from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .common import BenchError, digest, new_dir, read_json, seal_files, write_json, write_jsonl
from .data import load_inputs


class DemoTokenizer:
    """Deterministic byte tokenizer for synthetic plumbing tests, NOT CodeBERT."""
    pad_token_id = 0
    def encode(self, text, add_special_tokens=True):
        ids = [b + 3 for b in text.encode("utf-8")]
        return [1] + ids + [2] if add_special_tokens else ids
    def decode(self, ids, **kwargs):
        return bytes(i - 3 for i in ids if 3 <= i <= 258).decode("utf-8", errors="replace")
    def get_vocab(self):
        return {str(i): i for i in range(259)}
    @property
    def special_tokens_map(self):
        return {"pad": 0, "cls": 1, "sep": 2}


def get_tokenizer(name: str, revision: str | None = None, allow_download: bool = False):
    if name == "demo-byte":
        return DemoTokenizer()
    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        raise BenchError("Install the neural extra: pip install -e '.[neural]'") from e
    return AutoTokenizer.from_pretrained(name, revision=revision, local_files_only=not allow_download,
                                         trust_remote_code=False, use_fast=True)


def tokenizer_signature(tokenizer) -> str:
    value = {"vocab": tokenizer.get_vocab(), "special": tokenizer.special_tokens_map}
    if hasattr(tokenizer, "backend_tokenizer"):
        value["backend"] = tokenizer.backend_tokenizer.to_str()
    return digest(value)


def head_tail(ids: list[int], budget: int, marker: list[int]) -> list[int]:
    if len(ids) <= budget:
        return ids
    if budget < len(marker) + 2:
        return ids[:max(0, budget)]
    body = budget - len(marker)
    head = (body + 1) // 2
    return ids[:head] + marker + ids[-(body - head):] if body > head else ids[:head] + marker


def render_view(tokenizer, message: str, diff: str, max_tokens: int, message_tokens: int) -> dict:
    if max_tokens < 32 or not 1 <= message_tokens < max_tokens // 2:
        raise BenchError("Use max_tokens>=32 and 1 <= message_tokens < max_tokens/2")
    enc = lambda text: tokenizer.encode(text, add_special_tokens=False)
    marker = enc("\n[TRUNCATED]\n")
    msg_raw = enc(message)
    msg = tokenizer.decode(head_tail(msg_raw, message_tokens, marker), skip_special_tokens=True,
                           clean_up_tokenization_spaces=False)
    prefix = "COMMIT MESSAGE:\n" + msg + "\nCODE DIFF:\n"
    raw_diff = enc(diff)
    special_n = len(tokenizer.encode("", add_special_tokens=True))
    budget = max_tokens - special_n - len(enc(prefix))
    if budget <= len(marker):
        raise BenchError("Too little diff budget; increase max_tokens or reduce message_tokens")
    # Byte/BPE decode + encode can alter token counts at the join. Bound final actual input.
    for _ in range(max_tokens):
        chosen = head_tail(raw_diff, budget, marker)
        patch = tokenizer.decode(chosen, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        text = prefix + patch
        ids = tokenizer.encode(text, add_special_tokens=True)
        if len(ids) <= max_tokens:
            return {"text": text, "input_ids": ids, "original_diff_tokens": len(raw_diff),
                    "original_message_tokens": len(msg_raw), "input_tokens": len(ids),
                    "truncated": len(raw_diff) > budget or len(msg_raw) > message_tokens,
                    "text_hash": digest(text)}
        budget -= max(1, len(ids) - max_tokens)
        if budget <= len(marker):
            break
    raise BenchError("Could not fit a stable token view")


def prepare_view(source: str, out: str, tokenizer_name: str, max_tokens: int = 512,
                 message_tokens: int = 64, revision: str | None = None, allow_download: bool = False) -> dict:
    rows, old = load_inputs(source)
    if any("text" in r for r in rows):
        raise BenchError("Already prepared views; use the original raw partition")
    if tokenizer_name == "demo-byte" and not all(r["synthetic"] for r in rows):
        raise BenchError("Demo tokenizer is restricted to explicit synthetic data")
    if any(r["domain"] == "internal" for r in rows) and allow_download:
        raise BenchError("Internal view preparation must use a local tokenizer (no model downloads)")
    tok = get_tokenizer(tokenizer_name, revision, allow_download)
    root = new_dir(out); prepared = []
    config = {"tokenizer": tokenizer_name, "revision": revision, "tokenizer_signature": tokenizer_signature(tok),
              "max_tokens": max_tokens, "message_tokens": message_tokens, "strategy": "message-cap+diff-head-tail-v1",
              "pad_token_id": tok.pad_token_id, "synthetic_tokenizer": tokenizer_name == "demo-byte"}
    # Path/revision text may differ after moving offline; policy identity uses tokenizer content.
    config["policy_hash"] = digest({k: config[k] for k in ("tokenizer_signature", "max_tokens", "message_tokens", "strategy")})
    for r in rows:
        x = {k: v for k, v in r.items() if k not in ("message", "diff")}
        x.update(render_view(tok, r["message"], r["diff"], max_tokens, message_tokens))
        x["view_policy_hash"] = config["policy_hash"]
        prepared.append(x)
    write_jsonl(root / "inputs.jsonl", prepared)
    # Packaging operation only; no label value influences the input view.
    if (Path(source) / "labels.jsonl").exists():
        shutil.copyfile(Path(source) / "labels.jsonl", root / "labels.jsonl")
    names = ["inputs.jsonl"] + (["labels.jsonl"] if (root / "labels.jsonl").exists() else [])
    manifest = dict(old, files=seal_files(root, names), view=config,
                    parent_inputs_sha256=old["files"]["inputs.jsonl"],
                    view_stats={"n": len(rows), "truncated": sum(r["truncated"] for r in prepared)})
    write_json(root / "manifest.json", manifest)
    return manifest["view_stats"]


def require_view(rows: list[dict], manifest: dict) -> dict:
    if not manifest.get("view") or any("text" not in r or "input_ids" not in r for r in rows):
        raise BenchError("Prepare a common tokenizer view before code/LLM runs")
    view = manifest["view"]
    for r in rows:
        if (r.get("view_policy_hash") != view["policy_hash"] or r.get("text_hash") != digest(r["text"])
                or len(r["input_ids"]) > view["max_tokens"]):
            raise BenchError("Input view contract mismatch")
    return view
