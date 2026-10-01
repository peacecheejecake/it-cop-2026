"""Common EvidenceView, renderer `message-add-del-text-v2` (spec protocol §5, FR-07/20, AT-07/08).

One label-free query span per change, chosen against the CodeBERT 512-token budget and
then shared verbatim by B1 (TF-IDF), B2-B5 (encoder) and L0/L1 (LLM). No new special
tokens: added lines get a "+ " prefix, deleted lines "- " (native tokenizer).

Rule (fixed before any model): message <= max_message_tokens, cut at a token's char
offset so the span is a substring of the original. Code budget = max_length - 2 - message
tokens, split between add/delete in proportion to their token totals; each op keeps a
prefix of its (lexicographically sorted, see adapter) lines; unused budget moves to the
other op. The rendered text is re-tokenized and lines are dropped from the end until it
fits. Every surviving span and the rendered-text hash are recorded.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .util import ConfigError, IntegrityError, PolicyError, atomic_write_json, read_json, sha256_file, sha256_json, sha256_text

RENDERER = "message-add-del-text-v2"
TOKENIZER_FILES = ("vocab.json", "merges.txt", "tokenizer_config.json", "special_tokens_map.json", "tokenizer.json",
                   "added_tokens.json")


def tokenizer_files_digest(local_path: str | Path) -> str:
    p = Path(local_path)
    return sha256_json({f: sha256_file(p / f) for f in TOKENIZER_FILES if (p / f).exists()})


def load_tokenizer(local_path: str | Path, revision: str):  # noqa: ANN201
    from transformers import AutoTokenizer

    p = Path(local_path)
    src = read_json(p / "SOURCE.json")
    if src.get("resolved_revision") != revision:
        raise ConfigError(f"tokenizer snapshot revision {src.get('resolved_revision')} != pinned {revision}")
    tok = AutoTokenizer.from_pretrained(str(p), local_files_only=True, use_fast=True)
    added = tok.get_added_vocab()
    digest = tokenizer_files_digest(p)
    return tok, {"revision": revision, "files_sha256": digest, "vocab_size": tok.vocab_size,
                 "added_tokens": sorted(added), "class": type(tok).__name__}


def _allocate(add_len: list[int], del_len: list[int], budget: int) -> tuple[int, int]:
    def prefix(lengths: list[int], b: int) -> int:
        c = np.cumsum(lengths) if lengths else np.array([], dtype=int)
        return int(np.searchsorted(c, b, side="right"))

    a_tot, d_tot = sum(add_len), sum(del_len)
    if a_tot + d_tot <= budget:
        return len(add_len), len(del_len)
    b_add = int(budget * a_tot / (a_tot + d_tot)) if a_tot + d_tot else 0
    ka = prefix(add_len, b_add)
    used_a = sum(add_len[:ka])
    kd = prefix(del_len, budget - used_a)
    used_d = sum(del_len[:kd])
    ka = prefix(add_len, budget - used_d)
    return ka, kd


def render(msg_span: str, adds: list[str], dels: list[str]) -> tuple[str, str]:
    code = "\n".join([f"+ {x}" for x in adds] + [f"- {x}" for x in dels])
    return (msg_span + "\n" + code) if code else msg_span, code


def build_evidence(snapshot_dir: str | Path, split_dir: str | Path, local_path: str, revision: str,
                   max_length: int, max_message_tokens: int, out_dir: str | Path) -> dict:
    out = Path(out_dir)
    if out.exists():
        raise PolicyError(f"{out} exists; evidence artifacts are content-addressed and immutable")
    tok, tinfo = load_tokenizer(local_path, revision)
    if tinfo["added_tokens"] not in ([], ["<mask>"]) and set(tinfo["added_tokens"]) - set(tok.all_special_tokens):
        raise ConfigError(f"tokenizer has non-native added tokens {tinfo['added_tokens']}")
    snap = Path(snapshot_dir)
    changes = pd.read_parquet(snap / "changes.parquet")
    edits = pd.read_parquet(snap / "edits.parquet").sort_values(["change_id", "op", "order"])
    members = set(pd.read_parquet(Path(split_dir) / "splits.parquet")["change_id"])
    changes = changes[changes["change_id"].isin(members)].reset_index(drop=True)
    grouped = {k: g["text"].tolist() for k, g in edits.groupby(["change_id", "op"])}
    pieces = [f"\n+ {x}" for x in edits.loc[edits.op == "add", "text"]] + [f"\n- {x}" for x in edits.loc[edits.op == "delete", "text"]]
    lens = dict(zip(pieces, (len(i) for i in tok(pieces, add_special_tokens=False)["input_ids"]), strict=True)) if pieces else {}
    msg_enc = tok(changes["message"].tolist(), add_special_tokens=False, return_offsets_mapping=True)
    body = max_length - 2
    rows = []
    texts = []
    for i, r in enumerate(changes.itertuples()):
        ids, offs = msg_enc["input_ids"][i], msg_enc["offset_mapping"][i]
        keep_m = min(len(ids), max_message_tokens)
        msg_span = r.message[: offs[keep_m - 1][1]] if keep_m else ""
        adds, dels = grouped.get((r.change_id, "add"), []), grouped.get((r.change_id, "delete"), [])
        al, dl = [lens[f"\n+ {x}"] for x in adds], [lens[f"\n- {x}"] for x in dels]
        ka, kd = _allocate(al, dl, max(body - keep_m, 0))
        rows.append([r.change_id, msg_span, adds, dels, ka, kd, len(ids), keep_m])
        texts.append(render(msg_span, adds[:ka], dels[:kd])[0])
    n_tokens = [len(x) for x in tok(texts, add_special_tokens=True)["input_ids"]]
    out_rows = []
    for (cid, msg_span, adds, dels, ka, kd, m_tot, m_keep), n in zip(rows, n_tokens, strict=True):
        while n > max_length:
            if kd > 0 and (kd >= ka or ka == 0):
                kd -= 1
            elif ka > 0:
                ka -= 1
            else:
                raise IntegrityError(f"{cid}: message alone exceeds budget")
            n = len(tok(render(msg_span, adds[:ka], dels[:kd])[0], add_special_tokens=True)["input_ids"])
        text, code = render(msg_span, adds[:ka], dels[:kd])
        out_rows.append({"change_id": cid, "evidence_profile": "matched", "renderer": RENDERER,
                         "query_text": text, "message_text": msg_span, "code_text": code,
                         "source_span_ids": json.dumps({"message_chars": [0, len(msg_span)], "add": [0, ka], "delete": [0, kd]}),
                         "query_content_hash": sha256_text(text), "encoder_tokens": n,
                         "msg_tokens_total": m_tot, "msg_tokens_kept": m_keep, "add_lines_total": len(adds),
                         "add_lines_kept": ka, "del_lines_total": len(dels), "del_lines_kept": kd,
                         "truncated": bool(m_keep < m_tot or ka < len(adds) or kd < len(dels))})
    ev = pd.DataFrame(out_rows)
    if (ev["encoder_tokens"] > max_length).any():
        raise IntegrityError("evidence exceeds max_length after verification")
    out.mkdir(parents=True)
    ev.to_parquet(out / "evidence.parquet", index=False)
    cov = {"n": int(len(ev)), "truncated": int(ev.truncated.sum()),
           "message_truncated": int((ev.msg_tokens_kept < ev.msg_tokens_total).sum()),
           "add_lines_kept_ratio": float(ev.add_lines_kept.sum() / max(ev.add_lines_total.sum(), 1)),
           "del_lines_kept_ratio": float(ev.del_lines_kept.sum() / max(ev.del_lines_total.sum(), 1)),
           "changes_with_all_code_kept": int(((ev.add_lines_kept == ev.add_lines_total) & (ev.del_lines_kept == ev.del_lines_total)).sum()),
           "encoder_tokens_median": float(ev.encoder_tokens.median()),
           "coverage_scope": "within preprocessed_lines package (not repository diff coverage)"}
    manifest = {"artifact_kind": "evidence_view", "renderer": RENDERER, "evidence_profile": "matched",
                "max_length": max_length, "max_message_tokens": max_message_tokens, "tokenizer": tinfo,
                "snapshot_manifest_sha256": sha256_file(snap / "manifest.json"),
                "split_manifest_sha256": sha256_file(Path(split_dir) / "manifest.json"),
                "coverage": cov, "evidence_parquet_sha256": sha256_file(out / "evidence.parquet"),
                "query_hash_digest": sha256_json(sorted(zip(ev.change_id, ev.query_content_hash, strict=True)))}
    manifest["cache_key"] = sha256_json({k: manifest[k] for k in ("renderer", "max_length", "max_message_tokens", "tokenizer",
                                                                  "snapshot_manifest_sha256", "split_manifest_sha256")})
    atomic_write_json(out / "manifest.json", manifest)
    return manifest
