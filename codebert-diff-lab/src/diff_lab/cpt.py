"""Diff MLM continued pre-training for B4-S (spec FR-09, protocol §6.1/§6.3, T16/T17, AT-03/10/11).

Corpus: EvidenceView `query_text` of the label-free CPT-train subset (public train only).
Masking: native special tokens and renderer structure (the message/code "\\n" separators and
the "+ "/"- " line prefixes, located by char offsets) are never targets or replacement
candidates; code operators like `-` inside a line stay eligible. Rate 0.15, 80/10/10.
A sample with eligible tokens but no draw gets one uniformly chosen target; a sample with
no eligible token is excluded with a reason.

Every random stream is derived from (seed, stream, index) rather than carried state, so the
update plan, masks and resume are reproducible: the window plan is fixed before the first
step (planned == actual updates), masks for update u come from rng([seed, MASK, u]), and
CPT-dev masks from a fixed seed independent of the training seed.
Budget = cumulative non-padding input tokens (specials included); training stops at the
first update boundary at or past the budget. Only the encoder is exported; the MLM head
and optimizer never reach fine-tuning.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CptCfg, StudyConfig
from .neural import _torch, load_encoder, pick_device, seed_everything, state_hash
from .util import (
    ExecutionError,
    IntegrityError,
    PolicyError,
    atomic_write_json,
    commit_generation,
    read_json,
    resolve_generation,
    sha256_file,
    sha256_json,
    sha256_text,
)

PLAN_STREAM, MASK_STREAM, DEV_STREAM = 101, 202, 303
CPT_COLUMNS = ("change_id", "split", "cpt_role", "query_text", "message_text", "code_text")
FORBIDDEN_COLUMNS = ("label", "labels", "bug", "buggy", "defect")


@dataclass(frozen=True)
class CptCorpusView:
    """Label-free CPT input. Every row must be public train with the requested CPT role (AT-03)."""
    role: str
    frame: pd.DataFrame
    lineage: dict = field(repr=False)

    def __post_init__(self) -> None:
        if self.role not in ("cpt_train", "cpt_dev"):
            raise PolicyError(f"unknown CPT role {self.role}")
        if self.lineage.get("visibility") != "public":
            raise PolicyError("CPT corpus must come from an approved public source")
        leaked = [c for c in self.frame.columns if c.lower() in FORBIDDEN_COLUMNS]
        if leaked:
            raise PolicyError(f"CPT corpus carries label columns {leaked}")
        missing = [c for c in CPT_COLUMNS if c not in self.frame.columns]
        if missing:
            raise PolicyError(f"CPT corpus missing columns {missing}")
        bad = self.frame[(self.frame["split"] != "train") | (self.frame["cpt_role"] != self.role)]
        if len(bad):
            raise PolicyError(f"{len(bad)} rows outside public train/{self.role} in CPT corpus, e.g. {bad['change_id'].iloc[0]}")

    @property
    def ids(self) -> list[str]:
        return self.frame["change_id"].tolist()


def structure_chars(message: str, code: str) -> set[int]:
    """Char positions inserted by renderer message-add-del-text-v2 (separators and line prefixes)."""
    if not code:
        return set()
    pos = {len(message)}
    start = len(message) + 1
    for i, line in enumerate(code.split("\n")):
        if not (line.startswith("+ ") or line.startswith("- ")):
            raise IntegrityError(f"code line without renderer prefix: {line[:40]!r}")
        if i:
            pos.add(start - 1)
        pos.update((start, start + 1))
        start += len(line) + 1
    return pos


@dataclass
class Tokenized:
    ids: list[np.ndarray]
    eligible: list[np.ndarray]
    change_ids: list[str]
    excluded: dict[str, str]

    @property
    def lengths(self) -> np.ndarray:
        return np.array([len(x) for x in self.ids], dtype=np.int64)


def tokenize_corpus(tok, view: CptCorpusView, max_length: int) -> Tokenized:  # noqa: ANN001
    f = view.frame
    enc = tok(f["query_text"].tolist(), add_special_tokens=True, truncation=False, return_offsets_mapping=True,
              return_special_tokens_mask=True)
    ids, elig, cids, excluded = [], [], [], {}
    for i, r in enumerate(f.itertuples(index=False)):
        if r.query_text != ((r.message_text + "\n" + r.code_text) if r.code_text else r.message_text):
            raise IntegrityError(f"{r.change_id}: query_text is not the renderer output of its spans")
        x = enc["input_ids"][i]
        if len(x) > max_length:
            raise IntegrityError(f"{r.change_id}: {len(x)} tokens exceed {max_length}; EvidenceView contract violated")
        protected = structure_chars(r.message_text, r.code_text)
        e = np.array([not s and not any(c in protected for c in range(a, b)) and b > a
                      for s, (a, b) in zip(enc["special_tokens_mask"][i], enc["offset_mapping"][i], strict=True)])
        if not e.any():
            excluded[r.change_id] = "no_eligible_tokens"
            continue
        ids.append(np.asarray(x, dtype=np.int64))
        elig.append(e)
        cids.append(r.change_id)
    return Tokenized(ids, elig, cids, excluded)


def replacement_candidates(tok) -> np.ndarray:  # noqa: ANN001
    special = set(tok.all_special_ids)
    return np.array([i for i in range(len(tok)) if i not in special], dtype=np.int64)


def mask_sample(ids: np.ndarray, eligible: np.ndarray, rng: np.random.Generator, p: float, mask_frac: float,
                rand_frac: float, mask_id: int, candidates: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    sel = eligible & (rng.random(len(ids)) < p)
    forced = not sel.any()
    if forced:
        sel[rng.choice(np.flatnonzero(eligible))] = True
    labels = np.full(len(ids), -100, dtype=np.int64)
    labels[sel] = ids[sel]
    out = ids.copy()
    pos = np.flatnonzero(sel)
    r = rng.random(len(pos))
    out[pos[r < mask_frac]] = mask_id
    rnd = pos[(r >= mask_frac) & (r < mask_frac + rand_frac)]
    out[rnd] = candidates[rng.integers(0, len(candidates), len(rnd))]
    return out, labels, forced


def plan_windows(lengths: np.ndarray, seed: int, window: int, budget: int) -> dict:
    """Fixed sequence of optimizer-update windows over a seeded permutation stream until the token budget is met."""
    if window < 1 or budget < 1 or len(lengths) == 0:
        raise ExecutionError("empty CPT plan")
    windows, total, epoch, stream = [], 0, 0, np.array([], dtype=np.int64)
    while total < budget:
        while len(stream) < window:
            stream = np.concatenate([stream, np.random.default_rng([seed, PLAN_STREAM, epoch]).permutation(len(lengths))])
            epoch += 1
        w, stream = stream[:window], stream[window:]
        windows.append(w)
        total += int(lengths[w].sum())
    seen = np.unique(np.concatenate(windows))
    return {"windows": windows, "planned_tokens": total, "planned_updates": len(windows), "permutation_epochs": epoch,
            "unique_items": int(len(seen)), "plan_sha256": sha256_json([w.tolist() for w in windows])}


def lr_at(update: int, planned: int, warmup: int, base: float) -> float:
    if update < warmup:
        return base * (update + 1) / warmup
    return base * max(0.0, (planned - update) / max(planned - warmup, 1))


def _pad(rows: list[np.ndarray], fill: int):  # noqa: ANN202
    torch = _torch()
    L = max(len(r) for r in rows)
    out = torch.full((len(rows), L), fill, dtype=torch.long)
    for j, r in enumerate(rows):
        out[j, :len(r)] = torch.from_numpy(r)
    return out


def cpt_id_for(cfg: StudyConfig, seed: int, lineage: dict, code: dict, task: str) -> str:
    return sha256_json({"cpt": cfg.cpt.model_dump(), "revision": cfg.model.revision, "seed": seed, "task": task,
                        "lineage": lineage, "code": code})[:16]


class MlmCpt:
    """One seeded diff-MLM CPT run in `out_dir`, resumable at optimizer-update boundaries."""

    def __init__(self, cfg: StudyConfig, seed: int, base_path: Path, out_dir: Path, tokenizer) -> None:  # noqa: ANN001
        if not isinstance(cfg.cpt, CptCfg):
            raise ExecutionError("cpt section is not pinned")
        self.cfg, self.cc, self.seed, self.base, self.out, self.tok = cfg, cfg.cpt, seed, base_path, out_dir, tokenizer

    def _autocast(self, device):  # noqa: ANN001, ANN202
        torch = _torch()
        if self.cc.precision == "bf16_encoder_autocast":
            return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        import contextlib
        return contextlib.nullcontext()

    def _loss_sum(self, model, ids, labels, device):  # noqa: ANN001, ANN202
        torch = _torch()
        attn = (ids != self.tok.pad_token_id).long()
        with self._autocast(device):
            h = model.roberta(input_ids=ids, attention_mask=attn).last_hidden_state
        sel = labels != -100
        logits = model.lm_head(h[sel].float()).float()
        return torch.nn.functional.cross_entropy(logits, labels[sel], reduction="sum"), int(sel.sum()), int(attn.sum())

    def _dev_batches(self, dev: Tokenized, cand: np.ndarray) -> list:
        rng = np.random.default_rng([self.cc.dev_mask_seed, DEV_STREAM])
        cc = self.cc
        masked = [mask_sample(x, e, rng, cc.mlm_probability, cc.mask_replace_fraction, cc.random_replace_fraction,
                              self.tok.mask_token_id, cand)[:2] for x, e in zip(dev.ids, dev.eligible, strict=True)]
        bs = cc.eval_batch_size
        return [(_pad([m[0] for m in masked[i:i + bs]], self.tok.pad_token_id), _pad([m[1] for m in masked[i:i + bs]], -100))
                for i in range(0, len(masked), bs)]

    def _dev_loss(self, model, batches, device) -> dict:  # noqa: ANN001
        torch = _torch()
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for ids, lab in batches:
                s, k, _ = self._loss_sum(model, ids.to(device), lab.to(device), device)
                tot += float(s)
                n += k
        model.train()
        return {"dev_mlm_loss": tot / n, "dev_targets": n}

    def run(self, train: CptCorpusView, dev: CptCorpusView) -> dict:
        torch = _torch()
        from transformers import AutoModelForMaskedLM

        if train.role != "cpt_train" or dev.role != "cpt_dev":
            raise PolicyError("CPT needs a cpt_train view for gradients and a cpt_dev view for diagnostics")
        if set(train.ids) & set(dev.ids):
            raise PolicyError("cpt_train and cpt_dev overlap")
        dev_hashes = {sha256_text(t) for t in dev.frame["query_text"]}
        content_overlap = int(sum(sha256_text(t) in dev_hashes for t in train.frame["query_text"]))
        export = self.out / "encoder"
        if (self.out / "cpt.json").exists() and read_json(self.out / "cpt.json")["status"] == "completed":
            return read_json(self.out / "cpt.json")
        cc = self.cc
        device = pick_device(cc)
        seed_everything(self.seed)
        tr = tokenize_corpus(self.tok, train, self.cfg.model.max_length)
        dv = tokenize_corpus(self.tok, dev, self.cfg.model.max_length)
        cand = replacement_candidates(self.tok)
        window = cc.micro_batch_size * cc.gradient_accumulation_steps
        plan = plan_windows(tr.lengths, self.seed, window, cc.budget)
        N = plan["planned_updates"]
        warmup = max(1, math.ceil(cc.warmup_fraction * N))
        base_encoder, enc_info = load_encoder(self.base, self.cfg.model.revision,
                                              (self.cfg.model.weights_file, self.cfg.model.weights_sha256))
        encoder_init_hash = state_hash(base_encoder)
        del base_encoder
        model, info = AutoModelForMaskedLM.from_pretrained(str(self.base), local_files_only=True, output_loading_info=True)
        if state_hash(model.roberta) != encoder_init_hash:
            raise IntegrityError("MLM model encoder differs from the pinned encoder init (AT-09)")
        mlm_head_new = sorted(k for k in info["missing_keys"] if k.startswith("lm_head"))
        lm_head_init_hash = state_hash(model.lm_head)
        model.to(device).train()
        opt = torch.optim.AdamW(model.parameters(), lr=cc.learning_rate, weight_decay=cc.weight_decay)
        dev_batches = self._dev_batches(dv, cand)
        ck = self.out / "checkpoints"
        st = {"update": 0, "input_tokens": 0, "target_tokens": 0, "forced_single_target": 0, "history": [], "dev": []}
        last = resolve_generation(ck, "last")
        if last is not None:
            st = self._resume(last, model, opt, plan["plan_sha256"])
        else:
            atomic_write_json(self.out / "cpt.json", {"status": "running", "seed": self.seed, "task": "mlm"})
            st["dev"].append({"update": 0, **self._dev_loss(model, dev_batches, device)})
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        mb = cc.micro_batch_size
        while st["update"] < N:
            u = st["update"]
            w = plan["windows"][u]
            rng = np.random.default_rng([self.seed, MASK_STREAM, u])
            masked = [mask_sample(tr.ids[i], tr.eligible[i], rng, cc.mlm_probability, cc.mask_replace_fraction,
                                  cc.random_replace_fraction, self.tok.mask_token_id, cand) for i in w]
            n_targets = sum(int((m[1] != -100).sum()) for m in masked)
            for g in opt.param_groups:
                g["lr"] = lr_at(u, N, warmup, cc.learning_rate)
            opt.zero_grad(set_to_none=True)
            loss_tot, tok_in = 0.0, 0
            for j in range(0, len(masked), mb):
                part = masked[j:j + mb]
                ids = _pad([m[0] for m in part], self.tok.pad_token_id).to(device)
                lab = _pad([m[1] for m in part], -100).to(device)
                s, _, k = self._loss_sum(model, ids, lab, device)
                (s / n_targets).backward()  # exact mean over all targets of the update window
                loss_tot += float(s.detach())
                tok_in += k
            torch.nn.utils.clip_grad_norm_(model.parameters(), cc.max_grad_norm)
            opt.step()
            loss = loss_tot / n_targets
            if not math.isfinite(loss):
                raise ExecutionError(f"non-finite MLM loss at update {u + 1}")
            st["update"] = u + 1
            st["input_tokens"] += tok_in
            st["target_tokens"] += n_targets
            st["forced_single_target"] += sum(m[2] for m in masked)
            st["history"].append({"update": u + 1, "mlm_loss": loss, "lr": opt.param_groups[0]["lr"], "input_tokens": tok_in,
                                  "target_tokens": n_targets})
            if st["update"] % cc.dev_eval_every_updates == 0 or st["update"] == N:
                st["dev"].append({"update": st["update"], **self._dev_loss(model, dev_batches, device)})
                print(f"CPT-MLM seed={self.seed} update={st['update']}/{N} loss={loss:.4f} "
                      f"dev={st['dev'][-1]['dev_mlm_loss']:.4f} tokens={st['input_tokens']}", flush=True)
            if st["update"] % cc.checkpoint_every_updates == 0 and st["update"] < N:
                self._checkpoint(ck, model, opt, st, plan["plan_sha256"])
        train_s = time.perf_counter() - t0
        if st["input_tokens"] != plan["planned_tokens"]:
            raise IntegrityError(f"actual input tokens {st['input_tokens']} != planned {plan['planned_tokens']}")
        enc_hash = self._export(export, model, enc_info, st)
        accounting = {"budget_unit": cc.budget_unit, "budget": cc.budget, "mlm_input_tokens": st["input_tokens"],
                      "rmi_input_tokens": 0, "mlm_target_tokens": st["target_tokens"], "forward_tokens": st["input_tokens"],
                      "optimizer_updates": st["update"], "planned_updates": N, "warmup_updates": warmup,
                      "unique_changes_seen": plan["unique_items"], "corpus_changes": len(tr.ids),
                      "corpus_tokens_per_pass": int(tr.lengths.sum()), "passes": st["input_tokens"] / int(tr.lengths.sum()),
                      "excluded": {"train": tr.excluded, "dev": dv.excluded},
                      "forced_single_target_samples": st["forced_single_target"],
                      "dev_forward_tokens_per_eval": int(dv.lengths.sum()), "dev_evals": len(st["dev"]),
                      "cpt_train_rows_with_dev_identical_text": content_overlap,
                      "train_seconds_this_attempt": train_s}
        result = {"status": "completed", "seed": self.seed, "task": "mlm", "plan_sha256": plan["plan_sha256"],
                  "token_accounting": accounting, "encoder_init_state_sha256": encoder_init_hash,
                  "lm_head_newly_initialized": mlm_head_new, "lm_head_init_state_sha256": lm_head_init_hash,
                  "exported_encoder_state_sha256": enc_hash, "device": str(device), "precision": cc.precision,
                  "peak_cuda_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
                  "dev": st["dev"], "train_ids_sha256": sha256_json(sorted(tr.change_ids)),
                  "dev_ids_sha256": sha256_json(sorted(dv.change_ids))}
        atomic_write_json(self.out / "token-accounting.json", accounting)
        (self.out / "metrics.jsonl").write_text("".join(f"{json.dumps(h)}\n" for h in st["history"]))
        shutil.rmtree(self.out / "checkpoints", ignore_errors=True)
        atomic_write_json(self.out / "cpt.json", result)
        return result

    def _export(self, path: Path, model, enc_info: dict, st: dict) -> str:  # noqa: ANN001
        from safetensors.torch import save_file
        tmp = path.with_name(path.name + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        save_file({k: v.detach().cpu().contiguous() for k, v in model.roberta.state_dict().items()}, str(tmp / "encoder.safetensors"))
        h = state_hash(model.roberta)
        atomic_write_json(tmp / "manifest.json", {
            "artifact_kind": "cpt_encoder_export", "base_revision": self.cfg.model.revision, "task": "mlm", "seed": self.seed,
            "hidden_size": enc_info["hidden_size"], "encoder_state_sha256": h,
            "encoder_safetensors_sha256": sha256_file(tmp / "encoder.safetensors"),
            "includes_mlm_head": False, "includes_optimizer": False, "optimizer_updates": st["update"]})
        shutil.rmtree(path, ignore_errors=True)
        os.replace(tmp, path)
        return h

    def _checkpoint(self, ck: Path, model, opt, st: dict, plan_hash: str) -> None:  # noqa: ANN001
        torch = _torch()
        tmp = ck / "last.tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}, tmp / "state.pt")
        atomic_write_json(tmp / "trainer.json", {**st, "plan_sha256": plan_hash})
        commit_generation(ck, "last", tmp)

    def _resume(self, path: Path, model, opt, plan_hash: str) -> dict:  # noqa: ANN001
        torch = _torch()
        st = read_json(path / "trainer.json")
        if st.pop("plan_sha256") != plan_hash:
            raise IntegrityError("CPT resume refused: update plan changed (start a new run)")
        # map to CPU: RNG states must stay CPU ByteTensors; load_state_dict copies weights onto the live device.
        ck = torch.load(path / "state.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        torch.set_rng_state(ck["torch_rng"])
        if ck["cuda_rng"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(ck["cuda_rng"])
        return st


def load_cpt_encoder(export_dir: Path, base_path: Path, revision: str, weights: tuple[str, str] | None = None):  # noqa: ANN201
    """Fine-tuning init from a CPT export: pinned architecture + exported encoder weights only (AT-10)."""
    from safetensors.torch import load_file

    m = read_json(export_dir / "manifest.json")
    if m.get("artifact_kind") != "cpt_encoder_export" or m.get("base_revision") != revision:
        raise IntegrityError(f"{export_dir} is not a CPT export of revision {revision}")
    if sha256_file(export_dir / "encoder.safetensors") != m["encoder_safetensors_sha256"]:
        raise IntegrityError("CPT encoder export hash mismatch")
    encoder, info = load_encoder(base_path, revision, weights)
    encoder.load_state_dict(load_file(str(export_dir / "encoder.safetensors")), strict=True)
    if state_hash(encoder) != m["encoder_state_sha256"]:
        raise IntegrityError("CPT encoder state hash mismatch after load")
    return encoder, {**info, "cpt_export": {k: m[k] for k in ("task", "seed", "encoder_state_sha256", "optimizer_updates")}}
