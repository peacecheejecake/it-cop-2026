"""Diff continued pre-training: MLM for B4-S, MLM+RMI for B5-S (spec FR-09/10, protocol §6, T16/T17, AT-03/10/11).

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
first update boundary at or past the budget. Only the encoder is exported; the MLM/RMI heads
and optimizer never reach fine-tuning.

RMI (B5-S): rmi_target=1 keeps the change's own message, 0 swaps in the message of another
CPT-train change drawn uniformly, rejecting the same change, the same normalized message hash
and the same exact-duplicate group (no fallback to any other pool). A change with an empty
message or no code is rmi_ineligible. The swapped input is rendered like the evidence
(message + "\n" + own code lines) and code lines are dropped from the end until it fits.
One optimizer update is one task; updates cycle MLM, RMI (1:1) or MLM, MLM, RMI (2:1) per `b5_task_schedule` and the budget counts
both tasks' non-padding input tokens together.
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
RMI_PLAN_STREAM, RMI_STREAM, DEV_RMI_STREAM = 404, 505, 606
RMI_SAMPLER = "rmi-uniform-reject-v1"
TASKS = ("mlm", "mlm+rmi")
CPT_COLUMNS = ("change_id", "split", "cpt_role", "group_id", "query_text", "message_text", "code_text")
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


@dataclass
class RmiPool:
    change_ids: list[str]
    messages: list[str]
    codes: list[str]
    msg_hash: list[str]
    groups: list[str]
    ineligible: dict[str, str]
    sha256: str

    def __len__(self) -> int:
        return len(self.change_ids)


def build_rmi_pool(view: CptCorpusView) -> RmiPool:
    cids, msgs, codes, hashes, groups, bad = [], [], [], [], [], {}
    for r in view.frame.itertuples(index=False):
        if not r.message_text.strip():
            bad[r.change_id] = "empty_message"
            continue
        if not r.code_text:
            bad[r.change_id] = "no_code"
            continue
        cids.append(r.change_id)
        msgs.append(r.message_text)
        codes.append(r.code_text)
        hashes.append(sha256_text(" ".join(r.message_text.split())))
        groups.append(str(r.group_id))
    if len(set(hashes)) < 2:
        raise ExecutionError(f"RMI pool for {view.role} has fewer than two distinct messages")
    digest = sha256_json(sorted(zip(cids, hashes, groups, strict=True)))
    return RmiPool(cids, msgs, codes, hashes, groups, bad, digest)


def _fit_rmi(tok, msg: str, code: str, max_length: int) -> tuple[list[int], bool]:  # noqa: ANN001
    lines = code.split("\n")
    ids = tok(msg + "\n" + "\n".join(lines), add_special_tokens=True)["input_ids"]
    cut = False
    while len(ids) > max_length and lines:
        lines = lines[:-1]
        cut = True
        ids = tok((msg + "\n" + "\n".join(lines)) if lines else msg, add_special_tokens=True)["input_ids"]
    if len(ids) > max_length:
        raise IntegrityError("RMI message alone exceeds max_length")
    return ids, cut


def rmi_examples(tok, pool: RmiPool, anchors: np.ndarray, rng: np.random.Generator, p_replace: float,  # noqa: ANN001
                 max_length: int) -> tuple[list[np.ndarray], np.ndarray, int, list[int]]:
    """Inputs, rmi_target, truncation count and partner positions (-1 = own message) for one window of anchors."""
    pairs, targets, partners = [], [], []
    for a in anchors:
        if rng.random() < p_replace:
            for _ in range(10_000):
                j = int(rng.integers(len(pool)))
                if j != a and pool.msg_hash[j] != pool.msg_hash[a] and pool.groups[j] != pool.groups[a]:
                    break
            else:
                raise ExecutionError(f"no valid RMI negative found for {pool.change_ids[a]}")
            pairs.append((pool.messages[j], pool.codes[a]))
            targets.append(0)
            partners.append(j)
        else:
            pairs.append((pool.messages[a], pool.codes[a]))
            targets.append(1)
            partners.append(-1)
    texts = [m + "\n" + c for m, c in pairs]
    enc = tok(texts, add_special_tokens=True)["input_ids"]
    out, truncated = [], 0
    for (m, c), ids in zip(pairs, enc, strict=True):
        if len(ids) > max_length:
            ids, cut = _fit_rmi(tok, m, c, max_length)
            truncated += int(cut)
        out.append(np.asarray(ids, dtype=np.int64))
    return out, np.asarray(targets, dtype=np.float32), truncated, partners


MLM_PER_RMI = {"alternating_1_to_1": 1, "alternating_2_to_1": 2}


def plan_mlm_rmi(mlm_lengths: np.ndarray, tok, pool: RmiPool, seed: int, window: int, budget: int,  # noqa: ANN001
                 p_replace: float, max_length: int, mlm_per_rmi: int = 1) -> dict:
    """MLM/RMI update plan fixed before the first step: every cycle is `mlm_per_rmi` MLM updates, then one RMI update."""
    if window < 1 or budget < 1 or len(mlm_lengths) == 0 or mlm_per_rmi < 1:
        raise ExecutionError("empty CPT plan")
    steps, total, digests = [], 0, []
    streams = {"mlm": [np.array([], dtype=np.int64), 0, len(mlm_lengths), PLAN_STREAM],
               "rmi": [np.array([], dtype=np.int64), 0, len(pool), RMI_PLAN_STREAM]}
    rmi_k = 0
    while total < budget:
        task = "rmi" if len(steps) % (mlm_per_rmi + 1) == mlm_per_rmi else "mlm"
        st = streams[task]
        while len(st[0]) < window:
            st[0] = np.concatenate([st[0], np.random.default_rng([seed, st[3], st[1]]).permutation(st[2])])
            st[1] += 1
        w, st[0] = st[0][:window], st[0][window:]
        if task == "mlm":
            steps.append({"task": "mlm", "items": w})
            total += int(mlm_lengths[w].sum())
            digests.append(["mlm", w.tolist()])
        else:
            ids, targets, trunc, _ = rmi_examples(tok, pool, w, np.random.default_rng([seed, RMI_STREAM, rmi_k]), p_replace, max_length)
            rmi_k += 1
            steps.append({"task": "rmi", "items": w, "ids": ids, "targets": targets, "truncated": trunc})
            total += int(sum(len(x) for x in ids))
            digests.append(["rmi", w.tolist(), targets.astype(int).tolist(), sha256_json([x.tolist() for x in ids])])
    mlm_seen = {int(i) for s in steps if s["task"] == "mlm" for i in s["items"]}
    rmi_seen = {pool.change_ids[int(i)] for s in steps if s["task"] == "rmi" for i in s["items"]}
    return {"steps": steps, "planned_tokens": total, "planned_updates": len(steps), "mlm_seen": mlm_seen, "rmi_seen": rmi_seen,
            "permutation_epochs": {k: v[1] for k, v in streams.items()}, "plan_sha256": sha256_json(digests)}


def build_rmi_head(hidden: int, seed: int, dropout: float = 0.1):  # noqa: ANN201
    torch = _torch()
    nn = torch.nn
    head = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Dropout(dropout), nn.Linear(hidden, 1))
    g = torch.Generator().manual_seed(seed + 1_000_003)
    with torch.no_grad():
        for p in head.parameters():
            if p.dim() > 1:
                bound = 1 / np.sqrt(p.shape[1])
                p.copy_((torch.rand(p.shape, generator=g) * 2 - 1) * bound)
            else:
                p.zero_()
    return head


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


class CptRun:
    """One seeded CPT run (task "mlm" for B4-S, "mlm+rmi" for B5-S) in `out_dir`, resumable at update boundaries."""

    def __init__(self, cfg: StudyConfig, seed: int, base_path: Path, out_dir: Path, tokenizer, task: str = "mlm") -> None:  # noqa: ANN001
        if not isinstance(cfg.cpt, CptCfg):
            raise ExecutionError("cpt section is not pinned")
        if task not in TASKS:
            raise ExecutionError(f"unknown CPT task {task}")
        self.cfg, self.cc, self.seed, self.base, self.out, self.tok = cfg, cfg.cpt, seed, base_path, out_dir, tokenizer
        self.task = task

    def _autocast(self, device):  # noqa: ANN001, ANN202
        torch = _torch()
        if self.cc.precision == "bf16_encoder_autocast":
            return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        import contextlib
        return contextlib.nullcontext()

    def _hidden(self, model, ids, device):  # noqa: ANN001, ANN202
        attn = (ids != self.tok.pad_token_id).long()
        with self._autocast(device):
            h = model.roberta(input_ids=ids, attention_mask=attn).last_hidden_state
        return h, int(attn.sum())

    def _loss_sum(self, model, ids, labels, device):  # noqa: ANN001, ANN202
        torch = _torch()
        h, n_tok = self._hidden(model, ids, device)
        sel = labels != -100
        logits = model.lm_head(h[sel].float()).float()
        return torch.nn.functional.cross_entropy(logits, labels[sel], reduction="sum"), int(sel.sum()), n_tok

    def _rmi_logits(self, model, rmi_head, ids, device):  # noqa: ANN001, ANN202
        h, n_tok = self._hidden(model, ids, device)
        return rmi_head(h[:, 0].float()).squeeze(-1), n_tok

    def _dev_batches(self, dev: Tokenized, cand: np.ndarray) -> list:
        rng = np.random.default_rng([self.cc.dev_mask_seed, DEV_STREAM])
        cc = self.cc
        masked = [mask_sample(x, e, rng, cc.mlm_probability, cc.mask_replace_fraction, cc.random_replace_fraction,
                              self.tok.mask_token_id, cand)[:2] for x, e in zip(dev.ids, dev.eligible, strict=True)]
        bs = cc.eval_batch_size
        return [(_pad([m[0] for m in masked[i:i + bs]], self.tok.pad_token_id), _pad([m[1] for m in masked[i:i + bs]], -100))
                for i in range(0, len(masked), bs)]

    def _dev_loss(self, model, batches, device, rmi_head=None, rmi_dev=None) -> dict:  # noqa: ANN001
        torch = _torch()
        model.eval()
        tot, n = 0.0, 0
        out: dict = {}
        with torch.no_grad():
            for ids, lab in batches:
                s, k, _ = self._loss_sum(model, ids.to(device), lab.to(device), device)
                tot += float(s)
                n += k
            out.update(dev_mlm_loss=tot / n, dev_targets=n)
            if rmi_head is not None:
                rmi_head.eval()
                ids_l, y = rmi_dev
                scores = []
                for i in range(0, len(ids_l), self.cc.eval_batch_size):
                    lg, _ = self._rmi_logits(model, rmi_head, _pad(ids_l[i:i + self.cc.eval_batch_size], self.tok.pad_token_id)
                                             .to(device), device)
                    scores.append(lg.float().cpu().numpy())
                lg = np.concatenate(scores)
                p = 1 / (1 + np.exp(-lg))
                eps = 1e-7
                from sklearn.metrics import roc_auc_score
                out.update(dev_rmi_loss=float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))),
                           dev_rmi_accuracy=float(np.mean((p >= 0.5) == (y == 1))), dev_rmi_auc=float(roc_auc_score(y, p)),
                           dev_rmi_n=int(len(y)))
                rmi_head.train()
        model.train()
        return out

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
        cc, rmi = self.cc, self.task == "mlm+rmi"
        device = pick_device(cc)
        seed_everything(self.seed)
        max_len = self.cfg.model.max_length
        tr = tokenize_corpus(self.tok, train, max_len)
        dv = tokenize_corpus(self.tok, dev, max_len)
        cand = replacement_candidates(self.tok)
        window = cc.micro_batch_size * cc.gradient_accumulation_steps
        if rmi:
            pool, dev_pool = build_rmi_pool(train), build_rmi_pool(dev)
            plan = plan_mlm_rmi(tr.lengths, self.tok, pool, self.seed, window, cc.budget, cc.rmi_replacement_probability, max_len,
                                MLM_PER_RMI[cc.b5_task_schedule])
            dev_rmi_ids, dev_rmi_y, _, _ = rmi_examples(self.tok, dev_pool, np.arange(len(dev_pool)),
                                                     np.random.default_rng([cc.dev_mask_seed, DEV_RMI_STREAM]),
                                                     cc.rmi_replacement_probability, max_len)
        else:
            mp = plan_windows(tr.lengths, self.seed, window, cc.budget)
            plan = {"steps": [{"task": "mlm", "items": w} for w in mp["windows"]], "planned_tokens": mp["planned_tokens"],
                    "planned_updates": mp["planned_updates"], "mlm_seen": {int(i) for w in mp["windows"] for i in w},
                    "rmi_seen": set(), "permutation_epochs": {"mlm": mp["permutation_epochs"]}, "plan_sha256": mp["plan_sha256"]}
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
        rmi_head = build_rmi_head(enc_info["hidden_size"], self.seed) if rmi else None
        model.to(device).train()
        params = list(model.parameters()) + (list(rmi_head.to(device).train().parameters()) if rmi else [])
        opt = torch.optim.AdamW(params, lr=cc.learning_rate, weight_decay=cc.weight_decay)
        dev_batches = self._dev_batches(dv, cand)
        rmi_dev = (dev_rmi_ids, dev_rmi_y) if rmi else None
        ck = self.out / "checkpoints"
        st = {"update": 0, "input_tokens": 0, "target_tokens": 0, "forced_single_target": 0, "history": [], "dev": [],
              "mlm_updates": 0, "rmi_updates": 0, "mlm_input_tokens": 0, "rmi_input_tokens": 0, "rmi_targets": 0,
              "rmi_positive": 0, "rmi_truncated_inputs": 0}
        last = resolve_generation(ck, "last")
        if last is not None:
            st = self._resume(last, model, rmi_head, opt, plan["plan_sha256"])
        else:
            atomic_write_json(self.out / "cpt.json", {"status": "running", "seed": self.seed, "task": self.task})
            st["dev"].append({"update": 0, **self._dev_loss(model, dev_batches, device, rmi_head, rmi_dev)})
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        mb = cc.micro_batch_size
        bce = torch.nn.BCEWithLogitsLoss(reduction="sum")
        while st["update"] < N:
            u = st["update"]
            step = plan["steps"][u]
            for g in opt.param_groups:
                g["lr"] = lr_at(u, N, warmup, cc.learning_rate)
            opt.zero_grad(set_to_none=True)
            loss_tot, tok_in = 0.0, 0
            if step["task"] == "mlm":
                rng = np.random.default_rng([self.seed, MASK_STREAM, u])
                masked = [mask_sample(tr.ids[i], tr.eligible[i], rng, cc.mlm_probability, cc.mask_replace_fraction,
                                      cc.random_replace_fraction, self.tok.mask_token_id, cand) for i in step["items"]]
                denom = sum(int((m[1] != -100).sum()) for m in masked)
                for j in range(0, len(masked), mb):
                    part = masked[j:j + mb]
                    s, _, k = self._loss_sum(model, _pad([m[0] for m in part], self.tok.pad_token_id).to(device),
                                             _pad([m[1] for m in part], -100).to(device), device)
                    (s / denom).backward()  # exact mean over all targets of the update window
                    loss_tot += float(s.detach())
                    tok_in += k
                st["mlm_updates"] += 1
                st["mlm_input_tokens"] += tok_in
                st["target_tokens"] += denom
                st["forced_single_target"] += sum(m[2] for m in masked)
            else:
                ids_l, y = step["ids"], step["targets"]
                denom = len(ids_l)
                for j in range(0, denom, mb):
                    lg, k = self._rmi_logits(model, rmi_head, _pad(ids_l[j:j + mb], self.tok.pad_token_id).to(device), device)
                    s = bce(lg, torch.as_tensor(y[j:j + mb], device=device))
                    (s / denom).backward()  # exact mean over the RMI examples of the update window
                    loss_tot += float(s.detach())
                    tok_in += k
                st["rmi_updates"] += 1
                st["rmi_input_tokens"] += tok_in
                st["rmi_targets"] += denom
                st["rmi_positive"] += int(y.sum())
                st["rmi_truncated_inputs"] += step["truncated"]
            torch.nn.utils.clip_grad_norm_(params, cc.max_grad_norm)
            opt.step()
            loss = loss_tot / denom
            if not math.isfinite(loss):
                raise ExecutionError(f"non-finite {step['task']} loss at update {u + 1}")
            st["update"] = u + 1
            st["input_tokens"] += tok_in
            st["history"].append({"update": u + 1, "task": step["task"], "loss": loss, "lr": opt.param_groups[0]["lr"],
                                  "input_tokens": tok_in, "targets": denom})
            if st["update"] % cc.dev_eval_every_updates == 0 or st["update"] == N:
                st["dev"].append({"update": st["update"], **self._dev_loss(model, dev_batches, device, rmi_head, rmi_dev)})
                d = st["dev"][-1]
                extra = f" dev_rmi_acc={d['dev_rmi_accuracy']:.3f}" if rmi else ""
                print(f"CPT-{self.task.upper()} seed={self.seed} update={st['update']}/{N} {step['task']}_loss={loss:.4f} "
                      f"dev_mlm={d['dev_mlm_loss']:.4f}{extra} tokens={st['input_tokens']}", flush=True)
            if st["update"] % cc.checkpoint_every_updates == 0 and st["update"] < N:
                self._checkpoint(ck, model, rmi_head, opt, st, plan["plan_sha256"])
        train_s = time.perf_counter() - t0
        if st["input_tokens"] != plan["planned_tokens"]:
            raise IntegrityError(f"actual input tokens {st['input_tokens']} != planned {plan['planned_tokens']}")
        enc_hash = self._export(export, model, enc_info, st)
        seen = {tr.change_ids[i] for i in plan["mlm_seen"]} | plan["rmi_seen"]
        accounting = {"task": self.task, "budget_unit": cc.budget_unit, "budget": cc.budget,
                      "mlm_input_tokens": st["mlm_input_tokens"], "rmi_input_tokens": st["rmi_input_tokens"],
                      "mlm_target_tokens": st["target_tokens"], "rmi_targets": st["rmi_targets"],
                      "rmi_positive_targets": st["rmi_positive"], "rmi_negative_targets": st["rmi_targets"] - st["rmi_positive"],
                      "rmi_truncated_inputs": st["rmi_truncated_inputs"], "forward_tokens": st["input_tokens"],
                      "optimizer_updates": st["update"], "mlm_updates": st["mlm_updates"], "rmi_updates": st["rmi_updates"],
                      "planned_updates": N, "warmup_updates": warmup, "unique_changes_seen": len(seen),
                      "corpus_changes": len(tr.ids), "corpus_tokens_per_pass": int(tr.lengths.sum()),
                      "mlm_passes": st["mlm_input_tokens"] / int(tr.lengths.sum()),
                      "permutation_epochs": plan["permutation_epochs"],
                      "excluded": {"train": tr.excluded, "dev": dv.excluded},
                      "forced_single_target_samples": st["forced_single_target"],
                      "dev_forward_tokens_per_eval": int(dv.lengths.sum()), "dev_evals": len(st["dev"]),
                      "cpt_train_rows_with_dev_identical_text": content_overlap,
                      "train_seconds_this_attempt": train_s}
        if rmi:
            accounting["rmi"] = {"sampler": RMI_SAMPLER, "replacement_probability": cc.rmi_replacement_probability,
                                 "pool_sha256": pool.sha256, "pool_size": len(pool), "ineligible": pool.ineligible,
                                 "ineligible_rate": len(pool.ineligible) / len(train.ids), "dev_pool_sha256": dev_pool.sha256,
                                 "dev_examples": int(len(dev_rmi_y)), "dev_positive": int(dev_rmi_y.sum()),
                                 "target_convention": "1 = original message, 0 = message from another CPT-train change",
                                 "false_negative_limit": "a swapped message can still describe the change; not filtered"}
        result = {"status": "completed", "seed": self.seed, "task": self.task, "plan_sha256": plan["plan_sha256"],
                  "token_accounting": accounting, "encoder_init_state_sha256": encoder_init_hash,
                  "lm_head_newly_initialized": mlm_head_new, "lm_head_init_state_sha256": lm_head_init_hash,
                  "rmi_head_init": "seeded uniform, seed+1000003" if rmi else None,
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
            "artifact_kind": "cpt_encoder_export", "base_revision": self.cfg.model.revision, "task": self.task, "seed": self.seed,
            "hidden_size": enc_info["hidden_size"], "encoder_state_sha256": h,
            "encoder_safetensors_sha256": sha256_file(tmp / "encoder.safetensors"),
            "includes_mlm_head": False, "includes_rmi_head": False, "includes_optimizer": False, "optimizer_updates": st["update"]})
        shutil.rmtree(path, ignore_errors=True)
        os.replace(tmp, path)
        return h

    def _checkpoint(self, ck: Path, model, rmi_head, opt, st: dict, plan_hash: str) -> None:  # noqa: ANN001
        torch = _torch()
        tmp = ck / "last.tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        torch.save({"model": model.state_dict(), "rmi_head": None if rmi_head is None else rmi_head.state_dict(),
                    "optimizer": opt.state_dict(), "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}, tmp / "state.pt")
        atomic_write_json(tmp / "trainer.json", {**st, "plan_sha256": plan_hash})
        commit_generation(ck, "last", tmp)

    def _resume(self, path: Path, model, rmi_head, opt, plan_hash: str) -> dict:  # noqa: ANN001
        torch = _torch()
        st = read_json(path / "trainer.json")
        if st.pop("plan_sha256") != plan_hash:
            raise IntegrityError("CPT resume refused: update plan changed (start a new run)")
        # map to CPU: RNG states must stay CPU ByteTensors; load_state_dict copies weights onto the live device.
        ck = torch.load(path / "state.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        if rmi_head is not None:
            rmi_head.load_state_dict(ck["rmi_head"])
        opt.load_state_dict(ck["optimizer"])
        torch.set_rng_state(ck["torch_rng"])
        if ck["cuda_rng"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(ck["cuda_rng"])
        return st


MlmCpt = CptRun


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
