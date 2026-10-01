"""Downstream encoder runs: B2-S (frozen + fusion head), B3-S/B4-S (full fine-tuning) (spec FR-08/09/11/19, AT-09/10/14/28).

B4-S/B5-S are B3-S started from a diff-MLM (B5: MLM+RMI) CPT encoder export (`init_encoder`); only the encoder
transfers, the head is initialised by the same seeded rule and the optimizer is fresh.

Both start from the same pinned CodeBERT snapshot and the same seeded head init and share
batch/epoch/loss/optimizer; B2 only freezes the encoder (eval mode, no grads, excluded from
the optimizer; state hash checked before/after). Head `cls-feature-tanh-v1`:
logit = Linear(dropout(concat(CLS, tanh(Linear(features))))). Loss BCEWithLogits.

Precision `bf16_encoder_autocast` autocasts only the encoder forward; the head and loss run
in fp32 (a whole-epoch autocast including the head diverged in experiment 003).
Encoder and head may use different learning rates (v3); B2 has no encoder parameter group.
Checkpoints are immutable generations behind an atomically replaced pointer (`last.json`,
`best.json`); the best generation holds weights and its validation scores together. Resume
loads on CPU (RNG states must stay CPU tensors) and restores model, optimizer, RNG, epoch
and early-stopping state. Requests for an unavailable device fail (no fallback).
"""
from __future__ import annotations

import hashlib
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import FinetuneCfg, StudyConfig
from .features import StructuredPipeline
from .metrics import evaluate
from .policy import QueryView, TrainingDatasetView, require_training_view
from .util import (
    ExecutionError,
    IntegrityError,
    atomic_write_json,
    commit_generation,
    read_json,
    resolve_generation,
    sha256_file,
    sha256_json,
)

SCORE_SEMANTICS = "supervised_sigmoid_uncalibrated_public_defect_score"


def _torch():  # noqa: ANN202
    try:
        import torch
    except ImportError as e:
        raise ExecutionError("torch not installed; uv sync --extra neural") from e
    return torch


def pick_device(ft):  # noqa: ANN001, ANN201 - FinetuneCfg or CptCfg
    torch = _torch()
    if ft.device == "cuda" and not torch.cuda.is_available():
        raise ExecutionError("CUDA requested but unavailable (no silent CPU fallback)")
    if ft.device == "mps" and not torch.backends.mps.is_available():
        raise ExecutionError("MPS requested but unavailable")
    if ft.device == "cpu" and not ft.allow_cpu:
        raise ExecutionError("CPU encoder training requires allow_cpu: true in the finetune/cpt section")
    return torch.device(ft.device)


def seed_everything(seed: int) -> None:
    torch = _torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def state_hash(module) -> str:  # noqa: ANN001
    h = hashlib.sha256()
    for k, v in sorted(module.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().float().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def load_encoder(local_path: Path, revision: str, weights: tuple[str, str] | None = None):  # noqa: ANN201
    """`weights` = (file name, sha256) pins the actual checkpoint bytes, not just the revision string."""
    from transformers import AutoConfig, AutoModel

    src = read_json(local_path / "SOURCE.json")
    if src.get("resolved_revision") != revision:
        raise IntegrityError(f"encoder snapshot revision {src.get('resolved_revision')} != pinned {revision}")
    if weights is not None and sha256_file(local_path / weights[0]) != weights[1]:
        raise IntegrityError(f"{local_path / weights[0]} sha256 does not match the pinned weights_sha256")
    config = AutoConfig.from_pretrained(str(local_path), local_files_only=True)
    model, info = AutoModel.from_pretrained(str(local_path), config=config, local_files_only=True, add_pooling_layer=False,
                                            output_loading_info=True)
    return model, {"revision": revision, "missing_keys": sorted(info["missing_keys"]),
                   "unexpected_keys": sorted(info["unexpected_keys"])[:20], "hidden_size": config.hidden_size}


def build_head(hidden: int, n_feat: int, dropout: float, seed: int):  # noqa: ANN201
    torch = _torch()
    nn = torch.nn

    class FusionHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.feature_dense = nn.Linear(n_feat, hidden)
            self.dropout = nn.Dropout(dropout)
            self.out_proj = nn.Linear(2 * hidden, 1)

        def forward(self, cls, feats):  # noqa: ANN001, ANN202
            x = torch.cat([cls, torch.tanh(self.feature_dense(feats))], dim=-1)
            return self.out_proj(self.dropout(x)).squeeze(-1)

    # Dedicated generator: B2/B3/B4/B5 with the same seed get bit-identical head init (AT-09).
    g = torch.Generator().manual_seed(seed)
    head = FusionHead()
    with torch.no_grad():
        for p in head.parameters():
            if p.dim() > 1:
                bound = 1 / np.sqrt(p.shape[1])
                p.copy_((torch.rand(p.shape, generator=g) * 2 - 1) * bound)
            else:
                p.zero_()
    return head


@dataclass
class Encoded:
    ids: list[list[int]]
    feats: np.ndarray


def encode_inputs(tok, texts: list[str], max_length: int) -> list[list[int]]:  # noqa: ANN001
    ids = tok(texts, add_special_tokens=True, truncation=False)["input_ids"]
    over = sum(len(x) > max_length for x in ids)
    if over:
        raise IntegrityError(f"{over} inputs exceed {max_length} tokens; EvidenceView contract violated")
    return ids


def _batches(n: int, bs: int, order: np.ndarray):  # noqa: ANN202
    for i in range(0, n, bs):
        yield order[i:i + bs]


def _collate(enc: Encoded, idx: np.ndarray, pad_id: int, device):  # noqa: ANN001, ANN202
    torch = _torch()
    rows = [enc.ids[i] for i in idx]
    L = max(len(r) for r in rows)
    ids = torch.full((len(rows), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(rows), L), dtype=torch.long)
    for j, r in enumerate(rows):
        ids[j, :len(r)] = torch.tensor(r)
        mask[j, :len(r)] = 1
    return ids.to(device), mask.to(device), torch.tensor(enc.feats[idx], dtype=torch.float32, device=device)


class NeuralRun:
    """One B2-S or B3-S training run inside a run directory."""

    def __init__(self, cfg: StudyConfig, variant: str, seed: int, encoder_path: Path, run_dir: Path, tokenizer,  # noqa: ANN001
                 init_encoder: Path | None = None):
        if variant not in ("B2-S", "B3-S", "B4-S", "B5-S"):
            raise ExecutionError(f"NeuralRun does not implement {variant}")
        if (variant in ("B4-S", "B5-S")) != (init_encoder is not None):
            raise ExecutionError("B4-S/B5-S (and only they) fine-tune from a CPT encoder export")
        self.init_encoder = init_encoder
        if not isinstance(cfg.finetune, FinetuneCfg):
            raise ExecutionError("finetune section is not pinned")
        self.cfg, self.ft, self.variant, self.seed = cfg, cfg.finetune, variant, seed
        self.frozen = variant == "B2-S"
        self.encoder_path, self.run_dir, self.tok = encoder_path, run_dir, tokenizer
        self.pipe = StructuredPipeline(list(cfg.structured.features))

    def _autocast(self, device):  # noqa: ANN001, ANN202
        torch = _torch()
        if self.ft.precision == "bf16_encoder_autocast":
            return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        import contextlib
        return contextlib.nullcontext()

    def _load_encoder(self):  # noqa: ANN202
        pin = (self.cfg.model.weights_file, self.cfg.model.weights_sha256)
        if self.init_encoder is None:
            return load_encoder(self.encoder_path, self.cfg.model.revision, pin)
        from .cpt import load_cpt_encoder
        return load_cpt_encoder(self.init_encoder, self.encoder_path, self.cfg.model.revision, pin)

    def _optimizer(self, encoder, head):  # noqa: ANN001, ANN202
        torch = _torch()
        groups = [{"params": list(head.parameters()), "lr": self.ft.head_lr, "name": "head"}]
        if not self.frozen:
            groups.insert(0, {"params": list(encoder.parameters()), "lr": self.ft.encoder_lr, "name": "encoder"})
        return torch.optim.AdamW(groups, weight_decay=self.ft.weight_decay), [p for g in groups for p in g["params"]]

    def _cls(self, encoder, ids, mask, device):  # noqa: ANN001, ANN202
        with self._autocast(device):
            h = encoder(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]
        return h.float()

    def fit_predict(self, train: TrainingDatasetView, valid: QueryView, y_valid: np.ndarray, salt: str) -> dict:
        torch = _torch()
        require_training_view(train)
        device = pick_device(self.ft)
        seed_everything(self.seed)
        encoder, enc_info = self._load_encoder()
        encoder.to(device)
        x_tr = self.pipe.fit(train).transform(train.frame).astype(np.float32)
        x_va = self.pipe.transform(valid.frame).astype(np.float32)
        tr = Encoded(encode_inputs(self.tok, train.frame["query_text"].tolist(), self.cfg.model.max_length), x_tr)
        va = Encoded(encode_inputs(self.tok, valid.frame["query_text"].tolist(), self.cfg.model.max_length), x_va)
        y_tr = train.labels.to_numpy().astype(np.float32)
        head = build_head(enc_info["hidden_size"], x_tr.shape[1], self.ft.head_dropout, self.seed).to(device)
        head_init_hash = state_hash(head)
        enc_hash0 = state_hash(encoder)
        pad = self.tok.pad_token_id
        cache_info = None
        if self.frozen:
            for p in encoder.parameters():
                p.requires_grad_(False)
            encoder.eval()
            t0 = time.perf_counter()
            cls_tr = self._embed(encoder, tr, pad, device)
            cls_va = self._embed(encoder, va, pad, device)
            cache_info = {"pooling": "cls", "encoder_mode": "eval", "precision": self.ft.precision,
                          "encoder_state_sha256": enc_hash0, "seconds": time.perf_counter() - t0,
                          "key": sha256_json([enc_hash0, "cls", "eval", self.ft.precision,
                                              sha256_json(train.frame["query_text"].tolist()),
                                              sha256_json(valid.frame["query_text"].tolist())])}
        opt, params = self._optimizer(encoder, head)
        loss_fn = torch.nn.BCEWithLogitsLoss(reduction="sum")
        ck = self.run_dir / "checkpoints"
        state = {"epoch": 0, "best_ap": -1.0, "best_epoch": 0, "stale": 0, "history": [], "input_tokens": 0, "updates": 0}
        last = resolve_generation(ck, "last")
        if last is not None:
            state = self._resume(last, encoder, head, opt)
        gen = np.random.default_rng(self.seed)
        for _ in range(state["epoch"]):
            gen.permutation(len(y_tr))  # replay the shuffle stream so resumed epochs see the same order
        t_train = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        while state["epoch"] < self.ft.max_epochs and state["stale"] < self.ft.patience_epochs:
            epoch = state["epoch"] + 1
            order = gen.permutation(len(y_tr))
            head.train()
            if not self.frozen:
                encoder.train()
            mb, acc = self.ft.micro_batch_size, self.ft.gradient_accumulation_steps
            window = mb * acc
            loss_sum, n_seen, updates, tok_in = 0.0, 0, 0, 0
            for w in range(0, len(order), window):
                widx = order[w:w + window]
                opt.zero_grad(set_to_none=True)
                for idx in _batches(len(widx), mb, widx):
                    yb = torch.tensor(y_tr[idx], device=device)
                    if self.frozen:
                        cls = cls_tr[torch.as_tensor(idx, device=cls_tr.device)].to(device)
                        feats = torch.tensor(tr.feats[idx], device=device)
                    else:
                        ids, mask, feats = _collate(tr, idx, pad, device)
                        cls = self._cls(encoder, ids, mask, device)
                        tok_in += int(mask.sum())
                    loss = loss_fn(head(cls, feats), yb)
                    (loss / len(widx)).backward()  # exact mean over the (possibly short) accumulation window
                    loss_sum += float(loss.detach())
                    n_seen += len(idx)
                torch.nn.utils.clip_grad_norm_(params, self.ft.max_grad_norm)
                opt.step()
                updates += 1
            scores = self._predict(encoder, head, va, pad, device, cls_va if self.frozen else None)
            m = evaluate(valid.ids, y_valid, scores, salt)
            state["history"].append({"epoch": epoch, "train_loss": loss_sum / max(n_seen, 1), "optimizer_updates": updates,
                                     "validation_ap": m["ap"], "validation_roc_auc": m["roc_auc"]})
            state["epoch"] = epoch
            state["updates"] += updates
            state["input_tokens"] += tok_in
            if m["ap"] is not None and m["ap"] > state["best_ap"]:  # strict: ties keep the earlier checkpoint
                state.update(best_ap=m["ap"], best_epoch=epoch, stale=0)
                self._save_best(ck, encoder, head, valid.ids, scores, epoch)
            else:
                state["stale"] += 1
            self._checkpoint(ck, encoder, head, opt, state)
            print(f"{self.variant} seed={self.seed} epoch={epoch} loss={state['history'][-1]['train_loss']:.4f} "
                  f"val_ap={m['ap']:.4f} best={state['best_ap']:.4f}@{state['best_epoch']}", flush=True)
        train_s = time.perf_counter() - t_train
        if self.frozen and state_hash(encoder) != enc_hash0:
            raise IntegrityError("frozen encoder state changed during B2 training (AT-28)")
        best_dir = resolve_generation(ck, "best")
        best_scores = pd.read_parquet(best_dir / "validation-scores.parquet")
        if best_scores["change_id"].tolist() != valid.ids or read_json(best_dir / "epoch.json")["epoch"] != state["best_epoch"]:
            raise IntegrityError("best validation scores misaligned with the best checkpoint")
        peak = torch.cuda.max_memory_allocated() if device.type == "cuda" else None
        peak_reserved = torch.cuda.max_memory_reserved() if device.type == "cuda" else None
        frozen_tokens = int(sum(len(x) for x in tr.ids)) if self.frozen else 0
        return {"scores": best_scores["score"].to_numpy(), "history": state["history"], "best_epoch": state["best_epoch"],
                "best_validation_ap": state["best_ap"], "stopped_epoch": state["epoch"],
                "early_stopped": state["stale"] >= self.ft.patience_epochs, "encoder": enc_info,
                "encoder_init_state_sha256": enc_hash0, "head_init_state_sha256": head_init_hash,
                "frozen_encoder": self.frozen, "embedding_cache": cache_info, "pipeline": self.pipe.state(),
                "device": str(device), "train_seconds": train_s, "peak_cuda_bytes": peak, "peak_cuda_reserved_bytes": peak_reserved,
                "best_checkpoint_generation": best_dir.name,
                "learning_rates": {"encoder": None if self.frozen else self.ft.encoder_lr, "head": self.ft.head_lr},
                "token_accounting": {"unit": "nonpadding_input_tokens", "train_tokens_per_epoch": int(sum(len(x) for x in tr.ids)),
                                     "encoder_forward_train_tokens": state["input_tokens"] if not self.frozen else frozen_tokens,
                                     "frozen_embedding_pass_tokens": frozen_tokens, "optimizer_updates": state["updates"],
                                     "epochs": state["epoch"], "train_examples": len(tr.ids)}}

    def profile(self, train: TrainingDatasetView, updates: int) -> dict:
        """Spec T14: `updates` optimizer updates of B3-style full FT from the pinned init on a fixed batch order."""
        torch = _torch()
        require_training_view(train)
        device = pick_device(self.ft)
        seed_everything(self.seed)
        encoder, enc_info = self._load_encoder()
        encoder.to(device).train()
        x = self.pipe.fit(train).transform(train.frame).astype(np.float32)
        enc = Encoded(encode_inputs(self.tok, train.frame["query_text"].tolist(), self.cfg.model.max_length), x)
        y = train.labels.to_numpy().astype(np.float32)
        head = build_head(enc_info["hidden_size"], x.shape[1], self.ft.head_dropout, self.seed).to(device).train()
        opt, params = self._optimizer(encoder, head)
        loss_fn = torch.nn.BCEWithLogitsLoss(reduction="sum")
        order = np.random.default_rng(self.seed).permutation(len(y))
        mb, acc = self.ft.micro_batch_size, self.ft.gradient_accumulation_steps
        losses, n_ex, n_tok = [], 0, 0
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for u in range(updates):
            widx = order[(u * mb * acc) % len(order):][: mb * acc]
            opt.zero_grad(set_to_none=True)
            total = 0.0
            for idx in _batches(len(widx), mb, widx):
                ids, mask, feats = _collate(enc, idx, self.tok.pad_token_id, device)
                loss = loss_fn(head(self._cls(encoder, ids, mask, device), feats), torch.tensor(y[idx], device=device))
                (loss / len(widx)).backward()
                total += float(loss.detach())
                n_tok += int(mask.sum())
            torch.nn.utils.clip_grad_norm_(params, self.ft.max_grad_norm)
            opt.step()
            losses.append(total / len(widx))
            n_ex += len(widx)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        return {"precision": self.ft.precision, "device": str(device), "updates": updates, "examples": n_ex,
                "seconds": dt, "examples_per_second": n_ex / dt, "nonpad_tokens_per_second": n_tok / dt,
                "peak_cuda_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
                "losses": losses, "finite": bool(np.isfinite(losses).all())}

    def _embed(self, encoder, enc: Encoded, pad: int, device):  # noqa: ANN001, ANN202
        torch = _torch()
        out = []
        with torch.no_grad():
            for idx in _batches(len(enc.ids), self.ft.eval_batch_size, np.arange(len(enc.ids))):
                ids, mask, _ = _collate(enc, idx, pad, device)
                out.append(self._cls(encoder, ids, mask, device))
        return torch.cat(out)

    def _predict(self, encoder, head, enc: Encoded, pad: int, device, cached=None) -> np.ndarray:  # noqa: ANN001
        torch = _torch()
        head.eval()
        encoder.eval()
        scores = []
        with torch.no_grad():
            for idx in _batches(len(enc.ids), self.ft.eval_batch_size, np.arange(len(enc.ids))):
                feats = torch.tensor(enc.feats[idx], device=device)
                if cached is not None:
                    cls = cached[torch.as_tensor(idx, device=cached.device)].to(device)
                else:
                    ids, mask, _ = _collate(enc, idx, pad, device)
                    cls = self._cls(encoder, ids, mask, device)
                scores.append(torch.sigmoid(head(cls, feats)).float().cpu().numpy())
        s = np.concatenate(scores).astype(np.float64)
        if not np.isfinite(s).all():
            raise IntegrityError("non-finite validation scores")
        return s

    def _save_best(self, ck: Path, encoder, head, ids: list[str], scores: np.ndarray, epoch: int) -> None:  # noqa: ANN001
        from safetensors.torch import save_file
        tmp = ck / "best.tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        save_file({k: v.detach().cpu().contiguous() for k, v in head.state_dict().items()}, str(tmp / "head.safetensors"))
        if not self.frozen:
            save_file({k: v.detach().cpu().contiguous() for k, v in encoder.state_dict().items()}, str(tmp / "encoder.safetensors"))
        pd.DataFrame({"change_id": ids, "score": scores}).to_parquet(tmp / "validation-scores.parquet")
        atomic_write_json(tmp / "epoch.json", {"epoch": epoch})
        commit_generation(ck, "best", tmp)

    def _checkpoint(self, ck: Path, encoder, head, opt, state: dict) -> None:  # noqa: ANN001
        torch = _torch()
        tmp = ck / "last.tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        torch.save({"head": head.state_dict(), "encoder": None if self.frozen else encoder.state_dict(),
                    "optimizer": opt.state_dict(), "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
                   tmp / "state.pt")
        atomic_write_json(tmp / "trainer.json", state)
        commit_generation(ck, "last", tmp)

    def _resume(self, path: Path, encoder, head, opt) -> dict:  # noqa: ANN001
        torch = _torch()
        # map to CPU: RNG states must stay CPU ByteTensors; load_state_dict copies weights onto the live device.
        st = torch.load(path / "state.pt", map_location="cpu", weights_only=False)
        head.load_state_dict(st["head"])
        if not self.frozen:
            encoder.load_state_dict(st["encoder"])
        opt.load_state_dict(st["optimizer"])
        torch.set_rng_state(st["torch_rng"])
        if st["cuda_rng"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(st["cuda_rng"])
        return read_json(path / "trainer.json")
