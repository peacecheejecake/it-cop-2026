from __future__ import annotations

import math
import random
import time
from itertools import islice
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .artifacts import seal_predictions, verify_lock
from .common import (BenchError, assert_binary, fingerprint_record, guard_overlap, new_dir,
                     read_json, seal_files, verify_files, versions, write_json)
from .data import load_inputs, load_labeled
from .metrics import ranking_metrics
from .views import get_tokenizer, require_view, tokenizer_signature


class DemoEncoder(nn.Module):
    """Random tiny Transformer. Synthetic pipeline checks only; NOT a pretrained model."""
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=24, max_position_embeddings=1024)
        self.embedding = nn.Embedding(259, 24, padding_idx=0)
        self.position = nn.Embedding(1024, 24)
        layer = nn.TransformerEncoderLayer(24, 4, 48, dropout=0.0, batch_first=True)
        self.layers = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
    def forward(self, input_ids, attention_mask):
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)[None, :]
        x = self.embedding(input_ids) + self.position(positions)
        return SimpleNamespace(last_hidden_state=self.layers(x, src_key_padding_mask=~attention_mask.bool()))


def mean_pool(hidden, mask):
    weights = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * weights).sum(1) / weights.sum(1).clamp_min(1)


class RiskNet(nn.Module):
    """A and B have exactly the same masked mean pooling and linear binary head."""
    def __init__(self, encoder, frozen: bool):
        super().__init__()
        self.encoder = encoder
        self.frozen = frozen
        self.head = nn.Linear(encoder.config.hidden_size, 1)
        if frozen:
            for p in encoder.parameters():
                p.requires_grad_(False)
            encoder.eval()
    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.encoder.eval()  # Dropout must stay off for frozen embedding extraction.
        return self
    def embeddings(self, ids, mask):
        if self.frozen:
            with torch.no_grad():
                hidden = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        else:
            hidden = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        return mean_pool(hidden, mask)
    def forward(self, input_ids=None, attention_mask=None, embeddings=None):
        x = embeddings if embeddings is not None else self.embeddings(input_ids, attention_mask)
        return self.head(x).squeeze(-1)


class Rows(Dataset):
    def __init__(self, rows, labels=None):
        self.rows = rows; self.labels = labels
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, i):
        return {"ids": self.rows[i]["input_ids"], "label": None if self.labels is None else self.labels[i]}


class Embeddings(Dataset):
    def __init__(self, path, labels):
        self.x = np.load(path, mmap_mode="r", allow_pickle=False); self.y = labels
    def __len__(self):
        return len(self.y)
    def __getitem__(self, i):
        return {"embeddings": torch.tensor(np.array(self.x[i], copy=True)),
                "labels": torch.tensor(self.y[i], dtype=torch.float32)}


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id
    def __call__(self, batch):
        max_len = max(len(x["ids"]) for x in batch)
        ids = torch.full((len(batch), max_len), self.pad_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, row in enumerate(batch):
            n = len(row["ids"]); ids[i, :n] = torch.tensor(row["ids"]); mask[i, :n] = 1
        result = {"input_ids": ids, "attention_mask": mask}
        if batch[0]["label"] is not None:
            result["labels"] = torch.tensor([r["label"] for r in batch], dtype=torch.float32)
        return result


def device_for(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise BenchError("CUDA requested but unavailable")
    if name == "mps" and not torch.backends.mps.is_available():
        raise BenchError("MPS requested but unavailable")
    return torch.device(name)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Reduces avoidable nondeterminism; cross-device bitwise equivalence is not promised.
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def train_epoch(model, loader, optimizer, device, accumulation=1, pos_weight=None):
    if accumulation < 1:
        raise BenchError("accumulation must be positive")
    model.train(); stream = iter(loader); loss_sum = count = steps = 0
    while True:
        group = list(islice(stream, accumulation))
        if not group:
            break
        window_n = sum(len(batch["labels"]) for batch in group)
        optimizer.zero_grad(set_to_none=True)
        for batch in group:
            b = {k:v.to(device) for k,v in batch.items()}; y = b.pop("labels")
            logits = model(**b)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight, reduction="sum")
            # Correct denominator for a final short accumulation window and uneven microbatches.
            (loss/window_n).backward()
            loss_sum += loss.detach().item(); count += len(y)
        nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step(); steps += 1
    return {"loss": loss_sum/max(count, 1), "optimizer_steps": steps, "examples": count}


def score_loader(model, loader, device):
    model.eval(); scores = []
    with torch.inference_mode():
        for batch in loader:
            b = {k:v.to(device) for k,v in batch.items() if k != "labels"}
            scores.extend(torch.sigmoid(model(**b)).cpu().float().tolist())
    return scores


def cache_embeddings(model, rows, labels, path, batch_size, pad_id, device):
    loader = DataLoader(Rows(rows), batch_size=batch_size, shuffle=False, collate_fn=Collator(pad_id), num_workers=0)
    model.eval()
    x = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32,
                                 shape=(len(rows), model.encoder.config.hidden_size))
    offset = 0
    with torch.inference_mode():
        for b in loader:
            emb = model.embeddings(b["input_ids"].to(device), b["attention_mask"].to(device)).cpu().float().numpy()
            x[offset:offset+len(emb)] = emb; offset += len(emb)
    x.flush(); del x
    return Embeddings(path, labels)


def save_network(model, root, demo):
    from safetensors.torch import save_file
    if demo:
        write_json(root / "encoder_config.json", {"model_type": "riskbench_demo"})
    else:
        write_json(root / "encoder_config.json", model.encoder.config.to_dict())
    save_file({name: value.detach().cpu().contiguous() for name,value in model.state_dict().items()},
              str(root / "model.safetensors"))


def load_network(root, meta):
    from safetensors.torch import load_file
    if meta["demo_encoder"]:
        encoder = DemoEncoder()
    else:
        from transformers import AutoConfig, AutoModel
        conf = read_json(root / "encoder_config.json"); model_type = conf.pop("model_type")
        config = AutoConfig.for_model(model_type, **conf)
        encoder = AutoModel.from_config(config, trust_remote_code=False)
    net = RiskNet(encoder, frozen=meta["mode"] == "frozen")
    net.load_state_dict(load_file(str(root / "model.safetensors"), device="cpu"), strict=True)
    return net


def train_neural(train: str, valid: str, out: str, model_name: str, mode: str,
                 epochs: int = 5, batch_size: int = 8, accumulation: int = 4,
                 encoder_lr: float = 2e-5, head_lr: float = 1e-3, patience: int = 2,
                 seed: int = 42, device: str = "auto", revision: str | None = None,
                 allow_download: bool = False, allow_cpu_training: bool = False,
                 class_weight: str = "none") -> dict:
    if mode not in ("frozen", "finetune") or epochs < 1 or batch_size < 1 or patience < 1 or accumulation < 1:
        raise BenchError("Invalid neural training configuration")
    if encoder_lr <= 0 or head_lr <= 0:
        raise BenchError("Learning rates must be positive")
    tr, tl, tm = load_labeled(train, "train"); va, vl, vm = load_labeled(valid, "valid")
    y = assert_binary(tl, True); vy = assert_binary(vl, True)
    tv = require_view(tr, tm); vv = require_view(va, vm)
    if tv["policy_hash"] != vv["policy_hash"]:
        raise BenchError("Public train/valid input views differ")
    guard_overlap([fingerprint_record(r) for r in tr], va)
    types = {r["label_type"] for r in tl+vl}
    if len(types) != 1:
        raise BenchError("Public outcome definitions differ")
    demo = model_name == "demo-random"
    if demo and (not all(r["synthetic"] for r in tr+va) or not tv["synthetic_tokenizer"]):
        raise BenchError("Random demo encoder can only use explicit synthetic data and demo tokenizer")
    if not demo and tv["synthetic_tokenizer"]:
        raise BenchError("Real CodeBERT requires its actual tokenizer; never use demo token IDs")
    d = device_for(device)
    if mode == "finetune" and d.type == "cpu" and not (allow_cpu_training or demo):
        raise BenchError("Full CPU fine-tuning requires explicit --allow-cpu-training; prefer an approved training GPU")
    seed_all(seed)
    if demo:
        encoder = DemoEncoder()
    else:
        from transformers import AutoModel
        tok = get_tokenizer(model_name, revision, allow_download)
        if tokenizer_signature(tok) != tv["tokenizer_signature"]:
            raise BenchError("Model tokenizer does not match the prepared input view")
        encoder = AutoModel.from_pretrained(model_name, revision=revision, local_files_only=not allow_download,
                                             trust_remote_code=False)
    model = RiskNet(encoder, frozen=mode == "frozen").to(d)
    root = new_dir(out); pad = tv["pad_token_id"]; cache = root / "cache"; cache.mkdir()
    generator = torch.Generator().manual_seed(seed)
    if mode == "frozen":
        train_ds = cache_embeddings(model, tr, y, cache/"train.npy", batch_size, pad, d)
        valid_ds = cache_embeddings(model, va, vy, cache/"valid.npy", batch_size, pad, d)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=generator, num_workers=0)
        valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    else:
        train_loader = DataLoader(Rows(tr, y), batch_size=batch_size, shuffle=True, generator=generator,
                                  collate_fn=Collator(pad), num_workers=0)
        valid_loader = DataLoader(Rows(va, vy), batch_size=batch_size, shuffle=False,
                                  collate_fn=Collator(pad), num_workers=0)
    groups = [{"params": model.head.parameters(), "lr": head_lr}]
    if mode == "finetune":
        groups.append({"params": model.encoder.parameters(), "lr": encoder_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
    pw = None
    if class_weight == "balanced":
        pw = torch.tensor((len(y)-sum(y))/sum(y), dtype=torch.float32, device=d)
    elif class_weight != "none":
        raise BenchError("class_weight must be none/balanced")
    best = -float("inf"); stale = 0; history = []; best_epoch = 0; started = time.perf_counter()
    for epoch in range(1, epochs+1):
        fit = train_epoch(model, train_loader, optimizer, d, accumulation, pw)
        metrics = ranking_metrics(vy, score_loader(model, valid_loader, d))
        history.append({"epoch": epoch, **fit, "validation_ap": metrics["ap"]})
        print(f"{mode}: epoch={epoch} public_validation_ap={metrics['ap']:.6f}", flush=True)
        if metrics["ap"] > best:
            best = metrics["ap"]; best_epoch = epoch; stale = 0; save_network(model, root, demo)
        else:
            stale += 1
        if stale >= patience:
            break
    # Caches are derived public embeddings, not needed for inference. Avoid packaging them into model artifacts.
    if mode == "frozen":
        del train_loader, valid_loader, train_ds, valid_ds
    for p in cache.glob("*.npy"):
        p.unlink()
    cache.rmdir()
    meta = {"kind": "codebert", "mode": mode, "base_model": model_name, "revision": revision,
            "resolved_revision": getattr(encoder.config, "_commit_hash", None), "demo_encoder": demo,
            "training_label_type": next(iter(types)), "pooling": "attention-masked-mean", "head": "linear-logit",
            "view": tv, "seed": seed, "epochs_requested": epochs, "selected_epoch": best_epoch,
            "encoder_lr": encoder_lr, "head_lr": head_lr, "batch_size": batch_size, "accumulation": accumulation,
            "patience": patience, "class_weight": class_weight, "history": history,
            "training_seconds": time.perf_counter()-started, "device": str(d),
            "train_fingerprints": [fingerprint_record(r) for r in tr],
            "selection_fingerprints": [fingerprint_record(r) for r in va],
            "train_inputs_sha256": tm["files"]["inputs.jsonl"], "valid_inputs_sha256": vm["files"]["inputs.jsonl"],
            "retrospective_labels": tm["retrospective_labels"], "versions": versions()}
    write_json(root/"model.json", meta)
    write_json(root/"manifest.json", {"kind": "codebert", "files": seal_files(root, ["model.json", "encoder_config.json", "model.safetensors"])})
    return {"mode": mode, "best_public_validation_ap": best, "selected_epoch": best_epoch, "demo_encoder": demo}


def predict_neural(model_dir: str, dataset: str, out: str, run_name: str, batch_size: int = 8,
                   device: str = "auto", lock: str | None = None) -> None:
    root = Path(model_dir); mm = read_json(root/"manifest.json"); verify_files(root, mm["files"])
    meta = read_json(root/"model.json"); rows, dm = load_inputs(dataset); view = require_view(rows, dm)
    if view["policy_hash"] != meta["view"]["policy_hash"]:
        raise BenchError("Target view differs from frozen source preprocessing")
    if meta["demo_encoder"] and not all(r["synthetic"] for r in rows):
        raise BenchError("Random demo model cannot score real data")
    internal = any(r["domain"] == "internal" for r in rows)
    lock_hash = verify_lock(lock, [model_dir], internal)
    guard_overlap(meta["train_fingerprints"]+(meta["selection_fingerprints"] if internal else []), rows)
    d = device_for(device); model = load_network(root, meta).to(d)
    loader = DataLoader(Rows(rows), batch_size=batch_size, shuffle=False, collate_fn=Collator(view["pad_token_id"]), num_workers=0)
    start = time.perf_counter(); scores = score_loader(model, loader, d); elapsed = time.perf_counter()-start
    seal_predictions(out, [{"id": r["id"], "score": float(s), "status": "ok"} for r,s in zip(rows,scores)],
                     {"run_name": run_name, "kind": meta["mode"], "training_label_type": meta["training_label_type"],
                      "inputs_sha256": dm["files"]["inputs.jsonl"], "lock_sha256": lock_hash,
                      "inference_ms": elapsed*1000, "device": str(d), "batch_size": batch_size,
                      "demo_encoder": meta["demo_encoder"], "score_semantics": "uncalibrated source-classifier sigmoid"})
