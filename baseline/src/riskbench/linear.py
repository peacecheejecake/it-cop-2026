from __future__ import annotations

import time
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression

from .artifacts import seal_predictions, verify_lock
from .common import (BenchError, assert_binary, fingerprint_record, guard_overlap, new_dir, read_json,
                     seal_files, verify_files, versions, write_json)
from .data import FEATURES, FEATURE_VERSION, load_inputs, load_labeled
from .metrics import ranking_metrics


def matrix(rows: list[dict]) -> np.ndarray:
    values = [[r["features"][k] for k in FEATURES] for r in rows]
    x = np.array([[np.nan if v is None else float(v) for v in row] for row in values], dtype=np.float64)
    if np.isinf(x).any() or (x[~np.isnan(x)] < 0).any():
        raise BenchError("Static features must be nonnegative finite values or null")
    return np.log1p(x)


def fit_scaler(x: np.ndarray) -> dict:
    medians = [float(np.median(col[~np.isnan(col)])) if np.isfinite(col).any() else 0.0 for col in x.T]
    z = np.concatenate([np.where(np.isnan(x), np.array(medians), x), np.isnan(x).astype(float)], axis=1)
    mean, scale = z.mean(0), z.std(0)
    scale[scale == 0] = 1.0
    return {"median": medians, "mean": mean.tolist(), "scale": scale.tolist()}


def transform(x: np.ndarray, scaler: dict) -> np.ndarray:
    z = np.concatenate([np.where(np.isnan(x), np.array(scaler["median"]), x), np.isnan(x).astype(float)], axis=1)
    return (z - np.array(scaler["mean"])) / np.array(scaler["scale"])


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -700, 700)
    out = np.empty_like(z, dtype=float); pos = z >= 0
    out[pos] = 1/(1+np.exp(-z[pos])); e = np.exp(z[~pos]); out[~pos] = e/(1+e)
    return out


def train_tabular(train: str, valid: str, out: str, cs: list[float] | None = None, seed: int = 42) -> dict:
    tr, tl, tm = load_labeled(train, "train"); va, vl, vm = load_labeled(valid, "valid")
    y = assert_binary(tl, True); vy = assert_binary(vl, True)
    guard_overlap([fingerprint_record(r) for r in tr], va)
    label_types = {r["label_type"] for r in tl + vl}
    if len(label_types) != 1:
        raise BenchError("Public train/validation outcome definitions differ")
    root = new_dir(out); x = matrix(tr); scaler = fit_scaler(x)
    x = transform(x, scaler); vx = transform(matrix(va), scaler)
    trials = []; best = None
    for c in cs or [0.1, 1.0, 10.0]:
        if c <= 0:
            raise BenchError("C must be positive")
        model = LogisticRegression(C=c, max_iter=2000, random_state=seed, class_weight=None)
        model.fit(x, y)
        ap = ranking_metrics(vy, model.predict_proba(vx)[:,1])["ap"]
        trials.append({"C": c, "validation_ap": ap})
        if best is None or ap > best[0]:
            best = (ap, model, c)
    model = best[1]
    artifact = {"kind": "tabular_lr", "feature_names": list(FEATURES), "feature_version": FEATURE_VERSION,
                "scaler": scaler, "coef": model.coef_[0].tolist(), "intercept": float(model.intercept_[0]),
                "C": best[2], "seed": seed, "training_label_type": next(iter(label_types)),
                "train_fingerprints": [fingerprint_record(r) for r in tr],
                "selection_fingerprints": [fingerprint_record(r) for r in va],
                "train_inputs_sha256": tm["files"]["inputs.jsonl"], "valid_inputs_sha256": vm["files"]["inputs.jsonl"],
                "retrospective_labels": tm["retrospective_labels"], "public_validation_trials": trials,
                "synthetic": any(r["synthetic"] for r in tr), "versions": versions()}
    write_json(root / "model.json", artifact)
    write_json(root / "manifest.json", {"kind": "tabular_lr", "files": seal_files(root, ["model.json"])})
    return {"kind": artifact["kind"], "selected_C": best[2], "validation_ap": best[0], "synthetic": artifact["synthetic"]}


def predict_tabular(model_dir: str, dataset: str, out: str, run_name: str, lock: str | None = None) -> None:
    root = Path(model_dir); mm = read_json(root / "manifest.json"); verify_files(root, mm["files"])
    model = read_json(root / "model.json"); rows, dm = load_inputs(dataset)
    internal = any(r["domain"] == "internal" for r in rows)
    lock_hash = verify_lock(lock, [model_dir], internal)
    guard_overlap(model["train_fingerprints"] + (model["selection_fingerprints"] if internal else []), rows)
    t = time.perf_counter()
    scores = sigmoid(transform(matrix(rows), model["scaler"]) @ np.array(model["coef"]) + model["intercept"])
    elapsed = (time.perf_counter()-t)*1000
    preds = [{"id": r["id"], "score": float(s), "status": "ok"} for r,s in zip(rows, scores)]
    seal_predictions(out, preds, {"run_name": run_name, "kind": "tabular_lr", "training_label_type": model["training_label_type"],
                                 "inputs_sha256": dm["files"]["inputs.jsonl"], "lock_sha256": lock_hash,
                                 "inference_ms": elapsed, "score_semantics": "uncalibrated source-model score"})


def predict_rule(dataset: str, out: str, run_name: str = "rule_churn") -> None:
    rows, dm = load_inputs(dataset)
    # Fixed and public a priori: no internal thresholds or learned quantities.
    preds = [{"id": r["id"], "score": float(r["features"]["lines_added"] + r["features"]["lines_deleted"]),
              "status": "ok"} for r in rows]
    seal_predictions(out, preds, {"run_name": run_name, "kind": "fixed_churn", "training_label_type": "bug_inducing_commit",
                                 "inputs_sha256": dm["files"]["inputs.jsonl"], "score_semantics": "added+deleted lines; ranking only"})
