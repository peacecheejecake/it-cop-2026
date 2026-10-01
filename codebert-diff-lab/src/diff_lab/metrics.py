"""Evaluation metrics (spec protocol §8, FR-12, AT-06/18).

AP is sklearn.metrics.average_precision_score with pos_label=1 (not trapezoidal PR-AUC).
AP is null (with a reason) only when the cohort has no positives (all-positive -> 1.0);
ROC-AUC is null unless both classes are present; an empty cohort makes every metric null. Recall@q uses
K = ceil(q*N) clamped to [1, N]; ties are broken by sha256(salt:change_id), never by
labels. Thresholds come from validation max F1, ties -> highest threshold, positive iff
score >= threshold.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .util import IntegrityError


def _check(change_ids: list[str], y: np.ndarray, s: np.ndarray) -> None:
    if not (len(change_ids) == len(y) == len(s)):
        raise IntegrityError("ids/labels/scores length mismatch")
    if len(set(change_ids)) != len(change_ids):
        raise IntegrityError("duplicate change_id in evaluation cohort")
    if not np.isfinite(s).all():
        raise IntegrityError("non-finite score; failed predictions must be reported, not scored")
    if not set(np.unique(y)) <= {0, 1}:
        raise IntegrityError("labels must be 0/1")


def tie_order(change_ids: list[str], scores: np.ndarray, salt: str) -> np.ndarray:
    tb = [hashlib.sha256(f"{salt}:{c}".encode()).hexdigest() for c in change_ids]
    return np.array(sorted(range(len(scores)), key=lambda i: (-scores[i], tb[i])))


def recall_at(change_ids: list[str], y: np.ndarray, s: np.ndarray, q: float, salt: str) -> dict:
    n, pos = len(y), int(y.sum())
    if n == 0:
        return {"k": 0, "selection_rate": None, "tp": 0, "recall": None, "precision": None, "boundary_tie_size": 0,
                "null_reason": "empty cohort"}
    k = min(max(math.ceil(q * n), 1), n)
    order = tie_order(change_ids, s, salt)
    sel = order[:k]
    boundary = s[order[k - 1]]
    tie_size = int((s == boundary).sum())
    tp = int(y[sel].sum())
    return {"k": k, "selection_rate": k / n, "tp": tp, "recall": (tp / pos) if pos else None,
            "precision": tp / k, "boundary_tie_size": tie_size, "null_reason": None if pos else "no positives in cohort"}


def best_threshold(y: np.ndarray, s: np.ndarray) -> dict:
    order = np.argsort(-s, kind="mergesort")
    ss, yy = s[order], y[order]
    tp = np.cumsum(yy)
    fp = np.cumsum(1 - yy)
    last = np.r_[ss[1:] != ss[:-1], True]
    thr, tp, fp = ss[last], tp[last], fp[last]
    pos = y.sum()
    prec = tp / (tp + fp)
    rec = tp / pos if pos else np.zeros_like(tp, dtype=float)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-300), 0.0)
    best = f1.max()
    i = int(np.flatnonzero(f1 == best)[0])
    return {"threshold": float(thr[i]), "f1": float(f1[i]), "policy": "validation_max_f1_then_highest_threshold"}


def at_threshold(y: np.ndarray, s: np.ndarray, threshold: float) -> dict:
    pred = s >= threshold
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"threshold": threshold, "precision": p, "recall": r, "f1": (2 * p * r / (p + r)) if p + r else 0.0,
            "predicted_positive": int(pred.sum())}


def evaluate(change_ids: list[str], y, s, salt: str, threshold: float | None = None) -> dict:
    y, s = np.asarray(y, dtype=int), np.asarray(s, dtype=float)
    _check(change_ids, y, s)
    n, pos = len(y), int(y.sum())
    two = len(np.unique(y)) == 2
    ap_reason = "empty cohort" if n == 0 else ("no positives in cohort" if pos == 0 else None)
    auc_reason = "empty cohort" if n == 0 else (None if two else "single class in cohort")
    out = {"n": n, "positives": pos, "prevalence": float(y.mean()) if n else None,
           "ap": float(average_precision_score(y, s, pos_label=1)) if ap_reason is None else None,
           "roc_auc": float(roc_auc_score(y, s)) if auc_reason is None else None,
           "null_reasons": {"ap": ap_reason, "roc_auc": auc_reason},
           "null_reason": ap_reason or auc_reason,
           "recall_at_5pct": recall_at(change_ids, y, s, 0.05, salt),
           "recall_at_10pct": recall_at(change_ids, y, s, 0.10, salt),
           "unique_scores": int(len(np.unique(s)))}
    if threshold is not None:
        out["at_threshold"] = at_threshold(y, s, threshold)
    return out
