"""Commit-level evaluation of offline predictions against hand-made labels (internal trial, not the M7 protocol).

Labels live only in the labeling sheet and are read only here, after `predict` has written its
label-free output; nothing feeds back into the model, threshold or input. The sheet is made
before scoring and never shows scores: a sample is chosen by a salted sha256 rank of change_id,
so the labeled cohort is independent of the model. Labeling only high-scoring commits would bias
AP and recall; `--sample` exists so that a random subset can be labeled instead of everything.

Metrics reuse `metrics.evaluate` (AP, ROC-AUC, Recall@5/10% with hash tie-break, F1 at the frozen
validation threshold). Thresholds and scores stay uncalibrated public-defect scores.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from .metrics import evaluate
from .util import ConfigError, IntegrityError, PolicyError, atomic_write_json, read_json, sha256_file

SHEET_COLUMNS = ["change_id", "project", "commit_sha", "authored_at", "primary_language", "subject", "n_added_lines",
                 "n_deleted_lines", "label", "note"]


def _rank(change_ids: pd.Series, salt: str) -> pd.Series:
    return change_ids.map(lambda c: hashlib.sha256(f"{salt}:{c}".encode()).hexdigest())


def label_sheet(extract_dir: Path, out: Path, sample: int | None, salt: str) -> dict:
    """Score-blind CSV for labeling (label column empty). With --sample, a salted-hash random subset."""
    if out.exists():
        raise PolicyError(f"{out} exists; labeling sheets are never overwritten")
    meta = pd.read_parquet(extract_dir / "meta.parquet")
    if sample is not None:
        if sample <= 0:
            raise ConfigError("--sample must be positive")
        meta = meta.assign(_r=_rank(meta["change_id"], salt)).sort_values("_r").head(sample).drop(columns="_r")
    sheet = meta.sort_values(["project", "authored_at"]).assign(label="", note="")[SHEET_COLUMNS]
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.to_csv(out, index=False, encoding="utf-8-sig")
    info = {"rows": int(len(sheet)), "of": int(len(pd.read_parquet(extract_dir / "meta.parquet"))),
            "selection": "all" if sample is None else f"salted sha256 rank of change_id (salt={salt!r}), score-blind",
            "extract_manifest_sha256": sha256_file(extract_dir / "extract-manifest.json"), "sheet_sha256": sha256_file(out)}
    atomic_write_json(out.with_suffix(".manifest.json"), info)
    return info


def read_labels(path: Path) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if not {"change_id", "label"} <= set(raw.columns):
        raise ConfigError("labels file needs columns change_id,label")
    if not raw["change_id"].is_unique:
        raise IntegrityError("duplicate change_id in labels file")
    lab = raw["label"].str.strip()
    bad = sorted(set(lab[~lab.isin(["0", "1", ""])]))
    if bad:
        raise ConfigError(f"labels must be 1 (defect-inducing), 0 (clean) or empty (unlabeled); found {bad[:5]}")
    done = raw.loc[lab != "", ["change_id"]].assign(label=lab[lab != ""].astype(int).to_numpy())
    return done, {"rows": int(len(raw)), "labeled": int(len(done)), "unlabeled": int((lab == "").sum()),
                  "positives": int(done["label"].sum()), "labels_sha256": sha256_file(path)}


def _bootstrap_ap(y: np.ndarray, s: np.ndarray, n_boot: int, seed: int) -> list[float] | None:
    if y.sum() < 2 or (1 - y).sum() < 2:
        return None
    rng = np.random.default_rng(seed)
    aps = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            aps.append(average_precision_score(y[i], s[i]))
    return [float(np.quantile(aps, 0.025)), float(np.quantile(aps, 0.975))]


def _slim(m: dict) -> dict:
    out = {k: m[k] for k in ("n", "positives", "prevalence", "ap", "roc_auc")}
    out["lift_ap_over_prevalence"] = (m["ap"] / m["prevalence"]) if m["ap"] is not None and m["prevalence"] else None
    for q in ("5pct", "10pct"):
        r = m[f"recall_at_{q}"]
        out[f"recall_at_{q}"], out[f"precision_at_{q}"], out[f"k_at_{q}"] = r["recall"], r["precision"], r["k"]
    if "at_threshold" in m:
        out["at_threshold"] = m["at_threshold"]
    return out


def evaluate_labels(predictions: Path, labels: Path, meta_path: Path, bundle_dir: Path, out_dir: Path, salt: str = "label-eval-v1",
                    n_boot: int = 2000, min_group: int = 30) -> dict:
    if out_dir.exists():
        raise PolicyError(f"{out_dir} exists; evaluations are never overwritten")
    pm = read_json(predictions.parent / "offline-manifest.json")
    if pm.get("labels_read") is not False:
        raise PolicyError("predictions were not produced by the label-free offline predictor")
    pred = pd.read_parquet(predictions)
    lab, lab_info = read_labels(labels)
    unknown = set(lab["change_id"]) - set(pred["change_id"])
    if unknown:
        raise IntegrityError(f"{len(unknown)} labeled change_id(s) have no prediction (different extract?)")
    meta = pd.read_parquet(meta_path)[["change_id", "project", "commit_sha", "primary_language", "authored_at"]]
    fz = read_json(bundle_dir / "freeze.json")
    if fz["freeze_id"] != pm["freeze_id"]:
        raise IntegrityError("bundle freeze differs from the one that produced the predictions")
    thr = {e["run_id"]: e["threshold"] for e in fz["runs"]}
    df = pred.merge(lab, on="change_id", how="inner").merge(meta, on="change_id", how="left", validate="m:1")
    runs, scored = {}, []
    for run_id, g in df.groupby("run_id", sort=True):
        g = g.reset_index(drop=True)
        ids, y, s = g["change_id"].tolist(), g["label"].to_numpy(), g["score"].to_numpy()
        overall = _slim(evaluate(ids, y, s, salt, thr[run_id]))
        overall["ap_bootstrap_95ci"] = _bootstrap_ap(y, s, n_boot, 0)
        groups = {}
        for col in ("project", "primary_language"):
            groups[col] = {}
            for key, h in g.groupby(col, sort=True):
                if len(h) < min_group:
                    groups[col][key] = {"n": int(len(h)), "positives": int(h["label"].sum()), "note": f"fewer than {min_group} labeled"}
                    continue
                groups[col][key] = _slim(evaluate(h["change_id"].tolist(), h["label"].to_numpy(), h["score"].to_numpy(), salt, thr[run_id]))
        runs[run_id] = {"variant_id": g["variant_id"].iloc[0], "seed": int(g["seed"].iloc[0]), "threshold": thr[run_id],
                        "overall": overall, "by": groups}
        rank = (-g["score"]).rank(method="first").astype(int)
        scored.append(g.assign(rank=rank, above_threshold=g["score"] >= thr[run_id])[
            ["run_id", "variant_id", "seed", "change_id", "project", "commit_sha", "primary_language", "authored_at", "score", "rank",
             "above_threshold", "label"]])
    if not runs:
        raise ConfigError("no labeled change has a prediction")
    out_dir.mkdir(parents=True)
    pd.concat(scored).sort_values(["run_id", "rank"]).to_csv(out_dir / "scored-labeled.csv", index=False, encoding="utf-8-sig")
    report = {"freeze_id": pm["freeze_id"], "labels": lab_info, "predicted_changes": int(pred["change_id"].nunique()),
              "evaluated_changes": int(df["change_id"].nunique()),
              "label_coverage": df["change_id"].nunique() / pred["change_id"].nunique(),
              "score_semantics": pm.get("score_semantics"), "tie_salt": salt, "bootstrap": {"unit": "change", "reps": n_boot},
              "runs": runs,
              "caveats": ["scores are uncalibrated public-defect scores; the threshold is the frozen public-validation max-F1 threshold",
                          "AP/recall are unbiased only if the labeled cohort was not chosen by score (use label-sheet --sample)",
                          "the model was trained on 21 Java projects; non-Java groups are out of distribution"]}
    atomic_write_json(out_dir / "label-eval.json", report)
    return {r: {"variant": v["variant_id"], "seed": v["seed"], **{k: v["overall"][k] for k in
                                                                  ("n", "positives", "ap", "ap_bootstrap_95ci", "roc_auc",
                                                                   "recall_at_10pct", "lift_ap_over_prevalence")}}
            for r, v in runs.items()}
