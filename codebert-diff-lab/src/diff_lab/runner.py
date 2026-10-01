"""Run execution and the run-folder contract (spec implementation-plan §4.1, FR-11/12/24).

run_id = hash(study config, variant, seed, data lineage, code SHA). A completed run is
never overwritten; a failed attempt is kept with failure.json. Predictors see label-free
QueryViews; labels are joined only here, in the evaluator step. This runner evaluates
public validation only; the public test is sealed until a study freeze exists.
"""
from __future__ import annotations

import os
import platform
import subprocess
import time
import traceback
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

import pandas as pd

from .config import StudyConfig, require_pinned, resolve_local_path
from .metrics import best_threshold, evaluate
from .models import MODELS, SCORE_SEMANTICS
from .policy import QueryView, TrainingDatasetView
from .registry import MATRIX_VERSION, Registry
from .util import ExecutionError, IntegrityError, atomic_write_json, read_json, sha256_file, sha256_json

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NEEDS_EVIDENCE = {"B1-TFIDF-S", "B2-S", "B3-S"}
NEURAL = {"B2-S", "B3-S"}


def paths(cfg: StudyConfig, data_dir: Path) -> dict[str, Path]:
    snap = data_dir / "snapshots" / cfg.dataset.snapshot_id
    split = data_dir / "splits" / cfg.dataset.snapshot_id / cfg.dataset.split_id
    m = cfg.model
    ev = data_dir / "evidence" / cfg.dataset.snapshot_id / cfg.dataset.split_id / \
        f"{m.renderer}-L{m.max_length}-M{m.max_message_tokens}-{m.revision[:12]}"
    return {"snapshot": snap, "split": split, "evidence": ev}


def git_state() -> dict:
    def run(*a: str) -> str:
        return subprocess.run(["git", *a], capture_output=True, text=True, cwd=PROJECT_ROOT, check=True).stdout.strip()
    try:
        return {"sha": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain", "--", ".")), "source": "git"}
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    # Remote bundles are `git archive` exports without .git; the bundler writes the commit here.
    marker = PROJECT_ROOT / "CODE_SHA"
    if marker.is_file():
        return {"sha": marker.read_text().strip(), "dirty": None, "source": "bundle-marker"}
    return {"sha": None, "dirty": None, "source": None}


def environment() -> dict:
    pkgs = {p: metadata.version(p) for p in ("numpy", "pandas", "pyarrow", "scikit-learn", "scipy", "transformers",
                                             "tokenizers", "pydantic") if _installed(p)}
    if _installed("lightgbm"):
        pkgs["lightgbm"] = metadata.version("lightgbm")
    lock = PROJECT_ROOT / "uv.lock"
    env = {"python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine(),
           "cpu_count": os.cpu_count(), "packages": pkgs, "uv_lock_sha256": sha256_file(lock) if lock.exists() else None,
           "git": git_state()}
    if _installed("torch"):
        import torch
        env["packages"]["torch"] = torch.__version__
        env["cuda"] = {"available": torch.cuda.is_available(), "version": torch.version.cuda,
                       "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                       "tf32_matmul": torch.backends.cuda.matmul.allow_tf32}
        env["mps_available"] = torch.backends.mps.is_available()
    return env


def _installed(name: str) -> bool:
    try:
        metadata.version(name)
        return True
    except metadata.PackageNotFoundError:
        return False


def load_frames(cfg: StudyConfig, data_dir: Path, variant: str) -> tuple[pd.DataFrame, pd.Series, dict]:
    p = paths(cfg, data_dir)
    snap_m, split_m = p["snapshot"] / "manifest.json", p["split"] / "manifest.json"
    for f in (snap_m, split_m):
        if not f.exists():
            raise IntegrityError(f"missing {f}; run data import/split first")
    snap = read_json(snap_m)
    for t, digest in snap["table_sha256"].items():
        if sha256_file(p["snapshot"] / f"{t}.parquet") != digest:
            raise IntegrityError(f"snapshot table {t} hash mismatch")
    sp = pd.read_parquet(p["split"] / "splits.parquet")
    if sha256_file(p["split"] / "splits.parquet") != read_json(split_m)["splits_parquet_sha256"]:
        raise IntegrityError("split membership hash mismatch")
    feats = pd.read_parquet(p["snapshot"] / "features.parquet")
    labels = pd.read_parquet(p["snapshot"] / "labels.parquet")[["change_id", "label"]]
    frame = sp[sp["split"].isin(["train", "valid"])].merge(feats, on="change_id", how="left", validate="1:1")
    lineage = {"visibility": snap["visibility"], "snapshot_id": cfg.dataset.snapshot_id, "split_id": cfg.dataset.split_id,
               "snapshot_manifest_sha256": sha256_file(snap_m), "split_manifest_sha256": sha256_file(split_m),
               "feature_schema_id": snap["feature_schema"]["feature_schema_id"]}
    if variant in NEEDS_EVIDENCE:
        em = p["evidence"] / "manifest.json"
        if not em.exists():
            raise IntegrityError(f"missing evidence {em}; run evidence build first")
        if sha256_file(p["evidence"] / "evidence.parquet") != read_json(em)["evidence_parquet_sha256"]:
            raise IntegrityError("evidence hash mismatch")
        ev_cols = ["change_id", "query_text", "message_text", "code_text", "query_content_hash"]
        ev = pd.read_parquet(p["evidence"] / "evidence.parquet")[ev_cols]
        frame = frame.merge(ev, on="change_id", how="left", validate="1:1")
        if frame["query_content_hash"].isna().any():
            raise IntegrityError("evidence missing for some changes")
        lineage["evidence_manifest_sha256"] = sha256_file(em)
        lineage["evidence_query_hash_digest"] = read_json(em)["query_hash_digest"]
    y = frame[["change_id"]].merge(labels, on="change_id", how="left", validate="1:1")["label"]
    if y.isna().any():
        raise IntegrityError("unlabelled rows in train/valid")
    return frame, y.astype(int), lineage


def run_variant(cfg: StudyConfig, raw: dict, study_hash: str, variant: str, seed: int,
                data_dir: Path, artifacts_dir: Path) -> dict:
    registry = Registry()
    v = registry.resolve(variant)
    require_pinned(raw, variant)
    if variant not in MODELS and variant not in NEURAL:
        raise ExecutionError(f"{variant} is not implemented in this milestone")
    frame, y, lineage = load_frames(cfg, data_dir, variant)
    git = git_state()
    run_id = sha256_json({"study": study_hash, "variant": variant, "seed": seed, "lineage": lineage,
                          "code": git["sha"], "dirty": git["dirty"]})[:16]
    run_dir = artifacts_dir / "runs" / run_id
    if (run_dir / "run.json").exists() and read_json(run_dir / "run.json")["status"] == "completed":
        return read_json(run_dir / "run.json")
    attempt = 1 + len(list(run_dir.glob("attempt-*"))) if run_dir.exists() else 1
    base = {"run_id": run_id, "attempt": attempt, "study_id": cfg.study_id, "matrix_version": MATRIX_VERSION,
            "variant_id": variant, "family": v.family, "adaptation_mode": v.adaptation_mode, "track": cfg.track,
            "protocol_version": cfg.protocol_version, "evidence_profile": cfg.evidence_profile,
            "information_profile": v.input_profile, "seed": seed, "split_evaluated": "valid",
            "registry_sha256": registry.digest, "started_at": datetime.now(UTC).isoformat()}
    atomic_write_json(run_dir / "run.json", {**base, "status": "running"})
    atomic_write_json(run_dir / "resolved-config.json", raw)
    atomic_write_json(run_dir / "environment.json", environment())
    atomic_write_json(run_dir / "data-lineage.json", lineage)
    try:
        tr, va = frame["split"] == "train", frame["split"] == "valid"
        train_view = TrainingDatasetView(split="train", frame=frame[tr].reset_index(drop=True), lineage=lineage,
                                         labels=y[tr].reset_index(drop=True))
        query = QueryView(split="valid", frame=frame[va].reset_index(drop=True), lineage=lineage)
        device = "cpu"
        if variant in NEURAL:
            from .evidence import load_tokenizer
            from .neural import NeuralRun
            local = resolve_local_path(cfg.model.local_path, PROJECT_ROOT)
            tok, _ = load_tokenizer(local, cfg.model.revision)
            t0 = time.perf_counter()
            state = NeuralRun(cfg, variant, seed, local, run_dir, tok).fit_predict(
                train_view, query, y[va].to_numpy(), cfg.evaluation.tie_salt)
            scores = state.pop("scores")
            fit_s, pred_s, device = time.perf_counter() - t0, 0.0, state["device"]
        else:
            model = MODELS[variant](cfg, seed)
            t0 = time.perf_counter()
            state = model.fit(train_view)
            fit_s = time.perf_counter() - t0
            t1 = time.perf_counter()
            scores = model.predict(query)
            pred_s = time.perf_counter() - t1
        ids = query.ids
        pred = pd.DataFrame({"change_id": ids, "run_id": run_id, "score": scores.astype(float),
                             "score_semantics": SCORE_SEMANTICS, "prediction_status": "ok",
                             "latency_ms": pred_s * 1000 / max(len(ids), 1)})
        if len(pred) != int(va.sum()) or not pred["score"].between(0, 1).all():
            raise IntegrityError("prediction coverage/score range check failed")
        (run_dir / "predictions").mkdir(parents=True, exist_ok=True)
        pred.to_parquet(run_dir / "predictions" / "validation.parquet", index=False)
        booster = state.pop("booster_text", None)
        if booster is not None:
            (run_dir / "model").mkdir(exist_ok=True)
            (run_dir / "model" / "lightgbm.txt").write_text(booster)
        atomic_write_json(run_dir / "model" / "state.json", state)
        yv = y[va].to_numpy()
        thr = best_threshold(yv, scores)
        metrics = {"split": "valid", **evaluate(ids, yv, scores, cfg.evaluation.tie_salt, thr["threshold"]),
                   "threshold_selection": thr, "note": "threshold chosen on this same validation set (F1 is optimistic)"}
        atomic_write_json(run_dir / "metrics.json", metrics)
        atomic_write_json(run_dir / "label-access-ledger.json", {
            "gradient_label_count": int(tr.sum()), "demo_unique_label_count": 0, "index_labeled_count": 0,
            "selection_label_count": int(va.sum()), "rmi_synthetic_target_count": 0, "adaptation_mode": v.adaptation_mode,
            "selection_use": "threshold (max F1) and reporting on public validation"})
        atomic_write_json(run_dir / "evidence-manifest.json", {
            "evidence_profile": cfg.evidence_profile, "structured_feature_profile": cfg.dataset.feature_profile,
            "structured_columns": list(cfg.structured.features),
            "evidence_manifest_sha256": lineage.get("evidence_manifest_sha256"),
            "query_hash_digest": lineage.get("evidence_query_hash_digest")})
        atomic_write_json(run_dir / "cost.json", {"wall_seconds": fit_s + pred_s, "fit_seconds": fit_s, "predict_seconds": pred_s,
                                                  "device": device, "gpu_seconds": fit_s if device.startswith("cuda") else 0,
                                                  "peak_cuda_bytes": state.get("peak_cuda_bytes"),
                                                  "note": "neural fit_seconds include per-epoch validation scoring"})
        done = {**base, "status": "completed", "finished_at": datetime.now(UTC).isoformat(),
                "validation_ap": metrics["ap"]}
        atomic_write_json(run_dir / "run.json", done)
        return done
    except Exception as e:
        atomic_write_json(run_dir / f"attempt-{attempt}" / "failure.json",
                          {"error": type(e).__name__, "message": str(e), "traceback": traceback.format_exc()})
        atomic_write_json(run_dir / "run.json", {**base, "status": "failed", "error": f"{type(e).__name__}: {e}"})
        raise
