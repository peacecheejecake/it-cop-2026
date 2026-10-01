"""Run execution and the run-folder contract (spec implementation-plan §4.1, FR-11/12/24).

run_id = hash(study config, variant, seed, data lineage, code SHA). A completed run is
never overwritten; a failed attempt is kept with failure.json. Predictors see label-free
QueryViews; labels are joined only here, in the evaluator step. This runner evaluates
public validation only; the public test is sealed until a study freeze exists.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
import traceback
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

import pandas as pd

from .config import SECTIONS_READ, StudyConfig, require_pinned, resolve_local_path
from .metrics import best_threshold, evaluate
from .models import MODELS, SCORE_SEMANTICS
from .policy import QueryView, TrainingDatasetView
from .registry import MATRIX_VERSION, Registry
from .util import ExecutionError, IntegrityError, atomic_write_json, read_json, sha256_file, sha256_json, tree_digest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NEEDS_EVIDENCE = {"B1-TFIDF-S", "B2-S", "B3-S", "B4-S", "B5-S", "L0-S", "L1-S"}
NEURAL = {"B2-S", "B3-S", "B4-S", "B5-S"}
LLM = {"L0-S", "L1-S"}
CPT_VARIANTS = {"B4-S": "mlm", "B5-S": "mlm+rmi"}
SOURCE_PATTERNS = ("src/**/*.py", "pyproject.toml", "uv.lock")
LLM_RUN_FILES = ("prompts/manifest.json", "usage.jsonl")
REQUIRED_RUN_FILES = ("resolved-config.json", "environment.json", "data-lineage.json", "metrics.json", "metrics.jsonl",
                      "label-access-ledger.json", "evidence-manifest.json", "cost.json", "model/state.json",
                      "predictions/validation.parquet", "reports/summary.md")


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


def environment(neural: bool) -> dict:
    pkgs = {p: metadata.version(p) for p in ("numpy", "pandas", "pyarrow", "scikit-learn", "scipy", "transformers",
                                             "tokenizers", "pydantic") if _installed(p)}
    if _installed("lightgbm"):
        pkgs["lightgbm"] = metadata.version("lightgbm")
    lock = PROJECT_ROOT / "uv.lock"
    env = {"python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine(),
           "cpu_count": os.cpu_count(), "packages": pkgs, "uv_lock_sha256": sha256_file(lock) if lock.exists() else None,
           "git": git_state()}
    if _installed("torch"):
        env["packages"]["torch"] = metadata.version("torch")
    # Importing torch loads its OpenMP runtime; with LightGBM's libomp in the same process that segfaults (exp 009).
    if neural and _installed("torch"):
        import torch
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


def load_frames(cfg: StudyConfig, data_dir: Path, variant: str, test_unlock: dict | None = None) -> tuple[pd.DataFrame, pd.Series, dict]:
    """train+valid frames; test rows are added only with an unlocked study-freeze record (spec AT-17)."""
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
    splits = ["train", "valid"] + (["test"] if test_unlock is not None and test_unlock.get("status") == "frozen" else [])
    frame = sp[sp["split"].isin(splits)].merge(feats, on="change_id", how="left", validate="1:1")
    from .lineage import validate_lineage
    tok_digest = None
    if variant in NEEDS_EVIDENCE:
        from .evidence import tokenizer_files_digest
        tok_digest = tokenizer_files_digest(resolve_local_path(cfg.model.local_path, PROJECT_ROOT))
    roles = ("supervised_train", "selection") + (("cpt_train",) if variant in CPT_VARIANTS else ())
    approved = validate_lineage(resolve_local_path(cfg.dataset.source_approval, PROJECT_ROOT), p["snapshot"], p["split"],
                                p["evidence"] if variant in NEEDS_EVIDENCE else None, tok_digest, roles)
    lineage = {"visibility": snap["visibility"], "snapshot_id": cfg.dataset.snapshot_id, "split_id": cfg.dataset.split_id,
               **approved,
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
        raise IntegrityError("unlabelled rows in the loaded splits")
    return frame, y.astype(int), lineage


IDENTITY_KEYS = ("schema_version", "matrix_version", "study_id", "track", "protocol_version", "evidence_profile",
                 "information_profile", "evaluation", "policy")


def scoped_config_hash(raw: dict, variant: str) -> str:
    """Hash of the study identity plus only the sections this variant reads, so pinning a later section keeps earlier run ids."""
    return sha256_json({k: raw.get(k) for k in (*IDENTITY_KEYS, *SECTIONS_READ[variant])})


def _run_cpt(cfg: StudyConfig, seed: int, frame: pd.DataFrame, lineage: dict, git: dict, local: Path, tok,  # noqa: ANN001
             artifacts_dir: Path, task: str) -> dict:
    from .cpt import CPT_COLUMNS, CptCorpusView, CptRun, cpt_id_for
    cols = list(CPT_COLUMNS)
    tr = frame[frame["cpt_role"] == "cpt_train"][cols].reset_index(drop=True)
    dv = frame[frame["cpt_role"] == "cpt_dev"][cols].reset_index(drop=True)
    cid = cpt_id_for(cfg, seed, lineage, git, task)
    out = artifacts_dir / "cpt" / cid
    res = CptRun(cfg, seed, local, out, tok, task).run(CptCorpusView("cpt_train", tr, lineage), CptCorpusView("cpt_dev", dv, lineage))
    return {"cpt_id": cid, **res}


def run_variant(cfg: StudyConfig, raw: dict, study_hash: str, variant: str, seed: int,
                data_dir: Path, artifacts_dir: Path) -> dict:
    registry = Registry()
    v = registry.resolve(variant)
    require_pinned(raw, variant)
    if variant not in MODELS and variant not in NEURAL and variant not in LLM:
        raise ExecutionError(f"{variant} is not implemented in this milestone")
    if variant == "B0-LGBM" and "torch" in sys.modules:
        raise ExecutionError("B0-LGBM must run in a process that has not imported torch (two OpenMP runtimes segfault); "
                             "run it in a separate `experiment run` invocation from the neural variants")
    frame, y, lineage = load_frames(cfg, data_dir, variant)
    git = git_state()
    if git["sha"] is None:
        raise ExecutionError("unknown code version (no git and no CODE_SHA marker); refusing a run that cannot be traced")
    source = tree_digest(PROJECT_ROOT, SOURCE_PATTERNS)
    run_id = sha256_json({"study": scoped_config_hash(raw, variant), "variant": variant, "seed": seed, "lineage": lineage,
                          "code": git["sha"], "source_digest": source})[:16]
    run_dir = artifacts_dir / "runs" / run_id
    if (run_dir / "run.json").exists() and read_json(run_dir / "run.json")["status"] == "completed":
        done = read_json(run_dir / "run.json")
        for rel, digest in done.get("artifacts_sha256", {}).items():
            if not (run_dir / rel).is_file() or sha256_file(run_dir / rel) != digest:
                raise IntegrityError(f"completed run {run_id} has a missing or modified artifact {rel}; not reusing it")
        if not done.get("artifacts_sha256"):
            raise IntegrityError(f"completed run {run_id} lacks an artifact manifest; not reusing it")
        return done
    attempt = 1 + len(list(run_dir.glob("attempt-*"))) if run_dir.exists() else 1
    base = {"run_id": run_id, "attempt": attempt, "study_id": cfg.study_id, "matrix_version": MATRIX_VERSION,
            "variant_id": variant, "family": v.family, "adaptation_mode": v.adaptation_mode, "track": cfg.track,
            "protocol_version": cfg.protocol_version, "evidence_profile": cfg.evidence_profile,
            "information_profile": v.input_profile, "seed": seed, "split_evaluated": "valid",
            "registry_sha256": registry.digest, "study_sha256": study_hash, "source_digest": source, "code_git": git,
            "scoped_config_sha256": scoped_config_hash(raw, variant), "started_at": datetime.now(UTC).isoformat()}
    atomic_write_json(run_dir / "run.json", {**base, "status": "running"})
    atomic_write_json(run_dir / "resolved-config.json", raw)
    atomic_write_json(run_dir / "environment.json", environment(variant in NEURAL or variant in LLM))
    atomic_write_json(run_dir / "data-lineage.json", lineage)
    try:
        tr, va = frame["split"] == "train", frame["split"] == "valid"
        train_view = TrainingDatasetView(split="train", frame=frame[tr].reset_index(drop=True), lineage=lineage,
                                         labels=y[tr].reset_index(drop=True))
        query = QueryView(split="valid", frame=frame[va].reset_index(drop=True), lineage=lineage)
        device = "cpu"
        cpt = None
        semantics = SCORE_SEMANTICS
        if variant in LLM:
            from .llm import SCORE_SEMANTICS as LLM_SEMANTICS
            from .llm import LlmRun
            t0 = time.perf_counter()
            state = LlmRun(cfg, variant, seed, resolve_local_path(cfg.llm.local_path, PROJECT_ROOT), run_dir).fit_predict(
                train_view, query)
            scores = state.pop("scores")
            pred_s, device = state["inference_seconds"], state["device"]
            fit_s = time.perf_counter() - t0 - pred_s  # prompt building and model load; scoring time is pred_s
            semantics = LLM_SEMANTICS
        elif variant in NEURAL:
            from .evidence import load_tokenizer
            from .neural import NeuralRun
            local = resolve_local_path(cfg.model.local_path, PROJECT_ROOT)
            tok, _ = load_tokenizer(local, cfg.model.revision)
            t0 = time.perf_counter()
            init = None
            if variant in CPT_VARIANTS:
                cpt = _run_cpt(cfg, seed, frame, lineage, {"sha": git["sha"], "source_digest": source}, local, tok, artifacts_dir,
                               CPT_VARIANTS[variant])
                init = artifacts_dir / "cpt" / cpt["cpt_id"] / "encoder"
            state = NeuralRun(cfg, variant, seed, local, run_dir, tok, init_encoder=init).fit_predict(
                train_view, query, y[va].to_numpy(), cfg.evaluation.tie_salt)
            if cpt is not None:
                state["cpt"] = {k: cpt[k] for k in ("cpt_id", "task", "plan_sha256", "lm_head_newly_initialized",
                                                    "encoder_init_state_sha256",
                                                    "exported_encoder_state_sha256", "dev", "precision")}
            scores = state.pop("scores")
            # Validation scores come from per-epoch evaluation inside fit; inference latency is not measured here.
            fit_s, pred_s, device = time.perf_counter() - t0, None, state["device"]
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
                             "score_semantics": semantics, "prediction_status": "ok",
                             "latency_ms": None if pred_s is None else pred_s * 1000 / max(len(ids), 1)})
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
        history = state.get("history") or [{"epoch": None, "note": "single fit (no epochs)"}]
        metrics = {"split": "valid", "evaluation_role": "selection_validation",
                   **evaluate(ids, yv, scores, cfg.evaluation.tie_salt, thr["threshold"]), "threshold_selection": thr,
                   "note": ("selection-validation: the threshold" + (" and the checkpoint epoch" if variant in NEURAL else "")
                            + " were chosen on this same validation set, so AP/recall/F1 are optimistic; "
                            "only the sealed test after freeze is an independent estimate")}
        atomic_write_json(run_dir / "metrics.json", metrics)
        (run_dir / "metrics.jsonl").write_text("".join(json.dumps({"kind": "epoch", **h}, sort_keys=True) + "\n" for h in history))
        events = [{"kind": "threshold", "rule": thr["policy"], "chosen": thr["threshold"], "metric": "f1",
                   "split": "valid"}]
        if variant in LLM:
            events.insert(0, {"kind": "llm_configuration", "rule": "single pre-registered model/template/scorer; no sweep",
                              "candidates": 1, "split": "none"})
        if variant in NEURAL:
            events.insert(0, {"kind": "checkpoint_epoch", "rule": "max validation AP, ties -> earlier", "metric": "validation_ap",
                              "chosen_epoch": state["best_epoch"], "candidates": len(state["history"]), "split": "valid"})
        atomic_write_json(run_dir / "label-access-ledger.json", {
            "gradient_label_count": 0 if variant in LLM else int(tr.sum()),
            "demo_unique_label_count": state.get("demo_unique_label_count", 0) if variant in LLM else 0, "index_labeled_count": 0,
            "selection_label_count": int(va.sum()), "adaptation_mode": v.adaptation_mode,
            "rmi_synthetic_target_count": cpt["token_accounting"]["rmi_targets"] if cpt else 0,
            "cpt_label_count": 0, "cpt_unlabeled_input_count": cpt["token_accounting"]["corpus_changes"] if cpt else 0,
            "selection_events": events,
            "selection_use": "checkpoint epoch / threshold selection and reporting on public validation"})
        atomic_write_json(run_dir / "evidence-manifest.json", {
            "evidence_profile": cfg.evidence_profile, "structured_feature_profile": cfg.dataset.feature_profile,
            "structured_columns": list(cfg.structured.features),
            "evidence_manifest_sha256": lineage.get("evidence_manifest_sha256"),
            "query_hash_digest": lineage.get("evidence_query_hash_digest")})
        if variant in NEURAL:
            acct = {"downstream": state["token_accounting"]}
            if cpt is not None:
                acct["cpt"] = {"cpt_id": cpt["cpt_id"], **cpt["token_accounting"]}
            atomic_write_json(run_dir / "token-accounting.json", acct)
        atomic_write_json(run_dir / "cost.json", {
            "wall_seconds": fit_s + (pred_s or 0.0), "fit_seconds": fit_s, "predict_seconds": pred_s,
            "device": device, "gpu_seconds": (fit_s + (pred_s or 0.0)) if device.startswith("cuda") else 0,
            "peak_cuda_bytes": state.get("peak_cuda_bytes"), "peak_cuda_reserved_bytes": state.get("peak_cuda_reserved_bytes"),
            "inference_latency": "not measured (null); needs a separate inference benchmark" if pred_s is None else "measured",
            "note": "neural fit_seconds include per-epoch validation scoring"})
        (run_dir / "reports").mkdir(exist_ok=True)
        (run_dir / "reports" / "summary.md").write_text(
            f"# {variant} seed {seed} ({run_id})\n\nstudy `{cfg.study_id}` / split `{cfg.dataset.split_id}` / code `{git['sha']}`\n\n"
            f"selection-validation AP {metrics['ap']}, ROC-AUC {metrics['roc_auc']}, "
            f"Recall@5% {metrics['recall_at_5pct']['recall']}, Recall@10% {metrics['recall_at_10pct']['recall']}\n\n"
            + (f"best epoch {state['best_epoch']} of {len(state['history'])}\n" if variant in NEURAL else ""))
        artifacts = {rel: sha256_file(run_dir / rel) for rel in REQUIRED_RUN_FILES + (LLM_RUN_FILES if variant in LLM else ())}
        done = {**base, "status": "completed", "finished_at": datetime.now(UTC).isoformat(), "artifacts_sha256": artifacts,
                "validation_ap": metrics["ap"], **({"cpt_id": cpt["cpt_id"]} if cpt else {})}
        atomic_write_json(run_dir / "run.json", done)
        return done
    except Exception as e:
        atomic_write_json(run_dir / f"attempt-{attempt}" / "failure.json",
                          {"error": type(e).__name__, "message": str(e), "traceback": traceback.format_exc()})
        atomic_write_json(run_dir / "run.json", {**base, "status": "failed", "error": f"{type(e).__name__}: {e}"})
        raise
