"""Evaluation-only offline prediction from an export bundle (spec FR-14/17/22, T24, DoD-Internal-Ready, AT-19/20/21/25/35).

The predictor never sees labels: an input with a label-like column is refused, and no command
fits, calibrates, re-normalises scores, or adds internal changes to demos/indices. Network
egress is blocked for the whole call (HF offline env + socket guard). Every bundle file is
re-hashed against the export manifest before use; base models are referenced by pinned hashes
and must already be on local disk. Internal changes are rendered with the same EvidenceView
rule as the public data, with add/delete lines canonicalised by lexicographic sort (the
public snapshot stores lines as unordered sets). A missing structured column is never filled
with zeros: every variant needs the jit14 profile, so such a dataset is reported unavailable.
"""
from __future__ import annotations

import contextlib
import os
import socket
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

from .config import FEATURE_PROFILE_JIT14, StudyConfig
from .policy import QueryView
from .util import ConfigError, IntegrityError, PolicyError, atomic_write_json, read_json, sha256_file

REQUIRED_COLUMNS = ("change_id", "message", "added_lines", "deleted_lines")
LABEL_LIKE = ("label", "labels", "bug", "buggy", "defect", "is_bug", "target", "y")
OFFLINE_ENV = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1"}
INTERNAL_SEMANTICS = "uncalibrated public-defect score (not an internal incident probability; no re-normalisation)"


@contextlib.contextmanager
def offline_guard():  # noqa: ANN201
    """Block outbound connections and force Hugging Face offline mode for the duration (AT-21)."""
    saved_env = {k: os.environ.get(k) for k in OFFLINE_ENV}
    saved = (socket.socket.connect, socket.socket.connect_ex, socket.create_connection)

    def refuse(*_a, **_k):  # noqa: ANN002, ANN003, ANN202
        raise PolicyError("network egress is blocked during offline evaluation")

    os.environ.update(OFFLINE_ENV)
    socket.socket.connect, socket.socket.connect_ex, socket.create_connection = refuse, refuse, refuse  # type: ignore[method-assign]
    try:
        yield
    finally:
        socket.socket.connect, socket.socket.connect_ex, socket.create_connection = saved  # type: ignore[method-assign]
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def unpack_bundle(bundle: Path, out_dir: Path) -> dict:
    if out_dir.exists():
        raise PolicyError(f"{out_dir} exists; unpack into a fresh directory")
    with tarfile.open(bundle) as tar:
        for m in tar.getmembers():
            p = Path(m.name)
            if p.is_absolute() or ".." in p.parts or m.issym() or m.islnk() or not (m.isfile() or m.isdir()):
                raise PolicyError(f"unsafe bundle member {m.name!r}")
        tar.extractall(out_dir, filter="data")
    return verify_bundle(out_dir)


def verify_bundle(bundle_dir: Path) -> dict:
    """Re-hash every bundled file against the export manifest (AT-20)."""
    man = read_json(bundle_dir / "export-manifest.json")
    for rel, digest in man["files_sha256"].items():
        f = bundle_dir / rel
        if not f.is_file() or sha256_file(f) != digest:
            raise IntegrityError(f"bundle file {rel} missing or modified")
    fz = read_json(bundle_dir / "freeze.json")
    if fz.get("status") != "frozen" or fz.get("freeze_id") != man["freeze_id"]:
        raise IntegrityError("bundle freeze record does not match the export manifest")
    return {"manifest": man, "freeze": fz}


def study_from_bundle(bundle_dir: Path) -> StudyConfig:
    from pydantic import ValidationError
    try:
        return StudyConfig.model_validate(read_json(bundle_dir / "study.json"))
    except ValidationError as e:
        raise ConfigError(f"bundle study config invalid: {e}") from e


def load_internal(path: Path) -> tuple[pd.DataFrame, dict]:
    df = pd.read_parquet(path)
    leaked = [c for c in df.columns if c.lower() in LABEL_LIKE]
    if leaked:
        raise PolicyError(f"offline prediction input must not carry labels; found {leaked}")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ConfigError(f"internal dataset missing required columns {missing}")
    if not df["change_id"].is_unique:
        raise IntegrityError("duplicate change_id in internal dataset")
    absent = [c for c in FEATURE_PROFILE_JIT14 if c not in df.columns]
    present = [c for c in FEATURE_PROFILE_JIT14 if c in df.columns]
    nan_counts = {c: int(df[c].isna().sum()) for c in present}
    return df, {"rows": int(len(df)), "absent_feature_columns": absent, "feature_nan_counts": nan_counts,
                "input_sha256": sha256_file(path)}


def render_internal(tok, df: pd.DataFrame, cfg: StudyConfig) -> pd.DataFrame:  # noqa: ANN001
    from .evidence import render_changes
    grouped = {}
    for r in df.itertuples(index=False):
        for op, lines in (("add", r.added_lines), ("delete", r.deleted_lines)):
            lines = sorted(str(x) for x in (lines if lines is not None else []))
            if lines:
                grouped[(r.change_id, op)] = lines
    rows = render_changes(tok, df[["change_id", "message"]].astype({"message": str}), grouped, cfg.model.max_length,
                          cfg.model.max_message_tokens)
    ev = pd.DataFrame(rows)[["change_id", "query_text", "message_text", "code_text", "query_content_hash", "encoder_tokens", "truncated"]]
    return df.drop(columns=["added_lines", "deleted_lines"]).merge(ev, on="change_id", how="left", validate="1:1")


def predict(bundle_dir: Path, dataset: Path, out_dir: Path, device: str, variants: tuple[str, ...] | None,
            encoder_path: Path | None, llm_path: Path | None) -> dict:
    """Score an internal dataset with frozen bundle predictors only. Writes predictions + a label-free manifest."""
    from .runner import LLM, NEURAL
    with offline_guard():
        info = verify_bundle(bundle_dir)
        cfg = study_from_bundle(bundle_dir)
        df, coverage = load_internal(dataset)
        entries = [e for e in info["freeze"]["runs"] if variants is None or e["variant_id"] in variants]
        status: dict[str, str] = {}
        frame = None
        if coverage["absent_feature_columns"]:
            for e in entries:
                status[e["run_id"]] = f"unavailable: absent feature columns {coverage['absent_feature_columns']} (no zero fill)"
            entries = []
        elif any(e["variant_id"] != "B0-LR" and e["variant_id"] != "B0-LGBM" for e in entries):
            if encoder_path is None:
                raise ConfigError("text variants need the pinned CodeBERT tokenizer/encoder on local disk (--encoder-path)")
            from .evidence import load_tokenizer
            tok, tinfo = load_tokenizer(encoder_path, cfg.model.revision)
            frame = render_internal(tok, df, cfg)
            coverage["rendered_truncated"] = int(frame["truncated"].sum())
        if frame is None:
            frame = df.drop(columns=["added_lines", "deleted_lines"])
        view = QueryView(split="internal", frame=frame.reset_index(drop=True), lineage={"visibility": "internal"})
        out_dir.mkdir(parents=True, exist_ok=True)
        preds = []
        for e in entries:
            v, s, d = e["variant_id"], e["seed"], bundle_dir / "runs" / e["run_id"]
            try:
                scores = _bundle_predictor(cfg, v, s, d, bundle_dir, device, encoder_path, llm_path, NEURAL, LLM).predict(
                    *((view, out_dir / "llm" / e["run_id"]) if v in LLM else (view,)))
            except FileNotFoundError as err:
                status[e["run_id"]] = f"unavailable: {err}"
                continue
            if not np.isfinite(scores).all():
                raise IntegrityError(f"non-finite scores from {v}")
            status[e["run_id"]] = "ok"
            preds.append(pd.DataFrame({"change_id": view.ids, "variant_id": v, "seed": s, "run_id": e["run_id"],
                                       "score": scores, "score_semantics": INTERNAL_SEMANTICS}))
        if preds:
            pd.concat(preds, ignore_index=True).to_parquet(out_dir / "predictions.parquet", index=False)
        manifest = {"freeze_id": info["freeze"]["freeze_id"], "study_id": cfg.study_id,
                    "bundle_files": len(info["manifest"]["files_sha256"]),
                    "dataset": {k: v for k, v in coverage.items()}, "run_status": status, "labels_read": False,
                    "network": "blocked (socket guard + HF offline)", "fit_or_calibration": "none",
                    "score_semantics": INTERNAL_SEMANTICS}
        atomic_write_json(out_dir / "offline-manifest.json", manifest)
    return {"freeze_id": manifest["freeze_id"], "rows": coverage["rows"],
            "ok": sum(1 for x in status.values() if x == "ok"), "unavailable": sum(1 for x in status.values() if x != "ok")}


def _bundle_predictor(cfg: StudyConfig, v: str, s: int, run_dir: Path, bundle_dir: Path, device: str, encoder_path: Path | None,
                      llm_path: Path | None, neural: set, llm: set):  # noqa: ANN202
    from .frozen import FrozenB0LGBM, FrozenB0LR, FrozenB1, FrozenEncoder, FrozenLlm
    if v == "B0-LR":
        return FrozenB0LR(run_dir)
    if v == "B0-LGBM":
        return FrozenB0LGBM(run_dir, cfg)
    if v == "B1-TFIDF-S":
        return FrozenB1(run_dir, cfg)
    if v in neural:
        from .evidence import load_tokenizer
        return FrozenEncoder(run_dir, cfg, v, encoder_path, load_tokenizer(encoder_path, cfg.model.revision)[0], device)
    if v in llm:
        if llm_path is None or not llm_path.exists():
            raise FileNotFoundError("pinned LLM files not present locally (--llm-path)")
        demos = None
        if v == "L1-S":
            demos = pd.read_parquet(bundle_dir / "demos" / f"{run_dir.name}.parquet")
        return FrozenLlm(run_dir, cfg, v, s, llm_path, train=None, demos=demos)
    raise ConfigError(f"no offline predictor for {v}")
