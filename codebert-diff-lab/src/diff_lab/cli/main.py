"""diff-lab CLI (spec implementation-plan §5). Exit codes: 0 ok, 2 config, 3 policy, 4 integrity, 5 execution."""
# No `from __future__ import annotations`: Typer must see real types on the command signatures.
import functools
import json
import platform
import shutil
from pathlib import Path

import typer

from ..config import load_study, resolve_local_path
from ..policy import require_test_unlocked
from ..util import DiffLabError, read_json

PROJECT_ROOT = Path(__file__).resolve().parents[3]
app = typer.Typer(no_args_is_help=True, add_completion=False)
data_app = typer.Typer(no_args_is_help=True)
evidence_app = typer.Typer(no_args_is_help=True)
experiment_app = typer.Typer(no_args_is_help=True)
study_app = typer.Typer(no_args_is_help=True)
app.add_typer(data_app, name="data")
app.add_typer(evidence_app, name="evidence")
app.add_typer(experiment_app, name="experiment")
app.add_typer(study_app, name="study")

DATA = typer.Option(PROJECT_ROOT / "data", "--data-dir")
ARTIFACTS = typer.Option(PROJECT_ROOT / "artifacts", "--artifacts-dir")


def _emit(obj: object) -> None:
    typer.echo(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True, default=str))


def _guard(fn):  # noqa: ANN001, ANN202
    @functools.wraps(fn)
    def wrapper(*a, **k):  # noqa: ANN002, ANN003, ANN202
        try:
            return fn(*a, **k)
        except DiffLabError as e:
            typer.echo(f"error[{type(e).__name__}]: {e}", err=True)
            raise typer.Exit(e.exit_code) from e
    return wrapper


@app.command()
@_guard
def doctor() -> None:
    """Environment, sandbox and optional-dependency check (no network, no downloads)."""
    from importlib import metadata
    out = {"python": platform.python_version(), "platform": platform.platform(),
           "network_sandbox": "sandbox-exec" if shutil.which("sandbox-exec") else ("unshare" if shutil.which("unshare") else None)}
    try:
        import lightgbm  # noqa: F401
        out["lightgbm"] = metadata.version("lightgbm")
    except Exception as e:  # noqa: BLE001 - report any import failure verbatim
        out["lightgbm"] = f"unavailable: {type(e).__name__}: {str(e)[:120]}"
    try:
        import torch
        out["torch"] = torch.__version__
        out["cuda"] = torch.cuda.is_available()
        out["mps"] = torch.backends.mps.is_available()
    except ImportError:
        out["torch"] = "not installed (extra 'neural')"
    _emit(out)


@data_app.command("import")
@_guard
def data_import(source: str = typer.Option(...), archive: Path = typer.Option(...), approval: Path = typer.Option(...),
                snapshot_id: str = typer.Option(...), data_dir: Path = DATA) -> None:
    """Import an approved archive into an immutable snapshot (legacy pickles via isolated worker)."""
    if source != "jit-defects4j":
        typer.echo(f"unknown source {source}", err=True)
        raise typer.Exit(2)
    from ..adapters.jit_defects4j import import_archive
    m = import_archive(archive, approval, data_dir / "snapshots" / snapshot_id)
    _emit({"snapshot": snapshot_id, "split_stats": m["split_stats"], "isolation": m["import_isolation"]})


@data_app.command("audit")
@_guard
def data_audit(snapshot_id: str = typer.Option(...), data_dir: Path = DATA) -> None:
    """Label-free duplicate / label / temporal / feature audit."""
    from ..data.audit import audit
    _emit(audit(data_dir / "snapshots" / snapshot_id, data_dir / "audit" / snapshot_id))


@data_app.command("split")
@_guard
def data_split(snapshot_id: str = typer.Option(...), split_id: str = typer.Option("upstream-clean1"), data_dir: Path = DATA) -> None:
    """Build the controlled split (eval membership kept, contaminated train removed, CPT-dev by ID hash)."""
    from ..data.splits import SPLIT_IDS, make_controlled_split
    if split_id not in SPLIT_IDS:
        typer.echo(f"split must be one of {SPLIT_IDS}", err=True)
        raise typer.Exit(2)
    _emit(make_controlled_split(data_dir / "snapshots" / snapshot_id, data_dir / "splits" / snapshot_id / split_id, split_id))


@evidence_app.command("build")
@_guard
def evidence_build(study: Path = typer.Option(...), data_dir: Path = DATA) -> None:
    """Build the matched EvidenceView with the pinned local tokenizer (CPU)."""
    from ..evidence import build_evidence
    from ..runner import paths
    cfg, _, _ = load_study(study)
    p = paths(cfg, data_dir)
    m = cfg.model
    local = resolve_local_path(m.local_path, PROJECT_ROOT)
    out = build_evidence(p["snapshot"], p["split"], str(local), m.revision, m.max_length, m.max_message_tokens, p["evidence"])
    _emit({"evidence_dir": str(p["evidence"]), "coverage": out["coverage"], "cache_key": out["cache_key"]})


@experiment_app.command("run")
@_guard
def experiment_run(study: Path = typer.Option(...), models: str = typer.Option(...), seeds: str = typer.Option("42"),
                   split: str = typer.Option("valid"), data_dir: Path = DATA, artifacts_dir: Path = ARTIFACTS) -> None:
    """Train on public train and evaluate on public validation. Public test is sealed until freeze."""
    from ..runner import run_variant
    require_test_unlocked(split, None)
    if split != "valid":
        typer.echo("experiment run evaluates public validation only", err=True)
        raise typer.Exit(2)
    cfg, raw, h = load_study(study)
    results = []
    for vid in [m.strip() for m in models.split(",") if m.strip()]:
        if vid not in cfg.experiments:
            typer.echo(f"{vid} is not registered in study {cfg.study_id}", err=True)
            raise typer.Exit(2)
        for seed in [int(s) for s in seeds.split(",")]:
            r = run_variant(cfg, raw, h, vid, seed, data_dir, artifacts_dir)
            results.append({k: r.get(k) for k in ("run_id", "variant_id", "seed", "status", "validation_ap")})
    _emit(results)


@experiment_app.command("profile")
@_guard
def experiment_profile(study: Path = typer.Option(...), precision: str = typer.Option(...), updates: int = typer.Option(100),
                       seed: int = typer.Option(42), data_dir: Path = DATA, out: Path = typer.Option(...)) -> None:
    """Spec T14: N-update B3-style profile (throughput, peak VRAM, loss trajectory) for one precision. Train split only."""
    from ..evidence import load_tokenizer
    from ..neural import NeuralRun
    from ..policy import TrainingDatasetView
    from ..runner import load_frames
    from ..util import atomic_write_json
    cfg, raw, _ = load_study(study)
    if precision not in ("fp32", "bf16_encoder_autocast"):
        typer.echo("precision must be fp32 or bf16_encoder_autocast", err=True)
        raise typer.Exit(2)
    cfg = cfg.model_copy(update={"finetune": cfg.finetune.model_copy(update={"precision": precision})})
    frame, y, lineage = load_frames(cfg, data_dir, "B3-S")
    tr = frame["split"] == "train"
    view = TrainingDatasetView(split="train", frame=frame[tr].reset_index(drop=True), lineage=lineage,
                               labels=y[tr].reset_index(drop=True))
    local = resolve_local_path(cfg.model.local_path, PROJECT_ROOT)
    tok, _ = load_tokenizer(local, cfg.model.revision)
    result = NeuralRun(cfg, "B3-S", seed, local, out.parent, tok).profile(view, updates)
    atomic_write_json(out, {**result, "seed": seed, "lineage": lineage})
    _emit({k: v for k, v in result.items() if k != "losses"} | {"loss_first_last": [result["losses"][0], result["losses"][-1]]})


@study_app.command("freeze")
@_guard
def study_freeze(study: Path = typer.Option(...), out: Path = typer.Option(...), device: str = typer.Option("cpu"),
                 data_dir: Path = DATA, artifacts_dir: Path = ARTIFACTS, skip: str = typer.Option("")) -> None:
    """Freeze every registered variant x replicate after re-deriving its validation predictions from frozen state."""
    from ..study import freeze
    cfg, raw, _ = load_study(study)
    skipped = tuple(v.strip() for v in skip.split(",") if v.strip())
    rec = freeze(cfg, raw, data_dir, artifacts_dir, out, device, skip_variants=skipped)
    _emit({"freeze_id": rec["freeze_id"], "runs": len(rec["runs"]), "skipped": rec["skipped_variants"], "out": str(out)})


@experiment_app.command("test")
@_guard
def experiment_test(study: Path = typer.Option(...), freeze: Path = typer.Option(...), out_dir: Path = typer.Option(...),
                    device: str = typer.Option("cpu"), variants: str = typer.Option(""), data_dir: Path = DATA,
                    artifacts_dir: Path = ARTIFACTS) -> None:
    """One-shot public test evaluation of frozen runs (never overwrites existing results)."""
    from ..study import evaluate_public_test
    cfg, raw, _ = load_study(study)
    sel = tuple(v.strip() for v in variants.split(",") if v.strip()) or None
    _emit(evaluate_public_test(cfg, raw, data_dir, artifacts_dir, freeze, out_dir, device, sel))


@study_app.command("report")
@_guard
def study_report(study: Path = typer.Option(...), freeze: Path = typer.Option(...), test_dir: Path = typer.Option(...),
                 out: Path = typer.Option(...), data_dir: Path = DATA, artifacts_dir: Path = ARTIFACTS,
                 bootstrap: int = typer.Option(2000)) -> None:
    """Validation/test table, seed-paired test AP differences and project-bootstrap intervals."""
    from ..study import report
    cfg, _, _ = load_study(study)
    r = report(cfg, data_dir, artifacts_dir, freeze, test_dir, out, n_boot=bootstrap)
    _emit({v: {"test_ap": t["test_ap"], "valid_ap": t["valid_ap"]} for v, t in r["variants"].items()})


@study_app.command("export")
@_guard
def study_export(study: Path = typer.Option(...), freeze: Path = typer.Option(...), out: Path = typer.Option(...),
                 artifacts_dir: Path = ARTIFACTS) -> None:
    """Inference-only export bundle of the frozen study (no optimizer state, CPT heads or training data)."""
    from ..study import export_bundle
    cfg, raw, _ = load_study(study)
    _emit(export_bundle(cfg, raw, artifacts_dir, freeze, out))


@study_app.command("summary")
@_guard
def study_summary(study: Path = typer.Option(...), artifacts_dir: Path = ARTIFACTS) -> None:
    """Validation summary of completed runs for this study."""
    cfg, _, _ = load_study(study)
    rows = []
    for rj in sorted((artifacts_dir / "runs").glob("*/run.json")):
        r = read_json(rj)
        if r.get("study_id") != cfg.study_id:
            continue
        row = {k: r.get(k) for k in ("run_id", "variant_id", "seed", "status")}
        mpath = rj.parent / "metrics.json"
        if r["status"] == "completed" and mpath.exists():
            m = read_json(mpath)
            row.update({"ap": m["ap"], "roc_auc": m["roc_auc"], "recall@5%": m["recall_at_5pct"]["recall"],
                        "recall@10%": m["recall_at_10pct"]["recall"], "f1@thr": m["at_threshold"]["f1"]})
        rows.append(row)
    _emit(rows)


if __name__ == "__main__":
    app()
