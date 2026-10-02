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
bundle_app = typer.Typer(no_args_is_help=True)
internal_app = typer.Typer(no_args_is_help=True)
diffllm_app = typer.Typer(no_args_is_help=True)
app.add_typer(data_app, name="data")
app.add_typer(evidence_app, name="evidence")
app.add_typer(experiment_app, name="experiment")
app.add_typer(study_app, name="study")
app.add_typer(bundle_app, name="bundle")
app.add_typer(internal_app, name="internal")
app.add_typer(diffllm_app, name="diffllm")

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


@data_app.command("rebuild-git")
@_guard
def data_rebuild_git(snapshot_id: str = typer.Option(...), mirrors: Path = typer.Option(...), out_snapshot_id: str = typer.Option(...),
                     data_dir: Path = DATA) -> None:
    """Derived snapshot: package IDs/labels/split, message + lines + jit14 re-extracted from the public git mirrors."""
    from ..adapters.jit_defects4j_git import rebuild_from_git
    _emit(rebuild_from_git(data_dir / "snapshots" / snapshot_id, mirrors, data_dir / "snapshots" / out_snapshot_id))


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
                 artifacts_dir: Path = ARTIFACTS, data_dir: Path = DATA, variants: str = typer.Option(""),
                 seeds: str = typer.Option("")) -> None:
    """Inference-only export bundle of the frozen study (no optimizer state, CPT heads or training data)."""
    from ..study import export_bundle
    cfg, raw, _ = load_study(study)
    sel = tuple(v.strip() for v in variants.split(",") if v.strip()) or None
    sds = tuple(int(x) for x in seeds.split(",") if x.strip()) or None
    _emit(export_bundle(cfg, raw, artifacts_dir, freeze, out, data_dir, sel, sds))


@bundle_app.command("unpack")
@_guard
def bundle_unpack(bundle: Path = typer.Option(...), out: Path = typer.Option(...)) -> None:
    """Safely extract an export bundle and verify every file against its manifest."""
    from ..offline import unpack_bundle
    info = unpack_bundle(bundle, out)
    _emit({"freeze_id": info["freeze"]["freeze_id"], "files": len(info["manifest"]["files_sha256"]), "verified": True})


@bundle_app.command("verify")
@_guard
def bundle_verify(bundle_dir: Path = typer.Option(...)) -> None:
    """Re-hash an unpacked bundle against its export manifest (AT-20)."""
    from ..offline import verify_bundle
    info = verify_bundle(bundle_dir)
    _emit({"freeze_id": info["freeze"]["freeze_id"], "files": len(info["manifest"]["files_sha256"]), "verified": True})


@internal_app.command("evaluate")
@_guard
def internal_evaluate(protocol: Path = typer.Option(...), predictions: Path = typer.Option(...), mapping: Path = typer.Option(...),
                      deployments: Path = typer.Option(...), out: Path = typer.Option(...)) -> None:
    """Deployment-level evaluation of label-free offline predictions under a pre-registered protocol (prints counts only)."""
    from ..internal_eval import evaluate_internal
    _emit(evaluate_internal(protocol, predictions, mapping, deployments, out))


@internal_app.command("extract")
@_guard
def internal_extract(repo: list[Path] = typer.Option(...), project: list[str] = typer.Option(None), out: Path = typer.Option(...),
                     rev: str = typer.Option("HEAD"), since: str = typer.Option(None), until: str = typer.Option(None),
                     max_commits: int = typer.Option(None), extensions: str = typer.Option(""),
                     author_key: str = typer.Option("name")) -> None:
    """Label-free dataset (predict input) + metadata from local git history, in the public package's representation."""
    from ..gitextract import DEFAULT_EXTENSIONS, extract
    names = project or [r.resolve().name.removesuffix(".git") for r in repo]
    if len(names) != len(repo):
        raise typer.BadParameter("give one --project per --repo, or none")
    if author_key not in ("name", "email"):
        raise typer.BadParameter("--author-key must be name or email")
    exts = tuple(e.strip() if e.strip().startswith(".") else f".{e.strip()}" for e in extensions.split(",") if e.strip())
    _emit(extract(list(zip(repo, names, strict=True)), out, rev, since, until, max_commits, exts or DEFAULT_EXTENSIONS, author_key))


@internal_app.command("label-sheet")
@_guard
def internal_label_sheet(extract_dir: Path = typer.Option(...), out: Path = typer.Option(...), sample: int = typer.Option(None),
                         salt: str = typer.Option("label-sheet-v1")) -> None:
    """Score-blind labeling CSV (empty label column); --sample N picks a salted-hash random subset."""
    from ..label_eval import label_sheet
    _emit(label_sheet(extract_dir, out, sample, salt))


@internal_app.command("label-eval")
@_guard
def internal_label_eval(predictions: Path = typer.Option(...), labels: Path = typer.Option(...), meta: Path = typer.Option(...),
                        bundle_dir: Path = typer.Option(...), out_dir: Path = typer.Option(...), bootstrap: int = typer.Option(2000),
                        min_group: int = typer.Option(30)) -> None:
    """Commit-level AP / ROC-AUC / Recall@q / F1@frozen threshold of offline predictions against hand labels."""
    from ..label_eval import evaluate_labels
    _emit(evaluate_labels(predictions, labels, meta, bundle_dir, out_dir, n_boot=bootstrap, min_group=min_group))


@diffllm_app.command("prepare")
@_guard
def diffllm_prepare(config: Path = typer.Option(...), fulldiff: Path = typer.Option(...), tokenizer: Path = typer.Option(...),
                    out: Path = typer.Option(...), data_dir: Path = DATA) -> None:
    """Render full-diff prompts (message + whole hunks within budget); test rows stay label-free."""
    from ..diffllm import load_cfg, prepare
    cfg, _ = load_cfg(config)
    d = cfg["dataset"]
    m = prepare(cfg, data_dir / "snapshots" / d["snapshot_id"], data_dir / "splits" / d["snapshot_id"] / d["split_id"], fulldiff,
                tokenizer, out)
    _emit({k: m[k] for k in ("rows", "by_split", "truncated_share", "content_tokens_mean")})


@diffllm_app.command("prepare-evidence")
@_guard
def diffllm_prepare_evidence(config: Path = typer.Option(...), evidence: Path = typer.Option(...), tokenizer: Path = typer.Option(...),
                             out: Path = typer.Option(...), data_dir: Path = DATA) -> None:
    """Render the registered EvidenceView prompts (v2); test rows stay label-free."""
    from ..diffllm import load_cfg, prepare_evidence
    cfg, _ = load_cfg(config)
    d = cfg["dataset"]
    m = prepare_evidence(cfg, data_dir / "snapshots" / d["snapshot_id"], data_dir / "splits" / d["snapshot_id"] / d["split_id"],
                         evidence, tokenizer, out)
    _emit({k: m[k] for k in ("rows", "by_split", "prompt_tokens")})


@diffllm_app.command("cpt")
@_guard
def diffllm_cpt(config: Path = typer.Option(...), name: str = typer.Option(None), data_dir: Path = DATA,
                artifacts_dir: Path = ARTIFACTS) -> None:
    """LoRA diff continued pretraining on the registered disjoint-repository corpus."""
    from ..diffllm import load_cfg, run_cpt
    cfg, h = load_cfg(config)
    _emit(run_cpt(cfg, h, data_dir, artifacts_dir / "diffllm", name))


@diffllm_app.command("arm")
@_guard
def diffllm_arm(config: Path = typer.Option(...), arms: str = typer.Option(...), view: Path = typer.Option(...),
                seeds: str = typer.Option(""), artifacts_dir: Path = ARTIFACTS) -> None:
    """Train/select one or more registered arms on public train/valid (per seed when the study sets per_seed)."""
    from ..diffllm import _seeds, load_cfg, run_arm
    cfg, h = load_cfg(config)
    sds = [int(x) for x in seeds.split(",") if x.strip()] or _seeds(cfg)
    _emit([{k: r[k] for k in ("arm", "seed", "validation_ap")}
           for r in (run_arm(cfg, h, a.strip(), view, artifacts_dir / "diffllm", s) for a in arms.split(",") for s in sds)])


@diffllm_app.command("probe-cpt")
@_guard
def diffllm_probe_cpt(config: Path = typer.Option(...), view: Path = typer.Option(...), cpt: str = typer.Option(...),
                      out: Path = typer.Option(...), rows: int = typer.Option(512), artifacts_dir: Path = ARTIFACTS) -> None:
    """Label-free check of what diff CPT changed: next-token loss on public-train diffs, base vs base+CPT adapter."""
    from ..diffllm import load_cfg, probe_cpt_loss
    cfg, _ = load_cfg(config)
    _emit(probe_cpt_loss(cfg, view, artifacts_dir / "diffllm", cpt, rows, out))


@diffllm_app.command("freeze")
@_guard
def diffllm_freeze(config: Path = typer.Option(...), view: Path = typer.Option(...), out: Path = typer.Option(...),
                   artifacts_dir: Path = ARTIFACTS) -> None:
    """Freeze all diffllm arms after reproducing a validation sample from frozen artifacts."""
    from ..diffllm import freeze, load_cfg
    cfg, h = load_cfg(config)
    rec = freeze(cfg, h, view, artifacts_dir / "diffllm", out)
    _emit({"freeze_id": rec["freeze_id"], "arms": [e["arm"] for e in rec["arms"]]})


@diffllm_app.command("test")
@_guard
def diffllm_test(config: Path = typer.Option(...), view: Path = typer.Option(...), freeze: Path = typer.Option(...),
                 out_dir: Path = typer.Option(...), data_dir: Path = DATA, artifacts_dir: Path = ARTIFACTS) -> None:
    """One-shot public test evaluation of the frozen diffllm arms (never overwrites)."""
    from ..diffllm import load_cfg, test_arms_once
    cfg, h = load_cfg(config)
    _emit(test_arms_once(cfg, h, view, data_dir / "snapshots" / cfg["dataset"]["snapshot_id"], artifacts_dir / "diffllm", freeze,
                         out_dir))


@app.command("predict")
@_guard
def offline_predict(bundle_dir: Path = typer.Option(...), dataset: Path = typer.Option(...), out_dir: Path = typer.Option(...),
                    device: str = typer.Option("cpu"), variants: str = typer.Option(""),
                    encoder_path: Path = typer.Option(None), llm_path: Path = typer.Option(None)) -> None:
    """Evaluation-only offline scoring of an internal dataset (no labels, no fitting, network blocked). Prints counts only."""
    from ..offline import predict
    sel = tuple(v.strip() for v in variants.split(",") if v.strip()) or None
    _emit(predict(bundle_dir, dataset, out_dir, device, sel, encoder_path, llm_path))


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
