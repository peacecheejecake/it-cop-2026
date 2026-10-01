"""Study freeze, one-shot public test evaluation, report and export (spec protocol §9/§10, FR-13/14/15, AT-17).

freeze: every registered primary variant x required replicate must have exactly one completed
validation run of this study (same scoped config). Each run's artifact hashes and best
checkpoint are verified, its validation predictions are re-derived from the frozen state and
must match, and the validation threshold is fixed. The record is immutable.

test: requires the frozen record (all referenced files re-hashed first). Predictors are rebuilt
from frozen state only; labels are joined in the evaluator; thresholds are the frozen ones.
The public test is evaluated once: an existing result set is never overwritten.

report: validation and test metrics per variant (mean, sd over replicates), seed-paired test
AP differences for the pre-registered pairs, and a project-level bootstrap interval that
resamples projects (not rows) and is reported separately from seed variance.
"""
from __future__ import annotations

import json
import shutil
import tarfile
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import StudyConfig, resolve_local_path
from .metrics import evaluate
from .policy import QueryView, TrainingDatasetView, require_test_unlocked
from .registry import Registry
from .runner import LLM, NEURAL, PROJECT_ROOT, environment, git_state, load_frames, scoped_config_hash
from .util import ExecutionError, IntegrityError, PolicyError, atomic_write_json, read_json, sha256_file, sha256_json

REPLICATES = {"L0-S": [42]}
PAIRS = [("B3-S", "B2-S"), ("B4-S", "B3-S"), ("B5-S", "B4-S"), ("B5-S", "B3-S"), ("L1-S", "L0-S"), ("B1-TFIDF-S", "B0-LR"),
         ("B2-S", "B1-TFIDF-S"), ("B3-S", "B1-TFIDF-S"), ("B3-S", "L1-S")]
VALID_TOL = {"cpu": 1e-9, "neural": 2e-3}
LLM_TOL = {"max_abs": 0.05, "spearman": 0.99}


def _replicates(cfg: StudyConfig, variant: str) -> list[int]:
    return REPLICATES.get(variant, list(cfg.seeds))


def _verify_run(run_dir: Path) -> dict:
    run = read_json(run_dir / "run.json")
    if run.get("status") != "completed" or not run.get("artifacts_sha256"):
        raise IntegrityError(f"{run_dir.name}: not a completed run with an artifact manifest")
    for rel, digest in run["artifacts_sha256"].items():
        if sha256_file(run_dir / rel) != digest:
            raise IntegrityError(f"{run_dir.name}: artifact {rel} changed since the run completed")
    return run


def _checkpoint_files(run_dir: Path) -> dict:
    ptr = run_dir / "checkpoints" / "best.json"
    if not ptr.exists():
        return {}
    meta = read_json(ptr)
    out = {}
    for rel, digest in meta["files_sha256"].items():
        f = run_dir / "checkpoints" / meta["generation"] / rel
        if not f.is_file() or sha256_file(f) != digest:
            raise IntegrityError(f"{run_dir.name}: best checkpoint file {rel} missing or corrupted")
        out[f"checkpoints/{meta['generation']}/{rel}"] = digest
    out["checkpoints/best.json"] = sha256_file(ptr)
    return out


def _frames(cfg: StudyConfig, data_dir: Path, variant: str, unlock: dict | None = None):  # noqa: ANN202
    frame, y, lineage = load_frames(cfg, data_dir, variant, unlock)
    tr = frame["split"] == "train"
    train = TrainingDatasetView(split="train", frame=frame[tr].reset_index(drop=True), lineage=lineage,
                                labels=y[tr].reset_index(drop=True))
    return frame, y, lineage, train


def predictor(cfg: StudyConfig, variant: str, seed: int, run_dir: Path, train: TrainingDatasetView, device: str):  # noqa: ANN201
    from .frozen import FrozenB0LGBM, FrozenB0LR, FrozenB1, FrozenEncoder, FrozenLlm
    if variant == "B0-LR":
        return FrozenB0LR(run_dir)
    if variant == "B0-LGBM":
        return FrozenB0LGBM(run_dir, cfg)
    if variant == "B1-TFIDF-S":
        return FrozenB1(run_dir, cfg)
    if variant in NEURAL:
        from .evidence import load_tokenizer
        local = resolve_local_path(cfg.model.local_path, PROJECT_ROOT)
        return FrozenEncoder(run_dir, cfg, variant, local, load_tokenizer(local, cfg.model.revision)[0], device)
    if variant in LLM:
        return FrozenLlm(run_dir, cfg, variant, seed, resolve_local_path(cfg.llm.local_path, PROJECT_ROOT), train)
    raise ExecutionError(f"no frozen predictor for {variant}")


def _predict(pred, variant: str, view: QueryView, scratch: Path) -> np.ndarray:  # noqa: ANN001
    return pred.predict(view, scratch) if variant in LLM else pred.predict(view)


def freeze(cfg: StudyConfig, raw: dict, data_dir: Path, artifacts_dir: Path, out: Path, device: str,
           llm_check_rows: int = 64, skip_variants: tuple[str, ...] = ()) -> dict:
    if out.exists():
        raise PolicyError(f"{out} exists; a study freeze is immutable")
    registry = Registry()
    runs: dict[tuple[str, int], Path] = {}
    for rj in sorted((artifacts_dir / "runs").glob("*/run.json")):
        r = read_json(rj)
        if r.get("study_id") != cfg.study_id or r.get("status") != "completed":
            continue
        if r.get("scoped_config_sha256") != scoped_config_hash(raw, r["variant_id"]):
            continue
        key = (r["variant_id"], r["seed"])
        if key in runs:
            raise IntegrityError(f"two completed runs for {key}: {runs[key].name}, {rj.parent.name}; resolve before freezing")
        runs[key] = rj.parent
    missing = [(v, s) for v in cfg.experiments for s in _replicates(cfg, v) if (v, s) not in runs and v not in skip_variants]
    if missing:
        raise PolicyError(f"cannot freeze: missing completed validation runs {missing}")
    entries = []
    for v in cfg.experiments:
        if v in skip_variants:
            continue
        frame = y = train = None
        for s in _replicates(cfg, v):
            d = runs[(v, s)]
            run = _verify_run(d)
            metrics = read_json(d / "metrics.json")
            if frame is None:
                frame, y, _, train = _frames(cfg, data_dir, v)
            va = frame["split"] == "valid"
            query = QueryView(split="valid", frame=frame[va].reset_index(drop=True), lineage={})
            stored = pd.read_parquet(d / "predictions" / "validation.parquet")
            if stored["change_id"].tolist() != query.ids:
                raise IntegrityError(f"{d.name}: stored validation predictions are not aligned with the split")
            pred = predictor(cfg, v, s, d, train, device)
            kind = "llm" if v in LLM else ("neural" if v in NEURAL else "cpu")
            if kind == "llm":
                rows = np.random.default_rng([s, 909]).choice(len(query.ids), size=min(llm_check_rows, len(query.ids)), replace=False)
                sub = QueryView(split="valid", frame=query.frame.iloc[np.sort(rows)].reset_index(drop=True), lineage={})
                again = _predict(pred, v, sub, out.parent / ".freeze-check" / d.name)
                ref = stored["score"].to_numpy()[np.sort(rows)]
            else:
                again, ref = _predict(pred, v, query, out.parent), stored["score"].to_numpy()
            diff = float(np.max(np.abs(again - ref)))
            rank = None
            if kind == "llm":
                # bf16 logits depend on batch composition (padding/batch size); the same batch is bit-identical, so the
                # check is agreement within LLM_TOL plus rank agreement on the sample (exp 011: max 0.031, Spearman 0.995).
                from scipy.stats import spearmanr
                rank = float(spearmanr(again, ref).correlation)
                ok = diff <= LLM_TOL["max_abs"] and rank >= LLM_TOL["spearman"]
            else:
                ok = diff <= VALID_TOL[kind]
            if not ok:
                raise IntegrityError(f"{d.name} ({v} seed {s}): frozen predictor reproduces validation only to {diff:.2e}"
                                     + (f" (Spearman {rank:.4f})" if rank is not None else ""))
            entries.append({"variant_id": v, "seed": s, "run_id": run["run_id"], "family": registry.resolve(v).family,
                            "run_json_sha256": sha256_file(d / "run.json"), "artifacts_sha256": run["artifacts_sha256"],
                            "checkpoint_files_sha256": _checkpoint_files(d), "code_git": run.get("code_git"),
                            "source_digest": run.get("source_digest"),
                            "threshold": metrics["threshold_selection"]["threshold"], "validation_ap": metrics["ap"],
                            "validation_reproduction_max_abs_diff": diff, "validation_reproduction_spearman": rank,
                            "validation_reproduction_rows": len(ref)})
            print(f"freeze: {v} seed {s} {run['run_id']} reproduced validation (max |diff| {diff:.1e})", flush=True)
    record = {"status": "frozen", "study_id": cfg.study_id, "protocol_version": cfg.protocol_version,
              "frozen_at": datetime.now(UTC).isoformat(), "study_sha256": sha256_json(raw), "registry_sha256": registry.digest,
              "code_git": git_state(), "replicates": {v: _replicates(cfg, v) for v in cfg.experiments if v not in skip_variants},
              "skipped_variants": list(skip_variants), "runs": entries, "artifacts_root": str(artifacts_dir.name)}
    record["freeze_id"] = sha256_json({k: record[k] for k in ("study_sha256", "runs")})[:16]
    shutil.rmtree(out.parent / ".freeze-check", ignore_errors=True)
    atomic_write_json(out, record)
    return record


def verify_freeze(freeze_path: Path, artifacts_dir: Path) -> dict:
    rec = require_test_unlocked("test", freeze_path)
    for e in rec["runs"]:
        d = artifacts_dir / "runs" / e["run_id"]
        if sha256_file(d / "run.json") != e["run_json_sha256"]:
            raise IntegrityError(f"{e['run_id']}: run.json changed after freeze")
        for rel, digest in {**e["artifacts_sha256"], **e["checkpoint_files_sha256"]}.items():
            if sha256_file(d / rel) != digest:
                raise IntegrityError(f"{e['run_id']}: {rel} changed after freeze")
    return rec


def evaluate_public_test(cfg: StudyConfig, raw: dict, data_dir: Path, artifacts_dir: Path, freeze_path: Path, out_dir: Path,
                  device: str, variants: tuple[str, ...] | None = None) -> dict:
    rec = verify_freeze(freeze_path, artifacts_dir)
    if rec["study_sha256"] != sha256_json(raw):
        raise IntegrityError("study config changed after freeze")
    entries = [e for e in rec["runs"] if variants is None or e["variant_id"] in variants]
    done = []
    for e in entries:
        d = out_dir / e["run_id"]
        if (d / "metrics.json").exists():
            raise PolicyError(f"public test already evaluated for {e['run_id']}; results are never overwritten")
    cache: dict = {}
    for e in entries:
        v, s = e["variant_id"], e["seed"]
        if v not in cache:
            frame, y, lineage, train = _frames(cfg, data_dir, v, rec)
            te = frame["split"] == "test"
            cache = {v: (QueryView(split="test", frame=frame[te].reset_index(drop=True), lineage=lineage),
                         y[te].to_numpy(), train, frame[te]["change_id"].str.split(":").str[1].to_numpy())}
        query, yt, train, projects = cache[v]
        pred = predictor(cfg, v, s, artifacts_dir / "runs" / e["run_id"], train, device)
        d = out_dir / e["run_id"]
        t0 = time.perf_counter()
        scores = _predict(pred, v, query, d / "llm")
        secs = time.perf_counter() - t0
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"change_id": query.ids, "project": projects, "score": scores}).to_parquet(d / "predictions.parquet", index=False)
        m = {"split": "test", "evaluation_role": "frozen_final_test", "variant_id": v, "seed": s, "run_id": e["run_id"],
             "freeze_id": rec["freeze_id"], **evaluate(query.ids, yt, scores, cfg.evaluation.tie_salt, e["threshold"]),
             "threshold_source": "frozen validation max-F1 threshold", "predict_seconds": secs, "device": device,
             "evaluated_at": datetime.now(UTC).isoformat()}
        atomic_write_json(d / "metrics.json", m)
        done.append({"variant_id": v, "seed": s, "run_id": e["run_id"], "ap": m["ap"]})
        print(f"test: {v} seed {s} AP {m['ap']:.4f}", flush=True)
    heavy = any(e["variant_id"] in NEURAL | LLM for e in entries)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    atomic_write_json(out_dir / f"evaluation-{stamp}.json",
                      {"freeze_id": rec["freeze_id"], "evaluated": done, "environment": environment(heavy)})
    return {"freeze_id": rec["freeze_id"], "evaluated": done}


def _ap(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y, s)) if y.sum() else float("nan")


def report(cfg: StudyConfig, data_dir: Path, artifacts_dir: Path, freeze_path: Path, test_dir: Path, out: Path,
           n_boot: int = 2000, seed: int = 20261001) -> dict:
    rec = verify_freeze(freeze_path, artifacts_dir)
    frame, y, _ = load_frames(cfg, data_dir, "B0-LR", rec)
    lab = dict(zip(frame["change_id"], y, strict=True))
    per: dict[str, dict[int, dict]] = {}
    preds: dict[str, dict[int, pd.DataFrame]] = {}
    for e in rec["runs"]:
        tm = test_dir / e["run_id"] / "metrics.json"
        if not tm.exists():
            continue
        v, s = e["variant_id"], e["seed"]
        vm, t = read_json(artifacts_dir / "runs" / e["run_id"] / "metrics.json"), read_json(tm)
        per.setdefault(v, {})[s] = {"valid": vm, "test": t}
        preds.setdefault(v, {})[s] = pd.read_parquet(test_dir / e["run_id"] / "predictions.parquet")

    fields = {"ap": ("ap", None), "roc_auc": ("roc_auc", None), "recall_at_5pct": ("recall_at_5pct", "recall"),
              "recall_at_10pct": ("recall_at_10pct", "recall"), "f1_at_threshold": ("at_threshold", "f1"),
              "precision_at_threshold": ("at_threshold", "precision"), "recall_at_threshold": ("at_threshold", "recall")}

    def agg(v: str, split: str, key: str) -> dict:
        top, sub = fields[key]
        vals = []
        for r in per[v].values():
            x = r[split][top]
            vals.append(x[sub] if sub else x)
        a = np.asarray(vals, dtype=float)
        return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)) if len(a) > 1 else None, "n": len(a)}

    table = {v: {f"{split}_{k}": agg(v, split, k) for split in ("valid", "test")
                 for k in ("ap", "roc_auc", "recall_at_5pct", "recall_at_10pct")}
             | {f"test_{k}": agg(v, "test", k) for k in ("f1_at_threshold", "precision_at_threshold", "recall_at_threshold")}
             for v in per}
    any_v = next(iter(preds))
    base = preds[any_v][next(iter(preds[any_v]))][["change_id", "project"]]
    yv = base["change_id"].map(lab).to_numpy().astype(int)
    projects = base["project"].to_numpy()
    uniq = np.unique(projects)
    idx_by_p = {p: np.flatnonzero(projects == p) for p in uniq}
    rng = np.random.default_rng(seed)
    boots = [np.concatenate([idx_by_p[p] for p in rng.choice(uniq, size=len(uniq), replace=True)]) for _ in range(n_boot)]

    def scores(v: str) -> dict[int, np.ndarray]:
        out = {}
        for s, df in preds[v].items():
            if df["change_id"].tolist() != base["change_id"].tolist():
                raise IntegrityError("test predictions are not aligned across runs")
            out[s] = df["score"].to_numpy()
        return out

    pairs = []
    for a, b in PAIRS:
        if a not in preds or b not in preds:
            continue
        sa, sb = scores(a), scores(b)
        # Same replicate seeds -> pair by seed; otherwise (e.g. single-run L0) every combination is a pair.
        combos = [(k, k) for k in sorted(sa)] if sorted(sa) == sorted(sb) else [(ka, kb) for ka in sorted(sa) for kb in sorted(sb)]
        diffs = [_ap(yv, sa[ka]) - _ap(yv, sb[kb]) for ka, kb in combos]
        bd = [np.mean([_ap(yv[ix], sa[ka][ix]) - _ap(yv[ix], sb[kb][ix]) for ka, kb in combos]) for ix in boots]
        bd = np.asarray(bd)
        bd = bd[np.isfinite(bd)]
        pairs.append({"a": a, "b": b, "pairing": [list(c) for c in combos], "test_ap_diff_per_pair": diffs,
                      "test_ap_diff_mean": float(np.mean(diffs)), "seed_sd": float(np.std(diffs, ddof=1)) if len(diffs) > 1 else None,
                      "project_bootstrap_95ci": [float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))],
                      "project_bootstrap_p_le_0": float((bd <= 0).mean()), "bootstrap_reps": int(len(bd))})
    per_project = {}
    for v in preds:
        sv = scores(v)
        per_project[v] = {p: float(np.mean([_ap(yv[ix], sv[s][ix]) for s in sv])) for p, ix in idx_by_p.items()}
    result = {"freeze_id": rec["freeze_id"], "study_id": cfg.study_id, "test_n": int(len(yv)), "test_positives": int(yv.sum()),
              "projects": int(len(uniq)), "variants": table, "paired_differences": pairs, "per_project_test_ap": per_project,
              "bootstrap": {"unit": "project", "reps": n_boot, "seed": seed,
                            "note": "project resampling interval; separate from seed variance (seed_sd)"},
              **study_checks(cfg, rec, artifacts_dir, test_dir)}
    atomic_write_json(out, result)
    write_report_tables(result, out.with_suffix(".md"), out.with_suffix(".csv"))
    return result


CORE = ("B0-LR", "B0-LGBM", "B1-TFIDF-S", "B2-S", "B3-S", "B4-S", "B5-S")
COMPARATIVE = CORE + ("L0-S", "L1-S")


def study_checks(cfg: StudyConfig, rec: dict, artifacts_dir: Path, test_dir: Path) -> dict:
    """DoD-Core / DoD-Comparative status (spec requirements §10, AT-29/38/40) plus ledger, cost and failure summaries."""
    by_v: dict[str, list[dict]] = {}
    for e in rec["runs"]:
        by_v.setdefault(e["variant_id"], []).append(e)
    ledger, cost, failures, evidence, replication = {}, {}, {}, {}, {}
    for v, es in by_v.items():
        led = [read_json(artifacts_dir / "runs" / e["run_id"] / "label-access-ledger.json") for e in es]
        ledger[v] = {k: sorted({x.get(k, 0) for x in led}) for k in ("gradient_label_count", "demo_unique_label_count",
                                                                       "selection_label_count", "cpt_label_count",
                                                                       "rmi_synthetic_target_count", "index_labeled_count")}
        ledger[v]["adaptation_mode"] = led[0]["adaptation_mode"]
        c = [read_json(artifacts_dir / "runs" / e["run_id"] / "cost.json") for e in es]
        cpt_s = []
        for e in es:
            ta = artifacts_dir / "runs" / e["run_id"] / "token-accounting.json"
            if ta.exists() and "cpt" in read_json(ta):
                cpt_s.append(read_json(ta)["cpt"]["train_seconds_this_attempt"])
        cost[v] = {"device": sorted({x["device"] for x in c}), "gpu_seconds_total": float(sum(x.get("gpu_seconds") or 0 for x in c)),
                   "wall_seconds_mean": float(np.mean([x["wall_seconds"] for x in c])),
                   "cpt_seconds_mean": float(np.mean(cpt_s)) if cpt_s else None,
                   "peak_cuda_bytes_max": max((x.get("peak_cuda_bytes") or 0) for x in c) or None}
        bad = 0
        for e in es:
            vp = pd.read_parquet(artifacts_dir / "runs" / e["run_id"] / "predictions" / "validation.parquet")
            bad += int((vp["prediction_status"] != "ok").sum())
            tp = test_dir / e["run_id"] / "predictions.parquet"
            if tp.exists():
                bad += int((~np.isfinite(pd.read_parquet(tp)["score"].to_numpy())).sum())
        failures[v] = bad
        em = [read_json(artifacts_dir / "runs" / e["run_id"] / "evidence-manifest.json") for e in es]
        evidence[v] = {"evidence_manifest_sha256": sorted({x.get("evidence_manifest_sha256") or "none" for x in em}),
                       "query_hash_digest": sorted({x.get("query_hash_digest") or "none" for x in em}),
                       "structured_columns": sorted({",".join(x["structured_columns"]) for x in em})}
        seeds = sorted(e["seed"] for e in es)
        deterministic = v in ("B0-LR", "B0-LGBM", "B1-TFIDF-S", "L0-S")
        replication[v] = {"replicate_axis": "demo_set" if v == "L1-S" else ("none" if v == "L0-S" else "training_seed"),
                          "replicates": seeds,
                          "n_training_seeds_effective": 0 if v.startswith("L") else (1 if deterministic else len(seeds)),
                          "n_demo_sets": len(seeds) if v == "L1-S" else 0,
                          "note": "deterministic: identical replicates are not independent" if deterministic and len(seeds) > 1 else None}
    text_variants = [v for v in by_v if v != "B0-LR" and v != "B0-LGBM"]
    # Matched evidence is judged on content: experiments rebuild the evidence (manifest hashes differ by import time),
    # but the per-change query content hashes must be identical for every text-reading variant (AT-29).
    digests = {d for v in text_variants for d in evidence[v]["query_hash_digest"]}
    col_sets = {tuple(evidence[v]["structured_columns"]) for v in by_v}
    matched = len(digests) == 1 and "none" not in digests
    evidence_builds = sorted({m for v in text_variants for m in evidence[v]["evidence_manifest_sha256"]})
    tested = {v: all((test_dir / e["run_id"] / "metrics.json").exists() for e in by_v[v]) for v in by_v}

    def status(required: tuple[str, ...]) -> dict:
        missing = [v for v in required if v not in by_v]
        untested = [v for v in required if v in by_v and not tested[v]]
        failing = [v for v in required if failures.get(v)]
        checks = {"all_variants_frozen": not missing, "all_variants_tested_once": not untested, "matched_evidence": matched,
                  "same_structured_columns": len(col_sets) == 1, "no_prediction_failures": not failing,
                  "label_ledger_present": all(v in ledger for v in required if v in by_v)}
        return {"complete": all(checks.values()), "checks": checks, "missing": missing, "untested": untested, "failing": failing}

    comp = status(COMPARATIVE)
    if any(v not in by_v for v in ("L0-S", "L1-S")):
        comp["complete"] = False  # AT-40: no Comparative claim without the LLM variants
    return {"dod": {"core": status(CORE), "comparative": comp,
                    "internal_ready": {"complete": False, "note": "evaluated separately (bundle verify + offline predict fixture)"},
                    "internal_evaluated": {"complete": False, "note": "requires approved internal data (M7)"},
                    "l2": "not executed (extension, not required)"},
            "label_ledger": ledger, "cost": cost, "prediction_failures": failures, "evidence": evidence,
            "evidence_matching": {"query_hash_digests": sorted(digests), "distinct_evidence_builds": len(evidence_builds),
                                  "criterion": "identical per-change query content hashes across text-reading variants"},
            "replication": replication}


def write_report_tables(r: dict, md: Path, csv: Path) -> None:
    rows = []
    for v, t in r["variants"].items():
        rows.append({"variant": v, **{k: t[k]["mean"] for k in t}, "test_ap_sd": t["test_ap"]["sd"], "n": t["test_ap"]["n"]})
    pd.DataFrame(rows).to_csv(csv, index=False)
    lines = [f"# Study report {r['study_id']} (freeze {r['freeze_id']})", "",
             f"Public test: {r['test_n']} changes, {r['test_positives']} positives, {r['projects']} projects. "
             "Validation numbers are selection-validation; test was evaluated once after freeze.", "",
             "| variant | valid AP | test AP | sd | ROC-AUC | Recall@5% | Recall@10% | F1@thr |", "|---|---|---|---|---|---|---|---|"]
    for row in rows:
        sd = "" if row["test_ap_sd"] is None else f"{row['test_ap_sd']:.3f}"
        lines.append(f"| {row['variant']} | {row['valid_ap']:.3f} | {row['test_ap']:.3f} | {sd} | {row['test_roc_auc']:.3f} | "
                     f"{row['test_recall_at_5pct']:.3f} | {row['test_recall_at_10pct']:.3f} | {row['test_f1_at_threshold']:.3f} |")
    lines += ["", "| pair | test AP diff | project bootstrap 95% CI | P(diff<=0) |", "|---|---|---|---|"]
    for p in r["paired_differences"]:
        lo, hi = p["project_bootstrap_95ci"]
        pr = p["project_bootstrap_p_le_0"]
        lines.append(f"| {p['a']} - {p['b']} | {p['test_ap_diff_mean']:+.4f} | [{lo:+.3f}, {hi:+.3f}] | {pr:.3f} |")
    lines += ["", "## Definition-of-done status", ""]
    for k in ("core", "comparative"):
        d = r["dod"][k]
        lines.append(f"- **DoD-{k.capitalize()}**: {'complete' if d['complete'] else 'NOT complete'} — "
                     + ", ".join(f"{c}={'ok' if ok else 'FAIL'}" for c, ok in d["checks"].items()))
    lines += [f"- DoD-Internal-Ready: {r['dod']['internal_ready']['note']}",
              f"- DoD-Internal-Evaluated: {r['dod']['internal_evaluated']['note']}",
              f"- L2: {r['dod']['l2']}", "", "## Replication axes", ""]
    for v, x in r["replication"].items():
        lines.append(f"- {v}: axis={x['replicate_axis']}, replicates={x['replicates']}, effective training seeds="
                     f"{x['n_training_seeds_effective']}, demo sets={x['n_demo_sets']}" + (f" ({x['note']})" if x["note"] else ""))
    lines += ["", "## Label access (per run)", "",
              "| variant | mode | gradient | demo | selection | CPT labels | RMI synthetic |",
              "|---|---|---|---|---|---|---|"]
    for v, x in r["label_ledger"].items():
        lines.append(f"| {v} | {x['adaptation_mode']} | {x['gradient_label_count']} | {x['demo_unique_label_count']} | "
                     f"{x['selection_label_count']} | {x['cpt_label_count']} | {x['rmi_synthetic_target_count']} |")
    lines += ["", "## Cost", "", "| variant | device | GPU s (all replicates) | CPT s (mean) |", "|---|---|---|---|"]
    for v, x in r["cost"].items():
        cs = "" if x["cpt_seconds_mean"] is None else f"{x['cpt_seconds_mean']:.0f}"
        lines.append(f"| {v} | {','.join(x['device'])} | {x['gpu_seconds_total']:.0f} | {cs} |")
    md.write_text("\n".join(lines) + "\n")


def export_bundle(cfg: StudyConfig, raw: dict, artifacts_dir: Path, freeze_path: Path, out: Path) -> dict:
    """Inference-only bundle: frozen run states, best checkpoints, prompt manifests, study config and freeze record.

    Excludes optimizer/resume state, CPT heads, training data and base model weights (referenced by pinned revision/sha256).
    """
    rec = verify_freeze(freeze_path, artifacts_dir)
    if out.exists():
        raise PolicyError(f"{out} exists")
    files: dict[str, str] = {}
    out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out, "w:gz") as tar:
        def add(path: Path, arc: str) -> None:
            files[arc] = sha256_file(path)
            tar.add(path, arcname=arc, recursive=False)
        add(freeze_path, "freeze.json")
        for e in rec["runs"]:
            d = artifacts_dir / "runs" / e["run_id"]
            for rel in ["run.json", "model/state.json", "resolved-config.json", "evidence-manifest.json", "metrics.json",
                        *[r for r in e["checkpoint_files_sha256"] if not r.endswith("validation-scores.parquet")],
                        *(["model/lightgbm.txt"] if (d / "model" / "lightgbm.txt").exists() else []),
                        *(["prompts/manifest.json"] if (d / "prompts" / "manifest.json").exists() else [])]:
                add(d / rel, f"runs/{e['run_id']}/{rel}")
        manifest = {"freeze_id": rec["freeze_id"], "study_id": cfg.study_id, "files_sha256": files,
                    "base_models": {"encoder": {"id": cfg.model.base, "revision": cfg.model.revision,
                                                "weights_sha256": cfg.model.weights_sha256},
                                    "llm": None if not hasattr(cfg.llm, "model_id") else
                                    {"id": cfg.llm.model_id, "revision": cfg.llm.revision, "files_sha256": cfg.llm.files_sha256}},
                    "excluded": ["optimizer/resume state", "CPT MLM/RMI heads", "training data", "base model weights"],
                    "use": "offline inference only; internal data must not be used for training, selection or calibration"}
        mpath = out.parent / "export-manifest.json"
        atomic_write_json(mpath, manifest)
        tar.add(mpath, arcname="export-manifest.json")
        cfg_path = out.parent / "study.json"
        cfg_path.write_text(json.dumps(raw, indent=2, sort_keys=True))
        tar.add(cfg_path, arcname="study.json")
    return {"bundle": str(out), "bundle_sha256": sha256_file(out), "files": len(files), "freeze_id": rec["freeze_id"]}
