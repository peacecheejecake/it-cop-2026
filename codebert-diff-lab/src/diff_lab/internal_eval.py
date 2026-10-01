"""Deployment-level evaluation of offline internal predictions (spec FR-17, protocol §10, DoD-Internal-Evaluated).

This is the only place internal labels are read; predictions are produced earlier by `predict`
without labels, and nothing here feeds back into a model, threshold, prompt or index. Every rule
comes from a protocol file registered before the internal results are seen:

- unit: deployment; commit scores are aggregated with the registered rule (first candidate: max,
  whose bias toward deployments with many commits is reported, never "fixed" with internal labels);
- labels: incident linked to the deployment; a deployment counts only when its observation
  window is complete (observation_end >= deployed_at + window) and its linkage confidence meets the
  registered minimum; otherwise it is reported as unobserved/low-confidence, not as negative;
- evaluation period and time order: deployments inside the registered period, reported by month;
- budget: Recall@q with K = ceil(q * number of evaluated deployments) (unit deployment_count).
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .metrics import evaluate
from .util import ConfigError, IntegrityError, PolicyError, atomic_write_json, read_json, sha256_file, sha256_json


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class InternalProtocol(_Strict):
    protocol_id: str
    freeze_id: str
    unit: Literal["deployment"]
    aggregation: Literal["max", "mean", "noisy_or"]
    budget_unit: Literal["deployment_count"]
    q: list[Literal[0.05, 0.1]]
    label_definition: str = Field(min_length=10)
    observation_window_days: int = Field(ge=1)
    evaluation_start: str
    evaluation_end: str
    min_linkage_confidence: float | None = None
    stratify_by: str | None = None
    tie_salt: str


def load_protocol(path: Path) -> tuple[InternalProtocol, str]:
    raw = yaml.safe_load(path.read_text())
    pins = [k for k, v in raw.items() if v == "PIN_REQUIRED"]
    if pins:
        raise ConfigError(f"internal protocol has unpinned fields {pins}; register them before evaluating")
    try:
        return InternalProtocol.model_validate(raw), sha256_json(raw)
    except ValidationError as e:
        raise ConfigError(f"invalid internal protocol: {e}") from e


def _aggregate(scores: pd.Series, rule: str) -> float:
    s = scores.to_numpy(dtype=float)
    if rule == "max":
        return float(s.max())
    if rule == "mean":
        return float(s.mean())
    return float(1.0 - np.prod(1.0 - s))


def evaluate_internal(protocol_path: Path, predictions: Path, mapping: Path, deployments: Path, out: Path) -> dict:
    proto, proto_hash = load_protocol(protocol_path)
    pred_manifest = read_json(predictions.parent / "offline-manifest.json")
    if pred_manifest["freeze_id"] != proto.freeze_id:
        raise PolicyError(f"predictions come from freeze {pred_manifest['freeze_id']}, protocol registered {proto.freeze_id}")
    if pred_manifest.get("labels_read") is not False:
        raise PolicyError("predictions were not produced by the label-free offline predictor")
    pred = pd.read_parquet(predictions)
    mp = pd.read_parquet(mapping)[["change_id", "deployment_id"]]
    dep = pd.read_parquet(deployments)
    need = {"deployment_id", "deployed_at", "label", "observation_end"}
    if not need <= set(dep.columns):
        raise ConfigError(f"deployments file needs columns {sorted(need)}")
    if not dep["deployment_id"].is_unique or mp.duplicated().any():
        raise IntegrityError("duplicate deployment_id or mapping rows")
    dep = dep.copy()
    dep["deployed_at"] = pd.to_datetime(dep["deployed_at"], utc=True)
    dep["observation_end"] = pd.to_datetime(dep["observation_end"], utc=True)
    start, end = pd.Timestamp(proto.evaluation_start, tz="UTC"), pd.Timestamp(proto.evaluation_end, tz="UTC")
    in_period = (dep["deployed_at"] >= start) & (dep["deployed_at"] < end)
    window = pd.Timedelta(days=proto.observation_window_days)
    observed = dep["observation_end"] >= dep["deployed_at"] + window
    known = dep["label"].notna()
    conf_ok = pd.Series(True, index=dep.index)
    if proto.min_linkage_confidence is not None:
        if "linkage_confidence" not in dep.columns:
            raise ConfigError("protocol sets min_linkage_confidence but deployments lack linkage_confidence")
        conf_ok = dep["linkage_confidence"].fillna(-1.0) >= proto.min_linkage_confidence
    scored_changes = set(pred["change_id"])
    coverage = {
        "deployments_total": int(len(dep)), "deployments_in_period": int(in_period.sum()),
        "excluded_unobserved_window": int((in_period & ~observed).sum()),
        "excluded_unknown_label": int((in_period & observed & ~known).sum()),
        "excluded_low_linkage": int((in_period & observed & known & ~conf_ok).sum()),
        "mapped_changes": int(mp["change_id"].nunique()), "scored_changes": len(scored_changes),
        "scored_changes_without_deployment": len(scored_changes - set(mp["change_id"])),
        "mapped_changes_without_score": len(set(mp["change_id"]) - scored_changes)}
    eligible = dep[in_period & observed & known & conf_ok].copy()
    eligible["label"] = eligible["label"].astype(int)
    results, size_bias = [], {}
    for rid, g in pred.groupby("run_id"):
        joined = mp.merge(g[["change_id", "score"]], on="change_id", how="inner")
        agg = joined.groupby("deployment_id")["score"].apply(lambda s: _aggregate(s, proto.aggregation)).rename("score")
        n_changes = joined.groupby("deployment_id").size().rename("n_changes")
        ev = eligible.merge(agg, on="deployment_id", how="left").merge(n_changes, on="deployment_id", how="left")
        unscored = int(ev["score"].isna().sum())
        ev = ev[ev["score"].notna()].sort_values("deployed_at")
        full = evaluate(ev["deployment_id"].astype(str).tolist(), ev["label"].to_numpy(), ev["score"].to_numpy(), proto.tie_salt)
        m = {k: full[k] for k in ("n", "positives", "prevalence", "ap", "roc_auc", "null_reasons")}
        m["recall_at_5pct"], m["recall_at_10pct"] = full["recall_at_5pct"]["recall"], full["recall_at_10pct"]["recall"]
        by_month = {str(p): {"n": int(len(x)), "positives": int(x["label"].sum())}
                    for p, x in ev.groupby(ev["deployed_at"].dt.strftime("%Y-%m"))}
        strata = {}
        if proto.stratify_by and proto.stratify_by in ev.columns:
            for k, x in ev.groupby(proto.stratify_by):
                sm = evaluate(x["deployment_id"].astype(str).tolist(), x["label"].to_numpy(), x["score"].to_numpy(), proto.tie_salt)
                strata[str(k)] = {"n": sm["n"], "positives": sm["positives"], "ap": sm["ap"]}
        from scipy.stats import spearmanr
        rho = spearmanr(ev["score"], ev["n_changes"]).correlation if ev["n_changes"].nunique() > 1 else None
        size_bias[rid] = None if rho is None or rho != rho else float(rho)
        results.append({"run_id": rid, "variant_id": g["variant_id"].iloc[0], "seed": int(g["seed"].iloc[0]),
                        "deployments_without_scored_change": unscored, **m, "by_month": by_month, "by_stratum": strata,
                        "score_vs_n_changes_spearman": size_bias[rid]})
    size_ref = None
    if len(eligible):
        n_all = mp.groupby("deployment_id").size().rename("n_changes")
        ref = eligible.merge(n_all, on="deployment_id", how="inner")
        if len(ref) and ref["label"].nunique() == 2:
            size_ref = evaluate(ref["deployment_id"].astype(str).tolist(), ref["label"].to_numpy(),
                                ref["n_changes"].to_numpy(dtype=float), proto.tie_salt)["ap"]
    report = {"protocol_id": proto.protocol_id, "protocol_sha256": proto_hash, "freeze_id": proto.freeze_id,
              "aggregation": proto.aggregation, "budget_unit": proto.budget_unit, "coverage": coverage,
              "inputs_sha256": {"predictions": sha256_file(predictions), "mapping": sha256_file(mapping),
                                "deployments": sha256_file(deployments)},
              "size_bias_reference_ap_by_n_changes": size_ref, "runs": results,
              "note": "labels used for evaluation only; no model, threshold, prompt, index or aggregation choice uses them"}
    atomic_write_json(out, report)
    return {"runs": len(results), "eligible_deployments": int(len(eligible)), "coverage": coverage}
