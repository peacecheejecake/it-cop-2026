"""Deployment-level internal evaluator on synthetic fixtures (FR-17): aggregation, observation window, exclusions."""
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from diff_lab.internal_eval import evaluate_internal, load_protocol
from diff_lab.util import ConfigError, PolicyError

PROTO = {"protocol_id": "t", "freeze_id": "f1", "unit": "deployment", "aggregation": "max", "budget_unit": "deployment_count",
         "q": [0.05, 0.1], "label_definition": "deployment linked to an incident within the window", "observation_window_days": 14,
         "evaluation_start": "2026-01-01", "evaluation_end": "2026-07-01", "min_linkage_confidence": 0.5,
         "stratify_by": "service", "tie_salt": "s"}


def _write(tmp, proto=PROTO, freeze="f1", labels_read=False):  # noqa: ANN001, ANN202
    rng = np.random.default_rng(0)
    p = tmp / "proto.yaml"
    p.write_text(yaml.safe_dump(proto))
    pred_dir = tmp / "pred"
    pred_dir.mkdir()
    (pred_dir / "offline-manifest.json").write_text(json.dumps({"freeze_id": freeze, "labels_read": labels_read}))
    n_dep = 60
    changes = [(f"c{d}-{k}", f"d{d}") for d in range(n_dep) for k in range(1 + d % 3)]
    mp = pd.DataFrame(changes, columns=["change_id", "deployment_id"])
    label = np.array([int(d % 5 == 0) for d in range(n_dep)])
    score = {cid: float(0.9 if label[int(dep[1:])] and cid.endswith("-0") else rng.uniform(0, 0.6)) for cid, dep in changes}
    pred = pd.DataFrame({"change_id": list(score), "score": list(score.values()), "variant_id": "B3-S", "seed": 42, "run_id": "r1"})
    pred.to_parquet(pred_dir / "predictions.parquet")
    mp.to_parquet(tmp / "map.parquet")
    dep = pd.DataFrame({"deployment_id": [f"d{d}" for d in range(n_dep)],
                        "deployed_at": pd.date_range("2026-01-05", periods=n_dep, freq="2D", tz="UTC"),
                        "label": label.astype(float), "service": ["a" if d % 2 else "b" for d in range(n_dep)],
                        "linkage_confidence": 0.9})
    dep["observation_end"] = dep["deployed_at"] + pd.Timedelta(days=30)
    dep.loc[3, "observation_end"] = dep.loc[3, "deployed_at"] + pd.Timedelta(days=2)
    dep.loc[4, "label"] = np.nan
    dep.loc[6, "linkage_confidence"] = 0.1
    dep.to_parquet(tmp / "dep.parquet")
    return p, pred_dir / "predictions.parquet", tmp / "map.parquet", tmp / "dep.parquet"


def test_internal_evaluation_rules(tmp_path):
    p, pred, mp, dep = _write(tmp_path)
    res = evaluate_internal(p, pred, mp, dep, tmp_path / "report.json")
    rep = json.loads((tmp_path / "report.json").read_text())
    cov = rep["coverage"]
    assert cov["excluded_unobserved_window"] == 1 and cov["excluded_unknown_label"] == 1 and cov["excluded_low_linkage"] == 1
    assert res["eligible_deployments"] == cov["deployments_in_period"] - 3
    run = rep["runs"][0]
    assert run["ap"] == pytest.approx(1.0) and run["recall_at_10pct"] is not None
    assert set(run["by_stratum"]) == {"a", "b"} and run["score_vs_n_changes_spearman"] is not None


def test_unpinned_protocol_and_foreign_predictions_refused(tmp_path):
    with pytest.raises(ConfigError, match="unpinned"):
        q = tmp_path / "x.yaml"
        q.write_text(yaml.safe_dump({**PROTO, "observation_window_days": "PIN_REQUIRED"}))
        load_protocol(q)
    p, pred, mp, dep = _write(tmp_path, freeze="other")
    with pytest.raises(PolicyError, match="freeze"):
        evaluate_internal(p, pred, mp, dep, tmp_path / "r.json")


def test_predictions_that_read_labels_refused(tmp_path):
    p, pred, mp, dep = _write(tmp_path, labels_read=True)
    with pytest.raises(PolicyError, match="label-free"):
        evaluate_internal(p, pred, mp, dep, tmp_path / "r.json")


def test_registered_template_blocks_until_pinned():
    from conftest import ROOT
    with pytest.raises(ConfigError, match="unpinned"):
        load_protocol(ROOT / "configs" / "internal" / "internal-eval-v1.yaml")
