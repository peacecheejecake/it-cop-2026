"""M6 freeze -> one-shot test -> report -> export on a synthetic snapshot with B0-LR (AT-17 and freeze integrity)."""
import tarfile

import numpy as np
import pandas as pd
import pytest
import yaml
from conftest import ROOT, STUDY_V3

from diff_lab.config import FEATURE_PROFILE_JIT14, load_study
from diff_lab.policy import SourceApproval
from diff_lab.runner import run_variant
from diff_lab.study import evaluate_public_test, export_bundle, freeze, report
from diff_lab.util import IntegrityError, PolicyError, atomic_write_json, sha256_file

APPROVAL = ROOT / "configs" / "sources" / "jit-defects4j.yaml"


def _dataset(root):  # noqa: ANN001, ANN202
    rng = np.random.default_rng(0)
    n = 300
    split = np.array(["train"] * 180 + ["valid"] * 60 + ["test"] * 60)
    ids = [f"jitd4j:proj{i % 5}:{i:040x}" for i in range(n)]
    x = rng.normal(size=(n, len(FEATURE_PROFILE_JIT14)))
    y = (x[:, 4] + rng.normal(scale=1.0, size=n) > 0.8).astype(int)
    snap = root / "snapshots" / "snap"
    sp = root / "splits" / "snap" / "upstream-clean2"
    snap.mkdir(parents=True)
    sp.mkdir(parents=True)
    feats = pd.DataFrame(np.abs(x) * 10, columns=FEATURE_PROFILE_JIT14).assign(change_id=ids)
    feats.to_parquet(snap / "features.parquet")
    pd.DataFrame({"change_id": ids, "label": y}).to_parquet(snap / "labels.parquet")
    pd.DataFrame({"change_id": ids, "split": split}).to_parquet(snap / "splits.parquet")
    a = SourceApproval.load(APPROVAL)
    atomic_write_json(snap / "manifest.json", {
        "visibility": "public", "source": {"source_id": a.source_id, "archive_sha256": a.archive_sha256,
                                           "approval_file_sha256": sha256_file(APPROVAL)},
        "feature_schema": {"feature_schema_id": "jit14-provided-v1"},
        "table_sha256": {t: sha256_file(snap / f"{t}.parquet") for t in ("features", "labels", "splits")}})
    pd.DataFrame({"change_id": ids, "split": split, "cpt_role": None, "group_id": ids}).to_parquet(sp / "splits.parquet")
    atomic_write_json(sp / "manifest.json", {"parent_snapshot_manifest_sha256": sha256_file(snap / "manifest.json"),
                                             "splits_parquet_sha256": sha256_file(sp / "splits.parquet")})
    return y[split == "test"]


@pytest.fixture()
def study(tmp_path):
    raw = yaml.safe_load(STUDY_V3.read_text())
    raw["dataset"]["snapshot_id"] = "snap"
    raw["experiments"] = ["B0-LR"]
    raw["seeds"] = [42, 43]
    p = tmp_path / "study.yaml"
    p.write_text(yaml.safe_dump(raw, sort_keys=False))
    data, arts = tmp_path / "data", tmp_path / "artifacts"
    _dataset(data)
    cfg, raw, h = load_study(p)
    for s in (42, 43):
        run_variant(cfg, raw, h, "B0-LR", s, data, arts)
    return cfg, raw, data, arts, tmp_path


def test_at17_freeze_test_report_export_end_to_end(study):
    cfg, raw, data, arts, tmp = study
    fz = tmp / "freeze" / "freeze.json"
    with pytest.raises(PolicyError, match="sealed"):
        evaluate_public_test(cfg, raw, data, arts, fz, tmp / "test", "cpu")
    rec = freeze(cfg, raw, data, arts, fz, "cpu")
    assert rec["status"] == "frozen" and len(rec["runs"]) == 2
    assert all(e["validation_reproduction_max_abs_diff"] < 1e-9 for e in rec["runs"])
    with pytest.raises(PolicyError, match="immutable"):
        freeze(cfg, raw, data, arts, fz, "cpu")
    res = evaluate_public_test(cfg, raw, data, arts, fz, tmp / "test", "cpu")
    assert len(res["evaluated"]) == 2
    with pytest.raises(PolicyError, match="never overwritten"):
        evaluate_public_test(cfg, raw, data, arts, fz, tmp / "test", "cpu")
    r = report(cfg, data, arts, fz, tmp / "test", tmp / "report.json", n_boot=50)
    assert r["test_n"] == 60 and r["projects"] == 5 and "test_ap" in r["variants"]["B0-LR"]
    import json
    f1s = [json.loads((tmp / "test" / e["run_id"] / "metrics.json").read_text())["at_threshold"]["f1"] for e in rec["runs"]]
    assert r["variants"]["B0-LR"]["test_f1_at_threshold"]["mean"] == pytest.approx(float(np.mean(f1s)))
    b = export_bundle(cfg, raw, arts, fz, tmp / "export" / "bundle.tar.gz")
    with tarfile.open(b["bundle"]) as t:
        names = t.getnames()
    assert "freeze.json" in names and "export-manifest.json" in names
    assert not any("state.pt" in n or "data/" in n for n in names)


def test_freeze_requires_all_replicates_and_detects_tampering(study):
    cfg, raw, data, arts, tmp = study
    raw2 = {**raw, "seeds": [42, 43, 44]}
    cfg2 = cfg.model_copy(update={"seeds": [42, 43, 44]})
    with pytest.raises(PolicyError, match="missing completed validation runs"):
        freeze(cfg2, raw2, data, arts, tmp / "f2" / "freeze.json", "cpu")
    fz = tmp / "freeze" / "freeze.json"
    rec = freeze(cfg, raw, data, arts, fz, "cpu")
    pred = arts / "runs" / rec["runs"][0]["run_id"] / "metrics.json"
    pred.write_text(pred.read_text() + " ")
    with pytest.raises(IntegrityError, match="changed after freeze"):
        evaluate_public_test(cfg, raw, data, arts, fz, tmp / "test", "cpu")


def test_frozen_b1_reproduces_predictions_after_sorted_json_roundtrip(tmp_path):
    from diff_lab.frozen import FrozenB1
    from diff_lab.models import B1TFIDF
    from diff_lab.policy import QueryView, TrainingDatasetView
    cfg = load_study(STUDY_V3)[0]
    rng = np.random.default_rng(1)
    n = 80
    f = pd.DataFrame(rng.integers(0, 30, (n, len(FEATURE_PROFILE_JIT14))).astype(float), columns=FEATURE_PROFILE_JIT14)
    f["change_id"] = [f"c{i}" for i in range(n)]
    f["message_text"] = [f"fix bug {i % 7} in parser" if i % 2 else f"add feature {i % 5}" for i in range(n)]
    f["code_text"] = [f"+ if (x{i % 3} == null) return;" if i % 2 else f"- old{i % 4}();" for i in range(n)]
    y = pd.Series([i % 2 for i in range(n)])
    tr = TrainingDatasetView(split="train", frame=f.iloc[:60].reset_index(drop=True), lineage={"visibility": "public"},
                             labels=y.iloc[:60].reset_index(drop=True))
    q = QueryView(split="valid", frame=f.iloc[60:].reset_index(drop=True), lineage={})
    m = B1TFIDF(cfg, 42)
    state = m.fit(tr)
    d = tmp_path / "run"
    atomic_write_json(d / "model" / "state.json", state)
    assert np.abs(FrozenB1(d, cfg).predict(q) - m.predict(q)).max() < 1e-12
