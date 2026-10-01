"""Lineage cross-validation (AT-04), training-view alignment, and the upstream-clean2 CPT-dev grouping."""
import shutil
from pathlib import Path

import pandas as pd
import pytest

from diff_lab.data.splits import _components, _dev_ids
from diff_lab.lineage import validate_lineage
from diff_lab.policy import TrainingDatasetView
from diff_lab.util import IntegrityError, PolicyError, atomic_write_json, sha256_file

ROOT = Path(__file__).resolve().parents[1]
APPROVAL = ROOT / "configs" / "sources" / "jit-defects4j.yaml"


def _fake(tmp_path: Path, approval: Path = APPROVAL) -> tuple[Path, Path, Path]:
    import yaml
    a = yaml.safe_load(approval.read_text())
    snap, split, ev = tmp_path / "snap", tmp_path / "split", tmp_path / "ev"
    for d in (snap, split, ev):
        d.mkdir(parents=True)
    provided = pd.DataFrame({"change_id": ["t1", "t2", "v1", "x1"], "split": ["train", "train", "valid", "test"]})
    provided.to_parquet(snap / "splits.parquet")
    atomic_write_json(snap / "manifest.json", {"visibility": "public", "source": {
        "source_id": a["source_id"], "archive_sha256": a["archive_sha256"], "approval_file_sha256": sha256_file(approval)}})
    provided.iloc[[0, 2, 3]].to_parquet(split / "splits.parquet")
    atomic_write_json(split / "manifest.json", {"parent_snapshot_manifest_sha256": sha256_file(snap / "manifest.json")})
    atomic_write_json(ev / "manifest.json", {"snapshot_manifest_sha256": sha256_file(snap / "manifest.json"),
                                             "split_manifest_sha256": sha256_file(split / "manifest.json"),
                                             "tokenizer": {"files_sha256": "tokdigest"}})
    return snap, split, ev


def test_at04_lineage_accepts_consistent_chain(tmp_path):
    snap, split, ev = _fake(tmp_path)
    out = validate_lineage(APPROVAL, snap, split, ev, "tokdigest", ("supervised_train", "selection", "cpt_train"))
    assert out["approval_source_id"]


def test_at04_lineage_rejects_forged_parents_and_membership(tmp_path):
    snap, split, ev = _fake(tmp_path)
    with pytest.raises(IntegrityError, match="tokenizer"):
        validate_lineage(APPROVAL, snap, split, ev, "other", ("selection",))
    atomic_write_json(split / "manifest.json", {"parent_snapshot_manifest_sha256": "0" * 64})
    with pytest.raises(IntegrityError, match="different snapshot"):
        validate_lineage(APPROVAL, snap, split, None, None, ("selection",))
    snap, split, ev = _fake(tmp_path / "b")
    pd.DataFrame({"change_id": ["t1", "v1", "x1"], "split": ["train", "train", "test"]}).to_parquet(split / "splits.parquet")
    with pytest.raises(IntegrityError, match="valid membership"):
        validate_lineage(APPROVAL, snap, split, None, None, ("selection",))


def test_at04_edited_approval_or_unapproved_role_blocks(tmp_path):
    snap, split, _ = _fake(tmp_path)
    edited = tmp_path / "approval.yaml"
    shutil.copy(APPROVAL, edited)
    edited.write_text(edited.read_text() + "\n# edited\n")
    with pytest.raises(PolicyError, match="approval file changed"):
        validate_lineage(edited, snap, split, None, None, ("selection",))
    with pytest.raises(PolicyError, match="not approved"):
        validate_lineage(APPROVAL, snap, split, None, None, ("internal_training",))


def test_training_view_rejects_misaligned_labels():
    f = pd.DataFrame({"change_id": ["a", "b"]})
    with pytest.raises(PolicyError, match="row index"):
        TrainingDatasetView(split="train", frame=f, lineage={"visibility": "public"}, labels=pd.Series([0, 1], index=[5, 6]))
    with pytest.raises(PolicyError, match="label column"):
        TrainingDatasetView(split="train", frame=f.assign(label=[0, 1]), lineage={"visibility": "public"}, labels=pd.Series([0, 1]))


def test_components_link_exact_and_code_keys():
    comp = _components(["a", "b", "c", "d"], [["x", "y", "x", "z"], [None, "k", "k", None]])
    assert comp["a"] == comp["b"] == comp["c"] and comp["d"] == "d"


def test_clean2_keeps_duplicate_groups_on_one_side():
    n = 400
    ids = [f"c{i:03d}" for i in range(n)]
    exact = [f"e{i // 4}" for i in range(n)]
    code = [f"k{i // 2}" if i % 7 else None for i in range(n)]
    clean = pd.DataFrame({"change_id": ids, "exact_key": exact, "code_key": code})
    dev = _dev_ids(clean, "upstream-clean2")
    groups = clean.assign(g=[e for e in exact]).groupby("g")["change_id"].apply(set)
    assert all(g <= dev or not (g & dev) for g in groups)
    assert 0.04 <= len(dev) / n <= 0.08
    assert _dev_ids(clean, "upstream-clean2") == dev
    assert len(_dev_ids(clean, "upstream-clean1")) == round(n * 0.05)


def test_environment_for_non_neural_runs_does_not_import_torch():
    import subprocess
    import sys
    code = "import sys; from diff_lab.runner import environment; e = environment(False); assert 'torch' not in sys.modules, 'torch'"
    assert subprocess.run([sys.executable, "-c", code], capture_output=True, text=True).returncode == 0
