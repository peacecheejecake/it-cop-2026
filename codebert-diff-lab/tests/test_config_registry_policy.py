from pathlib import Path

import pandas as pd
import pytest
import yaml

from diff_lab.config import load_study, require_pinned
from diff_lab.policy import QueryView, TrainingDatasetView, require_test_unlocked, require_training_view
from diff_lab.registry import Registry
from diff_lab.util import ConfigError, PolicyError

STUDY = Path(__file__).resolve().parents[1] / "configs" / "studies" / "public-comparison-v2.yaml"


def _write(tmp_path: Path, mutate) -> Path:  # noqa: ANN001
    raw = yaml.safe_load(STUDY.read_text())
    mutate(raw)
    p = tmp_path / "s.yaml"
    p.write_text(yaml.safe_dump(raw, sort_keys=False))
    return p


def test_study_loads_and_registers_nine_primary():
    cfg, raw, h = load_study(STUDY)
    assert cfg.experiments == Registry().primary and len(h) == 64


def test_at01_unknown_field_rejected(tmp_path):
    with pytest.raises(ConfigError, match="extra"):
        load_study(_write(tmp_path, lambda r: r["structured"].update(typo_field=1)))


def test_at01_duplicate_key_rejected(tmp_path):
    p = tmp_path / "dup.yaml"
    p.write_text(STUDY.read_text() + "\nstudy_id: other\n")
    with pytest.raises(ConfigError, match="duplicate YAML key"):
        load_study(p)


def test_at01_forbidden_policy_cannot_be_enabled(tmp_path):
    with pytest.raises(ConfigError):
        load_study(_write(tmp_path, lambda r: r["policy"].update(allow_internal_training=True)))


def test_at01_pin_required_blocks_only_variants_that_read_it():
    _, raw, _ = load_study(STUDY)
    require_pinned(raw, "B0-LR")
    require_pinned(raw, "B1-TFIDF-S")
    require_pinned(raw, "B3-S")
    require_pinned(raw, "B4-S")
    with pytest.raises(ConfigError, match="cannot run"):
        require_pinned(raw, "L0-S")


def test_feature_allowlist_order_enforced(tmp_path):
    with pytest.raises(ConfigError):
        load_study(_write(tmp_path, lambda r: r["structured"]["features"].reverse()))


def test_at39_legacy_b1_rejected_and_explicit_migration():
    reg = Registry()
    with pytest.raises(ConfigError, match="v0.1"):
        reg.resolve("B1")
    m = reg.migrate_legacy("B1")
    assert m["mapped_variant_id"] == "B2-T" and m["legacy_matrix_version"] == 1


def _frame(n: int) -> pd.DataFrame:
    return pd.DataFrame({"change_id": [f"c{i}" for i in range(n)], "x": range(n)})


def test_at05_training_view_rejects_validation():
    with pytest.raises(PolicyError):
        TrainingDatasetView(split="valid", frame=_frame(3), lineage={"visibility": "public"}, labels=pd.Series([0, 1, 0]))
    with pytest.raises(PolicyError):
        TrainingDatasetView(split="train", frame=_frame(3), lineage={"visibility": "internal"}, labels=pd.Series([0, 1, 0]))
    q = QueryView(split="valid", frame=_frame(3), lineage={})
    with pytest.raises(PolicyError, match="TrainingDatasetView"):
        require_training_view(q)


def test_query_view_cannot_carry_labels():
    with pytest.raises(PolicyError):
        QueryView(split="test", frame=_frame(2).assign(label=[0, 1]), lineage={})


def test_at17_test_sealed_without_freeze(tmp_path):
    with pytest.raises(PolicyError, match="sealed"):
        require_test_unlocked("test", None)
    with pytest.raises(PolicyError, match="sealed"):
        require_test_unlocked("test", tmp_path / "missing.json")
    assert require_test_unlocked("valid", None) is None
