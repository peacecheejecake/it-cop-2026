"""Shared fixtures: real CodeBERT tokenizer (skip if the shared cache is absent) and a tiny random RoBERTa encoder."""
import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
STUDY_V2 = ROOT / "configs" / "studies" / "public-comparison-v2.yaml"
STUDY_V3 = ROOT / "configs" / "studies" / "public-comparison-v3.yaml"
REV = "3b0952feddeffad0063f274080e3c23d75e7eb39"


@pytest.fixture(scope="module")
def tok():
    from diff_lab.config import resolve_local_path
    from diff_lab.evidence import load_tokenizer
    try:
        path = resolve_local_path("cache:models/codebert-base", ROOT)
    except Exception:  # noqa: BLE001
        pytest.skip("shared cache not found")
    if not path.exists():
        pytest.skip("CodeBERT tokenizer snapshot not available")
    return load_tokenizer(path, REV)[0]


@pytest.fixture()
def tiny_encoder(tmp_path):
    torch = pytest.importorskip("torch")
    from transformers import RobertaConfig, RobertaModel
    torch.manual_seed(0)
    cfg = RobertaConfig(vocab_size=50265, hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32,
                        max_position_embeddings=514, type_vocab_size=1, pad_token_id=1)
    d = tmp_path / "tiny-encoder"
    RobertaModel(cfg, add_pooling_layer=False).save_pretrained(d)
    (d / "SOURCE.json").write_text(json.dumps({"resolved_revision": REV, "note": "random tiny test encoder"}))
    return d


def study_cfg(tmp_path: Path, encoder: Path | None = None, base: Path = STUDY_V3, **sections):  # noqa: ANN201
    """Load a copy of a registered study with sections replaced; pins the tiny encoder's weights when given."""
    from diff_lab.config import load_study
    from diff_lab.util import sha256_file
    raw = yaml.safe_load(base.read_text())
    raw.update(sections)
    if encoder is not None:
        raw["model"] = {**raw["model"], "weights_file": "model.safetensors",
                        "weights_sha256": sha256_file(encoder / "model.safetensors")}
    p = tmp_path / f"study-{len(list(tmp_path.glob('study-*.yaml')))}.yaml"
    p.write_text(yaml.safe_dump(raw, sort_keys=False))
    return load_study(p)[0]
