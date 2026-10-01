"""Neural contract tests on a tiny random RoBERTa (NOT CodeBERT; spec NFR-08/AT-23: no GPU/real-model claim)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

torch = pytest.importorskip("torch")
from transformers import RobertaConfig, RobertaModel  # noqa: E402

from diff_lab.config import FEATURE_PROFILE_JIT14, load_study, resolve_local_path  # noqa: E402
from diff_lab.evidence import load_tokenizer  # noqa: E402
from diff_lab.neural import NeuralRun, load_encoder, state_hash  # noqa: E402
from diff_lab.policy import QueryView, TrainingDatasetView  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "configs" / "studies" / "public-comparison-v2.yaml"
REV = "3b0952feddeffad0063f274080e3c23d75e7eb39"
FT = {"max_epochs": 2, "micro_batch_size": 4, "gradient_accumulation_steps": 2, "learning_rate": 1e-3, "weight_decay": 0.01,
      "lr_schedule": "constant", "max_grad_norm": 1.0, "head_dropout": 0.1, "selection_metric": "validation_ap",
      "patience_epochs": 5, "tie_rule": "earlier_checkpoint", "precision": "fp32", "device": "cpu", "allow_cpu": True,
      "eval_batch_size": 8}


@pytest.fixture(scope="module")
def tok():
    try:
        path = resolve_local_path("cache:models/codebert-base", ROOT)
    except Exception:  # noqa: BLE001
        pytest.skip("shared cache not found")
    if not path.exists():
        pytest.skip("CodeBERT tokenizer snapshot not available")
    return load_tokenizer(path, REV)[0]


@pytest.fixture()
def tiny_encoder(tmp_path):
    torch.manual_seed(0)
    cfg = RobertaConfig(vocab_size=50265, hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32,
                        max_position_embeddings=514, type_vocab_size=1, pad_token_id=1)
    d = tmp_path / "tiny-encoder"
    RobertaModel(cfg, add_pooling_layer=False).save_pretrained(d)
    (d / "SOURCE.json").write_text(json.dumps({"resolved_revision": REV, "note": "random tiny test encoder"}))
    return d


def _cfg(tmp_path, ft=FT):  # noqa: ANN001
    raw = yaml.safe_load(STUDY.read_text())
    raw["finetune"] = ft
    p = tmp_path / "study.yaml"
    p.write_text(yaml.safe_dump(raw, sort_keys=False))
    return load_study(p)[0]


def _views(n=24):  # noqa: ANN001
    rng = np.random.default_rng(1)
    f = pd.DataFrame(rng.integers(0, 20, (2 * n, len(FEATURE_PROFILE_JIT14))).astype(float), columns=FEATURE_PROFILE_JIT14)
    f["change_id"] = [f"c{i}" for i in range(2 * n)]
    f["query_text"] = [("fix null check\n+ if ( x == null )" if i % 2 else "update docs\n- old line") for i in range(2 * n)]
    y = pd.Series([i % 2 for i in range(2 * n)])
    tr = TrainingDatasetView(split="train", frame=f.iloc[:n].reset_index(drop=True), lineage={"visibility": "public"},
                             labels=y.iloc[:n].reset_index(drop=True))
    va = QueryView(split="valid", frame=f.iloc[n:].reset_index(drop=True), lineage={})
    return tr, va, y.iloc[n:].to_numpy()


def test_at09_b2_b3_share_initial_encoder_and_head(tmp_path, tok, tiny_encoder):
    cfg = _cfg(tmp_path)
    tr, va, yv = _views()
    r2 = NeuralRun(cfg, "B2-S", 42, tiny_encoder, tmp_path / "b2", tok).fit_predict(tr, va, yv, "salt")
    r3 = NeuralRun(cfg, "B3-S", 42, tiny_encoder, tmp_path / "b3", tok).fit_predict(tr, va, yv, "salt")
    assert r2["encoder_init_state_sha256"] == r3["encoder_init_state_sha256"]
    assert r2["head_init_state_sha256"] == r3["head_init_state_sha256"]
    assert r2["frozen_encoder"] and not r3["frozen_encoder"]
    assert len(r2["scores"]) == len(va) and np.isfinite(r3["scores"]).all()


def test_at28_frozen_encoder_unchanged_and_b3_updates(tmp_path, tok, tiny_encoder):
    cfg = _cfg(tmp_path)
    tr, va, yv = _views()
    NeuralRun(cfg, "B3-S", 42, tiny_encoder, tmp_path / "b3", tok).fit_predict(tr, va, yv, "salt")
    from safetensors.torch import load_file
    enc0, _ = load_encoder(tiny_encoder, REV)
    tuned = load_file(str(tmp_path / "b3" / "checkpoints" / "best" / "encoder.safetensors"))
    assert any(not torch.equal(enc0.state_dict()[k], tuned[k]) for k in tuned)
    r2 = NeuralRun(cfg, "B2-S", 42, tiny_encoder, tmp_path / "b2", tok).fit_predict(tr, va, yv, "salt")
    assert r2["embedding_cache"]["encoder_state_sha256"] == state_hash(enc0)
    assert not (tmp_path / "b2" / "checkpoints" / "best" / "encoder.safetensors").exists()


def test_at14_resume_continues_from_last_epoch(tmp_path, tok, tiny_encoder):
    tr, va, yv = _views()
    one = _cfg(tmp_path, {**FT, "max_epochs": 1})
    NeuralRun(one, "B3-S", 7, tiny_encoder, tmp_path / "run", tok).fit_predict(tr, va, yv, "salt")
    two = _cfg(tmp_path, {**FT, "max_epochs": 2})
    r = NeuralRun(two, "B3-S", 7, tiny_encoder, tmp_path / "run", tok).fit_predict(tr, va, yv, "salt")
    assert [h["epoch"] for h in r["history"]] == [1, 2]
    fresh = NeuralRun(two, "B3-S", 7, tiny_encoder, tmp_path / "fresh", tok).fit_predict(tr, va, yv, "salt")
    assert r["history"][1]["train_loss"] == pytest.approx(fresh["history"][1]["train_loss"], rel=1e-5)


def test_cuda_request_without_cuda_fails(tmp_path, tok, tiny_encoder):
    if torch.cuda.is_available():
        pytest.skip("CUDA present")
    cfg = _cfg(tmp_path, {**FT, "device": "cuda"})
    tr, va, yv = _views()
    with pytest.raises(Exception, match="no silent CPU fallback"):
        NeuralRun(cfg, "B3-S", 1, tiny_encoder, tmp_path / "x", tok).fit_predict(tr, va, yv, "salt")
