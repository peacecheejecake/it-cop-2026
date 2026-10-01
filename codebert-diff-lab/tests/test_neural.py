"""Neural contract tests on a tiny random RoBERTa (NOT CodeBERT; spec NFR-08/AT-23: no GPU/real-model claim)."""
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from conftest import REV, study_cfg  # noqa: E402

from diff_lab.config import FEATURE_PROFILE_JIT14  # noqa: E402
from diff_lab.neural import NeuralRun, load_encoder, state_hash  # noqa: E402
from diff_lab.policy import QueryView, TrainingDatasetView  # noqa: E402
from diff_lab.util import resolve_generation  # noqa: E402

FT = {"max_epochs": 2, "micro_batch_size": 4, "gradient_accumulation_steps": 2, "learning_rate": 1e-3, "weight_decay": 0.01,
      "lr_schedule": "constant", "max_grad_norm": 1.0, "head_dropout": 0.1, "selection_metric": "validation_ap",
      "patience_epochs": 5, "tie_rule": "earlier_checkpoint", "precision": "fp32", "device": "cpu", "allow_cpu": True,
      "eval_batch_size": 8}


def _cfg(tmp_path, encoder, ft=FT):  # noqa: ANN001
    return study_cfg(tmp_path, encoder, finetune=ft)


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
    cfg = _cfg(tmp_path, tiny_encoder)
    tr, va, yv = _views()
    r2 = NeuralRun(cfg, "B2-S", 42, tiny_encoder, tmp_path / "b2", tok).fit_predict(tr, va, yv, "salt")
    r3 = NeuralRun(cfg, "B3-S", 42, tiny_encoder, tmp_path / "b3", tok).fit_predict(tr, va, yv, "salt")
    assert r2["encoder_init_state_sha256"] == r3["encoder_init_state_sha256"]
    assert r2["head_init_state_sha256"] == r3["head_init_state_sha256"]
    assert r2["frozen_encoder"] and not r3["frozen_encoder"]
    assert len(r2["scores"]) == len(va) and np.isfinite(r3["scores"]).all()


def test_at28_frozen_encoder_unchanged_and_b3_updates(tmp_path, tok, tiny_encoder):
    cfg = _cfg(tmp_path, tiny_encoder)
    tr, va, yv = _views()
    NeuralRun(cfg, "B3-S", 42, tiny_encoder, tmp_path / "b3", tok).fit_predict(tr, va, yv, "salt")
    from safetensors.torch import load_file
    enc0, _ = load_encoder(tiny_encoder, REV)
    tuned = load_file(str(resolve_generation(tmp_path / "b3" / "checkpoints", "best") / "encoder.safetensors"))
    assert any(not torch.equal(enc0.state_dict()[k], tuned[k]) for k in tuned)
    r2 = NeuralRun(cfg, "B2-S", 42, tiny_encoder, tmp_path / "b2", tok).fit_predict(tr, va, yv, "salt")
    assert r2["embedding_cache"]["encoder_state_sha256"] == state_hash(enc0)
    assert not (resolve_generation(tmp_path / "b2" / "checkpoints", "best") / "encoder.safetensors").exists()


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
def test_at14_resume_continues_from_last_epoch(tmp_path, tok, tiny_encoder, device, monkeypatch):
    tr, va, yv = _views()
    ft = {**FT, "device": device}
    one = _cfg(tmp_path, tiny_encoder, {**ft, "max_epochs": 1})
    NeuralRun(one, "B3-S", 7, tiny_encoder, tmp_path / "run", tok).fit_predict(tr, va, yv, "salt")
    two = _cfg(tmp_path, tiny_encoder, {**ft, "max_epochs": 2})
    seen = []
    real_load = torch.load
    monkeypatch.setattr(torch, "load", lambda *a, **k: seen.append(k.get("map_location")) or real_load(*a, **k))
    r = NeuralRun(two, "B3-S", 7, tiny_encoder, tmp_path / "run", tok).fit_predict(tr, va, yv, "salt")
    assert seen == ["cpu"] and [h["epoch"] for h in r["history"]] == [1, 2]
    fresh = NeuralRun(two, "B3-S", 7, tiny_encoder, tmp_path / "fresh", tok).fit_predict(tr, va, yv, "salt")
    tol = 1e-5 if device == "cpu" else 1e-3
    assert r["history"][1]["train_loss"] == pytest.approx(fresh["history"][1]["train_loss"], rel=tol)


def test_v3_split_learning_rates_and_b2_has_no_encoder_group(tmp_path, tok, tiny_encoder):
    ft = {k: v for k, v in FT.items() if k != "learning_rate"} | {"encoder_learning_rate": 1e-5, "head_learning_rate": 1e-3}
    cfg = _cfg(tmp_path, tiny_encoder, ft)
    run3 = NeuralRun(cfg, "B3-S", 1, tiny_encoder, tmp_path / "x", tok)
    enc, _ = run3._load_encoder()
    from diff_lab.neural import build_head
    head = build_head(16, 14, 0.1, 1)
    opt, params = run3._optimizer(enc, head)
    assert [(g["name"], g["lr"]) for g in opt.param_groups] == [("encoder", 1e-5), ("head", 1e-3)]
    opt2, _ = NeuralRun(cfg, "B2-S", 1, tiny_encoder, tmp_path / "y", tok)._optimizer(enc, head)
    assert [(g["name"], g["lr"]) for g in opt2.param_groups] == [("head", 1e-3)]
    tr, va, yv = _views()
    r = NeuralRun(cfg, "B3-S", 1, tiny_encoder, tmp_path / "z", tok).fit_predict(tr, va, yv, "salt")
    assert r["learning_rates"] == {"encoder": 1e-5, "head": 1e-3}
    assert r["token_accounting"]["optimizer_updates"] == 2 * 3  # 24 rows / window 8, 2 epochs


def test_finetune_lr_form_is_exclusive(tmp_path, tiny_encoder):
    bad = {**FT, "encoder_learning_rate": 1e-5}
    with pytest.raises(Exception, match="learning_rate"):
        _cfg(tmp_path, tiny_encoder, bad)


def test_corrupted_checkpoint_generation_is_refused(tmp_path, tok, tiny_encoder):
    tr, va, yv = _views()
    NeuralRun(_cfg(tmp_path, tiny_encoder, {**FT, "max_epochs": 1}), "B3-S", 3, tiny_encoder, tmp_path / "run", tok).fit_predict(
        tr, va, yv, "salt")
    last = resolve_generation(tmp_path / "run" / "checkpoints", "last")
    (last / "trainer.json").write_text("{}")
    with pytest.raises(Exception, match="corrupted"):
        NeuralRun(_cfg(tmp_path, tiny_encoder), "B3-S", 3, tiny_encoder, tmp_path / "run", tok).fit_predict(tr, va, yv, "salt")


def test_weights_pin_mismatch_is_refused(tmp_path, tok, tiny_encoder):
    import yaml
    from conftest import STUDY_V3

    from diff_lab.config import load_study
    raw = yaml.safe_load(STUDY_V3.read_text())
    raw["finetune"] = FT
    raw["model"] = {**raw["model"], "weights_file": "model.safetensors", "weights_sha256": "0" * 64}
    p = tmp_path / "s.yaml"
    p.write_text(yaml.safe_dump(raw))
    tr, va, yv = _views()
    with pytest.raises(Exception, match="weights_sha256"):
        NeuralRun(load_study(p)[0], "B3-S", 1, tiny_encoder, tmp_path / "r", tok).fit_predict(tr, va, yv, "salt")


def test_cuda_request_without_cuda_fails(tmp_path, tok, tiny_encoder):
    if torch.cuda.is_available():
        pytest.skip("CUDA present")
    cfg = _cfg(tmp_path, tiny_encoder, {**FT, "device": "cuda"})
    tr, va, yv = _views()
    with pytest.raises(Exception, match="no silent CPU fallback"):
        NeuralRun(cfg, "B3-S", 1, tiny_encoder, tmp_path / "x", tok).fit_predict(tr, va, yv, "salt")
