"""DoD-Internal-Ready on synthetic fixtures: bundle integrity, label-free offline predict, network block, no zero fill,
frozen public demos only (AT-19/20/21/25/30/35). No real internal data is used."""
import io
import socket
import tarfile

import numpy as np
import pandas as pd
import pytest
from test_study import study  # noqa: F401  (fixture: synthetic snapshot + two B0-LR runs)

from diff_lab.config import FEATURE_PROFILE_JIT14, load_study
from diff_lab.offline import load_internal, offline_guard, predict, render_internal, unpack_bundle, verify_bundle
from diff_lab.study import export_bundle, freeze
from diff_lab.util import IntegrityError, PolicyError


def _internal(path, n=12, drop=None, label=False):  # noqa: ANN001, ANN202
    rng = np.random.default_rng(5)
    df = pd.DataFrame(np.abs(rng.normal(size=(n, len(FEATURE_PROFILE_JIT14)))) * 10, columns=FEATURE_PROFILE_JIT14)
    df.loc[0, "age"] = np.nan
    df["change_id"] = [f"internal:{i}" for i in range(n)]
    df["message"] = [f"사내 변경 {i}: fix null handling" for i in range(n)]
    df["added_lines"] = [["z = 1;", "if (x == null) return;"] if i % 2 else ["a = b - c;"] for i in range(n)]
    df["deleted_lines"] = [["old();"] if i % 3 else [] for i in range(n)]
    if drop:
        df = df.drop(columns=[drop])
    if label:
        df["label"] = 0
    df.to_parquet(path)
    return df


@pytest.fixture()
def bundle(study):  # noqa: F811
    cfg, raw, data, arts, tmp = study
    fz = tmp / "freeze" / "freeze.json"
    freeze(cfg, raw, data, arts, fz, "cpu")
    b = export_bundle(cfg, raw, arts, fz, tmp / "export" / "bundle.tar.gz")
    out = tmp / "unpacked"
    unpack_bundle(b["bundle"], out)
    return cfg, arts, tmp, out


def test_at20_bundle_verifies_and_detects_tampering(bundle):
    _, _, tmp, out = bundle
    assert verify_bundle(out)["freeze"]["status"] == "frozen"
    state = next(out.glob("runs/*/model/state.json"))
    state.write_text(state.read_text().replace('"intercept"', '"intercept" ', 1))
    with pytest.raises(IntegrityError, match="modified"):
        verify_bundle(out)


def test_unsafe_bundle_member_refused(tmp_path):
    p = tmp_path / "evil.tar.gz"
    with tarfile.open(p, "w:gz") as t:
        data = b"x"
        info = tarfile.TarInfo("../escape.txt")
        info.size = len(data)
        t.addfile(info, io.BytesIO(data))
    with pytest.raises(PolicyError, match="unsafe"):
        unpack_bundle(p, tmp_path / "out")


def test_offline_predict_is_label_free_and_matches_frozen_predictor(bundle):
    cfg, arts, tmp, out = bundle
    df = _internal(tmp / "internal.parquet")
    res = predict(out, tmp / "internal.parquet", tmp / "pred", "cpu", ("B0-LR",), None, None)
    assert res == {"freeze_id": res["freeze_id"], "rows": 12, "ok": 2, "unavailable": 0}
    man = pd.read_json(tmp / "pred" / "offline-manifest.json", typ="series")
    assert man["labels_read"] is False and man["fit_or_calibration"] == "none"
    pred = pd.read_parquet(tmp / "pred" / "predictions.parquet")
    from diff_lab.frozen import FrozenB0LR
    from diff_lab.policy import QueryView
    run = pred["run_id"].iloc[0]
    ref = FrozenB0LR(out / "runs" / run).predict(QueryView(split="internal", frame=df, lineage={}))
    assert np.allclose(pred.loc[pred.run_id == run, "score"].to_numpy(), ref)
    assert pred["score_semantics"].str.contains("uncalibrated").all()


def test_at19_labels_refused_and_at25_missing_feature_unavailable(bundle):
    _, _, tmp, out = bundle
    _internal(tmp / "with-label.parquet", label=True)
    with pytest.raises(PolicyError, match="labels"):
        predict(out, tmp / "with-label.parquet", tmp / "p1", "cpu", None, None, None)
    _internal(tmp / "no-lt.parquet", drop="lt")
    res = predict(out, tmp / "no-lt.parquet", tmp / "p2", "cpu", None, None, None)
    assert res["ok"] == 0 and res["unavailable"] == 2 and not (tmp / "p2" / "predictions.parquet").exists()


def test_at21_network_is_blocked_inside_offline_guard():
    with offline_guard():
        with pytest.raises(PolicyError, match="egress"):
            socket.create_connection(("example.com", 80), timeout=1)
        with pytest.raises(PolicyError, match="egress"):
            socket.socket().connect(("127.0.0.1", 9))
    s = socket.socket()
    s.close()


def test_internal_rendering_matches_public_renderer(tmp_path, tok):
    from conftest import STUDY_V3
    cfg = load_study(STUDY_V3)[0]
    raw = _internal(tmp_path / "i.parquet")
    df, cov = load_internal(tmp_path / "i.parquet")
    assert cov["feature_nan_counts"]["age"] == 1 and not cov["absent_feature_columns"]
    frame = render_internal(tok, df, cfg)
    row = frame.iloc[1]
    assert row["query_text"] == raw.loc[1, "message"] + "\n+ if (x == null) return;\n+ z = 1;\n- old();"
    assert (frame["encoder_tokens"] <= 512).all()


def test_at35_bundled_public_demos_only_and_at30_no_demos_for_l0(tmp_path, tok):
    pytest.importorskip("torch")
    from test_llm import _llm_cfg, _train  # tiny Qwen fixtures

    from diff_lab.frozen import FrozenLlm
    from diff_lab.llm import LlmRun, select_demos
    from diff_lab.policy import QueryView
    tiny_model = _tiny_qwen(tmp_path / "tiny")
    cfg = _llm_cfg(tmp_path, tiny_model)
    train = _train()
    run_dir = tmp_path / "run"
    LlmRun(cfg, "L1-S", 42, tiny_model, run_dir).fit_predict(train, QueryView(split="valid", frame=train.frame.iloc[:2], lineage={}))
    demos = select_demos(train, cfg.llm, 42)
    FrozenLlm(run_dir, cfg, "L1-S", 42, tiny_model, None, demos)
    altered = demos.copy()
    altered.loc[0, "query_text"] = "internal change content"
    with pytest.raises(IntegrityError, match="frozen public demo manifest"):
        FrozenLlm(run_dir, cfg, "L1-S", 42, tiny_model, None, altered)
    with pytest.raises(PolicyError, match="zero-shot"):
        LlmRun(cfg, "L0-S", 42, tiny_model, tmp_path / "l0").fit_predict(None, QueryView(split="v", frame=train.frame.iloc[:2],
                                                                                         lineage={}), demos)


def _tiny_qwen(d):  # noqa: ANN001, ANN202
    import shutil

    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from diff_lab.config import resolve_local_path
    src = resolve_local_path("cache:models/qwen2.5-coder-7b-instruct-tokenizer", __import__("conftest").ROOT)
    if not (src / "tokenizer.json").exists():
        pytest.skip("Qwen tokenizer snapshot not available")
    d.mkdir()
    for f in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
        shutil.copy(src / f, d / f)
    torch.manual_seed(0)
    Qwen2ForCausalLM(Qwen2Config(vocab_size=151936, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                                 num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=32768,
                                 tie_word_embeddings=True)).save_pretrained(d)
    return d
