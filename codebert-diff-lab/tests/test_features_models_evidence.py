from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from diff_lab.config import FEATURE_PROFILE_JIT14, load_study, resolve_local_path
from diff_lab.evidence import _allocate, load_tokenizer, render
from diff_lab.features import StructuredPipeline
from diff_lab.models import B0LR, B1TFIDF
from diff_lab.policy import QueryView, TrainingDatasetView

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "configs" / "studies" / "public-comparison-v2.yaml"


def _views(n: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    f = pd.DataFrame(rng.integers(0, 50, size=(2 * n, len(FEATURE_PROFILE_JIT14))).astype(float), columns=FEATURE_PROFILE_JIT14)
    f["change_id"] = [f"c{i}" for i in range(2 * n)]
    f["message_text"] = ["fix bug in parser" if i % 3 == 0 else "add feature" for i in range(2 * n)]
    f["code_text"] = ["+ int x = 1 ;" if i % 2 else "- return null ;" for i in range(2 * n)]
    y = pd.Series((f["la"] > 30).astype(int))
    tr = TrainingDatasetView(split="train", frame=f.iloc[:n].reset_index(drop=True), lineage={"visibility": "public"},
                             labels=y.iloc[:n].reset_index(drop=True))
    q = QueryView(split="valid", frame=f.iloc[n:].reset_index(drop=True), lineage={})
    return tr, q, y.iloc[n:].to_numpy()


def test_pipeline_statistics_come_from_train_only():
    tr, q, _ = _views()
    p = StructuredPipeline(FEATURE_PROFILE_JIT14).fit(tr)
    z = np.sign(tr.frame[FEATURE_PROFILE_JIT14].to_numpy()) * np.log1p(np.abs(tr.frame[FEATURE_PROFILE_JIT14].to_numpy()))
    assert np.allclose(p.mean, z.mean(axis=0))
    with pytest.raises(Exception, match="TrainingDatasetView"):
        StructuredPipeline(FEATURE_PROFILE_JIT14).fit(q)


def test_b0lr_and_b1_fit_predict_and_b1_stays_sparse():
    cfg, _, _ = load_study(STUDY)
    tr, q, _ = _views()
    s0 = B0LR(cfg, 42)
    s0.fit(tr)
    assert s0.predict(q).shape == (len(q),)
    b1 = B1TFIDF(cfg, 42)
    state = b1.fit(tr)
    assert sp.issparse(b1._x(q.frame)) and state["fit_role"] == "supervised_train"


def test_at26_tfidf_vocabulary_is_train_only():
    cfg, _, _ = load_study(STUDY)
    tr, q, _ = _views()
    q.frame.loc[:, "message_text"] = "zzqcanaryzzq " * 5
    b1 = B1TFIDF(cfg, 42)
    state = b1.fit(tr)
    assert not any("zzq" in t for t in state["tfidf"]["message_text"]["vocabulary"])


def test_allocation_keeps_both_ops_and_respects_budget():
    ka, kd = _allocate([10] * 50, [10] * 50, 100)
    assert ka > 0 and kd > 0 and 10 * (ka + kd) <= 100
    assert _allocate([5, 5], [5], 100) == (2, 1)


def test_at07_render_preserves_direction():
    text, code = render("msg", ["a"], ["b"])
    assert text == "msg\n+ a\n- b" and code == "+ a\n- b"
    assert render("msg", [], [])[0] == "msg"


def test_pinned_tokenizer_is_native():
    cfg, _, _ = load_study(STUDY)
    try:
        path = resolve_local_path(cfg.model.local_path, ROOT)
    except Exception:  # noqa: BLE001
        pytest.skip("shared cache not found (network-free CI)")
    if not path.exists():
        pytest.skip("pinned CodeBERT tokenizer snapshot not available (network-free CI)")
    tok, info = load_tokenizer(path, cfg.model.revision)
    assert set(info["added_tokens"]) <= set(tok.all_special_tokens)
