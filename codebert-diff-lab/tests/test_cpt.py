"""MLM CPT contract tests (T16/T17, AT-03/09/10/11) on a tiny random RoBERTa with the real CodeBERT tokenizer."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

torch = pytest.importorskip("torch")
from transformers import RobertaConfig, RobertaModel  # noqa: E402

import diff_lab.cpt as cpt_mod  # noqa: E402
from diff_lab.config import FEATURE_PROFILE_JIT14, load_study, resolve_local_path  # noqa: E402
from diff_lab.cpt import (  # noqa: E402
    CptCorpusView,
    MlmCpt,
    load_cpt_encoder,
    lr_at,
    mask_sample,
    plan_windows,
    replacement_candidates,
    structure_chars,
    tokenize_corpus,
)
from diff_lab.evidence import load_tokenizer  # noqa: E402
from diff_lab.neural import NeuralRun, load_encoder, state_hash  # noqa: E402
from diff_lab.policy import QueryView, TrainingDatasetView  # noqa: E402
from diff_lab.util import PolicyError  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "configs" / "studies" / "public-comparison-v2.yaml"
REV = "3b0952feddeffad0063f274080e3c23d75e7eb39"
FT = {"max_epochs": 1, "micro_batch_size": 4, "gradient_accumulation_steps": 2, "learning_rate": 1e-3, "weight_decay": 0.01,
      "lr_schedule": "constant", "max_grad_norm": 1.0, "head_dropout": 0.1, "selection_metric": "validation_ap",
      "patience_epochs": 5, "tie_rule": "earlier_checkpoint", "precision": "fp32", "device": "cpu", "allow_cpu": True,
      "eval_batch_size": 8}
CPT = {"permitted_role": "cpt_train", "corpus": "evidence_query_text", "budget_unit": "nonpadding_input_token_exposures",
       "budget": 300, "micro_batch_size": 2, "gradient_accumulation_steps": 2, "learning_rate": 1e-3, "weight_decay": 0.01,
       "lr_schedule": "linear_warmup_linear_decay", "warmup_fraction": 0.2, "max_grad_norm": 1.0, "mlm_probability": 0.15,
       "mask_replace_fraction": 0.8, "random_replace_fraction": 0.1, "protect_renderer_structure": True,
       "zero_selection_rule": "force_one_uniform", "b5_task_schedule": "alternating_1_to_1", "rmi_replacement_probability": 0.5,
       "dev_eval_every_updates": 2, "dev_mask_seed": 7, "checkpoint_every_updates": 2, "precision": "fp32", "device": "cpu",
       "allow_cpu": True, "eval_batch_size": 4}
PUBLIC = {"visibility": "public"}


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


def _cfg(tmp_path, cpt=CPT, ft=FT):  # noqa: ANN001
    raw = yaml.safe_load(STUDY.read_text())
    raw["cpt"], raw["finetune"] = cpt, ft
    p = tmp_path / "study.yaml"
    p.write_text(yaml.safe_dump(raw, sort_keys=False))
    return load_study(p)[0]


def _row(cid: str, msg: str, adds: list[str], dels: list[str], split="train", role="cpt_train") -> dict:  # noqa: ANN001
    code = "\n".join([f"+ {a}" for a in adds] + [f"- {d}" for d in dels])
    return {"change_id": cid, "split": split, "cpt_role": role, "message_text": msg, "code_text": code,
            "query_text": (msg + "\n" + code) if code else msg}


def _corpus(n: int, role: str = "cpt_train") -> CptCorpusView:
    rows = [_row(f"{role}{i}", f"fix issue {i}", [f"if (x{i} == null) return -1;"], ["y = a - b;"] if i % 2 else [], role=role)
            for i in range(n)]
    return CptCorpusView(role, pd.DataFrame(rows), PUBLIC)


def test_at11_structure_protected_but_code_operators_eligible(tok):
    view = CptCorpusView("cpt_train", pd.DataFrame([_row("c", "Fix NPE", ["if (x == null) return -1;"], ["x.foo();"])]), PUBLIC)
    t = tokenize_corpus(tok, view, 512)
    toks = tok.convert_ids_to_tokens(t.ids[0].tolist())
    elig = dict(zip(range(len(toks)), t.eligible[0], strict=True))
    assert not elig[0] and not elig[len(toks) - 1]
    newline = [i for i, s in enumerate(toks) if s == "Ċ"]
    prefixes = [i + 1 for i in newline]
    assert newline and all(not elig[i] for i in newline + prefixes)
    assert {toks[i] for i in prefixes} == {"+", "-"}
    minus_op = [i for i, s in enumerate(toks) if s == "Ġ-"]
    assert minus_op and all(elig[i] for i in minus_op)


def test_structure_chars_rejects_unprefixed_code():
    with pytest.raises(Exception, match="prefix"):
        structure_chars("m", "x = 1")


def test_at11_mask_rates_specials_and_replacements(tok):
    rng = np.random.default_rng(0)
    cand = replacement_candidates(tok)
    assert not set(tok.all_special_ids) & set(cand.tolist())
    ids = rng.integers(10, 50000, 200_000)
    elig = np.ones(len(ids), dtype=bool)
    elig[::10] = False
    out, lab, forced = mask_sample(ids, elig, rng, 0.15, 0.8, 0.1, tok.mask_token_id, cand)
    sel = lab != -100
    assert not forced and not sel[~elig].any()
    assert sel.sum() / elig.sum() == pytest.approx(0.15, abs=0.005)
    masked = (out[sel] == tok.mask_token_id).mean()
    kept = (out[sel] == ids[sel]).mean()
    assert masked == pytest.approx(0.8, abs=0.01) and kept == pytest.approx(0.1 + 0.1 / len(cand), abs=0.01)
    assert not np.isin(out[sel], [i for i in tok.all_special_ids if i != tok.mask_token_id]).any()


def test_at11_zero_draw_forces_one_target_and_no_eligible_is_excluded(tok):
    ids = np.array([0, 500, 600, 2])
    elig = np.array([False, False, True, False])
    _, lab, forced = mask_sample(ids, elig, np.random.default_rng(0), 1e-9, 0.8, 0.1, tok.mask_token_id, replacement_candidates(tok))
    assert forced and (lab != -100).tolist() == [False, False, True, False]
    view = CptCorpusView("cpt_train", pd.DataFrame([_row("empty", "", [], []), _row("ok", "fix", [], [])]), PUBLIC)
    t = tokenize_corpus(tok, view, 512)
    assert t.change_ids == ["ok"] and t.excluded == {"empty": "no_eligible_tokens"}


def test_at03_cpt_corpus_rejects_eval_ids_and_labels():
    bad = pd.DataFrame([_row("v1", "m", ["a"], [], split="valid", role=None)])
    with pytest.raises(PolicyError, match="outside public train"):
        CptCorpusView("cpt_train", bad, PUBLIC)
    dev_row = pd.DataFrame([_row("d1", "m", ["a"], [], role="cpt_dev")])
    with pytest.raises(PolicyError):
        CptCorpusView("cpt_train", dev_row, PUBLIC)
    with pytest.raises(PolicyError, match="label"):
        CptCorpusView("cpt_train", pd.DataFrame([_row("t", "m", ["a"], [])]).assign(label=1), PUBLIC)
    with pytest.raises(PolicyError, match="public"):
        CptCorpusView("cpt_train", pd.DataFrame([_row("t", "m", ["a"], [])]), {"visibility": "internal"})


def test_plan_is_deterministic_minimal_and_meets_budget():
    lengths = np.random.default_rng(3).integers(5, 50, 37)
    a, b = plan_windows(lengths, 42, 8, 1000), plan_windows(lengths, 42, 8, 1000)
    assert a["plan_sha256"] == b["plan_sha256"] and a["planned_tokens"] >= 1000
    assert a["planned_tokens"] - int(lengths[a["windows"][-1]].sum()) < 1000
    assert plan_windows(lengths, 43, 8, 1000)["plan_sha256"] != a["plan_sha256"]


def test_lr_schedule_warmup_then_decay_stays_positive():
    lrs = [lr_at(u, 10, 3, 1.0) for u in range(10)]
    assert lrs[:3] == pytest.approx([1 / 3, 2 / 3, 1.0]) and all(x > 0 for x in lrs)
    assert all(lrs[i] >= lrs[i + 1] for i in range(2, 9))


def test_at10_cpt_exports_encoder_only_and_b4_starts_from_it(tmp_path, tok, tiny_encoder):
    cfg = _cfg(tmp_path)
    res = MlmCpt(cfg, 42, tiny_encoder, tmp_path / "cpt", tok).run(_corpus(24), _corpus(6, "cpt_dev"))
    acc = res["token_accounting"]
    assert acc["mlm_input_tokens"] >= CPT["budget"] and acc["optimizer_updates"] == acc["planned_updates"]
    assert acc["rmi_input_tokens"] == 0 and acc["mlm_target_tokens"] > 0
    assert res["lm_head_newly_initialized"] and all(np.isfinite(d["dev_mlm_loss"]) for d in res["dev"])
    from safetensors.torch import load_file
    keys = load_file(str(tmp_path / "cpt" / "encoder" / "encoder.safetensors")).keys()
    assert not any(k.startswith("lm_head") for k in keys)
    base, _ = load_encoder(tiny_encoder, REV)
    assert res["encoder_init_state_sha256"] == state_hash(base) != res["exported_encoder_state_sha256"]
    enc, info = load_cpt_encoder(tmp_path / "cpt" / "encoder", tiny_encoder, REV)
    assert state_hash(enc) == res["exported_encoder_state_sha256"] and info["cpt_export"]["task"] == "mlm"
    tr, va, yv = _views()
    r4 = NeuralRun(cfg, "B4-S", 42, tiny_encoder, tmp_path / "b4", tok, init_encoder=tmp_path / "cpt" / "encoder").fit_predict(
        tr, va, yv, "salt")
    r3 = NeuralRun(cfg, "B3-S", 42, tiny_encoder, tmp_path / "b3", tok).fit_predict(tr, va, yv, "salt")
    assert r4["head_init_state_sha256"] == r3["head_init_state_sha256"]
    assert r4["encoder_init_state_sha256"] == res["exported_encoder_state_sha256"] != r3["encoder_init_state_sha256"]
    with pytest.raises(Exception, match="CPT encoder export"):
        NeuralRun(cfg, "B4-S", 42, tiny_encoder, tmp_path / "x", tok)


class _Stop(Exception):
    pass


def test_cpt_resume_matches_uninterrupted(tmp_path, tok, tiny_encoder, monkeypatch):
    cfg = _cfg(tmp_path)
    full = MlmCpt(cfg, 5, tiny_encoder, tmp_path / "full", tok).run(_corpus(24), _corpus(6, "cpt_dev"))
    orig = cpt_mod.MlmCpt._checkpoint

    def crash(self, path, model, opt, st, plan_hash):  # noqa: ANN001, ANN202
        orig(self, path, model, opt, st, plan_hash)
        raise _Stop

    monkeypatch.setattr(cpt_mod.MlmCpt, "_checkpoint", crash)
    with pytest.raises(_Stop):
        MlmCpt(cfg, 5, tiny_encoder, tmp_path / "resumed", tok).run(_corpus(24), _corpus(6, "cpt_dev"))
    monkeypatch.setattr(cpt_mod.MlmCpt, "_checkpoint", orig)
    resumed = MlmCpt(cfg, 5, tiny_encoder, tmp_path / "resumed", tok).run(_corpus(24), _corpus(6, "cpt_dev"))
    assert resumed["token_accounting"]["mlm_input_tokens"] == full["token_accounting"]["mlm_input_tokens"]
    assert resumed["exported_encoder_state_sha256"] == full["exported_encoder_state_sha256"]
    assert [d["dev_mlm_loss"] for d in resumed["dev"]] == pytest.approx([d["dev_mlm_loss"] for d in full["dev"]], rel=1e-6)


def _views(n=16):  # noqa: ANN001
    rng = np.random.default_rng(1)
    f = pd.DataFrame(rng.integers(0, 20, (2 * n, len(FEATURE_PROFILE_JIT14))).astype(float), columns=FEATURE_PROFILE_JIT14)
    f["change_id"] = [f"c{i}" for i in range(2 * n)]
    f["query_text"] = [("fix null check\n+ if ( x == null )" if i % 2 else "update docs\n- old line") for i in range(2 * n)]
    y = pd.Series([i % 2 for i in range(2 * n)])
    tr = TrainingDatasetView(split="train", frame=f.iloc[:n].reset_index(drop=True), lineage=PUBLIC,
                             labels=y.iloc[:n].reset_index(drop=True))
    return tr, QueryView(split="valid", frame=f.iloc[n:].reset_index(drop=True), lineage={}), y.iloc[n:].to_numpy()
