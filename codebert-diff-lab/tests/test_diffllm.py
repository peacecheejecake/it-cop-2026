"""diffllm-v1 pieces on tiny fixtures: diff cleaning/rendering, CPT corpus rules, LoRA CPT, risk SFT, embedding + MLP."""
import subprocess

import numpy as np
import pandas as pd
import pytest

from diff_lab.fulldiff import build_cpt_corpus, clean_diff, render_prompt, split_hunks

DIFF = """diff --git a/A.java b/A.java
index 1..2 100644
--- a/A.java
+++ b/A.java
@@ -1,3 +1,4 @@
 class A {
+  int x = 1;
 }
@@ -10,2 +11,3 @@
 void f() {
+  g();
 }
diff --git a/logo.png b/logo.png
Binary files a/logo.png and b/logo.png differ
diff --git a/package-lock.json b/package-lock.json
--- a/package-lock.json
+++ b/package-lock.json
@@ -1 +1 @@
-x
+y"""


def test_clean_diff_drops_binary_and_lock_sections():
    d, st = clean_diff(DIFF)
    assert st == {"files": 1, "dropped_binary": 1, "dropped_lock": 1, "hunks": 2}
    assert "logo.png" not in d and "package-lock" not in d and d.startswith("diff --git a/A.java")


def test_render_keeps_whole_hunks_within_budget(tok):
    d, _ = clean_diff(DIFF)
    units = split_hunks(d)
    assert len(units) == 2 and units[0].startswith("diff --git") and units[1].startswith("@@ -10")
    msg, diff, n, cut = render_prompt(tok, "Fix NPE in parser", d, budget=10_000)
    assert not cut and diff == d and msg == "Fix NPE in parser"
    msg, diff2, n2, cut2 = render_prompt(tok, "Fix NPE in parser", d, budget=len(tok(units[0])["input_ids"]) + 8)
    assert cut2 and diff2 == units[0] and n2 <= len(tok(units[0])["input_ids"]) + 8


def _repo(path, n):  # noqa: ANN001, ANN202
    path.mkdir()
    run = lambda *a: subprocess.run(["git", "-C", str(path), *a], check=True, capture_output=True)  # noqa: E731
    run("init", "-q")
    run("config", "user.email", "t@t")
    run("config", "user.name", "t")
    for i in range(n):
        (path / f"F{i % 3}.java").write_text(f"class F{i % 3} {{ int v = {i}; }}\n")
        run("add", ".")
        run("commit", "-q", "-m", f"change {i}")
    bare = path.parent / (path.name + ".git")
    subprocess.run(["git", "clone", "-q", "--bare", str(path), str(bare)], check=True)
    return bare


def test_cpt_corpus_budget_dedupe_and_exclusion(tmp_path, tok):
    pytest.importorskip("transformers")
    root = tmp_path / "git"
    root.mkdir()
    _repo(root / "r1", 12)
    from diff_lab.fulldiff import commit_diff
    from diff_lab.util import sha256_text
    shas = subprocess.run(["git", "-C", str(root / "r1.git"), "rev-list", "HEAD"], capture_output=True, text=True).stdout.split()
    keep = shas[len(shas) // 2]
    # Rank order depends on commit SHAs (new every run): exclude all but one so the outcome is order-independent.
    excluded = {sha256_text(commit_diff(root / "r1.git", s)[0]) for s in shas if s != keep}
    m = build_cpt_corpus(root, ["r1"], excluded, tok, budget=1, max_tokens=512, out=tmp_path / "cpt")
    df = pd.read_parquet(tmp_path / "cpt" / "corpus.parquet")
    assert set(df["sha"]) == {keep} and m["tokens"] >= 1
    assert m["skipped"]["excluded_jd4j"] + m["skipped"]["duplicate"] + m["skipped"]["empty"] + 1 <= len(shas)
    m2 = build_cpt_corpus(root, ["r1"], set(), tok, budget=200, max_tokens=512, out=tmp_path / "cpt-b")
    assert m2["tokens"] >= 200 and pd.read_parquet(tmp_path / "cpt-b" / "corpus.parquet")["diff_sha256"].is_unique
    with pytest.raises(Exception, match="only"):
        build_cpt_corpus(root, ["r1"], set(), tok, budget=10**9, max_tokens=512, out=tmp_path / "cpt2")


@pytest.fixture()
def tiny_qwen(tmp_path):
    torch = pytest.importorskip("torch")
    from conftest import ROOT
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from diff_lab.config import resolve_local_path
    src = resolve_local_path("cache:models/qwen2.5-coder-7b-instruct-tokenizer", ROOT)
    if not (src / "tokenizer.json").exists():
        pytest.skip("Qwen tokenizer snapshot not available")
    import shutil
    d = tmp_path / "tiny-qwen"
    d.mkdir()
    for f in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
        shutil.copy(src / f, d / f)
    torch.manual_seed(0)
    Qwen2ForCausalLM(Qwen2Config(vocab_size=151936, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                 num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096,
                                 tie_word_embeddings=True)).save_pretrained(d)
    return d


def _frames(n=48):  # noqa: ANN202
    rows = [{"change_id": f"c{i}", "message": f"fix issue {i}" if i % 2 else f"add feature {i}",
             "diff": (f"diff --git a/X.java b/X.java\n@@ -1 +1 @@\n-  a = {i};\n+  if (x == null) return {i};" if i % 2
                      else f"diff --git a/Y.java b/Y.java\n@@ -1 +1,2 @@\n+  log({i});"), "label": i % 2} for i in range(n)]
    df = pd.DataFrame(rows)
    return df.iloc[: n // 2].reset_index(drop=True), df.iloc[n // 2:].reset_index(drop=True)


def test_lora_cpt_then_risk_sft_and_embedding_mlp(tmp_path, tiny_qwen):
    import torch

    from diff_lab.llm_train import (
        CptLoraCfg,
        MlpCfg,
        SftCfg,
        cpt_lora,
        embed,
        encode_risk,
        label_ids_for,
        load_base,
        load_sft,
        mlp_predict,
        mlp_train,
        score,
        sft_risk,
    )
    targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    corpus = pd.DataFrame({"text": [f"Commit message:\nchange {i}\n\nDiff:\n+ int v = {i};" for i in range(200)]})
    corpus.to_parquet(tmp_path / "corpus.parquet")
    ccfg = CptLoraCfg(seq_len=256, micro_batch=2, accumulation=2, learning_rate=1e-3, warmup_fraction=0.1, lora_r=4,
                      lora_alpha=8, lora_dropout=0.0, target_modules=targets, checkpoint_every=2, gradient_checkpointing=False)
    res = cpt_lora(tiny_qwen, tmp_path / "corpus.parquet", tmp_path / "cpt", ccfg, 42, "cpu")
    assert res["updates"] >= 1 and (tmp_path / "cpt" / "adapter" / "adapter_config.json").exists()
    tr, va = _frames()
    scfg = SftCfg(max_prompt_tokens=512, micro_batch=4, accumulation=2, learning_rate=1e-3, max_epochs=2, evals_per_epoch=2,
                  patience_evals=3, selection_subset=100, lora_r=4, lora_alpha=8, lora_dropout=0.0, target_modules=targets,
                  gradient_checkpointing=False, eval_token_budget=4096)
    out = sft_risk(tiny_qwen, tmp_path / "cpt" / "adapter", tr, va, tmp_path / "sft", scfg, 42, "cpu", "salt")
    assert out["best"]["step"] >= 1 and out["history"]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(tiny_qwen))
    lids = label_ids_for(tok)
    m = load_sft(tiny_qwen, tmp_path / "cpt" / "adapter", tmp_path / "sft", torch.device("cpu"))
    s = score(m, [encode_risk(tok, a, b, lids) for a, b in zip(va["message"], va["diff"], strict=True)], lids,
              tok.pad_token_id, torch.device("cpu"), 4096)
    assert s.shape == (len(va),) and ((s > 0) & (s < 1)).all()
    base = load_base(tiny_qwen, torch.device("cpu"))
    x_tr, x_va = embed(base, tok, tr, torch.device("cpu"), 4096), embed(base, tok, va, torch.device("cpu"), 4096)
    single = embed(base, tok, tr.iloc[:1], torch.device("cpu"), 4096)
    assert np.allclose(single[0], x_tr[0], atol=1e-4)
    mcfg = MlpCfg(hidden=16, dropout=0.1, learning_rate=1e-2, max_epochs=10, patience=3, batch_size=8, pooling="max")
    state, info = mlp_train(x_tr, tr["label"].to_numpy(), x_va, va["label"].to_numpy(), va["change_id"].tolist(), mcfg, 42, "salt")
    p = mlp_predict(state, x_va, mcfg)
    assert p.shape == (len(va),) and info["best_epoch"] >= 1


def test_diffllm_orchestration_freeze_and_single_test(tmp_path, tiny_qwen):
    import json

    import yaml
    from conftest import ROOT

    from diff_lab.diffllm import freeze, load_cfg, run_arm, run_cpt, test_arms_once
    from diff_lab.util import PolicyError, atomic_write_json, sha256_file
    raw = yaml.safe_load((ROOT / "configs" / "studies" / "diffllm-v1.yaml").read_text())
    files = {p.name: sha256_file(p) for p in sorted(tiny_qwen.iterdir()) if p.is_file()}
    raw["models"]["qwen7b"] = {**raw["models"]["qwen7b"], "local_path": str(tiny_qwen), "files_sha256": files}
    raw["device"] = "cpu"
    raw["cpt"] = {**raw["cpt"], "seq_len": 256, "micro_batch": 2, "accumulation": 2, "lora_r": 4, "lora_alpha": 8,
                  "gradient_checkpointing": False, "checkpoint_every": 2}
    raw["sft"] = {**raw["sft"], "max_prompt_tokens": 512, "micro_batch": 4, "accumulation": 2, "lora_r": 4, "lora_alpha": 8,
                  "gradient_checkpointing": False, "selection_subset": 100, "eval_token_budget": 4096}
    raw["mlp"] = {**raw["mlp"], "hidden": 16, "max_epochs": 5, "batch_size": 8}
    raw["embed_token_budget"] = 4096
    cfgp = tmp_path / "cfg.yaml"
    cfgp.write_text(yaml.safe_dump(raw))
    cfg, h = load_cfg(cfgp)
    data = tmp_path / "data"
    corpus = data / "cpt" / cfg["cpt_corpus"]["id"]
    corpus.mkdir(parents=True)
    pd.DataFrame({"text": [f"Commit message:\nchange {i}\n\nDiff:\n+ int v = {i};" for i in range(120)]}).to_parquet(
        corpus / "corpus.parquet")
    atomic_write_json(corpus / "manifest.json", {"corpus_parquet_sha256": sha256_file(corpus / "corpus.parquet")})
    tr, va = _frames(48)
    te = va.copy()
    te["change_id"] = [f"jitd4j:p{i % 3}:t{i}" for i in range(len(te))]
    view = tmp_path / "view"
    view.mkdir()
    v = pd.concat([tr.assign(split="train"), va.assign(split="valid"), te.assign(split="test")])
    v.drop(columns=["label"]).to_parquet(view / "view.parquet")
    pd.concat([tr, va])[["change_id", "label"]].to_parquet(view / "labels-train-valid.parquet")
    atomic_write_json(view / "manifest.json", {"view_sha256": sha256_file(view / "view.parquet"),
                                               "labels_sha256": sha256_file(view / "labels-train-valid.parquet")})
    snap = tmp_path / "snap"
    snap.mkdir()
    pd.DataFrame({"change_id": te["change_id"], "label": te["label"]}).to_parquet(snap / "labels.parquet")
    art = tmp_path / "art"
    run_cpt(cfg, h, data, art)
    for arm in ("R-base", "E-base", "R-diff", "E-diff"):
        assert run_arm(cfg, h, arm, view, art)["status"] == "completed"
    fz = tmp_path / "freeze.json"
    with pytest.raises(PolicyError, match="sealed"):
        test_arms_once(cfg, h, view, snap, art, fz, tmp_path / "test")
    rec = freeze(cfg, h, view, art, fz, check_rows=8)
    assert [e["arm"] for e in rec["arms"]] == ["R-base", "E-base", "R-diff", "E-diff"]
    res = test_arms_once(cfg, h, view, snap, art, fz, tmp_path / "test")
    assert set(res) == {"R-base", "E-base", "R-diff", "E-diff"}
    with pytest.raises(PolicyError, match="never overwritten"):
        test_arms_once(cfg, h, view, snap, art, fz, tmp_path / "test")
    assert json.loads((tmp_path / "test" / "R-diff" / "metrics.json").read_text())["evaluation_role"] == "frozen_final_test"


def test_right_padded_training_logits_match_unpadded(tiny_qwen):
    import torch

    from diff_lab.llm_train import last_token_logits, load_base
    m = load_base(tiny_qwen, torch.device("cpu")).float().eval()
    rows = [[1, 2, 3, 4, 5, 6], [7, 8, 9], [10, 11, 12, 13]]
    batched = last_token_logits(m, rows, 0, torch.device("cpu"))
    for i, r in enumerate(rows):
        single = m(input_ids=torch.tensor([r])).logits[0, -1].float()
        assert torch.allclose(batched[i], single, atol=1e-4)
