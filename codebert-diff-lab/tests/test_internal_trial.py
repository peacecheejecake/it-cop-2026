"""Internal trial path: git extraction in the public representation, subset export, score-blind sheet, label evaluation."""
import subprocess

import numpy as np
import pandas as pd
import pytest
from test_offline import bundle  # noqa: F401  (fixture: frozen synthetic study, unpacked bundle)
from test_study import study  # noqa: F401

from diff_lab.config import FEATURE_PROFILE_JIT14
from diff_lab.gitextract import clean_line, clean_message, extract
from diff_lab.label_eval import evaluate_labels, label_sheet
from diff_lab.offline import predict, unpack_bundle
from diff_lab.study import export_bundle
from diff_lab.util import ConfigError, IntegrityError, PolicyError, read_json


def test_line_and_message_rules_match_public_package():
    assert clean_line("    if (a != null && b_c >= 1.5) { // why", ".java") == "if ( a ! = null & & b c > = 1 . 5 ) {"
    assert clean_line('  String s = "http://x//y"; // c', ".java") == 'String s = "http : / / x / / y" ;'
    assert clean_line("   * javadoc line", ".java") == ""
    assert clean_line("  // only a comment", ".java") == ""
    assert clean_line("x = 1  # note", ".py") == "x = 1"
    assert clean_line("a.b()", ".xyz") == "a . b ( )"
    assert clean_message("Fix NPE (IVY-20)\n\ngit-svn-id: https://svn/x@1 abc") == "Fix NPE ( IVY - 20 )"


def _git(repo, *args, env=None):  # noqa: ANN001, ANN002, ANN202
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, env=env)


def _commit(repo, files, msg, author, when):  # noqa: ANN001, ANN202
    import os
    for path, text in files.items():
        p = repo / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    env = {**os.environ, "GIT_AUTHOR_NAME": author, "GIT_AUTHOR_EMAIL": f"{author}@x", "GIT_AUTHOR_DATE": f"{when} +0000",
           "GIT_COMMITTER_NAME": author, "GIT_COMMITTER_EMAIL": f"{author}@x", "GIT_COMMITTER_DATE": f"{when} +0000"}
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", msg, env=env)


@pytest.fixture()
def repo(tmp_path):
    r = tmp_path / "svc"
    r.mkdir()
    _git(r, "init", "-q")
    day = 86400
    _commit(r, {"core/A.java": "class A {\n  int x = 1;\n}\n", "README.md": "hi\n"}, "initial", "ann", 1_600_000_000)
    _commit(r, {"core/A.java": "class A {\n  int x = 2; // tweak\n  int y;\n}\n"}, "change x", "bob", 1_600_000_000 + 2 * day)
    _commit(r, {"docs/notes.md": "only docs\n"}, "docs", "bob", 1_600_000_000 + 3 * day)
    _commit(r, {"core/A.java": "class A {\n  int x = 2; // tweak\n  int y;\n  // comment only\n}\n",
                "web/b.py": "def f():\n    return 1  # one\n"}, "fix bug in A", "ann", 1_600_000_000 + 4 * day)
    return r


def test_extract_features_lines_and_skips(repo, tmp_path):
    res = extract([(repo, "svc")], tmp_path / "ex", "HEAD", None, None, None, (".java", ".py"), "name")
    assert res["rows"] == 3
    rep = res["repos"][0]
    assert rep["skipped"] == {"merge_or_no_code_file": 1}
    df = pd.read_parquet(tmp_path / "ex" / "dataset.parquet").set_index("change_id")
    meta = pd.read_parquet(tmp_path / "ex" / "meta.parquet").set_index("change_id")
    assert not ({"label", "bug", "defect"} & set(df.columns))
    assert list(df.columns[-14:]) == FEATURE_PROFILE_JIT14
    second = meta.index[meta["subject"] == "change x"][0]
    r = df.loc[second]
    assert list(r["added_lines"]) == ["int x = 2 ;", "int y ;"] and list(r["deleted_lines"]) == ["int x = 1 ;"]
    assert (r["nf"], r["la"], r["ld"], r["lt"], r["ndev"], r["nuc"], r["exp"], r["fix"]) == (1, 2, 1, 3, 1, 1, 0, 0)
    assert r["age"] == pytest.approx(2.0)
    last = df.loc[meta.index[meta["subject"] == "fix bug in A"][0]]
    # the comment-only line is dropped; the .py file joins; author ann has one earlier code commit
    assert list(last["added_lines"]) == ["def f ( ) :", "return 1"] and last["fix"] == 1 and last["exp"] == 1
    assert (last["nf"], last["ns"], last["nd"]) == (2, 2, 2) and last["entropy"] == pytest.approx(0.9183, abs=1e-4)  # 1 vs 2 changed lines
    assert meta.loc[meta["subject"] == "fix bug in A", "languages"].iloc[0] == "java,python"
    with pytest.raises(PolicyError, match="exists"):
        extract([(repo, "svc")], tmp_path / "ex", "HEAD", None, None, None, (".java",), "name")


def test_subset_export_and_label_eval(bundle, repo, tmp_path):  # noqa: F811
    cfg, arts, tmp, full = bundle
    import json
    raw = json.loads((full / "study.json").read_text())
    fz = tmp / "freeze" / "freeze.json"
    runs = read_json(fz)["runs"]
    keep = runs[0]
    b = export_bundle(cfg, raw, arts, fz, tmp_path / "sub" / "bundle.tar.gz", variants=(keep["variant_id"],), seeds=(keep["seed"],))
    with pytest.raises(ConfigError, match="no frozen run"):
        export_bundle(cfg, raw, arts, fz, tmp_path / "none" / "bundle.tar.gz", variants=("B3-S",))
    sub = tmp_path / "sub-unpacked"
    info = unpack_bundle(b["bundle"], sub)
    assert info["manifest"]["exported_runs"] == [keep["run_id"]] and len(info["freeze"]["runs"]) == len(runs)

    ex = tmp_path / "ex"
    extract([(repo, "svc")], ex, "HEAD", None, None, None, (".java", ".py"), "name")
    res = predict(sub, ex / "dataset.parquet", tmp_path / "pred", "cpu", None, None, None)
    assert res["ok"] == 1 and res["unavailable"] == 0

    info = label_sheet(ex, tmp_path / "sheet.csv", 2, "s")
    sheet = pd.read_csv(tmp_path / "sheet.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    assert info["rows"] == 2 and "score" not in sheet.columns and (sheet["label"] == "").all()
    full_sheet = label_sheet(ex, tmp_path / "all.csv", None, "s")
    assert full_sheet["rows"] == 3

    lab = pd.read_csv(tmp_path / "all.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    lab["label"] = ["1", "0", ""]
    lab.to_csv(tmp_path / "labels.csv", index=False)
    out = evaluate_labels(tmp_path / "pred" / "predictions.parquet", tmp_path / "labels.csv", ex / "meta.parquet", sub,
                          tmp_path / "eval", n_boot=50, min_group=1)
    r = out[keep["run_id"]]
    assert r["n"] == 2 and r["positives"] == 1 and r["ap"] is not None
    rep = read_json(tmp_path / "eval" / "label-eval.json")
    assert rep["labels"]["unlabeled"] == 1 and rep["runs"][keep["run_id"]]["threshold"] == keep["threshold"]
    scored = pd.read_csv(tmp_path / "eval" / "scored-labeled.csv", encoding="utf-8-sig")
    assert len(scored) == 2 and np.isfinite(scored["score"]).all()

    lab["label"] = ["1", "maybe", ""]
    lab.to_csv(tmp_path / "bad.csv", index=False)
    with pytest.raises(ConfigError, match="maybe"):
        evaluate_labels(tmp_path / "pred" / "predictions.parquet", tmp_path / "bad.csv", ex / "meta.parquet", sub, tmp_path / "e2")
    pd.DataFrame({"change_id": ["other:1"], "label": ["1"]}).to_csv(tmp_path / "unknown.csv", index=False)
    with pytest.raises(IntegrityError, match="no prediction"):
        evaluate_labels(tmp_path / "pred" / "predictions.parquet", tmp_path / "unknown.csv", ex / "meta.parquet", sub, tmp_path / "e3")
