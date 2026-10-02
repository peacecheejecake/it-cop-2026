"""Extractor fidelity on the public JD4J test split (already opened once in exp 011; this checks inputs, not models).

Builds four internal-format datasets for the test commits found in the cached git mirrors and scores
each with the exported B3-S bundle through the real `predict` path:
  orig       package lines/message + package jit14 (should reproduce the frozen test scores)
  git        extractor lines/message + extractor jit14 (what the internal trial will see)
  git_lines  extractor lines/message + package jit14
  git_feats  package lines/message + extractor jit14
Writes per-feature/line agreement and AP per input to <out>/validate.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from diff_lab.config import FEATURE_PROFILE_JIT14 as F
from diff_lab.gitextract import extract_repo
from diff_lab.offline import predict

E011 = Path(sys.argv[1])
MIRRORS, BUNDLE, CODEBERT, OUT = (Path(a) for a in sys.argv[2:6])
DEVICE = sys.argv[6]
snap = E011 / "data/snapshots/jitd4j-audit1"
splits = pd.read_parquet(E011 / "data/splits/jitd4j-audit1/upstream-clean2/splits.parquet")
c = pd.read_parquet(snap / "changes.parquet")
f = pd.read_parquet(snap / "features.parquet")
e = pd.read_parquet(snap / "edits.parquet")
lab = pd.read_parquet(snap / "labels.parquet")[["change_id", "label"]]
test = c.merge(splits[splits.split == "test"][["change_id"]], on="change_id")
OUT.mkdir(parents=True, exist_ok=False)

mine = []
for proj, g in test.groupby("project_id"):
    d, _, rep = extract_repo(MIRRORS / f"{proj}.git", proj, extensions=(".java",), only=set(g.commit_sha))
    mine.append(d)
    print(proj, rep["extracted"], "/", len(g), rep["skipped"], flush=True)
mine = pd.concat(mine, ignore_index=True)
mine["change_id"] = "jitd4j:" + mine["change_id"]
ids = sorted(set(mine.change_id) & set(test.change_id))
mine = mine.set_index("change_id").loc[ids]

lines = {(cid, op): sorted(gr.text) for (cid, op), gr in e[e.change_id.isin(ids)].groupby(["change_id", "op"])}
orig = pd.DataFrame({"change_id": ids})
orig["message"] = orig.change_id.map(c.set_index("change_id").message)
orig["added_lines"] = [lines.get((i, "add"), []) for i in ids]
orig["deleted_lines"] = [lines.get((i, "delete"), []) for i in ids]
orig = orig.merge(f[["change_id", *F]], on="change_id").set_index("change_id").loc[ids]

text = ["message", "added_lines", "deleted_lines"]
sets = {"orig": orig, "git": mine[text + F],
        "git_lines": pd.concat([mine[text], orig[F]], axis=1), "git_feats": pd.concat([orig[text], mine[F]], axis=1)}
y = pd.Series(ids).map(lab.set_index("change_id").label).to_numpy()
res = {"test_commits": int(len(test)), "found_in_mirror": len(ids), "positives": int(y.sum()), "inputs": {}}
for name, df in sets.items():
    p = OUT / f"{name}.parquet"
    df.reset_index().to_parquet(p, index=False)
    predict(BUNDLE, p, OUT / f"pred-{name}", DEVICE, ("B3-S",), CODEBERT, None)
    s = pd.read_parquet(OUT / f"pred-{name}" / "predictions.parquet").set_index("change_id").loc[ids, "score"].to_numpy()
    res["inputs"][name] = {"ap": average_precision_score(y, s), "roc_auc": roc_auc_score(y, s)}
    np.save(OUT / f"scores-{name}.npy", s)
    print(name, res["inputs"][name], flush=True)
base = np.load(OUT / "scores-orig.npy")
for name in sets:
    s = np.load(OUT / f"scores-{name}.npy")
    res["inputs"][name]["spearman_vs_orig"] = float(pd.Series(s).corr(pd.Series(base), method="spearman"))
    res["inputs"][name]["max_abs_diff_vs_orig"] = float(np.abs(s - base).max())
res["feature_exact_rate"] = {k: float(np.mean(np.isclose(orig[k], mine[k], atol=1e-3, rtol=1e-3))) for k in F}
res["feature_spearman"] = {k: float(orig[k].corr(mine[k], method="spearman")) for k in F}
jac = []
for i in ids:
    for col in ("added_lines", "deleted_lines"):
        a, b = set(orig.at[i, col]), set(mine.at[i, col])
        if a or b:
            jac.append(len(a & b) / len(a | b))
res["line_jaccard_mean"] = float(np.mean(jac))
res["message_exact_rate"] = float(np.mean(orig.message.to_numpy() == mine.message.to_numpy()))
(OUT / "validate.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
