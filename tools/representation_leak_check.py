"""Does the text of a snapshot cover a label-dependent share of the commit? (exp 015 finding)

Needs only a diff-lab snapshot directory (edits, features, labels, splits): no git, no model.
coverage = (text lines + 1) / (la + ld + 1). If low coverage predicts the label far better than
the base rate, the line sets were built differently for buggy and clean commits, and any text
model can read the label from how much of the commit the text contains.

usage: python tools/representation_leak_check.py <snapshot_dir> [<snapshot_dir> ...]
"""
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def check(snap: Path) -> pd.DataFrame:
    n = pd.read_parquet(snap / "edits.parquet").groupby("change_id").size().rename("lines").reset_index()
    d = (pd.read_parquet(snap / "splits.parquet")[["change_id", "split"]]
         .merge(pd.read_parquet(snap / "labels.parquet")[["change_id", "label"]], on="change_id")
         .merge(pd.read_parquet(snap / "features.parquet")[["change_id", "la", "ld"]], on="change_id")
         .merge(n, on="change_id", how="left").fillna({"lines": 0}))
    d["coverage"] = (d["lines"] + 1) / (d["la"] + d["ld"] + 1)
    rows = []
    for s, g in d.groupby("split"):
        rows.append({"split": s, "n": len(g), "base_rate": g["label"].mean(),
                     "ap_low_coverage": average_precision_score(g["label"], -g["coverage"]),
                     "auc_low_coverage": roc_auc_score(g["label"], -g["coverage"]),
                     "median_coverage_buggy": g.loc[g["label"] == 1, "coverage"].median(),
                     "median_coverage_clean": g.loc[g["label"] == 0, "coverage"].median()})
    return pd.DataFrame(rows).round(3)


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(p)
        print(check(Path(p)).to_string(index=False))
