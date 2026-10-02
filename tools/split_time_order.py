"""Per-project time order of a diff-lab snapshot's provided split (author dates; committer dates are absent in JD4J).

Pooled date ranges overlap whenever projects cover different eras, so they cannot show whether a split is time-ordered.
Per project this reports P(test later than train/valid), P(valid later than train) (rank AUCs) and the share of test
changes dated before the 99th percentile of train/valid.
Usage: python tools/split_time_order.py <snapshot_dir> [--out result.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    ch = pd.read_parquet(a.snapshot / "changes.parquet")[["change_id", "project_id", "authored_at_unix"]]
    sp = pd.read_parquet(a.snapshot / "splits.parquet")[["change_id", "split"]]
    d = ch.merge(sp, on="change_id", validate="1:1")
    if d["authored_at_unix"].isna().any():
        raise SystemExit("changes without an author date")
    rows = []
    for p, g in d.groupby("project_id"):
        t, te, va = g["authored_at_unix"], g["split"] == "test", g["split"] == "valid"
        rows.append({"project": p, "n": len(g), "test_share": float(te.mean()), "valid_share": float(va.mean()),
                     "p_test_later_than_trainvalid": float(roc_auc_score(te, t)),
                     "p_valid_later_than_train": float(roc_auc_score(va[~te], t[~te])),
                     "test_before_p99_trainvalid": float((t[te] < t[~te].quantile(0.99)).mean())})
    r = pd.DataFrame(rows)
    print(r.round(3).to_string(index=False))
    med = r.drop(columns=["project", "n"]).median().round(4).to_dict()
    print("median:", med)
    if a.out:
        a.out.write_text(json.dumps({"snapshot": str(a.snapshot), "time_field": "authored_at_unix", "projects": rows, "median": med}, indent=1))


if __name__ == "__main__":
    main()
