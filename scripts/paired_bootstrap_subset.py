"""Week-cluster paired bootstrap of AP(model_b) - AP(model_a) on full test and on test
without the Hadoop-family projects (riskbench's evaluate only bootstraps the full set).
Clusters follow riskbench's default (ISO week of prediction_at), 500 reps, seed 42.
Usage: python paired_bootstrap_subset.py <eval_dir> <test_split_dir> <model_a> <model_b>
"""
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score as ap

HADOOP = {"apache/hadoop", "apache/hadoop-hdfs", "apache/hadoop-mapreduce"}


def main():
    ev, split, a, b = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
    y = {json.loads(l)["id"]: json.loads(l)["label"] for l in open(split / "labels.jsonl")}
    week = {}
    for l in open(split / "inputs.jsonl"):
        r = json.loads(l); week[r["id"]] = tuple(datetime.fromisoformat(r["prediction_at"]).isocalendar()[:2])
    s = {m: {r["id"]: r["score"] for r in map(json.loads, open(ev / f"{m}.jsonl"))} for m in (a, b)}
    rng = np.random.default_rng(42)
    for name, keep in (("full", lambda i: True), ("no_hadoop_family", lambda i: i.split(":")[1] not in HADOOP)):
        ids = [i for i in y if keep(i)]; clusters = defaultdict(list)
        for i in ids:
            clusters[week[i]].append(i)
        keys = list(clusters); deltas = []
        for _ in range(500):
            sample = [i for k in rng.choice(len(keys), len(keys)) for i in clusters[keys[k]]]
            yy = [y[i] for i in sample]
            if 0 < sum(yy) < len(yy):
                deltas.append(ap(yy, [s[b][i] for i in sample]) - ap(yy, [s[a][i] for i in sample]))
        yy = [y[i] for i in ids]
        point = ap(yy, [s[b][i] for i in ids]) - ap(yy, [s[a][i] for i in ids])
        print(f"{name}: AP({b})-AP({a}) = {point:+.4f}  95% week-cluster CI "
              f"[{np.percentile(deltas, 2.5):+.4f}, {np.percentile(deltas, 97.5):+.4f}]  clusters={len(keys)}")


if __name__ == "__main__":
    main()
