"""Pre-registered sensitivity analysis on public test predictions (see RUNLOG "데이터 점검").

Recomputes riskbench.metrics.ranking_metrics on: full test, test without the
Hadoop-family projects (label-dependent project assignment in ApacheJIT), and test
without train near-duplicates (leak_check.py). Also reports a train-only
project-prior baseline and per-project AP (removes between-project base-rate signal).
Usage: python sensitivity.py <eval_dir> <splits_dir> <near_dup_ids.json> <out.json>
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

from sklearn.metrics import average_precision_score

from riskbench.metrics import ranking_metrics

HADOOP = {"apache/hadoop", "apache/hadoop-hdfs", "apache/hadoop-mapreduce"}
KEYS = ("ap", "roc_auc", "top5_recall", "top10_recall", "top10_precision")


def labels(split_dir):
    return {json.loads(l)["id"]: json.loads(l)["label"] for l in open(Path(split_dir) / "labels.jsonl")}


def main():
    ev, splits, dup_path, out = map(Path, sys.argv[1:5])
    y = labels(splits / "test"); train = labels(splits / "train")
    repo = {i: i.split(":")[1] for i in y}
    rate = defaultdict(lambda: [0, 0])
    for i, v in train.items():
        rate[i.split(":")[1]][0] += v; rate[i.split(":")[1]][1] += 1
    scores = {"project_prior": {i: rate[repo[i]][0] / rate[repo[i]][1] for i in y}}
    for m in ("rule", "tabular", "frozen", "finetune"):
        scores[m] = {r["id"]: r["score"] for r in map(json.loads, open(ev / f"{m}.jsonl"))}
    dups = set(json.loads(dup_path.read_text()))
    subsets = {"full": list(y), "no_hadoop_family": [i for i in y if repo[i] not in HADOOP],
               "no_near_duplicates": [i for i in y if i not in dups]}
    report = {"subsets": {}, "per_project_ap": {}}
    for name, ids in subsets.items():
        yy = [y[i] for i in ids]
        report["subsets"][name] = {"n": len(ids), "positives": sum(yy), "models": {
            m: {k: round(ranking_metrics(yy, [s[i] for i in ids])[k], 4) for k in KEYS} for m, s in scores.items()}}
    projects = sorted({repo[i] for i in y})
    for p in projects:
        ids = [i for i in y if repo[i] == p]; yy = [y[i] for i in ids]
        if 0 < sum(yy) < len(yy) and sum(yy) >= 20:
            report["per_project_ap"][p] = {"n": len(ids), "positives": sum(yy), "base_rate": round(sum(yy) / len(ids), 3),
                                           **{m: round(average_precision_score(yy, [s[i] for i in ids]), 4)
                                              for m, s in scores.items() if m != "project_prior"}}
    pp = report["per_project_ap"]
    report["per_project_mean_ap"] = {m: round(sum(v[m] for v in pp.values()) / len(pp), 4)
                                     for m in ("rule", "tabular", "frozen", "finetune")}
    report["per_project_mean_base_rate"] = round(sum(v["base_rate"] for v in pp.values()) / len(pp), 4)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
