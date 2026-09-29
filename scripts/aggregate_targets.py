"""Aggregate 005 v2: per evaluation target, mean ± sd across seeds of AP, AP/base-rate
lift, ROC-AUC, Recall@10%, per-project mean AP (projects with >= 20 positives), and a
week-cluster paired bootstrap of AP(finetune) - AP(tabular) per seed.
Usage: python aggregate_targets.py <runs_pod_dir> <out.json>
Targets map to local label splits: public-test -> ../../002-no-hadoop/.../public/test,
jd4j-* -> data/splits/jd4j/*."""
import json
import statistics as st
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score as ap

MODELS = ("rule", "tabular", "frozen", "finetune")
TARGETS = {"public-test": Path("../../002-no-hadoop/baseline/data/splits/public/test"),
           "jd4j-train": Path("data/splits/jd4j/train"), "jd4j-valid": Path("data/splits/jd4j/valid"),
           "jd4j-test": Path("data/splits/jd4j/test")}


def ms(xs):
    return {"mean": round(st.mean(xs), 4), "sd": round(st.stdev(xs), 4) if len(xs) > 1 else 0.0,
            "values": [round(x, 4) for x in xs]}


def paired_boot(y, week, a, b, reps=500, seed=42):
    ids = list(y); clusters = defaultdict(list)
    for i in ids:
        clusters[week[i]].append(i)
    keys = list(clusters); rng = np.random.default_rng(seed); d = []
    for _ in range(reps):
        s = [i for k in rng.choice(len(keys), len(keys)) for i in clusters[keys[k]]]
        yy = [y[i] for i in s]
        if 0 < sum(yy) < len(yy):
            d.append(ap(yy, [b[i] for i in s]) - ap(yy, [a[i] for i in s]))
    yy = [y[i] for i in ids]
    return {"delta": round(ap(yy, [b[i] for i in ids]) - ap(yy, [a[i] for i in ids]), 4),
            "ci95": [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)],
            "clusters": len(keys)}


def main():
    runs, out = Path(sys.argv[1]), Path(sys.argv[2])
    seeds = sorted({p.name.split("-")[0] for p in (runs / "evaluation").iterdir() if p.is_dir()})
    report = {"seeds": seeds, "targets": {}}
    for target, split in TARGETS.items():
        y = {json.loads(l)["id"]: json.loads(l)["label"] for l in open(split / "labels.jsonl")}
        week, repo = {}, {}
        for l in open(split / "inputs.jsonl"):
            r = json.loads(l)
            week[r["id"]] = tuple(datetime.fromisoformat(r["prediction_at"]).isocalendar()[:2]); repo[r["id"]] = r["repository"]
        base = sum(y.values()) / len(y)
        projects = defaultdict(list)
        for i in y:
            projects[repo[i]].append(i)
        eligible = {p: ids for p, ids in projects.items() if 20 <= sum(y[i] for i in ids) < len(ids)}
        per = {m: defaultdict(list) for m in MODELS}; boots = []
        for s in seeds:
            d = runs / "evaluation" / f"{s}-{target}"
            metrics = json.loads((d / "metrics.json").read_text())["metrics"]
            scores = {m: {r["id"]: r["score"] for r in map(json.loads, open(d / f"{m}.jsonl"))} for m in MODELS}
            if set(scores["rule"]) != set(y):
                raise SystemExit(f"{s}-{target}: prediction ids differ from label ids")
            for m in MODELS:
                per[m]["ap"].append(metrics[m]["ap"]); per[m]["lift"].append(metrics[m]["ap"] / base)
                per[m]["roc_auc"].append(metrics[m]["roc_auc"]); per[m]["top10_recall"].append(metrics[m]["top10_recall"])
                if eligible:
                    per[m]["per_project_mean_ap"].append(st.mean(
                        ap([y[i] for i in ids], [scores[m][i] for i in ids]) for ids in eligible.values()))
            boots.append({"seed": s, **paired_boot(y, week, scores["tabular"], scores["finetune"])})
        report["targets"][target] = {
            "n": len(y), "positives": sum(y.values()), "base_rate": round(base, 4),
            "projects_in_per_project_mean": sorted(eligible),
            "models": {m: {k: ms(v) for k, v in per[m].items()} for m in MODELS},
            "finetune_minus_tabular_ap_bootstrap": boots}
    out.write_text(json.dumps(report, indent=2))
    for t, r in report["targets"].items():
        print(f"== {t}  n={r['n']} pos={r['positives']} base={r['base_rate']}  projects={len(r['projects_in_per_project_mean'])}")
        for m, v in r["models"].items():
            pp = v.get("per_project_mean_ap", {"mean": float("nan"), "sd": 0})
            print(f"  {m:9} AP {v['ap']['mean']:.3f}±{v['ap']['sd']:.3f}  lift {v['lift']['mean']:.2f}  "
                  f"AUC {v['roc_auc']['mean']:.3f}±{v['roc_auc']['sd']:.3f}  R@10 {v['top10_recall']['mean']:.3f}  "
                  f"perProjAP {pp['mean']:.3f}±{pp['sd']:.3f}")
        print("  B-tab:", [(b["seed"], b["delta"], b["ci95"]) for b in r["finetune_minus_tabular_ap_bootstrap"]])


if __name__ == "__main__":
    main()
