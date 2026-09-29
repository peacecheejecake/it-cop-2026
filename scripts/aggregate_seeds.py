"""Aggregate per-seed public-test results: mean ± sd across seeds for full-test metrics,
per-project mean AP (from tools/sensitivity.py outputs) and per-year AP.
Usage: python aggregate_seeds.py <runs_dir> <test_split_dir> <out.json>"""
import json
import statistics as st
import sys
from pathlib import Path

from sklearn.metrics import average_precision_score as ap

MODELS = ("rule", "tabular", "frozen", "finetune")
KEYS = ("ap", "roc_auc", "top5_recall", "top10_recall", "top10_precision")


def ms(xs):
    return {"mean": round(st.mean(xs), 4), "sd": round(st.stdev(xs), 4) if len(xs) > 1 else 0.0,
            "values": [round(x, 4) for x in xs]}


def main():
    runs, split, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    seeds = sorted(p.name for p in (runs / "evaluation").iterdir() if p.is_dir() and p.name.startswith("seed"))
    y = {json.loads(l)["id"]: json.loads(l)["label"] for l in open(split / "labels.jsonl")}
    year = {json.loads(l)["id"]: json.loads(l)["prediction_at"][:4] for l in open(split / "inputs.jsonl")}
    full = {m: {k: [] for k in KEYS} for m in MODELS}
    per_project = {m: [] for m in MODELS}
    per_year = {}
    for s in seeds:
        metrics = json.loads((runs / "evaluation" / s / "metrics.json").read_text())["metrics"]
        for m in MODELS:
            for k in KEYS:
                full[m][k].append(metrics[m][k])
        sens = json.loads((runs / "evaluation" / f"{s}-sensitivity.json").read_text())
        for m in MODELS:
            per_project[m].append(sens["per_project_mean_ap"][m])
        scores = {m: {r["id"]: r["score"] for r in map(json.loads, open(runs / "evaluation" / s / f"{m}.jsonl"))}
                  for m in MODELS}
        for yr in sorted(set(year.values())):
            ids = [i for i in y if year[i] == yr]; yy = [y[i] for i in ids]
            for m in MODELS:
                per_year.setdefault(yr, {}).setdefault(m, []).append(ap(yy, [scores[m][i] for i in ids]))
    report = {"seeds": seeds,
              "full_test": {m: {k: ms(v) for k, v in full[m].items()} for m in MODELS},
              "per_project_mean_ap": {m: ms(v) for m, v in per_project.items()},
              "per_year_ap": {yr: {m: ms(v) for m, v in d.items()} for yr, d in per_year.items()},
              "finetune_minus_tabular_ap": {
                  "full": ms([a - b for a, b in zip(full["finetune"]["ap"], full["tabular"]["ap"])]),
                  "per_project_mean": ms([a - b for a, b in zip(per_project["finetune"], per_project["tabular"])]),
                  **{yr: ms([a - b for a, b in zip(d["finetune"], d["tabular"])]) for yr, d in per_year.items()}}}
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
