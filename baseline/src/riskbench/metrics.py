from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .common import BenchError, check_score, read_json, read_jsonl, unique, utc, write_json
from .data import load_labeled


def ranking_metrics(y, scores, fractions=(0.05, 0.10)) -> dict:
    y = np.asarray(y, dtype=int); s = np.asarray(scores, dtype=float)
    if not len(y) or len(y) != len(s) or not np.isfinite(s).all() or not set(y).issubset({0, 1}):
        raise BenchError("Invalid metric inputs")
    n = len(y); p = int(y.sum())
    out = {"n": n, "positive": p, "prevalence": p / n,
           "ap": float(average_precision_score(y, s)) if p else None,
           "roc_auc": float(roc_auc_score(y, s)) if 0 < p < n else None,
           "unique_scores": int(len(np.unique(s)))}
    for q in fractions:
        if not 0 < q <= 1:
            raise BenchError("Review fraction must be in (0,1]")
        k = max(1, math.ceil(n*q))
        cut = np.partition(s, n-k)[n-k]
        above = s > cut; tied = s == cut
        slots = k - int(above.sum())
        # Expectation under random tie breaking, not label-ordered optimistic ties.
        tp = float(y[above].sum()) + float(y[tied].sum()) * slots / int(tied.sum())
        name = f"top{q*100:g}"
        out[f"{name}_k"] = k; out[f"{name}_review_rate"] = k/n
        out[f"{name}_expected_tp"] = tp
        out[f"{name}_recall"] = tp / p if p else None
        out[f"{name}_precision"] = tp / k
    return out


def _cluster_key(row, method):
    if method == "group":
        return row["group_id"]
    if method == "week":
        t = utc(row["prediction_at"]).isocalendar()
        return f"{t.year}-W{t.week:02}"
    raise BenchError("cluster must be group or week")


def evaluate(dataset: str, predictions: list[str], out: str, allow_partial: bool = False,
             allow_label_transfer: bool = False, bootstrap: int = 0, seed: int = 42,
             cluster: str = "week") -> dict:
    rows, labels, manifest = load_labeled(dataset)
    ids = [r["id"] for r in rows]; idset = set(ids); y = np.array([r["label"] for r in labels])
    label_types = sorted({r["label_type"] for r in labels})
    if len(label_types) != 1:
        raise BenchError("Do not pool different outcome definitions in one evaluation")
    predmaps = {}; metas = {}; coverage = {}; common = set(ids)
    for path in predictions:
        meta = read_json(path + ".meta.json")
        if meta["inputs_sha256"] != manifest["files"]["inputs.jsonl"]:
            raise BenchError("Prediction file belongs to a different dataset/view")
        from .common import file_hash
        if file_hash(path) != meta["predictions_sha256"]:
            raise BenchError("Prediction file was altered after sealing")
        name = meta["run_name"]
        if name in predmaps:
            raise BenchError("Each prediction run_name must be unique")
        if meta.get("training_label_type") != label_types[0] and not allow_label_transfer:
            raise BenchError("Source/target label mismatch; explicitly use --allow-label-transfer for transfer evaluation")
        pmap = unique(read_jsonl(path))
        if set(pmap) != idset:
            raise BenchError("Missing/extra prediction IDs; partial failures must be explicit rows")
        success = set()
        for rid, p in pmap.items():
            if p.get("status") == "ok":
                check_score(p.get("score")); success.add(rid)
        coverage[name] = {"total": len(ids), "successful": len(success), "failed": len(ids)-len(success),
                          "coverage": len(success)/len(ids),
                          "positives_in_failures": int(sum(label["label"] for label in labels if label["id"] not in success))}
        if success != idset and not allow_partial:
            raise BenchError("Failed predictions present; resolve them or explicitly use --allow-partial (common-success subset)")
        common &= success; predmaps[name] = pmap; metas[name] = meta
    selected = [i for i, rid in enumerate(ids) if rid in common]
    if not selected:
        raise BenchError("No common successfully scored rows")
    yc = y[selected]; rc = [rows[i] for i in selected]
    scores = {name: np.array([pm[ids[i]]["score"] for i in selected]) for name, pm in predmaps.items()}
    results = {name: ranking_metrics(yc, s) for name, s in scores.items()}
    clusters = defaultdict(list)
    for i, r in enumerate(rc):
        clusters[_cluster_key(r, cluster)].append(i)
    cluster_arrays = [np.array(v) for v in clusters.values()]
    keys = ("ap", "roc_auc", "top5_recall", "top10_recall", "top5_precision", "top10_precision")
    samples = {name: {key: [] for key in keys} for name in scores}
    deltas = {name: {key: [] for key in keys} for name in list(scores)[1:]}
    rng = np.random.default_rng(seed); first = next(iter(scores))
    if bootstrap < 0:
        raise BenchError("bootstrap must be nonnegative")
    for _ in range(bootstrap):
        choose = rng.integers(len(cluster_arrays), size=len(cluster_arrays))
        ix = np.concatenate([cluster_arrays[i] for i in choose])
        ms = {name: ranking_metrics(yc[ix], s[ix]) for name, s in scores.items()}
        for name, m in ms.items():
            for key in keys:
                if m[key] is not None:
                    samples[name][key].append(m[key])
                if name != first and m[key] is not None and ms[first][key] is not None:
                    deltas[name][key].append(m[key] - ms[first][key])
    def intervals(d):
        return {name: {key: {"valid_replicates": len(values),
                            "percentile95": [float(x) for x in np.quantile(values, [.025, .975])] if values else None}
                       for key, values in values_by_key.items()} for name, values_by_key in d.items()}
    report = {"evaluation_label_type": label_types[0], "source_label_types": {k: m["training_label_type"] for k,m in metas.items()},
              "dataset_role": manifest["role"], "synthetic": any(r["synthetic"] for r in rows),
              "retrospective_labels": manifest["retrospective_labels"], "coverage": coverage,
              "population": "entire dataset" if len(selected)==len(ids) else "COMMON SUCCESS SUBSET: selection bias possible",
              "common_n": len(selected), "original_n": len(ids), "metrics": results,
              "tie_policy": "expected true positives under random tie breaking at ceil(n*fraction)",
              "topk_policy": "retrospective global ranking; NOT an online threshold policy",
              "bootstrap": {"requested": bootstrap, "seed": seed, "cluster": cluster, "n_clusters": len(clusters),
                            "warning": "Few independent clusters; intervals may be unreliable" if len(clusters)<10 else None,
                            "intervals": intervals(samples), "paired_delta_reference": first,
                            "paired_delta_intervals": intervals(deltas)},
              "note": "Scores are uncalibrated rankings. Captured changes are not proven prevented incidents."}
    target = Path(out)
    if target.exists():
        raise BenchError("Refusing to overwrite evaluation report")
    write_json(target, report)
    csv_path = target.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        fields = ["model", "n", "positive", "ap", "roc_auc", "top5_recall", "top5_precision", "top10_recall", "top10_precision"]
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for name, m in results.items():
            writer.writerow({"model": name, **{k:m[k] for k in fields if k != "model"}})
    return report
