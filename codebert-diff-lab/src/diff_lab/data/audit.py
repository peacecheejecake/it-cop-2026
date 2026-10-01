"""Dataset audit (spec FR-04, protocol §4.1, AT-02).

Label-free duplicate rules fixed before any model is trained:
  exact      content_hash (message + sorted added + sorted deleted lines)
  code       identical non-empty sorted added/deleted line sets, message ignored
  normalized exact after collapsing whitespace (reported only; never used to exclude)
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from ..util import IntegrityError, atomic_write_json

TABLES = ("changes", "edits", "features", "labels", "splits")


def load_snapshot(snapshot_dir: str | Path) -> dict[str, pd.DataFrame]:
    d = Path(snapshot_dir)
    return {t: pd.read_parquet(d / f"{t}.parquet") for t in TABLES}


def _h(obj: object) -> str:
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def duplicate_keys(t: dict[str, pd.DataFrame]) -> pd.DataFrame:
    e = t["edits"].sort_values(["change_id", "op", "order"])
    lines = e.groupby(["change_id", "op"])["text"].apply(list).unstack(fill_value=[])
    lines = lines.reindex(t["changes"]["change_id"]).apply(lambda c: c.map(lambda v: v if isinstance(v, list) else []))
    adds = lines.get("add", pd.Series([[]] * len(lines), index=lines.index))
    dels = lines.get("delete", pd.Series([[]] * len(lines), index=lines.index))
    msg = t["changes"].set_index("change_id")["message"]
    ws = re.compile(r"\s+")
    out = pd.DataFrame({"change_id": msg.index})
    out["exact_key"] = t["changes"]["content_hash"].to_numpy()
    out["code_key"] = [(_h([a, d]) if (a or d) else None) for a, d in zip(adds, dels, strict=True)]
    out["normalized_key"] = [_h([ws.sub(" ", m).strip(), sorted(ws.sub(" ", x).strip() for x in a),
                                 sorted(ws.sub(" ", x).strip() for x in d)]) for m, a, d in zip(msg, adds, dels, strict=True)]
    out["n_add"] = [len(a) for a in adds]
    out["n_del"] = [len(d) for d in dels]
    return out


def audit(snapshot_dir: str | Path, out_dir: str | Path) -> dict:
    t = load_snapshot(snapshot_dir)
    c, lab, sp, feat = t["changes"], t["labels"], t["splits"], t["features"]
    if not c["change_id"].is_unique:
        raise IntegrityError("duplicate change_id")
    if lab["label"].isna().any() or not set(lab["label"].unique()) <= {0, 1}:
        raise IntegrityError("labels must be 0/1 for this snapshot")
    keys = duplicate_keys(t)
    df = (keys.merge(sp[["change_id", "split"]], on="change_id").merge(lab[["change_id", "label"]], on="change_id")
          .merge(c[["change_id", "project_id", "message", "authored_at_unix"]], on="change_id"))
    report: dict = {"n_changes": int(len(c)), "splits": {}, "duplicates": {}, "label_conflicts": {}, "content": {},
                    "temporal": {}, "features": {}}
    for s, g in df.groupby("split"):
        report["splits"][s] = {"n": int(len(g)), "positives": int(g["label"].sum()),
                               "positive_rate": round(float(g["label"].mean()), 4), "projects": int(g["project_id"].nunique())}
        ts = pd.to_datetime(g["authored_at_unix"], unit="s", utc=True)
        report["temporal"][s] = {"min": ts.min().isoformat(), "max": ts.max().isoformat()}
    report["temporal"]["provided_split_time_ordered"] = bool(
        report["temporal"]["train"]["max"] < report["temporal"]["valid"]["min"]
        and report["temporal"]["valid"]["max"] < report["temporal"]["test"]["min"])
    for kind in ("exact_key", "code_key", "normalized_key"):
        k = df.dropna(subset=[kind])
        groups = k.groupby(kind)
        multi = groups.filter(lambda g: len(g) > 1)
        cross = multi.groupby(kind)["split"].nunique()
        cross_keys = set(cross[cross > 1].index)
        pairs = Counter()
        for _, g in multi[multi[kind].isin(cross_keys)].groupby(kind):
            ss = sorted(set(g["split"]))
            for i in range(len(ss)):
                for j in range(i + 1, len(ss)):
                    pairs[f"{ss[i]}-{ss[j]}"] += 1
        conflicts = multi.groupby(kind)["label"].nunique()
        report["duplicates"][kind] = {"rows_in_duplicate_groups": int(len(multi)), "groups": int(multi[kind].nunique()),
                                      "cross_split_groups": len(cross_keys), "cross_split_pairs": dict(pairs)}
        report["label_conflicts"][kind] = int((conflicts > 1).sum())
    report["content"] = {"empty_message": int((df["message"].str.strip() == "").sum()),
                         "add_only": int(((df.n_add > 0) & (df.n_del == 0)).sum()),
                         "delete_only": int(((df.n_add == 0) & (df.n_del > 0)).sum()),
                         "no_code_lines": int(((df.n_add == 0) & (df.n_del == 0)).sum()),
                         "added_lines_median": float(df.n_add.median()), "deleted_lines_median": float(df.n_del.median())}
    fcols = [x for x in feat.columns if x not in ("change_id", "feature_schema_id", "provenance")]
    report["features"] = {col: {"negative": int((feat[col] < 0).sum()), "nan": int(feat[col].isna().sum()),
                                "min": float(feat[col].min()), "median": float(feat[col].median()), "max": float(feat[col].max())}
                          for col in fcols}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    keys.to_parquet(out / "duplicate-keys.parquet", index=False)
    atomic_write_json(out / "audit.json", report)
    return report
