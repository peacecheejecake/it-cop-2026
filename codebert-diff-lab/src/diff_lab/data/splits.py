"""Controlled split from the provided split (spec protocol §2.1, §4.1).

upstream-clean1: evaluation membership (valid/test) is kept exactly as provided. Train
rows sharing an exact or code-identical key with any valid/test row are excluded from
train and from CPT. CPT-dev is ~5% of the clean train chosen by sha256(salt:change_id),
independent of labels and model seeds.

upstream-clean2: identical train/valid/test membership and train exclusions, but CPT-dev is
drawn by duplicate group (connected components of exact and code-identical keys within the
clean train), so no CPT-train row shares text or code with a CPT-dev row. Group order is
sha256(cpt-dev-v2:min change_id); groups are taken until ~5% of rows.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from ..util import IntegrityError, PolicyError, atomic_write_json, sha256_file, sha256_json
from .audit import duplicate_keys, load_snapshot

SPLIT_IDS = ("upstream-clean1", "upstream-clean2")
CPT_DEV_FRACTION = 0.05
CPT_DEV_SALT = {"upstream-clean1": "cpt-dev-v1", "upstream-clean2": "cpt-dev-v2"}


def _rank(salt: str, cid: str) -> str:
    return hashlib.sha256(f"{salt}:{cid}".encode()).hexdigest()


def _components(ids: list[str], key_cols: list[list]) -> dict[str, str]:
    parent = {i: i for i in ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for keys in key_cols:
        first: dict = {}
        for cid, k in zip(ids, keys, strict=True):
            if k is None or (isinstance(k, float) and k != k):
                continue
            if k in first:
                a, b = find(cid), find(first[k])
                if a != b:
                    parent[max(a, b)] = min(a, b)
            else:
                first[k] = cid
    return {i: find(i) for i in ids}


def _dev_ids(clean: pd.DataFrame, split_id: str) -> set[str]:
    salt = CPT_DEV_SALT[split_id]
    n_dev = round(len(clean) * CPT_DEV_FRACTION)
    if split_id == "upstream-clean1":
        ranked = clean.sort_values("change_id", key=lambda s: s.map(lambda c: _rank(salt, c)))
        return set(ranked["change_id"].iloc[:n_dev])
    ids = clean["change_id"].tolist()
    comp = _components(ids, [clean["exact_key"].tolist(), clean["code_key"].tolist()])
    groups: dict[str, list[str]] = {}
    for cid, root in comp.items():
        groups.setdefault(root, []).append(cid)
    dev: set[str] = set()
    for root in sorted(groups, key=lambda r: _rank(salt, min(groups[r]))):
        if len(dev) >= n_dev:
            break
        dev.update(groups[root])
    return dev


def make_controlled_split(snapshot_dir: str | Path, out_dir: str | Path, split_id: str = "upstream-clean1") -> dict:
    if split_id not in SPLIT_IDS:
        raise PolicyError(f"unknown split {split_id}")
    out = Path(out_dir)
    if out.exists():
        raise PolicyError(f"{out} exists; split versions are immutable")
    t = load_snapshot(snapshot_dir)
    keys = duplicate_keys(t).merge(t["splits"][["change_id", "split"]], on="change_id")
    if set(keys["split"]) != {"train", "valid", "test"}:
        raise IntegrityError("provided split must contain train/valid/test")
    evals = keys[keys["split"] != "train"]
    eval_exact, eval_code = set(evals["exact_key"]), set(evals["code_key"].dropna())
    train = keys[keys["split"] == "train"].copy()
    train["excluded_reason"] = None
    train.loc[train["code_key"].isin(eval_code), "excluded_reason"] = "code_identical_to_eval"
    train.loc[train["exact_key"].isin(eval_exact), "excluded_reason"] = "exact_duplicate_of_eval"
    clean = train[train["excluded_reason"].isna()].copy()
    dev_ids = _dev_ids(clean, split_id)
    rows = []
    for r in keys.itertuples():
        if r.split == "train":
            if r.change_id not in set(clean["change_id"]):
                continue
            cpt = "cpt_dev" if r.change_id in dev_ids else "cpt_train"
        else:
            cpt = None
        rows.append({"change_id": r.change_id, "split": r.split, "split_version": split_id,
                     "group_id": r.exact_key, "cpt_role": cpt})
    sp = pd.DataFrame(rows)
    out.mkdir(parents=True)
    sp.to_parquet(out / "splits.parquet", index=False)
    excluded = train[train["excluded_reason"].notna()][["change_id", "excluded_reason"]]
    excluded.to_parquet(out / "excluded.parquet", index=False)
    lab = t["labels"].set_index("change_id")["label"]
    counts = {s: {"n": int(len(g)), "positives": int(lab.loc[g["change_id"]].sum())} for s, g in sp.groupby("split")}
    counts["cpt_train"] = int((sp["cpt_role"] == "cpt_train").sum())
    counts["cpt_dev"] = int((sp["cpt_role"] == "cpt_dev").sum())
    manifest = {"split_id": split_id, "parent_snapshot_manifest_sha256": sha256_file(Path(snapshot_dir) / "manifest.json"),
                "rules": {"eval_membership": "provided valid/test kept unchanged",
                          "train_exclusion": ["exact_duplicate_of_eval", "code_identical_to_eval"],
                          "cpt_dev": {"fraction": CPT_DEV_FRACTION, "salt": CPT_DEV_SALT[split_id], "selection": "sha256 rank, label-free",
                                      "unit": "change" if split_id == "upstream-clean1" else "exact/code duplicate group"},
                          "time_ordering": "not time-ordered (provided split); reported as upstream_holdout"},
                "counts": counts, "excluded": excluded["excluded_reason"].value_counts().to_dict(),
                "excluded_positives": int(lab.loc[excluded["change_id"]].sum()) if len(excluded) else 0,
                "membership_sha256": sha256_json(sorted(zip(sp["change_id"], sp["split"], sp["cpt_role"].fillna(""), strict=True))),
                "splits_parquet_sha256": sha256_file(out / "splits.parquet")}
    atomic_write_json(out / "manifest.json", manifest)
    return manifest
