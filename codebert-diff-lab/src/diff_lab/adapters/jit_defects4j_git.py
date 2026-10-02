"""Derived JIT-Defects4J snapshot whose text and jit14 come from the public git repositories (exp 015 finding).

The package's add/delete line sets are label-dependent: for buggy commits it keeps only some of the
changed files (presumably those with buggy lines), for clean commits all of them, so a model can read the label from
"how much of the commit the text covers". This snapshot keeps the package's change IDs, labels and
provided split (the approved source) and replaces message, lines and jit14 with `gitextract` output
from the commit itself, which is exactly what offline prediction sees on new repositories. Commits
missing from a mirror are dropped and counted per split/label; nothing is imputed.
"""
from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from ..config import FEATURE_PROFILE_JIT14
from ..gitextract import EXTRACTOR_VERSION, extract_repo
from ..util import IntegrityError, PolicyError, atomic_write_json, read_json, sha256_file, sha256_json

TABLES = ("changes", "edits", "features", "labels", "splits")
EXTENSIONS = (".java",)


def rebuild_from_git(parent: Path, mirrors: Path, out: Path) -> dict:
    if out.exists():
        raise PolicyError(f"output {out} exists; snapshots are immutable")
    pm = read_json(parent / "manifest.json")
    t = {name: pd.read_parquet(parent / f"{name}.parquet") for name in TABLES}
    ch = t["changes"]
    rows, repos = [], {}
    for proj, g in ch.groupby("project_id", sort=True):
        repo = mirrors / f"{proj}.git"
        if not repo.exists():
            raise IntegrityError(f"mirror for {proj} not found at {repo}")
        d, _, rep = extract_repo(repo, proj, extensions=EXTENSIONS, only=set(g["commit_sha"]))
        d["commit_sha"] = d["change_id"].str.split(":", n=1).str[1]
        rows.append(g[["change_id", "commit_sha"]].merge(d.drop(columns="change_id"), on="commit_sha", validate="1:1"))
        repos[proj] = {"head": rep["rev_sha"], "package_commits": int(len(g)), "extracted": rep["extracted"], "skipped": rep["skipped"]}
    got = pd.concat(rows, ignore_index=True).set_index("change_id")
    dropped = t["splits"].merge(t["labels"][["change_id", "label"]], on="change_id")
    dropped = dropped[~dropped["change_id"].isin(got.index)]
    out_t = {name: df[df["change_id"].isin(got.index)].reset_index(drop=True) for name, df in t.items() if name != "edits"}
    c = out_t["changes"]
    c["message"] = c["change_id"].map(got["message"])
    c["representation_kind"] = "git_extracted_lines"
    edits = []
    for cid, r in got.iterrows():
        for op, col in (("add", "added_lines"), ("delete", "deleted_lines")):
            edits += [(cid, None, None, i, op, x) for i, x in enumerate(r[col])]
    out_t["edits"] = pd.DataFrame(edits, columns=list(t["edits"].columns))
    f = out_t["features"]
    for k in FEATURE_PROFILE_JIT14:
        f[k] = f["change_id"].map(got[k]).astype(float)
    f["feature_schema_id"] = "jit14-gitextract-v1"
    f["provenance"] = EXTRACTOR_VERSION
    out.mkdir(parents=True)
    for name, df in out_t.items():
        df.to_parquet(out / f"{name}.parquet", index=False)
    merged = out_t["splits"].merge(out_t["labels"][["change_id", "label"]], on="change_id")
    by_split_label = {f"{s}/{y}": int(n) for (s, y), n in dropped.groupby(["split", "label"]).size().items()}
    manifest = {
        **{k: pm[k] for k in ("artifact_kind", "schema_version", "dataset_id", "visibility", "privacy", "source", "allowed_uses")},
        "adapter_version": "jit-defects4j-git-v1", "representation_kind": "git_extracted_lines",
        "imported_at": datetime.now(UTC).isoformat(), "code_git_sha": _git_sha(),
        "derived_from": {"parent_manifest_sha256": sha256_file(parent / "manifest.json"), "parent_adapter": pm["adapter_version"],
                         "kept_from_parent": ["change_id", "label", "provided split", "project", "commit/parent sha", "author date"],
                         "replaced": ["message", "add/delete lines", "jit14"], "extractor_version": EXTRACTOR_VERSION,
                         "extensions": list(EXTENSIONS), "mirrors": repos,
                         "reason": "package line sets are label-dependent (buggy commits keep only some files); see exp 015"},
        "dropped_missing_from_mirror": {"n": int(len(dropped)), "by_split_label": by_split_label},
        "representation_notes": ["lines from `git log -p --unified=0 -M` of the commit, .java files only, comments dropped, "
                                 "JD4J tokenization, set semantics, lexicographic order (edits.order is the sort rank)",
                                 "jit14 recomputed from git history (gitextract); fidelity vs package in exp 015 RESULTS"],
        "feature_schema": {"feature_schema_id": "jit14-gitextract-v1", "provenance": EXTRACTOR_VERSION},
        "split_stats": {s: {"n": int(len(g)), "positives": int(g["label"].sum())} for s, g in merged.groupby("split")},
        "table_sha256": {name: sha256_file(out / f"{name}.parquet") for name in TABLES},
        "split_membership_sha256": sha256_json(sorted(zip(out_t["splits"]["change_id"], out_t["splits"]["split"], strict=True))),
    }
    atomic_write_json(out / "manifest.json", manifest)
    return {"rows": int(len(c)), "dropped": manifest["dropped_missing_from_mirror"], "split_stats": manifest["split_stats"]}


def _git_sha() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
                              cwd=Path(__file__).parent).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
