from __future__ import annotations

import csv
import hashlib
import math
import os
import re
import shlex
import subprocess
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import httpx

from .common import (BenchError, assert_binary, canonical, digest, file_hash, new_dir, now,
                     read_json, read_jsonl, seal_files, unique, utc, verify_files, write_json, write_jsonl)

FEATURE_VERSION = "patch-static-v1"
FEATURES = ("lines_added", "lines_deleted", "files_changed", "directories_changed",
            "change_entropy", "binary_files", "hunks")
ARCHIVE_URL = "https://zenodo.org/records/5907847/files/apachejit_dataset_replication.zip?download=1"
ARCHIVE_MD5 = "528bf0ee04b15976be6bf15f8efddc65"  # Zenodo v2 record; integrity, not authenticity.
HEX = re.compile(r"^[0-9a-fA-F]{7,64}$")
REPO = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


def normalize_patch(diff: str) -> str:
    lines = []
    for line in diff.replace("\r\n", "\n").splitlines():
        if line.startswith("index "):
            continue
        if line.startswith("@@"):
            line = re.sub(r"^@@.*?@@", "@@", line)
        lines.append(line.rstrip())
    return "\n".join(lines).strip()


def static_features(diff: str) -> dict:
    """Same parser for public/internal patches. No supplied CSV 'fix' or history columns."""
    churn = []; directories = set(); added = deleted = binary = hunks = 0
    in_hunk = False
    for line in diff.replace("\r\n", "\n").splitlines():
        if line.startswith("diff --git "):
            churn.append(0)
            in_hunk = False
            same_path = re.fullmatch(r"diff --git a/(.+) b/\1", line)
            if same_path:
                path = same_path.group(1)
            else:
                try:
                    parts = shlex.split(line)
                    path = parts[-1]
                except ValueError:
                    path = line[len("diff --git "):]
                path = path[2:] if path.startswith(("a/", "b/")) else path
            directories.add(str(Path(path).parent))
        elif line.startswith("@@"):
            in_hunk = True; hunks += 1
        elif line.startswith(("Binary files ", "GIT binary patch")):
            binary += 1; in_hunk = False
        elif in_hunk and line.startswith(("+", "-")):
            if not churn:
                churn.append(0); directories.add(".")
            churn[-1] += 1
            added += line.startswith("+")
            deleted += line.startswith("-")
    total = sum(churn)
    entropy = -sum((n/total)*math.log2(n/total) for n in churn if n) if total else 0.0
    return dict(zip(FEATURES, (added, deleted, len(churn), len(directories), entropy, binary, hunks)))


def binary_value(x: object) -> int:
    s = str(x).strip().lower()
    if s in ("1", "true", "1.0"):
        return 1
    if s in ("0", "false", "0.0"):
        return 0
    raise BenchError("Invalid or missing binary label")


def canonical_row(row: dict) -> dict:
    r = dict(row)
    for key in ("id", "repository", "commit", "message", "diff", "group_id", "label_type"):
        if not isinstance(r.get(key), str):
            raise BenchError(f"Missing string field: {key}")
    if not r["id"] or not r["group_id"] or not r["diff"].strip():
        raise BenchError("Empty ID, group, or patch")
    if r.get("domain") not in ("public", "internal"):
        raise BenchError("domain must be public/internal")
    if r.get("label_status") not in ("confirmed_positive", "observed_negative"):
        raise BenchError("Unresolved/censored labels must not be mapped to 0")
    r["label"] = binary_value(r.get("label"))
    if (r["label"] == 1) != (r["label_status"] == "confirmed_positive"):
        raise BenchError("Label/status mismatch")
    r["prediction_at"] = utc(r["prediction_at"]).isoformat()
    r["input_available_at"] = utc(r["input_available_at"]).isoformat()
    if utc(r["input_available_at"]) > utc(r["prediction_at"]):
        raise BenchError("Input contains information available after prediction time")
    if r.get("label_available_at"):
        r["label_available_at"] = utc(r["label_available_at"]).isoformat()
        if utc(r["label_available_at"]) < utc(r["prediction_at"]):
            raise BenchError("Outcome label predates the prediction event")
    else:
        r["label_available_at"] = None
    r["diff"] = r["diff"].replace("\r\n", "\n")
    r["patch_hash"] = digest(normalize_patch(r["diff"]))
    r["features"] = static_features(r["diff"])
    r["feature_version"] = FEATURE_VERSION
    if type(r.get("synthetic", False)) is not bool:
        raise BenchError("synthetic must be a JSON boolean")
    r["synthetic"] = r.get("synthetic", False)
    return r


def git(repo: Path, *args: str, input_bytes: bytes | None = None) -> bytes:
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0", GIT_CONFIG_NOSYSTEM="1",
               GIT_PAGER="cat", GIT_EXTERNAL_DIFF="", GIT_LFS_SKIP_SMUDGE="1")
    cmd = ["git", "-c", "core.hooksPath=" + os.devnull, "-c", "diff.external=",
           "-c", "core.quotePath=false", "-C", str(repo), *args]
    p = subprocess.run(cmd, input=input_bytes, capture_output=True, timeout=600, env=env)
    if p.returncode:
        # stderr can contain internal paths/URLs; never echo it in shared logs.
        raise BenchError(f"Git operation {args[0]!r} failed with exit code {p.returncode}")
    return p.stdout


def extract_commit(repo: Path, commit: str, first_parent: bool = False) -> dict:
    if not HEX.fullmatch(commit):
        raise BenchError("Invalid Git commit SHA")
    raw = git(repo, "show", "-s", "--format=%H%x00%P%x00%cI%x00%B", commit)
    sha, parents, date, message = raw.decode("utf-8", errors="replace").split("\x00", 3)
    ps = parents.split()
    if len(ps) > 1 and not first_parent:
        raise BenchError("merge_commit_excluded")
    base = ps[0] if ps else git(repo, "hash-object", "-t", "tree", "--stdin", input_bytes=b"").decode().strip()
    patch = git(repo, "diff", "--no-ext-diff", "--no-textconv", "--no-renames", "--unified=3", base, sha, "--")
    if len(patch) > 2_000_000:
        raise BenchError("patch_over_2MB_excluded")
    return {"commit": sha.strip(), "message": message.strip(),
            "diff": patch.decode("utf-8", errors="replace"), "prediction_at": date.strip(),
            "input_available_at": date.strip()}


def download_apache(out: str, archive: str | None = None) -> dict:
    root = new_dir(out)
    zpath = Path(archive) if archive else root / "apachejit_dataset_replication.zip"
    if not archive:
        try:
            with httpx.stream("GET", ARCHIVE_URL, timeout=120, follow_redirects=True) as r:
                r.raise_for_status()
                with zpath.open("wb") as f:
                    for chunk in r.iter_bytes():
                        f.write(chunk)
        except httpx.HTTPError as e:
            raise BenchError("Download failed; use --archive with an approved offline copy") from e
    md5 = hashlib.md5(zpath.read_bytes()).hexdigest()
    if md5 != ARCHIVE_MD5:
        raise BenchError("Archive checksum differs from pinned Zenodo v2; inspect the source version")
    # Never extract/execute bundled scripts or blindly extract ZIP paths.
    with zipfile.ZipFile(zpath) as z:
        names = [n for n in z.namelist() if n.endswith("/dataset/apachejit_total.csv")]
        if len(names) != 1 or z.getinfo(names[0]).file_size > 100_000_000:
            raise BenchError("Unexpected archive layout/size")
        (root / "apachejit_total.csv").write_bytes(z.read(names[0]))
    report = {"source_url": ARCHIVE_URL, "source_md5": md5,
              "source_sha256": file_hash(zpath), "csv_sha256": file_hash(root / "apachejit_total.csv"),
              "downloaded_at": now(), "note": "CSV only; source code repositories carry their own licenses"}
    write_json(root / "source.json", report)
    return report


def apache_build(csv_path: str, repos: str, out: str, allow_network: bool = False,
                 limit: int | None = None, seed: int = 42, first_parent: bool = False,
                 commit_col: str = "commit_id", repo_col: str = "project", label_col: str = "buggy") -> dict:
    root = new_dir(out); cache = Path(repos); cache.mkdir(parents=True, exist_ok=True)
    with Path(csv_path).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not {commit_col, repo_col, label_col}.issubset(reader.fieldnames or []):
            raise BenchError(f"CSV columns required: {commit_col}, {repo_col}, {label_col}; got {reader.fieldnames}")
        rows = list(reader)
    # Label-blind sample for plumbing only. Default: use all rows without resampling.
    if limit is not None:
        if limit < 1:
            raise BenchError("limit must be positive")
        rows = sorted(rows, key=lambda r: digest([seed, r[repo_col], r[commit_col]]))[:limit]
    accepted = []; rejected = []; initialized = {}; failed_repos = {}; seen = set()
    for source in rows:
        name = source[repo_col].strip(); commit = source[commit_col].strip()
        rid = f"apachejit:{name}:{commit}"
        try:
            if not REPO.fullmatch(name) or any(p in (".", "..") for p in name.split("/")):
                raise BenchError("Expected owner/repository; provide a corrected explicit repository map")
            if name in failed_repos:
                raise BenchError(failed_repos[name])
            if name not in initialized:
                path = cache / (name + ".git")
                if not path.exists() and (cache / name).exists():
                    path = cache / name
                if not path.exists():
                    if not allow_network:
                        failed_repos[name] = "repository_missing_offline"
                        raise BenchError("repository_missing_offline")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    # Public import only. Never called for internal manifests.
                    p = subprocess.run(["git", "-c", "core.hooksPath=" + os.devnull, "clone", "--mirror",
                                        f"https://github.com/{name}.git", str(path)],
                                       capture_output=True, timeout=1200,
                                       env=dict(os.environ, GIT_TERMINAL_PROMPT="0"))
                    if p.returncode:
                        failed_repos[name] = "repository_clone_failed"
                        raise BenchError("repository_clone_failed")
                initialized[name] = path
            x = extract_commit(initialized[name], commit, first_parent)
            rid = f"apachejit:{name}:{x['commit']}"
            if rid in seen:
                raise BenchError("duplicate_commit")
            y = binary_value(source[label_col])
            record = canonical_row(dict(x, id=rid, repository=name, group_id=f"{name}:{x['commit']}",
                                        domain="public", label=y, label_type="bug_inducing_commit",
                                        label_status="confirmed_positive" if y else "observed_negative",
                                        label_available_at=None, synthetic=False))
            accepted.append(record); seen.add(rid)
        except (BenchError, subprocess.TimeoutExpired) as e:
            rejected.append({"id": rid, "reason": str(e) if isinstance(e, BenchError) else "git_timeout"})
    write_jsonl(root / "records.jsonl", accepted)
    write_jsonl(root / "rejected.jsonl", rejected)
    report = {"source_csv_sha256": file_hash(csv_path), "selected": len(rows), "accepted": len(accepted),
              "rejected": len(rejected), "reasons": dict(Counter(r["reason"] for r in rejected)),
              "sampling": "none" if limit is None else f"label-blind hash sample, limit={limit}, seed={seed}",
              "merge_policy": "first-parent" if first_parent else "exclude",
              "timestamp": "Git committer date, NOT dataset author_date",
              "features": FEATURE_VERSION, "label_availability": "unknown; retrospective benchmark only",
              "warning": "Archive labels are not re-derived or guaranteed noise-free; original CSV historical features are not used"}
    write_json(root / "build_report.json", report)
    if not accepted:
        raise BenchError("No accepted commits; inspect build_report.json and repository cache")
    return report


def internal_build(manifest: str, out: str) -> dict:
    """Read local exported patches only. Labels must already be adjudicated upstream."""
    root = new_dir(out); source_path = Path(manifest).resolve(); base = source_path.parent
    accepted = []; rejected = []
    with source_path.open(encoding="utf-8-sig", newline="") as f:
        for i, row in enumerate(csv.DictReader(f), 2):
            try:
                path = (base / row["diff_path"]).resolve()
                if not path.is_relative_to(base):
                    raise BenchError("diff_path must remain under manifest directory")
                if path.stat().st_size > 2_000_000:
                    raise BenchError("patch_over_2MB_excluded")
                item = {"id": row["id"], "repository": row["repository"],
                        "commit": row.get("commit", ""), "group_id": row["group_id"],
                        "prediction_at": row["prediction_at"], "input_available_at": row["input_available_at"],
                        "message": row.get("message", ""), "diff": path.read_text(encoding="utf-8"),
                        "label": row["label"], "label_type": row["label_type"], "domain": "internal",
                        "label_status": row["label_status"], "label_available_at": row.get("label_available_at"),
                        "synthetic": bool(binary_value(row.get("synthetic") or "0"))}
                accepted.append(canonical_row(item))
            except (KeyError, ValueError, OSError) as e:
                rejected.append({"id": row.get("id", f"line:{i}"), "reason": type(e).__name__ + ": " +
                                 (str(e) if isinstance(e, BenchError) else "invalid local export")})
    unique(accepted)
    write_jsonl(root / "records.jsonl", accepted); write_jsonl(root / "rejected.jsonl", rejected)
    report = {"selected": len(accepted) + len(rejected), "accepted": len(accepted), "rejected": len(rejected),
              "source_sha256": file_hash(manifest), "network_used": False,
              "warning": "Only provided resolved labels are included. Eligibility bias and incident causality require upstream audit."}
    write_json(root / "build_report.json", report)
    if not accepted:
        raise BenchError("No resolved internal rows; inspect rejected.jsonl")
    return report


def write_partition(rows: list[dict], out: Path, role: str, retrospective: bool, metadata: dict | None = None) -> None:
    new_dir(out)
    inputs = []; labels = []
    for r in rows:
        inputs.append({k: r[k] for k in ("id", "domain", "repository", "commit", "group_id", "prediction_at",
                                         "input_available_at", "message", "diff", "patch_hash", "features",
                                         "feature_version", "synthetic")})
        labels.append({k: r[k] for k in ("id", "label", "label_type", "label_status", "label_available_at")})
    write_jsonl(out / "inputs.jsonl", inputs); write_jsonl(out / "labels.jsonl", labels)
    write_json(out / "manifest.json", {"schema": "riskbench-dataset-v1", "role": role, "n": len(rows),
               "created_at": now(), "retrospective_labels": retrospective,
               "files": seal_files(out, ["inputs.jsonl", "labels.jsonl"]), "metadata": metadata or {}})


def split_public(records: str, out: str, train_before: str, valid_before: str,
                 allow_retrospective: bool = False, holdout_repos: list[str] | None = None) -> dict:
    root = new_dir(out); rows = [canonical_row(r) for r in read_jsonl(records)]; unique(rows)
    if not rows or any(r["domain"] != "public" for r in rows):
        raise BenchError("Public splitting accepts public records only")
    t1, t2 = utc(train_before), utc(valid_before)
    if t1 >= t2:
        raise BenchError("train_before must precede valid_before")
    holdout = set(holdout_repos or [])
    roles = []
    for r in rows:
        t = utc(r["prediction_at"])
        roles.append("test" if r["repository"] in holdout or t >= t2 else "train" if t < t1 else "valid")
    # Exclude entire connected components spanning split boundaries. Handles exact
    # patches, cherry-picks and release groups transitively without moving future rows backward.
    parent = list(range(len(rows)))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    seen = {}
    for i, r in enumerate(rows):
        for field in ("patch_hash", "group_id"):
            key = (field, r[field])
            if key in seen:
                parent[find(i)] = find(seen[key])
            else:
                seen[key] = i
    comps = defaultdict(list)
    for i in range(len(rows)):
        comps[find(i)].append(i)
    rejected = []; kept = {k: [] for k in ("train", "valid", "test")}; dedup = set()
    for members in comps.values():
        if len({roles[i] for i in members}) > 1:
            rejected.extend({"id": rows[i]["id"], "reason": "group_or_exact_patch_crosses_split"} for i in members)
            continue
        for i in sorted(members, key=lambda i: (rows[i]["prediction_at"], rows[i]["id"])):
            r = rows[i]; role = roles[i]
            if r["patch_hash"] in dedup:
                rejected.append({"id": r["id"], "reason": "duplicate_patch"}); continue
            # Known late labels are excluded even when unknown-date retrospective mode is allowed.
            cutoff = t1 if role == "train" else t2 if role == "valid" else None
            if cutoff and r["label_available_at"] and utc(r["label_available_at"]) >= cutoff:
                rejected.append({"id": r["id"], "reason": "label_not_available_at_fit_cutoff"}); continue
            if cutoff and not r["label_available_at"] and not allow_retrospective:
                raise BenchError("Missing label_available_at; add real availability or explicitly use --allow-retrospective")
            kept[role].append(r); dedup.add(r["patch_hash"])
    for role in ("train", "valid"):
        assert_binary(kept[role], both=True)
    if not kept["test"]:
        raise BenchError("Public test partition is empty")
    metadata = {"train_before": train_before, "valid_before": valid_before, "heldout_repositories": sorted(holdout),
                "source_sha256": file_hash(records)}
    for role, part in kept.items():
        write_partition(sorted(part, key=lambda r: (r["prediction_at"], r["id"])), root / role, role,
                        allow_retrospective, metadata)
    write_jsonl(root / "excluded.jsonl", rejected)
    report = {"n": {k: len(v) for k, v in kept.items()}, "excluded": len(rejected),
              "exclusion_reasons": dict(Counter(r["reason"] for r in rejected)), **metadata,
              "retrospective_labels": allow_retrospective}
    write_json(root / "split_report.json", report)
    return report


def internal_partition(records: str, out: str) -> None:
    rows = [canonical_row(r) for r in read_jsonl(records)]; unique(rows)
    if not rows or any(r["domain"] != "internal" for r in rows):
        raise BenchError("Internal partition requires internal records only")
    write_partition(rows, Path(out), "internal_test", False, {"source_sha256": file_hash(records)})


def load_inputs(directory: str | Path) -> tuple[list[dict], dict]:
    p = Path(directory); m = read_json(p / "manifest.json")
    if m.get("schema") != "riskbench-dataset-v1":
        raise BenchError("Unknown dataset schema")
    # Inference does not read, hash, or require the outcome label file.
    verify_files(p, {"inputs.jsonl": m["files"]["inputs.jsonl"]})
    rows = read_jsonl(p / "inputs.jsonl"); unique(rows)
    if not rows or len(rows) != m["n"]:
        raise BenchError("Empty or inconsistent dataset")
    for r in rows:
        if any(k in r for k in ("label", "label_type", "label_status", "label_available_at", "incident_id", "buggy")):
            raise BenchError("Outcome field in inference inputs")
        if r.get("feature_version") != FEATURE_VERSION or set(r.get("features", {})) != set(FEATURES):
            raise BenchError("Feature contract mismatch")
        if utc(r["input_available_at"]) > utc(r["prediction_at"]):
            raise BenchError("Future input")
        if r.get("domain") not in ("public", "internal"):
            raise BenchError("Unknown domain")
    if m["role"] == "internal_test" and any(r["domain"] != "internal" for r in rows):
        raise BenchError("Internal role/domain mismatch")
    return rows, m


def load_labeled(directory: str | Path, required_role: str | None = None) -> tuple[list[dict], list[dict], dict]:
    rows, m = load_inputs(directory)
    if required_role and (m["role"] != required_role or any(r["domain"] != "public" for r in rows)):
        raise BenchError(f"Public {required_role} partition required; internal data cannot train or select models")
    p = Path(directory); verify_files(p, {"labels.jsonl": m["files"]["labels.jsonl"]})
    index = unique(read_jsonl(p / "labels.jsonl"))
    if set(index) != {r["id"] for r in rows}:
        raise BenchError("Inputs/labels ID mismatch")
    labels = [index[r["id"]] for r in rows]; assert_binary(labels)
    return rows, labels, m
