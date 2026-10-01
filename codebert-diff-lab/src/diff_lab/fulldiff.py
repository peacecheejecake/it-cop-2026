"""Full before->after diffs from public git history for study diffllm-v1 (docs/diffllm-study-v1.md).

`extract_labeled`: `git show <sha> -U3` (first parent) for every JIT-Defects4J commit, keeping file
headers, hunk order and context lines; binary and lock-file sections are dropped and counted.
`render_prompts`: message (<= 128 tokens) + diff truncated to whole hunks within the token budget.
`build_cpt_corpus`: non-merge commits of repositories disjoint from JD4J, ranked by
sha256(salt:repo:sha) and taken until the token budget; duplicate diffs and any diff identical to a
JD4J labeled diff are excluded. Repositories are read-only bare clones; nothing is fetched here.
"""
from __future__ import annotations

import hashlib
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from .util import ExecutionError, IntegrityError, atomic_write_json, sha256_file, sha256_text

LOCK_FILES = re.compile(r"(^|/)(package-lock\.json|yarn\.lock|pnpm-lock\.yaml|Gemfile\.lock|poetry\.lock|Cargo\.lock|go\.sum)$")
CPT_SALT = "cpt-diff-v1"
FILE_SPLIT = re.compile(r"(?m)^diff --git ")


def _git(repo: Path, *args: str) -> str:
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, errors="replace")
    if r.returncode != 0:
        raise ExecutionError(r.stderr.strip()[:200])
    return r.stdout


def clean_diff(raw: str) -> tuple[str, dict]:
    """Drop binary and lock-file sections; return the diff and section counts."""
    parts = FILE_SPLIT.split(raw)
    kept, dropped_binary, dropped_lock = [], 0, 0
    for p in parts[1:]:
        header = p.split("\n", 1)[0]
        path = header.split(" b/", 1)[-1].strip()
        if "\nBinary files " in p or "GIT binary patch" in p:
            dropped_binary += 1
            continue
        if LOCK_FILES.search(path):
            dropped_lock += 1
            continue
        kept.append("diff --git " + p.rstrip("\n"))
    return "\n".join(kept), {"files": len(kept), "dropped_binary": dropped_binary, "dropped_lock": dropped_lock,
                             "hunks": sum(k.count("\n@@ ") for k in kept)}


def commit_diff(repo: Path, sha: str) -> tuple[str, dict]:
    raw = _git(repo, "show", sha, "--format=", "-U3", "--no-color", "--no-ext-diff", "--first-parent")
    return clean_diff(raw)


def extract_labeled(snapshot_dir: Path, git_root: Path, out: Path, workers: int = 8) -> dict:
    if out.exists():
        raise IntegrityError(f"{out} exists; full-diff extractions are immutable")
    changes = pd.read_parquet(snapshot_dir / "changes.parquet")[["change_id", "project_id", "commit_sha", "message"]]

    def one(r) -> dict:  # noqa: ANN001
        repo = git_root / f"{r.project_id}.git"
        try:
            diff, stats = commit_diff(repo, r.commit_sha)
            status = "ok" if diff else "empty_after_filter"
        except ExecutionError as e:
            diff, stats, status = "", {"files": 0, "dropped_binary": 0, "dropped_lock": 0, "hunks": 0}, f"unavailable: {e}"[:200]
        return {"change_id": r.change_id, "project_id": r.project_id, "commit_sha": r.commit_sha, "diff_text": diff,
                "diff_sha256": sha256_text(diff), "status": status, **stats}

    with ThreadPoolExecutor(workers) as ex:
        rows = list(ex.map(one, changes.itertuples(index=False)))
    df = pd.DataFrame(rows)
    out.mkdir(parents=True)
    df.to_parquet(out / "fulldiff.parquet", index=False)
    heads = {p: _git(git_root / f"{p}.git", "rev-parse", "HEAD").strip() for p in sorted(df["project_id"].unique())}
    manifest = {"artifact_kind": "fulldiff_labeled", "snapshot_manifest_sha256": sha256_file(snapshot_dir / "manifest.json"),
                "git_cmd": "git show <sha> --format= -U3 --no-color --no-ext-diff --first-parent", "repo_heads": heads,
                "status_counts": df["status"].str.split(":").str[0].value_counts().to_dict(),
                "dropped_binary_sections": int(df["dropped_binary"].sum()), "dropped_lock_sections": int(df["dropped_lock"].sum()),
                "fulldiff_parquet_sha256": sha256_file(out / "fulldiff.parquet")}
    atomic_write_json(out / "manifest.json", manifest)
    return manifest


def split_hunks(diff: str) -> list[str]:
    """Units that can be dropped whole: each file header joined with its first hunk, then one unit per later hunk."""
    units = []
    for f in [("diff --git " + p) for p in FILE_SPLIT.split(diff)[1:]]:
        pieces = re.split(r"(?m)^(?=@@ )", f)
        if len(pieces) <= 2:
            units.append(f.rstrip("\n"))
        else:
            units.append((pieces[0] + pieces[1]).rstrip("\n"))
            units += [p.rstrip("\n") for p in pieces[2:]]
    return units


def render_prompt(tok, message: str, diff: str, budget: int, max_msg_tokens: int = 128) -> tuple[str, str, int, bool]:  # noqa: ANN001
    """Title/body (<= max_msg_tokens) and the diff cut to whole hunks so that both fit `budget` tokens."""
    enc = tok(message, add_special_tokens=False, return_offsets_mapping=True)
    keep = min(len(enc["input_ids"]), max_msg_tokens)
    msg = message[: enc["offset_mapping"][keep - 1][1]] if keep else ""
    left = budget - keep
    units, used, out = split_hunks(diff), 0, []
    truncated = keep < len(enc["input_ids"])
    for u in units:
        n = len(tok("\n" + u, add_special_tokens=False)["input_ids"])
        if used + n > left:
            truncated = True
            break
        out.append(u)
        used += n
    return msg, "\n".join(out), keep + used, truncated


def build_cpt_corpus(git_root: Path, repos: list[str], exclude_hashes: set[str], tok, budget: int, max_tokens: int,  # noqa: ANN001
                     out: Path, workers: int = 8) -> dict:
    if out.exists():
        raise IntegrityError(f"{out} exists; CPT corpora are immutable")
    ranked = []
    heads = {}
    for repo in repos:
        rp = git_root / f"{repo}.git"
        heads[repo] = _git(rp, "rev-parse", "HEAD").strip()
        for sha in _git(rp, "rev-list", "--no-merges", "HEAD").split():
            ranked.append((hashlib.sha256(f"{CPT_SALT}:{repo}:{sha}".encode()).hexdigest(), repo, sha))
    ranked.sort()
    seen, rows, total, stats = set(exclude_hashes), [], 0, {"empty": 0, "duplicate": 0, "excluded_jd4j": 0}

    def fetch(item):  # noqa: ANN001, ANN202
        _, repo, sha = item
        rp = git_root / f"{repo}.git"
        try:
            msg = _git(rp, "log", "-1", "--format=%B", sha).strip()
            diff, _ = commit_diff(rp, sha)
        except ExecutionError:
            return repo, sha, "", ""
        return repo, sha, msg, diff

    with ThreadPoolExecutor(workers) as ex:
        for start in range(0, len(ranked), 2000):
            for repo, sha, msg, diff in ex.map(fetch, ranked[start:start + 2000]):
                if not diff:
                    stats["empty"] += 1
                    continue
                h = sha256_text(diff)
                if h in seen:
                    stats["excluded_jd4j" if h in exclude_hashes else "duplicate"] += 1
                    continue
                seen.add(h)
                m, d, n, cut = render_prompt(tok, msg, diff, max_tokens)
                if not d:
                    stats["empty"] += 1
                    continue
                text = f"Commit message:\n{m}\n\nDiff:\n{d}"
                n_tok = len(tok(text, add_special_tokens=False)["input_ids"])
                rows.append({"repo": repo, "sha": sha, "text": text, "n_tokens": n_tok, "truncated": cut, "diff_sha256": h})
                total += n_tok
                if total >= budget:
                    break
            if total >= budget:
                break
    if total < budget:
        raise ExecutionError(f"CPT corpus reached only {total} tokens < budget {budget}; add repositories (no silent shortfall)")
    df = pd.DataFrame(rows)
    out.mkdir(parents=True)
    df.to_parquet(out / "corpus.parquet", index=False)
    manifest = {"artifact_kind": "cpt_diff_corpus", "salt": CPT_SALT, "repos": repos, "repo_heads": heads, "budget_tokens": budget,
                "tokens": int(total), "commits": int(len(df)), "per_repo": df.groupby("repo")["n_tokens"].agg(["count", "sum"]).to_dict(),
                "truncated_commits": int(df["truncated"].sum()), "skipped": stats, "max_tokens_per_commit": max_tokens,
                "candidates_ranked": len(ranked), "corpus_parquet_sha256": sha256_file(out / "corpus.parquet")}
    atomic_write_json(out / "manifest.json", manifest)
    return manifest
