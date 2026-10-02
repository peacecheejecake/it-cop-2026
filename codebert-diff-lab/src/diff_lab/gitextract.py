"""Build the label-free internal dataset (docs/internal-offline-eval.md format) from local git history.

The public JIT-Defects4J package is the reference for every rule here, because the frozen
models only ever saw that representation (checked against the cached public mirrors with
`tools/check_gitextract.py`):

- lines: code files only; pure comment lines dropped and trailing comments cut; operators and
  punctuation split into single-character tokens (string literals included, quotes kept);
  `_` becomes a space; whitespace collapsed; empty lines dropped; each side is a set, sorted
  lexicographically;
- message: the full commit message without `git-svn-id:` trailers, same tokenization;
- jit14: Kamei change metrics over the code files of the commit, history accumulated over
  non-merge commits in topological order. `fix` is the `fix|bug` message keyword rule
  (95.5% agreement with the package), `lt` is the mean of per-file line counts accumulated from
  numstat (so it can be negative, as in the package), `age` is in days from author dates.

Merges are skipped (the package has none). Commits that touch no code file, or whose code lines
are all comments/blank, are skipped and counted in the manifest. Nothing here reads labels.
"""
from __future__ import annotations

import math
import re
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from .config import FEATURE_PROFILE_JIT14
from .util import ConfigError, ExecutionError, PolicyError, atomic_write_json, sha256_file, sha256_json

EXTRACTOR_VERSION = "gitextract-jd4j-v1"
C_LIKE = (".java", ".kt", ".kts", ".scala", ".groovy", ".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".cs", ".go", ".js", ".jsx",
          ".ts", ".tsx", ".swift", ".m", ".mm", ".rs", ".dart", ".php")
HASH_COMMENT = (".py", ".rb", ".sh", ".bash", ".pl", ".r")
DEFAULT_EXTENSIONS = C_LIKE + HASH_COMMENT + (".sql",)
LANGUAGE = {".java": "java", ".kt": "kotlin", ".kts": "kotlin", ".scala": "scala", ".groovy": "groovy", ".c": "c", ".h": "c",
            ".cc": "cpp", ".cpp": "cpp", ".cxx": "cpp", ".hpp": "cpp", ".cs": "csharp", ".go": "go", ".js": "javascript",
            ".jsx": "javascript", ".ts": "typescript", ".tsx": "typescript", ".swift": "swift", ".m": "objc", ".mm": "objc",
            ".rs": "rust", ".dart": "dart", ".php": "php", ".py": "python", ".rb": "ruby", ".sh": "shell", ".bash": "shell",
            ".pl": "perl", ".r": "r", ".sql": "sql"}
_SPLIT = re.compile(r"([.,;:(){}\[\]+\-*/=<>&|?@%^~\\])")
_FIX = re.compile(r"fix|bug", re.IGNORECASE)
# Patch and numstat lines never start with this, so splitting on "\n" + _SEP is unambiguous.
_SEP = "\x1e\x1eDIFFLAB "


def tokenize(text: str) -> str:
    return " ".join(_SPLIT.sub(r" \1 ", text.replace("_", " ")).split())


def clean_message(msg: str) -> str:
    return tokenize("\n".join(ln for ln in msg.split("\n") if not ln.startswith("git-svn-id:")))


def _cut_comment(line: str, marker: str) -> str:
    """Cut a trailing comment that starts outside string/char literals."""
    quote, i = None, 0
    while i < len(line):
        ch = line[i]
        if quote:
            if ch == "\\":
                i += 1
            elif ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
        elif line.startswith(marker, i):
            return line[:i]
        i += 1
    return line


def clean_line(raw: str, ext: str) -> str:
    s = raw.strip()
    if ext in C_LIKE:
        if s.startswith(("//", "/*", "*")):
            return ""
        s = _cut_comment(s, "//")
    elif ext in HASH_COMMENT:
        if s.startswith("#"):
            return ""
        s = _cut_comment(s, "#")
    return tokenize(s)


def _ext(path: str) -> str:
    return Path(path).suffix.lower()


def _git(repo: Path, *args: str, stdin: str | None = None) -> str:
    r = subprocess.run(["git", "-C", str(repo), *args], input=stdin, capture_output=True, text=True, errors="replace")
    if r.returncode != 0:
        raise ExecutionError(f"git {args[0]} failed in {repo}: {r.stderr.strip()[-500:]}")
    return r.stdout


@dataclass
class _History:
    loc: Counter = field(default_factory=Counter)
    devs: defaultdict = field(default_factory=lambda: defaultdict(set))
    last: dict = field(default_factory=dict)
    changes: defaultdict = field(default_factory=lambda: defaultdict(set))
    dev_times: defaultdict = field(default_factory=lambda: defaultdict(list))
    dev_subsys: defaultdict = field(default_factory=lambda: defaultdict(Counter))


def _subsystem(p: str) -> str:
    return p.split("/", 1)[0] if "/" in p else ""


def _features(h: _History, dev: str, at: int, files: list[tuple[str, int, int]]) -> dict:
    subs = {_subsystem(p) for p, _, _ in files}
    dirs = {p.rsplit("/", 1)[0] if "/" in p else "" for p, _, _ in files}
    la, ld = sum(a for _, a, _ in files), sum(d for _, _, d in files)
    tot = la + ld
    entropy = -sum(((a + d) / tot) * math.log2((a + d) / tot) for _, a, d in files if a + d > 0) if tot else 0.0
    ages = [(at - h.last[p]) / 86400 for p, _, _ in files if p in h.last]
    prev = h.dev_times[dev]
    return {"ns": len(subs), "nd": len(dirs), "nf": len(files), "entropy": entropy, "la": la, "ld": ld,
            "lt": sum(h.loc[p] for p, _, _ in files) / len(files),
            "ndev": len(set().union(*(h.devs[p] for p, _, _ in files))), "age": sum(ages) / len(files),
            "nuc": sum(len(h.changes[p]) for p, _, _ in files), "exp": len(prev),
            "rexp": sum(1 / ((at - t) / 86400 / 365 + 1) for t in prev), "sexp": sum(h.dev_subsys[dev][s] for s in subs)}


def _update(h: _History, sha: str, dev: str, at: int, files: list[tuple[str, int, int]]) -> None:
    for p, a, d in files:
        h.loc[p] += a - d
        h.devs[p].add(dev)
        h.last[p] = at
        h.changes[p].add(sha)
    h.dev_times[dev].append(at)
    for s in {_subsystem(p) for p, _, _ in files}:
        h.dev_subsys[dev][s] += 1


def _numstat_log(repo: Path, rev: str, author_key: str) -> list[tuple[str, str, int, list[tuple[str, int, int]]]]:
    who = "%an" if author_key == "name" else "%ae"
    out = _git(repo, "log", rev, "--reverse", "--topo-order", "--no-merges", "-M", "--numstat",
               f"--format=%x1e%x1eDIFFLAB %H%x00{who}%x00%at")
    commits = []
    for block in ("\n" + out).split("\n" + _SEP)[1:]:
        lines = block.split("\n")
        sha, dev, at = lines[0].split("\x00")
        files = []
        for ln in lines[1:]:
            if not ln.strip():
                continue
            a, d, p = ln.split("\t", 2)
            files.append((_renamed_to(p), 0 if a == "-" else int(a), 0 if d == "-" else int(d)))
        commits.append((sha, dev, int(at), files))
    return commits


def _renamed_to(p: str) -> str:
    if " => " not in p:
        return p
    m = re.match(r"(.*)\{(.*) => (.*)\}(.*)", p)
    if m:
        a, _, c, d = m.groups()
        return (a + c + d).replace("//", "/")
    return p.split(" => ", 1)[1]


def _patches(repo: Path, shas: list[str]) -> dict[str, dict]:
    """Message and per-file +/- lines of each target commit (unified=0, renames detected)."""
    out = _git(repo, "log", "--no-walk=unsorted", "--stdin", "-p", "--unified=0", "-M", "--no-color", "--no-ext-diff",
               "--format=%x1e%x1eDIFFLAB %H%x00%P%x00%aI%x00%B%x00", stdin="\n".join(shas) + "\n")
    res = {}
    for block in ("\n" + out).split("\n" + _SEP)[1:]:
        sha, parents, authored, rest = block.split("\x00", 3)
        msg, _, patch = rest.partition("\x00")
        files: dict[str, dict] = {}
        cur, in_hunk = None, False
        for ln in patch.split("\n"):
            if ln.startswith("diff --git "):
                cur, in_hunk = None, False
            elif not in_hunk and ln.startswith("+++ "):
                path = ln[4:]
                cur = path[2:] if path.startswith("b/") else None
                if cur is not None:
                    files.setdefault(cur, {"add": [], "delete": []})
            elif not in_hunk and ln.startswith("--- "):
                path = ln[4:]
                if path.startswith("a/"):
                    cur = path[2:]
                    files.setdefault(cur, {"add": [], "delete": []})
            elif ln.startswith("@@"):
                in_hunk = cur is not None
            elif in_hunk and ln.startswith("+"):
                files[cur]["add"].append(ln[1:])
            elif in_hunk and ln.startswith("-"):
                files[cur]["delete"].append(ln[1:])
        res[sha] = {"parent_sha": parents.split()[0] if parents.strip() else None, "authored_at": authored,
                    "message": msg.strip(), "files": files}
    return res


def _targets(repo: Path, rev: str, since: str | None, until: str | None, max_commits: int | None) -> list[str]:
    args = ["log", rev, "--no-merges", "--format=%H"]
    if since:
        args.append(f"--since={since}")
    if until:
        args.append(f"--until={until}")
    if max_commits:
        args.append(f"--max-count={max_commits}")
    return [x for x in _git(repo, *args).split() if x]


def extract_repo(repo: Path, project: str, rev: str = "HEAD", since: str | None = None, until: str | None = None,
                 max_commits: int | None = None, extensions: tuple[str, ...] = DEFAULT_EXTENSIONS,
                 author_key: str = "name", only: set[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return (dataset rows, metadata rows, per-repo report) for target commits of one repository."""
    if not (repo / ".git").exists() and not (repo / "HEAD").exists():
        raise ConfigError(f"{repo} is not a git repository")
    exts = tuple(e.lower() for e in extensions)
    targets = only if only is not None else set(_targets(repo, rev, since, until, max_commits))
    hist, feats = _History(), {}
    for sha, dev, at, files in _numstat_log(repo, rev, author_key):
        code = [f for f in files if _ext(f[0]) in exts]
        if not code:
            continue
        if sha in targets:
            feats[sha] = _features(hist, dev, at, code)
        _update(hist, sha, dev, at, code)
    patches = _patches(repo, sorted(feats)) if feats else {}
    rows, meta = [], []
    skipped = Counter({"merge_or_no_code_file": len(targets) - len(feats)})
    for sha, f in feats.items():
        p = patches[sha]
        adds, dels, langs = set(), set(), Counter()
        for path, lines in p["files"].items():
            ext = _ext(path)
            if ext not in exts:
                continue
            langs[LANGUAGE.get(ext, ext)] += 1
            adds.update(x for x in (clean_line(s, ext) for s in lines["add"]) if x)
            dels.update(x for x in (clean_line(s, ext) for s in lines["delete"]) if x)
        if not adds and not dels:
            skipped["no_code_lines_after_filter"] += 1
            continue
        f["fix"] = float(bool(_FIX.search(p["message"])))
        cid = f"{project}:{sha}"
        rows.append({"change_id": cid, "message": clean_message(p["message"]), "added_lines": sorted(adds),
                     "deleted_lines": sorted(dels), **{k: float(f[k]) for k in FEATURE_PROFILE_JIT14}})
        meta.append({"change_id": cid, "project": project, "commit_sha": sha, "parent_sha": p["parent_sha"],
                     "authored_at": p["authored_at"], "subject": p["message"].split("\n", 1)[0][:200],
                     "code_files": sorted(path for path in p["files"] if _ext(path) in exts),
                     "languages": ",".join(sorted(langs)), "primary_language": langs.most_common(1)[0][0] if langs else "",
                     "n_added_lines": len(adds), "n_deleted_lines": len(dels)})
    head = _git(repo, "rev-parse", rev).strip()
    report = {"project": project, "repo": str(repo), "rev": rev, "rev_sha": head, "since": since, "until": until,
              "max_commits": max_commits, "target_commits": len(targets), "extracted": len(rows), "skipped": dict(skipped)}
    return pd.DataFrame(rows), pd.DataFrame(meta), report


def extract(repos: list[tuple[Path, str]], out_dir: Path, rev: str, since: str | None, until: str | None,
            max_commits: int | None, extensions: tuple[str, ...], author_key: str) -> dict:
    """Write dataset.parquet (predict input, no labels), meta.parquet (labeling/breakdown) and extract-manifest.json."""
    if out_dir.exists():
        raise PolicyError(f"{out_dir} exists; extract into a fresh directory")
    names = [p for _, p in repos]
    if len(set(names)) != len(names):
        raise ConfigError(f"duplicate project names {names}; pass --project to disambiguate")
    data, meta, reports = [], [], []
    for repo, project in repos:
        d, m, r = extract_repo(repo, project, rev, since, until, max_commits, extensions, author_key)
        data.append(d)
        meta.append(m)
        reports.append(r)
    df = pd.concat(data, ignore_index=True) if data else pd.DataFrame()
    if df.empty:
        raise ExecutionError("no commits extracted; check --rev/--since/--until and --extensions")
    out_dir.mkdir(parents=True)
    df.to_parquet(out_dir / "dataset.parquet", index=False)
    pd.concat(meta, ignore_index=True).to_parquet(out_dir / "meta.parquet", index=False)
    manifest = {"extractor_version": EXTRACTOR_VERSION, "created_at": datetime.now(UTC).isoformat(),
                "extensions": list(extensions), "author_key": author_key, "repos": reports, "rows": int(len(df)),
                "rule_sha256": sha256_json({"split": _SPLIT.pattern, "fix": _FIX.pattern, "c_like": C_LIKE, "hash": HASH_COMMENT}),
                "dataset_sha256": sha256_file(out_dir / "dataset.parquet"), "labels": "none (label-free by construction)"}
    atomic_write_json(out_dir / "extract-manifest.json", manifest)
    return {"rows": manifest["rows"], "repos": [{k: r[k] for k in ("project", "target_commits", "extracted", "skipped")}
                                                 for r in reports]}
