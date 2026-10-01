"""JIT-Defects4J (JIT-Fine data.zip) import (spec FR-02/03, §5.1, AT-22).

The calling process never unpickles. It verifies the approved archive hash, rejects
unsafe zip members (path traversal, absolute paths, symlinks, oversized or highly
compressed entries), copies only the allowlisted pickles into a temp dir and runs
adapters.pickle_worker in a separate network-denied process with a scrubbed env.
"""
from __future__ import annotations

import platform
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import pandas as pd

from ..policy import SourceApproval
from ..util import IntegrityError, PolicyError, atomic_write_json, read_json, sha256_file, sha256_json

ADAPTER_VERSION = "jit-defects4j-adapter-v1"
SCHEMA_VERSION = 2
MEMBERS = {f"data/jitfine/{kind}_{split}.pkl": f"{kind}_{split}.pkl"
           for kind in ("changes", "features") for split in ("train", "valid", "test")}
MAX_MEMBER_BYTES = 200 * 1024 * 1024
MAX_COMPRESSION_RATIO = 200
TABLES = ("changes", "edits", "features", "labels", "splits")


def _safe_member(info: zipfile.ZipInfo) -> None:
    name = info.filename
    p = PurePosixPath(name)
    if name.startswith("/") or "\\" in name or ".." in p.parts:
        raise PolicyError(f"unsafe archive path: {name!r}")
    if stat.S_ISLNK(info.external_attr >> 16):
        raise PolicyError(f"symlink in archive: {name!r}")
    if info.file_size > MAX_MEMBER_BYTES:
        raise PolicyError(f"archive member too large: {name!r} {info.file_size}")
    if info.compress_size and info.file_size / info.compress_size > MAX_COMPRESSION_RATIO:
        raise PolicyError(f"suspicious compression ratio for {name!r}")


def extract_allowlisted(archive: Path, dest: Path) -> dict[str, str]:
    hashes = {}
    with zipfile.ZipFile(archive) as z:
        infos = {i.filename: i for i in z.infolist()}
        for info in infos.values():
            _safe_member(info)
        missing = set(MEMBERS) - set(infos)
        if missing:
            raise IntegrityError(f"archive missing members: {sorted(missing)}")
        for member, local in MEMBERS.items():
            data = z.read(infos[member])
            (dest / local).write_bytes(data)
            hashes[member] = sha256_file(dest / local)
    return hashes


def _sandbox_cmd(cmd: list[str]) -> tuple[list[str], str]:
    if platform.system() == "Darwin" and shutil.which("sandbox-exec"):
        return ["sandbox-exec", "-p", "(version 1)(allow default)(deny network*)", *cmd], "macos-sandbox-exec-deny-network"
    if platform.system() == "Linux" and shutil.which("unshare"):
        return ["unshare", "--user", "--net", "--map-root-user", *cmd], "linux-unshare-net"
    raise PolicyError("no network sandbox available for legacy pickle import (sandbox-exec/unshare)")


def _git_sha() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
                              cwd=Path(__file__).parent).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def import_archive(archive: str | Path, approval_path: str | Path, out_dir: str | Path) -> dict:
    archive, out = Path(archive), Path(out_dir)
    if out.exists():
        raise PolicyError(f"output {out} exists; snapshots are immutable")
    approval = SourceApproval.load(approval_path)
    archive_sha = approval.verify_archive(archive)
    with tempfile.TemporaryDirectory(prefix="diff-lab-import-") as tmp:
        tmp_p = Path(tmp)
        stage, result = tmp_p / "in", tmp_p / "out"
        stage.mkdir()
        member_hashes = extract_allowlisted(archive, stage)
        cmd, sandbox = _sandbox_cmd([sys.executable, "-m", "diff_lab.adapters.pickle_worker", str(stage), str(result)])
        env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_p), "PYTHONHASHSEED": "0",
               "PYTHONPATH": str(Path(__file__).resolve().parents[2]), "PYTHONDONTWRITEBYTECODE": "1"}
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1800)
        if proc.returncode != 0:
            raise IntegrityError(f"isolated pickle worker failed ({sandbox}):\n{proc.stderr[-4000:]}")
        report = read_json(result / "worker-report.json")
        out.mkdir(parents=True)
        for t in TABLES:
            shutil.copy2(result / f"{t}.parquet", out / f"{t}.parquet")
    return _write_manifest(out, approval, archive, archive_sha, member_hashes, report, sandbox)


def _write_manifest(out: Path, approval: SourceApproval, archive: Path, archive_sha: str,
                    member_hashes: dict, report: dict, sandbox: str) -> dict:
    t = {name: pd.read_parquet(out / f"{name}.parquet") for name in TABLES}
    ids = t["changes"]["change_id"]
    if not ids.is_unique:
        raise IntegrityError("duplicate change_id after import")
    for name in ("features", "labels", "splits"):
        if not t[name]["change_id"].is_unique or set(t[name]["change_id"]) != set(ids):
            raise IntegrityError(f"{name} is not 1:1 with changes")
    if not set(t["edits"]["change_id"]) <= set(ids):
        raise IntegrityError("edits reference unknown change_id")
    merged = t["splits"].merge(t["labels"][["change_id", "label"]], on="change_id").merge(
        t["changes"][["change_id", "project_id"]], on="change_id")
    split_stats = {s: {"n": int(len(g)), "positives": int(g["label"].sum()), "unique_ids": int(g["change_id"].nunique()),
                       "projects": int(g["project_id"].nunique()), "language": {"java": int(len(g))}}
                   for s, g in merged.groupby("split")}
    for s, st in split_stats.items():
        exp = report["splits"][s]
        if (st["n"], st["positives"]) != (exp["n"], exp["positives"]):
            raise IntegrityError(f"{s}: parquet counts differ from worker report")
    manifest = {
        "artifact_kind": "dataset_snapshot", "schema_version": SCHEMA_VERSION, "adapter_version": ADAPTER_VERSION,
        "dataset_id": "jit-defects4j", "visibility": "public", "privacy": "public",
        "source": {"source_id": approval.source_id, "origin_uri": approval.origin_uri, "archive_path": str(archive),
                   "archive_sha256": archive_sha, "archive_bytes": archive.stat().st_size,
                   "upstream_revision": approval.upstream_revision, "license_status": approval.license_status,
                   "approval_file_sha256": sha256_file(approval.path), "members_sha256": member_hashes},
        "imported_at": datetime.now(UTC).isoformat(), "code_git_sha": _git_sha(), "import_isolation": sandbox,
        "unpickle_globals": report["globals"], "representation_kind": "preprocessed_lines",
        "representation_notes": [
            "upstream stores added/removed lines as Python sets: original order and duplicate lines are lost",
            "edits.order is the rank in lexicographic sort (deterministic), not the patch order",
            "file and hunk mapping of lines is not available (file_id/hunk_id null)",
            "messages are upstream-tokenized text identical between changes and features pickles"],
        "excluded_upstream_content": {
            "changes_complete_buggy_line_level.pkl": "line-level localisation ground truth; never imported",
            "features.author_name/author_email": "personal data, not needed",
            "features.classification": "commit category derived from message; not in feature allowlist",
            "features.commit_message/fileschanged/author_date": "message duplicated in changes; others kept out of features"},
        "time": {"available": "author_date (unix)", "committed_at": "not provided", "provided_split_time_ordered": "no (M0 audit)"},
        "feature_schema": {"feature_schema_id": "jit14-provided-v1", "provenance": "provided_unverified",
                           "negative_value_counts": report["anomalies"]},
        "split_stats": split_stats, "allowed_uses": list(approval.approved_roles),
        "table_sha256": {name: sha256_file(out / f"{name}.parquet") for name in TABLES},
        "split_membership_sha256": sha256_json(sorted(zip(t["splits"]["change_id"], t["splits"]["split"], strict=True))),
    }
    atomic_write_json(out / "manifest.json", manifest)
    return manifest
