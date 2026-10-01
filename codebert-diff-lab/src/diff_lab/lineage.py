"""Cross-artifact lineage validation before any fit (spec FR-02/04, AT-04).

Each artifact already checks its own file hashes; this module checks that the pieces belong
together: approval -> snapshot -> split -> evidence -> tokenizer, and that the controlled
split kept the provided valid/test membership (no eval row re-labelled as train).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .policy import SourceApproval
from .util import IntegrityError, PolicyError, read_json, sha256_file


def validate_lineage(approval_path: Path, snapshot_dir: Path, split_dir: Path, evidence_dir: Path | None,
                     tokenizer_digest: str | None, required_roles: tuple[str, ...]) -> dict:
    approval = SourceApproval.load(approval_path)
    for role in required_roles:
        approval.require_role(role)
    snap_m = read_json(snapshot_dir / "manifest.json")
    src = snap_m.get("source", {})
    if src.get("source_id") != approval.source_id or src.get("archive_sha256") != approval.archive_sha256:
        raise PolicyError("snapshot was not imported from the approved source archive")
    if src.get("approval_file_sha256") != sha256_file(approval_path):
        raise PolicyError("approval file changed since the snapshot was imported; re-import under the new approval")
    if snap_m.get("visibility") != approval.visibility:
        raise PolicyError("snapshot visibility does not match its approval")
    snap_sha = sha256_file(snapshot_dir / "manifest.json")
    split_m = read_json(split_dir / "manifest.json")
    if split_m.get("parent_snapshot_manifest_sha256") != snap_sha:
        raise IntegrityError("split was built from a different snapshot manifest")
    split_sha = sha256_file(split_dir / "manifest.json")
    provided = pd.read_parquet(snapshot_dir / "splits.parquet")[["change_id", "split"]]
    controlled = pd.read_parquet(split_dir / "splits.parquet")[["change_id", "split"]]
    for s in ("valid", "test"):
        if set(provided.loc[provided.split == s, "change_id"]) != set(controlled.loc[controlled.split == s, "change_id"]):
            raise IntegrityError(f"controlled split changed the provided {s} membership")
    if not set(controlled.loc[controlled.split == "train", "change_id"]) <= set(provided.loc[provided.split == "train", "change_id"]):
        raise IntegrityError("controlled train contains rows that are not provided-train")
    out = {"approval_sha256": sha256_file(approval_path), "approval_source_id": approval.source_id}
    if evidence_dir is not None:
        ev_m = read_json(evidence_dir / "manifest.json")
        if ev_m.get("snapshot_manifest_sha256") != snap_sha or ev_m.get("split_manifest_sha256") != split_sha:
            raise IntegrityError("evidence was built from a different snapshot/split")
        if tokenizer_digest is not None and ev_m.get("tokenizer", {}).get("files_sha256") != tokenizer_digest:
            raise IntegrityError("evidence was tokenized with different tokenizer files than the pinned snapshot")
    return out
