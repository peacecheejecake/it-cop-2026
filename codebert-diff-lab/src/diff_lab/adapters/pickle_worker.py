"""Isolated JIT-Defects4J pickle converter. Run only via adapters.jit_defects4j.import_archive.

Executed as a separate process inside a network-denied sandbox with a scrubbed
environment. Unpickling goes through an allowlist Unpickler (the pandas/numpy
reconstruction globals found by M0 opcode inspection); anything else raises. Writes
schema_version=2 Parquet tables and a worker report; never reads the line-level
localisation pickle (ground truth that must not reach the commit classifier).
"""
from __future__ import annotations

import io
import json
import pickle
import pickletools
import re
import sys
from pathlib import Path

ALLOW = {("pandas.core.frame", "DataFrame"), ("pandas.core.internals.managers", "BlockManager"),
         ("pandas.core.indexes.base", "_new_Index"), ("pandas.core.indexes.base", "Index"),
         ("pandas.core.indexes.range", "RangeIndex"), ("numpy", "ndarray"), ("numpy", "dtype"),
         ("numpy.core.multiarray", "_reconstruct"), ("numpy.core.numeric", "_frombuffer"),
         ("builtins", "slice")}
FEATURES = ["ns", "nd", "nf", "entropy", "la", "ld", "lt", "fix", "ndev", "age", "nuc", "exp", "rexp", "sexp"]
SPLITS = ("train", "valid", "test")


class AllowlistUnpickler(pickle.Unpickler):
    """find_class is the enforcement point; it also records what was actually resolved."""

    def __init__(self, file: io.BytesIO):
        super().__init__(file)
        self.resolved: set[str] = set()

    def find_class(self, module: str, name: str):  # noqa: ANN201
        if (module, name) not in ALLOW:
            raise pickle.UnpicklingError(f"blocked global {module}.{name}")
        self.resolved.add(f"{module}.{name}")
        return super().find_class(module, name)


def safe_load(data: bytes):  # noqa: ANN201
    # Structural pre-check only: opcode stream must parse; arguments of STACK_GLOBAL come from
    # the stack/memo, so allowlisting is enforced in find_class rather than guessed here.
    for op, _, _ in pickletools.genops(data):
        if op.name in ("INST", "OBJ"):
            raise pickle.UnpicklingError(f"opcode {op.name} not allowed")
    u = AllowlistUnpickler(io.BytesIO(data))
    return u.load(), sorted(u.resolved)


def main(in_dir: str, out_dir: str) -> None:
    import hashlib

    import pandas as pd

    src, out = Path(in_dir), Path(out_dir)
    report: dict = {"globals": {}, "splits": {}, "anomalies": {}}
    changes, edits, feats, labels, splits = [], [], [], [], []
    for split in SPLITS:
        c, g1 = safe_load((src / f"changes_{split}.pkl").read_bytes())
        f, g2 = safe_load((src / f"features_{split}.pkl").read_bytes())
        report["globals"][split] = {"changes": g1, "features": g2}
        if not (isinstance(c, list) and len(c) == 4 and len({len(x) for x in c}) == 1):
            raise ValueError(f"changes_{split}: expected 4 aligned columns")
        ids, labs, msgs, codes = c
        if len(set(ids)) != len(ids) or f["commit_hash"].nunique() != len(f) or set(ids) != set(f["commit_hash"]):
            raise ValueError(f"{split}: ids not unique or changes/features not 1:1")
        fm = f.set_index("commit_hash")
        n_pos = 0
        for sha, lab, msg, code in zip(ids, labs, msgs, codes, strict=True):
            row = fm.loc[sha]
            if float(lab) != float(row["is_buggy_commit"]) or float(lab) not in (0.0, 1.0):
                raise ValueError(f"{split}:{sha} label mismatch between changes and features")
            if set(code) != {"added_code", "removed_code"}:
                raise ValueError(f"{split}:{sha} unexpected code keys {set(code)}")
            adds = sorted(str(x) for x in code["added_code"])
            dels = sorted(str(x) for x in code["removed_code"])
            project = str(row["project"])
            cid = f"jitd4j:{project}:{sha}"
            content = json.dumps([msg, adds, dels], ensure_ascii=False, separators=(",", ":"))
            parents = re.findall(r"\b[0-9a-f]{40}\b", str(row["parent_hashes"]))
            changes.append({"change_id": cid, "dataset_id": "jit-defects4j", "project_id": project,
                            "source_record_id": sha, "visibility": "public", "message": str(msg), "language": "java",
                            "content_hash": hashlib.sha256(content.encode()).hexdigest(),
                            "representation_kind": "preprocessed_lines", "commit_sha": sha,
                            "parent_sha": parents[0] if len(parents) == 1 else None, "committed_at": None,
                            "available_at": None, "authored_at_unix": int(str(row["author_date_unix_timestamp"])),
                            "time_provenance": "author_date_from_package_unverified", "source_uri": f"data/jitfine/changes_{split}.pkl"})
            for op, lines in (("add", adds), ("delete", dels)):
                edits.extend({"change_id": cid, "file_id": None, "hunk_id": None, "order": i, "op": op, "text": t}
                             for i, t in enumerate(lines))
            fr = {"change_id": cid, "feature_schema_id": "jit14-provided-v1", "provenance": "provided_unverified"}
            for k in FEATURES:
                v = row[k]
                if k == "fix":
                    if str(v) not in ("True", "False"):
                        raise ValueError(f"{sha}: fix={v!r}")
                    fr[k] = 1.0 if str(v) == "True" else 0.0
                else:
                    fr[k] = float(str(v))
            feats.append(fr)
            labels.append({"change_id": cid, "label": int(float(lab)), "label_kind": "defect_inducing",
                           "label_source": "jit-defects4j:is_buggy_commit", "label_version": "upstream-584799f",
                           "observed_at": None, "observation_end": None, "linkage_confidence": "unknown"})
            splits.append({"change_id": cid, "split": split, "split_version": "upstream-provided", "group_id": None})
            n_pos += int(float(lab))
        report["splits"][split] = {"n": len(ids), "positives": n_pos}
    fdf = pd.DataFrame(feats)
    report["anomalies"] = {k: int((fdf[k] < 0).sum()) for k in FEATURES if (fdf[k] < 0).any()}
    out.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(changes).to_parquet(out / "changes.parquet", index=False)
    pd.DataFrame(edits, columns=["change_id", "file_id", "hunk_id", "order", "op", "text"]).to_parquet(out / "edits.parquet", index=False)
    fdf.to_parquet(out / "features.parquet", index=False)
    pd.DataFrame(labels).to_parquet(out / "labels.parquet", index=False)
    pd.DataFrame(splits).to_parquet(out / "splits.parquet", index=False)
    (out / "worker-report.json").write_text(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
