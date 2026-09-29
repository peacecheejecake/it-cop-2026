"""Convert JIT-Defects4J (JIT-Fine data.zip) labels to a riskbench build-apache CSV
without executing untrusted pickle code.

The archive ships only pickles. jitfine/features_{train,valid,test}.pkl are pandas
DataFrames; they are loaded with an Unpickler whose find_class allows exactly the
pandas/numpy reconstruction globals those files reference (checked with pickletools
first) and raises on anything else. Only repo, commit, label, original split and
timestamp are exported; author names/emails are dropped. Diffs are NOT taken from the
archive: build-apache re-extracts them from the public git repositories so the code
views match ApacheJIT's exactly.

Needs pandas (use a throwaway venv). Usage:
  python convert_jit_defects4j.py <data.zip> <out.csv>
"""
import csv
import pickle
import pickletools
import sys
import zipfile

ALLOW = {("pandas.core.frame", "DataFrame"), ("pandas.core.internals.managers", "BlockManager"),
         ("pandas.core.indexes.base", "_new_Index"), ("pandas.core.indexes.base", "Index"),
         ("pandas.core.indexes.range", "RangeIndex"), ("numpy", "ndarray"), ("numpy", "dtype"),
         ("numpy.core.multiarray", "_reconstruct"), ("numpy.core.numeric", "_frombuffer"),
         ("builtins", "slice")}


class AllowlistUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) not in ALLOW:
            raise pickle.UnpicklingError(f"blocked global {module}.{name}")
        return super().find_class(module, name)


def main():
    import io
    archive, out = sys.argv[1], sys.argv[2]
    rows = []
    with zipfile.ZipFile(archive) as z:
        for split in ("train", "valid", "test"):
            data = z.read(f"data/jitfine/features_{split}.pkl")
            pickletools.dis(data, out=io.StringIO())  # fails fast on malformed opcodes before loading
            df = AllowlistUnpickler(io.BytesIO(data)).load()
            for r in df[["project", "commit_hash", "is_buggy_commit", "author_date_unix_timestamp"]].itertuples(index=False):
                rows.append({"repo": f"apache/{r.project}", "project": r.project, "commit_hash": r.commit_hash,
                             "is_buggy_commit": "True" if int(float(r.is_buggy_commit)) else "False",
                             "orig_split": split, "author_date_unix_timestamp": r.author_date_unix_timestamp})
    if len({r["commit_hash"] for r in rows}) != len(rows):
        raise SystemExit("duplicate commit hashes across splits")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} commits, {sum(r['is_buggy_commit'] == 'True' for r in rows)} buggy -> {out}")


if __name__ == "__main__":
    main()
