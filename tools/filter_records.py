"""Drop whole repositories from a canonical records.jsonl before split-public.

Streams the (multi-GB) input line by line. Writes records.jsonl plus
filter_report.json recording the source hash, excluded repositories, and counts, so
the derived dataset stays traceable to the shared canonical build.
Usage: python filter_records.py <in_records.jsonl> <out_dir> --exclude-repos owner/repo ...
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("records"); p.add_argument("out")
    p.add_argument("--exclude-repos", nargs="+", required=True)
    a = p.parse_args()
    src, out = Path(a.records), Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    excluded = set(a.exclude_repos); kept = Counter(); dropped = Counter(); pos = Counter()
    with src.open() as fin, (out / "records.jsonl").open("w") as fout:
        for line in fin:
            r = json.loads(line); repo = r["repository"]
            if repo in excluded:
                dropped[repo] += 1
                continue
            kept[repo] += 1; pos[repo] += r["label"]
            fout.write(line)
    unknown = excluded - set(dropped)
    if unknown:
        raise SystemExit(f"excluded repositories not present in source: {sorted(unknown)}")
    report = {"source_records": str(src.resolve()), "source_sha256": sha256(src),
              "source_build_report": json.loads((src.parent / "build_report.json").read_text()),
              "excluded_repositories": sorted(excluded), "dropped": dict(dropped),
              "kept_total": sum(kept.values()), "kept": {k: {"n": v, "positives": pos[k]} for k, v in sorted(kept.items())},
              "output_sha256": sha256(out / "records.jsonl")}
    (out / "filter_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("excluded_repositories", "dropped", "kept_total")}, indent=2))


if __name__ == "__main__":
    main()
