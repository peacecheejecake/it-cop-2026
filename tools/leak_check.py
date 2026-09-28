"""Near-duplicate check between public splits (baseline only dedupes exact patches).

A change is keyed by the hash of its +/- line contents with whitespace removed and
file headers / hunk line numbers ignored, so the same edit re-applied elsewhere
(backport, cherry-pick onto another branch, re-commit) collides. Also checks
identical normalized commit messages. Reports how many eval rows collide with train
and how often the colliding train row carries the same label.
Usage: python leak_check.py <splits_dir> [--write-exclusions <out.json>]
"""
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


def change_key(diff: str) -> str | None:
    body = [re.sub(r"\s+", "", l[1:]) for l in diff.splitlines()
            if l[:1] in "+-" and not l.startswith(("+++", "---"))]
    body = [l for l in body if l]
    return hashlib.sha256("\n".join(body).encode()).hexdigest() if body else None


def msg_key(msg: str) -> str | None:
    m = re.sub(r"\s+", " ", msg.strip().lower())
    return hashlib.sha256(m.encode()).hexdigest() if len(m) >= 20 else None


def load(split_dir: Path):
    labels = {json.loads(l)["id"]: json.loads(l)["label"] for l in open(split_dir / "labels.jsonl")}
    rows = {}
    with open(split_dir / "inputs.jsonl") as f:
        for l in f:
            r = json.loads(l)
            rows[r["id"]] = (change_key(r["diff"]), msg_key(r["message"]), labels[r["id"]])
    return rows


def main():
    root = Path(sys.argv[1]); write = sys.argv[sys.argv.index("--write-exclusions") + 1] if "--write-exclusions" in sys.argv else None
    train = load(root / "train")
    by_change = defaultdict(list); by_msg = defaultdict(list)
    for ck, mk, y in train.values():
        if ck: by_change[ck].append(y)
        if mk: by_msg[mk].append(y)
    report = {}
    for name in ("valid", "test"):
        ev = load(root / name)
        out = {"n": len(ev), "positives": sum(y for *_, y in ev.values())}
        for kind, index, pos in (("change", by_change, 0), ("message", by_msg, 1)):
            hits = {i: v for i, v in ev.items() if v[pos] and v[pos] in index}
            same = sum(1 for v in hits.values() if round(sum(index[v[pos]]) / len(index[v[pos]])) == v[2])
            out[kind] = {"rows_matching_train": len(hits), "rate": round(len(hits) / len(ev), 4),
                         "positives_matching": sum(v[2] for v in hits.values()),
                         "label_agrees_with_train_majority": same}
            out[kind]["_ids"] = set(hits)
        report[name] = out
    if write:
        test_hits = report["test"]["change"]["_ids"] | report["test"]["message"]["_ids"]
        report["test"]["excluding_matches_n"] = report["test"]["n"] - len(test_hits)
        report["test"]["_exclude"] = sorted(test_hits)
    for name in report:
        for kind in ("change", "message"):
            report[name][kind].pop("_ids")
    exclude = report["test"].pop("_exclude", [])
    print(json.dumps(report, indent=2))
    if write:
        Path(write).write_text(json.dumps(exclude))
        print(f"wrote {len(exclude)} ids to {write}", file=sys.stderr)


if __name__ == "__main__":
    main()
