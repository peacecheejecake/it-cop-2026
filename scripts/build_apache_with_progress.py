"""Run `riskbench build-apache` unchanged, but log progress to stderr.

build-apache is silent until it finishes (~2h for full ApacheJIT); this wraps
riskbench.data.extract_commit to print a progress line every N commits.
Usage: python build_apache_with_progress.py <build-apache args...>
"""
import csv
import sys
import time

import riskbench.data as data
from riskbench.cli import main

EVERY = 500
args = sys.argv[1:]
csv_path = args[args.index("--csv") + 1]
with open(csv_path, encoding="utf-8-sig", newline="") as f:
    total = sum(1 for _ in csv.DictReader(f))
if "--limit" in args:
    total = min(total, int(args[args.index("--limit") + 1]))

_orig = data.extract_commit
state = {"n": 0, "err": 0, "t0": time.monotonic()}


def extract_commit(repo, commit, first_parent=False):
    state["n"] += 1
    n = state["n"]
    try:
        return _orig(repo, commit, first_parent)
    except Exception:
        state["err"] += 1
        raise
    finally:
        if n % EVERY == 0 or n == total:
            el = time.monotonic() - state["t0"]
            eta = (total - n) * el / n
            print(f"{time.strftime('%H:%M:%S')} {n:>6}/{total} ({100 * n / total:5.1f}%) "
                  f"repo={repo.name:<24} rejected_in_extract={state['err']:<5} "
                  f"{n / el * 60:6.0f}/min ETA {eta / 60:5.1f}m", file=sys.stderr, flush=True)


data.extract_commit = extract_commit
sys.argv = ["riskbench", "build-apache", *args]
main()
