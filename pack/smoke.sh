#!/usr/bin/env bash
# End-to-end self test on a throwaway git repo: extract -> sheet -> predict -> fake labels -> eval.
# Usage: ./smoke.sh [cuda|cpu]   (default cpu). Uses a temp dir and deletes it afterwards.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DEV="${1:-cpu}"
T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT
R="$T/demo-repo"
mkdir -p "$R/src/main/java/demo" && cd "$R"
git init -q
c() { git add -A
      GIT_AUTHOR_NAME="$1" GIT_AUTHOR_EMAIL="$1@example" GIT_COMMITTER_NAME="$1" GIT_COMMITTER_EMAIL="$1@example" \
      GIT_AUTHOR_DATE="$2" GIT_COMMITTER_DATE="$2" git commit -q -m "$3"; }
F=src/main/java/demo/Calc.java
printf 'package demo;\npublic class Calc {\n  public int add(int a, int b) { return a + b; }\n}\n' > $F; c alice "2024-01-01T10:00:00" "initial calculator"
printf 'package demo;\npublic class Calc {\n  public int add(int a, int b) { return a + b; }\n  public int div(int a, int b) { return a / b; }\n}\n' > $F; c bob "2024-01-05T10:00:00" "add division"
printf 'package demo;\npublic class Calc {\n  public int add(int a, int b) { return a + b; }\n  public int div(int a, int b) {\n    if (b == 0) throw new IllegalArgumentException("b");\n    return a / b;\n  }\n}\n' > $F; c alice "2024-02-01T10:00:00" "fix division by zero"
printf '# demo\n' > README.md; c bob "2024-02-02T10:00:00" "docs only"
cd "$HERE"
./jit.sh extract "$T/extract" --repo "$R" --project demo
./jit.sh sheet "$T/extract" "$T/labels.csv"
./jit.sh predict "$T/extract" "$T/pred" "$DEV"
.venv/bin/python - "$T/labels.csv" <<'PY'
import sys, pandas as pd
p = sys.argv[1]
d = pd.read_csv(p, dtype=str, keep_default_na=False, encoding="utf-8-sig")
d["label"] = ["1" if s == "add division" else "0" for s in d["subject"]]
d.to_csv(p, index=False, encoding="utf-8-sig")
PY
./jit.sh eval "$T/extract" "$T/pred" "$T/labels.csv" "$T/eval"
test -s "$T/eval/label-eval.json" && test -s "$T/eval/scored-labeled.csv"
echo "smoke ok (device=$DEV)"
