#!/usr/bin/env bash
# Thin wrapper around diff-lab with the package paths filled in. Usage: see README.md.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DL="$HERE/.venv/bin/diff-lab"
[ -x "$DL" ] || { echo "run ./install.sh first" >&2; exit 2; }
cmd="${1:-}"; shift || true
case "$cmd" in
  extract)  # ./jit.sh extract <out_dir> --repo <path> [--repo <path> ...] [--since YYYY-MM-DD] [--until ...] [--extensions .java,.kt]
    out="$1"; shift
    exec "$DL" internal extract --out "$out" "$@" ;;
  sheet)    # ./jit.sh sheet <extract_dir> <labels.csv> [--sample N]
    ex="$1"; csv="$2"; shift 2
    exec "$DL" internal label-sheet --extract-dir "$ex" --out "$csv" "$@" ;;
  predict)  # ./jit.sh predict <extract_dir> <pred_dir> [cuda|cpu]
    ex="$1"; pred="$2"; dev="${3:-cuda}"
    exec "$DL" predict --bundle-dir "$HERE/bundle/unpacked" --dataset "$ex/dataset.parquet" --out-dir "$pred" \
      --device "$dev" --encoder-path "$HERE/models/codebert-base" --variants B3-S ;;
  eval)     # ./jit.sh eval <extract_dir> <pred_dir> <labels.csv> <eval_dir>
    ex="$1"; pred="$2"; csv="$3"; out="$4"
    exec "$DL" internal label-eval --predictions "$pred/predictions.parquet" --labels "$csv" --meta "$ex/meta.parquet" \
      --bundle-dir "$HERE/bundle/unpacked" --out-dir "$out" ;;
  *)
    echo "usage: ./jit.sh {extract|sheet|predict|eval} ...  (README.md)" >&2; exit 2 ;;
esac
