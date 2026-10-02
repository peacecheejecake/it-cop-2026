#!/usr/bin/env bash
# Offline install: bundled CPython 3.11 + wheels -> .venv, then unpack and verify the model bundle.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
echo "[1/5] checking package integrity (SHA256SUMS)"
sha256sum --quiet -c SHA256SUMS
echo "[2/5] extracting Python"
if [ ! -x runtime/python/bin/python3 ]; then
  mkdir -p runtime
  tar -xzf python/cpython-3.11.*-x86_64-unknown-linux-gnu-install_only.tar.gz -C runtime
fi
echo "[3/5] creating .venv and installing wheels (no network)"
[ -d .venv ] || runtime/python/bin/python3 -m venv .venv
.venv/bin/python -m pip install --quiet --no-index --find-links wheels -r wheels/requirements-offline.txt
.venv/bin/python -m pip install --quiet --no-index --no-deps wheels/codebert_diff_lab-*.whl
echo "[4/5] unpacking model bundle"
if [ ! -d bundle/unpacked ]; then
  .venv/bin/diff-lab bundle unpack --bundle bundle/bundle.tar.gz --out bundle/unpacked
else
  .venv/bin/diff-lab bundle verify --bundle-dir bundle/unpacked
fi
echo "[5/5] environment"
.venv/bin/python - <<'PY'
import torch
print(f"torch {torch.__version__} (CUDA build {torch.version.cuda}); cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
echo "install ok. next: ./smoke.sh"
