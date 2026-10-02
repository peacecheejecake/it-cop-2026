#!/bin/bash
# 018 step 1: label-free CPT probe on a small GPU pod. Unpack, build env, download pinned Qwen 7B, run probe-cpt.
set -euo pipefail
export PATH=$HOME/.local/bin:$PATH
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
mkdir -p /root/jit018 && cd /root/jit018
tar --no-same-owner --exclude='._*' -xzf /root/jit018-bundle.tar.gz
cd codebert-diff-lab
uv python install 3.11
uv sync --extra neural --extra dev
uv run --no-sync python - <<'PY'
import yaml
from huggingface_hub import snapshot_download
m = yaml.safe_load(open("configs/studies/study-m-stage1.yaml"))["models"]["qwen7b"]
snapshot_download(m["id"], revision=m["revision"], local_dir="/root/jit018/experiments/.cache/models/qwen2.5-coder-7b-instruct")
PY
uv run --no-sync diff-lab diffllm probe-cpt --config configs/studies/study-m-stage1.yaml --view data/diffllm-view/study-m-stage1 \
  --cpt qwen7b-apache50m --rows 512 --out results/probe-cpt.json --artifacts-dir artifacts
echo PROBE_DONE
