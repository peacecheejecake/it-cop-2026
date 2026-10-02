#!/bin/bash
# 016 pod setup: unpack the bundle (code @ CODE_SHA, jitd4j-git1 snapshot/split/evidence, pinned CodeBERT), build env.
set -euo pipefail
export PATH=$HOME/.local/bin:$PATH
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
mkdir -p /workspace/jit016 && cd /workspace/jit016
tar --no-same-owner --exclude='._*' -xzf /workspace/jit016-bundle.tar.gz
cd codebert-diff-lab
uv python install 3.11
uv sync --extra neural --extra lgbm --extra dev
uv run diff-lab doctor
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
