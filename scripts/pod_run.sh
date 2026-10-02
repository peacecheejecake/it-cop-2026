#!/bin/bash
# 016 on-pod: B3-S seeds 42/43/44 on public validation (study v4-gitlines). Test stays sealed.
set -euo pipefail
export PATH=$HOME/.local/bin:$PATH
cd /workspace/jit016/codebert-diff-lab
mkdir -p logs
step() { echo "$(date -u +%FT%TZ) $1" >> logs/steps; }
step start
for s in 42 43 44; do
  uv run diff-lab experiment run --study configs/studies/public-comparison-v4-gitlines.yaml --models B3-S --seeds $s >> logs/B3-S.out 2>> logs/B3-S.err
  step "B3-S seed $s"
done
step done
