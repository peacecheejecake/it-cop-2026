#!/bin/bash
# 016 on-pod, step 4: after pod_run.sh (B3-S) is done, B2-S, B4-S, B5-S seeds 42/43/44 on public validation.
set -euo pipefail
export PATH=$HOME/.local/bin:$PATH
cd /workspace/jit016/codebert-diff-lab
step() { echo "$(date -u +%FT%TZ) $1" >> logs/steps; }
until grep -q " done$" logs/steps; do sleep 30; done
# The extended registration (commit 7a25603) replaces the config only after the B3-S loop has finished.
cp /workspace/v4.yaml configs/studies/public-comparison-v4-gitlines.yaml
for m in B2-S B4-S B5-S; do
  for s in 42 43 44; do
    uv run diff-lab experiment run --study configs/studies/public-comparison-v4-gitlines.yaml --models $m --seeds $s >> logs/$m.out 2>> logs/$m.err
    step "$m seed $s"
  done
done
step done2
