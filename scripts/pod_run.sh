#!/bin/bash
# 012 on-pod sequence (H100): base arms -> diff CPT -> diff arms -> freeze -> one test; then the L2 extension study.
set -euo pipefail
export PATH=$HOME/.local/bin:$PATH
cd /workspace/jit012/codebert-diff-lab
C=configs/studies/diffllm-v1.yaml
V=data/diffllm/view-full-diff-v1
L2=configs/studies/public-comparison-v3-l2ext.yaml
mkdir -p logs
step() { echo "$(date -u +%FT%TZ) $1" >> logs/steps; }
step start
uv run diff-lab diffllm arm --config $C --arms R-base --view $V >> logs/arms.out 2>> logs/arms.err; step R-base
uv run diff-lab diffllm arm --config $C --arms E-base --view $V >> logs/arms.out 2>> logs/arms.err; step E-base
uv run diff-lab diffllm cpt --config $C >> logs/cpt.out 2>> logs/cpt.err; step cpt
uv run diff-lab diffllm arm --config $C --arms R-diff --view $V >> logs/arms.out 2>> logs/arms.err; step R-diff
uv run diff-lab diffllm arm --config $C --arms E-diff --view $V >> logs/arms.out 2>> logs/arms.err; step E-diff
uv run diff-lab diffllm freeze --config $C --view $V --out freeze-diffllm/freeze.json >> logs/freeze.out 2>> logs/freeze.err; step diffllm-frozen
uv run diff-lab diffllm test --config $C --view $V --freeze freeze-diffllm/freeze.json --out-dir test-diffllm >> logs/test.out 2>> logs/test.err; step diffllm-tested
uv run diff-lab experiment run --study $L2 --models L2-S --seeds 42,43,44 >> logs/l2.out 2>> logs/l2.err; step l2-valid
uv run diff-lab study freeze --study $L2 --out freeze-l2/freeze.json --device cuda >> logs/l2-freeze.out 2>> logs/l2-freeze.err; step l2-frozen
uv run diff-lab experiment test --study $L2 --freeze freeze-l2/freeze.json --out-dir test-l2 --device cuda >> logs/l2-test.out 2>> logs/l2-test.err; step l2-tested
step done
