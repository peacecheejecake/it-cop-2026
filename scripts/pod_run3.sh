#!/bin/bash
# After the four B4/B5 lanes: L0-S, L1-S on validation, then freeze all 9 variants and score the public test once.
export PATH=$HOME/.local/bin:$PATH
cd /workspace/jit016/codebert-diff-lab
S=configs/studies/public-comparison-v4-gitlines.yaml
step() { echo "$(date -u +%FT%TZ) [L] $1" >> logs/steps; }
until [ "$(grep -c '\] lane done' logs/steps)" -ge 4 ] || grep -q FAILED logs/steps; do sleep 20; done
grep -q FAILED logs/steps && { step "not started: a lane failed"; exit 1; }
grep -q QWEN_PINS_OK /workspace/qwen.log || { step "FAILED qwen pins"; exit 1; }
run() { name=$1; shift; step "start $name"; "$@" >> logs/$name.out 2>> logs/$name.err; rc=$?; [ $rc -ne 0 ] && { step "FAILED $name rc=$rc"; exit $rc; }; step "$name"; }
run L0-S uv run --no-sync diff-lab experiment run --study $S --models L0-S --seeds 42
run L1-S uv run --no-sync diff-lab experiment run --study $S --models L1-S --seeds 42,43,44
run freeze uv run --no-sync diff-lab study freeze --study $S --out freeze/freeze.json --device cuda
run test uv run --no-sync diff-lab experiment test --study $S --freeze freeze/freeze.json --out-dir test-results --device cuda
step done3
