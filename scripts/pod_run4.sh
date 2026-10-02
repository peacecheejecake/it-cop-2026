#!/bin/bash
# The first freeze refused: B0-LR, B0-LGBM, B1-TFIDF-S had only seed 42 (run locally). Run seeds 43/44 here (CPU), then freeze and test once.
export PATH=$HOME/.local/bin:$PATH
cd /workspace/jit016/codebert-diff-lab
S=configs/studies/public-comparison-v4-gitlines.yaml
step() { echo "$(date -u +%FT%TZ) [F] $1" >> logs/steps; }
run() { name=$1; shift; step "start $name"; "$@" >> logs/$name.out 2>> logs/$name.err; rc=$?; [ $rc -ne 0 ] && { step "FAILED $name rc=$rc"; exit $rc; }; step "$name"; }
run cpu-baselines uv run --no-sync diff-lab experiment run --study $S --models B0-LR,B0-LGBM,B1-TFIDF-S --seeds 43,44
run freeze2 uv run --no-sync diff-lab study freeze --study $S --out freeze/freeze.json --device cuda
run test uv run --no-sync diff-lab experiment test --study $S --freeze freeze/freeze.json --out-dir test-results --device cuda
step done4
