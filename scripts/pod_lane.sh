#!/bin/bash
# One lane of sequential runs; several lanes share the single H100 (a run uses ~8 GB and one CPU core).
# usage: pod_lane.sh <lane-name> <variant>:<seed> [...]
export PATH=$HOME/.local/bin:$PATH
cd /workspace/jit016/codebert-diff-lab
lane=$1; shift
step() { echo "$(date -u +%FT%TZ) [$lane] $1" >> logs/steps; }
for job in "$@"; do
  m=${job%%:*}; s=${job##*:}
  step "start $m seed $s"
  uv run --no-sync diff-lab experiment run --study configs/studies/public-comparison-v4-gitlines.yaml --models $m --seeds $s >> logs/$m-$s.out 2>> logs/$m-$s.err
  rc=$?
  if [ $rc -ne 0 ]; then step "FAILED $m seed $s rc=$rc"; exit $rc; fi
  step "$m seed $s"
done
step "lane done"
