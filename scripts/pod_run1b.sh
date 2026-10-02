#!/bin/bash
# Resume B3-S seed 44, then let pod_run2.sh continue. The first attempt died silently at 09:35 UTC: the 20 GB /workspace
# volume hit its quota while writing the epoch-2 checkpoint. artifacts/ now lives on the container disk (symlink).
export PATH=$HOME/.local/bin:$PATH
cd /workspace/jit016/codebert-diff-lab
step() { echo "$(date -u +%FT%TZ) $1" >> logs/steps; }
step "B3-S seed 44 restart"
uv run diff-lab experiment run --study configs/studies/public-comparison-v4-gitlines.yaml --models B3-S --seeds 44 >> logs/B3-S.out 2>> logs/B3-S.err
rc=$?
if [ $rc -ne 0 ]; then step "FAILED B3-S seed 44 rc=$rc"; exit $rc; fi
step "B3-S seed 44"
step done
