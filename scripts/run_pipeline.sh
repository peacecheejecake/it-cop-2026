#!/usr/bin/env bash
# 002 pipeline after split-public; run from experiments/002-no-hadoop/baseline with the venv active.
set -euo pipefail
L=runs/logs; mkdir -p "$L"
step() { echo "$(date '+%F %T') $*" | tee -a "$L/pipeline.log"; }

step "prepare train/valid/test"
for s in train valid test; do
  riskbench prepare --dataset data/splits/public/$s --out data/views/public/$s \
    --tokenizer models/codebert-base > "$L/prepare-$s.log" 2>&1 &
done
wait

step "train tabular (cpu) in background"
riskbench train-tabular --public-train data/views/public/train --public-valid data/views/public/valid \
  --out runs/models/tabular > "$L/train-tabular.log" 2>&1 &
TAB=$!

step "train A frozen (mps)"
python ../tools/train_with_progress.py --every 200 --mode frozen --model models/codebert-base \
  --public-train data/views/public/train --public-valid data/views/public/valid --out runs/models/frozen \
  --epochs 5 --batch-size 8 --accumulation 4 --seed 42 --device mps > "$L/train-frozen.log" 2>&1

step "train B finetune (mps)"
python ../tools/train_with_progress.py --every 200 --mode finetune --model models/codebert-base \
  --public-train data/views/public/train --public-valid data/views/public/valid --out runs/models/finetune \
  --epochs 5 --batch-size 8 --accumulation 4 --encoder-lr 2e-5 --head-lr 1e-3 --seed 42 --device mps \
  > "$L/train-finetune.log" 2>&1
wait "$TAB"

step "predict_suite on public test"
python scripts/predict_suite.py --dataset data/views/public/test --models rule tabular frozen finetune \
  --model-root runs/models --device mps --out runs/evaluation/public-test --bootstrap 500 \
  > "$L/predict-public-test.log" 2>&1

step "analysis"
python ../tools/leak_check.py data/splits/public --write-exclusions runs/test_near_dup_ids.json > "$L/leak_check.json"
python ../tools/sensitivity.py runs/evaluation/public-test data/splits/public runs/test_near_dup_ids.json \
  runs/evaluation/public-test-sensitivity.json > /dev/null
python ../tools/paired_bootstrap_subset.py runs/evaluation/public-test data/splits/public/test tabular finetune \
  > runs/evaluation/paired_bootstrap_subset.txt
step "done"
