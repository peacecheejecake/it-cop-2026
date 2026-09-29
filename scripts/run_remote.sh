#!/usr/bin/env bash
# 003 on a Runpod CUDA pod. Run from <bundle>/baseline after unpacking the bundle (see RUNLOG).
set -euo pipefail
SEEDS="${SEEDS:-42 43 44}"
L=runs/logs; mkdir -p "$L"
step() { echo "$(date '+%F %T') $*" | tee -a "$L/pipeline.log"; }

step "env"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv | tee "$L/gpu.txt"
python -m pip install -q -e '.[neural,dev]' 2>&1 | tail -3
python -c "import torch,transformers;print('torch',torch.__version__,'cuda',torch.version.cuda,'tf32',torch.backends.cuda.matmul.allow_tf32,'transformers',transformers.__version__)" | tee -a "$L/gpu.txt"
python -m pip freeze > "$L/requirements-lock.txt"
pytest -q 2>&1 | tail -2 | tee -a "$L/pipeline.log"

step "train tabular"
riskbench train-tabular --public-train data/views/public/train --public-valid data/views/public/valid \
  --out runs/models/tabular > "$L/train-tabular.log" 2>&1

for seed in $SEEDS; do
  M=runs/models/seed$seed; mkdir -p "$M"; ln -sfn ../tabular "$M/tabular"
  for mode in frozen finetune; do
    step "seed $seed train $mode"
    python ../tools/train_with_progress.py --every 100 --mode $mode --model models/codebert-base \
      --public-train data/views/public/train --public-valid data/views/public/valid --out "$M/$mode" \
      --epochs 100 --patience 2 --batch-size 32 --accumulation 1 --encoder-lr 2e-5 --head-lr 1e-3 \
      --seed $seed --device cuda > "$L/train-$mode-seed$seed.log" 2>&1
  done
  step "seed $seed predict_suite"
  python scripts/predict_suite.py --dataset data/views/public/test --models rule tabular frozen finetune \
    --model-root "$M" --device cuda --batch-size 64 --out runs/evaluation/seed$seed --bootstrap 500 \
    > "$L/predict-seed$seed.log" 2>&1
done
step "done"
