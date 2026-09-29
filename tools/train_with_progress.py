"""Run `riskbench train-codebert` unchanged, but log step-level progress to stderr.

riskbench only prints at epoch end (~2h per fine-tuning epoch on MPS); this wraps
riskbench.neural.train_epoch's microbatch loop to print progress/throughput/ETA every N
optimizer steps. The original train_epoch runs as-is.

--amp bf16 runs each training epoch under torch.autocast(bfloat16): weights, optimizer
state and the BCE-with-logits loss stay fp32 (autocast keeps that op in fp32), and no
GradScaler is needed because bf16 has fp32's exponent range. Validation scoring and
embedding extraction stay fp32. The choice is written to <out>/training_wrapper.json
(model.json is sealed by riskbench's manifest, so it is not edited).
Usage: python train_with_progress.py [--every N] [--amp none|bf16] <train-codebert args...>
"""
import json
import sys
import time
from pathlib import Path

import torch

import riskbench.neural as neural
from riskbench.cli import main

args = sys.argv[1:]
every, amp = 50, "none"
while args[:1] in (["--every"], ["--amp"]):
    if args[0] == "--every":
        every = int(args[1])
    else:
        amp = args[1]
    args = args[2:]
if amp not in ("none", "bf16"):
    raise SystemExit(f"--amp must be none or bf16, got {amp!r}")

_orig_train_epoch = neural.train_epoch
state = {"epoch": 0}


class ProgressLoader:
    def __init__(self, loader, accumulation):
        self.loader, self.accumulation = loader, accumulation

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        total = len(self.loader); t0 = time.monotonic(); seen = 0
        for i, batch in enumerate(self.loader, 1):
            yield batch
            seen += len(batch["labels"])
            if i % (every * self.accumulation) == 0 or i == total:
                el = time.monotonic() - t0
                print(f"{time.strftime('%H:%M:%S')} epoch={state['epoch']} batch {i}/{total} "
                      f"({100 * i / total:5.1f}%) {seen / el:6.1f} ex/s "
                      f"ETA {(total - i) * el / i / 60:6.1f}m", file=sys.stderr, flush=True)


def train_epoch(model, loader, optimizer, device, accumulation=1, pos_weight=None):
    state["epoch"] += 1
    loader = ProgressLoader(loader, accumulation)
    if amp == "bf16":
        with torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16):
            return _orig_train_epoch(model, loader, optimizer, device, accumulation, pos_weight)
    return _orig_train_epoch(model, loader, optimizer, device, accumulation, pos_weight)


neural.train_epoch = train_epoch
sys.argv = ["riskbench", "train-codebert", *args]
started = time.time()
main()
out = Path(args[args.index("--out") + 1])
(out / "training_wrapper.json").write_text(json.dumps({
    "wrapper": "tools/train_with_progress.py", "amp": amp, "every": every,
    "wall_seconds": round(time.time() - started, 1), "torch": torch.__version__}, indent=2))
