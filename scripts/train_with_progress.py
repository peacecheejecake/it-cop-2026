"""Run `riskbench train-codebert` unchanged, but log step-level progress to stderr.

riskbench only prints at epoch end (~2h per fine-tuning epoch on MPS); this wraps
riskbench.neural.train_epoch's microbatch loop to print progress/throughput/ETA every N
optimizer steps. Training math is untouched: the original train_epoch runs as-is.
Usage: python train_with_progress.py [--every N] <train-codebert args...>
"""
import sys
import time

import riskbench.neural as neural
from riskbench.cli import main

args = sys.argv[1:]
every = 50
if args[:1] == ["--every"]:
    every, args = int(args[1]), args[2:]

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
    return _orig_train_epoch(model, ProgressLoader(loader, accumulation), optimizer, device,
                             accumulation, pos_weight)


neural.train_epoch = train_epoch
sys.argv = ["riskbench", "train-codebert", *args]
main()
