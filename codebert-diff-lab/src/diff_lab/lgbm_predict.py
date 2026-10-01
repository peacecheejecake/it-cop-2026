"""Score a frozen LightGBM booster in a clean interpreter: `python -m diff_lab.lgbm_predict booster.txt in.parquet out.npy`.

LightGBM's libomp and torch's OpenMP runtime segfault when loaded into one process (exp 009),
so frozen B0-LGBM prediction never runs inside a process that may have imported torch.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd


def main() -> None:
    import lightgbm as lgb
    booster_path, inp, out = sys.argv[1:4]
    booster = lgb.Booster(model_file=booster_path)
    np.save(out, booster.predict(pd.read_parquet(inp)))


if __name__ == "__main__":
    main()
