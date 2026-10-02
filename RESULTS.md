# RESULTS — 017-dl-d1-learning-curve (validation only, exploratory)

| train fraction | train rows | B0-LR | B1-TFIDF-S | B2-S on MPS (seeds 42/43/44) | B2-S mean |
|---|---|---|---|---|---|
| 25% | 4,032 | 0.291 | 0.318 | 0.345 / 0.342 / 0.332 | 0.340 |
| 50% | 8,143 | 0.297 | 0.344 | 0.342 / 0.356 / 0.343 | 0.347 |
| 100% | 16,119 | 0.302 | 0.345 | 0.320 / 0.315 / 0.289 | 0.308 |

- **B0-LR and B1 (deterministic, CPU) are flat from 50% to 100%** (+0.005 and +0.001). For these two, more data of the same kind is unlikely to help.
- **The B2-S column is not usable as a learning curve.** The 100% runs on MPS (0.308) disagree with the same configuration on the H100 in study v4 (0.372 ± 0.005). Only the device differs, so MPS training runs cannot be compared with CUDA runs. The cause was not investigated.
- B2-S validation AP also moves by 0.02–0.03 from one epoch to the next (v4 seed 42: 0.359, 0.346, 0.347, 0.334, 0.324, 0.367, …), so its best-epoch value is partly selection noise.
- **Open:** rerun the nine B2-S runs (and B3-S if wanted) on CUDA; about 40 minutes on one GPU. The D2 gate (is the curve still rising at 100%?) is not passed for B0-LR/B1 and undecided for the encoder variants.
