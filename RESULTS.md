# RESULTS — 017-dl-d1-learning-curve (validation only, exploratory)

| train fraction | train rows | B0-LR | B1-TFIDF-S | B2-S on MPS (seeds 42/43/44) | B2-S mean |
|---|---|---|---|---|---|
| 25% | 4,032 | 0.291 | 0.318 | 0.345 / 0.342 / 0.332 | 0.340 |
| 50% | 8,143 | 0.297 | 0.344 | 0.342 / 0.356 / 0.343 | 0.347 |
| 100% | 16,119 | 0.302 | 0.345 | 0.320 / 0.315 / 0.289 | 0.308 |

- **B0-LR and B1 (deterministic, CPU) are flat from 50% to 100%** (+0.005 and +0.001). For these two, more data of the same kind is unlikely to help.
- **The B2-S column is not usable as a learning curve: local training and H100 training of the same B2-S configuration give different results.** The 100% runs here (0.308) disagree with v4 on the H100 (0.372 ± 0.005).

### Is the data different? No (checked 2026-10-03, `scripts/b2_head_platform_check.py`)

| check | result |
|---|---|
| data lineage of the 017 f100 and the 016 v4 runs (snapshot, split, evidence manifests and digests) | identical |
| B0-LR, B1-TFIDF-S (deterministic, same loader) | identical to v4 to 4 decimals (0.3021, 0.3447) |
| encoder init, head init, structured-pipeline state hashes | identical |
| frozen CodeBERT CLS embeddings, Mac CPU vs Mac MPS (64 changes) | max abs diff 9.5e-6 |
| v4's H100-trained seed-42 head applied to locally computed embeddings and features | reproduces the stored H100 validation scores to 1.6e-5, AP 0.3668 |
| head training on the same cached embeddings, Mac CPU vs Mac MPS, seed 42, 6 epochs | same trajectory (epoch 1: 0.3201 vs 0.3202) |
| head training on Mac CPU, seeds 40–51, v4 early stopping | best validation AP 0.302–0.324, mean 0.313 ± 0.007 |
| the same on the H100 (v4, seeds 42–44) | 0.367, 0.374, 0.375 |

- Inputs, labels and initial states are the same. Only the head's training trajectory differs, and the gap holds across 12 local seeds, so it is not seed luck.
- Local training reaches a lower training loss (epoch 6: 0.235 vs 0.241 on the H100) and a lower validation AP: it fits train harder and generalizes worse.
- The only device-dependent parts of the loop found in the code are the dropout masks (CPU vs CUDA RNG streams) and floating-point reduction order. Neither obviously explains a systematic 0.06; the cause is open.
- Inference is not affected: a frozen model scores the same on CPU, MPS and CUDA (1.6e-5).
- Consequence: v4's B2-S validation lead over B3-S may partly be a property of training on CUDA. On test, the v4 B2-S − B3-S gap was already within noise (+0.011, interval includes 0).
- Next check (about 15 minutes on any CUDA GPU, cheap): run `scripts/b2_head_platform_check.py --seeds` on CUDA. If CUDA lands at 0.36–0.38 for all seeds, the platform effect is confirmed; then run the same seeds with dropout off on both platforms.

- B2-S validation AP also moves by 0.02–0.03 from one epoch to the next (v4 seed 42: 0.359, 0.346, 0.347, 0.334, 0.324, 0.367, …), so its best-epoch value is partly selection noise.
- **Open:** the CUDA check above, then the nine B2-S runs on CUDA (about 40 minutes). The D2 gate (is the curve still rising at 100%?) is not passed for B0-LR/B1 and undecided for the encoder variants.
