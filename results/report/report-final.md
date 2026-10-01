# Study report public-comparison-v3 (freeze b6afa59f3a2a9550)

Public test: 5480 changes, 475 positives, 21 projects. Validation numbers are selection-validation; test was evaluated once after freeze.

| variant | valid AP | test AP | sd | ROC-AUC | Recall@5% | Recall@10% | F1@thr |
|---|---|---|---|---|---|---|---|
| B0-LR | 0.319 | 0.215 | 0.000 | 0.723 | 0.160 | 0.286 | 0.293 |
| B0-LGBM | 0.211 | 0.197 | 0.000 | 0.676 | 0.154 | 0.267 | 0.257 |
| B1-TFIDF-S | 0.546 | 0.440 | 0.000 | 0.852 | 0.307 | 0.491 | 0.453 |
| B2-S | 0.616 | 0.475 | 0.018 | 0.851 | 0.347 | 0.528 | 0.481 |
| B3-S | 0.687 | 0.598 | 0.023 | 0.909 | 0.425 | 0.615 | 0.580 |
| B4-S | 0.674 | 0.590 | 0.034 | 0.903 | 0.412 | 0.608 | 0.565 |
| B5-S | 0.669 | 0.580 | 0.029 | 0.902 | 0.410 | 0.590 | 0.555 |
| L0-S | 0.140 | 0.125 |  | 0.634 | 0.074 | 0.162 | 0.207 |
| L1-S | 0.204 | 0.172 | 0.006 | 0.677 | 0.144 | 0.244 | 0.226 |

| pair | test AP diff | project bootstrap 95% CI | P(diff<=0) |
|---|---|---|---|
| B3-S - B2-S | +0.1235 | [+0.081, +0.164] | 0.000 |
| B4-S - B3-S | -0.0082 | [-0.025, +0.008] | 0.841 |
| B5-S - B4-S | -0.0100 | [-0.020, -0.002] | 0.993 |
| B5-S - B3-S | -0.0182 | [-0.039, +0.002] | 0.958 |
| L1-S - L0-S | +0.0467 | [+0.022, +0.081] | 0.000 |
| B1-TFIDF-S - B0-LR | +0.2253 | [+0.164, +0.279] | 0.000 |
| B2-S - B1-TFIDF-S | +0.0346 | [-0.002, +0.076] | 0.033 |
| B3-S - B1-TFIDF-S | +0.1582 | [+0.121, +0.194] | 0.000 |
| B3-S - L1-S | +0.4266 | [+0.356, +0.485] | 0.000 |

## Definition-of-done status

- **DoD-Core**: complete — all_variants_frozen=ok, all_variants_tested_once=ok, matched_evidence=ok, same_structured_columns=ok, no_prediction_failures=ok, label_ledger_present=ok
- **DoD-Comparative**: complete — all_variants_frozen=ok, all_variants_tested_once=ok, matched_evidence=ok, same_structured_columns=ok, no_prediction_failures=ok, label_ledger_present=ok
- DoD-Internal-Ready: evaluated separately (bundle verify + offline predict fixture)
- DoD-Internal-Evaluated: requires approved internal data (M7)
- L2: not executed (extension, not required)

## Replication axes

- B0-LR: axis=training_seed, replicates=[42, 43, 44], effective training seeds=1, demo sets=0 (deterministic: identical replicates are not independent)
- B0-LGBM: axis=training_seed, replicates=[42, 43, 44], effective training seeds=1, demo sets=0 (deterministic: identical replicates are not independent)
- B1-TFIDF-S: axis=training_seed, replicates=[42, 43, 44], effective training seeds=1, demo sets=0 (deterministic: identical replicates are not independent)
- B2-S: axis=training_seed, replicates=[42, 43, 44], effective training seeds=3, demo sets=0
- B3-S: axis=training_seed, replicates=[42, 43, 44], effective training seeds=3, demo sets=0
- B4-S: axis=training_seed, replicates=[42, 43, 44], effective training seeds=3, demo sets=0
- B5-S: axis=training_seed, replicates=[42, 43, 44], effective training seeds=3, demo sets=0
- L0-S: axis=none, replicates=[42], effective training seeds=0, demo sets=0
- L1-S: axis=demo_set, replicates=[42, 43, 44], effective training seeds=0, demo sets=3

## Label access (per run)

| variant | mode | gradient | demo | selection | CPT labels | RMI synthetic |
|---|---|---|---|---|---|---|
| B0-LR | supervised_classifier | [16184] | [0] | [5465] | [0] | [0] |
| B0-LGBM | supervised_classifier | [16184] | [0] | [5465] | [0] | [0] |
| B1-TFIDF-S | supervised_classifier | [16184] | [0] | [5465] | [0] | [0] |
| B2-S | frozen_encoder_train_head | [16184] | [0] | [5465] | [0] | [0] |
| B3-S | supervised_encoder_head | [16184] | [0] | [5465] | [0] | [0] |
| B4-S | mlm_cpt_then_supervised | [16184] | [0] | [5465] | [0] | [0] |
| B5-S | mlm_rmi_cpt_then_supervised | [16184] | [0] | [5465] | [0] | [16896] |
| L0-S | zero_shot | [0] | [0] | [5465] | [0] | [0] |
| L1-S | static_few_shot | [0] | [4] | [5465] | [0] | [0] |

## Cost

| variant | device | GPU s (all replicates) | CPT s (mean) |
|---|---|---|---|
| B0-LR | cpu | 0 |  |
| B0-LGBM | cpu | 0 |  |
| B1-TFIDF-S | cpu | 0 |  |
| B2-S | cuda | 374 |  |
| B3-S | cuda | 3437 |  |
| B4-S | cuda | 4156 | 330 |
| B5-S | cuda | 4214 | 346 |
| L0-S | cuda | 124 |  |
| L1-S | cuda | 1394 |  |
