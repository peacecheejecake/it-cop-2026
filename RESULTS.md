# RESULTS — 009-dl-v3-rerun (study `public-comparison-v3`, public validation)

JIT-Defects4J `jitd4j-audit1/upstream-clean2`. Validation has 5,465 changes, 467 of them positive (base rate 0.085). All runs use the same EvidenceView (`message-add-del-text-v2`, 512 tokens). Seeds are 42, 43 and 44. **The public test is still sealed.**

> These numbers are **selection-validation** values. For the neural variants, both the checkpoint epoch and the threshold were chosen on this same validation set, so the values are optimistic. The B0/B1 thresholds are also chosen on validation. Only the sealed test, evaluated once after study freeze, gives an independent estimate. Do not treat the ordering below as a final conclusion.

| variant | AP (mean ± sd, n=3) | ROC-AUC | Recall@5% | Recall@10% | F1@thr* | Notes |
|---|---|---|---|---|---|---|
| B0-LR | 0.319 | 0.792 | 0.244 | 0.396 | 0.369 | Deterministic, so effectively one seed |
| B0-LGBM | 0.211 | 0.710 | 0.167 | 0.293 | 0.289 | Deterministic. The registered starting config overfits |
| B1-TFIDF-S | 0.546 | 0.885 | 0.398 | 0.565 | 0.530 | Deterministic |
| B2-S frozen CodeBERT + head | 0.616 ± 0.011 | 0.886 | 0.430 | 0.615 | 0.582 | best epoch 14–27 |
| **B3-S** full FT | **0.687 ± 0.008** | **0.912** | **0.483** | **0.667** | 0.652 | best epoch 2–3 |
| B4-S diff MLM CPT (10M tokens) + FT | 0.674 ± 0.014 | 0.911 | 0.470 | 0.657 | 0.633 | best epoch 1–2 |

\* The threshold is the max-F1 point on the same validation set.

Seed-paired differences (AP):

| Comparison | seed 42 | 43 | 44 | Mean |
|---|---|---|---|---|
| B3-S − B2-S | +0.058 | +0.092 | +0.063 | **+0.071** |
| B4-S − B3-S | −0.022 | −0.010 | −0.008 | **−0.013** |
| B2-S − B1-TFIDF-S | +0.076 | +0.058 | +0.075 | +0.070 |

## Comparison with v2 (initial protocol, shared encoder/head lr of 1e-5; 007/008)

| variant | v2 AP | v3 AP | Change |
|---|---|---|---|
| B2-S | 0.377 ± 0.005 | 0.616 ± 0.011 | +0.239 |
| B3-S | 0.527 ± 0.024 | 0.687 ± 0.008 | +0.160 |
| B4-S | 0.551 ± 0.008 (seeds 43/44 from logs only) | 0.674 ± 0.014 | +0.123 |

- With the head learning rate separated, every encoder variant improves sharply, and seed variance also drops for B3-S. Under v2 the randomly initialised head was **under-trained**, which held back the encoder variants as a group. B2 suffered most because its encoder is frozen and the head is the only thing that learns.
- In v2, B4-S beat B3-S by +0.024. **In v3 the order flips to −0.013.** The v2 advantage looks like an artefact of under-training: the CPT encoder simply made up for some of the slow head learning. Under v3, 10M tokens of diff-MLM CPT shows **no added value** on validation (RQ2). The gap is about the size of the seed spread (sd 0.008–0.014), so the defensible conclusion is "no benefit observed", not "harmful".

## Interpretation and caveats

1. **Encoder ordering (validation):** B3-S > B4-S > B2-S > B1-TFIDF-S > B0-LR > B0-LGBM. Full fine-tuning adds +0.07 over the frozen representation. The frozen representation also beats TF-IDF by +0.07.
2. **Selection bias:** B2-S scans validation over 19–32 epochs, while B3-S/B4-S scan only 6–8 epochs, so the number of selection opportunities differs (B2's is larger). The final comparison should use paired differences on the test set together with a project-level bootstrap.
3. **Head lr 1e-3 was chosen after looking at validation** (the 008 sensitivity run, with two configurations seen). The same single setting was applied to every variant, and the selection process is disclosed in the v3 header.
4. **CPT diagnostic:** the CPT-dev split is now grouped by duplicates (upstream-clean2), but one CPT-train row still matches CPT-dev text after evidence truncation. Dev loss is only a diagnostic and is not used for checkpoint selection.

## Next steps (spec order)

- **M5:** B5-S (MLM + RMI) under the same v3 FT protocol. The B4-S→B5-S comparison is defined as "same total CPT budget". Given B4's result, a "CPT has no effect" outcome is quite possible. The spec treats that as a valid result.
- **M6L:** L0-S/L1-S. Needs a local LLM candidate pin and user approval.
- **M6:** after implementing freeze → public test → report, evaluate all nine primary variants on test in one pass.
