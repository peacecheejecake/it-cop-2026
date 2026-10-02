# Study `public-comparison-v3-cpt100m` — CodeBERT diff pretraining at 10× the budget, and the 2:1 MLM:RMI mix (exploratory, pre-registered 2026-10-02)

**Why:** in v3, 10M tokens of diff MLM gave no gain over plain fine-tuning (B4 − B3 = −0.008 test AP, interval [−0.025, +0.008]), and splitting the same budget 1:1 between MLM and RMI was lower than MLM alone (B5 − B4 = −0.010, [−0.020, −0.002]). An external review (2026-10-02) and our own check raised three reasons the v3 pretraining may simply have been too short to say much:

- **Exposure.** v3 pretraining was 265 updates, 2.2 passes over 4.55M tokens, about 5.5 minutes. BiCC-BERT trains for 100 epochs on a corpus of similar size (27,391 changes).
- **A new MLM head.** `microsoft/codebert-base` ships without an MLM head, so `lm_head` (dense, layer norm, biases) starts from random values. Dev MLM loss fell from about 15 to 1.5 during the 265 updates, so much of the budget went to fitting that head.
- **Mix ratio.** BiCC-BERT's final setting is MLM:RMI = 2:1; its own 1:1 ablation is also below MLM alone (F1 0.436 vs 0.452). v3 ran only 1:1, and RMI got 132 updates (dev accuracy 0.57–0.70 across seeds).

**Status:** exploratory extension, registered **after** the v3 public test was opened (freeze `b6afa59f3a2a9550`). It is not part of the untouched v3 comparison (spec §12). The user approved the two arms and the budget on 2026-10-02.

## 1. Arms

Config: `configs/studies/public-comparison-v3-cpt100m.yaml`. Everything outside the `cpt` section is v3's: data (`jitd4j-audit1`, `upstream-clean2`), EvidenceView, jit14, fine-tuning (encoder lr 1e-5, head lr 1e-3, patience 5), seeds 42/43/44.

| arm (variant in this study) | pretraining | MLM tokens | RMI tokens |
|---|---|---|---|
| `MLM-100M` (B4-S) | MLM only, 100M tokens | 100M | 0 |
| `MIX21-100M` (B5-S) | updates cycle MLM, MLM, RMI; 100M tokens in total | about 67M | about 33M |
| *v3 references* | | | |
| B3-S | none | 0 | 0 |
| B4-S | MLM only, 10M | 10M | 0 |
| B5-S | MLM, RMI alternating (1:1), 10M | 5M | 5M |

- **Budget:** 100M non-padding input tokens, about 22 passes over CPT-train, about 2,650 updates. The lr schedule (linear warmup 6%, linear decay) scales with the number of updates.
- **Corpus:** unchanged, CPT-train from the public train split only. BiCC-BERT's corpus is described as 27,391 changes from the same 21 projects; whether it includes the test period was not verified.
- **Dev diagnostics:** every 250 updates (v3: 25), the same number of points per run. Dev MLM loss and dev RMI accuracy are reported; a rising dev MLM loss marks overfitting to the repeated corpus.
- **Not run:** an MLM-only arm at 67M tokens (the exact partner of `MIX21-100M`), and 10M-budget arms. See §3.

## 2. Evaluation

- **Selection and reporting:** public validation, as in v3 (AP, ROC-AUC, Recall@5%/10%, F1 at the validation max-F1 threshold; 3 seeds, mean ± sd).
- **Comparisons (validation AP, seed-paired, project bootstrap with 2,000 reps):**
  - **C1 (exposure):** `MLM-100M` − B3-S, and `MLM-100M` − B4-S (10M).
  - **C2 (mix at the paper's ratio):** `MIX21-100M` − `MLM-100M`.
  - **C3:** `MIX21-100M` − B3-S.
- **Test:** the v3 test set stays closed for these arms unless the rule below fires. Rule, fixed now: if an arm's mean validation AP exceeds B3-S's (0.687) by **at least 0.010** and the seed-paired difference is positive for at least 2 of 3 seeds, both arms are frozen together and evaluated on test once. Otherwise the study reports validation only.

## 3. How results will be worded (fixed before the run)

These follow the corrections made to `results-final-2026-10-01.md` on 2026-10-02.

- **No equivalence claims.** A difference whose interval includes 0 is reported as "no gain observed under these conditions", with the estimate and the interval. No equivalence margin is tested.
- **C2 is a budget-allocation effect, not a pure RMI effect.** `MIX21-100M` has a third less MLM exposure than `MLM-100M`. The cause of any difference (RMI itself, less MLM, or their interaction) is not separated by this pair alone. If C1 shows MLM exposure makes no measurable difference between 0, 10M and 100M, the "less MLM" explanation becomes less likely; that is an inference, and it will be labelled as one.
- **Bootstrap column.** The last column is "share of bootstrap resamples with difference ≤ 0", not a p-value. Intervals are per comparison, with no multiple-comparison correction; all 21 evaluation projects also appear in train.
- **Seed noise.** The sd of validation AP over 3 seeds was 0.008–0.014 in v3. Differences near 0.01 will be described by direction and consistency, not as established effects.
- **Scope.** A null result does not refute BiCC-BERT: the checkpoint rule (validation AP vs F1), the downstream protocol (separate head lr) and the exposure (22 passes vs 100 epochs) still differ. A positive result is exploratory, because the v3 test has been seen.
- **Split.** The provided split is mostly time-ordered within each project (median P(test later) 0.989, boundary violations of about 7%; `m0-audit.md`). Results remain fixed-holdout results.

## 4. Cost guards

- Runpod H100 ($3.49/h at v3 time). v3 timings: CPT 330 s per seed at 10M; B5 with fine-tuning took about 70 minutes for 3 seeds.
- Estimate: CPT about 55 minutes per seed at 100M, plus about 17 minutes of fine-tuning. Two arms × 3 seeds: **about 7 hours, about $25**.
- The run can share a pod with diffllm-v2 (exp 013). Check the balance against the estimate plus the $13 margin first; pull small results after every run and weights last.
