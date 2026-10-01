# RESULTS — 011-dl-m6-final (study `public-comparison-v3`, freeze `b6afa59f3a2a9550`)

The public test is JIT-Defects4J `upstream-clean2` test: **5,480 changes, 475 positives (base rate 0.087), 21 projects**. It was evaluated **once**, after freezing all 9 primary variants. Thresholds are the frozen validation max-F1 thresholds. Numbers come from `results/report/report-corrected.json`.

## Test results

| variant | test AP (mean ± sd) | ROC-AUC | Recall@5% | Recall@10% | F1@thr | (validation AP) |
|---|---|---|---|---|---|---|
| B0-LR | 0.215 | 0.723 | 0.160 | 0.286 | 0.293 | 0.319 |
| B0-LGBM | 0.197 | 0.676 | 0.154 | 0.267 | 0.257 | 0.211 |
| B1-TFIDF-S | 0.440 | 0.852 | 0.307 | 0.491 | 0.453 | 0.546 |
| B2-S frozen CodeBERT | 0.475 ± 0.018 | 0.851 | 0.347 | 0.528 | 0.481 | 0.616 |
| **B3-S full FT** | **0.598 ± 0.023** | **0.909** | **0.425** | **0.615** | **0.580** | 0.687 |
| B4-S MLM CPT + FT | 0.590 ± 0.034 | 0.903 | 0.412 | 0.608 | 0.565 | 0.674 |
| B5-S MLM+RMI CPT + FT | 0.580 ± 0.029 | 0.902 | 0.410 | 0.590 | 0.555 | 0.669 |
| L0-S Qwen2.5-Coder-7B zero-shot | 0.125 | 0.634 | 0.074 | 0.162 | 0.207 | 0.140 |
| L1-S Qwen 4-shot | 0.172 ± 0.006 | 0.677 | 0.144 | 0.244 | 0.226 | 0.204 |

sd is the spread across the three training seeds (demo seeds for L1). B0/B1 are deterministic. L0 is a single run.

## Pre-registered pairwise comparisons (test AP; seed-paired, project bootstrap 2,000 reps)

| Comparison | Mean difference | Per seed | Project-bootstrap 95% CI | P(diff ≤ 0) |
|---|---|---|---|---|
| B3 − B2 (value of fine-tuning) | **+0.124** | +0.080 / +0.158 / +0.133 | [+0.081, +0.164] | 0.000 |
| B4 − B3 (value of MLM CPT, RQ2) | −0.008 | −0.021 / −0.005 / +0.001 | [−0.025, +0.008] | 0.84 |
| B5 − B4 (value of RMI, RQ3) | −0.010 | −0.005 / −0.012 / −0.013 | [−0.020, −0.002] | 0.99 |
| B5 − B3 | −0.018 | −0.025 / −0.017 / −0.013 | [−0.039, +0.002] | 0.96 |
| L1 − L0 (value of few-shot) | **+0.047** | +0.053 / +0.040 / +0.046 | [+0.022, +0.081] | 0.000 |
| B1 − B0-LR (value of text) | **+0.225** | (deterministic) | [+0.164, +0.279] | 0.000 |
| B2 − B1 (frozen encoder vs TF-IDF) | +0.035 | +0.052 / +0.016 / +0.036 | [−0.002, +0.076] | 0.03 |
| B3 − B1 | **+0.158** | +0.131 / +0.174 / +0.169 | [+0.121, +0.194] | 0.000 |
| B3 − L1 | **+0.427** | +0.393 / +0.449 / +0.438 | [+0.356, +0.485] | 0.000 |

The project-bootstrap interval reflects uncertainty from resampling projects. It is a different quantity from seed variance (the per-seed spread).

## Conclusions (sealed test, study v3)

1. **Fully fine-tuned CodeBERT (B3-S) is best.** It beats TF-IDF by +0.158 AP and the frozen representation by +0.124, and both intervals exclude 0. Across projects, it scores at or above B1 in 19 of 21.
2. **RQ2, diff MLM CPT (10M tokens): no added value.** B4 − B3 = −0.008 with a CI that includes 0. B4 beats B3 in 11 of 21 projects, essentially a tie.
3. **RQ3, adding RMI: no added value, possibly slightly harmful.** B5 − B4 = −0.010 with CI [−0.020, −0.002]. This is at the same total budget, so MLM exposure is halved. RMI learning was also unstable across seeds (dev accuracy 0.57–0.70).
4. **The local LLM (Qwen2.5-Coder-7B) is weak as a frozen scorer.**
   - L0 0.125 and L1 0.172 are well above the base rate (0.087) but below even the structured-feature LR (0.215).
   - 4-shot examples do help (+0.047, CI excludes 0).
   - This is one model, one template, one scorer; it should not be generalized to "encoders beat LLMs" (spec §8).
5. **The frozen encoder (B2) beats TF-IDF by +0.035, but the CI just touches 0.** The test-set gap is smaller than on validation.

## Validation-to-test gap

Every variant drops on test: B3 −0.089, B1 −0.106, and deterministic B0-LR −0.104 even though it selects no epoch.

- Because B0-LR drops this much, most of the gap looks like **a difference between the validation and test cohorts**, not epoch-selection bias. The cause (e.g. per-project composition) has not been analyzed.
- The rankings, however, are consistent between validation and test. The B2−B1 gap (validation +0.070 → test +0.035) and the B4−B3 gap (−0.013 → −0.008) narrowed on test.
- Per-project AP is in `per_project_test_ap` in the report.

## Limitations

- **Data:** one public dataset (JIT-Defects4J, 21 Java projects). The provided split is not chronological. The license is unverified.
- **Protocol:** v3's head lr 1e-3 was chosen after seeing validation results (disclosed). Under v2 the CodeBERT variants were systematically under-trained.
- **CPT scale:** CPT was tried only at 10M tokens; scale-up was not done (the spec allows stopping when there is no effect).
- **L0/L1 reproducibility:** freeze verified it with a tolerance (bf16 batch composition), not exact reproduction.
- **Internal (company) data:** not evaluated yet. The `study export` inference-only bundle is ready.
