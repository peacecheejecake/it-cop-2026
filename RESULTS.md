# RESULTS — 016-dl-v4-gitlines (study `public-comparison-v4-gitlines`, freeze `7340ee71c876c8b7`)

All 9 variants were re-run on snapshot `jitd4j-git1`, whose text and jit14 come from git instead of the JIT-Fine package (`codebert-diff-lab/docs/leak-finding-2026-10-02.md`). Hyperparameters are v3's, unchanged. 25 runs were frozen, then the public test (5,459 changes, 458 positives, base rate 0.084, 21 projects) was scored once. Numbers come from `results/report.json`.

## Test results

| variant | test AP (mean ± sd) | ROC-AUC | Recall@5% | Recall@10% | F1@thr | validation AP | v3 test AP (package text) |
|---|---|---|---|---|---|---|---|
| B0-LGBM | 0.182 | 0.678 | 0.159 | 0.258 | 0.244 | 0.240 | 0.197 |
| B0-LR | 0.224 | 0.728 | 0.164 | 0.299 | 0.282 | 0.302 | 0.215 |
| B1-TFIDF-S | 0.247 | 0.773 | 0.175 | 0.320 | 0.317 | 0.345 | 0.440 |
| **B2-S** frozen CodeBERT | **0.264 ± 0.009** | 0.777 | 0.202 | 0.335 | 0.323 | 0.372 | 0.475 |
| B3-S full FT | 0.253 ± 0.016 | 0.781 | 0.189 | 0.333 | 0.318 | 0.359 | 0.598 |
| B4-S MLM CPT + FT | 0.243 ± 0.019 | 0.781 | 0.172 | 0.325 | 0.306 | 0.351 | 0.590 |
| B5-S MLM+RMI CPT + FT | 0.243 ± 0.022 | 0.779 | 0.184 | 0.321 | 0.318 | 0.343 | 0.580 |
| L0-S Qwen 7B zero-shot | 0.126 | 0.650 | 0.094 | 0.197 | 0.192 | 0.138 | 0.125 |
| L1-S Qwen 7B 4-shot | 0.157 ± 0.021 | 0.700 | 0.114 | 0.227 | 0.239 | 0.188 | 0.172 |

sd is across the three training seeds (demo seeds for L1). B0 and B1 are deterministic; their seed-42 runs were made on macOS and seeds 43/44 on the pod, which differ in the fourth decimal.

## Pre-registered comparisons (test AP; seed-paired; project bootstrap, 2,000 resamples)

| comparison | mean difference | per seed | 95% interval | P(diff ≤ 0) |
|---|---|---|---|---|
| B1 − B0-LR (value of text) | +0.022 | +0.022 / +0.022 / +0.022 | [−0.013, +0.050] | 0.090 |
| B2 − B1 (frozen encoder vs TF-IDF) | +0.017 | +0.010 / +0.027 / +0.014 | [−0.009, +0.042] | 0.101 |
| B3 − B2 (value of fine-tuning) | −0.011 | +0.011 / −0.017 / −0.025 | [−0.029, +0.009] | 0.858 |
| B3 − B1 | +0.006 | +0.021 / +0.009 / −0.011 | [−0.014, +0.029] | 0.283 |
| B4 − B3 (MLM CPT) | −0.010 | −0.008 / −0.008 / −0.014 | [−0.018, +0.004] | 0.922 |
| B5 − B4 (RMI) | +0.001 | +0.004 / −0.000 / −0.002 | [−0.011, +0.007] | 0.517 |
| B5 − B3 | −0.010 | −0.004 / −0.008 / −0.016 | [−0.018, +0.000] | 0.969 |
| L1 − L0 (few-shot) | **+0.031** | +0.052 / +0.010 / +0.030 | [+0.015, +0.052] | 0.000 |
| B3 − L1 | **+0.096** | +0.089 / +0.120 / +0.079 | [+0.062, +0.124] | 0.000 |

Not pre-registered, computed after the test was opened (same bootstrap, seed 0): B2 − B0-LR = +0.039, interval [+0.002, +0.069]; B3 − B0-LR = +0.029, interval [−0.009, +0.062].

## Conclusions

1. **On leak-free text every model is far lower, and the gaps between text models disappear.** B3-S falls from 0.598 to 0.253. B1 through B5 lie within 0.243–0.264, and no pre-registered difference among them has an interval that excludes 0.
2. **The v3 headline is not reproduced.** "Full fine-tuning is best" and "fine-tuning adds +0.12 over the frozen encoder" came from the label-dependent package text. Here B3 − B2 is −0.011 with an interval that includes 0.
3. **Text adds a little over the structured features.** B2-S is the highest variant (0.264 vs B0-LR 0.224). The registered B1 − B0-LR comparison is not established; the post hoc B2 − B0-LR interval just excludes 0.
4. **Diff pretraining (10M tokens) still adds nothing** (B4 − B3, B5 − B4 include 0), now on clean text.
5. **Few-shot examples help the frozen LLM** (+0.031, interval excludes 0), and the frozen LLM stays below B0-LR. L0/L1 changed little from v3 (0.125 → 0.126, 0.172 → 0.157), so the frozen LLM was not exploiting the artifact.
6. **Reference:** exp 012 R-diff (Qwen 7B LoRA on git full diffs) has test AP 0.283 on 5,480 test commits, above every v4 variant. It is one seed and a slightly different cohort.

## Model for the internal trial (rule in `docs/v4-correction-plan.md` §4)

- Highest mean validation AP among B2–B5: **B2-S** (0.372). Its best-validation seed: **44**, run `d4987435844df076`, validation AP 0.375, frozen threshold 0.155, test AP for that seed in `results/report.json`.
- B2-S is the pinned `microsoft/codebert-base` plus a fusion head (about 1 MB of trained weights).
- The validation gap to B0-LR (0.07) is larger than the seed spread, so the rule does not require shipping B0-LR. Given the small test gap (+0.039), shipping B0-LR as a second scorer is still recommended; it costs nothing.

## Limitations

- One public dataset, 21 Java projects; provided split, mostly time-ordered within project.
- The test commits were scored once in v3 with package text. v4 hyperparameters (including head lr 1e-3) are v3's, which were chosen on leaked validation text.
- 110 commits absent from the mirrors are excluded (21 in test, 17 of them buggy).
- The validation-to-test drop (about 0.08–0.11 for every variant, including deterministic B0-LR) is still unexplained.
- EvidenceView keeps 26–28% of lines at 512 tokens; large commits are heavily truncated.
