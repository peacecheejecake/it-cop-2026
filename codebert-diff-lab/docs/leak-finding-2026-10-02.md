# Finding: the JIT-Defects4J package text depends on the label (2026-10-02)

**Summary.** In the JIT-Fine `data.zip` package, the added/removed line sets of a commit were built differently for buggy and clean commits. Buggy commits keep only part of the commit; clean commits keep all of it. A text model can therefore read the label from how much of the commit its text contains. Every text variant of studies v2 and v3 was trained and evaluated on this text. The structured baselines (B0) and every experiment that took its text from git are not affected.

Found in exp `015-dl-internal-pack` while validating the git extractor for the internal trial. Records: `experiments/015-dl-internal-pack/RESULTS.md`, `results/validate/validate.json`.

## 1. What kind of leak this is

- It is **not** contamination between splits. No validation or test row, label or statistic reaches training.
- It is **target leakage through preprocessing**: the input representation itself is a function of the label, in train, validation and test alike.
- Because all three splits carry the same artifact, a model learns it in training and is rewarded for it in evaluation. Split-overlap checks cannot see it.
- The artifact does not exist in a real git diff, so it is unavailable at prediction time on any new repository.

## 2. Evidence

### 2.1 Package text against the commit's git diff

27,209 of the 27,319 package commits were found in the cached public mirrors (`experiments/.cache/git/jd4j`). Lines were extracted with `diff_lab.gitextract` (.java files, comments dropped, package tokenization).

| cohort | commits | line sets identical to git | more than 20% of the commit's lines missing | mean share missing | lines, package / git |
|---|---|---|---|---|---|
| train, buggy | 1,375 | 20% | **71%** | 46% | 112k / 285k |
| valid, buggy | 461 | 21% | **69%** | 46% | 31k / 76k |
| test, buggy | 458 | 27% | **67%** | 44% | 25k / 87k |
| train, clean | 14,930 | 92% | 0.9% | 0.6% | 789k / 759k |
| valid, clean | 4,984 | 94% | 0.5% | 0.4% | 224k / 219k |
| test, clean | 5,001 | 94% | 0.6% | 0.5% | 301k / 292k |

- Of 2,294 buggy commits, 1,604 (70%) lack a substantial part of the commit. Of 24,915 clean commits, 193 (0.8%) do.
- In multi-file buggy commits of commons-io and commons-lang, 43% of the changed files are present in the package text; in clean commits, 99.8%. A file is either kept whole or dropped.
- The dropping rule was not verified. It looks like a restriction to files that contain buggy lines (the package also ships a line-level ground-truth pickle), but that is an inference.

### 2.2 The same frozen model on both inputs

B3-S seed 43 (freeze `b6afa59f3a2a9550`), scored through the offline `predict` path on the 5,459 test commits found in the mirrors. The test split was already opened once in exp 011; this run checks inputs, not models.

| text | jit14 | test AP | ROC-AUC |
|---|---|---|---|
| package | package | 0.606 | 0.913 |
| package | recomputed from git | 0.607 | 0.912 |
| **git** | package | **0.164** | 0.639 |
| **git** | recomputed from git | **0.164** | 0.640 |

Base rate 0.084. Replacing jit14 changes nothing; replacing the text removes most of the performance.

### 2.3 A check that needs only the package

`tools/representation_leak_check.py <snapshot_dir>` computes one number per commit, coverage = (text lines + 1) / (la + ld + 1), and scores "low coverage" as a predictor of the label. No git and no model are involved.

| snapshot | split | base rate | AP of low coverage | ROC-AUC |
|---|---|---|---|---|
| package (`jitd4j-audit1`) | train | 0.085 | 0.323 | 0.776 |
| | valid | 0.085 | 0.372 | 0.818 |
| | test | 0.087 | **0.324** | 0.754 |
| git (`jitd4j-git1`) | train | 0.084 | 0.089 | 0.560 |
| | valid | 0.085 | 0.105 | 0.616 |
| | test | 0.084 | 0.089 | 0.562 |

- In the package, this single ratio reaches test AP 0.324 with no training. B0-LR with all 14 features reaches 0.215.
- On git text the same ratio is at the base rate.

### 2.4 Retraining on git text (study v4, validation)

| variant | v3 validation AP (package text) | v4 validation AP (git text) |
|---|---|---|
| B0-LR | 0.319 | 0.302 |
| B0-LGBM | 0.211 | 0.240 |
| B1-TFIDF-S | 0.546 | **0.345** |
| B3-S | 0.687 | running (exp 016) |

The structured baselines barely move. The text baseline loses 0.20.

## 3. Scope

| experiments | input text | status |
|---|---|---|
| 006–011 (studies v2, v3): B1, B2, B3, B4, B5, L0, L1 | package EvidenceView | **affected** |
| 009/010 CPT corpora (B4, B5) | package EvidenceView | affected (downstream evaluation on leaked text) |
| 013 diffllm-v2, EvidenceView arms | package EvidenceView | **affected** (running on 2026-10-02) |
| 014 CPT-100M | package EvidenceView | **affected** (running on 2026-10-02) |
| L2 extension | package EvidenceView | affected if run as registered |
| B0-LR, B0-LGBM in every study | jit14 only | not affected (confirmed by 2.2 and 2.4) |
| 012 diffllm-v1 | `git show -U3` full diff | not affected |
| 005 cross-project | rebuilt from git by `build-apache` | not affected |
| 001–003 | ApacheJIT, built from git | not affected |

Conclusions of `docs/results-final-2026-10-01.md` that can no longer be claimed:

- **Absolute levels and rankings of text variants**: "B3-S is best (0.598)", B3 − B1, B3 − B2, B3 − L1, B1 − B0-LR. Each mixes real signal with skill at reading the artifact.
- **Relative comparisons measured on leaked text**: "MLM CPT adds nothing" (B4 − B3), "RMI adds nothing" (B5 − B4), "4-shot helps" (L1 − L0). They may or may not hold on clean text; they are unknown, not refuted.
- Still valid: B0-LR and B0-LGBM numbers, and exp 012's R-diff test AP 0.283 (git full diff), which is currently the best leak-free test result on this dataset.

## 4. Exchange with the agent running 013/014 (2026-10-02)

**Their conclusion:** no leakage found in the data that 013 and 014 train on. Items they checked:

- No change_id or commit SHA shared across train/valid/test; the 190 excluded rows are gone from the split table.
- 014 CPT rows (15,375 train + 809 dev) are all train rows; the code refuses rows outside train or a label column.
- 013 loads labels only for train+valid (21,649 rows); test is loaded without labels.
- No prompt contains its own SHA or change_id.
- Feature scaling is fit on train only.
- "Label proxy features": each jit14 column alone predicts the train label with AUC at most 0.745 (la).
- Residual overlaps (one truncated-input pair between train and valid, 6–24 valid/test duplicates, 408 identical messages) and caveats (pretraining overlap, time order, validation reuse, jit14 computation time).

**Review:**

- Every item above is correct, and we agree there is no split contamination.
- The checks answer "does evaluation data reach training?". The finding here answers a different question, "was the input built from the label?". The two do not conflict.
- The proxy check looked at one jit14 column at a time. The artifact is in the **relation between the text and jit14**: the text of a buggy commit covers less of the commit than la/ld say it should. The ratio in section 2.3 exposes it with AUC 0.75–0.82 and AP about four times the base rate, using package data only.
- 013 and 014 are well isolated across splits but read the package EvidenceView, so their results inherit the artifact.
- Request passed on: run `tools/representation_leak_check.py` on the snapshot to confirm independently.

## 5. What was changed

- `diff_lab.gitextract` and `diff-lab internal extract`: text and jit14 from git history, in the package's tokenization.
- `diff-lab data rebuild-git`: snapshot `jitd4j-git1`. It keeps the package's change IDs, labels and provided split, and replaces message, lines and jit14 with git-extracted values. 110 commits missing from the mirrors are dropped (train 69, valid 20, test 21; 38 of them buggy).
- Study `public-comparison-v4-gitlines`: v3 hyperparameters unchanged, run on `jitd4j-git1`.
- Correction plan: `docs/v4-correction-plan.md`.

## 6. Limits of this finding

- The cause inside the upstream pipeline is inferred, not read from upstream code.
- The extractor does not reproduce the package exactly even for clean commits (6–8% of clean line sets differ, by under 1% of lines). jit14 agreement is high for size metrics and lower for ndev (21% exact) and exp (54%); section 2.2 shows the model is insensitive to that.
- Whether other published results on this package are affected was not examined.
