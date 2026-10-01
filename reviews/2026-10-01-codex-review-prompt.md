You are reviewing a research codebase and the experiments run with it so far. Do not modify any files; this is a read-only review. Answer in Korean, citing `path:line` for every code finding.

## Context
- Repo root: this directory. The active project is `codebert-diff-lab/` (package `diff_lab`). It implements the spec in `codebert-diff-lab/docs/spec-v0.2/` (start with README.md, requirements.md, experiment-protocol.md, implementation-plan.md, comparison-matrix.md). `AGENTS.md` §0 explains the research direction. `baseline/` and experiments 001–005 are a legacy direction; ignore them.
- Data: JIT-Defects4J public dataset; split `upstream-clean1` (train 16,184 / valid 5,465 / test 5,480). The public test is sealed until study freeze.
- Milestones done: M0 audit (`codebert-diff-lab/docs/m0-audit.md`), M1/M2 CPU baselines (experiment 006), M3 B2-S/B3-S on H100 (`experiments/007-dl-m3-encoder/RUNLOG.md`, `RESULTS.md`). M4 B4-S (diff MLM CPT, `src/diff_lab/cpt.py`) is running now (`experiments/008-dl-m4-mlm-cpt/RUNLOG.md`).
- Validation AP so far (base rate 0.085): B0-LR 0.319, B0-LGBM 0.211, B1-TFIDF-S 0.546, B2-S 0.377±0.005 (registered protocol lr 1e-5, hit the 20-epoch cap still improving), B3-S 0.527±0.024 (best epoch 2–3, then strong overfitting), B4-S seed 42 0.560 (others pending).
- Supplementary B2 sensitivity (head lr 1e-3, max 100 epochs, `configs/studies/b2-head-sensitivity-v1.yaml` on branch exp/008): validation AP 0.623 / 0.604 / 0.622. That is higher than B3-S and B1. The validation curve swings about ±0.03 between epochs, and the best epoch is chosen on the same validation set.

## What to review (in priority order)
1. **Correctness bugs** in `codebert-diff-lab/src/diff_lab/` (especially neural.py, cpt.py, runner.py, metrics.py, evidence.py, features.py, data/splits.py, adapters/). Look for: loss or accumulation math, masking and eligibility, resume, RNG and determinism, hash and provenance, score alignment, threshold and metric definitions versus the spec.
2. **Data and label leakage, and spec policy violations**: can a validation, test, or CPT-dev ID or label reach any gradient, fit, or selection step? Check the train-only fitting of features and TF-IDF, CPT-dev versus CPT-train, and the duplicate removal.
3. **Methodology**: is the comparison fair given the evidence above? In particular, does the shared fine-tuning protocol (lr 1e-5 for both encoder and head, no head-specific lr) systematically disadvantage the encoder variants? How should the study handle this without p-hacking or violating the pre-registration rules in the spec (protocol version changes, selection only on public validation)? Also judge whether best-epoch selection on a noisy validation curve, then reporting AP on that same validation set, inflates the numbers. Suggest a principled fix.
4. **Gaps versus the spec** for the milestones claimed done (M0–M4): missing run-folder files, acceptance tests not covered, metrics not reported.
5. Anything else important (test quality, silent fallbacks).

Output: findings ranked by severity (Critical / High / Medium / Low), each with evidence and a concrete fix, then a short section with recommended next steps for the study. Be specific; skip generic advice.
