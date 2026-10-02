# RESULTS — 015-dl-internal-pack (stopped before packaging)

## Finding: the JIT-Defects4J (JIT-Fine) line sets leak the label

B3-S seed 43 on the public test, scoring the same 5,459 commits through the real offline `predict` path:

| input | AP | ROC-AUC |
|---|---|---|
| package lines + package jit14 (= frozen path) | 0.606 | 0.913 |
| package lines + git-extracted jit14 | 0.607 | 0.912 |
| git-extracted lines + package jit14 | **0.164** | 0.639 |
| git-extracted lines + git jit14 (what internal data looks like) | **0.164** | 0.640 |

Base rate 0.084. For comparison B0-LR (jit14 only) has test AP 0.215.

Cause:
- **Clean commits**: the package's added/deleted line sets equal the git diff (Jaccard 0.98; 99.4% of changed files kept).
- **Buggy commits**: the package keeps only a subset. Jaccard is 0.55, and git has about 3× more lines.
  - In multi-file buggy commits, only 43% of files are kept (commons-io/lang check).
  - Kept files are kept whole; the others are dropped. This looks like a restriction to files containing the buggy lines.
- The jit14 features (la/ld/nf…) still describe the full commit. So "the text covers fewer files and lines than the commit" is a label signal available only in this package.
- The extractor reproduces the clean-commit representation almost exactly. The gap therefore comes from the package, not from the extractor.

Consequences:
- The v3 text-variant results (B1–B5, L0/L1, diffllm) are likely inflated by this artifact. B0-LR and B0-LGBM read only jit14 and are not affected.
- Bringing B3-S in as is would measure a model that, on real diffs, ranks worse than the structured LR.
- Packaging was stopped to let the user decide.
