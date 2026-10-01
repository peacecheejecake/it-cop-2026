# RUNLOG — 012-dl-diffllm

Branch `exp/012-dl-diffllm`, branched from main `089c994` and fast-forwarded to `345b988`.

Study **`diffllm-v1`** (exploratory, `codebert-diff-lab/docs/diffllm-study-v1.md`, `configs/studies/diffllm-v1.yaml`). The same pod also runs **the L2-S extension study** (`public-comparison-v3-l2ext`).

## User decisions

- Re-collect full public diffs (before → after) and add a pretraining phase on the iDiffLlama recipe, so that the risk alignment built on top of it can be compared.
- CPT corpus: **Apache Java projects disjoint from JD4J**. Scale: **50M tokens, seed 1**.
- Qwen2.5-Coder-14B vs 7B: **pre-registered** in the design document. After the 7B results the user chooses between the minimum (R-base-14B only) and the full set (14B 2×2).
- Test L2-S (retrieval few-shot). Since it is registered after the v3 test was opened, it is labelled as an extension.

## Data collection (local, free)

- `scripts/clone_repos.sh` writes bare clones to `experiments/.cache/git/{jd4j,cpt}`. Completed in about 4 minutes; the per-repository HEAD SHAs are in `logs/clone.out`.
  - JD4J's 21 repositories: 454 MB.
  - CPT's 6 repositories (zookeeper, zeppelin, activemq, kafka, cassandra, groovy): 1.5 GB, about 80k non-merge commits.
- **Labeled full diffs**: `fulldiff.extract_labeled` → `data/fulldiff/jitd4j-audit1`.
  - All 27,319 commits are ok; 615 binary sections dropped, 0 lock sections. Command: `git show <sha> --format= -U3 --no-color --no-ext-diff --first-parent`.
  - Consistency check on a sample of 200 commits: only 8.5% of JIT-Fine's lines appear in our diffs exactly as stored. **Ignoring whitespace, coverage is 91% for added lines and 88% for deleted lines (median 100%).** The low raw number is because JIT-Fine **tokenized** its lines (`LOG . error (`); the commits themselves match. So the v3 EvidenceView was also built from tokenized lines, applied identically to every model.
- **Input views**: `diffllm prepare` → `data/diffllm/view-full-diff-v1`.
  - 27,129 rows (train 16,184 / valid 5,465 / test 5,480, the same membership as upstream-clean2). Test rows carry no labels.
  - Mean content length 1,012 Qwen tokens; 32.6% truncated (whole hunks dropped from the end).
- **CPT corpus**: `build_cpt_corpus` → `data/cpt/apache-disjoint-diffs-v1`.
  - **50,000,641 tokens, 48,570 commits** (of 80,117 candidates, taken in seeded rank order).
  - 17,926 commits truncated at 2,048 tokens. Skipped: 791 empty, 178 duplicates, 0 identical to JD4J diffs.
  - Tokens by repository: kafka 13.8M, groovy 12.7M, cassandra 11.6M, activemq 6.7M, zeppelin 3.4M, zookeeper 1.8M.
- The snapshot was rebuilt inside the worktree; its parquet files are identical to 011. The L2 evidence is also identical to v3 (query digest `7d8774b4…`, parquet `d24c807d…`).

## Code (main)

- `fulldiff.py`: extraction, rendering, corpus construction.
- `llm_train.py`: LoRA CPT, risk SFT with loss on the label token only, embeddings, MLP.
- `diffllm.py`: prepare, arms, freeze, one test evaluation.
- CLI: `diff-lab diffllm …`.
- 97 tests, including an end-to-end run on a tiny Qwen.
- **Process mistake**: commit `8f782de` was made while a test was failing, because the command checked `tail`'s exit code instead of pytest's. The cause was flakiness in the test design (the ranking depends on commit SHAs); there is no implementation bug. Fixed in `575e359`. Pytest's exit code is now checked directly.

## Pod run (pending)

- Bundle: `.cache/bundles/jit012-345b988.tar.gz` (140 MB, sha256 `cf3e1ecc…ef99`). Contents: code, CODE_SHA, data (snapshots/splits/evidence/diffllm/cpt) and the CodeBERT tokenizer files. Qwen is downloaded from HF on the pod and checked against its hashes.
- Sequence: `scripts/pod_run.sh`
  1. R-base, E-base
  2. CPT
  3. R-diff, E-diff
  4. diffllm freeze, then one test evaluation
  5. L2-S validation, then L2 freeze and test
- Estimate: H100 SXM, 7–8 hours, about $25–30. **Waiting for a balance top-up** (currently $2.30).
