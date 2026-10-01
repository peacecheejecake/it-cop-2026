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

## Pod run 1 (`90hu1gx1wtmmtq`, US-NE-1, H100 SXM, $3.49/h, 2026-10-01 15:49–21:30 UTC)

- Bundle: `.cache/bundles/jit012-345b988.tar.gz` (140 MB, sha256 `cf3e1ecc…ef99`). Contents: code, CODE_SHA, data (snapshots/splits/evidence/diffllm/cpt) and the CodeBERT tokenizer files. Qwen is downloaded from HF on the pod and checked against its hashes.
- Sequence: `scripts/pod_run.sh`
  1. R-base, E-base
  2. CPT
  3. R-diff, E-diff
  4. diffllm freeze, then one test evaluation
  5. L2-S validation, then L2 freeze and test
- Estimate: H100 SXM, 7–8 hours, about $25–30. **Waiting for a balance top-up** (currently $2.30).

### Results of run 1

- **First attempt failed (16:02)**: the R-base SFT stopped at step 43 with a non-finite value. The step loss was finite, but the **gradient norm was NaN**.
  - Diagnostic (`diag_sft.py`, replay with per-step logging): reproducible at update 41 → 42.
    - Switching to right padding left it unchanged.
    - **Running each example of that update alone gave finite gradients under both sdpa and eager.** Only padded micro-batches (4 examples of different lengths) produced the NaN.
  - Fixes:
    - main `432c951`: right padding, last-token gather, abort on a non-finite gradient.
    - main `0d0fdb8`: **SFT micro-batch 1 × accumulation 16**, i.e. no padding and the same effective batch of 16. Recorded in the config header before any SFT result.
  - Debugging cost: about $2.
- **Restarted (16:19)** with code `0d0fdb8` (CODE_SHA).

| Step | Finished | Result |
|---|---|---|
| R-base | 18:18 | selection-subset AP: 0.234 (step 505), **0.292 (step 1010, best)**, 0.291 (step 1011), 0.247 (step 1515), then early stop. **Full validation AP 0.3224**, ROC-AUC 0.836, R@5% 0.251, R@10% 0.400 |
| E-base | 18:32 | MLP over max-pooled embeddings. **Validation AP 0.2499**, ROC-AUC 0.766, R@5% 0.208, R@10% 0.328 |
| CPT | 21:25 | 381 updates, 49.9M tokens, loss 0.976 → 0.746, 10,340 s (4,830 tok/s) |

- **Known deviation (applies equally to all arms)**: in `sft_risk`, the every-505-steps evaluation (step 1010) and the end-of-epoch evaluation (step 1011) **run back to back**. The duplicate evaluation (0.291) counts against patience, so early stopping triggers one evaluation sooner. R-diff uses the same code and therefore the same rule. This will be fixed on main after this study ends.
- **Stop for balance**: when CPT finished, the balance was $7.67, below the $13 threshold. The gate script (`cpt_gate12`) stopped the pipeline before R-diff and pulled the artifacts. All 24 files match by sha256; the initial "mismatch" was only a GNU/BSD sort-order difference.
  - Pulled: CPT adapter (625 MB), R-base best LoRA, E-base MLP state.
  - **Pod deleted.** Cost about $20 (balance $27.30 → $7.38).
- Remaining: R-diff → E-diff → freeze → test (about 3.5 hours, about $12), then L2 (about $4). Resume on a new pod with the same code (`0d0fdb8`) after the balance is topped up.
