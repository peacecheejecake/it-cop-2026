# Study `diffllm-v1` — change-aware pretraining and risk alignment for an LLM (exploratory, pre-registered 2026-10-01)

**Purpose:** reproduce, on public data, the comparison structure of the risk-alignment paper (iCodeLlama/iDiffLlama). Two questions:

1. Does **diff (before → after) pretraining** of the LLM help, the "change-aware" claim (**Q1**)?
2. Does **risk alignment** (SFT on labels, scoring with P(1)) beat **embedding + MLP** (**Q2**)?

**Status:** this is a new study **beyond** spec v0.2, which excludes LLM training, and it **runs after the v3 public test has been opened**. So it is an **exploratory follow-up study**, not the untouched v3 comparison (spec §12). Every choice below is fixed before training. The test set is evaluated only once, after this study's own freeze.

## 1. Data

**Labeled data:** JIT-Defects4J `jitd4j-audit1` with **the same membership as split `upstream-clean2`**: train 16,184 / valid 5,465 / test 5,480. Labels are as provided.

**Input representation `full-diff-v1`:** the unified diff extracted from the public git repository for each labeled commit, `git show <sha> --format= -U3 --no-color` against the first parent. It keeps file headers, hunk order and context lines, i.e. **before → after**.
- The prompt is the commit message (title + body, at most 128 tokens) followed by the diff, at most 2,048 Qwen tokens in total.
- When over budget, whole hunks are dropped from the end, and the dropped amount is recorded.
- Binary files, deleted binaries and lock files are excluded.
- Changes whose diff cannot be extracted are recorded as unavailable; they are not dropped silently.

**Pretraining (CPT) corpus `apache-disjoint-diffs-v1`:** commits from **Java Apache projects disjoint from JD4J**: zookeeper, zeppelin, activemq, kafka, cassandra, groovy.
- Disjoint repositories are used so that later fix commits from JD4J's own history (the basis of its SZZ labels) never enter pretraining.
- Non-merge commits only, in the same text format (message + diff, at most 2,048 tokens per commit).
- Duplicate diff hashes are removed, and any commit whose diff hash matches a JD4J commit diff is excluded.
- Commits are drawn in order of `sha256("cpt-diff-v1:" + repo + sha)` until **50M tokens**. The token counts per repository are recorded.

**Not used:** internal data, the JD4J test set, or any data for selection other than JD4J valid.

## 2. Models and arms

- **Base:** Qwen2.5-Coder-7B-Instruct `c03e6d358207e414f1eca0bb1891e29f1db0e242` (pinned in v3).
- **14B arm (pre-registered, run after the 7B results):** Qwen2.5-Coder-14B-Instruct `aedcc2d42b622764e023cf882b6652e646b95671`, Apache-2.0, 6 shards with sha256 recorded via their LFS oids. After the 7B results, the user chooses between:
  - **minimum:** `R-base-14B` only
  - **full:** the full 2×2 at 14B

**2×2 (7B):**

| | Embedding + MLP (method A) | Risk alignment (SFT, P(1)) |
|---|---|---|
| base model | `E-base` | `R-base` (≈ iCodeLlama risk-aligned) |
| + diff CPT | `E-diff` | `R-diff` (≈ iDiffLlama risk-aligned) |

- **diff CPT:** causal-LM (next-token) loss over the CPT corpus.
  - LoRA: r=64, α=128, dropout 0.05, applied to all linear layers (q, k, v, o, gate, up, down).
  - lr 1e-4 with cosine decay, 3% warmup, bf16.
  - Sequences are packed to 2,048 tokens, about 131k tokens per update, for about 380 updates (50M tokens).
  - The LoRA weights are then merged into the base to give the `diff` model.
- **Risk alignment:** SFT with a fresh LoRA (r=16, α=32, dropout 0.05, all linear layers) on top of the base or diff model.
  - Prompt: one fixed system instruction plus `[DRS]\nTitle: …\nCode changes:\n<diff>\n[/DRS]`, then the assistant answers `1` or `0`.
  - **Loss is computed on the label token only.** Training uses the natural class distribution (same as the v3 B variants, no resampling).
  - lr 1e-4, at most 2 epochs, effective batch 16, validation AP every half epoch; the best checkpoint is kept, and ties go to the earlier one.
  - Score = P(1) / (P(0) + P(1)), the same normalization as L0/L1.
- **Embedding + MLP:** the base or diff model is frozen.
  - Embedding: max pooling of the last hidden states over the prompt tokens (the paper's method A).
  - Head: a 3-layer MLP (hidden 512, ReLU, dropout 0.1), BCE loss, Adam lr 1e-3, at most 50 epochs, early stopping on validation AP with patience 5.
- **Seed:** 42 (one seed, user decision). With one seed, seed variance cannot be estimated, so differences are reported with the project bootstrap only, and this limitation is stated explicitly.

## 3. Evaluation

- **Selection:** JD4J validation only (checkpoint and MLP epoch).
- **Freeze:** all study arms are frozen together, then the public test is evaluated once.
- **Metrics:** the same as v3 (AP, ROC-AUC, Recall@5% and Recall@10% = the paper's top-5% and top-10% gates, F1 at the validation max-F1 threshold).
- **Pre-registered comparisons:**
  - **Q1:** R-diff − R-base and E-diff − E-base.
  - **Q2:** R-base − E-base and R-diff − E-diff.
  - **References:**
    - v3 L0/L1 (the same 7B with no alignment, on the EvidenceView)
    - v3 B3-S (CodeBERT FT + structured features)
    - B0-LR
    - The input representation and feature use differ, so these are **reference comparisons only**.
- **Size (14B arm):** R-base-14B vs R-diff-7B (the paper's "bigger general model vs smaller change-aware model").
- **Leakage and limitation notes:**
  - The labeled repositories (JD4J) and the CPT repositories are disjoint.
  - The JD4J split is not time-ordered.
  - The CPT corpus is several orders of magnitude smaller than the paper's.
  - There is no test plan field.

## 4. Cost guards

- GPU: Runpod H100 SXM. 7B arms: about 6–8 hours, about $20–25. The 14B arm is decided separately.
- Before each GPU step, check that the balance exceeds the estimate. Pull small results immediately after each run, and fetch weights (LoRA adapters) last.
