# Study `diffllm-v2` — input representation, then model size, for a risk-aligned LLM (exploratory, pre-registered 2026-10-02)

**Why:** diffllm-v1 (exp 012) left aligned 7B models far below CodeBERT (R-diff − B3-S = −0.315 test AP). Its post-hoc diagnostics (validation only) point at the **input**, not the model:
- With the same TF-IDF model, the v1 full diff is much weaker than the v3 EvidenceView: 0.239 vs 0.338 text only, and 0.358 vs 0.546 with jit14.
- Adding jit14 to R-base's score gives +0.07 (0.322 → 0.394).

So v2 first puts the LLM on **the same input as B1–B5**, then asks whether a larger model helps. The user chose this order (input first, then model) and the model candidate (Qwen2.5-Coder-14B only) on 2026-10-02.

**Status:** exploratory, registered **after** the v3 and diffllm-v1 public tests were opened. It is not an untouched comparison (spec §12). Every choice below is fixed before v2 training; v2's test set is evaluated once, after v2's own freeze.

## 1. Data and input

- **Labeled data:** JD4J `jitd4j-audit1`, split `upstream-clean2` (train 16,184 / valid 5,465 / test 5,480), as in v3 and v1.
- **Representation `evidence-v2`:** the v3 EvidenceView `message-add-del-text-v2` (512 CodeBERT tokens, parquet sha256 `d24c807d…`). It is not truncated again.
- **Two prompts, fixed in `configs/studies/diffllm-v2.yaml`:**
  - `ev-jit14`: **exactly the L0/L1 prompt** (system prompt plus the jit14 features serialized by name, then the evidence). This is checked by a test.
  - `ev`: the same template without the metrics block, and without the words "and change metrics" in the system prompt.
- **Prompt length:** about 565 Qwen tokens on average, at most 1,437. v1 averaged about 1,100. `max_prompt_tokens` is 1,536, and the code refuses any prompt over it.
- **No CPT, no internal data.** Selection uses JD4J validation only.

## 2. Arms

| arm | model | prompt | stage |
|---|---|---|---|
| `R-ev` | Qwen2.5-Coder-7B-Instruct `c03e6d35` | `ev` | 1 |
| `R-ev-jit14` | Qwen2.5-Coder-7B-Instruct `c03e6d35` | `ev-jit14` | 1 |
| `R-14B` | Qwen2.5-Coder-14B-Instruct `aedcc2d4` (Apache-2.0, 14.8B, bf16 about 30 GB) | **stage-1 winner** | 2 |

- **Stage-1 winner:** the stage-1 arm with the higher **full-validation AP**; a tie goes to `R-ev-jit14`. The choice is resolved by the code and recorded in `R-14B/run.json`.
- **Risk SFT is the same as v1 R-base:**
  - LoRA r=16, α=32, dropout 0.05 on all linear layers.
  - Loss on the label token only; natural class distribution.
  - lr 1e-4, at most 2 epochs, micro-batch 1 × accumulation 16, patience 2.
  - Selection on the fixed 2,000-change validation subset.
  - Score = P(1) / (P(0) + P(1)).
- **Only change from v1: `eval_schedule: even`.** Evaluations happen at 2 evenly spaced points per epoch (updates 506 and 1,011). In v1, the end-of-epoch evaluation ran one update after the periodic one, counted twice against patience, and stopped R-base early.
- **Seed:** 42, one seed. Seed variance is not estimated, as in v1.

## 3. Evaluation and comparisons

- **Metrics and freeze:** the same as v1.
  - Metrics: AP, ROC-AUC, Recall@5%/10%, and F1 at the validation max-F1 threshold.
  - All three arms are frozen together, then the public test is evaluated once.
  - Intervals use the project bootstrap (2,000 reps).
- **Pre-registered comparisons:**
  - **P1 (input):** `R-ev` − v1 `R-base`. Same model and SFT; the input is the EvidenceView vs the full diff. This crosses studies, so it is a reference comparison.
  - **P2 (jit14 in the prompt):** `R-ev-jit14` − `R-ev`.
  - **P3 (size):** `R-14B` − the 7B arm with the same prompt.
- **References:**
  - v3 L0-S: the same 7B and the same `ev-jit14` prompt, with no SFT.
  - v3 B3-S, B1-TFIDF-S and B0-LR.
  - v1 R-base and R-diff.
- **What this does not test:** diff CPT scale-up (the paper's 855k-diff corpus scale) and other model families. Those wait for v2's result and are decided separately.

## 4. Cost guards

- **Hardware and time:** Runpod H100 SXM ($3.49/h at v1 time).
  - Each 7B arm takes about 1 h (half of v1's prompt length).
  - The 14B arm takes about 2–2.5 h.
  - Downloads, freeze and test add about 1 h.
  - **Total: about 5 h, about $18–20.**
- **Balance:** before starting, check that it exceeds the estimate plus the $13 margin used in exp 012. Pull small results after each arm; fetch adapters last.
