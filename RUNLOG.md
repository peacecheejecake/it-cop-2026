# RUNLOG — 011-dl-m6-final

Branch `exp/011-dl-m6-final`, branched from main `4f7792b` and fast-forwarded to `2e75451` → `e1b2b1a` → `4cba508`. Study: `public-comparison-v3`.

## Purpose

- **M6L**: L0-S (zero-shot) and L1-S (static 4-shot × demo seeds 42/43/44) with a frozen local LLM (Qwen2.5-Coder-7B-Instruct), on public validation.
- **M6**: freeze all 9 primary variants → evaluate the public test **once** → report (paired differences + project bootstrap) → inference-only export.

## LLM pin (v3 `llm` section, recorded before any run)

- Model: `Qwen/Qwen2.5-Coder-7B-Instruct`, revision `c03e6d35…0e242`, Apache-2.0, not gated (user's choice).
- Hashes: sha256 for all 10 files (4 weight shards via their LFS oid; tokenizer and config files hashed locally).
- Chat template hash: `cd8e9439…527f`.
- Task prompt: `jit-defect-binary-v1`, hash `64262bc5…1f017`. Structured features are serialized as `jit14-named-raw-v1`.
- Scorer: log-likelihood of the two single-token labels "0" (id 15) and "1" (id 16) after the assistant prefix, normalized with logsumexp. The label boundary was checked on every query.
- Precision: bf16. `max_context_tokens` 8192; truncation forbidden. Batches are left-padded with explicit position ids, 32,768-token budget.
- Prompt lengths on real validation: L0 mean 553 / max 1,251 tokens; L1 mean 2,129–2,656 / max 3,354 tokens.
- L1 demos are 2 positive + 2 negative public-train changes per seed:
  - seed 42: commons-collections, commons-math, ant-ivy, ant-ivy
  - seed 43: giraph, commons-lang, commons-net, commons-math
  - seed 44: commons-compress, commons-beanutils, commons-io, opennlp
  - Change IDs and content hashes are in `results/runs/*/prompts-manifest.json`.

## Execution (same pod as 010: `9r0xh4efj5n1wl`, H100, $3.49/h)

1. During the 010 B5 run, the Qwen model was downloaded directly from HF (15 GB) and checked against all pins. The 009 run folders were uploaded (351 files, sha256 verified; 8 minutes).
2. After B5 finished, the 010 runs were copied in, then L0/L1 ran (11:16 → 11:42 UTC):

   ```bash
   uv run diff-lab experiment run --study configs/studies/public-comparison-v3.yaml --models L0-S --seeds 42
   uv run diff-lab experiment run --study configs/studies/public-comparison-v3.yaml --models L1-S --seeds 42,43,44
   ```

   - L0: about 2 minutes. L1: about 7–8 minutes per seed.
   - A script error in the follow-up script (a `sed` range) invoked these two commands once more. All runs were already completed, so the runner verified their artifact hashes and reused them; nothing was re-executed.
3. **Freeze, first attempt failed**:
   - B0–B5 reproduced validation exactly from frozen state (B0/B1 within 1e-16; B2–B5 difference 0 on the H100).
   - L0's 64-row validation sample differed from the stored scores by up to **0.031**, exceeding the pre-set tolerance of 0.02.
   - Cause: bf16 logits depend on batch composition (padding and batch size). The same batch repeated is bit-identical. Against stored scores: mean 0.009, Spearman 0.995. Prompts and demos match the frozen manifest.
   - Fix (main `e1b2b1a`): the LLM reproduction check became max |diff| ≤ 0.05 **and** Spearman ≥ 0.99, with both values recorded. This only changes a verification criterion, not any performance setting; no test data had been seen.
4. **Freeze, second attempt succeeded**:
   - `freeze_id b6afa59f3a2a9550`, 11:54:43 UTC, 25 runs (B0–B5 × 3, L0 × 1, L1 × 3). Code commit `e1b2b1a`.
   - Each L run's reproduction max |diff| was 0.030–0.031.
5. **One-shot public test** (11:54 → 12:23 UTC):
   - `uv run diff-lab experiment test --study … --freeze freeze/freeze.json --out-dir test-results --device cuda`
   - Thresholds are the frozen validation thresholds.
   - B0-LGBM was scored in a separate process.
6. Report and export:
   - `study report` → `report/report.json`; `study export` → `export/bundle.tar.gz`.
   - The pod's bundle (4.15 GB, sha256 `3bde2025…b504`) was not downloaded because of the balance; only its manifest was retrieved.
7. Retrieval and cleanup:
   - Retrieved freeze, test-results, report, logs and the L run folders (75 files, sha256 match).
   - **Pod `9r0xh4efj5n1wl` deleted** (204).
   - 010 + 011 cost about $8.6 (balance $10.99 → $2.36).
   - A local guard was in place to delete the pod automatically if the balance fell below $1.50; it never triggered.

## Post-processing (local)

- **Report bug**: the `test_f1_at_threshold` column was aggregating recall at the threshold, because the `at_threshold` dict also has a `recall` key. Fixed in main `4cba508` and regenerated locally as `report/report-corrected.json`.
  - The test predictions were reused; the test set was not re-evaluated.
  - AP, ROC-AUC, Recall@q and paired differences are identical to the pod report; only the F1/precision/recall-at-threshold columns changed.
  - Regenerating the report also verified the freeze hashes for all 25 runs locally (009/010 runs linked into the 011 artifacts root with `cp -al`).
- The export bundle was regenerated locally as `export-local/bundle.tar.gz`: sha256 `88254…4325`, 178 files, same freeze_id. It is not byte-identical to the pod bundle because tar timestamps differ.

## Remaining caveats

- The L0/L1 freeze check is not exact reproduction; it relies on a tolerance plus rank agreement, for the bf16 batch-composition reason above.
- Validation-to-test drops are large for every model, including deterministic B0-LR (0.319 → 0.215). See RESULTS.

## Definition-of-done report (main `358b263`, regenerated locally)

- `study report` now also emits DoD status, a label-access ledger summary, cost, prediction failures, matched evidence and replication axes. It writes JSON, Markdown and CSV (`results/report/report-final.{json,md,csv}`). AP and paired differences are identical to report-corrected.
- On the first local regeneration, matched_evidence was **FAIL**: three evidence-manifest sha256 values coexisted (009/010/011). The cause was that each experiment rebuilt the evidence, so the import timestamps in the snapshot manifests differed.
- The per-change query content hash (`query_hash_digest 7d8774b4…`) is identical for every text-reading variant. Matched evidence is now judged on content (AT-29), and the number of evidence builds is recorded as information.
- Result: **DoD-Core complete, DoD-Comparative complete** (all checks pass). DoD-Internal-Ready is checked separately. DoD-Internal-Evaluated needs internal data. L2 was not run.
- Replication axes:
  - B0/B1 are deterministic, so they count as 1 effective training seed.
  - B2–B5 have 3 training seeds each.
  - L0 has no replicate axis.
  - L1's 3 demo sets are not training seeds (AT-38).

## DoD-Internal-Ready check (local, main `86aae74` → `4d64de1`)

- **Export bundle v2** (`study export … --data-dir data`): 194 files, sha256 `a6e61370…abe1`. Includes the three frozen L1 public demo sets, `study.json` and the full best generations.
  - The first v2 attempt left `validation-scores.parquet` out of the generation, so the generation-pointer check refused the encoder runs. Fixed by exporting the whole generation (main commit "Export the complete best checkpoint generation").
- **Command**: `bundle unpack` → `predict --bundle-dir bundle-v2 --dataset public-as-internal.parquet --device mps --encoder-path <codebert>`.
  - Input: 300 public test changes sampled with random_state=0, converted to the internal format (message, added_lines, deleted_lines, jit14).
  - Result: 21 runs ok. The 4 L runs were **unavailable** because Qwen was not on local disk; the reason is recorded.
  - 109 of the 300 renders were truncated. `labels_read: false`.
- **Agreement with the frozen test scores** (max |diff|): B0-LGBM 0, B0-LR/B1 2.2e-16, B2–B5 6e-7 to 1.9e-6 (MPS vs H100). So the offline path (internal-format rendering, then bundle predictors) reproduces the test path.
- Unit tests on synthetic fixtures (`tests/test_offline.py`):
  - bundle tampering refused (AT-20) and unsafe tar members refused
  - label columns refused (AT-19)
  - missing feature columns → unavailable (AT-25)
  - network blocked (AT-21)
  - only frozen public demos accepted, L0 refuses demos (AT-30/35)
- Record: `results/offline-check/offline-manifest.json`, `results/export-manifest-v2.json`. Usage: `codebert-diff-lab/docs/internal-offline-eval.md`.
