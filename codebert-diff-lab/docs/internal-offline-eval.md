# Offline evaluation on internal data (evaluation-only)

This document describes how to score internal company changes **without any training, calibration or external transfer**, using the frozen study `public-comparison-v3` (freeze `b6afa59f3a2a9550`). It covers **DoD-Internal-Ready**: verified only with synthetic fixtures and with public test changes reformatted into the internal format. Actual internal evaluation (DoD-Internal-Evaluated, M7) requires separate approval to bring internal data in, and the code, language and deployment-mapping coverage must be reported.

## What to bring in

1. **Export bundle**: run `diff-lab study export … --data-dir <public data>`.
   - The bundle contains the frozen per-run state, best checkpoints, prompt manifests, the frozen public L1 demos, the study config, the freeze record and a sha256 manifest.
   - It contains no optimizer state, CPT heads or training data.
2. **Base models, checked against the hashes pinned in the config**:
   - CodeBERT: `microsoft/codebert-base` `3b0952fe…`, `pytorch_model.bin` sha256 `28b61fd8…`. Needed for B1–B5 (the tokenizer, and the encoder for B2–B5).
   - Qwen2.5-Coder-7B-Instruct: `c03e6d35…`, sha256 for all 10 files. Needed for L0/L1; if it is absent, those runs are reported unavailable.
3. **Code**: the matching commit of `codebert-diff-lab` and `uv.lock`, installed with `uv sync --frozen` (`--extra neural` for B2–B5 and L0/L1).

## Internal data format (parquet, no labels)

| Column | Type | Notes |
|---|---|---|
| `change_id` | str | Unique. Never printed to stdout. |
| `message` | str | Commit message. An empty string is allowed. |
| `added_lines` | list[str] | Added lines without the `+` marker. Sorted lexicographically, the same canonicalization as the public data. |
| `deleted_lines` | list[str] | Deleted lines. |
| `ns, nd, nf, entropy, la, ld, lt, fix, ndev, age, nuc, exp, rexp, sexp` | float | The jit14 features (meaning: `docs/m0-audit.md`, `llm.FEATURE_NAMES`). Per-row NaN is allowed and handled only by the frozen policy (training-median imputation; NaN as-is for LGBM; "unknown" for the LLM). |

- **Columns that look like labels** (`label`, `bug`, `defect`, `target`, etc.) cause the input to be rejected. The predictor does not read labels; label joining happens in a separate evaluator.
- **If any jit14 column is missing entirely**, every variant is reported unavailable. Zero-fill is not used.

## Commands

```bash
uv run diff-lab bundle unpack --bundle bundle.tar.gz --out bundle/        # safe extraction + full sha256 verification
uv run diff-lab bundle verify --bundle-dir bundle/                         # re-verify (tamper check)
uv run diff-lab predict --bundle-dir bundle/ --dataset internal.parquet --out-dir pred/ \
    --device cuda --encoder-path <codebert dir> --llm-path <qwen dir>      # --variants B3-S,... to select
```

Outputs:

- `pred/predictions.parquet`: change_id, variant, seed, run_id, score, score_semantics.
- `pred/offline-manifest.json`: run_status (ok/unavailable with reason), dataset coverage (missing-value counts, number of truncated renders, input sha256), `labels_read: false`, `fit_or_calibration: none`.
- stdout shows only counts.

## Guarantees and limits

- **Network**: blocked for the whole `predict` call (socket guard plus HF/transformers offline mode). Organization-level egress blocking must be applied separately; this code guard does not replace it.
- **No training or modification of any kind**: there is no command that trains, fits TF-IDF/scalers, calibrates, re-normalizes scores, or adds internal examples to demos or an index. L1 accepts only the frozen public demos whose IDs and content hashes match the manifest. L0 rejects demos.
- **Integrity**: every bundle file is re-hashed against the export manifest, and each checkpoint is re-checked against its generation pointer.
- **Score meaning**: scores are uncalibrated public-defect scores, not probabilities of internal incidents. Do not re-normalize them using internal min/max or means.
- **Reproduction check**: 300 public test changes were converted to the internal format and scored offline (MPS). Against the frozen test scores, B0/B1 match within 2e-16 and B2–B5 within 2e-6. L0/L1 were unavailable locally because Qwen was absent.
- **Not yet verified**: coverage of internal code languages, missing-feature patterns, deployment-level aggregation (FR-17) and the label observation window. These belong to M7 and must be reported separately.
