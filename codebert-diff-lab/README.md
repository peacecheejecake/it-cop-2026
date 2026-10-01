# codebert-diff-lab

Implementation of the **CodeBERT Diff Lab spec v0.2** (`docs/spec-v0.2/`, copied verbatim with `SHA256SUMS`).
Public-only training, public-validation selection, sealed public test until study freeze, internal data evaluation-only.

## Status (what exists vs. what the spec plans)

| Milestone | Scope | State |
|---|---|---|
| M0 | data/source audit | done for JIT-Defects4J — `docs/m0-audit.md` |
| M1 | uv/CLI/strict config/registry v2/policy/tests | done (CPU, network-free tests) |
| M2 | import → audit → split → EvidenceView → B0-LR / B0-LGBM / B1-TFIDF-S on public validation | done (exp 006); B0-LGBM needs `libomp` on macOS |
| M3 | B2-S (frozen CodeBERT + fusion head) / B3-S (full FT), resume, T14 profile | implemented, CPU/MPS-verified on tiny + real CodeBERT; CUDA run pending |
| M4 | B4-S diff-MLM CPT (10M-token plan, structure-protected masking, resume, encoder-only export) → B3-S fine-tuning | implemented, CPU-tested; real CodeBERT 1-update check on MPS; CUDA run pending |
| M5–M6 | MLM+RMI CPT, local LLM L0/L1, freeze/test/report/export | not implemented yet |

## Commands (run from this directory)

```bash
uv sync --system-certs --extra lgbm --extra dev          # Python 3.11, locked by uv.lock
uv run diff-lab doctor
uv run diff-lab data import --source jit-defects4j --archive <data.zip> \
    --approval configs/sources/jit-defects4j.yaml --snapshot-id jitd4j-audit1
uv run diff-lab data audit --snapshot-id jitd4j-audit1
uv run diff-lab data split --snapshot-id jitd4j-audit1 --split-id upstream-clean1
uv run diff-lab evidence build --study configs/studies/public-comparison-v2.yaml
uv run diff-lab experiment run --study configs/studies/public-comparison-v2.yaml --models B0-LR,B1-TFIDF-S --seeds 42
uv run diff-lab experiment profile --study configs/studies/public-comparison-v2.yaml --precision fp32 --updates 100 --out profile/fp32.json
uv run diff-lab study summary --study configs/studies/public-comparison-v2.yaml
uv run pytest
```

`--system-certs` lets uv use the OS trust store (needed behind a TLS-inspecting proxy). `data/` and `artifacts/` are
git-ignored; snapshots, splits and evidence are immutable and content-hashed; runs live in `artifacts/runs/<run_id>/`.

## Layout

```text
configs/registry/matrix-v2.json   variant registry (= spec experiment-registry.json)
configs/sources/*.yaml            approved public sources (archive sha256, roles, license status)
configs/studies/*.yaml            study registration; PIN_REQUIRED blocks only variants that read that section
src/diff_lab/
  adapters/                       jit_defects4j (zip safety + sandboxed allowlist unpickler worker)
  data/audit.py, data/splits.py   label-free duplicate/temporal/feature audit; upstream-clean1 split
  evidence.py                     matched EvidenceView, renderer message-add-del-text-v2
  features.py, models.py          train-only structured pipeline; B0-LR, B0-LGBM, B1-TFIDF-S
  neural.py                        B2-S/B3-S/B4-S fusion head, frozen-encoder cache, training loop, resume, T14 profile
  cpt.py                           diff-MLM CPT corpus/masking/plan/resume and encoder export (B4-S)
  metrics.py                      AP / ROC-AUC / Recall@q (ceil, hash ties) / validation max-F1 threshold
  policy.py, config.py, registry.py
  runner.py, cli/main.py          run-folder contract, label-access ledger, public-test gate
```
