# M0 audit — JIT-Defects4J (JIT-Fine package) — 2026-10-01

Spec tasks T00/T01, requirements §8. Numbers below come from running `diff-lab data import/audit/split` on the
approved archive (sha256 `9e5ca1a3…2b47`, upstream `jacknichao/JIT-Fine@584799fd`). The experiment worktree that ran
the pinned commands holds the generated `audit.json` / manifests.

## Source and license

| Item | Finding | Status |
|---|---|---|
| Archive | `data.zip` 75,326,372 bytes, 99 members, only `.pkl` + `ngram/*.txt` | pinned by sha256 |
| License | upstream repository declares **no license** | research analysis only; internal import needs policy check |
| Paper counts | BiCC-BERT abstract 27,391 vs §4.2 27,319 (S01) | **package = 27,319** (16,374 / 5,465 / 5,480) |

## Format (decides the adapter)

| Item | Finding | Consequence |
|---|---|---|
| `changes_{split}.pkl` | 4 aligned lists `[commit_id, label(float), message, {added_code, removed_code}]`, builtins only | no globals needed |
| added/removed code | **Python `set`s of upstream-tokenized lines** | `representation_kind=preprocessed_lines`; original line order and duplicate lines are lost; no file/hunk mapping |
| set iteration order | depends on PYTHONHASHSEED | adapter sorts lines lexicographically; `edits.order` = sort rank, not patch order |
| `features_{split}.pkl` | pandas DataFrame, 25 columns, **all values stored as strings** | parsed to float; `fix` 'True'/'False' → 1/0 |
| unpickle globals | only pandas DataFrame/BlockManager/Index/RangeIndex, numpy ndarray/dtype/_reconstruct/_frombuffer, builtins.slice | allowlist Unpickler in a network-denied sandbox (`sandbox-exec`), verified that DNS fails inside |
| line-level pickle | `changes_complete_buggy_line_level.pkl` = localisation ground truth | **never imported** |
| PII | author_name/author_email | dropped |

## Labels, IDs, joins

- changes ↔ features: same ID set per split, unique IDs, **0 label disagreements**; labels ∈ {0,1}.
- Positives: train 1,390 (8.5%), valid 467 (8.5%), test 475 (8.7%); 21 projects in every split (within-project split).
- change_id = `jitd4j:<project>:<commit_sha>`.

## Time

- author date present (unix), committer date absent → `committed_at=null`, `time_provenance=author_date_from_package_unverified`.
- **Provided split is not time-ordered**: train 2001-09 → 2015-09, valid 2006-10 → 2018-01, test overlaps both.
  Results are reported as `upstream_holdout`, never as a historical/time-ordered evaluation.

## Features (allowlist jit14 = NS ND NF Entropy LA LD LT FIX NDEV AGE NUC EXP REXP SEXP)

- All 14 parse as numbers. **Negative values that cannot be physical: `lt` 1,580 rows (min −1,920), `age` 74 rows (min −95.9).**
  Kept as provided (no silent clipping); LR uses a signed log1p transform.
- `fix` aligns with the message-derived `classification=Corrective` column → computed from the current commit message,
  available at prediction time; still marked `provided_unverified`.
- `classification` (message-derived category) is not in the allowlist.

## Duplicates and conflicts (label-free keys)

| Key | rows in dup groups | groups | cross-split groups | label-conflicting groups |
|---|---|---|---|---|
| exact (message + sorted lines) | 559 | 237 | 17 (train-valid 11, train-test 6, valid-test 6) | 5 |
| code-identical (lines only) | 1,095 | 336 | 48 (train-valid 31, train-test 22, valid-test 25) | 10 |
| normalized whitespace | same as exact | | | |

Content: empty message 16, add-only 5,421, delete-only 483, no code lines 0; median 10 added / 3 deleted lines.

## Controlled split `upstream-clean1`

Valid/test membership kept exactly as provided (valid-test duplicates are reported, not hidden). Train rows exact- or
code-identical to any valid/test row removed from train and CPT: **190 rows (40 exact, 150 code), 3 positives**.
Result: train 16,184 (1,387 pos), valid 5,465 (467), test 5,480 (475); CPT-train 15,375 / CPT-dev 809 (5% by
sha256 rank, label-free).

## EvidenceView (`message-add-del-text-v2`, 512 / message 64, CodeBERT `3b0952fe`)

Native tokenizer only (no added tokens). 27,129 changes; 17,660 (65%) keep all code lines; 10,057 truncated; 1,471
messages truncated; median 280 encoder tokens. Line-level kept ratio ≈30% (dominated by very large commits).
Coverage is within the preprocessed package, not repository-diff coverage.

## Open items

- License/redistribution: unresolved (no upstream license).
- Upstream `only_adds` and reference-track audit (JIT-Fine / JIT-BiCC code paths): not done (reference track not started).
- Base-model pretraining overlap with these public repositories: cannot be ruled out (limitation).
- LightGBM on macOS requires `libomp`.
