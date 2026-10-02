# Plan: redo the public comparison on leak-free text (study `public-comparison-v4-gitlines`)

Registered 2026-10-02, before any v4 model has seen the test split. Background: `docs/leak-finding-2026-10-02.md`.

## 1. Goal

1. Produce a B3-S model trained on text that exists at prediction time, for the internal trial.
2. Re-measure the v3 research questions on that text, so the record states what is known and what is not.

## 2. Fixed protocol

- **Data:** snapshot `jitd4j-git1`, split `upstream-clean2` (rebuilt on that snapshot). 27,209 commits: train 16,305 / valid 5,445 / test 5,459, base rate 0.084.
  - IDs, labels and the provided split come from the package. Message, lines and jit14 come from git (`gitextract-jd4j-v1`).
  - The coverage check (`tools/representation_leak_check.py`) is at the base rate on this snapshot.
- **Hyperparameters:** copied unchanged from v3 for every variant. Nothing is tuned for v4.
- **Selection:** validation only. All registered variants and seeds are frozen before the test is scored, and the test is scored once.
- **Disclosure:** the same test commits were scored once in v3 with the package text. No v4 model has scored the v4 test inputs. v3's head learning rate was chosen on leaked validation text; it is kept as is and reported as such.
- **Comparisons (same as v3):** B1 − B0-LR, B2 − B1, B3 − B2, B3 − B1, B4 − B3, B5 − B4, B5 − B3, L1 − L0, B3 − L1. Seed-paired differences, project bootstrap with 2,000 resamples.
- **Added reference:** exp 012 R-diff (Qwen 7B LoRA on git full diffs, test AP 0.283) is reported beside v4 as a leak-free result from another input form. Its cohort differs by 21 test commits.

## 3. Steps

| step | what | where | status / estimate |
|---|---|---|---|
| 1 | B0-LR, B0-LGBM, B1-TFIDF-S | local CPU | done: validation AP 0.302 / 0.240 / 0.345 |
| 2 | B3-S, seeds 42/43/44 | H100 pod `jit016-v4-gitlines` | running; about 1 h, $4 |
| 3 | Extend the v4 registration to all 9 variants: add the `cpt` and `llm` sections copied from v3. Run ids of steps 1–2 do not change (the variant list is not part of a run's identity). | main | before step 4 |
| 4 | B2-S, B4-S, B5-S, seeds 42/43/44 | same pod, right after step 2 | about 2.5 h, $9 |
| 5 | L0-S, L1-S (Qwen2.5-Coder-7B, pinned revision; 15 GB download to the pod) | same pod | about 1.5 h, $5–6 |
| 6 | `study freeze` → `experiment test` (once) → `study report` | pod or local MPS | under 30 min |
| 7 | Results document `docs/results-v4-<date>.md`; correction notes on the v3 documents | main | — |
| 8 | Internal package (zip) with the frozen model chosen by the rule in section 4 | exp 015 worktree | CPU only |

Total GPU estimate: 5–6 h, **$18–22** at $3.49/h.

**Budget warning.** The balance was $26.25 at 2026-10-02 08:45 UTC, and pods 013 and 014 are spending $4.25/h in addition to this pod. With all three running the balance lasts about 3.4 h, less than steps 2–5 need. Steps 4–5 start only after the user confirms the budget (top-up, or 013/014 finished or stopped). Small result files are pulled after every run; weights last.

## 4. Rules fixed in advance

- **Model for the internal trial:** the CodeBERT variant (B2–B5) with the highest mean validation AP; among its seeds, the one with the highest validation AP. If step 4 is delayed, the package is built from B3-S alone after freezing the variants finished so far, and that freeze is recorded as a partial one (a later full freeze gets a new freeze id).
- **If no text variant beats B0-LR on validation by more than the seed spread:** the package ships B0-LR as well, and the internal trial reports both.
- **Interpretation:** a difference is called established only if its project-bootstrap 95% interval excludes 0. v3 conclusions are neither confirmed nor refuted until step 6.

## 5. Follow-up studies (not started; each needs its own approval)

| study | what must change | estimate |
|---|---|---|
| 013 diffllm-v2 | EvidenceView arms must read `jitd4j-git1` evidence. Full-diff arms are unaffected. Results from the current run on package EvidenceView are kept as a record of the leaked input only. | about 6 h, $13 |
| 014 CPT-100M | Corpus and downstream fine-tuning on `jitd4j-git1`. Its pre-registered gate ("mean validation AP above B3 by 0.010") must be restated against the v4 B3-S value. | about 5 h, $11–12 |
| L2 extension | Index and demos from `jitd4j-git1` train. | not estimated |

## 6. Documents to correct after step 6

- `docs/results-final-2026-10-01.md`, `docs/results-2026-10-01.md`: banner pointing to the finding (added 2026-10-02) and, later, to the v4 results.
- `AGENTS.md` §0 and the experiment table: v3 text results marked as measured on label-dependent text.
- `docs/internal-offline-eval.md`: the "reproduction check" there compared package-format inputs with themselves; add that real git inputs differ and name the v4 bundle.
- `DATASETS.md`: known issue for JIT-Defects4J.
