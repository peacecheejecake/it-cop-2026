# RUNLOG — 015-dl-internal-pack

Goal: bring only the best fine-tuned CodeBERT (B3-S, freeze `b6afa59f3a2a9550`) into the internal network as one zip, with
git commit extraction and a hand-label evaluator. Code: main `20bba96` (`internal extract / label-sheet / label-eval`,
`study export --variants/--seeds`).

1. `uv sync --offline --extra neural --extra dev` (codebert-diff-lab)
2. Export B3-S seed 43 (best validation AP 0.696):
   `diff-lab study export --study configs/studies/public-comparison-v3.yaml --freeze ../../011-dl-m6-final/codebert-diff-lab/freeze/freeze.json --artifacts-dir ../../011-dl-m6-final/codebert-diff-lab/artifacts --out export-b3s43/bundle.tar.gz --variants B3-S --seeds 43`
   → 12 files, sha256 `f1e85e52…35e0`; `bundle unpack` verified.
3. Extractor fidelity vs the package, all 21 cached mirrors (`.cache/git/jd4j`): 27,209/27,319 commits found; messages 100% equal;
   line-set Jaccard 0.95; ns/nd/nf/entropy/la/ld ≥ 99% exact; age 95%; lt/nuc 74%; exp ≈54% (constant offsets); ndev 21%.
4. End-to-end on the public test (already opened in 011; this checks inputs, not models):
   `python ../scripts/validate_extract.py ../../011-dl-m6-final/codebert-diff-lab ../../.cache/git/jd4j bundle-b3s43 ../../.cache/models/codebert-base ../results/validate mps`
   → `results/validate/validate.json` (5,459 of 5,480 test commits, 458 positives).
5. `scripts/fetch_wheels.py` (Linux x86_64 cp311, torch 2.14.1+cu126), python-build-standalone 3.11.17 (sha256 checked).
   Packaging (`scripts/build_pack.sh`) **not run**: stopped after step 4 (see RESULTS.md).
