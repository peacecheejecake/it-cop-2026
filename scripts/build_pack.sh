#!/usr/bin/env bash
# Assemble the single-zip internal package from this worktree. Run from the worktree root after
# fetch_wheels.py, the bundle export and validate_extract.py. Prints the zip sha256.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NAME=jit-b3s-internal
OUT="$ROOT/pack-build/$NAME"
CODE_SHA="$(git -C "$ROOT" rev-parse HEAD)"
[ -z "$(git -C "$ROOT" status --porcelain -- codebert-diff-lab/src)" ] || { echo "codebert-diff-lab/src is dirty" >&2; exit 1; }
rm -rf "$OUT" "$ROOT/pack-build/$NAME.zip"
mkdir -p "$OUT"/{python,wheels,models,bundle,docs,src}
cp "$ROOT"/pack/{README.md,install.sh,jit.sh,smoke.sh} "$OUT"/
cp "$ROOT"/pack-build/python/cpython-3.11.*-install_only.tar.gz "$OUT/python/"
cp "$ROOT"/pack-build/wheels/*.whl "$ROOT/pack-build/wheels/requirements-offline.txt" "$OUT/wheels/"
(cd "$ROOT/codebert-diff-lab" && rm -rf dist && uv build --offline --wheel -q && cp dist/codebert_diff_lab-*.whl "$OUT/wheels/")
cp -RL "$ROOT/../.cache/models/codebert-base" "$OUT/models/codebert-base"
cp "$ROOT/codebert-diff-lab/export-b3s43/bundle.tar.gz" "$OUT/bundle/"
cp "$ROOT/results/validate/validate.json" "$OUT/docs/extract-fidelity-public-test.json"
cp "$ROOT/codebert-diff-lab/docs/internal-offline-eval.md" "$OUT/docs/"
git -C "$ROOT" archive --format=tar.gz --prefix=codebert-diff-lab/ -o "$OUT/src/codebert-diff-lab-$CODE_SHA.tar.gz" HEAD:codebert-diff-lab
cat > "$OUT/PROVENANCE.json" <<JSON
{
  "package": "$NAME",
  "built_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "code_commit": "$CODE_SHA",
  "study": "public-comparison-v3",
  "freeze_id": "b6afa59f3a2a9550",
  "model": {"variant": "B3-S (CodeBERT full fine-tuning + jit14 fusion head)", "seed": 43, "run_id": "733fb98bec2f83f4",
            "public_validation_ap": 0.6957, "public_test_ap_seed43": "see docs/extract-fidelity-public-test.json (orig)"},
  "bundle_sha256": "$(shasum -a 256 "$OUT/bundle/bundle.tar.gz" | cut -d' ' -f1)",
  "base_encoder": {"id": "microsoft/codebert-base", "revision": "3b0952feddeffad0063f274080e3c23d75e7eb39"},
  "python": "$(basename "$OUT"/python/*.tar.gz)",
  "torch": "2.14.1+cu126 (CUDA 12.6 build; CPU also works)",
  "training_data": "public JIT-Defects4J only (21 Java projects); no internal data used for training, selection or calibration"
}
JSON
(cd "$OUT" && find . -type f ! -name SHA256SUMS | sed 's|^\./||' | LC_ALL=C sort | while read -r f; do shasum -a 256 "$f"; done > SHA256SUMS)
(cd "$ROOT/pack-build" && zip -q -r -1 "$NAME.zip" "$NAME")
shasum -a 256 "$ROOT/pack-build/$NAME.zip" | tee "$ROOT/pack-build/$NAME.zip.sha256"
du -h "$ROOT/pack-build/$NAME.zip"
