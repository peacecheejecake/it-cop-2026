#!/usr/bin/env bash
# Move the git-ignored state that later experiments need to another machine.
#
#   tools/handoff_transfer.sh pack <out_dir>       # on the old machine, from the repo root
#   tools/handoff_transfer.sh restore <in_dir>     # on the new machine, from the root of a fresh clone
#
# pack writes one uncompressed tar per item plus SHA256SUMS. restore verifies the sums, creates the experiment
# worktrees from their branches (git refuses to add a worktree into a non-empty directory, so this comes first),
# then extracts every tar at the repo root. Nothing is deleted on either side.
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

# name|paths (relative to the repo root)|why
ITEMS=(
  "cache-models|experiments/.cache/models|CodeBERT (pinned) and the Qwen tokenizer"
  "cache-git|experiments/.cache/git|JD4J mirrors (gitextract, rebuild-git) and the 6 CPT-corpus mirrors"
  "cache-raw|experiments/.cache/raw|JIT-Fine data.zip (approved archive, for re-deriving jitd4j-audit1)"
  "e016-data|experiments/016-dl-v4-gitlines/codebert-diff-lab/data|snapshots jitd4j-audit1/jitd4j-git1, split, evidence"
  "e016-results|experiments/016-dl-v4-gitlines/codebert-diff-lab/freeze experiments/016-dl-v4-gitlines/codebert-diff-lab/test-results experiments/016-dl-v4-gitlines/codebert-diff-lab/report|v4 freeze 7340ee71c876c8b7, test predictions, report"
  "e016-runs|experiments/016-dl-v4-gitlines/codebert-diff-lab/artifacts/runs|v4 runs with best checkpoints (needed for export and external-test scoring)"
  "e016-cpt|experiments/016-dl-v4-gitlines/codebert-diff-lab/artifacts/cpt|v4 B4/B5 CPT encoders (only for re-fine-tuning)"
  "e012-fulldiff|experiments/012-dl-diffllm/codebert-diff-lab/data/fulldiff|git show -U3 diffs of all JD4J commits (Study M input)"
  "e012-cpt|experiments/012-dl-diffllm/codebert-diff-lab/artifacts/diffllm/cpt|50M-token Qwen 7B diff-CPT adapter (Study M R-diff)"
  "e012-corpus|experiments/012-dl-diffllm/codebert-diff-lab/data/cpt|CPT corpus apache-disjoint-diffs-v1"
  "e018-data|experiments/018-dl-study-m/codebert-diff-lab/data|Study M stage-1 view on the v4 cohort"
  "e015-python|experiments/015-dl-internal-pack/pack-build/python|Linux CPython 3.11 for the internal package"
)
WORKTREES=(012-dl-diffllm 015-dl-internal-pack 016-dl-v4-gitlines 017-dl-d1-learning-curve 018-dl-study-m)

branch_of() { git for-each-ref --format='%(refname:short)' "refs/heads/exp/$1" "refs/remotes/origin/exp/$1" | head -1; }

case "${1:-}" in
  pack)
    out="${2:?out_dir}"; mkdir -p "$out"
    : > "$out/ITEMS.txt"
    for it in "${ITEMS[@]}"; do
      IFS='|' read -r name paths why <<<"$it"
      present=()
      for p in $paths; do [ -e "$p" ] && present+=("$p"); done
      if [ ${#present[@]} -eq 0 ]; then echo "skip $name (not present)"; continue; fi
      echo "pack $name: ${present[*]}"
      COPYFILE_DISABLE=1 tar -cf "$out/$name.tar" --exclude '.DS_Store' --exclude '*/last-*' --exclude '*/last.json' "${present[@]}"
      echo "$name.tar  $why" >> "$out/ITEMS.txt"
    done
    (cd "$out" && shasum -a 256 ./*.tar > SHA256SUMS)
    git rev-parse HEAD > "$out/REPO_HEAD"
    du -sh "$out"
    ;;
  restore)
    in="${2:?in_dir}"
    (cd "$in" && shasum -a 256 -c SHA256SUMS)
    git fetch --all --quiet
    for w in "${WORKTREES[@]}"; do
      [ -d "experiments/$w" ] && { echo "worktree experiments/$w exists"; continue; }
      b=$(branch_of "$w"); [ -n "$b" ] || { echo "no branch for $w" >&2; exit 1; }
      case "$b" in origin/*) git branch "exp/$w" "$b"; b="exp/$w";; esac
      git worktree add "experiments/$w" "$b"
    done
    for t in "$in"/*.tar; do echo "extract $(basename "$t")"; tar -xf "$t"; done
    echo "restored. next: uv sync in each worktree's codebert-diff-lab (HANDOFF.md)"
    ;;
  *) echo "usage: $0 {pack <out_dir>|restore <in_dir>}" >&2; exit 2 ;;
esac
