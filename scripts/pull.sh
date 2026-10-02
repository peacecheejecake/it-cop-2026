#!/bin/bash
# Pull pod results. Default: small files only; "all" also pulls best checkpoints (never resume state).
set -euo pipefail
cd "$(dirname "$0")/../codebert-diff-lab"
INFO=$(runpodctl ssh info 2spnrcaqwyaa55)
IP=$(echo "$INFO" | python3 -c "import json,sys;print(json.load(sys.stdin)['ip'])"); PORT=$(echo "$INFO" | python3 -c "import json,sys;print(json.load(sys.stdin)['port'])")
LIMIT=(--max-size=20M); [ "${1:-}" = all ] && LIMIT=()
rsync -aL "${LIMIT[@]}" --exclude 'last-*' --exclude 'last.json' --exclude '*.tmp' -e "ssh -i $HOME/.runpod/ssh/runpodctl-ssh-key -p $PORT -o ConnectTimeout=15" \
  root@$IP:/workspace/jit016/codebert-diff-lab/artifacts/ artifacts/
mkdir -p ../logs/pod && rsync -a -e "ssh -i $HOME/.runpod/ssh/runpodctl-ssh-key -p $PORT -o ConnectTimeout=15" root@$IP:/workspace/jit016/codebert-diff-lab/logs/ ../logs/pod/
