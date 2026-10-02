#!/bin/bash
# Download the pinned Qwen2.5-Coder-7B-Instruct revision to the container disk (the /workspace quota is 20 GB) and check the pins.
export PATH=$HOME/.local/bin:$PATH
cd /workspace/jit016/codebert-diff-lab
mkdir -p /root/qwen && ln -sfn /root/qwen /workspace/jit016/experiments/.cache/models/qwen2.5-coder-7b-instruct
uv run --no-sync python - <<'PY'
from huggingface_hub import snapshot_download
import yaml, hashlib, pathlib
cfg = yaml.safe_load(open("configs/studies/public-comparison-v4-gitlines.yaml"))["llm"] if False else yaml.safe_load(open("/workspace/v4.yaml"))["llm"]
d = pathlib.Path("/root/qwen")
snapshot_download(cfg["model_id"], revision=cfg["revision"], local_dir=str(d), allow_patterns=list(cfg["files_sha256"]))
bad = []
for name, digest in cfg["files_sha256"].items():
    h = hashlib.sha256()
    with open(d / name, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    if h.hexdigest() != digest:
        bad.append(name)
print("QWEN_PINS_OK" if not bad else f"QWEN_PINS_BAD {bad}")
PY
