"""Diagnose collapsed (constant) scores: compare CodeBERT pooled embeddings for the same
inputs on cpu fp32, cuda fp32 and cuda bf16-autocast, and the spread across samples."""
import json

import torch
from transformers import AutoModel

from riskbench.neural import Collator, RiskNet, Rows

rows = [json.loads(l) for _, l in zip(range(16), open("data/views/public/valid/inputs.jsonl"))]
pad = json.load(open("data/views/public/valid/manifest.json"))["view"]["pad_token_id"]
batch = Collator(pad)([{"ids": r["input_ids"], "label": None} for r in rows])
print("seq len", batch["input_ids"].shape, "pad", pad, "attn attn_implementation?")

out = {}
for name, dev, amp in (("cpu32", "cpu", False), ("cuda32", "cuda", False), ("cuda_bf16", "cuda", True)):
    enc = AutoModel.from_pretrained("models/codebert-base", local_files_only=True)
    print(name, "attn_impl", enc.config._attn_implementation)
    net = RiskNet(enc, frozen=True).to(dev).eval()
    ids, mask = batch["input_ids"].to(dev), batch["attention_mask"].to(dev)
    with torch.inference_mode(), torch.autocast(device_type=dev, dtype=torch.bfloat16, enabled=amp):
        e = net.embeddings(ids, mask).float().cpu()
    out[name] = e
    print(f"{name}: finite={torch.isfinite(e).all().item()} mean|e|={e.abs().mean():.4f} "
          f"across-sample std={e.std(0).mean():.6f}")
for a, b in (("cpu32", "cuda32"), ("cpu32", "cuda_bf16")):
    d = (out[a] - out[b]).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(out[a], out[b]).min().item()
    print(f"{a} vs {b}: max|diff|={d:.5f} min cosine={cos:.5f}")
