from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .common import BenchError


def parser():
    p = argparse.ArgumentParser(prog="riskbench", description="Public-only change-risk baselines; internal labels are evaluation-only")
    subs = p.add_subparsers(dest="command", required=True)
    s = subs.add_parser("fetch-apache", help="Download pinned ApacheJIT v2 CSV (or use a local archive)")
    s.add_argument("--out", required=True); s.add_argument("--archive")
    s = subs.add_parser("build-apache", help="CSV labels + local Git patches; optional explicit public cloning")
    for key in ("csv", "repos", "out"): s.add_argument("--"+key, required=True)
    s.add_argument("--allow-network", action="store_true"); s.add_argument("--limit", type=int)
    s.add_argument("--seed", type=int, default=42); s.add_argument("--first-parent", action="store_true")
    s.add_argument("--commit-col", default="commit_id"); s.add_argument("--repo-col", default="project"); s.add_argument("--label-col", default="buggy")
    s = subs.add_parser("build-internal", help="Import a local manifest and resolved labels; no network")
    s.add_argument("--manifest", required=True); s.add_argument("--out", required=True)
    s = subs.add_parser("split-public")
    for key in ("records", "out", "train-before", "valid-before"): s.add_argument("--"+key, required=True)
    s.add_argument("--allow-retrospective", action="store_true"); s.add_argument("--holdout-repos", nargs="*", default=[])
    s = subs.add_parser("internal-test")
    s.add_argument("--records", required=True); s.add_argument("--out", required=True)
    s = subs.add_parser("prepare", help="Create identical bounded code views for A/B/C/D")
    for key in ("dataset", "out", "tokenizer"): s.add_argument("--"+key, required=True)
    s.add_argument("--max-tokens", type=int, default=512); s.add_argument("--message-tokens", type=int, default=64)
    s.add_argument("--revision"); s.add_argument("--allow-download", action="store_true")
    s = subs.add_parser("download-codebert")
    s.add_argument("--out", required=True); s.add_argument("--model", default="microsoft/codebert-base"); s.add_argument("--revision", default="main")
    s = subs.add_parser("train-tabular")
    for key in ("public-train", "public-valid", "out"): s.add_argument("--"+key, required=True)
    s.add_argument("--cs", type=float, nargs="+", default=[0.1,1.0,10.0]); s.add_argument("--seed", type=int, default=42)
    s = subs.add_parser("train-codebert")
    for key in ("public-train", "public-valid", "out", "model"): s.add_argument("--"+key, required=True)
    s.add_argument("--mode", choices=["frozen","finetune"], required=True)
    s.add_argument("--epochs", type=int, default=5); s.add_argument("--batch-size", type=int, default=8)
    s.add_argument("--accumulation", type=int, default=4); s.add_argument("--encoder-lr", type=float, default=2e-5)
    s.add_argument("--head-lr", type=float, default=1e-3); s.add_argument("--patience", type=int, default=2)
    s.add_argument("--seed", type=int, default=42); s.add_argument("--device", choices=["auto","cpu","cuda","mps"], default="auto")
    s.add_argument("--revision"); s.add_argument("--allow-download", action="store_true"); s.add_argument("--allow-cpu-training", action="store_true")
    s.add_argument("--class-weight", choices=["none","balanced"], default="none")
    s = subs.add_parser("examples")
    s.add_argument("--public-train", required=True); s.add_argument("--out", required=True)
    s.add_argument("--k", type=int, default=4); s.add_argument("--seed", type=int, default=42)
    s = subs.add_parser("freeze")
    s.add_argument("--paths", nargs="+", required=True); s.add_argument("--out", required=True)
    s = subs.add_parser("predict")
    s.add_argument("--kind", choices=["rule","tabular","codebert"], required=True)
    s.add_argument("--artifact"); s.add_argument("--dataset", required=True); s.add_argument("--out", required=True)
    s.add_argument("--name", required=True); s.add_argument("--lock")
    s.add_argument("--batch-size", type=int, default=8); s.add_argument("--device", default="auto")
    s = subs.add_parser("llm")
    for key in ("config", "dataset", "out", "name"): s.add_argument("--"+key, required=True)
    s.add_argument("--examples"); s.add_argument("--lock"); s.add_argument("--resume", action="store_true")
    s = subs.add_parser("evaluate")
    s.add_argument("--dataset", required=True); s.add_argument("--predictions", nargs="+", required=True); s.add_argument("--out", required=True)
    s.add_argument("--allow-partial", action="store_true"); s.add_argument("--allow-label-transfer", action="store_true")
    s.add_argument("--bootstrap", type=int, default=0); s.add_argument("--seed", type=int, default=42)
    s.add_argument("--cluster", choices=["week","group"], default="week")
    s = subs.add_parser("demo-data")
    s.add_argument("--out", required=True)
    return p


def main(argv=None):
    a = parser().parse_args(argv)
    try:
        result = dispatch(a)
        if result is not None:
            print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    except (BenchError, FileNotFoundError, ImportError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    return 0


def dispatch(a):
    c = a.command
    if c == "fetch-apache":
        from .data import download_apache
        return download_apache(a.out, a.archive)
    if c == "build-apache":
        from .data import apache_build
        return apache_build(a.csv, a.repos, a.out, a.allow_network, a.limit, a.seed, a.first_parent, a.commit_col, a.repo_col, a.label_col)
    if c == "build-internal":
        from .data import internal_build
        return internal_build(a.manifest, a.out)
    if c == "split-public":
        from .data import split_public
        return split_public(a.records, a.out, a.train_before, a.valid_before, a.allow_retrospective, a.holdout_repos)
    if c == "internal-test":
        from .data import internal_partition
        internal_partition(a.records, a.out); return {"out": a.out}
    if c == "prepare":
        from .views import prepare_view
        return prepare_view(a.dataset, a.out, a.tokenizer, a.max_tokens, a.message_tokens, a.revision, a.allow_download)
    if c == "download-codebert":
        from huggingface_hub import HfApi, snapshot_download
        from .common import new_dir, write_json
        root = new_dir(a.out)
        resolved = HfApi().model_info(a.model, revision=a.revision).sha
        snapshot_download(a.model, revision=resolved, local_dir=str(root), allow_patterns=["config.json", "tokenizer*", "vocab.json", "merges.txt", "special_tokens_map.json", "pytorch_model.bin", "model.safetensors"])
        info = {"model": a.model, "requested_revision": a.revision, "resolved_revision": resolved}
        write_json(root/"SOURCE.json", info)
        return info
    if c == "train-tabular":
        from .linear import train_tabular
        return train_tabular(a.public_train, a.public_valid, a.out, a.cs, a.seed)
    if c == "train-codebert":
        from .neural import train_neural
        return train_neural(a.public_train, a.public_valid, a.out, a.model, a.mode, a.epochs, a.batch_size,
                            a.accumulation, a.encoder_lr, a.head_lr, a.patience, a.seed, a.device, a.revision,
                            a.allow_download, a.allow_cpu_training, a.class_weight)
    if c == "examples":
        from .llm import select_examples
        return select_examples(a.public_train, a.out, a.k, a.seed)
    if c == "freeze":
        from .artifacts import freeze
        report = freeze(a.paths, a.out)
        return {"out": a.out, "frozen_files": len(report["files"])}
    if c == "predict":
        if a.kind != "rule" and not a.artifact:
            raise BenchError("--artifact required for learned models")
        if a.kind == "rule":
            from .linear import predict_rule
            predict_rule(a.dataset, a.out, a.name)
        elif a.kind == "tabular":
            from .linear import predict_tabular
            predict_tabular(a.artifact, a.dataset, a.out, a.name, a.lock)
        else:
            from .neural import predict_neural
            predict_neural(a.artifact, a.dataset, a.out, a.name, a.batch_size, a.device, a.lock)
        return {"out": a.out}
    if c == "llm":
        from .llm import predict_llm
        return predict_llm(a.config, a.dataset, a.out, a.name, a.examples, a.lock, a.resume)
    if c == "evaluate":
        from .metrics import evaluate
        report = evaluate(a.dataset, a.predictions, a.out, a.allow_partial, a.allow_label_transfer, a.bootstrap, a.seed, a.cluster)
        return {"out": a.out, "coverage": report["coverage"], "metrics": report["metrics"], "synthetic": report["synthetic"]}
    if c == "demo-data":
        from .demo import make_demo
        return make_demo(a.out)
    raise BenchError("Unknown command")


if __name__ == "__main__":
    raise SystemExit(main())
