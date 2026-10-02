"""Orchestration for study diffllm-v1 (docs/diffllm-study-v1.md): prepare -> arms -> freeze -> one test evaluation.

Prepared views carry labels only for public train/valid; test rows are label-free and their labels are
joined only in `test_arms` after this study's own freeze record exists. Arms write run folders under
artifacts/diffllm/<arm>/ with the code SHA, config hash, validation predictions and metrics.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .config import resolve_local_path
from .fulldiff import render_prompt
from .metrics import best_threshold, evaluate
from .policy import require_test_unlocked
from .util import ConfigError, IntegrityError, PolicyError, atomic_write_json, read_json, sha256_file, sha256_json

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARMS = ("R-base", "E-base", "R-diff", "E-diff")


def load_cfg(path: Path) -> tuple[dict, str]:
    raw = yaml.safe_load(path.read_text())
    if raw.get("study_id") != "diffllm-v1":
        raise ConfigError("not a diffllm-v1 study config")
    from .llm_train import CptLoraCfg, MlpCfg, SftCfg
    CptLoraCfg.model_validate(raw["cpt"])
    SftCfg.model_validate(raw["sft"])
    MlpCfg.model_validate(raw["mlp"])
    return raw, sha256_json(raw)


def prepare(cfg: dict, snapshot: Path, split_dir: Path, fulldiff: Path, tokenizer_path: Path, out: Path) -> dict:
    from transformers import AutoTokenizer
    if out.exists():
        raise IntegrityError(f"{out} exists; prepared views are immutable")
    tok = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    rep = cfg["representation"]
    fd = pd.read_parquet(fulldiff / "fulldiff.parquet")
    ch = pd.read_parquet(snapshot / "changes.parquet")[["change_id", "message"]]
    sp = pd.read_parquet(split_dir / "splits.parquet")[["change_id", "split"]]
    lab = pd.read_parquet(snapshot / "labels.parquet")[["change_id", "label"]]
    df = sp.merge(fd, on="change_id", how="left", validate="1:1").merge(ch, on="change_id", how="left", validate="1:1")
    if df["status"].ne("ok").any():
        raise IntegrityError(f"{int(df['status'].ne('ok').sum())} changes lack a full diff")
    rows = []
    for r in df.itertuples(index=False):
        msg, diff, n, cut = render_prompt(tok, r.message, r.diff_text, rep["max_content_tokens"], rep["max_message_tokens"])
        rows.append({"change_id": r.change_id, "split": r.split, "message": msg, "diff": diff, "content_tokens": n, "truncated": cut})
    view = pd.DataFrame(rows)
    out.mkdir(parents=True)
    view.to_parquet(out / "view.parquet", index=False)
    tv = view[view["split"].isin(["train", "valid"])][["change_id"]].merge(lab, on="change_id", how="left", validate="1:1")
    if tv["label"].isna().any():
        raise IntegrityError("unlabelled train/valid rows")
    tv.to_parquet(out / "labels-train-valid.parquet", index=False)
    man = {"artifact_kind": "diffllm_view", "representation": rep, "rows": int(len(view)),
           "by_split": view["split"].value_counts().to_dict(),
           "truncated_share": float(view["truncated"].mean()), "content_tokens_mean": float(view["content_tokens"].mean()),
           "fulldiff_manifest_sha256": sha256_file(fulldiff / "manifest.json"),
           "split_manifest_sha256": sha256_file(split_dir / "manifest.json"),
           "view_sha256": sha256_file(out / "view.parquet"), "labels_sha256": sha256_file(out / "labels-train-valid.parquet"),
           "test_labels": "not included (joined only after freeze)"}
    atomic_write_json(out / "manifest.json", man)
    return man


def _frames(view_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    man = read_json(view_dir / "manifest.json")
    if (sha256_file(view_dir / "view.parquet") != man["view_sha256"]
            or sha256_file(view_dir / "labels-train-valid.parquet") != man["labels_sha256"]):
        raise IntegrityError("prepared view hash mismatch")
    v = pd.read_parquet(view_dir / "view.parquet")
    lab = pd.read_parquet(view_dir / "labels-train-valid.parquet")
    tr = v[v.split == "train"].merge(lab, on="change_id").reset_index(drop=True)
    va = v[v.split == "valid"].merge(lab, on="change_id").reset_index(drop=True)
    te = v[v.split == "test"].reset_index(drop=True)
    if "label" in te.columns:
        raise PolicyError("test view must be label-free before freeze")
    return tr, va, te


def _model_path(cfg: dict, key: str = "qwen7b") -> Path:
    m = cfg["models"][key]
    p = resolve_local_path(m["local_path"], PROJECT_ROOT)
    for name, digest in m.get("files_sha256", m.get("weights_sha256", {})).items():
        if sha256_file(p / name) != digest:
            raise IntegrityError(f"{key} file {name} does not match its pin")
    return p


def _git_sha() -> str:
    from .runner import git_state
    g = git_state()
    if g["sha"] is None:
        raise ConfigError("unknown code version; refusing an untraceable run")
    return g["sha"]


def run_cpt(cfg: dict, cfg_hash: str, data_dir: Path, art: Path) -> dict:
    from .llm_train import CptLoraCfg, cpt_lora
    out = art / "cpt"
    if (out / "cpt.json").exists():
        return read_json(out / "cpt.json")
    corpus = data_dir / "cpt" / cfg["cpt_corpus"]["id"]
    if read_json(corpus / "manifest.json")["corpus_parquet_sha256"] != sha256_file(corpus / "corpus.parquet"):
        raise IntegrityError("CPT corpus hash mismatch")
    res = cpt_lora(_model_path(cfg), corpus / "corpus.parquet", out, CptLoraCfg.model_validate(cfg["cpt"]), cfg["seeds"][0], cfg["device"])
    atomic_write_json(out / "cpt.json", {**read_json(out / "cpt.json"), "config_sha256": cfg_hash, "code_git_sha": _git_sha(),
                                         "corpus_manifest_sha256": sha256_file(corpus / "manifest.json")})
    return res


def run_arm(cfg: dict, cfg_hash: str, arm: str, view_dir: Path, art: Path) -> dict:
    from .llm_train import MlpCfg, SftCfg, embed, encode_risk, label_ids_for, load_base, load_sft, mlp_predict, mlp_train, score, sft_risk
    from .neural import _torch
    torch = _torch()
    if arm not in ARMS:
        raise ConfigError(f"unknown arm {arm}")
    out = art / arm
    if (out / "run.json").exists() and read_json(out / "run.json")["status"] == "completed":
        return read_json(out / "run.json")
    spec = cfg["arms"][arm]
    tr, va, _ = _frames(view_dir)
    base = _model_path(cfg, spec["model"])
    cpt = (art / "cpt" / "adapter") if spec["cpt"] else None
    if cpt is not None and not (cpt / "adapter_config.json").exists():
        raise ConfigError(f"{arm} needs the CPT adapter; run cpt first")
    dev = torch.device(cfg["device"])
    out.mkdir(parents=True, exist_ok=True)
    started = datetime.now(UTC).isoformat()
    seed = cfg["seeds"][0]
    if spec["head"] == "risk_sft":
        info = sft_risk(base, cpt, tr, va, out, SftCfg.model_validate(cfg["sft"]), seed, cfg["device"], cfg["tie_salt"])
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(base), local_files_only=True)
        lids = label_ids_for(tok)
        model = load_sft(base, cpt, out, dev)
        s = score(model, [encode_risk(tok, m, d, lids) for m, d in zip(va["message"], va["diff"], strict=True)], lids,
                  tok.pad_token_id, dev, cfg["sft"]["eval_token_budget"])
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(base), local_files_only=True)
        model = load_base(base, dev, cpt)
        x_tr = embed(model, tok, tr, dev, cfg["embed_token_budget"])
        x_va = embed(model, tok, va, dev, cfg["embed_token_budget"])
        mcfg = MlpCfg.model_validate(cfg["mlp"])
        state, info = mlp_train(x_tr, tr["label"].to_numpy(), x_va, va["label"].to_numpy(), va["change_id"].tolist(), mcfg, seed,
                                cfg["tie_salt"], cfg["device"])
        atomic_write_json(out / "mlp.json", state)
        s = mlp_predict(state, x_va, mcfg)
    y = va["label"].to_numpy().astype(int)
    thr = best_threshold(y, s)
    metrics = {"split": "valid", "evaluation_role": "selection_validation",
               **evaluate(va["change_id"].tolist(), y, s, cfg["tie_salt"], thr["threshold"]), "threshold_selection": thr}
    pd.DataFrame({"change_id": va["change_id"], "score": s}).to_parquet(out / "validation.parquet", index=False)
    atomic_write_json(out / "metrics.json", metrics)
    atomic_write_json(out / "state.json", {k: v for k, v in info.items() if k != "history"} | {"history": info.get("history")})
    files = {p.relative_to(out).as_posix(): sha256_file(p) for p in sorted(out.rglob("*")) if p.is_file() and p.name != "run.json"}
    run = {"study_id": cfg["study_id"], "arm": arm, "seed": seed, "status": "completed", "config_sha256": cfg_hash,
           "code_git_sha": _git_sha(), "started_at": started, "finished_at": datetime.now(UTC).isoformat(),
           "validation_ap": metrics["ap"], "artifacts_sha256": files, "cpt_adapter": str(cpt) if cpt else None}
    atomic_write_json(out / "run.json", run)
    print(f"{arm}: validation AP {metrics['ap']:.4f}", flush=True)
    return run


def _scores(cfg: dict, arm: str, frame: pd.DataFrame, art: Path) -> np.ndarray:
    from transformers import AutoTokenizer

    from .llm_train import MlpCfg, embed, encode_risk, label_ids_for, load_base, load_sft, mlp_predict, score
    from .neural import _torch
    torch = _torch()
    spec = cfg["arms"][arm]
    base = _model_path(cfg, spec["model"])
    cpt = (art / "cpt" / "adapter") if spec["cpt"] else None
    dev = torch.device(cfg["device"])
    tok = AutoTokenizer.from_pretrained(str(base), local_files_only=True)
    if spec["head"] == "risk_sft":
        lids = label_ids_for(tok)
        m = load_sft(base, cpt, art / arm, dev)
        return score(m, [encode_risk(tok, a, b, lids) for a, b in zip(frame["message"], frame["diff"], strict=True)], lids,
                     tok.pad_token_id, dev, cfg["sft"]["eval_token_budget"])
    m = load_base(base, dev, cpt)
    x = embed(m, tok, frame, dev, cfg["embed_token_budget"])
    return mlp_predict(read_json(art / arm / "mlp.json"), x, MlpCfg.model_validate(cfg["mlp"]))


def freeze(cfg: dict, cfg_hash: str, view_dir: Path, art: Path, out: Path, check_rows: int = 64) -> dict:
    if out.exists():
        raise PolicyError(f"{out} exists; a study freeze is immutable")
    _, va, _ = _frames(view_dir)
    entries = []
    for arm in ARMS:
        run = read_json(art / arm / "run.json")
        if run["status"] != "completed" or run["config_sha256"] != cfg_hash:
            raise IntegrityError(f"{arm} is not a completed run of this config")
        for rel, digest in run["artifacts_sha256"].items():
            if sha256_file(art / arm / rel) != digest:
                raise IntegrityError(f"{arm}: {rel} changed")
        # Embeddings are batched by length over the whole frame; a subsample changes bf16 batch composition, which the
        # standardize+MLP head amplifies (0.109 on 64 rows, Spearman 0.997), so embedding arms re-embed all of validation.
        if cfg["arms"][arm]["head"] == "risk_sft":
            rows = np.sort(np.random.default_rng([cfg["seeds"][0], 919]).choice(len(va), size=check_rows, replace=False))
        else:
            rows = np.arange(len(va))
        again = _scores(cfg, arm, va.iloc[rows].reset_index(drop=True), art)
        ref = pd.read_parquet(art / arm / "validation.parquet")["score"].to_numpy()[rows]
        from scipy.stats import spearmanr
        diff, rho = float(np.abs(again - ref).max()), float(spearmanr(again, ref).correlation)
        if diff > 0.05 or rho < 0.99:
            raise IntegrityError(f"{arm}: frozen artifacts reproduce validation only to {diff:.3f} (Spearman {rho:.3f})")
        entries.append({"arm": arm, "run_json_sha256": sha256_file(art / arm / "run.json"), "artifacts_sha256": run["artifacts_sha256"],
                        "threshold": read_json(art / arm / "metrics.json")["threshold_selection"]["threshold"],
                        "validation_ap": run["validation_ap"], "validation_reproduction_max_abs_diff": diff, "spearman": rho})
        print(f"freeze: {arm} reproduced validation (max |diff| {diff:.1e}, Spearman {rho:.4f})", flush=True)
    rec = {"status": "frozen", "study_id": cfg["study_id"], "config_sha256": cfg_hash, "frozen_at": datetime.now(UTC).isoformat(),
           "code_git_sha": _git_sha(), "cpt_json_sha256": sha256_file(art / "cpt" / "cpt.json"), "arms": entries}
    rec["freeze_id"] = sha256_json({"config": cfg_hash, "arms": entries})[:16]
    atomic_write_json(out, rec)
    return rec


def test_arms_once(cfg: dict, cfg_hash: str, view_dir: Path, snapshot: Path, art: Path, freeze_path: Path, out_dir: Path) -> dict:
    rec = require_test_unlocked("test", freeze_path)
    if rec["config_sha256"] != cfg_hash:
        raise IntegrityError("config changed after freeze")
    _, _, te = _frames(view_dir)
    lab = pd.read_parquet(snapshot / "labels.parquet")[["change_id", "label"]]
    y = te[["change_id"]].merge(lab, on="change_id", how="left", validate="1:1")["label"].to_numpy().astype(int)
    done = {}
    for e in rec["arms"]:
        d = out_dir / e["arm"]
        if (d / "metrics.json").exists():
            raise PolicyError(f"public test already evaluated for {e['arm']}; never overwritten")
        for rel, digest in e["artifacts_sha256"].items():
            if sha256_file(art / e["arm"] / rel) != digest:
                raise IntegrityError(f"{e['arm']}: {rel} changed after freeze")
        s = _scores(cfg, e["arm"], te, art)
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"change_id": te["change_id"], "project": te["change_id"].str.split(":").str[1], "score": s}).to_parquet(
            d / "predictions.parquet", index=False)
        m = {"split": "test", "evaluation_role": "frozen_final_test", "study_id": cfg["study_id"], "freeze_id": rec["freeze_id"],
             "arm": e["arm"], **evaluate(te["change_id"].tolist(), y, s, cfg["tie_salt"], e["threshold"]),
             "evaluated_at": datetime.now(UTC).isoformat()}
        atomic_write_json(d / "metrics.json", m)
        done[e["arm"]] = m["ap"]
        print(f"test: {e['arm']} AP {m['ap']:.4f}", flush=True)
    atomic_write_json(out_dir / "evaluation.json", {"freeze_id": rec["freeze_id"], "test_ap": done})
    return done


def summarize(art: Path, test_dir: Path) -> str:
    rows = []
    for arm in ARMS:
        v = read_json(art / arm / "metrics.json")
        t = read_json(test_dir / arm / "metrics.json") if (test_dir / arm / "metrics.json").exists() else {}
        rows.append({"arm": arm, "valid_ap": v["ap"], "test_ap": t.get("ap"),
                     "test_recall_at_5pct": (t.get("recall_at_5pct") or {}).get("recall"),
                     "test_recall_at_10pct": (t.get("recall_at_10pct") or {}).get("recall")})
    return json.dumps(rows, indent=1)
