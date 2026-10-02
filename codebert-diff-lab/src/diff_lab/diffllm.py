"""Orchestration for studies diffllm-v1/v2 (docs/diffllm-study-v1.md, docs/diffllm-study-v2.md): prepare -> arms -> freeze ->
one test evaluation.

v1 views hold message/diff (full-diff-v1). v2 views (`prepare_evidence`) hold pre-rendered prompts per representation in
`system:<rep>`/`user:<rep>` columns over the v3 EvidenceView; an arm with representation `stage1-best` takes whichever of the
registered stage-1 arms has the higher full-validation AP (ties: the first listed), recorded in its run.json.

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
STUDIES = ("diffllm-v1", "diffllm-v2")
STUDY_PREFIXES = ("study-m",)
STAGE1_BEST = "stage1-best"
V1_ARMS = ("R-base", "E-base", "R-diff", "E-diff")


def load_cfg(path: Path) -> tuple[dict, str]:
    raw = yaml.safe_load(path.read_text())
    sid = raw.get("study_id") or ""
    if sid not in STUDIES and not sid.startswith(STUDY_PREFIXES):
        raise ConfigError(f"not a diffllm study config ({STUDIES}, {STUDY_PREFIXES})")
    from .llm_train import CptLoraCfg, MlpCfg, SftCfg
    SftCfg.model_validate(raw["sft"])
    if any(s["cpt"] for s in raw["arms"].values()):
        CptLoraCfg.model_validate(raw["cpt"])
    if any(s["head"] == "mlp" for s in raw["arms"].values()):
        MlpCfg.model_validate(raw["mlp"])
    reps = set(raw["representation"].get("prompts") or {})
    for arm, spec in raw["arms"].items():
        r = spec.get("representation")
        if r is not None and r != STAGE1_BEST and r not in reps:
            raise ConfigError(f"{arm}: unknown representation {r}")
        if r is not None and spec["head"] != "risk_sft":
            raise ConfigError(f"{arm}: prompt representations are only implemented for risk_sft")
        if r == STAGE1_BEST and not raw.get("stage1_arms"):
            raise ConfigError(f"{arm}: {STAGE1_BEST} needs stage1_arms")
    for arm, spec in raw["arms"].items():
        if isinstance(spec["cpt"], str) and spec["cpt"] not in (raw.get("cpts") or {}):
            raise ConfigError(f"{arm}: unknown cpt {spec['cpt']}")
        if isinstance(spec["cpt"], str) and raw["cpts"][spec["cpt"]]["model"] != spec["model"]:
            raise ConfigError(f"{arm}: cpt {spec['cpt']} was trained for another model")
    _arms(raw)
    return raw, sha256_json(raw)


def _seeds(cfg: dict) -> list[int]:
    """Study M trains every arm per seed (`per_seed: true`); diffllm-v1/v2 used seeds[0] only."""
    return list(cfg["seeds"]) if cfg.get("per_seed") else [cfg["seeds"][0]]


def _run_dir(cfg: dict, art: Path, arm: str, seed: int) -> Path:
    return art / arm / f"seed-{seed}" if cfg.get("per_seed") else art / arm


def _cpt_dir(cfg: dict, art: Path, arm: str) -> Path | None:
    """`cpt: true` is the single v1 CPT (art/cpt); `cpt: <name>` is a per-model CPT registered under `cpts` (art/cpt/<name>)."""
    c = cfg["arms"][arm]["cpt"]
    if not c:
        return None
    return (art / "cpt" / c if isinstance(c, str) else art / "cpt") / "adapter"


def _pairs(cfg: dict) -> list[tuple[str, int]]:
    return [(arm, seed) for arm in _arms(cfg) for seed in _seeds(cfg)]


def _arms(cfg: dict) -> list[str]:
    """Run/freeze order: `arm_order` from the config (v2+); diffllm-v1 predates it."""
    order = cfg.get("arm_order", V1_ARMS)
    if sorted(order) != sorted(cfg["arms"]):
        raise ConfigError("arm_order must list every arm exactly once")
    return list(order)


def prepare_evidence(cfg: dict, snapshot: Path, split_dir: Path, evidence_dir: Path, tokenizer_path: Path, out: Path) -> dict:
    """v2 view: the v3 EvidenceView text (already budgeted; never truncated here) rendered by each registered prompt."""
    from transformers import AutoTokenizer

    from .llm import serialize_features
    if out.exists():
        raise IntegrityError(f"{out} exists; prepared views are immutable")
    rep = cfg["representation"]
    man_ev = read_json(evidence_dir / "manifest.json")
    if sha256_file(evidence_dir / "evidence.parquet") != man_ev["evidence_parquet_sha256"]:
        raise IntegrityError("evidence parquet hash mismatch")
    if man_ev["evidence_parquet_sha256"] != rep["evidence_parquet_sha256"]:
        raise IntegrityError("evidence view is not the registered one")
    tok = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    ev = pd.read_parquet(evidence_dir / "evidence.parquet")[["change_id", "query_text"]]
    feat = pd.read_parquet(snapshot / "features.parquet")
    sp = pd.read_parquet(split_dir / "splits.parquet")[["change_id", "split"]]
    lab = pd.read_parquet(snapshot / "labels.parquet")[["change_id", "label"]]
    df = sp.merge(ev, on="change_id", how="left", validate="1:1").merge(feat, on="change_id", how="left", validate="1:1")
    if df["query_text"].isna().any():
        raise IntegrityError(f"{int(df['query_text'].isna().sum())} changes lack an EvidenceView")
    view = df[["change_id", "split"]].copy()
    lengths = {}
    for name, p in rep["prompts"].items():
        users = [p["user"].format(evidence=r.query_text, features=serialize_features(pd.Series(r._asdict())) if "{features}" in p["user"]
                                  else "") for r in df.itertuples(index=False)]
        view[f"system:{name}"] = p["system"]
        view[f"user:{name}"] = users
        n = np.array([len(tok(tok.apply_chat_template([{"role": "system", "content": p["system"]}, {"role": "user", "content": u}],
                                                      tokenize=False, add_generation_prompt=True), add_special_tokens=False)["input_ids"])
                      for u in users])
        if n.max() > cfg["sft"]["max_prompt_tokens"]:
            raise IntegrityError(f"{name}: {int((n > cfg['sft']['max_prompt_tokens']).sum())} prompts exceed max_prompt_tokens")
        lengths[name] = {"mean": float(n.mean()), "p99": float(np.percentile(n, 99)), "max": int(n.max())}
    out.mkdir(parents=True)
    view.to_parquet(out / "view.parquet", index=False)
    tv = view[view["split"].isin(["train", "valid"])][["change_id"]].merge(lab, on="change_id", how="left", validate="1:1")
    if tv["label"].isna().any():
        raise IntegrityError("unlabelled train/valid rows")
    tv.to_parquet(out / "labels-train-valid.parquet", index=False)
    man = {"artifact_kind": "diffllm_view", "representation": rep, "rows": int(len(view)),
           "by_split": view["split"].value_counts().to_dict(), "prompt_tokens": lengths,
           "evidence_manifest_sha256": sha256_file(evidence_dir / "manifest.json"),
           "split_manifest_sha256": sha256_file(split_dir / "manifest.json"),
           "view_sha256": sha256_file(out / "view.parquet"), "labels_sha256": sha256_file(out / "labels-train-valid.parquet"),
           "test_labels": "not included (joined only after freeze)"}
    atomic_write_json(out / "manifest.json", man)
    return man


def _rep(cfg: dict, arm: str, art: Path) -> str | None:
    r = cfg["arms"][arm].get("representation")
    if r != STAGE1_BEST:
        return r
    run = art / arm / "run.json"
    if run.exists():
        return read_json(run)["representation"]
    aps = []
    for a in cfg["stage1_arms"]:
        rr = read_json(art / a / "run.json")
        if rr["status"] != "completed":
            raise ConfigError(f"{arm} needs completed stage-1 arm {a}")
        aps.append((read_json(art / a / "metrics.json")["ap"], a))
    best = max(aps, key=lambda t: t[0])
    winner = next(a for ap, a in aps if ap == best[0])
    return cfg["arms"][winner]["representation"]


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


def run_cpt(cfg: dict, cfg_hash: str, data_dir: Path, art: Path, name: str | None = None) -> dict:
    from .llm_train import CptLoraCfg, cpt_lora
    out = art / "cpt" / name if name else art / "cpt"
    if (out / "cpt.json").exists():
        return read_json(out / "cpt.json")
    spec = cfg["cpts"][name] if name else {"model": "qwen7b", "corpus": cfg["cpt_corpus"]["id"]}
    corpus = data_dir / "cpt" / spec["corpus"]
    if read_json(corpus / "manifest.json")["corpus_parquet_sha256"] != sha256_file(corpus / "corpus.parquet"):
        raise IntegrityError("CPT corpus hash mismatch")
    res = cpt_lora(_model_path(cfg, spec["model"]), corpus / "corpus.parquet", out, CptLoraCfg.model_validate(cfg["cpt"]),
                   cfg["seeds"][0], cfg["device"])
    atomic_write_json(out / "cpt.json", {**read_json(out / "cpt.json"), "config_sha256": cfg_hash, "code_git_sha": _git_sha(),
                                         "corpus_manifest_sha256": sha256_file(corpus / "manifest.json")})
    return res


def run_arm(cfg: dict, cfg_hash: str, arm: str, view_dir: Path, art: Path, seed: int | None = None) -> dict:
    from .llm_train import MlpCfg, SftCfg, embed, encode_frame, label_ids_for, load_base, load_sft, mlp_predict, mlp_train, score, sft_risk
    from .neural import _torch
    torch = _torch()
    if arm not in cfg["arms"]:
        raise ConfigError(f"unknown arm {arm}")
    seed = cfg["seeds"][0] if seed is None else seed
    if seed not in _seeds(cfg):
        raise ConfigError(f"seed {seed} is not registered for {arm}")
    out = _run_dir(cfg, art, arm, seed)
    if (out / "run.json").exists() and read_json(out / "run.json")["status"] == "completed":
        return read_json(out / "run.json")
    spec = cfg["arms"][arm]
    tr, va, _ = _frames(view_dir)
    base = _model_path(cfg, spec["model"])
    cpt = _cpt_dir(cfg, art, arm)
    if cpt is not None and not (cpt / "adapter_config.json").exists():
        raise ConfigError(f"{arm} needs the CPT adapter; run cpt first")
    dev = torch.device(cfg["device"])
    out.mkdir(parents=True, exist_ok=True)
    started = datetime.now(UTC).isoformat()
    rep = _rep(cfg, arm, art)
    if spec["head"] == "risk_sft":
        info = sft_risk(base, cpt, tr, va, out, SftCfg.model_validate(cfg["sft"]), seed, cfg["device"], cfg["tie_salt"], rep)
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(base), local_files_only=True)
        lids = label_ids_for(tok)
        model = load_sft(base, cpt, out, dev)
        s = score(model, encode_frame(tok, va, lids, rep), lids, tok.pad_token_id, dev, cfg["sft"]["eval_token_budget"])
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
           "validation_ap": metrics["ap"], "artifacts_sha256": files, "cpt_adapter": str(cpt) if cpt else None,
           "model": cfg["models"][spec["model"]]["id"], "representation": rep}
    atomic_write_json(out / "run.json", run)
    print(f"{arm} seed {seed}: validation AP {metrics['ap']:.4f}", flush=True)
    return run


def _scores(cfg: dict, arm: str, frame: pd.DataFrame, art: Path, seed: int | None = None) -> np.ndarray:
    from transformers import AutoTokenizer

    from .llm_train import MlpCfg, embed, encode_frame, label_ids_for, load_base, load_sft, mlp_predict, score
    from .neural import _torch
    torch = _torch()
    spec = cfg["arms"][arm]
    base = _model_path(cfg, spec["model"])
    cpt = _cpt_dir(cfg, art, arm)
    run_dir = _run_dir(cfg, art, arm, cfg["seeds"][0] if seed is None else seed)
    dev = torch.device(cfg["device"])
    tok = AutoTokenizer.from_pretrained(str(base), local_files_only=True)
    if spec["head"] == "risk_sft":
        lids = label_ids_for(tok)
        m = load_sft(base, cpt, run_dir, dev)
        return score(m, encode_frame(tok, frame, lids, _rep(cfg, arm, art)), lids, tok.pad_token_id, dev, cfg["sft"]["eval_token_budget"])
    m = load_base(base, dev, cpt)
    x = embed(m, tok, frame, dev, cfg["embed_token_budget"])
    return mlp_predict(read_json(run_dir / "mlp.json"), x, MlpCfg.model_validate(cfg["mlp"]))


def freeze(cfg: dict, cfg_hash: str, view_dir: Path, art: Path, out: Path, check_rows: int = 64) -> dict:
    if out.exists():
        raise PolicyError(f"{out} exists; a study freeze is immutable")
    _, va, _ = _frames(view_dir)
    entries = []
    for arm, seed in _pairs(cfg):
        d = _run_dir(cfg, art, arm, seed)
        run = read_json(d / "run.json")
        if run["status"] != "completed" or run["config_sha256"] != cfg_hash:
            raise IntegrityError(f"{arm} seed {seed} is not a completed run of this config")
        for rel, digest in run["artifacts_sha256"].items():
            if sha256_file(d / rel) != digest:
                raise IntegrityError(f"{arm} seed {seed}: {rel} changed")
        # Embeddings are batched by length over the whole frame; a subsample changes bf16 batch composition, which the
        # standardize+MLP head amplifies (0.109 on 64 rows, Spearman 0.997), so embedding arms re-embed all of validation.
        if cfg["arms"][arm]["head"] == "risk_sft":
            rows = np.sort(np.random.default_rng([seed, 919]).choice(len(va), size=check_rows, replace=False))
        else:
            rows = np.arange(len(va))
        again = _scores(cfg, arm, va.iloc[rows].reset_index(drop=True), art, seed)
        ref = pd.read_parquet(d / "validation.parquet")["score"].to_numpy()[rows]
        from scipy.stats import spearmanr
        diff, rho = float(np.abs(again - ref).max()), float(spearmanr(again, ref).correlation)
        if diff > 0.05 or rho < 0.99:
            raise IntegrityError(f"{arm} seed {seed}: frozen artifacts reproduce validation only to {diff:.3f} (Spearman {rho:.3f})")
        entries.append({"arm": arm, "seed": seed, "run_json_sha256": sha256_file(d / "run.json"),
                        "artifacts_sha256": run["artifacts_sha256"],
                        "threshold": read_json(d / "metrics.json")["threshold_selection"]["threshold"],
                        "validation_ap": run["validation_ap"], "validation_reproduction_max_abs_diff": diff, "spearman": rho})
        print(f"freeze: {arm} seed {seed} reproduced validation (max |diff| {diff:.1e}, Spearman {rho:.4f})", flush=True)
    rec = {"status": "frozen", "study_id": cfg["study_id"], "config_sha256": cfg_hash, "frozen_at": datetime.now(UTC).isoformat(),
           "code_git_sha": _git_sha(), "arms": entries,
           "cpt_json_sha256": {str(p.parent.relative_to(art)): sha256_file(p.parent / "cpt.json")
                               for p in sorted({_cpt_dir(cfg, art, a) for a in cfg["arms"] if cfg["arms"][a]["cpt"]})} or None}
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
        seed = e.get("seed", cfg["seeds"][0])
        d = _run_dir(cfg, out_dir, e["arm"], seed)
        if (d / "metrics.json").exists():
            raise PolicyError(f"public test already evaluated for {e['arm']} seed {seed}; never overwritten")
        for rel, digest in e["artifacts_sha256"].items():
            if sha256_file(_run_dir(cfg, art, e["arm"], seed) / rel) != digest:
                raise IntegrityError(f"{e['arm']} seed {seed}: {rel} changed after freeze")
        s = _scores(cfg, e["arm"], te, art, seed)
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"change_id": te["change_id"], "project": te["change_id"].str.split(":").str[1], "score": s}).to_parquet(
            d / "predictions.parquet", index=False)
        m = {"split": "test", "evaluation_role": "frozen_final_test", "study_id": cfg["study_id"], "freeze_id": rec["freeze_id"],
             "arm": e["arm"], "seed": seed, **evaluate(te["change_id"].tolist(), y, s, cfg["tie_salt"], e["threshold"]),
             "evaluated_at": datetime.now(UTC).isoformat()}
        atomic_write_json(d / "metrics.json", m)
        done[f"{e['arm']}:{seed}" if cfg.get("per_seed") else e["arm"]] = m["ap"]
        print(f"test: {e['arm']} seed {seed} AP {m['ap']:.4f}", flush=True)
    atomic_write_json(out_dir / "evaluation.json", {"freeze_id": rec["freeze_id"], "test_ap": done})
    return done


def summarize(cfg: dict, art: Path, test_dir: Path) -> str:
    rows = []
    for arm, seed in _pairs(cfg):
        v = read_json(_run_dir(cfg, art, arm, seed) / "metrics.json")
        tp = _run_dir(cfg, test_dir, arm, seed) / "metrics.json"
        t = read_json(tp) if tp.exists() else {}
        rows.append({"arm": arm, "seed": seed, "valid_ap": v["ap"], "test_ap": t.get("ap"),
                     "test_recall_at_5pct": (t.get("recall_at_5pct") or {}).get("recall"),
                     "test_recall_at_10pct": (t.get("recall_at_10pct") or {}).get("recall")})
    return json.dumps(rows, indent=1)


def probe_cpt_loss(cfg: dict, view_dir: Path, art: Path, cpt: str, rows: int, out: Path) -> dict:
    """Mean next-token loss on public-train full diffs in the CPT text format, for the base model and base + CPT adapter.

    Only train rows are used; the sample is a salted sha256 rank of change_id (label-free). Labels are read afterwards to
    split the loss by class, which is allowed for train. A CPT that barely moves this loss changed little about how the
    model reads these diffs, so scaling its corpus is the less likely fix.
    """
    import hashlib

    from transformers import AutoTokenizer

    from .llm_train import load_base
    from .neural import _torch
    torch = _torch()
    if out.exists():
        raise PolicyError(f"{out} exists")
    spec = cfg["cpts"][cpt] if cpt in (cfg.get("cpts") or {}) else {"model": "qwen7b"}
    adapter = (art / "cpt" / cpt if cpt in (cfg.get("cpts") or {}) else art / "cpt") / "adapter"
    if not (adapter / "adapter_config.json").exists():
        raise ConfigError(f"no CPT adapter at {adapter}")
    tr, _, _ = _frames(view_dir)
    rank = tr["change_id"].map(lambda c: hashlib.sha256(f"probe-cpt-v1:{c}".encode()).hexdigest())
    sample = tr.assign(_r=rank).sort_values("_r").head(rows).reset_index(drop=True)
    base = _model_path(cfg, spec["model"])
    tok = AutoTokenizer.from_pretrained(str(base), local_files_only=True)
    texts = [f"Commit message:\n{m}\n\nDiff:\n{d}" for m, d in zip(sample["message"], sample["diff"], strict=True)]
    ids = [tok(t, add_special_tokens=False)["input_ids"][: cfg["cpt"]["seq_len"]] for t in texts]
    dev = torch.device(cfg["device"])
    res = {}
    for name, adp in (("base", None), ("cpt", adapter)):
        model = load_base(base, dev, adp).eval()
        losses = []
        with torch.no_grad():
            for x in ids:
                t = torch.tensor([x], device=dev)
                losses.append(float(model(input_ids=t, labels=t).loss))
        res[name] = np.asarray(losses)
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    y = sample["label"].to_numpy()
    d = res["base"] - res["cpt"]
    summary = {"rows": int(len(sample)), "cpt": cpt, "adapter": str(adapter),
               "mean_loss": {k: float(v.mean()) for k, v in res.items()},
               "mean_reduction": float(d.mean()), "share_rows_improved": float((d > 0).mean()),
               "by_label": {str(k): {"rows": int((y == k).sum()), "base": float(res["base"][y == k].mean()),
                                     "cpt": float(res["cpt"][y == k].mean())} for k in (0, 1)},
               "tokens_mean": float(np.mean([len(x) for x in ids]))}
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"change_id": sample["change_id"], "label": y, "loss_base": res["base"], "loss_cpt": res["cpt"]}).to_parquet(
        out.with_suffix(".parquet"), index=False)
    atomic_write_json(out, summary)
    return summary
