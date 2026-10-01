"""Predictors rebuilt from frozen run artifacts, never refitted (spec protocol §9, FR-13/14).

B0-LR / B1-TFIDF-S: JSON state (pipeline statistics, TF-IDF vocabulary/idf, LR coefficients).
B0-LGBM: LightGBM booster text. B2-S..B5-S: pinned base encoder (B2) or the best-generation
encoder weights (B3-B5) plus the head. L0-S/L1-S: the pinned LLM with the frozen demo IDs.
Every loader checks the hashes recorded at training time before predicting.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .config import StudyConfig
from .features import StructuredPipeline
from .models import B1_FIELDS
from .policy import QueryView, TrainingDatasetView
from .util import IntegrityError, read_json, resolve_generation, sha256_json


def pipeline_from_state(state: dict) -> StructuredPipeline:
    s = {k: state[k] for k in ("kind", "columns", "median", "mean", "scale")}
    if sha256_json(s) != state["sha256"]:
        raise IntegrityError("structured pipeline state hash mismatch")
    return StructuredPipeline(list(state["columns"]), state["median"], state["mean"], state["scale"])


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


class FrozenB0LR:
    def __init__(self, run_dir: Path) -> None:
        st = read_json(run_dir / "model" / "state.json")
        self.pipe = pipeline_from_state(st["pipeline"])
        self.coef, self.intercept = np.asarray(st["model"]["coef"]), st["model"]["intercept"]

    def predict(self, view: QueryView) -> np.ndarray:
        return _sigmoid(self.pipe.transform(view.frame) @ self.coef + self.intercept)


class FrozenB0LGBM:
    """Scores in a subprocess (see lgbm_predict) so torch and LightGBM never share an OpenMP runtime."""

    def __init__(self, run_dir: Path, cfg: StudyConfig) -> None:
        st = read_json(run_dir / "model" / "state.json")
        self.booster_path = run_dir / "model" / "lightgbm.txt"
        if sha256_json(self.booster_path.read_text()) != st["booster_sha256"]:
            raise IntegrityError("LightGBM booster text hash mismatch")
        self.pipe = StructuredPipeline(list(cfg.structured.features))

    def predict(self, view: QueryView) -> np.ndarray:
        import subprocess
        import sys
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            inp, out = Path(tmp) / "x.parquet", Path(tmp) / "s.npy"
            pd.DataFrame(self.pipe.raw(view.frame), columns=self.pipe.columns).to_parquet(inp)
            r = subprocess.run([sys.executable, "-m", "diff_lab.lgbm_predict", str(self.booster_path), str(inp), str(out)],
                               capture_output=True, text=True)
            if r.returncode != 0:
                raise IntegrityError(f"LightGBM scoring subprocess failed ({r.returncode}): {r.stderr[-500:]}")
            return np.load(out)


class FrozenB1:
    def __init__(self, run_dir: Path, cfg: StudyConfig) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        st = read_json(run_dir / "model" / "state.json")
        if sha256_json(st["tfidf"]) != st["tfidf_sha256"]:
            raise IntegrityError("TF-IDF state hash mismatch")
        t = cfg.text_baseline
        self.vecs = {}
        # Block order must match training (message, code); the state JSON is written with sorted keys.
        if set(st["tfidf"]) != set(B1_FIELDS):
            raise IntegrityError(f"unexpected TF-IDF fields {sorted(st['tfidf'])}")
        for f in B1_FIELDS:
            v = st["tfidf"][f]
            vec = TfidfVectorizer(analyzer=t.analyzer, ngram_range=tuple(t.ngram_range), lowercase=t.lowercase,
                                  vocabulary=v["vocabulary"], dtype=np.float32)
            vec.idf_ = np.asarray(v["idf"], dtype=np.float64)
            self.vecs[f] = vec
        self.pipe = pipeline_from_state(st["pipeline"])
        self.coef, self.intercept = np.asarray(st["model"]["coef"]), st["model"]["intercept"]

    def predict(self, view: QueryView) -> np.ndarray:
        blocks = [self.vecs[f].transform(view.frame[f].fillna("")) for f in self.vecs]
        blocks.append(sp.csr_matrix(self.pipe.transform(view.frame).astype(np.float32)))
        x = sp.hstack(blocks, format="csr")
        return _sigmoid(np.asarray(x @ self.coef).ravel() + self.intercept)


class FrozenEncoder:
    def __init__(self, run_dir: Path, cfg: StudyConfig, variant: str, base_path: Path, tokenizer, device: str) -> None:  # noqa: ANN001
        from safetensors.torch import load_file

        from .neural import _torch, build_head, load_encoder, state_hash
        torch = _torch()
        st = read_json(run_dir / "model" / "state.json")
        best = resolve_generation(run_dir / "checkpoints", "best")
        if best is None or best.name != st["best_checkpoint_generation"]:
            raise IntegrityError(f"{run_dir.name}: best checkpoint generation missing or not the recorded one")
        encoder, info = load_encoder(base_path, cfg.model.revision, (cfg.model.weights_file, cfg.model.weights_sha256))
        if variant == "B2-S":
            if state_hash(encoder) != st["encoder_init_state_sha256"]:
                raise IntegrityError("B2 base encoder differs from the frozen encoder state")
        else:
            encoder.load_state_dict(load_file(str(best / "encoder.safetensors")), strict=True)
        head = build_head(info["hidden_size"], len(cfg.structured.features), cfg.finetune.head_dropout, 0)
        head.load_state_dict(load_file(str(best / "head.safetensors")), strict=True)
        self.device = torch.device(device)
        self.encoder, self.head = encoder.to(self.device).eval(), head.to(self.device).eval()
        self.pipe = pipeline_from_state(st["pipeline"])
        self.tok, self.cfg = tokenizer, cfg
        self.precision = cfg.finetune.precision

    def predict(self, view: QueryView) -> np.ndarray:
        from .neural import Encoded, _collate, _torch, encode_inputs
        torch = _torch()
        enc = Encoded(encode_inputs(self.tok, view.frame["query_text"].tolist(), self.cfg.model.max_length),
                      self.pipe.transform(view.frame).astype(np.float32))
        out, bs = [], self.cfg.finetune.eval_batch_size
        with torch.no_grad():
            for i in range(0, len(enc.ids), bs):
                idx = np.arange(i, min(i + bs, len(enc.ids)))
                ids, mask, feats = _collate(enc, idx, self.tok.pad_token_id, self.device)
                ctx = (torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)
                       if self.precision == "bf16_encoder_autocast" else torch.no_grad())
                with ctx:
                    cls = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]
                out.append(torch.sigmoid(self.head(cls.float(), feats)).float().cpu().numpy())
        s = np.concatenate(out).astype(np.float64)
        if not np.isfinite(s).all():
            raise IntegrityError("non-finite scores")
        return s


class FrozenLlm:
    def __init__(self, run_dir: Path, cfg: StudyConfig, variant: str, seed: int, model_path: Path, train: TrainingDatasetView) -> None:
        from .llm import select_demos
        self.manifest = read_json(run_dir / "prompts" / "manifest.json")
        self.cfg, self.variant, self.seed, self.path, self.train = cfg, variant, seed, model_path, train
        if variant == "L1-S":
            demos = select_demos(train, cfg.llm, seed)
            if demos["change_id"].tolist() != [d["change_id"] for d in self.manifest["demos"]]:
                raise IntegrityError("re-derived L1 demo set differs from the frozen prompt manifest")

    def predict(self, view: QueryView, out_dir: Path) -> np.ndarray:
        from .llm import LlmRun
        out_dir.mkdir(parents=True, exist_ok=True)
        res = LlmRun(self.cfg, self.variant, self.seed, self.path, out_dir).fit_predict(self.train, view)
        if read_json(out_dir / "prompts" / "manifest.json") != self.manifest:
            raise IntegrityError("test-time prompt manifest differs from the frozen manifest")
        return res["scores"]
