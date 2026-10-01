"""CPU baselines B0-LR, B0-LGBM, B1-TFIDF-S (spec FR-06/18, T09/T30/T31, AT-26/27).

All fit() methods take a TrainingDatasetView. Fitted state is exported as JSON / LightGBM
text, never pickle. Non-convergence is an error, not a completed run.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .config import StudyConfig
from .features import StructuredPipeline
from .policy import QueryView, TrainingDatasetView, require_training_view
from .util import ExecutionError, IntegrityError, sha256_json

SCORE_SEMANTICS = "supervised_sigmoid_uncalibrated_public_defect_score"
B1_FIELDS = ("message_text", "code_text")


def _fit_lr(x, y: np.ndarray, C: float, max_iter: int, solver: str, seed: int) -> LogisticRegression:  # noqa: N803
    lr = LogisticRegression(C=C, max_iter=max_iter, solver=solver, class_weight=None, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        try:
            lr.fit(x, y)
        except ConvergenceWarning as e:
            raise ExecutionError(f"logistic regression did not converge ({solver}, max_iter={max_iter})") from e
    return lr


def _lr_state(lr: LogisticRegression) -> dict:
    return {"coef": lr.coef_.ravel().tolist(), "intercept": float(lr.intercept_[0]), "n_iter": int(np.max(lr.n_iter_)),
            "solver": lr.solver, "C": lr.C, "max_iter": lr.max_iter}


class B0LR:
    variant = "B0-LR"

    def __init__(self, cfg: StudyConfig, seed: int):
        self.cfg, self.seed = cfg, seed
        self.pipe = StructuredPipeline(list(cfg.structured.features))

    def fit(self, view: TrainingDatasetView) -> dict:
        require_training_view(view)
        x = self.pipe.fit(view).transform(view.frame)
        self.lr = _fit_lr(x, view.labels.to_numpy(), self.cfg.structured.lr.C, self.cfg.structured.lr.max_iter, "lbfgs", self.seed)
        return {"pipeline": self.pipe.state(), "model": _lr_state(self.lr)}

    def predict(self, view: QueryView) -> np.ndarray:
        return self.lr.predict_proba(self.pipe.transform(view.frame))[:, 1]


class B0LGBM:
    variant = "B0-LGBM"

    def __init__(self, cfg: StudyConfig, seed: int):
        self.cfg, self.seed = cfg, seed
        self.pipe = StructuredPipeline(list(cfg.structured.features))

    def fit(self, view: TrainingDatasetView) -> dict:
        require_training_view(view)
        try:
            import lightgbm as lgb
        except OSError as e:
            raise ExecutionError(f"LightGBM native library unavailable (macOS needs libomp): {e}") from e
        p = self.cfg.structured.lgbm
        params = {"objective": "binary", "num_leaves": p.num_leaves, "learning_rate": p.learning_rate,
                  "n_estimators": p.n_estimators, "min_child_samples": p.min_child_samples, "random_state": self.seed,
                  "n_jobs": p.num_threads, "deterministic": True, "force_row_wise": True, "verbose": -1}
        self.model = lgb.LGBMClassifier(**params)
        self.model.fit(pd.DataFrame(self.pipe.raw(view.frame), columns=self.pipe.columns), view.labels.to_numpy())
        booster = self.model.booster_.model_to_string()
        return {"params": params, "lightgbm_version": lgb.__version__, "booster_sha256": sha256_json(booster),
                "booster_text": booster, "categorical": [], "early_stopping": None}

    def predict(self, view: QueryView) -> np.ndarray:
        return self.model.predict_proba(pd.DataFrame(self.pipe.raw(view.frame), columns=self.pipe.columns))[:, 1]


class B1TFIDF:
    variant = "B1-TFIDF-S"

    def __init__(self, cfg: StudyConfig, seed: int):
        self.cfg, self.seed = cfg, seed
        self.pipe = StructuredPipeline(list(cfg.structured.features))
        t = cfg.text_baseline
        self.vecs = {f: TfidfVectorizer(analyzer=t.analyzer, ngram_range=tuple(t.ngram_range), lowercase=t.lowercase,
                                        min_df=t.min_df, max_features=t.max_features_per_field, dtype=np.float32)
                     for f in B1_FIELDS}

    def _x(self, frame: pd.DataFrame) -> sp.csr_matrix:
        blocks = [self.vecs[f].transform(frame[f].fillna("")) for f in self.vecs]
        blocks.append(sp.csr_matrix(self.pipe.transform(frame).astype(np.float32)))
        x = sp.hstack(blocks, format="csr")
        if not sp.issparse(x):
            raise IntegrityError("B1 design matrix must stay sparse")
        return x

    def fit(self, view: TrainingDatasetView) -> dict:
        require_training_view(view)
        self.pipe.fit(view)
        t0 = time.perf_counter()
        for f, v in self.vecs.items():
            v.fit(view.frame[f].fillna(""))
        x = self._x(view.frame)
        t = self.cfg.text_baseline
        self.lr = _fit_lr(x, view.labels.to_numpy(), t.lr.C, t.lr.max_iter, "liblinear", self.seed)
        cols = [f"{f}:{term}" for f in self.vecs for term in self.vecs[f].get_feature_names_out()] + \
               [f"structured:{c}" for c in self.pipe.columns]
        vocab = {f: {"vocabulary": {k: int(i) for k, i in self.vecs[f].vocabulary_.items()},
                     "idf": self.vecs[f].idf_.tolist()} for f in self.vecs}
        return {"pipeline": self.pipe.state(), "model": _lr_state(self.lr), "tfidf": vocab,
                "tfidf_sha256": sha256_json(vocab), "design_columns_sha256": sha256_json(cols),
                "n_features": int(x.shape[1]), "nnz": int(x.nnz), "fit_seconds": time.perf_counter() - t0,
                "fit_role": "supervised_train"}

    def predict(self, view: QueryView) -> np.ndarray:
        return self.lr.predict_proba(self._x(view.frame))[:, 1]


MODELS = {"B0-LR": B0LR, "B0-LGBM": B0LGBM, "B1-TFIDF-S": B1TFIDF}
