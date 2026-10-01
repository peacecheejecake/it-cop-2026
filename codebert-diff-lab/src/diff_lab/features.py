"""Structured feature pipeline (spec FR-05, AT-05). Fit only on TrainingDatasetView.

LR profile `signed_log1p_median_impute_standardize`: x -> sign(x)*log1p(|x|) (keeps the
audited negative lt/age values instead of clipping them), median imputation and
standardisation with train-only statistics. LightGBM reads the raw allowlisted values.
Parameters are stored as JSON, never pickled.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .policy import TrainingDatasetView, require_training_view
from .util import IntegrityError, sha256_json


def _matrix(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise IntegrityError(f"missing feature columns {missing}; never zero-filled")
    x = frame[columns].to_numpy(dtype=np.float64)
    if np.isinf(x).any():
        raise IntegrityError("infinite feature values")
    return x


@dataclass
class StructuredPipeline:
    columns: list[str]
    median: list[float] | None = None
    mean: list[float] | None = None
    scale: list[float] | None = None

    def fit(self, view: TrainingDatasetView) -> StructuredPipeline:
        require_training_view(view)
        z = self._signed_log(_matrix(view.frame, self.columns))
        med = np.nanmedian(z, axis=0)
        z = np.where(np.isnan(z), med, z)
        sd = z.std(axis=0)
        self.median, self.mean, self.scale = med.tolist(), z.mean(axis=0).tolist(), np.where(sd > 0, sd, 1.0).tolist()
        return self

    @staticmethod
    def _signed_log(x: np.ndarray) -> np.ndarray:
        return np.sign(x) * np.log1p(np.abs(x))

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.median is None:
            raise IntegrityError("pipeline not fitted")
        z = self._signed_log(_matrix(frame, self.columns))
        z = np.where(np.isnan(z), np.asarray(self.median), z)
        return (z - np.asarray(self.mean)) / np.asarray(self.scale)

    def raw(self, frame: pd.DataFrame) -> np.ndarray:
        return _matrix(frame, self.columns)

    def state(self) -> dict:
        s = {"kind": "signed_log1p_median_impute_standardize", "columns": self.columns,
             "median": self.median, "mean": self.mean, "scale": self.scale}
        return {**s, "sha256": sha256_json(s)}
