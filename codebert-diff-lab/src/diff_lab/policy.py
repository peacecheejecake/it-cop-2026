"""Source approval, dataset views and the public-test gate (spec FR-02/04/05/13, AT-04/05/17).

fit-style APIs accept only TrainingDatasetView (public train). Predictors receive
QueryView objects that carry no labels; labels are joined only by the evaluator. A
public-test QueryView can be opened only with a study freeze record. These are guards
against logical mistakes, not a security boundary against a malicious operator.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pandas as pd
import yaml

from .util import IntegrityError, PolicyError, read_json, sha256_file

Role = Literal["supervised_train", "cpt_train", "selection", "evaluation_only"]
SPLIT_ROLES: dict[str, Role] = {"train": "supervised_train", "valid": "selection", "test": "evaluation_only"}


@dataclass(frozen=True)
class SourceApproval:
    source_id: str
    archive_sha256: str
    origin_uri: str
    upstream_revision: str
    visibility: Literal["public"]
    license_status: str
    approved_roles: tuple[str, ...]
    approved_by: str
    approved_at: str
    path: Path

    @classmethod
    def load(cls, path: str | Path) -> SourceApproval:
        raw = yaml.safe_load(Path(path).read_text())
        fields = set(cls.__dataclass_fields__) - {"path"}
        if set(raw) != fields:
            raise PolicyError(f"source approval fields mismatch: extra={set(raw) - fields} missing={fields - set(raw)}")
        if raw["visibility"] != "public":
            raise PolicyError("only public sources can be approved for training in this study")
        return cls(**{**raw, "approved_roles": tuple(raw["approved_roles"])}, path=Path(path))

    def verify_archive(self, archive: str | Path) -> str:
        digest = sha256_file(archive)
        if digest != self.archive_sha256:
            raise IntegrityError(f"archive sha256 {digest} does not match approval {self.archive_sha256}")
        return digest

    def require_role(self, role: str) -> None:
        if role not in self.approved_roles:
            raise PolicyError(f"source {self.source_id} not approved for role {role}")


@dataclass(frozen=True)
class _View:
    split: str
    frame: pd.DataFrame
    lineage: dict = field(repr=False)

    @property
    def ids(self) -> list[str]:
        return self.frame["change_id"].tolist()

    def __len__(self) -> int:
        return len(self.frame)


@dataclass(frozen=True)
class TrainingDatasetView(_View):
    labels: pd.Series = field(repr=False, default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.split != "train":
            raise PolicyError(f"TrainingDatasetView requires the public train split, got '{self.split}'")
        if self.lineage.get("visibility") != "public":
            raise PolicyError("training views must come from an approved public source")
        if self.labels is None or len(self.labels) != len(self.frame):
            raise PolicyError("training view requires aligned labels")
        if not self.labels.index.equals(self.frame.index):
            raise PolicyError("training labels must share the frame's row index (positional misalignment)")
        if "label" in self.frame.columns:
            raise PolicyError("training frame must not also carry a label column; labels travel separately")


@dataclass(frozen=True)
class QueryView(_View):
    """Label-free view used for prediction (validation, test, internal)."""

    def __post_init__(self) -> None:
        if "label" in self.frame.columns:
            raise PolicyError("query views must not carry labels")


def require_training_view(view: object) -> TrainingDatasetView:
    if not isinstance(view, TrainingDatasetView):
        raise PolicyError(f"fit requires TrainingDatasetView, got {type(view).__name__}")
    return view


def require_test_unlocked(split: str, freeze_path: str | Path | None) -> dict | None:
    if split != "test":
        return None
    if freeze_path is None or not Path(freeze_path).exists():
        raise PolicyError("public test is sealed: run 'study freeze' first and pass the freeze record")
    record = read_json(freeze_path)
    if record.get("status") != "frozen":
        raise PolicyError("freeze record is not in 'frozen' state")
    return record
