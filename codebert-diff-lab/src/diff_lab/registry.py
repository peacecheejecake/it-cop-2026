"""Matrix v2 variant registry (spec comparison-matrix §2, experiment-registry.json).

Family IDs (B0..L2) and v0.1 IDs are not run identifiers. A bare legacy ID such as
"B1" is rejected because v0.1 B1 (frozen encoder, code-only) is v0.2 B2-T, while v0.2
B1 is TF-IDF; only an explicit migrate_legacy() call maps it.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .util import ConfigError, read_json, sha256_file

MATRIX_VERSION = 2
REGISTRY_PATH = Path(__file__).resolve().parents[2] / "configs" / "registry" / "matrix-v2.json"


@dataclass(frozen=True)
class Variant:
    id: str
    family: str
    priority: str
    primary: bool
    input_profile: str
    adaptation_mode: str
    task_label_gradient: bool
    encoder_updates: bool
    task_label_demonstrations: bool
    task_label_retrieval_index: bool


class Registry:
    def __init__(self, path: Path = REGISTRY_PATH):
        raw = read_json(path)
        if raw.get("matrix_version") != MATRIX_VERSION:
            raise ConfigError(f"registry matrix_version {raw.get('matrix_version')} != {MATRIX_VERSION}")
        self.digest = sha256_file(path)
        self.variants = {v["id"]: Variant(**{k: v[k] for k in Variant.__dataclass_fields__}) for v in raw["variants"]}
        if len(self.variants) != len(raw["variants"]):
            raise ConfigError("duplicate variant ids in registry")
        self.primary = list(raw["primary_variant_ids"])
        self.core = list(raw["core_variant_ids"])
        self.legacy = dict(raw["legacy_v1_mapping"])
        self.primary_comparisons = [tuple(p) for p in raw["primary_comparisons"]]
        self.constraints = dict(raw["constraints"])
        if any(v.get("internal_training_allowed") or v.get("remote_llm_allowed") for v in raw["variants"]):
            raise ConfigError("registry may not allow internal training or remote LLMs")

    def resolve(self, variant_id: str) -> Variant:
        if variant_id in self.variants:
            return self.variants[variant_id]
        if variant_id in self.legacy:
            raise ConfigError(
                f"'{variant_id}' is a v0.1 / family identifier, not a matrix_version=2 run id; "
                f"use an explicit variant id or migrate_legacy() (v0.1 {variant_id} -> {self.legacy[variant_id]})")
        raise ConfigError(f"unknown variant id '{variant_id}'")

    def migrate_legacy(self, legacy_model_id: str) -> dict:
        if legacy_model_id not in self.legacy:
            raise ConfigError(f"no v0.1 mapping for '{legacy_model_id}'")
        return {"legacy_model_id": legacy_model_id, "legacy_matrix_version": 1,
                "mapped_variant_id": self.legacy[legacy_model_id], "matrix_version": MATRIX_VERSION,
                "note": "id mapping only; renderer changed in v0.2, conditions are not certified identical"}
