"""Strict study configuration (spec implementation-plan §6, FR-01).

Unknown fields, duplicate YAML keys and type coercion fail before any data or model
is touched. Forbidden policies are typed Literal[False]: setting them to true is a
schema error, not an override. PIN_REQUIRED placeholders may remain for families that
are not being run; running a variant requires every section it reads to be pinned.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .registry import Registry
from .util import ConfigError, sha256_json

PIN = "PIN_REQUIRED"
FEATURE_PROFILE_JIT14 = ["ns", "nd", "nf", "entropy", "la", "ld", "lt", "fix", "ndev", "age", "nuc", "exp", "rexp", "sexp"]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class DatasetCfg(Strict):
    snapshot_id: str
    split_id: Literal["upstream-clean1", "upstream-clean2"]
    source_approval: str
    train_visibility: Literal["public"]
    feature_profile: Literal["jit14-audited-v1"]


class ModelCfg(Strict):
    base: Literal["microsoft/codebert-base"]
    revision: str
    local_path: str
    weights_file: str
    weights_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    renderer: Literal["message-add-del-text-v2"]
    max_length: int = Field(512, ge=16, le=512)
    max_message_tokens: int = Field(64, ge=0, le=512)
    fusion_head: Literal["cls-feature-tanh-v1"]


class LRCfg(Strict):
    C: float = Field(gt=0)
    max_iter: int = Field(ge=100)
    class_weight: None = None


class LGBMCfg(Strict):
    num_leaves: int
    learning_rate: float
    n_estimators: int
    min_child_samples: int
    num_threads: int = Field(ge=1)
    deterministic: Literal[True] = True


class StructuredCfg(Strict):
    features: list[str]
    lr_transform: Literal["signed_log1p_median_impute_standardize"]
    lr: LRCfg
    lgbm: LGBMCfg

    @field_validator("features")
    @classmethod
    def _allowlist(cls, v: list[str]) -> list[str]:
        if v != FEATURE_PROFILE_JIT14:
            raise ValueError(f"features must equal the audited jit14 allowlist in order: {FEATURE_PROFILE_JIT14}")
        return v


class TextBaselineCfg(Strict):
    classifier: Literal["logistic_regression"]
    analyzer: Literal["char"]
    ngram_range: list[int]
    lowercase: bool
    min_df: int = Field(ge=1)
    max_features_per_field: int = Field(ge=1)
    fit_role: Literal["supervised_train"]
    lr: LRCfg

    @field_validator("ngram_range")
    @classmethod
    def _ngram(cls, v: list[int]) -> list[int]:
        if len(v) != 2 or not 1 <= v[0] <= v[1]:
            raise ValueError("ngram_range must be [min, max] with 1 <= min <= max")
        return v


class EvaluationCfg(Strict):
    primary_metrics: list[Literal["ap", "recall_at_5pct", "recall_at_10pct"]]
    budget_unit: Literal["change_count"]
    topk_rounding: Literal["ceil"]
    tie_salt: str
    threshold_policy: Literal["validation_max_f1_then_highest_threshold"]
    calibration: Literal["none"]
    require_freeze_for_test: Literal[True]


class PolicyCfg(Strict):
    allow_internal_training: Literal[False]
    allow_internal_demonstrations: Literal[False]
    allow_internal_index_members: Literal[False]
    allow_remote_llm: Literal[False]
    allow_test_selection: Literal[False]
    automatic_remote_logging: Literal[False]


class FinetuneCfg(Strict):
    """Shared downstream protocol for B2-S/B3-S/B4-S/B5-S (spec protocol §2.1). B2 differs only by a frozen encoder.

    v2 used one `learning_rate` for encoder and head; v3 sets `encoder_learning_rate` and
    `head_learning_rate` separately (exactly one of the two forms must be given).
    """
    max_epochs: int = Field(ge=1)
    micro_batch_size: int = Field(ge=1)
    gradient_accumulation_steps: int = Field(ge=1)
    learning_rate: float | None = Field(default=None, gt=0)
    encoder_learning_rate: float | None = Field(default=None, gt=0)
    head_learning_rate: float | None = Field(default=None, gt=0)
    weight_decay: float = Field(ge=0)
    lr_schedule: Literal["constant"]
    max_grad_norm: float = Field(gt=0)
    head_dropout: float = Field(ge=0, lt=1)
    selection_metric: Literal["validation_ap"]
    patience_epochs: int = Field(ge=1)
    tie_rule: Literal["earlier_checkpoint"]
    class_weight: None = None
    precision: Literal["fp32", "bf16_encoder_autocast"]
    device: Literal["cuda", "mps", "cpu"]
    allow_cpu: bool = False
    eval_batch_size: int = Field(ge=1)

    @model_validator(mode="after")
    def _lr_form(self) -> FinetuneCfg:
        shared = self.learning_rate is not None
        split = self.encoder_learning_rate is not None and self.head_learning_rate is not None
        partial = (self.encoder_learning_rate is None) != (self.head_learning_rate is None)
        if partial or shared == split:
            raise ValueError("set either learning_rate (shared) or both encoder_learning_rate and head_learning_rate")
        return self

    @property
    def encoder_lr(self) -> float:
        return self.learning_rate if self.learning_rate is not None else self.encoder_learning_rate  # type: ignore[return-value]

    @property
    def head_lr(self) -> float:
        return self.learning_rate if self.learning_rate is not None else self.head_learning_rate  # type: ignore[return-value]


class CptCfg(Strict):
    """Continued pre-training shared by B4-S (MLM) and B5-S (MLM+RMI) (spec protocol §2.1/§6)."""
    permitted_role: Literal["cpt_train"]
    corpus: Literal["evidence_query_text"]
    budget_unit: Literal["nonpadding_input_token_exposures"]
    budget: int = Field(ge=1)
    micro_batch_size: int = Field(ge=1)
    gradient_accumulation_steps: int = Field(ge=1)
    learning_rate: float = Field(gt=0)
    weight_decay: float = Field(ge=0)
    lr_schedule: Literal["linear_warmup_linear_decay"]
    warmup_fraction: float = Field(gt=0, lt=1)
    max_grad_norm: float = Field(gt=0)
    mlm_probability: float = Field(gt=0, lt=1)
    mask_replace_fraction: float = Field(ge=0, le=1)
    random_replace_fraction: float = Field(ge=0, le=1)
    protect_renderer_structure: Literal[True]
    zero_selection_rule: Literal["force_one_uniform"]
    b5_task_schedule: Literal["alternating_1_to_1", "alternating_2_to_1"]
    rmi_replacement_probability: float = Field(gt=0, lt=1)
    dev_eval_every_updates: int = Field(ge=1)
    dev_mask_seed: int
    checkpoint_every_updates: int = Field(ge=1)
    precision: Literal["fp32", "bf16_encoder_autocast"]
    device: Literal["cuda", "mps", "cpu"]
    allow_cpu: bool = False
    eval_batch_size: int = Field(ge=1)

    @field_validator("random_replace_fraction")
    @classmethod
    def _fractions(cls, v: float, info) -> float:  # noqa: ANN001
        if info.data.get("mask_replace_fraction", 0) + v > 1:
            raise ValueError("mask_replace_fraction + random_replace_fraction must be <= 1")
        return v


class LlmExamplesCfg(Strict):
    source_role: Literal["supervised_train"]
    visibility: Literal["public"]
    k: int = Field(ge=1)
    class_counts: dict[Literal["negative", "positive"], int]
    seeds: list[int]
    index_write_from_query: Literal[False]

    @model_validator(mode="after")
    def _k(self) -> LlmExamplesCfg:
        if sum(self.class_counts.values()) != self.k or set(self.class_counts) != {"negative", "positive"}:
            raise ValueError("class_counts must cover negative and positive and sum to k")
        return self


class RetrievalCfg(Strict):
    """L2-S retriever (spec comparison-matrix §6, T39): class-conditional char n-gram TF-IDF cosine over public train."""
    method: Literal["class_conditional_tfidf_char_cosine"]
    fit_role: Literal["supervised_train"]
    field: Literal["query_text"]
    ngram_range: list[int]
    lowercase: bool
    min_df: int = Field(ge=1)
    max_features: int = Field(ge=1)
    exclude_same_content: Literal[True]
    time_filter: Literal["none_static_benchmark"]
    order: Literal["seeded_permutation_per_query"]
    tie_break: Literal["seeded_hash"]


class LlmCfg(Strict):
    """Frozen local LLM for L0-S/L1-S (spec implementation-plan §6 llm, protocol §11/§12)."""
    backend: Literal["local_hf"]
    model_id: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    tokenizer_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    license: str
    local_path: str
    files_sha256: dict[str, str]
    precision_profile: Literal["bf16", "fp32"]
    chat_template_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    task_prompt_id: Literal["jit-defect-binary-v1"]
    task_prompt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    feature_serializer: Literal["jit14-named-raw-v1"]
    scorer: Literal["candidate_loglikelihood"]
    labels: list[Literal["0", "1"]]
    max_context_tokens: int = Field(ge=512)
    query_truncation: Literal["forbidden_after_evidence_freeze"]
    train_weights: Literal[False]
    examples: LlmExamplesCfg
    retrieval_enabled: bool
    retrieval: RetrievalCfg | None = None
    device: Literal["cuda", "mps", "cpu"]
    allow_cpu: bool = False
    batch_token_budget: int = Field(ge=512)

    @model_validator(mode="after")
    def _retrieval(self) -> LlmCfg:
        if self.retrieval_enabled != (self.retrieval is not None):
            raise ValueError("retrieval_enabled requires a retrieval section (and only then)")
        return self

    @field_validator("labels")
    @classmethod
    def _labels(cls, v: list[str]) -> list[str]:
        if v != ["0", "1"]:
            raise ValueError('labels must be ["0", "1"] (index 1 is the positive class)')
        return v


class StudyConfig(Strict):
    schema_version: Literal[2]
    matrix_version: Literal[2]
    study_id: str
    track: Literal["controlled", "reference"]
    protocol_version: str
    dataset: DatasetCfg
    model: ModelCfg
    experiments: list[str]
    evidence_profile: Literal["matched", "native"]
    information_profile: Literal["structured_fused"]
    structured: StructuredCfg
    text_baseline: TextBaselineCfg
    seeds: list[int]
    evaluation: EvaluationCfg
    policy: PolicyCfg
    llm: LlmCfg | Literal["PIN_REQUIRED"] | None = None
    cpt: CptCfg | Literal["PIN_REQUIRED"] | None = None
    finetune: FinetuneCfg | Literal["PIN_REQUIRED"] | None = None


SECTIONS_READ = {
    "B0-LR": ["dataset", "structured"], "B0-LGBM": ["dataset", "structured"],
    "B1-TFIDF-S": ["dataset", "model", "structured", "text_baseline"],
    "B2-S": ["dataset", "model", "structured", "finetune"], "B3-S": ["dataset", "model", "structured", "finetune"],
    "B4-S": ["dataset", "model", "structured", "finetune", "cpt"], "B5-S": ["dataset", "model", "structured", "finetune", "cpt"],
    "L0-S": ["dataset", "model", "structured", "llm"], "L1-S": ["dataset", "model", "structured", "llm"],
    "L2-S": ["dataset", "model", "structured", "llm"],
}


class _NoDupLoader(yaml.SafeLoader):
    pass


def _construct_mapping(loader: _NoDupLoader, node: yaml.MappingNode, deep: bool = False) -> dict:
    seen = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in seen:
            raise ConfigError(f"duplicate YAML key '{key}' at line {key_node.start_mark.line + 1}")
        seen.add(key)
    return loader.construct_mapping(node, deep=deep)


_NoDupLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


def _find_pins(obj: Any, path: str = "") -> list[str]:
    if obj == PIN:
        return [path or "<root>"]
    if isinstance(obj, dict):
        return [p for k, v in obj.items() for p in _find_pins(v, f"{path}.{k}" if path else k)]
    if isinstance(obj, list):
        return [p for i, v in enumerate(obj) for p in _find_pins(v, f"{path}[{i}]")]
    return []


def load_study(path: str | Path, registry: Registry | None = None) -> tuple[StudyConfig, dict, str]:
    with open(path, encoding="utf-8") as f:
        raw = yaml.load(f, Loader=_NoDupLoader)  # noqa: S506 - SafeLoader subclass
    if not isinstance(raw, dict):
        raise ConfigError("study config must be a mapping")
    try:
        cfg = StudyConfig.model_validate(raw)
    except ValidationError as e:
        raise ConfigError(f"invalid study config {path}:\n{e}") from e
    registry = registry or Registry()
    for vid in cfg.experiments:
        registry.resolve(vid)
    if len(set(cfg.experiments)) != len(cfg.experiments):
        raise ConfigError("duplicate experiment ids")
    return cfg, raw, sha256_json(raw)


def resolve_local_path(value: str, project_root: Path) -> Path:
    """'cache:<rel>' resolves against the nearest ancestor holding experiments/.cache (works in worktrees)."""
    if not value.startswith("cache:"):
        p = Path(value)
        return p if p.is_absolute() else (project_root / p).resolve()
    rel = value.removeprefix("cache:")
    for parent in [project_root, *project_root.parents]:
        cache = parent / "experiments" / ".cache"
        if cache.is_dir():
            return cache / rel
    raise ConfigError(f"shared cache not found above {project_root} for {value}")


def require_pinned(raw: dict, variant_id: str) -> None:
    sections = SECTIONS_READ.get(variant_id)
    if sections is None:
        raise ConfigError(f"no section map for variant {variant_id}; not implemented in this milestone")
    missing = [p for s in sections for p in _find_pins(raw.get(s), s)]
    absent = [s for s in sections if raw.get(s) is None]
    if missing or absent:
        raise ConfigError(f"{variant_id} cannot run: unpinned {missing} / missing sections {absent}")
