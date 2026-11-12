from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION: Literal["0.1.0"] = "0.1.0"
Identifier = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")]


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Capture(Model):
    tier: Literal["L0", "L1", "L2"] = "L1"
    source: str = "normalized-import"
    sampling_probability: float = Field(default=1.0, gt=0, le=1)
    selection_rule: str = "full"


class Run(Model):
    run_id: Identifier
    name: str
    schema_version: Literal["0.1.0", "0.2.0"] = SCHEMA_VERSION
    framework: str
    framework_version: str
    algorithm: Literal["grpo"] = "grpo"
    recipe_id: str
    config_hash: str
    model_id: str = "unknown"
    tokenizer_hash: str | None = None
    git_sha: str | None = None
    seed: int | None = None
    material: Literal["synthetic example", "real captured trace"]
    capture: Capture = Field(default_factory=Capture)
    config: dict[str, str | float | int | bool | None] = Field(default_factory=dict)
    expected_total_groups: int | None = Field(default=None, ge=0)
    drop_count: int = Field(default=0, ge=0)


class GenerationBatch(Model):
    generation_batch_id: Identifier
    run_id: Identifier
    policy_version: str
    generation_step: int = Field(ge=0)
    event_time: datetime
    worker_rank: int = Field(default=0, ge=0)


class PromptGroup(Model):
    group_id: Identifier
    generation_batch_id: Identifier
    prompt_id: str
    prompt: str | None = None
    expected_group_size: int = Field(ge=1, le=4096)
    capture: Capture = Field(default_factory=Capture)
    worker_rank: int = Field(default=0, ge=0)


class Grade(Model):
    source: str
    value: float


class Completion(Model):
    completion_id: Identifier
    group_id: Identifier
    sequence_index: int = Field(ge=0)
    text: str | None = None
    text_hash: str | None = None
    reward_total: float | None
    reward_status: Literal["valid", "missing", "invalid", "error"] = "valid"
    reward_components: dict[str, float | None] = Field(default_factory=dict)
    inputs: dict[str, float | bool | None] = Field(default_factory=dict)
    grade: Grade | None = None
    completion_token_count: int | None = Field(default=None, ge=0)
    stop_reason: str | None = None
    token_blob_ref: str | None = None
    worker_rank: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def valid_reward(self) -> "Completion":
        if self.reward_status == "valid" and self.reward_total is None:
            raise ValueError("valid reward requires a finite reward_total; null is not zero")
        if self.reward_status != "valid" and self.reward_total is not None:
            raise ValueError("non-valid reward must have null reward_total")
        return self


class OptimizerStep(Model):
    run_id: Identifier
    optimizer_step: int = Field(ge=0)
    timestamp: datetime
    algorithm_recipe_id: str
    generation_batch_ids: list[Identifier] = Field(default_factory=list)
    loss: float | None = None
    effective_token_count: int | None = Field(default=None, ge=0)


class Metric(Model):
    generation_step: int = Field(ge=0)
    name: str
    value: float
    source: str
    alignment: Literal["generation_step"] = "generation_step"


class TokenBlob(Model):
    token_blob_id: Identifier
    completion_id: Identifier
    tokenizer_hash: str = Field(min_length=1)
    optimizer_step: int = Field(ge=0)
    worker_rank: int = Field(default=0, ge=0)
    microbatch_index: int = Field(default=0, ge=0)
    token_ids: list[int] = Field(min_length=1, max_length=1000000)
    token_pieces: list[str]
    loss_mask: list[Literal[0, 1]]
    behavior_logprobs: list[float] | None = None
    old_policy_logprobs: list[float] | None = None
    current_policy_logprobs: list[float] | None = None
    reference_logprobs: list[float] | None = None
    behavior_policy_version: str | None = None
    old_policy_version: str | None = None
    current_policy_version: str | None = None
    reference_policy_version: str | None = None
    observed_advantage: float | None = None
    advantage_recipe_id: str | None = None
    source: str = Field(min_length=1)
    source_sha256: str | None = None
    dtype: Literal["float32", "float64"] = "float32"

    @model_validator(mode="after")
    def alignment(self) -> "TokenBlob":
        n = len(self.token_ids)
        if len(self.token_pieces) != n or len(self.loss_mask) != n or any(i < 0 for i in self.token_ids):
            raise ValueError("Token IDs, captured tokenizer pieces and loss mask must align")
        for logprobs, version in [
            (self.behavior_logprobs, self.behavior_policy_version),
            (self.old_policy_logprobs, self.old_policy_version),
            (self.current_policy_logprobs, self.current_policy_version),
            (self.reference_logprobs, self.reference_policy_version),
        ]:
            if logprobs is not None and (len(logprobs) != n or version is None or any(v > 1e-6 for v in logprobs)):
                raise ValueError("Each logprob array requires aligned finite log probabilities and an explicit policy version")
        if self.observed_advantage is not None and self.advantage_recipe_id is None:
            raise ValueError("Observed sequence advantage requires its recipe ID")
        return self


class ObjectiveBatch(Model):
    objective_id: Identifier
    run_id: Identifier
    optimizer_step: int = Field(ge=0)
    token_blob_ids: list[Identifier] = Field(min_length=1)
    expected_completion_count: int = Field(ge=1)
    declared_complete: bool = False
    policy_version: str
    old_policy_version: str
    reference_policy_version: str | None = None
    loss_type: Literal["grpo", "dapo", "dr_grpo", "bnpo"] = "dapo"
    epsilon_low: float = Field(default=.2, ge=0, lt=1)
    epsilon_high: float = Field(default=.2, ge=0)
    kl_coefficient: float = Field(default=0, ge=0)
    num_processes: int = Field(default=1, ge=1)
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    global_effective_tokens: int = Field(ge=0)
    max_completion_length: int = Field(ge=1)
    source: str = Field(min_length=1)
    source_loss: float | None = None


class TraceTurn(Model):
    turn_id: Identifier
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_name: str | None = None
    policy_version: str | None = None


class Trajectory(Model):
    trajectory_id: Identifier
    completion_id: Identifier
    turns: list[TraceTurn] = Field(min_length=1)
    source: str


class Snapshot(Model):
    run: Run
    batches: list[GenerationBatch]
    groups: list[PromptGroup]
    completions: list[Completion]
    metrics: list[Metric] = Field(default_factory=list)
    optimizer_steps: list[OptimizerStep] = Field(default_factory=list)
    tokens: list[TokenBlob] = Field(default_factory=list)
    objectives: list[ObjectiveBatch] = Field(default_factory=list)
    trajectories: list[Trajectory] = Field(default_factory=list)

    @model_validator(mode="after")
    def references(self) -> "Snapshot":
        entities_by_key: list[tuple[list[Any], str]] = [
            (self.batches, "generation_batch_id"), (self.groups, "group_id"),
            (self.completions, "completion_id"),
        ]
        for entities, key in entities_by_key:
            ids = [getattr(e, key) for e in entities]
            if len(ids) != len(set(ids)):
                raise ValueError(f"duplicate {key}")
        batches = {b.generation_batch_id for b in self.batches}
        groups = {g.group_id: g for g in self.groups}
        if any(b.run_id != self.run.run_id for b in self.batches):
            raise ValueError("batch run_id foreign key mismatch")
        if any(g.generation_batch_id not in batches for g in self.groups):
            raise ValueError("group generation_batch_id foreign key missing")
        seen: set[tuple[str, int]] = set()
        for c in self.completions:
            if c.group_id not in groups:
                raise ValueError("completion group_id foreign key missing")
            g = groups[c.group_id]
            if c.sequence_index >= g.expected_group_size:
                raise ValueError("sequence_index exceeds expected_group_size")
            sequence_key = (c.group_id, c.sequence_index)
            if sequence_key in seen:
                raise ValueError("duplicate group sequence_index")
            seen.add(sequence_key)
        for step in self.optimizer_steps:
            if step.run_id != self.run.run_id or not set(step.generation_batch_ids) <= batches:
                raise ValueError("optimizer step foreign key mismatch")
        metric_keys = [(m.name, m.generation_step) for m in self.metrics]
        if len(metric_keys) != len(set(metric_keys)):
            raise ValueError("duplicate metric name/generation_step; use distinct names for evaluator sources")
        sources: dict[str, str] = {}
        for metric in self.metrics:
            if metric.name in sources and sources[metric.name] != metric.source:
                raise ValueError("metric source changes within a series; use a distinct metric name")
            sources[metric.name] = metric.source
        if (self.tokens or self.objectives or self.trajectories) and self.run.schema_version != "0.2.0":
            raise ValueError("Advanced capture fields require schema_version 0.2.0")
        if self.tokens and self.run.capture.tier != "L2":
            raise ValueError("Token blobs require explicitly declared L2 capture")
        cs = {c.completion_id for c in self.completions}
        blobs = {t.token_blob_id: t for t in self.tokens}
        if len(blobs) != len(self.tokens):
            raise ValueError("duplicate token_blob_id")
        if any(t.completion_id not in cs for t in self.tokens):
            raise ValueError("Token completion foreign key missing")
        if any(self.run.tokenizer_hash and t.tokenizer_hash != self.run.tokenizer_hash for t in self.tokens):
            raise ValueError("Token tokenizer hash differs from run")
        if len({o.objective_id for o in self.objectives}) != len(self.objectives):
            raise ValueError("duplicate objective_id")
        for objective in self.objectives:
            if (objective.run_id != self.run.run_id or len(set(objective.token_blob_ids)) != len(objective.token_blob_ids)
                    or not set(objective.token_blob_ids) <= set(blobs)):
                raise ValueError("Objective identity or Token foreign keys invalid")
        if any(t.completion_id not in cs for t in self.trajectories):
            raise ValueError("Trajectory completion foreign key missing")
        return self

    def ordered(self) -> dict[str, Any]:
        value = self.model_dump(mode="json")
        for key, field in [
            ("batches", "generation_batch_id"), ("groups", "group_id"),
            ("completions", "completion_id"), ("metrics", "generation_step"),
            ("optimizer_steps", "optimizer_step"),
            ("tokens", "token_blob_id"), ("objectives", "objective_id"), ("trajectories", "trajectory_id"),
        ]:
            value[key] = sorted(value[key], key=lambda x: (x[field], str(sorted(x.items()))))
        if self.run.schema_version == "0.1.0":
            for key in ("tokens", "objectives", "trajectories"):
                value.pop(key)
        return value
