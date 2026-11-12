import json
import os
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, Literal, Union

import numpy as np
import pyarrow as pa
import yaml
from pydantic import Field, TypeAdapter, model_validator

from ptm import __version__
from ptm.algorithms import advantages, get_recipe, population_variance, ranks
from ptm.errors import PTMError
from ptm.importer import check_depth
from ptm.schemas import Completion, Model, Snapshot
from ptm.storage import Store, atomic_write, canonical, digest, doctor, file_hash, write_parquet


class TermBase(Model):
    id: str = Field(min_length=1, max_length=80)
    source: str = Field(min_length=1, max_length=120)
    weight: float = Field(ge=-1e6, le=1e6)


class Identity(TermBase):
    type: Literal["identity"]


class BooleanBonus(TermBase):
    type: Literal["boolean_bonus"]


class ExcessLengthPenalty(TermBase):
    type: Literal["excess_length_penalty"]
    threshold: float = Field(gt=0, le=1e9)
    weight: float = Field(ge=0, le=1e6)


Term = Annotated[Union[Identity, BooleanBonus, ExcessLengthPenalty], Field(discriminator="type")]


class RewardSpec(Model):
    spec_version: Literal[1] = 1
    name: str = Field(min_length=1, max_length=100)
    inputs: list[str] = Field(min_length=1, max_length=64)
    terms: list[Term] = Field(min_length=1, max_length=64)
    aggregation: Literal["sum"] = "sum"
    missing_data: Literal["fail_group"] = "fail_group"

    @model_validator(mode="after")
    def closed_inputs(self) -> "RewardSpec":
        if len({t.id for t in self.terms}) != len(self.terms):
            raise ValueError("term IDs must be unique")
        if len(set(self.inputs)) != len(self.inputs):
            raise ValueError("declared inputs must be unique")
        if {t.source for t in self.terms} != set(self.inputs):
            raise ValueError("inputs must exactly declare all term sources")
        return self


def load_spec(path: Path) -> RewardSpec:
    if path.stat().st_size > 1024 * 1024:
        raise PTMError("PTM_SPEC_INVALID", "Spec exceeds 1 MiB")
    try:
        data = yaml.safe_load(path.read_text())
        check_depth(data)
        return RewardSpec.model_validate(data)
    except (ValueError, yaml.YAMLError, RecursionError) as exc:
        raise PTMError("PTM_SPEC_INVALID", str(exc)) from None


def evaluate(completions: list[Completion], spec: RewardSpec) -> tuple[list[float], list[dict[str, float]]]:
    columns: dict[str, np.ndarray[Any, Any]] = {}
    for source in spec.inputs:
        input_values = []
        for c in completions:
            if source.startswith("component:"):
                value = c.reward_components.get(source.removeprefix("component:"))
            elif source == "completion_token_count":
                value = c.completion_token_count
            elif source == "reward_total":
                value = c.reward_total
            else:
                value = c.inputs.get(source)
            if value is None:
                raise PTMError("PTM_REWARD_INPUT_MISSING", f"Missing {source} for {c.completion_id}",
                               missing_fields=[source])
            input_values.append(value)
        columns[source] = np.asarray(input_values, dtype=np.float64)
    terms: dict[str, np.ndarray[Any, Any]] = {}
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            for term in spec.terms:
                values = columns[term.source]
                if isinstance(term, BooleanBonus):
                    if not np.isin(values, [0, 1]).all():
                        raise PTMError("PTM_REWARD_INPUT_INVALID", f"{term.source} must be boolean or 0/1")
                    contribution = values * term.weight
                elif isinstance(term, ExcessLengthPenalty):
                    if (values < 0).any():
                        raise PTMError("PTM_REWARD_INPUT_INVALID", "Length cannot be negative")
                    contribution = -term.weight * np.maximum(0, values-term.threshold) / term.threshold
                else:
                    contribution = values * term.weight
                terms[term.id] = contribution
            totals = np.sum(list(terms.values()), axis=0)
            if not np.isfinite(totals).all():
                raise FloatingPointError()
    except FloatingPointError:
        raise PTMError("PTM_NUMERICAL_OVERFLOW", "RewardSpec produced nonfinite totals") from None
    return totals.tolist(), [{key: float(value[i]) for key, value in terms.items()} for i in range(len(completions))]


class GroupFilter(Model):
    generation_step: tuple[int, int] | None = None
    group_ids: list[str] | None = Field(default=None, max_length=10000)


class AnalysisRequest(Model):
    run_id: str
    snapshot_id: str | None = None
    reward_spec: RewardSpec
    algorithm_recipe_id: str = "trl-0.24.0-grpo-group-v1"
    group_filter: GroupFilter = Field(default_factory=GroupFilter)


def analysis_identity(store: Store, request: AnalysisRequest) -> tuple[str, dict[str, Any]]:
    snapshot = store.manifest(request.run_id, request.snapshot_id)["snapshot_id"]
    recipe = get_recipe(request.algorithm_recipe_id)
    params = {"ptm_version": __version__, "engine_revision": "reward-diff-v1",
              "snapshot_id": snapshot, "run_id": request.run_id,
              "recipe": recipe.model_dump(), "reward_spec": request.reward_spec.model_dump(),
              "group_filter": request.group_filter.model_dump()}
    return "ana_" + digest(params), params


def compute_diff(snapshot: Snapshot, request: AnalysisRequest) -> dict[str, Any]:
    if snapshot.run.recipe_id != request.algorithm_recipe_id:
        raise PTMError("PTM_RECIPE_MISMATCH", "Requested recipe does not match recorded run recipe")
    if snapshot.run.capture.tier == "L0":
        raise PTMError("PTM_CAPTURE_LEVEL_TOO_LOW", "Completion-level capture is required")
    recipe = get_recipe(request.algorithm_recipe_id, framework=snapshot.run.framework,
                        framework_version=snapshot.run.framework_version)
    batches = {b.generation_batch_id: b for b in snapshot.batches}
    selected = sorted([g for g in snapshot.groups if
        (request.group_filter.group_ids is None or g.group_id in request.group_filter.group_ids) and
        (request.group_filter.generation_step is None or
         request.group_filter.generation_step[0] <= batches[g.generation_batch_id].generation_step <=
         request.group_filter.generation_step[1])], key=lambda g: g.group_id)
    if not selected:
        raise PTMError("PTM_EMPTY_SELECTION", "No prompt groups match this selection")
    if request.group_filter.group_ids is not None:
        if set(request.group_filter.group_ids) - {g.group_id for g in selected}:
            raise PTMError("PTM_NOT_FOUND", "One or more selected groups do not belong to the filtered snapshot")
    if recipe.scale == "batch" and (snapshot.run.config.get("reward_batch_scope") != "full_snapshot" or
                                    len(selected) != len(snapshot.groups) or
                                    snapshot.run.capture.sampling_probability != 1 or
                                    snapshot.run.expected_total_groups != len(snapshot.groups) or
                                    snapshot.run.drop_count != 0 or snapshot.run.config.get("capture_errors", 0) != 0 or
                                    any(g.capture.sampling_probability != 1 for g in selected)):
        raise PTMError("PTM_BATCH_SCOPE_UNAVAILABLE", "Batch scaling needs explicitly complete full_snapshot scope")
    by_group: dict[str, list[Completion]] = {}
    for c in snapshot.completions:
        by_group.setdefault(c.group_id, []).append(c)
    groups = [sorted(by_group.get(g.group_id, []), key=lambda c: c.sequence_index) for g in selected]
    for g, rows in zip(selected, groups, strict=True):
        if len(rows) != g.expected_group_size:
            raise PTMError("PTM_INCOMPLETE_GROUP", f"{g.group_id}: {len(rows)}/{g.expected_group_size} recorded")
        if any(c.reward_status != "valid" for c in rows):
            raise PTMError("PTM_INVALID_REWARD", f"{g.group_id} contains non-valid rewards")
    flat = [c for group in groups for c in group]
    totals, components = evaluate(flat, request.reward_spec)
    offsets, after, position = [], [], 0
    for group in groups:
        offsets.append(position)
        after.append(totals[position:position + len(group)])
        position += len(group)
    before = [[float(c.reward_total) for c in group] for group in groups]  # type: ignore[arg-type]
    old_adv, new_adv = advantages(before, recipe), advantages(after, recipe)
    changes, group_stats = [], []
    for index, (group, old, new) in enumerate(zip(groups, before, after, strict=True)):
        rank_old, rank_new = ranks(old), ranks(new)
        reversals = sum((old[i]-old[j])*(new[i]-new[j]) < 0 for i in range(len(group)) for j in range(i))
        variance_old, variance_new = population_variance(old), population_variance(new)
        group_stats.append({"group_id": selected[index].group_id, "variance_before": variance_old,
                            "variance_after": variance_new, "variance_ddof": 0, "rank_reversals": int(reversals),
                            "zero_variance_rescued": variance_old == 0 and variance_new > 0,
                            "activation_change": (variance_new > 0) != (variance_old > 0),
                            "top1_changed": {c.completion_id for c, r in zip(group, rank_old) if r == 1} !=
                                            {c.completion_id for c, r in zip(group, rank_new) if r == 1}})
        for i, c in enumerate(group):
            a, b = old_adv[index][i], new_adv[index][i]
            changes.append({"completion_id": c.completion_id, "group_id": c.group_id,
                            "reward_before": old[i], "reward_after": new[i], "delta_reward": new[i]-old[i],
                            "components_before": c.reward_components, "components_after": components[offsets[index]+i],
                            "rank_before": rank_old[i], "rank_after": rank_new[i],
                            "advantage_before": a, "advantage_after": b, "sign_flip": a*b < 0,
                            "affected": old[i] != new[i] or rank_old[i] != rank_new[i] or a != b or
                                        c.reward_components != components[offsets[index]+i],
                            "grade": c.grade.model_dump() if c.grade else None})
    affected = {c["group_id"] for c in changes if c["affected"]}
    return {
        "summary": {"groups": len(groups), "completions": len(flat), "affected_groups": len(affected),
                    "rank_reversals": sum(g["rank_reversals"] for g in group_stats),
                    "sign_flips": sum(c["sign_flip"] for c in changes),
                    "zero_variance_rescued": sum(g["zero_variance_rescued"] for g in group_stats)},
        "changes": changes, "groups": group_stats,
        "coverage": {**doctor(snapshot)["coverage"], "selected_groups": len(groups)},
        "evidence": {"completion": "observed", "reward_after": "recomputed", "advantage": "recomputed",
                     "future_outcome": "unavailable", "loss_replay": "unavailable", "estimated": []},
        "limits": ["Fixed saved completions only; retrain validation pending.",
                   "Statistics describe captured groups, not unbiased population estimates.",
                   "Zero variance does not imply zero full loss gradient."],
    }


def run_analysis(
    store: Store, request: AnalysisRequest, *, cancelled: Callable[[], bool] | None = None
) -> dict[str, Any]:
    analysis_id, params = analysis_identity(store, request)
    target = store.path("analyses", analysis_id)
    if target.exists():
        return load_analysis(store, analysis_id)
    snapshot = store.load(request.run_id, params["snapshot_id"], include_advanced=False)
    result = {"analysis_id": analysis_id, "status": "succeeded", **params, **compute_diff(snapshot, request)}
    result["result_hash"] = digest(result)
    if cancelled and cancelled():
        raise PTMError("PTM_CANCELLED", "Cancelled before publishing analysis")
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".pending-", dir=target.parent))
    try:
        atomic_write(stage / "result.json", canonical(result))
        write_parquet(stage / "diff.parquet", result["changes"], {
            "completion_id": pa.string(), "group_id": pa.string(), "delta_reward": pa.float64(),
            "advantage_before": pa.float64(), "advantage_after": pa.float64()})
        atomic_write(stage / "manifest.json", canonical({"files": {p.name: file_hash(p) for p in stage.iterdir()}}))
        try:
            os.rename(stage, target)
        except OSError:
            if not target.exists():
                raise
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return load_analysis(store, analysis_id)


def load_analysis(store: Store, analysis_id: str) -> dict[str, Any]:
    path = store.path("analyses", analysis_id)
    try:
        manifest = json.loads((path / "manifest.json").read_bytes())
        for name in ("result.json", "diff.parquet"):
            if file_hash(path / name) != manifest["files"][name]:
                raise PTMError("PTM_INTEGRITY_ERROR", "Analysis file checksum mismatch")
        result: dict[str, Any] = json.loads((path / "result.json").read_bytes())
        expected = result.pop("result_hash")
        if digest(result) != expected:
            raise PTMError("PTM_INTEGRITY_ERROR", "Analysis logical hash mismatch")
        result["result_hash"] = expected
        return result
    except FileNotFoundError:
        raise PTMError("PTM_NOT_FOUND", "Analysis not found", status=404) from None


# Exposed for schema generation and editors; this is not a Python-code evaluator.
TERM_SCHEMA = TypeAdapter(Term).json_schema()
