"""Deterministic sampled-token signals and full-batch objective replay; no gradients inferred."""

from typing import Any

import numpy as np

from ptm.algorithms import get_recipe
from ptm.errors import PTMError
from ptm.replay import AnalysisRequest, GroupFilter, RewardSpec, compute_diff
from ptm.schemas import ObjectiveBatch, Snapshot, TokenBlob
from ptm.storage import Store, atomic_write, canonical, digest


def token_signals(
    blob: TokenBlob, advantage: float | None = None, epsilon_low: float = 0.2, epsilon_high: float = 0.2
) -> dict[str, Any]:
    a = blob.observed_advantage if advantage is None else advantage
    current, old, reference = blob.current_policy_logprobs, blob.old_policy_logprobs, blob.reference_logprobs
    ratio = log_ratio = surrogate = kl = None
    try:
        with np.errstate(over="raise", invalid="raise"):
            if current is not None and old is not None:
                log_ratio = np.asarray(current) - np.asarray(old)
                ratio = np.exp(log_ratio)
                if a is not None:
                    surrogate = np.minimum(ratio * a, np.clip(ratio, 1 - epsilon_low, 1 + epsilon_high) * a)
            if current is not None and reference is not None:
                delta = np.asarray(reference) - np.asarray(current)
                kl = np.expm1(delta) - delta
    except FloatingPointError:
        raise PTMError(
            "PTM_NUMERICAL_OVERFLOW", "Token logprob difference exceeds stable exponential range"
        ) from None
    missing: dict[str, list[str]] = {}
    if ratio is None:
        missing["ratio"] = [
            name
            for name, value in [("current_policy_logprobs", current), ("old_policy_logprobs", old)]
            if value is None
        ]
    if kl is None:
        missing["sampled_kl"] = [
            name
            for name, value in [("current_policy_logprobs", current), ("reference_logprobs", reference)]
            if value is None
        ]
    if surrogate is None:
        missing["surrogate"] = missing.get("ratio", []) + (["observed_advantage"] if a is None else [])
    rows = []
    for i, (token_id, piece, mask) in enumerate(
        zip(blob.token_ids, blob.token_pieces, blob.loss_mask, strict=True)
    ):
        rows.append(
            {
                "index": i,
                "token_id": token_id,
                "piece": piece,
                "loss_mask": mask,
                "log_ratio": float(log_ratio[i]) if log_ratio is not None and mask else None,
                "ratio": float(ratio[i]) if ratio is not None and mask else None,
                "sampled_kl": float(kl[i]) if kl is not None and mask else None,
                "surrogate": float(surrogate[i]) if surrogate is not None and mask else None,
            }
        )
    return {
        "rows": rows,
        "sequence_advantage": a,
        "missing": missing,
        "evidence": {
            "token_ids": "observed",
            "mask": "observed",
            "logprobs": "observed",
            "ratio_kl_surrogate": "recomputed",
            "gradient": "unavailable",
        },
        "limits": [
            "Sequence advantage is broadcast across effective tokens, not learned independently per token.",
            "Sampled KL estimator is not exact full-vocabulary KL. Masked tokens are unavailable, not zero.",
        ],
    }


def replay_objective(
    snapshot: Snapshot, objective: ObjectiveBatch, spec: RewardSpec | None = None
) -> dict[str, Any]:
    get_recipe(
        snapshot.run.recipe_id,
        framework=snapshot.run.framework,
        framework_version=snapshot.run.framework_version,
    )
    config = snapshot.run.config
    if (
        config.get("importance_sampling_level", "token") != "token"
        or config.get("top_entropy_quantile", 1.0) != 1.0
        or config.get("delta") is not None
        or (config.get("use_vllm", False) and config.get("vllm_importance_sampling_correction", True))
    ):
        raise PTMError(
            "PTM_RECIPE_MISMATCH", "Uncaptured sequence/entropy/vLLM/delta loss modifiers are unsupported"
        )
    blobs_by_id = {b.token_blob_id: b for b in snapshot.tokens}
    blobs = [blobs_by_id[k] for k in objective.token_blob_ids if k in blobs_by_id]
    if (
        not objective.declared_complete
        or len(blobs) != objective.expected_completion_count
        or len(blobs) != len(objective.token_blob_ids)
    ):
        raise PTMError("PTM_INCOMPLETE_OBJECTIVE", "Full explicitly declared optimizer batch is required")
    if (
        snapshot.run.capture.sampling_probability != 1
        or snapshot.run.drop_count
        or snapshot.run.config.get("capture_errors", 0)
    ):
        raise PTMError(
            "PTM_INCOMPLETE_OBJECTIVE", "Objective replay requires full capture without drops/errors"
        )
    cells: dict[tuple[int, int], list[TokenBlob]] = {}
    for blob in blobs:
        if (
            blob.optimizer_step != objective.optimizer_step
            or blob.current_policy_version != objective.policy_version
            or blob.old_policy_version != objective.old_policy_version
        ):
            raise PTMError("PTM_POLICY_MISMATCH", "Optimizer step or current/old policy identity differs")
        if blob.current_policy_logprobs is None or blob.old_policy_logprobs is None:
            raise PTMError(
                "PTM_TOKEN_INPUT_MISSING",
                "Current and old policy logprobs are required",
                missing_fields=["current_policy_logprobs", "old_policy_logprobs"],
            )
        if objective.kl_coefficient and (
            blob.reference_logprobs is None
            or blob.reference_policy_version != objective.reference_policy_version
        ):
            raise PTMError(
                "PTM_TOKEN_INPUT_MISSING",
                "KL term needs matching reference policy logprobs",
                missing_fields=["reference_logprobs", "reference_policy_version"],
            )
        if len(blob.token_ids) > objective.max_completion_length:
            raise PTMError("PTM_MASK_MISMATCH", "Captured completion exceeds declared maximum length")
        if blob.advantage_recipe_id != snapshot.run.recipe_id:
            raise PTMError("PTM_RECIPE_MISMATCH", "Captured advantage recipe differs from run")
        cells.setdefault((blob.worker_rank, blob.microbatch_index), []).append(blob)
    grid = {
        (r, m) for r in range(objective.num_processes) for m in range(objective.gradient_accumulation_steps)
    }
    if set(cells) != grid or len({len(v) for v in cells.values()}) != 1:
        raise PTMError(
            "PTM_INCOMPLETE_OBJECTIVE", "Rank/microbatch grid must be complete with uniform completion counts"
        )
    token_count = sum(sum(b.loss_mask) for b in blobs)
    if token_count != objective.global_effective_tokens:
        raise PTMError(
            "PTM_DENOMINATOR_MISMATCH", "Global effective token denominator does not match recorded masks"
        )
    if not token_count:
        raise PTMError("PTM_DENOMINATOR_MISMATCH", "No effective training tokens")
    overrides: dict[str, float] = {}
    if spec is not None:
        completion_groups = {c.completion_id: c.group_id for c in snapshot.completions}
        ids = sorted({completion_groups[b.completion_id] for b in blobs})
        diff = compute_diff(
            snapshot,
            AnalysisRequest(
                run_id=snapshot.run.run_id,
                reward_spec=spec,
                algorithm_recipe_id=snapshot.run.recipe_id,
                group_filter=GroupFilter(group_ids=ids),
            ),
        )
        overrides = {c["completion_id"]: c["advantage_after"] for c in diff["changes"]}
    losses: dict[str, tuple[float, int]] = {}
    per_completion = []
    for blob in blobs:
        advantage = overrides.get(blob.completion_id, blob.observed_advantage)
        if advantage is None:
            raise PTMError(
                "PTM_TOKEN_INPUT_MISSING", "Missing sequence advantage", missing_fields=["observed_advantage"]
            )
        signals = token_signals(blob, advantage, objective.epsilon_low, objective.epsilon_high)
        total = 0.0
        for row in signals["rows"]:
            if row["loss_mask"]:
                value = -row["surrogate"]
                if objective.kl_coefficient:
                    value += objective.kl_coefficient * row["sampled_kl"]
                total += value
        losses[blob.token_blob_id] = (total, sum(blob.loss_mask))
        per_completion.append(
            {
                "completion_id": blob.completion_id,
                "token_blob_id": blob.token_blob_id,
                "masked_loss_sum": total,
                "effective_tokens": sum(blob.loss_mask),
            }
        )
    numerator = sum(v[0] for v in losses.values())
    if objective.loss_type == "grpo":
        loss = sum(total / max(n, 1) for total, n in losses.values()) / len(blobs)
    elif objective.loss_type == "dr_grpo":
        loss = numerator / (len(blobs) * objective.max_completion_length)
    elif objective.loss_type == "bnpo":
        means = [
            sum(losses[b.token_blob_id][0] for b in cell) / max(sum(sum(b.loss_mask) for b in cell), 1)
            for cell in cells.values()
        ]
        loss = sum(means) / len(means)
    else:
        loss = numerator / objective.global_effective_tokens
    if not np.isfinite(loss):
        raise PTMError("PTM_NUMERICAL_OVERFLOW", "Objective replay produced a nonfinite value")
    return {
        "objective_id": objective.objective_id,
        "loss": float(loss),
        "loss_type": objective.loss_type,
        "normalization_scope": "full global optimizer batch",
        "effective_tokens": token_count,
        "completions": len(blobs),
        "per_completion": per_completion,
        "advantage_source": "recomputed reward spec" if spec else "observed trainer sequence advantage",
        "source_loss": objective.source_loss,
        "source_loss_delta": None if spec or objective.source_loss is None else loss - objective.source_loss,
        "evidence": "recomputed",
        "limits": [
            "Fixed saved policy tensors only; no future model-quality prediction.",
            "Scalar objective replay does not reconstruct gradients or perform an update.",
        ],
    }


def save_objective_analysis(
    store: Store, run_id: str, objective_id: str, spec: RewardSpec | None
) -> dict[str, Any]:
    snapshot = store.load(run_id)
    objective = next((o for o in snapshot.objectives if o.objective_id == objective_id), None)
    if objective is None:
        raise PTMError("PTM_NOT_FOUND", "Objective batch not found", status=404)
    result = replay_objective(snapshot, objective, spec)
    identity = {
        "snapshot_id": store.manifest(run_id)["snapshot_id"],
        "objective": objective.model_dump(),
        "reward_spec": spec.model_dump() if spec else None,
        "engine_revision": "objective-v1",
    }
    result.update({"analysis_id": "obj_" + digest(identity), **identity})
    atomic_write(store.path("objective-analyses", result["analysis_id"] + ".json"), canonical(result))
    return result
