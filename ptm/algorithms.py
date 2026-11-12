from typing import Literal

import numpy as np
from pydantic import Field

from ptm.errors import PTMError
from ptm.schemas import Model


class AlgorithmRecipe(Model):
    recipe_id: str = "trl-0.24.0-grpo-group-v1"
    framework: Literal["trl"] = "trl"
    framework_version: Literal["0.24.0"] = "0.24.0"
    algorithm: Literal["grpo"] = "grpo"
    scale: Literal["group", "none", "batch"] = "group"
    ddof: Literal[1] = 1
    epsilon: float = Field(default=1e-4, ge=1e-4, le=1e-4)
    revision: Literal[1] = 1
    scope: Literal["reward_advantage_only"] = "reward_advantage_only"


RECIPES = {f"trl-0.24.0-grpo-{scale}-v1": AlgorithmRecipe(
    recipe_id=f"trl-0.24.0-grpo-{scale}-v1", scale=scale
) for scale in ("group", "none", "batch")}


def get_recipe(
    recipe_id: str, *, framework: str | None = None, framework_version: str | None = None
) -> AlgorithmRecipe:
    if recipe_id not in RECIPES:
        raise PTMError("PTM_RECIPE_MISMATCH", f"Unsupported numerical recipe: {recipe_id}")
    recipe = RECIPES[recipe_id]
    if framework == "trl" and framework_version != recipe.framework_version:
        raise PTMError("PTM_RECIPE_MISMATCH", "Recorded TRL version differs from the verified recipe version")
    return recipe


def advantages(rewards: list[list[float]], recipe: AlgorithmRecipe) -> list[list[float]]:
    """Independent float64 kernel. Every row is one explicitly complete prompt group."""
    if not rewards or any(not group for group in rewards):
        raise PTMError("PTM_EMPTY_GROUP", "No complete nonempty groups")
    if recipe.scale == "group" and any(len(group) < 2 for group in rewards):
        raise PTMError("PTM_GROUP_TOO_SMALL", "Sample standard deviation needs G >= 2")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            arrays = [np.asarray(group, dtype=np.float64) for group in rewards]
            if any(not np.isfinite(a).all() for a in arrays):
                raise PTMError("PTM_INVALID_REWARD", "Reward contains NaN/Inf")
            batch_std = float(np.concatenate(arrays).std(ddof=1)) if recipe.scale == "batch" else None
            result = []
            for array in arrays:
                centered = array - array.mean()
                if recipe.scale == "group":
                    centered = centered / (array.std(ddof=1) + recipe.epsilon)
                elif recipe.scale == "batch":
                    centered = centered / (batch_std + recipe.epsilon)  # type: ignore[operator]
                if not np.isfinite(centered).all():
                    raise FloatingPointError("nonfinite advantage")
                result.append(centered.tolist())
            return result
    except FloatingPointError:
        raise PTMError("PTM_NUMERICAL_OVERFLOW", "Reward magnitude exceeds stable float64 computation") from None


def ranks(values: list[float]) -> list[int]:
    """Competition ranks, descending: equal values have equal rank (1,1,3)."""
    ordered = sorted(values, reverse=True)
    mapping = {v: ordered.index(v) + 1 for v in set(values)}
    return [mapping[v] for v in values]


def population_variance(values: list[float]) -> float:
    try:
        with np.errstate(over="raise", invalid="raise"):
            result = float(np.var(values, ddof=0))
            if not np.isfinite(result):
                raise FloatingPointError()
            return result
    except FloatingPointError:
        raise PTMError("PTM_NUMERICAL_OVERFLOW", "Group variance exceeds stable float64 computation") from None
