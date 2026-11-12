from typing import Any

from ptm.query import Repository
from ptm.schemas import Model


class CompareRequest(Model):
    left_run_id: str
    left_group_id: str
    right_run_id: str
    right_group_id: str


def compare_groups(repository: Repository, request: CompareRequest) -> dict[str, Any]:
    left = repository.group(request.left_run_id, request.left_group_id)
    right = repository.group(request.right_run_id, request.right_group_id)
    matched = left["prompt_id"] == right["prompt_id"]
    recipe_match = left["recipe_id"] == right["recipe_id"]
    comparable = matched and recipe_match and left["advantage_available"] and right["advantage_available"]
    return {
        "left": left,
        "right": right,
        "prompt_match": matched,
        "recipe_match": recipe_match,
        "advantage_comparable": comparable,
        "comparison_level": "same recipe / same prompt" if comparable else "descriptive only",
        "limits": [
            "Separate rollouts are not one-to-one counterfactual samples; no causal improvement is inferred.",
            "Recipe mismatch: no cross-run advantage delta."
            if not recipe_match
            else "Recipe equality is a prerequisite, not evidence of equivalent sampling distributions.",
        ],
    }
