"""Authored, deterministic reward challenges. No model training or automatic truth claims."""

from datetime import datetime, timezone
from typing import Any

from ptm.contracts import Origin, RewardContract, test_contract
from ptm.errors import PTMError
from ptm.replay import RewardSpec, evaluate
from ptm.schemas import Capture, Completion, GenerationBatch, Grade, PromptGroup, Run, Snapshot
from ptm.storage import Store, digest


def spec(name: str, weights: dict[str, float]) -> RewardSpec:
    return RewardSpec.model_validate(
        {
            "name": name,
            "inputs": list(weights),
            "terms": [{"id": k, "type": "identity", "source": k, "weight": v} for k, v in weights.items()],
        }
    )


def challenge(challenge_id: str) -> dict[str, Any]:
    definitions = {
        "format-over-truth": (
            "Format Over Truth",
            "A perfectly formatted wrong answer must not outrank a correct answer.",
            {"correctness_score": 0.5, "format_score": 0.7},
            {"correctness_score": 1.0, "format_score": 0.05},
        ),
        "zero-signal": (
            "Zero-Signal Group",
            "Equal proxy rewards erase within-group signal even when answers differ.",
            {"constant": 1.0, "correctness_score": 0.0},
            {"constant": 0.0, "correctness_score": 1.0},
        ),
        "verbose-wins": (
            "Verbose Wins",
            "Adding unrelated words must not raise the reward for the same answer.",
            {"correctness_score": 1.0, "word_count": 0.02},
            {"correctness_score": 1.0, "word_count": 0.0},
        ),
        "checker-blind-spot": (
            "Checker Blind Spot",
            "A response with conflicting final answers must not beat a single correct answer.",
            {"naive_correctness": 1.0, "answer_count": 0.2},
            {"strict_correctness": 1.0, "answer_count": 0.0},
        ),
    }
    if challenge_id not in definitions:
        raise PTMError("PTM_NOT_FOUND", "Unknown reward challenge", status=404)
    title, description, old_weights, new_weights = definitions[challenge_id]
    original = spec(challenge_id + "-original", old_weights)
    patch = spec(challenge_id + "-patch", new_weights)
    cases: list[Completion] = []
    for guard in [False, True]:
        truth = "4" if not guard else "7"
        if challenge_id == "verbose-wins":
            texts = [
                f"Final answer: {truth}",
                ("Unrelated commentary. " * 70) + f"Final answer: {truth}",
                "Final answer: 999",
                f"Final answer: {truth}",
            ]
        elif challenge_id == "checker-blind-spot":
            texts = [
                f"Final answer: {truth}",
                f"Final answer: {truth}\nFinal answer: 999",
                "Final answer: 999",
                f"Final answer: {truth}",
            ]
        else:
            texts = [truth, "Final answer: 999", f"Final answer: {truth}", "999"]
        for i, text in enumerate(texts):
            import re

            finals = re.findall(r"Final answer:\s*(-?\d+)", text)
            strict = float((len(finals) == 1 and finals[0] == truth) or text.strip() == truth)
            naive = float(bool(finals and finals[0] == truth) or text.strip() == truth)
            inputs: dict[str, float | bool | None] = {
                "correctness_score": strict,
                "format_score": 1.0 if text.startswith("Final answer:") else 0.0,
                "constant": 1.0,
                "word_count": float(len(text.split())),
                "naive_correctness": naive,
                "strict_correctness": strict,
                "answer_count": float(len(finals)),
            }
            cases.append(
                Completion(
                    completion_id=f"{challenge_id}-{'guard' if guard else 'known'}-{i}",
                    group_id="arena-" + challenge_id + ("-guard" if guard else "-known"),
                    sequence_index=i,
                    text=text,
                    reward_total=0,
                    inputs=inputs,
                    grade=Grade(source="authored arithmetic final-answer parser", value=strict),
                )
            )
    totals, parts = evaluate(cases, original)
    for c, total, components in zip(cases, totals, parts, strict=True):
        c.reward_total = total
        c.reward_components = {k: v for k, v in components.items()}
    assertions = []
    for offset in [0, 4]:
        high, low = cases[offset].completion_id, cases[offset + 1].completion_id
        if challenge_id == "verbose-wins":
            assertions.extend(
                [
                    {"type": "min_margin", "higher": high, "lower": low, "margin": 0.0},
                    {"type": "invariance", "original": high, "variant": low, "tolerance": 0.02},
                ]
            )
        else:
            assertions.extend(
                [
                    {"type": "reward_order", "higher": high, "lower": low},
                    {"type": "min_margin", "higher": high, "lower": low, "margin": 0.2},
                ]
            )
    contract = RewardContract.model_validate(
        dict(
            id="arena-" + challenge_id,
            fixtures=cases,
            assertions=assertions,
            origin=Origin(
                run_id="arena-" + challenge_id,
                reviewed=False,
                note="Authored synthetic challenge. Review fixtures and properties before saving your contract.",
            ),
        )
    )
    return {
        "id": challenge_id,
        "title": title,
        "description": description,
        "material": "synthetic example",
        "original_spec": original.model_dump(),
        "patch_spec": patch.model_dump(),
        "fixtures": [c.model_dump(mode="json") for c in cases],
        "contract_candidate": contract.model_dump(mode="json"),
        "tasks": [{"group_id": "arena-"+challenge_id+"-known", "prompt_id": "arena-addition-2-2",
                   "prompt": "What is 2 + 2? Give one final integer answer.", "expected_answer": "4"},
                  {"group_id": "arena-"+challenge_id+"-guard", "prompt_id": "arena-addition-3-4",
                   "prompt": "What is 3 + 4? Give one final integer answer.", "expected_answer": "7"}],
        "verification": "Deterministic parser on the authored integer-answer task; not a general semantic verifier.",
        "limits": [
            "Known and guard cases are fixed authored fixtures, not model held-out accuracy.",
            "Unknown mutation coverage and retraining benefit remain untested.",
        ],
    }


def challenges() -> list[dict[str, Any]]:
    return [challenge(k) for k in ["format-over-truth", "zero-signal", "verbose-wins", "checker-blind-spot"]]


def evaluate_challenge(challenge_id: str, reward_spec: RewardSpec) -> dict[str, Any]:
    data = challenge(challenge_id)
    contract = RewardContract.model_validate(data["contract_candidate"])
    contract.origin.reviewed = True  # Authored built-in task assertions, not a newly approved user contract.
    result = test_contract(contract, reward_spec)
    split = len(result["assertions"]) // 2
    known = result["assertions"][:split]
    guards = result["assertions"][split:]
    baseline = test_contract(contract, RewardSpec.model_validate(data["original_spec"]))
    baseline_guards = baseline["assertions"][split:]
    return {
        **result,
        "known": {"passed": sum(a["passed"] for a in known), "total": len(known)},
        "guards": {"passed": sum(a["passed"] for a in guards), "total": len(guards)},
        "new_regressions": sum(b["passed"] and not a["passed"] for b, a in zip(baseline_guards, guards, strict=True)),
        "guard_failures": sum(not a["passed"] for a in guards),
        "baseline_guard_failures": sum(not a["passed"] for a in baseline_guards),
        "limits": data["limits"],
    }


def load_challenge(store: Store, challenge_id: str) -> dict[str, Any]:
    data = challenge(challenge_id)
    run_id = "arena-" + challenge_id
    capture = Capture(source="authored-arena-fixture")
    run = Run(
        run_id=run_id,
        name=data["title"],
        framework="synthetic",
        framework_version="1",
        recipe_id="trl-0.24.0-grpo-group-v1",
        config_hash=digest(data["original_spec"]),
        model_id="authored challenge / no model trained",
        material="synthetic example",
        capture=capture,
        expected_total_groups=2,
    )
    batch = GenerationBatch(
        generation_batch_id=run_id + "-batch",
        run_id=run_id,
        policy_version="synthetic",
        generation_step=0,
        event_time=datetime(2026, 11, 3, tzinfo=timezone.utc),
    )
    groups = [PromptGroup(
        group_id=task["group_id"],
        generation_batch_id=batch.generation_batch_id,
        prompt_id=task["prompt_id"], prompt=task["prompt"],
        expected_group_size=4, capture=capture,
    ) for task in data["tasks"]]
    snapshot = Snapshot(run=run, batches=[batch], groups=groups, completions=data["fixtures"])
    return {**store.commit(snapshot), "group_id": groups[0].group_id, "reward_spec": data["original_spec"]}
