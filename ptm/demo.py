from datetime import datetime, timedelta, timezone

from ptm.schemas import Capture, Completion, GenerationBatch, Grade, Metric, PromptGroup, Run, Snapshot
from ptm.storage import Store, digest


def demo_snapshot() -> Snapshot:
    capture = Capture(source="ptm-synthetic-generator", selection_rule="full synthetic fixture")
    config: dict[str, str | float | int | bool | None] = {
        "correctness_weight": 0.5, "format_weight": 0.7, "num_generations": 4}
    run = Run(run_id="demo-format-truth", name="Format over truth / controlled challenge", framework="synthetic",
              framework_version="1", recipe_id="trl-0.24.0-grpo-group-v1", config_hash=digest(config), config=config,
              model_id="synthetic / no model trained", material="synthetic example", capture=capture,
              expected_total_groups=96, seed=42)
    batches: list[GenerationBatch] = []
    groups: list[PromptGroup] = []
    completions: list[Completion] = []
    metrics: list[Metric] = []
    start = datetime(2026, 11, 2, tzinfo=timezone.utc)
    for step_index in range(24):
        step = 6400 + step_index * 160
        bid = f"gb-{step}"
        batches.append(GenerationBatch(generation_batch_id=bid, run_id=run.run_id, generation_step=step,
                                       policy_version=f"synthetic-{step}", event_time=start + timedelta(minutes=step_index)))
        step_start = len(completions)
        for j in range(4):
            gid = f"group-{step}-{j}"
            a, b = step_index + 3, j + 2
            groups.append(PromptGroup(group_id=gid, generation_batch_id=bid, prompt_id=f"math-{a}-{b}",
                                      prompt=f"What is {a} + {b}? Give one final answer.", expected_group_size=4,
                                      capture=capture))
            collapse = j == 3 and step_index > 12
            values = [(1.0, 0.0), (0.0, 1.0), (1.0, 0.6), (0.0, 0.3)]
            if j > 0:
                # A controlled candidate distribution shifts from mostly correct to polished wrong answers.
                # Metrics below are exact aggregates of these saved groups, never an unrelated decorative curve.
                wrong_count = min(3, step_index // 6)
                values = [(1.0, 0.0)] + [
                    (0.0, step_index/23) if k <= wrong_count else (1.0, k*0.1)
                    for k in range(1, 4)]
            if collapse:
                values = [(0.0, 1.0)] * 4
            for k, (correct, fmt) in enumerate(values):
                answer = a+b if correct else a+b+1
                text = f"{answer}" if k == 0 else f"<think>Let me work carefully.</think>\nFinal answer: {answer}"
                completions.append(Completion(
                    completion_id=f"cmp-{step}-{j}-{k}", group_id=gid, sequence_index=k, text=text,
                    reward_total=0.5*correct + 0.7*fmt, reward_components={"correctness": 0.5*correct, "format": 0.7*fmt},
                    inputs={"correctness_score": correct, "has_valid_final_answer": bool(fmt == 1),
                            "format_score": fmt, "completion_token_count": float(4 + k*22)},
                    grade=Grade(source="synthetic arithmetic verifier", value=correct),
                    completion_token_count=4+k*22, stop_reason="eos"))
        captured = completions[step_start:]
        reward_values, grades = [], []
        for c in captured:
            assert c.reward_total is not None and c.grade is not None
            reward_values.append(c.reward_total)
            grades.append(c.grade.value)
        for name, value in [("reward", sum(reward_values)/len(captured)),
                            ("verifier_accuracy", sum(grades)/len(captured)),
                            ("zero_variance_fraction", 0.25 if step_index > 12 else 0.0)]:
            metrics.append(Metric(generation_step=step, name=name, value=value,
                                  source="synthetic captured-group arithmetic aggregate"))
    return Snapshot(run=run, batches=batches, groups=groups, completions=completions, metrics=metrics)


def create_demo(store: Store) -> str:
    snapshot = demo_snapshot()
    store.commit(snapshot)
    return snapshot.run.run_id
