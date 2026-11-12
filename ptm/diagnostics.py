from typing import Any

import numpy as np

from ptm.schemas import Snapshot
from ptm.storage import doctor


def detect(snapshot: Snapshot) -> list[dict[str, Any]]:
    """Explanatory signals, never automatic proof of reward hacking or causality."""
    groups = {g.group_id: g for g in snapshot.groups}
    by_group: dict[str, list[Any]] = {}
    for c in snapshot.completions:
        by_group.setdefault(c.group_id, []).append(c)
    coverage = doctor(snapshot)["coverage"]
    results = []

    def add(kind: str, title: str, reason: str, ids: list[str], limits: str) -> None:
        if ids:
            results.append({"id": kind, "title": title, "severity": "warning", "why_flagged": reason,
                            "supporting_group_ids": sorted(set(ids)), "coverage": coverage,
                            "limits": limits, "detector_version": "1.0.0"})

    complete = {gid: cs for gid, cs in by_group.items() if len(cs) == groups[gid].expected_group_size and
                all(c.reward_status == "valid" for c in cs)}
    collapsed = [gid for gid, cs in complete.items() if len({c.reward_total for c in cs}) == 1]
    add("variance-collapse", "Zero-signal groups", "All captured rewards are equal within each listed complete group.",
        collapsed, "Zero relative advantage need not imply zero full loss gradient.")
    mismatch, dominant, length = [], [], []
    for gid, cs in complete.items():
        for c in cs:
            if c.grade is not None and c.grade.value == 0 and any(
                d.grade is not None and d.grade.value == 1 and d.grade.source == c.grade.source and
                c.reward_total > d.reward_total for d in cs):
                mismatch.append(gid)
            parts = c.reward_components
            total_abs = sum(abs(v) for v in parts.values() if v is not None)
            if total_abs > 0 and abs(parts.get("format") or 0)/total_abs > 0.8:
                dominant.append(gid)
        long = [c for c in cs if c.completion_token_count is not None]
        if (len(long) >= 3 and len({c.completion_token_count for c in long}) > 1 and
                len({c.reward_total for c in long}) > 1):
            corr = np.corrcoef([c.completion_token_count for c in long], [c.reward_total for c in long])[0, 1]
            if np.isfinite(corr) and corr > 0.8:
                length.append(gid)
    add("reward-truth-mismatch", "Format over truth", "Verifier-failed completion outranks a verifier-passed peer.",
        mismatch, "Depends on the supplied independent verifier; not proof of learned exploitation.")
    add("component-dominance", "Format component dominates", "Format supplies >80% of absolute component contribution.",
        dominant, "Contribution magnitude does not establish causality.")
    add("length-drift", "Reward correlates with length", "Within-group Pearson correlation exceeds 0.8.",
        length, "Long answers can be valid. Captured-group statistic only.")
    metrics: dict[str, dict[int, float]] = {}
    for m in snapshot.metrics:
        metrics.setdefault(m.name, {})[m.generation_step] = m.value
    accuracy_key = "heldout_accuracy" if "heldout_accuracy" in metrics else "verifier_accuracy"
    accuracy_label = "held-out accuracy" if accuracy_key == "heldout_accuracy" else "saved-sample verifier accuracy"
    reward, heldout = metrics.get("reward", {}), metrics.get(accuracy_key, {})
    aligned = sorted(set(reward) & set(heldout))
    if len(aligned) >= 2 and reward[aligned[-1]] > reward[aligned[0]] and heldout[aligned[-1]] < heldout[aligned[0]]:
        batches = {b.generation_batch_id: b for b in snapshot.batches}
        ids = [gid for gid in groups if aligned[0] <= batches[groups[gid].generation_batch_id].generation_step <= aligned[-1]]
        add("reward-accuracy-divergence", f"Reward ↑ / {accuracy_label} ↓",
            f"Explicit generation-step alignment: reward Δ={reward[aligned[-1]]-reward[aligned[0]]:.3f}; "
            f"{accuracy_label} Δ={heldout[aligned[-1]]-heldout[aligned[0]]:.3f}.", ids,
            "Temporal association only. Metric source and evaluator independence are supplied by the capture. " +
            ("Saved-cohort verifier correctness; no independent held-out split." if accuracy_key == "verifier_accuracy" else
             "Held-out split must be established by the metric producer."))
    priority = {"reward-accuracy-divergence": 0, "reward-truth-mismatch": 1, "variance-collapse": 2}
    return sorted(results, key=lambda r: priority.get(r["id"], 3))
