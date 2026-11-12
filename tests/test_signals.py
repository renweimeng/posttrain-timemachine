import math

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from ptm.api import create_app
from ptm.demo import demo_snapshot
from ptm.errors import PTMError
from ptm.schemas import ObjectiveBatch, Snapshot, TokenBlob
from ptm.signals import replay_objective, token_signals
from ptm.storage import Store


def l2_snapshot():
    s = demo_snapshot()
    s.run.schema_version = "0.2.0"
    s.run.capture.tier = "L2"
    s.run.tokenizer_hash = "fixture-tokenizer"
    s.run.expected_total_groups = 1
    s.groups = s.groups[:1]
    s.batches = s.batches[:1]
    s.completions = s.completions[:4]
    s.metrics = []
    masks = [[1, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0]]
    for i, c in enumerate(s.completions):
        s.tokens.append(
            TokenBlob(
                token_blob_id=f"tok-{i}",
                completion_id=c.completion_id,
                tokenizer_hash="fixture-tokenizer",
                optimizer_step=1,
                token_ids=[10, 11, 12],
                token_pieces=["中文", "🤖", "<EOS>"],
                loss_mask=masks[i],
                old_policy_logprobs=[-2, -1, -3],
                current_policy_logprobs=[-1.8, -1.3, -3],
                reference_logprobs=[-1.7, -1.2, -3],
                old_policy_version="old",
                current_policy_version="current",
                reference_policy_version="ref",
                observed_advantage=[1, 2, -1, -2][i],
                advantage_recipe_id=s.run.recipe_id,
                source="synthetic numerical fixture",
            )
        )
    s.objectives = [
        ObjectiveBatch(
            objective_id="obj-fixture",
            run_id=s.run.run_id,
            optimizer_step=1,
            token_blob_ids=[t.token_blob_id for t in s.tokens],
            expected_completion_count=4,
            declared_complete=True,
            policy_version="current",
            old_policy_version="old",
            reference_policy_version="ref",
            global_effective_tokens=6,
            max_completion_length=3,
            kl_coefficient=0.1,
            source="synthetic numerical fixture",
        )
    ]
    return Snapshot.model_validate(s.model_dump())


def independent_loss(s, kind):
    totals = []
    for t in s.tokens:
        total = 0
        for current, old, ref, mask in zip(
            t.current_policy_logprobs, t.old_policy_logprobs, t.reference_logprobs, t.loss_mask
        ):
            ratio = math.exp(current - old)
            contribution = min(ratio * t.observed_advantage, min(max(ratio, 0.8), 1.2) * t.observed_advantage)
            total += (-contribution + 0.1 * (math.exp(ref - current) - (ref - current) - 1)) * mask
        totals.append((total, sum(t.loss_mask)))
    if kind == "grpo":
        return sum(v / max(n, 1) for v, n in totals) / 4
    if kind == "dr_grpo":
        return sum(v for v, n in totals) / 12
    return sum(v for v, n in totals) / 6


@pytest.mark.parametrize("kind", ["grpo", "dapo", "dr_grpo", "bnpo"])
def test_objective_against_independent_scalar_reference(kind):
    s = l2_snapshot()
    o = s.objectives[0]
    o.loss_type = kind
    assert replay_objective(s, o)["loss"] == pytest.approx(independent_loss(s, kind), abs=1e-12)


def test_token_mask_and_missing_inputs_are_explicit():
    t = l2_snapshot().tokens[0]
    got = token_signals(t)
    assert got["rows"][0]["ratio"] == pytest.approx(math.exp(0.2))
    assert got["rows"][2]["ratio"] is None
    assert got["rows"][0]["sampled_kl"] == pytest.approx(math.expm1(0.1) - 0.1)
    t.old_policy_logprobs = None
    got = token_signals(t)
    assert got["missing"]["ratio"] == ["old_policy_logprobs"]
    assert got["rows"][0]["ratio"] is None and got["rows"][0]["sampled_kl"] is not None
    t.token_pieces.pop()
    with pytest.raises(ValidationError, match="align"):
        TokenBlob.model_validate(t.model_dump())


@pytest.mark.parametrize("failure", ["partial", "mask_denominator", "policy", "rank_grid", "missing_old"])
def test_objective_rejects_insufficient_capture(failure):
    s = l2_snapshot()
    o = s.objectives[0]
    if failure == "partial":
        o.declared_complete = False
    elif failure == "mask_denominator":
        o.global_effective_tokens = 999
    elif failure == "policy":
        s.tokens[0].current_policy_version = "another"
    elif failure == "rank_grid":
        o.num_processes = 2
    else:
        s.tokens[0].old_policy_logprobs = None
    with pytest.raises(PTMError):
        replay_objective(s, o)


def test_distributed_microbatch_bnpo_uses_local_means():
    s = l2_snapshot()
    o = s.objectives[0]
    o.loss_type = "bnpo"
    o.num_processes = 2
    o.gradient_accumulation_steps = 2
    for i, t in enumerate(s.tokens):
        t.worker_rank = i // 2
        t.microbatch_index = i % 2
    # One sequence per local microbatch: BNPO becomes the average of sequence token means.
    assert replay_objective(s, o)["loss"] == pytest.approx(independent_loss(s, "grpo"))


def test_token_api_roundtrip_and_legacy_snapshot_hash(tmp_path):
    store = Store(tmp_path)
    legacy = demo_snapshot()
    old = store.commit(legacy)
    assert store.load(legacy.run.run_id).ordered() == legacy.ordered()
    assert store.commit(legacy)["snapshot_id"] == old["snapshot_id"]
    s = l2_snapshot()
    store.commit(s)
    assert store.load(s.run.run_id).ordered() == s.ordered()
    assert store.load(s.run.run_id, include_advanced=False).tokens == []
    with TestClient(create_app(tmp_path)) as client:
        result = client.get(
            f"/api/v1/completions/{s.completions[0].completion_id}/tokens?run_id={s.run.run_id}"
        ).json()
        assert result["selected"]["token_pieces"][1] == "🤖"
        assert result["signals"]["rows"][2]["ratio"] is None
        replay = client.post("/api/v1/objectives/obj-fixture/replay", json={"run_id": s.run.run_id})
        assert replay.status_code == 200, replay.text
        assert replay.json()["evidence"] == "recomputed"


def test_token_lens_uses_captured_objective_clipping_and_labels_fallback(tmp_path):
    from ptm.query import Repository

    snapshot = l2_snapshot()
    snapshot.objectives[0].epsilon_high = 0.05
    store = Store(tmp_path)
    store.commit(snapshot)
    token = snapshot.tokens[0]
    response = Repository(store).tokens(token.completion_id, snapshot.run.run_id)
    assert response["clipping"]["source"] == "captured objective"
    assert response["clipping"]["epsilon_high"] == 0.05
    assert response["signals"]["rows"][0]["surrogate"] == pytest.approx(1.05)
    snapshot.objectives = []
    store.commit(snapshot)
    fallback = Repository(store).tokens(token.completion_id, snapshot.run.run_id)
    assert fallback["clipping"]["objective_id"] is None
    assert "display calculator default" in fallback["clipping"]["source"]
    assert fallback["signals"]["rows"][0]["surrogate"] == pytest.approx(1.2)
