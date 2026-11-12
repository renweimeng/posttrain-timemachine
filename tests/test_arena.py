import pytest
from fastapi.testclient import TestClient

from ptm.api import create_app
from ptm.arena import challenge, evaluate_challenge, load_challenge
from ptm.replay import RewardSpec
from ptm.storage import Store


@pytest.mark.parametrize("name", ["format-over-truth", "zero-signal", "verbose-wins", "checker-blind-spot"])
def test_every_arena_has_real_failure_patch_and_reviewable_contract(name, tmp_path):
    data = challenge(name)
    assert not data["contract_candidate"]["origin"]["reviewed"]
    before = evaluate_challenge(name, RewardSpec.model_validate(data["original_spec"]))
    after = evaluate_challenge(name, RewardSpec.model_validate(data["patch_spec"]))
    assert not before["passed"] and after["passed"]
    assert before["new_regressions"] == 0 and before["guard_failures"] > 0
    assert after["new_regressions"] == 0 and after["guard_failures"] == 0
    assert after["known"]["passed"] == after["known"]["total"]
    assert after["guards"]["passed"] == after["guards"]["total"]
    store = Store(tmp_path)
    loaded = load_challenge(store, name)
    snapshot = store.load(loaded["run_id"])
    assert len(snapshot.completions) == 8
    assert len(snapshot.groups) == 2
    for group in snapshot.groups:
        truth = "7" if group.group_id.endswith("-guard") else "4"
        members = [c for c in snapshot.completions if c.group_id == group.group_id]
        assert len(members) == group.expected_group_size == 4
        assert [c.sequence_index for c in members] == list(range(4))
        assert truth in members[0].text
    with TestClient(create_app(tmp_path)) as client:
        result = client.post(f"/api/v1/arena/{name}/evaluate", json=data["patch_spec"])
        assert result.status_code == 200 and result.json()["passed"]
        saved = client.post("/api/v1/contracts", json=data["contract_candidate"])
        assert saved.status_code == 422 and saved.json()["code"] == "PTM_CONTRACT_UNREVIEWED"
