from pathlib import Path

import pytest
from pydantic import ValidationError

from ptm.demo import demo_snapshot
from ptm.errors import PTMError
from ptm.replay import AnalysisRequest, GroupFilter, RewardSpec, compute_diff, load_spec, run_analysis
from ptm.storage import Store, digest

PATCH = Path(__file__).resolve().parents[1] / "examples/format-over-truth/patch.yaml"


def test_deterministic_diff_and_no_source_mutation(tmp_path):
    snapshot = demo_snapshot()
    before = digest(snapshot.ordered())
    store = Store(tmp_path)
    store.commit(snapshot)
    request = AnalysisRequest(run_id=snapshot.run.run_id, reward_spec=load_spec(PATCH))
    a, b = run_analysis(store, request), run_analysis(store, request)
    assert a == b
    assert a["summary"]["rank_reversals"] > 0
    assert a["summary"]["sign_flips"] > 0
    assert digest(store.load(snapshot.run.run_id).ordered()) == before
    assert a["evidence"]["future_outcome"] == "unavailable"


@pytest.mark.parametrize("problem,code", [
    ("partial", "PTM_INCOMPLETE_GROUP"), ("input", "PTM_REWARD_INPUT_MISSING"),
    ("recipe", "PTM_RECIPE_MISMATCH"), ("invalid", "PTM_INVALID_REWARD"),
])
def test_fail_closed(problem, code):
    snapshot = demo_snapshot()
    request = AnalysisRequest(run_id=snapshot.run.run_id, reward_spec=load_spec(PATCH))
    if problem == "partial":
        snapshot.completions.pop()
    elif problem == "input":
        snapshot.completions[0].inputs.clear()
    elif problem == "recipe":
        request.algorithm_recipe_id = "trl-0.24.0-grpo-none-v1"
    else:
        snapshot.completions[0].reward_status = "missing"
        snapshot.completions[0].reward_total = None
    with pytest.raises(PTMError) as exc:
        compute_diff(snapshot, request)
    assert exc.value.code == code


def test_closed_ast_no_code_execution():
    data = load_spec(PATCH).model_dump()
    data["terms"][0]["type"] = "python"
    with pytest.raises(ValidationError):
        RewardSpec.model_validate(data)
    data = load_spec(PATCH).model_dump()
    data["code"] = "__import__('os').system('false')"
    with pytest.raises(ValidationError):
        RewardSpec.model_validate(data)


def test_group_filter():
    snapshot = demo_snapshot()
    request = AnalysisRequest(run_id=snapshot.run.run_id, reward_spec=load_spec(PATCH),
                              group_filter=GroupFilter(group_ids=[snapshot.groups[0].group_id]))
    assert compute_diff(snapshot, request)["summary"]["groups"] == 1


def batch_case():
    snapshot = demo_snapshot()
    snapshot.groups = snapshot.groups[:2]
    ids = {g.group_id for g in snapshot.groups}
    snapshot.completions = [c for c in snapshot.completions if c.group_id in ids]
    snapshot.run.recipe_id = 'trl-0.24.0-grpo-batch-v1'
    snapshot.run.config['reward_batch_scope'] = 'full_snapshot'
    snapshot.run.expected_total_groups = 2
    for c in snapshot.completions:
        c.inputs['after'] = c.reward_total*(2 if c.group_id == snapshot.groups[0].group_id else 1)
    spec = RewardSpec(name='batch-change',inputs=['after'],terms=[
        {'id':'score','type':'identity','source':'after','weight':1}])
    return snapshot, AnalysisRequest(run_id=snapshot.run.run_id,reward_spec=spec,
                                     algorithm_recipe_id=snapshot.run.recipe_id)


def test_advantage_only_change_is_affected_and_variance_is_labeled():
    snapshot, request = batch_case()
    result = compute_diff(snapshot, request)
    unchanged_rewards = [c for c in result['changes'] if c['group_id'] == snapshot.groups[1].group_id]
    assert all(c['delta_reward'] == 0 and c['rank_before'] == c['rank_after'] for c in unchanged_rewards)
    assert all(c['advantage_before'] != c['advantage_after'] and c['affected'] for c in unchanged_rewards)
    assert all(g['variance_ddof'] == 0 for g in result['groups'])


@pytest.mark.parametrize('problem',['missing_whole_group','drop','capture_error','group_sampling'])
def test_batch_denominator_requires_verified_full_capture(problem):
    snapshot, request = batch_case()
    if problem == 'missing_whole_group':
        snapshot.run.expected_total_groups = 3
    elif problem == 'drop':
        snapshot.run.drop_count = 1
    elif problem == 'capture_error':
        snapshot.run.config['capture_errors'] = 1
    else:
        snapshot.groups[0].capture.sampling_probability = .5
    with pytest.raises(PTMError) as exc:
        compute_diff(snapshot, request)
    assert exc.value.code == 'PTM_BATCH_SCOPE_UNAVAILABLE'


def test_native_trl_version_mismatch_is_not_silently_replayed():
    snapshot = demo_snapshot()
    snapshot.run.framework = 'trl'
    snapshot.run.framework_version = '9.9'
    request = AnalysisRequest(run_id=snapshot.run.run_id,reward_spec=load_spec(PATCH))
    with pytest.raises(PTMError) as exc:
        compute_diff(snapshot,request)
    assert exc.value.code == 'PTM_RECIPE_MISMATCH'


def test_component_only_changes_are_affected():
    snapshot = demo_snapshot()
    spec = load_spec(PATCH.parent/'original.yaml').model_dump()
    spec['terms'][0]['id'] = 'renamed_correctness'
    result = compute_diff(snapshot,AnalysisRequest(run_id=snapshot.run.run_id,
                          reward_spec=RewardSpec.model_validate(spec)))
    assert all(c['delta_reward'] == 0 for c in result['changes'])
    assert all(c['affected'] for c in result['changes'])
