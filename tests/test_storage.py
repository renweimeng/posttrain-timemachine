import json

import pytest
from pydantic import ValidationError

from ptm.demo import demo_snapshot
from ptm.errors import PTMError
from ptm.schemas import Snapshot
from ptm.storage import Store, doctor, file_hash


def test_snapshot_roundtrip_and_immutable(tmp_path):
    store = Store(tmp_path)
    snapshot = demo_snapshot()
    first = store.commit(snapshot)
    path = store.snapshot_path(snapshot.run.run_id)
    before = {p.name: file_hash(p) for p in path.iterdir()}
    assert store.commit(snapshot) == first
    assert store.load(snapshot.run.run_id).ordered() == snapshot.ordered()
    assert before == {p.name: file_hash(p) for p in path.iterdir()}
    assert doctor(snapshot)["complete_groups"] == 96


def test_synthetic_metrics_reconcile_to_clickable_group_evidence():
    snapshot = demo_snapshot()
    batches = {b.generation_batch_id: b.generation_step for b in snapshot.batches}
    groups = {g.group_id: batches[g.generation_batch_id] for g in snapshot.groups}
    for metric in snapshot.metrics:
        cs = [c for c in snapshot.completions if groups[c.group_id] == metric.generation_step]
        if metric.name == 'reward':
            assert metric.value == sum(c.reward_total for c in cs)/len(cs)
        elif metric.name == 'verifier_accuracy':
            assert metric.value == sum(c.grade.value for c in cs)/len(cs)


def test_missing_zero_nan_and_foreign_keys():
    data = demo_snapshot().model_dump(mode="json")
    data["completions"][0]["reward_total"] = 0.0
    assert Snapshot.model_validate(data).completions[0].reward_total == 0
    for invalid in [None, float("nan"), float("inf")]:
        data["completions"][0]["reward_total"] = invalid
        with pytest.raises(ValidationError):
            Snapshot.model_validate(data)
    data["completions"][0]["reward_total"] = 1
    data["completions"][0]["group_id"] = "unknown"
    with pytest.raises(ValidationError, match="foreign key"):
        Snapshot.model_validate(data)


def test_missing_recipe_and_ambiguous_metric_sources_fail_closed():
    data = demo_snapshot().ordered()
    del data['run']['recipe_id']
    with pytest.raises(ValidationError, match='recipe_id'):
        Snapshot.model_validate(data)
    data = demo_snapshot().ordered()
    data['metrics'].append(dict(data['metrics'][0],source='other evaluator',value=.123))
    with pytest.raises(ValidationError, match='duplicate metric'):
        Snapshot.model_validate(data)
    data = demo_snapshot().ordered()
    name = data['metrics'][0]['name']
    next(m for m in data['metrics'][1:] if m['name']==name)['source']='other evaluator'
    with pytest.raises(ValidationError, match='source changes'):
        Snapshot.model_validate(data)


def test_partial_group_is_visible(tmp_path):
    snapshot = demo_snapshot()
    snapshot.completions.pop()
    store = Store(tmp_path)
    store.commit(snapshot)
    result = doctor(store.load(snapshot.run.run_id))
    assert len(result["incomplete_groups"]) == 1
    assert not result["capabilities"]["advantage"]


def test_integrity_and_path_boundary(tmp_path):
    store = Store(tmp_path / "data")
    store.commit(demo_snapshot())
    with pytest.raises(PTMError, match="identifier"):
        store.path("..", "secret")
    (store.root / "escape").symlink_to(tmp_path)
    with pytest.raises(PTMError, match="Symlink"):
        store.path("escape", "outside")
    path = store.snapshot_path("demo-format-truth") / "run.json"
    data = json.loads(path.read_bytes())
    data["name"] = "tampered"
    path.write_text(json.dumps(data))
    with pytest.raises(PTMError, match="Checksum"):
        store.load("demo-format-truth")
