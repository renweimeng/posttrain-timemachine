
import pytest

from ptm.capture import CaptureContext, SpoolWriter, TRLRewardCapture
from ptm.demo import demo_snapshot
from ptm.importer import read_snapshot
from ptm.storage import Store, doctor


def capture(tmp_path, monkeypatch, **kwargs):
    monkeypatch.setattr('ptm.capture.importlib.metadata.version', lambda name: '0.24.0')
    monkeypatch.delenv('WORLD_SIZE', raising=False)
    writer = SpoolWriter(tmp_path/'rank-0/events.jsonl', demo_snapshot().run, **kwargs)
    adapter = TRLRewardCapture(writer, lambda kwargs: CaptureContext(
        generation_batch_id='batch-explicit', generation_step=7, policy_version='policy-x'), group_size=2)
    return writer, adapter


def test_explicit_interleaved_groups_and_serialization_failure_preserves_rewards(tmp_path, monkeypatch):
    writer, adapter = capture(tmp_path, monkeypatch)
    values = [1.0, 0.5, 0.0, None]
    wrapped = adapter.wrap(lambda **kwargs: values)
    kwargs = dict(prompts=['a','b','a','b'], completions=['中文 🤖','bad','good','missing'],
                  completion_ids=[[1,2],[3],[4],[5]], ptm_group_id=['explicit-a','explicit-b','explicit-a','explicit-b'])
    assert wrapped(**kwargs) is values
    writer.close()
    snapshot = read_snapshot(writer.path)
    assert len(snapshot.groups) == 2
    assert all(g.expected_group_size == 2 for g in snapshot.groups)
    assert sorted(c.sequence_index for c in snapshot.completions) == [0,0,1,1]
    assert any(c.reward_status == 'missing' for c in snapshot.completions)
    assert any(c.text == '中文 🤖' for c in snapshot.completions)
    failing = adapter.wrap(lambda **kwargs: values, fields=lambda i,k: {'inputs': {'bad': object()}})
    assert failing(**kwargs) is values
    assert adapter.capture_errors == 1
    assert writer.capture_errors == 1


def test_missing_group_id_never_infers_groups(tmp_path, monkeypatch):
    writer, adapter = capture(tmp_path, monkeypatch)
    values = [1,2]
    wrapped = adapter.wrap(lambda **kwargs: values)
    assert wrapped(prompts=['same']*2, completions=['a','b']) is values
    writer.close()
    assert adapter.capture_errors == 1
    assert read_snapshot(writer.path).groups == []


def test_bounded_backpressure_drops_whole_groups(tmp_path, monkeypatch):
    writer, adapter = capture(tmp_path, monkeypatch, max_batch_bytes=1)
    wrapped = adapter.wrap(lambda **kwargs: [1,0])
    assert wrapped(prompts=['same']*2, completions=['a','b'], ptm_group_id=['g','g']) == [1,0]
    writer.compact(Store(tmp_path/'store'))
    assert writer.dropped_groups == 1 and writer.dropped_batches == 1
    assert read_snapshot(writer.path).groups == []


def test_unsupported_adapter_version_fails_closed(tmp_path, monkeypatch):
    writer, adapter = capture(tmp_path, monkeypatch)
    monkeypatch.setattr('ptm.capture.importlib.metadata.version', lambda name: '999')
    with pytest.raises(Exception, match='0.24.0'):
        TRLRewardCapture(writer, adapter.context, group_size=2)
    writer.close()


def test_capture_failure_makes_population_coverage_unknown(tmp_path, monkeypatch):
    writer, adapter = capture(tmp_path, monkeypatch)
    wrapped = adapter.wrap(lambda **kwargs: [1,0])
    assert wrapped(prompts=['a','a'], completions=['a','b']) == [1,0]
    assert wrapped(prompts=['a','a'], completions=['a','b'],ptm_group_id=['q','q']) == [1,0]
    store = Store(tmp_path/'data')
    writer.compact(store)
    health = doctor(store.load('demo-format-truth'))
    assert health['capture_errors'] == 1
    assert health['coverage']['expected_groups'] is None
    assert health['coverage']['fraction'] is None
