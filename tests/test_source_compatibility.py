from fastapi.testclient import TestClient

from ptm.api import create_app
from ptm.demo import demo_snapshot
from ptm.storage import Store, doctor


def test_native_version_mismatch_disables_all_advantage_read_paths(tmp_path):
    snapshot = demo_snapshot()
    snapshot.run.framework = 'trl'
    snapshot.run.framework_version = '9.9'
    Store(tmp_path).commit(snapshot)
    assert not doctor(snapshot)['capabilities']['advantage']
    with TestClient(create_app(tmp_path)) as client:
        summary = client.get('/api/v1/runs/demo-format-truth/summary').json()
        assert not summary['compatibility']['reward_advantage']
        group = client.get('/api/v1/groups/group-6400-0?run_id=demo-format-truth').json()
        assert not group['advantage_available']
        assert group['unavailable']['code'] == 'PTM_RECIPE_MISMATCH'
        assert all(c['advantage'] is None for c in group['completions'])
