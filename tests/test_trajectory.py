import json

import pytest
from fastapi.testclient import TestClient

from ptm.api import create_app
from ptm.demo import demo_snapshot
from ptm.schemas import Trajectory
from ptm.storage import Store


@pytest.mark.parametrize('retention', ['full', 'off', 'hash_only'])
def test_recorded_tool_trajectory_import_retention_query_and_export(tmp_path, retention):
    snapshot = demo_snapshot()
    snapshot.run.schema_version = '0.2.0'
    completion = snapshot.completions[1]
    snapshot.trajectories = [Trajectory.model_validate({
        'trajectory_id': 'authored-recorded-trace', 'completion_id': completion.completion_id,
        'source': 'synthetic explicit trace fixture; tools are never executed',
        'turns': [
            {'turn_id': 'turn-user', 'role': 'user', 'content': '中文 🤖 calculate 2+2'},
            {'turn_id': 'turn-assistant', 'role': 'assistant', 'content': 'calculator(2+2)',
             'tool_name': 'calculator', 'policy_version': 'declared-policy'},
            {'turn_id': 'turn-tool', 'role': 'tool', 'content': '4', 'tool_name': 'calculator'},
            {'turn_id': 'turn-final', 'role': 'assistant', 'content': '4',
             'policy_version': 'declared-policy'},
        ],
    })]
    with TestClient(create_app(tmp_path)) as client:
        response = client.post('/api/v1/imports', json={
            'content': json.dumps(snapshot.ordered()), 'format': 'ptm-json', 'text_capture': retention})
        assert response.status_code == 201
        route = f'/api/v1/completions/{completion.completion_id}/trajectory?run_id={snapshot.run.run_id}'
        trace = client.get(route).json()['items'][0]
        assert [t['role'] for t in trace['turns']] == ['user', 'assistant', 'tool', 'assistant']
        assert trace['turns'][1]['policy_version'] == 'declared-policy'
        if retention == 'full':
            assert trace['turns'][0]['content'] == '中文 🤖 calculate 2+2'
        else:
            assert all('calculate' not in turn['content'] for turn in trace['turns'])
        stored = Store(tmp_path).load(snapshot.run.run_id)
        assert stored.trajectories[0].model_dump(mode='json') == trace
        exported = client.get(f'/api/v1/runs/{snapshot.run.run_id}/export').text
        assert ('calculate' in exported) == (retention == 'full')
