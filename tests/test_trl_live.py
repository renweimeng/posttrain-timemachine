import importlib.util
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from ptm.algorithms import RECIPES, advantages
from ptm.api import create_app


@pytest.mark.trl
def test_actual_trainer_capture_parquet_api(tmp_path):
    pytest.importorskip('trl')
    source = Path(__file__).resolve().parents[1]/'examples/quickstart-trl/train_tiny.py'
    spec = importlib.util.spec_from_file_location('ptm_tiny_training', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fixture = module.train(tmp_path/'capture', tmp_path/'store', steps=2)
    assert fixture['weights_changed'] and fixture['capture_errors'] == 0
    for batch in fixture['batches']:
        computed = advantages([batch['rewards']], RECIPES['trl-0.24.0-grpo-group-v1'])
        np.testing.assert_allclose(computed[0], batch['advantages'], atol=2e-6, rtol=1e-5)
    with TestClient(create_app(tmp_path/'store')) as client:
        run = client.get('/api/v1/runs').json()['items'][0]
        assert run['material'] == 'real captured trace' and run['completion_count'] == 8
        groups = client.get('/api/v1/runs/tiny-trl-group/groups').json()['items']
        detail = client.get('/api/v1/groups/'+groups[0]['group_id']+'?run_id=tiny-trl-group').json()
        assert len(detail['completions']) == 4 and detail['advantage_available']
