import json
from pathlib import Path

import numpy as np
import pytest

from ptm.algorithms import RECIPES, advantages

FIXTURES = Path(__file__).resolve().parents[1]/'fixtures/trl'


@pytest.mark.parametrize('scale', ['group','none','batch'])
def test_actual_trainer_golden(scale):
    data = json.loads((FIXTURES/f'{scale}.json').read_text())
    assert data['provenance']['version'] == '0.24.0'
    assert data['provenance']['dtype'] == 'torch.float32'
    assert data['provenance']['source_sha256'] == 'bb6905182acf2ec426f1dc9ffc37e5210d58ee47d2764f7cf12dbb90a4a28954'
    assert data['weights_changed'] and data['capture_errors'] == 0
    for batch in data['batches']:
        rewards, size = batch['rewards'], batch['group_size']
        groups = [rewards[i:i+size] for i in range(0,len(rewards),size)]
        actual = advantages(groups, RECIPES[f'trl-0.24.0-grpo-{scale}-v1'])
        np.testing.assert_allclose(np.array(actual).ravel(), batch['advantages'], atol=2e-6, rtol=1e-5)
