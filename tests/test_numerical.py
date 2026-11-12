import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from ptm.algorithms import RECIPES, advantages, population_variance, ranks
from ptm.errors import PTMError

GROUP = RECIPES["trl-0.24.0-grpo-group-v1"]


def independent_reference(values):
    # Scalar reference deliberately avoids the implementation's NumPy reductions.
    import math
    mean = sum(values) / len(values)
    std = math.sqrt(sum((v-mean)**2 for v in values) / (len(values)-1))
    return [(v-mean)/(std+0.0001) for v in values]


@given(st.lists(st.floats(-10000, 10000, allow_nan=False, allow_infinity=False), min_size=2, max_size=32))
def test_independent_reference_and_permutation(values):
    got = advantages([values], GROUP)[0]
    np.testing.assert_allclose(got, independent_reference(values), atol=1e-11, rtol=1e-10)
    np.testing.assert_allclose(advantages([values[::-1]], GROUP)[0], got[::-1], atol=1e-11)


def test_constant_shift_scaling_ties_singleton_overflow():
    values = [0, 1, 0.5, 0.7]
    a = advantages([values], GROUP)[0]
    np.testing.assert_allclose(advantages([[v+20 for v in values]], GROUP)[0], a, atol=1e-12)
    # Epsilon breaks exact scale invariance; use the derived denominator correction.
    std = np.std(values, ddof=1)
    scaled = advantages([[v*3 for v in values]], GROUP)[0]
    np.testing.assert_allclose(scaled, np.array(a)*3*(std+1e-4)/(3*std+1e-4))
    assert advantages([[5, 5, 5]], GROUP) == [[0, 0, 0]]
    assert ranks([2, 2, 0, 1]) == [1, 1, 4, 3]
    with pytest.raises(PTMError, match="G >= 2"):
        advantages([[1]], GROUP)
    for values in [[float("nan"), 1], [float("inf"), 1], [1e308, -1e308]]:
        with pytest.raises(PTMError):
            advantages([values], GROUP)


def test_none_and_batch():
    assert advantages([[2]], RECIPES["trl-0.24.0-grpo-none-v1"]) == [[0]]
    values = [[0, 2], [1, 5]]
    got = advantages(values, RECIPES["trl-0.24.0-grpo-batch-v1"])
    np.testing.assert_allclose(got, [[-1, 1], [-2, 2]] / (np.std([0, 2, 1, 5], ddof=1)+1e-4))
    with pytest.raises(PTMError, match='variance'):
        population_variance([-1e200,1e200])
