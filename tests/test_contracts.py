from pathlib import Path

import pytest

from ptm.contracts import Origin, RewardContract, export_pytest, junit, mutate, test_contract
from ptm.demo import demo_snapshot
from ptm.errors import PTMError
from ptm.replay import load_spec

EXAMPLES = Path(__file__).resolve().parents[1] / "examples/format-over-truth"


def example_contract():
    cs = demo_snapshot().completions[:2]
    return RewardContract(id="format-cannot-beat-truth", fixtures=cs,
                          assertions=[{"type": "min_margin", "higher": cs[0].completion_id,
                                       "lower": cs[1].completion_id, "margin": 0.2}],
                          origin=Origin(run_id="demo-format-truth", reviewed=True))


def test_known_bad_reward_fails_patch_passes(tmp_path):
    contract = example_contract()
    assert not test_contract(contract, load_spec(EXAMPLES/"original.yaml"))["passed"]
    result = test_contract(contract, load_spec(EXAMPLES/"patch.yaml"))
    assert result["passed"]
    assert b'failures="0"' in junit([result])
    output = tmp_path / "test_exported.py"
    export_pytest([contract], load_spec(EXAMPLES/"patch.yaml"), output)
    namespace = {}
    exec(compile(output.read_text(), str(output), "exec"), namespace)
    namespace["test_reward_invariant"](contract.model_dump(mode="json"))
    contract.origin.reviewed = False
    with pytest.raises(PTMError, match="review"):
        test_contract(contract, load_spec(EXAMPLES/"patch.yaml"))


@pytest.mark.parametrize("kind", ["irrelevant_padding", "equivalent_format", "repeated_answer", "correctness_change"])
def test_mutations_never_invent_verifier_facts(kind):
    original = demo_snapshot().completions[0]
    variant = mutate(original, kind)
    assert variant.grade is None and variant.completion_token_count is None
    assert variant.reward_status == "missing" and variant.text != original.text
