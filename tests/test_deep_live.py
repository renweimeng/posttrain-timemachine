import importlib.util
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ptm.errors import PTMError
from ptm.replay import AnalysisRequest, GroupFilter, RewardSpec, run_analysis
from ptm.reports import build_bundle, verify_bundle
from ptm.signals import replay_objective
from ptm.storage import Store, file_hash
from ptm.updates import replay_update


@pytest.mark.trl
@pytest.mark.parametrize(
    "loss,optimizer",
    [("grpo", "AdamW"), ("dapo", "AdamW"), ("bnpo", "AdamW"), ("dr_grpo", "AdamW"), ("dapo", "SGD")],
)
def test_actual_deep_capture_objective_and_real_optimizer_replay(tmp_path, loss, optimizer):
    pytest.importorskip("trl")
    source = Path(__file__).resolve().parents[1] / "examples/quickstart-trl/train_tiny.py"
    spec = importlib.util.spec_from_file_location("ptm_deep_live_reference", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = module.train(
        tmp_path / "capture",
        tmp_path / "store",
        steps=2,
        deep_capture=True,
        capture_updates=True,
        loss_type=loss,
        optimizer_kind=optimizer,
    )
    assert not output["deep_capture_errors"] and output["weights_changed"]
    store = Store(tmp_path / "store")
    snapshot = store.load(output["run_id"])
    assert snapshot.run.capture.tier == "L2" and len(snapshot.tokens) == 8
    for objective in snapshot.objectives:
        replayed = replay_objective(snapshot, objective)
        assert replayed["loss"] == pytest.approx(objective.source_loss, abs=2e-6, rel=1e-5)
    assert all(t.behavior_logprobs is None for t in snapshot.tokens)  # Never invented from a policy label.
    capture = output["update_capture_id"]
    before_file = store.path("update-captures", capture, "model.safetensors")
    original_hash = file_hash(before_file)
    replayed = replay_update(store, capture)
    assert replayed["parameters_changed"] and replayed["parameter_delta_l2"] > 0
    assert replayed["reference_match"] and replayed["max_reference_weight_error"] <= 2e-6
    assert file_hash(before_file) == original_hash
    assert replay_update(store, capture) == replayed
    # A patched branch executes a genuine second update from the same before-state, not future training prediction.
    reward = RewardSpec.model_validate(
        {
            "name": "proxy-patch",
            "inputs": ["token_proxy"],
            "terms": [{"id": "proxy", "type": "identity", "source": "token_proxy", "weight": 1.0}],
        }
    )
    branch = replay_update(store, capture, reward)
    assert branch["reference_match"] is None and branch["capture_id"] == capture
    if loss == "dapo" and optimizer == "AdamW":
        analysis = run_analysis(
            store,
            AnalysisRequest(
                run_id=snapshot.run.run_id,
                reward_spec=reward,
                group_filter=GroupFilter(group_ids=[snapshot.groups[0].group_id]),
            ),
        )
        bundle = tmp_path / "l2-report.zip"
        bundle.write_bytes(build_bundle(store, analysis["analysis_id"]))
        assert verify_bundle(bundle, recompute=True)["verified"]
        from ptm.api import create_app

        with TestClient(create_app(store.root)) as client:
            assert client.get("/api/v1/update-captures").json()["items"]
            assert client.post(
                f"/api/v1/update-captures/{capture}/replay", json={"run_id": snapshot.run.run_id}
            ).json()["reference_match"]
            assert (
                client.post(
                    f"/api/v1/update-captures/{capture}/replay", json={"run_id": "other-run"}
                ).status_code
                == 404
            )
    before_file.write_bytes(before_file.read_bytes() + b"tamper")
    with pytest.raises(PTMError, match="checksum"):
        replay_update(store, capture)
