import json
from pathlib import Path

from fastapi.testclient import TestClient

from ptm.api import create_app
from ptm.demo import create_demo, demo_snapshot
from ptm.replay import AnalysisRequest, load_spec, run_analysis
from ptm.storage import Store


def test_import_export_history_and_restart(tmp_path):
    store = Store(tmp_path)
    snapshot = demo_snapshot()
    spec = load_spec(Path(__file__).resolve().parents[1] / "examples/format-over-truth/patch.yaml")
    with TestClient(create_app(tmp_path)) as client:
        imported = client.post(
            "/api/v1/imports",
            json={
                "content": json.dumps(snapshot.ordered()),
                "format": "ptm-json",
                "text_capture": "hash_only",
            },
        )
        assert imported.status_code == 201, imported.text
        assert store.load(snapshot.run.run_id).completions[0].text is None
        exported = client.get("/api/v1/runs/demo-format-truth/export")
        assert exported.status_code == 200 and "event_id" in exported.text
        result = run_analysis(store, AnalysisRequest(run_id=snapshot.run.run_id, reward_spec=spec))
        assert client.get("/api/v1/analyses").json()["items"][0]["analysis_id"] == result["analysis_id"]
    with TestClient(create_app(tmp_path)) as client:
        assert client.get("/api/v1/reports").json()["items"][0]["analysis_id"] == result["analysis_id"]
        assert client.get("/api/v1/reports/" + result["analysis_id"] + "/export").content.startswith(b"PK")
        bad = client.post("/api/v1/imports", json={"content": "{}", "format": "ptm-json"})
        assert bad.status_code == 422


def test_contract_history_is_persistent_and_exportable(tmp_path):
    from ptm.contracts import Origin, RewardContract, save_contract

    store = Store(tmp_path)
    create_demo(store)
    cs = store.load("demo-format-truth").completions[:2]
    contract = RewardContract(
        id="saved-contract",
        fixtures=cs,
        assertions=[{"type": "reward_order", "higher": cs[0].completion_id, "lower": cs[1].completion_id}],
        origin=Origin(run_id="demo-format-truth", reviewed=True),
    )
    save_contract(store, contract)
    spec = load_spec(Path(__file__).resolve().parents[1] / "examples/format-over-truth/patch.yaml")
    with TestClient(create_app(tmp_path)) as client:
        first = client.post("/api/v1/contracts/saved-contract/test", json=spec.model_dump()).json()
        second = client.post("/api/v1/contracts/saved-contract/test", json=spec.model_dump()).json()
        assert first == second and first["passed"]
    with TestClient(create_app(tmp_path)) as client:
        assert len(client.get("/api/v1/contract-results").json()["items"]) == 1
        assert b"testsuite" in client.get("/api/v1/contract-results/export?format=junit").content


def test_contract_preview_resets_file_review_and_malformed_rank_is_typed(tmp_path):
    import json

    from fastapi.testclient import TestClient

    from ptm.api import create_app
    from ptm.contracts import RewardContract
    from ptm.demo import demo_snapshot

    snapshot = demo_snapshot()
    contract = RewardContract(id="import-review", fixtures=snapshot.completions[:2],
                              assertions=[{"type": "reward_order", "higher": snapshot.completions[0].completion_id,
                                           "lower": snapshot.completions[1].completion_id}],
                              origin={"run_id": snapshot.run.run_id, "reviewed": True})
    with TestClient(create_app(tmp_path)) as client:
        result = client.post("/api/v1/contracts/preview", json={"content": contract.model_dump_json()})
        assert result.status_code == 200 and not result.json()["origin"]["reviewed"]
        assert client.post("/api/v1/contracts", json=result.json()).status_code == 422
        assert client.post("/api/v1/contracts/preview", json={"content": "not: [valid"}).status_code == 422
        for value in ["not json", "[]", json.dumps({"world_size": 2, "shards": [3]})]:
            response = client.post("/api/v1/imports", json={"content": value, "format": "rank-jsonl"})
            assert response.status_code == 422 and response.json()["code"] == "PTM_IMPORT_INVALID"
        capability = client.get("/api/v1/system/capabilities").json()
        assert isinstance(capability["update_replay_available"], bool)
