import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ptm.api import create_app
from ptm.contracts import Origin, RewardContract
from ptm.demo import create_demo
from ptm.replay import load_spec
from ptm.storage import Store

PATCH = Path(__file__).resolve().parents[1] / "examples/format-over-truth/patch.yaml"


@pytest.fixture()
def client(tmp_path):
    create_demo(Store(tmp_path))
    with TestClient(create_app(tmp_path)) as client:
        yield client


def wait_result(client, aid):
    for _ in range(150):
        result = client.get(f"/api/v1/analyses/{aid}").json()
        if result["status"] in {"succeeded", "failed", "cancelled"}:
            return result
        time.sleep(0.02)
    pytest.fail("Job did not finish")


def test_query_evidence_and_error_semantics(client):
    run = client.get("/api/v1/runs").json()["items"][0]
    assert run["material"] == "synthetic example"
    rid = run["run_id"]
    first = client.get(f"/api/v1/runs/{rid}/groups", params={"limit": 3}).json()
    second = client.get(f"/api/v1/runs/{rid}/groups", params={"limit": 3, "cursor": first["next_cursor"]}).json()
    assert not {g["group_id"] for g in first["items"]} & {g["group_id"] for g in second["items"]}
    gid = first["items"][0]["group_id"]
    group = client.get(f"/api/v1/groups/{gid}", params={"run_id": rid}).json()
    assert group["advantage_available"] and len(group["completions"]) == 4
    missing = client.get("/api/v1/completions/unknown/tokens")
    assert missing.status_code == 422
    assert missing.json()["code"] == "PTM_CAPTURE_LEVEL_TOO_LOW"
    assert client.get(f"/api/v1/runs/{rid}/groups", params={"search": "' OR 1=1 --"}).json()["items"] == []
    anomalies = client.get(f"/api/v1/runs/{rid}/anomalies").json()["items"]
    assert any(a["id"] == "reward-accuracy-divergence" for a in anomalies)
    assert all(a["supporting_group_ids"] and a["limits"] for a in anomalies)


def test_job_idempotency_events_changes_contract_export(client):
    spec = load_spec(PATCH).model_dump()
    request = {"run_id": "demo-format-truth", "reward_spec": spec}
    response = client.post("/api/v1/analyses/reward-diff", json=request)
    assert response.status_code == 202
    aid = response.json()["analysis_id"]
    assert client.post("/api/v1/analyses/reward-diff", json=request).json()["analysis_id"] == aid
    result = wait_result(client, aid)
    assert result["status"] == "succeeded", result
    assert "succeeded" in client.get(f"/api/v1/analyses/{aid}/events").text
    assert client.get(f"/api/v1/analyses/{aid}/changes?sign_flips=true").json()["items"]
    group = client.get("/api/v1/groups/group-6400-0?run_id=demo-format-truth").json()
    fixtures = [{k: v for k, v in c.items() if k != "advantage"} for c in group["completions"][:2]]
    contract = RewardContract(id="format-regression", fixtures=fixtures,
        assertions=[{"type": "reward_order", "higher": fixtures[0]["completion_id"], "lower": fixtures[1]["completion_id"]}],
        origin=Origin(run_id="demo-format-truth", analysis_id=aid, reviewed=True))
    assert client.post("/api/v1/contracts", json=contract.model_dump(mode="json")).status_code == 201
    assert client.post("/api/v1/contracts/format-regression/test", json=spec).json()["passed"]
    assert "def test_reward_invariant" in client.post("/api/v1/contracts/format-regression/export-pytest", json=spec).text
    assert client.get(f"/api/v1/reports/{aid}/export").content.startswith(b"PK")


def test_job_failure_origin_host_and_spec_validation(client):
    spec = load_spec(PATCH).model_dump()
    request = {"run_id": "demo-format-truth", "reward_spec": spec,
               "algorithm_recipe_id": "trl-0.24.0-grpo-none-v1"}
    aid = client.post("/api/v1/analyses/reward-diff", json=request).json()["analysis_id"]
    result = wait_result(client, aid)
    assert result["status"] == "failed" and result["error"]["code"] == "PTM_RECIPE_MISMATCH"
    assert client.post("/api/v1/reward-spec/validate", json=spec, headers={"Origin": "https://evil.invalid"}).status_code == 403
    assert client.get("/api/v1/runs", headers={"host": "evil.invalid"}).status_code == 403
    assert client.post("/api/v1/reward-spec/validate", json={**spec, "code": "x"}).status_code == 422


def test_chunked_body_size_limit(client):
    response = client.post('/api/v1/reward-spec/validate',
        content=iter([b'x'*(1024*1024)]*9),headers={'content-type':'application/json'})
    assert response.status_code == 413
    assert response.json()['code'] == 'PTM_IMPORT_LIMIT'
