"""Persistent local workspace operations shared by API and CLI."""

import json
import math
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from ptm.contracts import RewardContract, junit, load_contract, test_contract
from ptm.errors import PTMError
from ptm.importer import export_events, read_snapshot, redact_snapshot
from ptm.replay import RewardSpec, load_analysis
from ptm.schemas import Model
from ptm.storage import Store, atomic_write, canonical, digest


class ImportRequest(Model):
    content: str
    format: Literal["ptm-json", "ptm-jsonl", "verl-jsonl", "rank-jsonl"] = "ptm-json"
    text_capture: Literal["full", "off", "hash_only"] = "full"


def import_content(store: Store, request: ImportRequest) -> dict[str, Any]:
    directory = store.path("imports")
    directory.mkdir(exist_ok=True)
    suffix = ".json" if request.format == "ptm-json" else ".jsonl"
    with tempfile.NamedTemporaryFile(dir=directory, suffix=suffix, delete=False) as stream:
        path = Path(stream.name)
        stream.write(request.content.encode())
    try:
        if request.format == "rank-jsonl":
            from ptm.interop import rank_import_content

            snapshot = rank_import_content(request.content)
        elif request.format == "verl-jsonl":
            from ptm.interop import read_verl

            snapshot = read_verl(path, request.text_capture)
        else:
            snapshot = read_snapshot(path, request.text_capture)
        return store.commit(redact_snapshot(snapshot, request.text_capture))
    finally:
        path.unlink(missing_ok=True)


def export_run(store: Store, run_id: str) -> bytes:
    directory = store.path("exports")
    directory.mkdir(exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=directory, suffix=".jsonl", delete=False) as stream:
        path = Path(stream.name)
    try:
        export_events(store.load(run_id), path)
        return path.read_bytes()
    finally:
        path.unlink(missing_ok=True)


def recorded_reward_spec(store: Store, run_id: str) -> dict[str, Any]:
    snapshot = store.load(run_id, include_advanced=False)
    cs = snapshot.completions
    keys = sorted({k for c in cs for k in c.reward_components})
    complete = bool(keys) and all(
        c.reward_total is not None
        and all(c.reward_components.get(k) is not None for k in keys)
        and math.isclose(
            math.fsum(v for v in c.reward_components.values() if v is not None),
            c.reward_total,
            rel_tol=1e-10,
            abs_tol=1e-10,
        )
        for c in cs
    )
    if complete and all(len(k) < 75 for k in keys):
        spec = RewardSpec.model_validate(
            dict(
                name="recorded-component-sum",
                inputs=["component:" + k for k in keys],
                terms=[
                    {"id": k, "type": "identity", "source": "component:" + k, "weight": 1.0} for k in keys
                ],
            )
        )
        description = "Sum of all recorded components; totals match every captured completion."
    else:
        spec = RewardSpec.model_validate(
            dict(
                name="recorded-scalar-reward",
                inputs=["reward_total"],
                terms=[
                    {"id": "captured_reward", "type": "identity", "source": "reward_total", "weight": 1.0}
                ],
            )
        )
        description = "Identity on the captured scalar reward. Component coverage is insufficient to reconstruct its rule."
    return {"spec": spec.model_dump(), "component_coverage_complete": complete, "description": description}


def analysis_history(store: Store, run_id: str | None = None) -> list[dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in store.path("jobs").glob("*.json"):
        if path.is_symlink():
            continue
        job = json.loads(path.read_bytes())
        request = job.get("request", {})
        records[job["analysis_id"]] = {
            **job,
            "run_id": request.get("run_id"),
            "spec_name": request.get("reward_spec", {}).get("name"),
        }
    for path in store.path("analyses").glob("ana_*"):
        if not path.is_dir() or path.is_symlink():
            continue
        result = load_analysis(store, path.name)
        record = records.get(path.name, {})
        records[path.name] = {
            **record,
            "analysis_id": path.name,
            "status": "succeeded",
            "run_id": result["run_id"],
            "snapshot_id": result["snapshot_id"],
            "spec_name": result["reward_spec"]["name"],
            "summary": result["summary"],
            "result_hash": result["result_hash"],
            "group_filter": result["group_filter"],
        }
    return sorted(
        [v for v in records.values() if run_id is None or v.get("run_id") == run_id],
        key=lambda v: (v.get("updated_at", ""), v["analysis_id"]),
        reverse=True,
    )


def record_contract_test(store: Store, contract: RewardContract, spec: RewardSpec) -> dict[str, Any]:
    result = test_contract(contract, spec)
    payload = {"contract": contract.model_dump(mode="json"), "reward_spec": spec.model_dump(mode="json")}
    test_id = "test_" + digest(payload)
    result.update(
        {
            "test_id": test_id,
            "spec_hash": digest(spec.model_dump()),
            "reward_spec": spec.model_dump(),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    path = store.path("contract-runs", test_id + ".json")
    if path.exists():
        return json.loads(path.read_bytes())
    atomic_write(path, canonical(result))
    return result


def contract_history(store: Store, contract_id: str | None = None) -> list[dict[str, Any]]:
    records = [
        json.loads(p.read_bytes()) for p in store.path("contract-runs").glob("*.json") if not p.is_symlink()
    ]
    return sorted(
        [r for r in records if contract_id is None or r["contract_id"] == contract_id],
        key=lambda r: r["created_at"],
        reverse=True,
    )


def contract_results_export(store: Store, kind: str) -> bytes:
    results = contract_history(store)
    if kind == "json":
        return canonical(results)
    if kind == "junit":
        return junit(results)
    raise PTMError("PTM_REPORT_FORMAT", "Use json or junit")


def import_contract(store: Store, content: str) -> RewardContract:
    from ptm.contracts import save_contract

    directory = store.path("imports")
    directory.mkdir(exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=directory, suffix=".yaml", delete=False) as stream:
        path = Path(stream.name)
        stream.write(content.encode())
    try:
        contract = load_contract(path)
        save_contract(store, contract)
        return contract
    finally:
        path.unlink(missing_ok=True)
