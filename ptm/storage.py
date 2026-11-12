import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from ptm.errors import PTMError
from ptm.schemas import Snapshot


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def safe_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
        raise PTMError("PTM_UNSAFE_PATH", "Invalid identifier")
    return value


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def write_parquet(path: Path, rows: list[dict[str, Any]], columns: dict[str, pa.DataType]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema([*(pa.field(k, v) for k, v in columns.items()), pa.field("payload", pa.string())])
    records = [{**{k: row.get(k) for k in columns}, "payload": canonical(row).decode()} for row in rows]
    pq.write_table(
        pa.Table.from_pylist(records, schema=schema),
        path,
        compression="zstd",
        row_group_size=128 if path.stem == "tokens" else 65536,
    )
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


TABLES: dict[str, dict[str, pa.DataType]] = {
    "batches": {"generation_batch_id": pa.string(), "generation_step": pa.int64()},
    "groups": {
        "group_id": pa.string(),
        "generation_batch_id": pa.string(),
        "prompt_id": pa.string(),
        "expected_group_size": pa.int32(),
    },
    "completions": {
        "completion_id": pa.string(),
        "group_id": pa.string(),
        "sequence_index": pa.int32(),
        "reward_total": pa.float64(),
        "reward_status": pa.string(),
    },
    "metrics": {"generation_step": pa.int64(), "name": pa.string(), "value": pa.float64()},
    "optimizer_steps": {"optimizer_step": pa.int64()},
    "tokens": {"token_blob_id": pa.string(), "completion_id": pa.string(), "optimizer_step": pa.int64()},
    "objectives": {"objective_id": pa.string(), "optimizer_step": pa.int64()},
    "trajectories": {"trajectory_id": pa.string(), "completion_id": pa.string()},
}


class Store:
    def __init__(self, root: Path | str = ".ptm") -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, *parts: str) -> Path:
        path = self.root.joinpath(*(safe_id(p) for p in parts))
        if not path.resolve().is_relative_to(self.root):
            raise PTMError("PTM_UNSAFE_PATH", "Symlink leaves data root")
        return path

    def commit(self, snapshot: Snapshot) -> dict[str, Any]:
        data = snapshot.ordered()
        snapshot_id = "snap_" + digest(data)
        run_dir = self.path("runs", snapshot.run.run_id)
        target = self.path("runs", snapshot.run.run_id, "snapshots", snapshot_id)
        if target.exists():
            return self.manifest(snapshot.run.run_id, snapshot_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=".pending-", dir=target.parent))
        try:
            atomic_write(stage / "run.json", canonical(data["run"]))
            for name, columns in TABLES.items():
                write_parquet(stage / f"{name}.parquet", data.get(name, []), columns)
            hashes = {p.name: file_hash(p) for p in sorted(stage.iterdir())}
            manifest = {
                "snapshot_id": snapshot_id,
                "schema_version": snapshot.run.schema_version,
                "run_id": snapshot.run.run_id,
                "files": hashes,
                "completion_count": len(snapshot.completions),
                "group_count": len(snapshot.groups),
            }
            atomic_write(stage / "manifest.json", canonical(manifest))
            try:
                os.rename(stage, target)
            except OSError:
                if not target.exists():
                    raise
            atomic_write(run_dir / "manifest.json", canonical(manifest))
            return manifest
        finally:
            if stage.exists():
                shutil.rmtree(stage)

    def manifest(self, run_id: str, snapshot_id: str | None = None) -> dict[str, Any]:
        path = (
            self.path("runs", run_id, "snapshots", snapshot_id) if snapshot_id else self.path("runs", run_id)
        )
        try:
            result: dict[str, Any] = json.loads((path / "manifest.json").read_bytes())
            return result
        except FileNotFoundError:
            raise PTMError("PTM_NOT_FOUND", "Run or snapshot not found", status=404) from None

    def snapshot_path(self, run_id: str, snapshot_id: str | None = None) -> Path:
        manifest = self.manifest(run_id, snapshot_id)
        return self.path("runs", run_id, "snapshots", manifest["snapshot_id"])

    def load(
        self,
        run_id: str,
        snapshot_id: str | None = None,
        *,
        verify: bool = True,
        include_advanced: bool = True,
    ) -> Snapshot:
        manifest = self.manifest(run_id, snapshot_id)
        path = self.snapshot_path(run_id, manifest["snapshot_id"])
        if verify:
            for name, expected in manifest["files"].items():
                if name not in {"run.json", *(f"{k}.parquet" for k in TABLES)}:
                    raise PTMError("PTM_INTEGRITY_ERROR", "Unexpected manifest file")
                if file_hash(path / name) != expected:
                    raise PTMError("PTM_INTEGRITY_ERROR", f"Checksum mismatch: {name}")
        data = {"run": json.loads((path / "run.json").read_bytes())}
        for name in TABLES:
            if not include_advanced and name in {"tokens", "objectives", "trajectories"}:
                data[name] = []
                continue
            data[name] = (
                [json.loads(row["payload"]) for row in pq.read_table(path / f"{name}.parquet").to_pylist()]
                if f"{name}.parquet" in manifest["files"]
                else []
            )
        snapshot = Snapshot.model_validate(data)
        if verify and include_advanced and "snap_" + digest(snapshot.ordered()) != manifest["snapshot_id"]:
            raise PTMError("PTM_INTEGRITY_ERROR", "Snapshot content hash mismatch")
        return snapshot

    def runs(self) -> list[dict[str, Any]]:
        parent = self.path("runs")
        if not parent.exists():
            return []
        result = []
        for path in sorted(parent.iterdir()):
            if path.name.startswith(".") or not (path / "manifest.json").is_file():
                continue
            manifest = self.manifest(path.name)
            run = json.loads((self.snapshot_path(path.name) / "run.json").read_bytes())
            result.append(
                {
                    **run,
                    **manifest,
                    "coverage": coverage_from_counts(
                        manifest["group_count"], run.get("expected_total_groups")
                    ),
                }
            )
        return result


def coverage_from_counts(captured: int, expected: int | None) -> dict[str, Any]:
    return {
        "captured_groups": captured,
        "expected_groups": expected,
        "fraction": captured / expected if expected else None,
        "scope": "captured snapshot; not a population estimate",
    }


def doctor(snapshot: Snapshot) -> dict[str, Any]:
    from ptm.algorithms import get_recipe

    try:
        get_recipe(
            snapshot.run.recipe_id,
            framework=snapshot.run.framework,
            framework_version=snapshot.run.framework_version,
        )
        recipe_supported = True
    except PTMError:
        recipe_supported = False
    objective_errors = {}
    available_objectives = []
    if snapshot.objectives:
        from ptm.signals import replay_objective

        for objective in snapshot.objectives:
            try:
                replay_objective(snapshot, objective)
                available_objectives.append(objective.objective_id)
            except PTMError as exc:
                objective_errors[objective.objective_id] = exc.as_dict()
    counts: dict[str, int] = {}
    for c in snapshot.completions:
        counts[c.group_id] = counts.get(c.group_id, 0) + 1
    incomplete = [g.group_id for g in snapshot.groups if counts.get(g.group_id, 0) != g.expected_group_size]
    invalid = [c.completion_id for c in snapshot.completions if c.reward_status != "valid"]
    return {
        "run_id": snapshot.run.run_id,
        "schema_version": snapshot.run.schema_version,
        "capture_tier": snapshot.run.capture.tier,
        "source": snapshot.run.capture.source,
        "material": snapshot.run.material,
        "recipe_supported": recipe_supported,
        "complete_groups": len(snapshot.groups) - len(incomplete),
        "incomplete_groups": incomplete,
        "invalid_rewards": invalid,
        "drop_count": snapshot.run.drop_count,
        "capture_errors": snapshot.run.config.get("capture_errors", 0),
        "coverage": coverage_from_counts(len(snapshot.groups), snapshot.run.expected_total_groups),
        "sampling": snapshot.run.capture.model_dump(),
        "capabilities": {
            "reward_diff": recipe_supported
            and snapshot.run.capture.tier != "L0"
            and bool(snapshot.groups)
            and not incomplete
            and not invalid,
            "advantage": recipe_supported
            and snapshot.run.capture.tier != "L0"
            and bool(snapshot.groups)
            and not incomplete
            and not invalid,
            "token_signals": any(
                t.current_policy_logprobs is not None
                and (t.old_policy_logprobs is not None or t.reference_logprobs is not None)
                for t in snapshot.tokens
            ),
            "token_viewer": bool(snapshot.tokens),
            "loss_replay": bool(available_objectives),
            "update_replay": False,
        },
        "update_capture_declared": snapshot.run.config.get("update_capture_id"),
        "token_coverage": {
            "observations": len(snapshot.tokens),
            "completions_with_tokens": len({t.completion_id for t in snapshot.tokens}),
            "total_completions": len(snapshot.completions),
        },
        "available_objectives": available_objectives,
        "objective_errors": objective_errors,
        "missing_fields": []
        if snapshot.tokens
        else ["old_policy_logprob", "current_policy_logprob", "reference_logprob", "loss_mask"],
    }
