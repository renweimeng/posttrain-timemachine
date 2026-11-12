"""Explicit-field verl import and verified rank-spool assembly; no ordering heuristics."""

import json
import tempfile
from pathlib import Path
from typing import Any, Literal

from ptm.errors import PTMError
from ptm.importer import MAX_IMPORT_BYTES, check_depth, read_json, read_snapshot, redact_snapshot
from ptm.schemas import Completion, GenerationBatch, PromptGroup, Run, Snapshot, TokenBlob
from ptm.storage import canonical, file_hash


def read_verl(path: Path, text_capture: Literal["full", "off", "hash_only"] = "full") -> Snapshot:
    if path.is_symlink() or path.stat().st_size > MAX_IMPORT_BYTES:
        raise PTMError("PTM_IMPORT_LIMIT", "Invalid or oversized verl input")
    try:
        lines = path.read_text().splitlines()
        header = json.loads(lines[0])
        check_depth(header)
        run = Run.model_validate(header["ptm_run"])
        if run.framework != "verl":
            raise ValueError("verl header must explicitly declare framework=verl and its source version")
        mapping = header.get("field_map", {})
        if not isinstance(mapping, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in mapping.items()
        ):
            raise ValueError("field_map must map canonical keys to explicit dotted source keys")
        batches: dict[str, GenerationBatch] = {}
        groups: dict[str, PromptGroup] = {}
        completions = []
        tokens = []
        fields = {
            "generation_batch_id",
            "generation_step",
            "event_time",
            "policy_version",
            "group_id",
            "prompt_id",
            "prompt",
            "expected_group_size",
            "completion_id",
            "sequence_index",
            "text",
            "reward_total",
            "reward_status",
            "reward_components",
            "inputs",
            "grade",
            "completion_token_count",
            "stop_reason",
            "worker_rank",
            "token_blob",
        }
        required = {
            "generation_batch_id",
            "generation_step",
            "event_time",
            "policy_version",
            "group_id",
            "prompt_id",
            "expected_group_size",
            "completion_id",
            "sequence_index",
            "reward_total",
            "reward_status",
        }
        for line in lines[1:]:
            if not line.strip():
                continue
            raw = json.loads(line)
            check_depth(raw)
            row: dict[str, Any] = {}
            for key in fields:
                value: Any = raw
                for part in mapping.get(key, key).split("."):
                    value = value.get(part) if isinstance(value, dict) else None
                if value is not None or key == "reward_total":
                    row[key] = value
            if required - set(row):
                raise ValueError(f"Missing explicit verl fields: {sorted(required - set(row))}")
            batch = GenerationBatch(
                **{
                    k: row[k]
                    for k in ["generation_batch_id", "generation_step", "event_time", "policy_version"]
                },
                run_id=run.run_id,
                worker_rank=row.get("worker_rank", 0),
            )
            group = PromptGroup(
                **{k: row[k] for k in ["group_id", "prompt_id", "expected_group_size"]},
                generation_batch_id=batch.generation_batch_id,
                prompt=row.get("prompt"),
                capture=run.capture,
            )
            if batch.generation_batch_id in batches and batches[batch.generation_batch_id] != batch:
                raise ValueError(f"Conflicting explicit batch {batch.generation_batch_id}")
            batches[batch.generation_batch_id] = batch
            if group.group_id in groups and groups[group.group_id] != group:
                raise ValueError(f"Conflicting explicit group {group.group_id}")
            groups[group.group_id] = group
            completion_fields = set(Completion.model_fields)
            completions.append(
                Completion.model_validate({k: v for k, v in row.items() if k in completion_fields})
            )
            if "token_blob" in row:
                tokens.append(TokenBlob.model_validate(row["token_blob"]))
        snapshot = Snapshot(
            run=run,
            batches=list(batches.values()),
            groups=list(groups.values()),
            completions=completions,
            tokens=tokens,
        )
        return redact_snapshot(snapshot, text_capture)
    except PTMError:
        raise
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise PTMError("PTM_IMPORT_INVALID", str(exc)) from None


def merge_rank_manifest(manifest_path: Path) -> Snapshot:
    manifest = read_json(manifest_path)
    root = manifest_path.parent.resolve()
    world_size = manifest.get("world_size")
    shards = manifest.get("shards", [])
    if (
        not isinstance(world_size, int)
        or world_size < 1
        or {s.get("worker_rank") for s in shards} != set(range(world_size))
        or len(shards) != world_size
    ):
        raise PTMError(
            "PTM_INCOMPLETE_RANKS", "Manifest must contain exactly one explicit shard per expected rank"
        )
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as stream:
        combined = Path(stream.name)
        try:
            total = 0
            for shard in shards:
                path = root / shard["file"]
                if path.is_symlink() or not path.resolve().is_relative_to(root) or not path.is_file():
                    raise PTMError(
                        "PTM_UNSAFE_PATH", "Rank shard must be a regular file inside the manifest directory"
                    )
                if path.stat().st_size > MAX_IMPORT_BYTES or file_hash(path) != shard.get("sha256"):
                    raise PTMError("PTM_INTEGRITY_ERROR", "Rank shard size/checksum mismatch")
                data = path.read_bytes()
                total += len(data)
                if total > MAX_IMPORT_BYTES:
                    raise PTMError("PTM_IMPORT_LIMIT", "Combined rank events exceed 128 MiB")
                stream.write(data + b"\n")
            stream.flush()
            snapshot = read_snapshot(combined)
            snapshot.run.drop_count = sum(int(s.get("drop_count", 0)) for s in shards)
            errors = sum(int(s.get("capture_errors", 0)) for s in shards)
            snapshot.run.config["capture_errors"] = errors
            snapshot.run.config["world_size"] = world_size
            expected_ids = {group for shard in shards for group in shard.get("expected_group_ids", [])}
            if expected_ids:
                snapshot.run.expected_total_groups = len(expected_ids)
            if errors:
                snapshot.run.expected_total_groups = None
            return snapshot
        finally:
            combined.unlink(missing_ok=True)


def _rank_import_content(content: str) -> Snapshot:
    payload = json.loads(content)
    check_depth(payload)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        manifest = {"world_size": payload["world_size"], "shards": []}
        for i, shard in enumerate(payload["shards"]):
            path = root / f"rank-{i}.jsonl"
            path.write_text(shard["content"])
            manifest["shards"].append(
                {
                    "file": path.name,
                    "worker_rank": shard["worker_rank"],
                    "sha256": file_hash(path),
                    "drop_count": shard.get("drop_count", 0),
                    "capture_errors": shard.get("capture_errors", 0),
                    "expected_group_ids": shard.get("expected_group_ids", []),
                }
            )
        path = root / "manifest.json"
        path.write_bytes(canonical(manifest))
        return merge_rank_manifest(path)


def rank_import_content(content: str) -> Snapshot:
    try:
        return _rank_import_content(content)
    except PTMError:
        raise
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        raise PTMError("PTM_IMPORT_INVALID", f"Invalid rank import: {exc}") from None
