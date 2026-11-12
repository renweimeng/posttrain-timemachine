import json
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError

from ptm.errors import PTMError
from ptm.schemas import Snapshot
from ptm.storage import digest

MAX_IMPORT_BYTES = 128 * 1024 * 1024
MAX_LINE_BYTES = 2 * 1024 * 1024


def check_depth(value: Any, depth: int = 0, seen: set[int] | None = None) -> None:
    if depth > 24:
        raise PTMError("PTM_IMPORT_LIMIT", "Input nesting exceeds 24 levels")
    if seen is None:
        seen = set()
    if isinstance(value, (dict, list)):
        if id(value) in seen:
            raise PTMError("PTM_IMPORT_LIMIT", "Shared/cyclic YAML containers are not supported")
        seen.add(id(value))
    if isinstance(value, dict):
        for child in value.values():
            check_depth(child, depth + 1, seen)
    elif isinstance(value, list):
        for child in value:
            check_depth(child, depth + 1, seen)


def read_json(path: Path) -> Any:
    if path.is_symlink() or path.stat().st_size > MAX_IMPORT_BYTES:
        raise PTMError("PTM_IMPORT_LIMIT", "Input is a symlink or exceeds 128 MiB")
    try:
        result = json.loads(path.read_bytes())
        check_depth(result)
        return result
    except (ValueError, RecursionError) as exc:
        raise PTMError("PTM_IMPORT_INVALID", str(exc)) from None


def redact_snapshot(snapshot: Snapshot, mode: Literal['full','off','hash_only']) -> Snapshot:
    if mode == 'full':
        return snapshot
    for c in snapshot.completions:
        c.text_hash = (digest(c.text) if c.text is not None else c.text_hash) if mode == 'hash_only' else None
        c.text = None
    for group in snapshot.groups:
        group.prompt = None
    for blob in snapshot.tokens:
        blob.token_pieces = ['[redacted]']*len(blob.token_ids)
    for trajectory in snapshot.trajectories:
        for turn in trajectory.turns:
            turn.content = '[redacted]'
    return snapshot


def read_snapshot(path: Path, text_capture: Literal["full", "off", "hash_only"] = "full") -> Snapshot:
    try:
        if path.suffix == ".jsonl":
            data: dict[str, Any] = {k: [] for k in ["batches", "groups", "completions", "metrics", "optimizer_steps",
                                                   "tokens", "objectives", "trajectories"]}
            seen: dict[str, str] = {}
            total = 0
            if path.is_symlink():
                raise PTMError("PTM_UNSAFE_PATH", "Symlink imports are not accepted")
            with path.open("rb") as stream:
                while line := stream.readline(MAX_LINE_BYTES + 1):
                    total += len(line)
                    if total > MAX_IMPORT_BYTES or len(line) > MAX_LINE_BYTES:
                        raise PTMError("PTM_IMPORT_LIMIT", "JSONL exceeds file/line limit")
                    if not line.strip():
                        continue
                    event = json.loads(line)
                    check_depth(event)
                    if set(event) != {"event_id", "entity", "data"}:
                        raise PTMError("PTM_IMPORT_INVALID", "Event requires event_id, entity, data")
                    eid, entity, value = event["event_id"], event["entity"], event["data"]
                    if not isinstance(eid, str) or len(eid) > 256:
                        raise PTMError("PTM_IMPORT_INVALID", "Invalid event_id")
                    fingerprint = digest(event)
                    if eid in seen:
                        if seen[eid] != fingerprint:
                            raise PTMError("PTM_EVENT_CONFLICT", f"Conflicting repeated event_id {eid}")
                        continue
                    seen[eid] = fingerprint
                    if entity == "run":
                        if "run" in data and data["run"] != value:
                            raise PTMError("PTM_EVENT_CONFLICT", "Multiple run identities in one import")
                        data["run"] = value
                    elif entity in data and entity != "run":
                        data[entity].append(value)
                    else:
                        raise PTMError("PTM_IMPORT_INVALID", "Unknown event entity")
        elif path.suffix == ".json":
            data = read_json(path)
        else:
            raise PTMError("PTM_IMPORT_FORMAT", "Use canonical .json snapshot or .jsonl typed events")
        # v0.0.1 used the same entity fields; migration only promotes the declared schema version.
        if data.get("run", {}).get("schema_version") == "0.0.1":
            data["run"]["schema_version"] = "0.1.0"
        snapshot = Snapshot.model_validate(data)
        return redact_snapshot(snapshot,text_capture)
    except PTMError:
        raise
    except (ValidationError, ValueError, TypeError, KeyError, RecursionError) as exc:
        raise PTMError("PTM_IMPORT_INVALID", str(exc)) from None


def export_events(snapshot: Snapshot, path: Path) -> None:
    with path.open("w") as stream:
        for entity, items in snapshot.ordered().items():
            for value in ([items] if entity == "run" else items):
                event = {"entity": entity, "data": value, "event_id": digest([entity, value])}
                stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
