"""Bounded, best-effort L1 capture. Capturing must never change reward return values."""

import functools
import importlib.metadata
import json
import logging
import math
import os
import queue
import threading
from collections import Counter
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from ptm.errors import PTMError
from ptm.importer import read_snapshot
from ptm.schemas import Capture, Completion, GenerationBatch, Grade, Model, PromptGroup, Run
from ptm.storage import Store, atomic_write, canonical, digest


class CaptureContext(Model):
    generation_batch_id: str
    generation_step: int
    policy_version: str


class SpoolWriter:
    def __init__(
        self,
        path: Path,
        run: Run,
        *,
        max_batches: int = 8,
        max_batch_bytes: int = 8 * 1024 * 1024,
        worker_rank: int = 0,
        world_size: int = 1,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.run_id = run.run_id
        self.worker_rank = worker_rank
        self.world_size = world_size
        self.expected_group_ids: set[str] = set()
        self.expected_groups = 0
        self.capture_errors = 0
        self.max_batch_bytes = max_batch_bytes
        self.queue: queue.Queue[bytes | None] = queue.Queue(maxsize=max_batches)
        self.dropped_batches = 0
        self.dropped_groups = 0
        self.error: str | None = None
        self.closed = False
        # Metadata is durable before training starts. Never append to a previous capture session.
        self.stream = path.open("xb")
        self.stream.write(self.encode([("run", run.model_dump(mode="json"))]))
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.thread = threading.Thread(target=self.work, daemon=True, name="ptm-spool")
        self.thread.start()

    @staticmethod
    def encode(events: list[tuple[str, dict[str, Any]]]) -> bytes:
        return b"".join(
            canonical({"event_id": digest([kind, data]), "entity": kind, "data": data}) + b"\n"
            for kind, data in events
        )

    def submit(self, events: list[tuple[str, dict[str, Any]]], group_count: int) -> bool:
        try:
            if self.closed or self.error:
                raise ValueError("capture writer closed or failed")
            payload = self.encode(events)
            if len(payload) > self.max_batch_bytes:
                raise ValueError("capture batch exceeds bounded buffer size")
            self.queue.put_nowait(payload)
            return True
        except (ValueError, TypeError, queue.Full):
            self.dropped_batches += 1
            self.dropped_groups += group_count
            return False

    def work(self) -> None:
        while True:
            payload = self.queue.get()
            try:
                if payload is None:
                    return
                if not self.error:
                    try:
                        self.stream.write(payload)
                        self.stream.flush()
                    except OSError as exc:
                        self.error = str(exc)
                        # Count all groups in a dropped serialized batch.
                        self.dropped_batches += 1
                        self.dropped_groups += sum(
                            json.loads(line)["entity"] == "groups" for line in payload.splitlines()
                        )
                else:
                    self.dropped_batches += 1
                    self.dropped_groups += sum(
                        json.loads(line)["entity"] == "groups" for line in payload.splitlines()
                    )
            finally:
                self.queue.task_done()

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.queue.put(None)
        self.thread.join()
        try:
            self.stream.flush()
            os.fsync(self.stream.fileno())
        finally:
            self.stream.close()
        atomic_write(
            self.path.with_suffix(".health.json"),
            canonical(
                {
                    "dropped_batches": self.dropped_batches,
                    "dropped_groups": self.dropped_groups,
                    "capture_errors": self.capture_errors,
                    "error": self.error,
                    "worker_rank": self.worker_rank,
                    "world_size": self.world_size,
                    "expected_group_ids": sorted(self.expected_group_ids),
                }
            ),
        )

    def compact(self, store: Store) -> dict[str, Any]:
        self.close()
        if self.error:
            raise PTMError("PTM_CAPTURE_IO_ERROR", self.error)
        snapshot = read_snapshot(self.path)
        snapshot.run.drop_count = self.dropped_groups
        snapshot.run.expected_total_groups = None if self.capture_errors else self.expected_groups
        snapshot.run.config["capture_errors"] = self.capture_errors
        return store.commit(snapshot)


class TRLRewardCapture:
    """TRL 0.24.0 single-process L1 adapter. Explicit dataset group keys are mandatory.

    Wrap ONE combined reward callable (trainer reward_weight=1). Internal tensor
    hooks and distributed groups are intentionally unsupported.
    """

    def __init__(
        self,
        writer: SpoolWriter,
        context: Callable[[dict[str, Any]], CaptureContext],
        *,
        group_size: int,
        sample_probability: float = 1.0,
        text_capture: Literal["full", "off", "hash_only"] = "full",
        _allow_distributed: bool = False,
    ) -> None:
        version = importlib.metadata.version("trl")
        if version != "0.24.0" or (int(os.environ.get("WORLD_SIZE", "1")) != 1 and not _allow_distributed):
            raise PTMError(
                "PTM_ADAPTER_UNSUPPORTED", "Tested adapter requires TRL 0.24.0, single-process capture"
            )
        if not 0 < sample_probability <= 1 or group_size < 2:
            raise ValueError("sample_probability must be (0,1], group_size >= 2")
        self.writer, self.context, self.group_size = writer, context, group_size
        self.sample_probability, self.text_capture = sample_probability, text_capture
        self.capture_errors = 0
        self.seen_groups = 0
        self.last_completion_ids: list[str] = []

    def wrap(
        self,
        reward: Callable[..., Any],
        fields: Callable[[int, dict[str, Any]], dict[str, Any]] | None = None,
    ) -> Callable[..., Any]:
        @functools.wraps(reward)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            values = reward(
                *args, **kwargs
            )  # Exceptions from the reward itself keep their original semantics.
            try:
                self.record(values, kwargs, fields)
            except Exception as exc:
                self.capture_errors += 1
                self.writer.capture_errors += 1
                logging.getLogger(__name__).warning(
                    "PTM capture failed (%s); trainer reward return preserved", type(exc).__name__
                )
            return values  # Preserve exact identity/type; a serializer can never replace this value.

        return wrapped

    def record(
        self,
        values: Any,
        kwargs: dict[str, Any],
        fields: Callable[[int, dict[str, Any]], dict[str, Any]] | None,
    ) -> None:
        keys = kwargs.get("ptm_group_id")
        prompts, texts = kwargs["prompts"], kwargs["completions"]
        if not isinstance(keys, list) or len(keys) != len(texts) or len(values) != len(texts):
            raise PTMError("PTM_GROUP_ID_REQUIRED", "Provide explicit ptm_group_id per completion")
        counts = Counter(keys)
        if any(count != self.group_size for count in counts.values()):
            raise PTMError("PTM_INCOMPLETE_GROUP", "Only whole groups may be captured by this adapter")
        context = self.context(kwargs)
        selected = {
            key
            for key in counts
            if int(digest([context.generation_batch_id, key])[:16], 16) / 2**64 < self.sample_probability
        }
        self.seen_groups += len(counts)
        self.writer.expected_groups += len(counts)
        if not selected:
            return
        capture = Capture(
            source="trl-0.24.0-reward-wrapper",
            sampling_probability=self.sample_probability,
            selection_rule="deterministic uniform group hash",
        )
        batch = GenerationBatch(
            **context.model_dump(), run_id=self.run_id(), event_time=datetime.now(timezone.utc)
        )
        events: list[tuple[str, dict[str, Any]]] = [("batches", batch.model_dump(mode="json"))]
        sequence: dict[str, int] = {}
        for i, key in enumerate(keys):
            if key not in selected:
                continue
            gid = "grp_" + digest([context.generation_batch_id, key])[:32]
            n = sequence.get(key, 0)
            sequence[key] = n + 1
            if n == 0:
                prompt = (
                    prompts[i] if isinstance(prompts[i], str) else json.dumps(prompts[i], ensure_ascii=False)
                )
                events.append(
                    (
                        "groups",
                        PromptGroup(
                            group_id=gid,
                            generation_batch_id=context.generation_batch_id,
                            prompt_id=digest(prompt),
                            prompt=prompt if self.text_capture == "full" else None,
                            expected_group_size=self.group_size,
                            capture=capture,
                        ).model_dump(mode="json"),
                    )
                )
            text = texts[i] if isinstance(texts[i], str) else json.dumps(texts[i], ensure_ascii=False)
            raw = values[i]
            status: Literal["missing", "valid", "invalid"] = (
                "missing" if raw is None else ("valid" if math.isfinite(float(raw)) else "invalid")
            )
            extra = dict(fields(i, kwargs)) if fields else {}
            grade = extra.pop("grade", None)
            tokens = kwargs.get("completion_ids", [None] * len(texts))[i]
            completion = Completion(
                completion_id=f"{gid}-c{n}",
                group_id=gid,
                sequence_index=n,
                text=text if self.text_capture == "full" else None,
                text_hash=digest(text) if self.text_capture == "hash_only" else None,
                reward_status=status,
                reward_total=float(raw) if status == "valid" else None,
                grade=Grade.model_validate(grade) if grade else None,
                completion_token_count=len(tokens) if tokens is not None else None,
                **extra,
            )
            events.append(("completions", completion.model_dump(mode="json")))
        if selected:
            self.writer.submit(events, len(selected))

    def run_id(self) -> str:
        return self.writer.run_id


class DistributedTRLRewardCapture(TRLRewardCapture):
    """Explicit-ID L1 capture for source-pinned TRL on an initialized trusted process group.

    A small group-ID collective establishes sequence indices. No completion text
    or policy tensors are gathered by the capture layer. Per-rank files are
    assembled only after all rank manifests are present and verified.
    """

    def __init__(
        self,
        writer: SpoolWriter,
        context: Callable[[dict[str, Any]], CaptureContext],
        *,
        group_size: int,
        sample_probability: float = 1.0,
        text_capture: Literal["full", "off", "hash_only"] = "full",
    ) -> None:
        import torch.distributed as dist

        if (
            not dist.is_initialized()
            or dist.get_world_size() != writer.world_size
            or dist.get_rank() != writer.worker_rank
        ):
            raise PTMError(
                "PTM_ADAPTER_UNSUPPORTED",
                "Explicit writer rank/world size must match the initialized process group",
            )
        super().__init__(
            writer,
            context,
            group_size=group_size,
            sample_probability=sample_probability,
            text_capture=text_capture,
            _allow_distributed=True,
        )

    def record(
        self,
        values: Any,
        kwargs: dict[str, Any],
        fields: Callable[[int, dict[str, Any]], dict[str, Any]] | None,
    ) -> None:
        import torch.distributed as dist

        payload: dict[str, Any] = {}
        try:
            keys = kwargs.get("ptm_group_id")
            if not isinstance(keys, list) or not all(isinstance(k, str) for k in keys):
                raise ValueError("Explicit string group IDs required")
            if len(keys) != len(values) or len(keys) != len(kwargs["prompts"]):
                raise ValueError("Reward and metadata length mismatch")
            context = self.context(kwargs)
            payload = {
                "keys": keys,
                "context": context.model_dump(),
                "prompt_hashes": [digest(p) for p in kwargs["prompts"]],
                "event_time": datetime.now(timezone.utc).isoformat() if dist.get_rank() == 0 else None,
            }
        except Exception as exc:
            payload = {"error": type(exc).__name__}
        gathered: list[Any] = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, payload)
        if any("error" in part for part in gathered):
            raise PTMError(
                "PTM_GROUP_ID_REQUIRED", "One or more ranks could not validate explicit capture metadata"
            )
        contexts = [part["context"] for part in gathered]
        if any(c != contexts[0] for c in contexts):
            raise PTMError("PTM_POLICY_MISMATCH", "Ranks disagree on generation-batch/policy identity")
        context = CaptureContext.model_validate(contexts[0])
        parts = [part["keys"] for part in gathered]
        counts = Counter(k for part in parts for k in part)
        if any(n != self.group_size for n in counts.values()):
            raise PTMError("PTM_INCOMPLETE_GROUP", "Global explicit group counts differ from configured G")
        hashes: dict[str, str] = {}
        for part in gathered:
            for key, value in zip(part["keys"], part["prompt_hashes"], strict=True):
                if key in hashes and hashes[key] != value:
                    raise PTMError(
                        "PTM_GROUP_ID_CONFLICT", "The same explicit group key refers to different prompts"
                    )
                hashes[key] = value
        rank = dist.get_rank()
        keys = parts[rank]
        capture = Capture(
            source="trl-0.24.0-distributed-explicit-ids",
            sampling_probability=self.sample_probability,
            selection_rule="deterministic uniform global group hash",
        )
        self.writer.expected_group_ids.update(
            "grp_" + digest([context.generation_batch_id, k])[:32] for k in counts
        )
        self.writer.expected_groups = len(self.writer.expected_group_ids)
        self.last_completion_ids = []
        prefix = Counter(k for part in parts[:rank] for k in part)
        owners = {k: next(r for r, part in enumerate(parts) if k in part) for k in counts}
        batch = GenerationBatch(
            **context.model_dump(),
            run_id=self.writer.run_id,
            event_time=gathered[0]["event_time"],
            worker_rank=0,
        )
        events: list[tuple[str, dict[str, Any]]] = [("batches", batch.model_dump(mode="json"))]
        local_groups = set()
        for i, key in enumerate(keys):
            seq = prefix[key]
            prefix[key] += 1
            gid = "grp_" + digest([context.generation_batch_id, key])[:32]
            cid = f"{gid}-c{seq}"
            self.last_completion_ids.append(cid)
            if int(digest([context.generation_batch_id, key])[:16], 16) / 2**64 >= self.sample_probability:
                continue
            if key not in local_groups and owners[key] == rank:
                prompt = kwargs["prompts"][i]
                prompt = prompt if isinstance(prompt, str) else json.dumps(prompt, ensure_ascii=False)
                events.append(
                    (
                        "groups",
                        PromptGroup(
                            group_id=gid,
                            generation_batch_id=context.generation_batch_id,
                            prompt_id=digest(prompt),
                            prompt=prompt if self.text_capture == "full" else None,
                            expected_group_size=self.group_size,
                            capture=capture,
                            worker_rank=rank,
                        ).model_dump(mode="json"),
                    )
                )
            local_groups.add(key)
            raw = values[i]
            status: Literal["valid", "missing", "invalid"] = (
                "missing" if raw is None else "valid" if math.isfinite(float(raw)) else "invalid"
            )
            text = kwargs["completions"][i]
            text = text if isinstance(text, str) else json.dumps(text, ensure_ascii=False)
            extra = dict(fields(i, kwargs)) if fields else {}
            grade = extra.pop("grade", None)
            token_ids = kwargs.get("completion_ids", [None] * len(keys))[i]
            c = Completion(
                completion_id=cid,
                group_id=gid,
                sequence_index=seq,
                worker_rank=rank,
                text=text if self.text_capture == "full" else None,
                text_hash=digest(text) if self.text_capture == "hash_only" else None,
                reward_total=float(raw) if status == "valid" else None,
                reward_status=status,
                completion_token_count=len(token_ids) if token_ids is not None else None,
                grade=Grade.model_validate(grade) if grade else None,
                **extra,
            )
            events.append(("completions", c.model_dump(mode="json")))
        if len(events) > 1:
            self.writer.submit(events, len(local_groups))
