"""Opt-in, source-pinned TRL loss-boundary capture. Imported only by reference training."""

import hashlib
import inspect
from pathlib import Path
from typing import Any

from ptm.errors import PTMError
from ptm.schemas import ObjectiveBatch, Snapshot, TokenBlob

TRL_SOURCE_SHA = "bb6905182acf2ec426f1dc9ffc37e5210d58ee47d2764f7cf12dbb90a4a28954"


class LossCollector:
    def __init__(self, capture_updates: bool = False) -> None:
        self.capture_updates = capture_updates
        self.batch_id = ""
        self.completion_ids: list[str] = []
        self.frames: list[dict[str, Any]] = []
        self.current: Any = None
        self.errors: list[str] = []
        self.before: dict[str, Any] | None = None
        self.after: dict[str, Any] | None = None

    def before_loss(self, trainer: Any, model: Any) -> None:
        if (
            trainer.accelerator.num_processes != 1
            or trainer.args.gradient_accumulation_steps != 1
            or trainer.args.num_iterations != 1
            or trainer.importance_sampling_level != "token"
            or trainer.top_entropy_quantile != 1
            or trainer.args.delta is not None
            or trainer.use_vllm
            or trainer.use_liger_loss
        ):
            raise PTMError(
                "PTM_ADAPTER_UNSUPPORTED",
                "Deep reference adapter requires single-process token-level non-vLLM GRPO",
            )
        if self.capture_updates and self.before is None:
            from ptm.updates import capture_before

            self.before = capture_before(model, trainer.optimizer, trainer.args)

    def record(self, trainer: Any, inputs: dict[str, Any], loss: Any) -> None:
        if self.current is None:
            raise PTMError("PTM_CAPTURE_MISSING", "Native current-policy forward was not observed")
        if len(self.completion_ids) != len(inputs["advantages"]):
            raise PTMError(
                "PTM_CAPTURE_MISSING", "Explicit captured completion IDs do not align with loss batch"
            )

        def values(x: Any) -> Any:
            return x.detach().float().cpu().tolist() if x is not None else None

        current = values(self.current)
        old = values(inputs.get("old_per_token_logps"))
        old_source = "captured native old_per_token_logps"
        if old is None:
            old = current
            old_source = "native TRL old=current.detach() branch at the same loss forward"
        version = f"optimizer-{trainer.state.global_step}"
        native_inputs = {
            k: values(inputs[k])
            for k in [
                "prompt_ids",
                "prompt_mask",
                "completion_ids",
                "completion_mask",
                "advantages",
                "num_items_in_batch",
            ]
            if k in inputs
        }
        self.frames.append(
            {
                "batch_id": self.batch_id,
                "completion_ids": list(self.completion_ids),
                "optimizer_step": trainer.state.global_step + 1,
                "policy_version": version,
                "old_policy_version": version,
                "old_source": old_source,
                "current": current,
                "old": old,
                "reference": values(inputs.get("ref_per_token_logps")),
                "native_inputs": native_inputs,
                "loss": float(loss.detach().cpu()),
                "loss_type": trainer.loss_type,
                "epsilon_low": trainer.epsilon_low,
                "epsilon_high": trainer.epsilon_high,
                "beta": trainer.beta,
                "max_completion_length": trainer.max_completion_length,
            }
        )

    def capture_after(self, model: Any) -> None:
        if self.capture_updates and self.after is None and self.frames:
            self.after = {k: v.detach().cpu().clone().contiguous() for k, v in model.state_dict().items()}

    def enrich(self, snapshot: Snapshot, tokenizer: Any) -> Snapshot:
        if self.errors or not self.frames:
            raise PTMError("PTM_DEEP_CAPTURE_FAILED", "; ".join(self.errors) or "No loss boundary captured")
        snapshot.run.schema_version = "0.2.0"
        snapshot.run.capture.tier = "L2"
        snapshot.run.capture.source = "trl-0.24.0-loss-boundary"
        for group in snapshot.groups:
            group.capture.tier = "L2"
        for frame_index, frame in enumerate(self.frames):
            blob_ids = []
            ids = frame["native_inputs"]["completion_ids"]
            masks = frame["native_inputs"]["completion_mask"]
            advantages = frame["native_inputs"]["advantages"]
            for i, cid in enumerate(frame["completion_ids"]):
                blob_id = f"tokens-{frame_index}-{i}"
                blob_ids.append(blob_id)
                token_ids = [int(x) for x in ids[i]]
                snapshot.tokens.append(
                    TokenBlob.model_validate(
                        dict(
                            token_blob_id=blob_id,
                            completion_id=cid,
                            tokenizer_hash=snapshot.run.tokenizer_hash,
                            optimizer_step=frame["optimizer_step"],
                            token_ids=token_ids,
                            token_pieces=tokenizer.convert_ids_to_tokens(token_ids),
                            loss_mask=[int(x) for x in masks[i]],
                            current_policy_logprobs=frame["current"][i],
                            old_policy_logprobs=frame["old"][i],
                            reference_logprobs=frame["reference"][i]
                            if frame["reference"] is not None
                            else None,
                            current_policy_version=frame["policy_version"],
                            old_policy_version=frame["old_policy_version"],
                            reference_policy_version="initial-reference"
                            if frame["reference"] is not None
                            else None,
                            behavior_policy_version=next(
                                b.policy_version
                                for b in snapshot.batches
                                if b.generation_batch_id == frame["batch_id"]
                            ),
                            observed_advantage=advantages[i],
                            advantage_recipe_id=snapshot.run.recipe_id,
                            source="actual GRPOTrainer.compute_loss; " + frame["old_source"],
                            source_sha256=TRL_SOURCE_SHA,
                        )
                    )
                )
            snapshot.objectives.append(
                ObjectiveBatch.model_validate(
                    dict(
                        objective_id=f"objective-{frame_index}",
                        run_id=snapshot.run.run_id,
                        optimizer_step=frame["optimizer_step"],
                        token_blob_ids=blob_ids,
                        expected_completion_count=len(blob_ids),
                        declared_complete=True,
                        policy_version=frame["policy_version"],
                        old_policy_version=frame["old_policy_version"],
                        reference_policy_version="initial-reference"
                        if frame["reference"] is not None
                        else None,
                        loss_type=frame["loss_type"],
                        epsilon_low=frame["epsilon_low"],
                        epsilon_high=frame["epsilon_high"],
                        kl_coefficient=frame["beta"],
                        global_effective_tokens=sum(sum(int(v) for v in row) for row in masks),
                        max_completion_length=frame["max_completion_length"],
                        source="actual TRL loss forward",
                        source_loss=frame["loss"],
                    )
                )
            )
        return Snapshot.model_validate(snapshot.model_dump())


def trainer_class(collector: LossCollector) -> Any:
    from trl import GRPOTrainer

    if hashlib.sha256(Path(inspect.getfile(GRPOTrainer)).read_bytes()).hexdigest() != TRL_SOURCE_SHA:
        raise PTMError(
            "PTM_ADAPTER_UNSUPPORTED", "Native TRL source hash differs from the verified deep adapter"
        )

    class InstrumentedGRPOTrainer(GRPOTrainer):
        _ptm_in_loss = False

        def _get_per_token_logps_and_entropies(self, *args: Any, **kwargs: Any) -> Any:
            result = super()._get_per_token_logps_and_entropies(*args, **kwargs)
            if self._ptm_in_loss:
                collector.current = result[0]
            return result

        def compute_loss(
            self, model: Any, inputs: Any, return_outputs: bool = False, num_items_in_batch: Any = None
        ) -> Any:
            try:
                collector.before_loss(self, model)
            except Exception as exc:
                collector.errors.append(type(exc).__name__ + ": " + str(exc))
            self._ptm_in_loss = True
            try:
                result = super().compute_loss(
                    model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch
                )
            finally:
                self._ptm_in_loss = False
            try:
                collector.record(self, inputs, result)
            except Exception as exc:
                collector.errors.append(type(exc).__name__ + ": " + str(exc))
            finally:
                collector.current = None
            return result

    return InstrumentedGRPOTrainer
