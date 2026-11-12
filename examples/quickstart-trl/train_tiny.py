"""Actual, fully offline CPU GRPO training and source-pinned numerical fixture capture.

No pretrained weights, model downloads or network graders. This is an integration
experiment, not evidence of useful language-model performance.
"""

import argparse
import hashlib
import inspect
import json
import os
import time
from pathlib import Path
from typing import Any

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import transformers
import trl
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

from ptm.capture import CaptureContext, SpoolWriter, TRLRewardCapture
from ptm.schemas import Capture, OptimizerStep, Run
from ptm.storage import Store, atomic_write, canonical, digest


def train(
    output: Path,
    store_root: Path,
    scale: str = "group",
    steps: int = 3,
    capture_on: bool = True,
    sample_probability: float = 1.0,
    deep_capture: bool = False,
    capture_updates: bool = False,
    loss_type: str = "dapo",
    beta: float | None = None,
    device: str = "cpu",
    optimizer_kind: str = "AdamW",
) -> dict[str, Any]:
    if trl.__version__ != "0.24.0" or transformers.__version__ != "4.57.1":
        raise RuntimeError("Install the pinned .[trl] extra first")
    if device not in {"cpu", "cuda"} or (device == "cuda" and not torch.cuda.is_available()):
        raise RuntimeError("Requested CUDA device is unavailable; use --device cpu for actual CPU validation")
    if deep_capture and (not capture_on or sample_probability != 1 or device != "cpu"):
        raise RuntimeError(
            "Verified deep reference capture requires full CPU capture; GPU parity is not established"
        )
    if loss_type not in {"grpo", "dapo", "bnpo", "dr_grpo"}:
        raise ValueError("Unsupported loss type")
    effective_beta = (0.02 if deep_capture else 0.0) if beta is None else beta
    collector = None
    if deep_capture:
        from ptm.deep_capture import LossCollector, trainer_class

        collector = LossCollector(capture_updates=capture_updates)
        BaseTrainer = trainer_class(collector)
    else:
        BaseTrainer = GRPOTrainer
    torch.set_num_threads(1)
    torch.manual_seed(42)
    output.mkdir(parents=True, exist_ok=True)
    vocabulary = {
        s: i
        for i, s in enumerate(
            [
                "[PAD]",
                "[EOS]",
                "[UNK]",
                "What",
                "is",
                "1",
                "+",
                "2",
                "3",
                "4",
                "?",
                "Final",
                "answer",
                ":",
                "5",
                "6",
            ]
        )
    }
    backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", eos_token="[EOS]", unk_token="[UNK]"
    )
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=len(vocabulary),
            n_embd=32,
            n_layer=1,
            n_head=2,
            n_positions=64,
            bos_token_id=1,
            eos_token_id=1,
            pad_token_id=0,
            resid_pdrop=0,
            embd_pdrop=0,
            attn_pdrop=0,
        )
    )
    if effective_beta:
        base_path = output / "initial-local-model"
        model.save_pretrained(base_path, safe_serialization=True)
        tokenizer.save_pretrained(base_path)
        model.config._name_or_path = str(base_path.resolve())
    dataset = Dataset.from_dict(
        {
            "prompt": ["What is 1 + 2 ?", "What is 2 + 2 ?", "What is 1 + 1 ?", "What is 2 + 3 ?"],
            "ptm_group_id": ["q-1-2", "q-2-2", "q-1-1", "q-2-3"],
        }
    )
    config = {
        "num_generations": 4,
        "scale_rewards": scale,
        "loss_type": loss_type,
        "kl_coefficient": effective_beta,
        "importance_sampling_level": "token",
        "top_entropy_quantile": 1.0,
        "use_vllm": False,
        "reward_batch_scope": "full_snapshot" if steps == 1 else "multiple_generation_batches",
    }
    recipe_id = f"trl-0.24.0-grpo-{scale}-v1"
    run_id = f"tiny-trl-{scale}" + (f"-l2-{loss_type}-{optimizer_kind.lower()}" if deep_capture else "")
    run = Run(
        run_id=run_id,
        name="Actual tiny random GPT-2 CPU integration",
        framework="trl",
        framework_version=trl.__version__,
        recipe_id=recipe_id,
        config_hash=digest(config),
        config=config,
        seed=42,
        model_id="local-random-gpt2-1layer-32dim",
        tokenizer_hash=digest(vocabulary),
        material="real captured trace",
        capture=Capture(
            source="trl-0.24.0-reward-wrapper",
            sampling_probability=sample_probability,
            selection_rule="deterministic uniform group hash",
        ),
    )
    spool = output / "rank-0-events.jsonl"
    if spool.exists():
        spool.unlink()
    writer = SpoolWriter(spool, run)
    counter = 0
    last_rewards: list[float] = []
    last_inputs: list[dict[str, float]] = []
    goldens: list[dict[str, Any]] = []
    batch_ids: list[str] = []

    def context(kwargs: dict[str, Any]) -> CaptureContext:
        nonlocal counter
        counter += 1
        bid = f"gb-{scale}-{counter}"
        batch_ids.append(bid)
        if collector is not None:
            collector.batch_id = bid
        return CaptureContext(
            generation_batch_id=bid,
            generation_step=counter,
            policy_version=f"optimizer-{kwargs['trainer_state'].global_step}",
        )

    def reward(completions: list[str], completion_ids: list[list[int]], **kwargs: Any) -> list[float]:
        nonlocal last_rewards, last_inputs
        # Explicit proxy reward on generated tokens; independent task accuracy is not established.
        last_inputs = [
            {"token_proxy": float(sum(ids) % 7) / 6, "has_final": float("Final" in text)}
            for text, ids in zip(completions, completion_ids, strict=True)
        ]
        last_rewards = [v["token_proxy"] + 0.1 * v["has_final"] for v in last_inputs]
        return last_rewards

    def fields(i: int, kwargs: dict[str, Any]) -> dict[str, Any]:
        if collector is not None:
            if i == 0:
                collector.completion_ids = []
            key = kwargs["ptm_group_id"][i]
            seq = kwargs["ptm_group_id"][:i].count(key)
            collector.completion_ids.append("grp_" + digest([batch_ids[-1], key])[:32] + f"-c{seq}")
        return {
            "inputs": last_inputs[i],
            "reward_components": {
                "token_proxy": last_inputs[i]["token_proxy"],
                "format": 0.1 * last_inputs[i]["has_final"],
            },
        }

    adapter = TRLRewardCapture(writer, context, group_size=4, sample_probability=sample_probability)

    class RecordingTrainer(BaseTrainer):
        def _generate_and_score_completions(self, inputs: list[dict[str, Any]]) -> dict[str, Any]:
            result = super()._generate_and_score_completions(inputs)
            goldens.append(
                {
                    "rewards": list(last_rewards),
                    "advantages": result["advantages"].detach().cpu().tolist(),
                    "group_ids_from_dataset": [v["ptm_group_id"] for v in inputs],
                    "group_size": 4,
                    "generation_batch_id": batch_ids[-1] if batch_ids else None,
                }
            )
            return result

    args = GRPOConfig(
        output_dir=str(output / "trainer"),
        use_cpu=device == "cpu",
        bf16=False,
        fp16=False,
        max_steps=steps,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        num_generations=4,
        max_prompt_length=16,
        max_completion_length=8,
        generation_batch_size=4,
        gradient_checkpointing=False,
        optim="adamw_torch",
        save_strategy="no",
        report_to=[],
        logging_steps=1,
        disable_tqdm=True,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        scale_rewards=scale,
        beta=effective_beta,
        loss_type=loss_type,
        seed=42,
    )

    class AfterStep(TrainerCallback):
        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            if collector is not None:
                collector.capture_after(kwargs["model"])

    supplied_optimizer = (
        torch.optim.SGD(model.parameters(), lr=1e-6, momentum=0.9) if optimizer_kind == "SGD" else None
    )
    trainer = RecordingTrainer(
        model=model,
        args=args,
        processing_class=tokenizer,
        train_dataset=dataset,
        reward_funcs=adapter.wrap(reward, fields) if capture_on else reward,
        callbacks=[AfterStep()] if collector is not None else [],
        optimizers=(supplied_optimizer, None),
    )
    initial = next(model.parameters()).detach().clone()
    start = time.perf_counter()
    try:
        trained = trainer.train()
    finally:
        writer.close()
    duration = time.perf_counter() - start
    if not capture_on:
        return {"seconds": duration, "capture_on": False}
    store = Store(store_root)
    manifest = writer.compact(store)
    snapshot = store.load(run_id)
    from datetime import datetime, timezone

    snapshot.optimizer_steps = [
        OptimizerStep(
            run_id=run_id,
            optimizer_step=i + 1,
            timestamp=datetime.now(timezone.utc),
            algorithm_recipe_id=recipe_id,
            generation_batch_ids=[bid],
        )
        for i, bid in enumerate(batch_ids)
    ]
    if collector is not None:
        snapshot = collector.enrich(snapshot, tokenizer)
    manifest = store.commit(snapshot)
    if collector is not None and capture_updates:
        from ptm.updates import save_update_capture

        capture_id = save_update_capture(store, run_id, manifest["snapshot_id"], collector)
        snapshot.run.config["update_capture_id"] = capture_id
        manifest = store.commit(snapshot)
    source = Path(inspect.getfile(GRPOTrainer))
    fixture = {
        "provenance": {
            "framework": "trl",
            "version": trl.__version__,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "source_url": "https://github.com/huggingface/trl/blob/v0.24.0/trl/trainer/grpo_trainer.py",
            "method": "actual GRPOTrainer._generate_and_score_completions return, after super()",
            "dtype": "torch.float32",
            "seed": 42,
            "scale_rewards": scale,
            "epsilon": 1e-4,
            "ddof": 1,
            "model": run.model_id,
            "useful_model_quality": "not evaluated",
            "seconds": duration,
        },
        "batches": goldens,
        "snapshot_id": manifest["snapshot_id"],
        "capture_errors": adapter.capture_errors,
        "deep_capture_errors": collector.errors if collector else [],
        "run_id": run_id,
        "update_capture_id": snapshot.run.config.get("update_capture_id"),
        "weights_changed": not torch.equal(initial, next(model.parameters()).detach()),
        "train_loss": trained.training_loss,
    }
    atomic_write(output / "golden.json", canonical(fixture))
    return fixture


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("output/trl-capture"))
    parser.add_argument("--root", type=Path, default=Path(".ptm"))
    parser.add_argument("--scale", choices=["group", "none", "batch"], default="group")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--deep", action="store_true")
    parser.add_argument("--capture-update", action="store_true")
    parser.add_argument("--loss", choices=["grpo", "dapo", "bnpo", "dr_grpo"], default="dapo")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    result = train(
        args.output,
        args.root,
        args.scale,
        args.steps,
        deep_capture=args.deep,
        capture_updates=args.capture_update,
        loss_type=args.loss,
        device=args.device,
    )
    print(json.dumps(result, indent=2))
