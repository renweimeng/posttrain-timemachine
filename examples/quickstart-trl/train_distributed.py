"""Actual two-rank CPU TRL training with explicit-ID distributed L1 capture.

Run with torch.distributed.run --nproc_per_node=2. No downloaded weights/data.
"""

import argparse
import json
import os
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch
import torch.distributed as dist
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
from trl import GRPOConfig, GRPOTrainer

from ptm.algorithms import advantages, get_recipe
from ptm.capture import CaptureContext, DistributedTRLRewardCapture, SpoolWriter
from ptm.interop import merge_rank_manifest
from ptm.schemas import Capture, Run
from ptm.storage import Store, atomic_write, canonical, digest, file_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("output/distributed-trl"))
    parser.add_argument("--root", type=Path, default=Path(".ptm"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    if world != 2:
        raise RuntimeError("This reference requires exactly two ranks")
    vocab = {
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
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", eos_token="[EOS]", unk_token="[UNK]"
    )
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=len(vocab),
            n_embd=32,
            n_layer=1,
            n_head=2,
            n_positions=64,
            eos_token_id=1,
            pad_token_id=0,
            resid_pdrop=0,
            embd_pdrop=0,
            attn_pdrop=0,
        )
    )
    data = Dataset.from_dict(
        {"prompt": ["What is 1 + 2 ?"] * 4, "ptm_group_id": [f"explicit-q{i}" for i in range(4)]}
    )
    config = {
        "num_generations": 4,
        "world_size": world,
        "per_device_train_batch_size": 2,
        "loss_type": "dapo",
    }
    run = Run(
        run_id="tiny-trl-distributed",
        name="Actual two-rank CPU GRPO capture",
        framework="trl",
        framework_version="0.24.0",
        recipe_id="trl-0.24.0-grpo-group-v1",
        config_hash=digest(config),
        config=config,
        model_id="local-random-gpt2",
        tokenizer_hash=digest(vocab),
        material="real captured trace",
        expected_total_groups=1,
        capture=Capture(source="trl-0.24.0-distributed-explicit-ids"),
    )
    output = args.output / f"rank-{rank}"
    output.mkdir(parents=True, exist_ok=True)
    spool = output / "events.jsonl"
    spool.unlink(missing_ok=True)
    writer = SpoolWriter(spool, run, worker_rank=rank, world_size=world)
    state = {"call": 0, "rewards": []}
    recorded = []
    adapter = None

    def context(kwargs):
        state["call"] += 1
        return CaptureContext(
            generation_batch_id=f"distributed-batch-{state['call']}",
            generation_step=state["call"],
            policy_version=f"optimizer-{kwargs['trainer_state'].global_step}",
        )

    def reward(completion_ids, **kwargs):
        state["rewards"] = [float(sum(ids) % 7) / 6 for ids in completion_ids]
        return state["rewards"]

    def fields(i, kwargs):
        return {
            "inputs": {"token_proxy": state["rewards"][i]},
            "reward_components": {"token_proxy": state["rewards"][i]},
        }

    # Accelerator initializes the actual CPU process group during GRPOTrainer construction.
    def wrapped(**kwargs):
        nonlocal adapter
        if adapter is None:
            adapter = DistributedTRLRewardCapture(writer, context, group_size=4)
        return adapter.wrap(reward, fields)(**kwargs)

    wrapped.__name__ = "distributed_captured_reward"

    class ReferenceTrainer(GRPOTrainer):
        def _generate_and_score_completions(self, inputs):
            result = super()._generate_and_score_completions(inputs)
            recorded.append(
                {
                    "completion_ids": adapter.last_completion_ids,
                    "advantages": result["advantages"].detach().cpu().tolist(),
                }
            )
            return result

    trainer = ReferenceTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=data,
        reward_funcs=wrapped,
        args=GRPOConfig(
            output_dir=str(output / "trainer"),
            use_cpu=True,
            bf16=False,
            fp16=False,
            max_steps=1,
            per_device_train_batch_size=2,
            generation_batch_size=4,
            num_generations=4,
            max_completion_length=8,
            gradient_checkpointing=False,
            optim="adamw_torch",
            save_strategy="no",
            report_to=[],
            disable_tqdm=True,
            dataloader_pin_memory=False,
            ddp_backend="gloo",
            ddp_find_unused_parameters=False,
            beta=0.0,
            seed=42,
        ),
    )
    initial = next(model.parameters()).detach().clone()
    try:
        trainer.train()
    finally:
        writer.close()
    atomic_write(
        output / "native.json",
        canonical(
            {
                "rank": rank,
                "records": recorded,
                "weights_changed": not torch.equal(initial, next(model.parameters()).detach()),
            }
        ),
    )
    dist.barrier()
    if rank == 0:
        shards = []
        for r in range(world):
            path = args.output / f"rank-{r}/events.jsonl"
            health = json.loads(path.with_suffix(".health.json").read_bytes())
            shards.append(
                {
                    "worker_rank": r,
                    "file": str(path.relative_to(args.output)),
                    "sha256": file_hash(path),
                    "drop_count": health["dropped_groups"],
                    "capture_errors": health["capture_errors"],
                    "expected_group_ids": health["expected_group_ids"],
                }
            )
        manifest = args.output / "rank-manifest.json"
        atomic_write(manifest, canonical({"world_size": world, "shards": shards}))
        snapshot = merge_rank_manifest(manifest)
        stored = Store(args.root).commit(snapshot)
        expected = {}
        for r in range(world):
            native = json.loads((args.output / f"rank-{r}/native.json").read_bytes())
            for batch in native["records"]:
                expected.update(zip(batch["completion_ids"], batch["advantages"], strict=True))
        for group in snapshot.groups:
            cs = sorted(
                [c for c in snapshot.completions if c.group_id == group.group_id],
                key=lambda c: c.sequence_index,
            )
            actual = advantages([[c.reward_total for c in cs]], get_recipe(run.recipe_id))[0]
            np.testing.assert_allclose(actual, [expected[c.completion_id] for c in cs], rtol=1e-5, atol=2e-6)
        result = {
            "snapshot_id": stored["snapshot_id"],
            "ranks": world,
            "groups": len(snapshot.groups),
            "completions": len(snapshot.completions),
            "reference_parity": True,
            "native_advantages": expected,
            "scope": "actual CPU gloo GRPO training and L1 capture; GPU throughput not measured",
        }
        atomic_write(args.output / "result.json", canonical(result))
        print(json.dumps(result, indent=2))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
