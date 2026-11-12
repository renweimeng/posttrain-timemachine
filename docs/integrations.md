# Capture integration and limits

| Adapter | Verified source | Supported evidence | Boundary |
|---|---|---|---|
| Canonical JSON/JSONL | schemas 0.1.0 and 0.2.0 | Explicit L0/L1/L2 entities, Token/Objective/Trajectory IDs | Missing tensors are never reconstructed |
| TRL combined reward wrapper | TRL 0.24.0 | Single-process L1, explicit group IDs, saved text/reward/components/grades | One combined callable and unit outer weight |
| TRL distributed wrapper | TRL 0.24.0, actual 2-rank CPU gloo run | Global explicit groups spanning ranks, native advantage parity, health sidecars and verified merge | Trusted initialized process group; no live distributed L2 claim |
| TRL deep capture | Exact source fingerprint in numerical guide | Actual current/old/reference logprobs, masks, sequence advantage and complete native objective | Opt-in, CPU, one process, accumulation=1, num_iterations=1, token importance, no entropy/delta/vLLM/Liger modifiers |
| Numeric checkpoint replay | Tiny local GPT-2, native AdamW/SGD | Before-state, numeric optimizer/RNG state, fixed inputs, actual native after-state reference | CPU only; ≤20M parameters, bounded config; no pickle/custom model code |
| verl importer | Explicit declared source version + field map | Canonical IDs and optional saved L2 Token blobs | Not a native verl trainer hook or validated verl loss recipe |
| Multi-turn import | Explicit canonical Trajectory entities | Recorded system/user/assistant/tool turns and policy provenance | No tool execution or automatic Agent framework instrumentation |

Install the optional reference environment with
`.venv/bin/pip install -e '.[dev,trl]'`. No model download is required:

```bash
HF_HUB_OFFLINE=1 .venv/bin/python examples/quickstart-trl/train_tiny.py --root .ptm
HF_HUB_OFFLINE=1 .venv/bin/python examples/quickstart-trl/train_tiny.py --deep --capture-update --loss dapo --root .ptm
OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 .venv/bin/python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29647 examples/quickstart-trl/train_distributed.py --output output/distributed-trl --root .ptm
.venv/bin/ptm preflight
.venv/bin/ptm serve .ptm --open
```

Use `TRLRewardCapture.wrap()` around one combined reward function. Dataset rows
provide explicit `ptm_group_id`; the context factory declares batch identity,
generation step and policy version. The distributed subclass gathers only IDs,
prompt hashes and sanitized context across the trusted local process group.
It validates global group size and prompt agreement before assigning sequence
indices; group identity is never inferred from order. Every rank writes its own
spool and health record. `ptm merge manifest.json` requires every rank and checks
shard checksums before import. Browser rank bundles carry explicit `world_size`
and `{worker_rank, content, drop_count, capture_errors, expected_group_ids}` shards.

verl JSONL begins with `{"ptm_run": <Run>, "field_map": {"group_id":
"metadata.explicit_group"}}`. Each following row explicitly declares batch,
generation_step, event_time, policy, group, prompt_id, expected_group_size,
completion_id, sequence_index, reward_total and reward_status. Source keys may
be mapped with dotted paths. Optional `token_blob` is already captured data;
text logs alone do not supply logprobs. A known PTM recipe must independently
support the requested calculation.

Wrappers return the exact original reward object even if optional capture
serialization fails. Whole-group deterministic sampling, a bounded writer queue,
drop/error counts and atomic compaction preserve the training return path.
The run and initial metadata are saved before training. Deep capture refuses
sampling or unsupported hooks instead of inventing a full batch. Native loss is
returned unchanged. The actual after-step weights are captured by a trainer
callback, separate from the PTM numerical replay implementation.

`full`, `off` and `hash_only` text retention apply before persistence, including
captured Token pieces and trajectory content. Numeric token IDs are retained;
this is not an anonymization guarantee. Mutations clear rewards, grades, lengths
and derived facts; users must rescore/regrade and review expected properties.
The tiny model reward is a token-ID proxy, not arithmetic or held-out accuracy.
