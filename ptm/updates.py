"""Bounded, real local optimizer replay from numeric state; never load pickle or model code."""

import json
import os
import platform
import shutil
import tempfile
import threading
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from pydantic import Field, model_validator

from ptm.errors import PTMError
from ptm.replay import AnalysisRequest, GroupFilter, RewardSpec, compute_diff
from ptm.schemas import Model
from ptm.storage import Store, atomic_write, canonical, digest, file_hash

LOCK = threading.Lock()
CONFIG_KEYS = [
    "vocab_size",
    "n_positions",
    "n_embd",
    "n_layer",
    "n_head",
    "activation_function",
    "resid_pdrop",
    "embd_pdrop",
    "attn_pdrop",
    "bos_token_id",
    "eos_token_id",
    "pad_token_id",
    "layer_norm_epsilon",
]
ALLOWED_FILES = {"model.safetensors", "reference.safetensors", "optimizer.safetensors", "capture.json"}


class GPT2Shape(Model):
    vocab_size: int = Field(ge=3, le=4096)
    n_positions: int = Field(ge=8, le=2048)
    n_embd: int = Field(ge=8, le=512)
    n_layer: int = Field(ge=1, le=4)
    n_head: int = Field(ge=1, le=16)
    activation_function: str = "gelu_new"
    resid_pdrop: float = Field(default=0, ge=0, le=1)
    embd_pdrop: float = Field(default=0, ge=0, le=1)
    attn_pdrop: float = Field(default=0, ge=0, le=1)
    bos_token_id: int | None = None
    eos_token_id: int | None = None
    pad_token_id: int | None = None
    layer_norm_epsilon: float = Field(default=1e-5, gt=0)

    @model_validator(mode="after")
    def supported(self) -> "GPT2Shape":
        if self.n_embd % self.n_head or self.activation_function not in {"gelu_new", "gelu", "relu", "silu"}:
            raise ValueError("Unsupported bounded GPT-2 configuration")
        return self


def reference_environment() -> dict[str, Any]:
    """Inspect optional package metadata without importing Torch on the web startup path."""
    required = {"torch": "2.9.0", "transformers": "4.57.1", "trl": "0.24.0", "accelerate": "1.11.0"}
    installed: dict[str, str | None] = {}
    for name in required:
        try:
            installed[name] = version(name)
        except PackageNotFoundError:
            installed[name] = None
    ready = all((installed[k] or "").split("+")[0] == v for k, v in required.items())
    return {"update_replay_available": ready, "required": required, "installed": installed,
            "device_scope": "CPU reference only", "reason": None if ready else
            "Install the pinned [trl] extra in a reference environment to execute optimizer replay."}


def capture_before(model: Any, optimizer: Any, args: Any) -> dict[str, Any]:
    import torch

    if model.config.model_type != "gpt2" or sum(p.numel() for p in model.parameters()) > 20_000_000:
        raise PTMError(
            "PTM_UPDATE_UNSUPPORTED",
            "Bounded reference replay supports local GPT-2 models up to 20M parameters",
        )
    native = getattr(optimizer, "optimizer", optimizer)
    if not isinstance(native, (torch.optim.AdamW, torch.optim.SGD)):
        raise PTMError("PTM_UPDATE_UNSUPPORTED", "Capture supports explicit AdamW or SGD state")
    config = GPT2Shape.model_validate({k: getattr(model.config, k) for k in CONFIG_KEYS}).model_dump()
    names = {id(p): name for name, p in model.named_parameters()}
    tensors = {"rng_cpu": torch.random.get_rng_state().clone()}
    groups = []
    states = {}
    for group in native.param_groups:
        record = {
            k: v
            for k, v in group.items()
            if k
            in {
                "lr",
                "betas",
                "eps",
                "weight_decay",
                "amsgrad",
                "momentum",
                "dampening",
                "nesterov",
                "maximize",
                "foreach",
                "fused",
                "capturable",
                "differentiable",
            }
        }
        record["param_names"] = [names[id(p)] for p in group["params"]]
        groups.append(record)
        for parameter in group["params"]:
            state: dict[str, Any] = {}
            for key, value in native.state.get(parameter, {}).items():
                if key not in {"step", "exp_avg", "exp_avg_sq", "max_exp_avg_sq", "momentum_buffer"}:
                    raise PTMError("PTM_UPDATE_UNSUPPORTED", "Unsupported optimizer state field")
                if isinstance(value, torch.Tensor):
                    tensor_key = names[id(parameter)] + "::" + key
                    tensors[tensor_key] = value.detach().cpu().clone().contiguous()
                    state[key] = {"tensor": tensor_key}
                elif isinstance(value, (int, float, bool)):
                    state[key] = value
                else:
                    raise PTMError("PTM_UPDATE_UNSUPPORTED", "Optimizer state is not a numeric tensor/scalar")
            if state:
                states[names[id(parameter)]] = state
    return {
        "config": config,
        "weights": {k: v.detach().cpu().clone().contiguous() for k, v in model.state_dict().items()},
        "optimizer_tensors": tensors,
        "optimizer": {
            "kind": "AdamW" if isinstance(native, torch.optim.AdamW) else "SGD",
            "groups": groups,
            "states": states,
            "max_grad_norm": args.max_grad_norm,
        },
        "temperature": args.temperature,
        "environment": {
            "torch": torch.__version__,
            "device": str(next(model.parameters()).device),
            "platform": platform.platform(),
        },
    }


def save_update_capture(store: Store, run_id: str, snapshot_id: str, collector: Any) -> str:
    from safetensors.torch import save_file

    if collector.before is None or collector.after is None or not collector.frames:
        raise PTMError(
            "PTM_UPDATE_CAPTURE_MISSING", "Before state, actual after state and loss inputs are required"
        )
    before = collector.before
    if before["environment"]["device"] != "cpu":
        raise PTMError("PTM_UPDATE_UNSUPPORTED", "This reference release validates CPU optimizer replay only")
    directory = store.path("update-captures")
    directory.mkdir(exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".pending-", dir=directory))
    try:
        save_file(before["weights"], stage / "model.safetensors")
        save_file(collector.after, stage / "reference.safetensors")
        save_file(before["optimizer_tensors"], stage / "optimizer.safetensors")
        metadata = {
            "format_version": 1,
            "run_id": run_id,
            "snapshot_id": snapshot_id,
            "model_type": "gpt2",
            "model_config": before["config"],
            "optimizer": before["optimizer"],
            "temperature": before["temperature"],
            "environment": before["environment"],
            "frame": collector.frames[0],
            "scope": "one captured optimizer step on fixed token IDs; no generation or future-quality prediction",
        }
        atomic_write(stage / "capture.json", canonical(metadata))
        hashes = {p.name: file_hash(p) for p in stage.iterdir()}
        capture_id = "upd_" + digest(hashes)
        manifest = {
            "capture_id": capture_id,
            "files": hashes,
            "run_id": run_id,
            "snapshot_id": snapshot_id,
            "optimizer_step": metadata["frame"]["optimizer_step"],
            "scope": metadata["scope"],
        }
        atomic_write(stage / "manifest.json", canonical(manifest))
        target = store.path("update-captures", capture_id)
        if not target.exists():
            os.rename(stage, target)
        return capture_id
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def read_capture(store: Store, capture_id: str) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    path = store.path("update-captures", capture_id)
    if not (path / "manifest.json").is_file():
        raise PTMError("PTM_NOT_FOUND", "Update capture not found", status=404)
    manifest = json.loads((path / "manifest.json").read_bytes())
    if set(manifest["files"]) != ALLOWED_FILES or "upd_" + digest(manifest["files"]) != capture_id:
        raise PTMError("PTM_INTEGRITY_ERROR", "Update capture file set or identity mismatch")
    for name, expected in manifest["files"].items():
        file = path / name
        if file.is_symlink() or file.stat().st_size > 256 * 1024 * 1024 or file_hash(file) != expected:
            raise PTMError("PTM_INTEGRITY_ERROR", "Update capture checksum/size/path mismatch")
    metadata = json.loads((path / "capture.json").read_bytes())
    GPT2Shape.model_validate(metadata["model_config"])
    return path, manifest, metadata


def list_captures(store: Store, run_id: str | None = None) -> list[dict[str, Any]]:
    results = []
    for p in store.path("update-captures").glob("upd_*"):
        if p.is_symlink() or not p.is_dir():
            continue
        _, manifest, metadata = read_capture(store, p.name)
        if run_id is None or manifest["run_id"] == run_id:
            results.append(
                {
                    **manifest,
                    "model_type": metadata["model_type"],
                    "environment": metadata["environment"],
                    "optimizer": metadata["optimizer"]["kind"],
                }
            )
    return results


def replay_update(store: Store, capture_id: str, spec: RewardSpec | None = None) -> dict[str, Any]:
    try:
        import torch
        import transformers
        import trl
        from safetensors.torch import load_file, save_file
        from transformers import GPT2Config, GPT2LMHeadModel
        from trl.trainer.utils import selective_log_softmax
    except ImportError:
        raise PTMError(
            "PTM_REFERENCE_ENV_MISSING", "Install the pinned [trl] reference extra to execute a real update"
        ) from None
    path, manifest, metadata = read_capture(store, capture_id)
    if (
        torch.__version__ != metadata["environment"]["torch"]
        or metadata["environment"]["device"] != "cpu"
        or transformers.__version__ != "4.57.1"
        or trl.__version__ != "0.24.0"
    ):
        raise PTMError(
            "PTM_UPDATE_ENV_MISMATCH", "Replay requires the captured Torch version and CPU environment"
        )
    identity = {
        "capture_id": capture_id,
        "reward_spec": spec.model_dump() if spec else None,
        "engine_revision": "bounded-update-v1",
    }
    aid = "update_" + digest(identity)
    target = store.path("update-analyses", aid)
    if (target / "result.json").exists():
        return json.loads((target / "result.json").read_bytes())
    with LOCK, torch.random.fork_rng(devices=[]):
        if (target / "result.json").exists():
            return json.loads((target / "result.json").read_bytes())
        torch.set_num_threads(1)
        model = GPT2LMHeadModel(GPT2Config(**GPT2Shape.model_validate(metadata["model_config"]).model_dump()))
        model.load_state_dict(load_file(path / "model.safetensors"), strict=True)
        parameters = dict(model.named_parameters())
        groups = []
        for group in metadata["optimizer"]["groups"]:
            names = group["param_names"]
            if not names or any(n not in parameters for n in names):
                raise PTMError("PTM_UPDATE_STATE_INVALID", "Optimizer parameter names do not match the model")
            opts = {k: v for k, v in group.items() if k != "param_names"}
            if any(
                k
                not in {
                    "lr",
                    "betas",
                    "eps",
                    "weight_decay",
                    "amsgrad",
                    "momentum",
                    "dampening",
                    "nesterov",
                    "maximize",
                    "foreach",
                    "fused",
                    "capturable",
                    "differentiable",
                }
                for k in opts
            ):
                raise PTMError("PTM_UPDATE_STATE_INVALID", "Unknown optimizer option")
            if "betas" in opts:
                opts["betas"] = tuple(opts["betas"])
            groups.append({"params": [parameters[n] for n in names], **opts})
        kind = metadata["optimizer"]["kind"]
        if kind not in {"AdamW", "SGD"}:
            raise PTMError("PTM_UPDATE_UNSUPPORTED", "Only numeric AdamW and SGD state is supported")
        optimizer = (torch.optim.AdamW if kind == "AdamW" else torch.optim.SGD)(groups)
        state_tensors = load_file(path / "optimizer.safetensors")
        for name, state in metadata["optimizer"]["states"].items():
            if name not in parameters:
                raise PTMError("PTM_UPDATE_STATE_INVALID", "Unknown optimizer state parameter")
            optimizer.state[parameters[name]] = {
                k: state_tensors[v["tensor"]].clone() if isinstance(v, dict) else v for k, v in state.items()
            }
        torch.random.set_rng_state(state_tensors["rng_cpu"])
        frame = metadata["frame"]
        raw = frame["native_inputs"]
        prompts = torch.tensor(raw["prompt_ids"], dtype=torch.long)
        completions = torch.tensor(raw["completion_ids"], dtype=torch.long)
        mask = torch.tensor(raw["completion_mask"], dtype=torch.float32)
        attention = torch.cat([torch.tensor(raw["prompt_mask"], dtype=torch.long), mask.long()], dim=1)
        if (
            prompts.ndim != 2
            or completions.ndim != 2
            or len(prompts) != len(completions)
            or len(prompts) > 64
            or prompts.shape[1] + completions.shape[1] > metadata["model_config"]["n_positions"]
        ):
            raise PTMError("PTM_UPDATE_STATE_INVALID", "Captured model input shape exceeds replay bounds")
        if torch.any(completions < 0) or torch.any(completions >= metadata["model_config"]["vocab_size"]):
            raise PTMError("PTM_UPDATE_STATE_INVALID", "Token IDs exceed the captured vocabulary")
        advantage_values = raw["advantages"]
        if spec is not None:
            snapshot = store.load(manifest["run_id"], manifest["snapshot_id"])
            by_id = {c.completion_id: c.group_id for c in snapshot.completions}
            group_ids = sorted({by_id[cid] for cid in frame["completion_ids"]})
            diff = compute_diff(
                snapshot,
                AnalysisRequest(
                    run_id=snapshot.run.run_id,
                    reward_spec=spec,
                    algorithm_recipe_id=snapshot.run.recipe_id,
                    group_filter=GroupFilter(group_ids=group_ids),
                ),
            )
            new_adv = {c["completion_id"]: c["advantage_after"] for c in diff["changes"]}
            advantage_values = [new_adv[cid] for cid in frame["completion_ids"]]
        advantages = torch.tensor(advantage_values, dtype=torch.float32).unsqueeze(1)
        model.train()
        logits = model(
            input_ids=torch.cat([prompts, completions], dim=1),
            attention_mask=attention,
            use_cache=False,
            logits_to_keep=completions.shape[1] + 1,
        ).logits[:, :-1, :]
        current = selective_log_softmax(logits / metadata["temperature"], completions)
        old = torch.tensor(frame["old"], dtype=current.dtype)
        ratio = torch.exp(current - old)
        per_token = -torch.minimum(
            ratio * advantages,
            torch.clamp(ratio, 1 - frame["epsilon_low"], 1 + frame["epsilon_high"]) * advantages,
        )
        if frame["beta"]:
            ref = torch.tensor(frame["reference"], dtype=current.dtype)
            delta = ref - current
            per_token = per_token + frame["beta"] * (torch.exp(delta) - delta - 1)
        if frame["loss_type"] == "grpo":
            loss = ((per_token * mask).sum(-1) / mask.sum(-1).clamp(min=1)).mean()
        elif frame["loss_type"] == "dr_grpo":
            loss = (per_token * mask).sum() / (len(prompts) * frame["max_completion_length"])
        else:
            loss = (per_token * mask).sum() / mask.sum().clamp(min=1)
        before = {k: p.detach().clone() for k, p in model.named_parameters()}
        optimizer.zero_grad()
        loss.backward()
        max_norm = metadata["optimizer"]["max_grad_norm"]
        if max_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        distance = float(
            torch.stack([((p.detach() - before[k]) ** 2).sum() for k, p in model.named_parameters()])
            .sum()
            .sqrt()
        )
        reference = load_file(path / "reference.safetensors")
        max_error = max(float((v.detach() - reference[k]).abs().max()) for k, v in model.state_dict().items())
        result = {
            "analysis_id": aid,
            "capture_id": capture_id,
            "run_id": manifest["run_id"],
            "optimizer_step": frame["optimizer_step"],
            "evidence": "replayed update" if distance > 0 else "replayed step; zero parameter delta",
            "loss": float(loss.detach()),
            "parameter_delta_l2": distance,
            "source_loss": frame["loss"],
            "source_loss_delta": float(loss.detach()) - frame["loss"],
            "max_reference_weight_error": max_error if spec is None else None,
            "reference_match": max_error <= 2e-6 if spec is None else None,
            "reference_tolerance": 2e-6,
            "parameters_changed": distance > 0,
            "reward_spec_hash": digest(spec.model_dump()) if spec else None,
            "scope": metadata["scope"],
            "limits": [
                "One actual optimizer step on fixed captured tokens, not a new rollout distribution.",
                "No claim about future accuracy, long-term training or general checkpoint compatibility.",
            ],
        }
        target.parent.mkdir(exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=".pending-", dir=target.parent))
        try:
            save_file(
                {k: v.detach().cpu().clone().contiguous() for k, v in model.state_dict().items()},
                stage / "replayed.safetensors",
            )
            atomic_write(stage / "result.json", canonical(result))
            if not target.exists():
                os.rename(stage, target)
        finally:
            if stage.exists():
                shutil.rmtree(stage)
        return result
