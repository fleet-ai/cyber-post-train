"""Small GLM-MoE-DSA LoRA primitives; no launcher and no implicit full-size gate.

The caller must independently validate the immutable base payload/sidecars and
training plan. This module checks their identity binding, loads only rank zero
on CPU, keeps other ranks on meta, and saves no frozen base weights. Distributed
FSDP2 calls are implemented with PyTorch state-dict APIs but require a GPU gate;
the synthetic CPU tests do not certify model size, GPU kernels, or throughput.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import inspect
import json
import math
import random
import re
import threading
from pathlib import Path

TARGETS = ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj")
FP8_INTEGRATION_SHA256 = "cad925d1ceae22b1bf9ad769971e5d53b7829b73fd00ddf0959a8818763dec4b"
_DEQUANTIZE_LOCK = threading.Lock()


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted(_jsonable(v) for v in value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def canonical(value) -> bytes:
    return json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _sha(value: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[a-f0-9]{64}", value) is None:
        raise ValueError("expected an exact SHA-256")
    return value


@dataclasses.dataclass(frozen=True)
class BaseIdentity:
    repository: str
    revision: str
    config_sha256: str
    weights_manifest_sha256: str
    tokenizer_manifest_sha256: str

    def __post_init__(self):
        if (
            self.repository != "zai-org/GLM-5.3"
            or re.fullmatch(r"[a-f0-9]{40}", self.revision) is None
        ):
            raise ValueError("expected exact full GLM-5.3 identity, not Flash or mutable revision")
        for value in (
            self.config_sha256,
            self.weights_manifest_sha256,
            self.tokenizer_manifest_sha256,
        ):
            _sha(value)


def bf16_meta_parameters(model):
    """Mirror HF BF16 load while preserving strict FP32 and nonpersistent buffers."""
    import torch

    strict = set(getattr(model, "_keep_in_fp32_modules_strict", ()) or ())
    nonpersistent = set()
    for prefix, module in model.named_modules():
        for name in getattr(module, "_non_persistent_buffers_set", ()):
            nonpersistent.add(f"{prefix}.{name}" if prefix else name)
    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        if name in nonpersistent or not tensor.dtype.is_floating_point:
            continue
        dtype = torch.float32 if strict.intersection(name.split(".")) else torch.bfloat16
        tensor.data = tensor.data.to(dtype)
    return model


@contextlib.contextmanager
def bf16_fp8_conversion():
    """Exact-runtime, process-scoped per-tensor cast before HF expert merging.

    Transformers 5.8 chooses scale dtype, not requested model dtype. FP32
    scales otherwise make FP32 weights and double the dominant base allocation.
    This hook preserves the official FP32 arithmetic then rounds each result to
    BF16 before it enters the model. Use a dedicated single-loader process;
    this does not modify installed files, and the original method is restored.
    """
    import torch
    from transformers.integrations import finegrained_fp8

    if sha_file(Path(inspect.getfile(finegrained_fp8))) != FP8_INTEGRATION_SHA256:
        raise ValueError("unqualified FP8 integration source; revalidate before use")
    with _DEQUANTIZE_LOCK:
        original = finegrained_fp8.Fp8Dequantize._dequantize_one

        def convert(operation, weight, scale):
            # HF may already widen E4M3 to BF16 before the conversion op.
            # Source E4M3 provenance is checked by the caller's payload audit.
            if weight.dtype not in {torch.float8_e4m3fn, torch.bfloat16, torch.float32}:
                raise ValueError("only the bound E4M3 FP8 payload is qualified")
            # The pinned integration infers block size by dividing weight/grid
            # dimensions. GLM's 576-row KV projection has five 128-row blocks:
            # its final block has 64 rows, so that inference raises. Apply the
            # exact base's declared 128x128 scales, cropping only the edge block.
            if weight.ndim == scale.ndim == 2:
                rows, cols = weight.shape
                sr, sc = scale.shape
                if (sr, sc) != (math.ceil(rows / 128), math.ceil(cols / 128)):
                    raise ValueError("FP8 scale grid does not match the bound 128x128 blocks")
                if rows % 128 or cols % 128:
                    expanded = scale.float().repeat_interleave(128, 0).repeat_interleave(128, 1)
                    return (weight.float() * expanded[:rows, :cols]).to(torch.bfloat16)
            return original(operation, weight, scale).to(torch.bfloat16)

        finegrained_fp8.Fp8Dequantize._dequantize_one = convert
        try:
            yield
        finally:
            finegrained_fp8.Fp8Dequantize._dequantize_one = original


def load_lora_model(
    root: Path,
    identity: BaseIdentity,
    *,
    meta_init: bool,
    rank: int = 16,
    alpha: int = 32,
    attention: str = "sdpa",
):
    """Load a previously verified FP8 base as frozen BF16; optimizer is independent.

    ``meta_init`` is chosen by the caller's FSDP mesh, not by local CUDA rank.
    Never set SkyRL inference_only_init to enable this path: it disables training.
    """
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoConfig, AutoModelForCausalLM, FineGrainedFP8Config

    root = Path(root)
    if (
        root.is_symlink()
        or not root.is_dir()
        or sha_file(root / "config.json") != identity.config_sha256
    ):
        raise ValueError("base config identity mismatch")
    if type(rank) is not int or not 1 <= rank <= 64 or type(alpha) is not int or alpha <= 0:
        raise ValueError("invalid bounded adapter rank/alpha")
    if attention not in {"eager", "sdpa"}:
        raise ValueError("unqualified attention implementation")
    config = AutoConfig.from_pretrained(str(root), local_files_only=True, trust_remote_code=False)
    if config.model_type != "glm_moe_dsa":
        raise ValueError("wrong architecture")
    quant = getattr(config, "quantization_config", {})
    if (
        quant.get("quant_method") != "fp8"
        or quant.get("fmt", "e4m3") != "e4m3"
        or list(quant.get("weight_block_size", [])) != [128, 128]
    ):
        raise ValueError("base is not the bound block-FP8 artifact")
    config._attn_implementation = attention
    if meta_init:
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=False)
        bf16_meta_parameters(model)
    else:
        dequant = FineGrainedFP8Config(
            dequantize=True, weight_block_size=(128, 128), activation_scheme="dynamic"
        )
        with bf16_fp8_conversion():
            model, loading = AutoModelForCausalLM.from_pretrained(
                str(root),
                config=config,
                local_files_only=True,
                trust_remote_code=False,
                dtype=torch.bfloat16,
                device_map={"": "cpu"},
                quantization_config=dequant,
                attn_implementation=attention,
                output_loading_info=True,
            )
        if any(
            loading.get(key)
            for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
        ):
            raise ValueError("base loader reported missing, extra, mismatched, or failed weights")
    model.requires_grad_(False)
    model.config.use_cache = False
    adapted = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=rank,
            lora_alpha=alpha,
            lora_dropout=0.0,
            bias="none",
            target_modules=list(TARGETS),
        ),
    )
    # Required for gradient checkpointing through a frozen embedding/base.
    adapted.enable_input_require_grads()
    adapter_parameters(adapted)
    return adapted


def adapter_parameters(model) -> dict:
    selected = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if not selected or any(".lora_" not in n or ".indexer." in n for n in selected):
        raise ValueError("only attention adapters may be trainable")
    return selected


def make_optimizer(model, *, lr: float = 1e-5):
    import torch

    if not math.isfinite(lr) or not 0 < lr <= 0.01:
        raise ValueError("invalid learning rate")
    return torch.optim.AdamW(adapter_parameters(model).values(), lr=lr)


def _topology():
    import torch.distributed as dist

    return (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)


def _rank0(function):
    """Synchronize preflight/I/O failures rather than strand the other ranks."""
    import torch.distributed as dist

    rank, size = _topology()
    message = [None]
    if rank == 0:
        try:
            message[0] = {"ok": True, "value": function()}
        except Exception as exc:
            message[0] = {"ok": False, "error": type(exc).__name__}
    if size > 1:
        dist.broadcast_object_list(message, src=0)
    if not message[0]["ok"]:
        raise ValueError("checkpoint rank-zero operation failed: " + message[0]["error"])
    return message[0]["value"]


def _validate_optimizer(model, optimizer):
    expected = {id(p) for p in adapter_parameters(model).values()}
    got = [id(p) for group in optimizer.param_groups for p in group["params"]]
    if len(got) != len(set(got)) or set(got) != expected:
        raise ValueError("optimizer must contain exactly the trainable adapters")


def _layout(model):
    return {
        n: {"shape": list(p.shape), "dtype": str(p.dtype)}
        for n, p in adapter_parameters(model).items()
    }


def _rng():
    import torch

    return {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _write_json(path, value):
    import os

    with Path(path).open("xb") as stream:
        stream.write(canonical(value))
        stream.flush()
        os.fsync(stream.fileno())


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    scheduler,
    *,
    identity: BaseIdentity,
    plan_sha256: str,
    optimizer_step: int,
    next_batch: int,
    epoch: int,
    at_optimizer_boundary: bool,
) -> dict:
    """Collect adapters/optimizer only. The COMPLETE marker is published last.

    Every FSDP rank must call together, on the same mesh. Partial destinations
    are preserved and cannot be overwritten. No base weight enters this format.
    """
    import os

    import torch
    import torch.distributed as dist
    from safetensors.torch import save_file
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_optimizer_state_dict

    _sha(plan_sha256)
    _validate_optimizer(model, optimizer)
    if at_optimizer_boundary is not True or any(
        type(x) is not int or x < 0 for x in (optimizer_step, next_batch, epoch)
    ):
        raise ValueError("checkpoint requires a complete optimizer boundary and numeric cursor")
    path = Path(path)
    rank, size = _topology()
    _rank0(lambda: path.mkdir(mode=0o700, parents=False, exist_ok=False))
    adapters = {}
    for name, value in adapter_parameters(model).items():
        tensor = value.detach()
        if hasattr(tensor, "full_tensor"):
            tensor = tensor.full_tensor()
        if rank == 0:
            adapters[name] = tensor.cpu().contiguous()
    optim = get_optimizer_state_dict(
        model, optimizer, options=StateDictOptions(full_state_dict=True, cpu_offload=True)
    )
    local_rng = _rng()
    states = [None] * size
    if size > 1:
        dist.all_gather_object(states, local_rng)
    else:
        states[0] = local_rng

    def publish():
        save_file(adapters, str(path / "adapter_tensors.safetensors"))
        torch.save(
            {"optimizer": optim, "scheduler": scheduler.state_dict(), "rng_by_rank": states},
            path / "training_state.pt",
        )
        peft = _jsonable(model.peft_config["default"].to_dict())
        _write_json(path / "peft_config.json", peft)
        payloads = {}
        for name in ("adapter_tensors.safetensors", "training_state.pt", "peft_config.json"):
            f = path / name
            with f.open("rb") as stream:
                os.fsync(stream.fileno())
            payloads[name] = {"sha256": sha_file(f), "bytes": f.stat().st_size}
        receipt = {
            "schema": "glm53_lora_resumable_checkpoint_v1",
            "base": dataclasses.asdict(identity),
            "plan_sha256": plan_sha256,
            "world_size": size,
            "optimizer_step": optimizer_step,
            "next_batch": next_batch,
            "epoch": epoch,
            "adapter_layout": _layout(model),
            "peft_config": peft,
            "payloads": payloads,
            "frozen_base_saved": False,
        }
        receipt["receipt_sha256"] = hashlib.sha256(canonical(receipt)).hexdigest()
        _write_json(path / "COMPLETE.json", receipt)
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        return receipt

    return _rank0(publish)


def _verify(path: Path, identity, plan_sha256, layout, peft, size):
    if path.is_symlink() or (path / "COMPLETE.json").is_symlink():
        raise ValueError("symlink checkpoint")
    receipt = json.loads((path / "COMPLETE.json").read_text())
    expected = receipt.pop("receipt_sha256")
    if hashlib.sha256(canonical(receipt)).hexdigest() != expected:
        raise ValueError("checkpoint receipt digest mismatch")
    receipt["receipt_sha256"] = expected
    if (
        receipt["schema"] != "glm53_lora_resumable_checkpoint_v1"
        or receipt["base"] != dataclasses.asdict(identity)
        or receipt["plan_sha256"] != plan_sha256
        or receipt["world_size"] != size
        or receipt["adapter_layout"] != layout
        or receipt["peft_config"] != peft
        or receipt["frozen_base_saved"] is not False
    ):
        raise ValueError("checkpoint identity, layout, topology or recipe mismatch")
    names = {"adapter_tensors.safetensors", "training_state.pt", "peft_config.json"}
    if set(receipt["payloads"]) != names or {p.name for p in path.iterdir()} != names | {
        "COMPLETE.json"
    }:
        raise ValueError("unexpected or missing checkpoint payload")
    for name, spec in receipt["payloads"].items():
        f = path / name
        if f.is_symlink() or f.stat().st_size != spec["bytes"] or sha_file(f) != spec["sha256"]:
            raise ValueError("checkpoint payload integrity mismatch")
    return receipt


def load_checkpoint(
    path: Path, model, optimizer, scheduler, *, identity: BaseIdentity, plan_sha256: str
) -> dict:
    """Restore the next update, not just adapter weights; require identical mesh."""
    import torch
    import torch.distributed as dist
    from safetensors.torch import load_file
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        set_model_state_dict,
        set_optimizer_state_dict,
    )

    _sha(plan_sha256)
    _validate_optimizer(model, optimizer)
    rank, size = _topology()
    path = Path(path)
    receipt = _rank0(
        lambda: _verify(
            path,
            identity,
            plan_sha256,
            _layout(model),
            _jsonable(model.peft_config["default"].to_dict()),
            size,
        )
    )
    tensors, state = {}, None

    def decode():
        nonlocal tensors, state
        tensors = load_file(str(path / "adapter_tensors.safetensors"), device="cpu")
        state = torch.load(path / "training_state.pt", map_location="cpu", weights_only=True)
        if set(tensors) != set(receipt["adapter_layout"]):
            raise ValueError("unexpected adapter tensor names")
        for name, tensor in tensors.items():
            if {"shape": list(tensor.shape), "dtype": str(tensor.dtype)} != receipt[
                "adapter_layout"
            ][name] or not torch.isfinite(tensor).all():
                raise ValueError("adapter payload layout or numeric defect")
        if (
            set(state) != {"optimizer", "scheduler", "rng_by_rank"}
            or len(state["rng_by_rank"]) != size
        ):
            raise ValueError("training state structure or RNG topology mismatch")
        return {"scheduler": state["scheduler"], "rng_by_rank": state["rng_by_rank"]}

    metadata = _rank0(decode)
    rng = metadata["rng_by_rank"][rank]
    expected_cuda = torch.cuda.device_count() if torch.cuda.is_available() else 0
    valid_rng = set(rng) == {"python", "torch", "cuda"} and len(rng["cuda"]) == expected_cuda
    checks = [None] * size
    if size > 1:
        dist.all_gather_object(checks, valid_rng)
    else:
        checks[0] = valid_rng
    if not all(checks):
        raise ValueError("CUDA RNG topology mismatch")
    options = StateDictOptions(full_state_dict=True, broadcast_from_rank0=(size > 1), strict=False)
    set_model_state_dict(model, tensors, options=options)
    set_optimizer_state_dict(
        model, optimizer, state["optimizer"] if rank == 0 else {}, options=options
    )
    scheduler.load_state_dict(metadata["scheduler"])
    random.setstate(rng["python"])
    torch.set_rng_state(rng["torch"])
    if rng["cuda"]:
        torch.cuda.set_rng_state_all(rng["cuda"])
    return receipt


def identity_from_plan(plan: dict) -> BaseIdentity:
    model = plan["model"]
    config = next(f for f in model["files"] if f["path"] == "config.json")
    return BaseIdentity(
        model["repo"],
        model["revision"],
        config["sha256"].removeprefix("sha256:"),
        model["weight_manifest_sha256"].removeprefix("sha256:"),
        model["tokenizer_manifest_sha256"].removeprefix("sha256:"),
    )


def worker_class(plan: dict):
    """Use the native FSDP worker, changing only base loading and adapter saves.

    The scoped constructor replacement is process-local inside one Ray actor.
    Native FSDP sharding, loss, backward, optimizer/scheduler and dispatch remain
    unchanged. This module must be present at the same bound shared path on every
    worker; the Jobs bundle supplies it, not a mutable pip installation.
    """
    import math

    from skyrl.backends.skyrl_train.workers.fsdp import fsdp_worker

    identity = identity_from_plan(plan)
    lora = plan["lora"]

    class GlmLoraWorker(fsdp_worker.FSDPPolicyWorkerBase):
        def init_model(self, model_path, num_training_steps=None):
            if sha_file(Path(__file__)) != plan["glm_runtime_sha256"]:
                raise ValueError("worker GLM runtime differs from frozen plan")
            if self.cfg.policy.inference_only_init:
                raise ValueError("LoRA training must not disable its optimizer")
            original = fsdp_worker.HFModelWrapper

            def wrap(path, **kwargs):
                if path != plan["model"]["root"]:
                    raise ValueError("worker base path mismatch")
                model = load_lora_model(
                    Path(path),
                    identity,
                    meta_init=kwargs["meta_init"],
                    rank=lora["rank"],
                    alpha=lora["alpha"],
                    attention="sdpa",
                )
                return original(model, **kwargs)

            fsdp_worker.HFModelWrapper = wrap
            try:
                super().init_model(model_path, num_training_steps=num_training_steps)
            finally:
                fsdp_worker.HFModelWrapper = original
            # The native optimizer is still empty immediately after creation.
            # Remove frozen params before its first step or checkpoint; retain
            # the exact native AdamW configuration and scheduler object.
            if self.optimizer is None or self.optimizer.state:
                raise ValueError("expected a fresh native LoRA optimizer")
            for group in self.optimizer.param_groups:
                group["params"] = [p for p in group["params"] if p.requires_grad]
            _validate_optimizer(self.model.model, self.optimizer)
            self._completed_optimizer_steps = 0

        def optim_step(self, *args, **kwargs):
            result = super().optim_step(*args, **kwargs)
            self._completed_optimizer_steps += 1
            steps = [float(s["step"].item()) for s in self.optimizer.state.values() if "step" in s]
            if not steps or any(s != self._completed_optimizer_steps for s in steps):
                raise ValueError("native adapter optimizer counter mismatch")
            return result

        def save_checkpoint(self, ckpt_dir, tokenizer=None):
            path = Path(ckpt_dir)
            step = self._completed_optimizer_steps
            if path.name != "policy" or path.parent.name != f"global_step_{step}" or step <= 0:
                raise ValueError("checkpoint path differs from actual adapter optimizer step")
            per_epoch = math.ceil(plan["datasets"]["train"]["rows"] / plan["recipe"]["batch_size"])
            return save_checkpoint(
                path,
                self.model.model,
                self.optimizer,
                self.scheduler,
                identity=identity,
                plan_sha256=plan["plan_sha256"],
                optimizer_step=step,
                next_batch=(step - 1) % per_epoch + 1,
                epoch=(step - 1) // per_epoch,
                at_optimizer_boundary=True,
            )

        def load_checkpoint(
            self, ckpt_dir, load_optimizer_states=True, load_lr_scheduler_states=True
        ):
            if not load_optimizer_states or not load_lr_scheduler_states:
                raise ValueError("adapter resume must restore optimizer and scheduler")
            result = load_checkpoint(
                Path(ckpt_dir),
                self.model.model,
                self.optimizer,
                self.scheduler,
                identity=identity,
                plan_sha256=plan.get("recovery", {})
                .get("checkpoint", {})
                .get("source_plan_sha256", plan["plan_sha256"]),
            )
            self._completed_optimizer_steps = result["optimizer_step"]
            return result

        def save_hf_model(self, *args, **kwargs):
            raise ValueError("full frozen-base export is not an adapter checkpoint")

    return GlmLoraWorker


@contextlib.contextmanager
def use_worker(plan: dict):
    """Scope one native trainer's worker selection; never patch installed files."""
    import ray
    from skyrl.backends.skyrl_train.workers.fsdp import fsdp_worker

    original = fsdp_worker.PolicyWorker
    fsdp_worker.PolicyWorker = ray.remote(num_gpus=1)(worker_class(plan))
    try:
        yield
    finally:
        fsdp_worker.PolicyWorker = original
