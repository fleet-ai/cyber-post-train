"""Exact dense runtime for the one-epoch 32K cyber-domain anchor."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import math
import re
import threading
from pathlib import Path

from training import sft_runtime as _base_runtime

RUNTIME_VARIANT = "qwen38_sft_32k_domain_anchor_v1"
RUNTIME_BINDING_SCHEMA = "cyber_sft_runtime_variant_v1"
RUN_NAME = "chris-q38-d32-b16-lr5e6-v1"
OUTPUT_ROOT = f"/mnt/sfs/jobs/{RUN_NAME}"
DENSE_IMAGE = (
    "661864827319.dkr.ecr.us-east-1.amazonaws.com/fleet/skyrl-train@sha256:"
    "ba288751cd227c5be146d28f4a03237545d87d2cbd4c48464945b17fde566ff4"
)
SEED = 20260921
OPTIMIZER = {
    "adam_betas": [0.9, 0.999],
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "scheduler": "constant_with_warmup",
    "num_warmup_steps": 0,
    "offload_after_step": False,
}
WANDB = {
    "entity": "thefleet",
    "project": "cyber-post-train",
    "group": "qwen38-sft-32k-domain-anchor-v1",
    "run_id": RUN_NAME,
    "name": RUN_NAME,
    "tags": [
        "qwen38",
        "teacher-sft",
        "broad-data",
        "57m-unique-supervised-tokens",
        "32k",
        "batch16",
        "lr5e-6",
        "one-epoch",
        "task-outcomes-only",
        "full-weight",
        "ctf-dojo-domain-anchor-v1",
        "seed20260921",
        "explicit-optimizer",
    ],
}
_SHA256 = re.compile(r"[a-f0-9]{64}")
_PATCH_LOCK = threading.RLock()
_BASE_VALIDATE_PLAN = _base_runtime.validate_plan
_BASE_SFT_OVERRIDES = _base_runtime.sft_overrides


def _file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def base_plan_binding() -> dict:
    """Return the only base-runtime identity admitted by this wrapper."""
    source = copy.deepcopy(_base_runtime.qwen38_lora_broad_full_plan_binding())
    source.update(
        {
            "plan_keys": [
                "checkpoint_recovery_horizon_seconds",
                "corpus_manifest_sha256",
                "datasets",
                "execution",
                "model",
                "output_root",
                "recipe",
                "run_name",
                "runtime_sha256",
                "schema",
                "split_manifest_sha256",
                "validation_mode",
                "wandb",
            ],
            "run_name": RUN_NAME,
            "output_root": OUTPUT_ROOT,
            "recipe": {
                "epochs": 1,
                "batch_size": 16,
                "microbatch_per_gpu": 1,
                "nodes": 1,
                "gpus_per_node": 8,
                "lr": 5e-6,
                "max_length": 32768,
                "eval_interval": 0,
                "checkpoint_interval": 50,
                "keep_checkpoints": 2,
                "seed": SEED,
                "max_steps": 919,
            },
            "wandb": copy.deepcopy(WANDB),
        }
    )
    return source


def runtime_binding(base_runtime_path: Path | None = None) -> dict[str, str]:
    path = base_runtime_path or Path(_base_runtime.__file__)
    return {
        "schema": RUNTIME_BINDING_SCHEMA,
        "name": RUNTIME_VARIANT,
        "base_runtime_sha256": _file_sha256(path),
    }


def delegated_plan(plan: dict) -> dict:
    delegated = copy.deepcopy(plan)
    delegated.pop("runtime_variant", None)
    delegated.pop("optimizer", None)
    delegated["runtime_sha256"] = runtime_binding()["base_runtime_sha256"]
    return delegated


def validate_optimizer(value: object) -> None:
    if not isinstance(value, dict) or set(value) != set(OPTIMIZER):
        raise ValueError("domain-anchor optimizer binding is incomplete")
    if value.get("adam_betas") != OPTIMIZER["adam_betas"]:
        raise ValueError("domain-anchor Adam betas drifted")
    for key in ("weight_decay", "max_grad_norm"):
        number = value.get(key)
        if type(number) not in {int, float} or not math.isfinite(number):
            raise ValueError("domain-anchor optimizer scalar is invalid")
    if value != OPTIMIZER:
        raise ValueError("domain-anchor optimizer binding drifted")


def validate_runtime_binding(
    plan: dict,
    *,
    runtime_path: Path | None = None,
    base_runtime_path: Path | None = None,
    check_files: bool = True,
) -> None:
    binding = plan.get("runtime_variant")
    if (
        not isinstance(binding, dict)
        or set(binding) != {"schema", "name", "base_runtime_sha256"}
        or binding.get("schema") != RUNTIME_BINDING_SCHEMA
        or binding.get("name") != RUNTIME_VARIANT
        or _SHA256.fullmatch(binding.get("base_runtime_sha256", "")) is None
        or _SHA256.fullmatch(plan.get("runtime_sha256", "")) is None
        or plan["runtime_sha256"] == binding["base_runtime_sha256"]
        or plan.get("run_name") != RUN_NAME
        or "lora" in plan
        or "recovery" in plan
        or "pause_after_step" in plan
        or plan.get("model", {}).get("repo") != "Qwen/Qwen3.8-27B"
        or plan.get("execution", {}).get("image") != DENSE_IMAGE
        or plan.get("checkpoint_recovery_horizon_seconds") != 53300
    ):
        raise ValueError("32K domain-anchor runtime binding is invalid")
    validate_optimizer(plan.get("optimizer"))
    if _base_runtime._qwen38_lora_one_step_identity(delegated_plan(plan)) != base_plan_binding():
        raise ValueError("32K domain-anchor scientific identity drifted")
    if check_files:
        wrapper = runtime_path or Path(__file__)
        base = base_runtime_path or Path(_base_runtime.__file__)
        if (
            _file_sha256(wrapper) != plan["runtime_sha256"]
            or _file_sha256(base) != binding["base_runtime_sha256"]
        ):
            raise ValueError("32K domain-anchor runtime source digest mismatch")


def validate_plan(plan: dict, *, check_files: bool = True) -> None:
    validate_runtime_binding(plan, check_files=check_files)
    _BASE_VALIDATE_PLAN(delegated_plan(plan), check_files=check_files)


def sft_overrides(plan: dict) -> dict:
    validate_runtime_binding(plan, check_files=False)
    options = _BASE_SFT_OVERRIDES(delegated_plan(plan))
    optimizer = plan["optimizer"]
    options.update(
        {
            "optimizer_config.adam_betas": list(optimizer["adam_betas"]),
            "optimizer_config.weight_decay": optimizer["weight_decay"],
            "optimizer_config.max_grad_norm": optimizer["max_grad_norm"],
            "optimizer_config.scheduler": optimizer["scheduler"],
            "optimizer_config.num_warmup_steps": optimizer["num_warmup_steps"],
            "optimizer_config.offload_after_step": optimizer["offload_after_step"],
        }
    )
    return options


@contextlib.contextmanager
def execution_context():
    with _PATCH_LOCK:
        previous_validate = _base_runtime.validate_plan
        previous_overrides = _base_runtime.sft_overrides
        _base_runtime.validate_plan = validate_plan
        _base_runtime.sft_overrides = sft_overrides
        try:
            yield
        finally:
            _base_runtime.validate_plan = previous_validate
            _base_runtime.sft_overrides = previous_overrides


def main() -> None:
    with execution_context():
        _base_runtime.main()


if __name__ == "__main__":
    main()
