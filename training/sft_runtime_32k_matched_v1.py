"""Versioned runtime for the matched 32K dense-versus-LoRA SFT pair.

The historical runtime bytes remain bound to prior plans and receipts.  This
wrapper admits exactly two fresh create-once identities, adds one explicit
optimizer contract to their plans, and delegates every training operation to
the unchanged qualified runtime.  Full-weight training remains native FSDP and
LoRA remains native Megatron TP8; this wrapper does not pretend those runtimes
are the same treatment.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import math
import re
import threading
from pathlib import Path

from training import sft_runtime as _base_runtime

RUNTIME_VARIANT = "qwen38_sft_32k_matched_v1"
RUNTIME_BINDING_SCHEMA = "cyber_sft_runtime_variant_v1"
DENSE_RUN_NAME = "chris-q38-sft32-dense-m1-v1"
LORA_RUN_NAME = "chris-q38-sft32-lora-m1-v1"
DENSE_OUTPUT_ROOT = f"/mnt/sfs/jobs/{DENSE_RUN_NAME}"
LORA_OUTPUT_ROOT = f"/mnt/sfs/jobs/{LORA_RUN_NAME}"
DENSE_IMAGE = (
    "661864827319.dkr.ecr.us-east-1.amazonaws.com/fleet/skyrl-train@sha256:"
    "ba288751cd227c5be146d28f4a03237545d87d2cbd4c48464945b17fde566ff4"
)
MATCHED_SEED = 20260921
MATCHED_OPTIMIZER = {
    "adam_betas": [0.9, 0.999],
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "scheduler": "constant_with_warmup",
    "num_warmup_steps": 0,
    "offload_after_step": False,
}
_SHA256 = re.compile(r"[a-f0-9]{64}")
_PATCH_LOCK = threading.RLock()
_BASE_VALIDATE_PLAN = _base_runtime.validate_plan
_BASE_SFT_OVERRIDES = _base_runtime.sft_overrides
_BASE_VERIFY_PRODUCTION_QUALIFICATION = _base_runtime._verify_qwen38_production_qualification

DENSE_WANDB = {
    "entity": "thefleet",
    "project": "cyber-post-train",
    "group": "qwen38-sft-32k-dense-lora-matched-v1",
    "run_id": DENSE_RUN_NAME,
    "name": DENSE_RUN_NAME,
    "tags": [
        "qwen38",
        "teacher-sft",
        "broad-data",
        "57m-unique-supervised-tokens",
        "32k",
        "batch8",
        "lr3e-6",
        "one-epoch",
        "task-outcomes-only",
        "full-weight",
        "matched-dense-lora-v1",
        "seed20260921",
        "explicit-optimizer",
    ],
}
LORA_WANDB = {
    "entity": "thefleet",
    "project": "cyber-post-train",
    "group": "qwen38-sft-32k-dense-lora-matched-v1",
    "run_id": LORA_RUN_NAME,
    "name": LORA_RUN_NAME,
    "tags": [
        "qwen38",
        "lora",
        "teacher-sft",
        "rank64",
        "alpha32",
        "57m-unique-supervised-tokens",
        "32k",
        "batch8",
        "lr3e-5",
        "one-epoch",
        "task-outcomes-only",
        "matched-dense-lora-v1",
        "seed20260921",
        "explicit-optimizer",
    ],
}


def _file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def lora_base_plan_binding() -> dict:
    """Return the one base-runtime LoRA identity admitted by this wrapper."""
    source = copy.deepcopy(_base_runtime.qwen38_lora_broad_full_plan_binding())
    source["run_name"] = LORA_RUN_NAME
    source["output_root"] = LORA_OUTPUT_ROOT
    source["recipe"] = {**source["recipe"], "seed": MATCHED_SEED}
    source["wandb"] = copy.deepcopy(LORA_WANDB)
    return source


def dense_base_plan_binding() -> dict:
    """Return the exact compact identity of the matched dense base plan."""
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
            "run_name": DENSE_RUN_NAME,
            "output_root": DENSE_OUTPUT_ROOT,
            "recipe": {
                "epochs": 1,
                "batch_size": 8,
                "microbatch_per_gpu": 1,
                "nodes": 1,
                "gpus_per_node": 8,
                "lr": 3e-6,
                "max_length": 32768,
                "eval_interval": 0,
                "checkpoint_interval": 100,
                "keep_checkpoints": 2,
                "seed": MATCHED_SEED,
                "max_steps": 1837,
            },
            "wandb": copy.deepcopy(DENSE_WANDB),
        }
    )
    return source


def runtime_binding(base_runtime_path: Path | None = None) -> dict[str, str]:
    """Bind this wrapper to exactly the unchanged historical runtime bytes."""
    path = base_runtime_path or Path(_base_runtime.__file__)
    return {
        "schema": RUNTIME_BINDING_SCHEMA,
        "name": RUNTIME_VARIANT,
        "base_runtime_sha256": _file_sha256(path),
    }


@contextlib.contextmanager
def base_plan_context():
    """Expose the fresh LoRA identity only inside this versioned wrapper."""
    with _PATCH_LOCK:
        original = _base_runtime.QWEN38_LORA_BROAD_FULL_PLANS
        expected = lora_base_plan_binding()
        if LORA_RUN_NAME in original:
            if original[LORA_RUN_NAME] != expected:
                raise ValueError("matched LoRA binding drifted in memory")
            yield original
            return
        replacement = {**original, LORA_RUN_NAME: expected}
        _base_runtime.QWEN38_LORA_BROAD_FULL_PLANS = replacement
        try:
            yield replacement
        finally:
            _base_runtime.QWEN38_LORA_BROAD_FULL_PLANS = original


def delegated_plan(plan: dict) -> dict:
    """Remove only versioned-wrapper metadata before base validation."""
    delegated = copy.deepcopy(plan)
    delegated.pop("runtime_variant", None)
    delegated.pop("optimizer", None)
    delegated["runtime_sha256"] = runtime_binding()["base_runtime_sha256"]
    return delegated


def _validate_optimizer(value: object) -> None:
    if not isinstance(value, dict) or set(value) != set(MATCHED_OPTIMIZER):
        raise ValueError("matched SFT optimizer binding is incomplete")
    if value.get("adam_betas") != MATCHED_OPTIMIZER["adam_betas"]:
        raise ValueError("matched SFT Adam betas drifted")
    for key in ("weight_decay", "max_grad_norm"):
        number = value.get(key)
        if type(number) not in {int, float} or not math.isfinite(number):
            raise ValueError("matched SFT optimizer scalar is invalid")
    if value != MATCHED_OPTIMIZER:
        raise ValueError("matched SFT optimizer binding drifted")


def validate_runtime_binding(
    plan: dict,
    *,
    runtime_path: Path | None = None,
    base_runtime_path: Path | None = None,
    check_files: bool = True,
) -> None:
    """Fail closed unless the wrapper, optimizer, and exact arm agree."""
    binding = plan.get("runtime_variant")
    expected_identity = {
        DENSE_RUN_NAME: dense_base_plan_binding(),
        LORA_RUN_NAME: lora_base_plan_binding(),
    }.get(plan.get("run_name"))
    if (
        not isinstance(binding, dict)
        or set(binding) != {"schema", "name", "base_runtime_sha256"}
        or binding.get("schema") != RUNTIME_BINDING_SCHEMA
        or binding.get("name") != RUNTIME_VARIANT
        or _SHA256.fullmatch(binding.get("base_runtime_sha256", "")) is None
        or _SHA256.fullmatch(plan.get("runtime_sha256", "")) is None
        or plan["runtime_sha256"] == binding["base_runtime_sha256"]
        or expected_identity is None
        or "recovery" in plan
        or "pause_after_step" in plan
        or plan.get("model", {}).get("repo") != "Qwen/Qwen3.8-27B"
    ):
        raise ValueError("matched 32K SFT runtime binding is invalid")
    _validate_optimizer(plan.get("optimizer"))
    delegated = delegated_plan(plan)
    if _base_runtime._qwen38_lora_one_step_identity(delegated) != expected_identity:
        raise ValueError("matched 32K SFT scientific identity drifted")
    if plan["run_name"] == DENSE_RUN_NAME:
        if (
            "lora" in plan
            or plan.get("execution", {}).get("image") != DENSE_IMAGE
            or plan.get("checkpoint_recovery_horizon_seconds") != 53300
        ):
            raise ValueError("matched dense runtime identity drifted")
    elif (
        plan.get("lora") != _base_runtime.QWEN38_LORA
        or "checkpoint_recovery_horizon_seconds" in plan
    ):
        raise ValueError("matched LoRA runtime identity drifted")
    if check_files:
        wrapper = runtime_path or Path(__file__)
        base = base_runtime_path or Path(_base_runtime.__file__)
        if (
            _file_sha256(wrapper) != plan["runtime_sha256"]
            or _file_sha256(base) != binding["base_runtime_sha256"]
        ):
            raise ValueError("matched 32K SFT runtime source digest mismatch")


def validate_plan(plan: dict, *, check_files: bool = True) -> None:
    """Validate the wrapper contract, then the unchanged base plan."""
    validate_runtime_binding(plan, check_files=check_files)
    with base_plan_context():
        _BASE_VALIDATE_PLAN(delegated_plan(plan), check_files=check_files)


def sft_overrides(plan: dict) -> dict:
    """Render every optimizer field supported by both pinned runtimes."""
    validate_runtime_binding(plan, check_files=False)
    with base_plan_context():
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


def verify_production_qualification(plan: dict) -> None:
    """Reopen the inherited LoRA receipt against the delegated base plan."""
    validate_runtime_binding(plan, check_files=False)
    with base_plan_context():
        _BASE_VERIFY_PRODUCTION_QUALIFICATION(delegated_plan(plan))


@contextlib.contextmanager
def execution_context():
    """Install the exact wrapper hooks only for this process and plan."""
    with _PATCH_LOCK, base_plan_context():
        previous_validate = _base_runtime.validate_plan
        previous_overrides = _base_runtime.sft_overrides
        previous_verify = _base_runtime._verify_qwen38_production_qualification
        _base_runtime.validate_plan = validate_plan
        _base_runtime.sft_overrides = sft_overrides
        _base_runtime._verify_qwen38_production_qualification = verify_production_qualification
        try:
            yield
        finally:
            _base_runtime.validate_plan = previous_validate
            _base_runtime.sft_overrides = previous_overrides
            _base_runtime._verify_qwen38_production_qualification = previous_verify


def main() -> None:
    """Run the historical trainer only while exact matched hooks are bound."""
    with execution_context():
        _base_runtime.main()


if __name__ == "__main__":
    main()
