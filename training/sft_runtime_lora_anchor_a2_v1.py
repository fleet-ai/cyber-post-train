"""Versioned runtime binding for the fresh A2 LoRA-anchor identity.

The historical ``sft_runtime.py`` bytes are already bound into completed plans
and receipts.  A2 therefore stages this wrapper as its entrypoint, binds the
unchanged base runtime separately, and admits only one fresh create-once
identity derived from the reviewed A1 broad LoRA plan.  It does not alter the
trainer, data, model, optimizer, or resource recipe.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import re
import threading
from pathlib import Path

from training import sft_runtime as _base_runtime

RUNTIME_VARIANT = "qwen38_lora_anchor_a2_v1"
RUNTIME_BINDING_SCHEMA = "cyber_sft_runtime_variant_v1"
RUN_NAME = "chris-q38-lora-sft-a2-v1"
OUTPUT_ROOT = "/mnt/sfs/jobs/chris-q38-lora-sft-a2-v1"
_SHA256 = re.compile(r"[a-f0-9]{64}")
_PATCH_LOCK = threading.RLock()
_BASE_VALIDATE_PLAN = _base_runtime.validate_plan


def _file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def base_plan_binding() -> dict:
    """Return A2's one exact base-runtime identity, never a parameter menu."""
    source = copy.deepcopy(_base_runtime.qwen38_lora_broad_full_plan_binding())
    if source.get("run_name") != "chris-q38-lora-sft-a1-v1":
        raise ValueError("A2 source plan is not the reviewed LoRA anchor")
    source["run_name"] = RUN_NAME
    source["output_root"] = OUTPUT_ROOT
    source["wandb"] = {
        **source["wandb"],
        "run_id": RUN_NAME,
        "name": RUN_NAME,
    }
    return source


def runtime_binding(base_runtime_path: Path | None = None) -> dict[str, str]:
    """Bind this wrapper to exactly the historical base runtime bytes."""
    path = base_runtime_path or Path(_base_runtime.__file__)
    return {
        "schema": RUNTIME_BINDING_SCHEMA,
        "name": RUNTIME_VARIANT,
        "base_runtime_sha256": _file_sha256(path),
    }


@contextlib.contextmanager
def base_plan_context():
    """Temporarily expose A2 only to this isolated versioned runtime.

    The base module remains byte-identical.  A process executing this wrapper
    has one plan, and the lock ensures even local validation cannot leak the
    in-memory A2 binding into another caller.
    """
    with _PATCH_LOCK:
        original = _base_runtime.QWEN38_LORA_BROAD_FULL_PLANS
        expected = base_plan_binding()
        if RUN_NAME in original:
            if original[RUN_NAME] != expected:
                raise ValueError("A2 runtime identity binding drifted in memory")
            # ``main`` installs this context for its whole lifetime and the
            # base runtime then calls our validator recursively.  Re-entering
            # the same exact sealed context is safe; a different binding is
            # not.
            yield original
            return
        replacement = {**original, RUN_NAME: expected}
        _base_runtime.QWEN38_LORA_BROAD_FULL_PLANS = replacement
        try:
            yield replacement
        finally:
            _base_runtime.QWEN38_LORA_BROAD_FULL_PLANS = original


def validate_runtime_binding(
    plan: dict,
    *,
    runtime_path: Path | None = None,
    base_runtime_path: Path | None = None,
    check_files: bool = True,
) -> None:
    """Fail closed unless plan, wrapper, base bytes, and A2 identity agree."""
    binding = plan.get("runtime_variant")
    wandb = plan.get("wandb")
    if (
        not isinstance(binding, dict)
        or set(binding) != {"schema", "name", "base_runtime_sha256"}
        or binding.get("schema") != RUNTIME_BINDING_SCHEMA
        or binding.get("name") != RUNTIME_VARIANT
        or _SHA256.fullmatch(binding.get("base_runtime_sha256", "")) is None
        or _SHA256.fullmatch(plan.get("runtime_sha256", "")) is None
        or plan["runtime_sha256"] == binding["base_runtime_sha256"]
        or plan.get("run_name") != RUN_NAME
        or plan.get("output_root") != OUTPUT_ROOT
        or not isinstance(wandb, dict)
        or wandb.get("run_id") != RUN_NAME
        or wandb.get("name") != RUN_NAME
        or "recovery" in plan
        or plan.get("model", {}).get("repo") != "Qwen/Qwen3.8-27B"
        or "lora" not in plan
    ):
        raise ValueError("A2 LoRA successor runtime binding is invalid")
    if check_files:
        wrapper = runtime_path or Path(__file__)
        base = base_runtime_path or Path(_base_runtime.__file__)
        if (
            _file_sha256(wrapper) != plan["runtime_sha256"]
            or _file_sha256(base) != binding["base_runtime_sha256"]
        ):
            raise ValueError("A2 LoRA successor runtime source digest mismatch")


def delegated_plan(plan: dict) -> dict:
    """Remove only wrapper transport metadata before base-runtime validation."""
    delegated = copy.deepcopy(plan)
    delegated.pop("runtime_variant", None)
    delegated["runtime_sha256"] = runtime_binding()["base_runtime_sha256"]
    return delegated


def validate_plan(plan: dict, *, check_files: bool = True) -> None:
    """Validate the wrapper then the unchanged base plan under A2's identity."""
    validate_runtime_binding(plan, check_files=check_files)
    with base_plan_context():
        _BASE_VALIDATE_PLAN(delegated_plan(plan), check_files=check_files)


def main() -> None:
    """Run the historical runtime only while its exact A2 identity is bound."""
    previous_validate = _base_runtime.validate_plan
    with base_plan_context():
        _base_runtime.validate_plan = validate_plan
        try:
            _base_runtime.main()
        finally:
            _base_runtime.validate_plan = previous_validate


if __name__ == "__main__":
    main()
