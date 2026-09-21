"""Receipt contract for Qwen3.8 LoRA's native zero-GPU SFT preflight.

The generic SFT CPU-preflight Job already executes the actual compiler and
native loader for a prepared SFT plan.  Qwen3.8 LoRA needs a small typed
wrapper around that receipt so an operator cannot accidentally describe a
generic SFT preflight as a LoRA gate.  This module adds no new workload
shape: it binds the generic Job receipt to the exact Qwen/LoRA plan instead.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from .jobs import digest, validate_request

SCHEMA = "cyber_qwen38_lora_sft_cpu_preflight_v1"
NATIVE_SCHEMA = "cyber_sft_cpu_preflight_v1"
MODEL_REPO = "Qwen/Qwen3.8-27B"
RUNTIME_VARIANT = "qwen38_lora_anchor_a2_v1"
MATCHED_RUNTIME_VARIANT = "qwen38_sft_32k_matched_v1"
RUNTIME_VARIANTS = frozenset({RUNTIME_VARIANT, MATCHED_RUNTIME_VARIANT})
LORA_ANCHOR = {
    "type": "lora",
    "target_modules": "all-linear",
    "rank": 64,
    "alpha": 32,
    "init_method": "kaiming",
    "dropout": 0.0,
}
_REVISION = re.compile(r"[0-9a-f]{40}")


def is_qwen38_lora_plan(plan: Mapping[str, Any]) -> bool:
    """Recognize only the two reviewed Qwen3.8 rank-64 LoRA runtimes."""

    model = plan.get("model")
    runtime = plan.get("runtime_variant")
    return (
        isinstance(model, Mapping)
        and model.get("repo") == MODEL_REPO
        and plan.get("lora") == LORA_ANCHOR
        and isinstance(runtime, Mapping)
        and runtime.get("name") in RUNTIME_VARIANTS
    )


def _digest_valid(value: Mapping[str, Any], label: str) -> None:
    supplied = value.get("sha256")
    body = {key: item for key, item in value.items() if key != "sha256"}
    if supplied != digest(body):
        raise ValueError(f"{label} digest mismatch")


def _qwen38_lora_identity(plan: Mapping[str, Any], request: Mapping[str, Any]) -> dict[str, Any]:
    """Return only public plan fields that distinguish this receipt family."""

    if not is_qwen38_lora_plan(plan):
        raise ValueError("Qwen3.8 LoRA CPU preflight requires a reviewed rank-64 runtime")
    model = plan["model"]
    lora = plan["lora"]
    runtime = plan.get("runtime_variant")
    execution = plan.get("execution")
    recipe = plan.get("recipe")
    if (
        not isinstance(model, Mapping)
        or not isinstance(lora, Mapping)
        or not isinstance(execution, Mapping)
        or not isinstance(recipe, Mapping)
        or not isinstance(request, Mapping)
    ):
        raise ValueError("Qwen3.8 LoRA CPU preflight plan shape is invalid")
    validate_request(dict(request))
    if (
        request.get("name") != plan.get("run_name")
        or request.get("title") != plan.get("run_name")
        or request.get("run_dir") != plan.get("output_root")
        or request.get("failureAlerts") is not False
        or request.get("priority_class") != "c1"
    ):
        raise ValueError("Qwen3.8 LoRA CPU preflight request identity or policy drifted")
    # The compiled SFT plan describes accelerator topology in ``recipe``;
    # ``execution.resources`` deliberately contains only CPU and memory
    # quantities.  Check the former rather than inventing a ``gpus`` field in
    # the latter, so this gate tracks the reviewed one-node/eight-GPU anchor.
    if (
        recipe.get("nodes") != 1
        or recipe.get("gpus_per_node") != 8
        or request.get("workers") != 1
        or request.get("gpus_per_worker") != 8
        or request.get("image") != execution.get("image")
    ):
        raise ValueError("Qwen3.8 LoRA CPU preflight must bind the one-node eight-GPU plan")
    revision = model.get("revision")
    if not isinstance(revision, str) or _REVISION.fullmatch(revision) is None:
        raise ValueError("Qwen3.8 LoRA CPU preflight requires an exact model revision")
    if not isinstance(runtime, Mapping) or not runtime:
        raise ValueError("Qwen3.8 LoRA CPU preflight requires a versioned runtime binding")
    return {
        "model_revision": model.get("revision"),
        "lora": dict(lora),
        "runtime_variant": dict(runtime),
    }


def validate_plan_request(plan: Mapping[str, Any], request: Mapping[str, Any]) -> None:
    """Fail closed unless this is the reviewed one-node Qwen3.8 LoRA shape.

    The generic SFT CPU Job still runs the actual compiler and native loader.
    This small public boundary exists so the named CLI route can reject a
    dense, stale, or topology-drifted prepared directory before it reaches the
    generic Job creator.
    """

    _qwen38_lora_identity(plan, request)


def build_receipt(
    plan: Mapping[str, Any], request: Mapping[str, Any], native_receipt: Mapping[str, Any]
) -> dict[str, Any]:
    """Bind a verified generic native receipt to the Qwen3.8 LoRA identity."""

    identity = _qwen38_lora_identity(plan, request)
    native = dict(native_receipt)
    _digest_valid(native, "native SFT CPU preflight receipt")
    if (
        native.get("schema") != NATIVE_SCHEMA
        or native.get("status") != "passed"
        or native.get("gpus") != 0
        or native.get("plan_sha256") != digest(dict(plan))
        or native.get("request_sha256") != digest(dict(request))
    ):
        raise ValueError("native SFT CPU preflight receipt does not bind this LoRA plan")
    receipt = {
        "schema": SCHEMA,
        "status": "passed",
        "gpus": 0,
        "plan_sha256": digest(dict(plan)),
        "request_sha256": digest(dict(request)),
        "model_revision": identity["model_revision"],
        "lora": identity["lora"],
        "runtime_variant": identity["runtime_variant"],
        "native_sft_preflight": native,
    }
    return {**receipt, "sha256": digest(receipt)}


def validate_receipt(
    receipt: Mapping[str, Any], plan: Mapping[str, Any], request: Mapping[str, Any]
) -> None:
    """Fail closed unless the persisted receipt exactly matches this prepared run."""

    value = dict(receipt)
    _digest_valid(value, "Qwen3.8 LoRA CPU preflight receipt")
    identity = _qwen38_lora_identity(plan, request)
    expected = {
        "schema": SCHEMA,
        "status": "passed",
        "gpus": 0,
        "plan_sha256": digest(dict(plan)),
        "request_sha256": digest(dict(request)),
        "model_revision": identity["model_revision"],
        "lora": identity["lora"],
        "runtime_variant": identity["runtime_variant"],
    }
    if any(value.get(key) != item for key, item in expected.items()):
        raise ValueError("Qwen3.8 LoRA CPU preflight receipt identity drifted")
    native = value.get("native_sft_preflight")
    if not isinstance(native, Mapping):
        raise ValueError("Qwen3.8 LoRA CPU preflight native receipt is missing")
    # Rebuild validates the nested digest, generic schema, exact plan/request
    # digests, and its zero-GPU status instead of trusting a copied hash.
    rebuilt = build_receipt(plan, request, native)
    if value != rebuilt:
        raise ValueError("Qwen3.8 LoRA CPU preflight receipt has unknown or missing fields")
