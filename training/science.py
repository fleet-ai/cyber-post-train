"""Fail-closed protocol bindings for causal pre/post comparisons."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from .io import digest_json

SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
REVISION_RE = re.compile(r"^[0-9a-f]{40,64}$")


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"evaluation protocol {field} must be an object")
    return value


def _required_text(value: Mapping[str, Any], field: str) -> str:
    item = value.get(field)
    if not isinstance(item, str) or not item.strip():
        raise ValueError(f"evaluation protocol field {field} is empty")
    if item.upper().startswith(("PIN", "TODO", "UNKNOWN", "UNRESOLVED")):
        raise ValueError(f"evaluation protocol field {field} is unresolved")
    return item


def _sha256(value: Mapping[str, Any], field: str) -> str:
    item = _required_text(value, field)
    if not SHA256_RE.fullmatch(item):
        raise ValueError(f"evaluation protocol field {field} must be sha256:<64 hex>")
    return item


def _revision(value: Mapping[str, Any], field: str) -> str:
    item = _required_text(value, field)
    if not REVISION_RE.fullmatch(item):
        raise ValueError(f"evaluation protocol field {field} must be an immutable commit")
    return item


def validate_eval_protocol(value: Mapping[str, Any]) -> str:
    """Validate every controlled variable and return the canonical digest."""

    if value.get("schema") != "cyber_prepost_eval_protocol_v1":
        raise ValueError("invalid evaluation protocol schema")

    model = _mapping(value.get("model"), "model")
    _required_text(model, "repo")
    _revision(model, "base_checkpoint_revision")
    _sha256(model, "weights_manifest_sha256")
    _revision(model, "tokenizer_revision")
    _sha256(model, "tokenizer_manifest_sha256")
    _sha256(model, "chat_template_sha256")

    serving = _mapping(value.get("serving"), "serving")
    image = _required_text(serving, "image")
    if "@sha256:" not in image:
        raise ValueError("evaluation protocol serving.image must be digest-pinned")
    _revision(serving, "engine_revision")
    _required_text(serving, "precision")
    _sha256(serving, "quantization_manifest_sha256")

    harness = _mapping(value.get("harness"), "harness")
    harness_image = _required_text(harness, "image")
    if "@sha256:" not in harness_image:
        raise ValueError("evaluation protocol harness.image must be digest-pinned")
    _revision(harness, "source_revision")
    _sha256(harness, "tool_schema_sha256")
    _sha256(harness, "system_prompt_sha256")

    benchmark = _mapping(value.get("benchmark"), "benchmark")
    _required_text(benchmark, "id")
    _revision(benchmark, "source_revision")
    for field in (
        "task_manifest_sha256",
        "environment_manifest_sha256",
        "verifier_manifest_sha256",
        "prompt_manifest_sha256",
    ):
        _sha256(benchmark, field)

    sampling = _mapping(value.get("sampling"), "sampling")
    for field in ("temperature", "top_p", "max_output_tokens"):
        if not isinstance(sampling.get(field), (int, float)):
            raise ValueError(f"evaluation protocol sampling.{field} must be numeric")
    budgets = _mapping(value.get("budgets"), "budgets")
    for field in ("max_agent_steps", "max_duration_minutes"):
        if not isinstance(budgets.get(field), int) or budgets[field] < 1:
            raise ValueError(f"evaluation protocol budgets.{field} must be positive")
    seeds = value.get("random_seeds")
    if (
        not isinstance(seeds, list)
        or not seeds
        or not all(isinstance(seed, int) for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("evaluation protocol random_seeds must be unique integers")

    unsigned = {key: item for key, item in value.items() if key != "protocol_digest"}
    actual = digest_json(unsigned)
    if value.get("protocol_digest") != actual:
        raise ValueError("evaluation protocol digest mismatch")
    return actual


def assert_protocol_matches_model(protocol: Mapping[str, Any], model: Mapping[str, Any]) -> None:
    """Prove training and both evaluation arms share the exact base artifacts."""

    protocol_model = _mapping(protocol.get("model"), "model")
    if protocol_model.get("repo") != model.get("name"):
        raise ValueError("evaluation protocol model repo differs from training model")
    if protocol_model.get("base_checkpoint_revision") != model.get("revision"):
        raise ValueError("evaluation protocol checkpoint differs from training model")
    if protocol_model.get("weights_manifest_sha256") != model.get("weights_manifest_sha256"):
        raise ValueError("evaluation protocol weights manifest differs from training model")


def build_comparison_manifest(
    *,
    protocol_digest: str,
    base_checkpoint_manifest_sha256: str,
    intervention_checkpoint_manifest_sha256: str,
    training_plan_digest: str,
) -> dict[str, Any]:
    """Bind the only allowed pre/post difference: intervention checkpoint bytes."""

    for name, value in {
        "protocol_digest": protocol_digest,
        "base_checkpoint_manifest_sha256": base_checkpoint_manifest_sha256,
        "intervention_checkpoint_manifest_sha256": intervention_checkpoint_manifest_sha256,
        "training_plan_digest": training_plan_digest,
    }.items():
        if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
            raise ValueError(f"{name} must be sha256:<64 hex>")
    manifest = {
        "schema": "cyber_prepost_comparison_v1",
        "controlled_protocol_digest": protocol_digest,
        "arms": {
            "base": {"checkpoint_manifest_sha256": base_checkpoint_manifest_sha256},
            "intervention": {
                "checkpoint_manifest_sha256": intervention_checkpoint_manifest_sha256,
                "training_plan_digest": training_plan_digest,
            },
        },
        "allowed_difference": "checkpoint_or_adapter_bytes_only",
    }
    manifest["comparison_digest"] = digest_json(manifest)
    return manifest
