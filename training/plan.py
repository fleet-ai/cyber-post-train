"""Validate model compatibility and produce an immutable run plan."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .io import atomic_write_json, digest_json, file_sha256
from .science import assert_protocol_matches_model, validate_eval_protocol

REQUIRED_COMPATIBILITY_CHECKS = (
    "hf_checkpoint_load",
    "tokenizer_chat_template_roundtrip",
    "forward_backward_bf16",
    "moe_expert_partition",
    "lora_attach_and_optimizer_step",
    "checkpoint_save_resume",
    "inference_export_equivalence",
)
REQUIRED_RECEIPT_BINDINGS = (
    "tokenizer_revision",
    "chat_template_sha256",
    "training_image_digest",
    "framework_revision",
    "weights_manifest_sha256",
    "inference_image_digest",
    "evaluation_protocol_digest",
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def validate_compatibility(
    receipt: Mapping[str, Any], model_revision: str, evaluation_protocol_digest: str
) -> None:
    if receipt.get("schema") != "glm52_training_compatibility_v1":
        raise ValueError("invalid compatibility receipt schema")
    if receipt.get("model_revision") != model_revision:
        raise ValueError("compatibility receipt does not pin configured model_revision")
    if receipt.get("evaluation_protocol_digest") != evaluation_protocol_digest:
        raise ValueError("compatibility receipt does not pin the evaluation protocol")
    checks = receipt.get("checks")
    if not isinstance(checks, Mapping):
        raise ValueError("compatibility receipt is missing checks")
    failed = [name for name in REQUIRED_COMPATIBILITY_CHECKS if checks.get(name) is not True]
    if failed:
        raise ValueError("model compatibility gate failed: " + ", ".join(failed))
    if not receipt.get("cluster_gpu_topology"):
        raise ValueError("compatibility receipt must name the tested cluster GPU topology")
    missing_bindings = [name for name in REQUIRED_RECEIPT_BINDINGS if not receipt.get(name)]
    if missing_bindings:
        raise ValueError(
            "compatibility receipt is missing exact bindings: " + ", ".join(missing_bindings)
        )


def create_run_plan(
    config_path: Path,
    manifest_path: Path,
    receipt_path: Path,
    evaluation_protocol_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    config = _load(config_path)
    manifest = _load(manifest_path)
    receipt = _load(receipt_path)
    evaluation_protocol = _load(evaluation_protocol_path)
    evaluation_protocol_digest = validate_eval_protocol(evaluation_protocol)
    if config.get("schema") != "cyber_post_train_config_v1":
        raise ValueError("invalid training config schema")
    if manifest.get("schema") != "fleet_cyber_dataset_manifest_v1":
        raise ValueError("invalid dataset manifest schema")
    expected_manifest_digest = manifest.get("manifest_digest")
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_digest"}
    if expected_manifest_digest != digest_json(unsigned):
        raise ValueError("dataset manifest digest mismatch")
    model = config.get("model") or {}
    model_revision = model.get("revision")
    mutable_revision = model_revision == "PIN_BEFORE_LAUNCH"
    if not isinstance(model_revision, str) or len(model_revision) < 8 or mutable_revision:
        raise ValueError("model.revision must be an immutable commit SHA")
    assert_protocol_matches_model(evaluation_protocol, model)
    validate_compatibility(receipt, model_revision, evaluation_protocol_digest)
    if config.get("data", {}).get("use_all_eligible_fleet_data") is not True:
        raise ValueError("this experiment requires all eligible exported Fleet data")

    plan = {
        "schema": "cyber_post_train_run_plan_v1",
        "model": model,
        "inputs": {
            "dataset_manifest": str(manifest_path),
            "dataset_manifest_digest": expected_manifest_digest,
            "compatibility_receipt": str(receipt_path),
            "compatibility_receipt_sha256": file_sha256(receipt_path),
            "evaluation_protocol": str(evaluation_protocol_path),
            "evaluation_protocol_digest": evaluation_protocol_digest,
        },
        "stages": [
            {
                "name": "sft",
                "dataset": "sft_train.jsonl",
                "validation_dataset": "sft_dev.jsonl",
                "config": config.get("sft"),
                "required": True,
            },
            {"name": "pre_rl_eval", "hook": config.get("evaluation"), "required": True},
            {
                "name": "online_rl",
                "dataset": "rl_prompts.jsonl",
                "config": config.get("online_rl"),
                "required": True,
            },
            {"name": "post_rl_eval", "hook": config.get("evaluation"), "required": True},
        ],
        "safety": config.get("safety"),
        "distributed": config.get("distributed"),
        "checkpointing": config.get("checkpointing"),
    }
    plan["plan_digest"] = digest_json(plan)
    atomic_write_json(output_path, plan)
    return plan
