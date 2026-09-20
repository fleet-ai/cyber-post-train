"""Strict receipts for the pending Qwen3.8 Megatron-LoRA qualification handoff.

The native Megatron checkpoint is not the FSDP checkpoint consumed by
``training.export`` and it is not the GLM PEFT checkpoint consumed by
``training.checkpoints``.  This module therefore validates a separate,
create-once evidence surface instead of guessing either existing layout.

It reads JSON metadata only.  The producer that inventories the adapter rank
files and merges the adapter into a complete BF16 Hugging Face export remains a
separate zero-update job.  Evaluation may use the export only after these
receipts, serving registration, and live parity all form one exact chain.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Any

from .io import digest_json
from .sft_runtime import (
    QWEN38_LORA,
    QWEN38_MEGATRON_SKYRL_REVISION,
    QWEN38_MEGATRON_SOURCE_SHA256,
    validate_plan,
)

CHECKPOINT_SCHEMA = "cyber_qwen38_megatron_lora_checkpoint_manifest_v1"
EXPORT_SCHEMA = "cyber_qwen38_megatron_lora_merged_hf_export_v1"
MODEL_REPOSITORY = "Qwen/Qwen3.8-27B"
SHA256 = re.compile(r"^[0-9a-f]{64}$")

# Exact physical target order emitted by the reviewed SkyRL target resolver in
# exact evidence-capable SkyRL revision. Counts may be zero for mutually
# exclusive MLA layouts, but the census must contain every key and must prove
# all selected GDN, attention and MLP blocks are complete.  The import remains
# fail-closed until this source is built and independently qualified.
PHYSICAL_TARGETS = (
    "linear_qkv",
    "linear_proj",
    "linear_fc1",
    "linear_fc2",
    "in_proj",
    "out_proj",
    "linear_q_proj",
    "linear_q_down_proj",
    "linear_q_up_proj",
    "linear_kv_down_proj",
    "linear_kv_up_proj",
    "linear_qkv_down_proj",
)


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return dict(value)


def _exact(value: Mapping[str, Any], fields: set[str], label: str) -> None:
    if set(value) != fields:
        raise ValueError(f"{label} has unknown or missing fields")


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be one bare SHA-256")
    return value


def _receipt_digest(receipt: Mapping[str, Any], label: str) -> str:
    expected = _sha(receipt.get("receipt_sha256"), f"{label} self-digest")
    actual = digest_json({key: value for key, value in receipt.items() if key != "receipt_sha256"})
    if expected != actual.removeprefix("sha256:"):
        raise ValueError(f"{label} self-digest mismatch")
    return expected


def _absolute_path(value: Any, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value.startswith("/") or "//" in value:
        raise ValueError(f"{label} must be one canonical absolute path")
    path = PurePosixPath(value)
    if str(path) != value or any(part in {"", ".", ".."} for part in path.parts[1:]):
        raise ValueError(f"{label} must be one canonical absolute path")
    return path


def _inventory(value: Any, label: str) -> tuple[dict[str, dict[str, Any]], int]:
    rows = _object(value, label)
    if not rows:
        raise ValueError(f"{label} must not be empty")
    total = 0
    for name, raw in rows.items():
        if not isinstance(name, str):
            raise ValueError(f"{label} contains a non-string path")
        path = PurePosixPath(name)
        row = _object(raw, f"{label} entry {name}")
        if path.is_absolute() or ".." in path.parts or str(path) != name or not path.parts:
            raise ValueError(f"{label} contains an unsafe path")
        _exact(row, {"bytes", "sha256"}, f"{label} entry {name}")
        if type(row["bytes"]) is not int or row["bytes"] <= 0:
            raise ValueError(f"{label} contains an invalid size")
        _sha(row["sha256"], f"{label} entry {name}")
        total += row["bytes"]
    return rows, total


def _validate_plan(plan: dict[str, Any]) -> None:
    # Reuse the exact one-step source/image/topology/training gate.  This does
    # not touch model or data files; those were bound and checked before the
    # GPU run and remain present in the signed source plan.
    validate_plan(plan, check_files=False)
    if (
        plan.get("model", {}).get("repo") != MODEL_REPOSITORY
        or plan.get("lora") != QWEN38_LORA
        or plan.get("skyrl_runtime")
        != {
            "source_commit": QWEN38_MEGATRON_SKYRL_REVISION,
            "source_files_sha256": QWEN38_MEGATRON_SOURCE_SHA256,
        }
    ):
        raise ValueError("checkpoint does not bind the qualified Qwen3.8 Megatron-LoRA plan")


def _validate_target_census(value: Any) -> dict[str, Any]:
    census = _object(value, "target census")
    _exact(
        census,
        {
            "selector",
            "lora_type",
            "physical_targets",
            "target_counts",
            "gated_delta_net_blocks",
            "standard_attention_blocks",
            "multi_latent_attention_blocks",
            "mlp_projection_pairs",
        },
        "target census",
    )
    if (
        census["selector"] != "all-linear"
        or census["lora_type"] != "lora"
        or census["physical_targets"] != list(PHYSICAL_TARGETS)
    ):
        raise ValueError("target census differs from the pinned all-linear resolver")
    counts = _object(census["target_counts"], "target counts")
    if set(counts) != set(PHYSICAL_TARGETS) or any(
        type(counts[name]) is not int or counts[name] < 0 for name in PHYSICAL_TARGETS
    ):
        raise ValueError("target counts are incomplete or invalid")
    for field in (
        "gated_delta_net_blocks",
        "standard_attention_blocks",
        "multi_latent_attention_blocks",
        "mlp_projection_pairs",
    ):
        if type(census[field]) is not int or census[field] < 0:
            raise ValueError("target block counts must be nonnegative integers")
    if (
        counts["in_proj"] != census["gated_delta_net_blocks"]
        or counts["out_proj"] != census["gated_delta_net_blocks"]
        or counts["linear_qkv"] != census["standard_attention_blocks"]
        or counts["linear_proj"]
        != census["standard_attention_blocks"] + census["multi_latent_attention_blocks"]
        or counts["linear_fc1"] != census["mlp_projection_pairs"]
        or counts["linear_fc2"] != census["mlp_projection_pairs"]
        or census["mlp_projection_pairs"]
        != census["gated_delta_net_blocks"]
        + census["standard_attention_blocks"]
        + census["multi_latent_attention_blocks"]
        or census["mlp_projection_pairs"] <= 0
    ):
        raise ValueError("target census block/projection counts do not reconcile")
    mla_projection_total = sum(
        counts[name]
        for name in (
            "linear_q_proj",
            "linear_q_down_proj",
            "linear_q_up_proj",
            "linear_kv_down_proj",
            "linear_kv_up_proj",
            "linear_qkv_down_proj",
        )
    )
    if (census["multi_latent_attention_blocks"] == 0) != (mla_projection_total == 0):
        raise ValueError("MLA target counts do not match the MLA block census")
    return census


def _validate_adapter_parameters(value: Any) -> dict[str, Any]:
    parameters = _object(value, "adapter parameter census")
    if not parameters:
        raise ValueError("adapter parameter census must not be empty")
    for name, raw in parameters.items():
        if not isinstance(name, str) or ".adapter" not in name.lower():
            raise ValueError("adapter parameter census contains a non-adapter name")
        row = _object(raw, f"adapter parameter {name}")
        _exact(
            row,
            {"logical_target", "dtype", "global_shape", "sharding", "rank_shards"},
            name,
        )
        if row["logical_target"] not in PHYSICAL_TARGETS or row["dtype"] != "BF16":
            raise ValueError("adapter parameter target/dtype differs from the qualified contract")
        shape = row["global_shape"]
        if (
            not isinstance(shape, list)
            or not shape
            or any(type(dimension) is not int or dimension <= 0 for dimension in shape)
        ):
            raise ValueError("adapter parameter has an invalid global shape")
        sharding = _object(row["sharding"], f"adapter parameter sharding {name}")
        _exact(
            sharding,
            {"tensor_parallel", "partition_dim", "partition_stride"},
            f"adapter parameter sharding {name}",
        )
        if (
            type(sharding["tensor_parallel"]) is not bool
            or type(sharding["partition_stride"]) is not int
            or sharding["partition_stride"] <= 0
            or (
                sharding["tensor_parallel"]
                and (
                    type(sharding["partition_dim"]) is not int
                    or not 0 <= sharding["partition_dim"] < len(shape)
                )
            )
            or (not sharding["tensor_parallel"] and sharding["partition_dim"] is not None)
        ):
            raise ValueError("adapter parameter sharding metadata is invalid")
        shards = row["rank_shards"]
        if not isinstance(shards, list) or not shards:
            raise ValueError("adapter parameter has no rank shards")
        seen = set()
        for shard in shards:
            shard = _object(shard, f"adapter parameter shard {name}")
            _exact(
                shard,
                {
                    "tp_rank",
                    "shape",
                    "before_sha256",
                    "after_sha256",
                    "checkpoint_sha256",
                },
                f"adapter parameter shard {name}",
            )
            rank = shard["tp_rank"]
            local_shape = shard["shape"]
            if (
                type(rank) is not int
                or not 0 <= rank < 8
                or rank in seen
                or not isinstance(local_shape, list)
                or not local_shape
                or len(local_shape) != len(shape)
                or any(type(dimension) is not int or dimension <= 0 for dimension in local_shape)
            ):
                raise ValueError("adapter parameter rank shards are invalid")
            _sha(shard["before_sha256"], f"adapter parameter before shard {name}")
            after = _sha(shard["after_sha256"], f"adapter parameter after shard {name}")
            checkpoint = _sha(
                shard["checkpoint_sha256"], f"adapter parameter checkpoint shard {name}"
            )
            if after != checkpoint:
                raise ValueError(
                    "adapter checkpoint tensor differs from the live post-update tensor"
                )
            seen.add(rank)
        if seen != set(range(8)):
            raise ValueError("adapter parameter census is missing one or more TP8 rank shards")
    return parameters


def validate_checkpoint_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    value = _object(receipt, "Qwen LoRA checkpoint receipt")
    _exact(
        value,
        {
            "schema",
            "source_plan_sha256",
            "source_plan",
            "checkpoint_path",
            "optimizer_step",
            "topology",
            "target_census",
            "adapter_parameters",
            "trainable_parameter_census",
            "frozen_base",
            "checkpoint_finalization",
            "source_inventory",
            "wandb",
            "files",
            "file_roles",
            "total_bytes",
            "evidence",
            "receipt_sha256",
        },
        "Qwen LoRA checkpoint receipt",
    )
    if value["schema"] != CHECKPOINT_SCHEMA:
        raise ValueError("unsupported Qwen LoRA checkpoint schema")
    self_digest = _receipt_digest(value, "Qwen LoRA checkpoint receipt")
    plan = _object(value["source_plan"], "Qwen LoRA source plan")
    _validate_plan(plan)
    if _sha(value["source_plan_sha256"], "source plan") != digest_json(plan).removeprefix(
        "sha256:"
    ):
        raise ValueError("Qwen LoRA source-plan digest mismatch")
    step = value["optimizer_step"]
    if type(step) is not int or step != 1:
        raise ValueError("the qualified Qwen LoRA checkpoint must be optimizer step 1")
    expected_path = PurePosixPath(plan["output_root"]) / "checkpoints" / "global_step_1"
    if _absolute_path(value["checkpoint_path"], "checkpoint path") != expected_path:
        raise ValueError("Qwen LoRA checkpoint path differs from its plan and step")

    topology = _object(value["topology"], "Qwen LoRA topology")
    _exact(
        topology,
        {
            "world_size",
            "tensor_parallel",
            "pipeline_parallel",
            "context_parallel",
            "data_parallel",
            "expert_parallel",
            "expert_tensor_parallel",
        },
        "Qwen LoRA topology",
    )
    if topology != {
        "world_size": 8,
        "tensor_parallel": 8,
        "pipeline_parallel": 1,
        "context_parallel": 1,
        "data_parallel": 1,
        "expert_parallel": 1,
        "expert_tensor_parallel": 1,
    }:
        raise ValueError("Qwen LoRA checkpoint topology differs from the qualified TP8 gate")
    finalization = _object(value["checkpoint_finalization"], "checkpoint finalization")
    _exact(
        finalization,
        {"async_writes_finalized", "rank_acknowledgements"},
        "checkpoint finalization",
    )
    acknowledgements = finalization["rank_acknowledgements"]
    if (
        finalization["async_writes_finalized"] is not True
        or not isinstance(acknowledgements, list)
        or len(acknowledgements) != 8
    ):
        raise ValueError("checkpoint finalization does not cover TP8")
    seen_finalized = set()
    for acknowledgement in acknowledgements:
        acknowledgement = _object(acknowledgement, "checkpoint finalization rank")
        _exact(
            acknowledgement,
            {"world_rank", "tp_rank", "finalized"},
            "checkpoint finalization rank",
        )
        rank = acknowledgement["tp_rank"]
        if (
            type(rank) is not int
            or acknowledgement["world_rank"] != rank
            or acknowledgement["finalized"] is not True
            or rank in seen_finalized
            or not 0 <= rank < 8
        ):
            raise ValueError("checkpoint finalization rank acknowledgement is invalid")
        seen_finalized.add(rank)
    if seen_finalized != set(range(8)):
        raise ValueError("checkpoint finalization does not cover TP8")
    census = _validate_target_census(value["target_census"])
    parameters = _validate_adapter_parameters(value["adapter_parameters"])
    represented_targets = {row["logical_target"] for row in parameters.values()}
    required_targets = {name for name, count in census["target_counts"].items() if count > 0}
    if represented_targets != required_targets:
        raise ValueError("adapter parameter census does not cover every present target type")
    changed = sum(
        shard["before_sha256"] != shard["after_sha256"]
        for row in parameters.values()
        for shard in row["rank_shards"]
    )

    trainable = _object(value["trainable_parameter_census"], "trainable parameter census")
    _exact(
        trainable,
        {"parameter_count", "elements", "rank_manifests"},
        "trainable parameter census",
    )
    if (
        trainable["parameter_count"] != len(parameters)
        or type(trainable["elements"]) is not int
        or trainable["elements"] <= 0
        or not isinstance(trainable["rank_manifests"], list)
        or len(trainable["rank_manifests"]) != 8
    ):
        raise ValueError("trainable parameter census differs from the adapter inventory")
    frozen = _object(value["frozen_base"], "frozen-base census")
    _exact(
        frozen,
        {"parameter_count", "elements", "bytes", "rank_manifests"},
        "frozen-base census",
    )
    if any(
        type(frozen[field]) is not int or frozen[field] <= 0
        for field in ("parameter_count", "elements", "bytes")
    ):
        raise ValueError("frozen-base census counts are invalid")
    for label, manifests in (
        ("trainable", trainable["rank_manifests"]),
        ("frozen", frozen["rank_manifests"]),
    ):
        if not isinstance(manifests, list) or len(manifests) != 8:
            raise ValueError(f"{label} rank manifests must cover TP8")
        seen_ranks = set()
        for manifest in manifests:
            manifest = _object(manifest, f"{label} rank manifest")
            _exact(
                manifest,
                {"tp_rank", "before_sha256", "after_sha256"},
                f"{label} rank manifest",
            )
            rank = manifest["tp_rank"]
            if type(rank) is not int or not 0 <= rank < 8 or rank in seen_ranks:
                raise ValueError(f"{label} rank manifests contain an invalid TP rank")
            before = _sha(manifest["before_sha256"], f"{label} before manifest")
            after = _sha(manifest["after_sha256"], f"{label} after manifest")
            if label == "frozen" and before != after:
                raise ValueError("a frozen base shard changed during the optimizer update")
            if label == "trainable" and before != after:
                raise ValueError("the trainable parameter census changed during training")
            seen_ranks.add(rank)
        if seen_ranks != set(range(8)):
            raise ValueError(f"{label} rank manifests do not cover TP8")

    source_inventory = _object(value["source_inventory"], "source inventory")
    _exact(
        source_inventory,
        {"model_and_data", "runtime"},
        "source inventory",
    )
    for label, raw in source_inventory.items():
        inventory = _object(raw, f"{label} source inventory")
        _exact(
            inventory,
            {"file_count", "total_bytes", "before_sha256", "after_sha256"},
            f"{label} source inventory",
        )
        if (
            type(inventory["file_count"]) is not int
            or inventory["file_count"] <= 0
            or type(inventory["total_bytes"]) is not int
            or inventory["total_bytes"] <= 0
            or _sha(inventory["before_sha256"], f"{label} source inventory before")
            != _sha(inventory["after_sha256"], f"{label} source inventory after")
        ):
            raise ValueError(f"{label} source inventory changed or is empty")

    wandb = _object(value["wandb"], "W&B binding")
    _exact(
        wandb,
        {
            "entity",
            "project",
            "group",
            "run_id",
            "name",
            "url",
            "optimizer_step",
            "local_metrics_sha256",
            "scalar_keys",
            "finish_succeeded",
        },
        "W&B binding",
    )
    _sha(wandb["local_metrics_sha256"], "local W&B scalar stream")
    if (
        any(
            wandb[field] != plan["wandb"][field]
            for field in ("entity", "project", "group", "run_id", "name")
        )
        or wandb["optimizer_step"] != 1
        or not isinstance(wandb["url"], str)
        or not wandb["url"].startswith("https://")
        or not isinstance(wandb["scalar_keys"], list)
        or any(not isinstance(key, str) or not key for key in wandb["scalar_keys"])
        or len(set(wandb["scalar_keys"])) != len(wandb["scalar_keys"])
        or not {"train/loss", "train/grad_norm", "train/lr"}.issubset(wandb["scalar_keys"])
        or wandb["finish_succeeded"] is not True
    ):
        raise ValueError("W&B run binding or scalar evidence differs from the source plan")

    files, total = _inventory(value["files"], "checkpoint file inventory")
    roles = _object(value["file_roles"], "checkpoint file roles")
    _exact(roles, {"adapter", "optimizer_and_rng", "metadata"}, "checkpoint file roles")
    if any(
        not isinstance(roles[role], list)
        or not roles[role]
        or len(set(roles[role])) != len(roles[role])
        for role in roles
    ):
        raise ValueError("checkpoint file roles must be nonempty unique lists")
    expected_adapters = [f"policy/adapter_tp{rank}_pp0_cp0_dp0_ep0_etp0.pt" for rank in range(8)]
    if roles["adapter"] != expected_adapters:
        raise ValueError("checkpoint adapter files differ from the exact TP8 rank layout")
    if not {"data.pt", "trainer_state.pt", "policy/huggingface/config.json"}.issubset(
        roles["metadata"]
    ):
        raise ValueError("checkpoint metadata lacks trainer, sampler, or model configuration")
    optimizer_and_rng = set(roles["optimizer_and_rng"])
    if "policy/.metadata" not in optimizer_and_rng or any(
        name != "policy/.metadata" and re.fullmatch(r"policy/__\d+_\d+\.distcp", name) is None
        for name in optimizer_and_rng
    ):
        raise ValueError("optimizer/RNG checkpoint files differ from torch-dist layout")
    flattened = [
        name for role in ("adapter", "optimizer_and_rng", "metadata") for name in roles[role]
    ]
    if len(flattened) != len(set(flattened)) or set(flattened) != set(files):
        raise ValueError("checkpoint file roles are overlapping or not exhaustive")
    if value["total_bytes"] != total:
        raise ValueError("checkpoint byte total differs from its inventory")

    evidence = _object(value["evidence"], "Qwen LoRA checkpoint evidence")
    _exact(
        evidence,
        {
            "forward_loss",
            "lora_gradient_norm",
            "optimizer_updates",
            "adapter_tensors_changed",
            "adapter_updated_tensor_count",
            "frozen_base_tensors_unchanged",
            "unexpected_trainable_parameters",
            "source_inventory_unchanged",
            "wandb_run_id",
        },
        "Qwen LoRA checkpoint evidence",
    )
    if (
        type(evidence["forward_loss"]) not in {int, float}
        or not math.isfinite(evidence["forward_loss"])
        or evidence["forward_loss"] < 0
        or type(evidence["lora_gradient_norm"]) not in {int, float}
        or not math.isfinite(evidence["lora_gradient_norm"])
        or evidence["lora_gradient_norm"] <= 0
        or evidence["optimizer_updates"] != 1
        or evidence["adapter_tensors_changed"] is not True
        or evidence["adapter_updated_tensor_count"] != changed
        or changed <= 0
        or evidence["frozen_base_tensors_unchanged"] is not True
        or evidence["unexpected_trainable_parameters"] != []
        or evidence["source_inventory_unchanged"] is not True
        or evidence["wandb_run_id"] != plan["wandb"]["run_id"]
    ):
        raise ValueError("Qwen LoRA checkpoint lacks exact optimizer/frozen-base evidence")
    return {
        "schema": CHECKPOINT_SCHEMA,
        "receipt_sha256": self_digest,
        "source_plan_sha256": value["source_plan_sha256"],
        "checkpoint_path": value["checkpoint_path"],
        "optimizer_step": step,
        "model_repository": plan["model"]["repo"],
        "base_model_revision": plan["model"]["revision"],
        "checkpoint_inventory_sha256": digest_json(files).removeprefix("sha256:"),
        "target_census_sha256": digest_json(census).removeprefix("sha256:"),
        "adapter_parameter_inventory_sha256": digest_json(parameters).removeprefix("sha256:"),
        "base_model_inventory_sha256": digest_json(plan["model"]["files"]).removeprefix("sha256:"),
    }


def validate_export_receipt(
    receipt: Mapping[str, Any],
    checkpoint_receipt: Mapping[str, Any],
    *,
    checkpoint_file_sha256: str,
) -> dict[str, Any]:
    checkpoint = validate_checkpoint_receipt(checkpoint_receipt)
    value = _object(receipt, "Qwen LoRA merged export receipt")
    _exact(
        value,
        {
            "schema",
            "source_checkpoint_receipt_sha256",
            "source_manifest_file_sha256",
            "source_plan_sha256",
            "code_sha256",
            "model_repo",
            "model_revision",
            "output_root",
            "optimizer_step",
            "optimizer_steps_executed",
            "merge_method",
            "dtype",
            "adapter_parameter_inventory_sha256",
            "target_census_sha256",
            "base_model_inventory_sha256",
            "base_tensor_layout_sha256",
            "merged_tensor_layout_sha256",
            "base_tensor_count",
            "merged_tensor_count",
            "adapter_updated_tensor_count",
            "tensor_values",
            "tensor_bytes",
            "sidecars",
            "files",
            "source_checkpoint_unchanged",
            "source_base_unchanged",
            "adapter_checkpoint_reload_verified",
            "optimizer_resume_verified",
            "all_output_tensors_reopened_equal",
            "adapter_payloads_absent",
            "deterministic_merge",
            "merged_model_reload_verified",
            "finite_logits",
            "gpu_reload_verified",
            "receipt_sha256",
        },
        "Qwen LoRA merged export receipt",
    )
    if value["schema"] != EXPORT_SCHEMA:
        raise ValueError("unsupported Qwen LoRA merged export schema")
    self_digest = _receipt_digest(value, "Qwen LoRA merged export receipt")
    links = {
        "source_checkpoint_receipt_sha256": checkpoint["receipt_sha256"],
        "source_manifest_file_sha256": _sha(
            checkpoint_file_sha256.removeprefix("sha256:"), "checkpoint manifest file"
        ),
        "source_plan_sha256": checkpoint["source_plan_sha256"],
        "adapter_parameter_inventory_sha256": checkpoint["adapter_parameter_inventory_sha256"],
        "target_census_sha256": checkpoint["target_census_sha256"],
        "base_model_inventory_sha256": checkpoint["base_model_inventory_sha256"],
    }
    if any(_sha(value.get(field), field) != expected for field, expected in links.items()):
        raise ValueError("Qwen LoRA merged export binds a different checkpoint/census/base")
    if (
        value["model_repo"] != checkpoint["model_repository"]
        or value["model_revision"] != checkpoint["base_model_revision"]
        or value["optimizer_step"] != checkpoint["optimizer_step"]
        or value["optimizer_steps_executed"] != 0
        or value["merge_method"] != "megatron_bridge_lora_merge_v1"
        or value["dtype"] != "BF16"
    ):
        raise ValueError("Qwen LoRA merged export model/step/merge identity mismatch")
    output_root = _absolute_path(value["output_root"], "merged export root")
    checkpoint_root = PurePosixPath(checkpoint["checkpoint_path"])
    if output_root == checkpoint_root or output_root.is_relative_to(checkpoint_root):
        raise ValueError("merged export aliases or is nested in its checkpoint")
    code = _object(value["code_sha256"], "merged export producer identity")
    if not code or any(
        not isinstance(name, str) or not name or not SHA256.fullmatch(digest)
        for name, digest in code.items()
    ):
        raise ValueError("merged export producer identity is incomplete")
    files, _ = _inventory(value["files"], "merged export file inventory")
    names = set(files)
    required = {
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "model.safetensors.index.json",
    }
    if not required <= names or not any(
        re.fullmatch(r"model-\d{5}-of-\d{5}\.safetensors", name) for name in names
    ):
        raise ValueError("merged export lacks the complete model/runtime surface")
    if any(len(PurePosixPath(name).parts) != 1 for name in names):
        raise ValueError("merged export payload must be one flat Hugging Face directory")
    if any("adapter" in name.lower() for name in names):
        raise ValueError("merged export still contains adapter payload files")
    sidecars = _object(value["sidecars"], "merged export sidecars")
    if not required - {"model.safetensors.index.json"} <= set(sidecars):
        raise ValueError("merged export sidecars are incomplete")
    if any(
        not isinstance(name, str)
        or name not in files
        or _sha(digest, f"merged export sidecar {name}") != files[name]["sha256"]
        for name, digest in sidecars.items()
    ):
        raise ValueError("merged export sidecars differ from the payload inventory")
    for field in (
        "base_tensor_count",
        "merged_tensor_count",
        "adapter_updated_tensor_count",
        "tensor_values",
        "tensor_bytes",
    ):
        if type(value[field]) is not int or value[field] <= 0:
            raise ValueError(f"merged export {field} must be positive")
    if (
        _sha(value["base_tensor_layout_sha256"], "base tensor layout")
        != _sha(value["merged_tensor_layout_sha256"], "merged tensor layout")
        or value["base_tensor_count"] != value["merged_tensor_count"]
        or value["adapter_updated_tensor_count"] > value["merged_tensor_count"]
        or value["source_checkpoint_unchanged"] is not True
        or value["source_base_unchanged"] is not True
        or value["adapter_checkpoint_reload_verified"] is not True
        or value["optimizer_resume_verified"] is not True
        or value["all_output_tensors_reopened_equal"] is not True
        or value["adapter_payloads_absent"] is not True
        or value["deterministic_merge"] is not True
        or value["merged_model_reload_verified"] is not True
        or value["finite_logits"] is not True
        or value["gpu_reload_verified"] is not True
    ):
        raise ValueError("merged export lacks exact layout/merge/reload/source evidence")
    return {
        "schema": EXPORT_SCHEMA,
        "receipt_sha256": self_digest,
        "source_checkpoint_receipt_sha256": checkpoint["receipt_sha256"],
        "source_plan_sha256": checkpoint["source_plan_sha256"],
        "optimizer_step": checkpoint["optimizer_step"],
        "model_repository": checkpoint["model_repository"],
        "base_model_revision": checkpoint["base_model_revision"],
        "export_revision": self_digest,
        "output_root": str(output_root),
        "model_artifact_sha256": digest_json(files).removeprefix("sha256:"),
        "runtime_sidecars_sha256": digest_json(sidecars).removeprefix("sha256:"),
        "files": files,
    }
