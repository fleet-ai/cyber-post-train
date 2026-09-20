from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from evals.webexploitbench.tensorlake.collection_provenance import (
    ProvenanceError,
    validate_checkpoint_export,
)
from training import qwen38_lora_artifacts, sft_runtime
from training.io import digest_json, file_sha256
from training.qwen38_lora_artifacts import (
    CHECKPOINT_SCHEMA,
    EXPORT_SCHEMA,
    PHYSICAL_TARGETS,
    validate_checkpoint_receipt,
    validate_export_receipt,
)
from training.sft_runtime import (
    QWEN38_LORA,
    QWEN38_LORA_QUALIFICATION,
    _unsigned_digest,
)

QUALIFIED_COMMIT = "1" * 40
QUALIFIED_IMAGE = "ghcr.io/fleet-ai/skyrl-fleet-v2/trainer@sha256:" + "4" * 64
QUALIFIED_SOURCES = {"skyrl/train/sft_trainer.py": "5" * 64}


@pytest.fixture(autouse=True)
def qualified_runtime_binding(monkeypatch):
    # Keep validator fixtures independent from the production source/image pair.
    # The immutable production binding has its own qualification receipt test.
    monkeypatch.setattr(sft_runtime, "QWEN38_MEGATRON_SKYRL_REVISION", QUALIFIED_COMMIT)
    monkeypatch.setattr(sft_runtime, "QWEN38_MEGATRON_IMAGE", QUALIFIED_IMAGE)
    monkeypatch.setattr(sft_runtime, "QWEN38_MEGATRON_SOURCE_SHA256", dict(QUALIFIED_SOURCES))
    monkeypatch.setattr(
        sft_runtime,
        "QWEN38_MEGATRON_SOURCE_CENSUS_SHA256",
        _unsigned_digest(QUALIFIED_SOURCES),
    )
    monkeypatch.setattr(qwen38_lora_artifacts, "QWEN38_MEGATRON_SKYRL_REVISION", QUALIFIED_COMMIT)
    monkeypatch.setattr(
        qwen38_lora_artifacts,
        "QWEN38_MEGATRON_SOURCE_SHA256",
        dict(QUALIFIED_SOURCES),
    )
    # Artifact tests exercise a small synthetic checkpoint chain, not the
    # production corpus. Give that chain its own complete reviewed identity so
    # production remains blocked on the pending leak-free corpus materialization.
    monkeypatch.setattr(
        sft_runtime,
        "QWEN38_LORA_ONE_STEP_PLAN",
        sft_runtime._qwen38_lora_one_step_identity(plan()),
    )


def signed(value: dict) -> dict:
    return {**value, "receipt_sha256": _unsigned_digest(value)}


def plan() -> dict:
    name = "qwen38-lora-artifact-test"
    return {
        "schema": "cyber_sft_runtime_v2",
        "run_name": name,
        "output_root": "/mnt/sfs/jobs/qwen38-lora-artifact-test",
        "model": {
            "root": "/mnt/sfs/models/qwen38",
            "repo": "Qwen/Qwen3.8-27B",
            "revision": "1" * 40,
            "weight_manifest_sha256": "sha256:" + "6" * 64,
            "files": [
                {"path": filename, "sha256": "2" * 64}
                for filename in (
                    "config.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "chat_template.jinja",
                )
            ],
        },
        "datasets": {
            "train": {
                "path": "/mnt/sfs/data/train.parquet",
                "sha256": "3" * 64,
                "rows": 2,
                "task_keys": ["train-task"],
            }
        },
        "recipe": {
            "epochs": 1,
            "batch_size": 1,
            "microbatch_per_gpu": 1,
            "nodes": 1,
            "gpus_per_node": 8,
            "lr": 3e-5,
            "max_length": 16384,
            "eval_interval": 0,
            "checkpoint_interval": 1,
            "keep_checkpoints": 3,
            "max_steps": 2,
            "seed": 7,
        },
        "wandb": {
            "entity": "thefleet",
            "project": "cyber-post-train",
            "group": "qwen38-lora-sft-goal-v1",
            "run_id": name,
            "name": name,
            "tags": ["qwen38", "lora", "exact-model-gate"],
        },
        "validation_mode": "task_outcomes_only",
        "split_manifest_sha256": "sha256:" + "7" * 64,
        "corpus_manifest_sha256": "sha256:" + "8" * 64,
        "lora": dict(QWEN38_LORA),
        "skyrl_runtime": {
            "source_commit": QUALIFIED_COMMIT,
            "source_files_sha256": dict(QUALIFIED_SOURCES),
        },
        "qualification_gate": copy.deepcopy(QWEN38_LORA_QUALIFICATION),
        "pause_after_step": 1,
        "execution": {
            "image": QUALIFIED_IMAGE,
            "priority": "c1",
            "resources": {},
        },
    }


def target_census() -> dict:
    counts = dict.fromkeys(PHYSICAL_TARGETS, 0)
    counts.update(
        {
            "linear_qkv": 2,
            "linear_proj": 2,
            "linear_fc1": 4,
            "linear_fc2": 4,
            "in_proj": 2,
            "out_proj": 2,
        }
    )
    return {
        "selector": "all-linear",
        "lora_type": "lora",
        "physical_targets": list(PHYSICAL_TARGETS),
        "target_counts": counts,
        "gated_delta_net_blocks": 2,
        "standard_attention_blocks": 2,
        "multi_latent_attention_blocks": 0,
        "mlp_projection_pairs": 4,
    }


def adapter_parameters(census: dict) -> dict:
    return {
        f"model.layers.{index}.{target}.adapter.linear_in.weight": {
            "logical_target": target,
            "dtype": "BF16",
            "global_shape": [64, 128],
            "sharding": {
                "tensor_parallel": True,
                "partition_dim": 1,
                "partition_stride": 1,
            },
            "rank_shards": [
                {
                    "tp_rank": rank,
                    "shape": [64, 16],
                    "before_sha256": f"{(index + 10) * 16 + rank:064x}",
                    "after_sha256": f"{(index + 10) * 16 + rank + 8:064x}",
                    "checkpoint_sha256": f"{(index + 10) * 16 + rank + 8:064x}",
                }
                for rank in range(8)
            ],
        }
        for index, (target, count) in enumerate(census["target_counts"].items())
        if count
    }


def checkpoint() -> dict:
    source = plan()
    adapters = [f"policy/adapter_tp{rank}_pp0_cp0_dp0_ep0_etp0.pt" for rank in range(8)]
    optimizer = ["policy/.metadata", "policy/__0_0.distcp"]
    metadata = [
        "data.pt",
        "trainer_state.pt",
        "policy/huggingface/config.json",
        "policy/huggingface/tokenizer.json",
    ]
    names = adapters + optimizer + metadata
    files = {
        name: {"bytes": index + 1, "sha256": f"{index + 100:064x}"}
        for index, name in enumerate(names)
    }
    census = target_census()
    parameters = adapter_parameters(census)
    trainable_manifests = [
        {
            "tp_rank": rank,
            "before_sha256": f"{1000 + rank:064x}",
            "after_sha256": f"{1000 + rank:064x}",
        }
        for rank in range(8)
    ]
    frozen_manifests = [
        {
            "tp_rank": rank,
            "before_sha256": f"{2000 + rank:064x}",
            "after_sha256": f"{2000 + rank:064x}",
        }
        for rank in range(8)
    ]
    return signed(
        {
            "schema": CHECKPOINT_SCHEMA,
            "source_plan_sha256": digest_json(source).removeprefix("sha256:"),
            "source_plan": source,
            "checkpoint_path": source["output_root"] + "/checkpoints/global_step_1",
            "optimizer_step": 1,
            "topology": {
                "world_size": 8,
                "tensor_parallel": 8,
                "pipeline_parallel": 1,
                "context_parallel": 1,
                "data_parallel": 1,
                "expert_parallel": 1,
                "expert_tensor_parallel": 1,
            },
            "target_census": census,
            "adapter_parameters": parameters,
            "trainable_parameter_census": {
                "parameter_count": len(parameters),
                "elements": len(parameters) * 64 * 128,
                "rank_manifests": trainable_manifests,
            },
            "frozen_base": {
                "parameter_count": 100,
                "elements": 1000,
                "bytes": 2000,
                "rank_manifests": frozen_manifests,
            },
            "checkpoint_finalization": {
                "async_writes_finalized": True,
                "rank_acknowledgements": [
                    {"world_rank": rank, "tp_rank": rank, "finalized": True} for rank in range(8)
                ],
            },
            "source_inventory": {
                "model_and_data": {
                    "file_count": 5,
                    "total_bytes": 500,
                    "before_sha256": "a" * 64,
                    "after_sha256": "a" * 64,
                },
                "runtime": {
                    "file_count": 14,
                    "total_bytes": 1400,
                    "before_sha256": "b" * 64,
                    "after_sha256": "b" * 64,
                },
            },
            "wandb": {
                "entity": source["wandb"]["entity"],
                "project": source["wandb"]["project"],
                "group": source["wandb"]["group"],
                "run_id": source["wandb"]["run_id"],
                "name": source["wandb"]["name"],
                "url": "https://wandb.ai/thefleet/cyber-post-train/runs/test",
                "optimizer_step": 1,
                "local_metrics_sha256": "c" * 64,
                "scalar_keys": [
                    "train/global_step",
                    "train/grad_norm",
                    "train/loss",
                    "train/lr",
                ],
                "finish_succeeded": True,
            },
            "files": files,
            "file_roles": {
                "adapter": adapters,
                "optimizer_and_rng": optimizer,
                "metadata": metadata,
            },
            "total_bytes": sum(item["bytes"] for item in files.values()),
            "evidence": {
                "forward_loss": 1.25,
                "lora_gradient_norm": 0.5,
                "optimizer_updates": 1,
                "adapter_tensors_changed": True,
                "adapter_updated_tensor_count": len(parameters) * 8,
                "frozen_base_tensors_unchanged": True,
                "unexpected_trainable_parameters": [],
                "source_inventory_unchanged": True,
                "wandb_run_id": source["wandb"]["run_id"],
            },
        }
    )


def export(checkpoint_receipt: dict, checkpoint_file_sha256: str) -> dict:
    checkpoint_identity = validate_checkpoint_receipt(checkpoint_receipt)
    files = {
        "config.json": {"bytes": 1, "sha256": "5" * 64},
        "tokenizer.json": {"bytes": 2, "sha256": "6" * 64},
        "tokenizer_config.json": {"bytes": 3, "sha256": "7" * 64},
        "chat_template.jinja": {"bytes": 4, "sha256": "8" * 64},
        "model.safetensors.index.json": {"bytes": 5, "sha256": "9" * 64},
        "model-00001-of-00001.safetensors": {"bytes": 6, "sha256": "a" * 64},
    }
    return signed(
        {
            "schema": EXPORT_SCHEMA,
            "source_checkpoint_receipt_sha256": checkpoint_identity["receipt_sha256"],
            "source_manifest_file_sha256": checkpoint_file_sha256.removeprefix("sha256:"),
            "source_plan_sha256": checkpoint_identity["source_plan_sha256"],
            "code_sha256": {"merge.py": "b" * 64, "artifacts.py": "c" * 64},
            "model_repo": checkpoint_identity["model_repository"],
            "model_revision": checkpoint_identity["base_model_revision"],
            "output_root": "/mnt/sfs/jobs/qwen38-lora-artifact-test/merged-step-1",
            "optimizer_step": 1,
            "optimizer_steps_executed": 0,
            "merge_method": "megatron_bridge_lora_merge_v1",
            "dtype": "BF16",
            "adapter_parameter_inventory_sha256": checkpoint_identity[
                "adapter_parameter_inventory_sha256"
            ],
            "target_census_sha256": checkpoint_identity["target_census_sha256"],
            "base_model_inventory_sha256": checkpoint_identity["base_model_inventory_sha256"],
            "base_tensor_layout_sha256": "d" * 64,
            "merged_tensor_layout_sha256": "d" * 64,
            "base_tensor_count": 100,
            "merged_tensor_count": 100,
            "adapter_updated_tensor_count": 12,
            "tensor_values": 1000,
            "tensor_bytes": 2000,
            "sidecars": {
                name: row["sha256"]
                for name, row in files.items()
                if name
                in {
                    "config.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "chat_template.jinja",
                }
            },
            "files": files,
            "source_checkpoint_unchanged": True,
            "source_base_unchanged": True,
            "adapter_checkpoint_reload_verified": True,
            "optimizer_resume_verified": True,
            "all_output_tensors_reopened_equal": True,
            "adapter_payloads_absent": True,
            "deterministic_merge": True,
            "merged_model_reload_verified": True,
            "finite_logits": True,
            "gpu_reload_verified": True,
        }
    )


def write(path: Path, value: dict) -> dict[str, str]:
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return {"path": str(path), "file_sha256": file_sha256(path)}


def test_exact_qwen_megatron_checkpoint_and_merged_export_are_accepted(
    tmp_path: Path,
) -> None:
    checkpoint_receipt = checkpoint()
    checkpoint_reference = write(tmp_path / "checkpoint.json", checkpoint_receipt)
    export_receipt = export(checkpoint_receipt, checkpoint_reference["file_sha256"])
    export_reference = write(tmp_path / "export.json", export_receipt)

    checkpoint_identity = validate_checkpoint_receipt(checkpoint_receipt)
    export_identity = validate_export_receipt(
        export_receipt,
        checkpoint_receipt,
        checkpoint_file_sha256=checkpoint_reference["file_sha256"],
    )
    chain = validate_checkpoint_export(
        {"checkpoint": checkpoint_reference, "export": export_reference}
    )

    assert checkpoint_identity["optimizer_step"] == 1
    assert export_identity["schema"] == EXPORT_SCHEMA
    assert chain["checkpoint"]["schema"] == CHECKPOINT_SCHEMA
    assert chain["export"]["schema"] == EXPORT_SCHEMA
    assert chain["export"]["source_checkpoint_file_sha256"] == checkpoint_reference["file_sha256"]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda value: value["target_census"]["target_counts"].__setitem__("in_proj", 1),
            "counts do not reconcile",
        ),
        (
            lambda value: value["file_roles"]["adapter"].pop(),
            "TP8 rank layout",
        ),
        (
            lambda value: value["evidence"].__setitem__("frozen_base_tensors_unchanged", False),
            "optimizer/frozen-base",
        ),
        (
            lambda value: value["adapter_parameters"][next(iter(value["adapter_parameters"]))][
                "rank_shards"
            ].pop(),
            "missing one or more TP8 rank shards",
        ),
        (
            lambda value: value["source_inventory"]["runtime"].__setitem__(
                "after_sha256", "f" * 64
            ),
            "source inventory changed",
        ),
        (
            lambda value: value["wandb"].__setitem__("finish_succeeded", False),
            "W&B run binding",
        ),
    ],
)
def test_checkpoint_fails_closed_on_census_layout_or_evidence(mutate, message: str) -> None:
    value = checkpoint()
    mutate(value)
    value["receipt_sha256"] = _unsigned_digest(
        {key: item for key, item in value.items() if key != "receipt_sha256"}
    )
    with pytest.raises(ValueError, match=message):
        validate_checkpoint_receipt(value)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("adapter_parameter_inventory_sha256", "f" * 64, "different checkpoint"),
        ("adapter_payloads_absent", False, "layout/merge/reload/source"),
        ("adapter_checkpoint_reload_verified", False, "layout/merge/reload/source"),
        ("gpu_reload_verified", False, "layout/merge/reload/source"),
        ("base_tensor_count", 99, "layout/merge/reload/source"),
    ],
)
def test_merged_export_fails_closed_on_binding_or_acceptance_defect(
    tmp_path: Path, field: str, value, message: str
) -> None:
    checkpoint_receipt = checkpoint()
    checkpoint_reference = write(tmp_path / "checkpoint.json", checkpoint_receipt)
    export_receipt = export(checkpoint_receipt, checkpoint_reference["file_sha256"])
    export_receipt[field] = value
    export_receipt["receipt_sha256"] = _unsigned_digest(
        {key: item for key, item in export_receipt.items() if key != "receipt_sha256"}
    )
    with pytest.raises(ValueError, match=message):
        validate_export_receipt(
            export_receipt,
            checkpoint_receipt,
            checkpoint_file_sha256=checkpoint_reference["file_sha256"],
        )


def test_dense_and_qwen_receipt_schemas_cannot_be_crossed(tmp_path: Path) -> None:
    checkpoint_receipt = checkpoint()
    checkpoint_reference = write(tmp_path / "checkpoint.json", checkpoint_receipt)
    export_receipt = export(checkpoint_receipt, checkpoint_reference["file_sha256"])
    export_receipt["schema"] = "cyber_native_checkpoint_hf_export_v1"
    export_receipt["receipt_sha256"] = _unsigned_digest(
        {key: item for key, item in export_receipt.items() if key != "receipt_sha256"}
    )
    export_reference = write(tmp_path / "export.json", export_receipt)
    with pytest.raises(ProvenanceError, match="dense HF export requires"):
        validate_checkpoint_export({"checkpoint": checkpoint_reference, "export": export_reference})


def test_merged_export_rejects_nested_adapter_payload(tmp_path: Path) -> None:
    checkpoint_receipt = checkpoint()
    checkpoint_reference = write(tmp_path / "checkpoint.json", checkpoint_receipt)
    export_receipt = export(checkpoint_receipt, checkpoint_reference["file_sha256"])
    export_receipt["files"]["adapter/payload.bin"] = {
        "bytes": 1,
        "sha256": "e" * 64,
    }
    export_receipt["receipt_sha256"] = _unsigned_digest(
        {key: item for key, item in export_receipt.items() if key != "receipt_sha256"}
    )
    with pytest.raises(ValueError, match="flat Hugging Face directory"):
        validate_export_receipt(
            export_receipt,
            checkpoint_receipt,
            checkpoint_file_sha256=checkpoint_reference["file_sha256"],
        )
