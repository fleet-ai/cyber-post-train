import copy
import hashlib
import json
from pathlib import Path

import pytest

from training.io import digest_json
from training.post_sft import (
    assemble_hf_export_receipt,
    build_zero_step_hf_export_request,
    freeze_final_promoted_checkpoint,
    normalize_zero_step_stored_config,
    render_zero_step_sft_command,
    validate_hf_export_receipt,
)
from training.post_sft_export_observation import collect_zero_step_export_run_observation

RUN = "ft-run-574bd7b3"
IMAGE = "registry.example/trainer@sha256:" + "1" * 64
STAGING_IMAGE = "registry.example/stager@sha256:" + "2" * 64
STAGING_COMMAND_SHA256 = "sha256:" + "3" * 64
TOKENIZER_SHA256 = "sha256:" + "4" * 64
CHAT_SHA256 = "sha256:" + "5" * 64
CONFIG_SHA256 = "sha256:" + "6" * 64
SOURCE_SHA256 = "sha256:" + "7" * 64
RAW_WEIGHTS_SHA256 = "sha256:" + "8" * 64
RAW_FILES_SHA256 = "sha256:" + "9" * 64
STRUCTURAL_SHA256 = "sha256:" + "3" * 64
CAST_WEIGHTS_SHA256 = "sha256:" + "0" * 64
COMPOSED_FILES_SHA256 = "sha256:" + "a" * 64
RAW_LAYOUT_SHA256 = digest_json({"weight": {"shape": [1], "dtype": "F32"}})
BASE_LAYOUT_SHA256 = digest_json(
    {
        "mtp.fc.weight": {"shape": [1], "dtype": "BF16"},
        "weight": {"shape": [1], "dtype": "BF16"},
    }
)
SIDECARS = {
    "chat_template.jinja": CHAT_SHA256,
    "config.json": CONFIG_SHA256,
    "configuration.json": "sha256:" + "7" * 64,
    "generation_config.json": "sha256:" + "8" * 64,
    "merges.txt": "sha256:" + "9" * 64,
    "preprocessor_config.json": "sha256:" + "a" * 64,
    "tokenizer.json": "sha256:" + "b" * 64,
    "tokenizer_config.json": "sha256:" + "c" * 64,
    "video_preprocessor_config.json": "sha256:" + "d" * 64,
    "vocab.json": "sha256:" + "e" * 64,
}
STAGING_CODE_SHA256 = {
    "training/__init__.py": "sha256:" + "c" * 64,
    "training/io.py": "sha256:" + "d" * 64,
    "training/post_sft_artifacts.py": "sha256:" + "e" * 64,
    "training/post_sft_base_surface.py": "sha256:" + "1" * 64,
    "training/post_sft_staging.py": "sha256:" + "f" * 64,
}
CAST_CODE_SHA256 = {"training/post_sft_cast.py": "sha256:" + "1" * 64}
EVIDENCE_CODE_SHA256 = {"training/post_sft_artifacts.py": "sha256:" + "2" * 64}
BASE_WEIGHT_ROWS = [
    {
        "path": f"model-{index:05d}-of-00015.safetensors",
        "size": index,
        "sha256": f"{index:x}" * 64,
    }
    for index in range(1, 16)
]
BASE_WEIGHTS_SHA256 = digest_json(BASE_WEIGHT_ROWS)


def _evidence_execution_plan():
    return {
        "schema": "cyber_sft_sfs_evidence_execution_plan_v1",
        "namespace": "fleet-train-jobs",
        "job_name": "chris-cyber-evidence-v4",
        "config_map_name": "chris-cyber-evidence-v4",
        "service_account_name": "chris-cyber-evidence-observer-v4",
        "container_name": "evidence",
        "image": "registry.example/evidence@sha256:" + "2" * 64,
        "image_digest": "sha256:" + "2" * 64,
        "command_sha256": "sha256:" + "3" * 64,
        "registration_sha256": "sha256:" + "d" * 64,
        "config_map_code_sha256": EVIDENCE_CODE_SHA256,
    }


def _no_speculative_proof():
    return {
        "schema": "cyber_post_sft_no_speculative_decoding_proof_v1",
        "registration_sha256": "sha256:" + "d" * 64,
        "runtime_args_sha256": "sha256:" + "4" * 64,
        "prohibited_runtime_args": ["--speculative-algorithm"],
        "prohibited_runtime_args_absent": True,
        "no_speculative_or_draft_argument": True,
    }


def _evidence_runtime():
    plan = _evidence_execution_plan()
    return {
        "schema": "cyber_sft_sfs_evidence_runtime_provenance_v1",
        "image": plan["image"],
        "image_id": "containerd://" + plan["image_digest"],
        "resolved_image_digest": plan["image_digest"],
        "command_sha256": plan["command_sha256"],
        "service_account_name": plan["service_account_name"],
        "container_name": plan["container_name"],
        "plan_file_sha256": "sha256:" + "5" * 64,
        "execution_plan_sha256": digest_json(plan),
        "job": {
            "namespace": plan["namespace"],
            "name": plan["job_name"],
            "uid": "evidence-job-uid",
            "resource_version": "1",
            "spec_sha256": "sha256:" + "6" * 64,
        },
        "pod": {
            "namespace": plan["namespace"],
            "name": "evidence-pod",
            "uid": "evidence-pod-uid",
            "resource_version": "2",
            "spec_sha256": "sha256:" + "7" * 64,
        },
        "config_map": {
            "namespace": plan["namespace"],
            "name": plan["config_map_name"],
            "uid": "evidence-config-uid",
            "resource_version": "3",
            "immutable": True,
            "mounted_file_sha256": EVIDENCE_CODE_SHA256,
            "registration_sha256": plan["registration_sha256"],
        },
        "no_speculative_decoding_proof": _no_speculative_proof(),
    }


def _base_artifact_surface():
    return {
        "schema": "cyber_sft_base_inference_artifact_surface_v1",
        "policy": "exact_top_level_inference_artifacts_with_reviewed_control_exclusions_v1",
        "weight_shard_count": 15,
        "weights_manifest_sha256": BASE_WEIGHTS_SHA256,
        "index": {
            "path": "model.safetensors.index.json",
            "sha256": "sha256:" + "d" * 64,
        },
        "required_runtime_sidecar_sha256": SIDECARS,
        "allowed_non_artifact_top_level_files": {
            "README.md": "test documentation; not loaded by inference"
        },
        "excluded_non_artifact_directory_prefixes": {
            ".cache/": "test cache; not loaded by inference"
        },
        "unknown_top_level_entries": "reject",
        "symlinks": "reject",
    }


def _base_artifact_manifest():
    rows = [
        *BASE_WEIGHT_ROWS,
        {
            "path": "model.safetensors.index.json",
            "size": 3,
            "sha256": "d" * 64,
        },
        *[
            {
                "path": name,
                "size": index + 4,
                "sha256": digest.removeprefix("sha256:"),
            }
            for index, (name, digest) in enumerate(sorted(SIDECARS.items()))
        ],
    ]
    rows = sorted(rows, key=lambda row: row["path"])
    return {
        "schema": "cyber_sft_base_inference_artifact_manifest_v1",
        "root": "/models/base-revision",
        "surface_sha256": digest_json(_base_artifact_surface()),
        "file_count": len(rows),
        "total_bytes": sum(row["size"] for row in rows),
        "files": rows,
        "manifest_sha256": digest_json(rows),
        "weights_manifest_sha256": BASE_WEIGHTS_SHA256,
        "runtime_sidecar_sha256": SIDECARS,
        "excluded_non_artifact_files": [
            {
                "path": "README.md",
                "reviewed_reason": "test documentation; not loaded by inference",
            }
        ],
        "excluded_non_artifact_directory_prefixes": [
            {
                "path_prefix": ".cache/",
                "reviewed_reason": "test cache; not loaded by inference",
            }
        ],
    }


def _tamper_base_artifact_manifest(manifest: dict, kind: str) -> None:
    rows = manifest["files"]
    if kind == "evil_path":
        rows.append({"path": "evil.bin", "size": 1, "sha256": "1" * 64})
    elif kind == "extra_path":
        rows.append({"path": "unknown.txt", "size": 1, "sha256": "4" * 64})
    elif kind == "missing_path":
        rows.pop(0)
    elif kind == "duplicate_path":
        rows.append(copy.deepcopy(rows[0]))
    elif kind == "changed_index_hash":
        next(row for row in rows if row["path"] == "model.safetensors.index.json")["sha256"] = (
            "1" * 64
        )
    elif kind == "changed_sidecar_hash":
        next(row for row in rows if row["path"] == "config.json")["sha256"] = "0" * 64
    elif kind == "wrong_shard_aggregate":
        next(row for row in rows if row["path"].endswith(".safetensors"))["sha256"] = "3" * 64
    elif kind == "wrong_exclusions":
        manifest["excluded_non_artifact_files"] = []
    rows.sort(key=lambda row: row["path"])
    manifest["file_count"] = len(rows)
    manifest["total_bytes"] = sum(row["size"] for row in rows)
    manifest["manifest_sha256"] = digest_json(rows)
    if kind == "wrong_shard_aggregate":
        manifest["weights_manifest_sha256"] = digest_json(
            [row for row in rows if row["path"].endswith(".safetensors")]
        )
    if kind == "wrong_count":
        manifest["file_count"] += 1
    elif kind == "wrong_total":
        manifest["total_bytes"] += 1


def _cast_execution_plan():
    return {
        "schema": "cyber_sft_fp32_to_bf16_cast_execution_plan_v2",
        "namespace": "fleet-train-jobs",
        "job_name": "chris-cyber-cast-v1",
        "config_map_name": "chris-cyber-cast-v1",
        "service_account_name": "chris-cyber-cast-observer-v1",
        "container_name": "cast",
        "image": "registry.example/caster@sha256:" + "1" * 64,
        "image_digest": "sha256:" + "1" * 64,
        "command_sha256": "sha256:" + "2" * 64,
        "base_model_path": "/models/base-revision",
        "base_inference_artifact_surface": _base_artifact_surface(),
        "config_map_code_sha256": CAST_CODE_SHA256,
    }


def _cast_runtime(cast_input_sha256):
    plan = _cast_execution_plan()
    return {
        "schema": "cyber_sft_fp32_to_bf16_cast_execution_v1",
        "image": plan["image"],
        "image_id": "containerd://" + plan["image_digest"],
        "resolved_image_digest": plan["image_digest"],
        "command_sha256": plan["command_sha256"],
        "service_account_name": plan["service_account_name"],
        "container_name": plan["container_name"],
        "cast_input_sha256": cast_input_sha256,
        "job": {
            "namespace": plan["namespace"],
            "name": plan["job_name"],
            "uid": "cast-job-uid",
            "resource_version": "1",
            "spec_sha256": "sha256:" + "3" * 64,
        },
        "pod": {
            "namespace": plan["namespace"],
            "name": "cast-pod",
            "uid": "cast-pod-uid",
            "resource_version": "2",
            "spec_sha256": "sha256:" + "4" * 64,
        },
        "config_map": {
            "namespace": plan["namespace"],
            "name": plan["config_map_name"],
            "uid": "cast-config-uid",
            "resource_version": "3",
            "immutable": True,
            "mounted_file_sha256": CAST_CODE_SHA256,
        },
    }


def _sign(value, field):
    value[field] = digest_json(value)
    return value


def _stage_execution_plan(image=STAGING_IMAGE, command_sha256=STAGING_COMMAND_SHA256):
    return {
        "schema": "cyber_sft_inference_stage_execution_plan_v1",
        "namespace": "inference",
        "job_name": "chris-cyber-stage-v1",
        "config_map_name": "chris-cyber-stage-v1",
        "service_account_name": "chris-cyber-stage-observer-v1",
        "container_name": "stage",
        "image": image,
        "image_digest": image.rsplit("@", 1)[1],
        "command_sha256": command_sha256,
        "config_map_code_sha256": STAGING_CODE_SHA256,
    }


def _stage_runtime_execution(
    stage_input, image=STAGING_IMAGE, command_sha256=STAGING_COMMAND_SHA256
):
    stage_input_file_sha256 = (
        "sha256:"
        + hashlib.sha256(
            (json.dumps(stage_input, indent=2, sort_keys=True) + "\n").encode()
        ).hexdigest()
    )
    return {
        "schema": "cyber_sft_inference_stage_execution_v1",
        "image": image,
        "image_id": "containerd://" + image.rsplit("@", 1)[1],
        "resolved_image_digest": image.rsplit("@", 1)[1],
        "command_sha256": command_sha256,
        "service_account_name": "chris-cyber-stage-observer-v1",
        "container_name": "stage",
        "job": {
            "namespace": "inference",
            "name": "chris-cyber-stage-v1",
            "uid": "job-uid",
            "resource_version": "11",
            "spec_sha256": "sha256:" + "1" * 64,
        },
        "pod": {
            "namespace": "inference",
            "name": "chris-cyber-stage-v1-pod",
            "uid": "pod-uid",
            "resource_version": "12",
            "spec_sha256": "sha256:" + "2" * 64,
        },
        "config_map": {
            "namespace": "inference",
            "name": "chris-cyber-stage-v1",
            "uid": "config-map-uid",
            "resource_version": "13",
            "immutable": True,
            "reviewed_code_sha256": STAGING_CODE_SHA256,
            "mounted_file_sha256": {
                **STAGING_CODE_SHA256,
                "stage-input.json": stage_input_file_sha256,
            },
            "stage_input_file_sha256": stage_input_file_sha256,
        },
        "stage_input_sha256": stage_input["stage_input_sha256"],
    }


def _binding():
    output_root = f"/mnt/sfs/exports/cyber-sft/{RUN}/step-318-v1"
    return {
        "strategy": "resume_final_checkpoint_with_zero_optimizer_steps_v1",
        "source_run_name": RUN,
        "source_global_step": 318,
        "source_checkpoint_path": f"/mnt/sfs/checkpoints/{RUN}/global_step_318",
        "output_root": output_root,
        "expected_output_path": f"{output_root}/global_step_318/policy",
        "bf16_cast_destination": (
            f"/mnt/sfs/exports/cyber-sft/{RUN}/step-318-bf16-v2/global_step_318/policy"
        ),
        "inference_staging_destination": f"/models/cyber-sft/{RUN}/step-318",
        "destination_preflight": {
            "observed_at": "2026-08-31T16:05:07Z",
            "state": "absent",
            "matching_rayjobs": [{"name": "ft-run-export", "uid": "export-uid"}],
        },
        "export_run": {
            "name": "ft-run-export",
            "run_id": "export-run-id",
            "rayjob_uid": "export-uid",
            "title": "Chris cyber test zero-step export",
            "trainer_version_id": "trainer-version",
            "trainer_image": IMAGE,
            "resume_from": f"/mnt/sfs/checkpoints/{RUN}/global_step_318",
            "num_steps": 318,
            "hf_save_interval": 319,
            "optimizer_steps_expected": 0,
            "server_normalization": {
                "schema": "fleet_training_stored_config_normalization_v1",
                "submitted_by_email": "scientist@example.com",
                "node_pool": "fleetai-training-ng-gpu",
                "data_defaults": {
                    "env_keys": None,
                    "models": None,
                    "session_ids": None,
                    "since": None,
                    "team_ids": ["a1025f0b-ad67-49fc-a023-51800ab43e84"],
                    "until": None,
                },
                "trainer_defaults": {"command": None, "env": {}},
            },
        },
        "raw_export_auxiliary_head_omission": {
            "schema": "cyber_sft_exact_auxiliary_head_omission_v1",
            "role": "speculative_draft_heads",
            "base_repository": "Qwen/Qwen3.6-27B",
            "base_revision": "base-revision",
            "base_weights_manifest_sha256": BASE_WEIGHTS_SHA256,
            "serving_inference_effect": "inert_without_speculative_decoding",
            "serving_registration_sha256": "sha256:" + "d" * 64,
            "prohibited_runtime_args": ["--speculative-algorithm"],
            "restoration_policy": "copy_exact_frozen_base_bf16_tensor_bits",
            "base_tensor_count": 2,
            "base_parameter_count": 2,
            "raw_export_tensor_count": 1,
            "raw_export_parameter_count": 1,
            "missing_tensor_count": 1,
            "missing_parameter_count": 1,
            "tensors": [{"key": "mtp.fc.weight", "shape": [1], "dtype": "BF16", "elements": 1}],
        },
    }


def _collector_inputs_for_request(request):
    entrypoint = render_zero_step_sft_command(request["request"], "ft-run-export")
    stored_config = normalize_zero_step_stored_config(request["request"], request["export_run"])
    stored_config_env = [
        {
            "name": "FLEET_RUN_CONFIG",
            "value": json.dumps(stored_config, sort_keys=True, separators=(",", ":")),
        }
    ]
    head_resources = {
        "requests": {"cpu": "2", "memory": "8Gi"},
        "limits": {"cpu": "4", "memory": "16Gi"},
    }
    worker_resources = {
        "requests": {"cpu": "184", "memory": "2560Gi", "nvidia.com/gpu": "8"},
        "limits": {"cpu": "184", "memory": "2560Gi", "nvidia.com/gpu": "8"},
    }
    submitter_resources = {
        "requests": {"cpu": "200m", "ephemeral-storage": "2Gi", "memory": "512Mi"},
        "limits": {"cpu": "1", "memory": "1Gi"},
    }
    cluster_spec = {
        "headGroupSpec": {
            "template": {
                "spec": {
                    "containers": [
                        {
                            "name": "ray-head",
                            "image": IMAGE,
                            "resources": head_resources,
                            "env": stored_config_env,
                        }
                    ]
                }
            }
        },
        "workerGroupSpecs": [
            {
                "groupName": "gpu-worker",
                "replicas": 1,
                "minReplicas": 1,
                "maxReplicas": 1,
                "template": {
                    "spec": {
                        "containers": [
                            {
                                "name": "ray-worker",
                                "image": IMAGE,
                                "resources": worker_resources,
                            }
                        ]
                    }
                },
            }
        ],
    }
    rayjob = {
        "metadata": {
            "namespace": "fleet-train-jobs",
            "name": "ft-run-export",
            "uid": "export-uid",
            "labels": {"kueue.x-k8s.io/queue-name": "training-lq"},
        },
        "spec": {
            "backoffLimit": 0,
            "entrypoint": entrypoint,
            "rayClusterSpec": copy.deepcopy(cluster_spec),
            "shutdownAfterJobFinishes": True,
            "submissionMode": "K8sJobMode",
            "submitterPodTemplate": {
                "spec": {
                    "containers": [
                        {
                            "name": "ray-job-submitter",
                            "image": "anyscale/ray:2.56.0-slim-py312",
                            "resources": submitter_resources,
                        }
                    ]
                }
            },
            "ttlSecondsAfterFinished": 0,
        },
        "status": {"jobStatus": "SUCCEEDED", "rayClusterName": "export-cluster"},
    }
    raycluster = {
        "metadata": {
            "namespace": "fleet-train-jobs",
            "name": "export-cluster",
            "uid": "cluster-uid",
            "ownerReferences": [
                {
                    "kind": "RayJob",
                    "name": "ft-run-export",
                    "uid": "export-uid",
                    "controller": True,
                }
            ],
        },
        "spec": copy.deepcopy(cluster_spec),
    }
    pod = {
        "metadata": {
            "namespace": "fleet-train-jobs",
            "name": "export-head",
            "uid": "pod-uid",
            "labels": {"ray.io/cluster": "export-cluster", "ray.io/node-type": "head"},
            "ownerReferences": [
                {
                    "kind": "RayCluster",
                    "name": "export-cluster",
                    "uid": "cluster-uid",
                    "controller": True,
                }
            ],
        },
        "spec": {"containers": [{"name": "ray-head", "image": IMAGE}]},
        "status": {
            "phase": "Running",
            "containerStatuses": [
                {
                    "name": "ray-head",
                    "imageID": "resolved.example/trainer@sha256:" + "1" * 64,
                }
            ],
        },
    }
    submitter_job = {
        "metadata": {
            "namespace": "fleet-train-jobs",
            "name": "ft-run-export",
            "uid": "submitter-job-uid",
            "ownerReferences": [
                {
                    "kind": "RayJob",
                    "name": "ft-run-export",
                    "uid": "export-uid",
                    "controller": True,
                }
            ],
        },
        "spec": {
            "template": {
                "spec": {
                    "containers": [
                        {
                            "name": "ray-job-submitter",
                            "image": "anyscale/ray:2.56.0-slim-py312",
                            "resources": submitter_resources,
                        }
                    ]
                }
            }
        },
    }
    submitter_pod = {
        "metadata": {
            "namespace": "fleet-train-jobs",
            "name": "ft-run-export-submitter",
            "uid": "submitter-pod-uid",
            "ownerReferences": [
                {
                    "kind": "Job",
                    "name": "ft-run-export",
                    "uid": "submitter-job-uid",
                    "controller": True,
                }
            ],
        },
        "spec": {
            "containers": [
                {
                    "name": "ray-job-submitter",
                    "image": "anyscale/ray:2.56.0-slim-py312",
                    "resources": submitter_resources,
                }
            ]
        },
    }
    api = {
        "name": "ft-run-export",
        "id": "export-run-id",
        "kind": "sft",
        "status": "succeeded",
        "trainer_version_id": "trainer-version",
        "config": stored_config,
        "current_step": 318,
        "latest_metrics": {"global_step": 318, "optimizer_steps": 0},
        "steps": [],
        "step_metrics": [],
    }
    log = ("submitted exact entrypoint: " + entrypoint + "\nrun succeeded\n").encode()
    return request, rayjob, raycluster, pod, submitter_job, submitter_pod, api, log


def _artifacts():
    selection = freeze_final_promoted_checkpoint(
        {
            "schema": "fleet_training_run_observation_v1",
            "run_name": RUN,
            "status": "succeeded",
            "rayjob_uid": "source-uid",
            "run_config_sha256": "sha256:" + "c" * 64,
            "trainer_image": IMAGE,
            "entrypoint_sha256": "sha256:" + "d" * 64,
            "latest_checkpoint_step": 318,
        },
        [
            {
                "run_name": RUN,
                "step": 318,
                "complete": True,
                "is_promoted": True,
                "s3_available": True,
                "s3_uri": "s3://example/checkpoint",
                "archive_manifest_sha256": SOURCE_SHA256,
                "sfs_path": f"/mnt/sfs/checkpoints/{RUN}/global_step_318",
                "sfs_available": True,
            }
        ],
        expected_run_name=RUN,
        expected_run_config_sha256="sha256:" + "c" * 64,
        expected_rayjob_uid="source-uid",
        expected_trainer_image=IMAGE,
        expected_entrypoint_sha256="sha256:" + "d" * 64,
    )
    request = build_zero_step_hf_export_request(
        {
            "kind": "sft",
            "data": {},
            "objective": {
                "train_on_what": "last_assistant_message",
                "max_length": 16384,
            },
            "model": {"staged_model": "qwen3.6-27b", "precision": "bf16"},
            "sft": {
                "strategy": "fsdp",
                "learning_rate": 1e-6,
                "batch_size": 8,
                "micro_train_batch_size_per_gpu": 1,
                "num_epochs": 1,
                "max_steps": 318,
                "checkpoint_interval": 10,
                "max_checkpoints_to_keep": 3,
                "sequence_parallel_size": 1,
            },
            "trainer": {"trainer_version_id": "trainer-version", "args": []},
            "wandb": {"enabled": True, "project": "cyber-post-train"},
            "gpus_per_worker": 8,
            "num_workers": 1,
        },
        selection,
        expected_trainer_version_id="trainer-version",
        expected_trainer_image=IMAGE,
        expected_export_binding=_binding(),
    )
    run_observation = collect_zero_step_export_run_observation(
        *_collector_inputs_for_request(request)
    )
    raw_inspection = {
        "root": _binding()["expected_output_path"],
        "format": "safetensors",
        "dtype": "f32",
        "parameter_count": 1,
        "base_parameter_count": 2,
        "omitted_parameter_count": 1,
        "parameter_count_expectation": (
            "declared_raw_export_after_exact_auxiliary_head_omission_v1"
        ),
        "exact_auxiliary_omission_sha256": digest_json(
            [
                {
                    "key": "mtp.fc.weight",
                    "shape": [1],
                    "dtype": "BF16",
                    "elements": 1,
                    "base_shard": "base.safetensors",
                }
            ]
        ),
        "weights_manifest_sha256": RAW_WEIGHTS_SHA256,
        "files_manifest_sha256": RAW_FILES_SHA256,
        "all_shards_present": True,
        "safetensors_load_passed": True,
        "parameter_count_matches": True,
    }
    export_observation = _sign(
        {
            "schema": "fleet_sft_sfs_checkpoint_observation_v1",
            "run_name": RUN,
            "step": 318,
            "sfs_path": f"/mnt/sfs/checkpoints/{RUN}/global_step_318",
            "structural_manifest_before_sha256": STRUCTURAL_SHA256,
            "structural_manifest_after_sha256": STRUCTURAL_SHA256,
            "markers": {
                "promoted": True,
                "milestone": True,
                "expected_shards": 1,
                "complete_shards": 1,
                "latest_step": 318,
            },
            "full_file_manifest_sha256": SOURCE_SHA256,
            "output_inspection": raw_inspection,
            "raw_export_full_manifest_sha256": RAW_FILES_SHA256,
            "weight_layout_exact_auxiliary_omission": {
                "schema": "cyber_sft_safetensors_exact_auxiliary_omission_v1",
                "base_tensor_count": 2,
                "base_parameter_count": 2,
                "raw_export_tensor_count": 1,
                "raw_export_parameter_count": 1,
                "missing_tensor_count": 1,
                "missing_parameter_count": 1,
                "missing_tensors": [
                    {
                        "key": "mtp.fc.weight",
                        "shape": [1],
                        "dtype": "BF16",
                        "elements": 1,
                        "base_shard": "base.safetensors",
                    }
                ],
                "missing_tensors_sha256": raw_inspection["exact_auxiliary_omission_sha256"],
                "unexpected_key_count": 0,
                "shape_mismatch_count": 0,
                "candidate_wrong_dtype_count": 0,
                "base_layout_sha256": BASE_LAYOUT_SHA256,
                "candidate_layout_sha256": RAW_LAYOUT_SHA256,
                "exact_allowlist_match": True,
                "all_present_keys_and_shapes_match": True,
                "parameter_arithmetic_closes": True,
            },
            "model_config_architecture_equivalence": {
                "all_architecture_and_vocab_fields_identical": True,
                "normalized_architecture_sha256": "sha256:" + "2" * 64,
            },
            "execution": _evidence_runtime(),
            "no_speculative_decoding_proof": _no_speculative_proof(),
        },
        "observation_sha256",
    )
    cast_payload_rows = [{"path": "model.safetensors", "size": 10, "sha256": "e" * 64}]
    cast_payload_sha256 = digest_json(cast_payload_rows)
    cast_inspection = {
        "root": _binding()["bf16_cast_destination"],
        "format": "safetensors",
        "dtype": "bf16",
        "parameter_count": 2,
        "weights_manifest_sha256": CAST_WEIGHTS_SHA256,
        "files_manifest_sha256": cast_payload_sha256,
        "all_shards_present": True,
        "safetensors_load_passed": True,
        "parameter_count_matches": True,
    }
    cast_rows = [
        {
            "kind": "trained_fp32_to_bf16",
            "key": "weight",
            "shape": [1],
            "elements": 1,
            "source_dtype": "F32",
            "destination_dtype": "BF16",
            "source_shard": "raw.safetensors",
            "destination_shard": "model-00001-of-00001.safetensors",
            "source_tensor_sha256": "sha256:" + "4" * 64,
            "destination_tensor_sha256": "sha256:" + "5" * 64,
            "exact_cast_bits_verified": True,
        }
    ]
    restoration_rows = [
        {
            "kind": "frozen_base_auxiliary_head_restoration",
            "key": "mtp.fc.weight",
            "shape": [1],
            "elements": 1,
            "source_dtype": "BF16",
            "destination_dtype": "BF16",
            "base_revision": "base-revision",
            "base_shard": "base.safetensors",
            "destination_shard": "model-00001-of-00001.safetensors",
            "base_tensor_sha256": "sha256:" + "6" * 64,
            "destination_tensor_sha256": "sha256:" + "6" * 64,
            "exact_base_bits_verified": True,
        }
    ]
    cast_receipt = _sign(
        {
            "schema": "cyber_sft_fp32_to_bf16_cast_receipt_v2",
            "cast_input_sha256": "sha256:" + "1" * 64,
            "source": {
                "path": _binding()["expected_output_path"],
                "observation_sha256": export_observation["observation_sha256"],
                "checkpoint_full_manifest_sha256": SOURCE_SHA256,
                "raw_full_manifest_sha256": RAW_FILES_SHA256,
                "raw_full_manifest_after_sha256": RAW_FILES_SHA256,
                "raw_source_stable_during_cast": True,
                "raw_weights_manifest_sha256": RAW_WEIGHTS_SHA256,
                "dtype": "F32",
            },
            "frozen_base_auxiliary_source": {
                "path": "/models/base-revision",
                "repository": "Qwen/Qwen3.6-27B",
                "revision": "base-revision",
                "weights_manifest_sha256": BASE_WEIGHTS_SHA256,
                "inference_artifact_surface": _base_artifact_surface(),
                "inference_artifact_surface_sha256": digest_json(_base_artifact_surface()),
                "inference_artifact_manifest_before": _base_artifact_manifest(),
                "inference_artifact_manifest_after": _base_artifact_manifest(),
                "manifest_scope": "exact_inference_artifact_surface_v1",
                "full_manifest_before_sha256": _base_artifact_manifest()["manifest_sha256"],
                "full_manifest_after_sha256": _base_artifact_manifest()["manifest_sha256"],
                "base_source_stable_during_cast": True,
                "exact_omission_evidence_sha256": raw_inspection["exact_auxiliary_omission_sha256"],
                "role": "speculative_draft_heads",
                "serving_inference_effect": "inert_without_speculative_decoding",
                "serving_registration_sha256": "sha256:" + "d" * 64,
                "speculative_decoding_enabled": False,
                "restoration_semantics": (
                    "frozen_base_auxiliary_head_restoration_not_trained_weights"
                ),
            },
            "conversion": {
                "schema": "cyber_sft_fp32_to_bf16_cast_and_restore_proof_v2",
                "policy": ("deterministic_trained_fp32_to_bf16_plus_frozen_base_mtp_restore_v1"),
                "source_dtype": "F32",
                "destination_dtype": "BF16",
                "trained_parameter_count": 1,
                "trained_tensor_count": 1,
                "restored_auxiliary_parameter_count": 1,
                "restored_auxiliary_tensor_count": 1,
                "final_parameter_count": 2,
                "final_tensor_count": 2,
                "source_layout_sha256": RAW_LAYOUT_SHA256,
                "final_layout_sha256": BASE_LAYOUT_SHA256,
                "exact_omission_evidence_sha256": raw_inspection["exact_auxiliary_omission_sha256"],
                "cast_rows_sha256": digest_json(cast_rows),
                "cast_rows": cast_rows,
                "restoration_rows_sha256": digest_json(restoration_rows),
                "restoration_rows": restoration_rows,
                "all_source_values_finite": True,
                "all_destination_bits_equal_direct_bf16_cast": True,
                "all_restored_auxiliary_bits_equal_frozen_base": True,
                "final_layout_exactly_matches_frozen_base": True,
            },
            "execution_plan": _cast_execution_plan(),
            "execution": _cast_runtime("sha256:" + "1" * 64),
            "destination": {
                "path": _binding()["bf16_cast_destination"],
                "dtype": "BF16",
                "inspection": cast_inspection,
                "exact_base_layout_equivalence": {
                    "schema": "cyber_sft_safetensors_layout_equivalence_v2",
                    "tensor_count": 2,
                    "missing_key_count": 0,
                    "unexpected_key_count": 0,
                    "shape_mismatch_count": 0,
                    "dtype_mismatch_count": 0,
                    "dtype_match_required": True,
                    "base_layout_sha256": BASE_LAYOUT_SHA256,
                    "candidate_layout_sha256": BASE_LAYOUT_SHA256,
                    "all_keys_and_shapes_match": True,
                    "all_keys_shapes_and_dtypes_match": True,
                },
                "payload_manifest_sha256": cast_payload_sha256,
            },
        },
        "cast_receipt_sha256",
    )
    cast_manifest_rows = [
        {
            "path": ".fleet-bf16-cast-acceptance.json",
            "size": 100,
            "sha256": "d" * 64,
        },
        *cast_payload_rows,
    ]
    cast_full_manifest = {
        "schema": "cyber_sft_full_file_manifest_v1",
        "root": _binding()["bf16_cast_destination"],
        "file_count": 2,
        "total_bytes": 110,
        "files": cast_manifest_rows,
        "manifest_sha256": digest_json(cast_manifest_rows),
    }
    stage_input = _sign(
        {
            "schema": "cyber_sft_inference_stage_input_v1",
            "source": {
                "sfs_path": _binding()["bf16_cast_destination"],
                "bf16_full_manifest": cast_full_manifest,
                "bf16_inspection": cast_inspection,
                "observation_sha256": export_observation["observation_sha256"],
                "cast_receipt_sha256": cast_receipt["cast_receipt_sha256"],
                "trained_cast_rows_sha256": cast_receipt["conversion"]["cast_rows_sha256"],
                "restoration_rows_sha256": cast_receipt["conversion"]["restoration_rows_sha256"],
                "exact_auxiliary_omission_sha256": raw_inspection[
                    "exact_auxiliary_omission_sha256"
                ],
                "restoration_semantics": (
                    "frozen_base_auxiliary_head_restoration_not_trained_weights"
                ),
            },
            "composition": {
                "runtime_sidecar_sha256": SIDECARS,
                "tokenizer_equivalence_evidence_sha256": "sha256:" + "0" * 64,
                "expected_tokenizer_manifest_sha256": TOKENIZER_SHA256,
                "expected_chat_template_sha256": CHAT_SHA256,
                "expected_config_sha256": CONFIG_SHA256,
            },
            "execution": _stage_execution_plan(),
            "destination": {
                "path": _binding()["inference_staging_destination"],
                "must_be_absent": True,
            },
        },
        "stage_input_sha256",
    )
    composed = {
        "root": _binding()["inference_staging_destination"],
        "weights_manifest_sha256": CAST_WEIGHTS_SHA256,
        "files_manifest_sha256": COMPOSED_FILES_SHA256,
        "tokenizer_manifest_sha256": TOKENIZER_SHA256,
        "chat_template_sha256": CHAT_SHA256,
        "config_sha256": CONFIG_SHA256,
        "sidecar_sha256": SIDECARS,
        "all_shards_present": True,
        "safetensors_load_passed": True,
        "parameter_count_matches": True,
    }
    staging_receipt = _sign(
        {
            "schema": "cyber_sft_inference_stage_receipt_v1",
            "stage_input_sha256": stage_input["stage_input_sha256"],
            "source_observation_sha256": export_observation["observation_sha256"],
            "source_bf16_manifest_sha256": cast_full_manifest["manifest_sha256"],
            "source_cast_receipt_sha256": cast_receipt["cast_receipt_sha256"],
            "execution": _stage_runtime_execution(stage_input),
            "composition": {
                "policy": (
                    "verified_trained_bf16_plus_frozen_base_auxiliary_heads_and_"
                    "exact_base_runtime_sidecars_v2"
                ),
                "tokenizer_equivalence_evidence_sha256": "sha256:" + "0" * 64,
                "inspection": composed,
            },
            "destination": {
                "path": _binding()["inference_staging_destination"],
                "acceptance_receipt_path": (
                    _binding()["inference_staging_destination"] + "/.fleet-acceptance.json"
                ),
                "payload_manifest_excludes": [".fleet-acceptance.json"],
                "payload_manifest_sha256": COMPOSED_FILES_SHA256,
                "payload_file_count": 12,
                "payload_total_bytes": 100,
                "atomic_transaction": "directory_rename_noreplace_v1",
                "atomic_promotion": True,
            },
        },
        "staging_receipt_sha256",
    )
    return (
        selection,
        request,
        run_observation,
        export_observation,
        cast_receipt,
        cast_full_manifest,
        stage_input,
        staging_receipt,
    )


def _assemble(artifacts):
    return assemble_hf_export_receipt(
        *artifacts,
        expected_tokenizer_manifest_sha256=TOKENIZER_SHA256,
        expected_chat_template_sha256=CHAT_SHA256,
        expected_config_sha256=CONFIG_SHA256,
        expected_export_binding=_binding(),
        expected_runtime_sidecar_sha256=SIDECARS,
        expected_tokenizer_equivalence_evidence_sha256="sha256:" + "0" * 64,
        expected_cast_execution=_cast_execution_plan(),
        expected_sfs_evidence_execution=_evidence_execution_plan(),
        expected_source_structural_sha256=STRUCTURAL_SHA256,
        expected_staging_image=STAGING_IMAGE,
        expected_staging_command_sha256=STAGING_COMMAND_SHA256,
    )


def test_assembly_builds_and_self_validates_final_export_receipt():
    assert _base_artifact_surface()["weight_shard_count"] == 15
    assert len(_base_artifact_surface()["required_runtime_sidecar_sha256"]) == 10
    assert _base_artifact_manifest()["file_count"] == 26
    receipt = _assemble(_artifacts())
    assert receipt["schema"] == "cyber_sft_hf_export_v1"
    assert receipt["conversion"]["optimizer_steps"] == 0
    assert receipt["output"]["weights_manifest_sha256"] == CAST_WEIGHTS_SHA256
    assert receipt["precision_correction"]["source_weights_manifest_sha256"] == (RAW_WEIGHTS_SHA256)
    assert receipt["output"]["sidecar_sha256"] == SIDECARS
    assert receipt["staging"]["acceptance_manifest_sha256"].startswith("sha256:")
    assert receipt["export_receipt_sha256"] == digest_json(
        {key: value for key, value in receipt.items() if key != "export_receipt_sha256"}
    )
    run_observation = _artifacts()[2]
    assert (
        run_observation["rayjob"]["entrypoint_sha256"]
        == run_observation["request_identity"]["command_sha256"]
    )
    assert run_observation["pod"]["resolved_image_digest"] == "sha256:" + "1" * 64
    assert run_observation["zero_step_evidence"]["metrics"] == {
        "api_observation_sha256": run_observation["zero_step_evidence"]["metrics"][
            "api_observation_sha256"
        ],
        "request_owned_config_sha256": run_observation["zero_step_evidence"]["metrics"][
            "request_owned_config_sha256"
        ],
        "rayjob_fleet_run_config_sha256": run_observation["zero_step_evidence"]["metrics"][
            "rayjob_fleet_run_config_sha256"
        ],
        "checked_step_fields": {
            "current_step": 318,
            "latest_metrics.global_step": 318,
            "latest_metrics.optimizer_steps": 0,
        },
        "reported_optimizer_steps": 0,
    }


@pytest.mark.parametrize(
    "kind",
    (
        "evil_path",
        "extra_path",
        "missing_path",
        "duplicate_path",
        "changed_index_hash",
        "changed_sidecar_hash",
        "wrong_shard_aggregate",
        "wrong_count",
        "wrong_total",
        "wrong_exclusions",
    ),
)
def test_final_assembly_rejects_resigned_base_artifact_manifest_tamper(kind):
    artifacts = list(copy.deepcopy(_artifacts()))
    cast_receipt = artifacts[4]
    base_source = cast_receipt["frozen_base_auxiliary_source"]
    tampered = copy.deepcopy(base_source["inference_artifact_manifest_before"])
    _tamper_base_artifact_manifest(tampered, kind)
    base_source["inference_artifact_manifest_before"] = tampered
    base_source["inference_artifact_manifest_after"] = copy.deepcopy(tampered)
    base_source["full_manifest_before_sha256"] = tampered["manifest_sha256"]
    base_source["full_manifest_after_sha256"] = tampered["manifest_sha256"]
    cast_receipt["cast_receipt_sha256"] = digest_json(
        {key: value for key, value in cast_receipt.items() if key != "cast_receipt_sha256"}
    )

    with pytest.raises(ValueError, match="base inference artifact"):
        _assemble(tuple(artifacts))


@pytest.mark.parametrize(
    "field",
    ("base_weights_manifest_sha256", "serving_registration_sha256"),
)
def test_final_receipt_rejects_re_signed_frozen_omission_binding_tamper(field):
    receipt = copy.deepcopy(_assemble(_artifacts()))
    correction = receipt["precision_correction"]
    correction[field] = "sha256:" + "f" * 64
    correction["restoration_binding_sha256"] = digest_json(
        {
            "restoration_rows_sha256": correction["restoration_rows_sha256"],
            "omission_policy_sha256": correction["omission_policy_sha256"],
            "base_weights_manifest_sha256": correction["base_weights_manifest_sha256"],
            "serving_registration_sha256": correction["serving_registration_sha256"],
            "restored_auxiliary_tensors": correction["restored_auxiliary_tensors"],
        }
    )
    receipt["export_receipt_sha256"] = digest_json(
        {key: value for key, value in receipt.items() if key != "export_receipt_sha256"}
    )
    with pytest.raises(ValueError, match="precision correction"):
        validate_hf_export_receipt(
            receipt,
            _artifacts()[0],
            expected_tokenizer_manifest_sha256=TOKENIZER_SHA256,
            expected_chat_template_sha256=CHAT_SHA256,
            expected_config_sha256=CONFIG_SHA256,
            expected_export_binding=_binding(),
            expected_runtime_sidecar_sha256=SIDECARS,
        )


@pytest.mark.parametrize("field", ("restoration_rows_sha256", "exact_omission_evidence_sha256"))
def test_final_receipt_rejects_re_signed_omission_evidence_digest_tamper(field):
    receipt = copy.deepcopy(_assemble(_artifacts()))
    receipt["precision_correction"][field] = "sha256:" + "f" * 64
    receipt["export_receipt_sha256"] = digest_json(
        {key: value for key, value in receipt.items() if key != "export_receipt_sha256"}
    )
    with pytest.raises(ValueError, match="precision correction"):
        validate_hf_export_receipt(
            receipt,
            _artifacts()[0],
            expected_tokenizer_manifest_sha256=TOKENIZER_SHA256,
            expected_chat_template_sha256=CHAT_SHA256,
            expected_config_sha256=CONFIG_SHA256,
            expected_export_binding=_binding(),
            expected_runtime_sidecar_sha256=SIDECARS,
        )


@pytest.mark.parametrize(
    ("index", "mutation", "message"),
    [
        (2, lambda value: value.__setitem__("optimizer_steps", 1), "executed optimizer steps"),
        (
            7,
            lambda value: value["execution"].__setitem__(
                "image", "registry.example/wrong@sha256:" + "f" * 64
            ),
            "staging image differs",
        ),
        (
            7,
            lambda value: value["composition"]["inspection"].__setitem__(
                "weights_manifest_sha256", "sha256:" + "1" * 64
            ),
            "weights differ",
        ),
    ],
)
def test_assembly_fails_closed_on_zero_step_or_staging_mismatch(index, mutation, message):
    artifacts = list(copy.deepcopy(_artifacts()))
    mutation(artifacts[index])
    digest_field = "observation_sha256" if index == 2 else "staging_receipt_sha256"
    artifacts[index][digest_field] = digest_json(
        {key: value for key, value in artifacts[index].items() if key != digest_field}
    )
    with pytest.raises(ValueError, match=message):
        _assemble(artifacts)


def test_assembly_rejects_tampered_embedded_receipt_before_field_use():
    artifacts = list(_artifacts())
    artifacts[6]["destination"]["path"] = "/models/collision"
    with pytest.raises(ValueError, match="stage_input_sha256 digest mismatch"):
        _assemble(artifacts)


def test_assembly_rejects_self_asserted_or_tampered_per_tensor_cast_proof():
    artifacts = list(copy.deepcopy(_artifacts()))
    cast_receipt = artifacts[4]
    cast_receipt["conversion"]["cast_rows"][0]["exact_cast_bits_verified"] = False
    cast_receipt["conversion"]["cast_rows_sha256"] = digest_json(
        cast_receipt["conversion"]["cast_rows"]
    )
    cast_receipt["cast_receipt_sha256"] = digest_json(
        {key: value for key, value in cast_receipt.items() if key != "cast_receipt_sha256"}
    )
    with pytest.raises(ValueError, match="per-tensor cast proof is incomplete"):
        _assemble(artifacts)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value["request"].__setitem__("kind", "rl"), "kind must be sft"),
        (
            lambda value: value["request"]["model"].__setitem__("precision", "fp8"),
            "precision must be exactly bf16",
        ),
        (
            lambda value: value["request"]["sft"].__setitem__("strategy", "deepspeed"),
            "strategy must be fsdp",
        ),
        (
            lambda value: value["request"]["trainer"].__setitem__(
                "trainer_version_id", "other-trainer"
            ),
            "expected execution identity is not exact",
        ),
        (
            lambda value: value["expected_execution"].__setitem__(
                "command_sha256", "sha256:" + "f" * 64
            ),
            "expected execution identity is not exact",
        ),
    ],
)
def test_assembler_compares_every_request_owned_execution_field(mutation, message):
    artifacts = list(copy.deepcopy(_artifacts()))
    mutation(artifacts[1])
    artifacts[1]["export_request_receipt_sha256"] = digest_json(
        {
            key: value
            for key, value in artifacts[1].items()
            if key != "export_request_receipt_sha256"
        }
    )
    with pytest.raises(ValueError, match=message):
        _assemble(artifacts)


def test_assembler_rejects_arbitrary_observed_command_digest():
    artifacts = list(copy.deepcopy(_artifacts()))
    artifacts[2]["command_sha256"] = "sha256:" + "e" * 64
    artifacts[2]["rayjob"]["entrypoint_sha256"] = "sha256:" + "e" * 64
    artifacts[2]["request_identity"]["command_sha256"] = "sha256:" + "e" * 64
    artifacts[2]["observation_sha256"] = digest_json(
        {key: value for key, value in artifacts[2].items() if key != "observation_sha256"}
    )
    with pytest.raises(ValueError, match="exact expected command"):
        _assemble(artifacts)


def _collector_replay_inputs():
    return _collector_inputs_for_request(_artifacts()[1])


def test_export_observation_collector_fails_closed_on_runtime_or_metric_drift():
    values = list(_collector_replay_inputs())
    values[3]["status"]["containerStatuses"][0]["imageID"] = (
        "resolved.example/trainer@sha256:" + "f" * 64
    )
    with pytest.raises(ValueError, match="different trainer image digest"):
        collect_zero_step_export_run_observation(*values)

    values = list(_collector_replay_inputs())
    values[6]["step_metrics"] = [{"step": 319, "loss": 0.1}]
    with pytest.raises(ValueError, match="beyond the resume boundary"):
        collect_zero_step_export_run_observation(*values)

    values = list(_collector_replay_inputs())
    values[7] += b"optimizer_step=319\n"
    with pytest.raises(ValueError, match="optimizer event"):
        collect_zero_step_export_run_observation(*values)

    values = list(_collector_replay_inputs())
    values[1]["metadata"]["uid"] = "replacement-rayjob"
    with pytest.raises(ValueError, match="RayJob identity differs"):
        collect_zero_step_export_run_observation(*values)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda v: v[1]["metadata"]["labels"].__setitem__(
                "kueue.x-k8s.io/queue-name", "other-queue"
            ),
            "queue differs",
        ),
        (lambda v: v[1]["spec"].__setitem__("entrypoint", "python wrong.py"), "entrypoint"),
        (
            lambda v: v[1]["spec"]["rayClusterSpec"]["headGroupSpec"]["template"]["spec"][
                "containers"
            ][0].__setitem__("image", "wrong/image@sha256:" + "f" * 64),
            "head image",
        ),
        (
            lambda v: v[1]["spec"]["rayClusterSpec"]["headGroupSpec"]["template"]["spec"][
                "containers"
            ][0].__setitem__("command", ["python", "wrong.py"]),
            "unreviewed container command",
        ),
        (
            lambda v: v[1]["spec"]["rayClusterSpec"]["headGroupSpec"]["template"][
                "spec"
            ].__setitem__("serviceAccountName", "privileged"),
            "service account",
        ),
        (
            lambda v: v[1]["spec"]["rayClusterSpec"]["workerGroupSpecs"][0].__setitem__(
                "replicas", 2
            ),
            "worker replicas",
        ),
        (
            lambda v: v[1]["spec"]["rayClusterSpec"]["workerGroupSpecs"][0]["template"]["spec"][
                "containers"
            ][0]["resources"]["requests"].__setitem__("nvidia.com/gpu", "7"),
            "worker resources",
        ),
        (
            lambda v: v[1]["spec"]["submitterPodTemplate"]["spec"].__setitem__(
                "serviceAccountName", "privileged"
            ),
            "submitter template",
        ),
        (
            lambda v: v[2]["metadata"]["ownerReferences"][0].__setitem__("uid", "wrong"),
            "RayCluster.*ownership",
        ),
        (
            lambda v: v[3]["metadata"]["ownerReferences"][0].__setitem__("uid", "wrong"),
            "head Pod.*ownership",
        ),
        (
            lambda v: v[4]["metadata"]["ownerReferences"][0].__setitem__("uid", "wrong"),
            "submitter Job.*ownership",
        ),
        (
            lambda v: v[5]["metadata"]["ownerReferences"][0].__setitem__("uid", "wrong"),
            "submitter Pod.*ownership",
        ),
        (
            lambda v: v[6]["config"].__setitem__("unreviewed_execution_toggle", True),
            "exact server-normalized config",
        ),
        (
            lambda v: v[6]["config"].__setitem__("run_id", "wrong-run"),
            "stored config run identity",
        ),
        (lambda v: v[6].__setitem__("id", "wrong-run"), "API run identity"),
        (lambda v: v[6].__setitem__("current_step", 319), "beyond the resume boundary"),
        (lambda v: v[6].__setitem__("optimizer_steps", 1), "optimizer activity"),
    ],
)
def test_export_observation_rejects_every_reviewed_execution_mutation(mutation, message):
    values = list(copy.deepcopy(_collector_replay_inputs()))
    mutation(values)
    with pytest.raises(ValueError, match=message):
        collect_zero_step_export_run_observation(*values)


@pytest.mark.parametrize(
    "event",
    [
        "Started: 'optim_step'",
        "Step 319: loss=0.1",
        "global_step=319",
        '"global_step": 319',
        '"current_step": 319',
        "latest step: 319",
    ],
)
def test_export_observation_rejects_optimizer_and_step_log_variants(event):
    values = list(_collector_replay_inputs())
    values[7] += (event + "\n").encode()
    with pytest.raises(ValueError, match="optimizer event|steps after the resume boundary"):
        collect_zero_step_export_run_observation(*values)


def test_export_observation_rejects_log_without_terminal_success_marker():
    values = list(_collector_replay_inputs())
    entrypoint = render_zero_step_sft_command(values[0]["request"], "ft-run-export")
    values[7] = ("submitted exact entrypoint: " + entrypoint + "\n").encode()
    with pytest.raises(ValueError, match="terminal success marker"):
        collect_zero_step_export_run_observation(*values)


def test_export_observation_accepts_exact_ray_cli_terminal_success_marker():
    values = list(_collector_replay_inputs())
    entrypoint = render_zero_step_sft_command(values[0]["request"], "ft-run-export")
    values[7] = (
        "submitted exact entrypoint: "
        + entrypoint
        + "\nSFT training complete!\n\x1b[32mJob 'ft-run-export-abcde' succeeded\x1b[39m\n"
    ).encode()
    receipt = collect_zero_step_export_run_observation(*values)
    assert receipt["zero_step_evidence"]["optimizer_step_events"] == 0


def test_export_evidence_submission_is_create_only_and_immutable():
    root = Path(__file__).resolve().parents[1]
    script = (root / "evals/post_sft/scripts/submit_evidence_v4.sh").read_text()
    manifest = (root / "evals/post_sft/cluster/qwen36-sft-evidence-v4-job.yaml").read_text()
    assert "kubectl apply" not in script
    assert 'value["immutable"]=True' in script
    assert script.count("kubectl create --dry-run=server") >= 2
    assert 'kubectl create -f "$config_map"' in script
    assert 'kubectl create -f "$JOB"' in script
    assert script.count("require_absent") >= 3
    assert "validate-evidence-bundle" in script
    assert "structural_before=" in manifest
    assert "structural_after=" in manifest
    assert "latest_before=" in manifest
    assert "latest_after=" in manifest
    assert manifest.index("raw-export-full-manifest.json") < manifest.index("structural_after=")
    assert 'test "$structural_after" = "$structural_before"' in manifest
    assert 'test "$latest_after" = "$latest_before"' in manifest
    assert "runtime-provenance.json" in manifest
    assert "serviceAccountName: chris-cyber-qwen36-sft-evidence-observer-v4" in manifest
