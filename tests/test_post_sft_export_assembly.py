import copy
import hashlib
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import training.post_sft_cli as post_sft_cli
from training.io import digest_json
from training.post_sft import (
    assemble_hf_export_receipt,
    build_zero_step_hf_export_request,
    freeze_final_promoted_checkpoint,
    normalize_zero_step_stored_config,
    render_zero_step_sft_command,
)
from training.post_sft_cli import app
from training.post_sft_export_observation import collect_zero_step_export_run_observation
from training.post_sft_staging import STAGING_COMMAND_SHA256 as REAL_STAGING_COMMAND_SHA256
from training.post_sft_staging import STAGING_IMAGE as REAL_STAGING_IMAGE

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
COMPOSED_FILES_SHA256 = "sha256:" + "a" * 64
SIDECARS = {"config.json": CONFIG_SHA256, "tokenizer.json": "sha256:" + "b" * 64}
STAGING_CODE_SHA256 = {
    "training/__init__.py": "sha256:" + "c" * 64,
    "training/io.py": "sha256:" + "d" * 64,
    "training/post_sft_artifacts.py": "sha256:" + "e" * 64,
    "training/post_sft_staging.py": "sha256:" + "f" * 64,
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
    stage_input_file_sha256 = "sha256:" + hashlib.sha256(
        (json.dumps(stage_input, indent=2, sort_keys=True) + "\n").encode()
    ).hexdigest()
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
    }


def _collector_inputs_for_request(request):
    entrypoint = render_zero_step_sft_command(request["request"], "ft-run-export")
    stored_config = normalize_zero_step_stored_config(
        request["request"], request["export_run"]
    )
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
        "dtype": "bf16",
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
            "full_file_manifest_sha256": SOURCE_SHA256,
            "output_inspection": raw_inspection,
            "raw_export_full_manifest_sha256": RAW_FILES_SHA256,
        },
        "observation_sha256",
    )
    raw_full_manifest = {
        "schema": "cyber_sft_full_file_manifest_v1",
        "root": _binding()["expected_output_path"],
        "file_count": 1,
        "total_bytes": 10,
        "files": [{"path": "model.safetensors", "size": 10, "sha256": "f" * 64}],
        "manifest_sha256": RAW_FILES_SHA256,
    }
    stage_input = _sign(
        {
            "schema": "cyber_sft_inference_stage_input_v1",
            "source": {
                "sfs_path": _binding()["expected_output_path"],
                "raw_full_manifest": raw_full_manifest,
                "raw_inspection": raw_inspection,
                "observation_sha256": export_observation["observation_sha256"],
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
        "weights_manifest_sha256": RAW_WEIGHTS_SHA256,
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
            "source_raw_manifest_sha256": RAW_FILES_SHA256,
            "execution": _stage_runtime_execution(stage_input),
            "composition": {
                "policy": "raw_post_weights_and_index_plus_exact_base_runtime_sidecars_v1",
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
    return selection, request, run_observation, export_observation, stage_input, staging_receipt


def _assemble(artifacts):
    return assemble_hf_export_receipt(
        *artifacts,
        expected_tokenizer_manifest_sha256=TOKENIZER_SHA256,
        expected_chat_template_sha256=CHAT_SHA256,
        expected_config_sha256=CONFIG_SHA256,
        expected_export_binding=_binding(),
        expected_runtime_sidecar_sha256=SIDECARS,
        expected_tokenizer_equivalence_evidence_sha256="sha256:" + "0" * 64,
        expected_staging_image=STAGING_IMAGE,
        expected_staging_command_sha256=STAGING_COMMAND_SHA256,
    )


def test_assembly_builds_and_self_validates_final_export_receipt():
    receipt = _assemble(_artifacts())
    assert receipt["schema"] == "cyber_sft_hf_export_v1"
    assert receipt["conversion"]["optimizer_steps"] == 0
    assert receipt["output"]["weights_manifest_sha256"] == RAW_WEIGHTS_SHA256
    assert receipt["output"]["sidecar_sha256"] == SIDECARS
    assert receipt["staging"]["acceptance_manifest_sha256"].startswith("sha256:")
    assert receipt["export_receipt_sha256"] == digest_json(
        {key: value for key, value in receipt.items() if key != "export_receipt_sha256"}
    )
    run_observation = _artifacts()[2]
    assert run_observation["rayjob"]["entrypoint_sha256"] == run_observation[
        "request_identity"
    ]["command_sha256"]
    assert run_observation["pod"]["resolved_image_digest"] == "sha256:" + "1" * 64
    assert run_observation["zero_step_evidence"]["metrics"] == {
        "api_observation_sha256": run_observation["zero_step_evidence"]["metrics"][
            "api_observation_sha256"
        ],
        "request_owned_config_sha256": run_observation["zero_step_evidence"]["metrics"][
            "request_owned_config_sha256"
        ],
        "rayjob_fleet_run_config_sha256": run_observation["zero_step_evidence"][
            "metrics"
        ]["rayjob_fleet_run_config_sha256"],
        "checked_step_fields": {
            "current_step": 318,
            "latest_metrics.global_step": 318,
            "latest_metrics.optimizer_steps": 0,
        },
        "reported_optimizer_steps": 0,
    }


@pytest.mark.parametrize(
    ("index", "mutation", "message"),
    [
        (2, lambda value: value.__setitem__("optimizer_steps", 1), "executed optimizer steps"),
        (
            5,
            lambda value: value["execution"].__setitem__(
                "image", "registry.example/wrong@sha256:" + "f" * 64
            ),
            "staging image differs",
        ),
        (
            5,
            lambda value: value["composition"]["inspection"].__setitem__(
                "weights_manifest_sha256", "sha256:" + "0" * 64
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
    artifacts[4]["destination"]["path"] = "/models/collision"
    with pytest.raises(ValueError, match="stage_input_sha256 digest mismatch"):
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
            lambda v: v[1]["spec"]["rayClusterSpec"]["headGroupSpec"]["template"][
                "spec"
            ]["containers"][0].__setitem__("image", "wrong/image@sha256:" + "f" * 64),
            "head image",
        ),
        (
            lambda v: v[1]["spec"]["rayClusterSpec"]["headGroupSpec"]["template"][
                "spec"
            ]["containers"][0].__setitem__("command", ["python", "wrong.py"]),
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
            lambda v: v[1]["spec"]["rayClusterSpec"]["workerGroupSpecs"][0]["template"][
                "spec"
            ]["containers"][0]["resources"]["requests"].__setitem__("nvidia.com/gpu", "7"),
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


def test_export_collector_shell_keeps_bearer_token_out_of_argv_and_logs_exact_pod():
    script = (
        Path(__file__).resolve().parents[1]
        / "evals/post_sft/scripts/collect_export_run_observation.sh"
    ).read_text(encoding="utf-8")
    assert '--header "Authorization: Bearer $FLEET_TRAINING_API_TOKEN"' not in script
    assert '--config "$CURL_CONFIG"' in script
    assert 'chmod 0600 "$CURL_CONFIG"' in script
    assert 'logs "pod/$SUBMITTER_POD" -c ray-job-submitter' in script
    assert "terminal-capture)" in script
    assert 'mkdir -m 0700 -- "$SNAPSHOT_DIR/terminal"' in script
    assert 'ln -- "$TEMP_DIR/$name" "$SNAPSHOT_DIR/terminal/$name"' in script
    assert '--terminal-rayjob "$SNAPSHOT_DIR/terminal/rayjob-terminal.json"' in script
    assert '--api-run "$SNAPSHOT_DIR/terminal/api-run.json"' in script
    assert '--driver-log "$SNAPSHOT_DIR/terminal/driver.log"' in script
    for snapshot in (
        "raycluster-runtime.json",
        "submitter-job.json",
        "submitter-pod.json",
    ):
        assert snapshot in script


def test_assemble_export_cli_writes_the_only_validated_aggregate(tmp_path):
    artifacts = list(_artifacts())
    artifacts[4]["execution"] = _stage_execution_plan(
        REAL_STAGING_IMAGE, REAL_STAGING_COMMAND_SHA256
    )
    artifacts[4]["stage_input_sha256"] = digest_json(
        {key: value for key, value in artifacts[4].items() if key != "stage_input_sha256"}
    )
    artifacts[5]["stage_input_sha256"] = artifacts[4]["stage_input_sha256"]
    artifacts[5]["execution"] = _stage_runtime_execution(
        artifacts[4], REAL_STAGING_IMAGE, REAL_STAGING_COMMAND_SHA256
    )
    artifacts[5]["staging_receipt_sha256"] = digest_json(
        {key: value for key, value in artifacts[5].items() if key != "staging_receipt_sha256"}
    )
    plan = {
        "schema": "cyber_post_sft_eval_plan_v1",
        "base_model": {
            "tokenizer_manifest_sha256": TOKENIZER_SHA256,
            "chat_template_sha256": CHAT_SHA256,
            "config_sha256": CONFIG_SHA256,
            "runtime_sidecar_sha256": SIDECARS,
            "tokenizer_equivalence_evidence": {"sha256": "sha256:" + "0" * 64},
        },
        "serving": {"engine_image": REAL_STAGING_IMAGE},
        "export": _binding(),
    }
    names = (
        "selection",
        "export-request",
        "export-run-observation",
        "export-observation",
        "stage-input",
        "staging-receipt",
    )
    paths = {}
    for name, value in zip(names, artifacts, strict=True):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        paths[name] = path
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    output = tmp_path / "final-export.json"

    cli_args = [
        "assemble-export",
        "--plan",
        str(plan_path),
        "--selection",
        str(paths["selection"]),
        "--export-request",
        str(paths["export-request"]),
        "--export-run-observation",
        str(paths["export-run-observation"]),
        "--export-observation",
        str(paths["export-observation"]),
        "--stage-input",
        str(paths["stage-input"]),
        "--staging-receipt",
        str(paths["staging-receipt"]),
        "--output",
        str(output),
    ]
    result = CliRunner().invoke(app, cli_args)
    assert result.exit_code == 0, result.exception
    assert json.loads(output.read_text())["schema"] == "cyber_sft_hf_export_v1"

    original = output.read_bytes()
    collision = CliRunner().invoke(app, cli_args)
    assert collision.exit_code != 0
    assert "pre-existing output path" in str(collision.exception)
    assert output.read_bytes() == original

    for name, make_collision in (
        ("directory", lambda path: path.mkdir()),
        ("symlink", lambda path: path.symlink_to(output)),
        ("broken-symlink", lambda path: path.symlink_to(tmp_path / "missing")),
    ):
        target = tmp_path / name
        make_collision(target)
        args = list(cli_args)
        args[-1] = str(target)
        blocked = CliRunner().invoke(app, args)
        assert blocked.exit_code != 0
        assert "pre-existing output path" in str(blocked.exception)


def test_immutable_output_publication_loses_a_creation_race_without_overwrite(
    tmp_path, monkeypatch
):
    output = tmp_path / "receipt.json"

    def competing_link(_source, destination):
        destination = type(output)(destination)
        destination.write_text("competitor\n")
        raise FileExistsError(destination)

    monkeypatch.setattr(post_sft_cli.os, "link", competing_link)
    with pytest.raises(ValueError, match="pre-existing output path"):
        post_sft_cli._atomic_write_json_new(output, {"ours": True})
    assert output.read_text() == "competitor\n"


def test_multi_output_preflight_rejects_files_directories_and_dangling_symlinks(tmp_path):
    available = tmp_path / "available.json"
    for name, make_collision in (
        ("file", lambda path: path.write_text("immutable\n")),
        ("directory", lambda path: path.mkdir()),
        ("broken-symlink", lambda path: path.symlink_to(tmp_path / "missing")),
    ):
        collision = tmp_path / name
        make_collision(collision)
        with pytest.raises(ValueError, match="pre-existing output path"):
            post_sft_cli._require_outputs_absent([available, collision])
        assert not available.exists()


def test_export_evidence_submission_is_create_only_and_immutable():
    script = (
        Path(__file__).resolve().parents[1]
        / "evals/post_sft/scripts/submit_evidence.sh"
    ).read_text()
    assert "kubectl apply" not in script
    assert 'value["immutable"]=True' in script
    assert script.count('kubectl create --dry-run=server') >= 2
    assert 'kubectl create -f "$config_map"' in script
    assert 'kubectl create -f "$JOB"' in script
    assert script.count("require_absent") >= 3
