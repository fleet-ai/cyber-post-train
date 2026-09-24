#!/usr/bin/env python3
"""Prepare, but never launch, the Qwen3.8 seed-44 Fleet two-arm packets.

The only runtime input is one fresh prompt-free base/Fresh75 receipt from
``training.checkpoint_serving_parity``.  The script verifies that receipt,
materializes immutable ConfigMap and Job source, seals one held-out launch
packet per arm, and runs the repository's offline packet validator.  It has no
Kubernetes, Fleet, PostgreSQL, or serving client.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from evals.fleet import heldout_launch

ROOT = Path(__file__).resolve().parents[1]
READINESS = (
    ROOT / "configs/evaluation/qwen38-fleet-dev17-seed44-fresh75-setup-successor-readiness-v1.json"
)
PRIMARY_FRESH_CONFIG = (
    ROOT / "configs/evaluation/qwen38-fresh75-fleet-dev17-opencode-seed44-pass1-v1.json"
)
PRIMARY_FRESH_ARTIFACT = (
    ROOT / "configs/evaluation/qwen38-fresh75-fleet-dev17-opencode-seed44-pass1-v1.artifact.json"
)
LIVE_PARITY_SCHEMA = "cyber_checkpoint_serving_live_parity_v2"
EXPECTED_SERVING_IMAGE = "sha256:d6e7288627be8b02be88e4bba38e73f6d50e2826869f753c13a4c4385ab3eda9"
EXPECTED_DIND_IMAGE = (
    "docker.io/library/docker@"
    "sha256:f649ef046008ca7f926a2571c32b0ac22e5c59eb61b959617f9acc2a4c638cf5"
)
EXPECTED_EVALUATOR_IMAGE = (
    "ghcr.io/astral-sh/uv:python3.12-bookworm@"
    "sha256:9aa60c50016c0485636ab9a830246a6ef3399aa4a8bab3d17ef4a2358fba2ca7"
)
MAX_PROOF_AGE_SECONDS = 3600
MAX_FUTURE_SKEW_SECONDS = 300
UID = re.compile(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}")
SHA256 = re.compile(r"sha256:[0-9a-f]{64}")
SOURCE_FILES = {
    "jobs.py": ROOT / "cyber_post_train/jobs.py",
    "cluster_entry.py": ROOT / "evals/fleet/cluster_entry.py",
    "evaluate.py": ROOT / "evals/fleet/evaluate.py",
    "exact_pass4_crypto.py": ROOT / "evals/fleet/exact_pass4_crypto.py",
    "exact_pass4_universe.py": ROOT / "evals/fleet/exact_pass4_universe.py",
    "fixed_proxy.py": ROOT / "evals/fleet/fixed_proxy.py",
    "model_artifact.py": ROOT / "evals/fleet/model_artifact.py",
    "model_artifact_v2.py": ROOT / "evals/fleet/model_artifact_v2.py",
    "model_artifact_v3.py": ROOT / "evals/fleet/model_artifact_v3.py",
    "opencode_self_hosted.py": ROOT / "evals/fleet/opencode_self_hosted.py",
    "rollout_campaign.py": ROOT / "evals/fleet/rollout_campaign.py",
    "rollout_ledger.py": ROOT / "evals/fleet/rollout_ledger.py",
    "rollout_postgres.py": ROOT / "evals/fleet/rollout_postgres.py",
    "rollout_worker.py": ROOT / "evals/fleet/rollout_worker.py",
    "run.sh": ROOT / "evals/fleet/scripts/run_qwen38_dev17_single_arm_v3.sh",
}


def _canonical_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be one regular nonsymlink file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not readable JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be one JSON object")
    return value


def _repo_path(value: Any, label: str) -> Path:
    if (
        not isinstance(value, str)
        or not value
        or value.startswith("/")
        or ".." in Path(value).parts
    ):
        raise ValueError(f"{label} must be one repository-relative path")
    path = (ROOT / value).resolve()
    if ROOT not in path.parents or not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} is not one regular repository file")
    return path


def _verify_self_digest(value: dict[str, Any], label: str, *, field: str = "sha256") -> None:
    expected = value.get(field)
    if not isinstance(expected, str) or SHA256.fullmatch(expected) is None:
        raise ValueError(f"{label} has no prefixed self digest")
    actual = _canonical_digest({key: item for key, item in value.items() if key != field})
    if actual != expected:
        raise ValueError(f"{label} self digest differs")


def _verify_file_binding(binding: dict[str, Any], label: str) -> Path:
    path = _repo_path(binding.get("path"), f"{label} path")
    if _file_sha256(path) != binding.get("file_sha256"):
        raise ValueError(f"{label} file digest differs")
    return path


def _validate_setup_successor(
    readiness: dict[str, Any],
    *,
    protocol: dict[str, Any],
    config: dict[str, Any],
    artifact: dict[str, Any],
) -> None:
    """Prove the Fresh75 v2 identity is only a zero-claim setup retry."""

    review = readiness.get("reviewed_setup_successor")
    arm = readiness.get("arms", {}).get("fresh75")
    if (
        readiness.get("schema")
        != "cyber_qwen38_fleet_dev17_seed44_fresh75_setup_successor_readiness_v1"
        or readiness.get("status") != "source_prepared_setup_successor_not_launchable"
        or readiness.get("launchable") is not False
        or not isinstance(review, dict)
        or not isinstance(arm, dict)
        or protocol.get("retry_limit") != 1
    ):
        raise ValueError("Fresh75 setup-successor readiness is invalid")
    primary = review.get("primary")
    cause = review.get("cause")
    retry = review.get("retry_authority")
    successor = review.get("successor")
    if (
        not isinstance(primary, dict)
        or primary.get("job_name") != "chris-q38-dev17-s44-fresh75-p1-v1"
        or primary.get("root_failure_alerts") != "off"
        or primary.get("job_failed") != 1
        or primary.get("pod_restart_count") != 0
        or primary.get("evaluator_exit_code") != 1
        or primary.get("gpu_requests") != 0
        or primary.get("output_root_absent") is not True
        or not isinstance(cause, dict)
        or cause.get("error_type") != "OSError"
        or cause.get("errno") != 40
        or cause.get("message")
        != "Too many levels of symbolic links: /bootstrap/model-artifact.json"
        or cause.get("failure_preceded_output_fleet_and_database_side_effects") is not True
        or cause.get("fresh75_cells_or_sessions_created") != 0
        or retry
        != {
            "protocol_retry_limit": 1,
            "automatic_retry": False,
            "valid_outcome_replayed": False,
            "same_seed_tasks_harness_sampling_and_model_revision": True,
        }
        or successor
        != {
            "job_name": arm.get("job_name"),
            "config_map_name": arm.get("config_map_name"),
            "output_root": arm.get("output_root"),
            "database": arm.get("database"),
        }
        or review.get("authorization") != "source_preparation_only_not_launch_authorization"
    ):
        raise ValueError("Fresh75 setup-successor evidence is incomplete")

    repair = review.get("repair")
    if not isinstance(repair, dict):
        raise ValueError("Fresh75 setup-successor repair is invalid")
    for path_field, digest_field in (
        ("single_arm_wrapper", "single_arm_wrapper_file_sha256"),
        ("matched_wrapper", "matched_wrapper_file_sha256"),
    ):
        path = _repo_path(repair.get(path_field), f"Fresh75 repair {path_field}")
        if _file_sha256(path) != repair.get(digest_field):
            raise ValueError("Fresh75 setup-successor wrapper bytes changed")

    primary_config = _read_json(PRIMARY_FRESH_CONFIG, "Fresh75 primary config")
    successor_config = dict(config)
    if (
        primary_config.pop("name") != "q38-dev17-s44-fresh75-p1-v1"
        or successor_config.pop("name") != "q38-dev17-s44-fresh75-p1-v2"
    ):
        raise ValueError("Fresh75 setup-successor campaign identity is invalid")
    primary_binding = dict(primary_config.pop("model_artifact_binding"))
    successor_binding = dict(successor_config.pop("model_artifact_binding"))
    for field in ("packet_path", "packet_file_sha256", "packet_sha256"):
        primary_binding.pop(field)
        successor_binding.pop(field)
    if primary_config != successor_config or primary_binding != successor_binding:
        raise ValueError("Fresh75 setup successor changes scientific configuration")

    primary_artifact = _read_json(PRIMARY_FRESH_ARTIFACT, "Fresh75 primary artifact")
    primary_artifact.pop("campaign_name")
    primary_artifact.pop("sha256")
    successor_artifact = dict(artifact)
    if successor_artifact.pop("campaign_name") != "q38-dev17-s44-fresh75-p1-v2":
        raise ValueError("Fresh75 setup-successor artifact campaign is invalid")
    successor_artifact.pop("sha256")
    if primary_artifact != successor_artifact:
        raise ValueError("Fresh75 setup successor changes checkpoint provenance")


def _timestamp(value: Any, label: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a timezone-aware timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} must be a timezone-aware timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} must be a timezone-aware timestamp")
    return parsed.astimezone(UTC)


def _uid_list(value: Any, label: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or len(value) != len(set(value))
        or any(not isinstance(item, str) or UID.fullmatch(item) is None for item in value)
    ):
        raise ValueError(f"{label} must contain unique Kubernetes UIDs")
    return value


def _validate_live_arm(
    value: Any,
    *,
    receipt_label: str,
    arm: dict[str, Any],
    config: dict[str, Any],
) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{receipt_label} live arm is invalid")
    expected = {
        "served_model",
        "resource_version",
        "kubernetes_resource_version",
        "model_revision",
        "source_path",
        "serving_path",
        "normalized_contract_sha256",
        "registration_spec_sha256",
        "catalog",
        "model_info",
        "server_info",
        "kubernetes",
        "probes",
    }
    if set(value) != expected:
        raise ValueError(f"{receipt_label} live arm shape is invalid")
    route = next(iter(config["routes"].values()))
    if (
        value.get("served_model") != arm["served_id"]
        or value.get("model_revision") != arm["model_revision"]
        or value.get("source_path") != arm["expected_source_path"]
        or value.get("serving_path") != route["model_info"]["model_path"]
        or not isinstance(value.get("resource_version"), str)
        or not value["resource_version"]
        or not isinstance(value.get("kubernetes_resource_version"), str)
        or not value["kubernetes_resource_version"]
    ):
        raise ValueError(f"{receipt_label} live route identity differs")
    for field in ("normalized_contract_sha256", "registration_spec_sha256"):
        if not isinstance(value[field], str) or SHA256.fullmatch(value[field]) is None:
            raise ValueError(f"{receipt_label} {field} is invalid")

    catalog = value.get("catalog")
    if not isinstance(catalog, dict) or any(
        (
            catalog.get("id") != arm["served_id"],
            catalog.get("model_revision") != arm["model_revision"],
            catalog.get("status") != "ready",
            catalog.get("routed") is not True,
            type(catalog.get("ready_replicas")) is not int,
            catalog.get("ready_replicas", 0) < 1,
            catalog.get("engine") != route["catalog"]["engine"],
            catalog.get("precision") != route["catalog"]["precision"],
            catalog.get("tensor_parallel_size") != route["catalog"]["tensor_parallel_size"],
        )
    ):
        raise ValueError(f"{receipt_label} catalog projection differs")
    model_info = value.get("model_info")
    if not isinstance(model_info, dict) or any(
        (
            model_info.get("model_type") != route["model_info"]["model_type"],
            model_info.get("architectures") != route["model_info"]["architectures"],
            model_info.get("reasoning_parser") != route["server_info"]["reasoning_parser"],
            model_info.get("tool_call_parser") != route["server_info"]["tool_call_parser"],
        )
    ):
        raise ValueError(f"{receipt_label} model-info projection differs")
    server_info = value.get("server_info")
    if not isinstance(server_info, dict) or any(
        server_info.get(field) != route["server_info"][field]
        for field in (
            "context_length",
            "tp_size",
            "dp_size",
            "quantization",
            "kv_cache_dtype",
            "reasoning_parser",
            "tool_call_parser",
        )
    ):
        raise ValueError(f"{receipt_label} server-info projection differs")

    kubernetes = value.get("kubernetes")
    if not isinstance(kubernetes, dict):
        raise ValueError(f"{receipt_label} Kubernetes projection is invalid")
    for field in ("inference_model_uid", "deployment_uid", "service_uid"):
        if not isinstance(kubernetes.get(field), str) or UID.fullmatch(kubernetes[field]) is None:
            raise ValueError(f"{receipt_label} Kubernetes {field} is invalid")
    for field in ("replicaset_uids", "pod_uids", "endpoint_slice_uids"):
        _uid_list(kubernetes.get(field), f"{receipt_label} Kubernetes {field}")
    replicas = kubernetes.get("ready_replicas")
    if (
        type(replicas) is not int
        or replicas < 1
        or len(kubernetes["pod_uids"]) != replicas
        or kubernetes.get("ready_endpoint_count") != replicas
        or kubernetes.get("restart_counts") != [0] * replicas
        or kubernetes.get("image_digest") != EXPECTED_SERVING_IMAGE
        or kubernetes.get("normalized_inference_spec_sha256") != value["normalized_contract_sha256"]
    ):
        raise ValueError(f"{receipt_label} Kubernetes route is not exact and ready")

    probes = value.get("probes")
    if not isinstance(probes, dict) or any(
        (
            probes.get("served_model") != arm["served_id"],
            probes.get("tool_call_passed") is not True,
            probes.get("tool_name") != "identity",
            probes.get("tool_argument_keys") != ["value"],
            probes.get("logits_finite") is not True,
            probes.get("logit_token_identity_deterministic") is not True,
            type(probes.get("logit_token_count")) is not int,
            probes.get("logit_token_count", 0) < 1,
        )
    ):
        raise ValueError(f"{receipt_label} content-free probes did not pass")
    for field in (
        "tool_response_sha256",
        "logit_projection_sha256",
        "logit_token_identity_sha256",
    ):
        if not isinstance(probes.get(field), str) or SHA256.fullmatch(probes[field]) is None:
            raise ValueError(f"{receipt_label} probe {field} is invalid")
    logit_receipts = probes.get("logit_response_sha256")
    if (
        not isinstance(logit_receipts, list)
        or len(logit_receipts) != 2
        or any(
            not isinstance(item, str) or SHA256.fullmatch(item) is None for item in logit_receipts
        )
    ):
        raise ValueError(f"{receipt_label} logit receipt digests are invalid")


def _live_parity(
    path: Path,
    *,
    readiness: dict[str, Any],
    configs: dict[str, dict[str, Any]],
    now: datetime,
) -> dict[str, Any]:
    value = _read_json(path, "base/Fresh75 live parity")
    expected = {
        "schema",
        "status",
        "observed_at",
        "endpoint_origin",
        "arms",
        "held_constant",
        "benchmark_content_included",
        "response_content_recorded",
        "scores_observed",
        "task_content_included",
        "external_mutations_performed",
        "resource_versions_are_non_atomic_live_observations",
        "fixed_probe_logit_projection_differs_between_weights",
        "receipt_sha256",
    }
    if set(value) != expected or value.get("schema") != LIVE_PARITY_SCHEMA:
        raise ValueError("base/Fresh75 live-parity schema is unsupported")
    _verify_self_digest(value, "base/Fresh75 live parity", field="receipt_sha256")
    observed = _timestamp(value.get("observed_at"), "live-parity observed_at")
    age = (now.astimezone(UTC) - observed).total_seconds()
    if (
        age > MAX_PROOF_AGE_SECONDS
        or age < -MAX_FUTURE_SKEW_SECONDS
        or value.get("status") != "passed"
        or value.get("endpoint_origin") != "https://inference.flt.build"
        or value.get("benchmark_content_included") is not False
        or value.get("response_content_recorded") is not False
        or value.get("scores_observed") is not False
        or value.get("task_content_included") is not False
        or value.get("external_mutations_performed") != 0
        or value.get("resource_versions_are_non_atomic_live_observations") is not True
    ):
        raise ValueError("base/Fresh75 live-parity receipt is stale or unsafe")
    arms = value.get("arms")
    if not isinstance(arms, dict) or set(arms) != {"base", "candidate"}:
        raise ValueError("base/Fresh75 live-parity arms are invalid")
    _validate_live_arm(
        arms["base"],
        receipt_label="base",
        arm=readiness["arms"]["base"],
        config=configs["base"],
    )
    _validate_live_arm(
        arms["candidate"],
        receipt_label="Fresh75",
        arm=readiness["arms"]["fresh75"],
        config=configs["fresh75"],
    )
    held = value.get("held_constant")
    if not isinstance(held, dict) or any(
        (
            held.get("normalized_contract_sha256") != arms["base"]["normalized_contract_sha256"],
            arms["candidate"]["normalized_contract_sha256"]
            != arms["base"]["normalized_contract_sha256"],
            held.get("inference_precision") != "bf16",
            held.get("max_context_size") != 262144,
            held.get("weight_quantization") != "none",
        )
    ):
        raise ValueError("base/Fresh75 held-constant serving contract differs")
    return value


def _job(arm: dict[str, Any], config_name: str, task_set_name: str) -> dict[str, Any]:
    experiment = arm["job_name"]
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": experiment,
            "namespace": heldout_launch.NAMESPACE,
            "labels": {
                "cyber-post-train.fleet.ai/experiment": experiment,
                "cyber-post-train.fleet.ai/owner": "chris",
                "kueue.x-k8s.io/queue-name": "training-lq",
            },
            "annotations": {
                heldout_launch.FAILURE_ALERT_ANNOTATION: heldout_launch.FAILURE_ALERT_OFF,
                heldout_launch.CREATE_ONCE_ANNOTATION: "true",
            },
        },
        "spec": {
            "backoffLimit": 0,
            "activeDeadlineSeconds": 172800,
            "template": {
                "metadata": {
                    "labels": {
                        "cyber-post-train.fleet.ai/experiment": experiment,
                        "cyber-post-train.fleet.ai/owner": "chris",
                        "cyber-post-train.fleet.ai/postgres-client": "true",
                    }
                },
                "spec": {
                    "priorityClassName": "c1",
                    "restartPolicy": "Never",
                    "nodeSelector": {
                        "kubernetes.io/arch": "amd64",
                        "workload": "fleetai-training-ng-cpu",
                    },
                    "tolerations": [
                        {
                            "key": "workload",
                            "operator": "Equal",
                            "value": "fleetai-training-ng-cpu",
                            "effect": "NoSchedule",
                        }
                    ],
                    "initContainers": [
                        {
                            "name": "dind",
                            "image": EXPECTED_DIND_IMAGE,
                            "restartPolicy": "Always",
                            "args": ["--host=unix:///var/run/docker.sock", "--tls=false"],
                            "securityContext": {"privileged": True},
                            "resources": {
                                "requests": {
                                    "cpu": "2",
                                    "memory": "2Gi",
                                    "ephemeral-storage": "10Gi",
                                },
                                "limits": {
                                    "cpu": "4",
                                    "memory": "4Gi",
                                    "ephemeral-storage": "40Gi",
                                },
                            },
                            "volumeMounts": [
                                {"name": "docker-socket", "mountPath": "/var/run"},
                                {"name": "docker-data", "mountPath": "/var/lib/docker"},
                                {"name": "docker-bind", "mountPath": "/docker-bind"},
                                {"name": "workspace", "mountPath": "/workspace"},
                                {"name": "sfs", "mountPath": "/mnt/sfs"},
                            ],
                        }
                    ],
                    "containers": [
                        {
                            "name": "evaluator",
                            "image": EXPECTED_EVALUATOR_IMAGE,
                            "command": ["/bin/bash", "-ceu", "--"],
                            "args": [
                                "apt-get update\n"
                                "apt-get install --yes --no-install-recommends docker.io\n"
                                "exec /bin/bash /bootstrap/run.sh\n"
                            ],
                            "env": [
                                {"name": "DOCKER_HOST", "value": "unix:///var/run/docker.sock"},
                                {"name": "DOCKER_TLS_CERTDIR", "value": ""},
                                {"name": "DOCKER_BIND_ROOT", "value": "/docker-bind"},
                                {"name": "EVAL_CONFIG_NAME", "value": config_name},
                                {"name": "EVAL_TASK_SET_NAME", "value": task_set_name},
                                {"name": "EVAL_OUTPUT", "value": arm["output_root"]},
                                {"name": "EVAL_DATABASE", "value": arm["database"]},
                                {
                                    "name": "FLEET_API_KEY",
                                    "valueFrom": {
                                        "secretKeyRef": {
                                            "name": "chris-cyber-opencode-evals-v2",
                                            "key": "FLEET_API_KEY",
                                        }
                                    },
                                },
                                {
                                    "name": "ROLLOUT_DATABASE_URL",
                                    "valueFrom": {
                                        "secretKeyRef": {
                                            "name": "chris-cyber-rollout-postgres-v1",
                                            "key": "ROLLOUT_DATABASE_URL",
                                        }
                                    },
                                },
                            ],
                            "resources": {
                                "requests": {
                                    "cpu": "4",
                                    "memory": "32Gi",
                                    "ephemeral-storage": "10Gi",
                                },
                                "limits": {
                                    "cpu": "8",
                                    "memory": "48Gi",
                                    "ephemeral-storage": "40Gi",
                                },
                            },
                            "volumeMounts": [
                                {"name": "docker-socket", "mountPath": "/var/run"},
                                {
                                    "name": "bootstrap",
                                    "mountPath": "/bootstrap",
                                    "readOnly": True,
                                },
                                {"name": "docker-bind", "mountPath": "/docker-bind"},
                                {"name": "workspace", "mountPath": "/workspace"},
                                {"name": "sfs", "mountPath": "/mnt/sfs"},
                            ],
                        }
                    ],
                    "volumes": [
                        {
                            "name": "bootstrap",
                            "configMap": {"name": arm["config_map_name"]},
                        },
                        {"name": "docker-data", "emptyDir": {"sizeLimit": "80Gi"}},
                        {"name": "docker-socket", "emptyDir": {}},
                        {"name": "docker-bind", "emptyDir": {"sizeLimit": "2Gi"}},
                        {"name": "workspace", "emptyDir": {"sizeLimit": "5Gi"}},
                        {
                            "name": "sfs",
                            "persistentVolumeClaim": {"claimName": "sfs-shared"},
                        },
                    ],
                },
            },
        },
    }


def _write(path: Path, raw: bytes) -> None:
    path.write_bytes(raw)


def _prepare_arm(
    directory: Path,
    *,
    arm_id: str,
    arm: dict[str, Any],
    config_path: Path,
    config: dict[str, Any],
    task_set_path: Path,
    split_path: Path,
    protocol_path: Path,
    protocol: dict[str, Any],
    checkpoint_path: Path,
    proof_path: Path,
    ledger_path: Path,
    source_files: dict[str, Path] | None = None,
) -> dict[str, Any]:
    local_models = {
        alias: model
        for alias, model in config.get("models", {}).items()
        if isinstance(model, dict)
        and isinstance(model.get("repository"), str)
        and model["repository"].startswith("/mnt/sfs/jobs/")
    }
    artifact_binding = config.get("model_artifact_binding")
    if local_models and artifact_binding is None:
        raise ValueError(
            f"{arm_id} local SFS model requires a staged artifact binding before render"
        )
    actual_source_files = source_files or SOURCE_FILES
    run_script = actual_source_files.get("run.sh")
    if run_script is None:
        raise ValueError(f"{arm_id} run script is not staged")
    run_text = run_script.read_text(encoding="utf-8")
    for dependency in ("model_artifact_v2.py", "model_artifact_v3.py"):
        if f"/bootstrap/{dependency}" in run_text and dependency not in actual_source_files:
            raise ValueError(f"{arm_id} runtime dependency {dependency} is not staged")
    if artifact_binding is not None:
        artifact = _read_json(checkpoint_path, f"{arm_id} model artifact packet")
        schema = artifact.get("schema")
        required: set[str] = set()
        if schema == "cyber_fleet_eval_model_artifact_packet_v2":
            required.add("model_artifact_v2.py")
        elif schema == "cyber_fleet_eval_model_artifact_packet_v3":
            required.update({"model_artifact_v2.py", "model_artifact_v3.py"})
        elif schema != "cyber_fleet_eval_model_artifact_packet_v1":
            raise ValueError(f"{arm_id} model artifact packet schema is unsupported")
        if (
            not required.issubset(actual_source_files)
            or "/bootstrap/model-artifact.json" not in run_text
        ):
            raise ValueError(f"{arm_id} artifact validator/runtime files are not staged")
    directory.mkdir(mode=0o700)
    files = {
        "evaluation_config": directory / config_path.name,
        "task_set": directory / task_set_path.name,
        "split_manifest": directory / "split-manifest.json",
        "comparison_protocol": directory / "comparison-protocol.json",
        "checkpoint_provenance": directory / "checkpoint-provenance.json",
        "serving_route_proof": directory / "serving-route-proof.json",
        "evaluation_ledger": directory / "evaluation-ledger.json",
        "config_map": directory / "config-map.yaml",
        "job": directory / "job.yaml",
    }
    for destination, source in (
        (files["evaluation_config"], config_path),
        (files["task_set"], task_set_path),
        (files["split_manifest"], split_path),
        (files["comparison_protocol"], protocol_path),
        (files["checkpoint_provenance"], checkpoint_path),
        (files["serving_route_proof"], proof_path),
        (files["evaluation_ledger"], ledger_path),
    ):
        _write(destination, source.read_bytes())
    config_text = json.dumps(config, sort_keys=True, separators=(",", ":"))
    config_name = config_path.name
    data = {name: path.read_text(encoding="utf-8") for name, path in actual_source_files.items()}
    data.update(
        {
            "config.json": config_text,
            config_name: config_text,
            "task-set.json": task_set_path.read_text(encoding="utf-8"),
        }
    )
    if artifact_binding is not None:
        data["model-artifact.json"] = checkpoint_path.read_text(encoding="utf-8")
        acceptance_path = _repo_path(
            artifact_binding.get("acceptance_evidence_path"),
            f"{arm_id} model artifact acceptance",
        )
        if _file_sha256(acceptance_path) != artifact_binding.get("acceptance_evidence_file_sha256"):
            raise ValueError(f"{arm_id} model artifact acceptance bytes changed")
        data["model-artifact-acceptance.json"] = acceptance_path.read_text(encoding="utf-8")
    config_map = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": arm["config_map_name"],
            "namespace": heldout_launch.NAMESPACE,
        },
        "immutable": True,
        "data": data,
    }
    task_set_name = config.get("task_set")
    if (
        not isinstance(task_set_name, str)
        or not task_set_name
        or Path(task_set_name).name != task_set_name
    ):
        raise ValueError(f"{arm_id} evaluation config task set must be a basename")
    job = _job(arm, config_name, task_set_name)
    files["config_map"].write_text(yaml.safe_dump(config_map, sort_keys=False), encoding="utf-8")
    files["job"].write_text(yaml.safe_dump(job, sort_keys=False), encoding="utf-8")
    file_digests = {
        name: _file_sha256(path) for name, path in files.items() if name != "evaluation_ledger"
    }
    identity = {
        "protocol_id": protocol["protocol_id"],
        "comparison_arms": protocol["comparison_arms"],
        "arm_id": arm_id,
        "evaluation_config_name": config["name"],
        "evaluation_config_sha256": file_digests["evaluation_config"],
        "task_selection_sha256": file_digests["task_set"],
        "split_manifest_file_sha256": file_digests["split_manifest"],
        "split_manifest_sha256": protocol.get(
            "split_manifest_sha256", protocol.get("split_manifest", {}).get("sha256")
        ),
        "comparison_protocol_file_sha256": file_digests["comparison_protocol"],
        "comparison_protocol_sha256": protocol["sha256"],
        "checkpoint_provenance_sha256": file_digests["checkpoint_provenance"],
        "serving_route_proof_sha256": file_digests["serving_route_proof"],
        "model_revision": arm["model_revision"],
        "harness": config["harness"]["harness"],
        "harness_version": config["harness"]["harness_version"],
        "context_management": config["harness"]["context_management"],
        "sampling_seed": config["sampling"]["seed"],
        "pass_k": config["pass_k"],
        "retry_limit": config["max_reviewed_infrastructure_retries"],
        "output_root": arm["output_root"],
        "database": arm["database"],
    }
    packet = {
        "schema": heldout_launch.PACKET_SCHEMA,
        "namespace": heldout_launch.NAMESPACE,
        "job_name": arm["job_name"],
        "config_map_name": arm["config_map_name"],
        "output_root": arm["output_root"],
        "database": arm["database"],
        "files": {
            name: {
                "path": path.name,
                **({"sha256": file_digests[name]} if name != "evaluation_ledger" else {}),
            }
            for name, path in files.items()
        },
        "evaluation_identity": identity,
        "evaluation_identity_sha256": _canonical_digest(identity),
    }
    packet_path = directory / "LAUNCH_PACKET.json"
    packet_path.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    heldout_launch.build_package(packet_path)
    return {
        "arm_id": arm_id,
        "packet_path": str(Path(arm_id) / packet_path.name),
        "packet_file_sha256": _file_sha256(packet_path),
        "evaluation_identity_sha256": packet["evaluation_identity_sha256"],
        "serving_proof_file_sha256": file_digests["serving_route_proof"],
    }


def prepare(
    *,
    output: Path,
    live_parity: Path,
    now: datetime | None = None,
) -> dict[str, Any]:
    if output.exists() or output.is_symlink():
        raise FileExistsError("packet output already exists")
    if not output.parent.is_dir():
        raise ValueError("packet output parent does not exist")
    readiness = _read_json(READINESS, "seed-44 readiness")
    _verify_self_digest(readiness, "seed-44 readiness")
    protocol_path = _verify_file_binding(readiness["protocol"], "comparison protocol")
    protocol = _read_json(protocol_path, "comparison protocol")
    _verify_self_digest(protocol, "comparison protocol")
    task_set_path = _repo_path(readiness["selection"]["task_set_path"], "task set")
    split_path = _repo_path(readiness["selection"]["split_path"], "split manifest")
    if (
        _file_sha256(task_set_path) != readiness["selection"]["task_set_file_sha256"]
        or _file_sha256(split_path) != readiness["selection"]["split_file_sha256"]
    ):
        raise ValueError("seed-44 task selection or split bytes changed")
    ledger_path = ROOT / "configs/evaluation/qwen38-checkpoint-eval-ledger-v2.json"
    configs: dict[str, dict[str, Any]] = {}
    sources: dict[str, tuple[Path, Path]] = {}
    for arm_id, arm in readiness["arms"].items():
        config_path = _repo_path(arm["config_path"], f"{arm_id} config")
        checkpoint_path = _repo_path(
            arm["checkpoint_provenance_path"], f"{arm_id} checkpoint provenance"
        )
        if (
            _file_sha256(config_path) != arm["config_file_sha256"]
            or _file_sha256(checkpoint_path) != arm["checkpoint_provenance_file_sha256"]
        ):
            raise ValueError(f"{arm_id} source bytes changed")
        config = _read_json(config_path, f"{arm_id} config")
        configs[arm_id] = config
        sources[arm_id] = (config_path, checkpoint_path)
    _validate_setup_successor(
        readiness,
        protocol=protocol,
        config=configs["fresh75"],
        artifact=_read_json(sources["fresh75"][1], "Fresh75 successor artifact"),
    )
    parity = _live_parity(
        live_parity,
        readiness=readiness,
        configs=configs,
        now=now or datetime.now(UTC),
    )
    temporary = Path(tempfile.mkdtemp(prefix=".seed44-packets-", dir=output.parent))
    try:
        arms = []
        for arm_id in protocol["comparison_arms"]:
            config = configs[arm_id]
            config_path, checkpoint_path = sources[arm_id]
            arms.append(
                _prepare_arm(
                    temporary / arm_id,
                    arm_id=arm_id,
                    arm=readiness["arms"][arm_id],
                    config_path=config_path,
                    config=config,
                    task_set_path=task_set_path,
                    split_path=split_path,
                    protocol_path=protocol_path,
                    protocol=protocol,
                    checkpoint_path=checkpoint_path,
                    proof_path=live_parity,
                    ledger_path=ledger_path,
                )
            )
        receipt = {
            "schema": "cyber_qwen38_fleet_seed44_two_arm_packet_preparation_v1",
            "protocol_id": protocol["protocol_id"],
            "comparison_protocol_sha256": protocol["sha256"],
            "live_parity_receipt_sha256": parity["receipt_sha256"],
            "live_parity_file_sha256": _file_sha256(live_parity),
            "live_parity_observed_at": parity["observed_at"],
            "arms": arms,
            "external_mutations": 0,
            "launch_performed": False,
            "credentials_read": False,
        }
        receipt["sha256"] = _canonical_digest(receipt)
        (temporary / "PREPARATION_RECEIPT.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live-parity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                output=args.output,
                live_parity=args.live_parity,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
