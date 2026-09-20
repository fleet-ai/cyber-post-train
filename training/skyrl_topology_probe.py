"""One-pod SkyRL setup probe with no task, rollout, optimizer or checkpoint path.

The probe is deliberately a different schema, name, output directory and
entrypoint from scientific training.  It may only target the development Jobs
API.  Its 20 minute setup budget plus five minute cleanup budget is a hard
25-minute process limit; Kubernetes release is accepted only by a separate
post-terminal receipt bound to the exact API and Kubernetes identities.  The
Ray head owns all eight GPUs so this probe qualifies the same one-pod topology
used by the scientific reward canary. FleetJob requires one GPU worker group,
so the supported GPU-head pattern declares an autoscaling group at zero
replicas. The probe never asks Ray to scale it, and the observer rejects any
allocation above the head's exact eight GPUs.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import signal
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from cyber_post_train.jobs import API_URLS, JobsError, bundled_request, digest, quantity

from . import skyrl

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "cyber_skyrl_topology_probe_v1"
CONFIG_SCHEMA = "cyber_skyrl_topology_probe_config_v1"
RECEIPT_SCHEMA = "cyber_skyrl_topology_probe_receipt_v1"
RELEASE_SCHEMA = "cyber_skyrl_topology_probe_release_v1"
FLEETJOB_PACKET_SCHEMA = "cyber_skyrl_topology_probe_fleetjob_packet_v1"
FLEETJOB_PREVIEW_SCHEMA = "cyber_skyrl_topology_probe_fleetjob_preview_v1"
PREFLIGHT_PACKET_SCHEMA = "cyber_skyrl_topology_probe_preflight_job_packet_v1"
PREFLIGHT_PREVIEW_SCHEMA = "cyber_skyrl_topology_probe_preflight_job_preview_v1"
PREFLIGHT_FAILURE_SCHEMA = "cyber_skyrl_topology_probe_cpu_preflight_rejection_v1"
PROBE_FAILURE_SCHEMA = "cyber_skyrl_topology_probe_failure_v1"
RECEIPT_VERIFY_SCHEMA = "cyber_skyrl_topology_probe_receipt_verification_v1"
RECEIPT_VERIFY_FAILURE_SCHEMA = "cyber_skyrl_topology_probe_receipt_verification_rejection_v1"
RECEIPT_VERIFY_PACKET_SCHEMA = "cyber_skyrl_topology_probe_receipt_verification_job_packet_v1"
RECEIPT_VERIFY_PREVIEW_SCHEMA = "cyber_skyrl_topology_probe_receipt_verification_job_preview_v1"
PREFLIGHT_NAME = "chris-q38-skyrl-probe-preflight-v24"
RECEIPT_VERIFY_NAME = "chris-q38-skyrl-probe-receipt-v13"
PREFLIGHT_RECEIPT = "/dev/termination-log"
FAILURE_RECEIPT = "TOPOLOGY_PROBE_FAILED.json"
MODULE = "training.skyrl_topology_probe"
CONFIG_PATH = ROOT / "configs/qualification/qwen38-skyrl-topology-probe-dev-v1.json"
IMAGE = (
    "661864827319.dkr.ecr.us-east-1.amazonaws.com/fleet/skyrl-train@sha256:"
    "89758df2b5f35cdb19efe948c7f6ef54f11e2e2ab47a45d600c25f36914e308f"
)
MODEL_BINDING_SHA256 = "e71d00df4e29a312746e1c17b794f18b0163ddc456406b5c02c809fb189bd9be"
RUNTIME_FILES = (
    "training/skyrl_topology_probe.py",
    "training/skyrl.py",
    "training/skyrl_episode.py",
    "training/dense.py",
    "training/io.py",
    "training/qwen_tools.py",
    "training/rl_episode.py",
    "evals/fleet/opencode_self_hosted.py",
    "cyber_post_train/jobs.py",
)


class ProbeGateError(ValueError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def _seal(value: dict) -> dict:
    body = {key: item for key, item in value.items() if key != "sha256"}
    return {**body, "sha256": "sha256:" + digest(body)}


def _validate_seal(value: dict, schema: str) -> None:
    if value.get("schema") != schema or value != _seal(value):
        raise ValueError("SkyRL topology probe digest changed")


def _runtime() -> dict[str, str]:
    return {name: (ROOT / name).read_text() for name in RUNTIME_FILES}


def _expected_execution() -> dict:
    return {
        "cluster_target": "dev",
        "jobs_api_base_url": API_URLS["dev"],
        "submission_transport": "fleetjob",
        "kubernetes_context": "nebius-mk8s-fleetai-training-dev-e04p03enwk5c0va9tb",
        "namespace": "fleet-train-jobs",
        "project_name": "fleetjob-dev",
        "auth_secret": {"name": "fleet-api", "key": "FLEET_API_KEY"},
        "mount_root": "/mnt/sfs/jobs/chris-q38-skyrl-probe-v13",
        "output_pvc": "sfs-shared",
        "output_registry_mount": "/mnt/cyber-output-registry",
        "output_registry_subpath": "models/fleetjob-dev",
        "model_artifact": {
            "path": "fleetjob-dev/qwen38-27b-1d4bf0f2-skyrl-v4",
            "mount_path": "base",
            "read_only": True,
            "required": True,
        },
        "preflight_model_pvc_subpath": ("models/fleetjob-dev/qwen38-27b-1d4bf0f2-skyrl-v4"),
        "ray_version": "2.49.2",
        "priority": "c1",
        "queue_priority": "q1",
        # The scientific canary is one eight-GPU Pod. Keep that exact initial
        # GPU shape here. FleetJob rejects CPU-only worker groups and requires
        # one GPU group, so use its documented GPU-head pattern: an elastic GPU
        # group starts at zero and has no demand during this zero-work probe.
        "workers": 0,
        "max_workers": 1,
        "gpus_per_worker": 1,
        "gpus_on_head": 8,
        "head_resources": {
            "cpu_request": "4",
            "cpu_limit": "8",
            "memory_request": "16Gi",
            "memory_limit": "32Gi",
        },
        "worker_resources": {
            "cpu_request": "2",
            "cpu_limit": "4",
            "memory_request": "4Gi",
            "memory_limit": "8Gi",
        },
        "resources": {
            "cpu_request": "64",
            "cpu_limit": "64",
            "memory_request": "512Gi",
            "memory_limit": "768Gi",
        },
    }


def _config(path: Path) -> dict:
    from .sft import read_mapping

    if path.resolve() != CONFIG_PATH.resolve() or path.is_symlink():
        raise ValueError("unknown SkyRL topology probe configuration")
    value = read_mapping(path)
    _validate_seal(value, CONFIG_SCHEMA)
    expected = _expected_execution()
    if (
        value["execution"] != expected
        # FleetJob artifact paths are SFS-relative below ``models/``.  The
        # zero-GPU PVC mount and GPU FleetJob must therefore name the same
        # source bytes, not merely expose them at the same destination.
        or "models/" + expected["model_artifact"]["path"] != expected["preflight_model_pvc_subpath"]
        or value["engine"] != {"num_engines": 2, "tensor_parallel_size": 4, "context_tokens": 98304}
        or value["deadlines"]
        != {"setup_seconds": 1200, "cleanup_seconds": 300, "total_seconds": 1500}
        or value["deadlines"]["setup_seconds"] + value["deadlines"]["cleanup_seconds"]
        != value["deadlines"]["total_seconds"]
        or value["submission_gate"].get("cpu_preflight_authorized") is not True
        or value["submission_gate"].get("preview_authorized") is not False
        or value["submission_gate"].get("fleetjob_preview_authorized") is not True
        or value["submission_gate"].get("submission_authorized") is not False
    ):
        raise ValueError("SkyRL topology probe contract changed")
    return value


def _bound_model(value: dict) -> dict:
    from .models import bound_model
    from .sft import read_mapping

    model = value["model"]
    return bound_model(
        read_mapping(CONFIG_PATH.parent / model["lock"]),
        read_mapping(CONFIG_PATH.parent / model["weights"]),
        model["root"],
    )


def compile_probe(path: Path) -> dict:
    value = _config(path)
    bound = _bound_model(value)
    arguments = skyrl.SkyRLConfig(
        name=value["name"],
        output_root=value["output_root"],
        model_root=bound["root"],
        train_data="/mnt/sfs/datasets/q38-probe/unused-train.jsonl",
        dev_data="/mnt/sfs/datasets/q38-probe/unused-dev.jsonl",
        data_manifest="/mnt/sfs/datasets/q38-probe/unused-manifest.json",
        train_rows=2,
        dev_rows=1,
        wandb_entity="disabled",
        wandb_project="disabled",
        wandb_run_id=value["name"],
        groups=1,
        samples_per_prompt=8,
        context_tokens=value["engine"]["context_tokens"],
        response_tokens=81920,
        tokens_per_turn=4096,
        max_turns=2,
    )
    plan = {
        "schema": SCHEMA,
        "run_name": value["name"],
        "output_root": value["output_root"],
        "model": bound,
        "arguments": arguments.__dict__,
        "native_overrides": skyrl.overrides(arguments),
        "engine": value["engine"],
        "deadlines": value["deadlines"],
        "execution": {"image": IMAGE, **value["execution"]},
        "qualification": {
            "profile": "qwen38_skyrl_topology_probe_dev_v1",
            "submission_gate": value["submission_gate"],
        },
        "runtime_sha256": digest(_runtime()),
        "scientific_work": {
            "task_rows": 0,
            "rollout_episodes": 0,
            "verifier_calls": 0,
            "optimizer_steps": 0,
            "checkpoints": 0,
        },
    }
    request(plan)
    return plan


def _validate(plan: dict) -> skyrl.SkyRLConfig:
    arguments = skyrl.SkyRLConfig(**plan["arguments"])
    overrides = skyrl.overrides(arguments)
    execution = plan["execution"]
    if (
        plan.get("schema") != SCHEMA
        or plan.get("runtime_sha256") != digest(_runtime())
        or plan.get("run_name") != arguments.name
        or plan.get("output_root") != arguments.output_root
        or plan.get("native_overrides") != overrides
        # Runtime bundles intentionally do not carry local config/model-lock
        # files.  This exact digest binds the complete repository, revision,
        # root and per-file size/SHA inventory compiled from those sealed
        # sources, without creating a second mutable runtime authority.
        or digest(plan.get("model")) != MODEL_BINDING_SHA256
        or plan.get("engine")
        != {"num_engines": 2, "tensor_parallel_size": 4, "context_tokens": 98304}
        or plan.get("deadlines")
        != {"setup_seconds": 1200, "cleanup_seconds": 300, "total_seconds": 1500}
        or execution != {"image": IMAGE, **_expected_execution()}
        or plan.get("output_root") != execution.get("mount_root", "") + "/models/run"
        # The FleetJob controller mounts the resolved artifact contents at the
        # requested mountPath itself; it does not add the Hugging Face revision
        # as another directory.  Exact revision authority still comes from the
        # sealed lock and every file is size+digest checked before setup.
        or arguments.model_root
        != execution.get("mount_root", "")
        + "/models/"
        + execution.get("model_artifact", {}).get("mount_path", "")
        or plan.get("qualification")
        != {
            "profile": "qwen38_skyrl_topology_probe_dev_v1",
            "submission_gate": {
                "cpu_preflight_authorized": True,
                "preview_authorized": False,
                "fleetjob_preview_authorized": True,
                "submission_authorized": False,
                "blockers": [
                    "cpu_preflight_not_yet_recorded",
                    "external_release_observer_not_yet_bound",
                ],
            },
        }
        or plan.get("scientific_work")
        != {
            "task_rows": 0,
            "rollout_episodes": 0,
            "verifier_calls": 0,
            "optimizer_steps": 0,
            "checkpoints": 0,
        }
    ):
        raise ValueError("SkyRL topology probe plan changed")
    return arguments


def request(
    plan: dict,
    *,
    fleetjob_transport: bool = False,
    cpu_preflight: bool = False,
    receipt_verify: bool = False,
) -> dict:
    if cpu_preflight and receipt_verify:
        raise ValueError("topology probe CPU modes are mutually exclusive")
    arguments = _validate(plan)
    execution = plan["execution"]
    files = _runtime()
    files.update(
        {
            "training/__init__.py": "",
            "evals/__init__.py": "",
            "evals/fleet/__init__.py": "",
            "cyber_post_train/__init__.py": "",
        }
    )
    files["plan.json"] = json.dumps(plan, sort_keys=True, separators=(",", ":"))
    options = (
        {"transport_split_threshold": 30000, "transport_chunk_size": 30000}
        if fleetjob_transport
        else {}
    )
    argv = ["--plan", "plan.json", "--sha256", digest(plan)]
    if fleetjob_transport:
        argv.extend(["--receipt", PREFLIGHT_RECEIPT])
    if cpu_preflight:
        argv.append("--cpu-preflight")
    if receipt_verify:
        argv.append("--verify-durable-receipt")
    return bundled_request(
        {
            "name": arguments.name,
            "title": arguments.name + " zero-update topology probe",
            "run_dir": arguments.output_root,
            "image": IMAGE,
            # bundled_request is only the sealed code transport here. Its
            # generic schema requires a positive GPU worker, but the FleetJob
            # manifest below is the sole resource authority and binds all
            # eight GPUs to the Ray head.
            "workers": 1,
            "gpus_per_worker": 8,
            "resources": execution["resources"],
            "priority_class": "c1",
            "requeueIfPreempted": False,
            "secrets": [],
            "env": {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONUNBUFFERED": "1",
                "CYBER_EXPECTED_RUNTIME_UID": "1000",
                "CYBER_EXPECTED_RUNTIME_GID": "100",
            },
        },
        files,
        MODULE,
        argv,
        **options,
    )


def _env(value: dict[str, str]) -> list[dict[str, str]]:
    return [{"name": key, "value": item} for key, item in sorted(value.items())]


def _container(
    plan: dict,
    environment: dict[str, str],
    resources: dict,
    *,
    gpus: int = 0,
) -> dict:
    requests = {
        "cpu": resources["cpu_request"],
        "memory": resources["memory_request"],
    }
    limits = {
        "cpu": resources["cpu_limit"],
        "memory": resources["memory_limit"],
    }
    if gpus:
        requests["nvidia.com/gpu"] = str(gpus)
        limits["nvidia.com/gpu"] = str(gpus)
    return {
        "name": "ray",
        "image": plan["execution"]["image"],
        "env": _env(environment),
        "securityContext": {
            "allowPrivilegeEscalation": False,
            "privileged": False,
            "runAsGroup": 100,
            "runAsNonRoot": True,
            "runAsUser": 1000,
        },
        "resources": {"requests": requests, "limits": limits},
    }


def _fleetjob(
    plan: dict,
    *,
    name: str,
    command: str,
    head: dict,
    workers: list[dict],
    active_deadline_seconds: int,
) -> dict:
    execution = plan["execution"]
    return {
        "apiVersion": "fleet.ai/v1alpha1",
        "kind": "FleetJob",
        "metadata": {"name": name, "namespace": execution["namespace"]},
        "spec": {
            "fleet": {
                "projectName": execution["project_name"],
                "auth": {"secretRef": execution["auth_secret"]},
                "mountRoot": execution["mount_root"],
                "models": [
                    {
                        "path": execution["model_artifact"]["path"],
                        "mountPath": execution["model_artifact"]["mount_path"],
                        "readOnly": execution["model_artifact"]["read_only"],
                        "required": execution["model_artifact"]["required"],
                    }
                ],
                "wandb": {"mode": "disabled"},
            },
            "kueue": {
                "queueName": "training-lq",
                "queuePriorityClass": execution["queue_priority"],
                "head": {
                    "queuePriorityClass": execution["queue_priority"],
                    "topology": {"mode": "unconstrained"},
                },
                "workerGroups": {
                    "gpu": {
                        "queuePriorityClass": execution["queue_priority"],
                        "topology": {"mode": "unconstrained"},
                    }
                },
            },
            "job": {
                "apiVersion": "ray.io/v1",
                "kind": "RayJob",
                "metadata": {"annotations": {"ray/kueue-admission-scope": "job"}},
                "spec": {
                    "entrypoint": command,
                    "activeDeadlineSeconds": active_deadline_seconds,
                    "backoffLimit": 0,
                    # Kueue-managed RayJobs must clean their clusters up after
                    # the entrypoint returns. The success/failure receipt is
                    # durable on SFS and is checked by a separate zero-GPU Job.
                    "shutdownAfterJobFinishes": True,
                    "rayClusterSpec": {
                        "rayVersion": execution["ray_version"],
                        "enableInTreeAutoscaling": True,
                        "headGroupSpec": head,
                        "workerGroupSpecs": workers,
                    },
                },
            },
        },
    }


def fleetjob_manifest(plan: dict) -> dict:
    """Render the current, dev-only FleetJob launch transport."""
    _validate(plan)
    execution = plan["execution"]
    bundled = request(plan, fleetjob_transport=True)
    head_env = {**bundled["env"], "RUN_DIR": plan["output_root"]}
    worker_env = {
        key: value for key, value in head_env.items() if not key.startswith("CYBER_RUNTIME_BUNDLE")
    }
    head = {
        "rayStartParams": {
            "num-cpus": "0",
            "num-gpus": str(execution["gpus_on_head"]),
        },
        "template": {
            "metadata": {},
            "spec": {
                "priorityClassName": execution["priority"],
                "containers": [
                    _container(
                        plan,
                        head_env,
                        execution["resources"],
                        gpus=execution["gpus_on_head"],
                    )
                ],
            },
        },
    }
    head_container = head["template"]["spec"]["containers"][0]
    head_container.update(
        {
            "terminationMessagePath": PREFLIGHT_RECEIPT,
            "terminationMessagePolicy": "File",
        }
    )
    worker = {
        "groupName": "gpu",
        "replicas": execution["workers"],
        "minReplicas": execution["workers"],
        "maxReplicas": execution["max_workers"],
        "rayStartParams": {"num-gpus": str(execution["gpus_per_worker"])},
        "template": {
            "metadata": {},
            "spec": {
                "priorityClassName": execution["priority"],
                "containers": [
                    _container(
                        plan,
                        worker_env,
                        execution["worker_resources"],
                        gpus=execution["gpus_per_worker"],
                    )
                ],
            },
        },
    }
    return _fleetjob(
        plan,
        name=plan["run_name"],
        command=bundled["command"],
        head=head,
        workers=[worker],
        active_deadline_seconds=1800,
    )


def preflight_job_manifest(plan: dict) -> dict:
    """Render the zero-GPU, exact-image/model Kubernetes preflight Job."""
    _validate(plan)
    execution = plan["execution"]
    gate = plan["qualification"]["submission_gate"]
    if gate["cpu_preflight_authorized"] is not True:
        raise JobsError("topology probe CPU preflight is not authorized")
    bundled = request(plan, fleetjob_transport=True, cpu_preflight=True)
    head_env = {
        **bundled["env"],
        "RUN_DIR": plan["output_root"],
        "CYBER_CREATE_ONCE_ROOT": execution["output_registry_mount"],
    }
    container = _container(plan, head_env, execution["head_resources"])
    container.update(
        {
            "command": ["/bin/sh", "-lc", "exec " + bundled["command"]],
            "terminationMessagePath": PREFLIGHT_RECEIPT,
            "terminationMessagePolicy": "File",
            "volumeMounts": [
                {
                    "name": "model",
                    "mountPath": plan["model"]["root"],
                    "readOnly": True,
                    "subPath": execution["preflight_model_pvc_subpath"],
                },
                {
                    "name": "output",
                    "mountPath": plan["output_root"],
                },
                {
                    "name": "output-registry",
                    "mountPath": execution["output_registry_mount"],
                    "readOnly": True,
                    "subPath": execution["output_registry_subpath"],
                },
            ],
        }
    )
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": PREFLIGHT_NAME, "namespace": execution["namespace"]},
        "spec": {
            "activeDeadlineSeconds": 1200,
            "backoffLimit": 0,
            "template": {
                "metadata": {},
                "spec": {
                    "automountServiceAccountToken": False,
                    "containers": [container],
                    "priorityClassName": execution["priority"],
                    "restartPolicy": "Never",
                    "volumes": [
                        {
                            "name": "model",
                            "persistentVolumeClaim": {"claimName": execution["output_pvc"]},
                        },
                        {
                            "name": "output-registry",
                            "persistentVolumeClaim": {
                                "claimName": execution["output_pvc"],
                                "readOnly": True,
                            },
                        },
                        {"name": "output", "emptyDir": {}},
                    ],
                },
            },
        },
    }


def receipt_verify_job_manifest(plan: dict) -> dict:
    """Render the zero-GPU, read-only durable-receipt verification Job."""
    _validate(plan)
    execution = plan["execution"]
    bundled = request(plan, fleetjob_transport=True, receipt_verify=True)
    environment = {**bundled["env"], "RUN_DIR": plan["output_root"]}
    container = _container(plan, environment, execution["head_resources"])
    container.update(
        {
            "command": ["/bin/sh", "-lc", "exec " + bundled["command"]],
            "terminationMessagePath": PREFLIGHT_RECEIPT,
            "terminationMessagePolicy": "File",
            "volumeMounts": [
                {
                    "name": "output",
                    "mountPath": plan["output_root"],
                    "readOnly": True,
                    "subPath": (execution["output_registry_subpath"] + "/" + plan["run_name"]),
                }
            ],
        }
    )
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": RECEIPT_VERIFY_NAME,
            "namespace": execution["namespace"],
        },
        "spec": {
            "activeDeadlineSeconds": 600,
            "backoffLimit": 0,
            "template": {
                "metadata": {},
                "spec": {
                    "automountServiceAccountToken": False,
                    "containers": [container],
                    "priorityClassName": execution["priority"],
                    "restartPolicy": "Never",
                    "volumes": [
                        {
                            "name": "output",
                            "persistentVolumeClaim": {
                                "claimName": execution["output_pvc"],
                                "readOnly": True,
                            },
                        }
                    ],
                },
            },
        },
    }


def fleetjob_packet(plan: dict) -> dict:
    manifest = fleetjob_manifest(plan)
    execution = plan["execution"]
    return _seal(
        {
            "schema": FLEETJOB_PACKET_SCHEMA,
            "plan_sha256": digest(plan),
            "manifest_sha256": digest(manifest),
            "kubernetes_context": execution["kubernetes_context"],
            "namespace": execution["namespace"],
            "name": plan["run_name"],
            "submitted": False,
        }
    )


def preflight_job_packet(plan: dict) -> dict:
    manifest = preflight_job_manifest(plan)
    execution = plan["execution"]
    return _seal(
        {
            "schema": PREFLIGHT_PACKET_SCHEMA,
            "plan_sha256": digest(plan),
            "manifest_sha256": digest(manifest),
            "kubernetes_context": execution["kubernetes_context"],
            "namespace": execution["namespace"],
            "name": PREFLIGHT_NAME,
            "submitted": False,
        }
    )


def receipt_verify_job_packet(plan: dict) -> dict:
    manifest = receipt_verify_job_manifest(plan)
    execution = plan["execution"]
    return _seal(
        {
            "schema": RECEIPT_VERIFY_PACKET_SCHEMA,
            "plan_sha256": digest(plan),
            "manifest_sha256": digest(manifest),
            "kubernetes_context": execution["kubernetes_context"],
            "namespace": execution["namespace"],
            "name": RECEIPT_VERIFY_NAME,
            "submitted": False,
        }
    )


def _validate_server_render(expected: dict, rendered: dict) -> None:
    expected = copy.deepcopy(expected)
    actual = copy.deepcopy(rendered)
    metadata = actual.get("metadata", {})
    finalizers = metadata.pop("finalizers", [])
    creation_timestamp = metadata.pop("creationTimestamp", None)
    generation = metadata.pop("generation", None)
    uid = metadata.pop("uid", None)
    if (
        finalizers != ["fleet.ai/fleetjob-cleanup"]
        or not isinstance(creation_timestamp, str)
        or not creation_timestamp
        or generation != 1
        or not isinstance(uid, str)
        or not uid
        or actual != expected
    ):
        raise JobsError("FleetJob server dry-run changed the topology probe")


def _validate_preflight_job_render(expected: dict, rendered: dict) -> None:
    """Allow only deterministic Kubernetes Job defaults in server dry-run."""
    expected = copy.deepcopy(expected)
    actual = copy.deepcopy(rendered)
    status = actual.pop("status", None)
    metadata = actual.get("metadata", {})
    timestamp = metadata.pop("creationTimestamp", None)
    generation = metadata.pop("generation", None)
    uid = metadata.pop("uid", None)
    labels = metadata.pop("labels", None)
    name = expected["metadata"]["name"]
    generated_labels = {
        "batch.kubernetes.io/controller-uid": uid,
        "batch.kubernetes.io/job-name": name,
        "controller-uid": uid,
        "job-name": name,
    }
    spec = actual.get("spec", {})
    defaults = {
        "completionMode": spec.pop("completionMode", None),
        "completions": spec.pop("completions", None),
        "manualSelector": spec.pop("manualSelector", None),
        "parallelism": spec.pop("parallelism", None),
        "podReplacementPolicy": spec.pop("podReplacementPolicy", None),
        "selector": spec.pop("selector", None),
        "suspend": spec.pop("suspend", None),
    }
    template = spec.get("template", {})
    template_labels = template.get("metadata", {}).pop("labels", None)
    pod = template.get("spec", {})
    pod_defaults = {
        "dnsPolicy": pod.pop("dnsPolicy", None),
        "schedulerName": pod.pop("schedulerName", None),
        "securityContext": pod.pop("securityContext", None),
        "terminationGracePeriodSeconds": pod.pop("terminationGracePeriodSeconds", None),
    }
    containers = pod.get("containers", [])
    image_pull_policy = containers[0].pop("imagePullPolicy", None) if len(containers) == 1 else None
    if (
        status != {}
        or not isinstance(timestamp, str)
        or not timestamp
        or generation != 1
        or not isinstance(uid, str)
        or not uid
        or labels != generated_labels
        or template_labels != generated_labels
        or defaults
        != {
            "completionMode": "NonIndexed",
            "completions": 1,
            "manualSelector": False,
            "parallelism": 1,
            "podReplacementPolicy": "TerminatingOrFailed",
            "selector": {"matchLabels": {"batch.kubernetes.io/controller-uid": uid}},
            "suspend": False,
        }
        or pod_defaults
        != {
            "dnsPolicy": "ClusterFirst",
            "schedulerName": "default-scheduler",
            "securityContext": {},
            "terminationGracePeriodSeconds": 30,
        }
        or image_pull_policy != "IfNotPresent"
        or actual != expected
    ):
        raise JobsError("CPU preflight server dry-run changed the exact Job")


def validate_fleetjob_preview(plan: dict, manifest: dict, rendered: dict) -> dict:
    """Validate the server-dry-run object without retaining its private env."""
    if manifest != fleetjob_manifest(plan):
        raise JobsError("topology probe FleetJob differs from its immutable plan")
    _validate_server_render(manifest, rendered)
    execution = plan["execution"]
    return _seal(
        {
            "schema": FLEETJOB_PREVIEW_SCHEMA,
            "status": "passed",
            "plan_sha256": digest(plan),
            "manifest_sha256": digest(manifest),
            "server_render_sha256": digest(rendered),
            "kubernetes_context": execution["kubernetes_context"],
            "namespace": execution["namespace"],
            "name": plan["run_name"],
            "gpu_nodes": 1,
            "gpus": 8,
            "runtime_user": {"uid": 1000, "gid": 100},
            "submitted": False,
        }
    )


def validate_preflight_job_preview(plan: dict, manifest: dict, rendered: dict) -> dict:
    if manifest != preflight_job_manifest(plan):
        raise JobsError("topology probe preflight differs from its immutable plan")
    _validate_preflight_job_render(manifest, rendered)
    execution = plan["execution"]
    return _seal(
        {
            "schema": PREFLIGHT_PREVIEW_SCHEMA,
            "status": "passed",
            "plan_sha256": digest(plan),
            "manifest_sha256": digest(manifest),
            "server_render_sha256": digest(rendered),
            "kubernetes_context": execution["kubernetes_context"],
            "namespace": execution["namespace"],
            "name": PREFLIGHT_NAME,
            "gpu_nodes": 0,
            "gpus": 0,
            "runtime_user": {"uid": 1000, "gid": 100},
            "submitted": False,
        }
    )


def validate_receipt_verify_job_preview(plan: dict, manifest: dict, rendered: dict) -> dict:
    if manifest != receipt_verify_job_manifest(plan):
        raise JobsError("topology probe receipt verifier differs from its plan")
    _validate_preflight_job_render(manifest, rendered)
    execution = plan["execution"]
    return _seal(
        {
            "schema": RECEIPT_VERIFY_PREVIEW_SCHEMA,
            "status": "passed",
            "plan_sha256": digest(plan),
            "manifest_sha256": digest(manifest),
            "server_render_sha256": digest(rendered),
            "kubernetes_context": execution["kubernetes_context"],
            "namespace": execution["namespace"],
            "name": RECEIPT_VERIFY_NAME,
            "gpu_nodes": 0,
            "gpus": 0,
            "runtime_user": {"uid": 1000, "gid": 100},
            "submitted": False,
        }
    )


def preflight(plan: dict, progress: Callable[[str], None] | None = None) -> dict:
    mark = progress or (lambda _: None)
    mark("runtime_identity")
    if (os.geteuid(), os.getegid()) != (1000, 100):
        raise ValueError("probe preflight requires image user 1000:100")
    mark("runtime_imports")
    import torch
    from skyrl.backends.skyrl_train.inference_servers.utils import build_vllm_cli_args

    mark("zero_gpu")
    if torch.cuda.is_available():
        raise ValueError("probe preflight is CPU-only")
    mark("plan_validation")
    arguments = _validate(plan)
    mark("sealed_bootstrap_destination")
    _validate_destination(plan)
    mark("create_once_destination_absence")
    _validate_create_once_absence(plan)
    mark("model_inventory")
    _verify_model(plan)
    mark("native_engine_arguments")
    cfg = skyrl.diagnostic_native_config(arguments)
    build_vllm_cli_args(cfg)
    mark("receipt")
    req = request(plan, fleetjob_transport=True, cpu_preflight=True)
    return {
        "schema": "cyber_skyrl_topology_probe_cpu_preflight_v1",
        "status": "passed",
        "gpus": 0,
        "runtime_user": {"uid": 1000, "gid": 100},
        "plan_sha256": digest(plan),
        "request_sha256": digest(req),
        "fleetjob_manifest_sha256": digest(fleetjob_manifest(plan)),
        "preflight_job_manifest_sha256": digest(preflight_job_manifest(plan)),
        "task_rows_read": 0,
        "rollout_episodes": 0,
        "optimizer_steps": 0,
        "create_once_output_absent": True,
    }


def validate_preview(plan: dict, req: dict, preview: dict) -> dict:
    """Validate the legacy Jobs-API rendering with normal cluster cleanup."""
    import yaml

    from .skyrl_training import validate_gpu_runtime_preview

    if request(plan) != req:
        raise JobsError("topology probe request differs from its immutable plan")
    try:
        rendered = yaml.safe_load(preview["manifest_yaml"])
        if rendered["spec"]["shutdownAfterJobFinishes"] is not True:
            raise JobsError("topology probe preview must release its Ray cluster")
    except (KeyError, TypeError, yaml.YAMLError) as exc:
        raise JobsError("malformed topology probe Jobs API preview") from exc
    result = validate_gpu_runtime_preview(req, preview)
    return {
        **result,
        "manifest_sha256": digest(rendered),
        "shutdown_after_job_finishes": True,
        "cleanup_authority": "kuberay_plus_uid_bound_observer",
        "receipt_authority": "durable_sfs_receipt_verifier",
    }


class _Deadline:
    def __init__(self, seconds: int, phase: str):
        self.seconds, self.phase = seconds, phase

    def __enter__(self):
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("topology-probe deadline requires the main thread")
        if signal.getitimer(signal.ITIMER_REAL) != (0.0, 0.0):
            raise RuntimeError("topology-probe deadline cannot replace an existing timer")
        self.previous = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, self._expired)
        signal.setitimer(signal.ITIMER_REAL, self.seconds)
        return self

    def _expired(self, *_):
        raise TimeoutError(f"{self.phase} exceeded its plan-bound deadline")

    def __exit__(self, *_):
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, self.previous)


def _verify_model(plan: dict) -> None:
    """Prove that the FleetJob model mount matches the exact model lock."""
    root = Path(plan["model"]["root"])
    for index, item in enumerate(plan["model"]["files"]):
        path = root / item["path"]
        is_link = path.is_symlink()
        try:
            metadata = path.stat()
        except FileNotFoundError as exc:
            category = "broken_symlink" if is_link else "missing"
            raise ProbeGateError(f"model_file_{category}_{index:02d}") from exc
        except PermissionError as exc:
            raise ProbeGateError(f"model_file_inaccessible_{index:02d}") from exc
        if not path.is_file():
            raise ProbeGateError(f"model_file_not_regular_{index:02d}")
        if "size" in item and metadata.st_size != item["size"]:
            raise ProbeGateError(f"model_file_size_mismatch_{index:02d}")
        try:
            with path.open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
        except PermissionError as exc:
            raise ProbeGateError(f"model_file_inaccessible_{index:02d}") from exc
        if actual != item["sha256"].removeprefix("sha256:"):
            raise ProbeGateError(f"model_file_digest_mismatch_{index:02d}")


def _validate_destination(plan: dict) -> Path:
    """Allow only the digest-checked bootstrap that necessarily precedes us."""
    root = Path(plan["output_root"])
    if not root.is_dir() or not os.access(root, os.W_OK | os.X_OK):
        raise PermissionError("probe output is not a writable directory")
    runtime = root / ".runtime"
    if runtime.is_symlink() or not runtime.is_dir() or set(root.iterdir()) != {runtime}:
        raise FileExistsError("probe output contains more than its sealed bootstrap")
    expected = {
        **_runtime(),
        "training/__init__.py": "",
        "evals/__init__.py": "",
        "evals/fleet/__init__.py": "",
        "cyber_post_train/__init__.py": "",
        "plan.json": json.dumps(plan, sort_keys=True, separators=(",", ":")),
    }
    files = {str(path.relative_to(runtime)): path for path in runtime.rglob("*") if path.is_file()}
    if set(files) != set(expected) or any(
        path.is_symlink() or path.read_text() != expected[name] for name, path in files.items()
    ):
        raise ValueError("probe bootstrap differs from its digest-bound runtime")
    return root


def _validate_create_once_absence(plan: dict) -> Path:
    """Prove the controller-owned destination does not already exist on SFS."""
    execution = plan["execution"]
    registry_root = os.environ.get("CYBER_CREATE_ONCE_ROOT")
    if registry_root != execution["output_registry_mount"]:
        raise ValueError("create-once registry mount binding mismatch")
    target = (
        Path(registry_root)
        / plan["run_name"]
        / Path(plan["output_root"]).relative_to(execution["mount_root"])
    )
    if target.exists() or target.is_symlink():
        raise FileExistsError("create-once topology-probe output already exists")
    return target


def _write_receipt(path: Path, receipt: dict, *, exclusive: bool) -> None:
    flags = os.O_WRONLY | os.O_CREAT | (os.O_EXCL if exclusive else os.O_TRUNC)
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, "w") as stream:
        stream.write(json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n")


def _persist_probe_failure(plan: dict, failure: dict) -> None:
    """Retain one sanitized receipt when KubeRay removes a failed pod quickly."""
    _validate_seal(failure, PROBE_FAILURE_SCHEMA)
    if failure.get("plan_sha256") != digest(plan):
        raise ValueError("probe failure receipt is not bound to its plan")
    root = Path(plan["output_root"])
    if root.is_symlink() or not root.is_dir() or not os.access(root, os.W_OK | os.X_OK):
        raise ValueError("probe failure destination is unavailable")
    _write_receipt(root / FAILURE_RECEIPT, failure, exclusive=True)


def _stop_setup(setup, ray) -> None:
    if setup is None:
        return
    errors = []
    try:
        setup.router.shutdown()
    except Exception as exc:
        errors.append(exc)
    actors, placement_groups = [], []
    for group in reversed(tuple(setup.server_groups)):
        try:
            actors.extend(tuple(group.get_actors()))
        except Exception as exc:
            errors.append(exc)
        for name in ("_internal_pg", "_external_pg"):
            value = getattr(group, name, None)
            if value is not None:
                placement_groups.append(getattr(value, "pg", value))
    refs = []
    for actor in actors:
        with suppress(Exception):
            refs.append(actor.shutdown.remote())
    if refs:
        with suppress(Exception):
            ray.get(refs, timeout=30)
    for actor in actors:
        with suppress(Exception):
            ray.kill(actor, no_restart=True)
    if placement_groups:
        from ray.util.placement_group import remove_placement_group

        for group in placement_groups:
            with suppress(Exception):
                remove_placement_group(group)
    if errors:
        raise ExceptionGroup("one or more topology-probe resources failed to stop", errors)


def run(plan: dict) -> dict:
    """Start both TP4 engines and release them; no scientific path is imported."""
    arguments = _validate(plan)
    if os.environ.get("RUN_DIR") != plan["output_root"]:
        raise ValueError("Jobs API output binding mismatch")
    if (os.geteuid(), os.getegid()) != (1000, 100):
        raise ValueError("probe GPU runtime requires image user 1000:100")
    root = _validate_destination(plan)
    forbidden = ("episodes", "checkpoints", "exports")
    if any((root / name).exists() for name in forbidden):
        raise FileExistsError("probe output contains a scientific artifact")

    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
    from skyrl.backends.skyrl_train.inference_servers.setup import create_inference_servers
    from skyrl.backends.skyrl_train.inference_servers.utils import build_vllm_cli_args

    cfg = skyrl.diagnostic_native_config(arguments)
    setup = None
    started = time.time()
    setup_error = None
    gpu_node_runtime_users = None
    try:
        with _Deadline(plan["deadlines"]["setup_seconds"], "engine setup"):
            _verify_model(plan)
            ray.init(address="auto", log_to_driver=False)
            nodes = [
                row
                for row in ray.nodes()
                if row.get("Alive") and row.get("Resources", {}).get("GPU")
            ]
            if len(nodes) != 1 or quantity(nodes[0]["Resources"]["GPU"]) != 8:
                raise ValueError("probe requires exactly one eight-GPU Ray pod")

            @ray.remote(num_cpus=0)
            def runtime_user():
                import os

                return {
                    "uid": os.geteuid(),
                    "gid": os.getegid(),
                    "physical_node": os.environ.get("FLEET_NODE_NAME", ""),
                }

            gpu_node_runtime_users = ray.get(
                [
                    runtime_user.options(
                        scheduling_strategy=NodeAffinitySchedulingStrategy(
                            row["NodeID"], soft=False
                        )
                    ).remote()
                    for row in sorted(nodes, key=lambda item: item["NodeID"])
                ],
                timeout=30,
            )
            if any(
                row.get("uid") != 1000 or row.get("gid") != 100 or not row.get("physical_node")
                for row in gpu_node_runtime_users
            ):
                raise ValueError("probe GPU pods must use image user 1000:100")
            physical_nodes = {row["physical_node"] for row in gpu_node_runtime_users}
            if physical_nodes != {os.environ.get("FLEET_NODE_NAME")}:
                raise ValueError("probe GPU pods are not on one physical node")
            setup = create_inference_servers(
                cfg.generator.inference_engine,
                build_vllm_cli_args(cfg),
                log_path=cfg.trainer.log_path,
            )
            if len(tuple(setup.server_groups)) != 2 or len(tuple(setup.server_urls)) != 2:
                raise ValueError("probe did not start both TP4 engine groups")
    except BaseException as exc:
        setup_error = exc
    try:
        with _Deadline(plan["deadlines"]["cleanup_seconds"], "engine cleanup"):
            _stop_setup(setup, ray)
            ray.shutdown()
    finally:
        if ray.is_initialized():
            ray.shutdown()
    if setup_error is not None:
        raise setup_error
    if any((root / name).exists() for name in forbidden):
        raise ValueError("probe created a scientific artifact")
    return _seal(
        {
            "schema": RECEIPT_SCHEMA,
            "status": "setup_and_internal_cleanup_passed",
            "plan_sha256": digest(plan),
            "elapsed_seconds": time.time() - started,
            **plan["scientific_work"],
            "engines_started": 2,
            "tensor_parallel_size": 4,
            "runtime_users": {
                "driver": {
                    "uid": 1000,
                    "gid": 100,
                    "physical_node": os.environ["FLEET_NODE_NAME"],
                },
                "gpu_pods": gpu_node_runtime_users,
            },
            "ray_shutdown_called": True,
            "external_release_required": True,
        }
    )


def _validate_probe_receipt(plan: dict, receipt: dict) -> None:
    """Validate the durable zero-science GPU receipt against its exact plan."""
    _validate(plan)
    _validate_seal(receipt, RECEIPT_SCHEMA)
    if (
        receipt.get("status") != "setup_and_internal_cleanup_passed"
        or receipt.get("plan_sha256") != digest(plan)
        or not isinstance(receipt.get("runtime_users"), dict)
        or receipt["runtime_users"].get("driver", {}).get("uid") != 1000
        or receipt["runtime_users"].get("driver", {}).get("gid") != 100
        or not receipt["runtime_users"].get("driver", {}).get("physical_node")
        or not isinstance(receipt["runtime_users"].get("gpu_pods"), list)
        or len(receipt["runtime_users"]["gpu_pods"]) != 1
        or any(
            row.get("uid") != 1000
            or row.get("gid") != 100
            or row.get("physical_node") != receipt["runtime_users"]["driver"]["physical_node"]
            for row in receipt["runtime_users"]["gpu_pods"]
        )
        or any(receipt.get(key) != 0 for key in plan["scientific_work"])
        or receipt.get("engines_started") != 2
        or receipt.get("tensor_parallel_size") != 4
        or receipt.get("ray_shutdown_called") is not True
        or receipt.get("external_release_required") is not True
    ):
        raise ValueError("probe receipt is not the exact accepted internal cleanup")


def verify_durable_receipt(plan: dict) -> dict:
    """Read the post-cleanup success receipt from a read-only SFS mount."""
    if (os.geteuid(), os.getegid()) != (1000, 100):
        raise ValueError("probe receipt verification requires image user 1000:100")
    import torch

    if torch.cuda.is_available():
        raise ValueError("probe receipt verification is CPU-only")
    root = Path(plan["output_root"])
    if root.is_symlink() or not root.is_dir():
        raise ProbeGateError("durable_output_absent")
    success = root / "TOPOLOGY_PROBE.json"
    failure = root / FAILURE_RECEIPT
    if failure.exists() or failure.is_symlink():
        raise ProbeGateError("durable_failure_receipt_present")
    try:
        receipt = json.loads(success.read_bytes())
    except FileNotFoundError as exc:
        raise ProbeGateError("durable_success_receipt_absent") from exc
    except (OSError, ValueError) as exc:
        raise ProbeGateError("durable_success_receipt_unreadable") from exc
    if not isinstance(receipt, dict):
        raise ProbeGateError("durable_success_receipt_invalid")
    try:
        _validate_probe_receipt(plan, receipt)
    except ValueError as exc:
        raise ProbeGateError("durable_success_receipt_invalid") from exc
    return _seal(
        {
            "schema": RECEIPT_VERIFY_SCHEMA,
            "status": "passed",
            "gpus": 0,
            "runtime_user": {"uid": 1000, "gid": 100},
            "plan_sha256": digest(plan),
            "receipt_sha256": receipt["sha256"],
            **plan["scientific_work"],
        }
    )


def validate_release(plan: dict, receipt: dict, observation: dict) -> dict:
    """Seal a later read-only observation that the exact allocation disappeared."""
    _validate_probe_receipt(plan, receipt)
    required = {
        "kubernetes_context",
        "namespace",
        "fleetjob_name",
        "job_id",
        "fleetjob_uid",
        "rayjob_uid",
        "workload_uid",
        "raycluster_uid",
        "pod_uids",
        "terminal_status",
        "fleetjob_present",
        "rayjob_present",
        "workload_present",
        "raycluster_present",
        "pods_present",
        "active_gpus",
        "created_at",
        "deletion_requested_at",
        "release_observed_at",
    }
    presence = {
        "fleetjob_present",
        "rayjob_present",
        "workload_present",
        "raycluster_present",
        "pods_present",
    }
    identities = {
        "job_id",
        "fleetjob_uid",
        "rayjob_uid",
        "workload_uid",
        "raycluster_uid",
    }
    try:
        timestamps = [
            datetime.strptime(observation[key], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
            for key in ("created_at", "deletion_requested_at", "release_observed_at")
        ]
        uuid_values = [UUID(observation[key]) for key in identities]
        uuid_values.extend(UUID(value) for value in observation["pod_uids"])
    except (AttributeError, KeyError, TypeError, ValueError):
        timestamps, uuid_values = [], []
    if (
        set(observation) != required
        or observation.get("kubernetes_context") != plan["execution"]["kubernetes_context"]
        or observation.get("namespace") != plan["execution"]["namespace"]
        or observation.get("fleetjob_name") != plan["run_name"]
        or any(
            not isinstance(observation[key], str) or not observation[key]
            for key in required - presence - {"active_gpus", "pod_uids"}
        )
        or len(uuid_values) != 6
        or not isinstance(observation["pod_uids"], list)
        or len(observation["pod_uids"]) != 1
        or len(set(observation["pod_uids"])) != 1
        or any(not isinstance(value, str) or not value for value in observation["pod_uids"])
        or observation["terminal_status"] not in {"Succeeded", "Failed", "Deleted"}
        or any(observation[key] is not False for key in presence)
        or type(observation["active_gpus"]) is not int
        or observation["active_gpus"] != 0
        or len(timestamps) != 3
        or not timestamps[0] <= timestamps[1] <= timestamps[2]
        or (timestamps[1] - timestamps[0]).total_seconds() > 1800
    ):
        raise ValueError("exact topology-probe release was not proven")
    return _seal(
        {
            "schema": RELEASE_SCHEMA,
            "status": "released",
            "plan_sha256": digest(plan),
            "probe_receipt_sha256": receipt["sha256"],
            "observation": observation,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--cpu-preflight", action="store_true")
    parser.add_argument("--verify-durable-receipt", action="store_true")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    phase = "plan_loading"
    plan = None

    def mark(value: str) -> None:
        nonlocal phase
        phase = value

    try:
        plan = json.loads(args.plan.read_bytes())
        phase = "plan_digest"
        if digest(plan) != args.sha256:
            raise ValueError("plan digest mismatch")
        phase = "request_validation"
        request(plan)
        if args.cpu_preflight and args.verify_durable_receipt:
            raise ValueError("topology probe CPU modes are mutually exclusive")
        if args.cpu_preflight:
            if args.receipt != Path(PREFLIGHT_RECEIPT):
                raise ValueError("CPU preflight receipt binding mismatch")
            receipt = _seal(preflight(plan, mark))
            _write_receipt(Path(PREFLIGHT_RECEIPT), receipt, exclusive=False)
            print(json.dumps({"status": receipt["status"], "sha256": receipt["sha256"]}))
            return
        if args.verify_durable_receipt:
            if args.receipt != Path(PREFLIGHT_RECEIPT):
                raise ValueError("durable receipt verification binding mismatch")
            phase = "durable_receipt_verification"
            receipt = verify_durable_receipt(plan)
            _write_receipt(Path(PREFLIGHT_RECEIPT), receipt, exclusive=False)
            print(json.dumps({"status": receipt["status"], "sha256": receipt["sha256"]}))
            return
        if args.receipt != Path(PREFLIGHT_RECEIPT):
            raise ValueError("GPU probe receipt binding mismatch")
        phase = "gpu_topology_probe"
        receipt = run(plan)
        path = Path(plan["output_root"]) / "TOPOLOGY_PROBE.json"
        _write_receipt(path, receipt, exclusive=True)
        _write_receipt(Path(PREFLIGHT_RECEIPT), receipt, exclusive=False)
        print(json.dumps({"status": receipt["status"], "sha256": receipt["sha256"]}))
    except BaseException as exc:
        failure = _seal(
            {
                "schema": (
                    PREFLIGHT_FAILURE_SCHEMA
                    if args.cpu_preflight
                    else RECEIPT_VERIFY_FAILURE_SCHEMA
                    if args.verify_durable_receipt
                    else PROBE_FAILURE_SCHEMA
                ),
                "status": (
                    "rejected" if args.cpu_preflight or args.verify_durable_receipt else "failed"
                ),
                "phase": phase,
                "error_class": type(exc).__name__,
                "error_code": getattr(exc, "code", "unclassified"),
                "plan_sha256": args.sha256,
                "task_rows_read": 0,
                "rollout_episodes": 0,
                "optimizer_steps": 0,
                "checkpoints": 0,
            }
        )
        if args.receipt == Path(PREFLIGHT_RECEIPT):
            with suppress(Exception):
                _write_receipt(Path(PREFLIGHT_RECEIPT), failure, exclusive=False)
        if not args.cpu_preflight and phase == "gpu_topology_probe" and isinstance(plan, dict):
            with suppress(Exception):
                _persist_probe_failure(plan, failure)
        print(
            json.dumps(
                {
                    "status": failure["status"],
                    "phase": phase,
                    "error_class": type(exc).__name__,
                    "error_code": getattr(exc, "code", "unclassified"),
                    "sha256": failure["sha256"],
                }
            )
        )
        if args.cpu_preflight or args.verify_durable_receipt:
            return
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
