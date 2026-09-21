"""Fail-closed, create-once fallbacks when the Jobs API omits one annotation.

The Jobs API remains the rendering authority.  This module accepts its live
preview, changes only the run identity, removes the API-only Fleet credential
Secret that SFT does not consume, and adds the project-required root alert
annotation.  It never calls the Jobs API create endpoint and never applies or
patches a Kubernetes object.  The non-SFT exception is restricted to one exact
LR30 step-76 HF inference-forward qualification schema.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import uuid
from collections.abc import Callable
from copy import deepcopy
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import yaml

from .jobs import (
    FAILURE_ALERT_ANNOTATION,
    FAILURE_ALERT_OFF,
    JobsError,
    digest,
    validate_preview,
    validate_request,
)
from .lora_cpu_preflight import (
    CPU_PREFLIGHT_BINDING_ANNOTATIONS,
    CPU_PREFLIGHT_BUNDLE_SHA256_ANNOTATION,
    CPU_PREFLIGHT_ENV,
    CPU_SFS_MEMORY_FLOOR_MIB_ANNOTATION,
    CPU_SFS_OUTPUT_ROOT_ANNOTATION,
    CPU_SFS_OWNED_ROOT_ANNOTATION,
    CPU_SOURCE_ARCHIVE_SHA256_ANNOTATION,
    CPU_SOURCE_COMMIT_ANNOTATION,
    LoraCpuPreflightPackage,
    validate_lora_cpu_preflight_config_map_response,
    validate_lora_cpu_preflight_package,
)
from .lora_cpu_preflight_driver import (
    ENV_BUNDLE_SHA256,
    ENV_SFS_CONTROL_MOUNT,
    ENV_SFS_OUTPUT_ROOT,
    ENV_SFS_OWNED_ROOT,
    ENV_SOURCE_ARCHIVE,
    ENV_SOURCE_ARCHIVE_SHA256,
    ENV_SOURCE_COMMIT,
)
from .sfs_output import SFS_JOBS_ROOT, prove_output_absent
from .sfs_output_job import (
    build_sfs_output_job,
    collect_sfs_output_receipt,
    validate_completed_sfs_output_job,
    validate_sfs_output_job_node_fit,
    validate_sfs_output_job_package,
    validate_sfs_output_job_response,
)
from .sfs_write_identity import TRAINER_GID, TRAINER_UID, validate_owned_output_binding
from .sft_cpu_preflight_job import (
    build_sft_cpu_preflight_job,
    collect_sft_cpu_preflight_receipt,
    validate_completed_sft_cpu_preflight_job,
    validate_sft_cpu_preflight_job_node_fit,
    validate_sft_cpu_preflight_job_package,
    validate_sft_cpu_preflight_job_response,
)
from .source_bundle import canonical_source_commit_bytes

NAMESPACE = "fleet-train-jobs"
ZERO_RUN_ID = "00000000-0000-0000-0000-000000000000"
SFT_SCHEMAS = {"cyber_sft_runtime_v2", "cyber_sft_runtime_dense_v1"}
SFT_SECRET = "wandb-api"
DIRECT_JOURNAL = "DIRECT_SUBMISSION.jsonl"
CPU_CHECKPOINT_OPERATION_ANNOTATION = "cyber-post-train.fleet.ai/cpu-checkpoint-operation"
CPU_CHECKPOINT_OPERATIONS = {"seal", "verify"}
LORA_TRAINER_IMAGE = (
    "ghcr.io/fleet-ai/skyrl-fleet-v2/trainer@"
    "sha256:7da4adba80d032509dba69fb4dd23bedca17fde3e2b88643815f80d6ee6c5317"
)
CPU_NODE_SELECTOR = {
    "kubernetes.io/arch": "amd64",
    "workload": "fleetai-training-ng-cpu",
}
KUBERNETES_UID_PATTERN = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}"
)


def _cpu_millicores(value: object) -> int:
    if not isinstance(value, str) or not value:
        raise JobsError("CPU checkpoint Pod has an invalid CPU quantity")
    raw = value[:-1] if value.endswith("m") else value
    try:
        amount = Decimal(raw)
    except InvalidOperation:
        raise JobsError("CPU checkpoint Pod has an invalid CPU quantity") from None
    millicores = amount if value.endswith("m") else amount * 1000
    if millicores <= 0 or millicores != millicores.to_integral_value():
        raise JobsError("CPU checkpoint Pod has an invalid CPU quantity")
    return int(millicores)


def _memory_bytes(value: object) -> int:
    if not isinstance(value, str) or not value:
        raise JobsError("CPU checkpoint Pod has an invalid memory quantity")
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([EPTGMK]i?|)", value)
    if match is None:
        raise JobsError("CPU checkpoint Pod has an invalid memory quantity")
    suffix = match.group(2)
    power = {"": 0, "K": 1, "M": 2, "G": 3, "T": 4, "P": 5, "E": 6}
    binary = suffix.endswith("i")
    unit = suffix[:-1] if binary else suffix
    try:
        amount = Decimal(match.group(1)) * (1024 if binary else 1000) ** power[unit]
    except (InvalidOperation, KeyError):
        raise JobsError("CPU checkpoint Pod has an invalid memory quantity") from None
    if amount <= 0 or amount != amount.to_integral_value():
        raise JobsError("CPU checkpoint Pod has an invalid memory quantity")
    return int(amount)


def _cpu_checkpoint_request(manifest: dict) -> tuple[int, int]:
    spec = manifest["spec"]
    if spec.get("initContainers") not in (None, []):
        raise JobsError("CPU checkpoint Pod must not use init containers")
    if spec.get("overhead") not in (None, {}):
        raise JobsError("CPU checkpoint Pod must not add scheduling overhead")
    cpu = 0
    memory = 0
    for container in spec["containers"]:
        requests = container.get("resources", {}).get("requests", {})
        if set(requests) != {"cpu", "memory"}:
            raise JobsError("CPU checkpoint Pod must request exactly CPU and memory")
        cpu += _cpu_millicores(requests["cpu"])
        memory += _memory_bytes(requests["memory"])
    return cpu, memory


def validate_cpu_checkpoint_node_fit(manifest: dict, inventory: dict) -> dict:
    """Prove the requested CPU and memory fit one currently observed eligible node."""
    validate_cpu_checkpoint_pod(manifest)
    if inventory.get("kind") != "List" or not isinstance(inventory.get("items"), list):
        raise JobsError("CPU node inventory is incomplete")
    requested_cpu, requested_memory = _cpu_checkpoint_request(manifest)
    eligible = []
    for node in inventory["items"]:
        try:
            metadata = node["metadata"]
            spec = node["spec"]
            status = node["status"]
            labels = metadata["labels"]
            allocatable = status["allocatable"]
            conditions = status["conditions"]
        except (KeyError, TypeError):
            raise JobsError("CPU node inventory contains a malformed node") from None
        if not all(labels.get(key) == value for key, value in CPU_NODE_SELECTOR.items()):
            continue
        if spec.get("unschedulable") is True:
            continue
        if not any(
            condition.get("type") == "Ready" and condition.get("status") == "True"
            for condition in conditions
            if isinstance(condition, dict)
        ):
            continue
        eligible.append(
            {
                "name": metadata.get("name"),
                "cpu_millicores": _cpu_millicores(allocatable.get("cpu")),
                "memory_bytes": _memory_bytes(allocatable.get("memory")),
            }
        )
    if not eligible:
        raise JobsError("no Ready eligible CPU node is visible")
    fitting = [
        node
        for node in eligible
        if node["cpu_millicores"] >= requested_cpu and node["memory_bytes"] >= requested_memory
    ]
    if not fitting:
        raise JobsError("CPU checkpoint Pod cannot fit any observed eligible node")
    return {
        "requested_cpu_millicores": requested_cpu,
        "requested_memory_bytes": requested_memory,
        "eligible_node_count": len(eligible),
        "fitting_node_count": len(fitting),
    }


def _has_lora_cpu_preflight_surface(manifest: dict) -> bool:
    """Detect a partially stripped package instead of silently taking the generic path."""

    metadata = manifest.get("metadata")
    spec = manifest.get("spec")
    if not isinstance(metadata, dict) or not isinstance(spec, dict):
        return False
    annotations = metadata.get("annotations")
    if isinstance(annotations, dict) and any(
        name in annotations for name in CPU_PREFLIGHT_BINDING_ANNOTATIONS
    ):
        return True
    containers = spec.get("containers")
    if not isinstance(containers, list):
        return False
    for container in containers:
        if not isinstance(container, dict):
            continue
        if container.get("command") == ["python", "/bundle/preflight_driver.py"]:
            return True
        entries = container.get("env")
        if isinstance(entries, list) and any(
            isinstance(entry, dict) and entry.get("name") in CPU_PREFLIGHT_ENV for entry in entries
        ):
            return True
    return False


def validate_cpu_checkpoint_pod(manifest: dict) -> dict:
    """Reject GPU use and host-specific placement for a CPU checkpoint operation."""
    try:
        metadata = manifest["metadata"]
        spec = manifest["spec"]
        annotations = metadata["annotations"]
        containers = spec["containers"]
    except (KeyError, TypeError) as exc:
        raise JobsError("malformed CPU checkpoint Pod") from exc
    if manifest.get("apiVersion") != "v1" or manifest.get("kind") != "Pod":
        raise JobsError("CPU checkpoint operation must be one v1 Pod")
    if metadata.get("namespace") != NAMESPACE or not isinstance(metadata.get("name"), str):
        raise JobsError("CPU checkpoint Pod namespace/name drift")
    if annotations.get(FAILURE_ALERT_ANNOTATION) != FAILURE_ALERT_OFF:
        raise JobsError("CPU checkpoint Pod must opt out of failed-job alerts before create")
    if annotations.get(CPU_CHECKPOINT_OPERATION_ANNOTATION) not in CPU_CHECKPOINT_OPERATIONS:
        raise JobsError("CPU checkpoint Pod must name a supported seal/verify operation")
    if spec.get("priorityClassName") != "c1":
        raise JobsError("CPU checkpoint Pod must use c1 priority")
    if spec.get("nodeSelector") != CPU_NODE_SELECTOR:
        raise JobsError("CPU checkpoint Pod must select only the shared CPU pool and architecture")
    if spec.get("nodeName") is not None or spec.get("affinity") is not None:
        raise JobsError("CPU checkpoint Pod must not pin one host or add placement affinity")
    if spec.get("restartPolicy") != "Never":
        raise JobsError("CPU checkpoint Pod must use restartPolicy Never")
    if not isinstance(containers, list) or not containers:
        raise JobsError("CPU checkpoint Pod has no containers")
    for container in [*spec.get("initContainers", []), *containers]:
        if not isinstance(container, dict):
            raise JobsError("CPU checkpoint Pod contains a malformed container")
        resources = container.get("resources", {})
        if not isinstance(resources, dict):
            raise JobsError("CPU checkpoint Pod resources are malformed")
        for field in ("requests", "limits"):
            values = resources.get(field, {})
            if not isinstance(values, dict):
                raise JobsError("CPU checkpoint Pod resource quantities are malformed")
            if "nvidia.com/gpu" in values:
                raise JobsError("CPU checkpoint Pod must not request or limit GPUs")
    sfs_fields = {field: annotations.get(field) for field in CPU_PREFLIGHT_BINDING_ANNOTATIONS}
    if _has_lora_cpu_preflight_surface(manifest):
        if not all(isinstance(value, str) and value for value in sfs_fields.values()):
            raise JobsError("CPU preflight source and SFS annotations must be complete")
        _validate_cpu_sfs_control(manifest, sfs_fields)
    return manifest


def _validate_cpu_sfs_control(manifest: dict, fields: dict[str, str]) -> None:
    """Bind one non-root control transaction below an already-owned SFS run tree."""

    spec = manifest["spec"]
    containers = spec["containers"]
    if len(containers) != 1 or spec.get("initContainers") not in (None, []):
        raise JobsError("CPU SFS control must use one non-root container and no initializer")
    if spec.get("automountServiceAccountToken") is not False:
        raise JobsError("CPU SFS control must not mount a service account token")
    if not isinstance(spec.get("activeDeadlineSeconds"), int) or not (
        1 <= spec["activeDeadlineSeconds"] <= 3600
    ):
        raise JobsError("CPU SFS control must have a fixed deadline of at most one hour")
    if any(spec.get(field) not in (None, False) for field in ("hostNetwork", "hostPID", "hostIPC")):
        raise JobsError("CPU SFS control must not join host namespaces")
    if spec.get("securityContext") != {
        "runAsNonRoot": True,
        "runAsUser": TRAINER_UID,
        "runAsGroup": TRAINER_GID,
        "fsGroup": TRAINER_GID,
    }:
        raise JobsError("CPU SFS control must use the proven trainer and SFS group identity")
    if spec.get("imagePullSecrets") != [{"name": "ghcr-pull"}]:
        raise JobsError("CPU SFS control may use only the reviewed image pull Secret")
    container = containers[0]
    if container.get("image") != LORA_TRAINER_IMAGE:
        raise JobsError("CPU SFS control must use the exact reviewed LoRA trainer image")
    if container.get("command") != ["python", "/bundle/preflight_driver.py"]:
        raise JobsError("CPU SFS control must run only the reviewed preflight driver")
    if container.get("securityContext") != {
        "allowPrivilegeEscalation": False,
        "capabilities": {"drop": ["ALL"]},
    }:
        raise JobsError("CPU SFS control must drop capabilities and privilege escalation")
    if container.get("envFrom") not in (None, []):
        raise JobsError("CPU SFS control must not import Secret-backed environment")
    environment = {}
    for entry in container.get("env", []):
        if (
            not isinstance(entry, dict)
            or set(entry) != {"name", "value"}
            or not isinstance(entry.get("name"), str)
            or not isinstance(entry.get("value"), str)
            or entry["name"] in environment
        ):
            raise JobsError("CPU SFS control environment must contain literal unique values")
        environment[entry["name"]] = entry["value"]
    owned_root = fields[CPU_SFS_OWNED_ROOT_ANNOTATION]
    output_root = fields[CPU_SFS_OUTPUT_ROOT_ANNOTATION]
    if set(environment) != set(CPU_PREFLIGHT_ENV):
        raise JobsError("CPU SFS control environment differs from the packaged driver contract")
    if (
        environment.get(ENV_SFS_OWNED_ROOT) != owned_root
        or environment.get(ENV_SFS_OUTPUT_ROOT) != output_root
    ):
        raise JobsError("CPU SFS control path annotations and environment differ")
    if environment.get(ENV_SFS_CONTROL_MOUNT) != "/controls":
        raise JobsError("CPU SFS control must bind the reviewed writable subpath mount")
    if (
        environment.get(ENV_BUNDLE_SHA256) != fields[CPU_PREFLIGHT_BUNDLE_SHA256_ANNOTATION]
        or environment.get(ENV_SOURCE_ARCHIVE_SHA256)
        != fields[CPU_SOURCE_ARCHIVE_SHA256_ANNOTATION]
        or environment.get(ENV_SOURCE_COMMIT) != fields[CPU_SOURCE_COMMIT_ANNOTATION]
    ):
        raise JobsError("CPU preflight source annotations and environment differ")
    for annotation in (
        CPU_PREFLIGHT_BUNDLE_SHA256_ANNOTATION,
        CPU_SOURCE_ARCHIVE_SHA256_ANNOTATION,
    ):
        if re.fullmatch(r"[0-9a-f]{64}", fields[annotation]) is None:
            raise JobsError("CPU preflight package digest is invalid")
    try:
        canonical_source_commit_bytes(fields[CPU_SOURCE_COMMIT_ANNOTATION])
    except ValueError as exc:
        raise JobsError(str(exc)) from None
    runtime_archive = environment.get(ENV_SOURCE_ARCHIVE, "")
    if not runtime_archive.startswith("/mnt/sfs/") or "/../" in runtime_archive:
        raise JobsError("CPU preflight source archive must use the read-only SFS mount")
    try:
        validate_owned_output_binding(owned_root, output_root)
    except ValueError as exc:
        raise JobsError(str(exc)) from None
    try:
        floor_mib = int(fields[CPU_SFS_MEMORY_FLOOR_MIB_ANNOTATION])
    except ValueError:
        raise JobsError("CPU SFS control memory floor is invalid") from None
    if floor_mib <= 0:
        raise JobsError("CPU SFS control memory floor is invalid")
    requests = container.get("resources", {}).get("requests", {})
    limits = container.get("resources", {}).get("limits", {})
    request_memory = _memory_bytes(requests.get("memory"))
    limit_memory = _memory_bytes(limits.get("memory"))
    if request_memory < floor_mib * 1024**2 or limit_memory < request_memory:
        raise JobsError("CPU SFS control memory does not cover its observed contract")
    mounts = container.get("volumeMounts", [])
    if mounts != [
        {"name": "bundle", "mountPath": "/bundle", "readOnly": True},
        {"name": "sfs-readonly", "mountPath": "/mnt/sfs", "readOnly": True},
        {
            "name": "sfs-control",
            "mountPath": "/controls",
            "subPath": "jobs/chris-q38-study-corpora-v1/launch-controls",
        },
    ]:
        raise JobsError("CPU SFS control must expose only read-only SFS plus its control subpath")
    volumes = spec.get("volumes", [])
    expected_bundle = {
        "name": "bundle",
        "configMap": {"name": manifest["metadata"]["name"] + "-source", "defaultMode": 292},
    }
    if volumes != [
        expected_bundle,
        {"name": "sfs-readonly", "persistentVolumeClaim": {"claimName": "sfs-shared"}},
        {"name": "sfs-control", "persistentVolumeClaim": {"claimName": "sfs-shared"}},
    ]:
        raise JobsError("CPU SFS control must use only its immutable bundle and shared SFS")


def _templates(obj: dict) -> list[tuple[str, dict]]:
    try:
        cluster = obj["spec"]["rayClusterSpec"]
        values = [("head", cluster["headGroupSpec"]["template"])]
        values.extend(
            (f"worker[{index}]", group["template"])
            for index, group in enumerate(cluster.get("workerGroupSpecs", []))
        )
        return values
    except (KeyError, TypeError) as exc:
        raise JobsError("malformed RayJob pod templates") from exc


def _env(container: dict) -> dict[str, str]:
    entries = container.get("env", [])
    if not isinstance(entries, list):
        raise JobsError("preview container environment is not a list")
    result: dict[str, str] = {}
    for entry in entries:
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("name"), str)
            or set(entry) != {"name", "value"}
            or not isinstance(entry.get("value"), str)
            or entry["name"] in result
        ):
            raise JobsError("preview has ambiguous runtime environment")
        result[entry["name"]] = entry["value"]
    return result


def _secret_names(container: dict) -> list[str]:
    entries = container.get("envFrom", [])
    if not isinstance(entries, list):
        raise JobsError("preview container envFrom is not a list")
    result = []
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"secretRef"}:
            raise JobsError("preview contains an unsupported envFrom source")
        ref = entry["secretRef"]
        if (
            not isinstance(ref, dict)
            or set(ref) != {"name"}
            or not isinstance(ref.get("name"), str)
        ):
            raise JobsError("preview Secret reference is not exact and required")
        result.append(ref["name"])
    if len(result) != len(set(result)):
        raise JobsError("preview has duplicate Secret references")
    return result


def _require_preview_keys(
    value: object,
    *,
    required: set[str],
    optional: set[str] = frozenset(),
    context: str,
) -> dict:
    if not isinstance(value, dict):
        raise JobsError(f"preview {context} is not an object")
    keys = set(value)
    if not required.issubset(keys) or not keys.issubset(required | optional):
        raise JobsError(f"preview {context} contains an unreviewed field")
    return value


def _standard_training_storage(request: dict) -> tuple[list[dict], list[dict], list[dict]]:
    mounts = [
        {"name": "sfs", "mountPath": "/mnt/sfs"},
        {"name": "shm", "mountPath": "/dev/shm"},
        {
            "name": "trajectory-spool",
            "mountPath": "/mnt/fleet/trajectory-spool",
        },
    ]
    volumes = [
        {
            "name": "sfs",
            "persistentVolumeClaim": {"claimName": "sfs-shared"},
        },
        {
            "name": "shm",
            "emptyDir": {"medium": "Memory", "sizeLimit": "64Gi"},
        },
        {
            "name": "trajectory-spool",
            "hostPath": {
                "path": "/scratch/trajectories",
                "type": "DirectoryOrCreate",
            },
        },
    ]
    init_containers = [
        {
            "name": "sfs-init",
            "image": "busybox:1.36",
            "command": [
                "sh",
                "-c",
                f"mkdir -p {request['run_dir']} && chown 1000:100 {request['run_dir']}",
            ],
            "securityContext": {"runAsUser": 0},
            "volumeMounts": [{"name": "sfs", "mountPath": "/mnt/sfs"}],
        },
        {
            "name": "prepare-trajectory-spool",
            "image": "public.ecr.aws/docker/library/busybox:1.37.0",
            "command": ["sh", "-ec"],
            "args": ["chgrp 2000 /trajectory-spool; chmod 2770 /trajectory-spool"],
            "securityContext": {
                "allowPrivilegeEscalation": False,
                "capabilities": {"add": ["CHOWN", "FOWNER"], "drop": ["ALL"]},
                "readOnlyRootFilesystem": True,
                "runAsNonRoot": False,
                "runAsUser": 0,
            },
            "volumeMounts": [{"name": "trajectory-spool", "mountPath": "/trajectory-spool"}],
        },
    ]
    return mounts, volumes, init_containers


def _assert_source_preview_pod(
    template: object,
    request: dict,
    *,
    group_name: str,
    container_name: str,
) -> None:
    template = _require_preview_keys(
        template,
        required={"metadata", "spec"},
        context=f"{group_name} template",
    )
    metadata = _require_preview_keys(
        template["metadata"],
        required={"labels", "annotations"},
        context=f"{group_name} template metadata",
    )
    if metadata["labels"] != {
        "fleet.ai/run-id": ZERO_RUN_ID,
        "fleet.ai/run-name": request["name"],
    } or metadata["annotations"] != {
        "kueue.x-k8s.io/podset-preferred-topology": "topology.nebius.com/tier-1"
    }:
        raise JobsError(f"preview {group_name} template metadata drift")
    pod = _require_preview_keys(
        template["spec"],
        required={
            "containers",
            "initContainers",
            "nodeSelector",
            "priorityClassName",
            "securityContext",
            "tolerations",
            "volumes",
        },
        optional={
            "imagePullSecrets",
            "priority",
        },
        context=f"{group_name} PodSpec",
    )
    if pod["priorityClassName"] != "c1" or pod.get("priority") not in (None, 10_000):
        raise JobsError(f"preview {group_name} priority drift")
    selector = pod["nodeSelector"]
    if selector not in (
        {"workload": "fleetai-training-ng-gpu"},
        {
            "kubernetes.io/os": "linux",
            "workload": "fleetai-training-ng-gpu",
        },
    ):
        raise JobsError(f"preview {group_name} node selector drift")
    expected_pulls = [{"name": name} for name in request.get("image_pull_secrets", [])]
    if pod.get("imagePullSecrets", []) != expected_pulls:
        raise JobsError(f"preview {group_name} image-pull Secret drift")
    containers = pod["containers"]
    if not isinstance(containers, list) or len(containers) != 1:
        raise JobsError(f"preview {group_name} must contain one app container")
    container = _require_preview_keys(
        containers[0],
        required={
            "name",
            "image",
            "env",
            "envFrom",
            "resources",
            "volumeMounts",
        },
        optional={"securityContext"},
        context=f"{group_name} app container",
    )
    if container["name"] != container_name or container["image"] != request["image"]:
        raise JobsError(f"preview {group_name} app container identity drift")
    resources = _require_preview_keys(
        container["resources"],
        required={"requests", "limits"},
        context=f"{group_name} resources",
    )
    for resource_kind in ("requests", "limits"):
        values = _require_preview_keys(
            resources[resource_kind],
            required={"cpu", "memory", "nvidia.com/gpu"},
            context=f"{group_name} {resource_kind}",
        )
        if len(values) != 3:
            raise JobsError(f"preview {group_name} {resource_kind} resource drift")
    if container.get("securityContext") not in (None, {"privileged": False}):
        raise JobsError(f"preview {group_name} app security context drift")

    standard_mounts, standard_volumes, standard_init = _standard_training_storage(request)
    if (
        container.get("volumeMounts") != standard_mounts
        or pod.get("volumes") != standard_volumes
        or pod.get("initContainers") != standard_init
        or pod.get("securityContext") != {"supplementalGroups": [2000]}
        or pod.get("tolerations")
        != [
            {
                "effect": "NoSchedule",
                "key": "workload",
                "operator": "Equal",
                "value": "fleetai-training-ng-gpu",
            }
        ]
    ):
        raise JobsError(f"preview {group_name} storage or init surface drift")


def _assert_source_preview_surface(obj: dict, request: dict) -> None:
    """Allow only the reviewed Jobs API preview shape before it becomes intent."""
    _require_preview_keys(
        obj,
        required={"apiVersion", "kind", "metadata", "spec"},
        context="RayJob",
    )
    metadata = _require_preview_keys(
        obj["metadata"],
        required={"name", "namespace", "labels", "annotations"},
        context="RayJob metadata",
    )
    expected_labels = {
        "app": "fleet-rl-job",
        "fleet.ai/requeue-if-preempted": "false",
        "fleet.ai/run-id": ZERO_RUN_ID,
        "fleet.ai/run-name": request["name"],
        "kueue.x-k8s.io/priority-class": "q1",
        "kueue.x-k8s.io/queue-name": "training-lq",
    }
    if metadata["labels"] != expected_labels or set(metadata["annotations"]) != {
        "fleet.ai/job-image",
        "fleet.ai/run-dir",
        "fleet.ai/run-id",
        "fleet.ai/submitted-by",
        "fleet.ai/submitted-by-profile",
    }:
        raise JobsError("preview root scheduling or annotation surface drift")
    spec = _require_preview_keys(
        obj["spec"],
        required={
            "entrypoint",
            "rayClusterSpec",
            "shutdownAfterJobFinishes",
            "submissionMode",
            "suspend",
        },
        context="RayJob spec",
    )
    if (
        spec["entrypoint"] != request["command"]
        or spec["submissionMode"] != "HTTPMode"
        or spec["suspend"] is not True
        or spec["shutdownAfterJobFinishes"] is not True
    ):
        raise JobsError("preview RayJob execution policy drift")
    cluster = _require_preview_keys(
        spec["rayClusterSpec"],
        required={"enableInTreeAutoscaling", "headGroupSpec"},
        optional={"workerGroupSpecs"},
        context="RayCluster spec",
    )
    if cluster["enableInTreeAutoscaling"] is not False:
        raise JobsError("preview RayCluster autoscaling drift")
    head = _require_preview_keys(
        cluster["headGroupSpec"],
        required={"rayStartParams", "template"},
        context="head group",
    )
    if head["rayStartParams"] != {"dashboard-host": "0.0.0.0"}:
        raise JobsError("preview head Ray start parameters drift")
    _assert_source_preview_pod(
        head["template"],
        request,
        group_name="head",
        container_name="ray-head",
    )
    workers = cluster.get("workerGroupSpecs", [])
    expected_worker_groups = 0 if request["workers"] == 1 else 1
    if not isinstance(workers, list) or len(workers) != expected_worker_groups:
        raise JobsError("preview worker group count drift")
    if workers:
        worker = _require_preview_keys(
            workers[0],
            required={
                "groupName",
                "replicas",
                "minReplicas",
                "maxReplicas",
                "rayStartParams",
                "template",
            },
            context="worker group",
        )
        replicas = request["workers"] - 1
        if (
            worker["groupName"] != "gpu"
            or worker["replicas"] != replicas
            or worker["minReplicas"] != replicas
            or worker["maxReplicas"] != replicas
            or worker["rayStartParams"] != {}
        ):
            raise JobsError("preview fixed worker group drift")
        _assert_source_preview_pod(
            worker["template"],
            request,
            group_name="worker[0]",
            container_name="ray-worker",
        )


def _assert_sft_contract(plan: dict, request: dict) -> None:
    validate_request(request)
    if plan.get("schema") not in SFT_SCHEMAS:
        raise JobsError("direct fallback is restricted to the qualified SFT runtime")
    if request.get("secrets") != [SFT_SECRET]:
        raise JobsError("SFT direct fallback requires exactly the W&B Secret")
    if any("fleet" in name.lower() for name in request["secrets"]):
        raise JobsError("SFT request unexpectedly requires a Fleet credential Secret")
    if request.get("priority_class") != "c1":
        raise JobsError("direct fallback is restricted to current c1/q1 policy")


def _assert_lr30_contract(plan: dict, request: dict, *, require_launchable: bool) -> None:
    from training.qwen38_lr30_step76_gate import validate_submission_contract

    try:
        validate_submission_contract(plan, request, require_launchable=require_launchable)
    except ValueError as exc:
        raise JobsError(str(exc)) from None


def _expected_generated_env(request: dict, placeholder_name: str) -> dict[str, str]:
    return {
        "FLEET_EXTERNAL_RAY": "1",
        "FLEET_GPUS_PER_WORKER": str(request["gpus_per_worker"]),
        "FLEET_RUN_ID": ZERO_RUN_ID,
        "FLEET_RUN_NAME": placeholder_name,
        "FLEET_TRACE_ROOT": "/mnt/fleet/trajectory-spool",
        "RAY_memory_usage_threshold": "0.98",
        "RUN_DIR": request["run_dir"],
        "SKYPILOT_NUM_GPUS_PER_NODE": str(request["gpus_per_worker"]),
        "WORKERS": str(request["workers"]),
    }


def _walk_strings(value: Any, path: tuple[Any, ...] = ()) -> list[tuple[tuple[Any, ...], str]]:
    if isinstance(value, dict):
        result = []
        for key, item in value.items():
            result.extend(_walk_strings(item, (*path, key)))
        return result
    if isinstance(value, list):
        result = []
        for index, item in enumerate(value):
            result.extend(_walk_strings(item, (*path, index)))
        return result
    return [(path, value)] if isinstance(value, str) else []


def _parse_preview(preview: dict) -> dict:
    if preview.get("errors") or preview.get("warnings"):
        raise JobsError("preview reported errors/warnings; direct rendering stopped")
    try:
        obj = yaml.safe_load(preview["manifest_yaml"])
    except (KeyError, TypeError, yaml.YAMLError) as exc:
        raise JobsError("malformed Jobs API preview") from exc
    if not isinstance(obj, dict):
        raise JobsError("Jobs API preview is not one Kubernetes object")
    return obj


def _render_rayjob(
    request: dict,
    preview: dict,
    *,
    expected_secrets: list[str],
    run_id: str | None = None,
) -> tuple[dict, dict]:
    """Return one reviewed RayJob and a sanitized proof; create nothing."""
    source = _parse_preview(preview)
    obj = deepcopy(source)
    expected_placeholder_name = request["name"] + "-00000000"
    try:
        meta = obj["metadata"]
        if obj.get("apiVersion") != "ray.io/v1" or obj.get("kind") != "RayJob":
            raise JobsError("preview is not a ray.io/v1 RayJob")
        if meta.get("name") != expected_placeholder_name or meta.get("namespace") != NAMESPACE:
            raise JobsError("preview placeholder name/namespace drift")
        if meta.get("generateName") is not None or any(
            key in meta for key in ("uid", "resourceVersion", "creationTimestamp")
        ):
            raise JobsError("preview unexpectedly contains persisted object identity")
        labels = meta["labels"]
        annotations = meta["annotations"]
        if labels.get("fleet.ai/run-id") != ZERO_RUN_ID:
            raise JobsError("preview root run ID is not the zero placeholder")
        if labels.get("fleet.ai/run-name") != request["name"]:
            raise JobsError("preview root run-name label drift")
        if annotations.get("fleet.ai/run-id") != ZERO_RUN_ID:
            raise JobsError("preview root run ID annotation drift")
        if FAILURE_ALERT_ANNOTATION in annotations:
            raise JobsError("preview already carries an alert setting; use the normal API rail")
        if annotations.get("fleet.ai/job-image") != request["image"]:
            raise JobsError("preview root image annotation drift")
        if annotations.get("fleet.ai/run-dir") != request["run_dir"]:
            raise JobsError("preview root output annotation drift")
        submitted_by = annotations.get("fleet.ai/submitted-by")
        submitted_by_profile = annotations.get("fleet.ai/submitted-by-profile")
        if (
            not isinstance(submitted_by, str)
            or len(submitted_by) > 254
            or re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9._%+\-]{0,63}@[A-Za-z0-9]"
                r"(?:[A-Za-z0-9.\-]{0,251}[A-Za-z0-9])?",
                submitted_by,
            )
            is None
        ):
            raise JobsError("preview submitter identity is missing")
        try:
            canonical_profile = str(uuid.UUID(submitted_by_profile))
        except (AttributeError, TypeError, ValueError):
            raise JobsError("preview submitter profile is not a canonical UUID") from None
        if canonical_profile != submitted_by_profile:
            raise JobsError("preview submitter profile is not a canonical UUID")
    except (KeyError, TypeError) as exc:
        raise JobsError("malformed Jobs API preview identity") from exc

    _assert_source_preview_surface(obj, request)
    template_records = _templates(obj)
    expected_env = {
        **request.get("env", {}),
        **_expected_generated_env(request, expected_placeholder_name),
    }
    generated_secret = expected_placeholder_name + "-fleet-key"
    removed = 0
    for group_name, template in template_records:
        try:
            pod_labels = template["metadata"]["labels"]
            if pod_labels.get("fleet.ai/run-id") != ZERO_RUN_ID:
                raise JobsError(f"{group_name} template run ID drift")
            if pod_labels.get("fleet.ai/run-name") != request["name"]:
                raise JobsError(f"{group_name} template run-name drift")
            pod_spec = template["spec"]
            if pod_spec.get("priority") not in (None, 10_000):
                raise JobsError(f"{group_name} numeric priority drift")
            # Bind effective c1 numerically in the object itself. Fleet's live
            # RayJob API server preserves this field but does not default it.
            pod_spec["priority"] = 10_000
            containers = pod_spec["containers"]
            if not isinstance(containers, list) or len(containers) != 1:
                raise JobsError(f"{group_name} must contain exactly one workload container")
            container = containers[0]
            if _env(container) != expected_env:
                raise JobsError(f"{group_name} generated environment drift")
            names = _secret_names(container)
            if names != [*expected_secrets, generated_secret]:
                raise JobsError(f"{group_name} Secret injection drift")
            container["envFrom"] = [
                entry
                for entry in container["envFrom"]
                if entry["secretRef"]["name"] != generated_secret
            ]
            removed += 1
        except (KeyError, TypeError) as exc:
            raise JobsError(f"malformed {group_name} SFT template") from exc

    raw_strings = _walk_strings(source)
    allowed_old_name_paths = {("metadata", "name")}
    allowed_zero_paths = {
        ("metadata", "labels", "fleet.ai/run-id"),
        ("metadata", "annotations", "fleet.ai/run-id"),
    }
    for group_index, (_, template) in enumerate(template_records):
        # Resolve the original path from object identity instead of assuming a
        # worker exists for one-node previews.
        prefix: tuple[Any, ...]
        if group_index == 0:
            prefix = ("spec", "rayClusterSpec", "headGroupSpec", "template")
        else:
            prefix = (
                "spec",
                "rayClusterSpec",
                "workerGroupSpecs",
                group_index - 1,
                "template",
            )
        allowed_zero_paths.add((*prefix, "metadata", "labels", "fleet.ai/run-id"))
        # Environment ordering is platform-owned, so find the identity entries.
        entries = template["spec"]["containers"][0]["env"]
        for env_index, entry in enumerate(entries):
            if entry["name"] == "FLEET_RUN_ID":
                allowed_zero_paths.add(
                    (*prefix, "spec", "containers", 0, "env", env_index, "value")
                )
            if entry["name"] == "FLEET_RUN_NAME":
                allowed_old_name_paths.add(
                    (*prefix, "spec", "containers", 0, "env", env_index, "value")
                )

    actual_zero_paths = {path for path, value in raw_strings if value == ZERO_RUN_ID}
    actual_old_name_paths = {
        path for path, value in raw_strings if value == expected_placeholder_name
    }
    if actual_zero_paths != allowed_zero_paths or actual_old_name_paths != allowed_old_name_paths:
        raise JobsError("preview placeholder identity appeared at an unreviewed location")
    if sum(value == generated_secret for _, value in raw_strings) != len(template_records):
        raise JobsError("preview generated Fleet Secret appeared at an unreviewed location")

    selected_run_id = run_id or str(uuid.uuid4())
    uuid4_pattern = r"[a-f0-9]{8}-[a-f0-9]{4}-4[a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}"
    if not re.fullmatch(uuid4_pattern, selected_run_id):
        raise JobsError("direct run identity must be a lowercase UUIDv4")
    run_name = request["name"] + "-" + selected_run_id[:8]
    meta["name"] = run_name
    meta["labels"]["fleet.ai/run-id"] = selected_run_id
    meta["annotations"]["fleet.ai/run-id"] = selected_run_id
    meta["annotations"][FAILURE_ALERT_ANNOTATION] = FAILURE_ALERT_OFF
    for _, template in template_records:
        template["metadata"]["labels"]["fleet.ai/run-id"] = selected_run_id
        entries = template["spec"]["containers"][0]["env"]
        for entry in entries:
            if entry["name"] == "FLEET_RUN_ID":
                entry["value"] = selected_run_id
            elif entry["name"] == "FLEET_RUN_NAME":
                entry["value"] = run_name

    final_strings = _walk_strings(obj)
    if any(
        value in {ZERO_RUN_ID, expected_placeholder_name, generated_secret}
        for _, value in final_strings
    ):
        raise JobsError("placeholder identity or generated Fleet Secret survived rendering")
    if removed != len(template_records):
        raise JobsError("generated Fleet Secret removal count drift")

    validated = validate_preview(
        request,
        {"manifest_yaml": yaml.safe_dump(obj, sort_keys=False), "warnings": [], "errors": []},
    )
    return obj, {
        **validated,
        "preview_manifest_sha256": digest(source),
        "manifest_sha256": digest(obj),
        "run_id": selected_run_id,
        "name": run_name,
        "removed_api_fleet_secrets": removed,
    }


def render_sft_rayjob(
    plan: dict, request: dict, preview: dict, *, run_id: str | None = None
) -> tuple[dict, dict]:
    """Render the maintained SFT-only direct-create fallback."""
    _assert_sft_contract(plan, request)
    return _render_rayjob(request, preview, expected_secrets=[SFT_SECRET], run_id=run_id)


def render_lr30_qualification_rayjob(
    plan: dict, request: dict, preview: dict, *, run_id: str | None = None
) -> tuple[dict, dict]:
    """Render only the exact LR30 step-76 HF inference-forward qualification."""
    _assert_lr30_contract(plan, request, require_launchable=False)
    return _render_rayjob(request, preview, expected_secrets=[], run_id=run_id)


def _json_object(payload: str, operation: str) -> dict:
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise JobsError(f"kubectl {operation} returned invalid JSON") from exc
    if not isinstance(value, dict):
        raise JobsError(f"kubectl {operation} returned a non-object")
    return value


class Kubectl:
    """Small no-shell Kubernetes boundary; mutation is create-only and never retried."""

    def __init__(self, context: str, *, binary: str = "kubectl"):
        valid_context = (
            isinstance(context, str)
            and context
            and not context.startswith("-")
            and re.fullmatch(r"[-A-Za-z0-9_.:@/]+", context)
        )
        if not valid_context:
            raise JobsError("an explicit valid Kubernetes context is required")
        if not binary or binary.startswith("-"):
            raise JobsError("invalid kubectl binary")
        self.context = context
        self.binary = binary

    def _run(self, args: list[str], *, manifest: dict | None = None) -> dict:
        return _json_object(self._run_text(args, manifest=manifest), args[0])

    def _run_text(self, args: list[str], *, manifest: dict | None = None) -> str:
        command = [self.binary, "--context", self.context, "--request-timeout=60s", *args]
        try:
            result = subprocess.run(
                command,
                input=(
                    json.dumps(manifest, sort_keys=True, separators=(",", ":"))
                    if manifest is not None
                    else None
                ),
                text=True,
                capture_output=True,
                timeout=75,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            raise JobsError("kubectl transport failed; do not infer cluster state") from None
        if result.returncode:
            raise JobsError("kubectl operation failed; private server output suppressed")
        return result.stdout

    def list(self, resource: str) -> dict:
        if resource not in {"rayjobs.ray.io", "jobs.batch"}:
            raise JobsError("unsupported duplicate-check resource")
        return self._run(["get", resource, "--namespace", NAMESPACE, "--output=json"])

    def dry_run(self, manifest: dict) -> dict:
        return self._run(
            ["create", "--dry-run=server", "--filename=-", "--output=json"],
            manifest=manifest,
        )

    def create_once(self, manifest: dict) -> dict:
        return self._run(["create", "--filename=-", "--output=json"], manifest=manifest)

    def _cpu_node_inventory(self) -> dict:
        return self._run(
            [
                "get",
                "nodes",
                "--selector=kubernetes.io/arch=amd64,workload=fleetai-training-ng-cpu",
                "--output=json",
            ]
        )

    def get_output_check_job(self, name: str) -> dict:
        if re.fullmatch(r"[a-z0-9](?:[-a-z0-9]{0,61}[a-z0-9])?", name) is None:
            raise JobsError("invalid output-check Job name")
        return self._run(["get", "job", name, "--namespace", NAMESPACE, "--output=json"])

    def list_output_check_pods(self, name: str) -> dict:
        if re.fullmatch(r"[a-z0-9](?:[-a-z0-9]{0,61}[a-z0-9])?", name) is None:
            raise JobsError("invalid output-check Job name")
        return self._run(
            [
                "get",
                "pods",
                "--namespace",
                NAMESPACE,
                "--selector=job-name=" + name,
                "--output=json",
            ]
        )

    def list_output_check_workloads(self, job_uid: str) -> dict:
        if KUBERNETES_UID_PATTERN.fullmatch(job_uid) is None:
            raise JobsError("invalid output-check Job UID")
        return self._run(
            [
                "get",
                "workloads.kueue.x-k8s.io",
                "--namespace",
                NAMESPACE,
                "--selector=kueue.x-k8s.io/job-uid=" + job_uid,
                "--output=json",
            ]
        )

    def get_output_check_service_account(self) -> dict:
        return self._run(
            [
                "get",
                "serviceaccount",
                "default",
                "--namespace",
                NAMESPACE,
                "--output=json",
            ]
        )

    def output_check_logs(self, pod: str) -> str:
        if re.fullmatch(r"[a-z0-9](?:[-a-z0-9]{0,251}[a-z0-9])?", pod) is None:
            raise JobsError("invalid output-check Pod name")
        return self._run_text(
            [
                "logs",
                pod,
                "--namespace",
                NAMESPACE,
                "--container=output-check",
            ]
        )

    def sft_cpu_preflight_logs(self, pod: str) -> str:
        if re.fullmatch(r"[a-z0-9](?:[-a-z0-9]{0,251}[a-z0-9])?", pod) is None:
            raise JobsError("invalid dense-SFT CPU-preflight Pod name")
        return self._run_text(
            [
                "logs",
                pod,
                "--namespace",
                NAMESPACE,
                "--container=preflight",
            ]
        )

    @staticmethod
    def _require_generic_cpu_checkpoint(manifest: dict) -> None:
        validate_cpu_checkpoint_pod(manifest)
        if _has_lora_cpu_preflight_surface(manifest):
            raise JobsError("LoRA CPU preflight must use the exact packaged submission path")

    @staticmethod
    def _require_lora_cpu_preflight_package(package: LoraCpuPreflightPackage) -> dict:
        try:
            proof = validate_lora_cpu_preflight_package(package)
        except (OSError, ValueError) as exc:
            raise JobsError(str(exc)) from None
        validate_cpu_checkpoint_pod(package.pod)
        return proof

    @staticmethod
    def _require_config_map_response(response: dict, package: LoraCpuPreflightPackage) -> None:
        try:
            validate_lora_cpu_preflight_config_map_response(response, package)
        except ValueError as exc:
            raise JobsError(str(exc)) from None

    def dry_run_cpu_checkpoint_pod(self, manifest: dict) -> dict:
        """Server-preview one CPU seal/verifier after the local placement gate."""
        self._require_generic_cpu_checkpoint(manifest)
        inventory = self._cpu_node_inventory()
        validate_cpu_checkpoint_node_fit(manifest, inventory)
        response = self._run(
            ["create", "--dry-run=server", "--filename=-", "--output=json"],
            manifest=manifest,
        )
        validate_cpu_checkpoint_pod(response)
        return response

    def create_cpu_checkpoint_pod_once(self, manifest: dict) -> dict:
        """Create exactly one locally validated CPU seal/verifier Pod."""
        self._require_generic_cpu_checkpoint(manifest)
        inventory = self._cpu_node_inventory()
        validate_cpu_checkpoint_node_fit(manifest, inventory)
        response = self._run(["create", "--filename=-", "--output=json"], manifest=manifest)
        validate_cpu_checkpoint_pod(response)
        return response

    def dry_run_lora_cpu_preflight(self, package: LoraCpuPreflightPackage) -> dict:
        """Server-preview the exact immutable code package and its non-root Pod."""

        proof = self._require_lora_cpu_preflight_package(package)
        inventory = self._cpu_node_inventory()
        validate_cpu_checkpoint_node_fit(package.pod, inventory)
        config_map = self._run(
            ["create", "--dry-run=server", "--filename=-", "--output=json"],
            manifest=package.config_map,
        )
        self._require_config_map_response(config_map, package)
        pod = self._run(
            ["create", "--dry-run=server", "--filename=-", "--output=json"],
            manifest=package.pod,
        )
        validate_cpu_checkpoint_pod(pod)
        return {"config_map": config_map, "pod": pod, "proof": proof}

    def create_lora_cpu_preflight_once(self, package: LoraCpuPreflightPackage) -> dict:
        """Create one immutable bundle and one exact Pod; never retry either call."""

        proof = self._require_lora_cpu_preflight_package(package)
        inventory = self._cpu_node_inventory()
        validate_cpu_checkpoint_node_fit(package.pod, inventory)
        config_map = self._run(
            ["create", "--filename=-", "--output=json"], manifest=package.config_map
        )
        self._require_config_map_response(config_map, package)
        pod = self._run(["create", "--filename=-", "--output=json"], manifest=package.pod)
        validate_cpu_checkpoint_pod(pod)
        return {"config_map": config_map, "pod": pod, "proof": proof}


def _is_direct_run_name(name: str, request_name: str) -> bool:
    """Match only the exact name shape emitted by the direct renderer."""
    return re.fullmatch(re.escape(request_name) + r"-[a-f0-9]{8}", name) is not None


def _assert_api_unique(rows: list[dict], request: dict) -> None:
    if not isinstance(rows, list):
        raise JobsError("Jobs API history is incomplete")
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("name"), str):
            raise JobsError("Jobs API history contains an invalid record")
        if (
            row.get("run_dir") == request["run_dir"]
            or row["name"] == request["name"]
            or _is_direct_run_name(row["name"], request["name"])
            or (request.get("title") is not None and row.get("title") == request["title"])
        ):
            raise JobsError("a Jobs API run already owns this name/title/output")


def _assert_kubernetes_unique(inventories: list[dict], request: dict, proof: dict) -> None:
    for inventory in inventories:
        items = inventory.get("items")
        kind = inventory.get("kind")
        if not isinstance(kind, str) or not kind.endswith("List") or not isinstance(items, list):
            raise JobsError("Kubernetes duplicate inventory is incomplete")
        for item in items:
            if not isinstance(item, dict) or not isinstance(item.get("metadata"), dict):
                raise JobsError("Kubernetes duplicate inventory contains an invalid object")
            meta = item["metadata"]
            name = meta.get("name", "")
            labels = meta.get("labels") or {}
            annotations = meta.get("annotations") or {}
            if (
                not isinstance(name, str)
                or not isinstance(labels, dict)
                or not isinstance(annotations, dict)
            ):
                raise JobsError("Kubernetes duplicate inventory contains invalid metadata")
            if (
                name == proof["name"]
                or name == request["name"]
                or _is_direct_run_name(name, request["name"])
                or labels.get("fleet.ai/run-name") == request["name"]
                or labels.get("fleet.ai/run-id") == proof["run_id"]
                or annotations.get("fleet.ai/run-id") == proof["run_id"]
                or annotations.get("fleet.ai/run-dir") == request["run_dir"]
            ):
                raise JobsError("a Kubernetes object already owns this name/output/run identity")


def _assert_created_identity(
    obj: dict,
    proof: dict,
    request: dict,
    expected_manifest: dict,
    *,
    require_uid: bool,
) -> None:
    try:
        meta = obj["metadata"]
        if (
            obj["apiVersion"] != "ray.io/v1"
            or obj["kind"] != "RayJob"
            or meta["name"] != proof["name"]
            or meta["namespace"] != NAMESPACE
            or meta["labels"]["fleet.ai/run-id"] != proof["run_id"]
            or meta["annotations"]["fleet.ai/run-id"] != proof["run_id"]
            or meta["annotations"][FAILURE_ALERT_ANNOTATION] != FAILURE_ALERT_OFF
        ):
            raise JobsError("Kubernetes response identity differs from create intent")
        if require_uid and not re.fullmatch(r"[a-f0-9-]{36}", meta.get("uid", "")):
            raise JobsError("created RayJob response omitted its immutable UID")
    except (KeyError, TypeError) as exc:
        raise JobsError("malformed Kubernetes create response") from exc
    # The API server may default fields, but it must not change the effective
    # queue, priority, release, resource, image, environment, or Secret
    # contract between the reviewed local render and the persisted object.
    validate_preview(
        request,
        {
            "manifest_yaml": yaml.safe_dump(obj, sort_keys=False),
            "warnings": [],
            "errors": [],
        },
    )
    _assert_exact_rayjob_runtime_surface(obj, expected_manifest)


_RAY_POD_SERVER_DEFAULTS = {
    "dnsPolicy": "ClusterFirst",
    "enableServiceLinks": True,
    "preemptionPolicy": "PreemptLowerPriority",
    "schedulerName": "default-scheduler",
    "serviceAccount": "default",
    "serviceAccountName": "default",
    "terminationGracePeriodSeconds": 30,
}
_RAY_CONTAINER_SERVER_DEFAULTS = {
    "terminationMessagePath": "/dev/termination-log",
    "terminationMessagePolicy": "File",
}
_RAYJOB_SPEC_SERVER_DEFAULTS = {
    "backoffLimit": 0,
    "ttlSecondsAfterFinished": 0,
}
_RAY_WORKER_GROUP_SERVER_DEFAULTS = {"numOfHosts": 1}
_RAY_SERVER_METADATA_FIELDS = {
    "creationTimestamp",
    "generation",
    "managedFields",
    "resourceVersion",
    "uid",
}


def _strip_exact_server_defaults(actual: dict, expected: dict, defaults: dict) -> dict:
    if not isinstance(actual, dict) or not isinstance(expected, dict):
        raise JobsError("Kubernetes response contains a malformed runtime surface")
    normalized = deepcopy(actual)
    for field, value in defaults.items():
        if field not in expected and field in normalized:
            if normalized[field] != value:
                raise JobsError("Kubernetes response changed an effective runtime default")
            normalized.pop(field)
    return normalized


def _normalize_ray_pod_spec(actual: dict, expected: dict) -> dict:
    if actual.get("priority") != 10_000:
        raise JobsError("Kubernetes response did not prove effective c1 priority 10000")
    normalized = _strip_exact_server_defaults(actual, expected, _RAY_POD_SERVER_DEFAULTS)
    actual_containers = normalized.get("containers")
    expected_containers = expected.get("containers")
    if (
        not isinstance(actual_containers, list)
        or not isinstance(expected_containers, list)
        or len(actual_containers) != len(expected_containers)
    ):
        raise JobsError("Kubernetes response changed the exact Ray app containers")
    normalized["containers"] = [
        _strip_exact_server_defaults(
            actual_container,
            expected_container,
            _RAY_CONTAINER_SERVER_DEFAULTS,
        )
        for actual_container, expected_container in zip(
            actual_containers, expected_containers, strict=True
        )
    ]
    return normalized


def _assert_exact_rayjob_runtime_surface(actual: dict, expected: dict) -> None:
    """Strictly compare the full RayJob behavior after harmless API defaults."""
    if not isinstance(actual, dict) or not isinstance(expected, dict):
        raise JobsError("malformed Kubernetes RayJob runtime surface")
    normalized = deepcopy(actual)
    metadata = normalized.get("metadata")
    expected_metadata = expected.get("metadata")
    if not isinstance(metadata, dict) or not isinstance(expected_metadata, dict):
        raise JobsError("malformed Kubernetes RayJob metadata")
    for field in _RAY_SERVER_METADATA_FIELDS:
        if field not in expected_metadata:
            metadata.pop(field, None)
    status = normalized.pop("status", None)
    if status is not None and not isinstance(status, dict):
        raise JobsError("malformed Kubernetes RayJob status")
    try:
        expected_spec = expected["spec"]
        normalized["spec"] = _strip_exact_server_defaults(
            normalized["spec"],
            expected_spec,
            _RAYJOB_SPEC_SERVER_DEFAULTS,
        )
        actual_cluster = normalized["spec"]["rayClusterSpec"]
        expected_cluster = expected_spec["rayClusterSpec"]
        actual_groups = [
            actual_cluster["headGroupSpec"],
            *actual_cluster.get("workerGroupSpecs", []),
        ]
        expected_groups = [
            expected_cluster["headGroupSpec"],
            *expected_cluster.get("workerGroupSpecs", []),
        ]
    except (KeyError, TypeError) as exc:
        raise JobsError("malformed Kubernetes RayJob runtime surface") from exc
    if len(actual_groups) != len(expected_groups):
        raise JobsError("Kubernetes response changed the exact Ray worker groups")
    for index, (actual_group, expected_group) in enumerate(
        zip(actual_groups, expected_groups, strict=True)
    ):
        if index:
            normalized_group = _strip_exact_server_defaults(
                actual_group,
                expected_group,
                _RAY_WORKER_GROUP_SERVER_DEFAULTS,
            )
            actual_cluster["workerGroupSpecs"][index - 1] = normalized_group
            actual_group = normalized_group
        try:
            actual_spec = actual_group["template"]["spec"]
            expected_spec = expected_group["template"]["spec"]
            actual_group["template"]["spec"] = _normalize_ray_pod_spec(actual_spec, expected_spec)
        except (KeyError, TypeError) as exc:
            raise JobsError("malformed Kubernetes Ray Pod template") from exc
    if normalized != expected:
        raise JobsError("Kubernetes response changed the exact RayJob runtime surface")


def _write_intent(path: Path, value: dict) -> None:
    if path.exists() or path.is_symlink():
        raise JobsError("direct-create journal already exists; reconcile, never retry")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        stream.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _append_journal(path: Path, value: dict) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _assert_output_check_job_absent(inventory: dict, name: str) -> None:
    if inventory.get("kind") not in {"List", "JobList"} or not isinstance(
        inventory.get("items"), list
    ):
        raise JobsError("Kubernetes Job inventory is incomplete")
    for item in inventory["items"]:
        try:
            existing = item["metadata"]["name"]
        except (KeyError, TypeError):
            raise JobsError("Kubernetes Job inventory contains invalid metadata") from None
        if existing == name:
            raise JobsError("the exact output-check Job already exists; reconcile, never replay")


def create_sfs_output_check_once(
    *,
    plan: dict,
    request: dict,
    attempt: int,
    kubectl: Kubectl,
    journal: Path,
) -> dict:
    """Create one bounded zero-GPU read-only SFS observer; never retry create."""
    if journal.exists() or journal.is_symlink():
        raise JobsError("output-check journal already exists; reconcile, never retry")
    try:
        package = build_sfs_output_job(plan, request, attempt)
        proof = validate_sfs_output_job_package(package)
        validate_sfs_output_job_node_fit(package, kubectl._cpu_node_inventory())
    except (OSError, ValueError) as exc:
        raise JobsError(str(exc)) from None
    _assert_output_check_job_absent(kubectl.list("jobs.batch"), proof["name"])
    server_object = kubectl.dry_run(package.job)
    try:
        validate_sfs_output_job_response(server_object, package, require_uid=False)
    except ValueError as exc:
        raise JobsError(str(exc)) from None
    _assert_output_check_job_absent(kubectl.list("jobs.batch"), proof["name"])
    _write_intent(
        journal,
        {
            "state": "KUBECTL_CREATE_INTENT_DO_NOT_RETRY",
            "operation": "sfs_output_check",
            "attempt": attempt,
            "name": proof["name"],
            "namespace": NAMESPACE,
            "kubernetes_context": kubectl.context,
            **{key: proof[key] for key in ("plan_sha256", "request_sha256", "manifest_sha256")},
        },
    )
    created = kubectl.create_once(package.job)
    try:
        validate_sfs_output_job_response(created, package, require_uid=True)
    except ValueError as exc:
        raise JobsError(str(exc)) from None
    result = {
        "submitted": True,
        "gpus": 0,
        "name": proof["name"],
        "uid": created["metadata"]["uid"],
        "attempt": attempt,
    }
    _append_journal(journal, {"state": "KUBECTL_CREATE_RESPONSE", **result})
    return result


def collect_sfs_output_check(*, plan: dict, request: dict, attempt: int, kubectl: Kubectl) -> dict:
    """Read one successful exact Job and return its still-fresh sanitized receipt."""
    try:
        package = build_sfs_output_job(plan, request, attempt)
        name = package.job["metadata"]["name"]
        job = kubectl.get_output_check_job(name)
        job_uid = job.get("metadata", {}).get("uid", "")
        if KUBERNETES_UID_PATTERN.fullmatch(job_uid) is None:
            raise JobsError("output-check Job readback omitted its immutable UID")
        workloads = kubectl.list_output_check_workloads(job_uid)
        pods = kubectl.list_output_check_pods(name)
        service_account = kubectl.get_output_check_service_account()
        pod_name = validate_completed_sfs_output_job(
            package,
            job,
            workloads,
            pods,
            service_account,
        )
        logs = kubectl.output_check_logs(pod_name)
        return collect_sfs_output_receipt(
            package,
            job,
            workloads,
            pods,
            service_account,
            logs,
        )
    except (OSError, ValueError) as exc:
        raise JobsError(str(exc)) from None


def create_sft_cpu_preflight_once(
    *,
    directory: Path,
    source_commit: str,
    attempt: int,
    kubectl: Kubectl,
    journal: Path,
) -> dict:
    """Create one tracked dense-SFT zero-GPU CPU preflight; never retry create."""
    if journal.exists() or journal.is_symlink():
        raise JobsError("CPU-preflight journal already exists; reconcile, never retry")
    try:
        package = build_sft_cpu_preflight_job(
            directory,
            source_commit=source_commit,
            attempt=attempt,
        )
        proof = validate_sft_cpu_preflight_job_package(package)
        validate_sft_cpu_preflight_job_node_fit(package, kubectl._cpu_node_inventory())
    except (OSError, ValueError) as exc:
        raise JobsError(str(exc)) from None
    _assert_output_check_job_absent(kubectl.list("jobs.batch"), proof["name"])
    server_object = kubectl.dry_run(package.job)
    try:
        validate_sft_cpu_preflight_job_response(server_object, package, require_uid=False)
    except ValueError as exc:
        raise JobsError(str(exc)) from None
    _assert_output_check_job_absent(kubectl.list("jobs.batch"), proof["name"])
    _write_intent(
        journal,
        {
            "state": "KUBECTL_CREATE_INTENT_DO_NOT_RETRY",
            "operation": "sft_cpu_preflight",
            "attempt": attempt,
            "name": proof["name"],
            "namespace": NAMESPACE,
            "kubernetes_context": kubectl.context,
            **{
                key: proof[key]
                for key in (
                    "plan_sha256",
                    "request_sha256",
                    "manifest_sha256",
                    "bundle_sha256",
                    "driver_sha256",
                    "source_commit",
                )
            },
        },
    )
    created = kubectl.create_once(package.job)
    try:
        validate_sft_cpu_preflight_job_response(created, package, require_uid=True)
    except ValueError as exc:
        raise JobsError(str(exc)) from None
    result = {
        "submitted": True,
        "gpus": 0,
        "name": proof["name"],
        "uid": created["metadata"]["uid"],
        "attempt": attempt,
    }
    _append_journal(journal, {"state": "KUBECTL_CREATE_RESPONSE", **result})
    return result


def collect_sft_cpu_preflight(
    *,
    directory: Path,
    source_commit: str,
    attempt: int,
    kubectl: Kubectl,
) -> dict:
    """Collect one exact admitted, successful dense-SFT CPU-preflight receipt."""
    try:
        package = build_sft_cpu_preflight_job(
            directory,
            source_commit=source_commit,
            attempt=attempt,
        )
        name = package.job["metadata"]["name"]
        job = kubectl.get_output_check_job(name)
        job_uid = job.get("metadata", {}).get("uid", "")
        if KUBERNETES_UID_PATTERN.fullmatch(job_uid) is None:
            raise JobsError("CPU-preflight Job readback omitted its immutable UID")
        workloads = kubectl.list_output_check_workloads(job_uid)
        pods = kubectl.list_output_check_pods(name)
        service_account = kubectl.get_output_check_service_account()
        pod_name, _ = validate_completed_sft_cpu_preflight_job(
            package,
            job,
            workloads,
            pods,
            service_account,
        )
        logs = kubectl.sft_cpu_preflight_logs(pod_name)
        return collect_sft_cpu_preflight_receipt(
            package,
            job,
            workloads,
            pods,
            service_account,
            logs,
        )
    except (OSError, ValueError) as exc:
        raise JobsError(str(exc)) from None


def _direct_submit_once(
    *,
    plan: dict,
    request: dict,
    jobs: Any,
    kubectl: Kubectl,
    journal: Path,
    renderer: Any,
    run_id: str | None = None,
    output_absence_gate: Callable[[], dict] | None = None,
) -> dict:
    """Preview, prove, journal and issue exactly one direct Kubernetes create."""
    if journal.exists() or journal.is_symlink():
        raise JobsError("direct-create journal already exists; reconcile, never retry")

    if output_absence_gate is not None:
        output_absence_gate()
    _assert_api_unique(jobs.all_runs(), request)
    preview = jobs.raw_preview(request)
    manifest, proof = renderer(plan, request, preview, run_id=run_id)
    _assert_kubernetes_unique(
        [kubectl.list("rayjobs.ray.io"), kubectl.list("jobs.batch")], request, proof
    )
    server_object = kubectl.dry_run(manifest)
    _assert_created_identity(
        server_object,
        proof,
        request,
        manifest,
        require_uid=False,
    )

    # Close the read/dry-run race as far as the two authorities permit.  The
    # exact-name Kubernetes create remains the final atomic create-once gate.
    _assert_api_unique(jobs.all_runs(), request)
    _assert_kubernetes_unique(
        [kubectl.list("rayjobs.ray.io"), kubectl.list("jobs.batch")], request, proof
    )
    output_absence_proof = output_absence_gate() if output_absence_gate is not None else None
    _write_intent(
        journal,
        {
            "state": "KUBECTL_CREATE_INTENT_DO_NOT_RETRY",
            "request_sha256": digest(request),
            "plan_sha256": digest(plan),
            "preview_manifest_sha256": proof["preview_manifest_sha256"],
            "manifest_sha256": proof["manifest_sha256"],
            "run_id": proof["run_id"],
            "name": proof["name"],
            "namespace": NAMESPACE,
            "kubernetes_context": kubectl.context,
            **(
                {"output_absence_receipt_sha256": output_absence_proof["sha256"]}
                if output_absence_proof is not None
                else {}
            ),
        },
    )

    # Never wrap this call in retry logic.  Any error after the durable intent
    # is ambiguous until the exact name/run ID is reconciled read-only.
    created = kubectl.create_once(manifest)
    _assert_created_identity(
        created,
        proof,
        request,
        manifest,
        require_uid=True,
    )
    result = {
        "name": proof["name"],
        "run_id": proof["run_id"],
        "namespace": NAMESPACE,
        "uid": created["metadata"]["uid"],
        "manifest_sha256": proof["manifest_sha256"],
        "submitted": True,
        "transport": "direct-kubectl-create",
    }
    _append_journal(journal, {"state": "KUBECTL_CREATE_RESPONSE", **result})
    return result


def direct_submit_sft_once(
    *,
    plan: dict,
    request: dict,
    jobs: Any,
    kubectl: Kubectl,
    journal: Path,
    run_id: str | None = None,
    jobs_root: Path = SFS_JOBS_ROOT,
    output_absence_receipt: dict | None = None,
) -> dict:
    """Create one source-bound SFT RayJob through the maintained fallback."""
    _assert_sft_contract(plan, request)
    from training.sft import job_request

    if job_request(plan) != request:
        raise JobsError("saved SFT request differs from the current source-bound renderer")

    def output_absence_gate() -> dict:
        try:
            return prove_output_absent(
                plan,
                request,
                jobs_root=jobs_root,
                receipt=output_absence_receipt,
            )
        except ValueError as exc:
            raise JobsError(str(exc)) from None

    return _direct_submit_once(
        plan=plan,
        request=request,
        jobs=jobs,
        kubectl=kubectl,
        journal=journal,
        renderer=render_sft_rayjob,
        run_id=run_id,
        output_absence_gate=output_absence_gate,
    )


def direct_submit_lr30_qualification_once(
    *,
    plan: dict,
    request: dict,
    jobs: Any,
    kubectl: Kubectl,
    journal: Path,
    run_id: str | None = None,
) -> dict:
    """Create only the approved LR30 step-76 HF inference-forward qualification."""
    _assert_lr30_contract(plan, request, require_launchable=True)
    from training.qwen38_lr30_step76_gate import job_request

    if job_request() != request:
        raise JobsError("saved LR30 request differs from the current source-bound renderer")
    return _direct_submit_once(
        plan=plan,
        request=request,
        jobs=jobs,
        kubectl=kubectl,
        journal=journal,
        renderer=render_lr30_qualification_rayjob,
        run_id=run_id,
    )
