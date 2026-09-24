"""Create-once, zero-GPU Job for a fresh read-only SFS observation."""

from __future__ import annotations

import base64
import hashlib
import json
import re
from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from .gpu_capacity import _effective_pod_quantity
from .jobs import FAILURE_ALERT_ANNOTATION, FAILURE_ALERT_OFF, digest, validate_request
from .sfs_output import validate_output_absence_receipt
from .sfs_output_driver import (
    ENV_DRIVER_SHA256,
    ENV_DRIVER_SOURCE,
    ENV_PLAN_SHA256,
    ENV_REQUEST_SHA256,
    ENV_RUN_DIR,
    ENV_RUN_NAME,
    LOG_PREFIX,
)

NAMESPACE = "fleet-train-jobs"
CPU_NODE_SELECTOR = {
    "kubernetes.io/arch": "amd64",
    "workload": "fleetai-training-ng-cpu",
}
JOB_ROLE = "sfs-output-check"
JOB_SUFFIX = "-sfs-a{attempt:02d}"
MAX_ATTEMPT = 99
PLAN_ANNOTATION = "cyber-post-train.fleet.ai/plan-sha256"
REQUEST_ANNOTATION = "cyber-post-train.fleet.ai/request-sha256"
OUTPUT_ANNOTATION = "cyber-post-train.fleet.ai/training-output"
DRIVER_ANNOTATION = "cyber-post-train.fleet.ai/driver-sha256"
ROLE_LABEL = "cyber-post-train.fleet.ai/role"
OWNER_LABEL = "cyber-post-train.fleet.ai/owner"
QUEUE_LABEL = "kueue.x-k8s.io/queue-name"
QUEUE_PRIORITY_LABEL = "kueue.x-k8s.io/priority-class"
QUEUE = "training-lq"
QUEUE_PRIORITY = "q1"
EFFECTIVE_C1_PRIORITY = 10_000
WORKLOAD_PRIORITY_CLASS_REF = {
    "group": "kueue.x-k8s.io",
    "kind": "WorkloadPriorityClass",
    "name": QUEUE_PRIORITY,
}
WORKLOAD_TOPOLOGY_REQUEST = {
    "podIndexLabel": "batch.kubernetes.io/job-completion-index",
}
OUTPUT_CHECK_PLAN_SCHEMAS = {
    "cyber_sft_runtime_v2",
    "cyber_sft_runtime_dense_v1",
    "cyber_qwen38_miles96_mechanics_canary_v1",
    "cyber_qwen38_miles96_signal_qualification_v1",
}
_DRIVER_PATH = Path(__file__).with_name("sfs_output_driver.py")
_SERVER_METADATA_KEYS = {
    "annotations",
    "creationTimestamp",
    "generation",
    "labels",
    "managedFields",
    "name",
    "namespace",
    "resourceVersion",
    "uid",
}
_SERVER_JOB_SPEC_KEYS = {
    "activeDeadlineSeconds",
    "backoffLimit",
    "completionMode",
    "completions",
    "manualSelector",
    "parallelism",
    "podReplacementPolicy",
    "selector",
    "suspend",
    "template",
}
_SERVER_POD_DEFAULTS = {
    "dnsPolicy": "ClusterFirst",
    "enableServiceLinks": True,
    "preemptionPolicy": "PreemptLowerPriority",
    "schedulerName": "default-scheduler",
    "serviceAccount": "default",
    "serviceAccountName": "default",
}
_SERVER_CONTAINER_DEFAULTS = {
    "terminationMessagePath": "/dev/termination-log",
    "terminationMessagePolicy": "File",
}
_BATCH_SELECTOR_LABELS = {
    "batch.kubernetes.io/controller-uid",
    "batch.kubernetes.io/job-name",
    "controller-uid",
    "job-name",
}
_KUEUE_WORKLOAD_ANNOTATION = "kueue.x-k8s.io/workload"
_KUEUE_ADMISSION_LABELS = {
    "kueue.x-k8s.io/cluster-queue-name": "cluster_queue",
    "kueue.x-k8s.io/local-queue-name": "local_queue",
    "kueue.x-k8s.io/podset": "podset",
}
_TOPOLOGY_REGION_LABEL = "topology.kubernetes.io/region"
_DEFAULT_LIVE_TOLERATIONS = (
    {
        "key": "node.kubernetes.io/not-ready",
        "operator": "Exists",
        "effect": "NoExecute",
        "tolerationSeconds": 300,
    },
    {
        "key": "node.kubernetes.io/unreachable",
        "operator": "Exists",
        "effect": "NoExecute",
        "tolerationSeconds": 300,
    },
)
_DEFAULT_SCHEDULED_POD_PULL_SECRETS = [{"name": "ecr-pull"}]
_KUBERNETES_UID = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}"
)


@dataclass(frozen=True)
class SfsOutputJobPackage:
    job: dict[str, Any]
    plan: dict[str, Any]
    request: dict[str, Any]
    attempt: int


def job_name(request: dict, attempt: int) -> str:
    if type(attempt) is not int or not 1 <= attempt <= MAX_ATTEMPT:
        raise ValueError(f"output-check attempt must be between 1 and {MAX_ATTEMPT}")
    value = request.get("name", "") + JOB_SUFFIX.format(attempt=attempt)
    if re.fullmatch(r"[a-z0-9](?:[-a-z0-9]{0,61}[a-z0-9])?", value) is None:
        raise ValueError("prepared run name is too long for the output-check Job suffix")
    return value


def _driver() -> tuple[str, str, str]:
    source_bytes = _DRIVER_PATH.read_bytes()
    try:
        source = source_bytes.decode()
    except UnicodeDecodeError:
        raise ValueError("output-check driver must be UTF-8") from None
    if source.encode() != source_bytes:
        raise ValueError("output-check driver text is not byte-stable")
    return (
        source,
        hashlib.sha256(source_bytes).hexdigest(),
        base64.b64encode(source_bytes).decode(),
    )


def _render(plan: dict, request: dict, attempt: int) -> dict:
    validate_request(request)
    if plan.get("schema") not in OUTPUT_CHECK_PLAN_SCHEMAS:
        raise ValueError("output-check Job plan schema is not supported")
    if request.get("failureAlerts") is not False or request.get("priority_class") != "c1":
        raise ValueError("output-check source request must use alert-off c1 policy")
    expected_name = job_name(request, attempt)
    source, driver_sha256, encoded_source = _driver()
    annotations = {
        FAILURE_ALERT_ANNOTATION: FAILURE_ALERT_OFF,
        PLAN_ANNOTATION: digest(plan),
        REQUEST_ANNOTATION: digest(request),
        OUTPUT_ANNOTATION: request["run_dir"],
        DRIVER_ANNOTATION: driver_sha256,
    }
    labels = {
        OWNER_LABEL: "chris",
        ROLE_LABEL: JOB_ROLE,
        QUEUE_LABEL: QUEUE,
        QUEUE_PRIORITY_LABEL: QUEUE_PRIORITY,
    }
    environment = {
        ENV_DRIVER_SHA256: driver_sha256,
        ENV_DRIVER_SOURCE: encoded_source,
        ENV_PLAN_SHA256: digest(plan),
        ENV_REQUEST_SHA256: digest(request),
        ENV_RUN_NAME: request["name"],
        ENV_RUN_DIR: request["run_dir"],
        "CUDA_VISIBLE_DEVICES": "",
        "NVIDIA_VISIBLE_DEVICES": "none",
        "WANDB_MODE": "disabled",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": expected_name,
            "namespace": NAMESPACE,
            "annotations": deepcopy(annotations),
            "labels": deepcopy(labels),
        },
        "spec": {
            "activeDeadlineSeconds": 300,
            "backoffLimit": 0,
            "completionMode": "NonIndexed",
            "completions": 1,
            "parallelism": 1,
            # Kueue-managed batch Jobs enter the normal queue suspended.  The
            # controller is the only component allowed to unsuspend this Job
            # after quota admission.
            "suspend": True,
            "template": {
                "metadata": {
                    "annotations": deepcopy(annotations),
                    "labels": deepcopy(labels),
                },
                "spec": {
                    "activeDeadlineSeconds": 300,
                    "automountServiceAccountToken": False,
                    "containers": [
                        {
                            "name": "output-check",
                            "image": request["image"],
                            "imagePullPolicy": "IfNotPresent",
                            "command": ["python", "-u", "-c", source],
                            "env": [
                                {"name": name, "value": value}
                                for name, value in environment.items()
                            ],
                            "resources": {
                                "requests": {"cpu": "1", "memory": "1Gi"},
                                "limits": {"cpu": "1", "memory": "1Gi"},
                            },
                            "securityContext": {
                                "allowPrivilegeEscalation": False,
                                "capabilities": {"drop": ["ALL"]},
                                "privileged": False,
                                "readOnlyRootFilesystem": True,
                                "runAsNonRoot": True,
                            },
                            "volumeMounts": [
                                {"name": "sfs", "mountPath": "/mnt/sfs", "readOnly": True}
                            ],
                        }
                    ],
                    "hostIPC": False,
                    "hostNetwork": False,
                    "hostPID": False,
                    "imagePullSecrets": [
                        {"name": name} for name in request.get("image_pull_secrets", [])
                    ],
                    "nodeSelector": CPU_NODE_SELECTOR,
                    "priority": EFFECTIVE_C1_PRIORITY,
                    "priorityClassName": "c1",
                    "restartPolicy": "Never",
                    "securityContext": {
                        "runAsNonRoot": True,
                        "runAsUser": 1000,
                        "runAsGroup": 100,
                        "fsGroup": 100,
                        "seccompProfile": {"type": "RuntimeDefault"},
                    },
                    "terminationGracePeriodSeconds": 30,
                    "tolerations": [
                        {
                            "key": "workload",
                            "operator": "Equal",
                            "value": "fleetai-training-ng-cpu",
                            "effect": "NoSchedule",
                        }
                    ],
                    "volumes": [
                        {
                            "name": "sfs",
                            "persistentVolumeClaim": {
                                "claimName": "sfs-shared",
                                "readOnly": True,
                            },
                        }
                    ],
                },
            },
        },
    }


def build_sfs_output_job(plan: dict, request: dict, attempt: int) -> SfsOutputJobPackage:
    package = SfsOutputJobPackage(
        job=_render(plan, request, attempt),
        plan=deepcopy(plan),
        request=deepcopy(request),
        attempt=attempt,
    )
    validate_sfs_output_job_package(package)
    return package


def validate_sfs_output_job_package(package: SfsOutputJobPackage) -> dict:
    if not isinstance(package, SfsOutputJobPackage):
        raise ValueError("output check requires one source-bound package")
    expected = _render(package.plan, package.request, package.attempt)
    if package.job != expected:
        raise ValueError("output-check Job differs from the current source-bound renderer")
    return {
        "name": expected["metadata"]["name"],
        "plan_sha256": digest(package.plan),
        "request_sha256": digest(package.request),
        "manifest_sha256": digest(expected),
        "driver_sha256": expected["metadata"]["annotations"][DRIVER_ANNOTATION],
        "gpus": 0,
    }


def _contains(actual: Any, expected: Any) -> bool:
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and _contains(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(
                _contains(actual_value, expected_value)
                for actual_value, expected_value in zip(actual, expected, strict=True)
            )
        )
    return actual == expected


def _only_keys(value: Any, allowed: set[str], label: str) -> dict:
    if not isinstance(value, dict) or not set(value).issubset(allowed):
        raise ValueError(f"server output-check {label} contains an unreviewed field")
    return value


def _validate_selector_and_template_labels(
    actual_spec: dict,
    expected: dict,
    *,
    admitted_workload: dict | None,
    job_name: str,
    job_uid: str | None,
) -> None:
    selector = actual_spec.get("selector")
    selector_labels: dict[str, str] = {}
    if selector is not None:
        if (
            not isinstance(selector, dict)
            or set(selector) != {"matchLabels"}
            or not isinstance(selector["matchLabels"], dict)
            or not selector["matchLabels"]
            or not set(selector["matchLabels"]).issubset(_BATCH_SELECTOR_LABELS)
            or any(
                not isinstance(value, str) or not value
                for value in selector["matchLabels"].values()
            )
        ):
            raise ValueError("server output-check Job selector is not a generated batch identity")
        selector_labels = selector["matchLabels"]
    template_metadata = actual_spec.get("template", {}).get("metadata", {})
    _only_keys(
        template_metadata,
        {"annotations", "creationTimestamp", "labels"},
        "Pod template metadata",
    )
    if template_metadata.get("creationTimestamp") is not None:
        raise ValueError("server output-check Pod template timestamp is not a server default")
    expected_annotations = expected["spec"]["template"]["metadata"]["annotations"]
    expected_labels = deepcopy(expected["spec"]["template"]["metadata"]["labels"])
    batch_labels = {
        "batch.kubernetes.io/controller-uid": job_uid,
        "batch.kubernetes.io/job-name": job_name,
        "controller-uid": job_uid,
        "job-name": job_name,
    }
    if admitted_workload is None:
        if selector_labels:
            if _KUBERNETES_UID.fullmatch(job_uid or "") is None or any(
                batch_labels.get(key) != value for key, value in selector_labels.items()
            ):
                raise ValueError("server output-check Job selector identity drifted")
            expected_labels.update(batch_labels)
    else:
        expected_annotations = {
            **expected_annotations,
            _KUEUE_WORKLOAD_ANNOTATION: admitted_workload["name"],
        }
        if any(batch_labels.get(key) != value for key, value in selector_labels.items()):
            raise ValueError("server output-check Job selector identity drifted")
        expected_labels.update(batch_labels)
        expected_labels.update(
            {label: admitted_workload[value] for label, value in _KUEUE_ADMISSION_LABELS.items()}
        )
    if template_metadata.get("annotations") != expected_annotations:
        raise ValueError("server output-check Pod template annotations drifted")
    if template_metadata.get("labels") != expected_labels:
        raise ValueError("server output-check Pod template labels drifted")


def _normalize_container_serialization(actual: Any, expected: dict) -> Any:
    if not isinstance(actual, dict):
        return actual
    normalized = deepcopy(actual)
    actual_env = normalized.get("env")
    expected_env = expected.get("env")
    if (
        isinstance(actual_env, list)
        and isinstance(expected_env, list)
        and len(actual_env) == len(expected_env)
    ):
        for actual_entry, expected_entry in zip(actual_env, expected_env, strict=True):
            if expected_entry == {"name": "CUDA_VISIBLE_DEVICES", "value": ""} and actual_entry == {
                "name": "CUDA_VISIBLE_DEVICES"
            }:
                actual_entry["value"] = ""
    return normalized


def _validate_container(actual: Any, expected: dict) -> None:
    actual = _normalize_container_serialization(actual, expected)
    container = _only_keys(
        actual,
        {*expected, *_SERVER_CONTAINER_DEFAULTS},
        "container",
    )
    for key, value in _SERVER_CONTAINER_DEFAULTS.items():
        if key in container and key not in expected and container[key] != value:
            raise ValueError("server output-check container default drifted")
        if key in expected and container.get(key) != expected[key]:
            raise ValueError(f"server output-check container {key} drifted")
    if any(container.get(name) not in (None, []) for name in ("envFrom", "ports", "volumeDevices")):
        raise ValueError("server output-check container gained an unreviewed source or device")
    if any(
        container.get(name) is not None
        for name in ("lifecycle", "livenessProbe", "readinessProbe", "startupProbe")
    ):
        raise ValueError("server output-check container gained an unreviewed lifecycle surface")
    for key in (
        "name",
        "image",
        "imagePullPolicy",
        "command",
        "env",
        "resources",
        "securityContext",
        "volumeMounts",
    ):
        if container.get(key) != expected.get(key):
            raise ValueError(f"server output-check container {key} drifted")
    entries = container["env"]
    if any(
        not isinstance(entry, dict)
        or set(entry) != {"name", "value"}
        or not isinstance(entry["name"], str)
        or not isinstance(entry["value"], str)
        for entry in entries
    ):
        raise ValueError("server output-check environment is not literal and exact")


def _validate_tolerations(actual: Any, expected: list[dict], *, scheduled: bool) -> None:
    if not isinstance(actual, list):
        raise ValueError("server output-check tolerations are malformed")
    if not scheduled:
        if actual != expected:
            raise ValueError("server output-check queue tolerations drifted")
        return
    remaining = [deepcopy(item) for item in actual]
    for required in expected:
        if required not in remaining:
            raise ValueError("live output-check Pod lost its reviewed CPU toleration")
        remaining.remove(required)
    if any(item not in _DEFAULT_LIVE_TOLERATIONS for item in remaining) or len(remaining) != len(
        {json.dumps(item, sort_keys=True) for item in remaining}
    ):
        raise ValueError("live output-check Pod gained an unreviewed toleration")


def _normalize_pod_serialization(actual: Any, expected: dict) -> Any:
    if not isinstance(actual, dict):
        return actual
    normalized = deepcopy(actual)
    for key in ("hostIPC", "hostNetwork", "hostPID"):
        if expected.get(key) is False and key not in normalized:
            normalized[key] = False
    if expected.get("imagePullSecrets") == [] and "imagePullSecrets" not in normalized:
        normalized["imagePullSecrets"] = []
    actual_containers = normalized.get("containers")
    expected_containers = expected.get("containers")
    if (
        isinstance(actual_containers, list)
        and isinstance(expected_containers, list)
        and len(actual_containers) == len(expected_containers)
    ):
        normalized["containers"] = [
            _normalize_container_serialization(actual_container, expected_container)
            for actual_container, expected_container in zip(
                actual_containers, expected_containers, strict=True
            )
        ]
    return normalized


def _validate_default_service_account(actual: Any) -> None:
    if not isinstance(actual, dict):
        raise ValueError("output-check ServiceAccount readback is malformed")
    metadata = actual.get("metadata")
    if (
        actual.get("apiVersion") != "v1"
        or actual.get("kind") != "ServiceAccount"
        or not isinstance(metadata, dict)
        or metadata.get("name") != "default"
        or metadata.get("namespace") != NAMESPACE
        or actual.get("imagePullSecrets") != _DEFAULT_SCHEDULED_POD_PULL_SECRETS
        or actual.get("secrets") not in (None, [])
    ):
        raise ValueError("output-check default ServiceAccount pull-secret binding drifted")


def _validate_pod_spec(
    actual: Any,
    expected: dict,
    *,
    scheduled: bool,
    service_account: dict | None = None,
) -> None:
    actual = _normalize_pod_serialization(actual, expected)
    allowed = {*expected, *_SERVER_POD_DEFAULTS}
    if scheduled:
        allowed.add("nodeName")
    pod_spec = _only_keys(actual, allowed, "Pod spec")
    if scheduled:
        _validate_default_service_account(service_account)
        if pod_spec.get("serviceAccountName") != "default":
            raise ValueError("output-check scheduled Pod ServiceAccount drifted")
    elif service_account is not None:
        raise ValueError("pre-admission output-check validation received a ServiceAccount")
    if scheduled and expected.get("imagePullSecrets") == []:
        if pod_spec.get("imagePullSecrets") != _DEFAULT_SCHEDULED_POD_PULL_SECRETS:
            raise ValueError("server output-check Pod imagePullSecrets drifted")
        # The production namespace's default ServiceAccount injects exactly
        # this ECR pull reference into the scheduled Pod, not the Job template
        # or Kueue PodSet. It is never accepted as an env/volume Secret.
        pod_spec["imagePullSecrets"] = []
    for key, value in _SERVER_POD_DEFAULTS.items():
        if key in pod_spec and pod_spec[key] != value:
            raise ValueError(f"server output-check Pod default {key} drifted")
    if pod_spec.get("priority") != EFFECTIVE_C1_PRIORITY:
        raise ValueError("server output-check Pod effective c1 priority drifted")
    if scheduled:
        if not isinstance(pod_spec.get("nodeName"), str) or not pod_spec["nodeName"]:
            raise ValueError("successful output-check Pod has no assigned node")
    elif pod_spec.get("nodeName") is not None:
        raise ValueError("server output-check Job pinned one node before admission")
    for key in (
        "affinity",
        "ephemeralContainers",
        "initContainers",
        "overhead",
        "runtimeClassName",
        "schedulingGates",
    ):
        if pod_spec.get(key) not in (None, [], {}):
            raise ValueError(f"server output-check Pod gained unreviewed {key}")
    for key in (
        "activeDeadlineSeconds",
        "automountServiceAccountToken",
        "hostIPC",
        "hostNetwork",
        "hostPID",
        "imagePullSecrets",
        "nodeSelector",
        "priority",
        "priorityClassName",
        "restartPolicy",
        "securityContext",
        "terminationGracePeriodSeconds",
        "volumes",
    ):
        if pod_spec.get(key) != expected.get(key):
            raise ValueError(f"server output-check Pod {key} drifted")
    containers = pod_spec.get("containers")
    if not isinstance(containers, list) or len(containers) != 1:
        raise ValueError("output-check Job must keep exactly one app container")
    _validate_container(containers[0], expected["containers"][0])
    _validate_tolerations(pod_spec.get("tolerations"), expected["tolerations"], scheduled=scheduled)
    for field in ("requests", "limits"):
        if _effective_pod_quantity(pod_spec, field) != 0:
            raise ValueError("output-check Pod must remain effectively zero-GPU")


def _validate_server_job_surface(
    actual: dict,
    expected: dict,
    *,
    admitted_workload: dict | None,
) -> None:
    admitted = admitted_workload is not None
    _only_keys(actual, {"apiVersion", "kind", "metadata", "spec", "status"}, "Job")
    if not admitted and actual.get("status") not in (None, {}):
        raise ValueError("server dry-run output-check Job unexpectedly contains live status")
    metadata = _only_keys(actual.get("metadata"), _SERVER_METADATA_KEYS, "Job metadata")
    if metadata.get("annotations") != expected["metadata"]["annotations"]:
        raise ValueError("server output-check root annotations drifted")
    if metadata.get("labels") != expected["metadata"]["labels"]:
        raise ValueError("server output-check root labels drifted")
    spec = _only_keys(actual.get("spec"), _SERVER_JOB_SPEC_KEYS, "Job spec")
    if spec.get("manualSelector") not in (None, False):
        raise ValueError("server output-check Job enabled manual selector control")
    if spec.get("podReplacementPolicy") not in (None, "Failed", "TerminatingOrFailed"):
        raise ValueError("server output-check Job replacement policy drifted")
    _validate_selector_and_template_labels(
        spec,
        expected,
        admitted_workload=admitted_workload,
        job_name=metadata.get("name", ""),
        job_uid=metadata.get("uid"),
    )
    _validate_pod_spec(
        spec.get("template", {}).get("spec"),
        expected["spec"]["template"]["spec"],
        scheduled=False,
    )


def _normalize_job_serialization(actual: Any, expected: dict) -> Any:
    if not isinstance(actual, dict):
        return actual
    normalized = deepcopy(actual)
    try:
        actual_spec = normalized["spec"]["template"]["spec"]
        expected_spec = expected["spec"]["template"]["spec"]
    except (KeyError, TypeError):
        return normalized
    normalized["spec"]["template"]["spec"] = _normalize_pod_serialization(
        actual_spec,
        expected_spec,
    )
    return normalized


def validate_sfs_output_job_response(
    actual: dict,
    package: SfsOutputJobPackage,
    *,
    require_uid: bool,
    admitted: bool = False,
    admitted_workload: dict | None = None,
) -> dict:
    proof = validate_sfs_output_job_package(package)
    expected = deepcopy(package.job)
    if admitted:
        # A successfully admitted Kueue Job is the exact submitted manifest
        # with only its queue-controlled suspension state released.  All
        # reviewed labels, annotations, Pod controls, and resource fields must
        # still match.
        expected["spec"]["suspend"] = False
        if admitted_workload is None:
            raise ValueError("admitted output-check Job requires exact Workload evidence")
    elif admitted_workload is not None:
        raise ValueError("pre-admission output-check Job cannot carry Workload evidence")
    actual = _normalize_job_serialization(actual, expected)
    if not _contains(actual, expected):
        raise ValueError("server output-check Job differs from the reviewed manifest")
    _validate_server_job_surface(
        actual,
        expected,
        admitted_workload=admitted_workload,
    )
    metadata = actual.get("metadata", {})
    if require_uid and _KUBERNETES_UID.fullmatch(metadata.get("uid", "")) is None:
        raise ValueError("created output-check Job response omitted its immutable UID")
    pod_spec = actual.get("spec", {}).get("template", {}).get("spec", {})
    containers = pod_spec.get("containers")
    if not isinstance(containers, list) or len(containers) != 1:
        raise ValueError("output-check Job must keep exactly one container")
    for container in [*pod_spec.get("initContainers", []), *containers]:
        for values in container.get("resources", {}).values():
            if isinstance(values, dict) and "nvidia.com/gpu" in values:
                raise ValueError("output-check Job must remain zero-GPU")
    return proof


def _cpu_millicores(value: object) -> int:
    if not isinstance(value, str) or not value:
        raise ValueError("CPU node inventory quantity is invalid")
    raw = value[:-1] if value.endswith("m") else value
    try:
        amount = Decimal(raw)
    except InvalidOperation:
        raise ValueError("CPU node inventory quantity is invalid") from None
    amount = amount if value.endswith("m") else amount * 1000
    if amount <= 0 or amount != amount.to_integral_value():
        raise ValueError("CPU node inventory quantity is invalid")
    return int(amount)


def _memory_bytes(value: object) -> int:
    if not isinstance(value, str):
        raise ValueError("CPU node inventory memory is invalid")
    match = re.fullmatch(r"([0-9]+)([KMGT]i)", value)
    if match is None:
        raise ValueError("CPU node inventory memory is invalid")
    power = {"Ki": 1, "Mi": 2, "Gi": 3, "Ti": 4}[match.group(2)]
    return int(match.group(1)) * 1024**power


def validate_sfs_output_job_node_fit(package: SfsOutputJobPackage, inventory: dict) -> dict:
    validate_sfs_output_job_package(package)
    if inventory.get("kind") != "List" or not isinstance(inventory.get("items"), list):
        raise ValueError("CPU node inventory is incomplete")
    eligible = []
    for node in inventory["items"]:
        try:
            labels = node["metadata"]["labels"]
            spec = node["spec"]
            status = node["status"]
            ready = any(
                item.get("type") == "Ready" and item.get("status") == "True"
                for item in status["conditions"]
            )
            cpu = _cpu_millicores(status["allocatable"]["cpu"])
            memory = _memory_bytes(status["allocatable"]["memory"])
        except (KeyError, TypeError):
            raise ValueError("CPU node inventory contains a malformed node") from None
        if (
            all(labels.get(key) == value for key, value in CPU_NODE_SELECTOR.items())
            and spec.get("unschedulable") is not True
            and ready
        ):
            eligible.append((node["metadata"].get("name"), cpu, memory))
    fitting = [item for item in eligible if item[1] >= 1000 and item[2] >= 1024**3]
    if not fitting:
        raise ValueError("output-check Job cannot fit a Ready eligible CPU node")
    return {"eligible_nodes": len(eligible), "fitting_nodes": len(fitting)}


def _validate_workload_podset(
    package: SfsOutputJobPackage,
    job: dict,
    workload_spec: dict,
    admission: dict,
) -> str:
    podsets = workload_spec.get("podSets")
    assignments = admission.get("podSetAssignments")
    if (
        not isinstance(podsets, list)
        or len(podsets) != 1
        or not isinstance(assignments, list)
        or len(assignments) != 1
    ):
        raise ValueError("output-check Workload must contain one admitted PodSet")
    podset = podsets[0]
    assignment = assignments[0]
    if (
        not isinstance(podset, dict)
        or podset.get("name") != "main"
        or podset.get("count") != 1
        or not isinstance(assignment, dict)
        or assignment.get("name") != "main"
        or assignment.get("count") != 1
    ):
        raise ValueError("output-check Workload main PodSet identity or count drifted")
    _only_keys(
        podset,
        {"name", "count", "template", "topologyRequest"},
        "Workload PodSet",
    )
    if podset.get("topologyRequest") != WORKLOAD_TOPOLOGY_REQUEST:
        raise ValueError("output-check Workload topology request drifted")
    _only_keys(
        assignment,
        {"name", "count", "flavors", "resourceUsage"},
        "Workload PodSet assignment",
    )
    template = podset.get("template")
    if not isinstance(template, dict):
        raise ValueError("output-check Workload PodSet template is malformed")
    _only_keys(template, {"metadata", "spec"}, "Workload PodSet template")
    metadata = template.get("metadata")
    expected_metadata = package.job["spec"]["template"]["metadata"]
    if not isinstance(metadata, dict):
        raise ValueError("output-check Workload PodSet metadata is malformed")
    _only_keys(
        metadata,
        {"annotations", "creationTimestamp", "labels"},
        "Workload PodSet metadata",
    )
    if metadata.get("creationTimestamp") is not None:
        raise ValueError("output-check Workload PodSet timestamp is not a server default")
    if metadata.get("annotations") != expected_metadata["annotations"] or metadata.get(
        "labels"
    ) != {
        **expected_metadata["labels"],
        "batch.kubernetes.io/job-name": job["metadata"]["name"],
    }:
        raise ValueError("output-check Workload PodSet metadata drifted")
    pod_spec = deepcopy(template.get("spec"))
    if not isinstance(pod_spec, dict) or pod_spec.get("priority") not in (
        None,
        EFFECTIVE_C1_PRIORITY,
    ):
        raise ValueError("output-check Workload PodSet priority drifted")
    pod_spec["priority"] = EFFECTIVE_C1_PRIORITY
    _validate_pod_spec(
        pod_spec,
        package.job["spec"]["template"]["spec"],
        scheduled=False,
    )
    resource_usage = assignment.get("resourceUsage")
    flavors = assignment.get("flavors")
    if not isinstance(resource_usage, dict) or not isinstance(flavors, dict):
        raise ValueError("output-check Workload admission resources are malformed")
    if "nvidia.com/gpu" in resource_usage or "nvidia.com/gpu" in flavors:
        raise ValueError("output-check Workload must remain effectively zero-GPU")
    return "main"


def _validate_admitted_workload(
    package: SfsOutputJobPackage,
    job: dict,
    workloads: dict,
) -> dict:
    if workloads.get("kind") != "List" or not isinstance(workloads.get("items"), list):
        raise ValueError("output-check Workload inventory is incomplete")
    expected_owner = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "name": job["metadata"]["name"],
        "uid": job["metadata"]["uid"],
        "controller": True,
        "blockOwnerDeletion": True,
    }
    owned = []
    for workload in workloads["items"]:
        if not isinstance(workload, dict):
            raise ValueError("output-check Workload inventory contains a malformed item")
        metadata = workload.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError("output-check Workload inventory contains malformed metadata")
        references = metadata.get("ownerReferences", [])
        if not isinstance(references, list):
            raise ValueError("output-check Workload ownership is malformed")
        if any(
            isinstance(reference, dict)
            and reference.get("apiVersion") == "batch/v1"
            and reference.get("kind") == "Job"
            and reference.get("name") == expected_owner["name"]
            for reference in references
        ):
            owned.append(workload)
    if len(owned) != 1:
        raise ValueError("exactly one Kueue Workload must be bound to the output-check Job")
    workload = owned[0]
    metadata = workload["metadata"]
    if workload.get("apiVersion") != "kueue.x-k8s.io/v1beta2" or workload.get("kind") != "Workload":
        raise ValueError("output-check admission object is not one Kueue Workload")
    if (
        metadata.get("namespace") != NAMESPACE
        or not isinstance(metadata.get("name"), str)
        or not metadata["name"]
        or _KUBERNETES_UID.fullmatch(metadata.get("uid", "")) is None
        or metadata.get("ownerReferences") != [expected_owner]
        or not isinstance(metadata.get("labels"), dict)
        or metadata["labels"].get("kueue.x-k8s.io/job-uid") != expected_owner["uid"]
    ):
        raise ValueError("output-check Workload identity or Job UID binding drifted")
    spec = workload.get("spec")
    if not isinstance(spec, dict):
        raise ValueError("output-check Workload spec is malformed")
    _only_keys(
        spec,
        {
            "active",
            "podSets",
            "priority",
            "priorityClassRef",
            "priorityClassName",
            "priorityClassSource",
            "queueName",
        },
        "Workload spec",
    )
    if (
        spec.get("queueName") != QUEUE
        or spec.get("priority") != EFFECTIVE_C1_PRIORITY
        or spec.get("priorityClassRef") != WORKLOAD_PRIORITY_CLASS_REF
        or spec.get("active", True) is not True
    ):
        raise ValueError("output-check Workload queue or effective priority drifted")
    # Current Fleet Kueue v1beta2 Workloads bind q1 through priorityClassRef.
    # The retired scalar class/source fields must remain absent, while numeric
    # priority proves the effective policy that Kueue admitted.
    if spec.get("priorityClassName") is not None or spec.get("priorityClassSource") is not None:
        raise ValueError("output-check Workload priority shape differs from live policy")
    status = workload.get("status")
    if not isinstance(status, dict) or not isinstance(status.get("admission"), dict):
        raise ValueError("output-check Workload has no live Kueue admission")
    admission = status["admission"]
    if (
        admission.get("clusterQueue") != "training-cq"
        or not isinstance(admission.get("podSetAssignments"), list)
        or not admission["podSetAssignments"]
    ):
        raise ValueError("output-check Workload has no complete Kueue admission")
    conditions = status.get("conditions")
    if not isinstance(conditions, list) or any(
        not isinstance(condition, dict) for condition in conditions
    ):
        raise ValueError("output-check Workload conditions are malformed")
    admitted = [condition for condition in conditions if condition.get("type") == "Admitted"]
    finished = [condition for condition in conditions if condition.get("type") == "Finished"]
    evicted = [condition for condition in conditions if condition.get("type") == "Evicted"]
    if (
        len(admitted) != 1
        or admitted[0].get("status") != "True"
        or len(finished) != 1
        or finished[0].get("status") != "True"
        or finished[0].get("reason") != "Succeeded"
    ):
        raise ValueError("output-check Workload is not both admitted and terminally finished")
    if any(condition.get("status") == "True" for condition in evicted):
        raise ValueError("output-check Workload was evicted")
    podset = _validate_workload_podset(package, job, spec, admission)
    return {
        "name": metadata.get("name"),
        "uid": metadata["uid"],
        "priority": spec["priority"],
        "local_queue": spec["queueName"],
        "cluster_queue": admission["clusterQueue"],
        "podset": podset,
    }


def validate_completed_sfs_output_job(
    package: SfsOutputJobPackage,
    job: dict,
    workloads: dict,
    pods: dict,
    service_account: dict,
) -> str:
    """Bind one successful Pod to the exact created Job before reading its logs."""
    admitted_workload = _validate_admitted_workload(package, job, workloads)
    validate_sfs_output_job_response(
        job,
        package,
        require_uid=True,
        admitted=True,
        admitted_workload=admitted_workload,
    )
    conditions = job.get("status", {}).get("conditions", [])
    if not any(
        item.get("type") == "Complete" and item.get("status") == "True"
        for item in conditions
        if isinstance(item, dict)
    ) or any(
        item.get("type") == "Failed" and item.get("status") == "True"
        for item in conditions
        if isinstance(item, dict)
    ):
        raise ValueError("output-check Job is not terminally successful")
    if pods.get("kind") != "List" or not isinstance(pods.get("items"), list):
        raise ValueError("output-check Pod inventory is incomplete")
    if len(pods["items"]) != 1:
        raise ValueError("output-check Job does not own exactly one Pod")
    pod = pods["items"][0]
    metadata = pod.get("metadata", {})
    if _KUBERNETES_UID.fullmatch(metadata.get("uid", "")) is None:
        raise ValueError("output-check Pod is missing its immutable UID")
    owner_references = metadata.get("ownerReferences", [])
    if owner_references != [
        {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "name": job["metadata"]["name"],
            "uid": job["metadata"]["uid"],
            "controller": True,
            "blockOwnerDeletion": True,
        }
    ]:
        raise ValueError("output-check Pod ownership differs from the exact Job UID")
    _validate_pod_spec(
        pod.get("spec", {}),
        package.job["spec"]["template"]["spec"],
        scheduled=True,
        service_account=service_account,
    )
    live_template_metadata = job["spec"]["template"]["metadata"]
    if metadata.get("annotations") != live_template_metadata["annotations"]:
        raise ValueError("output-check Pod annotations differ from the admitted Job")
    labels = metadata.get("labels")
    expected_labels = live_template_metadata["labels"]
    if not isinstance(labels, dict):
        raise ValueError("output-check Pod labels differ from the reviewed queue binding")
    region = labels.get(_TOPOLOGY_REGION_LABEL)
    if region is not None and (not isinstance(region, str) or not region):
        raise ValueError("output-check Pod topology region label is invalid")
    normalized_labels = deepcopy(labels)
    normalized_labels.pop(_TOPOLOGY_REGION_LABEL, None)
    if normalized_labels != expected_labels:
        raise ValueError("output-check Pod labels differ from the reviewed queue binding")
    statuses = pod.get("status", {}).get("containerStatuses", [])
    if pod.get("status", {}).get("phase") != "Succeeded" or len(statuses) != 1:
        raise ValueError("output-check Pod is not terminally successful")
    terminated = statuses[0].get("state", {}).get("terminated", {})
    if statuses[0].get("restartCount") != 0 or terminated.get("exitCode") != 0:
        raise ValueError("output-check container restarted or exited nonzero")
    image_digest = package.request["image"].rsplit("@", 1)[-1]
    if not statuses[0].get("imageID", "").endswith("@" + image_digest):
        raise ValueError("output-check Pod image identity differs from the prepared request")
    name = metadata.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("output-check Pod name is missing")
    return name


def collect_sfs_output_receipt(
    package: SfsOutputJobPackage,
    job: dict,
    workloads: dict,
    pods: dict,
    service_account: dict,
    logs: str,
    *,
    now: float | None = None,
) -> dict:
    validate_completed_sfs_output_job(package, job, workloads, pods, service_account)
    lines = logs.splitlines()
    if len(lines) != 1 or not lines[0].startswith(LOG_PREFIX):
        raise ValueError("output-check logs do not contain one sanitized receipt")
    try:
        receipt = json.loads(lines[0].removeprefix(LOG_PREFIX))
    except json.JSONDecodeError as exc:
        raise ValueError("output-check receipt is invalid JSON") from exc
    return validate_output_absence_receipt(
        receipt,
        package.plan,
        package.request,
        now=now,
    )
