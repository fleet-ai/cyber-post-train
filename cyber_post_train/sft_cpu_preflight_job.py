"""Tracked create-once zero-GPU Job for the native dense-SFT CPU preflight."""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
import re
from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .gpu_capacity import _effective_pod_quantity
from .jobs import FAILURE_ALERT_ANNOTATION, FAILURE_ALERT_OFF, digest, validate_request
from .sfs_output_job import (
    _KUBERNETES_UID,
    CPU_NODE_SELECTOR,
    EFFECTIVE_C1_PRIORITY,
    NAMESPACE,
    OWNER_LABEL,
    QUEUE,
    QUEUE_LABEL,
    QUEUE_PRIORITY,
    QUEUE_PRIORITY_LABEL,
    ROLE_LABEL,
    _contains,
    _normalize_job_serialization,
    _validate_admitted_workload,
    _validate_pod_spec,
    _validate_server_job_surface,
)
from .sft_cpu_preflight_driver import (
    BUNDLE_SCHEMA,
    CHUNK_PREFIX,
    ENV_BUNDLE_SHA256,
    ENV_DRIVER_SHA256,
    ENV_RUN_DIR,
    ENV_RUN_NAME,
    ENVELOPE_SCHEMA,
    LOG_PREFIX,
    MAX_CHUNK_BYTES,
    MAX_CHUNKS,
    PREFLIGHT_SCHEMA,
)

JOB_ROLE = "sft-cpu-preflight"
JOB_SUFFIX = "-pre-a{attempt:02d}"
MAX_ATTEMPT = 99
PLAN_ANNOTATION = "cyber-post-train.fleet.ai/plan-sha256"
REQUEST_ANNOTATION = "cyber-post-train.fleet.ai/request-sha256"
OUTPUT_ANNOTATION = "cyber-post-train.fleet.ai/training-output"
BUNDLE_ANNOTATION = "cyber-post-train.fleet.ai/preflight-bundle-sha256"
DRIVER_ANNOTATION = "cyber-post-train.fleet.ai/preflight-driver-sha256"
SOURCE_COMMIT_ANNOTATION = "cyber-post-train.fleet.ai/source-commit"
SFT_SCHEMAS = {"cyber_sft_runtime_v2", "cyber_sft_runtime_dense_v1"}
_SOURCE_ROOT = Path(__file__).resolve().parents[1]
_DRIVER_PATH = Path(__file__).with_name("sft_cpu_preflight_driver.py")
_GIT_SHA = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class SftCpuPreflightJobPackage:
    job: dict[str, Any]
    plan: dict[str, Any]
    request: dict[str, Any]
    prepared: dict[str, Any]
    prepared_directory: Path
    source_commit: str
    attempt: int


def job_name(request: dict, attempt: int) -> str:
    if type(attempt) is not int or not 1 <= attempt <= MAX_ATTEMPT:
        raise ValueError(f"SFT CPU preflight attempt must be between 1 and {MAX_ATTEMPT}")
    value = request.get("name", "") + JOB_SUFFIX.format(attempt=attempt)
    if re.fullmatch(r"[a-z0-9](?:[-a-z0-9]{0,61}[a-z0-9])?", value) is None:
        raise ValueError("prepared run name is too long for the CPU-preflight Job suffix")
    return value


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    try:
        text = raw.decode()
    except UnicodeDecodeError:
        raise ValueError(f"preflight source is not UTF-8: {path.name}") from None
    if text.encode() != raw:
        raise ValueError(f"preflight source is not byte-stable: {path.name}")
    return text


def _source_files() -> dict[str, str]:
    files: dict[str, str] = {}
    for package in ("cyber_post_train", "training"):
        root = _SOURCE_ROOT / package
        for path in sorted(root.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            name = "src/" + path.relative_to(_SOURCE_ROOT).as_posix()
            files[name] = _read_text(path)
    if (
        not files
        or "src/training/sft.py" not in files
        or "src/training/sft_runtime.py" not in files
    ):
        raise ValueError("dense-SFT CPU preflight source inventory is incomplete")
    return files


def _driver() -> tuple[str, str]:
    source = _read_text(_DRIVER_PATH)
    return source, hashlib.sha256(source.encode()).hexdigest()


def _prepared_files(directory: Path) -> tuple[dict, dict, dict, dict[str, str]]:
    names = ("plan.json", "request.json", "PREPARED.json")
    values: dict[str, dict] = {}
    texts: dict[str, str] = {}
    for name in names:
        text = _read_text(directory / name)
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"prepared {name} is not valid JSON") from exc
        if not isinstance(value, dict):
            raise ValueError(f"prepared {name} is not an object")
        values[name] = value
        texts["prepared/" + name] = text
    plan = values["plan.json"]
    request = values["request.json"]
    prepared = values["PREPARED.json"]
    expected = {
        "schema": "cyber_post_train_prepared_v1",
        "gate_version": 2,
        "plan_sha256": digest(plan),
        "request_sha256": digest(request),
    }
    if prepared != expected:
        raise ValueError("prepared dense-SFT request predates or differs from the current gate")
    return plan, request, prepared, texts


def _bundle(
    directory: Path,
    *,
    source_commit: str,
) -> tuple[dict, dict, dict, bytes, str, str]:
    if _GIT_SHA.fullmatch(source_commit) is None:
        raise ValueError("dense-SFT CPU preflight source commit must be a full Git SHA")
    plan, request, prepared, prepared_text = _prepared_files(directory)
    validate_request(request)
    if plan.get("schema") not in SFT_SCHEMAS:
        raise ValueError("dense-SFT CPU preflight is restricted to SFT plans")
    if (
        request.get("name") != plan.get("run_name")
        or request.get("title") != plan.get("run_name")
        or request.get("run_dir") != plan.get("output_root")
        or request.get("failureAlerts") is not False
        or request.get("priority_class") != "c1"
    ):
        raise ValueError("prepared dense-SFT run identity or alert/priority policy drifted")
    source, driver_sha256 = _driver()
    files = {**_source_files(), **prepared_text}
    inventory = {name: hashlib.sha256(text.encode()).hexdigest() for name, text in files.items()}
    manifest = {
        "schema": BUNDLE_SCHEMA,
        "source_commit": source_commit,
        "run_name": request["name"],
        "training_output_root": request["run_dir"],
        "image": request["image"],
        "driver_sha256": driver_sha256,
        "plan_sha256": digest(plan),
        "request_sha256": digest(request),
        "files": inventory,
    }
    manifest["sha256"] = digest(manifest)
    raw = json.dumps(
        {"manifest": manifest, "files": files},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    blob = gzip.compress(raw, compresslevel=9, mtime=0)
    encoded = base64.b64encode(blob).decode()
    chunks = [
        encoded[index : index + MAX_CHUNK_BYTES]
        for index in range(0, len(encoded), MAX_CHUNK_BYTES)
    ]
    if not 1 <= len(chunks) <= MAX_CHUNKS:
        raise ValueError("dense-SFT CPU preflight bundle exceeds the Kubernetes chunk bound")
    environment = {
        ENV_BUNDLE_SHA256: hashlib.sha256(blob).hexdigest(),
        ENV_DRIVER_SHA256: driver_sha256,
        ENV_RUN_NAME: request["name"],
        ENV_RUN_DIR: request["run_dir"],
        **{f"{CHUNK_PREFIX}{index:03d}": value for index, value in enumerate(chunks)},
    }
    return plan, request, prepared, blob, source, environment


def _render(
    directory: Path,
    *,
    source_commit: str,
    attempt: int,
) -> tuple[dict, dict, dict, dict]:
    plan, request, prepared, blob, source, environment = _bundle(
        directory, source_commit=source_commit
    )
    name = job_name(request, attempt)
    driver_sha256 = environment[ENV_DRIVER_SHA256]
    annotations = {
        FAILURE_ALERT_ANNOTATION: FAILURE_ALERT_OFF,
        PLAN_ANNOTATION: digest(plan),
        REQUEST_ANNOTATION: digest(request),
        OUTPUT_ANNOTATION: request["run_dir"],
        BUNDLE_ANNOTATION: hashlib.sha256(blob).hexdigest(),
        DRIVER_ANNOTATION: driver_sha256,
        SOURCE_COMMIT_ANNOTATION: source_commit,
    }
    pod_environment = {
        "CUDA_VISIBLE_DEVICES": "",
        "NVIDIA_VISIBLE_DEVICES": "none",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "WANDB_MODE": "disabled",
        "PYTHONDONTWRITEBYTECODE": "1",
        **environment,
    }
    job = render_cpu_preflight_job(
        name=name,
        image=request["image"],
        role=JOB_ROLE,
        command=["python", "-u", "-c", source],
        environment=pod_environment,
        annotations=annotations,
        resources={
            "requests": {
                "cpu": "4",
                "memory": "32Gi",
                "ephemeral-storage": "2Gi",
            },
            "limits": {
                "cpu": "8",
                "memory": "48Gi",
                "ephemeral-storage": "4Gi",
            },
        },
        image_pull_secrets=request.get("image_pull_secrets", []),
        read_only_sfs=True,
    )
    return job, plan, request, prepared


def render_cpu_preflight_job(
    *,
    name: str,
    image: str,
    role: str,
    command: list[str],
    environment: dict[str, str],
    annotations: dict[str, str],
    resources: dict[str, dict[str, str]],
    image_pull_secrets: list[str] | tuple[str, ...] = (),
    read_only_sfs: bool = False,
    termination_message_path: str | None = None,
) -> dict[str, Any]:
    """Render the repository's one reviewed Kueue-managed zero-GPU Job shape.

    Callers remain responsible for rebuilding and comparing their immutable
    driver, bundle, and plan bytes.  This helper owns only the shared cluster
    safety surface used by both SFT and Miles CPU preflights.
    """
    if re.fullmatch(r"[a-z0-9](?:[-a-z0-9]{0,61}[a-z0-9])?", name) is None:
        raise ValueError("CPU-preflight Job name is not a DNS label")
    if re.fullmatch(r"[^\s]+@sha256:[a-f0-9]{64}", image) is None:
        raise ValueError("CPU-preflight image is not immutable")
    if re.fullmatch(r"[a-z0-9](?:[-a-z0-9]{0,61}[a-z0-9])?", role) is None:
        raise ValueError("CPU-preflight role is not a DNS label")
    if (
        not command
        or any(not isinstance(value, str) or not value for value in command)
        or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in environment.items()
        )
        or "JOB_NAME" in environment
        or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in annotations.items()
        )
        or annotations.get(FAILURE_ALERT_ANNOTATION, FAILURE_ALERT_OFF) != FAILURE_ALERT_OFF
        or type(read_only_sfs) is not bool
        or any(
            not isinstance(value, str)
            or re.fullmatch(r"[a-z0-9](?:[-a-z0-9.]{0,251}[a-z0-9])?", value) is None
            for value in image_pull_secrets
        )
    ):
        raise ValueError("CPU-preflight command, environment, or policy is invalid")
    if set(resources) != {"requests", "limits"} or any(
        not isinstance(values, dict)
        or set(values) != {"cpu", "memory", "ephemeral-storage"}
        or any(not isinstance(value, str) or not value for value in values.values())
        or "nvidia.com/gpu" in values
        for values in resources.values()
    ):
        raise ValueError("CPU-preflight resources are invalid")
    if termination_message_path is not None and (
        not isinstance(termination_message_path, str)
        or not termination_message_path.startswith("/tmp/")
        or ".." in Path(termination_message_path).parts
    ):
        raise ValueError("CPU-preflight termination message path is unsafe")

    annotations = {**annotations, FAILURE_ALERT_ANNOTATION: FAILURE_ALERT_OFF}
    labels = {
        OWNER_LABEL: "chris",
        ROLE_LABEL: role,
        QUEUE_LABEL: QUEUE,
        QUEUE_PRIORITY_LABEL: QUEUE_PRIORITY,
    }
    volume_mounts = [{"name": "tmp", "mountPath": "/tmp"}]
    volumes = [{"name": "tmp", "emptyDir": {"sizeLimit": "1Gi"}}]
    if read_only_sfs:
        volume_mounts.insert(0, {"name": "sfs", "mountPath": "/mnt/sfs", "readOnly": True})
        volumes.insert(
            0,
            {
                "name": "sfs",
                "persistentVolumeClaim": {"claimName": "sfs-shared", "readOnly": True},
            },
        )
    container = {
        "name": "preflight",
        "image": image,
        "imagePullPolicy": "IfNotPresent",
        "command": command,
        "env": [{"name": "JOB_NAME", "value": name}]
        + [{"name": key, "value": value} for key, value in environment.items()],
        "resources": deepcopy(resources),
        "securityContext": {
            "allowPrivilegeEscalation": False,
            "capabilities": {"drop": ["ALL"]},
            "privileged": False,
            "readOnlyRootFilesystem": True,
            "runAsNonRoot": True,
        },
        "volumeMounts": volume_mounts,
    }
    if termination_message_path is not None:
        container.update(
            {
                "terminationMessagePath": termination_message_path,
                "terminationMessagePolicy": "File",
            }
        )
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": name,
            "namespace": NAMESPACE,
            "annotations": deepcopy(annotations),
            "labels": deepcopy(labels),
        },
        "spec": {
            "activeDeadlineSeconds": 1800,
            "backoffLimit": 0,
            "completionMode": "NonIndexed",
            "completions": 1,
            "parallelism": 1,
            "suspend": True,
            "template": {
                "metadata": {
                    "annotations": deepcopy(annotations),
                    "labels": deepcopy(labels),
                },
                "spec": {
                    "activeDeadlineSeconds": 1800,
                    "automountServiceAccountToken": False,
                    "containers": [
                        {
                            **container,
                        }
                    ],
                    "hostIPC": False,
                    "hostNetwork": False,
                    "hostPID": False,
                    "imagePullSecrets": [{"name": value} for value in image_pull_secrets],
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
                    "volumes": volumes,
                },
            },
        },
    }


def build_sft_cpu_preflight_job(
    directory: Path,
    *,
    source_commit: str,
    attempt: int,
) -> SftCpuPreflightJobPackage:
    job, plan, request, prepared = _render(
        Path(directory), source_commit=source_commit, attempt=attempt
    )
    package = SftCpuPreflightJobPackage(
        job=job,
        plan=plan,
        request=request,
        prepared=prepared,
        prepared_directory=Path(directory),
        source_commit=source_commit,
        attempt=attempt,
    )
    validate_sft_cpu_preflight_job_package(package)
    return package


def validate_sft_cpu_preflight_job_package(package: SftCpuPreflightJobPackage) -> dict:
    if not isinstance(package, SftCpuPreflightJobPackage):
        raise ValueError("dense-SFT CPU preflight requires one source-bound package")
    expected, plan, request, prepared = _render(
        package.prepared_directory,
        source_commit=package.source_commit,
        attempt=package.attempt,
    )
    if (
        package.job != expected
        or package.plan != plan
        or package.request != request
        or package.prepared != prepared
    ):
        raise ValueError("dense-SFT CPU-preflight Job differs from current source/prepared bytes")
    return {
        "name": expected["metadata"]["name"],
        "plan_sha256": digest(plan),
        "request_sha256": digest(request),
        "manifest_sha256": digest(expected),
        "bundle_sha256": expected["metadata"]["annotations"][BUNDLE_ANNOTATION],
        "driver_sha256": expected["metadata"]["annotations"][DRIVER_ANNOTATION],
        "source_commit": package.source_commit,
        "gpus": 0,
    }


def validate_sft_cpu_preflight_job_response(
    actual: dict,
    package: SftCpuPreflightJobPackage,
    *,
    require_uid: bool,
    admitted: bool = False,
    admitted_workload: dict | None = None,
) -> dict:
    proof = validate_sft_cpu_preflight_job_package(package)
    validate_cpu_preflight_job_response(
        actual,
        package.job,
        require_uid=require_uid,
        admitted=admitted,
        admitted_workload=admitted_workload,
    )
    return proof


def validate_cpu_preflight_job_response(
    actual: dict,
    expected_job: dict,
    *,
    require_uid: bool,
    admitted: bool = False,
    admitted_workload: dict | None = None,
) -> dict[str, Any]:
    """Validate one server-returned Job against an exact rebuilt CPU manifest."""
    expected = deepcopy(expected_job)
    if admitted:
        expected["spec"]["suspend"] = False
        if admitted_workload is None:
            raise ValueError("admitted CPU-preflight Job requires exact Workload evidence")
    elif admitted_workload is not None:
        raise ValueError("pre-admission CPU-preflight Job cannot carry Workload evidence")
    normalized = _normalize_job_serialization(actual, expected)
    if not _contains(normalized, expected):
        raise ValueError("server CPU-preflight Job differs from the reviewed manifest")
    _validate_server_job_surface(
        normalized,
        expected,
        admitted_workload=admitted_workload,
    )
    uid = normalized.get("metadata", {}).get("uid", "")
    if require_uid and _KUBERNETES_UID.fullmatch(uid) is None:
        raise ValueError("created CPU-preflight Job omitted its immutable UID")
    pod_spec = normalized.get("spec", {}).get("template", {}).get("spec", {})
    if (
        _effective_pod_quantity(pod_spec, "requests") != 0
        or _effective_pod_quantity(pod_spec, "limits") != 0
    ):
        raise ValueError("dense-SFT CPU preflight must remain effectively zero-GPU")
    return {
        "name": expected_job["metadata"]["name"],
        "manifest_sha256": digest(expected_job),
        "job_uid": uid,
        "gpus": 0,
    }


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


def validate_sft_cpu_preflight_job_node_fit(
    package: SftCpuPreflightJobPackage, inventory: dict
) -> dict:
    validate_sft_cpu_preflight_job_package(package)
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
    fitting = [item for item in eligible if item[1] >= 4000 and item[2] >= 32 * 1024**3]
    if not fitting:
        raise ValueError("dense-SFT CPU-preflight Job cannot fit a Ready eligible CPU node")
    return {"eligible_nodes": len(eligible), "fitting_nodes": len(fitting)}


def validate_completed_sft_cpu_preflight_job(
    package: SftCpuPreflightJobPackage,
    job: dict,
    workloads: dict,
    pods: dict,
    service_account: dict,
) -> tuple[str, dict]:
    return validate_completed_cpu_preflight_job(
        package.job,
        job,
        workloads,
        pods,
        service_account,
    )


def validate_completed_cpu_preflight_job(
    expected_job: dict,
    job: dict,
    workloads: dict,
    pods: dict,
    service_account: dict,
) -> tuple[str, dict]:
    """Bind one successful CPU-only Job, Workload, and Pod to exact bytes."""
    surface = SimpleNamespace(job=expected_job)
    admitted_workload = _validate_admitted_workload(surface, job, workloads)
    owned = [
        item
        for item in workloads.get("items", [])
        if item.get("metadata", {}).get("uid") == admitted_workload["uid"]
    ]
    if len(owned) != 1:
        raise ValueError("dense-SFT CPU-preflight Workload identity is ambiguous")
    assignments = owned[0]["status"]["admission"]["podSetAssignments"]
    expected_resources = expected_job["spec"]["template"]["spec"]["containers"][0]["resources"][
        "requests"
    ]
    if (
        len(assignments) != 1
        or assignments[0].get("resourceUsage") != expected_resources
        or set(assignments[0].get("flavors", {})) != set(expected_resources)
        or any(
            not isinstance(value, str) or not value
            for value in assignments[0].get("flavors", {}).values()
        )
    ):
        raise ValueError("dense-SFT CPU-preflight admitted resources drifted")
    validate_cpu_preflight_job_response(
        job,
        expected_job,
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
        raise ValueError("dense-SFT CPU-preflight Job is not terminally successful")
    if pods.get("kind") != "List" or not isinstance(pods.get("items"), list):
        raise ValueError("dense-SFT CPU-preflight Pod inventory is incomplete")
    if len(pods["items"]) != 1:
        raise ValueError("dense-SFT CPU-preflight Job does not own exactly one Pod")
    pod = pods["items"][0]
    metadata = pod.get("metadata", {})
    if _KUBERNETES_UID.fullmatch(metadata.get("uid", "")) is None:
        raise ValueError("dense-SFT CPU-preflight Pod omitted its immutable UID")
    if metadata.get("ownerReferences") != [
        {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "name": job["metadata"]["name"],
            "uid": job["metadata"]["uid"],
            "controller": True,
            "blockOwnerDeletion": True,
        }
    ]:
        raise ValueError("dense-SFT CPU-preflight Pod ownership differs from the Job UID")
    _validate_pod_spec(
        pod.get("spec", {}),
        expected_job["spec"]["template"]["spec"],
        scheduled=True,
        service_account=service_account,
    )
    live_template = job["spec"]["template"]["metadata"]
    if metadata.get("annotations") != live_template["annotations"]:
        raise ValueError("dense-SFT CPU-preflight Pod annotations drifted")
    labels = deepcopy(metadata.get("labels"))
    if not isinstance(labels, dict):
        raise ValueError("dense-SFT CPU-preflight Pod labels are malformed")
    labels.pop("topology.kubernetes.io/region", None)
    if labels != live_template["labels"]:
        raise ValueError("dense-SFT CPU-preflight Pod labels drifted")
    statuses = pod.get("status", {}).get("containerStatuses", [])
    if pod.get("status", {}).get("phase") != "Succeeded" or len(statuses) != 1:
        raise ValueError("dense-SFT CPU-preflight Pod is not terminally successful")
    terminated = statuses[0].get("state", {}).get("terminated", {})
    if statuses[0].get("restartCount") != 0 or terminated.get("exitCode") != 0:
        raise ValueError("dense-SFT CPU-preflight container restarted or exited nonzero")
    image_digest = expected_job["spec"]["template"]["spec"]["containers"][0]["image"].rsplit(
        "@", 1
    )[-1]
    image_id = statuses[0].get("imageID", "")
    match = re.search(r"(?:@|://)(sha256:[0-9a-f]{64})$", image_id)
    if match is None or match.group(1) != image_digest:
        raise ValueError("dense-SFT CPU-preflight image differs from the prepared request")
    name = metadata.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("dense-SFT CPU-preflight Pod name is missing")
    return name, admitted_workload


def collect_sft_cpu_preflight_receipt(
    package: SftCpuPreflightJobPackage,
    job: dict,
    workloads: dict,
    pods: dict,
    service_account: dict,
    logs: str,
) -> dict:
    validate_completed_sft_cpu_preflight_job(package, job, workloads, pods, service_account)
    lines = logs.splitlines()
    if len(lines) != 1 or not lines[0].startswith(LOG_PREFIX):
        raise ValueError("dense-SFT CPU-preflight logs do not contain one sanitized receipt")
    try:
        envelope = json.loads(lines[0].removeprefix(LOG_PREFIX))
    except json.JSONDecodeError as exc:
        raise ValueError("dense-SFT CPU-preflight receipt is invalid JSON") from exc
    proof = validate_sft_cpu_preflight_job_package(package)
    if (
        not isinstance(envelope, dict)
        or envelope.get("schema") != ENVELOPE_SCHEMA
        or envelope.get("status") != "passed"
        or envelope.get("gpus") != 0
        or envelope.get("job_name") != job["metadata"]["name"]
        or envelope.get("bundle_sha256") != proof["bundle_sha256"]
        or envelope.get("driver_sha256") != proof["driver_sha256"]
        or envelope.get("plan_sha256") != proof["plan_sha256"]
        or envelope.get("request_sha256") != proof["request_sha256"]
        or not isinstance(envelope.get("observed_at_unix"), (int, float))
    ):
        raise ValueError("dense-SFT CPU-preflight observation identity drifted")
    receipt = envelope.get("preflight")
    if not isinstance(receipt, dict):
        raise ValueError("dense-SFT CPU-preflight native receipt is missing")
    receipt_body = {key: value for key, value in receipt.items() if key != "sha256"}
    if (
        receipt.get("sha256") != digest(receipt_body)
        or receipt_body.get("schema") != PREFLIGHT_SCHEMA
        or receipt_body.get("status") != "passed"
        or receipt_body.get("gpus") != 0
        or receipt_body.get("plan_sha256") != proof["plan_sha256"]
        or receipt_body.get("request_sha256") != proof["request_sha256"]
    ):
        raise ValueError("dense-SFT CPU-preflight native receipt failed its binding")
    return receipt
