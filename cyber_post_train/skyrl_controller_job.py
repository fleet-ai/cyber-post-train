"""Typed, zero-GPU bootstrap Job for the prod10 SkyRL controller.

The Fleet Jobs API accepts only GPU RayJobs, so it cannot host the SFS-local
create-once coordinator used by the reviewed prod9 rail. This module packages
one ordinary ``batch/v1`` Job behind a small typed API. The caller interface
requires an API-server dry-run and an exact create response; it deliberately
has no ``kubectl`` implementation.

This first bootstrap is intentionally non-submitting. It proves the pinned
image, runtime identity, SFS mounts and durable receipt hand-off without a
Kubernetes service-account token or a Fleet credential. The target POST stays
closed until the platform supplies immutable-ID cleanup and a metadata-only
capacity receipt.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import os
import re
import stat
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from .gpu_capacity import _effective_pod_quantity
from .jobs import FAILURE_ALERT_ANNOTATION, FAILURE_ALERT_OFF, JobsError, digest

NAMESPACE = "fleet-train-jobs"
PROD_CONTEXT = "nebius-mk8s-fleetai-training-e04zw4ye1k7wczqdw6"
IMAGE = (
    "661864827319.dkr.ecr.us-east-1.amazonaws.com/fleet/skyrl-train@sha256:"
    "ba288751cd227c5be146d28f4a03237545d87d2cbd4c48464945b17fde566ff4"
)
PVC = "sfs-shared"
NAME = "chris-q38-prod10-controller-v1"
ROLE = "skyrl-prod10-controller"
OWNER = "chris"
QUEUE = "training-lq"
QUEUE_PRIORITY = "q1"
POD_PRIORITY = "c1"
EFFECTIVE_C1_PRIORITY = 10_000
RUNTIME_UID = 1000
RUNTIME_GID = 100
CONTROLS_SUBPATH = "jobs/chris-q38-study-corpora-v1/launch-controls"
CONTROLS_MOUNT = "/mnt/sfs/" + CONTROLS_SUBPATH
CREATE_ONCE_ROOT = CONTROLS_MOUNT + "/prod9-create-once-v1"
BOOTSTRAP_ROOT = CONTROLS_MOUNT + "/prod10-controller-bootstrap-v1"
ACTIVE_DEADLINE_SECONDS = 15 * 60
PACKET_SCHEMA = "cyber_skyrl_prod10_controller_packet_v1"
CREATOR_SCHEMA = "cyber_skyrl_prod10_controller_created_v1"
TERMINAL_SCHEMA = "cyber_skyrl_prod10_controller_terminal_v1"

ROLE_LABEL = "cyber-post-train.fleet.ai/role"
OWNER_LABEL = "cyber-post-train.fleet.ai/owner"
QUEUE_LABEL = "kueue.x-k8s.io/queue-name"
QUEUE_PRIORITY_LABEL = "kueue.x-k8s.io/priority-class"
PACKET_ANNOTATION = "cyber-post-train.fleet.ai/controller-packet-sha256"
DRIVER_ANNOTATION = "cyber-post-train.fleet.ai/controller-driver-sha256"
SOURCE_ANNOTATION = "cyber-post-train.fleet.ai/source-commit"

ENV_DRIVER_SHA256 = "CYBER_SKYRL_CONTROLLER_DRIVER_SHA256"
ENV_DRIVER_SOURCE = "CYBER_SKYRL_CONTROLLER_DRIVER_SOURCE_B64"
ENV_PACKET = "CYBER_SKYRL_CONTROLLER_PACKET"
ENV_JOB_UID = "CYBER_SKYRL_CONTROLLER_JOB_UID"
ENV_POD_NAME = "CYBER_SKYRL_CONTROLLER_POD_NAME"
ENV_POD_UID = "CYBER_SKYRL_CONTROLLER_POD_UID"

CPU_NODE_SELECTOR = {
    "kubernetes.io/arch": "amd64",
    "workload": "fleetai-training-ng-cpu",
}
CPU_TOLERATION = {
    "key": "workload",
    "operator": "Equal",
    "value": "fleetai-training-ng-cpu",
    "effect": "NoSchedule",
}

_DRIVER_PATH = Path(__file__).with_name("skyrl_controller_driver.py")
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_KUBERNETES_UID = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}"
)


@dataclass(frozen=True)
class ControllerJobPackage:
    """One reviewed bootstrap packet and its exact Kubernetes Job."""

    packet: dict[str, Any]
    job: dict[str, Any]


class ControllerJobApi(Protocol):
    """Narrow transport required by the create-once caller.

    Implementations are platform-owned. In particular, this repository does
    not silently fall back to a local ``kubectl create`` subprocess.
    """

    def get_job(self, namespace: str, name: str) -> dict[str, Any] | None: ...

    def server_dry_run_job(self, manifest: dict[str, Any]) -> dict[str, Any]: ...

    def create_job_once(self, manifest: dict[str, Any]) -> dict[str, Any]: ...


class ControllerJournal(Protocol):
    """Durable SFS journal used by the one-create caller."""

    path: Path

    def exists(self) -> bool: ...

    def write_once(self, value: dict[str, Any]) -> None: ...

    def append(self, value: dict[str, Any]) -> None: ...

    def read_rows(self) -> list[dict[str, Any]]: ...


class ControllerTerminalApi(Protocol):
    """Scoped read surface used after one exact creator receipt."""

    def terminal_evidence(
        self,
        *,
        job_uid: str,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], str]: ...


def _seal(value: dict[str, Any]) -> dict[str, Any]:
    body = {key: item for key, item in value.items() if key != "sha256"}
    return {**body, "sha256": "sha256:" + digest(body)}


def controller_packet(
    *,
    source_commit: str,
    identity_sha256: str,
    plan_sha256: str,
    request_sha256: str,
    target_manifest_sha256: str,
) -> dict[str, Any]:
    """Seal public prod10 bindings without authorizing a target POST."""

    values = (identity_sha256, plan_sha256, request_sha256, target_manifest_sha256)
    if _COMMIT.fullmatch(source_commit) is None or any(
        _SHA256.fullmatch(value) is None for value in values
    ):
        raise ValueError("prod10 controller source or digest binding is invalid")
    body = {
        "schema": PACKET_SCHEMA,
        "status": "prepared_not_authorized",
        "source_commit": source_commit,
        "identity_sha256": identity_sha256,
        "plan_sha256": plan_sha256,
        "request_sha256": request_sha256,
        "target_manifest_sha256": target_manifest_sha256,
        "target": {
            "name": "chris-q38-rlreward-prod10",
            "nodes": 1,
            "gpus": 8,
            "priority": POD_PRIORITY,
            "queue_priority": QUEUE_PRIORITY,
            "failure_alerts": FAILURE_ALERT_OFF,
        },
        "controller": {
            "name": NAME,
            "context": PROD_CONTEXT,
            "namespace": NAMESPACE,
            "image": IMAGE,
            "runtime_user": {"uid": RUNTIME_UID, "gid": RUNTIME_GID},
            "controls_mount": CONTROLS_MOUNT,
            "create_once_root": CREATE_ONCE_ROOT,
            "bootstrap_root": BOOTSTRAP_ROOT,
        },
        "required_target_receipts": {
            "creator_exact_run_id_and_rayjob_uid": True,
            "observer_exact_uid_release": True,
            "terminal_reward_update_checkpoint": True,
        },
        "platform_gates": {
            "immutable_run_id_and_rayjob_uid_release": False,
            "metadata_only_capacity_receipt": False,
            "dedicated_least_privilege_controller_service_account": False,
        },
        "launch_authorized": False,
        "submitted": False,
    }
    body["controller"]["bootstrap_receipt"] = (
        BOOTSTRAP_ROOT + "/" + plan_sha256.removeprefix("sha256:") + ".json"
    )
    body["controller"]["create_journal"] = (
        BOOTSTRAP_ROOT + "/" + plan_sha256.removeprefix("sha256:") + ".create.jsonl"
    )
    return _seal(body)


def validate_controller_packet(packet: object) -> dict[str, Any]:
    if not isinstance(packet, dict) or packet != _seal(packet):
        raise ValueError("prod10 controller packet digest changed")
    expected = controller_packet(
        source_commit=packet.get("source_commit", ""),
        identity_sha256=packet.get("identity_sha256", ""),
        plan_sha256=packet.get("plan_sha256", ""),
        request_sha256=packet.get("request_sha256", ""),
        target_manifest_sha256=packet.get("target_manifest_sha256", ""),
    )
    if packet != expected:
        raise ValueError("prod10 controller packet contract changed")
    return packet


def _driver() -> tuple[str, str, str]:
    source_bytes = _DRIVER_PATH.read_bytes()
    try:
        source = source_bytes.decode()
    except UnicodeDecodeError:
        raise ValueError("prod10 controller driver must be UTF-8") from None
    if source.encode() != source_bytes:
        raise ValueError("prod10 controller driver text is not byte-stable")
    driver_sha256 = hashlib.sha256(source_bytes).hexdigest()
    return source, driver_sha256, base64.b64encode(source_bytes).decode()


def _annotations(packet: dict[str, Any], driver_sha256: str) -> dict[str, str]:
    return {
        FAILURE_ALERT_ANNOTATION: FAILURE_ALERT_OFF,
        PACKET_ANNOTATION: packet["sha256"],
        DRIVER_ANNOTATION: driver_sha256,
        SOURCE_ANNOTATION: packet["source_commit"],
    }


def _labels() -> dict[str, str]:
    return {
        OWNER_LABEL: OWNER,
        ROLE_LABEL: ROLE,
        QUEUE_LABEL: QUEUE,
        QUEUE_PRIORITY_LABEL: QUEUE_PRIORITY,
    }


def _literal_env(name: str, value: str) -> dict[str, str]:
    return {"name": name, "value": value}


def _render(packet: dict[str, Any]) -> dict[str, Any]:
    packet = validate_controller_packet(packet)
    source, driver_sha256, encoded_source = _driver()
    annotations = _annotations(packet, driver_sha256)
    labels = _labels()
    environment = [
        _literal_env(ENV_DRIVER_SHA256, driver_sha256),
        _literal_env(ENV_DRIVER_SOURCE, encoded_source),
        _literal_env(ENV_PACKET, json.dumps(packet, sort_keys=True, separators=(",", ":"))),
        _literal_env(ENV_JOB_UID, "/controller/job_uid"),
        _literal_env(ENV_POD_NAME, "/controller/pod_name"),
        _literal_env(ENV_POD_UID, "/controller/pod_uid"),
        _literal_env("CUDA_VISIBLE_DEVICES", ""),
        _literal_env("NVIDIA_VISIBLE_DEVICES", "none"),
        _literal_env("PYTHONDONTWRITEBYTECODE", "1"),
        _literal_env("PYTHONUNBUFFERED", "1"),
        _literal_env("WANDB_MODE", "disabled"),
    ]
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": NAME,
            "namespace": NAMESPACE,
            "annotations": copy.deepcopy(annotations),
            "labels": copy.deepcopy(labels),
        },
        "spec": {
            "activeDeadlineSeconds": ACTIVE_DEADLINE_SECONDS,
            "backoffLimit": 0,
            "completionMode": "NonIndexed",
            "completions": 1,
            "parallelism": 1,
            "suspend": True,
            "template": {
                "metadata": {
                    "annotations": copy.deepcopy(annotations),
                    "labels": copy.deepcopy(labels),
                },
                "spec": {
                    "activeDeadlineSeconds": ACTIVE_DEADLINE_SECONDS,
                    "automountServiceAccountToken": False,
                    "containers": [
                        {
                            "name": "controller",
                            "image": IMAGE,
                            "imagePullPolicy": "IfNotPresent",
                            "command": ["python", "-u", "-c", source],
                            "env": environment,
                            "resources": {
                                "requests": {
                                    "cpu": "2",
                                    "memory": "2Gi",
                                    "ephemeral-storage": "1Gi",
                                },
                                "limits": {
                                    "cpu": "4",
                                    "memory": "4Gi",
                                    "ephemeral-storage": "2Gi",
                                },
                            },
                            "securityContext": {
                                "allowPrivilegeEscalation": False,
                                "capabilities": {"drop": ["ALL"]},
                                "privileged": False,
                                "readOnlyRootFilesystem": True,
                                "runAsGroup": RUNTIME_GID,
                                "runAsNonRoot": True,
                                "runAsUser": RUNTIME_UID,
                            },
                            "terminationMessagePath": "/dev/termination-log",
                            "terminationMessagePolicy": "File",
                            "volumeMounts": [
                                {"name": "sfs", "mountPath": "/mnt/sfs", "readOnly": True},
                                {
                                    "name": "controls",
                                    "mountPath": CONTROLS_MOUNT,
                                    "readOnly": False,
                                    "subPath": CONTROLS_SUBPATH,
                                },
                                {
                                    "name": "identity",
                                    "mountPath": "/controller",
                                    "readOnly": True,
                                },
                                {"name": "tmp", "mountPath": "/tmp"},
                            ],
                        }
                    ],
                    "hostIPC": False,
                    "hostNetwork": False,
                    "hostPID": False,
                    "imagePullSecrets": [],
                    "nodeSelector": copy.deepcopy(CPU_NODE_SELECTOR),
                    "priority": EFFECTIVE_C1_PRIORITY,
                    "priorityClassName": POD_PRIORITY,
                    "restartPolicy": "Never",
                    "securityContext": {
                        "fsGroup": RUNTIME_GID,
                        "runAsGroup": RUNTIME_GID,
                        "runAsNonRoot": True,
                        "runAsUser": RUNTIME_UID,
                        "seccompProfile": {"type": "RuntimeDefault"},
                    },
                    "terminationGracePeriodSeconds": 30,
                    "tolerations": [copy.deepcopy(CPU_TOLERATION)],
                    "volumes": [
                        {
                            "name": "sfs",
                            "persistentVolumeClaim": {"claimName": PVC, "readOnly": True},
                        },
                        {
                            "name": "controls",
                            "persistentVolumeClaim": {"claimName": PVC},
                        },
                        {
                            "name": "identity",
                            "downwardAPI": {
                                "items": [
                                    {
                                        "path": "job_uid",
                                        "fieldRef": {
                                            "fieldPath": (
                                                "metadata.labels"
                                                "['batch.kubernetes.io/controller-uid']"
                                            )
                                        },
                                    },
                                    {
                                        "path": "pod_name",
                                        "fieldRef": {"fieldPath": "metadata.name"},
                                    },
                                    {
                                        "path": "pod_uid",
                                        "fieldRef": {"fieldPath": "metadata.uid"},
                                    },
                                ]
                            },
                        },
                        {"name": "tmp", "emptyDir": {"sizeLimit": "2Gi"}},
                    ],
                },
            },
        },
    }


def build_controller_job(packet: dict[str, Any]) -> ControllerJobPackage:
    package = ControllerJobPackage(packet=copy.deepcopy(packet), job=_render(packet))
    validate_controller_job_package(package)
    return package


def _validate_zero_gpu(job: dict[str, Any]) -> None:
    pod = job.get("spec", {}).get("template", {}).get("spec", {})
    containers = [*pod.get("initContainers", []), *pod.get("containers", [])]
    if len(containers) != 1 or any(
        _effective_pod_quantity(pod, field) != 0 for field in ("requests", "limits")
    ):
        raise ValueError("prod10 controller must remain effectively zero-GPU")
    if "nvidia.com/gpu" in json.dumps(job, sort_keys=True):
        raise ValueError("prod10 controller must not request a GPU")


def validate_controller_job_package(package: ControllerJobPackage) -> dict[str, Any]:
    if not isinstance(package, ControllerJobPackage):
        raise ValueError("prod10 controller requires one typed package")
    packet = validate_controller_packet(package.packet)
    expected = _render(packet)
    if package.job != expected:
        raise ValueError("prod10 controller Job differs from its typed renderer")
    _validate_zero_gpu(expected)
    pod = expected["spec"]["template"]["spec"]
    if (
        pod.get("automountServiceAccountToken") is not False
        or "serviceAccountName" in pod
        or any("secret" in volume for volume in pod["volumes"])
    ):
        raise ValueError("prod10 bootstrap must not receive a Kubernetes or data credential")
    return {
        "name": NAME,
        "namespace": NAMESPACE,
        "packet_sha256": packet["sha256"],
        "manifest_sha256": "sha256:" + digest(expected),
        "driver_sha256": expected["metadata"]["annotations"][DRIVER_ANNOTATION],
        "failure_alerts": FAILURE_ALERT_OFF,
        "priority": POD_PRIORITY,
        "queue_priority": QUEUE_PRIORITY,
        "gpus": 0,
        "submitted": False,
    }


def _normalize_server_job(actual: object, expected: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(actual, dict):
        raise ValueError("server prod10 controller Job is not an object")
    value = copy.deepcopy(actual)
    if set(value) - {"apiVersion", "kind", "metadata", "spec", "status"}:
        raise ValueError("server prod10 controller Job added an unreviewed field")
    if value.pop("status", None) not in (None, {}):
        raise ValueError("server prod10 controller Job unexpectedly has live status")
    metadata = value.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("server prod10 controller Job metadata is malformed")
    uid = metadata.pop("uid", None)
    created_at = metadata.pop("creationTimestamp", None)
    generation = metadata.pop("generation", None)
    resource_version = metadata.pop("resourceVersion", None)
    metadata.pop("managedFields", None)
    if uid is not None and _KUBERNETES_UID.fullmatch(uid) is None:
        raise ValueError("server prod10 controller Job UID default changed")
    if created_at is not None and (not isinstance(created_at, str) or not created_at):
        raise ValueError("server prod10 controller Job timestamp default changed")
    if generation is not None and (type(generation) is not int or generation < 1):
        raise ValueError("server prod10 controller Job generation default changed")
    if resource_version is not None and (
        not isinstance(resource_version, str) or not resource_version
    ):
        raise ValueError("server prod10 controller Job resource version default changed")
    generated = {
        "batch.kubernetes.io/controller-uid": uid,
        "batch.kubernetes.io/job-name": NAME,
        "controller-uid": uid,
        "job-name": NAME,
    }
    labels = metadata.get("labels")
    if isinstance(labels, dict):
        for key, item in generated.items():
            if key in labels and labels[key] == item:
                labels.pop(key)
    spec = value.get("spec")
    if not isinstance(spec, dict):
        raise ValueError("server prod10 controller Job spec is malformed")
    selector = spec.pop("selector", None)
    if selector not in (
        None,
        {"matchLabels": {"batch.kubernetes.io/controller-uid": uid}},
    ):
        raise ValueError("server prod10 controller selector changed")
    defaults = {
        "manualSelector": False,
        "podReplacementPolicy": "TerminatingOrFailed",
    }
    for key, item in defaults.items():
        if key in spec and spec.pop(key) != item:
            raise ValueError("server prod10 controller Job default changed")
    template = spec.get("template", {})
    template_metadata = template.get("metadata", {})
    template_metadata.pop("creationTimestamp", None)
    template_labels = template_metadata.get("labels")
    if isinstance(template_labels, dict):
        for key, item in generated.items():
            if key in template_labels and template_labels[key] == item:
                template_labels.pop(key)
    pod = template.get("spec", {})
    pod_defaults = {
        "dnsPolicy": "ClusterFirst",
        "enableServiceLinks": True,
        "preemptionPolicy": "PreemptLowerPriority",
        "schedulerName": "default-scheduler",
        "serviceAccount": "default",
        "serviceAccountName": "default",
    }
    for key, item in pod_defaults.items():
        if key in pod and pod.pop(key) != item:
            raise ValueError("server prod10 controller Pod default changed")
    for key in ("hostIPC", "hostNetwork", "hostPID"):
        if key not in pod and expected["spec"]["template"]["spec"].get(key) is False:
            pod[key] = False
    if (
        expected["spec"]["template"]["spec"].get("imagePullSecrets") == []
        and "imagePullSecrets" not in pod
    ):
        pod["imagePullSecrets"] = []
    containers = pod.get("containers", [])
    if len(containers) == 1:
        environment = containers[0].get("env", [])
        expected_environment = expected["spec"]["template"]["spec"]["containers"][0]["env"]
        if len(environment) == len(expected_environment):
            for actual_entry, expected_entry in zip(environment, expected_environment, strict=True):
                if expected_entry == {
                    "name": "CUDA_VISIBLE_DEVICES",
                    "value": "",
                } and actual_entry == {"name": "CUDA_VISIBLE_DEVICES"}:
                    actual_entry["value"] = ""
    return value


def validate_controller_job_response(
    actual: dict[str, Any],
    package: ControllerJobPackage,
    *,
    require_uid: bool,
) -> dict[str, Any]:
    proof = validate_controller_job_package(package)
    normalized = _normalize_server_job(actual, package.job)
    if normalized != package.job:
        raise ValueError("server prod10 controller Job differs from the reviewed manifest")
    metadata = actual.get("metadata", {})
    uid = metadata.get("uid")
    if require_uid and _KUBERNETES_UID.fullmatch(uid or "") is None:
        raise ValueError("created prod10 controller Job omitted its immutable UID")
    if metadata.get("annotations", {}).get(FAILURE_ALERT_ANNOTATION) != FAILURE_ALERT_OFF:
        raise ValueError("prod10 controller root alert annotation changed")
    _validate_zero_gpu(actual)
    return {
        **proof,
        "server_render_sha256": "sha256:" + digest(actual),
        "job_uid": uid if require_uid else None,
    }


def _write_once(path: Path, value: dict[str, Any]) -> None:
    try:
        with suppress(FileExistsError):
            path.parent.mkdir(mode=0o700, parents=False, exist_ok=False)
        parent_identity = path.parent.lstat()
        if (
            not stat.S_ISDIR(parent_identity.st_mode)
            or path.parent.is_symlink()
            or path.parent.resolve() != path.parent
            or (parent_identity.st_uid, parent_identity.st_gid) != (RUNTIME_UID, RUNTIME_GID)
        ):
            raise JobsError("prod10 controller journal parent identity changed")
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        parent = os.open(path.parent, directory_flags)
        try:
            fd = os.open(
                path.name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=parent,
            )
            with os.fdopen(fd, "w") as stream:
                stream.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.fsync(parent)
        finally:
            os.close(parent)
    except JobsError:
        raise
    except OSError:
        raise JobsError("prod10 controller create intent write failed; reconcile") from None


def _append(path: Path, value: dict[str, Any]) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0))
    identity = os.fstat(fd)
    if not stat.S_ISREG(identity.st_mode) or (identity.st_uid, identity.st_gid) != (
        RUNTIME_UID,
        RUNTIME_GID,
    ):
        os.close(fd)
        raise JobsError("prod10 controller journal is not one regular file")
    with os.fdopen(fd, "a") as stream:
        stream.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            identity = os.fstat(fd)
            if (
                not stat.S_ISREG(identity.st_mode)
                or (identity.st_uid, identity.st_gid) != (RUNTIME_UID, RUNTIME_GID)
                or identity.st_size > 64 * 1024
            ):
                raise JobsError("prod10 controller journal is not one bounded regular file")
            raw = os.read(fd, identity.st_size + 1)
        finally:
            os.close(fd)
        if len(raw) != identity.st_size or not raw.endswith(b"\n"):
            raise JobsError("prod10 controller journal bytes are incomplete")
        rows = [json.loads(line) for line in raw.splitlines()]
    except JobsError:
        raise
    except (OSError, ValueError):
        raise JobsError("prod10 controller journal is unreadable") from None
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise JobsError("prod10 controller journal rows are malformed")
    return rows


@dataclass(frozen=True)
class SfsControllerJournal:
    """Concrete create-once journal at the packet's canonical SFS path."""

    path: Path

    @classmethod
    def for_package(cls, package: ControllerJobPackage) -> SfsControllerJournal:
        packet = validate_controller_job_package(package)
        del packet
        return cls(Path(package.packet["controller"]["create_journal"]))

    def exists(self) -> bool:
        return self.path.exists() or self.path.is_symlink()

    def write_once(self, value: dict[str, Any]) -> None:
        _write_once(self.path, value)

    def append(self, value: dict[str, Any]) -> None:
        _append(self.path, value)

    def read_rows(self) -> list[dict[str, Any]]:
        return _read_jsonl(self.path)


def _journal_path(package: ControllerJobPackage) -> Path:
    validate_controller_job_package(package)
    path = Path(package.packet["controller"]["create_journal"])
    if not path.is_absolute() or path != Path(BOOTSTRAP_ROOT) / (
        package.packet["plan_sha256"].removeprefix("sha256:") + ".create.jsonl"
    ):
        raise JobsError("prod10 controller create journal binding changed")
    return path


def _validate_journal(journal: ControllerJournal, package: ControllerJobPackage) -> Path:
    expected = _journal_path(package)
    if not isinstance(getattr(journal, "path", None), Path) or journal.path != expected:
        raise JobsError("prod10 controller journal is not the canonical SFS path")
    return expected


def create_controller_job_once(
    package: ControllerJobPackage,
    api: ControllerJobApi,
    journal: ControllerJournal,
) -> dict[str, Any]:
    """Dry-run then create exactly one bootstrap through a typed cluster API."""

    proof = validate_controller_job_package(package)
    journal_path = _validate_journal(journal, package)
    if journal.exists():
        raise JobsError("prod10 controller create intent exists; reconcile, never retry")
    if api.get_job(NAMESPACE, NAME) is not None:
        raise JobsError("prod10 controller Job already exists")
    rendered = api.server_dry_run_job(copy.deepcopy(package.job))
    try:
        preview = validate_controller_job_response(rendered, package, require_uid=False)
    except ValueError as exc:
        raise JobsError(str(exc)) from None
    if api.get_job(NAMESPACE, NAME) is not None:
        raise JobsError("prod10 controller Job appeared after server preview")
    journal.write_once(
        {
            "state": "CREATE_INTENT_DO_NOT_RETRY",
            "name": NAME,
            "namespace": NAMESPACE,
            "journal_path": str(journal_path),
            "packet_sha256": proof["packet_sha256"],
            "manifest_sha256": proof["manifest_sha256"],
            "server_render_sha256": preview["server_render_sha256"],
        },
    )
    created = api.create_job_once(copy.deepcopy(package.job))
    try:
        accepted = validate_controller_job_response(created, package, require_uid=True)
    except ValueError as exc:
        raise JobsError("prod10 controller create response is ambiguous; reconcile") from exc
    receipt = _seal(
        {
            "schema": CREATOR_SCHEMA,
            "status": "created_once",
            "name": NAME,
            "namespace": NAMESPACE,
            "job_uid": accepted["job_uid"],
            "journal_path": str(journal_path),
            "packet_sha256": proof["packet_sha256"],
            "manifest_sha256": proof["manifest_sha256"],
            "server_render_sha256": preview["server_render_sha256"],
            "created_render_sha256": accepted["server_render_sha256"],
            "failure_alerts": FAILURE_ALERT_OFF,
            "priority": POD_PRIORITY,
            "queue_priority": QUEUE_PRIORITY,
            "gpus": 0,
            "target_posts": 0,
        }
    )
    journal.append({"state": "CREATE_RESPONSE", **receipt})
    return receipt


def _validate_creator(
    receipt: object,
    package: ControllerJobPackage,
    journal: ControllerJournal,
) -> dict[str, Any]:
    proof = validate_controller_job_package(package)
    journal_path = _validate_journal(journal, package)
    if not isinstance(receipt, dict) or receipt != _seal(receipt):
        raise ValueError("prod10 controller creator receipt digest changed")
    if (
        set(receipt)
        != {
            "schema",
            "status",
            "name",
            "namespace",
            "job_uid",
            "journal_path",
            "packet_sha256",
            "manifest_sha256",
            "server_render_sha256",
            "created_render_sha256",
            "failure_alerts",
            "priority",
            "queue_priority",
            "gpus",
            "target_posts",
            "sha256",
        }
        or receipt.get("schema") != CREATOR_SCHEMA
        or receipt.get("status") != "created_once"
        or receipt.get("name") != NAME
        or receipt.get("namespace") != NAMESPACE
        or receipt.get("journal_path") != str(journal_path)
        or receipt.get("packet_sha256") != proof["packet_sha256"]
        or receipt.get("manifest_sha256") != proof["manifest_sha256"]
        or receipt.get("failure_alerts") != FAILURE_ALERT_OFF
        or receipt.get("priority") != POD_PRIORITY
        or receipt.get("queue_priority") != QUEUE_PRIORITY
        or receipt.get("gpus") != 0
        or receipt.get("target_posts") != 0
        or _SHA256.fullmatch(receipt.get("server_render_sha256", "")) is None
        or _SHA256.fullmatch(receipt.get("created_render_sha256", "")) is None
        or _KUBERNETES_UID.fullmatch(receipt.get("job_uid", "")) is None
    ):
        raise ValueError("prod10 controller creator receipt changed")
    try:
        rows = journal.read_rows()
    except JobsError as exc:
        raise ValueError("prod10 controller canonical create journal is unreadable") from exc
    expected_intent = {
        "state": "CREATE_INTENT_DO_NOT_RETRY",
        "name": NAME,
        "namespace": NAMESPACE,
        "journal_path": str(journal_path),
        "packet_sha256": proof["packet_sha256"],
        "manifest_sha256": proof["manifest_sha256"],
        "server_render_sha256": receipt["server_render_sha256"],
    }
    if rows != [expected_intent, {"state": "CREATE_RESPONSE", **receipt}]:
        raise ValueError("prod10 controller creator receipt differs from its canonical journal")
    return receipt


def collect_controller_terminal_receipt(
    package: ControllerJobPackage,
    creator: dict[str, Any],
    journal: ControllerJournal,
    job: dict[str, Any],
    workloads: dict[str, Any],
    pods: dict[str, Any],
    service_account: dict[str, Any],
    runtime_logs: str,
) -> dict[str, Any]:
    """Bind one successful bootstrap Pod and its sanitized SFS receipt."""

    from .sfs_output_job import (
        _contains,
        _normalize_job_serialization,
        _validate_admitted_workload,
        _validate_pod_spec,
        _validate_server_job_surface,
    )
    from .skyrl_controller_driver import LOG_PREFIX, canonical_json, validate_receipt

    created = _validate_creator(creator, package, journal)
    uid = created["job_uid"]
    metadata = job.get("metadata", {}) if isinstance(job, dict) else {}
    conditions = job.get("status", {}).get("conditions", []) if isinstance(job, dict) else []
    normalized_workloads = copy.deepcopy(workloads)
    if (
        normalized_workloads.get("apiVersion") != "kueue.x-k8s.io/v1beta2"
        or normalized_workloads.get("kind") != "WorkloadList"
    ):
        raise ValueError("prod10 controller Workload inventory GVK changed")
    normalized_workloads["kind"] = "List"
    admitted_workload = _validate_admitted_workload(package, job, normalized_workloads)
    expected_job = copy.deepcopy(package.job)
    expected_job["spec"]["suspend"] = False
    normalized_job = _normalize_job_serialization(job, expected_job)
    if not _contains(normalized_job, expected_job):
        raise ValueError("terminal prod10 controller Job differs from the reviewed manifest")
    _validate_server_job_surface(
        normalized_job,
        expected_job,
        admitted_workload=admitted_workload,
    )
    complete = [
        row
        for row in conditions
        if isinstance(row, dict) and row.get("type") == "Complete" and row.get("status") == "True"
    ]
    failed = [
        row
        for row in conditions
        if isinstance(row, dict) and row.get("type") == "Failed" and row.get("status") == "True"
    ]
    if (
        metadata.get("name") != NAME
        or metadata.get("namespace") != NAMESPACE
        or metadata.get("uid") != uid
        or len(complete) != 1
        or failed
    ):
        raise ValueError("prod10 controller terminal Job identity/status changed")
    rows = pods.get("items") if isinstance(pods, dict) else None
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
        raise ValueError("prod10 controller requires one exact terminal Pod")
    if pods.get("apiVersion") != "v1" or pods.get("kind") != "PodList":
        raise ValueError("prod10 controller Pod inventory GVK changed")
    pod = rows[0]
    pod_metadata = pod.get("metadata", {})
    expected_owner = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "name": NAME,
        "uid": uid,
        "controller": True,
        "blockOwnerDeletion": True,
    }
    owners = pod_metadata.get("ownerReferences", [])
    statuses = pod.get("status", {}).get("containerStatuses", [])
    pod_spec = pod.get("spec", {})
    expected_pod = package.job["spec"]["template"]["spec"]
    _validate_pod_spec(
        pod_spec,
        expected_pod,
        scheduled=True,
        service_account=service_account,
    )
    live_template_metadata = job["spec"]["template"]["metadata"]
    live_labels = pod_metadata.get("labels")
    normalized_labels = copy.deepcopy(live_labels) if isinstance(live_labels, dict) else {}
    region = normalized_labels.pop("topology.kubernetes.io/region", None)
    statuses = pod.get("status", {}).get("containerStatuses", [])
    image_digest = IMAGE.rsplit("@", 1)[1]
    terminated = statuses[0].get("state", {}).get("terminated", {}) if len(statuses) == 1 else {}
    termination_message = terminated.get("message")
    if (
        pod.get("apiVersion") != "v1"
        or pod.get("kind") != "Pod"
        or pod_metadata.get("namespace") != NAMESPACE
        or _KUBERNETES_UID.fullmatch(pod_metadata.get("uid", "")) is None
        or owners != [expected_owner]
        or pod_metadata.get("annotations") != live_template_metadata.get("annotations")
        or normalized_labels != live_template_metadata.get("labels")
        or (region is not None and (not isinstance(region, str) or not region))
        or pod.get("status", {}).get("phase") != "Succeeded"
        or len(statuses) != 1
        or statuses[0].get("name") != "controller"
        or statuses[0].get("image") != IMAGE
        or statuses[0].get("restartCount") != 0
        or terminated.get("exitCode") != 0
        or terminated.get("reason") != "Completed"
        or not isinstance(termination_message, str)
        or not termination_message
        or not statuses[0].get("imageID", "").endswith("@" + image_digest)
    ):
        raise ValueError("prod10 controller terminal Pod was not one clean success")
    if not isinstance(runtime_logs, str) or runtime_logs != LOG_PREFIX + termination_message + "\n":
        raise ValueError("prod10 controller Pod log and termination receipt differ")
    try:
        runtime_receipt = json.loads(termination_message)
    except json.JSONDecodeError as exc:
        raise ValueError("prod10 controller Pod termination receipt is invalid JSON") from exc
    if canonical_json(runtime_receipt).decode() != termination_message:
        raise ValueError("prod10 controller Pod termination receipt is not canonical")
    checked_runtime = validate_receipt(
        runtime_receipt,
        packet=package.packet,
        job_uid=uid,
        pod_name=pod_metadata.get("name", ""),
        pod_uid=pod_metadata.get("uid", ""),
    )
    proof = validate_controller_job_package(package)
    if checked_runtime.get("driver_sha256") != proof["driver_sha256"]:
        raise ValueError("prod10 controller runtime driver binding changed")
    return _seal(
        {
            "schema": TERMINAL_SCHEMA,
            "status": "accepted_non_submitting_bootstrap",
            "name": NAME,
            "namespace": NAMESPACE,
            "job_uid": uid,
            "pod_name": pod_metadata["name"],
            "pod_uid": pod_metadata["uid"],
            "workload_name": admitted_workload["name"],
            "workload_uid": admitted_workload["uid"],
            "local_queue": admitted_workload["local_queue"],
            "cluster_queue": admitted_workload["cluster_queue"],
            "effective_priority": admitted_workload["priority"],
            "resolved_image_id": statuses[0]["imageID"],
            "packet_sha256": package.packet["sha256"],
            "creator_receipt_sha256": created["sha256"],
            "runtime_receipt_sha256": checked_runtime["sha256"],
            "runtime_receipt_path": package.packet["controller"]["bootstrap_receipt"],
            "termination_message_sha256": "sha256:"
            + hashlib.sha256(termination_message.encode()).hexdigest(),
            "sanitized_log_sha256": "sha256:" + hashlib.sha256(runtime_logs.encode()).hexdigest(),
            "failure_alerts": FAILURE_ALERT_OFF,
            "priority": POD_PRIORITY,
            "queue_priority": QUEUE_PRIORITY,
            "gpus": 0,
            "target_posts": 0,
            "launch_authorized": False,
        }
    )


def collect_controller_terminal_from_api(
    package: ControllerJobPackage,
    creator: dict[str, Any],
    journal: ControllerJournal,
    api: ControllerTerminalApi,
) -> dict[str, Any]:
    """Fetch only exact-identity live evidence, then seal terminal acceptance."""

    created = _validate_creator(creator, package, journal)
    job, workloads, pods, service_account, logs = api.terminal_evidence(job_uid=created["job_uid"])
    return collect_controller_terminal_receipt(
        package,
        creator,
        journal,
        job,
        workloads,
        pods,
        service_account,
        logs,
    )


def assert_target_post_disabled(packet: dict[str, Any]) -> None:
    """Fail closed until immutable cleanup and metadata-only capacity exist."""

    checked = validate_controller_packet(packet)
    if checked["launch_authorized"] is not False or any(checked["platform_gates"].values()):
        raise ValueError("prod10 controller packet attempted to override closed platform gates")
    raise ValueError("prod10 target POST is disabled pending exact platform capabilities")


def validate_controls_path(path: str) -> None:
    value = PurePosixPath(path)
    if str(value) != CONTROLS_MOUNT or value.parts[-1] != "launch-controls":
        raise ValueError("prod10 controller controls mount changed")
