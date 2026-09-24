"""Exact-identity release supervision for the held Qwen3.8 262K 4x8 canary.

This module does not create a workload.  A launcher must arm it before its one
external create attempt, persist that intent, and pass the resulting exact
RayJob through :meth:`FourNodeReleaseSupervisor.reconcile_exact`.  The only
destructive operation it can request is foreground deletion of that exact
RayJob with its immutable UID as a Kubernetes precondition.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import stat
import tempfile
from collections.abc import Callable
from contextlib import contextmanager, suppress
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Protocol
from uuid import UUID

from cyber_post_train.jobs import JobsError, digest, quantity
from training import sft_262k_4node_v1 as compiler
from training import sft_262k_runtime as runtime

SCHEMA = "qwen38_262k_4node_release_supervisor_v1"
ARMED_SCHEMA = SCHEMA + ":armed"
INTENT_SCHEMA = SCHEMA + ":create_intent"
BINDING_SCHEMA = SCHEMA + ":binding"
AUTHORIZATION_SCHEMA = SCHEMA + ":release_authorization"
DELEGATION_SCHEMA = SCHEMA + ":release_delegation"
DRIFT_DELEGATION_SCHEMA = SCHEMA + ":drift_cleanup_delegation"
DRIFT_BINDING_SCHEMA = SCHEMA + ":drift_cleanup_binding"
DRIFT_AUTHORIZATION_SCHEMA = SCHEMA + ":drift_cleanup_authorization"
DRIFT_DELETE_INTENT_SCHEMA = SCHEMA + ":drift_cleanup_delete_intent"
DRIFT_RESULT_SCHEMA = SCHEMA + ":drift_cleanup_result"
DRIFT_RESOURCE_SCHEMA = SCHEMA + ":drift_cleanup_resource"
BRANCH_SCHEMA = SCHEMA + ":branch_decision"
RESULT_SCHEMA = SCHEMA + ":result"
DELETE_INTENT_SCHEMA = SCHEMA + ":delete_intent"
STATE_SCHEMA = SCHEMA + ":state"

RUN_NAME = "chris-q38-t3k262-4n-can-v1"
RUN_DIR = "/mnt/sfs/jobs/chris-q38-t3k262-4n-can-v1"
NAMESPACE = "fleet-train-jobs"
CONTEXT = "nebius-mk8s-fleetai-training-e04zw4ye1k7wczqdw6"
IMAGE = compiler.IMAGE
NODES = 4
GPUS_PER_NODE = 8
TOTAL_GPUS = NODES * GPUS_PER_NODE
POLL_SECONDS = 60
ALLOCATION_TO_STARTED_SECONDS = 1800
STARTED_TO_ACTION_SECONDS = 29100
ALLOCATION_TO_ACTION_SECONDS = 30900
ALLOCATION_TO_RELEASE_SECONDS = 31500
TERMINAL_GRACE_SECONDS = 300
DELETE_CONFIRM_SECONDS = 300
RELEASE_REASONS = {
    "authenticated_started_deadline_elapsed",
    "allocation_started_receipt_deadline_elapsed",
    "observation_contract_defect",
    "terminal_self_release_grace_elapsed",
}
DELEGATION_DERIVATION_RULE = (
    "prearm_exact_name_absence_single_post_exact_surface_unique_uid_binding"
)
TERMINAL_STATUSES = {"SUCCEEDED": "Succeeded", "FAILED": "Failed"}


class SupervisorError(RuntimeError):
    """A sanitized release-supervision contract failure."""


class ClusterBackend(Protocol):
    """The narrow Kubernetes surface required by this exact supervisor."""

    def get_rayjob(self, name: str) -> dict | None: ...

    def list_workloads_by_job_uid(self, rayjob_uid: str) -> list[dict]: ...

    def get_workload(self, name: str) -> dict | None: ...

    def list_rayclusters_by_rayjob_uid(self, name: str, uid: str) -> list[dict]: ...

    def get_raycluster(self, name: str) -> dict | None: ...

    def list_pods(self, cluster_name: str, controller_uid: str) -> list[dict]: ...

    def list_pods_by_run_identity(self, run_id: str, run_name: str) -> list[dict]: ...

    def get_pod(self, name: str) -> dict | None: ...

    def delete_rayjob_uid_foreground(self, name: str, uid: str) -> None: ...


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _seal(value: dict) -> dict:
    return {**value, "sha256": "sha256:" + digest(value)}


def _parse_time(value: object) -> datetime:
    if not isinstance(value, str):
        raise SupervisorError("resource timestamp is invalid")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SupervisorError("resource timestamp is invalid") from exc
    if parsed.tzinfo is None:
        raise SupervisorError("resource timestamp is invalid")
    return parsed.astimezone(UTC)


def _stamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_once(path: Path, value: object) -> None:
    fd, temporary_name = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(_canonical(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.close(fd)
        fd = -1
        os.link(temporary, path)
        _fsync_dir(path.parent)
    finally:
        if fd >= 0:
            os.close(fd)
        temporary.unlink(missing_ok=True)


def _write_atomic(path: Path, value: object) -> None:
    fd, temporary_name = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(_canonical(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.close(fd)
        fd = -1
        os.replace(temporary, path)
        os.chmod(path, 0o600)
        _fsync_dir(path.parent)
    finally:
        if fd >= 0:
            os.close(fd)
        temporary.unlink(missing_ok=True)


@contextmanager
def _exclusive_state_lock(operation_dir: Path):
    """Serialize state/result publication through one persistent private inode."""
    path = operation_dir / "STATE.lock"
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags, 0o600)
    except OSError as exc:
        raise SupervisorError("observation-state lock is unavailable") from exc
    try:
        metadata = os.fstat(fd)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) != 0o600
        ):
            raise SupervisorError("observation-state lock is not a private file")
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    except OSError as exc:
        raise SupervisorError("observation-state lock failed") from exc
    finally:
        with suppress(OSError):
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, ValueError) as exc:
        raise SupervisorError(f"{path.name} is unavailable or invalid") from exc
    if not isinstance(value, dict):
        raise SupervisorError(f"{path.name} is unavailable or invalid")
    return value


def _validate_seal(value: dict, schema: str) -> dict:
    body = {key: item for key, item in value.items() if key != "sha256"}
    if value.get("schema") != schema or value.get("sha256") != "sha256:" + digest(body):
        raise SupervisorError(f"{schema} receipt is invalid")
    return value


def _metadata(resource: dict, kind: str) -> tuple[str, str, datetime]:
    metadata = resource.get("metadata")
    if resource.get("kind") != kind or not isinstance(metadata, dict):
        raise SupervisorError(f"{kind} is malformed")
    name, uid = metadata.get("name"), metadata.get("uid")
    try:
        UUID(uid)
    except (TypeError, ValueError) as exc:
        raise SupervisorError(f"{kind} identity is invalid") from exc
    if not isinstance(name, str) or not name:
        raise SupervisorError(f"{kind} identity is invalid")
    return name, uid, _parse_time(metadata.get("creationTimestamp"))


def _owned_by(resource: dict, *, kind: str, name: str, uid: str) -> bool:
    metadata = resource.get("metadata")
    owners = metadata.get("ownerReferences") if isinstance(metadata, dict) else None
    return isinstance(owners, list) and any(
        isinstance(owner, dict)
        and owner.get("kind") == kind
        and owner.get("name") == name
        and owner.get("uid") == uid
        and owner.get("controller") is True
        for owner in owners
    )


def _merge_unique(resources: list[dict], additions: list[dict], kind: str) -> list[dict]:
    by_name: dict[str, str] = {}
    by_uid: dict[str, str] = {}
    merged = []
    for resource in [*resources, *additions]:
        name, uid, _ = _metadata(resource, kind)
        if (name in by_name and by_name[name] != uid) or (uid in by_uid and by_uid[uid] != name):
            raise SupervisorError(f"{kind} census identity changed")
        if uid not in by_uid:
            merged.append(resource)
        by_name[name] = uid
        by_uid[uid] = name
    return merged


def _validate_final_census(value: object, *, confirmed: bool) -> dict:
    if not isinstance(value, dict):
        raise SupervisorError("terminal release census is invalid")
    expected = {"root", "workloads", "rayclusters", "pods", "active_gpus"}
    if set(value) == {"reason"}:
        if confirmed or not isinstance(value["reason"], str) or not value["reason"]:
            raise SupervisorError("terminal release census reason is invalid")
        return value
    if set(value) != expected or any(type(value[key]) is not int for key in expected):
        raise SupervisorError("terminal release census is invalid")
    if any(value[key] < 0 for key in expected) or value["active_gpus"] > TOTAL_GPUS:
        raise SupervisorError("terminal release census counts are invalid")
    if confirmed and any(value.values()):
        raise SupervisorError("confirmed terminal release census is not zero")
    return value


def _pod_gpu_count(pod: dict) -> int:
    spec = pod.get("spec")
    if not isinstance(spec, dict):
        raise SupervisorError("Pod is malformed")

    def gpu(container: object) -> int:
        if not isinstance(container, dict):
            raise SupervisorError("Pod is malformed")
        resources = container.get("resources", {})
        requests = resources.get("requests", {}) if isinstance(resources, dict) else {}
        limits = resources.get("limits", {}) if isinstance(resources, dict) else {}
        requested = quantity(requests.get("nvidia.com/gpu", 0))
        limited = quantity(limits.get("nvidia.com/gpu", 0))
        if requested != limited or requested != int(requested):
            raise SupervisorError("Pod GPU request and limit differ")
        return int(requested)

    regular = spec.get("containers")
    init = spec.get("initContainers", [])
    if not isinstance(regular, list) or not isinstance(init, list):
        raise SupervisorError("Pod is malformed")
    overhead = spec.get("overhead", {})
    if not isinstance(overhead, dict) or quantity(overhead.get("nvidia.com/gpu", 0)):
        raise SupervisorError("Pod GPU overhead changed the exact topology")
    if spec.get("resourceClaims") not in (None, []):
        raise SupervisorError("Pod resource claims changed the exact topology")
    restartable = sum(
        gpu(container)
        for container in init
        if isinstance(container, dict) and container.get("restartPolicy") == "Always"
    )
    ordinary = [
        gpu(container)
        for container in init
        if not isinstance(container, dict) or container.get("restartPolicy") != "Always"
    ]
    return sum(gpu(container) for container in regular) + restartable + max(ordinary, default=0)


def _pod_allocation_time(pod: dict, created: datetime) -> datetime | None:
    """Return a demonstrated PodScheduled transition, never Pod creation time."""
    spec = pod.get("spec")
    status = pod.get("status")
    if not isinstance(spec, dict) or not isinstance(status, dict):
        raise SupervisorError("Pod is malformed")
    node = spec.get("nodeName")
    conditions = status.get("conditions", [])
    if not node:
        return None
    if not isinstance(node, str) or not isinstance(conditions, list):
        raise SupervisorError("Pod scheduling evidence is malformed")
    scheduled = [
        item
        for item in conditions
        if isinstance(item, dict)
        and item.get("type") == "PodScheduled"
        and item.get("status") == "True"
    ]
    if not scheduled:
        return None
    if len(scheduled) != 1:
        raise SupervisorError("Pod scheduling evidence is ambiguous")
    allocated = _parse_time(scheduled[0].get("lastTransitionTime"))
    if allocated < created:
        raise SupervisorError("Pod scheduling evidence predates Pod creation")
    return allocated


def _validate_template(template: dict, replicas: int) -> None:
    try:
        spec = template["template"]["spec"]
        containers = spec["containers"]
    except (KeyError, TypeError) as exc:
        raise SupervisorError("rendered RayJob topology is malformed") from exc
    if replicas < 1 or spec.get("priorityClassName") != "c1" or not isinstance(containers, list):
        raise SupervisorError("rendered RayJob topology changed")
    gpu_containers = []
    for container in containers:
        if not isinstance(container, dict):
            raise SupervisorError("rendered RayJob topology is malformed")
        resources = container.get("resources", {})
        requests = resources.get("requests", {}) if isinstance(resources, dict) else {}
        limits = resources.get("limits", {}) if isinstance(resources, dict) else {}
        requested = quantity(requests.get("nvidia.com/gpu", 0))
        limited = quantity(limits.get("nvidia.com/gpu", 0))
        if requested != limited:
            raise SupervisorError("rendered RayJob topology changed")
        if requested:
            gpu_containers.append((container, int(requested)))
    if (
        len(gpu_containers) != 1
        or gpu_containers[0][0].get("image") != IMAGE
        or gpu_containers[0][1] != GPUS_PER_NODE
    ):
        raise SupervisorError("rendered RayJob image or GPU topology changed")
    if any(
        _pod_gpu_count({"spec": {"containers": [], "initContainers": [item]}})
        for item in spec.get("initContainers", [])
    ):
        raise SupervisorError("rendered RayJob init container requests a GPU")


def validate_exact_candidate(plan: dict, request: dict, manifest: dict) -> None:
    """Validate the one held candidate without making its GPU gate launchable."""
    runtime.validate_plan(plan, check_files=False)
    if request != compiler.job_request(plan):
        raise SupervisorError("candidate request differs from the exact compiled request")
    metadata = manifest.get("metadata")
    spec = manifest.get("spec")
    if (
        manifest.get("kind") != "RayJob"
        or not isinstance(metadata, dict)
        or not isinstance(spec, dict)
    ):
        raise SupervisorError("rendered RayJob is malformed")
    labels, annotations = metadata.get("labels"), metadata.get("annotations")
    run_id = annotations.get("fleet.ai/run-id") if isinstance(annotations, dict) else None
    try:
        UUID(run_id)
    except (TypeError, ValueError) as exc:
        raise SupervisorError("rendered RayJob run ID is invalid") from exc
    expected_name = f"{RUN_NAME}-{run_id[:8]}"
    if (
        metadata.get("name") != expected_name
        or metadata.get("namespace") != NAMESPACE
        or not isinstance(labels, dict)
        or labels.get("fleet.ai/run-id") != run_id
        or labels.get("fleet.ai/run-name") != RUN_NAME
        or labels.get("fleet.ai/requeue-if-preempted") != "false"
        or labels.get("kueue.x-k8s.io/priority-class") != "q1"
        or labels.get("kueue.x-k8s.io/queue-name") != "training-lq"
        or annotations.get("fleet.ai/failure-alerts") != "off"
        or annotations.get("fleet.ai/run-dir") != RUN_DIR
        or annotations.get("fleet.ai/job-image") != IMAGE
        or spec.get("shutdownAfterJobFinishes") is not True
        or "ttlSecondsAfterFinished" in spec
        or spec.get("backoffLimit", 0) != 0
    ):
        raise SupervisorError("rendered RayJob identity or release contract changed")
    try:
        cluster = spec["rayClusterSpec"]
        head = cluster["headGroupSpec"]
        workers = cluster["workerGroupSpecs"]
    except (KeyError, TypeError) as exc:
        raise SupervisorError("rendered RayJob topology is malformed") from exc
    if not isinstance(workers, list) or len(workers) != 1 or workers[0].get("replicas") != 3:
        raise SupervisorError("rendered RayJob is not the exact four-node topology")
    if workers[0].get("minReplicas") != 3 or workers[0].get("maxReplicas") != 3:
        raise SupervisorError("rendered RayJob worker replica bounds changed")
    _validate_template(head, 1)
    _validate_template(workers[0], 3)


def _validate_armed(value: dict, plan: dict, request: dict, manifest: dict) -> dict:
    _validate_seal(value, ARMED_SCHEMA)
    expected = {
        "status": "armed_before_external_create",
        "context": CONTEXT,
        "namespace": NAMESPACE,
        "rayjob_name": manifest["metadata"]["name"],
        "jobs_api_run_id": manifest["metadata"]["annotations"]["fleet.ai/run-id"],
        "run_dir": RUN_DIR,
        "image": IMAGE,
        "nodes": NODES,
        "gpus_per_node": GPUS_PER_NODE,
        "total_gpus": TOTAL_GPUS,
        "plan_sha256": "sha256:" + digest(plan),
        "request_sha256": "sha256:" + digest(request),
        "manifest_sha256": "sha256:" + digest(manifest),
        "exact_name_absent_before_arm": True,
    }
    if set(value) != {
        "schema",
        *expected,
        "armed_at",
        "observer_pid",
        "observer_receipt_sha256",
        "sha256",
    } or any(value.get(key) != item for key, item in expected.items()):
        raise SupervisorError("armed packet differs from the exact candidate")
    if type(value.get("observer_pid")) is not int or value["observer_pid"] < 1:
        raise SupervisorError("armed packet observer identity is invalid")
    if re.fullmatch(r"sha256:[a-f0-9]{64}", value.get("observer_receipt_sha256", "")) is None:
        raise SupervisorError("armed packet observer identity is invalid")
    _parse_time(value.get("armed_at"))
    return value


def _validate_create_intent(value: dict, armed: dict) -> dict:
    _validate_seal(value, INTENT_SCHEMA)
    expected = {
        "status": "external_create_may_be_attempted_once",
        "armed_sha256": armed["sha256"],
        "observer_pid": armed["observer_pid"],
        "observer_receipt_sha256": armed["observer_receipt_sha256"],
        "rayjob_name": armed["rayjob_name"],
        "jobs_api_run_id": armed["jobs_api_run_id"],
        "manifest_sha256": armed["manifest_sha256"],
        "uncertain_create_policy": "reconcile_this_exact_name_never_retry_or_discover",
    }
    if set(value) != {"schema", *expected, "written_at", "sha256"} or any(
        value.get(key) != item for key, item in expected.items()
    ):
        raise SupervisorError("create intent differs from the armed exact name")
    _parse_time(value.get("written_at"))
    return value


def _validate_delegation(value: dict, armed: dict) -> dict:
    _validate_seal(value, DELEGATION_SCHEMA)
    expected = {
        "status": "derive_exact_uid_authorization_after_validated_binding_only",
        "armed_sha256": armed["sha256"],
        "context": CONTEXT,
        "namespace": NAMESPACE,
        "jobs_api_run_id": armed["jobs_api_run_id"],
        "rayjob_name": armed["rayjob_name"],
        "run_dir": RUN_DIR,
        "image": IMAGE,
        "nodes": NODES,
        "gpus_per_node": GPUS_PER_NODE,
        "total_gpus": TOTAL_GPUS,
        "plan_sha256": armed["plan_sha256"],
        "request_sha256": armed["request_sha256"],
        "manifest_sha256": armed["manifest_sha256"],
        "authorized_reasons": sorted(RELEASE_REASONS),
        "derivation_rule": DELEGATION_DERIVATION_RULE,
        "delete_route": "root_rayjob_uid_cas_only",
    }
    if set(value) != {*expected, "schema", "delegated_at", "sha256"} or any(
        value.get(key) != item for key, item in expected.items()
    ):
        raise SupervisorError("release delegation differs from the armed packet")
    _parse_time(value.get("delegated_at"))
    return value


def _validate_drift_delegation(value: dict, armed: dict) -> dict:
    _validate_seal(value, DRIFT_DELEGATION_SCHEMA)
    expected = {
        "status": "cleanup_only_never_accept",
        "armed_sha256": armed["sha256"],
        "context": CONTEXT,
        "namespace": NAMESPACE,
        "jobs_api_run_id": armed["jobs_api_run_id"],
        "rayjob_name": armed["rayjob_name"],
        "manifest_sha256": armed["manifest_sha256"],
        "one_shot_intent_required": True,
        "authorized_reason": "observation_contract_defect",
        "delete_route": "root_rayjob_uid_cas_only",
        "acceptance_forbidden": True,
    }
    if set(value) != {"schema", *expected, "delegated_at", "sha256"} or any(
        value.get(key) != item for key, item in expected.items()
    ):
        raise SupervisorError("drift cleanup delegation differs from the armed packet")
    _parse_time(value.get("delegated_at"))
    return value


def _validate_drift_binding(value: dict, armed: dict, intent: dict, delegation: dict) -> dict:
    _validate_seal(value, DRIFT_BINDING_SCHEMA)
    expected = {
        "status": "cleanup_only_exact_uid_not_accepted",
        "armed_sha256": armed["sha256"],
        "intent_sha256": intent["sha256"],
        "delegation_sha256": delegation["sha256"],
        "context": CONTEXT,
        "namespace": NAMESPACE,
        "jobs_api_run_id": armed["jobs_api_run_id"],
        "rayjob_name": armed["rayjob_name"],
        "manifest_sha256": armed["manifest_sha256"],
        "surface_validation": "rejected",
        "acceptance_forbidden": True,
    }
    if set(value) != {
        "schema",
        *expected,
        "rayjob_uid",
        "rayjob_created_at",
        "bound_at",
        "sha256",
    } or any(value.get(key) != item for key, item in expected.items()):
        raise SupervisorError("drift cleanup binding is invalid")
    try:
        uid = UUID(value["rayjob_uid"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SupervisorError("drift cleanup binding UID is invalid") from exc
    if str(uid) != value["rayjob_uid"]:
        raise SupervisorError("drift cleanup binding UID is invalid")
    _parse_time(value.get("rayjob_created_at"))
    _parse_time(value.get("bound_at"))
    return value


def _validate_drift_authorization(value: dict, binding: dict, delegation: dict) -> dict:
    _validate_seal(value, DRIFT_AUTHORIZATION_SCHEMA)
    expected = {
        "status": "cleanup_only_exact_root_uid_delete",
        "binding_sha256": binding["sha256"],
        "delegation_sha256": delegation["sha256"],
        "rayjob_name": binding["rayjob_name"],
        "rayjob_uid": binding["rayjob_uid"],
        "authorized_reason": "observation_contract_defect",
        "delete_route": "root_rayjob_uid_cas_only",
        "acceptance_forbidden": True,
    }
    if set(value) != {"schema", *expected, "authorized_at", "sha256"} or any(
        value.get(key) != item for key, item in expected.items()
    ):
        raise SupervisorError("drift cleanup authorization is invalid")
    _parse_time(value.get("authorized_at"))
    return value


def _validate_authorization_value(value: dict, binding: dict, delegation: dict) -> dict:
    _validate_seal(value, AUTHORIZATION_SCHEMA)
    if (
        set(value)
        != {
            "schema",
            "status",
            "binding_sha256",
            "delegation_sha256",
            "rayjob_name",
            "rayjob_uid",
            "authorized_reasons",
            "authorized_at",
            "delete_route",
            "sha256",
        }
        or value.get("status") != "authorized_exact_root_uid_delete"
        or value.get("binding_sha256") != binding["sha256"]
        or value.get("delegation_sha256") != delegation["sha256"]
        or value.get("rayjob_name") != binding["rayjob_name"]
        or value.get("rayjob_uid") != binding["rayjob_uid"]
        or value.get("authorized_reasons") != sorted(RELEASE_REASONS)
        or value.get("delete_route") != "root_rayjob_uid_cas_only"
    ):
        raise SupervisorError("release authorization is invalid")
    _parse_time(value.get("authorized_at"))
    return value


def _validate_binding_for_authorization(
    binding: dict, armed: dict, delegation: dict, intent: dict
) -> dict:
    _validate_seal(binding, BINDING_SCHEMA)
    expected = {
        "status": "bound_exact_uid",
        "armed_sha256": armed["sha256"],
        "intent_sha256": intent["sha256"],
        "context": CONTEXT,
        "namespace": NAMESPACE,
        "jobs_api_run_id": armed["jobs_api_run_id"],
        "rayjob_name": armed["rayjob_name"],
        "manifest_sha256": armed["manifest_sha256"],
    }
    if set(binding) != {
        "schema",
        *expected,
        "rayjob_uid",
        "rayjob_created_at",
        "bound_at",
        "sha256",
    } or any(binding.get(key) != item for key, item in expected.items()):
        raise SupervisorError("exact UID binding differs from release delegation")
    try:
        uid = UUID(binding["rayjob_uid"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SupervisorError("exact UID binding is invalid") from exc
    if str(uid) != binding["rayjob_uid"]:
        raise SupervisorError("exact UID binding is invalid")
    # Parse and seal both clocks as evidence, but never compare the Kubernetes
    # API server's wall clock with this launcher's wall clock.  Causality is
    # established by exact-name absence, the locally generated run UUID, the
    # fsynced one-shot intent, and the exact persisted runtime surface.
    _parse_time(binding.get("rayjob_created_at"))
    _parse_time(binding.get("bound_at"))
    if delegation.get("armed_sha256") != armed["sha256"]:
        raise SupervisorError("release delegation differs from the armed packet")
    return binding


def make_release_authorization(binding: dict, *, authorized_at: datetime) -> dict:
    """Return the creator's exact-UID authorization for all bounded stop reasons."""
    _validate_seal(binding, BINDING_SCHEMA)
    return _seal(
        {
            "schema": AUTHORIZATION_SCHEMA,
            "status": "authorized_exact_root_uid_delete",
            "binding_sha256": binding["sha256"],
            "rayjob_name": binding["rayjob_name"],
            "rayjob_uid": binding["rayjob_uid"],
            "authorized_reasons": sorted(RELEASE_REASONS),
            "authorized_at": _stamp(authorized_at),
            "delete_route": "root_rayjob_uid_cas_only",
        }
    )


def persist_release_authorization(
    operation_dir: Path, binding: dict, *, authorized_at: datetime
) -> dict:
    """Fsync the creator's sealed authorization exactly once with mode 0600."""
    plan = _read_json(operation_dir / "PLAN.json")
    request = _read_json(operation_dir / "REQUEST.json")
    manifest = _read_json(operation_dir / "RENDERED_RAYJOB.json")
    validate_exact_candidate(plan, request, manifest)
    armed = _validate_armed(_read_json(operation_dir / "ARMED.json"), plan, request, manifest)
    delegation = _validate_delegation(_read_json(operation_dir / "RELEASE_DELEGATION.json"), armed)
    intent = _validate_create_intent(_read_json(operation_dir / "CREATE_INTENT.json"), armed)
    _validate_binding_for_authorization(binding, armed, delegation, intent)
    path = operation_dir / "RELEASE_AUTHORIZATION.json"
    if path.exists():
        return _validate_authorization_value(_read_json(path), binding, delegation)
    base = make_release_authorization(binding, authorized_at=authorized_at)
    value = _seal(
        {
            **{key: item for key, item in base.items() if key != "sha256"},
            "delegation_sha256": delegation["sha256"],
        }
    )
    try:
        _write_once(path, value)
    except FileExistsError:
        return _validate_authorization_value(_read_json(path), binding, delegation)
    return value


class FourNodeReleaseSupervisor:
    """Persistent exact-name/UID observer for the one held 4x8 candidate."""

    def __init__(
        self,
        operation_dir: Path,
        backend: ClusterBackend,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        sleep: Callable[[float], None] | None = None,
        started_reader: Callable[[], dict | None] = lambda: None,
        process_alive: Callable[[int], bool] = lambda pid: _process_alive(pid),
    ) -> None:
        self.operation_dir = operation_dir
        self.backend = backend
        self.clock = clock
        self.sleep = sleep
        self.started_reader = started_reader
        self.process_alive = process_alive
        self.plan = _read_json(operation_dir / "PLAN.json")
        self.request = _read_json(operation_dir / "REQUEST.json")
        self.manifest = _read_json(operation_dir / "RENDERED_RAYJOB.json")
        validate_exact_candidate(self.plan, self.request, self.manifest)
        self.armed = _validate_armed(
            _read_json(operation_dir / "ARMED.json"),
            self.plan,
            self.request,
            self.manifest,
        )
        self.delegation = _validate_delegation(
            _read_json(operation_dir / "RELEASE_DELEGATION.json"), self.armed
        )
        self.drift_delegation = _validate_drift_delegation(
            _read_json(operation_dir / "DRIFT_CLEANUP_DELEGATION.json"), self.armed
        )
        self.root_name = self.manifest["metadata"]["name"]
        self.run_id = self.manifest["metadata"]["annotations"]["fleet.ai/run-id"]
        self.binding: dict | None = None
        self.branch_decision: dict | None = None
        branch_path = operation_dir / "BRANCH_DECISION.json"
        if branch_path.exists():
            self.branch_decision = self._validate_branch_decision(_read_json(branch_path))
            if self.branch_decision["branch"] == "ordinary":
                self.binding = self.branch_decision["binding"]
        binding_path = operation_dir / "BOUND.json"
        if binding_path.exists():
            persisted_binding = self._validate_binding(_read_json(binding_path))
            if self.binding is not None and persisted_binding != self.binding:
                raise SupervisorError("ordinary binding differs from atomic branch binding")
            self.binding = persisted_binding
        self.authorization: dict | None = None
        authorization_path = operation_dir / "RELEASE_AUTHORIZATION.json"
        if authorization_path.exists():
            self.authorization = self._validate_authorization(_read_json(authorization_path))
        self.workload_name = ""
        self.workload_uid = ""
        self.cluster_name = ""
        self.cluster_uid = ""
        self.pods: dict[str, str] = {}
        self.runtime_images: dict[str, str] = {}
        self.current_pods: set[str] = set()
        self.current_runtime_images: set[str] = set()
        self.current_gpus = 0
        self.topology_observed = False
        self.allocation_at: datetime | None = None
        self.started_at: datetime | None = None
        self.terminal_status = ""
        self.terminal_observed_at: datetime | None = None
        self.delete_requested_at: datetime | None = None
        self.delete_reason = ""
        self.peak_gpus = 0
        self.zero_scan_at: datetime | None = None
        self.drift_zero_scan_at: datetime | None = None
        self.state_revision = 0
        self.last_fault_class = ""
        self.last_fault_at: datetime | None = None
        state_path = operation_dir / "STATE.json"
        if state_path.exists():
            self._restore_state(_read_json(state_path))
        self.result: dict | None = None
        result_path = operation_dir / "RESULT.json"
        if result_path.exists():
            self.result = self._validate_result(_read_json(result_path))
        self._reload_delete_intent()

    def _reload_delete_intent(self) -> dict | None:
        path = self.operation_dir / "DELETE_INTENT.json"
        if not path.exists():
            return None
        delete_intent = _validate_seal(_read_json(path), DELETE_INTENT_SCHEMA)
        if (
            self.binding is None
            or self.authorization is None
            or delete_intent.get("binding_sha256") != self.binding["sha256"]
            or delete_intent.get("authorization_sha256") != self.authorization["sha256"]
            or delete_intent.get("rayjob_name") != self.root_name
            or delete_intent.get("rayjob_uid") != self.binding["rayjob_uid"]
            or delete_intent.get("reason") not in RELEASE_REASONS
            or delete_intent.get("status") != "one_exact_uid_delete_may_be_attempted"
            or delete_intent.get("retry_policy") != "same_exact_uid_cas_until_release_deadline"
        ):
            raise SupervisorError("exact UID delete intent is invalid")
        self.delete_reason = delete_intent["reason"]
        self.delete_requested_at = _parse_time(delete_intent["written_at"])
        return delete_intent

    @classmethod
    def arm(
        cls,
        operation_dir: Path,
        backend: ClusterBackend,
        *,
        plan: dict,
        request: dict,
        manifest: dict,
        observer_pid: int,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        sleep: Callable[[float], None] | None = None,
        started_reader: Callable[[], dict | None] = lambda: None,
        process_alive: Callable[[int], bool] = lambda pid: _process_alive(pid),
        observer_receipt_sha256: str | None = None,
    ) -> FourNodeReleaseSupervisor:
        validate_exact_candidate(plan, request, manifest)
        if type(observer_pid) is not int or observer_pid < 1 or not process_alive(observer_pid):
            raise SupervisorError("release supervisor process is not live before arm")
        if observer_receipt_sha256 is None:
            observer_receipt_sha256 = "sha256:" + digest(
                {"schema": SCHEMA + ":inprocess_observer", "observer_pid": observer_pid}
            )
        if re.fullmatch(r"sha256:[a-f0-9]{64}", observer_receipt_sha256) is None:
            raise SupervisorError("release supervisor readiness receipt hash is invalid")
        operation_dir.mkdir(mode=0o700, parents=False, exist_ok=False)
        _fsync_dir(operation_dir.parent)
        _write_once(operation_dir / "PLAN.json", plan)
        _write_once(operation_dir / "REQUEST.json", request)
        _write_once(operation_dir / "RENDERED_RAYJOB.json", manifest)
        name = manifest["metadata"]["name"]
        if backend.get_rayjob(name) is not None:
            raise SupervisorError("exact rendered RayJob name already exists before arm")
        armed = _seal(
            {
                "schema": ARMED_SCHEMA,
                "status": "armed_before_external_create",
                "context": CONTEXT,
                "namespace": NAMESPACE,
                "rayjob_name": name,
                "jobs_api_run_id": manifest["metadata"]["annotations"]["fleet.ai/run-id"],
                "run_dir": RUN_DIR,
                "image": IMAGE,
                "nodes": NODES,
                "gpus_per_node": GPUS_PER_NODE,
                "total_gpus": TOTAL_GPUS,
                "plan_sha256": "sha256:" + digest(plan),
                "request_sha256": "sha256:" + digest(request),
                "manifest_sha256": "sha256:" + digest(manifest),
                "armed_at": _stamp(clock()),
                "observer_pid": observer_pid,
                "observer_receipt_sha256": observer_receipt_sha256,
                "exact_name_absent_before_arm": True,
            }
        )
        _write_once(operation_dir / "ARMED.json", armed)
        delegation = _seal(
            {
                "schema": DELEGATION_SCHEMA,
                "status": "derive_exact_uid_authorization_after_validated_binding_only",
                "armed_sha256": armed["sha256"],
                "context": CONTEXT,
                "namespace": NAMESPACE,
                "jobs_api_run_id": armed["jobs_api_run_id"],
                "rayjob_name": name,
                "run_dir": RUN_DIR,
                "image": IMAGE,
                "nodes": NODES,
                "gpus_per_node": GPUS_PER_NODE,
                "total_gpus": TOTAL_GPUS,
                "plan_sha256": armed["plan_sha256"],
                "request_sha256": armed["request_sha256"],
                "manifest_sha256": armed["manifest_sha256"],
                "authorized_reasons": sorted(RELEASE_REASONS),
                "derivation_rule": DELEGATION_DERIVATION_RULE,
                "delete_route": "root_rayjob_uid_cas_only",
                "delegated_at": _stamp(clock()),
            }
        )
        _write_once(operation_dir / "RELEASE_DELEGATION.json", delegation)
        drift_delegation = _seal(
            {
                "schema": DRIFT_DELEGATION_SCHEMA,
                "status": "cleanup_only_never_accept",
                "armed_sha256": armed["sha256"],
                "context": CONTEXT,
                "namespace": NAMESPACE,
                "jobs_api_run_id": armed["jobs_api_run_id"],
                "rayjob_name": name,
                "manifest_sha256": armed["manifest_sha256"],
                "one_shot_intent_required": True,
                "authorized_reason": "observation_contract_defect",
                "delete_route": "root_rayjob_uid_cas_only",
                "acceptance_forbidden": True,
                "delegated_at": _stamp(clock()),
            }
        )
        _write_once(operation_dir / "DRIFT_CLEANUP_DELEGATION.json", drift_delegation)
        return cls(
            operation_dir,
            backend,
            clock=clock,
            sleep=sleep,
            started_reader=started_reader,
            process_alive=process_alive,
        )

    def write_create_intent(self) -> dict:
        """Fsync a one-shot external-create intent immediately before its POST."""
        if not self.process_alive(self.armed["observer_pid"]):
            raise SupervisorError("release supervisor process died before create intent")
        if self.backend.get_rayjob(self.root_name) is not None:
            raise SupervisorError("exact rendered RayJob name appeared before create intent")
        value = _seal(
            {
                "schema": INTENT_SCHEMA,
                "status": "external_create_may_be_attempted_once",
                "armed_sha256": self.armed["sha256"],
                "observer_pid": self.armed["observer_pid"],
                "observer_receipt_sha256": self.armed["observer_receipt_sha256"],
                "rayjob_name": self.root_name,
                "jobs_api_run_id": self.run_id,
                "manifest_sha256": self.armed["manifest_sha256"],
                "written_at": _stamp(self.clock()),
                "uncertain_create_policy": "reconcile_this_exact_name_never_retry_or_discover",
            }
        )
        _write_once(self.operation_dir / "CREATE_INTENT.json", value)
        return value

    def _validate_live_root(self, root: dict) -> tuple[str, datetime]:
        expected = deepcopy(self.manifest)
        actual = deepcopy(root)
        try:
            expected_suspend = expected["spec"].get("suspend")
            actual_suspend = actual["spec"].get("suspend")
        except (KeyError, TypeError) as exc:
            raise SupervisorError("rendered RayJob surface is malformed") from exc
        if type(actual_suspend) is not bool:
            raise SupervisorError("rendered RayJob suspend state is malformed")
        # Kueue owns this one mutable admission bit.  Normalize only this known
        # controller field; every other effective runtime field remains an
        # exact comparison after ordinary API-server defaults are removed.
        actual["spec"]["suspend"] = expected_suspend
        try:
            from cyber_post_train.direct_submit import _assert_exact_rayjob_runtime_surface

            _assert_exact_rayjob_runtime_surface(actual, expected)
        except JobsError as exc:
            raise SupervisorError("rendered RayJob runtime surface changed") from exc
        name, uid, created = _metadata(root, "RayJob")
        if name != self.root_name or root["metadata"].get("namespace") != NAMESPACE:
            raise SupervisorError("exact RayJob root identity changed")
        return uid, created

    def _drift_cleanup_identity(self, root: dict) -> tuple[str, datetime]:
        name, uid, created = _metadata(root, "RayJob")
        if str(UUID(uid)) != uid:
            raise SupervisorError("drift cleanup exact root UID is not canonical")
        metadata = root.get("metadata", {})
        labels = metadata.get("labels")
        annotations = metadata.get("annotations")
        if (
            name != self.root_name
            or metadata.get("namespace") != NAMESPACE
            or not isinstance(labels, dict)
            or not isinstance(annotations, dict)
            or labels.get("fleet.ai/run-id") != self.run_id
            or annotations.get("fleet.ai/run-id") != self.run_id
            or labels.get("fleet.ai/run-name") != RUN_NAME
            or annotations.get("fleet.ai/run-dir") != RUN_DIR
        ):
            raise SupervisorError("drift cleanup exact root identity is not proven")
        return uid, created

    def _validate_branch_decision(self, value: dict) -> dict:
        _validate_seal(value, BRANCH_SCHEMA)
        intent = _validate_create_intent(
            _read_json(self.operation_dir / "CREATE_INTENT.json"), self.armed
        )
        expected = {
            "schema",
            "status",
            "armed_sha256",
            "intent_sha256",
            "rayjob_name",
            "rayjob_uid",
            "branch",
            "binding",
            "binding_sha256",
            "selected_at",
            "sha256",
        }
        branch = value.get("branch")
        binding = value.get("binding")
        if (
            set(value) != expected
            or value.get("status") != "exact_uid_branch_and_binding_selected"
            or value.get("armed_sha256") != self.armed["sha256"]
            or value.get("intent_sha256") != intent["sha256"]
            or value.get("rayjob_name") != self.root_name
            or branch not in {"ordinary", "cleanup_only"}
            or not isinstance(binding, dict)
        ):
            raise SupervisorError("release branch decision is invalid")
        if branch == "ordinary":
            binding = _validate_binding_for_authorization(
                binding, self.armed, self.delegation, intent
            )
        else:
            binding = _validate_drift_binding(binding, self.armed, intent, self.drift_delegation)
        if (
            value.get("binding_sha256") != binding["sha256"]
            or value.get("rayjob_uid") != binding["rayjob_uid"]
        ):
            raise SupervisorError("release branch decision binding is invalid")
        _parse_time(value.get("selected_at"))
        return value

    def _select_branch(self, branch: str, binding: dict) -> dict:
        if branch not in {"ordinary", "cleanup_only"}:
            raise SupervisorError("release branch is invalid")
        if branch == "ordinary":
            intent = _validate_create_intent(
                _read_json(self.operation_dir / "CREATE_INTENT.json"), self.armed
            )
            binding = _validate_binding_for_authorization(
                binding, self.armed, self.delegation, intent
            )
        else:
            intent = _validate_create_intent(
                _read_json(self.operation_dir / "CREATE_INTENT.json"), self.armed
            )
            binding = _validate_drift_binding(binding, self.armed, intent, self.drift_delegation)
        conflicting = (
            ["DRIFT_CLEANUP_BOUND.json", "DRIFT_CLEANUP_AUTHORIZATION.json"]
            if branch == "ordinary"
            else ["BOUND.json", "RELEASE_AUTHORIZATION.json", "OBSERVER_TAKEOVER.json"]
        )
        if any((self.operation_dir / name).exists() for name in conflicting):
            raise SupervisorError("ordinary and cleanup-only branches are mutually exclusive")
        value = _seal(
            {
                "schema": BRANCH_SCHEMA,
                "status": "exact_uid_branch_and_binding_selected",
                "armed_sha256": self.armed["sha256"],
                "intent_sha256": intent["sha256"],
                "rayjob_name": self.root_name,
                "rayjob_uid": binding["rayjob_uid"],
                "branch": branch,
                "binding": binding,
                "binding_sha256": binding["sha256"],
                "selected_at": _stamp(self.clock()),
            }
        )
        path = self.operation_dir / "BRANCH_DECISION.json"
        try:
            _write_once(path, value)
        except FileExistsError:
            existing = self._validate_branch_decision(_read_json(path))
            if (
                existing["branch"] != branch
                or existing["rayjob_uid"] != binding["rayjob_uid"]
                or existing["binding"].get("rayjob_created_at") != binding["rayjob_created_at"]
            ):
                raise SupervisorError(
                    "ordinary and cleanup-only branches are mutually exclusive"
                ) from None
            value = existing
        self.branch_decision = value
        if branch == "ordinary":
            self.binding = value["binding"]
        return value

    def _require_branch(self, branch: str, uid: str) -> dict:
        value = self._validate_branch_decision(
            _read_json(self.operation_dir / "BRANCH_DECISION.json")
        )
        if value.get("rayjob_uid") != uid or value.get("branch") != branch:
            raise SupervisorError("release branch decision is invalid")
        return value

    def cleanup_rejected_exact_root(self, returned_root: dict) -> dict:
        """Delete, but never accept, one exact post-intent root with surface drift."""
        intent = _validate_create_intent(
            _read_json(self.operation_dir / "CREATE_INTENT.json"), self.armed
        )
        try:
            self._validate_live_root(returned_root)
        except SupervisorError:
            pass
        else:
            raise SupervisorError("drift cleanup cannot replace ordinary exact acceptance")
        returned_uid, _ = self._drift_cleanup_identity(returned_root)
        live = self.backend.get_rayjob(self.root_name)
        if live is None:
            raise SupervisorError("drift cleanup exact root disappeared before UID binding")
        live_uid, created = self._drift_cleanup_identity(live)
        if live_uid != returned_uid:
            raise SupervisorError("drift cleanup returned and persisted UIDs differ")
        binding = _seal(
            {
                "schema": DRIFT_BINDING_SCHEMA,
                "status": "cleanup_only_exact_uid_not_accepted",
                "armed_sha256": self.armed["sha256"],
                "intent_sha256": intent["sha256"],
                "delegation_sha256": self.drift_delegation["sha256"],
                "context": CONTEXT,
                "namespace": NAMESPACE,
                "jobs_api_run_id": self.run_id,
                "rayjob_name": self.root_name,
                "rayjob_uid": live_uid,
                "rayjob_created_at": _stamp(created),
                "manifest_sha256": self.armed["manifest_sha256"],
                "surface_validation": "rejected",
                "acceptance_forbidden": True,
                "bound_at": _stamp(self.clock()),
            }
        )
        decision = self._select_branch("cleanup_only", binding)
        binding = decision["binding"]
        binding_path = self.operation_dir / "DRIFT_CLEANUP_BOUND.json"
        try:
            _write_once(binding_path, binding)
        except FileExistsError:
            binding = _validate_drift_binding(
                _read_json(binding_path), self.armed, intent, self.drift_delegation
            )
            if binding["rayjob_uid"] != live_uid:
                raise SupervisorError("drift cleanup binding selected another UID") from None
        binding = _validate_drift_binding(binding, self.armed, intent, self.drift_delegation)
        return self._persist_drift_cleanup_chain(binding)

    def _persist_drift_cleanup_chain(self, binding: dict) -> dict:
        intent = _validate_create_intent(
            _read_json(self.operation_dir / "CREATE_INTENT.json"), self.armed
        )
        binding = _validate_drift_binding(binding, self.armed, intent, self.drift_delegation)
        self._require_branch("cleanup_only", binding["rayjob_uid"])
        binding_path = self.operation_dir / "DRIFT_CLEANUP_BOUND.json"
        try:
            _write_once(binding_path, binding)
        except FileExistsError:
            persisted = _validate_drift_binding(
                _read_json(binding_path), self.armed, intent, self.drift_delegation
            )
            if persisted != binding:
                raise SupervisorError("drift cleanup binding changed after selection") from None
        authorization_path = self.operation_dir / "DRIFT_CLEANUP_AUTHORIZATION.json"
        authorization = _seal(
            {
                "schema": DRIFT_AUTHORIZATION_SCHEMA,
                "status": "cleanup_only_exact_root_uid_delete",
                "binding_sha256": binding["sha256"],
                "delegation_sha256": self.drift_delegation["sha256"],
                "rayjob_name": self.root_name,
                "rayjob_uid": binding["rayjob_uid"],
                "authorized_reason": "observation_contract_defect",
                "delete_route": "root_rayjob_uid_cas_only",
                "acceptance_forbidden": True,
                "authorized_at": _stamp(self.clock()),
            }
        )
        try:
            _write_once(authorization_path, authorization)
        except FileExistsError:
            authorization = _validate_drift_authorization(
                _read_json(authorization_path), binding, self.drift_delegation
            )
        authorization = _validate_drift_authorization(authorization, binding, self.drift_delegation)

        delete_path = self.operation_dir / "DRIFT_CLEANUP_DELETE_INTENT.json"
        delete_intent = _seal(
            {
                "schema": DRIFT_DELETE_INTENT_SCHEMA,
                "status": "one_cleanup_only_UID_delete_may_be_attempted",
                "binding_sha256": binding["sha256"],
                "authorization_sha256": authorization["sha256"],
                "rayjob_name": self.root_name,
                "rayjob_uid": binding["rayjob_uid"],
                "reason": "observation_contract_defect",
                "written_at": _stamp(self.clock()),
                "retry_policy": "same_exact_uid_cas_until_release_deadline",
                "acceptance_forbidden": True,
            }
        )
        try:
            _write_once(delete_path, delete_intent)
        except FileExistsError:
            existing = _validate_seal(_read_json(delete_path), DRIFT_DELETE_INTENT_SCHEMA)
            expected = {
                key: value
                for key, value in delete_intent.items()
                if key not in {"written_at", "sha256"}
            }
            if set(existing) != {"written_at", "sha256", *expected} or any(
                existing.get(key) != value for key, value in expected.items()
            ):
                raise SupervisorError("drift cleanup delete intent is invalid") from None
            _parse_time(existing.get("written_at"))
            delete_intent = existing
        return {
            "drift_cleanup_binding_sha256": binding["sha256"],
            "drift_cleanup_authorization_sha256": authorization["sha256"],
            "drift_cleanup_delete_intent_sha256": delete_intent["sha256"],
            "rayjob_uid": binding["rayjob_uid"],
            "accepted": False,
        }

    def resume_selected_cleanup_branch(self) -> dict | None:
        """Complete a durably selected cleanup branch after any local crash."""
        branch_path = self.operation_dir / "BRANCH_DECISION.json"
        if not branch_path.exists():
            return None
        decision = self._validate_branch_decision(_read_json(branch_path))
        if decision["branch"] != "cleanup_only":
            return None
        return self._persist_drift_cleanup_chain(decision["binding"])

    def drive_drift_cleanup_delete(self) -> dict | None:
        """Re-drive only the one journaled cleanup UID-CAS root deletion."""
        binding, authorization, delete_intent = self._drift_cleanup_chain()
        root = self.backend.get_rayjob(self.root_name)
        if root is None:
            return None
        try:
            live_uid, _ = self._drift_cleanup_identity(root)
        except SupervisorError:
            return self._finish_drift_cleanup(
                binding,
                authorization,
                delete_intent,
                release_confirmed=False,
                final={"reason": "exact_name_identity_mismatch"},
            )
        if live_uid != binding["rayjob_uid"]:
            return self._finish_drift_cleanup(
                binding,
                authorization,
                delete_intent,
                release_confirmed=False,
                final={"reason": "exact_name_uid_mismatch"},
            )
        self.backend.delete_rayjob_uid_foreground(self.root_name, binding["rayjob_uid"])
        return None

    def _drift_cleanup_chain(self) -> tuple[dict, dict, dict]:
        delete_path = self.operation_dir / "DRIFT_CLEANUP_DELETE_INTENT.json"
        if not delete_path.exists():
            raise SupervisorError("drift cleanup delete intent is absent")
        intent = _validate_create_intent(
            _read_json(self.operation_dir / "CREATE_INTENT.json"), self.armed
        )
        binding = _validate_drift_binding(
            _read_json(self.operation_dir / "DRIFT_CLEANUP_BOUND.json"),
            self.armed,
            intent,
            self.drift_delegation,
        )
        self._require_branch("cleanup_only", binding["rayjob_uid"])
        authorization = _validate_drift_authorization(
            _read_json(self.operation_dir / "DRIFT_CLEANUP_AUTHORIZATION.json"),
            binding,
            self.drift_delegation,
        )
        delete_intent = _validate_seal(_read_json(delete_path), DRIFT_DELETE_INTENT_SCHEMA)
        expected = {
            "status": "one_cleanup_only_UID_delete_may_be_attempted",
            "binding_sha256": binding["sha256"],
            "authorization_sha256": authorization["sha256"],
            "rayjob_name": self.root_name,
            "rayjob_uid": binding["rayjob_uid"],
            "reason": "observation_contract_defect",
            "retry_policy": "same_exact_uid_cas_until_release_deadline",
            "acceptance_forbidden": True,
        }
        if set(delete_intent) != {"schema", *expected, "written_at", "sha256"} or any(
            delete_intent.get(key) != value for key, value in expected.items()
        ):
            raise SupervisorError("drift cleanup delete intent is invalid")
        _parse_time(delete_intent.get("written_at"))
        return binding, authorization, delete_intent

    def _drift_resource_path(self, kind: str, uid: str) -> Path:
        if kind == "Workload":
            name = "DRIFT_CLEANUP_WORKLOAD.json"
        elif kind == "RayCluster":
            name = "DRIFT_CLEANUP_RAYCLUSTER.json"
        elif kind == "Pod":
            name = f"DRIFT_CLEANUP_POD_{str(UUID(uid))}.json"
        else:
            raise SupervisorError("drift cleanup resource kind is invalid")
        return self.operation_dir / name

    def _persist_drift_resource(
        self,
        binding: dict,
        resource: dict,
        *,
        kind: str,
        owner_kind: str,
        owner_name: str,
        owner_uid: str,
    ) -> dict:
        name, uid, _ = _metadata(resource, kind)
        if not _owned_by(
            resource,
            kind=owner_kind,
            name=owner_name,
            uid=owner_uid,
        ):
            raise SupervisorError(f"drift cleanup {kind} owner is not exact")
        value = _seal(
            {
                "schema": DRIFT_RESOURCE_SCHEMA,
                "status": "cleanup_only_observed_exact_owned_resource",
                "binding_sha256": binding["sha256"],
                "kind": kind,
                "name": name,
                "uid": uid,
                "owner_kind": owner_kind,
                "owner_name": owner_name,
                "owner_uid": owner_uid,
                "observed_at": _stamp(self.clock()),
            }
        )
        path = self._drift_resource_path(kind, uid)
        try:
            _write_once(path, value)
        except FileExistsError:
            existing = _validate_seal(_read_json(path), DRIFT_RESOURCE_SCHEMA)
            expected = {
                key: item for key, item in value.items() if key not in {"observed_at", "sha256"}
            }
            if any(existing.get(key) != item for key, item in expected.items()):
                raise SupervisorError(f"drift cleanup {kind} identity changed") from None
            _parse_time(existing.get("observed_at"))
            value = existing
        return value

    def _load_drift_resources(self, binding: dict, kind: str) -> list[dict]:
        if kind == "Workload":
            paths = [self.operation_dir / "DRIFT_CLEANUP_WORKLOAD.json"]
        elif kind == "RayCluster":
            paths = [self.operation_dir / "DRIFT_CLEANUP_RAYCLUSTER.json"]
        elif kind == "Pod":
            paths = sorted(self.operation_dir.glob("DRIFT_CLEANUP_POD_*.json"))
        else:
            raise SupervisorError("drift cleanup resource kind is invalid")
        values = []
        for path in paths:
            if not path.exists():
                continue
            value = _validate_seal(_read_json(path), DRIFT_RESOURCE_SCHEMA)
            try:
                uid_is_canonical = str(UUID(value.get("uid"))) == value.get("uid")
                owner_uid_is_canonical = str(UUID(value.get("owner_uid"))) == value.get("owner_uid")
            except (TypeError, ValueError, AttributeError):
                uid_is_canonical = False
                owner_uid_is_canonical = False
            if (
                value.get("status") != "cleanup_only_observed_exact_owned_resource"
                or value.get("binding_sha256") != binding["sha256"]
                or value.get("kind") != kind
                or not isinstance(value.get("name"), str)
                or not value["name"]
                or not uid_is_canonical
                or not isinstance(value.get("owner_kind"), str)
                or not isinstance(value.get("owner_name"), str)
                or not owner_uid_is_canonical
            ):
                raise SupervisorError("drift cleanup resource receipt is invalid")
            if path != self._drift_resource_path(kind, value["uid"]):
                raise SupervisorError("drift cleanup resource receipt path is invalid")
            _parse_time(value.get("observed_at"))
            values.append(value)
        if kind != "Pod" and len(values) > 1:
            raise SupervisorError(f"drift cleanup found multiple persisted {kind}s")
        return values

    def _pods_by_run_identity(self) -> list[dict]:
        pods = self.backend.list_pods_by_run_identity(self.run_id, RUN_NAME)
        for pod in pods:
            _metadata(pod, "Pod")
            metadata = pod.get("metadata", {})
            labels = metadata.get("labels") if isinstance(metadata, dict) else None
            if (
                not isinstance(labels, dict)
                or labels.get("fleet.ai/run-id") != self.run_id
                or labels.get("fleet.ai/run-name") != RUN_NAME
            ):
                raise SupervisorError("run-identity Pod census includes another candidate")
        return pods

    def _fresh_drift_cleanup_observation(self, binding: dict) -> tuple[bool, dict]:
        root = self.backend.get_rayjob(self.root_name)
        if root is not None:
            root_uid, _ = self._drift_cleanup_identity(root)
            if root_uid != binding["rayjob_uid"]:
                raise SupervisorError("drift cleanup root UID changed during reconciliation")
        workloads = self.backend.list_workloads_by_job_uid(binding["rayjob_uid"])
        persisted_workloads = self._load_drift_resources(binding, "Workload")
        if persisted_workloads:
            persisted = persisted_workloads[0]
            exact = self.backend.get_workload(persisted["name"])
            if exact is not None and all(
                _metadata(item, "Workload")[1] != persisted["uid"] for item in workloads
            ):
                workloads.append(exact)
        if len(workloads) > 1:
            raise SupervisorError("drift cleanup found multiple owned Workloads")
        for workload in workloads:
            self._persist_drift_resource(
                binding,
                workload,
                kind="Workload",
                owner_kind="RayJob",
                owner_name=self.root_name,
                owner_uid=binding["rayjob_uid"],
            )
        clusters = self.backend.list_rayclusters_by_rayjob_uid(
            self.root_name, binding["rayjob_uid"]
        )
        persisted_clusters = self._load_drift_resources(binding, "RayCluster")
        if persisted_clusters:
            persisted = persisted_clusters[0]
            exact = self.backend.get_raycluster(persisted["name"])
            if exact is not None and all(
                _metadata(item, "RayCluster")[1] != persisted["uid"] for item in clusters
            ):
                clusters.append(exact)
        if len(clusters) > 1:
            raise SupervisorError("drift cleanup found multiple owned RayClusters")
        pods = self._pods_by_run_identity()
        if clusters:
            cluster_name, cluster_uid, _ = _metadata(clusters[0], "RayCluster")
            self._persist_drift_resource(
                binding,
                clusters[0],
                kind="RayCluster",
                owner_kind="RayJob",
                owner_name=self.root_name,
                owner_uid=binding["rayjob_uid"],
            )
        elif persisted_clusters:
            cluster_name = persisted_clusters[0]["name"]
            cluster_uid = persisted_clusters[0]["uid"]
        else:
            cluster_name = ""
            cluster_uid = ""
        if cluster_name:
            pods = _merge_unique(
                pods,
                self.backend.list_pods(cluster_name, cluster_uid),
                "Pod",
            )
            for persisted in self._load_drift_resources(binding, "Pod"):
                exact = self.backend.get_pod(persisted["name"])
                if exact is not None:
                    pods = _merge_unique(pods, [exact], "Pod")
            for pod in pods:
                self._persist_drift_resource(
                    binding,
                    pod,
                    kind="Pod",
                    owner_kind="RayCluster",
                    owner_name=cluster_name,
                    owner_uid=cluster_uid,
                )
        if len(pods) > NODES:
            raise SupervisorError("drift cleanup found too many candidate Pods")
        counts = {
            "root": int(root is not None),
            "workloads": len(workloads),
            "rayclusters": len(clusters),
            "pods": len(pods),
            "active_gpus": sum(_pod_gpu_count(pod) for pod in pods),
        }
        return not any(counts.values()), counts

    def _finish_drift_cleanup(
        self,
        binding: dict,
        authorization: dict,
        delete_intent: dict,
        *,
        release_confirmed: bool,
        final: dict,
    ) -> dict:
        _validate_final_census(final, confirmed=release_confirmed)
        value = _seal(
            {
                "schema": DRIFT_RESULT_SCHEMA,
                "status": (
                    "cleanup_only_full_release_confirmed"
                    if release_confirmed
                    else "cleanup_only_release_uncertain"
                ),
                "accepted": False,
                "release_confirmed": release_confirmed,
                "binding_sha256": binding["sha256"],
                "authorization_sha256": authorization["sha256"],
                "delete_intent_sha256": delete_intent["sha256"],
                "rayjob_name": self.root_name,
                "rayjob_uid": binding["rayjob_uid"],
                "fresh_final_relist": final,
                "finished_at": _stamp(self.clock()),
            }
        )
        path = self.operation_dir / "DRIFT_CLEANUP_RESULT.json"
        try:
            _write_once(path, value)
        except FileExistsError:
            existing = _validate_seal(_read_json(path), DRIFT_RESULT_SCHEMA)
            if set(existing) != set(value) or any(
                existing.get(key) != item
                for key, item in value.items()
                if key not in {"fresh_final_relist", "finished_at", "sha256"}
            ):
                raise SupervisorError("drift cleanup result differs from exact UID chain") from None
            _validate_final_census(
                existing.get("fresh_final_relist"),
                confirmed=existing.get("release_confirmed") is True,
            )
            _parse_time(existing.get("finished_at"))
            value = existing
        return value

    def reconcile_drift_cleanup_absence(self) -> dict | None:
        """Observe the full cleanup-only owner chain; never repeat its root DELETE."""
        delete_path = self.operation_dir / "DRIFT_CLEANUP_DELETE_INTENT.json"
        if not delete_path.exists():
            return None
        binding, authorization, delete_intent = self._drift_cleanup_chain()
        result_path = self.operation_dir / "DRIFT_CLEANUP_RESULT.json"
        if result_path.exists():
            return self._finish_drift_cleanup(
                binding,
                authorization,
                delete_intent,
                release_confirmed=bool(_read_json(result_path).get("release_confirmed")),
                final=_read_json(result_path).get("fresh_final_relist", {}),
            )
        released, final = self._fresh_drift_cleanup_observation(binding)
        if released:
            now = self.clock()
            if self.drift_zero_scan_at is None:
                self.drift_zero_scan_at = now
                return None
            if now < self.drift_zero_scan_at + timedelta(seconds=POLL_SECONDS):
                return None
            return self._finish_drift_cleanup(
                binding,
                authorization,
                delete_intent,
                release_confirmed=True,
                final=final,
            )
        self.drift_zero_scan_at = None
        if self.clock() >= _parse_time(delete_intent["written_at"]) + timedelta(
            seconds=DELETE_CONFIRM_SECONDS
        ):
            return self._finish_drift_cleanup(
                binding,
                authorization,
                delete_intent,
                release_confirmed=False,
                final=final,
            )
        return None

    def record_drift_cleanup_uncertain(self) -> dict:
        """Seal the final full-chain census after the caller's bounded grace."""
        binding, authorization, delete_intent = self._drift_cleanup_chain()
        _, final = self._fresh_drift_cleanup_observation(binding)
        return self._finish_drift_cleanup(
            binding,
            authorization,
            delete_intent,
            release_confirmed=False,
            final=final,
        )

    def reconcile_exact(self) -> dict | None:
        """Adopt only the journaled exact name after a certain or uncertain create."""
        _validate_create_intent(_read_json(self.operation_dir / "CREATE_INTENT.json"), self.armed)
        root = self.backend.get_rayjob(self.root_name)
        if self.binding is not None:
            if root is not None:
                self._validate_bound_root(root)
            return self.binding
        if root is None:
            return None
        return self.bind_created(root)

    def bind_created(self, root: dict) -> dict:
        """Persist the exact create-returned UID for the already-armed name."""
        intent = _validate_create_intent(
            _read_json(self.operation_dir / "CREATE_INTENT.json"), self.armed
        )
        if self.binding is None and (self.operation_dir / "BOUND.json").exists():
            self.binding = self._validate_binding(_read_json(self.operation_dir / "BOUND.json"))
        uid, created = self._validate_live_root(root)
        live = self.backend.get_rayjob(self.root_name)
        if live is None:
            raise SupervisorError("exact RayJob disappeared before ordinary UID binding")
        live_uid, live_created = self._validate_live_root(live)
        if live_uid != uid or live_created != created:
            raise SupervisorError("create-returned and persisted exact roots differ")
        if self.binding is not None:
            if (
                uid != self.binding["rayjob_uid"]
                or _stamp(created) != self.binding["rayjob_created_at"]
            ):
                raise SupervisorError("create-returned RayJob differs from exact UID binding")
            return self.binding
        binding = _seal(
            {
                "schema": BINDING_SCHEMA,
                "status": "bound_exact_uid",
                "armed_sha256": self.armed["sha256"],
                "intent_sha256": intent["sha256"],
                "context": CONTEXT,
                "namespace": NAMESPACE,
                "jobs_api_run_id": self.run_id,
                "rayjob_name": self.root_name,
                "rayjob_uid": uid,
                "rayjob_created_at": _stamp(created),
                "manifest_sha256": self.armed["manifest_sha256"],
                "bound_at": _stamp(self.clock()),
            }
        )
        decision = self._select_branch("ordinary", binding)
        binding = decision["binding"]
        try:
            _write_once(self.operation_dir / "BOUND.json", binding)
        except FileExistsError:
            existing = self._validate_binding(_read_json(self.operation_dir / "BOUND.json"))
            if existing["rayjob_uid"] != uid or existing["rayjob_created_at"] != _stamp(created):
                raise SupervisorError(
                    "concurrent exact UID binding selected another root"
                ) from None
            binding = existing
        self.binding = binding
        return binding

    def _validate_binding(self, value: dict) -> dict:
        intent = _validate_create_intent(
            _read_json(self.operation_dir / "CREATE_INTENT.json"), self.armed
        )
        binding = _validate_binding_for_authorization(value, self.armed, self.delegation, intent)
        self._require_branch("ordinary", binding["rayjob_uid"])
        return binding

    def _validate_authorization(self, value: dict) -> dict:
        if self.binding is None:
            raise SupervisorError("release authorization exists before exact UID binding")
        return _validate_authorization_value(value, self.binding, self.delegation)

    def authorize_bound_uid(self) -> dict:
        """Derive one UID-bound authorization from the sealed pre-create delegation."""
        if self.binding is None:
            raise SupervisorError("cannot authorize release before exact UID binding")
        persist_release_authorization(
            self.operation_dir,
            self.binding,
            authorized_at=self.clock(),
        )
        return self.reload_authorization()

    def reconcile_and_authorize_exact(self) -> dict | None:
        """Recover one exact create and make its delegated UID release-operable."""
        if self.binding is not None and self.authorization is None:
            # The atomic ordinary branch receipt already sealed the complete
            # strict binding. Derive its delegated release authority before a
            # later live-surface defect can interrupt observation.
            self.authorize_bound_uid()
        binding = self.reconcile_exact()
        if binding is None:
            return None
        if self.authorization is None:
            self.authorize_bound_uid()
        return binding

    def reload_authorization(self) -> dict:
        self.authorization = self._validate_authorization(
            _read_json(self.operation_dir / "RELEASE_AUTHORIZATION.json")
        )
        return self.authorization

    def _restore_state(self, value: dict) -> None:
        if self.binding is None:
            raise SupervisorError("observation state exists before exact UID binding")
        _validate_seal(value, STATE_SCHEMA)
        expected = {
            "schema",
            "status",
            "binding_sha256",
            "revision",
            "workload",
            "raycluster",
            "pods",
            "runtime_images",
            "allocated_at",
            "started_at",
            "terminal_status",
            "terminal_observed_at",
            "topology_and_images_observed",
            "peak_gpus",
            "last_fault_class",
            "last_fault_at",
            "updated_at",
            "sha256",
        }
        if (
            set(value) != expected
            or value.get("status") != "exact_uid_observation_state"
            or value.get("binding_sha256") != self.binding["sha256"]
            or type(value.get("revision")) is not int
            or value["revision"] < 1
        ):
            raise SupervisorError("observation state differs from exact UID binding")
        workload = value.get("workload", {})
        cluster = value.get("raycluster", {})
        pods = value.get("pods", {})
        images = value.get("runtime_images", {})
        if not all(isinstance(item, dict) for item in (workload, cluster, pods, images)):
            raise SupervisorError("observation state UID chain is malformed")
        if workload:
            self.workload_name, self.workload_uid = self._restore_identity(workload, "Workload")
        if cluster:
            self.cluster_name, self.cluster_uid = self._restore_identity(cluster, "RayCluster")
        if len(pods) > NODES or set(images) - set(pods):
            raise SupervisorError("observation state Pod chain is malformed")
        for name, uid in pods.items():
            self._restore_named_uid(name, uid, "Pod")
        for image_id in images.values():
            if not isinstance(image_id, str) or not image_id:
                raise SupervisorError("observation state image chain is malformed")
        self.pods = dict(pods)
        self.runtime_images = dict(images)
        self.allocation_at = self._optional_state_time(value.get("allocated_at"))
        self.started_at = self._optional_state_time(value.get("started_at"))
        self.terminal_observed_at = self._optional_state_time(value.get("terminal_observed_at"))
        self.last_fault_at = self._optional_state_time(value.get("last_fault_at"))
        _parse_time(value.get("updated_at"))
        self.terminal_status = value.get("terminal_status", "")
        self.last_fault_class = value.get("last_fault_class", "")
        if self.terminal_status not in {"", *TERMINAL_STATUSES.values()} or not isinstance(
            self.last_fault_class, str
        ):
            raise SupervisorError("observation state status is malformed")
        if self.started_at is not None and self.allocation_at is None:
            raise SupervisorError("observation state STARTED predates allocation")
        if self.started_at is not None and self.started_at < self.allocation_at:
            raise SupervisorError("observation state STARTED predates allocation")
        topology = value.get("topology_and_images_observed")
        peak = value.get("peak_gpus")
        if type(topology) is not bool or type(peak) is not int or not 0 <= peak <= TOTAL_GPUS:
            raise SupervisorError("observation state topology is malformed")
        self.topology_observed = topology
        self.peak_gpus = peak
        self.state_revision = value["revision"]

    @staticmethod
    def _restore_identity(value: dict, kind: str) -> tuple[str, str]:
        if set(value) != {"name", "uid"}:
            raise SupervisorError(f"observation state {kind} identity is malformed")
        FourNodeReleaseSupervisor._restore_named_uid(value["name"], value["uid"], kind)
        return value["name"], value["uid"]

    @staticmethod
    def _restore_named_uid(name: object, uid: object, kind: str) -> None:
        if not isinstance(name, str) or not name:
            raise SupervisorError(f"observation state {kind} identity is malformed")
        try:
            UUID(uid)
        except (TypeError, ValueError) as exc:
            raise SupervisorError(f"observation state {kind} identity is malformed") from exc

    @staticmethod
    def _optional_state_time(value: object) -> datetime | None:
        return None if value == "" else _parse_time(value)

    def _state_value(self) -> dict:
        if self.binding is None:
            raise SupervisorError("cannot persist observations before exact UID binding")
        return _seal(
            {
                "schema": STATE_SCHEMA,
                "status": "exact_uid_observation_state",
                "binding_sha256": self.binding["sha256"],
                "revision": self.state_revision + 1,
                "workload": (
                    {"name": self.workload_name, "uid": self.workload_uid}
                    if self.workload_name
                    else {}
                ),
                "raycluster": (
                    {"name": self.cluster_name, "uid": self.cluster_uid}
                    if self.cluster_name
                    else {}
                ),
                "pods": dict(sorted(self.pods.items())),
                "runtime_images": dict(sorted(self.runtime_images.items())),
                "allocated_at": _stamp(self.allocation_at) if self.allocation_at else "",
                "started_at": _stamp(self.started_at) if self.started_at else "",
                "terminal_status": self.terminal_status,
                "terminal_observed_at": (
                    _stamp(self.terminal_observed_at) if self.terminal_observed_at else ""
                ),
                "topology_and_images_observed": self.topology_observed,
                "peak_gpus": self.peak_gpus,
                "last_fault_class": self.last_fault_class,
                "last_fault_at": _stamp(self.last_fault_at) if self.last_fault_at else "",
                "updated_at": _stamp(self.clock()),
            }
        )

    @staticmethod
    def _merge_identity(
        current_name: str,
        current_uid: str,
        local_name: str,
        local_uid: str,
        kind: str,
    ) -> tuple[str, str]:
        if (
            current_name
            and local_name
            and (current_name, current_uid)
            != (
                local_name,
                local_uid,
            )
        ):
            raise SupervisorError(f"concurrent {kind} observation changed identity")
        return (current_name, current_uid) if current_name else (local_name, local_uid)

    @staticmethod
    def _merge_named_uids(current: dict[str, str], local: dict[str, str], kind: str) -> dict:
        merged = dict(current)
        reverse = {uid: name for name, uid in merged.items()}
        for name, uid in local.items():
            if (name in merged and merged[name] != uid) or (
                uid in reverse and reverse[uid] != name
            ):
                raise SupervisorError(f"concurrent {kind} observation changed identity")
            merged[name] = uid
            reverse[uid] = name
        return merged

    @staticmethod
    def _merge_mapping(current: dict[str, str], local: dict[str, str], kind: str) -> dict:
        merged = dict(current)
        for name, value in local.items():
            if name in merged and merged[name] != value:
                raise SupervisorError(f"concurrent {kind} observation changed")
            merged[name] = value
        return merged

    @staticmethod
    def _earliest(first: datetime | None, second: datetime | None) -> datetime | None:
        values = [value for value in (first, second) if value is not None]
        return min(values) if values else None

    def _merge_persisted_state(self, persisted: dict) -> None:
        """Merge monotonic evidence after another observer won the state lock."""
        local = {
            "workload": (self.workload_name, self.workload_uid),
            "raycluster": (self.cluster_name, self.cluster_uid),
            "pods": dict(self.pods),
            "runtime_images": dict(self.runtime_images),
            "allocation_at": self.allocation_at,
            "started_at": self.started_at,
            "terminal_status": self.terminal_status,
            "terminal_observed_at": self.terminal_observed_at,
            "topology_observed": self.topology_observed,
            "peak_gpus": self.peak_gpus,
            "last_fault_class": self.last_fault_class,
            "last_fault_at": self.last_fault_at,
        }
        self._restore_state(persisted)
        self.workload_name, self.workload_uid = self._merge_identity(
            self.workload_name,
            self.workload_uid,
            *local["workload"],
            "Workload",
        )
        self.cluster_name, self.cluster_uid = self._merge_identity(
            self.cluster_name,
            self.cluster_uid,
            *local["raycluster"],
            "RayCluster",
        )
        self.pods = self._merge_named_uids(self.pods, local["pods"], "Pod")
        self.runtime_images = self._merge_mapping(
            self.runtime_images,
            local["runtime_images"],
            "runtime image",
        )
        if len(self.pods) > NODES or set(self.runtime_images) - set(self.pods):
            raise SupervisorError("concurrent Pod observation is malformed")
        self.allocation_at = self._earliest(self.allocation_at, local["allocation_at"])
        self.started_at = self._earliest(self.started_at, local["started_at"])
        if self.terminal_status and local["terminal_status"]:
            if self.terminal_status != local["terminal_status"]:
                raise SupervisorError("concurrent terminal observations disagree")
        elif local["terminal_status"]:
            self.terminal_status = local["terminal_status"]
        self.terminal_observed_at = self._earliest(
            self.terminal_observed_at,
            local["terminal_observed_at"],
        )
        self.topology_observed = self.topology_observed or local["topology_observed"]
        self.peak_gpus = max(self.peak_gpus, local["peak_gpus"])
        local_fault_at = local["last_fault_at"]
        if local_fault_at is not None and (
            self.last_fault_at is None or local_fault_at > self.last_fault_at
        ):
            self.last_fault_class = local["last_fault_class"]
            self.last_fault_at = local_fault_at

    def _persist_state_locked(self) -> dict:
        path = self.operation_dir / "STATE.json"
        if path.exists():
            self._merge_persisted_state(_read_json(path))
        value = self._state_value()
        _write_atomic(path, value)
        self.state_revision = value["revision"]
        return value

    def _persist_state(self) -> dict:
        with _exclusive_state_lock(self.operation_dir):
            result_path = self.operation_dir / "RESULT.json"
            if result_path.exists():
                self.result = self._validate_result(_read_json(result_path))
                return _read_json(self.operation_dir / "STATE.json")
            return self._persist_state_locked()

    def _validate_result(self, value: dict) -> dict:
        if self.binding is None:
            raise SupervisorError("terminal result exists before exact UID binding")
        _validate_seal(value, RESULT_SCHEMA)
        expected = {
            "schema",
            "status",
            "reason",
            "release_confirmed",
            "binding_sha256",
            "rayjob_name",
            "rayjob_uid",
            "nodes",
            "gpus_per_node",
            "topology_and_images_observed",
            "ownership_state_sha256",
            "ownership_chain",
            "allocated_at",
            "started_at",
            "terminal_status",
            "delete_requested",
            "delete_reason",
            "fresh_final_relist",
            "active_gpus",
            "private_logs_read",
            "completed_at",
            "sha256",
        }
        if (
            set(value) != expected
            or value.get("binding_sha256") != self.binding["sha256"]
            or value.get("rayjob_name") != self.root_name
            or value.get("rayjob_uid") != self.binding["rayjob_uid"]
            or value.get("nodes") != NODES
            or value.get("gpus_per_node") != GPUS_PER_NODE
            or type(value.get("topology_and_images_observed")) is not bool
            or type(value.get("release_confirmed")) is not bool
            or type(value.get("delete_requested")) is not bool
            or value.get("private_logs_read") is not False
        ):
            raise SupervisorError("terminal result differs from exact UID binding")
        confirmed = value["release_confirmed"]
        if value.get("status") != ("released" if confirmed else "release_uncertain"):
            raise SupervisorError("terminal result status is invalid")
        allowed_uncertain_reasons = {
            "allocation_release_outer_bound_elapsed",
            "owned_resources_remain_after_release_deadline",
            "exact_name_uid_mismatch",
            "release_confirmation_outer_bound_elapsed",
        }
        if (confirmed and value.get("reason") != "fresh_exact_inventory_absent") or (
            not confirmed and value.get("reason") not in allowed_uncertain_reasons
        ):
            raise SupervisorError("terminal result reason is invalid")
        final = _validate_final_census(value.get("fresh_final_relist"), confirmed=confirmed)
        active_gpus = value.get("active_gpus")
        if confirmed:
            if type(active_gpus) is not int or active_gpus != 0:
                raise SupervisorError("confirmed terminal result has active GPUs")
        elif (active_gpus is not None and type(active_gpus) is not int) or active_gpus != final.get(
            "active_gpus"
        ):
            raise SupervisorError("uncertain terminal result GPU count is invalid")
        if value.get("terminal_status") not in {"", *TERMINAL_STATUSES.values()}:
            raise SupervisorError("terminal result RayJob status is invalid")
        delete_reason = value.get("delete_reason")
        if not isinstance(delete_reason, str) or (
            delete_reason and delete_reason not in RELEASE_REASONS
        ):
            raise SupervisorError("terminal result delete reason is invalid")
        if bool(delete_reason) != value["delete_requested"]:
            raise SupervisorError("terminal result delete evidence is inconsistent")
        delete_path = self.operation_dir / "DELETE_INTENT.json"
        if value["delete_requested"]:
            if not delete_path.exists():
                raise SupervisorError("terminal result delete intent is missing")
            delete_intent = self._reload_delete_intent()
            if delete_intent is None or delete_intent["reason"] != delete_reason:
                raise SupervisorError("terminal result delete intent is inconsistent")
        elif delete_path.exists():
            raise SupervisorError("terminal result omitted its durable delete intent")
        for key in ("allocated_at", "started_at"):
            timestamp = value.get(key)
            if timestamp != "":
                _parse_time(timestamp)
        _parse_time(value.get("completed_at"))
        state = _validate_seal(_read_json(self.operation_dir / "STATE.json"), STATE_SCHEMA)
        if (
            value.get("ownership_state_sha256") != state["sha256"]
            or value.get("allocated_at") != state.get("allocated_at")
            or value.get("started_at") != state.get("started_at")
            or value.get("terminal_status") != state.get("terminal_status")
            or value.get("topology_and_images_observed")
            != state.get("topology_and_images_observed")
        ):
            raise SupervisorError("terminal result differs from its sealed observation state")
        state_cluster = state.get("raycluster", {})
        ownership_chain = value.get("ownership_chain")
        expected_chain = {
            "workload": state.get("workload", {}),
            "raycluster": (state_cluster if state_cluster else {"name": "", "uid": ""}),
            "pods": state.get("pods", {}),
            "runtime_images": state.get("runtime_images", {}),
        }
        if ownership_chain != expected_chain:
            raise SupervisorError("terminal result ownership chain is invalid")
        return value

    def _validate_bound_root(self, root: dict) -> str:
        if self.binding is None:
            raise SupervisorError("supervision has no exact UID binding")
        uid, created = self._validate_live_root(root)
        if (
            uid != self.binding["rayjob_uid"]
            or _stamp(created) != self.binding["rayjob_created_at"]
        ):
            raise SupervisorError("bound RayJob UID or creation time changed")
        status = root.get("status", {})
        if not isinstance(status, dict):
            raise SupervisorError("bound RayJob status is malformed")
        terminal = TERMINAL_STATUSES.get(status.get("jobStatus"), "")
        if terminal:
            if self.terminal_status and self.terminal_status != terminal:
                raise SupervisorError("bound RayJob terminal status changed")
            self.terminal_status = terminal
            if self.terminal_observed_at is None:
                self.terminal_observed_at = self.clock()
        cluster_name = status.get("rayClusterName", "")
        if cluster_name and (
            not isinstance(cluster_name, str)
            or not re.fullmatch(r"[a-z0-9][-a-z0-9]{0,62}", cluster_name)
        ):
            raise SupervisorError("bound RayCluster name is invalid")
        return cluster_name

    def _validate_bound_root_identity(self, root: dict) -> None:
        """Validate only immutable identity before an authorized emergency delete."""
        if self.binding is None:
            raise SupervisorError("supervision has no exact UID binding")
        name, uid, created = _metadata(root, "RayJob")
        metadata = root.get("metadata", {})
        if (
            name != self.root_name
            or metadata.get("namespace") != NAMESPACE
            or uid != self.binding["rayjob_uid"]
            or _stamp(created) != self.binding["rayjob_created_at"]
        ):
            raise SupervisorError("bound RayJob immutable identity changed")

    def _record(self, seen: dict[str, str], name: str, uid: str) -> None:
        if name in seen and seen[name] != uid:
            raise SupervisorError("owned resource name was reused with another UID")
        seen[name] = uid

    def _observe_inventory(self, cluster_name: str) -> None:
        if self.binding is None:
            raise SupervisorError("supervision has no exact UID binding")
        root_uid = self.binding["rayjob_uid"]
        workloads = self.backend.list_workloads_by_job_uid(root_uid)
        if self.workload_name:
            exact_workload = self.backend.get_workload(self.workload_name)
            if exact_workload is not None and all(
                _metadata(item, "Workload")[1] != self.workload_uid for item in workloads
            ):
                workloads.append(exact_workload)
        if len(workloads) > 1:
            raise SupervisorError("more than one Workload selects the bound RayJob UID")
        for item in workloads:
            name, uid, _ = _metadata(item, "Workload")
            if not _owned_by(item, kind="RayJob", name=self.root_name, uid=root_uid):
                raise SupervisorError("Workload is not owned by the bound RayJob UID")
            if self.workload_name and (self.workload_name != name or self.workload_uid != uid):
                raise SupervisorError("bound Workload identity changed")
            self.workload_name, self.workload_uid = name, uid
        owned_clusters = self.backend.list_rayclusters_by_rayjob_uid(self.root_name, root_uid)
        if len(owned_clusters) > 1:
            raise SupervisorError("more than one RayCluster is owned by the bound RayJob UID")
        status_cluster = self.backend.get_raycluster(cluster_name) if cluster_name else None
        if status_cluster is not None and not _owned_by(
            status_cluster, kind="RayJob", name=self.root_name, uid=root_uid
        ):
            raise SupervisorError("status-named RayCluster is not owned by the bound RayJob UID")
        cluster = owned_clusters[0] if owned_clusters else status_cluster
        if owned_clusters and status_cluster is not None:
            owned_identity = _metadata(owned_clusters[0], "RayCluster")[:2]
            status_identity = _metadata(status_cluster, "RayCluster")[:2]
            if owned_identity != status_identity:
                raise SupervisorError("status and ownerRef RayCluster identities disagree")
        if (
            cluster_name
            and cluster is not None
            and _metadata(cluster, "RayCluster")[0] != cluster_name
        ):
            raise SupervisorError("status and ownerRef RayCluster names disagree")
        if cluster is None:
            if self.cluster_name:
                if cluster_name and cluster_name != self.cluster_name:
                    raise SupervisorError("bound RayCluster status name changed")
                # A controller may disappear before its Pods.  The persisted
                # UID still permits an exact controller-owner census.
                name, uid = self.cluster_name, self.cluster_uid
            else:
                self.current_pods = set()
                self.current_runtime_images = set()
                self.current_gpus = 0
                return
        else:
            name, uid, _ = _metadata(cluster, "RayCluster")
            if not _owned_by(cluster, kind="RayJob", name=self.root_name, uid=root_uid):
                raise SupervisorError("RayCluster is not owned by the bound RayJob UID")
            if self.cluster_name and (self.cluster_name != name or self.cluster_uid != uid):
                raise SupervisorError("bound RayCluster identity changed")
            self.cluster_name, self.cluster_uid = name, uid
        # The backend must use the Ray controller-UID selector.  The ownerRef
        # check below independently proves that every returned Pod belongs to
        # the exact bound RayCluster rather than merely sharing its name.
        pods = self.backend.list_pods(name, uid)
        if len(pods) > NODES:
            raise SupervisorError("bound RayCluster exceeds the exact four-Pod topology")
        now_gpus = 0
        current_pods: set[str] = set()
        current_runtime_images: set[str] = set()
        for pod in pods:
            pod_name, pod_uid, created = _metadata(pod, "Pod")
            if not _owned_by(pod, kind="RayCluster", name=name, uid=uid):
                raise SupervisorError("Pod is not owned by the bound RayCluster UID")
            if _pod_gpu_count(pod) != GPUS_PER_NODE:
                raise SupervisorError("owned Pod is not an exact eight-GPU node")
            spec = pod["spec"]
            containers = spec.get("containers")
            gpu_containers = [
                item
                for item in containers
                if isinstance(item, dict)
                and quantity(item.get("resources", {}).get("requests", {}).get("nvidia.com/gpu", 0))
            ]
            if len(gpu_containers) != 1 or gpu_containers[0].get("image") != IMAGE:
                raise SupervisorError("owned Pod image differs from the exact candidate")
            statuses = (pod.get("status") or {}).get("containerStatuses", [])
            init_statuses = (pod.get("status") or {}).get("initContainerStatuses", [])
            if not isinstance(statuses, list) or not isinstance(init_statuses, list):
                raise SupervisorError("owned Pod status is malformed")
            for status in [*init_statuses, *statuses]:
                if (
                    not isinstance(status, dict)
                    or type(status.get("restartCount", 0)) is not int
                    or status.get("restartCount", 0) != 0
                ):
                    raise SupervisorError("owned Pod restarted")
            matches = [
                item
                for item in statuses
                if isinstance(item, dict) and item.get("name") == gpu_containers[0].get("name")
            ]
            self._record(self.pods, pod_name, pod_uid)
            current_pods.add(pod_name)
            if len(self.pods) > NODES:
                raise SupervisorError("owned GPU Pod identity was replaced")
            if len(matches) > 1:
                raise SupervisorError("owned GPU Pod runtime image identity is ambiguous")
            allocated = _pod_allocation_time(pod, created)
            if allocated is not None:
                if allocated > self.clock():
                    raise SupervisorError("Pod scheduling evidence is in the future")
                self.allocation_at = min(self.allocation_at or allocated, allocated)
                now_gpus += GPUS_PER_NODE
            if not matches or matches[0].get("imageID") in (None, ""):
                continue
            image_id = matches[0]["imageID"]
            if not isinstance(image_id, str):
                raise SupervisorError("owned GPU Pod runtime image identity is invalid")
            expected_digest = IMAGE.rsplit("@", 1)[1]
            if not (
                image_id.endswith("@" + expected_digest)
                or image_id.endswith("://" + expected_digest)
            ):
                raise SupervisorError("owned GPU Pod resolved another image digest")
            prior = self.runtime_images.get(pod_name)
            if prior is not None and prior != image_id:
                raise SupervisorError("owned GPU Pod runtime image identity changed")
            self.runtime_images[pod_name] = image_id
            current_runtime_images.add(pod_name)
        self.current_pods = current_pods
        self.current_runtime_images = current_runtime_images
        self.current_gpus = now_gpus
        if now_gpus and not self.workload_name:
            raise SupervisorError("GPU allocation exists without an exact bound Workload")
        self.peak_gpus = max(self.peak_gpus, now_gpus)
        self.topology_observed = self.topology_observed or self._topology_complete()

    def _observe_started(self) -> None:
        if self.started_at is not None or self.allocation_at is None:
            return
        receipt = self.started_reader()
        if receipt is None:
            return
        if not isinstance(receipt, dict) or set(receipt) != {
            "plan_sha256",
            "started_at_unix",
            "receipt_sha256",
        }:
            raise SupervisorError("STARTED.json receipt is invalid")
        body = {key: item for key, item in receipt.items() if key != "receipt_sha256"}
        if (
            receipt.get("receipt_sha256") != digest(body)
            or receipt.get("plan_sha256") != self.armed["plan_sha256"].removeprefix("sha256:")
            or type(receipt.get("started_at_unix")) not in {int, float}
        ):
            raise SupervisorError("STARTED.json receipt is invalid")
        started = datetime.fromtimestamp(receipt["started_at_unix"], UTC)
        if started < self.allocation_at or started > self.clock():
            raise SupervisorError("STARTED.json timestamp is outside the allocation window")
        if started > self.allocation_at + timedelta(seconds=ALLOCATION_TO_STARTED_SECONDS):
            raise SupervisorError("STARTED.json arrived outside the bounded startup window")
        self.started_at = started

    def _topology_complete(self) -> bool:
        return (
            len(self.current_pods) == NODES
            and len(self.current_runtime_images) == NODES
            and self.current_gpus == TOTAL_GPUS
        )

    def _release_reason(self, now: datetime) -> str:
        if self.terminal_observed_at is not None and now >= self.terminal_observed_at + timedelta(
            seconds=TERMINAL_GRACE_SECONDS
        ):
            return "terminal_self_release_grace_elapsed"
        if self.allocation_at is None:
            return ""
        if self.started_at is None:
            if now >= self.allocation_at + timedelta(seconds=ALLOCATION_TO_STARTED_SECONDS):
                return "allocation_started_receipt_deadline_elapsed"
            return ""
        action_at = min(
            self.started_at + timedelta(seconds=STARTED_TO_ACTION_SECONDS),
            self.allocation_at + timedelta(seconds=ALLOCATION_TO_ACTION_SECONDS),
        )
        return "authenticated_started_deadline_elapsed" if now >= action_at else ""

    def _delete_bound_root(self, reason: str) -> None:
        if reason not in RELEASE_REASONS:
            raise SupervisorError("root deletion reason is not authorized")
        if self.binding is None or self.authorization is None:
            raise SupervisorError("exact UID root deletion lacks sealed authorization")
        if self.delete_requested_at is not None:
            return
        root = self.backend.get_rayjob(self.root_name)
        if root is None:
            return
        # A runtime-surface defect is itself an authorized release reason.  It
        # must not prevent cleanup.  The immutable root identity and UID-CAS
        # precondition protect peers without depending on the defective field.
        self._validate_bound_root_identity(root)
        status = root.get("status")
        cluster_name = status.get("rayClusterName", "") if isinstance(status, dict) else ""
        if isinstance(cluster_name, str) and re.fullmatch(r"[a-z0-9][-a-z0-9]{0,62}", cluster_name):
            # Inventory evidence is best-effort here.  A defective child
            # surface must not block the separately authorized root UID-CAS
            # cleanup path; lack of a complete chain prevents a later
            # release-confirmed claim.
            with suppress(Exception):
                self._observe_inventory(cluster_name)
        intent = _seal(
            {
                "schema": DELETE_INTENT_SCHEMA,
                "status": "one_exact_uid_delete_may_be_attempted",
                "binding_sha256": self.binding["sha256"],
                "authorization_sha256": self.authorization["sha256"],
                "rayjob_name": self.root_name,
                "rayjob_uid": self.binding["rayjob_uid"],
                "reason": reason,
                "written_at": _stamp(self.clock()),
                "retry_policy": "same_exact_uid_cas_until_release_deadline",
            }
        )
        intent_path = self.operation_dir / "DELETE_INTENT.json"
        with _exclusive_state_lock(self.operation_dir):
            result_path = self.operation_dir / "RESULT.json"
            if result_path.exists():
                self.result = self._validate_result(_read_json(result_path))
                return
            if intent_path.exists():
                self._reload_delete_intent()
                return
            _write_once(intent_path, intent)
            self.delete_reason = reason
            self.delete_requested_at = _parse_time(intent["written_at"])

    def _drive_bound_root_delete(self) -> str:
        """Re-drive only the immutable ordinary root UID-CAS delete intent."""
        if self.binding is None or self.authorization is None:
            raise SupervisorError("exact UID root deletion lacks sealed authorization")
        if self.delete_requested_at is None:
            return "not_requested"
        root = self.backend.get_rayjob(self.root_name)
        if root is None:
            return "absent"
        try:
            self._validate_bound_root_identity(root)
        except SupervisorError:
            return "uid_mismatch"
        self.backend.delete_rayjob_uid_foreground(self.root_name, self.binding["rayjob_uid"])
        return "requested"

    def journal_bound_root_release_after_observer_failure(self) -> None:
        """Journal release for a recovery observer; never send DELETE here."""
        if self.binding is None:
            raise SupervisorError("observer failure cleanup has no exact UID binding")
        if self.authorization is None:
            self.reload_authorization()
        self._delete_bound_root("observation_contract_defect")

    def fresh_release_observation(self) -> tuple[bool, dict]:
        """Return one fresh, UID-bound full-chain absence/GPU-zero census."""
        return self._fresh_release_observation()

    def _fresh_release_observation(self) -> tuple[bool, dict]:
        if self.binding is None:
            return False, {"reason": "exact_root_UID_was_not_bound"}
        root = self.backend.get_rayjob(self.root_name)
        if root is not None:
            self._validate_bound_root_identity(root)
        workloads = self.backend.list_workloads_by_job_uid(self.binding["rayjob_uid"])
        if self.workload_name:
            exact_workload = self.backend.get_workload(self.workload_name)
            if exact_workload is not None and all(
                _metadata(item, "Workload")[1] != self.workload_uid for item in workloads
            ):
                workloads.append(exact_workload)
        for workload in workloads:
            workload_name, workload_uid, _ = _metadata(workload, "Workload")
            if self.workload_name and (
                workload_name != self.workload_name or workload_uid != self.workload_uid
            ):
                raise SupervisorError("fresh Workload census changed the bound identity")
            if not _owned_by(
                workload,
                kind="RayJob",
                name=self.root_name,
                uid=self.binding["rayjob_uid"],
            ):
                raise SupervisorError("fresh Workload census includes another owner")

        clusters = self.backend.list_rayclusters_by_rayjob_uid(
            self.root_name, self.binding["rayjob_uid"]
        )
        if self.cluster_name:
            exact_cluster = self.backend.get_raycluster(self.cluster_name)
            if exact_cluster is not None and all(
                _metadata(item, "RayCluster")[1] != self.cluster_uid for item in clusters
            ):
                clusters.append(exact_cluster)
        if len(clusters) > 1:
            raise SupervisorError("fresh RayCluster census found multiple owned controllers")
        for cluster in clusters:
            if not _owned_by(
                cluster,
                kind="RayJob",
                name=self.root_name,
                uid=self.binding["rayjob_uid"],
            ):
                raise SupervisorError("fresh RayCluster census includes another owner")

        cluster_name = self.cluster_name
        cluster_uid = self.cluster_uid
        if clusters:
            live_name, live_uid, _ = _metadata(clusters[0], "RayCluster")
            if cluster_name and (live_name != cluster_name or live_uid != cluster_uid):
                raise SupervisorError("fresh RayCluster census changed the bound identity")
            cluster_name, cluster_uid = live_name, live_uid
        pods = self._pods_by_run_identity()
        if cluster_name:
            pods = _merge_unique(
                pods,
                self.backend.list_pods(cluster_name, cluster_uid),
                "Pod",
            )
        for pod_name in self.pods:
            exact_pod = self.backend.get_pod(pod_name)
            if exact_pod is not None:
                pods = _merge_unique(pods, [exact_pod], "Pod")
        if cluster_name:
            for pod_item in pods:
                pod_name, pod_uid, _ = _metadata(pod_item, "Pod")
                if pod_name in self.pods and self.pods[pod_name] != pod_uid:
                    raise SupervisorError("fresh Pod census changed a bound UID")
                if pod_uid in self.pods.values() and self.pods.get(pod_name) != pod_uid:
                    raise SupervisorError("fresh Pod census changed a bound name")
                if not _owned_by(
                    pod_item,
                    kind="RayCluster",
                    name=cluster_name,
                    uid=cluster_uid,
                ):
                    raise SupervisorError("fresh Pod census includes another owner")
        if len(pods) > NODES:
            raise SupervisorError("fresh Pod census exceeds the exact candidate topology")
        active_gpus = sum(_pod_gpu_count(pod) for pod in pods)
        counts = {
            "root": int(root is not None),
            "workloads": len(workloads),
            "rayclusters": len(clusters),
            "pods": len(pods),
            "active_gpus": active_gpus,
        }
        return not any(counts.values()), counts

    def _finish(self, *, status: str, reason: str, release_confirmed: bool, final: dict) -> dict:
        if self.result is not None:
            return self.result
        result_path = self.operation_dir / "RESULT.json"
        with _exclusive_state_lock(self.operation_dir):
            if result_path.exists():
                self.result = self._validate_result(_read_json(result_path))
                return self.result
            if (
                self.delete_requested_at is None
                and (self.operation_dir / "DELETE_INTENT.json").exists()
            ):
                self._reload_delete_intent()
            state = self._persist_state_locked()
            value = _seal(
                {
                    "schema": RESULT_SCHEMA,
                    "status": status,
                    "reason": reason,
                    "release_confirmed": release_confirmed,
                    "binding_sha256": self.binding["sha256"] if self.binding else "",
                    "rayjob_name": self.root_name,
                    "rayjob_uid": self.binding["rayjob_uid"] if self.binding else "",
                    "nodes": NODES,
                    "gpus_per_node": GPUS_PER_NODE,
                    "topology_and_images_observed": self.topology_observed,
                    "ownership_state_sha256": state["sha256"],
                    "ownership_chain": {
                        "workload": (
                            {"name": self.workload_name, "uid": self.workload_uid}
                            if self.workload_name
                            else {}
                        ),
                        "raycluster": {
                            "name": self.cluster_name,
                            "uid": self.cluster_uid,
                        },
                        "pods": dict(sorted(self.pods.items())),
                        "runtime_images": dict(sorted(self.runtime_images.items())),
                    },
                    "allocated_at": _stamp(self.allocation_at) if self.allocation_at else "",
                    "started_at": _stamp(self.started_at) if self.started_at else "",
                    "terminal_status": self.terminal_status,
                    "delete_requested": self.delete_requested_at is not None,
                    "delete_reason": self.delete_reason,
                    "fresh_final_relist": final,
                    "active_gpus": 0 if release_confirmed else final.get("active_gpus"),
                    "private_logs_read": False,
                    "completed_at": _stamp(self.clock()),
                }
            )
            value = self._validate_result(value)
            try:
                _write_once(result_path, value)
            except FileExistsError:
                self.result = self._validate_result(_read_json(result_path))
            else:
                self.result = value
            return self.result

    def step(self) -> dict | None:
        """Perform one observation; return a terminal receipt or ``None``."""
        if self.result is not None:
            return self.result
        if self.binding is None:
            raise SupervisorError("supervision has no exact UID binding")
        authorization_path = self.operation_dir / "RELEASE_AUTHORIZATION.json"
        if self.authorization is None and authorization_path.exists():
            self.reload_authorization()
        if self.delete_requested_at is None:
            self._reload_delete_intent()
        now = self.clock()
        root = self.backend.get_rayjob(self.root_name)
        if self.delete_requested_at is not None:
            delete_state = self._drive_bound_root_delete()
            if delete_state == "uid_mismatch":
                return self._finish(
                    status="release_uncertain",
                    reason="exact_name_uid_mismatch",
                    release_confirmed=False,
                    final={"reason": "exact_name_uid_mismatch"},
                )
        if root is not None:
            cluster_name = self._validate_bound_root(root)
            self._observe_inventory(cluster_name)
            self._observe_started()
            self._persist_state()
            if self.result is not None:
                return self.result
            reason = self._release_reason(now)
            if reason and self.delete_requested_at is None:
                self._delete_bound_root(reason)
                if self.result is not None:
                    return self.result
                delete_state = self._drive_bound_root_delete()
                if delete_state == "uid_mismatch":
                    return self._finish(
                        status="release_uncertain",
                        reason="exact_name_uid_mismatch",
                        release_confirmed=False,
                        final={"reason": "exact_name_uid_mismatch"},
                    )
            delete_deadline = (
                self.delete_requested_at + timedelta(seconds=DELETE_CONFIRM_SECONDS)
                if self.delete_requested_at is not None
                else None
            )
            outer_deadline = (
                self.allocation_at + timedelta(seconds=ALLOCATION_TO_RELEASE_SECONDS)
                if self.allocation_at is not None
                else None
            )
            if (delete_deadline is not None and now >= delete_deadline) or (
                outer_deadline is not None and now >= outer_deadline
            ):
                _, final = self._fresh_release_observation()
                # The root was visible at the beginning of this observation.
                # One later zero census is not stable release proof: a child
                # from that controller can still appear before the next poll.
                # The immutable outer bound requires a terminal answer now, so
                # preserve the census but report uncertainty.
                return self._finish(
                    status="release_uncertain",
                    reason="release_confirmation_outer_bound_elapsed",
                    release_confirmed=False,
                    final=final,
                )
            return None

        released, final = self._fresh_release_observation()
        if released:
            if self.zero_scan_at is None:
                self.zero_scan_at = now
                return None
            if now < self.zero_scan_at + timedelta(seconds=POLL_SECONDS):
                return None
            return self._finish(
                status="released",
                reason="fresh_exact_inventory_absent",
                release_confirmed=True,
                final=final,
            )
        self.zero_scan_at = None
        deadline = (
            self.delete_requested_at + timedelta(seconds=DELETE_CONFIRM_SECONDS)
            if self.delete_requested_at is not None
            else (
                self.allocation_at + timedelta(seconds=ALLOCATION_TO_RELEASE_SECONDS)
                if self.allocation_at is not None
                else now
            )
        )
        if now >= deadline:
            return self._finish(
                status="release_uncertain",
                reason="owned_resources_remain_after_release_deadline",
                release_confirmed=False,
                final=final,
            )
        return None

    def _record_fault(self, error: BaseException) -> None:
        self.last_fault_class = type(error).__name__
        self.last_fault_at = self.clock()
        if self.binding is not None:
            self._persist_state()

    def run(self) -> dict:
        """Poll until release is confirmed or its immutable outer bound expires."""
        if self.sleep is None:
            raise SupervisorError("run requires an injected sleep function")
        while True:
            try:
                result = self.step()
                if result is not None:
                    return result
            except SupervisorError as exc:
                # A validated contract defect is evidence that this exact
                # allocation is unsafe to retain.  Persist the defect and use
                # only the separately authorized root UID-CAS release route.
                self._record_fault(exc)
                authorization_path = self.operation_dir / "RELEASE_AUTHORIZATION.json"
                if self.authorization is None and authorization_path.exists():
                    self.reload_authorization()
                if (
                    self.binding is not None
                    and self.authorization is not None
                    and self.delete_requested_at is None
                ):
                    try:
                        self._delete_bound_root("observation_contract_defect")
                    except Exception as delete_error:  # the intent is already durable
                        self._record_fault(delete_error)
            except Exception as exc:
                # Transport/read failures are unknown state, never an idle
                # finding.  Keep the independent supervisor alive and retry
                # observations; immutable time bounds remain in STATE.json.
                self._record_fault(exc)
            self.sleep(POLL_SECONDS)


def _process_alive(_pid: int) -> bool:
    """Fail closed when the caller did not retain a reapable process handle."""
    return False
