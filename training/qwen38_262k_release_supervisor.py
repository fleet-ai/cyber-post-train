"""Exact-identity release supervision for the held Qwen3.8 262K 4x8 canary.

This module does not create a workload.  A launcher must arm it before its one
external create attempt, persist that intent, and pass the resulting exact
RayJob through :meth:`FourNodeReleaseSupervisor.reconcile_exact`.  The only
destructive operation it can request is foreground deletion of that exact
RayJob with its immutable UID as a Kubernetes precondition.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Protocol
from uuid import UUID

from cyber_post_train.jobs import digest, quantity
from training import sft_262k_4node_v1 as compiler
from training import sft_262k_runtime as runtime

SCHEMA = "qwen38_262k_4node_release_supervisor_v1"
ARMED_SCHEMA = SCHEMA + ":armed"
INTENT_SCHEMA = SCHEMA + ":create_intent"
BINDING_SCHEMA = SCHEMA + ":binding"
AUTHORIZATION_SCHEMA = SCHEMA + ":release_authorization"
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
POLL_SECONDS = runtime.RELEASE_SUPERVISION["poll_seconds"]
ALLOCATION_TO_STARTED_SECONDS = runtime.RELEASE_SUPERVISION["external_deadlines"][
    "gpu_allocation_to_authenticated_started_seconds"
]
STARTED_TO_ACTION_SECONDS = runtime.RELEASE_SUPERVISION["external_deadlines"][
    "authenticated_started_to_forced_terminal_action_seconds"
]
ALLOCATION_TO_ACTION_SECONDS = runtime.RELEASE_SUPERVISION["external_deadlines"][
    "gpu_allocation_to_forced_terminal_action_seconds"
]
ALLOCATION_TO_RELEASE_SECONDS = runtime.RELEASE_SUPERVISION["external_deadlines"][
    "gpu_allocation_to_release_confirmation_outer_bound_seconds"
]
TERMINAL_GRACE_SECONDS = runtime.RELEASE_SUPERVISION["terminal_release_grace_seconds"]
DELETE_CONFIRM_SECONDS = runtime.RELEASE_SUPERVISION["post_delete_confirmation_seconds"]
RELEASE_REASONS = {
    "authenticated_started_deadline_elapsed",
    "allocation_started_receipt_deadline_elapsed",
    "observation_contract_defect",
    "terminal_self_release_grace_elapsed",
}
TERMINAL_STATUSES = {"SUCCEEDED": "Succeeded", "FAILED": "Failed"}


class SupervisorError(RuntimeError):
    """A sanitized release-supervision contract failure."""


class ClusterBackend(Protocol):
    """The narrow Kubernetes surface required by this exact supervisor."""

    def get_rayjob(self, name: str) -> dict | None: ...

    def list_workloads_by_job_uid(self, rayjob_uid: str) -> list[dict]: ...

    def get_raycluster(self, name: str) -> dict | None: ...

    def list_pods(self, cluster_name: str, controller_uid: str) -> list[dict]: ...

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
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(_canonical(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(fd)
    _fsync_dir(path.parent)


def _write_atomic(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".next")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(_canonical(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(fd)
    os.replace(temporary, path)
    os.chmod(path, 0o600)
    _fsync_dir(path.parent)


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


def _assert_subset(expected: object, actual: object, path: str = "root") -> None:
    """Require every persisted manifest field while allowing server metadata."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            raise SupervisorError(f"rendered RayJob surface changed at {path}")
        for key, value in expected.items():
            if path == "root.spec" and key == "suspend":
                # Kueue owns this single field after admission.
                continue
            if key not in actual:
                raise SupervisorError(f"rendered RayJob surface changed at {path}.{key}")
            _assert_subset(value, actual[key], f"{path}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise SupervisorError(f"rendered RayJob surface changed at {path}")
        for index, value in enumerate(expected):
            _assert_subset(value, actual[index], f"{path}[{index}]")
        return
    if actual != expected:
        raise SupervisorError(f"rendered RayJob surface changed at {path}")


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
    value = make_release_authorization(binding, authorized_at=authorized_at)
    _write_once(operation_dir / "RELEASE_AUTHORIZATION.json", value)
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
        self.armed = _validate_seal(_read_json(operation_dir / "ARMED.json"), ARMED_SCHEMA)
        validate_exact_candidate(self.plan, self.request, self.manifest)
        expected_hashes = {
            "plan_sha256": "sha256:" + digest(self.plan),
            "request_sha256": "sha256:" + digest(self.request),
            "manifest_sha256": "sha256:" + digest(self.manifest),
        }
        if any(self.armed.get(key) != value for key, value in expected_hashes.items()):
            raise SupervisorError("armed packet hashes differ from persisted inputs")
        self.root_name = self.manifest["metadata"]["name"]
        self.run_id = self.manifest["metadata"]["annotations"]["fleet.ai/run-id"]
        self.binding: dict | None = None
        binding_path = operation_dir / "BOUND.json"
        if binding_path.exists():
            self.binding = self._validate_binding(_read_json(binding_path))
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
        delete_intent_path = operation_dir / "DELETE_INTENT.json"
        if delete_intent_path.exists():
            delete_intent = _validate_seal(_read_json(delete_intent_path), DELETE_INTENT_SCHEMA)
            if (
                self.binding is None
                or self.authorization is None
                or (
                    delete_intent.get("binding_sha256") != self.binding["sha256"]
                    or delete_intent.get("authorization_sha256") != self.authorization["sha256"]
                    or delete_intent.get("rayjob_name") != self.root_name
                    or delete_intent.get("rayjob_uid") != self.binding["rayjob_uid"]
                    or delete_intent.get("reason") not in RELEASE_REASONS
                    or delete_intent.get("status") != "one_exact_uid_delete_may_be_attempted"
                    or delete_intent.get("uncertain_delete_policy")
                    != "reconcile_read_only_never_repeat"
                )
            ):
                raise SupervisorError("exact UID delete intent is invalid")
            self.delete_reason = delete_intent["reason"]
            self.delete_requested_at = _parse_time(delete_intent["written_at"])

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
    ) -> FourNodeReleaseSupervisor:
        validate_exact_candidate(plan, request, manifest)
        if type(observer_pid) is not int or observer_pid < 1 or not process_alive(observer_pid):
            raise SupervisorError("release supervisor process is not live before arm")
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
                "exact_name_absent_before_arm": True,
            }
        )
        _write_once(operation_dir / "ARMED.json", armed)
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
                "observer_receipt_sha256": self.armed["sha256"],
                "rayjob_name": self.root_name,
                "jobs_api_run_id": self.run_id,
                "manifest_sha256": self.armed["manifest_sha256"],
                "written_at": _stamp(self.clock()),
                "uncertain_create_policy": "reconcile_this_exact_name_never_retry_or_discover",
            }
        )
        _write_once(self.operation_dir / "CREATE_INTENT.json", value)
        return value

    def _validate_live_root(self, root: dict, *, require_uid: bool = True) -> tuple[str, datetime]:
        expected = deepcopy(self.manifest)
        _assert_subset(expected, root)
        if not require_uid:
            return "", _parse_time(root["metadata"]["creationTimestamp"])
        name, uid, created = _metadata(root, "RayJob")
        if name != self.root_name or root["metadata"].get("namespace") != NAMESPACE:
            raise SupervisorError("exact RayJob root identity changed")
        return uid, created

    def reconcile_exact(self) -> dict | None:
        """Adopt only the journaled exact name after a certain or uncertain create."""
        intent = _validate_seal(
            _read_json(self.operation_dir / "CREATE_INTENT.json"), INTENT_SCHEMA
        )
        if (
            intent.get("rayjob_name") != self.root_name
            or intent.get("armed_sha256") != self.armed["sha256"]
        ):
            raise SupervisorError("create intent differs from the armed exact name")
        root = self.backend.get_rayjob(self.root_name)
        if self.binding is not None:
            if root is not None:
                self._validate_bound_root(root)
            return self.binding
        if root is None:
            return None
        uid, created = self._validate_live_root(root)
        if created < _parse_time(self.armed["armed_at"]):
            raise SupervisorError("exact RayJob predates the armed supervisor")
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
        _write_once(self.operation_dir / "BOUND.json", binding)
        self.binding = binding
        self._persist_state()
        return binding

    def _validate_binding(self, value: dict) -> dict:
        _validate_seal(value, BINDING_SCHEMA)
        if (
            value.get("armed_sha256") != self.armed["sha256"]
            or value.get("rayjob_name") != self.root_name
            or value.get("jobs_api_run_id") != self.run_id
            or value.get("manifest_sha256") != self.armed["manifest_sha256"]
            or value.get("context") != CONTEXT
            or value.get("namespace") != NAMESPACE
        ):
            raise SupervisorError("exact UID binding differs from the armed packet")
        try:
            UUID(value["rayjob_uid"])
            _parse_time(value["rayjob_created_at"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SupervisorError("exact UID binding is invalid") from exc
        return value

    def _validate_authorization(self, value: dict) -> dict:
        if self.binding is None:
            raise SupervisorError("release authorization exists before exact UID binding")
        _validate_seal(value, AUTHORIZATION_SCHEMA)
        if (
            value.get("status") != "authorized_exact_root_uid_delete"
            or value.get("binding_sha256") != self.binding["sha256"]
            or value.get("rayjob_name") != self.root_name
            or value.get("rayjob_uid") != self.binding["rayjob_uid"]
            or value.get("authorized_reasons") != sorted(RELEASE_REASONS)
            or value.get("delete_route") != "root_rayjob_uid_cas_only"
        ):
            raise SupervisorError("release authorization is invalid")
        if _parse_time(value.get("authorized_at")) < _parse_time(self.binding["bound_at"]):
            raise SupervisorError("release authorization predates the exact UID binding")
        return value

    def reload_authorization(self) -> dict:
        self.authorization = self._validate_authorization(
            _read_json(self.operation_dir / "RELEASE_AUTHORIZATION.json")
        )
        return self.authorization

    def _restore_state(self, value: dict) -> None:
        if self.binding is None:
            raise SupervisorError("observation state exists before exact UID binding")
        _validate_seal(value, STATE_SCHEMA)
        if (
            value.get("status") != "exact_uid_observation_state"
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

    def _persist_state(self) -> dict:
        value = self._state_value()
        _write_atomic(self.operation_dir / "STATE.json", value)
        self.state_revision = value["revision"]
        return value

    def _validate_result(self, value: dict) -> dict:
        if self.binding is None:
            raise SupervisorError("terminal result exists before exact UID binding")
        _validate_seal(value, RESULT_SCHEMA)
        if (
            value.get("binding_sha256") != self.binding["sha256"]
            or value.get("rayjob_name") != self.root_name
            or value.get("rayjob_uid") != self.binding["rayjob_uid"]
        ):
            raise SupervisorError("terminal result differs from exact UID binding")
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

    def _record(self, seen: dict[str, str], name: str, uid: str) -> None:
        if name in seen and seen[name] != uid:
            raise SupervisorError("owned resource name was reused with another UID")
        seen[name] = uid

    def _observe_inventory(self, cluster_name: str) -> None:
        if self.binding is None:
            raise SupervisorError("supervision has no exact UID binding")
        root_uid = self.binding["rayjob_uid"]
        workloads = self.backend.list_workloads_by_job_uid(root_uid)
        if len(workloads) > 1:
            raise SupervisorError("more than one Workload selects the bound RayJob UID")
        for item in workloads:
            name, uid, _ = _metadata(item, "Workload")
            if not _owned_by(item, kind="RayJob", name=self.root_name, uid=root_uid):
                raise SupervisorError("Workload is not owned by the bound RayJob UID")
            if self.workload_name and (self.workload_name != name or self.workload_uid != uid):
                raise SupervisorError("bound Workload identity changed")
            self.workload_name, self.workload_uid = name, uid
        if not cluster_name:
            self.current_pods = set()
            self.current_runtime_images = set()
            self.current_gpus = 0
            return
        cluster = self.backend.get_raycluster(cluster_name)
        if cluster is None:
            if self.cluster_name != cluster_name or not self.cluster_uid:
                self.current_pods = set()
                self.current_runtime_images = set()
                self.current_gpus = 0
                return
            # A controller may disappear before its Pods.  The persisted UID
            # still permits an exact controller-UID census of those orphans.
            name, uid = self.cluster_name, self.cluster_uid
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
        root = self.backend.get_rayjob(self.root_name)
        if root is None:
            return
        self._validate_bound_root(root)
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
                "uncertain_delete_policy": "reconcile_read_only_never_repeat",
            }
        )
        _write_once(self.operation_dir / "DELETE_INTENT.json", intent)
        self.delete_reason = reason
        self.delete_requested_at = _parse_time(intent["written_at"])
        self.backend.delete_rayjob_uid_foreground(self.root_name, self.binding["rayjob_uid"])

    def _fresh_release_observation(self) -> tuple[bool, dict]:
        if self.binding is None or not self.cluster_name:
            return False, {"reason": "complete_UID_chain_was_not_observed"}
        root = self.backend.get_rayjob(self.root_name)
        workloads = self.backend.list_workloads_by_job_uid(self.binding["rayjob_uid"])
        cluster = self.backend.get_raycluster(self.cluster_name)
        pods = self.backend.list_pods(self.cluster_name, self.cluster_uid)
        active_gpus = sum(_pod_gpu_count(pod) for pod in pods)
        counts = {
            "root": int(root is not None),
            "workloads": len(workloads),
            "rayclusters": int(cluster is not None),
            "pods": len(pods),
            "active_gpus": active_gpus,
        }
        return not any(counts.values()), counts

    def _finish(self, *, status: str, reason: str, release_confirmed: bool, final: dict) -> dict:
        if self.result is not None:
            return self.result
        state = self._persist_state()
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
        _write_once(self.operation_dir / "RESULT.json", value)
        self.result = value
        return value

    def step(self) -> dict | None:
        """Perform one observation; return a terminal receipt or ``None``."""
        if self.result is not None:
            return self.result
        if self.binding is None:
            raise SupervisorError("supervision has no exact UID binding")
        authorization_path = self.operation_dir / "RELEASE_AUTHORIZATION.json"
        if self.authorization is None and authorization_path.exists():
            self.reload_authorization()
        now = self.clock()
        root = self.backend.get_rayjob(self.root_name)
        if root is not None:
            cluster_name = self._validate_bound_root(root)
            self._observe_inventory(cluster_name)
            self._observe_started()
            self._persist_state()
            reason = self._release_reason(now)
            if reason and self.delete_requested_at is None:
                self._delete_bound_root(reason)
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
                released, final = self._fresh_release_observation()
                return self._finish(
                    status="released" if released else "release_uncertain",
                    reason="fresh_exact_inventory_absent"
                    if released
                    else "allocation_release_outer_bound_elapsed",
                    release_confirmed=released,
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


def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True
