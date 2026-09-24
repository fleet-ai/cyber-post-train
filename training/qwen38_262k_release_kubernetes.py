"""Exact production Kubernetes adapter and direct-create hook for the 262K canary.

Recovery is intentionally bounded to a live launch host and the next external
monitor turn.  Durable receipts make that recovery exact; total launch-host
loss can still end in truthful release uncertainty rather than invented proof.
"""

from __future__ import annotations

import json
import os
import re
import stat
import subprocess
import sys
import time
from collections.abc import Callable
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

from cyber_post_train.jobs import digest
from training.qwen38_262k_release_supervisor import (
    CONTEXT,
    DELETE_CONFIRM_SECONDS,
    GPUS_PER_NODE,
    IMAGE,
    NAMESPACE,
    NODES,
    POLL_SECONDS,
    RUN_DIR,
    RUN_NAME,
    SCHEMA,
    TOTAL_GPUS,
    FourNodeReleaseSupervisor,
    SupervisorError,
    _read_json,
    _write_once,
)

_NAME = re.compile(r"[a-z0-9](?:[-a-z0-9]{0,61}[a-z0-9])?")
_RECEIPT = re.compile(r"sha256:[a-f0-9]{64}")
READY_SCHEMA = SCHEMA + ":observer_ready"
TAKEOVER_SCHEMA = SCHEMA + ":observer_takeover"
RECOVERY_READY_SCHEMA = SCHEMA + ":recovery_observer_ready"
GUARD_UNCERTAINTY_SCHEMA = SCHEMA + ":direct_guard_release_uncertainty"
OBSERVER_READY_SECONDS = 30
OBSERVER_PRECREATE_SECONDS = 600
OBSERVER_TAKEOVER_SECONDS = 30
OBSERVER_CLEANUP_JOIN_SECONDS = 30
_OBSERVER_PROCESSES: dict[int, subprocess.Popen] = {}


class ReleaseKubernetesError(RuntimeError):
    """A sanitized failure at the exact production Kubernetes boundary."""


def _stamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _seal(value: dict) -> dict:
    return {**value, "sha256": "sha256:" + digest(value)}


def _ready_path(operation_dir: Path) -> Path:
    return operation_dir.with_name(operation_dir.name + ".OBSERVER_READY.json")


def _read_private_receipt(path: Path, label: str) -> dict:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise SupervisorError(f"release supervisor {label} receipt is unavailable") from exc
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) != 0o600
    ):
        raise SupervisorError(f"release supervisor {label} receipt is not a private file")
    return _read_json(path)


def _read_ready_receipt(path: Path) -> dict:
    value = _read_private_receipt(path, "readiness")
    body = {key: item for key, item in value.items() if key != "sha256"}
    if (
        value.get("schema") != READY_SCHEMA
        or set(value)
        != {
            "schema",
            "status",
            "pid",
            "operation_dir",
            "context",
            "namespace",
            "run_name",
            "run_dir",
            "image",
            "nodes",
            "gpus_per_node",
            "total_gpus",
            "plan_sha256",
            "request_sha256",
            "manifest_sha256",
            "rayjob_name",
            "jobs_api_run_id",
            "ready_at",
            "sha256",
        }
        or value.get("sha256") != "sha256:" + digest(body)
    ):
        raise SupervisorError("release supervisor readiness receipt is invalid")
    return value


def _takeover_path(operation_dir: Path, recovery_token: str = "") -> Path:
    name = (
        f"RECOVERY_OBSERVER_TAKEOVER_{recovery_token}.json"
        if recovery_token
        else "OBSERVER_TAKEOVER.json"
    )
    return operation_dir / name


def _recovery_ready_path(operation_dir: Path, token: str) -> Path:
    return operation_dir / f"RECOVERY_OBSERVER_READY_{token}.json"


def _recovery_ready_receipt(
    supervisor: FourNodeReleaseSupervisor, *, token: str, pid: int, now: datetime
) -> dict:
    return _seal(
        {
            "schema": RECOVERY_READY_SCHEMA,
            "status": "same_intent_recovery_observer_ready",
            "recovery_token": token,
            "observer_pid": pid,
            "armed_sha256": supervisor.armed["sha256"],
            "intent_sha256": _read_json(supervisor.operation_dir / "CREATE_INTENT.json")["sha256"],
            "rayjob_name": supervisor.root_name,
            "jobs_api_run_id": supervisor.run_id,
            "ready_at": _stamp(now),
        }
    )


def _validate_recovery_ready_receipt(
    value: dict,
    supervisor: FourNodeReleaseSupervisor,
    *,
    token: str,
    pid: int,
) -> dict:
    body = {key: item for key, item in value.items() if key != "sha256"}
    expected = {
        "schema": RECOVERY_READY_SCHEMA,
        "status": "same_intent_recovery_observer_ready",
        "recovery_token": token,
        "observer_pid": pid,
        "armed_sha256": supervisor.armed["sha256"],
        "intent_sha256": _read_json(supervisor.operation_dir / "CREATE_INTENT.json")["sha256"],
        "rayjob_name": supervisor.root_name,
        "jobs_api_run_id": supervisor.run_id,
    }
    if (
        set(value) != {*expected, "ready_at", "sha256"}
        or any(value.get(key) != item for key, item in expected.items())
        or value.get("sha256") != "sha256:" + digest(body)
    ):
        raise SupervisorError("recovery observer readiness receipt is invalid")
    _parse_receipt_time(value.get("ready_at"))
    return value


def _takeover_receipt(
    supervisor: FourNodeReleaseSupervisor,
    binding: dict,
    authorization: dict,
    *,
    pid: int,
    now: datetime,
) -> dict:
    return _seal(
        {
            "schema": TAKEOVER_SCHEMA,
            "status": "detached_observer_owns_exact_uid_supervision",
            "observer_pid": pid,
            "observer_receipt_sha256": supervisor.armed["observer_receipt_sha256"],
            "armed_sha256": supervisor.armed["sha256"],
            "binding_sha256": binding["sha256"],
            "authorization_sha256": authorization["sha256"],
            "rayjob_name": binding["rayjob_name"],
            "rayjob_uid": binding["rayjob_uid"],
            "taken_over_at": _stamp(now),
        }
    )


def _validate_takeover_receipt(
    value: dict,
    supervisor: FourNodeReleaseSupervisor,
    binding: dict,
    authorization: dict,
    *,
    expected_pid: int | None = None,
) -> dict:
    body = {key: item for key, item in value.items() if key != "sha256"}
    expected = {
        "status": "detached_observer_owns_exact_uid_supervision",
        "observer_pid": (
            supervisor.armed["observer_pid"] if expected_pid is None else expected_pid
        ),
        "observer_receipt_sha256": supervisor.armed["observer_receipt_sha256"],
        "armed_sha256": supervisor.armed["sha256"],
        "binding_sha256": binding["sha256"],
        "authorization_sha256": authorization["sha256"],
        "rayjob_name": binding["rayjob_name"],
        "rayjob_uid": binding["rayjob_uid"],
    }
    if (
        value.get("schema") != TAKEOVER_SCHEMA
        or set(value) != {"schema", *expected, "taken_over_at", "sha256"}
        or value.get("sha256") != "sha256:" + digest(body)
        or any(value.get(key) != item for key, item in expected.items())
    ):
        raise SupervisorError("release supervisor takeover receipt is invalid")
    # Receipt hashes and write-once dependency bindings establish causality;
    # wall clocks are sealed evidence only and are never an ordering gate.
    _parse_receipt_time(value.get("taken_over_at"))
    return value


def _parse_receipt_time(value: object) -> datetime:
    if not isinstance(value, str):
        raise SupervisorError("release supervisor receipt timestamp is invalid")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SupervisorError("release supervisor receipt timestamp is invalid") from exc
    if parsed.tzinfo is None:
        raise SupervisorError("release supervisor receipt timestamp is invalid")
    return parsed.astimezone(UTC)


def _validate_ready_receipt(
    receipt: dict,
    operation_dir: Path,
    plan: dict,
    request: dict,
    manifest: dict,
) -> dict:
    expected = {
        "status": "detached_observer_ready_before_arm",
        "operation_dir": str(operation_dir.resolve()),
        "context": CONTEXT,
        "namespace": NAMESPACE,
        "run_name": RUN_NAME,
        "run_dir": RUN_DIR,
        "image": IMAGE,
        "nodes": NODES,
        "gpus_per_node": GPUS_PER_NODE,
        "total_gpus": TOTAL_GPUS,
        "plan_sha256": "sha256:" + digest(plan),
        "request_sha256": "sha256:" + digest(request),
        "manifest_sha256": "sha256:" + digest(manifest),
        "rayjob_name": manifest["metadata"]["name"],
        "jobs_api_run_id": manifest["metadata"]["annotations"]["fleet.ai/run-id"],
    }
    if type(receipt.get("pid")) is not int or receipt["pid"] < 1:
        raise SupervisorError("release supervisor readiness PID is invalid")
    if any(receipt.get(key) != item for key, item in expected.items()):
        raise SupervisorError("release supervisor readiness receipt differs from candidate")
    try:
        _parse_receipt_time(receipt["ready_at"])
    except (KeyError, TypeError, ValueError, SupervisorError) as exc:
        raise SupervisorError("release supervisor readiness timestamp is invalid") from exc
    return receipt


class ProductionReleaseKubernetesBackend:
    """No-shell, exact-context transport implementing only the release protocol."""

    def __init__(
        self,
        *,
        binary: str = "kubectl",
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ) -> None:
        if not binary or binary.startswith("-"):
            raise ReleaseKubernetesError("invalid kubectl binary")
        self.binary = binary
        self.runner = runner

    def _run_text(self, args: list[str], *, body: dict | None = None) -> str:
        command = [self.binary, "--context", CONTEXT, "--request-timeout=60s", *args]
        try:
            result = self.runner(
                command,
                input=(
                    json.dumps(body, sort_keys=True, separators=(",", ":"))
                    if body is not None
                    else None
                ),
                text=True,
                capture_output=True,
                timeout=75,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise ReleaseKubernetesError("kubectl transport failed") from exc
        if result.returncode:
            raise ReleaseKubernetesError("kubectl operation failed; server output suppressed")
        return result.stdout

    def _run(self, args: list[str], *, body: dict | None = None) -> dict:
        try:
            value = json.loads(self._run_text(args, body=body))
        except (TypeError, ValueError) as exc:
            raise ReleaseKubernetesError("kubectl response is invalid") from exc
        if not isinstance(value, dict):
            raise ReleaseKubernetesError("kubectl response is invalid")
        return value

    def _get_optional(self, resource: str, name: str) -> dict | None:
        self._name(name)
        output = self._run_text(
            [
                "get",
                resource,
                name,
                "--namespace",
                NAMESPACE,
                "--ignore-not-found",
                "--output=json",
            ]
        ).strip()
        if not output:
            return None
        try:
            value = json.loads(output)
        except ValueError as exc:
            raise ReleaseKubernetesError("kubectl response is invalid") from exc
        if not isinstance(value, dict):
            raise ReleaseKubernetesError("kubectl response is invalid")
        return value

    def _list_all(self, resource: str) -> list[dict]:
        return self._list_args(resource, [])

    def _list_args(self, resource: str, extra: list[str]) -> list[dict]:
        value = self._run(["get", resource, "--namespace", NAMESPACE, *extra, "--output=json"])
        items = value.get("items")
        if not isinstance(value.get("kind"), str) or not value["kind"].endswith("List"):
            raise ReleaseKubernetesError("kubectl list response is invalid")
        if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
            raise ReleaseKubernetesError("kubectl list response is invalid")
        return items

    @staticmethod
    def _name(value: str) -> None:
        if not isinstance(value, str) or _NAME.fullmatch(value) is None:
            raise ReleaseKubernetesError("Kubernetes name is invalid")

    @staticmethod
    def _uid(value: str) -> None:
        try:
            parsed = UUID(value)
        except (TypeError, ValueError) as exc:
            raise ReleaseKubernetesError("Kubernetes UID is invalid") from exc
        if str(parsed) != value:
            raise ReleaseKubernetesError("Kubernetes UID is invalid")

    def get_rayjob(self, name: str) -> dict | None:
        return self._get_optional("rayjobs.ray.io", name)

    def list_workloads_by_job_uid(self, rayjob_uid: str) -> list[dict]:
        self._uid(rayjob_uid)
        matches = []
        for workload in self._list_all("workloads.kueue.x-k8s.io"):
            metadata = workload.get("metadata")
            labels = metadata.get("labels") if isinstance(metadata, dict) else None
            owners = metadata.get("ownerReferences") if isinstance(metadata, dict) else None
            label_match = (
                isinstance(labels, dict) and labels.get("kueue.x-k8s.io/job-uid") == rayjob_uid
            )
            owner_match = isinstance(owners, list) and any(
                isinstance(owner, dict)
                and owner.get("kind") == "RayJob"
                and owner.get("uid") == rayjob_uid
                and owner.get("controller") is True
                for owner in owners
            )
            if label_match or owner_match:
                matches.append(workload)
        return matches

    def get_workload(self, name: str) -> dict | None:
        return self._get_optional("workloads.kueue.x-k8s.io", name)

    def list_rayclusters_by_rayjob_uid(self, name: str, uid: str) -> list[dict]:
        self._name(name)
        self._uid(uid)
        matches = []
        for cluster in self._list_all("rayclusters.ray.io"):
            metadata = cluster.get("metadata")
            owners = metadata.get("ownerReferences") if isinstance(metadata, dict) else None
            if isinstance(owners, list) and any(
                isinstance(owner, dict)
                and owner.get("kind") == "RayJob"
                and owner.get("name") == name
                and owner.get("uid") == uid
                and owner.get("controller") is True
                for owner in owners
            ):
                matches.append(cluster)
        return matches

    def get_raycluster(self, name: str) -> dict | None:
        return self._get_optional("rayclusters.ray.io", name)

    def list_pods(self, cluster_name: str, controller_uid: str) -> list[dict]:
        self._name(cluster_name)
        self._uid(controller_uid)
        pods = []
        for pod in self._list_all("pods"):
            metadata = pod.get("metadata")
            owners = metadata.get("ownerReferences") if isinstance(metadata, dict) else None
            if isinstance(owners, list) and any(
                isinstance(owner, dict)
                and owner.get("kind") == "RayCluster"
                and owner.get("name") == cluster_name
                and owner.get("uid") == controller_uid
                and owner.get("controller") is True
                for owner in owners
            ):
                pods.append(pod)
        return pods

    def list_pods_by_run_identity(self, run_id: str, run_name: str) -> list[dict]:
        try:
            if str(UUID(run_id)) != run_id:
                raise ValueError
        except (TypeError, ValueError) as exc:
            raise ReleaseKubernetesError("run ID is invalid") from exc
        if run_name != RUN_NAME:
            raise ReleaseKubernetesError("run name is invalid")
        matches = []
        for pod in self._list_all("pods"):
            metadata = pod.get("metadata")
            labels = metadata.get("labels") if isinstance(metadata, dict) else None
            if (
                isinstance(labels, dict)
                and labels.get("fleet.ai/run-id") == run_id
                and labels.get("fleet.ai/run-name") == run_name
            ):
                matches.append(pod)
        return matches

    def get_pod(self, name: str) -> dict | None:
        return self._get_optional("pods", name)

    def delete_rayjob_uid_foreground(self, name: str, uid: str) -> None:
        self._name(name)
        self._uid(uid)
        self._run(
            [
                "delete",
                "--raw",
                f"/apis/ray.io/v1/namespaces/{NAMESPACE}/rayjobs/{name}",
                "-f",
                "-",
            ],
            body={
                "apiVersion": "v1",
                "kind": "DeleteOptions",
                "propagationPolicy": "Foreground",
                "preconditions": {"uid": uid},
            },
        )


def _ready_receipt(
    operation_dir: Path,
    plan: dict,
    request: dict,
    manifest: dict,
    *,
    pid: int,
    now: datetime,
) -> dict:
    return _seal(
        {
            "schema": READY_SCHEMA,
            "status": "detached_observer_ready_before_arm",
            "pid": pid,
            "operation_dir": str(operation_dir.resolve()),
            "context": CONTEXT,
            "namespace": NAMESPACE,
            "run_name": RUN_NAME,
            "run_dir": RUN_DIR,
            "image": IMAGE,
            "nodes": NODES,
            "gpus_per_node": GPUS_PER_NODE,
            "total_gpus": TOTAL_GPUS,
            "plan_sha256": "sha256:" + digest(plan),
            "request_sha256": "sha256:" + digest(request),
            "manifest_sha256": "sha256:" + digest(manifest),
            "rayjob_name": manifest["metadata"]["name"],
            "jobs_api_run_id": manifest["metadata"]["annotations"]["fleet.ai/run-id"],
            "ready_at": _stamp(now),
        }
    )


def launch_detached_release_observer(
    operation_dir: Path,
    plan: dict,
    request: dict,
    manifest: dict,
    *,
    popen: Callable[..., subprocess.Popen] = subprocess.Popen,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict:
    """Start the exact observer and return its sealed, candidate-bound readiness."""
    operation_dir = operation_dir.resolve()
    if not operation_dir.parent.is_dir() or operation_dir.exists() or operation_dir.is_symlink():
        raise SupervisorError("release supervisor operation path is not fresh")
    ready_path = _ready_path(operation_dir)
    if ready_path.exists() or ready_path.is_symlink():
        raise SupervisorError("release supervisor readiness receipt already exists")
    command = [
        sys.executable,
        "-m",
        "training.qwen38_262k_release_kubernetes",
        "observe",
        str(operation_dir),
        "sha256:" + digest(plan),
        "sha256:" + digest(request),
        "sha256:" + digest(manifest),
        manifest["metadata"]["name"],
        manifest["metadata"]["annotations"]["fleet.ai/run-id"],
    ]
    try:
        process = popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=Path(__file__).resolve().parents[1],
            start_new_session=True,
            close_fds=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SupervisorError("release supervisor process could not start") from exc
    _OBSERVER_PROCESSES[process.pid] = process
    deadline = monotonic() + OBSERVER_READY_SECONDS
    while monotonic() < deadline:
        if ready_path.exists():
            receipt = _validate_ready_receipt(
                _read_ready_receipt(ready_path), operation_dir, plan, request, manifest
            )
            if receipt["pid"] != process.pid or process.poll() is not None:
                _OBSERVER_PROCESSES.pop(process.pid, None)
                raise SupervisorError("release supervisor process is not live after readiness")
            return receipt
        if process.poll() is not None:
            _OBSERVER_PROCESSES.pop(process.pid, None)
            raise SupervisorError("release supervisor exited before readiness")
        sleep(0.05)
    process.terminate()
    _OBSERVER_PROCESSES.pop(process.pid, None)
    raise SupervisorError("release supervisor readiness deadline elapsed")


def launch_detached_recovery_observer(
    operation_dir: Path,
    supervisor: FourNodeReleaseSupervisor,
    *,
    popen: Callable[..., subprocess.Popen] = subprocess.Popen,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict:
    """Start a same-intent observer after the original local observer dies."""
    token = str(uuid4())
    ready_path = _recovery_ready_path(operation_dir, token)
    command = [
        sys.executable,
        "-m",
        "training.qwen38_262k_release_kubernetes",
        "recover",
        str(operation_dir.resolve()),
        supervisor.armed["plan_sha256"],
        supervisor.armed["request_sha256"],
        supervisor.armed["manifest_sha256"],
        supervisor.root_name,
        supervisor.run_id,
        token,
    ]
    try:
        process = popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=Path(__file__).resolve().parents[1],
            start_new_session=True,
            close_fds=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SupervisorError("recovery observer process could not start") from exc
    _OBSERVER_PROCESSES[process.pid] = process
    deadline = monotonic() + OBSERVER_READY_SECONDS
    while monotonic() < deadline:
        if ready_path.exists():
            receipt = _validate_recovery_ready_receipt(
                _read_private_receipt(ready_path, "recovery readiness"),
                supervisor,
                token=token,
                pid=process.pid,
            )
            if process.poll() is not None:
                raise SupervisorError("recovery observer exited after readiness")
            return receipt
        if process.poll() is not None:
            _OBSERVER_PROCESSES.pop(process.pid, None)
            raise SupervisorError("recovery observer exited before readiness")
        sleep(0.05)
    process.terminate()
    _OBSERVER_PROCESSES.pop(process.pid, None)
    raise SupervisorError("recovery observer readiness deadline elapsed")


def _started_receipt() -> dict | None:
    path = Path(RUN_DIR) / "STARTED.json"
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return None
    except OSError:
        return None
    if not stat.S_ISREG(mode) or stat.S_ISLNK(mode):
        raise SupervisorError("STARTED.json receipt is not a regular file")
    try:
        value = json.loads(path.read_bytes())
    except (OSError, ValueError):
        # The shared runtime streams this create-once file directly to its
        # final path. A concurrent read can therefore observe ordinary partial
        # JSON; absence/incompleteness remains unknown until the startup bound.
        return None
    if not isinstance(value, dict):
        raise SupervisorError("STARTED.json receipt is invalid")
    return value


def run_detached_release_observer(
    operation_dir: Path,
    expected_plan_sha256: str,
    expected_request_sha256: str,
    expected_manifest_sha256: str,
    expected_name: str,
    expected_run_id: str,
    *,
    recovery: bool = False,
    recovery_token: str = "",
) -> dict | None:
    """Own exact-name recovery and release after advertising process readiness."""
    operation_dir = operation_dir.resolve()
    for value in (
        expected_plan_sha256,
        expected_request_sha256,
        expected_manifest_sha256,
    ):
        if _RECEIPT.fullmatch(value) is None:
            raise SupervisorError("release observer expected hash is invalid")
    if _NAME.fullmatch(expected_name) is None:
        raise SupervisorError("release observer expected name is invalid")
    try:
        UUID(expected_run_id)
    except ValueError as exc:
        raise SupervisorError("release observer expected run ID is invalid") from exc
    if recovery:
        try:
            if str(UUID(recovery_token)) != recovery_token:
                raise ValueError
        except (TypeError, ValueError) as exc:
            raise SupervisorError("release recovery token is invalid") from exc
    elif recovery_token:
        raise SupervisorError("ordinary observer cannot carry a recovery token")
    ready = None
    if not recovery:
        ready = _seal(
            {
                "schema": READY_SCHEMA,
                "status": "detached_observer_ready_before_arm",
                "pid": os.getpid(),
                "operation_dir": str(operation_dir),
                "context": CONTEXT,
                "namespace": NAMESPACE,
                "run_name": RUN_NAME,
                "run_dir": RUN_DIR,
                "image": IMAGE,
                "nodes": NODES,
                "gpus_per_node": GPUS_PER_NODE,
                "total_gpus": TOTAL_GPUS,
                "plan_sha256": expected_plan_sha256,
                "request_sha256": expected_request_sha256,
                "manifest_sha256": expected_manifest_sha256,
                "rayjob_name": expected_name,
                "jobs_api_run_id": expected_run_id,
                "ready_at": _stamp(datetime.now(UTC)),
            }
        )
        _write_once(_ready_path(operation_dir), ready)
    deadline = time.monotonic() + OBSERVER_PRECREATE_SECONDS
    supervisor: FourNodeReleaseSupervisor | None = None
    intent_seen = recovery
    if recovery:
        supervisor = FourNodeReleaseSupervisor(
            operation_dir,
            ProductionReleaseKubernetesBackend(),
            sleep=time.sleep,
            started_reader=_started_receipt,
        )
        if (
            supervisor.armed["plan_sha256"] != expected_plan_sha256
            or supervisor.armed["request_sha256"] != expected_request_sha256
            or supervisor.armed["manifest_sha256"] != expected_manifest_sha256
            or supervisor.root_name != expected_name
            or supervisor.run_id != expected_run_id
        ):
            raise SupervisorError("recovery observer differs from durable exact intent")
        recovery_ready = _recovery_ready_receipt(
            supervisor,
            token=recovery_token,
            pid=os.getpid(),
            now=datetime.now(UTC),
        )
        _write_once(_recovery_ready_path(operation_dir, recovery_token), recovery_ready)
    while True:
        intent_seen = intent_seen or (operation_dir / "CREATE_INTENT.json").exists()
        if not intent_seen and time.monotonic() >= deadline:
            return None
        if intent_seen:
            if supervisor is None:
                supervisor = FourNodeReleaseSupervisor(
                    operation_dir,
                    ProductionReleaseKubernetesBackend(),
                    sleep=time.sleep,
                    started_reader=_started_receipt,
                )
                if (
                    supervisor.armed["plan_sha256"] != expected_plan_sha256
                    or supervisor.armed["request_sha256"] != expected_request_sha256
                    or supervisor.armed["manifest_sha256"] != expected_manifest_sha256
                    or supervisor.root_name != expected_name
                    or supervisor.run_id != expected_run_id
                    or (
                        not recovery
                        and (
                            supervisor.armed["observer_pid"] != ready["pid"]
                            or supervisor.armed["observer_receipt_sha256"] != ready["sha256"]
                        )
                    )
                ):
                    raise SupervisorError("armed packet differs from observer readiness")
            supervisor.resume_selected_cleanup_branch()
            cleanup_intent = operation_dir / "DRIFT_CLEANUP_DELETE_INTENT.json"
            try:
                cleanup_result = None
                if cleanup_intent.exists():
                    cleanup_result = supervisor.drive_drift_cleanup_delete()
                if cleanup_result is None:
                    cleanup_result = supervisor.reconcile_drift_cleanup_absence()
            except ReleaseKubernetesError:
                time.sleep(1)
                continue
            if cleanup_result is not None:
                return cleanup_result
            if cleanup_intent.exists():
                # The exact UID intent is immutable; keep its idempotent
                # foreground root deletion and full release proof alive.
                time.sleep(1)
                continue
            try:
                binding = supervisor.reconcile_and_authorize_exact()
            except ReleaseKubernetesError:
                # A lost launcher may have completed the create while the
                # control plane is briefly unreadable. Keep retrying only the
                # one journaled exact name; never infer absence or retry POST.
                time.sleep(1)
                continue
            except SupervisorError:
                if cleanup_intent.exists():
                    continue
                if supervisor.binding is not None and supervisor.authorization is not None:
                    # An ordinary atomic binding already exists. A later live
                    # surface defect is handled by its ordinary authorized
                    # exact-UID release path, never by the cleanup-only branch.
                    return supervisor.run()
                try:
                    root = supervisor.backend.get_rayjob(supervisor.root_name)
                except ReleaseKubernetesError:
                    time.sleep(1)
                    continue
                if root is None:
                    join_deadline = time.monotonic() + OBSERVER_CLEANUP_JOIN_SECONDS
                    while time.monotonic() < join_deadline and not cleanup_intent.exists():
                        time.sleep(0.05)
                    if cleanup_intent.exists():
                        continue
                    raise
                try:
                    supervisor.cleanup_rejected_exact_root(root)
                except SupervisorError:
                    if cleanup_intent.exists():
                        continue
                    raise
                continue
            if binding is not None:
                if supervisor.authorization is None:
                    raise SupervisorError(
                        "release authorization is missing after exact UID binding"
                    )
                takeover = _takeover_receipt(
                    supervisor,
                    binding,
                    supervisor.authorization,
                    pid=os.getpid(),
                    now=datetime.now(UTC),
                )
                try:
                    _write_once(_takeover_path(operation_dir, recovery_token), takeover)
                except FileExistsError:
                    _validate_takeover_receipt(
                        _read_private_receipt(
                            _takeover_path(operation_dir, recovery_token), "takeover"
                        ),
                        supervisor,
                        binding,
                        supervisor.authorization,
                        expected_pid=os.getpid(),
                    )
                return supervisor.run()
        time.sleep(1)


class ExactCandidateDirectCreateGuard:
    """Bridge the one reviewed direct create to its independent supervisor."""

    def __init__(
        self,
        operation_dir: Path,
        backend: ProductionReleaseKubernetesBackend | None = None,
        *,
        launch_observer: Callable[[Path, dict, dict, dict], dict] | None = None,
        launch_recovery_observer: Callable[[Path, FourNodeReleaseSupervisor], dict] | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        process_alive: Callable[[int], bool] = lambda pid: _process_alive(pid),
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        started_visible: Callable[[], bool] = lambda: _local_started_visible(),
    ) -> None:
        self.operation_dir = operation_dir
        self.backend = backend or ProductionReleaseKubernetesBackend()
        if type(self.backend) is not ProductionReleaseKubernetesBackend:
            raise SupervisorError("release guard requires the exact production backend")
        self.launch_observer = launch_observer or launch_detached_release_observer
        self.launch_recovery_observer = (
            launch_recovery_observer or launch_detached_recovery_observer
        )
        self.clock = clock
        self.process_alive = process_alive
        self.monotonic = monotonic
        self.sleep = sleep
        self.started_visible = started_visible
        self.observer_pid = 0
        self.observer_receipt_sha256 = ""
        self.recovery_token = ""
        self.supervisor: FourNodeReleaseSupervisor | None = None

    def arm(self, plan: dict, request: dict, manifest: dict) -> None:
        if self.supervisor is not None:
            raise SupervisorError("release supervision was already armed")
        if not self.started_visible():
            raise SupervisorError(
                "exact 262K release supervision requires direct shared-SFS visibility"
            )
        launched = self.launch_observer(self.operation_dir, plan, request, manifest)
        receipt = _validate_ready_receipt(
            _read_ready_receipt(_ready_path(self.operation_dir)),
            self.operation_dir,
            plan,
            request,
            manifest,
        )
        if launched != receipt:
            raise SupervisorError("release supervisor readiness result differs from its receipt")
        self.observer_pid = receipt["pid"]
        self.observer_receipt_sha256 = receipt["sha256"]
        if not self.process_alive(self.observer_pid):
            raise SupervisorError("release supervisor process died before arm")
        self.supervisor = FourNodeReleaseSupervisor.arm(
            self.operation_dir,
            self.backend,
            plan=plan,
            request=request,
            manifest=manifest,
            observer_pid=self.observer_pid,
            observer_receipt_sha256=self.observer_receipt_sha256,
            clock=self.clock,
            process_alive=self.process_alive,
        )
        _write_once(self.operation_dir / "OBSERVER_READY.json", receipt)

    def create_intent_evidence(self) -> dict:
        supervisor = self._armed()
        return {
            "release_supervisor_pid": self.observer_pid,
            "release_supervisor_receipt_sha256": self.observer_receipt_sha256,
            "release_supervision_armed_sha256": supervisor.armed["sha256"],
        }

    def prepare_immediately_before_post(self) -> None:
        supervisor = self._armed()
        supervisor.write_create_intent()
        if self.backend.get_rayjob(supervisor.root_name) is not None:
            raise SupervisorError("exact RayJob name appeared immediately before create")
        if not self.process_alive(self.observer_pid):
            raise SupervisorError("release supervisor died immediately before create")

    def reconcile_ambiguous_create(self) -> dict | None:
        """Boundedly reread only the journaled exact name; never bind here."""
        supervisor = self._armed()
        deadline = self.monotonic() + OBSERVER_TAKEOVER_SECONDS
        recovery_error: SupervisorError | None = None
        while self.monotonic() < deadline:
            try:
                root = self.backend.get_rayjob(supervisor.root_name)
            except ReleaseKubernetesError:
                root = None
            if root is not None:
                return root
            error = self._ensure_recovery_observer()
            if error is not None:
                recovery_error = error
            self.sleep(0.05)
        if recovery_error is not None and not self.process_alive(self.observer_pid):
            raise SupervisorError(
                "possible resource leak: exact-name create remained uncertain and "
                "same-intent recovery observer handoff failed"
            ) from recovery_error
        return None

    def accept_created(self, root: dict) -> dict:
        supervisor = self._armed()
        try:
            binding = supervisor.bind_created(root)
            authorization = supervisor.authorize_bound_uid()
        except Exception:
            # After POST, the durable one-shot intent is enough for the live
            # observer to reconcile only this exact name and choose the one
            # ordinary-or-cleanup branch. Do not create a competing writer.
            self._wait_for_live_observer_or_uncertain()
            raise SupervisorError(
                "post-create binding did not complete; same-intent observer owns recovery"
            ) from None
        if supervisor.authorization is None:
            raise SupervisorError("release authorization was not persisted")
        try:
            takeover = self._wait_for_takeover(binding, authorization)
        except SupervisorError:
            if not self.process_alive(self.observer_pid):
                try:
                    self._wait_for_live_observer_or_uncertain()
                    takeover = self._wait_for_takeover(binding, authorization)
                except SupervisorError as recovery_exc:
                    self._release_after_takeover_failure()
                    raise SupervisorError(
                        "release supervisor takeover failed after bounded same-intent recovery"
                    ) from recovery_exc
            else:
                self._release_after_takeover_failure()
                raise SupervisorError(
                    "live release supervisor missed takeover; release intent was journaled"
                ) from None
        return {
            "release_binding_sha256": binding["sha256"],
            "release_authorization_sha256": supervisor.authorization["sha256"],
            "release_takeover_sha256": takeover["sha256"],
        }

    def cleanup_rejected_created(self, _root: dict) -> dict:
        """Reread and clean only the exact persisted root; never trust or accept a response."""
        supervisor = self._armed()
        root = self.reconcile_ambiguous_create()
        if root is None:
            raise SupervisorError(
                "possible resource leak: rejected exact-candidate root was not readable "
                "by its journaled name"
            )
        evidence = supervisor.cleanup_rejected_exact_root(root)
        recovery_error = self._ensure_recovery_observer()
        deadline = self.monotonic() + DELETE_CONFIRM_SECONDS
        while self.monotonic() < deadline:
            error = self._ensure_recovery_observer()
            if error is not None:
                recovery_error = error
            try:
                result = supervisor.reconcile_drift_cleanup_absence()
            except ReleaseKubernetesError:
                self.sleep(1)
                continue
            if result is not None:
                if result["release_confirmed"]:
                    return {**evidence, "release_result_sha256": result["sha256"]}
                raise SupervisorError(
                    "possible resource leak: cleanup-only full-chain release is uncertain"
                )
            self.sleep(1)
        result = supervisor.record_drift_cleanup_uncertain()
        detail = (
            f"; recovery handoff failed with {type(recovery_error).__name__}"
            if recovery_error is not None
            else ""
        )
        raise SupervisorError(
            "possible resource leak: cleanup-only full-chain release was not confirmed "
            f"({result['sha256']}{detail})"
        )

    def _wait_for_takeover(self, binding: dict, authorization: dict) -> dict:
        supervisor = self._armed()
        deadline = self.monotonic() + OBSERVER_TAKEOVER_SECONDS
        path = _takeover_path(self.operation_dir, self.recovery_token)
        while self.monotonic() < deadline:
            if path.exists():
                value = _validate_takeover_receipt(
                    _read_private_receipt(path, "takeover"),
                    supervisor,
                    binding,
                    authorization,
                    expected_pid=self.observer_pid,
                )
                if not self.process_alive(self.observer_pid):
                    raise SupervisorError("release supervisor died after takeover acknowledgement")
                return value
            if not self.process_alive(self.observer_pid):
                raise SupervisorError("release supervisor died before takeover acknowledgement")
            self.sleep(0.05)
        raise SupervisorError("release supervisor takeover deadline elapsed")

    def _start_recovery_observer(self) -> None:
        supervisor = self._armed()
        launched = self.launch_recovery_observer(self.operation_dir, supervisor)
        if not isinstance(launched, dict):
            raise SupervisorError("same-intent recovery observer receipt is missing")
        token = launched.get("recovery_token")
        pid = launched.get("observer_pid")
        if not isinstance(token, str) or not isinstance(pid, int):
            raise SupervisorError("same-intent recovery observer receipt is invalid")
        receipt = _validate_recovery_ready_receipt(
            _read_private_receipt(
                _recovery_ready_path(self.operation_dir, token), "recovery readiness"
            ),
            supervisor,
            token=token,
            pid=pid,
        )
        if launched != receipt or not self.process_alive(pid):
            raise SupervisorError("same-intent recovery observer is not live")
        self.observer_pid = pid
        self.recovery_token = token

    def _ensure_recovery_observer(self) -> SupervisorError | None:
        """Keep the guard read-only while it boundedly hands work to an observer."""
        if self.process_alive(self.observer_pid):
            return None
        try:
            self._start_recovery_observer()
        except SupervisorError as exc:
            return exc
        return None

    def _wait_for_live_observer_or_uncertain(self) -> None:
        """Retry the sealed handoff while rereading only the exact root."""
        supervisor = self._armed()
        deadline = self.monotonic() + OBSERVER_TAKEOVER_SECONDS
        recovery_error: SupervisorError | None = None
        while self.monotonic() < deadline:
            error = self._ensure_recovery_observer()
            if error is None:
                return
            recovery_error = error
            with suppress(ReleaseKubernetesError):
                self.backend.get_rayjob(supervisor.root_name)
            self.sleep(0.05)
        raise SupervisorError(
            "possible resource leak: no live same-intent release observer after "
            "bounded exact-name supervision"
        ) from recovery_error

    def _release_after_takeover_failure(self) -> None:
        supervisor = self._armed()
        delete_error: Exception | None = None
        try:
            supervisor.journal_bound_root_release_after_observer_failure()
        except Exception as exc:
            # The guard only journals. A live observer may re-drive the same
            # immutable UID-CAS request; this process remains read-only.
            delete_error = exc
        deadline = self.monotonic() + DELETE_CONFIRM_SECONDS
        zero_seen_at: float | None = None
        final: dict = {"reason": "release_confirmation_not_started"}
        while self.monotonic() < deadline:
            if not self.process_alive(self.observer_pid):
                try:
                    self._start_recovery_observer()
                except SupervisorError as exc:
                    delete_error = exc
            try:
                released, final = supervisor.fresh_release_observation()
                if released:
                    now = self.monotonic()
                    if zero_seen_at is not None and now - zero_seen_at >= POLL_SECONDS:
                        return
                    zero_seen_at = now
                else:
                    zero_seen_at = None
            except ReleaseKubernetesError:
                zero_seen_at = None
            self.sleep(1)
        detail = final.get("reason", "owned_resources_remain")
        if delete_error is not None:
            detail = type(delete_error).__name__ + ":" + str(detail)
        evidence = self._persist_guard_release_uncertainty(final, detail)
        raise SupervisorError(
            "possible resource leak: exact UID full-chain release could not be confirmed "
            f"after takeover failure ({detail}; {evidence['sha256']})"
        )

    def _persist_guard_release_uncertainty(self, final: dict, detail: str) -> dict:
        """Preserve the guard's final read-only evidence without minting RESULT."""
        supervisor = self._armed()
        if supervisor.binding is None:
            raise SupervisorError("guard uncertainty has no exact UID binding")
        delete_intent = _read_json(self.operation_dir / "DELETE_INTENT.json")
        if (
            delete_intent.get("sha256")
            != "sha256:"
            + digest({key: value for key, value in delete_intent.items() if key != "sha256"})
            or delete_intent.get("rayjob_name") != supervisor.root_name
            or delete_intent.get("rayjob_uid") != supervisor.binding["rayjob_uid"]
        ):
            raise SupervisorError("guard uncertainty delete intent is invalid")
        value = _seal(
            {
                "schema": GUARD_UNCERTAINTY_SCHEMA,
                "status": "release_uncertain_after_observer_handoff_failure",
                "binding_sha256": supervisor.binding["sha256"],
                "delete_intent_sha256": delete_intent["sha256"],
                "rayjob_name": supervisor.root_name,
                "rayjob_uid": supervisor.binding["rayjob_uid"],
                "fresh_final_relist": final,
                "detail": detail,
                "recorded_at": _stamp(self.clock()),
            }
        )
        path = self.operation_dir / "GUARD_RELEASE_UNCERTAINTY.json"
        try:
            _write_once(path, value)
        except FileExistsError:
            existing = _read_private_receipt(path, "guard uncertainty")
            body = {key: item for key, item in existing.items() if key != "sha256"}
            if (
                set(existing)
                != {
                    "schema",
                    "status",
                    "binding_sha256",
                    "delete_intent_sha256",
                    "rayjob_name",
                    "rayjob_uid",
                    "fresh_final_relist",
                    "detail",
                    "recorded_at",
                    "sha256",
                }
                or existing.get("schema") != GUARD_UNCERTAINTY_SCHEMA
                or existing.get("status") != "release_uncertain_after_observer_handoff_failure"
                or existing.get("binding_sha256") != supervisor.binding["sha256"]
                or existing.get("delete_intent_sha256") != delete_intent["sha256"]
                or existing.get("rayjob_name") != supervisor.root_name
                or existing.get("rayjob_uid") != supervisor.binding["rayjob_uid"]
                or not isinstance(existing.get("fresh_final_relist"), dict)
                or not isinstance(existing.get("detail"), str)
                or existing.get("sha256") != "sha256:" + digest(body)
            ):
                raise SupervisorError("guard uncertainty receipt is invalid") from None
            _parse_receipt_time(existing.get("recorded_at"))
            value = existing
        return value

    def _armed(self) -> FourNodeReleaseSupervisor:
        if self.supervisor is None:
            raise SupervisorError("release supervision is not armed")
        return self.supervisor


def _process_alive(pid: int) -> bool:
    process = _OBSERVER_PROCESSES.get(pid)
    if process is None:
        return False
    if process.poll() is None:
        return True
    _OBSERVER_PROCESSES.pop(pid, None)
    return False


def _local_started_visible() -> bool:
    root = Path(RUN_DIR).parent
    return root.is_dir() and os.access(root, os.R_OK | os.X_OK)


def _main(argv: list[str]) -> int:
    if (
        (argv[1:2] == ["observe"] and len(argv) != 8)
        or (argv[1:2] == ["recover"] and len(argv) != 9)
        or argv[1:2] not in (["observe"], ["recover"])
    ):
        return 2
    operation_dir = Path(argv[2])
    try:
        run_detached_release_observer(
            operation_dir,
            argv[3],
            argv[4],
            argv[5],
            argv[6],
            argv[7],
            recovery=argv[1] == "recover",
            recovery_token=argv[8] if argv[1] == "recover" else "",
        )
        return 0
    except BaseException as exc:
        if operation_dir.is_dir():
            failure = _seal(
                {
                    "schema": SCHEMA + ":observer_failure",
                    "status": "failed",
                    "error_class": type(exc).__name__,
                    "failed_at": _stamp(datetime.now(UTC)),
                }
            )
            with suppress(OSError, SupervisorError):
                _write_once(operation_dir / "OBSERVER_FAILURE.json", failure)
        return 1


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
