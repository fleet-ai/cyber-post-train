"""Arm a local, UID-bound cleanup observer for one exact workload.

The observer starts before the workload is created.  It records only Kubernetes
identity, lifecycle, resource and validated receipt metadata.  It never reads
container logs.  Once the exact workload becomes terminal, or reaches its
allocation-bound active-runtime deadline, the observer deletes that UID-bound
object and waits for all discovered children to disappear.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath
from urllib.parse import quote
from uuid import UUID

from cyber_post_train.jobs import digest, quantity
from scripts import probe_qwen38_prod8_terminal as prod8

DEV_CONTEXT = "nebius-mk8s-fleetai-training-dev-e04p03enwk5c0va9tb"
PROD_CONTEXT = "nebius-mk8s-fleetai-training-e04zw4ye1k7wczqdw6"
NAMESPACE = "fleet-train-jobs"
ARMED_SCHEMA = "cyber_dev_cleanup_observer_armed_v1"
RESULT_SCHEMA = "cyber_dev_cleanup_observer_result_v1"
DIRECT_ARMED_SCHEMA = "cyber_direct_cleanup_observer_armed_v1"
DIRECT_RESULT_SCHEMA = "cyber_direct_cleanup_observer_result_v1"
CREATOR_BINDING_SCHEMA = "cyber_direct_cleanup_creator_binding_v1"
PROD9_RELOAD_RESULT_SCHEMA = "cyber_skyrl_prod9_reload_observer_result_v1"
RECOVERY_ARMED_SCHEMA = "cyber_direct_cleanup_recovery_observer_armed_v1"
RECOVERY_RESULT_SCHEMA = "cyber_direct_cleanup_recovery_observer_result_v1"
JOBS_API_PREFIX_GUARD_SCHEMA = "cyber_jobs_api_prefix_guard_armed_v1"
JOBS_API_EXACT_BINDING_SCHEMA = "cyber_jobs_api_exact_rayjob_binding_v1"
JOBS_API_RELEASE_CONTRACT_SCHEMA = "cyber_jobs_api_exact_uid_release_contract_v1"
JOBS_API_EXACT_OBSERVER_SCHEMA = "cyber_jobs_api_exact_uid_observer_result_v1"
TERMINAL_RAY_STATUSES = {"SUCCEEDED": "Succeeded", "FAILED": "Failed"}
KUBECTL_ATTEMPTS = 3
KUBECTL_TIMEOUT_SECONDS = 20
MAX_CONSECUTIVE_OBSERVATION_FAILURES = 5
DELETE_REQUEST_MARGIN_SECONDS = 60
TERMINAL_RECEIPT_GRACE_SECONDS = 30
JOBS_API_POD_STARTUP_ALLOWANCE_SECONDS = 300
_RUN_NAME_PREFIX = re.compile(r"[a-z0-9](?:[-a-z0-9]{0,29}[a-z0-9])?")
_KUBERNETES_DNS_LABEL = re.compile(r"[a-z0-9](?:[-a-z0-9]{0,61}[a-z0-9])?")


class ObserverError(RuntimeError):
    """A sanitized cleanup-observer failure."""

    def __init__(self, message: str, *, code: str = "observer_error") -> None:
        super().__init__(message)
        self.code = code


class CleanupAuthorizedError(ObserverError):
    """Positive contract evidence that authorizes exact-target cleanup."""


def _seal(value: dict) -> dict:
    body = {key: item for key, item in value.items() if key != "sha256"}
    return {**body, "sha256": "sha256:" + digest(body)}


def _now() -> datetime:
    return datetime.now(UTC)


def _stamp(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_stamp(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)


def _write_create_once(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        stream.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")


def _canonical_jobs_run_dir(value: object) -> str:
    """Validate the public, per-run SFS path used by the generic Jobs API."""
    if not isinstance(value, str):
        raise ObserverError("Jobs API run directory is invalid")
    root = PurePosixPath(value)
    if (
        root.parts[:4] != ("/", "mnt", "sfs", "jobs")
        or len(root.parts) < 5
        or ".." in root.parts
        or str(root) != value
    ):
        raise ObserverError("Jobs API run directory is invalid")
    return value


def _digest_binding(value: object) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"sha256:[a-f0-9]{64}", value):
        raise ObserverError("cleanup observer digest binding is invalid")
    return value


class JobsApiPrefixGuard:
    """Non-destructive pre-POST guard for generated generic-Jobs RayJob names.

    The generic Jobs API assigns a suffix after POST, so an observer cannot know
    the exact Kubernetes name while it checks for a pre-existing collision.  This
    class deliberately stops at that safety check.  A later ``bind_exact`` call
    receives the creator-returned *exact* API identity and looks up only that
    name.  Prefix matching is never ownership evidence and this class never
    deletes, waits on, or releases a workload.
    """

    def __init__(
        self,
        *,
        context: str,
        namespace: str,
        run_name_prefix: str,
        run_dir: str,
        image: str,
        plan_sha256: str,
        manifest_sha256: str,
        maximum_seconds: int,
        expected_gpus: int,
        armed_path: Path,
        binding_path: Path,
        bind_wait_seconds: float = 120.0,
        run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if context not in {DEV_CONTEXT, PROD_CONTEXT} or namespace != NAMESPACE:
            raise ObserverError("Jobs API prefix guard context or namespace is invalid")
        if _RUN_NAME_PREFIX.fullmatch(run_name_prefix) is None:
            raise ObserverError("Jobs API run-name prefix is invalid")
        if not re.fullmatch(r"[^\s]+@sha256:[a-f0-9]{64}", image):
            raise ObserverError("Jobs API image binding is invalid")
        if (
            maximum_seconds < 1
            or maximum_seconds > (1800 if context == DEV_CONTEXT else 24 * 60 * 60)
            or expected_gpus not in {1, 8}
            or not 0 <= bind_wait_seconds <= 300
        ):
            raise ObserverError("Jobs API guard resource or deadline binding is invalid")
        # A separate post-POST process may need to bind the durable pre-POST
        # guard.  It may reuse the armed path only after ``bind_exact``
        # validates its sealed contents against these immutable constructor
        # bindings.  A binding receipt, in contrast, is single-use.
        if binding_path.exists():
            raise ObserverError("Jobs API exact binding evidence path already exists")
        self.context = context
        self.namespace = namespace
        self.run_name_prefix = run_name_prefix
        self.run_dir = _canonical_jobs_run_dir(run_dir)
        self.image = image
        self.plan_sha256 = _digest_binding(plan_sha256)
        self.manifest_sha256 = _digest_binding(manifest_sha256)
        self.maximum_seconds = maximum_seconds
        self.expected_gpus = expected_gpus
        self.armed_path = armed_path
        self.binding_path = binding_path
        self.bind_wait_seconds = bind_wait_seconds
        self._run = run
        self._sleep = sleep
        self._monotonic = monotonic
        self.armed_at = ""

    @property
    def generated_name_pattern(self) -> str:
        return "^" + re.escape(self.run_name_prefix) + r"-[a-f0-9]{8}$"

    def _kubectl(self, *arguments: str) -> str:
        command = [
            "kubectl",
            "--context",
            self.context,
            "--namespace",
            self.namespace,
            *arguments,
        ]
        result = None
        for attempt in range(KUBECTL_ATTEMPTS):
            try:
                result = self._run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=KUBECTL_TIMEOUT_SECONDS,
                )
                if result.returncode == 0:
                    return result.stdout
            except subprocess.TimeoutExpired:
                result = None
            if attempt + 1 < KUBECTL_ATTEMPTS:
                time.sleep(1)
        if result is None:
            raise ObserverError("Jobs API guard observation timed out", code="kubectl_timeout")
        raise ObserverError("Jobs API guard observation failed", code="kubectl_failed")

    def _list_rayjobs(self) -> list[dict]:
        output = self._kubectl("get", "rayjob", "--output", "json")
        try:
            value = json.loads(output)
        except ValueError as exc:
            raise ObserverError("Jobs API guard response was not JSON") from exc
        rows = value.get("items") if isinstance(value, dict) else None
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise ObserverError("Jobs API guard response had the wrong shape")
        return rows

    def _get_exact_rayjob(self, name: str) -> dict | None:
        output = self._kubectl(
            "get", "rayjob", name, "--ignore-not-found", "--output", "json"
        ).strip()
        if not output:
            return None
        try:
            value = json.loads(output)
        except ValueError as exc:
            raise ObserverError("Jobs API exact RayJob response was not JSON") from exc
        if not isinstance(value, dict):
            raise ObserverError("Jobs API exact RayJob response had the wrong shape")
        return value

    def _prefix_collisions(self, rows: list[dict]) -> int:
        prefix = self.run_name_prefix + "-"
        count = 0
        for row in rows:
            metadata = row.get("metadata")
            name = metadata.get("name") if isinstance(metadata, dict) else None
            if isinstance(name, str) and (name == self.run_name_prefix or name.startswith(prefix)):
                count += 1
        return count

    def _validate_armed(self, value: object) -> dict:
        if not isinstance(value, dict):
            raise ObserverError("Jobs API prefix guard receipt is invalid")
        fields = {
            "schema",
            "status",
            "context",
            "namespace",
            "run_name_prefix",
            "generated_name_pattern",
            "run_dir",
            "image",
            "plan_sha256",
            "manifest_sha256",
            "maximum_seconds",
            "expected_gpus",
            "armed_at",
            "observer_pid",
            "prefix_collision_count_before_post",
            "sha256",
        }
        body = {key: item for key, item in value.items() if key != "sha256"}
        if (
            set(value) != fields
            or value.get("schema") != JOBS_API_PREFIX_GUARD_SCHEMA
            or value.get("status") != "armed_non_destructive_prefix_guard"
            or value.get("sha256") != "sha256:" + digest(body)
            or value.get("context") != self.context
            or value.get("namespace") != self.namespace
            or value.get("run_name_prefix") != self.run_name_prefix
            or value.get("generated_name_pattern") != self.generated_name_pattern
            or value.get("run_dir") != self.run_dir
            or value.get("image") != self.image
            or value.get("plan_sha256") != self.plan_sha256
            or value.get("manifest_sha256") != self.manifest_sha256
            or value.get("maximum_seconds") != self.maximum_seconds
            or value.get("expected_gpus") != self.expected_gpus
            or value.get("prefix_collision_count_before_post") != 0
        ):
            raise ObserverError("Jobs API prefix guard receipt is invalid")
        try:
            _parse_stamp(value["armed_at"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ObserverError("Jobs API prefix guard receipt is invalid") from exc
        if type(value.get("observer_pid")) is not int or value["observer_pid"] < 1:
            raise ObserverError("Jobs API prefix guard receipt is invalid")
        return value

    def arm(self) -> dict:
        if self.armed_at or self.armed_path.exists() or self.binding_path.exists():
            raise ObserverError("Jobs API prefix guard was already armed")
        collisions = self._prefix_collisions(self._list_rayjobs())
        if collisions:
            raise ObserverError("Jobs API generated-name prefix is already in use")
        self.armed_at = _stamp(_now())
        value = _seal(
            {
                "schema": JOBS_API_PREFIX_GUARD_SCHEMA,
                "status": "armed_non_destructive_prefix_guard",
                "context": self.context,
                "namespace": self.namespace,
                "run_name_prefix": self.run_name_prefix,
                "generated_name_pattern": self.generated_name_pattern,
                "run_dir": self.run_dir,
                "image": self.image,
                "plan_sha256": self.plan_sha256,
                "manifest_sha256": self.manifest_sha256,
                "maximum_seconds": self.maximum_seconds,
                "expected_gpus": self.expected_gpus,
                "armed_at": self.armed_at,
                "observer_pid": os.getpid(),
                "prefix_collision_count_before_post": 0,
            }
        )
        _write_create_once(self.armed_path, value)
        return value

    def _creator_identity(self, value: Mapping[str, object]) -> tuple[str, str]:
        expected = {"jobs_api_run_name", "jobs_api_run_id", "run_dir"}
        if set(value) != expected:
            raise ObserverError("Jobs API creator identity has an unexpected shape")
        name, run_id, run_dir = (
            value.get("jobs_api_run_name"),
            value.get("jobs_api_run_id"),
            value.get("run_dir"),
        )
        if (
            not isinstance(name, str)
            or re.fullmatch(self.generated_name_pattern, name) is None
            or not isinstance(run_id, str)
            or run_dir != self.run_dir
        ):
            raise ObserverError("Jobs API creator identity is invalid")
        try:
            UUID(run_id)
        except ValueError as exc:
            raise ObserverError("Jobs API creator identity is invalid") from exc
        return name, run_id

    def _validate_exact_rayjob(self, resource: dict, *, name: str) -> tuple[str, str]:
        metadata = resource.get("metadata")
        spec = resource.get("spec")
        if (
            resource.get("kind") != "RayJob"
            or not isinstance(metadata, dict)
            or not isinstance(spec, dict)
        ):
            raise ObserverError("Jobs API exact RayJob is malformed")
        uid, created = Observer._metadata(resource)
        annotations = metadata.get("annotations")
        labels = metadata.get("labels")
        if (
            metadata.get("namespace") != self.namespace
            or metadata.get("name") != name
            or not isinstance(annotations, dict)
            or annotations.get("fleet.ai/run-dir") != self.run_dir
            or annotations.get("fleet.ai/failure-alerts") != "off"
            or not isinstance(labels, dict)
            or labels.get("kueue.x-k8s.io/queue-name") != "training-lq"
            or labels.get("kueue.x-k8s.io/priority-class") != "q1"
            or labels.get("fleet.ai/requeue-if-preempted") != "false"
            or spec.get("shutdownAfterJobFinishes") is not True
            or spec.get("backoffLimit") != 0
        ):
            raise ObserverError("Jobs API exact RayJob does not match the armed guard")
        if _parse_stamp(created) < _parse_stamp(self.armed_at):
            raise ObserverError("Jobs API exact RayJob predates the armed guard")
        try:
            cluster = spec["rayClusterSpec"]
            head = cluster["headGroupSpec"]
            head_pod = head["template"]["spec"]
            workers = cluster.get("workerGroupSpecs", [])
            if (
                not isinstance(head_pod, dict)
                or not isinstance(workers, list)
                or any(
                    not isinstance(group, dict)
                    or type(group.get("replicas")) is not int
                    or group["replicas"] < 0
                    for group in workers
                )
            ):
                raise TypeError
            if 1 + sum(group["replicas"] for group in workers) != 1:
                raise ValueError
            if head_pod.get("priorityClassName") != "c1":
                raise ValueError
            containers = head_pod["containers"]
            if not isinstance(containers, list):
                raise TypeError
            gpu_containers = []
            for container in containers:
                if not isinstance(container, dict):
                    raise TypeError
                resources = container.get("resources", {})
                requested = quantity(resources.get("requests", {}).get("nvidia.com/gpu", 0))
                limited = quantity(resources.get("limits", {}).get("nvidia.com/gpu", 0))
                if requested != limited:
                    raise ValueError
                if limited:
                    gpu_containers.append((container, limited))
            if (
                len(gpu_containers) != 1
                or gpu_containers[0][0].get("image") != self.image
                or gpu_containers[0][1] != self.expected_gpus
            ):
                raise ValueError
            init_containers = head_pod.get("initContainers", [])
            if not isinstance(init_containers, list):
                raise TypeError
            for container in init_containers:
                if not isinstance(container, dict):
                    raise TypeError
                resources = container.get("resources", {})
                if quantity(resources.get("requests", {}).get("nvidia.com/gpu", 0)) or quantity(
                    resources.get("limits", {}).get("nvidia.com/gpu", 0)
                ):
                    raise ValueError
        except (KeyError, TypeError, ValueError):
            raise ObserverError("Jobs API exact RayJob resource binding changed") from None
        return uid, created

    def bind_exact(self, creator_identity: Mapping[str, object]) -> dict:
        """Bind one exact API-returned name; this method never performs cleanup.

        ``creator_identity`` must be a narrow, caller-normalized copy of the
        authenticated API response.  In particular, prefix discovery after POST
        is intentionally prohibited: only ``jobs_api_run_name`` is queried.
        """
        if not self.armed_at:
            try:
                stored = json.loads(self.armed_path.read_text())
            except (OSError, ValueError) as exc:
                raise ObserverError("Jobs API prefix guard has not been armed") from exc
            self.armed_at = self._validate_armed(stored)["armed_at"]
        else:
            try:
                stored = json.loads(self.armed_path.read_text())
            except (OSError, ValueError) as exc:
                raise ObserverError("Jobs API prefix guard receipt is unavailable") from exc
            self._validate_armed(stored)
        if self.binding_path.exists():
            raise ObserverError("Jobs API exact binding already exists")
        name, run_id = self._creator_identity(creator_identity)
        deadline = self._monotonic() + self.bind_wait_seconds
        while True:
            resource = self._get_exact_rayjob(name)
            if resource is not None:
                break
            remaining = deadline - self._monotonic()
            if remaining <= 0:
                raise ObserverError("Jobs API exact RayJob is absent after bounded bind wait")
            self._sleep(min(1.0, remaining))
        uid, created = self._validate_exact_rayjob(resource, name=name)
        self._validate_creator_run_id(resource, name=name, run_id=run_id)
        value = _seal(
            {
                "schema": JOBS_API_EXACT_BINDING_SCHEMA,
                "status": "bound_exact_uid_cleanup_not_started",
                "prefix_guard_sha256": stored["sha256"],
                "context": self.context,
                "namespace": self.namespace,
                "jobs_api_run_name": name,
                "jobs_api_run_id": run_id,
                "run_dir": self.run_dir,
                "image": self.image,
                "rayjob_name": name,
                "rayjob_uid": uid,
                "rayjob_created_at": created,
                "bound_at": _stamp(_now()),
                "failure_alerts": "off",
                "maximum_seconds": self.maximum_seconds,
                "expected_gpus": self.expected_gpus,
                "cleanup_started": False,
            }
        )
        _write_create_once(self.binding_path, value)
        return value

    def reconcile_accepted_post(
        self,
        *,
        title: str,
        jobs_reader: Callable[[], list[dict]],
        returned_name: str | None,
    ) -> tuple[dict, dict]:
        """Bind one ambiguously accepted POST without ever repeating the POST.

        Reconciliation is intentionally narrower than ordinary discovery.  The
        armed guard must be durable and there must be exactly one Jobs row and
        one RayJob related by either immutable
        name/output signal.  The selected pair must then agree on both signals
        and on the canonical server run UUID.  The deployed history schema does
        not expose title, so its sealed request hash remains journal evidence.
        """
        if not self.armed_at:
            try:
                stored = json.loads(self.armed_path.read_text())
            except (OSError, ValueError) as exc:
                raise ObserverError("Jobs API prefix guard has not been armed") from exc
            self.armed_at = self._validate_armed(stored)["armed_at"]
        else:
            try:
                stored = json.loads(self.armed_path.read_text())
            except (OSError, ValueError) as exc:
                raise ObserverError("Jobs API prefix guard receipt is unavailable") from exc
            self._validate_armed(stored)
        if self.binding_path.exists():
            raise ObserverError("Jobs API exact binding already exists")
        if not isinstance(title, str) or not title or len(title) > 256:
            raise ObserverError("Jobs API reconciliation title is invalid")
        if returned_name is not None and (
            not isinstance(returned_name, str)
            or re.fullmatch(self.generated_name_pattern, returned_name) is None
        ):
            raise ObserverError("Jobs API reconciliation returned name is invalid")
        deadline = self._monotonic() + self.bind_wait_seconds
        while True:
            rows = jobs_reader()
            if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
                raise ObserverError("Jobs API reconciliation rows have the wrong shape")
            rayjobs = self._list_rayjobs()
            related_rows = [
                row
                for row in rows
                if (
                    re.fullmatch(self.generated_name_pattern, str(row.get("name", "")))
                    or row.get("run_dir") == self.run_dir
                )
            ]
            related_rayjobs = []
            for row in rayjobs:
                metadata = row.get("metadata")
                if not isinstance(metadata, dict):
                    continue
                annotations = metadata.get("annotations")
                name = metadata.get("name")
                if (isinstance(name, str) and re.fullmatch(self.generated_name_pattern, name)) or (
                    isinstance(annotations, dict)
                    and annotations.get("fleet.ai/run-dir") == self.run_dir
                ):
                    related_rayjobs.append(row)
            if len(related_rows) > 1 or len(related_rayjobs) > 1:
                raise ObserverError("Jobs API reconciliation identity is not unique")
            if len(related_rows) == len(related_rayjobs) == 1:
                break
            remaining = deadline - self._monotonic()
            if remaining <= 0:
                raise ObserverError("Jobs API accepted POST is absent after bounded reconciliation")
            self._sleep(min(1.0, remaining))
        api_row, resource = related_rows[0], related_rayjobs[0]
        name = api_row.get("name")
        if (
            not isinstance(name, str)
            or re.fullmatch(self.generated_name_pattern, name) is None
            or api_row.get("run_dir") != self.run_dir
            or (returned_name is not None and returned_name != name)
        ):
            raise ObserverError("Jobs API reconciliation row does not match the armed intent")
        listed_metadata = resource.get("metadata")
        if not isinstance(listed_metadata, dict) or listed_metadata.get("name") != name:
            raise ObserverError("Jobs API reconciliation RayJob name does not agree")
        exact = self._get_exact_rayjob(name)
        if exact is None or exact.get("metadata", {}).get("uid") != listed_metadata.get("uid"):
            raise ObserverError("Jobs API reconciliation exact RayJob changed")
        resource = exact
        uid, created = self._validate_exact_rayjob(resource, name=name)
        canonical_run_id = self._rayjob_run_id(resource, name=name)
        api_run_ids = [api_row.get(field) for field in ("job_id", "run_id")]
        if any(run_id is not None and run_id != canonical_run_id for run_id in api_run_ids):
            raise ObserverError("Jobs API reconciliation run ID does not agree")
        value = _seal(
            {
                "schema": JOBS_API_EXACT_BINDING_SCHEMA,
                "status": "bound_exact_uid_cleanup_not_started",
                "prefix_guard_sha256": stored["sha256"],
                "context": self.context,
                "namespace": self.namespace,
                "jobs_api_run_name": name,
                "jobs_api_run_id": canonical_run_id,
                "run_dir": self.run_dir,
                "image": self.image,
                "rayjob_name": name,
                "rayjob_uid": uid,
                "rayjob_created_at": created,
                "bound_at": _stamp(_now()),
                "failure_alerts": "off",
                "maximum_seconds": self.maximum_seconds,
                "expected_gpus": self.expected_gpus,
                "cleanup_started": False,
            }
        )
        _write_create_once(self.binding_path, value)
        return value, {
            "jobs_api_rows_checked": len(rows),
            "kubernetes_rayjobs_checked": len(rayjobs),
            "matching_jobs_api_rows": len(related_rows),
            "matching_kubernetes_rayjobs": len(related_rayjobs),
            "jobs_api_status": api_row.get("status"),
            "jobs_api_created_at": api_row.get("created_at"),
        }

    def _rayjob_run_id(self, resource: dict, *, name: str) -> str:
        metadata = resource["metadata"]
        annotation_run_id = metadata["annotations"].get("fleet.ai/run-id")
        label_run_id = metadata["labels"].get("fleet.ai/run-id")
        try:
            canonical_run_id = str(UUID(annotation_run_id))
        except (TypeError, ValueError) as exc:
            raise ObserverError("Jobs API exact RayJob run ID evidence is invalid") from exc
        if (
            annotation_run_id != canonical_run_id
            or label_run_id != canonical_run_id
            or name != self.run_name_prefix + "-" + canonical_run_id.split("-", 1)[0]
        ):
            raise ObserverError("Jobs API exact RayJob run ID evidence does not agree")
        return canonical_run_id

    def _validate_creator_run_id(self, resource: dict, *, name: str, run_id: str) -> None:
        del resource, name, run_id


class Prod10JobsApiPrefixGuard(JobsApiPrefixGuard):
    """Prod10-only exact topology and server-run-ID binding."""

    def _validate_exact_rayjob(self, resource: dict, *, name: str) -> tuple[str, str]:
        uid, created = super()._validate_exact_rayjob(resource, name=name)
        try:
            cluster = resource["spec"]["rayClusterSpec"]
            head = cluster["headGroupSpec"]
            annotations = head["template"]["metadata"]["annotations"]
            pod = head["template"]["spec"]
            if (
                cluster.get("workerGroupSpecs", []) != []
                or annotations.get("kueue.x-k8s.io/podset-preferred-topology")
                != "topology.nebius.com/tier-1"
                or "kueue.x-k8s.io/podset-required-topology" in annotations
                or pod.get("nodeSelector")
                != {"kubernetes.io/os": "linux", "workload": "fleetai-training-ng-gpu"}
            ):
                raise ValueError
        except (KeyError, TypeError, ValueError):
            raise ObserverError("prod10 Jobs API exact RayJob topology changed") from None
        return uid, created

    def _validate_creator_run_id(self, resource: dict, *, name: str, run_id: str) -> None:
        if self._rayjob_run_id(resource, name=name) != run_id:
            raise ObserverError("Jobs API response and exact RayJob run IDs do not agree")


class JobsApiExactUidObserver:
    """Bounded, exact-UID observer for one generic-Jobs-created RayJob.

    This is deliberately not a generic cleanup framework.  It consumes the
    sealed post-POST binding from :class:`JobsApiPrefixGuard`, reads only that
    RayJob and children whose owner UID proves the relationship, and never
    selects resources by prefix.  It does not create workloads or read logs.

    A terminal RayJob normally removes itself because the rendered object has
    ``shutdownAfterJobFinishes``.  The only destructive fallback is an exact
    Kubernetes raw DELETE of that one bound RayJob, with a UID precondition,
    and it is unavailable unless a separately sealed creator/observer contract
    authorizes it.  There is intentionally no generic-Jobs run-name deletion:
    a mutable API name is not deletion authority.
    """

    def __init__(
        self,
        *,
        binding_path: Path,
        result_path: Path,
        release_contract_path: Path | None = None,
        poll_seconds: float = 2.0,
        run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    ) -> None:
        if result_path.exists():
            raise ObserverError("Jobs API exact observer evidence path already exists")
        if poll_seconds <= 0:
            raise ObserverError("Jobs API exact observer poll interval is invalid")
        try:
            binding = json.loads(binding_path.read_text())
        except (OSError, ValueError) as exc:
            raise ObserverError("Jobs API exact binding evidence is unavailable") from exc
        self.binding = self._validate_binding(binding)
        self.binding_path = binding_path
        self.result_path = result_path
        self.poll_seconds = poll_seconds
        self._run = run
        self.context = self.binding["context"]
        self.namespace = self.binding["namespace"]
        self.root_name = self.binding["rayjob_name"]
        self.root_uid = self.binding["rayjob_uid"]
        self.created_at = self.binding["rayjob_created_at"]
        self.maximum_seconds = self.binding["maximum_seconds"]
        # Queueing is not active GPU runtime.  The exact Jobs API object may
        # legitimately wait for scheduler admission longer than the plan's
        # active-runtime allowance, so do not derive a destructive deadline
        # from RayJob creation.  Once an owned GPU Pod exists, its immutable
        # creation timestamp starts the bounded active-runtime clock.
        self.contract_deadline_at = _parse_stamp(self.created_at) + timedelta(
            seconds=self.maximum_seconds
        )
        self.allocated_at: datetime | None = None
        self.deadline_at: datetime | None = None
        self.release_contract = self._load_release_contract(release_contract_path)
        self.root_seen = False
        self.terminal_status = ""
        self.inventory_seen = False
        self.raycluster_identity_observed = False
        self.pod_inventory_deadline_at: datetime | None = None
        self.cleanup_requested = False
        self.cleanup_status = (
            "authorized_not_requested" if self.release_contract is not None else "not_authorized"
        )
        self.peak_gpus = 0
        self.gpu_pods: dict[str, dict[str, object]] = {}
        self.runtime_images: dict[str, dict[str, str]] = {}
        self.receipt: dict | None = None
        self.pod_restarts: dict[str, int] = {}
        self.pod_exit_codes: dict[str, tuple[int, ...]] = {}
        self.known: dict[str, dict[str, str]] = {
            "workload": {},
            "raycluster": {},
            "pod": {},
        }

    def _validate_binding(self, value: object) -> dict:
        fields = {
            "schema",
            "status",
            "prefix_guard_sha256",
            "context",
            "namespace",
            "jobs_api_run_name",
            "jobs_api_run_id",
            "run_dir",
            "image",
            "rayjob_name",
            "rayjob_uid",
            "rayjob_created_at",
            "bound_at",
            "failure_alerts",
            "maximum_seconds",
            "expected_gpus",
            "cleanup_started",
            "sha256",
        }
        if not isinstance(value, dict):
            raise ObserverError("Jobs API exact binding evidence is invalid")
        body = {key: item for key, item in value.items() if key != "sha256"}
        if (
            set(value) != fields
            or value.get("schema") != JOBS_API_EXACT_BINDING_SCHEMA
            or value.get("status") != "bound_exact_uid_cleanup_not_started"
            or value.get("sha256") != "sha256:" + digest(body)
            or value.get("context") not in {DEV_CONTEXT, PROD_CONTEXT}
            or value.get("namespace") != NAMESPACE
            or value.get("failure_alerts") != "off"
            or value.get("expected_gpus") not in {1, 8}
            or value.get("cleanup_started") is not False
            or value.get("rayjob_name") != value.get("jobs_api_run_name")
        ):
            raise ObserverError("Jobs API exact binding evidence is invalid")
        try:
            UUID(value["jobs_api_run_id"])
            UUID(value["rayjob_uid"])
            _parse_stamp(value["rayjob_created_at"])
            _parse_stamp(value["bound_at"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ObserverError("Jobs API exact binding evidence is invalid") from exc
        if (
            not isinstance(value.get("rayjob_name"), str)
            or _KUBERNETES_DNS_LABEL.fullmatch(value["rayjob_name"]) is None
            or _canonical_jobs_run_dir(value.get("run_dir")) != value["run_dir"]
            or not isinstance(value.get("image"), str)
            or re.fullmatch(r"[^\s]+@sha256:[a-f0-9]{64}", value["image"]) is None
            or not 1
            <= value.get("maximum_seconds", 0)
            <= (1800 if value.get("context") == DEV_CONTEXT else 24 * 60 * 60)
            or not isinstance(value.get("prefix_guard_sha256"), str)
            or not re.fullmatch(r"sha256:[a-f0-9]{64}", value["prefix_guard_sha256"])
        ):
            raise ObserverError("Jobs API exact binding evidence is invalid")
        return value

    def _load_release_contract(self, path: Path | None) -> dict | None:
        if path is None:
            return None
        try:
            value = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            raise ObserverError("Jobs API release contract is unavailable") from exc
        fields = {
            "schema",
            "status",
            "binding_sha256",
            "context",
            "namespace",
            "jobs_api_run_name",
            "jobs_api_run_id",
            "rayjob_name",
            "rayjob_uid",
            "authorized_at",
            "release_route",
            "sha256",
        }
        if not isinstance(value, dict):
            raise ObserverError("Jobs API release contract is invalid")
        body = {key: item for key, item in value.items() if key != "sha256"}
        if (
            set(value) != fields
            or value.get("schema") != JOBS_API_RELEASE_CONTRACT_SCHEMA
            or value.get("status") != "creator_authorized_exact_uid_release"
            or value.get("sha256") != "sha256:" + digest(body)
            or value.get("binding_sha256") != self.binding["sha256"]
            or value.get("context") != self.context
            or value.get("namespace") != self.namespace
            or value.get("jobs_api_run_name") != self.binding["jobs_api_run_name"]
            or value.get("jobs_api_run_id") != self.binding["jobs_api_run_id"]
            or value.get("rayjob_name") != self.root_name
            or value.get("rayjob_uid") != self.root_uid
            or value.get("release_route") != "raw_rayjob_uid_precondition_v1"
        ):
            raise ObserverError("Jobs API release contract is invalid")
        try:
            authorized = _parse_stamp(value["authorized_at"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ObserverError("Jobs API release contract is invalid") from exc
        if authorized > self.contract_deadline_at:
            raise ObserverError("Jobs API release contract is outside the bound deadline")
        return value

    def _kubectl(self, *arguments: str, input_text: str | None = None) -> str:
        command = [
            "kubectl",
            "--context",
            self.context,
            "--namespace",
            self.namespace,
            *arguments,
        ]
        result = None
        for attempt in range(KUBECTL_ATTEMPTS):
            try:
                result = self._run(
                    command,
                    input=input_text,
                    capture_output=True,
                    text=True,
                    timeout=KUBECTL_TIMEOUT_SECONDS,
                )
                if result.returncode == 0:
                    return result.stdout
            except subprocess.TimeoutExpired:
                result = None
            if attempt + 1 < KUBECTL_ATTEMPTS:
                time.sleep(min(self.poll_seconds, 1.0))
        if result is None:
            raise ObserverError("Jobs API exact observer timed out", code="kubectl_timeout")
        raise ObserverError("Jobs API exact observer failed", code="kubectl_failed")

    def _get(self, resource: str, name: str) -> dict | None:
        output = self._kubectl(
            "get", resource, name, "--ignore-not-found", "--output", "json"
        ).strip()
        if not output:
            return None
        try:
            value = json.loads(output)
        except ValueError as exc:
            raise ObserverError("Jobs API exact observer response was not JSON") from exc
        if not isinstance(value, dict):
            raise ObserverError("Jobs API exact observer response had the wrong shape")
        return value

    def _list_owned(self, resource: str, selector: str) -> list[dict]:
        output = self._kubectl("get", resource, "--selector", selector, "--output", "json")
        try:
            value = json.loads(output)
        except ValueError as exc:
            raise ObserverError("Jobs API exact observer response was not JSON") from exc
        items = value.get("items") if isinstance(value, dict) else None
        if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
            raise ObserverError("Jobs API exact observer response had the wrong shape")
        return items

    @staticmethod
    def _metadata(resource: dict, *, expected_kind: str) -> tuple[str, str, str]:
        metadata = resource.get("metadata")
        if resource.get("kind") != expected_kind or not isinstance(metadata, dict):
            raise ObserverError("Jobs API exact observer resource is malformed")
        name, uid, created = (
            metadata.get("name"),
            metadata.get("uid"),
            metadata.get("creationTimestamp"),
        )
        try:
            UUID(uid)
            _parse_stamp(created)
        except (TypeError, ValueError) as exc:
            raise ObserverError("Jobs API exact observer resource identity is invalid") from exc
        if not isinstance(name, str) or not name:
            raise ObserverError("Jobs API exact observer resource identity is invalid")
        return name, uid, created

    @staticmethod
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

    def _record_known(self, resource: str, *, name: str, uid: str) -> None:
        existing = self.known[resource].get(name)
        if existing is not None and existing != uid:
            raise ObserverError("Jobs API exact observer detected UID name reuse")
        self.known[resource][name] = uid

    @staticmethod
    def _pod_gpus(pod: dict) -> int:
        spec = pod.get("spec")
        if not isinstance(spec, dict):
            raise ObserverError("Jobs API exact observer Pod is malformed")

        def gpu(container: object) -> int:
            if not isinstance(container, dict):
                raise ObserverError("Jobs API exact observer Pod is malformed")
            resources = container.get("resources", {})
            if not isinstance(resources, dict):
                raise ObserverError("Jobs API exact observer Pod is malformed")
            requests = resources.get("requests", {})
            limits = resources.get("limits", {})
            if not isinstance(requests, dict) or not isinstance(limits, dict):
                raise ObserverError("Jobs API exact observer Pod is malformed")
            requested = quantity(requests.get("nvidia.com/gpu", 0))
            limited = quantity(limits.get("nvidia.com/gpu", 0))
            if requested != limited or requested != int(requested):
                raise ObserverError("Jobs API exact observer Pod GPU contract changed")
            return int(requested)

        containers = spec.get("containers")
        init_containers = spec.get("initContainers", [])
        if not isinstance(containers, list) or not isinstance(init_containers, list):
            raise ObserverError("Jobs API exact observer Pod is malformed")
        regular = sum(gpu(container) for container in containers)
        restartable = sum(
            gpu(container)
            for container in init_containers
            if isinstance(container, dict) and container.get("restartPolicy") == "Always"
        )
        ordinary = [
            gpu(container)
            for container in init_containers
            if not isinstance(container, dict) or container.get("restartPolicy") != "Always"
        ]
        return regular + restartable + max(ordinary, default=0)

    def _observe_runtime_image(self, pod: dict, *, name: str, uid: str, gpus: int) -> None:
        """Bind the GPU container's actual imageID to the requested digest."""
        if gpus == 0:
            return
        spec = pod.get("spec")
        status = pod.get("status", {})
        if not isinstance(spec, dict) or not isinstance(status, dict):
            raise ObserverError("Jobs API exact observer Pod is malformed")
        gpu_containers = []
        for container in spec.get("containers", []):
            if not isinstance(container, dict):
                raise ObserverError("Jobs API exact observer Pod is malformed")
            resources = container.get("resources", {})
            requests = resources.get("requests", {}) if isinstance(resources, dict) else {}
            limits = resources.get("limits", {}) if isinstance(resources, dict) else {}
            requested = quantity(requests.get("nvidia.com/gpu", 0))
            limited = quantity(limits.get("nvidia.com/gpu", 0))
            if requested or limited:
                gpu_containers.append(container)
        if (
            len(gpu_containers) != 1
            or gpu_containers[0].get("image") != self.binding["image"]
            or not isinstance(gpu_containers[0].get("name"), str)
            or not gpu_containers[0]["name"]
        ):
            raise ObserverError("Jobs API exact observer Pod image binding changed")
        self.gpu_pods[name] = {"uid": uid, "gpus": gpus}
        statuses = status.get("containerStatuses", [])
        if not isinstance(statuses, list) or any(not isinstance(item, dict) for item in statuses):
            raise ObserverError("Jobs API exact observer Pod status is malformed")
        matches = [item for item in statuses if item.get("name") == gpu_containers[0]["name"]]
        if not matches:
            return
        if len(matches) != 1:
            raise ObserverError("Jobs API exact observer Pod status is ambiguous")
        image_id = matches[0].get("imageID")
        if image_id in (None, ""):
            return
        if not isinstance(image_id, str):
            raise ObserverError("Jobs API exact observer runtime imageID is invalid")
        match = re.search(r"(sha256:[a-f0-9]{64})$", image_id)
        expected = self.binding["image"].rsplit("@", 1)[1]
        if match is None or match.group(1) != expected:
            raise ObserverError("Jobs API exact observer runtime image digest changed")
        observed = {"image_id": image_id, "digest": match.group(1)}
        if name in self.runtime_images and self.runtime_images[name] != observed:
            raise ObserverError("Jobs API exact observer runtime image identity changed")
        self.runtime_images[name] = observed

    def _runtime_image_identity_complete(self) -> bool:
        return (
            len(self.gpu_pods) == 1
            and sum(int(value["gpus"]) for value in self.gpu_pods.values())
            == self.binding["expected_gpus"]
            and set(self.runtime_images) == set(self.gpu_pods)
        )

    def _validate_root(self, resource: dict) -> str:
        name, uid, created = self._metadata(resource, expected_kind="RayJob")
        metadata = resource["metadata"]
        annotations = metadata.get("annotations")
        if (
            name != self.root_name
            or uid != self.root_uid
            or created != self.created_at
            or metadata.get("namespace") != self.namespace
            or not isinstance(annotations, dict)
            or annotations.get("fleet.ai/run-dir") != self.binding["run_dir"]
            or annotations.get("fleet.ai/failure-alerts") != "off"
        ):
            raise ObserverError("Jobs API exact observer root binding changed")
        status = resource.get("status", {})
        if not isinstance(status, dict):
            raise ObserverError("Jobs API exact observer root is malformed")
        terminal = TERMINAL_RAY_STATUSES.get(status.get("jobStatus"), "")
        if terminal:
            self.terminal_status = terminal
        cluster_name = status.get("rayClusterName", "")
        if cluster_name and (
            not isinstance(cluster_name, str)
            or _KUBERNETES_DNS_LABEL.fullmatch(cluster_name) is None
        ):
            raise ObserverError("Jobs API exact observer RayCluster name is invalid")
        return cluster_name

    def _observe_owned_children(self, cluster_name: str) -> None:
        workloads = self._list_owned("workload", f"kueue.x-k8s.io/job-uid={self.root_uid}")
        if len(workloads) > 1:
            raise ObserverError("Jobs API exact observer found multiple owned Workloads")
        for workload in workloads:
            name, uid, _ = self._metadata(workload, expected_kind="Workload")
            if not self._owned_by(workload, kind="RayJob", name=self.root_name, uid=self.root_uid):
                raise ObserverError("Jobs API exact observer Workload owner binding changed")
            self._record_known("workload", name=name, uid=uid)
        # An empty or already-gone RayCluster name cannot prove that all GPU
        # children were observed.  Keep polling while the root exists, but
        # never turn that gap into a release confirmation after the root goes
        # away.  The safe terminal result in that case is ``release_uncertain``.
        if not cluster_name:
            return
        cluster = self._get("raycluster", cluster_name)
        if cluster is None:
            return
        name, uid, created = self._metadata(cluster, expected_kind="RayCluster")
        if not self._owned_by(cluster, kind="RayJob", name=self.root_name, uid=self.root_uid):
            raise ObserverError("Jobs API exact observer RayCluster owner binding changed")
        self._record_known("raycluster", name=name, uid=uid)
        self.raycluster_identity_observed = True
        if self.pod_inventory_deadline_at is None:
            self.pod_inventory_deadline_at = _parse_stamp(created) + timedelta(
                seconds=JOBS_API_POD_STARTUP_ALLOWANCE_SECONDS
            )
        try:
            pods = self._list_owned("pod", f"ray.io/cluster={name}")
        except ObserverError as exc:
            if exc.code == "observer_error" and not self.inventory_seen and not self.known["pod"]:
                if _now() < self.pod_inventory_deadline_at:
                    return
                raise ObserverError(
                    "Jobs API exact observer Pod inventory startup allowance elapsed",
                    code="pod_inventory_startup_timeout",
                ) from exc
            raise
        if not pods:
            if _now() < self.pod_inventory_deadline_at:
                return
            raise ObserverError(
                "Jobs API exact observer Pod inventory did not appear within startup allowance",
                code="pod_inventory_startup_timeout",
            )
        for pod in pods:
            try:
                pod_name, pod_uid, pod_created = self._metadata(pod, expected_kind="Pod")
                if not self._owned_by(pod, kind="RayCluster", name=name, uid=uid):
                    raise ObserverError("Jobs API exact observer Pod owner binding changed")
            except ObserverError as exc:
                if (
                    exc.code == "observer_error"
                    and not self.inventory_seen
                    and not self.known["pod"]
                ):
                    if _now() < self.pod_inventory_deadline_at:
                        return
                    raise ObserverError(
                        "Jobs API exact observer Pod inventory startup allowance elapsed",
                        code="pod_inventory_startup_timeout",
                    ) from exc
                raise
            self._record_known("pod", name=pod_name, uid=pod_uid)
            pod_gpus = self._pod_gpus(pod)
            self._observe_runtime_image(pod, name=pod_name, uid=pod_uid, gpus=pod_gpus)
            self.peak_gpus = max(self.peak_gpus, pod_gpus)
            if self.peak_gpus > self.binding["expected_gpus"]:
                raise ObserverError("Jobs API exact observer GPU contract exceeded")
            if pod_gpus > 0:
                pod_allocated_at = _parse_stamp(pod_created)
                if self.allocated_at is None or pod_allocated_at < self.allocated_at:
                    self.allocated_at = pod_allocated_at
                    self.deadline_at = self.allocated_at + timedelta(seconds=self.maximum_seconds)
            status = pod.get("status") or {}
            statuses = [
                *(status.get("initContainerStatuses") or []),
                *(status.get("containerStatuses") or []),
            ]
            if any(not isinstance(item, dict) for item in statuses):
                raise ObserverError("Jobs API exact observer Pod status is malformed")
            restarts = []
            exit_codes = []
            for item in statuses:
                restart_count = item.get("restartCount", 0)
                if type(restart_count) is not int or restart_count < 0:
                    raise ObserverError("Jobs API exact observer Pod restart count is invalid")
                restarts.append(restart_count)
                terminated = (item.get("state") or {}).get("terminated")
                if terminated is None:
                    continue
                if not isinstance(terminated, dict) or type(terminated.get("exitCode")) is not int:
                    raise ObserverError("Jobs API exact observer termination state is malformed")
                exit_codes.append(terminated["exitCode"])
                message = terminated.get("message")
                if message:
                    receipt = _validated_receipt(message, kind="rayjob")
                    if receipt is None:
                        raise ObserverError(
                            "Jobs API exact observer termination receipt is invalid"
                        )
                    if self.receipt is not None and self.receipt != receipt:
                        raise ObserverError("Jobs API exact observer termination receipt changed")
                    self.receipt = receipt
            self.pod_restarts[pod_uid] = max(restarts, default=0)
            if exit_codes:
                self.pod_exit_codes[pod_uid] = tuple(exit_codes)
        self.inventory_seen = True

    def _known_children_absent(self) -> bool:
        kinds = {"workload": "Workload", "raycluster": "RayCluster", "pod": "Pod"}
        for resource, identities in self.known.items():
            expected_kind = kinds[resource]
            for name, uid in identities.items():
                value = self._get(resource, name)
                if value is None:
                    continue
                _, observed_uid, _ = self._metadata(value, expected_kind=expected_kind)
                if observed_uid != uid:
                    raise ObserverError("Jobs API exact observer detected UID name reuse")
                return False
        return True

    def _request_exact_uid_cleanup(self) -> None:
        if self.release_contract is None or self.cleanup_requested:
            return
        # Re-read only the bound root immediately before the destructive call.
        root = self._get("rayjob", self.root_name)
        if root is None:
            self.cleanup_status = "not_requested_root_already_absent"
            return
        self._validate_root(root)
        body = json.dumps(
            {
                "apiVersion": "v1",
                "kind": "DeleteOptions",
                "propagationPolicy": "Foreground",
                "preconditions": {"uid": self.root_uid},
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        self._kubectl(
            "delete",
            "--raw",
            f"/apis/ray.io/v1/namespaces/{quote(self.namespace, safe='')}/"
            f"rayjobs/{quote(self.root_name, safe='')}",
            "-f",
            "-",
            input_text=body,
        )
        self.cleanup_requested = True
        self.cleanup_status = "requested_exact_uid_precondition"

    def _result(self, *, status: str, reason: str, release_confirmed: bool) -> dict:
        value = _seal(
            {
                "schema": JOBS_API_EXACT_OBSERVER_SCHEMA,
                "status": status,
                "reason": reason,
                "release_confirmed": release_confirmed,
                "context": self.context,
                "namespace": self.namespace,
                "binding_sha256": self.binding["sha256"],
                "jobs_api_run_name": self.binding["jobs_api_run_name"],
                "jobs_api_run_id": self.binding["jobs_api_run_id"],
                "rayjob_name": self.root_name,
                "rayjob_uid": self.root_uid,
                "requested_image": self.binding["image"],
                "created_at": self.created_at,
                "allocated_at": _stamp(self.allocated_at) if self.allocated_at else "",
                "deadline_at": _stamp(self.deadline_at) if self.deadline_at else "",
                "maximum_seconds": self.maximum_seconds,
                "terminal_status": self.terminal_status,
                "owned_inventory_observed": self.inventory_seen,
                "raycluster_identity_observed": self.raycluster_identity_observed,
                "workloads": [
                    {"name": name, "uid": uid}
                    for name, uid in sorted(self.known["workload"].items())
                ],
                "rayclusters": [
                    {"name": name, "uid": uid}
                    for name, uid in sorted(self.known["raycluster"].items())
                ],
                "pods": [
                    {
                        "name": name,
                        "uid": uid,
                        "gpus": int(self.gpu_pods.get(name, {}).get("gpus", 0)),
                        "runtime_image_id": self.runtime_images.get(name, {}).get("image_id", ""),
                        "runtime_image_digest": self.runtime_images.get(name, {}).get("digest", ""),
                    }
                    for name, uid in sorted(self.known["pod"].items())
                ],
                "runtime_image_identity_complete": self._runtime_image_identity_complete(),
                "peak_gpus": self.peak_gpus,
                "active_gpus": 0 if release_confirmed else self.peak_gpus,
                "restarts": sum(self.pod_restarts.values()),
                "exit_codes": sorted(
                    code for codes in self.pod_exit_codes.values() for code in codes
                ),
                "receipt": self.receipt,
                "cleanup_status": self.cleanup_status,
                "cleanup_requested": self.cleanup_requested,
                "private_logs_read": False,
            }
        )
        _write_create_once(self.result_path, value)
        return value

    def run(self) -> dict:
        """Observe to terminal/release or the allocation-bound active deadline."""
        while True:
            try:
                root = self._get("rayjob", self.root_name)
                if root is None:
                    if not self.root_seen:
                        return self._result(
                            status="release_uncertain",
                            reason="bound_root_absent_before_observation",
                            release_confirmed=False,
                        )
                    if (
                        not self.terminal_status
                        or not self.inventory_seen
                        or not self.raycluster_identity_observed
                    ):
                        return self._result(
                            status="release_uncertain",
                            reason="bound_root_absent_without_terminal_inventory",
                            release_confirmed=False,
                        )
                    if self._known_children_absent():
                        if not self._runtime_image_identity_complete():
                            return self._result(
                                status="release_uncertain",
                                reason="runtime_image_identity_not_observed",
                                release_confirmed=False,
                            )
                        return self._result(
                            status="released_after_terminal",
                            reason="exact_root_and_observed_children_absent",
                            release_confirmed=True,
                        )
                    if self.deadline_at is not None and _now() >= self.deadline_at:
                        return self._result(
                            status="release_uncertain",
                            reason="owned_child_still_present_at_deadline",
                            release_confirmed=False,
                        )
                    time.sleep(self.poll_seconds)
                    continue

                self.root_seen = True
                cluster_name = self._validate_root(root)
                self._observe_owned_children(cluster_name)
                deadline_reached = self.deadline_at is not None and _now() >= self.deadline_at
                if (
                    self.terminal_status
                    and self._runtime_image_identity_complete()
                    and not self.cleanup_requested
                ):
                    self._request_exact_uid_cleanup()
                    continue
                if deadline_reached:
                    if not self.cleanup_requested:
                        self._request_exact_uid_cleanup()
                        if self.cleanup_requested:
                            continue
                    return self._result(
                        status="release_uncertain",
                        reason="allocation_bound_deadline_elapsed",
                        release_confirmed=False,
                    )
                if self.deadline_at is None:
                    time.sleep(self.poll_seconds)
                else:
                    time.sleep(
                        min(
                            self.poll_seconds,
                            max(0.01, (self.deadline_at - _now()).total_seconds()),
                        )
                    )
            except ObserverError as exc:
                return self._result(
                    status="release_uncertain",
                    reason=exc.code,
                    release_confirmed=False,
                )


def _validated_receipt(message: object, *, kind: str) -> dict | None:
    if not isinstance(message, str) or not message:
        return None
    # Kubernetes bounds a termination message by bytes, but refuse an
    # obviously oversized Python string before allocating a second UTF-8
    # buffer.  The post-encode cap remains authoritative for multibyte text.
    if len(message) > 16384:
        return None
    try:
        raw = message.encode("utf-8")
    except UnicodeEncodeError:
        return None
    # Preserve the legacy Kubernetes termination-message bound before schema
    # dispatch.  The v2 reader applies its stricter 3500-byte raw cap before
    # parsing its canonical receipt a second time below.
    if len(raw) > 16384:
        return None
    try:
        value = prod8.parse_receipt_json(raw)
    except (TypeError, ValueError, RecursionError):
        return None
    if not isinstance(value, dict):
        return None
    if value.get("schema") == prod8.RECEIPT_SCHEMA:
        try:
            return prod8.validate_canonical_receipt_bytes(raw)
        except ValueError:
            return None
    # Fresh prod9 CPU gates write their fixed-path receipt with a
    # ``receipt_sha256`` field because Kubernetes' termination file is not a
    # create-once artifact.  The prod9 direct rail validates every semantic
    # field before accepting either receipt; this observer only needs to bind
    # the sealed object to the exact terminal Pod without parsing private data.
    if kind == "job" and value.get("schema") in {
        "cyber_skyrl_prod9_rebind_stage_receipt_v1",
        "cyber_skyrl_prod9_training_cpu_preflight_v1",
    }:
        body = {key: item for key, item in value.items() if key != "receipt_sha256"}
        return value if value.get("receipt_sha256") == digest(body) else None
    if kind == "rayjob" and value.get("schema") == "cyber_hf_export_check_v1":
        body = {key: item for key, item in value.items() if key != "receipt_sha256"}
        return value if value.get("receipt_sha256") == digest(body) else None
    if kind == "rayjob" and value.get("status") == "native_loop_returned":
        body = {key: item for key, item in value.items() if key != "sha256"}
        if set(value) == {
            "status",
            "plan_sha256",
            "checkpoint_global_step",
            "completed_batches",
            "completed_at",
            "optimizer_update_independently_verified",
            "checkpoint_reload_verified",
            "sha256",
        } and value.get("sha256") == digest(body):
            return value
        return None
    body = {key: item for key, item in value.items() if key != "sha256"}
    schemas = {
        "job": {
            "cyber_skyrl_topology_probe_cpu_preflight_v1",
            "cyber_skyrl_topology_probe_cpu_preflight_rejection_v1",
            "cyber_skyrl_model_artifact_stage_receipt_v1",
            "cyber_skyrl_model_artifact_stage_rejection_v1",
            "cyber_skyrl_topology_probe_receipt_verification_v1",
            "cyber_skyrl_topology_probe_receipt_verification_rejection_v1",
            "cyber_skyrl_reward_data_stage_receipt_v1",
            "cyber_skyrl_reward_cpu_preflight_v1",
            "cyber_skyrl_reward_cpu_preflight_rejection_v1",
            "cyber_skyrl_prod10_operator_termination_v1",
            "cyber_skyrl_prod10_rebound_manifest_result_v1",
            "cyber_skyrl_prod10_operator_failure_v1",
            "cyber_skyrl_prod10_bootstrap_failure_v1",
            "cyber_skyrl_prod10_preview_difference_result_v1",
        },
        "fleetjob": {
            "cyber_skyrl_topology_probe_receipt_v1",
            "cyber_skyrl_topology_probe_failure_v1",
        },
        "rayjob": {
            "cyber_skyrl_topology_probe_receipt_v1",
            "cyber_skyrl_topology_probe_failure_v1",
        },
    }
    if value.get("schema") not in schemas[kind] or value.get("sha256") != "sha256:" + digest(body):
        return None
    return value


def _receipt_execution_accepted(value: dict | None) -> bool:
    if value is None:
        return False
    if value.get("schema") == "cyber_skyrl_prod10_preview_difference_result_v1":
        return value.get("status") == "diagnostic_completed"
    if value.get("schema") != prod8.RECEIPT_SCHEMA:
        return value.get("status") in {
            "passed",
            "published",
            "setup_and_internal_cleanup_passed",
            "native_loop_returned",
        }
    return prod8.receipt_execution_accepted(value)


@dataclass
class Snapshot:
    uid: str = ""
    created_at: str = ""
    terminal_status: str = ""
    job_id: str = ""
    rayjob_name: str = ""
    rayjob_uid: str = ""
    workload_name: str = ""
    workload_uid: str = ""
    raycluster_name: str = ""
    raycluster_uid: str = ""
    pod_names: set[str] = field(default_factory=set)
    pod_uids: set[str] = field(default_factory=set)
    image_ids: set[str] = field(default_factory=set)
    exit_codes: set[int] = field(default_factory=set)
    termination_reasons: set[str] = field(default_factory=set)
    restarts: int = 0
    peak_gpus: int = 0
    receipt: dict | None = None
    prod8_normalized_manifest: dict | None = None
    prod8_job_template: dict | None = None
    prod8_last_listed_pod_count: int = 0
    prod8_container_observations: dict[str, dict] = field(default_factory=dict)
    prod8_receipt_bound: bool = False
    prod8_manifest_error: str = ""
    prod8_binding_error: str = ""
    prod8_receipt_error: str = ""


class Observer:
    def __init__(
        self,
        *,
        context: str,
        namespace: str,
        kind: str,
        name: str,
        maximum_seconds: int,
        expected_gpus: int,
        plan_sha256: str,
        manifest_sha256: str,
        armed_path: Path,
        result_path: Path,
        poll_seconds: float = 2.0,
        release_seconds: int = 300,
        profile: str = "development",
        expected_uid: str = "",
        run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    ) -> None:
        profiles = {
            "development": (DEV_CONTEXT, ARMED_SCHEMA, RESULT_SCHEMA, 1800),
            "production-direct": (
                PROD_CONTEXT,
                DIRECT_ARMED_SCHEMA,
                DIRECT_RESULT_SCHEMA,
                16 * 60 * 60,
            ),
            "production-cpu": (
                PROD_CONTEXT,
                DIRECT_ARMED_SCHEMA,
                DIRECT_RESULT_SCHEMA,
                1800,
            ),
            "production-operator": (
                PROD_CONTEXT,
                DIRECT_ARMED_SCHEMA,
                DIRECT_RESULT_SCHEMA,
                2700,
            ),
            "production-reload": (
                PROD_CONTEXT,
                DIRECT_ARMED_SCHEMA,
                PROD9_RELOAD_RESULT_SCHEMA,
                1800,
            ),
            "production-recovery": (
                PROD_CONTEXT,
                RECOVERY_ARMED_SCHEMA,
                RECOVERY_RESULT_SCHEMA,
                16 * 60 * 60,
            ),
        }
        if profile not in profiles:
            raise ObserverError("cleanup observer profile is invalid")
        expected_context, armed_schema, result_schema, maximum_bound = profiles[profile]
        if context != expected_context or namespace != NAMESPACE:
            if profile == "development":
                raise ObserverError("cleanup observer is bound to the development cluster")
            raise ObserverError("cleanup observer production binding is invalid")
        if kind not in {"job", "fleetjob", "rayjob"}:
            raise ObserverError("cleanup observer kind is invalid")
        if not name or maximum_seconds < 1 or maximum_seconds > maximum_bound:
            raise ObserverError("cleanup observer deadline is invalid")
        if profile == "production-direct" and kind != "rayjob":
            raise ObserverError("production direct observer requires a root RayJob")
        if profile == "production-recovery" and kind != "rayjob":
            raise ObserverError("production recovery observer requires a root RayJob")
        if profile == "production-cpu" and kind != "job":
            raise ObserverError("production CPU observer requires a root Job")
        if profile == "production-operator" and kind != "job":
            raise ObserverError("production operator observer requires a root Job")
        if profile == "production-reload" and kind != "rayjob":
            raise ObserverError("production reload observer requires a root RayJob")
        is_prod8_terminal_probe = kind == "job" and name == prod8.NAME
        if profile == "production-recovery":
            try:
                UUID(expected_uid)
            except (TypeError, ValueError) as exc:
                raise ObserverError("production recovery observer requires an exact UID") from exc
        elif is_prod8_terminal_probe:
            if profile != "production-cpu":
                raise ObserverError("prod8 terminal probe requires the production CPU observer")
            try:
                UUID(expected_uid)
            except (TypeError, ValueError) as exc:
                raise ObserverError(
                    "prod8 terminal probe requires its creation-bound exact UID"
                ) from exc
        elif expected_uid:
            raise ObserverError("only a recovery observer may bind an existing UID")
        if expected_gpus not in {0, 1, 8} or (kind == "job") != (expected_gpus == 0):
            raise ObserverError("cleanup observer GPU contract is invalid")
        for value in (plan_sha256, manifest_sha256):
            if len(value.removeprefix("sha256:")) != 64:
                raise ObserverError("cleanup observer digest binding is invalid")
        if (
            is_prod8_terminal_probe
            and plan_sha256.removeprefix("sha256:") != prod8.TRAINING_PLAN_SHA256
        ):
            raise ObserverError("prod8 terminal probe training-plan binding is invalid")
        if armed_path.exists() or result_path.exists():
            raise ObserverError("cleanup observer evidence path already exists")
        self.context = context
        self.namespace = namespace
        self.kind = kind
        self.name = name
        self.maximum_seconds = maximum_seconds
        self.expected_gpus = expected_gpus
        self.plan_sha256 = plan_sha256
        self.manifest_sha256 = manifest_sha256
        self.armed_path = armed_path
        self.result_path = result_path
        self.poll_seconds = poll_seconds
        self.release_seconds = release_seconds
        self.profile = profile
        self.armed_schema = armed_schema
        self.result_schema = result_schema
        self.expected_uid = expected_uid
        self.creator_binding_path = armed_path.with_name(armed_path.name + ".created.json")
        self.requires_creator_binding = profile in {
            "production-direct",
            "production-cpu",
            "production-operator",
            "production-reload",
        } and not (is_prod8_terminal_probe or expected_uid)
        if self.requires_creator_binding and (
            self.creator_binding_path.exists() or self.creator_binding_path.is_symlink()
        ):
            raise ObserverError("cleanup observer creator binding already exists")
        self._run = run
        self.snapshot = Snapshot()
        self.armed_at = ""
        self.deletion_requested_at = ""
        self.deletion_reason = ""
        self.observation_failures = 0
        self.max_consecutive_observation_failures = 0
        self.last_observation_error_code = ""

    def _is_prod8_terminal_probe(self) -> bool:
        return self.kind == "job" and self.name == prod8.NAME

    def _is_fatal_prod8_binding_error(self, error: ObserverError) -> bool:
        return self._is_prod8_terminal_probe() and error.code in {
            "prod8_manifest_binding_failed",
            "prod8_creation_uid_mismatch",
            "prod8_pod_binding_failed",
            "prod8_receipt_rejected",
        }

    def _kubectl(self, *arguments: str, input_text: str | None = None) -> str:
        command = [
            "kubectl",
            "--context",
            self.context,
            "--namespace",
            self.namespace,
            *arguments,
        ]
        result = None
        for attempt in range(KUBECTL_ATTEMPTS):
            try:
                result = self._run(
                    command,
                    input=input_text,
                    capture_output=True,
                    text=True,
                    timeout=KUBECTL_TIMEOUT_SECONDS,
                )
                if result.returncode == 0:
                    break
            except subprocess.TimeoutExpired:
                result = None
            if attempt + 1 < KUBECTL_ATTEMPTS:
                time.sleep(min(self.poll_seconds, 1.0))
        if result is None:  # pragma: no cover - the loop either returns or raises
            raise ObserverError(
                "development-cluster observation timed out",
                code="kubectl_timeout",
            )
        if result.returncode:
            raise ObserverError(
                "development-cluster observation failed",
                code="kubectl_failed",
            )
        return result.stdout

    def _get(self, resource: str, name: str) -> dict | None:
        output = self._kubectl(
            "get", resource, name, "--ignore-not-found", "--output", "json"
        ).strip()
        if not output:
            return None
        try:
            value = json.loads(output)
        except ValueError as exc:
            raise ObserverError("development-cluster response was not JSON") from exc
        if not isinstance(value, dict):
            raise ObserverError("development-cluster response had the wrong shape")
        return value

    def _list(self, resource: str, *arguments: str) -> list[dict]:
        output = self._kubectl("get", resource, *arguments, "--output", "json")
        try:
            value = json.loads(output)
        except ValueError as exc:
            raise ObserverError("development-cluster response was not JSON") from exc
        items = value.get("items") if isinstance(value, dict) else None
        if not isinstance(items, list) or not all(isinstance(row, dict) for row in items):
            raise ObserverError("development-cluster list had the wrong shape")
        return items

    def _target_resource(self) -> str:
        return self.kind

    def _target(self) -> dict | None:
        return self._get(self._target_resource(), self.name)

    def arm(self) -> dict:
        target = self._target()
        recovery = {}
        if self.expected_uid:
            if target is None and not self._is_prod8_terminal_probe():
                raise ObserverError("recovery cleanup target is absent")
            if target is not None:
                uid, created = self._metadata(target)
                if uid != self.expected_uid:
                    raise ObserverError("recovery cleanup target UID differs")
                recovery = (
                    {
                        "creation_bound_uid": uid,
                        "target_created_at": created,
                    }
                    if self._is_prod8_terminal_probe()
                    else {
                        "recovered_existing_target": True,
                        "expected_uid": uid,
                        "target_created_at": created,
                    }
                )
            elif self._is_prod8_terminal_probe():
                recovery = {
                    "creation_bound_uid": self.expected_uid,
                    "target_created_at": "",
                }
        elif target is not None:
            raise ObserverError("cleanup target already exists")
        self.armed_at = _stamp(_now())
        evidence = _seal(
            {
                "schema": self.armed_schema,
                "status": "armed",
                "context": self.context,
                "namespace": self.namespace,
                "kind": self.kind,
                "name": self.name,
                "maximum_seconds": self.maximum_seconds,
                "expected_gpus": self.expected_gpus,
                "plan_sha256": self.plan_sha256,
                "manifest_sha256": self.manifest_sha256,
                "armed_at": self.armed_at,
                "observer_pid": os.getpid(),
                **(
                    {"creator_binding_path": str(self.creator_binding_path)}
                    if self.requires_creator_binding
                    else {}
                ),
                **recovery,
            }
        )
        _write_create_once(self.armed_path, evidence)
        return evidence

    @staticmethod
    def _metadata(resource: dict) -> tuple[str, str]:
        metadata = resource.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ObserverError("workload identity metadata is invalid")
        uid, created = metadata.get("uid"), metadata.get("creationTimestamp")
        try:
            UUID(uid)
            _parse_stamp(created)
        except (TypeError, ValueError) as exc:
            raise ObserverError("workload identity metadata is invalid") from exc
        return uid, created

    def _bind(self, resource: dict) -> None:
        uid, created = self._metadata(resource)
        if self.expected_uid and uid != self.expected_uid:
            raise ObserverError(
                "cleanup target UID changed",
                code=(
                    "prod8_creation_uid_mismatch"
                    if self._is_prod8_terminal_probe()
                    else "observer_error"
                ),
            )
        if self.snapshot.uid and uid != self.snapshot.uid:
            raise ObserverError("cleanup target name was reused with another UID")
        if not self.expected_uid and _parse_stamp(created) < _parse_stamp(self.armed_at):
            raise ObserverError("cleanup target predates the armed observer")
        self.snapshot.uid = uid
        self.snapshot.created_at = created

    def _creator_uid(self) -> str | None:
        """Load the create response handoff before adopting a production name."""
        if not self.requires_creator_binding:
            return self.expected_uid or None
        path = self.creator_binding_path
        if not path.exists():
            return None
        if path.is_symlink() or not path.is_file():
            raise ObserverError("cleanup observer creator binding is indirect")
        try:
            value = json.loads(path.read_bytes())
        except (OSError, ValueError) as exc:
            raise ObserverError("cleanup observer creator binding is invalid") from exc
        body = {key: item for key, item in value.items() if key != "sha256"}
        try:
            uid = str(UUID(value.get("uid")))
        except (TypeError, ValueError) as exc:
            raise ObserverError("cleanup observer creator binding UID is invalid") from exc
        if (
            value.get("schema") != CREATOR_BINDING_SCHEMA
            or value.get("sha256") != "sha256:" + digest(body)
            or value.get("status") != "created_once"
            or value.get("context") != self.context
            or value.get("namespace") != self.namespace
            or value.get("kind") != self.kind
            or value.get("name") != self.name
            or value.get("plan_sha256") != self.plan_sha256
            or value.get("manifest_sha256") != self.manifest_sha256
        ):
            raise ObserverError("cleanup observer creator binding changed")
        self.expected_uid = uid
        self.requires_creator_binding = False
        return uid

    def _bind_prod8_manifest(self, resource: dict) -> None:
        if self.manifest_sha256.removeprefix("sha256:") != prod8.manifest_digest():
            self.snapshot.prod8_manifest_error = "manifest_digest_mismatch"
            return
        try:
            normalized = prod8.normalized_manifest(resource)
        except ValueError:
            self.snapshot.prod8_manifest_error = "manifest_malformed"
            return
        if not prod8._exactly_equal(normalized, prod8.expected_normalized_manifest()):
            self.snapshot.prod8_manifest_error = "manifest_mismatch"
            return
        template = resource.get("spec", {}).get("template", {}).get("spec")
        if not isinstance(template, dict):
            self.snapshot.prod8_manifest_error = "manifest_malformed"
            return
        self.snapshot.prod8_normalized_manifest = normalized
        self.snapshot.prod8_job_template = template

    def _capture_prod8_pods(self, pods: list[dict], *, job_template: object) -> None:
        expected_digest = prod8.IMAGE.rsplit("@", 1)[1]
        for pod in pods:
            try:
                uid, _ = self._metadata(pod)
            except ObserverError:
                self.snapshot.prod8_binding_error = "pod_identity_invalid"
                continue
            metadata = pod.get("metadata", {})
            name = metadata.get("name")
            if not isinstance(name, str) or not name:
                self.snapshot.prod8_binding_error = "pod_name_missing"
                continue
            if (
                self.snapshot.prod8_container_observations
                and uid not in self.snapshot.prod8_container_observations
            ):
                self.snapshot.prod8_binding_error = "multiple_pods"
                continue
            try:
                observation = prod8.normalized_terminal_pod(
                    pod,
                    job_uid=self.snapshot.uid,
                    job_template=job_template,
                )
            except ValueError:
                self.snapshot.prod8_binding_error = "pod_binding_mismatch"
                continue
            message = observation.pop("termination_message")
            self.snapshot.pod_names.add(name)
            self.snapshot.pod_uids.add(uid)
            self.snapshot.prod8_container_observations[uid] = observation
            runtime_image = observation.get("runtime_image_id")
            if isinstance(runtime_image, str):
                self.snapshot.image_ids.add(runtime_image)
            restart_count = observation["restart_count"]
            self.snapshot.restarts = max(self.snapshot.restarts, restart_count)
            exit_code = observation.get("exit_code")
            if exit_code is not None:
                self.snapshot.exit_codes.add(exit_code)
            reason = observation.get("termination_reason")
            if isinstance(reason, str) and reason:
                self.snapshot.termination_reasons.add(reason)
            receipt = _validated_receipt(message, kind=self.kind) if message is not None else None
            if message is not None and receipt is None:
                # A termination message that is present but cannot prove the
                # sealed v2 contract is a receipt rejection, not permission
                # to delete the otherwise creation-bound Job.
                self.snapshot.prod8_receipt_error = "receipt_invalid"
                continue
            if (
                not self.snapshot.prod8_binding_error
                and observation.get("runtime_image_digest") == expected_digest
                and observation.get("exit_code") == 0
                and restart_count == 0
                and receipt is not None
            ):
                self.snapshot.receipt = receipt
                self.snapshot.prod8_receipt_bound = True

    def _capture_pods(self, pods: list[dict], *, job_template: object | None = None) -> None:
        if self._is_prod8_terminal_probe():
            template = (
                job_template if job_template is not None else self.snapshot.prod8_job_template
            )
            if template is None:
                self.snapshot.prod8_binding_error = "job_template_missing"
                return
            self._capture_prod8_pods(pods, job_template=template)
            return
        current_gpus = 0
        for pod in pods:
            uid, _ = self._metadata(pod)
            metadata = pod.get("metadata", {})
            name = metadata.get("name")
            if not isinstance(name, str) or not name:
                raise ObserverError("observed pod has no name")
            self.snapshot.pod_names.add(name)
            self.snapshot.pod_uids.add(uid)
            spec = pod.get("spec", {})
            for container in spec.get("containers", []):
                requests = container.get("resources", {}).get("requests", {})
                current_gpus += int(quantity(requests.get("nvidia.com/gpu", 0)))
            for status in pod.get("status", {}).get("containerStatuses", []):
                self.snapshot.restarts = max(
                    self.snapshot.restarts, int(status.get("restartCount", 0))
                )
                image_id = status.get("imageID")
                if isinstance(image_id, str) and image_id:
                    self.snapshot.image_ids.add(image_id)
                terminated = status.get("state", {}).get("terminated")
                if isinstance(terminated, dict):
                    exit_code = terminated.get("exitCode")
                    reason = terminated.get("reason")
                    if type(exit_code) is int:
                        self.snapshot.exit_codes.add(exit_code)
                    if isinstance(reason, str) and reason:
                        self.snapshot.termination_reasons.add(reason)
                    receipt = _validated_receipt(terminated.get("message"), kind=self.kind)
                    if receipt is not None:
                        self.snapshot.receipt = receipt
        self.snapshot.peak_gpus = max(self.snapshot.peak_gpus, current_gpus)

    def _observe_job(self, resource: dict) -> None:
        if self._is_prod8_terminal_probe():
            job_template = self.snapshot.prod8_job_template
            pods = self._list(
                "pod",
                "--selector",
                f"batch.kubernetes.io/controller-uid={self.snapshot.uid}",
            )
            self.snapshot.prod8_last_listed_pod_count = len(pods)
            self._capture_pods(pods, job_template=job_template)
        else:
            pods = self._list("pod", "--selector", f"job-name={self.name}")
            self._capture_pods(pods)
        labels = resource.get("metadata", {}).get("labels", {})
        queued = (
            isinstance(labels, dict)
            and labels.get("kueue.x-k8s.io/queue-name") == "training-lq"
            and labels.get("kueue.x-k8s.io/priority-class") == "q1"
        )
        workloads = (
            self._list("workload", "--selector", f"kueue.x-k8s.io/job-uid={self.snapshot.uid}")
            if queued
            else []
        )
        if len(workloads) > 1:
            raise ObserverError("more than one Workload owns the exact Job")
        if workloads:
            workload = workloads[0]
            workload_name = workload.get("metadata", {}).get("name")
            workload_uid, _ = self._metadata(workload)
            owners = workload.get("metadata", {}).get("ownerReferences", [])
            if (
                not isinstance(workload_name, str)
                or not workload_name
                or not any(
                    owner.get("kind") == "Job"
                    and owner.get("name") == self.name
                    and owner.get("uid") == self.snapshot.uid
                    for owner in owners
                    if isinstance(owner, dict)
                )
            ):
                raise ObserverError("Kueue Workload does not own the exact Job")
            if self.snapshot.workload_uid and self.snapshot.workload_uid != workload_uid:
                raise ObserverError("Workload UID changed")
            self.snapshot.workload_name = workload_name
            self.snapshot.workload_uid = workload_uid
        conditions = resource.get("status", {}).get("conditions", [])
        for condition in conditions:
            if condition.get("status") != "True":
                continue
            if condition.get("type") == "Complete":
                self.snapshot.terminal_status = "Succeeded"
            elif condition.get("type") == "Failed":
                self.snapshot.terminal_status = "Failed"

    def _find_workload(self, rayjob_name: str) -> dict | None:
        matches = []
        for workload in self._list("workload"):
            owners = workload.get("metadata", {}).get("ownerReferences", [])
            if any(owner.get("name") == rayjob_name for owner in owners):
                matches.append(workload)
        if len(matches) > 1:
            raise ObserverError("more than one Workload owns the exact RayJob")
        return matches[0] if matches else None

    def _observe_fleetjob(self, resource: dict) -> None:
        status = resource.get("status", {})
        job_id = status.get("jobId")
        rayjob_name = status.get("rayJobName")
        if isinstance(job_id, str) and job_id:
            if self.snapshot.job_id and self.snapshot.job_id != job_id:
                raise ObserverError("FleetJob API job identity changed")
            try:
                UUID(job_id)
            except ValueError as exc:
                raise ObserverError("FleetJob API job identity is invalid") from exc
            self.snapshot.job_id = job_id
        if isinstance(rayjob_name, str) and rayjob_name:
            if self.snapshot.rayjob_name and self.snapshot.rayjob_name != rayjob_name:
                raise ObserverError("FleetJob RayJob identity changed")
            self.snapshot.rayjob_name = rayjob_name
        if not self.snapshot.rayjob_name:
            return
        rayjob = self._get("rayjob", self.snapshot.rayjob_name)
        if rayjob is None:
            return
        rayjob_uid, _ = self._metadata(rayjob)
        if self.snapshot.rayjob_uid and self.snapshot.rayjob_uid != rayjob_uid:
            raise ObserverError("RayJob UID changed")
        self.snapshot.rayjob_uid = rayjob_uid
        ray_status = rayjob.get("status", {})
        cluster_name = ray_status.get("rayClusterName")
        if isinstance(cluster_name, str) and cluster_name:
            if self.snapshot.raycluster_name and self.snapshot.raycluster_name != cluster_name:
                raise ObserverError("RayCluster identity changed")
            self.snapshot.raycluster_name = cluster_name
        terminal = TERMINAL_RAY_STATUSES.get(ray_status.get("jobStatus"))
        if terminal:
            self.snapshot.terminal_status = terminal
        workload = self._find_workload(self.snapshot.rayjob_name)
        if workload is not None:
            name = workload.get("metadata", {}).get("name")
            uid, _ = self._metadata(workload)
            if not isinstance(name, str) or not name:
                raise ObserverError("Workload identity is invalid")
            if self.snapshot.workload_uid and self.snapshot.workload_uid != uid:
                raise ObserverError("Workload UID changed")
            self.snapshot.workload_name = name
            self.snapshot.workload_uid = uid
        if not self.snapshot.raycluster_name:
            return
        cluster = self._get("raycluster", self.snapshot.raycluster_name)
        if cluster is not None:
            uid, _ = self._metadata(cluster)
            if self.snapshot.raycluster_uid and self.snapshot.raycluster_uid != uid:
                raise ObserverError("RayCluster UID changed")
            self.snapshot.raycluster_uid = uid
        pods = self._list("pod", "--selector", f"ray.io/cluster={self.snapshot.raycluster_name}")
        self._capture_pods(pods)

    def _observe_rayjob(self, resource: dict) -> None:
        """Observe a direct root RayJob without a FleetJob parent."""
        uid, _ = self._metadata(resource)
        if self.snapshot.rayjob_uid and self.snapshot.rayjob_uid != uid:
            raise ObserverError("RayJob UID changed")
        self.snapshot.rayjob_name = self.name
        self.snapshot.rayjob_uid = uid
        ray_status = resource.get("status", {})
        cluster_name = ray_status.get("rayClusterName")
        if isinstance(cluster_name, str) and cluster_name:
            if self.snapshot.raycluster_name and self.snapshot.raycluster_name != cluster_name:
                raise ObserverError("RayCluster identity changed")
            self.snapshot.raycluster_name = cluster_name
        terminal = TERMINAL_RAY_STATUSES.get(ray_status.get("jobStatus"))
        if terminal:
            self.snapshot.terminal_status = terminal
        workload = self._find_workload(self.name)
        if workload is not None:
            name = workload.get("metadata", {}).get("name")
            workload_uid, _ = self._metadata(workload)
            if not isinstance(name, str) or not name:
                raise ObserverError("Workload identity is invalid")
            if self.snapshot.workload_uid and self.snapshot.workload_uid != workload_uid:
                raise ObserverError("Workload UID changed")
            self.snapshot.workload_name = name
            self.snapshot.workload_uid = workload_uid
        if not self.snapshot.raycluster_name:
            return
        cluster = self._get("raycluster", self.snapshot.raycluster_name)
        if cluster is not None:
            cluster_uid, _ = self._metadata(cluster)
            if self.snapshot.raycluster_uid and self.snapshot.raycluster_uid != cluster_uid:
                raise ObserverError("RayCluster UID changed")
            self.snapshot.raycluster_uid = cluster_uid
        pods = self._list("pod", "--selector", f"ray.io/cluster={self.snapshot.raycluster_name}")
        self._capture_pods(pods)

    def observe(self, resource: dict) -> None:
        if self._is_prod8_terminal_probe():
            # Do not adopt the name (and therefore do not make it deletable)
            # until the live object proves both its immutable manifest and the
            # caller-supplied UID captured at creation.
            self._bind_prod8_manifest(resource)
            if (
                self.snapshot.prod8_manifest_error
                or self.snapshot.prod8_normalized_manifest is None
            ):
                raise ObserverError(
                    "prod8 terminal probe manifest binding failed",
                    code="prod8_manifest_binding_failed",
                )
        self._bind(resource)
        if self.kind == "job":
            self._observe_job(resource)
        elif self.kind == "fleetjob":
            self._observe_fleetjob(resource)
        else:
            self._observe_rayjob(resource)
        # A Pod that differs from the rendered template may be an admission
        # mutation or another provenance failure.  Even though the root Job
        # UID was creation-bound, do not turn a failed live-Pod binding into a
        # destructive cleanup action: the terminal probe is evidence-only and
        # must leave an unproven workload for explicit operator review.
        if self._is_prod8_terminal_probe() and self.snapshot.prod8_binding_error:
            raise ObserverError(
                "prod8 terminal probe Pod binding failed",
                code="prod8_pod_binding_failed",
            )
        if self._is_prod8_terminal_probe() and self.snapshot.prod8_receipt_error:
            raise ObserverError(
                "prod8 terminal probe receipt was rejected",
                code="prod8_receipt_rejected",
            )
        if self.snapshot.peak_gpus > self.expected_gpus:
            raise CleanupAuthorizedError(
                "workload exceeded its plan-bound GPU count",
                code="gpu_contract_exceeded",
            )

    def _record_observation_failure(self, error: ObserverError, consecutive_failures: int) -> None:
        self.observation_failures += 1
        self.max_consecutive_observation_failures = max(
            self.max_consecutive_observation_failures,
            consecutive_failures,
        )
        self.last_observation_error_code = error.code

    def _same_target(self) -> dict | None:
        resource = self._target()
        if resource is None:
            return None
        uid, _ = self._metadata(resource)
        if uid != self.snapshot.uid:
            raise ObserverError(
                "refusing to delete a different workload UID",
                code=(
                    "prod8_creation_uid_mismatch"
                    if self._is_prod8_terminal_probe()
                    else "observer_error"
                ),
            )
        return resource

    def _uid_delete_path(self) -> str:
        """Return the fixed Kubernetes REST route for an observed root kind."""
        routes = {
            "job": ("batch/v1", "jobs"),
            "fleetjob": ("fleet.ai/v1alpha1", "fleetjobs"),
            "rayjob": ("ray.io/v1", "rayjobs"),
        }
        version, plural = routes[self.kind]
        return (
            f"/apis/{version}/namespaces/{quote(self.namespace, safe='')}/"
            f"{plural}/{quote(self.name, safe='')}"
        )

    def _delete_uid_conditioned(self) -> None:
        """Issue the sole destructive request as API DELETE with a UID CAS."""
        body = json.dumps(
            {
                "apiVersion": "v1",
                "kind": "DeleteOptions",
                "propagationPolicy": "Foreground",
                "preconditions": {"uid": self.snapshot.uid},
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        # kubectl's raw-delete stdin path sends this exact body to the API.
        # Unlike `kubectl delete <name>`, a recreated same-name object cannot
        # satisfy DeleteOptions.preconditions.uid.
        self._kubectl(
            "delete",
            "--raw",
            self._uid_delete_path(),
            "-f",
            "-",
            input_text=body,
        )

    def delete(self) -> None:
        if not self.snapshot.uid or self.deletion_requested_at:
            return
        target = self._same_target()
        if target is None:
            self.deletion_requested_at = _stamp(_now())
            return
        if self._is_prod8_terminal_probe():
            # Re-read the exact root and its controller-selected Pod in the
            # destructive path.  A mutable label could otherwise hide a live
            # Job-owned Pod from an earlier observation, allowing a deadline
            # or terminal transition to delete without current Pod provenance.
            self.observe(target)
            if self.snapshot.prod8_last_listed_pod_count != 1:
                self.snapshot.prod8_binding_error = "final_pod_not_observed"
                raise ObserverError(
                    "prod8 terminal probe has no uniquely bound live Pod",
                    code="prod8_pod_binding_failed",
                )
        self.deletion_requested_at = _stamp(_now())
        self._delete_uid_conditioned()

    def _present(self, resource: str, name: str) -> bool:
        return bool(name and self._get(resource, name) is not None)

    def wait_for_release(self) -> dict:
        deadline = time.monotonic() + self.release_seconds
        while True:
            # A same-name replacement is neither proof that the bound target
            # remains nor proof that it was released.  Never turn it into a
            # successful release observation.
            target_present = self._same_target() is not None
            live_pods = []
            for name in self.snapshot.pod_names:
                pod = self._get("pod", name)
                if pod is not None:
                    live_pods.append(pod)
            if live_pods:
                self._capture_pods(live_pods)
            pod_present = bool(live_pods)
            rayjob_present = self._present("rayjob", self.snapshot.rayjob_name)
            workload_present = self._present("workload", self.snapshot.workload_name)
            cluster_present = self._present("raycluster", self.snapshot.raycluster_name)
            if not any(
                (
                    target_present,
                    pod_present,
                    rayjob_present,
                    workload_present,
                    cluster_present,
                )
            ):
                # Re-read the root immediately before certifying release.  A
                # recreated same-name target raises from _same_target rather
                # than being mistaken for the just-deleted UID.
                if self._same_target() is not None:
                    continue
                return {
                    "target_present": False,
                    "pods_present": False,
                    "rayjob_present": False,
                    "workload_present": False,
                    "raycluster_present": False,
                    "active_gpus": 0,
                    "release_observed_at": _stamp(_now()),
                }
            if time.monotonic() >= deadline:
                raise ObserverError("workload cleanup exceeded its release allowance")
            time.sleep(self.poll_seconds)

    def result(self, release: dict, *, observer_error_class: str = "") -> dict:
        prod8_bound = not self._is_prod8_terminal_probe() or (
            self.snapshot.prod8_normalized_manifest is not None
            and self.snapshot.prod8_receipt_bound
            and len(self.snapshot.prod8_container_observations) == 1
            and not self.snapshot.prod8_manifest_error
            and not self.snapshot.prod8_binding_error
            and not self.snapshot.prod8_receipt_error
        )
        receipt = self.snapshot.receipt if prod8_bound else None
        accepted = (
            not observer_error_class
            and self.snapshot.terminal_status == "Succeeded"
            and _receipt_execution_accepted(receipt)
            and self.snapshot.peak_gpus == self.expected_gpus
            and prod8_bound
            and (
                not isinstance(receipt, dict)
                or receipt.get("schema") != prod8.RECEIPT_SCHEMA
                or self._is_prod8_terminal_probe()
            )
        )
        status = "released" if accepted else "released_without_accepted_execution"
        return _seal(
            {
                "schema": self.result_schema,
                "status": status,
                "context": self.context,
                "namespace": self.namespace,
                "kind": self.kind,
                "name": self.name,
                "plan_sha256": self.plan_sha256,
                "manifest_sha256": self.manifest_sha256,
                "expected_gpus": self.expected_gpus,
                "maximum_seconds": self.maximum_seconds,
                "armed_at": self.armed_at,
                "uid": self.snapshot.uid,
                "created_at": self.snapshot.created_at,
                "terminal_status": self.snapshot.terminal_status or "Deleted",
                "job_id": self.snapshot.job_id,
                "rayjob_name": self.snapshot.rayjob_name,
                "rayjob_uid": self.snapshot.rayjob_uid,
                "workload_name": self.snapshot.workload_name,
                "workload_uid": self.snapshot.workload_uid,
                "raycluster_name": self.snapshot.raycluster_name,
                "raycluster_uid": self.snapshot.raycluster_uid,
                "pod_names": sorted(self.snapshot.pod_names),
                "pod_uids": sorted(self.snapshot.pod_uids),
                "image_ids": sorted(self.snapshot.image_ids),
                "exit_codes": sorted(self.snapshot.exit_codes),
                "termination_reasons": sorted(self.snapshot.termination_reasons),
                "restarts": self.snapshot.restarts,
                "peak_gpus": self.snapshot.peak_gpus,
                "receipt": receipt,
                "observer_error_class": observer_error_class,
                "observation_failures": self.observation_failures,
                "max_consecutive_observation_failures": (self.max_consecutive_observation_failures),
                "last_observation_error_code": self.last_observation_error_code,
                "deletion_reason": self.deletion_reason,
                "deletion_requested_at": self.deletion_requested_at,
                **(
                    {
                        "prod8_live_binding": {
                            "job_uid": self.snapshot.uid,
                            "normalized_manifest": self.snapshot.prod8_normalized_manifest,
                            "normalized_manifest_sha256": (
                                digest(self.snapshot.prod8_normalized_manifest)
                                if self.snapshot.prod8_normalized_manifest is not None
                                else ""
                            ),
                            "containers": [
                                self.snapshot.prod8_container_observations[uid]
                                for uid in sorted(self.snapshot.prod8_container_observations)
                            ],
                            "receipt_container_bound": self.snapshot.prod8_receipt_bound,
                            "binding_error": (
                                self.snapshot.prod8_manifest_error
                                or self.snapshot.prod8_binding_error
                                or self.snapshot.prod8_receipt_error
                            ),
                        }
                    }
                    if self._is_prod8_terminal_probe()
                    else {}
                ),
                **(
                    {"recovered_existing_target_uid": self.expected_uid}
                    if self.profile == "production-recovery" and self.expected_uid
                    else (
                        {"creation_bound_uid": self.expected_uid}
                        if self._is_prod8_terminal_probe()
                        else {}
                    )
                ),
                **release,
            }
        )

    def run(self) -> dict:
        self.arm()
        creation_allowance = 300 if self.profile == "production-direct" else 120
        creation_deadline = time.monotonic() + min(creation_allowance, self.maximum_seconds)
        observer_error: BaseException | None = None
        delete_failure: BaseException | None = None
        deletion_authorized = False
        try:
            consecutive_observation_failures = 0
            while not self.snapshot.uid:
                if self._creator_uid() is None and self.requires_creator_binding:
                    if time.monotonic() >= creation_deadline:
                        raise ObserverError("cleanup target was not creation-bound after arming")
                    time.sleep(self.poll_seconds)
                    continue
                try:
                    resource = self._target()
                    if resource is not None:
                        self.observe(resource)
                        consecutive_observation_failures = 0
                        break
                except ObserverError as exc:
                    consecutive_observation_failures += 1
                    self._record_observation_failure(exc, consecutive_observation_failures)
                    if self._is_fatal_prod8_binding_error(exc):
                        raise
                    # _bind runs before child discovery.  If the exact target
                    # UID was bound and a later read failed, continue through
                    # the normal observation loop instead of treating that
                    # transient child read as a failed workload.
                    if self.snapshot.uid:
                        break
                if time.monotonic() >= creation_deadline:
                    raise ObserverError("cleanup target was not created after arming")
                time.sleep(self.poll_seconds)
            created = _parse_stamp(self.snapshot.created_at)
            terminal_receipt_deadline: float | None = None
            while True:
                elapsed = (_now() - created).total_seconds()
                if elapsed >= self.maximum_seconds - DELETE_REQUEST_MARGIN_SECONDS:
                    deletion_authorized = True
                    self.deletion_reason = "plan_deadline"
                    break
                try:
                    resource = self._same_target()
                    if resource is None:
                        self.snapshot.terminal_status = self.snapshot.terminal_status or "Deleted"
                        self.deletion_reason = "target_absent"
                        break
                    self.observe(resource)
                    consecutive_observation_failures = 0
                except CleanupAuthorizedError as exc:
                    consecutive_observation_failures += 1
                    self._record_observation_failure(exc, consecutive_observation_failures)
                    observer_error = exc
                    deletion_authorized = True
                    self.deletion_reason = exc.code
                    break
                except ObserverError as exc:
                    consecutive_observation_failures += 1
                    self._record_observation_failure(exc, consecutive_observation_failures)
                    if self._is_fatal_prod8_binding_error(exc):
                        observer_error = exc
                        break
                    # An observation failure is not evidence that a healthy
                    # workload is terminal, broken, or stalled.  In
                    # particular, repeated Kubernetes read timeouts must not
                    # turn the cleanup observer into a job killer.  Keep
                    # watching until a positive terminal/resource signal or
                    # the plan-bound deadline authorizes exact-UID cleanup.
                    if consecutive_observation_failures >= MAX_CONSECUTIVE_OBSERVATION_FAILURES:
                        time.sleep(min(max(self.poll_seconds, 1.0), 30.0))
                    continue
                if self.snapshot.terminal_status:
                    # A RayJob can report terminal before its long-running Ray
                    # container terminates.  Kubernetes exposes the declared
                    # termination-message file only after that container
                    # terminates, so deleting immediately races and loses the
                    # public sealed receipt.  Give shutdownAfterJobFinishes a
                    # short, fixed grace period; the normal deadline and
                    # unconditional UID-bound delete remain authoritative.
                    now = time.monotonic()
                    if self.kind in {"fleetjob", "rayjob"} and self.snapshot.receipt is None:
                        if terminal_receipt_deadline is None:
                            remaining = max(
                                0.0,
                                self.maximum_seconds - DELETE_REQUEST_MARGIN_SECONDS - elapsed,
                            )
                            terminal_receipt_deadline = now + min(
                                TERMINAL_RECEIPT_GRACE_SECONDS,
                                remaining,
                            )
                        if now < terminal_receipt_deadline:
                            time.sleep(min(self.poll_seconds, terminal_receipt_deadline - now))
                            continue
                    deletion_authorized = True
                    self.deletion_reason = "terminal_status"
                    break
                time.sleep(
                    min(
                        self.poll_seconds,
                        max(
                            0.1,
                            self.maximum_seconds - DELETE_REQUEST_MARGIN_SECONDS - elapsed,
                        ),
                    )
                )
        except BaseException as exc:
            observer_error = exc
        finally:
            if self.snapshot.uid and deletion_authorized:
                try:
                    self.delete()
                except BaseException as exc:
                    delete_failure = exc
                    observer_error = observer_error or exc
        # A DELETE conflict/time-out is intentionally not followed by name
        # polling: the object at that name may now be a different workload.
        if delete_failure is not None:
            raise delete_failure
        if isinstance(observer_error, ObserverError) and self._is_fatal_prod8_binding_error(
            observer_error
        ):
            raise observer_error
        if not self.snapshot.uid:
            if observer_error is not None:
                raise observer_error
            raise ObserverError("cleanup target identity was never bound")
        release = self.wait_for_release()
        value = self.result(
            release,
            observer_error_class=(
                type(observer_error).__name__ if observer_error is not None else ""
            ),
        )
        _write_create_once(self.result_path, value)
        return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--kind", choices=("job", "fleetjob", "rayjob"), required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--maximum-seconds", type=int, required=True)
    parser.add_argument("--expected-gpus", type=int, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--armed", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--expected-uid", default="")
    parser.add_argument(
        "--profile",
        choices=(
            "development",
            "production-direct",
            "production-cpu",
            "production-reload",
            "production-recovery",
        ),
        default="development",
    )
    args = parser.parse_args()
    try:
        observer = Observer(
            context=args.context,
            namespace=args.namespace,
            kind=args.kind,
            name=args.name,
            maximum_seconds=args.maximum_seconds,
            expected_gpus=args.expected_gpus,
            plan_sha256=args.plan_sha256,
            manifest_sha256=args.manifest_sha256,
            armed_path=args.armed,
            result_path=args.result,
            profile=args.profile,
            expected_uid=args.expected_uid,
        )
        result = observer.run()
        print(json.dumps({"status": result["status"], "sha256": result["sha256"]}))
        if (
            args.profile
            in {
                "development",
                "production-cpu",
                "production-reload",
            }
            and result["status"] != "released"
        ):
            raise SystemExit(1)
    except BaseException as exc:
        print(json.dumps({"status": "failed", "error_class": type(exc).__name__}))
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
