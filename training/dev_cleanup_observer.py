"""Arm a local, UID-bound cleanup observer for one exact workload.

The observer starts before the workload is created.  It records only Kubernetes
identity, lifecycle, resource and validated receipt metadata.  It never reads
container logs.  Once the exact workload becomes terminal, or reaches its
plan-bound deadline, the observer deletes that UID-bound object and waits for
all discovered children to disappear.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
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
RECOVERY_ARMED_SCHEMA = "cyber_direct_cleanup_recovery_observer_armed_v1"
RECOVERY_RESULT_SCHEMA = "cyber_direct_cleanup_recovery_observer_result_v1"
TERMINAL_RAY_STATUSES = {"SUCCEEDED": "Succeeded", "FAILED": "Failed"}
KUBECTL_ATTEMPTS = 3
KUBECTL_TIMEOUT_SECONDS = 20
MAX_CONSECUTIVE_OBSERVATION_FAILURES = 5
DELETE_REQUEST_MARGIN_SECONDS = 60
TERMINAL_RECEIPT_GRACE_SECONDS = 30


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
    if value.get("schema") != prod8.RECEIPT_SCHEMA:
        return value.get("status") in {
            "passed",
            "published",
            "setup_and_internal_cleanup_passed",
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
        if expected_gpus not in {0, 8} or (kind == "job") != (expected_gpus == 0):
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
        if args.profile in {"development", "production-cpu"} and result["status"] != "released":
            raise SystemExit(1)
    except BaseException as exc:
        print(json.dumps({"status": "failed", "error_class": type(exc).__name__}))
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
