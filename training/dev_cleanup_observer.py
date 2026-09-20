"""Arm a local, UID-bound cleanup observer for one development workload.

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
from uuid import UUID

from cyber_post_train.jobs import digest, quantity

DEV_CONTEXT = "nebius-mk8s-fleetai-training-dev-e04p03enwk5c0va9tb"
NAMESPACE = "fleet-train-jobs"
ARMED_SCHEMA = "cyber_dev_cleanup_observer_armed_v1"
RESULT_SCHEMA = "cyber_dev_cleanup_observer_result_v1"
TERMINAL_RAY_STATUSES = {"SUCCEEDED": "Succeeded", "FAILED": "Failed"}


class ObserverError(RuntimeError):
    """A sanitized cleanup-observer failure."""


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
    if not isinstance(message, str) or not message or len(message) > 16384:
        return None
    try:
        value = json.loads(message)
    except (TypeError, ValueError):
        return None
    if not isinstance(value, dict):
        return None
    body = {key: item for key, item in value.items() if key != "sha256"}
    schemas = {
        "job": {
            "cyber_skyrl_topology_probe_cpu_preflight_v1",
            "cyber_skyrl_topology_probe_cpu_preflight_rejection_v1",
            "cyber_skyrl_model_artifact_stage_receipt_v1",
            "cyber_skyrl_model_artifact_stage_rejection_v1",
        },
        "fleetjob": {
            "cyber_skyrl_topology_probe_receipt_v1",
            "cyber_skyrl_topology_probe_failure_v1",
        },
    }
    if (
        value.get("schema") not in schemas[kind]
        or value.get("sha256") != "sha256:" + digest(body)
    ):
        return None
    return value


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
        run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    ) -> None:
        if context != DEV_CONTEXT or namespace != NAMESPACE:
            raise ObserverError("cleanup observer is bound to the development cluster")
        if kind not in {"job", "fleetjob"}:
            raise ObserverError("cleanup observer kind is invalid")
        if not name or maximum_seconds < 1 or maximum_seconds > 1800:
            raise ObserverError("cleanup observer deadline is invalid")
        if expected_gpus not in {0, 8} or (kind == "job") != (expected_gpus == 0):
            raise ObserverError("cleanup observer GPU contract is invalid")
        for value in (plan_sha256, manifest_sha256):
            if len(value.removeprefix("sha256:")) != 64:
                raise ObserverError("cleanup observer digest binding is invalid")
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
        self._run = run
        self.snapshot = Snapshot()
        self.armed_at = ""
        self.deletion_requested_at = ""

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
        for attempt in range(3):
            try:
                result = self._run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=45,
                )
                break
            except subprocess.TimeoutExpired:
                if attempt == 2:
                    raise ObserverError(
                        "development-cluster observation timed out"
                    ) from None
                time.sleep(min(self.poll_seconds, 1.0))
        if result is None:  # pragma: no cover - the loop either returns or raises
            raise ObserverError("development-cluster observation failed")
        if result.returncode:
            raise ObserverError("development-cluster observation failed")
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

    def _target(self) -> dict | None:
        return self._get("job" if self.kind == "job" else "fleetjob", self.name)

    def arm(self) -> dict:
        if self._target() is not None:
            raise ObserverError("cleanup target already exists")
        self.armed_at = _stamp(_now())
        evidence = _seal(
            {
                "schema": ARMED_SCHEMA,
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
            }
        )
        _write_create_once(self.armed_path, evidence)
        return evidence

    @staticmethod
    def _metadata(resource: dict) -> tuple[str, str]:
        metadata = resource.get("metadata", {})
        uid, created = metadata.get("uid"), metadata.get("creationTimestamp")
        try:
            UUID(uid)
            _parse_stamp(created)
        except (TypeError, ValueError) as exc:
            raise ObserverError("workload identity metadata is invalid") from exc
        return uid, created

    def _bind(self, resource: dict) -> None:
        uid, created = self._metadata(resource)
        if self.snapshot.uid and uid != self.snapshot.uid:
            raise ObserverError("cleanup target name was reused with another UID")
        if _parse_stamp(created) < _parse_stamp(self.armed_at):
            raise ObserverError("cleanup target predates the armed observer")
        self.snapshot.uid = uid
        self.snapshot.created_at = created

    def _capture_pods(self, pods: list[dict]) -> None:
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
                    receipt = _validated_receipt(
                        terminated.get("message"), kind=self.kind
                    )
                    if receipt is not None:
                        self.snapshot.receipt = receipt
        self.snapshot.peak_gpus = max(self.snapshot.peak_gpus, current_gpus)

    def _observe_job(self, resource: dict) -> None:
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
            if (
                self.snapshot.raycluster_name
                and self.snapshot.raycluster_name != cluster_name
            ):
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
        pods = self._list(
            "pod", "--selector", f"ray.io/cluster={self.snapshot.raycluster_name}"
        )
        self._capture_pods(pods)

    def observe(self, resource: dict) -> None:
        self._bind(resource)
        if self.kind == "job":
            self._observe_job(resource)
        else:
            self._observe_fleetjob(resource)
        if self.snapshot.peak_gpus > self.expected_gpus:
            raise ObserverError("workload exceeded its plan-bound GPU count")

    def _same_target(self) -> dict | None:
        resource = self._target()
        if resource is None:
            return None
        uid, _ = self._metadata(resource)
        if uid != self.snapshot.uid:
            raise ObserverError("refusing to delete a different workload UID")
        return resource

    def delete(self) -> None:
        if not self.snapshot.uid or self.deletion_requested_at:
            return
        if self._same_target() is None:
            self.deletion_requested_at = _stamp(_now())
            return
        self.deletion_requested_at = _stamp(_now())
        self._kubectl(
            "delete",
            "job" if self.kind == "job" else "fleetjob",
            self.name,
            "--cascade=foreground",
            "--wait=false",
            "--output=name",
        )

    def _present(self, resource: str, name: str) -> bool:
        return bool(name and self._get(resource, name) is not None)

    def wait_for_release(self) -> dict:
        deadline = time.monotonic() + self.release_seconds
        while True:
            target_present = self._present(
                "job" if self.kind == "job" else "fleetjob", self.name
            )
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
        receipt = self.snapshot.receipt
        accepted = (
            not observer_error_class
            and self.snapshot.terminal_status == "Succeeded"
            and receipt is not None
            and receipt.get("status")
            in {"passed", "published", "setup_and_internal_cleanup_passed"}
            and self.snapshot.peak_gpus == self.expected_gpus
        )
        status = "released" if accepted else "released_without_accepted_execution"
        return _seal(
            {
                "schema": RESULT_SCHEMA,
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
                "deletion_requested_at": self.deletion_requested_at,
                **release,
            }
        )

    def run(self) -> dict:
        self.arm()
        creation_deadline = time.monotonic() + min(120, self.maximum_seconds)
        observer_error: BaseException | None = None
        try:
            while not self.snapshot.uid:
                resource = self._target()
                if resource is not None:
                    self.observe(resource)
                    break
                if time.monotonic() >= creation_deadline:
                    raise ObserverError("cleanup target was not created after arming")
                time.sleep(self.poll_seconds)
            created = _parse_stamp(self.snapshot.created_at)
            while True:
                resource = self._same_target()
                if resource is None:
                    self.snapshot.terminal_status = self.snapshot.terminal_status or "Deleted"
                    break
                self.observe(resource)
                elapsed = (_now() - created).total_seconds()
                if self.snapshot.terminal_status or elapsed >= self.maximum_seconds:
                    break
                time.sleep(min(self.poll_seconds, max(0.1, self.maximum_seconds - elapsed)))
        except BaseException as exc:
            observer_error = exc
        finally:
            if self.snapshot.uid:
                try:
                    self.delete()
                except BaseException as exc:
                    observer_error = observer_error or exc
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
    parser.add_argument("--kind", choices=("job", "fleetjob"), required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--maximum-seconds", type=int, required=True)
    parser.add_argument("--expected-gpus", type=int, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--armed", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
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
        )
        result = observer.run()
        print(json.dumps({"status": result["status"], "sha256": result["sha256"]}))
        if result["status"] != "released":
            raise SystemExit(1)
    except BaseException as exc:
        print(json.dumps({"status": "failed", "error_class": type(exc).__name__}))
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
