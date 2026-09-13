"""Durable, value-free Kubernetes event evidence for TTL-zero Miles runs.

The Jobs API deliberately configures RayJobs for immediate cleanup.  A poll
after completion can therefore miss the RayJob, Workload, RayCluster, and Pod.
This module projects read-only watch events into small create-once receipts
*while the objects exist*, then compiles the same terminal controller evidence
after Kubernetes has removed them.  It never stores logs, environment values,
task content, metrics, or credentials.
"""

from __future__ import annotations

import math
import numbers
import re
import uuid
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from cyber_post_train.jobs import digest

from .miles_conversion import _write
from .rl_runtime import sealed

START_SCHEMA = "cyber_miles_uid_event_capture_start_v1"
EVENT_SCHEMA = "cyber_miles_uid_event_v1"
CONTROLLER_SCHEMA = "cyber_miles_controller_terminal_observation_v1"
RELEASE_SCHEMA = "cyber_miles_external_release_v1"
NAMESPACE = "fleet-train-jobs"
WORLD_SIZE = 8

_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}")


def _time(value: object, label: str) -> float:
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        result = float(value)
    elif isinstance(value, str):
        try:
            result = datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except ValueError as error:
            raise ValueError(f"{label} is not an RFC3339 timestamp") from error
    else:
        raise ValueError(f"{label} is not a timestamp")
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{label} is not a positive finite timestamp")
    return result


def _uuid(value: object, label: str) -> str:
    try:
        result = str(uuid.UUID(str(value)))
    except ValueError as error:
        raise ValueError(f"{label} is not a UUID") from error
    if result == "00000000-0000-0000-0000-000000000000":
        raise ValueError(f"{label} is a zero UUID")
    return result


def _read(path: Path, schema: str) -> tuple[dict[str, Any], str]:
    from .miles_acceptance import _json_snapshot

    value, file_sha256 = _json_snapshot(path)
    if not isinstance(value, dict):
        raise ValueError("event evidence must be a JSON object")
    sealed(value, schema)
    return value, file_sha256


def _owner(metadata: Mapping[str, Any]) -> dict[str, str] | None:
    owners = metadata.get("ownerReferences")
    if owners in (None, []):
        return None
    if not isinstance(owners, list):
        raise ValueError("Kubernetes ownerReferences is not an array")
    controllers = [
        row for row in owners if isinstance(row, Mapping) and row.get("controller") is True
    ]
    if len(controllers) != 1:
        raise ValueError("Kubernetes object has no unique controller owner")
    row = controllers[0]
    if not all(isinstance(row.get(key), str) and row[key] for key in ("kind", "name", "uid")):
        raise ValueError("Kubernetes controller owner identity is incomplete")
    return {"kind": row["kind"], "name": row["name"], "uid": _uuid(row["uid"], "owner UID")}


def _condition_time(status: Mapping[str, Any], condition_type: str) -> object | None:
    values = [
        row.get("lastTransitionTime")
        for row in status.get("conditions") or []
        if isinstance(row, Mapping)
        and row.get("type") == condition_type
        and str(row.get("status")).lower() == "true"
    ]
    if len(values) > 1:
        raise ValueError("Kubernetes condition is ambiguous")
    return values[0] if values else None


def start_capture(
    plan: dict[str, Any],
    submission: dict[str, Any],
    *,
    namespace_uid: str,
    started_at: object,
    directory: Path,
) -> dict[str, Any]:
    """Create the capture root before admission; never overwrite a prior watch."""

    from .miles_acceptance import validate_submission_binding

    submitted = validate_submission_binding(submission, plan, check_files=False)
    run_id = _uuid(submitted["api"]["run_id"], "API run ID")
    namespace = _uuid(namespace_uid, "namespace UID")
    directory.mkdir(parents=True, mode=0o700, exist_ok=False)
    return _write(
        directory / "STARTED.json",
        {
            "schema": START_SCHEMA,
            "status": "watching",
            "source_plan_sha256": "sha256:" + digest(plan),
            "source_request_sha256": submitted["request_sha256"],
            "api_base_url": submitted["api"]["base_url"],
            "api_run_id": run_id,
            "api_run_name": submitted["api"]["run_name"],
            "cluster": "dev",
            "namespace": NAMESPACE,
            "namespace_uid": namespace,
            "started_at": started_at,
            "private_logs_included": False,
            "metric_values_included": False,
            "task_content_included": False,
        },
    )


def _pod_projection(spec: Mapping[str, Any], status: Mapping[str, Any]) -> dict[str, Any]:
    containers = spec.get("containers")
    statuses = status.get("containerStatuses")
    if not isinstance(containers, list) or len(containers) != 1 or not isinstance(statuses, list):
        raise ValueError("Miles Pod does not expose one unambiguous main container")
    container = containers[0]
    matches = [
        row
        for row in statuses
        if isinstance(row, Mapping) and row.get("name") == container.get("name")
    ]
    if len(matches) != 1:
        raise ValueError("Miles Pod main container status is absent or ambiguous")
    observed = matches[0]
    terminated = (observed.get("state") or {}).get("terminated")
    terminated = terminated if isinstance(terminated, Mapping) else {}
    resources = container.get("resources")
    if not isinstance(resources, Mapping):
        raise ValueError("Miles Pod resources are absent")
    limits = resources.get("limits")
    if not isinstance(limits, Mapping):
        raise ValueError("Miles Pod resource limits are absent")
    gpu_value = limits.get("nvidia.com/gpu")
    try:
        gpus = int(gpu_value)
    except (TypeError, ValueError) as error:
        raise ValueError("Miles Pod GPU allocation is invalid") from error
    return {
        "phase": status.get("phase"),
        "runtime_image_id": observed.get("imageID"),
        "container_restarts": observed.get("restartCount"),
        "gpus": gpus,
        "exit_code": terminated.get("exitCode"),
        "termination_reason": terminated.get("reason"),
        "terminated_at": terminated.get("finishedAt"),
    }


def _project_event(start: dict[str, Any], raw: Mapping[str, Any], observed_at: object) -> dict:
    event_type = raw.get("type")
    obj = raw.get("object")
    if event_type not in {"ADDED", "MODIFIED", "DELETED"} or not isinstance(obj, Mapping):
        raise ValueError("Kubernetes watch event is invalid")
    kind = obj.get("kind")
    if kind not in {"RayJob", "Workload", "RayCluster", "Pod"}:
        raise ValueError("Kubernetes watch event is outside the Miles lifecycle")
    metadata = obj.get("metadata")
    spec = obj.get("spec") or {}
    status = obj.get("status") or {}
    if not all(isinstance(item, Mapping) for item in (metadata, spec, status)):
        raise ValueError("Kubernetes watch object is incomplete")
    if metadata.get("namespace") != start["namespace"]:
        raise ValueError("Kubernetes watch object escaped the Fleet namespace")
    created = metadata.get("creationTimestamp")
    _time(created, "Kubernetes creation timestamp")
    projection: dict[str, Any] = {
        "schema": EVENT_SCHEMA,
        "source_plan_sha256": start["source_plan_sha256"],
        "source_request_sha256": start["source_request_sha256"],
        "api_run_id": start["api_run_id"],
        "api_run_name": start["api_run_name"],
        "event_type": event_type,
        "observed_at": observed_at,
        "kind": kind,
        "namespace": metadata["namespace"],
        "name": metadata.get("name"),
        "uid": _uuid(metadata.get("uid"), f"{kind} UID"),
        "resource_version": metadata.get("resourceVersion"),
        "creation_timestamp": created,
        "deletion_timestamp": metadata.get("deletionTimestamp"),
        "owner": _owner(metadata),
        "controller_status": None,
        "admitted_at": None,
        "raycluster_name": None,
        "priority_class": None,
        "shutdown_after_job_finishes": None,
        "ttl_seconds_after_finished": None,
        "pod": None,
        "private_logs_included": False,
        "metric_values_included": False,
        "task_content_included": False,
    }
    if not isinstance(projection["name"], str) or not projection["name"]:
        raise ValueError("Kubernetes watch object name is absent")
    if not isinstance(projection["resource_version"], str) or not projection["resource_version"]:
        raise ValueError("Kubernetes resource version is absent")
    if projection["deletion_timestamp"] is not None:
        _time(projection["deletion_timestamp"], "Kubernetes deletion timestamp")
    if kind == "RayJob":
        labels = metadata.get("labels") or {}
        head = ((spec.get("rayClusterSpec") or {}).get("headGroupSpec") or {}).get("template") or {}
        pod_spec = head.get("spec") or {}
        projection.update(
            controller_status=status.get("jobStatus"),
            raycluster_name=status.get("rayClusterName"),
            priority_class=pod_spec.get("priorityClassName"),
            shutdown_after_job_finishes=spec.get("shutdownAfterJobFinishes"),
            ttl_seconds_after_finished=spec.get("ttlSecondsAfterFinished"),
        )
        if (
            projection["name"] != start["api_run_name"]
            or labels.get("fleet.ai/run-id") != start["api_run_id"]
        ):
            raise ValueError("RayJob event identity differs from the exact API run")
    elif kind == "Workload":
        projection["admitted_at"] = _condition_time(status, "Admitted")
        if projection["admitted_at"] is not None:
            _time(projection["admitted_at"], "Workload admission timestamp")
    elif kind == "Pod":
        projection["pod"] = _pod_projection(spec, status)
    return projection


def record_event(
    directory: Path, raw: Mapping[str, Any], *, observed_at: object, sequence: int
) -> dict[str, Any]:
    """Project and atomically persist one read-only watch event."""

    if type(sequence) is not int or sequence < 0:
        raise ValueError("event sequence must be a nonnegative integer")
    start, _ = _read(directory / "STARTED.json", START_SCHEMA)
    _time(observed_at, "event observation")
    value = _project_event(start, raw, observed_at)
    value["sequence"] = sequence
    return _write(directory / f"EVENT-{sequence:06d}.json", value)


def _event_files(directory: Path) -> list[Path]:
    paths = sorted(directory.glob("EVENT-*.json"))
    if not paths or [path.name for path in paths] != [
        f"EVENT-{index:06d}.json" for index in range(len(paths))
    ]:
        raise ValueError("Miles event journal is absent or non-contiguous")
    return paths


def compile_controller(
    plan: dict[str, Any],
    submission: dict[str, Any],
    *,
    directory: Path,
    api_status: str,
    observed_at: object,
    output: Path,
) -> dict[str, Any]:
    """Compile terminal evidence from the durable journal after TTL cleanup."""

    from .miles_acceptance import validate_submission_binding

    submitted = validate_submission_binding(submission, plan, check_files=False)
    start, _ = _read(directory / "STARTED.json", START_SCHEMA)
    if (
        start["source_plan_sha256"].removeprefix("sha256:") != digest(plan)
        or start["source_request_sha256"] != submitted["request_sha256"]
        or start["api_run_id"] != submitted["api"]["run_id"]
        or start["api_run_name"] != submitted["api"]["run_name"]
        or api_status != "SUCCEEDED"
    ):
        raise ValueError("Miles event capture differs from the terminal API run")
    events = [_read(path, EVENT_SCHEMA)[0] for path in _event_files(directory)]
    if [event["sequence"] for event in events] != list(range(len(events))):
        raise ValueError("Miles event journal sequence changed")
    for event in events:
        if (
            event["source_plan_sha256"] != start["source_plan_sha256"]
            or event["source_request_sha256"] != start["source_request_sha256"]
            or event["api_run_id"] != start["api_run_id"]
            or event["api_run_name"] != start["api_run_name"]
            or event["private_logs_included"] is not False
            or event["metric_values_included"] is not False
            or event["task_content_included"] is not False
        ):
            raise ValueError("Miles event journal identity or minimization changed")
    by_kind = {
        kind: [row for row in events if row["kind"] == kind]
        for kind in ("RayJob", "Workload", "RayCluster", "Pod")
    }
    if any(not rows for rows in by_kind.values()):
        raise ValueError("Miles TTL-zero journal missed a lifecycle object")
    identities: dict[str, tuple[str, str]] = {}
    for kind, rows in by_kind.items():
        values = {(row["name"], row["uid"]) for row in rows}
        if len(values) != 1:
            raise ValueError(f"Miles event journal contains multiple {kind} identities")
        identities[kind] = next(iter(values))
    rayjob_name, rayjob_uid = identities["RayJob"]
    workload_name, workload_uid = identities["Workload"]
    raycluster_name, raycluster_uid = identities["RayCluster"]
    pod_name, pod_uid = identities["Pod"]
    if (
        rayjob_name != start["api_run_name"]
        or any(
            row["owner"] != {"kind": "RayJob", "name": rayjob_name, "uid": rayjob_uid}
            for kind in ("Workload", "RayCluster")
            for row in by_kind[kind]
        )
        or any(
            row["owner"] != {"kind": "RayCluster", "name": raycluster_name, "uid": raycluster_uid}
            for row in by_kind["Pod"]
        )
    ):
        raise ValueError("Miles event journal controller ownership changed")
    admissions = [row["admitted_at"] for row in by_kind["Workload"] if row["admitted_at"]]
    if not admissions:
        raise ValueError("Miles event journal missed Workload admission")
    admitted_at = min(_time(value, "Workload admission") for value in admissions)
    if _time(start["started_at"], "capture start") > admitted_at:
        raise ValueError("Miles event capture began after admission")
    rayjob_terminal = [row for row in by_kind["RayJob"] if row["controller_status"] == "SUCCEEDED"]
    pod_terminal = [
        row
        for row in by_kind["Pod"]
        if isinstance(row["pod"], dict)
        and row["pod"]["phase"] == "Succeeded"
        and row["pod"]["exit_code"] == 0
        and row["pod"]["termination_reason"] == "Completed"
    ]
    if not rayjob_terminal or not pod_terminal:
        raise ValueError("Miles event journal missed terminal success before TTL cleanup")
    terminal_rayjob = rayjob_terminal[-1]
    terminal_pod = pod_terminal[-1]
    pod = terminal_pod["pod"]
    if (
        terminal_rayjob["shutdown_after_job_finishes"] is not True
        or terminal_rayjob["ttl_seconds_after_finished"] != 0
        or terminal_rayjob["priority_class"] != "c1"
        or terminal_rayjob["raycluster_name"] != raycluster_name
        or pod["container_restarts"] != 0
        or pod["gpus"] != WORLD_SIZE
        or not isinstance(pod["runtime_image_id"], str)
        or not pod["runtime_image_id"].endswith(plan["execution"]["image"].rsplit("@", 1)[1])
    ):
        raise ValueError("Miles event journal terminal execution differs from the plan")
    _time(pod["terminated_at"], "Pod termination")
    terminal_observed = max(
        _time(observed_at, "API terminal observation"),
        _time(terminal_rayjob["observed_at"], "RayJob terminal observation"),
        _time(terminal_pod["observed_at"], "Pod terminal observation"),
    )
    return _write(
        output,
        {
            "schema": CONTROLLER_SCHEMA,
            "status": "succeeded",
            "source_plan_sha256": "sha256:" + digest(plan),
            "source_request_sha256": submitted["request_sha256"],
            "api": {
                "base_url": start["api_base_url"],
                "run_id": start["api_run_id"],
                "run_name": start["api_run_name"],
                "status": api_status,
            },
            "kubernetes": {
                "cluster": start["cluster"],
                "namespace": start["namespace"],
                "namespace_uid": start["namespace_uid"],
                "rayjob": {"name": rayjob_name, "uid": rayjob_uid, "status": "SUCCEEDED"},
                "workload": {
                    "name": workload_name,
                    "uid": workload_uid,
                    "owner_rayjob_uid": rayjob_uid,
                },
                "raycluster": {
                    "name": raycluster_name,
                    "uid": raycluster_uid,
                    "owner_rayjob_uid": rayjob_uid,
                },
                "pods": [
                    {
                        "name": pod_name,
                        "uid": pod_uid,
                        "owner_raycluster_uid": raycluster_uid,
                        "phase": pod["phase"],
                        "exit_code": pod["exit_code"],
                        "termination_reason": pod["termination_reason"],
                        "runtime_image_id": pod["runtime_image_id"],
                        "container_restarts": pod["container_restarts"],
                        "gpus": pod["gpus"],
                    }
                ],
            },
            "execution": {
                "requested_image": plan["execution"]["image"],
                "priority_class": "c1",
                "effective_priority": 10000,
                "automatic_requeue": False,
                "workers": 1,
                "gpus_per_worker": WORLD_SIZE,
                "total_gpus": WORLD_SIZE,
            },
            "observed_at": terminal_observed,
        },
    )


def compile_release(
    plan: dict[str, Any],
    submission: dict[str, Any],
    *,
    controller_path: Path,
    absence: Mapping[str, Any],
    observed_at: object,
    output: Path,
) -> dict[str, Any]:
    """Seal a separate post-terminal absence observation."""

    from .miles_acceptance import validate_controller_observation, validate_submission_binding

    submitted = validate_submission_binding(submission, plan, check_files=False)
    controller, controller_file_sha256 = _read(controller_path, CONTROLLER_SCHEMA)
    validate_controller_observation(controller, plan, submission)
    expected_absence = {
        "raycluster_present": False,
        "rayjob_present": False,
        "workload_present": False,
        "quota_reservation_present": False,
        "gpu_pods_present": False,
        "active_gpus": 0,
    }
    if dict(absence) != expected_absence:
        raise ValueError("Miles post-terminal Kubernetes absence is incomplete")
    when = _time(observed_at, "release observation")
    if when < _time(controller["observed_at"], "controller terminal observation"):
        raise ValueError("Miles release observation predates terminal evidence")
    api = controller["api"]
    kube = controller["kubernetes"]
    return _write(
        output,
        {
            "schema": RELEASE_SCHEMA,
            "status": "released",
            "source_plan_sha256": "sha256:" + digest(plan),
            "source_request_sha256": submitted["request_sha256"],
            "controller_observation_sha256": controller["sha256"],
            "controller_observation_file_sha256": "sha256:" + controller_file_sha256,
            "api_status": "SUCCEEDED",
            "controller_status": "SUCCEEDED",
            "identities": {
                "api_run_id": api["run_id"],
                "api_run_name": api["run_name"],
                "rayjob_uid": kube["rayjob"]["uid"],
                "workload_uid": kube["workload"]["uid"],
                "raycluster_uid": kube["raycluster"]["uid"],
                "pod_uids": [kube["pods"][0]["uid"]],
            },
            **expected_absence,
            "gpu_release_proven": True,
            "observed_at": when,
        },
    )
