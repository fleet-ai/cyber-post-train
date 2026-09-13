"""Durable, value-free Kubernetes event evidence for TTL-zero Miles runs.

Both direct batch Jobs and Jobs API RayJobs are configured for immediate
cleanup.  This module projects read-only watch events into small create-once
receipts while their exact owner chain exists, then compiles terminal and
release evidence after Kubernetes removes it.  It never stores logs,
environment values, task content, metrics, or credentials.
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
DEV_KUBE_CONTEXT = "nebius-mk8s-fleetai-training-dev-e04p03enwk5c0va9tb"
_HF_START_FIELDS = {
    "schema",
    "status",
    "source_plan_sha256",
    "source_request_sha256",
    "api_base_url",
    "api_run_id",
    "api_run_name",
    "cluster",
    "kube_context",
    "namespace",
    "namespace_uid",
    "started_at",
    "private_logs_included",
    "metric_values_included",
    "task_content_included",
    "sha256",
}
_EVENT_FIELDS = {
    "schema",
    "source_plan_sha256",
    "source_request_sha256",
    "api_run_id",
    "api_run_name",
    "event_type",
    "observed_at",
    "kind",
    "namespace",
    "name",
    "uid",
    "resource_version",
    "creation_timestamp",
    "deletion_timestamp",
    "owner",
    "controller_status",
    "admitted_at",
    "raycluster_name",
    "priority_class",
    "shutdown_after_job_finishes",
    "ttl_seconds_after_finished",
    "pod",
    "private_logs_included",
    "metric_values_included",
    "task_content_included",
    "sequence",
    "sha256",
}

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
    kube_context: str = DEV_KUBE_CONTEXT,
) -> dict[str, Any]:
    """Create the capture root before admission; never overwrite a prior watch."""

    submitted = _validate_submission(submission, plan, check_files=False)
    run_id = _uuid(submitted["api"]["run_id"], "API run ID")
    namespace = _uuid(namespace_uid, "namespace UID")
    if kube_context != DEV_KUBE_CONTEXT:
        raise ValueError("Miles event capture is not using the exact dev Kubernetes context")
    if (
        plan.get("schema") == "cyber_miles_hf_export_job_plan_v1"
        and namespace != "10394b76-e1d4-40b1-a8e2-7575e95df216"
    ):
        raise ValueError("Miles HF event capture namespace UID differs from exact dev")
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
            "kube_context": kube_context,
            "namespace": NAMESPACE,
            "namespace_uid": namespace,
            "started_at": started_at,
            "private_logs_included": False,
            "metric_values_included": False,
            "task_content_included": False,
        },
    )


def _validate_submission(
    submission: dict[str, Any], plan: dict[str, Any], *, check_files: bool
) -> dict[str, Any]:
    if plan.get("schema") == "cyber_miles_hf_export_job_plan_v1":
        from .miles_hf_export_job import validate_submission_binding
    else:
        from .miles_acceptance import validate_submission_binding

    return validate_submission_binding(submission, plan, check_files=check_files)


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
    gpu_value = limits.get("nvidia.com/gpu", 0)
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
    if kind not in {"Job", "RayJob", "Workload", "RayCluster", "Pod"}:
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
    if kind == "Job":
        complete = _condition_time(status, "Complete")
        failed = _condition_time(status, "Failed")
        if complete is not None and failed is not None:
            raise ValueError("Kubernetes Job has contradictory terminal conditions")
        pod_spec = (spec.get("template") or {}).get("spec") or {}
        terminal_status = (
            "SUCCEEDED" if complete is not None else "FAILED" if failed is not None else None
        )
        projection.update(
            controller_status=terminal_status,
            priority_class=pod_spec.get("priorityClassName"),
            ttl_seconds_after_finished=spec.get("ttlSecondsAfterFinished"),
        )
        if (
            projection["name"] != start["api_run_name"]
            or projection["uid"] != start["api_run_id"]
        ):
            raise ValueError("Job event identity differs from the exact Kubernetes create")
    elif kind == "RayJob":
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
        projection["priority_class"] = spec.get("priorityClassName")
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

    submitted = _validate_submission(submission, plan, check_files=False)
    start_path = directory / "STARTED.json"
    start, start_file_sha256 = _read(start_path, START_SCHEMA)
    hf_plan = plan.get("schema") == "cyber_miles_hf_export_job_plan_v1"
    direct_export = hf_plan and plan.get("stage") == "export"
    expected_gpus = submitted.get("request", {}).get(
        "gpus" if direct_export else "gpus_per_worker", WORLD_SIZE
    )
    if (
        start["source_plan_sha256"].removeprefix("sha256:") != digest(plan)
        or start["source_request_sha256"] != submitted["request_sha256"]
        or start["api_run_id"] != submitted["api"]["run_id"]
        or start["api_run_name"] != submitted["api"]["run_name"]
        or api_status != "SUCCEEDED"
    ):
        raise ValueError("Miles event capture differs from the terminal API run")
    event_rows = [(path, *_read(path, EVENT_SCHEMA)) for path in _event_files(directory)]
    events = [value for _, value, _ in event_rows]
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
    kinds = ("Job", "Workload", "Pod") if direct_export else (
        "RayJob",
        "Workload",
        "RayCluster",
        "Pod",
    )
    by_kind = {kind: [row for row in events if row["kind"] == kind] for kind in kinds}
    if any(not rows for rows in by_kind.values()):
        raise ValueError("Miles TTL-zero journal missed a lifecycle object")
    identities: dict[str, tuple[str, str]] = {}
    for kind, rows in by_kind.items():
        values = {(row["name"], row["uid"]) for row in rows}
        if len(values) != 1:
            raise ValueError(f"Miles event journal contains multiple {kind} identities")
        identities[kind] = next(iter(values))
    workload_name, workload_uid = identities["Workload"]
    pod_name, pod_uid = identities["Pod"]
    if direct_export:
        job_name, job_uid = identities["Job"]
        if (
            job_name != start["api_run_name"]
            or job_uid != start["api_run_id"]
            or any(
                row["owner"] != {"kind": "Job", "name": job_name, "uid": job_uid}
                for kind in ("Workload", "Pod")
                for row in by_kind[kind]
            )
        ):
            raise ValueError("Miles event journal controller ownership changed")
        controller_kind = "Job"
        expected_priority = "c2"
    else:
        rayjob_name, rayjob_uid = identities["RayJob"]
        raycluster_name, raycluster_uid = identities["RayCluster"]
        if (
            rayjob_name != start["api_run_name"]
            or any(
                row["owner"] != {"kind": "RayJob", "name": rayjob_name, "uid": rayjob_uid}
                for kind in ("Workload", "RayCluster")
                for row in by_kind[kind]
            )
            or any(
                row["owner"]
                != {"kind": "RayCluster", "name": raycluster_name, "uid": raycluster_uid}
                for row in by_kind["Pod"]
            )
        ):
            raise ValueError("Miles event journal controller ownership changed")
        controller_kind = "RayJob"
        expected_priority = "c1"
    admissions = [row["admitted_at"] for row in by_kind["Workload"] if row["admitted_at"]]
    if not admissions:
        raise ValueError("Miles event journal missed Workload admission")
    if direct_export and any(row["priority_class"] != "q2" for row in by_kind["Workload"]):
        raise ValueError("Miles direct export Workload queue priority changed")
    admitted_at = min(_time(value, "Workload admission") for value in admissions)
    if _time(start["started_at"], "capture start") > admitted_at:
        raise ValueError("Miles event capture began after admission")
    controller_terminal = [
        row for row in by_kind[controller_kind] if row["controller_status"] == "SUCCEEDED"
    ]
    pod_terminal = [
        row
        for row in by_kind["Pod"]
        if isinstance(row["pod"], dict)
        and row["pod"]["phase"] == "Succeeded"
        and row["pod"]["exit_code"] == 0
        and row["pod"]["termination_reason"] == "Completed"
    ]
    if not controller_terminal or not pod_terminal:
        raise ValueError("Miles event journal missed terminal success before TTL cleanup")
    terminal_controller = controller_terminal[-1]
    terminal_pod = pod_terminal[-1]
    pod = terminal_pod["pod"]
    if (
        terminal_controller["ttl_seconds_after_finished"] != 0
        or terminal_controller["priority_class"] != expected_priority
        or (
            not direct_export
            and (
                terminal_controller["shutdown_after_job_finishes"] is not True
                or terminal_controller["raycluster_name"] != raycluster_name
            )
        )
        or pod["container_restarts"] != 0
        or pod["gpus"] != expected_gpus
        or not isinstance(pod["runtime_image_id"], str)
        or not pod["runtime_image_id"].endswith(plan["execution"]["image"].rsplit("@", 1)[1])
    ):
        raise ValueError("Miles event journal terminal execution differs from the plan")
    _time(pod["terminated_at"], "Pod termination")
    terminal_observed = max(
        _time(observed_at, "API terminal observation"),
        _time(terminal_controller["observed_at"], f"{controller_kind} terminal observation"),
        _time(terminal_pod["observed_at"], "Pod terminal observation"),
    )
    if hf_plan:
        from .miles_hf_export_job import controller_from_event_journal

        return _write(
            output,
            controller_from_event_journal(
                plan=plan,
                submission=submission,
                start=start,
                identities=identities,
                pod=pod,
                event_journal={
                    "directory": str(directory.resolve()),
                    "start": {
                        "path": str(start_path.resolve()),
                        "file_sha256": start_file_sha256,
                        "receipt_sha256": start["sha256"].removeprefix("sha256:"),
                    },
                    "events": [
                        {
                            "path": str(path.resolve()),
                            "file_sha256": file_sha256,
                            "receipt_sha256": value["sha256"].removeprefix("sha256:"),
                        }
                        for path, value, file_sha256 in event_rows
                    ],
                },
                observed_at=terminal_observed,
            ),
        )
    assert not direct_export
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
                "gpus_per_worker": expected_gpus,
                "total_gpus": expected_gpus,
            },
            "observed_at": terminal_observed,
        },
    )


def validate_hf_event_journal(
    plan: dict[str, Any],
    submission: dict[str, Any],
    binding: object,
    controller: Mapping[str, Any],
) -> dict[str, Any]:
    """Reopen the exact TTL-zero watch journal supporting an HF controller."""
    from .miles_hf_export_job import DEV_NAMESPACE_UID

    submitted = _validate_submission(submission, plan, check_files=False)
    if (
        not isinstance(binding, Mapping)
        or set(binding) != {"directory", "start", "events"}
        or not isinstance(binding.get("start"), Mapping)
        or not isinstance(binding.get("events"), list)
        or not binding["events"]
    ):
        raise ValueError("Miles HF controller event-journal binding is malformed")
    directory = Path(str(binding["directory"]))
    if not directory.is_absolute() or directory.is_symlink() or not directory.is_dir():
        raise ValueError("Miles HF controller event-journal root is indirect or absent")

    def reopen(reference: Mapping[str, Any], expected: Path, schema: str) -> dict[str, Any]:
        if (
            set(reference) != {"path", "file_sha256", "receipt_sha256"}
            or Path(str(reference.get("path"))) != expected
            or re.fullmatch(r"[a-f0-9]{64}", str(reference.get("file_sha256", ""))) is None
            or re.fullmatch(r"[a-f0-9]{64}", str(reference.get("receipt_sha256", ""))) is None
        ):
            raise ValueError("Miles HF controller event-journal reference changed")
        value, file_sha256 = _read(expected, schema)
        if (
            file_sha256 != reference["file_sha256"]
            or value["sha256"].removeprefix("sha256:") != reference["receipt_sha256"]
        ):
            raise ValueError("Miles HF controller event-journal file changed")
        return value

    start = reopen(binding["start"], directory / "STARTED.json", START_SCHEMA)
    expected_paths = _event_files(directory)
    if len(binding["events"]) != len(expected_paths):
        raise ValueError("Miles HF controller event-journal length changed")
    events = [
        reopen(reference, path, EVENT_SCHEMA)
        for reference, path in zip(binding["events"], expected_paths, strict=True)
        if isinstance(reference, Mapping)
    ]
    if len(events) != len(expected_paths):
        raise ValueError("Miles HF controller event-journal reference is malformed")
    if (
        set(start) != _HF_START_FIELDS
        or start.get("status") != "watching"
        or start.get("source_plan_sha256") != "sha256:" + digest(plan)
        or start.get("source_request_sha256") != submitted["request_sha256"]
        or start.get("api_base_url") != submitted["api"]["base_url"]
        or start.get("api_run_id") != submitted["api"]["run_id"]
        or start.get("api_run_name") != submitted["api"]["run_name"]
        or start.get("kube_context") != DEV_KUBE_CONTEXT
        or start.get("cluster") != "dev"
        or start.get("namespace") != NAMESPACE
        or start.get("namespace_uid") != DEV_NAMESPACE_UID
        or any(
            start.get(key) is not False
            for key in ("private_logs_included", "metric_values_included", "task_content_included")
        )
    ):
        raise ValueError("Miles HF controller capture start changed")
    started_at = _time(start.get("started_at"), "capture start")
    previous_observed = started_at
    for sequence, event in enumerate(events):
        observed = _time(event.get("observed_at"), "event observation")
        if (
            set(event) != _EVENT_FIELDS
            or event.get("sequence") != sequence
            or observed < previous_observed
            or event.get("source_plan_sha256") != start["source_plan_sha256"]
            or event.get("source_request_sha256") != start["source_request_sha256"]
            or event.get("api_run_id") != start["api_run_id"]
            or event.get("api_run_name") != start["api_run_name"]
            or event.get("namespace") != NAMESPACE
            or any(
                event.get(key) is not False
                for key in (
                    "private_logs_included",
                    "metric_values_included",
                    "task_content_included",
                )
            )
        ):
            raise ValueError("Miles HF controller event-journal sequence or identity changed")
        previous_observed = observed
    direct_export = plan.get("stage") == "export"
    kinds = ("Job", "Workload", "Pod") if direct_export else (
        "RayJob",
        "Workload",
        "RayCluster",
        "Pod",
    )
    by_kind = {kind: [row for row in events if row.get("kind") == kind] for kind in kinds}
    if any(not rows for rows in by_kind.values()):
        raise ValueError("Miles HF controller event-journal missed a lifecycle object")
    identities: dict[str, tuple[str, str]] = {}
    for kind, rows in by_kind.items():
        values = {(str(row.get("name")), str(row.get("uid"))) for row in rows}
        if len(values) != 1:
            raise ValueError("Miles HF controller event-journal identity changed")
        identities[kind] = next(iter(values))
    workload_name, workload_uid = identities["Workload"]
    pod_name, pod_uid = identities["Pod"]
    if direct_export:
        job_name, job_uid = identities["Job"]
        if (
            job_name != start["api_run_name"]
            or job_uid != start["api_run_id"]
            or any(
                row.get("owner") != {"kind": "Job", "name": job_name, "uid": job_uid}
                for kind in ("Workload", "Pod")
                for row in by_kind[kind]
            )
        ):
            raise ValueError("Miles HF controller event-journal ownership changed")
        controller_kind = "Job"
        expected_priority = "c2"
        expected_gpus = submitted["request"]["gpus"]
    else:
        rayjob_name, rayjob_uid = identities["RayJob"]
        raycluster_name, raycluster_uid = identities["RayCluster"]
        if (
            rayjob_name != start["api_run_name"]
            or any(
                row.get("owner")
                != {"kind": "RayJob", "name": rayjob_name, "uid": rayjob_uid}
                for kind in ("Workload", "RayCluster")
                for row in by_kind[kind]
            )
            or any(
                row.get("owner")
                != {"kind": "RayCluster", "name": raycluster_name, "uid": raycluster_uid}
                for row in by_kind["Pod"]
            )
        ):
            raise ValueError("Miles HF controller event-journal ownership changed")
        controller_kind = "RayJob"
        expected_priority = "c1"
        expected_gpus = submitted["request"]["gpus_per_worker"]
    admissions = [row["admitted_at"] for row in by_kind["Workload"] if row.get("admitted_at")]
    if not admissions or started_at > min(
        _time(value, "Workload admission") for value in admissions
    ):
        raise ValueError("Miles HF controller capture did not precede admission")
    if direct_export and any(
        row.get("priority_class") != "q2" for row in by_kind["Workload"]
    ):
        raise ValueError("Miles HF direct export Workload queue priority changed")
    terminal_controllers = [
        row for row in by_kind[controller_kind] if row.get("controller_status") == "SUCCEEDED"
    ]
    terminal_pods = [
        row
        for row in by_kind["Pod"]
        if isinstance(row.get("pod"), Mapping)
        and row["pod"].get("phase") == "Succeeded"
        and row["pod"].get("exit_code") == 0
        and row["pod"].get("termination_reason") == "Completed"
    ]
    if not terminal_controllers or not terminal_pods:
        raise ValueError("Miles HF controller event-journal missed terminal success")
    terminal_controller = terminal_controllers[-1]
    pod = dict(terminal_pods[-1]["pod"])
    if (
        terminal_controller.get("ttl_seconds_after_finished") != 0
        or terminal_controller.get("priority_class") != expected_priority
        or (
            not direct_export
            and (
                terminal_controller.get("shutdown_after_job_finishes") is not True
                or terminal_controller.get("raycluster_name") != raycluster_name
            )
        )
        or pod.get("container_restarts") != 0
        or pod.get("gpus") != expected_gpus
        or not isinstance(pod.get("runtime_image_id"), str)
        or not pod["runtime_image_id"].endswith(plan["execution"]["image"].rsplit("@", 1)[1])
        or _time(pod.get("terminated_at"), "Pod termination")
        > _time(terminal_pods[-1]["observed_at"], "terminal Pod observation")
    ):
        raise ValueError("Miles HF controller event-journal execution changed")
    controller_pods = controller.get("pods")
    controller_pod = (
        controller_pods[0]
        if isinstance(controller_pods, list) and len(controller_pods) == 1
        else {}
    )
    identity_matches = (
        controller.get("job_name") == job_name
        and controller.get("job_uid") == job_uid
        if direct_export
        else controller.get("rayjob_name") == rayjob_name
        and controller.get("rayjob_uid") == rayjob_uid
        and controller.get("raycluster_name") == raycluster_name
        and controller.get("raycluster_uid") == raycluster_uid
    )
    if (
        controller.get("event_journal") != binding
        or controller.get("api_run_id") != start["api_run_id"]
        or controller.get("api_run_name") != start["api_run_name"]
        or not identity_matches
        or controller.get("workload_name") != workload_name
        or controller.get("workload_uid") != workload_uid
        or not isinstance(controller_pod, Mapping)
        or controller_pod.get("name") != pod_name
        or controller_pod.get("uid") != pod_uid
        or any(
            controller_pod.get(key) != pod.get(key)
            for key in (
                "phase",
                "exit_code",
                "termination_reason",
                "terminated_at",
                "runtime_image_id",
                "container_restarts",
                "gpus",
            )
        )
        or _time(controller.get("observed_at"), "controller observation")
        < max(
            _time(
                terminal_controller["observed_at"],
                f"terminal {controller_kind} observation",
            ),
            _time(terminal_pods[-1]["observed_at"], "terminal Pod observation"),
        )
    ):
        raise ValueError("Miles HF controller differs from its event journal")
    return {"events": events, "identities": identities, "pod": pod}


def validate_hf_release_journal(
    plan: dict[str, Any],
    submission: dict[str, Any],
    controller: Mapping[str, Any],
    observed_at: object,
) -> None:
    """Require durable DELETED events for every exact allocated UID."""
    result = validate_hf_event_journal(
        plan, submission, controller.get("event_journal"), controller
    )
    released_at = _time(observed_at, "release observation")
    expected = {kind: identity[1] for kind, identity in result["identities"].items()}
    for kind, uid in expected.items():
        deleted = [
            row
            for row in result["events"]
            if row.get("kind") == kind
            and row.get("uid") == uid
            and row.get("event_type") == "DELETED"
        ]
        if (
            not deleted
            or max(_time(row["observed_at"], "deletion observation") for row in deleted)
            > released_at
        ):
            raise ValueError("Miles HF release lacks a durable exact-UID deletion event")


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

    submitted = _validate_submission(submission, plan, check_files=False)
    if plan.get("schema") == "cyber_miles_hf_export_job_plan_v1":
        from .miles_hf_export import RELOAD_CONTROLLER_SCHEMA
        from .miles_hf_export_job import EXPORT_CONTROLLER_SCHEMA, validate_event_controller

        controller_schema = (
            EXPORT_CONTROLLER_SCHEMA if plan.get("stage") == "export" else RELOAD_CONTROLLER_SCHEMA
        )
        controller, controller_file_sha256 = _read(controller_path, controller_schema)
        validate_event_controller(controller, plan, submission)
    else:
        from .miles_acceptance import validate_controller_observation

        controller, controller_file_sha256 = _read(controller_path, CONTROLLER_SCHEMA)
        validate_controller_observation(controller, plan, submission)
    if (
        plan.get("schema") == "cyber_miles_hf_export_job_plan_v1"
        and plan.get("stage") == "export"
    ):
        expected_absence = {
            "job_present": False,
            "workload_present": False,
            "quota_reservation_present": False,
            "gpu_pods_present": False,
            "active_gpus": 0,
        }
    else:
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
    if plan.get("schema") == "cyber_miles_hf_export_job_plan_v1":
        validate_hf_release_journal(plan, submission, controller, when)
        from .miles_hf_export_job import release_from_event_journal

        return _write(
            output,
            release_from_event_journal(
                plan=plan,
                submission=submission,
                controller=controller,
                controller_path=controller_path,
                controller_file_sha256=controller_file_sha256,
                observed_at=when,
            ),
        )
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
