"""Build a score-blind GLM v32 create authorization from one live observation.

The builder admits either no active project serving server, or one six-GPU
Qwen server whose terminal score-free qualification authority is independently
self-digested and exactly matches the live Jobs API and Kubernetes UID chain.
It never accepts caller-authored footprint counters as authority.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from evals.fleet import exact_pass4_crypto as crypto

LIVE_SCHEMA = "fleet-glm53-dedicated-v32-live-create-observation-v1"
QWEN_AUTHORITY_SCHEMA = "fleet-qwen-dp6-scorefree-coexistence-authority-v1"
NAMESPACE = "fleet-train-jobs"
BASE_URL = "https://api.ft.flt.build"
ACTIVE_API_STATUSES = {"SUBMITTED", "SUSPENDED", "RUNNING"}
TERMINAL_JOB_CONDITIONS = {"Complete"}
MAX_NODES = 2
MAX_GPUS = 16

QWEN_AUTHORITY_KEYS = {
    "schema_version",
    "status",
    "qualified_at_epoch",
    "api_run_id",
    "server_title",
    "server_run_dir",
    "rayjob_name",
    "rayjob_uid",
    "workload_name",
    "workload_uid",
    "raycluster_name",
    "raycluster_uid",
    "head_pod_name",
    "head_pod_uid",
    "head_pod_node",
    "service_name",
    "service_uid",
    "qualifier_job_name",
    "qualifier_job_uid",
    "qualifier_pod_name",
    "qualifier_pod_uid",
    "qualification_result_path",
    "qualification_result_file_sha256",
    "qualification_result_receipt_sha256",
    "requested_nodes",
    "requested_gpus",
    "head_pod_running_ready",
    "head_pod_restarts",
    "qualified_score_free",
    "fleet_task_instance_calls",
    "fleet_session_calls",
    "verifier_calls",
    "scoring_calls",
    "protected_content_included",
    "receipt_sha256",
}

LIVE_KEYS = {
    "schema_version",
    "status",
    "observed_at_epoch",
    "mode",
    "jobs_api_pages",
    "active_project_runs",
    "project_rayjob_count",
    "project_gpu_pod_count",
    "coexisting_qwen_server",
    "qwen_qualification_authority",
    "qwen_qualification_authority_receipt_sha256",
    "sfs_observation",
    "preview_http_status",
    "active_dedicated_nodes",
    "active_dedicated_gpus",
    "planned_nodes_after_create",
    "planned_gpus_after_create",
    "maximum_nodes",
    "maximum_gpus",
    "api_mutation_calls",
    "fleet_task_instance_calls",
    "fleet_session_calls",
    "verifier_calls",
    "scoring_calls",
    "protected_content_included",
    "receipt_sha256",
}


class LiveAuthorizationError(RuntimeError):
    """The score-blind live authorization observation failed closed."""


class Backend(Protocol):
    """Read-only observation surface used by the live builder."""

    def list_runs(self) -> tuple[list[dict[str, Any]], int]: ...

    def get_run(self, api_run_id: str) -> dict[str, Any] | None: ...

    def kubernetes_inventory(self) -> dict[str, Any]: ...

    def sfs_observation(
        self, *, absent_paths: tuple[str, ...], qualification_path: str | None
    ) -> dict[str, Any]: ...

    def preview(self, payload: Mapping[str, Any]) -> int: ...


def _nonzero_uuid(value: object) -> bool:
    try:
        return uuid.UUID(str(value)).int != 0
    except (AttributeError, TypeError, ValueError):
        return False


def _condition_true(value: Mapping[str, Any], condition_type: str) -> bool:
    return any(
        row.get("type") == condition_type and row.get("status") == "True"
        for row in value.get("status", {}).get("conditions") or []
        if isinstance(row, dict)
    )


def _pod_gpu_requests(pod: Mapping[str, Any]) -> int:
    try:
        return sum(
            int(
                ((container.get("resources") or {}).get("requests") or {}).get(
                    "nvidia.com/gpu", 0
                )
            )
            for container in (pod.get("spec") or {}).get("containers") or []
        )
    except (TypeError, ValueError) as exc:
        raise LiveAuthorizationError("v32_kubernetes_gpu_shape_invalid") from exc


def _pod_run_dir(pod: Mapping[str, Any]) -> str | None:
    values = [
        env.get("value")
        for container in (pod.get("spec") or {}).get("containers") or []
        for env in container.get("env") or []
        if env.get("name") in {"RUN_DIR", "GLM53_RUN_DIR", "QWEN38_RUN_DIR"}
        and isinstance(env.get("value"), str)
    ]
    unique = set(values)
    return next(iter(unique)) if len(unique) == 1 else None


def _is_project_run_dir(value: object) -> bool:
    return isinstance(value, str) and value.startswith(
        "/mnt/sfs/jobs/chris-cyber-evalserve-"
    )


def validate_qwen_authority(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the injected terminal, score-free Qwen authority exactly."""

    if (
        set(value) != QWEN_AUTHORITY_KEYS
        or value.get("schema_version") != QWEN_AUTHORITY_SCHEMA
        or value.get("status") != "QUALIFIED_SCORE_FREE_TERMINAL"
        or value.get("receipt_sha256")
        != crypto.digest_without(dict(value), "receipt_sha256")
        or not isinstance(value.get("qualified_at_epoch"), (int, float))
        or isinstance(value.get("qualified_at_epoch"), bool)
        or not isinstance(value.get("api_run_id"), str)
        or not str(value["api_run_id"]).startswith("ft-run-")
        or not _is_project_run_dir(value.get("server_run_dir"))
        or "q38" not in str(value.get("server_run_dir"))
        or value.get("rayjob_name") != value.get("api_run_id")
        or value.get("requested_nodes") != 1
        or value.get("requested_gpus") != 6
        or value.get("head_pod_running_ready") is not True
        or value.get("head_pod_restarts") != 0
        or value.get("qualified_score_free") is not True
        or value.get("fleet_task_instance_calls") != 0
        or value.get("fleet_session_calls") != 0
        or value.get("verifier_calls") != 0
        or value.get("scoring_calls") != 0
        or value.get("protected_content_included") is not False
        or not isinstance(value.get("qualification_result_path"), str)
        or not str(value["qualification_result_path"]).startswith("/mnt/sfs/jobs/")
        or not isinstance(value.get("qualification_result_file_sha256"), str)
        or not str(value["qualification_result_file_sha256"]).startswith("sha256:")
        or not isinstance(value.get("qualification_result_receipt_sha256"), str)
        or not str(value["qualification_result_receipt_sha256"]).startswith("sha256:")
    ):
        raise LiveAuthorizationError("v32_qwen_qualification_authority_invalid")
    for field in (
        "rayjob_uid",
        "workload_uid",
        "raycluster_uid",
        "head_pod_uid",
        "service_uid",
        "qualifier_job_uid",
        "qualifier_pod_uid",
    ):
        if not _nonzero_uuid(value.get(field)):
            raise LiveAuthorizationError("v32_qwen_qualification_authority_invalid")
    for field in (
        "server_title",
        "workload_name",
        "raycluster_name",
        "head_pod_name",
        "head_pod_node",
        "service_name",
        "qualifier_job_name",
        "qualifier_pod_name",
    ):
        if not isinstance(value.get(field), str) or not value.get(field):
            raise LiveAuthorizationError("v32_qwen_qualification_authority_invalid")
    return dict(value)


def _metadata(value: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = value.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _owned_by(value: Mapping[str, Any], kind: str, uid: str) -> bool:
    return any(
        row.get("kind") == kind
        and row.get("uid") == uid
        and row.get("controller") is not False
        for row in _metadata(value).get("ownerReferences") or []
        if isinstance(row, dict)
    )


def _exact_item(
    items: list[dict[str, Any]], kind: str, name: str, uid: str
) -> dict[str, Any]:
    matches = [
        row
        for row in items
        if row.get("kind") == kind
        and _metadata(row).get("name") == name
        and _metadata(row).get("uid") == uid
    ]
    if len(matches) != 1:
        raise LiveAuthorizationError("v32_qwen_kubernetes_uid_chain_invalid")
    return matches[0]


def _validate_qwen_live_chain(
    authority: Mapping[str, Any],
    api_run: Mapping[str, Any],
    items: list[dict[str, Any]],
    sfs: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        api_run.get("run_dir") != authority.get("server_run_dir")
        or api_run.get("title")
        not in {None, authority.get("server_title")}
        or str(api_run.get("status") or "").upper() not in ACTIVE_API_STATUSES
    ):
        raise LiveAuthorizationError("v32_qwen_jobs_api_identity_invalid")
    rayjob = _exact_item(
        items, "RayJob", str(authority["rayjob_name"]), str(authority["rayjob_uid"])
    )
    workload = _exact_item(
        items,
        "Workload",
        str(authority["workload_name"]),
        str(authority["workload_uid"]),
    )
    raycluster = _exact_item(
        items,
        "RayCluster",
        str(authority["raycluster_name"]),
        str(authority["raycluster_uid"]),
    )
    pod = _exact_item(
        items, "Pod", str(authority["head_pod_name"]), str(authority["head_pod_uid"])
    )
    service = _exact_item(
        items, "Service", str(authority["service_name"]), str(authority["service_uid"])
    )
    qualifier_job = _exact_item(
        items,
        "Job",
        str(authority["qualifier_job_name"]),
        str(authority["qualifier_job_uid"]),
    )
    qualifier_pod = _exact_item(
        items,
        "Pod",
        str(authority["qualifier_pod_name"]),
        str(authority["qualifier_pod_uid"]),
    )
    pod_statuses = (pod.get("status") or {}).get("containerStatuses") or []
    restarts = sum(int(row.get("restartCount") or 0) for row in pod_statuses)
    qualifier_restarts = sum(
        int(row.get("restartCount") or 0)
        for row in (qualifier_pod.get("status") or {}).get("containerStatuses") or []
    )
    if (
        not _condition_true(pod, "Ready")
        or (pod.get("status") or {}).get("phase") != "Running"
        or restarts != 0
        or _pod_gpu_requests(pod) != 6
        or (pod.get("spec") or {}).get("nodeName") != authority.get("head_pod_node")
        or _pod_run_dir(pod) != authority.get("server_run_dir")
        or not _condition_true(qualifier_job, "Complete")
        or (qualifier_pod.get("status") or {}).get("phase") != "Succeeded"
        or qualifier_restarts != 0
        or not _owned_by(workload, "RayJob", str(authority["rayjob_uid"]))
        or not _owned_by(raycluster, "RayJob", str(authority["rayjob_uid"]))
        or not _owned_by(pod, "RayCluster", str(authority["raycluster_uid"]))
        or not _owned_by(service, "RayCluster", str(authority["raycluster_uid"]))
        or not _owned_by(
            qualifier_pod, "Job", str(authority["qualifier_job_uid"])
        )
        or sfs.get("qualification_result_present") is not True
        or sfs.get("qualification_result_file_sha256")
        != authority.get("qualification_result_file_sha256")
    ):
        raise LiveAuthorizationError("v32_qwen_live_qualification_invalid")
    return {
        key: authority[key]
        for key in (
            "api_run_id",
            "server_title",
            "server_run_dir",
            "rayjob_name",
            "rayjob_uid",
            "workload_name",
            "workload_uid",
            "raycluster_name",
            "raycluster_uid",
            "head_pod_name",
            "head_pod_uid",
            "head_pod_node",
            "service_name",
            "service_uid",
            "qualifier_job_name",
            "qualifier_job_uid",
            "qualifier_pod_name",
            "qualifier_pod_uid",
            "qualification_result_path",
            "qualification_result_file_sha256",
            "qualification_result_receipt_sha256",
            "requested_nodes",
            "requested_gpus",
            "qualified_score_free",
            "head_pod_running_ready",
            "head_pod_restarts",
        )
    } | {
        "api_status": api_run["status"],
        "rayjob_status": (rayjob.get("status") or {}).get("jobStatus"),
    }


def _active_project_runs(
    backend: Backend, rows: list[dict[str, Any]], title: str, run_dir: str
) -> list[dict[str, Any]]:
    active: list[dict[str, Any]] = []
    for row in rows:
        name = row.get("name")
        row_run_dir = row.get("run_dir")
        if row.get("title") == title or row_run_dir == run_dir:
            raise LiveAuthorizationError("v32_jobs_api_identity_already_exists")
        if not isinstance(name, str) or not _is_project_run_dir(row_run_dir):
            continue
        current = backend.get_run(name)
        if current is None:
            continue
        if current.get("run_dir") != row_run_dir:
            raise LiveAuthorizationError("v32_jobs_api_history_live_drift")
        if str(current.get("status") or "").upper() in ACTIVE_API_STATUSES:
            active.append({**current, "_observed_api_run_id": name})
    return active


def _project_kubernetes(
    inventory: Mapping[str, Any], title: str, run_dir: str
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    rows = inventory.get("items")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise LiveAuthorizationError("v32_kubernetes_inventory_invalid")
    items: list[dict[str, Any]] = rows
    serialized = json.dumps(inventory, sort_keys=True)
    if title in serialized or run_dir in serialized:
        raise LiveAuthorizationError("v32_kubernetes_identity_already_exists")
    rayjobs = [
        row
        for row in items
        if row.get("kind") == "RayJob"
        and "/mnt/sfs/jobs/chris-cyber-evalserve-" in json.dumps(row, sort_keys=True)
        and str((row.get("status") or {}).get("jobStatus") or "").upper()
        not in {"SUCCEEDED", "FAILED", "STOPPED"}
    ]
    gpu_pods = [
        row
        for row in items
        if row.get("kind") == "Pod"
        and (row.get("status") or {}).get("phase") in {"Pending", "Running"}
        and _pod_gpu_requests(row) > 0
        and _is_project_run_dir(_pod_run_dir(row))
    ]
    project_objects = [
        row
        for row in items
        if row.get("kind") in {"RayJob", "RayCluster", "Workload", "Service"}
        and not (
            row.get("kind") == "RayJob"
            and str((row.get("status") or {}).get("jobStatus") or "").upper()
            in {"SUCCEEDED", "FAILED", "STOPPED"}
        )
        and (
            "/mnt/sfs/jobs/chris-cyber-evalserve-"
            in json.dumps(row, sort_keys=True)
            or (_metadata(row).get("labels") or {}).get(
                "cyber-post-train.fleet.ai/owner"
            )
            == "chris"
        )
    ]
    return items, rayjobs, gpu_pods, project_objects


def build_live_authorization(
    *,
    backend: Backend,
    payload: Mapping[str, Any],
    title: str,
    run_dir: str,
    control_result_path: str,
    request_sha256: str,
    priority_class: str,
    qwen_authority: Mapping[str, Any] | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    """Observe live state once and produce the only admissible authorization."""

    observed_at = time.time() if now is None else now
    rows, page_count = backend.list_runs()
    active = _active_project_runs(backend, rows, title, run_dir)
    inventory = backend.kubernetes_inventory()
    items, project_rayjobs, project_gpu_pods, project_objects = _project_kubernetes(
        inventory, title, run_dir
    )
    authority = validate_qwen_authority(qwen_authority) if qwen_authority else None
    qualification_path = (
        str(authority["qualification_result_path"]) if authority is not None else None
    )
    sfs = backend.sfs_observation(
        absent_paths=(run_dir, control_result_path),
        qualification_path=qualification_path,
    )
    if (
        set(sfs)
        != {
            "observer_pod_name",
            "observer_pod_uid",
            "observer_sfs_mount_path",
            "absent_paths",
            "qualification_result_present",
            "qualification_result_file_sha256",
        }
        or not _nonzero_uuid(sfs.get("observer_pod_uid"))
        or sfs.get("absent_paths") != [run_dir, control_result_path]
    ):
        raise LiveAuthorizationError("v32_sfs_observation_invalid")

    coexisting: dict[str, Any] | None
    if authority is None:
        if active or project_rayjobs or project_gpu_pods or project_objects:
            raise LiveAuthorizationError("v32_zero_state_has_project_server")
        mode, active_nodes, active_gpus = "ZERO_PROJECT_SERVER", 0, 0
        coexisting = None
        authority_digest = None
    else:
        if (
            len(active) != 1
            or active[0].get("_observed_api_run_id") != authority.get("api_run_id")
        ):
            raise LiveAuthorizationError("v32_qwen_jobs_api_cardinality_invalid")
        coexisting = _validate_qwen_live_chain(authority, active[0], items, sfs)
        exact_rayjob = _exact_item(
            items,
            "RayJob",
            str(authority["rayjob_name"]),
            str(authority["rayjob_uid"]),
        )
        project_rayjobs = list(
            {
                str(_metadata(row).get("uid")): row
                for row in [*project_rayjobs, exact_rayjob]
            }.values()
        )
        expected_gpu_uids = {authority["head_pod_uid"]}
        if {
            _metadata(row).get("uid") for row in project_gpu_pods
        } != expected_gpu_uids:
            raise LiveAuthorizationError("v32_qwen_orphan_gpu_pod_detected")
        expected_objects = {
            ("RayJob", authority["rayjob_uid"]),
            ("RayCluster", authority["raycluster_uid"]),
            ("Workload", authority["workload_uid"]),
            ("Service", authority["service_uid"]),
        }
        observed_objects = {
            (str(row.get("kind")), _metadata(row).get("uid"))
            for row in project_objects
        } | expected_objects
        if observed_objects != expected_objects:
            raise LiveAuthorizationError("v32_qwen_orphan_project_object_detected")
        mode, active_nodes, active_gpus = "QUALIFIED_QWEN_DP6_COEXISTENCE", 1, 6
        authority_digest = authority["receipt_sha256"]

    planned_nodes, planned_gpus = active_nodes + 1, active_gpus + 8
    if planned_nodes > MAX_NODES or planned_gpus > MAX_GPUS:
        raise LiveAuthorizationError("v32_project_capacity_exceeded")
    preview_status = backend.preview(payload)
    if preview_status != 200:
        raise LiveAuthorizationError("v32_jobs_api_preview_invalid")

    live: dict[str, Any] = {
        "schema_version": LIVE_SCHEMA,
        "status": "PASSED_SCORE_BLIND_LIVE_CREATE_OBSERVATION",
        "observed_at_epoch": observed_at,
        "mode": mode,
        "jobs_api_pages": page_count,
        "active_project_runs": [
            {
                "api_run_id": row.get("_observed_api_run_id"),
                "run_dir": row.get("run_dir"),
                "status": row.get("status"),
            }
            for row in active
        ],
        "project_rayjob_count": len(project_rayjobs),
        "project_gpu_pod_count": len(project_gpu_pods),
        "coexisting_qwen_server": coexisting,
        "qwen_qualification_authority": authority,
        "qwen_qualification_authority_receipt_sha256": authority_digest,
        "sfs_observation": sfs,
        "preview_http_status": preview_status,
        "active_dedicated_nodes": active_nodes,
        "active_dedicated_gpus": active_gpus,
        "planned_nodes_after_create": planned_nodes,
        "planned_gpus_after_create": planned_gpus,
        "maximum_nodes": MAX_NODES,
        "maximum_gpus": MAX_GPUS,
        "api_mutation_calls": 0,
        "fleet_task_instance_calls": 0,
        "fleet_session_calls": 0,
        "verifier_calls": 0,
        "scoring_calls": 0,
        "protected_content_included": False,
    }
    live["receipt_sha256"] = crypto.digest_without(live, "receipt_sha256")
    authorization: dict[str, Any] = {
        "schema_version": "fleet-glm53-dedicated-v32-create-authorization-v1",
        "status": "PASSED_LIVE_CREATE_GATES",
        "observed_at_epoch": observed_at,
        "server_title": title,
        "server_run_dir": run_dir,
        "request_sha256": request_sha256,
        "preview_http_status": preview_status,
        "jobs_api_title_matches": 0,
        "jobs_api_run_dir_matches": 0,
        "kubernetes_identity_or_remnant_matches": 0,
        "sfs_run_dir_absent": True,
        "control_result_absent": True,
        "active_dedicated_nodes": active_nodes,
        "active_dedicated_gpus": active_gpus,
        "planned_nodes_after_create": planned_nodes,
        "planned_gpus_after_create": planned_gpus,
        "coexisting_qwen_server": coexisting,
        "live_observation": live,
        "priority_class": priority_class,
        "preemption_policy": "Never",
        "server_launch_authorized": True,
        "watchdog_handoff_required_immediately": True,
        "qualification_launch_authorized": False,
        "scored_launch_authorized": False,
        "api_mutation_calls": 0,
        "protected_content_included": False,
    }
    authorization["receipt_sha256"] = crypto.digest_without(
        authorization, "receipt_sha256"
    )
    return authorization


def validate_live_observation(
    value: Mapping[str, Any], authorization: Mapping[str, Any]
) -> None:
    if (
        set(value) != LIVE_KEYS
        or value.get("schema_version") != LIVE_SCHEMA
        or value.get("status") != "PASSED_SCORE_BLIND_LIVE_CREATE_OBSERVATION"
        or value.get("receipt_sha256")
        != crypto.digest_without(dict(value), "receipt_sha256")
        or value.get("observed_at_epoch") != authorization.get("observed_at_epoch")
        or value.get("preview_http_status") != 200
        or value.get("active_dedicated_nodes")
        != authorization.get("active_dedicated_nodes")
        or value.get("active_dedicated_gpus")
        != authorization.get("active_dedicated_gpus")
        or value.get("planned_nodes_after_create")
        != authorization.get("planned_nodes_after_create")
        or value.get("planned_gpus_after_create")
        != authorization.get("planned_gpus_after_create")
        or value.get("coexisting_qwen_server")
        != authorization.get("coexisting_qwen_server")
        or value.get("maximum_nodes") != MAX_NODES
        or value.get("maximum_gpus") != MAX_GPUS
        or not isinstance(value.get("jobs_api_pages"), int)
        or value.get("jobs_api_pages", 0) < 1
        or value.get("api_mutation_calls") != 0
        or value.get("fleet_task_instance_calls") != 0
        or value.get("fleet_session_calls") != 0
        or value.get("verifier_calls") != 0
        or value.get("scoring_calls") != 0
        or value.get("protected_content_included") is not False
    ):
        raise LiveAuthorizationError("v32_live_observation_invalid")
    sfs = value.get("sfs_observation")
    absent_paths = sfs.get("absent_paths") if isinstance(sfs, dict) else None
    if (
        not isinstance(sfs, dict)
        or set(sfs)
        != {
            "observer_pod_name",
            "observer_pod_uid",
            "observer_sfs_mount_path",
            "absent_paths",
            "qualification_result_present",
            "qualification_result_file_sha256",
        }
        or not _nonzero_uuid(sfs.get("observer_pod_uid"))
        or not isinstance(sfs.get("observer_pod_name"), str)
        or not sfs.get("observer_pod_name")
        or not isinstance(sfs.get("observer_sfs_mount_path"), str)
        or not isinstance(absent_paths, list)
        or len(absent_paths) != 2
        or absent_paths[0] != authorization.get("server_run_dir")
        or not str(absent_paths[1]).endswith("/CREATED.json")
    ):
        raise LiveAuthorizationError("v32_live_observation_invalid")
    if value.get("mode") == "ZERO_PROJECT_SERVER":
        if (
            value.get("active_project_runs") != []
            or value.get("project_rayjob_count") != 0
            or value.get("project_gpu_pod_count") != 0
            or value.get("qwen_qualification_authority_receipt_sha256") is not None
            or value.get("qwen_qualification_authority") is not None
            or value.get("coexisting_qwen_server") is not None
            or sfs.get("qualification_result_present") is not False
            or sfs.get("qualification_result_file_sha256") is not None
        ):
            raise LiveAuthorizationError("v32_live_observation_invalid")
    elif value.get("mode") == "QUALIFIED_QWEN_DP6_COEXISTENCE":
        coexisting = value.get("coexisting_qwen_server")
        authority = value.get("qwen_qualification_authority")
        try:
            validated_authority = (
                validate_qwen_authority(authority)
                if isinstance(authority, dict)
                else None
            )
        except LiveAuthorizationError as exc:
            raise LiveAuthorizationError("v32_live_observation_invalid") from exc
        if (
            not isinstance(coexisting, dict)
            or validated_authority is None
            or value.get("active_dedicated_nodes") != 1
            or value.get("active_dedicated_gpus") != 6
            or value.get("planned_nodes_after_create") != 2
            or value.get("planned_gpus_after_create") != 14
            or len(value.get("active_project_runs") or []) != 1
            or value.get("project_rayjob_count") != 1
            or value.get("project_gpu_pod_count") != 1
            or not isinstance(
                value.get("qwen_qualification_authority_receipt_sha256"), str
            )
            or value.get("qwen_qualification_authority_receipt_sha256")
            != validated_authority.get("receipt_sha256")
            or sfs.get("qualification_result_present") is not True
            or sfs.get("qualification_result_file_sha256")
            != validated_authority.get("qualification_result_file_sha256")
            or any(
                coexisting.get(key) != validated_authority.get(key)
                for key in (
                    "api_run_id",
                    "server_title",
                    "server_run_dir",
                    "rayjob_name",
                    "rayjob_uid",
                    "workload_name",
                    "workload_uid",
                    "raycluster_name",
                    "raycluster_uid",
                    "head_pod_name",
                    "head_pod_uid",
                    "head_pod_node",
                    "service_name",
                    "service_uid",
                    "qualifier_job_name",
                    "qualifier_job_uid",
                    "qualifier_pod_name",
                    "qualifier_pod_uid",
                    "qualification_result_path",
                    "qualification_result_file_sha256",
                    "qualification_result_receipt_sha256",
                    "requested_nodes",
                    "requested_gpus",
                    "qualified_score_free",
                    "head_pod_running_ready",
                    "head_pod_restarts",
                )
            )
        ):
            raise LiveAuthorizationError("v32_live_observation_invalid")
        active_row = value["active_project_runs"][0]
        if (
            not isinstance(active_row, dict)
            or set(active_row) != {"api_run_id", "run_dir", "status"}
            or active_row.get("api_run_id") != validated_authority.get("api_run_id")
            or active_row.get("run_dir") != validated_authority.get("server_run_dir")
            or str(active_row.get("status") or "").upper() not in ACTIVE_API_STATUSES
        ):
            raise LiveAuthorizationError("v32_live_observation_invalid")
    else:
        raise LiveAuthorizationError("v32_live_observation_invalid")


class SystemBackend:
    """Read-only deployed Jobs API/Kubernetes/SFS observation backend."""

    def __init__(self) -> None:
        token = os.environ.get("FLEET_API_KEY", "")
        if not token:
            raise LiveAuthorizationError("v32_jobs_api_credential_absent")
        self._headers = {
            "Authorization": "Bearer " + token,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        self._inventory: dict[str, Any] | None = None

    def _request(
        self, method: str, route: str, body: Mapping[str, Any] | None = None
    ) -> tuple[int, dict[str, Any]]:
        request = urllib.request.Request(
            BASE_URL + route,
            method=method,
            headers=self._headers,
            data=crypto.canonical_json(dict(body)) if body is not None else None,
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                status, raw = int(response.status), response.read()
        except urllib.error.HTTPError as exc:
            status, raw = int(exc.code), exc.read()
        try:
            value = json.loads(raw) if raw else {}
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise LiveAuthorizationError("v32_jobs_api_json_invalid") from exc
        if not isinstance(value, dict):
            raise LiveAuthorizationError("v32_jobs_api_json_invalid")
        return status, value

    def list_runs(self) -> tuple[list[dict[str, Any]], int]:
        rows: list[dict[str, Any]] = []
        offset = 0
        pages = 0
        while True:
            status, page = self._request(
                "GET", f"/v1/runs?limit=200&offset={offset}"
            )
            if status != 200 or not isinstance(page.get("items"), list):
                raise LiveAuthorizationError("v32_jobs_api_inventory_invalid")
            current = [row for row in page["items"] if isinstance(row, dict)]
            if len(current) != len(page["items"]):
                raise LiveAuthorizationError("v32_jobs_api_inventory_invalid")
            pages += 1
            rows.extend(current)
            if not page.get("has_more"):
                return rows, pages
            if not current:
                raise LiveAuthorizationError("v32_jobs_api_pagination_stalled")
            offset += len(current)

    def get_run(self, api_run_id: str) -> dict[str, Any] | None:
        status, value = self._request(
            "GET", "/v1/runs/" + urllib.parse.quote(api_run_id, safe="")
        )
        if status == 404:
            return None
        if status != 200:
            raise LiveAuthorizationError("v32_jobs_api_exact_get_invalid")
        return value

    def kubernetes_inventory(self) -> dict[str, Any]:
        command = [
            "kubectl",
            "-n",
            NAMESPACE,
            "get",
            (
                "rayjobs.ray.io,rayclusters.ray.io,workloads.kueue.x-k8s.io,"
                "jobs,pods,services"
            ),
            "-o",
            "json",
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise LiveAuthorizationError("v32_kubernetes_inventory_failed")
        try:
            value = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise LiveAuthorizationError("v32_kubernetes_inventory_invalid") from exc
        if not isinstance(value, dict):
            raise LiveAuthorizationError("v32_kubernetes_inventory_invalid")
        self._inventory = value
        return value

    def _observer(self) -> tuple[str, str, str]:
        inventory = self._inventory
        if inventory is None:
            raise LiveAuthorizationError("v32_kubernetes_inventory_not_observed")
        candidates: list[tuple[str, str, str]] = []
        for pod in inventory.get("items") or []:
            if pod.get("kind") != "Pod" or not _condition_true(pod, "Ready"):
                continue
            volumes = {
                row.get("name"): (row.get("persistentVolumeClaim") or {}).get(
                    "claimName"
                )
                for row in (pod.get("spec") or {}).get("volumes") or []
            }
            for container in (pod.get("spec") or {}).get("containers") or []:
                for mount in container.get("volumeMounts") or []:
                    if volumes.get(mount.get("name")) == "sfs-shared":
                        metadata = _metadata(pod)
                        candidates.append(
                            (
                                str(metadata.get("name")),
                                str(metadata.get("uid")),
                                str(mount.get("mountPath")),
                            )
                        )
        candidates = [row for row in candidates if _nonzero_uuid(row[1])]
        if not candidates:
            raise LiveAuthorizationError("v32_sfs_observer_absent")
        return sorted(set(candidates))[0]

    @staticmethod
    def _observed_path(mount: str, canonical: str) -> str:
        prefix = "/mnt/sfs/"
        if not canonical.startswith(prefix) or not mount.startswith("/"):
            raise LiveAuthorizationError("v32_sfs_path_invalid")
        relative = canonical.removeprefix(prefix)
        if not relative or ".." in Path(relative).parts:
            raise LiveAuthorizationError("v32_sfs_path_invalid")
        return str(Path(mount) / relative)

    def sfs_observation(
        self, *, absent_paths: tuple[str, ...], qualification_path: str | None
    ) -> dict[str, Any]:
        name, uid, mount = self._observer()
        observed_absent = [self._observed_path(mount, path) for path in absent_paths]
        command = ["kubectl", "-n", NAMESPACE, "exec", name, "--", "test"]
        for path in observed_absent:
            result = subprocess.run(
                [*command, "!", "-e", path], capture_output=True, check=False
            )
            if result.returncode != 0:
                raise LiveAuthorizationError("v32_sfs_create_identity_exists")
        present = False
        file_sha: str | None = None
        if qualification_path is not None:
            observed = self._observed_path(mount, qualification_path)
            result = subprocess.run(
                [
                    "kubectl",
                    "-n",
                    NAMESPACE,
                    "exec",
                    name,
                    "--",
                    "sha256sum",
                    observed,
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                raise LiveAuthorizationError("v32_qwen_qualification_result_absent")
            digest = result.stdout.split(maxsplit=1)[0]
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise LiveAuthorizationError("v32_qwen_qualification_result_digest_invalid")
            present, file_sha = True, "sha256:" + digest
        return {
            "observer_pod_name": name,
            "observer_pod_uid": uid,
            "observer_sfs_mount_path": mount,
            "absent_paths": list(absent_paths),
            "qualification_result_present": present,
            "qualification_result_file_sha256": file_sha,
        }

    def preview(self, payload: Mapping[str, Any]) -> int:
        status, _ = self._request("POST", "/v1/runs/preview", payload)
        return status


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise LiveAuthorizationError("v32_live_authority_input_invalid") from exc
    if not isinstance(value, dict):
        raise LiveAuthorizationError("v32_live_authority_input_invalid")
    return value


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise LiveAuthorizationError("v32_live_authorization_already_exists") from exc
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(crypto.canonical_json(dict(value)) + b"\n")


def main(argv: list[str] | None = None) -> int:
    """Observe exactly once and persist a fresh create authorization."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qwen-authority", type=Path)
    args = parser.parse_args(argv)
    from evals.fleet import glm53_dedicated_v32_create_v1 as server

    authority = (
        _load_object(args.qwen_authority) if args.qwen_authority is not None else None
    )
    value = build_live_authorization(
        backend=SystemBackend(),
        payload=server.payload(),
        title=server.TITLE,
        run_dir=server.RUN_DIR,
        control_result_path=server.RESULT_PATH,
        request_sha256=server.request_sha256(),
        priority_class=server.payload()["priority_class"],
        qwen_authority=authority,
    )
    server.validate_authorization(value)
    _write_once(args.output, value)
    print(
        json.dumps(
            {
                "status": value["status"],
                "mode": value["live_observation"]["mode"],
                "receipt_sha256": value["receipt_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
