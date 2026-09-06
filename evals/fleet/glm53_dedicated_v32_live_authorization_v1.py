"""Build the zero-project-server GLM v32 authorization from live evidence."""

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
from evals.fleet import glm53_dedicated_v32_stale_run_reconciliation_v1 as stale_runs

LIVE_SCHEMA = "fleet-glm53-dedicated-v32-live-create-observation-v1"
NAMESPACE = "fleet-train-jobs"
BASE_URL = "https://api.ft.flt.build"
ACTIVE_API_STATUSES = {"SUBMITTED", "SUSPENDED", "RUNNING"}
TERMINAL_API_STATUSES = {"CANCELLED", "COMPLETED", "FAILED", "STOPPED", "SUCCEEDED"}
MAX_NODES = 2
MAX_GPUS = 16
CONTROL_RESULT_PATH = (
    "/mnt/sfs/jobs/chris-cyber-evalserve-glm53-tp8-a-v32-create-control/CREATED.json"
)
SFS_OBSERVER_POD_NAME = "allie-dev"
SFS_OBSERVER_POD_UID = "73dabe56-60f8-4879-be9f-365196c502e3"
SFS_OBSERVER_MOUNT_PATH = "/shared"
LIVE_KEYS = {
    "schema_version",
    "status",
    "observed_at_epoch",
    "mode",
    "jobs_api_pages",
    "active_project_runs",
    "stale_run_reconciliation",
    "project_object_count",
    "project_gpu_pod_count",
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

    def sfs_observation(self, *, absent_paths: tuple[str, ...]) -> dict[str, Any]: ...

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


def _metadata(value: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = value.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


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


def _pod_run_dirs(pod: Mapping[str, Any]) -> set[str]:
    return {
        str(env["value"])
        for container in (pod.get("spec") or {}).get("containers") or []
        for env in container.get("env") or []
        if env.get("name") in {"RUN_DIR", "GLM53_RUN_DIR", "QWEN38_RUN_DIR"}
        and isinstance(env.get("value"), str)
    }


def _is_project_run_dir(value: object) -> bool:
    return isinstance(value, str) and value.startswith(
        "/mnt/sfs/jobs/chris-cyber-evalserve-"
    )


def _is_project_title(value: object) -> bool:
    return isinstance(value, str) and value.startswith("chris-cyber-evalserve-")


def _api_status(value: object) -> str:
    if not isinstance(value, str):
        raise LiveAuthorizationError("v32_jobs_api_status_invalid")
    status = value.upper()
    if status not in ACTIVE_API_STATUSES | TERMINAL_API_STATUSES:
        raise LiveAuthorizationError("v32_jobs_api_status_invalid")
    return status


def _active_project_runs(
    backend: Backend,
    rows: list[dict[str, Any]],
    title: str,
    run_dir: str,
    reconciliation: Mapping[str, Any],
    observed_at: float,
) -> list[dict[str, Any]]:
    try:
        stale_runs.validate_reconciliation(
            reconciliation,
            rows=rows,
            now=observed_at,
        )
    except stale_runs.ReconciliationError as exc:
        raise LiveAuthorizationError("v32_stale_run_reconciliation_invalid") from exc
    active: list[dict[str, Any]] = []
    for row in rows:
        name = row.get("name")
        row_run_dir = row.get("run_dir")
        if row.get("title") == title or row_run_dir == run_dir:
            raise LiveAuthorizationError("v32_jobs_api_identity_already_exists")
        if not (
            _is_project_run_dir(row_run_dir) or _is_project_title(row.get("title"))
        ):
            continue
        if not isinstance(name, str) or not _is_project_run_dir(row_run_dir):
            raise LiveAuthorizationError("v32_jobs_api_project_identity_invalid")
        row_status = _api_status(row.get("status"))
        current = backend.get_run(name)
        if current is None:
            if row_status in TERMINAL_API_STATUSES:
                continue
            raise LiveAuthorizationError("v32_jobs_api_history_live_drift")
        if current.get("run_dir") != row_run_dir:
            raise LiveAuthorizationError("v32_jobs_api_history_live_drift")
        current_status = _api_status(current.get("status"))
        if current_status in ACTIVE_API_STATUSES:
            active.append({**current, "_observed_api_run_id": name})
    return active


def _project_kubernetes(
    inventory: Mapping[str, Any], title: str, run_dir: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = inventory.get("items")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise LiveAuthorizationError("v32_kubernetes_inventory_invalid")
    serialized = json.dumps(inventory, sort_keys=True)
    if title in serialized or run_dir in serialized:
        raise LiveAuthorizationError("v32_kubernetes_identity_already_exists")
    project_objects = [
        row
        for row in rows
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
    gpu_pods = [
        row
        for row in rows
        if row.get("kind") == "Pod"
        and (row.get("status") or {}).get("phase") in {"Pending", "Running"}
        and _pod_gpu_requests(row) > 0
        and (
            any(_is_project_run_dir(value) for value in _pod_run_dirs(row))
            or (_metadata(row).get("labels") or {}).get(
                "cyber-post-train.fleet.ai/owner"
            )
            == "chris"
        )
    ]
    return project_objects, gpu_pods


def build_live_authorization(
    *,
    backend: Backend,
    payload: Mapping[str, Any],
    title: str,
    run_dir: str,
    control_result_path: str,
    request_sha256: str,
    priority_class: str,
    stale_run_reconciliation: Mapping[str, Any],
    now: float | None = None,
) -> dict[str, Any]:
    """Observe live state once and produce a zero-footprint authorization."""

    if control_result_path != CONTROL_RESULT_PATH:
        raise LiveAuthorizationError("v32_control_result_path_invalid")
    observed_at = time.time() if now is None else now
    rows, page_count = backend.list_runs()
    active = _active_project_runs(
        backend,
        rows,
        title,
        run_dir,
        stale_run_reconciliation,
        observed_at,
    )
    project_objects, project_gpu_pods = _project_kubernetes(
        backend.kubernetes_inventory(), title, run_dir
    )
    sfs = backend.sfs_observation(absent_paths=(run_dir, control_result_path))
    if (
        set(sfs)
        != {
            "observer_pod_name",
            "observer_pod_uid",
            "observer_sfs_mount_path",
            "absent_paths",
        }
        or sfs.get("observer_pod_name") != SFS_OBSERVER_POD_NAME
        or sfs.get("observer_pod_uid") != SFS_OBSERVER_POD_UID
        or sfs.get("observer_sfs_mount_path") != SFS_OBSERVER_MOUNT_PATH
        or not _nonzero_uuid(sfs.get("observer_pod_uid"))
        or sfs.get("absent_paths") != [run_dir, control_result_path]
    ):
        raise LiveAuthorizationError("v32_sfs_observation_invalid")
    if active or project_objects or project_gpu_pods:
        raise LiveAuthorizationError("v32_requires_zero_project_server")
    preview_status = backend.preview(payload)
    if preview_status != 200:
        raise LiveAuthorizationError("v32_jobs_api_preview_invalid")

    live: dict[str, Any] = {
        "schema_version": LIVE_SCHEMA,
        "status": "PASSED_SCORE_BLIND_LIVE_CREATE_OBSERVATION",
        "observed_at_epoch": observed_at,
        "mode": "ZERO_PROJECT_SERVER",
        "jobs_api_pages": page_count,
        "active_project_runs": [],
        "stale_run_reconciliation": dict(stale_run_reconciliation),
        "project_object_count": 0,
        "project_gpu_pod_count": 0,
        "sfs_observation": sfs,
        "preview_http_status": preview_status,
        "active_dedicated_nodes": 0,
        "active_dedicated_gpus": 0,
        "planned_nodes_after_create": 1,
        "planned_gpus_after_create": 8,
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
        "active_dedicated_nodes": 0,
        "active_dedicated_gpus": 0,
        "planned_nodes_after_create": 1,
        "planned_gpus_after_create": 8,
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
    sfs = value.get("sfs_observation")
    absent_paths = sfs.get("absent_paths") if isinstance(sfs, dict) else None
    reconciliation = value.get("stale_run_reconciliation")
    if (
        set(value) != LIVE_KEYS
        or value.get("schema_version") != LIVE_SCHEMA
        or value.get("status") != "PASSED_SCORE_BLIND_LIVE_CREATE_OBSERVATION"
        or value.get("receipt_sha256")
        != crypto.digest_without(dict(value), "receipt_sha256")
        or value.get("observed_at_epoch") != authorization.get("observed_at_epoch")
        or value.get("mode") != "ZERO_PROJECT_SERVER"
        or value.get("jobs_api_pages", 0) < 1
        or value.get("active_project_runs") != []
        or value.get("project_object_count") != 0
        or value.get("project_gpu_pod_count") != 0
        or value.get("preview_http_status") != 200
        or value.get("active_dedicated_nodes") != 0
        or value.get("active_dedicated_gpus") != 0
        or value.get("planned_nodes_after_create") != 1
        or value.get("planned_gpus_after_create") != 8
        or value.get("maximum_nodes") != MAX_NODES
        or value.get("maximum_gpus") != MAX_GPUS
        or value.get("api_mutation_calls") != 0
        or value.get("fleet_task_instance_calls") != 0
        or value.get("fleet_session_calls") != 0
        or value.get("verifier_calls") != 0
        or value.get("scoring_calls") != 0
        or value.get("protected_content_included") is not False
        or not isinstance(sfs, dict)
        or set(sfs)
        != {
            "observer_pod_name",
            "observer_pod_uid",
            "observer_sfs_mount_path",
            "absent_paths",
        }
        or sfs.get("observer_pod_name") != SFS_OBSERVER_POD_NAME
        or sfs.get("observer_pod_uid") != SFS_OBSERVER_POD_UID
        or sfs.get("observer_sfs_mount_path") != SFS_OBSERVER_MOUNT_PATH
        or not _nonzero_uuid(sfs.get("observer_pod_uid"))
        or not isinstance(absent_paths, list)
        or len(absent_paths) != 2
        or absent_paths[0] != authorization.get("server_run_dir")
        or absent_paths[1] != CONTROL_RESULT_PATH
        or any(
            value.get(field) != authorization.get(field)
            for field in (
                "active_dedicated_nodes",
                "active_dedicated_gpus",
                "planned_nodes_after_create",
                "planned_gpus_after_create",
            )
        )
    ):
        raise LiveAuthorizationError("v32_live_observation_invalid")
    if not isinstance(reconciliation, dict):
        raise LiveAuthorizationError("v32_live_observation_invalid")
    try:
        stale_runs.validate_reconciliation(
            reconciliation,
            rows=reconciliation.get("project_rows"),
            now=float(authorization["observed_at_epoch"]),
        )
    except (KeyError, TypeError, ValueError, stale_runs.ReconciliationError) as exc:
        raise LiveAuthorizationError("v32_live_observation_invalid") from exc


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
            status, page = self._request("GET", f"/v1/runs?limit=200&offset={offset}")
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
        result = subprocess.run(
            [
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
            ],
            capture_output=True,
            text=True,
            check=False,
        )
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
        if self._inventory is None:
            raise LiveAuthorizationError("v32_kubernetes_inventory_not_observed")
        candidates: list[tuple[str, str, str]] = []
        for pod in self._inventory.get("items") or []:
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
        exact = [
            row
            for row in sorted(set(candidates))
            if row
            == (
                SFS_OBSERVER_POD_NAME,
                SFS_OBSERVER_POD_UID,
                SFS_OBSERVER_MOUNT_PATH,
            )
        ]
        if len(exact) != 1:
            raise LiveAuthorizationError("v32_sfs_observer_absent")
        return exact[0]

    @staticmethod
    def _observed_path(mount: str, canonical: str) -> str:
        if not canonical.startswith("/mnt/sfs/") or not mount.startswith("/"):
            raise LiveAuthorizationError("v32_sfs_path_invalid")
        relative = canonical.removeprefix("/mnt/sfs/")
        if not relative or ".." in Path(relative).parts:
            raise LiveAuthorizationError("v32_sfs_path_invalid")
        return str(Path(mount) / relative)

    def sfs_observation(self, *, absent_paths: tuple[str, ...]) -> dict[str, Any]:
        name, uid, mount = self._observer()
        for path in absent_paths:
            result = subprocess.run(
                [
                    "kubectl",
                    "-n",
                    NAMESPACE,
                    "exec",
                    name,
                    "--",
                    "test",
                    "!",
                    "-e",
                    self._observed_path(mount, path),
                ],
                capture_output=True,
                check=False,
            )
            if result.returncode != 0:
                raise LiveAuthorizationError("v32_sfs_create_identity_exists")
        return {
            "observer_pod_name": name,
            "observer_pod_uid": uid,
            "observer_sfs_mount_path": mount,
            "absent_paths": list(absent_paths),
        }

    def preview(self, payload: Mapping[str, Any]) -> int:
        status, _ = self._request("POST", "/v1/runs/preview", payload)
        return status


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
    parser.add_argument("--stale-reconciliation", type=Path, required=True)
    args = parser.parse_args(argv)
    from evals.fleet import glm53_dedicated_v32_create_v1 as server

    try:
        stale_reconciliation = json.loads(args.stale_reconciliation.read_text())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise LiveAuthorizationError("v32_stale_run_reconciliation_invalid") from exc
    if not isinstance(stale_reconciliation, dict):
        raise LiveAuthorizationError("v32_stale_run_reconciliation_invalid")
    value = build_live_authorization(
        backend=SystemBackend(),
        payload=server.payload(),
        title=server.TITLE,
        run_dir=server.RUN_DIR,
        control_result_path=server.RESULT_PATH,
        request_sha256=server.request_sha256(),
        priority_class=server.payload()["priority_class"],
        stale_run_reconciliation=stale_reconciliation,
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
