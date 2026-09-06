"""Fresh-current successor guard for gap-free DP6-g qualification and release."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import yaml

from evals.fleet import glm53_dedicated_v8_live as shared
from evals.fleet import qwen38_dp6_g_live_release_v1 as server_live
from evals.fleet import qwen38_dp6_g_qualifier_package_v1 as qualifier
from evals.fleet import qwen38_dp6_g_qualifier_runtime_v1 as runtime
from evals.fleet import qwen38_dp6_g_scorefree_v1 as held
from evals.fleet import qwen38_dp6_qualification_guard_v2 as lifecycle_guard
from evals.fleet import self_hosted

POSTCREATE_RELEASE_SCHEMA = "fleet-qwen38-dp6-g-postcreate-release-v1"
BINDING_SCHEMA = runtime.BINDING_SCHEMA
QUALIFIER_RELEASE_SCHEMA = qualifier.RELEASE_SCHEMA
TERMINAL_SCHEMA = "fleet-qwen38-dp6-g-scorefree-terminal-v1"
POLL_SECONDS = 5
IDLE_SECONDS = 600
COUNTER_PRODUCER_TIMEOUT_SECONDS = (
    held.SERVER_PRE_READY_TIMEOUT_SECONDS + held.COUNTER_PRODUCER_STARTUP_GRACE_SECONDS
)
STARTUP_TIMEOUT_SECONDS = 600
DELETE_TIMEOUT_SECONDS = 180
CREATE_RECONCILE_TIMEOUT_SECONDS = 60


def _digest(value: Mapping[str, Any]) -> str:
    return self_hosted.digest_without(dict(value), "receipt_sha256")


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    shared._write_once(path, dict(value))  # noqa: SLF001 - create-once evidence helper


def _kubectl_json(*args: str) -> dict[str, Any]:
    result = subprocess.run(["kubectl", *args], check=True, capture_output=True, text=True)
    value = json.loads(result.stdout)
    if not isinstance(value, dict):
        raise RuntimeError("Kubernetes response is malformed")
    return value


def _uid(value: object, field: str) -> str:
    try:
        parsed = uuid.UUID(str(value))
    except (AttributeError, TypeError, ValueError) as exc:
        raise RuntimeError(f"{field} is not one UUID") from exc
    if parsed.int == 0:
        raise RuntimeError(f"{field} is zero")
    return str(parsed)


def _condition_true(value: Mapping[str, Any], kind: str) -> bool:
    return any(
        row.get("type") == kind and row.get("status") == "True"
        for row in value.get("status", {}).get("conditions") or []
        if isinstance(row, dict)
    )


def validate_release(
    value: Mapping[str, Any], server_release: Mapping[str, Any], source_commit: str
) -> None:
    expected = {
        "schema_version": POSTCREATE_RELEASE_SCHEMA,
        "status": "RELEASED_FOR_ONE_GAP_FREE_SCORE_FREE_DP6_G_SEQUENCE",
        "launch_authorized": True,
        "scoring_authorized": False,
        "source_commit": source_commit,
        "server_release_receipt_sha256": server_release.get("receipt_sha256"),
        "title": held.TITLE,
        "run_dir": held.RUN_DIR,
        "serving_block": held.SERVING_BLOCK,
        "qualifier_job": qualifier.JOB_NAME,
        "qualifier_output_root": qualifier.OUTPUT_ROOT,
        "server_create_limit": 1,
        "qualifier_create_limit": 1,
        "idle_release_seconds": IDLE_SECONDS,
        "gap_free_monitor_required": True,
        "statistical_cells_selected": 0,
        "scored_calls": 0,
        "prompts_traces_flags_or_scores_included": False,
    }
    if value.get("receipt_sha256") != _digest(value) or (
        {key: item for key, item in value.items() if key != "receipt_sha256"} != expected
    ):
        raise ValueError("DP6-g post-create release is not executable")


def _inventory() -> list[dict[str, Any]]:
    return list(
        _kubectl_json(
            "-n",
            shared.NAMESPACE,
            "get",
            "rayjobs.ray.io,rayclusters.ray.io,workloads.kueue.x-k8s.io,pods,services,"
            "jobs,configmaps",
            "-o",
            "json",
        ).get("items")
        or []
    )


def _owned(items: list[dict[str, Any]], kind: str, owner_uid: str) -> list[dict[str, Any]]:
    return [
        item
        for item in items
        if item.get("kind") == kind
        and any(
            owner.get("uid") == owner_uid
            for owner in item.get("metadata", {}).get("ownerReferences") or []
        )
    ]


def _one(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    if len(rows) != 1:
        raise RuntimeError(f"expected exactly one {label}; observed {len(rows)}")
    return rows[0]


def build_binding(
    api_value: Mapping[str, Any],
    submission: Mapping[str, Any],
    items: list[dict[str, Any]],
    root: Path,
) -> dict[str, Any]:
    api_run_id = submission["api_run_id"]
    if api_value.get("name") != api_run_id or api_value.get("run_dir") != held.RUN_DIR:
        raise RuntimeError("Jobs API run identity drifted")
    if str(api_value.get("status") or "").upper() != "RUNNING":
        raise RuntimeError("Jobs API server is not RUNNING")
    rayjob = _one(
        [
            item
            for item in items
            if item.get("kind") == "RayJob" and item.get("metadata", {}).get("name") == api_run_id
        ],
        "exact RayJob",
    )
    rayjob_uid = _uid(rayjob.get("metadata", {}).get("uid"), "RayJob UID")
    if str(rayjob.get("status", {}).get("jobStatus") or "").upper() != "RUNNING":
        raise RuntimeError("exact RayJob is not RUNNING")
    workload = _one(_owned(items, "Workload", rayjob_uid), "RayJob-owned Workload")
    workload_uid = _uid(workload.get("metadata", {}).get("uid"), "Workload UID")
    conditions = json.dumps(workload.get("status", {}).get("conditions") or [], sort_keys=True)
    if (
        not _condition_true(workload, "Admitted")
        or not _condition_true(workload, "QuotaReserved")
        or _condition_true(workload, "Evicted")
        or "preempt" in conditions.lower()
        or "noreservation" in conditions.lower()
    ):
        raise RuntimeError("Workload admission is not stable and nonpreempted")
    cluster_name = rayjob.get("status", {}).get("rayClusterName")
    cluster = _one(
        [
            item
            for item in items
            if item.get("kind") == "RayCluster"
            and item.get("metadata", {}).get("name") == cluster_name
            and any(
                owner.get("uid") == rayjob_uid
                for owner in item.get("metadata", {}).get("ownerReferences") or []
            )
        ],
        "RayJob-owned exact RayCluster",
    )
    cluster_uid = _uid(cluster.get("metadata", {}).get("uid"), "RayCluster UID")
    service_name = f"{cluster_name}-head-svc"
    service = _one(
        [
            item
            for item in _owned(items, "Service", cluster_uid)
            if item.get("metadata", {}).get("name") == service_name
        ],
        "RayCluster-owned head Service",
    )
    service_uid = _uid(service.get("metadata", {}).get("uid"), "Service UID")
    pods = [
        item
        for item in _owned(items, "Pod", cluster_uid)
        if str(item.get("metadata", {}).get("name") or "").startswith(f"{cluster_name}-head-")
        and item.get("metadata", {}).get("name") != service_name
    ]
    pod = _one(pods, "RayCluster-owned head Pod")
    statuses = pod.get("status", {}).get("containerStatuses") or []
    if (
        pod.get("status", {}).get("phase") != "Running"
        or not statuses
        or not all(row.get("ready") is True for row in statuses)
        or sum(int(row.get("restartCount") or 0) for row in statuses) != 0
    ):
        raise RuntimeError("head Pod is not Running/Ready/restart0")
    binding = {
        "schema_version": BINDING_SCHEMA,
        "status": "READY_NON_SCORED",
        "submission_receipt_sha256": submission["receipt_sha256"],
        "api_run_id": api_run_id,
        "title": held.TITLE,
        "run_dir": held.RUN_DIR,
        "serving_block": held.SERVING_BLOCK,
        "ray_cluster_name": cluster_name,
        "ray_cluster_uid": cluster_uid,
        "service_name": service_name,
        "service_origin": (f"http://{service_name}.{shared.NAMESPACE}.svc.cluster.local:8000"),
        "rayjob_uid": rayjob_uid,
        "workload_uid": workload_uid,
        "head_pod_name": pod["metadata"]["name"],
        "head_pod_uid": _uid(pod["metadata"].get("uid"), "head Pod UID"),
        "service_uid": service_uid,
        "image": held.runtime.IMAGE,
        "model_revision": held.runtime.MODEL_REVISION,
        "context_length": 262144,
        "tensor_parallel_size": 1,
        "data_parallel_size": 6,
        "head_pod_running_ready": True,
        "head_pod_restarts": 0,
        "kueue_preempted": False,
        "scoring_authorized": False,
        "prompts_traces_flags_or_scores_included": False,
    }
    binding["receipt_sha256"] = _digest(binding)
    runtime.validate_binding(binding, submission, root)
    return binding


def _stable_sfs_observer(excluded_pod_uid: str) -> tuple[str, str, str]:
    return lifecycle_guard.select_stable_sfs_observer(_inventory(), excluded_pod_uid)


def _validated_observer(
    observer: tuple[str, str, str],
) -> tuple[str, str, str]:
    observer_name, observer_uid, mount_path = observer
    current = _kubectl_json("-n", shared.NAMESPACE, "get", "pod", observer_name, "-o", "json")
    if lifecycle_guard.select_stable_sfs_observer(
        [current], excluded_pod_uid=""
    ) != (observer_name, observer_uid, mount_path):
        raise RuntimeError("bound SFS observer drifted")
    return observer


def _bound_sfs_json(observer: tuple[str, str, str], path: str) -> dict[str, Any]:
    observer_name, _, mount_path = _validated_observer(observer)
    observed_path = lifecycle_guard.observer_sfs_path(mount_path, path)
    result = subprocess.run(
        [
            "kubectl",
            "-n",
            shared.NAMESPACE,
            "exec",
            observer_name,
            "--",
            "cat",
            observed_path,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    value = json.loads(result.stdout)
    if not isinstance(value, dict):
        raise RuntimeError("sanitized SFS receipt is malformed")
    return value


def _counter_producer_gate(binding: Mapping[str, Any]) -> dict[str, Any]:
    observer = _stable_sfs_observer(str(binding["head_pod_uid"]))
    lifecycle = f"{held.RUN_DIR}/lifecycle"
    deadline = time.monotonic() + COUNTER_PRODUCER_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        try:
            baseline = _bound_sfs_json(observer, f"{lifecycle}/REQUEST-COUNTER-BASELINE.json")
            state = _bound_sfs_json(observer, f"{lifecycle}/.request-counters.json")
            status = _bound_sfs_json(observer, f"{lifecycle}/OBSERVER-STATUS.json")
            value = lifecycle_guard.validate_counter_producer_ready(
                baseline, state, status, binding, held.RUN_DIR
            )
            value = {
                **value,
                "observer_pod_name": observer[0],
                "observer_pod_uid": observer[1],
                "observer_sfs_mount_path": observer[2],
            }
            value["receipt_sha256"] = _digest(value)
            return value
        except (json.JSONDecodeError, subprocess.CalledProcessError, RuntimeError, ValueError):
            time.sleep(POLL_SECONDS)
    raise RuntimeError("bound request-counter producer did not become ready")


def qualifier_release(
    submission: Mapping[str, Any], binding: Mapping[str, Any], root: Path
) -> dict[str, Any]:
    items = _inventory()
    exact = [
        item
        for item in items
        if item.get("metadata", {}).get("name") in {qualifier.JOB_NAME, qualifier.CONFIGMAP_NAME}
    ]
    if exact:
        raise RuntimeError("DP6-g qualifier Kubernetes identity already exists")
    observer, _, mount_path = server_live._observer_pod()  # noqa: SLF001
    observed_output_root = lifecycle_guard.observer_sfs_path(
        mount_path, qualifier.OUTPUT_ROOT
    )
    result = subprocess.run(
        [
            "kubectl",
            "-n",
            shared.NAMESPACE,
            "exec",
            observer,
            "--",
            "test",
            "!",
            "-e",
            observed_output_root,
        ],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError("DP6-g qualifier output identity already exists")
    value = {
        "schema_version": QUALIFIER_RELEASE_SCHEMA,
        "status": "RELEASED_FOR_ONE_NON_SCORED_QUALIFIER",
        "launch_authorized": True,
        "scoring_authorized": False,
        "job_name": qualifier.JOB_NAME,
        "configmap_name": qualifier.CONFIGMAP_NAME,
        "output_root": qualifier.OUTPUT_ROOT,
        "serving_block": held.SERVING_BLOCK,
        "submission_receipt_sha256": submission["receipt_sha256"],
        "server_binding_receipt_sha256": binding["receipt_sha256"],
        "package_sha256": qualifier.package_sha256(root),
        "harness_runtime_image": qualifier.staged_image.identity(),
        "fresh_job_matches": 0,
        "fresh_configmap_matches": 0,
        "fresh_output_root_exists": False,
        "server_running_ready_restart0": True,
        "api_mutations_before_create": 0,
        "task_instance_session_verifier_scoring_calls": 0,
        "prompts_traces_flags_or_scores_included": False,
    }
    value["receipt_sha256"] = _digest(value)
    qualifier.validate_release(value, submission, binding, root)
    return value


def _sfs_json(path: str, observer: tuple[str, str, str] | None = None) -> dict[str, Any]:
    if observer is not None:
        return _bound_sfs_json(observer, path)
    observer_name, _, mount_path = _validated_observer(
        server_live._observer_pod()  # noqa: SLF001
    )
    observed_path = lifecycle_guard.observer_sfs_path(mount_path, path)
    result = subprocess.run(
        [
            "kubectl",
            "-n",
            shared.NAMESPACE,
            "exec",
            observer_name,
            "--",
            "cat",
            observed_path,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    value = json.loads(result.stdout)
    if not isinstance(value, dict):
        raise RuntimeError("sanitized SFS receipt is malformed")
    return value


def _sfs_exists(
    path: str, observer: tuple[str, str, str] | None = None
) -> bool:
    if observer is None:
        observer = server_live._observer_pod()  # noqa: SLF001
    observer_name, _, mount_path = _validated_observer(observer)
    observed_path = lifecycle_guard.observer_sfs_path(mount_path, path)
    return (
        subprocess.run(
            [
                "kubectl",
                "-n",
                shared.NAMESPACE,
                "exec",
                observer_name,
                "--",
                "test",
                "-e",
                observed_path,
            ],
            check=False,
            capture_output=True,
        ).returncode
        == 0
    )


def create_qualifier(
    submission: Mapping[str, Any],
    binding: Mapping[str, Any],
    release: Mapping[str, Any],
    root: Path,
    created_uids: dict[str, str],
) -> tuple[str, str]:
    rendered = qualifier.render(root, submission, binding, release)
    created = []
    for manifest in (rendered["configmap"], rendered["job"]):
        completed = subprocess.run(
            ["kubectl", "-n", shared.NAMESPACE, "create", "-f", "-", "-o", "json"],
            input=yaml.safe_dump(manifest, sort_keys=False),
            check=True,
            capture_output=True,
            text=True,
        )
        value = json.loads(completed.stdout)
        if not isinstance(value, dict) or value.get("kind") != manifest["kind"]:
            raise RuntimeError("qualifier create response drifted")
        created_uids[manifest["kind"]] = _uid(
            value.get("metadata", {}).get("uid"), f"qualifier {manifest['kind']} UID"
        )
        created.append(value)
    by_kind = {row["kind"]: row for row in created}
    return (
        _uid(by_kind["ConfigMap"]["metadata"].get("uid"), "qualifier ConfigMap UID"),
        _uid(by_kind["Job"]["metadata"].get("uid"), "qualifier Job UID"),
    )


def _release_server(
    client: httpx.Client, api_run_id: str, binding: Mapping[str, Any] | None
) -> dict[str, Any]:
    # Binding can fail before SERVER-BINDING.json exists.  Snapshot both the
    # exact owner closure and the Jobs-API-generated suffixed names before the
    # delete so cleanup remains observable even in that state.
    before_items = _inventory()
    tracked_uids = {
        str(binding[field])
        for field in (
            "rayjob_uid",
            "workload_uid",
            "ray_cluster_uid",
            "head_pod_uid",
            "service_uid",
        )
        if binding is not None and binding.get(field)
    }
    tracked_names = {api_run_id}
    changed = True
    while changed:
        changed = False
        for item in before_items:
            metadata = item.get("metadata", {})
            name = str(metadata.get("name") or "")
            uid = str(metadata.get("uid") or "")
            owners = {
                str(owner.get("uid") or "")
                for owner in metadata.get("ownerReferences") or []
                if isinstance(owner, dict)
            }
            generated_name = name == api_run_id or name.startswith(f"{api_run_id}-")
            if generated_name or owners.intersection(tracked_uids):
                if name and name not in tracked_names:
                    tracked_names.add(name)
                    changed = True
                if uid and uid not in tracked_uids:
                    tracked_uids.add(uid)
                    changed = True
    before = client.get(f"/v1/runs/{api_run_id}")
    if before.status_code == 404:
        delete_status: int | str = "already_absent"
    else:
        before.raise_for_status()
        deleted = client.delete(f"/v1/runs/{api_run_id}")
        if deleted.status_code != 204:
            raise RuntimeError("Jobs API release did not return HTTP 204")
        delete_status = 204
    deadline = time.monotonic() + DELETE_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        after = client.get(f"/v1/runs/{api_run_id}")
        if after.status_code == 404:
            items = _inventory()
            remaining = [
                item
                for item in items
                if str(item.get("metadata", {}).get("name") or "") in tracked_names
                or str(item.get("metadata", {}).get("name") or "").startswith(
                    f"{api_run_id}-"
                )
                or str(item.get("metadata", {}).get("uid") or "") in tracked_uids
            ]
            if not remaining:
                return {
                    "get_before": before.status_code,
                    "delete": delete_status,
                    "get_after": 404,
                    "rayjob_workload_raycluster_pod_service_absent": True,
                }
        time.sleep(POLL_SECONDS)
    raise RuntimeError("Jobs API run or UID-bound Kubernetes objects remained after release")


def _recover_created_api_run(client: httpx.Client) -> str | None:
    """Recover an uncertain successful POST from the exact target identity."""
    deadline = time.monotonic() + CREATE_RECONCILE_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        matches: list[str] = []
        for row in shared._runs(client):  # noqa: SLF001 - paginated GET authority
            title, run_dir, name = row.get("title"), row.get("run_dir"), row.get("name")
            if title != held.TITLE and run_dir != held.RUN_DIR:
                continue
            if title != held.TITLE or run_dir != held.RUN_DIR:
                raise RuntimeError("uncertain DP6-g create identity is contradictory")
            if not isinstance(name, str) or not name.startswith("ft-run-"):
                raise RuntimeError("uncertain DP6-g create omitted its run identity")
            matches.append(name)
        if len(matches) > 1:
            raise RuntimeError("ambiguous created DP6-g server during cleanup")
        if matches:
            return matches[0]
        time.sleep(POLL_SECONDS)
    return None


def _delete_qualifier_object(kind: str, name: str, expected_uid: str | None) -> dict[str, Any]:
    current = subprocess.run(
        ["kubectl", "-n", shared.NAMESPACE, "get", kind, name, "-o", "json"],
        check=False,
        capture_output=True,
        text=True,
    )
    if current.returncode != 0:
        return {"known": expected_uid is not None, "uid": expected_uid, "delete": "already_absent"}
    value = json.loads(current.stdout)
    observed_uid = _uid(value.get("metadata", {}).get("uid"), f"qualifier {kind} UID")
    if expected_uid is not None and observed_uid != expected_uid:
        raise RuntimeError(f"refusing to stop qualifier after {kind} UID drift")
    subprocess.run(
        ["kubectl", "-n", shared.NAMESPACE, "delete", kind, name, "--wait=true"],
        check=True,
        capture_output=True,
        text=True,
    )
    absent = subprocess.run(
        ["kubectl", "-n", shared.NAMESPACE, "get", kind, name, "-o", "name"],
        check=False,
        capture_output=True,
        text=True,
    )
    if absent.returncode == 0:
        raise RuntimeError(f"qualifier {kind} remained after delete")
    return {"known": True, "uid": observed_uid, "delete": "completed"}


def _stop_qualifier(job_uid: str | None, configmap_uid: str | None) -> dict[str, Any]:
    # Recover exact-name objects when create succeeded but response handling
    # failed before the caller could retain a UID.  The release gate proved
    # both names absent immediately before their create.
    job: dict[str, Any] | None = None
    configmap: dict[str, Any] | None = None
    errors: list[BaseException] = []
    try:
        job = _delete_qualifier_object("job", qualifier.JOB_NAME, job_uid)
    except BaseException as exc:
        errors.append(exc)
    try:
        configmap = _delete_qualifier_object(
            "configmap", qualifier.CONFIGMAP_NAME, configmap_uid
        )
    except BaseException as exc:
        errors.append(exc)
    if errors:
        raise RuntimeError("one or more qualifier objects failed cleanup") from errors[0]
    return {
        "job": job,
        "configmap": configmap,
        "job_and_configmap_absent": True,
    }


def _failure_cleanup(
    client: httpx.Client,
    api_run_id: str | None,
    binding: Mapping[str, Any] | None,
    qualifier_uids: Mapping[str, str],
    *,
    server_create_started: bool,
    qualifier_create_started: bool,
) -> tuple[str | None, dict[str, Any], dict[str, Any], dict[str, str | None]]:
    errors: dict[str, str | None] = {
        "create_reconciliation": None,
        "server": None,
        "qualifier": None,
    }
    if api_run_id is None and server_create_started:
        try:
            api_run_id = _recover_created_api_run(client)
        except BaseException as exc:
            errors["create_reconciliation"] = type(exc).__name__
    server_release: dict[str, Any] = {
        "get_before": "not_created",
        "delete": "not_created",
        "get_after": 404,
        "rayjob_workload_raycluster_pod_service_absent": True,
    }
    if api_run_id is not None:
        try:
            server_release = _release_server(client, api_run_id, binding)
        except BaseException as exc:
            errors["server"] = type(exc).__name__
    qualifier_stop: dict[str, Any] = {
        "job": {"known": False, "uid": None, "delete": "not_created"},
        "configmap": {"known": False, "uid": None, "delete": "not_created"},
        "job_and_configmap_absent": True,
    }
    if qualifier_create_started:
        try:
            qualifier_stop = _stop_qualifier(
                qualifier_uids.get("Job"), qualifier_uids.get("ConfigMap")
            )
        except BaseException as exc:
            errors["qualifier"] = type(exc).__name__
            qualifier_stop = {"cleanup_error_type": type(exc).__name__}
    return api_run_id, server_release, qualifier_stop, errors


def monitor_and_release(
    client: httpx.Client,
    api_run_id: str,
    submission: Mapping[str, Any],
    binding: Mapping[str, Any],
    qualifier_configmap_uid: str,
    qualifier_job_uid: str,
    root: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    terminal_reason = ""
    qualifier_phase = ""
    while True:
        job = _kubectl_json("-n", shared.NAMESPACE, "get", "job", qualifier.JOB_NAME, "-o", "json")
        if _uid(job.get("metadata", {}).get("uid"), "qualifier Job UID") != qualifier_job_uid:
            raise RuntimeError("qualifier Job UID drifted")
        if int(job.get("status", {}).get("succeeded") or 0) == 1:
            terminal_reason, qualifier_phase = "qualifier_succeeded", "Succeeded"
            break
        if int(job.get("status", {}).get("failed") or 0) > 0:
            terminal_reason, qualifier_phase = "qualifier_failed", "Failed"
            break
        if _sfs_exists(f"{held.RUN_DIR}/lifecycle/IDLE-TIMEOUT"):
            terminal_reason, qualifier_phase = "server_idle_timeout", "Running"
            break
        api = client.get(f"/v1/runs/{api_run_id}")
        if api.status_code != 200 or str(api.json().get("status") or "").upper() != "RUNNING":
            terminal_reason, qualifier_phase = "server_not_running", "Running"
            break
        if time.monotonic() - started > runtime.parity.TIMEOUT_SECONDS * sum(runtime.LEVELS):
            terminal_reason, qualifier_phase = "qualification_wall_timeout", "Running"
            break
        time.sleep(POLL_SECONDS)
    terminal_observer = _stable_sfs_observer(str(binding["head_pod_uid"]))
    released = _release_server(client, api_run_id, binding)
    qualifier_receipt: dict[str, Any] | None = None
    if qualifier_phase == "Succeeded":
        qualifier_receipt = _sfs_json(
            f"{qualifier.OUTPUT_ROOT}/RESULT.json", terminal_observer
        )
        reduced = runtime.validate_binding(binding, submission, root)
        plan = runtime.qualification_plan(reduced, str(binding["service_origin"]), root)
        runtime.validate_result(qualifier_receipt, plan, root)
    elif qualifier_phase == "Failed" and _sfs_exists(
        f"{qualifier.OUTPUT_ROOT}/FAILED.json", terminal_observer
    ):
        qualifier_receipt = _sfs_json(
            f"{qualifier.OUTPUT_ROOT}/FAILED.json", terminal_observer
        )
        if qualifier_receipt.get("receipt_sha256") != _digest(qualifier_receipt) or (
            qualifier_receipt.get("schema_version") != runtime.FAILURE_SCHEMA
            or qualifier_receipt.get("scoring_calls") != 0
            or qualifier_receipt.get("prompts_traces_flags_or_scores_included") is not False
        ):
            raise RuntimeError("qualifier failure receipt drifted")
    qualifier_stop = _stop_qualifier(qualifier_job_uid, qualifier_configmap_uid)
    value = {
        "schema_version": TERMINAL_SCHEMA,
        "status": "RELEASED_AFTER_SCORE_FREE_QUALIFICATION_BOUNDARY",
        "observed_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "api_run_id": api_run_id,
        "rayjob_uid": binding["rayjob_uid"],
        "workload_uid": binding["workload_uid"],
        "ray_cluster_uid": binding["ray_cluster_uid"],
        "head_pod_uid": binding["head_pod_uid"],
        "service_uid": binding["service_uid"],
        "qualifier_job_uid": qualifier_job_uid,
        "qualifier_phase": qualifier_phase,
        "terminal_reason": terminal_reason,
        "jobs_api_release": released,
        "qualifier_stop": qualifier_stop,
        "terminal_sfs_observer": {
            "pod_name": terminal_observer[0],
            "pod_uid": terminal_observer[1],
            "sfs_mount_path": terminal_observer[2],
            "excluded_target_head_pod_uid": binding["head_pod_uid"],
        },
        "qualifier_receipt_status": (
            qualifier_receipt.get("status") if qualifier_receipt is not None else None
        ),
        "qualifier_receipt_sha256": (
            qualifier_receipt.get("receipt_sha256") if qualifier_receipt is not None else None
        ),
        "scored_calls": 0,
        "prompts_traces_flags_or_scores_included": False,
    }
    value["receipt_sha256"] = _digest(value)
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-release", type=Path, required=True)
    parser.add_argument("--postcreate-release", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path.cwd().resolve()
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    if subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout:
        raise RuntimeError("gap-free DP6-g execution requires a clean exact source commit")
    server_release = json.loads(args.server_release.read_text())
    postcreate_release = json.loads(args.postcreate_release.read_text())
    server_live.validate_server_release(server_release, root, source_commit)
    validate_release(postcreate_release, server_release, source_commit)
    args.evidence_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    with httpx.Client(
        base_url=shared.BASE_URL,
        headers={"Authorization": f"Bearer {shared._token()}", "Accept": "application/json"},  # noqa: SLF001
        timeout=60,
    ) as client:
        api_run_id: str | None = None
        binding: dict[str, Any] | None = None
        qualifier_uids: dict[str, str] = {}
        server_create_started = False
        qualifier_create_started = False
        stage = "live_gate"
        try:
            payload, gate = server_live.live_gate(client, server_release, root, source_commit)
            lifecycle_guard.validate_observer_environment(
                payload.get("env") or {}, held.RUNTIME_DEPENDENCY_DIR
            )
            stage = "create_server"
            server_create_started = True
            api_run_id = server_live.submit_create_once(
                client, payload, gate, server_release, source_commit, root
            )
            stage = "write_submission"
            submission = {
                "schema_version": server_live.SUBMISSION_SCHEMA,
                "status": "SUBMITTED_SCORE_FREE_DP6_G_SERVER",
                "api_run_id": api_run_id,
                "source_commit": source_commit,
                "title": held.TITLE,
                "run_dir": held.RUN_DIR,
                "serving_block": held.SERVING_BLOCK,
                "config_sha256": held.config(root)["config_sha256"],
                "server_release_receipt_sha256": server_release["receipt_sha256"],
                "live_gate_receipt_sha256": gate["receipt_sha256"],
                "live_gate": gate,
                "request_sha256": held.config(root)["request_sha256"],
                "project_resource_shape": gate["project_resource_shape"],
                "route": "POST /v1/runs",
                "http_status": 202,
                "server_instances_created": 1,
                "scored_calls": 0,
                "prompts_traces_flags_or_scores_included": False,
            }
            submission["receipt_sha256"] = _digest(submission)
            server_live.validate_submission(submission, server_release, root, source_commit)
            _write_once(args.evidence_dir / "SUBMISSION.json", submission)
            stage = "bind_server"
            deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
            while time.monotonic() < deadline:
                response = client.get(f"/v1/runs/{api_run_id}")
                if response.status_code == 200:
                    try:
                        binding = build_binding(response.json(), submission, _inventory(), root)
                        break
                    except RuntimeError:
                        pass
                time.sleep(POLL_SECONDS)
            if binding is None:
                raise RuntimeError("binding timeout")
            _write_once(args.evidence_dir / "SERVER-BINDING.json", binding)
            stage = "counter_producer_ready"
            counter_producer = _counter_producer_gate(binding)
            _write_once(args.evidence_dir / "COUNTER-PRODUCER-GATE.json", counter_producer)
            stage = "release_qualifier"
            release = qualifier_release(submission, binding, root)
            _write_once(args.evidence_dir / "QUALIFIER-RELEASE.json", release)
            stage = "create_qualifier"
            qualifier_create_started = True
            configmap_uid, job_uid = create_qualifier(
                submission, binding, release, root, qualifier_uids
            )
            stage = "monitor_qualifier_and_server"
            terminal = monitor_and_release(
                client,
                api_run_id,
                submission,
                binding,
                configmap_uid,
                job_uid,
                root,
            )
            terminal["qualifier_configmap_uid"] = configmap_uid
            terminal["receipt_sha256"] = _digest(terminal)
            _write_once(args.evidence_dir / "TERMINAL.json", terminal)
        except BaseException as exc:
            api_run_id, server_release_result, qualifier_stop, cleanup_errors = (
                _failure_cleanup(
                    client,
                    api_run_id,
                    binding,
                    qualifier_uids,
                    server_create_started=server_create_started,
                    qualifier_create_started=qualifier_create_started,
                )
            )
            cleanup_complete = all(value is None for value in cleanup_errors.values())
            terminal = {
                "schema_version": TERMINAL_SCHEMA,
                "status": (
                    "RELEASED_AFTER_SCORE_FREE_QUALIFICATION_BOUNDARY"
                    if cleanup_complete
                    else "CLEANUP_INCOMPLETE"
                ),
                "observed_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "api_run_id": api_run_id,
                "terminal_reason": "orchestration_failure",
                "failure_stage": stage,
                "error_type": type(exc).__name__,
                "jobs_api_release": server_release_result,
                "qualifier_stop": qualifier_stop,
                "server_cleanup_error_type": cleanup_errors["server"],
                "qualifier_cleanup_error_type": cleanup_errors["qualifier"],
                "create_reconciliation_error_type": cleanup_errors[
                    "create_reconciliation"
                ],
                "scored_calls": 0,
                "prompts_traces_flags_or_scores_included": False,
            }
            terminal["receipt_sha256"] = _digest(terminal)
            _write_once(args.evidence_dir / "TERMINAL.json", terminal)
            return 1
    return 0 if terminal["qualifier_phase"] == "Succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
