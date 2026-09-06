"""Create-once live gate for a reviewed DP6-l score-free server release."""

from __future__ import annotations

import argparse
import json
import subprocess
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from evals.fleet import glm53_dedicated_v8_live as shared
from evals.fleet import qwen38_dp6_l_scorefree_v1 as held
from evals.fleet import qwen38_dp6_qualification_guard_v3 as lifecycle_guard
from evals.fleet import self_hosted

SERVER_RELEASE_SCHEMA = "fleet-qwen38-dp6-l-server-release-v1"
LIVE_GATE_SCHEMA = "fleet-qwen38-dp6-l-live-gate-v1"
SUBMISSION_SCHEMA = "fleet-qwen38-dp6-l-submission-v1"
ACTIVE_STATUSES = {"SUBMITTED", "SUSPENDED", "RUNNING"}
PROJECT_SHAPE_KEYS = {
    "current_gpu_nodes",
    "current_gpus",
    "projected_gpu_nodes",
    "projected_gpus",
    "maximum_gpu_nodes",
    "maximum_gpus",
    "active_project_rayjobs",
    "active_project_gpu_pods",
    "capacity",
}
CAPACITY_KEYS = {
    "eligible_six_gpu_node_count",
    "eligible_node_uids",
    "b300_nominal_gpu_quota",
    "b300_used_gpu_quota",
    "b300_gpu_quota_headroom",
    "local_queue_uid",
    "cluster_queue_uid",
    "priority_class",
    "peer_preemption_required",
}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def _global(*args: str) -> dict[str, Any]:
    result = subprocess.run(["kubectl", *args], check=True, capture_output=True, text=True)
    value = json.loads(result.stdout)
    if not isinstance(value, dict):
        raise RuntimeError("Kubernetes response is malformed")
    return value


def _condition_true(value: Mapping[str, Any], kind: str) -> bool:
    return any(
        row.get("type") == kind and row.get("status") == "True"
        for row in value.get("status", {}).get("conditions") or []
        if isinstance(row, dict)
    )


def _nonzero_uuid(value: object) -> bool:
    try:
        return uuid.UUID(str(value)).int != 0
    except (AttributeError, TypeError, ValueError):
        return False


def _pod_gpu_requests(pod: Mapping[str, Any]) -> int:
    try:
        return sum(
            int((container.get("resources", {}).get("requests") or {}).get("nvidia.com/gpu", 0))
            for container in pod.get("spec", {}).get("containers") or []
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError("GPU request shape drifted") from exc


def _pod_run_dir(pod: Mapping[str, Any]) -> str | None:
    values = [
        env.get("value")
        for container in pod.get("spec", {}).get("containers") or []
        for env in container.get("env") or []
        if env.get("name") == "RUN_DIR"
    ]
    return values[0] if len(values) == 1 and isinstance(values[0], str) else None


def validate_server_release(value: Mapping[str, Any], root: Path, source_commit: str) -> None:
    config, plan, preview, inventory, held_release = held.load_all(root)
    if value.get("receipt_sha256") != self_hosted.digest_without(dict(value), "receipt_sha256") or (
        value.get("schema_version") != SERVER_RELEASE_SCHEMA
        or value.get("status") != "RELEASED_FOR_ONE_SCORE_FREE_DP6_L_SERVER"
        or value.get("launch_authorized") is not True
        or value.get("scoring_authorized") is not False
        or value.get("source_commit") != source_commit
        or value.get("title") != held.TITLE
        or value.get("run_dir") != held.RUN_DIR
        or value.get("serving_block") != held.SERVING_BLOCK
        or value.get("config_sha256") != config["config_sha256"]
        or value.get("plan_receipt_sha256") != plan["receipt_sha256"]
        or value.get("preview_receipt_sha256") != preview["receipt_sha256"]
        or value.get("review_inventory_receipt_sha256") != inventory["receipt_sha256"]
        or value.get("held_release_receipt_sha256") != held_release["receipt_sha256"]
        or value.get("release_first_reconciled") is not True
        or value.get("server_create_limit") != 1
        or value.get("qualifier_create_limit") != 0
        or value.get("statistical_cells_selected") != 0
        or value.get("scored_calls") != 0
        or value.get("prompts_traces_flags_or_scores_included") is not False
    ):
        raise ValueError("DP6-l server release is not executable")


def _active_project_runs(client: httpx.Client) -> list[dict[str, str]]:
    active: list[dict[str, str]] = []
    for row in shared._runs(client):  # noqa: SLF001 - paginated GET authority
        name, run_dir = row.get("name"), row.get("run_dir")
        if not isinstance(name, str) or not isinstance(run_dir, str):
            continue
        if row.get("title") == held.TITLE or run_dir == held.RUN_DIR:
            raise RuntimeError("DP6-l Jobs API identity already exists")
        if "/mnt/sfs/jobs/chris-cyber-evalserve-" not in run_dir:
            continue
        current = client.get(f"/v1/runs/{name}")
        if current.status_code == 404:
            continue
        current.raise_for_status()
        value = current.json()
        if value.get("run_dir") != run_dir:
            raise RuntimeError("Jobs API history/live identity drifted")
        if str(value.get("status") or "").upper() in ACTIVE_STATUSES:
            active.append({"api_run_id": name, "run_dir": run_dir, "status": value["status"]})
    return active


def _priority_gate() -> dict[str, Any]:
    value = _global("get", "priorityclass", held.SERVER_PRIORITY_CLASS, "-o", "json")
    if value.get("value") != held.SERVER_PRIORITY_VALUE or value.get("preemptionPolicy") != "Never":
        raise RuntimeError("DP6-l PriorityClass drifted")
    return {
        "name": held.SERVER_PRIORITY_CLASS,
        "value": held.SERVER_PRIORITY_VALUE,
        "preemption_policy": "Never",
    }


def _resource_total(rows: list[dict[str, Any]], name: str) -> int:
    for row in rows:
        if row.get("name") == name:
            return int(row.get("total") or row.get("nominalQuota") or 0)
    raise RuntimeError(f"Kueue resource is absent: {name}")


def _capacity_gate() -> dict[str, Any]:
    nodes = (
        _global("get", "nodes", "-l", "workload=fleetai-training-ng-gpu", "-o", "json").get("items")
        or []
    )
    pods = _global("get", "pods", "--all-namespaces", "-o", "json").get("items") or []
    used: dict[str, int] = {}
    for pod in pods:
        if pod.get("status", {}).get("phase") not in {"Pending", "Running"}:
            continue
        node = pod.get("spec", {}).get("nodeName")
        if isinstance(node, str):
            used[node] = used.get(node, 0) + _pod_gpu_requests(pod)
    candidates = []
    for node in nodes:
        name = node.get("metadata", {}).get("name")
        labels = node.get("metadata", {}).get("labels") or {}
        allocatable = int(node.get("status", {}).get("allocatable", {}).get("nvidia.com/gpu", 0))
        free = allocatable - used.get(str(name), 0)
        if (
            isinstance(name, str)
            and _condition_true(node, "Ready")
            and node.get("spec", {}).get("unschedulable") is not True
            and allocatable == 8
            and free >= 6
            and labels.get("topology.nebius.com/tier-1")
            and "B300" in str(labels.get("nvidia.com/gpu.product") or "")
        ):
            candidates.append({"node_uid": node.get("metadata", {}).get("uid"), "free_gpus": free})
    if not candidates:
        raise RuntimeError("no schedulable six-GPU B300 fit exists")
    local = _global("get", "localqueue", "training-lq", "-n", shared.NAMESPACE, "-o", "json")
    queue = _global("get", "clusterqueue", "training-cq", "-o", "json")
    if (
        local.get("spec", {}).get("clusterQueue") != "training-cq"
        or not _condition_true(local, "Active")
        or not _condition_true(queue, "Active")
    ):
        raise RuntimeError("training queue is not active")
    flavor = next(
        row
        for group in queue.get("spec", {}).get("resourceGroups") or []
        for row in group.get("flavors") or []
        if row.get("name") == "b300-training"
    )
    usage = next(
        row
        for row in queue.get("status", {}).get("flavorsUsage") or []
        if row.get("name") == "b300-training"
    )
    nominal = _resource_total(flavor.get("resources") or [], "nvidia.com/gpu")
    used_quota = _resource_total(usage.get("resources") or [], "nvidia.com/gpu")
    if nominal - used_quota < 6:
        raise RuntimeError("training queue lacks six-GPU quota headroom")
    preemption = queue.get("spec", {}).get("preemption") or {}
    if preemption.get("reclaimWithinCohort") != "Never" or (
        (preemption.get("borrowWithinCohort") or {}).get("policy") != "Never"
    ):
        raise RuntimeError("training queue cohort preemption drifted")
    return {
        "eligible_six_gpu_node_count": len(candidates),
        "eligible_node_uids": sorted(str(row["node_uid"]) for row in candidates),
        "b300_nominal_gpu_quota": nominal,
        "b300_used_gpu_quota": used_quota,
        "b300_gpu_quota_headroom": nominal - used_quota,
        "local_queue_uid": local.get("metadata", {}).get("uid"),
        "cluster_queue_uid": queue.get("metadata", {}).get("uid"),
        "priority_class": _priority_gate(),
        "peer_preemption_required": False,
    }


def _observer_pod() -> tuple[str, str, str]:
    pods = _global("-n", shared.NAMESPACE, "get", "pods", "-o", "json").get("items") or []
    return lifecycle_guard.select_stable_sfs_observer(pods, excluded_pod_uid="")


def _sfs_absence(observer: tuple[str, str, str]) -> dict[str, Any]:
    name, uid, mount_path = observer
    current = _global("-n", shared.NAMESPACE, "get", "pod", name, "-o", "json")
    if lifecycle_guard.select_stable_sfs_observer([current], excluded_pod_uid="") != observer:
        raise RuntimeError("UID-bound SFS observer mount or readiness drifted")
    observed_run_dir = lifecycle_guard.observer_sfs_path(mount_path, held.RUN_DIR)
    result = subprocess.run(
        [
            "kubectl",
            "-n",
            shared.NAMESPACE,
            "exec",
            name,
            "--",
            "test",
            "!",
            "-e",
            observed_run_dir,
        ],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError("DP6-l SFS root already exists")
    return {
        "observer_pod_name": name,
        "observer_pod_uid": uid,
        "observer_sfs_mount_path": mount_path,
        "run_dir_exists": False,
    }


def _sfs_observation_valid(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "observer_pod_name",
        "observer_pod_uid",
        "observer_sfs_mount_path",
        "run_dir_exists",
    }:
        return False
    try:
        lifecycle_guard.observer_sfs_path(
            str(value.get("observer_sfs_mount_path")), held.RUN_DIR
        )
    except ValueError:
        return False
    return (
        isinstance(value.get("observer_pod_name"), str)
        and bool(value.get("observer_pod_name"))
        and _nonzero_uuid(value.get("observer_pod_uid"))
        and value.get("run_dir_exists") is False
    )


def _kubernetes_gate(active_runs: list[dict[str, str]]) -> dict[str, Any]:
    if active_runs:
        raise RuntimeError("release-first requires zero active project serving runs")
    inventory = _global(
        "-n",
        shared.NAMESPACE,
        "get",
        "rayjobs.ray.io,rayclusters.ray.io,workloads.kueue.x-k8s.io,jobs,configmaps,pods,services",
        "-o",
        "json",
    )
    items = inventory.get("items") or []
    serialized = json.dumps(inventory, sort_keys=True)
    if held.TITLE in serialized or held.RUN_DIR in serialized:
        raise RuntimeError("DP6-l Kubernetes identity already exists")
    terminal = {"SUCCEEDED", "FAILED", "STOPPED"}
    project_rayjobs = [
        row
        for row in items
        if row.get("kind") == "RayJob"
        and "/mnt/sfs/jobs/chris-cyber-evalserve-" in json.dumps(row, sort_keys=True)
        and str(row.get("status", {}).get("jobStatus") or "").upper() not in terminal
    ]
    project_pods = [
        row
        for row in items
        if row.get("kind") == "Pod"
        and row.get("status", {}).get("phase") in {"Pending", "Running"}
        and _pod_gpu_requests(row) > 0
        and isinstance(_pod_run_dir(row), str)
        and str(_pod_run_dir(row)).startswith("/mnt/sfs/jobs/chris-cyber-evalserve-")
    ]
    if project_rayjobs or project_pods:
        raise RuntimeError("active or orphan project serving Kubernetes object exists")
    capacity = _capacity_gate()
    return {
        "current_gpu_nodes": 0,
        "current_gpus": 0,
        "projected_gpu_nodes": 1,
        "projected_gpus": 6,
        "maximum_gpu_nodes": held.MAX_PROJECT_NODES,
        "maximum_gpus": held.MAX_PROJECT_GPUS,
        "active_project_rayjobs": 0,
        "active_project_gpu_pods": 0,
        "capacity": capacity,
    }


def _shape_safe(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    capacity = value.get("capacity") or {}
    if not isinstance(capacity, dict):
        return False
    eligible_uids = capacity.get("eligible_node_uids")
    return (
        set(value) == PROJECT_SHAPE_KEYS
        and set(capacity) == CAPACITY_KEYS
        and value.get("current_gpu_nodes") == 0
        and value.get("current_gpus") == 0
        and value.get("projected_gpu_nodes") == 1
        and value.get("projected_gpus") == 6
        and value.get("maximum_gpu_nodes") == 2
        and value.get("maximum_gpus") == 16
        and value.get("active_project_rayjobs") == 0
        and value.get("active_project_gpu_pods") == 0
        and isinstance(eligible_uids, list)
        and capacity.get("eligible_six_gpu_node_count") == len(eligible_uids)
        and len(eligible_uids) >= 1
        and all(_nonzero_uuid(item) for item in eligible_uids)
        and isinstance(capacity.get("b300_nominal_gpu_quota"), int)
        and isinstance(capacity.get("b300_used_gpu_quota"), int)
        and capacity.get("b300_gpu_quota_headroom", 0) >= 6
        and capacity.get("b300_nominal_gpu_quota") - capacity.get("b300_used_gpu_quota")
        == capacity.get("b300_gpu_quota_headroom")
        and _nonzero_uuid(capacity.get("local_queue_uid"))
        and _nonzero_uuid(capacity.get("cluster_queue_uid"))
        and capacity.get("priority_class")
        == {
            "name": held.SERVER_PRIORITY_CLASS,
            "value": held.SERVER_PRIORITY_VALUE,
            "preemption_policy": "Never",
        }
        and capacity.get("peer_preemption_required") is False
    )


def live_gate(
    client: httpx.Client, release: Mapping[str, Any], root: Path, source_commit: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    validate_server_release(release, root, source_commit)
    payload = held.jobs_payload(root)
    active = _active_project_runs(client)
    shape = _kubernetes_gate(active)
    preview = client.post("/v1/runs/preview", json=payload)
    preview.raise_for_status()
    rendered = held.preview_identity(preview.json()["manifest_yaml"], root)
    sfs = _sfs_absence(_observer_pod())
    value = {
        "schema_version": LIVE_GATE_SCHEMA,
        "status": "PASSED_IMMEDIATELY_BEFORE_CREATE",
        "observed_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "source_commit": source_commit,
        "server_release_receipt_sha256": release["receipt_sha256"],
        "config_sha256": held.config(root)["config_sha256"],
        "request_sha256": held.config(root)["request_sha256"],
        "active_project_serving_runs": 0,
        "project_resource_shape": shape,
        "target_identity_matches": {"jobs_api": 0, "kubernetes": 0, "sfs": 0},
        "sfs_observation": sfs,
        "rendered": rendered,
        "api_mutations": 0,
        "scored_calls": 0,
        "prompts_traces_flags_or_scores_included": False,
    }
    value["receipt_sha256"] = self_hosted.digest_without(value, "receipt_sha256")
    return payload, value


def submit_create_once(
    client: httpx.Client,
    payload: Mapping[str, Any],
    gate: Mapping[str, Any],
    release: Mapping[str, Any],
    source_commit: str,
    root: Path,
) -> str:
    if gate.get("receipt_sha256") != self_hosted.digest_without(dict(gate), "receipt_sha256") or (
        gate.get("schema_version") != LIVE_GATE_SCHEMA
        or gate.get("status") != "PASSED_IMMEDIATELY_BEFORE_CREATE"
        or gate.get("source_commit") != source_commit
        or gate.get("server_release_receipt_sha256") != release.get("receipt_sha256")
        or gate.get("active_project_serving_runs") != 0
        or gate.get("target_identity_matches") != {"jobs_api": 0, "kubernetes": 0, "sfs": 0}
        or gate.get("api_mutations") != 0
        or gate.get("scored_calls") != 0
        or not _sfs_observation_valid(gate.get("sfs_observation"))
        or not _shape_safe(gate.get("project_resource_shape"))
    ):
        raise ValueError("DP6-l live gate is not clear")
    late_active = _active_project_runs(client)
    if late_active:
        raise RuntimeError("release-first project serving state changed before create")
    late_shape = _kubernetes_gate(late_active)
    if not _shape_safe(late_shape):
        raise RuntimeError("DP6-l capacity changed before create")
    if not _sfs_observation_valid(_sfs_absence(_observer_pod())):
        raise RuntimeError("DP6-l SFS state changed before create")
    preview = client.post("/v1/runs/preview", json=dict(payload))
    preview.raise_for_status()
    if held.preview_identity(preview.json()["manifest_yaml"], root) != gate.get("rendered"):
        raise RuntimeError("DP6-l preview drifted at create boundary")
    response = client.post("/v1/runs", json=dict(payload))
    response.raise_for_status()
    if response.status_code != 202:
        raise RuntimeError("DP6-l create did not return HTTP 202")
    name = response.json().get("name")
    if not isinstance(name, str) or not name.startswith("ft-run-"):
        raise RuntimeError("DP6-l create omitted run identity")
    return name


def validate_submission(
    value: Mapping[str, Any], release: Mapping[str, Any], root: Path, source_commit: str
) -> None:
    gate = value.get("live_gate")
    if not isinstance(gate, dict):
        raise ValueError("DP6-l submission omitted its complete sanitized live gate")
    if gate.get("receipt_sha256") != self_hosted.digest_without(gate, "receipt_sha256"):
        raise ValueError("DP6-l nested live gate digest drifted")
    config = held.config(root)
    preview = _load(root / held.PREVIEW_PATH)
    held.validate_preview(preview, root)
    submission_keys = {
        "schema_version",
        "status",
        "api_run_id",
        "source_commit",
        "title",
        "run_dir",
        "serving_block",
        "config_sha256",
        "server_release_receipt_sha256",
        "live_gate_receipt_sha256",
        "live_gate",
        "request_sha256",
        "project_resource_shape",
        "route",
        "http_status",
        "server_instances_created",
        "scored_calls",
        "prompts_traces_flags_or_scores_included",
        "receipt_sha256",
    }
    gate_keys = {
        "schema_version",
        "status",
        "observed_at_utc",
        "source_commit",
        "server_release_receipt_sha256",
        "config_sha256",
        "request_sha256",
        "active_project_serving_runs",
        "project_resource_shape",
        "target_identity_matches",
        "sfs_observation",
        "rendered",
        "api_mutations",
        "scored_calls",
        "prompts_traces_flags_or_scores_included",
        "receipt_sha256",
    }
    expected = {
        "schema_version": SUBMISSION_SCHEMA,
        "status": "SUBMITTED_SCORE_FREE_DP6_L_SERVER",
        "source_commit": source_commit,
        "title": held.TITLE,
        "run_dir": held.RUN_DIR,
        "serving_block": held.SERVING_BLOCK,
        "config_sha256": config["config_sha256"],
        "server_release_receipt_sha256": release.get("receipt_sha256"),
        "live_gate_receipt_sha256": gate["receipt_sha256"],
        "request_sha256": config["request_sha256"],
        "project_resource_shape": gate.get("project_resource_shape"),
        "route": "POST /v1/runs",
        "http_status": 202,
        "server_instances_created": 1,
        "scored_calls": 0,
        "prompts_traces_flags_or_scores_included": False,
    }
    if (
        set(value) != submission_keys
        or set(gate) != gate_keys
        or (
            value.get("receipt_sha256") != self_hosted.digest_without(dict(value), "receipt_sha256")
            or any(value.get(field) != expected_value for field, expected_value in expected.items())
        )
    ):
        raise ValueError("DP6-l submission receipt drifted")
    if not isinstance(value.get("api_run_id"), str) or not str(value["api_run_id"]).startswith(
        "ft-run-"
    ):
        raise ValueError("DP6-l submission receipt omitted its Jobs API identity")
    sfs = gate.get("sfs_observation")
    if not isinstance(sfs, dict):
        raise ValueError("DP6-l SFS observation is absent")
    if gate.get("schema_version") != LIVE_GATE_SCHEMA or (
        gate.get("status") != "PASSED_IMMEDIATELY_BEFORE_CREATE"
        or not isinstance(gate.get("observed_at_utc"), str)
        or gate.get("source_commit") != source_commit
        or gate.get("server_release_receipt_sha256") != release.get("receipt_sha256")
        or gate.get("config_sha256") != config["config_sha256"]
        or gate.get("request_sha256") != config["request_sha256"]
        or gate.get("active_project_serving_runs") != 0
        or gate.get("target_identity_matches") != {"jobs_api": 0, "kubernetes": 0, "sfs": 0}
        or not _sfs_observation_valid(sfs)
        or gate.get("rendered") != preview.get("rendered")
        or gate.get("api_mutations") != 0
        or gate.get("scored_calls") != 0
        or gate.get("prompts_traces_flags_or_scores_included") is not False
        or not _shape_safe(gate.get("project_resource_shape"))
    ):
        raise ValueError("DP6-l nested live gate is not executable evidence")


def submission_receipt(
    api_run_id: str,
    gate: Mapping[str, Any],
    release: Mapping[str, Any],
    source_commit: str,
    root: Path,
) -> dict[str, Any]:
    receipt = {
        "schema_version": SUBMISSION_SCHEMA,
        "status": "SUBMITTED_SCORE_FREE_DP6_L_SERVER",
        "api_run_id": api_run_id,
        "source_commit": source_commit,
        "title": held.TITLE,
        "run_dir": held.RUN_DIR,
        "serving_block": held.SERVING_BLOCK,
        "config_sha256": held.config(root)["config_sha256"],
        "server_release_receipt_sha256": release["receipt_sha256"],
        "live_gate_receipt_sha256": gate["receipt_sha256"],
        "live_gate": dict(gate),
        "request_sha256": held.config(root)["request_sha256"],
        "project_resource_shape": gate["project_resource_shape"],
        "route": "POST /v1/runs",
        "http_status": 202,
        "server_instances_created": 1,
        "scored_calls": 0,
        "prompts_traces_flags_or_scores_included": False,
    }
    receipt["receipt_sha256"] = self_hosted.digest_without(receipt, "receipt_sha256")
    validate_submission(receipt, release, root, source_commit)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("gate", "submit"))
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path.cwd().resolve()
    source_commit = _git(root, "rev-parse", "HEAD")
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("refusing existing DP6-l receipt path")
    if args.command == "submit" and _git(root, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("DP6-l submit requires a clean exact source commit")
    release = _load(args.release)
    with httpx.Client(
        base_url=shared.BASE_URL,
        headers={"Authorization": f"Bearer {shared._token()}", "Accept": "application/json"},  # noqa: SLF001
        timeout=60,
    ) as client:
        payload, gate = live_gate(client, release, root, source_commit)
        if args.command == "gate":
            shared._write_once(args.output, gate)  # noqa: SLF001
            return 0
        api_run_id = submit_create_once(client, payload, gate, release, source_commit, root)
    receipt = submission_receipt(api_run_id, gate, release, source_commit, root)
    shared._write_once(args.output, receipt)  # noqa: SLF001
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
