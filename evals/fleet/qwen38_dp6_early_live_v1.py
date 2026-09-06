"""Create-once Jobs API rail for the reviewed early non-scored Qwen DP6 server."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from evals.fleet import glm53_dedicated_v8_live as shared
from evals.fleet import qwen38_dp6_early_qualification_v1 as early
from evals.fleet import self_hosted

SERVER_RELEASE_SCHEMA = "fleet-qwen38-dp6-early-server-launch-release-v2"
LIVE_GATE_SCHEMA = "fleet-qwen38-dp6-early-live-submit-gate-v2"
SUBMISSION_SCHEMA = "fleet-qwen38-dp6-early-submission-v2"
TP1_BINDING_PATH = Path(
    "docs/evidence/qwen38-study/2026-09-05-qwen38-dedicated-tp1-j-v1-server-binding.json"
)
ACTIVE_STATUSES = {"SUBMITTED", "SUSPENDED", "RUNNING"}
MAX_TP1_TRAFFIC_AGE_SECONDS = 300
SERVER_PRIORITY_VALUE = -1000


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def validate_server_release(value: Mapping[str, Any], root: Path, source_commit: str) -> None:
    _, plan, preview, inventory, held = early.load_all(root)
    binding = _load(root / TP1_BINDING_PATH)
    if value.get("receipt_sha256") != self_hosted.digest_without(dict(value), "receipt_sha256") or (
        value.get("schema_version") != SERVER_RELEASE_SCHEMA
        or value.get("status") != "RELEASED_FOR_ONE_NON_SCORED_DP6_SERVER"
        or value.get("launch_authorized") is not True
        or value.get("scoring_authorized") is not False
        or value.get("source_commit") != source_commit
        or value.get("title") != early.TITLE
        or value.get("run_dir") != early.RUN_DIR
        or value.get("serving_block") != early.SERVING_BLOCK
        or value.get("held_release_receipt_sha256") != held.get("receipt_sha256")
        or value.get("plan_receipt_sha256") != plan.get("receipt_sha256")
        or value.get("preview_receipt_sha256") != preview.get("receipt_sha256")
        or value.get("review_inventory_receipt_sha256") != inventory.get("receipt_sha256")
        or value.get("allowed_peer_binding_receipt_sha256") != binding.get("receipt_sha256")
        or value.get("server_create_limit") != 1
        or value.get("statistical_cells_selected") != 0
        or value.get("task_instance_session_verifier_scoring_calls") != 0
        or value.get("prompts_traces_flags_or_scores_included") is not False
    ):
        raise ValueError("early DP6 server release is not executable")


def _active_serving_runs(client: httpx.Client) -> list[dict[str, Any]]:
    active: list[dict[str, Any]] = []
    for row in shared._runs(client):  # noqa: SLF001 - shared paginated Jobs API authority
        name = row.get("name")
        run_dir = row.get("run_dir")
        if not isinstance(name, str) or not isinstance(run_dir, str):
            continue
        if row.get("title") == early.TITLE or run_dir == early.RUN_DIR:
            raise RuntimeError("early DP6 Jobs API identity already exists")
        if "chris-cyber-evalserve-" not in run_dir:
            continue
        current = client.get(f"/v1/runs/{name}")
        if current.status_code == 404:
            continue
        current.raise_for_status()
        live = current.json()
        if live.get("run_dir") != run_dir:
            raise RuntimeError("Jobs API serving history/live identity drifted")
        if str(live.get("status") or "").upper() in ACTIVE_STATUSES:
            active.append({"api_run_id": name, "run_dir": run_dir, "status": live["status"]})
    return active


def _pod_run_dir(pod: Mapping[str, Any]) -> str | None:
    values = [
        env.get("value")
        for container in pod.get("spec", {}).get("containers") or []
        for env in container.get("env") or []
        if env.get("name") == "RUN_DIR"
    ]
    return values[0] if len(values) == 1 and isinstance(values[0], str) else None


def _pod_gpu_requests(pod: Mapping[str, Any]) -> int:
    values = [
        (container.get("resources", {}).get("requests") or {}).get("nvidia.com/gpu", 0)
        for container in pod.get("spec", {}).get("containers") or []
    ]
    try:
        return sum(int(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("project GPU request shape drifted") from exc


def _condition_true(value: Mapping[str, Any], condition_type: str) -> bool:
    return any(
        row.get("type") == condition_type and row.get("status") == "True"
        for row in value.get("status", {}).get("conditions") or []
        if isinstance(row, dict)
    )


def _validate_peer_control_plane(found: Mapping[str, list[dict[str, Any]]]) -> None:
    rayjob = found["RayJob"][0]
    workload = found["Workload"][0]
    conditions = workload.get("status", {}).get("conditions") or []
    text = json.dumps(conditions, sort_keys=True).lower()
    if str(rayjob.get("status", {}).get("jobStatus") or "").upper() != "RUNNING":
        raise RuntimeError("exact TP1-j RayJob is not RUNNING")
    if (
        not _condition_true(workload, "Admitted")
        or not _condition_true(workload, "QuotaReserved")
        or _condition_true(workload, "Evicted")
        or "preempt" in text
        or "noreservation" in text
    ):
        raise RuntimeError("exact TP1-j Workload is not stably admitted")


def _priority_class_gate() -> dict[str, Any]:
    value = _kubectl_global_json(
        "get", "priorityclass", early.SERVER_PRIORITY_CLASS, "-o", "json"
    )
    if (
        value.get("metadata", {}).get("name") != early.SERVER_PRIORITY_CLASS
        or value.get("value") != SERVER_PRIORITY_VALUE
        or value.get("preemptionPolicy") != "Never"
    ):
        raise RuntimeError("DP6 server PriorityClass contract drifted")
    return {
        "name": early.SERVER_PRIORITY_CLASS,
        "value": SERVER_PRIORITY_VALUE,
        "preemption_policy": "Never",
    }


def _tp1_traffic_gate(observer_pod: str, binding: Mapping[str, Any]) -> dict[str, Any]:
    traffic = "/mnt/sfs/jobs/chris-cyber-evalserve-q38-tp1-j-v1/lifecycle/traffic"
    result = subprocess.run(
        [
            "kubectl",
            "-n",
            shared.NAMESPACE,
            "exec",
            observer_pod,
            "--",
            "stat",
            "-c",
            "%Y %s",
            traffic,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError("exact TP1-j traffic marker is absent")
    try:
        mtime, size = (int(item) for item in result.stdout.strip().split())
    except (TypeError, ValueError) as exc:
        raise RuntimeError("exact TP1-j traffic marker is malformed") from exc
    age = max(0, int(time.time()) - mtime)
    if age > MAX_TP1_TRAFFIC_AGE_SECONDS:
        raise RuntimeError("exact TP1-j productive traffic is stale")
    return {
        "api_run_id": binding["api_run_id"],
        "head_pod_uid": binding["head_pod_uid"],
        "traffic_marker_path": traffic,
        "traffic_mtime_epoch": mtime,
        "traffic_marker_bytes": size,
        "traffic_age_seconds": age,
        "maximum_age_seconds": MAX_TP1_TRAFFIC_AGE_SECONDS,
    }


def _resource_total(rows: list[dict[str, Any]], name: str) -> int:
    for row in rows:
        if row.get("name") == name:
            return int(row.get("total") or row.get("nominalQuota") or 0)
    raise RuntimeError(f"Kueue resource is absent: {name}")


def _kubectl_global_json(*args: str) -> dict[str, Any]:
    result = subprocess.run(
        ["kubectl", *args], check=True, capture_output=True, text=True
    )
    value = json.loads(result.stdout)
    if not isinstance(value, dict):
        raise RuntimeError("global Kubernetes response is malformed")
    return value


def _capacity_gate(tp1_node: str) -> dict[str, Any]:
    """Prove at least one non-preempting six-GPU fit while TP1 remains healthy."""
    nodes = (
        _kubectl_global_json(
            "get", "nodes", "-l", "workload=fleetai-training-ng-gpu", "-o", "json"
        ).get("items")
        or []
    )
    pods = (
        _kubectl_global_json("get", "pods", "--all-namespaces", "-o", "json").get(
            "items"
        )
        or []
    )
    used: dict[str, int] = {}
    active_gpu_pods: list[dict[str, Any]] = []
    for pod in pods:
        if pod.get("status", {}).get("phase") not in {"Pending", "Running"}:
            continue
        node = pod.get("spec", {}).get("nodeName")
        if not isinstance(node, str):
            continue
        gpus = _pod_gpu_requests(pod)
        used[node] = used.get(node, 0) + gpus
        if gpus:
            active_gpu_pods.append(
                {
                    "namespace": pod.get("metadata", {}).get("namespace"),
                    "name": pod.get("metadata", {}).get("name"),
                    "uid": pod.get("metadata", {}).get("uid"),
                    "node": node,
                    "gpus": gpus,
                }
            )
    candidates = []
    for node in nodes:
        metadata = node.get("metadata", {})
        labels = metadata.get("labels") or {}
        name = metadata.get("name")
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
            candidates.append({"name": name, "allocatable_gpus": allocatable, "free_gpus": free})
    tp1_candidates = [row for row in candidates if row["name"] == tp1_node]
    if not candidates or tp1_candidates != [
        {"name": tp1_node, "allocatable_gpus": 8, "free_gpus": 6}
    ]:
        raise RuntimeError("no safe six-GPU fit exists while TP1 remains healthy")
    target = [row for row in active_gpu_pods if row["node"] == tp1_node]
    if sorted(row["gpus"] for row in target) != [1, 1]:
        raise RuntimeError("shared TP1 node peer GPU shape drifted")

    local_queue = _kubectl_global_json(
        "get", "localqueue", "training-lq", "-n", shared.NAMESPACE, "-o", "json"
    )
    queue = _kubectl_global_json("get", "clusterqueue", "training-cq", "-o", "json")
    if (
        local_queue.get("spec", {}).get("clusterQueue") != "training-cq"
        or not _condition_true(local_queue, "Active")
        or not _condition_true(queue, "Active")
    ):
        raise RuntimeError("training queue is not active")
    flavor = next(
        row
        for group in queue.get("spec", {}).get("resourceGroups") or []
        for row in group.get("flavors") or []
        if row.get("name") == "b300-training"
    )
    nominal = _resource_total(flavor.get("resources") or [], "nvidia.com/gpu")
    usage = next(
        row
        for row in queue.get("status", {}).get("flavorsUsage") or []
        if row.get("name") == "b300-training"
    )
    used_quota = _resource_total(usage.get("resources") or [], "nvidia.com/gpu")
    if nominal - used_quota < 6:
        raise RuntimeError("training-cq lacks six-GPU nominal quota headroom")
    priority = _priority_class_gate()
    preemption = queue.get("spec", {}).get("preemption") or {}
    if preemption.get("reclaimWithinCohort") != "Never" or (
        (preemption.get("borrowWithinCohort") or {}).get("policy") != "Never"
    ):
        raise RuntimeError("training-cq cohort preemption policy drifted")
    return {
        "target_node": tp1_node,
        "target_allocatable_gpus": 8,
        "target_active_gpu_requests": 2,
        "target_free_gpus": 6,
        "eligible_six_gpu_node_count": len(candidates),
        "tp1_node_is_eligible": True,
        "active_gpu_pods_on_target": target,
        "b300_nominal_gpu_quota": nominal,
        "b300_used_gpu_quota": used_quota,
        "b300_gpu_quota_headroom": nominal - used_quota,
        "local_queue_uid": local_queue.get("metadata", {}).get("uid"),
        "cluster_queue_uid": queue.get("metadata", {}).get("uid"),
        "priority_class": priority,
        "peer_preemption_required": False,
    }


def _kubernetes_gate(
    binding: Mapping[str, Any], active_runs: list[dict[str, Any]]
) -> tuple[dict[str, Any], str]:
    inventory = json.loads(
        shared._kubectl(  # noqa: SLF001 - GET-only cluster authority
            "get",
            "rayjobs.ray.io,rayclusters.ray.io,workloads.kueue.x-k8s.io,jobs,pods,services",
            "-o",
            "json",
        )
    )
    items = inventory.get("items") or []
    expected = {
        "RayJob": binding["rayjob_uid"],
        "Workload": binding["workload_uid"],
        "Pod": binding["head_pod_uid"],
        "Service": binding["service_uid"],
    }
    found = {
        kind: [
            row
            for row in items
            if row.get("kind") == kind and row.get("metadata", {}).get("uid") == uid
        ]
        for kind, uid in expected.items()
    }
    if any(len(rows) != 1 for rows in found.values()):
        raise RuntimeError("exact TP1-j Kubernetes peer binding is absent")
    _validate_peer_control_plane(found)
    pod = found["Pod"][0]
    statuses = pod.get("status", {}).get("containerStatuses") or []
    if (
        pod.get("status", {}).get("phase") != "Running"
        or not statuses
        or not all(row.get("ready") is True for row in statuses)
        or sum(int(row.get("restartCount") or 0) for row in statuses) != 0
    ):
        raise RuntimeError("exact TP1-j peer is not Running/Ready/restart0")
    serialized = json.dumps(inventory, sort_keys=True)
    if early.TITLE in serialized or early.RUN_DIR in serialized:
        raise RuntimeError("early DP6 Kubernetes identity already exists")
    active_ids = {str(row["api_run_id"]) for row in active_runs}
    active_dirs = {str(row["run_dir"]) for row in active_runs}
    terminal_statuses = {"SUCCEEDED", "FAILED", "STOPPED"}
    all_project_rayjobs = [
        row
        for row in items
        if row.get("kind") == "RayJob"
        and (
            row.get("metadata", {}).get("name") in active_ids
            or "/mnt/sfs/jobs/chris-cyber-evalserve-" in json.dumps(row, sort_keys=True)
        )
    ]
    project_rayjobs = [
        row
        for row in all_project_rayjobs
        if row.get("metadata", {}).get("name") in active_ids
        or str(row.get("status", {}).get("jobStatus") or "").upper() not in terminal_statuses
    ]
    orphan_rayjobs = [
        row for row in project_rayjobs if row.get("metadata", {}).get("name") not in active_ids
    ]
    project_rayjob_uids = {row.get("metadata", {}).get("uid") for row in project_rayjobs}
    project_cluster_rows = [
        row
        for row in items
        if row.get("kind") == "RayCluster"
        and any(
            owner.get("uid") in project_rayjob_uids
            for owner in row.get("metadata", {}).get("ownerReferences") or []
        )
    ]
    project_cluster_names = {
        str(row.get("metadata", {}).get("name")) for row in project_cluster_rows
    }
    project_cluster_uids = {
        row.get("metadata", {}).get("uid") for row in project_cluster_rows
    }
    project_pods = []
    orphan_pods = []
    for row in items:
        if row.get("kind") != "Pod" or row.get("status", {}).get("phase") not in {
            "Pending",
            "Running",
        }:
            continue
        if _pod_gpu_requests(row) <= 0:
            continue
        metadata = row.get("metadata", {})
        run_dir = _pod_run_dir(row)
        labels = metadata.get("labels") or {}
        linked_cluster = (
            labels.get("ray.io/cluster") in project_cluster_names
            or any(
                owner.get("uid") in project_cluster_uids
                for owner in metadata.get("ownerReferences") or []
            )
        )
        linked_api_run = any(
            str(metadata.get("name") or "").startswith(api_run_id)
            for api_run_id in active_ids
        )
        explicit_project_run = isinstance(run_dir, str) and run_dir.startswith(
            "/mnt/sfs/jobs/chris-cyber-evalserve-"
        )
        if not (explicit_project_run or linked_cluster or linked_api_run):
            continue
        project_pods.append(row)
        if (run_dir is not None and run_dir not in active_dirs) or (
            run_dir is None and not (linked_cluster or linked_api_run)
        ):
            orphan_pods.append(row)
    if orphan_pods or orphan_rayjobs:
        raise RuntimeError("orphan project serving Kubernetes object exists")
    if len(project_pods) != len(active_runs) or len(project_rayjobs) != len(active_runs):
        raise RuntimeError("project serving API/Kubernetes cardinality drifted")
    current_gpus = sum(_pod_gpu_requests(row) for row in project_pods)
    current_nodes = {row.get("spec", {}).get("nodeName") for row in project_pods}
    if None in current_nodes or not current_nodes:
        raise RuntimeError("project serving GPU node placement is incomplete")
    if pod not in project_pods or current_gpus != 1 or len(current_nodes) != 1:
        raise RuntimeError("exact TP1-j project GPU shape drifted")
    capacity = _capacity_gate(str(pod.get("spec", {}).get("nodeName")))
    return {
        "current_gpu_nodes": len(current_nodes),
        "current_gpus": current_gpus,
        "projected_gpu_nodes_allowed": [len(current_nodes), len(current_nodes) + 1],
        "projected_gpus": current_gpus + 6,
        "maximum_gpu_nodes": 2,
        "maximum_gpus": 16,
        "tp1_head_pod_uid": binding["head_pod_uid"],
        "peer_workload_admitted": True,
        "peer_workload_quota_reserved": True,
        "peer_workload_preemption_observed": False,
        "scope": "project_chris_cyber_evalserve_runs_only",
        "unrelated_namespace_gpu_pods_counted": False,
        "physical_capacity": capacity,
        "project_object_inventory": {
            "active_jobs_api_run_ids": sorted(active_ids),
            "active_run_dirs": sorted(active_dirs),
            "rayjob_uids": sorted(row["metadata"]["uid"] for row in project_rayjobs),
            "raycluster_uids": sorted(
                str(row["metadata"]["uid"]) for row in project_cluster_rows
            ),
            "gpu_pod_uids": sorted(row["metadata"]["uid"] for row in project_pods),
            "gpu_node_names": sorted(str(value) for value in current_nodes),
            "gpu_requests": current_gpus,
            "orphan_project_rayjobs": 0,
            "orphan_project_gpu_pods": 0,
            "terminal_project_rayjobs_ignored": len(all_project_rayjobs) - len(project_rayjobs),
        },
    }, pod["metadata"]["name"]


def _validate_active_peer(
    active: list[dict[str, Any]], binding: Mapping[str, Any]
) -> dict[str, Any]:
    expected = {
        "api_run_id": binding["api_run_id"],
        "run_dir": "/mnt/sfs/jobs/chris-cyber-evalserve-q38-tp1-j-v1",
        "status": "RUNNING",
    }
    normalized = [{**row, "status": str(row["status"]).upper()} for row in active]
    if normalized != [expected]:
        raise RuntimeError("active serving peers are not exactly productive TP1-j")
    return expected


def _sfs_run_dir_absent(observer_pod: str) -> bool:
    result = subprocess.run(
        [
            "kubectl",
            "-n",
            shared.NAMESPACE,
            "exec",
            observer_pod,
            "--",
            "test",
            "!",
            "-e",
            early.RUN_DIR,
        ],
        check=False,
        capture_output=True,
    )
    return result.returncode == 0


def _project_shape_is_safe(shape: object) -> bool:
    if not isinstance(shape, dict):
        return False
    return not (
        shape.get("current_gpu_nodes") != 1
        or shape.get("current_gpus") != 1
        or shape.get("projected_gpu_nodes_allowed") != [1, 2]
        or shape.get("projected_gpus") != 7
        or shape.get("maximum_gpu_nodes") != 2
        or shape.get("maximum_gpus") != 16
        or shape.get("peer_workload_admitted") is not True
        or shape.get("peer_workload_quota_reserved") is not True
        or shape.get("peer_workload_preemption_observed") is not False
        or (shape.get("physical_capacity") or {}).get(
            "eligible_six_gpu_node_count", 0
        )
        < 1
        or (shape.get("physical_capacity") or {}).get("tp1_node_is_eligible")
        is not True
        or (shape.get("physical_capacity") or {}).get(
            "b300_gpu_quota_headroom", 0
        )
        < 6
        or (shape.get("physical_capacity") or {}).get("priority_class")
        != {
            "name": early.SERVER_PRIORITY_CLASS,
            "value": SERVER_PRIORITY_VALUE,
            "preemption_policy": "Never",
        }
        or (shape.get("project_object_inventory") or {}).get(
            "orphan_project_rayjobs"
        )
        != 0
        or (shape.get("project_object_inventory") or {}).get(
            "orphan_project_gpu_pods"
        )
        != 0
    )


def live_gate(
    client: httpx.Client, release: Mapping[str, Any], root: Path, source_commit: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    validate_server_release(release, root, source_commit)
    config = early.load_all(root)[0]
    payload = early.jobs_payload(root)
    whoami = client.get("/v1/whoami")
    whoami.raise_for_status()
    if not (whoami.json().get("login") or whoami.json().get("email")):
        raise RuntimeError("Jobs API identity is absent")
    openapi = client.get("/v1/openapi.json")
    openapi.raise_for_status()
    if not all(
        route in (openapi.json().get("paths") or {})
        for route in ("/v1/runs", "/v1/runs/preview", "/v1/runs/{name}")
    ):
        raise RuntimeError("Jobs API contract routes are absent")
    binding = _load(root / TP1_BINDING_PATH)
    active = _active_serving_runs(client)
    expected_peer = _validate_active_peer(active, binding)
    preview = client.post("/v1/runs/preview", json=payload)
    preview.raise_for_status()
    rendered = early.preview_identity(preview.json()["manifest_yaml"], root)
    project_shape, observer_pod = _kubernetes_gate(binding, active)
    tp1_traffic = _tp1_traffic_gate(observer_pod, binding)
    if not _sfs_run_dir_absent(observer_pod):
        raise RuntimeError("early DP6 SFS run root already exists")
    gate = {
        "schema_version": LIVE_GATE_SCHEMA,
        "status": "PASSED_IMMEDIATELY_BEFORE_CREATE",
        "observed_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "freshness_seconds_at_submit": 0,
        "source_commit": source_commit,
        "server_release_receipt_sha256": release["receipt_sha256"],
        "config_sha256": config["config_sha256"],
        "request_sha256": self_hosted.sha256(self_hosted.canonical_json(payload)),
        "allowed_active_peer": expected_peer,
        "allowed_active_peer_traffic": tp1_traffic,
        "project_resource_shape": project_shape,
        "jobs_api_title_matches": 0,
        "jobs_api_run_dir_matches": 0,
        "kubernetes_identity_matches": 0,
        "sfs_run_dir_exists": False,
        "rendered": rendered,
        "api_mutations": 0,
        "scoring_authorized": False,
        "statistical_cells_selected": 0,
        "prompts_traces_flags_or_scores_included": False,
    }
    gate["receipt_sha256"] = self_hosted.digest_without(gate, "receipt_sha256")
    return payload, gate


def submit_create_once(
    client: httpx.Client,
    payload: Mapping[str, Any],
    gate: Mapping[str, Any],
    release: Mapping[str, Any],
    source_commit: str,
    root: Path,
) -> str:
    shape = gate.get("project_resource_shape")
    if gate.get("receipt_sha256") != self_hosted.digest_without(dict(gate), "receipt_sha256") or (
        gate.get("schema_version") != LIVE_GATE_SCHEMA
        or gate.get("status") != "PASSED_IMMEDIATELY_BEFORE_CREATE"
        or gate.get("freshness_seconds_at_submit") != 0
        or gate.get("source_commit") != source_commit
        or gate.get("server_release_receipt_sha256") != release.get("receipt_sha256")
        or gate.get("jobs_api_title_matches") != 0
        or gate.get("jobs_api_run_dir_matches") != 0
        or gate.get("kubernetes_identity_matches") != 0
        or gate.get("sfs_run_dir_exists") is not False
        or gate.get("api_mutations") != 0
        or gate.get("scoring_authorized") is not False
        or gate.get("statistical_cells_selected") != 0
        or gate.get("request_sha256")
        != self_hosted.sha256(self_hosted.canonical_json(dict(payload)))
        or not isinstance(gate.get("allowed_active_peer_traffic"), dict)
        or gate["allowed_active_peer_traffic"].get("head_pod_uid")
        != shape.get("tp1_head_pod_uid")
        or gate["allowed_active_peer_traffic"].get("traffic_age_seconds", 999999)
        > MAX_TP1_TRAFFIC_AGE_SECONDS
        or not _project_shape_is_safe(shape)
    ):
        raise ValueError("early DP6 live submit gate is not clear")
    # Repeat the complete project-capacity gate at the create boundary. A
    # different project server, orphan, quota change, or fit change is just as
    # important as this exact identity appearing after the earlier review.
    binding = _load(root / TP1_BINDING_PATH)
    late_active = _active_serving_runs(client)
    late_peer = _validate_active_peer(late_active, binding)
    late_shape, late_observer = _kubernetes_gate(binding, late_active)
    late_traffic = _tp1_traffic_gate(late_observer, binding)
    if (
        late_peer != gate.get("allowed_active_peer")
        or not _project_shape_is_safe(late_shape)
        or late_traffic.get("head_pod_uid") != late_shape.get("tp1_head_pod_uid")
        or late_traffic.get("traffic_age_seconds", 999999)
        > MAX_TP1_TRAFFIC_AGE_SECONDS
        or not _sfs_run_dir_absent(late_observer)
    ):
        raise RuntimeError("early DP6 project capacity changed before create")
    preview = client.post("/v1/runs/preview", json=dict(payload))
    preview.raise_for_status()
    rendered = early.preview_identity(preview.json()["manifest_yaml"], root)
    if rendered != gate.get("rendered"):
        raise RuntimeError("early DP6 Jobs API preview drifted after the live gate")
    response = client.post("/v1/runs", json=dict(payload))
    response.raise_for_status()
    if response.status_code != 202:
        raise RuntimeError("early DP6 create did not return HTTP 202")
    api_run_id = response.json().get("name")
    if not isinstance(api_run_id, str) or not api_run_id.startswith("ft-run-"):
        raise RuntimeError("early DP6 create response omitted run identity")
    return api_run_id


def validate_submission(
    value: Mapping[str, Any],
    payload: Mapping[str, Any],
    release: Mapping[str, Any],
    source_commit: str,
    root: Path,
) -> None:
    config = early.load_all(root)[0]
    gate = value.get("live_gate")
    if not isinstance(gate, dict):
        raise ValueError("early DP6 submission omitted durable live gate")
    if value.get("receipt_sha256") != self_hosted.digest_without(dict(value), "receipt_sha256") or (
        value.get("schema_version") != SUBMISSION_SCHEMA
        or value.get("status") != "SUBMITTED_NON_SCORED_SERVER"
        or value.get("source_commit") != source_commit
        or value.get("title") != early.TITLE
        or value.get("run_dir") != early.RUN_DIR
        or value.get("serving_block") != early.SERVING_BLOCK
        or value.get("config_sha256") != config["config_sha256"]
        or value.get("request_sha256")
        != self_hosted.sha256(self_hosted.canonical_json(dict(payload)))
        or value.get("server_release_receipt_sha256") != release.get("receipt_sha256")
        or value.get("live_gate_receipt_sha256") != gate.get("receipt_sha256")
        or gate.get("receipt_sha256") != self_hosted.digest_without(gate, "receipt_sha256")
        or gate.get("source_commit") != source_commit
        or gate.get("server_release_receipt_sha256") != release.get("receipt_sha256")
        or gate.get("config_sha256") != config["config_sha256"]
        or gate.get("request_sha256") != value.get("request_sha256")
        or value.get("project_resource_shape") != gate.get("project_resource_shape")
        or value.get("route") != "POST /v1/runs"
        or value.get("http_status") != 202
        or value.get("server_instances_created") != 1
        or value.get("scored_calls") != 0
        or value.get("prompts_traces_flags_or_scores_included") is not False
    ):
        raise ValueError("early DP6 submission receipt drifted")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("gate", "submit"))
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path.cwd().resolve()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("refusing existing early DP6 receipt path")
    source_commit = _git(root, "rev-parse", "HEAD")
    if args.command == "submit" and _git(root, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("early DP6 submit requires a clean exact source commit")
    release = _load(args.release)
    with httpx.Client(
        base_url=shared.BASE_URL,
        headers={"Authorization": f"Bearer {shared._token()}", "Accept": "application/json"},  # noqa: SLF001
        timeout=60,
    ) as client:
        payload, gate = live_gate(client, release, root, source_commit)
        if args.command == "gate":
            shared._write_once(args.output, gate)  # noqa: SLF001
            print(json.dumps({"status": gate["status"], "receipt_sha256": gate["receipt_sha256"]}))
            return 0
        api_run_id = submit_create_once(client, payload, gate, release, source_commit, root)
    receipt = {
        "schema_version": SUBMISSION_SCHEMA,
        "status": "SUBMITTED_NON_SCORED_SERVER",
        "submitted_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "api_run_id": api_run_id,
        "title": early.TITLE,
        "run_dir": early.RUN_DIR,
        "serving_block": early.SERVING_BLOCK,
        "config_sha256": early.load_all(root)[0]["config_sha256"],
        "request_sha256": gate["request_sha256"],
        "server_release_receipt_sha256": release["receipt_sha256"],
        "live_gate_receipt_sha256": gate["receipt_sha256"],
        "live_gate": gate,
        "source_commit": source_commit,
        "project_resource_shape": gate["project_resource_shape"],
        "route": "POST /v1/runs",
        "http_status": 202,
        "server_instances_created": 1,
        "scored_calls": 0,
        "prompts_traces_flags_or_scores_included": False,
    }
    receipt["receipt_sha256"] = self_hosted.digest_without(receipt, "receipt_sha256")
    validate_submission(receipt, payload, release, source_commit, root)
    shared._write_once(args.output, receipt)  # noqa: SLF001
    print(json.dumps({"api_run_id": api_run_id, "status": receipt["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
