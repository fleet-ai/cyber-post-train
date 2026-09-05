"""Fail-closed create-once Jobs API rail for the pre-admitted GLM v15 server."""

from __future__ import annotations

import argparse
import json
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from evals.fleet import glm53_dedicated_v8_live as shared
from evals.fleet import glm53_dedicated_v11_preview as generic
from evals.fleet import glm53_dedicated_v15 as v15
from evals.fleet import jobs_api_topology_guard as topology
from evals.fleet import self_hosted

ACTIVE_STATUSES = {"submitted", "suspended", "running"}
ALLOWED_ACTIVE_EVAL_SERVERS = {
    "/mnt/sfs/jobs/chris-cyber-evalserve-q38-tp1-b-v2": {"nodes": 1, "gpus": 1},
}
MAX_OWNED_EVAL_NODES = 2
MAX_OWNED_EVAL_GPUS = 16


def _validate_prepared(raw: bytes, package: Any = v15) -> dict[str, Any]:
    admission = package.PRE_ADMISSION
    if self_hosted.sha256(raw) != admission["file_sha256"]:
        raise RuntimeError("v15 pre-admission receipt file digest drifted")
    try:
        receipt = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("v15 pre-admission receipt is invalid JSON") from exc
    required = {
        "schema_version": "fleet-glm53-dedicated-v14-scored-canary-preflight-v1",
        "status": "CLEAR_HELD",
        "launch_authorized": False,
        "controller_package_sha256": admission["controller_package_sha256"],
        "held_plan_sha256": admission["held_plan_sha256"],
        "selection_rank": admission["selection_rank"],
        "all_four_rank_cells_unstarted": True,
        "fleet_session_collisions": 0,
        "global_claim_collisions": 0,
        "kubernetes_object_collisions": 0,
        "sfs_output_collisions": 0,
        "mutation_calls": 0,
        "scores_read": False,
        "prompts_traces_flags_read": False,
        "fresh_server_binding_required_after_admission": True,
        "fresh_non_scored_parity_required_after_admission": True,
        "fresh_release_required_immediately_before_canary_create": True,
    }
    if any(receipt.get(key) != value for key, value in required.items()):
        raise RuntimeError("v15 pre-admission receipt fields drifted")
    if (
        receipt.get("receipt_sha256") != admission["receipt_sha256"]
        or receipt.get("receipt_sha256")
        != self_hosted.digest_without(receipt, "receipt_sha256")
    ):
        raise RuntimeError("v15 pre-admission self digest drifted")
    cells = receipt.get("reserved_cell_ids")
    executions = receipt.get("reserved_execution_ids")
    if (
        not isinstance(cells, list)
        or not isinstance(executions, list)
        or len(cells) != 4
        or len(set(cells)) != 4
        or len(executions) != 4
        or len(set(executions)) != 4
        or any(
            not isinstance(value, str)
            or not value.startswith("sha256:")
            or len(value) != 71
            for value in [*cells, *executions]
        )
    ):
        raise RuntimeError("v15 pre-admission cell reservation drifted")
    for field in ("observer_job_uid", "observer_pod_uid"):
        try:
            uuid.UUID(str(receipt.get(field)))
        except (TypeError, ValueError) as exc:
            raise RuntimeError("v15 pre-admission observer UID drifted") from exc
    return receipt


def _read_prepared(package: Any = v15) -> tuple[bytes, dict[str, Any]]:
    observer_path = shared._observer_sfs_path(package.PRE_ADMISSION["receipt_path"])
    completed = subprocess.run(
        [
            "kubectl",
            "-n",
            shared.NAMESPACE,
            "exec",
            shared.OBSERVER_POD,
            "--",
            "cat",
            observer_path,
        ],
        check=True,
        capture_output=True,
    )
    return completed.stdout, _validate_prepared(completed.stdout, package)


def _live_rows(client: httpx.Client, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    live = []
    for row in rows:
        run_dir = row.get("run_dir")
        if not isinstance(run_dir, str) or not run_dir.startswith(
            "/mnt/sfs/jobs/chris-cyber-evalserve-"
        ):
            continue
        name = row.get("name")
        if not isinstance(name, str) or not name.startswith("ft-run-"):
            raise RuntimeError("owned eval server row lacks immutable run identity")
        response = client.get(f"/v1/runs/{name}")
        if response.status_code == 404:
            continue
        response.raise_for_status()
        current = response.json()
        status = str(current.get("status") or row.get("status") or "").lower()
        if status in ACTIVE_STATUSES:
            live.append({**row, **current, "name": name, "run_dir": run_dir})
    return live


def _project_shape(
    rows: list[dict[str, Any]],
    *,
    target_run_dir: str = v15.RUN_DIR,
    allowed: dict[str, dict[str, int]] = ALLOWED_ACTIVE_EVAL_SERVERS,
) -> dict[str, Any]:
    peers = []
    nodes, gpus = 1, 8
    for row in rows:
        run_dir = row["run_dir"]
        if run_dir == target_run_dir:
            raise RuntimeError("GLM v15 Jobs API identity already exists")
        shape = allowed.get(run_dir)
        if shape is None:
            raise RuntimeError("unknown active owned eval server")
        nodes += shape["nodes"]
        gpus += shape["gpus"]
        peers.append({"api_run_id": row["name"], "run_dir": run_dir, **shape})
    if nodes > MAX_OWNED_EVAL_NODES or gpus > MAX_OWNED_EVAL_GPUS:
        raise RuntimeError("owned dedicated eval serving ceiling would be exceeded")
    return {
        "planned_nodes": nodes,
        "planned_gpus": gpus,
        "active_peers": peers,
        "max_nodes": MAX_OWNED_EVAL_NODES,
        "max_gpus": MAX_OWNED_EVAL_GPUS,
    }


def live_gate(
    root: Path,
    package: Any = v15,
    allowed: dict[str, dict[str, int]] = ALLOWED_ACTIVE_EVAL_SERVERS,
) -> dict[str, Any]:
    raw_prepared, prepared = _read_prepared(package)
    gate = generic.live_gate(root, package)
    payload = package.payload(package.spec(root), root)
    with httpx.Client(
        base_url=shared.BASE_URL,
        headers={"Authorization": f"Bearer {shared._token()}", "Accept": "application/json"},
        timeout=60,
    ) as client:
        rows = shared._runs(client)
        project_shape = _project_shape(
            _live_rows(client, rows), target_run_dir=package.RUN_DIR, allowed=allowed
        )
        preview = client.post("/v1/runs/preview", json=payload)
        preview.raise_for_status()
        manifest_yaml = preview.json()["manifest_yaml"]
    nodes = json.loads(shared._kubectl("get", "nodes", "-o", "json"))["items"]
    pods = json.loads(
        subprocess.run(
            ["kubectl", "get", "pods", "--all-namespaces", "-o", "json"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )["items"]
    flavor = json.loads(
        subprocess.run(
            ["kubectl", "get", "resourceflavor.kueue.x-k8s.io", "b300-training", "-o", "json"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    gate["node_capacity_gate"] = topology.require_single_pod_node_capacity(
        manifest_yaml, nodes, pods, flavor
    )
    gate["pre_admission"] = {
        "path": package.PRE_ADMISSION["receipt_path"],
        "file_sha256": self_hosted.sha256(raw_prepared),
        "receipt_sha256": prepared["receipt_sha256"],
        "controller_package_sha256": prepared["controller_package_sha256"],
        "held_plan_sha256": prepared["held_plan_sha256"],
        "selection_rank": prepared["selection_rank"],
        "status": prepared["status"],
    }
    gate["project_resource_shape"] = project_shape
    gate["post_ready_idle_seconds"] = 600
    gate["scored_tasks_launched"] = 0
    gate["receipt_sha256"] = self_hosted.digest_without(gate, "receipt_sha256")
    return gate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preview", "submit"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise RuntimeError("refusing existing output receipt")
    if args.command == "submit":
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        if dirty:
            raise RuntimeError("submit requires a clean immutable worktree")
    gate = live_gate(Path.cwd())
    if args.command == "preview":
        shared._write_once(args.output, gate)
        print(json.dumps({"status": gate["status"], "receipt_sha256": gate["receipt_sha256"]}))
        return 0

    payload = v15.payload(v15.spec(Path.cwd()), Path.cwd())
    with httpx.Client(
        base_url=shared.BASE_URL,
        headers={"Authorization": f"Bearer {shared._token()}", "Accept": "application/json"},
        timeout=60,
    ) as client:
        response = client.post("/v1/runs", json=payload)
        response.raise_for_status()
        if response.status_code != 202:
            raise RuntimeError("Jobs API submit did not return 202")
        api_run_id = response.json().get("name")
    if not isinstance(api_run_id, str) or not api_run_id.startswith("ft-run-"):
        raise RuntimeError("Jobs API response omitted the exact run identity")
    receipt = {
        "schema_version": "fleet-glm53-dedicated-serving-v15-submission-v1",
        "status": "SUBMITTED",
        "submitted_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "api_run_id": api_run_id,
        "title": v15.TITLE,
        "run_dir": v15.RUN_DIR,
        "request_sha256": gate["request_sha256"],
        "live_gate_receipt_sha256": gate["receipt_sha256"],
        "pre_admission": gate["pre_admission"],
        "project_resource_shape": gate["project_resource_shape"],
        "selected_priority": gate["selected_priority"],
        "eligible_flavor_uids": [row["uid"] for row in gate["topology_gate"]["eligible_flavors"]],
        "route": "POST /v1/runs",
        "http_status": 202,
        "scored_tasks_launched": 0,
        "credentials_included": False,
        "prompts_traces_flags_or_scores_included": False,
    }
    receipt["receipt_sha256"] = self_hosted.digest_without(receipt, "receipt_sha256")
    shared._write_once(args.output, receipt)
    print(json.dumps({"api_run_id": api_run_id, "status": "SUBMITTED"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
