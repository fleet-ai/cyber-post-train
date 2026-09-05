"""Authenticated create-once preview and submit rail for dedicated Qwen v3."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from evals.fleet import glm53_dedicated_v8_live as shared
from evals.fleet import priority_preemption_guard as priority_guard
from evals.fleet import qwen38_dedicated_v1_live as v1_live
from evals.fleet import qwen38_dedicated_v3 as v3
from evals.fleet import self_hosted

ACTIVE_STATUSES = {"Submitted", "Suspended", "Running"}
ALLOWED_ACTIVE_DEDICATED_PEERS = {
    "/mnt/sfs/jobs/chris-cyber-evalserve-glm53-tp8-a-v13": {
        "nodes": 1,
        "gpus": 8,
    },
}


def _project_shape(rows: list[dict[str, Any]]) -> dict[str, Any]:
    peers = []
    nodes = 1
    gpus = 1
    for row in rows:
        run_dir = row.get("run_dir")
        status = row.get("status")
        if run_dir == v3.RUN_DIR:
            raise RuntimeError("Qwen v3 Jobs API identity already exists")
        if not isinstance(run_dir, str) or "chris-cyber-evalserve-" not in run_dir:
            continue
        if status not in ACTIVE_STATUSES:
            continue
        shape = ALLOWED_ACTIVE_DEDICATED_PEERS.get(run_dir)
        if shape is None:
            raise RuntimeError("unknown active dedicated serving peer")
        nodes += shape["nodes"]
        gpus += shape["gpus"]
        peers.append({"api_run_id": row.get("name"), "run_dir": run_dir, **shape})
    if nodes > v3.MAX_PROJECT_GPU_NODES or gpus > v3.MAX_PROJECT_GPUS:
        raise RuntimeError("dedicated serving resource ceiling would be exceeded")
    return {
        "planned_nodes": nodes,
        "planned_gpus": gpus,
        "active_peers": peers,
        "max_nodes": v3.MAX_PROJECT_GPU_NODES,
        "max_gpus": v3.MAX_PROJECT_GPUS,
    }


def live_gate(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = v3.payload(v3.spec(root), root)
    headers = {"Authorization": f"Bearer {shared._token()}", "Accept": "application/json"}
    with httpx.Client(base_url=shared.BASE_URL, headers=headers, timeout=60) as client:
        whoami = client.get("/v1/whoami")
        whoami.raise_for_status()
        if not (whoami.json().get("login") or whoami.json().get("email")):
            raise RuntimeError("Jobs API identity is absent")
        openapi = client.get("/v1/openapi.json")
        openapi.raise_for_status()
        paths = openapi.json().get("paths") or {}
        if (
            not all(route in paths for route in ("/v1/runs", "/v1/runs/preview", "/v1/runs/{name}"))
            or "delete" not in paths["/v1/runs/{name}"]
        ):
            raise RuntimeError("Jobs API contract routes are absent")
        runs = shared._runs(client)
        title_matches = sum(row.get("title") == v3.TITLE for row in runs)
        run_dir_matches = sum(row.get("run_dir") == v3.RUN_DIR for row in runs)
        project_shape = _project_shape(runs)
        preview = client.post("/v1/runs/preview", json=payload)
        preview.raise_for_status()
        rendered = v1_live._preview_identity(preview.json()["manifest_yaml"], payload)

    inventory = json.loads(
        shared._kubectl(
            "get",
            "rayjobs.ray.io,rayclusters.ray.io,workloads.kueue.x-k8s.io,jobs,pods,services",
            "-o",
            "json",
        )
    )
    serialized = json.dumps(inventory, sort_keys=True)
    kubernetes_matches = int(v3.TITLE in serialized) + int(v3.RUN_DIR in serialized)
    observer_run_dir = shared._observer_sfs_path(v3.RUN_DIR)
    absent = subprocess.run(
        [
            "kubectl",
            "-n",
            shared.NAMESPACE,
            "exec",
            shared.OBSERVER_POD,
            "--",
            "test",
            "!",
            "-e",
            observer_run_dir,
        ],
        check=False,
        capture_output=True,
    )
    priorities = json.loads(
        subprocess.run(
            ["kubectl", "get", "priorityclass", "-o", "json"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    selected_priority = priority_guard.select_highest_nonpreempting(
        priorities.get("items") or [], allowed_names=v3.API_PRIORITY_CLASSES
    )
    if (
        title_matches
        or run_dir_matches
        or kubernetes_matches
        or absent.returncode != 0
        or selected_priority["name"] != v3.PRIORITY_CLASS
    ):
        raise RuntimeError("Qwen v3 create-once, SFS, or priority gate failed")
    staging = v1_live._staging_identity()
    gate = {
        "schema_version": "fleet-qwen38-dedicated-serving-v3-live-gate-v1",
        "status": "PASSED",
        "observed_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "request_sha256": self_hosted.sha256(self_hosted.canonical_json(payload)),
        "jobs_api_contract_valid": True,
        "jobs_api_identity_valid": True,
        "jobs_api_title_matches": title_matches,
        "jobs_api_run_dir_matches": run_dir_matches,
        "kubernetes_identity_matches": kubernetes_matches,
        "sfs_run_dir_exists": False,
        "sfs_observer_run_dir": observer_run_dir,
        "staging_identity": staging,
        "rendered": rendered,
        "selected_priority": selected_priority,
        "project_resource_shape": project_shape,
        "hardware_preflight_embedded": True,
        "pre_ready_exit_receipt_embedded": True,
        "traffic_idle_guard_seconds": 600,
        "fresh_actual_opencode_parity_required_before_scoring": True,
        "preemption_policy": "Never",
        "release_route": "DELETE /v1/runs/{name}",
        "credentials_included": False,
        "prompts_traces_flags_or_scores_included": False,
    }
    gate["receipt_sha256"] = self_hosted.digest_without(gate, "receipt_sha256")
    return payload, gate


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
    payload, gate = live_gate(Path.cwd())
    if args.command == "preview":
        shared._write_once(args.output, gate)
        print(json.dumps({"status": gate["status"], "receipt_sha256": gate["receipt_sha256"]}))
        return 0

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
        raise RuntimeError("Jobs API response omitted run identity")
    receipt = {
        "schema_version": "fleet-qwen38-dedicated-serving-v3-submission-v1",
        "status": "SUBMITTED",
        "submitted_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "api_run_id": api_run_id,
        "title": v3.TITLE,
        "run_dir": v3.RUN_DIR,
        "request_sha256": gate["request_sha256"],
        "live_gate_receipt_sha256": gate["receipt_sha256"],
        "selected_priority": gate["selected_priority"],
        "project_resource_shape": gate["project_resource_shape"],
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
