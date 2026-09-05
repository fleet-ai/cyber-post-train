"""Authenticated create-once preview and submit rail for GLM5.3 v14."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import httpx

from evals.fleet import glm53_dedicated_v8_live as shared
from evals.fleet import glm53_dedicated_v11_preview as generic
from evals.fleet import glm53_dedicated_v14 as v14
from evals.fleet import jobs_api_topology_guard as topology
from evals.fleet import self_hosted


def live_gate(root: Path) -> dict:
    gate = generic.live_gate(root, v14)
    payload = v14.payload(v14.spec(root), root)
    with httpx.Client(
        base_url=shared.BASE_URL,
        headers={"Authorization": f"Bearer {shared._token()}", "Accept": "application/json"},
        timeout=60,
    ) as client:
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

    payload = v14.payload(v14.spec(Path.cwd()), Path.cwd())
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
        "schema_version": "fleet-glm53-dedicated-serving-v14-submission-v1",
        "status": "SUBMITTED",
        "submitted_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "api_run_id": api_run_id,
        "title": v14.TITLE,
        "run_dir": v14.RUN_DIR,
        "request_sha256": gate["request_sha256"],
        "live_gate_receipt_sha256": gate["receipt_sha256"],
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
