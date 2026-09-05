"""Authenticated preview and create-once submit rail for dedicated Qwen v2."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from evals.fleet import glm53_dedicated_v8_live as shared
from evals.fleet import qwen38_dedicated_v1_live as v1_live
from evals.fleet import qwen38_dedicated_v2 as v2
from evals.fleet import self_hosted

OLD_RUNTIME_UIDS = {
    "9f1ecf8a-a047-44ae-8d51-45af1f4ae9ac",
    "6516f5ef-0e62-42e0-9e3a-c455e8009536",
    "4bb3c7a4-77cc-46cc-a3a6-e204b7f46b33",
    "205a9faf-8ee5-494c-b27c-bfeaeda812d8",
}


def live_gate(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = v2.payload(v2.spec(root), root)
    token = shared._token()
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
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
        title_matches = sum(row.get("title") == v2.TITLE for row in runs)
        run_dir_matches = sum(row.get("run_dir") == v2.RUN_DIR for row in runs)
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
    kubernetes_matches = int(v2.TITLE in serialized) + int(v2.RUN_DIR in serialized)
    live_object_uids = {
        str((item.get("metadata") or {}).get("uid"))
        for item in inventory.get("items") or []
        if isinstance(item, dict) and (item.get("metadata") or {}).get("uid")
    }
    old_runtime_uid_matches = sorted(OLD_RUNTIME_UIDS & live_object_uids)
    observer_run_dir = shared._observer_sfs_path(v2.RUN_DIR)
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
    if (
        title_matches
        or run_dir_matches
        or kubernetes_matches
        or old_runtime_uid_matches
        or absent.returncode != 0
    ):
        raise RuntimeError("Qwen v2 create-once or fresh-runtime identity gate failed")
    priority = json.loads(
        subprocess.run(
            ["kubectl", "get", "priorityclass", "fleet-infra-quiet", "-o", "json"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    if priority.get("preemptionPolicy") != "Never":
        raise RuntimeError("fleet-infra-quiet is not nonpreempting")
    staging = v1_live._staging_identity()
    gate = {
        "schema_version": "fleet-qwen38-dedicated-serving-v2-live-gate-v1",
        "status": "PASSED",
        "observed_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "request_sha256": self_hosted.sha256(self_hosted.canonical_json(payload)),
        "jobs_api_contract_valid": True,
        "jobs_api_identity_valid": True,
        "jobs_api_title_matches": title_matches,
        "jobs_api_run_dir_matches": run_dir_matches,
        "kubernetes_identity_matches": kubernetes_matches,
        "old_runtime_uid_matches": old_runtime_uid_matches,
        "sfs_run_dir_exists": False,
        "sfs_observer_run_dir": observer_run_dir,
        "staging_identity": staging,
        "rendered": rendered,
        "hardware_preflight_embedded": True,
        "pre_ready_exit_receipt_embedded": True,
        "traffic_idle_guard_seconds": 600,
        "fresh_parity_required_before_scoring": True,
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
    root = Path.cwd()
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
    payload, gate = live_gate(root)
    if args.command == "preview":
        shared._write_once(args.output, gate)
        print(json.dumps({"status": "PASSED", "receipt_sha256": gate["receipt_sha256"]}))
        return 0

    token = shared._token()
    with httpx.Client(
        base_url=shared.BASE_URL,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
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
        "schema_version": "fleet-qwen38-dedicated-serving-v2-submission-v1",
        "status": "SUBMITTED",
        "submitted_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "api_run_id": api_run_id,
        "title": v2.TITLE,
        "run_dir": v2.RUN_DIR,
        "request_sha256": gate["request_sha256"],
        "live_gate_receipt_sha256": gate["receipt_sha256"],
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
