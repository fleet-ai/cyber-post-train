"""Create-once Jobs API rail for metric-watchdog-qualified GLM v19."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from evals.fleet import glm53_dedicated_v8_live as shared
from evals.fleet import glm53_dedicated_v15_live as common
from evals.fleet import glm53_dedicated_v19 as v19
from evals.fleet import self_hosted

# Updated immediately before submission from an exact active Jobs API row.
ALLOWED: dict[str, dict[str, int]] = {
    "/mnt/sfs/jobs/chris-cyber-evalserve-q38-tp1-i-v1": {"nodes": 1, "gpus": 1}
}


def _watchdog() -> tuple[bytes, dict[str, Any]]:
    expected = v19.WATCHDOG_QUALIFICATION
    path = shared._observer_sfs_path(expected["receipt_path"])
    raw = subprocess.run(
        ["kubectl", "-n", shared.NAMESPACE, "exec", shared.OBSERVER_POD, "--", "cat", path],
        check=True,
        capture_output=True,
    ).stdout
    if self_hosted.sha256(raw) != expected["file_sha256"]:
        raise RuntimeError("v19 watchdog file digest drifted")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("v19 watchdog receipt is invalid JSON") from exc
    required = {
        "status": "PASSED_NON_SCORED",
        "lifecycle_file_sha256": expected["lifecycle_file_sha256"],
        "production_idle_seconds": 600,
        "canary_idle_seconds": 5,
        "health_only_released": True,
        "health_probes_counted_as_traffic": False,
        "actual_model_requests": 8,
        "actual_model_traffic_prevented_release": True,
        "server_local_uid_scope": True,
        "task_instance_session_verifier_scoring_calls": 0,
        "prompts_traces_flags_or_scores_read": False,
    }
    if any(value.get(key) != wanted for key, wanted in required.items()):
        raise RuntimeError("v19 watchdog receipt fields drifted")
    if (
        value.get("receipt_sha256") != expected["receipt_sha256"]
        or value.get("receipt_sha256")
        != self_hosted.digest_without(value, "receipt_sha256")
    ):
        raise RuntimeError("v19 watchdog self digest drifted")
    return raw, value


def live_gate(root: Path) -> dict[str, Any]:
    raw, watchdog = _watchdog()
    gate = common.live_gate(root, v19, ALLOWED)
    # The inherited v18 rank-51 admission receipt predates its accepted attempt
    # and cannot authorize another cell. It is deliberately excluded from the
    # v19 server authority. A fresh server-bound release/preclaim is mandatory
    # after READY and parity, immediately before any scored create.
    gate.pop("pre_admission", None)
    gate["watchdog_qualification"] = {
        "path": v19.WATCHDOG_QUALIFICATION["receipt_path"],
        "file_sha256": self_hosted.sha256(raw),
        "receipt_sha256": watchdog["receipt_sha256"],
        "lifecycle_file_sha256": watchdog["lifecycle_file_sha256"],
        "job_uid": v19.WATCHDOG_QUALIFICATION["job_uid"],
        "pod_uid": v19.WATCHDOG_QUALIFICATION["pod_uid"],
        "status": watchdog["status"],
    }
    gate["scored_tasks_launched"] = 0
    gate["scored_create_gate"] = {
        "status": "CLOSED",
        "fresh_uid_bound_parity_required": True,
        "fresh_server_bound_release_required": True,
        "fresh_server_bound_preclaim_required": True,
    }
    gate["receipt_sha256"] = self_hosted.digest_without(gate, "receipt_sha256")
    return gate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preview", "submit"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise RuntimeError("refusing existing output receipt")
    if args.command == "submit" and subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout:
        raise RuntimeError("submit requires a clean immutable worktree")
    gate = live_gate(Path.cwd())
    if args.command == "preview":
        shared._write_once(args.output, gate)
        print(json.dumps({"status": gate["status"], "receipt_sha256": gate["receipt_sha256"]}))
        return 0
    payload = v19.payload(v19.spec(Path.cwd()), Path.cwd())
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
        "schema_version": "fleet-glm53-dedicated-serving-v19-submission-v1",
        "status": "SUBMITTED",
        "submitted_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "api_run_id": api_run_id,
        "title": v19.TITLE,
        "run_dir": v19.RUN_DIR,
        "request_sha256": gate["request_sha256"],
        "live_gate_receipt_sha256": gate["receipt_sha256"],
        "watchdog_qualification": gate["watchdog_qualification"],
        "scored_create_gate": gate["scored_create_gate"],
        "project_resource_shape": gate["project_resource_shape"],
        "selected_priority": gate["selected_priority"],
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
