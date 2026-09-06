"""Create-once Jobs API rail for package-qualified GLM v21."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import httpx

from evals.fleet import glm53_dedicated_v8_live as shared
from evals.fleet import glm53_dedicated_v15_live as common
from evals.fleet import glm53_dedicated_v19_live as watchdog_authority
from evals.fleet import glm53_dedicated_v21 as v21
from evals.fleet import self_hosted

ALLOWED = {
    "/mnt/sfs/jobs/chris-cyber-evalserve-q38-tp1-j-v1": {"nodes": 1, "gpus": 1},
}


def _package_canary() -> tuple[bytes, dict]:
    expected = v21.PACKAGE_CANARY
    raw = subprocess.run(
        ["kubectl", "-n", shared.NAMESPACE, "exec", shared.OBSERVER_POD, "--", "cat", shared._observer_sfs_path(expected["receipt_path"])],
        check=True, capture_output=True,
    ).stdout
    if self_hosted.sha256(raw) != expected["file_sha256"]:
        raise RuntimeError("v21 package canary file digest drifted")
    value = json.loads(raw)
    required = {
        "schema_version": "fleet-glm53-dedicated-controller-bootstrap-v1",
        "status": "PASSED_PRE_MODEL",
        "controller_package_sha256": expected["controller_package_sha256"],
        "runtime_plan_sha256": expected["runtime_plan_sha256"],
        "job_uid": expected["job_uid"],
        "pod_uid": expected["pod_uid"],
        "model_requests": 0,
        "task_instance_session_verifier_scoring_calls": 0,
        "projected_evidence_copied_to_private_regular_files": True,
        "runtime_import_closure_valid": True,
        "server_bound_runtime_release_gate_valid": True,
        "prompts_traces_flags_or_scores_read": False,
    }
    if any(value.get(key) != wanted for key, wanted in required.items()):
        raise RuntimeError("v21 package canary fields drifted")
    if value.get("receipt_sha256") != expected["receipt_sha256"] or value.get(
        "receipt_sha256"
    ) != self_hosted.digest_without(value, "receipt_sha256"):
        raise RuntimeError("v21 package canary self digest drifted")
    return raw, value


def live_gate(root: Path) -> dict:
    canary_raw, canary = _package_canary()
    watchdog_raw, watchdog = watchdog_authority._watchdog()
    gate = common.live_gate(root, v21, ALLOWED)
    gate.pop("pre_admission", None)
    gate["watchdog_qualification"] = {
        "path": v21.WATCHDOG_QUALIFICATION["receipt_path"],
        "file_sha256": self_hosted.sha256(watchdog_raw),
        "receipt_sha256": watchdog["receipt_sha256"],
        "lifecycle_file_sha256": watchdog["lifecycle_file_sha256"],
        "status": watchdog["status"],
    }
    gate["package_canary"] = {
        "path": v21.PACKAGE_CANARY["receipt_path"],
        "file_sha256": self_hosted.sha256(canary_raw),
        "receipt_sha256": canary["receipt_sha256"],
        "controller_package_sha256": canary["controller_package_sha256"],
        "runtime_plan_sha256": canary["runtime_plan_sha256"],
        "status": canary["status"],
    }
    gate["scored_create_gate"] = {
        "status": "CLOSED",
        "fresh_uid_bound_parity_required": True,
        "fresh_v21_server_bound_release_required": True,
        "fresh_v21_server_bound_preclaim_required": True,
    }
    gate["scored_tasks_launched"] = 0
    gate["receipt_sha256"] = self_hosted.digest_without(gate, "receipt_sha256")
    return gate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preview", "submit"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise RuntimeError("output collision")
    if args.command == "submit" and subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        capture_output=True, text=True, check=True,
    ).stdout:
        raise RuntimeError("submit requires clean worktree")
    gate = live_gate(Path.cwd())
    if args.command == "preview":
        shared._write_once(args.output, gate)
        return 0
    payload = v21.payload(v21.spec(Path.cwd()), Path.cwd())
    with httpx.Client(
        base_url=shared.BASE_URL,
        headers={"Authorization": f"Bearer {shared._token()}", "Accept": "application/json"},
        timeout=60,
    ) as client:
        response = client.post("/v1/runs", json=payload)
        response.raise_for_status()
        api_run_id = response.json().get("name")
    if response.status_code != 202 or not isinstance(api_run_id, str) or not api_run_id.startswith("ft-run-"):
        raise RuntimeError("v21 Jobs API identity absent")
    receipt = {
        "schema_version": "fleet-glm53-dedicated-serving-v21-submission-v1",
        "status": "SUBMITTED",
        "submitted_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "api_run_id": api_run_id,
        "title": v21.TITLE,
        "run_dir": v21.RUN_DIR,
        "request_sha256": gate["request_sha256"],
        "live_gate_receipt_sha256": gate["receipt_sha256"],
        "watchdog_qualification": gate["watchdog_qualification"],
        "package_canary": gate["package_canary"],
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

