"""Create-once Jobs API rail for bootstrap-qualified GLM v17."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import httpx

from evals.fleet import glm53_dedicated_v8_live as shared
from evals.fleet import glm53_dedicated_v15_live as common
from evals.fleet import glm53_dedicated_v17 as v17
from evals.fleet import self_hosted

ALLOWED = {
    "/mnt/sfs/jobs/chris-cyber-evalserve-q38-tp1-h-v1": {"nodes": 1, "gpus": 1}
}


def _bootstrap() -> tuple[bytes, dict]:
    path = shared._observer_sfs_path(v17.CONTROLLER_BOOTSTRAP["receipt_path"])
    raw = subprocess.run(
        ["kubectl", "-n", shared.NAMESPACE, "exec", shared.OBSERVER_POD, "--", "cat", path],
        check=True,
        capture_output=True,
    ).stdout
    if self_hosted.sha256(raw) != v17.CONTROLLER_BOOTSTRAP["file_sha256"]:
        raise RuntimeError("v17 bootstrap file digest drifted")
    value = json.loads(raw)
    required = {
        "schema_version": "fleet-glm53-dedicated-controller-bootstrap-v1",
        "status": "PASSED_PRE_MODEL",
        "job_uid": v17.CONTROLLER_BOOTSTRAP["job_uid"],
        "pod_uid": v17.CONTROLLER_BOOTSTRAP["pod_uid"],
        "controller_package_sha256": v17.CONTROLLER_BOOTSTRAP["controller_package_sha256"],
        "harness_image": "chris/opencode:1.18.27-cyber-v1",
        "harness_version": "1.18.27",
        "projected_evidence_copied_to_private_regular_files": True,
        "runtime_import_closure_valid": True,
        "docker_build_and_version_check_completed_by_exact_run_sh": True,
        "claim_calls": 0,
        "model_requests": 0,
        "task_instance_session_verifier_scoring_calls": 0,
        "prompts_traces_flags_or_scores_read": False,
    }
    if any(value.get(k) != v for k, v in required.items()):
        raise RuntimeError("v17 bootstrap receipt fields drifted")
    if (
        value.get("receipt_sha256") != v17.CONTROLLER_BOOTSTRAP["receipt_sha256"]
        or value.get("receipt_sha256")
        != self_hosted.digest_without(value, "receipt_sha256")
    ):
        raise RuntimeError("v17 bootstrap self digest drifted")
    return raw, value


def _runtime_gate() -> tuple[bytes, dict]:
    path = shared._observer_sfs_path(v17.RUNTIME_GATE["receipt_path"])
    raw = subprocess.run(
        ["kubectl", "-n", shared.NAMESPACE, "exec", shared.OBSERVER_POD, "--", "cat", path],
        check=True, capture_output=True,
    ).stdout
    if self_hosted.sha256(raw) != v17.RUNTIME_GATE["file_sha256"]:
        raise RuntimeError("v18 runtime gate file digest drifted")
    value = json.loads(raw)
    if (
        value.get("schema_version") != "fleet-glm53-dedicated-runtime-gate-observer-v1"
        or value.get("status") != "PASSED_PRECLAIM"
        or value.get("job_uid") != v17.RUNTIME_GATE["job_uid"]
        or value.get("pod_uid") != v17.RUNTIME_GATE["pod_uid"]
        or value.get("plan_sha256") != v17.RUNTIME_GATE["plan_sha256"]
        or value.get("stages") != [
            "01-adapter-interface-valid", "02-runtime-plan-built",
            "03-runtime-plan-rebuilt-exactly", "04-release-gate-valid",
        ]
        or value.get("claim_calls") != 0
        or value.get("model_requests") != 0
        or value.get("task_instance_session_verifier_scoring_calls") != 0
        or value.get("prompts_traces_flags_or_scores_read") is not False
        or value.get("receipt_sha256") != v17.RUNTIME_GATE["receipt_sha256"]
        or value.get("receipt_sha256")
        != self_hosted.digest_without(value, "receipt_sha256")
    ):
        raise RuntimeError("v18 runtime gate receipt drifted")
    return raw, value


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
    raw_bootstrap, bootstrap = _bootstrap()
    raw_runtime_gate, runtime_gate = _runtime_gate()
    gate = common.live_gate(Path.cwd(), v17, ALLOWED)
    gate["controller_bootstrap"] = {
        "path": v17.CONTROLLER_BOOTSTRAP["receipt_path"],
        "file_sha256": self_hosted.sha256(raw_bootstrap),
        "receipt_sha256": bootstrap["receipt_sha256"],
        "controller_package_sha256": bootstrap["controller_package_sha256"],
        "status": bootstrap["status"],
        "claim_calls": bootstrap["claim_calls"],
        "model_requests": bootstrap["model_requests"],
    }
    gate["runtime_gate"] = {
        "path": v17.RUNTIME_GATE["receipt_path"],
        "file_sha256": self_hosted.sha256(raw_runtime_gate),
        "receipt_sha256": runtime_gate["receipt_sha256"],
        "plan_sha256": runtime_gate["plan_sha256"],
        "status": runtime_gate["status"],
    }
    gate["receipt_sha256"] = self_hosted.digest_without(gate, "receipt_sha256")
    if args.command == "preview":
        shared._write_once(args.output, gate)
        print(json.dumps({"status": gate["status"], "receipt_sha256": gate["receipt_sha256"]}))
        return 0
    payload = v17.payload(v17.spec(Path.cwd()), Path.cwd())
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
        "schema_version": "fleet-glm53-dedicated-serving-v18-submission-v1",
        "status": "SUBMITTED",
        "submitted_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "api_run_id": api_run_id,
        "title": v17.TITLE,
        "run_dir": v17.RUN_DIR,
        "request_sha256": gate["request_sha256"],
        "live_gate_receipt_sha256": gate["receipt_sha256"],
        "pre_admission": gate["pre_admission"],
        "controller_bootstrap": gate["controller_bootstrap"],
        "runtime_gate": gate["runtime_gate"],
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
