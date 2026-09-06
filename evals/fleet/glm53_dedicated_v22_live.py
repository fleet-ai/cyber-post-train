"""Create-once Jobs API rail for generation-qualified GLM v22."""

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
from evals.fleet import glm53_dedicated_v21_live as v21_live
from evals.fleet import glm53_dedicated_v22 as v22
from evals.fleet import glm53_dedicated_v22_generation_consistency_v1 as consistency
from evals.fleet import self_hosted

ALLOWED = {
    "/mnt/sfs/jobs/chris-cyber-evalserve-q38-tp1-j-v1": {"nodes": 1, "gpus": 1},
}


def _generation_qualification(root: Path) -> dict:
    expected = v22.GENERATION_QUALIFICATION
    raw = (root / expected["path"]).read_bytes()
    if self_hosted.sha256(raw) != expected["file_sha256"]:
        raise RuntimeError("v22 generation qualification file digest drifted")
    value = json.loads(raw)
    required = {
        "schema_version": consistency.CANARY_SCHEMA,
        "status": "PASSED_FULL_GENERATION_BINDING_PATH",
        "generation": 22,
        "job_uid": expected["job_uid"],
        "pod_uid": expected["pod_uid"],
        "server_title": v22.TITLE,
        "server_run_dir": v22.RUN_DIR,
        "model_requests": 0,
        "claims_sessions_verifier_or_scoring_calls": 0,
        "single_generation_valid": True,
        "projected_configmap_roundtrip_valid": True,
        "prompts_traces_flags_scores_or_model_outputs_included": False,
    }
    if any(value.get(key) != wanted for key, wanted in required.items()):
        raise RuntimeError("v22 generation qualification fields drifted")
    if value.get("receipt_sha256") != expected["receipt_sha256"] or value.get(
        "receipt_sha256"
    ) != self_hosted.digest_without(value, "receipt_sha256"):
        raise RuntimeError("v22 generation qualification self digest drifted")
    return value


def live_gate(root: Path) -> dict:
    generation = _generation_qualification(root)
    canary_raw, canary = v21_live._package_canary()
    watchdog_raw, watchdog = watchdog_authority._watchdog()
    gate = common.live_gate(root, v22, ALLOWED)
    gate.pop("pre_admission", None)
    gate["generation_qualification"] = {
        "path": v22.GENERATION_QUALIFICATION["path"],
        "file_sha256": v22.GENERATION_QUALIFICATION["file_sha256"],
        "receipt_sha256": generation["receipt_sha256"],
        "status": generation["status"],
    }
    gate["watchdog_qualification"] = {
        "path": v22.WATCHDOG_QUALIFICATION["receipt_path"],
        "file_sha256": self_hosted.sha256(watchdog_raw),
        "receipt_sha256": watchdog["receipt_sha256"],
        "lifecycle_file_sha256": watchdog["lifecycle_file_sha256"],
        "status": watchdog["status"],
    }
    gate["package_canary"] = {
        "path": v22.PACKAGE_CANARY["receipt_path"],
        "file_sha256": self_hosted.sha256(canary_raw),
        "receipt_sha256": canary["receipt_sha256"],
        "controller_package_sha256": canary["controller_package_sha256"],
        "runtime_plan_sha256": canary["runtime_plan_sha256"],
        "status": canary["status"],
    }
    gate["scored_create_gate"] = {
        "status": "CLOSED",
        "fresh_uid_bound_parity_required": True,
        "fresh_v22_server_bound_release_required": True,
        "fresh_v22_server_bound_preclaim_required": True,
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
    if (
        args.command == "submit"
        and subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    ):
        raise RuntimeError("submit requires clean worktree")
    gate = live_gate(Path.cwd())
    if args.command == "preview":
        shared._write_once(args.output, gate)
        return 0
    payload = v22.payload(v22.spec(Path.cwd()), Path.cwd())
    with httpx.Client(
        base_url=shared.BASE_URL,
        headers={"Authorization": f"Bearer {shared._token()}", "Accept": "application/json"},
        timeout=60,
    ) as client:
        response = client.post("/v1/runs", json=payload)
        response.raise_for_status()
        api_run_id = response.json().get("name")
    if (
        response.status_code != 202
        or not isinstance(api_run_id, str)
        or not api_run_id.startswith("ft-run-")
    ):
        raise RuntimeError("v22 Jobs API identity absent")
    receipt = {
        "schema_version": "fleet-glm53-dedicated-serving-v22-submission-v1",
        "status": "SUBMITTED",
        "submitted_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "api_run_id": api_run_id,
        "title": v22.TITLE,
        "run_dir": v22.RUN_DIR,
        "request_sha256": gate["request_sha256"],
        "live_gate_receipt_sha256": gate["receipt_sha256"],
        "generation_qualification": gate["generation_qualification"],
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
