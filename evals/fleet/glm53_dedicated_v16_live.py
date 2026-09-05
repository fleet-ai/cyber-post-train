"""Create-once Jobs API rail for corrected GLM v16."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import httpx

from evals.fleet import glm53_dedicated_v8_live as shared
from evals.fleet import glm53_dedicated_v15_live as common
from evals.fleet import glm53_dedicated_v16 as v16
from evals.fleet import self_hosted

ALLOWED = {
    "/mnt/sfs/jobs/chris-cyber-evalserve-q38-tp1-g-v1": {"nodes": 1, "gpus": 1}
}


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
    gate = common.live_gate(Path.cwd(), v16, ALLOWED)
    if args.command == "preview":
        shared._write_once(args.output, gate)
        print(json.dumps({"status": gate["status"], "receipt_sha256": gate["receipt_sha256"]}))
        return 0
    payload = v16.payload(v16.spec(Path.cwd()), Path.cwd())
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
        "schema_version": "fleet-glm53-dedicated-serving-v16-submission-v1",
        "status": "SUBMITTED",
        "submitted_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "api_run_id": api_run_id,
        "title": v16.TITLE,
        "run_dir": v16.RUN_DIR,
        "request_sha256": gate["request_sha256"],
        "live_gate_receipt_sha256": gate["receipt_sha256"],
        "pre_admission": gate["pre_admission"],
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
