"""Read-only live duplicate and route preflight for the Qwen G18 canary."""

from __future__ import annotations

import argparse
import os
from datetime import UTC, datetime
from pathlib import Path

import httpx

from evals.fleet import exact_pass4_bulk_runtime_v3 as engine
from evals.fleet import qwen_hosted_generation18 as g18
from evals.fleet import self_hosted


def run(plan_path: Path, parity_path: Path, output: Path) -> dict:
    plan = g18.load(plan_path)
    g18.validate_plan(plan)
    g18.validate_parity(g18.load(parity_path))
    item = plan["attempts"][0]
    claim = Path(g18.CLAIM_ROOT) / engine.claim_filename(item["execution_id"])
    if Path(plan["sfs_root"]).exists() or claim.exists():
        raise RuntimeError("Generation-18 output or claim already exists")
    key = os.environ.get("FLEET_API_KEY")
    if not key:
        raise RuntimeError("FLEET_API_KEY required")
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    with httpx.Client(headers=headers, timeout=180) as client:
        account = self_hosted._request(client, "GET", "/v1/account")  # noqa: SLF001
        sessions = self_hosted._task_sessions(client, item["task_key"])  # noqa: SLF001
        roster_response = client.get("https://inference.flt.build/v1/models")
        roster_response.raise_for_status()
        roster = roster_response.json()
    if account.get("team_name") != "fleet" or account.get("team_id") != self_hosted.FLEET_TEAM_ID:
        raise RuntimeError("Fleet account binding drifted")
    expected = {
        "run_id": item["run_id"],
        "execution_id": item["execution_id"],
        "cell_id": item["cell_id"],
    }
    collisions = [
        row
        for row in sessions
        if any((row.get("metadata") or {}).get(field) == value for field, value in expected.items())
    ]
    models = roster.get("data") if isinstance(roster, dict) else None
    selected = [
        row for row in models or [] if isinstance(row, dict) and row.get("id") == "qwen3.8-27b"
    ]
    if collisions or len(selected) != 1:
        raise RuntimeError("Generation-18 live duplicate or route gate failed")
    body = {
        "schema_version": "fleet-qwen-generation18-hosted-canary-preflight-v1",
        "status": "CLEAR",
        "plan_sha256": plan["plan_sha256"],
        "cell_id": item["cell_id"],
        "execution_id": item["execution_id"],
        "task_key": item["task_key"],
        "task_session_rows_checked": len(sessions),
        "api_identity_collisions": 0,
        "sfs_output_collisions": 0,
        "global_claim_collisions": 0,
        "qwen_hosted_route_present": True,
        "actual_opencode_parity_receipt_sha256": g18.load(parity_path)["receipt_sha256"],
        "mutation_calls": 0,
        "checked_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "prompts_traces_flags_or_scores_included": False,
        "credentials_included": False,
    }
    receipt = {**body, "receipt_sha256": self_hosted.digest_without(body, "receipt_sha256")}
    self_hosted.write_json_once(output, receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--parity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.plan, args.parity, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
