"""Score-blind release preflight for the two held Qwen G19 shards."""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from evals.fleet import exact_pass4_bulk_runtime_v3 as engine
from evals.fleet import qwen_bulk_generation16_preflight as inventory
from evals.fleet import qwen_hosted_generation18 as g18
from evals.fleet import qwen_hosted_generation19_bulk as g19
from evals.fleet import self_hosted


def run(output: Path) -> dict[str, Any]:
    plans = g19.validate_all(Path.cwd())
    rows = [row for plan in plans.values() for row in plan["attempts"]]
    if len(rows) != 384 or any(row["selection_rank"] in {2, 3, 4, 5} for row in rows):
        raise RuntimeError("Generation-19 held partition drifted")
    output_roots = [Path(plan["sfs_root"]) for plan in plans.values()]
    claims = [Path(g18.CLAIM_ROOT) / engine.claim_filename(row["execution_id"]) for row in rows]
    if any(path.exists() for path in [*output_roots, *claims]):
        raise RuntimeError("Generation-19 SFS output or claim collision")
    accepted = g18.load(
        Path("/mnt/sfs/jobs/chris-q38-ac-r005-a1-g18-v1/accepted/")
        / "chris-q38-ac-g18-r005-a1-deb44299.json"
    )
    terminal = g18.load(Path("/mnt/sfs/jobs/chris-q38-ac-r005-a1-g18-v1/TERMINAL.json"))
    if any(
        (
            accepted.get("receipt_sha256")
            != "sha256:083deadd604e090bb43fc4455fed032e763119e8e89ce4bae7632e02ef99eb30",
            accepted.get("accepted") is not True,
            accepted.get("credited") is not True,
            accepted.get("retry_allowed") is not False,
            accepted.get("cell_id")
            != "sha256:65143e0c71a7af857878a94c466ee4e9a3fe063c8af77df29ab6cdadb9429334",
            terminal.get("receipt_sha256")
            != "sha256:1f8be7be188789210d1561b1aa026a0906a8e5ff85230bdf58b987ef120d84c9",
            terminal.get("status") != "COMPLETE",
            terminal.get("accepted_cells") != 1,
            terminal.get("endpoint_lease_released") is not True,
        )
    ):
        raise RuntimeError("Generation-18 acceptance gate drifted")
    key = os.environ.get("FLEET_API_KEY")
    if not key:
        raise RuntimeError("FLEET_API_KEY required")
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[row["task_key"]].append(row)
    expected = {
        field: {row[field] for row in rows} for field in ("run_id", "execution_id", "cell_id")
    }
    with httpx.Client(headers=headers, timeout=30) as client:
        account = self_hosted._request(client, "GET", "/v1/account")  # noqa: SLF001
        roster_response = client.get("https://inference.flt.build/v1/models")
        roster_response.raise_for_status()
        roster = roster_response.json()
    sessions, requests = inventory.collect_task_sessions(sorted(by_task), headers)
    collisions = [
        row
        for row in sessions
        if any(
            (row.get("metadata") or {}).get(field) in values for field, values in expected.items()
        )
    ]
    models = roster.get("data") if isinstance(roster, dict) else None
    route = [
        row for row in models or [] if isinstance(row, dict) and row.get("id") == "qwen3.8-27b"
    ]
    if (
        account.get("team_name") != "fleet"
        or account.get("team_id") != self_hosted.FLEET_TEAM_ID
        or collisions
        or len(route) != 1
    ):
        raise RuntimeError("Generation-19 account, duplicate, or route gate failed")
    execution_ids = sorted(row["execution_id"] for row in rows)
    body = {
        "schema_version": "fleet-qwen-generation19-hosted-bulk-preflight-v1",
        "status": "CLEAR",
        "planned_cells": 384,
        "planned_tasks": 96,
        "planned_execution_ids_sha256": self_hosted.sha256(
            self_hosted.canonical_json(execution_ids)
        ),
        "g18_accepted_receipt_sha256": accepted["receipt_sha256"],
        "g18_terminal_receipt_sha256": terminal["receipt_sha256"],
        "task_session_rows_checked": len(sessions),
        "session_get_requests": requests,
        "api_identity_collisions": 0,
        "sfs_output_collisions": 0,
        "global_claim_collisions": 0,
        "qwen_hosted_route_present": True,
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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
