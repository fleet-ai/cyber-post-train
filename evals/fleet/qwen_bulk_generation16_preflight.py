"""Content-free SFS and Fleet-session duplicate preflight for Qwen G16."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from evals.fleet import qwen_bulk_generation16 as bulk
from evals.fleet import qwen_bulk_generation16_runtime as runtime
from evals.fleet import self_hosted

SCHEMA = "fleet-qwen-generation16-preflight-v1"


def _stage(output: Path, ordinal: int, name: str) -> None:
    body = {
        "schema_version": "fleet-qwen-generation16-preflight-stage-v1",
        "ordinal": ordinal,
        "name": name,
        "prompts_traces_flags_or_scores_included": False,
    }
    self_hosted.write_json_once(
        output.parent / f"STAGE-{ordinal:02d}.json",
        {**body, "receipt_sha256": self_hosted.digest_without(body, "receipt_sha256")},
    )


def run(plan_path: Path, output: Path) -> dict[str, Any]:
    _stage(output, 1, "started")
    raw = plan_path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict) or raw != self_hosted.canonical_json(payload) + b"\n":
        raise RuntimeError("Generation-16 preflight envelope drifted")
    plans = payload.get("plans")
    if not isinstance(plans, list) or len(plans) != 2:
        raise RuntimeError("Generation-16 preflight plan envelope drifted")
    rows = [row for plan in plans for row in plan.get("attempts", [])]
    if len(rows) != 395 or len({row["execution_id"] for row in rows}) != 395:
        raise RuntimeError("Generation-16 preflight cell universe drifted")
    output_roots = [Path(plan["sfs_root"]) for plan in plans]
    claims = [Path(bulk.CLAIM_ROOT) / row["execution_id"] for row in rows]
    if any(path.exists() for path in [*output_roots, *claims]):
        raise RuntimeError("Generation-16 SFS output or execution claim collision")
    _stage(output, 2, "sfs-clear")

    key = os.environ.get("FLEET_API_KEY")
    if not key:
        raise RuntimeError("FLEET_API_KEY is required")
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[row["task_key"]].append(row)
    expected_by_field = {
        field: {row[field] for row in rows} for field in ("run_id", "execution_id", "cell_id")
    }
    collisions = 0
    accepted_gate = bulk.load(Path(payload["g15_gate_path"]))
    runtime.validate_g15_gate(accepted_gate)
    accepted_matches: list[dict[str, Any]] = []
    request_count = 0
    with httpx.Client(headers=headers, timeout=1800) as client:
        account = self_hosted._request(client, "GET", "/v1/account")  # noqa: SLF001
        request_count += 1
        if (
            account.get("team_name") != "fleet"
            or account.get("team_id") != self_hosted.FLEET_TEAM_ID
        ):
            raise RuntimeError("Generation-16 Fleet team authority drifted")
        _stage(output, 3, "account-valid")
        for task_key in sorted(by_task):
            sessions = self_hosted._task_sessions(client, task_key)  # noqa: SLF001
            request_count += 1
            for session in sessions:
                metadata = session.get("metadata") or {}
                if any(
                    metadata.get(field) in values
                    for field, values in expected_by_field.items()
                ):
                    collisions += 1
                if session.get("session_id") == accepted_gate["api_session"]["session_id"]:
                    accepted_matches.append(session)
        _stage(output, 4, "session-inventory-complete")
        roster_response = client.get("https://inference.flt.build/v1/models")
        request_count += 1
        roster_response.raise_for_status()
        roster = roster_response.json()
    selected = [
        row
        for row in roster.get("data", [])
        if isinstance(row, dict) and row.get("id") == "qwen3.8-27b"
    ]
    if len(selected) != 1 or collisions != 0 or len(accepted_matches) != 1:
        raise RuntimeError("Generation-16 live duplicate or route gate failed")
    _stage(output, 5, "route-and-collision-gates-clear")
    accepted = accepted_matches[0]
    projected_model = accepted.get("model")
    verifier = accepted.get("verifier_execution") or {}
    if (
        projected_model not in (None, "qwen3.8-27b")
        or verifier.get("id") != accepted_gate["api_session"]["verifier_execution_id"]
    ):
        raise RuntimeError("Generation-15 accepted session authority contradicted")
    body = {
        "schema_version": SCHEMA,
        "status": "CLEAR",
        "planned_cells": 395,
        "planned_execution_ids_sha256": self_hosted.sha256(
            self_hosted.canonical_json(sorted(row["execution_id"] for row in rows))
        ),
        "planned_run_ids_sha256": self_hosted.sha256(
            self_hosted.canonical_json(sorted(row["run_id"] for row in rows))
        ),
        "task_keys_checked": len(by_task),
        "fleet_requests": request_count,
        "api_identity_collisions": collisions,
        "sfs_output_collisions": 0,
        "global_claim_collisions": 0,
        "g15_accepted_session_present": True,
        "g15_session_model_projection": "omitted" if projected_model is None else "matched",
        "qwen_hosted_route_present": True,
        "checked_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mutation_calls": 0,
        "prompts_traces_flags_or_scores_included": False,
        "credentials_included": False,
    }
    receipt = {**body, "receipt_sha256": self_hosted.digest_without(body, "receipt_sha256")}
    self_hosted.write_json_once(output, receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.plan, args.output)
    except Exception as exc:
        body = {
            "schema_version": "fleet-qwen-generation16-preflight-failure-v1",
            "status": "FAILED",
            "error_type": type(exc).__name__,
            "error_sha256": self_hosted.sha256(str(exc).encode()),
            "prompts_traces_flags_or_scores_included": False,
            "credentials_included": False,
        }
        self_hosted.write_json_once(
            args.output.parent / "FAILED.json",
            {**body, "receipt_sha256": self_hosted.digest_without(body, "receipt_sha256")},
        )
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
