"""Run one create-once OpenCode pass@4 plan with per-attempt acceptance receipts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import httpx

from evals.fleet import self_hosted

PLAN_SCHEMA = "fleet-selfhosted-opencode-pass4-plan-v1"


def _digest_without(value: dict[str, Any], field: str) -> str:
    return self_hosted.digest_without(value, field)


def validate_plan(plan: dict[str, Any]) -> None:
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("unsupported full-plan schema")
    if plan.get("plan_sha256") != _digest_without(plan, "plan_sha256"):
        raise ValueError("full-plan digest mismatch")
    task_count = int(plan.get("task_count") or 0)
    if task_count not in {50, 100} or plan.get("pass_k") != 4:
        raise ValueError("full-plan task/pass@k shape drifted")
    if len(plan.get("tasks") or []) != task_count:
        raise ValueError("full-plan task count drifted")
    if len(plan.get("attempts") or []) != task_count * 4 - 1:
        raise ValueError("full-plan attempt count drifted")
    if (plan.get("execution") or {}).get("max_concurrent") != 1:
        raise ValueError("full-plan concurrency must remain one per model")
    if (plan.get("execution") or {}).get("required_task_tools") != [
        "bash",
        "submit_report",
    ]:
        raise ValueError("full-plan tool contract drifted")


def _accepted_attempts(root: Path) -> dict[str, dict[str, Any]]:
    accepted: dict[str, dict[str, Any]] = {}
    attempts_root = root / "attempts"
    if not attempts_root.exists():
        return accepted
    for receipt_path in attempts_root.glob("*/ACCEPTED.json"):
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("receipt_sha256") != _digest_without(receipt, "receipt_sha256"):
            raise RuntimeError("attempt acceptance digest mismatch")
        run_id = receipt.get("run_id")
        if not isinstance(run_id, str) or run_id in accepted:
            raise RuntimeError("attempt acceptance identity is invalid or duplicated")
        accepted[run_id] = receipt
    return accepted


def _inventory_for_task(
    client: httpx.Client, task_key: str, persisted_model: str
) -> list[dict[str, Any]]:
    return [
        row
        for row in self_hosted._task_sessions(client, task_key)
        if row.get("model") == persisted_model
    ]


def _attempt_config(
    plan: dict[str, Any], task_row: dict[str, Any], attempt: dict[str, Any]
) -> dict[str, Any]:
    config = {
        "schema_version": "fleet-selfhosted-opencode-pass4-attempt-v1",
        "run_id": attempt["run_id"],
        "campaign_id": plan["campaign_id"],
        "source_job_id": plan["source_job_id"],
        "task": task_row["task"],
        "environment": task_row["environment"],
        "verifier": task_row["verifier"],
        "authority": plan["authority"],
        "model": plan["model"],
        "harness": plan["harness"],
        "execution": {
            "pass_k": 1,
            "planned_full_pass_k": 4,
            "max_concurrent": 1,
            "network": attempt["network"],
            "training_data_eligible": plan["execution"]["training_data_eligible"],
            "required_task_tools": plan["execution"]["required_task_tools"],
            "required_task_tool_catalog_sha256": plan["execution"][
                "required_task_tool_catalog_sha256"
            ],
        },
    }
    config["config_sha256"] = _digest_without(config, "config_sha256")
    return config


def _accept_attempt(out_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    result = json.loads((out_dir / "result.json").read_text())
    cleanup = json.loads((out_dir / "cleanup.json").read_text())
    ingest = json.loads((out_dir / "session-ingest.json").read_text())
    if any(
        (
            result.get("run_id") != config["run_id"],
            result.get("agent_exit_code") != 0,
            result.get("agent_termination") != "completed",
            result.get("session_ingest_status") != "completed",
            ingest.get("status") != "completed",
            cleanup
            != {
                "instance_created": True,
                "instance_closed": True,
                "containers_removed": True,
            },
        )
    ):
        raise RuntimeError("attempt is not an accepted valid model outcome")
    receipt = {
        "schema_version": "fleet-selfhosted-opencode-pass4-attempt-accepted-v1",
        "accepted": True,
        "run_id": config["run_id"],
        "config_sha256": config["config_sha256"],
        "session_id": result["session_id"],
        "verifier_execution_id": result["verifier_execution_id"],
        "cleanup_completed": True,
        "score_persisted_privately": True,
        "scores_included": False,
        "prompts_or_traces_included": False,
    }
    receipt["receipt_sha256"] = _digest_without(receipt, "receipt_sha256")
    self_hosted.write_json_once(out_dir / "ACCEPTED.json", receipt)
    return receipt


def run_plan(plan: dict[str, Any], root: Path, proxy_script: Path) -> dict[str, Any]:
    validate_plan(plan)
    root.mkdir(parents=True, exist_ok=False)
    root.chmod(0o700)
    (root / "attempts").mkdir(mode=0o700)
    self_hosted.write_json_once(root / "PLAN.json", plan)
    key = os.environ.get("FLEET_API_KEY")
    if not key:
        raise RuntimeError("FLEET_API_KEY is required")
    task_by_rank = {int(row["rank"]): row for row in plan["tasks"]}
    persisted_model = self_hosted.persisted_session_model_identity(plan)
    with httpx.Client(
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        timeout=1800,
    ) as client:
        account = self_hosted._request(client, "GET", "/v1/account")
        if (
            account.get("team_name") != "fleet"
            or account.get("team_id") != self_hosted.FLEET_TEAM_ID
        ):
            raise RuntimeError("FLEET_API_KEY is not scoped to the Fleet team")
        for item in plan["attempts"]:
            task_row = task_by_rank[int(item["rank"])]
            accepted = _accepted_attempts(root)
            allowed = set(task_row["baseline_session_ids"])
            allowed.update(
                receipt["session_id"]
                for receipt in accepted.values()
                if receipt["run_id"] != item["run_id"]
            )
            observed = _inventory_for_task(client, task_row["task"]["key"], persisted_model)
            unknown = {
                row.get("session_id")
                for row in observed
                if isinstance(row.get("session_id"), str) and row.get("session_id") not in allowed
            }
            if unknown:
                raise RuntimeError("unexpected equivalent-treatment session appeared; halting")
            config = _attempt_config(plan, task_row, item)
            out_dir = root / "attempts" / item["run_id"]
            if out_dir.exists():
                raise RuntimeError("planned attempt output already exists; refusing duplicate")
            self_hosted.run(config, out_dir, proxy_script)
            _accept_attempt(out_dir, config)
            progress = {
                "schema_version": "fleet-selfhosted-opencode-pass4-progress-v1",
                "plan_sha256": plan["plan_sha256"],
                "accepted_new_sessions": len(_accepted_attempts(root)),
                "planned_new_sessions": plan["new_session_count"],
                "last_run_id": item["run_id"],
                "scores_included": False,
            }
            progress["receipt_sha256"] = _digest_without(progress, "receipt_sha256")
            progress_dir = root / "progress"
            progress_dir.mkdir(exist_ok=True)
            self_hosted.write_json_once(
                progress_dir / f"{item['ordinal']:04d}.json", progress
            )
    final = {
        "schema_version": "fleet-selfhosted-opencode-pass4-accepted-v1",
        "accepted": True,
        "plan_sha256": plan["plan_sha256"],
        "credited_smoke_sessions": 1,
        "accepted_new_sessions": len(_accepted_attempts(root)),
        "total_sessions": plan["total_session_count"],
        "scores_included": False,
        "prompts_or_traces_included": False,
    }
    if final["accepted_new_sessions"] != plan["new_session_count"]:
        raise RuntimeError("terminal accepted-session count drifted")
    final["receipt_sha256"] = _digest_without(final, "receipt_sha256")
    self_hosted.write_json_once(root / "ACCEPTED.json", final)
    return final


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--proxy-script", type=Path, required=True)
    args = parser.parse_args()
    result = run_plan(json.loads(args.plan.read_text()), args.out_dir, args.proxy_script)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
