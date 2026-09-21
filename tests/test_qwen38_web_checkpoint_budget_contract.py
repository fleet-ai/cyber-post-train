from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).parents[1]
AUDIT_PATH = ROOT / "docs/evidence/qwen38-web-important-checkpoint-budget-audit-20260921.json"
REPAIR_AUDIT_PATH = (
    ROOT / "docs/evidence/qwen38-web-important-checkpoint-budget-repair-20260921.json"
)


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def test_historical_web_campaign_budget_audit_preserves_the_old_mismatch() -> None:
    audit = json.loads(AUDIT_PATH.read_bytes())
    unsigned = dict(audit)
    unsigned.pop("receipt_sha256")
    assert audit["receipt_sha256"] == ("sha256:" + hashlib.sha256(_canonical(unsigned)).hexdigest())

    source = audit["source"]
    plan_path = ROOT / source["campaign_plan_path"]
    assert _sha256(plan_path) == source["campaign_plan_file_sha256"]
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", source["collection_runner_file_sha256"])

    plan = json.loads(plan_path.read_bytes())
    plan_unsigned = dict(plan)
    plan_unsigned.pop("sha256")
    assert plan["sha256"] == ("sha256:" + hashlib.sha256(_canonical(plan_unsigned)).hexdigest())
    assert plan["sha256"] == source["campaign_plan_sha256"]

    budget = plan["protocol"]["budget"]
    observed = audit["observed_budget_seconds"]
    declared_task = budget["task_max_duration_minutes"] * 60
    outer_limit = observed["actual_outer_collection_dispatch_limit"]
    assert observed == {
        "declared_task_max": declared_task,
        "actual_outer_collection_dispatch_limit": outer_limit,
        "snapshot_restore_and_readiness_allowance": budget[
            "snapshot_restore_and_readiness_allowance_seconds"
        ],
        "collection_and_preservation_allowance": budget[
            "collection_and_preservation_allowance_seconds"
        ],
        "sandbox_lifetime": budget["sandbox_timeout_seconds"],
    }

    checks = audit["checks"]
    assert checks["sandbox_covers_declared_task_and_preservation"] is (
        observed["sandbox_lifetime"]
        >= declared_task
        + observed["snapshot_restore_and_readiness_allowance"]
        + observed["collection_and_preservation_allowance"]
    )
    assert checks["outer_collection_dispatch_covers_declared_task"] is (
        outer_limit >= declared_task
    )
    assert checks["campaign_is_marked_launchable"] is plan["execution"]["launchable_now"]

    # This receipt deliberately freezes the current mismatch. If the runner or
    # campaign budget changes, the file-digest checks above force a fresh audit
    # instead of silently treating the old preparation as launchable.
    assert outer_limit < declared_task
    assert plan["execution"]["launchable_now"] is False
    assert audit["classification"] == "operational_gate_incomplete"
    assert audit["operation"] == {
        "provider_requests": 0,
        "model_requests": 0,
        "scoring_requests": 0,
        "external_mutations": 0,
    }


def test_repaired_web_campaign_budget_binds_runner_supervisor_and_sandbox() -> None:
    audit = json.loads(REPAIR_AUDIT_PATH.read_bytes())
    unsigned = dict(audit)
    unsigned.pop("receipt_sha256")
    assert audit["receipt_sha256"] == ("sha256:" + hashlib.sha256(_canonical(unsigned)).hexdigest())

    source = audit["source"]
    paths = {
        name: ROOT / source[f"{name}_path"]
        for name in (
            "campaign_plan",
            "collection_runner",
            "collection_launcher",
            "collection_supervisor",
        )
    }
    for name, path in paths.items():
        assert _sha256(path) == source[f"{name}_file_sha256"]

    plan = json.loads(paths["campaign_plan"].read_bytes())
    plan_unsigned = dict(plan)
    plan_unsigned.pop("sha256")
    assert plan["sha256"] == ("sha256:" + hashlib.sha256(_canonical(plan_unsigned)).hexdigest())
    assert plan["sha256"] == source["campaign_plan_sha256"]

    runner_source = paths["collection_runner"].read_text(encoding="utf-8")
    runner_values = {
        name: int(match.group(1))
        for name in (
            "runner_pre_collection_budget_seconds",
            "collection_dispatch_timeout_seconds",
            "collection_export_timeout_seconds",
            "collection_acceptance_timeout_seconds",
            "runner_shutdown_reserve_seconds",
        )
        if (match := re.search(rf"^readonly {name}=([0-9]+)$", runner_source, flags=re.MULTILINE))
    }
    assert len(runner_values) == 5
    supervisor_source = paths["collection_supervisor"].read_text(encoding="utf-8")
    poll_interval = int(
        re.search(
            r"^SUPERVISOR_POLL_INTERVAL_SECONDS = ([0-9]+)$",
            supervisor_source,
            flags=re.MULTILINE,
        ).group(1)
    )
    poll_cycles = int(
        re.search(
            r"^SUPERVISOR_POLL_MAX_CYCLES = ([0-9]+)$",
            supervisor_source,
            flags=re.MULTILINE,
        ).group(1)
    )

    observed = audit["observed_budget_seconds"]
    task_budget = plan["protocol"]["budget"]["task_max_duration_minutes"] * 60
    preservation = (
        runner_values["collection_export_timeout_seconds"]
        + runner_values["collection_acceptance_timeout_seconds"]
    )
    runner_envelope = (
        runner_values["runner_pre_collection_budget_seconds"]
        + runner_values["collection_dispatch_timeout_seconds"]
        + preservation
        + runner_values["runner_shutdown_reserve_seconds"]
    )
    assert observed == {
        "declared_task_max": task_budget,
        "actual_outer_collection_dispatch_limit": runner_values[
            "collection_dispatch_timeout_seconds"
        ],
        "runner_pre_collection_budget": runner_values["runner_pre_collection_budget_seconds"],
        "runner_preservation_budget": preservation,
        "runner_shutdown_reserve": runner_values["runner_shutdown_reserve_seconds"],
        "runner_envelope": runner_envelope,
        "supervisor_poll_interval": poll_interval,
        "supervisor_poll_max_cycles": poll_cycles,
        "supervisor_poll_horizon": poll_interval * poll_cycles,
        "snapshot_restore_and_readiness_allowance": plan["protocol"]["budget"][
            "snapshot_restore_and_readiness_allowance_seconds"
        ],
        "campaign_collection_and_preservation_allowance": plan["protocol"]["budget"][
            "collection_and_preservation_allowance_seconds"
        ],
        "sandbox_lifetime": plan["protocol"]["budget"]["sandbox_timeout_seconds"],
    }
    assert task_budget <= observed["actual_outer_collection_dispatch_limit"]
    assert observed["actual_outer_collection_dispatch_limit"] < runner_envelope
    assert runner_envelope + 1800 <= observed["supervisor_poll_horizon"]
    assert observed["supervisor_poll_horizon"] <= observed["sandbox_lifetime"]
    assert plan["execution"]["launchable_now"] is False
    assert audit["classification"] == "runtime_budget_repaired_campaign_still_gated"
    assert audit["operation"] == {
        "provider_requests": 0,
        "model_requests": 0,
        "scoring_requests": 0,
        "external_mutations": 0,
    }


def test_web_campaign_budget_audit_contains_no_credentials_or_results() -> None:
    text = "\n".join(
        path.read_text(encoding="utf-8").lower() for path in (AUDIT_PATH, REPAIR_AUDIT_PATH)
    )
    assert "api_key" not in text
    assert "tensorlake-api-key" not in text
    assert "score_value" not in text
    assert "rollout_trace" not in text
