from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).parents[1]
AUDIT_PATH = ROOT / "docs/evidence/qwen38-web-important-checkpoint-budget-audit-20260921.json"


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def test_web_campaign_budget_audit_binds_the_actual_outer_timeout() -> None:
    audit = json.loads(AUDIT_PATH.read_bytes())
    unsigned = dict(audit)
    unsigned.pop("receipt_sha256")
    assert audit["receipt_sha256"] == ("sha256:" + hashlib.sha256(_canonical(unsigned)).hexdigest())

    source = audit["source"]
    plan_path = ROOT / source["campaign_plan_path"]
    runner_path = ROOT / source["collection_runner_path"]
    assert _sha256(plan_path) == source["campaign_plan_file_sha256"]
    assert _sha256(runner_path) == source["collection_runner_file_sha256"]

    plan = json.loads(plan_path.read_bytes())
    plan_unsigned = dict(plan)
    plan_unsigned.pop("sha256")
    assert plan["sha256"] == ("sha256:" + hashlib.sha256(_canonical(plan_unsigned)).hexdigest())
    assert plan["sha256"] == source["campaign_plan_sha256"]

    matches = re.findall(
        r"^readonly collection_dispatch_timeout_seconds=([0-9]+)$",
        runner_path.read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    )
    assert len(matches) == 1
    outer_limit = int(matches[0])

    budget = plan["protocol"]["budget"]
    observed = audit["observed_budget_seconds"]
    declared_task = budget["task_max_duration_minutes"] * 60
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


def test_web_campaign_budget_audit_contains_no_credentials_or_results() -> None:
    text = AUDIT_PATH.read_text(encoding="utf-8").lower()
    assert "api_key" not in text
    assert "tensorlake-api-key" not in text
    assert "score_value" not in text
    assert "rollout_trace" not in text
