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
SPLIT_PROBE_REPAIR_AUDIT_PATH = (
    ROOT / "docs/evidence/qwen38-web-important-checkpoint-budget-repair-split-probe-20260921.json"
)
NETPROXY_REPAIR_AUDIT_PATH = (
    ROOT
    / "docs/evidence/qwen38-web-important-checkpoint-budget-repair-netproxy-contract-20260921.json"
)
CURRENT_REPAIR_AUDIT_PATH = (
    ROOT
    / "docs/evidence/qwen38-web-important-checkpoint-budget-repair-prompt-preflight-20260921.json"
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


def test_historical_repaired_budget_receipt_remains_byte_bound_to_its_runner() -> None:
    audit = json.loads(REPAIR_AUDIT_PATH.read_bytes())
    unsigned = dict(audit)
    unsigned.pop("receipt_sha256")
    assert audit["receipt_sha256"] == ("sha256:" + hashlib.sha256(_canonical(unsigned)).hexdigest())
    assert audit["source"]["collection_runner_file_sha256"] == (
        "sha256:b1462f13a3be437d42e818c5ab943b11c71e01a8c8d0a602d4f7ff67dbdbe65c"
    )
    assert audit["classification"] == "runtime_budget_repaired_campaign_still_gated"
    assert audit["checks"]["model_free_runtime_qualification_still_required"] is True


def test_historical_split_probe_receipt_remains_byte_bound_to_its_runner() -> None:
    audit = json.loads(SPLIT_PROBE_REPAIR_AUDIT_PATH.read_bytes())
    unsigned = dict(audit)
    unsigned.pop("receipt_sha256")
    assert audit["receipt_sha256"] == ("sha256:" + hashlib.sha256(_canonical(unsigned)).hexdigest())
    assert _sha256(SPLIT_PROBE_REPAIR_AUDIT_PATH) == (
        "sha256:1a94875b9802556be9e2a048fcd0073ecfc75fdcf03ba3d12981b042e3a49782"
    )
    assert audit["source"]["collection_runner_file_sha256"] == (
        "sha256:bb6a2d586e53eec9e8908e57d73aa8af74ca8ca2506036b194aa87b47e0a396e"
    )
    assert audit["classification"] == (
        "runtime_budget_repaired_split_identity_probe_campaign_still_gated"
    )
    assert audit["checks"]["fresh_model_free_runtime_qualification_still_required"] is True


def test_historical_netproxy_receipt_remains_byte_bound_to_its_sources() -> None:
    audit = json.loads(NETPROXY_REPAIR_AUDIT_PATH.read_bytes())
    unsigned = dict(audit)
    unsigned.pop("receipt_sha256")
    assert audit["receipt_sha256"] == ("sha256:" + hashlib.sha256(_canonical(unsigned)).hexdigest())
    assert _sha256(NETPROXY_REPAIR_AUDIT_PATH) == (
        "sha256:9cb3ffc3170098d2bd0e966cd0b12c9f8a8a8142bd0750d23d1db4eb15b714dc"
    )
    assert audit["source"]["qualification_worker_file_sha256"] == (
        "sha256:33bdd8cc83047a3f3fd75b9907e8d3d958a750e23f24c1b3cfd9df6126b4f613"
    )
    assert audit["classification"] == (
        "runtime_budget_repaired_netproxy_contract_campaign_still_gated"
    )


def test_current_repaired_web_campaign_budget_binds_runner_supervisor_and_sandbox() -> None:
    audit = json.loads(CURRENT_REPAIR_AUDIT_PATH.read_bytes())
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
            "qualification_worker",
            "qualification_execution_packet",
        )
    }
    for name, path in paths.items():
        assert _sha256(path) == source[f"{name}_file_sha256"]
    assert (
        _sha256(ROOT / source["historical_netproxy_audit_path"])
        == source["historical_netproxy_audit_file_sha256"]
    )
    assert (
        _sha256(ROOT / source["runtime_failure_evidence_path"])
        == source["runtime_failure_evidence_file_sha256"]
    )

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
    assert (
        audit["classification"]
        == "runtime_budget_repaired_prompt_preflight_parity_campaign_still_gated"
    )
    assert audit["checks"]["prompt_runtime_dependency_probe_bound_to_current_source"] is True
    assert audit["checks"]["prompt_render_before_netproxy_bound_to_current_source"] is True
    assert audit["checks"]["non_strict_prompt_receipt_validation_bound_to_current_source"] is True
    assert (
        audit["checks"]["qualification_netproxy_network_restoration_bound_to_current_source"]
        is True
    )
    assert audit["checks"]["isolated_release_reconciliation_import_bound_to_current_source"] is True
    assert audit["checks"]["historical_netproxy_receipt_preserved"] is True
    assert audit["checks"]["fresh_model_free_runtime_qualification_still_required"] is True
    assert audit["operation"] == {
        "provider_requests": 0,
        "model_requests": 0,
        "scoring_requests": 0,
        "external_mutations": 0,
    }


def test_web_campaign_budget_audit_contains_no_credentials_or_results() -> None:
    text = "\n".join(
        path.read_text(encoding="utf-8").lower()
        for path in (
            AUDIT_PATH,
            REPAIR_AUDIT_PATH,
            SPLIT_PROBE_REPAIR_AUDIT_PATH,
            CURRENT_REPAIR_AUDIT_PATH,
        )
    )
    assert "api_key" not in text
    assert "tensorlake-api-key" not in text
    assert "score_value" not in text
    assert "rollout_trace" not in text
