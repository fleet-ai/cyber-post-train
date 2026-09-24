import hashlib
import json
from pathlib import Path

EVIDENCE = (
    Path(__file__).parents[1]
    / "docs/evidence/qwen38-study/2026-09-24-q38-step1000-weight-drift-a2-success.json"
)


def _digest(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_step1000_weight_drift_success_evidence_is_self_bound_and_sanitized():
    value = json.loads(EVIDENCE.read_text())
    claimed = value.pop("receipt_sha256")
    assert claimed == _digest(value)

    assert value["status"] == "terminal_success_released_structural_weight_drift_passed"
    assert value["attempt"]["job"]["uid"] == "1e4079ba-29e1-477f-88be-780f17ea3964"
    assert value["attempt"]["pod"]["exit_code"] == 0
    assert value["attempt"]["pod"]["restart_count"] == 0
    assert value["attempt"]["pod"]["gpu_request"] == 0
    assert value["attempt"]["resource_release"]["active_owned_pods"] == 0
    assert value["attempt"]["resource_release"]["workload_finished"] is True

    receipt = value["aggregate_receipt"]
    assert receipt["schema"] == "cyber_qwen38_weight_drift_audit_v1"
    assert receipt["status"] == "passed"
    assert receipt["self_sha256_verified"] is True
    assert receipt["base"]["base_matches_frozen_lock"] is True
    assert list(receipt["exports"]) == ["step600", "step700", "step800", "step900", "step1000"]
    assert receipt["export_contract"]["all_exact_payload_identities_verified"] is True
    assert receipt["drift_summary"]["all_structural_weight_checks_pass"] is True
    assert receipt["drift_summary"]["structural_issues"] == []
    assert receipt["drift_summary"]["diagnostic_warnings"] == []
    assert all(
        row["changed_non_mtp_tensors_vs_previous"] > 0
        for row in receipt["drift_summary"]["checkpoints"].values()
    )
    assert all(
        row["changed_mtp_tensors_vs_base"] == 0
        for row in receipt["drift_summary"]["checkpoints"].values()
    )

    assert all(flag is False for flag in value["privacy"].values())
