from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = ROOT / "docs/evidence/qwen36-study/2026-08-31-state-v1.json"


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_evidence(reference: dict[str, str]) -> None:
    path = ROOT / reference["path"]
    assert path.is_file(), reference["path"]
    assert _sha256(path) == reference["file_sha256"]


def test_study_state_binds_exact_model_and_terminal_base_receipts() -> None:
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    assert state["model"]["revision"] == "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
    _assert_evidence(state["model"]["evidence"])
    _assert_evidence(state["data"]["evidence"])

    web = state["base_evaluations"]["webexploitbench_level0"]
    assert (web["solved"], web["vulnerabilities"]) == (10, 110)
    assert web["infrastructure_invalid_targets"] == 0
    _assert_evidence(web["evidence"])

    fleet = state["base_evaluations"]["fleet_test20"]
    assert (fleet["solved"], fleet["tasks"]) == (1, 20)
    assert fleet["primary_eval_infrastructure_invalid_tasks"] == 0
    _assert_evidence(fleet["evidence"])

    gym = state["base_evaluations"]["exploitgym_pilot"]
    assert (gym["solved"], gym["valid_results"]) == (0, 5)
    assert gym["paired_control_eligible"] is False
    _assert_evidence(gym["evidence"])


def test_pending_work_never_claims_a_terminal_result() -> None:
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    pending = [
        state["sft"]["export"],
        state["native_fleet_rl"]["full_run_successor"],
        state["verified_miles_rl"]["rank_safe_canary_03"],
        *state["paired_evaluation_protocols"].values(),
    ]
    for item in pending:
        status = (
            item.get("observed_status", "") + item.get("queue_state", "") + item.get("status", "")
        )
        assert "pending" in status.lower() or "not_launched" in status.lower()
        assert item["terminal_result"] is None


def test_state_binds_post_training_protocol_evidence() -> None:
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    _assert_evidence(state["sft"]["plan"])
    _assert_evidence(state["sft"]["pre_export_tokenizer_gate"]["evidence"])
    _assert_evidence(state["native_fleet_rl"]["full_run_successor"]["pre_submit_evidence"])
    _assert_evidence(state["paired_evaluation_protocols"]["exploitgym"]["evidence"])

    gate = state["native_fleet_rl"]["one_step_gate"]
    assert gate["all_zero_group_fraction"] == 1.0
    assert gate["rollout_truncation_rate"] == 1.0
