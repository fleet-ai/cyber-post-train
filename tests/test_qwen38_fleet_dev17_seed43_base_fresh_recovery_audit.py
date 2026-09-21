import json
from pathlib import Path

from cyber_post_train.jobs import digest

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/qwen38-fleet-dev17-seed43-base-fresh-recovery-audit-20260921.json"


def test_recovery_audit_is_self_digesting_score_blind_and_does_not_apply():
    receipt = json.loads(EVIDENCE.read_text())
    assert receipt["sha256"] == digest(
        {key: value for key, value in receipt.items() if key != "sha256"}
    )
    assert receipt["private_manifest"]["status"] == "partial_observation_do_not_apply"
    assert receipt["authoritative_review"]["score_values_read_or_recorded"] is False
    assert receipt["privacy"]["final_eight_task_set_accessed"] is False
    assert receipt["launch_gate"]["open"] is False
    assert receipt["launch_gate"]["scoring_jobs_created"] == 0
    assert receipt["launch_gate"]["rollout_jobs_created"] == 0
    assert receipt["launch_gate"]["cells_replayed_or_mutated"] == 0


def test_stored_sessions_never_schedule_model_generation_or_redundant_scoring():
    arms = json.loads(EVIDENCE.read_text())["arms"]
    assert arms["base"]["reviewed_actions"] == {
        "accept_existing_scored_session": 10,
        "scoring_only_recovery": 0,
        "rollout_retry": 0,
        "manual_terminal_review": 0,
    }
    assert arms["fresh75"]["reviewed_actions"] == {
        "accept_existing_scored_session": 1,
        "scoring_only_recovery": 0,
        "rollout_retry_pending_full_absence_proof": 2,
        "manual_terminal_review": 0,
    }
