from pathlib import Path

import pytest

from evals.fleet import qwen38_dedicated_rank2_v3 as lane
from evals.fleet import self_hosted

ROOT = Path(__file__).resolve().parents[1]


def test_rank2_attempt2_plan_is_exact_and_sequential() -> None:
    plan = lane.build_plan(ROOT)
    assert plan["item"]["selection_rank"] == 2
    assert plan["item"]["attempt"] == 2
    assert plan["item"]["run_id"] == lane.RUN_ID
    assert plan["config"]["serving"]["serving_block"] == lane.SERVING_BLOCK
    assert plan["config"]["serving"]["service_uid"] == lane.SERVICE_UID
    assert plan["config"]["serving"]["rayjob_uid"] == lane.RAYJOB_UID
    assert plan["config"]["serving"]["parity_receipt_sha256"] == lane.PARITY_SHA256
    assert plan["plan_sha256"] == self_hosted.digest_without(plan, "plan_sha256")
    assert plan["config"]["config_sha256"] == self_hosted.digest_without(
        plan["config"], "config_sha256"
    )


def test_rank2_attempts3_and4_have_fresh_exact_identities() -> None:
    for attempt in (3, 4):
        plan = lane.build_plan(ROOT, attempt)
        assert plan["item"]["selection_rank"] == 2
        assert plan["item"]["attempt"] == attempt
        assert plan["item"]["run_id"] == lane.RUN_IDS[attempt]
        assert plan["output_root"].endswith(lane.RUN_IDS[attempt])
        assert plan["config"]["serving"]["serving_block"] == lane.SERVING_BLOCK


def test_rank2_v3_parity_receipt_is_exact_and_non_scored() -> None:
    receipt = lane._validate_parity(ROOT)
    assert receipt["execution"]["task_instance_session_verifier_scoring_calls"] == 0
    assert receipt["tool_contract"]["calls_observed_in_order"] == ["bash", "submit_report"]


def test_next_attempt_requires_full_validated_previous_chain(tmp_path: Path) -> None:
    plan = lane.build_plan(ROOT, 2)
    previous = tmp_path / lane.RUN_IDS[2]
    previous.mkdir()
    accepted = lane.legacy._seal(
        {
            "accepted": True,
            "attempt": 2,
            "serving_block": lane.SERVING_BLOCK,
            "cell_id": plan["item"]["cell_id"],
            "execution_id": plan["item"]["execution_id"],
            "session_id": "session",
            "verifier_execution_id": "verifier",
        }
    )
    terminal = lane.legacy._seal(
        {
            "status": "ACCEPTED",
            "accepted_receipt_sha256": accepted["receipt_sha256"],
        }
    )
    validated = lane.legacy._seal(
        {
            "schema_version": "fleet-qwen38-dedicated-tp1-accepted-validated-v2",
            "status": "ACCEPTED_VALIDATED",
            "accepted": True,
            "credited": True,
            "retry_allowed": False,
            "attempt": 2,
            "serving_block": lane.SERVING_BLOCK,
            "cell_id": accepted["cell_id"],
            "execution_id": accepted["execution_id"],
            "session_id": accepted["session_id"],
            "verifier_execution_id": accepted["verifier_execution_id"],
            "all_artifact_byte_digests_matched": True,
            "fresh_authoritative_session_reconciled": True,
            "fleet_api_mutations": 0,
            "prompts_or_traces_included": False,
            "scores_included": False,
        }
    )
    self_hosted.write_json_once(previous / "ACCEPTED.json", accepted)
    self_hosted.write_json_once(previous / "TERMINAL.json", terminal)
    self_hosted.write_json_once(previous / "ACCEPTED_VALIDATED.json", validated)
    lane._require_previous_accepted(3, tmp_path)

    validated["all_artifact_byte_digests_matched"] = False
    validated["receipt_sha256"] = self_hosted.digest_without(validated, "receipt_sha256")
    (previous / "ACCEPTED_VALIDATED.json").write_text(__import__("json").dumps(validated))
    with pytest.raises(RuntimeError, match="not accepted"):
        lane._require_previous_accepted(3, tmp_path)
