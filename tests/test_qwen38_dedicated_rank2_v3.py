from pathlib import Path

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
