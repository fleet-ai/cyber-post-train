import json
from pathlib import Path

import pytest

from evals.fleet import qwen38_dedicated_rank3_v3 as lane
from evals.fleet import self_hosted

ROOT = Path(__file__).resolve().parents[1]


def test_rank3_block_has_fresh_generation_exact_complete_cells_and_is_held() -> None:
    plans = [lane.build_plan(ROOT, attempt) for attempt in (1, 2, 3, 4)]
    assert [plan["item"]["attempt"] for plan in plans] == [1, 2, 3, 4]
    assert {plan["item"]["selection_rank"] for plan in plans} == {3}
    assert {plan["item"]["execution_generation"] for plan in plans} == {21}
    assert [plan["item"]["cell_id"] for plan in plans] == [
        lane.EXPECTED_IDENTITIES[index][0] for index in (1, 2, 3, 4)
    ]
    assert [plan["item"]["execution_id"] for plan in plans] == [
        lane.EXPECTED_IDENTITIES[index][1] for index in (1, 2, 3, 4)
    ]
    assert all(plan["launch_authorized"] is False for plan in plans)
    assert all(plan["config"]["serving"]["serving_block"] == lane.SERVING_BLOCK for plan in plans)
    assert all(plan["config"]["serving"]["service_uid"] == lane.SERVICE_UID for plan in plans)
    assert all(plan["config"]["serving"]["rayjob_uid"] == lane.RAYJOB_UID for plan in plans)
    assert all(
        plan["config"]["serving"]["parity_receipt_sha256"] == lane.PARITY_SHA256 for plan in plans
    )
    assert all(
        plan["plan_sha256"] == self_hosted.digest_without(plan, "plan_sha256") for plan in plans
    )


def test_rank3_release_must_transfer_laptop_hold_and_clear_global_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = {
        "schema_version": "fleet-qwen38-dedicated-rank3-release-v1",
        "status": "RELEASED_TO_DEDICATED",
        "selection_rank": 3,
        "cell_ids": [lane.EXPECTED_IDENTITIES[index][0] for index in (1, 2, 3, 4)],
        "execution_ids": [lane.EXPECTED_IDENTITIES[index][1] for index in (1, 2, 3, 4)],
        "serving_block": lane.SERVING_BLOCK,
        "laptop_hold_transferred": True,
        "fresh_global_ledger_clear": True,
        "fresh_authoritative_sessions_clear": True,
        "claims_clear": True,
        "prompts_traces_flags_or_scores_included": False,
    }
    receipt["receipt_sha256"] = self_hosted.digest_without(receipt, "receipt_sha256")
    path = tmp_path / "release.json"
    path.write_text(json.dumps(receipt))
    monkeypatch.setenv("QWEN_RANK3_RELEASE_PATH", str(path))
    plan = lane._released_plan(ROOT, 1)
    assert plan["launch_authorized"] is True
    assert plan["release_receipt_sha256"] == receipt["receipt_sha256"]

    receipt["claims_clear"] = False
    receipt["receipt_sha256"] = self_hosted.digest_without(receipt, "receipt_sha256")
    path.write_text(json.dumps(receipt))
    with pytest.raises(RuntimeError, match="release receipt drifted"):
        lane._released_plan(ROOT, 1)


def test_rank3_attempt_order_requires_validated_previous_cell(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        lane._require_previous_validated(2, tmp_path)


def test_rank3_parity_is_uid_bound_to_tp1_b_server() -> None:
    value = lane._validate_parity(ROOT)
    binding = value["endpoint"]["server_binding"]
    assert binding["api_run_id"] == "ft-run-8656260d"
    assert binding["rayjob_uid"] == lane.RAYJOB_UID
    assert binding["service_uid"] == lane.SERVICE_UID
    assert binding["context_length"] == 262144
