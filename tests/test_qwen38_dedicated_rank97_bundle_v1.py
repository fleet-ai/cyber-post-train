import json
from pathlib import Path

import pytest

from evals.fleet import qwen38_dedicated_rank97_bundle_v1 as lane
from evals.fleet import self_hosted

ROOT = Path(__file__).resolve().parents[1]


def test_rank97_preserves_g19_cells_but_uses_dedicated_block() -> None:
    plans = [lane.build_plan(ROOT, attempt) for attempt in (1, 2, 3, 4)]
    assert [plan["item"]["cell_id"] for plan in plans] == [
        lane.EXPECTED_IDENTITIES[index][0] for index in (1, 2, 3, 4)
    ]
    assert [plan["item"]["execution_id"] for plan in plans] == [
        lane.EXPECTED_IDENTITIES[index][1] for index in (1, 2, 3, 4)
    ]
    assert {plan["item"]["execution_generation"] for plan in plans} == {19}
    assert {plan["config"]["serving"]["serving_block"] for plan in plans} == {lane.SERVING_BLOCK}
    assert all(plan["launch_authorized"] is False for plan in plans)


def test_partial_reservation_rolls_back_only_exact_new_claims(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plans = [lane.build_plan(ROOT, attempt) for attempt in (1, 2, 3, 4)]
    claim_root = tmp_path / "claims"
    claim_root.mkdir()
    monkeypatch.setattr(lane.legacy, "CLAIM_ROOT", claim_root)
    calls = 0
    original = lane.legacy._claim

    def fail_second(plan, job_uid, pod_uid):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("collision")
        return original(plan, job_uid, pod_uid)

    monkeypatch.setattr(lane.legacy, "_claim", fail_second)
    with pytest.raises(RuntimeError, match="collision"):
        lane.reserve_all(
            plans,
            "11111111-1111-4111-8111-111111111111",
            "22222222-2222-4222-8222-222222222222",
        )
    assert list(claim_root.glob("*.json")) == []


def test_rank97_parity_is_exact_uid_bound() -> None:
    value = lane._validate_parity(ROOT)
    assert value["receipt_sha256"] == lane.PARITY_SHA256
    assert value["receipt_sha256"] == self_hosted.digest_without(value, "receipt_sha256")
    assert value["endpoint"]["server_binding"]["service_uid"] == lane.SERVICE_UID


def test_rank97_release_requires_complete_fresh_four_cell_transfer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = {
        "schema_version": "fleet-qwen38-dedicated-rank97-whole-task-release-v1",
        "status": "RELEASED_TO_DEDICATED",
        "selection_rank": 97,
        "cell_ids": [lane.EXPECTED_IDENTITIES[index][0] for index in (1, 2, 3, 4)],
        "execution_ids": [lane.EXPECTED_IDENTITIES[index][1] for index in (1, 2, 3, 4)],
        "serving_block": lane.SERVING_BLOCK,
        "prior_owner": "qwen-a-generation19-v4",
        "single_job_four_claim_reservation_required": True,
        "fresh_global_ledger_clear": True,
        "fresh_authoritative_sessions_clear": True,
        "claims_clear": True,
        "output_roots_clear": True,
        "prompts_traces_flags_or_scores_included": False,
    }
    release["receipt_sha256"] = self_hosted.digest_without(release, "receipt_sha256")
    path = tmp_path / "release.json"
    path.write_text(json.dumps(release))
    monkeypatch.setenv("QWEN_RANK97_RELEASE_PATH", str(path))
    plans = lane._released_plans(ROOT)
    assert len(plans) == 4
    assert all(plan["launch_authorized"] is True for plan in plans)
    release["claims_clear"] = False
    release["receipt_sha256"] = self_hosted.digest_without(release, "receipt_sha256")
    path.write_text(json.dumps(release))
    with pytest.raises(RuntimeError, match="release receipt drifted"):
        lane._released_plans(ROOT)
