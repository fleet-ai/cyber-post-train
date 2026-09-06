from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from evals.fleet import qwen_hosted_rank17_g22_held_v1 as held

ROOT = Path(__file__).parents[1]


def test_plan_freezes_one_complete_unstarted_task() -> None:
    plan = held.build_plan(ROOT)
    source = json.loads((ROOT / held.SOURCE_PATH).read_text())
    held.validate_plan(plan, source)
    assert [(row["selection_rank"], row["attempt"]) for row in plan["attempts"]] == [
        (17, 1),
        (17, 2),
        (17, 3),
        (17, 4),
    ]
    assert len({row["execution_id"] for row in plan["attempts"]}) == 4
    assert all(row["execution_generation"] == 22 for row in plan["attempts"])
    assert plan["launch_authorized"] is False


def test_plan_preserves_frozen_treatment() -> None:
    plan = held.build_plan(ROOT)
    assert plan["model"] == held.EXPECTED_MODEL
    assert plan["harness"] == held.EXPECTED_HARNESS
    assert plan["treatment"]["tools"] == ["bash", "submit_report"]
    assert plan["treatment"]["context_management"].endswith("autocontinue_v1")
    assert plan["execution"]["global_execution_claim_before_model_call"] is True
    assert plan["atomic_whole_task_reservation"]["claim_count"] == 4


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("launch_authorized", True),
        ("sfs_root", "/mnt/sfs/jobs/reused"),
        ("controller", "qwen-b"),
    ],
)
def test_plan_rejects_authority_mutations(field: str, value: object) -> None:
    source = json.loads((ROOT / held.SOURCE_PATH).read_text())
    plan = held.build_plan(ROOT)
    plan[field] = value
    with pytest.raises(ValueError, match="plan drifted"):
        held.validate_plan(plan, source)


def test_plan_rejects_cell_or_execution_mutation() -> None:
    source = json.loads((ROOT / held.SOURCE_PATH).read_text())
    for key, value in (("cell_id", "sha256:" + "0" * 64), ("execution_id", "sha256:" + "1" * 64)):
        plan = held.build_plan(ROOT)
        plan["attempts"][0][key] = value
        plan["plan_sha256"] = held.self_hosted.digest_without(plan, "plan_sha256")
        with pytest.raises(ValueError, match="plan drifted"):
            held.validate_plan(plan, source)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("tasks", 0, "verifier", "sha256"), "0" * 64),
        (
            ("tasks", 0, "environment", "version_id"),
            "00000000-0000-0000-0000-000000000000",
        ),
        (("attempts", 0, "environment_version_id"), "00000000-0000-0000-0000-000000000000"),
        (("treatment", "timeout_seconds"), 1),
        (("treatment", "max_model_requests"), 1),
        (("atomic_whole_task_reservation", "claim_count"), 3),
        (
            ("atomic_whole_task_reservation", "all_claims_validated_before_model_call"),
            False,
        ),
        (("execution_rollforward", "predecessor_execution_generation"), 18),
        (("authority", "scoring_mode"), "binary"),
        (("release_gates", "fresh_score_blind_global_ledger_required"), False),
        (("execution", "unexpected_field"), True),
        (("unexpected_top_level_field",), True),
        (("tasks", 0, "unexpected_field"), True),
        (("attempts", 0, "unexpected_field"), True),
    ],
)
def test_plan_rejects_rehashed_nested_or_extra_field_mutation(
    path: tuple[str | int, ...], value: object
) -> None:
    source = json.loads((ROOT / held.SOURCE_PATH).read_text())
    plan = held.build_plan(ROOT)
    target: object = plan
    for key in path[:-1]:
        target = target[key]  # type: ignore[index]
    target[path[-1]] = value  # type: ignore[index]
    plan["plan_sha256"] = held.self_hosted.digest_without(plan, "plan_sha256")
    with pytest.raises(ValueError, match="plan drifted"):
        held.validate_plan(plan, source)


def test_plan_rejects_rehashed_source_mutation() -> None:
    source = json.loads((ROOT / held.SOURCE_PATH).read_text())
    source["tasks"][0]["verifier"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="source plan digest drifted"):
        held.validate_plan(held.build_plan(ROOT), source)


def test_held_receipt_is_exact_and_non_executable() -> None:
    value = json.loads((ROOT / held.HELD_PATH).read_text())
    held.validate_held(value, ROOT)
    assert value["status"] == "HELD_PENDING_FRESH_SCORE_BLIND_RELEASE"
    assert value["launch_authorized"] is False
    assert value["scoring_authorized"] is False
    assert value["required_fresh_release"]["all_four_cells_exactly_unstarted"] is True
    assert set(value["side_effects"].values()) == {0}
    assert value["privacy"] == {
        "scores_read": False,
        "prompts_traces_flags_read": False,
        "credentials_included": False,
    }


def test_held_receipt_rejects_relaxed_gate_or_side_effect() -> None:
    for mutate in (
        lambda value: value["required_fresh_release"].__setitem__(
            "authoritative_session_collisions", 1
        ),
        lambda value: value["side_effects"].__setitem__("model_calls", 1),
        lambda value: value.__setitem__("launch_authorized", True),
    ):
        value = copy.deepcopy(held.held_receipt(ROOT))
        mutate(value)
        value["receipt_sha256"] = held.self_hosted.digest_without(value, "receipt_sha256")
        with pytest.raises(ValueError, match="receipt drifted"):
            held.validate_held(value, ROOT)
