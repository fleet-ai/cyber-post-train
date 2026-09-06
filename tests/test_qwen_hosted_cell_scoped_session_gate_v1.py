from __future__ import annotations

import copy

import pytest

from evals.fleet import qwen_hosted_cell_scoped_session_gate_v1 as gate

TASK_KEY = "task-a"
VERSION = "version-a"
MODEL = "provider/model-a"
PLANNED = "sha256:" + "1" * 64
OTHER_CELL = "sha256:" + "2" * 64
OTHER_EXECUTION = "sha256:" + "3" * 64
OTHER_RUN = "other-model-run"


def _row(**changes):
    value = {
        "session_id": "session-1",
        "eval_task_id": "task-id-1",
        "eval_task_version_id": VERSION,
        "task_key": TASK_KEY,
        "model_identity": MODEL,
        "model_identity_status": "resolved",
        "cell_id": None,
        "execution_id": None,
        "run_id": None,
        "status": "completed",
    }
    value.update(changes)
    return value


def _classify(row):
    return gate.classify(
        row,
        task_key=TASK_KEY,
        task_version_id=VERSION,
        session_model=MODEL,
        planned_cell_ids={PLANNED},
        known_non_target_executions={(OTHER_CELL, OTHER_EXECUTION, OTHER_RUN)},
    )


def test_exact_target_treatment_is_a_collision_without_caller_metadata() -> None:
    assert _classify(_row()) == "TARGET_TREATMENT_COLLISION"


def test_planned_cell_identity_is_always_a_collision() -> None:
    assert (
        _classify(
            _row(
                eval_task_version_id="different-version",
                model_identity=None,
                model_identity_status="ambiguous",
                cell_id=PLANNED,
                execution_id=OTHER_EXECUTION,
                run_id="some-run",
            )
        )
        == "TARGET_CELL_COLLISION"
    )


def test_null_model_does_not_blanket_block_a_different_exact_version() -> None:
    assert (
        _classify(
            _row(
                eval_task_version_id="different-version",
                model_identity=None,
                model_identity_status="ambiguous",
            )
        )
        == "NON_TARGET_VERSION"
    )


def test_exact_known_non_target_execution_can_disambiguate_null_model() -> None:
    assert (
        _classify(
            _row(
                model_identity=None,
                model_identity_status="ambiguous",
                cell_id=OTHER_CELL,
                execution_id=OTHER_EXECUTION,
                run_id=OTHER_RUN,
            )
        )
        == "KNOWN_NON_TARGET_EXECUTION"
    )


def test_resolved_different_model_is_non_target() -> None:
    assert _classify(_row(model_identity="provider/model-b")) == "NON_TARGET_MODEL"


@pytest.mark.parametrize(
    "row",
    [
        _row(model_identity=None, model_identity_status="ambiguous"),
        _row(eval_task_version_id=None),
        _row(cell_id=OTHER_CELL, execution_id=None, run_id=OTHER_RUN),
        _row(model_identity="provider/model-b", model_identity_status="ambiguous"),
        {**_row(), "score": 1},
    ],
)
def test_missing_conflicting_extra_or_unknown_identity_fails_closed(row) -> None:
    with pytest.raises(gate.IdentityAmbiguous):
        _classify(row)


def test_contract_is_exact_held_and_score_blind() -> None:
    value = gate.contract()
    gate.validate_contract(value)
    assert value["launch_authorized"] is False
    assert value["scoring_authorized"] is False
    assert value["null_model_policy"] == {
        "blanket_task_level_block_forbidden": True,
        "different_exact_task_version_may_be_ignored": True,
        "known_non_target_execution_tuple_may_be_ignored": True,
        "exact_target_version_without_known_cell_identity_still_blocks": True,
    }
    assert set(value["required_projection_fields"]).isdisjoint(
        value["forbidden_projection_fields"]
    )


def test_rehashed_contract_relaxation_is_rejected() -> None:
    value = copy.deepcopy(gate.contract())
    value["launch_authorized"] = True
    value["receipt_sha256"] = gate.self_hosted.digest_without(
        value, "receipt_sha256"
    )
    with pytest.raises(ValueError, match="contract drifted"):
        gate.validate_contract(value)
