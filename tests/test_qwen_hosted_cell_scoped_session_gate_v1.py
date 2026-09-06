from __future__ import annotations

import copy

import pytest

from evals.fleet import qwen_hosted_cell_scoped_session_gate_v1 as gate

TASK_KEY = "task-a"
VERSION = "version-a"
MODEL = "provider/model-a"


def _row(**changes):
    value = {
        "session_id": "session-1",
        "eval_task_id": "task-id-1",
        "eval_task_version_id": VERSION,
        "task_key": TASK_KEY,
        "model_id": "model-a",
        "model_identity": MODEL,
        "model_identity_status": "resolved",
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
    )


def test_exact_target_treatment_is_a_collision_without_caller_metadata() -> None:
    assert _classify(_row()) == "TARGET_TREATMENT_COLLISION"


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


def test_resolved_different_model_is_non_target() -> None:
    assert (
        _classify(_row(model_id="model-b", model_identity="provider/model-b")) == "NON_TARGET_MODEL"
    )


@pytest.mark.parametrize(
    "row",
    [
        _row(model_identity=None, model_identity_status="ambiguous"),
        _row(eval_task_version_id=None),
        _row(model_id=None),
        _row(model_id="model-b"),
        _row(model_id="provider/model-a"),
        _row(model_identity="/model-a"),
        _row(model_identity="provider/extra/model-a"),
        _row(model_identity=" provider/model-a"),
        _row(model_identity="provider/model-b", model_identity_status="ambiguous"),
        {key: value for key, value in _row().items() if key != "model_id"},
        {**_row(), "run_id": "sk_live_secret123"},
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
        "caller_cell_execution_run_metadata_is_untrusted": True,
        "exact_target_version_without_immutable_model_identity_blocks": True,
    }
    assert set(value["required_projection_fields"]).isdisjoint(value["forbidden_projection_fields"])


def test_rehashed_contract_relaxation_is_rejected() -> None:
    value = copy.deepcopy(gate.contract())
    value["launch_authorized"] = True
    value["receipt_sha256"] = gate.self_hosted.digest_without(value, "receipt_sha256")
    with pytest.raises(ValueError, match="contract drifted"):
        gate.validate_contract(value)
