"""Broad source-only teacher-visible-rationale campaign and matched control."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from training import teacher_visible_rationale_broad_campaign as broad
from training import teacher_visible_rationale_campaign as teacher
from training.io import digest_json

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT / "configs/collection/stronger-teacher-visible-rationale-current75-pass64-v1.source.json"
)
REQUIREMENTS = (
    ROOT
    / "configs/collection/stronger-teacher-visible-rationale-current75-pass64-v1.requirements.json"
)
REVIEW = (
    ROOT
    / "configs/collection/stronger-teacher-visible-rationale-current75-pass64-v1/broad-review.json"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _seal(value: dict) -> dict:
    result = copy.deepcopy(value)
    result.pop("sha256", None)
    result["sha256"] = digest_json(result)
    return result


def _digest(character: str) -> str:
    return "sha256:" + character * 64


def _authorization(*, revision: str = "gpt-5.6-sol-2026-09-20") -> dict:
    requirements = _load(REQUIREMENTS)
    value = {
        "schema": teacher.SOURCE_AUTHORIZATION_SCHEMA,
        "source": {
            "provider": "openai",
            "model": "gpt-5.6-sol",
            "immutable_provider_revision": revision,
            "session_model": "openai/gpt-5.6-sol",
            "route_profile_sha256": _digest("a"),
        },
        "teacher_strength_receipt_sha256": _digest("b"),
        "visible_output": {
            "surface": "ordinary_assistant_content_before_tool_call",
            "instruction_file_sha256": requirements["visible_rationale"]["instruction_file_sha256"],
            "serialization_contract_sha256": digest_json(requirements["serialization"]),
            "visible_to_student": True,
            "private_fields_rejected": list(teacher._PRIVATE_FIELD_NAMES),
            "provider_private_reasoning_ingested": False,
        },
        "training_use": {
            "authorized": True,
            "purpose": "qwen38_teacher_visible_rationale_plus_actions",
            "target_model": f"{teacher.QWEN_REPOSITORY}@{teacher.QWEN_REVISION}",
            "issuer": "fleet-data-authority",
            "issued_at": "2026-09-21T00:00:00Z",
        },
    }
    payload = digest_json(value)
    value["authority"] = {
        "kind": "fleet_artifact_registry_immutable_v1",
        "artifact_key": "cyber/runs/qwen38/teacher-visible-rationale/authorization/broad-v1",
        "version_index": 1,
        "content_sha256": payload,
    }
    value["registry_payload_sha256"] = payload
    return _seal(value)


def test_review_freezes_broad_scale_cost_and_zero_external_work() -> None:
    receipt = broad.review(_load(SOURCE), root=ROOT)
    assert receipt == _load(REVIEW)
    assert receipt["schema"] == broad.REVIEW_SCHEMA
    assert receipt["train_task_versions"] == receipt["train_families"] == 50
    assert receipt["attempts_per_task"] == 64
    assert receipt["planned_cells"] == 3_200
    assert receipt["minimum_unique_supervised_tokens_per_arm"] == 20_000_000
    assert receipt["minimum_selected_families"] == 40
    assert receipt["minimum_selected_family_fraction"] == 0.8
    assert receipt["historical_reference_is_mixed_teacher_corpus"] is True
    assert receipt["estimated_successes_needed_for_action_only_20m"] == 1_006
    assert receipt["estimated_required_success_fraction"] == {
        "numerator": 1_006,
        "denominator": 3_200,
    }
    assert receipt["maximum_daily_fleet_rollouts"] == 500
    assert receipt["minimum_calendar_days_at_daily_cap"] == 7
    assert receipt["aggregate_context_window_capacity_tokens"] == 838_860_800
    assert receipt["provider_cost_estimate"] == {
        "currency": "USD",
        "planned_paid_teacher_rollouts": 3_200,
        "aggregate_context_capacity_tokens": 838_860_800,
        "aggregate_context_capacity_is_not_billed_token_forecast": True,
        "dollar_estimate_available": False,
        "missing_inputs": [
            "authorized route billed input-token count and unit price",
            "authorized route billed output-token count and unit price",
        ],
        "exact_dollar_formula": (
            "billed_input_tokens * input_usd_per_token + "
            "billed_output_tokens * output_usd_per_token"
        ),
    }
    assert receipt["fleet_api_calls"] == receipt["model_calls"] == 0
    assert receipt["raw_private_trace_reads"] == 0
    assert receipt["external_submission_authorized"] is False
    assert receipt["training_authorized"] is False
    assert receipt["independent_review_required"] is True


def test_authorized_render_adds_exactly_once_operation_and_matched_control() -> None:
    rendered = broad.render(_load(SOURCE), _authorization(), root=ROOT)
    assert set(rendered) == {
        "source-profile.json",
        "collection-packet.json",
        "broad-review.json",
        "operation-authorization.json",
        "matched-materialization-plan.json",
    }
    profile = rendered["source-profile.json"]
    packet = rendered["collection-packet.json"]
    operation = rendered["operation-authorization.json"]
    matched = rendered["matched-materialization-plan.json"]
    assert profile["opencode"] == {
        **profile["opencode"],
        "context_window_tokens": 262_144,
        "context_management": teacher.ONLINE_COMPACTION,
    }
    assert profile["serialization"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert profile["visible_rationale"]["private_fields_rejected"] == list(
        teacher._PRIVATE_FIELD_NAMES
    )
    assert packet["planned_cells"] == operation["planned_cells"] == 3_200
    assert operation["collection_packet_sha256"] == packet["sha256"]
    assert operation["execution_contract"] == {
        "canonical_operation_root_required": True,
        "dedicated_empty_ledger_required": True,
        "exclusive_pre_mutation_intent_required": True,
        "same_path_retry_allowed": False,
        "alternate_path_retry_allowed": False,
        "ambiguous_create_replay_allowed": False,
    }
    assert operation["external_submission_authorized"] is False
    assert matched["single_selected_success_set"] is True
    assert matched["same_serialized_messages_actions_and_compaction"] is True
    assert matched["second_model_collection"] is False
    assert matched["loss_arms"] == broad.LOSS_ARMS
    assert matched["loss_arms"]["matched_actions_only"]["ordinary_visible_rationale"] == 0
    assert matched["loss_arms"]["visible_rationale_plus_actions"]["ordinary_visible_rationale"] == 1
    assert matched["external_benchmarks_excluded"] == ["WebExploitBench"]
    assert matched["heldout_roles_excluded"] == ["dev", "final_test"]
    assert matched["provider_private_or_hidden_reasoning_allowed"] is False
    assert matched["external_submission_authorized"] is False
    assert matched["training_authorized"] is False


def test_scientific_cell_universe_contains_3200_unique_predeclared_identities() -> None:
    requirements = teacher._requirements(_load(REQUIREMENTS), root=ROOT)
    roster = teacher._roster(requirements, root=ROOT)
    cells = broad._scientific_cells(requirements=requirements, roster=roster)
    assert len(cells) == 3_200
    assert len({cell["scientific_cell_id"] for cell in cells}) == 3_200
    assert {cell["attempt"] for cell in cells} == set(range(1, 65))


def test_operation_identity_is_stable_and_bound_to_exact_teacher_authorization() -> None:
    first = broad.render(_load(SOURCE), _authorization(), root=ROOT)
    same = broad.render(_load(SOURCE), _authorization(), root=ROOT)
    changed = broad.render(
        _load(SOURCE),
        _authorization(revision="gpt-5.6-sol-2026-09-21"),
        root=ROOT,
    )
    assert first["operation-authorization.json"] == same["operation-authorization.json"]
    assert (
        first["operation-authorization.json"]["scientific_cell_universe_sha256"]
        == changed["operation-authorization.json"]["scientific_cell_universe_sha256"]
    )
    assert (
        first["operation-authorization.json"]["operation_root_name"]
        != changed["operation-authorization.json"]["operation_root_name"]
    )
    assert (
        first["operation-authorization.json"]["authorized_execution_universe_sha256"]
        != changed["operation-authorization.json"]["authorized_execution_universe_sha256"]
    )


@pytest.mark.parametrize(
    ("path", "replacement", "message"),
    [
        (("exclusions", "split_roles"), ["final_test"], "holdout or benchmark"),
        (("exclusions", "external_benchmarks"), [], "holdout or benchmark"),
        (
            ("exclusions", "provider_private_or_hidden_reasoning"),
            True,
            "holdout or benchmark",
        ),
        (("safety", "external_submission_authorized"), True, "source-only boundary"),
        (("matched_comparison", "second_model_collection"), True, "matched comparison"),
    ],
)
def test_review_rejects_holdout_private_reasoning_launch_and_pairing_drift(
    path: tuple[str, str], replacement: object, message: str
) -> None:
    value = _load(SOURCE)
    value[path[0]][path[1]] = replacement
    value = _seal(value)
    with pytest.raises(ValueError, match=message):
        broad.review(value, root=ROOT)


def test_source_bundle_write_is_local_private_and_create_once(tmp_path: Path) -> None:
    rendered = broad.render(_load(SOURCE), _authorization(), root=ROOT)
    output = tmp_path / "broad-source-only"
    broad.write(output, rendered)
    assert sorted(path.name for path in output.iterdir()) == sorted(rendered)
    assert all((path.stat().st_mode & 0o077) == 0 for path in output.iterdir())
    with pytest.raises(FileExistsError, match="already exists"):
        broad.write(output, rendered)


def test_source_bundle_write_rejects_path_traversal(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="one local file"):
        broad.write(tmp_path / "bundle", {"../escape.json": {}})
    assert not (tmp_path / "bundle").exists()
    assert not (tmp_path / "escape.json").exists()
