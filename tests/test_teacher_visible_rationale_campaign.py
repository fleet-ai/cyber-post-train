"""Separate stronger-teacher visible-rationale collection/admission contract."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from training import teacher_visible_rationale_campaign as teacher
from training.io import digest_json, file_sha256

ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = (
    ROOT / "configs/collection/stronger-teacher-visible-rationale-current75-v1.requirements.json"
)
ROSTER = ROOT / "configs/collection/qwen38-base-current75-actions-pass4-v1"


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _seal(value: dict) -> dict:
    result = copy.deepcopy(value)
    result.pop("sha256", None)
    result["sha256"] = digest_json(result)
    return result


def _digest(character: str) -> str:
    return "sha256:" + character * 64


def _authorization() -> dict:
    value = {
        "schema": teacher.SOURCE_AUTHORIZATION_SCHEMA,
        "source": {
            "provider": "openai",
            "model": "gpt-5.6-sol",
            "immutable_provider_revision": "gpt-5.6-sol-2026-09-20",
            "session_model": "openai/gpt-5.6-sol",
            "route_profile_sha256": _digest("a"),
        },
        "teacher_strength_receipt_sha256": _digest("b"),
        "visible_output": {
            "surface": "ordinary_assistant_content_before_tool_call",
            "instruction_file_sha256": _load(REQUIREMENTS)["visible_rationale"][
                "instruction_file_sha256"
            ],
            "serialization_contract_sha256": digest_json(_load(REQUIREMENTS)["serialization"]),
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
        "artifact_key": "cyber/runs/qwen38/teacher-visible-rationale/authorization-v1",
        "version_index": 1,
        "content_sha256": payload,
    }
    value["registry_payload_sha256"] = payload
    return _seal(value)


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def _rendered(tmp_path: Path) -> tuple[dict, dict, dict]:
    requirements = _load(REQUIREMENTS)
    rendered = teacher.render(requirements, _authorization(), root=ROOT)
    profile = rendered["source-profile.json"]
    packet = rendered["collection-packet.json"]
    _write_json(tmp_path / "profile.json", profile)
    _write_json(tmp_path / "packet.json", packet)
    return requirements, profile, packet


def _attempt(profile: dict, packet: dict, *, task: dict, compaction: dict | None = None) -> dict:
    value = {
        "schema": teacher.ATTEMPT_SCHEMA,
        "record_id": "teacher-visible-record-1",
        "campaign_packet_sha256": packet["sha256"],
        "source_profile_sha256": profile["sha256"],
        "task_key": task["task_key"],
        "task_version_id": task["task_version_id"],
        "attempt": 1,
        "source_session_identity_sha256": _digest("c"),
        "normalized_record_sha256": _digest("d"),
        "normalized_trajectory_sha256": _digest("e"),
        "transcript_sha256": _digest("f"),
        "outcome": {
            "status": "completed",
            "verifier_process_success": True,
            "score_at_least_one": True,
            "verifier_execution_identity_sha256": _digest("1"),
            "verifier_receipt_sha256": _digest("2"),
            "verifier_authority": {
                "kind": "fleet_artifact_registry_immutable_v1",
                "artifact_key": "cyber/runs/qwen38/teacher-visible-rationale/verifier-1",
                "version_index": 1,
                "content_sha256": _digest("2"),
            },
        },
        "visible_rationale": {
            "surface": "ordinary_assistant_content_before_tool_call",
            "ordinary_content_only": True,
            "visible_to_student": True,
            "provider_private_reasoning_present": False,
            "rationale_target_tokens": 120,
            "visible_action_target_tokens": 80,
            "target_occurrence_manifest_sha256": _digest("3"),
        },
        "serialization": {
            "schema": teacher.ROUNDTRIP_SCHEMA,
            "qwen_target_sha256": digest_json(profile["qwen_target"]),
            "qwen_chat_template_sha256": profile["qwen_target"]["chat_template_sha256"],
            "roundtrip_receipt_sha256": _digest("4"),
            "message_surface": teacher.MESSAGE_SURFACE,
            "chat_template_kwargs": {"enable_thinking": False},
            "prompt_token_ids_equal_local_template": True,
            "prompt_token_ids_exact_prefix": True,
            "collection_training_serving_token_ids_match": True,
            "round_trip_verified": True,
        },
        "compaction": compaction or {"kind": "none", "boundaries": []},
    }
    return _seal(value)


def _admission_request(tmp_path: Path, attempts: Path) -> dict:
    authorization = tmp_path / "authorization.json"
    _write_json(authorization, _authorization())
    return {
        "schema": teacher.ADMISSION_REQUEST_SCHEMA,
        "requirements": {"path": str(REQUIREMENTS), "sha256": file_sha256(REQUIREMENTS)},
        "source_authorization": {
            "path": str(authorization),
            "sha256": file_sha256(authorization),
        },
        "source_profile": {
            "path": "profile.json",
            "sha256": file_sha256(tmp_path / "profile.json"),
        },
        "collection_packet": {
            "path": "packet.json",
            "sha256": file_sha256(tmp_path / "packet.json"),
        },
        "attempts": {"path": str(attempts), "sha256": file_sha256(attempts)},
        "output": "admitted",
    }


def test_renders_distinct_source_only_teacher_contract() -> None:
    requirements = _load(REQUIREMENTS)
    rendered = teacher.render(requirements, _authorization(), root=ROOT)
    profile = rendered["source-profile.json"]
    packet = rendered["collection-packet.json"]
    assert profile["schema"] == teacher.SOURCE_PROFILE_SCHEMA
    assert profile["source"]["kind"] == "teacher_visible_rationale"
    assert profile["qwen_target"]["revision"] == teacher.QWEN_REVISION
    assert profile["visible_rationale"]["surface"] == (
        "ordinary_assistant_content_before_tool_call"
    )
    assert profile["visible_rationale"]["private_fields_rejected"] == list(
        teacher._PRIVATE_FIELD_NAMES
    )
    assert profile["compaction"]["accepted_offline_kind"] == ("teacher_visible_exact_summary_v1")
    assert packet["schema"] == teacher.PACKET_SCHEMA
    assert packet["train_task_versions"] == 50
    assert packet["attempts_per_task"] == 4
    assert packet["planned_cells"] == 200
    assert packet["minimum_unique_supervised_tokens"] == 20_000_000
    assert packet["external_submission_authorized"] is False
    assert packet["training_data_eligible"] is True
    assert requirements["separation"] == {
        "reuse_action_only_packet": False,
        "reuse_qwen_self_packet": False,
        "mix_with_action_only_corpus": False,
        "mix_with_qwen_self_corpus": False,
        "teacher_hidden_reasoning_allowed": False,
    }


def test_exact_teacher_authorization_and_visible_surface_are_mandatory() -> None:
    authorization = _authorization()
    authorization["source"]["model"] = "grok-4.5"
    payload = digest_json(
        {
            key: item
            for key, item in authorization.items()
            if key not in {"authority", "registry_payload_sha256", "sha256"}
        }
    )
    authorization["registry_payload_sha256"] = payload
    authorization["authority"]["content_sha256"] = payload
    authorization = _seal(authorization)
    with pytest.raises(ValueError, match="reviewed provider/model"):
        teacher.render(_load(REQUIREMENTS), authorization, root=ROOT)

    authorization = _authorization()
    authorization["visible_output"]["provider_private_reasoning_ingested"] = True
    payload = digest_json(
        {
            key: item
            for key, item in authorization.items()
            if key not in {"authority", "registry_payload_sha256", "sha256"}
        }
    )
    authorization["registry_payload_sha256"] = payload
    authorization["authority"]["content_sha256"] = payload
    authorization = _seal(authorization)
    with pytest.raises(ValueError, match="ordinary visible rationale"):
        teacher.render(_load(REQUIREMENTS), authorization, root=ROOT)


def test_prompt_bytes_are_immutable() -> None:
    requirements = _load(REQUIREMENTS)
    requirements["visible_rationale"]["instruction_file_sha256"] = _digest("0")
    requirements = _seal(requirements)
    with pytest.raises(ValueError, match="prompt or safety contract drift"):
        teacher.render(requirements, _authorization(), root=ROOT)


def test_admits_only_verifier_success_and_emits_no_training_permit(tmp_path: Path) -> None:
    _requirements, profile, packet = _rendered(tmp_path)
    task = _load(ROSTER / "task-selection.json")["tasks"][0]
    attempts = tmp_path / "attempts.jsonl"
    attempts.write_text(json.dumps(_attempt(profile, packet, task=task)) + "\n")
    result = teacher.admit(_admission_request(tmp_path, attempts), relative_to=tmp_path)
    assert result == {
        "submitted": False,
        "selected_records": 1,
        "candidate_supervised_token_occurrences": 200,
        "candidate_occurrence_floor_reached": False,
        "sft_ready": False,
        "receipt_sha256": result["receipt_sha256"],
        "selection_sha256": result["selection_sha256"],
    }
    receipt = _load(tmp_path / "admitted/ADMISSION.json")
    assert receipt["heldout_families_admitted"] == 0
    assert receipt["source_text_read"] is False
    assert receipt["token_ids_read"] is False
    assert receipt["sft_ready"] is False
    assert receipt["next_gate"] == (
        "private_qwen_token_roundtrip_window_dedupe_and_exact_20m_coverage"
    )
    assert "record_id" not in json.dumps(receipt)
    assert "task_key" not in json.dumps(receipt)


def test_provider_private_reasoning_and_failed_verifier_are_not_admitted(tmp_path: Path) -> None:
    _requirements, profile, packet = _rendered(tmp_path)
    task = _load(ROSTER / "task-selection.json")["tasks"][0]
    private = _attempt(profile, packet, task=task)
    private["visible_rationale"]["provider_private_reasoning_present"] = True
    private = _seal(private)
    failed = _attempt(profile, packet, task=task)
    failed["record_id"] = "teacher-visible-record-2"
    failed["attempt"] = 2
    failed["source_session_identity_sha256"] = _digest("5")
    failed["normalized_trajectory_sha256"] = _digest("6")
    failed["outcome"]["score_at_least_one"] = False
    failed = _seal(failed)
    attempts = tmp_path / "attempts.jsonl"
    attempts.write_text(json.dumps(private) + "\n" + json.dumps(failed) + "\n")
    result = teacher.admit(_admission_request(tmp_path, attempts), relative_to=tmp_path)
    assert result["selected_records"] == 0
    receipt = _load(tmp_path / "admitted/ADMISSION.json")
    assert receipt["rejections"]["missing_visible_rationale"] == 1
    assert receipt["rejections"]["invalid_authoritative_success"] == 1


def test_exact_visible_compaction_summary_binds_true_next_prompt(tmp_path: Path) -> None:
    _requirements, profile, packet = _rendered(tmp_path)
    task = _load(ROSTER / "task-selection.json")["tasks"][0]
    boundary = {
        "boundary_id": _digest("7"),
        "pre_compaction_prompt_sha256": _digest("8"),
        "summary_generation_prompt_sha256": _digest("9"),
        "visible_summary_message_sha256": _digest("a"),
        "visible_summary_qwen_token_sha256": _digest("b"),
        "visible_summary_qwen_tokens": 42,
        "post_compaction_prompt_sha256": _digest("c"),
        "next_target_prompt_sha256": _digest("c"),
        "summary_visible_to_student": True,
        "summary_surface": "ordinary_assistant_content",
        "provider_private_reasoning_present": False,
        "summary_loss": "context_only_zero_loss",
    }
    attempt = _attempt(
        profile,
        packet,
        task=task,
        compaction={"kind": teacher.EXACT_VISIBLE_SUMMARY, "boundaries": [boundary]},
    )
    attempts = tmp_path / "attempts.jsonl"
    attempts.write_text(json.dumps(attempt) + "\n")
    teacher.admit(_admission_request(tmp_path, attempts), relative_to=tmp_path)
    receipt = _load(tmp_path / "admitted/ADMISSION.json")
    assert receipt["exact_visible_summary_sessions"] == 1
    assert receipt["opaque_compaction_sessions"] == 0

    invalid = copy.deepcopy(attempt)
    invalid["compaction"]["boundaries"][0]["next_target_prompt_sha256"] = _digest("d")
    invalid = _seal(invalid)
    with pytest.raises(ValueError, match="true visible next prompt"):
        teacher._attempt(invalid)


def test_admission_recomputes_authorized_profile_and_rejects_serialization_drift(
    tmp_path: Path,
) -> None:
    _requirements, profile, packet = _rendered(tmp_path)
    task = _load(ROSTER / "task-selection.json")["tasks"][0]
    attempt = _attempt(profile, packet, task=task)
    attempt["serialization"]["prompt_token_ids_exact_prefix"] = False
    attempt = _seal(attempt)
    attempts = tmp_path / "attempts.jsonl"
    attempts.write_text(json.dumps(attempt) + "\n")
    result = teacher.admit(_admission_request(tmp_path, attempts), relative_to=tmp_path)
    assert result["selected_records"] == 0
    receipt = _load(tmp_path / "admitted/ADMISSION.json")
    assert receipt["rejections"]["serialization_mismatch"] == 1

    other = tmp_path / "other"
    other.mkdir()
    _requirements, profile, packet = _rendered(other)
    profile["visible_rationale"]["maximum_sentences_before_tool"] = 3
    profile = _seal(profile)
    packet["source_profile_sha256"] = profile["sha256"]
    packet = _seal(packet)
    _write_json(other / "profile.json", profile)
    _write_json(other / "packet.json", packet)
    task = _load(ROSTER / "task-selection.json")["tasks"][0]
    attempts = other / "attempts.jsonl"
    attempts.write_text(json.dumps(_attempt(profile, packet, task=task)) + "\n")
    with pytest.raises(ValueError, match="not rendered from its authority"):
        teacher.admit(_admission_request(other, attempts), relative_to=other)


def test_heldout_family_cannot_enter_the_collection_roster(tmp_path: Path) -> None:
    requirements = _load(REQUIREMENTS)
    selection = _load(ROSTER / "task-selection.json")
    split = _load(ROSTER / "family-split.json")
    heldout = next(row for row in split["tasks"] if row["split"] == "dev")
    replacement = copy.deepcopy(selection["tasks"][0])
    replacement["task_key"] = heldout["task_key"]
    replacement["task_version_id"] = heldout["task_version_id"]
    selection["tasks"][0] = replacement
    selection = _seal(selection)
    selection_path = tmp_path / "selection.json"
    _write_json(selection_path, selection)
    requirements["roster"]["task_selection"] = {
        "path": str(selection_path),
        "sha256": file_sha256(selection_path),
    }
    requirements = _seal(requirements)
    with pytest.raises(ValueError, match="heldout or duplicate"):
        teacher.render(requirements, _authorization(), root=ROOT)
