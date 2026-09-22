"""Executable private materializer and matched-arm training permit tests."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from training import corpus as dense_corpus
from training import sft
from training import teacher_visible_rationale_broad_campaign as broad
from training import teacher_visible_rationale_campaign as teacher
from training import teacher_visible_rationale_corpus as corpus
from training.io import digest_json, file_sha256
from training.sft_runtime import DENSE_FORMAT, DENSE_SCHEMA

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT / "configs/collection/stronger-teacher-visible-rationale-current75-pass64-v1.source.json"
)
REQUIREMENTS = (
    ROOT
    / "configs/collection/stronger-teacher-visible-rationale-current75-pass64-v1.requirements.json"
)


class _FakeTokenizer:
    def _message(self, message: dict) -> list[int]:
        role = message["role"]
        if role == "assistant":
            values = [90]
            values.extend(ord(character) for character in message.get("content") or "")
            for call in message.get("tool_calls") or []:
                values.extend(ord(character) for character in json.dumps(call, sort_keys=True))
            return values + [91]
        values = [40 + {"system": 1, "user": 2, "tool": 3}[role]]
        values.extend(ord(character) for character in message.get("content") or "")
        return values + [49]

    def apply_chat_template(self, messages, *, add_generation_prompt=False, **_kwargs):
        result = []
        for message in messages:
            result.extend(self._message(message))
        if add_generation_prompt:
            result.append(90)
        return result


def _messages() -> list[dict]:
    return [
        {"role": "system", "content": "You are a security researcher."},
        {"role": "user", "content": "Test the target."},
        {
            "role": "assistant",
            "content": "The response exposes a useful input. I will test it next.",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": {"script": "printf test"},
                    },
                }
            ],
        },
        {"role": "tool", "content": "ok", "tool_call_id": "call-1"},
    ]


def _target_data() -> tuple[list[int], int, list[int], list[int]]:
    tokenizer = _FakeTokenizer()
    messages = _messages()
    prompt = tokenizer.apply_chat_template(messages[:2], add_generation_prompt=True)
    rendered = tokenizer.apply_chat_template(messages[:3], add_generation_prompt=False)
    rationale = [ord(character) for character in messages[2]["content"]]
    action = rendered[len(prompt) + len(rationale) :]
    assert rendered[: len(prompt)] == prompt
    return rendered, len(prompt), rationale, action


def _compaction_fixture() -> tuple[list[dict], list[dict], dict]:
    tokenizer = _FakeTokenizer()
    messages = [
        {"role": "system", "content": "You are a security researcher."},
        {"role": "user", "content": "Test the target."},
        {
            "role": "assistant",
            "content": "First visible rationale.",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "bash", "arguments": {"script": "printf first"}},
                }
            ],
        },
        {"role": "tool", "content": "first", "tool_call_id": "call-1"},
        {"role": "assistant", "content": "First visible compacted summary."},
        {
            "role": "assistant",
            "content": "Second visible rationale.",
            "tool_calls": [
                {
                    "id": "call-2",
                    "type": "function",
                    "function": {"name": "bash", "arguments": {"script": "printf second"}},
                }
            ],
        },
        {"role": "tool", "content": "second", "tool_call_id": "call-2"},
        {"role": "assistant", "content": "Second visible compacted summary."},
        {
            "role": "assistant",
            "content": "Third visible rationale.",
            "tool_calls": [
                {
                    "id": "call-3",
                    "type": "function",
                    "function": {"name": "bash", "arguments": {"script": "printf third"}},
                }
            ],
        },
    ]

    def window(
        window_id: str,
        sequence_index: int,
        prompt_indices: list[int],
        target_index: int,
        occurrence: str,
    ) -> dict:
        prompt = tokenizer.apply_chat_template(
            [messages[index] for index in prompt_indices], add_generation_prompt=True
        )
        return {
            "window_id": window_id,
            "sequence_index": sequence_index,
            "message_indices": [*prompt_indices, target_index],
            "target_message_index": target_index,
            "prompt_token_count": len(prompt),
            "prompt_token_sha256": digest_json(prompt),
            "target_spans": [
                {
                    "kind": corpus.ACTION_KIND,
                    "target_occurrence_sha256": occurrence,
                }
            ],
        }

    windows = [
        window("before", 0, [0, 1], 2, _digest("1")),
        window("middle", 1, [0, 1, 4], 5, _digest("2")),
        window("after", 2, [0, 1, 7], 8, _digest("3")),
    ]

    def boundary(
        *,
        boundary_index: int,
        previous_boundary_id: str | None,
        parent: dict,
        summary_index: int,
        summary_indices: list[int],
        target: dict,
    ) -> dict:
        pre_indices = parent["message_indices"][:-1]
        post_indices = target["message_indices"][:-1]
        pre_prompt = tokenizer.apply_chat_template(
            [messages[index] for index in pre_indices], add_generation_prompt=True
        )
        summary_prompt = tokenizer.apply_chat_template(
            [messages[index] for index in summary_indices], add_generation_prompt=True
        )
        summary_rendered = tokenizer.apply_chat_template(
            [*[messages[index] for index in summary_indices], messages[summary_index]],
            add_generation_prompt=False,
        )
        summary_tokens = summary_rendered[len(summary_prompt) :]
        post_prompt = tokenizer.apply_chat_template(
            [messages[index] for index in post_indices], add_generation_prompt=True
        )
        value = {
            "boundary_index": boundary_index,
            "previous_boundary_id": previous_boundary_id,
            "source_session_identity_sha256": _digest("4"),
            "normalized_trajectory_sha256": _digest("5"),
            "transcript_sha256": _digest("6"),
            "target_occurrence_manifest_sha256": _digest("7"),
            "parent_window_id": parent["window_id"],
            "parent_target_message_index": parent["target_message_index"],
            "pre_compaction_prompt_message_indices": pre_indices,
            "pre_compaction_prompt_sha256": digest_json(pre_prompt),
            "pre_compaction_prompt_tokens": len(pre_prompt),
            "summary_generation_message_indices": summary_indices,
            "summary_generation_prompt_sha256": digest_json(summary_prompt),
            "summary_generation_prompt_tokens": len(summary_prompt),
            "summary_message_index": summary_index,
            "visible_summary_message_sha256": digest_json(messages[summary_index]),
            "visible_summary_qwen_token_sha256": digest_json(summary_tokens),
            "visible_summary_qwen_tokens": len(summary_tokens),
            "post_compaction_prompt_message_indices": post_indices,
            "post_compaction_prompt_sha256": digest_json(post_prompt),
            "post_compaction_prompt_tokens": len(post_prompt),
            "next_target_window_id": target["window_id"],
            "next_target_message_index": target["target_message_index"],
            "next_target_prompt_sha256": digest_json(post_prompt),
            "next_target_occurrence_sha256": target["target_spans"][0]["target_occurrence_sha256"],
            "summary_visible_to_student": True,
            "summary_surface": "ordinary_assistant_content",
            "provider_private_reasoning_present": False,
            "summary_loss": "context_only_zero_loss",
        }
        value["boundary_id"] = digest_json(value)
        return value

    first = boundary(
        boundary_index=0,
        previous_boundary_id=None,
        parent=windows[0],
        summary_index=4,
        summary_indices=[0, 1, 2, 3],
        target=windows[1],
    )
    second = boundary(
        boundary_index=1,
        previous_boundary_id=first["boundary_id"],
        parent=windows[1],
        summary_index=7,
        summary_indices=[0, 1, 4, 5, 6],
        target=windows[2],
    )
    return (
        messages,
        windows,
        {
            "kind": teacher.EXACT_VISIBLE_SUMMARY,
            "boundaries": [first, second],
        },
    )


def _reseal_boundaries(boundaries: list[dict]) -> None:
    previous: str | None = None
    for index, boundary in enumerate(boundaries):
        boundary["boundary_index"] = index
        boundary["previous_boundary_id"] = previous
        boundary["boundary_id"] = digest_json(
            {key: item for key, item in boundary.items() if key != "boundary_id"}
        )
        previous = boundary["boundary_id"]


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
    requirements = _load(REQUIREMENTS)
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
        "artifact_key": "cyber/runs/qwen38/teacher-visible-rationale/authorization/test-v1",
        "version_index": 1,
        "content_sha256": payload,
    }
    value["registry_payload_sha256"] = payload
    return _seal(value)


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def _source_files(tmp_path: Path) -> tuple[dict, dict, dict, dict]:
    authorization = _authorization()
    rendered = broad.render(_load(SOURCE), authorization, root=ROOT)
    profile = rendered["source-profile.json"]
    packet = rendered["collection-packet.json"]
    matched = rendered["matched-materialization-plan.json"]
    _write(tmp_path / "profile.json", profile)
    _write(tmp_path / "packet.json", packet)
    _write(tmp_path / "matched.json", matched)
    _write(tmp_path / "authorization.json", authorization)
    return profile, packet, matched, authorization


def _attempt(profile: dict, packet: dict, task: dict, manifest_sha256: str) -> dict:
    _rendered, _prompt_count, rationale_ids, action_ids = _target_data()
    rationale = {
        "surface": "ordinary_assistant_content_before_tool_call",
        "ordinary_content_only": True,
        "visible_to_student": True,
        "provider_private_reasoning_present": False,
        "unknown_reasoning_fields_present": False,
        "forbidden_private_field_occurrences": {name: 0 for name in teacher._PRIVATE_FIELD_NAMES},
        "tool_calls": 1,
        "tool_calls_with_visible_rationale": 1,
        "rationale_segments": 1,
        "minimum_sentences_per_rationale": 2,
        "maximum_sentences_per_rationale": 2,
        "rationale_target_tokens": len(rationale_ids),
        "visible_action_target_tokens": len(action_ids),
        "target_occurrence_manifest_sha256": manifest_sha256,
    }
    serialization = {
        "schema": teacher.ROUNDTRIP_SCHEMA,
        "qwen_target_sha256": digest_json(profile["qwen_target"]),
        "qwen_chat_template_sha256": profile["qwen_target"]["chat_template_sha256"],
        "roundtrip_receipt_sha256": _digest("a"),
        "roundtrip_receipt_authority": {
            "kind": "fleet_artifact_registry_immutable_v1",
            "artifact_key": "cyber/runs/qwen38/teacher-visible-rationale/roundtrip/test-v1",
            "version_index": 1,
            "content_sha256": _digest("a"),
        },
        "message_surface": teacher.MESSAGE_SURFACE,
        "chat_template_kwargs": {"enable_thinking": False},
        "prompt_token_ids_equal_local_template": True,
        "prompt_token_ids_exact_prefix": True,
        "collection_training_serving_token_ids_match": True,
        "round_trip_verified": True,
    }
    compaction = {"kind": "none", "boundaries": []}
    roster = teacher._roster(teacher._requirements(_load(REQUIREMENTS), root=ROOT), root=ROOT)
    identity = (task["task_key"], task["task_version_id"])
    binding = {
        "campaign_packet_sha256": packet["sha256"],
        "source_profile_sha256": profile["sha256"],
        "record_id": "teacher-visible-record-1",
        "source_session_identity_sha256": _digest("1"),
        "normalized_record_sha256": _digest("2"),
        "normalized_trajectory_sha256": digest_json(_messages()),
        "transcript_sha256": _digest("4"),
        "task_key": task["task_key"],
        "task_version_id": task["task_version_id"],
        "attempt": 1,
        "runtime_binding_sha256": digest_json(roster["selected_runtime_bindings"][identity]),
        "teacher_source_sha256": digest_json(profile["source"]),
        "qwen_target_sha256": digest_json(profile["qwen_target"]),
        "qwen_chat_template_sha256": profile["qwen_target"]["chat_template_sha256"],
        "serialization_contract_sha256": digest_json(profile["serialization"]),
        "serialization_evidence_sha256": digest_json(serialization),
        "roundtrip_receipt_sha256": serialization["roundtrip_receipt_sha256"],
        "compaction_sha256": digest_json(compaction),
        "visible_rationale_evidence_sha256": digest_json(rationale),
        "target_occurrence_manifest_sha256": manifest_sha256,
    }
    outcome = {
        "status": "completed",
        "verifier_process_success": True,
        "score_at_least_one": True,
        "verifier_execution_identity_sha256": _digest("b"),
        "verifier_receipt_sha256": _digest("c"),
        "verifier_authority": {
            "kind": "fleet_artifact_registry_immutable_v1",
            "artifact_key": "cyber/runs/qwen38/teacher-visible-rationale/verifier/test-v1",
            "version_index": 1,
            "content_sha256": _digest("c"),
        },
        "attempt_binding": binding,
    }
    payload = teacher._registry_payload_sha256(outcome)
    outcome["authority"] = {
        "kind": "fleet_artifact_registry_immutable_v1",
        "artifact_key": "cyber/runs/qwen38/teacher-visible-rationale/success/test-v1",
        "version_index": 1,
        "content_sha256": payload,
    }
    outcome["registry_payload_sha256"] = payload
    return _seal(
        {
            "schema": teacher.ATTEMPT_SCHEMA,
            "record_id": binding["record_id"],
            "campaign_packet_sha256": packet["sha256"],
            "source_profile_sha256": profile["sha256"],
            "task_key": task["task_key"],
            "task_version_id": task["task_version_id"],
            "attempt": 1,
            "source_session_identity_sha256": binding["source_session_identity_sha256"],
            "normalized_record_sha256": binding["normalized_record_sha256"],
            "normalized_trajectory_sha256": binding["normalized_trajectory_sha256"],
            "transcript_sha256": binding["transcript_sha256"],
            "outcome": outcome,
            "visible_rationale": rationale,
            "serialization": serialization,
            "compaction": compaction,
        }
    )


def _selection(profile: dict, packet: dict, manifest_sha256: str, authorization: dict) -> dict:
    roster = teacher._roster(teacher._requirements(_load(REQUIREMENTS), root=ROOT), root=ROOT)
    identity, task = next(iter(sorted(roster["selected"].items())))
    attempt = _attempt(profile, packet, task, manifest_sha256)
    row = {
        "record_id": attempt["record_id"],
        "task_key": attempt["task_key"],
        "task_version_id": attempt["task_version_id"],
        "group_id": task["group_id"],
        "source_session_identity_sha256": attempt["source_session_identity_sha256"],
        "normalized_record_sha256": attempt["normalized_record_sha256"],
        "normalized_trajectory_sha256": attempt["normalized_trajectory_sha256"],
        "transcript_sha256": attempt["transcript_sha256"],
        "attempt": attempt["attempt"],
        "attempt_metadata_sha256": attempt["sha256"],
        "runtime_binding_sha256": digest_json(roster["selected_runtime_bindings"][identity]),
        "target_occurrence_manifest_sha256": manifest_sha256,
        "rationale_target_tokens": attempt["visible_rationale"]["rationale_target_tokens"],
        "visible_action_target_tokens": attempt["visible_rationale"][
            "visible_action_target_tokens"
        ],
        "outcome": attempt["outcome"],
        "visible_rationale": attempt["visible_rationale"],
        "serialization": attempt["serialization"],
        "compaction": attempt["compaction"],
    }
    value = {
        "schema": teacher.SELECTION_SCHEMA,
        "request_files_sha256": {},
        "contract_files_sha256": {},
        "roster_files_sha256": {},
        "admission_input_manifest_sha256": _digest("7"),
        "source_authorization_sha256": authorization["sha256"],
        "source_authorization_authority": authorization["authority"],
        "source_profile_sha256": profile["sha256"],
        "collection_packet_sha256": packet["sha256"],
        "family_split_sha256": packet["family_split_sha256"],
        "family_role_anchor_sha256": packet["family_role_anchor_sha256"],
        "protected_family_lock_sha256": packet["protected_family_lock_sha256"],
        "records": [row],
        "sft_ready": False,
        "next_gate": "private_qwen_token_roundtrip_window_dedupe_and_exact_20m_coverage",
    }
    return _seal(value)


def _receipt(profile: dict, packet: dict, selection: dict) -> dict:
    return _seal(
        {
            "schema": teacher.RECEIPT_SCHEMA,
            "source_profile_sha256": profile["sha256"],
            "collection_packet_sha256": packet["sha256"],
            "selection_sha256": selection["sha256"],
            "heldout_families_admitted": 0,
            "source_text_read": False,
            "token_ids_read": False,
            "sft_ready": False,
        }
    )


def _occurrence(kind: str, source_offset: int, token_ids: list[int]) -> str:
    return digest_json(
        {
            "normalized_record_sha256": _digest("2"),
            "kind": kind,
            "source_message_index": 2,
            "source_token_offset": source_offset,
            "source_token_count": len(token_ids),
            "token_ids_sha256": digest_json(token_ids),
        }
    )


def _record(profile: dict, packet: dict, selection: dict) -> dict:
    input_ids, prompt_count, rationale_ids, action_ids = _target_data()
    rationale_start = prompt_count
    rationale_end = rationale_start + len(rationale_ids)
    action_start = rationale_end
    action_end = action_start + len(action_ids)
    rationale_occurrence = _occurrence(corpus.RATIONALE_KIND, 0, rationale_ids)
    action_occurrence = _occurrence(corpus.ACTION_KIND, len(rationale_ids), action_ids)
    occurrences = [
        {
            "target_occurrence_sha256": rationale_occurrence,
            "kind": corpus.RATIONALE_KIND,
            "source_message_index": 2,
            "source_token_offset": 0,
            "source_token_count": len(rationale_ids),
            "source_token_sha256": digest_json(rationale_ids),
        },
        {
            "target_occurrence_sha256": action_occurrence,
            "kind": corpus.ACTION_KIND,
            "source_message_index": 2,
            "source_token_offset": len(rationale_ids),
            "source_token_count": len(action_ids),
            "source_token_sha256": digest_json(action_ids),
        },
    ]
    manifest_sha256 = digest_json(
        sorted(occurrences, key=lambda item: item["target_occurrence_sha256"])
    )
    assert selection["records"][0]["target_occurrence_manifest_sha256"] == manifest_sha256
    spans = [
        {
            "target_occurrence_sha256": rationale_occurrence,
            "kind": corpus.RATIONALE_KIND,
            "source_message_index": 2,
            "token_start": rationale_start,
            "token_end": rationale_end,
            "source_token_offset": 0,
            "token_ids_sha256": digest_json(rationale_ids),
        },
        {
            "target_occurrence_sha256": action_occurrence,
            "kind": corpus.ACTION_KIND,
            "source_message_index": 2,
            "token_start": action_start,
            "token_end": action_end,
            "source_token_offset": len(rationale_ids),
            "token_ids_sha256": digest_json(action_ids),
        },
    ]
    payload = {
        "window_id": "window-1",
        "sequence_index": 0,
        "message_indices": [0, 1, 2],
        "target_message_index": 2,
        "input_ids": input_ids,
        "prompt_token_count": prompt_count,
        "prompt_token_sha256": digest_json(input_ids[:prompt_count]),
        "serialized_token_ids_sha256": digest_json(input_ids),
        "target_spans": spans,
    }
    selected = selection["records"][0]
    value = {
        "schema": corpus.PRIVATE_RECORD_SCHEMA,
        "record_id": selected["record_id"],
        "source_profile_sha256": profile["sha256"],
        "collection_packet_sha256": packet["sha256"],
        "selection_record_sha256": selected["attempt_metadata_sha256"],
        "lineage": {
            "task_key": selected["task_key"],
            "task_version_id": selected["task_version_id"],
            "group_id": selected["group_id"],
            "attempt": selected["attempt"],
        },
        "evidence": {
            "source_session_identity_sha256": selected["source_session_identity_sha256"],
            "normalized_record_sha256": selected["normalized_record_sha256"],
            "normalized_trajectory_sha256": selected["normalized_trajectory_sha256"],
            "transcript_sha256": selected["transcript_sha256"],
            "target_occurrence_manifest_sha256": manifest_sha256,
        },
        "reasoning_visibility": "student_visible_ordinary_assistant_content",
        "private_reasoning_present": False,
        "original_task_digest": _digest("d"),
        "messages": _messages(),
        "windows": [{**payload, "window_payload_sha256": digest_json(payload)}],
        "compaction": {"kind": "none", "boundaries": []},
    }
    value["content_digest"] = digest_json(value)
    return value


def _request(tmp_path: Path, selection: dict, receipt: dict) -> dict:
    _write(tmp_path / "selection.json", selection)
    _write(tmp_path / "admission.json", receipt)
    return {
        "schema": corpus.REQUEST_SCHEMA,
        "requirements": {"path": str(REQUIREMENTS), "sha256": file_sha256(REQUIREMENTS)},
        "source_authorization": {
            "path": "authorization.json",
            "sha256": file_sha256(tmp_path / "authorization.json"),
        },
        "source_profile": {
            "path": "profile.json",
            "sha256": file_sha256(tmp_path / "profile.json"),
        },
        "collection_packet": {
            "path": "packet.json",
            "sha256": file_sha256(tmp_path / "packet.json"),
        },
        "admission_selection": {
            "path": "selection.json",
            "sha256": file_sha256(tmp_path / "selection.json"),
        },
        "admission_receipt": {
            "path": "admission.json",
            "sha256": file_sha256(tmp_path / "admission.json"),
        },
        "matched_plan": {
            "path": "matched.json",
            "sha256": file_sha256(tmp_path / "matched.json"),
        },
        "tokenizer_root": "tokenizer",
        "records": {"path": "records.jsonl", "sha256": file_sha256(tmp_path / "records.jsonl")},
        "output": "corpus",
    }


def _materialization_fixture(tmp_path: Path, monkeypatch) -> tuple[dict, dict]:
    profile, packet, _matched, authorization = _source_files(tmp_path)
    _input_ids, _prompt_count, rationale_ids, action_ids = _target_data()
    occurrences = [
        {
            "target_occurrence_sha256": _occurrence(corpus.RATIONALE_KIND, 0, rationale_ids),
            "kind": corpus.RATIONALE_KIND,
            "source_message_index": 2,
            "source_token_offset": 0,
            "source_token_count": len(rationale_ids),
            "source_token_sha256": digest_json(rationale_ids),
        },
        {
            "target_occurrence_sha256": _occurrence(
                corpus.ACTION_KIND, len(rationale_ids), action_ids
            ),
            "kind": corpus.ACTION_KIND,
            "source_message_index": 2,
            "source_token_offset": len(rationale_ids),
            "source_token_count": len(action_ids),
            "source_token_sha256": digest_json(action_ids),
        },
    ]
    manifest_sha256 = digest_json(
        sorted(occurrences, key=lambda item: item["target_occurrence_sha256"])
    )
    selection = _selection(profile, packet, manifest_sha256, authorization)
    record = _record(profile, packet, selection)
    receipt = _receipt(profile, packet, selection)
    (tmp_path / "records.jsonl").write_text(json.dumps(record) + "\n")
    (tmp_path / "tokenizer").mkdir()
    model_lock = _load(ROOT / "configs/models/qwen38-27b-1d4bf0f2.lock.json")
    monkeypatch.setattr(
        dense_corpus,
        "local_tokenizer",
        lambda _lock, _root: (
            _FakeTokenizer(),
            {
                "repo": model_lock["repo"],
                "revision": model_lock["revision"],
                "files": model_lock["tokenizer"]["files"],
                "chat_template_sha256": model_lock["tokenizer"]["files"][2]["sha256"],
                "backend_sha256": "f" * 64,
            },
        ),
    )
    return _request(tmp_path, selection, receipt), record


def test_materializes_exact_same_windows_with_only_the_loss_mask_changed(
    tmp_path: Path, monkeypatch
) -> None:
    request, _record_value = _materialization_fixture(tmp_path, monkeypatch)
    result = corpus.build(request, relative_to=tmp_path)
    assert result["submitted"] is False
    assert result["external_mutations"] == 0
    assert result["training_ready"] is False
    rationale_rows = pq.read_table(
        tmp_path / "corpus" / corpus.RATIONALE_ARM / "train.parquet"
    ).to_pylist()
    action_rows = pq.read_table(
        tmp_path / "corpus" / corpus.ACTION_ARM / "train.parquet"
    ).to_pylist()
    assert rationale_rows[0]["input_ids"] == action_rows[0]["input_ids"]
    assert rationale_rows[0]["source_window_sha256"] == action_rows[0]["source_window_sha256"]
    _input_ids, prompt_count, rationale_ids, action_ids = _target_data()
    assert rationale_rows[0]["loss_mask"] == [0] * prompt_count + [1] * (
        len(rationale_ids) + len(action_ids)
    )
    assert action_rows[0]["loss_mask"] == [0] * (prompt_count + len(rationale_ids)) + [1] * len(
        action_ids
    )
    coverage = _load(tmp_path / "corpus/coverage.private.json")
    assert coverage["arms"][corpus.RATIONALE_ARM][
        "packing_independent_unique_target_tokens"
    ] == len(rationale_ids) + len(action_ids)
    assert coverage["arms"][corpus.ACTION_ARM]["packing_independent_unique_target_tokens"] == len(
        action_ids
    )
    assert coverage["provider_private_or_hidden_reasoning_materialized"] is False
    assert coverage["raw_text_written"] is False
    with pytest.raises(ValueError, match="has not reached every corpus gate"):
        corpus._authorize_aggregate(
            _load(tmp_path / "corpus/matched-manifest.private.json"),
            coverage,
            _load(tmp_path / "corpus/MATERIALIZATION.json"),
        )


def test_hidden_reasoning_field_is_rejected_before_any_output(tmp_path: Path, monkeypatch) -> None:
    request, record = _materialization_fixture(tmp_path, monkeypatch)
    record["hidden_reasoning"] = [99]
    record["content_digest"] = digest_json(
        {key: item for key, item in record.items() if key != "content_digest"}
    )
    (tmp_path / "records.jsonl").write_text(json.dumps(record) + "\n")
    request["records"]["sha256"] = file_sha256(tmp_path / "records.jsonl")
    with pytest.raises(ValueError, match="forbidden private reasoning"):
        corpus.build(request, relative_to=tmp_path)
    assert not (tmp_path / "corpus").exists()


def test_hidden_reasoning_nested_in_visible_message_is_rejected(
    tmp_path: Path, monkeypatch
) -> None:
    request, record = _materialization_fixture(tmp_path, monkeypatch)
    record["messages"][2]["analysis"] = "provider-private text"
    record["content_digest"] = digest_json(
        {key: item for key, item in record.items() if key != "content_digest"}
    )
    (tmp_path / "records.jsonl").write_text(json.dumps(record) + "\n")
    request["records"]["sha256"] = file_sha256(tmp_path / "records.jsonl")
    with pytest.raises(ValueError, match="forbidden private reasoning"):
        corpus.build(request, relative_to=tmp_path)
    assert not (tmp_path / "corpus").exists()


def test_local_qwen_rerender_rejects_self_consistent_token_drift(
    tmp_path: Path, monkeypatch
) -> None:
    request, record = _materialization_fixture(tmp_path, monkeypatch)
    window = record["windows"][0]
    window["input_ids"][0] += 1
    window["prompt_token_sha256"] = digest_json(window["input_ids"][: window["prompt_token_count"]])
    window["serialized_token_ids_sha256"] = digest_json(window["input_ids"])
    window["window_payload_sha256"] = digest_json(
        {key: item for key, item in window.items() if key != "window_payload_sha256"}
    )
    record["content_digest"] = digest_json(
        {key: item for key, item in record.items() if key != "content_digest"}
    )
    (tmp_path / "records.jsonl").write_text(json.dumps(record) + "\n")
    request["records"]["sha256"] = file_sha256(tmp_path / "records.jsonl")
    with pytest.raises(ValueError, match="local Qwen/OpenCode rendering"):
        corpus.build(request, relative_to=tmp_path)
    assert not (tmp_path / "corpus").exists()


def test_local_qwen_rerender_rejects_swapped_rationale_and_action_labels(
    tmp_path: Path, monkeypatch
) -> None:
    _request_value, record = _materialization_fixture(tmp_path, monkeypatch)
    window = copy.deepcopy(record["windows"][0])
    for span in window["target_spans"]:
        span["kind"] = (
            corpus.ACTION_KIND if span["kind"] == corpus.RATIONALE_KIND else corpus.RATIONALE_KIND
        )
    with pytest.raises(ValueError, match="labels do not match"):
        corpus._rendered_window(_FakeTokenizer(), record["messages"], window)


def test_compaction_binds_two_monotone_immediate_parent_transitions() -> None:
    messages, windows, compaction = _compaction_fixture()
    checked = teacher._compaction(compaction)
    corpus._rendered_compaction(_FakeTokenizer(), messages, windows, checked)


def test_compaction_rejects_cross_wired_and_nonadjacent_windows() -> None:
    messages, windows, compaction = _compaction_fixture()
    cross_wired = copy.deepcopy(compaction)
    first = cross_wired["boundaries"][0]
    first["parent_window_id"] = "cross-wired-window"
    _reseal_boundaries(cross_wired["boundaries"])
    checked = teacher._compaction(cross_wired)
    with pytest.raises(ValueError, match="immediate parent/next-target"):
        corpus._rendered_compaction(_FakeTokenizer(), messages, windows, checked)

    nonadjacent = copy.deepcopy(compaction)
    first = nonadjacent["boundaries"][0]
    after = windows[2]
    first["next_target_window_id"] = after["window_id"]
    first["next_target_message_index"] = after["target_message_index"]
    first["next_target_occurrence_sha256"] = after["target_spans"][0]["target_occurrence_sha256"]
    nonadjacent["boundaries"] = [first]
    _reseal_boundaries(nonadjacent["boundaries"])
    checked = teacher._compaction(nonadjacent)
    with pytest.raises(ValueError, match="immediate parent/next-target"):
        corpus._rendered_compaction(_FakeTokenizer(), messages, windows, checked)


def test_compaction_rejects_reversed_or_future_message_chronology() -> None:
    _messages_value, _windows_value, compaction = _compaction_fixture()
    reversed_boundaries = copy.deepcopy(compaction)
    reversed_boundaries["boundaries"].reverse()
    _reseal_boundaries(reversed_boundaries["boundaries"])
    with pytest.raises(ValueError, match="chronology is not monotone"):
        teacher._compaction(reversed_boundaries)

    future = copy.deepcopy(compaction)
    first = future["boundaries"][0]
    first["summary_generation_message_indices"] = [0, 1, 2, 3, 4]
    _reseal_boundaries(future["boundaries"])
    with pytest.raises(ValueError, match="chronology is not monotone"):
        teacher._compaction(future)


def test_compaction_rejects_cross_wired_prompt_tokens() -> None:
    messages, windows, compaction = _compaction_fixture()
    wrong_tokens = copy.deepcopy(compaction)
    first = wrong_tokens["boundaries"][0]
    first["post_compaction_prompt_sha256"] = windows[2]["prompt_token_sha256"]
    first["next_target_prompt_sha256"] = windows[2]["prompt_token_sha256"]
    _reseal_boundaries(wrong_tokens["boundaries"])
    checked = teacher._compaction(wrong_tokens)
    with pytest.raises(ValueError, match="real visible continuation"):
        corpus._rendered_compaction(_FakeTokenizer(), messages, windows, checked)


def test_source_occurrence_cannot_be_repeated_by_repacking(tmp_path: Path, monkeypatch) -> None:
    request, record = _materialization_fixture(tmp_path, monkeypatch)
    duplicate = copy.deepcopy(record["windows"][0])
    duplicate["window_id"] = "window-2"
    duplicate["sequence_index"] = 1
    duplicate["window_payload_sha256"] = digest_json(
        {key: item for key, item in duplicate.items() if key != "window_payload_sha256"}
    )
    record["windows"].append(duplicate)
    record["content_digest"] = digest_json(
        {key: item for key, item in record.items() if key != "content_digest"}
    )
    (tmp_path / "records.jsonl").write_text(json.dumps(record) + "\n")
    request["records"]["sha256"] = file_sha256(tmp_path / "records.jsonl")
    with pytest.raises(ValueError, match="repeated across packed windows"):
        corpus.build(request, relative_to=tmp_path)


def _qualified_documents() -> tuple[dict, dict, dict, dict]:
    train_digest = _digest("a")
    pending = _seal(
        {
            "schema": corpus.PENDING_MANIFEST_SCHEMA,
            "source_sha256": _digest("b"),
            "split_sha256": _digest("c"),
            "tokenizer": {"repo": teacher.QWEN_REPOSITORY, "revision": teacher.QWEN_REVISION},
            "files": {
                "train": {
                    "path": "train.parquet",
                    "sha256": train_digest,
                    "rows": 100,
                    "task_keys": ["train-task"],
                    "format": DENSE_FORMAT,
                    "supervised_tokens": 20_000_000,
                    "assistant_responses": 50,
                    "source_sessions": 40,
                    "source_total_assistant_responses": 50,
                    "excluded_assistant_responses": 0,
                }
            },
            "validation_mode": corpus.PENDING_VALIDATION_MODE,
            "objective": corpus.RATIONALE_ARM,
            "training_permit_sha256": None,
        }
    )
    gate = {
        "packing_independent_unique_target_tokens": 20_000_000,
        "minimum_unique_target_tokens": 20_000_000,
        "families_with_targets": 40,
        "minimum_selected_families": 40,
        "selected_family_fraction": 0.8,
        "minimum_selected_family_fraction": 0.8,
        "largest_family_target_token_fraction": 0.025,
        "maximum_family_target_token_fraction": 0.25,
        "family_concentration_within_limit": True,
        "qualified": True,
    }
    coverage = _seal(
        {
            "schema": corpus.COVERAGE_SCHEMA,
            "source_profile_sha256": _digest("d"),
            "collection_packet_sha256": _digest("e"),
            "admission_selection_sha256": _digest("f"),
            "admission_receipt_sha256": _digest("1"),
            "matched_plan_sha256": _digest("2"),
            "selected_source_records": 40,
            "packed_windows": 100,
            "same_packed_window_selection": True,
            "same_input_ids_and_order": True,
            "only_loss_mask_differs": True,
            "packing_independent_target_occurrence_deduplication": True,
            "arms": {arm: copy.deepcopy(gate) for arm in corpus.ARMS},
            "heldout_families_materialized": 0,
            "external_benchmarks_materialized": [],
            "provider_private_or_hidden_reasoning_materialized": False,
            "raw_text_written": False,
            "training_ready": True,
        }
    )
    matched = _seal(
        {
            "schema": corpus.MATCHED_CORPUS_SCHEMA,
            "source_profile_sha256": coverage["source_profile_sha256"],
            "collection_packet_sha256": coverage["collection_packet_sha256"],
            "admission_selection_sha256": coverage["admission_selection_sha256"],
            "admission_receipt_sha256": coverage["admission_receipt_sha256"],
            "matched_plan_sha256": coverage["matched_plan_sha256"],
            "coverage_sha256": coverage["sha256"],
            "arms": {
                arm: {
                    "manifest_path": f"{arm}/manifest.pending.json",
                    "manifest_file_sha256": _digest("3"),
                    "manifest_sha256": pending["sha256"],
                    "train_file_sha256": train_digest,
                }
                for arm in corpus.ARMS
            },
            "training_ready": True,
        }
    )
    receipt = _seal(
        {
            "schema": corpus.RECEIPT_SCHEMA,
            "matched_manifest_file_sha256": _digest("4"),
            "matched_manifest_sha256": matched["sha256"],
            "coverage_file_sha256": _digest("5"),
            "coverage_sha256": coverage["sha256"],
            "selection_file_sha256": _digest("6"),
            "training_ready": True,
            "external_mutations": 0,
            "submitted": False,
        }
    )
    return pending, coverage, matched, receipt


def test_permit_requires_both_20m_arms_and_consumer_emits_dense_sft_manifest(
    tmp_path: Path,
) -> None:
    pending, coverage, matched, receipt = _qualified_documents()
    compile_config = {
        "name": "teacher-rationale-consumer-test",
        "output_root": "/mnt/sfs/jobs/teacher-visible-rationale-consumer-test",
        "model": {
            "lock": str(ROOT / "configs/models/qwen38-27b-1d4bf0f2.lock.json"),
            "weights": str(ROOT / "configs/models/qwen38-27b-1d4bf0f2.weights.json"),
            "root": "/mnt/sfs/models/qwen38-test",
        },
        "data": {
            "manifest": "manifest.json",
            "root": "/mnt/sfs/data/teacher-visible-rationale-test",
        },
        "recipe": {"eval_interval": 0},
        "wandb": {
            "entity": "fleet",
            "project": "cyber-post",
            "group": "teacher-visible-rationale-test",
            "run_id": "teacher-rationale-consumer-test",
            "name": "teacher-rationale-consumer-test",
        },
    }
    _write(tmp_path / "manifest.json", pending)
    with pytest.raises(ValueError, match="unsupported corpus validation mode"):
        sft.compile_sft(compile_config, relative_to=tmp_path)

    permit = corpus._authorize_aggregate(matched, coverage, receipt)
    consumed = corpus.consume(permit, pending, arm=corpus.RATIONALE_ARM)
    assert consumed["training_permit_sha256"] == permit["sha256"]
    assert consumed["files"]["train"]["format"] == DENSE_FORMAT
    assert consumed["validation_mode"] == "task_outcomes_only"
    assert consumed["sha256"] == digest_json(
        {key: item for key, item in consumed.items() if key != "sha256"}
    )
    _write(tmp_path / "manifest.json", consumed)
    plan = sft.compile_sft(compile_config, relative_to=tmp_path)
    assert plan["schema"] == DENSE_SCHEMA
    assert plan["datasets"]["train"]["supervised_tokens"] == 20_000_000

    weak = copy.deepcopy(coverage)
    weak["arms"][corpus.ACTION_ARM]["packing_independent_unique_target_tokens"] -= 1
    weak["sha256"] = digest_json({key: item for key, item in weak.items() if key != "sha256"})
    with pytest.raises(ValueError, match="matched_actions_only arm"):
        corpus._authorize_aggregate(matched, weak, receipt)


def test_path_authorizer_rejects_aggregate_file_tampering(tmp_path: Path) -> None:
    _pending, coverage, matched, receipt = _qualified_documents()
    matched_path = tmp_path / "matched.json"
    coverage_path = tmp_path / "coverage.json"
    receipt_path = tmp_path / "receipt.json"
    _write(matched_path, matched)
    _write(coverage_path, coverage)
    receipt["matched_manifest_file_sha256"] = file_sha256(matched_path)
    receipt["coverage_file_sha256"] = file_sha256(coverage_path)
    receipt = _seal(receipt)
    _write(receipt_path, receipt)
    coverage_path.write_text(coverage_path.read_text() + "\n")
    with pytest.raises(ValueError, match="does not bind the aggregate files"):
        corpus.authorize_paths(
            matched_path,
            coverage_path,
            receipt_path,
            tmp_path / "TRAINING-PERMIT.json",
        )
    assert not (tmp_path / "TRAINING-PERMIT.json").exists()


def test_training_consumer_rejects_cross_arm_manifest() -> None:
    pending, coverage, matched, receipt = _qualified_documents()
    permit = corpus._authorize_aggregate(matched, coverage, receipt)
    with pytest.raises(ValueError, match="permitted training arm"):
        corpus.consume(permit, pending, arm=corpus.ACTION_ARM)
