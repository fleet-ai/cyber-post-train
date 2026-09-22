"""Private matched corpora for the teacher-visible-rationale campaign.

This module has no collector or network client.  It consumes only a local,
metadata-admitted selection plus a private exact-message export.  It locally
re-renders every proposed target through the pinned Qwen template and creates two
corpora from the exact same windows: ordinary student-visible rationale plus
visible actions, and a matched control in which the rationale remains context
but has zero loss.  Provider-private or hidden reasoning is never accepted.
"""

from __future__ import annotations

import copy
import json
import os
import re
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import corpus as dense_corpus
from . import dense
from . import fleet_visible_reasoning_corpus as visible
from . import teacher_visible_rationale_broad_campaign as broad
from . import teacher_visible_rationale_campaign as teacher
from .io import atomic_write_json, digest_json, file_sha256, iter_jsonl
from .sft_runtime import DENSE_FORMAT

REQUEST_SCHEMA = "cyber_teacher_visible_rationale_matched_materialization_request_v1"
PRIVATE_RECORD_SCHEMA = "cyber_teacher_visible_rationale_private_token_record_v1"
MATCHED_CORPUS_SCHEMA = "cyber_teacher_visible_rationale_matched_corpus_v1"
COVERAGE_SCHEMA = "cyber_teacher_visible_rationale_matched_coverage_v1"
RECEIPT_SCHEMA = "cyber_teacher_visible_rationale_matched_materialization_receipt_v1"
PERMIT_SCHEMA = "cyber_teacher_visible_rationale_training_permit_v1"
PENDING_MANIFEST_SCHEMA = "cyber_teacher_visible_rationale_pending_arm_v1"
TRAINING_MANIFEST_SCHEMA = "cyber_dense_sft_corpus_v1"
PENDING_VALIDATION_MODE = "pending_teacher_visible_rationale_permit"

RATIONALE_ARM = "visible_rationale_plus_actions"
ACTION_ARM = "matched_actions_only"
ARMS = (RATIONALE_ARM, ACTION_ARM)
RATIONALE_KIND = "ordinary_visible_rationale"
ACTION_KIND = "visible_tool_action"
SUMMARY_KIND = "visible_compaction_summary"
SPAN_KINDS = (RATIONALE_KIND, ACTION_KIND)
_FORBIDDEN_PRIVATE_KEYS = set(teacher._PRIVATE_FIELD_NAMES) | {
    "chain_of_thought",
    "cot",
    "internal_monologue",
    "private_analysis",
}


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return dict(value)


def _exact(value: object, fields: set[str], label: str) -> dict[str, Any]:
    result = _mapping(value, label)
    if set(result) != fields:
        raise ValueError(f"{label} has unknown or missing fields")
    return result


def _sealed(value: Mapping[str, Any], schema: str, label: str) -> dict[str, Any]:
    result = _mapping(value, label)
    if result.get("schema") != schema or result.get("sha256") != digest_json(
        {key: item for key, item in result.items() if key != "sha256"}
    ):
        raise ValueError(f"invalid {label} schema or digest")
    return result


def _sha(value: object, label: str) -> str:
    return teacher._sha(value, label)


def _text(value: object, label: str) -> str:
    return teacher._text(value, label)


def _count(value: object, label: str, *, positive: bool = False) -> int:
    return teacher._count(value, label, positive=positive)


def _path(root: Path, value: object, label: str) -> Path:
    return teacher._path(root, value, label)


def _input(root: Path, value: object, label: str) -> tuple[Path, str]:
    return teacher._input(root, value, label)


def _private_input(root: Path, value: object) -> tuple[Path, str]:
    """Bind the private record file without reading it before public gates pass."""

    reference = _exact(value, {"path", "sha256"}, "records")
    path = _path(root, reference["path"], "records path")
    expected = _sha(reference["sha256"], "records file")
    if path.is_symlink() or not path.is_file():
        raise ValueError("records path is not a regular private file")
    return path, expected


def _json(path: Path, label: str) -> dict[str, Any]:
    return teacher._json(path, label)


def _reject_hidden_keys(value: object, *, path: str = "record") -> None:
    """Reject every hidden/private reasoning surface, even if newly nested."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            lowered = str(key).lower()
            if lowered in _FORBIDDEN_PRIVATE_KEYS:
                raise ValueError(f"{path} contains forbidden private reasoning field")
            _reject_hidden_keys(item, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_hidden_keys(item, path=f"{path}[{index}]")


def _selection(value: Mapping[str, Any]) -> dict[str, Any]:
    selection = _sealed(value, teacher.SELECTION_SCHEMA, "teacher rationale selection")
    _exact(
        selection,
        {
            "schema",
            "request_files_sha256",
            "contract_files_sha256",
            "roster_files_sha256",
            "admission_input_manifest_sha256",
            "source_authorization_sha256",
            "source_authorization_authority",
            "source_profile_sha256",
            "collection_packet_sha256",
            "family_split_sha256",
            "family_role_anchor_sha256",
            "protected_family_lock_sha256",
            "records",
            "sft_ready",
            "next_gate",
            "sha256",
        },
        "teacher rationale selection",
    )
    if (
        selection["sft_ready"] is not False
        or selection["next_gate"]
        != "private_qwen_token_roundtrip_window_dedupe_and_exact_20m_coverage"
    ):
        raise ValueError("teacher rationale selection bypassed the private materializer")
    records = selection["records"]
    if not isinstance(records, list):
        raise ValueError("teacher rationale selection records must be a list")
    seen: set[str] = set()
    for raw in records:
        row = _exact(
            raw,
            {
                "record_id",
                "task_key",
                "task_version_id",
                "group_id",
                "source_session_identity_sha256",
                "normalized_record_sha256",
                "normalized_trajectory_sha256",
                "transcript_sha256",
                "attempt",
                "attempt_metadata_sha256",
                "runtime_binding_sha256",
                "target_occurrence_manifest_sha256",
                "rationale_target_tokens",
                "visible_action_target_tokens",
                "outcome",
                "visible_rationale",
                "serialization",
                "compaction",
            },
            "teacher rationale selected record",
        )
        record_id = _text(row.get("record_id"), "selected record identity")
        if record_id in seen:
            raise ValueError("teacher rationale selection duplicates a record")
        seen.add(record_id)
        if row.get("visible_rationale", {}).get("provider_private_reasoning_present") is not False:
            raise ValueError("teacher rationale selection includes private reasoning")
        if row.get("visible_rationale", {}).get("unknown_reasoning_fields_present") is not False:
            raise ValueError("teacher rationale selection includes unknown reasoning")
        if any(
            row.get("visible_rationale", {}).get("forbidden_private_field_occurrences", {}).values()
        ):
            raise ValueError("teacher rationale selection includes a forbidden private field")
    return selection


def _revalidate_admitted_records(
    selection: dict[str, Any],
    *,
    packet: dict[str, Any],
    profile: dict[str, Any],
    roster: dict[str, Any],
) -> None:
    """Rebuild each attempt and repeat the verifier-success admission check."""

    for selected in selection["records"]:
        attempt = {
            "schema": teacher.ATTEMPT_SCHEMA,
            "record_id": selected["record_id"],
            "campaign_packet_sha256": packet["sha256"],
            "source_profile_sha256": profile["sha256"],
            "task_key": selected["task_key"],
            "task_version_id": selected["task_version_id"],
            "attempt": selected["attempt"],
            "source_session_identity_sha256": selected["source_session_identity_sha256"],
            "normalized_record_sha256": selected["normalized_record_sha256"],
            "normalized_trajectory_sha256": selected["normalized_trajectory_sha256"],
            "transcript_sha256": selected["transcript_sha256"],
            "outcome": selected["outcome"],
            "visible_rationale": selected["visible_rationale"],
            "serialization": selected["serialization"],
            "compaction": selected["compaction"],
            "sha256": selected["attempt_metadata_sha256"],
        }
        checked = teacher._attempt(attempt)
        reason = teacher._admission_reason(
            checked,
            packet=packet,
            profile=profile,
            selected_tasks=roster["selected"],
            selected_runtime_bindings=roster["selected_runtime_bindings"],
        )
        if reason is not None:
            raise ValueError(f"teacher selection no longer passes admission: {reason}")
        identity = (selected["task_key"], selected["task_version_id"])
        assignment = roster["selected"].get(identity)
        binding = roster["selected_runtime_bindings"].get(identity)
        if (
            assignment is None
            or binding is None
            or selected["group_id"] != assignment["group_id"]
            or selected["runtime_binding_sha256"] != digest_json(binding)
        ):
            raise ValueError("teacher selection changes the exact train-family runtime binding")


def _admission_receipt(value: Mapping[str, Any], selection: dict[str, Any]) -> dict[str, Any]:
    receipt = _sealed(value, teacher.RECEIPT_SCHEMA, "teacher rationale admission receipt")
    if (
        receipt.get("selection_sha256") != selection["sha256"]
        or receipt.get("source_profile_sha256") != selection["source_profile_sha256"]
        or receipt.get("collection_packet_sha256") != selection["collection_packet_sha256"]
        or receipt.get("heldout_families_admitted") != 0
        or receipt.get("source_text_read") is not False
        or receipt.get("token_ids_read") is not False
        or receipt.get("sft_ready") is not False
    ):
        raise ValueError("teacher rationale admission receipt differs from its private selection")
    return receipt


def _matched_plan(value: Mapping[str, Any], packet: dict[str, Any]) -> dict[str, Any]:
    plan = _sealed(value, broad.MATCHED_SCHEMA, "matched materialization plan")
    if (
        plan.get("collection_packet_sha256") != packet["sha256"]
        or plan.get("single_selected_success_set") is not True
        or plan.get("single_packed_window_selection") is not True
        or plan.get("same_serialized_messages_actions_and_compaction") is not True
        or plan.get("loss_arms") != broad.LOSS_ARMS
        or plan.get("minimum_unique_supervised_tokens_per_arm")
        != packet["minimum_unique_supervised_tokens"]
        or plan.get("provider_private_or_hidden_reasoning_allowed") is not False
        or plan.get("external_submission_authorized") is not False
        or plan.get("training_authorized") is not False
        or plan.get("submitted") is not False
    ):
        raise ValueError("matched materialization plan is not the reviewed two-arm contract")
    return plan


def _token_ids(value: object, label: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must contain token IDs")
    if any(type(item) is not int or item < 0 for item in value):
        raise ValueError(f"{label} contains an invalid token ID")
    return list(value)


def _span(value: object, input_ids: list[int]) -> dict[str, Any]:
    span = _exact(
        value,
        {
            "target_occurrence_sha256",
            "kind",
            "source_message_index",
            "token_start",
            "token_end",
            "source_token_offset",
            "token_ids_sha256",
        },
        "teacher visible target span",
    )
    _sha(span["target_occurrence_sha256"], "target occurrence")
    if span["kind"] not in SPAN_KINDS:
        raise ValueError("teacher target span is not ordinary visible rationale/action context")
    start = _count(span["token_start"], "target token start")
    end = _count(span["token_end"], "target token end", positive=True)
    _count(span["source_token_offset"], "source token offset")
    _count(span["source_message_index"], "source message index")
    if not start < end <= len(input_ids):
        raise ValueError("teacher target span escapes its packed window")
    if span["token_ids_sha256"] != digest_json(input_ids[start:end]):
        raise ValueError("teacher target span token digest mismatch")
    return span


def _source_occurrence_sha256(
    *,
    selected: Mapping[str, Any],
    kind: str,
    source_message_index: int,
    source_token_offset: int,
    token_ids_sha256: str,
    source_token_count: int,
) -> str:
    """Derive one source target identity before any packing decision.

    A caller-supplied opaque occurrence ID is not sufficient: the same source
    tokens could be repacked into several windows and assigned fresh IDs.  The
    identity therefore binds the admitted record, semantic target kind, exact
    source offset and width, and token digest.  Window IDs and packed offsets
    are deliberately absent.
    """

    return digest_json(
        {
            "normalized_record_sha256": selected["normalized_record_sha256"],
            "kind": kind,
            "source_message_index": source_message_index,
            "source_token_offset": source_token_offset,
            "source_token_count": source_token_count,
            "token_ids_sha256": token_ids_sha256,
        }
    )


def _template_ids(tokenizer: Any, messages: list[dict[str, Any]], *, generation: bool) -> list[int]:
    """Render only through the locally pinned Qwen template with thinking off."""

    try:
        rendered = tokenizer.apply_chat_template(
            [dense._normalized_for_template(message) for message in messages],
            tokenize=True,
            add_generation_prompt=generation,
            tools=[],
            enable_thinking=False,
        )
    except Exception as error:
        raise ValueError("pinned Qwen teacher-visible template rendering failed") from error
    if isinstance(rendered, Mapping):
        rendered = rendered.get("input_ids")
    if isinstance(rendered, list) and len(rendered) == 1 and isinstance(rendered[0], list):
        rendered = rendered[0]
    return _token_ids(rendered, "pinned Qwen teacher-visible token IDs")


def _load_tokenizer(
    requirements: dict[str, Any], tokenizer_root: Path
) -> tuple[Any, dict[str, Any]]:
    model_lock_path = _path(
        teacher.REPOSITORY_ROOT,
        requirements["qwen_target"]["model_lock_path"],
        "Qwen model lock",
    )
    lock = _json(model_lock_path, "Qwen model lock")
    target = requirements["qwen_target"]
    tokenizer, identity = dense_corpus.local_tokenizer(lock, tokenizer_root)
    expected_files = lock["tokenizer"]["files"]
    actual = {
        "repo": identity.get("repo"),
        "revision": identity.get("revision"),
        "files": identity.get("files"),
        "chat_template_sha256": "sha256:"
        + _text(identity.get("chat_template_sha256"), "local Qwen chat template digest"),
    }
    expected = {
        "repo": target["repository"],
        "revision": target["revision"],
        "files": expected_files,
        "chat_template_sha256": target["chat_template_sha256"],
    }
    if actual != expected:
        raise ValueError("local Qwen tokenizer/template differs from the exact target lock")
    return tokenizer, actual


def _sentence_count(text: str) -> int:
    """Count nonempty visible sentences using a fixed conservative boundary."""

    return len(
        [segment for segment in re.split(r"(?<=[.!?])(?:\s+|$)", text.strip()) if segment.strip()]
    )


def _messages(value: object, selected: Mapping[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError("private teacher record has no exact visible message stream")
    messages = [visible._message(message) for message in value]
    if [message["role"] for message in messages[:2]] != ["system", "user"] or any(
        message["role"] in {"system", "user"} for message in messages[2:]
    ):
        raise ValueError("teacher message stream changes the OpenCode system/task anchor")
    visible._conversation(messages)
    if digest_json(messages) != selected["normalized_trajectory_sha256"]:
        raise ValueError("teacher message stream differs from admitted trajectory evidence")
    tool_calls = 0
    rationale_segments = 0
    sentence_counts = []
    for message in messages:
        calls = message.get("tool_calls") or []
        if not calls:
            continue
        text = message["content"].strip()
        sentences = _sentence_count(text)
        if not text or not 1 <= sentences <= 4:
            raise ValueError("teacher tool call lacks one-to-four visible rationale sentences")
        tool_calls += len(calls)
        rationale_segments += 1
        sentence_counts.append(sentences)
    admitted = selected["visible_rationale"]
    if (
        not sentence_counts
        or tool_calls != admitted["tool_calls"]
        or tool_calls != admitted["tool_calls_with_visible_rationale"]
        or rationale_segments != admitted["rationale_segments"]
        or min(sentence_counts) != admitted["minimum_sentences_per_rationale"]
        or max(sentence_counts) != admitted["maximum_sentences_per_rationale"]
    ):
        raise ValueError("teacher visible-rationale counts differ from admitted evidence")
    return messages


def _rendered_window(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    window: dict[str, Any],
) -> None:
    indices = window["message_indices"]
    if max(indices) >= len(messages):
        raise ValueError("teacher packed window names a message outside its source record")
    selected = [messages[index] for index in indices]
    target_index = window["target_message_index"]
    target = messages[target_index]
    if (
        selected[-1] != target
        or target["role"] != "assistant"
        or not target.get("tool_calls")
        or not target["content"].strip()
    ):
        raise ValueError("teacher target is not a visible-rationale assistant tool turn")
    prompt = _template_ids(tokenizer, selected[:-1], generation=True)
    rendered = _template_ids(tokenizer, selected, generation=False)
    if (
        rendered != window["input_ids"]
        or prompt != rendered[: window["prompt_token_count"]]
        or len(prompt) != window["prompt_token_count"]
    ):
        raise ValueError("teacher packed window differs from local Qwen/OpenCode rendering")
    if any(span["source_message_index"] != target_index for span in window["target_spans"]):
        raise ValueError("teacher target span is not bound to its exact assistant message")
    covered = []
    for span in window["target_spans"]:
        covered.extend(range(span["token_start"], span["token_end"]))
    if covered != list(range(window["prompt_token_count"], len(rendered))):
        raise ValueError("teacher target spans do not cover the exact assistant continuation once")

    content_only = {"role": "assistant", "content": target["content"]}
    content_rendered = _template_ids(tokenizer, [*selected[:-1], content_only], generation=False)
    if content_rendered[: len(prompt)] != prompt:
        raise ValueError("teacher visible content changes the exact Qwen assistant boundary")
    full_continuation = rendered[len(prompt) :]
    content_continuation = content_rendered[len(prompt) :]
    rationale_width = 0
    for full_token, content_token in zip(full_continuation, content_continuation, strict=False):
        if full_token != content_token:
            break
        rationale_width += 1
    action_only = {
        "role": "assistant",
        "content": "",
        "tool_calls": target["tool_calls"],
    }
    action_rendered = _template_ids(tokenizer, [*selected[:-1], action_only], generation=False)
    if action_rendered[: len(prompt)] != prompt:
        raise ValueError("teacher visible action changes the exact Qwen assistant boundary")
    action_continuation = action_rendered[len(prompt) :]
    action_tokens = full_continuation[rationale_width:]
    if (
        rationale_width < 1
        or not action_tokens
        or len(action_continuation) < len(action_tokens)
        or action_continuation[-len(action_tokens) :] != action_tokens
    ):
        raise ValueError("teacher Qwen template cannot separate visible rationale from actions")
    expected_ranges = {
        RATIONALE_KIND: list(range(len(prompt), len(prompt) + rationale_width)),
        ACTION_KIND: list(range(len(prompt) + rationale_width, len(rendered))),
    }
    actual_ranges = {kind: [] for kind in (RATIONALE_KIND, ACTION_KIND)}
    for span in window["target_spans"]:
        actual_ranges[span["kind"]].extend(range(span["token_start"], span["token_end"]))
    if actual_ranges != expected_ranges:
        raise ValueError("teacher target labels do not match visible content and tool actions")


def _rendered_compaction(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    windows: list[dict[str, Any]],
    compaction: dict[str, Any],
) -> None:
    """Bind every visible compaction summary to one exact parent/child transition."""

    if compaction["kind"] == "none":
        return
    message_digests: dict[str, list[int]] = defaultdict(list)
    for index, message in enumerate(messages):
        message_digests[digest_json(message)].append(index)
    windows_by_id = {window["window_id"]: window for window in windows}
    if len(windows_by_id) != len(windows):
        raise ValueError("teacher compaction windows do not have unique identities")
    ordered_windows = sorted(windows, key=lambda window: window["sequence_index"])
    position = {window["window_id"]: index for index, window in enumerate(ordered_windows)}
    summary_indices: set[int] = set()
    previous_parent_position: int | None = None
    previous_next_position: int | None = None
    previous_summary_index: int | None = None
    previous_next_target_index: int | None = None
    previous_post_indices: list[int] | None = None
    for boundary in compaction["boundaries"]:
        summary_index = boundary["summary_message_index"]
        candidates = message_digests.get(boundary["visible_summary_message_sha256"], [])
        if len(candidates) != 1 or candidates[0] != summary_index or summary_index >= len(messages):
            raise ValueError("teacher compaction summary is not one exact visible message")
        summary = messages[summary_index]
        parent = windows_by_id.get(boundary["parent_window_id"])
        next_window = windows_by_id.get(boundary["next_target_window_id"])
        parent_position = position.get(boundary["parent_window_id"])
        next_position = position.get(boundary["next_target_window_id"])
        if (
            parent is None
            or next_window is None
            or parent_position is None
            or next_position is None
            or next_position != parent_position + 1
            or parent["target_message_index"] != boundary["parent_target_message_index"]
            or next_window["target_message_index"] != boundary["next_target_message_index"]
            or parent["target_message_index"] >= summary_index
            or summary_index >= next_window["target_message_index"]
            or boundary["next_target_occurrence_sha256"]
            not in {
                span["target_occurrence_sha256"]
                for span in next_window["target_spans"]
                if span["kind"] != SUMMARY_KIND
            }
            or summary_index in {window["target_message_index"] for window in windows}
        ):
            raise ValueError("teacher compaction is not one immediate parent/next-target boundary")
        if (
            summary_index in summary_indices
            or summary["role"] != "assistant"
            or set(summary) != {"role", "content"}
            or not summary["content"].strip()
        ):
            raise ValueError("teacher compaction summary is hidden, duplicated, or not plain text")
        summary_indices.add(summary_index)

        pre_indices = boundary["pre_compaction_prompt_message_indices"]
        summary_prompt_indices = boundary["summary_generation_message_indices"]
        post_indices = boundary["post_compaction_prompt_message_indices"]
        if previous_post_indices is None:
            expected_summary_indices = list(range(summary_index))
        elif previous_next_target_index is None:  # pragma: no cover - local invariant
            raise ValueError("teacher compaction chain lost its preceding target")
        else:
            expected_summary_indices = [
                *previous_post_indices,
                *range(previous_next_target_index, summary_index),
            ]
        if (
            pre_indices != parent["message_indices"][:-1]
            or summary_prompt_indices != expected_summary_indices
            or post_indices != next_window["message_indices"][:-1]
            or summary_index not in post_indices
            or pre_indices == post_indices
        ):
            raise ValueError("teacher compaction message ancestry is cross-wired")

        pre_prompt = _template_ids(
            tokenizer,
            [messages[index] for index in pre_indices],
            generation=True,
        )
        summary_prompt = _template_ids(
            tokenizer,
            [messages[index] for index in summary_prompt_indices],
            generation=True,
        )
        summary_rendered = _template_ids(
            tokenizer,
            [*[messages[index] for index in summary_prompt_indices], summary],
            generation=False,
        )
        continuation = (
            summary_rendered[len(summary_prompt) :]
            if summary_rendered[: len(summary_prompt)] == summary_prompt
            else []
        )
        post_prompt = _template_ids(
            tokenizer,
            [messages[index] for index in post_indices],
            generation=True,
        )
        if (
            not continuation
            or digest_json(pre_prompt) != boundary["pre_compaction_prompt_sha256"]
            or len(pre_prompt) != boundary["pre_compaction_prompt_tokens"]
            or digest_json(summary_prompt) != boundary["summary_generation_prompt_sha256"]
            or len(summary_prompt) != boundary["summary_generation_prompt_tokens"]
            or digest_json(continuation) != boundary["visible_summary_qwen_token_sha256"]
            or len(continuation) != boundary["visible_summary_qwen_tokens"]
            or digest_json(post_prompt) != boundary["post_compaction_prompt_sha256"]
            or len(post_prompt) != boundary["post_compaction_prompt_tokens"]
            or pre_prompt == post_prompt
            or parent["prompt_token_sha256"] != boundary["pre_compaction_prompt_sha256"]
            or parent["prompt_token_count"] != boundary["pre_compaction_prompt_tokens"]
            or next_window["prompt_token_sha256"] != boundary["post_compaction_prompt_sha256"]
            or next_window["prompt_token_count"] != boundary["post_compaction_prompt_tokens"]
            or next_window["prompt_token_sha256"] != boundary["next_target_prompt_sha256"]
        ):
            raise ValueError("teacher compaction does not bind a real visible continuation")
        if (
            (previous_parent_position is not None and parent_position <= previous_parent_position)
            or (previous_next_position is not None and parent_position < previous_next_position)
            or (previous_summary_index is not None and summary_index <= previous_summary_index)
            or (
                previous_next_target_index is not None
                and parent["target_message_index"] < previous_next_target_index
            )
        ):
            raise ValueError("teacher compaction boundaries reverse or cross their chronology")
        previous_parent_position = parent_position
        previous_next_position = next_position
        previous_summary_index = summary_index
        previous_next_target_index = next_window["target_message_index"]
        previous_post_indices = post_indices


def _private_record(
    value: Mapping[str, Any],
    selected: dict[str, Any],
    tokenizer: Any,
    *,
    context_window_tokens: int,
) -> dict[str, Any]:
    # The exact schema check rejects hidden fields at every declared level;
    # this recursive check also catches a newly nested provider field.
    _reject_hidden_keys(value)
    record = _exact(
        value,
        {
            "schema",
            "record_id",
            "source_profile_sha256",
            "collection_packet_sha256",
            "selection_record_sha256",
            "lineage",
            "evidence",
            "reasoning_visibility",
            "private_reasoning_present",
            "original_task_digest",
            "messages",
            "windows",
            "compaction",
            "content_digest",
        },
        "private teacher token record",
    )
    if record["schema"] != PRIVATE_RECORD_SCHEMA or record["content_digest"] != digest_json(
        {key: item for key, item in record.items() if key != "content_digest"}
    ):
        raise ValueError("private teacher token record digest or schema mismatch")
    expected = {
        "record_id": selected["record_id"],
        "source_profile_sha256": selected["source_profile_sha256"],
        "collection_packet_sha256": selected["collection_packet_sha256"],
        "selection_record_sha256": selected["attempt_metadata_sha256"],
    }
    if any(record.get(name) != target for name, target in expected.items()):
        raise ValueError("private teacher token record changes its admitted source")
    if (
        record["reasoning_visibility"] != "student_visible_ordinary_assistant_content"
        or record["private_reasoning_present"] is not False
    ):
        raise ValueError("private or hidden teacher reasoning cannot be materialized")
    _sha(record["original_task_digest"], "teacher original task")
    lineage = _exact(
        record["lineage"],
        {"task_key", "task_version_id", "group_id", "attempt"},
        "private teacher lineage",
    )
    if lineage != {
        "task_key": selected["task_key"],
        "task_version_id": selected["task_version_id"],
        "group_id": selected["group_id"],
        "attempt": selected["attempt"],
    }:
        raise ValueError("private teacher token record changes task/family/attempt lineage")
    evidence = _exact(
        record["evidence"],
        {
            "source_session_identity_sha256",
            "normalized_record_sha256",
            "normalized_trajectory_sha256",
            "transcript_sha256",
            "target_occurrence_manifest_sha256",
        },
        "private teacher evidence",
    )
    for name, target in evidence.items():
        _sha(target, f"private teacher evidence {name}")
        if target != selected[name]:
            raise ValueError("private teacher token record changes admitted evidence")
    checked_messages = _messages(record["messages"], selected)
    windows = record["windows"]
    if not isinstance(windows, list) or not windows:
        raise ValueError("private teacher token record has no packed windows")
    checked_windows = []
    for raw_window in windows:
        window = _exact(
            raw_window,
            {
                "window_id",
                "sequence_index",
                "message_indices",
                "target_message_index",
                "input_ids",
                "prompt_token_count",
                "prompt_token_sha256",
                "serialized_token_ids_sha256",
                "target_spans",
                "window_payload_sha256",
            },
            "private teacher packed window",
        )
        _text(window["window_id"], "packed window identity")
        _count(window["sequence_index"], "packed window sequence")
        message_indices = window["message_indices"]
        if (
            not isinstance(message_indices, list)
            or not message_indices
            or any(type(index) is not int or index < 0 for index in message_indices)
            or message_indices != sorted(set(message_indices))
        ):
            raise ValueError("teacher packed window message indices are not strictly ordered")
        target_message_index = _count(
            window["target_message_index"], "teacher target message index"
        )
        if target_message_index != message_indices[-1]:
            raise ValueError("teacher target message is not final in its packed window")
        input_ids = _token_ids(window["input_ids"], "packed window input")
        if len(input_ids) > context_window_tokens:
            raise ValueError("teacher packed window exceeds the exact Qwen context limit")
        prompt_token_count = _count(window["prompt_token_count"], "teacher prompt tokens")
        if not prompt_token_count < len(input_ids):
            raise ValueError("teacher packed window has no assistant continuation")
        if window["prompt_token_sha256"] != digest_json(input_ids[:prompt_token_count]) or window[
            "serialized_token_ids_sha256"
        ] != digest_json(input_ids):
            raise ValueError("teacher packed window token digests mismatch")
        spans = window["target_spans"]
        if not isinstance(spans, list) or not spans:
            raise ValueError("private teacher packed window has no target spans")
        checked_spans = [_span(span, input_ids) for span in spans]
        if {span["kind"] for span in checked_spans if span["kind"] != SUMMARY_KIND} != {
            RATIONALE_KIND,
            ACTION_KIND,
        }:
            raise ValueError(
                "every matched teacher window must contain visible rationale and action targets"
            )
        payload = {
            "window_id": window["window_id"],
            "sequence_index": window["sequence_index"],
            "message_indices": message_indices,
            "target_message_index": target_message_index,
            "input_ids": input_ids,
            "prompt_token_count": prompt_token_count,
            "prompt_token_sha256": window["prompt_token_sha256"],
            "serialized_token_ids_sha256": window["serialized_token_ids_sha256"],
            "target_spans": checked_spans,
        }
        if window["window_payload_sha256"] != digest_json(payload):
            raise ValueError("private teacher packed window payload digest mismatch")
        checked_window = {**payload, "window_payload_sha256": window["window_payload_sha256"]}
        _rendered_window(tokenizer, checked_messages, checked_window)
        checked_windows.append(checked_window)
    if len({window["window_id"] for window in checked_windows}) != len(checked_windows) or [
        window["sequence_index"] for window in checked_windows
    ] != list(range(len(checked_windows))):
        raise ValueError("private teacher windows duplicate or skip their exact order")
    target_message_indices = [window["target_message_index"] for window in checked_windows]
    if len(set(target_message_indices)) != len(target_message_indices):
        raise ValueError("teacher target assistant message is repeated across packed windows")
    if target_message_indices != sorted(target_message_indices):
        raise ValueError("teacher target assistant messages are not in exact chronological order")
    expected_offset = 0
    target_tokens = {RATIONALE_KIND: 0, ACTION_KIND: 0}
    for window in checked_windows:
        for span in window["target_spans"]:
            if span["kind"] == SUMMARY_KIND:
                continue
            if span["source_token_offset"] != expected_offset:
                raise ValueError("teacher source token offsets skip or repeat target tokens")
            width = span["token_end"] - span["token_start"]
            expected_offset += width
            target_tokens[span["kind"]] += width
    if (
        target_tokens[RATIONALE_KIND] != selected["rationale_target_tokens"]
        or target_tokens[ACTION_KIND] != selected["visible_action_target_tokens"]
    ):
        raise ValueError("teacher rendered target-token counts differ from admitted evidence")
    checked_compaction = teacher._compaction(
        record["compaction"],
        source_session_identity_sha256=evidence["source_session_identity_sha256"],
        normalized_trajectory_sha256=evidence["normalized_trajectory_sha256"],
        transcript_sha256=evidence["transcript_sha256"],
        target_occurrence_manifest_sha256=evidence["target_occurrence_manifest_sha256"],
    )
    _rendered_compaction(tokenizer, checked_messages, checked_windows, checked_compaction)
    return {
        **record,
        "messages": checked_messages,
        "windows": checked_windows,
        "compaction": checked_compaction,
    }


def _mask(length: int, spans: list[dict[str, Any]], arm: str) -> list[int]:
    mask = [0] * length
    included = {ACTION_KIND} if arm == ACTION_ARM else {RATIONALE_KIND, ACTION_KIND}
    for span in spans:
        if span["kind"] in included:
            start, end = span["token_start"], span["token_end"]
            if any(mask[start:end]):
                raise ValueError("teacher target spans overlap in one packed window")
            mask[start:end] = [1] * (end - start)
    return mask


def _arm_gate(
    *,
    unique_target_tokens: int,
    family_tokens: Mapping[str, int],
    packet: dict[str, Any],
) -> dict[str, Any]:
    unique = unique_target_tokens
    families = len(family_tokens)
    fraction = families / packet["train_task_versions"]
    largest = max(family_tokens.values()) / unique if unique else 0.0
    minimum = packet["minimum_unique_supervised_tokens"]
    within = largest <= packet["maximum_family_target_token_fraction"] if unique else False
    ready = (
        unique >= minimum
        and families >= packet["minimum_selected_families"]
        and fraction >= packet["minimum_selected_family_fraction"]
        and within
    )
    return {
        "packing_independent_unique_target_tokens": unique,
        "minimum_unique_target_tokens": minimum,
        "families_with_targets": families,
        "minimum_selected_families": packet["minimum_selected_families"],
        "selected_family_fraction": fraction,
        "minimum_selected_family_fraction": packet["minimum_selected_family_fraction"],
        "largest_family_target_token_fraction": largest,
        "maximum_family_target_token_fraction": packet["maximum_family_target_token_fraction"],
        "family_concentration_within_limit": within,
        "qualified": ready,
    }


def _tokenizer_manifest(requirements: dict[str, Any], root: Path) -> dict[str, Any]:
    lock = _json(
        _path(root, requirements["qwen_target"]["model_lock_path"], "Qwen model lock"),
        "Qwen model lock",
    )
    tokenizer = _mapping(lock.get("tokenizer"), "Qwen tokenizer lock")
    if (
        lock.get("repo") != teacher.QWEN_REPOSITORY
        or lock.get("revision") != teacher.QWEN_REVISION
        or tokenizer.get("manifest_sha256")
        != requirements["qwen_target"]["tokenizer_manifest_sha256"]
    ):
        raise ValueError("teacher corpus tokenizer differs from the exact Qwen target")
    return {
        "repo": lock["repo"],
        "revision": lock["revision"],
        "files": tokenizer["files"],
        "manifest_sha256": tokenizer["manifest_sha256"],
        "chat_template_sha256": requirements["qwen_target"]["chat_template_sha256"],
    }


def _arm_manifest(
    *,
    arm: str,
    packet: dict[str, Any],
    profile: dict[str, Any],
    selection: dict[str, Any],
    matched_plan: dict[str, Any],
    tokenizer: dict[str, Any],
    rows: list[dict[str, Any]],
    file_digest: str,
    task_keys: list[str],
    supervised_tokens: int,
    source_records: int,
    selected_occurrences: int,
    total_occurrences: int,
) -> dict[str, Any]:
    manifest = {
        "schema": PENDING_MANIFEST_SCHEMA,
        "source_sha256": digest_json(
            {
                "source_profile_sha256": profile["sha256"],
                "collection_packet_sha256": packet["sha256"],
                "selection_sha256": selection["sha256"],
                "matched_plan_sha256": matched_plan["sha256"],
                "arm": arm,
            }
        ),
        "split_sha256": packet["family_split_sha256"],
        "tokenizer": tokenizer,
        "files": {
            "train": {
                "path": "train.parquet",
                "sha256": file_digest,
                "rows": len(rows),
                "task_keys": task_keys,
                "format": DENSE_FORMAT,
                "supervised_tokens": supervised_tokens,
                "assistant_responses": selected_occurrences,
                "source_sessions": source_records,
                "source_total_assistant_responses": total_occurrences,
                "excluded_assistant_responses": total_occurrences - selected_occurrences,
            }
        },
        "validation_mode": PENDING_VALIDATION_MODE,
        "objective": arm,
        "training_permit_sha256": None,
    }
    manifest["sha256"] = digest_json(manifest)
    return manifest


def build(config: Mapping[str, Any], *, relative_to: Path) -> dict[str, Any]:
    """Create two private token-only arms without authorizing training."""

    import pyarrow as pa
    import pyarrow.parquet as pq

    request = _exact(
        config,
        {
            "schema",
            "requirements",
            "source_authorization",
            "source_profile",
            "collection_packet",
            "admission_selection",
            "admission_receipt",
            "matched_plan",
            "tokenizer_root",
            "records",
            "output",
        },
        "teacher matched materialization request",
    )
    if request["schema"] != REQUEST_SCHEMA:
        raise ValueError("unsupported teacher matched materialization request")
    output = _path(relative_to, request["output"], "teacher matched corpus output")
    if output.exists() or output.is_symlink():
        raise FileExistsError("teacher matched corpus destination already exists")
    public_names = (
        "requirements",
        "source_authorization",
        "source_profile",
        "collection_packet",
        "admission_selection",
        "admission_receipt",
        "matched_plan",
    )
    sources = {name: _input(relative_to, request[name], name) for name in public_names}
    records_path, records_sha256 = _private_input(relative_to, request["records"])
    sources["records"] = (records_path, records_sha256)
    paths = {name: item[0] for name, item in sources.items()}
    requirements = teacher._requirements(
        _json(paths["requirements"], "teacher rationale requirements"),
        root=teacher.REPOSITORY_ROOT,
    )
    authorization = teacher._authorization(
        _json(paths["source_authorization"], "teacher source authorization"), requirements
    )
    profile = teacher._profile(_json(paths["source_profile"], "teacher rationale profile"))
    packet = teacher._packet(_json(paths["collection_packet"], "teacher rationale packet"), profile)
    selection = _selection(
        _json(paths["admission_selection"], "teacher rationale admission selection")
    )
    receipt = _admission_receipt(
        _json(paths["admission_receipt"], "teacher rationale admission receipt"), selection
    )
    plan = _matched_plan(_json(paths["matched_plan"], "matched plan"), packet)
    expected = teacher.render(requirements, authorization, root=teacher.REPOSITORY_ROOT)
    if (
        profile != expected["source-profile.json"]
        or packet != expected["collection-packet.json"]
        or selection["source_authorization_sha256"] != authorization["sha256"]
        or selection["source_authorization_authority"] != authorization["authority"]
        or selection["source_profile_sha256"] != profile["sha256"]
        or selection["collection_packet_sha256"] != packet["sha256"]
        or receipt["collection_packet_sha256"] != packet["sha256"]
    ):
        raise ValueError("teacher materialization inputs change the exact campaign")
    roster = teacher._roster(requirements, root=teacher.REPOSITORY_ROOT)
    _revalidate_admitted_records(selection, packet=packet, profile=profile, roster=roster)
    tokenizer_root = _path(relative_to, request["tokenizer_root"], "local Qwen tokenizer root")
    if tokenizer_root.is_symlink() or not tokenizer_root.is_dir():
        raise ValueError("local Qwen tokenizer root must be a regular directory")
    tokenizer, tokenizer_identity = _load_tokenizer(requirements, tokenizer_root)
    if file_sha256(records_path) != records_sha256:
        raise ValueError("private teacher token record file digest mismatch")
    selected = {
        row["record_id"]: {
            **row,
            "source_profile_sha256": selection["source_profile_sha256"],
            "collection_packet_sha256": selection["collection_packet_sha256"],
        }
        for row in selection["records"]
    }
    records: dict[str, dict[str, Any]] = {}
    for raw in iter_jsonl(paths["records"]):
        record_id = raw.get("record_id")
        if record_id not in selected:
            raise ValueError("private teacher token record is not in the admitted selection")
        checked = _private_record(
            raw,
            selected[record_id],
            tokenizer,
            context_window_tokens=profile["opencode"]["context_window_tokens"],
        )
        if record_id in records:
            raise ValueError("private teacher token records duplicate a record")
        records[record_id] = checked
    if set(records) != set(selected):
        raise ValueError("private teacher token records do not cover the exact admitted selection")

    arm_rows: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
    arm_unique_tokens = {arm: 0 for arm in ARMS}
    arm_family_tokens: dict[str, dict[str, int]] = {arm: defaultdict(int) for arm in ARMS}
    occurrence_ids: set[str] = set()
    source_intervals: dict[str, list[tuple[int, int]]] = defaultdict(list)
    window_payloads: set[str] = set()
    total_occurrences = 0
    arm_span_counts = {arm: 0 for arm in ARMS}
    selection_rows = []
    for record_id, selected_row in sorted(selected.items()):
        record = records[record_id]
        record_occurrences: list[dict[str, Any]] = []
        for window in record["windows"]:
            if window["window_payload_sha256"] in window_payloads:
                raise ValueError("one packed teacher window is repeated")
            window_payloads.add(window["window_payload_sha256"])
            for span in window["target_spans"]:
                if span["kind"] == SUMMARY_KIND:
                    continue
                occurrence = span["target_occurrence_sha256"]
                start, end = span["token_start"], span["token_end"]
                width = end - start
                expected_occurrence = _source_occurrence_sha256(
                    selected=selected_row,
                    kind=span["kind"],
                    source_message_index=span["source_message_index"],
                    source_token_offset=span["source_token_offset"],
                    token_ids_sha256=span["token_ids_sha256"],
                    source_token_count=width,
                )
                if occurrence != expected_occurrence:
                    raise ValueError(
                        "teacher source target identity depends on packing or changes source bytes"
                    )
                if occurrence in occurrence_ids:
                    raise ValueError("one teacher source target occurs in more than one window")
                occurrence_ids.add(occurrence)
                source_start = span["source_token_offset"]
                source_end = source_start + width
                intervals = source_intervals[record_id]
                if any(
                    prior_start < source_end and source_start < prior_end
                    for prior_start, prior_end in intervals
                ):
                    raise ValueError("teacher source target token ranges overlap after repacking")
                intervals.append((source_start, source_end))
                record_occurrences.append(
                    {
                        "target_occurrence_sha256": occurrence,
                        "kind": span["kind"],
                        "source_message_index": span["source_message_index"],
                        "source_token_offset": source_start,
                        "source_token_count": width,
                        "source_token_sha256": span["token_ids_sha256"],
                    }
                )
                total_occurrences += 1
                included_arms = (
                    (RATIONALE_ARM, ACTION_ARM) if span["kind"] == ACTION_KIND else (RATIONALE_ARM,)
                )
                for arm in included_arms:
                    arm_span_counts[arm] += 1
                    # Target-occurrence identity is defined before packing and
                    # is globally unique above. Its exact source width can be
                    # counted directly; keeping 20M per-token hashes in memory
                    # would add no information and would be operationally
                    # wasteful.
                    arm_unique_tokens[arm] += width
                    arm_family_tokens[arm][selected_row["group_id"]] += width
            common = {
                "input_ids": window["input_ids"],
                "source_record_sha256": selected_row["normalized_record_sha256"],
                "source_task_version_id": selected_row["task_version_id"],
                "source_group_id": selected_row["group_id"],
                "source_window_sha256": window["window_payload_sha256"],
                "source_profile_sha256": profile["sha256"],
                "collection_packet_sha256": packet["sha256"],
            }
            for arm in ARMS:
                mask = _mask(len(window["input_ids"]), window["target_spans"], arm)
                if any(mask):
                    arm_rows[arm].append({**common, "loss_mask": mask})
        if (
            digest_json(
                sorted(record_occurrences, key=lambda item: item["target_occurrence_sha256"])
            )
            != record["evidence"]["target_occurrence_manifest_sha256"]
        ):
            raise ValueError(
                "private teacher target occurrences differ from their admitted manifest"
            )
        if not any(item["kind"] == RATIONALE_KIND for item in record_occurrences) or not any(
            item["kind"] == ACTION_KIND for item in record_occurrences
        ):
            raise ValueError("each teacher record needs paired visible rationale and actions")
        selection_rows.append(
            {
                "record_id": record_id,
                "normalized_record_sha256": selected_row["normalized_record_sha256"],
                "group_id": selected_row["group_id"],
                "target_occurrence_manifest_sha256": selected_row[
                    "target_occurrence_manifest_sha256"
                ],
            }
        )
    if not all(arm_rows.values()):
        raise ValueError("teacher matched materialization produced an empty arm")
    gates = {
        arm: _arm_gate(
            unique_target_tokens=arm_unique_tokens[arm],
            family_tokens=arm_family_tokens[arm],
            packet=packet,
        )
        for arm in ARMS
    }
    coverage = {
        "schema": COVERAGE_SCHEMA,
        "source_profile_sha256": profile["sha256"],
        "collection_packet_sha256": packet["sha256"],
        "admission_selection_sha256": selection["sha256"],
        "admission_receipt_sha256": receipt["sha256"],
        "matched_plan_sha256": plan["sha256"],
        "selected_source_records": len(selection_rows),
        "packed_windows": len(window_payloads),
        "same_packed_window_selection": True,
        "same_input_ids_and_order": True,
        "only_loss_mask_differs": True,
        "packing_independent_target_occurrence_deduplication": True,
        "arms": gates,
        "heldout_families_materialized": 0,
        "external_benchmarks_materialized": [],
        "provider_private_or_hidden_reasoning_materialized": False,
        "raw_text_written": False,
        "training_ready": all(gate["qualified"] for gate in gates.values()),
    }
    coverage["sha256"] = digest_json(coverage)
    tokenizer_manifest = _tokenizer_manifest(requirements, teacher.REPOSITORY_ROOT)
    if any(
        tokenizer_manifest[name] != tokenizer_identity[name]
        for name in ("repo", "revision", "files", "chat_template_sha256")
    ):
        raise ValueError("teacher corpus output tokenizer differs from materialization bytes")
    task_keys = sorted({row["task_key"] for row in selected.values()})

    output.parent.mkdir(parents=True, exist_ok=True)
    os.mkdir(output, 0o700)
    manifests: dict[str, dict[str, Any]] = {}
    try:
        for arm in ARMS:
            arm_dir = output / arm
            os.mkdir(arm_dir, 0o700)
            train_path = arm_dir / "train.parquet"
            pq.write_table(pa.Table.from_pylist(arm_rows[arm]), train_path, compression="zstd")
            os.chmod(train_path, 0o600)
            if pq.read_table(train_path).to_pylist() != arm_rows[arm]:
                raise ValueError("teacher matched Parquet readback differs")
            manifest = _arm_manifest(
                arm=arm,
                packet=packet,
                profile=profile,
                selection=selection,
                matched_plan=plan,
                tokenizer=tokenizer_manifest,
                rows=arm_rows[arm],
                file_digest=file_sha256(train_path),
                task_keys=task_keys,
                supervised_tokens=arm_unique_tokens[arm],
                source_records=len(selection_rows),
                selected_occurrences=arm_span_counts[arm],
                total_occurrences=total_occurrences,
            )
            manifests[arm] = manifest
            atomic_write_json(arm_dir / "manifest.pending.json", manifest, private=True)
        matched_manifest = {
            "schema": MATCHED_CORPUS_SCHEMA,
            "source_profile_sha256": profile["sha256"],
            "collection_packet_sha256": packet["sha256"],
            "admission_selection_sha256": selection["sha256"],
            "admission_receipt_sha256": receipt["sha256"],
            "matched_plan_sha256": plan["sha256"],
            "coverage_sha256": coverage["sha256"],
            "arms": {
                arm: {
                    "manifest_path": f"{arm}/manifest.pending.json",
                    "manifest_file_sha256": file_sha256(output / arm / "manifest.pending.json"),
                    "manifest_sha256": manifests[arm]["sha256"],
                    "train_file_sha256": manifests[arm]["files"]["train"]["sha256"],
                }
                for arm in ARMS
            },
            "training_ready": coverage["training_ready"],
        }
        matched_manifest["sha256"] = digest_json(matched_manifest)
        atomic_write_json(output / "selection.private.json", selection_rows, private=True)
        atomic_write_json(output / "coverage.private.json", coverage, private=True)
        atomic_write_json(output / "matched-manifest.private.json", matched_manifest, private=True)
        materialization = {
            "schema": RECEIPT_SCHEMA,
            "matched_manifest_file_sha256": file_sha256(output / "matched-manifest.private.json"),
            "matched_manifest_sha256": matched_manifest["sha256"],
            "coverage_file_sha256": file_sha256(output / "coverage.private.json"),
            "coverage_sha256": coverage["sha256"],
            "selection_file_sha256": file_sha256(output / "selection.private.json"),
            "training_ready": coverage["training_ready"],
            "external_mutations": 0,
            "submitted": False,
        }
        materialization["sha256"] = digest_json(materialization)
        atomic_write_json(output / "MATERIALIZATION.json", materialization, private=True)
    except BaseException:
        # A partial create-once directory is evidence and must not be replayed over.
        raise
    teacher._require_unchanged(sources, "teacher matched materialization input")
    return {
        "submitted": False,
        "external_mutations": 0,
        "matched_manifest_sha256": matched_manifest["sha256"],
        "coverage_sha256": coverage["sha256"],
        "arms": {arm: gates[arm] for arm in ARMS},
        "training_ready": coverage["training_ready"],
    }


def _coverage(value: Mapping[str, Any]) -> dict[str, Any]:
    coverage = _sealed(value, COVERAGE_SCHEMA, "teacher matched coverage")
    _exact(
        coverage,
        {
            "schema",
            "source_profile_sha256",
            "collection_packet_sha256",
            "admission_selection_sha256",
            "admission_receipt_sha256",
            "matched_plan_sha256",
            "selected_source_records",
            "packed_windows",
            "same_packed_window_selection",
            "same_input_ids_and_order",
            "only_loss_mask_differs",
            "packing_independent_target_occurrence_deduplication",
            "arms",
            "heldout_families_materialized",
            "external_benchmarks_materialized",
            "provider_private_or_hidden_reasoning_materialized",
            "raw_text_written",
            "training_ready",
            "sha256",
        },
        "teacher matched coverage",
    )
    if (
        coverage.get("heldout_families_materialized") != 0
        or coverage.get("external_benchmarks_materialized") != []
        or coverage.get("provider_private_or_hidden_reasoning_materialized") is not False
        or coverage.get("raw_text_written") is not False
        or coverage.get("same_packed_window_selection") is not True
        or coverage.get("same_input_ids_and_order") is not True
        or coverage.get("only_loss_mask_differs") is not True
        or coverage.get("packing_independent_target_occurrence_deduplication") is not True
    ):
        raise ValueError("teacher matched coverage violates the reviewed private boundary")
    arms = _exact(coverage.get("arms"), set(ARMS), "teacher matched coverage arms")
    for arm in ARMS:
        gate = _exact(
            arms[arm],
            {
                "packing_independent_unique_target_tokens",
                "minimum_unique_target_tokens",
                "families_with_targets",
                "minimum_selected_families",
                "selected_family_fraction",
                "minimum_selected_family_fraction",
                "largest_family_target_token_fraction",
                "maximum_family_target_token_fraction",
                "family_concentration_within_limit",
                "qualified",
            },
            f"teacher matched {arm} coverage gate",
        )
        if (
            gate.get("packing_independent_unique_target_tokens")
            < gate.get("minimum_unique_target_tokens")
            or gate.get("minimum_unique_target_tokens") < teacher.MINIMUM_UNIQUE_SUPERVISED_TOKENS
            or gate.get("families_with_targets") < gate.get("minimum_selected_families")
            or gate.get("selected_family_fraction") < gate.get("minimum_selected_family_fraction")
            or gate.get("largest_family_target_token_fraction")
            > gate.get("maximum_family_target_token_fraction")
            or gate.get("family_concentration_within_limit") is not True
            or gate.get("qualified") is not True
        ):
            raise ValueError(f"teacher matched {arm} arm has not reached every corpus gate")
    if coverage.get("training_ready") is not True:
        raise ValueError("teacher matched coverage is not training-ready")
    return coverage


def _matched_manifest(value: Mapping[str, Any], coverage: dict[str, Any]) -> dict[str, Any]:
    manifest = _sealed(value, MATCHED_CORPUS_SCHEMA, "teacher matched corpus manifest")
    _exact(
        manifest,
        {
            "schema",
            "source_profile_sha256",
            "collection_packet_sha256",
            "admission_selection_sha256",
            "admission_receipt_sha256",
            "matched_plan_sha256",
            "coverage_sha256",
            "arms",
            "training_ready",
            "sha256",
        },
        "teacher matched corpus manifest",
    )
    if (
        manifest.get("coverage_sha256") != coverage["sha256"]
        or manifest.get("source_profile_sha256") != coverage["source_profile_sha256"]
        or manifest.get("collection_packet_sha256") != coverage["collection_packet_sha256"]
        or manifest.get("admission_selection_sha256") != coverage["admission_selection_sha256"]
        or manifest.get("admission_receipt_sha256") != coverage["admission_receipt_sha256"]
        or manifest.get("matched_plan_sha256") != coverage["matched_plan_sha256"]
        or manifest.get("training_ready") is not True
        or set(_mapping(manifest.get("arms"), "matched manifest arms")) != set(ARMS)
    ):
        raise ValueError("teacher matched corpus manifest differs from coverage")
    for arm in ARMS:
        binding = _exact(
            manifest["arms"][arm],
            {
                "manifest_path",
                "manifest_file_sha256",
                "manifest_sha256",
                "train_file_sha256",
            },
            f"teacher matched {arm} binding",
        )
        for name in binding:
            if name.endswith("sha256"):
                _sha(binding[name], f"teacher matched {arm} {name}")
    return manifest


def _authorize_aggregate(
    matched_manifest: Mapping[str, Any],
    coverage: Mapping[str, Any],
    materialization: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate aggregate evidence; only ``authorize_paths`` may issue a permit."""

    checked_coverage = _coverage(coverage)
    checked_manifest = _matched_manifest(matched_manifest, checked_coverage)
    receipt = _sealed(materialization, RECEIPT_SCHEMA, "teacher materialization receipt")
    _exact(
        receipt,
        {
            "schema",
            "matched_manifest_file_sha256",
            "matched_manifest_sha256",
            "coverage_file_sha256",
            "coverage_sha256",
            "selection_file_sha256",
            "training_ready",
            "external_mutations",
            "submitted",
            "sha256",
        },
        "teacher materialization receipt",
    )
    if (
        receipt.get("matched_manifest_sha256") != checked_manifest["sha256"]
        or receipt.get("coverage_sha256") != checked_coverage["sha256"]
        or receipt.get("training_ready") is not True
        or receipt.get("external_mutations") != 0
        or receipt.get("submitted") is not False
    ):
        raise ValueError("teacher materialization receipt does not authorize this corpus")
    permit = {
        "schema": PERMIT_SCHEMA,
        "matched_manifest_sha256": checked_manifest["sha256"],
        "coverage_sha256": checked_coverage["sha256"],
        "materialization_receipt_sha256": receipt["sha256"],
        "source_profile_sha256": checked_manifest["source_profile_sha256"],
        "collection_packet_sha256": checked_manifest["collection_packet_sha256"],
        "admission_selection_sha256": checked_manifest["admission_selection_sha256"],
        "matched_plan_sha256": checked_manifest["matched_plan_sha256"],
        "arms": {
            arm: {
                "manifest_sha256": checked_manifest["arms"][arm]["manifest_sha256"],
                "manifest_file_sha256": checked_manifest["arms"][arm]["manifest_file_sha256"],
                "train_file_sha256": checked_manifest["arms"][arm]["train_file_sha256"],
                "packing_independent_unique_target_tokens": checked_coverage["arms"][arm][
                    "packing_independent_unique_target_tokens"
                ],
            }
            for arm in ARMS
        },
        "authorized_use": "qwen38_matched_teacher_visible_rationale_sft",
        "external_submission_authorized": False,
        "submitted": False,
    }
    permit["sha256"] = digest_json(permit)
    return permit


def authorize_paths(
    matched_manifest_path: Path,
    coverage_path: Path,
    materialization_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError("teacher training permit destination already exists")
    for path, label in (
        (matched_manifest_path, "teacher matched manifest"),
        (coverage_path, "teacher matched coverage"),
        (materialization_path, "teacher materialization receipt"),
    ):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"{label} must be a regular file")
    matched = _json(matched_manifest_path, "teacher matched manifest")
    coverage_value = _json(coverage_path, "teacher matched coverage")
    receipt = _json(materialization_path, "teacher materialization receipt")
    permit = _authorize_aggregate(matched, coverage_value, receipt)
    if receipt["matched_manifest_file_sha256"] != file_sha256(matched_manifest_path) or receipt[
        "coverage_file_sha256"
    ] != file_sha256(coverage_path):
        raise ValueError("teacher materialization receipt does not bind the aggregate files")

    import pyarrow.parquet as pq

    base = matched_manifest_path.resolve().parent
    arm_rows: dict[str, list[dict[str, Any]]] = {}
    for arm in ARMS:
        binding = matched["arms"][arm]
        relative = Path(binding["manifest_path"])
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise ValueError("teacher arm manifest path escapes its matched corpus")
        manifest_path = base / relative
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise ValueError("teacher arm manifest is not a regular bound file")
        if file_sha256(manifest_path) != binding["manifest_file_sha256"]:
            raise ValueError("teacher arm manifest file digest mismatch")
        pending = _sealed(
            _json(manifest_path, f"teacher {arm} pending manifest"),
            PENDING_MANIFEST_SCHEMA,
            f"teacher {arm} pending manifest",
        )
        if (
            pending["sha256"] != binding["manifest_sha256"]
            or pending.get("objective") != arm
            or pending.get("training_permit_sha256") is not None
            or pending.get("validation_mode") != PENDING_VALIDATION_MODE
            or set(pending.get("files", {})) != {"train"}
        ):
            raise ValueError("teacher arm manifest differs from its matched binding")
        train = _mapping(pending["files"]["train"], f"teacher {arm} train file")
        train_relative = Path(_text(train.get("path"), f"teacher {arm} train path"))
        if train_relative.is_absolute() or not train_relative.parts or ".." in train_relative.parts:
            raise ValueError("teacher train path escapes its permitted arm")
        train_path = manifest_path.parent / train_relative
        if train_path.is_symlink() or not train_path.is_file():
            raise ValueError("teacher train Parquet is not a regular bound file")
        if (
            file_sha256(train_path) != binding["train_file_sha256"]
            or train.get("sha256") != binding["train_file_sha256"]
            or train.get("format") != DENSE_FORMAT
        ):
            raise ValueError("teacher train Parquet digest or format mismatch")
        rows = pq.read_table(train_path).to_pylist()
        if len(rows) != train.get("rows"):
            raise ValueError("teacher train Parquet row count mismatch")
        supervised = 0
        family_tokens: dict[str, int] = defaultdict(int)
        for row in rows:
            ids = _token_ids(row.get("input_ids"), "teacher permitted input IDs")
            mask = row.get("loss_mask")
            family = _text(row.get("source_group_id"), "teacher permitted family")
            if (
                not isinstance(mask, list)
                or len(mask) != len(ids)
                or any(type(item) is not int or item not in {0, 1} for item in mask)
            ):
                raise ValueError("teacher permitted loss mask is invalid")
            selected_tokens = sum(mask)
            supervised += selected_tokens
            family_tokens[family] += selected_tokens
        gate = coverage_value["arms"][arm]
        if (
            supervised != train.get("supervised_tokens")
            or supervised != gate["packing_independent_unique_target_tokens"]
            or len(family_tokens) != gate["families_with_targets"]
            or (max(family_tokens.values()) / supervised if supervised else 0.0)
            != gate["largest_family_target_token_fraction"]
        ):
            raise ValueError("teacher arm aggregate evidence differs from its Parquet")
        arm_rows[arm] = rows

    rationale_rows = arm_rows[RATIONALE_ARM]
    action_rows = arm_rows[ACTION_ARM]
    if len(rationale_rows) != len(action_rows):
        raise ValueError("teacher matched arms do not contain the same packed windows")
    for rationale, action in zip(rationale_rows, action_rows, strict=True):
        rationale_common = {key: value for key, value in rationale.items() if key != "loss_mask"}
        action_common = {key: value for key, value in action.items() if key != "loss_mask"}
        if rationale_common != action_common or any(
            action_bit > rationale_bit
            for rationale_bit, action_bit in zip(
                rationale["loss_mask"], action["loss_mask"], strict=True
            )
        ):
            raise ValueError("teacher matched arms differ by more than rationale loss")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(permit, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return permit


def consume(
    permit: Mapping[str, Any],
    pending_manifest: Mapping[str, Any],
    *,
    arm: str,
) -> dict[str, Any]:
    """Turn one permitted arm into the manifest consumed by ``compile_sft``."""

    if arm not in ARMS:
        raise ValueError("unknown teacher matched training arm")
    checked_permit = _sealed(permit, PERMIT_SCHEMA, "teacher training permit")
    _exact(
        checked_permit,
        {
            "schema",
            "matched_manifest_sha256",
            "coverage_sha256",
            "materialization_receipt_sha256",
            "source_profile_sha256",
            "collection_packet_sha256",
            "admission_selection_sha256",
            "matched_plan_sha256",
            "arms",
            "authorized_use",
            "external_submission_authorized",
            "submitted",
            "sha256",
        },
        "teacher training permit",
    )
    if (
        checked_permit.get("authorized_use") != "qwen38_matched_teacher_visible_rationale_sft"
        or checked_permit.get("external_submission_authorized") is not False
        or checked_permit.get("submitted") is not False
    ):
        raise ValueError("teacher training permit changes its source-only authority")
    manifest = _sealed(pending_manifest, PENDING_MANIFEST_SCHEMA, "teacher pending SFT manifest")
    arms = _exact(checked_permit.get("arms"), set(ARMS), "teacher permit arms")
    arm_binding = _mapping(arms.get(arm), f"teacher permit {arm} arm")
    _exact(
        arm_binding,
        {
            "manifest_sha256",
            "manifest_file_sha256",
            "train_file_sha256",
            "packing_independent_unique_target_tokens",
        },
        f"teacher permit {arm} arm",
    )
    if arm_binding.get("manifest_sha256") != manifest["sha256"]:
        raise ValueError("teacher training permit does not bind the selected arm manifest")
    if (
        manifest.get("objective") != arm
        or manifest.get("training_permit_sha256") is not None
        or manifest.get("validation_mode") != PENDING_VALIDATION_MODE
        or set(manifest.get("files", {})) != {"train"}
        or manifest["files"]["train"].get("format") != DENSE_FORMAT
        or manifest["files"]["train"].get("sha256") != arm_binding.get("train_file_sha256")
        or manifest["files"]["train"].get("supervised_tokens")
        != arm_binding.get("packing_independent_unique_target_tokens")
    ):
        raise ValueError("teacher pending manifest differs from the permitted training arm")
    result = copy.deepcopy(manifest)
    result["schema"] = TRAINING_MANIFEST_SCHEMA
    result["validation_mode"] = "task_outcomes_only"
    result["training_permit_sha256"] = checked_permit["sha256"]
    result["sha256"] = digest_json({key: item for key, item in result.items() if key != "sha256"})
    return result


def consume_paths(permit_path: Path, pending_manifest_path: Path, arm: str, output: Path) -> dict:
    if output.exists() or output.is_symlink():
        raise FileExistsError("teacher training manifest destination already exists")
    if output.parent.resolve() != pending_manifest_path.parent.resolve():
        raise ValueError("teacher training manifest must remain beside its bound train Parquet")
    if (
        permit_path.is_symlink()
        or not permit_path.is_file()
        or pending_manifest_path.is_symlink()
        or not pending_manifest_path.is_file()
    ):
        raise ValueError("teacher permit and pending manifest must be regular files")
    permit = _json(permit_path, "teacher training permit")
    pending = _json(pending_manifest_path, "teacher pending SFT manifest")
    result = consume(permit, pending, arm=arm)
    binding = permit["arms"][arm]
    if file_sha256(pending_manifest_path) != binding["manifest_file_sha256"]:
        raise ValueError("teacher pending manifest file digest differs from the permit")
    train_relative = Path(_text(pending["files"]["train"]["path"], "teacher train path"))
    if train_relative.is_absolute() or not train_relative.parts or ".." in train_relative.parts:
        raise ValueError("teacher train path escapes its permitted arm")
    train_path = pending_manifest_path.parent / train_relative
    if (
        train_path.is_symlink()
        or not train_path.is_file()
        or file_sha256(train_path) != binding["train_file_sha256"]
    ):
        raise ValueError("teacher train Parquet differs from the permit")
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return result
