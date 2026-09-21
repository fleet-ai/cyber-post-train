"""Private Qwen/OpenCode materialization for explicitly visible reasoning.

This module is intentionally *not* an option on ``fleet_collection_corpus``.
The action-only corpus rejects all assistant prose because it cannot establish
whether that prose is visible reasoning or private provider state.  This
separate contract accepts only a narrowly described source profile, a sealed
train-family selection, and records whose token boundaries are re-rendered by
an independently pinned adapter.

It is an offline CPU-only builder.  It does not contact Fleet, launch a
workload, print records, or write source text to its output.  The only corpus
payload is private token IDs and loss masks.  A caller must supply the exact
Qwen/OpenCode serialization adapter named by the source profile; this module
will fail closed rather than guess how a provider represented reasoning.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from . import collection_campaign
from . import fleet_collection_admission as admission
from .io import atomic_write_json, atomic_write_jsonl, digest_json, file_sha256, iter_jsonl
from .sft import _known
from .task_family_split import (
    ANCHORED_SCHEMA,
    TRUSTED_FLEET_COLLECTION_ROOT_ID,
    require_trusted_fleet_collection_root_anchor,
)
from .task_family_split import validate as validate_split

SOURCE_PROFILE_SCHEMA = "cyber_qwen_opencode_student_visible_reasoning_source_profile_v1"
PACKET_SCHEMA = "cyber_qwen_opencode_student_visible_reasoning_packet_v1"
SELECTION_SCHEMA = "cyber_qwen_opencode_student_visible_reasoning_selection_v1"
REQUEST_SCHEMA = "cyber_qwen_opencode_visible_reasoning_materialization_request_v1"
RECORD_SCHEMA = "cyber_qwen_opencode_student_visible_reasoning_record_v1"
CORPUS_SCHEMA = "cyber_qwen_opencode_visible_reasoning_sft_corpus_v1"
COVERAGE_SCHEMA = "cyber_qwen_opencode_visible_reasoning_coverage_v1"
RECEIPT_SCHEMA = "cyber_qwen_opencode_visible_reasoning_materialization_receipt_v1"
ROUNDTRIP_SCHEMA = "cyber_qwen_opencode_template_roundtrip_v1"

QWEN_REPOSITORY = "Qwen/Qwen3.8-27B"
OPENCODE_HARNESS = "opencode"
OPENCODE_VERSION = "1.18.27"
ONLINE_COMPACTION = "opencode_1.18.27_native_compaction_autocontinue_v2"
EXACT_COMPACTION = "student_generated_exact_continuation_v1"
MINIMUM_SUPERVISED_TOKENS = 20_000_000
MAXIMUM_FAMILY_TOKEN_FRACTION = 0.25

_SHA256 = re.compile(r"sha256:[0-9a-f]{64}")
_OPAQUE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}")
_PRIVATE_FIELDS = frozenset(
    {
        "thinking",
        "reasoning",
        "reasoning_content",
        "analysis",
        "hidden_reasoning",
        "private_reasoning",
    }
)


def _sealed(value: Mapping[str, Any], schema: str, label: str) -> dict[str, Any]:
    result = _mapping(value, label)
    if result.get("schema") != schema or result.get("sha256") != digest_json(
        {key: item for key, item in result.items() if key != "sha256"}
    ):
        raise ValueError(f"invalid sealed {label}")
    return result


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return dict(value)


def _exact(value: object, expected: set[str], label: str) -> dict[str, Any]:
    result = _mapping(value, label)
    if set(result) != expected:
        raise ValueError(f"{label} has unknown or missing fields")
    return result


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"exact {label} digest is required")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"nonempty {label} is required")
    return value


def _count(value: object, label: str, *, positive: bool = False) -> int:
    if type(value) is not int or value < 0 or (positive and value == 0):
        raise ValueError(f"{label} must be a {'positive' if positive else 'nonnegative'} integer")
    return value


def _token_ids(value: object, label: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a nonempty token-id list")
    if any(type(token) is not int or token < 0 for token in value):
        raise ValueError(f"{label} must contain nonnegative integer token IDs")
    return list(value)


def _path(root: Path, value: object, label: str) -> Path:
    path = Path(_string(value, label))
    return path if path.is_absolute() else root / path


def _input(root: Path, value: object, label: str) -> tuple[Path, str]:
    reference = _exact(value, {"path", "sha256"}, label)
    path = _path(root, reference["path"], f"{label} path")
    expected = _sha(reference["sha256"], f"{label} file")
    if path.is_symlink() or not path.is_file() or file_sha256(path) != expected:
        raise ValueError(f"{label} file digest mismatch")
    return path, expected


def _private_records_input(root: Path, value: object) -> tuple[Path, str]:
    """Bind a private record path without opening it before public gates pass."""
    reference = _exact(value, {"path", "sha256"}, "records")
    path = _path(root, reference["path"], "records path")
    expected = _sha(reference["sha256"], "records file")
    if path.is_symlink() or not path.is_file():
        raise ValueError("records path is not a regular private file")
    return path, expected


def _json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label} JSON") from error
    return _mapping(value, label)


def _model(value: object, label: str) -> dict[str, str]:
    model = _exact(value, {"repository", "revision", "session_model"}, label)
    result = {field: _string(model[field], f"{label} {field}") for field in model}
    if not re.fullmatch(r"[0-9a-f]{40}", result["revision"]):
        raise ValueError(f"{label} revision must be an immutable 40-hex revision")
    return result


def _profile(value: Mapping[str, Any]) -> dict[str, Any]:
    profile = _sealed(value, SOURCE_PROFILE_SCHEMA, "visible-reasoning source profile")
    _exact(
        profile,
        {
            "schema",
            "source",
            "qwen_target",
            "opencode",
            "thinking",
            "serialization",
            "compaction",
            "sha256",
        },
        "visible-reasoning source profile",
    )
    source = _mapping(profile["source"], "visible-reasoning source")
    kind = source.get("kind")
    base = {
        "kind",
        "model_alias",
        "model",
        "source_authorization_receipt_sha256",
        "student_visible_reasoning_authorization_receipt_sha256",
    }
    if kind == "teacher_visible":
        base |= {
            "teacher_visibility_receipt_sha256",
            "teacher_training_authorization_receipt_sha256",
        }
    if kind not in {"qwen_self", "teacher_visible"} or set(source) != base:
        raise ValueError("source must use one explicit visible-reasoning authorization mode")
    _string(source["model_alias"], "source model alias")
    source_model = _model(source["model"], "source model")
    for name in base - {"kind", "model_alias", "model"}:
        _sha(source[name], f"source {name}")

    target = _exact(
        profile["qwen_target"],
        {"repository", "revision", "tokenizer_sha256", "chat_template_sha256"},
        "Qwen target",
    )
    if (
        target["repository"] != QWEN_REPOSITORY
        or not isinstance(target["revision"], str)
        or not re.fullmatch(r"[0-9a-f]{40}", target["revision"])
    ):
        raise ValueError("visible-reasoning target must pin Qwen3.8-27B exactly")
    for name in ("tokenizer_sha256", "chat_template_sha256"):
        _sha(target[name], f"Qwen target {name}")
    if kind == "qwen_self" and source_model != {
        "repository": target["repository"],
        "revision": target["revision"],
        "session_model": source_model["session_model"],
    }:
        raise ValueError("Qwen self reasoning must use the exact Qwen target revision")

    opencode = _exact(
        profile["opencode"],
        {
            "harness",
            "harness_version",
            "release_asset_sha256",
            "tool_catalog_sha256",
            "context_management",
            "context_window_tokens",
            "context_headroom_tokens",
            "tools",
        },
        "OpenCode treatment",
    )
    if (
        opencode["harness"] != OPENCODE_HARNESS
        or opencode["harness_version"] != OPENCODE_VERSION
        or opencode["context_management"] != ONLINE_COMPACTION
        or opencode["context_window_tokens"] != 262_144
        or opencode["context_headroom_tokens"] != 20_000
        or opencode["tools"] != ["bash", "submit_report"]
    ):
        raise ValueError("source must use the qualified Qwen/OpenCode treatment")
    for name in ("release_asset_sha256", "tool_catalog_sha256"):
        _sha(opencode[name], f"OpenCode {name}")

    thinking = _exact(profile["thinking"], {"enable_thinking", "preserve_thinking"}, "thinking")
    if thinking != {"enable_thinking": True, "preserve_thinking": True}:
        raise ValueError("source must explicitly preserve student-visible Qwen thinking")

    serialization = _exact(
        profile["serialization"],
        {
            "schema",
            "renderer_adapter_sha256",
            "roundtrip_fixture_sha256",
            "collection_template_sha256",
            "training_template_sha256",
            "serving_template_sha256",
            "round_trip_verified",
        },
        "Qwen/OpenCode serialization proof",
    )
    if (
        serialization["schema"] != ROUNDTRIP_SCHEMA
        or serialization["round_trip_verified"] is not True
        or any(
            serialization[name] != target["chat_template_sha256"]
            for name in (
                "collection_template_sha256",
                "training_template_sha256",
                "serving_template_sha256",
            )
        )
    ):
        raise ValueError("Qwen/OpenCode template round-trip is not exactly bound")
    for name in ("renderer_adapter_sha256", "roundtrip_fixture_sha256"):
        _sha(serialization[name], f"serialization {name}")

    compaction = _exact(
        profile["compaction"], {"accepted_kind", "opaque_compaction_rejected"}, "compaction"
    )
    if compaction != {
        "accepted_kind": EXACT_COMPACTION,
        "opaque_compaction_rejected": True,
    }:
        raise ValueError("source compaction policy is insufficient")
    return profile


def _roundtrip(value: Mapping[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    """Validate synthetic, non-task Qwen/OpenCode template equivalence evidence."""
    fixture = _sealed(value, ROUNDTRIP_SCHEMA, "Qwen/OpenCode round-trip fixture")
    _exact(
        fixture,
        {
            "schema",
            "qwen_target",
            "renderer_adapter_sha256",
            "cases",
            "sha256",
        },
        "Qwen/OpenCode round-trip fixture",
    )
    target = _exact(
        fixture["qwen_target"],
        {"repository", "revision", "tokenizer_sha256", "chat_template_sha256"},
        "round-trip Qwen target",
    )
    if target != profile["qwen_target"]:
        raise ValueError("round-trip fixture changes the exact Qwen target")
    if fixture["renderer_adapter_sha256"] != profile["serialization"]["renderer_adapter_sha256"]:
        raise ValueError("round-trip fixture changes the pinned serializer")
    cases = fixture["cases"]
    if not isinstance(cases, list) or not cases:
        raise ValueError("round-trip fixture requires synthetic cases")
    seen: set[str] = set()
    for case in cases:
        case = _exact(
            case,
            {
                "case_id",
                "collection_token_ids_sha256",
                "training_token_ids_sha256",
                "serving_token_ids_sha256",
                "assistant_start_token_index",
            },
            "round-trip case",
        )
        case_id = _string(case["case_id"], "round-trip case identity")
        if case_id in seen:
            raise ValueError("round-trip fixture duplicates a synthetic case")
        seen.add(case_id)
        for name in (
            "collection_token_ids_sha256",
            "training_token_ids_sha256",
            "serving_token_ids_sha256",
        ):
            _sha(case[name], f"round-trip {name}")
        if (
            case["collection_token_ids_sha256"] != case["training_token_ids_sha256"]
            or case["collection_token_ids_sha256"] != case["serving_token_ids_sha256"]
            or _count(case["assistant_start_token_index"], "assistant start token index") < 1
        ):
            raise ValueError("round-trip fixture does not prove exact token serialization")
    return fixture


def _packet(value: Mapping[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    packet = _sealed(value, PACKET_SCHEMA, "visible-reasoning collection packet")
    _exact(
        packet,
        {
            "schema",
            "source_profile_sha256",
            "catalog_inventory_sha256",
            "family_split_sha256",
            "root_role_anchor_id",
            "family_role_anchor_sha256",
            "protected_family_lock_sha256",
            "runtime_bindings_sha256",
            "training_data_eligible",
            "objective",
            "minimum_unique_supervised_tokens",
            "maximum_family_target_token_fraction",
            "deduplication_order",
            "rejection_policy",
            "sha256",
        },
        "visible-reasoning collection packet",
    )
    for name in (
        "source_profile_sha256",
        "catalog_inventory_sha256",
        "family_split_sha256",
        "family_role_anchor_sha256",
        "protected_family_lock_sha256",
        "runtime_bindings_sha256",
    ):
        _sha(packet[name], f"collection packet {name}")
    if (
        packet["source_profile_sha256"] != profile["sha256"]
        or packet["root_role_anchor_id"] != TRUSTED_FLEET_COLLECTION_ROOT_ID
        or packet["training_data_eligible"] is not True
        or packet["objective"] != "student_visible_reasoning_plus_visible_actions"
        or packet["minimum_unique_supervised_tokens"] < MINIMUM_SUPERVISED_TOKENS
        or packet["maximum_family_target_token_fraction"] != MAXIMUM_FAMILY_TOKEN_FRACTION
        or packet["deduplication_order"]
        != [
            "source_session_identity",
            "normalized_trajectory_digest",
            "packed_window_payload_digest",
        ]
        or packet["rejection_policy"]
        != {
            "private_or_unknown_reasoning": "reject",
            "opaque_compaction": "reject",
            "unknown_serialization": "reject",
            "heldout_family": "reject",
        }
    ):
        raise ValueError("visible-reasoning collection packet has an insufficient policy")
    return packet


def _task_boundary(
    inventory: dict[str, Any],
    split: dict[str, Any],
    role_anchor: dict[str, Any],
    lock: dict[str, Any],
    runtime: dict[str, Any],
    packet: dict[str, Any],
) -> dict[tuple[str, str], dict[str, Any]]:
    rows = collection_campaign._inventory_rows(inventory)
    bindings = collection_campaign._runtime_bindings(runtime, inventory, rows)
    if split.get("schema") != ANCHORED_SCHEMA or split.get("inventory_sha256") != inventory.get(
        "sha256"
    ):
        raise ValueError("family split is not an anchored exact catalog split")
    require_trusted_fleet_collection_root_anchor(role_anchor)
    validate_split(split, rows, role_anchor=role_anchor)
    if (
        packet["catalog_inventory_sha256"] != inventory["sha256"]
        or packet["family_split_sha256"] != split["sha256"]
        or packet["family_role_anchor_sha256"] != role_anchor["sha256"]
        or packet["protected_family_lock_sha256"] != lock["sha256"]
        or packet["runtime_bindings_sha256"] != runtime["sha256"]
    ):
        raise ValueError("collection packet does not bind the reviewed task boundary")
    protected = _sealed(lock, admission.PROTECTED_FAMILY_LOCK_SCHEMA, "protected-family lock")
    if protected.get("source_split_sha256") != split["sha256"]:
        raise ValueError("protected-family lock does not bind the family split")
    protected_groups = protected.get("heldout_group_ids")
    if not isinstance(protected_groups, list) or protected_groups != sorted(set(protected_groups)):
        raise ValueError("protected-family lock is malformed")
    role_rows = {(row["task_key"], row["task_version_id"]): row for row in split["tasks"]}
    expected_heldout = {row["group_id"] for row in role_rows.values() if row["split"] != "train"}
    if set(protected_groups) != expected_heldout:
        raise ValueError("protected-family lock must cover every held-out family")
    return bindings


def _selection(
    value: Mapping[str, Any],
    packet: dict[str, Any],
    split: dict[str, Any],
    lock: dict[str, Any],
    bindings: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    selection = _sealed(value, SELECTION_SCHEMA, "visible-reasoning private selection")
    _exact(
        selection,
        {
            "schema",
            "collection_packet_sha256",
            "verified_success_evidence_sha256",
            "catalog_inventory_sha256",
            "family_split_sha256",
            "root_role_anchor_id",
            "family_role_anchor_sha256",
            "protected_family_lock_sha256",
            "max_sessions_per_task_version",
            "selected",
            "sha256",
        },
        "visible-reasoning private selection",
    )
    for name in (
        "collection_packet_sha256",
        "verified_success_evidence_sha256",
        "catalog_inventory_sha256",
        "family_split_sha256",
        "family_role_anchor_sha256",
        "protected_family_lock_sha256",
    ):
        _sha(selection[name], f"selection {name}")
    if (
        selection["collection_packet_sha256"] != packet["sha256"]
        or selection["catalog_inventory_sha256"] != packet["catalog_inventory_sha256"]
        or selection["family_split_sha256"] != split["sha256"]
        or selection["root_role_anchor_id"] != TRUSTED_FLEET_COLLECTION_ROOT_ID
        or selection["family_role_anchor_sha256"] != packet["family_role_anchor_sha256"]
        or selection["protected_family_lock_sha256"] != lock["sha256"]
        or type(selection["max_sessions_per_task_version"]) is not int
        or selection["max_sessions_per_task_version"] < 1
    ):
        raise ValueError("private selection changes an immutable collection binding")
    protected = set(lock["heldout_group_ids"])
    roles = {(row["task_key"], row["task_version_id"]): row for row in split["tasks"]}
    fields = {
        "record_id",
        "source_session_identity_sha256",
        "normalized_record_sha256",
        "normalized_trajectory_sha256",
        "transcript_sha256",
        "task_key",
        "task_version_id",
        "group_id",
        "attempt",
        "reasoning_visibility",
        "compaction_kind",
    }
    rows = selection["selected"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("private selection must contain visible-reasoning records")
    result: dict[str, dict[str, Any]] = {}
    sessions: set[str] = set()
    trajectories: set[str] = set()
    per_task: dict[tuple[str, str], int] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != fields:
            raise ValueError("private selection row has an unsupported contract")
        record_id = _string(row["record_id"], "selected record identity")
        if record_id in result:
            raise ValueError("private selection duplicates a record identity")
        for name in (
            "source_session_identity_sha256",
            "normalized_record_sha256",
            "normalized_trajectory_sha256",
            "transcript_sha256",
        ):
            _sha(row[name], f"selected {name}")
        if row["source_session_identity_sha256"] in sessions:
            raise ValueError("private selection duplicates a source session")
        if row["normalized_trajectory_sha256"] in trajectories:
            raise ValueError("private selection duplicates a trajectory")
        sessions.add(row["source_session_identity_sha256"])
        trajectories.add(row["normalized_trajectory_sha256"])
        identity = (
            _string(row["task_key"], "selected task key"),
            _string(row["task_version_id"], "selected task version"),
        )
        role = roles.get(identity)
        if (
            identity not in bindings
            or role is None
            or role["split"] != "train"
            or row["group_id"] != role["group_id"]
            or row["group_id"] in protected
        ):
            raise ValueError("held-out or unbound family reached visible-reasoning selection")
        if type(row["attempt"]) is not int or row["attempt"] < 1:
            raise ValueError("selected attempt is invalid")
        if row["reasoning_visibility"] != "student_visible":
            raise ValueError("private or unknown reasoning cannot enter selection")
        if row["compaction_kind"] not in {"none", EXACT_COMPACTION}:
            raise ValueError("opaque compaction cannot enter selection")
        per_task[identity] = per_task.get(identity, 0) + 1
        if per_task[identity] > selection["max_sessions_per_task_version"]:
            raise ValueError("selection exceeds its deterministic per-task-version cap")
        result[record_id] = row
    return result


def _message(value: object) -> dict[str, Any]:
    message = _mapping(value, "visible-reasoning message")
    if _PRIVATE_FIELDS & set(message):
        raise ValueError("private or unknown reasoning fields are forbidden")
    role = message.get("role")
    if role not in {"system", "user", "assistant", "tool"}:
        raise ValueError("visible-reasoning message role is invalid")
    if not isinstance(message.get("content"), str):
        raise ValueError("visible-reasoning message content must be explicit text")
    if role in {"system", "user"}:
        if set(message) != {"role", "content"}:
            raise ValueError("system and user messages have an unsupported field")
    elif role == "assistant":
        if set(message) not in ({"role", "content"}, {"role", "content", "tool_calls"}):
            raise ValueError("assistant message has an unsupported field")
        calls = message.get("tool_calls")
        if calls is not None and not isinstance(calls, list):
            raise ValueError("assistant tool calls must be a list")
        for call in calls or []:
            call = _exact(call, {"id", "type", "function"}, "tool call")
            if not isinstance(call["id"], str) or _OPAQUE_ID.fullmatch(call["id"]) is None:
                raise ValueError("tool call identity is invalid")
            if call["type"] != "function":
                raise ValueError("tool call type is invalid")
            function = _exact(call["function"], {"name", "arguments"}, "tool function")
            if function["name"] not in {"bash", "submit_report"}:
                raise ValueError("visible-reasoning tool surface is not approved")
    else:
        if set(message) != {"role", "content", "tool_call_id"}:
            raise ValueError("tool result has an unsupported field")
        if (
            not isinstance(message.get("tool_call_id"), str)
            or _OPAQUE_ID.fullmatch(message["tool_call_id"]) is None
        ):
            raise ValueError("tool result must bind an approved opaque tool call")
    return message


def _span(value: object, ids: list[int]) -> dict[str, Any]:
    span = _exact(value, {"kind", "token_start", "token_end", "token_ids_sha256"}, "target span")
    if span["kind"] not in {"student_visible_reasoning", "visible_action"}:
        raise ValueError("target span has an unknown reasoning visibility")
    start = _count(span["token_start"], "target span start")
    end = _count(span["token_end"], "target span end", positive=True)
    if start >= end or end > len(ids):
        raise ValueError("target span is outside rendered token IDs")
    if span["token_ids_sha256"] != digest_json(ids[start:end]):
        raise ValueError("target span token identity differs from its rendered tokens")
    return span


def _window(value: object) -> dict[str, Any]:
    window = _exact(
        value,
        {
            "window_id",
            "assistant_turn_id",
            "input_ids",
            "prompt_token_count",
            "prompt_token_sha256",
            "serialized_token_ids_sha256",
            "target_spans",
            "window_payload_sha256",
        },
        "visible-reasoning window",
    )
    _string(window["window_id"], "window identity")
    _string(window["assistant_turn_id"], "assistant turn identity")
    ids = _token_ids(window["input_ids"], "window input IDs")
    prompt = _count(window["prompt_token_count"], "window prompt token count")
    if prompt >= len(ids):
        raise ValueError("window must contain a nonempty assistant continuation")
    if window["prompt_token_sha256"] != digest_json(ids[:prompt]):
        raise ValueError("window prompt digest differs from prompt token IDs")
    if window["serialized_token_ids_sha256"] != digest_json(ids):
        raise ValueError("window serialization digest differs from token IDs")
    spans_raw = window["target_spans"]
    if not isinstance(spans_raw, list) or not spans_raw:
        raise ValueError("window must have explicit supervised target spans")
    spans = [_span(span, ids) for span in spans_raw]
    ordered = sorted(spans, key=lambda span: span["token_start"])
    if spans != ordered or any(
        prior["token_end"] > current["token_start"]
        for prior, current in zip(spans[:-1], spans[1:], strict=True)
    ):
        raise ValueError("target spans must be ordered and non-overlapping")
    if any(span["token_start"] < prompt for span in spans):
        raise ValueError("supervised spans must start after the exact assistant boundary")
    if {span["kind"] for span in spans} != {"student_visible_reasoning", "visible_action"}:
        raise ValueError(
            "each reasoning window must prove both visible reasoning and action targets"
        )
    payload = {
        "input_ids": ids,
        "prompt_token_count": prompt,
        "target_spans": spans,
    }
    if window["window_payload_sha256"] != digest_json(payload):
        raise ValueError("window payload digest mismatch")
    return window


def _compaction(
    value: object,
    windows: dict[str, dict[str, Any]],
    messages: list[dict[str, Any]],
    source_kind: str,
) -> dict[str, Any]:
    compaction = _mapping(value, "record compaction")
    kind = compaction.get("kind")
    if kind == "none" and set(compaction) == {"kind"}:
        return compaction
    expected = {"kind", "boundaries"}
    if kind != EXACT_COMPACTION or set(compaction) != expected or source_kind != "qwen_self":
        raise ValueError("opaque or unapproved compaction is forbidden")
    boundaries = compaction["boundaries"]
    if not isinstance(boundaries, list) or not boundaries:
        raise ValueError("exact compaction requires continuation boundaries")
    fields = {
        "boundary_id",
        "parent_window_sha256",
        "original_task_digest",
        "prior_history_digest",
        "summary_message_digest",
        "continuation_token_sha256",
        "continuation_token_ids",
        "continuation_tokens",
        "pre_compaction_prompt_token_sha256",
        "pre_compaction_prompt_tokens",
        "post_compaction_prompt_token_sha256",
        "post_compaction_prompt_tokens",
        "next_target_window_id",
        "next_target_prompt_token_sha256",
        "summary_supervised",
    }
    seen: set[str] = set()
    window_payloads = {window["window_payload_sha256"] for window in windows.values()}
    message_digests = {digest_json(message) for message in messages}
    for boundary in boundaries:
        boundary = _exact(boundary, fields, "compaction boundary")
        boundary_id = _string(boundary["boundary_id"], "compaction boundary identity")
        if boundary_id in seen:
            raise ValueError("compaction boundary identity is duplicated")
        seen.add(boundary_id)
        for name in fields - {
            "boundary_id",
            "continuation_token_ids",
            "continuation_tokens",
            "pre_compaction_prompt_tokens",
            "post_compaction_prompt_tokens",
            "next_target_window_id",
            "summary_supervised",
        }:
            _sha(boundary[name], f"compaction {name}")
        for name in (
            "continuation_tokens",
            "pre_compaction_prompt_tokens",
            "post_compaction_prompt_tokens",
        ):
            _count(boundary[name], f"compaction {name}", positive=True)
        continuation = _token_ids(boundary["continuation_token_ids"], "compaction continuation")
        if (
            len(continuation) != boundary["continuation_tokens"]
            or digest_json(continuation) != boundary["continuation_token_sha256"]
        ):
            raise ValueError("compaction continuation tokens do not match their exact proof")
        if (
            boundary["parent_window_sha256"] not in window_payloads
            or boundary["summary_message_digest"] not in message_digests
        ):
            raise ValueError("compaction lineage is not bound to the exact private record")
        if boundary["summary_supervised"] is not False:
            raise ValueError("compaction summaries are context only, never reasoning targets")
        target = windows.get(boundary["next_target_window_id"])
        if target is None:
            raise ValueError("compaction must name an existing next target window")
        if (
            boundary["post_compaction_prompt_token_sha256"]
            != boundary["next_target_prompt_token_sha256"]
            or boundary["next_target_prompt_token_sha256"] != target["prompt_token_sha256"]
            or boundary["post_compaction_prompt_tokens"] != target["prompt_token_count"]
        ):
            raise ValueError("compaction continuation does not bind the true next target prompt")
    return compaction


def _record(value: Mapping[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    record = _exact(
        value,
        {
            "schema",
            "record_id",
            "source_profile_sha256",
            "lineage",
            "evidence",
            "reasoning_visibility",
            "messages",
            "windows",
            "compaction",
            "content_digest",
        },
        "private visible-reasoning record",
    )
    if record["schema"] != RECORD_SCHEMA:
        raise ValueError("private record uses an unsupported visible-reasoning schema")
    if record["content_digest"] != digest_json(
        {key: item for key, item in record.items() if key != "content_digest"}
    ):
        raise ValueError("private visible-reasoning record digest mismatch")
    if record["source_profile_sha256"] != profile["sha256"]:
        raise ValueError("private record changes the authorized source profile")
    _string(record["record_id"], "private record identity")
    lineage = _exact(
        record["lineage"], {"task_key", "task_version_id", "group_id"}, "record lineage"
    )
    for name in lineage:
        _string(lineage[name], f"record lineage {name}")
    evidence = _exact(
        record["evidence"],
        {
            "source_session_identity_sha256",
            "normalized_trajectory_sha256",
            "transcript_sha256",
            "verified_success_evidence_sha256",
        },
        "record evidence",
    )
    for name in evidence:
        _sha(evidence[name], f"record evidence {name}")
    if record["reasoning_visibility"] != "student_visible":
        raise ValueError("private or unknown reasoning cannot be materialized")
    messages = record["messages"]
    if not isinstance(messages, list) or not messages:
        raise ValueError("private visible-reasoning record has no messages")
    checked_messages = [_message(message) for message in messages]
    if digest_json(checked_messages) != evidence["normalized_trajectory_sha256"]:
        raise ValueError("record trajectory digest differs from its exact messages")
    raw_windows = record["windows"]
    if not isinstance(raw_windows, list) or not raw_windows:
        raise ValueError("private visible-reasoning record has no windows")
    windows = [_window(window) for window in raw_windows]
    by_id = {window["window_id"]: window for window in windows}
    if len(by_id) != len(windows):
        raise ValueError("private visible-reasoning record duplicates a window identity")
    _compaction(record["compaction"], by_id, checked_messages, profile["source"]["kind"])
    return record


def _load_renderer(
    path: Path, profile: dict[str, Any]
) -> Callable[[dict[str, Any], dict[str, Any]], Any]:
    expected = profile["serialization"]["renderer_adapter_sha256"]
    if file_sha256(path) != expected:
        raise ValueError("Qwen/OpenCode renderer adapter digest mismatch")
    spec = importlib.util.spec_from_file_location("_qwen_visible_reasoning_renderer", path)
    if spec is None or spec.loader is None:
        raise ValueError("Qwen/OpenCode renderer adapter is not loadable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    render = getattr(module, "render_visible_reasoning_window", None)
    if not callable(render):
        raise ValueError("Qwen/OpenCode renderer adapter lacks render_visible_reasoning_window")
    return render


def _rendered_window(
    renderer: Callable[[dict[str, Any], dict[str, Any]], Any],
    record: dict[str, Any],
    window: dict[str, Any],
) -> None:
    """Compare the stored private tokens with an exact pinned serialization adapter."""
    try:
        rendered = renderer(record, window)
    except Exception as error:  # no payload is included in the public error
        raise ValueError("Qwen/OpenCode serialization adapter failed") from error
    checked = _exact(
        rendered,
        {"input_ids", "prompt_token_count", "target_spans"},
        "renderer output",
    )
    expected = {
        "input_ids": window["input_ids"],
        "prompt_token_count": window["prompt_token_count"],
        "target_spans": window["target_spans"],
    }
    if checked != expected:
        raise ValueError(
            "Qwen/OpenCode template serialization differs from collected token boundaries"
        )


def _loss_mask(window: dict[str, Any]) -> list[int]:
    mask = [0] * len(window["input_ids"])
    for span in window["target_spans"]:
        mask[span["token_start"] : span["token_end"]] = [1] * (
            span["token_end"] - span["token_start"]
        )
    return mask


def build(config: Mapping[str, Any], *, relative_to: Path) -> dict[str, Any]:
    """Build a private, token-only Qwen visible-reasoning corpus.

    The method validates all public metadata and split boundaries before it
    opens the private records file.  It returns aggregates only.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    _known(
        config,
        {
            "schema",
            "source_profile",
            "collection_packet",
            "selection",
            "inventory",
            "family_split",
            "role_anchor",
            "protected_family_lock",
            "runtime_bindings",
            "roundtrip_fixture",
            "records",
            "renderer_adapter",
            "output",
        },
        "visible-reasoning corpus materialization request",
    )
    if config.get("schema") != REQUEST_SCHEMA:
        raise ValueError("unsupported visible-reasoning corpus materialization request")
    output = _path(relative_to, config.get("output"), "output")
    if output.exists() or output.is_symlink():
        raise FileExistsError("create-once visible-reasoning corpus destination exists")
    public_names = (
        "source_profile",
        "collection_packet",
        "selection",
        "inventory",
        "family_split",
        "role_anchor",
        "protected_family_lock",
        "runtime_bindings",
        "roundtrip_fixture",
        "renderer_adapter",
    )
    sources = {name: _input(relative_to, config.get(name), name) for name in public_names}
    records_path, records_sha256 = _private_records_input(relative_to, config.get("records"))
    sources["records"] = (records_path, records_sha256)
    paths = {name: item[0] for name, item in sources.items()}
    expected_files = {name: item[1] for name, item in sources.items()}
    profile = _profile(_json(paths["source_profile"], "source profile"))
    if expected_files["roundtrip_fixture"] != profile["serialization"]["roundtrip_fixture_sha256"]:
        raise ValueError("round-trip fixture file does not match the source profile")
    _roundtrip(_json(paths["roundtrip_fixture"], "round-trip fixture"), profile)
    packet = _packet(_json(paths["collection_packet"], "collection packet"), profile)
    inventory = _json(paths["inventory"], "catalog inventory")
    split = _json(paths["family_split"], "family split")
    role_anchor = _json(paths["role_anchor"], "family role anchor")
    lock = _json(paths["protected_family_lock"], "protected-family lock")
    runtime = _json(paths["runtime_bindings"], "runtime bindings")
    bindings = _task_boundary(inventory, split, role_anchor, lock, runtime, packet)
    selection = _json(paths["selection"], "private selection")
    selected = _selection(selection, packet, split, lock, bindings)
    renderer = _load_renderer(paths["renderer_adapter"], profile)

    # This is the first point at which the private payload is read.  A bad
    # catalog, split, anchor, lock, packet, or selection stops above without
    # hashing or parsing raw trajectory content.
    if file_sha256(paths["records"]) != expected_files["records"]:
        raise ValueError("records file digest mismatch")
    records: dict[str, dict[str, Any]] = {}
    for raw in iter_jsonl(paths["records"]):
        record = _record(raw, profile)
        if record["record_id"] in records:
            raise ValueError("private records duplicate a record identity")
        records[record["record_id"]] = record
    if set(records) != set(selected):
        raise ValueError("private records do not cover the exact sealed selection")

    rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    window_digests: set[str] = set()
    target_occurrences: set[str] = set()
    family_tokens: dict[str, int] = {}
    reasoning_tokens = action_tokens = 0
    compacted_windows = 0
    for record_id, selected_row in sorted(selected.items()):
        record = records[record_id]
        if (
            record["lineage"]
            != {
                "task_key": selected_row["task_key"],
                "task_version_id": selected_row["task_version_id"],
                "group_id": selected_row["group_id"],
            }
            or record["evidence"]["source_session_identity_sha256"]
            != selected_row["source_session_identity_sha256"]
            or record["evidence"]["normalized_trajectory_sha256"]
            != selected_row["normalized_trajectory_sha256"]
            or record["evidence"]["transcript_sha256"] != selected_row["transcript_sha256"]
            or record["evidence"]["verified_success_evidence_sha256"]
            != selection["verified_success_evidence_sha256"]
            or digest_json(record) != selected_row["normalized_record_sha256"]
            or ("none" if record["compaction"]["kind"] == "none" else EXACT_COMPACTION)
            != selected_row["compaction_kind"]
        ):
            raise ValueError("private record differs from its sealed success selection")
        record_reasoning = record_action = 0
        for window in record["windows"]:
            _rendered_window(renderer, record, window)
            payload = window["window_payload_sha256"]
            if payload in window_digests:
                raise ValueError("duplicate packed window would repeat supervised targets")
            window_digests.add(payload)
            mask = _loss_mask(window)
            for span in window["target_spans"]:
                # Equal token bytes can occur in two genuinely distinct tool
                # actions.  Deduplicate the exact source occurrence, not a
                # token-content hash that would incorrectly erase one of them.
                occurrence = digest_json(
                    [
                        record["evidence"]["normalized_trajectory_sha256"],
                        payload,
                        span["kind"],
                        span["token_start"],
                        span["token_end"],
                    ]
                )
                if occurrence in target_occurrences:
                    raise ValueError("one visible target span appears more than once")
                target_occurrences.add(occurrence)
                width = span["token_end"] - span["token_start"]
                family_tokens[selected_row["group_id"]] = (
                    family_tokens.get(selected_row["group_id"], 0) + width
                )
                if span["kind"] == "student_visible_reasoning":
                    reasoning_tokens += width
                    record_reasoning += width
                else:
                    action_tokens += width
                    record_action += width
            rows.append(
                {
                    "input_ids": window["input_ids"],
                    "loss_mask": mask,
                    "source_record_sha256": selected_row["normalized_record_sha256"],
                    "source_task_version_id": selected_row["task_version_id"],
                    "source_window_sha256": payload,
                    "source_profile_sha256": profile["sha256"],
                    "collection_packet_sha256": packet["sha256"],
                }
            )
        if not record_reasoning or not record_action:
            raise ValueError("selected record lacks paired visible reasoning and action coverage")
        if record["compaction"]["kind"] == EXACT_COMPACTION:
            compacted_windows += len(record["windows"])
        selection_rows.append(
            {
                "record_id": record_id,
                "normalized_record_sha256": selected_row["normalized_record_sha256"],
                "task_key": selected_row["task_key"],
                "task_version_id": selected_row["task_version_id"],
                "group_id": selected_row["group_id"],
                "reasoning_target_tokens": record_reasoning,
                "action_target_tokens": record_action,
            }
        )
    if not rows:
        raise ValueError("visible-reasoning materialization produced no train windows")
    total_tokens = reasoning_tokens + action_tokens
    family_total = sum(family_tokens.values())
    family_fraction = max(family_tokens.values()) / family_total
    ready = total_tokens >= packet["minimum_unique_supervised_tokens"]
    if ready and family_fraction > packet["maximum_family_target_token_fraction"]:
        raise ValueError("target-ready corpus exceeds immutable family concentration limit")
    coverage = {
        "schema": COVERAGE_SCHEMA,
        "selection_sha256": selection["sha256"],
        "collection_packet_sha256": packet["sha256"],
        "source_profile_sha256": profile["sha256"],
        "selected_source_records": len(selection_rows),
        "visible_reasoning_windows": len(rows),
        "student_visible_reasoning_target_tokens": reasoning_tokens,
        "visible_action_target_tokens": action_tokens,
        "unique_supervised_tokens": total_tokens,
        "minimum_unique_supervised_tokens": packet["minimum_unique_supervised_tokens"],
        "target_goal_reached": ready,
        "family_token_concentration": {
            "families_with_targets": len(family_tokens),
            "largest_family_target_token_fraction": family_fraction,
            "maximum_allowed_fraction": packet["maximum_family_target_token_fraction"],
            "within_limit": family_fraction <= packet["maximum_family_target_token_fraction"],
        },
        "compaction": {
            "uncompacted_windows": len(rows) - compacted_windows,
            "exact_student_generated_continuation_windows": compacted_windows,
            "opaque_compaction_windows": 0,
        },
        "heldout_families_materialized": 0,
        "raw_text_written": False,
    }
    coverage["sha256"] = digest_json(coverage)
    manifest = {
        "schema": CORPUS_SCHEMA,
        "source_profile_sha256": profile["sha256"],
        "collection_packet_sha256": packet["sha256"],
        "selection_sha256": coverage["selection_sha256"],
        "catalog_inventory_sha256": packet["catalog_inventory_sha256"],
        "family_split_sha256": packet["family_split_sha256"],
        "root_role_anchor_id": TRUSTED_FLEET_COLLECTION_ROOT_ID,
        "family_role_anchor_sha256": packet["family_role_anchor_sha256"],
        "protected_family_lock_sha256": packet["protected_family_lock_sha256"],
        "runtime_bindings_sha256": packet["runtime_bindings_sha256"],
        "files": {"train": {"path": "train.parquet", "sha256": None, "rows": len(rows)}},
        "counts": {
            "source_records": len(selection_rows),
            "windows": len(rows),
            "student_visible_reasoning_target_tokens": reasoning_tokens,
            "visible_action_target_tokens": action_tokens,
            "supervised_tokens": total_tokens,
        },
        "validation_mode": "task_outcomes_only" if ready else "collection_pending_target",
        "coverage_sha256": coverage["sha256"],
        "limitations": [
            "Only explicitly authorized student-visible reasoning is supervised.",
            "Private or unknown reasoning fields and opaque compaction are rejected.",
            "Output contains token IDs and masks only; it contains no raw source text.",
        ],
        "builder_sha256": {"fleet_visible_reasoning_corpus.py": file_sha256(Path(__file__))},
    }
    for name, expected in expected_files.items():
        if file_sha256(paths[name]) != expected:
            raise ValueError(f"{name} changed during visible-reasoning materialization")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.mkdir(output, 0o700)
    except FileExistsError as error:
        raise FileExistsError("create-once visible-reasoning corpus destination exists") from error
    try:
        train_path = output / "train.parquet"
        pq.write_table(pa.Table.from_pylist(rows), train_path, compression="zstd")
        os.chmod(train_path, 0o600)
        if pq.read_table(train_path).to_pylist() != rows:
            raise ValueError("private visible-reasoning Parquet readback differs")
        manifest["files"]["train"]["sha256"] = file_sha256(train_path)
        manifest["sha256"] = digest_json(manifest)
        atomic_write_jsonl(output / "source-selection.private.jsonl", selection_rows, private=True)
        atomic_write_json(output / "coverage.private.json", coverage, private=True)
        atomic_write_json(output / "manifest.json", manifest, private=True)
        receipt = {
            "schema": RECEIPT_SCHEMA,
            "manifest_file_sha256": file_sha256(output / "manifest.json"),
            "manifest_sha256": manifest["sha256"],
            "coverage_file_sha256": file_sha256(output / "coverage.private.json"),
            "coverage_sha256": coverage["sha256"],
            "source_selection_file_sha256": file_sha256(output / "source-selection.private.jsonl"),
            "sft_ready": ready,
        }
        receipt["sha256"] = digest_json(receipt)
        atomic_write_json(output / "MATERIALIZATION.json", receipt, private=True)
    except BaseException:
        # Never overwrite a partial private destination: it is evidence to
        # inspect, not an invitation to replay over the same path.
        raise
    return {
        "submitted": False,
        "manifest_sha256": manifest["sha256"],
        "coverage_sha256": coverage["sha256"],
        "source_records": len(selection_rows),
        "student_visible_reasoning_target_tokens": reasoning_tokens,
        "visible_action_target_tokens": action_tokens,
        "sft_ready": ready,
    }
