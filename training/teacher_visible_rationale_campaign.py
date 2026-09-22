"""Source-only contract for stronger-teacher, student-visible rationales.

This lane is deliberately separate from both the existing teacher/action-only
corpus and the Qwen-self visible-reasoning corpus.  It never accepts a provider
``thinking``/``analysis`` field.  The only rationale surface is ordinary
assistant content produced in response to the reviewed visible instruction.

The renderer is offline and produces a profile plus collection packet after an
immutable teacher authorization is supplied.  The admission step reads only
sealed metadata, verifier evidence, digests, and token counts.  It never opens
prompts, transcripts, rationales, tool output, flags, or token IDs, and it never
creates a rollout or training workload.
"""

from __future__ import annotations

import collections
import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .io import atomic_write_json, digest_json, file_sha256, iter_jsonl
from .task_family_split import (
    ANCHORED_SCHEMA,
    TRUSTED_FLEET_COLLECTION_ROOT_ID,
    require_trusted_fleet_collection_root_anchor,
)
from .task_family_split import validate as validate_split

REQUIREMENTS_SCHEMA = "cyber_teacher_visible_rationale_campaign_requirements_v1"
SOURCE_AUTHORIZATION_SCHEMA = "cyber_teacher_visible_rationale_source_authorization_v1"
SOURCE_PROFILE_SCHEMA = "cyber_teacher_visible_rationale_source_profile_v1"
PACKET_SCHEMA = "cyber_teacher_visible_rationale_packet_v1"
ATTEMPT_SCHEMA = "cyber_teacher_visible_rationale_attempt_metadata_v1"
ADMISSION_REQUEST_SCHEMA = "cyber_teacher_visible_rationale_admission_request_v1"
SELECTION_SCHEMA = "cyber_teacher_visible_rationale_selection_v1"
RECEIPT_SCHEMA = "cyber_teacher_visible_rationale_admission_receipt_v1"
ROUNDTRIP_SCHEMA = "cyber_qwen_opencode_template_roundtrip_v1"

QWEN_REPOSITORY = "Qwen/Qwen3.8-27B"
QWEN_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
OPENCODE_VERSION = "1.18.27"
ONLINE_COMPACTION = "opencode_1.18.27_native_compaction_autocontinue_v2"
EXACT_VISIBLE_SUMMARY = "teacher_visible_exact_summary_v1"
MINIMUM_UNIQUE_SUPERVISED_TOKENS = 20_000_000
MAXIMUM_FAMILY_TOKEN_FRACTION = 0.25
MINIMUM_SELECTED_FAMILIES = 40
MINIMUM_SELECTED_FAMILY_FRACTION = 0.8
MESSAGE_SURFACE = "qwen_chat_messages_with_opencode_tool_calls_v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

_SHA256 = re.compile(r"sha256:[0-9a-f]{64}")
_OPAQUE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/+-]{0,255}")
_PRIVATE_FIELD_NAMES = (
    "analysis",
    "hidden_reasoning",
    "private_reasoning",
    "reasoning",
    "reasoning_content",
    "thinking",
)
_SOURCE_AUTHORITY_PREFIX = "cyber/runs/qwen38/teacher-visible-rationale/authorization/"
_SUCCESS_EVIDENCE_AUTHORITY_PREFIX = "cyber/runs/qwen38/teacher-visible-rationale/success/"
_VERIFIER_AUTHORITY_PREFIX = "cyber/runs/qwen38/teacher-visible-rationale/verifier/"
_ROUNDTRIP_AUTHORITY_PREFIX = "cyber/runs/qwen38/teacher-visible-rationale/roundtrip/"
_REJECTIONS = (
    "ambiguous_campaign_cell",
    "duplicate_record",
    "duplicate_source_session",
    "duplicate_trajectory",
    "heldout_or_nontrain",
    "invalid_authoritative_success",
    "invalid_rationale_coverage",
    "missing_visible_rationale",
    "opaque_or_invalid_compaction",
    "per_task_cap",
    "private_or_unknown_reasoning",
    "serialization_mismatch",
    "success_evidence_binding_mismatch",
    "unplanned_campaign_cell",
    "wrong_source_binding",
)
_PUBLIC_FORBIDDEN_KEYS = {
    "answer",
    "content",
    "credential",
    "flag",
    "message",
    "prompt",
    "record_id",
    "session_id",
    "task_key",
    "task_version_id",
    "trace",
    "transcript",
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


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be an exact sha256 digest")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or _OPAQUE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a nonempty opaque string")
    return value


def _count(value: object, label: str, *, positive: bool = False) -> int:
    if type(value) is not int or value < 0 or (positive and value == 0):
        raise ValueError(f"{label} must be a {'positive' if positive else 'nonnegative'} integer")
    return value


def _sealed(value: Mapping[str, Any], schema: str, label: str) -> dict[str, Any]:
    result = _mapping(value, label)
    if result.get("schema") != schema or result.get("sha256") != digest_json(
        {key: item for key, item in result.items() if key != "sha256"}
    ):
        raise ValueError(f"invalid {label} schema or digest")
    return result


def _path(root: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} path is required")
    path = Path(value)
    return path if path.is_absolute() else root / path


def _input(root: Path, value: object, label: str) -> tuple[Path, str]:
    reference = _exact(value, {"path", "sha256"}, label)
    path = _path(root, reference["path"], label)
    expected = _sha(reference["sha256"], f"{label} file")
    if path.is_symlink() or not path.is_file() or file_sha256(path) != expected:
        raise ValueError(f"{label} file digest mismatch")
    return path, expected


def _require_unchanged(inputs: Mapping[str, tuple[Path, str]], label: str) -> None:
    for name, (path, expected) in inputs.items():
        if path.is_symlink() or not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"{label} {name} changed during processing")


def _requirements_local_inputs(
    requirements: Mapping[str, Any], *, root: Path
) -> dict[str, tuple[Path, str]]:
    return {
        "qwen_model_lock": (
            _path(root, requirements["qwen_target"]["model_lock_path"], "model lock"),
            requirements["qwen_target"]["model_lock_file_sha256"],
        ),
        "visible_rationale_instruction": (
            _path(
                root,
                requirements["visible_rationale"]["instruction_path"],
                "rationale instruction",
            ),
            requirements["visible_rationale"]["instruction_file_sha256"],
        ),
    }


def _json(path: Path, label: str) -> dict[str, Any]:
    try:
        return _mapping(json.loads(path.read_text()), label)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label} JSON") from error


def _immutable_authority(value: object, label: str) -> dict[str, Any]:
    result = _exact(
        value,
        {"kind", "artifact_key", "version_index", "content_sha256"},
        label,
    )
    if (
        result["kind"] != "fleet_artifact_registry_immutable_v1"
        or not isinstance(result["artifact_key"], str)
        or not result["artifact_key"].startswith("cyber/runs/")
        or type(result["version_index"]) is not int
        or result["version_index"] < 1
    ):
        raise ValueError(f"{label} lacks an immutable Fleet artifact binding")
    _sha(result["content_sha256"], f"{label} content")
    return result


def _namespaced_authority(value: object, label: str, prefix: str) -> dict[str, Any]:
    result = _immutable_authority(value, label)
    if not result["artifact_key"].startswith(prefix):
        raise ValueError(f"{label} uses the wrong immutable Registry namespace")
    return result


def _registry_payload_sha256(value: Mapping[str, Any]) -> str:
    return digest_json(
        {
            key: item
            for key, item in value.items()
            if key not in {"authority", "registry_payload_sha256", "sha256"}
        }
    )


def _public_only(value: object) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).lower() in _PUBLIC_FORBIDDEN_KEYS:
                raise ValueError("public receipt would expose private collection metadata")
            _public_only(item)
    elif isinstance(value, list):
        for item in value:
            _public_only(item)


def _requirements(value: Mapping[str, Any], *, root: Path) -> dict[str, Any]:
    req = _sealed(value, REQUIREMENTS_SCHEMA, "teacher-visible-rationale requirements")
    _exact(
        req,
        {
            "schema",
            "campaign_name",
            "status",
            "source",
            "qwen_target",
            "opencode",
            "visible_rationale",
            "serialization",
            "compaction",
            "roster",
            "collection",
            "admission",
            "separation",
            "submitted",
            "sha256",
        },
        "teacher-visible-rationale requirements",
    )
    if req["status"] != "source_only_requires_exact_teacher_authorization":
        raise ValueError("teacher-visible-rationale campaign must fail closed on source authority")
    source = _exact(
        req["source"],
        {
            "kind",
            "provider",
            "model",
            "authorization_schema",
            "teacher_strength_receipt_required",
        },
        "teacher source requirement",
    )
    if source != {
        "kind": "teacher_visible_rationale",
        "provider": "openai",
        "model": "gpt-5.6-sol",
        "authorization_schema": SOURCE_AUTHORIZATION_SCHEMA,
        "teacher_strength_receipt_required": True,
    }:
        raise ValueError("teacher source requirement drift")
    target = _exact(
        req["qwen_target"],
        {
            "repository",
            "revision",
            "model_lock_path",
            "model_lock_file_sha256",
            "tokenizer_manifest_sha256",
            "chat_template_sha256",
            "local_roundtrip_required",
        },
        "Qwen target",
    )
    if (
        target["repository"] != QWEN_REPOSITORY
        or target["revision"] != QWEN_REVISION
        or target["local_roundtrip_required"] is not True
    ):
        raise ValueError("teacher rationale target must be exact Qwen3.8-27B")
    for name in ("model_lock_file_sha256", "tokenizer_manifest_sha256", "chat_template_sha256"):
        _sha(target[name], f"Qwen target {name}")
    model_lock = _path(root, target["model_lock_path"], "model lock")
    if (
        model_lock.is_symlink()
        or not model_lock.is_file()
        or file_sha256(model_lock) != target["model_lock_file_sha256"]
    ):
        raise ValueError("Qwen model lock file drift")
    opencode = _exact(
        req["opencode"],
        {
            "harness",
            "version",
            "release_asset_sha256",
            "tool_catalog_sha256",
            "tools",
            "context_window_tokens",
            "compaction_headroom_tokens",
            "context_management",
        },
        "OpenCode contract",
    )
    if (
        opencode["harness"] != "opencode"
        or opencode["version"] != OPENCODE_VERSION
        or opencode["tools"] != ["bash", "submit_report"]
        or opencode["context_window_tokens"] != 262_144
        or opencode["compaction_headroom_tokens"] != 20_000
        or opencode["context_management"] != ONLINE_COMPACTION
    ):
        raise ValueError("teacher rationale OpenCode contract drift")
    for name in ("release_asset_sha256", "tool_catalog_sha256"):
        _sha(opencode[name], f"OpenCode {name}")
    rationale = _exact(
        req["visible_rationale"],
        {
            "instruction_path",
            "instruction_file_sha256",
            "surface",
            "minimum_sentences_before_tool",
            "maximum_sentences_before_tool",
            "every_tool_call_requires_visible_rationale",
            "private_fields_rejected",
            "inference_or_reconstruction_allowed",
        },
        "visible rationale contract",
    )
    instruction = _path(root, rationale["instruction_path"], "rationale instruction")
    if (
        instruction.is_symlink()
        or not instruction.is_file()
        or file_sha256(instruction) != rationale["instruction_file_sha256"]
        or rationale["surface"] != "ordinary_assistant_content_before_tool_call"
        or rationale["minimum_sentences_before_tool"] != 1
        or rationale["maximum_sentences_before_tool"] != 4
        or rationale["every_tool_call_requires_visible_rationale"] is not True
        or rationale["private_fields_rejected"] != list(_PRIVATE_FIELD_NAMES)
        or rationale["inference_or_reconstruction_allowed"] is not False
    ):
        raise ValueError("visible rationale prompt or safety contract drift")
    serialization = _exact(
        req["serialization"],
        {
            "schema",
            "message_surface",
            "assistant_rationale_surface",
            "chat_template_kwargs",
            "prompt_messages",
            "prompt_add_generation_prompt",
            "prompt_token_ids_must_equal_local_apply_chat_template",
            "prompt_token_ids_exact_prefix_of_every_full_serialization",
            "collection_training_and_serving_token_ids_must_match",
            "private_provider_fields_forbidden",
        },
        "Qwen/OpenCode serialization contract",
    )
    if serialization != {
        "schema": ROUNDTRIP_SCHEMA,
        "message_surface": MESSAGE_SURFACE,
        "assistant_rationale_surface": "ordinary_assistant_content",
        "chat_template_kwargs": {"enable_thinking": False},
        "prompt_messages": "messages_without_target",
        "prompt_add_generation_prompt": True,
        "prompt_token_ids_must_equal_local_apply_chat_template": True,
        "prompt_token_ids_exact_prefix_of_every_full_serialization": True,
        "collection_training_and_serving_token_ids_must_match": True,
        "private_provider_fields_forbidden": list(_PRIVATE_FIELD_NAMES),
    }:
        raise ValueError("teacher rationale Qwen/OpenCode serialization contract drift")
    compaction = _exact(
        req["compaction"],
        {
            "online_kind",
            "accepted_offline_kind",
            "summary_visibility",
            "summary_loss",
            "exact_pre_summary_post_next_prompt_lineage_required",
            "opaque_rejected",
        },
        "teacher rationale compaction",
    )
    if compaction != {
        "online_kind": ONLINE_COMPACTION,
        "accepted_offline_kind": EXACT_VISIBLE_SUMMARY,
        "summary_visibility": "student_visible",
        "summary_loss": "context_only_zero_loss",
        "exact_pre_summary_post_next_prompt_lineage_required": True,
        "opaque_rejected": True,
    }:
        raise ValueError("teacher rationale compaction contract drift")
    collection = _exact(
        req["collection"],
        {
            "attempts_per_task",
            "concurrency",
            "minimum_unique_supervised_tokens",
            "maximum_family_target_token_fraction",
            "maximum_sessions_per_task_version",
            "minimum_selected_families",
            "minimum_selected_family_fraction",
        },
        "teacher rationale collection budget",
    )
    for name in (
        "attempts_per_task",
        "concurrency",
        "maximum_sessions_per_task_version",
        "minimum_selected_families",
    ):
        _count(collection[name], name, positive=True)
    if (
        type(collection["minimum_selected_family_fraction"]) is not float
        or not 0 < collection["minimum_selected_family_fraction"] <= 1
        or collection["minimum_unique_supervised_tokens"] < MINIMUM_UNIQUE_SUPERVISED_TOKENS
        or collection["maximum_family_target_token_fraction"] != MAXIMUM_FAMILY_TOKEN_FRACTION
        or collection["minimum_selected_families"] < MINIMUM_SELECTED_FAMILIES
        or collection["minimum_selected_family_fraction"] < MINIMUM_SELECTED_FAMILY_FRACTION
    ):
        raise ValueError("teacher rationale collection scale gate drift")
    admission = _exact(
        req["admission"],
        {
            "required_outcome",
            "required_rationale_surface",
            "qwen_roundtrip_required",
            "deduplication_order",
            "heldout_roles_excluded",
            "opaque_compaction_rejected",
            "private_or_unknown_reasoning_rejected",
            "immutable_success_evidence_required",
            "durable_evidence_handoff_required",
            "final_unique_token_gate_requires_private_materializer",
        },
        "teacher rationale admission policy",
    )
    if admission != {
        "required_outcome": "completed_authoritative_verifier_success",
        "required_rationale_surface": "ordinary_assistant_content_before_tool_call",
        "qwen_roundtrip_required": True,
        "deduplication_order": [
            "record_identity",
            "source_session_identity",
            "normalized_trajectory_digest",
            "packed_window_payload_digest",
        ],
        "heldout_roles_excluded": ["dev", "final_test"],
        "opaque_compaction_rejected": True,
        "private_or_unknown_reasoning_rejected": True,
        "immutable_success_evidence_required": True,
        "durable_evidence_handoff_required": True,
        "final_unique_token_gate_requires_private_materializer": True,
    }:
        raise ValueError("teacher rationale admission policy drift")
    if (
        req["separation"]
        != {
            "reuse_action_only_packet": False,
            "reuse_qwen_self_packet": False,
            "mix_with_action_only_corpus": False,
            "mix_with_qwen_self_corpus": False,
            "teacher_hidden_reasoning_allowed": False,
        }
        or req["submitted"] is not False
    ):
        raise ValueError("teacher rationale separation or source-only status drift")
    return req


def _authorization(value: Mapping[str, Any], requirements: dict[str, Any]) -> dict[str, Any]:
    auth = _sealed(value, SOURCE_AUTHORIZATION_SCHEMA, "teacher source authorization")
    _exact(
        auth,
        {
            "schema",
            "authority",
            "registry_payload_sha256",
            "source",
            "teacher_strength_receipt_sha256",
            "visible_output",
            "training_use",
            "sha256",
        },
        "teacher source authorization",
    )
    authority = _namespaced_authority(
        auth["authority"],
        "teacher authorization authority",
        _SOURCE_AUTHORITY_PREFIX,
    )
    payload = _sha(auth["registry_payload_sha256"], "teacher authorization payload")
    if payload != _registry_payload_sha256(auth) or authority["content_sha256"] != payload:
        raise ValueError("teacher authorization is not bound to its immutable payload")
    source = _exact(
        auth["source"],
        {
            "provider",
            "model",
            "immutable_provider_revision",
            "session_model",
            "route_profile_sha256",
        },
        "authorized teacher source",
    )
    for name in source:
        _sha(source[name], f"teacher {name}") if name.endswith("sha256") else _text(
            source[name], f"teacher {name}"
        )
    required = requirements["source"]
    if source["provider"] != required["provider"] or source["model"] != required["model"]:
        raise ValueError("teacher authorization changes the reviewed provider/model")
    _sha(auth["teacher_strength_receipt_sha256"], "teacher strength receipt")
    visible = _exact(
        auth["visible_output"],
        {
            "surface",
            "instruction_file_sha256",
            "serialization_contract_sha256",
            "visible_to_student",
            "private_fields_rejected",
            "provider_private_reasoning_ingested",
        },
        "authorized visible output",
    )
    rationale = requirements["visible_rationale"]
    if visible != {
        "surface": rationale["surface"],
        "instruction_file_sha256": rationale["instruction_file_sha256"],
        "serialization_contract_sha256": digest_json(requirements["serialization"]),
        "visible_to_student": True,
        "private_fields_rejected": list(_PRIVATE_FIELD_NAMES),
        "provider_private_reasoning_ingested": False,
    }:
        raise ValueError("teacher authorization does not prove an ordinary visible rationale")
    training = _exact(
        auth["training_use"],
        {"authorized", "purpose", "target_model", "issuer", "issued_at"},
        "teacher training-use authorization",
    )
    if (
        training["authorized"] is not True
        or training["purpose"] != "qwen38_teacher_visible_rationale_plus_actions"
        or training["target_model"] != f"{QWEN_REPOSITORY}@{QWEN_REVISION}"
    ):
        raise ValueError("teacher rationale is not explicitly authorized for Qwen training")
    _text(training["issuer"], "authorization issuer")
    _text(training["issued_at"], "authorization timestamp")
    return auth


def _roster(requirements: dict[str, Any], *, root: Path) -> dict[str, Any]:
    refs = _exact(
        requirements["roster"],
        {
            "inventory",
            "family_split",
            "role_anchor",
            "protected_family_lock",
            "runtime_bindings",
            "task_selection",
        },
        "teacher rationale roster",
    )
    loaded: dict[str, dict[str, Any]] = {}
    files: dict[str, str] = {}
    paths: dict[str, Path] = {}
    for name, reference in refs.items():
        path, expected = _input(root, reference, name.replace("_", " "))
        loaded[name] = _json(path, name.replace("_", " "))
        files[name] = expected
        paths[name] = path
    inventory = _sealed(
        loaded["inventory"],
        "cyber_collection_metadata_inventory_v1",
        "teacher rationale inventory",
    )
    split = _sealed(loaded["family_split"], ANCHORED_SCHEMA, "teacher rationale family split")
    anchor = _sealed(
        loaded["role_anchor"],
        "cyber_task_family_role_anchor_v1",
        "teacher rationale role anchor",
    )
    lock = _sealed(
        loaded["protected_family_lock"],
        "cyber_protected_task_family_lock_v1",
        "teacher rationale heldout lock",
    )
    runtime = _sealed(
        loaded["runtime_bindings"],
        "cyber_collection_runtime_bindings_v1",
        "teacher rationale runtime bindings",
    )
    selection = _sealed(
        loaded["task_selection"],
        "cyber_collection_task_selection_v1",
        "teacher rationale task selection",
    )
    require_trusted_fleet_collection_root_anchor(anchor)
    rows = inventory.get("task_versions")
    if not isinstance(rows, list) or not rows:
        raise ValueError("teacher rationale inventory has no task versions")
    if split.get("inventory_sha256") != inventory.get("sha256"):
        raise ValueError("teacher rationale split is not anchored to its inventory")
    validate_split(split, rows, role_anchor=anchor)
    if (
        split.get("parent_role_anchor_sha256") != anchor.get("sha256")
        or lock.get("source_split_sha256") != split.get("sha256")
        or runtime.get("metadata_inventory_sha256") != inventory.get("sha256")
        or runtime.get("task_validity_receipt_sha256")
        != inventory.get("task_validity_receipt_sha256")
        or selection.get("inventory_sha256") != inventory.get("sha256")
        or selection.get("family_split_sha256") != split.get("sha256")
        or selection.get("family_role_anchor_sha256") != anchor.get("sha256")
        or selection.get("runtime_bindings_sha256") != runtime.get("sha256")
    ):
        raise ValueError("teacher rationale role anchor or heldout lock drift")
    roles = {(row["task_key"], row["task_version_id"]): row for row in split.get("tasks", [])}
    protected = sorted(
        {row["group_id"] for row in roles.values() if row["split"] in {"dev", "final_test"}}
    )
    if lock.get("heldout_group_ids") != protected:
        raise ValueError("teacher rationale heldout lock is incomplete")
    selected = selection.get("tasks")
    if not isinstance(selected, list) or not selected:
        raise ValueError("teacher rationale task selection is empty")
    identities: dict[tuple[str, str], dict[str, Any]] = {}
    groups: set[str] = set()
    runtime_rows = {
        (row.get("task_key"), row.get("task_version_id")): row
        for row in runtime.get("task_versions", [])
    }
    inventory_identities = {(row.get("task_key"), row.get("task_version_id")) for row in rows}
    if set(runtime_rows) != inventory_identities:
        raise ValueError("teacher rationale runtime bindings do not cover the exact inventory")
    for task in selected:
        identity = (task.get("task_key"), task.get("task_version_id"))
        assignment = roles.get(identity)
        if identity in identities or assignment is None or assignment["split"] != "train":
            raise ValueError("teacher rationale task selection reaches a heldout or duplicate task")
        if assignment["group_id"] in protected or assignment["group_id"] in groups:
            raise ValueError("teacher rationale selection repeats or reaches a heldout family")
        if runtime_rows.get(identity) != task:
            raise ValueError("teacher rationale selection changes an exact runtime binding")
        groups.add(assignment["group_id"])
        identities[identity] = assignment
    return {
        "files": files,
        "paths": paths,
        "loaded": loaded,
        "roles": roles,
        "protected": set(protected),
        "selected": identities,
        "selected_runtime_bindings": {identity: runtime_rows[identity] for identity in identities},
    }


def render(
    requirements: Mapping[str, Any], authorization: Mapping[str, Any], *, root: Path
) -> dict[str, dict[str, Any]]:
    """Render the immutable source profile and packet without external calls."""

    req = _requirements(requirements, root=root)
    requirements_local_inputs = _requirements_local_inputs(req, root=root)
    auth = _authorization(authorization, req)
    roster = _roster(req, root=root)
    if len(roster["selected"]) < req["collection"]["minimum_selected_families"]:
        raise ValueError("teacher rationale roster is too narrow for the reviewed breadth gate")
    profile = {
        "schema": SOURCE_PROFILE_SCHEMA,
        "source": {"kind": "teacher_visible_rationale", **auth["source"]},
        "source_authorization_sha256": auth["sha256"],
        "teacher_strength_receipt_sha256": auth["teacher_strength_receipt_sha256"],
        "qwen_target": req["qwen_target"],
        "opencode": req["opencode"],
        "visible_rationale": req["visible_rationale"],
        "serialization": req["serialization"],
        "compaction": req["compaction"],
    }
    profile["sha256"] = digest_json(profile)
    collection = req["collection"]
    packet = {
        "schema": PACKET_SCHEMA,
        "campaign_name": req["campaign_name"],
        "source_profile_sha256": profile["sha256"],
        "source_authorization_sha256": auth["sha256"],
        "task_selection_sha256": roster["loaded"]["task_selection"]["sha256"],
        "catalog_inventory_sha256": roster["loaded"]["inventory"]["sha256"],
        "family_split_sha256": roster["loaded"]["family_split"]["sha256"],
        "root_role_anchor_id": TRUSTED_FLEET_COLLECTION_ROOT_ID,
        "family_role_anchor_sha256": roster["loaded"]["role_anchor"]["sha256"],
        "protected_family_lock_sha256": roster["loaded"]["protected_family_lock"]["sha256"],
        "runtime_bindings_sha256": roster["loaded"]["runtime_bindings"]["sha256"],
        "train_task_versions": len(roster["selected"]),
        "attempts_per_task": collection["attempts_per_task"],
        "planned_cells": len(roster["selected"]) * collection["attempts_per_task"],
        "minimum_unique_supervised_tokens": collection["minimum_unique_supervised_tokens"],
        "maximum_family_target_token_fraction": collection["maximum_family_target_token_fraction"],
        "maximum_sessions_per_task_version": collection["maximum_sessions_per_task_version"],
        "minimum_selected_families": collection["minimum_selected_families"],
        "minimum_selected_family_fraction": collection["minimum_selected_family_fraction"],
        "admission_policy": req["admission"],
        "training_data_eligible": True,
        "external_submission_authorized": False,
    }
    packet["sha256"] = digest_json(packet)
    _require_unchanged(requirements_local_inputs, "teacher rationale contract input")
    _require_unchanged(
        {name: (roster["paths"][name], expected) for name, expected in roster["files"].items()},
        "teacher rationale roster input",
    )
    return {"source-profile.json": profile, "collection-packet.json": packet}


def write_contract(output: Path, rendered: Mapping[str, Mapping[str, Any]]) -> None:
    if output.exists() or output.is_symlink():
        raise FileExistsError("teacher rationale contract destination already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    os.mkdir(output, 0o700)
    for name, value in rendered.items():
        path = output / name
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())


def _profile(value: Mapping[str, Any]) -> dict[str, Any]:
    profile = _sealed(value, SOURCE_PROFILE_SCHEMA, "teacher rationale source profile")
    _exact(
        profile,
        {
            "schema",
            "source",
            "source_authorization_sha256",
            "teacher_strength_receipt_sha256",
            "qwen_target",
            "opencode",
            "visible_rationale",
            "serialization",
            "compaction",
            "sha256",
        },
        "teacher rationale source profile",
    )
    source = _exact(
        profile["source"],
        {
            "kind",
            "provider",
            "model",
            "immutable_provider_revision",
            "session_model",
            "route_profile_sha256",
        },
        "teacher rationale profile source",
    )
    if source["kind"] != "teacher_visible_rationale":
        raise ValueError("teacher rationale profile has the wrong source lane")
    for name in ("source_authorization_sha256", "teacher_strength_receipt_sha256"):
        _sha(profile[name], f"teacher rationale profile {name}")
    _sha(source["route_profile_sha256"], "teacher rationale route profile")
    return profile


def _packet(value: Mapping[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    packet = _sealed(value, PACKET_SCHEMA, "teacher rationale collection packet")
    _exact(
        packet,
        {
            "schema",
            "campaign_name",
            "source_profile_sha256",
            "source_authorization_sha256",
            "task_selection_sha256",
            "catalog_inventory_sha256",
            "family_split_sha256",
            "root_role_anchor_id",
            "family_role_anchor_sha256",
            "protected_family_lock_sha256",
            "runtime_bindings_sha256",
            "train_task_versions",
            "attempts_per_task",
            "planned_cells",
            "minimum_unique_supervised_tokens",
            "maximum_family_target_token_fraction",
            "maximum_sessions_per_task_version",
            "minimum_selected_families",
            "minimum_selected_family_fraction",
            "admission_policy",
            "training_data_eligible",
            "external_submission_authorized",
            "sha256",
        },
        "teacher rationale collection packet",
    )
    for name in (
        "source_profile_sha256",
        "source_authorization_sha256",
        "task_selection_sha256",
        "catalog_inventory_sha256",
        "family_split_sha256",
        "family_role_anchor_sha256",
        "protected_family_lock_sha256",
        "runtime_bindings_sha256",
    ):
        _sha(packet[name], f"teacher rationale packet {name}")
    if (
        packet.get("source_profile_sha256") != profile["sha256"]
        or packet.get("source_authorization_sha256") != profile["source_authorization_sha256"]
        or packet.get("root_role_anchor_id") != TRUSTED_FLEET_COLLECTION_ROOT_ID
        or packet.get("training_data_eligible") is not True
        or packet.get("external_submission_authorized") is not False
        or packet.get("minimum_unique_supervised_tokens") < MINIMUM_UNIQUE_SUPERVISED_TOKENS
        or packet.get("maximum_family_target_token_fraction") != MAXIMUM_FAMILY_TOKEN_FRACTION
        or packet.get("minimum_selected_families") < MINIMUM_SELECTED_FAMILIES
        or packet.get("minimum_selected_family_fraction") < MINIMUM_SELECTED_FAMILY_FRACTION
    ):
        raise ValueError("teacher rationale packet changes the reviewed collection contract")
    return packet


def _compaction(
    value: object,
    *,
    source_session_identity_sha256: str | None = None,
    normalized_trajectory_sha256: str | None = None,
    transcript_sha256: str | None = None,
    target_occurrence_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    result = _mapping(value, "teacher rationale compaction evidence")
    kind = result.get("kind")
    if kind == "none" and result == {"kind": "none", "boundaries": []}:
        return result
    if kind != EXACT_VISIBLE_SUMMARY or set(result) != {"kind", "boundaries"}:
        raise ValueError("opaque or unknown teacher compaction is forbidden")
    boundaries = result["boundaries"]
    if not isinstance(boundaries, list) or not boundaries:
        raise ValueError("teacher visible-summary compaction needs exact boundaries")
    seen: set[str] = set()
    fields = {
        "boundary_id",
        "boundary_index",
        "previous_boundary_id",
        "source_session_identity_sha256",
        "normalized_trajectory_sha256",
        "transcript_sha256",
        "target_occurrence_manifest_sha256",
        "parent_window_id",
        "parent_target_message_index",
        "pre_compaction_prompt_message_indices",
        "pre_compaction_prompt_sha256",
        "pre_compaction_prompt_tokens",
        "summary_generation_message_indices",
        "summary_generation_prompt_sha256",
        "summary_generation_prompt_tokens",
        "summary_message_index",
        "visible_summary_message_sha256",
        "visible_summary_qwen_token_sha256",
        "visible_summary_qwen_tokens",
        "post_compaction_prompt_message_indices",
        "post_compaction_prompt_sha256",
        "post_compaction_prompt_tokens",
        "next_target_window_id",
        "next_target_message_index",
        "next_target_prompt_sha256",
        "next_target_occurrence_sha256",
        "summary_visible_to_student",
        "summary_surface",
        "provider_private_reasoning_present",
        "summary_loss",
    }
    previous_boundary_id: str | None = None
    previous_summary_message_index: int | None = None
    previous_post_indices: list[int] | None = None
    previous_next_target_index: int | None = None

    def ordered_indices(value: object, label: str) -> list[int]:
        if (
            not isinstance(value, list)
            or not value
            or any(type(item) is not int or item < 0 for item in value)
            or value != sorted(set(value))
        ):
            raise ValueError(f"{label} must be one exact ordered message-index list")
        return value

    for index, raw in enumerate(boundaries):
        boundary = _exact(raw, fields, "teacher visible-summary boundary")
        boundary_id = _sha(boundary["boundary_id"], "compaction boundary")
        if boundary_id in seen:
            raise ValueError("teacher visible-summary boundary is duplicated")
        seen.add(boundary_id)
        _count(boundary["boundary_index"], "compaction boundary index")
        if boundary["previous_boundary_id"] is not None:
            _sha(boundary["previous_boundary_id"], "previous compaction boundary")
        if boundary["boundary_index"] != index or boundary["previous_boundary_id"] != (
            previous_boundary_id
        ):
            raise ValueError("teacher visible-summary boundaries are not one ordered chain")
        _text(boundary["parent_window_id"], "compaction parent window")
        _text(boundary["next_target_window_id"], "compaction next-target window")
        parent_target_index = _count(
            boundary["parent_target_message_index"], "compaction parent target message"
        )
        summary_index = _count(
            boundary["summary_message_index"], "compaction summary message", positive=True
        )
        next_target_index = _count(
            boundary["next_target_message_index"], "compaction next target message"
        )
        pre_indices = ordered_indices(
            boundary["pre_compaction_prompt_message_indices"],
            "pre-compaction prompt indices",
        )
        summary_indices = ordered_indices(
            boundary["summary_generation_message_indices"],
            "summary-generation prompt indices",
        )
        post_indices = ordered_indices(
            boundary["post_compaction_prompt_message_indices"],
            "post-compaction prompt indices",
        )
        for name in (
            "pre_compaction_prompt_tokens",
            "summary_generation_prompt_tokens",
            "post_compaction_prompt_tokens",
        ):
            _count(boundary[name], f"compaction {name}", positive=True)
        if previous_post_indices is None:
            expected_summary_indices = list(range(summary_index))
        elif previous_next_target_index is None:  # pragma: no cover - local invariant
            raise ValueError("teacher visible-summary chain lost its preceding target")
        else:
            expected_summary_indices = [
                *previous_post_indices,
                *range(previous_next_target_index, summary_index),
            ]
        if (
            max(pre_indices) >= parent_target_index
            or summary_indices != expected_summary_indices
            or parent_target_index not in summary_indices
            or parent_target_index >= summary_index
            or summary_index not in post_indices
            or max(post_indices) >= next_target_index
            or summary_index >= next_target_index
            or pre_indices == post_indices
            or boundary["pre_compaction_prompt_sha256"] == boundary["post_compaction_prompt_sha256"]
            or (
                previous_summary_message_index is not None
                and summary_index <= previous_summary_message_index
            )
        ):
            raise ValueError("teacher visible-summary message chronology is not monotone")
        for name in fields:
            if name.endswith("sha256"):
                _sha(boundary[name], f"compaction {name}")
        expected_boundary_id = digest_json(
            {name: item for name, item in boundary.items() if name != "boundary_id"}
        )
        if boundary_id != expected_boundary_id:
            raise ValueError("teacher visible-summary boundary digest mismatch")
        _count(boundary["visible_summary_qwen_tokens"], "visible summary tokens", positive=True)
        if (
            boundary["summary_visible_to_student"] is not True
            or boundary["summary_surface"] != "ordinary_assistant_content"
            or boundary["provider_private_reasoning_present"] is not False
            or boundary["summary_loss"] != "context_only_zero_loss"
            or boundary["post_compaction_prompt_sha256"] != boundary["next_target_prompt_sha256"]
            or (
                source_session_identity_sha256 is not None
                and boundary["source_session_identity_sha256"] != source_session_identity_sha256
            )
            or (
                normalized_trajectory_sha256 is not None
                and boundary["normalized_trajectory_sha256"] != normalized_trajectory_sha256
            )
            or (
                transcript_sha256 is not None and boundary["transcript_sha256"] != transcript_sha256
            )
            or (
                target_occurrence_manifest_sha256 is not None
                and boundary["target_occurrence_manifest_sha256"]
                != target_occurrence_manifest_sha256
            )
        ):
            raise ValueError("teacher compaction does not preserve the true visible next prompt")
        previous_boundary_id = boundary_id
        previous_summary_message_index = summary_index
        previous_post_indices = post_indices
        previous_next_target_index = next_target_index
    return result


def _attempt(value: Mapping[str, Any]) -> dict[str, Any]:
    row = _sealed(value, ATTEMPT_SCHEMA, "teacher rationale attempt metadata")
    _exact(
        row,
        {
            "schema",
            "record_id",
            "campaign_packet_sha256",
            "source_profile_sha256",
            "task_key",
            "task_version_id",
            "attempt",
            "source_session_identity_sha256",
            "normalized_record_sha256",
            "normalized_trajectory_sha256",
            "transcript_sha256",
            "outcome",
            "visible_rationale",
            "serialization",
            "compaction",
            "sha256",
        },
        "teacher rationale attempt metadata",
    )
    _text(row["record_id"], "teacher rationale record identity")
    _text(row["task_key"], "teacher rationale task key")
    _text(row["task_version_id"], "teacher rationale task version")
    _count(row["attempt"], "teacher rationale attempt", positive=True)
    for name in (
        "campaign_packet_sha256",
        "source_profile_sha256",
        "source_session_identity_sha256",
        "normalized_record_sha256",
        "normalized_trajectory_sha256",
        "transcript_sha256",
    ):
        _sha(row[name], name)
    rationale = _exact(
        row["visible_rationale"],
        {
            "surface",
            "ordinary_content_only",
            "visible_to_student",
            "provider_private_reasoning_present",
            "unknown_reasoning_fields_present",
            "forbidden_private_field_occurrences",
            "tool_calls",
            "tool_calls_with_visible_rationale",
            "rationale_segments",
            "minimum_sentences_per_rationale",
            "maximum_sentences_per_rationale",
            "rationale_target_tokens",
            "visible_action_target_tokens",
            "target_occurrence_manifest_sha256",
        },
        "teacher visible rationale evidence",
    )
    for name in (
        "tool_calls",
        "tool_calls_with_visible_rationale",
        "rationale_segments",
        "minimum_sentences_per_rationale",
        "maximum_sentences_per_rationale",
        "rationale_target_tokens",
        "visible_action_target_tokens",
    ):
        _count(rationale[name], name.replace("_", " "), positive=True)
    private_occurrences = _exact(
        rationale["forbidden_private_field_occurrences"],
        set(_PRIVATE_FIELD_NAMES),
        "teacher private-field occurrence census",
    )
    for name, count in private_occurrences.items():
        _count(count, f"private field {name} occurrences")
    _sha(rationale["target_occurrence_manifest_sha256"], "target occurrence manifest")
    serialization = _exact(
        row["serialization"],
        {
            "schema",
            "qwen_target_sha256",
            "qwen_chat_template_sha256",
            "roundtrip_receipt_sha256",
            "roundtrip_receipt_authority",
            "message_surface",
            "chat_template_kwargs",
            "prompt_token_ids_equal_local_template",
            "prompt_token_ids_exact_prefix",
            "collection_training_serving_token_ids_match",
            "round_trip_verified",
        },
        "teacher rationale Qwen serialization",
    )
    for name in serialization:
        if name.endswith("sha256"):
            _sha(serialization[name], name)
    roundtrip_authority = _namespaced_authority(
        serialization["roundtrip_receipt_authority"],
        "teacher rationale round-trip authority",
        _ROUNDTRIP_AUTHORITY_PREFIX,
    )
    if roundtrip_authority["content_sha256"] != serialization["roundtrip_receipt_sha256"]:
        raise ValueError("teacher rationale round-trip authority does not bind its receipt")
    if type(serialization["round_trip_verified"]) is not bool:
        raise ValueError("teacher rationale round-trip status must be boolean")
    _compaction(
        row["compaction"],
        source_session_identity_sha256=row["source_session_identity_sha256"],
        normalized_trajectory_sha256=row["normalized_trajectory_sha256"],
        transcript_sha256=row["transcript_sha256"],
        target_occurrence_manifest_sha256=rationale["target_occurrence_manifest_sha256"],
    )
    outcome = _exact(
        row["outcome"],
        {
            "authority",
            "registry_payload_sha256",
            "status",
            "verifier_process_success",
            "score_at_least_one",
            "verifier_execution_identity_sha256",
            "verifier_receipt_sha256",
            "verifier_authority",
            "attempt_binding",
        },
        "teacher rationale outcome",
    )
    for name in (
        "registry_payload_sha256",
        "verifier_execution_identity_sha256",
        "verifier_receipt_sha256",
    ):
        _sha(outcome[name], name)
    verifier_authority = _namespaced_authority(
        outcome["verifier_authority"],
        "teacher rationale verifier authority",
        _VERIFIER_AUTHORITY_PREFIX,
    )
    if verifier_authority["content_sha256"] != outcome["verifier_receipt_sha256"]:
        raise ValueError("teacher rationale verifier authority does not bind its receipt")
    success_authority = _namespaced_authority(
        outcome["authority"],
        "teacher rationale success-evidence authority",
        _SUCCESS_EVIDENCE_AUTHORITY_PREFIX,
    )
    if (
        outcome["registry_payload_sha256"] != _registry_payload_sha256(outcome)
        or success_authority["content_sha256"] != outcome["registry_payload_sha256"]
    ):
        raise ValueError("teacher rationale success evidence is not immutably bound")
    binding = _exact(
        outcome["attempt_binding"],
        {
            "campaign_packet_sha256",
            "source_profile_sha256",
            "record_id",
            "source_session_identity_sha256",
            "normalized_record_sha256",
            "normalized_trajectory_sha256",
            "transcript_sha256",
            "task_key",
            "task_version_id",
            "attempt",
            "runtime_binding_sha256",
            "teacher_source_sha256",
            "qwen_target_sha256",
            "qwen_chat_template_sha256",
            "serialization_contract_sha256",
            "serialization_evidence_sha256",
            "roundtrip_receipt_sha256",
            "compaction_sha256",
            "visible_rationale_evidence_sha256",
            "target_occurrence_manifest_sha256",
        },
        "teacher rationale success-evidence attempt binding",
    )
    for name, item in binding.items():
        if name.endswith("sha256"):
            _sha(item, f"success-evidence {name}")
    _text(binding["record_id"], "success-evidence record identity")
    _text(binding["task_key"], "success-evidence task key")
    _text(binding["task_version_id"], "success-evidence task version")
    _count(binding["attempt"], "success-evidence attempt", positive=True)
    return row


def _admission_reason(
    row: dict[str, Any],
    *,
    packet: dict[str, Any],
    profile: dict[str, Any],
    selected_tasks: Mapping[tuple[str, str], Mapping[str, Any]],
    selected_runtime_bindings: Mapping[tuple[str, str], Mapping[str, Any]],
) -> str | None:
    if (
        row["campaign_packet_sha256"] != packet["sha256"]
        or row["source_profile_sha256"] != profile["sha256"]
    ):
        return "wrong_source_binding"
    identity = (row["task_key"], row["task_version_id"])
    if identity not in selected_tasks or row["attempt"] > packet["attempts_per_task"]:
        return "unplanned_campaign_cell"
    outcome = row["outcome"]
    if (
        outcome["status"] != "completed"
        or outcome["verifier_process_success"] is not True
        or outcome["score_at_least_one"] is not True
    ):
        return "invalid_authoritative_success"
    rationale = row["visible_rationale"]
    if (
        rationale["surface"] != "ordinary_assistant_content_before_tool_call"
        or rationale["ordinary_content_only"] is not True
        or rationale["visible_to_student"] is not True
    ):
        return "missing_visible_rationale"
    if (
        rationale["provider_private_reasoning_present"] is not False
        or rationale["unknown_reasoning_fields_present"] is not False
        or any(rationale["forbidden_private_field_occurrences"].values())
    ):
        return "private_or_unknown_reasoning"
    if (
        rationale["tool_calls_with_visible_rationale"] != rationale["tool_calls"]
        or rationale["rationale_segments"] != rationale["tool_calls"]
        or rationale["minimum_sentences_per_rationale"] < 1
        or rationale["maximum_sentences_per_rationale"] > 4
        or rationale["minimum_sentences_per_rationale"]
        > rationale["maximum_sentences_per_rationale"]
    ):
        return "invalid_rationale_coverage"
    target = profile["qwen_target"]
    contract = profile["serialization"]
    if (
        row["serialization"]["schema"] != ROUNDTRIP_SCHEMA
        or row["serialization"]["qwen_target_sha256"] != digest_json(target)
        or row["serialization"]["qwen_chat_template_sha256"] != target["chat_template_sha256"]
        or row["serialization"]["message_surface"] != contract["message_surface"]
        or row["serialization"]["chat_template_kwargs"] != contract["chat_template_kwargs"]
        or row["serialization"]["prompt_token_ids_equal_local_template"] is not True
        or row["serialization"]["prompt_token_ids_exact_prefix"] is not True
        or row["serialization"]["collection_training_serving_token_ids_match"] is not True
        or row["serialization"]["round_trip_verified"] is not True
    ):
        return "serialization_mismatch"
    try:
        _compaction(
            row["compaction"],
            source_session_identity_sha256=row["source_session_identity_sha256"],
            normalized_trajectory_sha256=row["normalized_trajectory_sha256"],
            transcript_sha256=row["transcript_sha256"],
            target_occurrence_manifest_sha256=rationale["target_occurrence_manifest_sha256"],
        )
    except ValueError:
        return "opaque_or_invalid_compaction"
    expected_binding = {
        "campaign_packet_sha256": packet["sha256"],
        "source_profile_sha256": profile["sha256"],
        "record_id": row["record_id"],
        "source_session_identity_sha256": row["source_session_identity_sha256"],
        "normalized_record_sha256": row["normalized_record_sha256"],
        "normalized_trajectory_sha256": row["normalized_trajectory_sha256"],
        "transcript_sha256": row["transcript_sha256"],
        "task_key": row["task_key"],
        "task_version_id": row["task_version_id"],
        "attempt": row["attempt"],
        "runtime_binding_sha256": digest_json(selected_runtime_bindings[identity]),
        "teacher_source_sha256": digest_json(profile["source"]),
        "qwen_target_sha256": digest_json(target),
        "qwen_chat_template_sha256": target["chat_template_sha256"],
        "serialization_contract_sha256": digest_json(contract),
        "serialization_evidence_sha256": digest_json(row["serialization"]),
        "roundtrip_receipt_sha256": row["serialization"]["roundtrip_receipt_sha256"],
        "compaction_sha256": digest_json(row["compaction"]),
        "visible_rationale_evidence_sha256": digest_json(rationale),
        "target_occurrence_manifest_sha256": rationale["target_occurrence_manifest_sha256"],
    }
    if outcome["attempt_binding"] != expected_binding:
        return "success_evidence_binding_mismatch"
    return None


def _write_admission(output: Path, selection: dict[str, Any], receipt: dict[str, Any]) -> None:
    if output.exists() or output.is_symlink():
        raise FileExistsError("teacher rationale admission destination already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    os.mkdir(output, 0o700)
    atomic_write_json(output / "selection.private.json", selection, private=True)
    atomic_write_json(output / "ADMISSION.json", receipt, private=True)


def admit(config: Mapping[str, Any], *, relative_to: Path) -> dict[str, Any]:
    """Admit only metadata-backed, verifier-success visible-rationale records."""

    request = _exact(
        config,
        {
            "schema",
            "requirements",
            "source_authorization",
            "source_profile",
            "collection_packet",
            "attempts",
            "output",
        },
        "teacher rationale admission request",
    )
    if request["schema"] != ADMISSION_REQUEST_SCHEMA:
        raise ValueError("unsupported teacher rationale admission request")
    output = _path(relative_to, request["output"], "teacher rationale admission output")
    if output.exists() or output.is_symlink():
        raise FileExistsError("teacher rationale admission destination already exists")
    names = (
        "requirements",
        "source_authorization",
        "source_profile",
        "collection_packet",
        "attempts",
    )
    sources = {name: _input(relative_to, request[name], name) for name in names}
    paths = {name: pair[0] for name, pair in sources.items()}
    input_files_sha256 = {name: expected for name, (_, expected) in sources.items()}
    requirements = _requirements(
        _json(paths["requirements"], "teacher rationale requirements"),
        root=REPOSITORY_ROOT,
    )
    requirements_local_inputs = _requirements_local_inputs(requirements, root=REPOSITORY_ROOT)
    authorization = _authorization(
        _json(paths["source_authorization"], "teacher source authorization"),
        requirements,
    )
    roster = _roster(requirements, root=REPOSITORY_ROOT)
    expected = render(requirements, authorization, root=REPOSITORY_ROOT)
    profile = _profile(_json(paths["source_profile"], "teacher rationale source profile"))
    packet = _packet(_json(paths["collection_packet"], "teacher rationale packet"), profile)
    if profile != expected["source-profile.json"] or packet != expected["collection-packet.json"]:
        raise ValueError("teacher rationale profile or packet was not rendered from its authority")
    split = roster["loaded"]["family_split"]
    anchor = roster["loaded"]["role_anchor"]
    lock = roster["loaded"]["protected_family_lock"]
    selected_tasks = roster["selected"]
    selected_runtime_bindings = roster["selected_runtime_bindings"]
    if len(selected_tasks) != packet["train_task_versions"]:
        raise ValueError("teacher rationale packet train-task count drift")

    rows = [_attempt(row) for row in iter_jsonl(paths["attempts"])]
    rejected = collections.Counter({reason: 0 for reason in _REJECTIONS})
    candidates: list[dict[str, Any]] = []
    by_cell: dict[tuple[str, str, int], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        reason = _admission_reason(
            row,
            packet=packet,
            profile=profile,
            selected_tasks=selected_tasks,
            selected_runtime_bindings=selected_runtime_bindings,
        )
        if reason is not None:
            rejected[reason] += 1
            continue
        by_cell[(row["task_key"], row["task_version_id"], row["attempt"])].append(row)
    for cell_rows in by_cell.values():
        if len(cell_rows) != 1:
            rejected["ambiguous_campaign_cell"] += len(cell_rows)
        else:
            candidates.append(cell_rows[0])

    def dedupe(items: list[dict[str, Any]], field: str, reason: str) -> list[dict[str, Any]]:
        grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        for item in items:
            grouped[item[field]].append(item)
        result = []
        for group in grouped.values():
            ordered = sorted(group, key=lambda row: row["sha256"])
            result.append(ordered[0])
            rejected[reason] += len(ordered) - 1
        return result

    candidates = dedupe(candidates, "record_id", "duplicate_record")
    candidates = dedupe(candidates, "source_session_identity_sha256", "duplicate_source_session")
    candidates = dedupe(candidates, "normalized_trajectory_sha256", "duplicate_trajectory")
    by_task: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in candidates:
        by_task[(row["task_key"], row["task_version_id"])].append(row)
    selected: list[dict[str, Any]] = []
    cap = packet["maximum_sessions_per_task_version"]
    for task_rows in by_task.values():
        ordered = sorted(task_rows, key=lambda row: row["sha256"])
        selected.extend(ordered[:cap])
        rejected["per_task_cap"] += max(0, len(ordered) - cap)
    selected.sort(key=lambda row: row["sha256"])

    family_tokens: dict[str, int] = collections.defaultdict(int)
    rationale_tokens = action_tokens = compacted_sessions = 0
    private_rows = []
    for row in selected:
        assignment = selected_tasks[(row["task_key"], row["task_version_id"])]
        rationale = row["visible_rationale"]
        total = rationale["rationale_target_tokens"] + rationale["visible_action_target_tokens"]
        family_tokens[assignment["group_id"]] += total
        rationale_tokens += rationale["rationale_target_tokens"]
        action_tokens += rationale["visible_action_target_tokens"]
        compacted_sessions += row["compaction"]["kind"] != "none"
        private_rows.append(
            {
                "record_id": row["record_id"],
                "task_key": row["task_key"],
                "task_version_id": row["task_version_id"],
                "group_id": assignment["group_id"],
                "source_session_identity_sha256": row["source_session_identity_sha256"],
                "normalized_record_sha256": row["normalized_record_sha256"],
                "normalized_trajectory_sha256": row["normalized_trajectory_sha256"],
                "transcript_sha256": row["transcript_sha256"],
                "attempt": row["attempt"],
                "attempt_metadata_sha256": row["sha256"],
                "runtime_binding_sha256": digest_json(
                    selected_runtime_bindings[(row["task_key"], row["task_version_id"])]
                ),
                "target_occurrence_manifest_sha256": rationale["target_occurrence_manifest_sha256"],
                "rationale_target_tokens": rationale["rationale_target_tokens"],
                "visible_action_target_tokens": rationale["visible_action_target_tokens"],
                "outcome": row["outcome"],
                "visible_rationale": rationale,
                "serialization": row["serialization"],
                "compaction": row["compaction"],
            }
        )
    candidate_tokens = rationale_tokens + action_tokens
    maximum_family_fraction = (
        max(family_tokens.values()) / candidate_tokens if candidate_tokens else 0.0
    )
    selected_family_fraction = (
        len(family_tokens) / packet["train_task_versions"] if packet["train_task_versions"] else 0.0
    )
    candidate_occurrence_floor = (
        candidate_tokens >= packet["minimum_unique_supervised_tokens"]
        and maximum_family_fraction <= packet["maximum_family_target_token_fraction"]
        and len(family_tokens) >= packet["minimum_selected_families"]
        and selected_family_fraction >= packet["minimum_selected_family_fraction"]
    )
    roster_files_sha256 = dict(sorted(roster["files"].items()))
    admission_input_manifest = {
        "request_files_sha256": dict(sorted(input_files_sha256.items())),
        "contract_files_sha256": {
            name: expected for name, (_, expected) in sorted(requirements_local_inputs.items())
        },
        "roster_files_sha256": roster_files_sha256,
    }
    selection = {
        "schema": SELECTION_SCHEMA,
        "request_files_sha256": admission_input_manifest["request_files_sha256"],
        "contract_files_sha256": admission_input_manifest["contract_files_sha256"],
        "roster_files_sha256": roster_files_sha256,
        "admission_input_manifest_sha256": digest_json(admission_input_manifest),
        "source_authorization_sha256": authorization["sha256"],
        "source_authorization_authority": authorization["authority"],
        "source_profile_sha256": profile["sha256"],
        "collection_packet_sha256": packet["sha256"],
        "family_split_sha256": split["sha256"],
        "family_role_anchor_sha256": anchor["sha256"],
        "protected_family_lock_sha256": lock["sha256"],
        "records": private_rows,
        "sft_ready": False,
        "next_gate": "private_qwen_token_roundtrip_window_dedupe_and_exact_20m_coverage",
    }
    selection["sha256"] = digest_json(selection)
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "source_profile_sha256": profile["sha256"],
        "collection_packet_sha256": packet["sha256"],
        "selected_records": len(private_rows),
        "selected_task_versions": len(
            {(row["task_key"], row["task_version_id"]) for row in private_rows}
        ),
        "selected_families": len(family_tokens),
        "candidate_rationale_target_tokens": rationale_tokens,
        "candidate_visible_action_target_tokens": action_tokens,
        "candidate_supervised_token_occurrences": candidate_tokens,
        "minimum_unique_supervised_tokens": packet["minimum_unique_supervised_tokens"],
        "observed_maximum_family_target_token_fraction": maximum_family_fraction,
        "selected_family_fraction": selected_family_fraction,
        "minimum_selected_families": packet["minimum_selected_families"],
        "minimum_selected_family_fraction": packet["minimum_selected_family_fraction"],
        "candidate_occurrence_floor_reached": candidate_occurrence_floor,
        "uncompacted_sessions": len(private_rows) - compacted_sessions,
        "exact_visible_summary_sessions": compacted_sessions,
        "opaque_compaction_sessions": 0,
        "heldout_families_admitted": 0,
        "rejections": {reason: rejected[reason] for reason in _REJECTIONS},
        "source_text_read": False,
        "token_ids_read": False,
        "sft_ready": False,
        "next_gate": "private_qwen_token_roundtrip_window_dedupe_and_exact_20m_coverage",
        "selection_sha256": selection["sha256"],
        "admission_input_manifest_sha256": selection["admission_input_manifest_sha256"],
    }
    _public_only(receipt)
    receipt["sha256"] = digest_json(receipt)
    for name, (path, expected) in sources.items():
        if file_sha256(path) != expected:
            raise ValueError(f"{name} changed during teacher rationale admission")
    _require_unchanged(requirements_local_inputs, "teacher rationale contract input")
    _require_unchanged(
        {name: (roster["paths"][name], expected) for name, expected in roster["files"].items()},
        "teacher rationale roster input",
    )
    _write_admission(output, selection, receipt)
    return {
        "submitted": False,
        "selected_records": len(private_rows),
        "candidate_supervised_token_occurrences": candidate_tokens,
        "candidate_occurrence_floor_reached": candidate_occurrence_floor,
        "sft_ready": False,
        "receipt_sha256": receipt["sha256"],
        "selection_sha256": selection["sha256"],
    }
