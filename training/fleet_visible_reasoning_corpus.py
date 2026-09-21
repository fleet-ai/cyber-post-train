"""Private Qwen/OpenCode materialization for explicitly visible reasoning.

This module is intentionally *not* an option on ``fleet_collection_corpus``.
The action-only corpus rejects all assistant prose because it cannot establish
whether that prose is visible reasoning or private provider state.  This
separate contract accepts only a narrowly described source profile, a sealed
train-family selection, and records whose token boundaries are re-rendered by
the exact local Qwen tokenizer and chat template.

It is an offline CPU-only builder.  It does not contact Fleet, launch a
workload, print records, or write source text to its output.  The only corpus
payload is private token IDs and loss masks. It never loads a caller-supplied
renderer: a reflection adapter could simply repeat stored token IDs. Instead it
uses the exact locally pinned tokenizer and fails closed when the recorded
OpenCode/Qwen boundary cannot be reproduced.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import collection_campaign, dense, visible_reasoning_census
from . import corpus as dense_corpus
from . import fleet_collection_admission as admission
from .io import atomic_write_json, atomic_write_jsonl, digest_json, file_sha256, iter_jsonl
from .sft import _known, read_mapping
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
ARM_MANIFEST_SCHEMA = "cyber_qwen_opencode_visible_reasoning_sft_arm_manifest_v1"
COVERAGE_SCHEMA = "cyber_qwen_opencode_visible_reasoning_coverage_v1"
RECEIPT_SCHEMA = "cyber_qwen_opencode_visible_reasoning_materialization_receipt_v1"
ROUNDTRIP_SCHEMA = "cyber_qwen_opencode_template_roundtrip_v1"
SUCCESS_EVIDENCE_SCHEMA = "cyber_qwen_opencode_student_visible_reasoning_success_evidence_v1"
SOURCE_AUTHORIZATION_SCHEMA = (
    "cyber_qwen_opencode_student_visible_reasoning_source_authorization_v1"
)
OPERATION_AUTHORIZATION_SCHEMA = (
    "cyber_qwen_opencode_student_visible_reasoning_operation_authorization_v1"
)
CAMPAIGN_PLAN_SCHEMA = "cyber_qwen_opencode_visible_reasoning_campaign_plan_v1"
WAVE_PLAN_SCHEMA = "cyber_qwen_opencode_visible_reasoning_wave_plan_v1"

QWEN_REPOSITORY = "Qwen/Qwen3.8-27B"
OPENCODE_HARNESS = "opencode"
OPENCODE_VERSION = "1.18.27"
ONLINE_COMPACTION = "opencode_1.18.27_native_compaction_autocontinue_v2"
EXACT_COMPACTION = "student_generated_exact_continuation_v1"
MINIMUM_SUPERVISED_TOKENS = 20_000_000
MAXIMUM_FAMILY_TOKEN_FRACTION = 0.25
MINIMUM_SUCCESSFUL_FAMILIES = 20
CAMPAIGN_ATTEMPTS_PER_TASK = 400
CAMPAIGN_ATTEMPTS_PER_WAVE = 10
CAMPAIGN_WAVES = CAMPAIGN_ATTEMPTS_PER_TASK // CAMPAIGN_ATTEMPTS_PER_WAVE
CAMPAIGN_BASE_SEED = 43
CELL_ID_PREFIX = "qvrc-"

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


def _immutable_authority(value: object, label: str) -> dict[str, Any]:
    """Validate an immutable Registry locator without contacting the Registry.

    The private builder is deliberately offline.  The collection authority must
    therefore publish the canonical payload under this exact locator before a
    caller materializes it locally.  We bind the advertised registry content
    digest to a deterministic payload digest below; a bare, syntactically valid
    locator is not evidence by itself.
    """

    authority = _exact(
        value,
        {"kind", "artifact_key", "version_index", "content_sha256"},
        label,
    )
    if (
        authority["kind"] != "fleet_artifact_registry_immutable_v1"
        or not isinstance(authority["artifact_key"], str)
        or not authority["artifact_key"].startswith("cyber/runs/")
        or type(authority["version_index"]) is not int
        or authority["version_index"] < 1
    ):
        raise ValueError(f"{label} lacks an immutable Fleet artifact binding")
    _sha(authority["content_sha256"], f"{label} content")
    return authority


def _registry_payload_sha256(value: Mapping[str, Any]) -> str:
    """Digest the exact registry payload, excluding only the locator wrapper."""

    return digest_json(
        {
            name: item
            for name, item in value.items()
            if name not in {"authority", "registry_payload_sha256", "sha256"}
        }
    )


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
        "authorization_sha256",
    }
    if kind != "qwen_self" or set(source) != base:
        raise ValueError("v1 accepts only the explicit Qwen-self visible-reasoning source")
    _string(source["model_alias"], "source model alias")
    source_model = _model(source["model"], "source model")
    for name in base - {"kind", "model_alias", "model"}:
        _sha(source[name], f"source {name}")

    target = _exact(
        profile["qwen_target"],
        {
            "repository",
            "revision",
            "tokenizer_sha256",
            "tokenizer_backend_sha256",
            "chat_template_sha256",
        },
        "Qwen target",
    )
    if (
        target["repository"] != QWEN_REPOSITORY
        or not isinstance(target["revision"], str)
        or not re.fullmatch(r"[0-9a-f]{40}", target["revision"])
    ):
        raise ValueError("visible-reasoning target must pin Qwen3.8-27B exactly")
    for name in ("tokenizer_sha256", "tokenizer_backend_sha256", "chat_template_sha256"):
        _sha(target[name], f"Qwen target {name}")
    if source_model != {
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
    _sha(serialization["roundtrip_fixture_sha256"], "serialization round-trip fixture")

    compaction = _exact(
        profile["compaction"], {"accepted_kind", "opaque_compaction_rejected"}, "compaction"
    )
    if compaction != {
        "accepted_kind": EXACT_COMPACTION,
        "opaque_compaction_rejected": True,
    }:
        raise ValueError("source compaction policy is insufficient")
    return profile


def _source_authorization(value: Mapping[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    """Require explicit model-visible Qwen reasoning authorization.

    This is intentionally a separate immutable artifact, not a boolean in the
    source profile.  It makes generic teacher/provider ``thinking`` fields
    unable to reach the Qwen-self lane by merely changing a digest string.
    """

    authorization = _sealed(
        value,
        SOURCE_AUTHORIZATION_SCHEMA,
        "visible-reasoning source authorization",
    )
    _exact(
        authorization,
        {
            "schema",
            "authority",
            "registry_payload_sha256",
            "source",
            "qwen_target",
            "opencode",
            "thinking",
            "reasoning_visibility",
            "private_or_unknown_reasoning",
            "purpose",
            "sha256",
        },
        "visible-reasoning source authorization",
    )
    authority = _immutable_authority(
        authorization["authority"], "source-authorization immutable authority"
    )
    payload_sha256 = _sha(
        authorization["registry_payload_sha256"], "source-authorization registry payload"
    )
    if (
        payload_sha256 != _registry_payload_sha256(authorization)
        or authority["content_sha256"] != payload_sha256
    ):
        raise ValueError("source authorization does not bind its immutable Registry payload")
    source = _exact(
        authorization["source"],
        {"kind", "model_alias", "model"},
        "authorized visible-reasoning source",
    )
    if source != {
        "kind": profile["source"]["kind"],
        "model_alias": profile["source"]["model_alias"],
        "model": profile["source"]["model"],
    }:
        raise ValueError("source authorization changes the Qwen-self source")
    if (
        authorization["qwen_target"] != profile["qwen_target"]
        or authorization["opencode"] != profile["opencode"]
        or authorization["thinking"] != profile["thinking"]
        or authorization["reasoning_visibility"] != "student_visible"
        or authorization["private_or_unknown_reasoning"] != "reject"
        or authorization["purpose"] != "student_visible_reasoning_plus_visible_actions"
    ):
        raise ValueError(
            "source authorization does not prove the qualified visible-reasoning treatment"
        )
    if profile["source"]["authorization_sha256"] != authorization["sha256"]:
        raise ValueError("source profile does not bind the source authorization receipt")
    return authorization


def _template_ids(tokenizer: Any, messages: list[dict[str, Any]], *, generation: bool) -> list[int]:
    """Render only through the pinned local Qwen chat template."""

    normalized = []
    for message in messages:
        rendered_message = dense._normalized_for_template(message)
        # Raw provider fields named ``reasoning_content`` remain forbidden by
        # ``_message``.  Only the separately authorized, model-visible field
        # below may enter Qwen's exact reasoning slot, and it is translated
        # locally only after the private record has passed that schema gate.
        if "student_visible_reasoning" in message:
            rendered_message["reasoning_content"] = message["student_visible_reasoning"]
        normalized.append(rendered_message)
    try:
        rendered = tokenizer.apply_chat_template(
            normalized,
            tokenize=True,
            add_generation_prompt=generation,
            tools=[],
            enable_thinking=True,
        )
    except Exception as error:
        raise ValueError("pinned Qwen chat-template rendering failed") from error
    if isinstance(rendered, Mapping):
        rendered = rendered.get("input_ids")
    if isinstance(rendered, list) and len(rendered) == 1 and isinstance(rendered[0], list):
        rendered = rendered[0]
    return _token_ids(rendered, "pinned Qwen chat-template token IDs")


def _roundtrip(
    value: Mapping[str, Any],
    profile: dict[str, Any],
    tokenizer: Any,
    tokenizer_identity: dict[str, Any],
) -> dict[str, Any]:
    """Re-render non-task fixtures through the exact local Qwen template."""
    fixture = _sealed(value, ROUNDTRIP_SCHEMA, "Qwen/OpenCode round-trip fixture")
    _exact(
        fixture,
        {
            "schema",
            "qwen_target",
            "cases",
            "sha256",
        },
        "Qwen/OpenCode round-trip fixture",
    )
    target = _exact(
        fixture["qwen_target"],
        {
            "repository",
            "revision",
            "tokenizer_sha256",
            "tokenizer_backend_sha256",
            "chat_template_sha256",
        },
        "round-trip Qwen target",
    )
    if target != profile["qwen_target"]:
        raise ValueError("round-trip fixture changes the exact Qwen target")
    if tokenizer_identity != {
        "repository": profile["qwen_target"]["repository"],
        "revision": profile["qwen_target"]["revision"],
        "tokenizer_sha256": profile["qwen_target"]["tokenizer_sha256"],
        "tokenizer_backend_sha256": profile["qwen_target"]["tokenizer_backend_sha256"],
        "chat_template_sha256": profile["qwen_target"]["chat_template_sha256"],
    }:
        raise ValueError("local Qwen tokenizer identity differs from the source profile")
    cases = fixture["cases"]
    if not isinstance(cases, list) or not cases:
        raise ValueError("round-trip fixture requires synthetic cases")
    seen: set[str] = set()
    for case in cases:
        case = _exact(
            case,
            {
                "case_id",
                "messages",
                "target_message_index",
                "prompt_token_ids",
                "collection_token_ids",
                "training_token_ids",
                "serving_token_ids",
            },
            "round-trip case",
        )
        case_id = _string(case["case_id"], "round-trip case identity")
        if case_id in seen:
            raise ValueError("round-trip fixture duplicates a synthetic case")
        seen.add(case_id)
        raw_messages = case["messages"]
        if not isinstance(raw_messages, list) or not raw_messages:
            raise ValueError("round-trip case requires synthetic messages")
        messages = [_message(message) for message in raw_messages]
        target_index = _count(case["target_message_index"], "round-trip target message")
        if target_index != len(messages) - 1 or messages[target_index]["role"] != "assistant":
            raise ValueError("round-trip case must end at one assistant target")
        prompt = _token_ids(case["prompt_token_ids"], "round-trip prompt token IDs")
        collection = _token_ids(case["collection_token_ids"], "round-trip collection token IDs")
        training = _token_ids(case["training_token_ids"], "round-trip training token IDs")
        serving = _token_ids(case["serving_token_ids"], "round-trip serving token IDs")
        actual_prompt = _template_ids(tokenizer, messages[:-1], generation=True)
        actual = _template_ids(tokenizer, messages, generation=False)
        if (
            prompt != actual_prompt
            or actual[: len(prompt)] != prompt
            or collection != training
            or collection != serving
            or collection != actual
        ):
            raise ValueError("round-trip fixture does not prove exact token serialization")
    return fixture


def _load_tokenizer(
    model_lock: Mapping[str, Any], tokenizer_root: Path, profile: dict[str, Any]
) -> tuple[Any, dict[str, str]]:
    """Load only the exact local Qwen tokenizer bytes named by the profile."""
    target = profile["qwen_target"]
    lock = _mapping(model_lock, "Qwen model lock")
    tokenizer = _mapping(lock.get("tokenizer"), "Qwen model-lock tokenizer")
    identity_lock = {
        "repository": lock.get("repo"),
        "revision": lock.get("revision"),
        "tokenizer_sha256": tokenizer.get("manifest_sha256"),
    }
    if identity_lock != {
        "repository": target["repository"],
        "revision": target["revision"],
        "tokenizer_sha256": target["tokenizer_sha256"],
    }:
        raise ValueError("model lock differs from the exact Qwen source profile")
    loaded, raw_identity = dense_corpus.local_tokenizer(lock, tokenizer_root)
    identity = {
        "repository": raw_identity.get("repo"),
        "revision": raw_identity.get("revision"),
        "tokenizer_sha256": tokenizer["manifest_sha256"],
        "tokenizer_backend_sha256": "sha256:"
        + _string(raw_identity.get("backend_sha256"), "local tokenizer backend digest"),
        "chat_template_sha256": "sha256:"
        + _string(raw_identity.get("chat_template_sha256"), "local chat-template digest"),
    }
    if identity != target:
        raise ValueError("local tokenizer bytes differ from the Qwen source profile")
    return loaded, identity


def _packet(value: Mapping[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    packet = _sealed(value, PACKET_SCHEMA, "visible-reasoning collection packet")
    _exact(
        packet,
        {
            "schema",
            "source_profile_sha256",
            "source_authorization_sha256",
            "campaign_name",
            "campaign_plan_sha256",
            "wave_plan_sha256",
            "cell_identity_universe_sha256",
            "operation_authorization_sha256",
            "task_selection_sha256",
            "attempts_per_task_version",
            "base_seed",
            "catalog_inventory_sha256",
            "family_split_sha256",
            "root_role_anchor_id",
            "family_role_anchor_sha256",
            "protected_family_lock_sha256",
            "runtime_bindings_sha256",
            "training_data_eligible",
            "objective",
            "minimum_unique_supervised_tokens",
            "minimum_successful_families",
            "maximum_family_target_token_fraction",
            "matched_action_only_required",
            "deduplication_order",
            "rejection_policy",
            "sha256",
        },
        "visible-reasoning collection packet",
    )
    for name in (
        "source_profile_sha256",
        "source_authorization_sha256",
        "campaign_plan_sha256",
        "wave_plan_sha256",
        "cell_identity_universe_sha256",
        "operation_authorization_sha256",
        "task_selection_sha256",
        "catalog_inventory_sha256",
        "family_split_sha256",
        "family_role_anchor_sha256",
        "protected_family_lock_sha256",
        "runtime_bindings_sha256",
    ):
        _sha(packet[name], f"collection packet {name}")
    if (
        packet["source_profile_sha256"] != profile["sha256"]
        or packet["source_authorization_sha256"] != profile["source"]["authorization_sha256"]
        or not isinstance(packet["campaign_name"], str)
        or not packet["campaign_name"]
        or packet["attempts_per_task_version"] != CAMPAIGN_ATTEMPTS_PER_TASK
        or packet["base_seed"] != CAMPAIGN_BASE_SEED
        or packet["root_role_anchor_id"] != TRUSTED_FLEET_COLLECTION_ROOT_ID
        or packet["training_data_eligible"] is not True
        or packet["objective"] != "student_visible_reasoning_plus_visible_actions"
        or packet["minimum_unique_supervised_tokens"] < MINIMUM_SUPERVISED_TOKENS
        or packet["minimum_successful_families"] != MINIMUM_SUCCESSFUL_FAMILIES
        or packet["maximum_family_target_token_fraction"] != MAXIMUM_FAMILY_TOKEN_FRACTION
        or packet["matched_action_only_required"] is not True
        or packet["deduplication_order"]
        != [
            "source_session_identity",
            "normalized_trajectory_digest",
            "source_target_digest",
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


def _planned_cell_id(packet: Mapping[str, Any], row: Mapping[str, Any]) -> str:
    attempt = _count(row.get("attempt"), "campaign attempt", positive=True)
    seed = _count(row.get("seed"), "campaign seed")
    wave = _count(row.get("wave"), "campaign wave", positive=True)
    if (
        attempt > packet["attempts_per_task_version"]
        or seed != packet["base_seed"] + attempt - 1
        or wave != (attempt - 1) // CAMPAIGN_ATTEMPTS_PER_WAVE + 1
        or wave > CAMPAIGN_WAVES
    ):
        raise ValueError("record is outside the frozen campaign attempt and seed universe")
    payload = {
        "campaign_name": packet["campaign_name"],
        "task_key": _string(row.get("task_key"), "campaign task key"),
        "task_version_id": _string(row.get("task_version_id"), "campaign task version"),
        "group_id": _string(row.get("group_id"), "campaign task family"),
        "attempt": attempt,
        "seed": seed,
    }
    return CELL_ID_PREFIX + digest_json(payload).removeprefix("sha256:")[:24]


def _operation_authorization(value: Mapping[str, Any], packet: Mapping[str, Any]) -> dict[str, Any]:
    authorization = _sealed(
        value,
        OPERATION_AUTHORIZATION_SCHEMA,
        "visible-reasoning operation authorization",
    )
    _exact(
        authorization,
        {
            "schema",
            "authority",
            "registry_payload_sha256",
            "campaign_name",
            "campaign_plan_sha256",
            "wave_plan_sha256",
            "cell_identity_universe_sha256",
            "task_selection_sha256",
            "operation_root_name",
            "dedicated_ledger_id",
            "policy",
            "sha256",
        },
        "visible-reasoning operation authorization",
    )
    authority = _immutable_authority(
        authorization["authority"], "operation-authorization immutable authority"
    )
    payload_sha256 = _sha(
        authorization["registry_payload_sha256"], "operation-authorization Registry payload"
    )
    if (
        payload_sha256 != _registry_payload_sha256(authorization)
        or authority["content_sha256"] != payload_sha256
        or packet["operation_authorization_sha256"] != authorization["sha256"]
    ):
        raise ValueError("operation authorization is not the packet-bound immutable artifact")
    for name in (
        "campaign_plan_sha256",
        "wave_plan_sha256",
        "cell_identity_universe_sha256",
        "task_selection_sha256",
    ):
        if authorization[name] != packet[name]:
            raise ValueError("operation authorization changes the frozen campaign identity")
    if authorization["campaign_name"] != packet["campaign_name"]:
        raise ValueError("operation authorization changes the campaign name")
    _string(authorization["operation_root_name"], "operation root")
    _sha(authorization["dedicated_ledger_id"], "dedicated ledger identity")
    if authorization["policy"] != {
        "canonical_private_operation_root_required": True,
        "dedicated_empty_ledger_required": True,
        "exclusive_pre_mutation_intent_required": True,
        "ambiguous_external_mutation_replay_allowed": False,
        "same_cell_retry_allowed": False,
    }:
        raise ValueError("operation authorization weakens create-once collection")
    return authorization


def _campaign_artifacts(
    plan_value: Mapping[str, Any],
    wave_value: Mapping[str, Any],
    packet: Mapping[str, Any],
    task_selection: Mapping[str, Any],
    split: Mapping[str, Any],
    profile: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    plan = _sealed(plan_value, CAMPAIGN_PLAN_SCHEMA, "visible-reasoning campaign plan")
    wave = _sealed(wave_value, WAVE_PLAN_SCHEMA, "visible-reasoning wave plan")
    identity = _mapping(plan.get("identity"), "campaign cell identity")
    collection = _mapping(plan.get("collection"), "campaign collection gate")
    boundary = _mapping(plan.get("task_boundary"), "campaign task boundary")
    source = _mapping(plan.get("source_treatment"), "campaign source treatment")
    if (
        packet["campaign_plan_sha256"] != plan["sha256"]
        or packet["wave_plan_sha256"] != wave["sha256"]
        or wave.get("campaign_plan_sha256") != plan["sha256"]
        or plan.get("campaign_name") != packet["campaign_name"]
        or identity.get("cell_identity_universe_sha256") != packet["cell_identity_universe_sha256"]
        or boundary.get("task_selection_sha256") != packet["task_selection_sha256"]
        or collection.get("attempts_per_task_version") != packet["attempts_per_task_version"]
        or collection.get("minimum_unique_supervised_tokens")
        != packet["minimum_unique_supervised_tokens"]
        or collection.get("minimum_successful_families") != packet["minimum_successful_families"]
        or collection.get("maximum_family_target_token_fraction")
        != packet["maximum_family_target_token_fraction"]
    ):
        raise ValueError("collection packet changes the frozen campaign or cell universe")
    campaign_tokenizer = _mapping(source.get("tokenizer"), "campaign tokenizer identity")
    campaign_opencode = _mapping(source.get("opencode"), "campaign OpenCode treatment")
    campaign_thinking = _mapping(source.get("thinking"), "campaign thinking treatment")
    profile_opencode = _mapping(profile.get("opencode"), "source-profile OpenCode treatment")
    if (
        source.get("kind") != profile["source"]["kind"]
        or source.get("model") != profile["source"]["model"]
        or campaign_tokenizer
        != {
            "manifest_sha256": profile["qwen_target"]["tokenizer_sha256"],
            "backend_sha256": profile["qwen_target"]["tokenizer_backend_sha256"],
            "chat_template_sha256": profile["qwen_target"]["chat_template_sha256"],
        }
        or any(
            campaign_opencode.get(name) != profile_opencode.get(name)
            for name in (
                "harness",
                "harness_version",
                "release_asset_sha256",
                "tool_catalog_sha256",
                "context_management",
                "context_window_tokens",
                "context_headroom_tokens",
                "tools",
            )
        )
        or campaign_thinking
        != {
            **profile["thinking"],
            "reasoning_visibility": "student_visible",
        }
    ):
        raise ValueError("campaign plan changes the authorized Qwen/OpenCode source treatment")
    role_by_identity = {
        (row.get("task_key"), row.get("task_version_id")): row
        for row in split.get("tasks", [])
        if isinstance(row, Mapping)
    }
    selected_rows = task_selection.get("tasks")
    if not isinstance(selected_rows, list) or not selected_rows:
        raise ValueError("campaign task selection is empty")
    tasks: list[dict[str, str]] = []
    for row in selected_rows:
        checked = _mapping(row, "campaign task selection row")
        key = (
            _string(checked.get("task_key"), "campaign task key"),
            _string(checked.get("task_version_id"), "campaign task version"),
        )
        role = role_by_identity.get(key)
        if role is None or role.get("split") != "train":
            raise ValueError("campaign cell universe includes a non-training task")
        tasks.append(
            {
                "task_key": key[0],
                "task_version_id": key[1],
                "group_id": _string(role.get("group_id"), "campaign task family"),
            }
        )
    tasks.sort(key=lambda row: (row["task_key"], row["task_version_id"]))
    expected_opencode = {
        **profile_opencode,
        "provider_adapter": "@ai-sdk/openai-compatible",
        "max_output_tokens": 32_768,
        "max_model_requests": 600,
        "timeout_seconds": 28_800,
    }
    expected_route = {
        "model": "qwen3.8-27b-base",
        "served_id": "qwen3.8-27b",
        "task_selection_sha256": packet["task_selection_sha256"],
        "runtime_bindings_sha256": packet["runtime_bindings_sha256"],
        "task_versions_sha256": digest_json([row["task_version_id"] for row in tasks]),
        "task_version_count": len(tasks),
        "catalog": {
            "engine": "sglang",
            "precision": "bf16",
            "tensor_parallel_size": 1,
        },
        "model_info": {
            "model_path": f"/scratch/models/qwen3.8-27b/{profile['qwen_target']['revision']}",
            "model_type": "qwen3_5",
            "architectures": ["Qwen3_5ForConditionalGeneration"],
        },
        "server_info": {
            "model_path": f"/scratch/models/qwen3.8-27b/{profile['qwen_target']['revision']}",
            "context_length": 262_144,
            "tp_size": 1,
            "dp_size": 8,
            "load_balance_method": "total_tokens",
            "quantization": None,
            "kv_cache_dtype": "fp8_e4m3",
            "reasoning_parser": "qwen3",
            "tool_call_parser": "qwen3_coder",
        },
        "endpoint_origin": "https://inference.flt.build",
    }
    if (
        campaign_opencode != expected_opencode
        or source.get("route") != expected_route
        or source.get("images")
        != {
            "agent": "sha256:c7d048c98e6b8e52e5b76ab4006a7626b1ccf63a37bfa4b47ecd0fe9028e1f92",
            "proxy": (
                "ghcr.io/astral-sh/uv:python3.12-bookworm@"
                "sha256:9aa60c50016c0485636ab9a830246a6ef3399aa4a8bab3d17ef4a2358fba2ca7"
            ),
        }
        or source.get("sampling")
        != {
            "temperature": 0.6,
            "top_p": 0.95,
            "base_seed": packet["base_seed"],
            "seed_rule": "base_seed_plus_attempt_minus_one",
        }
    ):
        raise ValueError("campaign source is not the exact train-family Qwen/OpenCode route")
    universe: list[dict[str, Any]] = []
    expected_waves: list[dict[str, Any]] = []
    for wave_index in range(CAMPAIGN_WAVES):
        first = wave_index * CAMPAIGN_ATTEMPTS_PER_WAVE + 1
        last = first + CAMPAIGN_ATTEMPTS_PER_WAVE - 1
        cells: list[dict[str, Any]] = []
        for task in tasks:
            for attempt in range(first, last + 1):
                cell = {
                    "campaign_name": packet["campaign_name"],
                    **task,
                    "attempt": attempt,
                    "seed": packet["base_seed"] + attempt - 1,
                }
                cell["cell_id"] = CELL_ID_PREFIX + digest_json(cell).removeprefix("sha256:")[:24]
                cells.append(cell)
        universe.extend(cells)
        expected_waves.append(
            {
                "wave": wave_index + 1,
                "attempt_first": first,
                "attempt_last": last,
                "seed_first": packet["base_seed"] + first - 1,
                "seed_last": packet["base_seed"] + last - 1,
                "task_versions": len(tasks),
                "planned_cells": len(cells),
                "cell_intents_sha256": digest_json(cells),
            }
        )
    if (
        wave.get("waves") != expected_waves
        or identity.get("planned_unique_cell_ids") != len(universe)
        or identity.get("cell_identity_universe_sha256") != digest_json(universe)
        or len({row["cell_id"] for row in universe}) != len(universe)
    ):
        raise ValueError("wave plan does not cover the exact campaign cell universe")
    return plan, wave


def _task_boundary(
    inventory: dict[str, Any],
    split: dict[str, Any],
    role_anchor: dict[str, Any],
    lock: dict[str, Any],
    runtime: dict[str, Any],
    task_selection: dict[str, Any],
    packet: dict[str, Any],
) -> dict[tuple[str, str], dict[str, Any]]:
    rows = collection_campaign._inventory_rows(inventory)
    bindings = collection_campaign._runtime_bindings(runtime, inventory, rows)
    collection_campaign._sealed(task_selection, collection_campaign.SELECTION_SCHEMA)  # noqa: SLF001
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
        or packet["task_selection_sha256"] != task_selection["sha256"]
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
    selected_rows = task_selection.get("tasks")
    if not isinstance(selected_rows, list) or not selected_rows:
        raise ValueError("campaign task selection is empty")
    selected = {(row.get("task_key"), row.get("task_version_id")) for row in selected_rows}
    expected_train = {identity for identity, row in role_rows.items() if row["split"] == "train"}
    selected_groups = {
        role_rows[identity]["group_id"] for identity in selected if identity in role_rows
    }
    if (
        len(selected) != len(selected_rows)
        or selected != expected_train
        or len(selected_groups) != len(selected)
    ):
        raise ValueError("campaign task selection is not the exact train-family roster")
    return bindings


def _success_evidence(
    value: Mapping[str, Any],
    packet: dict[str, Any],
    split: dict[str, Any],
    lock: dict[str, Any],
    source_authorization: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Validate the immutable success-to-task mapping before record content is read.

    This file is expected to be materialized from one immutable Fleet Artifact
    Registry version by the collection authority.  The offline builder cannot
    contact that authority, so it verifies the locator/version/content binding
    and requires an exact mapping for every selected private record.
    """

    evidence = _sealed(value, SUCCESS_EVIDENCE_SCHEMA, "visible-reasoning success evidence")
    _exact(
        evidence,
        {
            "schema",
            "authority",
            "registry_payload_sha256",
            "source_profile_sha256",
            "source_authorization_sha256",
            "collection_packet_sha256",
            "campaign_plan_sha256",
            "wave_plan_sha256",
            "cell_identity_universe_sha256",
            "operation_authorization_sha256",
            "task_selection_sha256",
            "catalog_inventory_sha256",
            "family_split_sha256",
            "root_role_anchor_id",
            "family_role_anchor_sha256",
            "protected_family_lock_sha256",
            "records",
            "sha256",
        },
        "visible-reasoning success evidence",
    )
    authority = _immutable_authority(evidence["authority"], "success-evidence immutable authority")
    payload_sha256 = _sha(evidence["registry_payload_sha256"], "success-evidence registry payload")
    if (
        payload_sha256 != _registry_payload_sha256(evidence)
        or authority["content_sha256"] != payload_sha256
    ):
        raise ValueError("success evidence does not bind its immutable Registry payload")
    expected = {
        "collection_packet_sha256": packet["sha256"],
        "campaign_plan_sha256": packet["campaign_plan_sha256"],
        "wave_plan_sha256": packet["wave_plan_sha256"],
        "cell_identity_universe_sha256": packet["cell_identity_universe_sha256"],
        "operation_authorization_sha256": packet["operation_authorization_sha256"],
        "task_selection_sha256": packet["task_selection_sha256"],
        "catalog_inventory_sha256": packet["catalog_inventory_sha256"],
        "family_split_sha256": split["sha256"],
        "root_role_anchor_id": TRUSTED_FLEET_COLLECTION_ROOT_ID,
        "family_role_anchor_sha256": packet["family_role_anchor_sha256"],
        "protected_family_lock_sha256": lock["sha256"],
    }
    for name, target in expected.items():
        checked_target = (
            _string(target, f"success evidence {name}")
            if name == "root_role_anchor_id"
            else _sha(target, f"success evidence {name}")
        )
        if evidence.get(name) != checked_target:
            raise ValueError("success evidence changes an immutable task binding")
    _sha(evidence["source_profile_sha256"], "success evidence source profile")
    if evidence["source_authorization_sha256"] != source_authorization["sha256"]:
        raise ValueError("success evidence changes the source authorization receipt")
    rows = evidence["records"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("success evidence requires at least one immutable record mapping")
    fields = {
        "record_id",
        "cell_id",
        "wave",
        "attempt",
        "seed",
        "source_session_identity_sha256",
        "normalized_record_sha256",
        "normalized_trajectory_sha256",
        "transcript_sha256",
        "task_key",
        "task_version_id",
        "group_id",
        "verifier_execution_sha256",
        "outcome",
    }
    result: dict[str, dict[str, Any]] = {}
    sessions: set[str] = set()
    cells: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != fields:
            raise ValueError("success evidence record has an unsupported contract")
        record_id = _string(row["record_id"], "success-evidence record identity")
        if record_id in result:
            raise ValueError("success evidence duplicates a record identity")
        for name in (
            "source_session_identity_sha256",
            "normalized_record_sha256",
            "normalized_trajectory_sha256",
            "transcript_sha256",
            "verifier_execution_sha256",
        ):
            _sha(row[name], f"success evidence {name}")
        if row["source_session_identity_sha256"] in sessions:
            raise ValueError("success evidence duplicates a source session")
        sessions.add(row["source_session_identity_sha256"])
        if row["outcome"] != "verified_success":
            raise ValueError("visible-reasoning evidence must be an authoritative success")
        for name in ("task_key", "task_version_id", "group_id"):
            _string(row[name], f"success evidence {name}")
        if row["cell_id"] != _planned_cell_id(packet, row):
            raise ValueError("success evidence is outside the exact campaign cell universe")
        if row["cell_id"] in cells:
            raise ValueError("success evidence duplicates a create-once campaign cell")
        cells.add(row["cell_id"])
        result[record_id] = row
    return evidence, result


def _selection(
    value: Mapping[str, Any],
    packet: dict[str, Any],
    split: dict[str, Any],
    lock: dict[str, Any],
    bindings: dict[tuple[str, str], dict[str, Any]],
    success_evidence: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    selection = _sealed(value, SELECTION_SCHEMA, "visible-reasoning private selection")
    _exact(
        selection,
        {
            "schema",
            "collection_packet_sha256",
            "verified_success_evidence_sha256",
            "campaign_plan_sha256",
            "wave_plan_sha256",
            "cell_identity_universe_sha256",
            "operation_authorization_sha256",
            "task_selection_sha256",
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
        "campaign_plan_sha256",
        "wave_plan_sha256",
        "cell_identity_universe_sha256",
        "operation_authorization_sha256",
        "task_selection_sha256",
        "catalog_inventory_sha256",
        "family_split_sha256",
        "family_role_anchor_sha256",
        "protected_family_lock_sha256",
    ):
        _sha(selection[name], f"selection {name}")
    if (
        selection["collection_packet_sha256"] != packet["sha256"]
        or selection["campaign_plan_sha256"] != packet["campaign_plan_sha256"]
        or selection["wave_plan_sha256"] != packet["wave_plan_sha256"]
        or selection["cell_identity_universe_sha256"] != packet["cell_identity_universe_sha256"]
        or selection["operation_authorization_sha256"] != packet["operation_authorization_sha256"]
        or selection["task_selection_sha256"] != packet["task_selection_sha256"]
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
        "cell_id",
        "wave",
        "source_session_identity_sha256",
        "normalized_record_sha256",
        "normalized_trajectory_sha256",
        "transcript_sha256",
        "task_key",
        "task_version_id",
        "group_id",
        "attempt",
        "seed",
        "reasoning_visibility",
        "compaction_kind",
    }
    rows = selection["selected"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("private selection must contain visible-reasoning records")
    result: dict[str, dict[str, Any]] = {}
    sessions: set[str] = set()
    trajectories: set[str] = set()
    cells: set[str] = set()
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
        if row["cell_id"] != _planned_cell_id(packet, row):
            raise ValueError("selected record is outside the exact campaign cell universe")
        if row["cell_id"] in cells:
            raise ValueError("private selection duplicates a create-once campaign cell")
        cells.add(row["cell_id"])
        if row["reasoning_visibility"] != "student_visible":
            raise ValueError("private or unknown reasoning cannot enter selection")
        if row["compaction_kind"] not in {"none", EXACT_COMPACTION}:
            raise ValueError("opaque compaction cannot enter selection")
        evidence = success_evidence.get(record_id)
        if evidence is None or any(
            row[name] != evidence[name]
            for name in (
                "cell_id",
                "wave",
                "attempt",
                "seed",
                "source_session_identity_sha256",
                "normalized_record_sha256",
                "normalized_trajectory_sha256",
                "transcript_sha256",
                "task_key",
                "task_version_id",
                "group_id",
            )
        ):
            raise ValueError("private selection is not covered by immutable success evidence")
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
        allowed = {"role", "content"}
        if "student_visible_reasoning" in message:
            allowed.add("student_visible_reasoning")
        if "tool_calls" in message:
            allowed.add("tool_calls")
        if set(message) != allowed:
            raise ValueError("assistant message has an unsupported field")
        reasoning = message.get("student_visible_reasoning")
        if reasoning is not None and (not isinstance(reasoning, str) or not reasoning.strip()):
            raise ValueError("student-visible reasoning must be explicit nonempty text")
        if "tool_calls" in message and reasoning is None:
            raise ValueError("assistant tool actions require explicit student-visible reasoning")
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
            _tool_arguments(function["name"], function["arguments"])
    else:
        if set(message) != {"role", "content", "tool_call_id"}:
            raise ValueError("tool result has an unsupported field")
        if (
            not isinstance(message.get("tool_call_id"), str)
            or _OPAQUE_ID.fullmatch(message["tool_call_id"]) is None
        ):
            raise ValueError("tool result must bind an approved opaque tool call")
    return message


def _reject_private_argument_fields(value: object) -> None:
    """Reject hidden-reasoning-shaped fields at every nested argument level."""

    if isinstance(value, Mapping):
        if _PRIVATE_FIELDS & set(value):
            raise ValueError("private or unknown reasoning fields are forbidden in tool arguments")
        for item in value.values():
            _reject_private_argument_fields(item)
    elif isinstance(value, list):
        for item in value:
            _reject_private_argument_fields(item)


def _tool_arguments(name: str, value: object) -> None:
    """Enforce the reviewed OpenCode action grammar before template rendering."""

    if not isinstance(value, Mapping):
        raise ValueError("OpenCode tool arguments must be an object")
    arguments = dict(value)
    _reject_private_argument_fields(arguments)
    if name == "bash":
        if (
            set(arguments) - {"script", "timeoutMs"}
            or not isinstance(arguments.get("script"), str)
            or not arguments["script"]
            or (
                "timeoutMs" in arguments
                and (
                    type(arguments["timeoutMs"]) is not int
                    or not 1 <= arguments["timeoutMs"] <= 300_000
                )
            )
        ):
            raise ValueError("OpenCode bash arguments are outside the reviewed grammar")
        return
    if (
        set(arguments) - {"flag", "flags", "explanation"}
        or arguments.get("explanation") != ""
        or not (
            (isinstance(arguments.get("flag"), str) and "flags" not in arguments)
            or (
                isinstance(arguments.get("flags"), list)
                and "flag" not in arguments
                and all(isinstance(flag, str) for flag in arguments["flags"])
            )
        )
    ):
        raise ValueError("OpenCode submit_report arguments are outside the reviewed grammar")


def _conversation(messages: list[dict[str, Any]]) -> None:
    """Bind every OpenCode tool result to one preceding visible call."""

    if len(messages) < 3 or [message["role"] for message in messages[:2]] != ["system", "user"]:
        raise ValueError("visible-reasoning record lacks the OpenCode system/user anchor")
    pending: set[str] = set()
    seen: set[str] = set()
    for message in messages:
        if message["role"] == "assistant":
            if pending:
                raise ValueError("assistant turn appears before its OpenCode tool result")
            for call in message.get("tool_calls") or []:
                call_id = call["id"]
                if call_id in seen:
                    raise ValueError("OpenCode tool call identity is duplicated")
                seen.add(call_id)
                pending.add(call_id)
        elif message["role"] == "tool":
            call_id = message["tool_call_id"]
            if call_id not in pending:
                raise ValueError("OpenCode tool result is orphaned or duplicated")
            pending.remove(call_id)
    if pending:
        raise ValueError("visible-reasoning record ends before an OpenCode tool result")


def _span(value: object, ids: list[int]) -> dict[str, Any]:
    span = _exact(
        value,
        {"kind", "source_message_index", "token_start", "token_end", "token_ids_sha256"},
        "target span",
    )
    if span["kind"] not in {"student_visible_reasoning", "visible_action"}:
        raise ValueError("target span has an unknown reasoning visibility")
    start = _count(span["token_start"], "target span start")
    end = _count(span["token_end"], "target span end", positive=True)
    if start >= end or end > len(ids):
        raise ValueError("target span is outside rendered token IDs")
    if span["token_ids_sha256"] != digest_json(ids[start:end]):
        raise ValueError("target span token identity differs from its rendered tokens")
    _count(span["source_message_index"], "target span source message")
    return span


def _window(value: object) -> dict[str, Any]:
    window = _exact(
        value,
        {
            "window_id",
            "sequence_index",
            "assistant_turn_id",
            "message_indices",
            "target_message_index",
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
    _count(window["sequence_index"], "window sequence index")
    _string(window["assistant_turn_id"], "assistant turn identity")
    message_indices = window["message_indices"]
    if (
        not isinstance(message_indices, list)
        or not message_indices
        or any(type(index) is not int or index < 0 for index in message_indices)
        or message_indices != sorted(set(message_indices))
    ):
        raise ValueError("window message indices must be strictly ordered nonnegative integers")
    if window["target_message_index"] != message_indices[-1]:
        raise ValueError("window target message must be the final serialized message")
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
        "sequence_index": window["sequence_index"],
        "message_indices": message_indices,
        "target_message_index": window["target_message_index"],
        "input_ids": ids,
        "prompt_token_count": prompt,
        "target_spans": spans,
    }
    if window["window_payload_sha256"] != digest_json(payload):
        raise ValueError("window payload digest mismatch")
    return window


def _source_target_identity(record: Mapping[str, Any], window: Mapping[str, Any]) -> dict[str, Any]:
    """Identify one supervised assistant target independently of packing.

    Window ids, sequence numbers, prompt length, and absolute packed-token
    offsets are deliberately absent from these identities.  The assistant
    turn is rooted in the exact normalized trajectory and target message;
    each supervised span is then normalized to offsets within that assistant
    continuation.  This prevents a collector from counting the same source
    turn twice merely by repacking it into another window.
    """

    evidence = _mapping(record.get("evidence"), "record evidence")
    trajectory_sha256 = _sha(
        evidence.get("normalized_trajectory_sha256"), "record normalized trajectory"
    )
    messages = record.get("messages")
    if not isinstance(messages, list):
        raise ValueError("private visible-reasoning record has no messages")
    target_index = _count(window.get("target_message_index"), "window target message")
    if target_index >= len(messages):
        raise ValueError("window target message is outside its private record")
    target_message = _mapping(messages[target_index], "window target message")
    if target_message.get("role") != "assistant":
        raise ValueError("window target message is not an assistant turn")
    assistant_turn_id = _string(window.get("assistant_turn_id"), "assistant turn identity")
    prompt_tokens = _count(window.get("prompt_token_count"), "window prompt token count")
    spans = window.get("target_spans")
    if not isinstance(spans, list) or not spans:
        raise ValueError("window must have explicit supervised target spans")

    source_turn = {
        "normalized_trajectory_sha256": trajectory_sha256,
        "target_message_index": target_index,
        "target_message_sha256": digest_json(target_message),
    }
    source_turn_sha256 = digest_json(source_turn)
    normalized_spans: list[dict[str, Any]] = []
    span_sha256s: list[str] = []
    for span in spans:
        checked = _mapping(span, "target span")
        if checked.get("source_message_index") != target_index:
            raise ValueError("a supervised target span is not bound to this assistant turn")
        start = _count(checked.get("token_start"), "target span start")
        end = _count(checked.get("token_end"), "target span end", positive=True)
        normalized = {
            "kind": checked.get("kind"),
            "source_message_index": target_index,
            "assistant_token_start": start - prompt_tokens,
            "assistant_token_end": end - prompt_tokens,
            "token_ids_sha256": checked.get("token_ids_sha256"),
        }
        normalized_spans.append(normalized)
        span_sha256s.append(
            digest_json(
                {
                    "source_turn_sha256": source_turn_sha256,
                    "normalized_supervised_span": normalized,
                }
            )
        )
    source_target_sha256 = digest_json(
        {
            "source_turn_sha256": source_turn_sha256,
            "assistant_turn_id": assistant_turn_id,
            "normalized_supervised_spans": normalized_spans,
        }
    )
    return {
        "source_turn_sha256": source_turn_sha256,
        "source_target_sha256": source_target_sha256,
        "source_span_sha256s": span_sha256s,
    }


def _compaction(
    value: object,
    windows: dict[str, dict[str, Any]],
    messages: list[dict[str, Any]],
    *,
    source_kind: str,
    original_task_digest: str,
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
        "parent_window_id",
        "original_task_digest",
        "prior_history_digest",
        "summary_message_index",
        "summary_message_digest",
        "continuation_token_sha256",
        "continuation_token_ids",
        "continuation_tokens",
        "summary_generation_prompt_token_sha256",
        "summary_generation_prompt_tokens",
        "pre_compaction_prompt_token_sha256",
        "pre_compaction_prompt_tokens",
        "post_compaction_prompt_token_sha256",
        "post_compaction_prompt_tokens",
        "post_compaction_message_indices",
        "next_target_window_id",
        "next_target_prompt_token_sha256",
    }
    seen: set[str] = set()
    next_targets: set[str] = set()
    for boundary in boundaries:
        boundary = _exact(boundary, fields, "compaction boundary")
        boundary_id = _string(boundary["boundary_id"], "compaction boundary identity")
        if boundary_id in seen:
            raise ValueError("compaction boundary identity is duplicated")
        seen.add(boundary_id)
        for name in fields - {
            "boundary_id",
            "parent_window_id",
            "continuation_token_ids",
            "continuation_tokens",
            "summary_generation_prompt_tokens",
            "pre_compaction_prompt_tokens",
            "post_compaction_prompt_tokens",
            "summary_message_index",
            "post_compaction_message_indices",
            "next_target_window_id",
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
        parent = windows.get(boundary["parent_window_id"])
        target = windows.get(boundary["next_target_window_id"])
        summary_index = _count(boundary["summary_message_index"], "compaction summary message")
        post_indices = boundary["post_compaction_message_indices"]
        if (
            parent is None
            or target is None
            or summary_index >= len(messages)
            or not isinstance(post_indices, list)
            or post_indices != target["message_indices"][:-1]
            or summary_index not in post_indices
            or parent["sequence_index"] >= target["sequence_index"]
            or boundary["next_target_window_id"] in next_targets
        ):
            raise ValueError("compaction lineage is not bound to the exact private record")
        next_targets.add(boundary["next_target_window_id"])
        summary = messages[summary_index]
        if (
            boundary["original_task_digest"] != original_task_digest
            or boundary["prior_history_digest"] != digest_json(messages[:summary_index])
            or boundary["summary_message_digest"] != digest_json(summary)
            or summary["role"] != "assistant"
            or set(summary) != {"role", "content"}
            or any(window["target_message_index"] == summary_index for window in windows.values())
        ):
            raise ValueError("compaction summary is not an exact zero-loss context message")
        if (
            boundary["pre_compaction_prompt_token_sha256"] != parent["prompt_token_sha256"]
            or boundary["pre_compaction_prompt_tokens"] != parent["prompt_token_count"]
            or boundary["post_compaction_prompt_token_sha256"]
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
            "cell_id",
            "wave",
            "attempt",
            "seed",
            "source_profile_sha256",
            "lineage",
            "original_task_digest",
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
    _string(record["cell_id"], "private record campaign cell identity")
    _count(record["wave"], "private record campaign wave", positive=True)
    _count(record["attempt"], "private record campaign attempt", positive=True)
    _count(record["seed"], "private record campaign seed")
    lineage = _exact(
        record["lineage"], {"task_key", "task_version_id", "group_id"}, "record lineage"
    )
    for name in lineage:
        _string(lineage[name], f"record lineage {name}")
    _sha(record["original_task_digest"], "record original task")
    evidence = _exact(
        record["evidence"],
        {
            "source_session_identity_sha256",
            "normalized_trajectory_sha256",
            "transcript_sha256",
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
    _conversation(checked_messages)
    if digest_json(checked_messages) != evidence["normalized_trajectory_sha256"]:
        raise ValueError("record trajectory digest differs from its exact messages")
    raw_windows = record["windows"]
    if not isinstance(raw_windows, list) or not raw_windows:
        raise ValueError("private visible-reasoning record has no windows")
    windows = [_window(window) for window in raw_windows]
    by_id = {window["window_id"]: window for window in windows}
    if len(by_id) != len(windows) or len({window["sequence_index"] for window in windows}) != len(
        windows
    ):
        raise ValueError("private visible-reasoning record duplicates a window identity or order")
    source_turns: set[str] = set()
    source_targets: set[str] = set()
    assistant_turns: dict[str, str] = {}
    for window in windows:
        identity = _source_target_identity(record, window)
        source_turn = identity["source_turn_sha256"]
        source_target = identity["source_target_sha256"]
        assistant_turn_id = window["assistant_turn_id"]
        if source_turn in source_turns or source_target in source_targets:
            raise ValueError(
                "private visible-reasoning record duplicates a source assistant target"
            )
        prior_turn = assistant_turns.setdefault(assistant_turn_id, source_turn)
        if prior_turn != source_turn:
            raise ValueError("assistant turn identity names more than one source target")
        source_turns.add(source_turn)
        source_targets.add(source_target)
    _compaction(
        record["compaction"],
        by_id,
        checked_messages,
        source_kind=profile["source"]["kind"],
        original_task_digest=record["original_task_digest"],
    )
    return record


def _rendered_window(
    tokenizer: Any,
    record: dict[str, Any],
    window: dict[str, Any],
) -> None:
    """Re-render one source window; no caller-controlled adapter is trusted."""
    indices = window["message_indices"]
    messages = record["messages"]
    if max(indices) >= len(messages):
        raise ValueError("window names a message outside its private record")
    selected = [messages[index] for index in indices]
    if selected[-1]["role"] != "assistant":
        raise ValueError("window target message is not an assistant turn")
    prompt = _template_ids(tokenizer, selected[:-1], generation=True)
    rendered = _template_ids(tokenizer, selected, generation=False)
    if (
        rendered != window["input_ids"]
        or prompt != window["input_ids"][: window["prompt_token_count"]]
        or len(prompt) != window["prompt_token_count"]
    ):
        raise ValueError(
            "Qwen/OpenCode template serialization differs from the collected token boundary"
        )
    target_index = window["target_message_index"]
    if any(span["source_message_index"] != target_index for span in window["target_spans"]):
        raise ValueError("a supervised target span is not bound to this assistant turn")

    target_message = selected[-1]
    reasoning = target_message.get("student_visible_reasoning")
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise ValueError("target assistant turn lacks explicit student-visible reasoning")
    if not target_message.get("content", "").strip() and not target_message.get("tool_calls"):
        raise ValueError("target assistant turn lacks a visible action")

    # Derive the component boundary from the pinned Qwen template itself.  The
    # counterfactual keeps the exact authorized reasoning but removes both
    # structured action channels.  For the pinned template it must equal a
    # prefix of the full rendering followed by the exact terminal suffix.  A
    # caller-provided span label or boundary is never used to derive this split.
    reasoning_only_message = {
        "role": "assistant",
        "content": "",
        "student_visible_reasoning": reasoning,
    }
    reasoning_only = _template_ids(
        tokenizer,
        [*selected[:-1], reasoning_only_message],
        generation=False,
    )
    common_prefix = 0
    for actual, counterfactual in zip(rendered, reasoning_only, strict=False):
        if actual != counterfactual:
            break
        common_prefix += 1
    common_suffix = 0
    for actual, counterfactual in zip(reversed(rendered), reversed(reasoning_only), strict=False):
        if actual != counterfactual:
            break
        common_suffix += 1
    if (
        common_prefix <= len(prompt)
        or common_prefix >= len(rendered)
        or common_suffix == 0
        or common_prefix + common_suffix != len(reasoning_only)
        or common_prefix + common_suffix > len(rendered)
    ):
        raise ValueError("Qwen template cannot derive an unambiguous reasoning/action boundary")
    expected_spans = [
        {
            "kind": "student_visible_reasoning",
            "source_message_index": target_index,
            "token_start": len(prompt),
            "token_end": common_prefix,
            "token_ids_sha256": digest_json(rendered[len(prompt) : common_prefix]),
        },
        {
            "kind": "visible_action",
            "source_message_index": target_index,
            "token_start": common_prefix,
            "token_end": len(rendered),
            "token_ids_sha256": digest_json(rendered[common_prefix:]),
        },
    ]
    if window["target_spans"] != expected_spans:
        raise ValueError(
            "caller target spans differ from template-derived reasoning/action components"
        )
    covered = []
    for span in window["target_spans"]:
        covered.extend(range(span["token_start"], span["token_end"]))
    target = list(range(window["prompt_token_count"], len(window["input_ids"])))
    if covered != target:
        raise ValueError("supervised spans must cover the exact assistant continuation once")


def _rendered_compaction(tokenizer: Any, record: dict[str, Any]) -> None:
    """Prove every exact continuation serializes the declared summary message.

    A digest of caller-provided token IDs is not enough: this re-renders the
    Qwen assistant turn which generated the summary and requires that its
    continuation exactly equals the recorded continuation bytes.  The summary
    remains zero-loss context; this only proves the later prompt did not hide
    opaque or private text behind a self-consistent digest.
    """

    compaction = record["compaction"]
    if compaction["kind"] == "none":
        return
    messages = record["messages"]
    for boundary in compaction["boundaries"]:
        summary_index = boundary["summary_message_index"]
        prompt = _template_ids(tokenizer, messages[:summary_index], generation=True)
        rendered = _template_ids(tokenizer, messages[: summary_index + 1], generation=False)
        continuation = boundary["continuation_token_ids"]
        if (
            prompt != rendered[: len(prompt)]
            or len(prompt) != boundary["summary_generation_prompt_tokens"]
            or digest_json(prompt) != boundary["summary_generation_prompt_token_sha256"]
            or continuation != rendered[len(prompt) :]
            or len(continuation) != boundary["continuation_tokens"]
            or digest_json(continuation) != boundary["continuation_token_sha256"]
        ):
            raise ValueError("compaction continuation does not serialize its exact summary message")


def _compacted_target_window_ids(record: Mapping[str, Any]) -> set[str]:
    """Return only targets that actually follow a proven compaction boundary."""

    compaction = _mapping(record.get("compaction"), "record compaction")
    if compaction.get("kind") == "none":
        return set()
    boundaries = compaction.get("boundaries")
    if not isinstance(boundaries, list):  # Validated by ``_record`` before production use.
        raise ValueError("exact compaction has no boundaries")
    return {
        _string(
            _mapping(boundary, "compaction boundary").get("next_target_window_id"), "next target"
        )
        for boundary in boundaries
    }


def _loss_mask(window: dict[str, Any]) -> list[int]:
    mask = [0] * len(window["input_ids"])
    for span in window["target_spans"]:
        mask[span["token_start"] : span["token_end"]] = [1] * (
            span["token_end"] - span["token_start"]
        )
    return mask


def _action_only_loss_mask(window: dict[str, Any]) -> list[int]:
    mask = [0] * len(window["input_ids"])
    for span in window["target_spans"]:
        if span["kind"] == "visible_action":
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
            "source_authorization",
            "campaign_plan",
            "wave_plan",
            "operation_authorization",
            "collection_packet",
            "selection",
            "success_evidence",
            "reasoning_census",
            "inventory",
            "family_split",
            "role_anchor",
            "protected_family_lock",
            "runtime_bindings",
            "task_selection",
            "roundtrip_fixture",
            "model_lock",
            "tokenizer_root",
            "records",
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
        "source_authorization",
        "campaign_plan",
        "wave_plan",
        "operation_authorization",
        "collection_packet",
        "selection",
        "success_evidence",
        "reasoning_census",
        "inventory",
        "family_split",
        "role_anchor",
        "protected_family_lock",
        "runtime_bindings",
        "task_selection",
        "roundtrip_fixture",
        "model_lock",
    )
    sources = {name: _input(relative_to, config.get(name), name) for name in public_names}
    records_path, records_sha256 = _private_records_input(relative_to, config.get("records"))
    sources["records"] = (records_path, records_sha256)
    paths = {name: item[0] for name, item in sources.items()}
    expected_files = {name: item[1] for name, item in sources.items()}
    profile = _profile(_json(paths["source_profile"], "source profile"))
    source_authorization = _source_authorization(
        _json(paths["source_authorization"], "source authorization"), profile
    )
    if expected_files["roundtrip_fixture"] != profile["serialization"]["roundtrip_fixture_sha256"]:
        raise ValueError("round-trip fixture file does not match the source profile")
    tokenizer_root = _path(relative_to, config.get("tokenizer_root"), "tokenizer root")
    tokenizer, tokenizer_identity = _load_tokenizer(
        read_mapping(paths["model_lock"]), tokenizer_root, profile
    )
    _roundtrip(
        _json(paths["roundtrip_fixture"], "round-trip fixture"),
        profile,
        tokenizer,
        tokenizer_identity,
    )
    packet = _packet(_json(paths["collection_packet"], "collection packet"), profile)
    _operation_authorization(
        _json(paths["operation_authorization"], "operation authorization"), packet
    )
    inventory = _json(paths["inventory"], "catalog inventory")
    split = _json(paths["family_split"], "family split")
    role_anchor = _json(paths["role_anchor"], "family role anchor")
    lock = _json(paths["protected_family_lock"], "protected-family lock")
    runtime = _json(paths["runtime_bindings"], "runtime bindings")
    task_selection = _json(paths["task_selection"], "campaign task selection")
    _campaign_artifacts(
        _json(paths["campaign_plan"], "campaign plan"),
        _json(paths["wave_plan"], "wave plan"),
        packet,
        task_selection,
        split,
        profile,
    )
    bindings = _task_boundary(inventory, split, role_anchor, lock, runtime, task_selection, packet)
    selection = _json(paths["selection"], "private selection")
    success_evidence_document, success_evidence = _success_evidence(
        _json(paths["success_evidence"], "success evidence"),
        packet,
        split,
        lock,
        source_authorization,
    )
    if success_evidence_document["source_profile_sha256"] != profile["sha256"]:
        raise ValueError("success evidence changes the exact Qwen source profile")
    if selection.get("verified_success_evidence_sha256") != success_evidence_document["sha256"]:
        raise ValueError("private selection does not bind the immutable success evidence mapping")
    selected = _selection(selection, packet, split, lock, bindings, success_evidence)
    source_census = visible_reasoning_census.validate_source_census(
        _json(paths["reasoning_census"], "visible-reasoning source census"),
        source_profile_sha256=profile["sha256"],
        source_authorization_sha256=source_authorization["sha256"],
        collection_packet_sha256=packet["sha256"],
        private_selection_sha256=selection["sha256"],
        success_evidence_sha256=success_evidence_document["sha256"],
        selected_sessions=len(selected),
    )

    # This is the first point at which the private payload is read.  A bad
    # catalog, split, anchor, lock, packet, or selection stops above without
    # hashing or parsing raw trajectory content.
    if file_sha256(paths["records"]) != expected_files["records"]:
        raise ValueError("records file digest mismatch")
    records: dict[str, dict[str, Any]] = {}
    for raw in iter_jsonl(paths["records"]):
        record = _record(raw, profile)
        _rendered_compaction(tokenizer, record)
        if record["record_id"] in records:
            raise ValueError("private records duplicate a record identity")
        records[record["record_id"]] = record
    if set(records) != set(selected):
        raise ValueError("private records do not cover the exact sealed selection")

    rows: list[dict[str, Any]] = []
    action_only_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    window_digests: set[str] = set()
    source_turns: set[str] = set()
    source_targets: set[str] = set()
    source_span_occurrences: set[str] = set()
    family_tokens: dict[str, int] = {}
    reasoning_tokens = action_tokens = 0
    compacted_windows = 0
    for record_id, selected_row in sorted(selected.items()):
        record = records[record_id]
        if (
            record["cell_id"] != selected_row["cell_id"]
            or record["wave"] != selected_row["wave"]
            or record["attempt"] != selected_row["attempt"]
            or record["seed"] != selected_row["seed"]
            or record["lineage"]
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
            or digest_json(record) != selected_row["normalized_record_sha256"]
            or ("none" if record["compaction"]["kind"] == "none" else EXACT_COMPACTION)
            != selected_row["compaction_kind"]
        ):
            raise ValueError("private record differs from its sealed success selection")
        record_reasoning = record_action = 0
        for window in record["windows"]:
            _rendered_window(tokenizer, record, window)
            identity = _source_target_identity(record, window)
            source_turn = identity["source_turn_sha256"]
            source_target = identity["source_target_sha256"]
            if source_turn in source_turns or source_target in source_targets:
                raise ValueError("duplicate source assistant target would repeat supervision")
            source_turns.add(source_turn)
            source_targets.add(source_target)
            payload = window["window_payload_sha256"]
            if payload in window_digests:
                raise ValueError("duplicate packed window would repeat supervised targets")
            window_digests.add(payload)
            mask = _loss_mask(window)
            for span, occurrence in zip(
                window["target_spans"], identity["source_span_sha256s"], strict=True
            ):
                # Equal token bytes can occur in two genuinely distinct tool
                # actions.  Deduplicate the exact source occurrence, not a
                # token-content hash that would incorrectly erase one of them.
                if occurrence in source_span_occurrences:
                    raise ValueError("one visible target span appears more than once")
                source_span_occurrences.add(occurrence)
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
            paired_window_sha256 = digest_json(
                {
                    "source_record_sha256": selected_row["normalized_record_sha256"],
                    "source_task_version_id": selected_row["task_version_id"],
                    "source_target_sha256": source_target,
                    "source_window_sha256": payload,
                    "input_ids_sha256": digest_json(window["input_ids"]),
                }
            )
            common = {
                "input_ids": window["input_ids"],
                "source_record_sha256": selected_row["normalized_record_sha256"],
                "source_task_version_id": selected_row["task_version_id"],
                "source_turn_sha256": source_turn,
                "source_target_sha256": source_target,
                "source_window_sha256": payload,
                "source_profile_sha256": profile["sha256"],
                "collection_packet_sha256": packet["sha256"],
                "paired_window_sha256": paired_window_sha256,
            }
            rows.append({**common, "loss_mask": mask})
            action_only_rows.append({**common, "loss_mask": _action_only_loss_mask(window)})
        if not record_reasoning or not record_action:
            raise ValueError("selected record lacks paired visible reasoning and action coverage")
        compacted_window_ids = _compacted_target_window_ids(record)
        compacted_windows += sum(
            window["window_id"] in compacted_window_ids for window in record["windows"]
        )
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
    if len(source_turns) != len(rows) or len(source_targets) != len(rows):
        raise ValueError("source-target identity count differs from materialized windows")
    if len(action_only_rows) != len(rows) or any(
        reasoning_row["paired_window_sha256"] != action_row["paired_window_sha256"]
        or reasoning_row["source_target_sha256"] != action_row["source_target_sha256"]
        or reasoning_row["input_ids"] != action_row["input_ids"]
        for reasoning_row, action_row in zip(rows, action_only_rows, strict=True)
    ):
        raise ValueError("matched action-only arm differs from the reasoning-arm source windows")
    total_tokens = reasoning_tokens + action_tokens
    family_total = sum(family_tokens.values())
    family_fraction = max(family_tokens.values()) / family_total
    token_goal_reached = total_tokens >= packet["minimum_unique_supervised_tokens"]
    family_goal_reached = len(family_tokens) >= packet["minimum_successful_families"]
    target_goal_reached = (
        token_goal_reached
        and family_goal_reached
        and family_fraction <= packet["maximum_family_target_token_fraction"]
    )
    if token_goal_reached and family_fraction > packet["maximum_family_target_token_fraction"]:
        raise ValueError("target-ready corpus exceeds immutable family concentration limit")
    coverage = {
        "schema": COVERAGE_SCHEMA,
        "selection_sha256": selection["sha256"],
        "success_evidence_sha256": success_evidence_document["sha256"],
        "source_census_sha256": source_census["sha256"],
        "collection_packet_sha256": packet["sha256"],
        "source_profile_sha256": profile["sha256"],
        "source_authorization_sha256": source_authorization["sha256"],
        "selected_source_records": len(selection_rows),
        "visible_reasoning_windows": len(rows),
        "student_visible_reasoning_target_tokens": reasoning_tokens,
        "visible_action_target_tokens": action_tokens,
        "unique_supervised_tokens": total_tokens,
        "minimum_unique_supervised_tokens": packet["minimum_unique_supervised_tokens"],
        "token_goal_reached": token_goal_reached,
        "minimum_successful_families": packet["minimum_successful_families"],
        "successful_family_goal_reached": family_goal_reached,
        "target_goal_reached": target_goal_reached,
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
        "source_authorization_sha256": source_authorization["sha256"],
        "collection_packet_sha256": packet["sha256"],
        "selection_sha256": coverage["selection_sha256"],
        "success_evidence_sha256": success_evidence_document["sha256"],
        "source_census_sha256": source_census["sha256"],
        "catalog_inventory_sha256": packet["catalog_inventory_sha256"],
        "family_split_sha256": packet["family_split_sha256"],
        "root_role_anchor_id": TRUSTED_FLEET_COLLECTION_ROOT_ID,
        "family_role_anchor_sha256": packet["family_role_anchor_sha256"],
        "protected_family_lock_sha256": packet["protected_family_lock_sha256"],
        "runtime_bindings_sha256": packet["runtime_bindings_sha256"],
        "files": {
            "reasoning_plus_action": {
                "path": "train-reasoning-plus-action.parquet",
                "sha256": None,
                "rows": len(rows),
            },
            "matched_action_only": {
                "path": "train-matched-action-only.parquet",
                "sha256": None,
                "rows": len(action_only_rows),
            },
        },
        "counts": {
            "source_records": len(selection_rows),
            "windows": len(rows),
            "student_visible_reasoning_target_tokens": reasoning_tokens,
            "visible_action_target_tokens": action_tokens,
            "supervised_tokens": total_tokens,
            "matched_action_only_supervised_tokens": action_tokens,
        },
        "campaign_identity": {
            "campaign_plan_sha256": packet["campaign_plan_sha256"],
            "wave_plan_sha256": packet["wave_plan_sha256"],
            "cell_identity_universe_sha256": packet["cell_identity_universe_sha256"],
            "operation_authorization_sha256": packet["operation_authorization_sha256"],
            "task_selection_sha256": packet["task_selection_sha256"],
        },
        "matched_ablation": {
            "paired_window_identity_sha256": digest_json(
                [row["paired_window_sha256"] for row in rows]
            ),
            "source_target_identity_sha256": digest_json(
                [row["source_target_sha256"] for row in rows]
            ),
            "same_selected_windows": True,
            "same_input_ids": True,
            "only_reasoning_loss_mask_differs": True,
            "cross_arm_sha256": None,
        },
        "arm_manifests": {},
        "validation_mode": "pending_reasoning_selection",
        "coverage_sha256": coverage["sha256"],
        "limitations": [
            "Only explicitly authorized student-visible reasoning is supervised.",
            "Private or unknown reasoning fields and opaque compaction are rejected.",
            (
                "Unique tokens count each normalized source assistant span once, "
                "independent of packing metadata."
            ),
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
        reasoning_path = output / "train-reasoning-plus-action.parquet"
        action_path = output / "train-matched-action-only.parquet"
        pq.write_table(pa.Table.from_pylist(rows), reasoning_path, compression="zstd")
        pq.write_table(pa.Table.from_pylist(action_only_rows), action_path, compression="zstd")
        os.chmod(reasoning_path, 0o600)
        os.chmod(action_path, 0o600)
        if pq.read_table(reasoning_path).to_pylist() != rows:
            raise ValueError("private visible-reasoning Parquet readback differs")
        if pq.read_table(action_path).to_pylist() != action_only_rows:
            raise ValueError("private matched action-only Parquet readback differs")
        manifest["files"]["reasoning_plus_action"]["sha256"] = file_sha256(reasoning_path)
        manifest["files"]["matched_action_only"]["sha256"] = file_sha256(action_path)
        paired_identity = manifest["matched_ablation"]["paired_window_identity_sha256"]
        source_target_identity = manifest["matched_ablation"]["source_target_identity_sha256"]
        arm_values = {
            "reasoning_plus_action": {
                "path": reasoning_path.name,
                "sha256": manifest["files"]["reasoning_plus_action"]["sha256"],
                "rows": len(rows),
                "target_mode": "student_visible_reasoning_plus_visible_actions",
                "supervised_tokens": total_tokens,
            },
            "matched_action_only": {
                "path": action_path.name,
                "sha256": manifest["files"]["matched_action_only"]["sha256"],
                "rows": len(action_only_rows),
                "target_mode": "visible_actions_only",
                "supervised_tokens": action_tokens,
            },
        }
        arm_manifests: dict[str, dict[str, Any]] = {}
        for arm_name, arm_file in arm_values.items():
            arm_manifest = {
                "schema": ARM_MANIFEST_SCHEMA,
                "arm": arm_name,
                "parquet": arm_file,
                "source_profile_sha256": profile["sha256"],
                "collection_packet_sha256": packet["sha256"],
                "selection_sha256": selection["sha256"],
                "paired_window_identity_sha256": paired_identity,
                "source_target_identity_sha256": source_target_identity,
            }
            arm_manifest["sha256"] = digest_json(arm_manifest)
            arm_manifests[arm_name] = arm_manifest
        cross_arm = {
            "reasoning_plus_action_manifest_sha256": arm_manifests["reasoning_plus_action"][
                "sha256"
            ],
            "matched_action_only_manifest_sha256": arm_manifests["matched_action_only"]["sha256"],
            "paired_window_identity_sha256": paired_identity,
            "source_target_identity_sha256": source_target_identity,
        }
        manifest["matched_ablation"]["cross_arm_sha256"] = digest_json(cross_arm)
        for arm_name, arm_manifest in arm_manifests.items():
            arm_path = output / f"{arm_name.replace('_', '-')}.manifest.json"
            atomic_write_json(arm_path, arm_manifest, private=True)
            manifest["arm_manifests"][arm_name] = {
                "path": arm_path.name,
                "file_sha256": file_sha256(arm_path),
                "logical_sha256": arm_manifest["sha256"],
            }
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
            "sft_ready": False,
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
        "matched_action_only_target_tokens": action_tokens,
        "sft_ready": False,
    }
