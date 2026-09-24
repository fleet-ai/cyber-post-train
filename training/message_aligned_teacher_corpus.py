"""Rebuild broad teacher SFT data from raw, complete message boundaries.

This is deliberately a new corpus identity.  It never rechunks a token row:
each output window is rendered again from the original system/task anchor,
the exact model-facing tool schemas, and whole source messages.  The builder
also refuses to infer task-family aliases or a successful final report.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from cyber_post_train.jobs import digest

from .corpus import local_tokenizer
from .dense import Excluded, _normalized_for_template, native_helper, segment_record
from .io import atomic_write_json, atomic_write_jsonl, digest_json, file_sha256, iter_jsonl
from .post_sft_staging import _rename_noreplace
from .sft import read_mapping
from .sft_runtime import DENSE_FORMAT, dense_rows
from .task_family_split import TRUSTED_FLEET_COLLECTION_ROOT_ID

REQUEST_SCHEMA = "cyber_message_aligned_teacher_corpus_request_v1"
# Keep the proven trainer envelope/row format.  The request, materialization,
# window, algorithm, and final digest are new identities; minting an unrelated
# runtime format would only fork the already-qualified dense loader.
CORPUS_SCHEMA = "cyber_dense_sft_corpus_v1"
MATERIALIZATION_SCHEMA = "cyber_message_aligned_teacher_materialization_v1"
EVIDENCE_SCHEMA = "fleet_broad_success_evidence_v2"
FAMILY_ROSTER_SCHEMA = "cyber_exact_task_family_role_roster_v1"
TOOL_CONTRACT_SCHEMA = "cyber_opencode_model_tool_contract_v1"
MODEL_REQUEST_CAPTURE_SCHEMA = "cyber_opencode_model_request_tool_capture_v1"
ALGORITHM = "anchored_complete_message_rounds_with_exact_tool_contract_v1"
WINDOW_SCHEMA = "cyber_message_aligned_sft_window_v1"
RECORD_SCHEMA = "fleet_cyber_trajectory_v1"
TARGET_TOOL_NAMES = ("fleet_bash", "fleet_submit_report")
OPENCODE_VERSION = "1.18.27"
OPENCODE_RELEASE_SHA256 = "sha256:4af5494f9433f59db8c1e344198f0ee72a50c06ec009fb4a8aeab4c2d4abd702"
SOURCE_TOOL_CATALOG_SHA256 = (
    "sha256:85fad6bdc3a835bf52a11a99b3387740eb06eb3d1720ad9bb33f3feac215b44a"
)
TARGET_TOOLS_SHA256 = "sha256:585574ec1a459141a2e79f4945d140864876224ebef1260be65f06c6d237610f"
TRANSCRIPT_ROUTE = "GET /v1/sessions/{session_id}/transcript"
SOURCE_TOOL_ALIASES = {
    "bash": "bash",
    "fleet_environment__bash": "bash",
    "fleet_environment__fleet_environment__bash": "bash",
    "mcp__fleet_environment__bash": "bash",
    "submit_report": "submit_report",
    "fleet_environment__submit_report": "submit_report",
    "mcp__fleet_environment__submit_report": "submit_report",
    "mcp__fleet_environment__mcp__fleet_environment__submit_report": "submit_report",
}


def encode_record(
    messages: list[dict], tokenizer: Any, helper: Any, *, tools: list[dict]
) -> tuple[list[int], list[dict]]:
    """Render complete messages with the successor's bound tool-schema anchor."""
    try:
        anchor = tokenizer.apply_chat_template(
            messages[:2],
            tokenize=True,
            add_generation_prompt=False,
            return_dict=False,
            tools=tools,
        )
        chunks = []
        ordinal = 0
        for index, message in enumerate(messages[2:], 2):
            ids, mask, _ = helper([message], tokenizer, tokenizer_kwargs={"tools": []})
            if len(ids) != len(mask):
                raise ValueError("native lengths")
            assistant = message["role"] == "assistant"
            if assistant:
                positions = [position for position, value in enumerate(mask) if value]
                if not positions or positions != list(range(positions[0], positions[-1] + 1)):
                    raise ValueError("native assistant mask not contiguous")
                target = (positions[0], positions[-1] + 1)
            else:
                if any(mask):
                    raise ValueError("native tool masked incorrectly")
                target = None
            chunks.append(
                {
                    "ids": ids,
                    "mask": mask,
                    "message_index": index,
                    "assistant_index": ordinal if assistant else None,
                    "target": target,
                }
            )
            ordinal += assistant
        return anchor, chunks
    except Exception:
        # Template failures may contain private task text or token IDs.
        raise Excluded("native_template_or_mask_contract") from None


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 71 or not value.startswith("sha256:"):
        raise ValueError(f"exact {label} digest is required")
    try:
        int(value[7:], 16)
    except ValueError as error:
        raise ValueError(f"exact {label} digest is required") from error
    return value


def _schema_value(value: object, schema: object, *, definition: bool = False) -> None:
    """Validate the small JSON-Schema subset used by the two Fleet tools."""
    if not isinstance(schema, dict) or set(schema) - {
        "type",
        "properties",
        "required",
        "additionalProperties",
        "description",
        "minLength",
        "minimum",
        "maximum",
        "items",
        "enum",
    }:
        raise ValueError("tool schema uses an unsupported JSON-Schema feature")
    kind = schema.get("type")
    if kind not in {"object", "string", "integer", "array"}:
        raise ValueError("tool schema has an unsupported or missing type")
    if kind == "object":
        properties = schema.get("properties")
        required = schema.get("required", [])
        if (
            not isinstance(properties, dict)
            or not isinstance(required, list)
            or any(not isinstance(item, str) or item not in properties for item in required)
            or schema.get("additionalProperties", False) is not False
        ):
            raise ValueError("tool object schema must explicitly reject unknown fields")
        for child in properties.values():
            _schema_value(None, child, definition=True)
        if definition:
            return
        if (
            not isinstance(value, dict)
            or any(item not in properties for item in value)
            or any(item not in value for item in required)
        ):
            raise Excluded("tool_argument_schema_mismatch")
        for name, item in value.items():
            _schema_value(item, properties[name])
        return
    if definition:
        if kind == "array":
            _schema_value(None, schema.get("items"), definition=True)
        return
    if kind == "string":
        if not isinstance(value, str) or len(value) < schema.get("minLength", 0):
            raise Excluded("tool_argument_schema_mismatch")
    elif kind == "integer":
        if type(value) is not int or not schema.get("minimum", value) <= value <= schema.get(
            "maximum", value
        ):
            raise Excluded("tool_argument_schema_mismatch")
    elif kind == "array":
        if not isinstance(value, list):
            raise Excluded("tool_argument_schema_mismatch")
        for item in value:
            _schema_value(item, schema["items"])
    if "enum" in schema and value not in schema["enum"]:
        raise Excluded("tool_argument_schema_mismatch")


def _wire_digest(value: object) -> str:
    """Match the ASCII-escaped canonical JSON used by the Fleet harness."""
    body = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(body.encode()).hexdigest()


def derive_opencode_tools(value: object) -> list[dict[str, Any]]:
    """Reproduce OpenCode 1.18.27's exact Fleet MCP-to-model transform."""
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("source catalog must contain exactly two Fleet MCP tools")
    result = []
    raw_names = []
    for raw in value:
        if not isinstance(raw, dict) or set(raw) != {"name", "description", "inputSchema"}:
            raise ValueError("source catalog tool is malformed")
        name = _string(raw["name"], "source tool name")
        description = _string(raw["description"], "source tool description")
        schema = raw["inputSchema"]
        _schema_value(None, schema, definition=True)
        parameters = {**schema, "additionalProperties": False}
        raw_names.append(name)
        result.append(
            {
                "type": "function",
                "function": {
                    "name": f"fleet_{re.sub(r'[^a-zA-Z0-9_-]', '_', name)}",
                    "description": description,
                    "parameters": parameters,
                },
            }
        )
    result.sort(key=lambda item: item["function"]["name"])
    if (
        raw_names != ["bash", "submit_report"]
        or _wire_digest(value) != SOURCE_TOOL_CATALOG_SHA256
        or tuple(item["function"]["name"] for item in result) != TARGET_TOOL_NAMES
        or _wire_digest(result) != TARGET_TOOLS_SHA256
    ):
        raise ValueError("source catalog or OpenCode tool transform differs")
    return result


def _sealed(value: dict[str, Any], schema: str) -> None:
    if value.get("schema") != schema or value.get("sha256") != digest_json(
        {key: item for key, item in value.items() if key != "sha256"}
    ):
        raise ValueError(f"invalid sealed {schema}")


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"nonempty {label} is required")
    return value


def _input(root: Path, value: object, label: str) -> Path:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} must bind one path and digest")
    path = Path(_string(value["path"], f"{label} path"))
    path = path if path.is_absolute() else root / path
    if path.is_symlink() or not path.is_file() or file_sha256(path) != value["sha256"]:
        raise ValueError(f"{label} file digest mismatch")
    return path


def _json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label} JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _json_value(path: Path, label: str) -> object:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label} JSON") from error


def tool_contract(value: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Validate the exact OpenCode model-facing names and JSON schemas."""
    _sealed(value, TOOL_CONTRACT_SCHEMA)
    if set(value) != {
        "schema",
        "source_catalog_sha256",
        "target_harness",
        "target_model",
        "provider_schema_transform_sha256",
        "model_request_capture_sha256",
        "source_aliases",
        "target_tools",
        "sha256",
    }:
        raise ValueError("tool contract has unexpected fields")
    if value["target_harness"] != {
        "name": "opencode",
        "version": OPENCODE_VERSION,
        "release_asset_sha256": OPENCODE_RELEASE_SHA256,
        "mcp_server": "fleet",
    }:
        raise ValueError("tool contract is not the OpenCode Fleet model surface")
    model = value["target_model"]
    if (
        not isinstance(model, dict)
        or set(model) != {"repo", "revision"}
        or model.get("repo") != "Qwen/Qwen3.8-27B"
        or not isinstance(model.get("revision"), str)
        or re.fullmatch(r"[0-9a-f]{40}", model["revision"]) is None
    ):
        raise ValueError("tool contract is not bound to an exact Qwen3.8 model")
    for field in ("provider_schema_transform_sha256", "model_request_capture_sha256"):
        _sha(value[field], f"tool contract {field}")
    if value["source_catalog_sha256"] != SOURCE_TOOL_CATALOG_SHA256:
        raise ValueError("tool contract source catalog differs")
    aliases = value["source_aliases"]
    if aliases != {"bash": "fleet_bash", "submit_report": "fleet_submit_report"}:
        raise ValueError("tool contract needs the exact Fleet source aliases")
    tools = value["target_tools"]
    if not isinstance(tools, list) or len(tools) != 2:
        raise ValueError("tool contract needs exactly two model-facing tools")
    names = []
    for tool in tools:
        if not isinstance(tool, dict) or set(tool) != {"type", "function"}:
            raise ValueError("tool contract must use OpenAI function-tool objects")
        function = tool["function"]
        if not isinstance(function, dict) or set(function) != {
            "name",
            "description",
            "parameters",
        }:
            raise ValueError("tool contract function is incomplete")
        names.append(_string(function["name"], "target tool name"))
        _string(function["description"], "target tool description")
        _schema_value(None, function["parameters"], definition=True)
    if tuple(names) != TARGET_TOOL_NAMES:
        raise ValueError("tool contract names differ from the OpenCode Fleet surface")
    if _wire_digest(tools) != TARGET_TOOLS_SHA256:
        raise ValueError("tool contract schemas differ from the exact OpenCode transform")
    return tools, dict(aliases)


def model_request_capture(
    value: dict[str, Any], contract: dict[str, Any], tools: list[dict[str, Any]]
) -> None:
    """Require a sealed, sanitized capture of the tools sent to Qwen."""
    _sealed(value, MODEL_REQUEST_CAPTURE_SCHEMA)
    if set(value) != {
        "schema",
        "target_harness",
        "target_model",
        "provider_schema_transform_sha256",
        "request_envelope_sha256",
        "captured_tools",
        "sha256",
    }:
        raise ValueError("model request capture has unexpected fields")
    if (
        value["target_harness"] != contract["target_harness"]
        or value["target_model"] != contract["target_model"]
        or value["provider_schema_transform_sha256"] != contract["provider_schema_transform_sha256"]
        or value["sha256"] != contract["model_request_capture_sha256"]
    ):
        raise ValueError("model request capture differs from the tool contract")
    _sha(value["request_envelope_sha256"], "captured model request envelope")
    if (
        value["captured_tools"] != tools
        or _wire_digest(value["captured_tools"]) != TARGET_TOOLS_SHA256
    ):
        raise ValueError("captured model request tools differ from the exact OpenCode contract")


def family_roster(value: dict[str, Any]) -> dict[tuple[str, str], dict[str, str]]:
    """Load exact identities while enforcing one role per reviewed family."""
    _sealed(value, FAMILY_ROSTER_SCHEMA)
    if (
        set(value)
        != {
            "schema",
            "root_role_anchor_id",
            "family_role_anchor_sha256",
            "heldout_group_ids",
            "identities",
            "sha256",
        }
        or value["root_role_anchor_id"] != TRUSTED_FLEET_COLLECTION_ROOT_ID
    ):
        raise ValueError("family roster is not bound to the trusted study root")
    heldout = value["heldout_group_ids"]
    rows = value["identities"]
    _sha(value["family_role_anchor_sha256"], "family role anchor")
    if (
        not isinstance(heldout, list)
        or heldout != sorted(set(heldout))
        or not heldout
        or not isinstance(rows, list)
        or not rows
    ):
        raise ValueError("family roster is empty or malformed")
    roles: dict[str, str] = {}
    result: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "task_key",
            "task_version_id",
            "group_id",
            "split",
        }:
            raise ValueError("family roster identity is malformed")
        identity = (
            _string(row["task_key"], "task key"),
            _string(row["task_version_id"], "task version"),
        )
        group = _sha(row["group_id"], "family group")
        split = row["split"]
        if split not in {"train", "dev", "final_test"} or identity in result:
            raise ValueError("family roster has an invalid or duplicate identity")
        if roles.setdefault(group, split) != split:
            raise ValueError("one reviewed family has conflicting split roles")
        result[identity] = {"group_id": group, "split": split}
    if set(heldout) != {group for group, split in roles.items() if split != "train"}:
        raise ValueError("held-out groups do not equal every nontraining family")
    return result


def _arguments(function: dict[str, Any]) -> dict[str, Any]:
    arguments = function.get("arguments")
    try:
        arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
    except (TypeError, ValueError):
        raise Excluded("invalid_tool_arguments") from None
    if not isinstance(arguments, dict):
        raise Excluded("invalid_tool_arguments")
    return arguments


def _canonical_source_tool(name: object, arguments: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Reuse only the exact historical broad-export aliases and wrapper shape."""
    if name == "use_tool" and set(arguments) >= {"tool_name", "tool_input"}:
        name, arguments = arguments["tool_name"], arguments["tool_input"]
        if not isinstance(name, str) or not name or not isinstance(arguments, dict):
            raise Excluded("invalid_use_tool_wrapper")
    if name == "shell" and set(arguments) == {"command"}:
        name, arguments = "bash", {"script": arguments["command"]}
    canonical = SOURCE_TOOL_ALIASES.get(name) if isinstance(name, str) else None
    if canonical is None:
        raise Excluded("tool_name_contract_mismatch")
    return canonical, arguments


def _verified_success(record: dict[str, Any], proof: dict[str, Any], teachers: set[str]) -> bool:
    eligibility, outcome = record.get("eligibility"), record.get("outcome")
    score = outcome.get("score") if isinstance(outcome, dict) else None
    return bool(
        record.get("schema") == RECORD_SCHEMA
        and record.get("source", {}).get("model") in teachers
        and isinstance(eligibility, dict)
        and eligibility.get("sft") is True
        and isinstance(outcome, dict)
        and outcome.get("infra_valid") is True
        and outcome.get("success") is True
        and type(score) in (int, float)
        and not isinstance(score, bool)
        and math.isfinite(score)
        and score >= 1
        and proof.get("outcome")
        == {
            "score_at_least_one": True,
            "status": "completed",
            "verifier_process_success": True,
        }
    )


def _source_provenance(
    record: dict[str, Any], proof: dict[str, Any]
) -> tuple[tuple[str, str], dict[str, Any]]:
    if record.get("content_digest") != digest_json(
        {key: value for key, value in record.items() if key != "content_digest"}
    ) or proof["normalized_record_sha256"] != digest_json(record):
        raise ValueError("normalized record digest mismatch")
    lineage, source = record.get("lineage"), record.get("source")
    if not isinstance(lineage, dict) or not isinstance(source, dict):
        raise ValueError("normalized source provenance is missing")
    identity = (
        _string(lineage.get("task_key"), "source task key"),
        _string(lineage.get("eval_task_version_id"), "source task version"),
    )
    if any(
        (
            proof.get("task_key") != identity[0],
            proof.get("task_version_id") != identity[1],
            proof.get("session_id") != source.get("session_id"),
            proof.get("model_id") != source.get("model"),
            proof.get("routes", {}).get("transcript") != TRANSCRIPT_ROUTE,
        )
    ):
        raise ValueError("source/evidence provenance mismatch")
    _string(source.get("harness_mode"), "source harness mode")
    _sha(source.get("harness_sha256"), "source harness")
    return identity, source


def _normalize_messages(
    record: dict[str, Any], proof: dict[str, Any], contract: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Normalize exact aliases and keep only a complete, successful prefix."""
    tools, aliases = tool_contract(contract)
    schemas = {tool["function"]["name"]: tool["function"]["parameters"] for tool in tools}
    raw = copy.deepcopy(record.get("messages"))
    if (
        not isinstance(raw, list)
        or len(raw) < 4
        or not all(isinstance(message, dict) for message in raw)
        or [message.get("role") for message in raw[:2]] != ["system", "user"]
        or not all(
            isinstance(raw[index].get("content"), str) and raw[index]["content"] for index in (0, 1)
        )
    ):
        raise Excluded("missing_original_system_and_task_anchor")
    successful_report = _string(proof.get("successful_report_call_id"), "successful report call id")
    pending: dict[str, tuple[int, str]] = {}
    seen: set[str] = set()
    operations: list[dict[str, Any]] = []
    completed: set[str] = set()
    cut = len(raw)
    issue: str | None = None
    successful_result_index: int | None = None
    report_calls = 0
    for index, message in enumerate(raw):
        if not isinstance(message, dict) or message.get("role") not in {
            "system",
            "user",
            "assistant",
            "tool",
        }:
            cut, issue = index, "unsupported_message_role"
            break
        role = message["role"]
        if index >= 2 and role in {"system", "user"}:
            cut, issue = index, "unproven_additional_system_or_user"
            break
        if role != "tool" and pending:
            cut, issue = min(item[0] for item in pending.values()), "missing_tool_result"
            break
        calls = message.get("tool_calls") or []
        if role != "assistant" and calls:
            cut, issue = index, "tool_call_outside_assistant"
            break
        if role == "assistant" and any(
            message.get(field) for field in ("thinking", "reasoning", "reasoning_content")
        ):
            raise Excluded("private_reasoning_field")
        local: list[tuple[str, str]] = []
        for call in calls:
            function = call.get("function") if isinstance(call, dict) else None
            call_id = call.get("id") if isinstance(call, dict) else None
            if (
                not isinstance(call_id, str)
                or not call_id
                or call_id in seen
                or not isinstance(function, dict)
            ):
                cut, issue = index, "invalid_tool_call"
                break
            arguments = _arguments(function)
            source_name = function.get("name")
            canonical, arguments = _canonical_source_tool(source_name, arguments)
            target = aliases[canonical]
            _schema_value(arguments, schemas[target])
            if target == "fleet_submit_report":
                report_calls += 1
                if report_calls != 1 or call_id != successful_report:
                    raise Excluded("unbound_or_multiple_report_submission")
            operations.append(
                {
                    "source_message_index": index,
                    "call_id": call_id,
                    "from": source_name,
                    "to": target,
                }
            )
            function["name"], function["arguments"] = target, arguments
            local.append((call_id, target))
        if issue:
            break
        if any(target == "fleet_submit_report" for _, target in local) and len(local) != 1:
            raise Excluded("successful_report_must_be_one_complete_tool_round")
        for call_id, target in local:
            seen.add(call_id)
            pending[call_id] = (index, target)
        if role == "tool":
            call_id = message.get("tool_call_id")
            if call_id not in pending:
                cut, issue = index, "orphan_or_duplicate_tool_result"
                break
            pending.pop(call_id)
            completed.add(call_id)
            if call_id == successful_report:
                if pending:
                    raise Excluded("successful_report_must_be_one_complete_tool_round")
                successful_result_index = index
                cut = index + 1
                break
        if successful_result_index is not None:
            break
    if issue is None and pending:
        cut, issue = min(item[0] for item in pending.values()), "missing_terminal_tool_result"
    retained = raw[:cut]
    if successful_result_index is not None:
        for message in raw[cut:]:
            for call in message.get("tool_calls") or []:
                function = call.get("function") if isinstance(call, dict) else None
                if not isinstance(function, dict):
                    continue
                with_arguments = function.get("arguments")
                try:
                    with_arguments = (
                        json.loads(with_arguments)
                        if isinstance(with_arguments, str)
                        else with_arguments
                    )
                except (TypeError, ValueError):
                    continue
                if not isinstance(with_arguments, dict):
                    continue
                try:
                    canonical, _ = _canonical_source_tool(function.get("name"), with_arguments)
                except Excluded:
                    continue
                if canonical == "submit_report":
                    raise Excluded("unbound_or_multiple_report_submission")
    retained_calls = {
        call["id"]: (call.get("function") or {}).get("name")
        for message in retained
        for call in message.get("tool_calls") or []
    }
    if (
        retained_calls.get(successful_report) != "fleet_submit_report"
        or successful_report not in completed
    ):
        reason = (
            "lifecycle_salvage_lost_successful_report" if issue else "missing_successful_report"
        )
        raise Excluded(reason)
    if not any(message.get("role") == "assistant" for message in retained):
        raise Excluded("no_assistant_targets")
    visible = [_normalized_for_template(message) for message in retained]
    transform = {
        "schema": "cyber_message_aligned_source_transform_v1",
        "input_messages_sha256": digest_json(record["messages"]),
        "tool_contract_sha256": contract["sha256"],
        "exact_tool_alias_operations": [
            item for item in operations if item["source_message_index"] < cut
        ],
        "lifecycle_prefix_salvage": (
            None
            if issue is None
            else {
                "reason": issue,
                "kept_messages": cut,
                "dropped_messages": len(raw) - cut,
            }
        ),
        "terminal_report_cut": (
            None
            if successful_result_index is None or cut == len(raw)
            else {"kept_messages": cut, "dropped_messages": len(raw) - cut}
        ),
        "successful_report_call_id_sha256": digest_json(successful_report),
        "output_messages_sha256": digest_json(visible),
    }
    transform["sha256"] = digest_json(transform)
    return visible, transform


def _selection_row(
    record: dict[str, Any],
    proof: dict[str, Any],
    rows: list[dict[str, Any]],
    group_id: str,
    transform: dict[str, Any],
) -> dict[str, Any]:
    """Preserve the qualified broad-corpus task-rich source gate."""
    assistants = [message for message in record["messages"] if message["role"] == "assistant"]
    eligible = {span["assistant_index"] for row in rows for span in row["target_spans"]}
    if len(eligible) != sum(len(row["target_spans"]) for row in rows):
        raise ValueError("one source assistant target appeared more than once")
    counts: Counter[str] = Counter()
    for index in sorted(eligible):
        names = [
            (call.get("function") or {}).get("name")
            for call in assistants[index].get("tool_calls") or []
        ]
        if not names and assistants[index].get("content"):
            counts["decision_responses"] += 1
        elif names and set(names) == {"fleet_bash"}:
            counts["non_submit_tool_responses"] += 1
            counts["completed_non_submit_tool_rounds"] += 1
        elif names == ["fleet_submit_report"]:
            counts["submit_report_responses"] += 1
        else:
            counts["other_responses"] += 1
    submit_indices = {
        index
        for index, message in enumerate(assistants)
        if [(call.get("function") or {}).get("name") for call in message.get("tool_calls") or []]
        == ["fleet_submit_report"]
    }
    submit_tokens = sum(
        span["token_end"] - span["token_start"]
        for row in rows
        for span in row["target_spans"]
        if span["assistant_index"] in submit_indices
    )
    supervised = sum(row["target_token_count"] for row in rows)
    assistant_count = len(eligible)
    if not (
        counts["non_submit_tool_responses"] >= 1
        and counts["completed_non_submit_tool_rounds"] >= 1
        and assistant_count > counts["submit_report_responses"] >= 0
        and counts["submit_report_responses"] / assistant_count <= 0.5
        and supervised > 0
        and submit_tokens / supervised <= 0.5
    ):
        raise Excluded("insufficient_task_rich_targets")
    source = record["source"]
    return {
        "session_id": record["record_id"],
        "task_key": record["lineage"]["task_key"],
        "task_version_id": record["lineage"]["eval_task_version_id"],
        "group_id": group_id,
        "model_id": source["model"],
        "harness_mode": source["harness_mode"],
        "harness_sha256": source["harness_sha256"],
        "trace_sha256": proof["transcript_sha256"],
        "acceptance_sha256": proof["sha256"],
        "normalized_record_sha256": proof["normalized_record_sha256"],
        "transform_sha256": transform["sha256"],
        "windows": len(rows),
        "supervised_tokens": supervised,
        "assistant_responses": assistant_count,
        "decision_responses": counts["decision_responses"],
        "non_submit_tool_responses": counts["non_submit_tool_responses"],
        "completed_non_submit_tool_rounds": counts["completed_non_submit_tool_rounds"],
        "submit_report_responses": counts["submit_report_responses"],
        "submit_report_tokens": submit_tokens,
        "other_responses": counts["other_responses"],
        "terminal_report_cut": transform["terminal_report_cut"],
        "lifecycle_prefix_salvage": transform["lifecycle_prefix_salvage"],
    }


def materialize_record(
    record: dict[str, Any],
    proof: dict[str, Any],
    roles: dict[tuple[str, str], dict[str, str]],
    contract: dict[str, Any],
    tokenizer: Any,
    helper: Any,
    *,
    max_length: int,
    context_tokens: int,
    family_role_anchor_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Render one verified success into anchored, message-aligned windows."""
    identity = (
        record.get("lineage", {}).get("task_key"),
        record.get("lineage", {}).get("eval_task_version_id"),
    )
    role = roles.get(identity)
    if role is None:
        raise ValueError("source identity lacks reviewed task-family lineage")
    if role["split"] != "train":
        raise ValueError("held-out task-family alias reached training materialization")
    if proof.get("normalized_record_sha256") != digest_json(record):
        raise ValueError("success evidence does not bind the normalized record")
    messages, transform = _normalize_messages(record, proof, contract)
    transformed = copy.deepcopy(record)
    transformed["messages"] = messages
    tools, _ = tool_contract(contract)
    anchor, chunks = encode_record(messages, tokenizer, helper, tools=tools)
    rows = segment_record(
        transformed, anchor, chunks, max_tokens=max_length, context_budget=context_tokens
    )
    positions = {chunk["message_index"]: index for index, chunk in enumerate(chunks)}
    for row in rows:
        start = positions.get(row["context_start_message_index"])
        target_messages = [span["source_message_index"] for span in row["target_spans"]]
        end = positions.get(max(target_messages)) if target_messages else None
        expected_body = (
            [token for chunk in chunks[start : end + 1] for token in chunk["ids"]]
            if start is not None and end is not None
            else None
        )
        selected_messages = (
            messages[:2] + messages[row["context_start_message_index"] : max(target_messages) + 1]
        )
        try:
            full_render = tokenizer.apply_chat_template(
                selected_messages,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=False,
                tools=tools,
            )
            if isinstance(full_render, Mapping):
                full_render = full_render["input_ids"]
            full_render = list(full_render)
        except Exception:
            raise Excluded("full_template_render_contract") from None
        if (
            row["input_ids"][: len(anchor)] != anchor
            or row["input_ids"][len(anchor) :] != expected_body
            or row["input_ids"] != full_render
        ):
            raise ValueError(
                "window does not start at its complete repeated anchor/message boundary"
            )
        row.update(
            {
                "window_schema": WINDOW_SCHEMA,
                "window_algorithm": ALGORITHM,
                "source_task_version_id": identity[1],
                "source_group_id": role["group_id"],
                "family_role_anchor_sha256": family_role_anchor_sha256,
                "tool_contract_sha256": contract["sha256"],
                "anchor_sha256": digest_json(anchor),
                "source_messages_sha256": transform["output_messages_sha256"],
                "source_transform_sha256": transform["sha256"],
            }
        )
    return rows, transform, _selection_row(transformed, proof, rows, role["group_id"], transform)


def _evidence(path: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for proof in iter_jsonl(path):
        _sealed(proof, EVIDENCE_SCHEMA)
        if set(proof) != {
            "schema",
            "session_id",
            "model_id",
            "task_key",
            "task_version_id",
            "normalized_record_sha256",
            "transcript_sha256",
            "verifier_execution_id",
            "routes",
            "successful_report_call_id",
            "outcome",
            "sha256",
        }:
            raise ValueError("success evidence has unexpected fields")
        session = _string(proof.get("session_id"), "evidence session")
        outcome = proof.get("outcome")
        if (
            session in result
            or not isinstance(outcome, dict)
            or set(outcome) != {"status", "verifier_process_success", "score_at_least_one"}
            or outcome.get("status") != "completed"
            or outcome.get("verifier_process_success") is not True
            or outcome.get("score_at_least_one") is not True
        ):
            raise ValueError("success evidence is duplicate or not authoritative success")
        _sha(proof.get("normalized_record_sha256"), "normalized record")
        _sha(proof.get("transcript_sha256"), "transcript")
        for field in (
            "model_id",
            "task_key",
            "task_version_id",
            "verifier_execution_id",
        ):
            _string(proof.get(field), f"evidence {field}")
        routes = proof.get("routes")
        if (
            not isinstance(routes, dict)
            or set(routes) != {"summary", "transcript"}
            or not isinstance(routes.get("summary"), str)
            or not routes["summary"]
            or routes.get("transcript") != TRANSCRIPT_ROUTE
        ):
            raise ValueError("success evidence routes differ from the broad export")
        _string(proof.get("successful_report_call_id"), "successful report call id")
        result[session] = proof
    return result


def _arrow_schema(pa: Any) -> Any:
    span = pa.struct(
        [
            ("assistant_index", pa.int64()),
            ("source_message_index", pa.int64()),
            ("token_start", pa.int64()),
            ("token_end", pa.int64()),
            ("source_target_sha256", pa.string()),
        ]
    )
    excluded = pa.struct(
        [
            ("assistant_index", pa.int64()),
            ("source_message_index", pa.int64()),
            ("reason", pa.string()),
        ]
    )
    return pa.schema(
        [
            ("input_ids", pa.list_(pa.int64())),
            ("loss_mask", pa.list_(pa.int64())),
            ("token_count", pa.int64()),
            ("target_token_count", pa.int64()),
            ("task_key", pa.string()),
            ("window_id", pa.string()),
            ("segment_id", pa.string()),
            ("source_session_id", pa.string()),
            ("source_model", pa.string()),
            ("source_assistant_count", pa.int64()),
            ("eligible_assistant_indices", pa.list_(pa.int64())),
            ("excluded_assistant_targets", pa.list_(excluded)),
            ("target_spans", pa.list_(span)),
            ("copied_context_assistant_indices", pa.list_(pa.int64())),
            ("context_start_message_index", pa.int64()),
            ("split", pa.string()),
            ("source_harness_mode", pa.string()),
            ("source_harness_sha256", pa.string()),
            ("source_acceptance_sha256", pa.string()),
            ("source_trace_sha256", pa.string()),
            ("source_normalized_record_sha256", pa.string()),
            ("window_schema", pa.string()),
            ("window_algorithm", pa.string()),
            ("source_task_version_id", pa.string()),
            ("source_group_id", pa.string()),
            ("family_role_anchor_sha256", pa.string()),
            ("tool_contract_sha256", pa.string()),
            ("anchor_sha256", pa.string()),
            ("source_messages_sha256", pa.string()),
            ("source_transform_sha256", pa.string()),
        ]
    )


def build(config: dict[str, Any], *, relative_to: Path) -> dict[str, Any]:
    """Create one immutable corpus only after every evidence binding exists."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    _sealed(config, REQUEST_SCHEMA)
    fields = {
        "schema",
        "normalized",
        "evidence",
        "family_roster",
        "tool_catalog",
        "tool_contract",
        "model_request_capture",
        "model_lock",
        "tokenizer_root",
        "native_helper",
        "teacher_models",
        "max_length",
        "context_tokens",
        "output",
        "sha256",
    }
    if set(config) != fields:
        raise ValueError("materialization request has unexpected fields")
    paths = {
        name: _input(relative_to, config[name], name)
        for name in (
            "normalized",
            "evidence",
            "family_roster",
            "tool_catalog",
            "tool_contract",
            "model_request_capture",
            "model_lock",
            "native_helper",
        )
    }
    roster_value = _json(paths["family_roster"], "family roster")
    roles = family_roster(roster_value)
    derived_tools = derive_opencode_tools(_json_value(paths["tool_catalog"], "tool catalog"))
    contract = _json(paths["tool_contract"], "tool contract")
    contract_tools, _ = tool_contract(contract)
    if contract_tools != derived_tools:
        raise ValueError("tool contract differs from the bound source catalog transform")
    capture = _json(paths["model_request_capture"], "model request capture")
    model_request_capture(capture, contract, contract_tools)
    proofs = _evidence(paths["evidence"])
    teachers = config["teacher_models"]
    if (
        not isinstance(teachers, list)
        or not teachers
        or any(not isinstance(model, str) or not model for model in teachers)
        or len(teachers) != len(set(teachers))
    ):
        raise ValueError("explicit unique teacher model list is required")
    maximum, context = config["max_length"], config["context_tokens"]
    if (
        type(maximum) is not int
        or type(context) is not int
        or maximum < 2
        or not 0 <= context <= maximum
    ):
        raise ValueError("invalid message-aligned window bounds")
    output = Path(_string(config["output"], "output"))
    output = output if output.is_absolute() else relative_to / output
    if output.exists() or output.is_symlink():
        raise FileExistsError("create-once corpus destination exists")
    tokenizer_root = Path(_string(config["tokenizer_root"], "tokenizer root"))
    tokenizer_root = (
        tokenizer_root if tokenizer_root.is_absolute() else relative_to / tokenizer_root
    )
    model_lock = read_mapping(paths["model_lock"])
    tokenizer, tokenizer_identity = local_tokenizer(model_lock, tokenizer_root)
    if contract["target_model"] != {
        "repo": tokenizer_identity["repo"],
        "revision": tokenizer_identity["revision"],
    }:
        raise ValueError("tool contract model differs from the corpus tokenizer")
    helper = native_helper(paths["native_helper"])
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(tempfile.mkdtemp(prefix=f".{output.name}.partial-", dir=output.parent))
    os.chmod(partial, 0o700)
    writer = None
    selected, exclusions, seen, task_keys, task_versions = [], Counter(), set(), set(), set()
    trajectories, window_payloads, written_row_digests = set(), set(), []
    models, harnesses, token_lengths = Counter(), Counter(), []
    counts = Counter()
    try:
        writer = pq.ParquetWriter(partial / "train.parquet", _arrow_schema(pa), compression="zstd")
        records = sorted(
            iter_jsonl(paths["normalized"]), key=lambda item: item.get("record_id", "")
        )
        for record in records:
            session = _string(record.get("record_id"), "source session")
            if session in seen:
                raise ValueError("normalized source duplicates a session identity")
            seen.add(session)
            proof = proofs.get(session)
            if proof is None:
                raise ValueError("normalized source lacks exact success evidence")
            identity, source = _source_provenance(record, proof)
            if not _verified_success(record, proof, set(teachers)):
                exclusions["not_selected_verified_teacher_success"] += 1
                continue
            try:
                rows, transform, selection = materialize_record(
                    record,
                    proof,
                    roles,
                    contract,
                    tokenizer,
                    helper,
                    max_length=maximum,
                    context_tokens=context,
                    family_role_anchor_sha256=roster_value["family_role_anchor_sha256"],
                )
            except Excluded as error:
                exclusions[error.reason] += 1
                continue
            trajectory = rows[0]["source_messages_sha256"]
            if trajectory in trajectories:
                exclusions["exact_trajectory_duplicate"] += 1
                continue
            fingerprints = [digest_json([row["input_ids"], row["loss_mask"]]) for row in rows]
            if len(fingerprints) != len(set(fingerprints)) or any(
                fingerprint in window_payloads for fingerprint in fingerprints
            ):
                exclusions["exact_window_payload_duplicate"] += 1
                continue
            for row in rows:
                row.update(
                    {
                        "source_harness_mode": source["harness_mode"],
                        "source_harness_sha256": source["harness_sha256"],
                        "source_acceptance_sha256": proof["sha256"],
                        "source_trace_sha256": proof["transcript_sha256"],
                        "source_normalized_record_sha256": proof["normalized_record_sha256"],
                    }
                )
            local = {
                "rows": len(rows),
                "task_keys": [rows[0]["task_key"]],
                "format": DENSE_FORMAT,
                "source_sessions": 1,
                "supervised_tokens": sum(row["target_token_count"] for row in rows),
                "assistant_responses": sum(len(row["target_spans"]) for row in rows),
                "source_total_assistant_responses": rows[0]["source_assistant_count"],
                "excluded_assistant_responses": len(rows[0]["excluded_assistant_targets"]),
            }
            dense_rows(rows, local, max_length=maximum, vocab_size=len(tokenizer))
            writer.write_table(pa.Table.from_pylist(rows, schema=_arrow_schema(pa)))
            written_row_digests.extend(digest_json(row) for row in rows)
            trajectories.add(trajectory)
            window_payloads.update(fingerprints)
            selected.append(selection)
            task_keys.add(rows[0]["task_key"])
            task_versions.add(identity)
            models[source["model"]] += 1
            harnesses[source["harness_mode"]] += 1
            token_lengths.extend(row["token_count"] for row in rows)
            if transform["lifecycle_prefix_salvage"]:
                counts["lifecycle_prefix_salvages"] += 1
                counts["lifecycle_prefix_dropped_messages"] += transform[
                    "lifecycle_prefix_salvage"
                ]["dropped_messages"]
            if transform["terminal_report_cut"]:
                counts["terminal_report_cuts"] += 1
                counts["terminal_report_dropped_messages"] += transform["terminal_report_cut"][
                    "dropped_messages"
                ]
            counts.update({key: value for key, value in local.items() if isinstance(value, int)})
        completed_writer, writer = writer, None
        completed_writer.close()
        if set(proofs) != seen or not selected:
            raise ValueError("source/evidence are not one-to-one or corpus is empty")
        os.chmod(partial / "train.parquet", 0o600)
        readback = [
            digest_json(row)
            for batch in pq.ParquetFile(partial / "train.parquet").iter_batches()
            for row in batch.to_pylist()
        ]
        if readback != written_row_digests:
            raise ValueError("Parquet readback differs from the validated dense rows")
        atomic_write_jsonl(partial / "source-selection.private.jsonl", selected, private=True)
        train = {
            "path": "train.parquet",
            "sha256": file_sha256(partial / "train.parquet"),
            "rows": counts["rows"],
            "task_keys": sorted(task_keys),
            "format": DENSE_FORMAT,
            "source_sessions": counts["source_sessions"],
            "supervised_tokens": counts["supervised_tokens"],
            "assistant_responses": counts["assistant_responses"],
            "source_total_assistant_responses": counts["source_total_assistant_responses"],
            "excluded_assistant_responses": counts["excluded_assistant_responses"],
        }
        token_lengths.sort()

        def percentile(p: float) -> int:
            return token_lengths[min(len(token_lengths) - 1, math.ceil(p * len(token_lengths)) - 1)]

        manifest = {
            "schema": CORPUS_SCHEMA,
            "algorithm": ALGORITHM,
            "source_sha256": file_sha256(partial / "source-selection.private.jsonl"),
            "split_sha256": roster_value["family_role_anchor_sha256"],
            "tokenizer": tokenizer_identity,
            "files": {"train": train},
            "train_models": sorted(models),
            "max_length": maximum,
            "context_tokens": context,
            "dev_windows": 0,
            "validation_mode": "task_outcomes_only",
            "whole_source_exclusions": dict(sorted(exclusions.items())),
            "catalog_provenance": {
                "selected_task_keys": len(task_keys),
                "selected_task_versions": len(task_versions),
                "selected_source_sessions": len(selected),
                "teacher_model_sessions": dict(sorted(models.items())),
                "harness_mode_sessions": dict(sorted(harnesses.items())),
                "lifecycle_prefix_salvages": counts["lifecycle_prefix_salvages"],
                "lifecycle_prefix_dropped_messages": counts["lifecycle_prefix_dropped_messages"],
                "terminal_report_cuts": counts["terminal_report_cuts"],
                "terminal_report_dropped_messages": counts["terminal_report_dropped_messages"],
                "transcript_route": TRANSCRIPT_ROUTE,
            },
            "token_length_distribution": {
                "min": token_lengths[0],
                "p50": percentile(0.50),
                "p90": percentile(0.90),
                "p99": percentile(0.99),
                "max": token_lengths[-1],
            },
            "materialization": {
                "schema": MATERIALIZATION_SCHEMA,
                "request_sha256": config["sha256"],
                "normalized_sha256": config["normalized"]["sha256"],
                "success_evidence_sha256": config["evidence"]["sha256"],
                "family_roster_sha256": roster_value["sha256"],
                "family_role_anchor_sha256": roster_value["family_role_anchor_sha256"],
                "source_tool_catalog_file_sha256": config["tool_catalog"]["sha256"],
                "source_tool_catalog_sha256": SOURCE_TOOL_CATALOG_SHA256,
                "target_tools_sha256": TARGET_TOOLS_SHA256,
                "tool_contract_sha256": contract["sha256"],
                "model_request_capture_sha256": capture["sha256"],
                "token_splicing": False,
                "every_window_repeats_system_task_and_tool_schema": True,
            },
            "builder_sha256": {
                "message_aligned_teacher_corpus.py": file_sha256(Path(__file__)),
                "dense.py": file_sha256(Path(__file__).with_name("dense.py")),
                "corpus.py": file_sha256(Path(__file__).with_name("corpus.py")),
                "native_helper": file_sha256(paths["native_helper"]),
            },
            "limitations": [
                "Visible assistant actions only; private reasoning is omitted.",
                "Every selected source is an exact transcript-verified stronger-teacher success.",
                "Task-level QA certification was not part of broad-v1 and is not inferred; "
                "the reviewed family role and exact transcript outcome are the admission evidence.",
                "Every reviewed dev/final-test family is excluded across its exact aliases.",
                "Malformed lifecycle suffixes are retained only when the evidence-bound "
                "successful report remains complete.",
            ],
        }
        manifest["sha256"] = "sha256:" + digest(manifest)
        atomic_write_json(partial / "manifest.json", manifest, private=True)
        receipt = {
            "schema": "cyber_message_aligned_teacher_corpus_receipt_v1",
            "manifest_file_sha256": file_sha256(partial / "manifest.json"),
            "manifest_sha256": manifest["sha256"],
            "train_parquet_sha256": file_sha256(partial / "train.parquet"),
            "source_selection_sha256": file_sha256(partial / "source-selection.private.jsonl"),
            "rows": train["rows"],
            "source_sessions": train["source_sessions"],
            "task_versions": len(task_versions),
            "supervised_tokens": train["supervised_tokens"],
        }
        receipt["sha256"] = "sha256:" + digest(receipt)
        atomic_write_json(partial / "RECEIPT.json", receipt, private=True)
        if any(file_sha256(path) != config[name]["sha256"] for name, path in paths.items()):
            raise ValueError("a bound materialization input changed during the build")
        for entry in model_lock["tokenizer"]["files"]:
            if file_sha256(tokenizer_root / entry["path"]) != "sha256:" + entry[
                "sha256"
            ].removeprefix("sha256:"):
                raise ValueError("a tokenizer input changed during the build")
        _rename_noreplace(partial, output)
        return {"manifest": manifest["sha256"], "receipt": receipt["sha256"], "train": train}
    except BaseException:
        if writer is not None:
            writer.close()
        shutil.rmtree(partial, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    result = build(read_mapping(args.config), relative_to=args.config.parent)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
