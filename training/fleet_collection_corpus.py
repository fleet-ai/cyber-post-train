"""Build a private dense SFT corpus from sealed Fleet collection evidence.

This is the second half of the generic collection path.  Unlike
``fleet_collection_admission``, it reads private normalized records, but it
does so only after binding every record to both the metadata-only admission
selection and the immutable collection packet.  It never contacts Fleet,
starts a workload, or prints a prompt, trace, tool result, token array, or
private reasoning.

Only the visible-action packet schema is accepted here.  A future
student-visible-reasoning corpus must use a distinct packet and materializer;
it cannot opt into this path with a request flag.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from cyber_post_train.jobs import digest

from . import collection_campaign
from . import fleet_collection_admission as admission
from .corpus import local_tokenizer
from .dense import Excluded, compatible_messages, encode_record, native_helper, segment_record
from .io import atomic_write_json, atomic_write_jsonl, digest_json, file_sha256, iter_jsonl
from .sft import _known, read_mapping
from .sft_runtime import DENSE_FORMAT, dense_rows
from .task_family_split import is_supported_schema
from .task_family_split import validate as validate_split

SCHEMA = "cyber_fleet_private_corpus_materialization_request_v1"
RECORD_SCHEMA = "cyber_fleet_visible_action_record_v1"
COVERAGE_SCHEMA = "cyber_fleet_collection_coverage_v1"
RECEIPT_SCHEMA = "cyber_fleet_collection_materialization_receipt_v1"

_SHA256 = re.compile(r"sha256:[0-9a-f]{64}")
_PRIVATE_REASONING_MARKER = re.compile(r"<\s*/?\s*think\b", re.IGNORECASE)
_OPAQUE_TOOL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}")
_POLICY = {
    "target_mode": "visible_actions_only",
    "reasoning_visibility": "absent",
    "compaction": "none",
}
_EXPECTED_PACKET_POLICY = {
    "eligible_terminal_outcome": "verified_success_v1",
    "reasoning_policy": collection_campaign.VISIBLE_ACTIONS_ONLY,
    "offline_compaction_policy": collection_campaign.OPAQUE_COMPACTION_REJECT,
    "minimum_non_submit_tool_responses": 1,
    "minimum_completed_non_submit_tool_rounds": 1,
    "maximum_submit_report_response_fraction": 0.5,
    "maximum_submit_report_target_token_fraction": 0.5,
    "maximum_family_target_token_fraction": 0.25,
    "deduplication_order": [
        "source_session_identity",
        "normalized_trajectory_digest",
        "packed_window_payload_digest",
    ],
    "held_out_roles_excluded": ["dev", "final_test"],
    "adapter_must_bind": ["collection_packet_sha256", "eval_plan_sha256"],
}
_EXPECTED_PACKET_METRICS = [
    "valid_success_count",
    "opaque_compaction_rejection_count",
    "private_or_unknown_reasoning_rejection_count",
    "distinct_visible_action_target_tokens",
    "family_token_concentration",
]


def _sealed(value: Mapping[str, Any], schema: str) -> None:
    if value.get("schema") != schema or value.get("sha256") != "sha256:" + digest(
        {key: item for key, item in value.items() if key != "sha256"}
    ):
        raise ValueError(f"invalid sealed {schema}")


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"exact {label} digest is required")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"nonempty {label} is required")
    return value


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _path(root: Path, value: object, label: str) -> Path:
    path = Path(_string(value, label))
    return path if path.is_absolute() else root / path


def _input(root: Path, value: object, label: str) -> tuple[Path, str]:
    reference = _mapping(value, label)
    _known(reference, {"path", "sha256"}, label)
    path = _path(root, reference.get("path"), f"{label} path")
    expected = _sha(reference.get("sha256"), f"{label} file")
    if path.is_symlink() or not path.is_file() or file_sha256(path) != expected:
        raise ValueError(f"{label} file digest mismatch")
    return path, expected


def _json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label} JSON") from error
    return _mapping(value, label)


def _selection(value: dict[str, Any]) -> dict[str, dict[str, Any]]:
    _sealed(value, admission.SELECTION_SCHEMA)
    required = {
        "schema",
        "artifact_kind",
        "trainable_corpus_created",
        "parquet_created",
        "source_text_read",
        "next_required_gate",
        "campaign_plan_sha256",
        "catalog_inventory_sha256",
        "family_split_sha256",
        "protected_family_lock_sha256",
        "source_kind",
        "source_model_alias",
        "source_model",
        "harness_treatment_sha256",
        "tool_catalog_sha256",
        "template_sha256",
        "max_sessions_per_task_version",
        "selected",
        "sha256",
    }
    if set(value) != required or value["artifact_kind"] != admission.HANDOFF_KIND:
        raise ValueError("admission selection has an unexpected contract")
    metadata_only = ("trainable_corpus_created", "parquet_created", "source_text_read")
    if any(value[field] is not False for field in metadata_only):
        raise ValueError("admission selection is not metadata-only")
    if value["next_required_gate"] != (
        "bind_private_normalized_records_then_run_existing_dense_sft_builder"
    ):
        raise ValueError("admission selection has an unexpected downstream gate")
    for field in (
        "campaign_plan_sha256",
        "catalog_inventory_sha256",
        "family_split_sha256",
        "protected_family_lock_sha256",
        "harness_treatment_sha256",
        "tool_catalog_sha256",
        "template_sha256",
    ):
        _sha(value.get(field), f"admission selection {field}")
    model = _mapping(value.get("source_model"), "admission selection model")
    if set(model) != {"repository", "revision", "session_model"} or any(
        not isinstance(model.get(field), str) or not model[field]
        for field in ("repository", "revision", "session_model")
    ):
        raise ValueError("admission selection model is malformed")
    if (
        value.get("source_kind") not in {"teacher", "self"}
        or not isinstance(value.get("source_model_alias"), str)
        or not value["source_model_alias"]
        or type(value.get("max_sessions_per_task_version")) is not int
        or value["max_sessions_per_task_version"] < 1
    ):
        raise ValueError("admission selection source policy is malformed")
    rows = value.get("selected")
    if not isinstance(rows, list) or not rows:
        raise ValueError("admission selection must contain selected private records")
    result: dict[str, dict[str, Any]] = {}
    seen_trajectories: set[str] = set()
    fields = {
        "session_id",
        "campaign_plan_sha256",
        "cell_sha256",
        "task_key",
        "task_version_id",
        "group_id",
        "attempt",
        "source_kind",
        "model",
        "harness",
        "template_sha256",
        "normalized_record_sha256",
        "normalized_trajectory_sha256",
        "transcript_sha256",
    }
    for row in rows:
        if not isinstance(row, dict) or set(row) != fields:
            raise ValueError("admission selection row is malformed")
        session = _string(row.get("session_id"), "selected session")
        if session in result or type(row.get("attempt")) is not int or row["attempt"] < 1:
            raise ValueError("admission selection has duplicate or invalid sessions")
        if row["campaign_plan_sha256"] != value["campaign_plan_sha256"]:
            raise ValueError("admission selection row changes campaign binding")
        if row["source_kind"] != value["source_kind"] or row["model"] != model:
            raise ValueError("admission selection row changes source model binding")
        if row["template_sha256"] != value["template_sha256"]:
            raise ValueError("admission selection row changes template binding")
        harness = _mapping(row.get("harness"), "selected session harness")
        _known(harness, {"treatment_sha256", "tool_catalog_sha256"}, "selected session harness")
        if (
            harness.get("treatment_sha256") != value["harness_treatment_sha256"]
            or harness.get("tool_catalog_sha256") != value["tool_catalog_sha256"]
        ):
            raise ValueError("admission selection row changes harness binding")
        for field in (
            "cell_sha256",
            "normalized_record_sha256",
            "normalized_trajectory_sha256",
            "transcript_sha256",
        ):
            _sha(row.get(field), f"selected {field}")
        if row["normalized_trajectory_sha256"] in seen_trajectories:
            raise ValueError("admission selection contains a duplicate trajectory")
        seen_trajectories.add(row["normalized_trajectory_sha256"])
        _string(row.get("task_key"), "selected task key")
        _string(row.get("task_version_id"), "selected task version")
        _string(row.get("group_id"), "selected group")
        result[session] = row
    return result


def _admission_receipt(value: dict[str, Any], selection: dict[str, Any]) -> None:
    _sealed(value, admission.RECEIPT_SCHEMA)
    required = {
        "schema",
        "artifact_kind",
        "trainable_corpus_created",
        "parquet_created",
        "source_text_read",
        "next_required_gate",
        "campaign_plan_sha256",
        "input_file_sha256",
        "catalog_inventory_sha256",
        "family_split_sha256",
        "protected_family_lock_sha256",
        "protected_family_count",
        "source_kind",
        "source_model_alias",
        "source_model_identity_sha256",
        "harness_treatment_sha256",
        "template_sha256",
        "tool_catalog_sha256",
        "max_sessions_per_task_version",
        "counts",
        "policy",
        "selection_sha256",
        "sha256",
    }
    if set(value) != required or value.get("selection_sha256") != selection["sha256"]:
        raise ValueError("admission receipt does not bind the private selection")
    if (
        value.get("artifact_kind") != admission.HANDOFF_KIND
        or any(
            value.get(field) is not False
            for field in ("trainable_corpus_created", "parquet_created", "source_text_read")
        )
        or value.get("next_required_gate")
        != "bind_private_normalized_records_then_run_existing_dense_sft_builder"
    ):
        raise ValueError("admission receipt is not the exact metadata-only handoff")
    for field in (
        "campaign_plan_sha256",
        "catalog_inventory_sha256",
        "family_split_sha256",
        "protected_family_lock_sha256",
        "harness_treatment_sha256",
        "template_sha256",
        "tool_catalog_sha256",
    ):
        if value.get(field) != selection.get(field):
            raise ValueError("admission receipt binding differs from selection")
    if (
        value.get("source_kind") != selection["source_kind"]
        or value.get("source_model_alias") != selection["source_model_alias"]
        or value.get("source_model_identity_sha256")
        != "sha256:" + digest(selection["source_model"])
        or value.get("tool_catalog_sha256") != selection["tool_catalog_sha256"]
        or value.get("max_sessions_per_task_version") != selection["max_sessions_per_task_version"]
    ):
        raise ValueError("admission receipt source binding differs from selection")
    input_files = _mapping(value.get("input_file_sha256"), "admission receipt input files")
    if set(input_files) != {
        "campaign",
        "inventory",
        "family_split",
        "protected_family_lock",
        "attempts",
    }:
        raise ValueError("admission receipt input-file bindings are incomplete")
    for name, value_sha in input_files.items():
        _sha(value_sha, f"admission receipt input file {name}")
    counts = _mapping(value.get("counts"), "admission receipt counts")
    expected_counts = {
        "attempt_metadata_records",
        "planned_cells_for_source_model",
        "train_cells_for_source_model",
        "admitted_sessions",
        "admitted_task_versions",
        "rejections",
    }
    if set(counts) != expected_counts or any(
        type(counts.get(field)) is not int or counts[field] < 0
        for field in expected_counts - {"rejections"}
    ):
        raise ValueError("admission receipt counts are malformed")
    rejections = _mapping(counts["rejections"], "admission receipt rejections")
    if set(rejections) != set(admission._REJECTION_REASONS) or any(
        type(count) is not int or count < 0 for count in rejections.values()
    ):
        raise ValueError("admission receipt rejection counts are malformed")
    selected = selection["selected"]
    if (
        counts["admitted_sessions"] != len(selected)
        or counts["admitted_task_versions"]
        != len({(row["task_key"], row["task_version_id"]) for row in selected})
        or counts["train_cells_for_source_model"] > counts["planned_cells_for_source_model"]
    ):
        raise ValueError("admission receipt counts differ from its private selection")
    policy = _mapping(value.get("policy"), "admission receipt policy")
    required_policy = {
        "training_data_eligible_required": True,
        "success_requires_completed_authoritative_verifier": True,
        "visible_actions_only": True,
        "private_or_unknown_reasoning_rejected": True,
        "opaque_or_unapproved_compaction_rejected": True,
        "exact_session_and_trajectory_deduplication": True,
        "per_task_version_cap_is_deterministic": True,
        "heldout_families_admitted": 0,
        "raw_source_payload_read": False,
    }
    if policy != required_policy:
        raise ValueError("admission receipt policy is insufficient for corpus materialization")


def _packet(
    value: dict[str, Any],
    selection: dict[str, Any],
    task_selection: dict[str, Any],
    eval_config: dict[str, Any],
) -> None:
    _sealed(value, collection_campaign.PACKET_SCHEMA)
    expected = {
        "schema",
        "source",
        "task_selection_sha256",
        "runtime_bindings_sha256",
        "eval_config_sha256",
        "eval_plan_sha256",
        "generic_plan_source_job_id",
        "training_data_eligible",
        "corpus_scope",
        "admission_policy",
        "metrics_required",
        "sha256",
    }
    if set(value) != expected or value.get("training_data_eligible") is not True:
        raise ValueError("collection packet has an unexpected contract")
    source = _mapping(value.get("source"), "collection packet source")
    allowed_source = {
        "kind",
        "model_alias",
        "model",
        "template_sha256",
        "source_authorization_receipt_sha256",
    }
    if source.get("kind") == "teacher":
        allowed_source.add("teacher_strength_receipt_sha256")
    if set(source) != allowed_source:
        raise ValueError("collection packet source is malformed")
    if (
        source.get("kind") != selection["source_kind"]
        or source.get("model_alias") != selection["source_model_alias"]
        or source.get("model") != selection["source_model"]
        or source.get("template_sha256") != selection["template_sha256"]
    ):
        raise ValueError("collection packet source does not match admission selection")
    for field in ("source_authorization_receipt_sha256", "template_sha256"):
        _sha(source.get(field), f"collection packet {field}")
    if source.get("kind") == "teacher":
        _sha(source.get("teacher_strength_receipt_sha256"), "teacher strength receipt")
    if (
        value.get("task_selection_sha256") != task_selection.get("sha256")
        or value.get("eval_config_sha256") != collection_campaign.canonical_digest(eval_config)
        or value.get("eval_plan_sha256") != selection["campaign_plan_sha256"]
    ):
        raise ValueError("collection packet artifact binding differs from admission selection")
    # The packet binds a generic evaluator plan, not merely a hand-written
    # configuration.  Re-render it through the same offline compiler so a
    # changed task assignment, runtime contract, or evaluator identity cannot
    # be relabeled as the admitted campaign.
    compiled = collection_campaign._compile_local(task_selection, eval_config)
    if value["eval_plan_sha256"] != "sha256:" + compiled["sha256"]:
        raise ValueError("collection packet evaluation-plan digest is not reproducible")
    if value.get("generic_plan_source_job_id") is not None:
        raise ValueError("source-only collection packet fabricated a source job")
    scope = value.get("corpus_scope")
    if scope != {
        "current": "verified_success_visible_actions_only_v1",
        "visible_reasoning_included": False,
        "future_visible_reasoning_requires": [
            "separate_immutable_campaign_packet",
            "student_visible_source_evidence",
            "explicit_authorization_and_safety_evidence",
        ],
    }:
        raise ValueError("this materializer accepts only the visible-action packet scope")
    policy = _mapping(value.get("admission_policy"), "collection packet admission policy")
    expected_policy = {
        **_EXPECTED_PACKET_POLICY,
        "minimum_unique_visible_action_target_tokens": policy.get(
            "minimum_unique_visible_action_target_tokens"
        ),
    }
    if (
        policy != expected_policy
        or type(policy["minimum_unique_visible_action_target_tokens"]) is not int
        or policy["minimum_unique_visible_action_target_tokens"]
        < collection_campaign.MINIMUM_VISIBLE_TARGET_TOKENS
    ):
        raise ValueError("collection packet admission policy is insufficient")
    if value.get("metrics_required") != _EXPECTED_PACKET_METRICS:
        raise ValueError("collection packet lacks required aggregate safety metrics")
    if eval_config.get("training_data_eligible") is not True:
        raise ValueError("collection eval config is not training-data eligible")
    if eval_config.get("models") != {selection["source_model_alias"]: selection["source_model"]}:
        raise ValueError("collection eval config model differs from admission selection")
    harness = _mapping(eval_config.get("harness"), "collection eval harness")
    if (
        harness.get("tools") != ["bash", "submit_report"]
        or harness.get("tool_catalog_sha256") != selection["tool_catalog_sha256"]
        or "sha256:" + digest(harness) != selection["harness_treatment_sha256"]
    ):
        raise ValueError("collection eval harness differs from admission selection")


def _task_boundary(
    inventory: dict[str, Any],
    split: dict[str, Any],
    lock: dict[str, Any],
    runtime: dict[str, Any],
    task_selection: dict[str, Any],
    packet: dict[str, Any],
    selection: dict[str, Any],
) -> dict[tuple[str, str], dict[str, Any]]:
    rows = collection_campaign._inventory_rows(inventory)
    bindings = collection_campaign._runtime_bindings(runtime, inventory, rows)
    if not is_supported_schema(split) or split.get("inventory_sha256") != inventory.get("sha256"):
        raise ValueError("family split is not bound to the reviewed catalog")
    validate_split(split, rows)
    if lock.get("schema") != admission.PROTECTED_FAMILY_LOCK_SCHEMA:
        raise ValueError("invalid protected-family lock")
    _sealed(lock, admission.PROTECTED_FAMILY_LOCK_SCHEMA)
    if lock.get("source_split_sha256") != split.get("sha256"):
        raise ValueError("protected-family lock is not bound to the family split")
    protected = lock.get("heldout_group_ids")
    if not isinstance(protected, list) or protected != sorted(set(protected)) or not protected:
        raise ValueError("protected-family lock groups are malformed")
    roles = {(row["task_key"], row["task_version_id"]): row for row in split["tasks"]}
    protected_set = set(protected)
    nontrain = {row["group_id"] for row in roles.values() if row["split"] != "train"}
    if protected_set != nontrain:
        raise ValueError("protected-family lock must cover every nontraining family exactly")
    collection_campaign._sealed(task_selection, collection_campaign.SELECTION_SCHEMA)
    collection_campaign._no_private_content_keys(task_selection)
    if set(task_selection) != {
        "schema",
        "inventory_sha256",
        "family_split_sha256",
        "runtime_bindings_sha256",
        "task_validity_receipt_sha256",
        "tasks",
        "held_out_task_version_count",
        "family_leakage_check",
        "sha256",
    }:
        raise ValueError("collection task selection has an unexpected contract")
    if (
        task_selection.get("schema") != collection_campaign.SELECTION_SCHEMA
        or task_selection.get("sha256") != packet.get("task_selection_sha256")
        or task_selection.get("inventory_sha256") != inventory["sha256"]
        or task_selection.get("family_split_sha256") != split["sha256"]
        or task_selection.get("runtime_bindings_sha256") != runtime["sha256"]
    ):
        raise ValueError("collection task selection is not bound to the reviewed split")
    if task_selection.get("family_leakage_check") != {
        "exact_identity_overlap": 0,
        "reviewed_family_overlap": 0,
        "held_out_roles": ["dev", "final_test"],
    }:
        raise ValueError("collection task selection lacks family-leakage proof")
    task_rows = task_selection.get("tasks")
    if not isinstance(task_rows, list):
        raise ValueError("collection task selection is malformed")
    selected_tasks: dict[tuple[str, str], dict[str, Any]] = {}
    for row in task_rows:
        if not isinstance(row, dict) or set(row) != set(collection_campaign.EXACT_TASK_FIELDS):
            raise ValueError("collection task selection row is malformed")
        identity = (row["task_key"], row["task_version_id"])
        if identity in selected_tasks or identity not in bindings or row != bindings[identity]:
            raise ValueError("collection task selection changes a runtime binding")
        role = roles.get(identity)
        if role is None or role["split"] != "train" or role["group_id"] in protected_set:
            raise ValueError("held-out family reached the collection task selection")
        selected_tasks[identity] = row
    expected_train = {identity for identity, role in roles.items() if role["split"] == "train"}
    if set(selected_tasks) != expected_train:
        raise ValueError("collection task selection does not cover the exact train family roster")
    if (
        selection["catalog_inventory_sha256"] != inventory["sha256"]
        or selection["family_split_sha256"] != split["sha256"]
        or selection["protected_family_lock_sha256"] != lock["sha256"]
        or packet.get("runtime_bindings_sha256") != runtime["sha256"]
    ):
        raise ValueError("admission selection changes the reviewed task boundary")
    for row in selection["selected"]:
        identity = (row["task_key"], row["task_version_id"])
        role = roles.get(identity)
        if (
            identity not in selected_tasks
            or role is None
            or role["split"] != "train"
            or row["group_id"] != role["group_id"]
            or row["group_id"] in protected_set
        ):
            raise ValueError("admission selection contains a held-out family")
    return bindings


def _message(value: object) -> None:
    message = _mapping(value, "visible-action message")
    if set(message) - {"role", "content", "tool_calls", "tool_call_id"}:
        raise ValueError("private or unknown message fields are not allowed")
    if message.get("role") not in {"system", "user", "assistant", "tool"}:
        raise ValueError("visible-action message role is invalid")
    content = message.get("content")
    if content is not None and not isinstance(content, str):
        raise ValueError("visible-action message content is invalid")
    if isinstance(content, str) and _PRIVATE_REASONING_MARKER.search(content):
        raise ValueError("private reasoning marker in visible-action content")
    calls = message.get("tool_calls")
    if calls is not None:
        if not isinstance(calls, list):
            raise ValueError("visible-action tool calls are invalid")
        call_names: list[str] = []
        for call in calls:
            call = _mapping(call, "visible-action tool call")
            if set(call) - {"id", "type", "function"}:
                raise ValueError("private or unknown tool-call fields are not allowed")
            if (
                not isinstance(call.get("id"), str)
                or _OPAQUE_TOOL_ID.fullmatch(call["id"]) is None
                or call.get("type") != "function"
            ):
                raise ValueError("visible-action tool call identity/type is invalid")
            function = _mapping(call.get("function"), "visible-action tool function")
            if set(function) - {"name", "arguments"}:
                raise ValueError("private or unknown tool-function fields are not allowed")
            name = function.get("name")
            if name not in {"bash", "submit_report"}:
                raise ValueError("visible-action tool function is not approved")
            call_names.append(name)
            # ``submit_report.explanation`` is an assistant-generated string
            # and is therefore a supervised target in the dense template.  A
            # generic report explanation can contain untagged private CoT, so
            # this action-only contract admits only the empty form.  A future
            # student-visible report/reasoning corpus needs a separate packet
            # and exporter proof rather than weakening this check.
            if function.get("name") == "submit_report":
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        raise ValueError("visible-action report arguments are invalid") from None
                if not isinstance(arguments, dict) or arguments.get("explanation") != "":
                    raise ValueError("visible-action submit_report explanation must be empty")
        # A final report has a different admission cap from ordinary tool
        # work.  Letting it share a message with ``bash`` would let it be
        # supervised without being counted in either cap below.
        if "submit_report" in call_names and call_names != ["submit_report"]:
            raise ValueError("visible-action submit_report must not share an assistant turn")
    role = message["role"]
    if role == "assistant":
        # This corpus teaches only observable tool actions.  Untagged prose
        # could be hidden or unknown reasoning, so it is not a safe fallback
        # merely because it lacks a literal ``<think>`` marker.
        if content not in {None, ""}:
            raise ValueError("visible-action assistant must not contain prose or reasoning")
        if not isinstance(calls, list) or not calls:
            raise ValueError("visible-action assistant must contain a reviewed tool action")
    elif calls is not None:
        raise ValueError("only assistant messages may contain tool calls")
    if role == "tool" and (
        not isinstance(message.get("tool_call_id"), str)
        or _OPAQUE_TOOL_ID.fullmatch(message["tool_call_id"]) is None
    ):
        raise ValueError("tool result must bind a reviewed opaque tool action")


def _record(value: dict[str, Any]) -> dict[str, Any]:
    required = {
        "schema",
        "record_id",
        "campaign_plan_sha256",
        "cell_sha256",
        "source",
        "lineage",
        "content_policy",
        "ingestion",
        "messages",
        "content_digest",
    }
    if set(value) != required or value.get("schema") != RECORD_SCHEMA:
        raise ValueError("private normalized record has an unsupported schema")
    if value.get("content_digest") != digest_json(
        {key: item for key, item in value.items() if key != "content_digest"}
    ):
        raise ValueError("private normalized record digest mismatch")
    _string(value.get("record_id"), "private normalized record identity")
    _sha(value.get("campaign_plan_sha256"), "private normalized record campaign")
    _sha(value.get("cell_sha256"), "private normalized record cell")
    source = _mapping(value.get("source"), "private normalized record source")
    _known(source, {"model_alias", "model", "harness", "template_sha256"}, "record source")
    _mapping(source.get("model"), "record source model")
    harness = _mapping(source.get("harness"), "record source harness")
    _known(harness, {"treatment_sha256", "tool_catalog_sha256"}, "record source harness")
    _sha(source.get("template_sha256"), "record source template")
    lineage = _mapping(value.get("lineage"), "private normalized record lineage")
    _known(lineage, {"task_key", "task_version_id", "group_id"}, "record lineage")
    for field in lineage.values():
        _string(field, "record lineage field")
    if value.get("content_policy") != _POLICY:
        raise ValueError("private normalized record has private reasoning or unapproved compaction")
    ingestion = _mapping(value.get("ingestion"), "private normalized record ingestion")
    _known(
        ingestion,
        {"normalized_trajectory_sha256", "transcript_sha256"},
        "record ingestion",
    )
    for field in ingestion.values():
        _sha(field, "record ingestion")
    messages = value.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("private normalized record has no visible messages")
    for message in messages:
        _message(message)
    if digest_json(messages) != ingestion["normalized_trajectory_sha256"]:
        raise ValueError("private normalized record trajectory digest mismatch")
    return value


def _coverage(
    rows: list[dict[str, Any]], messages: list[dict[str, Any]], policy: dict[str, Any]
) -> dict[str, int]:
    if not rows:
        raise ValueError("selected record has no fitting visible-action windows")
    eligible = rows[0]["eligible_assistant_indices"]
    excluded = rows[0]["excluded_assistant_targets"]
    spans = [span for row in rows for span in row["target_spans"]]
    if len(spans) != len(eligible) or {span["assistant_index"] for span in spans} != set(eligible):
        raise ValueError("visible assistant target coverage is incomplete")
    assistants = [message for message in messages if message["role"] == "assistant"]
    non_submit = submit = submit_tokens = 0
    for span in spans:
        calls = assistants[span["assistant_index"]].get("tool_calls") or []
        names = [(call.get("function") or {}).get("name") for call in calls]
        if names and set(names) == {"bash"}:
            non_submit += 1
        if names == ["submit_report"]:
            submit += 1
            submit_tokens += span["token_end"] - span["token_start"]
    target_tokens = sum(span["token_end"] - span["token_start"] for span in spans)
    if not (
        non_submit >= policy["minimum_non_submit_tool_responses"]
        and non_submit >= policy["minimum_completed_non_submit_tool_rounds"]
        and submit / len(spans) <= policy["maximum_submit_report_response_fraction"]
        and target_tokens > 0
        and submit_tokens / target_tokens <= policy["maximum_submit_report_target_token_fraction"]
    ):
        raise ValueError("selected record lacks task-rich visible-action target coverage")
    return {
        "assistant_targets": len(spans),
        "excluded_assistant_targets": len(excluded),
        "non_submit_tool_responses": non_submit,
        "submit_report_responses": submit,
        "supervised_tokens": target_tokens,
        "submit_report_tokens": submit_tokens,
    }


def build(config: dict[str, Any], *, relative_to: Path) -> dict[str, Any]:
    """Materialize a private, evidence-bound dense corpus without external actions."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    _known(
        config,
        {
            "schema",
            "admission_selection",
            "admission_receipt",
            "collection_packet",
            "collection_task_selection",
            "collection_eval_config",
            "inventory",
            "family_split",
            "protected_family_lock",
            "runtime_bindings",
            "records",
            "model_lock",
            "tokenizer_root",
            "native_helper",
            "max_length",
            "context_tokens",
            "output",
        },
        "Fleet collection corpus materialization",
    )
    if config.get("schema") != SCHEMA:
        raise ValueError("unsupported Fleet collection corpus materialization request")
    output = _path(relative_to, config.get("output"), "output")
    if output.exists() or output.is_symlink():
        raise FileExistsError("create-once corpus destination exists")
    names = (
        "admission_selection",
        "admission_receipt",
        "collection_packet",
        "collection_task_selection",
        "collection_eval_config",
        "inventory",
        "family_split",
        "protected_family_lock",
        "runtime_bindings",
        "records",
        "model_lock",
        "native_helper",
    )
    sources = {name: _input(relative_to, config.get(name), name) for name in names}
    paths = {name: item[0] for name, item in sources.items()}
    expected_files = {name: item[1] for name, item in sources.items()}
    selection = _json(paths["admission_selection"], "admission selection")
    selected = _selection(selection)
    receipt = _json(paths["admission_receipt"], "admission receipt")
    _admission_receipt(receipt, selection)
    packet = _json(paths["collection_packet"], "collection packet")
    task_selection = _json(paths["collection_task_selection"], "collection task selection")
    eval_config = _json(paths["collection_eval_config"], "collection eval config")
    inventory = _json(paths["inventory"], "inventory")
    split = _json(paths["family_split"], "family split")
    lock = _json(paths["protected_family_lock"], "protected-family lock")
    runtime = _json(paths["runtime_bindings"], "runtime bindings")
    # Validate the catalog-derived selection before the offline evaluator
    # compiler serializes it into a temporary file.  This preserves the
    # metadata-only boundary even for a malformed input that carries a payload
    # under an unexpected field.
    _task_boundary(inventory, split, lock, runtime, task_selection, packet, selection)
    _packet(packet, selection, task_selection, eval_config)
    tokenizer_root = _path(relative_to, config.get("tokenizer_root"), "tokenizer root")
    tokenizer, tokenizer_identity = local_tokenizer(
        read_mapping(paths["model_lock"]), tokenizer_root
    )
    helper = native_helper(paths["native_helper"])
    maximum = config.get("max_length", 16384)
    context = config.get("context_tokens", 4096)
    if type(maximum) is not int or type(context) is not int or maximum < 2 or context < 0:
        raise ValueError("invalid dense window bounds")

    records: dict[str, dict[str, Any]] = {}
    for raw in iter_jsonl(paths["records"]):
        record = _record(raw)
        identity = record["record_id"]
        if identity in records:
            raise ValueError("private normalized records duplicate a session identity")
        records[identity] = record
    if set(records) != set(selected):
        raise ValueError("private normalized records do not cover the exact admission selection")

    packed_rows: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    window_payloads: set[str] = set()
    target_lengths: dict[str, int] = {}
    family_target_tokens: dict[str, int] = {}
    for session_id, admitted in sorted(selected.items()):
        record = records[session_id]
        if (
            record["campaign_plan_sha256"] != selection["campaign_plan_sha256"]
            or record["cell_sha256"] != admitted["cell_sha256"]
            or record["source"]
            != {
                "model_alias": selection["source_model_alias"],
                "model": selection["source_model"],
                "harness": admitted["harness"],
                "template_sha256": selection["template_sha256"],
            }
            or record["lineage"]
            != {
                "task_key": admitted["task_key"],
                "task_version_id": admitted["task_version_id"],
                "group_id": admitted["group_id"],
            }
            or digest_json(record) != admitted["normalized_record_sha256"]
            or record["ingestion"]["normalized_trajectory_sha256"]
            != admitted["normalized_trajectory_sha256"]
            or record["ingestion"]["transcript_sha256"] != admitted["transcript_sha256"]
        ):
            raise ValueError("private normalized record differs from admission selection")
        try:
            messages, _ = compatible_messages(record)
            anchor, chunks = encode_record(messages, tokenizer, helper)
            rows = segment_record(
                record, anchor, chunks, max_tokens=maximum, context_budget=context
            )
        except Excluded as error:
            raise ValueError(
                f"selected record failed visible-action materialization: {error.reason}"
            ) from None
        coverage = _coverage(rows, messages, packet["admission_policy"])
        fingerprints = {digest_json([row["input_ids"], row["loss_mask"]]) for row in rows}
        if len(fingerprints) != len(rows) or window_payloads & fingerprints:
            raise ValueError("duplicate packed window would break source target coverage")
        window_payloads.update(fingerprints)
        for row in rows:
            for span in row["target_spans"]:
                width = span["token_end"] - span["token_start"]
                prior = target_lengths.setdefault(span["source_target_sha256"], width)
                if prior != width:
                    raise ValueError("target digest has inconsistent token coverage")
                family_target_tokens[admitted["group_id"]] = (
                    family_target_tokens.get(admitted["group_id"], 0) + width
                )
            row.update(
                {
                    "source_campaign_plan_sha256": selection["campaign_plan_sha256"],
                    "source_collection_packet_sha256": packet["sha256"],
                    "source_admission_selection_sha256": selection["sha256"],
                    "source_normalized_record_sha256": admitted["normalized_record_sha256"],
                    "source_task_version_id": admitted["task_version_id"],
                }
            )
        packed_rows.extend(rows)
        selections.append(
            {
                "session_id": session_id,
                "task_key": admitted["task_key"],
                "task_version_id": admitted["task_version_id"],
                "group_id": admitted["group_id"],
                "normalized_record_sha256": admitted["normalized_record_sha256"],
                "normalized_trajectory_sha256": admitted["normalized_trajectory_sha256"],
                "coverage": coverage,
            }
        )
    packed_rows.sort(key=lambda row: (row["task_key"], row["source_session_id"], row["segment_id"]))
    if not packed_rows:
        raise ValueError("collection materialization produced no visible-action windows")
    counts = {
        "rows": len(packed_rows),
        "task_keys": sorted({row["task_key"] for row in packed_rows}),
        "format": DENSE_FORMAT,
        "source_sessions": len(selections),
        "supervised_tokens": sum(row["target_token_count"] for row in packed_rows),
        "assistant_responses": sum(len(row["target_spans"]) for row in packed_rows),
        # ``segment_record`` repeats these inventories on every packed window,
        # so count them once per source session from the materialization
        # evidence rather than summing the rows.
        "source_total_assistant_responses": sum(
            item["coverage"]["assistant_targets"] + item["coverage"]["excluded_assistant_targets"]
            for item in selections
        ),
        "excluded_assistant_responses": sum(
            item["coverage"]["excluded_assistant_targets"] for item in selections
        ),
    }
    dense_rows(packed_rows, counts, max_length=maximum, vocab_size=len(tokenizer))
    goal = packet["admission_policy"]["minimum_unique_visible_action_target_tokens"]
    unique_target_tokens = sum(target_lengths.values())
    target_ready = unique_target_tokens >= goal
    total_family_target_tokens = sum(family_target_tokens.values())
    if total_family_target_tokens <= 0:
        raise ValueError("collection materialization has no family target-token coverage")
    largest_family_tokens = max(family_target_tokens.values())
    largest_family_fraction = largest_family_tokens / total_family_target_tokens
    family_limit = packet["admission_policy"]["maximum_family_target_token_fraction"]
    family_balance_ok = largest_family_fraction <= family_limit
    if target_ready and not family_balance_ok:
        raise ValueError(
            "target-ready corpus exceeds its immutable family-token concentration limit"
        )
    coverage = {
        "schema": COVERAGE_SCHEMA,
        "admission_selection_sha256": selection["sha256"],
        "collection_packet_sha256": packet["sha256"],
        "selected_source_sessions": len(selected),
        "materialized_source_sessions": len(selections),
        "visible_action_windows": len(packed_rows),
        "visible_assistant_targets": counts["assistant_responses"],
        "unique_visible_action_target_tokens": unique_target_tokens,
        "minimum_unique_visible_action_target_tokens": goal,
        "target_goal_reached": target_ready,
        "family_token_concentration": {
            "families_with_visible_action_targets": len(family_target_tokens),
            "largest_family_target_token_fraction": largest_family_fraction,
            "maximum_allowed_fraction": family_limit,
            "within_limit": family_balance_ok,
        },
        "family_split_sha256": split["sha256"],
        "heldout_families_materialized": 0,
        "target_coverage": "every fitting visible assistant target appears exactly once",
    }
    coverage["sha256"] = "sha256:" + digest(coverage)
    manifest = {
        "schema": "cyber_dense_sft_corpus_v1",
        "source_sha256": expected_files["records"],
        "split_sha256": split["sha256"],
        "tokenizer": tokenizer_identity,
        "files": {"train": {"path": "train.parquet", "sha256": None, **counts}},
        "train_models": [selection["source_model"]["session_model"]],
        "max_length": maximum,
        "context_tokens": context,
        "dev_windows": 0,
        # A below-target collection is useful immutable evidence but deliberately
        # not a corpus SFT may compile.  ``sft.compile_sft`` rejects this mode.
        "validation_mode": "task_outcomes_only" if target_ready else "collection_pending_target",
        "whole_source_exclusions": {},
        "collection_materialization": {
            "admission_selection_sha256": selection["sha256"],
            "admission_receipt_sha256": receipt["sha256"],
            "collection_packet_sha256": packet["sha256"],
            "collection_task_selection_sha256": task_selection["sha256"],
            "collection_eval_config_sha256": packet["eval_config_sha256"],
            "catalog_inventory_sha256": inventory["sha256"],
            "family_split_sha256": split["sha256"],
            "protected_family_lock_sha256": lock["sha256"],
            "runtime_bindings_sha256": runtime["sha256"],
            "coverage_sha256": coverage["sha256"],
            "sft_ready": target_ready,
        },
        "catalog_provenance": {
            "selected_task_versions": len({row["task_version_id"] for row in selections}),
            "selected_source_sessions": len(selections),
            "heldout_task_families_excluded_across_all_versions": True,
            "source_kind": selection["source_kind"],
            "source_model_alias": selection["source_model_alias"],
        },
        "builder_sha256": {
            "fleet_collection_corpus.py": file_sha256(Path(__file__)),
            "dense.py": file_sha256(Path(__file__).with_name("dense.py")),
            "corpus.py": file_sha256(Path(__file__).with_name("corpus.py")),
        },
        "limitations": [
            "Visible assistant actions only; private and unknown reasoning are rejected.",
            "Opaque or unapproved compaction is rejected before windowing.",
            "A student-visible-reasoning corpus requires a separate immutable packet.",
        ],
    }
    for path, expected in expected_files.items():
        if file_sha256(paths[path]) != expected:
            raise ValueError(f"{path} changed during private corpus materialization")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.mkdir(output, 0o700)
    except FileExistsError as error:
        raise FileExistsError("create-once corpus destination exists") from error
    try:
        data_path = output / "train.parquet"
        pq.write_table(pa.Table.from_pylist(packed_rows), data_path, compression="zstd")
        os.chmod(data_path, 0o600)
        if pq.read_table(data_path).to_pylist() != packed_rows:
            raise ValueError("private corpus Parquet readback differs")
        manifest["files"]["train"]["sha256"] = file_sha256(data_path)
        manifest["sha256"] = "sha256:" + digest(manifest)
        atomic_write_jsonl(output / "source-selection.private.jsonl", selections, private=True)
        atomic_write_json(output / "coverage.private.json", coverage, private=True)
        atomic_write_json(output / "manifest.json", manifest, private=True)
        receipt_out = {
            "schema": RECEIPT_SCHEMA,
            "manifest_file_sha256": file_sha256(output / "manifest.json"),
            "manifest_sha256": manifest["sha256"],
            "coverage_file_sha256": file_sha256(output / "coverage.private.json"),
            "coverage_sha256": coverage["sha256"],
            "source_selection_file_sha256": file_sha256(output / "source-selection.private.jsonl"),
            "sft_ready": target_ready,
        }
        receipt_out["sha256"] = "sha256:" + digest(receipt_out)
        atomic_write_json(output / "MATERIALIZATION.json", receipt_out, private=True)
    except BaseException:
        # Leave the create-once directory in place rather than risk overwriting
        # partial private evidence on a retry.  Callers must investigate it.
        raise
    return {
        "submitted": False,
        "manifest_sha256": manifest["sha256"],
        "coverage_sha256": coverage["sha256"],
        "source_sessions": len(selections),
        "visible_action_windows": len(packed_rows),
        "unique_visible_action_target_tokens": unique_target_tokens,
        "sft_ready": target_ready,
    }
