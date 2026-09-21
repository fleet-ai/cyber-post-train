"""Offline plan for a broad, paired teacher-visible-rationale campaign.

The underlying source and admission contract lives in
``teacher_visible_rationale_campaign``.  This module adds the pieces that must
be reviewed before a scaled collection can exist: a conservative scale model,
an exact scientific-cell universe, create-once operation identities, explicit
benchmark/holdout exclusions, and a matched action-only materialization arm.

No function in this module performs network I/O, reads a transcript, calls a
model, or authorizes a submission.  ``review`` needs no teacher authorization.
``render`` additionally requires the immutable authorization defined by the
base contract, but still produces only local source artifacts.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import teacher_visible_rationale_campaign as base
from .io import digest_json, file_sha256

SOURCE_SPEC_SCHEMA = "cyber_teacher_visible_rationale_broad_source_spec_v1"
REVIEW_SCHEMA = "cyber_teacher_visible_rationale_broad_review_v1"
OPERATION_SCHEMA = "cyber_teacher_visible_rationale_operation_authorization_v1"
MATCHED_SCHEMA = "cyber_teacher_visible_rationale_matched_materialization_v1"
EXECUTION_MAP_SCHEMA = "cyber_teacher_visible_rationale_execution_map_v1"

SOURCE_ONLY_STATUS = "source_only_waiting_for_independent_review_and_teacher_authorization"
MINIMUM_ATTEMPTS_PER_TASK = 64
MAXIMUM_DAILY_FLEET_ROLLOUTS = 500
EXCLUDED_BENCHMARKS = ["WebExploitBench"]
EXCLUDED_SPLIT_ROLES = ["dev", "final_test"]
LOSS_ARMS = {
    "visible_rationale_plus_actions": {
        "ordinary_visible_rationale": 1,
        "visible_tool_actions": 1,
        "visible_compaction_summary": 0,
    },
    "matched_actions_only": {
        "ordinary_visible_rationale": 0,
        "visible_tool_actions": 1,
        "visible_compaction_summary": 0,
    },
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


def _reference(value: object, *, root: Path, label: str) -> tuple[Path, str]:
    ref = _exact(value, {"path", "file_sha256", "logical_sha256"}, label)
    path = Path(ref["path"])
    path = path if path.is_absolute() else root / path
    if path.is_symlink() or not path.is_file() or file_sha256(path) != ref["file_sha256"]:
        raise ValueError(f"{label} file digest mismatch")
    payload = _mapping(json.loads(path.read_text()), label)
    if payload.get("sha256") != ref["logical_sha256"]:
        raise ValueError(f"{label} logical digest mismatch")
    return path, ref["file_sha256"]


def _source_spec(value: Mapping[str, Any], *, root: Path) -> tuple[dict[str, Any], Path, Path]:
    spec = _sealed(value, SOURCE_SPEC_SCHEMA, "broad teacher-rationale source spec")
    _exact(
        spec,
        {
            "schema",
            "campaign_name",
            "status",
            "requirements",
            "historical_scale_reference",
            "scale",
            "operation_identity",
            "matched_comparison",
            "exclusions",
            "safety",
            "submitted",
            "sha256",
        },
        "broad teacher-rationale source spec",
    )
    if spec["status"] != SOURCE_ONLY_STATUS or spec["submitted"] is not False:
        raise ValueError("broad teacher-rationale campaign is not source-only")

    requirements_path, _ = _reference(
        spec["requirements"], root=root, label="teacher-rationale requirements"
    )
    history_path, _ = _reference(
        spec["historical_scale_reference"], root=root, label="historical scale reference"
    )

    scale = _exact(
        spec["scale"],
        {
            "minimum_unique_supervised_tokens_per_arm",
            "attempts_per_task",
            "maximum_sessions_per_task_version",
            "concurrency",
            "maximum_daily_fleet_rollouts",
            "historical_action_target_tokens",
            "historical_successful_sessions",
            "historical_reference_is_mixed_teacher_corpus",
            "historical_mean_is_planning_only",
            "final_private_materializer_must_measure_exact_unique_tokens",
        },
        "broad teacher-rationale scale",
    )
    if (
        scale["minimum_unique_supervised_tokens_per_arm"] != base.MINIMUM_UNIQUE_SUPERVISED_TOKENS
        or scale["attempts_per_task"] < MINIMUM_ATTEMPTS_PER_TASK
        or scale["maximum_sessions_per_task_version"] != scale["attempts_per_task"]
        or type(scale["concurrency"]) is not int
        or not 1 <= scale["concurrency"] <= 100
        or scale["maximum_daily_fleet_rollouts"] != MAXIMUM_DAILY_FLEET_ROLLOUTS
        or type(scale["historical_action_target_tokens"]) is not int
        or scale["historical_action_target_tokens"] <= 0
        or type(scale["historical_successful_sessions"]) is not int
        or scale["historical_successful_sessions"] <= 0
        or scale["historical_reference_is_mixed_teacher_corpus"] is not True
        or scale["historical_mean_is_planning_only"] is not True
        or scale["final_private_materializer_must_measure_exact_unique_tokens"] is not True
    ):
        raise ValueError("broad teacher-rationale scale policy drift")

    operation = _exact(
        spec["operation_identity"],
        {
            "scientific_cell_identity",
            "authorized_execution_identity",
            "canonical_operation_root_required",
            "dedicated_empty_ledger_required",
            "exclusive_pre_mutation_intent_required",
            "same_path_retry_allowed",
            "alternate_path_retry_allowed",
            "ambiguous_create_replay_allowed",
        },
        "broad teacher-rationale operation policy",
    )
    if operation != {
        "scientific_cell_identity": "sha256-requirements-task-runtime-attempt-v1",
        "authorized_execution_identity": "sha256-packet-scientific-cell-v1",
        "canonical_operation_root_required": True,
        "dedicated_empty_ledger_required": True,
        "exclusive_pre_mutation_intent_required": True,
        "same_path_retry_allowed": False,
        "alternate_path_retry_allowed": False,
        "ambiguous_create_replay_allowed": False,
    }:
        raise ValueError("broad teacher-rationale operation policy drift")

    matched = _exact(
        spec["matched_comparison"],
        {
            "single_collection",
            "same_selected_successes",
            "same_actions",
            "same_context_and_compaction",
            "same_packing_and_window_selection",
            "second_model_collection",
            "loss_arms",
            "minimum_unique_supervised_tokens_per_arm",
        },
        "broad teacher-rationale matched comparison",
    )
    if matched != {
        "single_collection": True,
        "same_selected_successes": True,
        "same_actions": True,
        "same_context_and_compaction": True,
        "same_packing_and_window_selection": True,
        "second_model_collection": False,
        "loss_arms": LOSS_ARMS,
        "minimum_unique_supervised_tokens_per_arm": base.MINIMUM_UNIQUE_SUPERVISED_TOKENS,
    }:
        raise ValueError("broad teacher-rationale matched comparison drift")

    exclusions = _exact(
        spec["exclusions"],
        {
            "split_roles",
            "external_benchmarks",
            "benchmark_prompts_traces_outputs_metadata_solutions_or_hints",
            "provider_private_or_hidden_reasoning",
            "inferred_or_reconstructed_rationale",
        },
        "broad teacher-rationale exclusions",
    )
    if exclusions != {
        "split_roles": EXCLUDED_SPLIT_ROLES,
        "external_benchmarks": EXCLUDED_BENCHMARKS,
        "benchmark_prompts_traces_outputs_metadata_solutions_or_hints": False,
        "provider_private_or_hidden_reasoning": False,
        "inferred_or_reconstructed_rationale": False,
    }:
        raise ValueError("broad teacher-rationale holdout or benchmark exclusion drift")

    safety = _exact(
        spec["safety"],
        {
            "source_only",
            "independent_review_required_before_external_calls",
            "fleet_api_calls",
            "model_calls",
            "raw_private_trace_reads",
            "external_submission_authorized",
            "training_authorized",
        },
        "broad teacher-rationale safety",
    )
    if safety != {
        "source_only": True,
        "independent_review_required_before_external_calls": True,
        "fleet_api_calls": 0,
        "model_calls": 0,
        "raw_private_trace_reads": 0,
        "external_submission_authorized": False,
        "training_authorized": False,
    }:
        raise ValueError("broad teacher-rationale source-only boundary drift")
    return spec, requirements_path, history_path


def _load_requirements(path: Path, *, root: Path) -> dict[str, Any]:
    value = _mapping(json.loads(path.read_text()), "teacher-rationale requirements")
    return base._requirements(value, root=root)


def _historical_scale(path: Path, spec: dict[str, Any]) -> tuple[int, int]:
    manifest = _sealed(
        _mapping(json.loads(path.read_text()), "historical action-only manifest"),
        "cyber_dense_sft_corpus_v1",
        "historical action-only manifest",
    )
    train = _mapping(_mapping(manifest["files"], "historical manifest files")["train"], "train")
    tokens = train.get("supervised_tokens")
    sessions = train.get("source_sessions")
    scale = spec["scale"]
    if (
        tokens != scale["historical_action_target_tokens"]
        or sessions != scale["historical_successful_sessions"]
    ):
        raise ValueError("historical action-only scale reference drift")
    return tokens, sessions


def _scientific_cells(
    *, requirements: dict[str, Any], roster: dict[str, Any]
) -> list[dict[str, Any]]:
    cells = []
    for identity, binding in sorted(roster["selected_runtime_bindings"].items()):
        for attempt in range(1, requirements["collection"]["attempts_per_task"] + 1):
            source = {
                "campaign_name": requirements["campaign_name"],
                "requirements_sha256": requirements["sha256"],
                "task_key": identity[0],
                "task_version_id": identity[1],
                "runtime_binding_sha256": digest_json(binding),
                "attempt": attempt,
            }
            cells.append({**source, "scientific_cell_id": digest_json(source)})
    ids = [cell["scientific_cell_id"] for cell in cells]
    if len(ids) != len(set(ids)):
        raise ValueError("broad teacher-rationale scientific cell identity collision")
    return cells


def review(source_spec: Mapping[str, Any], *, root: Path) -> dict[str, Any]:
    """Validate and summarize the scaled plan without a source authorization."""

    spec, requirements_path, history_path = _source_spec(source_spec, root=root)
    requirements = _load_requirements(requirements_path, root=root)
    if requirements["campaign_name"] != spec["campaign_name"]:
        raise ValueError("broad source spec and requirements campaign names differ")
    scale = spec["scale"]
    collection = requirements["collection"]
    if (
        collection["attempts_per_task"] != scale["attempts_per_task"]
        or collection["maximum_sessions_per_task_version"]
        != scale["maximum_sessions_per_task_version"]
        or collection["concurrency"] != scale["concurrency"]
        or collection["minimum_unique_supervised_tokens"]
        != scale["minimum_unique_supervised_tokens_per_arm"]
    ):
        raise ValueError("broad source spec and teacher-rationale requirements scale differ")

    roster = base._roster(requirements, root=root)
    cells = _scientific_cells(requirements=requirements, roster=roster)
    train_families = len(roster["selected"])
    planned_cells = len(cells)
    historical_tokens, historical_sessions = _historical_scale(history_path, spec)
    successes_for_action_floor = math.ceil(
        scale["minimum_unique_supervised_tokens_per_arm"] * historical_sessions / historical_tokens
    )
    if successes_for_action_floor > planned_cells:
        raise ValueError("planned cells cannot reach the 20M action-only comparison floor")
    aggregate_context_capacity = planned_cells * requirements["opencode"]["context_window_tokens"]
    result = {
        "schema": REVIEW_SCHEMA,
        "campaign_name": requirements["campaign_name"],
        "requirements_sha256": requirements["sha256"],
        "status": SOURCE_ONLY_STATUS,
        "train_task_versions": train_families,
        "train_families": train_families,
        "attempts_per_task": collection["attempts_per_task"],
        "planned_cells": planned_cells,
        "scientific_cell_universe_sha256": digest_json(
            [cell["scientific_cell_id"] for cell in cells]
        ),
        "minimum_unique_supervised_tokens_per_arm": scale[
            "minimum_unique_supervised_tokens_per_arm"
        ],
        "minimum_selected_families": collection["minimum_selected_families"],
        "minimum_selected_family_fraction": collection["minimum_selected_family_fraction"],
        "historical_action_target_tokens": historical_tokens,
        "historical_successful_sessions": historical_sessions,
        "historical_reference_is_mixed_teacher_corpus": True,
        "historical_mean_action_tokens_fraction": {
            "numerator": historical_tokens,
            "denominator": historical_sessions,
        },
        "estimated_successes_needed_for_action_only_20m": successes_for_action_floor,
        "estimated_required_success_fraction": {
            "numerator": successes_for_action_floor,
            "denominator": planned_cells,
        },
        "maximum_daily_fleet_rollouts": scale["maximum_daily_fleet_rollouts"],
        "minimum_calendar_days_at_daily_cap": math.ceil(
            planned_cells / scale["maximum_daily_fleet_rollouts"]
        ),
        "aggregate_context_window_capacity_tokens": aggregate_context_capacity,
        "provider_cost_estimate": {
            "currency": "USD",
            "planned_paid_teacher_rollouts": planned_cells,
            "aggregate_context_capacity_tokens": aggregate_context_capacity,
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
        },
        "cost_estimate_is_not_a_success_or_budget_guarantee": True,
        "matched_loss_arms": spec["matched_comparison"]["loss_arms"],
        "excluded_split_roles": spec["exclusions"]["split_roles"],
        "excluded_external_benchmarks": spec["exclusions"]["external_benchmarks"],
        "fleet_api_calls": 0,
        "model_calls": 0,
        "raw_private_trace_reads": 0,
        "external_submission_authorized": False,
        "training_authorized": False,
        "independent_review_required": True,
        "submitted": False,
    }
    result["sha256"] = digest_json(result)
    return result


def _operation_authorization(
    *,
    review_receipt: dict[str, Any],
    packet: dict[str, Any],
    requirements: dict[str, Any],
    roster: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    scientific = _scientific_cells(requirements=requirements, roster=roster)
    identities = []
    for cell in scientific:
        source = {
            "collection_packet_sha256": packet["sha256"],
            "scientific_cell_id": cell["scientific_cell_id"],
        }
        authorized_execution_id = digest_json(source)
        intent = {
            "collection_packet_sha256": packet["sha256"],
            "authorized_execution_id": authorized_execution_id,
            "operation": "create_teacher_visible_rationale_rollout_once",
        }
        identities.append(
            {
                **cell,
                "collection_packet_sha256": packet["sha256"],
                "authorized_execution_id": authorized_execution_id,
                "exclusive_intent_id": digest_json(intent),
                "operation_relative_path": (
                    f"cells/{authorized_execution_id.removeprefix('sha256:')}.json"
                ),
            }
        )
    universe = digest_json([item["authorized_execution_id"] for item in identities])
    root_payload = {
        "campaign_name": packet["campaign_name"],
        "collection_packet_sha256": packet["sha256"],
        "authorized_execution_universe_sha256": universe,
    }
    root_digest = digest_json(root_payload)
    execution_map = {
        "schema": EXECUTION_MAP_SCHEMA,
        "campaign_name": packet["campaign_name"],
        "collection_packet_sha256": packet["sha256"],
        "review_receipt_sha256": review_receipt["sha256"],
        "planned_cells": len(identities),
        "scientific_cell_universe_sha256": review_receipt["scientific_cell_universe_sha256"],
        "authorized_execution_universe_sha256": universe,
        "identity_map_sha256": digest_json(identities),
        "cells": identities,
        "external_submission_authorized": False,
        "submitted": False,
    }
    execution_map["sha256"] = digest_json(execution_map)
    value = {
        "schema": OPERATION_SCHEMA,
        **root_payload,
        "review_receipt_sha256": review_receipt["sha256"],
        "planned_cells": len(identities),
        "scientific_cell_universe_sha256": review_receipt["scientific_cell_universe_sha256"],
        "authorized_execution_universe_sha256": universe,
        "identity_map_sha256": digest_json(identities),
        "execution_map_sha256": execution_map["sha256"],
        "operation_root_name": (
            f"q38-teacher-visible-rationale-{root_digest.removeprefix('sha256:')[:12]}"
        ),
        "dedicated_ledger_id": digest_json(
            {"operation_root_sha256": root_digest, "kind": "dedicated-empty-ledger-v1"}
        ),
        "execution_contract": {
            "canonical_operation_root_required": True,
            "dedicated_empty_ledger_required": True,
            "exclusive_pre_mutation_intent_required": True,
            "same_path_retry_allowed": False,
            "alternate_path_retry_allowed": False,
            "ambiguous_create_replay_allowed": False,
        },
        "external_submission_authorized": False,
        "submitted": False,
    }
    value["sha256"] = digest_json(value)
    return value, execution_map


def _matched_plan(
    *, review_receipt: dict[str, Any], packet: dict[str, Any], operation: dict[str, Any]
) -> dict[str, Any]:
    value = {
        "schema": MATCHED_SCHEMA,
        "campaign_name": packet["campaign_name"],
        "collection_packet_sha256": packet["sha256"],
        "operation_authorization_sha256": operation["sha256"],
        "review_receipt_sha256": review_receipt["sha256"],
        "single_selected_success_set": True,
        "single_packed_window_selection": True,
        "same_serialized_messages_actions_and_compaction": True,
        "loss_arms": LOSS_ARMS,
        "action_only_arm_retains_visible_rationale_as_zero_loss_context": True,
        "minimum_unique_supervised_tokens_per_arm": base.MINIMUM_UNIQUE_SUPERVISED_TOKENS,
        "final_unique_token_gate_requires_private_materializer": True,
        "second_model_collection": False,
        "heldout_roles_excluded": EXCLUDED_SPLIT_ROLES,
        "external_benchmarks_excluded": EXCLUDED_BENCHMARKS,
        "provider_private_or_hidden_reasoning_allowed": False,
        "external_submission_authorized": False,
        "training_authorized": False,
        "submitted": False,
    }
    value["sha256"] = digest_json(value)
    return value


def render(
    source_spec: Mapping[str, Any], authorization: Mapping[str, Any], *, root: Path
) -> dict[str, dict[str, Any]]:
    """Render the authorized source packet and paired plan without external calls."""

    spec, requirements_path, _ = _source_spec(source_spec, root=root)
    requirements = _load_requirements(requirements_path, root=root)
    review_receipt = review(spec, root=root)
    rendered = base.render(requirements, authorization, root=root)
    roster = base._roster(requirements, root=root)
    packet = rendered["collection-packet.json"]
    operation, execution_map = _operation_authorization(
        review_receipt=review_receipt,
        packet=packet,
        requirements=requirements,
        roster=roster,
    )
    matched = _matched_plan(
        review_receipt=review_receipt,
        packet=packet,
        operation=operation,
    )
    return {
        **rendered,
        "broad-review.json": review_receipt,
        "operation-authorization.json": operation,
        "execution-map.private.json": execution_map,
        "matched-materialization-plan.json": matched,
    }


def write(output: Path, rendered: Mapping[str, Mapping[str, Any]]) -> None:
    """Create one immutable local source bundle."""

    for name in rendered:
        if not isinstance(name, str) or Path(name).name != name or name in {"", ".", ".."}:
            raise ValueError("broad teacher-rationale artifact name must be one local file")
    if output.exists() or output.is_symlink():
        raise FileExistsError("broad teacher-rationale destination already exists")
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
