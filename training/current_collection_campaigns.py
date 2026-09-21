"""Materialize reviewed inventory metadata into a frozen collection campaign.

This adapter is deliberately source-only.  It reads sealed task identities,
family roles, runtime bindings, and one approved serving profile.  It never
contacts Fleet, creates a workload, calls a model, or opens a trace or score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from training import collection_campaign as campaign
from training import fleet_collection_admission as admission
from training import task_family_split

SPEC_SCHEMA = "cyber_collection_campaign_source_spec_v1"
RECEIPT_SCHEMA = "cyber_collection_campaign_materialization_receipt_v1"
TEACHER_REQUIREMENTS_SCHEMA = "cyber_stronger_teacher_collection_requirements_v1"
VISIBLE_REASONING_REQUIREMENTS_SCHEMA = "cyber_student_visible_reasoning_collection_requirements_v1"
TASK_QUALITY_REQUIREMENTS_SCHEMA = "cyber_task_quality_qualification_wave_requirements_v1"
CURRENT_INVENTORY_SCHEMA = "fleet_current_blackbox_receipt_proven_filter_v1"
CURRENT_SPLIT_SCHEMA = "cyber_representative_study_split_v2"
RUNTIME_CATALOG_SCHEMA = "cyber_rl_task_set_v1"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _file_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _sealed(value: dict[str, Any], schema: str) -> None:
    if value.get("schema") != schema or value.get("sha256") != campaign.canonical_digest(
        {key: item for key, item in value.items() if key != "sha256"}
    ):
        raise ValueError(f"invalid sealed {schema}")


def _reference(root: Path, value: object, label: str) -> tuple[Path, dict[str, Any]]:
    if not isinstance(value, dict) or set(value) != {"path", "file_sha256", "logical_sha256"}:
        raise ValueError(f"{label} needs an exact file and logical reference")
    path_value = value["path"]
    if not isinstance(path_value, str) or not path_value:
        raise ValueError(f"{label} path is required")
    path = root / path_value
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular tracked file")
    if _file_sha256(path) != value["file_sha256"]:
        raise ValueError(f"{label} file digest mismatch")
    loaded = _load(path)
    logical_sha256 = loaded.get("sha256", campaign.canonical_digest(loaded))
    if logical_sha256 != value["logical_sha256"]:
        raise ValueError(f"{label} logical digest mismatch")
    return path, loaded


def _source_inventory(value: dict[str, Any], expected_count: int) -> dict[str, Any]:
    _sealed(value, CURRENT_INVENTORY_SCHEMA)
    if value.get("private_content_persisted") is not False:
        raise ValueError("current inventory must remain metadata-only")
    rows = value.get("task_versions")
    if (
        value.get("task_count") != expected_count
        or not isinstance(rows, list)
        or len(rows) != expected_count
    ):
        raise ValueError("current inventory count drift")
    adapted = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("current inventory row is not an object")
        lineage = row.get("lineage")
        if not isinstance(lineage, dict):
            raise ValueError("current inventory row lacks reviewed lineage")
        adapted.append(
            {
                "task_key": row.get("task_key"),
                "task_version_id": row.get("task_version_id"),
                "lineage": {field: lineage.get(field) for field in campaign.LINEAGE_FIELDS},
            }
        )
    adapted.sort(key=lambda row: (row["task_key"], row["task_version_id"]))
    return campaign.sealed(
        {
            "schema": campaign.INVENTORY_SCHEMA,
            "task_validity_receipt_sha256": value["sha256"],
            "task_versions": adapted,
        }
    )


def _source_split(
    value: dict[str, Any],
    source_inventory: dict[str, Any],
    inventory_reference: dict[str, Any],
    adapted_inventory: dict[str, Any],
    expected_counts: dict[str, int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    _sealed(value, CURRENT_SPLIT_SCHEMA)
    expected_inventory_binding = {
        "path": inventory_reference["path"],
        "file_sha256": inventory_reference["file_sha256"],
        "logical_sha256": inventory_reference["logical_sha256"],
        "task_versions": len(source_inventory["task_versions"]),
    }
    if value.get("inventory") != expected_inventory_binding:
        raise ValueError("representative split inventory binding drift")
    anchor = task_family_split.trusted_fleet_collection_root_anchor()
    if anchor.get("source") != {
        "split_schema": CURRENT_SPLIT_SCHEMA,
        "split_sha256": value["sha256"],
        "inventory_sha256": source_inventory["sha256"],
    }:
        raise ValueError("representative split differs from the reviewed collection root")
    total = sum(expected_counts.values())
    split = task_family_split.build_anchored(
        adapted_inventory["task_versions"],
        inventory_sha256=adapted_inventory["sha256"],
        role_anchor=anchor,
        seed="qwen38-current75-anchored-v1",
        ratios={role: count / total for role, count in expected_counts.items()},
        max_group_task_version_fraction=0.25,
    )
    task_family_split.validate(split, adapted_inventory["task_versions"], role_anchor=anchor)
    source_roles = {
        (row["task_key"], row["task_version_id"]): (row["group_id"], row["split"])
        for row in value.get("tasks", [])
    }
    anchored_roles = {
        (row["task_key"], row["task_version_id"]): (row["group_id"], row["split"])
        for row in split["tasks"]
    }
    if (
        source_roles != anchored_roles
        or {role: split["counts"][role]["task_versions"] for role in expected_counts}
        != expected_counts
    ):
        raise ValueError("anchored split changed an immutable study role or identity")
    return split, anchor


def _runtime_bindings(
    value: dict[str, Any],
    inventory: dict[str, Any],
    expected_catalog_count: int,
) -> tuple[dict[str, Any], int]:
    _sealed(value, RUNTIME_CATALOG_SCHEMA)
    rows = value.get("tasks")
    if not isinstance(rows, list) or len(rows) != expected_catalog_count:
        raise ValueError("runtime catalog count drift")
    wanted = {(row["task_key"], row["task_version_id"]) for row in inventory["task_versions"]}
    found: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("runtime catalog row is not an object")
        identity = (row.get("task_key"), row.get("task_version_id"))
        if identity in wanted:
            if identity in found:
                raise ValueError("runtime catalog duplicates an admitted task identity")
            found[identity] = {field: row.get(field) for field in campaign.EXACT_TASK_FIELDS}
    if set(found) != wanted:
        raise ValueError("runtime catalog does not cover the admitted inventory exactly")
    adapted = campaign.sealed(
        {
            "schema": campaign.RUNTIME_BINDINGS_SCHEMA,
            "metadata_inventory_sha256": inventory["sha256"],
            "task_validity_receipt_sha256": inventory["task_validity_receipt_sha256"],
            "task_versions": sorted(
                found.values(), key=lambda row: (row["task_key"], row["task_version_id"])
            ),
        }
    )
    return adapted, len(rows) - len(found)


def _request(
    spec: dict[str, Any],
    profile: dict[str, Any],
) -> dict[str, Any]:
    source = spec["source"]
    model_alias = source["model_alias"]
    route_name = source["route_name"]
    model = profile.get("models", {}).get(model_alias)
    route = profile.get("routes", {}).get(route_name)
    if not isinstance(model, dict) or not isinstance(route, dict):
        raise ValueError("source profile lacks the selected model or route")
    route = {key: item for key, item in route.items() if key not in {"model", "task_versions"}}
    route["name"] = route_name
    harness = dict(profile.get("harness", {}))
    harness["thinking_mode"] = campaign.THINKING_DISABLED
    request = {
        "schema": campaign.REQUEST_SCHEMA,
        "campaign_name": spec["campaign_name"],
        "source_kind": source["kind"],
        "source_model": model,
        "template_sha256": source["template_sha256"],
        "source_authorization_receipt_sha256": spec["sha256"],
        "route": route,
        "harness": harness,
        "images": profile.get("images"),
        "sampling": profile.get("sampling"),
        "concurrency": spec["collection"]["concurrency"],
        "attempts_per_task": spec["collection"]["attempts_per_task"],
        "target_unique_visible_action_tokens": spec["collection"][
            "target_unique_visible_action_tokens"
        ],
        "reasoning_policy": campaign.VISIBLE_ACTIONS_ONLY,
        "offline_compaction_policy": campaign.OPAQUE_COMPACTION_REJECT,
        "execution_mode": campaign.LOCAL_CPU_EXECUTION,
        "maximum_task_versions_per_family": spec["selection"]["maximum_task_versions_per_family"],
        "maximum_planned_cells": spec["collection"]["maximum_planned_cells"],
    }
    if source["kind"] == "teacher":
        request["teacher_strength_receipt_sha256"] = source["teacher_strength_receipt_sha256"]
    return request


def _validate_spec(spec: dict[str, Any]) -> None:
    _sealed(spec, SPEC_SCHEMA)
    if set(spec) != {
        "schema",
        "campaign_name",
        "purpose",
        "inputs",
        "source",
        "selection",
        "collection",
        "authorization",
        "safety",
        "sha256",
    }:
        raise ValueError("collection source spec has unknown or missing fields")
    if set(spec.get("inputs", {})) != {
        "inventory",
        "family_split",
        "runtime_catalog",
        "runtime_catalog_expected_task_versions",
        "source_profile",
    }:
        raise ValueError("collection source spec input set drift")
    source = spec.get("source", {})
    source_fields = {
        "status",
        "kind",
        "model_alias",
        "route_name",
        "template_sha256",
    }
    if source.get("kind") == "teacher":
        source_fields.add("teacher_strength_receipt_sha256")
    if set(source) != source_fields:
        raise ValueError("collection source identity has unknown or missing fields")
    if set(spec.get("selection", {})) != {
        "roles",
        "expected_counts",
        "maximum_task_versions_per_family",
        "root_role_anchor_id",
        "family_role_anchor_sha256",
    }:
        raise ValueError("collection selection policy drift")
    if (
        spec["selection"].get("root_role_anchor_id")
        != task_family_split.TRUSTED_FLEET_COLLECTION_ROOT_ID
        or spec["selection"].get("family_role_anchor_sha256")
        != task_family_split.trusted_fleet_collection_root_anchor()["sha256"]
    ):
        raise ValueError("collection selection does not bind the trusted role root")
    if spec.get("source", {}).get("status") != "authorized_action_source":
        raise ValueError("collection source is not authorized for visible-action collection")
    authorization = spec.get("authorization")
    if authorization != {
        "source_visible_action_collection": True,
        "student_visible_reasoning": False,
        "private_or_unknown_reasoning_as_target": False,
        "external_launch_by_this_artifact": False,
    }:
        raise ValueError("collection authorization scope drift")
    if spec.get("selection", {}).get("roles") != ["train"]:
        raise ValueError("collection may select only the immutable train role")
    if spec.get("collection") != {
        "attempts_per_task": 4,
        "concurrency": 8,
        "maximum_planned_cells": 200,
        "target_unique_visible_action_tokens": campaign.MINIMUM_VISIBLE_TARGET_TOKENS,
    }:
        raise ValueError("current collection budget drift")
    safety = spec.get("safety")
    if safety != {
        "source_only_materialization": True,
        "fleet_api_calls": 0,
        "model_calls": 0,
        "trace_or_score_reads": 0,
        "exact_prelaunch_duplicate_census_required": True,
        "ambiguous_cell_replay_allowed": False,
        "cluster_wrapper_supported": False,
        "cluster_wrapper_requires_two_stable_server_previews": True,
        "required_root_annotation": {campaign.FAILURE_ALERT_ANNOTATION: "off"},
    }:
        raise ValueError("collection safety policy drift")


def render(spec: dict[str, Any], *, root: Path) -> dict[str, dict[str, Any]]:
    """Render all local packet files without contacting any external service."""
    _validate_spec(spec)
    inventory_ref = spec["inputs"]["inventory"]
    _inventory_path, source_inventory = _reference(root, inventory_ref, "inventory")
    _split_path, source_split = _reference(root, spec["inputs"]["family_split"], "family split")
    _runtime_path, source_runtime = _reference(
        root, spec["inputs"]["runtime_catalog"], "runtime catalog"
    )
    _profile_path, source_profile = _reference(
        root, spec["inputs"]["source_profile"], "source profile"
    )
    expected = spec["selection"]["expected_counts"]
    inventory = _source_inventory(source_inventory, sum(expected.values()))
    split, role_anchor = _source_split(
        source_split,
        source_inventory,
        inventory_ref,
        inventory,
        expected,
    )
    protected_family_lock = campaign.sealed(
        {
            "schema": admission.PROTECTED_FAMILY_LOCK_SCHEMA,
            "source_split_sha256": split["sha256"],
            "heldout_group_ids": sorted(
                {row["group_id"] for row in split["tasks"] if row["split"] in {"dev", "final_test"}}
            ),
        }
    )
    runtime_bindings, excluded_runtime_rows = _runtime_bindings(
        source_runtime,
        inventory,
        spec["inputs"]["runtime_catalog_expected_task_versions"],
    )
    request = _request(spec, source_profile)
    rendered = campaign.render(
        request,
        inventory,
        split,
        runtime_bindings,
        role_anchor=role_anchor,
    )
    train_tasks = len(rendered["task-selection.json"]["tasks"])
    planned_cells = train_tasks * request["attempts_per_task"]
    if (
        train_tasks != expected["train"]
        or planned_cells != spec["collection"]["maximum_planned_cells"]
    ):
        raise ValueError("materialized campaign count or cell budget drift")
    outputs = {
        "metadata-inventory.json": inventory,
        "family-split.json": split,
        "role-anchor.json": role_anchor,
        "protected-family-lock.json": protected_family_lock,
        "runtime-bindings.json": runtime_bindings,
        "collection-request.json": request,
        **rendered,
    }
    receipt = campaign.sealed(
        {
            "schema": RECEIPT_SCHEMA,
            "source_spec_sha256": spec["sha256"],
            "source_input_logical_sha256": {
                name: reference["logical_sha256"]
                for name, reference in spec["inputs"].items()
                if isinstance(reference, dict) and "logical_sha256" in reference
            },
            "output_logical_sha256": {
                name: value.get("sha256", campaign.canonical_digest(value))
                for name, value in outputs.items()
            },
            "inventory_task_versions": sum(expected.values()),
            "train_task_versions": train_tasks,
            "held_out_task_versions": expected["dev"] + expected["final_test"],
            "root_role_anchor_id": task_family_split.TRUSTED_FLEET_COLLECTION_ROOT_ID,
            "family_role_anchor_sha256": role_anchor["sha256"],
            "protected_family_lock_sha256": protected_family_lock["sha256"],
            "excluded_unqualified_runtime_catalog_rows": excluded_runtime_rows,
            "attempts_per_task": request["attempts_per_task"],
            "planned_cells": planned_cells,
            "training_data_eligible": True,
            "reasoning_generation": campaign.THINKING_DISABLED,
            "reasoning_targets_included": False,
            "context_window_size": request["harness"]["context_window_size"],
            "context_management": request["harness"]["context_management"],
            "submitted": False,
            "fleet_api_calls": 0,
            "model_calls": 0,
            "trace_or_score_reads": 0,
        }
    )
    outputs["materialization-receipt.json"] = receipt
    return outputs


def write_once(output: Path, rendered: dict[str, dict[str, Any]]) -> None:
    try:
        output.mkdir(mode=0o700)
    except FileExistsError as error:
        raise FileExistsError("collection campaign destination already exists") from error
    for name, value in rendered.items():
        path = output / name
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(campaign.raw(value))
            stream.flush()
            os.fsync(stream.fileno())


def check(output: Path, rendered: dict[str, dict[str, Any]]) -> None:
    expected_names = set(rendered)
    actual_names = {path.name for path in output.iterdir()} if output.is_dir() else set()
    if actual_names != expected_names:
        raise ValueError("collection campaign output file set drift")
    for name, value in rendered.items():
        if (output / name).read_bytes() != campaign.raw(value):
            raise ValueError(f"collection campaign output drift: {name}")


def validate_teacher_requirements(value: dict[str, Any]) -> None:
    _sealed(value, TEACHER_REQUIREMENTS_SCHEMA)
    _validate_admitted_roster(value.get("admitted_roster"), budget_field="planned_cells")
    if value.get("status") != "blocked_missing_approved_immutable_source_profile":
        raise ValueError("stronger-teacher requirements must fail closed until approved")
    missing = value.get("missing_required_issuer_fields")
    if not isinstance(missing, list) or not missing or len(missing) != len(set(missing)):
        raise ValueError("stronger-teacher requirements need exact missing issuer fields")
    historical = value.get("historical_aggregate_visible_action_supply")
    if not isinstance(historical, dict) or set(historical) != {
        "manifest_path",
        "manifest_file_sha256",
        "manifest_logical_sha256",
        "manifest_target_mode",
        "verified_success_source_sessions",
        "selected_task_keys",
        "exact_task_versions",
        "rechunked_windows",
        "assistant_responses",
        "supervised_target_tokens",
        "heldout_task_families_excluded",
        "source_max_length",
        "materialized_max_length",
        "all_teacher_model_session_counts",
        "allowed_source_session_counts_observed",
        "allowed_source_sessions_observed",
        "mixed_teacher_sources",
        "eligible_for_new_materializer_without_per_record_evidence",
        "raw_trace_read_for_this_requirements_packet",
    }:
        raise ValueError("stronger-teacher historical aggregate supply shape drift")
    if (
        historical["manifest_path"] != "configs/data/qwen38-teacher3k-32k-v1.manifest.json"
        or historical["manifest_file_sha256"]
        != "sha256:ebcf6677e51adb22f73602c210281a3f4a847c257155da6ed48cd1c6cba9f3c8"
        or historical["manifest_logical_sha256"]
        != "sha256:a8d08609991d7960cdcdab826bd07c3216bfcd3aa8f6fb33ab7a5fd29648d6f5"
        or historical["manifest_target_mode"]
        != "visible_assistant_actions_only_private_reasoning_omitted"
        or historical["verified_success_source_sessions"] != 2886
        or historical["selected_task_keys"] != 496
        or historical["exact_task_versions"] != 1176
        or historical["rechunked_windows"] != 14693
        or historical["assistant_responses"] != 176654
        or historical["supervised_target_tokens"] != 57384881
        or historical["heldout_task_families_excluded"] != 25
        or historical["source_max_length"] != 262144
        or historical["materialized_max_length"] != 32768
        or historical["mixed_teacher_sources"] is not True
        or historical["eligible_for_new_materializer_without_per_record_evidence"] is not False
        or historical["raw_trace_read_for_this_requirements_packet"] is not False
    ):
        raise ValueError("stronger-teacher historical aggregate supply evidence drift")
    all_sources = historical["all_teacher_model_session_counts"]
    allowed_sources = historical["allowed_source_session_counts_observed"]
    if (
        all_sources
        != {
            "claude-opus-4-8": 45,
            "claude-opus-4.8": 142,
            "claude-opus-5": 216,
            "claude-sonnet-5": 30,
            "gemini-3.1-pro-preview": 172,
            "gpt-5.6-sol": 645,
            "gpt-5.6-terra": 45,
            "grok-4.5": 273,
            "grok-4.6": 1283,
            "kimi-k3": 35,
        }
        or allowed_sources != {"gpt-5.6-sol": 645, "grok-4.5": 273}
        or set(allowed_sources) != set(value.get("allowed_source_names", []))
        or sum(all_sources.values()) != historical["verified_success_source_sessions"]
        or sum(allowed_sources.values()) != historical["allowed_source_sessions_observed"]
        or historical["supervised_target_tokens"] < campaign.MINIMUM_VISIBLE_TARGET_TOKENS
    ):
        raise ValueError("stronger-teacher historical source census drift")


def validate_visible_reasoning_requirements(value: dict[str, Any]) -> None:
    _sealed(value, VISIBLE_REASONING_REQUIREMENTS_SCHEMA)
    _validate_admitted_roster(value.get("admitted_roster"), budget_field="maximum_planned_cells")
    if value.get("status") != "blocked_pending_visible_reasoning_qualification":
        raise ValueError("student-visible reasoning requirements must fail closed")
    gates = value.get("missing_launch_gates")
    if not isinstance(gates, list) or not gates or len(gates) != len(set(gates)):
        raise ValueError("student-visible reasoning requirements need exact launch gates")
    if not {
        "pinned_immutable_reasoning_success_evidence_mapping",
        "aggregate_reasoning_census_and_selection_receipt",
    } <= set(gates):
        raise ValueError("student-visible reasoning lacks success-evidence selection gates")
    if value.get("separation_contract") != {
        "new_collection_arm": True,
        "new_corpus_schema_required": True,
        "mix_with_visible_action_corpus": False,
        "teacher_private_or_unknown_reasoning_allowed": False,
        "inferred_or_reconstructed_reasoning_allowed": False,
    }:
        raise ValueError("student-visible reasoning arm separation drift")
    if value.get("distinct_contract_schemas") != {
        "source_profile": "cyber_qwen_opencode_student_visible_reasoning_source_profile_v1",
        "packet": "cyber_qwen_opencode_student_visible_reasoning_packet_v1",
        "selection": "cyber_qwen_opencode_student_visible_reasoning_selection_v1",
        "record": "cyber_qwen_opencode_student_visible_reasoning_record_v1",
        "corpus": "cyber_qwen_opencode_visible_reasoning_sft_corpus_v1",
    }:
        raise ValueError("student-visible reasoning contract-schema identity drift")
    if value.get("required_reasoning_admission_policy") != {
        "action_packet_reuse_allowed": False,
        "training_data_eligible": True,
        "minimum_unique_target_tokens": campaign.MINIMUM_VISIBLE_TARGET_TOKENS,
        "maximum_family_target_token_fraction": campaign.MAXIMUM_FAMILY_TARGET_TOKEN_FRACTION,
        "deduplication_order": [
            "source_session_identity",
            "normalized_trajectory_digest",
            "packed_window_payload_digest",
        ],
        "reject": [
            "private_or_unknown_reasoning",
            "opaque_compaction",
            "unknown_serialization",
            "heldout_family",
        ],
    }:
        raise ValueError("student-visible reasoning admission policy drift")
    if value.get("required_serialization_round_trip") != {
        "prompt_token_ids_per_case": True,
        "collection_training_and_serving_full_token_ids_per_case": True,
        "prompt_messages": "messages_without_target",
        "prompt_add_generation_prompt": True,
        "prompt_token_ids_must_equal_local_apply_chat_template": True,
        "prompt_token_ids_exact_prefix_of_every_full_serialization": True,
    }:
        raise ValueError("student-visible reasoning serialization boundary drift")


def validate_task_quality_requirements(value: dict[str, Any]) -> None:
    _sealed(value, TASK_QUALITY_REQUIREMENTS_SCHEMA)
    if value.get("status") != "blocked_pending_reviewed_lineage_and_runtime_roster":
        raise ValueError("task-quality qualification requirements must fail closed")
    if value.get("public_evidence_policy") != {
        "aggregate_only": True,
        "task_or_session_identifiers": False,
        "prompts_traces_answers_flags_credentials": False,
        "numeric_scores": False,
        "training_data_eligible": False,
    }:
        raise ValueError("task-quality public evidence policy drift")
    if value.get("final_family_policy", {}).get("qualification_allowed") is not False:
        raise ValueError("immutable final families must remain outside qualification")
    final_policy = value.get("final_family_policy", {})
    trusted_anchor = task_family_split.trusted_fleet_collection_root_anchor()
    if (
        final_policy.get("root_role_anchor_id")
        != task_family_split.TRUSTED_FLEET_COLLECTION_ROOT_ID
        or final_policy.get("family_role_anchor_sha256") != trusted_anchor["sha256"]
    ):
        raise ValueError("task-quality wave does not bind the trusted family-role root")
    proven = value.get("receipt_proven_supply")
    if proven != {
        "total_with_exact_execution_receipt": 80,
        "admitted_not_known_broken": 75,
        "admitted_exact_success": 42,
        "admitted_exact_canonical_failure": 33,
        "excluded_known_broken_exact_success": 2,
        "excluded_known_broken_exact_canonical_failure": 3,
        "protected_train": 50,
        "protected_dev": 17,
        "protected_final_test": 8,
    }:
        raise ValueError("task-quality receipt-proven supply drift")
    waves = value.get("expansion_wave_plan")
    if not isinstance(waves, dict):
        raise ValueError("task-quality expansion wave plan is required")
    priority = waves.get("priority_wave")
    backlog = waves.get("not_analyzed_waves")
    if priority != {
        "candidate_versions": 17,
        "qa_clean_versions": 7,
        "qa_agent_failure_versions": 10,
        "maximum_versions_per_batch": 64,
        "maximum_batch_count": 1,
    } or backlog != {
        "candidate_versions": 949,
        "maximum_versions_per_batch": 64,
        "full_batch_size": 64,
        "full_batch_count": 14,
        "final_batch_size": 53,
        "maximum_batch_count": 15,
    }:
        raise ValueError("task-quality bounded wave schedule drift")
    if (
        backlog["full_batch_size"] * backlog["full_batch_count"] + backlog["final_batch_size"]
        != backlog["candidate_versions"]
        or waves.get("known_broken_without_exact_receipt_excluded") != 47
        or waves.get("nonbroken_unproven_candidate_upper_bound") != 966
        or waves.get("receipt_proven_plus_candidate_upper_bound") != 1041
        or waves.get("upper_bounds_are_not_admission_or_training_authority") is not True
    ):
        raise ValueError("task-quality candidate upper-bound arithmetic drift")
    if value.get("collection_growth_policy") != {
        "attempts_per_admitted_train_task_version": 4,
        "base_qwen_visible_action_packet_after_each_sealed_roster": True,
        "separate_approved_stronger_teacher_visible_action_packet_after_each_sealed_roster": True,
        "student_visible_reasoning_requires_distinct_contract_and_gates": True,
        "use_every_admitted_train_family": True,
        "maximum_task_versions_per_family": 1,
        "campaign_cell_formula": (
            "admitted_train_task_versions * attempts_per_admitted_train_task_version"
        ),
        "exact_cell_count_declared_only_after_anchored_split": True,
        "outcome_blind_before_collection": True,
    }:
        raise ValueError("task-quality collection growth policy drift")


def _validate_admitted_roster(value: object, *, budget_field: str) -> None:
    if not isinstance(value, dict) or set(value) != {
        "inventory_sha256",
        "family_split_sha256",
        "root_role_anchor_id",
        "family_role_anchor_sha256",
        "protected_family_lock_sha256",
        "runtime_bindings_sha256",
        "train_task_versions",
        "held_out_task_versions",
        "maximum_task_versions_per_family",
        budget_field,
    }:
        raise ValueError("campaign requirements do not bind the exact admitted roster")
    for field in (
        "inventory_sha256",
        "family_split_sha256",
        "family_role_anchor_sha256",
        "protected_family_lock_sha256",
        "runtime_bindings_sha256",
    ):
        campaign._sha256(value.get(field), f"admitted roster {field}")
    trusted_anchor = task_family_split.trusted_fleet_collection_root_anchor()
    if (
        value["root_role_anchor_id"] != task_family_split.TRUSTED_FLEET_COLLECTION_ROOT_ID
        or value["family_role_anchor_sha256"] != trusted_anchor["sha256"]
        or value["train_task_versions"] != 50
        or value["held_out_task_versions"] != 25
        or value["maximum_task_versions_per_family"] != 1
        or value[budget_field] != 200
    ):
        raise ValueError("campaign requirements drift from the frozen admitted roster")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rendered = render(_load(args.spec), root=args.root)
    if args.write:
        write_once(args.output, rendered)
    else:
        check(args.output, rendered)
    print(
        json.dumps(
            {
                "files": sorted(rendered),
                "planned_cells": rendered["materialization-receipt.json"]["planned_cells"],
                "submitted": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
