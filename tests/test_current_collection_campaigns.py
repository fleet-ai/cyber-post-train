"""Production source-only collection campaign contracts; no Fleet or trace access."""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from training import collection_campaign as campaign
from training import current_collection_campaigns as current
from training import task_family_split

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "configs/collection/qwen38-base-current75-actions-pass4-v1.source.json"
OUTPUT = ROOT / "configs/collection/qwen38-base-current75-actions-pass4-v1"
TEACHER = ROOT / "configs/collection/stronger-teacher-current75-actions-pass4-v1.requirements.json"
VISIBLE_REASONING = (
    ROOT / "configs/collection/qwen38-self-visible-reasoning-current75-pass4-v1.requirements.json"
)
TASK_QUALITY = (
    ROOT / "configs/qualification/fleet-blackbox-unproven-task-quality-wave-v1.requirements.json"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def test_committed_base_campaign_is_exactly_reproducible_and_source_only() -> None:
    rendered = current.render(_load(SPEC), root=ROOT)
    current.check(OUTPUT, rendered)
    receipt = rendered["materialization-receipt.json"]
    assert receipt["inventory_task_versions"] == 75
    assert receipt["train_task_versions"] == 50
    assert receipt["held_out_task_versions"] == 25
    assert receipt["attempts_per_task"] == 4
    assert receipt["planned_cells"] == 200
    assert receipt["excluded_unqualified_runtime_catalog_rows"] == 14
    assert receipt["training_data_eligible"] is True
    assert receipt["reasoning_generation"] == "disabled"
    assert receipt["reasoning_targets_included"] is False
    assert receipt["context_window_size"] == 262144
    assert receipt["submitted"] is False
    assert receipt["fleet_api_calls"] == receipt["model_calls"] == 0
    assert receipt["trace_or_score_reads"] == 0


def test_base_campaign_preserves_exact_roles_families_and_runtime_identities() -> None:
    rendered = current.render(_load(SPEC), root=ROOT)
    inventory = rendered["metadata-inventory.json"]
    split = rendered["family-split.json"]
    anchor = rendered["role-anchor.json"]
    lock = rendered["protected-family-lock.json"]
    selection = rendered["task-selection.json"]
    assert split["schema"] == task_family_split.ANCHORED_SCHEMA
    assert split["parent_role_anchor_sha256"] == anchor["sha256"]
    assert split["policy"]["require_representable_labels"] is False
    assert split["policy"]["representative_coverage_exception"] == ("all_current_groups_inherited")
    assert split["anchor_audit"]["all_current_groups_immutable"] is True
    assert split["new_group_ids"] == []
    assert lock["source_split_sha256"] == split["sha256"]
    assert lock["heldout_group_ids"] == sorted(
        {row["group_id"] for row in split["tasks"] if row["split"] != "train"}
    )
    assert len(lock["heldout_group_ids"]) == 25
    roles = Counter(row["split"] for row in split["tasks"])
    assert roles == Counter(train=50, dev=17, final_test=8)
    selected = {(row["task_key"], row["task_version_id"]) for row in selection["tasks"]}
    role_by_identity = {(row["task_key"], row["task_version_id"]): row for row in split["tasks"]}
    assert len(selected) == 50
    assert all(role_by_identity[identity]["split"] == "train" for identity in selected)
    assert (
        not {
            identity
            for identity, row in role_by_identity.items()
            if row["split"] in {"dev", "final_test"}
        }
        & selected
    )

    family_roles = defaultdict(set)
    for row in split["tasks"]:
        family_roles[row["group_id"]].add(row["split"])
    assert len(family_roles) == len(inventory["task_versions"]) == 75
    assert all(len(values) == 1 for values in family_roles.values())

    runtime = rendered["runtime-bindings.json"]
    assert {(row["task_key"], row["task_version_id"]) for row in runtime["task_versions"]} == {
        (row["task_key"], row["task_version_id"]) for row in inventory["task_versions"]
    }


def test_base_campaign_freezes_action_only_budget_dedupe_and_cluster_fail_closed_policy() -> None:
    rendered = current.render(_load(SPEC), root=ROOT)
    config = rendered["eval-config.json"]
    packet = rendered["collection-packet.json"]
    assert config["training_data_eligible"] is True
    assert config["pass_k"] == 4
    assert config["concurrency"] == 8
    assert config["harness"]["thinking_mode"] == campaign.THINKING_DISABLED
    assert config["harness"]["context_window_size"] == 262144
    assert config["harness"]["context_management"] == campaign.ONLINE_COMPACTION
    assert config["collection_runtime"]["reasoning_request_override"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }
    assert config["collection_runtime"]["opencode_model_reasoning"] is False
    assert config["collection_runtime"]["opencode_cli_thinking_flag"] is False
    assert packet["source"]["model_alias"] == "source"
    assert packet["source"]["template_sha256"].startswith("sha256:")
    assert packet["root_role_anchor_id"] == "fleet-blackbox-current-study-20260914-v2"
    assert packet["family_role_anchor_sha256"] == (
        "sha256:48350b8fc23143abe297db3b2364590c72d564553ebb4e9a349fa06ec246a26b"
    )
    assert packet["admission_policy"]["deduplication_order"] == [
        "source_session_identity",
        "normalized_trajectory_digest",
        "packed_window_payload_digest",
    ]
    assert packet["admission_policy"]["maximum_admitted_sessions_per_family"] == 4
    assert packet["admission_policy"]["maximum_family_target_token_fraction"] == 0.25
    safety = packet["execution_safety"]
    assert safety["planned_cells"] == safety["maximum_planned_cells"] == 200
    assert safety["prelaunch_exact_cell_duplicate_census_required"] is True
    assert safety["duplicate_census_max_age_seconds"] == 600
    assert safety["duplicate_census_required_coverage"] == (
        "all_authoritative_collection_ledgers_and_fleet_sessions_v1"
    )
    assert safety["automatic_replay_of_ambiguous_cells"] is False
    assert safety["external_submission"] is False
    assert safety["cluster_wrapper_supported"] is False
    assert safety["cluster_wrapper_enablement_requires"]["required_top_level_annotation"] == {
        "fleet.ai/failure-alerts": "off"
    }
    assert (
        safety["cluster_wrapper_enablement_requires"][
            "request_or_pod_template_annotation_is_insufficient"
        ]
        is True
    )


def test_cell_budget_and_source_spec_tampering_fail_closed() -> None:
    spec = _load(SPEC)
    spec["collection"]["maximum_planned_cells"] = 201
    spec["sha256"] = campaign.canonical_digest(
        {key: value for key, value in spec.items() if key != "sha256"}
    )
    with pytest.raises(ValueError, match="budget drift"):
        current.render(spec, root=ROOT)

    spec = _load(SPEC)
    spec["source"]["status"] = "pending"
    spec["sha256"] = campaign.canonical_digest(
        {key: value for key, value in spec.items() if key != "sha256"}
    )
    with pytest.raises(ValueError, match="not authorized"):
        current.render(spec, root=ROOT)


def test_stronger_teacher_spec_names_exact_missing_issuer_fields_and_cannot_render() -> None:
    requirements = _load(TEACHER)
    current.validate_teacher_requirements(requirements)
    assert requirements["status"] == "blocked_missing_approved_immutable_source_profile"
    assert requirements["allowed_source_names"] == ["gpt-5.6-sol", "grok-4.5"]
    missing = set(requirements["missing_required_issuer_fields"])
    assert {
        "source_model.immutable_revision",
        "source_authorization_receipt.issuer",
        "teacher_strength_receipt.issuer",
        "teacher_strength_receipt.capability_basis",
        "template.enable_thinking_false",
        "route.server_info",
    } <= missing
    assert requirements["prohibitions"]["relabel_qwen_teacher_sft_checkpoint_as_teacher"] is False
    historical = requirements["historical_aggregate_visible_action_supply"]
    historical_manifest_path = ROOT / historical["manifest_path"]
    assert (
        "sha256:" + hashlib.sha256(historical_manifest_path.read_bytes()).hexdigest()
        == (historical["manifest_file_sha256"])
    )
    historical_manifest = _load(historical_manifest_path)
    assert historical_manifest["sha256"] == historical["manifest_logical_sha256"]
    assert historical_manifest["files"]["train"]["source_sessions"] == 2886
    assert historical_manifest["files"]["train"]["rows"] == 14693
    assert historical_manifest["files"]["train"]["supervised_tokens"] == 57384881
    assert historical["allowed_source_session_counts_observed"] == {
        "gpt-5.6-sol": 645,
        "grok-4.5": 273,
    }
    assert historical["allowed_source_sessions_observed"] == 918
    assert historical["eligible_for_new_materializer_without_per_record_evidence"] is False
    assert historical["raw_trace_read_for_this_requirements_packet"] is False
    assert requirements["admitted_roster"]["root_role_anchor_id"] == (
        task_family_split.TRUSTED_FLEET_COLLECTION_ROOT_ID
    )
    rendered_anchor = _load(OUTPUT / "role-anchor.json")["sha256"]
    assert requirements["admitted_roster"]["family_role_anchor_sha256"] == rendered_anchor
    assert rendered_anchor == task_family_split.trusted_fleet_collection_root_anchor()["sha256"]
    assert (
        requirements["admitted_roster"]["inventory_sha256"]
        == _load(OUTPUT / "metadata-inventory.json")["sha256"]
    )
    assert (
        requirements["admitted_roster"]["family_split_sha256"]
        == _load(OUTPUT / "family-split.json")["sha256"]
    )
    assert (
        requirements["admitted_roster"]["protected_family_lock_sha256"]
        == _load(OUTPUT / "protected-family-lock.json")["sha256"]
    )
    assert (
        requirements["admitted_roster"]["runtime_bindings_sha256"]
        == _load(OUTPUT / "runtime-bindings.json")["sha256"]
    )

    forged = copy.deepcopy(requirements)
    forged["status"] = "ready"
    forged["sha256"] = campaign.canonical_digest(
        {key: value for key, value in forged.items() if key != "sha256"}
    )
    with pytest.raises(ValueError, match="fail closed"):
        current.validate_teacher_requirements(forged)


def test_student_visible_reasoning_is_a_separate_fail_closed_arm() -> None:
    requirements = _load(VISIBLE_REASONING)
    current.validate_visible_reasoning_requirements(requirements)
    assert requirements["status"] == "blocked_pending_visible_reasoning_qualification"
    assert requirements["known_base_identity"]["model_revision"] == (
        "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
    )
    assert requirements["required_profile"]["enable_thinking"] is True
    assert requirements["required_profile"]["reasoning_visibility"] == "student_visible"
    assert requirements["separation_contract"] == {
        "new_collection_arm": True,
        "new_corpus_schema_required": True,
        "mix_with_visible_action_corpus": False,
        "teacher_private_or_unknown_reasoning_allowed": False,
        "inferred_or_reconstructed_reasoning_allowed": False,
    }
    assert requirements["distinct_contract_schemas"] == {
        "source_profile": "cyber_qwen_opencode_student_visible_reasoning_source_profile_v1",
        "packet": "cyber_qwen_opencode_student_visible_reasoning_packet_v1",
        "selection": "cyber_qwen_opencode_student_visible_reasoning_selection_v1",
        "record": "cyber_qwen_opencode_student_visible_reasoning_record_v1",
        "corpus": "cyber_qwen_opencode_visible_reasoning_sft_corpus_v1",
    }
    assert (
        requirements["required_reasoning_admission_policy"]["action_packet_reuse_allowed"] is False
    )
    assert requirements["required_reasoning_admission_policy"]["reject"] == [
        "private_or_unknown_reasoning",
        "opaque_compaction",
        "unknown_serialization",
        "heldout_family",
    ]
    assert requirements["required_serialization_round_trip"] == {
        "prompt_token_ids_per_case": True,
        "collection_training_and_serving_full_token_ids_per_case": True,
        "prompt_messages": "messages_without_target",
        "prompt_add_generation_prompt": True,
        "prompt_token_ids_must_equal_local_apply_chat_template": True,
        "prompt_token_ids_exact_prefix_of_every_full_serialization": True,
    }
    assert requirements["admitted_roster"]["root_role_anchor_id"] == (
        task_family_split.TRUSTED_FLEET_COLLECTION_ROOT_ID
    )
    assert (
        requirements["admitted_roster"]["family_split_sha256"]
        == _load(OUTPUT / "family-split.json")["sha256"]
    )
    assert {
        "authorized_reasoning_schema_allowlist_entry",
        "template_serialization_token_round_trip_receipt",
        "continuation_and_reload_token_equivalence_receipt",
        "heldout_zero_admission_regression_receipt",
        "source_session_trajectory_and_window_dedupe_regression_receipt",
        "pinned_immutable_reasoning_success_evidence_mapping",
        "aggregate_reasoning_census_and_selection_receipt",
    } <= set(requirements["missing_launch_gates"])


def test_unproven_task_quality_wave_is_aggregate_only_and_final_family_excluding() -> None:
    requirements = _load(TASK_QUALITY)
    current.validate_task_quality_requirements(requirements)
    assert requirements["candidate_supply"] == {
        "current_production_blackbox": 1093,
        "current_with_exact_execution_receipt": 80,
        "without_exact_execution_receipt": 1013,
        "qa_clean_without_exact_receipt": 7,
        "qa_agent_failure_without_exact_receipt": 10,
        "qa_not_analyzed_without_exact_receipt": 949,
        "known_broken_without_exact_receipt_excluded": 47,
    }
    assert requirements["batch_policy"]["initial_priority_candidates"] == 17
    assert requirements["batch_policy"]["next_unreviewed_backlog"] == 949
    assert requirements["batch_policy"]["maximum_task_versions_per_family"] == 1
    assert requirements["receipt_proven_supply"] == {
        "total_with_exact_execution_receipt": 80,
        "admitted_not_known_broken": 75,
        "admitted_exact_success": 42,
        "admitted_exact_canonical_failure": 33,
        "excluded_known_broken_exact_success": 2,
        "excluded_known_broken_exact_canonical_failure": 3,
        "protected_train": 50,
        "protected_dev": 17,
        "protected_final_test": 8,
    }
    waves = requirements["expansion_wave_plan"]
    assert waves["priority_wave"]["candidate_versions"] == 17
    assert waves["not_analyzed_waves"]["full_batch_count"] == 14
    assert waves["not_analyzed_waves"]["final_batch_size"] == 53
    assert 14 * 64 + 53 == waves["not_analyzed_waves"]["candidate_versions"] == 949
    assert waves["nonbroken_unproven_candidate_upper_bound"] == 966
    assert waves["receipt_proven_plus_candidate_upper_bound"] == 1041
    assert waves["upper_bounds_are_not_admission_or_training_authority"] is True
    growth = requirements["collection_growth_policy"]
    assert growth["attempts_per_admitted_train_task_version"] == 4
    assert growth["use_every_admitted_train_family"] is True
    assert growth["exact_cell_count_declared_only_after_anchored_split"] is True
    assert growth["student_visible_reasoning_requires_distinct_contract_and_gates"] is True
    assert requirements["final_family_policy"]["protected_final_family_count"] == 8
    assert requirements["final_family_policy"]["qualification_allowed"] is False
    assert requirements["final_family_policy"]["root_role_anchor_id"] == (
        task_family_split.TRUSTED_FLEET_COLLECTION_ROOT_ID
    )
    assert (
        requirements["final_family_policy"]["family_role_anchor_sha256"]
        == (task_family_split.trusted_fleet_collection_root_anchor()["sha256"])
    )
    assert requirements["public_evidence_policy"]["training_data_eligible"] is False
    assert requirements["safety"]["cluster_root_annotation"] == {"fleet.ai/failure-alerts": "off"}
    assert requirements["safety"]["promotion_to_training_automatic"] is False
    assert {
        "exact_environment_startup",
        "exact_bash_and_submit_report_tool_reachability",
        "verifier_process_completion",
        "finite_authoritative_outcome_recording",
        "new_metadata_only_session_ingestion_completion",
        "environment_and_container_cleanup",
    } == set(requirements["qualification_checks"])

    production = _load(ROOT / requirements["inputs"]["current_inventory"]["path"])
    coverage = _load(ROOT / requirements["inputs"]["receipt_coverage"]["path"])
    split = _load(ROOT / requirements["inputs"]["protected_family_split"]["path"])
    missing = set(coverage["no_exact_receipt_in_this_refresh_task_keys"])
    statuses = Counter(
        row["qa_status"] for row in production["tasks"] if row["task_key"] in missing
    )
    assert statuses == Counter(not_analyzed=949, broken_task=47, agent_failure=10, clean=7)
    final_groups = sorted(row["group_id"] for row in split["tasks"] if row["split"] == "final_test")
    final_digest = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(final_groups, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )
    assert final_digest == requirements["final_family_policy"]["protected_final_group_ids_sha256"]
