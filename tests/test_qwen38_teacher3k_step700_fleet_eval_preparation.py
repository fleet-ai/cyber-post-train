"""No-launch Fleet dev17 packet for sealed Teacher3K 32K step 700."""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path

from cyber_post_train.jobs import digest

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "configs/evaluation"
PACKET = EVAL / "qwen38-teacher3k-32k-step700-fleet-dev17-seed43-preparation-v1.json"
BINDINGS = EVAL / "qwen38-fleet-dev17-exact-binding-roster-20260922-v1.json"
TASK_SET = EVAL / "qwen38-fresh75-fleet-dev17-task-set-v1.json"
BASE = EVAL / "qwen38-base-fleet-dev17-opencode-seed43-pass1-v1.json"
PROTOCOL = EVAL / "qwen38-fleet-dev17-seed43-matched-protocol-v1.json"
SPLIT = ROOT / "configs/data/fleet-blackbox-current-study-split-20260914-v2.json"
CORPUS = ROOT / "configs/data/qwen38-teacher3k-32k-v1.manifest.json"
EVIDENCE = ROOT / "docs/evidence/qwen38-teacher3k32-step700-preservation-20260922.json"


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def file_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def assert_self_digest(value: dict, *, prefixed: bool) -> None:
    expected = digest({key: item for key, item in value.items() if key != "sha256"})
    assert value["sha256"] == (f"sha256:{expected}" if prefixed else expected)


def test_packet_is_self_digesting_and_evaluation_created_nothing() -> None:
    packet = read(PACKET)
    assert_self_digest(packet, prefixed=False)
    assert packet["status"] == "blocked_not_launchable"
    assert packet["launchable"] is False
    assert packet["scientific_boundary"]["capability_claimed"] is False
    assert packet["scientific_boundary"]["evaluation_only"] is True
    assert packet["operation"] == {
        "fleet_gets_for_sanitized_exact_binding_roster": 18,
        "fleet_mutating_requests": 0,
        "jobs_submitted": 0,
        "config_maps_created": 0,
        "databases_created": 0,
        "outputs_created": 0,
        "routes_resumed_or_created": 1,
        "evaluation_sessions_created": 0,
        "gh_pages_writes": 0,
    }


def test_promotion_evidence_records_accepted_stage_and_paused_zero_gpu_route() -> None:
    evidence = read(EVIDENCE)
    assert_self_digest(evidence, prefixed=False)
    assert evidence["status"] == "accepted_through_paused_serving_registration"
    assert evidence["staging"]["status"] == "accepted_atomic_create_once_stage"
    assert evidence["staging"]["pod_and_config_map_absent_after_cleanup"] is True
    registration = evidence["serving_registration"]
    assert registration["phase"] == "paused"
    assert registration["active_pods"] == registration["ready_replicas"] == 0
    assert registration["routing_enabled"] is False
    assert registration["serving_qualified"] is False
    assert evidence["live_parity"]["state"] == "unaccepted_not_run_capacity_gate_closed"


def test_dev17_is_family_held_out_but_not_application_held_out() -> None:
    packet, split, corpus = read(PACKET), read(SPLIT), read(CORPUS)
    assessment = packet["scientific_assessment"]
    assert assessment["verdict"].startswith("suitable_for_family_held_out_development")
    assert "application_held_out_claims" in assessment["not_suitable_for"]
    assert split["generalization_scope"] == (
        "task-family-held-out; applications are represented in every split"
    )
    assert split["leakage_checks"] == {
        "all_inventory_versions_assigned_once": True,
        "exact_identity_overlap": 0,
        "reviewed_family_overlap": 0,
    }
    application_counts = split["representation"]["application"]["split_counts"]
    assert set(application_counts) == {"train", "dev", "final_test"}
    assert (
        set(application_counts["train"])
        == set(application_counts["dev"])
        == set(application_counts["final_test"])
    )
    assert all(count > 0 for counts in application_counts.values() for count in counts.values())
    assert assessment["split"]["file_sha256"] == file_sha256(SPLIT)
    assert assessment["split"]["counts"] == {"train": 50, "dev": 17, "final_test": 8}

    isolation = assessment["training_isolation"]
    assert isolation["corpus_manifest_file_sha256"] == file_sha256(CORPUS)
    assert corpus["split_sha256"] == split["sha256"] == isolation["split_sha256"]
    assert corpus["catalog_provenance"]["held_out_task_families_excluded_across_all_versions"] == 25
    assert corpus["rechunk_provenance"]["held_out_task_families_excluded_across_all_versions"] == 25
    assert isolation["dev_plus_final_task_families"] == 17 + 8


def test_exact_binding_roster_freezes_task_runtime_seed_and_verifier_identities() -> None:
    packet, roster, task_set = read(PACKET), read(BINDINGS), read(TASK_SET)
    assert_self_digest(roster, prefixed=True)
    assert roster["binding_count"] == len(roster["bindings"]) == len(task_set["tasks"]) == 17
    assert roster["bindings_sha256"] == "sha256:" + digest(roster["bindings"])
    assert roster["source"]["task_set_file_sha256"] == file_sha256(TASK_SET)
    assert roster["source"]["selection_sha256"] == task_set["selection_sha256"]
    assert roster["source"]["team_name"] == "fleet"
    assert roster["source"]["team_id"] == "a1025f0b-ad67-49fc-a023-51800ab43e84"

    selected = {row["task_version_id"]: row for row in task_set["tasks"]}
    observed = {row["task"]["version_id"]: row for row in roster["bindings"]}
    assert set(observed) == set(selected)
    for version_id, binding in observed.items():
        source = selected[version_id]
        task, environment, verifier = (
            binding["task"],
            binding["environment"],
            binding["verifier"],
        )
        assert task["key"] == source["task_key"]
        assert environment == {
            "id": source["env_key"],
            "version": source["env_version"],
            "version_id": source["environment_version_id"],
            "data_id": source["data_key"],
            "data_version": source["data_version"],
            "runtime_seed_content_sha256": environment["runtime_seed_content_sha256"],
            "ttl_seconds": 32400,
        }
        assert task["cyber_contract"] == {
            "evidence_schema": "1.0.0",
            "submission_protocol": "2.0.0",
            "verifier_contract": "3.0.0",
        }
        for field in ("id", "version_id"):
            uuid.UUID(verifier[field])
        assert verifier["version"] == 1
        assert verifier["function_name"] == "verify"
        assert len(verifier["sha256"]) == len(environment["runtime_seed_content_sha256"]) == 64

    frozen = packet["scientific_assessment"]["exact_binding_roster"]
    assert frozen["path"] == str(BINDINGS.relative_to(ROOT))
    assert frozen["file_sha256"] == file_sha256(BINDINGS)
    assert frozen["bindings_sha256"] == roster["bindings_sha256"]
    assert frozen["receipt_sha256"] == roster["sha256"]
    assert all(
        not roster["privacy"][field]
        for field in (
            "prompts_included",
            "environment_variables_included",
            "output_schemas_included",
            "runtime_seed_content_included",
            "verifier_code_included",
            "scores_sessions_rollouts_or_results_included",
        )
    )


def test_accepted_base_is_complete_and_fixed_without_copying_outcomes() -> None:
    packet = read(PACKET)
    baseline = packet["accepted_baseline"]
    assert baseline["state"] == "accepted_complete"
    assert baseline["required_cells"] == baseline["accepted_cells"] == baseline["valid_cells"] == 17
    assert baseline["technical_invalid_cells"] == 0
    assert baseline["automatic_checker_results"] == 17
    assert baseline["distinct_automatic_checker_execution_ids"] == 17
    assert baseline["missing_automatic_checker_results"] == 0
    assert baseline["missing_local_records"] == baseline["duplicate_local_records"] == 0
    assert baseline["capability_outcomes_included_in_this_packet"] is False
    assert baseline["config"]["file_sha256"] == file_sha256(BASE)
    assert baseline["aggregate_evidence"]["commit"] == ("ac8f79e51fc322461d14f80806d1f9aa81d28e68")
    assert baseline["receipt_integrity"]["distinct_receipt_digests"] == 17
    assert baseline["receipt_integrity"]["missing_receipt_digests"] == 0
    assert baseline["scientific_use"].startswith("reuse_without_rerun_only_if")


def test_step700_binds_promotion_through_paused_registration_but_not_live_parity() -> None:
    packet = read(PACKET)
    checkpoint = packet["candidate_checkpoint"]
    assert checkpoint["optimizer_step"] == 700
    assert checkpoint["checkpoint_path"].endswith("/checkpoints/global_step_700")
    assert checkpoint["checkpoint_receipt"]["path"].endswith(
        "/checkpoint_receipts/step-000700.json"
    )
    assert checkpoint["checkpoint_receipt"]["receipt_sha256"] == (
        "sha256:cab127f88955396894b1676756f8d6723363db6eaddcbe55f9233b12d2be9f44"
    )
    assert checkpoint["seal_manifest"]["path"].endswith("/checkpoint-seals-v1/step-700.json")
    assert checkpoint["seal_manifest"]["receipt_sha256"] == (
        "sha256:ef28389120f998a8da8c2ab17a2aaefdd123378c69dde55e2bce88195c6e14c5"
    )
    assert checkpoint["file_count"] == 33
    assert checkpoint["world_size"] == 8
    assert checkpoint["total_bytes"] == 324627486795
    assert checkpoint["supervised_tokens_at_step"] == 21596883
    assert checkpoint["gpu_reload_verified"] is True

    gates = packet["promotion_gates"]
    assert gates["export"]["state"] == "accepted"
    assert gates["export"]["optimizer_steps"] == 0
    assert gates["export"]["gpu_reload_verified"] is False
    assert gates["export"]["payload_manifest_sha256"] is None
    assert gates["cpu_layout"]["state"] == "accepted_synthetic_only_not_serving_qualification"
    assert gates["cpu_layout"]["synthetic_only"] is True
    assert gates["cpu_layout"]["serving_qualified"] is False
    assert gates["cpu_layout"]["gpu_reload_verified"] is False
    reload_gate = gates["gpu_reload"]
    assert reload_gate["state"] == "accepted_bounded_zero_update_reload"
    assert reload_gate["gpus"] == 1
    assert reload_gate["optimizer_steps_executed"] == 0
    assert reload_gate["finite_logits"] is True
    assert reload_gate["generated_tokens"] == 2
    assert reload_gate["source_unchanged"] is True
    assert reload_gate["serving_qualified"] is False
    assert reload_gate["pod_absent_after_cleanup"] is True
    assert reload_gate["receipt_sha256"] == (
        "sha256:7fca6bb17f1660579bac96a57e3b5e8c1fe161ee28e7508239f4c6ebe886d306"
    )
    assert reload_gate["complete_receipt_sha256"] == (
        "sha256:d590c924a8b8eedd06dc5a1b86477a485096db428c43aab49c97fdb3631c660a"
    )
    assert reload_gate["receipt_sfs_path"] == (
        "/jobs/chris-q38-study-corpora-v1/launch-controls/"
        "chris-q38-t3k32-s700-gpu-check-v1/GPU_CHECK.json"
    )
    assert "accepted_one_gpu_finite_reload_receipt" not in packet["remaining_before_launch_packet"]
    stage = gates["stage"]
    assert stage["state"] == "accepted_atomic_create_once_stage"
    assert stage["payload_manifest_sha256"] == (
        "sha256:41310cfdc8ea0d096d784c2f86c2d01fba9ad12263426d1e0fcd812bedcde013"
    )
    assert stage["pod_and_config_map_absent_after_cleanup"] is True
    registration = gates["serving_registration"]
    assert registration["state"] == "accepted_paused_zero_gpu_not_live_parity_qualified"
    assert registration["served_id"] == "chris-q38-t3k32-s700-v1"
    assert registration["session_model"] == "chris-q38-t3k32-s700-v1"
    assert registration["active_pods"] == 0
    assert registration["routing_enabled"] is False
    assert registration["serving_qualified"] is False
    parity = gates["live_parity"]
    assert parity["state"] == "unaccepted_not_run_capacity_gate_closed"
    assert parity["receipt_sha256"] is None
    assert parity["capacity_gate"] == {
        "active_project_gpu_nodes": 6,
        "candidate_gpu_nodes_if_resumed": 1,
        "limit_gpu_nodes": 8,
        "queued_rl_gpu_node_claims": 2,
        "state": "closed",
    }
    assert (
        "accepted_atomic_stage_receipt_with_payload_manifest_sha256"
        not in packet["remaining_before_launch_packet"]
    )
    assert "accepted_serving_registration_receipt" not in packet["remaining_before_launch_packet"]


def test_future_config_can_change_only_checkpoint_and_resource_identity() -> None:
    packet, protocol, base = read(PACKET), read(PROTOCOL), read(BASE)
    matched = packet["matched_protocol"]
    assert matched["file_sha256"] == file_sha256(PROTOCOL)
    assert matched["sha256"] == "sha256:" + protocol["sha256"]
    assert matched["selection"]["task_set_file_sha256"] == file_sha256(TASK_SET)
    assert matched["selection"]["binding_roster_file_sha256"] == file_sha256(BINDINGS)
    assert matched["treatment"] == protocol["treatment"]
    assert matched["images"] == protocol["images"]
    assert matched["sampling"] == protocol["sampling"] == base["sampling"]
    assert matched["pass_k"] == protocol["pass_k"] == base["pass_k"] == 1
    assert matched["training_data_eligible"] is False

    future = packet["future_candidate_config"]
    assert future["state"].startswith("deferred")
    assert future["file_sha256"] is future["evaluation_plan_sha256"] is None
    assert future["model_revision_from_staged_payload_manifest_sha256"] == (
        "sha256:41310cfdc8ea0d096d784c2f86c2d01fba9ad12263426d1e0fcd812bedcde013"
    )
    assert future["session_model"] == future["served_id"] == "chris-q38-t3k32-s700-v1"
    assert all(
        future[field] is None
        for field in ("route_catalog", "route_model_info", "route_server_info")
    )
    assert set(matched["allowed_differences"]) == {
        "candidate_model_weight_artifact",
        "candidate_served_model_binding",
        "candidate_resource_identity",
    }
    assert "sampling" in future["copy_exactly_from_accepted_base"]
    assert "max_reviewed_infrastructure_retries" in future["copy_exactly_from_accepted_base"]


def test_duplicate_server_preview_and_result_handoffs_fail_closed() -> None:
    packet = read(PACKET)
    duplicate = packet["duplicate_and_create_once_gates"]
    assert duplicate["baseline_rerun_forbidden"] is True
    assert duplicate["fresh_immediately_before_create"] is True
    assert duplicate["create_requests_authorized_by_this_packet"] == 0
    assert duplicate["uncertain_create_must_not_be_repeated"] is True
    assert duplicate["preview"]["server_dry_runs_required"] == 2
    assert duplicate["preview"]["normalized_previews_must_match"] is True
    assert duplicate["preview"]["required_root_annotation"] == {"fleet.ai/failure-alerts": "off"}
    assert duplicate["preview"]["pod_template_annotation_is_not_sufficient"] is True
    assert duplicate["preview"]["priority_class"] == "c1"
    assert duplicate["preview"]["evaluator_gpu_request"] == 0
    assert packet["future_resource_identities"]["minted"] is False
    assert all(
        packet["future_resource_identities"][field] is None
        for field in ("job", "config_map", "database", "output_root", "pod", "workload")
    )

    handoff = packet["result_and_webapp_handoff"]
    assert handoff["blocked_until_candidate_is_terminal_and_all_17_cells_are_accepted"] is True
    assert handoff["private_result_authority"]["do_not_pair_from_aggregate_public_receipts"] is True
    assert handoff["candidate_acceptance"]["accepted_cells_required"] == 17
    assert handoff["candidate_acceptance"]["valid_automatic_checker_results_required"] == 17
    assert handoff["candidate_acceptance"]["selective_replay_forbidden"] is True
    assert handoff["public_aggregate_receipt"]["must_be_absent_before_terminal_acceptance"] is True
    assert handoff["webapp"]["publish_before_candidate_terminal_acceptance"] is False
    assert handoff["webapp"]["external_mutations_authorized_by_this_packet"] == 0


def test_committed_files_contain_no_raw_results_or_credentials() -> None:
    for path in (PACKET, BINDINGS):
        serialized = path.read_text().lower()
        assert "bearer " not in serialized
        assert "authorization:" not in serialized
        assert '"api_key"' not in serialized
        assert '"prompt":' not in serialized
        assert '"verifier_code":' not in serialized
        assert '"rollout_trace":' not in serialized
        assert '"aggregate_result":' not in serialized
        assert '"score":' not in serialized
        assert '"flag":' not in serialized
