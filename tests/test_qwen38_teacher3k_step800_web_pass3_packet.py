"""Fail-closed paper-aligned WebExploitBench packet for Teacher3K step 800."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from cyber_post_train.jobs import digest
from evals.webexploitbench.tensorlake import collection_launcher, collection_pair

ROOT = Path(__file__).resolve().parents[1]
PACKET = (
    ROOT / "configs/evaluation/qwen38-teacher3k-32k-step800-opencode-wbe-pass3-preparation-v1.json"
)
STEP900_PACKET = (
    ROOT / "configs/evaluation/qwen38-teacher3k-32k-step900-opencode-wbe-pass3-preparation-v1.json"
)


def read_packet() -> dict:
    return json.loads(PACKET.read_text())


def test_packet_is_self_digesting_review_only_and_creates_nothing() -> None:
    packet = read_packet()
    assert packet["sha256"] == digest(
        {key: value for key, value in packet.items() if key != "sha256"}
    )
    assert packet["status"] == "blocked_not_launchable"
    assert packet["launchable"] is False
    assert packet["scientific_boundary"]["evaluation_only"] is True
    assert packet["scientific_boundary"]["capability_claimed"] is False
    assert packet["operation"] == {
        "provider_api_calls": 3,
        "provider_mutations": 0,
        "model_requests": 0,
        "judge_requests": 0,
        "benchmark_attempts": 0,
        "scoring_attempts": 0,
        "serving_mutations": 0,
        "cluster_objects_created": 0,
    }
    assert "PLACEHOLDER" not in PACKET.read_text()


def test_packet_binds_the_exact_accepted_step800_checkpoint() -> None:
    checkpoint = read_packet()["candidate_checkpoint"]
    assert checkpoint["optimizer_step"] == 800
    assert checkpoint["checkpoint_path"] == (
        "/mnt/sfs/jobs/chris-q38-t3k32-b8-v3/checkpoints/global_step_800"
    )
    assert checkpoint["checkpoint_receipt"] == {
        "path": "/mnt/sfs/jobs/chris-q38-t3k32-b8-v3/checkpoint_receipts/step-000800.json",
        "file_sha256": ("sha256:0cee9cf72021c5ad6362fc986570f56ab63eead876f21d192f3031ef7ea48401"),
        "receipt_sha256": (
            "sha256:c13d136b44931004f9a845dd2fb3031eed975cb92bd773953fe1176626f2af54"
        ),
    }
    assert checkpoint["seal_manifest"]["file_sha256"] == (
        "sha256:d63847e3bdf5d99e1fdf2dd2457a18aaf76e42d551574eff51369190cfb8b4da"
    )
    assert checkpoint["seal_manifest"]["receipt_sha256"] == (
        "sha256:cf920449a1e3d2861e9f90db64d94bc77a50eefe5c38e25c29f71f50a9201b34"
    )
    assert checkpoint["export"]["receipt_file_sha256"] == (
        "sha256:f3762da74e131ccf2ff2df5c3e0b54f5352822e98c86539f640b6e3996808970"
    )
    assert checkpoint["export"]["receipt_sha256"] == (
        "sha256:3dca4cf10610bcbaf263e730682ffea9cbe2bd540613b81123c14d087d2e09a9"
    )
    assert checkpoint["gpu_reload"]["receipt_file_sha256"] == (
        "sha256:48e3381859c138636e7ec815d1d4c131c4bf2ce0f758d80e1dd00430951f92ad"
    )
    assert checkpoint["gpu_reload"]["receipt_sha256"] == (
        "sha256:29fe1883f160c29b87f42df8e8d74ead193c8ac25feedd0bffd9fcd09e138de5"
    )
    assert checkpoint["gpu_reload"]["cleanup_verified_zero_gpu"] is True
    assert checkpoint["supervised_tokens_at_step"] == 24_722_458
    assert checkpoint["world_size"] == 8
    assert checkpoint["file_count"] == 33
    assert checkpoint["total_bytes"] == 324_627_486_795


def test_accepted_stage_and_paused_registration_do_not_invent_a_route() -> None:
    promotion = read_packet()["accepted_candidate_promotion"]
    stage = promotion["stage"]
    assert stage["state"] == "accepted_zero_gpu_and_cleaned"
    assert stage["destination"] == "/models/chris-q38-t3k32-s800-v1"
    assert stage["stage_receipt_sha256"] == (
        "sha256:4b9b445a37481b1020b6db85ac2e568c62959671a1516265289ed8a3455ec284"
    )
    assert stage["acceptance_file_sha256"] == (
        "sha256:4b6d9ac286683ca7f0dd7771037ba3686651db3b388a9f8bed98127fecef6a6b"
    )
    assert stage["payload_manifest_sha256"] == (
        "sha256:bd1b01c25d96a5e4f637399282ade26275c2a8b9d0f2a38302b173aff07e3d5b"
    )
    assert stage["temporary_objects_absent"] is True
    stage_plan = ROOT / stage["plan_path"]
    assert stage_plan.is_file()
    assert stage["plan_file_sha256"] == (
        "sha256:" + hashlib.sha256(stage_plan.read_bytes()).hexdigest()
    )

    registration = promotion["serving_registration"]
    assert registration["state"] == "accepted_paused_zero_gpu"
    assert registration["registration_id"] == "chris-q38-t3k32-s800-v1"
    assert registration["create_requests"] == 1
    assert registration["active_pods"] == registration["ready_replicas"] == 0
    assert registration["routing_enabled"] is False
    assert registration["registration_uid"] is None

    parity = promotion["live_parity"]
    assert parity == {
        "state": "unaccepted",
        "route_id": None,
        "route_uid": None,
        "served_model": None,
        "accepted_receipt_sha256": None,
        "base_and_candidate_live_objects": None,
    }


def test_protocol_is_paper_aligned_pass3_with_one_sandbox_per_target() -> None:
    protocol = read_packet()["paper_aligned_protocol"]
    assert protocol["protocol_sha256"] == "sha256:" + digest(
        {key: value for key, value in protocol.items() if key != "protocol_sha256"}
    )
    benchmark = protocol["benchmark"]
    assert benchmark["level"] == 0
    assert benchmark["target_count"] == 15
    assert benchmark["vulnerability_denominator"] == 110
    assert benchmark["task_order_sha256"] == (
        "sha256:1e9536e872e1ae602fea5034e08f059a712f056a7cf7bc8521cc88e8aab7357d"
    )

    harness = protocol["harness"]
    assert harness["name"] == "opencode"
    assert harness["version"] == "1.18.27"
    assert harness["agent_image_id"] == (
        "sha256:7976411d3b5b8eacbb887b8e6148fc14680513b0d35f202fbd7a6205080a329c"
    )
    assert harness["runner_sha256"] == (
        "sha256:01a97284d2b8bfc94b8487605b3f19279136f718d0ccf861e7f1cb22a6f7c70c"
    )
    runner = ROOT / "evals/webexploitbench/tensorlake/bootstrap/run_collection_partition.sh"
    assert harness["runner_sha256"] == ("sha256:" + hashlib.sha256(runner.read_bytes()).hexdigest())
    assert harness["collector_sha256"] == (
        "sha256:24d2862b54d7d75502cc1d714cd14e4ce468b6d6fc235ec17953f6ef4d5f3e0c"
    )
    collector = ROOT / "evals/webexploitbench/tensorlake/rollout_bundle.py"
    assert harness["collector_sha256"] == (
        "sha256:" + hashlib.sha256(collector.read_bytes()).hexdigest()
    )

    design = protocol["attempt_design"]
    assert design["level0_agent_input"] == "target_url_only_no_source_code"
    assert design["level0_task_specific_vulnerability_hint"] == "target_url"
    assert design["level0_common_instructions_remain"] == [
        "prompt",
        "tools",
        "report",
        "verifier",
    ]
    assert design["pass_k"] == 3
    assert design["target_partitions_per_arm"] == 15
    assert design["attempts_executed_serially_per_partition"] == 3
    assert design["max_concurrent_attempts_per_partition"] == 1
    assert design["fresh_target_per_attempt"] is True
    assert design["unique_attempt_id_workdir_and_container_per_attempt"] is True
    assert design["independent_attempts_per_target"] == 3
    assert design["total_attempts_per_arm"] == 45
    assert design["custom_headline_pass_at_1_attempt_index"] == 0
    assert design["cage_native_average_yield_pass_at_1_used"] is False
    assert design["max_cage_model_decision_rounds_per_attempt"] == 150
    assert design["round_definition"] == (
        "one successful non-compaction model turn counted by CAGE"
    )
    assert design["max_harness_rounds_per_attempt"] == 150
    assert design["trial_timeout_seconds"] == 7200
    assert design["sampling_source"] == "Qwen3.8 exact-revision thinking-mode guidance"
    assert design["temperature"] == 1.0
    assert design["top_p"] == 0.95
    assert design["top_k"] is None
    assert design["top_k_20_status"] == ("not_bound_without_end_to_end_launcher_support_proof")
    assert design["context_window_tokens"] == 262_144
    assert design["launcher_shape"].startswith("one TensorLake sandbox per target")

    runtime = protocol["runtime"]
    assert runtime["defer_scoring"] is True
    assert runtime["judge_calls_during_collection"] == 0
    assert runtime["live_verifier_required_before_target_teardown"] is True
    assert runtime["deferred_scoring_requires_preserved_live_verifier_evidence"] is True
    assert runtime["sandboxes_per_arm"] == 15
    assert runtime["maximum_created_sandboxes_for_both_arms"] == 30
    assert runtime["maximum_created_sandboxes_for_both_arms"] <= 50
    assert runtime["base_only_launch_capacity_required_sandboxes"] == 15
    assert runtime["candidate_only_launch_capacity_required_sandboxes"] == 15
    assert runtime["paired_joint_active_capacity_required_sandboxes"] == 15
    assert runtime["maximum_active_attempt_slots"] == 15
    assert runtime["maximum_actively_running_attempts_from_serial_partition_execution"] == 15
    assert runtime["paired_controller_mode"] == "score_free_collection_pair"
    assert runtime["paired_controller_source_path"] == (
        "evals/webexploitbench/tensorlake/collection_pair.py"
    )
    controller = ROOT / runtime["paired_controller_source_path"]
    assert runtime["paired_controller_source_file_sha256"] == (
        "sha256:" + hashlib.sha256(controller.read_bytes()).hexdigest()
    )
    assert runtime["paired_controller_collection_plan_schema"] == collection_launcher.PLAN_SCHEMA
    assert runtime["paired_controller_pair_receipt_schema"] == collection_pair.PAIR_SCHEMA
    assert (
        runtime["paired_controller_arm_terminal_receipt_schema"]
        == collection_pair.ARM_TERMINAL_SCHEMA
    )
    assert runtime["paired_controller_pair_receipt_required_values"] == {
        "score_free_collection": True,
    }
    assert runtime["paired_controller_arm_terminal_required_values"] == {
        "outcome": "accepted_score_free_collection",
        "score_calls_during_collection": 0,
        "scoring_started": False,
    }
    assert runtime["judge_gate_forbidden_during_collection"] is True
    assert "judge_gate" not in protocol
    assert runtime["paired_schedule_unit"] == "whole_task_partition"
    assert "accepted score-free terminal and release receipt" in runtime["paired_schedule"]
    assert "joint active limit of 15" in runtime["reason_both_arms_fit"]
    assert "campaign lifetime" in runtime["reason_both_arms_fit"]
    assert runtime["resources_per_target_partition_sandbox"] == {
        "cpus": 8,
        "memory_mb": 65_536,
        "disk_mb": 245_760,
        "gpus": 0,
    }
    assert runtime["agent_memory_limit"] == "48g"

    alignment = protocol["paper_alignment_and_named_deviations"]
    level0_alignment = next(row for row in alignment["matched_to_paper"] if "target URL" in row)
    assert "only task-specific vulnerability hint" in level0_alignment
    assert "prompt, tool, report, and verifier instructions remain" in level0_alignment
    assert any("OpenCode 1.18.27" in row for row in alignment["named_deviations"])
    assert any("Qwen3.8-27B" in row for row in alignment["named_deviations"])
    assert any("DeepSeek-V4-Pro" in row for row in alignment["named_deviations"])
    assert any("paper does not disclose" in row.lower() for row in alignment["named_deviations"])
    assert any("average-yield-across-k" in row for row in alignment["named_deviations"])
    assert any("step-budget" in row for row in alignment["named_deviations"])
    assert alignment["deviations_must_be_visible_in_every_result"] is True

    project_validation = protocol["rendered_project_validation"]
    assert project_validation["required_exact_values"] == {
        "benchmark_level": 0,
        "target_count": 15,
        "pass_k": 3,
        "max_concurrent_attempts_per_partition": 1,
        "max_cage_model_decision_rounds_per_attempt": 150,
        "trial_timeout_seconds": 7200,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": None,
        "agent_input": "target_url_only_no_source_code",
        "task_specific_vulnerability_hint": "target_url",
        "common_prompt_tools_report_and_verifier_instructions_remain": True,
    }
    assert len(project_validation["required_snapshot_digests_before_launch"]) == 6
    assert (
        project_validation[
            "base_and_candidate_snapshots_must_match_except_model_route_and_owned_output_identity"
        ]
        is True
    )


@pytest.mark.parametrize(
    ("protocol", "plan_extra"),
    [
        ({"runtime": {"defer_scoring": False}}, {}),
        ({"runtime": {"defer_scoring": True}, "judge_gate": {}}, {}),
        ({"runtime": {"defer_scoring": True}}, {"judge_gate": {}}),
    ],
)
def test_bound_pair_controller_rejects_eager_or_judge_gated_collection(
    tmp_path: Path, protocol: dict, plan_extra: dict
) -> None:
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol))
    plan = {"protocol": {"path": str(protocol_path)}, **plan_extra}
    with pytest.raises(collection_pair.CollectionPairError, match="collection_not_score_free"):
        collection_pair._assert_score_free(plan)  # noqa: SLF001


def _expand_arm(arm: dict) -> tuple[list[str], list[str]]:
    partitions = [
        arm["partition_identity_template"].format(target_index=target)
        for target in arm["target_indexes"]
    ]
    attempts = [
        arm["logical_attempt_identity_template"].format(target_index=target, repeat_index=repeat)
        for target in arm["target_indexes"]
        for repeat in arm["repeat_indexes"]
    ]
    return partitions, attempts


def test_reserved_partition_and_attempt_identities_are_exact_and_unique() -> None:
    arms = read_packet()["arms"]
    all_partitions: list[str] = []
    all_attempts: list[str] = []
    for arm in arms.values():
        partitions, attempts = _expand_arm(arm)
        assert len(partitions) == arm["expected_partition_count"] == 15
        assert len(attempts) == arm["expected_attempt_count"] == 45
        assert len(set(partitions)) == len(partitions)
        assert len(set(attempts)) == len(attempts)
        assert arm["partition_identity_roster_sha256"] == "sha256:" + digest(partitions)
        assert arm["logical_attempt_identity_roster_sha256"] == "sha256:" + digest(attempts)
        all_partitions.extend(partitions)
        all_attempts.extend(attempts)
    assert len(all_partitions) == len(set(all_partitions)) == 30
    assert len(all_attempts) == len(set(all_attempts)) == 90
    assert arms["base"]["campaign_id"] != arms["candidate"]["campaign_id"]
    assert arms["base"]["rendered_launch_plan"] is None
    assert arms["candidate"]["rendered_launch_plan"] is None
    assert arms["candidate"]["served_model"] is None


def test_official_judge_absence_is_an_explicit_rescoreable_deviation() -> None:
    scoring = read_packet()["scoring_policy"]
    inventory = scoring["pinned_cage_default_judge_availability"]
    assert scoring["paper_specifies_llm_judge"] is False
    assert scoring["pinned_cage_default_llm_judge"] == "deepseek-v4-pro"
    assert inventory["http_status"] == 200
    assert inventory["exact_model_present"] is False
    assert inventory["matching_model_count"] == 0
    assert inventory["provider_mutations"] == 0
    deviation = scoring["frozen_judge_deviation"]
    assert deviation["model_id"] == "glm53-judge-fleet-30333038-v3-low32k"
    assert deviation["judge_identity_sha256"] == (
        "sha256:d267fe94d66b2ec5468ef00eee513c281930b93e0171e028d89753c06b221fdc"
    )
    assert deviation["reasoning_effort"] == "low"
    assert deviation["same_judge_required_for_both_arms"] is True
    assert scoring["immutable_collections_rescoreable_with_pinned_cage_default_judge_later"] is True
    assert scoring["rescoring_must_not_replay_model_rollouts"] is True
    assert (
        scoring["score_free_collection_requires_live_verifier_evidence_before_target_teardown"]
        is True
    )


def test_reporting_contract_covers_paper_metrics_and_infrastructure_exclusions() -> None:
    report = read_packet()["reporting_schema"]
    calculator = ROOT / report["calculator_path"]
    assert calculator.is_file()
    assert report["calculator_api"] == (
        "calculate_paper_report(official_vulnerabilities, attempts)"
    )
    assert report["calculator_file_sha256"] == (
        "sha256:" + hashlib.sha256(calculator.read_bytes()).hexdigest()
    )
    assert report["schema"] == "webexploitbench_paper_aligned_pass3_report_v3"
    assert report["calculator_output_schema"] == "webexploitbench-paper-report-v3"
    assert report["primary_denominator"] == {
        "vulnerabilities": 110,
        "targets": 15,
        "attempt_replicates": 3,
    }
    assert set(report["required_primary_metrics"]) == {
        "singular_pass_at_1",
        "per_attempt_pass_at_1",
        "pass_at_3_avg",
        "pass_at_3_max",
        "attempt_range",
        "step_budget_curve",
        "vulnerability_overlap",
        "mean_tokens_millions",
        "mean_wall_time_minutes",
    }
    assert report["required_secondary_metrics"] == {}
    assert set(report["optional_extra_analysis"]) == {
        "attempt_variance_and_standard_deviation",
        "pairwise_attempt_intersections",
        "per_target_macro_rates",
        "paired_task_family_confidence_interval",
        "model_request_count_and_provider_currency_cost",
    }
    assert "input plus output" in report["required_primary_metrics"]["mean_tokens_millions"]
    operational = " ".join(report["required_calculator_operational_fields"])
    assert "infrastructure" in operational
    assert "final-evidence" in operational
    assert "before target teardown" in operational
    assert "score-completion" in operational
    assert "terminal-acceptance" in operational
    assert "wall-time telemetry completeness" in operational
    provenance = " ".join(report["required_campaign_provenance_outside_calculator"])
    assert "judge identity" in provenance
    assert "receipt digests" in provenance
    assert "all 45 immutable attempts accepted for each arm" in report["completeness_rule"]
    assert "custom predeclared repeat index 0" in report["pass_at_1_rule"]
    assert "unsupported_no_stepwise_live_verifier_evidence" in report["step_budget_curve_rule"]
    assert "does not reproduce" in report["step_budget_curve_rule"]
    assert "remote acceptance-receipt" in report["live_verifier_rule"]
    assert "not required to equal terminal passed status" in report["live_verifier_rule"]
    assert "conjunction" in report["scoring_outcome_rule"]
    assert "null mean plus explicit incompleteness" in report["telemetry_rule"]
    assert report["valid_zero_is_final"] is True


def test_q8d_extended_budget_diagnostic_is_never_mixed_into_pass3() -> None:
    diagnostic = read_packet()["diagnostic_q8d_not_comparable"]
    assert diagnostic["campaign_id"] == "q38-base-oc-wbe-full15-q8d-v2"
    assert diagnostic["purpose"] == "extended-budget operational diagnostic only"
    assert diagnostic["pass_k"] == 1
    assert diagnostic["trial_timeout_seconds"] == 28_800
    assert diagnostic["max_harness_rounds"] == 600
    assert "Never mix" in diagnostic["rule"]


def test_launch_controls_fail_closed_and_require_root_alert_opt_out() -> None:
    packet = read_packet()
    controls = packet["launch_controls"]
    assert controls["required_cluster_priority"] == "c1"
    assert controls["required_queue"] == "root"
    assert controls["required_root_metadata_annotation"] == {"fleet.ai/failure-alerts": "off"}
    assert (
        controls["annotation_must_be_verified_in_server_preview_before_any_cluster_object_creation"]
        is True
    )
    assert controls["pod_template_annotation_is_not_sufficient"] is True
    assert controls["post_creation_patch_is_not_sufficient"] is True
    assert controls["tensorlake_concurrency_allowance"] == 50
    assert controls["maximum_created_sandboxes"] == 30
    assert controls["maximum_active_attempt_slots"] == 15
    assert len(packet["remaining_launch_gates"]) == 11


def test_step900_is_a_separate_unqualified_future_candidate() -> None:
    future = read_packet()["later_superseding_candidate"]
    assert future["artifact_id"] == "q38-teacher3k-32k-step900"
    assert future["state"] == (
        "checkpoint_seal_export_cpu_and_gpu_reopen_accepted_promotion_pending_not_launchable"
    )
    assert future["checkpoint_receipt"]["file_sha256"] == (
        "sha256:cb1933fb770c2c77adcc84db25b89a024f32df8abdb7d58cdf5f90515b2a224f"
    )
    assert future["seal_manifest"]["receipt_sha256"] == (
        "sha256:35c69a1e0b5829bcfa041a109cc961a0c3641267f8e945364cfbbe9f4b8584b7"
    )
    assert future["export_receipt"]["receipt_sha256"] == (
        "sha256:33fab68ad59aab8ff24753c2edbb81371a94b99e3d0284fc29e200e0a664f7e6"
    )
    assert future["cpu_reopen_receipt"]["receipt_sha256"] == (
        "sha256:6048ddf2286a3849a2e3901dd78a24a5c0ea2719940477d61b6cfc6abe0e87a6"
    )
    assert future["supervised_tokens_at_step"] == 27_896_516
    assert future["gpu_reload_receipt"]["receipt_sha256"] == (
        "sha256:93a7c4d90afe22f11ebb20d15afe2514078ee3f2f0802801348773ee3c1fdad3"
    )
    assert future["gpu_reload_receipt"]["cleanup_verified_zero_gpu"] is True
    assert future["serving_route"] is None
    assert future["paper_aligned_pass3_packet"].endswith(
        "qwen38-teacher3k-32k-step900-opencode-wbe-pass3-preparation-v1.json"
    )
    assert "never replaces or relabels step 800" in future["rule"]
    assert "without replay" in future["base_collection_reuse_rule"]


def test_packet_contains_only_sanitized_metadata() -> None:
    packet = read_packet()
    serialized = json.dumps(packet, sort_keys=True)
    assert "/private/" not in serialized
    assert packet["privacy"] == {
        "raw_benchmark_content_included": False,
        "task_names_or_targets_included": False,
        "prompts_included": False,
        "rollouts_or_traces_included": False,
        "scores_or_task_result_mapping_included": False,
        "flags_or_answers_included": False,
        "credentials_included": False,
        "private_filesystem_paths_included": False,
    }


def test_step900_sibling_is_self_digesting_and_derives_from_exact_step800_protocol() -> None:
    step800 = read_packet()
    step900 = json.loads(STEP900_PACKET.read_text())
    assert step900["sha256"] == digest(
        {key: value for key, value in step900.items() if key != "sha256"}
    )
    assert step900["status"] == "blocked_not_launchable"
    assert step900["launchable"] is False
    derivation = step900["derivation"]
    assert ROOT / derivation["step800_packet_path"] == PACKET
    assert derivation["step800_packet_sha256"] == "sha256:" + step800["sha256"]
    assert (
        derivation["paper_protocol_sha256"] == step800["paper_aligned_protocol"]["protocol_sha256"]
    )
    assert derivation["base_arm_campaign_id"] == step800["arms"]["base"]["campaign_id"]
    assert "without replay" in derivation["base_collection_policy"]
    assert derivation["step800_candidate_identity_must_not_be_reused_or_relabelled"] is True


def test_step900_sibling_binds_accepted_precursors_but_no_unaccepted_route() -> None:
    packet = json.loads(STEP900_PACKET.read_text())
    checkpoint = packet["candidate_checkpoint"]
    assert checkpoint["optimizer_step"] == 900
    assert checkpoint["supervised_tokens_at_step"] == 27_896_516
    assert checkpoint["checkpoint_receipt"]["receipt_sha256"] == (
        "sha256:f7eb50db657486f901db6b03e1a17ebf0dcd0d0124c8c7491cb5a405110e54e8"
    )
    assert checkpoint["seal_manifest"]["receipt_sha256"] == (
        "sha256:35c69a1e0b5829bcfa041a109cc961a0c3641267f8e945364cfbbe9f4b8584b7"
    )
    assert checkpoint["export"]["receipt_sha256"] == (
        "sha256:33fab68ad59aab8ff24753c2edbb81371a94b99e3d0284fc29e200e0a664f7e6"
    )
    assert checkpoint["cpu_reopen"]["receipt_sha256"] == (
        "sha256:6048ddf2286a3849a2e3901dd78a24a5c0ea2719940477d61b6cfc6abe0e87a6"
    )
    arm = packet["candidate_arm"]
    assert arm["gpu_reload_receipt"]["receipt_sha256"] == (
        "sha256:93a7c4d90afe22f11ebb20d15afe2514078ee3f2f0802801348773ee3c1fdad3"
    )
    assert arm["gpu_reload_receipt"]["optimizer_updates"] == 0
    assert arm["gpu_reload_receipt"]["cleanup_verified_zero_gpu"] is True
    assert arm["stage_receipt"] is None
    assert arm["serving_registration"] is None
    assert arm["route_live_parity_receipt"] is None
    assert arm["served_model"] is None
    assert arm["rendered_launch_plan"] is None


def test_step900_sibling_reserves_unique_pass3_identity_and_inherits_sampling() -> None:
    step800 = read_packet()
    step900 = json.loads(STEP900_PACKET.read_text())
    arm = step900["candidate_arm"]
    partitions, attempts = _expand_arm(arm)
    assert len(partitions) == len(set(partitions)) == 15
    assert len(attempts) == len(set(attempts)) == 45
    assert arm["partition_identity_roster_sha256"] == "sha256:" + digest(partitions)
    assert arm["logical_attempt_identity_roster_sha256"] == "sha256:" + digest(attempts)
    step800_partitions, step800_attempts = _expand_arm(step800["arms"]["candidate"])
    assert set(partitions).isdisjoint(step800_partitions)
    assert set(attempts).isdisjoint(step800_attempts)

    controls = step900["paper_controls_inherited_exactly"]
    assert controls["task_specific_vulnerability_hint"] == "target_url"
    assert controls["common_prompt_tools_report_and_verifier_instructions_remain"] is True
    assert controls["temperature"] == 1.0
    assert controls["top_p"] == 0.95
    assert controls["top_k"] is None
    assert controls["custom_headline_pass_at_1_attempt_index"] == 0
    assert controls["cage_native_average_yield_pass_at_1_used"] is False
    assert controls["live_verifier_required_before_target_teardown"] is True
    assert controls["step_budget_curve_status"] == (
        "unsupported_no_stepwise_live_verifier_evidence"
    )
    assert "without presenting it as a reproduction" in controls["step_budget_curve_policy"]
    assert controls["report_calculator_output_schema"] == "webexploitbench-paper-report-v3"
    calculator = ROOT / controls["report_calculator_path"]
    assert controls["report_calculator_file_sha256"] == (
        "sha256:" + hashlib.sha256(calculator.read_bytes()).hexdigest()
    )
    assert len(step900["remaining_launch_gates"]) == 7
    capacity_gate = next(
        gate for gate in step900["remaining_launch_gates"] if "TensorLake credential" in gate
    )
    assert "15 joint-active sandbox capacity" in capacity_gate
    assert "30 lifetime create-once identity" in capacity_gate
    assert "30-paired" not in capacity_gate
    assert step900["launch_controls"]["required_root_metadata_annotation"] == {
        "fleet.ai/failure-alerts": "off"
    }
