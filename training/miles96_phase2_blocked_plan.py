"""Launch-blocked authority for one v004-shaped Miles cyber update."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from training import miles96_mechanics_canary as mechanics
from training import miles96_signal_qualification as signal
from training import miles_signal_launch_review, miles_signal_wave

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "configs/qualification/qwen38-miles96-phase2-blocked-v1.json"
SCHEMA = "cyber_qwen38_miles96_phase2_blocked_v1"
REVIEW_COMMIT = "7858ead1f2c498850816168fded1be9d3927fa32"
ADAPTER_COMMIT = "978df19a1f6b344e2f88d9502060700a59294681"
ADAPTER_SOURCE_CLOSURE = "sha256:de52b39f55e92a079bef1c0ea14b9e823e4af5096318dca5cf48cc800225fd15"
V004_COMMIT = "10afa8d064bb3dd1c11c50768590e432dfa69097"


def _file_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate_bindings(review: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "name": row["name"],
            "phase1_run_dir": row["run_dir"],
            "phase1_plan_sha256": row["plan_sha256"],
            "task_key": row["task"]["key"],
            "task_version_id": row["task"]["version_id"],
            "environment_version_id": row["environment"]["version_id"],
            "verifier_version_id": row["verifier"]["version_id"],
            "task_family": row["task_family"],
            "split": row["split"],
            "authority_receipt_sha256": row["authority_receipt_sha256"],
            "live_binding_receipt_sha256": row["live_binding_receipt_sha256"],
        }
        for row in review["candidate_bindings"]
    ]


def _prompt_schedule(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    schedule = []
    for replicate in range(2):
        for row in candidates:
            schedule.append(
                {
                    "prompt_group_index": len(schedule),
                    "candidate_name": row["name"],
                    "replicate_index": replicate,
                    "samples": 4,
                }
            )
    return schedule


def expected_body() -> dict[str, Any]:
    review = miles_signal_launch_review.load()
    wave = miles_signal_wave.load()
    candidates = _candidate_bindings(review)
    return {
        "schema": SCHEMA,
        "state": "blocked_pending_four_task_signal_and_v004_adapter",
        "launchable": False,
        "purpose": (
            "After all four exact train tasks independently prove fresh reward signal, run "
            "one v004-shaped 32-rollout wave, one finite optimizer update, one complete "
            "checkpoint/HF export, and a separate zero-update reload check."
        ),
        "authority": {
            "score_blind_review_commit": REVIEW_COMMIT,
            "score_blind_review_sha256": review["sha256"],
            "score_blind_review_file_sha256": _file_sha256(miles_signal_launch_review.REVIEW_PATH),
            "wave_sha256": wave["sha256"],
            "production_split_sha256": wave["authorities"]["production_split"]["self_sha256"],
            "task_set_sha256": wave["authorities"]["task_set"]["self_sha256"],
            "adapter_commit": ADAPTER_COMMIT,
            "runtime_source_closure_sha256": ADAPTER_SOURCE_CLOSURE,
            "model_revision": signal.HF_MODEL_REVISION,
            "model_root": signal.HF_MODEL_ROOT,
            "model_binding_sha256": signal.HF_MODEL_BINDING_SHA256,
            "trainer_image": mechanics.IMAGE,
            "theseus_commit": mechanics.THESEUS_COMMIT,
            "image_source_commit": mechanics.IMAGE_SOURCE_COMMIT,
            "miles_commit": mechanics.MILES_COMMIT,
            "fti_version": mechanics.FTI_VERSION,
        },
        "v004_reference": {
            "repository": "fleet-ai/dataminer_v2",
            "commit": V004_COMMIT,
            "sources": {
                "launcher": {
                    "path": "experiments/rl-transfer-v004/run_v004.py",
                    "git_blob": "60de44aac4659b0f48f0c876eb01c0d45b625486",
                },
                "protocol": {
                    "path": "experiments/rl-transfer-v004/protocol.md",
                    "git_blob": "dcc2cc3004526397b2904975bd61ef8211d8224a",
                },
                "base_easy_job": {
                    "path": (
                        "experiments/rl-transfer-v004/jobs/"
                        "neeraj-dataminer-transfer-v004-base-easy.yaml"
                    ),
                    "git_blob": "5ece2def0ca5cb41787cb6205e09b1a8636586ad",
                },
                "state": {
                    "path": "experiments/rl-transfer-v004/state.md",
                    "git_blob": "37ca36b187adab965c483f50f5e5abd74e72e922",
                },
            },
            "historical_image": (
                "radixark/miles@sha256:"
                "a7ef79b5c0d0cb1a5d6ab0f39a2693ea63ea369d9339d04b7b3e0b7ec23b5dfe"
            ),
            "proven_recipe_shape": {
                "nodes": 1,
                "gpus_per_node": 8,
                "accelerator": "B300",
                "full_parameter_training": True,
                "tensor_parallel": 4,
                "context_parallel": 2,
                "effective_context_tokens": 98_304,
                "max_train_tokens_per_gpu": 8_192,
                "optimizer_cpu_offload": True,
                "prompt_groups": 8,
                "samples_per_prompt": 4,
                "global_batch_size": 32,
                "max_concurrent_episodes": 32,
                "optimizer": "adam",
                "learning_rate": 2e-6,
                "weight_decay": 0.1,
                "adam_betas": [0.9, 0.98],
                "sampling_temperature": 0.7,
                "sampling_top_p": 0.8,
                "sampling_top_k": 20,
                "checkpoint_interval": 4,
            },
            "grounding_notes": [
                "The checked-in Job passes --lr 2e-6 and overrides --max-seq-len to 98304.",
                (
                    "run_v004.py sets TP4, CP2 on one colocated 8-GPU node, dynamic "
                    "8192 train tokens per GPU, optimizer CPU offload, Adam betas, and "
                    "sampling values."
                ),
                (
                    "protocol.md registers the wave-synchronous 8-prompt by 4-sample "
                    "shape, concurrency 32, and save every four updates."
                ),
                (
                    "state.md records that the base-easy arm completed 80 updates and "
                    "wrote periodic checkpoints through iteration 79."
                ),
            ],
        },
        "candidate_bindings": candidates,
        "phase1_prerequisite": {
            "required_terminal_phase1_lanes": 4,
            "required_qualified_lanes": 4,
            "each_lane_requires_eight_terminal_slots": True,
            "each_lane_minimum_completed_gradeable_episodes": 2,
            "each_lane_requires_distinct_finite_rewards": True,
            "each_lane_requires_unique_verifier_executions": True,
            "each_lane_requires_all_instances_released": True,
            "each_lane_requires_zero_optimizer_steps": True,
            "each_lane_requires_no_checkpoint_artifacts": True,
            "historical_scores_used": False,
            "historical_reward_magnitudes_used": False,
            "fallback_or_task_substitution": False,
        },
        "phase2_prompt_schedule": _prompt_schedule(candidates),
        "future_identity": {
            "train_name": "chris-q38-m96-p2-v004-oneupd-a1",
            "train_run_dir": "/mnt/sfs/jobs/chris-q38-m96-p2-v004-oneupd-a1",
            "wandb_run_id": "chris-q38-m96-p2-v004-oneupd-a1",
            "reload_name": "chris-q38-m96-p2-v004-reload-a1",
            "reload_run_dir": "/mnt/sfs/jobs/chris-q38-m96-p2-v004-reload-a1",
            "phase1_signal_evidence_sha256s": None,
            "phase2_plan_sha256": None,
            "train_request_sha256": None,
            "reload_request_sha256": None,
        },
        "train_contract": {
            "nodes": 1,
            "gpus_per_node": 8,
            "accelerator": "B300",
            "topology_mode": "preferred_single_host",
            "full_parameter_training": True,
            "tensor_parallel": 4,
            "context_parallel": 2,
            "pipeline_parallel": 1,
            "sequence_parallel": True,
            "context_tokens": 98_304,
            "response_tokens": 81_920,
            "max_train_tokens_per_gpu": 8_192,
            "optimizer_cpu_offload": True,
            "priority_class": "c1",
            "queue_priority": "q1",
            "root_annotations": {"fleet.ai/failure-alerts": "off"},
            "jobs_api_failure_alerts": False,
            "requeue_if_preempted": False,
            "prompt_groups": 8,
            "samples_per_prompt": 4,
            "fresh_rollout_count": 32,
            "global_batch_size": 32,
            "max_concurrent_episodes": 32,
            "max_turns": 32,
            "max_tokens_per_turn": 8_192,
            "episode_timeout_seconds": 2_400,
            "compaction": False,
            "reward_filter": mechanics.REWARD_FILTER,
            "optimizer_steps": 1,
            "rollout_iterations": 1,
            "optimizer": "adam",
            "learning_rate": 2e-6,
            "lr_decay_style": "constant",
            "weight_decay": 0.1,
            "adam_betas": [0.9, 0.98],
            "sampling_temperature": 0.7,
            "sampling_top_p": 0.8,
            "sampling_top_k": 20,
            "checkpoint_interval": 4,
            "force_terminal_checkpoint_after_one_update": True,
            "advantage_estimator": "grpo",
            "grpo_std_normalization": False,
            "per_token_loss": True,
            "kl_loss_coefficient": 0.001,
            "kl_loss_type": "low_var_kl",
            "ppo_clip_low": 0.2,
            "ppo_clip_high": 0.28,
            "seed": 20260924,
            "cpu_request": mechanics.TRAIN_RESOURCES["cpu_request"],
            "cpu_limit": mechanics.TRAIN_RESOURCES["cpu_limit"],
            "memory_request": mechanics.TRAIN_RESOURCES["memory_request"],
            "memory_limit": mechanics.TRAIN_RESOURCES["memory_limit"],
        },
        "optimizer_gate": {
            "prompt_group_count": 8,
            "episodes_per_group": 4,
            "selected_episode_count": 32,
            "every_group_all_four_normally_completed_and_gradeable": True,
            "every_group_all_rewards_finite": True,
            "every_group_minimum_distinct_reward_values": 2,
            "unique_task_instance_ids": 32,
            "unique_verifier_execution_ids": 32,
            "all_episodes_used_task_tools": True,
            "all_task_instances_released_before_update": True,
            "update_if_any_group_gate_fails": False,
        },
        "checkpoint_export_acceptance": {
            "optimizer_updates": 1,
            "periodic_checkpoint_interval_remains_v004_value_four": True,
            "forced_terminal_checkpoint_after_bounded_update": True,
            "native_checkpoint_nonempty_and_manifested": True,
            "raw_hf_one_update_export_complete": True,
            "complete_hf_one_update_export_create_once": True,
            "all_trained_tensors_finite": True,
            "minimum_changed_trained_tensors": 1,
            "frozen_tensor_prefixes": list(mechanics.FROZEN_TENSOR_PREFIXES),
            "frozen_tensors_byte_exact_from_base": True,
            "all_output_files_hashed": True,
            "train_receipt_required": True,
        },
        "reload_contract": {
            "starts_only_after_train_gpu_release": True,
            "nodes": 1,
            "gpus_per_node": 1,
            "priority_class": "c1",
            "queue_priority": "q1",
            "root_annotations": {"fleet.ai/failure-alerts": "off"},
            "jobs_api_failure_alerts": False,
            "requeue_if_preempted": False,
            "optimizer_steps": 0,
            "source_optimizer_updates": 1,
            "context_tokens": 98_304,
            "exact_complete_model_manifest_required": True,
            "fresh_gradeable_episode_required": True,
            "finite_reward_required": True,
            "tool_call_required": True,
            "task_cleanup_required": True,
            "engine_stop_required": True,
            "reload_receipt_required": True,
        },
        "cyber_only_deviations_from_v004": [
            {
                "field": "data",
                "v004": "eight distinct easy dataminer prompts per wave",
                "cyber": (
                    "two deterministic prompt groups for each of four exact train-only "
                    "Fleet cyber tasks"
                ),
                "reason": (
                    "make one 8x4 wave from the four independently qualified task "
                    "bindings without adding unqualified tasks"
                ),
            },
            {
                "field": "training_horizon",
                "v004": "80 optimizer updates",
                "cyber": "exactly one optimizer update",
                "reason": (
                    "phase two is the smallest real mechanics and checkpoint "
                    "qualification, not a full research run"
                ),
            },
            {
                "field": "terminal_checkpoint",
                "v004": "periodic saves every four updates",
                "cyber": (
                    "keep interval four and force a final save/export after the single "
                    "bounded update"
                ),
                "reason": (
                    "one update never reaches the first periodic save but must prove a "
                    "reloadable artifact"
                ),
            },
            {
                "field": "task_signal_gate",
                "v004": "aborted groups are dynamically filtered",
                "cyber": (
                    "all eight groups must be complete, finite, reward-varying, uniquely "
                    "graded, and released before update"
                ),
                "reason": (
                    "do not spend the only update on constant, fabricated, incomplete, "
                    "or leaked reward"
                ),
            },
            {
                "field": "runtime_binding",
                "v004": "historical dataminer Miles image and harness",
                "cyber": (
                    "frozen current FTI/Miles cyber adapter, exact task/env/verifier/tool "
                    "bindings, and exact immutable cyber image"
                ),
                "reason": (
                    "Fleet cyber tasks require the maintained FTI session and verifier contract"
                ),
            },
            {
                "field": "episode_envelope",
                "v004": "dataminer agent loop",
                "cyber": "32 turns, 8192 tokens per turn, 2400 seconds, no compaction",
                "reason": (
                    "reuse the already bounded phase-one signal envelope for this "
                    "mechanics qualification"
                ),
            },
            {
                "field": "model_export",
                "v004": "Megatron checkpoint followed by offline conversion",
                "cyber": (
                    "compose a complete HF model with byte-exact frozen visual/MTP "
                    "tensors, then run a separate zero-update reload"
                ),
                "reason": (
                    "prove the exported multimodal checkpoint is complete and actually reloadable"
                ),
            },
            {
                "field": "operational_safety",
                "v004": "historical c1/q1 Job",
                "cyber": (
                    "c1/q1 plus root failure-alert opt-out, create-once identities, exact "
                    "previews, receipts, and UID-bound release"
                ),
                "reason": "current cluster policy and auditable create-once operation",
            },
        ],
        "known_adapter_gap": {
            "current_first_qualification_shape": {
                "prompt_groups": 1,
                "samples_per_prompt": 8,
                "max_concurrent_episodes": 2,
                "max_train_tokens_per_gpu": 49_152,
                "optimizer_cpu_offload": False,
                "learning_rate": 1e-6,
                "checkpoint_interval": 1,
            },
            "required_successor_shape": "train_contract",
            "adapter_extension_required": True,
            "must_bind_all_four_task_tuples_and_eight_prompt_groups": True,
            "must_render_and_validate_v004_optimizer_and_offload_flags": True,
            "must_prove_forced_terminal_save_without_changing_periodic_interval": True,
            "exact_image_runtime_preflight_required": True,
        },
        "acceptance_contract": {
            "four_phase1_signal_receipts_revalidated_from_sfs": True,
            "prepared_model_rehashed_before_and_after_training": True,
            "exactly_one_optimizer_update": True,
            "finite_optimizer_metrics": True,
            "checkpoint_export_receipt_valid": True,
            "train_instances_released": True,
            "train_gpus_released": True,
            "reload_receipt_valid": True,
            "reload_gpus_released": True,
        },
        "stop_conditions": [
            "any_phase1_lane_is_not_terminally_accounted",
            "any_of_the_four_exact_phase1_tasks_lacks_fresh_reward_variation",
            "any_task_environment_verifier_or_tool_binding_drifts",
            "any_phase1_public_private_native_or_runtime_receipt_is_missing_or_invalid",
            "any_phase1_lane_contains_an_optimizer_or_checkpoint_artifact",
            "the_successor_adapter_does_not_exactly_render_the_v004_shape",
            "any_of_eight_phase2_group_reward_gates_fails",
            "optimizer_update_count_would_differ_from_one",
            "optimizer_loss_gradient_norm_or_updated_tensors_are_nonfinite",
            "forced_terminal_checkpoint_or_complete_export_proof_fails",
            "train_or_reload_identity_preview_duplicate_capacity_or_observer_gate_fails",
            "train_allocation_is_not_released_before_reload",
            "reload_model_episode_cleanup_or_zero_update_proof_fails",
            "any_terminal_gpu_release_is_unverified",
        ],
        "unresolved_launch_gates": [
            "four_fresh_phase1_terminal_and_signal_receipts",
            "successor_adapter_implements_and_tests_exact_v004_train_contract",
            "exact_four_task_eight_group_phase2_plan_and_request_digests",
            "fresh_task_get_and_live_tool_schema_receipts_for_all_four_tasks",
            "fresh_prepared_model_inventory_rehash",
            "genuine_exact_image_runtime_preflight_for_v004_phase2_and_reload",
            "fresh_jobs_kubernetes_sfs_and_wandb_duplicate_absence",
            "two_identical_train_server_previews_with_root_alert_off_c1_q1_and_backoff_zero",
            "fresh_eight_gpu_capacity_gate_and_prearmed_train_cleanup_observer",
            "durable_single_post_intent_and_independent_train_approval",
            "accepted_train_receipt_and_uid_bound_train_gpu_release",
            "two_identical_reload_server_previews_with_root_alert_off_c1_q1_and_backoff_zero",
            "fresh_one_gpu_capacity_gate_and_prearmed_reload_cleanup_observer",
            "durable_single_post_intent_and_independent_reload_approval",
        ],
        "privacy": {
            "prompt_text_included": False,
            "verifier_code_included": False,
            "environment_values_included": False,
            "credentials_included": False,
            "reward_values_included": False,
            "trajectory_content_included": False,
        },
    }


def load(path: Path = PLAN_PATH) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("phase-two blocked plan must be an object")
    body = {key: item for key, item in value.items() if key != "sha256"}
    if value.get("sha256") != "sha256:" + mechanics.digest(body):
        raise ValueError("phase-two blocked plan self digest changed")
    if body != expected_body():
        raise ValueError("phase-two blocked plan contract changed")
    return value
