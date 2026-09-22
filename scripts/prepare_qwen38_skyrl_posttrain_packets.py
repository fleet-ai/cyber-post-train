#!/usr/bin/env python3
"""Build offline post-training packets for the five broad Qwen3.8 SkyRL arms.

The packets describe gates; they do not stage weights, submit jobs, register
models, start routes, or launch evaluations.  Unknown terminal digests are
represented as required future receipts rather than invented values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts import prepare_qwen38_skyrl_production_queue as queue

ROOT = Path(__file__).resolve().parents[1]
MODEL_LOCK = ROOT / "configs/models/qwen38-27b-1d4bf0f2.lock.json"
MODEL_WEIGHTS = ROOT / "configs/models/qwen38-27b-1d4bf0f2.weights.json"
SERVING_TEMPLATE = ROOT / "configs/qualification/qwen38-fresh75-step230-inference-stage-v2.json"
PACKET_DIR = ROOT / "configs/qualification"
EVIDENCE = ROOT / "docs/evidence/qwen38-study/2026-09-20-skyrl-production-posttrain-queue-v1.json"
POSTTRAIN_ADAPTER = ROOT / "training/skyrl_posttrain.py"
RELOAD_ADAPTER = ROOT / "training/skyrl_prod9_reload.py"

IMAGE = (
    "661864827319.dkr.ecr.us-east-1.amazonaws.com/fleet/skyrl-train@sha256:"
    "89758df2b5f35cdb19efe948c7f6ef54f11e2e2ab47a45d600c25f36914e308f"
)
RELOAD_IMAGE = (
    "661864827319.dkr.ecr.us-east-1.amazonaws.com/fleet/skyrl-train@sha256:"
    "ba288751cd227c5be146d28f4a03237545d87d2cbd4c48464945b17fde566ff4"
)
EXECUTION_CONTRACT = "sha256:6092f664d93d2ff8037b5834135727759b9dd03e4b631bb563b5f4c5984d3b65"


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def object_digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(canonical(value)).hexdigest()


def file_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def source(path: Path) -> dict:
    result = {
        "path": str(path.relative_to(ROOT)),
        "file_sha256": file_digest(path),
    }
    if path.suffix == ".json" and (self_sha256 := load(path).get("sha256")) is not None:
        result["self_sha256"] = self_sha256
    return result


def seal(value: dict) -> dict:
    result = dict(value)
    result["sha256"] = object_digest(result)
    return result


def packet_path(arm: dict) -> Path:
    return PACKET_DIR / f"qwen38-skyrl-production-posttrain-{arm['id']}-v1.json"


def checkpoint_schedule(run: dict) -> dict:
    recipe = run["recipe"]
    steps = recipe["steps"]
    interval = recipe["checkpoint_interval"]
    saves = sorted(set(range(interval, steps + 1, interval)) | {steps})
    retained = saves[-recipe["keep_checkpoints"] :]
    return {
        "interval": interval,
        "expected_save_steps": saves,
        "expected_retained_steps_at_terminal": retained,
        "terminal_step": steps,
        "terminal_checkpoint": f"{run['output_root']}/checkpoints/global_step_{steps}",
        "native_latest_pointer": f"{run['output_root']}/checkpoints/latest_ckpt_global_step.txt",
        "seal_manifest": f"{run['output_root']}/checkpoint-seals-v1/step-{steps}.json",
    }


def dev_tasks() -> list[dict]:
    split = load(queue.SPLIT)
    task_set = load(queue.TASK_SET)
    facts = {(row["task_key"], row["task_version_id"]): row for row in task_set["tasks"]}
    rows = []
    for assignment in split["tasks"]:
        if assignment["split"] != "dev":
            continue
        key = (assignment["task_key"], assignment["task_version_id"])
        fact = facts[key]
        rows.append(
            {
                "task_key": fact["task_key"],
                "task_version_id": fact["task_version_id"],
                "env_key": fact["env_key"],
                "env_version": fact["env_version"],
                "environment_version_id": fact["environment_version_id"],
                "data_key": fact["data_key"],
                "data_version": fact["data_version"],
            }
        )
    if len(rows) != 20:
        raise ValueError("expected the frozen 20-task development split")
    return rows


def serving_template() -> dict:
    value = load(SERVING_TEMPLATE)
    registration = value["desired_registration"]["spec"]
    return {
        "source": source(SERVING_TEMPLATE),
        "spec_sha256": object_digest(registration),
        "runtime_image": registration["runtime"]["image"],
        "normalized_runtime_args": registration["runtime"]["args"],
        "precision": registration["model"]["precision"],
        "tensor_parallel_size": registration["model"]["tensorParallelSize"],
        "data_parallel_size": registration["model"]["dataParallelSize"],
        "priority_class": registration["placement"]["priorityClassName"],
        "initial_desired_state": registration["desiredState"],
    }


def terminal_gate(run: dict) -> dict:
    steps = run["recipe"]["steps"]
    train_rollouts = steps * run["recipe"]["groups"] * run["recipe"]["samples_per_prompt"]
    return {
        "state": "blocked_waiting_exact_training_acceptance",
        "required_terminal_marker": f"{run['output_root']}/NATIVE_TRAINING_COMPLETE.json",
        "forbidden_markers": [
            f"{run['output_root']}/FAILED.json",
            f"{run['output_root']}/REJECTED.json",
            f"{run['output_root']}/NATIVE_FAILURE.json",
            f"{run['output_root']}/NATIVE_REJECTED.json",
        ],
        "minimum_real_train_rollouts": train_rollouts,
        "requirements": [
            "all_planned_train_and_development_batches_have_digest_valid_COLLECTED_receipts",
            "every_counted_train_rollout_used_the_exact_Fleet_task_and_environment_version",
            "every_counted_train_rollout_has_an_authoritative_verifier_execution_id",
            "no_counted_train_rollout_is_synthetic_retried_truncated_or_context_full",
            "reward_max_is_strictly_greater_than_reward_min_over_counted_train_rollouts",
            "every_planned_optimizer_update_has_finite_loss_KL_entropy_gradient_and_parameters",
            "at_least_one_optimizer_update_has_a_finite_nonzero_parameter_delta",
            "terminal_checkpoint_and_sampler_state_are_complete_and_reloadable",
            "training_source_model_and_private_data_are_unchanged",
            "all_owned_training_GPU_resources_are_released",
        ],
        "expected_optimizer_updates": steps,
        "expected_native_train_batches": steps,
        "accepted": False,
    }


def export_and_reload(run: dict, schedule: dict) -> tuple[dict, dict]:
    step = schedule["terminal_step"]
    export_root = f"{run['output_root']}/hf-export-step{step}-v1"
    export = {
        "state": "blocked_waiting_training_acceptance",
        "launchable": False,
        "tooling_ready": True,
        "adapter": source(POSTTRAIN_ADAPTER),
        "checkpoint_manifest_schema": "cyber_native_skyrl_rl_checkpoint_manifest_v1",
        "export_receipt_schema": "cyber_native_skyrl_rl_hf_export_v1",
        "source_checkpoint": schedule["terminal_checkpoint"],
        "source_manifest": schedule["seal_manifest"],
        "output_root": export_root,
        "receipt_path": f"{export_root}/EXPORT.json",
        "create_once": True,
        "image": IMAGE,
        "gpus": 0,
        "optimizer_updates": 0,
        "requirements": [
            "complete_native_FSDP_checkpoint_inventory_rehashed_before_and_after_export",
            "exact_Qwen38_base_model_revision_and_tokenizer_restored",
            "all_trained_tensors_reassembled_and_written_as_BF16",
            "all_output_tensors_reopened_equal_to_the_export_input",
            "source_checkpoint_sizes_mtimes_and_digests_unchanged",
            "destination_and_partial_destination_absent_before_create",
        ],
    }
    reload = {
        "state": "blocked_waiting_digest_valid_BF16_export",
        "launchable": False,
        "tooling_ready": True,
        "image": RELOAD_IMAGE,
        "launcher": {
            "adapter": source(RELOAD_ADAPTER),
            "spec_schema": "cyber_skyrl_prod9_reload_spec_v1",
            "preview_schema": "cyber_skyrl_prod9_reload_preview_v1",
            "authorization_schema": "cyber_skyrl_prod9_reload_authorization_v1",
            "created_schema": "cyber_skyrl_prod9_reload_created_v1",
            "create_once": True,
            "retry_on_any_result": False,
            "exact_uid_cleanup_profile": "production-reload",
        },
        "cpu_check": {
            "gpus": 0,
            "output": f"{run['output_root']}/export-check-step{step}-cpu-v1.json",
            "requires": [
                "complete_payload_rehash",
                "exact_tensor_key_shape_dtype_parity",
                "configuration_and_tokenizer_reload",
                "zero_optimizer_updates",
                "source_unchanged",
            ],
        },
        "one_gpu_check": {
            "name": f"{run['name']}-p{step}-reload-v1",
            "run_dir": f"{run['output_root']}-p{step}-reload-v1",
            "priority_class": "c1",
            "workers": 1,
            "gpus_per_worker": 1,
            "requeue_if_preempted": False,
            "receipt": f"{run['output_root']}-p{step}-reload-v1/GPU_CHECK.json",
            "requires": [
                "zero_optimizer_updates",
                "complete_BF16_model_and_tokenizer_reload",
                "finite_logits",
                "finite_short_generation",
                "source_unchanged",
                "UID_bound_terminal_success_and_GPU_release",
            ],
        },
    }
    return export, reload


def serving(run: dict, schedule: dict, template: dict) -> dict:
    step = schedule["terminal_step"]
    model_id = f"{run['name']}-step{step}-v1"
    return {
        "state": "blocked_waiting_zero_update_export_and_reload_acceptance",
        "launchable": False,
        "model_id": model_id,
        "staged_root": f"/models/{model_id}",
        "acceptance_marker": f"/models/{model_id}/.fleet-acceptance.json",
        "create_once": True,
        "registration": {
            "must_be_absent_before_create": True,
            "must_receive_a_new_server_UID": True,
            "must_not_resume_or_replace_an_existing_registration": True,
            "initial_desired_state": "paused",
            "routing_must_not_serve_traffic_before_live_parity": True,
            "priority_class": "c1",
        },
        "fixed_template": template,
        "required_execution_contract_sha256": EXECUTION_CONTRACT,
        "permitted_template_changes": [
            "model_id_and_display_name",
            "source_and_runtime_model_paths",
            "served_model_name",
            "revision_bound_to_the_exact_export_payload_manifest",
        ],
        "forbidden_template_changes": [
            "tokenizer",
            "chat_template",
            "precision",
            "quantization",
            "serving_image",
            "normalized_runtime_arguments_other_than_model_identity_paths",
            "context_limit",
            "OpenCode_adapter_or_tool_transport",
        ],
        "temporary_dev_qualification": {
            "maximum_runtime_seconds": 1800,
            "persistent_endpoint_allowed": False,
            "release_and_exact_UID_absence_required_on_success_or_failure": True,
        },
        "live_parity": {
            "base_model_id": "qwen3.8-27b",
            "candidate_model_id": model_id,
            "only_permitted_difference": "exact_weight_manifest",
            "must_pass_before_candidate_route_activation": True,
        },
    }


def evaluations(run: dict, schedule: dict, model_id: str, development: list[dict]) -> dict:
    step = schedule["terminal_step"]
    fixed_harness = {
        "name": "opencode",
        "version": "1.18.27",
        "context_tokens": 262144,
        "native_compaction": True,
        "compaction_reserved_tokens": 32768,
        "sampling": {"temperature": 0.6, "top_p": 0.95, "max_output_tokens": 8192},
    }
    return {
        "state": "blocked_waiting_paused_registration_and_live_parity",
        "launchable": False,
        "checkpoint_selection": {
            "surface": "Fleet_development_split_only",
            "selected_step": step,
            "webexploitbench_may_select_or_change_the_checkpoint": False,
        },
        "fleet_development": {
            "schema": "cyber_qwen38_fleet_dev_matched_plan_v1",
            "split_sha256": load(queue.SPLIT)["sha256"],
            "task_set_sha256": load(queue.TASK_SET)["sha256"],
            "tasks": development,
            "task_count": 20,
            "pass_k": 1,
            "attempts_per_arm": 20,
            "arms": {"base": "qwen3.8-27b", "candidate": model_id},
            "harness": fixed_harness,
            "must_reuse_exact_environment_verifier_prompt_tool_and_budget_bindings": True,
            "candidate_only_difference": "exact_weight_manifest_and_model_route_identity",
            "final_test_tasks": "closed_and_forbidden",
        },
        "webexploitbench": {
            "schema": "webexploitbench_qwen38_matched_plan_intent_v1",
            "role": "sealed_external_reporting_only",
            "mode": "full",
            "provider": "tensorlake_sandbox",
            "task_count": 15,
            "pass_k": 4,
            "attempts_per_arm": 60,
            "joint_attempts": 120,
            "arms": {"base": "qwen3.8-27b", "candidate": model_id},
            "harness": fixed_harness,
            "collection_and_scoring_are_separate": True,
            "defer_scoring": True,
            "paired_schedule": "alternate_base_first_and_candidate_first_by_task_index",
            "requires": [
                "exact_accepted_qualified_benchmark_snapshot",
                "fresh_create_once_base_and_candidate_campaign_IDs_and_output_roots",
                "sealed_single_arm_plans_and_webexploitbench_tensorlake_checkpoint_pair_v2",
                "identical_benchmark_harness_tools_sampling_budgets_timeouts_retries_and_scoring",
                "candidate_post_checkpoint_gate_bound_to_export_reload_registration_and_parity",
                "infrastructure_invalid_attempts_reported_separately_from_valid_zero_scores",
            ],
            "may_not_influence_training_recipe_or_checkpoint_selection": True,
        },
    }


def build_packet(arm: dict, development: list[dict], template: dict) -> dict:
    run_path = queue.path_for("run", arm)
    run = load(run_path)
    schedule = checkpoint_schedule(run)
    export, reload = export_and_reload(run, schedule)
    serve = serving(run, schedule, template)
    extra_gate = (
        ["a1_step10_accepted_and_Fleet_development_stop_rule_passed"]
        if arm["id"] == "dose50"
        else []
    )
    return seal(
        {
            "schema": "cyber_qwen38_skyrl_posttrain_packet_v1",
            "state": "offline_blocked_waiting_training_acceptance",
            "launchable": False,
            "purpose": (
                "Fail-closed export, reload, paused serving and matched evaluation path for one "
                "predeclared SkyRL arm."
            ),
            "arm": {
                "id": arm["id"],
                "name": arm["name"],
                "treatment": arm["treatment"],
                "run_config": source(run_path),
                "output_root": run["output_root"],
                "recipe": run["recipe"],
                "additional_eligibility_gates": extra_gate,
            },
            "model": {
                "repo": "Qwen/Qwen3.8-27B",
                "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
                "lock": source(MODEL_LOCK),
                "weights": source(MODEL_WEIGHTS),
            },
            "training_acceptance": terminal_gate(run),
            "checkpoint": schedule,
            "export": export,
            "reload": reload,
            "serving": serve,
            "evaluation": evaluations(run, schedule, serve["model_id"], development),
            "firewall": {
                "final_test_tasks": "closed",
                "final_test_task_count": 10,
                "final_test_identity_or_outcomes_in_this_packet": False,
                "external_benchmark_content_or_outcomes_read": 0,
                "external_benchmark_may_change_training_or_selection": False,
            },
            "execution_record": {
                "offline_packet_only": True,
                "external_mutations": 0,
                "jobs_submitted": 0,
                "files_staged": 0,
                "routes_created_or_changed": 0,
                "evaluations_launched": 0,
            },
        }
    )


def build() -> dict[Path, dict]:
    development = dev_tasks()
    template = serving_template()
    packets = {packet_path(arm): build_packet(arm, development, template) for arm in queue.ARMS}
    evidence = seal(
        {
            "schema": "cyber_qwen38_skyrl_posttrain_queue_evidence_v1",
            "scope": "offline_no_submit_no_stage_no_serve_no_eval",
            "arms": [
                {
                    "id": arm["id"],
                    "name": arm["name"],
                    "packet": str(packet_path(arm).relative_to(ROOT)),
                    "packet_file_sha256": "sha256:"
                    + hashlib.sha256(raw(packets[packet_path(arm)])).hexdigest(),
                    "packet_self_sha256": packets[packet_path(arm)]["sha256"],
                    "terminal_checkpoint": packets[packet_path(arm)]["checkpoint"][
                        "terminal_checkpoint"
                    ],
                    "launchable": False,
                }
                for arm in queue.ARMS
            ],
            "common_blockers": [
                "prod4_canary_terminal_acceptance_and_release_not_yet_bound",
                "five_production_training_arms_not_yet_terminally_accepted",
                "exact_export_reload_staging_registration_and_live_parity_receipts_absent",
                "fresh_matched_Fleet_dev_and_WebExploitBench_plans_not_sealable_before_routes_exist",
            ],
            "ready_now": [
                "RL_specific_plan_bound_checkpoint_seal_and_BF16_export_adapter_implemented",
                "five_unique_output_and_checkpoint_schedules_bound",
                "terminal_reward_update_and_reload_acceptance_contract_bound",
                "create_once_zero_update_export_and_reload_contract_bound",
                "create_once_paused_registration_new_UID_and_live_parity_contract_bound",
                "matched_OpenCode_Fleet_dev_and_WebExploitBench_protocol_intents_bound",
                "final_test_firewall_bound",
            ],
            "external_mutations": 0,
        }
    )
    packets[EVIDENCE] = evidence
    return packets


def raw(value: dict) -> bytes:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--write", action="store_true")
    args = parser.parse_args()
    artifacts = build()
    if args.check:
        stale = [
            str(path.relative_to(ROOT))
            for path, value in artifacts.items()
            if not path.exists() or path.read_bytes() != raw(value)
        ]
        if stale:
            raise SystemExit("stale posttrain packets: " + ", ".join(stale))
    else:
        for path, value in artifacts.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw(value))
    print(json.dumps({"artifacts": len(artifacts), "external_mutations": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
