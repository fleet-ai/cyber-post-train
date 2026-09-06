"""Held, score-blind transition plan from rank99 TP1 to Qwen DP8 serving."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evals.fleet import qwen38_dedicated_dp8_v1 as prior
from evals.fleet import self_hosted

PLAN_PATH = Path(
    "docs/evidence/qwen38-study/"
    "2026-09-06-qwen38-dp8-post-rank99-held-plan-v1.json"
)
SCHEMA = "fleet-qwen38-dedicated-dp8-post-rank99-plan-v1"
TITLE = "chris-cyber-evalserve-q38-dp8-b-v1"
RUN_DIR = "/mnt/sfs/jobs/chris-cyber-evalserve-q38-dp8-b-v1"
SERVING_BLOCK = "dedicated-qwen-dp8-b-v1"


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("held DP8 plan must be an object")
    return value


def validate(value: dict[str, Any]) -> None:
    if value.get("receipt_sha256") != self_hosted.digest_without(value, "receipt_sha256"):
        raise ValueError("held DP8 plan digest drifted")
    if set(value) != {
        "schema_version",
        "status",
        "launch_authorized",
        "scoring_authorized",
        "purpose",
        "rank99_transition_gate",
        "node_budget_gate",
        "server",
        "non_scored_qualification",
        "scored_canary_gate",
        "bulk_gate",
        "lifecycle",
        "throughput_estimate",
        "privacy",
        "receipt_sha256",
    }:
        raise ValueError("held DP8 plan fields drifted")
    if (
        value.get("schema_version") != SCHEMA
        or value.get("status") != "HELD_NO_LAUNCH"
        or value.get("launch_authorized") is not False
        or value.get("scoring_authorized") is not False
        or value.get("purpose")
        != "post_rank99_qwen_data_parallel_throughput_qualification"
    ):
        raise ValueError("held DP8 plan authorization drifted")

    if value.get("rank99_transition_gate") != {
        "current_serving_block": "dedicated-qwen-tp1-j-v1",
        "controller_job": "chris-cyber-q38-r099-g23-package-reservation-v1",
        "all_four_rank99_cells_formally_accepted_required": True,
        "controller_terminal_success_required": True,
        "tp1_jobs_api_delete_204_then_get_404_required": True,
        "tp1_rayjob_workload_pod_service_absence_required": True,
        "tp1_gpu_release_required": True,
        "no_overlap_with_dp8_server": True,
    }:
        raise ValueError("rank99 transition gate drifted")
    if value.get("node_budget_gate") != {
        "max_project_gpu_nodes": 2,
        "max_project_gpus": 16,
        "fresh_jobs_api_and_kubernetes_inventory_required": True,
        "maximum_other_gpu_nodes_before_create": 1,
        "maximum_other_gpus_before_create": 8,
        "new_server_gpu_nodes": 1,
        "new_server_gpus": 8,
    }:
        raise ValueError("DP8 node budget gate drifted")

    server = value.get("server")
    if not isinstance(server, dict) or server != {
        "title": TITLE,
        "run_dir": RUN_DIR,
        "serving_block": SERVING_BLOCK,
        "create_once": True,
        "api_route": "POST /v1/runs",
        "image": prior.IMAGE,
        "model_repository": "Qwen/Qwen3.8-27B",
        "model_revision": prior.MODEL_REVISION,
        "served_id": "qwen3.8-27b",
        "context_length": 262144,
        "tensor_parallel_size": 1,
        "data_parallel_size": 8,
        "load_balance_method": "total_tokens",
        "server_arguments_sha256": prior.SERVER_ARGUMENTS_SHA256,
        "workers": 1,
        "gpus_per_worker": 8,
        "priority_class": "fleet-infra-quiet",
        "preemption_policy": "Never",
        "fresh_immutable_binding_receipt_required": True,
    }:
        raise ValueError("DP8 server identity or runtime drifted")

    if value.get("non_scored_qualification") != {
        "actual_opencode_1_18_27_parity_required": True,
        "exact_model_revision_and_context_required": True,
        "exact_bash_then_submit_report_tool_contract_required": True,
        "task_instance_session_verifier_scoring_calls": 0,
        "concurrency_ladder": [1, 2, 4, 8],
        "strict_tool_name_and_argument_validation": True,
        "zero_http_model_or_tool_protocol_errors_required": True,
        "persist_latency_error_and_throughput_receipt": True,
        "stop_at_first_failed_level": True,
        "maximum_scored_concurrency_after_pass": 8,
    }:
        raise ValueError("DP8 non-scored qualification drifted")
    if value.get("scored_canary_gate") != {
        "required_after_non_scored_qualification": True,
        "exactly_one_globally_unstarted_cell": True,
        "fresh_execution_generation_and_claim_namespace": True,
        "canonical_claim_before_model_call": True,
        "fresh_session_accepted_active_blocked_collision_scan": True,
        "separate_serving_block": SERVING_BLOCK,
        "formal_acceptance_required_before_bulk": True,
        "stop_on_failure": True,
    }:
        raise ValueError("DP8 scored canary gate drifted")
    if value.get("bulk_gate") != {
        "launch_authorized": False,
        "whole_task_boundaries_only": True,
        "all_four_attempt_claims_reserved_atomically_before_model": True,
        "rollback_all_new_claims_if_partial_reservation_fails_before_model": True,
        "same_serving_block_for_all_four_attempts": True,
        "hosted_and_dedicated_results_remain_explicit_blocks": True,
        "maximum_parallel_rollouts": 8,
        "recommended_parallel_shape_after_canary": "two_tasks_x_four_attempts",
        "fresh_launch_time_global_reconciliation_required": True,
    }:
        raise ValueError("DP8 bulk gate drifted")
    if value.get("lifecycle") != {
        "model_loading_counts_as_productive": True,
        "post_ready_idle_seconds": 600,
        "real_traffic_only_refreshes_idle_deadline": True,
        "release_immediately_on_failure_or_idle": True,
    }:
        raise ValueError("DP8 lifecycle drifted")
    if value.get("throughput_estimate") != {
        "tp1_current_parallel_model_streams": 1,
        "dp8_theoretical_parallel_model_streams": 8,
        "expected_concurrency_multiplier_ceiling": 8,
        "per_rollout_latency_speedup_claimed": False,
        "accepted_rollouts_per_minute_requires_measurement": True,
        "qualified_concurrency_is_highest_passing_ladder_level": True,
    }:
        raise ValueError("DP8 throughput estimate drifted")
    if value.get("privacy") != {
        "prompts_traces_flags_or_scores_included": False,
        "credentials_included": False,
    }:
        raise ValueError("DP8 plan privacy drifted")


def load_held(root: Path) -> dict[str, Any]:
    value = load(root / PLAN_PATH)
    validate(value)
    return value


if __name__ == "__main__":
    validate(load(PLAN_PATH))
