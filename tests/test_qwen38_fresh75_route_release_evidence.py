"""Sanitized, self-sealed proof that the drained Fresh75 route released GPUs."""

import json
from pathlib import Path

from cyber_post_train.jobs import digest

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/qwen38-fresh75-step230-route-release-20260921.json"


def test_fresh75_release_evidence_is_self_sealed_and_gpu_free():
    value = json.loads(EVIDENCE.read_text())
    assert value["sha256"] == digest({key: item for key, item in value.items() if key != "sha256"})
    assert value["consumer_gate"]["active_or_queued_owned_consumers"] == 0
    assert value["pause"] == {
        **value["pause"],
        "post_attempts": 1,
        "terminal_api_phase": "paused",
        "terminal_active_pods": 0,
        "terminal_ready_replicas": 0,
        "exact_serving_pod_absent": True,
        "deployment_replicas": 0,
        "matching_gpu_pods": 0,
        "matching_gpu_allocation": 0,
    }
    assert value["scope"] == {
        "fresh75_route_only": True,
        "self_route_mutated": False,
        "teacher_route_mutated": False,
        "evaluation_cells_mutated": False,
        "jobs_created_or_deleted": 0,
    }
    assert value["privacy"] == {
        "score_values_included": False,
        "prompts_responses_flags_rewards_or_traces_included": False,
        "credentials_or_private_logs_included": False,
        "final_eight_task_set_accessed": False,
    }
