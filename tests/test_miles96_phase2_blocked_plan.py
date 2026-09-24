from __future__ import annotations

import copy
import json

import pytest

from training import miles96_mechanics_canary as mechanics
from training import miles96_phase2_blocked_plan as blocked


def _write(tmp_path, value: dict):
    body = {key: item for key, item in value.items() if key != "sha256"}
    value["sha256"] = "sha256:" + mechanics.digest(body)
    path = tmp_path / "phase2.json"
    path.write_text(json.dumps(value))
    return path


def test_phase2_plan_is_exact_launch_blocked_and_unrendered() -> None:
    value = blocked.load()
    assert value["state"] == "blocked_pending_four_task_signal_and_v004_adapter"
    assert value["launchable"] is False
    assert value["future_identity"]["phase1_signal_evidence_sha256s"] is None
    assert value["future_identity"]["phase2_plan_sha256"] is None
    assert value["future_identity"]["train_request_sha256"] is None
    assert value["future_identity"]["reload_request_sha256"] is None
    assert value["known_adapter_gap"]["adapter_extension_required"] is True
    assert len(value["unresolved_launch_gates"]) == 14


def test_phase2_candidates_are_exact_train_bindings_and_score_blind() -> None:
    value = blocked.load()
    rows = value["candidate_bindings"]
    assert [row["name"] for row in rows] == sorted(row["name"] for row in rows)
    assert len(rows) == 4
    assert all(row["split"] == "train" for row in rows)
    forbidden = {
        "selection_prior",
        "campaign_rank",
        "passes",
        "reward_values",
        "completed_accepted_episodes",
    }
    assert all(not forbidden.intersection(row) for row in rows)
    prerequisite = value["phase1_prerequisite"]
    assert prerequisite["required_terminal_phase1_lanes"] == 4
    assert prerequisite["required_qualified_lanes"] == 4
    assert prerequisite["historical_scores_used"] is False
    assert prerequisite["historical_reward_magnitudes_used"] is False
    assert prerequisite["fallback_or_task_substitution"] is False


def test_phase2_uses_exact_v004_proven_mechanics() -> None:
    value = blocked.load()
    reference = value["v004_reference"]
    assert reference["commit"] == "10afa8d064bb3dd1c11c50768590e432dfa69097"
    proven = reference["proven_recipe_shape"]
    train = value["train_contract"]
    expected = {
        "nodes": 1,
        "gpus_per_node": 8,
        "tensor_parallel": 4,
        "context_parallel": 2,
        "max_train_tokens_per_gpu": 8192,
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
    }
    for key, item in expected.items():
        assert proven[key] == item
        assert train[key] == item
    assert proven["effective_context_tokens"] == train["context_tokens"] == 98_304
    assert train["fresh_rollout_count"] == 32
    assert train["optimizer_steps"] == train["rollout_iterations"] == 1
    assert train["force_terminal_checkpoint_after_one_update"] is True
    assert train["root_annotations"] == {"fleet.ai/failure-alerts": "off"}
    assert train["priority_class"] == "c1" and train["queue_priority"] == "q1"


def test_phase2_schedule_is_two_groups_per_task_and_four_samples_per_group() -> None:
    value = blocked.load()
    candidates = [row["name"] for row in value["candidate_bindings"]]
    schedule = value["phase2_prompt_schedule"]
    assert len(schedule) == 8
    assert [row["prompt_group_index"] for row in schedule] == list(range(8))
    assert [row["candidate_name"] for row in schedule[:4]] == candidates
    assert [row["candidate_name"] for row in schedule[4:]] == candidates
    assert [row["replicate_index"] for row in schedule] == [0] * 4 + [1] * 4
    assert all(row["samples"] == 4 for row in schedule)
    gate = value["optimizer_gate"]
    assert gate["prompt_group_count"] == 8
    assert gate["episodes_per_group"] == 4
    assert gate["selected_episode_count"] == 32
    assert gate["every_group_minimum_distinct_reward_values"] == 2
    assert gate["unique_task_instance_ids"] == 32
    assert gate["unique_verifier_execution_ids"] == 32
    assert gate["update_if_any_group_gate_fails"] is False


def test_phase2_checkpoint_and_reload_are_fail_closed() -> None:
    value = blocked.load()
    checkpoint = value["checkpoint_export_acceptance"]
    assert checkpoint["optimizer_updates"] == 1
    assert checkpoint["periodic_checkpoint_interval_remains_v004_value_four"] is True
    assert checkpoint["forced_terminal_checkpoint_after_bounded_update"] is True
    assert checkpoint["minimum_changed_trained_tensors"] == 1
    reload = value["reload_contract"]
    assert reload["starts_only_after_train_gpu_release"] is True
    assert reload["nodes"] == reload["gpus_per_node"] == 1
    assert reload["optimizer_steps"] == 0
    assert reload["root_annotations"] == {"fleet.ai/failure-alerts": "off"}


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.__setitem__("launchable", True),
        lambda value: value["candidate_bindings"].reverse(),
        lambda value: value["phase1_prerequisite"].__setitem__("required_qualified_lanes", 3),
        lambda value: value["train_contract"].__setitem__("optimizer_steps", 2),
        lambda value: value["train_contract"].__setitem__("max_train_tokens_per_gpu", 49_152),
        lambda value: value["train_contract"].__setitem__("optimizer_cpu_offload", False),
        lambda value: value["optimizer_gate"].__setitem__("update_if_any_group_gate_fails", True),
        lambda value: value["checkpoint_export_acceptance"].__setitem__(
            "minimum_changed_trained_tensors", 0
        ),
        lambda value: value["reload_contract"].__setitem__("optimizer_steps", 1),
        lambda value: value["unresolved_launch_gates"].pop(),
    ],
)
def test_phase2_plan_rejects_semantic_drift(tmp_path, mutate) -> None:
    value = copy.deepcopy(blocked.load())
    mutate(value)
    with pytest.raises(ValueError):
        blocked.load(_write(tmp_path, value))
