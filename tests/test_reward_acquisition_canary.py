from __future__ import annotations

import hashlib
import json
from pathlib import Path

from training.jobs_api import rl_paid_launch_blockers


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = (
    ROOT
    / "configs/runs/qwen36-27b-native-rl-reward-acquisition-canary.pre-submit.json"
)


def _plan() -> dict:
    return json.loads(PLAN_PATH.read_text())


def test_reward_acquisition_canary_is_exactly_shaped_and_explicitly_blocked() -> None:
    plan = _plan()
    request = plan["run_request_template"]
    source_path = ROOT / plan["source_evidence"]["two_task_gate"]
    source = json.loads(source_path.read_text())

    assert plan["status"] == "blocked_pre_submission"
    assert plan["paid_submission_authorized"] is False
    assert [row["source_task_version_id"] for row in plan["task_successors"]] == [
        "54425601-6fd2-43d8-8cb9-e565b767676a",
        "f31ebe83-0ff1-4660-bcba-59ffa4b82d5a",
    ]
    assert plan["source_evidence"]["two_task_gate_file_sha256"] == (
        f"sha256:{hashlib.sha256(source_path.read_bytes()).hexdigest()}"
    )
    source_by_id = {row["task_version_id"]: row for row in source["tasks"]}
    for successor in plan["task_successors"]:
        frozen = source_by_id[successor["source_task_version_id"]]
        assert successor["task_key"] == frozen["task_key"]
        assert successor["source_task_version"] == frozen["task_version"]
        for field in (
            "environment_version_id",
            "env_key",
            "env_version",
            "data_key",
            "data_version",
        ):
            assert successor[field] == frozen[field]
    assert all(
        row["status"] == "pending"
        and row["metadata_only_successor_task_version_id"] is None
        and row["metadata_only_successor_task_version"] is None
        for row in plan["task_successors"]
    )
    assert request["tasks"]["task_versions"] == []
    assert request["trainer"]["trainer_version_id"] is None

    assert request["num_workers"] == 1
    assert request["gpus_per_worker"] == 8
    assert request["model"] == {
        "staged_model": "qwen3.6-27b",
        "precision": "bf16",
        "engine_tensor_parallel_size": 4,
        "max_context_length": 65_536,
        "max_prompt_length": 16_384,
        "max_generate_length": 49_152,
        "sequence_parallel_size": 1,
    }
    assert request["grpo"] == {
        "group_size": 4,
        "train_batch_size": 2,
        "policy_mini_batch_size": 2,
        "kl_coefficient": 0.001,
        "kl_estimator": "k3",
        "policy_loss_type": "regular",
        "checkpoint_interval": 1,
        "max_checkpoints_to_keep": 1,
        "learning_rate": 0.000001,
        "max_steps": 1,
        "temperature": 1.0,
    }
    assert "trainer.max_training_steps=1" in request["trainer"]["args"]
    assert "generator.sampling_params.top_p=1.0" in request["trainer"]["args"]
    assert request["rollout"] == {
        "harness": "native",
        "mode": "tool-use",
        "max_turns": 600,
        "max_tokens_per_turn": 2048,
        "tool_result_max_chars": 16000,
        "required_task_tools": ["bash", "submit_report"],
        "partial_verifier_scoring": False,
        "pass_conversation_to_verifier": False,
        "multi_app_aggregation_mode": "binary",
    }
    assert request["eval"]["task_versions"] == []
    assert request["eval"]["before_train"] is False
    assert plan["derived_shape"] == {
        "accelerator": "B300",
        "workers": 1,
        "gpus_per_worker": 8,
        "total_gpus": 8,
        "engine_tensor_parallel_size": 4,
        "num_inference_engines": 2,
        "planned_rollouts": 8,
        "true_optimizer_steps": 1,
        "checkpoints_retained": 1,
        "evaluation_episodes": 0,
        "context_compaction": "disabled_single_flat_trajectory",
    }


def test_checked_in_canary_cannot_pass_the_current_paid_launch_gate() -> None:
    request = _plan()["run_request_template"]
    preview = {
        "manifest_yaml": (
            "spec:\n"
            "  entrypoint: python -m rl_rollout.entrypoint "
            "trainer.epochs=1 trainer.max_training_steps=1\n"
        )
    }

    blockers = rl_paid_launch_blockers(request, preview)

    assert blockers == [
        "paid RL requires at least one exact tasks.task_versions binding"
    ]


def test_canary_names_every_nonadministrative_launch_blocker() -> None:
    blocker_ids = [row["id"] for row in _plan()["launch_blockers"]]
    assert blocker_ids == [
        "trainer_catalog_readiness",
        "metadata_only_task_successors",
        "prompt_schema_token_preflight",
        "b300_65k_feasibility",
        "per_episode_verifier_identity",
        "single_tool_call_semantics",
        "explicit_paid_run_approval",
    ]
