"""Offline invariants for the inert Qwen3.8 long-context Miles full arm."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from cyber_post_train.jobs import digest
from training import miles, miles_training, rl_data

ROOT = Path(__file__).resolve().parents[1]
TASK_SET = ROOT / "configs/data/qwen38-rl-filtered-study-a-task-set-v1.json"
SPLIT = ROOT / "configs/data/qwen38-rl-filtered-study-a-split-v1.json"
DATA = ROOT / "configs/runs/qwen38-miles-opencode-long-context-full-v1.data.json"
ARM = ROOT / "configs/runs/qwen38-miles-opencode-long-context-full-v1.template.json"


def load(path: Path) -> dict:
    return json.loads(path.read_bytes())


def file_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def test_full_arm_preserves_59_train_20_dev_long_context_science() -> None:
    task_set, split, data, arm = map(load, (TASK_SET, SPLIT, DATA, ARM))
    selected = rl_data.selection(task_set, split)
    run = arm["candidate_run"]

    assert [row["split"] for row in selected].count("train") == 59
    assert [row["split"] for row in selected].count("dev") == 20
    assert task_set["boundaries"]["webexploitbench_rows_included"] == 0
    assert data["harness"] == {
        "name": "opencode",
        "version": "1.18.27",
        "context_management": "opencode_1.18.27_native_compaction_autocontinue_segmented_v1",
        "context_window_size": 262144,
        "max_output_tokens": 32768,
        "max_model_requests": 2048,
        "job_hard_seconds": 32400,
        "compaction_buffer_tokens": 20000,
        "compaction_reserved_tokens": 52768,
        "compaction_threshold_tokens": 176608,
        "summary_max_tokens": 4096,
        "preserve_recent_tokens": 15000,
        "summary_token_treatment": "excluded_separate_session_v1",
        "session_node_cap": 4096,
        "binary_sha256": (
            "sha256:bddf894e5c2bc3d8cf452bd6e5ab2273bbe4a37eeeb9aec848d3d7d20db1f256"
        ),
    }
    assert data["limits"] == {
        "context_tokens": 262144,
        "response_tokens": 245760,
        "max_tokens_per_turn": 32768,
        "max_turns": 2048,
        "episode_seconds": 28800,
        "tool_seconds": 330,
    }
    assert run["recipe"] == {
        "nodes": 4,
        "gpus_per_node": 8,
        "steps": 59,
        "groups": 1,
        "samples_per_prompt": 8,
        "lr": 2e-6,
        "temperature": 0.7,
        "kl_loss_coef": 0.001,
        "max_tokens_per_gpu": 65536,
        "eval_interval": 59,
        "checkpoint_interval": 10,
        "seed": 42,
        "native_profile": "qwen3.8-27b-256k",
        "harness": "opencode",
        "session_node_cap": 4096,
    }
    miles.MilesConfig(
        name=run["name"],
        output_root=run["output_root"],
        model_root=run["model"]["root"],
        torch_dist_root="/mnt/sfs/jobs/chris-q38-miles-lc-base1/torch-dist",
        train_data=data["output"] + "/train.jsonl",
        dev_data=data["output"] + "/dev.jsonl",
        data_manifest=run["data"]["manifest"],
        wandb_entity=run["wandb"]["entity"],
        wandb_project=run["wandb"]["project"],
        wandb_run_id=run["wandb"]["run_id"],
        context_tokens=data["limits"]["context_tokens"],
        response_tokens=data["limits"]["response_tokens"],
        tokens_per_turn=data["limits"]["max_tokens_per_turn"],
        runtime_image=run["runtime"]["image"],
        **run["recipe"],
    ).validate()


def test_full_arm_is_digest_bound_but_cannot_launch_before_canary() -> None:
    arm = load(ARM)
    assert arm["status"] == "blocked_not_launchable"
    assert arm["launchable"] is False
    assert arm["candidate_run"]["production_promotion"] is None
    assert arm["sha256"] == "sha256:" + digest(
        {key: value for key, value in arm.items() if key != "sha256"}
    )
    assert arm["candidate_run_sha256"] == "sha256:" + digest(
        {
            key: value
            for key, value in arm["candidate_run"].items()
            if key != "production_promotion"
        }
    )
    for source in arm["source_files"].values():
        assert source["file_sha256"] == file_sha256(ROOT / source["path"])
    with pytest.raises(ValueError):
        miles_training.compile_rl(arm["candidate_run"], relative_to=ARM.parent)


def test_full_arm_binds_exact_queued_canary_and_post_training_handoffs() -> None:
    arm = load(ARM)
    canary = arm["qualification"]["canary_submission"]
    assert canary == {
        "api_run_name": "chris-q38-miles-lc-canary1-2eb152dd",
        "api_run_id": "2eb152dd-8cd7-4802-8d26-788f2361346b",
        "rayjob_uid": "ad03cb9e-755f-491b-a997-28e75db97a56",
        "workload_uid": "f789db64-9158-46bd-acbc-5d99df88d4a0",
        "runtime_bundle_sha256": (
            "sha256:45ece11b1a0d3f138b43a8dd291d1c06744e969fbc86dafeee7baa2b0ddd1e9b"
        ),
        "priority": "c1",
        "queue_priority": 10000,
        "nodes": 4,
        "gpus_per_node": 8,
    }
    run = arm["candidate_run"]
    assert run["cluster"]["priority"] == "c1"
    assert '"c0"' not in json.dumps(arm).lower()
    assert '"q0"' not in json.dumps(arm).lower()
    assert "webexploitbench" not in json.dumps(run).lower()
    handoff = arm["post_training_handoff"]
    assert handoff["native_reload_template"].endswith(
        "qwen38-miles-opencode-long-context-reload-prod-v1.template.json"
    )
    wbe = ROOT / handoff["webexploitbench_paired_plan_template"]
    assert handoff["webexploitbench_paired_plan_template_file_sha256"] == file_sha256(wbe)
    assert handoff["webexploitbench_outcomes_are_training_inputs"] is False
