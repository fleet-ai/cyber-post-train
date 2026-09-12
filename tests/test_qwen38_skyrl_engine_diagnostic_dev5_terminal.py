"""Terminal zero-work and cleanup gates for the exact SkyRL dev5 run."""

import json
from pathlib import Path

from cyber_post_train.jobs import digest

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / (
    "docs/evidence/qwen38-study/"
    "2026-09-12-skyrl-engine-diagnostic-dev5-terminal-v1.json"
)


def load():
    return json.loads(EVIDENCE.read_bytes())


def test_dev5_terminal_evidence_is_sealed_and_exact():
    value = load()
    assert value["sha256"] == digest({k: v for k, v in value.items() if k != "sha256"})
    assert value["classification"] == "clean_diagnostic_rejection_engine_core_child_exit"
    assert value["run"]["api_run_id"] == "7a6fd5d6-626b-47b1-9ff2-d53a9437bd88"
    assert value["run"]["rayjob_uid"] == "aa61662e-435f-489b-92ad-4114e155dea5"
    assert value["run"]["workload_uid"] == "b554dc90-c48a-4e90-8e9b-46a159c917d6"
    assert value["run"]["api_status"] == value["run"]["controller_status"] == "SUCCEEDED"
    assert value["run"]["effective_priority"] == 10000
    assert value["run"]["automatic_requeue"] is False


def test_dev5_is_a_rejection_not_rl_qualification():
    value = load()
    terminal = value["receipts"]["terminal"]
    assert terminal["digest_valid"] is True
    assert terminal["status"] == "engine_start_rejected"
    assert terminal["engine_start_qualified"] is False
    assert terminal["training_qualified"] is False
    assert value["sanitized_cause"]["proven_branch"] == (
        "vllm_engine_core_child_process_exited_before_ready_handshake"
    )
    assert value["sanitized_cause"]["underlying_child_exception_captured"] is False
    assert all(
        state == "closed"
        for key, state in value["qualification"].items()
        if key not in {"engine_start", "replay_forbidden"}
    )
    assert value["qualification"]["replay_forbidden"] is True


def test_dev5_did_zero_training_work_and_released_everything():
    value = load()
    zero = value["zero_work"]
    assert not any(
        zero[key]
        for key in (
            "task_rows_read",
            "rollouts",
            "verifier_calls",
            "optimizer_steps",
            "checkpoints_created",
            "wandb_initialized",
            "checkpoint_artifacts",
            "episode_artifacts",
            "task_artifacts",
            "unexpected_output_artifacts",
        )
    )
    assert zero["runtime_files_unchanged"] is True
    cleanup = value["isolation_and_cleanup"]
    assert cleanup["cleanup_proven"] is True
    assert cleanup["active_owned_actors"] == cleanup["active_owned_placement_groups"] == 0
    assert value["release"] == {
        "workload_finished_at": "2026-09-12T12:53:23Z",
        "raycluster_absent": True,
        "pods_absent": True,
        "gpus_held": 0,
    }
    assert value["audit_boundaries"]["private_logs_read"] is False
