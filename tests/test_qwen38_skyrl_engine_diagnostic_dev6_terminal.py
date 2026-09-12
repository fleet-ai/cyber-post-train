"""Terminal falsification and zero-work gates for the exact SkyRL dev6 run."""

import json
from pathlib import Path

from cyber_post_train.jobs import digest

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / (
    "docs/evidence/qwen38-study/"
    "2026-09-12-skyrl-engine-diagnostic-dev6-terminal-v1.json"
)


def load():
    return json.loads(EVIDENCE.read_bytes())


def test_dev6_terminal_evidence_is_sealed_and_exact():
    value = load()
    assert value["sha256"] == digest({k: v for k, v in value.items() if k != "sha256"})
    assert value["classification"] == "clean_diagnostic_rejection_hypothesis_falsified"
    assert value["run"]["api_run_id"] == "f4bce6ba-d539-41f0-b4da-f12c3c519391"
    assert value["run"]["rayjob_uid"] == "5539fa91-95c1-4895-abf1-fc88f4ddf10f"
    assert value["run"]["workload_uid"] == "345c815a-00db-4c13-a9ca-fff697ff70cb"
    assert value["run"]["api_status"] == value["run"]["controller_status"] == "SUCCEEDED"
    assert value["run"]["effective_priority"] == 10000
    assert value["run"]["automatic_requeue"] is False


def test_dev6_falsified_the_in_process_hypothesis():
    value = load()
    terminal = value["receipts"]["terminal"]
    assert terminal["digest_valid"] is True
    assert terminal["status"] == "engine_start_rejected"
    assert terminal["vllm_v1_multiprocessing_disabled"] is True
    assert terminal["engine_start_qualified"] is terminal["training_qualified"] is False
    assert value["sanitized_cause"]["same_generic_branch_as_dev5"] is True
    assert value["sanitized_cause"]["underlying_child_exception_captured"] is False
    hypothesis = value["diagnostic_hypothesis"]
    assert hypothesis["result"] == "falsified"
    assert hypothesis["installed_source"]["async_constructor_call"] == (
        "EngineCoreClient.make_async_mp_client"
    )
    assert hypothesis["installed_source"]["private_log_read"] is False
    assert value["qualification"]["automatic_gpu_successor_forbidden"] is True


def test_dev6_did_zero_training_work_and_released_everything():
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
        "workload_finished_at": "2026-09-12T13:19:58Z",
        "raycluster_absent": True,
        "pods_absent": True,
        "gpus_held": 0,
    }
    assert value["audit_boundaries"]["private_logs_read"] is False

