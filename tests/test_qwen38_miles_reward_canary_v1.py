"""Static identity for the smallest real Qwen3.8 Miles reward/update canary."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "configs/qualification/qwen38-miles-rl-reward-canary-data-dev-v1.json"
RUN = ROOT / "configs/qualification/qwen38-miles-rl-reward-canary-dev-v1.json"
DATA_V2 = ROOT / "configs/qualification/qwen38-miles-rl-reward-canary-data-dev-v2.json"
RUN_V2 = ROOT / "configs/qualification/qwen38-miles-rl-reward-canary-dev-v2.json"
DATA_V3 = ROOT / "configs/qualification/qwen38-miles-rl-reward-canary-data-dev-v3.json"
RUN_V3 = ROOT / "configs/qualification/qwen38-miles-rl-reward-canary-dev-v3.json"
LAUNCH_V3 = ROOT / "configs/qualification/qwen38-miles-rl-reward-canary-launch-dev-v3.json"
DATA_V4 = ROOT / "configs/qualification/qwen38-miles-rl-reward-canary-data-dev-v4.json"
RUN_V4 = ROOT / "configs/qualification/qwen38-miles-rl-reward-canary-dev-v4.json"
LAUNCH_V4 = ROOT / "configs/qualification/qwen38-miles-rl-reward-canary-launch-dev-v4.json"
RUN_V5 = ROOT / "configs/qualification/qwen38-miles-rl-reward-canary-dev-v5.json"
LAUNCH_V5 = ROOT / "configs/qualification/qwen38-miles-rl-reward-canary-launch-dev-v5.json"
DEV3_RETIRED = (
    ROOT
    / "docs/evidence/qwen38-study/2026-09-13-miles-rlreward-dev3-retired-v1.json"
)
TASK_SET = ROOT / "configs/data/qwen38-rl-reward-canary-task-set-v1.json"
SPLIT = ROOT / "configs/data/qwen38-rl-reward-canary-split-v1.json"
TOOLS = ROOT / "configs/data/qwen38-rl-filtered-canary-tool-catalog-v1.json"


def load(path: Path) -> dict:
    return json.loads(path.read_bytes())


def test_miles_reward_canary_is_one_dev_update_on_the_exact_source_task() -> None:
    data, run, task_set, split = map(load, (DATA, RUN, TASK_SET, SPLIT))

    assert data["backend"] == run["backend"] == "miles"
    assert data["name"] == run["name"] == run["wandb"]["run_id"]
    assert data["name"] == "chris-q38-miles-rlreward-dev1"
    assert data["task_set"] == "../data/qwen38-rl-reward-canary-task-set-v1.json"
    assert data["split"] == "../data/qwen38-rl-reward-canary-split-v1.json"
    assert data["tool_catalog"] == "../data/qwen38-rl-filtered-canary-tool-catalog-v1.json"
    catalog = json.loads(TOOLS.read_bytes())
    canonical = json.dumps(catalog, sort_keys=True, separators=(",", ":")).encode()
    assert "sha256:" + hashlib.sha256(canonical).hexdigest() == task_set["tool_catalog_sha256"]
    assert [tool["name"] for tool in catalog] == ["bash", "submit_report"]
    assert task_set["training_data_eligible"] is True
    assert [row["split"] for row in split["tasks"]] == ["train", "dev"]
    assert len(task_set["tasks"]) == 2
    assert run["data"]["root"] == data["output"]
    assert run["data"]["manifest"] == data["output"] + "/manifest.json"

    assert run["recipe"] == {
        "nodes": 1,
        "steps": 1,
        "groups": 1,
        "samples_per_prompt": 8,
        "lr": 1e-6,
        "eval_interval": 1,
        "checkpoint_interval": 1,
        "seed": 42,
    }
    assert run["cluster"] == {
        "priority": "c1",
        "resources": {
            "cpu_request": "64",
            "cpu_limit": "128",
            "memory_request": "1536Gi",
            "memory_limit": "2048Gi",
        },
    }
    assert run["checkpoint"] == {
        "manifest": "/mnt/sfs/jobs/chris-cpt-cleanup-q38-miles-base-v1/NATIVE_CHECKPOINT.json",
        "sha256": "sha256:b3d772de9121f442ea7b9a4c9a996f2a0a99cab8c49fe3083c148fe3eebd089c",
    }
    assert run["model"]["root"] == data["model_root"]
    assert "webexploitbench" not in (DATA.read_text() + RUN.read_text()).lower()


def test_miles_reward_canary_keeps_full_horizon_and_no_retry_controls() -> None:
    data, run = load(DATA), load(RUN)

    assert data["limits"] == {
        "context_tokens": 98304,
        "response_tokens": 81920,
        "max_tokens_per_turn": 4096,
        "max_turns": 600,
        "episode_seconds": 2400,
        "tool_seconds": 330,
        "tool_result_chars": 50000,
    }
    assert run["output_root"] != data["output"]
    assert "requeue" not in json.dumps(run).lower()


def test_v3_is_a_fresh_native_one_by_eight_dev_canary() -> None:
    data_v1, run_v1, data_v2, run_v2, data_v3, run_v3, launch = map(
        load, (DATA, RUN, DATA_V2, RUN_V2, DATA_V3, RUN_V3, LAUNCH_V3)
    )

    assert run_v2["recipe"]["nodes"] == 2 and run_v2["recipe"]["gpus_per_node"] == 4
    assert data_v3["name"] == run_v3["name"] == run_v3["wandb"]["run_id"]
    assert data_v3["name"] == "chris-q38-miles-rlreward-dev3"
    assert data_v3["output"] == "/mnt/sfs/jobs/chris-q38-miles-rlreward-inputs-dev3/data"
    assert run_v3["output_root"] == "/mnt/sfs/jobs/chris-q38-miles-rlreward-dev3"
    assert run_v3["data"] == {
        "manifest": data_v3["output"] + "/manifest.json",
        "root": data_v3["output"],
    }
    assert run_v3["wandb"] == {
        **run_v1["wandb"],
        "run_id": "chris-q38-miles-rlreward-dev3",
    }

    data_identity = (
        "backend",
        "task_set",
        "split",
        "tool_catalog",
        "model_lock",
        "model_root",
        "limits",
    )
    for key in data_identity:
        assert data_v3[key] == data_v1[key]
    for key in ("backend", "model", "checkpoint", "cluster"):
        assert run_v3[key] == run_v1[key]

    assert run_v3["recipe"] == {**run_v1["recipe"], "gpus_per_node": 8}
    assert run_v3["cluster"]["priority"] == "c1"
    identities = launch["identities"]
    assert launch["status"] == "configuration_only_unsubmitted"
    assert launch["cluster_target"] == "dev"
    assert identities["run_name"] == identities["cyber_run_id"] == data_v3["name"]
    assert identities["data_root"] == data_v3["output"]
    assert identities["prepared_root"] == (
        "/mnt/sfs/jobs/chris-q38-miles-rlreward-inputs-dev3/prepared"
    )
    assert identities["output_root"] == run_v3["output_root"]
    assert identities["wandb_run_id"] == run_v3["wandb"]["run_id"]
    old = json.dumps([data_v1, run_v1, data_v2, run_v2])
    for value in identities.values():
        assert value not in old
    assert "webexploitbench" not in (DATA_V3.read_text() + RUN_V3.read_text()).lower()


def test_v4_preserves_science_but_uses_only_fresh_identities() -> None:
    data_v3, run_v3, data_v4, run_v4, launch = map(
        load, (DATA_V3, RUN_V3, DATA_V4, RUN_V4, LAUNCH_V4)
    )

    assert data_v4["name"] == run_v4["name"] == run_v4["wandb"]["run_id"]
    assert data_v4["name"] == "chris-q38-miles-rlreward-dev4"
    assert data_v4["output"] == "/mnt/sfs/jobs/chris-q38-miles-rlreward-inputs-dev4/data"
    assert run_v4["output_root"] == "/mnt/sfs/jobs/chris-q38-miles-rlreward-dev4"
    assert run_v4["data"] == {
        "manifest": data_v4["output"] + "/manifest.json",
        "root": data_v4["output"],
    }
    for key in (
        "backend",
        "task_set",
        "split",
        "tool_catalog",
        "model_lock",
        "model_root",
        "limits",
    ):
        assert data_v4[key] == data_v3[key]
    for key in ("backend", "model", "checkpoint", "recipe"):
        assert run_v4[key] == run_v3[key]
    assert run_v4["cluster"] == {**run_v3["cluster"], "target": "dev"}

    identities = launch["identities"]
    assert launch["status"] == "configuration_only_unsubmitted"
    assert launch["cluster_target"] == "dev"
    assert identities == {
        "run_name": data_v4["name"],
        "cyber_run_id": data_v4["name"],
        "data_root": data_v4["output"],
        "prepared_root": "/mnt/sfs/jobs/chris-q38-miles-rlreward-inputs-dev4/prepared",
        "output_root": run_v4["output_root"],
        "wandb_run_id": run_v4["wandb"]["run_id"],
    }
    old = json.dumps([data_v3, run_v3])
    assert all(value not in old for value in identities.values())
    assert launch["supersedes"]["run_name"] == data_v3["name"]
    assert launch["required_runtime"] == {
        "estimator_fix_commit": "652d1135dee0ae019c1255df8e734163bb6f5fd2",
        "source_commit_policy": (
            "the immutable bundle commit must equal or descend from estimator_fix_commit"
        ),
        "native_parsed_arguments": {
            "calculate_per_token_loss": True,
            "grpo_std_normalization": False,
        },
    }
    assert "webexploitbench" not in (DATA_V4.read_text() + RUN_V4.read_text()).lower()


def test_v5_reuses_accepted_dev4_data_but_has_fresh_run_identity() -> None:
    data_v4, run_v4, run_v5, launch = map(load, (DATA_V4, RUN_V4, RUN_V5, LAUNCH_V5))

    assert run_v5["name"] == run_v5["wandb"]["run_id"]
    assert run_v5["name"] == "chris-q38-miles-rlreward-dev5"
    assert run_v5["output_root"] == "/mnt/sfs/jobs/chris-q38-miles-rlreward-dev5"
    assert run_v5["data"] == {
        "manifest": data_v4["output"] + "/manifest.json",
        "root": data_v4["output"],
    }
    for key in ("backend", "model", "checkpoint"):
        assert run_v5[key] == run_v4[key]
    assert run_v5["recipe"] == {
        "nodes": 1,
        "gpus_per_node": 8,
        "steps": 1,
        "groups": 1,
        "samples_per_prompt": 8,
        "lr": 2e-6,
        "temperature": 0.7,
        "kl_loss_coef": 0.001,
        "max_tokens_per_gpu": 8192,
        "eval_interval": 1,
        "checkpoint_interval": 1,
        "seed": 42,
    }
    assert run_v5["cluster"] == {
        "target": "dev",
        "priority": "c1",
        "resources": {
            "cpu_request": "32",
            "cpu_limit": "32",
            "memory_request": "1800Gi",
            "memory_limit": "2400Gi",
        },
    }

    assert launch["status"] == "configuration_only_unsubmitted"
    assert launch["identities"] == {
        "run_name": run_v5["name"],
        "cyber_run_id": run_v5["name"],
        "data_root": data_v4["output"],
        "output_root": run_v5["output_root"],
        "wandb_run_id": run_v5["wandb"]["run_id"],
    }
    assert launch["parity"]["optimizer_updates"] == 1
    assert launch["parity"]["total_train_samples"] == 8
    assert launch["parity"]["parallelism"] == {
        "tensor": 4,
        "pipeline": 1,
        "context": 2,
        "sequence_parallel": True,
    }
    assert launch["parity"]["recompute"]["granularity"] == "full"
    api = launch["jobs_api_constraints"]
    assert api["priority_request"] == "c1"
    assert api["rendered_queue_priority"] == "q1"
    assert api["effective_priority"] == 10000
    assert api["requeue_if_preempted"] is False
    assert api["requested_shared_memory"] == "128Gi"
    assert api["expressible_shared_memory"] is False
    assert api["current_rendered_shared_memory"] == "64Gi"
    assert launch["supersedes"]["current_state"] == (
        "operator_cancelled_once_delete_204_then_reconciled_absent"
    )
    assert launch["supersedes"]["jobs_api_delete_calls"] == 1
    assert launch["supersedes"]["jobs_api_delete_status"] == 204
    assert launch["supersedes"]["jobs_api_get_after_delete_status"] == 404
    assert launch["supersedes"]["gpu_allocations"] == 0
    assert launch["supersedes"]["optimizer_updates"] == 0
    assert run_v5["name"] not in json.dumps([data_v4, run_v4])
    assert "webexploitbench" not in (RUN_V5.read_text() + LAUNCH_V5.read_text()).lower()


def test_dev3_retirement_is_append_only_and_proves_zero_execution() -> None:
    queued, retired = load(
        ROOT / "docs/evidence/qwen38-study/2026-09-12-miles-rlreward-dev3-queued-v1.json"
    ), load(DEV3_RETIRED)

    assert retired["recorded_at"] == "2026-09-13T07:40:23Z"
    assert retired["run"]["api_run_id"] == queued["run"]["api_run_id"]
    assert retired["run"]["rayjob_uid"] == queued["run"]["rayjob_uid"]
    assert retired["run"]["workload_uid"] == queued["run"]["workload_uid"]
    assert retired["retirement"] == {
        "jobs_api_delete_requests": 1,
        "jobs_api_delete_accepted": True,
        "jobs_api_get_after_delete_http_status": 404,
        "exact_rayjob_absent": True,
        "exact_workload_absent": True,
        "raycluster_ever_created": False,
        "pod_ever_created": False,
        "gpus_ever_allocated": 0,
        "resource_release_uncertain": False,
    }
    assert retired["scientific_accounting"] == {
        "task_interactions": 0,
        "verifier_calls": 0,
        "reward_batches": 0,
        "optimizer_updates": 0,
        "checkpoint_files": 0,
        "valid_model_outcome": False,
    }
    assert retired["reason"]["fresh_successor_required"] is True
    assert retired["reason"]["unchanged_resubmission_allowed"] is False
    assert not any(retired["privacy"].values())
    assert retired["production_authorized_by_this_record"] is False
