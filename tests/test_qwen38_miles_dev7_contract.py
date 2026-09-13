import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUALIFICATION = ROOT / "configs" / "qualification"


def load(name: str) -> dict:
    return json.loads((QUALIFICATION / name).read_bytes())


def test_dev7_data_run_launch_identities_are_fresh_and_exact() -> None:
    data = load("qwen38-miles-rl-reward-canary-data-dev-v7.json")
    run = load("qwen38-miles-rl-reward-canary-dev-v7.json")
    launch = load("qwen38-miles-rl-reward-canary-launch-dev-v7.json")
    run_id = "chris-q38-miles-rlreward-dev7"
    data_root = "/mnt/sfs/jobs/chris-q38-miles-rlreward-inputs-dev7-v1/data"
    prepared_root = "/mnt/sfs/jobs/chris-q38-miles-rlreward-preflight-dev7-v1"
    output_root = "/mnt/sfs/jobs/chris-q38-miles-rlreward-dev7"

    assert data["name"] == run["name"] == run_id
    assert data["output"] == run["data"]["root"] == data_root
    assert run["data"]["manifest"] == f"{data_root}/manifest.json"
    assert run["output_root"] == output_root
    assert run["wandb"] == {
        "entity": "thefleet",
        "project": "cyber-post-train",
        "run_id": run_id,
    }
    assert launch["identities"] == {
        "run_name": run_id,
        "cyber_run_id": run_id,
        "data_root": data_root,
        "prepared_root": prepared_root,
        "output_root": output_root,
        "wandb_run_id": run_id,
    }
    assert launch["data_config"].endswith("data-dev-v7.json")
    assert launch["run_config"].endswith("dev-v7.json")

    text = "\n".join(
        (QUALIFICATION / name).read_text()
        for name in (
            "qwen38-miles-rl-reward-canary-data-dev-v7.json",
            "qwen38-miles-rl-reward-canary-dev-v7.json",
            "qwen38-miles-rl-reward-canary-launch-dev-v7.json",
        )
    ).lower()
    assert "dev6" not in text
    assert "preflight-dev6" not in text
    assert '"c0"' not in text and '"q0"' not in text
    assert "priority_reason" not in text


def test_dev7_preserves_the_reviewed_one_update_c1_contract() -> None:
    data = load("qwen38-miles-rl-reward-canary-data-dev-v7.json")
    run = load("qwen38-miles-rl-reward-canary-dev-v7.json")
    launch = load("qwen38-miles-rl-reward-canary-launch-dev-v7.json")

    assert data["task_set"].endswith("qwen38-rl-filtered-canary-task-set-v2.json")
    assert data["split"].endswith("qwen38-rl-filtered-canary-split-v2.json")
    assert data["limits"]["tool_result_chars"] == 4000
    assert run["recipe"] == {
        "nodes": 1,
        "gpus_per_node": 8,
        "steps": 1,
        "groups": 1,
        "samples_per_prompt": 8,
        "lr": 2e-06,
        "temperature": 0.7,
        "kl_loss_coef": 0.001,
        "max_tokens_per_gpu": 8192,
        "eval_interval": 1,
        "checkpoint_interval": 1,
        "seed": 42,
    }
    assert run["cluster"] == {
        "target": "dev",
        "priority": "c1",
        "resources": {
            "cpu_request": "32",
            "cpu_limit": "32",
            "memory_request": "1800Gi",
            "memory_limit": "2400Gi",
        },
    }
    assert launch["jobs_api_constraints"] == {
        "priority_request": "c1",
        "rendered_queue_priority": "q1",
        "effective_priority": 10000,
        "maximum_allowed_priority": "c1",
        "requeue_if_preempted": False,
        "current_rendered_shared_memory": "64Gi",
    }
    assert launch["source"]["old_data_reused"] is False
    assert "webexploitbench" not in json.dumps((data, run, launch)).lower()
