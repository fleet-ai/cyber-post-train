"""Static identity for the smallest real Qwen3.8 Miles reward/update canary."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "configs/qualification/qwen38-miles-rl-reward-canary-data-dev-v1.json"
RUN = ROOT / "configs/qualification/qwen38-miles-rl-reward-canary-dev-v1.json"
TASK_SET = ROOT / "configs/data/qwen38-rl-reward-canary-task-set-v1.json"
SPLIT = ROOT / "configs/data/qwen38-rl-reward-canary-split-v1.json"


def load(path: Path) -> dict:
    return json.loads(path.read_bytes())


def test_miles_reward_canary_is_one_dev_update_on_the_exact_source_task() -> None:
    data, run, task_set, split = map(load, (DATA, RUN, TASK_SET, SPLIT))

    assert data["backend"] == run["backend"] == "miles"
    assert data["name"] == run["name"] == run["wandb"]["run_id"]
    assert data["name"] == "chris-q38-miles-rlreward-dev1"
    assert data["task_set"] == "../data/qwen38-rl-reward-canary-task-set-v1.json"
    assert data["split"] == "../data/qwen38-rl-reward-canary-split-v1.json"
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
