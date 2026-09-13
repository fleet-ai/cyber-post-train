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
