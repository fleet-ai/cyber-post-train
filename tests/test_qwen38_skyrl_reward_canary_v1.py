"""Static safety and provenance gates for the first real Qwen3.8 RL dev canary."""
# ruff: noqa: F811

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest
from test_rl_data import setup  # noqa: F401
from test_skyrl_data import skyrl as data_setup  # noqa: F401
from test_skyrl_training import prepared as skyrl_prepared  # noqa: F401

from cyber_post_train import cli
from cyber_post_train.jobs import JobsError, digest
from training import rl_data, skyrl_training

ROOT = Path(__file__).resolve().parents[1]
TASK_SET = ROOT / "configs/data/qwen38-rl-reward-canary-task-set-v1.json"
SPLIT = ROOT / "configs/data/qwen38-rl-reward-canary-split-v1.json"
DATA = ROOT / "configs/qualification/qwen38-rl-reward-canary-data-dev-v1.json"
RUN = ROOT / "configs/qualification/qwen38-rl-reward-canary-dev-v1.json"
RELOAD = ROOT / ("configs/qualification/qwen38-rl-reward-canary-reload-dev-v1.template.json")
FILTERED_TASK_SET = ROOT / "configs/data/qwen38-rl-filtered-canary-task-set-v1.json"
DEV8_DATA = ROOT / (
    "configs/qualification/qwen38-rl-filtered-skyrl-engine-diagnostic-data-dev-v8.json"
)
DEV8_RUN = ROOT / ("configs/qualification/qwen38-rl-filtered-skyrl-engine-diagnostic-dev-v8.json")
SUCCESSOR_EVIDENCE = ROOT / (
    "configs/runs/qwen36-27b-native-rl-reward-acquisition-canary.pre-submit.json"
)


def load(path: Path) -> dict:
    return json.loads(path.read_bytes())


def assert_self_digest(value: dict) -> None:
    assert value["sha256"] == "sha256:" + digest(
        {key: item for key, item in value.items() if key != "sha256"}
    )


def test_task_selection_uses_only_exact_receipt_bound_successors() -> None:
    task_set, split = load(TASK_SET), load(SPLIT)
    old_task_set, evidence = load(FILTERED_TASK_SET), load(SUCCESSOR_EVIDENCE)
    assert_self_digest(task_set)
    assert_self_digest(split)
    provenance = task_set["reward_signal_provenance"]
    assert provenance["kind"] == "historically_nonzero_source_to_metadata_only_successor"
    assert provenance["evidence_path"] == str(SUCCESSOR_EVIDENCE.relative_to(ROOT))
    assert (
        provenance["evidence_file_sha256"]
        == "sha256:" + hashlib.sha256(SUCCESSOR_EVIDENCE.read_bytes()).hexdigest()
    )
    assert task_set["source_manifest_sha256"] == old_task_set["source_manifest_sha256"]
    assert task_set["tool_catalog_sha256"] == old_task_set["tool_catalog_sha256"]

    successors = {
        row["source_task_version_id"]: row["metadata_only_successor_task_version_id"]
        for row in evidence["task_successors"]
    }
    old_by_key = {row["task_key"]: row for row in old_task_set["tasks"]}
    new_by_key = {row["task_key"]: row for row in task_set["tasks"]}
    assignments = {
        (row["task_key"], row["task_version_id"]): row["split"] for row in split["tasks"]
    }
    train = [
        row
        for row in task_set["tasks"]
        if assignments[(row["task_key"], row["task_version_id"])] == "train"
    ]
    dev = [
        row
        for row in task_set["tasks"]
        if assignments[(row["task_key"], row["task_version_id"])] == "dev"
    ]
    assert len(train) == 1 and len(dev) == 1
    for row in train:
        source = old_by_key[row["task_key"]]
        assert row["task_version_id"] == successors[source["task_version_id"]]
        assert {key: row[key] for key in row if key != "task_version_id"} == {
            key: source[key] for key in source if key != "task_version_id"
        }
    assert dev[0] == old_by_key[dev[0]["task_key"]]
    assert set(new_by_key) < set(old_by_key)
    assert [row["split"] for row in rl_data.selection(task_set, split)].count("train") == 1
    assert [row["split"] for row in rl_data.selection(task_set, split)].count("dev") == 1
    assert "webexploit" not in json.dumps(task_set).lower()
    assert all(row["reference_session_id"] is None for row in split["tasks"])


def test_real_canary_is_one_dev_only_update_with_durable_evidence() -> None:
    data, run = load(DATA), load(RUN)
    dev8_data, dev8_run = load(DEV8_DATA), load(DEV8_RUN)
    assert data["name"] == run["name"] == run["wandb"]["run_id"]
    assert data["name"] == "chris-q38-rlreward-dev1"
    assert data["backend"] == run["backend"] == "skyrl"
    assert run["output_root"] == "/mnt/sfs/jobs/chris-q38-rlreward-dev1"
    assert data["output"] == run["data"]["root"]
    assert run["data"]["manifest"] == data["output"] + "/manifest.json"
    assert data["task_set"].endswith("qwen38-rl-reward-canary-task-set-v1.json")
    assert data["split"].endswith("qwen38-rl-reward-canary-split-v1.json")

    for key in ("backend", "model_lock", "model_root", "limits"):
        assert data[key] == dev8_data[key]
    assert data["tool_catalog"] == dev8_data["tool_catalog"]
    assert run["model"] == dev8_run["model"]
    for key in (
        "nodes",
        "steps",
        "lr",
        "eval_interval",
        "checkpoint_interval",
        "keep_checkpoints",
        "seed",
    ):
        assert run["recipe"][key] == dev8_run["recipe"][key]
    assert (
        run["recipe"]["groups"] * run["recipe"]["samples_per_prompt"]
        == dev8_run["recipe"]["groups"] * dev8_run["recipe"]["samples_per_prompt"]
    )
    assert run["recipe"] == {
        "nodes": 1,
        "steps": 1,
        "groups": 1,
        "samples_per_prompt": 8,
        "lr": 1e-6,
        "eval_interval": 1,
        "checkpoint_interval": 1,
        "keep_checkpoints": 2,
        "seed": 42,
        "engine_start_timeout_seconds": 1800,
        "engine_cleanup_timeout_seconds": 300,
    }
    assert run["cluster"] == {
        "target": "dev",
        "priority": "c1",
        "resources": {
            "cpu_request": "64",
            "cpu_limit": "64",
            "memory_request": "512Gi",
            "memory_limit": "768Gi",
        },
    }
    assert run["recipe"]["groups"] * run["recipe"]["samples_per_prompt"] == 8
    assert run["recipe"]["steps"] == run["recipe"]["checkpoint_interval"] == 1
    assert run["wandb"] == {
        "entity": "thefleet",
        "project": "cyber-post-train",
        "run_id": run["name"],
    }
    assert data["output"] != dev8_data["output"]
    assert run["output_root"] != dev8_run["output_root"]


def test_cluster_target_is_immutable_and_request_never_requeues(skyrl_prepared) -> None:
    plan = deepcopy(skyrl_prepared.plan)
    plan["execution"]["cluster_target"] = "dev"
    request = skyrl_training.job_request(plan)
    cli._require_prepared_cluster(plan, cli.Cluster.dev)
    with pytest.raises(JobsError, match="dev-cluster-only"):
        cli._require_prepared_cluster(plan, cli.Cluster.prod)
    assert request["workers"] == 1 and request["gpus_per_worker"] == 8
    assert request["priority_class"] == "c1"
    assert request["requeueIfPreempted"] is False
    assert request["secrets"] == ["fleet-api", "wandb-api"]
    assert request["env"]["WANDB_MODE"] == "online"


def test_reload_gate_is_digest_bound_and_deliberately_not_launchable() -> None:
    reload = load(RELOAD)
    assert_self_digest(reload)
    assert reload["status"] == "blocked_not_launchable"
    assert reload["launchable"] is False
    assert reload["source"]["run_name"] == load(RUN)["name"]
    assert reload["source"]["checkpoint_global_step"] == 1
    assert reload["source"]["plan_sha256"] is None
    assert reload["source"]["checkpoint_manifest"] is None
    assert reload["source"]["checkpoint_manifest_file_sha256"] is None
    assert reload["target"] == {
        "run_name": "chris-q38-rlreward-reload-dev1",
        "output_root": "/mnt/sfs/jobs/chris-q38-rlreward-reload-dev1",
        "cluster": "dev",
        "priority_class": "c1",
        "priority_value": 10000,
        "workers": 1,
        "gpus_per_worker": 8,
        "requeue_if_preempted": False,
        "optimizer_updates_authorized": 0,
        "rollouts_authorized": 0,
        "wandb_run_id": "chris-q38-rlreward-reload-dev1",
    }
    assert len(reload["blockers"]) == 3
