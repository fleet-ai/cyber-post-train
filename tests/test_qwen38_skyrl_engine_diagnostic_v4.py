"""Offline identity and supersession gates for the fourth SkyRL diagnostic."""
# ruff: noqa: F811

import hashlib
import json
from pathlib import Path

from test_rl_data import setup  # noqa: F401
from test_skyrl_data import skyrl as data_setup  # noqa: F401
from test_skyrl_training import prepared as skyrl_prepared  # noqa: F401

from cyber_post_train.jobs import digest
from training import skyrl_training

ROOT = Path(__file__).resolve().parents[1]
DATA_V2 = ROOT / (
    "configs/qualification/"
    "qwen38-rl-filtered-skyrl-engine-diagnostic-data-dev-v2.json"
)
DATA_V4 = ROOT / (
    "configs/qualification/"
    "qwen38-rl-filtered-skyrl-engine-diagnostic-data-dev-v4.json"
)
RUN_V3 = ROOT / "configs/qualification/qwen38-rl-filtered-skyrl-engine-diagnostic-dev-v3.json"
RUN_V4 = ROOT / "configs/qualification/qwen38-rl-filtered-skyrl-engine-diagnostic-dev-v4.json"
EVIDENCE = ROOT / (
    "docs/evidence/qwen38-study/"
    "2026-09-12-skyrl-engine-diagnostic-dev3-supersession-v1.json"
)


def load(path):
    return json.loads(path.read_bytes())


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v4_uses_fresh_create_once_identities_with_unchanged_science():
    data_v2, data_v4 = load(DATA_V2), load(DATA_V4)
    run_v3, run_v4 = load(RUN_V3), load(RUN_V4)

    assert data_v4["name"] == run_v4["name"] == run_v4["wandb"]["run_id"]
    assert data_v4["name"] == "chris-q38-rldiag-dev4"
    assert run_v4["output_root"] == "/mnt/sfs/jobs/chris-q38-rldiag-dev4"
    assert data_v4["output"] == run_v4["data"]["root"]
    assert run_v4["data"]["manifest"] == data_v4["output"] + "/manifest.json"
    assert data_v4["output"] != data_v2["output"]
    assert run_v4["output_root"] != run_v3["output_root"]

    for key in ("backend", "task_set", "split", "tool_catalog", "model_lock", "model_root"):
        assert data_v4[key] == data_v2[key]
    assert data_v4["limits"] == data_v2["limits"]
    for key in ("backend", "model", "recipe", "cluster"):
        assert run_v4[key] == run_v3[key]


def test_v4_remains_dev_engine_only_c1_and_no_requeue_by_construction(
    skyrl_prepared,
):
    run = load(RUN_V4)
    assert run["cluster"]["priority"] == "c1"
    assert run["recipe"] == {
        "nodes": 1,
        "steps": 1,
        "groups": 2,
        "samples_per_prompt": 4,
        "lr": 0.000001,
        "eval_interval": 1,
        "checkpoint_interval": 1,
        "keep_checkpoints": 2,
        "seed": 42,
    }
    request = skyrl_training.engine_diagnostic_request(skyrl_prepared.plan)
    assert request["workers"] == 2
    assert request["gpus_per_worker"] == 4
    assert request["priority_class"] == "c1"
    assert request["requeueIfPreempted"] is False
    assert request["secrets"] == []
    assert "WANDB_MODE" not in request["env"]
    assert "FLEET_API_KEY" not in request["env"]


def test_dev3_supersession_evidence_is_self_digesting_and_binds_v4_runtime():
    value = load(EVIDENCE)
    assert value["sha256"] == digest({k: v for k, v in value.items() if k != "sha256"})
    assert value["classification"] == "infrastructure_invalid_runtime_identity_adapter"
    assert value["predecessor"]["plan_sha256"] == (
        "1201009189125f493ac98d179e78f5cc7e5904c7fd040f6030326aa9dd946806"
    )
    assert value["predecessor"]["qualification"] == {
        "task_rows_read": 0,
        "rollouts": 0,
        "verifier_calls": 0,
        "optimizer_steps": 0,
        "checkpoints_created": 0,
    }
    assert value["successor"]["data_config_sha256"] == file_sha256(DATA_V4)
    assert value["successor"]["run_config_sha256"] == file_sha256(RUN_V4)
    assert value["successor"]["runtime_sha256"] == digest(skyrl_training._runtime())
    assert value["successor"]["submitted"] is False
    assert value["successor"]["allowed_cluster"] == "dev"
    assert value["successor"]["predecessor_replay_forbidden"] is True
    assert value["successor"]["required_effective_priority"] == 10000
    assert value["successor"]["required_terminal_receipt"] == "ENGINE_DIAGNOSTIC.json"
