"""Fresh immutable identity and zero-training gates for SkyRL dev5."""
# ruff: noqa: F811

import ast
import hashlib
import json
import subprocess
from pathlib import Path

from test_rl_data import setup  # noqa: F401
from test_skyrl_data import skyrl as data_setup  # noqa: F401
from test_skyrl_training import prepared as skyrl_prepared  # noqa: F401

from cyber_post_train.jobs import digest
from training import skyrl_training

ROOT = Path(__file__).resolve().parents[1]
DATA_V4 = ROOT / (
    "configs/qualification/qwen38-rl-filtered-skyrl-engine-diagnostic-data-dev-v4.json"
)
DATA_V5 = ROOT / (
    "configs/qualification/qwen38-rl-filtered-skyrl-engine-diagnostic-data-dev-v5.json"
)
RUN_V4 = ROOT / "configs/qualification/qwen38-rl-filtered-skyrl-engine-diagnostic-dev-v4.json"
RUN_V5 = ROOT / "configs/qualification/qwen38-rl-filtered-skyrl-engine-diagnostic-dev-v5.json"
EVIDENCE = ROOT / (
    "docs/evidence/qwen38-study/2026-09-12-skyrl-engine-diagnostic-dev5-prelaunch-v1.json"
)
RUNTIME_SOURCE_COMMIT = "ae0f2b71733105074d41806560fe659a2035b9ba"


def load(path):
    return json.loads(path.read_bytes())


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_runtime():
    def source(path):
        return subprocess.check_output(
            ["git", "show", f"{RUNTIME_SOURCE_COMMIT}:{path}"], cwd=ROOT, text=True
        )

    module = ast.parse(source("training/skyrl_training.py"))
    files = next(
        ast.literal_eval(node.value)
        for node in module.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "RUNTIME_FILES" for target in node.targets
        )
    )
    return {path: source(path) for path in files}


def test_dev5_changes_only_create_once_identities_from_dev4():
    data_v4, data_v5 = load(DATA_V4), load(DATA_V5)
    run_v4, run_v5 = load(RUN_V4), load(RUN_V5)

    assert data_v5["name"] == run_v5["name"] == run_v5["wandb"]["run_id"]
    assert data_v5["name"] == "chris-q38-rldiag-dev5"
    assert run_v5["output_root"] == "/mnt/sfs/jobs/chris-q38-rldiag-dev5"
    assert data_v5["output"] == run_v5["data"]["root"]
    assert run_v5["data"]["manifest"] == data_v5["output"] + "/manifest.json"
    assert data_v5["output"] != data_v4["output"]
    assert run_v5["output_root"] != run_v4["output_root"]

    for key in ("backend", "task_set", "split", "tool_catalog", "model_lock", "model_root"):
        assert data_v5[key] == data_v4[key]
    assert data_v5["limits"] == data_v4["limits"]
    for key in ("backend", "model", "recipe", "cluster"):
        assert run_v5[key] == run_v4[key]


def test_dev5_request_is_dev_engine_only_c1_no_requeue_and_no_secrets(skyrl_prepared):
    request = skyrl_training.engine_diagnostic_request(skyrl_prepared.plan)
    assert request["workers"] == 2
    assert request["gpus_per_worker"] == 4
    assert request["priority_class"] == "c1"
    assert request["requeueIfPreempted"] is False
    assert request["secrets"] == []
    assert request["run_dir"] == skyrl_prepared.plan["output_root"]
    assert request["name"] == skyrl_prepared.plan["run_name"]
    assert "WANDB_MODE" not in request["env"]
    assert "FLEET_API_KEY" not in request["env"]

    # The engine diagnostic is a separate executable mode. It must not use the
    # native training request, even though both are derived from one frozen plan.
    assert request != skyrl_training.job_request(skyrl_prepared.plan)


def test_dev5_prelaunch_evidence_binds_frozen_runtime_and_zero_training_scope():
    value = load(EVIDENCE)
    assert value["sha256"] == digest({k: v for k, v in value.items() if k != "sha256"})
    assert value["classification"] == "operational_gate_prepared_not_submitted"
    assert value["correction"]["commit"] == ("7599706f1b5626d7ad2588804c75ea85ceeb099b")
    assert value["correction"]["causal_status"] == ("strong_inference_not_terminally_bound")

    successor = value["successor"]
    assert successor["data_config_sha256"] == file_sha256(DATA_V5)
    assert successor["run_config_sha256"] == file_sha256(RUN_V5)
    assert successor["runtime_source_commit"] == RUNTIME_SOURCE_COMMIT
    assert successor["runtime_sha256"] == digest(frozen_runtime())
    assert successor["allowed_cluster"] == "dev"
    assert successor["priority_class"] == "c1"
    assert successor["queue_priority_class"] == "q1"
    assert successor["required_effective_priority"] == 10000
    assert successor["automatic_requeue"] is False
    assert successor["submitted"] is False
    assert successor["workers"] * successor["gpus_per_worker"] == 8
    assert not any(
        successor[key]
        for key in (
            "task_rows_read",
            "rollouts",
            "verifier_calls",
            "optimizer_steps",
            "checkpoints_created",
            "wandb",
        )
    )
    assert value["cpu_preflight"]["gpus"] == 0
    assert value["cpu_preflight"]["status"] == "passed"
    assert value["cpu_preflight"]["engine_start_qualified"] is False
    assert value["dev_preview"] == {
        "api_base_url": "https://api.ft.dev.flt.build",
        "status": "passed",
        "manifest_sha256": "bbd3b8c441ef102ef720df6882a5e43d49833f9c281e60556e714da0064bd70b",
        "duplicate_matches": 0,
        "output_root_absent": True,
        "physical_resources_created": False,
    }
