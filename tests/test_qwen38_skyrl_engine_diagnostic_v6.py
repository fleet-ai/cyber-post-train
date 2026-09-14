"""Frozen prelaunch gates for the exact, now-retired SkyRL dev6 diagnostic."""
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
DATA_V5 = ROOT / (
    "configs/qualification/qwen38-rl-filtered-skyrl-engine-diagnostic-data-dev-v5.json"
)
DATA_V6 = ROOT / (
    "configs/qualification/qwen38-rl-filtered-skyrl-engine-diagnostic-data-dev-v6.json"
)
RUN_V5 = ROOT / "configs/qualification/qwen38-rl-filtered-skyrl-engine-diagnostic-dev-v5.json"
RUN_V6 = ROOT / "configs/qualification/qwen38-rl-filtered-skyrl-engine-diagnostic-dev-v6.json"
EVIDENCE = ROOT / (
    "docs/evidence/qwen38-study/2026-09-12-skyrl-engine-diagnostic-dev6-prelaunch-v1.json"
)
DEV5_TERMINAL = ROOT / (
    "docs/evidence/qwen38-study/2026-09-12-skyrl-engine-diagnostic-dev5-terminal-v1.json"
)
RUNTIME_SOURCE_COMMIT = "a29c4937a21b49735301e1abfd10fbc9f27bafe7"


def load(path):
    return json.loads(path.read_bytes())


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_source(path):
    return subprocess.check_output(
        ["git", "show", f"{RUNTIME_SOURCE_COMMIT}:{path}"], cwd=ROOT, text=True
    )


def frozen_runtime():
    module = ast.parse(frozen_source("training/skyrl_training.py"))
    files = next(
        ast.literal_eval(node.value)
        for node in module.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "RUNTIME_FILES" for target in node.targets
        )
    )
    return {path: frozen_source(path) for path in files}


def test_dev6_changes_only_create_once_identities_from_dev5():
    data_v5, data_v6 = load(DATA_V5), load(DATA_V6)
    run_v5, run_v6 = load(RUN_V5), load(RUN_V6)

    assert data_v6["name"] == run_v6["name"] == run_v6["wandb"]["run_id"]
    assert data_v6["name"] == "chris-q38-rldiag-dev6"
    assert run_v6["output_root"] == "/mnt/sfs/jobs/chris-q38-rldiag-dev6"
    assert data_v6["output"] == run_v6["data"]["root"]
    assert run_v6["data"]["manifest"] == data_v6["output"] + "/manifest.json"
    assert data_v6["output"] != data_v5["output"]
    assert run_v6["output_root"] != run_v5["output_root"]

    for key in ("backend", "task_set", "split", "tool_catalog", "model_lock", "model_root"):
        assert data_v6[key] == data_v5[key]
    assert data_v6["limits"] == data_v5["limits"]
    for key in ("backend", "model", "recipe", "cluster"):
        assert run_v6[key] == run_v5[key]


def test_dev6_toggle_is_frozen_history_and_absent_from_live_requests(skyrl_prepared):
    historical = frozen_source("training/skyrl_training.py")
    assert 'ENGINE_DIAGNOSTIC_VLLM_V1_MULTIPROCESSING = "0"' in historical
    assert '"VLLM_ENABLE_V1_MULTIPROCESSING"' in historical

    diagnostic = skyrl_training.engine_diagnostic_request(skyrl_prepared.plan)
    production = skyrl_training.job_request(skyrl_prepared.plan)
    assert "VLLM_ENABLE_V1_MULTIPROCESSING" not in diagnostic["env"]
    assert "VLLM_ENABLE_V1_MULTIPROCESSING" not in production["env"]
    assert diagnostic["workers"] == 2
    assert diagnostic["gpus_per_worker"] == 4
    assert diagnostic["priority_class"] == "c1"
    assert diagnostic["requeueIfPreempted"] is False
    assert diagnostic["secrets"] == []


def test_dev6_prelaunch_evidence_is_self_digesting_and_truthful():
    value = load(EVIDENCE)
    assert value["sha256"] == digest({k: v for k, v in value.items() if k != "sha256"})
    assert value["classification"] == "operational_gate_qualified_not_submitted"
    assert value["predecessor"]["diagnostic_status"] == "engine_start_rejected"
    assert value["predecessor"]["terminal_receipt"]["digest_valid"] is True
    terminal = load(DEV5_TERMINAL)
    assert value["predecessor"]["terminal_evidence"] == {
        "path": str(DEV5_TERMINAL.relative_to(ROOT)),
        "file_sha256": file_sha256(DEV5_TERMINAL),
        "embedded_sha256": terminal["sha256"],
    }
    assert value["predecessor"]["qualification"]["optimizer_steps"] == 0
    assert value["predecessor"]["release"]["gpus_held"] == 0
    assert value["predecessor"]["replay_forbidden"] is True

    correction = value["correction"]
    assert correction["environment_variable"] == "VLLM_ENABLE_V1_MULTIPROCESSING"
    assert correction["value"] == "0"
    assert correction["production_native_training_behavior_changed"] is False
    assert correction["causal_status"] == "unqualified_until_dev6_terminal_receipt"

    successor = value["successor"]
    assert successor["data_config_sha256"] == file_sha256(DATA_V6)
    assert successor["run_config_sha256"] == file_sha256(RUN_V6)
    assert (
        successor["skyrl_training_source_sha256"]
        == hashlib.sha256(frozen_source("training/skyrl_training.py").encode()).hexdigest()
    )
    assert successor["runtime_sha256"] == digest(frozen_runtime())
    assert successor["allowed_cluster"] == "dev"
    assert successor["priority_class"] == "c1"
    assert successor["queue_priority_class"] == "q1"
    assert successor["required_effective_priority"] == 10000
    assert successor["automatic_requeue"] is False
    assert successor["workers"] * successor["gpus_per_worker"] == 8
    assert successor["plan_sha256"] == (
        "dc34abf5325baff35875893ffec05c28564028d1f0f2ad1aa852e077d3fb9609"
    )
    assert successor["request_sha256"] == (
        "1923f8fedac4aba5d036acaf26d37cebd94621f6cb47e1d89844249dfda10dde"
    )
    assert all(
        successor[key]
        for key in (
            "data_materialized",
            "plan_compiled",
            "pinned_image_cpu_preflight_passed",
            "dev_api_preview_passed",
        )
    )
    assert not any(
        successor[key]
        for key in (
            "task_rows_read",
            "rollouts",
            "verifier_calls",
            "optimizer_steps",
            "checkpoints_created",
            "wandb",
            "submitted",
        )
    )
    assert value["data_preparation"]["manifest_digest_valid"] is True
    assert value["data_preparation"]["operator_prompt_content_inspected"] is False
    assert value["cpu_preflight"]["gpus"] == 0
    assert value["cpu_preflight"]["status"] == "passed"
    assert value["cpu_preflight"]["vllm_v1_multiprocessing_disabled"] is True
    assert value["cpu_preflight"]["engine_start_qualified"] is False
    assert value["dev_preview"]["status"] == "passed"
    assert value["dev_preview"]["physical_resources_created"] is False
    snapshot = value["immediate_readiness_snapshot"]
    assert snapshot["fleet_team_id"] == "a1025f0b-ad67-49fc-a023-51800ab43e84"
    assert snapshot["jobs_api_duplicate_matches"] == 0
    assert snapshot["kubernetes_duplicate_matches"] == 0
    assert snapshot["output_root_absent"] is snapshot["submission_journal_absent"] is True
    assert snapshot["nodes_with_at_least_four_free_gpus"] == 2
    assert value["audit_boundaries"]["cluster_mutations"] == 1
    assert value["audit_boundaries"]["zero_gpu_helpers_created"] == 1
    assert value["audit_boundaries"]["private_logs_read"] is False
