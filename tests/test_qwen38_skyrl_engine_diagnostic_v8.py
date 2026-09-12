"""Corrected relay-image and fresh-identity gates for the SkyRL dev8 smoke."""
# ruff: noqa: F811

import hashlib
import json
from pathlib import Path

import pytest
from test_rl_data import setup  # noqa: F401
from test_skyrl_data import skyrl as data_setup  # noqa: F401
from test_skyrl_training import prepared as skyrl_prepared  # noqa: F401

from cyber_post_train.jobs import digest
from training import skyrl_training

ROOT = Path(__file__).resolve().parents[1]
DATA_V7 = ROOT / (
    "configs/qualification/qwen38-rl-filtered-skyrl-engine-diagnostic-data-dev-v7.json"
)
DATA_V8 = ROOT / (
    "configs/qualification/qwen38-rl-filtered-skyrl-engine-diagnostic-data-dev-v8.json"
)
RUN_V7 = ROOT / "configs/qualification/qwen38-rl-filtered-skyrl-engine-diagnostic-dev-v7.json"
RUN_V8 = ROOT / "configs/qualification/qwen38-rl-filtered-skyrl-engine-diagnostic-dev-v8.json"
QUALIFICATION = ROOT / (
    "docs/evidence/qwen38-study/2026-09-12-skyrl-startup-relay-image-cpu-qualification-v1.json"
)
WORKER_RPC_QUALIFICATION = ROOT / (
    "docs/evidence/qwen38-study/2026-09-12-skyrl-worker-rpc-relay-image-cpu-qualification-v1.json"
)
TERMINAL = ROOT / (
    "docs/evidence/qwen38-study/2026-09-12-skyrl-startup-relay-image-terminal-v1.json"
)
DEV7_TERMINAL = ROOT / (
    "docs/evidence/qwen38-study/2026-09-12-skyrl-engine-diagnostic-dev7-terminal-v1.json"
)
HISTORICAL_IMAGE = (
    "661864827319.dkr.ecr.us-east-1.amazonaws.com/fleet/skyrl-train@sha256:"
    "e48827529b1cf5fafa153b2aed1b774c2eec86905baf5ccb62b36300533e252b"
)
HISTORICAL_SOURCE_COMMIT = "34de8d5753b8dfe44460ff9656db4ddc9a85a62c"
WORKER_RPC_SOURCE_COMMIT = "8d62868d6dc00eee793d83efe5738dc21e42758d"


def load(path: Path) -> dict:
    return json.loads(path.read_bytes())


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_historical_relay_image_is_bound_to_its_exact_cleanpull_receipt():
    receipt = load(QUALIFICATION)
    canonical = json.dumps(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    assert receipt["receipt_sha256"] == hashlib.sha256(canonical).hexdigest()
    assert receipt["receipt_sha256"] == (
        "4326ec6a28f1f2deee6d6ebaf9c04f80ec8ba5ac8856ae636e91aca5e4e53837"
    )
    assert receipt["requested_image"] == HISTORICAL_IMAGE
    assert receipt["expected_source_commit"] == HISTORICAL_SOURCE_COMMIT
    assert receipt["status"] == "qualified"
    assert receipt["classification"] == "operational_gate"
    assert receipt["details"]["installed_file_sha256"]["v1/engine/fleet_startup_error.py"] == (
        "077803a51e7c41473315ecf56eb58ecd2e23a5544ae0d3267a7059d3c3b63713"
    )
    assert receipt["details"]["qualification_source_sha256"] == {
        "/tmp/skyrl-fleet-owned/fleet_vllm_startup_error.py": (
            "077803a51e7c41473315ecf56eb58ecd2e23a5544ae0d3267a7059d3c3b63713"
        ),
        "/tmp/skyrl-fleet-owned/integration_test_fleet_vllm_startup_error.py": (
            "735c4216d70ded3de0c2452c7fb94ed45738e2d54b6dbb8e4cb347d7142e5c5b"
        ),
        "/tmp/skyrl-fleet-owned/test_fleet_vllm_startup_error.py": (
            "2e1d229634f85d91249053441355ce4a6aaa7204b711052fd4586232712ff25c"
        ),
    }
    assert receipt["details"]["unit_spawn_cases"] == 9
    assert receipt["details"]["installed_vllm_spawn_integration_cases"] == 6
    assert all(receipt["checks"].values())

    assert skyrl_training.IMAGE != HISTORICAL_IMAGE


def test_historical_relay_image_is_not_privacy_qualified_for_dev8(skyrl_prepared):
    plan = skyrl_prepared.plan
    plan["execution"]["image"] = HISTORICAL_IMAGE
    plan["run_name"] = skyrl_training.ENGINE_DIAGNOSTIC_CONFIG_NAME
    plan["arguments"]["name"] = skyrl_training.ENGINE_DIAGNOSTIC_CONFIG_NAME
    with pytest.raises(ValueError, match="worker-RPC evidence"):
        skyrl_training._require_engine_start_qualified_image(plan)


def test_worker_rpc_relay_image_is_cpu_qualified_and_bound_to_current_plan(
    skyrl_prepared,
):
    receipt = load(WORKER_RPC_QUALIFICATION)
    canonical = json.dumps(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    assert receipt["receipt_sha256"] == hashlib.sha256(canonical).hexdigest()
    assert receipt["receipt_sha256"] == (
        "28244d695896b8b9766df66caecd117a33fd5d9c2c5df35faba12bcc785d09f1"
    )
    assert receipt["requested_image"] == skyrl_training.IMAGE
    assert receipt["runtime"]["runtime_image_id"] == skyrl_training.IMAGE
    assert receipt["expected_source_commit"] == WORKER_RPC_SOURCE_COMMIT
    assert receipt["build"] == {
        "commit": WORKER_RPC_SOURCE_COMMIT,
        "tag": "q38-relay-v2-8d62868d",
        "workflow_run_id": 34724428519,
        "workflow_status": "success",
    }
    assert receipt["details"]["installed_file_sha256"] == {
        "v1/engine/core.py": ("73c72d2762d0cfef6f1fa3624dfb7a02c1b336c217cb3eaa4e9a049d58c0cc3f"),
        "v1/engine/fleet_startup_error.py": (
            "23d604c1619f39ffde72d89b89ef78847a09e28f4c2ea41e56c19b73b4f3aff1"
        ),
        "v1/engine/utils.py": ("60bd9439dd8c8b6d8c0fdabf4962ecfc1c425566ca0155071d6e978fa6098aa8"),
        "v1/executor/multiproc_executor.py": (
            "56ae7181effeb62ac3deb684457cad7ab7f478b01a466c64345c201432935e70"
        ),
    }
    assert receipt["details"]["qualification_source_sha256"] == {
        "/tmp/skyrl-fleet-owned/fleet_vllm_startup_error.py": (
            "23d604c1619f39ffde72d89b89ef78847a09e28f4c2ea41e56c19b73b4f3aff1"
        ),
        "/tmp/skyrl-fleet-owned/integration_test_fleet_vllm_startup_error.py": (
            "2203a25e43446620c463866d6e025a83250157cfaedf4fce10c04bfa1b3293d7"
        ),
        "/tmp/skyrl-fleet-owned/test_fleet_vllm_startup_error.py": (
            "31a09541250924d7a2ddd165a076a2f76d8039449ee516af96bec580c58ce3a6"
        ),
    }
    assert receipt["details"]["unit_spawn_cases"] == 10
    assert receipt["details"]["installed_vllm_spawn_integration_cases"] == 8
    assert receipt["runtime"]["gpu_request"] is None
    assert receipt["runtime"]["restart_count"] == 0
    assert all(receipt["checks"].values())
    assert all(receipt["cleanup"].values())

    plan = skyrl_prepared.plan
    assert plan["execution"]["image"] == skyrl_training.IMAGE
    assert plan["execution"]["image_cpu_qualification"] == (
        skyrl_training.ENGINE_IMAGE_CPU_QUALIFICATION
    )
    assert {
        "schema": receipt["schema"],
        "status": receipt["status"],
        "classification": receipt["classification"],
        "source_commit": receipt["expected_source_commit"],
        "receipt_sha256": receipt["receipt_sha256"],
        "evidence_path": str(WORKER_RPC_QUALIFICATION.relative_to(ROOT)),
    } == skyrl_training.ENGINE_IMAGE_CPU_QUALIFICATION


def test_cleanpull_terminal_evidence_cross_binds_identity_and_release():
    terminal = load(TERMINAL)
    assert terminal["receipt_sha256"] == digest(
        {key: value for key, value in terminal.items() if key != "receipt_sha256"}
    )
    assert terminal["status"] == "qualified_and_released"
    assert terminal["source"] == {
        "build_run_id": 34710564708,
        "build_status": "success",
        "requested_image": HISTORICAL_IMAGE,
        "runtime_image_id": HISTORICAL_IMAGE,
        "theseus_commit": HISTORICAL_SOURCE_COMMIT,
    }
    assert (
        terminal["qualification"]["receipt_embedded_sha256"]
        == (load(QUALIFICATION)["receipt_sha256"])
    )
    assert terminal["qualification"]["receipt_file_sha256"] == file_sha256(QUALIFICATION)
    assert terminal["qualification"]["helper_sha256"] == (
        "077803a51e7c41473315ecf56eb58ecd2e23a5544ae0d3267a7059d3c3b63713"
    )
    qualified_sources = load(QUALIFICATION)["details"]["qualification_source_sha256"]
    assert (
        terminal["qualification"]["integration_test_sha256"]
        == (
            qualified_sources["/tmp/skyrl-fleet-owned/integration_test_fleet_vllm_startup_error.py"]
        )
    )
    assert (
        terminal["qualification"]["unit_test_sha256"]
        == qualified_sources["/tmp/skyrl-fleet-owned/test_fleet_vllm_startup_error.py"]
    )
    assert terminal["qualification"]["all_six_checks_passed"] is True
    assert terminal["qualification"]["unit_spawn_cases"] == 9
    assert terminal["qualification"]["installed_vllm_spawn_integration_cases"] == 6
    assert terminal["pod"]["gpu_request"] is None
    assert terminal["pod"]["restart_count"] == 0
    assert terminal["cleanup"]["pod_absent_after_delete"] is True
    assert terminal["cleanup"]["pod_deleted"] is True
    assert terminal["claim_boundary"]["not_established"]


def test_dev7_terminal_copy_is_exact_and_remains_infrastructure_invalid():
    value = load(DEV7_TERMINAL)
    assert file_sha256(DEV7_TERMINAL) == (
        "0b083be5b52e8cb723920a451bcc3c5b0caabcae0db8cee27998ddf8d876bab7"
    )
    assert value["classification"] == "infrastructure_invalid_relay_transport"
    assert value["receipt"]["engine_start_qualified"] is False
    assert value["receipt"]["training_qualified"] is False
    assert value["kubernetes"]["gpu_release_proven"] is True
    assert value["diagnosis"]["theseus_fix_commit"] == HISTORICAL_SOURCE_COMMIT


def test_dev8_changes_only_create_once_identities_from_dev7():
    data_v7, data_v8 = load(DATA_V7), load(DATA_V8)
    run_v7, run_v8 = load(RUN_V7), load(RUN_V8)

    assert data_v8["name"] == run_v8["name"] == run_v8["wandb"]["run_id"]
    assert data_v8["name"] == "chris-q38-rldiag-dev8"
    assert run_v8["output_root"] == "/mnt/sfs/jobs/chris-q38-rldiag-dev8"
    assert data_v8["output"] == run_v8["data"]["root"]
    assert run_v8["data"]["manifest"] == data_v8["output"] + "/manifest.json"
    assert data_v8["output"] != data_v7["output"]
    assert run_v8["output_root"] != run_v7["output_root"]
    assert run_v8["wandb"]["run_id"] != run_v7["wandb"]["run_id"]

    for key in ("backend", "task_set", "split", "tool_catalog", "model_lock", "model_root"):
        assert data_v8[key] == data_v7[key]
    assert data_v8["limits"] == data_v7["limits"]
    for key in ("backend", "model", "recipe", "cluster"):
        assert run_v8[key] == run_v7[key]


def test_dev8_request_is_current_zero_work_tp4x2(skyrl_prepared):
    plan = skyrl_prepared.plan
    request = skyrl_training.engine_diagnostic_request(plan)

    skyrl_training._require_engine_start_qualified_image(plan)
    assert request["image"] == skyrl_training.IMAGE
    assert (request["workers"], request["gpus_per_worker"]) == (2, 4)
    assert request["priority_class"] == "c1"
    assert request["requeueIfPreempted"] is False
    assert request["secrets"] == []
    assert request["env"]["CYBER_EXPECTED_RUNTIME_UID"] == "1000"
    assert request["env"]["CYBER_EXPECTED_RUNTIME_GID"] == "100"
    assert "WANDB" not in json.dumps(request["env"])
    assert digest(request) != digest(skyrl_training.job_request(plan))
    assert request["title"].endswith("SkyRL engine-start diagnostic")
    assert request["run_dir"] == plan["output_root"]
