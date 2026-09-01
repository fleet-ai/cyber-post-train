from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from evals.fleet import holdout, qwen38_holdout_gate, self_hosted

PLAN_PATH = Path("evals/fleet/configs/qwen38-27b-qwen-code-test20-base-v1.json")
Q36_PLAN_PATH = Path("evals/fleet/configs/qwen36-27b-qwen-code-test20-base-v1.json")
SPLIT_PATH = Path("configs/data/fleet-a62-task-split-v1.json")
MODEL_LOCK_PATH = Path("configs/models/qwen38-27b-1d4bf0f2.lock.json")
SERVING_LOCK_PATH = Path("evals/webexploitbench/serving/qwen38-27b-1d4bf0f2.lock.json")
JOB_PATH = Path("evals/fleet/cluster/qwen38-code-selfhosted-test20-job.yaml")
SUBMIT_PATH = Path("evals/fleet/scripts/submit_selfhosted_qwen38_test20.sh")


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(self_hosted.canonical_json(value) + b"\n")


def _terminal_gate_fixture(tmp_path: Path, *, score: float = 0.0) -> tuple[dict, Path, Path]:
    plan = _json(PLAN_PATH)
    canary_version = plan["launch_gate"]["task_version_id"]
    tasks = [
        {"index": index, "task_key": f"task-{index}", "task_version_id": f"version-{index}"}
        for index in range(1, 21)
    ]
    tasks[0]["task_version_id"] = canary_version
    calibration_receipt = {
        "campaign_id": plan["launch_gate"]["campaign_id"],
        "task_count": 20,
        "tasks": tasks,
    }
    calibration_receipt["receipt_sha256"] = self_hosted.sha256(
        self_hosted.canonical_json(calibration_receipt)
    )
    plan["launch_gate"]["frozen_receipt_sha256"] = calibration_receipt["receipt_sha256"]
    receipt_path = tmp_path / "calibration-receipt.json"
    _write(receipt_path, calibration_receipt)

    root = tmp_path / plan["launch_gate"]["campaign_id"]
    task_root = root / "task-01"
    instance_id = "11111111-1111-4111-8111-111111111111"
    evidence_run_id = "22222222-2222-4222-8222-222222222222"
    verifier_id = "33333333-3333-4333-8333-333333333333"
    run_id = f"{plan['launch_gate']['campaign_id']}-t01-deadbeef"
    outcome = {
        "index": 1,
        "task_key": "task-1",
        "task_version_id": canary_version,
        "status": "model_outcome",
        "cleanup_verified": True,
        "score": score,
        "verifier_execution_id": verifier_id,
    }
    _write(
        root / "campaign-state.json",
        {
            "campaign_id": plan["launch_gate"]["campaign_id"],
            "planned_sessions": 20,
            "completed_attempts": 1,
            "outcomes": [outcome],
        },
    )
    _write(
        task_root / "binding.json",
        {
            "run_id": run_id,
            "task": {"key": "task-1", "version_id": canary_version},
            "model": {
                field: plan["model"][field]
                for field in ("repository", "revision", "served_id", "endpoint_origin")
            },
            "harness": {
                field: plan["harness"][field]
                for field in ("name", "version", "source_commit")
            },
        },
    )
    _write(
        task_root / "runtime-binding.json",
        {"instance_id": instance_id, "evidence_run_id": evidence_run_id},
    )
    _write(
        task_root / "reward-result.json",
        {
            "task_key": "task-1",
            "task_version_id": canary_version,
            "instance_id": instance_id,
            "reward": score,
            "verifier_execution_id": verifier_id,
        },
    )
    _write(
        task_root / "result.json",
        {"run_id": run_id, "score": score, "verifier_execution_id": verifier_id},
    )
    _write(
        task_root / "cleanup.json",
        {"instance_created": True, "instance_closed": True, "containers_removed": True},
    )
    return plan, receipt_path, root


def test_qwen38_holdout_inherits_exact_q36_sealed_test20() -> None:
    plan = _json(PLAN_PATH)
    q36_plan = _json(Q36_PLAN_PATH)
    split = _json(SPLIT_PATH)
    rows = holdout.validate_plan(plan, split)
    q36_rows = holdout.validate_plan(q36_plan, split)
    assert [row["task_version_id"] for row in rows] == [
        row["task_version_id"] for row in q36_rows
    ]
    assert len(rows) == 20
    assert {row["split"] for row in rows} == {"test"}


def test_qwen38_holdout_binds_exact_model_harness_and_runtime() -> None:
    plan = _json(PLAN_PATH)
    model_lock = _json(MODEL_LOCK_PATH)
    serving_lock = _json(SERVING_LOCK_PATH)
    assert plan["model"]["repository"] == model_lock["repo"]
    assert plan["model"]["revision"] == model_lock["revision"]
    assert plan["model"]["tokenizer_revision"] == model_lock["tokenizer"]["manifest_sha256"]
    assert plan["model"]["chat_template_revision"] == "sha256:" + next(
        row["sha256"]
        for row in model_lock["tokenizer"]["files"]
        if row["path"] == "chat_template.jinja"
    )
    assert plan["model"]["live_identity"]["catalog"]["id"] == serving_lock["id"]
    assert plan["model"]["live_identity"]["server_info"] == {
        "model_path": f"/scratch/models/qwen3.8-27b/{model_lock['revision']}",
        "served_model_name": serving_lock["id"],
        "context_length": 262144,
        "tp_size": 1,
        "quantization": None,
        "kv_cache_dtype": "fp8_e4m3",
        "reasoning_parser": "qwen3",
        "tool_call_parser": "qwen3_coder",
    }
    assert plan["harness"]["version"] == "0.22.3"
    assert plan["harness"]["source_commit"] == "09825973e7d3c3fd07e17909c396aa62f48ce51f"
    assert plan["harness"]["max_model_requests"] == 600
    assert plan["execution"]["required_task_tools"] == ["bash", "submit_report"]
    assert plan["execution"]["training_data_eligible"] is False


def test_qwen38_holdout_uses_unique_resource_prefix() -> None:
    plan = _json(PLAN_PATH)
    row = {
        "index": 1,
        "task_key": "task-key",
        "task_version_id": "task-version",
        "prompt_sha256": "sha256:prompt",
        "env_variables_sha256": "sha256:env",
        "output_json_schema_sha256": "sha256:schema",
        "env_key": "env",
        "env_version": "v1",
        "data_key": "data",
        "data_version": "v2",
        "runtime_seed_content_sha256": None,
        "verifier": {"id": "v", "version_id": "vv", "version": 1, "sha256": "v"},
    }
    config = holdout.task_config(plan, row)
    assert config["execution"]["network"].startswith("q38qcode-test20-b1-01-")


def test_terminal_valid_zero_releases_holdout_creation_gate(tmp_path: Path) -> None:
    plan, receipt_path, root = _terminal_gate_fixture(tmp_path, score=0.0)
    receipt = qwen38_holdout_gate.build_gate_receipt(plan, receipt_path, root)
    qwen38_holdout_gate.validate_gate_receipt(plan, receipt)
    assert receipt["canary"]["outcome_status"] == "model_outcome"
    assert receipt["canary"]["score"] == 0.0
    assert receipt["canary"]["cleanup_verified"] is True


def test_holdout_creation_gate_rejects_incomplete_cleanup(tmp_path: Path) -> None:
    plan, receipt_path, root = _terminal_gate_fixture(tmp_path, score=1.0)
    cleanup_path = root / "task-01" / "cleanup.json"
    cleanup = _json(cleanup_path)
    cleanup["instance_closed"] = False
    _write(cleanup_path, cleanup)
    with pytest.raises(ValueError, match="instance was not closed"):
        qwen38_holdout_gate.build_gate_receipt(plan, receipt_path, root)


def test_frozen_gate_receipt_rejects_digest_drift(tmp_path: Path) -> None:
    plan, receipt_path, root = _terminal_gate_fixture(tmp_path, score=1.0)
    receipt = qwen38_holdout_gate.build_gate_receipt(plan, receipt_path, root)
    drifted = copy.deepcopy(receipt)
    drifted["canary"]["cleanup_verified"] = False
    with pytest.raises(ValueError, match="digest mismatch"):
        qwen38_holdout_gate.validate_gate_receipt(plan, drifted)


def test_cluster_launcher_is_suspended_create_only_and_gate_first() -> None:
    job = yaml.safe_load(JOB_PATH.read_text())
    assert job["metadata"]["name"] == "chris-cyber-qwen38-qcode-fleet-test20-base-v1"
    assert job["spec"]["suspend"] is True
    assert job["spec"]["backoffLimit"] == 0
    assert job["metadata"]["labels"]["kueue.x-k8s.io/queue-name"] == "training-lq"
    script = SUBMIT_PATH.read_text()
    assert "qwen38_holdout_gate build" in script
    assert script.index("qwen38_holdout_gate build") < script.index(
        "configmap | kubectl create -f -"
    )
    assert "kubectl apply" not in script
