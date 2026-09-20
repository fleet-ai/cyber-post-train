import copy
import json
import os
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from cyber_post_train.jobs import validate_request
from training.fti_v1_training import (
    EVIDENCE_SCHEMA,
    IMAGE,
    _environment_only_wandb_args,
    _spool_reward_evidence,
    digest,
    job_request,
    native_arguments,
    task_rows,
    validate_plan,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "configs/runs/qwen38-27b-fti-v1-rl-reward-canary-v1.json"
PRODUCTION_PLAN_PATH = ROOT / "configs/runs/qwen38-27b-fti-v1-rl-production-a1.json"
ACTIVE_CANARY_PATH = ROOT / "configs/runs/qwen38-27b-fti-v1-rl-reward-canary-v3.json"


def plan():
    return json.loads(PLAN_PATH.read_text())


def production_plan():
    return json.loads(PRODUCTION_PLAN_PATH.read_text())


def test_exact_train_only_plan_and_native_recipe():
    value = validate_plan(plan(), root=ROOT)
    row = json.loads(task_rows(value))
    assert row["messages"] == [{"role": "user", "content": "Run the versioned Fleet task."}]
    assert row["metadata"]["task_version_id"] == value["data"]["tasks"][0]["task_version_id"]
    argv = native_arguments(value)
    assert "qwen3.8-27b-256k" in argv and "v1" in argv
    extra = argv[argv.index("--extra-args") + 1]
    assert "--num-rollout 1" in extra and "--save-interval 1" in extra
    assert "training.fti_v1_training.generate" in extra


def test_dev_or_final_task_cannot_enter_training():
    value = plan()
    split = json.loads((ROOT / value["data"]["study_split"]["path"]).read_text())
    held_out = next(row for row in split["tasks"] if row["split"] != "train")
    value["data"]["tasks"] = [
        {"task_key": held_out["task_key"], "task_version_id": held_out["task_version_id"]}
    ]
    with pytest.raises(ValueError, match="training split"):
        validate_plan(value, root=ROOT)


@pytest.mark.parametrize("fault", ["version", "duplicate", "priority", "image", "accepted"])
def test_plan_drift_fails_before_request(fault):
    value = plan()
    if fault == "version":
        value["data"]["tasks"][0]["task_version_id"] = "00000000-0000-0000-0000-000000000000"
    elif fault == "duplicate":
        value["data"]["tasks"] *= 2
        value["optimization"]["prompt_groups"] = 2
    elif fault == "priority":
        value["cluster"]["priority"] = "c0"
    elif fault == "image":
        value["trainer"]["image"] = "registry/image@sha256:" + "0" * 64
    else:
        value["acceptance"]["optimizer_step"] = True
    with pytest.raises(ValueError):
        validate_plan(value, root=ROOT)


def test_jobs_request_is_create_once_c1_and_contains_no_plain_task_prompt():
    value = json.loads(ACTIVE_CANARY_PATH.read_text())
    request = job_request(value, root=ROOT)
    validate_request(request)
    assert request["image"] == IMAGE
    assert request["workers"] == 4 and request["gpus_per_worker"] == 8
    assert request.get("privileged", False) is False
    assert request["priority_class"] == "c1" and request["requeueIfPreempted"] is False
    assert request["failureAlerts"] is False
    assert request["run_dir"] == "/mnt/sfs/jobs/" + request["name"]
    assert value["data"]["tasks"][0]["task_key"] not in request["command"]
    assert "Run the versioned Fleet task." not in request["command"]
    assert all("API_KEY" not in key for key in request["env"])
    assert all(
        not secret or secret not in request["command"]
        for secret in (os.environ.get("FLEET_API_KEY"), os.environ.get("WANDB_API_KEY"))
    )
    assert request["env"]["CYBER_PLAN_SHA256"] == digest(validate_plan(value, root=ROOT))
    assert request["env"]["CYBER_REWARD_EVIDENCE_DIR"].endswith("/evidence/verifier-executions")
    assert request["env"]["CYBER_REWARD_HMAC_KEY"].endswith("/.private/reward-hmac.key")


def test_sanitized_reward_evidence_is_create_once_and_idempotent(tmp_path, monkeypatch):
    value = validate_plan(json.loads(ACTIVE_CANARY_PATH.read_text()), root=ROOT)
    task = value["data"]["tasks"][0]
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    key_path = tmp_path / "reward.key"
    key_path.write_bytes(b"k" * 32)
    verifier_version_id = str(uuid.uuid4())
    verifier_execution_id = str(uuid.uuid4())
    fleet = {
        "graded": True,
        "reward": 0.5,
        "task_key": task["task_key"],
        "task_version_id": task["task_version_id"],
        "verifier_version_id": verifier_version_id,
        "verifier_execution_id": verifier_execution_id,
        "instance_id": str(uuid.uuid4()),
        "cleanup_error": None,
    }
    samples = [
        SimpleNamespace(metadata={"fleet_v1": dict(fleet)}, reward=0.5),
        SimpleNamespace(metadata={"fleet_v1": dict(fleet)}, reward=0.5),
    ]
    result = SimpleNamespace(samples=samples)
    monkeypatch.setenv("CYBER_PLAN_SHA256", digest(value))
    monkeypatch.setenv("CYBER_REWARD_EVIDENCE_DIR", str(evidence_dir))
    monkeypatch.setenv("CYBER_REWARD_HMAC_KEY", str(key_path))

    _spool_reward_evidence(result, task)
    _spool_reward_evidence(result, task)
    files = list(evidence_dir.iterdir())
    assert [path.name for path in files] == [verifier_execution_id + ".json"]
    payload = files[0].read_text()
    receipt = json.loads(payload)
    receipt_sha256 = receipt.pop("receipt_sha256")
    assert receipt_sha256 == digest(receipt)
    assert set(receipt) == {
        "schema",
        "plan_sha256",
        "task_key",
        "task_version_id",
        "verifier_version_id",
        "verifier_execution_id",
        "reward_class",
        "reward_fingerprint",
        "instance_cleanup_confirmed",
    }
    assert receipt["schema"] == EVIDENCE_SCHEMA
    assert receipt["reward_class"] == "finite_numeric"
    assert receipt["instance_cleanup_confirmed"] is True
    assert "0.5" not in payload
    assert fleet["instance_id"] not in payload


@pytest.mark.parametrize("fault", ["task", "cleanup", "reward", "execution"])
def test_invalid_reward_evidence_never_writes(tmp_path, monkeypatch, fault):
    value = validate_plan(json.loads(ACTIVE_CANARY_PATH.read_text()), root=ROOT)
    task = value["data"]["tasks"][0]
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    key_path = tmp_path / "reward.key"
    key_path.write_bytes(b"k" * 32)
    fleet = {
        "graded": True,
        "reward": 0.5,
        "task_key": task["task_key"],
        "task_version_id": task["task_version_id"],
        "verifier_version_id": str(uuid.uuid4()),
        "verifier_execution_id": str(uuid.uuid4()),
        "cleanup_error": None,
    }
    if fault == "task":
        fleet["task_version_id"] = str(uuid.uuid4())
    elif fault == "cleanup":
        fleet["cleanup_error"] = "not released"
    elif fault == "reward":
        fleet["reward"] = float("nan")
    else:
        fleet["verifier_execution_id"] = "missing"
    result = SimpleNamespace(
        samples=SimpleNamespace(metadata={"fleet_v1": fleet}, reward=fleet["reward"])
    )
    monkeypatch.setenv("CYBER_PLAN_SHA256", digest(value))
    monkeypatch.setenv("CYBER_REWARD_EVIDENCE_DIR", str(evidence_dir))
    monkeypatch.setenv("CYBER_REWARD_HMAC_KEY", str(key_path))
    with pytest.raises(ValueError):
        _spool_reward_evidence(result, task)
    assert not list(evidence_dir.iterdir())


def test_wandb_environment_auth_never_enters_arguments(monkeypatch):
    marker = "preflight-wandb-" + uuid.uuid4().hex
    monkeypatch.setenv("WANDB_API_KEY", marker)
    arguments = _environment_only_wandb_args("run_fleet.py", run_id="ignored")
    assert arguments == "--use-wandb --disable-wandb-random-suffix"
    assert marker not in arguments


def test_production_plan_expands_exact_locked_train_split_only():
    source = production_plan()
    assert source["data"]["tasks"] == "all_locked_train"
    value = validate_plan(source, root=ROOT)
    split = json.loads((ROOT / value["data"]["study_split"]["path"]).read_text())
    expected = [
        {"task_key": row["task_key"], "task_version_id": row["task_version_id"]}
        for row in split["tasks"]
        if row["split"] == "train"
    ]
    held_out = {
        (row["task_key"], row["task_version_id"])
        for row in split["tasks"]
        if row["split"] != "train"
    }
    actual = value["data"]["tasks"]
    assert actual == expected and len(actual) == 59
    assert not held_out.intersection((row["task_key"], row["task_version_id"]) for row in actual)
    assert source["data"]["tasks"] == "all_locked_train"


def test_production_request_is_bounded_c1_and_checkpointed():
    value = production_plan()
    request = job_request(value, root=ROOT)
    validate_request(request)
    normalized = validate_plan(value, root=ROOT)
    argv = native_arguments(normalized)
    extra = argv[argv.index("--extra-args") + 1]
    assert argv[argv.index("--rollout-batch-size") + 1] == "8"
    assert "--num-rollout 8" in extra and "--save-interval 1" in extra
    assert "--wandb-run-id chris-q38-v1rl-prod-a1" in extra
    assert request["workers"] == 4 and request["gpus_per_worker"] == 8
    assert request["priority_class"] == "c1" and request.get("privileged", False) is False


def test_production_explicit_task_drift_is_rejected():
    value = production_plan()
    split = json.loads((ROOT / value["data"]["study_split"]["path"]).read_text())
    held_out = next(row for row in split["tasks"] if row["split"] == "dev")
    value["data"]["tasks"] = [
        {"task_key": held_out["task_key"], "task_version_id": held_out["task_version_id"]}
    ]
    with pytest.raises(ValueError, match="every task"):
        validate_plan(value, root=ROOT)


def test_bound_manifest_byte_drift_is_fatal(tmp_path):
    value = copy.deepcopy(plan())
    original = ROOT / value["data"]["study_split"]["path"]
    changed = tmp_path / "split.json"
    changed.write_bytes(original.read_bytes() + b"\n")
    value["data"]["study_split"]["path"] = str(changed.relative_to(tmp_path))
    with pytest.raises(ValueError, match="bound data manifest changed"):
        validate_plan(value, root=tmp_path)
