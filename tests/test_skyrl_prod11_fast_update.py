"""Provider-free science and identity checks for the parallel fast-update canary."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from cyber_post_train.jobs import digest
from training import rl_reward_canary as canary
from training import sft, skyrl_training
from training import skyrl_reward_rayjob as historical

ROOT = Path(__file__).resolve().parents[1]
QUALIFICATION = ROOT / "configs/qualification"
PROD11_RUN = QUALIFICATION / "qwen38-rl-reward-canary-prod-v11.json"
FAST_RUN = QUALIFICATION / "qwen38-rl-reward-canary-prod-v11-fast1.json"
PROD11_DATA = QUALIFICATION / "qwen38-rl-reward-canary-data-prod-v11.json"
FAST_DATA = QUALIFICATION / "qwen38-rl-reward-canary-data-prod11-fast1.json"
FAST_IDENTITY = QUALIFICATION / "qwen38-rl-reward-canary-prod11-fast1-identity-v1.json"
FAST2_RUN = QUALIFICATION / "qwen38-rl-reward-canary-prod-v11-fast2.json"
FAST2_DATA = QUALIFICATION / "qwen38-rl-reward-canary-data-prod11-fast2.json"
FAST2_IDENTITY = QUALIFICATION / "qwen38-rl-reward-canary-prod11-fast2-identity-v1.json"
BASE_MANIFEST = QUALIFICATION / "qwen38-rl-reward-canary-manifest-prod-v8.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_bytes())


def _compile_fast(monkeypatch) -> dict:
    run = _load(FAST_RUN)
    manifest = copy.deepcopy(_load(BASE_MANIFEST))
    manifest["name"] = run["name"]
    manifest["sha256"] = "sha256:" + digest(
        {key: value for key, value in manifest.items() if key != "sha256"}
    )
    original = sft.read_mapping

    def read(path: Path) -> dict:
        if Path(path) == Path(run["data"]["manifest"]):
            return copy.deepcopy(manifest)
        return original(path)

    monkeypatch.setattr(sft, "read_mapping", read)
    return skyrl_training.compile_rl(run, relative_to=FAST_RUN.parent)


def _compile_fast2(monkeypatch) -> dict:
    run = _load(FAST2_RUN)
    manifest = copy.deepcopy(_load(BASE_MANIFEST))
    manifest["name"] = run["name"]
    manifest["sha256"] = "sha256:" + digest(
        {key: value for key, value in manifest.items() if key != "sha256"}
    )
    original = sft.read_mapping

    def read(path: Path) -> dict:
        if Path(path) == Path(run["data"]["manifest"]):
            return copy.deepcopy(manifest)
        return original(path)

    monkeypatch.setattr(sft, "read_mapping", read)
    return skyrl_training.compile_rl(run, relative_to=FAST2_RUN.parent)


def test_fast_update_changes_only_operational_identity_and_baseline_eval() -> None:
    prod11 = _load(PROD11_RUN)
    fast = _load(FAST_RUN)
    normalized = copy.deepcopy(fast)
    normalized["name"] = prod11["name"]
    normalized["output_root"] = prod11["output_root"]
    normalized["data"] = copy.deepcopy(prod11["data"])
    normalized["wandb"]["run_id"] = prod11["wandb"]["run_id"]
    assert normalized["recipe"].pop("eval_before_train") is False

    assert normalized == prod11
    assert fast["recipe"]["groups"] == 1
    assert fast["recipe"]["samples_per_prompt"] == 8
    assert fast["recipe"]["steps"] == 1
    assert fast["recipe"]["eval_interval"] == 1
    assert fast["recipe"]["checkpoint_interval"] == 1
    assert fast["recipe"]["keep_checkpoints"] == 2
    assert fast["cluster"]["priority"] == "c1"


def test_fast_update_reuses_exact_reviewed_task_and_compaction_contract() -> None:
    prod11 = _load(PROD11_DATA)
    fast = _load(FAST_DATA)
    normalized = copy.deepcopy(fast)
    normalized["name"] = prod11["name"]
    normalized["output"] = prod11["output"]

    assert normalized == prod11
    assert fast["task_set"] == "../data/qwen38-rl-reward-canary-task-set-v3.json"
    assert fast["limits"] == {
        "compaction_summary_tokens": 8192,
        "compaction_trigger_tokens": 163840,
        "context_tokens": 262144,
        "episode_seconds": 14400,
        "generation_chunk_tokens": 4096,
        "max_tokens_per_turn": 32768,
        "max_turns": 1200,
        "response_tokens": 4194304,
        "tool_result_chars": 50000,
        "tool_seconds": 330,
    }


def test_fast_update_identity_is_sealed_fresh_and_never_reuses_prod11_output() -> None:
    raw = _load(FAST_IDENTITY)
    identity = historical.load_identity(FAST_IDENTITY)
    body = {key: value for key, value in raw.items() if key != "sha256"}

    assert raw["sha256"] == "sha256:" + digest(body)
    assert identity.run_name == "chris-q38-rlreward-prod11-fast1"
    assert identity.predecessor_run_name == "chris-q38-rlreward-prod11"
    assert identity.output_root == "/mnt/sfs/jobs/chris-q38-rlreward-prod11-fast1"
    assert identity.output_root != "/mnt/sfs/jobs/chris-q38-rlreward-prod11"
    assert identity.data_root != identity.predecessor_data_root


def test_fast2_changes_only_identity_and_reuses_exact_fast_science() -> None:
    fast1 = _load(FAST_RUN)
    fast2 = _load(FAST2_RUN)
    normalized = copy.deepcopy(fast2)
    normalized["name"] = fast1["name"]
    normalized["output_root"] = fast1["output_root"]
    normalized["data"] = copy.deepcopy(fast1["data"])
    normalized["wandb"]["run_id"] = fast1["wandb"]["run_id"]
    assert normalized == fast1

    fast1_data = _load(FAST_DATA)
    fast2_data = _load(FAST2_DATA)
    fast2_data["name"] = fast1_data["name"]
    fast2_data["output"] = fast1_data["output"]
    assert fast2_data == fast1_data


def test_fast2_identity_is_sealed_and_retires_fast1() -> None:
    raw = _load(FAST2_IDENTITY)
    identity = historical.load_identity(FAST2_IDENTITY)
    body = {key: value for key, value in raw.items() if key != "sha256"}

    assert raw["sha256"] == "sha256:" + digest(body)
    assert identity.run_name == "chris-q38-rlreward-prod11-fast2"
    assert identity.predecessor_run_name == "chris-q38-rlreward-prod11-fast1"
    assert identity.output_root == "/mnt/sfs/jobs/chris-q38-rlreward-prod11-fast2"
    assert identity.output_root != "/mnt/sfs/jobs/chris-q38-rlreward-prod11-fast1"
    assert identity.data_root != identity.predecessor_data_root


def test_fast_update_exact_identity_compiles_and_plan_binding_revalidates(monkeypatch) -> None:
    plan = _compile_fast(monkeypatch)

    assert plan["arguments"]["eval_before_train"] is False
    assert plan["native_overrides"]["trainer.eval_before_train"] is False
    assert plan["qualification"]["fast_update"] == {
        "schema": "cyber_rl_reward_canary_fast_update_binding_v1",
        "identity_path": canary.FAST_UPDATE_IDENTITY_PATH,
        "identity_file_sha256": canary.FAST_UPDATE_IDENTITY_FILE_SHA256,
        "identity_self_sha256": canary.FAST_UPDATE_IDENTITY_SELF_SHA256,
        "eval_before_train": False,
        "sha256": plan["qualification"]["fast_update"]["sha256"],
    }
    assert (
        canary.validate_plan_binding(plan["qualification"], plan["data"], plan["arguments"])
        == plan["qualification"]
    )
    skyrl_training.job_request(plan)


def test_fast2_exact_identity_compiles_and_plan_binding_revalidates(monkeypatch) -> None:
    plan = _compile_fast2(monkeypatch)

    assert plan["arguments"]["eval_before_train"] is False
    assert plan["native_overrides"]["trainer.eval_before_train"] is False
    assert plan["qualification"]["fast_update"] == {
        "schema": "cyber_rl_reward_canary_fast_update_binding_v1",
        "identity_path": canary.FAST_UPDATE_2_IDENTITY_PATH,
        "identity_file_sha256": canary.FAST_UPDATE_2_IDENTITY_FILE_SHA256,
        "identity_self_sha256": canary.FAST_UPDATE_2_IDENTITY_SELF_SHA256,
        "eval_before_train": False,
        "sha256": plan["qualification"]["fast_update"]["sha256"],
    }
    assert (
        canary.validate_plan_binding(plan["qualification"], plan["data"], plan["arguments"])
        == plan["qualification"]
    )
    request = skyrl_training.job_request(plan)
    assert request["workers"] == 1
    assert request["gpus_per_worker"] == 8


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("recipe", "eval_before_train"), True),
        (("name",), "chris-q38-rlreward-prod11-fast1"),
        (("output_root",), "/mnt/sfs/jobs/chris-q38-rlreward-prod11-fast1"),
        (("data", "root"), canary.FAST_UPDATE_IDENTITY["data_root"]),
        (("wandb", "run_id"), "chris-q38-rlreward-prod11-fast1"),
    ],
)
def test_fast2_rejects_recipe_or_partial_identity_drift(monkeypatch, path, value) -> None:
    run = _load(FAST2_RUN)
    target = run
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    manifest = copy.deepcopy(_load(BASE_MANIFEST))
    manifest["name"] = run["name"]
    manifest["sha256"] = "sha256:" + digest(
        {key: item for key, item in manifest.items() if key != "sha256"}
    )
    original = sft.read_mapping

    def read(source: Path) -> dict:
        if Path(source) == Path(run["data"]["manifest"]):
            return copy.deepcopy(manifest)
        return original(source)

    monkeypatch.setattr(sft, "read_mapping", read)
    with pytest.raises(ValueError, match="reward-canary"):
        skyrl_training.compile_rl(run, relative_to=FAST2_RUN.parent)


def test_fast2_rejects_deleted_no_pre_eval_and_plan_binding(monkeypatch) -> None:
    run = _load(FAST2_RUN)
    run["recipe"].pop("eval_before_train")
    manifest = copy.deepcopy(_load(BASE_MANIFEST))
    manifest["name"] = run["name"]
    manifest["sha256"] = "sha256:" + digest(
        {key: item for key, item in manifest.items() if key != "sha256"}
    )
    original = sft.read_mapping

    def read(source: Path) -> dict:
        if Path(source) == Path(run["data"]["manifest"]):
            return copy.deepcopy(manifest)
        return original(source)

    monkeypatch.setattr(sft, "read_mapping", read)
    with pytest.raises(ValueError, match="reward-canary recipe changed"):
        skyrl_training.compile_rl(run, relative_to=FAST2_RUN.parent)

    plan = _compile_fast2(monkeypatch)
    plan["qualification"].pop("fast_update")
    plan["qualification"]["sha256"] = "sha256:" + digest(
        {key: value for key, value in plan["qualification"].items() if key != "sha256"}
    )
    with pytest.raises(ValueError, match="plan binding changed"):
        skyrl_training.job_request(plan)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("recipe", "eval_before_train"), True),
        (("recipe", "steps"), 2),
        (("name",), "chris-q38-rlreward-prod11-fast2"),
        (("output_root",), "/mnt/sfs/jobs/chris-q38-rlreward-prod11-fast2"),
        (("data", "root"), "/mnt/sfs/jobs/chris-q38-study-corpora-v1/other/data"),
        (("wandb", "run_id"), "chris-q38-rlreward-prod11-fast2"),
    ],
)
def test_fast_update_rejects_recipe_or_identity_mutation(monkeypatch, path, value) -> None:
    run = _load(FAST_RUN)
    target = run
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    manifest = copy.deepcopy(_load(BASE_MANIFEST))
    manifest["name"] = run["name"]
    manifest["sha256"] = "sha256:" + digest(
        {key: item for key, item in manifest.items() if key != "sha256"}
    )
    original = sft.read_mapping

    def read(source: Path) -> dict:
        if Path(source) == Path(run["data"]["manifest"]):
            return copy.deepcopy(manifest)
        return original(source)

    monkeypatch.setattr(sft, "read_mapping", read)
    with pytest.raises(ValueError, match="reward-canary"):
        skyrl_training.compile_rl(run, relative_to=FAST_RUN.parent)


def test_fast_update_rejects_deleting_explicit_no_pre_eval_mode(monkeypatch) -> None:
    run = _load(FAST_RUN)
    run["recipe"].pop("eval_before_train")
    manifest = copy.deepcopy(_load(BASE_MANIFEST))
    manifest["name"] = run["name"]
    manifest["sha256"] = "sha256:" + digest(
        {key: item for key, item in manifest.items() if key != "sha256"}
    )
    original = sft.read_mapping

    def read(source: Path) -> dict:
        if Path(source) == Path(run["data"]["manifest"]):
            return copy.deepcopy(manifest)
        return original(source)

    monkeypatch.setattr(sft, "read_mapping", read)
    with pytest.raises(ValueError, match="reward-canary recipe changed"):
        skyrl_training.compile_rl(run, relative_to=FAST_RUN.parent)


def test_fast_update_plan_binding_rejects_missing_receipt_or_identity_drift(monkeypatch) -> None:
    plan = _compile_fast(monkeypatch)
    missing = copy.deepcopy(plan)
    missing["qualification"].pop("fast_update")
    missing["qualification"]["sha256"] = "sha256:" + digest(
        {key: value for key, value in missing["qualification"].items() if key != "sha256"}
    )
    with pytest.raises(ValueError, match="plan binding changed"):
        skyrl_training.job_request(missing)

    baseline_shaped = copy.deepcopy(plan)
    baseline_shaped["arguments"].pop("eval_before_train")
    baseline_shaped["qualification"].pop("fast_update")
    baseline_shaped["qualification"]["sha256"] = "sha256:" + digest(
        {key: value for key, value in baseline_shaped["qualification"].items() if key != "sha256"}
    )
    with pytest.raises(ValueError, match="plan binding changed"):
        skyrl_training.job_request(baseline_shaped)

    drifted = copy.deepcopy(plan)
    drifted["arguments"]["wandb_run_id"] = "chris-q38-rlreward-prod11-fast2"
    with pytest.raises(ValueError, match="plan binding changed"):
        skyrl_training.job_request(drifted)
