"""Offline tests for the sealed Qwen Fleet final-test outcome protocol."""

import copy
import json
from pathlib import Path

import pytest

from evals.fleet import dev_outcome_protocol as protocol
from evals.fleet import evaluate
from training.io import digest_json

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs" / "evaluation"
DATA = ROOT / "configs" / "data"
TASK_SET_PATH = CONFIG / "qwen38-blackbox-fleet-final-task-set-v1.json"
PROTOCOL_PATH = CONFIG / "qwen38-blackbox-fleet-final-outcome-protocol-v1.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def test_frozen_final_set_matches_common_lock_and_exact_bindings():
    lock = load(DATA / "qwen-blackbox-study-final-test-v1.json")
    task_set = load(TASK_SET_PATH)
    frozen = load(PROTOCOL_PATH)
    protocol.validate_final_protocol(frozen, task_set)

    expected = {(row["task_key"], row["task_version_id"], row["group_id"]) for row in lock["tasks"]}
    actual = {
        (row["task_key"], row["task_version_id"], row["group_id"]) for row in task_set["tasks"]
    }
    assert actual == expected
    assert task_set["final_test_lock_sha256"] == lock["sha256"]
    assert task_set["task_count"] == 10
    assert all(
        row["exact_binding_sha256"]
        == digest_json(
            [
                row["exact_binding"]["task"],
                row["exact_binding"]["environment"],
                row["exact_binding"]["verifier"],
            ]
        )
        for row in task_set["tasks"]
    )


def test_final_protocol_is_paired_sealed_and_nonlaunchable():
    task_set = load(TASK_SET_PATH)
    frozen = load(PROTOCOL_PATH)
    protocol.validate_final_protocol(frozen, task_set)

    template = frozen["checkpoint_binding_template"]
    assert template["launchable"] is False
    assert set(template["arms"]) == {"base", "post_sft"}
    for arm in template["arms"].values():
        assert set(arm) == set(protocol.UNBOUND_CHECKPOINT_FIELDS)
        assert all(value is None for value in arm.values())
    assert frozen["model"]["only_intended_difference"] == "post-SFT weight manifest"
    assert frozen["harness"]["harness_version"] == "1.18.27"
    assert frozen["harness"]["tools"] == ["bash", "submit_report"]
    assert frozen["sampling"]["attempt_seeds"] == [42, 43, 44, 45]
    assert frozen["pass_k"] == 4
    assert frozen["paired_execution"]["same_tasks_and_attempt_seeds"] is True

    sealed = frozen["sealed_use_policy"]
    assert sealed["state"] == "sealed_during_hpo"
    assert sealed["training_data_eligible"] is False
    assert sealed["preference_or_reward_data_eligible"] is False
    assert sealed["hyperparameter_selection_eligible"] is False
    assert sealed["checkpoint_selection_eligible"] is False
    assert sealed["failure_analysis_during_hpo"] == "forbidden"


def test_final_lock_is_shared_by_both_study_splits():
    lock = load(DATA / "qwen-blackbox-study-final-test-v1.json")
    expected_versions = {row["task_version_id"] for row in lock["tasks"]}
    for variant in ("a", "b"):
        split = load(DATA / f"qwen-blackbox-study-split-{variant}-v1.json")
        selected_versions = {
            row["task_version_id"] for row in split["tasks"] if row["split"] == "final_test"
        }
        assert split["final_test_lock_sha256"] == lock["sha256"]
        assert selected_versions == expected_versions


def test_final_task_set_compiles_through_real_fleet_evaluator():
    task_set = load(TASK_SET_PATH)
    versions = [row["task_version_id"] for row in task_set["tasks"]]
    config = {
        "name": "synthetic-sealed-final",
        "task_set": TASK_SET_PATH.relative_to(ROOT).as_posix(),
        "models": {
            "qwen3.8-27b": {
                "repository": protocol.MODEL_REPOSITORY,
                "revision": protocol.MODEL_REVISION,
                "session_model": "synthetic-unlaunched-checkpoint",
            }
        },
        "routes": {
            "synthetic": {
                "model": "qwen3.8-27b",
                "served_id": "synthetic-unlaunched-checkpoint",
                "task_versions": versions,
                "endpoint_origin": "https://inference.flt.build",
                "catalog": {"engine": "sglang", "precision": "bf16", "tensor_parallel_size": 8},
                "model_info": {
                    "model_path": "/exact/unlaunched/checkpoint",
                    "model_type": "qwen3",
                    "architectures": ["Qwen3ForCausalLM"],
                },
                "server_info": {
                    "model_path": "/exact/unlaunched/checkpoint",
                    "context_length": 262144,
                    "tp_size": 8,
                    "quantization": None,
                    "kv_cache_dtype": "fp8_e4m3",
                    "reasoning_parser": "qwen3",
                    "tool_call_parser": "qwen3_coder",
                },
            }
        },
        "images": {
            "agent": "registry/agent@sha256:" + "a" * 64,
            "proxy": "registry/proxy@sha256:" + "b" * 64,
        },
        "harness": copy.deepcopy(protocol.HARNESS),
        "sampling": {"temperature": 0.6, "top_p": 0.95, "seed": 42},
        "pass_k": 4,
        "concurrency": 1,
        "training_data_eligible": False,
    }
    compiled = evaluate.compile_eval(config, relative_to=ROOT)
    assert len(compiled["tasks"]) == 10
    assert len(evaluate.plan_rows(compiled)) == 40
    assert compiled["training_data_eligible"] is False


def test_final_membership_and_sealed_policy_tamper_are_rejected():
    task_set = load(TASK_SET_PATH)
    frozen = load(PROTOCOL_PATH)

    tampered_tasks = copy.deepcopy(task_set)
    tampered_tasks["tasks"][0]["data_version"] = "different"
    tampered_tasks["sha256"] = digest_json(
        {key: value for key, value in tampered_tasks.items() if key != "sha256"}
    )
    with pytest.raises(ValueError, match="flat evaluator tuple"):
        protocol.validate_final_task_set(tampered_tasks)

    tampered_protocol = copy.deepcopy(frozen)
    tampered_protocol["sealed_use_policy"]["checkpoint_selection_eligible"] = True
    tampered_protocol["sha256"] = digest_json(
        {key: value for key, value in tampered_protocol.items() if key != "sha256"}
    )
    with pytest.raises(ValueError, match="sealed from training and selection"):
        protocol.validate_final_protocol(tampered_protocol, task_set)
