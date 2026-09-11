"""Offline tests for the exact Qwen Fleet development outcome protocols."""

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


def load(path: Path) -> dict:
    return json.loads(path.read_text())


@pytest.mark.parametrize("variant", ["a", "b"])
def test_frozen_protocol_and_task_set_are_exact_and_nonlaunchable(variant):
    task_set = load(CONFIG / f"qwen38-blackbox-fleet-dev-{variant}-task-set-v1.json")
    frozen = load(CONFIG / f"qwen38-blackbox-fleet-dev-{variant}-outcome-protocol-v1.json")
    protocol.validate_protocol(frozen, task_set)
    assert frozen["checkpoint_binding_template"]["launchable"] is False
    assert set(frozen["checkpoint_binding_template"]["fields"]) == set(
        protocol.UNBOUND_CHECKPOINT_FIELDS
    )
    assert all(value is None for value in frozen["checkpoint_binding_template"]["fields"].values())
    assert frozen["harness"]["tools"] == ["bash", "submit_report"]
    assert frozen["harness"]["context_management"].endswith("native_compaction_autocontinue_v2")
    assert frozen["sampling"]["attempt_seeds"] == [42, 43, 44, 45]
    assert frozen["pass_k"] == 4
    assert frozen["metrics"]["primary"]["name"] == "fleet_dev_pass_at_1"
    assert frozen["heldout_policy"]["teacher_reference_cross_entropy"] == "forbidden"
    assert task_set["task_count"] == 20
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


def test_split_variants_match_frozen_dev_membership_and_preserve_final_test():
    task_sets = {
        variant: load(CONFIG / f"qwen38-blackbox-fleet-dev-{variant}-task-set-v1.json")
        for variant in ("a", "b")
    }
    splits = {
        variant: load(DATA / f"qwen-blackbox-study-split-{variant}-v1.json")
        for variant in ("a", "b")
    }
    for variant in ("a", "b"):
        expected = {
            row["task_version_id"] for row in splits[variant]["tasks"] if row["split"] == "dev"
        }
        actual = {row["task_version_id"] for row in task_sets[variant]["tasks"]}
        assert actual == expected
        assert task_sets[variant]["study_split_sha256"] == splits[variant]["sha256"]
        assert (
            task_sets[variant]["final_test_lock_sha256"]
            == splits[variant]["final_test_lock_sha256"]
        )
    a = {row["task_version_id"] for row in task_sets["a"]["tasks"]}
    b = {row["task_version_id"] for row in task_sets["b"]["tasks"]}
    assert len(a & b) == 5


@pytest.mark.parametrize("variant", ["a", "b"])
def test_task_set_compiles_through_real_fleet_evaluator(variant):
    task_path = CONFIG / f"qwen38-blackbox-fleet-dev-{variant}-task-set-v1.json"
    task_set = load(task_path)
    versions = [row["task_version_id"] for row in task_set["tasks"]]
    config = {
        "name": f"synthetic-dev-{variant}",
        "task_set": task_path.relative_to(ROOT).as_posix(),
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
    assert len(compiled["tasks"]) == 20
    assert len(evaluate.plan_rows(compiled)) == 80
    assert compiled["training_data_eligible"] is False


def test_binding_or_checkpoint_template_tamper_is_rejected():
    task_set = load(CONFIG / "qwen38-blackbox-fleet-dev-a-task-set-v1.json")
    frozen = load(CONFIG / "qwen38-blackbox-fleet-dev-a-outcome-protocol-v1.json")
    tampered_tasks = copy.deepcopy(task_set)
    tampered_tasks["tasks"][0]["env_version"] = "different"
    tampered_tasks["sha256"] = digest_json(
        {key: value for key, value in tampered_tasks.items() if key != "sha256"}
    )
    with pytest.raises(ValueError, match="flat evaluator tuple"):
        protocol.validate_task_set(tampered_tasks)

    tampered_protocol = copy.deepcopy(frozen)
    tampered_protocol["checkpoint_binding_template"]["fields"]["served_model_id"] = "mutable"
    tampered_protocol["sha256"] = digest_json(
        {key: value for key, value in tampered_protocol.items() if key != "sha256"}
    )
    with pytest.raises(ValueError, match="checkpoint placeholder"):
        protocol.validate_protocol(tampered_protocol, task_set)
