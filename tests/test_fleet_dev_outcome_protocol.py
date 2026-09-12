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


@pytest.mark.parametrize("variant", ["a", "b"])
def test_v2_protocol_reuses_exact_split_and_all_four_attempts(variant):
    task_set = load(CONFIG / f"qwen38-blackbox-fleet-dev-{variant}-task-set-v1.json")
    v1 = load(CONFIG / f"qwen38-blackbox-fleet-dev-{variant}-outcome-protocol-v1.json")
    v2 = load(CONFIG / f"qwen38-blackbox-fleet-dev-{variant}-outcome-protocol-v2.json")
    control = load(CONFIG / f"qwen38-blackbox-fleet-dev-{variant}-base-control-v2.json")

    protocol.validate_protocol(v1, task_set)
    protocol.validate_protocol(v2, task_set)
    protocol.validate_base_control(control, v2, task_set)
    task_path = f"configs/evaluation/qwen38-blackbox-fleet-dev-{variant}-task-set-v1.json"
    v1_protocol_path = (
        f"configs/evaluation/qwen38-blackbox-fleet-dev-{variant}-outcome-protocol-v1.json"
    )
    protocol_path = (
        f"configs/evaluation/qwen38-blackbox-fleet-dev-{variant}-outcome-protocol-v2.json"
    )
    assert (
        protocol.build_protocol(
            task_set,
            variant=variant,
            task_set_path=task_path,
            schema=protocol.PROTOCOL_SCHEMA_V1,
        )
        == v1
    )
    assert protocol.build_protocol(task_set, variant=variant, task_set_path=task_path) == v2
    assert protocol.build_base_control(v1, task_set, protocol_path=v1_protocol_path) == load(
        CONFIG / f"qwen38-blackbox-fleet-dev-{variant}-base-control-v1.json"
    )
    assert protocol.build_base_control(v2, task_set, protocol_path=protocol_path) == control
    assert v1["sha256"] != v2["sha256"]
    assert v1["task_set"] == v2["task_set"]
    assert v2["schema"] == protocol.PROTOCOL_SCHEMA_V2
    assert v2["metrics"]["primary"] == {
        "name": "fleet_dev_paired_mean_success_delta_4fixed",
        "unit": "task_family",
        "attempt_seeds": [42, 43, 44, 45],
        "per_task_estimator": (
            "For each exact task version, average the four full-task solve indicators "
            "for the predeclared attempt seeds."
        ),
        "arm_estimator": "Mean the 20 per-task means with equal task weight.",
        "paired_comparison": (
            "Candidate minus base on each exact task-version and attempt-seed pair, "
            "then mean within task and across tasks."
        ),
        "hpo_direction": "maximize candidate-minus-base paired delta",
        "eligibility": (
            "All 80 task-seed pairs must contain authoritative valid outcomes for both base "
            "and candidate; no available fixed attempt may be discarded."
        ),
    }
    assert v2["metrics"]["uncertainty"]["cluster_unit"] == "exact task_version_id"
    assert "fleet_dev_pass_at_4" in v2["metrics"]["secondary_only"][0]
    assert v2["heldout_policy"]["webexploitbench"] == {
        "state": "sealed_during_fleet_dev_hpo",
        "hyperparameter_or_checkpoint_selection_eligible": False,
        "unseal_only_after": "the Fleet development selection decision is frozen",
    }
    assert control["pairing_contract"]["matched_task_seed_pairs"] == 80
    assert control["pairing_contract"]["all_predeclared_attempts_required"] is True


@pytest.mark.parametrize("variant", ["a", "b"])
def test_base_control_is_exact_outcome_only_and_nonlaunchable(variant):
    task_set = load(CONFIG / f"qwen38-blackbox-fleet-dev-{variant}-task-set-v1.json")
    parent = load(CONFIG / f"qwen38-blackbox-fleet-dev-{variant}-outcome-protocol-v1.json")
    control = load(CONFIG / f"qwen38-blackbox-fleet-dev-{variant}-base-control-v1.json")

    protocol.validate_base_control(control, parent, task_set)
    assert control["task_set"]["task_version_ids"] == sorted(
        row["task_version_id"] for row in task_set["tasks"]
    )
    assert control["model"]["revision"] == protocol.MODEL_REVISION
    assert control["frozen_treatment"]["harness_sha256"] == digest_json(parent["harness"])
    assert control["frozen_treatment"]["sampling_sha256"] == digest_json(parent["sampling"])
    assert control["frozen_treatment"]["attempt_seeds"] == [42, 43, 44, 45]
    assert control["pairing_contract"]["scientific_difference_allowed"] == [
        "weights_manifest_sha256"
    ]
    assert control["serving_binding"]["launchable"] is False
    assert set(control["serving_binding"]["fields"]) == set(protocol.UNBOUND_BASE_SERVING_FIELDS)
    assert all(value is None for value in control["serving_binding"]["fields"].values())
    assert control["result_policy"]["outcome_only"] is True
    assert control["result_policy"]["wandb_evaluation_metrics_or_scores"] == "forbidden"
    assert control["result_policy"]["teacher_reference_cross_entropy"] == "forbidden"


def test_base_control_rejects_route_parity_drift_even_when_resealed():
    task_set = load(CONFIG / "qwen38-blackbox-fleet-dev-a-task-set-v1.json")
    parent = load(CONFIG / "qwen38-blackbox-fleet-dev-a-outcome-protocol-v1.json")
    control = load(CONFIG / "qwen38-blackbox-fleet-dev-a-base-control-v1.json")
    tampered = copy.deepcopy(control)
    tampered["pairing_contract"]["same_runtime_fields"].remove("serving_image_digest")
    tampered["sha256"] = digest_json(
        {key: value for key, value in tampered.items() if key != "sha256"}
    )
    with pytest.raises(ValueError, match="differs from its frozen parent"):
        protocol.validate_base_control(tampered, parent, task_set)


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


def test_v2_estimator_contract_tamper_is_rejected_even_when_resealed():
    task_set = load(CONFIG / "qwen38-blackbox-fleet-dev-a-task-set-v1.json")
    frozen = load(CONFIG / "qwen38-blackbox-fleet-dev-a-outcome-protocol-v2.json")
    tampered = copy.deepcopy(frozen)
    tampered["metrics"]["primary"]["attempt_seeds"] = [42]
    tampered["sha256"] = digest_json(
        {key: value for key, value in tampered.items() if key != "sha256"}
    )
    with pytest.raises(ValueError, match="estimator or selection rule drifted"):
        protocol.validate_protocol(tampered, task_set)

    tampered = copy.deepcopy(frozen)
    tampered["heldout_policy"]["webexploitbench"][
        "hyperparameter_or_checkpoint_selection_eligible"
    ] = True
    tampered["sha256"] = digest_json(
        {key: value for key, value in tampered.items() if key != "sha256"}
    )
    with pytest.raises(ValueError, match="WebExploitBench"):
        protocol.validate_protocol(tampered, task_set)
