from __future__ import annotations

import json
from pathlib import Path

from training.io import digest_json, file_sha256

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "configs/qualification/qwen38-teacher-lr-extremes-dev-v1.template.json"


def _read() -> dict:
    value = json.loads(TEMPLATE.read_text())
    assert value["sha256"] == digest_json(
        {key: item for key, item in value.items() if key != "sha256"}
    )
    return value


def test_numeric_lr_canaries_are_distinct_bounded_and_inert() -> None:
    value = _read()
    assert value["launchable"] is False
    assert value["cluster_or_api_mutations_performed"] is False
    assert value["cluster"] == "dev"
    assert value["status"] == "blocked_waiting_enhanced_metrics_reload_and_runtime_preflight"

    execution = value["immutable_inputs"]["execution"]
    assert execution["workers"] == 1 and execution["gpus_per_worker"] == 4
    assert execution["global_batch_size"] == 8
    assert execution["microbatch_per_gpu"] == 1
    assert execution["gradient_accumulation_steps"] == 2
    assert execution["pause_after_optimizer_step"] == 2
    assert execution["checkpoint_interval"] == 1
    assert execution["priority_class"] == "c1"
    assert execution["expected_queue_priority"] == "q1"
    assert execution["expected_priority_value"] == 10_000
    assert execution["requeueIfPreempted"] is False

    arms = value["arms"]
    assert [arm["lr"] for arm in arms] == [1e-6, 3e-5]
    for field in ("run_name", "output_root", "wandb_run_id", "wandb_name", "prepared_dir"):
        assert len({arm[field] for arm in arms}) == 2


def test_numeric_canaries_bind_current_runtime_and_narrow_rejection_policy() -> None:
    value = _read()
    runtime = value["runtime"]
    assert runtime["sha256"] == file_sha256(ROOT / runtime["path"])
    assert runtime["implementation_commit"] == (
        "699fa577e96180695dad00d85b152f5302964699"
    )
    assert runtime["numeric_rejection"] == {
        "schema": "cyber_sft_numeric_rejection_policy_v1",
        "reason_codes": ["nonfinite_loss", "nonfinite_gradient_norm"],
        "latest_attempted_optimizer_step": 2,
    }

    contract = value["terminal_contract"]
    assert contract["finite_two_step_outcome"]["receipt"] == "TRAINING_PAUSED.json"
    rejected = contract["predeclared_numeric_rejection"]
    assert rejected["receipt"] == "TRAINING_REJECTED.json"
    assert rejected["training_artifact_accepted"] is False
    assert rejected["opens_next_gate"] is False
    unexpected = contract["unexpected_defect"]
    assert unexpected["controller_exit"] == 1
    assert unexpected["receipt"] == "FAILED.json"
    assert unexpected["may_be_reclassified_as_scientific_rejection"] is False


def test_numeric_canaries_require_exact_upstream_and_per_arm_bindings() -> None:
    value = _read()
    unresolved = value["unresolved_bindings"]
    assert len(unresolved) == len(set(unresolved)) == 17
    for path in unresolved:
        current = value
        for part in path.split("."):
            current = current[int(part)] if isinstance(current, list) else current[part]
        assert current in (None, False)

    assert "one_create_once_POST_per_arm_after_review" in value["pre_submit_gates"]
    assert "aggregate_active_study_node_count_rechecked_below_eight" in value[
        "pre_submit_gates"
    ]
