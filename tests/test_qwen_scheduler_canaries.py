"""The scheduler qualification template is inert, bounded and self-digested."""

from __future__ import annotations

import json
from pathlib import Path

from training.io import digest_json, file_sha256

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "configs/qualification/qwen38-teacher-scheduler-dev-v1.template.json"


def read() -> dict:
    value = json.loads(TEMPLATE.read_text())
    assert value["sha256"] == digest_json(
        {key: item for key, item in value.items() if key != "sha256"}
    )
    return value


def test_scheduler_canaries_are_bounded_dev_only_and_bind_current_runtime() -> None:
    value = read()
    assert value["launchable"] is False
    assert value["cluster_or_api_mutations_performed"] is False
    assert value["cluster"] == "dev"
    assert value["runtime"]["sha256"] == file_sha256(ROOT / value["runtime"]["path"])
    assert value["historical_two_step_canaries"]["classification"] == (
        "numeric_stability_only_not_scheduler_qualification_hpo_or_model_quality"
    )

    execution = value["immutable_inputs"]["execution"]
    assert (execution["workers"], execution["gpus_per_worker"]) == (1, 4)
    assert execution["global_batch_size"] == 8
    assert execution["gradient_accumulation_steps"] == 2
    assert execution["full_recipe_max_steps"] == 76
    assert execution["pause_after_optimizer_step"] == 6
    assert execution["priority_class"] == "c1"
    assert execution["expected_priority_value"] == 10_000


def test_scheduler_canaries_cross_the_exact_warmup_boundary_without_claiming_hpo() -> None:
    value = read()
    arms = value["arms"]
    assert [
        (arm["scheduler"], arm["warmup_ratio"], arm["expected_num_warmup_steps"]) for arm in arms
    ] == [("constant_with_warmup", 0.0, 0), ("cosine", 0.05, 4)]
    for field in ("run_name", "output_root", "wandb_run_id", "prepared_dir"):
        assert len({arm[field] for arm in arms}) == 2
    assert value["acceptance"]["purpose"] == (
        "scheduler_runtime_qualification_only_not_hyperparameter_selection"
    )
    assert "scheduler_quality_ranking" in value["acceptance"]["forbidden_interpretations"]


def test_scheduler_canaries_require_unresolved_preflight_and_per_arm_bindings() -> None:
    value = read()
    unresolved = value["unresolved_bindings"]
    assert len(unresolved) == len(set(unresolved)) == 7
    for path in unresolved:
        current = value
        for part in path.split("."):
            current = current[int(part)] if isinstance(current, list) else current[part]
        assert current is None
    assert "one_create_once_POST_per_arm_after_review" in value["pre_submit_gates"]
