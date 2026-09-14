from __future__ import annotations

import json
from pathlib import Path

from training.io import digest_json

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "configs/qualification/qwen38-teacher-scheduler-reload-dev-v2.template.json"


def _read() -> dict:
    value = json.loads(TEMPLATE.read_text())
    assert value["sha256"] == digest_json(
        {key: item for key, item in value.items() if key != "sha256"}
    )
    return value


def _resolve(value: object, path: str) -> object:
    current = value
    for part in path.split("."):
        current = current[int(part)] if isinstance(current, list) else current[part]
    return current


def test_scheduler_reload_pair_is_inert_until_both_sources_and_seals_are_bound() -> None:
    value = _read()
    assert value["launchable"] is False
    assert value["cluster_or_api_mutations_performed"] is False
    assert value["status"] == "blocked_waiting_both_sources_terminal_and_checkpoint_seals"
    unresolved = value["unresolved_bindings"]
    assert len(unresolved) == len(set(unresolved)) == 36
    assert all(_resolve(value, path) in (None, False) for path in unresolved)


def test_scheduler_reload_sources_are_the_exact_step_six_pair() -> None:
    value = _read()
    constant, cosine = value["arms"]
    assert [arm["source"]["api_name"] for arm in value["arms"]] == [
        "chris-q38-ta4-sched0-dev2-bb4aa2a5",
        "chris-q38-ta4-cos5-dev2-9af1c840",
    ]
    assert [
        (arm["source"]["scheduler"], arm["source"]["warmup_ratio"]) for arm in value["arms"]
    ] == [("constant_with_warmup", 0.0), ("cosine", 0.05)]
    assert [arm["source"]["expected_num_warmup_steps"] for arm in value["arms"]] == [
        0,
        4,
    ]
    assert value["common"]["source_optimizer_step"] == 6
    assert value["common"]["source_world_size"] == 4
    assert constant["source"]["output_root"] != cosine["source"]["output_root"]


def test_each_cpu_seal_binds_only_its_own_step_six_checkpoint() -> None:
    value = _read()
    for arm in value["arms"]:
        source, seal = arm["source"], arm["checkpoint_seal"]
        assert seal["operation"] == "cpu_only_create_once"
        assert seal["world_size"] == 4
        assert seal["gpu_reload_verified"] is False
        assert seal["source_checkpoint_path"] == (
            source["output_root"] + "/checkpoints/global_step_6"
        )
        assert seal["source_checkpoint_receipt_path"] == (
            source["output_root"] + "/checkpoint_receipts/step-000006.json"
        )
        assert seal["command"][2:5] == [
            "cyber-post-train",
            "checkpoint-seal",
            source["prepared_dir"],
        ]
        assert seal["command"][5] == "6"
        assert seal["command"][-1] == seal["manifest_path"]
    assert len({arm["checkpoint_seal"]["manifest_path"] for arm in value["arms"]}) == 2


def test_reload_is_exact_four_rank_zero_update_validation() -> None:
    value = _read()
    recovery = value["common"]["recovery"]
    assert recovery["mode"] == "validate"
    assert recovery["optimizer_steps_authorized"] == 0
    assert recovery["validation_scope"] == "checkpoint_and_sampler_reload_only_no_ce"
    assert recovery["resources"] == {
        "workers": 1,
        "gpus_per_worker": 4,
        "total_gpus": 4,
        "priority_class": "c1",
        "expected_queue_priority": "q1",
        "expected_priority_value": 10000,
        "requeueIfPreempted": False,
    }
    transform = recovery["source_plan_transform"]
    assert transform["remove"] == ["pause_after_step"]
    assert "recipe_including_scheduler_warmup_and_full_horizon" in transform["preserve_exactly"]
    assert recovery["tracking"]["training_loss_lr_gradient_and_supervised_token_events"] == 0


def test_pair_acceptance_requires_exact_scheduler_state_and_source_stability() -> None:
    value = _read()
    acceptance = set(value["acceptance_per_arm"])
    assert {
        "optimizer_steps_executed_equals_zero",
        "optimizer_step_equals_six",
        "each_rank_scheduler_state_exactly_matches_its_checkpoint_extra_state",
        "all_four_ranks_restore_exact_rng_state",
        "sampler_cursor_restored_exactly",
        "source_checkpoint_inventory_sizes_and_hashes_unchanged_after_reload",
        "reload_output_contains_no_checkpoint_or_training_receipt",
        "wandb_contains_no_new_loss_lr_gradient_or_supervised_token_training_events",
        "gpu_release_verified",
    } <= acceptance
    assert value["pair_completion"] == [
        "both_arms_independently_meet_every_acceptance_gate",
        "constant_checkpoint_is_never_cross_bound_to_cosine_reload",
        "cosine_checkpoint_is_never_cross_bound_to_constant_reload",
    ]
