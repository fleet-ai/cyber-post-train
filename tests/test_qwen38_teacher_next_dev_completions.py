"""Offline contract tests for the two next teacher-SFT completion arms."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

from training.io import digest_json, file_sha256

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "configs/studies/qwen-blackbox-teacher-next-dev-completions-v1.json"
LR30 = ROOT / "configs/qualification/qwen38-teacher-lr30-step6-resume-to76-dev-v1.template.json"
LR100 = ROOT / "configs/qualification/qwen38-teacher-lr100-step21-resume-to76-dev-v1.template.json"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sealed(path: Path) -> dict:
    value = read(path)
    assert value["sha256"] == digest_json(
        {key: item for key, item in value.items() if key != "sha256"}
    )
    return value


def bound(binding: dict) -> dict:
    path = ROOT / binding["path"]
    assert binding["file_sha256"] == file_sha256(path)
    value = read(path)
    if "embedded_sha256" in binding:
        assert value["sha256"] == binding["embedded_sha256"]
    return value


def set_dotted(value: dict, dotted: str, item: object) -> None:
    target = value
    pieces = dotted.split(".")
    for piece in pieces[:-1]:
        target = target[piece]
    target[pieces[-1]] = copy.deepcopy(item)


def materialize(template: dict) -> tuple[dict, dict]:
    source = bound(template["source"]["config"])
    result = copy.deepcopy(source)
    transform = template["materialization"]["source_config_transform"]
    for key in transform["remove"]:
        del result[key]
    for dotted, item in transform["replace"].items():
        set_dotted(result, dotted, item)
    result.update(copy.deepcopy(transform["add"]))
    return source, result


def expected_new_checkpoints(start: int, terminal: int, interval: int) -> list[int]:
    periodic = [step for step in range(interval, terminal + 1, interval) if step > start]
    return sorted(set(periodic + [terminal]))


def test_files_are_self_digesting_inert_and_study_bound() -> None:
    study, lr30, lr100 = sealed(STUDY), sealed(LR30), sealed(LR100)
    assert study["status"] == "offline_prepared_no_live_calls"
    assert study["launchable"] is False
    assert study["execution"] == {
        "kind": "sealed_metadata_only_not_a_job_request",
        "cluster_or_api_calls_performed": 0,
        "wandb_calls_performed": 0,
        "jobs_submitted": 0,
        "production_authorized": False,
    }
    for template in (lr30, lr100):
        assert template["launchable"] is False
        assert template["cluster_or_api_mutations_performed"] is False
        assert template["execution"] == {
            "kind": "offline_template_only_not_a_job_request",
            "kubernetes_calls_performed": 0,
            "jobs_api_preview_calls_performed": 0,
            "jobs_api_submit_calls_performed": 0,
            "wandb_runs_created": 0,
        }

    assert [arm["rank"] for arm in study["ranked_arms"]] == [1, 2]
    assert [arm["arm_id"] for arm in study["ranked_arms"]] == [
        lr30["arm_id"],
        lr100["arm_id"],
    ]
    for arm, path, template in zip(study["ranked_arms"], (LR30, LR100), (lr30, lr100), strict=True):
        assert arm["template"]["file_sha256"] == file_sha256(path)
        assert arm["template"]["embedded_sha256"] == template["sha256"]


def test_shared_model_split_corpus_and_outcome_protocol_are_exact() -> None:
    study, lr30, lr100 = sealed(STUDY), sealed(LR30), sealed(LR100)
    common = study["common_bindings"]

    model = common["model"]
    model_lock = read(ROOT / model["lock_path"])
    assert file_sha256(ROOT / model["lock_path"]) == model["lock_file_sha256"]
    assert (model_lock["repo"], model_lock["revision"]) == (
        model["repository"],
        model["revision"],
    )

    split_binding = common["filtered_split"]
    split = bound(split_binding)
    assert split["sha256"] == split_binding["embedded_sha256"]
    assert split["counts"] == {
        "train": {"groups": 59, "task_versions": 59},
        "dev": {"groups": 20, "task_versions": 20},
        "final_test": {"groups": 10, "task_versions": 10},
    }
    assert split["training_split"]["sha256"] == common["training_corpus"]["training_split_sha256"]
    assert split["inventory_audit"]["full_taxonomy_available"] is True
    assert not split["inventory_audit"]["quarantined_task_versions"]
    groups: dict[str, set[str]] = {name: set() for name in ("train", "dev", "final_test")}
    for task in split["tasks"]:
        groups[task["split"]].add(task["group_id"])
    assert groups["train"].isdisjoint(groups["dev"] | groups["final_test"])
    assert groups["dev"].isdisjoint(groups["final_test"])

    corpus = common["training_corpus"]
    successor = bound(corpus["successor_evidence"])
    stage = bound(corpus["stage_evidence"])
    preflight = bound(corpus["cpu_preflight_evidence"])
    assert successor["successor_manifest_sha256"] == corpus["manifest_sha256"]
    assert successor["protocol_sha256"] == common["selection_protocol"]["embedded_sha256"]
    assert successor["preserved"]["split_sha256"] == corpus["training_split_sha256"]
    assert successor["preserved"]["train_rows"] == corpus["dense_windows"] == 602
    assert successor["preserved"]["supervised_tokens"] == corpus["supervised_tokens"]
    assert stage["destination"]["embedded_sha256"] == corpus["manifest_sha256"]
    assert stage["destination"]["path"] == corpus["manifest"]
    assert stage["independent_readback"]["self_digest_valid"] is True
    preflight_counts = [arm["PREFLIGHT"]["counts"]["train"] for arm in preflight["arms"]]
    assert preflight_counts == [{"rows": 602, "tasks": 29, "supervised_tokens": 700359}] * 3
    assert corpus["represented_train_tasks"] == 29 < split["counts"]["train"]["groups"]
    assert corpus["teacher_source_harness_claimed_by_this_plan"] is False

    protocol_binding = common["selection_protocol"]
    protocol = bound(protocol_binding)
    assert protocol["sha256"] == protocol_binding["embedded_sha256"]
    assert protocol["task_set"]["task_count"] == protocol_binding["task_count"] == 20
    assert protocol["pass_k"] == protocol_binding["pass_k"] == 4
    assert protocol["sampling"]["attempt_seeds"] == protocol_binding["attempt_seeds"]
    assert protocol["metrics"]["primary"]["name"] == protocol_binding["primary_metric"]
    assert protocol["harness"] == {
        "harness": "opencode",
        "harness_version": "1.18.27",
        "release_asset_sha256": (
            "sha256:4af5494f9433f59db8c1e344198f0ee72a50c06ec009fb4a8aeab4c2d4abd702"
        ),
        "provider_adapter": "@ai-sdk/openai-compatible",
        "context_management": "opencode_1.18.27_native_compaction_autocontinue_v2",
        "context_window_size": 262144,
        "compaction_headroom_tokens": 20000,
        "max_output_tokens": 32768,
        "max_model_requests": 600,
        "timeout_seconds": 28800,
        "tools": ["bash", "submit_report"],
        "tool_catalog_sha256": (
            "sha256:85fad6bdc3a835bf52a11a99b3387740eb06eb3d1720ad9bb33f3feac215b44a"
        ),
    }
    assert protocol["sampling"]["temperature"] == 0.6
    assert protocol["sampling"]["top_p"] == 0.95

    for template in (lr30, lr100):
        frozen = template["frozen_scientific_contract"]
        assert frozen["model"] == {
            "repository": model["repository"],
            "revision": model["revision"],
            "lock_path": model["lock_path"],
            "lock_file_sha256": model["lock_file_sha256"],
        }
        assert frozen["split"]["embedded_sha256"] == split["sha256"]
        assert frozen["data"]["manifest_sha256"] == corpus["manifest_sha256"]
        assert frozen["data"]["train_file_sha256"] == corpus["train_file_sha256"]
        assert frozen["data"]["represented_train_tasks"] == corpus["represented_train_tasks"]
        assert template["selection_evaluation"]["protocol_sha256"] == protocol["sha256"]
        assert template["selection_evaluation"]["harness"] == "opencode"
        assert template["selection_evaluation"]["harness_version"] == "1.18.27"


def test_lr30_is_exact_continuation_of_accepted_step6_not_a_duplicate() -> None:
    template = sealed(LR30)
    evidence = bound(template["source"]["terminal_evidence"])
    source, materialized = materialize(template)
    transform = template["materialization"]["source_config_transform"]

    assert evidence["source"]["optimizer_step"] == template["source"]["optimizer_step"] == 6
    assert evidence["source"]["producer_state"] == "planned_pause_at_step_6_of_76"
    assert evidence["acceptance"]["zero_update_reload_accepted"] is True
    assert (
        evidence["independent_post_reload_verification"]["source_checkpoint_unchanged_after_reload"]
        is True
    )
    assert (
        evidence["checkpoint_seal"]["manifest_file_sha256"]
        == template["source"]["checkpoint_manifest_file_sha256"]
    )
    assert transform["add"]["recovery"] == {
        "manifest": template["source"]["checkpoint_manifest"],
        "sha256": template["source"]["checkpoint_manifest_file_sha256"].removeprefix("sha256:"),
        "mode": "resume",
    }
    assert "pause_after_step" not in materialized
    assert materialized["recovery"]["mode"] == "resume"
    assert materialized["name"] != source["name"]
    assert materialized["output_root"] != source["output_root"]
    assert materialized["wandb"]["run_id"] != source["wandb"]["run_id"]
    assert materialized["recipe"] == source["recipe"]
    assert materialized["data"] == source["data"]
    assert materialized["model"] == source["model"]

    recipe = template["frozen_scientific_contract"]["recipe"]
    assert recipe["total_optimizer_steps"] == math.ceil(
        template["frozen_scientific_contract"]["data"]["dense_windows"]
        / recipe["global_batch_size"]
    )
    assert recipe["remaining_optimizer_steps"] == 76 - 6 == 70
    checkpoints = template["checkpoint_and_reload_acceptance"]
    expected = expected_new_checkpoints(6, 76, recipe["checkpoint_interval"])
    assert checkpoints["expected_new_checkpoint_steps"] == expected
    assert checkpoints["expected_retained_new_checkpoint_steps"] == expected[-3:]
    assert template["wandb_acceptance"]["resume_expected_scalar_events"] == 70
    assert template["wandb_acceptance"]["resume_expected_global_step_range"] == [7, 76]


def test_lr100_remains_blocked_until_pair_proof_and_zero_update_reload() -> None:
    template = sealed(LR100)
    evidence = bound(template["source"]["terminal_evidence"])
    source, materialized = materialize(template)

    assert template["status"] == (
        "blocked_waiting_checkpoint_pair_and_zero_update_reload_acceptance"
    )
    assert evidence["bounded_training"]["observed_optimizer_steps"] == 21
    assert evidence["bounded_training"]["all_observed_losses_finite"] is True
    assert evidence["bounded_training"]["all_observed_gradient_norms_finite"] is True
    assert evidence["acceptance"]["independent_checkpoint_pair_verification_accepted"] is False
    assert evidence["acceptance"]["zero_optimizer_reload_accepted"] is False
    assert template["source"]["checkpoint_manifest_file_sha256"] is None
    assert materialized["recovery"] == {
        "manifest": template["source"]["checkpoint_manifest"],
        "sha256": None,
        "mode": "resume",
    }
    assert "source.checkpoint_manifest_file_sha256" in template["unresolved_live_bindings"]
    assert (
        "materialization.source_config_transform.add.recovery.sha256"
        in template["unresolved_live_bindings"]
    )
    pair = template["source"]["checkpoint_pair_verification_template"]
    assert pair["file_sha256"] == file_sha256(ROOT / pair["path"])
    reload = template["source"]["zero_update_reload_template"]
    assert reload["file_sha256"] == file_sha256(ROOT / reload["path"])
    assert read(ROOT / reload["path"])["sha256"] == reload["embedded_sha256"]
    assert materialized["recipe"] == source["recipe"]
    assert materialized["data"] == source["data"]
    assert materialized["model"] == source["model"]

    recipe = template["frozen_scientific_contract"]["recipe"]
    assert recipe["remaining_optimizer_steps"] == 76 - 21 == 55
    checkpoints = template["checkpoint_and_reload_acceptance"]
    expected = expected_new_checkpoints(21, 76, recipe["checkpoint_interval"])
    assert checkpoints["expected_new_checkpoint_steps"] == expected == [40, 60, 76]
    assert checkpoints["expected_retained_new_checkpoint_steps"] == expected
    assert template["wandb_acceptance"]["resume_expected_scalar_events"] == 55
    assert template["wandb_acceptance"]["resume_expected_global_step_range"] == [22, 76]


def test_two_arms_are_wide_lr_contrast_with_other_science_fixed() -> None:
    study, lr30, lr100 = sealed(STUDY), sealed(LR30), sealed(LR100)
    left = copy.deepcopy(lr30["frozen_scientific_contract"])
    right = copy.deepcopy(lr100["frozen_scientific_contract"])
    for value in (left["recipe"], right["recipe"]):
        for operational_or_progress_field in (
            "nodes",
            "gpus_per_node",
            "lr",
            "checkpoint_interval",
            "source_optimizer_steps",
            "remaining_optimizer_steps",
        ):
            value.pop(operational_or_progress_field)
    assert left == right
    assert (
        lr100["frozen_scientific_contract"]["recipe"]["lr"]
        / lr30["frozen_scientific_contract"]["recipe"]["lr"]
        == study["decision"]["learning_rate_ratio_high_over_low"]
    )
    assert study["decision"]["only_factor_varied_between_arms"] == "learning rate"
    assert study["decision"]["fresh_base_restarts_prepared"] is False
    assert study["decision"]["source_optimizer_steps_replayed"] is False

    for template in (lr30, lr100):
        admission = template["admission"]
        assert admission["cluster"] == "dev"
        assert admission["priority_class"] == "c1"
        assert admission["expected_queue_priority"] == "q1"
        assert admission["expected_priority_value"] == 10_000
        assert admission["requeue_if_preempted"] is False
        assert admission["production_submission_authorized"] is False
        assert admission["requested_nodes"] <= 8


def test_selection_firewall_never_uses_external_benchmark_or_training_loss() -> None:
    study = sealed(STUDY)
    firewall = study["selection_firewall"]
    assert firewall["training_loss"] == {
        "record_in_wandb": True,
        "selection_eligible": False,
        "role": "fitting diagnostic only",
    }
    assert firewall["fleet_dev"]["selection_eligible"] is True
    assert firewall["fleet_dev"]["only_metric"] == ("fleet_dev_paired_mean_success_delta_4fixed")
    assert firewall["fleet_final"]["selection_eligible"] is False
    assert (
        firewall["webexploitbench"][
            "hyperparameter_checkpoint_stopping_or_tiebreak_selection_eligible"
        ]
        is False
    )
    assert firewall["webexploitbench"]["state"] == (
        "sealed_until_the_fleet_dev_selection_decision_is_frozen"
    )
    for template_path in (LR30, LR100):
        evaluation = sealed(template_path)["selection_evaluation"]
        assert evaluation["primary_metric"] == firewall["fleet_dev"]["only_metric"]
        assert evaluation["training_loss_selection_eligible"] is False
        assert evaluation["external_benchmark_selection_eligible"] is False
