import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
INDEX = ROOT / "configs/discovery/qwen38-lora-evidence-index-v1.json"


def test_qwen38_lora_evidence_index_is_internally_consistent():
    value = json.loads(INDEX.read_text())

    assert value["schema"] == "cyber_qwen38_lora_evidence_index_v1"
    assert value["model"] == {
        "id": "Qwen/Qwen3.8-27B",
        "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    }
    accepted = [x for x in value["checkpoints"] if x["status"] == "accepted_checkpoint"]
    sft_names = [name for item in value["sft_attempts"] for name in item.get("run_names", [])]
    rl_names = [name for item in value["rl_attempts"] for name in item.get("run_names", [])]
    summary = value["summary"]
    assert len(sft_names) == len(set(sft_names)) == summary["sft_run_names_indexed"] == 37
    assert len(rl_names) == len(set(rl_names)) == summary["rl_run_names_indexed"] == 25
    all_names = sft_names + rl_names
    assert len(all_names) == len(set(all_names)) == summary["named_training_runs_indexed"] == 62
    assert len(value["checkpoints"]) == summary["checkpoint_entries_indexed"] == 7
    assert len(value["webexploitbench_evaluations"]) == 8
    assert summary["webexploitbench_evaluation_lineages_indexed"] == 8
    assert len(value["fleet_heldout_evaluations"]) == 3
    assert summary["fleet_evaluation_lineages_indexed"] == 3
    assert summary["accepted_sft_checkpoints"] == len(accepted) == 4
    assert summary["accepted_rl_checkpoints"] == 0
    assert summary["accepted_matched_webexploitbench_comparisons"] == 0
    assert summary["accepted_trained_checkpoint_fleet_heldout_evaluations"] == 0

    budget = value["failure_budget_reset"]
    assert budget["max_terminal_failed_cluster_jobs"] == 10
    assert budget["consumed_since_reset"] == 0
    assert budget[
        "any_submitted_goal_owned_dev_or_production_cluster_job_that_terminally_fails_consumes_budget"
    ]

    image_build = value["image_build_qualification"]
    assert image_build["status"] == "blocked_before_submission"
    assert image_build["failure_budget_consumed"] == 0
    assert len(image_build["blockers"]) == 3

    tasks = value["task_inventory"]
    assert tasks["current_blackbox_task_keys"] == 1055
    assert tasks["current_exact_receipt_task_keys"] == 80
    assert tasks["no_exact_receipt_in_this_refresh_task_keys"] == 975


def test_index_does_not_promote_controller_success_to_scientific_acceptance():
    value = json.loads(INDEX.read_text())

    assert all(item["disposition"] != "accepted_checkpoint" for item in value["rl_attempts"])
    teacher3k_262k = next(
        item for item in value["sft_attempts"] if item["id"] == "q38-teacher3k-262k-lineage"
    )
    assert teacher3k_262k["disposition"] == "failed_or_partial"
    assert teacher3k_262k["scientific_comparability"].startswith("mechanics_only")
    assert not any(
        item["scientific_comparability"].startswith("matched")
        for item in value["webexploitbench_evaluations"]
    )
    assert not any(
        item["scientific_comparability"] == "accepted_matched_heldout_comparison"
        for item in value["fleet_heldout_evaluations"]
    )


def test_index_ids_are_unique_within_each_evidence_class():
    value = json.loads(INDEX.read_text())

    for key in (
        "internal_recipe_matrix",
        "sft_attempts",
        "rl_attempts",
        "checkpoints",
        "webexploitbench_evaluations",
        "fleet_heldout_evaluations",
    ):
        ids = [item["id"] for item in value[key]]
        assert len(ids) == len(set(ids)), key


def test_teacher3k_262k_four_node_hypothesis_changes_only_topology_batch_and_steps():
    value = json.loads(INDEX.read_text())
    lineage = next(
        item for item in value["sft_attempts"] if item["id"] == "q38-teacher3k-262k-lineage"
    )
    hypothesis = lineage["four_node_hypothesis"]

    assert hypothesis["status"] == "not_qualified_or_launchable"
    assert hypothesis["exact_parent"] == "chris-q38-t3k262-can-v12-a421f0a3"
    assert hypothesis["only_recipe_deltas"] == {
        "nodes": {"from": 8, "to": 4},
        "batch_size": {"from": 64, "to": 32},
        "max_steps": {"from": 2, "to": 4},
    }
    assert hypothesis["unchanged_memory_controls"] == {
        "layer_checkpoint_group_size": 1,
        "gdn_chunk_tokens": 512,
        "lm_head_chunk_tokens": 1024,
        "mlp_chunk_tokens": 1024,
        "rmsnorm_chunk_tokens": 1024,
    }
