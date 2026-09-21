"""Predeclared, score-blind Fleet dev17 seed-43 comparison protocol."""

import copy
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from cyber_post_train.jobs import digest
from evals.fleet import evaluate

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "configs/evaluation"
CLUSTER = ROOT / "evals/fleet/cluster"
MASTER = EVAL / "qwen38-fleet-dev17-seed43-matched-protocol-v1.json"
PREVIEW = ROOT / "docs/evidence/qwen38-fleet-dev17-seed43-protocol-preview-20260921.json"
TASK_SET = EVAL / "qwen38-fresh75-fleet-dev17-task-set-v1.json"
SPLIT = ROOT / "configs/data/fleet-blackbox-current-study-split-20260914-v2.json"

ARMS = {
    "base": (
        EVAL / "qwen38-base-fleet-dev17-opencode-seed43-pass1-v1.json",
        CLUSTER / "qwen38-base-dev17-opencode-seed43-pass1-v1-job.yaml",
    ),
    "fresh75": (
        EVAL / "qwen38-fresh75-fleet-dev17-opencode-seed43-pass1-v1.json",
        CLUSTER / "qwen38-fresh75-dev17-opencode-seed43-pass1-v1-job.yaml",
    ),
    "self44": (
        EVAL / "qwen38-self-sft-step44-fleet-dev17-opencode-seed43-pass1-v1.json",
        CLUSTER / "qwen38-self-sft-step44-dev17-opencode-seed43-pass1-v1-job.yaml",
    ),
    "teacher186": (
        EVAL / "qwen38-teacher-v5-fleet-dev17-opencode-seed43-pass1-v1.json",
        CLUSTER / "qwen38-teacher-v5-dev17-opencode-seed43-pass1-v1-job.yaml",
    ),
}


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def test_master_is_self_digesting_score_blind_and_review_only():
    master = read(MASTER)
    assert master["sha256"] == digest(
        {key: value for key, value in master.items() if key != "sha256"}
    )
    assert master["status"] == "review_only_do_not_launch"
    assert master["independence"] == {
        "prior_capability_outcomes_accessed": False,
        "prior_capability_outcomes_may_influence_protocol": False,
        "distinguishing_intervention": "new_predeclared_sampling_seed_43",
        "final_eight_task_set_access": "sealed",
    }
    assert set(master["arms"]) == set(ARMS)


def test_preview_is_self_digesting_and_created_nothing():
    preview = read(PREVIEW)
    assert preview["sha256"] == digest(
        {key: value for key, value in preview.items() if key != "sha256"}
    )
    assert preview["interpretation"] == {
        "server_objects_created": False,
        "sessions_created": False,
        "capability_claim_made": False,
        "final_eight_task_set_accessed": False,
        "root_review_required_before_post": True,
    }
    assert all(
        row["generic_and_named_config_bytes_equal"]
        and row["job_absent"]
        and row["config_map_absent"]
        and row["owned_workloads"] == 0
        and row["pods_absent"]
        and row["output_absent"]
        and row["database_absent"]
        for row in preview["dry_run"]["arms"].values()
    )


def test_server_preview_digest_ignores_only_fresh_job_uid_projections():
    def preview(uid: str) -> dict:
        return {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {
                "name": "exact-job",
                "uid": uid,
                "labels": {"stable": "yes"},
            },
            "spec": {
                "suspend": True,
                "selector": {"matchLabels": {"batch.kubernetes.io/controller-uid": uid}},
                "template": {
                    "metadata": {
                        "labels": {
                            "batch.kubernetes.io/controller-uid": uid,
                            "controller-uid": uid,
                            "batch.kubernetes.io/job-name": "exact-job",
                            "stable": "yes",
                        }
                    },
                    "spec": {"priorityClassName": "c1"},
                },
            },
        }

    first = evaluate.stable_job_preview(preview("11111111-1111-1111-1111-111111111111"))
    second = evaluate.stable_job_preview(preview("22222222-2222-2222-2222-222222222222"))
    assert digest(first) == digest(second)
    second["spec"]["suspend"] = False
    assert digest(first) != digest(second)


def test_server_preview_normalizer_rejects_the_wrong_object_kind():
    with pytest.raises(ValueError, match="batch/v1 Job"):
        evaluate.stable_job_preview({"apiVersion": "v1", "kind": "Pod"})


def test_master_binds_exact_frozen_dev17_and_not_final8():
    master, task_set, split = read(MASTER), read(TASK_SET), read(SPLIT)
    assert master["selection"]["split_file_sha256"] == sha256(SPLIT)
    assert master["selection"]["task_set_file_sha256"] == sha256(TASK_SET)
    assert master["selection"]["selection_sha256"] == task_set["selection_sha256"]
    assert master["selection"]["task_version_ids"] == [
        row["task_version_id"] for row in task_set["tasks"]
    ]
    assert set(master["selection"]["task_version_ids"]) == {
        row["task_version_id"] for row in split["evaluation"]["dev"]["tasks"]
    }
    assert not set(master["selection"]["task_version_ids"]) & {
        row["task_version_id"] for row in split["evaluation"]["final_test"]["tasks"]
    }


def test_all_four_arms_compile_to_the_same_new_protocol():
    master = read(MASTER)
    reference = None
    for name, (config_path, _) in ARMS.items():
        config = read(config_path)
        plan = evaluate.compile_eval(config, relative_to=config_path.parent)
        rows = evaluate.plan_rows(plan)
        arm = master["arms"][name]
        assert sha256(config_path) == arm["config_sha256"]
        assert "sha256:" + plan["sha256"] == arm["evaluation_plan_sha256"]
        assert (
            config["sampling"]
            == master["sampling"]
            == {
                "temperature": 0.6,
                "top_p": 0.95,
                "seed": 43,
            }
        )
        assert plan["automatic_retry"] is False
        assert plan["max_reviewed_infrastructure_retries"] == 1
        assert len(rows) == 17
        assert {row["max_retries"] for row in rows} == {1}
        assert plan["training_data_eligible"] is False
        route = next(iter(plan["routes"].values()))
        model = next(iter(plan["models"].values()))
        assert model["session_model"] == f"qwen/{route['served_id']}"
        held_constant = {
            "tasks": plan["tasks"],
            "treatment": plan["treatment"],
            "images": plan["images"],
            "sampling": plan["sampling"],
            "pass_k": plan["pass_k"],
            "retry": plan["max_reviewed_infrastructure_retries"],
        }
        reference = held_constant if reference is None else reference
        assert held_constant == reference


def test_read_only_status_observer_is_outside_the_frozen_execution_runtime():
    assert "rollout_postgres_status.py" not in evaluate.RUNTIME_FILES


@pytest.mark.parametrize("value", [-1, 2, True, "1"])
def test_reviewed_retry_limit_is_narrow(value):
    config = read(next(iter(ARMS.values()))[0])
    config["max_reviewed_infrastructure_retries"] = value
    with pytest.raises(ValueError, match="retry limit"):
        evaluate.compile_eval(config, relative_to=EVAL)


def test_terminal_and_retry_rules_forbid_selective_replay_or_splicing():
    master = read(MASTER)
    assert master["retry_policy"] == {
        "automatic_retry": False,
        "max_reviewed_infrastructure_retries_per_cell": 1,
        "review_requires_score_blind_reconciliation_digest_and_proof_of_no_valid_outcome": True,
        "decision_rule_is_arm_blind": True,
        "valid_outcome_including_valid_zero_is_final": True,
        "one_predeclared_setup_successor_allowed_only_if_claim_count_is_zero": True,
        "setup_successor_forbidden_after_any_claim": True,
    }
    assert master["terminal_policy"] == {
        "accepted_cells_required_per_arm": 17,
        "comparison_requires_all_four_arms_complete": True,
        "pending_claimed_retry_review_or_terminal_invalid_means_incomplete": True,
        "incomplete_cells_are_not_capability_results": True,
        "cross_arm_splicing_forbidden": True,
        "selective_replay_forbidden": True,
    }


def test_primary_jobs_are_unique_alert_silent_cpu_only_create_once_packets():
    master = read(MASTER)
    jobs, outputs, databases = set(), set(), set()
    successor_jobs, successor_outputs, successor_databases, successor_maps = (
        set(),
        set(),
        set(),
        set(),
    )
    for name, (config_path, job_path) in ARMS.items():
        job = yaml.safe_load(job_path.read_text())
        assert job["metadata"]["name"] == master["arms"][name]["primary_job"]
        assert job["metadata"]["annotations"]["fleet.ai/failure-alerts"] == "off"
        assert job["metadata"]["annotations"]["cyber-post-train.fleet.ai/create-once"] == "true"
        assert job["spec"]["backoffLimit"] == 0
        assert job["spec"]["template"]["spec"]["priorityClassName"] == "c1"
        evaluator = next(
            row
            for row in job["spec"]["template"]["spec"]["containers"]
            if row["name"] == "evaluator"
        )
        assert evaluator["resources"]["requests"] == {
            "cpu": "4",
            "memory": "32Gi",
            "ephemeral-storage": "10Gi",
        }
        assert "nvidia.com/gpu" not in evaluator["resources"]["requests"]
        env = {row["name"]: row.get("value") for row in evaluator["env"]}
        assert env["EVAL_CONFIG_NAME"] == config_path.name
        assert env["EVAL_OUTPUT"] == master["arms"][name]["primary_output_root"]
        assert env["EVAL_DATABASE"] == master["arms"][name]["primary_database"]
        jobs.add(job["metadata"]["name"])
        outputs.add(env["EVAL_OUTPUT"])
        databases.add(env["EVAL_DATABASE"])
        successor_jobs.add(master["arms"][name]["zero_claim_setup_successor_job"])
        successor_outputs.add(master["arms"][name]["zero_claim_setup_successor_output_root"])
        successor_databases.add(master["arms"][name]["zero_claim_setup_successor_database"])
        successor_maps.add(master["arms"][name]["zero_claim_setup_successor_config_map"])
    assert len(jobs) == len(outputs) == len(databases) == 4
    assert (
        len(successor_jobs)
        == len(successor_outputs)
        == len(successor_databases)
        == len(successor_maps)
        == 4
    )
    assert not jobs & successor_jobs
    assert not outputs & successor_outputs
    assert not databases & successor_databases


def test_existing_configs_keep_the_zero_retry_default():
    old_path = EVAL / "qwen38-base-fleet-dev17-opencode-pass1-v1.json"
    old = read(old_path)
    assert "max_reviewed_infrastructure_retries" not in old
    plan = evaluate.compile_eval(copy.deepcopy(old), relative_to=old_path.parent)
    assert plan["max_reviewed_infrastructure_retries"] == 0
    assert {row["max_retries"] for row in evaluate.plan_rows(plan)} == {0}
