from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from cyber_post_train.jobs import digest
from evals.fleet import evaluate

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "configs/evaluation"
CONFIG = EVAL / "qwen38-lr30-step76-fleet-dev17-opencode-seed43-pass1-v1.json"
ADDENDUM = EVAL / "qwen38-fleet-dev17-seed43-lr30-step76-addendum-v1.json"
PARENT = EVAL / "qwen38-fleet-dev17-seed43-matched-protocol-v1.json"
JOB = ROOT / "evals/fleet/cluster/qwen38-lr30-step76-dev17-opencode-seed43-pass1-v1-job.yaml"
PREVIEW = ROOT / "docs/evidence/qwen38-lr30-step76-fleet-dev17-seed43-preview-20260921.json"


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _file_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def test_lr30_addendum_is_self_digesting_and_does_not_rewrite_parent() -> None:
    addendum = _read(ADDENDUM)
    parent = _read(PARENT)
    assert addendum["sha256"] == digest(
        {key: value for key, value in addendum.items() if key != "sha256"}
    )
    assert addendum["status"] == "review_only_do_not_launch"
    scope = addendum["scientific_scope"]
    assert scope["parent_protocol"] == parent["protocol_id"]
    assert scope["retroactively_changes_parent_four_arm_protocol"] is False
    assert scope["interpretation"] == (
        "separate_descriptive_checkpoint_evaluation_not_a_new_matched_parent_arm"
    )
    assert scope["prior_capability_scores_or_outcomes_accessed"] is False
    assert scope["final_eight_task_set_access"] == "sealed"
    assert set(parent["arms"]) == {"base", "fresh75", "self44", "teacher186"}


def test_lr30_plan_matches_parent_tasks_treatment_sampling_and_retry() -> None:
    config, addendum, parent = _read(CONFIG), _read(ADDENDUM), _read(PARENT)
    plan = evaluate.compile_eval(config, relative_to=CONFIG.parent)
    base_config = _read(EVAL / "qwen38-base-fleet-dev17-opencode-seed43-pass1-v1.json")
    base = evaluate.compile_eval(base_config, relative_to=CONFIG.parent)

    assert _file_sha256(CONFIG) == addendum["evaluation"]["config_file_sha256"]
    assert "sha256:" + plan["sha256"] == addendum["evaluation"]["evaluation_plan_sha256"]
    assert plan["tasks"] == base["tasks"]
    assert plan["treatment"] == base["treatment"]
    assert plan["images"] == base["images"]
    assert plan["sampling"] == base["sampling"] == parent["sampling"]
    assert plan["pass_k"] == base["pass_k"] == parent["pass_k"] == 1
    assert plan["max_reviewed_infrastructure_retries"] == 1
    assert plan["automatic_retry"] is False
    assert plan["training_data_eligible"] is False
    assert len(evaluate.plan_rows(plan)) == 17

    route = plan["routes"]["lr30"]
    model = plan["models"]["lr30-step76"]
    assert route["served_id"] == addendum["serving"]["served_id"]
    assert model["revision"] == addendum["checkpoint"]["revision"]
    assert model["session_model"] == addendum["serving"]["session_model"]


def test_lr30_job_is_create_once_alert_silent_c1_and_cpu_only() -> None:
    addendum = _read(ADDENDUM)
    job = yaml.safe_load(JOB.read_text())
    assert job["metadata"]["name"] == addendum["evaluation"]["job_name"]
    assert job["metadata"]["annotations"] == {
        "fleet.ai/failure-alerts": "off",
        "cyber-post-train.fleet.ai/create-once": "true",
    }
    assert job["spec"]["backoffLimit"] == 0
    assert job["spec"]["template"]["spec"]["priorityClassName"] == "c1"
    evaluator = next(
        row for row in job["spec"]["template"]["spec"]["containers"] if row["name"] == "evaluator"
    )
    assert evaluator["resources"]["requests"] == {
        "cpu": "4",
        "memory": "32Gi",
        "ephemeral-storage": "10Gi",
    }
    assert "nvidia.com/gpu" not in evaluator["resources"]["requests"]
    env = {row["name"]: row.get("value") for row in evaluator["env"]}
    assert env["EVAL_CONFIG_NAME"] == CONFIG.name
    assert env["EVAL_OUTPUT"] == addendum["evaluation"]["output_root"]
    assert env["EVAL_DATABASE"] == addendum["evaluation"]["database"]
    assert (
        job["spec"]["template"]["spec"]["volumes"][0]["configMap"]["name"]
        == (addendum["evaluation"]["config_map_name"])
    )


def test_lr30_preview_is_self_digesting_absent_and_created_nothing() -> None:
    preview = _read(PREVIEW)
    assert preview["sha256"] == digest(
        {key: value for key, value in preview.items() if key != "sha256"}
    )
    packet = preview["evaluation_packet"]
    assert packet["generic_and_named_config_bytes_equal"] is True
    assert packet["server_preview_root_failure_alerts"] == "off"
    assert packet["server_preview_priority_class"] == "c1"
    assert packet["server_preview_suspended_for_kueue"] is True
    assert packet["evaluator_gpu_request"] == 0
    assert all(
        packet[key]
        for key in (
            "job_absent",
            "config_map_absent",
            "workload_absent",
            "pod_absent",
            "output_absent",
            "database_absent",
        )
    )
    assert preview["route"]["phase"] == "paused"
    assert preview["route"]["active_pods"] == 0
    assert preview["route"]["gpus_allocated"] == 0
    assert preview["interpretation"] == {
        "root_review_required_before_create": True,
        "route_resumed": False,
        "server_objects_created": False,
        "sessions_created": False,
        "capability_claim_made": False,
        "final_eight_task_set_accessed": False,
    }
