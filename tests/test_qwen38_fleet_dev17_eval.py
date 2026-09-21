"""Frozen Fleet development comparison tests use identity metadata only."""

import hashlib
import json
from pathlib import Path

import yaml

from cyber_post_train.jobs import digest
from evals.fleet import evaluate

ROOT = Path(__file__).resolve().parents[1]
SPLIT = ROOT / "configs/data/fleet-blackbox-current-study-split-20260914-v2.json"
TASK_SET = ROOT / "configs/evaluation/qwen38-fresh75-fleet-dev17-task-set-v1.json"
CONFIG = ROOT / "configs/evaluation/qwen38-fresh75-fleet-dev17-matched-pass1-v2.json"
BASE_CONFIG = ROOT / "configs/evaluation/qwen38-base-fleet-dev17-opencode-pass1-v1.json"
CANDIDATE_CONFIG = ROOT / "configs/evaluation/qwen38-fresh75-fleet-dev17-opencode-pass1-v2.json"
TEACHER_CONFIG = ROOT / "configs/evaluation/qwen38-teacher-v5-fleet-dev17-opencode-pass1-v2.json"
TEACHER_V3_CONFIG = ROOT / "configs/evaluation/qwen38-teacher-v5-fleet-dev17-opencode-pass1-v3.json"
BACKLOG = ROOT / "configs/evaluation/qwen38-fleet-dev17-backlog-v1.json"
SUCCESSOR_JOB = ROOT / "evals/fleet/cluster/qwen38-base-dev17-opencode-pass1-v1-job.yaml"
TEACHER_V3_JOB = ROOT / "evals/fleet/cluster/qwen38-teacher-v5-dev17-opencode-pass1-v3-job.yaml"
SUCCESSOR_SCRIPT = ROOT / "evals/fleet/scripts/run_qwen38_dev17_single_arm_v1.sh"


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def test_dev17_task_set_is_exact_frozen_split_projection():
    split = read(SPLIT)
    task_set = read(TASK_SET)
    assert "sha256:" + hashlib.sha256(SPLIT.read_bytes()).hexdigest() == (
        "sha256:28a3dcaf31f14d724def9023d9435681772d5b8b3a8647cacc7f72ea8fc8adcb"
    )
    assert task_set["selection_sha256"] == split["evaluation"]["dev"]["sha256"]
    assert {row["task_version_id"] for row in task_set["tasks"]} == {
        row["task_version_id"] for row in split["evaluation"]["dev"]["tasks"]
    }
    assert not (
        {row["task_version_id"] for row in task_set["tasks"]}
        & {row["task_version_id"] for row in split["evaluation"]["final_test"]["tasks"]}
    )


def test_fresh75_matched_plan_covers_base_and_candidate_once():
    plan = evaluate.compile_eval(read(CONFIG), relative_to=CONFIG.parent)
    rows = evaluate.plan_rows(plan)
    assert plan["pass_k"] == 1
    assert plan["automatic_retry"] is False
    assert len(plan["tasks"]) == 17
    assert len(rows) == 34
    assert {row["model_id"] for row in rows} == {"qwen3.8-27b-base", "fresh75-step230"}
    assert all(sum(row["model_id"] == model for row in rows) == 17 for model in plan["models"])
    assert plan["routes"]["base"]["task_versions"] == plan["routes"]["fresh75"]["task_versions"]
    assert plan["models"]["fresh75-step230"]["revision"] == (
        "sha256:36eec01f1dd3f0d47f6b099e79970d9533cf59c37d8e15478907413dd162e029"
    )
    assert plan["sha256"] == digest({key: value for key, value in plan.items() if key != "sha256"})


def test_base_only_arm_preserves_exact_pairing_protocol():
    paired = evaluate.compile_eval(read(CONFIG), relative_to=CONFIG.parent)
    base = evaluate.compile_eval(read(BASE_CONFIG), relative_to=BASE_CONFIG.parent)
    assert len(evaluate.plan_rows(base)) == 17
    assert set(base["models"]) == {"qwen3.8-27b-base"}
    assert base["models"]["qwen3.8-27b-base"] == paired["models"]["qwen3.8-27b-base"]
    assert base["routes"]["base"] == paired["routes"]["base"]
    for field in ("selection", "treatment", "images", "sampling", "pass_k"):
        assert base[field] == paired[field]


def test_candidate_only_arm_preserves_exact_pairing_protocol():
    paired = evaluate.compile_eval(read(CONFIG), relative_to=CONFIG.parent)
    candidate = evaluate.compile_eval(read(CANDIDATE_CONFIG), relative_to=CANDIDATE_CONFIG.parent)
    assert len(evaluate.plan_rows(candidate)) == 17
    assert set(candidate["models"]) == {"fresh75-step230"}
    assert candidate["models"]["fresh75-step230"]["revision"] == (
        "sha256:36eec01f1dd3f0d47f6b099e79970d9533cf59c37d8e15478907413dd162e029"
    )
    assert candidate["routes"]["fresh75"]["served_id"] == ("chris-q38-fresh75-step230-web-v2")
    assert (
        candidate["routes"]["fresh75"]["task_versions"]
        == paired["routes"]["fresh75"]["task_versions"]
    )
    for field in ("selection", "treatment", "images", "sampling", "pass_k"):
        assert candidate[field] == paired[field]


def test_teacher_only_arm_preserves_exact_base_protocol():
    base = evaluate.compile_eval(read(BASE_CONFIG), relative_to=BASE_CONFIG.parent)
    teacher = evaluate.compile_eval(read(TEACHER_CONFIG), relative_to=TEACHER_CONFIG.parent)
    assert len(evaluate.plan_rows(teacher)) == 17
    assert set(teacher["models"]) == {"teacher-v5-step186"}
    assert teacher["models"]["teacher-v5-step186"]["revision"] == (
        "sha256:e6d59f5a58ce54b913d775cd71b81c64bd54637b53795df82baec0f71efefaf9"
    )
    assert teacher["routes"]["teacher"]["served_id"] == ("chris-q38-teacher-v5-step186-web-v1")
    assert teacher["routes"]["teacher"]["task_versions"] == base["routes"]["base"]["task_versions"]
    for field in ("selection", "treatment", "images", "sampling", "pass_k"):
        assert teacher[field] == base[field]


def test_teacher_v3_changes_only_create_once_experiment_identity():
    teacher_v2 = read(TEACHER_CONFIG)
    teacher_v3 = read(TEACHER_V3_CONFIG)
    assert teacher_v2["name"] == "q38-teacher-v5-dev17-p1-v2"
    assert teacher_v3["name"] == "q38-teacher-v5-dev17-p1-v3"
    assert {key: value for key, value in teacher_v3.items() if key != "name"} == {
        key: value for key, value in teacher_v2.items() if key != "name"
    }
    compiled = evaluate.compile_eval(teacher_v3, relative_to=TEACHER_V3_CONFIG.parent)
    assert compiled["automatic_retry"] is False
    assert len(evaluate.plan_rows(compiled)) == 17
    assert compiled["models"]["teacher-v5-step186"]["session_model"] == (
        "qwen/chris-q38-teacher-v5-step186-web-v1"
    )


def test_backlog_reuses_frozen_protocol_and_unique_create_once_targets():
    backlog = read(BACKLOG)
    base = evaluate.compile_eval(read(BASE_CONFIG), relative_to=BASE_CONFIG.parent)
    protocol = backlog["frozen_protocol"]
    assert protocol["selection_sha256"] == read(TASK_SET)["selection_sha256"]
    assert protocol["harness"] == base["treatment"]["harness"]
    assert protocol["sampling"] == base["sampling"]
    assert protocol["images"] == base["images"]
    assert protocol["pass_k"] == base["pass_k"] == 1
    outputs = [row["output_root"] for row in backlog["candidates"]]
    databases = [row["database"] for row in backlog["candidates"]]
    assert len(outputs) == len(set(outputs)) == 4
    assert len(databases) == len(set(databases)) == 4
    assert backlog["frozen_protocol"]["final_test_access"] == "sealed"


def test_successor_is_alert_silent_cpu_only_and_create_once():
    job = yaml.safe_load(SUCCESSOR_JOB.read_text())
    assert job["metadata"]["annotations"]["fleet.ai/failure-alerts"] == "off"
    assert job["spec"]["backoffLimit"] == 0
    assert job["spec"]["template"]["spec"]["priorityClassName"] == "c1"
    assert all(
        "nvidia.com/gpu" not in container.get("resources", {}).get("requests", {})
        for container in job["spec"]["template"]["spec"]["containers"]
    )
    script = SUCCESSOR_SCRIPT.read_text()
    assert '"${EVAL_OUTPUT:?EVAL_OUTPUT is required}"' in script
    assert '"${EVAL_DATABASE:?EVAL_DATABASE is required}"' in script
    assert 'cmp --silent /bootstrap/config.json "/bootstrap/$EVAL_CONFIG_NAME"' in script


def test_teacher_v3_job_is_alert_silent_cpu_only_and_capacity_matched():
    job = yaml.safe_load(TEACHER_V3_JOB.read_text())
    assert job["metadata"]["name"] == "chris-q38-teacher-v5-dev17-p1-v3"
    assert job["metadata"]["annotations"]["fleet.ai/failure-alerts"] == "off"
    assert job["spec"]["backoffLimit"] == 0
    assert job["spec"]["template"]["spec"]["priorityClassName"] == "c1"
    evaluator = next(
        item
        for item in job["spec"]["template"]["spec"]["containers"]
        if item["name"] == "evaluator"
    )
    assert evaluator["resources"]["requests"] == {
        "cpu": "4",
        "memory": "32Gi",
        "ephemeral-storage": "10Gi",
    }
    assert "nvidia.com/gpu" not in evaluator["resources"]["requests"]
    env = {row["name"]: row.get("value") for row in evaluator["env"]}
    assert env["EVAL_CONFIG_NAME"].endswith("pass1-v3.json")
    assert env["EVAL_DATABASE"] == "q38_teacher_v5_dev17_p1_v3"
