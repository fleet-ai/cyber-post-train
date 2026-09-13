"""Offline invariants for the inert Qwen3.8 Miles prod2 successor."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from cyber_post_train.jobs import digest
from training import miles, miles_training, rl_data

ROOT = Path(__file__).resolve().parents[1]
TASK_SET = ROOT / "configs/data/qwen38-rl-filtered-study-a-task-set-v1.json"
SPLIT = ROOT / "configs/data/qwen38-rl-filtered-study-a-split-v1.json"
DATA_V1 = ROOT / "configs/runs/qwen38-miles-rl-filtered-study-a-prod-v1.data.json"
ARM_V1 = ROOT / "configs/runs/qwen38-miles-rl-filtered-study-a-prod-v1.template.json"
DATA_V2 = ROOT / "configs/runs/qwen38-miles-rl-filtered-study-a-prod-v2.data.json"
ARM_V2 = ROOT / "configs/runs/qwen38-miles-rl-filtered-study-a-prod-v2.template.json"


def load(path: Path) -> dict:
    return json.loads(path.read_bytes())


def file_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def test_prod2_has_fresh_create_once_identities_and_no_priority_override() -> None:
    data_v1, arm_v1 = load(DATA_V1), load(ARM_V1)
    data_v2, arm_v2 = load(DATA_V2), load(ARM_V2)
    run_v1, run_v2 = arm_v1["candidate_run"], arm_v2["candidate_run"]

    assert data_v2["name"] == run_v2["name"] == run_v2["wandb"]["run_id"]
    assert run_v2["name"] == "chris-q38-miles-rl-prod2"
    assert run_v2["output_root"] == "/mnt/sfs/jobs/chris-q38-miles-rl-prod2"
    assert data_v2["output"] == run_v2["data"]["root"]
    assert run_v2["data"]["manifest"] == data_v2["output"] + "/manifest.json"
    assert {
        data_v2["name"],
        data_v2["output"],
        run_v2["output_root"],
        run_v2["wandb"]["run_id"],
    }.isdisjoint(
        {
            data_v1["name"],
            data_v1["output"],
            run_v1["output_root"],
            run_v1["wandb"]["run_id"],
        }
    )
    text = (DATA_V2.read_text() + ARM_V2.read_text()).lower()
    assert '"c0"' not in text and '"q0"' not in text
    assert "priority_reason" not in text
    assert "webexploitbench" not in text and '"wbe' not in text


def test_prod2_preserves_the_exact_full_study_science_with_4000_char_tools() -> None:
    task_set, split, data, arm = map(load, (TASK_SET, SPLIT, DATA_V2, ARM_V2))
    selected = rl_data.selection(task_set, split)
    run = arm["candidate_run"]

    assert [row["split"] for row in selected].count("train") == 59
    assert [row["split"] for row in selected].count("dev") == 20
    assert task_set["boundaries"]["webexploitbench_rows_included"] == 0
    assert data["limits"]["tool_result_chars"] == 4000
    assert arm["scientific_boundaries"] == {
        "optimizer_split": "train",
        "fleet_dev_is_evaluation_only": True,
        "train_rows": 59,
        "dev_rows": 20,
        "final_test_rows": 0,
        "external_benchmark_training_rows": 0,
        "external_benchmark_reward_inputs": 0,
        "external_benchmark_hpo_or_checkpoint_inputs": 0,
        "external_benchmark_retry_inputs": 0,
    }
    assert run["recipe"] == {
        "nodes": 1,
        "gpus_per_node": 8,
        "steps": 59,
        "groups": 1,
        "samples_per_prompt": 8,
        "lr": 2e-6,
        "temperature": 0.7,
        "kl_loss_coef": 0.001,
        "max_tokens_per_gpu": 8192,
        "eval_interval": 59,
        "checkpoint_interval": 10,
        "seed": 42,
    }
    assert run["cluster"]["target"] == "prod"
    assert run["cluster"]["priority"] == "c1"
    miles.MilesConfig(
        name=run["name"],
        output_root=run["output_root"],
        model_root=run["model"]["root"],
        torch_dist_root="/mnt/sfs/jobs/chris-cpt-cleanup-q38-miles-base-v1/torch-dist",
        train_data=data["output"] + "/train.jsonl",
        dev_data=data["output"] + "/dev.jsonl",
        data_manifest=run["data"]["manifest"],
        wandb_entity=run["wandb"]["entity"],
        wandb_project=run["wandb"]["project"],
        wandb_run_id=run["wandb"]["run_id"],
        context_tokens=data["limits"]["context_tokens"],
        response_tokens=data["limits"]["response_tokens"],
        tokens_per_turn=data["limits"]["max_tokens_per_turn"],
        **run["recipe"],
    ).validate()


def test_prod2_is_self_digesting_with_data_bound_and_training_unaccepted() -> None:
    arm = load(ARM_V2)

    assert arm["sha256"] == "sha256:" + digest(
        {key: value for key, value in arm.items() if key != "sha256"}
    )
    for source in arm["source_files"].values():
        assert source["file_sha256"] == file_sha256(ROOT / source["path"])
    assert arm["candidate_run_sha256"] == "sha256:" + digest(
        {key: value for key, value in arm["candidate_run"].items() if key != "production_promotion"}
    )
    assert arm["status"] == "blocked_not_launchable"
    assert arm["launchable"] is False
    assert arm["candidate_run"]["production_promotion"] is None
    qualification = arm["qualification"]
    assert qualification["dev7_source_run_name"] == "chris-q38-miles-rlreward-dev7"
    assert qualification["dev7_accepted_terminal"] is None
    assert qualification["active_canary_binding"] is None
    assert qualification["reward_terminal"] is None
    assert qualification["native_reload"] is None
    manifest = qualification["production_data_manifest"]
    assert manifest == {
        "schema": "cyber_miles_data_v1",
        "path": "/mnt/sfs/jobs/chris-q38-miles-rl-prod2-inputs/data/manifest.json",
        "file_sha256": ("sha256:6d5df2cde536234e69f38bb7d8f89e19d1f4493271fa4a2a7aa28d263fd669cd"),
        "receipt_sha256": (
            "sha256:c07bf3de80683f64735f02f9edb6e536fcfc4dda78e3f9408587a029cd57aa1a"
        ),
        "preparation_receipt": {
            "path": "/mnt/sfs/jobs/chris-q38-miles-rl-prod2-inputs/DATA_PREPARED.json",
            "file_sha256": (
                "sha256:72d8f7b94954fc64d5738d0a47a733fd39be2e3cdeaa610c845b1a6a0475bb64"
            ),
            "receipt_sha256": (
                "sha256:3bf0490262723c87ba951be6bfe43ddb3593c25d6edf794f3efa11f1166525d9"
            ),
        },
        "rows": {"train": 59, "dev": 20},
        "gpus": 0,
        "environment_creates": 0,
    }
    assert manifest["path"] == arm["candidate_run"]["data"]["manifest"]
    assert manifest["path"] == load(DATA_V2)["output"] + "/manifest.json"
    assert arm["dev7_to_production"]["dev7_submission_identity"] == {
        "api_run_name": "chris-q38-miles-rlreward-dev7-fa5c7491",
        "api_run_id": "fa5c7491-a778-4ac8-965c-82d37be70b4c",
        "rayjob_uid": "11bc3141-6261-49d5-98f4-a98f95ecf2d5",
        "workload_uid": "1a73677b-0161-4502-acc1-0ac94c1ce75c",
    }


def test_prod2_cannot_compile_or_borrow_a_prod1_promotion() -> None:
    prod2 = load(ARM_V2)["candidate_run"]

    with pytest.raises(ValueError, match="differs from the exact reviewed run"):
        miles_training.compile_rl(prod2, relative_to=ARM_V2.parent)

    borrowed = copy.deepcopy(prod2)
    borrowed["production_promotion"] = {
        "path": "/mnt/sfs/jobs/prod1/PROMOTION.json",
        "file_sha256": "1" * 64,
        "receipt_sha256": "2" * 64,
    }
    with pytest.raises(ValueError, match="differs from the exact reviewed run"):
        miles_training.compile_rl(borrowed, relative_to=ARM_V2.parent)
