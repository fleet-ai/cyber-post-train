"""Static release gates for the inert Qwen3.8 Miles production arm."""

import hashlib
import json
from pathlib import Path

from cyber_post_train.jobs import digest
from training import miles, rl_data

ROOT = Path(__file__).resolve().parents[1]
TASK_SET = ROOT / "configs/data/qwen38-rl-filtered-study-a-task-set-v1.json"
SPLIT = ROOT / "configs/data/qwen38-rl-filtered-study-a-split-v1.json"
DATA = ROOT / "configs/runs/qwen38-miles-rl-filtered-study-a-prod-v1.data.json"
ARM = ROOT / "configs/runs/qwen38-miles-rl-filtered-study-a-prod-v1.template.json"


def load(path: Path) -> dict:
    return json.loads(path.read_bytes())


def file_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_miles_data_is_exactly_59_train_and_20_dev() -> None:
    task_set, split, data = map(load, (TASK_SET, SPLIT, DATA))
    selected = rl_data.selection(task_set, split)

    assert data["backend"] == "miles"
    assert data["task_set"] == "../data/qwen38-rl-filtered-study-a-task-set-v1.json"
    assert data["split"] == "../data/qwen38-rl-filtered-study-a-split-v1.json"
    assert [row["split"] for row in selected].count("train") == 59
    assert [row["split"] for row in selected].count("dev") == 20
    assert task_set["boundaries"] == {
        "optimizer_split": "train",
        "native_evaluation_split": "dev",
        "final_test_rows_included": 0,
        "webexploitbench_rows_included": 0,
    }


def test_candidate_is_exact_inert_miles_one_by_eight_production_plan() -> None:
    arm, data = load(ARM), load(DATA)
    candidate = arm["candidate_run"]

    assert arm["sha256"] == "sha256:" + digest(
        {key: value for key, value in arm.items() if key != "sha256"}
    )
    assert arm["status"] == "blocked_not_launchable" and arm["launchable"] is False
    assert arm["candidate_run_sha256"] == "sha256:" + digest(
        {key: value for key, value in candidate.items() if key != "production_promotion"}
    )
    assert candidate["production_promotion"] is None
    assert data["name"] == candidate["name"] == candidate["wandb"]["run_id"]
    assert data["output"] == candidate["data"]["root"]
    assert candidate["cluster"]["target"] == "prod"
    assert candidate["cluster"]["priority"] == "c1"
    assert candidate["recipe"] == {
        "nodes": 1,
        "gpus_per_node": 8,
        "steps": 59,
        "groups": 1,
        "samples_per_prompt": 8,
        "lr": 1e-6,
        "eval_interval": 59,
        "checkpoint_interval": 10,
        "seed": 42,
    }
    config = miles.MilesConfig(
        name=candidate["name"],
        output_root=candidate["output_root"],
        model_root=candidate["model"]["root"],
        torch_dist_root="/mnt/sfs/jobs/chris-cpt-cleanup-q38-miles-base-v1/torch-dist",
        train_data=data["output"] + "/train.jsonl",
        dev_data=data["output"] + "/dev.jsonl",
        data_manifest=candidate["data"]["manifest"],
        wandb_entity=candidate["wandb"]["entity"],
        wandb_project=candidate["wandb"]["project"],
        wandb_run_id=candidate["wandb"]["run_id"],
        context_tokens=data["limits"]["context_tokens"],
        response_tokens=data["limits"]["response_tokens"],
        tokens_per_turn=data["limits"]["max_tokens_per_turn"],
        **candidate["recipe"],
    )
    config.validate()


def test_every_dev_proof_and_live_gate_starts_unbound() -> None:
    arm = load(ARM)
    exact = arm["active_dev3"]

    assert exact["source_plan_sha256"] == (
        "sha256:245b404f9507ac2603c53969dd4506aee811e1c02ebf468543994cff76e3e95e"
    )
    assert exact["source_request_sha256"] == (
        "sha256:d2f2514a33bdf05610c5765c11efaecfcd6eea39e4297115cc39ce3c86d3f927"
    )
    assert exact["runtime_bundle_sha256"] == (
        "sha256:0e3ff1646344abb1b4be13ea064143014e94c13bc22c6c7e8c1a552f8bbe32a6"
    )
    for gate in (arm["qualification"]["reward_terminal"], arm["qualification"]["native_reload"]):
        assert gate["file_sha256"] is None
        assert gate["receipt_sha256"] is None
    data = arm["qualification"]["production_data_manifest"]
    assert data["file_sha256"] == (
        "sha256:6a74daf4ba3f8b491caba73de8036ef648c6805202fe22a937447c6e4bf86b25"
    )
    assert data["receipt_sha256"] == (
        "sha256:0ff63c73d74c47e1fb97865b8b101885236deaf5ee9b92582f544a28abf9d546"
    )
    assert len(arm["blocked_reasons"]) == 3
    assert (
        "active_production_experiment_nodes_plus_candidate_at_most_eight"
        in arm["promotion_requirements"]["live_immediately_before_one_post"]
    )


def test_external_benchmark_is_only_a_sealed_post_training_handoff() -> None:
    arm, data = load(ARM), load(DATA)
    training_payload = json.dumps(
        {
            "data": data,
            "candidate": arm["candidate_run"],
            "qualification": arm["qualification"],
        },
        sort_keys=True,
    ).lower()
    handoff = arm["post_training_handoff"]["external_web_benchmark"]
    export = arm["post_training_handoff"]["concurrent_export_qualification"]

    assert "webexploitbench" not in training_payload
    assert export["required_before_production_training"] is False
    assert export["required_before_any_eval_or_serving"] is True
    assert handoff["provider"] == "tensorlake_sandbox"
    assert handoff["results_sealed_until_recipe_and_checkpoint_are_frozen"] is True
    assert handoff["checkpoint_or_hyperparameter_selection_allowed"] is False
    assert handoff["training_reward_prompt_retry_or_checkpoint_input_allowed"] is False
    assert handoff["parent_protocol_file_sha256"] == file_sha256(ROOT / handoff["parent_protocol"])
