"""Static release gates for the inert Qwen3.8 production SkyRL arm."""

import hashlib
import json
from pathlib import Path

from cyber_post_train.jobs import digest
from training import rl_data, skyrl

ROOT = Path(__file__).resolve().parents[1]
ELIGIBLE = ROOT / "configs/data/qwen-blackbox-eligible-v1.json"
INVENTORY = ROOT / "configs/data/qwen-blackbox-study-inventory-v1.json"
STUDY_SPLIT = ROOT / "configs/data/qwen-blackbox-study-split-a-v1.json"
TASK_SET = ROOT / "configs/data/qwen38-rl-filtered-study-a-task-set-v1.json"
RL_SPLIT = ROOT / "configs/data/qwen38-rl-filtered-study-a-split-v1.json"
DATA = ROOT / "configs/runs/qwen38-rl-filtered-study-a-prod-v1.data.json"
ARM = ROOT / "configs/runs/qwen38-rl-filtered-study-a-prod-v1.template.json"


def load(path: Path) -> dict:
    return json.loads(path.read_bytes())


def file_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def assert_self_digest(value: dict) -> None:
    assert value["sha256"] == "sha256:" + digest(
        {key: item for key, item in value.items() if key != "sha256"}
    )


def keyed(rows: list[dict]) -> dict[tuple[str, str], dict]:
    return {(row["task_key"], row["task_version_id"]): row for row in rows}


def test_rl_source_is_exact_representative_train_and_dev_only() -> None:
    eligible, inventory, source_split = load(ELIGIBLE), load(INVENTORY), load(STUDY_SPLIT)
    task_set, split = load(TASK_SET), load(RL_SPLIT)
    assert_self_digest(task_set)
    assert_self_digest(split)
    assert task_set["source_manifest_sha256"] == "sha256:" + eligible["sha256"]
    assert task_set["representative_inventory_sha256"] == inventory["sha256"]
    assert task_set["representative_split_sha256"] == source_split["sha256"]
    assert split["parent_split_sha256"] == source_split["sha256"]
    assert task_set["boundaries"] == {
        "optimizer_split": "train",
        "native_evaluation_split": "dev",
        "final_test_rows_included": 0,
        "webexploitbench_rows_included": 0,
    }

    expected = {
        (row["task_key"], row["task_version_id"]): row["split"]
        for row in source_split["tasks"]
        if row["split"] in {"train", "dev"}
    }
    assigned = {
        (row["task_key"], row["task_version_id"]): row["split"] for row in split["tasks"]
    }
    assert assigned == expected
    assert list(assigned.values()).count("train") == 59
    assert list(assigned.values()).count("dev") == 20
    assert len(task_set["tasks"]) == len(assigned) == 79
    assert set(keyed(task_set["tasks"])) == set(assigned)
    assert all(row["reference_session_id"] is None for row in split["tasks"])

    final_test = {
        (row["task_key"], row["task_version_id"])
        for row in source_split["tasks"]
        if row["split"] == "final_test"
    }
    assert len(final_test) == 10 and final_test.isdisjoint(assigned)
    assert "webexploit" not in json.dumps(task_set["tasks"]).lower()
    selected = rl_data.selection(task_set, split)
    assert [row["split"] for row in selected].count("train") == 59
    assert [row["split"] for row in selected].count("dev") == 20


def test_runtime_bindings_come_from_reviewed_eligibility_and_taxonomy() -> None:
    eligible = keyed(load(ELIGIBLE)["task_versions"])
    inventory = keyed(load(INVENTORY)["tasks"])
    for row in load(TASK_SET)["tasks"]:
        key = (row["task_key"], row["task_version_id"])
        source, taxonomy = eligible[key], inventory[key]["taxonomy"]
        environment = source["environment"]
        assert {
            "env_key": row["env_key"],
            "env_version": row["env_version"],
            "environment_version_id": row["environment_version_id"],
            "data_key": row["data_key"],
            "data_version": row["data_version"],
        } == {
            "env_key": environment["id"],
            "env_version": environment["version"],
            "environment_version_id": environment["version_id"],
            "data_key": environment["data_id"],
            "data_version": environment["data_version"],
        }
        assert taxonomy["application"]["status"] == "verified"
        assert taxonomy["task_family"]["status"] == "verified"
        assert row["lineage"] == {
            "application": taxonomy["application"]["value"],
            "task_family": taxonomy["task_family"]["value"],
        }


def test_production_candidate_is_digest_bound_but_inert() -> None:
    arm, data = load(ARM), load(DATA)
    assert_self_digest(arm)
    assert arm["status"] == "blocked_not_launchable"
    assert arm["launchable"] is False
    assert len(arm["blocked_reasons"]) == 4
    assert all(
        gate["terminal_receipt_path"] is None
        and gate["terminal_receipt_file_sha256"] is None
        for gate in arm["qualification"].values()
    )
    for source in arm["source_files"].values():
        assert source["file_sha256"] == file_sha256(ROOT / source["path"])
    run = arm["candidate_run"]
    assert arm["candidate_run_sha256"] == "sha256:" + digest(run)
    assert data["name"] == run["name"] == run["wandb"]["run_id"]
    assert run["output_root"] == "/mnt/sfs/jobs/chris-q38-rl-prod1"
    assert data["output"] == run["data"]["root"]
    assert run["data"]["manifest"] == data["output"] + "/manifest.json"
    assert run["cluster"]["target"] == "prod"
    assert run["cluster"]["priority"] == "c1"
    assert arm["live_release_gates"]["expected_request"] == {
        "workers": 1,
        "gpus_per_worker": 8,
        "priority_class": "c1",
        "priority_value": 10000,
        "requeue_if_preempted": False,
        "secrets": ["fleet-api", "wandb-api"],
    }


def test_recipe_matches_dev_reward_shape_and_never_optimizes_dev() -> None:
    arm = load(ARM)
    run, recipe = arm["candidate_run"], arm["candidate_run"]["recipe"]
    cfg = skyrl.SkyRLConfig(
        name=run["name"],
        output_root=run["output_root"],
        model_root=run["model"]["root"],
        train_data=run["data"]["root"] + "/train.jsonl",
        dev_data=run["data"]["root"] + "/dev.jsonl",
        data_manifest=run["data"]["manifest"],
        train_rows=59,
        dev_rows=20,
        wandb_entity=run["wandb"]["entity"],
        wandb_project=run["wandb"]["project"],
        wandb_run_id=run["wandb"]["run_id"],
        **recipe,
    )
    cfg.validate()
    native = skyrl.overrides(cfg)
    assert recipe == {
        "nodes": 1,
        "steps": 59,
        "groups": 1,
        "samples_per_prompt": 8,
        "lr": 1e-6,
        "eval_interval": 59,
        "checkpoint_interval": 10,
        "keep_checkpoints": 3,
        "seed": 42,
        "engine_start_timeout_seconds": 1800,
        "engine_cleanup_timeout_seconds": 300,
    }
    assert native["data.train_data"] == [cfg.train_data]
    assert native["data.val_data"] == [cfg.dev_data]
    assert cfg.dev_data not in native["data.train_data"]
    assert native["trainer.train_batch_size"] == 1
    assert native["trainer.eval_batch_size"] == 20
    assert native["generator.n_samples_per_prompt"] == 8
    assert native["trainer.epochs"] == 1
    assert native["trainer.max_training_steps"] == 59
    assert native["trainer.ckpt_interval"] == 10
    assert native["trainer.logger"] == "wandb"


def test_every_live_and_dev_qualification_gate_is_fail_closed() -> None:
    arm = load(ARM)
    assert arm["qualification"]["dev8_engine"]["required"] == [
        "digest_valid",
        "exact_corrected_image",
        "both_tp4_engines_started",
        "zero_task_rows",
        "zero_rollouts",
        "zero_verifier_calls",
        "zero_optimizer_steps",
        "zero_checkpoints",
        "gpu_release",
    ]
    reward = set(arm["qualification"]["reward_canary"]["required"])
    assert {
        "authoritative_nonempty_verifier_execution_ids",
        "all_rollouts_nontruncated",
        "exactly_one_optimizer_update",
        "independent_weight_or_optimizer_change",
        "sealed_step1_checkpoint",
        "wandb_identity_and_finite_scalars",
        "gpu_release",
    } <= reward
    reload = set(arm["qualification"]["native_reload"]["required"])
    assert {
        "all_eight_ranks_restored",
        "zero_optimizer_updates",
        "native_model_forward",
        "source_checkpoint_unchanged",
        "gpu_release",
    } <= reload
    live = arm["live_release_gates"]
    assert live["all_required"] is True
    assert live["must_be_proven_before_data_build"] == ["prod_data_output_absent"]
    assert len(live["must_be_rechecked_immediately_before_one_post"]) == 8
    assert arm["recipe_relationship"]["production_only_differences"]["cluster_target"] == (
        "prod instead of dev"
    )
