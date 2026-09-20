"""Offline contract for the broad, no-submit Qwen3.8 SkyRL queue."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from cyber_post_train.jobs import digest
from scripts import prepare_qwen38_skyrl_production_queue as queue
from training import rl_data, skyrl

ROOT = Path(__file__).resolve().parents[1]


def read(path: Path) -> dict:
    return json.loads(path.read_bytes())


def test_generated_queue_is_current_and_self_sealed() -> None:
    expected = queue.expected()
    assert len(expected) == 14
    for path, value in expected.items():
        assert path.read_bytes() == queue.raw(value)
        if "sha256" in value:
            assert value["sha256"] == "sha256:" + digest(
                {key: item for key, item in value.items() if key != "sha256"}
            )
    result = subprocess.run(
        [sys.executable, str(queue.__file__), "--check"],
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"checked": 14, "external_mutations": 0}


def test_broad_split_is_lineage_safe_and_final_test_never_enters_training() -> None:
    task_set, split = read(queue.TASK_SET), read(queue.SPLIT)
    selected = rl_data.selection(task_set, split)
    assert len(task_set["tasks"]) == len(split["tasks"]) == 89
    assert sum(row["split"] == "train" for row in selected) == 59
    assert sum(row["split"] == "dev" for row in selected) == 20
    assert sum(row["split"] == "test" for row in split["tasks"]) == 10
    selected_ids = {(row["task_key"], row["task_version_id"]) for row in selected}
    final_ids = {
        (row["task_key"], row["task_version_id"])
        for row in split["tasks"]
        if row["split"] == "test"
    }
    assert not selected_ids & final_ids
    assert task_set["training_data_eligible"] is True
    assert task_set["tool_catalog_sha256"] == (
        "sha256:85fad6bdc3a835bf52a11a99b3387740eb06eb3d1720ad9bb33f3feac215b44a"
    )


def test_five_arms_change_only_the_declared_treatment() -> None:
    runs = {arm["id"]: read(queue.path_for("run", arm)) for arm in queue.ARMS}
    anchor = runs["a1"]
    assert len(runs) == 5
    assert len({run["name"] for run in runs.values()}) == 5
    assert len({run["output_root"] for run in runs.values()}) == 5
    for arm in queue.ARMS:
        run = runs[arm["id"]]
        assert run["backend"] == "skyrl"
        assert run["cluster"]["priority"] == "c1"
        assert run["recipe"]["nodes"] == 1
        assert run["recipe"]["groups"] == 1
        assert run["recipe"]["samples_per_prompt"] == 8
        assert run["qualification"] == ("../qualification/qwen38-skyrl-production-queue-v1.json")
        assert run["wandb"] == {
            "entity": "thefleet",
            "project": "cyber-post-train",
            "run_id": run["name"],
        }
        # Exercise the actual native configuration validator without needing
        # task text, the image, or any external request.
        args = skyrl.SkyRLConfig(
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
            **run["recipe"],
        )
        assert skyrl.overrides(args)["trainer.max_training_steps"] == arm["steps"]

    common = {
        key: value for key, value in anchor["recipe"].items() if key not in {"lr", "steps", "seed"}
    }
    for arm_id, run in runs.items():
        assert {
            key: value for key, value in run["recipe"].items() if key not in {"lr", "steps", "seed"}
        } == common
        if arm_id == "lr3e7":
            assert (run["recipe"]["lr"], run["recipe"]["steps"], run["recipe"]["seed"]) == (
                3e-7,
                10,
                42,
            )
        elif arm_id == "lr3e6":
            assert (run["recipe"]["lr"], run["recipe"]["steps"], run["recipe"]["seed"]) == (
                3e-6,
                10,
                42,
            )
        elif arm_id == "seed43":
            assert (run["recipe"]["lr"], run["recipe"]["steps"], run["recipe"]["seed"]) == (
                1e-6,
                10,
                43,
            )
        elif arm_id == "dose50":
            assert (run["recipe"]["lr"], run["recipe"]["steps"], run["recipe"]["seed"]) == (
                1e-6,
                50,
                42,
            )
        else:
            assert (run["recipe"]["lr"], run["recipe"]["steps"], run["recipe"]["seed"]) == (
                1e-6,
                10,
                42,
            )


def test_queue_is_fail_closed_behind_canary_and_fresh_external_checks() -> None:
    qualification, evidence = read(queue.QUALIFICATION), read(queue.EVIDENCE)
    required = qualification["canary_prerequisite"]["required"]
    assert qualification["canary_prerequisite"]["run_name"] == "chris-q38-rlreward-prod4"
    assert required == [
        "eight_real_nontruncated_rollouts",
        "one_authoritative_verifier_execution_id_per_rollout",
        "reward_max_strictly_greater_than_reward_min",
        "one_finite_nonzero_optimizer_update",
        "reloadable_step_1_checkpoint",
        "all_owned_gpu_resources_released",
    ]
    assert qualification["submission_gate"]["preview_authorized"] is False
    assert qualification["submission_gate"]["submission_authorized"] is False
    assert [item["source"] for item in qualification["research_basis"]] == [
        "https://arxiv.org/abs/2402.03300",
        "https://www.jmlr.org/papers/v18/16-558.html",
    ]
    assert qualification["successive_halving"]["external_benchmark_selection_forbidden"] is True
    assert (
        "global_cluster_failure_budget_is_10_of_10_until_user_resets_it"
        in (qualification["submission_gate"]["blockers"])
    )
    assert (
        "production_qualification_dispatch_intentionally_disabled_until_prod4_acceptance"
        in qualification["submission_gate"]["blockers"]
    )
    assert (
        "broad_get_only_data_manifests_built_locally_but_not_staged"
        in qualification["submission_gate"]["blockers"]
    )
    assert "broad_get_only_data_manifests_not_built_or_staged" not in qualification[
        "submission_gate"
    ]["blockers"]
    assert qualification["private_data"]["state"] == "locally_built_not_staged"
    assert qualification["private_data"]["fleet_reads"]["mutating_requests"] == 0
    assert len(qualification["private_data"]["arms"]) == 5
    assert evidence["scope"] == "offline_no_submit_no_stage_no_serve_no_cancel"
    assert evidence["external_mutations"] == 0
    assert len(evidence["arms"]) == 5
    assert all(arm["submitted"] is False for arm in evidence["arms"])
    assert all(
        arm["plan_request_state"] == "not_prepared_until_exact_staged_manifest_exists"
        for arm in evidence["arms"]
    )
    duplicate = evidence["duplicate_output_checks"]
    assert duplicate["state"] == (
        "clean_read_only_observation_but_must_repeat_before_preview_or_submit"
    )
    assert duplicate["jobs_api"] == {
        "method": "GET_only",
        "history_rows": 934,
        "matching_name_or_output_rows": 0,
    }
    assert duplicate["kubernetes"]["matching_objects"] == 0
    assert duplicate["sfs"] == "not_observable_from_this_host"
    assert duplicate["wandb"] == "not_observable_without_a_local_read_credential"
    for arm in evidence["arms"]:
        for prefix in ("data_config", "run_config"):
            payload = (ROOT / arm[prefix]).read_bytes()
            assert arm[prefix + "_file_sha256"] == "sha256:" + hashlib.sha256(payload).hexdigest()


def test_data_configs_bind_same_science_but_unique_create_once_roots() -> None:
    configs = [read(queue.path_for("data", arm)) for arm in queue.ARMS]
    assert len({item["output"] for item in configs}) == 5
    for item in configs:
        assert item["backend"] == "skyrl"
        assert item["task_set"].endswith("qwen38-skyrl-production-task-set-v1.json")
        assert item["split"].endswith("qwen38-skyrl-production-split-v1.json")
        assert item["limits"] == configs[0]["limits"]
        assert item["limits"] == {
            "context_tokens": 98304,
            "response_tokens": 81920,
            "max_tokens_per_turn": 4096,
            "max_turns": 600,
            "episode_seconds": 2400,
            "tool_seconds": 330,
            "tool_result_chars": 50000,
        }
