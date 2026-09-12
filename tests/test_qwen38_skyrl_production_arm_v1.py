"""Static release gates for the inert Qwen3.8 production SkyRL arm."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from cyber_post_train.jobs import digest
from training import rl_data, skyrl, skyrl_promotion

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
    assigned = {(row["task_key"], row["task_version_id"]): row["split"] for row in split["tasks"]}
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
    assert (
        "No digest-valid production-promotion receipt cross-binding accepted dev8, "
        "reward-terminal, native checkpoint and all-rank reload evidence is bound."
        in arm["blocked_reasons"]
    )
    assert all(
        gate["terminal_receipt_path"] is None and gate["terminal_receipt_file_sha256"] is None
        for gate in arm["qualification"].values()
    )
    for source in arm["source_files"].values():
        assert source["file_sha256"] == file_sha256(ROOT / source["path"])
    reload = arm["qualification"]["native_reload"]
    assert reload["contract_file_sha256"] == file_sha256(ROOT / reload["contract_path"])
    run = arm["candidate_run"]
    assert run["production_promotion"] is None
    assert arm["candidate_run_sha256"] == "sha256:" + digest(
        {key: value for key, value in run.items() if key != "production_promotion"}
    )
    assert data["name"] == run["name"] == run["wandb"]["run_id"]
    assert run["output_root"] == "/mnt/sfs/jobs/chris-q38-rl-prod1"
    assert data["output"] == run["data"]["root"]
    assert data["limits"] == {
        "context_tokens": 98304,
        "response_tokens": 81920,
        "max_tokens_per_turn": 4096,
        "max_turns": 600,
        "episode_seconds": 2400,
        "tool_seconds": 330,
        "tool_result_chars": 50000,
    }
    assert (
        "max_turns_600_no_rollout_truncation" in arm["recipe_relationship"]["same_as_reward_canary"]
    )
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
        "zero_verifier_calls",
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


def compiled_candidate_plan() -> tuple[dict, dict]:
    run = load(ARM)["candidate_run"]
    config = skyrl.SkyRLConfig(
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
        context_tokens=98304,
        response_tokens=81920,
        tokens_per_turn=4096,
        max_turns=600,
        **run["recipe"],
    )
    model, native_sources = {"exact": "bound-model"}, {"exact": "native-sources"}
    plan = {
        "run_name": run["name"],
        "output_root": run["output_root"],
        "model": model,
        "native_sources": native_sources,
        "arguments": vars(config),
        "native_overrides": skyrl.overrides(config),
        "execution": {
            "image": "registry/skyrl@sha256:" + "1" * 64,
            "image_cpu_qualification": {"exact": "qualified"},
            "runtime_user": {"uid": 1000, "gid": 100, "run_as_non_root": True},
            "production_promotion": {"exact": "promotion"},
            "cluster_target": "prod",
            "priority": "c1",
            "resources": run["cluster"]["resources"],
        },
    }
    accepted = {
        "plan": {
            "source_manifest": {"source_plan": {"model": model, "native_sources": native_sources}}
        }
    }
    return plan, accepted


def test_every_prod_skyrl_plan_requires_exact_promotion() -> None:
    generic = {
        "backend": "skyrl",
        "name": "different-prod-run",
        "cluster": {"target": "prod"},
    }
    assert skyrl_promotion.requires_production_promotion(generic) is True
    with pytest.raises(ValueError, match="exact reviewed run"):
        skyrl_promotion.bind_production_promotion(generic, ROOT)


def test_compiled_plan_is_fully_bound_to_candidate_digest() -> None:
    plan, accepted = compiled_candidate_plan()
    assert digest(skyrl_promotion._plan_candidate_projection(plan)) == (
        skyrl_promotion.EXPECTED_CANDIDATE_SHA256
    )
    skyrl_promotion._exact_compiled_plan(plan, accepted)
    for target, key, invalid in (
        ("arguments", "steps", 60),
        ("arguments", "lr", 2e-6),
        ("arguments", "nodes", 2),
        ("arguments", "groups", 2),
        ("arguments", "samples_per_prompt", 4),
        ("arguments", "model", "Qwen/different"),
        ("arguments", "train_data", "/mnt/sfs/jobs/other/data/train.jsonl"),
        ("arguments", "dev_data", "/mnt/sfs/jobs/other/data/dev.jsonl"),
        ("arguments", "train_rows", 58),
        ("arguments", "dev_rows", 19),
        ("arguments", "context_tokens", 65536),
        ("arguments", "response_tokens", 49152),
        ("arguments", "tokens_per_turn", 2048),
        ("arguments", "max_turns", 80),
        ("execution", "resources", {**plan["execution"]["resources"], "cpu_request": "63"}),
        ("model", "exact", "different-model"),
    ):
        changed = copy.deepcopy(plan)
        changed[target][key] = invalid
        with pytest.raises(ValueError, match="exact promoted candidate"):
            skyrl_promotion._exact_compiled_plan(changed, accepted)


def test_promotion_binds_exact_reviewed_inference_endpoint_identities() -> None:
    reference = {
        "path": "/mnt/sfs/jobs/evidence.json",
        "file_sha256": "0" * 64,
        "receipt_self_sha256": "0" * 64,
    }
    value = {
        "schema": skyrl_promotion.SCHEMA,
        "status": "qualified_for_fresh_live_checks",
        "candidate_run_sha256": skyrl_promotion.EXPECTED_CANDIDATE_SHA256,
        "qualified_image": "registry/skyrl@sha256:" + "1" * 64,
        "source_plan_sha256": "0" * 64,
        "dev8_terminal": reference,
        "reward_terminal": reference,
        "checkpoint_manifest": reference,
        "reload_accepted": reference,
        "production_data_manifest": reference,
        "runtime_user": {"uid": 1000, "gid": 100, "run_as_non_root": True},
        "benchmark_isolation": {
            "optimizer_split": "train",
            "dev_is_evaluation_only": True,
            "final_test_rows": 0,
            "webexploitbench_rows": 0,
        },
        "live_requirements": skyrl_promotion.LIVE_REQUIREMENTS,
        "excluded_inference_endpoints": skyrl_promotion.EXCLUDED_INFERENCE_ENDPOINTS,
    }
    value["sha256"] = digest(value)
    skyrl_promotion.validate_promotion(value, check_files=False)
    changed = copy.deepcopy(value)
    changed["excluded_inference_endpoints"][0]["uid"] = "00000000-0000-4000-8000-000000000000"
    changed["sha256"] = digest({key: item for key, item in changed.items() if key != "sha256"})
    with pytest.raises(ValueError, match="invariant"):
        skyrl_promotion.validate_promotion(changed, check_files=False)


def test_live_node_gate_counts_unscheduled_gpu_pods_and_exact_namespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Client:
        @staticmethod
        def all_runs():
            return []

    class Wandb:
        @staticmethod
        def run(_):
            raise RuntimeError("Could not find run")

    pod_count = 6

    def kubectl(*args):
        if args[:2] == ("get", "namespace"):
            return {"metadata": {"uid": skyrl_promotion.PROD_NAMESPACE_UID}}
        if args[:2] == ("get", "deployments"):
            return {
                "items": [
                    {"metadata": endpoint}
                    for endpoint in skyrl_promotion.EXCLUDED_INFERENCE_ENDPOINTS
                ]
            }
        return {
            "items": [
                {
                    "metadata": {},
                    "status": {"phase": "Pending"},
                    "spec": {"containers": [{"resources": {"limits": {"nvidia.com/gpu": "8"}}}]},
                }
                for _ in range(pod_count)
            ]
            + [
                {
                    "metadata": {"deletionTimestamp": "2026-09-12T12:00:00Z"},
                    "status": {"phase": "Running"},
                    "spec": {
                        "nodeName": "terminating-gpu-node",
                        "containers": [{"resources": {"limits": {"nvidia.com/gpu": "8"}}}],
                    },
                }
            ]
        }

    monkeypatch.setattr(skyrl_promotion, "validate_embedded_promotion", lambda _: True)
    monkeypatch.setattr(skyrl_promotion, "_kubectl_json", kubectl)
    result = skyrl_promotion.require_live_external({}, Client(), wandb_api=Wandb())
    assert result["active_experiment_nodes"] == 7
    pod_count = 7
    with pytest.raises(ValueError, match="exceed eight"):
        skyrl_promotion.require_live_external({}, Client(), wandb_api=Wandb())

    def wrong_namespace(*args):
        if args[:2] == ("get", "namespace"):
            return {"metadata": {"uid": "00000000-0000-4000-8000-000000000000"}}
        return kubectl(*args)

    monkeypatch.setattr(skyrl_promotion, "_kubectl_json", wrong_namespace)
    with pytest.raises(ValueError, match="namespace identity"):
        skyrl_promotion.require_live_external({}, Client(), wandb_api=Wandb())
