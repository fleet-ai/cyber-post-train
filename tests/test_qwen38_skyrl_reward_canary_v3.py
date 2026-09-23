"""Offline contract tests for the queued one-step SkyRL reward canary."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from cyber_post_train import cli
from cyber_post_train.jobs import digest
from training import rl_reward_canary as canary
from training import sft, skyrl_training

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "configs/qualification/qwen38-rl-reward-canary-data-prod-v8.json"
RUN = ROOT / "configs/qualification/qwen38-rl-reward-canary-prod-v8.json"
MANIFEST = ROOT / "configs/qualification/qwen38-rl-reward-canary-manifest-prod-v8.json"
PROD6_TERMINAL_EVIDENCE = (
    ROOT / "docs/evidence/qwen38-study/2026-09-21-skyrl-prod6-observer-induced-release-v1.json"
)
QUALIFICATION = ROOT / canary.QUALIFICATION_PATH
QUEUE_EVIDENCE = ROOT / "docs/evidence/qwen38-study/2026-09-20-skyrl-next-gates-queue-v1.json"


def load(path: Path) -> dict:
    return json.loads(path.read_bytes())


def assert_sealed(value: dict) -> None:
    body = {key: item for key, item in value.items() if key != "sha256"}
    assert value["sha256"] == "sha256:" + digest(body)


def metadata(run: dict) -> dict:
    value = load(MANIFEST)
    assert value["name"] == run["name"]
    assert_sealed(value)
    return value


def compile_canary(monkeypatch) -> tuple[dict, dict]:
    run = load(RUN)
    manifest = metadata(run)
    original = sft.read_mapping

    def read(path: Path) -> dict:
        if Path(path) == Path(run["data"]["manifest"]):
            return copy.deepcopy(manifest)
        return original(path)

    monkeypatch.setattr(sft, "read_mapping", read)
    return skyrl_training.compile_rl(run, relative_to=RUN.parent), manifest


def test_prod6_external_identity_remains_frozen_historical_evidence() -> None:
    evidence = load(PROD6_TERMINAL_EVIDENCE)
    assert evidence["sha256"] == digest(
        {key: value for key, value in evidence.items() if key != "sha256"}
    )
    assert evidence["run"] == {
        "name": "chris-q38-rlreward-prod6",
        "rayjob_uid": "6c06625d-8730-4879-9144-a4cd00a58d8a",
        "workload_uid": "76b18017-e4c8-4f55-b62d-901df4f7978f",
        "raycluster_uid": "b706e216-b1b7-41db-aff6-4c4f057f2f47",
        "pod_uid": "d56ba2f6-9464-4bd7-8984-b86d035549f1",
        "plan_sha256": "44b17a6ad291c8ef4c2453e1bc28eaf911b358b361fd06f976e52dd8f2d7e31b",
        "request_sha256": "1ae655ae262af197e58a2c3f4ebe4cecc4488d682f99a4dd94f13118588e40b3",
    }
    assert evidence["result"]["optimizer_updates"] == 0
    assert evidence["result"]["checkpoints"] == 0
    assert evidence["result"]["all_owned_gpu_resources_released"] is True


def test_source_package_is_exact_and_historical_preflight_is_non_gating() -> None:
    data = load(DATA)
    proof = canary.validate_data_config(data, relative_to=DATA.parent)
    assert proof == canary.source_proof()

    task_set = load(ROOT / canary.TASK_SET_PATH)
    split = load(ROOT / canary.SPLIT_PATH)
    evidence = load(ROOT / canary.EVIDENCE_PATH)
    horizon = load(ROOT / canary.HORIZON_PATH)
    for value in (task_set, split, evidence, horizon):
        assert_sealed(value)
    assert [row["split"] for row in split["tasks"]] == ["train", "dev"]
    assert [task["task_key"] for task in canary.TASKS] == [
        row["task_key"] for row in task_set["tasks"]
    ]
    assert horizon["limits"] == canary.LIMITS
    assert horizon["required_task_tools"] == ["bash", "submit_report"]

    qualification = load(QUALIFICATION)
    assert_sealed(qualification)
    assert qualification["execution"]["cluster_target"] == "prod"
    assert qualification["submission_gate"] == {
        "preview_authorized": False,
        "submission_authorized": False,
        "blockers": canary.SUBMISSION_BLOCKERS,
    }
    assert all("failure_budget" not in item for item in canary.SUBMISSION_BLOCKERS)
    assert qualification["topology_successor"] == canary.TOPOLOGY_SUCCESSOR
    assert qualification["topology_successor"]["accepted"] is False
    assert qualification["topology_successor"]["terminal_receipt_grace_seconds"] == 30
    historical = qualification["historical_evidence"]["source_preflight"]
    assert historical["classification"] == "historical_preflight_provenance"
    assert historical["gating"] is False
    receipt = load(QUALIFICATION.parent / historical["path"])
    assert receipt["source"]["git_commit"] == canary.HISTORICAL_PORT_COMMITS["runtime_commit"]
    assert receipt["submission"] == {
        "submitted": False,
        "gpu_allocation": 0,
        "prod_submission_authorized": False,
        "next_gate": "explicit_dev_submission_authorization_after_review",
    }

    queue = load(QUEUE_EVIDENCE)
    assert_sealed(queue)
    assert queue["status"] == "prepared_not_submitted_failure_budget_closed"
    assert queue["failure_budget"] == {
        "used": 10,
        "limit": 10,
        "reset_recorded": False,
        "external_cluster_post_stop": True,
    }
    assert queue["topology_successor"]["submitted"] is False
    assert queue["scientific_canary"]["submitted"] is False


def test_canonical_source_receipts_remain_byte_identical() -> None:
    evidence = load(ROOT / canary.EVIDENCE_PATH)
    bindings = [
        evidence["eligible_source"]["version_receipt"],
        evidence["rejected_metadata_only_successor"]["version_receipt"],
    ]
    for binding in bindings:
        path = (ROOT / canary.EVIDENCE_PATH).parent / binding["path"]
        assert "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest() == (binding["file_sha256"])
        assert not path.read_bytes().endswith(b"\n")


def test_one_node_one_step_config_compiles_to_the_qualified_image(monkeypatch) -> None:
    plan, manifest = compile_canary(monkeypatch)
    request = skyrl_training.job_request(plan)
    assert digest(plan) == "e53e407a68a5ca56752381d76e48e1459cfa038595f381197c9271e6c7cc9b1e"
    assert digest(request) == "1ec530f63e3ef1c68e77b469d9c6a6e950d9829175c0b624913d6e1d9f05fa00"
    arguments, overrides = plan["arguments"], plan["native_overrides"]
    assert plan["data"] == manifest
    assert request["image"] == canary.IMAGE
    assert request["env"]["VLLM_USE_FLASHINFER_SAMPLER"] == "0"
    assert request["workers"] == 1
    assert request["gpus_per_worker"] == 8
    assert request["priority_class"] == "c1"
    assert request["failureAlerts"] is False
    assert request["resources"] == canary.RESOURCES
    assert request["requeueIfPreempted"] is False
    assert arguments["steps"] == overrides["trainer.max_training_steps"] == 1
    assert arguments["groups"] == overrides["trainer.train_batch_size"] == 1
    assert arguments["samples_per_prompt"] == overrides["generator.n_samples_per_prompt"] == 8
    assert overrides["generator.inference_engine.num_engines"] == 2
    assert overrides["generator.inference_engine.tensor_parallel_size"] == 4
    assert overrides["trainer.eval_before_train"] is True
    assert overrides["trainer.eval_interval"] == 1
    assert overrides["trainer.ckpt_interval"] == 1
    assert overrides["trainer.max_ckpts_to_keep"] == 2
    assert overrides["generator.max_turns"] == canary.LIMITS["max_turns"]
    assert overrides["generator.max_input_length"] == canary.LIMITS["context_tokens"]
    assert (
        overrides["generator.sampling_params"]["max_generate_length"]
        == (canary.LIMITS["generation_chunk_tokens"])
    )
    assert overrides["generator.step_wise_trajectories"] is True
    assert overrides["generator.merge_stepwise_output"] is False
    assert plan["qualification"]["submission_gate"]["submission_authorized"] is False


def test_fresh_create_once_identities_live_only_in_configs() -> None:
    data, run = load(DATA), load(RUN)
    assert data["name"] == run["name"] == run["wandb"]["run_id"]
    assert run["output_root"].endswith("/" + run["name"])
    assert data["output"] == run["data"]["root"]
    assert run["data"]["manifest"] == data["output"] + "/manifest.json"
    assert data["output"] != run["output_root"]
    identity_bytes = (
        (ROOT / "training/rl_reward_canary.py").read_bytes()
        + (ROOT / "training/skyrl_training.py").read_bytes()
        + (ROOT / "cyber_post_train/cli.py").read_bytes()
    )
    for identity in (run["name"], run["output_root"], data["output"], run["wandb"]["run_id"]):
        assert identity.encode() not in identity_bytes


def test_external_job_endpoints_are_blocked_before_client_creation(monkeypatch) -> None:
    plan, _ = compile_canary(monkeypatch)
    monkeypatch.setattr(cli, "_client", lambda: pytest.fail("blocked profile reached Jobs API"))
    with pytest.raises(ValueError, match="preview blocked by qualification gate"):
        cli._external_action_gate(plan, "preview")
    with pytest.raises(ValueError, match="submit blocked by qualification gate"):
        cli._external_action_gate(plan, "submit")


@pytest.mark.parametrize("fault", ["recipe", "source", "binding", "image", "route"])
def test_scientific_or_source_drift_fails_closed(monkeypatch, fault: str) -> None:
    run = load(RUN)
    manifest = metadata(run)
    bound = sft.read_mapping(ROOT / "configs/models/qwen38-27b-1d4bf0f2.lock.json")
    weights = sft.read_mapping(ROOT / "configs/models/qwen38-27b-1d4bf0f2.weights.json")
    from training.models import bound_model

    model = bound_model(bound, weights, run["model"]["root"])
    if fault == "recipe":
        run["recipe"]["steps"] = 2
        with pytest.raises(ValueError, match="recipe"):
            canary.validate_run_config(run, manifest, model, relative_to=RUN.parent)
    elif fault == "source":
        changed = copy.deepcopy(manifest)
        changed["selection_sha256"] = "sha256:" + "0" * 64
        with pytest.raises(ValueError, match="data"):
            canary.validate_run_config(run, changed, model, relative_to=RUN.parent)
    else:
        plan, _ = compile_canary(monkeypatch)
        if fault == "binding":
            plan["qualification"]["source_proof"]["limits"]["max_turns"] = 80
        elif fault == "image":
            plan["execution"]["image"] = skyrl_training.IMAGE
        else:
            plan["execution"]["cluster_target"] = "dev"
            plan["execution"]["jobs_api_base_url"] = "https://api.ft.dev.flt.build"
        with pytest.raises(ValueError):
            skyrl_training.job_request(plan)
