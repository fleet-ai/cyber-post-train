from __future__ import annotations

import json
from pathlib import Path

import yaml

from training.model_adapter import load_model_adapter

ROOT = Path(__file__).resolve().parents[1]
QUALIFICATION = ROOT / "configs/qualification/qwen38-27b-v1.json"
MODEL_LOCK = ROOT / "configs/models/qwen38-27b-1d4bf0f2.lock.json"
ADAPTER = ROOT / "training/configs/models/qwen38-27b.json"
Q38_SFT_GATE = ROOT / "configs/runs/qwen38-27b-sft-capability-gate.template.json"
Q38_SFT_FULL = ROOT / "configs/runs/qwen38-27b-sft-full.template.json"
Q38_RL_CANARY = (
    ROOT / "configs/runs/qwen38-27b-native-rl-reward-acquisition-canary.template.json"
)
Q38_RL_FULL = ROOT / "configs/runs/qwen38-27b-rl-base-full.template.json"
Q36_SFT_GATE = ROOT / "configs/runs/qwen36-27b-sft-smoke.json"
Q36_SFT_FULL = ROOT / "configs/runs/qwen36-27b-sft-full.json"
SPLIT = ROOT / "configs/data/fleet-a62-task-split-v1.json"
STAGE_JOB = ROOT / "cluster/jobs/chris-cyber-qwen38-stage-1d4bf0f2.yaml"
LINK_JOB = ROOT / "cluster/jobs/chris-cyber-qwen38-canonical-link.yaml"


def _read(path: Path) -> dict:
    value = json.loads(path.read_text())
    assert isinstance(value, dict)
    return value


def test_exact_model_lock_adapter_and_qualification_agree() -> None:
    qualification = _read(QUALIFICATION)
    lock = _read(MODEL_LOCK)
    adapter = load_model_adapter(ADAPTER)
    model = qualification["model"]

    assert qualification["status"] == "blocked_pre_submission"
    assert qualification["paid_training_authorized"] is False
    assert (model["repository"], model["revision"]) == (lock["repo"], lock["revision"])
    assert adapter.model_id == lock["repo"]
    assert adapter.revision == lock["revision"]
    assert adapter.weights_manifest_sha256 == lock["weights"]["manifest_sha256"]
    assert model["weights_manifest_sha256"] == lock["weights"]["manifest_sha256"]
    assert model["weights_index_sha256"] == lock["weights"]["index_sha256"]
    assert model["tokenizer_manifest_sha256"] == lock["tokenizer"]["manifest_sha256"]
    assert model["chat_template_sha256"] == (
        "sha256:"
        + next(
            row["sha256"]
            for row in lock["tokenizer"]["files"]
            if row["path"] == "chat_template.jinja"
        )
    )
    assert model["parameter_count"] == 27_781_427_952
    assert model["tensor_count"] == 1_199
    assert model["weight_shards"] == 18
    assert model["serialized_weight_bytes"] == 55_563_006_776


def test_live_catalog_observation_fail_closes_training() -> None:
    qualification = _read(QUALIFICATION)
    catalog = qualification["training_catalog_observation"]

    assert catalog["qwen38_catalog_row_present"] is False
    assert catalog["qwen38_staged_model_present"] is False
    assert catalog["sft_preview_http_status"] == 422
    assert "must be staged" in catalog["sft_preview_detail"]
    blocker_ids = [row["id"] for row in qualification["launch_blockers"]]
    assert blocker_ids == [
        "training_catalog_and_staging",
        "qwen38_corpus",
        "sft_trainer_capability",
        "reward_calibration",
        "rl_trainer_and_evidence",
    ]


def test_model_staging_is_cpu_only_queue_safe_and_not_submitted() -> None:
    qualification = _read(QUALIFICATION)
    staging = qualification["model_staging"]
    stage = yaml.safe_load(STAGE_JOB.read_text())
    link = yaml.safe_load(LINK_JOB.read_text())

    assert staging["status"] == "planned_not_submitted"
    assert staging["accelerator_request"] == 0
    assert stage["metadata"]["name"] == "chris-cyber-qwen38-stage-1d4bf0f2"
    assert link["metadata"]["name"] == "chris-cyber-qwen38-canonical-link"
    for job in (stage, link):
        assert job["metadata"]["namespace"] == "fleet-train-jobs"
        assert job["metadata"]["labels"]["kueue.x-k8s.io/queue-name"] == "training-lq"
        assert job["spec"]["suspend"] is True
        pod = job["spec"]["template"]["spec"]
        assert pod["nodeSelector"] == {"workload": "fleetai-training-ng-cpu"}
        assert all(
            "nvidia.com/gpu" not in container.get("resources", {}).get("requests", {})
            for container in pod["containers"]
        )

    stage_script = stage["spec"]["template"]["spec"]["containers"][0]["args"][0]
    assert "revision=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0" in stage_script
    assert (
        "expected_manifest=06c94e47c0e31fd331ed410665c830ab1b657f90f15a1b11e7bc45e2de00f352"
        in stage_script
    )
    assert 'assert len(shards) == 18' in stage_script
    assert '"verified_shards": 18' in stage_script
    assert 'mv "$partial" "$final"' in stage_script
    link_script = link["spec"]["template"]["spec"]["containers"][0]["args"][0]
    assert 'target = Path("/mnt/sfs/models/qwen3.8-27b")' in link_script
    assert "refusing to replace existing canonical path" in link_script


def test_qwen38_sft_templates_reuse_science_but_require_new_token_windows() -> None:
    for q36_path, q38_path in (
        (Q36_SFT_GATE, Q38_SFT_GATE),
        (Q36_SFT_FULL, Q38_SFT_FULL),
    ):
        q36 = _read(q36_path)
        q38 = _read(q38_path)

        assert q38["title"].startswith("chris-cyber-qwen38-27b-")
        assert q38["data"]["job_ids"] == ["chris-cyber-qwen38-windowed-v1"]
        assert q38["model"] == {"staged_model": "qwen3.8-27b", "precision": "bf16"}
        assert q38["trainer"]["trainer_version_id"] is None
        assert q38["trainer"]["args"] == q36["trainer"]["args"]

        assert q38["data"]["task_keys"] == q36["data"]["task_keys"]
        assert q38["data"]["min_reward"] == q36["data"]["min_reward"]
        assert q38["data"]["exclude_multimodal"] == q36["data"]["exclude_multimodal"]
        assert q38["objective"] == q36["objective"]
        assert q38["sft"] == q36["sft"]
        assert q38["eval"] == q36["eval"]
        assert q38["gpus_per_worker"] == q36["gpus_per_worker"] == 8
        assert q38["num_workers"] == q36["num_workers"] == 1


def test_reward_canary_has_exact_successors_long_horizon_and_no_test_leakage() -> None:
    canary = _read(Q38_RL_CANARY)
    split = _read(SPLIT)
    test_ids = {row["task_version_id"] for row in split["tasks"] if row["split"] == "test"}
    canary_ids = [row["task_version_id"] for row in canary["tasks"]["task_versions"]]

    assert canary["title"] == "chris-cyber-qwen38-27b-native-rl-reward-canary-v1"
    assert canary["trainer"]["trainer_version_id"] is None
    assert canary_ids == [
        "c99340e2-3801-5c3c-a50c-3b96cee4572f",
        "ab5f2956-9fbb-54fb-afd8-67fe11402fa4",
    ]
    assert test_ids.isdisjoint(canary_ids)
    assert canary["model"] == {
        "staged_model": "qwen3.8-27b",
        "precision": "bf16",
        "engine_tensor_parallel_size": 4,
        "max_context_length": 65_536,
        "max_prompt_length": 16_384,
        "max_generate_length": 49_152,
        "sequence_parallel_size": 1,
    }
    assert canary["rollout"]["required_task_tools"] == ["bash", "submit_report"]
    assert canary["rollout"]["max_turns"] == 600
    assert canary["rollout"]["max_tokens_per_turn"] == 2_048
    assert canary["rollout"]["tool_result_max_chars"] == 16_000
    assert canary["grpo"]["max_steps"] == 1
    assert canary["grpo"]["group_size"] * canary["grpo"]["train_batch_size"] == 8
    assert canary["trainer"]["args"].count("trainer.max_training_steps=1") == 1


def test_full_rl_template_preserves_step_cap_and_reward_acquisition_controls() -> None:
    request = _read(Q38_RL_FULL)
    qualification = _read(QUALIFICATION)

    assert request["title"] == "chris-cyber-qwen38-27b-rl-base-full-v1"
    assert request["trainer"]["trainer_version_id"] is None
    assert request["tasks"]["task_versions"] == []
    assert request["eval"]["task_versions"] == []
    assert request["grpo"]["max_steps"] == 130
    assert request["trainer"]["args"].count("trainer.max_training_steps=130") == 1
    assert request["rollout"]["required_task_tools"] == ["bash", "submit_report"]
    assert request["rollout"]["max_turns"] == 600
    assert request["model"]["max_generate_length"] == 49_152
    assert qualification["rl"]["full_run_release_rule"].startswith("reward_canary_must")


def test_topology_arithmetic_and_checkpoint_handoff_are_explicit() -> None:
    qualification = _read(QUALIFICATION)
    topology = qualification["topology"]
    sft = topology["sft"]
    state = sft["model_state_lower_bound_bytes"]

    assert topology["workers"] * topology["gpus_per_worker"] == 8
    assert state["bf16_weights"] == qualification["model"]["parameter_count"] * 2
    assert state["bf16_gradients"] == qualification["model"]["parameter_count"] * 2
    assert state["fp32_adam_moments"] == qualification["model"]["parameter_count"] * 8
    assert state["optional_fp32_master_weights"] == qualification["model"]["parameter_count"] * 4
    assert state["total_with_master_weights"] == sum(
        state[key]
        for key in (
            "bf16_weights",
            "bf16_gradients",
            "fp32_adam_moments",
            "optional_fp32_master_weights",
        )
    )
    assert state["per_gpu_if_fully_sharded"] * 8 == state["total_with_master_weights"]
    assert topology["rl"]["inference_engines"] * topology["rl"]["engine_tensor_parallel_size"] == 8
    assert topology["rl"]["rollouts_per_optimizer_step"] == 8

    handoff = qualification["checkpoint_handoff"]
    assert "zero_optimizer_steps" in handoff["export"]
    assert "never_relabel_dtype" in handoff["precision"]
    assert "do_not_assume_qwen36_15_tensor_result" in handoff["auxiliary_heads"]
    assert "atomic_no_replace" in handoff["staging"]


def test_serving_is_exact_base_clone_with_live_parity_gates() -> None:
    serving = _read(QUALIFICATION)["serving"]

    assert serving["served_id"] == "qwen3.8-27b"
    assert serving["image"].endswith(
        "@sha256:febfb971c7352570fc445c466ebd6ffc9d896024958e544a60f2137fd85856b1"
    )
    assert serving["tensor_parallel_size"] == 1
    assert serving["reasoning_parser"] == "qwen3"
    assert serving["tool_call_parser"] == "qwen3_coder"
    assert serving["post_training_rule"].startswith("clone_exact_base_spec")
    assert {
        "exact_model_revision_and_weights_receipt",
        "structured_bash_and_submit_report_tool_calls",
        "fixed_prompt_logit_parity",
    } <= set(serving["live_parity_gates"])
