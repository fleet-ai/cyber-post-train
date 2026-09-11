from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from training.io import digest_json, file_sha256

ROOT = Path(__file__).resolve().parents[1]
QUALIFICATION = ROOT / "configs/qualification/qwen38-27b-v1.json"
MODEL_LOCK = ROOT / "configs/models/qwen38-27b-1d4bf0f2.lock.json"
Q38_SFT_GATE = ROOT / "configs/runs/qwen38-27b-sft-capability-gate.template.json"
Q38_SFT_FULL = ROOT / "configs/runs/qwen38-27b-sft-full.template.json"
Q38_RL_CANARY = ROOT / "configs/runs/qwen38-27b-native-rl-reward-acquisition-canary.template.json"
Q38_RL_FULL = ROOT / "configs/runs/qwen38-27b-rl-base-full.template.json"
Q36_SFT_GATE = ROOT / "configs/runs/qwen36-27b-sft-smoke.json"
Q36_SFT_FULL = ROOT / "configs/runs/qwen36-27b-sft-full.json"
SPLIT = ROOT / "configs/data/fleet-a62-task-split-v1.json"
STAGE_JOB = ROOT / "cluster/jobs/chris-cyber-qwen38-stage-1d4bf0f2.yaml"
LINK_JOB = ROOT / "cluster/jobs/chris-cyber-qwen38-canonical-link.yaml"
READINESS = ROOT / "configs/qualification/qwen38-27b-readiness-2026-09-01-v2.json"
TRAINER_COMPATIBILITY = (
    ROOT / "configs/qualification/qwen38-27b-trainer-compatibility-2026-09-01-v1.json"
)
EXPECTED_WEIGHT_MANIFEST = "06c94e47c0e31fd331ed410665c830ab1b657f90f15a1b11e7bc45e2de00f352"
EXPECTED_WEIGHT_BYTES = 55_563_006_776
EXPECTED_SMALL_FILES = {
    "LICENSE": "bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a",
    "chat_template.jinja": "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041",
    "config.json": "191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab",
    "generation_config.json": "e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e",
    "merges.txt": "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d",
    "model.safetensors.index.json": (
        "77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df"
    ),
    "preprocessor_config.json": "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
    "tokenizer.json": "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3",
    "tokenizer_config.json": "b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27",
    "video_preprocessor_config.json": (
        "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13"
    ),
    "vocab.json": "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003",
}


def _read(path: Path) -> dict:
    value = json.loads(path.read_text())
    assert isinstance(value, dict)
    return value


def _verification_namespace(script: str) -> dict:
    tree = ast.parse(script)
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef))
        or (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "SMALL_FILES"
                for target in node.targets
            )
        )
    ]
    namespace: dict = {}
    exec(compile(ast.Module(body=selected, type_ignores=[]), "<job-verifier>", "exec"), namespace)
    return namespace


def _stage_verifier_script() -> str:
    script = yaml.safe_load(STAGE_JOB.read_text())["spec"]["template"]["spec"]["containers"][0][
        "args"
    ][0]
    marker = (
        'python - "$candidate" "$revision" "$expected_manifest" '
        '"$expected_bytes" "$already_promoted" <<\'PY\'\n'
    )
    return script.split(marker, 1)[1].split("\nPY\n", 1)[0]


def _link_verifier_script() -> str:
    return yaml.safe_load(LINK_JOB.read_text())["spec"]["template"]["spec"]["containers"][0][
        "args"
    ][0]


def _write_tiny_checkpoint(root: Path) -> tuple[str, int, dict[str, str]]:
    shards = []
    for index in range(1, 19):
        name = f"model-{index:05d}-of-00018.safetensors"
        payload = f"shard-{index}".encode()
        (root / name).write_bytes(payload)
        shards.append(
            {"path": name, "size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
        )
    sidecars = {"config.json": hashlib.sha256(b"config").hexdigest()}
    (root / "config.json").write_bytes(b"config")
    (root / "source-tree.json").write_text(
        json.dumps({"revision": "test-revision", "shards": shards}, sort_keys=True)
    )
    canonical = json.dumps(shards, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest(), sum(row["size"] for row in shards), sidecars


def _write_tiny_lock(
    root: Path, manifest: str, verified_bytes: int, sidecars: dict[str, str]
) -> dict:
    lock = {
        "schema": "cyber_post_train_checkpoint_lock_v1",
        "repo": "Qwen/Qwen3.8-27B",
        "revision": "test-revision",
        "weights_manifest_sha256": "sha256:" + manifest,
        "verified_shards": 18,
        "verified_bytes": verified_bytes,
        "small_file_sha256": sidecars,
    }
    (root / ".cyber-post-train-lock.json").write_text(json.dumps(lock))
    return lock


def test_historical_qualification_agrees_with_exact_model_lock() -> None:
    qualification = _read(QUALIFICATION)
    lock = _read(MODEL_LOCK)
    model = qualification["model"]

    assert qualification["status"] == "blocked_pre_submission"
    assert qualification["paid_training_authorized"] is False
    assert (model["repository"], model["revision"]) == (lock["repo"], lock["revision"])
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
        "qwen38_corpus_staging",
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
    assert "expected_bytes=55563006776" in stage_script
    assert "hashlib.sha256(canonical).hexdigest() == expected_manifest" in stage_script
    assert '"verified_bytes": expected_bytes' in stage_script
    assert "assert len(shards) == 18" in stage_script
    assert '"verified_shards": 18' in stage_script
    assert 'hf download "$repo" "${files[@]}"' in stage_script
    assert "assert actual_paths == expected_paths" in stage_script
    assert "assert json.loads(lock_path.read_text()) == lock" in stage_script
    assert 'mv "$partial" "$final"' in stage_script
    link_script = link["spec"]["template"]["spec"]["containers"][0]["args"][0]
    assert 'target = Path("/mnt/sfs/models/qwen3.8-27b")' in link_script
    assert 'hashlib.file_digest(handle, "sha256")' in link_script
    assert f'expected_manifest = "{EXPECTED_WEIGHT_MANIFEST}"' in link_script
    assert f"expected_bytes = {EXPECTED_WEIGHT_BYTES}" in link_script
    assert '"schema": "cyber_post_train_checkpoint_lock_v1"' in link_script
    assert '"verified_shards": 18' in link_script
    assert '"verified_bytes": expected_bytes' in link_script
    assert "assert lock == expected_lock" in link_script
    assert "assert actual_paths == expected_paths" in link_script
    assert "refusing to replace existing canonical path" in link_script
    assert _verification_namespace(_stage_verifier_script())["SMALL_FILES"] == EXPECTED_SMALL_FILES
    assert _verification_namespace(link_script)["SMALL_FILES"] == EXPECTED_SMALL_FILES


@pytest.mark.parametrize("script_factory", [_stage_verifier_script, _link_verifier_script])
def test_staging_verifiers_reject_jointly_tampered_tree_and_lock(
    script_factory, tmp_path: Path
) -> None:
    namespace = _verification_namespace(script_factory())
    manifest, verified_bytes, sidecars = _write_tiny_checkpoint(tmp_path)
    namespace["SMALL_FILES"] = sidecars
    _write_tiny_lock(tmp_path, manifest, verified_bytes, sidecars)
    namespace["verify_checkpoint"](tmp_path, "test-revision", manifest, verified_bytes)

    shard = tmp_path / "model-00001-of-00018.safetensors"
    shard.write_bytes(b"jointly-tampered-shard")
    tree = json.loads((tmp_path / "source-tree.json").read_text())
    tree["shards"][0]["size"] = shard.stat().st_size
    tree["shards"][0]["sha256"] = hashlib.sha256(shard.read_bytes()).hexdigest()
    (tmp_path / "source-tree.json").write_text(json.dumps(tree, sort_keys=True))
    tampered_manifest = hashlib.sha256(
        json.dumps(tree["shards"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    _write_tiny_lock(
        tmp_path, tampered_manifest, sum(row["size"] for row in tree["shards"]), sidecars
    )

    with pytest.raises(AssertionError):
        namespace["verify_checkpoint"](tmp_path, "test-revision", manifest, verified_bytes)


@pytest.mark.parametrize("script_factory", [_stage_verifier_script, _link_verifier_script])
def test_staging_verifiers_reject_jointly_tampered_sidecar_and_lock(
    script_factory, tmp_path: Path
) -> None:
    namespace = _verification_namespace(script_factory())
    manifest, verified_bytes, sidecars = _write_tiny_checkpoint(tmp_path)
    namespace["SMALL_FILES"] = sidecars
    _write_tiny_lock(tmp_path, manifest, verified_bytes, sidecars)
    namespace["verify_checkpoint"](tmp_path, "test-revision", manifest, verified_bytes)

    (tmp_path / "config.json").write_bytes(b"jointly-tampered-sidecar")
    tampered_sidecars = {"config.json": hashlib.sha256(b"jointly-tampered-sidecar").hexdigest()}
    _write_tiny_lock(tmp_path, manifest, verified_bytes, tampered_sidecars)

    with pytest.raises(AssertionError):
        namespace["verify_checkpoint"](tmp_path, "test-revision", manifest, verified_bytes)


def test_qwen38_sft_templates_reuse_science_but_require_new_token_windows() -> None:
    for q36_path, q38_path in (
        (Q36_SFT_GATE, Q38_SFT_GATE),
        (Q36_SFT_FULL, Q38_SFT_FULL),
    ):
        q36 = _read(q36_path)
        q38 = _read(q38_path)

        assert q38["title"].startswith("chris-cyber-qwen38-27b-")
        assert q38["data"]["job_ids"] == ["chris-cyber-qwen38-windowed-v2"]
        assert q38["model"] == {"staged_model": "qwen3.8-27b", "precision": "bf16"}
        assert q38["trainer"]["trainer_version_id"] is None
        assert q38["trainer"]["args"] == q36["trainer"]["args"]

        assert q38["data"]["task_keys"] == q36["data"]["task_keys"]
        assert q38["data"]["min_reward"] == q36["data"]["min_reward"]
        assert q38["data"]["exclude_multimodal"] == q36["data"]["exclude_multimodal"]
        assert q38["objective"] == q36["objective"]
        assert q38["sft"] == q36["sft"]
        assert q38["eval"] == q36["eval"]
        assert q38["node_pool"] == "fleetai-training-ng-gpu"
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


def test_readiness_snapshot_is_self_digested_and_remains_fail_closed() -> None:
    receipt = _read(READINESS)
    embedded = receipt.pop("receipt_sha256")

    assert embedded == digest_json(receipt)
    assert receipt["status"] == "blocked_pre_submission"
    assert receipt["paid_training_authorized"] is False
    assert receipt["training_catalog"]["qwen38_row_present"] is False
    assert receipt["reward_gate"]["satisfied"] is False
    assert receipt["model"]["model_lock_sha256"] == file_sha256(MODEL_LOCK)
    assert receipt["sft_corpus"]["included_splits"] == ["train"]
    assert receipt["sft_corpus"]["all_corpus_tasks_in_exact_train_allowlist"] is True
    assert receipt["sft_corpus"]["staged"] is False
    # This self-digesting historical receipt binds the retired fixed-window builder at
    # its original revision, not today's dense-data implementation.
    assert receipt["recommended_first_paid_run"]["execute_authorized"] is False
    assert receipt["recommended_first_paid_run"]["shape"] == {
        "workers": 1,
        "gpus_per_worker": 8,
        "accelerator": "NVIDIA B300",
        "node_pool": "fleetai-training-ng-gpu",
        "queue": "training-lq",
        "strategy": "fsdp",
        "precision": "bf16",
        "global_batch_size": 8,
        "micro_batch_size_per_gpu": 1,
        "optimizer_steps": 1,
        "checkpoint_interval": 1,
        "before_train_eval": True,
        "post_step_eval": True,
    }
    assert receipt["staging_plan"]["stage_manifest_sha256"] == file_sha256(STAGE_JOB)
    assert receipt["staging_plan"]["canonical_link_manifest_sha256"] == file_sha256(LINK_JOB)
    assert receipt["recommended_first_paid_run"]["template_sha256"] == file_sha256(Q38_SFT_GATE)
    assert all(value is False for value in receipt["mutation_attestation"].values())


def test_trainer_compatibility_receipt_distinguishes_static_fit_from_execution_proof() -> None:
    receipt = _read(TRAINER_COMPATIBILITY)
    embedded = receipt.pop("receipt_sha256")
    readiness = _read(READINESS)

    assert embedded == digest_json(receipt)
    assert receipt["status"] == "blocked_unproven"
    assert receipt["paid_training_authorized"] is False
    assert receipt["model"]["model_lock_sha256"] == file_sha256(MODEL_LOCK)
    serving_pvc = receipt["serving_observation"]["source_pvc"]
    training = receipt["training_storage_observation"]
    assert (serving_pvc["uid"], serving_pvc["storage_class"]) != (
        training["pvc"]["uid"],
        training["pvc"]["storage_class"],
    )
    assert training["candidate_paths_present"] is False
    assert training["qwen38_directory_found_under_models_root"] is False
    assert training["catalog_observation_receipt_sha256"] == readiness["receipt_sha256"]
    assert receipt["static_model_compatibility"]["trainer_fallback_surface"] == {
        "opt_in": "model_config_kwargs.fleet_force_qwen35_torch_gdn=true",
        "accepted_text_model_type": "qwen3_5_text",
        "requires_positive_linear_attention_count": True,
        "requires_exact_replacement_count": True,
        "preserves_parameter_identity": True,
        "qwen38_matches_static_gate": True,
    }
    assert len(receipt["missing_execution_capabilities"]) == 9
    assert receipt["shared_repo_change"] == {
        "catalog_pr_prepared": False,
        "reason": (
            "The requested prerequisite is not met: serving bytes are on a different PVC and no "
            "exact training-visible model path exists."
        ),
    }
    assert all(value is False for value in receipt["mutation_attestation"].values())
