"""Offline compiler and actual embedded-bootstrap tests: no GPU or network."""

import base64
import copy
import gzip
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from cyber_post_train.jobs import digest
from training import sft, sft_dispatch, sft_runtime

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("suffix", [".json", ".yaml", ""])
def test_json_numbers_survive_mapping_readback(tmp_path, suffix):
    value = {"lr": 1e-6, "large": 1e20, "count": 1, "enabled": True, "text": "1e-6"}
    path = tmp_path / ("configuration" + suffix)
    path.write_text(json.dumps(value))
    actual = sft.read_mapping(path)
    assert actual == value
    assert {k: type(v) for k, v in actual.items()} == {k: type(v) for k, v in value.items()}


def test_mapping_still_accepts_yaml_and_rejects_non_objects(tmp_path):
    path = tmp_path / "configuration.yaml"
    path.write_text("# Human-authored configuration\nrecipe:\n  lr: 1.0e-6\n  nodes: 1\n")
    assert sft.read_mapping(path) == {"recipe": {"lr": 1e-6, "nodes": 1}}
    for text in ("[]", "null", "- list\n- not-object"):
        path.write_text(text)
        with pytest.raises(ValueError, match="must be a mapping"):
            sft.read_mapping(path)


@pytest.fixture
def config(tmp_path):
    manifest = {
        "tokenizer": {
            "repo": "Qwen/Qwen3.8-27B",
            "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        },
        "split_sha256": "sha256:" + "a" * 64,
        "files": {
            "train": {
                "path": "train.parquet",
                "rows": 17,
                "sha256": "b" * 64,
                "task_keys": ["train"],
            },
            "dev": {
                "path": "dev.parquet",
                "rows": 2,
                "sha256": "c" * 64,
                "task_keys": ["dev-a", "dev-b"],
            },
        },
    }

    def save(value):
        value = {k: v for k, v in value.items() if k != "sha256"}
        value["sha256"] = "sha256:" + digest(value)
        (tmp_path / "corpus.json").write_text(json.dumps(value))

    save(manifest)
    return (
        {
            "name": "sft-compiler-test",
            "output_root": "/mnt/sfs/jobs/sft-compiler-test",
            "model": {
                "lock": str(ROOT / "configs/models/qwen38-27b-1d4bf0f2.lock.json"),
                "weights": str(ROOT / "configs/models/qwen38-27b-1d4bf0f2.weights.json"),
                "root": "/mnt/sfs/models/test-base",
            },
            "data": {
                "manifest": "corpus.json",
                "root": "/mnt/sfs/datasets/test-corpus",
            },
            "wandb": {
                "entity": "test-team",
                "project": "test-project",
                "group": "test",
                "run_id": "sft-compiler-test",
                "name": "test",
                "tags": ["synthetic"],
            },
        },
        manifest,
        save,
    )


def test_compile_uses_exact_model_manifest_and_complete_epochs(config, tmp_path):
    source, _, _ = config
    source["recipe"] = {"epochs": 3, "batch_size": 16, "lr": 2e-6}
    plan = sft.compile_sft(source, relative_to=tmp_path)
    assert plan["recipe"]["max_steps"] == 6
    assert plan["recipe"]["lr"] == 2e-6
    assert len(plan["model"]["files"]) == 28
    assert plan["datasets"]["train"]["path"] == "/mnt/sfs/datasets/test-corpus/train.parquet"
    request = sft.job_request(plan)
    assert request["workers"] == 1 and request["gpus_per_worker"] == 8
    assert request["priority_class"] == "c1"
    assert "cluster_target" not in plan["execution"]
    assert "jobs_api_base_url" not in plan["execution"]
    assert "queue_priority_class" not in request
    assert request["secrets"] == ["wandb-api"]
    assert "image_pull_secrets" not in request
    assert request == sft.job_request(plan)
    content = json.loads(gzip.decompress(base64.b64decode(request["env"]["CYBER_SFT_BUNDLE"])))
    assert json.loads(content["plan"]) == plan
    assert hashlib.sha256(content["runtime"].encode()).hexdigest() == plan["runtime_sha256"]


def test_four_node_262k_full_sft_fails_closed_without_runtime_qualification(config, tmp_path):
    source, manifest, save = config
    manifest["validation_mode"] = "task_outcomes_only"
    manifest["files"] = {"train": manifest["files"]["train"]}
    manifest["files"]["train"]["rows"] = 320
    save(manifest)
    source["recipe"] = {
        "epochs": 1,
        "batch_size": 32,
        "microbatch_per_gpu": 1,
        "nodes": 4,
        "gpus_per_node": 8,
        "lr": 1e-6,
        "max_length": 262_144,
        "eval_interval": 0,
        "checkpoint_interval": 5,
        "keep_checkpoints": 3,
        "seed": 42,
    }
    source["cluster"] = {"priority": "c1"}

    with pytest.raises(ValueError, match="separately qualified long-context runtime"):
        sft_dispatch.compiler_for_config(source)


def test_four_node_262k_held_shape_cannot_bypass_compile_gate(config, tmp_path):
    source, manifest, save = config
    manifest["validation_mode"] = "task_outcomes_only"
    manifest["files"] = {"train": manifest["files"]["train"]}
    manifest["files"]["train"]["rows"] = 320
    save(manifest)
    source["recipe"] = {
        "nodes": 4,
        "gpus_per_node": 8,
        "batch_size": 32,
        "max_length": 98_304,
        "eval_interval": 0,
        "checkpoint_interval": 5,
        "keep_checkpoints": 3,
    }
    plan = sft.compile_sft(source, relative_to=tmp_path)
    plan["recipe"]["max_length"] = 262_144

    with pytest.raises(ValueError, match="separately qualified long-context runtime"):
        sft_dispatch.compiler_for_plan(plan)


def test_four_node_262k_qualification_packet_is_bounded_and_not_launchable():
    packet = json.loads(
        (ROOT / "configs/qualification/qwen38-teacher3k-262k-4node-canary-v1.json").read_text()
    )
    assert packet["status"] == "held_nonlaunchable_hypothesis"
    assert packet["submission_authorized"] is False
    parent = packet["exact_parent"]
    assert parent["plan_sha256"] == (
        "sha256:3b9e81acb301327c40007a40ab13c3bf5c164152655fb49f96a8db52eb117690"
    )
    assert parent["runtime_source"] == (
        "git:45c04d709f855e20d931456233b85f427558525f:training/sft_runtime.py"
    )
    assert parent["runtime_sha256"] == (
        "sha256:b6b81c9876ddf8379a0837a0aaf5f0426acf8ccc4388d79bd93bad94c59e38d3"
    )
    evidence_path = (
        "docs/evidence/qwen38-study/2026-09-24-qwen38-teacher3k-262k-v12-recovered-parent.md"
    )
    assert parent["historical_evidence_document"] == evidence_path
    evidence = (ROOT / evidence_path).read_text()
    assert parent["plan_sha256"].removeprefix("sha256:") in evidence
    assert parent["image"] == (
        "661864827319.dkr.ecr.us-east-1.amazonaws.com/fleet/skyrl-train@"
        "sha256:ba288751cd227c5be146d28f4a03237545d87d2cbd4c48464945b17fde566ff4"
    )
    assert parent["model"] == {
        "repo": "Qwen/Qwen3.8-27B",
        "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        "weights_manifest_sha256": (
            "sha256:06c94e47c0e31fd331ed410665c830ab1b657f90f15a1b11e7bc45e2de00f352"
        ),
        "full_parameter_training": True,
    }
    assert parent["dataset"]["rows"] == 112
    assert parent["recipe"]["nodes"] == 8
    assert parent["recipe"]["global_batch"] == 64
    assert parent["recipe"]["gdn_chunk_tokens"] == 512
    assert parent["recipe"]["lm_head_chunk_tokens"] == 1024
    assert parent["recipe"]["layer_checkpoint_group_size"] == 1
    hypothesis = packet["four_node_hypothesis"]
    assert hypothesis["hypothesis_only"] is True
    assert hypothesis["mutations"] == {
        "nodes": 4,
        "world_size": 32,
        "data_parallel_size": 32,
        "global_batch": 32,
        "gradient_accumulation_rounds": 1,
        "max_steps": 4,
        "run_name_and_output_root": "new create-once identities",
        "wandb_identity": "new run identity and four-node topology tags",
    }
    assert hypothesis["jobs_api"] == {
        "target": "prod",
        "priority": "c1",
        "requeue_if_preempted": False,
        "required_new_root_annotations": {"fleet.ai/failure-alerts": "off"},
    }
    assert packet["remaining_blockers_before_any_post"]
    assert packet["current_port"]["status"] == ("implemented_and_locally_tested_not_live_qualified")
    assert packet["resource_accounting"] == {
        "candidate_nodes": 4,
        "planned_concurrent_nodes": 9,
        "authorized_concurrent_node_limit": 10,
    }
    assert packet["dev_qualification"] == {
        "exact_four_node_gpu_shape_available": False,
        "allowed_work": "zero_gpu_preflight_only",
        "cleanup_maximum_seconds": 1800,
        "priority": "c1",
    }


def test_compile_binds_and_enforces_first_checkpoint_recovery_horizon(config, tmp_path):
    source, manifest, save = config
    manifest["validation_mode"] = "task_outcomes_only"
    manifest["files"] = {"train": manifest["files"]["train"]}
    manifest["files"]["train"]["rows"] = 800
    save(manifest)
    source["recipe"] = {
        "batch_size": 8,
        "max_length": 32_768,
        "eval_interval": 0,
        "checkpoint_interval": 50,
    }
    expected = 1_800 + 50 * 512 + 300
    source["checkpoint_recovery_horizon_seconds"] = expected

    plan = sft.compile_sft(source, relative_to=tmp_path)
    assert plan["checkpoint_recovery_horizon_seconds"] == expected
    assert sft_runtime.sft_first_checkpoint_seconds(plan) == expected

    source["checkpoint_recovery_horizon_seconds"] = expected - 1
    with pytest.raises(ValueError, match="first recoverable checkpoint exceeds"):
        sft.compile_sft(source, relative_to=tmp_path)


def test_checkpoint_recovery_horizon_rejects_unbudgeted_teacher_ce(config, tmp_path):
    source, _, _ = config
    source["checkpoint_recovery_horizon_seconds"] = 100_000
    with pytest.raises(ValueError, match="only qualified for task-outcome training"):
        sft.compile_sft(source, relative_to=tmp_path)


def test_compile_training_loss_only_plan_has_no_reference_dev_dataset(config, tmp_path):
    source, manifest, save = config
    manifest["validation_mode"] = "task_outcomes_only"
    manifest["files"].pop("dev")
    save(manifest)
    source["recipe"] = {"eval_interval": 0, "checkpoint_interval": 2}

    plan = sft.compile_sft(source, relative_to=tmp_path)
    assert plan["validation_mode"] == "task_outcomes_only"
    assert set(plan["datasets"]) == {"train"}
    assert plan["recipe"]["eval_interval"] == 0
    assert sft.job_request(plan)["env"]["WANDB_MODE"] == "online"


@pytest.mark.parametrize("unexpected", [False, True])
def test_train_only_preflight_exercises_native_loader(monkeypatch, unexpected):
    from training import sft_runtime

    calls = []

    class Trainer:
        def load_dataset(self):
            calls.append((self.plan, "train"))
            return self._load_split("train")

        def load_eval_dataset(self):
            calls.append((self.plan, "eval"))
            return object() if unexpected else None

    monkeypatch.setattr(sft_runtime, "_make_trainer_class", lambda: Trainer)
    plan = {"datasets": {"train": {}}}
    rows = [{"input_ids": [1, 2]}]
    if unexpected:
        with pytest.raises(ValueError, match="unexpectedly produced"):
            sft._check_native_dataset_loader(plan, {"train": rows})
    else:
        sft._check_native_dataset_loader(plan, {"train": rows})
    assert calls == [(plan, "train"), (plan, "eval")]


def test_reference_dev_preflight_exercises_native_train_and_eval_loaders(monkeypatch):
    from training import sft_runtime

    class Trainer:
        def load_dataset(self):
            return self._load_split("train")

        def load_eval_dataset(self):
            return self._load_split("dev")

    monkeypatch.setattr(sft_runtime, "_make_trainer_class", lambda: Trainer)
    plan = {"datasets": {"train": {}, "dev": {}}}
    rows = {"train": [{"input_ids": [1]}], "dev": [{"input_ids": [2]}]}
    sft._check_native_dataset_loader(plan, rows)


@pytest.mark.parametrize(
    "validation_mode,eval_interval",
    [("task_outcomes_only", 50), ("teacher_cross_entropy", 0)],
)
def test_corpus_validation_mode_and_ce_interval_must_agree(
    config, tmp_path, validation_mode, eval_interval
):
    source, manifest, save = config
    manifest["validation_mode"] = validation_mode
    if validation_mode == "task_outcomes_only":
        manifest["files"].pop("dev")
    save(manifest)
    source["recipe"] = {"eval_interval": eval_interval}
    with pytest.raises(ValueError, match="task-outcome evaluation"):
        sft.compile_sft(source, relative_to=tmp_path)


def test_compiler_binds_planned_pause_without_shortening_recipe(config, tmp_path):
    source, _, _ = config
    original = sft.compile_sft(source, relative_to=tmp_path)
    source["pause_after_step"] = 1
    paused = sft.compile_sft(source, relative_to=tmp_path)
    assert paused["recipe"] == original["recipe"]
    assert paused["pause_after_step"] == 1
    request = sft.job_request(paused)
    content = json.loads(gzip.decompress(base64.b64decode(request["env"]["CYBER_SFT_BUNDLE"])))
    assert json.loads(content["plan"])["pause_after_step"] == 1
    assert request["requeueIfPreempted"] is False
    assert request["failureAlerts"] is False


@pytest.fixture
def glm_config(config):
    source, manifest, save = config
    model_dir = ROOT / "configs/models/glm53-30333038"
    source["model"].update(
        lock=str(model_dir / "model.lock.json"),
        weights=str(model_dir / "model.weights.json"),
        root="/mnt/sfs/models/glm-5.3-30333038",
    )
    manifest["tokenizer"] = {
        "repo": "zai-org/GLM-5.3",
        "revision": "30333038ada1f1dacb294a93270305a890b50c14",
    }
    save(manifest)
    source["lora"] = {"rank": 16, "alpha": 32}
    source["recipe"] = {"nodes": 2, "batch_size": 16}
    source["cluster"] = {"resources": {"memory_request": "2048Gi", "memory_limit": "2304Gi"}}
    return source


def test_glm_compiler_binds_full_base_adapter_and_worker_bundle(glm_config, tmp_path):
    plan = sft.compile_sft(glm_config, relative_to=tmp_path)
    request = sft.job_request(plan)
    assert plan["model"]["repo"] == "zai-org/GLM-5.3"
    assert plan["lora"] == {"rank": 16, "alpha": 32}
    assert request["workers"] == 2 and request["gpus_per_worker"] == 8
    assert "image_pull_secrets" not in request
    content = json.loads(gzip.decompress(base64.b64decode(request["env"]["CYBER_SFT_BUNDLE"])))
    assert content["extra_files"]["training/__init__.py"] == ""
    helper = content["extra_files"]["training/glm_runtime.py"]
    assert hashlib.sha256(helper.encode()).hexdigest() == plan["glm_runtime_sha256"]
    assert request["env"]["PYTHONPATH"] == plan["output_root"] + "/.runtime"
    plan["glm_runtime_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="GLM runtime changed"):
        sft.job_request(plan)


@pytest.fixture
def qwen38_lora_config(config, monkeypatch, tmp_path):
    monkeypatch.setattr(sft_runtime, "QWEN38_MEGATRON_SKYRL_REVISION", "f" * 40)
    monkeypatch.setattr(
        sft_runtime,
        "QWEN38_MEGATRON_IMAGE",
        "ghcr.io/fleet-ai/skyrl-fleet-v2/trainer@sha256:" + "d" * 64,
    )
    source_census = {"skyrl/train/sft_trainer.py": "e" * 64}
    monkeypatch.setattr(sft_runtime, "QWEN38_MEGATRON_SOURCE_SHA256", source_census)
    monkeypatch.setattr(
        sft_runtime,
        "QWEN38_MEGATRON_SOURCE_CENSUS_SHA256",
        hashlib.sha256(
            json.dumps(source_census, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    )
    source, manifest, save = config
    manifest["validation_mode"] = "task_outcomes_only"
    manifest["files"].pop("dev")
    save(manifest)
    source["name"] = "chris-q38-lora-sft-c1-v1"
    source["output_root"] = "/mnt/sfs/jobs/chris-q38-lora-sft-c1-v1"
    source["model"]["root"] = "/mnt/sfs/models/qwen3.8-27b-1d4bf0f2"
    source["recipe"] = {
        "epochs": 1,
        "batch_size": 1,
        "microbatch_per_gpu": 1,
        "nodes": 1,
        "gpus_per_node": 8,
        "lr": 3e-5,
        "max_length": 16384,
        "eval_interval": 0,
        "checkpoint_interval": 1,
        "keep_checkpoints": 3,
        "seed": 20260919,
    }
    source["lora"] = dict(sft_runtime.QWEN38_LORA)
    source["runtime"] = {
        "skyrl_source_commit": sft_runtime.QWEN38_MEGATRON_SKYRL_REVISION,
        "image": sft_runtime.QWEN38_MEGATRON_IMAGE,
    }
    source["pause_after_step"] = 1
    source["cluster"] = {
        "priority": "c1",
        "resources": {
            "cpu_request": "64",
            "cpu_limit": "64",
            "memory_request": "512Gi",
            "memory_limit": "768Gi",
        },
    }
    source["wandb"] = {
        "entity": "thefleet",
        "project": "cyber-post-train",
        "group": "qwen38-lora-sft-goal-v1",
        "run_id": source["name"],
        "name": source["name"],
        "tags": [
            "qwen38",
            "lora",
            "teacher-sft",
            "rank64",
            "alpha32",
            "exact-model-gate",
            "planned-pause-step1",
            "task-outcomes-only",
        ],
    }
    # Give this offline fixture a complete synthetic identity so the tests can
    # prove every reviewed field remains fail-closed independently of the real
    # V2 corpus binding.
    original_validate = sft.validate_plan
    monkeypatch.setattr(sft, "validate_plan", lambda *_args, **_kwargs: None)
    synthetic_plan = sft.compile_sft(source, relative_to=tmp_path)
    monkeypatch.setattr(sft, "validate_plan", original_validate)
    monkeypatch.setattr(
        sft_runtime,
        "QWEN38_LORA_ONE_STEP_PLAN",
        sft_runtime._qwen38_lora_one_step_identity(synthetic_plan),
    )
    return source


def test_qwen38_megatron_binding_rejects_incomplete_source_census(monkeypatch):
    monkeypatch.setattr(sft_runtime, "QWEN38_MEGATRON_SKYRL_REVISION", "f" * 40)
    monkeypatch.setattr(
        sft_runtime,
        "QWEN38_MEGATRON_IMAGE",
        "ghcr.io/fleet-ai/skyrl-fleet-v2/trainer@sha256:" + "d" * 64,
    )
    incomplete = dict(sft_runtime.QWEN38_MEGATRON_SOURCE_SHA256)
    incomplete.pop(next(iter(incomplete)))
    monkeypatch.setattr(sft_runtime, "QWEN38_MEGATRON_SOURCE_SHA256", incomplete)
    with pytest.raises(ValueError, match="qualification binding is unresolved"):
        sft_runtime.qwen38_megatron_binding()


def test_qwen38_production_canary_binds_verified_leak_free_corpus():
    binding = sft_runtime.qwen38_lora_one_step_plan_binding()
    assert binding["datasets_sha256"] == (
        "219efad0257ff29b3057c91de88144542f8617a124079c5a20c79cf80b071f72"
    )
    assert binding["corpus_manifest_sha256"] == (
        "sha256:5db5600ac9f403fd147b2ecbf6068b514d075d07ac683d97bc83319c6e89d149"
    )
    assert binding["split_manifest_sha256"] == (
        "sha256:b7b536940995a0d4b8674a4bdadd6240ef88dc1c07200a93f27b592ac8c046ea"
    )
    assert binding["recipe"]["max_steps"] == 866
    assert binding["pause_after_step"] == 1


def test_qwen38_megatron_lora_compiles_only_exact_one_step_gate(qwen38_lora_config, tmp_path):
    plan = sft.compile_sft(qwen38_lora_config, relative_to=tmp_path)
    request = sft.job_request(plan)
    options = sft_runtime.sft_overrides(plan)

    assert plan["lora"] == sft_runtime.QWEN38_LORA
    assert plan["skyrl_runtime"] == {
        "source_commit": sft_runtime.QWEN38_MEGATRON_SKYRL_REVISION,
        "source_files_sha256": sft_runtime.QWEN38_MEGATRON_SOURCE_SHA256,
    }
    assert plan["qualification_gate"]["accepted_for_production"] is False
    assert plan["qualification_gate"]["training_success_is_acceptance"] is False
    assert (
        "deterministic_merge_and_export" in plan["qualification_gate"]["later_zero_step_evidence"]
    )
    assert request["priority_class"] == "c1"
    assert request["image"] == qwen38_lora_config["runtime"]["image"]
    assert request["secrets"] == ["wandb-api"]
    assert request["image_pull_secrets"] == ["ghcr-pull"]
    assert request["env"]["WANDB_MODE"] == "online"
    assert request["env"]["FLA_TILELANG"] == "0"
    contents = json.loads(gzip.decompress(base64.b64decode(request["env"]["CYBER_SFT_BUNDLE"])))
    assert set(contents["extra_files"]) == {
        "training/__init__.py",
        "training/io.py",
        "training/qwen38_lora_artifacts.py",
        "training/sft_runtime.py",
    }
    assert contents["extra_files"]["training/sft_runtime.py"] == contents["runtime"]
    assert request["env"]["PYTHONPATH"] == plan["output_root"] + "/.runtime:/opt/skyrl"
    assert request["env"]["SKYRL_PYTHONPATH_EXPORT"] == "1"
    assert options == {
        **options,
        "strategy": "megatron",
        "language_model_only": True,
        "megatron_config.tensor_model_parallel_size": 8,
        "megatron_config.pipeline_model_parallel_size": 1,
        "megatron_config.context_parallel_size": 1,
        "megatron_config.lora_config.lora_type": "lora",
        "megatron_config.lora_config.merge_lora": True,
        "model.lora.rank": 64,
        "model.lora.alpha": 32,
        "model.lora.dropout": 0.0,
        "model.lora.init_method": "kaiming",
        "model.lora.target_modules": "all-linear",
        "optimizer_config.weight_decay": 0.01,
        "optimizer_config.max_grad_norm": 1.0,
        "remove_microbatch_padding": True,
        "use_sequence_packing": False,
    }
    assert "model_config_kwargs.fleet_force_qwen35_torch_gdn" not in options


def test_qwen38_bootstrap_stages_terminal_receipt_validator_without_checkout(
    qwen38_lora_config, tmp_path
):
    plan = sft.compile_sft(qwen38_lora_config, relative_to=tmp_path)
    request = sft.job_request(plan)
    contents = json.loads(gzip.decompress(base64.b64decode(request["env"]["CYBER_SFT_BUNDLE"])))
    root = tmp_path / "runtime"
    for relative, text in contents["extra_files"].items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            (
                "import sys;sys.path.insert(0,sys.argv[1]);"
                "from training.qwen38_lora_artifacts import validate_checkpoint_receipt;"
                "from training import sft_runtime;"
                "assert callable(validate_checkpoint_receipt);"
                "assert sft_runtime.QWEN38_MEGATRON_SOURCE_SHA256"
            ),
            str(root),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "defect",
    [
        "missing_runtime",
        "missing_image",
        "missing_lora_field",
        "source",
        "tag",
        "image_repo",
        "image_digest",
        "priority",
        "run_name",
        "output_root",
        "wandb_entity",
        "wandb_project",
        "wandb_group",
        "wandb_run_id",
        "wandb_name",
        "wandb_tag",
        "wandb_extra_tag",
        "wandb_tag_order",
        "nodes",
        "gpus",
        "batch",
        "microbatch",
        "length",
        "lr",
        "epochs",
        "seed",
        "checkpoint_interval",
        "keep_checkpoints",
        "cpu_request",
        "cpu_limit",
        "memory_request",
        "memory_limit",
        "model_root",
        "data_root",
        "pause",
        "lora_type",
        "target_modules",
        "rank",
        "alpha",
        "init_method",
        "dropout",
    ],
)
def test_qwen38_megatron_lora_rejects_unqualified_variants(qwen38_lora_config, tmp_path, defect):
    source = copy.deepcopy(qwen38_lora_config)
    if defect == "missing_runtime":
        source.pop("runtime")
    elif defect == "missing_image":
        source["runtime"].pop("image")
    elif defect == "missing_lora_field":
        source["lora"].pop("dropout")
    elif defect == "source":
        source["runtime"]["skyrl_source_commit"] = "0" * 40
    elif defect == "tag":
        source["runtime"]["image"] = "ghcr.io/fleet-ai/skyrl-fleet-v2/trainer:latest"
    elif defect == "image_repo":
        source["runtime"]["image"] = "ghcr.io/example/trainer@sha256:" + "d" * 64
    elif defect == "image_digest":
        source["runtime"]["image"] = "ghcr.io/fleet-ai/skyrl-fleet-v2/trainer@sha256:" + "c" * 64
    elif defect == "priority":
        source["cluster"]["priority"] = "c0"
    elif defect == "run_name":
        source["name"] = "other"
    elif defect == "output_root":
        source["output_root"] = "/mnt/sfs/jobs/other"
    elif defect == "wandb_entity":
        source["wandb"]["entity"] = "other"
    elif defect == "wandb_project":
        source["wandb"]["project"] = "other"
    elif defect == "wandb_group":
        source["wandb"]["group"] = "other"
    elif defect == "wandb_run_id":
        source["wandb"]["run_id"] = "other"
    elif defect == "wandb_name":
        source["wandb"]["name"] = "other"
    elif defect == "wandb_tag":
        source["wandb"]["tags"].remove("exact-model-gate")
    elif defect == "wandb_extra_tag":
        source["wandb"]["tags"].append("unreviewed")
    elif defect == "wandb_tag_order":
        source["wandb"]["tags"].reverse()
    elif defect == "pause":
        source.pop("pause_after_step")
    elif defect in {
        "nodes",
        "gpus",
        "batch",
        "microbatch",
        "length",
        "lr",
        "epochs",
        "seed",
        "checkpoint_interval",
        "keep_checkpoints",
    }:
        key = {
            "gpus": "gpus_per_node",
            "batch": "batch_size",
            "microbatch": "microbatch_per_gpu",
            "length": "max_length",
        }.get(defect, defect)
        source["recipe"][key] = {
            "nodes": 2,
            "gpus": 4,
            "batch": 2,
            "microbatch": 2,
            "length": 32768,
            "lr": 1e-5,
            "epochs": 2,
            "seed": 123,
            "checkpoint_interval": 2,
            "keep_checkpoints": 99,
        }[defect]
    elif defect in {
        "cpu_request",
        "cpu_limit",
        "memory_request",
        "memory_limit",
    }:
        source["cluster"]["resources"][defect] = "1" if defect.startswith("cpu") else "1Gi"
    elif defect == "model_root":
        source["model"]["root"] = "/mnt/sfs/models/other"
    elif defect == "data_root":
        source["data"]["root"] = "/mnt/sfs/datasets/other"
    else:
        source["lora"][defect] = {
            "lora_type": "canonical_lora",
            "target_modules": ["linear_qkv"],
            "rank": 32,
            "alpha": 64,
            "init_method": "normal",
            "dropout": 0.1,
        }[defect]
        if defect == "lora_type":
            source["lora"]["type"] = source["lora"].pop("lora_type")
    with pytest.raises(ValueError):
        sft.compile_sft(source, relative_to=tmp_path)


@pytest.mark.parametrize(
    "defect",
    [
        "corpus_manifest",
        "split_manifest",
        "dataset_path",
        "dataset_digest",
        "dataset_task_keys",
        "dataset_rows_and_max_steps",
        "model_revision",
        "model_weight_manifest",
        "model_file_digest",
        "runtime_source_file",
        "qualification_gate",
        "unexpected_plan_field",
    ],
)
def test_qwen38_megatron_lora_rejects_mutated_compiled_identity(
    qwen38_lora_config, tmp_path, defect
):
    plan = sft.compile_sft(qwen38_lora_config, relative_to=tmp_path)
    if defect == "corpus_manifest":
        plan["corpus_manifest_sha256"] = "sha256:" + "0" * 64
    elif defect == "split_manifest":
        plan["split_manifest_sha256"] = "sha256:" + "0" * 64
    elif defect == "dataset_path":
        plan["datasets"]["train"]["path"] = "/mnt/sfs/datasets/other/train.parquet"
    elif defect == "dataset_digest":
        plan["datasets"]["train"]["sha256"] = "sha256:" + "0" * 64
    elif defect == "dataset_task_keys":
        plan["datasets"]["train"]["task_keys"].append("unreviewed-task")
    elif defect == "dataset_rows_and_max_steps":
        plan["datasets"]["train"]["rows"] += 1
        plan["recipe"]["max_steps"] += 1
    elif defect == "model_revision":
        plan["model"]["revision"] = "0" * 40
    elif defect == "model_weight_manifest":
        plan["model"]["weight_manifest_sha256"] = "sha256:" + "0" * 64
    elif defect == "model_file_digest":
        plan["model"]["files"][0]["sha256"] = "0" * 64
    elif defect == "runtime_source_file":
        source_files = plan["skyrl_runtime"]["source_files_sha256"]
        source_files[next(iter(source_files))] = "0" * 64
    elif defect == "qualification_gate":
        plan["qualification_gate"]["accepted_for_production"] = True
    elif defect == "unexpected_plan_field":
        plan["unreviewed"] = True
    with pytest.raises(ValueError):
        sft_runtime.validate_plan(plan, check_files=False)


@pytest.mark.parametrize("defect", ["rank", "alpha", "missing", "extra", "memory", "nodes", "cpu"])
def test_glm_rejects_unsafe_or_ambiguous_configuration(glm_config, tmp_path, defect):
    if defect == "missing":
        glm_config.pop("lora")
    elif defect == "memory":
        glm_config.pop("cluster")
    elif defect == "nodes":
        glm_config["recipe"]["nodes"] = 1
    elif defect == "cpu":
        glm_config["cluster"]["resources"]["cpu_request"] = "15999m"
    else:
        glm_config["lora"][defect] = 0
    with pytest.raises(ValueError):
        sft.compile_sft(glm_config, relative_to=tmp_path)


def test_glm_bootstrap_imports_worker_in_child_without_checkout(glm_config, tmp_path, monkeypatch):
    plan = sft.compile_sft(glm_config, relative_to=tmp_path)
    helper = Path(sft.__file__).with_name("glm_runtime.py").read_text()
    (tmp_path / "glm_runtime.py").write_text(helper)
    runtime = tmp_path / "sft_runtime.py"
    runtime.write_text(
        "import os,subprocess,sys\nfrom training import glm_runtime\n"
        "assert glm_runtime.__file__.startswith(os.environ['RUN_DIR']), glm_runtime.__file__\n"
        "subprocess.run([sys.executable,'-S','-c',"
        '"from training import glm_runtime; '
        "assert glm_runtime.TARGETS[0]=='q_a_proj'\"],check=True)\n"
    )
    monkeypatch.setattr(sft, "__file__", str(tmp_path / "sft.py"))
    plan["runtime_sha256"] = hashlib.sha256(runtime.read_bytes()).hexdigest()
    request = sft.job_request(plan)
    run = tmp_path / "run"
    run.mkdir()
    argv = shlex.split(request["command"])
    argv[0] = sys.executable
    # Exclude this desktop's editable-install import hook as well as the cwd.
    argv.insert(1, "-S")
    env = {
        **os.environ,
        **request["env"],
        "RUN_DIR": str(run),
        "PYTHONPATH": str(run / ".runtime"),
    }
    result = subprocess.run(argv, cwd=run, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (run / ".runtime/training/glm_runtime.py").read_text() == helper


def test_recovery_bootstrap_imports_complete_package_without_checkout(
    config, tmp_path, monkeypatch
):
    from training import recovery

    plan = sft.compile_sft(config[0], relative_to=tmp_path)
    # Bootstrap import boundary only; manifest validity has independent tests.
    plan["recovery"] = {"mode": "validate"}
    plan["recovery_runtime_sha256"] = recovery.digest(Path(recovery.__file__))
    monkeypatch.setattr(sft, "validate_plan", lambda *_args, **_kwargs: None)
    request = sft.job_request(plan)
    files = json.loads(gzip.decompress(base64.b64decode(request["env"]["CYBER_SFT_BUNDLE"])))
    assert files["extra_files"]["training/sft_runtime.py"] == files["runtime"]
    # Replace only the executable entrypoint, retaining the real recovery,
    # checkpoint and runtime modules for both parent and fresh child imports.
    entry = (
        "import subprocess,sys\nfrom training import recovery\n"
        "assert callable(recovery.worker_class)\n"
        "subprocess.run([sys.executable,'-S','-c',"
        "'from training import recovery; assert callable(recovery.load)'],check=True)\n"
    )
    original = Path.read_bytes
    runtime_file = Path(sft.__file__).with_name("sft_runtime.py")
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda p: entry.encode() if p == runtime_file else original(p),
    )
    plan["runtime_sha256"] = hashlib.sha256(entry.encode()).hexdigest()
    request = sft.job_request(plan)
    run = tmp_path / "run"
    run.mkdir()
    argv = shlex.split(request["command"])
    argv[:1] = [sys.executable, "-S"]
    env = {
        **os.environ,
        **request["env"],
        "RUN_DIR": str(run),
        "PYTHONPATH": str(run / ".runtime"),
    }
    result = subprocess.run(argv, cwd=run, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    plan["recovery_runtime_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="recovery runtime changed"):
        sft.job_request(plan)


@pytest.mark.parametrize(
    "section,key",
    [
        (None, "typo"),
        ("recipe", "learning_rate"),
        ("cluster", "queue_priority_class"),
        ("model", "revision"),
        ("data", "taskset"),
    ],
)
def test_unknown_fields_fail_instead_of_silently_ignoring_overrides(config, tmp_path, section, key):
    source, _, _ = config
    target = source if section is None else source.setdefault(section, {})
    target[key] = "unexpected"
    with pytest.raises(ValueError, match="unknown fields"):
        sft.compile_sft(source, relative_to=tmp_path)


@pytest.mark.parametrize(
    "key,value",
    [
        ("batch_size", 0),
        ("epochs", 1.5),
        ("nodes", True),
        ("batch_size", 7),
        ("lr", float("nan")),
        ("lr", 1),
        ("checkpoint_interval", 10),
        ("keep_checkpoints", 0),
    ],
)
def test_bad_recipe_fails_before_network(config, tmp_path, key, value):
    source, _, _ = config
    source["recipe"] = {key: value}
    with pytest.raises(ValueError):
        sft.compile_sft(source, relative_to=tmp_path)


@pytest.mark.parametrize(
    "defect",
    [
        "unsigned",
        "tokenizer",
        "task_overlap",
        "path_escape",
        "same_file",
        "zero_rows",
        "missing_identity",
    ],
)
def test_corpus_identity_and_split_gates(config, tmp_path, defect):
    source, manifest, save = config
    if defect == "unsigned":
        manifest["sha256"] = "wrong"
        (tmp_path / "corpus.json").write_text(json.dumps(manifest))
    else:
        if defect == "tokenizer":
            manifest["tokenizer"]["revision"] = "a" * 40
        elif defect == "task_overlap":
            manifest["files"]["dev"]["task_keys"] = ["train"]
        elif defect == "path_escape":
            manifest["files"]["train"]["path"] = "../outside"
        elif defect == "same_file":
            manifest["files"]["dev"]["path"] = "train.parquet"
        elif defect == "zero_rows":
            manifest["files"]["train"]["rows"] = 0
        else:
            manifest["files"]["dev"]["sha256"] = ""
        save(manifest)
    with pytest.raises(ValueError):
        sft.compile_sft(source, relative_to=tmp_path)


@pytest.mark.parametrize(
    "defect", ["revision", "shard", "sidecar", "duplicate", "unsupported_model"]
)
def test_model_must_match_full_file_inventory(config, tmp_path, defect):
    source, manifest, save = config
    lock = sft.read_mapping(Path(source["model"]["lock"]))
    if defect == "revision":
        lock["revision"] = "a" * 40
    elif defect == "shard":
        lock["weights"]["shards"] -= 1
    elif defect == "sidecar":
        lock["configuration"]["config_sha256"] = ""
    elif defect == "duplicate":
        lock["tokenizer"]["files"].append(copy.deepcopy(lock["tokenizer"]["files"][0]))
    else:
        lock["repo"] = "zai-org/GLM-5.3-Flash"
        manifest["tokenizer"]["repo"] = lock["repo"]
        save(manifest)
    (tmp_path / "model.json").write_text(json.dumps(lock))
    source["model"]["lock"] = "model.json"
    with pytest.raises(ValueError):
        sft.compile_sft(source, relative_to=tmp_path)


@pytest.mark.parametrize(
    "section,root",
    [
        ("model", "/mnt/sfs"),
        ("data", "/tmp/data"),
        ("data", "/mnt/sfs/datasets/../test"),
        ("model", "/mnt/sfs//models/test"),
    ],
)
def test_roots_are_explicit_and_canonical(config, tmp_path, section, root):
    source, _, _ = config
    source[section]["root"] = root
    with pytest.raises(ValueError):
        sft.compile_sft(source, relative_to=tmp_path)


def test_bootstrap_executes_bound_bytes_and_never_overwrites(config, tmp_path, monkeypatch):
    source, _, _ = config
    plan = sft.compile_sft(source, relative_to=tmp_path)
    fake_module = tmp_path / "compiler.py"
    runtime = tmp_path / "sft_runtime.py"
    runtime.write_text("import sys; print('synthetic-bootstrap', sys.argv[4])\n")
    monkeypatch.setattr(sft, "__file__", str(fake_module))
    with pytest.raises(ValueError, match="runtime changed"):
        sft.job_request(plan)
    plan["runtime_sha256"] = hashlib.sha256(runtime.read_bytes()).hexdigest()
    request = sft.job_request(plan)
    argv = shlex.split(request["command"])
    argv[0] = sys.executable
    output = tmp_path / "missing" / "run"
    env = {**os.environ, **request["env"], "RUN_DIR": str(output)}
    result = subprocess.run(argv, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "synthetic-bootstrap" in result.stdout
    assert (output / ".runtime/sft_runtime.py").read_bytes() == runtime.read_bytes()
    assert subprocess.run(argv, env=env, capture_output=True).returncode != 0
    tampered = tmp_path / "tampered"
    tampered.mkdir()
    env.update(RUN_DIR=str(tampered), CYBER_SFT_BUNDLE=base64.b64encode(b"tampered").decode())
    assert subprocess.run(argv, env=env, capture_output=True).returncode != 0
    assert not (tampered / ".runtime").exists()


@pytest.mark.parametrize("defect", [None, "gpu", "native", "file", "tokens"])
def test_cpu_preflight_checks_files_and_actual_target_accounting(
    config, tmp_path, monkeypatch, defect
):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    import transformers

    from training import sft_runtime

    source, _, _ = config
    plan = sft.compile_sft(source, relative_to=tmp_path)
    root = tmp_path / "model"
    root.mkdir()
    plan["model"]["root"] = str(root)
    for item in plan["model"]["files"]:
        (root / item["path"]).write_bytes(b"synthetic-file")
        item["sha256"] = hashlib.sha256(b"synthetic-file").hexdigest()
    for split, spec in plan["datasets"].items():
        rows = [
            {
                "messages": [{"role": "assistant", "content": "synthetic"}],
                "task_key": spec["task_keys"][i % len(spec["task_keys"])],
                "window_id": str(i),
                "token_count": 4 if defect != "tokens" else 5,
            }
            for i in range(spec["rows"])
        ]
        path = tmp_path / f"{split}.parquet"
        pq.write_table(pa.Table.from_pylist(rows), path)
        spec.update(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    calls = []

    def native(_plan=None):
        calls.append("native")
        if defect == "native":
            raise ValueError("native source mismatch")

    def tokenize(row, tokenizer, max_length):
        assert max_length is None  # silently truncating is never acceptable
        return {
            "input_ids": [1, 2, 3, 4],
            "attention_mask": [1] * 4,
            "num_actions": 2,
            "loss_mask": [1, 1],
        }

    monkeypatch.setitem(
        sys.modules,
        "skyrl.train.sft_trainer",
        SimpleNamespace(tokenize_chat_example=tokenize),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: defect == "gpu")
    monkeypatch.setattr(sft_runtime, "validate_runtime_sources", native)
    monkeypatch.setattr(sft_runtime, "build_runtime_configs", lambda p: calls.append("config"))
    monkeypatch.setattr(
        sft_runtime,
        "_validate_sft_forward_backward_adapter",
        lambda p: calls.append("forward_backward_signature"),
    )

    class Trainer:
        def load_dataset(self):
            return self._load_split("train")

        def load_eval_dataset(self):
            return self._load_split("dev")

    monkeypatch.setattr(sft_runtime, "_make_trainer_class", lambda: Trainer)

    def local_loader(path, **kwargs):
        assert path == str(root)
        assert kwargs == {"local_files_only": True, "trust_remote_code": False}
        calls.append("local_model")

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", local_loader)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", local_loader)
    if defect == "file":
        (root / "config.json").write_bytes(b"tampered")
    if defect:
        with pytest.raises(ValueError):
            sft.preflight(plan)
        if defect in {"gpu", "file"}:
            assert calls == []
    else:
        receipt = sft.preflight(plan)
        assert calls == [
            "native",
            "config",
            "forward_backward_signature",
            "local_model",
            "local_model",
        ]
        assert receipt["plan_sha256"] == digest(plan)
        assert receipt["request_sha256"] == digest(sft.job_request(plan))
        assert receipt["counts"] == {
            "train": {"rows": 17, "tasks": 1, "supervised_tokens": 34},
            "dev": {"rows": 2, "tasks": 2, "supervised_tokens": 4},
        }
        assert receipt["status"] == "passed" and receipt["gpus"] == 0
        assert "native_forward_backward_signature" in receipt["checked"]
