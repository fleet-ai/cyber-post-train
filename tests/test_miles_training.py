"""Launch boundaries use synthetic metadata, never real GPU or Fleet requests."""

import ctypes
import hashlib
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from typer.testing import CliRunner

from cyber_post_train import cli
from cyber_post_train.jobs import digest
from training import miles, miles_conversion
from training import miles_training as train
from training.rl_runtime import progress

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config(tmp_path):
    base = {
        "lock": str(ROOT / "configs/models/qwen38-27b-1d4bf0f2.lock.json"),
        "weights": str(ROOT / "configs/models/qwen38-27b-1d4bf0f2.weights.json"),
        "root": "/mnt/sfs/models/synthetic-qwen",
    }
    model = miles_conversion.compile_conversion(
        {"name": "synthetic-base", "output_root": "/mnt/sfs/jobs/synthetic-base", "model": base},
        relative_to=tmp_path,
    )["model"]
    data = {
        "schema": "cyber_miles_data_v1",
        "name": "synthetic-miles-train",
        "tokenizer": {k: model[k] for k in ("repo", "revision")},
        "template_sha256": "sha256:" + miles.TEMPLATE_SHA256,
        "limits": {"context_tokens": 98304, "response_tokens": 81920, "max_tokens_per_turn": 4096},
        "files": {
            k: {"path": k + ".jsonl", "rows": 1, "sha256": "sha256:" + "a" * 64}
            for k in ("train", "dev")
        },
    }
    data["sha256"] = "sha256:" + digest(data)
    (tmp_path / "data.json").write_text(json.dumps(data))
    cp = {
        "schema": "cyber_miles_checkpoint_v1",
        "model": model,
        "image": miles.IMAGE,
        "root": "/mnt/sfs/jobs/synthetic-base/torch-dist",
        "optimizer_steps": 0,
        "files": [],
    }
    cp["sha256"] = digest(cp)
    (tmp_path / "checkpoint.json").write_text(json.dumps(cp))
    return {
        "backend": "miles",
        "name": data["name"],
        "output_root": "/mnt/sfs/jobs/synthetic-rl",
        "model": base,
        "data": {"manifest": "data.json", "root": "/mnt/sfs/data/synthetic-rl"},
        "checkpoint": {
            "manifest": "checkpoint.json",
            "sha256": train._hash(tmp_path / "checkpoint.json"),
        },
        "recipe": {"steps": 2, "lr": 1e-6},
        "wandb": {"entity": "synthetic", "project": "synthetic", "run_id": "synthetic"},
    }


@pytest.fixture
def plan(config, tmp_path):
    return train.compile_rl(config, relative_to=tmp_path)


def test_compiler_and_portable_job_have_native_identity(plan):
    request = train.job_request(plan)
    assert request == train.job_request(plan)
    assert request["secrets"] == ["fleet-api", "wandb-api"]
    assert request["workers"] == 1 and request["gpus_per_worker"] == 8
    assert request["env"]["MILES_USE_LEGACY_ROLLOUT_V1"] == "0"
    assert request["env"]["WANDB_RUN_ID"] == "synthetic"
    assert request["env"]["WANDB_CONSOLE"] == "off"
    assert request["env"]["PYTHONPATH"] == "/root/Megatron-LM"
    assert request["priority_class"] == "c1" and request["requeueIfPreempted"] is False
    assert plan["arguments"]["steps"] == 2
    assert plan["arguments"]["model_root"] == plan["model"]["root"]
    assert plan["arguments"]["policy_identity_root"] == plan["model"]["root"]
    assert "FLEET_API_KEY" not in request["env"]
    assert "WANDB_API_KEY" not in request["env"]


def test_miles_plan_can_bind_one_cluster(config, tmp_path):
    config["cluster"] = {"target": "dev", "priority": "c1"}
    plan = train.compile_rl(config, relative_to=tmp_path)
    assert plan["execution"]["cluster_target"] == "dev"
    cli._require_prepared_cluster(plan, cli.Cluster.dev)
    with pytest.raises(ValueError, match="dev-cluster-only"):
        cli._require_prepared_cluster(plan, cli.Cluster.prod)


def test_miles_rejects_unknown_cluster_target(config, tmp_path):
    config["cluster"] = {"target": "other"}
    with pytest.raises(ValueError, match="cluster target"):
        train.compile_rl(config, relative_to=tmp_path)


def test_miles_rejects_priority_reason_at_config_and_plan_boundaries(config, tmp_path):
    config["cluster"] = {"priority": "c1", "priority_reason": "override"}
    with pytest.raises(ValueError, match="unknown fields in cluster"):
        train.compile_rl(config, relative_to=tmp_path)

    config["cluster"] = {"priority": "c1"}
    compiled = train.compile_rl(config, relative_to=tmp_path)
    compiled["execution"]["priority_reason"] = "override"
    with pytest.raises(ValueError, match="Miles plan/runtime drift"):
        train.job_request(compiled)


def test_two_by_four_layout_is_rejected_before_request(config, tmp_path):
    config["recipe"].update({"nodes": 2, "gpus_per_node": 4})
    with pytest.raises(ValueError, match="unsupported Qwen Miles node/GPU layout"):
        train.compile_rl(config, relative_to=tmp_path)


def test_miles_has_independent_colocated_memory_defaults(plan):
    from training.sft import RESOURCES as sft_resources

    request = train.job_request(plan)
    assert request["resources"] == train.RESOURCES
    assert request["resources"] is not sft_resources
    assert request["resources"]["memory_request"] == "1536Gi"
    assert request["resources"]["memory_limit"] == "2048Gi"
    assert sft_resources["memory_limit"] == "768Gi"


@pytest.mark.parametrize(
    "resources",
    [
        {"memory_request": "512Gi", "memory_limit": "768Gi"},
        {"memory_request": "512Gi"},
        {"memory_limit": "768Gi"},
        {"cpu_request": "31"},
    ],
)
def test_miles_rejects_underreserved_or_known_oom_shape(config, tmp_path, resources):
    config["cluster"] = {"resources": resources}
    with pytest.raises(ValueError, match="colocated RAM"):
        train.compile_rl(config, relative_to=tmp_path)


def test_miles_allows_larger_explicit_resources(config, tmp_path):
    config["cluster"] = {"resources": {"memory_request": "2Ti", "memory_limit": "2304Gi"}}
    request = train.job_request(train.compile_rl(config, relative_to=tmp_path))
    assert request["resources"]["memory_request"] == "2Ti"
    assert request["resources"]["memory_limit"] == "2304Gi"


def test_miles_allows_peer_proven_cpu_and_memory_envelope(config, tmp_path):
    config["cluster"] = {
        "resources": {
            "cpu_request": "32",
            "cpu_limit": "32",
            "memory_request": "1800Gi",
            "memory_limit": "2400Gi",
        }
    }
    request = train.job_request(train.compile_rl(config, relative_to=tmp_path))
    assert request["resources"] == config["cluster"]["resources"]


@pytest.mark.parametrize(
    "fault",
    [
        "unknown",
        "backend",
        "checkpoint_hash",
        "checkpoint_schema",
        "trained_source",
        "data_name",
        "data_schema",
        "file_escape",
        "groups",
        "priority",
        "memory",
        "overlap",
        "runtime",
    ],
)
def test_invalid_bindings_rejected_before_native_import(config, tmp_path, fault):
    if fault == "unknown":
        config["extra_args"] = "--no-save-optim"
    elif fault == "backend":
        config["backend"] = "skyrl"
    elif fault == "checkpoint_hash":
        config["checkpoint"]["sha256"] = "b" * 64
    elif fault in {"checkpoint_schema", "trained_source"}:
        path = tmp_path / "checkpoint.json"
        value = json.loads(path.read_text())
        value["schema" if fault == "checkpoint_schema" else "optimizer_steps"] = "invalid"
        value["sha256"] = digest({k: v for k, v in value.items() if k != "sha256"})
        path.write_text(json.dumps(value))
        config["checkpoint"]["sha256"] = train._hash(path)
    elif fault in {"data_name", "data_schema", "file_escape"}:
        path = tmp_path / "data.json"
        value = json.loads(path.read_text())
        if fault == "file_escape":
            value["files"]["train"]["path"] = "../train.jsonl"
        else:
            value["name" if fault == "data_name" else "schema"] = "changed"
        value["sha256"] = "sha256:" + digest({k: v for k, v in value.items() if k != "sha256"})
        path.write_text(json.dumps(value))
    elif fault == "groups":
        config["recipe"]["groups"] = 2
    elif fault == "priority":
        config["cluster"] = {"priority": "c0"}
    elif fault == "memory":
        config["cluster"] = {"resources": {"memory_request": "32Gi"}}
    elif fault == "overlap":
        config["output_root"] = config["model"]["root"]
    else:
        result = train.compile_rl(config, relative_to=tmp_path)
        result["runtime_sha256"] = "b" * 64
        with pytest.raises(ValueError):
            train.job_request(result)
        return
    with pytest.raises(ValueError):
        train.compile_rl(config, relative_to=tmp_path)


@pytest.fixture
def execution(plan, tmp_path, monkeypatch):
    root = tmp_path / "output"
    root.mkdir()
    plan["output_root"] = str(root)
    plan["arguments"]["output_root"] = str(root)
    plan["arguments"]["steps"] = 1
    monkeypatch.setenv("RUN_DIR", str(root))
    monkeypatch.setattr(train, "check_artifacts", lambda p: {})
    monkeypatch.setattr(train, "native_source", lambda: Path("synthetic.py"))
    calls = []
    monkeypatch.setattr(os, "killpg", lambda pid, sig: calls.append((pid, sig)))
    return plan, root, calls


def batches(plan, root):
    count = plan["arguments"]["steps"]
    names = [
        "dev-baseline-r0",
        *(f"train-r{i}" for i in range(count)),
        *(
            f"dev-after-r{i}"
            for i in range(count)
            if (i + 1) % plan["arguments"]["eval_interval"] == 0
        ),
    ]
    for name in names:
        directory = root / "episodes/batches" / name
        directory.mkdir(parents=True)
        value = {
            "schema": "cyber_miles_batch_v1",
            "batch_id": name,
            "data_sha256": plan["data"]["sha256"],
        }
        value["sha256"] = "sha256:" + digest(value)
        (directory / "COLLECTED.json").write_text(json.dumps(value))


def fake_process(monkeypatch, root, *, code=0, pointer="0", plan=None):
    def create(*args, **kwargs):
        assert kwargs["start_new_session"] is True
        assert kwargs["stderr"] == subprocess.STDOUT
        if pointer is not None:
            (root / "checkpoints").mkdir()
            (root / "checkpoints/latest_checkpointed_iteration.txt").write_text(pointer)
        if plan is not None:
            batches(plan, root)
        return NS(pid=9876543, poll=lambda: code, wait=lambda **kw: code, returncode=code)

    monkeypatch.setattr(subprocess, "Popen", create)


def test_first_native_checkpoint_is_rollout_zero_not_optimizer_one(execution, monkeypatch):
    plan, root, calls = execution
    fake_process(monkeypatch, root, plan=plan)
    result = train.run(plan, root / "plan.json")
    assert result["checkpoint_rollout_index"] == 0
    assert result["optimizer_update_independently_verified"] is False
    assert result["checkpoint_reload_verified"] is False
    assert not (root / "FAILED.json").exists()
    assert not (root / "ACCEPTED.json").exists()
    assert len(calls) == 2 and all(c[0] == 9876543 for c in calls)
    assert (root / "private-miles.log").stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("code,pointer", [(1, None), (0, "1"), (0, None)])
def test_unexpected_exit_or_checkpoint_defect_is_not_success(execution, monkeypatch, code, pointer):
    plan, root, calls = execution
    fake_process(monkeypatch, root, code=code, pointer=pointer, plan=plan)
    with pytest.raises(RuntimeError, match="no automatic retry"):
        train.run(plan, root / "plan.json")
    assert (root / "FAILED.json").is_file()
    assert not (root / "NATIVE_TRAINING_COMPLETE.json").exists()
    assert len(calls) == 2


@pytest.mark.parametrize(
    "fault",
    [None, "nonzero", "digest", "plan", "reason", "failure", "complete", "cleanup"],
)
def test_rejected_native_child_is_not_training_completion(execution, monkeypatch, fault):
    from training import rl_runtime
    from training.rl_episode import EpisodeBudgetExceeded

    plan, root, calls = execution

    def spawn(*args, **kwargs):
        rl_runtime.native_rejection(plan, EpisodeBudgetExceeded("generation_incomplete_length"))
        path = root / "NATIVE_REJECTED.json"
        receipt = json.loads(path.read_bytes())
        if fault == "plan":
            receipt["plan_sha256"] = "b" * 64
        elif fault == "reason":
            receipt["reason"] = "unexpected_engine_error"
        receipt["sha256"] = digest({k: v for k, v in receipt.items() if k != "sha256"})
        if fault == "digest":
            receipt["sha256"] = "b" * 64
        path.write_text(json.dumps(receipt))
        if fault in {"failure", "complete"}:
            (
                root
                / ("NATIVE_FAILURE.json" if fault == "failure" else "NATIVE_TRAINING_COMPLETE.json")
            ).write_text("{}")

        def wait(**kwargs):
            if fault == "cleanup":
                raise subprocess.TimeoutExpired("synthetic", 30)
            return 1 if fault == "nonzero" else 0

        return NS(pid=9876543, poll=lambda: 0, wait=wait, returncode=1 if fault == "nonzero" else 0)

    monkeypatch.setattr(subprocess, "Popen", spawn)
    monkeypatch.setattr(train, "native_result", lambda _: pytest.fail("rejection cannot train"))
    if fault is None:
        result = train.run(plan, root / "plan.json")
        assert result["status"] == "rejected" and not (root / "FAILED.json").exists()
    else:
        with pytest.raises(RuntimeError, match="no automatic retry"):
            train.run(plan, root / "plan.json")
        assert (root / "FAILED.json").exists() and not (root / "REJECTED.json").exists()
    assert not (root / "ACCEPTED.json").exists() and not (root / "checkpoints").exists()
    assert len(calls) == 2 and {pid for pid, _ in calls} == {9876543}


def test_existing_intent_is_never_replayed(execution, monkeypatch):
    plan, root, _ = execution
    (root / "STARTED.json").write_text("preserve")
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: pytest.fail("replayed"))
    with pytest.raises(FileExistsError):
        train.run(plan, root / "plan.json")
    assert (root / "STARTED.json").read_text() == "preserve"


@pytest.mark.parametrize("fault", ["binding", "checkpoints", "episodes"])
def test_wrong_output_or_existing_progress_never_starts(execution, monkeypatch, fault):
    plan, root, calls = execution
    if fault == "binding":
        monkeypatch.delenv("RUN_DIR")
    else:
        (root / fault).mkdir()
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: pytest.fail("unexpected launch"))
    with pytest.raises((ValueError, FileExistsError)):
        train.run(plan, root / "plan.json")
    assert not calls and not (root / "STARTED.json").exists()


def test_compile_rejects_data_path_traversal(config, tmp_path):
    path = tmp_path / "data.json"
    data = json.loads(path.read_text())
    data["files"]["train"]["path"] = "other/train.jsonl"
    data["sha256"] = "sha256:" + digest({k: v for k, v in data.items() if k != "sha256"})
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="directly inside"):
        train.compile_rl(config, relative_to=tmp_path)


def test_plan_and_cli_roundtrip_are_offline(config, tmp_path, monkeypatch):
    source = tmp_path / "rl.json"
    source.write_text(json.dumps(config))
    output = tmp_path / "prepared"
    monkeypatch.setattr(cli, "_client", lambda: pytest.fail("offline compile used network"))
    result = CliRunner().invoke(cli.app, ["rl", str(source), "--output", str(output)])
    assert result.exit_code == 0, result.output
    plan, request = cli._prepared(output)
    assert plan["schema"] == train.SCHEMA and request == train.job_request(plan)
    assert json.loads(result.output)["submitted"] is False
    assert CliRunner().invoke(cli.app, ["rl", str(source), "--output", str(output)]).exit_code == 2


def _long_runtime_receipt(image, *, build_source_sha256="sha256:" + "b" * 64):
    value = {
        "schema": "cyber_miles_opencode_runtime_qualification_v1",
        "status": "image_qualified_for_dev",
        "image": image,
        "base_image": train.LONG_RUNTIME_BASE_IMAGE,
        "fti_version": "0.8.4",
        "native_profile": "qwen3.8-27b-256k",
        "model_config_sha256": (
            "sha256:191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab"
        ),
        "opencode_version": "1.18.27",
        "opencode_source_commit": "4b7e19e315cca414121ba1d61523fef74bb3ae8b",
        "opencode_binary_sha256": (
            "sha256:bddf894e5c2bc3d8cf452bd6e5ab2273bbe4a37eeeb9aec848d3d7d20db1f256"
        ),
        "miles_source_commit": "2799fe386320c156334bf763ad4d7ca0f85dca4e",
        "miles_tree_source_sha256": (
            "sha256:fd978a1ef2617f4bf30850fedd197e546cdc9c6542b00b03df502cbb285fc732"
        ),
        "native_driver_sha256": "sha256:" + miles.LONG_NATIVE_DRIVER_SHA256,
        "native_converter_sha256": "sha256:" + miles.LONG_NATIVE_CONVERTER_SHA256,
        "installed_session_tree_sha256": ("sha256:" + miles.LONG_INSTALLED_SESSION_TREE_SHA256),
        "build_source_sha256": build_source_sha256,
        "checks": {key: True for key in train.LONG_RUNTIME_CHECKS},
    }
    return {**value, "sha256": digest(value)}


def test_long_compiler_binds_four_node_image_data_checkpoint_and_session(config, tmp_path):
    from training import miles_opencode

    image = "registry.test/miles-opencode@sha256:" + "a" * 64
    root = ROOT / "training/images/miles-opencode-long-context"
    build_source = "sha256:" + digest(
        {
            str(path.relative_to(ROOT)): path.read_text()
            for path in sorted(root.glob("*"))
            if path.is_file()
        }
    )
    receipt = _long_runtime_receipt(image, build_source_sha256=build_source)
    runtime_path = tmp_path / "runtime.json"
    runtime_path.write_text(json.dumps(receipt))
    data_path = tmp_path / "data.json"
    data = json.loads(data_path.read_text())
    data.update(
        {
            "schema": "cyber_miles_data_v2",
            "limits": {
                "context_tokens": miles_opencode.CONTEXT_TOKENS,
                "response_tokens": miles_opencode.TOTAL_RESPONSE_TOKENS,
                "max_tokens_per_turn": miles_opencode.MAX_TOKENS_PER_TURN,
                "max_turns": miles_opencode.MAX_MODEL_REQUESTS,
                "episode_seconds": miles_opencode.EPISODE_SECONDS,
                "tool_seconds": 300,
            },
            "harness": miles_opencode.harness_contract(),
        }
    )
    data["sha256"] = "sha256:" + digest({k: v for k, v in data.items() if k != "sha256"})
    data_path.write_text(json.dumps(data))
    checkpoint_path = tmp_path / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    checkpoint["image"] = image
    checkpoint["sha256"] = digest({k: v for k, v in checkpoint.items() if k != "sha256"})
    checkpoint_path.write_text(json.dumps(checkpoint))
    config["checkpoint"]["sha256"] = train._hash(checkpoint_path)
    config["recipe"].update(
        {
            "nodes": 4,
            "gpus_per_node": 8,
            "max_tokens_per_gpu": 65536,
            "native_profile": "qwen3.8-27b-256k",
            "harness": "opencode",
            "session_node_cap": 4096,
        }
    )
    config["cluster"] = {"target": "dev", "priority": "c1"}
    config["runtime"] = {
        "image": image,
        "receipt": str(runtime_path),
        "sha256": "sha256:" + train._hash(runtime_path),
    }

    result = train.compile_rl(config, relative_to=tmp_path)
    request = train.job_request(result)

    assert result["schema"] == train.LONG_CONTEXT_SCHEMA
    assert result["runtime_qualification"] == receipt
    assert result["native_driver_sha256"] == miles.LONG_NATIVE_DRIVER_SHA256
    assert train.watchdog_hard_seconds(result) == miles_opencode.JOB_HARD_SECONDS
    assert request["image"] == image
    assert request["workers"] == 4 and request["gpus_per_worker"] == 8
    assert request["env"]["MILES_SESSION_MAX_NODES"] == "4096"
    assert request["priority_class"] == "c1"


def test_long_runtime_receipt_is_image_qualification_not_circular_rl_acceptance():
    image = "registry.test/miles-opencode@sha256:" + "a" * 64
    receipt = _long_runtime_receipt(image)
    assert train._validate_long_runtime_receipt(receipt, image=image) is receipt
    assert "finite_nonzero_update_checked" not in receipt["checks"]
    assert "zero_update_reload_checked" not in receipt["checks"]


def test_long_runtime_qualification_template_matches_source_and_contract():
    from training import miles_opencode

    path = (
        Path(__file__).parents[1]
        / "configs/qualification/qwen38-miles-opencode-long-context-runtime-v1.template.json"
    )
    template = json.loads(path.read_text())
    root = Path(__file__).parents[1]
    build_files = {
        str(source.relative_to(root)): source.read_text()
        for source in sorted((root / "training/images/miles-opencode-long-context").glob("*"))
        if source.is_file()
    }
    assert template["status"] == "not_run" and template["production_allowed"] is False
    assert template["derived_image"] is None
    assert template["sources"]["build_source_sha256"] == "sha256:" + digest(build_files)
    assert template["sources"]["native_driver_sha256"] == (
        "sha256:" + miles.LONG_NATIVE_DRIVER_SHA256
    )
    assert template["sources"]["native_converter_sha256"] == (
        "sha256:" + miles.LONG_NATIVE_CONVERTER_SHA256
    )
    assert template["sources"]["installed_session_tree_sha256"] == (
        "sha256:" + miles.LONG_INSTALLED_SESSION_TREE_SHA256
    )
    assert template["contract"] == {
        "model": "Qwen/Qwen3.8-27B",
        "tito_family": miles_opencode.TITO_FAMILY,
        "reasoning_parser": "qwen3",
        "tool_call_parser": "qwen3_coder",
        "chat_template_sha256": "sha256:" + miles_opencode.TEMPLATE_SHA256,
        "sglang_router_policy": "consistent_hashing",
        "nodes": 4,
        "gpus_per_node": 8,
        "context_tokens": miles_opencode.CONTEXT_TOKENS,
        "response_tokens": miles_opencode.TOTAL_RESPONSE_TOKENS,
        "max_tokens_per_model_call": miles_opencode.MAX_TOKENS_PER_TURN,
        "max_model_requests": miles_opencode.MAX_MODEL_REQUESTS,
        "job_hard_seconds": miles_opencode.JOB_HARD_SECONDS,
        "episode_seconds": miles_opencode.EPISODE_SECONDS,
        "environment_ttl_seconds": miles_opencode.INSTANCE_TTL_SECONDS,
        "compaction_threshold_tokens": miles_opencode.COMPACTION_THRESHOLD_TOKENS,
        "compaction_reserved_tokens": miles_opencode.COMPACTION_RESERVED_TOKENS,
        "compaction_summary_tokens": miles_opencode.SUMMARY_MAX_TOKENS,
        "preserve_recent_tokens": miles_opencode.PRESERVE_RECENT_TOKENS,
        "session_node_cap": miles_opencode.SESSION_NODE_CAP,
        "summary_token_treatment": miles_opencode.SUMMARY_TOKEN_TREATMENT,
        "primary_tool_result_prefix_truncation": False,
    }


@pytest.mark.parametrize("fault", ["image", "check", "extra_check", "source", "digest"])
def test_long_runtime_receipt_fails_closed(fault):
    image = "registry.test/miles-opencode@sha256:" + "a" * 64
    receipt = _long_runtime_receipt(image)
    if fault == "image":
        receipt["image"] = "registry.test/other@sha256:" + "a" * 64
    elif fault == "check":
        receipt["checks"]["native_256k_parser_checked"] = False
    elif fault == "extra_check":
        receipt["checks"]["finite_nonzero_update_checked"] = True
    elif fault == "source":
        receipt["build_source_sha256"] = "sha256:" + "c" * 64
    else:
        receipt["sha256"] = "0" * 64
    if fault != "digest":
        receipt["sha256"] = digest({k: v for k, v in receipt.items() if k != "sha256"})
    with pytest.raises(ValueError):
        train._validate_long_runtime_receipt(
            receipt,
            image=image,
            build_source_sha256="sha256:" + "b" * 64,
        )


@pytest.fixture
def artifacts(plan, tmp_path, monkeypatch):
    from evals.fleet import opencode_self_hosted as fleet
    from training.rl_data import AUTHORITY

    monkeypatch.setattr(train, "check_inputs", lambda value: None)
    root = tmp_path / "checkpoint"
    (root / "release").mkdir(parents=True)
    for name, content in {
        "latest_checkpointed_iteration.txt": b"release\n",
        "release/.metadata": b"synthetic metadata",
        "release/__0.distcp": b"synthetic tensor",
    }.items():
        (root / name).write_bytes(content)
    cp = plan["checkpoint"]
    cp["root"] = str(root)
    cp["files"] = [{**f, "sha256": train._hash(root / f["path"])} for f in train.inventory(root)]
    cp["sha256"] = digest({k: v for k, v in cp.items() if k != "sha256"})
    directory = tmp_path / "data"
    directory.mkdir()
    plan["arguments"]["data_manifest"] = str(directory / "manifest.json")
    data = plan["data"]
    data["tool_catalog_sha256"] = "sha256:" + "c" * 64
    data["limits"].update(max_turns=4, episode_seconds=60, tool_seconds=5, tool_result_chars=100)
    for group in ("train", "dev"):
        cfg = {
            "run_id": plan["run_name"],
            "model": {k: plan["model"][k] for k in ("root", "repo", "revision")},
            "authority": AUTHORITY,
            "environment": {"ttl_seconds": 3600},
            "execution": {
                "required_task_tools": ["bash", "submit_report"],
                "required_task_tool_catalog_sha256": data["tool_catalog_sha256"],
            },
            "rl": {k: v for k, v in data["limits"].items() if k != "response_tokens"},
            "task": {"key": "synthetic-" + group, "version_id": "synthetic-version"},
            "initial_prompt_sha256": fleet.sha256(b"synthetic prompt"),
        }
        cfg["config_sha256"] = fleet.digest_without(cfg, "config_sha256")
        row = {
            "input": "synthetic prompt",
            "metadata": {
                "cyber_config": cfg,
                "split": group,
                "lineage": {"application": "synthetic", "task_family": group},
            },
        }
        path = directory / (group + ".jsonl")
        path.write_text(json.dumps(row) + "\n")
        plan["arguments"][group + "_data"] = str(path)
        data["files"][group]["sha256"] = "sha256:" + train._hash(path)
    (directory / "manifest.json").write_text(json.dumps(data))
    return plan, root, directory


def test_artifacts_are_read_only_and_return_both_splits(artifacts):
    plan, root, _ = artifacts
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    rows = train.check_artifacts(plan)
    assert {k: len(v) for k, v in rows.items()} == {"train": 1, "dev": 1}
    assert before == {p: p.read_bytes() for p in before}


def _bind_sft_initial_policy(plan, *, runtime_stage=False):
    accepted_root = plan["model"]["root"]
    initial_policy = {
        "kind": "sft_hf_export",
        "sft_optimizer_step": 44,
        "accepted_root": accepted_root,
    }
    if runtime_stage:
        runtime_root = "/mnt/sfs/jobs/synthetic-sft-runtime/payload"
        initial_policy["runtime_stage"] = {"path": "/mnt/sfs/jobs/synthetic-sft-runtime/STAGE.json"}
        plan["model"]["root"] = runtime_root
        plan["arguments"]["model_root"] = runtime_root
    plan["model"]["initial_policy"] = initial_policy
    plan["checkpoint"]["model"] = plan["model"]
    plan["checkpoint"]["sha256"] = digest(
        {key: value for key, value in plan["checkpoint"].items() if key != "sha256"}
    )
    return accepted_root, initial_policy


def _set_episode_model_root(plan, directory, root):
    for split in ("train", "dev"):
        path = directory / f"{split}.jsonl"
        row = json.loads(path.read_text())
        config = row["metadata"]["cyber_config"]
        config["model"]["root"] = root
        from evals.fleet import opencode_self_hosted as fleet

        config["config_sha256"] = fleet.digest_without(config, "config_sha256")
        path.write_text(json.dumps(row) + "\n")
        plan["data"]["files"][split]["sha256"] = "sha256:" + train._hash(path)
    plan["data"]["sha256"] = "sha256:" + digest(
        {key: value for key, value in plan["data"].items() if key != "sha256"}
    )
    (directory / "manifest.json").write_text(json.dumps(plan["data"]))


def test_sft_initial_policy_without_runtime_stage_uses_accepted_root(artifacts):
    plan, _, _ = artifacts
    accepted_root, initial_policy = _bind_sft_initial_policy(plan)
    rows = train.check_artifacts(plan)
    assert {key: len(value) for key, value in rows.items()} == {"train": 1, "dev": 1}
    assert plan["checkpoint"]["model"]["initial_policy"] == initial_policy
    assert plan["arguments"]["model_root"] == accepted_root


def test_staged_sft_keeps_episode_identity_at_accepted_root(artifacts):
    plan, _, _ = artifacts
    accepted_root, _ = _bind_sft_initial_policy(plan, runtime_stage=True)
    rows = train.check_artifacts(plan)
    assert {key: len(value) for key, value in rows.items()} == {"train": 1, "dev": 1}
    assert plan["arguments"]["model_root"] == plan["model"]["root"]
    assert plan["arguments"]["model_root"] != accepted_root
    assert plan["arguments"]["policy_identity_root"] == accepted_root
    assert {
        row["metadata"]["cyber_config"]["model"]["root"]
        for values in rows.values()
        for row in values
    } == {accepted_root}


@pytest.mark.parametrize(
    "fault,error",
    [
        ("episode_uses_runtime", "episode identity changed"),
        ("runtime_argument", "runtime model identity changed"),
        ("policy_argument", "policy identity changed"),
        ("missing_stage", "unstaged SFT runtime identity changed"),
    ],
)
def test_sft_episode_and_runtime_root_mismatches_fail_closed(artifacts, fault, error):
    plan, _, directory = artifacts
    accepted_root, initial_policy = _bind_sft_initial_policy(plan, runtime_stage=True)
    if fault == "episode_uses_runtime":
        _set_episode_model_root(plan, directory, plan["model"]["root"])
    elif fault == "runtime_argument":
        plan["arguments"]["model_root"] += "-other"
    elif fault == "policy_argument":
        plan["arguments"]["policy_identity_root"] = plan["model"]["root"]
    else:
        initial_policy.pop("runtime_stage")
        plan["checkpoint"]["sha256"] = digest(
            {key: value for key, value in plan["checkpoint"].items() if key != "sha256"}
        )
    with pytest.raises(ValueError, match=error):
        train.check_artifacts(plan)
    assert accepted_root != plan["model"]["root"]


def test_training_checkpoint_cannot_masquerade_as_base(artifacts):
    plan, root, _ = artifacts
    (root / "latest_checkpointed_iteration.txt").write_text("1\n")
    with pytest.raises(ValueError, match="release checkpoint"):
        train.check_artifacts(plan)


@pytest.mark.parametrize(
    "fault", ["cp_extra", "cp_bytes", "data_manifest", "data_bytes", "count", "split", "prompt"]
)
def test_artifact_tampering_stops_before_native_loader(artifacts, fault):
    plan, root, directory = artifacts
    if fault == "cp_extra":
        (root / "extra").write_text("extra")
    elif fault == "cp_bytes":
        (root / "release/__0.distcp").write_bytes(b"different tensor")
    elif fault == "data_manifest":
        (directory / "manifest.json").write_text("{}")
    elif fault == "data_bytes":
        (directory / "train.jsonl").write_text("different")
    else:
        path = directory / "train.jsonl"
        row = json.loads(path.read_text())
        if fault == "split":
            row["metadata"]["split"] = "dev"
        elif fault == "prompt":
            row["input"] = "changed"
        else:
            plan["data"]["files"]["train"]["rows"] = 2
        path.write_text(json.dumps(row) + "\n")
        plan["data"]["files"]["train"]["sha256"] = "sha256:" + train._hash(path)
        (directory / "manifest.json").write_text(json.dumps(plan["data"]))
    with pytest.raises(ValueError):
        train.check_artifacts(plan)


def test_native_source_is_exact(tmp_path, monkeypatch):
    path = tmp_path / "train.py"
    path.write_text("synthetic native source")
    monkeypatch.setitem(
        sys.modules, "miles.utils.external_utils.command_utils", NS(repo_base_dir=tmp_path)
    )
    monkeypatch.setattr(train, "NATIVE_DRIVER_SHA256", train._hash(path))
    assert train.native_source() == path
    path.write_text("changed")
    with pytest.raises(ValueError):
        train.native_source()


def test_long_native_source_uses_only_the_new_image_digest(tmp_path, monkeypatch):
    path = tmp_path / "train.py"
    path.write_text("synthetic FTI 0.8.4 native source")
    monkeypatch.setitem(
        sys.modules, "miles.utils.external_utils.command_utils", NS(repo_base_dir=tmp_path)
    )
    monkeypatch.setattr(miles, "LONG_NATIVE_DRIVER_SHA256", train._hash(path))
    plan = {"arguments": {"harness": "opencode"}}
    assert train.native_source_for_plan(plan) == path
    monkeypatch.setattr(train, "NATIVE_DRIVER_SHA256", train._hash(path))
    path.write_text("changed")
    with pytest.raises(ValueError):
        train.native_source_for_plan(plan)


def test_native_namespace_uses_request_environment_and_fails_before_gpu(tmp_path):
    namespace = tmp_path / "megatron/post_training"
    namespace.mkdir(parents=True)
    (namespace / "checkpointing.py").write_text("raise AssertionError('must not import CUDA')")
    (tmp_path / "model_provider.py").write_text("raise AssertionError('must not import CUDA')")
    train.check_native_namespace(str(tmp_path))
    (namespace / "checkpointing.py").unlink()
    with pytest.raises(ValueError, match="source namespace unavailable"):
        train.check_native_namespace(str(tmp_path))


def test_parsed_native_semantics_and_argv_restoration(plan, monkeypatch):
    monkeypatch.setattr(train, "native_source", lambda _plan=None: Path("synthetic.py"))
    monkeypatch.setattr(miles, "arguments", lambda cfg: ["--chat-template-path", "synthetic"])
    args = NS(
        data_source_path="training.miles_text.TextDataSource",
        tool_key="tools",
        start_rollout_id=0,
        load=plan["checkpoint"]["root"],
        ref_load=plan["checkpoint"]["root"],
        hf_checkpoint=plan["model"]["root"],
        fleet_policy_identity_root=plan["arguments"]["policy_identity_root"],
        num_rollout=2,
        num_steps_per_rollout=1,
        global_batch_size=2,
        calculate_per_token_loss=True,
        grpo_std_normalization=False,
    )
    monkeypatch.setitem(sys.modules, "miles.utils.arguments", NS(parse_args=lambda: args))
    before = sys.argv
    assert train.native_args(plan) is args and sys.argv is before
    for key, value in (
        ("start_rollout_id", 1),
        ("calculate_per_token_loss", False),
        ("grpo_std_normalization", True),
    ):
        original = getattr(args, key)
        setattr(args, key, value)
        with pytest.raises(ValueError):
            train.native_args(plan)
        setattr(args, key, original)
    assert sys.argv is before


@pytest.fixture
def preflight_inputs(artifacts, tmp_path, monkeypatch):
    from training import rl_data

    plan, _, directory = artifacts
    plan["output_root"] = str(tmp_path / "new-run")
    for name, key in (("split", "split_sha256"), ("task-set", "selection_sha256")):
        plan["data"][key] = name + "-digest"
        (directory / (name + ".json")).write_text(json.dumps({"sha256": name + "-digest"}))
    (directory / "manifest.json").write_text(json.dumps(plan["data"]))
    rows = train.check_artifacts(plan)
    expected = [
        {
            "split": split,
            "task_key": row["metadata"]["cyber_config"]["task"]["key"],
            "task_version_id": "synthetic-version",
            "lineage": row["metadata"]["lineage"],
        }
        for split, values in rows.items()
        for row in values
    ]
    monkeypatch.setattr(train, "job_request", lambda p: {"env": {"PYTHONPATH": "synthetic"}})
    monkeypatch.setattr(train, "check_native_namespace", lambda path: None)
    monkeypatch.setattr(train, "check_artifacts", lambda p: rows)
    monkeypatch.setattr(rl_data, "selection", lambda *a: expected)
    monkeypatch.setattr(miles, "arguments", lambda cfg: ["--chat-template-path", "synthetic"])
    monkeypatch.setattr(miles, "TEMPLATE_SHA256", hashlib.sha256(b"synthetic").hexdigest())
    monkeypatch.setattr(train, "native_source", lambda: Path("synthetic.py"))
    monkeypatch.setitem(sys.modules, "torch", NS(cuda=NS(is_available=lambda: False)))
    monkeypatch.setitem(
        sys.modules, "transformers", NS(AutoTokenizer=NS(from_pretrained=lambda *a, **kw: NS()))
    )

    class Dataset:
        def __init__(self, path, *args, **kw):
            assert kw["apply_chat_template"] is False
            self.origin_samples = [
                NS(prompt=row["input"], metadata=row["metadata"]) for row in rows[Path(path).stem]
            ]

        def __len__(self):
            return len(self.origin_samples)

    monkeypatch.setitem(sys.modules, "miles.utils.data", NS(Dataset=Dataset))

    class TextDataSource:
        def __init__(self, args):
            from miles.utils.data import Dataset

            assert args.rollout_global_dataset and not args.apply_chat_template
            assert args.multimodal_keys is None and args.chat_template_path == "synthetic"
            assert args.tool_key == "tools"
            self.tokenizer = NS(chat_template="synthetic")
            self.dataset = Dataset(args.prompt_data, apply_chat_template=False)

    monkeypatch.setitem(sys.modules, "training.miles_text", NS(TextDataSource=TextDataSource))
    return plan, rows, expected, directory


def test_preflight_checks_split_native_parsing_and_dataset_without_qualification(preflight_inputs):
    plan, _, _, _ = preflight_inputs
    proof = train.preflight(plan)
    assert proof["counts"] == {"train": 1, "dev": 1}
    assert proof["rl_qualified"] is False and proof["gpus"] == 0
    assert proof["plan_sha256"] == digest(plan)
    assert proof["planned_steps"] == proof["planned_global_batch"] == 2
    assert proof["native_parser_checked"] is False
    assert proof["native_text_source_checked"] is True
    assert proof["native_megatron_namespace_checked"] is True


@pytest.mark.parametrize(
    "fault",
    [
        "gpu",
        "output",
        "split",
        "task-set",
        "duplicate",
        "lineage",
        "filtered",
        "prompt",
        "metadata",
        "template",
    ],
)
def test_preflight_rejects_native_or_split_drift(preflight_inputs, monkeypatch, fault):
    plan, rows, expected, directory = preflight_inputs
    if fault == "gpu":
        monkeypatch.setitem(sys.modules, "torch", NS(cuda=NS(is_available=lambda: True)))
    elif fault == "output":
        Path(plan["output_root"]).mkdir()
    elif fault in ("split", "task-set"):
        (directory / (fault + ".json")).write_text(json.dumps({"sha256": "different"}))
    elif fault == "duplicate":
        rows["train"] *= 2
    elif fault == "lineage":
        expected.pop()
    elif fault == "template":
        monkeypatch.setattr(miles, "TEMPLATE_SHA256", "changed")
    else:

        class AlteredDataset:
            def __init__(self, path, *args, **kw):
                self.origin_samples = [
                    NS(prompt=row["input"], metadata=row["metadata"])
                    for row in rows[Path(path).stem]
                ]
                if fault == "filtered":
                    self.origin_samples.clear()
                else:
                    setattr(self.origin_samples[0], fault, "changed")

            def __len__(self):
                return len(self.origin_samples)

        monkeypatch.setitem(sys.modules, "miles.utils.data", NS(Dataset=AlteredDataset))
    with pytest.raises((ValueError, FileExistsError)):
        train.preflight(plan)


@pytest.mark.parametrize("stalled", [False, True])
def test_watchdog_bounds_native_startup_and_only_cleans_own_process(
    execution, monkeypatch, stalled
):
    from training import sft_runtime

    plan, root, calls = execution
    waits = []

    def wait(**kw):
        waits.append(kw)
        if len(waits) == 1:
            raise subprocess.TimeoutExpired("synthetic", 60)
        return 0

    child = NS(pid=9876543, poll=lambda: None if not waits else 0, wait=wait, returncode=0)

    def create(*a, **kw):
        batches(plan, root)
        (root / "checkpoints").mkdir()
        (root / "checkpoints/latest_checkpointed_iteration.txt").write_text("0")
        return child

    monkeypatch.setattr(subprocess, "Popen", create)
    monkeypatch.setattr(train, "check_artifacts", lambda p: pytest.fail("unbounded parent check"))
    monkeypatch.setattr(sft_runtime, "_utilization_snapshot", lambda: (0, 0))
    monkeypatch.setattr(
        sft_runtime,
        "ProgressWatchdog",
        lambda started: NS(
            observe=lambda *a, **kw: "confirmed_no_progress_idle" if stalled else None
        ),
    )
    if stalled:
        with pytest.raises(RuntimeError):
            train.run(plan, root / "plan.json")
        assert json.loads((root / "FAILED.json").read_text())["error_class"] == "TimeoutError"
    else:
        assert train.run(plan, root / "plan.json")["status"] == "native_loop_returned"
    assert len(calls) == 2 and {pid for pid, _ in calls} == {9876543}


@pytest.mark.parametrize("fails", [None, "train", "tracking", "budget", "budget_cleanup"])
def test_native_wrapper_calls_real_driver_boundary_once_and_finishes_tracking(
    plan, tmp_path, monkeypatch, fails
):
    source = tmp_path / "train.py"
    source.write_text(
        "from training.rl_episode import EpisodeBudgetExceeded\n"
        "async def train(args):\n    args.calls.append('train')\n"
        "    if args.fails == 'train': raise RuntimeError('synthetic')\n"
        "    if args.fails in {'budget', 'budget_cleanup'}:\n"
        "        raise EpisodeBudgetExceeded('generation_incomplete_length')\n"
    )
    plan["output_root"] = str(tmp_path)
    calls = []
    monkeypatch.setenv("PYTHONPATH", "/root/Megatron-LM")
    args = NS(calls=calls, fails=fails)
    monkeypatch.setattr(train, "check_artifacts", lambda p: None)
    monkeypatch.setattr(train, "native_source", lambda: source)
    monkeypatch.setattr(train, "native_args", lambda p: args)
    monkeypatch.setitem(
        sys.modules,
        "ray",
        NS(init=lambda **kw: calls.append(kw), shutdown=lambda: calls.append("disconnect")),
    )

    def finish():
        calls.append("finish")
        if fails in {"tracking", "budget_cleanup"}:
            raise RuntimeError("synthetic tracking failure")

    monkeypatch.setitem(
        sys.modules,
        "miles.utils.tracking_utils.tracking",
        NS(finish_tracking=finish),
    )
    if fails not in {None, "budget"}:
        with pytest.raises(RuntimeError):
            train._native(plan)
    else:
        train._native(plan)
    assert calls[0]["address"] == "auto" and calls[0]["log_to_driver"] is False
    assert calls[0]["runtime_env"]["env_vars"]["PYTHONPATH"].endswith(":/root/Megatron-LM")
    assert calls[1:] == ["train", "finish", "disconnect"]
    assert (tmp_path / "NATIVE_REJECTED.json").exists() == (fails in {"budget", "budget_cleanup"})
    assert not (tmp_path / "ACCEPTED.json").exists()


def test_native_result_requires_every_expected_batch(execution):
    plan, root, _ = execution
    batches(plan, root)
    (root / "checkpoints").mkdir()
    (root / "checkpoints/latest_checkpointed_iteration.txt").write_text("0")
    assert train.native_result(plan)["completed_batches"] == 3
    directory = root / "episodes/batches/train-r0"
    (directory / "FAILED.json").write_text("{}")
    with pytest.raises(ValueError):
        train.native_result(plan)


def test_progress_uses_only_metadata(tmp_path):
    path = tmp_path / "episodes/synthetic/receipt.json"
    path.parent.mkdir(parents=True)
    path.write_text("synthetic")
    result = progress(tmp_path)
    assert len(result) == 1 and result[0][:2] == ("episodes/synthetic/receipt.json", 9)


@pytest.mark.parametrize("mode", ["parent", "native", "bad_digest"])
def test_entrypoint_routes_once_and_sanitizes_failures(plan, tmp_path, monkeypatch, capsys, mode):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    calls = []
    monkeypatch.setattr(
        train,
        "run",
        lambda p, f: calls.append("parent") or {"status": "synthetic", "sha256": "a" * 64},
    )
    monkeypatch.setattr(train, "_native", lambda p: calls.append("native"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "synthetic",
            "--plan",
            str(path),
            "--sha256",
            "wrong" if mode == "bad_digest" else digest(plan),
        ]
        + (["--native"] if mode == "native" else []),
    )
    if mode == "bad_digest":
        with pytest.raises(SystemExit) as error:
            train.main()
        assert error.value.code == 1 and calls == []
    else:
        train.main()
        assert calls == [mode]
    assert "synthetic prompt" not in capsys.readouterr().out


def test_real_native_parser_from_pinned_image_uses_base_without_resuming():
    pytest.importorskip("fti.trainers.miles.run_fleet")
    try:
        ctypes.CDLL("libcuda.so.1")
    except OSError:
        pytest.skip(
            "native Megatron parser requires CUDA driver libraries; CPU gate cannot qualify it"
        )
    model = "/mnt/sfs/models/qwen3.8-27b-1d4bf0f2"
    checkpoint = "/mnt/sfs/jobs/chris-cpt-cleanup-q38-miles-base-v1/torch-dist"
    if not Path(model).is_dir() or not Path(checkpoint).is_dir():
        pytest.skip("staged Qwen/native checkpoint not mounted")
    cfg = miles.MilesConfig(
        name="synthetic-parser",
        output_root="/mnt/sfs/jobs/synthetic-parser-never-submitted",
        model_root=model,
        torch_dist_root=checkpoint,
        train_data="/mnt/sfs/data/synthetic-parser/train.jsonl",
        dev_data="/mnt/sfs/data/synthetic-parser/dev.jsonl",
        data_manifest="/mnt/sfs/data/synthetic-parser/manifest.json",
        wandb_entity="synthetic",
        wandb_project="synthetic",
        wandb_run_id="synthetic",
    )
    from dataclasses import asdict

    args = train.native_args({"arguments": asdict(cfg), "checkpoint": {"root": checkpoint}})
    assert args.start_rollout_id == 0 and args.load == args.ref_load == checkpoint
    assert args.no_save_optim is False and args.no_save_rng is False
    assert args.num_rollout == args.num_steps_per_rollout == 1
    assert args.calculate_per_token_loss is True
    assert args.grpo_std_normalization is False
    assert args.custom_generate_function_path == "training.rl_episode.generate"


def test_checkpoint_rotation_during_progress_sample_is_safe(tmp_path, monkeypatch):
    path = tmp_path / "checkpoints" / "rotating-file"
    path.parent.mkdir()
    path.touch()
    original = Path.stat

    def stat(p, *args, **kwargs):
        if p == path:
            raise FileNotFoundError("rotated")
        return original(p, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", stat)
    assert progress(tmp_path) == ()


def test_module_entrypoint_rejects_bad_plan_before_any_runtime(tmp_path, monkeypatch, capsys):
    path = tmp_path / "plan.json"
    path.write_text("{}")
    monkeypatch.setattr(sys, "argv", ["miles_training", "--plan", str(path), "--sha256", "wrong"])
    monkeypatch.delitem(sys.modules, "training.miles_training")
    with pytest.raises(SystemExit) as error:
        runpy.run_module("training.miles_training", run_name="__main__")
    assert error.value.code == 1
    assert json.loads(capsys.readouterr().out) == {"status": "failed", "error_class": "ValueError"}
