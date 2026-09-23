"""No-network regressions for held-out terminal-observer publication."""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from evals.fleet import evaluate
from evals.fleet import heldout_terminal_observer as observer

VERSION = "11111111-1111-4111-8111-111111111111"


def _config(tmp_path: Path) -> dict:
    (tmp_path / "tasks.json").write_text(
        """{"tasks":[{"data_key":"synthetic-data","data_version":"v1","env_key":"synthetic-env","env_version":"v1","environment_version_id":"11111111-1111-4111-8111-111111111111","task_key":"synthetic-task","task_version_id":"11111111-1111-4111-8111-111111111111"}]}"""
    )
    return {
        "name": "synthetic-eval",
        "task_set": "tasks.json",
        "pass_k": 1,
        "concurrency": 1,
        "max_reviewed_infrastructure_retries": 0,
        "training_data_eligible": False,
        "sampling": {"temperature": 0.6, "top_p": 0.95, "seed": 42},
        "models": {
            "student": {
                "repository": "synthetic/model",
                "revision": "a" * 40,
                "session_model": "synthetic/student",
            }
        },
        "routes": {
            "shared": {
                "model": "student",
                "served_id": "synthetic-student",
                "task_versions": [VERSION],
                "endpoint_origin": "https://inference.flt.build",
                "catalog": {"engine": "sglang", "precision": "bf16", "tensor_parallel_size": 1},
                "model_info": {
                    "model_path": "/model",
                    "model_type": "synthetic",
                    "architectures": ["Synthetic"],
                },
                "server_info": {
                    "model_path": "/model",
                    "context_length": 262144,
                    "tp_size": 1,
                    "quantization": None,
                    "kv_cache_dtype": "fp8_e4m3",
                    "reasoning_parser": "qwen3",
                    "tool_call_parser": "qwen3_coder",
                },
            }
        },
        "harness": {
            "harness": "opencode",
            "harness_version": "1.18.27",
            "release_asset_sha256": "sha256:" + "b" * 64,
            "provider_adapter": "@ai-sdk/openai-compatible",
            "context_management": "opencode_1.18.27_native_compaction_autocontinue_v2",
            "context_window_size": 262144,
            "compaction_headroom_tokens": 20000,
            "max_output_tokens": 32768,
            "max_model_requests": 600,
            "timeout_seconds": 28800,
            "tools": ["bash", "submit_report"],
            "tool_catalog_sha256": "sha256:" + "c" * 64,
        },
        "images": {
            "agent": "registry/agent@sha256:" + "d" * 64,
            "proxy": "registry/proxy@sha256:" + "e" * 64,
        },
    }


def _package(tmp_path: Path) -> tuple[SimpleNamespace, list[dict]]:
    config = _config(tmp_path)
    scientific = copy.deepcopy(config)
    config["model_artifact_binding"] = {"execution_only": True}
    plan = evaluate.compile_eval(scientific, relative_to=tmp_path)
    runtime_root = Path(evaluate.__file__).parent
    data = {
        name: (runtime_root / name).read_text(encoding="utf-8")
        for name in observer.heldout_launch.SEALED_EVALUATOR_MODULES
    }
    data["task-set.json"] = (tmp_path / "tasks.json").read_text(encoding="utf-8")
    packet = SimpleNamespace(
        identity_sha256="sha256:" + "f" * 64,
        files={"evaluation_config": tmp_path / "config.json"},
    )
    return (
        SimpleNamespace(packet=packet, evaluation_config=config, config_map={"data": data}),
        evaluate.plan_rows(plan),
    )


def test_binding_comes_from_fresh_compiled_plan(tmp_path, monkeypatch):
    package, rows = _package(tmp_path)
    monkeypatch.setattr(observer.heldout_launch, "build_package", lambda _path: package)

    binding = observer.validate_plan_rows(tmp_path / "packet.json", rows)

    assert binding.harness_id == "protocol-" + binding.evaluation_plan_sha256
    assert binding.harness_id == rows[0]["harness_id"]
    assert binding.row_count == 1


def test_checkout_plan_row_drift_cannot_change_sealed_identity(tmp_path, monkeypatch):
    package, rows = _package(tmp_path)
    monkeypatch.setattr(observer.heldout_launch, "build_package", lambda _path: package)
    monkeypatch.setattr(evaluate, "plan_rows", lambda _plan: [{"checkout": "drifted"}])

    binding = observer.validate_plan_rows(tmp_path / "packet.json", rows)

    assert binding.harness_id == rows[0]["harness_id"]


def test_stale_harness_leaves_final_root_absent(tmp_path, monkeypatch):
    package, rows = _package(tmp_path)
    monkeypatch.setattr(observer.heldout_launch, "build_package", lambda _path: package)
    stale = [dict(rows[0], harness_id="protocol-" + "0" * 64)]
    output = tmp_path / "final-observer-output"

    with pytest.raises(observer.HeldoutObserverError, match="database identity differs"):
        observer.validate_then_publish(
            tmp_path / "packet.json",
            stale,
            output,
            lambda _binding: {"READY.json": b"{}\n"},
        )

    assert not output.exists()
    assert not list(tmp_path.glob(".final-observer-output.staging-*"))


def test_later_validation_failure_leaves_final_root_absent(tmp_path, monkeypatch):
    package, rows = _package(tmp_path)
    monkeypatch.setattr(observer.heldout_launch, "build_package", lambda _path: package)
    output = tmp_path / "final-observer-output"

    def reject_audit(_binding: observer.PlanBinding) -> dict[str, bytes]:
        raise observer.HeldoutObserverError("private audit differs")

    with pytest.raises(observer.HeldoutObserverError, match="private audit differs"):
        observer.validate_then_publish(tmp_path / "packet.json", rows, output, reject_audit)

    assert not output.exists()


def test_validated_publication_is_private_atomic_and_create_once(tmp_path, monkeypatch):
    package, rows = _package(tmp_path)
    monkeypatch.setattr(observer.heldout_launch, "build_package", lambda _path: package)
    output = tmp_path / "final-observer-output"

    binding = observer.validate_then_publish(
        tmp_path / "packet.json",
        rows,
        output,
        lambda value: {
            "READY.json": (value.harness_id + "\n").encode(),
            "TERMINAL.json": b"{}\n",
        },
    )

    assert output.is_dir()
    assert (output.stat().st_mode & 0o777) == 0o700
    assert (output / "READY.json").read_text() == binding.harness_id + "\n"
    assert (output / "READY.json").stat().st_mode & 0o777 == 0o600
    with pytest.raises(observer.HeldoutObserverError, match="already exists"):
        observer.validate_then_publish(
            tmp_path / "packet.json",
            rows,
            output,
            lambda _binding: {"READY.json": b"changed\n"},
        )
    assert (output / "READY.json").read_text() == binding.harness_id + "\n"


def test_artifact_write_failure_cleans_staging_and_leaves_final_absent(tmp_path, monkeypatch):
    package, rows = _package(tmp_path)
    monkeypatch.setattr(observer.heldout_launch, "build_package", lambda _path: package)
    output = tmp_path / "final-observer-output"
    original = observer._write_exclusive  # noqa: SLF001
    calls = 0

    def fail_second_write(path: Path, payload: bytes) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic write failure")
        original(path, payload)

    monkeypatch.setattr(observer, "_write_exclusive", fail_second_write)
    with pytest.raises(OSError, match="synthetic write failure"):
        observer.validate_then_publish(
            tmp_path / "packet.json",
            rows,
            output,
            lambda _binding: {"A.json": b"{}\n", "B.json": b"{}\n"},
        )

    assert not output.exists()
    assert not list(tmp_path.glob(".final-observer-output.staging-*"))
