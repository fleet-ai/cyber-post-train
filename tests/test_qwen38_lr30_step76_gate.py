import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from training import qwen38_lr30_step76_gate as gate
from training.sft_runtime import write_receipt

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/qualification/qwen38-lr30-step76-gpu-reload-v1.json"


def accepted_export_receipt():
    return {
        "receipt_sha256": gate.EXPORT_RECEIPT_SHA256,
        "source_manifest_file_sha256": gate.SOURCE_MANIFEST_FILE_SHA256,
        "source_checkpoint_receipt_sha256": gate.SOURCE_CHECKPOINT_RECEIPT_SHA256,
        "optimizer_step": 76,
        "optimizer_steps_executed": 0,
        "source_inventory_sizes_mtimes_unchanged": True,
        "output_root": str(gate.EXPORT.parent),
    }


def accepted_gpu_receipt():
    return {
        "checker_sha256": gate.CHECKER_SHA256,
        "export_receipt_sha256": gate.EXPORT_RECEIPT_SHA256,
        "finite_logits": True,
        "generated_tokens": 2,
        "gpu_reload_verified": True,
        "source_unchanged": True,
        "optimizer_steps_executed": 0,
        "serving_qualified": False,
    }


def test_validate_source_is_exact(monkeypatch):
    checked = []
    monkeypatch.setattr(gate, "_checked_file", lambda *args: checked.append(args))
    monkeypatch.setattr(gate, "receipt", lambda _: accepted_export_receipt())
    assert gate.validate_source()["optimizer_step"] == 76
    assert checked == [(gate.EXPORT, gate.EXPORT_FILE_SHA256)]

    for key, bad in (
        ("optimizer_step", 75),
        ("optimizer_steps_executed", 1),
        ("source_inventory_sizes_mtimes_unchanged", False),
        ("output_root", "/different"),
    ):
        value = accepted_export_receipt()
        value[key] = bad
        monkeypatch.setattr(gate, "receipt", lambda _, value=value: value)
        with pytest.raises(ValueError, match="accepted binding"):
            gate.validate_source()


@pytest.mark.parametrize(
    ("key", "bad"),
    [
        ("checker_sha256", "0" * 64),
        ("finite_logits", False),
        ("generated_tokens", 1),
        ("gpu_reload_verified", False),
        ("source_unchanged", False),
        ("optimizer_steps_executed", 1),
        ("serving_qualified", True),
    ],
)
def test_validate_gpu_receipt_fails_closed(key, bad):
    value = accepted_gpu_receipt()
    value[key] = bad
    with pytest.raises(ValueError, match="zero-update forward gate"):
        gate.validate_gpu_receipt(value)


def test_parent_is_bounded_create_once_and_uses_module_child(tmp_path, monkeypatch):
    output = tmp_path / "run"
    output.mkdir()
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        write_receipt(output / "GPU_CHECK.json", accepted_gpu_receipt())
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(gate.subprocess, "run", run)
    gate.run_parent(output)

    started = json.loads((output / "STARTED.json").read_text())
    complete = json.loads((output / "COMPLETE.json").read_text())
    assert started["gpus"] == 1 and started["optimizer_steps_executed"] == 0
    assert complete["gpus"] == 1 and complete["serving_qualified"] is False
    assert calls[0][0] == [
        gate.sys.executable,
        "-m",
        "training.qwen38_lr30_step76_gate",
        "--child",
    ]
    assert calls[0][1]["timeout"] == gate.HARD_CHILD_SECONDS
    assert (output / "private-synthetic-loader.log").stat().st_mode & 0o777 == 0o600

    with pytest.raises(FileExistsError):
        gate.run_parent(output)


def test_failure_receipt_is_sanitized(tmp_path, monkeypatch):
    output = tmp_path / "run"
    output.mkdir()

    def fail(*args, **kwargs):
        raise RuntimeError("private details must not be copied")

    monkeypatch.setattr(gate.subprocess, "run", fail)
    with pytest.raises(RuntimeError):
        gate.run_parent(output)
    failed = json.loads((output / "FAILED.json").read_text())
    assert failed["error_class"] == "RuntimeError"
    assert "private details" not in json.dumps(failed)
    assert failed["optimizer_steps_executed"] == 0


def test_request_is_one_gpu_c1_alert_off_and_immutable():
    from cyber_post_train.jobs import digest

    first = gate.job_request()
    second = gate.job_request()
    assert first == second
    assert first["name"] == gate.RUN_NAME
    assert first["run_dir"] == gate.RUN_DIR
    assert first["image"] == gate.IMAGE
    assert first["workers"] == first["gpus_per_worker"] == 1
    assert first["priority_class"] == "c1"
    assert first["failureAlerts"] is False
    assert first["requeueIfPreempted"] is False
    assert first["secrets"] == first["image_pull_secrets"] == []
    assert first["resources"] == {
        "cpu_request": "8",
        "cpu_limit": "8",
        "memory_request": "64Gi",
        "memory_limit": "128Gi",
    }
    assert first["command"].startswith("python -c ")
    assert any(key.startswith("CYBER_RUNTIME_BUNDLE") for key in first["env"])
    plan = json.loads(PLAN.read_text())
    assert plan["request"]["request_sha256"] == "sha256:" + digest(first)


def test_prepared_plan_is_self_bound_and_fail_closed():
    from training.sft_runtime import _unsigned_digest, digest

    plan = json.loads(PLAN.read_text())
    unsigned = {key: value for key, value in plan.items() if key != "sha256"}
    assert plan["sha256"] == "sha256:" + _unsigned_digest(unsigned)
    assert plan["launchable"] is False
    assert plan["execution_record"] == {
        "jobs_submitted": 0,
        "resources_created": 0,
        "gpus_allocated": 0,
    }
    assert plan["live_preview"]["root_failure_alert_annotation"] is None
    assert plan["duplicate_gates"]["jobs_api_history_matches"] == 0
    assert plan["duplicate_gates"]["kubernetes_rayjob_matches"] == 0
    for name, expected in plan["runtime"]["files"].items():
        assert "sha256:" + digest(ROOT / name) == expected
