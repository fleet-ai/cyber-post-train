from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from evals.external_ctf import cvebench_qualification as qualification
from evals.external_ctf.protocol import DEFAULT_PROTOCOL, digest, load_protocol


def _source() -> dict[str, object]:
    return {
        "source_commit": "3" * 40,
        "official_task_count": 40,
        "task_ids_sha256": "sha256:" + "4" * 64,
        "grader_sha256": "sha256:" + "5" * 64,
    }


def _runtime_result(protocol: dict) -> dict[str, object]:
    benchmark = protocol["benchmarks"][qualification.BENCHMARK]
    task_id = protocol["operational_canary"]["task_id"]
    images = [
        {
            "reference": "registry.invalid/cvebench/task5:v2.1.0",
            "image_id": "sha256:" + "1" * 64,
            "repository_digest": "registry.invalid/cvebench/task5@sha256:" + "2" * 64,
        }
    ]
    task = {
        "task_id": task_id,
        "startup_healthy": True,
        "checker_before_solution": False,
        "official_solution_executed": True,
        "checker_after_solution": True,
        "images": images,
        "image_set_sha256": digest(images),
    }
    return {
        "schema": "external_ctf_runtime_qualification_v1",
        "protocol_sha256": protocol["protocol_sha256"],
        "benchmark": qualification.BENCHMARK,
        "task_id": task_id,
        "arm": "qualification",
        "status": "runtime_preflight_passed",
        "source_commit": benchmark["source"]["commit"],
        "kali_size": "large",
        "model_requests": 0,
        "docker_available": True,
        "compose_available": True,
        "toolchain": {
            "docker_client_version": "27.5.1",
            "docker_server_version": "27.5.1",
            "docker_compose_version": "2.32.4",
            "inspect_ai_version": benchmark["harness"]["inspect_ai"]["version"],
            "inspect_limit_behavior_sha256": benchmark["harness"]["inspect_ai"][
                "limit_behavior_source_sha256"
            ],
        },
        "positive_grader_control": "passed",
        "task": task,
        "task_sha256": digest(task),
    }


def _ready(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(qualification, "_source_qualification", lambda *_args: _source())
    monkeypatch.setattr(
        qualification,
        "_missing_prerequisites",
        lambda *, source_qualified: [] if source_qualified else ["source"],
    )


def _assert_self_digested(receipt: dict[str, object]) -> None:
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    assert receipt["receipt_sha256"] == digest(unsigned)


def test_source_qualification_binds_all_40_and_native_grader(monkeypatch) -> None:
    protocol = load_protocol(DEFAULT_PROTOCOL)
    benchmark = protocol["benchmarks"][qualification.BENCHMARK]
    observed = {
        "verified": True,
        "task_count": 40,
        "task_ids_sha256": benchmark["task_ids_sha256"],
    }
    monkeypatch.setattr(qualification.subprocess, "check_output", lambda *_args, **_kwargs: b"")
    monkeypatch.setattr(qualification, "observed_source", lambda *_args: observed)
    monkeypatch.setattr(qualification.worker, "_bind_cve_source", lambda *_args: benchmark)

    result = qualification._source_qualification(  # noqa: SLF001
        protocol, Path("/synthetic/cvebench")
    )

    assert result["official_task_count"] == 40
    assert result["task_ids_sha256"] == benchmark["task_ids_sha256"]
    assert result["grader_sha256"] == benchmark["bindings"]["grader_sha256"]
    assert result["official_run_sha256"] == benchmark["harness"]["official_run_script"]["sha256"]


def test_source_qualification_rejects_tracked_checkout_drift(monkeypatch) -> None:
    protocol = load_protocol(DEFAULT_PROTOCOL)
    monkeypatch.setattr(
        qualification.subprocess,
        "check_output",
        lambda *_args, **_kwargs: b" M src/critical/challenges/example/file\n",
    )

    with pytest.raises(qualification.QualificationError, match="tracked_files_dirty"):
        qualification._source_qualification(  # noqa: SLF001
            protocol, Path("/synthetic/cvebench")
        )


def test_unavailable_local_platform_seals_exact_hold_without_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(qualification, "_source_qualification", lambda *_args: _source())
    monkeypatch.setattr(qualification.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(qualification.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(qualification.shutil, "which", lambda name: f"/bin/{name}")
    monkeypatch.setattr(qualification, "_command_available", lambda _command: True)
    monkeypatch.delenv("FLEET_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def no_runtime(*_args):
        raise AssertionError("runtime must not run when prerequisites are missing")

    monkeypatch.setattr(qualification.worker, "_run_cvebench_qualification", no_runtime)
    receipt = qualification.qualify(Path("/synthetic/cvebench"))

    assert receipt["status"] == "blocked_missing_prerequisites"
    assert receipt["missing_prerequisites"] == ["host_os_linux", "host_arch_x86_64"]
    assert receipt["source_qualification_completed"] is True
    assert receipt["native_grading_bindings_verified"] is True
    assert receipt["runtime_qualification_completed"] is False
    assert receipt["task5_positive_control_completed"] is False
    assert receipt["infrastructure_failures_count_as_scores"] is False
    assert "adapter_qualified" not in receipt
    assert protocol_task_id() not in json.dumps(receipt)
    _assert_self_digested(receipt)


def test_source_failure_is_sanitized_and_never_becomes_a_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_source(*_args):
        raise ValueError("sensitive-source-message")

    monkeypatch.setattr(qualification, "_source_qualification", fail_source)
    monkeypatch.setattr(qualification.platform, "system", lambda: "Linux")
    monkeypatch.setattr(qualification.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(qualification.shutil, "which", lambda name: f"/bin/{name}")
    monkeypatch.setattr(qualification, "_command_available", lambda _command: True)
    monkeypatch.delenv("FLEET_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    receipt = qualification.qualify(Path("/synthetic/cvebench"))
    encoded = json.dumps(receipt, sort_keys=True)

    assert receipt["missing_prerequisites"] == ["pinned_source_and_native_grading_bindings"]
    assert receipt["source_qualification_failure_class"] == "ValueError"
    assert receipt["runtime_qualification_completed"] is False
    assert "score" not in receipt
    assert "sensitive-source-message" not in encoded
    _assert_self_digested(receipt)


def test_even_empty_model_credential_variables_hold_runtime(monkeypatch) -> None:
    monkeypatch.setattr(qualification, "_source_qualification", lambda *_args: _source())
    monkeypatch.setattr(qualification.platform, "system", lambda: "Linux")
    monkeypatch.setattr(qualification.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(qualification.shutil, "which", lambda name: f"/bin/{name}")
    monkeypatch.setattr(qualification, "_command_available", lambda _command: True)
    monkeypatch.setenv("FLEET_API_KEY", "")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    receipt = qualification.qualify(Path("/synthetic/cvebench"))

    assert receipt["missing_prerequisites"] == ["model_credentials_absent"]
    assert receipt["runtime_qualification_completed"] is False


def test_exact_task5_positive_control_is_the_only_success_gate(monkeypatch) -> None:
    protocol = load_protocol(DEFAULT_PROTOCOL)
    runtime = _runtime_result(protocol)
    _ready(monkeypatch)
    monkeypatch.setattr(
        qualification.worker,
        "_run_cvebench_qualification",
        lambda *_args: runtime,
    )

    receipt = qualification.qualify(Path("/synthetic/cvebench"))

    assert receipt["status"] == "runtime_qualified"
    assert receipt["runtime_qualification_completed"] is True
    assert receipt["task5_positive_control_completed"] is True
    assert receipt["qualification_task_index"] == 5
    assert receipt["model_requests"] == 0
    assert receipt["provider_requests"] == 0
    assert receipt["scored_attempts"] == 0
    assert receipt["runtime_evidence_sha256"] == digest(runtime)
    assert receipt["task_image_set_sha256"] == runtime["task"]["image_set_sha256"]
    assert "adapter_qualified" not in receipt
    assert protocol_task_id() not in json.dumps(receipt)
    _assert_self_digested(receipt)


def test_false_to_true_mismatch_is_infrastructure_invalid_not_zero_score(monkeypatch) -> None:
    protocol = load_protocol(DEFAULT_PROTOCOL)
    runtime = _runtime_result(protocol)
    runtime["task"]["checker_after_solution"] = False
    runtime["task_sha256"] = digest(runtime["task"])
    _ready(monkeypatch)
    monkeypatch.setattr(
        qualification.worker,
        "_run_cvebench_qualification",
        lambda *_args: runtime,
    )

    receipt = qualification.qualify(Path("/synthetic/cvebench"))

    assert receipt["status"] == "infrastructure_invalid"
    assert receipt["runtime_failure_class"] == "QualificationError"
    assert receipt["runtime_qualification_completed"] is False
    assert receipt["task5_positive_control_completed"] is False
    assert "score" not in receipt
    _assert_self_digested(receipt)


def test_runtime_error_class_is_retained_without_message_or_score(monkeypatch) -> None:
    _ready(monkeypatch)

    def fail_runtime(*_args):
        raise RuntimeError("sensitive-runtime-message")

    monkeypatch.setattr(qualification.worker, "_run_cvebench_qualification", fail_runtime)
    receipt = qualification.qualify(Path("/synthetic/cvebench"))
    encoded = json.dumps(receipt, sort_keys=True)

    assert receipt["status"] == "infrastructure_invalid"
    assert receipt["runtime_failure_class"] == "RuntimeError"
    assert receipt["runtime_qualification_completed"] is False
    assert "score" not in receipt
    assert "sensitive-runtime-message" not in encoded
    _assert_self_digested(receipt)


def test_paired_settings_reject_any_non_weight_arm_drift() -> None:
    protocol = load_protocol(DEFAULT_PROTOCOL)
    protocol["arms"]["step_1000"]["chat_template_sha256"] = "sha256:" + "0" * 64

    with pytest.raises(qualification.QualificationError, match="paired_non_weight"):
        qualification._paired_settings_sha256(protocol)  # noqa: SLF001


def test_receipt_write_is_create_once_and_private(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    value = qualification._signed({"status": "blocked"})  # noqa: SLF001

    qualification._write_once(path, value)  # noqa: SLF001

    assert json.loads(path.read_bytes()) == value
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    with pytest.raises(FileExistsError):
        qualification._write_once(path, value)  # noqa: SLF001


def protocol_task_id() -> str:
    return load_protocol(DEFAULT_PROTOCOL)["operational_canary"]["task_id"]
