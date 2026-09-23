"""Qualify the pinned CVE-Bench source and task-5 runtime without model calls."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any

from . import worker
from .protocol import (
    DEFAULT_PROTOCOL,
    canonical,
    digest,
    file_digest,
    load_protocol,
    observed_source,
)

BENCHMARK = "cvebench_zero_day"
FRAMEWORK_BASE_COMMIT = "d5d8813599c77559189191bc6ece5a05eef69823"
RECEIPT_SCHEMA = "cvebench_zero_day_local_runtime_qualification_v1"
_SOURCE = Path(__file__).resolve()
_MODEL_CREDENTIALS = ("FLEET_API_KEY", "OPENAI_API_KEY")
_ARM_COMMON_FIELDS = (
    "model_id",
    "endpoint_origin_sha256",
    "tokenizer_sha256",
    "chat_template_sha256",
    "serving_runtime_sha256",
    "max_context_size",
    "inference_precision",
    "quantization",
)


class QualificationError(RuntimeError):
    """The local evidence does not prove the frozen qualification contract."""


def _signed(value: dict[str, Any]) -> dict[str, Any]:
    if "receipt_sha256" in value:
        raise QualificationError("receipt_digest_prepopulated")
    return {**value, "receipt_sha256": digest(value)}


def _command_available(command: list[str]) -> bool:
    try:
        result = subprocess.run(
            command,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def _source_qualification(protocol: dict[str, Any], checkout: Path) -> dict[str, Any]:
    try:
        tracked_status = subprocess.check_output(
            ["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=no"],
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise QualificationError("source_checkout_unreadable") from error
    if tracked_status:
        raise QualificationError("source_checkout_tracked_files_dirty")
    observed = observed_source(protocol, BENCHMARK, checkout)
    benchmark = worker._bind_cve_source(protocol, checkout)
    if (
        observed.get("verified") is not True
        or observed.get("task_count") != 40
        or benchmark.get("task_count") != 40
        or benchmark.get("task_ids_sha256") != observed.get("task_ids_sha256")
        or benchmark.get("scoring") != "official_deterministic_exploit_grader"
        or benchmark.get("harness", {}).get("variant") != "zero_day"
        or benchmark.get("harness", {}).get("kali_size") != "large"
    ):
        raise QualificationError("source_native_grading_contract_mismatch")
    return {
        "source_commit": benchmark["source"]["commit"],
        "source_release": benchmark["source"]["release"],
        "official_task_count": observed["task_count"],
        "task_ids_sha256": observed["task_ids_sha256"],
        "official_run_sha256": benchmark["harness"]["official_run_script"]["sha256"],
        "runner_sha256": benchmark["bindings"]["runner_sha256"],
        "grader_sha256": benchmark["bindings"]["grader_sha256"],
        "adapter_sha256": benchmark["bindings"]["adapter_sha256"],
        "inspect_ai_version": benchmark["harness"]["inspect_ai"]["version"],
        "inspect_ai_wheel_sha256": benchmark["harness"]["inspect_ai"]["wheel_sha256"],
    }


def _paired_settings_sha256(protocol: dict[str, Any]) -> str:
    arms = protocol["arms"]
    common = {field: arms["base"][field] for field in _ARM_COMMON_FIELDS}
    if any(arms["step_1000"][field] != value for field, value in common.items()):
        raise QualificationError("paired_non_weight_settings_mismatch")
    benchmark = protocol["benchmarks"][BENCHMARK]
    return digest(
        {
            "arms": ["base", "step_1000"],
            "arm_common": common,
            "harness": benchmark["harness"],
            "budget": benchmark["budget"],
            "bindings": benchmark["bindings"],
            "pass_k": benchmark["pass_k"],
            "scoring": benchmark["scoring"],
            "sampling": protocol["execution"]["sampling"],
            "retry": protocol["execution"]["retry"],
            "ordered_cells_sha256": protocol["execution_schedule"]["ordered_cells_sha256"],
        }
    )


def _missing_prerequisites(*, source_qualified: bool) -> list[str]:
    missing: list[str] = []
    if platform.system() != "Linux":
        missing.append("host_os_linux")
    if platform.machine() != "x86_64":
        missing.append("host_arch_x86_64")
    if shutil.which("uv") is None:
        missing.append("uv_executable")
    docker = shutil.which("docker")
    if docker is None:
        missing.extend(("docker_cli", "docker_engine", "docker_compose_v2"))
    else:
        if not _command_available(
            [docker, "version", "--format", "{{.Client.Version}}|{{.Server.Version}}"]
        ):
            missing.append("docker_engine")
        if not _command_available([docker, "compose", "version", "--short"]):
            missing.append("docker_compose_v2")
    if not source_qualified:
        missing.append("pinned_source_and_native_grading_bindings")
    if any(name in os.environ for name in _MODEL_CREDENTIALS):
        missing.append("model_credentials_absent")
    return missing


def _validate_runtime_result(protocol: dict[str, Any], value: object) -> dict[str, Any]:
    benchmark = protocol["benchmarks"][BENCHMARK]
    canary = protocol["operational_canary"]
    if not isinstance(value, dict):
        raise QualificationError("runtime_result_invalid")
    task = value.get("task")
    toolchain = value.get("toolchain")
    expected_fields = {
        "schema",
        "protocol_sha256",
        "benchmark",
        "task_id",
        "arm",
        "status",
        "source_commit",
        "kali_size",
        "model_requests",
        "docker_available",
        "compose_available",
        "toolchain",
        "positive_grader_control",
        "task",
        "task_sha256",
    }
    if (
        set(value) != expected_fields
        or value.get("schema") != "external_ctf_runtime_qualification_v1"
        or value.get("protocol_sha256") != protocol["protocol_sha256"]
        or value.get("benchmark") != BENCHMARK
        or value.get("task_id") != canary["task_id"]
        or value.get("arm") != "qualification"
        or value.get("status") != "runtime_preflight_passed"
        or value.get("source_commit") != benchmark["source"]["commit"]
        or value.get("kali_size") != "large"
        or value.get("model_requests") != 0
        or value.get("docker_available") is not True
        or value.get("compose_available") is not True
        or value.get("positive_grader_control") != "passed"
        or not isinstance(task, dict)
        or not isinstance(toolchain, dict)
    ):
        raise QualificationError("runtime_result_binding_mismatch")
    images = worker._validate_qualification_images(task.get("images"))
    if (
        set(task)
        != {
            "task_id",
            "startup_healthy",
            "checker_before_solution",
            "official_solution_executed",
            "checker_after_solution",
            "images",
            "image_set_sha256",
        }
        or task.get("task_id") != canary["task_id"]
        or task.get("startup_healthy") is not True
        or task.get("checker_before_solution") is not False
        or task.get("official_solution_executed") is not True
        or task.get("checker_after_solution") is not True
        or task.get("image_set_sha256") != digest(images)
        or value.get("task_sha256") != digest(task)
        or toolchain.get("inspect_ai_version") != benchmark["harness"]["inspect_ai"]["version"]
        or toolchain.get("inspect_limit_behavior_sha256")
        != benchmark["harness"]["inspect_ai"]["limit_behavior_source_sha256"]
    ):
        raise QualificationError("runtime_positive_control_mismatch")
    return value


def qualify(checkout: Path, protocol_path: Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    benchmark = protocol["benchmarks"][BENCHMARK]
    source: dict[str, Any] | None = None
    source_failure_class: str | None = None
    try:
        source = _source_qualification(protocol, checkout)
    except Exception as error:
        source_failure_class = type(error).__name__
    missing = _missing_prerequisites(source_qualified=source is not None)
    common = {
        "schema": RECEIPT_SCHEMA,
        "framework_base_commit": FRAMEWORK_BASE_COMMIT,
        "protocol_sha256": protocol["protocol_sha256"],
        "protocol_file_sha256": file_digest(protocol_path.read_bytes()),
        "qualification_module_sha256": file_digest(_SOURCE.read_bytes()),
        "benchmark": BENCHMARK,
        "source_commit": benchmark["source"]["commit"],
        "official_task_count": benchmark["task_count"],
        "task_ids_sha256": benchmark["task_ids_sha256"],
        "qualification_task_index": protocol["operational_canary"]["task_index"],
        "paired_settings_sha256": _paired_settings_sha256(protocol),
        "source_qualification_completed": source is not None,
        "native_grading_bindings_verified": source is not None,
        "source_qualification_sha256": None if source is None else digest(source),
        "source_qualification_failure_class": source_failure_class,
        "model_requests": 0,
        "provider_requests": 0,
        "scored_attempts": 0,
        "infrastructure_failures_count_as_scores": False,
        "contains_prompts_traces_solutions_flags_or_scores": False,
    }
    if missing:
        return _signed(
            {
                **common,
                "status": "blocked_missing_prerequisites",
                "runtime_qualification_completed": False,
                "task5_positive_control_completed": False,
                "missing_prerequisites": missing,
                "runtime_failure_class": None,
                "runtime_evidence_sha256": None,
                "task_image_set_sha256": None,
                "toolchain_sha256": None,
            }
        )
    try:
        raw_runtime = worker._run_cvebench_qualification(
            protocol,
            protocol["operational_canary"]["task_id"],
            checkout,
        )
        runtime = _validate_runtime_result(protocol, raw_runtime)
    except Exception as error:
        return _signed(
            {
                **common,
                "status": "infrastructure_invalid",
                "runtime_qualification_completed": False,
                "task5_positive_control_completed": False,
                "missing_prerequisites": [],
                "runtime_failure_class": type(error).__name__,
                "runtime_evidence_sha256": None,
                "task_image_set_sha256": None,
                "toolchain_sha256": None,
            }
        )
    return _signed(
        {
            **common,
            "status": "runtime_qualified",
            "runtime_qualification_completed": True,
            "task5_positive_control_completed": True,
            "missing_prerequisites": [],
            "runtime_failure_class": None,
            "runtime_evidence_sha256": digest(runtime),
            "task_image_set_sha256": runtime["task"]["image_set_sha256"],
            "toolchain_sha256": digest(runtime["toolchain"]),
        }
    )


def _write_once(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = qualify(args.checkout, args.protocol)
    _write_once(args.output, receipt)
    print(
        json.dumps(
            {"receipt_sha256": receipt["receipt_sha256"], "status": receipt["status"]},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
