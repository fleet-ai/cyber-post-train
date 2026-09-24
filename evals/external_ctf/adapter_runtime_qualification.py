"""One score-free Linux qualification for the shared NYU/Cybench adapter."""

from __future__ import annotations

import argparse
import os
import platform
import re
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import cybench_runtime_qualification as cybench
from . import nyu_runtime_qualification as nyu
from . import opencode_scored
from . import runtime_qualification as rt
from .protocol import DEFAULT_PROTOCOL, digest, file_digest, git_no_replace_env, load_protocol

SCHEMA = "external_ctf_adapter_runtime_qualification_v1"
SANDBOX_NAME = "extctf-nyu-cyb-adapter-qual-v1"
BENCHMARKS = (nyu.BENCHMARK, cybench.BENCHMARK)
SENSITIVE_ENVIRONMENT_NAMES = frozenset(opencode_scored.REAL_CREDENTIAL_ENVIRONMENT_NAMES)


class AdapterQualificationError(RuntimeError):
    """The model-free adapter qualification contract was not preserved."""


def source_sha256() -> str:
    files = {
        "adapter_runtime_qualification.py": Path(__file__),
        "cybench_runtime_qualification.py": Path(cybench.__file__),
        "nyu_runtime_qualification.py": Path(nyu.__file__),
        "opencode_scored": Path(opencode_scored.__file__),
    }
    return digest({name: file_digest(path.read_bytes()) for name, path in sorted(files.items())})


def contract_sha256(protocol: dict[str, Any]) -> str:
    return digest(
        {
            "schema": SCHEMA,
            "protocol_sha256": protocol["protocol_sha256"],
            "source_sha256": source_sha256(),
            "canaries": {
                name: protocol["benchmarks"][name]["runtime_qualification"]["canary_task_index"]
                for name in BENCHMARKS
            },
            "runtime_contracts": {
                name: protocol["benchmarks"][name]["runtime_qualification"]["contract_sha256"]
                for name in BENCHMARKS
            },
        }
    )


def _docker_ids(kind: str) -> set[str]:
    command = ["docker", kind, "ls", "--quiet"]
    if kind == "container":
        command.insert(3, "--all")
    return {line for line in rt.run(command, timeout=60).stdout.splitlines() if line}


def _cleanup_delta(containers: set[str], networks: set[str]) -> bool:
    extra_containers = _docker_ids("container") - containers
    if extra_containers:
        rt.docker("rm", "--force", *sorted(extra_containers), check=False, timeout=120)
    extra_networks = _docker_ids("network") - networks
    if extra_networks:
        rt.docker("network", "rm", *sorted(extra_networks), check=False, timeout=120)
    return _docker_ids("container") == containers and _docker_ids("network") == networks


def _clone(repository: str, commit: str, destination: Path) -> None:
    environment = git_no_replace_env()
    rt.run(
        ["git", "clone", "--quiet", "--no-checkout", repository, str(destination)],
        env=environment,
        timeout=900,
    )
    rt.run(
        [
            "git",
            "--no-replace-objects",
            "-C",
            str(destination),
            "checkout",
            "--quiet",
            "--detach",
            commit,
        ],
        env=environment,
        timeout=300,
    )


def _receipt_ok(value: object) -> bool:
    return isinstance(value, dict) and value.get("receipt_sha256") == digest(
        {key: item for key, item in value.items() if key != "receipt_sha256"}
    )


def _runtime_row(name: str, value: dict[str, Any]) -> dict[str, Any]:
    accepted = (
        _receipt_ok(value)
        and value.get("status") == "runtime_qualified"
        and value.get("error_code") is None
        and value.get("provider_calls") == 0
        and value.get("model_requests") == 0
        and value.get("scored_attempts") == 0
        and value.get("contains_prompts_flags_solutions_traces_or_scores") is False
    )
    return {
        "phase": name,
        "status": "qualified" if accepted else "infrastructure_invalid",
        "qualification_receipt_sha256": value.get("receipt_sha256") if _receipt_ok(value) else None,
        "qualification_contract_sha256": value.get("qualification_contract_sha256"),
        "source_commit": value.get("source_commit"),
        "task_index": value.get("task_index"),
        "task_id_sha256": value.get("task_id_sha256"),
        "provider_calls": 0,
        "model_requests": 0,
        "scored_attempts": 0,
        "error_code": None if accepted else value.get("error_code") or "receipt_invalid",
        "contains_prompts_flags_solutions_traces_or_scores": False,
    }


def _credential_row(value: dict[str, Any]) -> dict[str, Any]:
    accepted = (
        _receipt_ok(value)
        and value.get("status") == "qualified"
        and value.get("provider_calls") == 0
        and value.get("model_requests") == 0
        and value.get("scoring_calls") == 0
        and value.get("provider_credential_location") == "fixed_proxy_only"
        and value.get("ambient_real_credential_present") is False
        and value.get("agent_real_model_or_scoring_credential_present") is False
        and value.get("challenge_real_model_or_scoring_credential_present") is False
        and value.get("tensorlake_management_credential_forwarded") is False
        and value.get("agent_docker_socket_present") is False
        and value.get("challenge_docker_socket_present") is False
        and value.get("cleanup_verified") is True
        and value.get("contains_credentials_prompts_flags_solutions_traces_or_scores") is False
    )
    return {
        "phase": "credential_isolation",
        "status": "qualified" if accepted else "infrastructure_invalid",
        "qualification_receipt_sha256": value.get("receipt_sha256") if _receipt_ok(value) else None,
        "adapter_source_sha256": value.get("adapter_source_sha256"),
        "credential_environment_names_checked": value.get("credential_environment_names_checked"),
        "provider_credential_location": value.get("provider_credential_location"),
        "provider_calls": 0,
        "model_requests": 0,
        "scoring_calls": 0,
        "agent_real_model_or_scoring_credential_present": False,
        "challenge_real_model_or_scoring_credential_present": False,
        "tensorlake_management_credential_forwarded": False,
        "cleanup_verified": value.get("cleanup_verified") is True,
        "error_code": None if accepted else "credential_receipt_invalid",
        "contains_credentials_prompts_flags_solutions_traces_or_scores": False,
    }


def _phase(name: str, operation: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    containers, networks = _docker_ids("container"), _docker_ids("network")
    try:
        value = operation()
    except Exception as error:  # The next independent phase must still run.
        value = {"status": "infrastructure_invalid", "error_code": rt.failure_code(error)}
    cleanup_verified = _cleanup_delta(containers, networks)
    row = _credential_row(value) if name == "credential_isolation" else _runtime_row(name, value)
    row["cleanup_verified"] = cleanup_verified and row.get("cleanup_verified", True)
    if not row["cleanup_verified"]:
        row["status"] = "infrastructure_invalid"
        row["error_code"] = "phase_cleanup_failed"
    return row


def qualify(protocol_path: Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    """Run exactly one NYU canary, one Cybench canary, and the boundary proof."""
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise AdapterQualificationError("linux_amd64_required")
    if any(os.environ.get(name) for name in SENSITIVE_ENVIRONMENT_NAMES):
        raise AdapterQualificationError("real_credential_present_in_worker_environment")
    protocol = load_protocol(protocol_path)
    phases = [
        _phase(
            "credential_isolation",
            lambda: opencode_scored.qualify_credential_boundary(protocol),
        )
    ]
    for benchmark_name, module in ((nyu.BENCHMARK, nyu), (cybench.BENCHMARK, cybench)):
        benchmark = protocol["benchmarks"][benchmark_name]
        task_index = benchmark["runtime_qualification"]["canary_task_index"]

        def operation(
            benchmark: dict[str, Any] = benchmark,
            task_index: int = task_index,
            module: Any = module,
        ) -> dict[str, Any]:
            with tempfile.TemporaryDirectory(prefix="extctf-adapter-source-") as temporary:
                checkout = Path(temporary) / "source"
                _clone(benchmark["source"]["repository"], benchmark["source"]["commit"], checkout)
                if module is nyu:
                    return nyu.qualify(checkout, task_index, protocol_path)
                return cybench.qualify(
                    checkout,
                    task_index,
                    benchmark["runtime_qualification"]["controller_image"],
                    protocol_path,
                )

        phases.append(_phase(benchmark_name, operation))
    qualified = all(row["status"] == "qualified" for row in phases)
    unsigned = {
        "schema": SCHEMA,
        "protocol_sha256": protocol["protocol_sha256"],
        "qualification_contract_sha256": contract_sha256(protocol),
        "adapter_source_sha256": source_sha256(),
        "status": "runtime_qualified" if qualified else "infrastructure_invalid",
        "platform": "linux/amd64",
        "phase_order": ["credential_isolation", *BENCHMARKS],
        "phases": phases,
        "provider_calls": 0,
        "model_requests": 0,
        "scoring_calls": 0,
        "scored_attempts": 0,
        "real_credentials_present_in_worker_environment": False,
        "contains_credentials_prompts_flags_solutions_traces_or_scores": False,
    }
    return {**unsigned, "receipt_sha256": digest(unsigned)}


def validate_receipt(value: object, protocol: dict[str, Any]) -> dict[str, Any] | None:
    """Accept only the small, content-free receipt emitted by :func:`qualify`."""
    if not isinstance(value, dict):
        return None
    top_fields = {
        "schema",
        "protocol_sha256",
        "qualification_contract_sha256",
        "adapter_source_sha256",
        "status",
        "platform",
        "phase_order",
        "phases",
        "provider_calls",
        "model_requests",
        "scoring_calls",
        "scored_attempts",
        "real_credentials_present_in_worker_environment",
        "contains_credentials_prompts_flags_solutions_traces_or_scores",
        "receipt_sha256",
    }
    phases = value.get("phases")
    if (
        set(value) != top_fields
        or not _receipt_ok(value)
        or value.get("schema") != SCHEMA
        or value.get("protocol_sha256") != protocol["protocol_sha256"]
        or value.get("qualification_contract_sha256") != contract_sha256(protocol)
        or value.get("adapter_source_sha256") != source_sha256()
        or value.get("status") not in {"runtime_qualified", "infrastructure_invalid"}
        or value.get("platform") != "linux/amd64"
        or value.get("phase_order") != ["credential_isolation", *BENCHMARKS]
        or not isinstance(phases, list)
        or len(phases) != 3
        or any(
            value.get(field) != 0
            for field in ("provider_calls", "model_requests", "scoring_calls", "scored_attempts")
        )
        or value.get("real_credentials_present_in_worker_environment") is not False
        or value.get("contains_credentials_prompts_flags_solutions_traces_or_scores") is not False
    ):
        return None
    credential_fields = {
        "phase",
        "status",
        "qualification_receipt_sha256",
        "adapter_source_sha256",
        "credential_environment_names_checked",
        "provider_credential_location",
        "provider_calls",
        "model_requests",
        "scoring_calls",
        "agent_real_model_or_scoring_credential_present",
        "challenge_real_model_or_scoring_credential_present",
        "tensorlake_management_credential_forwarded",
        "cleanup_verified",
        "error_code",
        "contains_credentials_prompts_flags_solutions_traces_or_scores",
    }
    runtime_fields = {
        "phase",
        "status",
        "qualification_receipt_sha256",
        "qualification_contract_sha256",
        "source_commit",
        "task_index",
        "task_id_sha256",
        "provider_calls",
        "model_requests",
        "scored_attempts",
        "cleanup_verified",
        "error_code",
        "contains_prompts_flags_solutions_traces_or_scores",
    }
    if [row.get("phase") for row in phases if isinstance(row, dict)] != [
        "credential_isolation",
        *BENCHMARKS,
    ]:
        return None
    for row in phases:
        if (
            not isinstance(row, dict)
            or row.get("status") not in {"qualified", "infrastructure_invalid"}
            or row.get("cleanup_verified") is not True
            or any(row.get(field) != 0 for field in ("provider_calls", "model_requests"))
            or (row.get("status") == "qualified") is (row.get("error_code") is not None)
            or (
                row.get("error_code") is not None
                and re.fullmatch(r"[a-z0-9_]+", row["error_code"]) is None
            )
        ):
            return None
        if row["phase"] == "credential_isolation":
            if (
                set(row) != credential_fields
                or row.get("scoring_calls") != 0
                or row.get("adapter_source_sha256") != opencode_scored.source_sha256()
                or row.get("credential_environment_names_checked")
                != sorted(SENSITIVE_ENVIRONMENT_NAMES)
                or row.get("provider_credential_location") != "fixed_proxy_only"
                or row.get("agent_real_model_or_scoring_credential_present") is not False
                or row.get("challenge_real_model_or_scoring_credential_present") is not False
                or row.get("tensorlake_management_credential_forwarded") is not False
                or row.get("contains_credentials_prompts_flags_solutions_traces_or_scores")
                is not False
            ):
                return None
            continue
        benchmark = protocol["benchmarks"][row["phase"]]
        task_index = benchmark["runtime_qualification"]["canary_task_index"]
        if (
            set(row) != runtime_fields
            or row.get("qualification_contract_sha256")
            != benchmark["runtime_qualification"]["contract_sha256"]
            or row.get("source_commit") != benchmark["source"]["commit"]
            or row.get("task_index") != task_index
            or row.get("task_id_sha256") != file_digest(benchmark["task_ids"][task_index].encode())
            or row.get("scored_attempts") != 0
            or row.get("contains_prompts_flags_solutions_traces_or_scores") is not False
        ):
            return None
    expected = (
        "runtime_qualified"
        if all(row["status"] == "qualified" for row in phases)
        else "infrastructure_invalid"
    )
    return value if value["status"] == expected else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rt.write_once(args.output, qualify(args.protocol))


if __name__ == "__main__":
    main()
