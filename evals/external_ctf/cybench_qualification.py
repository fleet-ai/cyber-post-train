"""Provider-free qualification of the pinned Cybench source and native grader."""

from __future__ import annotations

import argparse
import ast
import copy
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .protocol import (
    canonical,
    digest,
    file_digest,
    git_no_replace_env,
    load_protocol,
    observed_source,
)

BENCHMARK = "cybench_web"
MODEL_CREDENTIAL_ENV_VARS = frozenset(
    {
        "ANTHROPIC_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "FLEET_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "HELM_API_KEY",
        "OPENAI_API_KEY",
        "TOGETHER_API_KEY",
    }
)
TASK_RUNTIME_FILES = (
    "metadata/metadata.json",
    "init_script.sh",
    "start_docker.sh",
    "stop_docker.sh",
    "stop_script.sh",
)
COMPOSE_FILES = ("docker-compose.yml", "compose.yml", "dist/docker-compose.yml")


class CybenchQualificationError(RuntimeError):
    """The pinned source or qualification contract is not exactly reproducible."""


def _git_blob(checkout: Path, commit: str, source_path: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(checkout), "show", f"{commit}:{source_path}"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        env=git_no_replace_env(),
    )
    if result.returncode != 0:
        raise CybenchQualificationError("pinned_source_blob_missing")
    return result.stdout


def _git_object_exists(checkout: Path, commit: str, source_path: str) -> bool:
    return (
        subprocess.run(
            ["git", "-C", str(checkout), "cat-file", "-e", f"{commit}:{source_path}"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=git_no_replace_env(),
        ).returncode
        == 0
    )


def _attribute_path(node: ast.expr) -> str | None:
    parts: list[str] = []
    current: ast.expr = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    parts.append(current.id)
    return ".".join(reversed(parts))


def _named_function(tree: ast.Module, name: str) -> ast.FunctionDef:
    functions = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(functions) != 1:
        raise CybenchQualificationError("native_runner_contract_missing")
    return functions[0]


def _native_acceptance_expression(agent_source: bytes) -> ast.Compare:
    tree = ast.parse(agent_source)
    simple_agent = [
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SimpleAgent"
    ]
    if len(simple_agent) != 1:
        raise CybenchQualificationError("native_acceptance_contract_missing")
    methods = [
        node
        for node in simple_agent[0].body
        if isinstance(node, ast.FunctionDef) and node.name == "_process_subtask"
    ]
    if len(methods) != 1:
        raise CybenchQualificationError("native_acceptance_contract_missing")
    candidates: list[ast.Compare] = []
    for node in ast.walk(methods[0]):
        if not isinstance(node, ast.Assign) or not any(
            isinstance(target, ast.Name) and target.id == "score" for target in node.targets
        ):
            continue
        value = node.value
        if (
            isinstance(value, ast.IfExp)
            and isinstance(value.body, ast.Constant)
            and value.body.value == 1
            and isinstance(value.orelse, ast.Constant)
            and value.orelse.value == 0
            and isinstance(value.test, ast.Compare)
            and len(value.test.ops) == 1
            and isinstance(value.test.ops[0], ast.Eq)
            and len(value.test.comparators) == 1
            and _attribute_path(value.test.left) == "iteration_result.execution_output.stdout"
            and _attribute_path(value.test.comparators[0]) == "subtask.answer"
        ):
            candidates.append(value.test)
    if len(candidates) != 1:
        raise CybenchQualificationError("native_acceptance_is_not_exact_equality")
    return candidates[0]


def _native_acceptance_controls(agent_source: bytes) -> dict[str, bool]:
    expression = ast.Expression(body=copy.deepcopy(_native_acceptance_expression(agent_source)))
    ast.fix_missing_locations(expression)
    compiled = compile(expression, "<pinned-cybench-agent-acceptance>", "eval")
    expected = "CYBENCH_QUALIFICATION_SENTINEL"

    def accepted(candidate: str) -> bool:
        result = eval(  # noqa: S307 - the expression is a shape-checked pinned AST node.
            compiled,
            {"__builtins__": {}},
            {
                "iteration_result": SimpleNamespace(
                    execution_output=SimpleNamespace(stdout=candidate)
                ),
                "subtask": SimpleNamespace(answer=expected),
            },
        )
        return result is True

    controls = {
        "exact_accept": accepted(expected),
        "prefix_reject": not accepted("prefix-" + expected),
        "suffix_reject": not accepted(expected + "-suffix"),
        "whitespace_reject": not accepted(expected + "\n"),
        "case_reject": not accepted(expected.lower()),
    }
    if not all(controls.values()):
        raise CybenchQualificationError("native_acceptance_control_failed")
    return controls


def _runner_controls(runner_source: bytes) -> dict[str, bool]:
    tree = ast.parse(runner_source)
    unguided = _named_function(tree, "unguided_mode")
    task_assignments = [
        node
        for node in ast.walk(unguided)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "task" for target in node.targets)
        and isinstance(node.value, ast.Call)
        and _attribute_path(node.value.func) == "Subtask"
    ]
    if len(task_assignments) != 1:
        raise CybenchQualificationError("native_runner_unguided_contract_missing")
    keywords = {keyword.arg: keyword.value for keyword in task_assignments[0].value.keywords}
    expected_answer = ast.parse("subtasks[-1].answer", mode="eval").body
    answer = keywords.get("answer")
    if (
        not isinstance(keywords.get("question"), ast.Constant)
        or keywords["question"].value != "What is the flag?"
        or not isinstance(answer, ast.expr)
        or ast.dump(answer, include_attributes=False)
        != ast.dump(expected_answer, include_attributes=False)
    ):
        raise CybenchQualificationError("native_runner_unguided_contract_missing")
    answer_expression = ast.Expression(body=copy.deepcopy(answer))
    ast.fix_missing_locations(answer_expression)
    selected = eval(  # noqa: S307 - the expression is an exact shape-checked pinned AST node.
        compile(answer_expression, "<pinned-cybench-runner-answer>", "eval"),
        {"__builtins__": {}},
        {
            "subtasks": [
                SimpleNamespace(answer="DECOY"),
                SimpleNamespace(answer="CYBENCH_QUALIFICATION_SENTINEL"),
            ]
        },
    )
    main = _named_function(tree, "main")
    expected_selection = ast.parse(
        "unguided_mode(subtasks) if args.unguided_mode else subtasks", mode="eval"
    ).body
    selections = [
        node.value
        for node in ast.walk(main)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "tasks" for target in node.targets)
    ]
    run_calls = [
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Call) and _attribute_path(node.func) == "runner.run_task"
    ]
    forwarded = False
    if len(run_calls) == 1:
        for keyword in run_calls[0].keywords:
            if keyword.arg == "unguided_mode" and _attribute_path(keyword.value) == (
                "args.unguided_mode"
            ):
                forwarded = True
    controls = {
        "official_unguided_question": True,
        "final_subtask_answer_selected": selected == "CYBENCH_QUALIFICATION_SENTINEL",
        "unguided_selection_bound": len(selections) == 1
        and ast.dump(selections[0], include_attributes=False)
        == ast.dump(expected_selection, include_attributes=False),
        "unguided_mode_forwarded": forwarded,
    }
    if not all(controls.values()):
        raise CybenchQualificationError("native_runner_control_failed")
    return controls


def _source_rows(checkout: Path, benchmark: dict[str, Any]) -> list[dict[str, Any]]:
    commit = benchmark["source"]["commit"]
    source_missing = set(benchmark["source_unavailable_task_ids"])
    rows: list[dict[str, Any]] = []
    for task_index, task_id in enumerate(benchmark["task_ids"]):
        task_exists = _git_object_exists(checkout, commit, task_id)
        expected_missing = task_id in source_missing
        if task_exists == expected_missing:
            raise CybenchQualificationError("pinned_task_source_availability_mismatch")
        row: dict[str, Any] = {
            "task_index": task_index,
            "task_id_sha256": file_digest(task_id.encode()),
            "source_state": "missing" if expected_missing else "present",
            "runtime_asset_manifest_sha256": None,
        }
        if not expected_missing:
            required = [f"{task_id}/{item}" for item in TASK_RUNTIME_FILES]
            compose = [
                f"{task_id}/{item}"
                for item in COMPOSE_FILES
                if _git_object_exists(checkout, commit, f"{task_id}/{item}")
            ]
            if len(compose) != 1 or any(
                not _git_object_exists(checkout, commit, source_path) for source_path in required
            ):
                raise CybenchQualificationError("pinned_task_runtime_assets_missing")
            assets = [
                {
                    "path": source_path.removeprefix(task_id + "/"),
                    "sha256": file_digest(_git_blob(checkout, commit, source_path)),
                }
                for source_path in sorted([*required, *compose])
            ]
            row["runtime_asset_manifest_sha256"] = digest(assets)
        rows.append(row)
    if [row["source_state"] for row in rows].count("present") != 5 or [
        row["source_state"] for row in rows
    ].count("missing") != 1:
        raise CybenchQualificationError("pinned_task_source_count_mismatch")
    return rows


def expected_remote_runtime_evidence(protocol: dict[str, Any]) -> dict[str, Any]:
    benchmark = protocol["benchmarks"][BENCHMARK]
    harness = benchmark["harness"]
    bindings = benchmark["bindings"]
    missing = set(benchmark["source_unavailable_task_ids"])
    tasks: list[dict[str, Any]] = []
    for task_index, task_id in enumerate(benchmark["task_ids"]):
        identity = {
            "task_index": task_index,
            "task_id_sha256": file_digest(task_id.encode()),
        }
        if task_id in missing:
            tasks.append(
                {
                    **identity,
                    "source_state": "missing",
                    "outcome": "infra_invalid_source_missing",
                    "sandbox_created": False,
                    "model_requests": 0,
                }
            )
            continue
        tasks.append(
            {
                **identity,
                "source_state": "present",
                "fresh_sandbox": True,
                "platform": "linux/amd64",
                "challenge_started": True,
                "challenge_health_verified": True,
                "native_init_script_executed": True,
                "agent_visible_only_native_init_artifacts": True,
                "agent_visible_benchmark_repository": False,
                "agent_visible_metadata": False,
                "agent_visible_solutions": False,
                "agent_visible_flags": False,
                "agent_network_scope": "task_challenge_only",
                "opencode_image_id": harness["image_id"],
                "opencode_version": harness["version"],
                "system_prompt_sha256": bindings["system_prompt_sha256"],
                "tool_schema_sha256": bindings["tool_schema_sha256"],
                "native_exact_accept_control": True,
                "native_near_miss_reject_control": True,
                "model_requests": 0,
                "cleanup_verified": True,
            }
        )
    return {
        "schema": "cybench_remote_runtime_qualification_v1",
        "protocol_sha256": protocol["protocol_sha256"],
        "benchmark": BENCHMARK,
        "source_commit": benchmark["source"]["commit"],
        "task_ids_sha256": benchmark["task_ids_sha256"],
        "platform": "linux/amd64",
        "harness_image_id": harness["image_id"],
        "harness_version": harness["version"],
        "model_requests": 0,
        "tasks": tasks,
    }


def validate_remote_runtime_evidence(protocol: dict[str, Any], value: object) -> None:
    expected = expected_remote_runtime_evidence(protocol)
    if not isinstance(value, dict):
        raise CybenchQualificationError("remote_runtime_evidence_invalid")
    unsigned = {key: item for key, item in value.items() if key != "receipt_sha256"}
    if unsigned != expected or value.get("receipt_sha256") != digest(expected):
        raise CybenchQualificationError("remote_runtime_evidence_invalid")


def _normalized_platform(system: str, machine: str) -> str:
    architecture = {"x86_64": "amd64", "aarch64": "arm64"}.get(machine.lower(), machine.lower())
    return f"{system.lower()}/{architecture}"


def _local_runtime_prerequisites(image_id: str) -> dict[str, Any]:
    process_platform = _normalized_platform(platform.system(), platform.machine())
    blockers: list[str] = []
    if process_platform != "linux/amd64":
        blockers.append("qualification_process_linux_amd64_required")
    docker = shutil.which("docker")
    docker_server_platform: str | None = None
    image_available = False
    if docker is None:
        blockers.append("docker_cli_unavailable")
    else:
        version = subprocess.run(
            [docker, "version", "--format", "{{.Server.Os}}/{{.Server.Arch}}"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=15,
        )
        if version.returncode != 0 or not version.stdout.strip():
            blockers.append("docker_daemon_unavailable")
        else:
            server = version.stdout.strip().split("/", maxsplit=1)
            if len(server) == 2:
                docker_server_platform = _normalized_platform(server[0], server[1])
            if docker_server_platform != "linux/amd64":
                blockers.append("docker_server_linux_amd64_required")
            inspected = subprocess.run(
                [
                    docker,
                    "image",
                    "inspect",
                    image_id,
                    "--format",
                    "{{.Id}}|{{.Os}}/{{.Architecture}}",
                ],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=15,
            )
            if inspected.returncode != 0:
                blockers.append("pinned_opencode_image_unavailable")
            else:
                observed = inspected.stdout.strip().split("|", maxsplit=1)
                if len(observed) != 2 or observed[0] != image_id:
                    blockers.append("pinned_opencode_image_identity_mismatch")
                elif observed[1] != "linux/amd64":
                    blockers.append("pinned_opencode_image_linux_amd64_required")
                else:
                    image_available = True
    return {
        "process_platform": process_platform,
        "docker_server_platform": docker_server_platform,
        "pinned_opencode_image_available": image_available,
        "blockers": blockers,
        "ready_for_remote_runtime_qualification": not blockers,
    }


def _reject_model_credentials() -> None:
    if any(os.environ.get(name) for name in MODEL_CREDENTIAL_ENV_VARS):
        raise CybenchQualificationError("model_credential_present")


def qualify(checkout: Path) -> dict[str, Any]:
    _reject_model_credentials()
    value = load_protocol()
    benchmark = value["benchmarks"][BENCHMARK]
    if benchmark["adapter_qualified"] is not False:
        raise CybenchQualificationError("qualification_requires_closed_adapter_gate")
    source = observed_source(value, BENCHMARK, checkout)
    commit = benchmark["source"]["commit"]
    native = benchmark["native_grading"]
    bindings = benchmark["bindings"]
    runner_source = _git_blob(checkout, commit, native["runner_path"])
    acceptance_source = _git_blob(checkout, commit, native["acceptance_path"])
    aggregate_source = _git_blob(checkout, commit, native["aggregate_report_path"])
    if (
        file_digest(runner_source) != native["runner_sha256"]
        or native["runner_sha256"] != bindings["runner_sha256"]
        or file_digest(acceptance_source) != native["acceptance_sha256"]
        or native["acceptance_sha256"] != bindings["grader_sha256"]
        or file_digest(aggregate_source) != native["aggregate_report_sha256"]
        or native["aggregate_report_is_task_acceptance_authority"] is not False
    ):
        raise CybenchQualificationError("native_grading_source_binding_mismatch")
    source_rows = _source_rows(checkout, benchmark)
    acceptance_controls = _native_acceptance_controls(acceptance_source)
    runner_controls = _runner_controls(runner_source)
    prerequisites = _local_runtime_prerequisites(benchmark["harness"]["image_id"])
    remote_contract = expected_remote_runtime_evidence(value)
    unsigned: dict[str, Any] = {
        "schema": "cybench_source_native_qualification_v1",
        "protocol_sha256": value["protocol_sha256"],
        "benchmark": BENCHMARK,
        "source_commit": commit,
        "task_ids_sha256": benchmark["task_ids_sha256"],
        "official_task_count": source["task_count"],
        "source_present_task_count": 5,
        "source_missing_task_count": source["source_unavailable_task_count"],
        "source_rows": source_rows,
        "native_grading": {
            "runner_path": native["runner_path"],
            "runner_sha256": native["runner_sha256"],
            "acceptance_path": native["acceptance_path"],
            "acceptance_sha256": native["acceptance_sha256"],
            "aggregate_report_path": native["aggregate_report_path"],
            "aggregate_report_sha256": native["aggregate_report_sha256"],
            "aggregate_report_excluded_from_task_acceptance": True,
            "acceptance_controls": acceptance_controls,
            "runner_controls": runner_controls,
        },
        "opencode_binding": {
            "name": benchmark["harness"]["name"],
            "version": benchmark["harness"]["version"],
            "image_id": benchmark["harness"]["image_id"],
            "platform": benchmark["harness"]["platform"],
            "system_prompt_sha256": bindings["system_prompt_sha256"],
            "tool_schema_sha256": bindings["tool_schema_sha256"],
        },
        "future_remote_runtime_evidence_sha256": digest(remote_contract),
        "local_runtime_prerequisites": prerequisites,
        "status": (
            "source_native_qualified_remote_runtime_pending"
            if prerequisites["ready_for_remote_runtime_qualification"]
            else "source_native_qualified_runtime_prerequisites_blocked"
        ),
        "adapter_qualified": False,
        "runtime_qualified": False,
        "provider_calls": 0,
        "model_requests": 0,
        "challenge_containers_started": 0,
        "contains_prompts_flags_solutions_traces_or_scores": False,
    }
    return {**unsigned, "receipt_sha256": digest(unsigned)}


def _write_once(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(canonical(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Qualify pinned Cybench source and native grading without a provider or model."
    )
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = qualify(args.checkout)
    _write_once(args.output, result)
    print(
        json.dumps(
            {"receipt_sha256": result["receipt_sha256"], "status": result["status"]},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
