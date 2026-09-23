"""Qualify one pinned Cybench web task without calling a model."""

from __future__ import annotations

import argparse
import ast
import copy
import json
import re
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

from . import cybench_qualification as source_gate
from . import runtime_qualification as rt
from .protocol import DEFAULT_PROTOCOL, digest, file_digest, load_protocol

BENCHMARK = source_gate.BENCHMARK
IMMUTABLE_IMAGE = re.compile(r"^[^\s@]+@sha256:[0-9a-f]{64}$")
HOST_GATEWAY_UNSAFE_TASKS = frozenset({0})
CybenchRuntimeError = rt.RuntimeGateError


def _compose_path(root: Path) -> Path:
    paths = [path for name in source_gate.COMPOSE_FILES if (path := root / name).is_file()]
    if len(paths) != 1:
        raise CybenchRuntimeError("compose_contract_invalid")
    return paths[0]


def _external_network(compose: object) -> str:
    networks = compose.get("networks") if isinstance(compose, dict) else None
    if not isinstance(networks, dict):
        raise CybenchRuntimeError("compose_network_contract_invalid")
    names = [
        name
        for name, value in networks.items()
        if isinstance(name, str) and isinstance(value, dict) and value.get("external") is True
    ]
    if len(names) != 1:
        raise CybenchRuntimeError("compose_external_network_contract_invalid")
    return names[0]


def _targets(metadata: object) -> list[tuple[str, int]]:
    raw = metadata.get("target_host") if isinstance(metadata, dict) else None
    if not isinstance(raw, str) or not raw.strip():
        raise CybenchRuntimeError("native_target_host_invalid")
    result: list[tuple[str, int]] = []
    for item in raw.split():
        host, separator, port = item.rpartition(":")
        if not separator or not host or not port.isdecimal() or not 0 < int(port) < 65536:
            raise CybenchRuntimeError("native_target_host_invalid")
        result.append((host, int(port)))
    return result


def _solution(root: Path) -> tuple[str, str, set[str]]:
    metadata = json.loads((root / "metadata/metadata.json").read_bytes())
    subtasks = metadata.get("subtasks") if isinstance(metadata, dict) else None
    answer = subtasks[-1].get("answer") if isinstance(subtasks, list) and subtasks else None
    solution = root / "metadata/solution"
    files = sorted(path for path in solution.rglob("*") if path.is_file())
    if (
        not isinstance(answer, str)
        or not answer
        or not (solution / "solution.sh").is_file()
        or not files
        or any(path.is_symlink() for path in solution.rglob("*"))
    ):
        raise CybenchRuntimeError("official_solution_contract_invalid")
    manifest = [
        {"path": path.relative_to(solution).as_posix(), "sha256": file_digest(path.read_bytes())}
        for path in files
    ]
    return answer, digest(manifest), {row["sha256"] for row in manifest}


def _workspace_hides_solution(workspace: Path, answer: str, solution_digests: set[str]) -> None:
    for path in workspace.rglob("*"):
        if path.is_symlink():
            raise CybenchRuntimeError("agent_workspace_symlink_invalid")
        if not path.is_file():
            continue
        content = path.read_bytes()
        if answer.encode() in content or file_digest(content) in solution_digests:
            raise CybenchRuntimeError("official_solution_visible_to_agent")


def _last_line(output: str) -> str:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        raise CybenchRuntimeError("official_solution_output_empty")
    return lines[-1]


def _native_accepts(source: bytes, candidate: str, answer: str) -> bool:
    expression = ast.Expression(
        body=copy.deepcopy(source_gate._native_acceptance_expression(source))  # noqa: SLF001
    )
    ast.fix_missing_locations(expression)
    return (
        eval(  # noqa: S307 - exact pinned, shape-checked expression is the authority.
            compile(expression, "<pinned-cybench-acceptance>", "eval"),
            {"__builtins__": {}},
            {
                "iteration_result": SimpleNamespace(
                    execution_output=SimpleNamespace(stdout=candidate)
                ),
                "subtask": SimpleNamespace(answer=answer),
            },
        )
        is True
    )


def _execute(
    protocol: dict[str, Any],
    checkout: Path,
    task_id: str,
    task_index: int,
    controller_image: str,
    acceptance_source: bytes,
    observed: dict[str, Any],
) -> None:
    benchmark = protocol["benchmarks"][BENCHMARK]
    with tempfile.TemporaryDirectory(prefix="extcyb-") as temporary:
        root = Path(temporary) / "task"
        workspace = Path(temporary) / "workspace"
        commit = benchmark["source"]["commit"]
        rt.materialize(checkout, commit, task_id, root)
        workspace.mkdir(mode=0o700)
        answer, solution_sha256, solution_digests = _solution(root)
        observed["official_solution_manifest_sha256"] = solution_sha256
        helper_sha256 = file_digest(  # Exact upstream helper; unsafe privileged parts are not run.
            source_gate._git_blob(checkout, commit, "run_solution.sh")  # noqa: SLF001
        )
        observed["official_solution_helper_sha256"] = helper_sha256
        metadata = json.loads((root / "metadata/metadata.json").read_bytes())
        compose_data = yaml.safe_load(_compose_path(root).read_bytes())
        external = _external_network(compose_data)
        compose, removed_ports, removed_names = rt.contained_compose(compose_data, root)
        observed["published_port_entries_removed"] = removed_ports
        observed["container_names_removed"] = removed_names
        with rt.Sandbox(root, compose, external, f"extcyb{task_index:02d}") as sandbox:
            sandbox.start(build=True, timeout=1800)
            images = sandbox.running_images()
            observed["challenge_images"] = images
            sandbox.down()
            sandbox.start(build=False, timeout=1800)
            if sandbox.running_images() != images:
                raise CybenchRuntimeError("challenge_image_binding_changed")
            controller = sandbox.create(
                "controller",
                "--network",
                "bridge",
                "--mount",
                f"type=bind,src={root.resolve()},dst=/task,readonly",
                "--mount",
                f"type=bind,src={workspace.resolve()},dst=/workspace",
                "--entrypoint",
                "/bin/sh",
                controller_image,
                "-c",
                "sleep 1200",
            )
            rt.docker("exec", controller, "bash", "/task/metadata/requirements.sh", timeout=1200)
            rt.docker(
                "exec",
                controller,
                "bash",
                "/task/init_script.sh",
                "/workspace",
                "false",
                timeout=600,
            )
            rt.docker("exec", controller, "chown", "-R", "1000:1000", "/workspace")
            _workspace_hides_solution(workspace, answer, solution_digests)
            rt.docker("network", "disconnect", "bridge", controller)
            rt.docker("network", "connect", sandbox.network, controller)
            rt.assert_isolated(
                controller,
                sandbox.network,
                {
                    (str(root.resolve()), "/task", False),
                    (str(workspace.resolve()), "/workspace", True),
                },
                hardened_user=False,
            )
            agent = sandbox.create(
                "agent",
                "--network",
                sandbox.network,
                "--user",
                "1000:1000",
                "--read-only",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges",
                "--tmpfs",
                "/tmp:rw,noexec,nosuid,size=16m",
                "--mount",
                f"type=bind,src={workspace.resolve()},dst=/home/agent/workspace",
                "--entrypoint",
                "/bin/sh",
                benchmark["harness"]["image_id"],
                "-c",
                "sleep 600",
            )
            rt.assert_isolated(
                agent,
                sandbox.network,
                {(str(workspace.resolve()), "/home/agent/workspace", True)},
                hardened_user=True,
            )
            rt.docker(
                "exec",
                agent,
                "/bin/sh",
                "-ceu",
                "test -r /home/agent/workspace; test -w /home/agent/workspace; "
                'p=/home/agent/workspace/.qualification-write-probe; test ! -e "$p"; '
                ': >"$p"; test -f "$p"; rm "$p"',
            )
            if not all(rt.probe(agent, host, port) for host, port in _targets(metadata)):
                raise CybenchRuntimeError("native_target_not_reachable")
            decoy = "EXTERNAL_CTF_RUNTIME_QUALIFICATION_DECOY"
            if _native_accepts(acceptance_source, decoy, answer):
                raise CybenchRuntimeError("native_grader_decoy_invalid")
            output = rt.docker(
                "exec",
                controller,
                "bash",
                "-lc",
                "cd /task/metadata/solution && exec bash ./solution.sh",
                timeout=1200,
            ).stdout
            # Pinned run_solution.sh applies tail -n 1 | tr -d '[:space:]' to both.
            solution_flag = "".join(_last_line(output).split())
            official_flag = "".join((root / "metadata/solution/flag.txt").read_text().split())
            if solution_flag != official_flag or not _native_accepts(
                acceptance_source, solution_flag, answer
            ):
                raise CybenchRuntimeError("official_solution_not_exactly_accepted")
            _workspace_hides_solution(workspace, answer, solution_digests)
            if sandbox.running_images() != images:
                raise CybenchRuntimeError("challenge_image_binding_changed")


def qualify(
    checkout: Path,
    task_index: int,
    controller_image: str,
    protocol_path: Path = DEFAULT_PROTOCOL,
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    benchmark = protocol["benchmarks"][BENCHMARK]
    task_ids = benchmark["task_ids"]
    task_id = (
        task_ids[task_index]
        if type(task_index) is int and 0 <= task_index < len(task_ids)
        else None
    )
    controller: dict[str, str | None] | None = None
    harness: dict[str, str | None] | None = None
    observed: dict[str, Any] = {
        "task_source": None,
        "official_solution_manifest_sha256": None,
        "official_solution_helper_sha256": None,
        "published_port_entries_removed": None,
        "container_names_removed": None,
        "challenge_images": None,
    }
    task_evidence: dict[str, Any] | None = None
    error_code: str | None = None
    admitted = False
    try:
        protocol, benchmark, task_id = rt.require_task(
            checkout, protocol_path, BENCHMARK, task_index
        )
        observed["task_source"] = source_gate._source_rows(checkout, benchmark)[task_index]  # noqa: SLF001
        if IMMUTABLE_IMAGE.fullmatch(controller_image) is None:
            raise CybenchRuntimeError("immutable_controller_image_required")
        native = benchmark["native_grading"]
        acceptance_source = source_gate._git_blob(  # noqa: SLF001
            checkout, benchmark["source"]["commit"], native["acceptance_path"]
        )
        acceptance = source_gate._native_acceptance_controls(acceptance_source)  # noqa: SLF001
        runner = source_gate._runner_controls(  # noqa: SLF001
            source_gate._git_blob(  # noqa: SLF001
                checkout, benchmark["source"]["commit"], native["runner_path"]
            )
        )
        if not all(acceptance.values()) or not all(runner.values()):
            raise CybenchRuntimeError("pinned_native_grader_control_failed")
        admitted = True
        if task_index in HOST_GATEWAY_UNSAFE_TASKS:
            raise CybenchRuntimeError("task_host_gateway_unsupported")
        rt.require_docker_linux_amd64()
        rt.docker("pull", controller_image, timeout=1800)
        controller = rt.image_lock(controller_image, repository_digest=True)
        if controller["repository_digest"] != controller_image:
            raise CybenchRuntimeError("controller_image_digest_mismatch")
        harness = rt.image_lock(benchmark["harness"]["image_id"], repository_digest=False)
        if harness["image_id"] != benchmark["harness"]["image_id"]:
            raise CybenchRuntimeError("pinned_opencode_image_identity_mismatch")
        _execute(
            protocol,
            checkout,
            task_id,
            task_index,
            controller_image,
            acceptance_source,
            observed,
        )
        task_evidence = source_gate.expected_remote_runtime_evidence(protocol)["tasks"][task_index]
    except Exception as error:
        error_code = rt.failure_code(error)
    unsigned = {
        "schema": (
            "cybench_remote_runtime_qualification_task_execution_v1"
            if admitted
            else "cybench_remote_runtime_qualification_precondition_v1"
        ),
        "protocol_sha256": protocol["protocol_sha256"],
        "benchmark": BENCHMARK,
        "qualification_name": f"extctf-cyb-t{task_index:02d}-qual-v1" if admitted else None,
        "task_index": task_index,
        "task_id_sha256": None if task_id is None else file_digest(task_id.encode()),
        "source_commit": benchmark["source"]["commit"],
        "task_source": observed["task_source"],
        "task_source_sha256": (
            None if observed["task_source"] is None else digest(observed["task_source"])
        ),
        "published_port_entries_removed": observed["published_port_entries_removed"],
        "container_names_removed": observed["container_names_removed"],
        "controller_image_reference": controller_image,
        "controller_image": controller,
        "controller_image_sha256": None if controller is None else digest(controller),
        "harness_image_id": benchmark["harness"]["image_id"],
        "harness_image": harness,
        "harness_image_sha256": None if harness is None else digest(harness),
        "challenge_images": observed["challenge_images"],
        "challenge_image_set_sha256": None
        if observed["challenge_images"] is None
        else digest(observed["challenge_images"]),
        "official_solution_manifest_sha256": observed["official_solution_manifest_sha256"],
        "official_solution_helper_sha256": observed["official_solution_helper_sha256"],
        "platform": "linux/amd64",
        "status": (
            "runtime_qualified"
            if task_evidence
            else "infrastructure_invalid"
            if admitted
            else "precondition_failed"
        ),
        "task_evidence": task_evidence,
        "error_code": error_code,
        "provider_calls": 0,
        "model_requests": 0,
        "scored_attempts": 0,
        "contains_prompts_flags_solutions_traces_or_scores": False,
    }
    return {**unsigned, "receipt_sha256": digest(unsigned)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--controller-image", required=True)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = qualify(args.checkout, args.task_index, args.controller_image, args.protocol)
    rt.write_once(args.output, receipt)
    print(json.dumps({"receipt_sha256": receipt["receipt_sha256"], "status": receipt["status"]}))


if __name__ == "__main__":
    main()
