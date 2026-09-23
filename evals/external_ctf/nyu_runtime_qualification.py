"""Qualify one pinned NYU web task without calling a model."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import yaml

from . import nyu_adapter
from . import runtime_qualification as rt
from .protocol import DEFAULT_PROTOCOL, digest, load_protocol

BENCHMARK = nyu_adapter.BENCHMARK
NyuRuntimeError = rt.RuntimeGateError


def _task_path(protocol: dict[str, Any], checkout: Path, task_id: str) -> str:
    benchmark = protocol["benchmarks"][BENCHMARK]
    manifest = json.loads(
        nyu_adapter._blob(  # noqa: SLF001 - the adapter owns the pinned source bytes.
            checkout, benchmark["source"]["commit"], "test_dataset.json"
        )
    )
    try:
        relative = nyu_adapter._safe_path(manifest[task_id]["path"])  # noqa: SLF001
    except (KeyError, TypeError) as error:
        raise NyuRuntimeError("task_source_path_missing") from error
    return relative


def _external_aliases(compose: object) -> tuple[str, set[str]]:
    if not isinstance(compose, dict):
        raise NyuRuntimeError("compose_contract_invalid")
    networks = compose.get("networks")
    services = compose.get("services")
    if not isinstance(networks, dict) or not isinstance(services, dict):
        raise NyuRuntimeError("compose_contract_invalid")
    external = [
        name
        for name, value in networks.items()
        if isinstance(name, str) and isinstance(value, dict) and value.get("external") is True
    ]
    if len(external) != 1:
        raise NyuRuntimeError("compose_external_network_contract_invalid")
    network = external[0]
    aliases: set[str] = set()
    for name, service in services.items():
        if not isinstance(name, str) or not isinstance(service, dict):
            raise NyuRuntimeError("compose_service_contract_invalid")
        attached = service.get("networks", [])
        if isinstance(attached, list):
            if network in attached:
                aliases.add(name)
            continue
        if not isinstance(attached, dict) or network not in attached:
            continue
        aliases.add(name)
        config = attached[network]
        if not isinstance(config, dict):
            continue
        listed = config.get("aliases", [])
        if not isinstance(listed, list) or any(not isinstance(item, str) for item in listed):
            raise NyuRuntimeError("compose_alias_contract_invalid")
        aliases.update(listed)
    return network, aliases


def _target_binding(target: str, aliases: set[str], reachable: bool) -> str:
    if not reachable:
        raise NyuRuntimeError("exact_target_not_reachable")
    return "compose_alias" if target in aliases else "runtime_proved_alias_mismatch"


def _qualification(
    protocol: dict[str, Any], task: nyu_adapter.NyuTask, images: list[dict[str, Any]]
) -> dict[str, Any]:
    benchmark = protocol["benchmarks"][BENCHMARK]
    unsigned = {
        "schema": "external_ctf_nyu_runtime_qualification_v1",
        "protocol_sha256": protocol["protocol_sha256"],
        "benchmark": BENCHMARK,
        "task_id": task.task_id,
        "source_commit": task.source_commit,
        "challenge_sha256": task.challenge_sha256,
        "compose_sha256": task.compose_sha256,
        "attachment_manifest_sha256": task.attachment_manifest_sha256,
        "target_host": task.target_host,
        "target_port": task.target_port,
        "status": "runtime_preflight_passed",
        "model_requests": 0,
        "startup_healthy": True,
        "target_reachable": True,
        "grader_negative_control": False,
        "grader_positive_control": True,
        "agent_image_id": benchmark["harness"]["image_id"],
        "opencode_version": benchmark["harness"]["version"],
        "platform": "linux/amd64",
        "isolated_task_network": True,
        "images": images,
    }
    receipt = {**unsigned, "receipt_sha256": digest(unsigned)}
    nyu_adapter._validate_qualification(protocol, task, receipt)  # noqa: SLF001
    return receipt


def _execute(
    protocol: dict[str, Any],
    checkout: Path,
    task: nyu_adapter.NyuTask,
    task_index: int,
    observed: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    benchmark = protocol["benchmarks"][BENCHMARK]
    with tempfile.TemporaryDirectory(prefix="extnyu-") as temporary:
        root = Path(temporary) / "task"
        rt.materialize(
            checkout,
            benchmark["source"]["commit"],
            _task_path(protocol, checkout, task.task_id),
            root,
        )
        compose = yaml.safe_load((root / "docker-compose.yml").read_bytes())
        external, aliases = _external_aliases(compose)
        compose_path, removed_ports, removed_names = rt.contained_compose(compose, root)
        observed["published_port_entries_removed"] = removed_ports
        observed["container_names_removed"] = removed_names
        with rt.Sandbox(root, compose_path, external, f"extnyu{task_index:02d}") as sandbox:
            references = sandbox.image_references()
            if references != list(task._compose_images):  # noqa: SLF001
                raise NyuRuntimeError("compose_image_set_mismatch")
            rt.run([*sandbox.command, "pull"], timeout=1800)
            images = [rt.image_lock(reference, repository_digest=True) for reference in references]
            sandbox.start(build=False, timeout=900)
            runtime_images = sandbox.running_images()
            observed["challenge_runtime_images"] = runtime_images
            if {row["image_id"] for row in runtime_images} != {row["image_id"] for row in images}:
                raise NyuRuntimeError("challenge_container_image_binding_mismatch")
            sandbox.down()
            sandbox.start(build=False, timeout=900)
            if sandbox.running_images() != runtime_images:
                raise NyuRuntimeError("challenge_image_binding_changed")
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
                "--entrypoint",
                "/bin/sh",
                benchmark["harness"]["image_id"],
                "-c",
                "sleep 600",
            )
            rt.assert_isolated(agent, sandbox.network, set(), hardened_user=True)
            binding = _target_binding(
                task.target_host,
                aliases,
                rt.probe(agent, task.target_host, task.target_port),
            )
            decoy = "EXTERNAL_CTF_RUNTIME_QUALIFICATION_DECOY"
            if task.score(decoy) != 0 or task.score(task._flag) != 1:  # noqa: SLF001
                raise NyuRuntimeError("hidden_flag_grader_control_failed")
            if (
                sandbox.running_images() != runtime_images
                or [rt.image_lock(item, repository_digest=True) for item in references] != images
            ):
                raise NyuRuntimeError("challenge_image_binding_changed")
            return _qualification(protocol, task, images), binding


def qualify(
    checkout: Path, task_index: int, protocol_path: Path = DEFAULT_PROTOCOL
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    benchmark = protocol["benchmarks"][BENCHMARK]
    task_ids = benchmark["task_ids"]
    task_id = (
        task_ids[task_index]
        if type(task_index) is int and 0 <= task_index < len(task_ids)
        else None
    )
    harness: dict[str, str | None] | None = None
    observed: dict[str, Any] = {
        "task_source": None,
        "published_port_entries_removed": None,
        "container_names_removed": None,
        "challenge_runtime_images": None,
    }
    qualification: dict[str, Any] | None = None
    binding: str | None = None
    error_code: str | None = None
    admitted = False
    try:
        protocol, benchmark, task_id = rt.require_task(
            checkout, protocol_path, BENCHMARK, task_index
        )
        task = nyu_adapter.load_task(protocol, checkout, task_id)
        observed["task_source"] = {
            "challenge_sha256": task.challenge_sha256,
            "compose_sha256": task.compose_sha256,
            "attachment_manifest_sha256": task.attachment_manifest_sha256,
        }
        admitted = True
        rt.require_docker_linux_amd64()
        harness = rt.image_lock(benchmark["harness"]["image_id"], repository_digest=False)
        if harness["image_id"] != benchmark["harness"]["image_id"]:
            raise NyuRuntimeError("pinned_opencode_image_identity_mismatch")
        qualification, binding = _execute(protocol, checkout, task, task_index, observed)
    except Exception as error:
        error_code = rt.failure_code(error)
    unsigned = {
        "schema": (
            "external_ctf_nyu_runtime_qualification_execution_v1"
            if admitted
            else "external_ctf_nyu_runtime_qualification_precondition_v1"
        ),
        "protocol_sha256": protocol["protocol_sha256"],
        "benchmark": BENCHMARK,
        "qualification_name": f"extctf-nyu-t{task_index:02d}-qual-v1" if admitted else None,
        "task_index": task_index,
        "task_id_sha256": (  # noqa: SLF001
            None if task_id is None else nyu_adapter._digest_bytes(task_id.encode())
        ),
        "source_commit": benchmark["source"]["commit"],
        "task_source": observed["task_source"],
        "task_source_sha256": (
            None if observed["task_source"] is None else digest(observed["task_source"])
        ),
        "published_port_entries_removed": observed["published_port_entries_removed"],
        "container_names_removed": observed["container_names_removed"],
        "harness_image_id": benchmark["harness"]["image_id"],
        "harness_image": harness,
        "harness_image_sha256": None if harness is None else digest(harness),
        "challenge_runtime_images": observed["challenge_runtime_images"],
        "challenge_runtime_images_sha256": (
            None
            if observed["challenge_runtime_images"] is None
            else digest(observed["challenge_runtime_images"])
        ),
        "platform": "linux/amd64",
        "status": (
            "runtime_qualified"
            if qualification
            else "infrastructure_invalid"
            if admitted
            else "precondition_failed"
        ),
        "target_address_binding": binding,
        "qualification": qualification,
        "error_code": error_code,
        "official_solution_solvability_claimed": False,
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
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = qualify(args.checkout, args.task_index, args.protocol)
    rt.write_once(args.output, receipt)
    print(json.dumps({"receipt_sha256": receipt["receipt_sha256"], "status": receipt["status"]}))


if __name__ == "__main__":
    main()
