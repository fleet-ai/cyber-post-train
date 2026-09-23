from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from evals.external_ctf import cybench_runtime_qualification as runtime
from evals.external_ctf.protocol import digest, load_protocol


def _source_rows(benchmark: dict) -> list[dict]:
    return [
        {
            "task_index": index,
            "task_id_sha256": runtime.file_digest(task_id.encode()),
            "source_state": "present",
            "runtime_asset_manifest_sha256": "sha256:" + "9" * 64,
        }
        for index, task_id in enumerate(benchmark["task_ids"])
    ]


def _preflight(monkeypatch: pytest.MonkeyPatch, index: int = 2):
    protocol = load_protocol()
    benchmark = protocol["benchmarks"][runtime.BENCHMARK]
    task_id = benchmark["task_ids"][index]
    controller = "example.invalid/controller@sha256:" + "1" * 64
    monkeypatch.setattr(
        runtime.rt,
        "require_task",
        lambda *_args: (protocol, benchmark, task_id),
    )
    monkeypatch.setattr(runtime.rt, "require_docker_linux_amd64", lambda: None)
    monkeypatch.setattr(runtime.source_gate, "_source_rows", lambda *_args: _source_rows(benchmark))
    monkeypatch.setattr(runtime.source_gate, "_git_blob", lambda *_args: b"pinned")
    monkeypatch.setattr(
        runtime.source_gate,
        "_native_acceptance_controls",
        lambda _source: {"exact": True, "near_miss": True},
    )
    monkeypatch.setattr(
        runtime.source_gate,
        "_runner_controls",
        lambda _source: {"official_solution": True},
    )
    monkeypatch.setattr(
        runtime.rt,
        "docker",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, "", ""),
    )

    def image_lock(reference: str, *, repository_digest: bool):
        if repository_digest:
            return {
                "reference": reference,
                "image_id": "sha256:" + "2" * 64,
                "repository_digest": controller,
            }
        return {
            "reference": reference,
            "image_id": benchmark["harness"]["image_id"],
            "repository_digest": None,
        }

    monkeypatch.setattr(runtime.rt, "image_lock", image_lock)
    return protocol, benchmark, task_id, controller


def test_mutable_controller_image_emits_precondition_receipt_before_docker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = load_protocol()
    benchmark = protocol["benchmarks"][runtime.BENCHMARK]
    monkeypatch.setattr(
        runtime.rt,
        "require_task",
        lambda *_args: (protocol, benchmark, benchmark["task_ids"][0]),
    )
    monkeypatch.setattr(runtime.source_gate, "_source_rows", lambda *_args: _source_rows(benchmark))
    monkeypatch.setattr(
        runtime.rt,
        "docker",
        lambda *_args, **_kwargs: pytest.fail("Docker must not run for a mutable image"),
    )
    receipt = runtime.qualify(Path("/unused"), 0, "cybench/cybench:latest")
    assert receipt["status"] == "precondition_failed"
    assert receipt["qualification_name"] is None
    assert receipt["error_code"] == "immutable_controller_image_required"
    assert receipt["task_source_sha256"] == digest(receipt["task_source"])


def test_agent_workspace_rejects_answer_or_solution_artifact(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "benign.txt").write_text("benign")
    runtime._workspace_hides_solution(workspace, "sealed-answer", {"sha256:" + "1" * 64})
    (workspace / "leak.txt").write_text("prefix sealed-answer suffix")
    with pytest.raises(runtime.CybenchRuntimeError, match="official_solution_visible_to_agent"):
        runtime._workspace_hides_solution(workspace, "sealed-answer", set())


def test_frozen_present_task_census_predeclares_only_host_gateway_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = load_protocol()
    benchmark = protocol["benchmarks"][runtime.BENCHMARK]
    present = {
        index
        for index, task_id in enumerate(benchmark["task_ids"])
        if task_id not in benchmark["source_unavailable_task_ids"]
    }
    assert present == {0, 2, 3, 4, 5}
    assert {0} == runtime.HOST_GATEWAY_UNSAFE_TASKS

    _protocol, _benchmark, _task_id, controller = _preflight(monkeypatch, 0)
    monkeypatch.setattr(
        runtime.rt,
        "require_docker_linux_amd64",
        lambda: pytest.fail("predeclared unsafe task must stop before Docker"),
    )
    receipt = runtime.qualify(Path("/unused"), 0, controller)
    assert receipt["status"] == "infrastructure_invalid"
    assert receipt["qualification_name"] == "extctf-cyb-t00-qual-v1"
    assert receipt["error_code"] == "task_host_gateway_unsupported"


def test_one_task_receipt_binds_native_evidence_and_images(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol, benchmark, task_id, controller = _preflight(monkeypatch, 2)

    def execute(*args):
        args[-1].update(
            {
                "official_solution_manifest_sha256": "sha256:" + "3" * 64,
                "official_solution_helper_sha256": "sha256:" + "5" * 64,
                "challenge_images": [
                    {
                        "service": "challenge",
                        "reference": "sha256:" + "4" * 64,
                        "image_id": "sha256:" + "4" * 64,
                        "repository_digest": None,
                    }
                ],
            }
        )

    monkeypatch.setattr(runtime, "_execute", execute)

    receipt = runtime.qualify(Path("/unused"), 2, controller)

    assert receipt["qualification_name"] == "extctf-cyb-t02-qual-v1"
    assert receipt["task_index"] == 2
    assert receipt["task_id_sha256"] == runtime.file_digest(task_id.encode())
    assert receipt["task_source_sha256"] == digest(receipt["task_source"])
    assert receipt["status"] == "runtime_qualified"
    assert (
        receipt["task_evidence"]
        == runtime.source_gate.expected_remote_runtime_evidence(protocol)["tasks"][2]
    )
    assert receipt["task_evidence"]["opencode_image_id"] == benchmark["harness"]["image_id"]
    assert receipt["controller_image_sha256"]
    assert receipt["controller_image"]["repository_digest"] == controller
    assert receipt["harness_image"]["image_id"] == benchmark["harness"]["image_id"]
    assert receipt["challenge_image_set_sha256"] == digest(receipt["challenge_images"])
    assert receipt["official_solution_helper_sha256"] == "sha256:" + "5" * 64
    assert receipt["receipt_sha256"] == digest(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )


def test_task_failure_withholds_runtime_evidence_and_private_detail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _protocol, _benchmark, _task_id, controller = _preflight(monkeypatch)
    images = [
        {
            "service": "challenge",
            "reference": "sha256:" + "4" * 64,
            "image_id": "sha256:" + "4" * 64,
            "repository_digest": None,
        }
    ]

    def fail(*args):
        args[-1].update(
            {
                "official_solution_manifest_sha256": "sha256:" + "3" * 64,
                "official_solution_helper_sha256": "sha256:" + "5" * 64,
                "challenge_images": images,
            }
        )
        raise runtime.CybenchRuntimeError("sealed_failure")

    monkeypatch.setattr(runtime, "_execute", fail)

    receipt = runtime.qualify(Path("/unused"), 2, controller)

    assert receipt["status"] == "infrastructure_invalid"
    assert receipt["task_evidence"] is None
    assert receipt["error_code"] == "sealed_failure"
    assert receipt["challenge_images"] == images
    assert receipt["challenge_image_set_sha256"] == digest(images)
    assert receipt["official_solution_manifest_sha256"] == "sha256:" + "3" * 64
    assert receipt["official_solution_helper_sha256"] == "sha256:" + "5" * 64


def test_controller_digest_must_resolve_to_exact_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _protocol, _benchmark, _task_id, controller = _preflight(monkeypatch)
    original = runtime.rt.image_lock

    def mismatch(reference: str, *, repository_digest: bool):
        row = original(reference, repository_digest=repository_digest)
        if repository_digest:
            row["repository_digest"] = "example.invalid/other@sha256:" + "f" * 64
        return row

    monkeypatch.setattr(runtime.rt, "image_lock", mismatch)
    receipt = runtime.qualify(Path("/unused"), 2, controller)
    assert receipt["status"] == "infrastructure_invalid"
    assert receipt["error_code"] == "controller_image_digest_mismatch"
    assert receipt["controller_image_reference"] == controller


def test_source_admission_failure_is_not_task_runtime_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = "example.invalid/controller@sha256:" + "1" * 64
    monkeypatch.setattr(
        runtime.rt,
        "require_task",
        lambda *_args: (_ for _ in ()).throw(
            runtime.CybenchRuntimeError("task_runtime_unavailable")
        ),
    )
    monkeypatch.setattr(
        runtime.rt,
        "docker",
        lambda *_args, **_kwargs: pytest.fail("Docker must not run after admission failure"),
    )
    receipt = runtime.qualify(Path("/unused"), 1, controller)
    assert receipt["status"] == "precondition_failed"
    assert receipt["qualification_name"] is None
    assert receipt["error_code"] == "task_runtime_unavailable"


def test_native_source_control_failure_emits_receipt_before_docker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = load_protocol()
    benchmark = protocol["benchmarks"][runtime.BENCHMARK]
    controller = "example.invalid/controller@sha256:" + "1" * 64
    monkeypatch.setattr(
        runtime.rt,
        "require_task",
        lambda *_args: (protocol, benchmark, benchmark["task_ids"][0]),
    )
    monkeypatch.setattr(runtime.source_gate, "_source_rows", lambda *_args: _source_rows(benchmark))
    monkeypatch.setattr(runtime.source_gate, "_git_blob", lambda *_args: b"pinned")
    monkeypatch.setattr(
        runtime.source_gate,
        "_native_acceptance_controls",
        lambda *_args: (_ for _ in ()).throw(runtime.CybenchRuntimeError("native_control_invalid")),
    )
    monkeypatch.setattr(
        runtime.rt,
        "docker",
        lambda *_args, **_kwargs: pytest.fail("Docker must not run after source rejection"),
    )

    receipt = runtime.qualify(Path("/unused"), 0, controller)

    assert receipt["status"] == "precondition_failed"
    assert receipt["error_code"] == "native_control_invalid"
    assert receipt["controller_image_reference"] == controller


def test_execute_restarts_before_native_false_then_true_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = load_protocol()
    benchmark = protocol["benchmarks"][runtime.BENCHMARK]
    task_id = benchmark["task_ids"][0]
    controller_image = "example.invalid/controller@sha256:" + "1" * 64
    running = [
        {
            "service": "challenge",
            "reference": "sha256:" + "2" * 64,
            "image_id": "sha256:" + "2" * 64,
            "repository_digest": None,
        }
    ]
    events: list[object] = []

    class Sandbox:
        network = "qualification-net"

        def __init__(self, *_args):
            pass

        def __enter__(self):
            events.append("enter")
            return self

        def __exit__(self, *_args):
            events.append("exit")

        def start(self, *, build: bool, timeout: int):
            events.append(("start", build, timeout))

        def running_images(self):
            events.append("running_images")
            return running

        def down(self):
            events.append("down")

        def create(self, role: str, *_args):
            events.append(("create", role))
            return role

    def materialize(_checkout, _commit, _path, destination):
        events.append("materialize")
        solution = destination / "metadata/solution"
        solution.mkdir(parents=True)
        (destination / "metadata/metadata.json").write_text(
            '{"target_host":"target.internal:8080","subtasks":[{"answer":"FLAG{sealed}"}]}'
        )
        (solution / "solution.sh").write_text("#!/bin/sh\necho FLAG{sealed}\n")
        (solution / "flag.txt").write_text("FLAG{sealed}\n")
        (destination / "docker-compose.yml").write_text(
            "networks:\n  challenge:\n    external: true\n"
            "services:\n  web:\n    image: sealed\n    networks: [challenge]\n"
        )

    def docker(*args: str, **_kwargs):
        if args[0] == "exec" and args[1] == "controller":
            command = " ".join(args)
            if "requirements.sh" in command:
                events.append("requirements")
            elif "init_script.sh" in command:
                events.append("init")
            elif "chown -R 1000:1000 /workspace" in command:
                events.append("chown")
            elif "exec bash ./solution.sh" in command:
                events.append("solution")
                return SimpleNamespace(stdout="noise\nFLAG{sealed}\n")
        elif args[0] == "exec" and args[1] == "agent":
            events.append("workspace_probe")
        elif args[:3] == ("network", "disconnect", "bridge"):
            events.append("disconnect")
        elif args[:2] == ("network", "connect"):
            events.append("connect")
        return SimpleNamespace(stdout="")

    def native_accepts(_source: bytes, candidate: str, answer: str) -> bool:
        events.append(("native", candidate))
        return candidate == answer

    monkeypatch.setattr(runtime.rt, "materialize", materialize)
    monkeypatch.setattr(
        runtime.rt,
        "contained_compose",
        lambda _compose, root: events.append("compose_safe") or (root / "docker-compose.yml", 0, 0),
    )
    monkeypatch.setattr(runtime.source_gate, "_git_blob", lambda *_args: b"helper")
    monkeypatch.setattr(runtime.rt, "Sandbox", Sandbox)
    monkeypatch.setattr(runtime.rt, "docker", docker)
    monkeypatch.setattr(
        runtime,
        "_workspace_hides_solution",
        lambda *_args: events.append("solution_hidden"),
    )
    monkeypatch.setattr(
        runtime.rt,
        "assert_isolated",
        lambda container, *_args, **_kwargs: events.append(("isolated", container)),
    )
    monkeypatch.setattr(
        runtime.rt,
        "probe",
        lambda *_args: events.append("probe") or True,
    )
    monkeypatch.setattr(runtime, "_native_accepts", native_accepts)
    observed = {
        "official_solution_manifest_sha256": None,
        "official_solution_helper_sha256": None,
        "challenge_images": None,
    }

    runtime._execute(
        protocol,
        Path("/unused"),
        task_id,
        0,
        controller_image,
        b"acceptance",
        observed,
    )

    assert observed["challenge_images"] == running
    assert observed["official_solution_manifest_sha256"]
    assert observed["official_solution_helper_sha256"]
    assert events == [
        "materialize",
        "compose_safe",
        "enter",
        ("start", True, 1800),
        "running_images",
        "down",
        ("start", False, 1800),
        "running_images",
        ("create", "controller"),
        "requirements",
        "init",
        "chown",
        "solution_hidden",
        "disconnect",
        "connect",
        ("isolated", "controller"),
        ("create", "agent"),
        ("isolated", "agent"),
        "workspace_probe",
        "probe",
        ("native", "EXTERNAL_CTF_RUNTIME_QUALIFICATION_DECOY"),
        "solution",
        ("native", "FLAG{sealed}"),
        "solution_hidden",
        "running_images",
        "exit",
    ]
