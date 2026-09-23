from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from evals.external_ctf import nyu_adapter
from evals.external_ctf import nyu_runtime_qualification as runtime
from evals.external_ctf.protocol import digest, load_protocol


def _task(benchmark: dict, index: int) -> nyu_adapter.NyuTask:
    return nyu_adapter.NyuTask(
        task_id=benchmark["task_ids"][index],
        source_commit=benchmark["source"]["commit"],
        challenge_sha256="sha256:" + "1" * 64,
        compose_sha256="sha256:" + "2" * 64,
        attachment_manifest_sha256="sha256:" + "3" * 64,
        target_host="target.internal",
        target_port=8080,
        _prompt="sealed",
        _flag="sealed",
        _attachments=(),
        _compose_images=("example.invalid/task:v1",),
    )


def test_alias_mismatch_requires_exact_runtime_reachability() -> None:
    assert runtime._target_binding("target", {"target"}, True) == "compose_alias"
    assert runtime._target_binding("declared", {"actual"}, True) == "runtime_proved_alias_mismatch"
    with pytest.raises(runtime.NyuRuntimeError, match="exact_target_not_reachable"):
        runtime._target_binding("declared", {"actual"}, False)


def test_external_aliases_are_read_from_the_exact_compose_network() -> None:
    compose = {
        "networks": {"challenge": {"external": True}, "private": {}},
        "services": {
            "web": {"networks": {"challenge": {"aliases": ["declared"]}}},
            "database": {"networks": ["private"]},
        },
    }
    assert runtime._external_aliases(compose) == ("challenge", {"web", "declared"})


def test_qualification_matches_the_existing_adapter_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = load_protocol()
    benchmark = protocol["benchmarks"][runtime.BENCHMARK]
    task = nyu_adapter.NyuTask(
        task_id=benchmark["task_ids"][0],
        source_commit=benchmark["source"]["commit"],
        challenge_sha256="sha256:" + "1" * 64,
        compose_sha256="sha256:" + "2" * 64,
        attachment_manifest_sha256="sha256:" + "3" * 64,
        target_host="target.internal",
        target_port=8080,
        _prompt="sealed",
        _flag="sealed",
        _attachments=(),
        _compose_images=("example.invalid/task:v1",),
    )
    images = [
        {
            "reference": "example.invalid/task:v1",
            "image_id": "sha256:" + "4" * 64,
            "repository_digest": "example.invalid/task@sha256:" + "5" * 64,
        }
    ]
    seen: list[dict] = []
    monkeypatch.setattr(
        nyu_adapter, "_validate_qualification", lambda _p, _t, row: seen.append(row)
    )

    receipt = runtime._qualification(protocol, task, images)

    assert seen == [receipt]
    assert receipt["grader_negative_control"] is False
    assert receipt["grader_positive_control"] is True
    assert receipt["receipt_sha256"] == digest(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    assert "flag" not in repr(receipt).lower()


def test_one_task_invocation_emits_one_sanitized_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = load_protocol()
    benchmark = protocol["benchmarks"][runtime.BENCHMARK]
    index = 6
    task_id = benchmark["task_ids"][index]
    harness = {
        "reference": benchmark["harness"]["image_id"],
        "image_id": benchmark["harness"]["image_id"],
        "repository_digest": None,
    }
    monkeypatch.setattr(
        runtime.rt,
        "require_task",
        lambda *_args: (protocol, benchmark, task_id),
    )
    monkeypatch.setattr(runtime.rt, "require_docker_linux_amd64", lambda: None)
    monkeypatch.setattr(runtime.rt, "image_lock", lambda *_args, **_kwargs: harness)
    monkeypatch.setattr(runtime.nyu_adapter, "load_task", lambda *_args: _task(benchmark, index))

    def execute(*args):
        args[-1]["challenge_runtime_images"] = [
            {
                "service": "web",
                "reference": "sha256:" + "b" * 64,
                "image_id": "sha256:" + "b" * 64,
                "repository_digest": None,
            }
        ]
        return {"receipt_sha256": "sha256:" + "a" * 64}, "compose_alias"

    monkeypatch.setattr(runtime, "_execute", execute)

    receipt = runtime.qualify(Path("/unused"), index)

    assert receipt["qualification_name"] == "extctf-nyu-t06-qual-v1"
    assert receipt["task_index"] == index
    assert receipt["task_source_sha256"] == digest(receipt["task_source"])
    assert receipt["status"] == "runtime_qualified"
    assert receipt["challenge_runtime_images_sha256"] == digest(receipt["challenge_runtime_images"])
    assert receipt["official_solution_solvability_claimed"] is False
    assert receipt["model_requests"] == receipt["provider_calls"] == 0
    assert receipt["receipt_sha256"] == digest(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )


def test_runtime_failure_is_sanitized_and_fails_only_that_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = load_protocol()
    benchmark = protocol["benchmarks"][runtime.BENCHMARK]
    monkeypatch.setattr(
        runtime.rt,
        "require_task",
        lambda *_args: (protocol, benchmark, benchmark["task_ids"][0]),
    )
    monkeypatch.setattr(runtime.rt, "require_docker_linux_amd64", lambda: None)
    monkeypatch.setattr(runtime.nyu_adapter, "load_task", lambda *_args: _task(benchmark, 0))
    monkeypatch.setattr(
        runtime.rt,
        "image_lock",
        lambda *_args, **_kwargs: {
            "reference": benchmark["harness"]["image_id"],
            "image_id": benchmark["harness"]["image_id"],
            "repository_digest": None,
        },
    )
    images = [
        {
            "service": "web",
            "reference": "sha256:" + "4" * 64,
            "image_id": "sha256:" + "4" * 64,
            "repository_digest": None,
        }
    ]

    def fail(*args):
        args[-1]["challenge_runtime_images"] = images
        raise runtime.NyuRuntimeError("sealed_failure")

    monkeypatch.setattr(runtime, "_execute", fail)

    receipt = runtime.qualify(Path("/unused"), 0)

    assert receipt["status"] == "infrastructure_invalid"
    assert receipt["error_code"] == "sealed_failure"
    assert receipt["qualification"] is None
    assert receipt["task_source_sha256"] == digest(receipt["task_source"])
    assert receipt["challenge_runtime_images"] == images
    assert receipt["challenge_runtime_images_sha256"] == digest(images)


def test_admission_failure_emits_non_denominator_precondition_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runtime.rt,
        "require_task",
        lambda *_args: (_ for _ in ()).throw(
            runtime.NyuRuntimeError("pinned_source_checkout_not_clean")
        ),
    )
    monkeypatch.setattr(
        runtime.rt,
        "require_docker_linux_amd64",
        lambda: pytest.fail("runtime must not start"),
    )

    receipt = runtime.qualify(Path("/unused"), 0)

    assert receipt["status"] == "precondition_failed"
    assert receipt["qualification_name"] is None
    assert receipt["error_code"] == "pinned_source_checkout_not_clean"
    assert receipt["qualification"] is None


def test_execute_uses_two_fresh_starts_before_grader_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = load_protocol()
    benchmark = protocol["benchmarks"][runtime.BENCHMARK]
    reference = "example.invalid/task@sha256:" + "1" * 64
    image_id = "sha256:" + "2" * 64
    lock = {
        "reference": reference,
        "image_id": image_id,
        "repository_digest": reference,
    }
    running = [
        {
            "service": "web",
            "reference": image_id,
            "image_id": image_id,
            "repository_digest": None,
        }
    ]
    events: list[object] = []

    class Task:
        task_id = benchmark["task_ids"][0]
        _compose_images = (reference,)
        _flag = "FLAG{sealed}"
        challenge_sha256 = "sha256:" + "3" * 64
        compose_sha256 = "sha256:" + "4" * 64
        attachment_manifest_sha256 = "sha256:" + "5" * 64
        target_host = "target.internal"
        target_port = 8080

        def score(self, answer: str) -> int:
            events.append(("score", answer))
            return int(answer == self._flag)

    class Sandbox:
        command = ["compose"]
        network = "qualification-net"

        def __init__(self, *_args):
            pass

        def __enter__(self):
            events.append("enter")
            return self

        def __exit__(self, *_args):
            events.append("exit")

        def image_references(self):
            events.append("references")
            return [reference]

        def start(self, *, build: bool, timeout: int):
            events.append(("start", build, timeout))

        def running_images(self):
            events.append("running_images")
            return running

        def down(self):
            events.append("down")

        def create(self, role: str, *_args):
            events.append(("create", role))
            return "agent"

    def materialize(_checkout, _commit, _path, destination):
        events.append("materialize")
        destination.mkdir(parents=True)
        (destination / "docker-compose.yml").write_text(
            "networks:\n  challenge:\n    external: true\n"
            "services:\n  web:\n    image: sealed\n    networks:\n"
            "      challenge:\n        aliases: [target.internal]\n"
        )

    monkeypatch.setattr(runtime, "_task_path", lambda *_args: "task")
    monkeypatch.setattr(runtime.nyu_adapter, "load_task", lambda *_args: Task())
    monkeypatch.setattr(runtime.rt, "materialize", materialize)
    monkeypatch.setattr(
        runtime.rt,
        "contained_compose",
        lambda _compose, root: events.append("compose_safe") or (root / "docker-compose.yml", 0, 0),
    )
    monkeypatch.setattr(runtime.rt, "Sandbox", Sandbox)
    monkeypatch.setattr(
        runtime.rt,
        "run",
        lambda *_args, **_kwargs: events.append("pull") or SimpleNamespace(stdout=""),
    )
    monkeypatch.setattr(
        runtime.rt,
        "image_lock",
        lambda *_args, **_kwargs: events.append("image_lock") or lock,
    )
    monkeypatch.setattr(
        runtime.rt,
        "assert_isolated",
        lambda *_args, **_kwargs: events.append("isolated"),
    )
    monkeypatch.setattr(runtime.rt, "probe", lambda *_args: events.append("probe") or True)
    monkeypatch.setattr(
        runtime,
        "_qualification",
        lambda *_args: events.append("qualification") or {"status": "passed"},
    )
    observed = {"challenge_runtime_images": None}

    qualification, binding = runtime._execute(protocol, Path("/unused"), Task(), 0, observed)

    assert qualification == {"status": "passed"}
    assert binding == "compose_alias"
    assert observed["challenge_runtime_images"] == running
    assert events == [
        "materialize",
        "compose_safe",
        "enter",
        "references",
        "pull",
        "image_lock",
        ("start", False, 900),
        "running_images",
        "down",
        ("start", False, 900),
        "running_images",
        ("create", "agent"),
        "isolated",
        "probe",
        ("score", "EXTERNAL_CTF_RUNTIME_QUALIFICATION_DECOY"),
        ("score", "FLAG{sealed}"),
        "running_images",
        "image_lock",
        "qualification",
        "exit",
    ]
