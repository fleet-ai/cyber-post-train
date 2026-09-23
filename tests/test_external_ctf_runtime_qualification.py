from __future__ import annotations

import json
import stat
import subprocess
from pathlib import Path

import pytest

from evals.external_ctf import runtime_qualification as runtime
from evals.external_ctf.protocol import DEFAULT_PROTOCOL, load_protocol


def _completed(output: str = "", code: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], code, output, "")


def test_unavailable_and_out_of_range_tasks_fail_before_runtime_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = load_protocol()
    monkeypatch.setattr(runtime.platform, "system", lambda: "Linux")
    monkeypatch.setattr(runtime.platform, "machine", lambda: "x86_64")
    for name in runtime.MODEL_CREDENTIALS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(runtime, "run", lambda *_args, **_kwargs: pytest.fail("no runtime action"))
    expected = {
        "nyu_ctf_web_test": {4, 5, 7},
        "cybench_web": {1},
    }
    for benchmark_name, unavailable in expected.items():
        benchmark = protocol["benchmarks"][benchmark_name]
        observed = {
            index
            for index, task_id in enumerate(benchmark["task_ids"])
            if task_id
            in set(benchmark["source_unavailable_task_ids"])
            | set(benchmark["execution_unavailable_task_ids"])
        }
        assert observed == unavailable
        for index in unavailable:
            with pytest.raises(runtime.RuntimeGateError, match="task_runtime_unavailable"):
                runtime.require_task(Path("/unused"), DEFAULT_PROTOCOL, benchmark_name, index)
        with pytest.raises(runtime.RuntimeGateError, match="task_index_invalid"):
            runtime.require_task(
                Path("/unused"), DEFAULT_PROTOCOL, benchmark_name, len(benchmark["task_ids"])
            )


def test_materialize_uses_exact_commit_and_excludes_ignored_bytes(tmp_path: Path) -> None:
    checkout = tmp_path / "source"
    checkout.mkdir()
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    subprocess.run(
        ["git", "-C", str(checkout), "config", "user.email", "test@example.com"], check=True
    )
    subprocess.run(["git", "-C", str(checkout), "config", "user.name", "Test"], check=True)
    (checkout / ".gitignore").write_text("task/ignored.txt\n")
    (checkout / "task").mkdir()
    (checkout / "task/committed.txt").write_text("pinned")
    (checkout / "task/linked.txt").symlink_to("committed.txt")
    subprocess.run(["git", "-C", str(checkout), "add", "."], check=True)
    subprocess.run(["git", "-C", str(checkout), "commit", "-qm", "pinned"], check=True)
    commit = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    (checkout / "task/ignored.txt").write_text("mutable")

    destination = tmp_path / "materialized"
    runtime.materialize(checkout, commit, "task", destination)

    assert (destination / "committed.txt").read_text() == "pinned"
    assert (destination / "linked.txt").is_symlink()
    assert (destination / "linked.txt").read_text() == "pinned"
    assert not (destination / "ignored.txt").exists()

    (checkout / "task/escape").symlink_to("../outside")
    subprocess.run(["git", "-C", str(checkout), "add", "task/escape"], check=True)
    subprocess.run(["git", "-C", str(checkout), "commit", "-qm", "escape"], check=True)
    unsafe_commit = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    with pytest.raises(runtime.RuntimeGateError, match="pinned_source_archive_invalid"):
        runtime.materialize(checkout, unsafe_commit, "task", tmp_path / "unsafe")


def test_compose_safety_rejects_host_access_and_allows_exact_read_only_inputs(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    path, removed, removed_names = runtime.contained_compose(
        {
            "services": {
                "web": {
                    "container_name": "ignored-fixed-name",
                    "volumes": ["./source:/app:ro"],
                    "ports": ["12345:80"],
                }
            }
        },
        tmp_path,
    )
    assert (removed, removed_names) == (1, 1)
    service = runtime.yaml.safe_load(path.read_bytes())["services"]["web"]
    assert "ports" not in service and "container_name" not in service
    unsafe = [
        {"services": {"web": {"volumes": ["/var/run/docker.sock:/var/run/docker.sock"]}}},
        {"services": {"web": {"privileged": True}}},
        {"services": {"web": {"network_mode": "host"}}},
        {"services": {"web": {"devices": ["/dev/kvm:/dev/kvm"]}}},
        {"services": {"web": {"extra_hosts": ["cache:host-gateway"]}}},
    ]
    for compose in unsafe:
        with pytest.raises(runtime.RuntimeGateError, match="unsafe_compose_contract"):
            runtime.require_safe_compose(compose, tmp_path)

    reserved = tmp_path / ".qualification-source-compose.yml"
    with pytest.raises(runtime.RuntimeGateError, match="qualification_control_path_exists"):
        runtime.contained_compose({"services": {"web": {}}}, tmp_path)
    assert "ports" not in runtime.yaml.safe_load(reserved.read_bytes())["services"]["web"]


def test_image_lock_requires_linux_amd64_and_exact_immutable_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = "example.invalid/controller@sha256:" + "1" * 64
    row = {
        "Id": "sha256:" + "2" * 64,
        "RepoDigests": [reference, "mirror.invalid/controller@sha256:" + "3" * 64],
        "Os": "linux",
        "Architecture": "amd64",
    }
    monkeypatch.setattr(runtime, "docker", lambda *_args, **_kwargs: _completed(json.dumps([row])))
    assert runtime.image_lock(reference, repository_digest=True)["repository_digest"] == reference
    with pytest.raises(runtime.RuntimeGateError, match="resolved_repository_digest_mismatch"):
        runtime.image_lock(
            "example.invalid/controller@sha256:" + "f" * 64,
            repository_digest=True,
        )


def test_docker_server_must_be_linux_amd64(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime, "docker", lambda *_args, **_kwargs: _completed("linux/amd64\n"))
    runtime.require_docker_linux_amd64()
    monkeypatch.setattr(runtime, "docker", lambda *_args, **_kwargs: _completed("linux/arm64\n"))
    with pytest.raises(runtime.RuntimeGateError, match="docker_server_linux_amd64_required"):
        runtime.require_docker_linux_amd64()


def test_only_owned_errors_can_supply_receipt_reason_codes() -> None:
    assert runtime.failure_code(runtime.RuntimeGateError("owned_reason")) == "owned_reason"
    assert runtime.failure_code(RuntimeError("forged_reason")) == "unexpected_runtime_error"
    assert (
        runtime.failure_code(subprocess.CalledProcessError(1, ["private", "command"]))
        == "runtime_command_failed"
    )


def test_container_isolation_is_exact(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workspace = tmp_path.resolve()
    row = {
        "HostConfig": {
            "Privileged": False,
            "ReadonlyRootfs": True,
            "CapDrop": ["ALL"],
            "SecurityOpt": ["no-new-privileges"],
        },
        "Config": {"User": "1000:1000"},
        "NetworkSettings": {"Networks": {"challenge": {}}},
        "Mounts": [{"Source": str(workspace), "Destination": "/workspace", "RW": True}],
    }
    monkeypatch.setattr(runtime, "docker", lambda *_args, **_kwargs: _completed(json.dumps([row])))
    runtime.assert_isolated(
        "agent", "challenge", {(str(workspace), "/workspace", True)}, hardened_user=True
    )
    row["NetworkSettings"]["Networks"]["bridge"] = {}
    with pytest.raises(runtime.RuntimeGateError, match="container_isolation_invalid"):
        runtime.assert_isolated(
            "agent", "challenge", {(str(workspace), "/workspace", True)}, hardened_user=True
        )


def test_running_images_are_bound_from_containers_not_mutable_tags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sandbox = runtime.Sandbox.__new__(runtime.Sandbox)
    sandbox.state = lambda: [{"Service": "web", "ID": "container-id"}]
    image_id = "sha256:" + "1" * 64
    monkeypatch.setattr(
        runtime, "docker", lambda *_args, **_kwargs: _completed(json.dumps([{"Image": image_id}]))
    )
    monkeypatch.setattr(
        runtime,
        "image_lock",
        lambda reference, **_kwargs: {
            "reference": reference,
            "image_id": reference,
            "repository_digest": None,
        },
    )
    assert sandbox.running_images() == [
        {
            "service": "web",
            "reference": image_id,
            "image_id": image_id,
            "repository_digest": None,
        }
    ]


def test_sandbox_cleanup_is_verified(monkeypatch: pytest.MonkeyPatch) -> None:
    sandbox = runtime.Sandbox.__new__(runtime.Sandbox)
    sandbox.containers = ["qualification-agent"]
    sandbox.stack_attempted = True
    sandbox.network = "qualification-net"
    sandbox.command = ["docker", "compose", "--project-name", "qualification"]
    calls: list[tuple[str, ...]] = []

    def docker(*args: str, **_kwargs):
        calls.append(args)
        return _completed()

    def run(command: list[str], **_kwargs):
        calls.append(tuple(command))
        return _completed()

    monkeypatch.setattr(runtime, "docker", docker)
    monkeypatch.setattr(runtime, "run", run)
    assert sandbox.__exit__(None, None, None) is False
    assert any("down" in call for call in calls)
    assert ("network", "rm", "qualification-net") in calls
    assert (
        "ps",
        "--all",
        "--filter",
        "name=^/qualification-agent$",
        "--format",
        "{{.ID}}",
    ) in calls
    assert (
        "network",
        "ls",
        "--filter",
        "name=^qualification-net$",
        "--format",
        "{{.ID}}",
    ) in calls


@pytest.mark.parametrize(
    "failure", ["container_query", "network_query", "compose_query", "residual_container"]
)
def test_sandbox_cleanup_query_failure_fails_closed(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    sandbox = runtime.Sandbox.__new__(runtime.Sandbox)
    sandbox.containers = ["qualification-agent"]
    sandbox.stack_attempted = True
    sandbox.network = "qualification-net"
    sandbox.command = ["docker", "compose", "--project-name", "qualification"]

    def docker(*args: str, **_kwargs):
        failed = (failure == "container_query" and args[0] == "ps") or (
            failure == "network_query" and args[:2] == ("network", "ls")
        )
        residual = failure == "residual_container" and args[0] == "ps"
        return _completed("remaining\n" if residual else "", 1 if failed else 0)

    def run(command: list[str], **_kwargs):
        failed = failure == "compose_query" and "ps" in command
        return _completed(code=1 if failed else 0)

    monkeypatch.setattr(runtime, "docker", docker)
    monkeypatch.setattr(runtime, "run", run)
    with pytest.raises(runtime.RuntimeGateError, match="qualification_cleanup_failed"):
        sandbox.__exit__(None, None, None)


def test_receipt_write_is_private_durable_and_create_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "receipt.json"
    fsyncs: list[int] = []
    original = runtime.os.fsync
    monkeypatch.setattr(
        runtime.os, "fsync", lambda descriptor: fsyncs.append(descriptor) or original(descriptor)
    )
    runtime.write_once(path, {"status": "sanitized"})
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert len(fsyncs) == 2 and len(set(fsyncs)) == 2
    with pytest.raises(FileExistsError):
        runtime.write_once(path, {"status": "replacement"})
    assert json.loads(path.read_bytes()) == {"status": "sanitized"}


def test_receipt_write_rejects_permissive_or_symlink_parent(tmp_path: Path) -> None:
    permissive = tmp_path / "permissive"
    permissive.mkdir(mode=0o755)
    permissive.chmod(0o755)
    with pytest.raises(runtime.RuntimeGateError, match="receipt_parent_not_private"):
        runtime.write_once(permissive / "receipt.json", {})

    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    linked = tmp_path / "linked"
    linked.symlink_to(private, target_is_directory=True)
    with pytest.raises(runtime.RuntimeGateError, match="receipt_parent_symlink_invalid"):
        runtime.write_once(linked / "new/receipt.json", {})
    assert not (private / "new").exists()
