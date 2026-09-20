import hashlib
import subprocess

import pytest
from typer.testing import CliRunner

from cyber_post_train import cli, pod_transfer
from cyber_post_train.jobs import JobsError
from cyber_post_train.pod_transfer import PodTransfer


class FakeRun:
    def __init__(self, digest, size, *, fail_at=None):
        self.digest = digest
        self.size = size
        self.fail_at = fail_at
        self.calls = []

    def __call__(self, command, **kwargs):
        self.calls.append((command, kwargs))
        index = len(self.calls)
        return subprocess.CompletedProcess(
            command,
            1 if self.fail_at == index else 0,
            stdout=f"{self.digest} {self.size}\n" if index == 3 else "",
            stderr="private remote output",
        )


def test_publish_copies_to_unique_partial_then_atomically_links(tmp_path, monkeypatch):
    source = tmp_path / "bundle.tar.gz"
    source.write_bytes(b"complete immutable archive")
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    run = FakeRun(expected, source.stat().st_size)
    monkeypatch.setattr(
        "cyber_post_train.pod_transfer.uuid.uuid4",
        lambda: type("U", (), {"hex": "12345678abcdef"})(),
    )

    result = PodTransfer("prod-context", run=run).publish(
        source,
        namespace="fleet-train-jobs",
        pod="cpu-preflight-abc",
        container="preflight",
        destination="/tmp/prepared.tar.gz",
    )

    assert result["sha256"] == expected and result["published"] is True
    assert len(run.calls) == 3
    precheck, copy, publish = (call[0] for call in run.calls)
    partial = f"/tmp/prepared.tar.gz.partial-{expected[:12]}-12345678"
    assert precheck[-2:] == [partial, "/tmp/prepared.tar.gz"]
    namespace_index = precheck.index("--namespace")
    assert precheck[namespace_index : namespace_index + 3] == [
        "--namespace",
        "fleet-train-jobs",
        "cpu-preflight-abc",
    ]
    assert copy[-2:] == [str(source.resolve()), f"fleet-train-jobs/cpu-preflight-abc:{partial}"]
    assert "ln --" in publish[publish.index("-c") + 1]
    assert publish[-4:] == [partial, "/tmp/prepared.tar.gz", expected, str(source.stat().st_size)]
    assert all(call[1]["capture_output"] is True for call in run.calls)


@pytest.mark.parametrize("fail_at", [1, 2, 3])
def test_publish_fails_closed_and_never_retries(tmp_path, fail_at):
    source = tmp_path / "bundle"
    source.write_bytes(b"bytes")
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    run = FakeRun(expected, source.stat().st_size, fail_at=fail_at)

    with pytest.raises(JobsError, match="remote output suppressed|transport"):
        PodTransfer("prod-context", run=run).publish(
            source,
            namespace="fleet-train-jobs",
            pod="cpu-preflight",
            destination="/tmp/input.tar.gz",
        )
    assert len(run.calls) == fail_at


@pytest.mark.parametrize(
    "source_name,destination",
    [("missing", "/tmp/file"), ("directory", "/tmp/file"), ("file", "relative")],
)
def test_publish_rejects_invalid_local_or_remote_paths_before_kubectl(
    tmp_path, source_name, destination
):
    source = tmp_path / source_name
    if source_name == "directory":
        source.mkdir()
    elif source_name == "file":
        source.write_text("x")
    run = FakeRun("0" * 64, 1)
    with pytest.raises(JobsError):
        PodTransfer("prod-context", run=run).publish(
            source,
            namespace="fleet-train-jobs",
            pod="cpu-preflight",
            destination=destination,
        )
    assert run.calls == []


def test_publish_rejects_local_symlink_before_kubectl(tmp_path):
    target = tmp_path / "target"
    target.write_bytes(b"bytes")
    source = tmp_path / "link"
    source.symlink_to(target)
    run = FakeRun("0" * 64, 1)
    with pytest.raises(JobsError, match="regular file"):
        PodTransfer("prod-context", run=run).publish(
            source,
            namespace="fleet-train-jobs",
            pod="cpu-preflight",
            destination="/tmp/input.tar.gz",
        )
    assert run.calls == []


def test_cli_dispatches_atomic_publisher(tmp_path, monkeypatch):
    source = tmp_path / "bundle"
    source.write_bytes(b"sealed")
    calls = []

    class FakeTransfer:
        def __init__(self, context):
            calls.append(("context", context))

        def publish(self, path, **kwargs):
            calls.append((path, kwargs))
            return {"published": True, "sha256": "a" * 64}

    monkeypatch.setattr(pod_transfer, "PodTransfer", FakeTransfer)
    result = CliRunner().invoke(
        cli.app,
        [
            "pod-publish-file",
            str(source),
            "--context",
            "prod-context",
            "--pod",
            "preflight-pod",
            "--container",
            "preflight",
            "--destination",
            "/tmp/prepared.tar.gz",
        ],
    )
    assert result.exit_code == 0
    assert calls == [
        ("context", "prod-context"),
        (
            source,
            {
                "namespace": "fleet-train-jobs",
                "pod": "preflight-pod",
                "destination": "/tmp/prepared.tar.gz",
                "container": "preflight",
            },
        ),
    ]
