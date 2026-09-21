"""Digest-checked, atomic file publication into an existing Kubernetes Pod."""

from __future__ import annotations

import hashlib
import re
import subprocess
import uuid
from collections.abc import Callable
from pathlib import Path, PurePosixPath

from .jobs import JobsError


def _sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _identifier(value: str, label: str) -> str:
    if not value or value.startswith("-") or not re.fullmatch(r"[-A-Za-z0-9_.:@/]+", value):
        raise JobsError(f"invalid {label}")
    return value


def _destination(value: str) -> str:
    path = PurePosixPath(value)
    if not path.is_absolute() or value.endswith("/") or any(c in value for c in "\n\r\0"):
        raise JobsError("destination must be an absolute file path")
    return value


class PodTransfer:
    """Publish a local file only after a complete remote digest check.

    The destination never names the in-progress copy.  A unique sibling is
    copied first, checked inside the Pod, and hard-linked to the create-once
    final path.  ``ln`` provides atomic no-replace publication on the Pod's
    filesystem; a watcher can therefore observe either no final file or the
    complete verified file, never partial bytes.
    """

    def __init__(
        self,
        context: str,
        *,
        binary: str = "kubectl",
        run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ):
        self.context = _identifier(context, "Kubernetes context")
        self.binary = _identifier(binary, "kubectl binary")
        self._run_command = run

    def _run(self, args: list[str], operation: str) -> str:
        try:
            result = self._run_command(
                [
                    self.binary,
                    "--context",
                    self.context,
                    "--request-timeout=300s",
                    *args,
                ],
                text=True,
                capture_output=True,
                timeout=330,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            raise JobsError(f"kubectl {operation} transport failed") from None
        if result.returncode:
            raise JobsError(f"kubectl {operation} failed; remote output suppressed")
        return result.stdout

    def publish(
        self,
        source: Path,
        *,
        namespace: str,
        pod: str,
        destination: str,
        container: str | None = None,
    ) -> dict:
        if source.is_symlink():
            raise JobsError("source must be one existing regular file")
        source = source.resolve()
        if not source.is_file():
            raise JobsError("source must be one existing regular file")
        namespace = _identifier(namespace, "namespace")
        pod = _identifier(pod, "Pod name")
        if container is not None:
            container = _identifier(container, "container name")
        destination = _destination(destination)
        expected_sha256 = _sha256(source)
        expected_size = source.stat().st_size
        partial = f"{destination}.partial-{expected_sha256[:12]}-{uuid.uuid4().hex[:8]}"
        copy_target = f"{namespace}/{pod}"
        container_args = ["--container", container] if container else []

        # Check both names before copying.  The final hard-link below repeats
        # the no-replace decision atomically, closing the intervening race.
        self._run(
            [
                "exec",
                "--namespace",
                namespace,
                pod,
                *container_args,
                "--",
                "sh",
                "-c",
                'test -d "$(dirname "$2")" && test ! -e "$1" && test ! -L "$1" '
                '&& test ! -e "$2" && test ! -L "$2"',
                "sh",
                partial,
                destination,
            ],
            "destination precheck",
        )
        self._run(
            ["cp", *container_args, str(source), f"{copy_target}:{partial}"],
            "copy",
        )
        output = self._run(
            [
                "exec",
                "--namespace",
                namespace,
                pod,
                *container_args,
                "--",
                "sh",
                "-c",
                'set -eu; actual=$(sha256sum "$1"); actual=${actual%% *}; '
                '[ "$actual" = "$3" ]; [ "$(wc -c < "$1" | tr -d "[:space:]")" = "$4" ]; '
                'ln -- "$1" "$2"; rm -- "$1"; '
                'actual=$(sha256sum "$2"); actual=${actual%% *}; '
                'size=$(wc -c < "$2" | tr -d "[:space:]"); printf "%s %s\\n" "$actual" "$size"',
                "sh",
                partial,
                destination,
                expected_sha256,
                str(expected_size),
            ],
            "digest check and atomic publication",
        )
        fields = output.strip().split()
        if fields != [expected_sha256, str(expected_size)]:
            raise JobsError("remote publication proof did not match the local file")
        return {
            "source": str(source),
            "destination": destination,
            "sha256": expected_sha256,
            "bytes": expected_size,
            "namespace": namespace,
            "pod": pod,
            "container": container,
            "published": True,
        }
