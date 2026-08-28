"""Pinned checkout helpers with no credential handling."""

from __future__ import annotations

import subprocess
from pathlib import Path

from .pins import BenchmarkPin


class WorkspaceError(RuntimeError):
    """Safe-to-display workspace failure."""


def _run(argv: list[str], *, cwd: Path | None = None) -> str:
    try:
        result = subprocess.run(
            argv,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise WorkspaceError(f"command failed: {argv[0]} {argv[1]}") from exc
    return result.stdout.strip()


def verify_checkout(path: Path, pin: BenchmarkPin) -> None:
    if not (path / ".git").is_dir():
        raise WorkspaceError(f"not a git checkout: {path}")
    head = _run(["git", "rev-parse", "HEAD"], cwd=path)
    if head != pin.commit:
        raise WorkspaceError(f"checkout HEAD is {head}; expected {pin.commit}")
    origin = _run(["git", "remote", "get-url", "origin"], cwd=path)
    normalized = origin.removesuffix("/").removesuffix(".git")
    expected = pin.repository.removesuffix("/").removesuffix(".git")
    if normalized != expected:
        raise WorkspaceError("checkout origin does not match the official repository")


def materialize_checkout(path: Path, pin: BenchmarkPin) -> None:
    if path.exists():
        verify_checkout(path, pin)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    _run(["git", "clone", "--filter=blob:none", "--no-checkout", pin.repository, str(path)])
    _run(["git", "checkout", "--detach", pin.commit], cwd=path)
    verify_checkout(path, pin)


def prepare_xben_compose(
    *,
    checkout: Path,
    target: str,
    agent_context: Path,
) -> Path:
    """Write a secret-free Compose wrapper with an outbound-denied runtime network."""

    challenge_compose = checkout / "benchmarks" / target / "docker-compose.yml"
    if not challenge_compose.is_file():
        raise WorkspaceError(f"challenge compose file is missing: {challenge_compose}")
    dockerfile = agent_context / "Dockerfile.agent"
    if not dockerfile.is_file():
        raise WorkspaceError(f"agent Dockerfile is missing: {dockerfile}")

    runtime_dir = checkout.parent / ".cyber-post-train-secondary" / target
    runtime_dir.mkdir(parents=True, exist_ok=True)
    compose_path = runtime_dir / "compose.yaml"
    # Compose's include keeps official application bytes and build contexts
    # untouched. The declared default network is internal at runtime; Inspect
    # makes model calls from the host rather than from this network.
    compose_path.write_text(
        "include:\n"
        f"  - path: {json_quote(str(challenge_compose.resolve()))}\n"
        "services:\n"
        "  default:\n"
        "    build:\n"
        f"      context: {json_quote(str(agent_context.resolve()))}\n"
        "      dockerfile: Dockerfile.agent\n"
        '    command: ["sleep", "infinity"]\n'
        "    init: true\n"
        "networks:\n"
        "  default:\n"
        "    internal: true\n",
        encoding="utf-8",
    )
    compose_path.chmod(0o600)
    return compose_path


def json_quote(value: str) -> str:
    """Use JSON string quoting, which is valid YAML and avoids path injection."""

    import json

    return json.dumps(value)
