"""Versioned hooks for the bounded semantic OpenCode startup probe.

The v1 evaluator is an immutable input to accepted historical plans. New
cluster studies bind these two hooks for the lifetime of their v2 entrypoint,
reusing the unchanged v1 scientific and MCP execution behavior.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

from evals.fleet import evaluate as v1
from evals.fleet import opencode_self_hosted as harness

RUNTIME_FILES = (
    "evaluate.py",
    "evaluate_v2.py",
    *v1.RUNTIME_FILES[1:],
)

# Safety ceiling, not an expected startup duration. The v1 probe launched two
# complete OpenCode processes under one 60-second budget, so a cold valid worker
# could fail. This probe starts OpenCode once and checks writable directories
# directly.
AGENT_STARTUP_PROBE_TIMEOUT_SECONDS = 300
AGENT_STARTUP_PROBE = """
test "$HOME" = /home/node
test "$(id -u)" -ne 0
for directory in \
  "$HOME/.config/opencode" \
  "$HOME/.cache/opencode" \
  "$HOME/.local/state/opencode" \
  "$HOME/.local/share/opencode"; do
  mkdir -p "$directory"
  probe=$(mktemp "$directory/.cyber-preflight.XXXXXX")
  rm "$probe"
done
opencode --version
""".strip()


def runtime_identity() -> dict[str, str]:
    """Return the exact v2 closure, including the immutable v1 implementation."""

    return {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in RUNTIME_FILES
    }


def check_images(plan: dict) -> None:
    """Validate immutable images and the bounded semantic OpenCode startup facts."""

    for image in plan["images"].values():
        result = subprocess.run(
            ["docker", "image", "inspect", image, "--format", "{{json .}}"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        if result.returncode:
            raise RuntimeError("immutable harness image is not staged on this worker")
        info = json.loads(result.stdout)
        local_id_matches = not image.startswith("sha256:") or info.get("Id") == image
        if not local_id_matches or info.get("Os") != "linux" or info.get("Architecture") != "amd64":
            raise RuntimeError("harness image bytes or platform differ")
        if image == plan["images"]["agent"]:
            labels = info.get("Config", {}).get("Labels") or {}
            if (
                labels.get("cyber.opencode.release-sha256")
                != plan["treatment"]["release_asset_sha256"]
            ):
                raise RuntimeError("harness release identity differs")

    bind_root = os.environ.get("DOCKER_BIND_ROOT")
    if bind_root is not None and not Path(bind_root).is_dir():
        raise RuntimeError("shared Docker bind root is unavailable")
    with tempfile.TemporaryDirectory(prefix="cpt-agent-preflight-", dir=bind_root) as home:
        if os.geteuid() == 0:
            os.chown(home, 1000, 1000)
        try:
            startup = subprocess.run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--platform",
                    "linux/amd64",
                    "--network",
                    "none",
                    "--pull",
                    "never",
                    *harness.agent_container_user_args(),
                    "-v",
                    f"{home}:/home/node",
                    plan["images"]["agent"],
                    "bash",
                    "-ceu",
                    "--",
                    AGENT_STARTUP_PROBE,
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=AGENT_STARTUP_PROBE_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("harness startup probe exceeded its safety deadline") from None
    if startup.returncode or startup.stdout.splitlines() != [plan["treatment"]["harness_version"]]:
        raise RuntimeError("harness version or writable agent-home startup differs")
