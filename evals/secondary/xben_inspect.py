"""Minimal Inspect task for one official XBEN target.

This module is loaded by ``inspect eval`` and is intentionally absent from the
training dependency graph.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.agent import react
from inspect_ai.dataset import Sample
from inspect_ai.scorer import includes
from inspect_ai.tool import bash

from evals.secondary.plan import render_xben_prompt

_FLAG_PATTERN = re.compile(r"^FLAG\{[A-Za-z0-9_-]{16,128}\}$")
_TARGET_PATTERN = re.compile(r"^http://[a-zA-Z0-9_.-]+:\d{1,5}/?$")


def _metadata(challenge_dir: Path) -> dict[str, object]:
    value = json.loads((challenge_dir / "benchmark.json").read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("description"), str):
        raise ValueError("XBEN benchmark.json has an invalid description")
    return value


@task
def xben(
    challenge_dir: str,
    compose_file: str,
    target_url: str,
    max_messages: int = 60,
    command_timeout: int = 180,
) -> Task:
    """Evaluate a single public XBEN challenge in its Docker sandbox."""

    challenge_path = Path(challenge_dir).resolve()
    compose_path = Path(compose_file).resolve()
    if not compose_path.is_file() or compose_path.name != "compose.yaml":
        raise ValueError("compose_file must be the generated evaluation wrapper")
    if not _TARGET_PATTERN.fullmatch(target_url):
        raise ValueError("target_url must name a Docker-internal HTTP service")
    expected_flag = os.environ.get("XBEN_FLAG", "")
    if not _FLAG_PATTERN.fullmatch(expected_flag):
        raise ValueError("XBEN_FLAG must be an ephemeral FLAG{...} value")

    metadata = _metadata(challenge_path)
    prompt = render_xben_prompt(
        description=str(metadata["description"]),
        target_url=target_url,
    )
    return Task(
        dataset=[
            Sample(
                id=challenge_path.name,
                input=prompt,
                target=expected_flag,
                metadata={
                    "benchmark": "xben",
                    "benchmark_id": challenge_path.name,
                    "prompt_version": "xben-blackbox-v1",
                },
            )
        ],
        solver=react(tools=[bash(timeout=command_timeout)], attempts=3),
        scorer=includes(ignore_case=False),
        sandbox=("docker", str(compose_path)),
        message_limit=max_messages,
    )
