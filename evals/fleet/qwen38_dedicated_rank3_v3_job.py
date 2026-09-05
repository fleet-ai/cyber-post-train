"""Render held create-once evaluator Jobs for dedicated Qwen rank3."""

# ruff: noqa: E501 -- bootstrap paths are immutable experiment bindings.

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from evals.fleet import qwen38_dedicated_rank2_v3_job as rank2_job
from evals.fleet import self_hosted

SOURCE = rank2_job.SOURCE
SOURCE_SHA256 = rank2_job.SOURCE_SHA256
SERVING_BLOCK = "dedicated-qwen-tp1-b-v2"
BOOTSTRAP = rank2_job.BOOTSTRAP.replace(
    'install -m 0644 /bootstrap/lane.py "$root/evals/fleet/qwen38_dedicated_rank2_v3.py"',
    'install -m 0644 /bootstrap/lane.py "$root/evals/fleet/qwen38_dedicated_rank3_v3.py"',
).replace(
    'install -m 0644 /bootstrap/parity.json "$root/docs/evidence/qwen38-study/2026-09-05-qwen38-dedicated-v3-actual-opencode-parity.json"',
    'install -m 0644 /bootstrap/parity.json "$root/docs/evidence/qwen38-study/2026-09-05-qwen38-dedicated-tp1-b-v2-actual-opencode-parity.json"',
)


def rank3_run_script(rank2_script: str) -> str:
    old = "-m evals.fleet.qwen38_dedicated_rank2_v3"
    if rank2_script.count(old) != 1:
        raise ValueError("rank2 evaluator run script module binding drifted")
    attempt_domain = 'case "$QWEN_DEDICATED_ATTEMPT" in 2|3|4)'
    if rank2_script.count(attempt_domain) != 1:
        raise ValueError("rank2 evaluator attempt-domain binding drifted")
    return rank2_script.replace(
        attempt_domain, 'case "$QWEN_DEDICATED_ATTEMPT" in 1|2|3|4)'
    ).replace(old, "-m evals.fleet.qwen38_dedicated_rank3_v3")


def render(root: Path, attempt: int) -> dict[str, Any]:
    if attempt not in (1, 2, 3, 4):
        raise ValueError("dedicated Qwen rank3 attempt must be 1 through 4")
    name = f"chris-cyber-q38-opencode11827-ded-tp1-r003-a{attempt}-g21-v2"
    experiment = f"q38-ded-tp1-r003-a{attempt}-g21-v2"
    source = root / SOURCE
    if self_hosted.sha256(source.read_bytes()) != SOURCE_SHA256:
        raise ValueError("source evaluator Job template drifted")
    value = copy.deepcopy(yaml.safe_load(source.read_text()))
    value["metadata"]["name"] = name
    value["metadata"]["labels"]["cyber-post-train.fleet.ai/experiment"] = experiment
    value["metadata"]["annotations"]["cyber-post-train.fleet.ai/serving-block"] = SERVING_BLOCK
    template = value["spec"]["template"]
    template["metadata"]["labels"]["cyber-post-train.fleet.ai/experiment"] = experiment
    template["spec"]["volumes"][0]["configMap"]["name"] = name
    container = template["spec"]["containers"][0]
    container["args"] = [BOOTSTRAP]
    container.setdefault("env", []).extend(
        [
            {"name": "QWEN_DEDICATED_ATTEMPT", "value": str(attempt)},
            {
                "name": "QWEN_RANK3_RELEASE_PATH",
                "value": "/bootstrap/rank3-release.json",
            },
        ]
    )
    return value
