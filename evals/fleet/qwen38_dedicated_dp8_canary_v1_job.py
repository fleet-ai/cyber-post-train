"""Render the create-once CPU evaluator Job for the Qwen DP8 canary."""

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
NAME = "chris-cyber-q38-opencode11827-ded-dp8-r004-a2-g22-v1"
EXPERIMENT = "q38-ded-dp8-r004-a2-g22-v1"
SERVING_BLOCK = "dedicated-qwen-dp8-a-v1"

BOOTSTRAP = rank2_job.BOOTSTRAP.replace(
    'install -m 0644 /bootstrap/lane.py "$root/evals/fleet/qwen38_dedicated_rank2_v3.py"',
    'install -m 0644 /bootstrap/lane.py "$root/evals/fleet/qwen38_dedicated_dp8_canary_v1.py"',
).replace(
    'install -m 0644 /bootstrap/parity.json "$root/docs/evidence/qwen38-study/2026-09-05-qwen38-dedicated-v3-actual-opencode-parity.json"',
    'install -m 0644 /bootstrap/parity.json "$root/docs/evidence/qwen38-study/2026-09-05-qwen38-dedicated-dp8-v1-actual-opencode-parity.json"',
)


def render(root: Path) -> dict[str, Any]:
    source = root / SOURCE
    if self_hosted.sha256(source.read_bytes()) != SOURCE_SHA256:
        raise ValueError("source evaluator Job template drifted")
    value = copy.deepcopy(yaml.safe_load(source.read_text()))
    value["metadata"]["name"] = NAME
    value["metadata"]["labels"]["cyber-post-train.fleet.ai/experiment"] = EXPERIMENT
    value["metadata"]["annotations"]["cyber-post-train.fleet.ai/serving-block"] = SERVING_BLOCK
    template = value["spec"]["template"]
    template["metadata"]["labels"]["cyber-post-train.fleet.ai/experiment"] = EXPERIMENT
    template["spec"]["volumes"][0]["configMap"]["name"] = NAME
    container = template["spec"]["containers"][0]
    container["args"] = [BOOTSTRAP]
    container.setdefault("env", []).append(
        {"name": "QWEN_DP8_RELEASE_PATH", "value": "/bootstrap/dp8-release.json"}
    )
    return value
