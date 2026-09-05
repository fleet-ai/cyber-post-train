"""Fresh TP1-d server for the rank97 whole-task dedicated block."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from evals.fleet import qwen38_dedicated_v6 as v6

SPEC_PATH = "evals/fleet/configs/qwen38-dedicated-serving-v7-authorized.json"
TITLE = "chris-cyber-evalserve-q38-tp1-d-v1"
RUN_DIR = "/mnt/sfs/jobs/chris-cyber-evalserve-q38-tp1-d-v1"
PRIORITY_CLASS = v6.PRIORITY_CLASS
API_PRIORITY_CLASSES = v6.API_PRIORITY_CLASSES
MAX_PROJECT_GPU_NODES = v6.MAX_PROJECT_GPU_NODES
MAX_PROJECT_GPUS = v6.MAX_PROJECT_GPUS


def _as_v6(value: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(value)
    result["schema_version"] = "fleet-qwen38-dedicated-serving-v6-authorized-spec-v1"
    result["title"] = v6.TITLE
    result["run_dir"] = v6.RUN_DIR
    result["scope"] = "one_non_scored_actual_harness_parity_server_then_disjoint_whole_task"
    return result


def spec(root: Path) -> dict[str, Any]:
    value = json.loads((root / SPEC_PATH).read_text())
    validate(value, root)
    return value


def validate(value: dict[str, Any], root: Path) -> None:
    if value.get("schema_version") != "fleet-qwen38-dedicated-serving-v7-authorized-spec-v1":
        raise ValueError("Qwen v7 schema drifted")
    if value.get("title") != TITLE or value.get("run_dir") != RUN_DIR:
        raise ValueError("Qwen v7 identity drifted")
    if value.get("scope") != "one_non_scored_actual_harness_parity_server_then_rank97_whole_task":
        raise ValueError("Qwen v7 scope drifted")
    v6.validate(_as_v6(value), root)


def payload(value: dict[str, Any], root: Path) -> dict[str, Any]:
    validate(value, root)
    result = v6.payload(_as_v6(value), root)
    result["title"] = TITLE
    result["run_dir"] = RUN_DIR
    result["env"]["QWEN38_RUN_DIR"] = RUN_DIR
    return result
