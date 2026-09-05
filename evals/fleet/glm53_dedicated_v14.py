"""Fresh one-pod GLM5.3 TP8 successor sized for a single free B300 node."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from evals.fleet import glm53_dedicated_v11 as v11

VERSION = "v14"
SPEC_PATH = "evals/fleet/configs/glm53-dedicated-serving-v14-authorized.json"
TITLE = "chris-cyber-evalserve-glm53-tp8-a-v14"
RUN_DIR = "/mnt/sfs/jobs/chris-cyber-evalserve-glm53-tp8-a-v14"
PRIORITY_CLASS = "fleet-infra-quiet"
API_PRIORITY_CLASSES = v11.API_PRIORITY_CLASSES
QUEUE = v11.QUEUE
TOPOLOGY_MODE = v11.TOPOLOGY_MODE
TOPOLOGY_LEVEL = v11.TOPOLOGY_LEVEL
IMAGE = v11.IMAGE
MODEL_REVISION = v11.MODEL_REVISION
MODEL_PATH = v11.MODEL_PATH
SERVER_ARGUMENTS = v11.SERVER_ARGUMENTS
SERVER_ARGUMENTS_SHA256 = v11.SERVER_ARGUMENTS_SHA256
EXPECTED_API_FIELDS = v11.EXPECTED_API_FIELDS
RESOURCES = {
    "workers": 1,
    "gpus_per_worker": 8,
    "cpu_request": "64",
    "cpu_limit": "192",
    "memory_request": "768Gi",
    "memory_limit": "2Ti",
    "privileged": True,
    "priority_class": PRIORITY_CLASS,
    "preemption_policy": "Never",
}


def load(path: Path) -> dict[str, Any]:
    return v11.load(path)


def _as_v11(value: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(value)
    normalized["schema_version"] = "fleet-glm53-dedicated-serving-v11-authorized-v1"
    normalized["title"] = v11.TITLE
    normalized["run_dir"] = v11.RUN_DIR
    normalized["resources"] = {
        "workers": 1,
        "gpus_per_worker": 8,
        "cpu_request": "96",
        "cpu_limit": "192",
        "memory_request": "1Ti",
        "memory_limit": "2Ti",
        "privileged": True,
        "priority_class": v11.PRIORITY_CLASS,
        "preemption_policy": "Never",
    }
    return normalized


def spec(root: Path) -> dict[str, Any]:
    value = load(root / SPEC_PATH)
    validate(value, root)
    return value


def validate(value: dict[str, Any], root: Path) -> None:
    if value.get("schema_version") != "fleet-glm53-dedicated-serving-v14-authorized-v1":
        raise ValueError("v14 schema drifted")
    if value.get("title") != TITLE or value.get("run_dir") != RUN_DIR:
        raise ValueError("v14 create-once identity drifted")
    if value.get("resources") != RESOURCES:
        raise ValueError("v14 single-node resource shape drifted")
    if value.get("request_shape_change") != {
        "predecessor": "v13",
        "cpu_request": {"from": "96", "to": "64"},
        "memory_request": {"from": "1Ti", "to": "768Gi"},
        "reason": "fit_one_schedulable_eight_gpu_b300_node_without_changing_runtime_limits",
    }:
        raise ValueError("v14 request-shape rationale drifted")
    v11.validate(_as_v11(value), root)


def payload(value: dict[str, Any], root: Path) -> dict[str, Any]:
    validate(value, root)
    result = v11.payload(_as_v11(value), root)
    result["title"] = TITLE
    result["run_dir"] = RUN_DIR
    result["env"]["GLM53_RUN_DIR"] = RUN_DIR
    result["resources"] = {
        "cpu_request": RESOURCES["cpu_request"],
        "cpu_limit": RESOURCES["cpu_limit"],
        "memory_request": RESOURCES["memory_request"],
        "memory_limit": RESOURCES["memory_limit"],
    }
    if set(result) != EXPECTED_API_FIELDS:
        raise AssertionError("v14 Jobs API payload shape drifted")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("validate", "payload"))
    args = parser.parse_args()
    root = Path.cwd()
    value = spec(root)
    if args.command == "payload":
        print(json.dumps(payload(value, root), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
