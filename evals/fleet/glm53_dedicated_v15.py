"""Fresh GLM5.3 TP8 server gated on an executable scored-canary package."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from evals.fleet import glm53_dedicated_v14 as v14

VERSION = "v15"
SPEC_PATH = "evals/fleet/configs/glm53-dedicated-serving-v15-authorized.json"
TITLE = "chris-cyber-evalserve-glm53-tp8-a-v15"
RUN_DIR = "/mnt/sfs/jobs/chris-cyber-evalserve-glm53-tp8-a-v15"
PRIORITY_CLASS = v14.PRIORITY_CLASS
API_PRIORITY_CLASSES = v14.API_PRIORITY_CLASSES
QUEUE = v14.QUEUE
TOPOLOGY_MODE = v14.TOPOLOGY_MODE
TOPOLOGY_LEVEL = v14.TOPOLOGY_LEVEL
IMAGE = v14.IMAGE
MODEL_REVISION = v14.MODEL_REVISION
MODEL_PATH = v14.MODEL_PATH
SERVER_ARGUMENTS = v14.SERVER_ARGUMENTS
SERVER_ARGUMENTS_SHA256 = v14.SERVER_ARGUMENTS_SHA256
EXPECTED_API_FIELDS = v14.EXPECTED_API_FIELDS
PRE_ADMISSION = {
    "receipt_path": "/mnt/sfs/jobs/chris-glm53-dedicated-v14-r051-preflight-v5/PREPARED.json",
    "receipt_sha256": "sha256:92a68098671ef4dff9fb481309258b733149b9283d36528c0b35cc2d3ab05011",
    "file_sha256": "sha256:be0c63e1af79e8154ae86ecbc9bff019c90c2da64d668313383ec77fa535372c",
    "controller_package_sha256": "sha256:e26bb6dc20e87b56fb9ed8d746e631d8f8c969bfe2717c7ab4d100975afdda3c",
    "held_plan_sha256": "sha256:1a6c28364ec36909aac4e08dfb8365c43864b9f7f7bb252ff30373fe10b93015",
    "selection_rank": 51,
    "all_four_rank_cells_unstarted": True,
}


def load(path: Path) -> dict[str, Any]:
    return v14.load(path)


def _as_v14(value: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(value)
    normalized["schema_version"] = "fleet-glm53-dedicated-serving-v14-authorized-v1"
    normalized["scope"] = "one_non_scored_dedicated_server_only"
    normalized["title"] = v14.TITLE
    normalized["run_dir"] = v14.RUN_DIR
    normalized["request_shape_change"] = {
        "predecessor": "v13",
        "cpu_request": {"from": "96", "to": "64"},
        "memory_request": {"from": "1Ti", "to": "768Gi"},
        "reason": "fit_one_schedulable_eight_gpu_b300_node_without_changing_runtime_limits",
    }
    normalized["evaluation"] = {
        "scored_tasks_allowed": False,
        "parity_required_before_scored_canary": True,
        "accepted_canary_required_before_bulk": True,
    }
    normalized.pop("pre_admission", None)
    return normalized


def spec(root: Path) -> dict[str, Any]:
    value = load(root / SPEC_PATH)
    validate(value, root)
    return value


def validate(value: dict[str, Any], root: Path) -> None:
    if value.get("schema_version") != "fleet-glm53-dedicated-serving-v15-authorized-v1":
        raise ValueError("v15 schema drifted")
    if value.get("status") != "AUTHORIZED" or value.get("launch_authorized") is not True:
        raise ValueError("v15 launch is not explicitly authorized")
    if value.get("scope") != "one_non_scored_parity_then_one_scored_canary":
        raise ValueError("v15 scope drifted")
    if value.get("title") != TITLE or value.get("run_dir") != RUN_DIR:
        raise ValueError("v15 create-once identity drifted")
    if value.get("pre_admission") != PRE_ADMISSION:
        raise ValueError("v15 canary pre-admission gate drifted")
    if value.get("request_shape_change") != {
        "predecessor": "v14",
        "cpu_request": {"from": "64", "to": "64"},
        "memory_request": {"from": "768Gi", "to": "768Gi"},
        "reason": "fresh_create_once_identity_with_exact_v14_runtime_after_pre_admission_canary_gate",
    }:
        raise ValueError("v15 request identity rationale drifted")
    if value.get("evaluation") != {
        "scored_tasks_allowed": "only_rank51_attempt1_after_fresh_uid_bound_parity_and_release",
        "parity_required_before_scored_canary": True,
        "accepted_canary_required_before_bulk": True,
        "fresh_duplicate_release_required_immediately_before_canary": True,
    }:
        raise ValueError("v15 evaluation gate drifted")
    v14.validate(_as_v14(value), root)


def payload(value: dict[str, Any], root: Path) -> dict[str, Any]:
    validate(value, root)
    result = v14.payload(_as_v14(value), root)
    result["title"] = TITLE
    result["run_dir"] = RUN_DIR
    result["env"]["GLM53_RUN_DIR"] = RUN_DIR
    if set(result) != EXPECTED_API_FIELDS:
        raise AssertionError("v15 Jobs API payload shape drifted")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("validate", "payload"))
    args = parser.parse_args()
    value = spec(Path.cwd())
    if args.command == "payload":
        print(json.dumps(payload(value, Path.cwd()), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
