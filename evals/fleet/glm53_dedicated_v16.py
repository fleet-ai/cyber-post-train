"""Fresh GLM TP8 server bound to the corrected regular-file canary package."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from evals.fleet import glm53_dedicated_v15 as v15

VERSION = "v16"
TITLE = "chris-cyber-evalserve-glm53-tp8-a-v16"
RUN_DIR = "/mnt/sfs/jobs/chris-cyber-evalserve-glm53-tp8-a-v16"
PRIORITY_CLASS = v15.PRIORITY_CLASS
API_PRIORITY_CLASSES = v15.API_PRIORITY_CLASSES
QUEUE = v15.QUEUE
TOPOLOGY_MODE = v15.TOPOLOGY_MODE
TOPOLOGY_LEVEL = v15.TOPOLOGY_LEVEL
IMAGE = v15.IMAGE
MODEL_REVISION = v15.MODEL_REVISION
MODEL_PATH = v15.MODEL_PATH
SERVER_ARGUMENTS = v15.SERVER_ARGUMENTS
SERVER_ARGUMENTS_SHA256 = v15.SERVER_ARGUMENTS_SHA256
EXPECTED_API_FIELDS = v15.EXPECTED_API_FIELDS
PRE_ADMISSION = {
    "receipt_path": "/mnt/sfs/jobs/chris-glm53-dedicated-v14-r051-preflight-v6/PREPARED.json",
    "receipt_sha256": "sha256:308a38408314f2972534b06daee5d854140a74c40c50425e964f38bda5f9339d",
    "file_sha256": "sha256:a2aa9321b0fe0a6b6104dc9a757c5bd6c81d78291d8a9f65244e02d37d439a52",
    "controller_package_sha256": "sha256:e6a7ea5a5831c737f1610269cf6deeaf3194895c14f2c15e333bdb0a62f47471",
    "held_plan_sha256": "sha256:1a6c28364ec36909aac4e08dfb8365c43864b9f7f7bb252ff30373fe10b93015",
    "selection_rank": 51,
    "all_four_rank_cells_unstarted": True,
}


def _normalized(root: Path) -> dict[str, Any]:
    value = copy.deepcopy(v15.spec(root))
    value["schema_version"] = "fleet-glm53-dedicated-serving-v16-authorized-v1"
    value["title"] = TITLE
    value["run_dir"] = RUN_DIR
    value["pre_admission"] = copy.deepcopy(PRE_ADMISSION)
    value["request_shape_change"] = {
        "predecessor": "v15",
        "cpu_request": {"from": "64", "to": "64"},
        "memory_request": {"from": "768Gi", "to": "768Gi"},
        "reason": "fresh_identity_bound_to_corrected_regular_file_controller_preflight",
    }
    return value


def _as_v15(value: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(value)
    normalized["schema_version"] = "fleet-glm53-dedicated-serving-v15-authorized-v1"
    normalized["title"] = v15.TITLE
    normalized["run_dir"] = v15.RUN_DIR
    normalized["pre_admission"] = copy.deepcopy(v15.PRE_ADMISSION)
    normalized["request_shape_change"] = {
        "predecessor": "v14",
        "cpu_request": {"from": "64", "to": "64"},
        "memory_request": {"from": "768Gi", "to": "768Gi"},
        "reason": "fresh_create_once_identity_with_exact_v14_runtime_after_pre_admission_canary_gate",
    }
    return normalized


def spec(root: Path) -> dict[str, Any]:
    value = _normalized(root)
    validate(value, root)
    return value


def validate(value: dict[str, Any], root: Path) -> None:
    if value.get("schema_version") != "fleet-glm53-dedicated-serving-v16-authorized-v1":
        raise ValueError("v16 schema drifted")
    if value.get("title") != TITLE or value.get("run_dir") != RUN_DIR:
        raise ValueError("v16 identity drifted")
    if value.get("pre_admission") != PRE_ADMISSION:
        raise ValueError("v16 pre-admission drifted")
    if value.get("request_shape_change") != {
        "predecessor": "v15",
        "cpu_request": {"from": "64", "to": "64"},
        "memory_request": {"from": "768Gi", "to": "768Gi"},
        "reason": "fresh_identity_bound_to_corrected_regular_file_controller_preflight",
    }:
        raise ValueError("v16 identity rationale drifted")
    v15.validate(_as_v15(value), root)


def payload(value: dict[str, Any], root: Path) -> dict[str, Any]:
    validate(value, root)
    result = v15.payload(_as_v15(value), root)
    result["title"] = TITLE
    result["run_dir"] = RUN_DIR
    result["env"]["GLM53_RUN_DIR"] = RUN_DIR
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
