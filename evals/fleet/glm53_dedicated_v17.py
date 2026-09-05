"""Fresh GLM TP8 server bound to exact bootstrap and duplicate preflight."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from evals.fleet import glm53_dedicated_v15 as v15

VERSION = "v17"
TITLE = "chris-cyber-evalserve-glm53-tp8-a-v17"
RUN_DIR = "/mnt/sfs/jobs/chris-cyber-evalserve-glm53-tp8-a-v17"
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
    "receipt_path": "/mnt/sfs/jobs/chris-glm53-dedicated-v14-r051-preflight-v7/PREPARED.json",
    "receipt_sha256": "sha256:5440efbe4408981ba60da33f4c3523b3d2a5486fecb89e517a638fd36b93c6ab",
    "file_sha256": "sha256:2373475d8d846d1d63e61f63280f372285a94ae534c97ed2b382f795f1fa3b15",
    "controller_package_sha256": "sha256:99727999b55f8be21decad799a994314e0068ec7cd390f1a72a3656889bfccc8",
    "held_plan_sha256": "sha256:1a6c28364ec36909aac4e08dfb8365c43864b9f7f7bb252ff30373fe10b93015",
    "selection_rank": 51,
    "all_four_rank_cells_unstarted": True,
}
CONTROLLER_BOOTSTRAP = {
    "receipt_path": "/mnt/sfs/jobs/chris-glm53-dedicated-v16-r051-bootstrap-v1/BOOTSTRAP.json",
    "receipt_sha256": "sha256:e11f23923b079bd88064ba97fae30f909777cb7f336a954b6b25d774b200cea5",
    "file_sha256": "sha256:e055c8e5a959377a6836b1d4a9ccc96219228b7e6e37799d006d2d961d06edb7",
    "controller_package_sha256": PRE_ADMISSION["controller_package_sha256"],
    "job_uid": "1362b6ba-92f2-4fb0-836d-cf9c8d43d856",
    "pod_uid": "4353a7ee-9e6a-4ba7-b7c5-e80ef88ddc19",
}


def _normalized(root: Path) -> dict[str, Any]:
    value = copy.deepcopy(v15.spec(root))
    value["schema_version"] = "fleet-glm53-dedicated-serving-v17-authorized-v1"
    value["title"] = TITLE
    value["run_dir"] = RUN_DIR
    value["pre_admission"] = copy.deepcopy(PRE_ADMISSION)
    value["controller_bootstrap"] = copy.deepcopy(CONTROLLER_BOOTSTRAP)
    value["request_shape_change"] = {
        "predecessor": "v16",
        "cpu_request": {"from": "64", "to": "64"},
        "memory_request": {"from": "768Gi", "to": "768Gi"},
        "reason": "fresh_identity_bound_to_exact_controller_bootstrap_and_preflight",
    }
    return value


def _as_v15(value: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(value)
    normalized["schema_version"] = "fleet-glm53-dedicated-serving-v15-authorized-v1"
    normalized["title"] = v15.TITLE
    normalized["run_dir"] = v15.RUN_DIR
    normalized["pre_admission"] = copy.deepcopy(v15.PRE_ADMISSION)
    normalized.pop("controller_bootstrap", None)
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
    if value.get("schema_version") != "fleet-glm53-dedicated-serving-v17-authorized-v1":
        raise ValueError("v17 schema drifted")
    if value.get("title") != TITLE or value.get("run_dir") != RUN_DIR:
        raise ValueError("v17 identity drifted")
    if value.get("pre_admission") != PRE_ADMISSION:
        raise ValueError("v17 pre-admission drifted")
    if value.get("controller_bootstrap") != CONTROLLER_BOOTSTRAP:
        raise ValueError("v17 controller bootstrap drifted")
    if value.get("request_shape_change", {}).get("reason") != (
        "fresh_identity_bound_to_exact_controller_bootstrap_and_preflight"
    ):
        raise ValueError("v17 identity rationale drifted")
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
