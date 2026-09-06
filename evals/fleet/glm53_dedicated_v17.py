"""Fresh GLM TP8 server bound to exact bootstrap and duplicate preflight."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from evals.fleet import glm53_dedicated_v15 as v15

VERSION = "v18"
TITLE = "chris-cyber-evalserve-glm53-tp8-a-v18"
RUN_DIR = "/mnt/sfs/jobs/chris-cyber-evalserve-glm53-tp8-a-v18"
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
    "receipt_path": "/mnt/sfs/jobs/chris-glm53-dedicated-v14-r051-preflight-v8/PREPARED.json",
    "receipt_sha256": "sha256:d02c7dc3306df815181e5e5d88404daf263635a4612e6aacb78938f9a4c1c5f9",
    "file_sha256": "sha256:be32c1b610a73f6d9552d8a033a92212933b89b61baace14b2361ea0e4180e46",
    "controller_package_sha256": "sha256:86468d5893a7f8be20315a82b44532fcfa9d53fc5fff937f431197aca9a0b249",
    "held_plan_sha256": "sha256:1a6c28364ec36909aac4e08dfb8365c43864b9f7f7bb252ff30373fe10b93015",
    "selection_rank": 51,
    "all_four_rank_cells_unstarted": True,
}
CONTROLLER_BOOTSTRAP = {
    "receipt_path": "/mnt/sfs/jobs/chris-glm53-dedicated-v17-r051-bootstrap-v2/BOOTSTRAP.json",
    "receipt_sha256": "sha256:6ad735fb385eb5d55eae03bbe60377d5b040a9ee08c70d26861da2f3e3661333",
    "file_sha256": "sha256:5411984e4cfffef52dfe0865602d83f7ebbec116c8c890f5d1d8d26fec9fe2ae",
    "controller_package_sha256": PRE_ADMISSION["controller_package_sha256"],
    "job_uid": "a8d79ce1-e1da-4332-bfb9-a42992795123",
    "pod_uid": "a17ab386-d905-494f-be88-385dae4272f0",
}
RUNTIME_GATE = {
    "receipt_path": "/mnt/sfs/jobs/chris-glm53-dedicated-v17-r051-runtime-gate-v1/GATE.json",
    "receipt_sha256": "sha256:9ef51367f3522178d9d34dba0527ad057e19695cc8c4fb74be308dc0819c7930",
    "file_sha256": "sha256:edd1ca1a0a420075b8c248ee0a6c293dfe439c80f6b31faa52f99a05f845b963",
    "job_uid": "46d491fb-10db-4bc6-8d3e-7f05067436fd",
    "pod_uid": "9829bf08-a20b-4e59-ac0c-bb18bdd5593c",
    "plan_sha256": "sha256:b5efd30a13224f8b408974ba3ac5f5e7cd9cee1827c30d8c4c9d223f5576bdca",
}


def _normalized(root: Path) -> dict[str, Any]:
    value = copy.deepcopy(v15.spec(root))
    value["schema_version"] = "fleet-glm53-dedicated-serving-v18-authorized-v1"
    value["title"] = TITLE
    value["run_dir"] = RUN_DIR
    value["pre_admission"] = copy.deepcopy(PRE_ADMISSION)
    value["controller_bootstrap"] = copy.deepcopy(CONTROLLER_BOOTSTRAP)
    value["runtime_gate"] = copy.deepcopy(RUNTIME_GATE)
    value["request_shape_change"] = {
        "predecessor": "v17",
        "cpu_request": {"from": "64", "to": "64"},
        "memory_request": {"from": "768Gi", "to": "768Gi"},
        "reason": "fresh_identity_bound_to_adapter_bootstrap_runtime_gate_and_preflight",
    }
    return value


def _as_v15(value: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(value)
    normalized["schema_version"] = "fleet-glm53-dedicated-serving-v15-authorized-v1"
    normalized["title"] = v15.TITLE
    normalized["run_dir"] = v15.RUN_DIR
    normalized["pre_admission"] = copy.deepcopy(v15.PRE_ADMISSION)
    normalized.pop("controller_bootstrap", None)
    normalized.pop("runtime_gate", None)
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
    if value.get("schema_version") != "fleet-glm53-dedicated-serving-v18-authorized-v1":
        raise ValueError("v18 schema drifted")
    if value.get("title") != TITLE or value.get("run_dir") != RUN_DIR:
        raise ValueError("v18 identity drifted")
    if value.get("pre_admission") != PRE_ADMISSION:
        raise ValueError("v18 pre-admission drifted")
    if value.get("controller_bootstrap") != CONTROLLER_BOOTSTRAP:
        raise ValueError("v18 controller bootstrap drifted")
    if value.get("runtime_gate") != RUNTIME_GATE:
        raise ValueError("v18 runtime gate drifted")
    if value.get("request_shape_change", {}).get("reason") != (
        "fresh_identity_bound_to_adapter_bootstrap_runtime_gate_and_preflight"
    ):
        raise ValueError("v18 identity rationale drifted")
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
