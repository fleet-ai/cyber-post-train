"""Fresh GLM TP8 v21 identity after the v20 package-closure qualification."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from evals.fleet import glm53_dedicated_v20 as v20

VERSION = "v21"
TITLE = "chris-cyber-evalserve-glm53-tp8-a-v21"
RUN_DIR = "/mnt/sfs/jobs/chris-cyber-evalserve-glm53-tp8-a-v21"
PRIORITY_CLASS = v20.PRIORITY_CLASS
API_PRIORITY_CLASSES = v20.API_PRIORITY_CLASSES
QUEUE = v20.QUEUE
TOPOLOGY_MODE = v20.TOPOLOGY_MODE
TOPOLOGY_LEVEL = v20.TOPOLOGY_LEVEL
WATCHDOG_QUALIFICATION = v20.WATCHDOG_QUALIFICATION
PRE_ADMISSION = v20.PRE_ADMISSION
PACKAGE_CANARY = {
    "receipt_path": "/mnt/sfs/jobs/chris-glm53-dedicated-v20-r051-a2-bootstrap-v3/BOOTSTRAP.json",
    "file_sha256": "sha256:a120b0f2ea190fb83a967814a0236f537805ce41bb0710f64c6688630cc4a59b",
    "receipt_sha256": "sha256:a7732199ccfa1ab3773405fb85403c1735dc3e7ff0c4b3464ded5090f1eccd92",
    "controller_package_sha256": "sha256:5de8f42c597eb824e54b7e89cd8e44ce92b10ce6113bfbc3597a8ee3522adc8b",
    "runtime_plan_sha256": "sha256:692b2a026cc9067c4387596aacfab6c2b2248e42c7d2c8196f463e9080c6b4e3",
    "job_uid": "e0056f59-1511-4bfb-82d8-6a6cdf98d4f9",
    "pod_uid": "c4530d33-f017-425e-9dc6-f943d65dfb29",
}


def spec(root: Path) -> dict[str, Any]:
    value = copy.deepcopy(v20.spec(root))
    value["schema_version"] = "fleet-glm53-dedicated-serving-v21-authorized-v1"
    value["title"] = TITLE
    value["run_dir"] = RUN_DIR
    value["request_shape_change"] = {
        "predecessor": "v20",
        "cpu_request": {"from": "64", "to": "64"},
        "memory_request": {"from": "768Gi", "to": "768Gi"},
        "reason": "fresh_identity_after_v20_release_race_with_full_package_closure_cpu_gate",
    }
    value["package_canary"] = copy.deepcopy(PACKAGE_CANARY)
    validate(value, root)
    return value


def _as_v20(value: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(value)
    normalized["schema_version"] = "fleet-glm53-dedicated-serving-v20-authorized-v1"
    normalized["title"] = v20.TITLE
    normalized["run_dir"] = v20.RUN_DIR
    normalized["request_shape_change"] = {
        "predecessor": "v19",
        "cpu_request": {"from": "64", "to": "64"},
        "memory_request": {"from": "768Gi", "to": "768Gi"},
        "reason": "fresh_identity_after_v19_preclaim_packaging_fix",
    }
    normalized.pop("package_canary", None)
    return normalized


def validate(value: dict[str, Any], root: Path) -> None:
    if value.get("schema_version") != "fleet-glm53-dedicated-serving-v21-authorized-v1":
        raise ValueError("v21 schema drifted")
    if value.get("title") != TITLE or value.get("run_dir") != RUN_DIR:
        raise ValueError("v21 identity drifted")
    if value.get("package_canary") != PACKAGE_CANARY:
        raise ValueError("v21 package canary drifted")
    if value.get("request_shape_change", {}).get("reason") != (
        "fresh_identity_after_v20_release_race_with_full_package_closure_cpu_gate"
    ):
        raise ValueError("v21 rationale drifted")
    v20.validate(_as_v20(value), root)


def payload(value: dict[str, Any], root: Path) -> dict[str, Any]:
    validate(value, root)
    result = v20.payload(_as_v20(value), root)
    result["title"] = TITLE
    result["run_dir"] = RUN_DIR
    result["env"]["GLM53_RUN_DIR"] = RUN_DIR
    return result

