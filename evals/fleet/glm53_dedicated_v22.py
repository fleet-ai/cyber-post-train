"""Fresh GLM TP8 v22 identity after generation-consistency qualification."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from evals.fleet import glm53_dedicated_v21 as v21

VERSION = "v22"
TITLE = "chris-cyber-evalserve-glm53-tp8-a-v22"
RUN_DIR = "/mnt/sfs/jobs/chris-cyber-evalserve-glm53-tp8-a-v22"
PRIORITY_CLASS = v21.PRIORITY_CLASS
API_PRIORITY_CLASSES = v21.API_PRIORITY_CLASSES
QUEUE = v21.QUEUE
TOPOLOGY_MODE = v21.TOPOLOGY_MODE
TOPOLOGY_LEVEL = v21.TOPOLOGY_LEVEL
WATCHDOG_QUALIFICATION = v21.WATCHDOG_QUALIFICATION
PRE_ADMISSION = v21.PRE_ADMISSION
PACKAGE_CANARY = v21.PACKAGE_CANARY
GENERATION_QUALIFICATION = {
    "path": (
        "docs/evidence/glm53-study/"
        "2026-09-06-glm53-dedicated-v22-generation-consistency-canary-v1.json"
    ),
    "file_sha256": "sha256:e5f82ccae8bebccce5165c3ae8489ec56b3ca37bb1d9f01e11cc30a7db155cea",
    "receipt_sha256": "sha256:8cfdb61b2c73633e5594cc2e01db765e5a46248e497e65a904d1b54497e15005",
    "job_uid": "af9eb5af-24e7-47fc-8adb-d12c2368ffe9",
    "pod_uid": "2fd12d50-6d71-4385-8c82-497fc476f804",
}


def spec(root: Path) -> dict[str, Any]:
    value = copy.deepcopy(v21.spec(root))
    value["schema_version"] = "fleet-glm53-dedicated-serving-v22-authorized-v1"
    value["title"] = TITLE
    value["run_dir"] = RUN_DIR
    value["request_shape_change"] = {
        "predecessor": "v21",
        "cpu_request": {"from": "64", "to": "64"},
        "memory_request": {"from": "768Gi", "to": "768Gi"},
        "reason": "fresh_identity_after_v21_generation_binding_failures",
    }
    value["generation_qualification"] = copy.deepcopy(GENERATION_QUALIFICATION)
    validate(value, root)
    return value


def _as_v21(value: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(value)
    normalized["schema_version"] = "fleet-glm53-dedicated-serving-v21-authorized-v1"
    normalized["title"] = v21.TITLE
    normalized["run_dir"] = v21.RUN_DIR
    normalized["request_shape_change"] = {
        "predecessor": "v20",
        "cpu_request": {"from": "64", "to": "64"},
        "memory_request": {"from": "768Gi", "to": "768Gi"},
        "reason": "fresh_identity_after_v20_release_race_with_full_package_closure_cpu_gate",
    }
    normalized.pop("generation_qualification", None)
    return normalized


def validate(value: dict[str, Any], root: Path) -> None:
    if value.get("schema_version") != "fleet-glm53-dedicated-serving-v22-authorized-v1":
        raise ValueError("v22 schema drifted")
    if value.get("title") != TITLE or value.get("run_dir") != RUN_DIR:
        raise ValueError("v22 identity drifted")
    if value.get("generation_qualification") != GENERATION_QUALIFICATION:
        raise ValueError("v22 generation qualification drifted")
    if value.get("request_shape_change", {}).get("reason") != (
        "fresh_identity_after_v21_generation_binding_failures"
    ):
        raise ValueError("v22 rationale drifted")
    v21.validate(_as_v21(value), root)


def payload(value: dict[str, Any], root: Path) -> dict[str, Any]:
    validate(value, root)
    result = v21.payload(_as_v21(value), root)
    result["title"] = TITLE
    result["run_dir"] = RUN_DIR
    result["env"]["GLM53_RUN_DIR"] = RUN_DIR
    return result
