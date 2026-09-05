"""Create-once authority for the fresh dedicated Qwen3.8 TP1 successor."""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any

from evals.fleet import glm53_dedicated_v7 as api
from evals.fleet import qwen38_dedicated_v1 as v1
from evals.fleet import self_hosted

SPEC_PATH = "evals/fleet/configs/qwen38-dedicated-serving-v2-authorized.json"
VALIDATED_RECEIPT = (
    "docs/evidence/qwen38-study/2026-09-05-dedicated-qwen-rank2-attempt1-accepted-validated-v1.json"
)
VALIDATED_RECEIPT_SHA = "sha256:108903b167e40d8628776a6bed35ef7fd871b687dd8c81413fe66ca0f2e43212"
TITLE = "chris-cyber-evalserve-q38-tp1-a-v2"
RUN_DIR = "/mnt/sfs/jobs/chris-cyber-evalserve-q38-tp1-a-v2"
LIFECYCLE_PATH = v1.LIFECYCLE_PATH
IMAGE = v1.IMAGE
MODEL_REVISION = v1.MODEL_REVISION
MODEL_PATH = v1.MODEL_PATH
SERVER_ARGUMENTS = v1.SERVER_ARGUMENTS
SERVER_ARGUMENTS_SHA256 = v1.SERVER_ARGUMENTS_SHA256


def load(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"unsafe or absent JSON: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("spec must be an object")
    return value


def spec(root: Path) -> dict[str, Any]:
    value = load(root / SPEC_PATH)
    validate(value, root)
    return value


def validate(value: dict[str, Any], root: Path) -> None:
    if value.get("schema_version") != "fleet-qwen38-dedicated-serving-v2-authorized-spec-v1":
        raise ValueError("Qwen v2 schema drifted")
    if value.get("status") != "AUTHORIZED" or value.get("launch_authorized") is not True:
        raise ValueError("Qwen v2 is not authorized")
    if value.get("title") != TITLE or value.get("run_dir") != RUN_DIR:
        raise ValueError("Qwen v2 identity drifted")
    if value.get("scope") != "one_non_scored_parity_server_then_rank2_attempts2_through4":
        raise ValueError("Qwen v2 scope drifted")
    if value.get("model") != {
        "repository": "Qwen/Qwen3.8-27B",
        "revision": MODEL_REVISION,
        "served_id": "qwen3.8-27b",
        "sfs_path": MODEL_PATH,
        "context_length": 262144,
        "staging_receipt_sha256": v1.STAGING_RECEIPT_SHA,
    }:
        raise ValueError("Qwen v2 model identity drifted")
    if value.get("runtime") != {
        "image": IMAGE,
        "opencode_version": "1.18.27",
        "reasoning_parser": "qwen3",
        "tool_call_parser": "qwen3_coder",
    }:
        raise ValueError("Qwen v2 runtime drifted")
    if value.get("resources") != {
        "workers": 1,
        "gpus_per_worker": 1,
        "cpu_request": "32",
        "cpu_limit": "96",
        "memory_request": "256Gi",
        "memory_limit": "768Gi",
        "privileged": False,
        "priority_class": "fleet-infra-quiet",
        "preemption_policy": "Never",
    }:
        raise ValueError("Qwen v2 resource policy drifted")
    if value.get("evaluation") != {
        "fresh_parity_required_before_scoring": True,
        "scored_tasks_allowed_before_parity": False,
        "rank2_attempts_allowed_after_parity": [2, 3, 4],
        "same_task_max_inflight": 1,
    }:
        raise ValueError("Qwen v2 evaluation policy drifted")
    receipt = load(root / VALIDATED_RECEIPT)
    if (
        receipt.get("receipt_sha256") != VALIDATED_RECEIPT_SHA
        or self_hosted.digest_without(receipt, "receipt_sha256") != VALIDATED_RECEIPT_SHA
    ):
        raise ValueError("prior validated cell receipt drifted")
    if value.get("prior_validated_cell") != {
        "selection_rank": 2,
        "attempt": 1,
        "accepted_validated_receipt_sha256": VALIDATED_RECEIPT_SHA,
    }:
        raise ValueError("prior validated cell binding drifted")
    if value.get("server_arguments_sha256") != SERVER_ARGUMENTS_SHA256:
        raise ValueError("Qwen v2 server arguments drifted")
    if not (root / LIFECYCLE_PATH).is_file():
        raise ValueError("Qwen lifecycle is absent")


def payload(value: dict[str, Any], root: Path) -> dict[str, Any]:
    validate(value, root)
    lifecycle = (root / LIFECYCLE_PATH).read_text()
    result = {
        "image": IMAGE,
        "command": "bash -lc " + shlex.quote(lifecycle) + " -- " + shlex.join(SERVER_ARGUMENTS),
        "workers": 1,
        "gpus_per_worker": 1,
        "env": {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "QWEN38_RUN_DIR": RUN_DIR,
        },
        "secrets": [],
        "resources": {
            "cpu_request": "32",
            "cpu_limit": "96",
            "memory_request": "256Gi",
            "memory_limit": "768Gi",
        },
        "priority_class": "fleet-infra-quiet",
        "privileged": False,
        "run_dir": RUN_DIR,
        "title": TITLE,
    }
    if set(result) != api.EXPECTED_API_FIELDS:
        raise AssertionError("Qwen v2 Jobs API payload shape drifted")
    return result
