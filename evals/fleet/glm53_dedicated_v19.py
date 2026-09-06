"""Fresh GLM TP8 server using the qualified local-metrics idle watchdog."""

from __future__ import annotations

import argparse
import copy
import json
import shlex
from pathlib import Path
from typing import Any

from evals.fleet import glm53_dedicated_v17 as v18
from evals.fleet import self_hosted

VERSION = "v19"
TITLE = "chris-cyber-evalserve-glm53-tp8-a-v19"
RUN_DIR = "/mnt/sfs/jobs/chris-cyber-evalserve-glm53-tp8-a-v19"
LIFECYCLE_PATH = "evals/fleet/scripts/glm53_dedicated_metric_lifecycle_v1.sh"
PRIORITY_CLASS = v18.PRIORITY_CLASS
API_PRIORITY_CLASSES = v18.API_PRIORITY_CLASSES
QUEUE = v18.QUEUE
TOPOLOGY_MODE = v18.TOPOLOGY_MODE
TOPOLOGY_LEVEL = v18.TOPOLOGY_LEVEL
IMAGE = v18.IMAGE
MODEL_REVISION = v18.MODEL_REVISION
MODEL_PATH = v18.MODEL_PATH
SERVER_ARGUMENTS = v18.SERVER_ARGUMENTS
SERVER_ARGUMENTS_SHA256 = v18.SERVER_ARGUMENTS_SHA256
EXPECTED_API_FIELDS = v18.EXPECTED_API_FIELDS
WATCHDOG_QUALIFICATION = {
    "receipt_path": (
        "/mnt/sfs/jobs/chris-glm53-dedicated-metric-watchdog-canary-v3/QUALIFIED.json"
    ),
    "receipt_sha256": (
        "sha256:375eccee69b52fe1d149339ccb4817105e47112bca6b79b2624f42f0a6fc452e"
    ),
    "file_sha256": (
        "sha256:b40a7eab8a67043be4e1de6286938d9964d73dac8c3202062735c99f77b4d768"
    ),
    "lifecycle_file_sha256": (
        "sha256:791a92ac5aa5dbacfe402f062fb49084204029f01110c976881c9293ebc727c8"
    ),
    "job_uid": "1187ac65-8f18-4c07-85f1-ddeb6104c9e1",
    "pod_uid": "3b1cb2c9-1332-4adf-8f02-cb391833dcb3",
}


def _normalized(root: Path) -> dict[str, Any]:
    value = copy.deepcopy(v18.spec(root))
    value["schema_version"] = "fleet-glm53-dedicated-serving-v19-authorized-v1"
    value["title"] = TITLE
    value["run_dir"] = RUN_DIR
    value["lifecycle"] = {
        "script": LIFECYCLE_PATH,
        "post_ready_idle_seconds": 600,
        "startup_is_outside_idle_budget": True,
        "release_when_idle": True,
        "productive_signal": "local_sglang_inference_metric_counter_change",
        "health_or_controller_liveness_refreshes_lease": False,
    }
    value["watchdog_qualification"] = copy.deepcopy(WATCHDOG_QUALIFICATION)
    value["request_shape_change"] = {
        "predecessor": "v18",
        "cpu_request": {"from": "64", "to": "64"},
        "memory_request": {"from": "768Gi", "to": "768Gi"},
        "reason": "fresh_identity_after_v18_false_idle_with_qualified_metric_watchdog",
    }
    value["evaluation"] = {
        "scored_tasks_allowed": (
            "only_after_fresh_uid_bound_parity_release_and_server_bound_preclaim"
        ),
        "parity_required_before_scored_canary": True,
        "accepted_canary_required_before_bulk": True,
        "fresh_duplicate_release_required_immediately_before_canary_create": True,
    }
    return value


def _as_v18(value: dict[str, Any], root: Path) -> dict[str, Any]:
    normalized = copy.deepcopy(value)
    normalized["schema_version"] = "fleet-glm53-dedicated-serving-v18-authorized-v1"
    normalized["title"] = v18.TITLE
    normalized["run_dir"] = v18.RUN_DIR
    normalized["lifecycle"] = copy.deepcopy(v18.spec(root)["lifecycle"])
    normalized.pop("watchdog_qualification", None)
    normalized["request_shape_change"] = {
        "predecessor": "v17",
        "cpu_request": {"from": "64", "to": "64"},
        "memory_request": {"from": "768Gi", "to": "768Gi"},
        "reason": "fresh_identity_bound_to_adapter_bootstrap_runtime_gate_and_preflight",
    }
    normalized["evaluation"] = copy.deepcopy(v18.spec(root)["evaluation"])
    return normalized


def spec(root: Path) -> dict[str, Any]:
    value = _normalized(root)
    validate(value, root)
    return value


def validate(value: dict[str, Any], root: Path) -> None:
    if value.get("schema_version") != "fleet-glm53-dedicated-serving-v19-authorized-v1":
        raise ValueError("v19 schema drifted")
    if value.get("title") != TITLE or value.get("run_dir") != RUN_DIR:
        raise ValueError("v19 identity drifted")
    if value.get("watchdog_qualification") != WATCHDOG_QUALIFICATION:
        raise ValueError("v19 watchdog qualification drifted")
    expected_lifecycle = {
        "script": LIFECYCLE_PATH,
        "post_ready_idle_seconds": 600,
        "startup_is_outside_idle_budget": True,
        "release_when_idle": True,
        "productive_signal": "local_sglang_inference_metric_counter_change",
        "health_or_controller_liveness_refreshes_lease": False,
    }
    if value.get("lifecycle") != expected_lifecycle:
        raise ValueError("v19 lifecycle drifted")
    lifecycle = root / LIFECYCLE_PATH
    if lifecycle.is_symlink() or not lifecycle.is_file():
        raise ValueError("v19 lifecycle is absent or unsafe")
    if self_hosted.sha256(lifecycle.read_bytes()) != WATCHDOG_QUALIFICATION[
        "lifecycle_file_sha256"
    ]:
        raise ValueError("v19 lifecycle differs from qualified bytes")
    if value.get("request_shape_change", {}).get("reason") != (
        "fresh_identity_after_v18_false_idle_with_qualified_metric_watchdog"
    ):
        raise ValueError("v19 rationale drifted")
    v18.validate(_as_v18(value, root), root)


def payload(value: dict[str, Any], root: Path) -> dict[str, Any]:
    validate(value, root)
    result = v18.payload(_as_v18(value, root), root)
    lifecycle = (root / LIFECYCLE_PATH).read_text()
    result["command"] = (
        "bash -lc " + shlex.quote(lifecycle) + " -- " + shlex.join(SERVER_ARGUMENTS)
    )
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
