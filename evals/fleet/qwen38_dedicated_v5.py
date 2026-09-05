"""Fresh Qwen3.8 second TP1 server successor using actual OpenCode parity."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from evals.fleet import opencode_actual_harness_parity_v1 as actual_parity
from evals.fleet import qwen38_dedicated_v2 as v2
from evals.fleet import self_hosted

SPEC_PATH = "evals/fleet/configs/qwen38-dedicated-serving-v5-authorized.json"
TITLE = "chris-cyber-evalserve-q38-tp1-b-v2"
RUN_DIR = "/mnt/sfs/jobs/chris-cyber-evalserve-q38-tp1-b-v2"
PRIORITY_CLASS = "fleet-infra-quiet"
API_PRIORITY_CLASSES = {"fleet-train-high", "fleet-infra-quiet"}
IMAGE = v2.IMAGE
MODEL_REVISION = v2.MODEL_REVISION
MODEL_PATH = v2.MODEL_PATH
SERVER_ARGUMENTS = v2.SERVER_ARGUMENTS
SERVER_ARGUMENTS_SHA256 = v2.SERVER_ARGUMENTS_SHA256
MAX_PROJECT_GPU_NODES = 2
MAX_PROJECT_GPUS = 16
PRIOR_VALIDATED_RECEIPT = (
    "docs/evidence/qwen38-study/2026-09-05-qwen38-dedicated-r002-a3-accepted-validated-v2.json"
)
PRIOR_VALIDATED_RECEIPT_SHA256 = (
    "sha256:0b0c81a825103ff66699cdfce8a166b4f45371f556fa0a46ea5e67e09509b769"
)


def load(path: Path) -> dict[str, Any]:
    return v2.load(path)


def _as_v2(value: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(value)
    normalized["schema_version"] = "fleet-qwen38-dedicated-serving-v2-authorized-spec-v1"
    normalized["title"] = v2.TITLE
    normalized["run_dir"] = v2.RUN_DIR
    normalized["scope"] = "one_non_scored_parity_server_then_rank2_attempts2_through4"
    normalized["evaluation"] = {
        "fresh_parity_required_before_scoring": True,
        "scored_tasks_allowed_before_parity": False,
        "rank2_attempts_allowed_after_parity": [2, 3, 4],
        "same_task_max_inflight": 1,
    }
    normalized["create_once"] = {
        "additional_replica_allowed": False,
        "api_route": "POST /v1/runs",
        "duplicate_checks_required": True,
    }
    normalized["prior_validated_cell"] = {
        "selection_rank": 2,
        "attempt": 1,
        "accepted_validated_receipt_sha256": v2.VALIDATED_RECEIPT_SHA,
    }
    return normalized


def spec(root: Path) -> dict[str, Any]:
    value = load(root / SPEC_PATH)
    validate(value, root)
    return value


def validate(value: dict[str, Any], root: Path) -> None:
    if value.get("schema_version") != "fleet-qwen38-dedicated-serving-v5-authorized-spec-v1":
        raise ValueError("Qwen v5 schema drifted")
    if value.get("title") != TITLE or value.get("run_dir") != RUN_DIR:
        raise ValueError("Qwen v5 identity drifted")
    if value.get("scope") != (
        "one_non_scored_actual_harness_parity_server_then_rank3_attempts1_through4"
    ):
        raise ValueError("Qwen v5 scope drifted")
    if value.get("evaluation") != {
        "fresh_actual_opencode_parity_required_before_scoring": True,
        "scored_tasks_allowed_before_parity": False,
        "parity_module": "evals.fleet.opencode_actual_harness_parity_v1",
        "rank3_attempts_allowed_after_parity_and_hold_transfer": [1, 2, 3, 4],
        "same_task_max_inflight": 1,
    }:
        raise ValueError("Qwen v5 parity policy drifted")
    if value.get("create_once") != {
        "additional_replica_allowed": True,
        "api_route": "POST /v1/runs",
        "duplicate_checks_required": True,
        "max_project_gpu_nodes": MAX_PROJECT_GPU_NODES,
        "max_project_gpus": MAX_PROJECT_GPUS,
    }:
        raise ValueError("Qwen v5 create-once ceiling drifted")
    prior = load(root / PRIOR_VALIDATED_RECEIPT)
    if (
        prior.get("receipt_sha256") != PRIOR_VALIDATED_RECEIPT_SHA256
        or self_hosted.digest_without(prior, "receipt_sha256") != PRIOR_VALIDATED_RECEIPT_SHA256
        or value.get("prior_validated_cell")
        != {
            "selection_rank": 2,
            "attempt": 3,
            "accepted_validated_receipt_sha256": PRIOR_VALIDATED_RECEIPT_SHA256,
        }
    ):
        raise ValueError("Qwen v5 prior validated cell drifted")
    if (
        actual_parity.IMAGE != "chris/opencode:1.18.27-cyber-v1"
        or actual_parity.IMAGE_ID
        != "sha256:ca4f0b8f50bd051d709c7c0ae5ec47ca31bbff7d2a2ad754c67b9cdf585567cb"
    ):
        raise ValueError("actual OpenCode parity image drifted")
    v2.validate(_as_v2(value), root)


def payload(value: dict[str, Any], root: Path) -> dict[str, Any]:
    validate(value, root)
    result = v2.payload(_as_v2(value), root)
    result["title"] = TITLE
    result["run_dir"] = RUN_DIR
    result["env"]["QWEN38_RUN_DIR"] = RUN_DIR
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
