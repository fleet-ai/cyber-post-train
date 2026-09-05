"""Held GLM5.3 TP8 successor with an explicit required TAS contract."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any

from evals.fleet import glm53_dedicated_v7 as v7
from evals.fleet import glm53_dedicated_v8 as v8

SPEC_PATH = "evals/fleet/configs/glm53-dedicated-serving-v11-authorized.json"
VERSION = "v11"
TITLE = "chris-cyber-evalserve-glm53-tp8-a-v11"
RUN_DIR = "/mnt/sfs/jobs/chris-cyber-evalserve-glm53-tp8-a-v11"
PRIORITY_CLASS = "fleet-infra-quiet"
API_PRIORITY_CLASSES = {"fleet-train-high", "fleet-infra-quiet"}
TOPOLOGY_MODE = "required"
TOPOLOGY_LEVEL = "topology.nebius.com/tier-1"
QUEUE = "training-lq"
IMAGE = v8.IMAGE
MODEL_REVISION = v8.MODEL_REVISION
MODEL_PATH = v8.MODEL_PATH
SERVER_ARGUMENTS = v8.SERVER_ARGUMENTS
SERVER_ARGUMENTS_SHA256 = v8.SERVER_ARGUMENTS_SHA256
EXPECTED_API_FIELDS = v7.EXPECTED_API_FIELDS | {"topology_mode"}


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
    if value.get("schema_version") != "fleet-glm53-dedicated-serving-v11-authorized-v1":
        raise ValueError("v11 schema drifted")
    if value.get("status") != "AUTHORIZED" or value.get("launch_authorized") is not True:
        raise ValueError("v11 launch is not explicitly authorized")
    if value.get("scope") != "one_non_scored_dedicated_server_only":
        raise ValueError("v11 authorization scope drifted")
    if value.get("title") != TITLE or value.get("run_dir") != RUN_DIR:
        raise ValueError("v11 create-once identity drifted")
    if value.get("model") != {
        "repository": "zai-org/GLM-5.3",
        "revision": MODEL_REVISION,
        "served_id": "glm-5.3",
        "sfs_path": MODEL_PATH,
        "context_length": 262144,
        "staging_receipt_sha256": v7.STAGING_RECEIPT_SHA,
    }:
        raise ValueError("v11 model identity drifted")
    if value.get("runtime", {}).get("image") != IMAGE:
        raise ValueError("v11 runtime image drifted")
    if value.get("resources") != {
        "workers": 1,
        "gpus_per_worker": 8,
        "cpu_request": "96",
        "cpu_limit": "192",
        "memory_request": "1Ti",
        "memory_limit": "2Ti",
        "privileged": True,
        "priority_class": PRIORITY_CLASS,
        "preemption_policy": "Never",
    }:
        raise ValueError("v11 resource policy drifted")
    if value.get("topology") != {
        "queue": QUEUE,
        "mode": TOPOLOGY_MODE,
        "requested_level": None,
        "rendered_default_level": TOPOLOGY_LEVEL,
        "required_live_flavor": "b300-training",
        "recheck_unreserved_quota_immediately_before_submit": True,
    }:
        raise ValueError("v11 topology policy drifted")
    if value.get("server_arguments_sha256") != SERVER_ARGUMENTS_SHA256:
        raise ValueError("v11 server arguments drifted")
    if value.get("two_node_tp16") != {
        "allowed": False,
        "reason": "current_entrypoint_is_single_node_tp8_and_workers_2_is_not_distributed_serving",
    }:
        raise ValueError("v11 two-node boundary drifted")
    lifecycle = root / v8.LIFECYCLE_PATH
    if lifecycle.is_symlink() or not lifecycle.is_file():
        raise ValueError("v11 lifecycle is absent or unsafe")


def payload(value: dict[str, Any], root: Path) -> dict[str, Any]:
    validate(value, root)
    lifecycle = (root / v8.LIFECYCLE_PATH).read_text()
    result = {
        "image": IMAGE,
        "command": "bash -lc " + shlex.quote(lifecycle) + " -- " + shlex.join(SERVER_ARGUMENTS),
        "workers": 1,
        "gpus_per_worker": 8,
        "env": {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "NVSHMEM_REMOTE_TRANSPORT": "none",
            "GLM53_RUN_DIR": RUN_DIR,
            "GLM53_REPLICA": "A",
        },
        "secrets": [],
        "resources": {
            "cpu_request": "96",
            "cpu_limit": "192",
            "memory_request": "1Ti",
            "memory_limit": "2Ti",
        },
        "priority_class": "fleet-infra-quiet",
        "privileged": True,
        "run_dir": RUN_DIR,
        "title": TITLE,
        "topology_mode": TOPOLOGY_MODE,
    }
    if set(result) != EXPECTED_API_FIELDS:
        raise AssertionError("v11 Jobs API payload shape drifted")
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
