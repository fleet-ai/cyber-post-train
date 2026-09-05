"""Create-once authority for one non-scored GLM5.3 dedicated v8 server."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any

from evals.fleet import glm53_dedicated_v7 as v7

SPEC_PATH = "evals/fleet/configs/glm53-dedicated-serving-v8-authorized.json"
LIFECYCLE_PATH = "evals/fleet/scripts/glm53_dedicated_v6_lifecycle.sh"
TITLE = "chris-cyber-evalserve-glm53-tp8-a-v8"
RUN_DIR = "/mnt/sfs/jobs/chris-cyber-evalserve-glm53-tp8-a-v8"
IMAGE = v7.IMAGE
MODEL_REVISION = v7.MODEL_REVISION
MODEL_PATH = v7.MODEL_PATH
SERVER_ARGUMENTS = v7.CANONICAL_SERVER_ARGUMENTS
SERVER_ARGUMENTS_SHA256 = v7.CANONICAL_SERVER_ARGUMENTS_SHA256


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
    if value.get("schema_version") != "fleet-glm53-dedicated-serving-v8-authorized-spec-v1":
        raise ValueError("v8 schema drifted")
    if value.get("status") != "AUTHORIZED" or value.get("launch_authorized") is not True:
        raise ValueError("v8 launch is not explicitly authorized")
    if value.get("scope") != "one_non_scored_dedicated_server_only":
        raise ValueError("v8 scope drifted")
    if value.get("title") != TITLE or value.get("run_dir") != RUN_DIR:
        raise ValueError("v8 create-once identity drifted")
    if value.get("model") != {
        "repository": "zai-org/GLM-5.3",
        "revision": MODEL_REVISION,
        "served_id": "glm-5.3",
        "sfs_path": MODEL_PATH,
        "context_length": 262144,
        "staging_receipt_sha256": v7.STAGING_RECEIPT_SHA,
    }:
        raise ValueError("v8 model identity drifted")
    if value.get("runtime", {}).get("image") != IMAGE:
        raise ValueError("v8 runtime image drifted")
    if value.get("resources") != {
        "workers": 1,
        "gpus_per_worker": 8,
        "cpu_request": "96",
        "cpu_limit": "192",
        "memory_request": "1Ti",
        "memory_limit": "2Ti",
        "privileged": True,
        "priority_class": "fleet-infra-quiet",
        "preemption_policy": "Never",
    }:
        raise ValueError("v8 resource policy drifted")
    if value.get("server_arguments_sha256") != SERVER_ARGUMENTS_SHA256:
        raise ValueError("v8 server arguments drifted")
    if value.get("create_once", {}).get("replica_b_allowed") is not False:
        raise ValueError("v8 unexpectedly authorizes replica B")
    if value.get("evaluation") != {
        "scored_tasks_allowed": False,
        "parity_required_before_scored_canary": True,
        "accepted_canary_required_before_bulk": True,
    }:
        raise ValueError("v8 evaluation gate drifted")
    lifecycle = root / LIFECYCLE_PATH
    if lifecycle.is_symlink() or not lifecycle.is_file():
        raise ValueError("v8 lifecycle is absent or unsafe")


def payload(value: dict[str, Any], root: Path) -> dict[str, Any]:
    validate(value, root)
    lifecycle = (root / LIFECYCLE_PATH).read_text()
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
    }
    if set(result) != v7.EXPECTED_API_FIELDS:
        raise AssertionError("v8 Jobs API payload shape drifted")
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
