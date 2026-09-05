"""Create-once authority for one full-node Qwen3.8 DP8 parity server."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any

from evals.fleet import glm53_dedicated_v7 as api
from evals.fleet import qwen38_dedicated_v1 as tp1
from evals.fleet import qwen38_dedicated_v5 as proven
from evals.fleet import self_hosted

SPEC_PATH = "evals/fleet/configs/qwen38-dedicated-dp8-v1-authorized.json"
LIFECYCLE_PATH = "evals/fleet/scripts/qwen38_dedicated_dp8_lifecycle_v1.sh"
TITLE = "chris-cyber-evalserve-q38-dp8-a-v1"
RUN_DIR = "/mnt/sfs/jobs/chris-cyber-evalserve-q38-dp8-a-v1"
PRIORITY_CLASS = "fleet-infra-quiet"
API_PRIORITY_CLASSES = {"fleet-train-high", "fleet-infra-quiet"}
IMAGE = tp1.IMAGE
MODEL_REVISION = tp1.MODEL_REVISION
MODEL_PATH = tp1.MODEL_PATH
MAX_PROJECT_GPU_NODES = 2
MAX_PROJECT_GPUS = 16
SERVER_ARGUMENTS = (
    "python3",
    "-m",
    "sglang.launch_server",
    "--model-path",
    MODEL_PATH,
    "--served-model-name",
    "qwen3.8-27b",
    "--host",
    "0.0.0.0",
    "--port",
    "8000",
    "--trust-remote-code",
    "--tp-size",
    "1",
    "--dp-size",
    "8",
    "--load-balance-method",
    "total_tokens",
    "--context-length",
    "262144",
    "--kv-cache-dtype",
    "fp8_e4m3",
    "--mem-fraction-static",
    "0.85",
    "--attention-backend",
    "trtllm_mha",
    "--chunked-prefill-size",
    "32768",
    "--max-prefill-tokens",
    "32768",
    "--mamba-full-memory-ratio",
    "4.58",
    "--reasoning-parser",
    "qwen3",
    "--tool-call-parser",
    "qwen3_coder",
    "--enable-metrics",
    "--enable-mfu-metrics",
    "--enable-metrics-for-all-schedulers",
)
SERVER_ARGUMENTS_SHA256 = self_hosted.sha256(self_hosted.canonical_json(list(SERVER_ARGUMENTS)))


def load(path: Path) -> dict[str, Any]:
    return tp1.load(path)


def spec(root: Path) -> dict[str, Any]:
    value = load(root / SPEC_PATH)
    validate(value, root)
    return value


def validate(value: dict[str, Any], root: Path) -> None:
    if value.get("schema_version") != "fleet-qwen38-dedicated-dp8-v1-authorized-spec-v1":
        raise ValueError("Qwen DP8 schema drifted")
    if value.get("status") != "AUTHORIZED" or value.get("launch_authorized") is not True:
        raise ValueError("Qwen DP8 is not authorized")
    if value.get("scope") != "one_non_scored_actual_harness_parity_server_only":
        raise ValueError("Qwen DP8 scope drifted")
    if value.get("title") != TITLE or value.get("run_dir") != RUN_DIR:
        raise ValueError("Qwen DP8 identity drifted")
    if value.get("model") != {
        "repository": "Qwen/Qwen3.8-27B",
        "revision": MODEL_REVISION,
        "served_id": "qwen3.8-27b",
        "sfs_path": MODEL_PATH,
        "context_length": 262144,
        "staging_receipt_sha256": tp1.STAGING_RECEIPT_SHA,
    }:
        raise ValueError("Qwen DP8 model identity drifted")
    if value.get("runtime") != {
        "image": IMAGE,
        "sglang_version": "0.5.18",
        "tensor_parallel_size": 1,
        "data_parallel_size": 8,
        "load_balance_method": "total_tokens",
        "reasoning_parser": "qwen3",
        "tool_call_parser": "qwen3_coder",
    }:
        raise ValueError("Qwen DP8 runtime drifted")
    if value.get("resources") != {
        "workers": 1,
        "gpus_per_worker": 8,
        "cpu_request": "32",
        "cpu_limit": "96",
        "memory_request": "256Gi",
        "memory_limit": "768Gi",
        "privileged": False,
        "priority_class": PRIORITY_CLASS,
        "preemption_policy": "Never",
    }:
        raise ValueError("Qwen DP8 resources drifted")
    if value.get("create_once") != {
        "api_route": "POST /v1/runs",
        "duplicate_checks_required": True,
        "max_project_gpu_nodes": MAX_PROJECT_GPU_NODES,
        "max_project_gpus": MAX_PROJECT_GPUS,
    }:
        raise ValueError("Qwen DP8 create-once ceiling drifted")
    if value.get("evaluation") != {
        "fresh_actual_opencode_parity_required_before_scoring": True,
        "scored_tasks_allowed": False,
        "scored_successor_requires_separate_release": True,
        "serving_block": "dedicated-qwen-dp8-a-v1",
    }:
        raise ValueError("Qwen DP8 evaluation policy drifted")
    prior = load(root / proven.PRIOR_VALIDATED_RECEIPT)
    if (
        prior.get("receipt_sha256") != proven.PRIOR_VALIDATED_RECEIPT_SHA256
        or self_hosted.digest_without(prior, "receipt_sha256")
        != proven.PRIOR_VALIDATED_RECEIPT_SHA256
        or value.get("prior_validated_cell_receipt_sha256")
        != proven.PRIOR_VALIDATED_RECEIPT_SHA256
    ):
        raise ValueError("Qwen DP8 prior validated cell drifted")
    if value.get("server_arguments_sha256") != SERVER_ARGUMENTS_SHA256:
        raise ValueError("Qwen DP8 server arguments drifted")
    lifecycle = root / LIFECYCLE_PATH
    if lifecycle.is_symlink() or not lifecycle.is_file():
        raise ValueError("Qwen DP8 lifecycle is absent or unsafe")


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
            "QWEN38_RUN_DIR": RUN_DIR,
        },
        "secrets": [],
        "resources": {
            "cpu_request": "32",
            "cpu_limit": "96",
            "memory_request": "256Gi",
            "memory_limit": "768Gi",
        },
        "priority_class": PRIORITY_CLASS,
        "privileged": False,
        "run_dir": RUN_DIR,
        "title": TITLE,
    }
    if set(result) != api.EXPECTED_API_FIELDS:
        raise AssertionError("Qwen DP8 Jobs API payload shape drifted")
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
