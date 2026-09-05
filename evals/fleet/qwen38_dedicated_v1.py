"""Create-once authority for one non-scored Qwen3.8-27B TP1 parity canary."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any

from evals.fleet import glm53_dedicated_v7 as v7
from evals.fleet import self_hosted

SPEC_PATH = "evals/fleet/configs/qwen38-dedicated-serving-v1-authorized.json"
LIFECYCLE_PATH = "evals/fleet/scripts/qwen38_dedicated_lifecycle_v1.sh"
TITLE = "chris-cyber-evalserve-q38-tp1-a-v1"
RUN_DIR = "/mnt/sfs/jobs/chris-cyber-evalserve-q38-tp1-a-v1"
IMAGE = v7.IMAGE
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
MODEL_PATH = "/mnt/sfs/models/qwen3.8-27b-1d4bf0f2"
STAGING_RECEIPT_SHA = "sha256:9a680f714de5c72e60dc6b08f363a8c82658dc1540bbb9aab90a3b8939445869"
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
SERVER_ARGUMENTS_SHA256 = "sha256:836d25fc3df3f0a1ecfc53788492cfa4f97df3462c4a4b0e6c23f272016f14c0"


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
    if value.get("schema_version") != "fleet-qwen38-dedicated-serving-v1-authorized-spec-v1":
        raise ValueError("Qwen v1 schema drifted")
    if value.get("status") != "AUTHORIZED" or value.get("launch_authorized") is not True:
        raise ValueError("Qwen v1 is not authorized")
    if value.get("scope") != "one_non_scored_parity_canary_only":
        raise ValueError("Qwen v1 scope drifted")
    if value.get("title") != TITLE or value.get("run_dir") != RUN_DIR:
        raise ValueError("Qwen v1 identity drifted")
    if value.get("model") != {
        "repository": "Qwen/Qwen3.8-27B",
        "revision": MODEL_REVISION,
        "served_id": "qwen3.8-27b",
        "sfs_path": MODEL_PATH,
        "context_length": 262144,
        "staging_receipt_sha256": STAGING_RECEIPT_SHA,
    }:
        raise ValueError("Qwen v1 model identity drifted")
    runtime = value.get("runtime") or {}
    if runtime.get("image") != IMAGE or runtime.get("sglang_version") != "0.5.18":
        raise ValueError("Qwen v1 runtime drifted")
    if not all(
        runtime.get(key) is True
        for key in (
            "qwen35_architecture_present",
            "qwen3_reasoning_parser_present",
            "qwen3_coder_tool_parser_present",
        )
    ):
        raise ValueError("Qwen v1 static runtime compatibility is absent")
    if value.get("resources") != {
        "workers": 1,
        "gpus_per_worker": 1,
        "cpu_request": "32",
        "cpu_limit": "96",
        "memory_request": "256Gi",
        "memory_limit": "768Gi",
        "privileged": True,
        "priority_class": "fleet-infra-quiet",
        "preemption_policy": "Never",
    }:
        raise ValueError("Qwen v1 resource policy drifted")
    basis = value.get("shape_basis") or {}
    if any(
        basis.get(key) != expected
        for key, expected in {
            "exact_live_inference_model_uid": "d06c0531-7181-41ad-a3fd-cd8e3e774ab9",
            "exact_live_revision_ready": True,
            "exact_live_context_length": 262144,
            "exact_live_tensor_parallel_size": 1,
            "exact_live_gpu_count": 1,
            "serialized_weight_bytes": 55563006776,
            "kv_cache_dtype": "fp8_e4m3",
        }.items()
    ):
        raise ValueError("Qwen v1 TP1 basis drifted")
    if value.get("server_arguments_sha256") != SERVER_ARGUMENTS_SHA256:
        raise ValueError("Qwen v1 server arguments drifted")
    actual_arguments_sha = self_hosted.sha256(self_hosted.canonical_json(list(SERVER_ARGUMENTS)))
    if actual_arguments_sha != SERVER_ARGUMENTS_SHA256:
        raise ValueError("Qwen v1 canonical server arguments do not match their digest")
    if value.get("create_once", {}).get("additional_replica_allowed") is not False:
        raise ValueError("Qwen v1 unexpectedly authorizes another replica")
    if value.get("evaluation", {}).get("scored_tasks_allowed") is not False:
        raise ValueError("Qwen v1 unexpectedly authorizes scoring")
    lifecycle = root / LIFECYCLE_PATH
    if lifecycle.is_symlink() or not lifecycle.is_file():
        raise ValueError("Qwen v1 lifecycle is absent or unsafe")


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
        "privileged": True,
        "run_dir": RUN_DIR,
        "title": TITLE,
    }
    if set(result) != v7.EXPECTED_API_FIELDS:
        raise AssertionError("Qwen v1 Jobs API payload shape drifted")
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
