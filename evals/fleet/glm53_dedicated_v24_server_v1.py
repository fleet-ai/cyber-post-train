"""Held GLM v24 TP8 server payload with one external idle-release authority."""

from __future__ import annotations

import re
import shlex
from typing import Any

from evals.fleet import exact_pass4_crypto as crypto
from evals.fleet import glm53_dedicated_v23_request_counter_watchdog_v1 as watchdog

SCHEMA = "fleet-glm53-dedicated-v24-server-held-v1"
TITLE = "chris-cyber-evalserve-glm53-tp8-a-v24"
RUN_DIR = "/mnt/sfs/jobs/chris-cyber-evalserve-glm53-tp8-a-v24"
IMAGE = (
    "ghcr.io/fleet-ai/cyber-post-train-glm53-runtime@"
    "sha256:ec93ba50613fd13fb4c0b0a9105767ab18209a1e0108dab0923aad694c0206ec"
)
MODEL_REVISION = "30333038ada1f1dacb294a93270305a890b50c14"
SERVED_ID = "glm-5.3"
CONTEXT_LENGTH = 262144
PRIORITY_CLASS = "fleet-infra-quiet"
PRIORITY_SELECTION = "highest_jobs_api_admitted_nonpreempting"
PREEMPTION_POLICY = "Never"
QUEUE = "training-lq"
SERVER_ARGV = (
    "python3",
    "-m",
    "sglang.launch_server",
    "--model-path",
    "/mnt/sfs/models/glm-5.3-30333038",
    "--served-model-name",
    SERVED_ID,
    "--host",
    "0.0.0.0",
    "--port",
    "8000",
    "--trust-remote-code",
    "--tp-size",
    "8",
    "--dp-size",
    "8",
    "--ep-size",
    "8",
    "--enable-dp-attention",
    "--context-length",
    str(CONTEXT_LENGTH),
    "--quantization",
    "fp8",
    "--kv-cache-dtype",
    "bfloat16",
    "--mem-fraction-static",
    "0.85",
    "--attention-backend",
    "dsa",
    "--dsa-prefill-backend",
    "trtllm",
    "--dsa-decode-backend",
    "trtllm",
    "--moe-a2a-backend",
    "deepep",
    "--speculative-algorithm",
    "EAGLE",
    "--speculative-draft-model-path",
    "/mnt/sfs/models/glm-5.3-30333038",
    "--speculative-num-steps",
    "1",
    "--speculative-eagle-topk",
    "1",
    "--speculative-num-draft-tokens",
    "2",
    "--reasoning-parser",
    "glm45",
    "--tool-call-parser",
    "glm47",
    "--enable-metrics",
    "--enable-mfu-metrics",
    "--enable-metrics-for-all-schedulers",
)


class ServerPlanError(RuntimeError):
    """The held v24 server contract is inconsistent."""


def payload() -> dict[str, Any]:
    """Return the exact preview-only Jobs API request body."""

    return {
        "title": TITLE,
        "image": IMAGE,
        "command": "bash -lc 'exec \"$@\"' -- " + shlex.join(SERVER_ARGV),
        "workers": 1,
        "gpus_per_worker": 8,
        "run_dir": RUN_DIR,
        "resources": {
            "cpu_request": "64",
            "cpu_limit": "192",
            "memory_request": "768Gi",
            "memory_limit": "2Ti",
        },
        "env": {
            "GLM53_REPLICA": "A",
            "GLM53_RUN_DIR": RUN_DIR,
            "HF_HUB_OFFLINE": "1",
            "NVSHMEM_REMOTE_TRANSPORT": "none",
            "TRANSFORMERS_OFFLINE": "1",
        },
        "secrets": [],
        "privileged": True,
        "priority_class": PRIORITY_CLASS,
        "topology_mode": "required",
    }


def validate_payload(value: dict[str, Any]) -> None:
    expected = payload()
    command = value.get("command")
    if (
        value != expected
        or not isinstance(command, str)
        or "sglang.launch_server" not in command
        or "--context-length 262144" not in command
        or "--enable-metrics" not in command
        or "GLM53_IDLE_SECONDS" in command
        or "IDLE_SECONDS" in command
        or "metrics_digest" in command
        or "MODEL-TRAFFIC" in command
        or "kill -TERM" in command
    ):
        raise ServerPlanError("v24_payload_or_idle_authority_invalid")


def build_held(package_commit: str) -> dict[str, Any]:
    if re.fullmatch(r"[0-9a-f]{40}", package_commit) is None:
        raise ServerPlanError("v24_package_commit_invalid")
    request = payload()
    validate_payload(request)
    body: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": "READY_PREVIEW_ONLY_HELD",
        "package_commit": package_commit,
        "server": {
            "title": TITLE,
            "run_dir": RUN_DIR,
            "image": IMAGE,
            "model_revision": MODEL_REVISION,
            "served_id": SERVED_ID,
            "context_length": CONTEXT_LENGTH,
            "nodes": 1,
            "gpus": 8,
            "priority_class": PRIORITY_CLASS,
            "priority_selection": PRIORITY_SELECTION,
            "preemption_policy": PREEMPTION_POLICY,
            "queue": QUEUE,
        },
        "request_sha256": crypto.sha256(crypto.canonical_json(request)),
        "idle_release_authority": {
            "server_internal_idle_killer_present": False,
            "external_watchdog_required": True,
            "implementation_module": (
                "evals.fleet.glm53_dedicated_v23_request_counter_watchdog_v1"
            ),
            "implementation_sha256": watchdog.source_sha256(),
            "idle_release_seconds": watchdog.IDLE_RELEASE_SECONDS,
            "activity_metrics": list(watchdog.ACTIVITY_METRICS),
            "active_request_or_queue_refreshes": True,
            "health_or_process_liveness_refreshes": False,
            "release_route": watchdog.RELEASE_ROUTE,
            "uid_bound_jobs_api_release_required": True,
            "uid_bound_kubernetes_absence_confirmation_required": True,
        },
        "fresh_preview_required_immediately_before_create": True,
        "fresh_duplicate_capacity_sfs_gates_required_immediately_before_create": True,
        "server_launch_authorized": False,
        "qualification_launch_authorized": False,
        "scored_launch_authorized": False,
        "api_mutation_calls": 0,
        "protected_content_included": False,
    }
    body["receipt_sha256"] = crypto.digest_without(body, "receipt_sha256")
    return body
