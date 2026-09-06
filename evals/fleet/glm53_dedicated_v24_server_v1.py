"""Held GLM v24 TP8 server payload with one external idle-release authority."""

from __future__ import annotations

import base64
import re
import shlex
import uuid
from typing import Any

from evals.fleet import exact_pass4_crypto as crypto
from evals.fleet import glm53_dedicated_v23_request_counter_watchdog_v1 as watchdog

SCHEMA = "fleet-glm53-dedicated-v24-server-held-v1"
READY_SCHEMA = "fleet-glm53-dedicated-v24-application-ready-v1"
TITLE = "chris-cyber-evalserve-glm53-tp8-a-v24"
RUN_DIR = "/mnt/sfs/jobs/chris-cyber-evalserve-glm53-tp8-a-v24"
READY_PATH = RUN_DIR + "/READY.json"
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

READY_OBSERVER = f"""
import datetime
import hashlib
import json
import os
import pathlib
import time
import urllib.request

pid = int(os.environ["GLM53_SERVER_PID"])
deadline = time.time() + 7200
while True:
    state = pathlib.Path(f"/proc/{{pid}}/stat")
    if not state.exists() or state.read_text().split()[2] == "Z":
        raise SystemExit("server exited before application readiness")
    try:
        with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=5) as response:
            if response.status == 200:
                break
    except OSError:
        pass
    if time.time() >= deadline:
        raise SystemExit("application readiness deadline exceeded")
    time.sleep(2)
ready_at = time.time()
body = {{
    "schema_version": {READY_SCHEMA!r},
    "status": "APPLICATION_HEALTH_HTTP_200",
    "server_title": {TITLE!r},
    "server_run_dir": {RUN_DIR!r},
    "served_id": {SERVED_ID!r},
    "model_revision": {MODEL_REVISION!r},
    "context_length": {CONTEXT_LENGTH},
    "ready_at_epoch": ready_at,
    "ready_at_utc": datetime.datetime.fromtimestamp(
        ready_at, datetime.UTC
    ).isoformat().replace("+00:00", "Z"),
    "health_http_status": 200,
    "prompts_traces_flags_scores_or_model_outputs_included": False,
}}
canonical = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
body["receipt_sha256"] = "sha256:" + hashlib.sha256(canonical).hexdigest()
target = pathlib.Path({READY_PATH!r})
target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
with os.fdopen(fd, "w") as handle:
    json.dump(body, handle, sort_keys=True, separators=(",", ":"))
    handle.write("\\n")
""".strip()


class ServerPlanError(RuntimeError):
    """The held v24 server contract is inconsistent."""


def validate_binding(binding: dict[str, Any]) -> None:
    expected_keys = {
        "server_title",
        "server_run_dir",
        "api_run_id",
        "rayjob_uid",
        "workload_uid",
        "head_pod_uid",
        "service_uid",
        "service_origin",
        "served_id",
        "model_revision",
        "context_length",
    }
    try:
        uids_valid = all(
            uuid.UUID(str(binding[key])).int != 0
            for key in ("rayjob_uid", "workload_uid", "head_pod_uid", "service_uid")
        )
    except (KeyError, ValueError):
        uids_valid = False
    if (
        set(binding) != expected_keys
        or binding.get("server_title") != TITLE
        or binding.get("server_run_dir") != RUN_DIR
        or not str(binding.get("api_run_id", "")).startswith("ft-run-")
        or not uids_valid
        or not str(binding.get("service_origin", "")).startswith("http://")
        or binding.get("served_id") != SERVED_ID
        or binding.get("model_revision") != MODEL_REVISION
        or binding.get("context_length") != CONTEXT_LENGTH
    ):
        raise ServerPlanError("v24_server_binding_invalid")


def render_server_command(
    *,
    observer_source: str = READY_OBSERVER,
    server_argv: tuple[str, ...] = SERVER_ARGV,
) -> str:
    """Render one command that survives the Jobs API's outer ``/bin/sh -c``."""

    if not observer_source or not server_argv or any(not item for item in server_argv):
        raise ServerPlanError("v24_server_command_input_invalid")
    observer_b64 = base64.b64encode(observer_source.encode("utf-8")).decode("ascii")
    observer_runner = shlex.join(
        (
            "python3",
            "-c",
            (
                "import base64,sys; "
                "source=base64.b64decode(sys.argv[1], validate=True); "
                "exec(compile(source, '<glm53-ready-observer>', 'exec'))"
            ),
            observer_b64,
        )
    )
    shell_program = "\n".join(
        (
            "set -euo pipefail",
            '"$@" &',
            "server_pid=$!",
            "export GLM53_SERVER_PID=$server_pid",
            'cleanup() { kill "$server_pid" 2>/dev/null || true; }',
            "trap cleanup EXIT",
            observer_runner,
            'wait "$server_pid"',
            "trap - EXIT",
        )
    )
    return shlex.join(("bash", "-lc", shell_program, "--", *server_argv))


def payload() -> dict[str, Any]:
    """Return the exact preview-only Jobs API request body."""

    return {
        "title": TITLE,
        "image": IMAGE,
        "command": render_server_command(),
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
        or base64.b64encode(READY_OBSERVER.encode("utf-8")).decode("ascii") not in command
        or READY_PATH not in READY_OBSERVER
        or "APPLICATION_HEALTH_HTTP_200" not in READY_OBSERVER
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
