"""Render a non-scored CPU qualification for the GLM server-local traffic watchdog."""

from __future__ import annotations

import argparse
import hashlib
import json
import textwrap
from pathlib import Path
from typing import Any

import yaml

from evals.fleet import self_hosted

JOB_NAME = "chris-glm53-dedicated-metric-watchdog-canary-v2"
CONFIGMAP_NAME = JOB_NAME + "-run"
OUTPUT_ROOT = f"/mnt/sfs/jobs/{JOB_NAME}"
LIFECYCLE = Path("evals/fleet/scripts/glm53_dedicated_metric_lifecycle_v1.sh")
IMAGE = (
    "ghcr.io/astral-sh/uv:python3.12-bookworm@"
    "sha256:9aa60c50016c0485636ab9a830246a6ef3399aa4a8bab3d17ef4a2358fba2ca7"
)

PROBE = r'''
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time
import urllib.request

output = pathlib.Path(os.environ["OUTPUT_ROOT"])
if output.exists() or output.is_symlink():
    raise RuntimeError("watchdog canary output collision")
output.mkdir(mode=0o700, parents=True)
projected = pathlib.Path("/bootstrap/lifecycle.sh")
resolved = projected.resolve()
if not projected.is_file() or pathlib.Path("/bootstrap") not in resolved.parents:
    raise RuntimeError("watchdog lifecycle projected source is invalid")
lifecycle = pathlib.Path("/tmp/glm53-metric-lifecycle.sh")
fd = os.open(lifecycle, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o700)
with os.fdopen(fd, "wb") as handle:
    handle.write(projected.read_bytes())
if lifecycle.is_symlink() or not lifecycle.is_file():
    raise RuntimeError("watchdog lifecycle private copy is not a regular file")
lifecycle_sha = "sha256:" + hashlib.sha256(lifecycle.read_bytes()).hexdigest()

server_source = r"""
import http.server
import pathlib
import sys

counter = pathlib.Path(sys.argv[1])
counter.write_text("0")

class Server(http.server.ThreadingHTTPServer):
    allow_reuse_address = True

class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health": body = b"ok\\n"
        elif self.path == "/metrics":
            value = int(counter.read_text())
            body = (
                f"sglang:num_requests_total{{model_name=\\\"glm-5.3\\\"}} {value}\\n"
                f"sglang:prompt_tokens_total{{model_name=\\\"glm-5.3\\\"}} {value * 2}\\n"
                f"sglang:generation_tokens_total{{model_name=\\\"glm-5.3\\\"}} {value * 3}\\n"
            ).encode()
        else:
            self.send_error(404); return
        self.send_response(200); self.send_header("Content-Length", str(len(body)))
        self.end_headers(); self.wfile.write(body)
    def do_POST(self):
        if self.path != "/v1/chat/completions": self.send_error(404); return
        counter.write_text(str(int(counter.read_text()) + 1))
        body = b"{}"; self.send_response(200)
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def log_message(self, *_args): return

Server(("127.0.0.1", 8000), Handler).serve_forever()
"""

def start(name):
    run_dir = output / name
    source = output / (name + ".py")
    count = output / (name + ".count")
    source.write_text(server_source)
    env = {
        **os.environ,
        "GLM53_RUN_DIR": str(run_dir),
        "GLM53_REPLICA": "A",
        "GLM53_IDLE_SECONDS": "5",
        "GLM53_POLL_SECONDS": "1",
    }
    proc = subprocess.Popen(
        ["bash", str(lifecycle), sys.executable, str(source), str(count)],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    deadline = time.time() + 15
    while time.time() < deadline and not (run_dir / "lifecycle/READY").is_file():
        if proc.poll() is not None: raise RuntimeError("watchdog canary exited before readiness")
        time.sleep(0.1)
    if not (run_dir / "lifecycle/READY").is_file(): raise RuntimeError("readiness timeout")
    return proc, run_dir

idle, idle_dir = start("idle")
for _ in range(5):
    if urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2).status != 200:
        raise RuntimeError("health probe failed")
    time.sleep(0.5)
if idle.wait(timeout=12) != 0 or not (idle_dir / "lifecycle/IDLE-TIMEOUT").is_file():
    raise RuntimeError("health-only workload did not release")
if (idle_dir / "lifecycle/MODEL-TRAFFIC.json").exists():
    raise RuntimeError("health probe incorrectly counted as model traffic")

active, active_dir = start("active")
request_count = 0
started = time.time()
try:
    for _ in range(8):
        request = urllib.request.Request(
            "http://127.0.0.1:8000/v1/chat/completions", data=b"{}", method="POST"
        )
        if urllib.request.urlopen(request, timeout=2).status != 200:
            raise RuntimeError("model request probe failed")
        request_count += 1
        time.sleep(1)
        if active.poll() is not None:
            raise RuntimeError("real model traffic did not preserve server")
    traffic = json.loads((active_dir / "lifecycle/MODEL-TRAFFIC.json").read_text())
    claimed = traffic.pop("receipt_sha256")
    actual = "sha256:" + hashlib.sha256(
        json.dumps(traffic, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if claimed != actual or traffic.get("source") != "local_sglang_inference_metrics":
        raise RuntimeError("model traffic receipt invalid")
    (active_dir / "lifecycle/DRAIN").write_text("qualified\\n")
    if active.wait(timeout=8) != 0:
        raise RuntimeError("qualified server did not drain")
finally:
    if active.poll() is None:
        active.terminate(); active.wait(timeout=5)

body = {
    "schema_version": "fleet-glm53-dedicated-metric-watchdog-canary-v1",
    "status": "PASSED_NON_SCORED",
    "lifecycle_file_sha256": lifecycle_sha,
    "production_idle_seconds": 600,
    "canary_idle_seconds": 5,
    "health_only_released": True,
    "health_probes_counted_as_traffic": False,
    "actual_model_requests": request_count,
    "active_survived_seconds": round(time.time() - started, 3),
    "actual_model_traffic_prevented_release": True,
    "server_local_uid_scope": True,
    "task_instance_session_verifier_scoring_calls": 0,
    "prompts_traces_flags_or_scores_read": False,
}
body["receipt_sha256"] = "sha256:" + hashlib.sha256(
    json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
target = output / "QUALIFIED.json"
fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, "w") as handle:
    handle.write(json.dumps(body, sort_keys=True, separators=(",", ":")) + "\\n")
'''.strip()


def render(root: Path) -> dict[str, Any]:
    lifecycle = root / LIFECYCLE
    if lifecycle.is_symlink() or not lifecycle.is_file():
        raise ValueError("watchdog lifecycle source must be a regular file")
    source = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": CONFIGMAP_NAME, "namespace": "fleet-train-jobs"},
        "immutable": True,
        "data": {"lifecycle.sh": lifecycle.read_text(), "probe.py": PROBE},
    }
    job = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": JOB_NAME,
            "namespace": "fleet-train-jobs",
            "labels": {
                "cyber-post-train.fleet.ai/experiment": JOB_NAME,
                "cyber-post-train.fleet.ai/owner": "chris",
                "kueue.x-k8s.io/queue-name": "training-lq",
            },
            "annotations": {
                "cyber-post-train.fleet.ai/create-once": "true",
                "cyber-post-train.fleet.ai/launch-authorized": "false",
            },
        },
        "spec": {
            "backoffLimit": 0,
            "activeDeadlineSeconds": 120,
            "ttlSecondsAfterFinished": 604800,
            "template": {
                "metadata": {"labels": {"cyber-post-train.fleet.ai/experiment": JOB_NAME}},
                "spec": {
                    "restartPolicy": "Never",
                    "priorityClassName": "fleet-infra-quiet",
                    "preemptionPolicy": "Never",
                    "nodeSelector": {
                        "kubernetes.io/arch": "amd64",
                        "workload": "fleetai-training-ng-cpu",
                    },
                    "tolerations": [{
                        "key": "workload", "operator": "Equal",
                        "value": "fleetai-training-ng-cpu", "effect": "NoSchedule",
                    }],
                    "containers": [{
                        "name": "watchdog-canary",
                        "image": IMAGE,
                        "command": ["python", "/bootstrap/probe.py"],
                        "env": [{"name": "OUTPUT_ROOT", "value": OUTPUT_ROOT}],
                        "resources": {
                            "requests": {"cpu": "50m", "memory": "64Mi"},
                            "limits": {"cpu": "250m", "memory": "256Mi"},
                        },
                        "volumeMounts": [
                            {"name": "bootstrap", "mountPath": "/bootstrap", "readOnly": True},
                            {"name": "sfs", "mountPath": "/mnt/sfs"},
                        ],
                    }],
                    "volumes": [
                        {"name": "bootstrap", "configMap": {"name": CONFIGMAP_NAME}},
                        {"name": "sfs", "persistentVolumeClaim": {"claimName": "sfs-shared"}},
                    ],
                },
            },
        },
    }
    objects = {"apiVersion": "v1", "kind": "List", "items": [source, job]}
    body = {
        "schema_version": "fleet-glm53-dedicated-metric-watchdog-package-v1",
        "status": "READY_NON_SCORED",
        "launch_authorized": False,
        "job_name": JOB_NAME,
        "output_root": OUTPUT_ROOT,
        "lifecycle_file_sha256": "sha256:" + hashlib.sha256(lifecycle.read_bytes()).hexdigest(),
        "objects_sha256": self_hosted.sha256(self_hosted.canonical_json(objects)),
        "objects": objects,
    }
    body["package_sha256"] = self_hosted.digest_without(body, "package_sha256")
    return body


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preview", "render"))
    args = parser.parse_args()
    value = render(Path.cwd())
    if args.command == "preview":
        print(json.dumps({k: v for k, v in value.items() if k != "objects"}, sort_keys=True))
    else:
        print(yaml.safe_dump(value["objects"], sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
