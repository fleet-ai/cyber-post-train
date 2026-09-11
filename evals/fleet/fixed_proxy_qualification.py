"""Render a zero-GPU dev Pod that qualifies the exact fixed proxy offline.

The helper clean-pulls a digest-pinned Python image and exercises the current
``fixed_proxy.py`` bytes against a loopback-only synthetic upstream.  It never
contacts a model, Fleet task, grader, or scoring service.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from training.io import canonical_json, file_sha256

SCHEMA = "cyber_fleet_fixed_proxy_cpu_qualification_v1"


def _image(value: str) -> str:
    if re.fullmatch(r"[^\s]+@sha256:[0-9a-f]{64}", value) is None:
        raise ValueError("proxy helper image must be digest pinned")
    return value


def _runner() -> str:
    return r'''
import base64
import hashlib
import http.client
import json
import os
import threading
import types
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

def canonical(value):
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)

def digest(value):
    return "sha256:" + hashlib.sha256(canonical(value).encode()).hexdigest()

config = json.loads(base64.b64decode(os.environ["QUALIFICATION_CONFIG_B64"], validate=True))
source = base64.b64decode(os.environ["FIXED_PROXY_SOURCE_B64"], validate=True)
if "sha256:" + hashlib.sha256(source).hexdigest() != config["proxy_source_sha256"]:
    raise RuntimeError("fixed proxy source digest differs")
module = types.ModuleType("qualified_fixed_proxy")
exec(compile(source, "fixed_proxy.py", "exec"), module.__dict__)
received = []

class Upstream(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("content-length", "0"))
        received.append({
            "path": self.path,
            "auth": self.headers.get("Authorization"),
            "body": json.loads(self.rfile.read(length)),
        })
        body = b"{}"
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        return

def request(port, method, path, body=b"", headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        response.read()
        return response.status
    finally:
        connection.close()

upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
os.environ.update({
    "FIXED_UPSTREAM": f"http://127.0.0.1:{upstream.server_port}/synthetic",
    "FIXED_ALLOWED_PATHS": "/v1/chat/completions",
    "FIXED_MAX_REQUEST_BYTES": "1024",
    "FIXED_MAX_REQUESTS": "1",
    "FIXED_AUTH_HEADER": "Authorization",
    "FIXED_AUTH_VALUE": "synthetic-only",
    "FIXED_COMPLETION_JSON": canonical(config["sampling_policy"]),
})
module.Handler.request_count = 0
proxy = ThreadingHTTPServer(("127.0.0.1", 0), module.Handler)
threads = [
    threading.Thread(target=server.serve_forever, daemon=True)
    for server in (upstream, proxy)
]
for thread in threads:
    thread.start()
try:
    health = request(proxy.server_port, "GET", "/healthz")
    denied = request(proxy.server_port, "GET", "/forbidden")
    oversized = request(
        proxy.server_port,
        "POST",
        "/v1/chat/completions",
        b"x" * 1025,
        {"Content-Length": "1025"},
    )
    synthetic = canonical({
        "model": "agent-cannot-select",
        "temperature": 2,
        "top_p": 0.1,
        "seed": 7,
        "max_tokens": 1,
        "max_completion_tokens": 2,
    }).encode()
    forwarded = request(
        proxy.server_port,
        "POST",
        "/v1/chat/completions",
        synthetic,
        {"Authorization": "agent-placeholder", "Content-Type": "application/json"},
    )
    exhausted = request(proxy.server_port, "POST", "/v1/chat/completions", b"{}")
finally:
    for server in (proxy, upstream):
        server.shutdown()
        server.server_close()
    for thread in threads:
        thread.join(timeout=3)

expected = config["sampling_policy"]
checks = {
    "loopback_only": True,
    "health_endpoint": health == 200,
    "allowed_path_enforced": denied == 404,
    "request_size_enforced": oversized == 413,
    "request_count_enforced": exhausted == 429,
    "fixed_sampling_seed_and_32768_output": (
        forwarded == 200 and len(received) == 1 and received[0]["body"] == expected
    ),
    "synthetic_auth_replaced": (
        len(received) == 1 and received[0]["auth"] == "synthetic-only"
    ),
    "upstream_path_fixed": (
        len(received) == 1
        and received[0]["path"] == "/synthetic/v1/chat/completions"
    ),
}
if not all(checks.values()):
    raise RuntimeError("fixed proxy qualification check failed")
unsigned = {
    "schema": config["schema"],
    "status": "passed",
    "plan_sha256": config["plan_sha256"],
    "qualified_at": datetime.now(timezone.utc).isoformat(),
    "helper": {
        "pod": os.environ["POD_NAME"],
        "pod_uid": os.environ["POD_UID"],
        "node": os.environ["NODE_NAME"],
    },
    "requested_image": config["image"],
    "image_pull_policy": "Always",
    "platform": "linux/amd64",
    "proxy_source_sha256": config["proxy_source_sha256"],
    "sampling_policy_sha256": digest(expected),
    "checks": checks,
    "scope": {
        "gpu_requests": 0,
        "external_network_requests": 0,
        "model_requests": 0,
        "prompt_requests": 0,
        "completion_requests": 0,
        "scoring_requests": 0,
        "task_or_grading_requests": 0,
        "synthetic_loopback_requests": 5,
    },
}
print(canonical({**unsigned, "sha256": digest(unsigned)}), flush=True)
'''


def build_config(plan: dict[str, Any], *, image: str, source: Path) -> dict[str, Any]:
    parent = json.loads(
        (Path.cwd() / plan["references"]["parent_protocol"]["path"]).read_text(encoding="utf-8")
    )
    sampling = parent["sampling"]
    policy = {
        "model": plan["candidate_base_route"]["served_model_id"],
        "temperature": sampling["temperature"],
        "top_p": sampling["top_p"],
        "seed": sampling["base_seed"],
        "max_tokens": parent["harness"]["max_output_tokens"],
    }
    if policy["max_tokens"] != 32768:
        raise ValueError("fixed output treatment changed")
    return {
        "schema": SCHEMA,
        "plan_sha256": plan["sha256"],
        "image": _image(image),
        "proxy_source_sha256": file_sha256(source),
        "sampling_policy": policy,
    }


def build_pod(
    config: dict[str, Any],
    *,
    source: Path,
    name: str,
    namespace: str,
    node: str,
) -> dict[str, Any]:
    source_bytes = source.read_bytes()
    if config["proxy_source_sha256"] != "sha256:" + hashlib.sha256(source_bytes).hexdigest():
        raise ValueError("fixed proxy source changed after configuration")
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": name,
            "namespace": namespace,
            "labels": {
                "cyber-post-train.fleet.ai/owner": "chris",
                "cyber-post-train.fleet.ai/purpose": "fixed-proxy-cpu-qualification",
            },
            "annotations": {"cyber-post-train.fleet.ai/plan-sha256": config["plan_sha256"]},
        },
        "spec": {
            "activeDeadlineSeconds": 300,
            "automountServiceAccountToken": False,
            "enableServiceLinks": False,
            "nodeName": node,
            "restartPolicy": "Never",
            "terminationGracePeriodSeconds": 5,
            "containers": [
                {
                    "name": "qualification",
                    "image": config["image"],
                    "imagePullPolicy": "Always",
                    "command": ["python3", "-c", _runner()],
                    "env": [
                        {
                            "name": "QUALIFICATION_CONFIG_B64",
                            "value": base64.b64encode(canonical_json(config).encode()).decode(),
                        },
                        {
                            "name": "FIXED_PROXY_SOURCE_B64",
                            "value": base64.b64encode(source_bytes).decode(),
                        },
                        {"name": "PYTHONDONTWRITEBYTECODE", "value": "1"},
                        {"name": "HOME", "value": "/tmp"},
                        {
                            "name": "POD_NAME",
                            "valueFrom": {"fieldRef": {"fieldPath": "metadata.name"}},
                        },
                        {
                            "name": "POD_UID",
                            "valueFrom": {"fieldRef": {"fieldPath": "metadata.uid"}},
                        },
                        {
                            "name": "NODE_NAME",
                            "valueFrom": {"fieldRef": {"fieldPath": "spec.nodeName"}},
                        },
                    ],
                    "resources": {
                        "requests": {"cpu": "50m", "memory": "64Mi"},
                        "limits": {"cpu": "500m", "memory": "256Mi"},
                    },
                    "securityContext": {
                        "allowPrivilegeEscalation": False,
                        "capabilities": {"drop": ["ALL"]},
                        "readOnlyRootFilesystem": True,
                        "runAsGroup": 65532,
                        "runAsNonRoot": True,
                        "runAsUser": 65532,
                    },
                    "volumeMounts": [{"name": "tmp", "mountPath": "/tmp"}],
                }
            ],
            "volumes": [{"name": "tmp", "emptyDir": {"sizeLimit": "16Mi"}}],
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=Path)
    parser.add_argument("--image", required=True)
    parser.add_argument("--source", type=Path, default=Path(__file__).with_name("fixed_proxy.py"))
    parser.add_argument("--name", required=True)
    parser.add_argument("--namespace", default="default")
    parser.add_argument("--node", required=True)
    args = parser.parse_args(argv)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    config = build_config(plan, image=args.image, source=args.source)
    print(
        canonical_json(
            build_pod(
                config,
                source=args.source,
                name=args.name,
                namespace=args.namespace,
                node=args.node,
            )
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
