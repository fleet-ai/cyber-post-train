"""Render a UID-bound lifecycle heartbeat for the active v15 scored canary."""

from __future__ import annotations

import argparse
import json
import textwrap
import uuid
from pathlib import Path
from typing import Any

import yaml

from evals.fleet import glm53_dedicated_v14_scored_canary_v1 as canary
from evals.fleet import glm53_dedicated_v15 as server
from evals.fleet import self_hosted

JOB_NAME = "chris-glm53-dedicated-v15-r051-heartbeat-v1"
OUTPUT_ROOT = f"/mnt/sfs/jobs/{JOB_NAME}"
HEARTBEAT = f"{server.RUN_DIR}/lifecycle/traffic-stream-1"
IMAGE = (
    "ghcr.io/astral-sh/uv:python3.12-bookworm@"
    "sha256:9aa60c50016c0485636ab9a830246a6ef3399aa4a8bab3d17ef4a2358fba2ca7"
)


def render(*, canary_job_uid: str, server_rayjob_uid: str) -> dict[str, Any]:
    for value in (canary_job_uid, server_rayjob_uid):
        if uuid.UUID(value).int == 0:
            raise ValueError("heartbeat requires nonzero bound UIDs")
    script = textwrap.dedent(
        f"""
        import hashlib, json, os, pathlib, ssl, time, urllib.request

        namespace = "fleet-train-jobs"
        canary_name = "{canary.JOB_NAME}"
        canary_uid = "{canary_job_uid}"
        server_name = "ft-run-16335b81"
        server_uid = "{server_rayjob_uid}"
        heartbeat = pathlib.Path("{HEARTBEAT}")
        output = pathlib.Path("{OUTPUT_ROOT}")
        token = pathlib.Path("/var/run/secrets/kubernetes.io/serviceaccount/token").read_text().strip()
        ca = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"
        context = ssl.create_default_context(cafile=ca)
        headers = {{"Authorization": "Bearer " + token}}

        def get(path):
            request = urllib.request.Request("https://kubernetes.default.svc" + path, headers=headers)
            with urllib.request.urlopen(request, context=context, timeout=20) as response:
                return json.load(response)

        def canonical(value):
            return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()

        if output.exists() or heartbeat.is_symlink():
            raise RuntimeError("heartbeat create-once gate failed")
        output.mkdir(mode=0o700, parents=True)
        touches = 0
        started = time.time()
        terminal = "unknown"
        while time.time() - started < 28800:
            server_row = get("/apis/ray.io/v1/namespaces/" + namespace + "/rayjobs/" + server_name)
            if server_row["metadata"]["uid"] != server_uid or server_row.get("status", {{}}).get("jobStatus") != "RUNNING":
                raise RuntimeError("bound v15 server is not running")
            job = get("/apis/batch/v1/namespaces/" + namespace + "/jobs/" + canary_name)
            if job["metadata"]["uid"] != canary_uid:
                raise RuntimeError("bound canary Job UID drifted")
            status = job.get("status") or {{}}
            if status.get("active") == 1:
                heartbeat.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                heartbeat.touch(exist_ok=True)
                touches += 1
                time.sleep(20)
                continue
            if status.get("succeeded") == 1:
                terminal = "succeeded"
                break
            if status.get("failed") == 1:
                terminal = "failed"
                break
            time.sleep(5)
        body = {{
            "schema_version": "fleet-glm53-dedicated-v15-canary-heartbeat-v1",
            "status": "TERMINAL",
            "canary_job_name": canary_name,
            "canary_job_uid": canary_uid,
            "server_api_run_id": server_name,
            "server_rayjob_uid": server_uid,
            "heartbeat_path": str(heartbeat),
            "heartbeat_touches": touches,
            "canary_terminal": terminal,
            "scores_read": False,
            "prompts_traces_flags_read": False,
        }}
        body["receipt_sha256"] = "sha256:" + hashlib.sha256(canonical(body)).hexdigest()
        target = output / "TERMINAL.json"
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(canonical(body) + b"\\n")
        """
    ).strip()
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
                "cyber-post-train.fleet.ai/launch-authorized": "true",
            },
        },
        "spec": {
            "backoffLimit": 0,
            "activeDeadlineSeconds": 29000,
            "ttlSecondsAfterFinished": 604800,
            "template": {
                "metadata": {
                    "labels": {
                        "cyber-post-train.fleet.ai/experiment": JOB_NAME,
                        "cyber-post-train.fleet.ai/owner": "chris",
                    }
                },
                "spec": {
                    "restartPolicy": "Never",
                    "serviceAccountName": "default",
                    "priorityClassName": "fleet-infra-quiet",
                    "preemptionPolicy": "Never",
                    "nodeSelector": {
                        "kubernetes.io/arch": "amd64",
                        "workload": "fleetai-training-ng-cpu",
                    },
                    "tolerations": [
                        {
                            "key": "workload",
                            "operator": "Equal",
                            "value": "fleetai-training-ng-cpu",
                            "effect": "NoSchedule",
                        }
                    ],
                    "containers": [
                        {
                            "name": "heartbeat",
                            "image": IMAGE,
                            "command": ["python", "-c", script],
                            "resources": {
                                "requests": {"cpu": "20m", "memory": "64Mi"},
                                "limits": {"cpu": "100m", "memory": "128Mi"},
                            },
                            "volumeMounts": [
                                {"name": "sfs", "mountPath": "/mnt/sfs"}
                            ],
                        }
                    ],
                    "volumes": [
                        {"name": "sfs", "persistentVolumeClaim": {"claimName": "sfs-shared"}}
                    ],
                },
            },
        },
    }
    body = {
        "schema_version": "fleet-glm53-dedicated-v15-canary-heartbeat-package-v1",
        "status": "READY",
        "launch_authorized": True,
        "canary_job_uid": canary_job_uid,
        "server_rayjob_uid": server_rayjob_uid,
        "heartbeat_path": HEARTBEAT,
        "object": job,
    }
    body["package_sha256"] = self_hosted.digest_without(body, "package_sha256")
    return body


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canary-job-uid", required=True)
    parser.add_argument("--server-rayjob-uid", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error("output must be unused")
    value = render(
        canary_job_uid=args.canary_job_uid,
        server_rayjob_uid=args.server_rayjob_uid,
    )
    args.output.write_text(yaml.safe_dump(value["object"], sort_keys=False))
    print(json.dumps({key: item for key, item in value.items() if key != "object"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
