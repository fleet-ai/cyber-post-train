"""Render one zero-GPU read-only proof that a reward-signal output is absent."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

SCHEMA = "cyber_qwen38_miles96_signal_absence_probe_packet_v1"
RECEIPT_SCHEMA = "cyber_qwen38_miles96_signal_absence_receipt_v1"
LOG_PREFIX = "CYBER_MILES96_SIGNAL_ABSENCE="
CONTEXT = "nebius-mk8s-fleetai-training-e04zw4ye1k7wczqdw6"
NAMESPACE = "fleet-train-jobs"
JOB_NAME = "chris-q38-m96-signal-abs-a2"
CONFIG_MAP_NAME = JOB_NAME + "-code"
OUTPUT = Path("/mnt/sfs/jobs/chris-q38-m96-signal-a2")
IMAGE = (
    "ghcr.io/astral-sh/uv:python3.12-bookworm@"
    "sha256:9aa60c50016c0485636ab9a830246a6ef3399aa4a8bab3d17ef4a2358fba2ca7"
)
DRIVER = "from training.miles96_signal_absence_probe import runtime_main;runtime_main()\n"


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def observe() -> dict[str, Any]:
    try:
        OUTPUT.lstat()
    except FileNotFoundError:
        status = "absent"
        error_class = None
    except OSError as exc:
        status = "inaccessible"
        error_class = type(exc).__name__
    else:
        status = "present"
        error_class = None
    body = {
        "schema": RECEIPT_SCHEMA,
        "status": status,
        "output": str(OUTPUT),
        "error_class": error_class,
        "checked_at_epoch": int(time.time()),
        "gpus": 0,
        "read_only": True,
        "external_writes": 0,
        "weights_prompts_traces_flags_answers_scores_or_credentials_included": False,
    }
    return {**body, "sha256": "sha256:" + digest(body)}


def runtime_main() -> None:
    if os.getuid() != 1000 or os.getgid() != 100:
        raise ValueError("absence-probe identity drift")
    receipt = observe()
    print(LOG_PREFIX + json.dumps(receipt, sort_keys=True, separators=(",", ":")), flush=True)
    if receipt["status"] != "absent":
        raise SystemExit(1)


def build_packet() -> dict[str, Any]:
    sources = {
        "training_init.py": "",
        "probe.py": Path(__file__).read_text(),
        "driver.py": DRIVER,
    }
    labels = {
        "cyber-post-train.fleet.ai/owner": "chris",
        "cyber-post-train.fleet.ai/role": "miles96-signal-absence",
        "kueue.x-k8s.io/queue-name": "training-lq",
        "kueue.x-k8s.io/priority-class": "q1",
    }
    annotations = {
        "fleet.ai/failure-alerts": "off",
        "cyber-post-train.fleet.ai/create-once": "true",
    }
    config_map = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": CONFIG_MAP_NAME, "namespace": NAMESPACE},
        "immutable": True,
        "data": sources,
    }
    job = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": JOB_NAME,
            "namespace": NAMESPACE,
            "labels": labels,
            "annotations": annotations,
        },
        "spec": {
            "activeDeadlineSeconds": 600,
            "backoffLimit": 0,
            "completions": 1,
            "parallelism": 1,
            "suspend": True,
            "template": {
                "metadata": {"labels": labels, "annotations": annotations},
                "spec": {
                    "automountServiceAccountToken": False,
                    "containers": [
                        {
                            "name": "probe",
                            "image": IMAGE,
                            "command": [
                                "python",
                                "-I",
                                "-c",
                                "import runpy,sys;sys.path.insert(0,'/probe');"
                                "runpy.run_path('/probe/driver.py',run_name='__main__')",
                            ],
                            "env": [
                                {"name": "CUDA_VISIBLE_DEVICES", "value": ""},
                                {"name": "NVIDIA_VISIBLE_DEVICES", "value": "none"},
                            ],
                            "resources": {
                                "requests": {"cpu": "100m", "memory": "128Mi"},
                                "limits": {"cpu": "1", "memory": "1Gi"},
                            },
                            "securityContext": {
                                "allowPrivilegeEscalation": False,
                                "capabilities": {"drop": ["ALL"]},
                                "privileged": False,
                                "readOnlyRootFilesystem": True,
                            },
                            "volumeMounts": [
                                {"name": "code", "mountPath": "/probe", "readOnly": True},
                                {"name": "jobs", "mountPath": "/mnt/sfs/jobs", "readOnly": True},
                            ],
                        }
                    ],
                    "imagePullSecrets": [{"name": "ecr-pull"}],
                    "nodeSelector": {
                        "kubernetes.io/arch": "amd64",
                        "workload": "fleetai-training-ng-cpu",
                    },
                    "priorityClassName": "c1",
                    "restartPolicy": "Never",
                    "securityContext": {
                        "runAsNonRoot": True,
                        "runAsUser": 1000,
                        "runAsGroup": 100,
                        "fsGroup": 100,
                        "seccompProfile": {"type": "RuntimeDefault"},
                    },
                    "tolerations": [
                        {
                            "key": "workload",
                            "operator": "Equal",
                            "value": "fleetai-training-ng-cpu",
                            "effect": "NoSchedule",
                        }
                    ],
                    "volumes": [
                        {
                            "name": "code",
                            "configMap": {
                                "name": CONFIG_MAP_NAME,
                                "items": [
                                    {"key": "training_init.py", "path": "training/__init__.py"},
                                    {
                                        "key": "probe.py",
                                        "path": "training/miles96_signal_absence_probe.py",
                                    },
                                    {"key": "driver.py", "path": "driver.py"},
                                ],
                            },
                        },
                        {
                            "name": "jobs",
                            "persistentVolumeClaim": {"claimName": "sfs-shared", "readOnly": True},
                        },
                    ],
                },
            },
        },
    }
    body = {
        "schema": SCHEMA,
        "context": CONTEXT,
        "namespace": NAMESPACE,
        "job_name": JOB_NAME,
        "config_map_name": CONFIG_MAP_NAME,
        "output": str(OUTPUT),
        "expected": {
            "status": "absent",
            "gpus": 0,
            "priority_class": "c1",
            "queue_priority": "q1",
            "failure_alerts": "off",
            "backoff_limit": 0,
            "read_only": True,
        },
        "execution_sequence": {
            "config_map_create_request_count": 1,
            "job_create_request_count": 1,
            "create_retries_allowed": False,
            "controller_managed_unsuspend": True,
            "operator_patch_request_count": 0,
            "exact_uid_monitor_and_cleanup_required": True,
        },
        "bundle": {"apiVersion": "v1", "kind": "List", "items": [config_map, job]},
    }
    return {**body, "sha256": "sha256:" + digest(body)}


def validate_packet(packet: dict[str, Any]) -> dict[str, Any]:
    if packet != build_packet():
        raise ValueError("Miles96 signal-absence packet differs from renderer")
    config_map, job = packet["bundle"]["items"]
    pod = job["spec"]["template"]["spec"]
    if (
        config_map.get("immutable") is not True
        or job["metadata"]["annotations"].get("fleet.ai/failure-alerts") != "off"
        or job["spec"].get("backoffLimit") != 0
        or job["spec"].get("suspend") is not True
        or pod.get("priorityClassName") != "c1"
        or "nvidia.com/gpu" in json.dumps(pod)
        or pod["containers"][0]["volumeMounts"][1].get("readOnly") is not True
    ):
        raise ValueError("Miles96 signal-absence safety contract drifted")
    return {
        "packet_sha256": packet["sha256"],
        "bundle_sha256": "sha256:" + digest(packet["bundle"]),
        "job": JOB_NAME,
        "config_map": CONFIG_MAP_NAME,
        "output": str(OUTPUT),
        "gpus": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    packet = build_packet()
    validate_packet(packet)
    encoded = json.dumps(packet, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            stream.write(encoded)


if __name__ == "__main__":
    main()
