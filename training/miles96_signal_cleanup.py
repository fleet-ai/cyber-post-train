"""Reconcile exact Fleet instances after the bounded Miles96 signal job failed."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ORCHESTRATOR = "https://orchestrator.fleetai.com"
SOURCE = Path("/mnt/sfs/jobs/chris-q38-m96-signal-a2")
RECEIPT = SOURCE / "LEAK_RECONCILED.json"
SOURCE_JOB_UID = "d3388003-61ce-4b5d-95cf-5b85d7e6ac8b"
TASK_VERSION_ID = "0920e798-c7e7-4da6-9d5e-ebeba45ec05a"
VERIFIER_VERSION_ID = "9356b7ca-43b4-4926-a871-d9a95b41f6e5"
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
SERVED_MODEL = "qwen/chris-q38-base-pass4-v1"
JOB_NAME = "chris-q38-m96-signal-a2-leak-a5"
CM_NAME = JOB_NAME + "-code"
NAMESPACE = "fleet-train-jobs"
IMAGE = (
    "ghcr.io/astral-sh/uv:python3.12-bookworm@sha256:"
    "9aa60c50016c0485636ab9a830246a6ef3399aa4a8bab3d17ef4a2358fba2ca7"
)


def digest(value: Any) -> str:
    return (
        "sha256:"
        + hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
    )


def _read(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("required private lifecycle receipt is absent")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("private lifecycle receipt is invalid")
    return value


def _instance_identifier(value: Any) -> str:
    """Apply Fleet's canonical opaque, DNS-safe environment-ID contract."""
    if not isinstance(value, str) or not re.fullmatch(
        r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", value
    ):
        raise ValueError("Fleet authoritative instance ID is not a DNS-safe identifier")
    return value


def _request(instance_id: str, method: str) -> tuple[int, dict[str, Any] | None]:
    key = os.environ.get("FLEET_API_KEY")
    if not key:
        raise ValueError("FLEET_API_KEY is required")
    request = urllib.request.Request(
        f"{ORCHESTRATOR}/v1/env/instances/{instance_id}",
        method=method,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            raw = response.read()
            value = json.loads(raw) if raw else None
            return response.status, value if isinstance(value, dict) else None
    except urllib.error.HTTPError as error:
        error.read()
        return error.code, None


def _write_once(path: Path, value: dict[str, Any]) -> None:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _release_state(instance_id: str, runtime: dict[str, Any]) -> str:
    """Return absent, terminated, or live for one exact owned instance."""
    status, value = _request(instance_id, "GET")
    if status == 404:
        return "absent"
    if status != 200 or value is None:
        raise RuntimeError("one exact instance status is unreadable")
    if value.get("instance_id") != instance_id or any(
        value.get(field) != runtime.get(runtime_field)
        for field, runtime_field in (
            ("env_key", "env_key"),
            ("version", "environment_version"),
            ("data_key", "data_key"),
            ("data_version", "data_version"),
        )
    ):
        raise ValueError("one instance differs from its exact runtime binding")
    terminated_at = value.get("terminated_at")
    if terminated_at is None:
        return "live"
    if not isinstance(terminated_at, str) or not terminated_at.strip():
        raise ValueError("one instance has invalid termination evidence")
    return "terminated"


def reconcile(source: Path = SOURCE) -> dict[str, Any]:
    receipt_path = source / RECEIPT.name
    if source.is_symlink() or not source.is_dir() or receipt_path.exists():
        raise ValueError("source output is absent, unsafe, or already reconciled")
    attempts = sorted((source / "attempts").iterdir())
    if len(attempts) != 8 or any(path.is_symlink() or not path.is_dir() for path in attempts):
        raise ValueError("expected exactly eight private attempt directories")
    rows: list[tuple[str, dict[str, Any]]] = []
    cleanup_count = 0
    for attempt in attempts:
        binding = _read(attempt / "binding.json")
        runtime = _read(attempt / "runtime-binding.json")
        instance_id = _instance_identifier(runtime.get("instance_id"))
        if (
            binding.get("task", {}).get("version_id") != TASK_VERSION_ID
            or binding.get("verifier", {}).get("version_id") != VERIFIER_VERSION_ID
            or binding.get("model", {}).get("revision") != MODEL_REVISION
            or binding.get("model", {}).get("session_model") != SERVED_MODEL
        ):
            raise ValueError("one private attempt binding differs")
        cleanup = attempt / "cleanup.json"
        if cleanup.exists():
            if _read(cleanup) != {
                "instance_created": True,
                "instance_closed": True,
                "containers_removed": True,
            }:
                raise ValueError("one existing cleanup receipt is incomplete")
            cleanup_count += 1
        rows.append((instance_id, runtime))
    if len({row[0] for row in rows}) != 8:
        raise ValueError("private instance roster is not unique")
    if cleanup_count not in {7, 8}:
        raise ValueError("private cleanup count is outside the reviewed seven-or-eight bound")

    states = [
        (instance_id, runtime, _release_state(instance_id, runtime))
        for instance_id, runtime in rows
    ]
    live = [(instance_id, runtime) for instance_id, runtime, state in states if state == "live"]
    if len(live) > 1:
        raise RuntimeError("more than one exact owned instance remains live")

    deleted = False
    deleted_digest = None
    if live:
        instance_id, runtime = live[0]
        status, value = _request(instance_id, "DELETE")
        if (
            status != 200
            or value is None
            or value.get("instance_id") != instance_id
            or any(
                value.get(field) != runtime.get(runtime_field)
                for field, runtime_field in (
                    ("env_key", "env_key"),
                    ("version", "environment_version"),
                    ("data_key", "data_key"),
                    ("data_version", "data_version"),
                )
            )
            or not isinstance(value.get("terminated_at"), str)
            or not value["terminated_at"].strip()
        ):
            raise RuntimeError("exact instance release failed")
        deleted = True
        deleted_digest = digest(instance_id)

    final_states: list[str] = []
    for attempt in range(6):
        final_states = [_release_state(instance_id, runtime) for instance_id, runtime in rows]
        if "live" not in final_states:
            break
        if attempt == 5:
            raise RuntimeError("one or more exact instances remain live after release")
        time.sleep(2**attempt)

    result_file_count = sum((attempt / "reward-result.json").is_file() for attempt in attempts)
    accepted_file_count = sum((attempt / "ACCEPTED.json").is_file() for attempt in attempts)
    signal_validated_present = (source / "SIGNAL_VALIDATED.json").is_file()

    body = {
        "schema": "cyber_qwen38_miles96_signal_leak_reconciliation_v3",
        "status": "all_exact_instances_released",
        "source_job_uid": SOURCE_JOB_UID,
        "attempt_count": 8,
        "reward_result_file_count": result_file_count,
        "preexisting_cleanup_receipt_count": cleanup_count,
        "accepted_receipt_count": accepted_file_count,
        "signal_validated_present": signal_validated_present,
        "reward_terminal_metadata_read": False,
        "live_instance_count_before": len(live),
        "exact_delete_attempted": deleted,
        "deleted_instance_identity_sha256": deleted_digest,
        "absent_instance_count_after": final_states.count("absent"),
        "terminated_instance_count_after": final_states.count("terminated"),
        "live_instance_count_after": final_states.count("live"),
        "all_instances_released_after": final_states.count("live") == 0,
        "prompts_traces_rewards_included": False,
    }
    receipt = {**body, "sha256": digest(body)}
    _write_once(receipt_path, receipt)
    return receipt


def packet() -> dict[str, Any]:
    source = Path(__file__).read_text()
    bundle = {
        "apiVersion": "v1",
        "kind": "List",
        "items": [
            {
                "apiVersion": "v1",
                "kind": "ConfigMap",
                "immutable": True,
                "metadata": {"name": CM_NAME, "namespace": NAMESPACE},
                "data": {"cleanup.py": source},
            },
            {
                "apiVersion": "batch/v1",
                "kind": "Job",
                "metadata": {
                    "name": JOB_NAME,
                    "namespace": NAMESPACE,
                    "annotations": {"fleet.ai/failure-alerts": "off"},
                    "labels": {
                        "cyber-post-train.fleet.ai/owner": "chris",
                        "cyber-post-train.fleet.ai/role": "miles96-signal-leak-reconcile",
                        "kueue.x-k8s.io/queue-name": "training-lq",
                        "kueue.x-k8s.io/priority-class": "q1",
                    },
                },
                "spec": {
                    "suspend": True,
                    "backoffLimit": 0,
                    "activeDeadlineSeconds": 900,
                    "template": {
                        "metadata": {"annotations": {"fleet.ai/failure-alerts": "off"}},
                        "spec": {
                            "restartPolicy": "Never",
                            "priorityClassName": "c1",
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
                                    "name": "reconcile",
                                    "image": IMAGE,
                                    "command": ["python", "/bootstrap/cleanup.py"],
                                    "env": [
                                        {
                                            "name": "FLEET_API_KEY",
                                            "valueFrom": {
                                                "secretKeyRef": {
                                                    "name": "chris-cyber-opencode-evals-v2",
                                                    "key": "FLEET_API_KEY",
                                                }
                                            },
                                        }
                                    ],
                                    "resources": {
                                        "requests": {"cpu": "1", "memory": "2Gi"},
                                        "limits": {"cpu": "2", "memory": "4Gi"},
                                    },
                                    "volumeMounts": [
                                        {
                                            "name": "bootstrap",
                                            "mountPath": "/bootstrap",
                                            "readOnly": True,
                                        },
                                        {"name": "sfs", "mountPath": "/mnt/sfs"},
                                    ],
                                }
                            ],
                            "volumes": [
                                {"name": "bootstrap", "configMap": {"name": CM_NAME}},
                                {
                                    "name": "sfs",
                                    "persistentVolumeClaim": {"claimName": "sfs-shared"},
                                },
                            ],
                        },
                    },
                },
            },
        ],
    }
    body = {
        "schema": "cyber_qwen38_miles96_signal_leak_reconcile_packet_v4",
        "job_name": JOB_NAME,
        "config_map_name": CM_NAME,
        "source_job_uid": SOURCE_JOB_UID,
        "bundle": bundle,
        "create_counts": {"config_map": 1, "job": 1, "retry": 0, "patch": 0},
        "expected": {
            "gpus": 0,
            "priority_class": "c1",
            "queue_priority": "q1",
            "failure_alerts": "off",
        },
    }
    return {**body, "sha256": digest(body)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", action="store_true")
    args = parser.parse_args()
    value = packet() if args.packet else reconcile()
    print(json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    main()
