"""Render create-once read-only preflight and scored Qwen G18 canary Jobs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from evals.fleet import qwen_hosted_generation18 as g18

NAMESPACE = "fleet-train-jobs"
UV_IMAGE = (
    "ghcr.io/astral-sh/uv:python3.12-bookworm@sha256:"
    "9aa60c50016c0485636ab9a830246a6ef3399aa4a8bab3d17ef4a2358fba2ca7"
)
DIND_IMAGE = (
    "docker.io/library/docker@sha256:"
    "f649ef046008ca7f926a2571c32b0ac22e5c59eb61b959617f9acc2a4c638cf5"
)
COMMON = (
    "evals/fleet/self_hosted.py",
    "evals/fleet/exact_pass4_bulk_v3.py",
    "evals/fleet/exact_pass4_bulk_runtime_v3.py",
    "evals/fleet/exact_pass4_universe.py",
    "evals/fleet/exact_pass4_crypto.py",
    "evals/fleet/endpoint_lease.py",
    "evals/fleet/qwen_hosted_generation18.py",
)


def _data(root: Path, paths: tuple[str, ...], plan: dict[str, Any]) -> dict[str, str]:
    data = {Path(path).name: (root / path).read_text() for path in paths}
    data["plan.json"] = json.dumps(plan, sort_keys=True, separators=(",", ":")) + "\n"
    data["parity.json"] = (root / g18.PARITY_PATH).read_text()
    return data


def _base_job(name: str, cm: str, *, scored: bool) -> dict[str, Any]:
    pod = {
        "restartPolicy": "Never",
        "priorityClassName": "fleet-serve-low",
        "preemptionPolicy": "Never",
        "nodeSelector": {"kubernetes.io/arch": "amd64", "workload": "fleetai-training-ng-cpu"},
        "tolerations": [
            {
                "key": "workload",
                "operator": "Equal",
                "value": "fleetai-training-ng-cpu",
                "effect": "NoSchedule",
            }
        ],
        "containers": [],
        "volumes": [
            {"name": "bootstrap", "configMap": {"name": cm}},
            {"name": "sfs", "persistentVolumeClaim": {"claimName": "sfs-shared"}},
        ],
    }
    env = [
        {
            "name": "FLEET_API_KEY",
            "valueFrom": {
                "secretKeyRef": {"name": g18.FLEET_API_KEY_SECRET, "key": "FLEET_API_KEY"}
            },
        },
        {
            "name": "JOB_UID",
            "valueFrom": {
                "fieldRef": {
                    "apiVersion": "v1",
                    "fieldPath": "metadata.labels['batch.kubernetes.io/controller-uid']",
                }
            },
        },
        {
            "name": "POD_UID",
            "valueFrom": {"fieldRef": {"apiVersion": "v1", "fieldPath": "metadata.uid"}},
        },
    ]
    if scored:
        pod["initContainers"] = [
            {
                "name": "dind",
                "image": DIND_IMAGE,
                "restartPolicy": "Always",
                "args": [
                    "--host=unix:///var/run/docker.sock",
                    "--host=tcp://0.0.0.0:2375",
                    "--tls=false",
                ],
                "readinessProbe": {
                    "tcpSocket": {"port": 2375},
                    "initialDelaySeconds": 2,
                    "periodSeconds": 2,
                },
                "securityContext": {"privileged": True},
                "resources": {
                    "requests": {"cpu": "500m", "memory": "2Gi", "ephemeral-storage": "20Gi"},
                    "limits": {"cpu": "2", "memory": "4Gi", "ephemeral-storage": "40Gi"},
                },
                "volumeMounts": [
                    {"name": "docker-socket", "mountPath": "/var/run"},
                    {"name": "docker-data", "mountPath": "/var/lib/docker"},
                    {"name": "workspace", "mountPath": "/workspace"},
                    {"name": "sfs", "mountPath": "/mnt/sfs"},
                ],
            }
        ]
        env.extend(
            [
                {"name": "DOCKER_HOST", "value": "unix:///var/run/docker.sock"},
                {"name": "DOCKER_TLS_CERTDIR", "value": ""},
                {"name": "G18_PREFLIGHT_PATH", "value": g18.PREFLIGHT_ROOT + "/CLEAR.json"},
                {"name": "G18_PREFLIGHT_SHA256", "value": "__PREFLIGHT_SHA256__"},
            ]
        )
        pod["volumes"].extend(
            [
                {"name": "docker-data", "emptyDir": {"sizeLimit": "40Gi"}},
                {"name": "docker-socket", "emptyDir": {}},
                {"name": "workspace", "emptyDir": {"sizeLimit": "5Gi"}},
            ]
        )
    mounts = [
        {"name": "bootstrap", "mountPath": "/bootstrap", "readOnly": True},
        {"name": "sfs", "mountPath": "/mnt/sfs"},
    ]
    if scored:
        mounts.extend(
            [
                {"name": "docker-socket", "mountPath": "/var/run"},
                {"name": "workspace", "mountPath": "/workspace"},
            ]
        )
    pod["containers"] = [
        {
            "name": "evaluator",
            "image": UV_IMAGE,
            "command": ["/bin/bash", "-ceu", "--"],
            "args": [
                (
                    "apt-get update; apt-get install --yes --no-install-recommends "
                    "docker.io=20.10.24+dfsg1-1+deb12u1+b6; "
                    if scored
                    else ""
                )
                + "exec /bin/bash /bootstrap/"
                + (
                    "run_qwen_hosted_generation18.sh"
                    if scored
                    else "run_qwen_hosted_generation18_preflight.sh"
                )
            ],
            "env": env,
            "resources": {
                "requests": {"cpu": "100m", "memory": "512Mi", "ephemeral-storage": "5Gi"},
                "limits": {"cpu": "1", "memory": "2Gi", "ephemeral-storage": "20Gi"},
            },
            "volumeMounts": mounts,
        }
    ]
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": name,
            "namespace": NAMESPACE,
            "labels": {
                "cyber-post-train.fleet.ai/owner": "chris",
                "cyber-post-train.fleet.ai/experiment": "q38-g18-hosted-canary",
                "kueue.x-k8s.io/queue-name": "training-lq",
            },
            "annotations": {
                "cyber-post-train.fleet.ai/create-once": "true",
                "cyber-post-train.fleet.ai/launch-authorized": "true" if scored else "false",
            },
        },
        "spec": {
            "activeDeadlineSeconds": 43200 if scored else 900,
            "backoffLimit": 0,
            "ttlSecondsAfterFinished": 604800,
            "template": {
                "metadata": {
                    "labels": {
                        "cyber-post-train.fleet.ai/owner": "chris",
                        "cyber-post-train.fleet.ai/experiment": "q38-g18-hosted-canary",
                    }
                },
                "spec": pod,
            },
        },
    }


def render(root: Path, preflight_sha256: str | None = None) -> dict[str, Any]:
    plan = g18.load(root / g18.PLAN_PATH)
    g18.validate_plan(plan)
    pre_paths = (
        *COMMON,
        "evals/fleet/qwen_hosted_generation18_preflight.py",
        "evals/fleet/scripts/run_qwen_hosted_generation18_preflight.sh",
    )
    scored_paths = (
        *COMMON,
        "evals/fleet/opencode_train_sweep_runner.py",
        "evals/fleet/qwen_hosted_generation18_runtime.py",
        "evals/fleet/fixed_proxy.py",
        "evals/fleet/Dockerfile.opencode",
        "evals/fleet/scripts/run_qwen_hosted_generation18.sh",
    )
    pre_cm = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": g18.PREFLIGHT_CONFIGMAP_NAME, "namespace": NAMESPACE},
        "data": _data(root, pre_paths, plan),
    }
    scored_cm = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": g18.CONFIGMAP_NAME, "namespace": NAMESPACE},
        "data": _data(root, scored_paths, plan),
    }
    for cm in (pre_cm, scored_cm):
        if len(json.dumps(cm).encode()) >= 900_000:
            raise ValueError("Generation-18 ConfigMap exceeds safety budget")
    pre_job = _base_job(g18.PREFLIGHT_JOB_NAME, g18.PREFLIGHT_CONFIGMAP_NAME, scored=False)
    scored_job = _base_job(g18.JOB_NAME, g18.CONFIGMAP_NAME, scored=True)
    if preflight_sha256 is not None:
        if g18.SHA256_RE.fullmatch(preflight_sha256) is None:
            raise ValueError("invalid Generation-18 preflight digest")
        for env in scored_job["spec"]["template"]["spec"]["containers"][0]["env"]:
            if env["name"] == "G18_PREFLIGHT_SHA256":
                env["value"] = preflight_sha256
    return {"apiVersion": "v1", "kind": "List", "items": [pre_cm, pre_job, scored_cm, scored_job]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-sha256")
    args = parser.parse_args()
    args.output.write_text(
        yaml.safe_dump(render(args.repo_root, args.preflight_sha256), sort_keys=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
