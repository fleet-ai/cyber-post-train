"""Immutable CPU/DinD package for the held GLM v22 score-free concurrency ramp."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

from evals.fleet import self_hosted

SCHEMA = "fleet-glm53-dedicated-v22-concurrency-package-v1"
NAMESPACE = "fleet-train-jobs"
JOB_NAME = "chris-glm53-dedicated-v22-concurrency-qualification-v1"
CONFIGMAP_NAME = JOB_NAME + "-package"
AUTHORIZATION_CONFIGMAP_NAME = JOB_NAME + "-authorization"
OUTPUT_ROOT = f"/mnt/sfs/jobs/{JOB_NAME}"
UV_IMAGE = (
    "ghcr.io/astral-sh/uv:python3.12-bookworm@sha256:"
    "9aa60c50016c0485636ab9a830246a6ef3399aa4a8bab3d17ef4a2358fba2ca7"
)
DIND_IMAGE = (
    "docker.io/library/docker@sha256:"
    "f649ef046008ca7f926a2571c32b0ac22e5c59eb61b959617f9acc2a4c638cf5"
)
COMMIT_RE = re.compile(r"[0-9a-f]{40}")
FILES = (
    "evals/fleet/glm53_dedicated_v22_concurrency_qualification_v1.py",
    "evals/fleet/glm53_dedicated_v22_concurrency_authorization_v1.py",
    "evals/fleet/opencode_actual_harness_parity_v1.py",
    "evals/fleet/production_blackbox_tool_catalog_v1.py",
    "evals/fleet/configs/blackbox-ctf-tool-catalog-v1.json",
    "evals/fleet/exact_pass4_crypto.py",
    "evals/fleet/exact_pass4_universe.py",
    "evals/fleet/endpoint_lease.py",
    "evals/fleet/self_hosted.py",
    "evals/fleet/Dockerfile.opencode",
    "docs/evidence/glm53-study/2026-09-06-glm53-dedicated-v22-server-binding.json",
    "docs/evidence/glm53-study/2026-09-06-glm53-dedicated-v22-actual-opencode-parity.json",
)
RUN = "evals/fleet/scripts/run_glm53_dedicated_v22_concurrency_qualification_v1.sh"


class PackageError(RuntimeError):
    pass


def _source(root: Path, commit: str, path: str) -> bytes:
    if COMMIT_RE.fullmatch(commit) is None:
        raise PackageError("package_commit_invalid")
    try:
        return subprocess.run(
            ["git", "-C", str(root), "show", f"{commit}:{path}"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PackageError(f"package_source_absent:{path}") from exc


def _key(path: str) -> str:
    return path.replace("/", "__SLASH__")


def build_configmap(root: Path, commit: str) -> dict[str, Any]:
    data: dict[str, str] = {}
    manifest: dict[str, str] = {}
    for path in FILES:
        raw = _source(root, commit, path)
        data[_key(path)] = raw.decode()
        manifest[path] = self_hosted.sha256(raw)
    run = _source(root, commit, RUN)
    data["run.sh"] = run.decode()
    manifest[RUN] = self_hosted.sha256(run)
    package = {
        "schema_version": SCHEMA,
        "package_commit": commit,
        "files": manifest,
        "job_name": JOB_NAME,
        "output_root": OUTPUT_ROOT,
        "concurrency_waves": [1, 2, 4],
        "score_free": True,
        "launch_authorized": False,
    }
    package["package_sha256"] = self_hosted.digest_without(package, "package_sha256")
    data["package.json"] = json.dumps(package, sort_keys=True, separators=(",", ":")) + "\n"
    data["package_commit"] = commit
    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": CONFIGMAP_NAME, "namespace": NAMESPACE},
        "immutable": True,
        "data": data,
    }


def build_authorization_configmap(authorization: dict[str, Any]) -> dict[str, Any]:
    if (
        authorization.get("qualification_launch_authorized") is not True
        or authorization.get("scored_successor_launch_authorized") is not False
        or authorization.get("receipt_sha256")
        != self_hosted.digest_without(authorization, "receipt_sha256")
    ):
        raise PackageError("authorization_invalid")
    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": AUTHORIZATION_CONFIGMAP_NAME, "namespace": NAMESPACE},
        "immutable": True,
        "data": {
            "authorization.json": json.dumps(authorization, sort_keys=True, separators=(",", ":"))
            + "\n"
        },
    }


def build_job(configmap: dict[str, Any], authorization: dict[str, Any]) -> dict[str, Any]:
    package = json.loads(configmap["data"]["package.json"])
    build_authorization_configmap(authorization)
    common_mounts = [
        {"name": "docker-socket", "mountPath": "/var/run"},
        {"name": "docker-data", "mountPath": "/var/lib/docker"},
        {"name": "workspace", "mountPath": "/workspace"},
        {"name": "sfs", "mountPath": "/mnt/sfs"},
    ]
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": JOB_NAME,
            "namespace": NAMESPACE,
            "labels": {
                "cyber-post-train.fleet.ai/owner": "chris",
                "cyber-post-train.fleet.ai/experiment": JOB_NAME,
                "kueue.x-k8s.io/queue-name": "training-lq",
            },
            "annotations": {
                "cyber-post-train.fleet.ai/create-once": "true",
                "cyber-post-train.fleet.ai/launch-authorized": "true",
                "cyber-post-train.fleet.ai/score-free": "true",
                "cyber-post-train.fleet.ai/authorization-receipt-sha256": authorization[
                    "receipt_sha256"
                ],
            },
        },
        "spec": {
            "backoffLimit": 0,
            "activeDeadlineSeconds": 7200,
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
                    "tolerations": [
                        {
                            "key": "workload",
                            "operator": "Equal",
                            "value": "fleetai-training-ng-cpu",
                            "effect": "NoSchedule",
                        }
                    ],
                    "initContainers": [
                        {
                            "name": "docker-cli",
                            "image": DIND_IMAGE,
                            "command": ["sh", "-ec"],
                            "args": ["install -D -m 0755 /usr/local/bin/docker /cli/docker"],
                            "volumeMounts": [{"name": "docker-cli", "mountPath": "/cli"}],
                        },
                        {
                            "name": "dind",
                            "image": DIND_IMAGE,
                            "restartPolicy": "Always",
                            "args": ["--host=unix:///var/run/docker.sock", "--tls=false"],
                            "securityContext": {"privileged": True},
                            "resources": {
                                "requests": {
                                    "cpu": "500m",
                                    "memory": "2Gi",
                                    "ephemeral-storage": "20Gi",
                                },
                                "limits": {
                                    "cpu": "2",
                                    "memory": "4Gi",
                                    "ephemeral-storage": "40Gi",
                                },
                            },
                            "volumeMounts": common_mounts,
                        },
                    ],
                    "containers": [
                        {
                            "name": "qualifier",
                            "image": UV_IMAGE,
                            "command": ["bash", "/bootstrap/run.sh"],
                            "env": [
                                {
                                    "name": "PATH",
                                    "value": "/docker-cli:/usr/local/bin:/usr/bin:/bin",
                                },
                                {"name": "DOCKER_HOST", "value": "unix:///var/run/docker.sock"},
                                {"name": "DOCKER_TLS_CERTDIR", "value": ""},
                                {
                                    "name": "QUALIFICATION_PACKAGE_COMMIT",
                                    "value": package["package_commit"],
                                },
                                {"name": "QUALIFICATION_OUTPUT_ROOT", "value": OUTPUT_ROOT},
                            ],
                            "resources": {
                                "requests": {
                                    "cpu": "4",
                                    "memory": "8Gi",
                                    "ephemeral-storage": "20Gi",
                                },
                                "limits": {
                                    "cpu": "8",
                                    "memory": "16Gi",
                                    "ephemeral-storage": "40Gi",
                                },
                            },
                            "volumeMounts": [
                                {"name": "docker-socket", "mountPath": "/var/run"},
                                {"name": "docker-cli", "mountPath": "/docker-cli"},
                                {"name": "workspace", "mountPath": "/workspace"},
                                {"name": "sfs", "mountPath": "/mnt/sfs"},
                                {"name": "package", "mountPath": "/bootstrap", "readOnly": True},
                                {
                                    "name": "authorization",
                                    "mountPath": "/authorization",
                                    "readOnly": True,
                                },
                            ],
                        }
                    ],
                    "volumes": [
                        {"name": "docker-socket", "emptyDir": {}},
                        {"name": "docker-cli", "emptyDir": {}},
                        {"name": "docker-data", "emptyDir": {}},
                        {"name": "workspace", "emptyDir": {}},
                        {"name": "sfs", "persistentVolumeClaim": {"claimName": "sfs-shared"}},
                        {"name": "package", "configMap": {"name": CONFIGMAP_NAME}},
                        {
                            "name": "authorization",
                            "configMap": {"name": AUTHORIZATION_CONFIGMAP_NAME},
                        },
                    ],
                },
            },
        },
    }
