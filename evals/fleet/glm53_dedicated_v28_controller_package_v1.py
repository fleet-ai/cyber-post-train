"""Immutable package renderer for the held credentialed GLM v28 controller."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

from evals.fleet import exact_pass4_crypto as crypto
from evals.fleet import glm53_dedicated_v23_scorefree_package_v1 as prior
from evals.fleet import glm53_dedicated_v28_controller_v1 as controller
from evals.fleet import glm53_dedicated_v28_create_v1 as server

JOB_NAME = "chris-glm53-v28-create-watchdog-controller-v1"
CONFIGMAP_NAME = JOB_NAME + "-package"
AUTHORIZATION_CONFIGMAP_NAME = JOB_NAME + "-authorization"
FLEET_SECRET_NAME = "chris-cyber-opencode-evals-v2"
SFS_PVC_NAME = "sfs-shared"
CPU_IMAGE = (
    "ghcr.io/astral-sh/uv:python3.12-bookworm@"
    "sha256:9aa60c50016c0485636ab9a830246a6ef3399aa4a8bab3d17ef4a2358fba2ca7"
)
KUBECTL_IMAGE = (
    "docker.io/bitnami/kubectl@"
    "sha256:7fc66a99e38500a5ceb81583856f89ee589bdffd885c895e42a76dce45a3bc73"
)
COMMIT_RE = re.compile(r"[0-9a-f]{40}")
FILES = tuple(
    sorted(
        set(prior.FILES)
        | {
            "evals/fleet/glm53_dedicated_v24_server_v1.py",
            "evals/fleet/glm53_dedicated_v24_watchdog_live_release_v1.py",
            "evals/fleet/glm53_dedicated_v24_watchdog_package_v1.py",
            "evals/fleet/glm53_dedicated_v25_create_v1.py",
            "evals/fleet/glm53_dedicated_v26_create_v1.py",
            "evals/fleet/glm53_dedicated_v27_create_v1.py",
            "evals/fleet/glm53_dedicated_v28_create_v1.py",
            "evals/fleet/glm53_dedicated_v28_controller_v1.py",
            "evals/fleet/glm53_dedicated_v28_watchdog_live_release_v1.py",
            "evals/fleet/glm53_dedicated_v28_watchdog_package_v1.py",
            "pyproject.toml",
            "uv.lock",
        }
    )
)


class PackageError(RuntimeError):
    """The v28 controller package is invalid."""


def _source(root: Path, commit: str, path: str) -> bytes:
    if COMMIT_RE.fullmatch(commit) is None:
        raise PackageError("v28_package_commit_invalid")
    try:
        return subprocess.run(
            ["git", "-C", str(root), "show", f"{commit}:{path}"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PackageError(f"v28_package_source_absent:{path}") from exc


def _key(path: str) -> str:
    return path.replace("/", "__SLASH__")


def build_source_configmap(root: Path, commit: str) -> dict[str, Any]:
    data: dict[str, str] = {}
    manifest: dict[str, str] = {}
    for path in FILES:
        raw = _source(root, commit, path)
        data[_key(path)] = raw.decode()
        manifest[path] = crypto.sha256(raw)
    package = {
        "schema_version": "fleet-glm53-dedicated-v28-controller-package-v1",
        "package_commit": commit,
        "files": manifest,
        "job_name": JOB_NAME,
        "server_title": server.TITLE,
        "server_run_dir": server.RUN_DIR,
        "ready_timeout_seconds": controller.READY_TIMEOUT_SECONDS,
        "credentialed_release_required": True,
        "score_free": True,
    }
    package["package_sha256"] = crypto.digest_without(package, "package_sha256")
    data["package.json"] = json.dumps(package, sort_keys=True, separators=(",", ":")) + "\n"
    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": CONFIGMAP_NAME, "namespace": "fleet-train-jobs"},
        "immutable": True,
        "data": data,
    }


def build_authorization_configmap(authorization: dict[str, Any]) -> dict[str, Any]:
    server.validate_authorization(authorization)
    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": AUTHORIZATION_CONFIGMAP_NAME,
            "namespace": "fleet-train-jobs",
        },
        "immutable": True,
        "data": {
            "authorization.json": json.dumps(
                authorization, sort_keys=True, separators=(",", ":")
            )
            + "\n"
        },
    }


def build_job(package_commit: str) -> dict[str, Any]:
    command = f"""
set -euo pipefail
mkdir -p /workspace
for source in /package/*__SLASH__*; do
  relative="${{source##*/}}"
  relative="${{relative//__SLASH__//}}"
  mkdir -p "/workspace/$(dirname "$relative")"
  cp "$source" "/workspace/$relative"
done
cp /package/pyproject.toml /workspace/pyproject.toml
cp /package/uv.lock /workspace/uv.lock
export PATH="/tools:$PATH"
export UV_PROJECT_ENVIRONMENT=/tmp/v28-env
uv sync --project /workspace --frozen --no-install-project
exec /tmp/v28-env/bin/python -m evals.fleet.glm53_dedicated_v28_controller_v1 \
  --root /workspace \
  --package-commit {package_commit} \
  --authorization /authorization/authorization.json
""".strip()
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": JOB_NAME,
            "namespace": "fleet-train-jobs",
            "labels": {
                "cyber-post-train.fleet.ai/owner": "chris",
                "cyber-post-train.fleet.ai/experiment": JOB_NAME,
                "kueue.x-k8s.io/queue-name": "training-lq",
            },
        },
        "spec": {
            "backoffLimit": 0,
            "activeDeadlineSeconds": 1200,
            "ttlSecondsAfterFinished": 604800,
            "template": {
                "metadata": {
                    "labels": {
                        "cyber-post-train.fleet.ai/owner": "chris",
                        "cyber-post-train.fleet.ai/experiment": JOB_NAME,
                        "kueue.x-k8s.io/queue-name": "training-lq",
                    }
                },
                "spec": {
                    "restartPolicy": "Never",
                    "priorityClassName": "fleet-serve-low",
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
                            "name": "install-kubectl",
                            "image": KUBECTL_IMAGE,
                            "resources": {
                                "requests": {"cpu": "50m", "memory": "64Mi"},
                                "limits": {"cpu": "500m", "memory": "256Mi"},
                            },
                            "command": [
                                "sh",
                                "-c",
                                'cp "$(command -v kubectl)" /tools/kubectl',
                            ],
                            "volumeMounts": [{"name": "tools", "mountPath": "/tools"}],
                        }
                    ],
                    "containers": [
                        {
                            "name": "controller",
                            "image": CPU_IMAGE,
                            "command": ["bash", "-lc", command],
                            "resources": {
                                "requests": {"cpu": "100m", "memory": "256Mi"},
                                "limits": {"cpu": "2", "memory": "2Gi"},
                            },
                            "env": [
                                {
                                    "name": "FLEET_API_KEY",
                                    "valueFrom": {
                                        "secretKeyRef": {
                                            "name": FLEET_SECRET_NAME,
                                            "key": "FLEET_API_KEY",
                                        }
                                    },
                                }
                            ],
                            "volumeMounts": [
                                {"name": "package", "mountPath": "/package", "readOnly": True},
                                {
                                    "name": "authorization",
                                    "mountPath": "/authorization",
                                    "readOnly": True,
                                },
                                {"name": "sfs", "mountPath": "/mnt/sfs"},
                                {"name": "tools", "mountPath": "/tools", "readOnly": True},
                            ],
                        }
                    ],
                    "volumes": [
                        {"name": "package", "configMap": {"name": CONFIGMAP_NAME}},
                        {
                            "name": "authorization",
                            "configMap": {"name": AUTHORIZATION_CONFIGMAP_NAME},
                        },
                        {"name": "sfs", "persistentVolumeClaim": {"claimName": SFS_PVC_NAME}},
                        {"name": "tools", "emptyDir": {}},
                    ],
                },
            },
        },
    }


def render(
    root: Path, commit: str, authorization: dict[str, Any]
) -> dict[str, Any]:
    return {
        "objects": {
            "apiVersion": "v1",
            "kind": "List",
            "items": [
                build_source_configmap(root, commit),
                build_authorization_configmap(authorization),
                build_job(commit),
            ],
        },
        "server_launch_authorized": True,
        "watchdog_handoff_required": True,
        "qualification_launch_authorized": False,
        "scored_launch_authorized": False,
    }


def build_held() -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema_version": "fleet-glm53-dedicated-v28-controller-held-v1",
        "status": "PASSED_HELD_NO_LAUNCH",
        "job_name": JOB_NAME,
        "fleet_secret_name": FLEET_SECRET_NAME,
        "cpu_priority_class": "fleet-serve-low",
        "preemption_policy": "Never",
        "ready_timeout_seconds": controller.READY_TIMEOUT_SECONDS,
        "credentialed_release_required": True,
        "server_launch_authorized": False,
        "qualification_launch_authorized": False,
        "scored_launch_authorized": False,
        "api_mutation_calls": 0,
        "protected_content_included": False,
    }
    body["receipt_sha256"] = crypto.digest_without(body, "receipt_sha256")
    return body
