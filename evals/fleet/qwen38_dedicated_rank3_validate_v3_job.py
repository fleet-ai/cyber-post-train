"""Render create-once chain-validator Jobs for dedicated Qwen rank3."""

# ruff: noqa: E501 -- bootstrap paths are immutable experiment bindings.

from __future__ import annotations

from typing import Any

IMAGE = (
    "ghcr.io/astral-sh/uv:python3.12-bookworm@"
    "sha256:9aa60c50016c0485636ab9a830246a6ef3399aa4a8bab3d17ef4a2358fba2ca7"
)
BOOTSTRAP = """root=/workspace/cyber-post-train
mkdir -p "$root/evals/fleet/configs" "$root/docs/evidence/qwen38-study"
touch "$root/evals/__init__.py" "$root/evals/fleet/__init__.py"
install -m 0644 /bootstrap/self_hosted.py "$root/evals/fleet/self_hosted.py"
install -m 0644 /bootstrap/exact_pass4_crypto.py "$root/evals/fleet/exact_pass4_crypto.py"
install -m 0644 /bootstrap/exact_pass4_universe.py "$root/evals/fleet/exact_pass4_universe.py"
install -m 0644 /bootstrap/legacy.py "$root/evals/fleet/qwen38_dedicated_scored_canary_v1.py"
install -m 0644 /bootstrap/lane.py "$root/evals/fleet/qwen38_dedicated_rank3_v3.py"
install -m 0644 /bootstrap/validator.py "$root/evals/fleet/qwen38_dedicated_rank3_validate_v3.py"
install -m 0644 /bootstrap/parity.json "$root/docs/evidence/qwen38-study/2026-09-05-qwen38-dedicated-tp1-b-v2-actual-opencode-parity.json"
install -m 0644 /bootstrap/rank3-release.json "$root/docs/evidence/qwen38-study/2026-09-05-qwen38-dedicated-rank3-g21-b-v2-release-v3.json"
install -m 0644 /bootstrap/campaign.json "$root/evals/fleet/configs/q38-glm53-exact-easiest100-pass4-campaign-v1.json"
install -m 0644 /bootstrap/selection.json "$root/evals/fleet/configs/opencode-easiest-train100-selection-v2.json"
install -m 0644 /bootstrap/source.json "$root/evals/fleet/configs/qwen38-opencode-train50-pass4-v3.json"
cd "$root"
exec uv run --no-project --with httpx==0.28.1 python -m evals.fleet.qwen38_dedicated_rank3_validate_v3
"""


def _identity(attempt: int) -> tuple[str, str]:
    if attempt not in (1, 2, 3, 4):
        raise ValueError("dedicated Qwen validator attempt must be 1 through 4")
    return (
        f"chris-cyber-q38-ded-r003-a{attempt}-validate-v1",
        f"q38-ded-r003-a{attempt}-validate-v1",
    )


def render(attempt: int) -> dict[str, Any]:
    name, experiment = _identity(attempt)
    labels = {
        "cyber-post-train.fleet.ai/experiment": experiment,
        "cyber-post-train.fleet.ai/owner": "chris",
        "kueue.x-k8s.io/queue-name": "training-lq",
    }
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": name, "namespace": "fleet-train-jobs", "labels": labels},
        "spec": {
            "activeDeadlineSeconds": 1800,
            "backoffLimit": 0,
            "ttlSecondsAfterFinished": 604800,
            "template": {
                "metadata": {
                    "labels": {
                        **labels,
                        "kueue.x-k8s.io/podset": "main",
                    }
                },
                "spec": {
                    "containers": [
                        {
                            "name": "validator",
                            "image": IMAGE,
                            "imagePullPolicy": "IfNotPresent",
                            "command": ["/bin/bash", "-ceu", "--"],
                            "args": [BOOTSTRAP],
                            "env": [
                                {
                                    "name": "JOB_UID",
                                    "valueFrom": {
                                        "fieldRef": {
                                            "apiVersion": "v1",
                                            "fieldPath": (
                                                "metadata.labels['batch.kubernetes.io/controller-uid']"
                                            ),
                                        }
                                    },
                                },
                                {
                                    "name": "POD_UID",
                                    "valueFrom": {
                                        "fieldRef": {
                                            "apiVersion": "v1",
                                            "fieldPath": "metadata.uid",
                                        }
                                    },
                                },
                                {
                                    "name": "FLEET_API_KEY",
                                    "valueFrom": {
                                        "secretKeyRef": {
                                            "name": "chris-cyber-opencode-evals-v2",
                                            "key": "FLEET_API_KEY",
                                        }
                                    },
                                },
                                {"name": "QWEN_DEDICATED_ATTEMPT", "value": str(attempt)},
                                {
                                    "name": "QWEN_RANK3_RELEASE_PATH",
                                    "value": "/workspace/cyber-post-train/docs/evidence/qwen38-study/2026-09-05-qwen38-dedicated-rank3-g21-b-v2-release-v3.json",
                                },
                            ],
                            "resources": {
                                "requests": {"cpu": "100m", "memory": "512Mi"},
                                "limits": {"cpu": "1", "memory": "2Gi"},
                            },
                            "volumeMounts": [
                                {"name": "bootstrap", "mountPath": "/bootstrap", "readOnly": True},
                                {"name": "sfs", "mountPath": "/mnt/sfs"},
                            ],
                        }
                    ],
                    "nodeSelector": {
                        "kubernetes.io/arch": "amd64",
                        "workload": "fleetai-training-ng-cpu",
                    },
                    "preemptionPolicy": "Never",
                    "priorityClassName": "fleet-serve-low",
                    "restartPolicy": "Never",
                    "tolerations": [
                        {
                            "effect": "NoSchedule",
                            "key": "workload",
                            "operator": "Equal",
                            "value": "fleetai-training-ng-cpu",
                        }
                    ],
                    "volumes": [
                        {"name": "bootstrap", "configMap": {"name": name}},
                        {"name": "sfs", "persistentVolumeClaim": {"claimName": "sfs-shared"}},
                    ],
                },
            },
        },
    }
