"""Render the create-once score-blind hosted GLM bulk release observer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from evals.fleet import hosted_glm_exact_bulk_package_v1 as bulk_package
from evals.fleet import hosted_glm_exact_bulk_runtime_v1 as bulk_runtime
from evals.fleet import self_hosted

JOB_NAME = "chris-glm53-exact100-hosted-bulk-v1-release"
CONFIGMAP_NAME = JOB_NAME + "-run"
OBSERVER_IMAGE = (
    "ghcr.io/astral-sh/uv:python3.12-bookworm@"
    "sha256:9aa60c50016c0485636ab9a830246a6ef3399aa4a8bab3d17ef4a2358fba2ca7"
)
PATHS = {
    **{name: relative for name, relative in bulk_package.PATHS.items() if name != "run.sh"},
    "release.py": "evals/fleet/hosted_glm_exact_bulk_release_v1.py",
    "run.sh": "evals/fleet/scripts/run_hosted_glm_exact_bulk_release_v1.sh",
}
INSTALL_PATHS = {
    **bulk_package.INSTALL_PATHS,
    "release.py": "evals/fleet/hosted_glm_exact_bulk_release_v1.py",
}


def render(
    root: Path, *, canary_accepted: Path | None = None, canary_terminal: Path | None = None
) -> dict[str, Any]:
    authorized = canary_accepted is not None or canary_terminal is not None
    if authorized:
        if canary_accepted is None or canary_terminal is None:
            raise ValueError("release authorization requires both canary receipts")
        bulk_runtime.validate_canary_receipts(
            bulk_package.bulk.load(canary_accepted),
            bulk_package.bulk.load(canary_terminal),
        )
    data = {}
    for name, relative in PATHS.items():
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"unsafe release package source: {relative}")
        data[name] = path.read_text()
    run_script = data["run.sh"]
    for name in INSTALL_PATHS:
        if f"{name}:" not in run_script and f"/bootstrap/{name}" not in run_script:
            raise ValueError(f"release runtime install closure omits {name}")
    configmap = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": CONFIGMAP_NAME, "namespace": "fleet-train-jobs"},
        "immutable": True,
        "data": data,
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
                "cyber-post-train.fleet.ai/launch-authorized": str(authorized).lower(),
            },
        },
        "spec": {
            "backoffLimit": 0,
            "ttlSecondsAfterFinished": 604800,
            "activeDeadlineSeconds": 3600,
            "template": {
                "metadata": {
                    "labels": {
                        "cyber-post-train.fleet.ai/experiment": JOB_NAME,
                        "cyber-post-train.fleet.ai/owner": "chris",
                    }
                },
                "spec": {
                    "restartPolicy": "Never",
                    "priorityClassName": "fleet-serve-low",
                    "preemptionPolicy": "Never",
                    "serviceAccountName": "default",
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
                            "name": "observer",
                            "image": OBSERVER_IMAGE,
                            "command": ["/bin/bash", "/bootstrap/run.sh"],
                            "env": [
                                {
                                    "name": "JOB_UID",
                                    "valueFrom": {
                                        "fieldRef": {
                                            "fieldPath": (
                                                "metadata.labels["
                                                "'batch.kubernetes.io/controller-uid']"
                                            )
                                        }
                                    },
                                },
                                {
                                    "name": "POD_UID",
                                    "valueFrom": {"fieldRef": {"fieldPath": "metadata.uid"}},
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
                            ],
                            "resources": {
                                "requests": {"cpu": "100m", "memory": "256Mi"},
                                "limits": {"cpu": "1", "memory": "1Gi"},
                            },
                            "volumeMounts": [
                                {"name": "bootstrap", "mountPath": "/bootstrap", "readOnly": True},
                                {"name": "workspace", "mountPath": "/workspace"},
                                {"name": "sfs", "mountPath": "/mnt/sfs"},
                            ],
                        }
                    ],
                    "volumes": [
                        {"name": "bootstrap", "configMap": {"name": CONFIGMAP_NAME}},
                        {"name": "workspace", "emptyDir": {}},
                        {"name": "sfs", "persistentVolumeClaim": {"claimName": "sfs-shared"}},
                    ],
                },
            },
        },
    }
    objects = {"apiVersion": "v1", "kind": "List", "items": [configmap, job]}
    if len(json.dumps(configmap).encode()) >= 900_000:
        raise ValueError("release observer ConfigMap exceeds safety budget")
    return {
        "objects": objects,
        "package_sha256": self_hosted.sha256(self_hosted.canonical_json(objects)),
        "launch_authorized": authorized,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preview", "render"))
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    parser.add_argument("--canary-accepted", type=Path)
    parser.add_argument("--canary-terminal", type=Path)
    args = parser.parse_args()
    value = render(
        args.repo.resolve(),
        canary_accepted=args.canary_accepted,
        canary_terminal=args.canary_terminal,
    )
    if args.command == "preview":
        print(json.dumps({key: item for key, item in value.items() if key != "objects"}))
        return 0
    if args.output is None or args.output.exists() or args.output.is_symlink():
        parser.error("render requires an unused --output")
    args.output.write_text(yaml.safe_dump(value["objects"], sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
