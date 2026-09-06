"""Render the CPU-only projected-ConfigMap GLM v22 generation canary."""

# ruff: noqa: E501 -- immutable image and downward-API fields are intentionally exact.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evals.fleet import glm53_dedicated_v22_generation_consistency_v1 as gate
from evals.fleet import self_hosted

NAME = "chris-glm53-dedicated-v22-generation-consistency-v1"
OUTPUT = f"/mnt/sfs/jobs/{NAME}/CANARY.json"


def render(root: Path) -> dict[str, Any]:
    fixture = gate.qualification_fixture()
    gate.validate(fixture)
    data = {
        "gate.py": (
            root / "evals/fleet/glm53_dedicated_v22_generation_consistency_v1.py"
        ).read_text(),
        "self_hosted.py": (root / "evals/fleet/self_hosted.py").read_text(),
        "fixture.json": self_hosted.canonical_json(fixture).decode(),
    }
    package_sha256 = self_hosted.sha256(self_hosted.canonical_json(data))
    configmap = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": NAME,
            "namespace": "fleet-train-jobs",
            "annotations": {"cyber-post-train.fleet.ai/package-sha256": package_sha256},
        },
        "immutable": True,
        "data": data,
    }
    script = (
        "set -euo pipefail\n"
        "root=$(mktemp -d)\n"
        'mkdir -p "$root/evals/fleet"\n'
        'touch "$root/evals/__init__.py" "$root/evals/fleet/__init__.py"\n'
        'cp /bootstrap/gate.py "$root/evals/fleet/glm53_dedicated_v22_generation_consistency_v1.py"\n'
        'cp /bootstrap/self_hosted.py "$root/evals/fleet/self_hosted.py"\n'
        "PYTHONPATH=\"$root\" python - <<'PY'\n"
        "import os\n"
        "from pathlib import Path\n"
        "from evals.fleet.glm53_dedicated_v22_generation_consistency_v1 import run_canary\n"
        f"run_canary(Path('/bootstrap/fixture.json'), Path('{OUTPUT}'))\n"
        "PY\n"
    )
    job = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": NAME,
            "namespace": "fleet-train-jobs",
            "labels": {"cyber-post-train.fleet.ai/experiment": NAME},
            "annotations": {
                "cyber-post-train.fleet.ai/launch-authorized": "true",
                "cyber-post-train.fleet.ai/package-sha256": package_sha256,
            },
        },
        "spec": {
            "backoffLimit": 0,
            "template": {
                "metadata": {"labels": {"cyber-post-train.fleet.ai/experiment": NAME}},
                "spec": {
                    "restartPolicy": "Never",
                    "priorityClassName": "fleet-infra-quiet",
                    "preemptionPolicy": "Never",
                    "containers": [
                        {
                            "name": "validator",
                            "image": "ghcr.io/astral-sh/uv:python3.12-bookworm@sha256:9aa60c50016c0485636ab9a830246a6ef3399aa4a8bab3d17ef4a2358fba2ca7",
                            "command": ["bash", "-lc"],
                            "args": [script],
                            "env": [
                                {
                                    "name": "JOB_UID",
                                    "valueFrom": {
                                        "fieldRef": {
                                            "fieldPath": "metadata.labels['batch.kubernetes.io/controller-uid']"
                                        }
                                    },
                                },
                                {
                                    "name": "POD_UID",
                                    "valueFrom": {"fieldRef": {"fieldPath": "metadata.uid"}},
                                },
                            ],
                            "volumeMounts": [
                                {"name": "bootstrap", "mountPath": "/bootstrap", "readOnly": True},
                                {"name": "sfs", "mountPath": "/mnt/sfs"},
                            ],
                            "resources": {
                                "requests": {"cpu": "100m", "memory": "128Mi"},
                                "limits": {"cpu": "500m", "memory": "512Mi"},
                            },
                            "securityContext": {
                                "allowPrivilegeEscalation": False,
                                "readOnlyRootFilesystem": True,
                                "runAsNonRoot": False,
                                "runAsUser": 0,
                                "runAsGroup": 0,
                            },
                        }
                    ],
                    "volumes": [
                        {"name": "bootstrap", "configMap": {"name": NAME}},
                        {"name": "sfs", "persistentVolumeClaim": {"claimName": "sfs-shared"}},
                    ],
                },
            },
        },
    }
    return {
        "schema_version": "fleet-glm53-dedicated-v22-generation-canary-package-v1",
        "status": "READY_CPU_ONLY",
        "gpu_requests": 0,
        "scoring_calls": 0,
        "output": OUTPUT,
        "package_sha256": package_sha256,
        "objects": {"apiVersion": "v1", "kind": "List", "items": [configmap, job]},
    }


def main() -> int:
    print(json.dumps(render(Path.cwd())["objects"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
