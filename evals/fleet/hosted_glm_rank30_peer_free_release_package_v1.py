"""Render the immutable score-blind rank-30 release observer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from evals.fleet import hosted_glm_rank30_peer_free_package_v2 as controller_package
from evals.fleet import hosted_glm_rank30_peer_free_release_observer_v1 as observer
from evals.fleet import hosted_glm_rank30_peer_free_successor_v2 as successor
from evals.fleet import self_hosted

JOB_NAME = "chris-glm53-peer-free-release-observer-v1"
CONFIGMAP_NAME = JOB_NAME + "-package"
SECRET_NAME = "chris-cyber-opencode-evals-v2"
UV_IMAGE = (
    "ghcr.io/astral-sh/uv:python3.12-bookworm@"
    "sha256:9aa60c50016c0485636ab9a830246a6ef3399aa4a8bab3d17ef4a2358fba2ca7"
)
EXTRA_FILES = {
    successor.LEDGER_AUTHORITY["path"],
    successor.LIVE_LEDGER_VALIDATION["path"],
    "evals/fleet/hosted_glm_rank30_peer_free_release_bootstrap_v1.py",
    "evals/fleet/hosted_glm_rank30_peer_free_release_observer_v1.py",
    "evals/fleet/hosted_glm_exact_bulk_runtime_v1.py",
    "evals/fleet/hosted_glm_exact_bulk_v1.py",
}
FILES = tuple(sorted(set(controller_package.PATHS.values()) | EXTRA_FILES))


def _seal(value: dict[str, Any]) -> dict[str, Any]:
    return {**value, "receipt_sha256": observer.digest(value)}


def build_binding(root: Path) -> dict[str, Any]:
    plan = successor.validate_all(root)[successor.CONTROLLER]
    body: dict[str, Any] = {
        "schema_version": observer.SCHEMA,
        "release_schema": successor.RELEASE_SCHEMA,
        "cell_ids": [row["cell_id"] for row in plan["attempts"]],
        "task_keys": sorted({row["task_key"] for row in plan["attempts"]}),
        "fresh_job_name": successor.JOB_NAME,
        "fresh_configmap_name": successor.CONFIGMAP_NAME,
        "fresh_sfs_root": successor.SFS_ROOT,
        "claim_root": str(successor.prior.CLAIM_ROOT),
        "jobs_root": str(successor.prior.JOBS_ROOT),
        "lease_root": str(successor.prior.LEASE_ROOT),
        "endpoint_key": successor.prior.LEASE_ENDPOINT_KEY,
        "allowed_rank30_paths": [successor.DIAGNOSTIC_V2["path"].rsplit("/", 1)[0]],
        "held_source_package_sha256": controller_package.source_package_sha256(root),
        "ledger_authority": successor.LEDGER_AUTHORITY,
        "live_ledger_validation": successor.LIVE_LEDGER_VALIDATION,
        "diagnostic_v2": successor.DIAGNOSTIC_V2,
        "superseded_identities": successor.SUPERSEDED_IDENTITIES,
    }
    binding = {
        **body,
        "binding_sha256": observer.sha256(observer.canonical(body)),
    }
    observer.validate_binding(binding)
    return binding


def _source_data(root: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    for relative in FILES:
        source = root / relative
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"unsafe observer package source: {relative}")
        data[relative.replace("/", "__SLASH__")] = source.read_text()
    return data


def render(root: Path, *, launch_authorized: bool = False) -> dict[str, Any]:
    if controller_package.PATHS != observer.CONTROLLER_SOURCE_PATHS:
        raise ValueError("controller source path authority drifted")
    root = root.resolve(strict=True)
    binding = build_binding(root)
    data = _source_data(root)
    files = {
        relative: self_hosted.sha256(data[relative.replace("/", "__SLASH__")].encode())
        for relative in FILES
    }
    package = _seal(
        {
            "schema_version": observer.PACKAGE_SCHEMA,
            "files": files,
            "file_count": len(files),
        }
    )
    data["binding.json"] = json.dumps(binding, sort_keys=True, separators=(",", ":")) + "\n"
    data["package.json"] = json.dumps(package, sort_keys=True, separators=(",", ":")) + "\n"
    configmap = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": CONFIGMAP_NAME, "namespace": observer.NAMESPACE},
        "immutable": True,
        "data": data,
    }
    command = f"""
set -euo pipefail
test ! -e {observer.OUTPUT_PATH}
mkdir -p /work
for source in /bootstrap/*__SLASH__*; do
  target="/work/$(basename "$source" | sed 's,__SLASH__,/,g')"
  install -D -m 0444 "$source" "$target"
done
export PYTHONPATH=/work
exec uv run --no-project --with pyyaml --with httpx==0.28.1 python \
  -m evals.fleet.hosted_glm_rank30_peer_free_release_bootstrap_v1 \
  --package /bootstrap/package.json --projected-root /bootstrap --repo /work -- \
  --binding /bootstrap/binding.json --package /bootstrap/package.json \
  --repo /work --output {observer.OUTPUT_PATH}
""".strip()
    job = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": JOB_NAME,
            "namespace": observer.NAMESPACE,
            "labels": {
                "cyber-post-train.fleet.ai/owner": "chris",
                "cyber-post-train.fleet.ai/experiment": JOB_NAME,
                "kueue.x-k8s.io/queue-name": "training-lq",
            },
            "annotations": {
                "cyber-post-train.fleet.ai/create-once": "true",
                "cyber-post-train.fleet.ai/score-free": "true",
                "cyber-post-train.fleet.ai/diagnostic-only": "true",
                "cyber-post-train.fleet.ai/launch-authorized": str(launch_authorized).lower(),
            },
        },
        "spec": {
            "backoffLimit": 0,
            "activeDeadlineSeconds": 900,
            "ttlSecondsAfterFinished": 604800,
            "template": {
                "metadata": {"labels": {"cyber-post-train.fleet.ai/owner": "chris"}},
                "spec": {
                    "restartPolicy": "Never",
                    "serviceAccountName": "default",
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
                    "containers": [
                        {
                            "name": "observer",
                            "image": UV_IMAGE,
                            "command": ["bash", "-ec", command],
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
                                            "name": SECRET_NAME,
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
                                {"name": "package", "mountPath": "/bootstrap", "readOnly": True},
                                {"name": "sfs", "mountPath": "/mnt/sfs"},
                            ],
                        }
                    ],
                    "volumes": [
                        {"name": "package", "configMap": {"name": CONFIGMAP_NAME}},
                        {"name": "sfs", "persistentVolumeClaim": {"claimName": "sfs-shared"}},
                    ],
                },
            },
        },
    }
    if len(json.dumps(configmap).encode()) >= 900_000:
        raise ValueError("observer ConfigMap exceeds safety budget")
    return {
        "apiVersion": "v1",
        "kind": "List",
        "items": [configmap, job],
        "launch_authorized": launch_authorized,
        "scoring_authorized": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    parser.add_argument("--authorize-score-free-observer", action="store_true")
    args = parser.parse_args(argv)
    rendered = render(
        args.root.resolve(strict=True),
        launch_authorized=args.authorize_score_free_observer,
    )
    payload = yaml.safe_dump(rendered, sort_keys=False)
    if args.output is None:
        print(payload, end="")
    else:
        args.output.write_text(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
