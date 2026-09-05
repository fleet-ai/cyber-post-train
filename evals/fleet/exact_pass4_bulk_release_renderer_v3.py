"""Render the four runnable bulk Jobs only after the append-only release validates."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import yaml

from evals.fleet import exact_pass4_bulk_package_v3 as package
from evals.fleet import exact_pass4_bulk_v3 as bulk

NAMESPACE = "fleet-train-jobs"
DIND_IMAGE = (
    "docker.io/library/docker@sha256:"
    "f649ef046008ca7f926a2571c32b0ac22e5c59eb61b959617f9acc2a4c638cf5"
)
RUNNER_IMAGE = (
    "ghcr.io/astral-sh/uv:python3.12-bookworm@sha256:"
    "9aa60c50016c0485636ab9a830246a6ef3399aa4a8bab3d17ef4a2358fba2ca7"
)
DOCKER_CLI_SHA256 = "242c7a8de606afba2acada7c7af00d77f92c3601678b2f3a60911b49a892c722"
DOCKER_BUILDX_SHA256 = "8c38f60308a895fa570f1410e453c5de11aafd65a99fa99965d96d24b6225a78"
DOCKER_CLI_TOTAL_BYTES = 105_594_160
DOCKER_CLI_VOLUME_BYTES = 256 * 1024 * 1024
DOCKER_CLI_VOLUME_SIZE = "256Mi"


def _job(controller: str, manifest: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    names = bulk.CONTROLLERS[controller]
    run_key = package.data_key(bulk.RUN_PATH)
    common_mounts = [
        {"name": "docker-socket", "mountPath": "/var/run"},
        {"name": "workspace", "mountPath": "/workspace"},
        {"name": "sfs", "mountPath": "/mnt/sfs"},
    ]
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": names["job_name"],
            "namespace": NAMESPACE,
            "labels": {
                "cyber-post-train.fleet.ai/owner": "chris",
                "cyber-post-train.fleet.ai/experiment": "exact100-bulk-v3",
                "kueue.x-k8s.io/queue-name": "training-lq",
            },
            "annotations": {
                "cyber-post-train.fleet.ai/launch-authorized": "true",
                "cyber-post-train.fleet.ai/create-once": "true",
                "cyber-post-train.fleet.ai/release-receipt-sha256": manifest[
                    "release_receipt_sha256"
                ],
            },
        },
        "spec": {
            # Two endpoint streams per model are intentional.  At the frozen
            # eight-hour per-cell ceiling, a 199/200-cell serial controller can
            # validly run for roughly 67 days; the Job deadline must not
            # truncate that scientifically valid tail.
            "activeDeadlineSeconds": 7776000,
            # A restarted Pod resumes the same immutable plan/output root and
            # skips every globally claimed cell.  This retries controller
            # plumbing only; it never repeats a statistical execution.
            "backoffLimit": 3,
            "ttlSecondsAfterFinished": 604800,
            "template": {
                "metadata": {
                    "labels": {
                        "cyber-post-train.fleet.ai/owner": "chris",
                        "cyber-post-train.fleet.ai/experiment": "exact100-bulk-v3",
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
                            "name": "docker-cli",
                            "image": DIND_IMAGE,
                            "command": ["/bin/sh", "-ec"],
                            "args": [
                                " ".join(
                                    (
                                        "install -D -m 0755 /usr/local/bin/docker /cli/bin/docker;",
                                        "install -D -m 0755",
                                        "/usr/local/libexec/docker/cli-plugins/docker-buildx",
                                        "/cli/plugins/docker-buildx;",
                                        "test \"$(sha256sum /cli/bin/docker | awk "
                                        "'{print $1}')\" =",
                                        f"{DOCKER_CLI_SHA256};",
                                        "test \"$(sha256sum /cli/plugins/docker-buildx | awk",
                                        "'{print $1}')\" =",
                                        f"{DOCKER_BUILDX_SHA256};",
                                        "test \"$(( $(wc -c < /cli/bin/docker) +",
                                        "$(wc -c < /cli/plugins/docker-buildx) ))\" -eq",
                                        str(DOCKER_CLI_TOTAL_BYTES),
                                    )
                                )
                            ],
                            "resources": {
                                "requests": {
                                    "cpu": "10m",
                                    "memory": "32Mi",
                                    "ephemeral-storage": "32Mi",
                                },
                                "limits": {
                                    "cpu": "100m",
                                    "memory": "128Mi",
                                    "ephemeral-storage": "128Mi",
                                },
                            },
                            "volumeMounts": [{"name": "docker-cli", "mountPath": "/cli"}],
                        },
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
                            "volumeMounts": [
                                *copy.deepcopy(common_mounts),
                                {"name": "docker-data", "mountPath": "/var/lib/docker"},
                            ],
                        }
                    ],
                    "containers": [
                        {
                            "name": "evaluator",
                            "image": RUNNER_IMAGE,
                            "command": ["/bin/bash", f"/bootstrap/{run_key}"],
                            "env": [
                                {
                                    "name": "PATH",
                                    "value": (
                                        "/docker-cli/bin:/usr/local/sbin:/usr/local/bin:"
                                        "/usr/sbin:/usr/bin:/sbin:/bin"
                                    ),
                                },
                                {"name": "DOCKER_CONFIG", "value": "/workspace/docker-config"},
                                {"name": "DOCKER_HOST", "value": "unix:///var/run/docker.sock"},
                                {"name": "DOCKER_TLS_CERTDIR", "value": ""},
                                {"name": "BULK_PLAN", "value": "/bootstrap/runtime-plan.json"},
                                {"name": "BULK_OUTPUT_ROOT", "value": plan["sfs_root"]},
                                {
                                    "name": "BULK_PACKAGE_AGGREGATE_SHA256",
                                    "value": manifest["package_aggregate_sha256"],
                                },
                                {
                                    "name": "BULK_PACKAGE_COMMIT",
                                    "value": manifest["package_commit"],
                                },
                                *[
                                    {"name": name, "value": value}
                                    for name, value in manifest["runtime_gates"].items()
                                ],
                                {
                                    "name": "JOB_UID",
                                    "valueFrom": {
                                        "fieldRef": {
                                            "apiVersion": "v1",
                                            "fieldPath": (
                                                "metadata.labels["
                                                "'batch.kubernetes.io/controller-uid']"
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
                                            "name": "chris-cyber-opencode-evals-v3",
                                            "key": "FLEET_API_KEY",
                                        }
                                    },
                                },
                            ],
                            "resources": {
                                "requests": {
                                    "cpu": "100m",
                                    "memory": "512Mi",
                                    "ephemeral-storage": "5Gi",
                                },
                                "limits": {
                                    "cpu": "1",
                                    "memory": "2Gi",
                                    "ephemeral-storage": "20Gi",
                                },
                            },
                            "volumeMounts": [
                                *copy.deepcopy(common_mounts),
                                {
                                    "name": "docker-cli",
                                    "mountPath": "/docker-cli",
                                    "readOnly": True,
                                },
                                {"name": "bootstrap", "mountPath": "/bootstrap", "readOnly": True},
                            ],
                        }
                    ],
                    "volumes": [
                        {
                            "name": "bootstrap",
                            "projected": {
                                "sources": [
                                    *[
                                        {"configMap": {"name": name}}
                                        for name in package.CORE_NAMES
                                    ],
                                    {"configMap": {"name": names["configmap_name"]}},
                                ]
                            },
                        },
                        {
                            "name": "docker-cli",
                            "emptyDir": {"sizeLimit": DOCKER_CLI_VOLUME_SIZE},
                        },
                        {"name": "docker-data", "emptyDir": {"sizeLimit": "40Gi"}},
                        {"name": "docker-socket", "emptyDir": {}},
                        {"name": "workspace", "emptyDir": {"sizeLimit": "5Gi"}},
                        {"name": "sfs", "persistentVolumeClaim": {"claimName": "sfs-shared"}},
                    ],
                },
            },
        },
    }


def render(
    release: dict[str, Any],
    inventory: dict[str, Any],
    root: Path,
    *,
    package_commit_authority: Path | None = None,
    evidence_root: Path | None = None,
    collector_job: dict[str, Any] | None = None,
    collector_pods: dict[str, Any] | None = None,
) -> dict[str, Any]:
    bulk.validate_release(
        release,
        root,
        package_commit_authority=package_commit_authority,
        evidence_root=evidence_root,
        collector_job=collector_job,
        collector_pods=collector_pods,
        reconciliation_max_age_seconds=900,
    )
    bulk.validate_inventory_gate(inventory, root)
    released_inventory = bulk.load(
        bulk._evidence_path(  # noqa: SLF001 - renderer enforces the gate's exact producer
            root,
            release["gate_receipts"]["exact100_inventory"],
            "inventory",
            evidence_root=evidence_root,
        )
    )
    if inventory != released_inventory:
        raise ValueError("renderer inventory differs from released inventory producer")
    built = package.build_package(root)
    released_package = release["package"]
    if (
        built["aggregate_sha256"] != released_package["aggregate_sha256"]
        or sorted(built["configmaps"]) != released_package["objects"]
        or built["object_json_bytes"] != released_package["object_json_bytes"]
    ):
        raise ValueError("final bulk package differs from the released package")
    items: list[dict[str, Any]] = []
    runtime_plans: dict[str, dict[str, Any]] = {}
    for controller in bulk.CONTROLLERS:
        plan = bulk.build_runtime_plan(controller, inventory, root)
        runtime_plans[controller] = plan
        name = bulk.CONTROLLERS[controller]["configmap_name"]
        configmap = copy.deepcopy(built["configmaps"][name])
        configmap["metadata"]["annotations"].update(
            {
                "cyber-post-train.fleet.ai/preview-only": "false",
                "cyber-post-train.fleet.ai/launch-authorized": "true",
                "cyber-post-train.fleet.ai/release-receipt-sha256": release["receipt_sha256"],
            }
        )
        configmap["data"]["runtime-plan.json"] = (
            json.dumps(plan, sort_keys=True, separators=(",", ":")) + "\n"
        )
        encoded_size = len(json.dumps(configmap, sort_keys=True, separators=(",", ":")).encode())
        if encoded_size >= 900000:
            raise ValueError("released controller ConfigMap exceeds the safety budget")
        items.append(configmap)
    core_configmaps = []
    for name in package.CORE_NAMES:
        core_configmap = copy.deepcopy(built["configmaps"][name])
        core_configmap["metadata"]["annotations"].update(
            {
                "cyber-post-train.fleet.ai/preview-only": "false",
                "cyber-post-train.fleet.ai/launch-authorized": "true",
                "cyber-post-train.fleet.ai/release-receipt-sha256": release[
                    "receipt_sha256"
                ],
            }
        )
        core_configmaps.append(core_configmap)
    items[0:0] = core_configmaps
    reconciliation = bulk.load(
        bulk._evidence_path(  # noqa: SLF001 - fixed release gate projection
            root,
            release["gate_receipts"]["duplicate_reconciliation"],
            "reconciliation",
            evidence_root=evidence_root,
        )
    )
    runtime_gates = {
        "BULK_INVENTORY_GATE_PATH": release["gate_receipts"]["exact100_inventory"],
        "BULK_INVENTORY_GATE_SHA256": released_inventory["receipt_sha256"],
        "BULK_RECONCILIATION_GATE_PATH": release["gate_receipts"]["duplicate_reconciliation"],
        "BULK_RECONCILIATION_GATE_SHA256": reconciliation["receipt_sha256"],
        "BULK_QWEN_CANARY_GATE_PATH": release["gate_receipts"]["qwen_generation7_canary"],
        "BULK_QWEN_CANARY_GATE_SHA256": reconciliation["generation7_gate_receipts"]["qwen3.8-27b"][
            "receipt_sha256"
        ],
        "BULK_GLM_CANARY_GATE_PATH": release["gate_receipts"]["glm_generation7_canary"],
        "BULK_GLM_CANARY_GATE_SHA256": reconciliation["generation7_gate_receipts"]["glm-5.3"][
            "receipt_sha256"
        ],
    }
    for controller in bulk.CONTROLLERS:
        plan = runtime_plans[controller]
        manifest = {
            "release_receipt_sha256": release["receipt_sha256"],
            "package_aggregate_sha256": built["controller_manifests"][controller][
                "aggregate_sha256"
            ],
            "package_commit": release["package_commit"],
            "runtime_gates": runtime_gates,
        }
        items.append(_job(controller, manifest, plan))
    return {"apiVersion": "v1", "kind": "List", "items": items}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--package-commit-authority",
        type=Path,
        help="Git repository used to verify an immutable materialized commit snapshot",
    )
    parser.add_argument(
        "--evidence-root",
        type=Path,
        required=True,
        help="Private local mirror of fixed sanitized /mnt/sfs/jobs evidence",
    )
    parser.add_argument("--collector-job", type=Path, required=True)
    parser.add_argument("--collector-pods", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    value = render(
        bulk.load(args.release),
        bulk.load(args.inventory),
        args.repo.resolve(),
        package_commit_authority=(
            args.package_commit_authority.resolve()
            if args.package_commit_authority is not None
            else None
        ),
        evidence_root=args.evidence_root.resolve(),
        collector_job=bulk.load(args.collector_job),
        collector_pods=bulk.load(args.collector_pods),
    )
    text = yaml.safe_dump(value, sort_keys=False)
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
