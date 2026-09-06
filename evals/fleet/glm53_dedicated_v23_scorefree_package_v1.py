"""Immutable held package renderer for the GLM v23 score-free qualifier."""

from __future__ import annotations

import copy
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from evals.fleet import exact_pass4_crypto as crypto
from evals.fleet import glm53_dedicated_v22_concurrency_package_v1 as prior
from evals.fleet import glm53_dedicated_v23_scorefree_qualifier_v1 as qualifier

SCHEMA = "fleet-glm53-dedicated-v23-scorefree-package-v1"
CONFIGMAP_NAME = qualifier.JOB_NAME + "-package"
AUTHORIZATION_CONFIGMAP_NAME = qualifier.JOB_NAME + "-authorization"
WATCHDOG_JOB_NAME = "chris-glm53-dedicated-v23-request-watchdog-v1"
WATCHDOG_CONFIGMAP_NAME = WATCHDOG_JOB_NAME + "-package"
WATCHDOG_RESULT_ROOT = f"/mnt/sfs/jobs/{WATCHDOG_JOB_NAME}"
COMMIT_RE = re.compile(r"[0-9a-f]{40}")
FILES = tuple(
    path
    for path in prior.FILES
    if "glm53_dedicated_v22_concurrency_authorization_v1.py" not in path
    and "glm53_dedicated_v22_concurrency_qualification_v1.py" not in path
) + (
    "evals/fleet/glm53_dedicated_v22_concurrency_qualification_v1.py",
    "evals/fleet/glm53_dedicated_v23_scorefree_qualifier_v1.py",
    "evals/fleet/glm53_dedicated_v23_request_counter_watchdog_v1.py",
)
RUN = "evals/fleet/scripts/run_glm53_dedicated_v23_scorefree_qualification_v1.sh"
OPERATOR_FILES = (
    "evals/fleet/glm53_dedicated_v23_scorefree_gpu_observer_v1.py",
    "evals/fleet/scripts/observe_glm53_dedicated_v23_scorefree_gpu_v1.sh",
)


class PackageError(RuntimeError):
    """The immutable v23 package cannot be rendered."""


def validate_cpu_priority_inventory(priority_classes: list[dict[str, Any]]) -> None:
    """Require the live eligible nonpreempting class that outranks the old class."""

    by_name = {
        row.get("metadata", {}).get("name"): row
        for row in priority_classes
        if isinstance(row, dict)
    }
    selected = by_name.get(qualifier.CPU_PRIORITY_CLASS) or {}
    old = by_name.get("fleet-infra-quiet") or {}
    if (
        selected.get("value") != qualifier.CPU_PRIORITY_VALUE
        or selected.get("preemptionPolicy") != "Never"
        or old.get("preemptionPolicy") != "Never"
        or not isinstance(old.get("value"), int)
        or old["value"] >= selected["value"]
        or any(
            row.get("preemptionPolicy") == "Never"
            and isinstance(row.get("value"), int)
            and row["value"] > selected["value"]
            for row in priority_classes
        )
    ):
        raise PackageError("v23_cpu_priority_contract_invalid")


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
        manifest[path] = crypto.sha256(raw)
    run = _source(root, commit, RUN)
    data["run.sh"] = run.decode()
    manifest[RUN] = crypto.sha256(run)
    package = {
        "schema_version": SCHEMA,
        "package_commit": commit,
        "files": manifest,
        "job_name": qualifier.JOB_NAME,
        "output_root": str(qualifier.RESULT_ROOT),
        "concurrency_waves": list(qualifier.CONCURRENCY),
        "score_free": True,
        "server_launch_authorized": False,
        "qualification_launch_authorized": False,
        "scored_launch_authorized": False,
        "external_uid_bound_operator_files": {
            path: crypto.sha256(_source(root, commit, path)) for path in OPERATOR_FILES
        },
    }
    package["package_sha256"] = crypto.digest_without(package, "package_sha256")
    data["package.json"] = json.dumps(package, sort_keys=True, separators=(",", ":")) + "\n"
    data["package_commit"] = commit
    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": CONFIGMAP_NAME, "namespace": prior.NAMESPACE},
        "immutable": True,
        "data": data,
    }


def build_authorization_configmap(authorization: dict[str, Any]) -> dict[str, Any]:
    if (
        authorization.get("schema_version") != qualifier.AUTH_SCHEMA
        or authorization.get("qualification_launch_authorized") is not True
        or authorization.get("scored_launch_authorized") is not False
        or authorization.get("receipt_sha256")
        != crypto.digest_without(authorization, "receipt_sha256")
    ):
        raise PackageError("v23_authorization_invalid")
    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": AUTHORIZATION_CONFIGMAP_NAME, "namespace": prior.NAMESPACE},
        "immutable": True,
        "data": {
            "authorization.json": json.dumps(authorization, sort_keys=True, separators=(",", ":"))
            + "\n"
        },
    }


def build_watchdog_configmap(
    root: Path,
    commit: str,
    binding: dict[str, Any],
    *,
    ready_at_epoch: float,
) -> dict[str, Any]:
    """Package the concrete watcher before score-free qualification authorization."""

    qualifier._validate_binding(binding)  # noqa: SLF001
    if ready_at_epoch <= 0:
        raise PackageError("v23_watchdog_ready_epoch_invalid")
    source_files = (
        "evals/fleet/exact_pass4_crypto.py",
        "evals/fleet/glm53_dedicated_v23_request_counter_watchdog_v1.py",
    )
    data = {_key(path): _source(root, commit, path).decode() for path in source_files}
    manifest = {path: crypto.sha256(_source(root, commit, path)) for path in source_files}
    data["binding.json"] = json.dumps(binding, sort_keys=True, separators=(",", ":")) + "\n"
    watchdog_package = {
        "schema_version": "fleet-glm53-dedicated-v23-watchdog-package-v1",
        "package_commit": commit,
        "files": manifest,
        "binding_sha256": crypto.sha256(crypto.canonical_json(binding)),
        "ready_at_epoch": ready_at_epoch,
        "job_name": WATCHDOG_JOB_NAME,
        "result_root": WATCHDOG_RESULT_ROOT,
        "idle_release_seconds": qualifier.IDLE_RELEASE_SECONDS,
        "release_route": "DELETE /v1/runs/{api_run_id}",
        "create_once": True,
        "score_free": True,
    }
    watchdog_package["package_sha256"] = crypto.digest_without(watchdog_package, "package_sha256")
    data["package.json"] = (
        json.dumps(watchdog_package, sort_keys=True, separators=(",", ":")) + "\n"
    )
    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": WATCHDOG_CONFIGMAP_NAME, "namespace": prior.NAMESPACE},
        "immutable": True,
        "data": data,
    }


def build_watchdog_job(configmap: dict[str, Any]) -> dict[str, Any]:
    package = json.loads(configmap["data"]["package.json"])
    command = """
set -euo pipefail
test ! -e "$WATCHDOG_RESULT_ROOT"
mkdir -p /work/evals/fleet "$WATCHDOG_RESULT_ROOT"
for source in /bootstrap/*__SLASH__*; do
  target="/work/$(basename "$source" | sed 's,__SLASH__,/,g')"
  install -D -m 0444 "$source" "$target"
done
export PYTHONPATH=/work
exec python -m evals.fleet.glm53_dedicated_v23_request_counter_watchdog_v1 watch \
  --binding /bootstrap/binding.json \
  --active-receipt "$WATCHDOG_RESULT_ROOT/ACTIVE.json" \
  --ready-at-epoch "$READY_AT_EPOCH"
""".strip()
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": WATCHDOG_JOB_NAME,
            "namespace": prior.NAMESPACE,
            "labels": {
                "cyber-post-train.fleet.ai/owner": "chris",
                "cyber-post-train.fleet.ai/experiment": WATCHDOG_JOB_NAME,
                "kueue.x-k8s.io/queue-name": "training-lq",
            },
            "annotations": {
                "cyber-post-train.fleet.ai/create-once": "true",
                "cyber-post-train.fleet.ai/score-free": "true",
                "cyber-post-train.fleet.ai/package-sha256": package["package_sha256"],
            },
        },
        "spec": {
            "backoffLimit": 0,
            "activeDeadlineSeconds": 604800,
            "ttlSecondsAfterFinished": 604800,
            "template": {
                "metadata": {
                    "labels": {
                        "cyber-post-train.fleet.ai/owner": "chris",
                        "cyber-post-train.fleet.ai/experiment": WATCHDOG_JOB_NAME,
                        "kueue.x-k8s.io/queue-name": "training-lq",
                    }
                },
                "spec": {
                    "restartPolicy": "Never",
                    "priorityClassName": qualifier.CPU_PRIORITY_CLASS,
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
                            "name": "watchdog",
                            "image": prior.UV_IMAGE,
                            "command": ["bash", "-ec", command],
                            "env": [
                                {
                                    "name": "WATCHDOG_RESULT_ROOT",
                                    "value": WATCHDOG_RESULT_ROOT,
                                },
                                {
                                    "name": "READY_AT_EPOCH",
                                    "value": str(package["ready_at_epoch"]),
                                },
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
                                    "name": "GH_TOKEN",
                                    "valueFrom": {
                                        "secretKeyRef": {
                                            "name": "img-build-secrets",
                                            "key": "GH_TOKEN",
                                        }
                                    },
                                },
                            ],
                            "resources": {
                                "requests": {"cpu": "100m", "memory": "128Mi"},
                                "limits": {"cpu": "500m", "memory": "512Mi"},
                            },
                            "volumeMounts": [
                                {
                                    "name": "package",
                                    "mountPath": "/bootstrap",
                                    "readOnly": True,
                                },
                                {"name": "sfs", "mountPath": "/mnt/sfs"},
                            ],
                        }
                    ],
                    "volumes": [
                        {
                            "name": "package",
                            "configMap": {"name": WATCHDOG_CONFIGMAP_NAME},
                        },
                        {
                            "name": "sfs",
                            "persistentVolumeClaim": {"claimName": "sfs-shared"},
                        },
                    ],
                },
            },
        },
    }


def render_watchdog(
    root: Path,
    commit: str,
    binding: dict[str, Any],
    *,
    ready_at_epoch: float,
    priority_classes: list[dict[str, Any]],
) -> dict[str, Any]:
    validate_cpu_priority_inventory(priority_classes)
    configmap = build_watchdog_configmap(root, commit, binding, ready_at_epoch=ready_at_epoch)
    return {
        "objects": {
            "apiVersion": "v1",
            "kind": "List",
            "items": [configmap, build_watchdog_job(configmap)],
        },
        "server_launch_authorized": False,
        "watchdog_launch_authorized": True,
        "qualification_launch_authorized": False,
        "scored_launch_authorized": False,
    }


def build_job(configmap: dict[str, Any], authorization: dict[str, Any]) -> dict[str, Any]:
    seed = {
        "qualification_launch_authorized": True,
        "scored_successor_launch_authorized": False,
    }
    seed["receipt_sha256"] = crypto.digest_without(seed, "receipt_sha256")
    old_configmap = {
        "data": {
            "package.json": json.dumps({"package_commit": "0" * 40}),
        }
    }
    job = copy.deepcopy(prior.build_job(old_configmap, seed))
    job["metadata"]["name"] = qualifier.JOB_NAME
    job["metadata"]["labels"]["cyber-post-train.fleet.ai/owner"] = "chris"
    job["metadata"]["labels"]["cyber-post-train.fleet.ai/experiment"] = qualifier.JOB_NAME
    job["metadata"]["labels"]["kueue.x-k8s.io/queue-name"] = "training-lq"
    job["metadata"]["annotations"]["cyber-post-train.fleet.ai/authorization-receipt-sha256"] = (
        authorization["receipt_sha256"]
    )
    job["spec"]["activeDeadlineSeconds"] = 1800
    pod = job["spec"]["template"]
    pod["spec"]["priorityClassName"] = qualifier.CPU_PRIORITY_CLASS
    pod["spec"]["preemptionPolicy"] = "Never"
    pod["metadata"]["labels"]["cyber-post-train.fleet.ai/owner"] = "chris"
    pod["metadata"]["labels"]["cyber-post-train.fleet.ai/experiment"] = qualifier.JOB_NAME
    pod["metadata"]["labels"]["kueue.x-k8s.io/queue-name"] = "training-lq"
    package_commit = json.loads(configmap["data"]["package.json"])["package_commit"]
    for row in pod["spec"]["containers"][0]["env"]:
        if row["name"] == "QUALIFICATION_PACKAGE_COMMIT":
            row["value"] = package_commit
        elif row["name"] == "QUALIFICATION_OUTPUT_ROOT":
            row["value"] = str(qualifier.RESULT_ROOT)
    pod["spec"]["containers"][0]["env"].extend(
        [
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
        ]
    )
    pod["spec"]["containers"][0]["env"].append(
        {
            "name": "DEDICATED_SERVICE_ORIGIN",
            "value": authorization["server_binding"]["service_origin"],
        }
    )
    for volume in pod["spec"]["volumes"]:
        if volume["name"] == "package":
            volume["configMap"]["name"] = CONFIGMAP_NAME
        elif volume["name"] == "authorization":
            volume["configMap"]["name"] = AUTHORIZATION_CONFIGMAP_NAME
    return job


def render(root: Path, commit: str, authorization: dict[str, Any]) -> dict[str, Any]:
    configmap = build_configmap(root, commit)
    auth = build_authorization_configmap(authorization)
    job = build_job(configmap, authorization)
    objects = {"apiVersion": "v1", "kind": "List", "items": [configmap, auth, job]}
    return {
        "objects": objects,
        "package_sha256": json.loads(configmap["data"]["package.json"])["package_sha256"],
        "server_launch_authorized": False,
        "qualification_launch_authorized": True,
        "scored_launch_authorized": False,
    }
