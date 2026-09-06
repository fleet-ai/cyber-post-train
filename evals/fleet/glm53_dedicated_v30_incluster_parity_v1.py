"""Held in-cluster DinD actual-OpenCode parity rail for the next GLM generation."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

from evals.fleet import exact_pass4_crypto as crypto
from evals.fleet import opencode_actual_harness_parity_v1 as parity
from evals.fleet import opencode_parity_support_v1 as support
from evals.fleet.glm53_dedicated_v22_concurrency_package_v1 import DIND_IMAGE, UV_IMAGE

SCHEMA = "fleet-glm53-dedicated-v30-incluster-parity-authorization-v1"
RESULT_SCHEMA = "fleet-glm53-dedicated-v30-incluster-parity-result-v1"
PACKAGE_SCHEMA = "fleet-glm53-dedicated-v30-incluster-parity-package-v1"
JOB_NAME = "chris-glm53-dedicated-v30-incluster-actual-parity-v1"
CONFIGMAP_NAME = JOB_NAME + "-package"
AUTHORIZATION_CONFIGMAP_NAME = JOB_NAME + "-authorization"
RESULT_ROOT = Path("/mnt/sfs/jobs") / JOB_NAME
SERVER_TITLE = "chris-cyber-evalserve-glm53-tp8-a-v30"
SERVER_RUN_DIR = "/mnt/sfs/jobs/chris-cyber-evalserve-glm53-tp8-a-v30"
SERVED_ID = "glm-5.3"
MODEL_REVISION = "30333038ada1f1dacb294a93270305a890b50c14"
CONTEXT_LENGTH = 262144
NAMESPACE = "fleet-train-jobs"
CPU_PRIORITY_CLASS = "fleet-serve-low"
FLEET_SECRET_NAME = "chris-cyber-opencode-evals-v2"
AUTH_MAX_AGE_SECONDS = 300
COMMIT_RE = re.compile(r"[0-9a-f]{40}")
FILES = (
    "evals/fleet/Dockerfile.opencode",
    "evals/fleet/configs/blackbox-ctf-tool-catalog-v1.json",
    "evals/fleet/exact_pass4_crypto.py",
    "evals/fleet/exact_pass4_universe.py",
    "evals/fleet/glm53_dedicated_v30_incluster_parity_v1.py",
    "evals/fleet/opencode_actual_harness_parity_v1.py",
    "evals/fleet/opencode_parity_support_v1.py",
    "evals/fleet/production_blackbox_tool_catalog_v1.py",
)
AUTH_KEYS = {
    "schema_version",
    "status",
    "observed_at_epoch",
    "server_binding",
    "server_ready_receipt_sha256",
    "watchdog_active_receipt_sha256",
    "job_identity_absent",
    "result_root_absent",
    "cpu_priority_class",
    "cpu_preemption_policy",
    "qualification_launch_authorized",
    "scored_launch_authorized",
    "fleet_task_instance_calls",
    "fleet_session_calls",
    "verifier_calls",
    "scoring_calls",
    "protected_content_included",
    "receipt_sha256",
}


class ParityPackageError(RuntimeError):
    """The in-cluster parity package or runtime failed closed."""


def _valid_uuid(value: Any) -> bool:
    try:
        return (
            str(uuid.UUID(str(value))) == value and value != "00000000-0000-0000-0000-000000000000"
        )
    except (ValueError, TypeError, AttributeError):
        return False


def validate_binding(binding: dict[str, Any]) -> None:
    keys = {
        "server_title",
        "server_run_dir",
        "api_run_id",
        "rayjob_uid",
        "workload_uid",
        "head_pod_uid",
        "service_uid",
        "service_origin",
        "served_id",
        "model_revision",
        "context_length",
    }
    if (
        set(binding) != keys
        or binding.get("server_title") != SERVER_TITLE
        or binding.get("server_run_dir") != SERVER_RUN_DIR
        or not re.fullmatch(r"ft-run-[0-9a-f]{8}", str(binding.get("api_run_id", "")))
        or any(
            not _valid_uuid(binding.get(field))
            for field in ("rayjob_uid", "workload_uid", "head_pod_uid", "service_uid")
        )
        or not re.fullmatch(
            r"http://[a-z0-9-]+-head-svc\.fleet-train-jobs\.svc:8000",
            str(binding.get("service_origin", "")),
        )
        or binding.get("served_id") != SERVED_ID
        or binding.get("model_revision") != MODEL_REVISION
        or binding.get("context_length") != CONTEXT_LENGTH
    ):
        raise ParityPackageError("v30_incluster_server_binding_invalid")


def canonical_binding(binding: dict[str, Any]) -> dict[str, Any]:
    validate_binding(binding)
    return {
        key: binding[key]
        for key in (
            "api_run_id",
            "rayjob_uid",
            "head_pod_uid",
            "service_uid",
            "served_id",
            "model_revision",
            "context_length",
        )
    }


def validate_authorization(value: dict[str, Any]) -> None:
    binding = value.get("server_binding")
    if not isinstance(binding, dict):
        raise ParityPackageError("v30_incluster_parity_authorization_invalid")
    validate_binding(binding)
    observed = value.get("observed_at_epoch")
    now = time.time()
    if (
        set(value) != AUTH_KEYS
        or value.get("schema_version") != SCHEMA
        or value.get("status") != "AUTHORIZED_SCORE_FREE_INCLUSTER_PARITY"
        or type(observed) not in {int, float}
        or observed > now + 5
        or now - observed > AUTH_MAX_AGE_SECONDS
        or any(
            not re.fullmatch(r"sha256:[0-9a-f]{64}", str(value.get(field, "")))
            for field in ("server_ready_receipt_sha256", "watchdog_active_receipt_sha256")
        )
        or value.get("job_identity_absent") is not True
        or value.get("result_root_absent") is not True
        or value.get("cpu_priority_class") != CPU_PRIORITY_CLASS
        or value.get("cpu_preemption_policy") != "Never"
        or value.get("qualification_launch_authorized") is not True
        or value.get("scored_launch_authorized") is not False
        or any(
            value.get(field) != 0
            for field in (
                "fleet_task_instance_calls",
                "fleet_session_calls",
                "verifier_calls",
                "scoring_calls",
            )
        )
        or value.get("protected_content_included") is not False
        or value.get("receipt_sha256") != crypto.digest_without(value, "receipt_sha256")
    ):
        raise ParityPackageError("v30_incluster_parity_authorization_invalid")


def validate_parity_receipt(value: dict[str, Any], binding: dict[str, Any]) -> None:
    execution = value.get("execution") or {}
    privacy = value.get("privacy") or {}
    if (
        value.get("receipt_sha256") != crypto.digest_without(value, "receipt_sha256")
        or value.get("schema_version") != parity.SCHEMA
        or value.get("status") != "PASSED_NON_SCORED"
        or value.get("classification") != "ACTUAL_HARNESS_PARITY"
        or value.get("endpoint", {}).get("origin") != binding["service_origin"]
        or value.get("endpoint", {}).get("kind") != "dedicated_uid_bound_inference"
        or value.get("endpoint", {}).get("server_binding") != canonical_binding(binding)
        or value.get("endpoint", {}).get("server_binding_sha256")
        != crypto.sha256(crypto.canonical_json(canonical_binding(binding)))
        or execution.get("harness_exit_code") != 0
        or execution.get("model_requests", 0) <= 0
        or execution.get("final_marker_observed") is not True
        or execution.get("docker_host_gateway_added") is not True
        or execution.get("task_instance_session_verifier_scoring_calls") != 0
        or execution.get("scored_launch_authorized") is not False
        or value.get("tool_contract", {}).get("calls_observed_in_order")
        != ["bash", "submit_report"]
        or any(item is not False for item in privacy.values())
    ):
        raise ParityPackageError("v30_incluster_parity_receipt_invalid")


def execute(authorization: dict[str, Any], *, job_uid: str, pod_uid: str) -> dict[str, Any]:
    validate_authorization(authorization)
    if not _valid_uuid(job_uid) or not _valid_uuid(pod_uid):
        raise ParityPackageError("v30_incluster_parity_runtime_identity_invalid")
    if RESULT_ROOT.exists() or RESULT_ROOT.is_symlink():
        raise ParityPackageError("v30_incluster_parity_result_collision")
    RESULT_ROOT.mkdir(parents=True)
    binding = authorization["server_binding"]
    receipt = parity.run(
        "glm-5.3",
        os.environ.get("FLEET_API_KEY", ""),
        upstream_origin=binding["service_origin"],
        server_binding=canonical_binding(binding),
        docker_add_host_gateway=True,
        docker_network_host=True,
    )
    validate_parity_receipt(receipt, binding)
    support.write_json_once(RESULT_ROOT / "PARITY.json", receipt)
    body: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "status": "PASSED_NON_SCORED_INCLUSTER_PARITY",
        "server_binding_sha256": crypto.sha256(crypto.canonical_json(binding)),
        "authorization_receipt_sha256": authorization["receipt_sha256"],
        "parity_receipt_sha256": receipt["receipt_sha256"],
        "job_uid": job_uid,
        "pod_uid": pod_uid,
        "nested_container_network": "host",
        "local_proxy_bind_address": "127.0.0.1",
        "endpoint_origin": binding["service_origin"],
        "fleet_task_instance_calls": 0,
        "fleet_session_calls": 0,
        "verifier_calls": 0,
        "scoring_calls": 0,
        "scored_launch_authorized": False,
        "protected_content_included": False,
    }
    body["receipt_sha256"] = crypto.digest_without(body, "receipt_sha256")
    support.write_json_once(RESULT_ROOT / "RESULT.json", body)
    return body


def _source(root: Path, commit: str, path: str) -> bytes:
    if COMMIT_RE.fullmatch(commit) is None:
        raise ParityPackageError("v30_incluster_package_commit_invalid")
    try:
        return subprocess.run(
            ["git", "-C", str(root), "show", f"{commit}:{path}"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ParityPackageError(f"v30_incluster_package_source_absent:{path}") from exc


def _key(path: str) -> str:
    return path.replace("/", "__SLASH__")


def build_configmap(root: Path, commit: str) -> dict[str, Any]:
    data = {_key(path): _source(root, commit, path).decode() for path in FILES}
    manifest = {path: crypto.sha256(_source(root, commit, path)) for path in FILES}
    package: dict[str, Any] = {
        "schema_version": PACKAGE_SCHEMA,
        "package_commit": commit,
        "files": manifest,
        "job_name": JOB_NAME,
        "result_root": str(RESULT_ROOT),
        "server_title": SERVER_TITLE,
        "server_run_dir": SERVER_RUN_DIR,
        "nested_container_network": "host",
        "local_proxy_bind_address": "127.0.0.1",
        "score_free": True,
    }
    package["package_sha256"] = crypto.digest_without(package, "package_sha256")
    data["package.json"] = json.dumps(package, sort_keys=True, separators=(",", ":")) + "\n"
    data["run.sh"] = """#!/usr/bin/env bash
set -euo pipefail
test "${DOCKER_HOST:-}" = unix:///var/run/docker.sock
test ! -e "$PARITY_RESULT_ROOT"
mkdir -p /work/evals/fleet/configs
for source in /bootstrap/*__SLASH__*; do
  target="/work/$(basename "$source" | sed 's,__SLASH__,/,g')"
  install -D -m 0444 "$source" "$target"
done
for _ in $(seq 1 120); do docker info >/dev/null 2>&1 && break; sleep 1; done
docker info >/dev/null
docker build --pull --platform linux/amd64 \
  --tag chris/opencode:1.18.27-cyber-v1 \
  --file /work/evals/fleet/Dockerfile.opencode /work/evals/fleet
test "$(docker run --rm chris/opencode:1.18.27-cyber-v1 opencode --version)" = 1.18.27
export PYTHONPATH=/work
exec uv run --with httpx==0.28.1 --with pyyaml==6.0.2 \
  python -m evals.fleet.glm53_dedicated_v30_incluster_parity_v1 run \
  --authorization /authorization/authorization.json
"""
    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": CONFIGMAP_NAME, "namespace": NAMESPACE},
        "immutable": True,
        "data": data,
    }


def build_authorization_configmap(authorization: dict[str, Any]) -> dict[str, Any]:
    validate_authorization(authorization)
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
    validate_authorization(authorization)
    package = json.loads(configmap["data"]["package.json"])
    common = [
        {"name": "docker-socket", "mountPath": "/var/run"},
        {"name": "docker-data", "mountPath": "/var/lib/docker"},
        {"name": "workspace", "mountPath": "/workspace"},
        {"name": "sfs", "mountPath": "/mnt/sfs"},
    ]
    labels = {
        "cyber-post-train.fleet.ai/owner": "chris",
        "cyber-post-train.fleet.ai/experiment": JOB_NAME,
        "kueue.x-k8s.io/queue-name": "training-lq",
    }
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": JOB_NAME,
            "namespace": NAMESPACE,
            "labels": labels,
            "annotations": {
                "cyber-post-train.fleet.ai/create-once": "true",
                "cyber-post-train.fleet.ai/score-free": "true",
                "cyber-post-train.fleet.ai/package-sha256": package["package_sha256"],
                "cyber-post-train.fleet.ai/authorization-receipt-sha256": authorization[
                    "receipt_sha256"
                ],
            },
        },
        "spec": {
            "backoffLimit": 0,
            "activeDeadlineSeconds": 1200,
            "ttlSecondsAfterFinished": 604800,
            "template": {
                "metadata": {"labels": labels},
                "spec": {
                    "restartPolicy": "Never",
                    "priorityClassName": CPU_PRIORITY_CLASS,
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
                            "volumeMounts": common,
                        },
                    ],
                    "containers": [
                        {
                            "name": "parity",
                            "image": UV_IMAGE,
                            "command": ["bash", "/bootstrap/run.sh"],
                            "env": [
                                {
                                    "name": "PATH",
                                    "value": "/docker-cli:/usr/local/bin:/usr/bin:/bin",
                                },
                                {"name": "DOCKER_HOST", "value": "unix:///var/run/docker.sock"},
                                {"name": "DOCKER_TLS_CERTDIR", "value": ""},
                                {"name": "PARITY_RESULT_ROOT", "value": str(RESULT_ROOT)},
                                {"name": "PACKAGE_COMMIT", "value": package["package_commit"]},
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
                                            "name": FLEET_SECRET_NAME,
                                            "key": "FLEET_API_KEY",
                                        }
                                    },
                                },
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


def render(root: Path, commit: str, authorization: dict[str, Any]) -> dict[str, Any]:
    configmap = build_configmap(root, commit)
    return {
        "objects": {
            "apiVersion": "v1",
            "kind": "List",
            "items": [
                configmap,
                build_authorization_configmap(authorization),
                build_job(configmap, authorization),
            ],
        },
        "qualification_launch_authorized": True,
        "scored_launch_authorized": False,
    }


def build_held() -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema_version": "fleet-glm53-dedicated-v30-incluster-parity-held-v1",
        "status": "PASSED_HELD_NO_LAUNCH",
        "server_title": SERVER_TITLE,
        "server_run_dir": SERVER_RUN_DIR,
        "job_name": JOB_NAME,
        "result_root": str(RESULT_ROOT),
        "nested_container_network": "host",
        "local_proxy_bind_address": "127.0.0.1",
        "endpoint_origin_must_equal_internal_service_origin": True,
        "qualification_launch_authorized": False,
        "scored_launch_authorized": False,
        "api_mutation_calls": 0,
        "protected_content_included": False,
    }
    body["receipt_sha256"] = crypto.digest_without(body, "receipt_sha256")
    return body


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "plan"))
    parser.add_argument("--authorization", type=Path)
    args = parser.parse_args()
    if args.command == "plan":
        print(json.dumps(build_held(), sort_keys=True, separators=(",", ":")))
        return 0
    if args.authorization is None:
        parser.error("run requires --authorization")
    authorization = json.loads(args.authorization.read_text())
    result = execute(
        authorization,
        job_uid=os.environ.get("JOB_UID", ""),
        pod_uid=os.environ.get("POD_UID", ""),
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
