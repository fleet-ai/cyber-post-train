"""Render one create-once CPU Job that pairs exact HF and Megatron inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

from training import miles96_mechanics_canary as mechanics

SCHEMA = "cyber_qwen38_miles96_model_stage_packet_v2"
RECEIPT_SCHEMA = "cyber_qwen38_miles96_model_stage_receipt_v1"
LOG_PREFIX = "CYBER_MILES96_MODEL_STAGE="
CONTEXT = "nebius-mk8s-fleetai-training-e04zw4ye1k7wczqdw6"
NAMESPACE = "fleet-train-jobs"
JOB_NAME = "chris-q38-m96-model-stage-a2"
CONFIG_MAP_NAME = JOB_NAME + "-code"
IMAGE = mechanics.IMAGE
HF_SOURCE = Path("/source/hf")
MEGATRON_SOURCE = Path("/source/megatron")
DESTINATION = Path("/mnt/sfs/jobs/chris-q38-m96-prepared-v1")
PARTIAL = Path("/mnt/sfs/jobs/.chris-q38-m96-prepared-v1.partial-a2")
MANIFEST = "PREPARED_MODEL.json"
COMPLETE = ".complete"
DRIVER = "from training.miles96_model_stage import runtime_main;runtime_main()\n"
ROOT_ACCESS_JUSTIFICATION = (
    "the exact read-only HF and Megatron source roots reject uid 1000; uid 0 is used only "
    "to inventory and copy those immutable sources into one fresh SFS destination"
)


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _sha(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def _source_inventory(root: Path, *, hf: bool) -> dict[str, Any]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("source is absent or unsafe")
    if hf:
        mechanics._hf_index(root)
    else:
        tracker = root / "latest_checkpointed_iteration.txt"
        if (
            tracker.is_symlink()
            or not tracker.is_file()
            or tracker.read_text().strip() != "release"
        ):
            raise ValueError("Megatron source is incomplete")
    files = mechanics._all_file_records(root, exclude_complete_model_markers=False)
    if not files:
        raise ValueError("source inventory is empty")
    body = {"files": files}
    return {
        "files": files,
        "file_count": len(files),
        "bytes": sum(row["bytes"] for row in files),
        "sha256": "sha256:" + mechanics.digest(body),
    }


def _copy_tree(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("staging child already exists")
    shutil.copytree(source, destination, symlinks=False, copy_function=shutil.copy2)


def _write_once(path: Path, value: str) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def stage() -> dict[str, Any]:
    if DESTINATION.exists() or DESTINATION.is_symlink():
        raise FileExistsError("prepared-model destination already exists")
    if PARTIAL.exists() or PARTIAL.is_symlink():
        raise FileExistsError("prepared-model partial destination already exists")
    hf_before = _source_inventory(HF_SOURCE, hf=True)
    megatron_before = _source_inventory(MEGATRON_SOURCE, hf=False)
    PARTIAL.mkdir(mode=0o755)
    _copy_tree(HF_SOURCE, PARTIAL / "Qwen3.8-27B")
    _copy_tree(MEGATRON_SOURCE, PARTIAL / "qwen3.8-27B_torch_dist")
    hf_after = _source_inventory(HF_SOURCE, hf=True)
    megatron_after = _source_inventory(MEGATRON_SOURCE, hf=False)
    if hf_before != hf_after or megatron_before != megatron_after:
        raise ValueError("source changed while staging")
    prepared = mechanics.prepared_model_inventory(PARTIAL)
    if prepared["hf_files"] != hf_before["files"]:
        raise ValueError("staged HF bytes differ from source")
    if prepared["megatron_files"] != megatron_before["files"]:
        raise ValueError("staged Megatron bytes differ from source")
    manifest_body = {
        "schema": mechanics.PREPARED_MODEL_SCHEMA,
        "prepared_model_binding_sha256": prepared["sha256"],
        "hf_source_inventory_sha256": hf_before["sha256"],
        "megatron_source_inventory_sha256": megatron_before["sha256"],
        "hf_file_count": hf_before["file_count"],
        "megatron_file_count": megatron_before["file_count"],
        "hf_bytes": hf_before["bytes"],
        "megatron_bytes": megatron_before["bytes"],
    }
    manifest = {**manifest_body, "sha256": "sha256:" + digest(manifest_body)}
    _write_once(PARTIAL / MANIFEST, json.dumps(manifest, sort_keys=True) + "\n")
    _write_once(PARTIAL / COMPLETE, manifest["sha256"] + "\n")
    mechanics._rename_noreplace(PARTIAL, DESTINATION)
    observed = mechanics.prepared_model_inventory(DESTINATION)
    if observed["sha256"] != prepared["sha256"]:
        raise ValueError("prepared-model binding changed after atomic promotion")
    return manifest


def runtime_main() -> None:
    started = int(time.time())
    try:
        manifest = stage()
        body = {
            "schema": RECEIPT_SCHEMA,
            "status": "succeeded",
            "prepared_model_root": str(DESTINATION),
            "prepared_model_binding_sha256": manifest["prepared_model_binding_sha256"],
            "manifest_sha256": manifest["sha256"],
            "hf_source_inventory_sha256": manifest["hf_source_inventory_sha256"],
            "megatron_source_inventory_sha256": manifest["megatron_source_inventory_sha256"],
            "hf_file_count": manifest["hf_file_count"],
            "megatron_file_count": manifest["megatron_file_count"],
            "hf_bytes": manifest["hf_bytes"],
            "megatron_bytes": manifest["megatron_bytes"],
            "started_at_epoch": started,
            "finished_at_epoch": int(time.time()),
            "gpus": 0,
            "container_uid": 0,
            "root_access_justification": ROOT_ACCESS_JUSTIFICATION,
            "root_filesystem_read_only": True,
            "source_mounts_read_only": True,
            "linux_capabilities": [],
            "privileged": False,
            "weights_prompts_traces_flags_answers_scores_or_credentials_included": False,
        }
        receipt = {**body, "sha256": "sha256:" + digest(body)}
        print(LOG_PREFIX + json.dumps(receipt, sort_keys=True, separators=(",", ":")), flush=True)
    except Exception as exc:
        body = {
            "schema": RECEIPT_SCHEMA,
            "status": "failed",
            "error_class": type(exc).__name__,
            "started_at_epoch": started,
            "finished_at_epoch": int(time.time()),
            "gpus": 0,
            "container_uid": 0,
            "root_access_justification": ROOT_ACCESS_JUSTIFICATION,
            "root_filesystem_read_only": True,
            "source_mounts_read_only": True,
            "linux_capabilities": [],
            "privileged": False,
            "weights_prompts_traces_flags_answers_scores_or_credentials_included": False,
        }
        receipt = {**body, "sha256": "sha256:" + digest(body)}
        print(LOG_PREFIX + json.dumps(receipt, sort_keys=True, separators=(",", ":")), flush=True)
        raise SystemExit(1) from None


def build_packet() -> dict[str, Any]:
    sources = {
        "training_init.py": "",
        "mechanics.py": Path(mechanics.__file__).read_text(),
        "stage_module.py": Path(__file__).read_text(),
        "driver.py": DRIVER,
    }
    labels = {
        "cyber-post-train.fleet.ai/owner": "chris",
        "cyber-post-train.fleet.ai/role": "miles96-model-stage",
        "kueue.x-k8s.io/queue-name": "training-lq",
        "kueue.x-k8s.io/priority-class": "q1",
    }
    annotations = {
        "fleet.ai/failure-alerts": "off",
        "cyber-post-train.fleet.ai/create-once": "true",
        "cyber-post-train.fleet.ai/source-sha256": _sha(
            json.dumps(sources, sort_keys=True, separators=(",", ":"))
        ),
    }
    config_map = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": CONFIG_MAP_NAME, "namespace": NAMESPACE},
        "immutable": True,
        "data": sources,
    }
    job = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": JOB_NAME,
            "namespace": NAMESPACE,
            "labels": labels,
            "annotations": annotations,
        },
        "spec": {
            "activeDeadlineSeconds": 7200,
            "backoffLimit": 0,
            "completions": 1,
            "parallelism": 1,
            "suspend": True,
            "template": {
                "metadata": {"labels": labels, "annotations": annotations},
                "spec": {
                    "activeDeadlineSeconds": 7200,
                    "automountServiceAccountToken": False,
                    "containers": [
                        {
                            "name": "stage",
                            "image": IMAGE,
                            "imagePullPolicy": "IfNotPresent",
                            "command": [
                                "python",
                                "-I",
                                "-c",
                                "import runpy,sys;sys.path.insert(0,'/stage');"
                                "runpy.run_path('/stage/driver.py',run_name='__main__')",
                            ],
                            "env": [
                                {"name": "CUDA_VISIBLE_DEVICES", "value": ""},
                                {"name": "NVIDIA_VISIBLE_DEVICES", "value": "none"},
                                {"name": "WANDB_MODE", "value": "disabled"},
                            ],
                            "resources": {
                                "requests": {"cpu": "8", "memory": "32Gi"},
                                "limits": {"cpu": "16", "memory": "64Gi"},
                            },
                            "securityContext": {
                                "allowPrivilegeEscalation": False,
                                "capabilities": {"drop": ["ALL"]},
                                "privileged": False,
                                "readOnlyRootFilesystem": True,
                                "runAsNonRoot": False,
                                "runAsUser": 0,
                                "runAsGroup": 0,
                            },
                            "volumeMounts": [
                                {"name": "code", "mountPath": "/stage", "readOnly": True},
                                {
                                    "name": "hf",
                                    "mountPath": str(HF_SOURCE),
                                    "readOnly": True,
                                    "subPath": "models/qwen3.8-27b-1d4bf0f2",
                                },
                                {
                                    "name": "megatron",
                                    "mountPath": str(MEGATRON_SOURCE),
                                    "readOnly": True,
                                    "subPath": (
                                        "jobs/chris-cpt-cleanup-q38-miles-base-v1/torch-dist"
                                    ),
                                },
                                {
                                    "name": "jobs",
                                    "mountPath": "/mnt/sfs/jobs",
                                    "subPath": "jobs",
                                },
                            ],
                        }
                    ],
                    "hostIPC": False,
                    "hostNetwork": False,
                    "hostPID": False,
                    "imagePullSecrets": [{"name": "ecr-pull"}],
                    "nodeSelector": {
                        "kubernetes.io/arch": "amd64",
                        "workload": "fleetai-training-ng-cpu",
                    },
                    "priorityClassName": "c1",
                    "restartPolicy": "Never",
                    "securityContext": {"seccompProfile": {"type": "RuntimeDefault"}},
                    "tolerations": [
                        {
                            "key": "workload",
                            "operator": "Equal",
                            "value": "fleetai-training-ng-cpu",
                            "effect": "NoSchedule",
                        }
                    ],
                    "volumes": [
                        {
                            "name": "code",
                            "configMap": {
                                "name": CONFIG_MAP_NAME,
                                "items": [
                                    {"key": "training_init.py", "path": "training/__init__.py"},
                                    {
                                        "key": "mechanics.py",
                                        "path": "training/miles96_mechanics_canary.py",
                                    },
                                    {
                                        "key": "stage_module.py",
                                        "path": "training/miles96_model_stage.py",
                                    },
                                    {"key": "driver.py", "path": "driver.py"},
                                ],
                            },
                        },
                        {
                            "name": "hf",
                            "persistentVolumeClaim": {"claimName": "sfs-shared", "readOnly": True},
                        },
                        {
                            "name": "megatron",
                            "persistentVolumeClaim": {"claimName": "sfs-shared", "readOnly": True},
                        },
                        {
                            "name": "jobs",
                            "persistentVolumeClaim": {"claimName": "sfs-shared"},
                        },
                    ],
                },
            },
        },
    }
    body = {
        "schema": SCHEMA,
        "context": CONTEXT,
        "namespace": NAMESPACE,
        "job_name": JOB_NAME,
        "config_map_name": CONFIG_MAP_NAME,
        "destination": str(DESTINATION),
        "partial": str(PARTIAL),
        "expected": {
            "gpus": 0,
            "priority_class": "c1",
            "queue_priority": "q1",
            "failure_alerts": "off",
            "backoff_limit": 0,
            "source_mounts_read_only": True,
            "container_uid": 0,
            "root_access_justification": ROOT_ACCESS_JUSTIFICATION,
            "root_filesystem_read_only": True,
            "linux_capabilities": [],
            "privileged": False,
            "destination_must_be_absent": True,
            "partial_must_be_absent": True,
        },
        "execution_sequence": {
            "exact_name_duplicate_census_required": True,
            "destination_absence_required": True,
            "partial_absence_required": True,
            "server_dry_run_count": 2,
            "stable_preview_digests_must_match": True,
            "config_map_create_request_count": 1,
            "config_map_create_retry_allowed": False,
            "job_create_request_count": 1,
            "job_create_retry_allowed": False,
            "job_created_suspended": True,
            "post_create_pre_unsuspend_checks": [
                "exact_config_map_uid_bound",
                "exact_job_uid_bound",
                "rendered_root_failure_alerts_off",
                "rendered_priority_class_c1",
                "rendered_gpu_requests_and_limits_zero",
                "rendered_suspend_true",
            ],
            "unsuspend": {
                "request_count": 1,
                "retry_allowed": False,
                "content_type": "application/json-patch+json",
                "uid_source": "exact_job_create_response.metadata.uid",
                "operations_template": [
                    {"op": "test", "path": "/metadata/uid", "value": "$JOB_UID"},
                    {"op": "test", "path": "/spec/suspend", "value": True},
                    {"op": "replace", "path": "/spec/suspend", "value": False},
                ],
            },
            "post_unsuspend_create_or_patch_requests_allowed": False,
            "exact_uid_terminal_monitor_required": True,
            "exact_uid_cleanup_required": True,
        },
        "bundle": {"apiVersion": "v1", "kind": "List", "items": [config_map, job]},
    }
    return {**body, "sha256": "sha256:" + digest(body)}


def validate_packet(packet: dict[str, Any]) -> dict[str, Any]:
    if packet != build_packet():
        raise ValueError("Miles96 model-stage packet differs from renderer")
    job = packet["bundle"]["items"][1]
    pod = job["spec"]["template"]["spec"]
    container = pod["containers"][0]
    mounts = {item["name"]: item for item in container["volumeMounts"]}
    if (
        job["metadata"]["annotations"].get("fleet.ai/failure-alerts") != "off"
        or job["spec"].get("backoffLimit") != 0
        or pod.get("priorityClassName") != "c1"
        or mounts["hf"].get("readOnly") is not True
        or mounts["megatron"].get("readOnly") is not True
        or container.get("securityContext", {}).get("runAsUser") != 0
        or container.get("securityContext", {}).get("readOnlyRootFilesystem") is not True
        or container.get("securityContext", {}).get("privileged") is not False
        or container.get("securityContext", {}).get("capabilities") != {"drop": ["ALL"]}
        or job["spec"].get("suspend") is not True
        or "nvidia.com/gpu" in json.dumps(container.get("resources", {}))
    ):
        raise ValueError("Miles96 model-stage safety contract drifted")
    sequence = packet["execution_sequence"]
    if (
        sequence.get("config_map_create_request_count") != 1
        or sequence.get("config_map_create_retry_allowed") is not False
        or sequence.get("job_create_request_count") != 1
        or sequence.get("job_create_retry_allowed") is not False
        or sequence.get("job_created_suspended") is not True
        or sequence.get("unsuspend", {}).get("request_count") != 1
        or sequence.get("unsuspend", {}).get("retry_allowed") is not False
        or sequence.get("unsuspend", {}).get("operations_template")
        != [
            {"op": "test", "path": "/metadata/uid", "value": "$JOB_UID"},
            {"op": "test", "path": "/spec/suspend", "value": True},
            {"op": "replace", "path": "/spec/suspend", "value": False},
        ]
        or sequence.get("post_unsuspend_create_or_patch_requests_allowed") is not False
    ):
        raise ValueError("Miles96 model-stage create/release sequence drifted")
    return {
        "packet_sha256": packet["sha256"],
        "bundle_sha256": "sha256:" + digest(packet["bundle"]),
        "job": JOB_NAME,
        "config_map": CONFIG_MAP_NAME,
        "destination": str(DESTINATION),
        "gpus": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    packet = build_packet()
    validate_packet(packet)
    encoded = json.dumps(packet, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(encoded, end="")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(encoded)


if __name__ == "__main__":
    main()
