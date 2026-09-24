"""Render one zero-GPU, read-only SFS probe for the Miles96 model inputs.

The packet is inert: this module never calls Kubernetes.  It embeds the exact
inventory implementation and a tiny driver in an immutable ConfigMap so an
operator can perform duplicate checks and two server dry-runs before deciding
whether to create the Job exactly once.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

SCHEMA = "cyber_qwen38_miles96_binding_probe_packet_v1"
RECEIPT_SCHEMA = "cyber_qwen38_miles96_binding_probe_receipt_v1"
LOG_PREFIX = "CYBER_MILES96_BINDING_PROBE="
CONTEXT = "nebius-mk8s-fleetai-training-e04zw4ye1k7wczqdw6"
NAMESPACE = "fleet-train-jobs"
JOB_NAME = "chris-q38-m96-bind-probe-a3"
CONFIG_MAP_NAME = JOB_NAME + "-code"
IMAGE = (
    "661864827319.dkr.ecr.us-east-1.amazonaws.com/fleet/miles-trainer@sha256:"
    "ee273bee346ad8e1cea63d18026b2bc5703bbd2e14d3c65efbbd74f19854874a"
)
HF_SOURCE = "/mnt/sfs/models/qwen3.8-27b-1d4bf0f2"
MEGATRON_SOURCE = "/mnt/sfs/jobs/chris-cpt-cleanup-q38-miles-base-v1/torch-dist"
PAIRED_CANDIDATES = tuple(
    ["/mnt/sfs/jobs/chris-cpt-cleanup-q38-rl-inputs-v1/prepare-v2/miles/prepared"]
    + [f"/mnt/sfs/jobs/chris-cpt-cleanup-q38-rl-inputs-v{i}/miles/prepared" for i in range(2, 8)]
)


DRIVER = "from training.miles96_binding_probe import runtime_main;runtime_main()\n"


def _sha(data: str) -> str:
    return "sha256:" + hashlib.sha256(data.encode()).hexdigest()


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _safe(callable_) -> dict[str, Any]:
    try:
        return {"status": "valid", **callable_()}
    except Exception as exc:
        return {"status": "invalid", "error_class": type(exc).__name__}


def _aggregate(root: Path, records: list[dict[str, Any]], mechanics: Any) -> dict[str, Any]:
    body = {"root": str(root), "files": records}
    return {
        "root": str(root),
        "file_count": len(records),
        "bytes": sum(row["bytes"] for row in records),
        "inventory_sha256": "sha256:" + mechanics.digest(body),
    }


def _paired_candidate(root: Path, mechanics: Any) -> dict[str, Any]:
    try:
        root.lstat()
    except FileNotFoundError:
        return {"root": str(root), "status": "absent"}
    except OSError as exc:
        return {"root": str(root), "status": "inaccessible", "error_class": type(exc).__name__}
    observed = _safe(lambda: mechanics.prepared_model_inventory(root))
    if observed["status"] != "valid":
        return {"root": str(root), **observed}
    return {
        "root": str(root),
        "status": "valid",
        "prepared_model_binding_sha256": observed["sha256"],
        "hf_file_count": len(observed["hf_files"]),
        "megatron_file_count": len(observed["megatron_files"]),
    }


def runtime_main() -> None:
    from training import miles96_mechanics_canary as mechanics

    if os.getuid() != 1000 or os.getgid() != 100:
        raise ValueError("probe identity drift")
    if (
        os.environ.get("CUDA_VISIBLE_DEVICES") != ""
        or os.environ.get("NVIDIA_VISIBLE_DEVICES") != "none"
    ):
        raise ValueError("probe has a visible GPU")

    def hf_source() -> dict[str, Any]:
        root = Path(HF_SOURCE)
        if root.is_symlink() or not root.is_dir() or not (root / "config.json").is_file():
            raise ValueError("HF source is absent or unsafe")
        mechanics._hf_index(root)
        return _aggregate(
            root,
            mechanics._all_file_records(root, exclude_complete_model_markers=False),
            mechanics,
        )

    def megatron_source() -> dict[str, Any]:
        root = Path(MEGATRON_SOURCE)
        tracker = root / "latest_checkpointed_iteration.txt"
        if (
            root.is_symlink()
            or not root.is_dir()
            or tracker.is_symlink()
            or not tracker.is_file()
            or tracker.read_text().strip() != "release"
        ):
            raise ValueError("Megatron source is absent or unsafe")
        result = _aggregate(
            root,
            mechanics._all_file_records(root, exclude_complete_model_markers=False),
            mechanics,
        )
        return {**result, "iteration": "release"}

    hf, megatron = _safe(hf_source), _safe(megatron_source)
    candidates = [_paired_candidate(Path(value), mechanics) for value in PAIRED_CANDIDATES]
    same_device = False
    if hf["status"] == megatron["status"] == "valid":
        same_device = Path(HF_SOURCE).stat().st_dev == Path(MEGATRON_SOURCE).stat().st_dev
    valid = [row for row in candidates if row["status"] == "valid"]
    body = {
        "schema": RECEIPT_SCHEMA,
        "status": "paired_candidate_found" if valid else "staging_required",
        "checked_at_epoch": int(time.time()),
        "hf_source": hf,
        "megatron_source": megatron,
        "sources_share_filesystem": same_device,
        "paired_candidates": candidates,
        "raw_weights_prompts_traces_or_scores_included": False,
    }
    receipt = {**body, "sha256": "sha256:" + mechanics.digest(body)}
    print(LOG_PREFIX + json.dumps(receipt, sort_keys=True, separators=(",", ":")), flush=True)


def build_packet() -> dict[str, Any]:
    mechanics = Path(__file__).with_name("miles96_mechanics_canary.py").read_text()
    module = Path(__file__).read_text()
    sources = {
        "training_init.py": "",
        "mechanics.py": mechanics,
        "probe_module.py": module,
        "driver.py": DRIVER,
    }
    source_sha = {name: _sha(value) for name, value in sources.items()}
    labels = {
        "cyber-post-train.fleet.ai/owner": "chris",
        "cyber-post-train.fleet.ai/role": "miles96-binding-probe",
        "kueue.x-k8s.io/queue-name": "training-lq",
        "kueue.x-k8s.io/priority-class": "q1",
    }
    annotations = {
        "fleet.ai/failure-alerts": "off",
        "cyber-post-train.fleet.ai/create-once": "true",
        "cyber-post-train.fleet.ai/source-sha256": _sha(
            json.dumps(source_sha, sort_keys=True, separators=(",", ":"))
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
            "activeDeadlineSeconds": 3600,
            "backoffLimit": 0,
            "completions": 1,
            "parallelism": 1,
            "suspend": True,
            "template": {
                "metadata": {"labels": labels, "annotations": annotations},
                "spec": {
                    "activeDeadlineSeconds": 3600,
                    "automountServiceAccountToken": False,
                    "containers": [
                        {
                            "name": "probe",
                            "image": IMAGE,
                            "imagePullPolicy": "IfNotPresent",
                            "command": [
                                "python",
                                "-I",
                                "-c",
                                "import runpy,sys;sys.path.insert(0,'/probe');"
                                "runpy.run_path('/probe/driver.py',run_name='__main__')",
                            ],
                            "env": [
                                {"name": "PYTHONPATH", "value": "/probe"},
                                {"name": "CUDA_VISIBLE_DEVICES", "value": ""},
                                {"name": "NVIDIA_VISIBLE_DEVICES", "value": "none"},
                                {"name": "WANDB_MODE", "value": "disabled"},
                            ],
                            "resources": {
                                "requests": {"cpu": "2", "memory": "8Gi"},
                                "limits": {"cpu": "4", "memory": "16Gi"},
                            },
                            "securityContext": {
                                "allowPrivilegeEscalation": False,
                                "capabilities": {"drop": ["ALL"]},
                                "privileged": False,
                                "readOnlyRootFilesystem": True,
                                "runAsNonRoot": True,
                            },
                            "volumeMounts": [
                                {"name": "code", "mountPath": "/probe", "readOnly": True},
                                {"name": "sfs", "mountPath": "/mnt/sfs", "readOnly": True},
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
                    "securityContext": {
                        "runAsNonRoot": True,
                        "runAsUser": 1000,
                        "runAsGroup": 100,
                        "fsGroup": 100,
                        "seccompProfile": {"type": "RuntimeDefault"},
                    },
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
                                        "key": "probe_module.py",
                                        "path": "training/miles96_binding_probe.py",
                                    },
                                    {"key": "driver.py", "path": "driver.py"},
                                ],
                            },
                        },
                        {
                            "name": "sfs",
                            "persistentVolumeClaim": {"claimName": "sfs-shared", "readOnly": True},
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
        "source_sha256": source_sha,
        "expected": {
            "gpus": 0,
            "priority_class": "c1",
            "queue_priority": "q1",
            "failure_alerts": "off",
            "backoff_limit": 0,
            "sfs_read_only": True,
            "external_writes": 0,
        },
        "precreate": {
            "exact_name_duplicate_census_required": True,
            "server_dry_run_count": 2,
            "stable_preview_digests_must_match": True,
            "create_request_count": 1,
            "automatic_create_retry": False,
        },
        "bundle": {"apiVersion": "v1", "kind": "List", "items": [config_map, job]},
    }
    return {**body, "sha256": "sha256:" + digest(body)}


def validate_packet(packet: dict[str, Any]) -> dict[str, Any]:
    expected = build_packet()
    if packet != expected:
        raise ValueError("Miles96 binding probe packet differs from the current renderer")
    job = packet["bundle"]["items"][1]
    pod = job["spec"]["template"]["spec"]
    if (
        job["metadata"]["annotations"].get("fleet.ai/failure-alerts") != "off"
        or job["spec"].get("backoffLimit") != 0
        or pod.get("priorityClassName") != "c1"
        or any(
            "nvidia.com/gpu" in values
            for container in pod.get("initContainers", []) + pod["containers"]
            for values in container.get("resources", {}).values()
        )
        or any(
            mount.get("readOnly") is not True
            for container in pod["containers"]
            for mount in container["volumeMounts"]
            if mount["name"] == "sfs"
        )
    ):
        raise ValueError("Miles96 binding probe safety contract drifted")
    return {
        "packet_sha256": packet["sha256"],
        "bundle_sha256": "sha256:" + digest(packet["bundle"]),
        "job": JOB_NAME,
        "config_map": CONFIG_MAP_NAME,
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
    descriptor = args.output.open("x")
    with descriptor:
        descriptor.write(encoded)


if __name__ == "__main__":
    main()
