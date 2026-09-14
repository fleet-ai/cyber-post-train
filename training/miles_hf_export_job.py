"""Prepared dev execution rails for an exact Miles canary HF export and reload.

The CPU conversion is one direct, zero-GPU Kubernetes ``batch/v1`` Job.  The
independent HF reopen/probe is one Jobs API RayJob with exactly one GPU.  Both
stages carry an immutable runtime bundle and a create-once output root.
"""

from __future__ import annotations

import base64
import binascii
import gzip
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import time
from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path
from typing import Any

from cyber_post_train.jobs import API_URLS, bundled_request, canonical_gzip, digest, quantity

from . import miles
from .miles_conversion import _write

CONFIG_SCHEMA = "cyber_miles_hf_export_job_config_v2"
PLAN_SCHEMA = "cyber_miles_hf_export_job_plan_v2"
PREFLIGHT_SCHEMA = "cyber_miles_hf_export_job_cpu_preflight_v1"
SUBMISSION_SCHEMA = "cyber_miles_hf_export_job_submission_v1"
EXPORT_CONTROLLER_SCHEMA = "cyber_miles_hf_export_controller_terminal_v1"
EXPORT_RELEASE_SCHEMA = "cyber_miles_hf_export_external_release_v1"
EXPORT_ACCEPTED_SCHEMA = "cyber_miles_hf_export_job_accepted_v2"
DEV_KUBE_CONTEXT = "nebius-mk8s-fleetai-training-dev-e04p03enwk5c0va9tb"
DEV_NAMESPACE = "fleet-train-jobs"
DEV_NAMESPACE_UID = "10394b76-e1d4-40b1-a8e2-7575e95df216"
DEV_KUBERNETES_API = "kubernetes://" + DEV_KUBE_CONTEXT
DEV_QUEUE = "training-lq"
DEV_GPU_NODE_POOL = "fleetai-training-ng-gpu"
SFS_CLAIM = "sfs-shared"
STAGES = frozenset({"export", "reload"})
DEADLINES = {"export": 6 * 60 * 60, "reload": 60 * 60}
RESOURCES = {
    "export": {
        "cpu_request": "64",
        "cpu_limit": "64",
        "memory_request": "512Gi",
        "memory_limit": "768Gi",
    },
    "reload": {
        "cpu_request": "32",
        "cpu_limit": "64",
        "memory_request": "192Gi",
        "memory_limit": "256Gi",
    },
}
PRIORITIES = {"export": "c2", "reload": "c1"}
RUNTIME_FILES = (
    "training/miles_hf_export_job.py",
    "training/miles_hf_export.py",
    "training/io.py",
    "training/miles.py",
    "training/miles_conversion.py",
    "training/miles_reload.py",
    "training/miles_promotion.py",
    "training/post_sft_artifacts.py",
    "training/post_sft_base_surface.py",
    "training/post_sft_cast.py",
    "training/rl_runtime.py",
    "cyber_post_train/jobs.py",
)
_REFERENCE_FIELDS = {"path", "file_sha256", "receipt_sha256"}
_PLAN_FIELDS = {
    "schema",
    "stage",
    "run_name",
    "output_root",
    "artifact_path",
    "source",
    "runtime_sha256",
    "deadline_seconds",
    "execution",
    "privacy",
}
_SHA = re.compile(r"(?:sha256:)?[a-f0-9]{64}")
_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}")
_RUN_NAME = re.compile(r"[a-z0-9](?:[-a-z0-9]{0,29}[a-z0-9])?")
_COMMON_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "WANDB_MODE": "disabled",
    "WANDB_DISABLED": "true",
    "PYTHONUNBUFFERED": "1",
}


def _fields(value: str) -> set[str]:
    return set(value.split())  # noqa: SIM905 - compact reviewed evidence schemas


_EXPORT_CONTROLLER_FIELDS = _fields(
    "schema status stage cluster api_base_url kube_context namespace namespace_uid run_name "
    "source_plan_sha256 source_request_sha256 submission_binding_sha256 runtime_bundle_sha256 "
    "event_journal artifact_path artifact_file_sha256 artifact_receipt_sha256 api_run_id "
    "api_run_name job_name job_uid workload_name workload_uid workload_owner_job_uid "
    "pods api_status controller_status "
    "effective_priority automatic_requeue workers gpus_per_worker total_gpus observed_at sha256"
)
_EXPORT_RELEASE_FIELDS = _fields(
    "schema status stage cluster api_base_url kube_context namespace namespace_uid run_name "
    "source_plan_sha256 source_request_sha256 submission_binding_sha256 runtime_bundle_sha256 "
    "artifact_path artifact_file_sha256 artifact_receipt_sha256 controller_terminal_path "
    "controller_terminal_file_sha256 controller_terminal_sha256 api_run_id api_run_name "
    "job_name job_uid workload_name workload_uid pod_uids api_status controller_status "
    "job_present workload_present quota_reservation_present gpu_pods_present "
    "active_gpu_pod_uids active_gpus observed_at sha256"
)
_EXPORT_ACCEPTED_FIELDS = _fields(
    "schema status export_plan_path export_plan_file_sha256 export_plan_sha256 "
    "submission_binding_path submission_binding_file_sha256 submission_binding_sha256 "
    "artifact_path artifact_file_sha256 artifact_receipt_sha256 tensor_inventory_sha256 "
    "controller_terminal_path controller_terminal_file_sha256 controller_terminal_sha256 "
    "external_release_path external_release_file_sha256 external_release_sha256 "
    "source_plan_sha256 source_request_sha256 runtime_bundle_sha256 active_canary_binding "
    "exact_active_canary_source_verified "
    "exact_job_execution_verified external_gpu_release_verified create_once_verified "
    "serving_qualified sha256"
)


def _runtime() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    return {name: (root / name).read_text() for name in RUNTIME_FILES}


def _sha(value: object, label: str) -> str:
    result = str(value).removeprefix("sha256:")
    if re.fullmatch(r"[a-f0-9]{64}", result) is None:
        raise ValueError(f"{label} is not an exact SHA-256")
    return result


def _sfs(path: object, label: str) -> str:
    value = Path(str(path))
    if (
        not value.is_absolute()
        or value.parts[:4] != ("/", "mnt", "sfs", "jobs")
        or len(value.parts) < 5
        or ".." in value.parts
        or str(value) != str(path)
    ):
        raise ValueError(f"{label} must be a canonical per-run path under /mnt/sfs/jobs")
    return str(value)


def _reference(value: object, label: str) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != {"path", "file_sha256"}:
        raise ValueError(f"{label} reference must bind one exact file")
    path = _sfs(value["path"], label)
    if _SHA.fullmatch(str(value["file_sha256"])) is None:
        raise ValueError(f"{label} reference is not digest-bound")
    return {"path": path, "file_sha256": _sha(value["file_sha256"], label)}


def _bound_reference(value: object, label: str) -> None:
    if (
        not isinstance(value, dict)
        or set(value) != _REFERENCE_FIELDS
        or _sfs(value.get("path"), label) != value.get("path")
        or any(
            _SHA.fullmatch(str(value.get(key, ""))) is None for key in _REFERENCE_FIELDS - {"path"}
        )
    ):
        raise ValueError(f"{label} prepared reference changed")


def _disjoint(output: Path, sources: list[Path]) -> None:
    resolved = output.resolve()
    if any(
        resolved == source.resolve()
        or resolved.is_relative_to(source.resolve())
        or source.resolve().is_relative_to(resolved)
        for source in sources
    ):
        raise ValueError("HF job output must be disjoint from every immutable input")


def compile_job(config: dict[str, Any]) -> dict[str, Any]:
    """Compile and reopen one exact canary export or its one-GPU reload."""
    from . import miles_hf_export as hf

    if set(config) != {"schema", "stage", "name", "output_root", "source", "cluster"}:
        raise ValueError("Miles HF job configuration fields changed")
    stage = config.get("stage")
    cluster = config.get("cluster")
    if (
        config.get("schema") != CONFIG_SCHEMA
        or stage not in STAGES
        or _RUN_NAME.fullmatch(str(config.get("name", ""))) is None
        or not isinstance(cluster, dict)
        or cluster
        != {
            "target": "dev",
            "priority": PRIORITIES[str(stage)],
            "resources": RESOURCES[str(stage)],
        }
    ):
        raise ValueError("Miles HF job is not the exact dev resource profile")
    output_root = _sfs(config["output_root"], "HF job output root")
    source = config.get("source")
    if not isinstance(source, dict):
        raise ValueError("Miles HF job source is absent")
    if stage == "export":
        if set(source) != {
            "active_canary_binding",
            "checkpoint",
            "terminal_acceptance",
            "native_reload_acceptance",
        }:
            raise ValueError("Miles HF export source fields changed")
        refs = {name: _reference(value, name) for name, value in source.items()}
        bound = hf.bind_source(
            active_canary_binding_path=Path(refs["active_canary_binding"]["path"]),
            active_canary_binding_sha256=refs["active_canary_binding"]["file_sha256"],
            checkpoint_path=Path(refs["checkpoint"]["path"]),
            checkpoint_sha256=refs["checkpoint"]["file_sha256"],
            terminal_path=Path(refs["terminal_acceptance"]["path"]),
            terminal_sha256=refs["terminal_acceptance"]["file_sha256"],
            native_reload_path=Path(refs["native_reload_acceptance"]["path"]),
            native_reload_sha256=refs["native_reload_acceptance"]["file_sha256"],
        )
        plan_source = {key: bound[key] for key in refs}
        plan_source.update(
            source_plan_sha256=bound["source_plan_sha256"],
            prediction_probe=bound["prediction_probe"],
        )
        artifact_path = str(Path(output_root) / "model")
        protected = [
            Path(bound["checkpoint_manifest"]["root"]),
            Path(bound["checkpoint_manifest"]["model"]["root"]),
        ]
    else:
        if set(source) != {"export_acceptance"}:
            raise ValueError("Miles HF reload source fields changed")
        reference = _reference(source["export_acceptance"], "accepted HF export")
        accepted, accepted_file_sha256 = _snapshot(Path(reference["path"]))
        if accepted_file_sha256 != reference["file_sha256"]:
            raise ValueError("accepted HF export file digest changed")
        validated = validate_export_accepted(accepted, check_files=True)
        plan_source = {
            "export_acceptance": {
                **reference,
                "receipt_sha256": accepted["sha256"].removeprefix("sha256:"),
            },
            "export": validated["export"],
        }
        artifact_path = str(Path(output_root) / "HF_RELOAD_VALIDATED.json")
        protected = [Path(reference["path"]).parent, Path(validated["export"]["path"]).parent]
    _disjoint(Path(output_root), protected)
    plan = {
        "schema": PLAN_SCHEMA,
        "stage": stage,
        "run_name": config["name"],
        "output_root": output_root,
        "artifact_path": artifact_path,
        "source": plan_source,
        "runtime_sha256": digest(_runtime()),
        "deadline_seconds": DEADLINES[str(stage)],
        "execution": _execution(str(stage)),
        "privacy": {
            "secrets": [],
            "network_downloads": False,
            "reward_values": False,
            "task_content": False,
            "benchmark_feedback": False,
        },
    }
    validate_plan(plan, check_files=True)
    job_request(plan)
    return plan


def validate_plan(
    plan: Mapping[str, Any], *, check_files: bool, validate_historical: bool = True
) -> None:
    from . import miles_hf_export as hf

    stage = plan.get("stage")
    execution = plan.get("execution")
    if (
        set(plan) != _PLAN_FIELDS
        or plan.get("schema") != PLAN_SCHEMA
        or stage not in STAGES
        or _RUN_NAME.fullmatch(str(plan.get("run_name", ""))) is None
        or _sfs(plan.get("output_root"), "HF job output root") != plan.get("output_root")
        or plan.get("artifact_path")
        != str(
            Path(str(plan.get("output_root")))
            / ("model" if stage == "export" else "HF_RELOAD_VALIDATED.json")
        )
        or plan.get("runtime_sha256") != digest(_runtime())
        or plan.get("deadline_seconds") != DEADLINES[str(stage)]
        or execution != _execution(str(stage))
        or plan.get("privacy")
        != {
            "secrets": [],
            "network_downloads": False,
            "reward_values": False,
            "task_content": False,
            "benchmark_feedback": False,
        }
    ):
        raise ValueError("Miles HF prepared plan drift")
    source = plan.get("source")
    if not isinstance(source, dict):
        raise ValueError("Miles HF prepared source is absent")
    if stage == "export":
        if set(source) != {
            "checkpoint",
            "terminal_acceptance",
            "native_reload_acceptance",
            "source_plan_sha256",
            "active_canary_binding",
            "prediction_probe",
        }:
            raise ValueError("Miles HF prepared export source fields changed")
        for key in (
            "active_canary_binding",
            "checkpoint",
            "terminal_acceptance",
            "native_reload_acceptance",
        ):
            _bound_reference(source.get(key), key)
        hf._validate_prediction_probe(source.get("prediction_probe"))
        if _SHA.fullmatch(str(source.get("source_plan_sha256", ""))) is None:
            raise ValueError("Miles HF export source plan is not digest-bound")
        if check_files:
            bound = hf.bind_source(
                active_canary_binding_path=Path(source["active_canary_binding"]["path"]),
                active_canary_binding_sha256=source["active_canary_binding"]["file_sha256"],
                checkpoint_path=Path(source["checkpoint"]["path"]),
                checkpoint_sha256=source["checkpoint"]["file_sha256"],
                terminal_path=Path(source["terminal_acceptance"]["path"]),
                terminal_sha256=source["terminal_acceptance"]["file_sha256"],
                native_reload_path=Path(source["native_reload_acceptance"]["path"]),
                native_reload_sha256=source["native_reload_acceptance"]["file_sha256"],
                validate_historical=validate_historical,
            )
            public = {key: item for key, item in bound.items() if key != "checkpoint_manifest"}
            if source != public:
                raise ValueError("Miles HF export source reference changed")
    else:
        if set(source) != {"export_acceptance", "export"}:
            raise ValueError("Miles HF prepared reload source fields changed")
        export_acceptance = source.get("export_acceptance")
        export = source.get("export")
        if (
            not isinstance(export_acceptance, dict)
            or set(export_acceptance) != _REFERENCE_FIELDS
            or _sfs(export_acceptance.get("path"), "accepted HF export")
            != export_acceptance.get("path")
            or any(
                _SHA.fullmatch(str(export_acceptance.get(key, ""))) is None
                for key in ("file_sha256", "receipt_sha256")
            )
            or not isinstance(export, dict)
            or set(export)
            != {
                "path",
                "file_sha256",
                "receipt_sha256",
                "tensor_inventory_sha256",
                "active_canary_binding",
            }
            or _sfs(export.get("path"), "HF export") != export.get("path")
            or any(
                _SHA.fullmatch(str(export.get(key, ""))) is None
                for key in ("file_sha256", "receipt_sha256", "tensor_inventory_sha256")
            )
        ):
            raise ValueError("Miles HF reload export binding changed")
        _bound_reference(export.get("active_canary_binding"), "active canary binding")
        if check_files:
            accepted, accepted_file_sha256 = _snapshot(Path(export_acceptance["path"]))
            if (
                accepted_file_sha256 != export_acceptance["file_sha256"]
                or accepted.get("sha256", "").removeprefix("sha256:")
                != export_acceptance["receipt_sha256"]
            ):
                raise ValueError("accepted HF export binding changed")
            validated = validate_export_accepted(
                accepted,
                check_files=validate_historical,
            )
            if validated["export"] != export:
                raise ValueError("accepted HF export selected different artifact bytes")
            receipt, _ = hf.inspect_export(
                Path(export["path"]),
                export["file_sha256"],
                prepared_export=export,
            )
            if (
                receipt["sha256"].removeprefix("sha256:") != export["receipt_sha256"]
                or receipt["tensor_inventory_sha256"] != export["tensor_inventory_sha256"]
            ):
                raise ValueError("Miles HF reload source export changed")


def _inspect_plan_bound_export(
    plan: dict[str, Any], path: Path, expected_sha256: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Reopen a post-Job export without materializing its full DCP values.

    The high-memory batch Job is the sole process that independently maps and
    compares every DCP value. Later lifecycle steps revalidate the exact source
    receipts and checkpoint file hashes through ``validate_plan``, then rehash
    the immutable HF artifact and its value inventory. This preserves source
    stability without moving a 27B DCP state dict into operator RAM.
    """
    from . import miles_hf_export as hf

    if plan.get("stage") != "export":
        raise ValueError("post-Job export inspection requires an export plan")
    expected_path = Path(plan["artifact_path"]) / "EXPORT.json"
    if path != expected_path:
        raise ValueError("post-Job export path differs from its exact plan")
    validate_plan(plan, check_files=True)
    artifact, artifact_file_sha256 = hf._snapshot(path)
    if artifact_file_sha256 != expected_sha256.removeprefix("sha256:"):
        raise ValueError("post-Job export file digest changed")
    prepared = {
        "path": str(path),
        "file_sha256": artifact_file_sha256,
        "receipt_sha256": str(artifact.get("sha256", "")).removeprefix("sha256:"),
        "tensor_inventory_sha256": artifact.get("tensor_inventory_sha256"),
        "active_canary_binding": plan["source"]["active_canary_binding"],
    }
    reopened, tensors = hf.inspect_export(
        path,
        artifact_file_sha256,
        prepared_export=prepared,
    )
    checkpoint_reference = plan["source"]["checkpoint"]
    checkpoint, _ = hf._snapshot_json(
        Path(checkpoint_reference["path"]), checkpoint_reference["file_sha256"]
    )
    generation = Path(checkpoint["root"]) / f"iter_{checkpoint['rollout_index']:07d}"
    equivalence = reopened["source_equivalence"]
    expected_sidecars = hf._sidecar_names(checkpoint["model"])
    source_files = {row["path"]: row for row in checkpoint["model"]["files"]}
    if (
        reopened != artifact
        or reopened.get("source") != plan["source"]
        or equivalence.get("converter_input") != str(generation)
        or equivalence.get("common_pt_sha256") != hf._hash(generation / "common.pt")
        or equivalence.get("dcp_metadata_sha256") != hf._hash(generation / ".metadata")
        or set(reopened["sidecars"]) != expected_sidecars
        or any(
            source_files[name]["sha256"].removeprefix("sha256:")
            != reopened["sidecars"][name]["sha256"]
            for name in expected_sidecars
        )
    ):
        raise ValueError("post-Job export differs from its plan-bound source evidence")
    return reopened, tensors


def _execution(stage: str) -> dict[str, Any]:
    if stage == "export":
        return {
            "cluster_target": "dev",
            "dispatch": "kubernetes_batch_v1",
            "image": miles.IMAGE,
            "priority": "c2",
            "queue": DEV_QUEUE,
            "queue_priority": "q2",
            "node_pool": DEV_GPU_NODE_POOL,
            "workers": 1,
            "gpus_per_worker": 0,
            "cpu_converter": True,
            "cuda_visible": False,
            "resources": RESOURCES[stage],
        }
    if stage == "reload":
        return {
            "cluster_target": "dev",
            "dispatch": "jobs_api_rayjob_v1",
            "image": miles.IMAGE,
            "priority": "c1",
            "workers": 1,
            "gpus_per_worker": 1,
            "cpu_converter": False,
            "cuda_visible": True,
            "resources": RESOURCES[stage],
        }
    raise ValueError("Miles HF job stage is unsupported")


def _runtime_payload(plan: dict[str, Any]) -> tuple[dict[str, Any], bytes, str]:
    files = _runtime()
    files.update({"training/__init__.py": "", "cyber_post_train/__init__.py": ""})
    files["plan.json"] = json.dumps(plan, sort_keys=True, separators=(",", ":"))
    payload = {
        "files": files,
        "module": "training.miles_hf_export_job",
        "argv": ["--plan", "plan.json", "--sha256", digest(plan)],
    }
    blob = canonical_gzip(json.dumps(payload, sort_keys=True).encode())
    return payload, blob, base64.b64encode(blob).decode()


def _runtime_transport(plan: dict[str, Any]) -> tuple[dict[str, str], str, str]:
    _payload, blob, encoded = _runtime_payload(plan)
    if len(encoded) > 512000:
        raise ValueError("Miles HF runtime bundle exceeds the reviewed transport bound")
    transport = {"CYBER_RUNTIME_BUNDLE": encoded}
    expression = "os.environ.pop('CYBER_RUNTIME_BUNDLE')"
    if len(encoded) > 120000:
        parts = [encoded[index : index + 48000] for index in range(0, len(encoded), 48000)]
        transport = {f"CYBER_RUNTIME_BUNDLE_{index}": part for index, part in enumerate(parts)}
        expression = (
            f"''.join(os.environ.pop('CYBER_RUNTIME_BUNDLE_'+str(i)) for i in range({len(parts)}))"
        )
    bootstrap = (
        "import base64,gzip,hashlib,importlib,json,os,pathlib,runpy,sys;"
        f"b=base64.b64decode({expression},validate=True);"
        f"assert hashlib.sha256(b).hexdigest()=={hashlib.sha256(blob).hexdigest()!r};"
        "v=json.loads(gzip.decompress(b));"
        "r=pathlib.Path(os.environ['RUN_DIR']);r.mkdir(mode=0o700);"
        "p=r/'.runtime';p.mkdir(mode=0o700);"
        "[((p/n).parent.mkdir(parents=True,exist_ok=True),(p/n).write_text(t)) "
        "for n,t in v['files'].items()];"
        "os.chdir(p);sys.path.insert(0,str(p));importlib.invalidate_caches();"
        "sys.argv=[v['module']]+v['argv'];"
        "runpy.run_module(v['module'],run_name='__main__',alter_sys=True)"
    )
    return transport, bootstrap, hashlib.sha256(blob).hexdigest()


def _kubernetes_job_request(plan: dict[str, Any]) -> dict[str, Any]:
    transport, bootstrap, bundle_sha256 = _runtime_transport(plan)
    ordinary = {**_COMMON_ENV, "CUDA_VISIBLE_DEVICES": "", "RUN_DIR": plan["output_root"]}
    env = [
        {"name": name, "value": value} for name, value in sorted({**ordinary, **transport}.items())
    ]
    labels = {
        "kueue.x-k8s.io/queue-name": DEV_QUEUE,
        "kueue.x-k8s.io/priority-class": "q2",
        "cyber-post-train.fleet.ai/owner": "chris",
        "cyber-post-train.fleet.ai/component": "miles-hf-export",
    }
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": plan["run_name"],
            "namespace": DEV_NAMESPACE,
            "labels": labels,
            "annotations": {
                "cyber-post-train.fleet.ai/source-plan-sha256": digest(plan),
                "cyber-post-train.fleet.ai/runtime-source-sha256": plan["runtime_sha256"],
                "cyber-post-train.fleet.ai/runtime-bundle-sha256": bundle_sha256,
            },
        },
        "spec": {
            "suspend": True,
            "backoffLimit": 0,
            "completions": 1,
            "parallelism": 1,
            "activeDeadlineSeconds": plan["deadline_seconds"],
            "ttlSecondsAfterFinished": 0,
            "template": {
                "metadata": {"labels": labels},
                "spec": {
                    "restartPolicy": "Never",
                    "automountServiceAccountToken": False,
                    "priorityClassName": "c2",
                    "nodeSelector": {"workload": DEV_GPU_NODE_POOL},
                    "tolerations": [
                        {
                            "key": "workload",
                            "operator": "Equal",
                            "value": DEV_GPU_NODE_POOL,
                            "effect": "NoSchedule",
                        }
                    ],
                    "containers": [
                        {
                            "name": "export",
                            "image": miles.IMAGE,
                            "imagePullPolicy": "IfNotPresent",
                            "command": ["python", "-c", bootstrap],
                            "env": env,
                            "resources": {
                                "requests": {
                                    "cpu": RESOURCES["export"]["cpu_request"],
                                    "memory": RESOURCES["export"]["memory_request"],
                                },
                                "limits": {
                                    "cpu": RESOURCES["export"]["cpu_limit"],
                                    "memory": RESOURCES["export"]["memory_limit"],
                                },
                            },
                            "volumeMounts": [{"name": "sfs", "mountPath": "/mnt/sfs"}],
                            "terminationMessagePolicy": "File",
                        }
                    ],
                    "volumes": [{"name": "sfs", "persistentVolumeClaim": {"claimName": SFS_CLAIM}}],
                },
            },
        },
    }


def job_request(plan: dict[str, Any]) -> dict[str, Any]:
    validate_plan(plan, check_files=False)
    if plan["stage"] == "export":
        return _kubernetes_job_request(plan)
    files = _runtime()
    files.update({"training/__init__.py": "", "cyber_post_train/__init__.py": ""})
    files["plan.json"] = json.dumps(plan, sort_keys=True, separators=(",", ":"))
    env = dict(_COMMON_ENV)
    return bundled_request(
        {
            "name": plan["run_name"],
            "title": plan["run_name"] + " exact Miles HF " + plan["stage"],
            "run_dir": plan["output_root"],
            "image": miles.IMAGE,
            "workers": 1,
            "gpus_per_worker": 1,
            "resources": plan["execution"]["resources"],
            "priority_class": plan["execution"]["priority"],
            "requeueIfPreempted": False,
            "secrets": [],
            "env": env,
        },
        files,
        "training.miles_hf_export_job",
        ["--plan", "plan.json", "--sha256", digest(plan)],
    )


def _existing_ancestor(path: Path) -> Path:
    while not path.exists() and path != path.parent:
        path = path.parent
    return path


def _storage_requirement(plan: Mapping[str, Any]) -> int:
    if plan["stage"] == "reload":
        return 8 * 1024**3
    from .miles_hf_export import _snapshot_json

    reference = plan["source"]["checkpoint"]
    checkpoint, _ = _snapshot_json(Path(reference["path"]), reference["file_sha256"])
    base = Path(checkpoint["model"]["root"])
    model_bytes = sum(
        item.get("size", (base / item["path"]).stat().st_size)
        for item in checkpoint["model"]["files"]
        if item["path"].endswith(".safetensors")
    )
    return model_bytes + 8 * 1024**3


def preflight(plan: dict[str, Any]) -> dict[str, Any]:
    import torch

    if torch.cuda.is_available():
        raise ValueError("Miles HF CPU preflight must run without a visible GPU")
    validate_plan(plan, check_files=True)
    request = job_request(plan)
    root = Path(plan["output_root"])
    if root.exists() or root.is_symlink():
        raise FileExistsError("Miles HF create-once output root already exists")
    required = _storage_requirement(plan)
    free = shutil.disk_usage(_existing_ancestor(root)).free
    if free < required:
        raise ValueError("Miles HF SFS free-space rail failed")
    resources = plan["execution"]["resources"]
    required_profile = RESOURCES[plan["stage"]]
    if any(
        quantity(resources[key]) < quantity(required_profile[key])
        for key in ("cpu_request", "cpu_limit", "memory_request", "memory_limit")
    ):
        raise ValueError("Miles HF memory rail failed")
    return {
        "schema": PREFLIGHT_SCHEMA,
        "status": "passed",
        "gpus": 0,
        "plan_sha256": digest(plan),
        "request_sha256": digest(request),
        "stage": plan["stage"],
        "source_reopened": True,
        "runtime_bundle_verified": True,
        "required_free_bytes": required,
        "observed_free_bytes": free,
        "memory_request_bytes": int(quantity(resources["memory_request"])),
        "memory_limit_bytes": int(quantity(resources["memory_limit"])),
        "submitted": False,
    }


def validate_preflight(
    plan: dict[str, Any], request: dict[str, Any], proof: dict[str, Any]
) -> None:
    resources = plan["execution"]["resources"]
    required_free_bytes = _storage_requirement(plan)
    expected_fields = {
        "schema",
        "status",
        "gpus",
        "plan_sha256",
        "request_sha256",
        "stage",
        "source_reopened",
        "runtime_bundle_verified",
        "required_free_bytes",
        "observed_free_bytes",
        "memory_request_bytes",
        "memory_limit_bytes",
        "submitted",
        "sha256",
    }
    if (
        set(proof) != expected_fields
        or request != job_request(plan)
        or any(
            proof.get(key) != value
            for key, value in {
                "schema": PREFLIGHT_SCHEMA,
                "status": "passed",
                "gpus": 0,
                "plan_sha256": digest(plan),
                "request_sha256": digest(request),
                "stage": plan["stage"],
                "source_reopened": True,
                "runtime_bundle_verified": True,
                "required_free_bytes": required_free_bytes,
                "memory_request_bytes": int(quantity(resources["memory_request"])),
                "memory_limit_bytes": int(quantity(resources["memory_limit"])),
                "submitted": False,
            }.items()
        )
    ):
        raise ValueError("Miles HF CPU preflight receipt is incomplete or mismatched")
    if proof.get("sha256") != digest(
        {key: value for key, value in proof.items() if key != "sha256"}
    ):
        raise ValueError("Miles HF CPU preflight receipt digest changed")
    if (
        type(proof.get("required_free_bytes")) is not int
        or type(proof.get("observed_free_bytes")) is not int
        or proof["observed_free_bytes"] < proof["required_free_bytes"]
    ):
        raise ValueError("Miles HF CPU preflight free-space proof changed")


def run(plan: dict[str, Any]) -> dict[str, Any]:
    from . import miles_hf_export as hf

    if os.environ.get("RUN_DIR") != plan["output_root"]:
        raise ValueError("HF runtime output binding drift")
    validate_plan(plan, check_files=True, validate_historical=False)
    root = Path(plan["output_root"])
    artifact = Path(plan["artifact_path"])
    if artifact.exists() or artifact.is_symlink() or (root / "FAILED.json").exists():
        raise FileExistsError("Miles HF job already has a terminal artifact")
    required = _storage_requirement(plan)
    if shutil.disk_usage(_existing_ancestor(root)).free < required:
        raise ValueError("Miles HF runtime SFS free-space rail failed")
    if plan["stage"] == "export":
        if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"", "-1"}:
            raise ValueError("CPU exporter received a visible GPU")
        source = plan["source"]
        return hf.export(
            active_canary_binding_path=Path(source["active_canary_binding"]["path"]),
            active_canary_binding_sha256=source["active_canary_binding"]["file_sha256"],
            checkpoint_path=Path(source["checkpoint"]["path"]),
            checkpoint_sha256=source["checkpoint"]["file_sha256"],
            terminal_path=Path(source["terminal_acceptance"]["path"]),
            terminal_sha256=source["terminal_acceptance"]["file_sha256"],
            native_reload_path=Path(source["native_reload_acceptance"]["path"]),
            native_reload_sha256=source["native_reload_acceptance"]["file_sha256"],
            output=artifact,
            prepared_source=source,
        )
    export = plan["source"]["export"]
    return hf.gpu_reload(
        Path(export["path"]),
        export["file_sha256"],
        artifact,
        run_name=plan["run_name"],
        prepared_export=export,
    )


def _transport(request: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    if request.get("kind") == "Job":
        containers = (((request.get("spec") or {}).get("template") or {}).get("spec") or {}).get(
            "containers"
        )
        if not isinstance(containers, list) or len(containers) != 1:
            raise ValueError("HF Kubernetes request container is absent")
        entries = containers[0].get("env")
        if not isinstance(entries, list) or any(
            not isinstance(entry, Mapping)
            or set(entry) != {"name", "value"}
            or not isinstance(entry["name"], str)
            or not isinstance(entry["value"], str)
            for entry in entries
        ):
            raise ValueError("HF Kubernetes request environment is malformed")
        env: Mapping[str, Any] = {entry["name"]: entry["value"] for entry in entries}
        if len(env) != len(entries):
            raise ValueError("HF Kubernetes request environment has duplicate names")
    else:
        candidate = request.get("env")
        if not isinstance(candidate, Mapping):
            raise ValueError("HF request environment is absent")
        env = candidate
    direct = env.get("CYBER_RUNTIME_BUNDLE")
    chunks = sorted(
        (
            (int(str(key).removeprefix("CYBER_RUNTIME_BUNDLE_")), value)
            for key, value in env.items()
            if re.fullmatch(r"CYBER_RUNTIME_BUNDLE_\d+", str(key))
        ),
        key=lambda row: row[0],
    )
    if direct is not None:
        if chunks:
            raise ValueError("HF runtime transport is ambiguous")
        encoded = direct
    else:
        if not chunks or [row[0] for row in chunks] != list(range(len(chunks))):
            raise ValueError("HF runtime transport is absent or non-contiguous")
        encoded = "".join(str(row[1]) for row in chunks)
    try:
        blob = base64.b64decode(encoded, validate=True)
        payload = json.loads(gzip.decompress(blob))
    except (binascii.Error, OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("HF runtime transport is invalid") from error
    if not isinstance(payload, dict):
        raise ValueError("HF runtime transport payload is not an object")
    return payload, hashlib.sha256(blob).hexdigest()


def _request_projection(
    plan: dict[str, Any], request: dict[str, Any], *, source_commit: str, repo_root: Path
) -> tuple[dict[str, Any], str]:
    expected = job_request(plan)
    if request != expected or re.fullmatch(r"[a-f0-9]{40}", source_commit) is None:
        raise ValueError("HF request or source commit differs from the prepared job")
    payload, bundle_sha256 = _transport(request)
    if plan["stage"] == "export":
        container = request["spec"]["template"]["spec"]["containers"][0]
        command = container.get("command")
    else:
        command = shlex.split(request["command"])
    if (
        set(payload) != {"files", "module", "argv"}
        or payload["module"] != "training.miles_hf_export_job"
        or payload["argv"] != ["--plan", "plan.json", "--sha256", digest(plan)]
        or payload["files"].get("plan.json")
        != json.dumps(plan, sort_keys=True, separators=(",", ":"))
        or not isinstance(command, list)
        or len(command) != 3
        or command[:2] != ["python", "-c"]
        or bundle_sha256 not in command[2]
    ):
        raise ValueError("HF runtime bundle is not exactly plan-bound")
    sources = {name: payload["files"].get(name) for name in RUNTIME_FILES}
    if (
        any(not isinstance(value, str) for value in sources.values())
        or digest(sources) != plan["runtime_sha256"]
    ):
        raise ValueError("HF runtime source inventory changed")
    for name, text in sources.items():
        observed = subprocess.run(
            ["git", "show", f"{source_commit}:{name}"],
            cwd=repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if observed.returncode or observed.stdout != text.encode():
            raise ValueError("HF runtime bundle differs from its source commit")
    if plan["stage"] == "export":
        request_env = {
            entry["name"]: entry["value"]
            for entry in request["spec"]["template"]["spec"]["containers"][0]["env"]
        }
    else:
        request_env = request["env"]
    ordinary = {
        key: value
        for key, value in request_env.items()
        if key != "CYBER_RUNTIME_BUNDLE" and not re.fullmatch(r"CYBER_RUNTIME_BUNDLE_\d+", key)
    }
    expected_env = dict(_COMMON_ENV)
    if plan["stage"] == "export":
        expected_env["CUDA_VISIBLE_DEVICES"] = ""
        expected_env["RUN_DIR"] = plan["output_root"]
    if ordinary != expected_env:
        raise ValueError("HF runtime environment changed")
    return _request_summary(request, ordinary, payload), bundle_sha256


def _request_summary(
    request: Mapping[str, Any], ordinary: Mapping[str, Any], payload: Mapping[str, Any]
) -> dict[str, Any]:
    if request.get("kind") == "Job":
        metadata = request["metadata"]
        spec = request["spec"]
        pod = spec["template"]["spec"]
        container = pod["containers"][0]
        return {
            "api_version": request["apiVersion"],
            "kind": "Job",
            "name": metadata["name"],
            "namespace": metadata["namespace"],
            "image": container["image"],
            "resources": container["resources"],
            "priority_class": pod["priorityClassName"],
            "queue": metadata["labels"]["kueue.x-k8s.io/queue-name"],
            "queue_priority": metadata["labels"]["kueue.x-k8s.io/priority-class"],
            "node_selector": pod["nodeSelector"],
            "tolerations": pod["tolerations"],
            "sfs_claim": pod["volumes"][0]["persistentVolumeClaim"]["claimName"],
            "suspend": spec["suspend"],
            "backoff_limit": spec["backoffLimit"],
            "active_deadline_seconds": spec["activeDeadlineSeconds"],
            "ttl_seconds_after_finished": spec["ttlSecondsAfterFinished"],
            "restart_policy": pod["restartPolicy"],
            "gpus": 0,
            "environment_names": sorted(ordinary),
            "runtime_module": payload["module"],
            "runtime_argv": payload["argv"],
        }
    return {
        "name": request["name"],
        "run_dir": request["run_dir"],
        "image": request["image"],
        "workers": request["workers"],
        "gpus_per_worker": request["gpus_per_worker"],
        "resources": request["resources"],
        "priority_class": request["priority_class"],
        "automatic_requeue": request["requeueIfPreempted"],
        "secret_names": request["secrets"],
        "environment_names": sorted(ordinary),
        "runtime_module": payload["module"],
        "runtime_argv": payload["argv"],
    }


def _snapshot(path: Path) -> tuple[dict[str, Any], str]:
    from .miles_acceptance import _json_snapshot

    value, file_sha256 = _json_snapshot(path)
    if not isinstance(value, dict):
        raise ValueError("HF job evidence must contain one JSON object")
    return value, file_sha256


def _jobs_submission_journal(
    path: Path, plan: dict[str, Any], request: dict[str, Any]
) -> tuple[dict[str, Any], str]:
    """Reopen the one-intent/one-response Jobs API journal as API authority."""
    if path.is_symlink() or not path.is_file():
        raise ValueError("HF submission journal is missing or indirect")
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    if any(
        getattr(before, key) != getattr(after, key)
        for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    ):
        raise ValueError("HF submission journal changed while reading")
    try:
        rows = [json.loads(line) for line in raw.splitlines()]
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("HF submission journal is not valid JSONL") from error
    if len(rows) != 2 or any(not isinstance(row, dict) for row in rows):
        raise ValueError("HF submission journal must contain one intent and one response")
    intent, response = rows
    expected_intent_fields = {
        "state",
        "api_base_url",
        "request_sha256",
        "manifest_sha256",
        "nodes",
        "gpus",
        "image",
    }
    expected_response_fields = {
        "state",
        "name",
        "job_id",
        "run_dir",
        "status",
        "created_at",
        "finished_at",
    }
    run_id = str(response.get("job_id", ""))
    run_name = plan["run_name"] + "-" + run_id[:8]
    from .miles_acceptance import _time

    if (
        set(intent) != expected_intent_fields
        or intent.get("state") != "POST_INTENT_DO_NOT_RETRY"
        or intent.get("api_base_url") != API_URLS["dev"]
        or intent.get("request_sha256") != digest(request)
        or _SHA.fullmatch(str(intent.get("manifest_sha256", ""))) is None
        or intent.get("nodes") != 1
        or intent.get("gpus") != 1
        or intent.get("image") != miles.IMAGE
        or set(response) != expected_response_fields
        or response.get("state") != "POST_RESPONSE"
        or _UUID.fullmatch(run_id) is None
        or response.get("name") != run_name
        or response.get("run_dir") != plan["output_root"]
        or response.get("status")
        not in {"queued", "PENDING", "QUEUED", "RUNNING", "SUCCEEDED", "FAILED", "STOPPED"}
        or (
            response.get("status") in {"queued", "PENDING", "QUEUED", "RUNNING"}
            and response.get("finished_at") is not None
        )
        or (
            response.get("status") in {"SUCCEEDED", "FAILED", "STOPPED"}
            and response.get("finished_at") is None
        )
    ):
        raise ValueError("HF submission journal differs from the exact dev API POST")
    created_at = _time(response.get("created_at"), "HF Jobs API creation time")
    if (
        response.get("finished_at") is not None
        and _time(response["finished_at"], "HF recovered terminal time") < created_at
    ):
        raise ValueError("HF recovered terminal time predates submission")
    return {"intent": intent, "response": response}, hashlib.sha256(raw).hexdigest()


def _kubernetes_submission_journal(
    path: Path, plan: dict[str, Any], request: dict[str, Any]
) -> tuple[dict[str, Any], str]:
    """Reopen one zero-GPU Kubernetes Job create intent/response."""

    if path.is_symlink() or not path.is_file():
        raise ValueError("HF Kubernetes submission journal is missing or indirect")
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    if any(
        getattr(before, key) != getattr(after, key)
        for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    ):
        raise ValueError("HF Kubernetes submission journal changed while reading")
    try:
        rows = [json.loads(line) for line in raw.splitlines()]
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("HF Kubernetes submission journal is not valid JSONL") from error
    if len(rows) != 2 or any(not isinstance(row, dict) for row in rows):
        raise ValueError("HF Kubernetes submission journal must contain one intent and response")
    intent, response = rows
    expected_intent = {
        "state",
        "api_base_url",
        "kube_context",
        "namespace",
        "namespace_uid",
        "request_sha256",
        "server_dry_run_sha256",
        "name",
        "image",
        "gpus",
    }
    expected_response = {
        "state",
        "api_version",
        "kind",
        "name",
        "namespace",
        "uid",
        "resource_version",
        "creation_timestamp",
    }
    from .miles_acceptance import _time

    if (
        set(intent) != expected_intent
        or intent.get("state") != "KUBERNETES_POST_INTENT_DO_NOT_RETRY"
        or intent.get("api_base_url") != DEV_KUBERNETES_API
        or intent.get("kube_context") != DEV_KUBE_CONTEXT
        or intent.get("namespace") != DEV_NAMESPACE
        or intent.get("namespace_uid") != DEV_NAMESPACE_UID
        or intent.get("request_sha256") != digest(request)
        or _SHA.fullmatch(str(intent.get("server_dry_run_sha256", ""))) is None
        or intent.get("name") != plan["run_name"]
        or intent.get("image") != miles.IMAGE
        or intent.get("gpus") != 0
        or set(response) != expected_response
        or response.get("state") != "KUBERNETES_POST_RESPONSE"
        or response.get("api_version") != "batch/v1"
        or response.get("kind") != "Job"
        or response.get("name") != plan["run_name"]
        or response.get("namespace") != DEV_NAMESPACE
        or _UUID.fullmatch(str(response.get("uid", ""))) is None
        or not isinstance(response.get("resource_version"), str)
        or not response["resource_version"]
    ):
        raise ValueError("HF Kubernetes submission journal differs from the exact dev API POST")
    _time(response.get("creation_timestamp"), "HF Kubernetes Job creation time")
    return {"intent": intent, "response": response}, hashlib.sha256(raw).hexdigest()


def _submission_journal(
    path: Path, plan: dict[str, Any], request: dict[str, Any]
) -> tuple[dict[str, Any], str]:
    if plan.get("stage") == "export":
        return _kubernetes_submission_journal(path, plan, request)
    return _jobs_submission_journal(path, plan, request)


def compile_submission_binding(
    *,
    plan_path: Path,
    request_path: Path,
    submission_journal_path: Path,
    source_commit: str,
    output: Path,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    plan_path = plan_path.resolve()
    request_path = request_path.resolve()
    submission_journal_path = submission_journal_path.resolve()
    output = output.resolve()
    prepared_root = plan_path.parent
    if (
        _sfs(str(prepared_root), "HF prepared evidence root") != str(prepared_root)
        or request_path != plan_path.parent / "request.json"
        or submission_journal_path != plan_path.parent / "SUBMISSION.jsonl"
        or output != plan_path.parent / "SUBMITTED.json"
    ):
        raise ValueError("HF submission inputs are outside the prepared directory")
    plan, plan_file_sha256 = _snapshot(plan_path)
    request, request_file_sha256 = _snapshot(request_path)
    validate_plan(plan, check_files=True)
    if prepared_root == Path(plan["output_root"]):
        raise ValueError("HF prepared evidence must be disjoint from the runtime output root")
    projection, bundle_sha256 = _request_projection(
        plan,
        request,
        source_commit=source_commit,
        repo_root=repo_root or Path(__file__).resolve().parents[1],
    )
    journal, submission_journal_file_sha256 = _submission_journal(
        submission_journal_path, plan, request
    )
    response = journal["response"]
    if plan["stage"] == "export":
        api = {
            "base_url": DEV_KUBERNETES_API,
            "run_id": response["uid"],
            "run_name": response["name"],
        }
        submitted_at = response["creation_timestamp"]
    else:
        api = {
            "base_url": API_URLS["dev"],
            "run_id": response["job_id"],
            "run_name": response["name"],
        }
        submitted_at = response["created_at"]
    return _write(
        output,
        {
            "schema": SUBMISSION_SCHEMA,
            "source_commit": source_commit,
            "source_plan_path": str(plan_path),
            "source_plan_sha256": "sha256:" + digest(plan),
            "source_plan_file_sha256": "sha256:" + plan_file_sha256,
            "source_request_path": str(request_path),
            "source_request_sha256": "sha256:" + digest(request),
            "source_request_file_sha256": "sha256:" + request_file_sha256,
            "submission_journal_path": str(submission_journal_path),
            "submission_journal_file_sha256": "sha256:" + submission_journal_file_sha256,
            "runtime_bundle_sha256": "sha256:" + bundle_sha256,
            "api": api,
            "request": projection,
            "dispatch": plan["execution"]["dispatch"],
            "create_call_count": 1,
            "submitted_at": submitted_at,
            "secret_values_included": False,
            "reward_values_included": False,
            "task_content_included": False,
            "benchmark_feedback_included": False,
        },
    )


def validate_submission_binding(
    value: dict[str, Any], plan: dict[str, Any], *, check_files: bool
) -> dict[str, Any]:
    from .miles_acceptance import _time
    from .rl_runtime import sealed

    sealed(value, SUBMISSION_SCHEMA)
    api = value.get("api")
    request = value.get("request")
    expected_request = job_request(plan)
    expected_payload, expected_bundle_sha256 = _transport(expected_request)
    if plan["stage"] == "export":
        expected_env = {
            entry["name"]: entry["value"]
            for entry in expected_request["spec"]["template"]["spec"]["containers"][0]["env"]
        }
    else:
        expected_env = expected_request["env"]
    expected_ordinary = {
        key: item
        for key, item in expected_env.items()
        if key != "CYBER_RUNTIME_BUNDLE" and not re.fullmatch(r"CYBER_RUNTIME_BUNDLE_\d+", key)
    }
    expected_projection = _request_summary(expected_request, expected_ordinary, expected_payload)
    expected_api_base = DEV_KUBERNETES_API if plan["stage"] == "export" else API_URLS["dev"]
    expected_fields = {
        "schema",
        "source_commit",
        "source_plan_path",
        "source_plan_sha256",
        "source_plan_file_sha256",
        "source_request_path",
        "source_request_sha256",
        "source_request_file_sha256",
        "submission_journal_path",
        "submission_journal_file_sha256",
        "runtime_bundle_sha256",
        "api",
        "request",
        "dispatch",
        "create_call_count",
        "submitted_at",
        "secret_values_included",
        "reward_values_included",
        "task_content_included",
        "benchmark_feedback_included",
        "sha256",
    }
    if (
        set(value) != expected_fields
        or re.fullmatch(r"[a-f0-9]{40}", str(value.get("source_commit", ""))) is None
        or not Path(str(value.get("source_plan_path", ""))).is_absolute()
        or not Path(str(value.get("source_request_path", ""))).is_absolute()
        or not Path(str(value.get("submission_journal_path", ""))).is_absolute()
        or _sfs(
            str(Path(str(value.get("source_plan_path", ""))).parent),
            "HF prepared evidence root",
        )
        != str(Path(str(value.get("source_plan_path", ""))).parent)
        or Path(str(value.get("source_request_path")))
        != Path(str(value.get("source_plan_path"))).parent / "request.json"
        or Path(str(value.get("submission_journal_path")))
        != Path(str(value.get("source_plan_path"))).parent / "SUBMISSION.jsonl"
        or any(
            not isinstance(value.get(key), str)
            or not value[key].startswith("sha256:")
            or _SHA.fullmatch(value[key]) is None
            for key in (
                "source_plan_sha256",
                "source_plan_file_sha256",
                "source_request_sha256",
                "source_request_file_sha256",
                "submission_journal_file_sha256",
                "runtime_bundle_sha256",
            )
        )
        or value.get("source_plan_sha256", "").removeprefix("sha256:") != digest(plan)
        or value.get("source_request_sha256", "").removeprefix("sha256:")
        != digest(expected_request)
        or value.get("runtime_bundle_sha256", "").removeprefix("sha256:") != expected_bundle_sha256
        or not isinstance(api, dict)
        or api.get("base_url") != expected_api_base
        or _UUID.fullmatch(str(api.get("run_id"))) is None
        or api.get("run_name")
        != (
            plan["run_name"]
            if plan["stage"] == "export"
            else plan["run_name"] + "-" + str(api.get("run_id"))[:8]
        )
        or request != expected_projection
        or value.get("dispatch") != plan["execution"]["dispatch"]
        or value.get("create_call_count") != 1
        or any(
            value.get(key) is not False
            for key in (
                "secret_values_included",
                "reward_values_included",
                "task_content_included",
                "benchmark_feedback_included",
            )
        )
    ):
        raise ValueError("HF submission binding is incomplete or mismatched")
    _time(value.get("submitted_at"), "HF submission time")
    if check_files:
        observed_plan, plan_file_sha256 = _snapshot(Path(value["source_plan_path"]))
        observed_request, request_file_sha256 = _snapshot(Path(value["source_request_path"]))
        journal, journal_file_sha256 = _submission_journal(
            Path(value["submission_journal_path"]), plan, observed_request
        )
        if (
            observed_plan != plan
            or plan_file_sha256 != value["source_plan_file_sha256"].removeprefix("sha256:")
            or digest(observed_request) != value["source_request_sha256"].removeprefix("sha256:")
            or request_file_sha256 != value["source_request_file_sha256"].removeprefix("sha256:")
            or journal_file_sha256
            != value["submission_journal_file_sha256"].removeprefix("sha256:")
            or (
                plan["stage"] == "export"
                and (
                    journal["response"]["uid"] != api["run_id"]
                    or journal["response"]["name"] != api["run_name"]
                    or journal["response"]["creation_timestamp"] != value["submitted_at"]
                )
            )
            or (
                plan["stage"] == "reload"
                and (
                    journal["response"]["job_id"] != api["run_id"]
                    or journal["response"]["name"] != api["run_name"]
                    or journal["response"]["created_at"] != value["submitted_at"]
                )
            )
        ):
            raise ValueError("HF prepared plan or request file changed")
        projection, bundle_sha256 = _request_projection(
            plan,
            observed_request,
            source_commit=value["source_commit"],
            repo_root=Path(__file__).resolve().parents[1],
        )
        if projection != request or bundle_sha256 != value["runtime_bundle_sha256"].removeprefix(
            "sha256:"
        ):
            raise ValueError("HF runtime bundle or request projection changed")
    return {"request_sha256": value["source_request_sha256"], "api": api, "request": request}


def controller_from_event_journal(
    *,
    plan: dict[str, Any],
    submission: dict[str, Any],
    start: dict[str, Any],
    identities: dict[str, tuple[str, str]],
    pod: dict[str, Any],
    event_journal: dict[str, Any],
    observed_at: float,
) -> dict[str, Any]:
    """Project trusted watch facts into the HF reload controller schema."""
    from . import miles_hf_export as hf

    submitted = validate_submission_binding(submission, plan, check_files=False)
    if (
        plan.get("stage") not in STAGES
        or start.get("kube_context") != DEV_KUBE_CONTEXT
        or start.get("namespace_uid") != DEV_NAMESPACE_UID
    ):
        raise ValueError("HF controller evidence is not for the exact job/watch context")
    if plan["stage"] == "export":
        job_name, job_uid = identities["Job"]
        workload_name, workload_uid = identities["Workload"]
        pod_name, pod_uid = identities["Pod"]
        artifact_path = Path(plan["artifact_path"]) / "EXPORT.json"
        artifact, artifact_file_sha256 = hf._snapshot(artifact_path)
        _inspect_plan_bound_export(plan, artifact_path, artifact_file_sha256)
        if observed_at < float(artifact["completed_at"]):
            raise ValueError("HF export artifact postdates its controller observation")
        return {
            "schema": EXPORT_CONTROLLER_SCHEMA,
            "status": "succeeded",
            "stage": "export",
            "cluster": "dev",
            "api_base_url": DEV_KUBERNETES_API,
            "kube_context": DEV_KUBE_CONTEXT,
            "namespace": DEV_NAMESPACE,
            "namespace_uid": DEV_NAMESPACE_UID,
            "run_name": plan["run_name"],
            "source_plan_sha256": "sha256:" + digest(plan),
            "source_request_sha256": submitted["request_sha256"],
            "submission_binding_sha256": submission["sha256"],
            "runtime_bundle_sha256": submission["runtime_bundle_sha256"],
            "event_journal": event_journal,
            "artifact_path": str(artifact_path),
            "artifact_file_sha256": artifact_file_sha256,
            "artifact_receipt_sha256": artifact["sha256"].removeprefix("sha256:"),
            "api_run_id": submitted["api"]["run_id"],
            "api_run_name": submitted["api"]["run_name"],
            "job_name": job_name,
            "job_uid": job_uid,
            "workload_name": workload_name,
            "workload_uid": workload_uid,
            "workload_owner_job_uid": job_uid,
            "pods": [
                {
                    "name": pod_name,
                    "uid": pod_uid,
                    "owner_job_uid": job_uid,
                    "phase": pod["phase"],
                    "exit_code": pod["exit_code"],
                    "termination_reason": pod["termination_reason"],
                    "terminated_at": pod["terminated_at"],
                    "runtime_image_id": pod["runtime_image_id"],
                    "container_restarts": pod["container_restarts"],
                    "gpus": pod["gpus"],
                }
            ],
            "api_status": "SUCCEEDED",
            "controller_status": "SUCCEEDED",
            "effective_priority": 5000,
            "automatic_requeue": False,
            "workers": 1,
            "gpus_per_worker": 0,
            "total_gpus": 0,
            "observed_at": observed_at,
        }
    if plan["stage"] != "reload":
        raise ValueError("HF controller stage is unsupported")
    rayjob_name, rayjob_uid = identities["RayJob"]
    workload_name, workload_uid = identities["Workload"]
    raycluster_name, raycluster_uid = identities["RayCluster"]
    pod_name, pod_uid = identities["Pod"]
    result_path = Path(plan["artifact_path"])
    result, result_file_sha256 = hf._snapshot(result_path)
    completed_at = hf._validate_reload_result(result)
    if result.get("run_name") != plan["run_name"] or observed_at < completed_at:
        raise ValueError("HF reload result differs from the watched plan")
    return {
        "schema": hf.RELOAD_CONTROLLER_SCHEMA,
        "status": "succeeded",
        "cluster": "dev",
        "api_base_url": API_URLS["dev"],
        "kube_context": DEV_KUBE_CONTEXT,
        "namespace": DEV_NAMESPACE,
        "namespace_uid": DEV_NAMESPACE_UID,
        "run_name": plan["run_name"],
        "source_plan_sha256": "sha256:" + digest(plan),
        "source_request_sha256": submitted["request_sha256"],
        "submission_binding_sha256": submission["sha256"],
        "runtime_bundle_sha256": submission["runtime_bundle_sha256"],
        "event_journal": event_journal,
        "reload_result_path": str(result_path),
        "reload_result_file_sha256": result_file_sha256,
        "reload_result_sha256": result["sha256"].removeprefix("sha256:"),
        "api_run_id": submitted["api"]["run_id"],
        "api_run_name": submitted["api"]["run_name"],
        "rayjob_name": rayjob_name,
        "rayjob_uid": rayjob_uid,
        "workload_name": workload_name,
        "workload_uid": workload_uid,
        "workload_owner_rayjob_uid": rayjob_uid,
        "raycluster_name": raycluster_name,
        "raycluster_uid": raycluster_uid,
        "raycluster_owner_rayjob_uid": rayjob_uid,
        "pods": [
            {
                "name": pod_name,
                "uid": pod_uid,
                "owner_raycluster_uid": raycluster_uid,
                "phase": pod["phase"],
                "exit_code": pod["exit_code"],
                "termination_reason": pod["termination_reason"],
                "terminated_at": pod["terminated_at"],
                "runtime_image_id": pod["runtime_image_id"],
                "container_restarts": pod["container_restarts"],
                "gpus": pod["gpus"],
            }
        ],
        "api_status": "SUCCEEDED",
        "controller_status": "SUCCEEDED",
        "effective_priority": 10000,
        "automatic_requeue": False,
        "workers": 1,
        "gpus_per_worker": 1,
        "total_gpus": 1,
        "observed_at": observed_at,
    }


def validate_event_controller(
    value: dict[str, Any], plan: dict[str, Any], submission: dict[str, Any]
) -> dict[str, Any]:
    from . import miles_hf_export as hf

    if plan.get("stage") == "export":
        return _validate_export_controller(value, plan, submission)
    result_path = Path(plan["artifact_path"])
    result, result_file_sha256 = hf._snapshot(result_path)
    hf._validate_controller(
        value,
        result,
        result_path,
        result_file_sha256,
        hf._validate_reload_result(result),
        plan,
        submission,
    )
    return result


def _validate_export_controller(
    value: dict[str, Any], plan: dict[str, Any], submission: dict[str, Any]
) -> dict[str, Any]:
    from . import miles_hf_export as hf
    from .miles_event_evidence import validate_hf_event_journal
    from .rl_runtime import sealed

    if plan.get("stage") != "export":
        raise ValueError("HF export controller requires an export plan")
    submitted = validate_submission_binding(submission, plan, check_files=False)
    sealed(value, EXPORT_CONTROLLER_SCHEMA)
    validate_hf_event_journal(plan, submission, value.get("event_journal"), value)
    artifact_path = Path(plan["artifact_path"]) / "EXPORT.json"
    artifact, artifact_file_sha256 = hf._snapshot(artifact_path)
    _inspect_plan_bound_export(plan, artifact_path, artifact_file_sha256)
    api = value.get("api_run_id")
    pod_rows = value.get("pods")
    pod = pod_rows[0] if isinstance(pod_rows, list) and len(pod_rows) == 1 else None
    expected_image = miles.IMAGE.rsplit("@sha256:", 1)[-1]
    if (
        set(value) != _EXPORT_CONTROLLER_FIELDS
        or value.get("status") != "succeeded"
        or value.get("stage") != "export"
        or value.get("cluster") != "dev"
        or value.get("api_base_url") != DEV_KUBERNETES_API
        or value.get("kube_context") != DEV_KUBE_CONTEXT
        or value.get("namespace") != DEV_NAMESPACE
        or value.get("namespace_uid") != DEV_NAMESPACE_UID
        or value.get("run_name") != plan["run_name"]
        or value.get("source_plan_sha256") != "sha256:" + digest(plan)
        or value.get("source_request_sha256") != submitted["request_sha256"]
        or value.get("submission_binding_sha256") != submission["sha256"]
        or value.get("runtime_bundle_sha256") != submission["runtime_bundle_sha256"]
        or value.get("artifact_path") != str(artifact_path)
        or value.get("artifact_file_sha256") != artifact_file_sha256
        or value.get("artifact_receipt_sha256") != artifact["sha256"].removeprefix("sha256:")
        or _UUID.fullmatch(str(api)) is None
        or api != submitted["api"]["run_id"]
        or value.get("api_run_name") != submitted["api"]["run_name"]
        or value.get("job_name") != submitted["api"]["run_name"]
        or value.get("job_uid") != submitted["api"]["run_id"]
        or any(_UUID.fullmatch(str(value.get(key))) is None for key in ("job_uid", "workload_uid"))
        or value.get("workload_owner_job_uid") != value.get("job_uid")
        or value.get("api_status") != "SUCCEEDED"
        or value.get("controller_status") != "SUCCEEDED"
        or value.get("effective_priority") != 5000
        or value.get("automatic_requeue") is not False
        or value.get("workers") != 1
        or value.get("gpus_per_worker") != 0
        or value.get("total_gpus") != 0
        or not isinstance(pod, dict)
        or set(pod)
        != {
            "name",
            "uid",
            "owner_job_uid",
            "phase",
            "exit_code",
            "termination_reason",
            "terminated_at",
            "runtime_image_id",
            "container_restarts",
            "gpus",
        }
        or _UUID.fullmatch(str(pod.get("uid"))) is None
        or pod.get("owner_job_uid") != value.get("job_uid")
        or pod.get("phase") != "Succeeded"
        or pod.get("exit_code") != 0
        or pod.get("termination_reason") != "Completed"
        or hf._runtime_image_digest(pod.get("runtime_image_id")) != expected_image
        or pod.get("container_restarts") != 0
        or pod.get("gpus") != 0
        or hf._evidence_time(pod.get("terminated_at")) < float(artifact["completed_at"])
        or hf._evidence_time(value.get("observed_at")) < hf._evidence_time(pod.get("terminated_at"))
    ):
        raise ValueError("HF export controller evidence is incomplete or mismatched")
    return artifact


def release_from_event_journal(
    *,
    plan: dict[str, Any],
    submission: dict[str, Any],
    controller: dict[str, Any],
    controller_path: Path,
    controller_file_sha256: str,
    observed_at: float,
) -> dict[str, Any]:
    """Project the post-terminal absence observation without inventing UIDs."""
    from . import miles_hf_export as hf

    result = validate_event_controller(controller, plan, submission)
    pod_uid = controller["pods"][0]["uid"]
    if plan.get("stage") == "export":
        return {
            "schema": EXPORT_RELEASE_SCHEMA,
            "status": "released",
            "stage": "export",
            "cluster": "dev",
            "api_base_url": DEV_KUBERNETES_API,
            "kube_context": DEV_KUBE_CONTEXT,
            "namespace": DEV_NAMESPACE,
            "namespace_uid": DEV_NAMESPACE_UID,
            "run_name": plan["run_name"],
            "source_plan_sha256": controller["source_plan_sha256"],
            "source_request_sha256": controller["source_request_sha256"],
            "submission_binding_sha256": controller["submission_binding_sha256"],
            "runtime_bundle_sha256": controller["runtime_bundle_sha256"],
            "artifact_path": controller["artifact_path"],
            "artifact_file_sha256": controller["artifact_file_sha256"],
            "artifact_receipt_sha256": result["sha256"].removeprefix("sha256:"),
            "controller_terminal_path": str(controller_path),
            "controller_terminal_file_sha256": controller_file_sha256,
            "controller_terminal_sha256": controller["sha256"].removeprefix("sha256:"),
            "api_run_id": controller["api_run_id"],
            "api_run_name": controller["api_run_name"],
            "job_name": controller["job_name"],
            "job_uid": controller["job_uid"],
            "workload_name": controller["workload_name"],
            "workload_uid": controller["workload_uid"],
            "pod_uids": [pod_uid],
            "api_status": "SUCCEEDED",
            "controller_status": "SUCCEEDED",
            "job_present": False,
            "workload_present": False,
            "quota_reservation_present": False,
            "gpu_pods_present": False,
            "active_gpu_pod_uids": [],
            "active_gpus": 0,
            "observed_at": observed_at,
        }
    return {
        "schema": hf.RELOAD_RELEASE_SCHEMA,
        "status": "released",
        "cluster": "dev",
        "api_base_url": API_URLS["dev"],
        "kube_context": DEV_KUBE_CONTEXT,
        "namespace": DEV_NAMESPACE,
        "namespace_uid": DEV_NAMESPACE_UID,
        "run_name": plan["run_name"],
        "source_plan_sha256": controller["source_plan_sha256"],
        "source_request_sha256": controller["source_request_sha256"],
        "submission_binding_sha256": controller["submission_binding_sha256"],
        "runtime_bundle_sha256": controller["runtime_bundle_sha256"],
        "reload_result_path": controller["reload_result_path"],
        "reload_result_file_sha256": controller["reload_result_file_sha256"],
        "reload_result_sha256": result["sha256"].removeprefix("sha256:"),
        "controller_terminal_path": str(controller_path),
        "controller_terminal_file_sha256": controller_file_sha256,
        "controller_terminal_sha256": controller["sha256"].removeprefix("sha256:"),
        "api_run_id": controller["api_run_id"],
        "api_run_name": controller["api_run_name"],
        "rayjob_name": controller["rayjob_name"],
        "rayjob_uid": controller["rayjob_uid"],
        "workload_name": controller["workload_name"],
        "workload_uid": controller["workload_uid"],
        "raycluster_name": controller["raycluster_name"],
        "raycluster_uid": controller["raycluster_uid"],
        "pod_uids": [pod_uid],
        "api_status": "SUCCEEDED",
        "controller_status": "SUCCEEDED",
        "rayjob_present": False,
        "workload_present": False,
        "quota_reservation_present": False,
        "raycluster_present": False,
        "gpu_pods_present": False,
        "active_gpu_pod_uids": [],
        "active_gpus": 0,
        "observed_at": observed_at,
    }


def _validate_export_release(
    value: dict[str, Any],
    plan: dict[str, Any],
    submission: dict[str, Any],
    controller: dict[str, Any],
    controller_path: Path,
    controller_file_sha256: str,
) -> None:
    from . import miles_hf_export as hf
    from .miles_event_evidence import validate_hf_release_journal
    from .rl_runtime import sealed

    artifact = _validate_export_controller(controller, plan, submission)
    sealed(value, EXPORT_RELEASE_SCHEMA)
    validate_hf_release_journal(plan, submission, controller, value.get("observed_at"))
    identities = (
        "api_run_id",
        "api_run_name",
        "job_name",
        "job_uid",
        "workload_name",
        "workload_uid",
    )
    if (
        set(value) != _EXPORT_RELEASE_FIELDS
        or value.get("status") != "released"
        or value.get("stage") != "export"
        or value.get("cluster") != "dev"
        or value.get("api_base_url") != DEV_KUBERNETES_API
        or value.get("kube_context") != DEV_KUBE_CONTEXT
        or value.get("namespace") != DEV_NAMESPACE
        or value.get("namespace_uid") != DEV_NAMESPACE_UID
        or value.get("run_name") != plan["run_name"]
        or value.get("source_plan_sha256") != controller["source_plan_sha256"]
        or value.get("source_request_sha256") != controller["source_request_sha256"]
        or value.get("submission_binding_sha256") != controller["submission_binding_sha256"]
        or value.get("runtime_bundle_sha256") != controller["runtime_bundle_sha256"]
        or value.get("artifact_path") != controller["artifact_path"]
        or value.get("artifact_file_sha256") != controller["artifact_file_sha256"]
        or value.get("artifact_receipt_sha256") != artifact["sha256"].removeprefix("sha256:")
        or value.get("controller_terminal_path") != str(controller_path)
        or value.get("controller_terminal_file_sha256") != controller_file_sha256
        or value.get("controller_terminal_sha256") != controller["sha256"].removeprefix("sha256:")
        or any(value.get(key) != controller[key] for key in identities)
        or value.get("pod_uids") != [controller["pods"][0]["uid"]]
        or value.get("api_status") != "SUCCEEDED"
        or value.get("controller_status") != "SUCCEEDED"
        or any(
            value.get(key) is not False
            for key in (
                "job_present",
                "workload_present",
                "quota_reservation_present",
                "gpu_pods_present",
            )
        )
        or value.get("active_gpu_pod_uids") != []
        or value.get("active_gpus") != 0
        or hf._evidence_time(value.get("observed_at"))
        < hf._evidence_time(controller.get("observed_at"))
    ):
        raise ValueError("HF export external release is incomplete or mismatched")


def accept_export_job(
    *,
    plan_path: Path,
    submission_path: Path,
    controller_path: Path,
    release_path: Path,
    output: Path,
) -> dict[str, Any]:
    """Join exact export bytes to their observed dev controller and UID release."""
    from . import miles_hf_export as hf

    if output.exists() or output.is_symlink():
        raise FileExistsError("HF export job acceptance already exists")
    plan, plan_file_sha256 = _snapshot(plan_path)
    submission, submission_file_sha256 = _snapshot(submission_path)
    validate_plan(plan, check_files=True)
    validate_submission_binding(submission, plan, check_files=True)
    if plan.get("stage") != "export":
        raise ValueError("HF export job acceptance requires an export plan")
    root = Path(plan["output_root"])
    expected = {
        controller_path: root / "HF_EXPORT_CONTROLLER_TERMINAL.json",
        release_path: root / "HF_EXPORT_RELEASE.json",
        output: root / "HF_EXPORT_ACCEPTED.json",
    }
    if any(actual != wanted for actual, wanted in expected.items()):
        raise ValueError("HF export job acceptance evidence is outside its exact output root")
    controller, controller_file_sha256 = _snapshot(controller_path)
    release, release_file_sha256 = _snapshot(release_path)
    artifact = _validate_export_controller(controller, plan, submission)
    _validate_export_release(
        release,
        plan,
        submission,
        controller,
        controller_path,
        controller_file_sha256,
    )
    artifact_path = Path(controller["artifact_path"])
    _, artifact_file_sha256 = hf._snapshot(artifact_path)
    return _write(
        output,
        {
            "schema": EXPORT_ACCEPTED_SCHEMA,
            "status": "accepted",
            "export_plan_path": str(plan_path),
            "export_plan_file_sha256": plan_file_sha256,
            "export_plan_sha256": digest(plan),
            "submission_binding_path": str(submission_path),
            "submission_binding_file_sha256": submission_file_sha256,
            "submission_binding_sha256": submission["sha256"],
            "artifact_path": str(artifact_path),
            "artifact_file_sha256": artifact_file_sha256,
            "artifact_receipt_sha256": artifact["sha256"].removeprefix("sha256:"),
            "tensor_inventory_sha256": artifact["tensor_inventory_sha256"],
            "controller_terminal_path": str(controller_path),
            "controller_terminal_file_sha256": controller_file_sha256,
            "controller_terminal_sha256": controller["sha256"].removeprefix("sha256:"),
            "external_release_path": str(release_path),
            "external_release_file_sha256": release_file_sha256,
            "external_release_sha256": release["sha256"].removeprefix("sha256:"),
            "source_plan_sha256": artifact["source"]["source_plan_sha256"],
            "source_request_sha256": submission["source_request_sha256"],
            "runtime_bundle_sha256": submission["runtime_bundle_sha256"],
            "active_canary_binding": artifact["source"]["active_canary_binding"],
            "exact_active_canary_source_verified": True,
            "exact_job_execution_verified": True,
            "external_gpu_release_verified": True,
            "create_once_verified": True,
            "serving_qualified": False,
        },
    )


def validate_export_accepted(value: dict[str, Any], *, check_files: bool = True) -> dict[str, Any]:
    """Reopen a fully observed export job before the reload stage consumes it."""
    from . import miles_hf_export as hf
    from .rl_runtime import sealed

    sealed(value, EXPORT_ACCEPTED_SCHEMA)
    digest_fields = (
        "export_plan_file_sha256",
        "export_plan_sha256",
        "submission_binding_file_sha256",
        "submission_binding_sha256",
        "artifact_file_sha256",
        "artifact_receipt_sha256",
        "tensor_inventory_sha256",
        "controller_terminal_file_sha256",
        "controller_terminal_sha256",
        "external_release_file_sha256",
        "external_release_sha256",
        "source_plan_sha256",
        "source_request_sha256",
        "runtime_bundle_sha256",
    )
    if (
        set(value) != _EXPORT_ACCEPTED_FIELDS
        or value.get("status") != "accepted"
        or any(_SHA.fullmatch(str(value.get(key, ""))) is None for key in digest_fields)
        or any(
            value.get(key) is not True
            for key in (
                "exact_active_canary_source_verified",
                "exact_job_execution_verified",
                "external_gpu_release_verified",
                "create_once_verified",
            )
        )
        or value.get("serving_qualified") is not False
    ):
        raise ValueError("accepted HF export job receipt is incomplete")
    _bound_reference(value.get("active_canary_binding"), "active canary binding")
    if check_files:
        plan, plan_file_sha256 = _snapshot(Path(value["export_plan_path"]))
        submission, submission_file_sha256 = _snapshot(Path(value["submission_binding_path"]))
        controller, controller_file_sha256 = _snapshot(Path(value["controller_terminal_path"]))
        release, release_file_sha256 = _snapshot(Path(value["external_release_path"]))
        validate_plan(plan, check_files=True)
        validate_submission_binding(submission, plan, check_files=True)
        artifact = _validate_export_controller(controller, plan, submission)
        _validate_export_release(
            release,
            plan,
            submission,
            controller,
            Path(value["controller_terminal_path"]),
            controller_file_sha256,
        )
        artifact_path = Path(value["artifact_path"])
        reopened, artifact_file_sha256 = hf._snapshot(artifact_path)
        _inspect_plan_bound_export(plan, artifact_path, artifact_file_sha256)
        if (
            plan_file_sha256 != value["export_plan_file_sha256"]
            or digest(plan) != value["export_plan_sha256"]
            or submission_file_sha256 != value["submission_binding_file_sha256"]
            or submission["sha256"].removeprefix("sha256:")
            != value["submission_binding_sha256"].removeprefix("sha256:")
            or controller_file_sha256 != value["controller_terminal_file_sha256"]
            or controller["sha256"].removeprefix("sha256:")
            != value["controller_terminal_sha256"].removeprefix("sha256:")
            or release_file_sha256 != value["external_release_file_sha256"]
            or release["sha256"].removeprefix("sha256:")
            != value["external_release_sha256"].removeprefix("sha256:")
            or reopened != artifact
            or artifact_file_sha256 != value["artifact_file_sha256"]
            or artifact["sha256"].removeprefix("sha256:")
            != value["artifact_receipt_sha256"].removeprefix("sha256:")
            or artifact["tensor_inventory_sha256"]
            != value["tensor_inventory_sha256"].removeprefix("sha256:")
            or artifact["source"]["active_canary_binding"] != value["active_canary_binding"]
            or plan["source"]["active_canary_binding"] != value["active_canary_binding"]
            or submission["source_request_sha256"].removeprefix("sha256:")
            != value["source_request_sha256"].removeprefix("sha256:")
            or submission["runtime_bundle_sha256"].removeprefix("sha256:")
            != value["runtime_bundle_sha256"].removeprefix("sha256:")
        ):
            raise ValueError("accepted HF export job evidence changed")
    return {
        "export": {
            "path": value["artifact_path"],
            "file_sha256": value["artifact_file_sha256"].removeprefix("sha256:"),
            "receipt_sha256": value["artifact_receipt_sha256"].removeprefix("sha256:"),
            "tensor_inventory_sha256": value["tensor_inventory_sha256"].removeprefix("sha256:"),
            "active_canary_binding": value["active_canary_binding"],
        },
        "source_request_sha256": value["source_request_sha256"],
        "runtime_bundle_sha256": value["runtime_bundle_sha256"],
    }


def _deadline_expired(_signum: int, _frame: object) -> None:
    raise TimeoutError("Miles HF job exceeded its prepared wall-clock deadline")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    try:
        plan = json.loads(args.plan.read_text())
        if not isinstance(plan, dict) or digest(plan) != args.sha256:
            raise ValueError("HF job plan digest changed")
        previous_alarm = signal.signal(signal.SIGALRM, _deadline_expired)
        signal.alarm(plan["deadline_seconds"])
        try:
            result = run(plan)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous_alarm)
        print(
            json.dumps({key: result[key] for key in ("schema", "status", "sha256")}, sort_keys=True)
        )
    except BaseException as error:
        root = Path(os.environ.get("RUN_DIR", "/nonexistent"))
        if root.is_dir():
            with suppress(BaseException):
                _write(
                    root / "FAILED.json",
                    {
                        "schema": PLAN_SCHEMA,
                        "error_class": type(error).__name__,
                        "failed_at": time.time(),
                    },
                )
        print(json.dumps({"status": "failed", "error_class": type(error).__name__}))
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
