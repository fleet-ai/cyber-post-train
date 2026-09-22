"""Exact one-GPU BF16 reload gate for a sealed prod9 SkyRL checkpoint.

This is a deliberately narrow successor to the proven Qwen3.8 LR30 reload
check.  It binds one prod9 plan, its terminal checkpoint receipt and its BF16
export, then requires the generic Jobs API to render one c1/q1,
one-node/one-GPU RayJob.  The live rail requires two stable server previews, a
fresh capacity census, an already armed exact-UID cleanup observer and one
durable POST intent.  It never retries a submission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import UUID

SPEC_SCHEMA = "cyber_skyrl_prod9_reload_spec_v1"
PREVIEW_SCHEMA = "cyber_skyrl_prod9_reload_preview_v1"
AUTHORIZATION_SCHEMA = "cyber_skyrl_prod9_reload_authorization_v1"
CAPACITY_SCHEMA = "cyber_skyrl_prod9_reload_capacity_gate_v1"
CREATED_SCHEMA = "cyber_skyrl_prod9_reload_created_v1"
IMAGE = (
    "661864827319.dkr.ecr.us-east-1.amazonaws.com/fleet/skyrl-train@sha256:"
    "ba288751cd227c5be146d28f4a03237545d87d2cbd4c48464945b17fde566ff4"
)
MODULE = "training.skyrl_prod9_reload"
MAXIMUM_SECONDS = 1800
HARD_CHILD_SECONDS = 1200
NAMESPACE = "fleet-train-jobs"
DEV_CONTEXT = "nebius-mk8s-fleetai-training-dev-e04p03enwk5c0va9tb"
PROD_CONTEXT = "nebius-mk8s-fleetai-training-e04zw4ye1k7wczqdw6"
FAILURE_ALERT_ANNOTATION = "fleet.ai/failure-alerts"
FAILURE_ALERT_OFF = "off"
RUNTIME_FILES = (
    "training/skyrl_prod9_reload.py",
    "training/checkpoints.py",
    "training/export_check.py",
    "training/io.py",
    "training/post_sft_artifacts.py",
    "training/sft_runtime.py",
)


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _seal(value: dict[str, Any]) -> dict[str, Any]:
    body = {key: item for key, item in value.items() if key != "sha256"}
    return {**body, "sha256": "sha256:" + _digest(body)}


def _validated_seal(value: object, schema: str) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema") != schema or value != _seal(value):
        raise ValueError("prod9 reload evidence digest/schema changed")
    return value


def _hex(value: object, *, prefix: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError("prod9 reload digest is invalid")
    raw = value.removeprefix("sha256:")
    if re.fullmatch(r"[0-9a-f]{64}", raw) is None or prefix != value.startswith("sha256:"):
        raise ValueError("prod9 reload digest is invalid")
    return value


def _file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _receipt(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("receipt_sha256") != _digest(
        {key: item for key, item in value.items() if key != "receipt_sha256"}
    ):
        raise ValueError("prod9 reload receipt digest changed")
    return value


def _spec_identity(spec: object) -> dict[str, Any]:
    value = _validated_seal(spec, SPEC_SCHEMA)
    required = {
        "schema",
        "plan_sha256",
        "name",
        "run_dir",
        "image",
        "model",
        "checkpoint",
        "export",
        "runtime",
        "resources",
        "scientific_work",
        "sha256",
    }
    model, checkpoint, export = value.get("model"), value.get("checkpoint"), value.get("export")
    if (
        set(value) != required
        or _hex(value.get("plan_sha256"), prefix=False) != value["plan_sha256"]
        or not isinstance(model, dict)
        or set(model) != {"repo", "revision", "base_root", "export_root"}
        or model.get("repo") != "Qwen/Qwen3.8-27B"
        or not isinstance(model.get("revision"), str)
        or not model["revision"]
        or not isinstance(checkpoint, dict)
        or set(checkpoint)
        != {
            "manifest_path",
            "manifest_file_sha256",
            "receipt_sha256",
            "checkpoint_path",
            "optimizer_step",
        }
        or not isinstance(export, dict)
        or set(export)
        != {
            "receipt_path",
            "file_sha256",
            "receipt_sha256",
            "source_checkpoint_receipt_sha256",
            "source_manifest_file_sha256",
            "source_plan_sha256",
            "optimizer_step",
        }
        or type(checkpoint.get("optimizer_step")) is not int
        or checkpoint["optimizer_step"] < 1
        or export.get("optimizer_step") != checkpoint["optimizer_step"]
        or export.get("source_checkpoint_receipt_sha256") != checkpoint.get("receipt_sha256")
        or export.get("source_manifest_file_sha256") != checkpoint.get("manifest_file_sha256")
        or export.get("source_plan_sha256") != value["plan_sha256"]
        or value.get("image") != IMAGE
        or value.get("runtime")
        != {
            "module": MODULE,
            "files_sha256": _runtime_hashes(),
            "hard_child_seconds": HARD_CHILD_SECONDS,
        }
        or value.get("resources")
        != {
            "nodes": 1,
            "gpus": 1,
            "priority_class": "c1",
            "queue_priority_class": "q1",
            "maximum_seconds": MAXIMUM_SECONDS,
        }
        or value.get("scientific_work")
        != {
            "optimizer_steps": 0,
            "synthetic_only": True,
            "serving_qualified": False,
        }
    ):
        raise ValueError("prod9 reload specification changed")
    for key in ("manifest_file_sha256", "receipt_sha256"):
        _hex(checkpoint.get(key), prefix=False)
    for key in (
        "file_sha256",
        "receipt_sha256",
        "source_checkpoint_receipt_sha256",
        "source_manifest_file_sha256",
        "source_plan_sha256",
    ):
        _hex(export.get(key), prefix=False)
    name, run_dir = value.get("name"), value.get("run_dir")
    if (
        not isinstance(name, str)
        or re.fullmatch(r"[a-z0-9](?:[-a-z0-9]{0,61}[a-z0-9])?", name) is None
        or run_dir != "/mnt/sfs/jobs/" + name
        or model.get("export_root") != str(Path(export["receipt_path"]).parent)
        or export["receipt_path"] != model["export_root"] + "/EXPORT.json"
    ):
        raise ValueError("prod9 reload path identity changed")
    return value


def _runtime_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    return {name: _file_sha256(root / name) for name in RUNTIME_FILES}


def _reload_run_dir(plan: dict[str, Any], step: int) -> str:
    root = PurePosixPath(plan.get("output_root", ""))
    run_name = plan.get("run_name")
    if (
        not isinstance(run_name, str)
        or root.parts[:4] != ("/", "mnt", "sfs", "jobs")
        or len(root.parts) != 5
        or root.name != run_name
        or str(root) != plan.get("output_root")
    ):
        raise ValueError("prod9 reload requires the exact canonical training output root")
    return str(root.with_name(root.name + f"-p{step}-reload-v1"))


def build_spec(
    plan: dict[str, Any],
    checkpoint_manifest: dict[str, Any],
    export_receipt: dict[str, Any],
    *,
    checkpoint_manifest_file_sha256: str,
    export_file_sha256: str,
) -> dict[str, Any]:
    """Bind public terminal receipts to one fixed prod9 reload identity."""
    from cyber_post_train.jobs import digest

    from . import skyrl_posttrain, skyrl_prod9_hardening

    if plan.get("schema") != "cyber_skyrl_prod9_training_v1":
        raise ValueError("prod9 reload requires the fresh training schema")
    args = skyrl_posttrain._validate_plan(plan)
    skyrl_posttrain.verify_manifest(checkpoint_manifest, check_files=False)
    export = _receipt(export_receipt)
    checkpoint_file = _hex(checkpoint_manifest_file_sha256, prefix=False)
    export_file = _hex(export_file_sha256, prefix=False)
    expected_name = plan["run_name"] + f"-p{args.steps}-reload-v1"
    expected_reload_root = _reload_run_dir(plan, args.steps)
    expected_export_root = plan["output_root"] + f"/hf-export-step{args.steps}-v1"
    expected_manifest_path = plan["output_root"] + f"/checkpoint-seals-v1/step-{args.steps}.json"
    expected_checkpoint_path = plan["output_root"] + f"/checkpoints/global_step_{args.steps}"
    terminal_paths = skyrl_prod9_hardening.terminal_paths(plan)
    if (
        str(terminal_paths["checkpoint_manifest"]) != expected_manifest_path
        or str(terminal_paths["export"].parent) != expected_export_root
        or checkpoint_manifest.get("source_plan") != plan
        or checkpoint_manifest.get("source_plan_sha256") != digest(plan)
        or checkpoint_manifest.get("checkpoint_path") != expected_checkpoint_path
        or checkpoint_manifest.get("optimizer_step") != args.steps
        or export.get("schema") != "cyber_native_skyrl_rl_hf_export_v1"
        or export.get("source_checkpoint_receipt_sha256")
        != checkpoint_manifest.get("receipt_sha256")
        or export.get("source_manifest_file_sha256") != checkpoint_file
        or export.get("source_plan_sha256") != digest(plan)
        or export.get("model_repo") != plan["model"]["repo"]
        or export.get("model_revision") != plan["model"]["revision"]
        or export.get("output_root") != expected_export_root
        or export.get("optimizer_step") != args.steps
        or export.get("optimizer_steps_executed") != 0
        or export.get("gpu_reload_verified") is not False
        or export.get("dtype") != "BF16"
        or export.get("all_output_tensors_reopened_equal") is not True
        or export.get("source_inventory_sizes_mtimes_unchanged") is not True
    ):
        raise ValueError("prod9 reload checkpoint/export identity changed")
    return _seal(
        {
            "schema": SPEC_SCHEMA,
            "plan_sha256": digest(plan),
            "name": expected_name,
            "run_dir": expected_reload_root,
            "image": IMAGE,
            "model": {
                "repo": plan["model"]["repo"],
                "revision": plan["model"]["revision"],
                "base_root": plan["model"]["root"],
                "export_root": expected_export_root,
            },
            "checkpoint": {
                "manifest_path": expected_manifest_path,
                "manifest_file_sha256": checkpoint_file,
                "receipt_sha256": checkpoint_manifest["receipt_sha256"],
                "checkpoint_path": expected_checkpoint_path,
                "optimizer_step": args.steps,
            },
            "export": {
                "receipt_path": expected_export_root + "/EXPORT.json",
                "file_sha256": export_file,
                "receipt_sha256": export["receipt_sha256"],
                "source_checkpoint_receipt_sha256": export["source_checkpoint_receipt_sha256"],
                "source_manifest_file_sha256": export["source_manifest_file_sha256"],
                "source_plan_sha256": export["source_plan_sha256"],
                "optimizer_step": args.steps,
            },
            "runtime": {
                "module": MODULE,
                "files_sha256": _runtime_hashes(),
                "hard_child_seconds": HARD_CHILD_SECONDS,
            },
            "resources": {
                "nodes": 1,
                "gpus": 1,
                "priority_class": "c1",
                "queue_priority_class": "q1",
                "maximum_seconds": MAXIMUM_SECONDS,
            },
            "scientific_work": {
                "optimizer_steps": 0,
                "synthetic_only": True,
                "serving_qualified": False,
            },
        }
    )


def job_request(spec: dict[str, Any]) -> dict[str, Any]:
    """Build the proven one-node Qwen3.8 runtime bundle without previewing it."""
    from cyber_post_train.jobs import bundled_request, validate_request

    value = _spec_identity(spec)
    root = Path(__file__).resolve().parents[1]
    files = {name: (root / name).read_text() for name in RUNTIME_FILES}
    files["training/__init__.py"] = ""
    files["spec.json"] = json.dumps(value, sort_keys=True, separators=(",", ":"))
    source_name = (
        "q38-rld-" + value["plan_sha256"][:8] + "-" + str(value["checkpoint"]["optimizer_step"])
    )
    request = bundled_request(
        {
            "name": source_name,
            "title": "Qwen3.8 prod9 exact BF16 one-GPU reload " + value["plan_sha256"][:12],
            "run_dir": value["run_dir"],
            "image": IMAGE,
            "workers": 1,
            "gpus_per_worker": 1,
            "resources": {
                "cpu_request": "8",
                "cpu_limit": "8",
                "memory_request": "64Gi",
                "memory_limit": "128Gi",
            },
            "priority_class": "c1",
            "requeueIfPreempted": False,
            "failureAlerts": False,
            "secrets": [],
            "image_pull_secrets": [],
            "env": {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "PYTHONUNBUFFERED": "1",
                "PYTHONPATH": value["run_dir"] + "/.runtime:/opt/skyrl",
            },
        },
        files,
        MODULE,
        ["--spec", "spec.json", "--sha256", value["sha256"]],
    )
    claim = value["run_dir"] + "/.prod9-reload-create-claim-v1"
    request["command"] = "mkdir " + shlex.quote(claim) + " && exec " + request["command"]
    validate_request(request)
    return request


def manifest(
    spec: dict[str, Any], request: dict[str, Any], source_preview: dict[str, Any]
) -> dict[str, Any]:
    """Accept the exact Jobs API render without transforming one byte."""
    from cyber_post_train.jobs import validate_preview, validate_request

    from . import skyrl_prod9_direct as direct

    value = _spec_identity(spec)
    if request != job_request(value):
        raise ValueError("prod9 reload request differs from the sealed specification")
    validate_request(request)
    source = direct._source(source_preview)
    placeholder = request["name"] + "-00000000"
    metadata = source.get("metadata", {})
    labels, annotations = metadata.get("labels", {}), metadata.get("annotations", {})
    if (
        source_preview.get("name") != placeholder
        or source.get("apiVersion") != "ray.io/v1"
        or source.get("kind") != "RayJob"
        or metadata.get("name") != placeholder
        or metadata.get("namespace") != NAMESPACE
        or labels.get("fleet.ai/run-id") != "00000000-0000-0000-0000-000000000000"
        or labels.get("fleet.ai/run-name") != request["name"]
        or labels.get("kueue.x-k8s.io/queue-name") != "training-lq"
        or labels.get("kueue.x-k8s.io/priority-class") != "q1"
        or annotations.get("fleet.ai/run-id") != "00000000-0000-0000-0000-000000000000"
        or annotations.get("fleet.ai/run-dir") != value["run_dir"]
        or annotations.get("fleet.ai/job-image") != IMAGE
        or annotations.get(FAILURE_ALERT_ANNOTATION) != FAILURE_ALERT_OFF
    ):
        raise ValueError("prod9 reload Jobs preview lacks exact root alert-off/admission")
    # The supported generic Jobs API validator independently proves the root
    # annotation, c1/q1 admission, one node/GPU, resources, image and env.
    validate_preview(request, source_preview)
    ray_spec = source.get("spec", {})
    if (
        ray_spec.get("entrypoint") != request["command"]
        or ray_spec.get("suspend") is not True
        or ray_spec.get("shutdownAfterJobFinishes") is not True
        or ray_spec.get("submissionMode") != "HTTPMode"
    ):
        raise ValueError("prod9 reload Jobs preview execution changed")
    if ray_spec.get("backoffLimit") != 0:
        raise ValueError("prod9 reload Jobs preview retry contract changed")
    cluster = ray_spec.get("rayClusterSpec", {})
    if cluster.get("workerGroupSpecs") not in (None, []):
        raise ValueError("prod9 reload must remain one physical GPU node")
    head = cluster.get("headGroupSpec", {}).get("template", {})
    pod = head.get("spec", {})
    containers = pod.get("containers", [])
    if len(containers) != 1 or pod.get("priorityClassName") != "c1":
        raise ValueError("prod9 reload preview has the wrong head Pod")
    container = containers[0]
    resources = container.get("resources", {})
    if (
        resources.get("requests", {}).get("nvidia.com/gpu") != 1
        or resources.get("limits", {}).get("nvidia.com/gpu") != 1
        or container.get("image") != IMAGE
    ):
        raise ValueError("prod9 reload preview is not exactly one GPU")
    generated_secret = placeholder + "-fleet-key"
    secret_names = [item.get("secretRef", {}).get("name") for item in container.get("envFrom", [])]
    if secret_names != [generated_secret]:
        raise ValueError("prod9 reload preview Secret bindings changed")
    if container.get("securityContext") != direct._runtime_context():
        raise ValueError("prod9 reload runtime user contract changed")
    if container.get("terminationMessagePath", "/dev/termination-log") != "/dev/termination-log":
        raise ValueError("prod9 reload termination receipt path changed")
    init = pod.get("initContainers", [])
    sfs = [item for item in init if item.get("name") == "sfs-init"]
    if len(sfs) != 1 or sfs[0].get("command") != [
        "sh",
        "-c",
        f"mkdir -p {value['run_dir']} && chown 1000:100 {value['run_dir']}",
    ]:
        raise ValueError("prod9 reload output initialization changed")
    _validate_manifest(value, request, source)
    return source


def _validate_manifest(
    spec: dict[str, Any], request: dict[str, Any], expected: dict[str, Any]
) -> None:
    value = _spec_identity(spec)
    try:
        metadata = expected["metadata"]
        ray_spec = expected["spec"]
        cluster = ray_spec["rayClusterSpec"]
        pod = cluster["headGroupSpec"]["template"]["spec"]
        container = pod["containers"][0]
        resources = container["resources"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("prod9 reload manifest is malformed") from exc
    if (
        expected.get("apiVersion") != "ray.io/v1"
        or expected.get("kind") != "RayJob"
        or metadata.get("name") != request["name"] + "-00000000"
        or metadata.get("namespace") != NAMESPACE
        or metadata.get("annotations", {}).get(FAILURE_ALERT_ANNOTATION) != FAILURE_ALERT_OFF
        or metadata.get("annotations", {}).get("fleet.ai/run-dir") != value["run_dir"]
        or metadata.get("labels", {}).get("kueue.x-k8s.io/queue-name") != "training-lq"
        or metadata.get("labels", {}).get("kueue.x-k8s.io/priority-class") != "q1"
        or ray_spec.get("entrypoint") != request["command"]
        or ray_spec.get("backoffLimit") != 0
        or ray_spec.get("shutdownAfterJobFinishes") is not True
        or ray_spec.get("suspend") is not True
        or cluster.get("workerGroupSpecs") not in (None, [])
        or pod.get("priorityClassName") != "c1"
        or len(pod.get("containers", [])) != 1
        or container.get("image") != IMAGE
        or [item.get("secretRef", {}).get("name") for item in container.get("envFrom", [])]
        != [request["name"] + "-00000000-fleet-key"]
        or container.get("securityContext")
        != {
            "allowPrivilegeEscalation": False,
            "privileged": False,
            "runAsGroup": 100,
            "runAsNonRoot": True,
            "runAsUser": 1000,
        }
        or resources.get("requests", {}).get("nvidia.com/gpu") != 1
        or resources.get("limits", {}).get("nvidia.com/gpu") != 1
    ):
        raise ValueError("prod9 reload manifest contract changed")


def validate_preview(
    spec: dict[str, Any],
    request: dict[str, Any],
    source_preview: dict[str, Any],
    expected: dict[str, Any],
    rendered: dict[str, Any],
    *,
    context: str,
) -> dict[str, Any]:
    """Prove a server-rendered root keeps the exact one-GPU/alert contract."""
    from cyber_post_train.jobs import digest

    from . import skyrl_prod9_direct as direct

    value = _spec_identity(spec)
    if expected != manifest(value, request, source_preview):
        raise ValueError("prod9 reload manifest changed before server preview")
    _validate_manifest(value, request, expected)
    if (
        context not in {DEV_CONTEXT, PROD_CONTEXT}
        or direct._strip_server_defaults(expected, rendered) != expected
    ):
        raise ValueError("prod9 reload server preview changed the exact RayJob")
    return _seal(
        {
            "schema": PREVIEW_SCHEMA,
            "status": "passed",
            "context": context,
            "spec_sha256": value["sha256"],
            "request_sha256": "sha256:" + digest(request),
            "manifest_sha256": "sha256:" + digest(expected),
            "server_render_sha256": "sha256:" + digest(rendered),
            "name": source_preview["name"],
            "run_name_prefix": request["name"],
            "nodes": 1,
            "gpus": 1,
            "priority": "c1",
            "queue_priority": "q1",
            "failure_alerts": "off",
            "submitted": False,
            "checked_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    )


def capacity_gate(
    spec: dict[str, Any],
    request: dict[str, Any],
    expected: dict[str, Any],
    *,
    reader: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Seal one fresh all-namespace 1-GPU capacity observation."""
    from cyber_post_train.gpu_capacity import ROLE_LABELS, CapacityError, live_capacity_census
    from cyber_post_train.jobs import digest

    from . import skyrl_prod9_hardening as hardening

    value = _spec_identity(spec)
    _validate_manifest(value, request, expected)
    reader = live_capacity_census if reader is None else reader
    try:
        census = reader(
            PROD_CONTEXT,
            owner_prefixes=hardening.PROJECT_OWNER_PREFIXES,
            max_nodes=hardening.PROJECT_MAX_NODES,
            max_gpus=hardening.PROJECT_MAX_GPUS,
            planned_nodes=1,
            planned_gpus=1,
        )
    except CapacityError as exc:
        raise ValueError("prod9 reload cross-namespace capacity census failed") from exc
    if not isinstance(census, dict):
        raise ValueError("prod9 reload capacity census is invalid")
    try:
        observed = datetime.fromisoformat(str(census.get("observed_at")).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("prod9 reload capacity timestamp is invalid") from exc
    current, projected = census.get("current"), census.get("projected")
    if (
        census.get("sha256")
        != digest({key: item for key, item in census.items() if key != "sha256"})
        or census.get("schema") != "cyber_project_gpu_capacity_census_v1"
        or census.get("scope")
        != {
            "kubernetes_namespaces": "all",
            "owner_prefixes": list(hardening.PROJECT_OWNER_PREFIXES),
            "ownership_labels": ROLE_LABELS,
        }
        or census.get("limits")
        != {"nodes": hardening.PROJECT_MAX_NODES, "gpus": hardening.PROJECT_MAX_GPUS}
        or census.get("planned") != {"nodes": 1, "gpus": 1}
        or census.get("qualified") is not True
        or census.get("problems") != []
        or not isinstance(current, dict)
        or not isinstance(projected, dict)
        or current.get("role_pod_counts", {}).get("unclassified") != 0
        or type(current.get("nodes")) is not int
        or type(current.get("gpus")) is not int
        or projected != {"nodes": current["nodes"] + 1, "gpus": current["gpus"] + 1}
        or projected["nodes"] > hardening.PROJECT_MAX_NODES
        or projected["gpus"] > hardening.PROJECT_MAX_GPUS
        or observed.tzinfo is None
        or not 0
        <= (datetime.now(UTC) - observed.astimezone(UTC)).total_seconds()
        <= hardening.CAPACITY_MAX_AGE_SECONDS
    ):
        raise ValueError("prod9 reload capacity proof is stale or incomplete")
    return _seal(
        {
            "schema": CAPACITY_SCHEMA,
            "status": "passed",
            "context": PROD_CONTEXT,
            "spec_sha256": value["sha256"],
            "request_sha256": "sha256:" + digest(request),
            "manifest_sha256": "sha256:" + digest(expected),
            "planned": {"nodes": 1, "gpus": 1},
            "observed_at": census["observed_at"],
            "capacity_census": census,
        }
    )


def _preview_set(
    spec: dict[str, Any],
    request: dict[str, Any],
    expected: dict[str, Any],
    previews: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    from cyber_post_train.jobs import digest

    from . import skyrl_prod9_direct as direct

    value = _spec_identity(spec)
    contexts = []
    for proof in previews:
        item = _validated_seal(proof, PREVIEW_SCHEMA)
        if (
            item.get("status") != "passed"
            or item.get("context") not in {DEV_CONTEXT, PROD_CONTEXT}
            or item.get("spec_sha256") != value["sha256"]
            or item.get("request_sha256") != "sha256:" + digest(request)
            or item.get("manifest_sha256") != "sha256:" + digest(expected)
            or item.get("name") != request["name"] + "-00000000"
            or item.get("run_name_prefix") != request["name"]
            or item.get("nodes") != 1
            or item.get("gpus") != 1
            or item.get("priority") != "c1"
            or item.get("queue_priority") != "q1"
            or item.get("failure_alerts") != "off"
            or item.get("submitted") is not False
        ):
            raise ValueError("prod9 reload server preview was not accepted")
        direct._fresh_at(item.get("checked_at"))
        contexts.append(item["context"])
    if len(previews) != 2 or sorted(contexts) != sorted({DEV_CONTEXT, PROD_CONTEXT}):
        raise ValueError("prod9 reload server preview set is incomplete")
    return previews


def _authorization(
    spec: dict[str, Any],
    request: dict[str, Any],
    source_preview: dict[str, Any],
    expected: dict[str, Any],
    *,
    dev_preview: dict[str, Any],
    prod_preview: dict[str, Any],
    observer: dict[str, Any],
) -> dict[str, Any]:
    from cyber_post_train.jobs import digest

    from . import skyrl_prod9_direct as direct
    from . import skyrl_prod9_hardening as hardening

    value = _spec_identity(spec)
    if expected != manifest(value, request, source_preview):
        raise ValueError("prod9 reload authorization manifest changed")
    previews = _preview_set(value, request, expected, [dev_preview, prod_preview])
    plan_sha256 = "sha256:" + value["plan_sha256"]
    manifest_sha256 = "sha256:" + digest(expected)
    operation_root = hardening.reload_operation_root(value)
    direct._canonical_operation_root(operation_root)
    armed = direct._jobs_api_prefix_guard(
        observer,
        operation_root=operation_root,
        purpose="reload",
        request=request,
        plan_sha256=plan_sha256,
        manifest_sha256=manifest_sha256,
        gpus=1,
        maximum_seconds=MAXIMUM_SECONDS,
    )
    return _seal(
        {
            "schema": AUTHORIZATION_SCHEMA,
            "status": "authorized_for_one_jobs_api_post",
            "spec": value,
            "request_sha256": "sha256:" + digest(request),
            "source_preview_sha256": "sha256:" + digest(source_preview),
            "manifest_sha256": manifest_sha256,
            "dev_preview": previews[0],
            "prod_preview": previews[1],
            "observer": armed,
            "operation_root": str(operation_root),
        }
    )


def authorize(
    spec: dict[str, Any],
    request: dict[str, Any],
    source_preview: dict[str, Any],
    expected: dict[str, Any],
    *,
    dev_preview: dict[str, Any],
    prod_preview: dict[str, Any],
    observer: dict[str, Any],
) -> dict[str, Any]:
    """Authorize one Jobs API POST only after previews and cleanup are live."""
    return _authorization(
        spec,
        request,
        source_preview,
        expected,
        dev_preview=dev_preview,
        prod_preview=prod_preview,
        observer=observer,
    )


def _duplicate_checks(
    spec: dict[str, Any],
    request: dict[str, Any],
    *,
    token: str,
    runner: Callable[..., subprocess.CompletedProcess[str]],
    jobs_factory: Any,
) -> dict[str, int]:
    from cyber_post_train.jobs import API_URLS

    value = _spec_identity(spec)
    if not isinstance(token, str) or not token:
        raise ValueError("prod9 reload Jobs API credential is required")
    inventories = 0
    names = {value["name"], request["name"]}
    for context in (DEV_CONTEXT, PROD_CONTEXT):
        for resource in ("rayjob", "raycluster", "job", "workload", "pod"):
            result = runner(
                [
                    "kubectl",
                    "--context",
                    context,
                    "--namespace",
                    NAMESPACE,
                    "get",
                    resource,
                    "--output=json",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode:
                raise ValueError("prod9 reload Kubernetes duplicate inventory failed")
            inventories += 1
            try:
                items = json.loads(result.stdout).get("items", [])
            except (AttributeError, ValueError) as exc:
                raise ValueError("prod9 reload Kubernetes duplicate inventory is invalid") from exc
            if not isinstance(items, list):
                raise ValueError("prod9 reload Kubernetes duplicate inventory is invalid")
            for item in items:
                metadata = item.get("metadata", {}) if isinstance(item, dict) else {}
                labels = metadata.get("labels", {}) if isinstance(metadata, dict) else {}
                annotations = metadata.get("annotations", {}) if isinstance(metadata, dict) else {}
                fields = [metadata.get("name", "")]
                if isinstance(labels, dict):
                    fields.extend(labels.values())
                if isinstance(annotations, dict):
                    fields.extend(annotations.values())
                serialized = json.dumps(item.get("spec", {}), sort_keys=True)
                if (
                    any(
                        field in names or any(str(field).startswith(name + "-") for name in names)
                        for field in fields
                    )
                    or value["run_dir"] in fields
                    or value["run_dir"] in serialized
                ):
                    raise ValueError("prod9 reload Kubernetes identity/output already exists")
    rows_checked = 0
    for target in ("dev", "prod"):
        with jobs_factory(token, base_url=API_URLS[target]) as client:
            rows = client.all_runs()
        rows_checked += len(rows)
        for row in rows:
            name = str(row.get("name", ""))
            if (
                row.get("run_dir") == value["run_dir"]
                or name in names
                or any(name.startswith(item + "-") for item in names)
                or row.get("title") == request["title"]
            ):
                raise ValueError("prod9 reload Jobs API history owns this identity/output")
    return {
        "kubernetes_inventories_checked": inventories,
        "jobs_api_rows_checked": rows_checked,
    }


def create_once(
    directory: Path,
    spec: dict[str, Any],
    request: dict[str, Any],
    source_preview: dict[str, Any],
    expected: dict[str, Any],
    authorization: dict[str, Any],
    *,
    token: str,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    jobs_factory: Any = None,
    capacity_reader: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Perform the sole reload Jobs API POST; an uncertain result stops replay."""
    from cyber_post_train.jobs import API_URLS, Jobs, digest

    from . import skyrl_prod9_direct as direct
    from . import skyrl_prod9_hardening as hardening

    value = _spec_identity(spec)
    jobs_factory = Jobs if jobs_factory is None else jobs_factory
    canonical = hardening.reload_operation_root(value)
    journal = canonical / "PROD9_RELOAD_RAYJOB_CREATE.jsonl"
    if journal.exists() or journal.is_symlink():
        raise ValueError("prod9 reload create intent exists; reconcile, never retry")
    auth = _validated_seal(authorization, AUTHORIZATION_SCHEMA)
    if (
        directory.is_symlink()
        or not directory.is_dir()
        or directory.resolve() != canonical
        or Path(auth.get("operation_root", "")) != canonical
    ):
        raise ValueError("prod9 reload create directory differs from its operation root")
    if auth != _authorization(
        value,
        request,
        source_preview,
        expected,
        dev_preview=auth["dev_preview"],
        prod_preview=auth["prod_preview"],
        observer=auth["observer"],
    ):
        raise ValueError("prod9 reload authorization changed")
    duplicate = _duplicate_checks(
        value,
        request,
        token=token,
        runner=runner,
        jobs_factory=jobs_factory,
    )
    plan_sha256 = "sha256:" + value["plan_sha256"]
    request_sha256 = "sha256:" + digest(request)
    manifest_sha256 = "sha256:" + digest(expected)
    with jobs_factory(token, base_url=API_URLS["prod"]) as client:
        live_source_preview = client.preview(request)
        live_expected = manifest(value, request, live_source_preview)
        if live_source_preview != source_preview or live_expected != expected:
            raise ValueError("prod9 reload live Jobs API preview changed after authorization")
        rendered = direct.server_dry_run(live_expected, context=PROD_CONTEXT, runner=runner)
        live_preview_proof = validate_preview(
            value,
            request,
            live_source_preview,
            live_expected,
            rendered,
            context=PROD_CONTEXT,
        )
        capacity = capacity_gate(
            value,
            request,
            live_expected,
            reader=capacity_reader,
        )
        direct._jobs_api_prefix_guard(
            auth["observer"],
            operation_root=canonical,
            purpose="reload",
            request=request,
            plan_sha256=plan_sha256,
            manifest_sha256=manifest_sha256,
            gpus=1,
            maximum_seconds=MAXIMUM_SECONDS,
        )
        direct._fresh_at(capacity.get("observed_at"), maximum_age=120)
        for preview in (auth["dev_preview"], auth["prod_preview"]):
            direct._fresh_at(preview.get("checked_at"))
        direct._write_once_fsynced(
            journal,
            {
                "state": "POST_INTENT_DO_NOT_RETRY",
                "spec_sha256": value["sha256"],
                "plan_sha256": plan_sha256,
                "request_sha256": request_sha256,
                "manifest_sha256": manifest_sha256,
                "authorization_sha256": auth["sha256"],
                "live_jobs_preview_sha256": "sha256:" + digest(live_source_preview),
                "live_preview_proof": live_preview_proof,
                "capacity_gate": capacity,
                "duplicate_checks": duplicate,
            },
        )
        response = client.request("POST", "/v1/runs", json=request)
    try:
        jobs_api_run_name = response["name"]
        jobs_api_run_id = str(UUID(response["job_id"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("prod9 reload API response is ambiguous; never retry") from exc
    if (
        not isinstance(jobs_api_run_name, str)
        or re.fullmatch(re.escape(request["name"]) + r"-[a-f0-9]{8}", jobs_api_run_name) is None
        or response.get("run_dir") not in (None, request["run_dir"])
    ):
        raise ValueError("prod9 reload API returned another identity; never retry")
    normalized_response = {
        "name": jobs_api_run_name,
        "job_id": jobs_api_run_id,
        "run_dir": response.get("run_dir") or request["run_dir"],
        "status": response.get("status"),
        "created_at": response.get("created_at"),
    }
    with journal.open("a") as stream:
        stream.write(
            json.dumps(
                {"state": "POST_RESPONSE", **normalized_response},
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )
        stream.flush()
        os.fsync(stream.fileno())
    binding = direct._bind_jobs_api_created(
        operation_root=canonical,
        purpose="reload",
        request=request,
        plan_sha256=plan_sha256,
        manifest_sha256=manifest_sha256,
        maximum_seconds=MAXIMUM_SECONDS,
        expected_gpus=1,
        response=normalized_response,
        runner=runner,
    )
    proof = _seal(
        {
            "schema": CREATED_SCHEMA,
            "status": "submitted_once_and_bound_exact_uid",
            "spec_sha256": value["sha256"],
            "plan_sha256": plan_sha256,
            "request_sha256": request_sha256,
            "manifest_sha256": manifest_sha256,
            "authorization_sha256": auth["sha256"],
            "live_jobs_preview_sha256": "sha256:" + digest(live_source_preview),
            "live_preview_proof_sha256": live_preview_proof["sha256"],
            "capacity_gate_sha256": capacity["sha256"],
            "jobs_api_run_name": jobs_api_run_name,
            "jobs_api_run_id": jobs_api_run_id,
            "rayjob_name": binding["rayjob_name"],
            "rayjob_uid": binding["rayjob_uid"],
            "creator_binding_sha256": binding["sha256"],
            "created_at": binding["rayjob_created_at"],
            "failure_alerts": binding["failure_alerts"],
            "priority": "c1",
            "queue_priority": "q1",
            "nodes": 1,
            "gpus": 1,
        }
    )
    with journal.open("a") as stream:
        stream.write(json.dumps(proof, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return proof


def _inspect_export(spec: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    from .checkpoints import receipt
    from .post_sft_artifacts import QWEN36_EXACT_MTP_OMISSION_KEYS, _safetensor_layout
    from .sft_runtime import _checked_file

    value = _spec_identity(spec)
    export = value["export"]
    path = Path(export["receipt_path"])
    _checked_file(path, export["file_sha256"])
    proof = receipt(path)
    root = path.parent
    if (
        proof.get("schema") != "cyber_native_skyrl_rl_hf_export_v1"
        or proof.get("receipt_sha256") != export["receipt_sha256"]
        or proof.get("source_checkpoint_receipt_sha256")
        != export["source_checkpoint_receipt_sha256"]
        or proof.get("source_manifest_file_sha256") != export["source_manifest_file_sha256"]
        or proof.get("source_plan_sha256") != value["plan_sha256"]
        or proof.get("model_repo") != value["model"]["repo"]
        or proof.get("model_revision") != value["model"]["revision"]
        or proof.get("output_root") != str(root)
        or proof.get("optimizer_step") != export["optimizer_step"]
        or proof.get("optimizer_steps_executed") != 0
        or proof.get("gpu_reload_verified") is not False
        or proof.get("dtype") != "BF16"
        or proof.get("all_output_tensors_reopened_equal") is not True
        or proof.get("source_inventory_sizes_mtimes_unchanged") is not True
        or set(proof.get("restored_base_tensors", [])) != set(QWEN36_EXACT_MTP_OMISSION_KEYS)
        or {item.name for item in root.iterdir()} != set(proof.get("files", {})) | {path.name}
    ):
        raise ValueError("prod9 reload export differs from the sealed identity")
    for name, file_spec in proof["files"].items():
        item = root / name
        _checked_file(item, file_spec["sha256"])
        if item.stat().st_size != file_spec["bytes"]:
            raise ValueError("prod9 reload export payload size changed")
    layout, _ = _safetensor_layout(root)
    if len(layout) != proof["trained_tensors"] + len(proof["restored_base_tensors"]) or any(
        item["dtype"] != "BF16" for item in layout.values()
    ):
        raise ValueError("prod9 reload export tensor contract changed")
    return proof, layout


def validate_source(spec: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reopen the exact checkpoint receipt and complete BF16 export."""
    from .checkpoints import receipt
    from .sft_runtime import _checked_file, _unsigned_digest

    value = _spec_identity(spec)
    checkpoint = value["checkpoint"]
    manifest_path = Path(checkpoint["manifest_path"])
    _checked_file(manifest_path, checkpoint["manifest_file_sha256"])
    manifest_receipt = receipt(manifest_path)
    source_plan = manifest_receipt.get("source_plan")
    source_model = source_plan.get("model") if isinstance(source_plan, dict) else None
    if (
        manifest_receipt.get("schema") != "cyber_native_skyrl_rl_checkpoint_manifest_v1"
        or manifest_receipt.get("receipt_sha256") != checkpoint["receipt_sha256"]
        or not isinstance(source_plan, dict)
        or _unsigned_digest(source_plan) != value["plan_sha256"]
        or manifest_receipt.get("source_plan_sha256") != value["plan_sha256"]
        or source_plan.get("schema") != "cyber_skyrl_prod9_training_v1"
        or not isinstance(source_model, dict)
        or source_model.get("repo") != value["model"]["repo"]
        or source_model.get("revision") != value["model"]["revision"]
        or source_model.get("root") != value["model"]["base_root"]
        or source_plan.get("run_name") + f"-p{checkpoint['optimizer_step']}-reload-v1"
        != value["name"]
        or manifest_receipt.get("checkpoint_path") != checkpoint["checkpoint_path"]
        or manifest_receipt.get("optimizer_step") != checkpoint["optimizer_step"]
        or manifest_receipt.get("optimizer_update_verified") is not True
        or manifest_receipt.get("source_inputs_unchanged") is not True
    ):
        raise ValueError("prod9 reload checkpoint identity changed")
    export_receipt, layout = _inspect_export(value)
    return export_receipt, layout


def run_check(spec: dict[str, Any], output: Path) -> dict[str, Any]:
    """Load the exact BF16 export and run the fixed finite two-token check."""
    import torch
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer

    from .export_check import enable_native_patch, model_contract, synthetic_forward
    from .sft_runtime import write_receipt

    value = _spec_identity(spec)
    if output != Path(value["run_dir"]) / "GPU_CHECK.json":
        raise ValueError("prod9 reload GPU receipt path changed")
    if torch.cuda.is_available() is not True or torch.cuda.device_count() != 1:
        raise ValueError("prod9 reload requires exactly one visible GPU")
    proof, layout = validate_source(value)
    root = Path(value["model"]["export_root"])
    kwargs = {"local_files_only": True, "trust_remote_code": False}
    config = AutoConfig.from_pretrained(root, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(root, **kwargs)
    config.dtype = torch.bfloat16
    config._attn_implementation = "eager"
    torch.cuda.set_device(0)
    model, info = AutoModelForImageTextToText.from_pretrained(
        root,
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        attn_implementation="eager",
        output_loading_info=True,
        **kwargs,
    )
    if any(info.get(key) for key in ("missing_keys", "mismatched_keys", "error_msgs")) or (
        set(info.get("unexpected_keys", [])) - set(proof["restored_base_tensors"])
    ):
        raise ValueError("prod9 reload HF load reported tensor drift")
    patched = enable_native_patch(model)
    contract = model_contract(model, layout, proof["restored_base_tensors"], gpu=True)
    smoke = synthetic_forward(model, tokenizer, device="cuda:0")
    torch.cuda.synchronize()
    _inspect_export(value)
    result = {
        "schema": "cyber_hf_export_check_v1",
        "status": "passed",
        "export_sha256": value["export"]["file_sha256"],
        "export_receipt_sha256": proof["receipt_sha256"],
        "checkpoint_manifest_file_sha256": value["checkpoint"]["manifest_file_sha256"],
        "checkpoint_receipt_sha256": value["checkpoint"]["receipt_sha256"],
        "source_plan_sha256": value["plan_sha256"],
        "model_repo": value["model"]["repo"],
        "model_revision": value["model"]["revision"],
        "checker_sha256": _file_sha256(Path(__file__)),
        "optimizer_steps_executed": 0,
        "gpus": 1,
        "gpu_reload_verified": True,
        "serving_qualified": False,
        "synthetic_only": True,
        "source_unchanged": True,
        "attention_implementation": "eager",
        "loader_contract": contract,
        "patched_linear_layers": patched,
        "gpu_name": torch.cuda.get_device_name(0),
        "peak_memory_bytes": torch.cuda.max_memory_allocated(0),
        **smoke,
    }
    write_receipt(output, result)
    return _receipt(json.loads(output.read_bytes()))


def _write_termination_receipt(value: dict[str, Any]) -> None:
    path = Path("/dev/termination-log")
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")))


def run_parent(spec: dict[str, Any]) -> None:
    from .checkpoints import receipt
    from .sft_runtime import write_receipt

    value = _spec_identity(spec)
    output = Path(value["run_dir"])
    write_receipt(
        output / "STARTED.json",
        {
            "schema": "cyber_skyrl_prod9_reload_start_v1",
            "started_at": time.time(),
            "spec_sha256": value["sha256"],
            "export_file_sha256": value["export"]["file_sha256"],
            "checkpoint_receipt_sha256": value["checkpoint"]["receipt_sha256"],
            "gpus": 1,
            "optimizer_steps_executed": 0,
            "hard_child_seconds": HARD_CHILD_SECONDS,
        },
    )
    try:
        with (output / "private-synthetic-loader.log").open("x") as log:
            os.chmod(log.name, 0o600)
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    MODULE,
                    "--spec",
                    "spec.json",
                    "--sha256",
                    value["sha256"],
                    "--child",
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=HARD_CHILD_SECONDS,
                check=True,
            )
        gpu = receipt(output / "GPU_CHECK.json")
        if (
            gpu.get("status") != "passed"
            or gpu.get("source_plan_sha256") != value["plan_sha256"]
            or gpu.get("checkpoint_receipt_sha256") != value["checkpoint"]["receipt_sha256"]
            or gpu.get("export_receipt_sha256") != value["export"]["receipt_sha256"]
            or gpu.get("gpus") != 1
            or gpu.get("optimizer_steps_executed") != 0
            or gpu.get("gpu_reload_verified") is not True
            or gpu.get("finite_logits") is not True
            or gpu.get("generated_tokens") != 2
            or gpu.get("source_unchanged") is not True
            or gpu.get("serving_qualified") is not False
        ):
            raise ValueError("prod9 reload child receipt changed")
        _write_termination_receipt(gpu)
        print(
            json.dumps(
                {
                    "status": "passed",
                    "receipt_sha256": gpu["receipt_sha256"],
                    "gpus": 1,
                    "optimizer_steps_executed": 0,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    except BaseException as exc:
        write_receipt(
            output / "FAILED.json",
            {
                "schema": "cyber_skyrl_prod9_reload_failure_v1",
                "error_class": type(exc).__name__,
                "spec_sha256": value["sha256"],
                "gpus": 1,
                "optimizer_steps_executed": 0,
            },
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    spec = json.loads(args.spec.read_bytes())
    value = _spec_identity(spec)
    if args.sha256 != value["sha256"]:
        raise ValueError("prod9 reload runtime specification digest changed")
    if args.child:
        run_check(value, Path(value["run_dir"]) / "GPU_CHECK.json")
    else:
        run_parent(value)


if __name__ == "__main__":
    main()
