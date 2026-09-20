"""Create-once direct root-RayJob rail for the sealed prod4 reward canary.

The Jobs API preview remains the topology source of truth.  This module removes
only API-controller bindings that do not exist for a direct root RayJob, adds
the mandatory failed-job-alert opt-out and explicit image user, and replaces
the preview placeholder identity with a deterministic direct identity.  Model,
data, reward, optimizer, checkpoint, image, resources and entrypoint are never
changed.
"""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

import yaml

from cyber_post_train.jobs import (
    API_URLS,
    FAILURE_ALERT_ANNOTATION,
    FAILURE_ALERT_OFF,
    Jobs,
    JobsError,
    bundled_request,
    digest,
)

from . import skyrl_training

DEV_CONTEXT = "nebius-mk8s-fleetai-training-dev-e04p03enwk5c0va9tb"
PROD_CONTEXT = "nebius-mk8s-fleetai-training-e04zw4ye1k7wczqdw6"
NAMESPACE = "fleet-train-jobs"
RUN_NAME = "chris-q38-rlreward-prod4"
STAGE_NAME = "chris-q38-prod4-data-v1"
PREFLIGHT_NAME = "chris-q38-prod4-preflight-v4"
STAGE_RECEIPT = "/dev/termination-log"
UPLOAD = Path("/tmp/autoresearch-upload.tar.gz")
PACKET_SCHEMA = "cyber_skyrl_reward_direct_rayjob_packet_v1"
PREVIEW_SCHEMA = "cyber_skyrl_reward_direct_rayjob_preview_v1"
STAGE_SCHEMA = "cyber_skyrl_reward_data_stage_v1"
STAGE_RECEIPT_SCHEMA = "cyber_skyrl_reward_data_stage_receipt_v1"
PREFLIGHT_RECEIPT_SCHEMA = "cyber_skyrl_reward_cpu_preflight_v1"
PREFLIGHT_REJECTION_SCHEMA = "cyber_skyrl_reward_cpu_preflight_rejection_v1"
CPU_PREVIEW_SCHEMA = "cyber_skyrl_reward_cpu_job_preview_v1"
AUTHORIZATION_SCHEMA = "cyber_skyrl_reward_direct_authorization_v1"
CREATED_SCHEMA = "cyber_skyrl_reward_direct_created_v1"
MAXIMUM_SECONDS = 8 * 60 * 60 + 5 * 60 + 30 * 60
RUNTIME_UID = 1000
RUNTIME_GID = 100
PVC = "sfs-shared"


def _seal(value: dict[str, Any]) -> dict[str, Any]:
    body = {key: item for key, item in value.items() if key != "sha256"}
    return {**body, "sha256": "sha256:" + digest(body)}


def _validate_seal(value: object, schema: str) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema") != schema or value != _seal(value):
        raise JobsError("prod4 direct evidence digest changed")
    return value


def _timestamp(value: object) -> datetime:
    if not isinstance(value, str):
        raise JobsError("prod4 evidence timestamp is missing")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise JobsError("prod4 evidence timestamp is invalid") from exc
    if parsed.tzinfo is None:
        raise JobsError("prod4 evidence timestamp has no timezone")
    return parsed.astimezone(UTC)


def _stamp() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _run_id(plan: dict[str, Any]) -> str:
    return str(uuid5(NAMESPACE_URL, f"fleet-direct-rayjob:{RUN_NAME}:{digest(plan)}"))


def _source(preview: dict[str, Any]) -> dict[str, Any]:
    if preview.get("warnings") or set(preview) != {"name", "warnings", "manifest_yaml"}:
        raise JobsError("prod4 Jobs preview reported warnings or changed shape")
    try:
        value = yaml.safe_load(preview["manifest_yaml"])
    except (KeyError, TypeError, yaml.YAMLError) as exc:
        raise JobsError("prod4 Jobs preview is malformed") from exc
    if not isinstance(value, dict):
        raise JobsError("prod4 Jobs preview is not an object")
    return value


def _replace_env(container: dict[str, Any], name: str, value: str) -> None:
    entries = container.get("env", [])
    matches = [entry for entry in entries if entry.get("name") == name]
    if len(matches) != 1 or set(matches[0]) != {"name", "value"}:
        raise JobsError("prod4 Jobs preview environment changed")
    matches[0]["value"] = value


def manifest(
    plan: dict[str, Any], request: dict[str, Any], preview: dict[str, Any]
) -> dict[str, Any]:
    """Project the accepted Jobs preview into one exact root RayJob."""
    if plan.get("run_name") != RUN_NAME or skyrl_training.job_request(plan) != request:
        raise JobsError("prod4 plan/request identity changed")
    source = _source(preview)
    expected_placeholder = RUN_NAME + "-00000000"
    metadata = source.get("metadata", {})
    labels, annotations = metadata.get("labels", {}), metadata.get("annotations", {})
    if (
        preview.get("name") != expected_placeholder
        or source.get("apiVersion") != "ray.io/v1"
        or source.get("kind") != "RayJob"
        or metadata.get("name") != expected_placeholder
        or metadata.get("namespace") != NAMESPACE
        or labels.get("fleet.ai/run-id") != "00000000-0000-0000-0000-000000000000"
        or labels.get("fleet.ai/run-name") != RUN_NAME
        or labels.get("kueue.x-k8s.io/queue-name") != "training-lq"
        or labels.get("kueue.x-k8s.io/priority-class") != "q1"
        or annotations.get("fleet.ai/run-id") != "00000000-0000-0000-0000-000000000000"
        or annotations.get("fleet.ai/run-dir") != plan["output_root"]
        or annotations.get("fleet.ai/job-image") != request["image"]
        or FAILURE_ALERT_ANNOTATION in annotations
    ):
        raise JobsError("prod4 Jobs preview identity or admission changed")
    result = copy.deepcopy(source)
    direct_id = _run_id(plan)
    result["metadata"]["name"] = RUN_NAME
    result["metadata"]["labels"]["fleet.ai/run-id"] = direct_id
    result["metadata"]["annotations"]["fleet.ai/run-id"] = direct_id
    result["metadata"]["annotations"][FAILURE_ALERT_ANNOTATION] = FAILURE_ALERT_OFF
    spec = result["spec"]
    if (
        spec.get("entrypoint") != request["command"]
        or spec.get("suspend") is not True
        or spec.get("shutdownAfterJobFinishes") is not True
        or spec.get("submissionMode") != "HTTPMode"
    ):
        raise JobsError("prod4 Jobs preview execution changed")
    spec["backoffLimit"] = 0
    spec["activeDeadlineSeconds"] = MAXIMUM_SECONDS
    cluster = spec.get("rayClusterSpec", {})
    if cluster.get("workerGroupSpecs") not in (None, []):
        raise JobsError("prod4 must remain one physical GPU node")
    head = cluster.get("headGroupSpec", {}).get("template", {}).get("spec", {})
    containers = head.get("containers", [])
    if len(containers) != 1:
        raise JobsError("prod4 preview must contain one head container")
    container = containers[0]
    _replace_env(container, "FLEET_RUN_ID", direct_id)
    _replace_env(container, "FLEET_RUN_NAME", RUN_NAME)
    secret_names = [row.get("secretRef", {}).get("name") for row in container.get("envFrom", [])]
    generated_secret = expected_placeholder + "-fleet-key"
    if secret_names != ["fleet-api", "wandb-api", generated_secret]:
        raise JobsError("prod4 Jobs preview Secret bindings changed")
    container["envFrom"] = [
        row
        for row in container["envFrom"]
        if row.get("secretRef", {}).get("name") != generated_secret
    ]
    container["securityContext"] = {
        "allowPrivilegeEscalation": False,
        "privileged": False,
        "runAsGroup": RUNTIME_GID,
        "runAsNonRoot": True,
        "runAsUser": RUNTIME_UID,
    }
    init = head.get("initContainers", [])
    sfs = [item for item in init if item.get("name") == "sfs-init"]
    if len(sfs) != 1 or sfs[0].get("command") != [
        "sh",
        "-c",
        f"mkdir -p {plan['output_root']} && chown 1000:100 {plan['output_root']}",
    ]:
        raise JobsError("prod4 output initialization changed")
    sfs[0]["command"] = [
        "sh",
        "-ec",
        f"mkdir {plan['output_root']}; chown 1000:100 {plan['output_root']}",
    ]
    # Validate the final root RayJob through the same scientific request gate.
    skyrl_training.validate_gpu_runtime_preview(
        request,
        {"manifest_yaml": yaml.safe_dump(result, sort_keys=True), "warnings": []},
    )
    return result


def packet(
    plan: dict[str, Any], request: dict[str, Any], preview: dict[str, Any]
) -> dict[str, Any]:
    value = manifest(plan, request, preview)
    return _seal(
        {
            "schema": PACKET_SCHEMA,
            "plan_sha256": digest(plan),
            "request_sha256": digest(request),
            "jobs_preview_sha256": digest(preview),
            "manifest_sha256": digest(value),
            "direct_run_id": _run_id(plan),
            "name": RUN_NAME,
            "namespace": NAMESPACE,
            "submitted": False,
        }
    )


def _strip_server_defaults(expected: dict[str, Any], rendered: dict[str, Any]) -> dict[str, Any]:
    actual = copy.deepcopy(rendered)
    metadata = actual.get("metadata", {})
    try:
        UUID(metadata.pop("uid"))
        _timestamp(metadata.pop("creationTimestamp"))
    except (KeyError, TypeError, ValueError) as exc:
        raise JobsError("prod4 server dry-run lacks identity defaults") from exc
    if metadata.pop("generation", None) != 1 or "resourceVersion" in metadata:
        raise JobsError("prod4 server dry-run metadata defaults changed")
    if actual.pop("status", None) not in (None, {}):
        raise JobsError("prod4 server dry-run added status")
    spec = actual.get("spec", {})
    if spec.pop("ttlSecondsAfterFinished", None) not in (None, 0):
        raise JobsError("prod4 RayJob TTL default changed")
    cluster = spec.get("rayClusterSpec", {})
    if cluster.pop("headServiceAnnotations", None) not in (None, {}):
        raise JobsError("prod4 head-service defaults changed")
    head = cluster.get("headGroupSpec", {})
    if head.pop("numOfHosts", None) not in (None, 1):
        raise JobsError("prod4 head host default changed")
    if head.pop("scaleStrategy", None) not in (None, {}):
        raise JobsError("prod4 head scale default changed")
    return actual


def validate_preview(
    plan: dict[str, Any],
    request: dict[str, Any],
    source_preview: dict[str, Any],
    expected: dict[str, Any],
    rendered: dict[str, Any],
    *,
    context: str,
) -> dict[str, Any]:
    if expected != manifest(plan, request, source_preview):
        raise JobsError("prod4 direct packet changed")
    if (
        context not in {DEV_CONTEXT, PROD_CONTEXT}
        or _strip_server_defaults(expected, rendered) != expected
    ):
        raise JobsError("prod4 server dry-run changed the direct RayJob")
    return _seal(
        {
            "schema": PREVIEW_SCHEMA,
            "status": "passed",
            "context": context,
            "plan_sha256": digest(plan),
            "request_sha256": digest(request),
            "manifest_sha256": digest(expected),
            "server_render_sha256": digest(rendered),
            "name": RUN_NAME,
            "nodes": 1,
            "gpus": 8,
            "priority": "c1",
            "queue_priority": "q1",
            "failure_alerts": "off",
            "submitted": False,
        }
    )


def build_stage_archive(source: Path, archive: Path) -> dict[str, Any]:
    """Write one deterministic private archive without exposing partial bytes."""
    expected = {"manifest.json", "split.json", "task-set.json", "train.jsonl", "dev.jsonl"}
    found = {path.name: path for path in source.iterdir() if path.is_file()}
    if set(found) != expected or archive.exists() or archive.is_symlink():
        raise JobsError("prod4 private stage archive inputs changed")
    archive.parent.mkdir(parents=True, exist_ok=True)
    temp = archive.with_name("." + archive.name + ".partial")
    if temp.exists() or temp.is_symlink():
        raise JobsError("prod4 private stage archive temporary path exists")
    try:
        with (
            temp.open("xb") as raw,
            gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed,
            tarfile.open(fileobj=compressed, mode="w") as tar,
        ):
            for name, path in sorted(found.items()):
                info = tarfile.TarInfo(name)
                info.size = path.stat().st_size
                info.mode = 0o600
                info.uid, info.gid, info.mtime = RUNTIME_UID, RUNTIME_GID, 0
                with path.open("rb") as stream:
                    tar.addfile(info, stream)
            raw.flush()
            os.fsync(raw.fileno())
        os.rename(temp, archive)
        directory = os.open(archive.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        if temp.exists() and temp.parent == archive.parent and temp.name.startswith("."):
            temp.unlink()
        raise
    return {
        "bytes": archive.stat().st_size,
        "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
    }


def stage_plan(plan: dict[str, Any], source: Path, archive: Path) -> dict[str, Any]:
    expected = {"manifest.json", "split.json", "task-set.json", "train.jsonl", "dev.jsonl"}
    found = {path.name: path for path in source.iterdir() if path.is_file()}
    if set(found) != expected:
        raise JobsError("prod4 private stage inventory changed")
    files = []
    for name, path in sorted(found.items()):
        files.append(
            {
                "path": name,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    metadata = json.loads(found["manifest.json"].read_bytes())
    if metadata != plan["data"]:
        raise JobsError("prod4 private manifest differs from the plan")
    by_name = {item["path"]: item for item in files}
    if any(
        by_name[plan["data"]["files"][split]["path"]]["sha256"]
        != plan["data"]["files"][split]["sha256"].removeprefix("sha256:")
        for split in ("train", "dev")
    ):
        raise JobsError("prod4 private train/dev payload differs from the plan")
    for name, expected_sha in (
        ("task-set.json", plan["data"]["selection_sha256"]),
        ("split.json", plan["data"]["split_sha256"]),
    ):
        value = json.loads(found[name].read_bytes())
        body = {key: item for key, item in value.items() if key != "sha256"}
        if value.get("sha256") != expected_sha or value["sha256"] != "sha256:" + digest(body):
            raise JobsError("prod4 private task-set/split evidence differs from the plan")
    archive_binding = build_stage_archive(source, archive)
    value = {
        "schema": STAGE_SCHEMA,
        "name": STAGE_NAME,
        "destination": plan["arguments"]["data_manifest"].removesuffix("/manifest.json"),
        "files": files,
        "archive": archive_binding,
        "plan_sha256": digest(plan),
        "image": plan["execution"]["image"],
        "scientific_work": {
            "task_rows_read": 0,
            "rollout_episodes": 0,
            "optimizer_steps": 0,
            "checkpoints": 0,
            "gpus": 0,
        },
    }
    return _seal(value)


def _safe_member(member: tarfile.TarInfo, expected: set[str]) -> None:
    path = PurePosixPath(member.name)
    if (
        member.name not in expected
        or path.is_absolute()
        or ".." in path.parts
        or not member.isfile()
        or member.issym()
        or member.islnk()
    ):
        raise ValueError("prod4 upload archive is unsafe")


def stage_runtime(value: dict[str, Any]) -> dict[str, Any]:
    plan = _validate_seal(value, STAGE_SCHEMA)
    if (os.geteuid(), os.getegid()) != (RUNTIME_UID, RUNTIME_GID):
        raise ValueError("prod4 stage runtime user changed")
    deadline = time.monotonic() + 600
    while not UPLOAD.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("prod4 upload did not arrive")
        time.sleep(1)
    upload_info = UPLOAD.stat()
    upload_sha256 = hashlib.sha256(UPLOAD.read_bytes()).hexdigest()
    if (
        upload_info.st_size != plan["archive"]["bytes"]
        or upload_sha256 != plan["archive"]["sha256"]
    ):
        raise ValueError("prod4 uploaded archive digest changed")
    destination = Path(plan["destination"])
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("prod4 data destination already exists")
    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    temp = parent / ("." + destination.name + ".tmp-" + digest(plan)[:12])
    if temp.exists() or temp.is_symlink():
        raise FileExistsError("prod4 stage temporary path already exists")
    temp.mkdir(mode=0o700)
    expected = {item["path"] for item in plan["files"]}
    try:
        with tarfile.open(UPLOAD, "r:gz") as archive:
            members = archive.getmembers()
            if {member.name for member in members} != expected:
                raise ValueError("prod4 upload inventory changed")
            for member in members:
                _safe_member(member, expected)
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError("prod4 upload member cannot be read")
                path = temp / member.name
                with path.open("xb") as output:
                    shutil.copyfileobj(stream, output, 1024 * 1024)
                    output.flush()
                    os.fsync(output.fileno())
                path.chmod(0o600)
        observed = []
        by_name = {item["path"]: item for item in plan["files"]}
        for path in sorted(temp.iterdir()):
            item = by_name[path.name]
            info = path.stat()
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if info.st_size != item["bytes"] or actual != item["sha256"]:
                raise ValueError("prod4 staged file digest changed")
            observed.append({"path": path.name, "bytes": info.st_size, "sha256": actual})
        os.rename(temp, destination)
        parent_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        return _seal(
            {
                "schema": STAGE_RECEIPT_SCHEMA,
                "status": "published",
                "stage_plan_sha256": digest(plan),
                "stage_plan": plan,
                "training_plan_sha256": plan["plan_sha256"],
                "destination": str(destination),
                "files": observed,
                "archive": plan["archive"],
                "runtime_user": {"uid": os.geteuid(), "gid": os.getegid()},
                **plan["scientific_work"],
            }
        )
    except BaseException:
        if temp.exists() and temp.parent == parent and temp.name.startswith("."):
            shutil.rmtree(temp)
        raise


def _runtime_files(plan: dict[str, Any], *, include_self: bool) -> dict[str, str]:
    files = skyrl_training._runtime()
    files.update(
        {
            path + "/__init__.py": ""
            for path in ("training", "evals", "evals/fleet", "cyber_post_train")
        }
    )
    if include_self:
        files["training/skyrl_reward_rayjob.py"] = Path(__file__).read_text()
    files["plan.json"] = json.dumps(plan, sort_keys=True, separators=(",", ":"))
    return files


def _cpu_job(
    name: str, image: str, command: str, env: dict[str, str], *, wandb: bool
) -> dict[str, Any]:
    container: dict[str, Any] = {
        "name": "preflight",
        "image": image,
        "command": ["/bin/sh", "-lc", "exec " + command],
        "env": [{"name": key, "value": value} for key, value in sorted(env.items())],
        "securityContext": {
            "allowPrivilegeEscalation": False,
            "privileged": False,
            "runAsGroup": RUNTIME_GID,
            "runAsNonRoot": True,
            "runAsUser": RUNTIME_UID,
        },
        "resources": {
            "requests": {"cpu": "4", "memory": "32Gi"},
            "limits": {"cpu": "8", "memory": "48Gi"},
        },
        "terminationMessagePath": STAGE_RECEIPT,
        "terminationMessagePolicy": "File",
        "volumeMounts": [{"name": "sfs", "mountPath": "/mnt/sfs"}],
    }
    if wandb:
        container["envFrom"] = [{"secretRef": {"name": "wandb-api"}}]
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": name,
            "namespace": NAMESPACE,
            "annotations": {FAILURE_ALERT_ANNOTATION: FAILURE_ALERT_OFF},
        },
        "spec": {
            "activeDeadlineSeconds": 1800,
            "backoffLimit": 0,
            "template": {
                "metadata": {},
                "spec": {
                    "automountServiceAccountToken": False,
                    "containers": [container],
                    "priorityClassName": "c1",
                    "restartPolicy": "Never",
                    "securityContext": {"supplementalGroups": [2000]},
                    "volumes": [{"name": "sfs", "persistentVolumeClaim": {"claimName": PVC}}],
                },
            },
        },
    }


def stage_job_manifest(value: dict[str, Any]) -> dict[str, Any]:
    plan = _validate_seal(value, STAGE_SCHEMA)
    files = skyrl_training._runtime()
    files.update(
        {
            path + "/__init__.py": ""
            for path in ("training", "evals", "evals/fleet", "cyber_post_train")
        }
    )
    files["training/skyrl_reward_rayjob.py"] = Path(__file__).read_text()
    files["stage.json"] = json.dumps(plan, sort_keys=True, separators=(",", ":"))
    bundled = bundled_request(
        {
            "name": STAGE_NAME,
            "run_dir": "/mnt/sfs/jobs/" + STAGE_NAME,
            "image": plan["image"],
            "workers": 1,
            "gpus_per_worker": 1,
            "resources": {
                "cpu_request": "4",
                "cpu_limit": "8",
                "memory_request": "32Gi",
                "memory_limit": "48Gi",
            },
            "priority_class": "c1",
            "requeueIfPreempted": False,
            "failureAlerts": False,
            "secrets": [],
            "env": {"PYTHONUNBUFFERED": "1"},
        },
        files,
        "training.skyrl_reward_rayjob",
        ["--stage", "stage.json", "--receipt", STAGE_RECEIPT],
    )
    return _cpu_job(
        STAGE_NAME,
        plan["image"],
        bundled["command"],
        {**bundled["env"], "RUN_DIR": "/tmp"},
        wandb=False,
    )


def preflight_job_manifest(plan: dict[str, Any]) -> dict[str, Any]:
    request = skyrl_training.job_request(plan)
    files = _runtime_files(plan, include_self=True)
    bundled = bundled_request(
        {
            "name": PREFLIGHT_NAME,
            "run_dir": "/mnt/sfs/jobs/" + PREFLIGHT_NAME,
            "image": request["image"],
            "workers": 1,
            "gpus_per_worker": 1,
            "resources": {
                "cpu_request": "4",
                "cpu_limit": "8",
                "memory_request": "32Gi",
                "memory_limit": "48Gi",
            },
            "priority_class": "c1",
            "requeueIfPreempted": False,
            "failureAlerts": False,
            "secrets": ["wandb-api"],
            "env": {"PYTHONUNBUFFERED": "1"},
        },
        files,
        "training.skyrl_reward_rayjob",
        ["--preflight", "plan.json", "--receipt", STAGE_RECEIPT],
    )
    return _cpu_job(
        PREFLIGHT_NAME,
        request["image"],
        bundled["command"],
        {**bundled["env"], "RUN_DIR": "/tmp"},
        wandb=True,
    )


class PreflightGateError(Exception):
    def __init__(self, phase: str, error: BaseException) -> None:
        super().__init__(phase)
        self.phase = phase
        self.error_class = type(error).__name__
        self.message_sha256 = hashlib.sha256(str(error).encode()).hexdigest()


def preflight_runtime(plan: dict[str, Any]) -> dict[str, Any]:
    try:
        proof = skyrl_training.preflight(plan)
    except BaseException as exc:
        raise PreflightGateError("scientific_preflight", exc) from None
    if proof.get("status") != "passed" or proof.get("gpus") != 0:
        raise PreflightGateError(
            "scientific_preflight_contract",
            ValueError("prod4 scientific CPU preflight failed"),
        )
    # W&B run creation is deliberately create-once at the training boundary:
    # ScalarTracking calls wandb.init(..., resume="never") with the immutable
    # plan-bound ID.  A pre-submit read of that ID is not an authoritative
    # existence test: W&B 0.21.1 wraps missing-run, subscription and transient
    # service failures in the same CommError surface.  Requiring that lookup
    # turned an optional observability read into a false scientific blocker.
    # Prove the credential injection and exact local binding here; the actual
    # W&B service accepts or rejects the create-once ID at runtime before any
    # rollout/optimizer work.
    if not os.environ.get("WANDB_API_KEY"):
        raise PreflightGateError(
            "wandb_credential", RuntimeError("prod4 W&B read credential is absent")
        )
    arguments = plan["arguments"]
    wandb_binding = {
        "entity": arguments.get("wandb_entity"),
        "project": arguments.get("wandb_project"),
        "run_id": arguments.get("wandb_run_id"),
        "resume": "never",
    }
    if wandb_binding != {
        "entity": "thefleet",
        "project": "cyber-post-train",
        "run_id": RUN_NAME,
        "resume": "never",
    }:
        raise PreflightGateError(
            "wandb_binding", ValueError("prod4 W&B create-once binding changed")
        )
    return _seal(
        {
            "schema": PREFLIGHT_RECEIPT_SCHEMA,
            "status": "passed",
            "plan_sha256": digest(plan),
            "request_sha256": digest(skyrl_training.job_request(plan)),
            "gpus": 0,
            "runtime_user": proof["runtime_user"],
            "counts": proof["counts"],
            "planned_steps": proof["planned_steps"],
            "native_parser_checked": proof["native_parser_checked"],
            "output_absent": True,
            "wandb_create_once": wandb_binding,
            "wandb_remote_lookup": "deferred_to_runtime_start",
            "checked_at": _stamp(),
        }
    )


def _write_receipt(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")


def _kubectl(
    runner: Callable[..., subprocess.CompletedProcess[str]], context: str, *arguments: str
) -> subprocess.CompletedProcess[str]:
    return runner(
        ["kubectl", "--context", context, "--namespace", NAMESPACE, *arguments],
        capture_output=True,
        text=True,
        timeout=60,
    )


def server_dry_run(
    value: dict[str, Any],
    *,
    context: str,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    result = runner(
        [
            "kubectl",
            "--context",
            context,
            "--namespace",
            NAMESPACE,
            "create",
            "--dry-run=server",
            "-f",
            "-",
            "-o",
            "json",
        ],
        input=json.dumps(value),
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode:
        raise JobsError("prod4 Kubernetes server dry-run failed")
    try:
        rendered = json.loads(result.stdout)
    except ValueError as exc:
        raise JobsError("prod4 Kubernetes server dry-run returned invalid JSON") from exc
    return rendered


def validate_cpu_preview(
    expected: dict[str, Any], rendered: dict[str, Any], *, context: str, purpose: str
) -> dict[str, Any]:
    """Accept only normal deterministic Kubernetes defaults on a zero-GPU Job."""
    if context not in {DEV_CONTEXT, PROD_CONTEXT} or purpose not in {"data_stage", "preflight"}:
        raise JobsError("prod4 CPU preview binding is invalid")
    actual = copy.deepcopy(rendered)
    status = actual.pop("status", None)
    metadata = actual.get("metadata", {})
    timestamp = metadata.pop("creationTimestamp", None)
    generation = metadata.pop("generation", None)
    uid = metadata.pop("uid", None)
    labels = metadata.pop("labels", None)
    name = expected["metadata"]["name"]
    generated_labels = {
        "batch.kubernetes.io/controller-uid": uid,
        "batch.kubernetes.io/job-name": name,
        "controller-uid": uid,
        "job-name": name,
    }
    spec = actual.get("spec", {})
    defaults = {
        key: spec.pop(key, None)
        for key in (
            "completionMode",
            "completions",
            "manualSelector",
            "parallelism",
            "podReplacementPolicy",
            "selector",
            "suspend",
        )
    }
    template = spec.get("template", {})
    template_labels = template.get("metadata", {}).pop("labels", None)
    pod = template.get("spec", {})
    pod_defaults: dict[str, Any] = {}
    for key in ("dnsPolicy", "schedulerName", "terminationGracePeriodSeconds"):
        if key not in expected["spec"]["template"]["spec"]:
            pod_defaults[key] = pod.pop(key, None)
    containers = pod.get("containers", [])
    pull = containers[0].pop("imagePullPolicy", None) if len(containers) == 1 else None
    try:
        UUID(uid)
    except (TypeError, ValueError) as exc:
        raise JobsError("prod4 CPU server dry-run lacks a UID") from exc
    if (
        status != {}
        or not isinstance(timestamp, str)
        or generation != 1
        or labels != generated_labels
        or template_labels != generated_labels
        or defaults
        != {
            "completionMode": "NonIndexed",
            "completions": 1,
            "manualSelector": False,
            "parallelism": 1,
            "podReplacementPolicy": "TerminatingOrFailed",
            "selector": {"matchLabels": {"batch.kubernetes.io/controller-uid": uid}},
            "suspend": False,
        }
        or pod_defaults
        != {
            "dnsPolicy": "ClusterFirst",
            "schedulerName": "default-scheduler",
            "terminationGracePeriodSeconds": 30,
        }
        or pull != "IfNotPresent"
        or actual != expected
        or "nvidia.com/gpu" in json.dumps(expected, sort_keys=True)
        or expected["metadata"]["annotations"].get(FAILURE_ALERT_ANNOTATION) != FAILURE_ALERT_OFF
        or expected["spec"]["template"]["spec"].get("priorityClassName") != "c1"
    ):
        raise JobsError("prod4 CPU server dry-run changed the exact Job")
    return _seal(
        {
            "schema": CPU_PREVIEW_SCHEMA,
            "status": "passed",
            "purpose": purpose,
            "context": context,
            "name": name,
            "manifest_sha256": digest(expected),
            "server_render_sha256": digest(rendered),
            "gpus": 0,
            "priority": "c1",
            "failure_alerts": "off",
            "submitted": False,
        }
    )


def duplicate_checks(
    plan: dict[str, Any],
    *,
    token: str,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    jobs_factory: Callable[..., Jobs] = Jobs,
) -> dict[str, int]:
    name, output = plan["run_name"], plan["output_root"]
    kube = 0
    for context in (DEV_CONTEXT, PROD_CONTEXT):
        for resource in ("rayjob", "raycluster", "job", "workload", "pod"):
            result = _kubectl(runner, context, "get", resource, "--output=json")
            if result.returncode:
                raise JobsError("prod4 Kubernetes duplicate inventory failed")
            kube += 1
            try:
                items = json.loads(result.stdout).get("items", [])
            except (AttributeError, ValueError) as exc:
                raise JobsError("prod4 Kubernetes duplicate inventory is invalid") from exc
            for item in items:
                metadata = item.get("metadata", {})
                values = [
                    metadata.get("name", ""),
                    *metadata.get("labels", {}).values(),
                    *metadata.get("annotations", {}).values(),
                ]
                serialized = json.dumps(item.get("spec", {}), sort_keys=True)
                if (
                    any(value == name or str(value).startswith(name + "-") for value in values)
                    or output in values
                    or output in serialized
                ):
                    raise JobsError("prod4 Kubernetes identity/output already exists")
    api_rows = 0
    for target in ("dev", "prod"):
        with jobs_factory(token, base_url=API_URLS[target]) as client:
            rows = client.all_runs()
        api_rows += len(rows)
        if any(
            row.get("run_dir") == output
            or row.get("name") == name
            or str(row.get("name", "")).startswith(name + "-")
            for row in rows
        ):
            raise JobsError("prod4 Jobs API history already owns this identity/output")
    return {"kubernetes_inventories_checked": kube, "jobs_api_rows_checked": api_rows}


def authorize(
    plan: dict[str, Any],
    request: dict[str, Any],
    source_preview: dict[str, Any],
    expected: dict[str, Any],
    *,
    dev_preview: dict[str, Any],
    prod_preview: dict[str, Any],
    cpu_previews: list[dict[str, Any]],
    stage_receipt: dict[str, Any],
    preflight_receipt: dict[str, Any],
    observer: dict[str, Any],
) -> dict[str, Any]:
    direct_sha = digest(manifest(plan, request, source_preview))
    for proof, context in ((dev_preview, DEV_CONTEXT), (prod_preview, PROD_CONTEXT)):
        _validate_seal(proof, PREVIEW_SCHEMA)
        if (
            proof.get("status") != "passed"
            or proof.get("context") != context
            or proof.get("manifest_sha256") != direct_sha
            or proof.get("failure_alerts") != "off"
            or proof.get("submitted") is not False
        ):
            raise JobsError("prod4 direct server preview was not accepted")
    _validate_seal(stage_receipt, STAGE_RECEIPT_SCHEMA)
    staged = _validate_seal(stage_receipt.get("stage_plan"), STAGE_SCHEMA)
    staged_files = staged.get("files")
    file_by_name = (
        {
            item.get("path"): item
            for item in staged_files
            if isinstance(item, dict) and isinstance(item.get("path"), str)
        }
        if isinstance(staged_files, list)
        else {}
    )
    archive = staged.get("archive", {})
    if (
        stage_receipt.get("status") != "published"
        or stage_receipt.get("stage_plan_sha256") != digest(staged)
        or staged.get("plan_sha256") != digest(plan)
        or staged.get("name") != STAGE_NAME
        or staged.get("image") != plan["execution"]["image"]
        or staged.get("destination")
        != plan["arguments"]["data_manifest"].removesuffix("/manifest.json")
        or set(file_by_name)
        != {"manifest.json", "split.json", "task-set.json", "train.jsonl", "dev.jsonl"}
        or any(
            file_by_name[plan["data"]["files"][split]["path"]].get("sha256")
            != plan["data"]["files"][split]["sha256"].removeprefix("sha256:")
            for split in ("train", "dev")
        )
        or not isinstance(archive.get("bytes"), int)
        or archive["bytes"] < 1
        or re.fullmatch(r"[a-f0-9]{64}", str(archive.get("sha256", ""))) is None
        or staged.get("scientific_work")
        != {
            "task_rows_read": 0,
            "rollout_episodes": 0,
            "optimizer_steps": 0,
            "checkpoints": 0,
            "gpus": 0,
        }
        or stage_receipt.get("training_plan_sha256") != digest(plan)
        or stage_receipt.get("destination")
        != plan["arguments"]["data_manifest"].removesuffix("/manifest.json")
        or stage_receipt.get("gpus") != 0
        or stage_receipt.get("files") != staged_files
        or stage_receipt.get("archive") != archive
        or stage_receipt.get("runtime_user") != {"uid": 1000, "gid": 100}
        or any(
            stage_receipt.get(key) != 0
            for key in ("task_rows_read", "rollout_episodes", "optimizer_steps", "checkpoints")
        )
    ):
        raise JobsError("prod4 data stage was not accepted")
    expected_cpu = {
        (context, purpose, digest(job))
        for context in (DEV_CONTEXT, PROD_CONTEXT)
        for purpose, job in (
            ("data_stage", stage_job_manifest(stage_receipt["stage_plan"])),
            ("preflight", preflight_job_manifest(plan)),
        )
    }
    observed_cpu = set()
    for proof in cpu_previews:
        _validate_seal(proof, CPU_PREVIEW_SCHEMA)
        if (
            proof.get("status") != "passed"
            or proof.get("gpus") != 0
            or proof.get("priority") != "c1"
            or proof.get("failure_alerts") != "off"
            or proof.get("submitted") is not False
        ):
            raise JobsError("prod4 CPU server preview was not accepted")
        observed_cpu.add((proof.get("context"), proof.get("purpose"), proof.get("manifest_sha256")))
    if observed_cpu != expected_cpu or len(cpu_previews) != 4:
        raise JobsError("prod4 CPU server preview set is incomplete")
    _validate_seal(preflight_receipt, PREFLIGHT_RECEIPT_SCHEMA)
    checked = _timestamp(preflight_receipt.get("checked_at"))
    if (
        preflight_receipt.get("status") != "passed"
        or preflight_receipt.get("plan_sha256") != digest(plan)
        or preflight_receipt.get("request_sha256") != digest(request)
        or preflight_receipt.get("gpus") != 0
        or preflight_receipt.get("runtime_user") != {"uid": 1000, "gid": 100}
        or preflight_receipt.get("output_absent") is not True
        or preflight_receipt.get("wandb_create_once")
        != {
            "entity": "thefleet",
            "project": "cyber-post-train",
            "run_id": RUN_NAME,
            "resume": "never",
        }
        or preflight_receipt.get("wandb_remote_lookup") != "deferred_to_runtime_start"
        or preflight_receipt.get("counts") != {"train": 1, "dev": 1}
        or preflight_receipt.get("planned_steps") != 1
        or preflight_receipt.get("native_parser_checked") is not True
        or (datetime.now(UTC) - checked).total_seconds() > 300
    ):
        raise JobsError("prod4 CPU preflight is stale or incomplete")
    _validate_seal(observer, "cyber_direct_cleanup_observer_armed_v1")
    pid = observer.get("observer_pid")
    if (
        observer.get("status") != "armed"
        or observer.get("context") != PROD_CONTEXT
        or observer.get("namespace") != NAMESPACE
        or observer.get("kind") != "rayjob"
        or observer.get("name") != RUN_NAME
        or observer.get("maximum_seconds") != MAXIMUM_SECONDS
        or observer.get("expected_gpus") != 8
        or observer.get("plan_sha256") != "sha256:" + digest(plan)
        or observer.get("manifest_sha256") != "sha256:" + direct_sha
        or type(pid) is not int
        or pid < 1
    ):
        raise JobsError("prod4 cleanup observer is not exactly armed")
    _timestamp(observer.get("armed_at"))
    return _seal(
        {
            "schema": AUTHORIZATION_SCHEMA,
            "status": "authorized_for_one_create",
            "plan_sha256": digest(plan),
            "request_sha256": digest(request),
            "manifest_sha256": direct_sha,
            "dev_preview": dev_preview,
            "prod_preview": prod_preview,
            "cpu_previews": cpu_previews,
            "stage_receipt": stage_receipt,
            "preflight_receipt": preflight_receipt,
            "observer": observer,
        }
    )


def write_once_fsynced(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        stream.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def create_once(
    directory: Path,
    plan: dict[str, Any],
    request: dict[str, Any],
    source_preview: dict[str, Any],
    expected: dict[str, Any],
    authorization: dict[str, Any],
    *,
    token: str,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    jobs_factory: Callable[..., Jobs] = Jobs,
) -> dict[str, Any]:
    if expected != manifest(plan, request, source_preview):
        raise JobsError("prod4 direct manifest changed")
    _validate_seal(authorization, AUTHORIZATION_SCHEMA)
    if authorization != authorize(
        plan,
        request,
        source_preview,
        expected,
        dev_preview=authorization["dev_preview"],
        prod_preview=authorization["prod_preview"],
        cpu_previews=authorization["cpu_previews"],
        stage_receipt=authorization["stage_receipt"],
        preflight_receipt=authorization["preflight_receipt"],
        observer=authorization["observer"],
    ):
        raise JobsError("prod4 direct authorization changed")
    try:
        os.kill(authorization["observer"]["observer_pid"], 0)
    except (KeyError, OSError, TypeError) as exc:
        raise JobsError("prod4 cleanup observer is not running") from exc
    duplicate = duplicate_checks(plan, token=token, runner=runner, jobs_factory=jobs_factory)
    final_render = server_dry_run(expected, context=PROD_CONTEXT, runner=runner)
    validate_preview(plan, request, source_preview, expected, final_render, context=PROD_CONTEXT)
    # Duplicate reconciliation and API inventory can be slow.  Recheck the
    # observer immediately before recording create intent so an observer that
    # exited during those reads can never authorize an unwatched workload.
    try:
        os.kill(authorization["observer"]["observer_pid"], 0)
    except (KeyError, OSError, TypeError) as exc:
        raise JobsError("prod4 cleanup observer exited before create") from exc
    journal = directory / "DIRECT_RAYJOB_CREATE.jsonl"
    if journal.exists() or journal.is_symlink():
        raise JobsError("prod4 create intent exists; reconcile, never retry")
    intent = {
        "state": "CREATE_INTENT_DO_NOT_RETRY",
        "recorded_at": _stamp(),
        "plan_sha256": digest(plan),
        "request_sha256": digest(request),
        "manifest_sha256": digest(expected),
        "authorization_sha256": authorization["sha256"],
        "duplicate_checks": duplicate,
    }
    write_once_fsynced(journal, intent)
    # The sole live create.  Do not add retries around this call.
    result = runner(
        [
            "kubectl",
            "--context",
            PROD_CONTEXT,
            "--namespace",
            NAMESPACE,
            "create",
            "-f",
            "-",
            "-o",
            "json",
        ],
        input=json.dumps(expected),
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode:
        raise JobsError("prod4 create returned failure; reconcile intent, never retry")
    try:
        created = json.loads(result.stdout)
        uid = str(UUID(created["metadata"]["uid"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise JobsError("prod4 create response is ambiguous; reconcile, never retry") from exc
    if created.get("metadata", {}).get("name") != RUN_NAME:
        raise JobsError("prod4 create returned another identity; reconcile, never retry")
    proof = _seal(
        {
            "schema": CREATED_SCHEMA,
            "status": "created_once",
            "plan_sha256": digest(plan),
            "manifest_sha256": digest(expected),
            "rayjob_name": RUN_NAME,
            "rayjob_uid": uid,
            "created_at": created["metadata"].get("creationTimestamp"),
        }
    )
    with journal.open("a") as stream:
        stream.write(json.dumps(proof, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return proof


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    try:
        if (args.stage is None) == (args.preflight is None) or args.receipt != Path(STAGE_RECEIPT):
            raise ValueError("prod4 CPU mode binding changed")
        if args.stage is not None:
            result = stage_runtime(json.loads(args.stage.read_bytes()))
        else:
            plan = json.loads(args.preflight.read_bytes())
            try:
                result = preflight_runtime(plan)
            except PreflightGateError as exc:
                result = _seal(
                    {
                        "schema": PREFLIGHT_REJECTION_SCHEMA,
                        "status": "rejected",
                        "phase": exc.phase,
                        "error_class": exc.error_class,
                        "message_sha256": exc.message_sha256,
                        "plan_sha256": digest(plan),
                        "request_sha256": digest(skyrl_training.job_request(plan)),
                        "gpus": 0,
                        "checked_at": _stamp(),
                    }
                )
        _write_receipt(args.receipt, result)
        print(json.dumps({"status": result["status"], "sha256": result["sha256"]}))
    except BaseException as exc:
        print(json.dumps({"status": "failed", "error_class": type(exc).__name__}))
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
