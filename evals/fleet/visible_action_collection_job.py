"""Create exactly one sealed amd64 CPU Job for visible-action collection.

This is deliberately narrower than a general Kubernetes launcher.  It packages
the exact merged source bytes and collection inputs into one immutable
ConfigMap, renders one fixed CPU Job, proves two identical server previews,
records a durable local intent, and performs one Kubernetes create request.

The collection Job itself owns the create-once SFS root and dedicated database
bound by an exact v2 or v3 visible-action plan.  V3 is the separately versioned
completion-budget successor; v2 inputs and artifacts remain unchanged.
Historical Fleet sessions are outside this operation identity.  A missing or
ambiguous create response is never retried.  Terminal cleanup is separately
bound to the returned Job and ConfigMap UIDs and uses Kubernetes UID
preconditions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Protocol
from urllib.parse import quote

from cyber_post_train.jobs import digest
from evals.fleet import visible_action_collection_v2 as collection_v2
from evals.fleet import visible_action_collection_v3 as collection_v3
from evals.fleet.evaluate import stable_job_preview

PACKET_SCHEMA = "cyber_fleet_visible_action_collection_job_packet_v1"
CREATE_INTENT_SCHEMA = "cyber_fleet_visible_action_collection_job_create_intent_v1"
CLEANUP_INTENT_SCHEMA = "cyber_fleet_visible_action_collection_job_cleanup_intent_v1"
CLEANUP_RECEIPT_SCHEMA = "cyber_fleet_visible_action_collection_job_cleanup_receipt_v1"
STATUS_SCHEMA = "cyber_fleet_visible_action_collection_job_status_v1"
CONTEXT = "nebius-mk8s-fleetai-training-e04zw4ye1k7wczqdw6"
NAMESPACE = "fleet-train-jobs"
QUEUE_NAME = "training-lq"
PRIORITY_CLASS = "c1"
SFS_PVC = "sfs-shared"
PRIVATE_ROOT = "/mnt/sfs/jobs"
FAILURE_ALERT_ANNOTATION = "fleet.ai/failure-alerts"
CREATE_ONCE_ANNOTATION = "cyber-post-train.fleet.ai/create-once"
OPERATION_AUTHORIZATION_ANNOTATION = "cyber-post-train.fleet.ai/operation-authorization-sha256"
COLLECTION_PACKET_ANNOTATION = "cyber-post-train.fleet.ai/collection-packet-sha256"
CONTROLLER_IMAGE = (
    "ghcr.io/astral-sh/uv:python3.12-bookworm@"
    "sha256:9aa60c50016c0485636ab9a830246a6ef3399aa4a8bab3d17ef4a2358fba2ca7"
)
DIND_IMAGE = (
    "docker.io/library/docker@"
    "sha256:f649ef046008ca7f926a2571c32b0ac22e5c59eb61b959617f9acc2a4c638cf5"
)
PROXY_IMAGE = CONTROLLER_IMAGE
AGENT_IMAGE_ID = "sha256:c7d048c98e6b8e52e5b76ab4006a7626b1ccf63a37bfa4b47ecd0fe9028e1f92"
HARNESS_TAR = (
    "/mnt/sfs/jobs/chris-q38-fleet-dev17-harness-build-v1/opencode-1.18.27-linux-amd64.tar"
)
HARNESS_TAR_SHA256 = "sha256:b14bdf558990087504a95e209ff0c0e84039633f6d28b2cdbc483cd845c0c410"
HARNESS_RECEIPT = "/mnt/sfs/jobs/chris-q38-fleet-dev17-harness-build-v1/BUILD.json"
HARNESS_RECEIPT_SHA256 = "sha256:4781514a8431d6d87c671c49120639062f5a8e3afb016a04360752c896181745"
FLEET_SECRET = "chris-cyber-opencode-evals-v2"
FLEET_SECRET_KEY = "FLEET_API_KEY"
POSTGRES_SECRET = "chris-cyber-rollout-postgres-v1"
POSTGRES_SECRET_KEY = "ROLLOUT_DATABASE_URL"
ACTIVE_DEADLINE_SECONDS = 768600
EXPECTED_CELLS = 200
CANARY_CELLS = 4
EXPECTED_CONCURRENCY = 8
CONFIG_MAP_MAX_SERIALIZED_BYTES = 900000
CLUSTER_JOB_REQUIREMENTS = {
    "active_deadline_seconds": ACTIVE_DEADLINE_SECONDS,
    "ambiguous_cluster_create_retry_allowed": False,
    "architecture": "amd64",
    "backoff_limit": 0,
    "cluster_create_attempts": 1,
    "completions": 1,
    "exact_launcher_bindings_required": [
        "operation_authorization_sha256",
        "operation_root_name",
        "dedicated_ledger_id",
        "source_git_commit",
        "source_git_tree",
        "agent_image_digest",
        "proxy_image_digest",
        "controller_image_digest",
        "database_identity",
        "sfs_pvc_identity",
        "kube_context",
        "namespace",
        "queue_name",
    ],
    "exact_uid_foreground_cleanup_required": True,
    "exclusive_local_cluster_create_intent_required": True,
    "identical_normalized_server_preview_digests_required": True,
    "owned_child_absence_and_no_idle_proof_required": True,
    "parallelism": 1,
    "priority_class_name": PRIORITY_CLASS,
    "required_top_level_annotation": {FAILURE_ALERT_ANNOTATION: "off"},
    "restart_policy": "Never",
    "root_kind": "Job",
    "server_preview_count": 2,
    "terminal_exact_name_and_uid_observation_required": True,
    "zero_gpu_requests_and_limits_across_regular_and_init_containers": True,
}
KUBERNETES_NAME = re.compile(r"[a-z0-9](?:[-a-z0-9]{0,61}[a-z0-9])?")
KUBERNETES_UID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
DATABASE_NAME = re.compile(r"[a-z][a-z0-9_]{0,62}")
SHA256 = re.compile(r"sha256:[0-9a-f]{64}")
GIT_ID = re.compile(r"[0-9a-f]{40}")

SOURCE_FILES_V2 = {
    "jobs.py": "cyber_post_train/jobs.py",
    "cluster_entry.py": "evals/fleet/cluster_entry.py",
    "collection_fixed_proxy.py": "evals/fleet/collection_fixed_proxy.py",
    "evaluate.py": "evals/fleet/evaluate.py",
    "exact_pass4_crypto.py": "evals/fleet/exact_pass4_crypto.py",
    "exact_pass4_universe.py": "evals/fleet/exact_pass4_universe.py",
    "fixed_proxy.py": "evals/fleet/fixed_proxy.py",
    "model_artifact.py": "evals/fleet/model_artifact.py",
    "opencode_self_hosted.py": "evals/fleet/opencode_self_hosted.py",
    "rollout_campaign.py": "evals/fleet/rollout_campaign.py",
    "rollout_ledger.py": "evals/fleet/rollout_ledger.py",
    "rollout_postgres.py": "evals/fleet/rollout_postgres.py",
    "rollout_worker.py": "evals/fleet/rollout_worker.py",
    "visible_action_collection.py": "evals/fleet/visible_action_collection.py",
    "visible_action_collection_v2.py": "evals/fleet/visible_action_collection_v2.py",
    "visible_action_collection_job_entry.py": (
        "evals/fleet/visible_action_collection_job_entry.py"
    ),
}
SOURCE_FILES_V3 = {
    **SOURCE_FILES_V2,
    "collection_completion_budget.py": "evals/fleet/collection_completion_budget.py",
    "collection_fixed_proxy_v2.py": "evals/fleet/collection_fixed_proxy_v2.py",
    "visible_action_collection_v3.py": "evals/fleet/visible_action_collection_v3.py",
}

BOOTSTRAP_SCRIPT = """#!/usr/bin/env bash
set -euo pipefail
umask 077

: "${FLEET_API_KEY:?FLEET_API_KEY is required}"
: "${ROLLOUT_DATABASE_URL:?ROLLOUT_DATABASE_URL is required}"
: "${COLLECTION_DATABASE:?COLLECTION_DATABASE is required}"
: "${COLLECTION_PRIVATE_ROOT:?COLLECTION_PRIVATE_ROOT is required}"
: "${COLLECTION_ROUTE:?COLLECTION_ROUTE is required}"
: "${COLLECTION_WORKER_ID:?COLLECTION_WORKER_ID is required}"
: "${COLLECTION_LIMIT:?COLLECTION_LIMIT is required}"

root=/workspace/cyber-post-train
mkdir -p "$root/cyber_post_train" "$root/evals/fleet" "$root/configs/collection"
touch "$root/cyber_post_train/__init__.py" "$root/evals/__init__.py" "$root/evals/fleet/__init__.py"
install -m 0644 /bootstrap/jobs.py "$root/cyber_post_train/jobs.py"
for name in cluster_entry.py collection_fixed_proxy.py collection_fixed_proxy_v2.py \
  collection_completion_budget.py evaluate.py exact_pass4_crypto.py \
  exact_pass4_universe.py fixed_proxy.py model_artifact.py opencode_self_hosted.py \
  rollout_campaign.py rollout_ledger.py rollout_postgres.py rollout_worker.py \
  visible_action_collection.py visible_action_collection_v2.py \
  visible_action_collection_v3.py \
  visible_action_collection_job_entry.py; do
  if [[ -f "/bootstrap/$name" ]]; then
    install -m 0644 "/bootstrap/$name" "$root/evals/fleet/$name"
  fi
done
install -m 0600 /bootstrap/eval-config.json "$root/configs/collection/eval-config.json"
install -m 0600 /bootstrap/task-selection.json "$root/configs/collection/task-selection.json"
install -m 0600 /bootstrap/operation-authorization.json \
  "$root/configs/collection/operation-authorization.json"

export PATH="/docker-cli:$PATH"
for _ in $(seq 1 120); do
  docker info >/dev/null 2>&1 && break
  sleep 1
done
docker info >/dev/null

cd "$root"
exec uv run --no-project \
  --with httpx==0.28.1 \
  --with pyyaml==6.0.3 \
  --with 'psycopg[binary]==3.3.5' \
  python -m evals.fleet.visible_action_collection_job_entry \
  --config configs/collection/eval-config.json \
  --authorization configs/collection/operation-authorization.json \
  --private-root "$COLLECTION_PRIVATE_ROOT" \
  --database "$COLLECTION_DATABASE" \
  --harness-tar \
    /mnt/sfs/jobs/chris-q38-fleet-dev17-harness-build-v1/opencode-1.18.27-linux-amd64.tar \
  --harness-tar-sha256 sha256:b14bdf558990087504a95e209ff0c0e84039633f6d28b2cdbc483cd845c0c410 \
  --harness-receipt /mnt/sfs/jobs/chris-q38-fleet-dev17-harness-build-v1/BUILD.json \
  --harness-receipt-sha256 sha256:4781514a8431d6d87c671c49120639062f5a8e3afb016a04360752c896181745 \
  --route "$COLLECTION_ROUTE" \
  --worker-id "$COLLECTION_WORKER_ID" \
  --limit "$COLLECTION_LIMIT"
"""


class CollectionJobError(ValueError):
    """A fail-closed packet, launch, observation, or cleanup error."""


class Cluster(Protocol):
    def list(
        self,
        resource: str,
        namespace: str,
        *,
        field_selector: str | None = None,
        label_selector: str | None = None,
    ) -> dict[str, Any]: ...

    def get(self, resource: str, namespace: str, name: str) -> dict[str, Any]: ...

    def get_optional(self, resource: str, namespace: str, name: str) -> dict[str, Any] | None: ...

    def server_dry_run(self, namespace: str, bundle: dict[str, Any]) -> dict[str, Any]: ...

    def create_once(self, namespace: str, bundle: dict[str, Any]) -> dict[str, Any]: ...

    def delete_uid(self, resource: str, namespace: str, name: str, uid: str) -> dict[str, Any]: ...


@dataclass(frozen=True)
class LaunchPacket:
    path: Path
    value: dict[str, Any]
    files: dict[str, Path]

    @property
    def packet_sha256(self) -> str:
        return _file_sha256(self.path)


@dataclass(frozen=True)
class Package:
    packet: LaunchPacket
    config_map: dict[str, Any]
    job: dict[str, Any]

    @property
    def bundle(self) -> dict[str, Any]:
        return {"apiVersion": "v1", "kind": "List", "items": [self.config_map, self.job]}


def _canonical_digest(value: Any) -> str:
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, ValueError) as error:
        raise CollectionJobError("value is not canonical JSON") from error
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return "sha256:" + value.hexdigest()


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CollectionJobError(f"{label} must be a mapping")
    return value


def _json(path: Path, label: str) -> dict[str, Any]:
    try:
        return _mapping(json.loads(path.read_text(encoding="utf-8")), label)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CollectionJobError(f"{label} is not readable JSON") from error


def _safe_file(root: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise CollectionJobError(f"{label} path is missing")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise CollectionJobError(f"{label} path leaves the packet")
    path = root / Path(relative)
    if path.is_symlink() or not path.is_file():
        raise CollectionJobError(f"{label} is not a regular file")
    resolved = path.resolve(strict=True)
    if root not in resolved.parents:
        raise CollectionJobError(f"{label} path leaves the packet")
    return resolved


def _write_once(path: Path, data: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _write_json_once(path: Path, value: dict[str, Any]) -> None:
    _write_once(path, json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")


def _write_jsonl_once(path: Path, value: dict[str, Any]) -> None:
    _write_once(
        path,
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n",
    )


def _append_json(path: Path, value: dict[str, Any]) -> None:
    data = json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_APPEND)
    except OSError:
        return
    with os.fdopen(descriptor, "a", encoding="utf-8") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _git(repo_root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise CollectionJobError("git source identity check failed") from error
    if result.returncode:
        raise CollectionJobError("git source identity check failed")
    return result.stdout.strip()


def _copy_input(source: Path, destination: Path) -> dict[str, str]:
    if source.is_symlink() or not source.is_file():
        raise CollectionJobError("packet input must be a regular file")
    data = source.read_bytes()
    _write_once(destination, data)
    return {
        "path": destination.relative_to(destination.parents[1]).as_posix(),
        "sha256": "sha256:" + hashlib.sha256(data).hexdigest(),
    }


def _collection_runtime(config: dict[str, Any]) -> Any:
    schema = config.get("collection_runtime", {}).get("schema")
    if schema == collection_v2.RUNTIME_SCHEMA:
        return collection_v2
    if schema == collection_v3.RUNTIME_SCHEMA:
        return collection_v3
    raise CollectionJobError("collection runtime schema is unsupported")


def _source_files(runtime: Any) -> dict[str, str]:
    return SOURCE_FILES_V3 if runtime is collection_v3 else SOURCE_FILES_V2


def _validate_admitted_packet(
    value: dict[str, Any], authorization: dict[str, Any], runtime: Any
) -> None:
    unsigned = {key: item for key, item in value.items() if key != "sha256"}
    safety = value.get("execution_safety")
    expected_cells = authorization["planned_cells"]
    packet_schema = (
        "cyber_trajectory_collection_packet_v3"
        if runtime is collection_v3
        else "cyber_trajectory_collection_packet_v2"
    )
    if (
        value.get("schema") != packet_schema
        or value.get("operation_authorization_sha256") != authorization["sha256"]
        or value.get("eval_plan_sha256") != authorization["plan_sha256"]
        or value.get("sha256") != "sha256:" + digest(unsigned)
        or not isinstance(safety, dict)
        or safety.get("execution_mode") != "authorized_amd64_cpu_job_v1"
        or safety.get("cluster_wrapper_supported") is not True
        or safety.get("external_submission") is not False
        or safety.get("operation_authorization_sha256") != authorization["sha256"]
        or safety.get("operation_root_name") != authorization["operation_root_name"]
        or safety.get("dedicated_ledger_id") != authorization["dedicated_ledger_id"]
        or safety.get("identity_map_sha256") != authorization["identity_map_sha256"]
        or safety.get("planned_cells") != expected_cells
        or safety.get("maximum_planned_cells") != expected_cells
        or safety.get("cluster_job_execution_requirements") != CLUSTER_JOB_REQUIREMENTS
    ):
        raise CollectionJobError("collection packet does not authorize this exact CPU Job rail")
    if runtime is collection_v3:
        runtime_sha = "sha256:" + digest(runtime.runtime_identity())
        if (
            expected_cells not in {CANARY_CELLS, EXPECTED_CELLS}
            or value.get("completion_budget_runtime_sha256") != runtime_sha
            or safety.get("completion_budget_runtime_sha256") != runtime_sha
            or safety.get("completion_budget") != runtime.COMPLETION_BUDGET_POLICY
        ):
            raise CollectionJobError("collection packet v3 completion budget binding differs")
    elif expected_cells != EXPECTED_CELLS:
        raise CollectionJobError("collection packet v2 is only the exact 200-cell wave")


def prepare_packet(
    *,
    repo_root: Path,
    config_path: Path,
    task_selection_path: Path,
    authorization_path: Path,
    collection_packet_path: Path,
    output: Path,
    expected_source_commit: str,
) -> dict[str, Any]:
    """Build one private, create-once packet without external reads or writes."""
    repo_root = repo_root.resolve(strict=True)
    if _git(repo_root, "status", "--porcelain"):
        raise CollectionJobError("source worktree must be completely clean")
    commit = _git(repo_root, "rev-parse", "HEAD")
    tree = _git(repo_root, "rev-parse", "HEAD^{tree}")
    if (
        GIT_ID.fullmatch(commit) is None
        or GIT_ID.fullmatch(tree) is None
        or commit != expected_source_commit
    ):
        raise CollectionJobError("source commit is not the explicitly expected merged commit")
    source_config = _json(config_path, "collection eval config")
    runtime = _collection_runtime(source_config)
    source_files = _source_files(runtime)
    try:
        output.mkdir(mode=0o700)
    except FileExistsError as error:
        raise CollectionJobError("launch packet directory already exists") from error
    source_dir = output / "source"
    input_dir = output / "inputs"
    source_dir.mkdir(mode=0o700)
    input_dir.mkdir(mode=0o700)

    source_entries: dict[str, dict[str, str]] = {}
    for key, relative in sorted(source_files.items()):
        source = repo_root / relative
        if source.is_symlink() or not source.is_file():
            raise CollectionJobError("required source file is unavailable")
        data = source.read_bytes()
        destination = source_dir / key
        _write_once(destination, data)
        source_entries[key] = {
            "path": destination.relative_to(output).as_posix(),
            "sha256": "sha256:" + hashlib.sha256(data).hexdigest(),
        }
    run_path = source_dir / "run.sh"
    _write_once(run_path, BOOTSTRAP_SCRIPT.encode())
    source_entries["run.sh"] = {
        "path": run_path.relative_to(output).as_posix(),
        "sha256": _file_sha256(run_path),
    }

    inputs = {
        "eval_config": _copy_input(config_path, input_dir / "eval-config.json"),
        "task_selection": _copy_input(task_selection_path, input_dir / "task-selection.json"),
        "operation_authorization": _copy_input(
            authorization_path, input_dir / "operation-authorization.json"
        ),
        "collection_packet": _copy_input(
            collection_packet_path, input_dir / "collection-packet.json"
        ),
    }
    config = _json(input_dir / "eval-config.json", "collection eval config")
    if _collection_runtime(config) is not runtime:
        raise CollectionJobError("collection runtime changed while packaging")
    selection = _json(input_dir / "task-selection.json", "collection task selection")
    authorization_input = _json(
        input_dir / "operation-authorization.json", "operation authorization"
    )
    admitted_packet = _json(input_dir / "collection-packet.json", "collection packet")
    plan = runtime.compile_eval(config, relative_to=input_dir)
    authorization = runtime.validate_operation_authorization(authorization_input, plan)
    planned_cells = plan.get("planned_cells")
    allowed_cells = {EXPECTED_CELLS} if runtime is collection_v2 else {CANARY_CELLS, EXPECTED_CELLS}
    if planned_cells not in allowed_cells or plan.get("concurrency") != EXPECTED_CONCURRENCY:
        raise CollectionJobError("launcher cell count or concurrency differs from its exact rail")
    if config.get("images") != {"agent": AGENT_IMAGE_ID, "proxy": PROXY_IMAGE}:
        raise CollectionJobError("collection image identity differs from the qualified Job rail")
    if config.get("task_set") != "task-selection.json":
        raise CollectionJobError("collection config task selection path is not canonical")
    if selection.get("sha256") is None or inputs["task_selection"]["sha256"] != _file_sha256(
        input_dir / "task-selection.json"
    ):
        raise CollectionJobError("collection task selection identity is invalid")
    _validate_admitted_packet(admitted_packet, authorization, runtime)

    plan_short = authorization["plan_sha256"].removeprefix("sha256:")[:8]
    if runtime is collection_v2:
        job_name = f"chris-q38-base-train50-p4-v2-{plan_short}"
        database = f"q38_base_train50_actions_p4_v2_{plan_short}"
        worker_id = "base-v2"
    else:
        job_name = f"chris-{authorization['campaign_id']}-{plan_short}"
        database = f"{authorization['campaign_id'].replace('-', '_')}_{plan_short}"
        worker_id = "base-v3"
    if KUBERNETES_NAME.fullmatch(job_name) is None or DATABASE_NAME.fullmatch(database) is None:
        raise CollectionJobError("derived Job or database name is invalid")
    config_map_name = f"{job_name}-code"
    if KUBERNETES_NAME.fullmatch(config_map_name) is None:
        raise CollectionJobError("derived ConfigMap name is invalid")
    operation_root = f"{PRIVATE_ROOT}/{authorization['operation_root_name']}"
    body = {
        "schema": PACKET_SCHEMA,
        "context": CONTEXT,
        "namespace": NAMESPACE,
        "job_name": job_name,
        "config_map_name": config_map_name,
        "source": {
            "git_commit": commit,
            "git_tree": tree,
            "launcher_sha256": _file_sha256(Path(__file__)),
            "files": source_entries,
        },
        "inputs": inputs,
        "operation": {
            "campaign_id": authorization["campaign_id"],
            "plan_sha256": authorization["plan_sha256"],
            "planned_cell_universe_sha256": authorization["planned_cell_universe_sha256"],
            "operation_authorization_sha256": authorization["sha256"],
            "collection_packet_sha256": admitted_packet["sha256"],
            "planned_cells": planned_cells,
            "private_root": PRIVATE_ROOT,
            "operation_root": operation_root,
            "database": database,
            "route": "base",
            "worker_id": worker_id,
        },
        "images": {
            "controller": CONTROLLER_IMAGE,
            "dind": DIND_IMAGE,
            "agent": {
                "image_id": AGENT_IMAGE_ID,
                "tar_sfs_path": HARNESS_TAR,
                "tar_sha256": HARNESS_TAR_SHA256,
                "build_receipt_sfs_path": HARNESS_RECEIPT,
                "build_receipt_sha256": HARNESS_RECEIPT_SHA256,
                "platform": "linux/amd64",
            },
            "proxy": PROXY_IMAGE,
        },
        "storage": {
            "pvc": SFS_PVC,
            "mount_path": "/mnt/sfs",
            "operation_root": operation_root,
        },
        "secrets": {
            "fleet": {"name": FLEET_SECRET, "key": FLEET_SECRET_KEY},
            "postgres": {"name": POSTGRES_SECRET, "key": POSTGRES_SECRET_KEY},
        },
        "execution": {
            "mode": "authorized_amd64_cpu_job_v1",
            "root_kind": "Job",
            "priority_class_name": PRIORITY_CLASS,
            "queue_name": QUEUE_NAME,
            "active_deadline_seconds": ACTIVE_DEADLINE_SECONDS,
            "backoff_limit": 0,
            "restart_policy": "Never",
            "requested_gpus": 0,
            "server_preview_count": 2,
            "create_request_count": 1,
            "automatic_create_retry": False,
            "exact_uid_cleanup_required": True,
            "failure_alerts": "off",
        },
    }
    packet = {**body, "sha256": "sha256:" + digest(body)}
    packet_path = output / "launch-packet.json"
    _write_json_once(packet_path, packet)
    return {
        "prepared": str(output),
        "packet": str(packet_path),
        "packet_file_sha256": _file_sha256(packet_path),
        "packet_sha256": packet["sha256"],
        "operation_authorization_sha256": authorization["sha256"],
        "planned_cells": planned_cells,
        "external_mutations": 0,
    }


def _binding_file(root: Path, value: object, label: str, expected_name: str | None = None) -> Path:
    entry = _mapping(value, label)
    if set(entry) != {"path", "sha256"} or SHA256.fullmatch(str(entry.get("sha256"))) is None:
        raise CollectionJobError(f"{label} binding is invalid")
    path = _safe_file(root, entry["path"], label)
    if expected_name is not None and path.name != expected_name:
        raise CollectionJobError(f"{label} filename is not canonical")
    if _file_sha256(path) != entry["sha256"]:
        raise CollectionJobError(f"{label} bytes differ from the packet")
    return path


def load_packet(path: Path) -> LaunchPacket:
    if path.is_symlink() or not path.is_file():
        raise CollectionJobError("launch packet must be a regular file")
    packet_path = path.resolve()
    value = _json(packet_path, "launch packet")
    if (
        set(value)
        != {
            "schema",
            "context",
            "namespace",
            "job_name",
            "config_map_name",
            "source",
            "inputs",
            "operation",
            "images",
            "storage",
            "secrets",
            "execution",
            "sha256",
        }
        or value.get("schema") != PACKET_SCHEMA
    ):
        raise CollectionJobError("launch packet schema is unsupported")
    unsigned = {key: item for key, item in value.items() if key != "sha256"}
    if value.get("sha256") != "sha256:" + digest(unsigned):
        raise CollectionJobError("launch packet seal changed")
    if value.get("context") != CONTEXT or value.get("namespace") != NAMESPACE:
        raise CollectionJobError("launch packet cluster identity is not authorized")
    for field in ("job_name", "config_map_name"):
        if KUBERNETES_NAME.fullmatch(str(value.get(field))) is None:
            raise CollectionJobError("launch packet Kubernetes name is invalid")
    root = packet_path.parent
    source = _mapping(value.get("source"), "source identity")
    if set(source) != {"git_commit", "git_tree", "launcher_sha256", "files"}:
        raise CollectionJobError("source identity has unknown or missing fields")
    if any(
        GIT_ID.fullmatch(str(source.get(field))) is None for field in ("git_commit", "git_tree")
    ):
        raise CollectionJobError("source git identity is invalid")
    if source.get("launcher_sha256") != _file_sha256(Path(__file__)):
        raise CollectionJobError("launcher source changed after packet preparation")
    source_files = _mapping(source.get("files"), "source files")
    allowed_source_sets = [set(files) | {"run.sh"} for files in (SOURCE_FILES_V2, SOURCE_FILES_V3)]
    if set(source_files) not in allowed_source_sets:
        raise CollectionJobError("source file set is incomplete")
    files: dict[str, Path] = {}
    for key in sorted(source_files):
        files[key] = _binding_file(root, source_files[key], f"source file {key}")
    inputs = _mapping(value.get("inputs"), "packet inputs")
    if set(inputs) != {
        "eval_config",
        "task_selection",
        "operation_authorization",
        "collection_packet",
    }:
        raise CollectionJobError("launch packet input set is incomplete")
    for key, filename in {
        "eval_config": "eval-config.json",
        "task_selection": "task-selection.json",
        "operation_authorization": "operation-authorization.json",
        "collection_packet": "collection-packet.json",
    }.items():
        files[key] = _binding_file(root, inputs[key], key, filename)
    return LaunchPacket(path=packet_path, value=value, files=files)


def _env_literal(name: str, value: str) -> dict[str, str]:
    # Kubernetes drops an explicitly empty EnvVar.value from server-rendered
    # objects.  Emit the canonical omitted form up front so the sealed manifest
    # and both server previews compare byte-stably.
    return {"name": name} if value == "" else {"name": name, "value": value}


def _env_secret(name: str, secret: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "valueFrom": {"secretKeyRef": {"name": secret["name"], "key": secret["key"]}},
    }


def _build_config_map(packet: LaunchPacket) -> dict[str, Any]:
    value = packet.value
    source_keys = set(_mapping(value["source"]["files"], "source files"))
    data = {key: packet.files[key].read_text(encoding="utf-8") for key in sorted(source_keys)}
    data.update(
        {
            "eval-config.json": packet.files["eval_config"].read_text(encoding="utf-8"),
            "task-selection.json": packet.files["task_selection"].read_text(encoding="utf-8"),
            "operation-authorization.json": packet.files["operation_authorization"].read_text(
                encoding="utf-8"
            ),
        }
    )
    config_map = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": value["config_map_name"],
            "namespace": NAMESPACE,
            "labels": {
                "cyber-post-train.fleet.ai/owner": "chris",
                "cyber-post-train.fleet.ai/experiment": value["job_name"],
            },
            "annotations": {
                OPERATION_AUTHORIZATION_ANNOTATION: value["operation"][
                    "operation_authorization_sha256"
                ],
                COLLECTION_PACKET_ANNOTATION: value["operation"]["collection_packet_sha256"],
            },
        },
        "immutable": True,
        "data": data,
    }
    if (
        len(json.dumps(config_map, sort_keys=True, separators=(",", ":")).encode())
        > CONFIG_MAP_MAX_SERIALIZED_BYTES
    ):
        raise CollectionJobError("immutable source ConfigMap exceeds its reviewed size ceiling")
    return config_map


def _build_job(packet: LaunchPacket) -> dict[str, Any]:
    value = packet.value
    operation = value["operation"]
    secrets = value["secrets"]
    annotations = {
        FAILURE_ALERT_ANNOTATION: "off",
        CREATE_ONCE_ANNOTATION: "true",
        OPERATION_AUTHORIZATION_ANNOTATION: operation["operation_authorization_sha256"],
        COLLECTION_PACKET_ANNOTATION: operation["collection_packet_sha256"],
    }
    labels = {
        "cyber-post-train.fleet.ai/owner": "chris",
        "cyber-post-train.fleet.ai/experiment": value["job_name"],
        "kueue.x-k8s.io/queue-name": QUEUE_NAME,
    }
    shared_mounts = [
        {"name": "docker-socket", "mountPath": "/var/run"},
        {"name": "docker-data", "mountPath": "/var/lib/docker"},
        {"name": "docker-bind", "mountPath": "/docker-bind"},
        {"name": "workspace", "mountPath": "/workspace"},
        {"name": "sfs", "mountPath": "/mnt/sfs"},
    ]
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": value["job_name"],
            "namespace": NAMESPACE,
            "labels": labels,
            "annotations": annotations,
        },
        "spec": {
            "backoffLimit": 0,
            "parallelism": 1,
            "completions": 1,
            "activeDeadlineSeconds": ACTIVE_DEADLINE_SECONDS,
            "template": {
                "metadata": {
                    "labels": {
                        "cyber-post-train.fleet.ai/owner": "chris",
                        "cyber-post-train.fleet.ai/experiment": value["job_name"],
                        "cyber-post-train.fleet.ai/postgres-client": "true",
                    },
                    "annotations": {
                        OPERATION_AUTHORIZATION_ANNOTATION: operation[
                            "operation_authorization_sha256"
                        ]
                    },
                },
                "spec": {
                    "priorityClassName": PRIORITY_CLASS,
                    "restartPolicy": "Never",
                    "terminationGracePeriodSeconds": 1800,
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
                            "command": [
                                "/bin/sh",
                                "-ceu",
                                "--",
                                "cp /usr/local/bin/docker /docker-cli/docker; "
                                "chmod 0555 /docker-cli/docker",
                            ],
                            "resources": {
                                "requests": {"cpu": "100m", "memory": "64Mi"},
                                "limits": {"cpu": "1", "memory": "256Mi"},
                            },
                            "volumeMounts": [{"name": "docker-cli", "mountPath": "/docker-cli"}],
                        },
                        {
                            "name": "dind",
                            "image": DIND_IMAGE,
                            "restartPolicy": "Always",
                            "args": ["--host=unix:///var/run/docker.sock", "--tls=false"],
                            "env": [_env_literal("DOCKER_TLS_CERTDIR", "")],
                            "startupProbe": {
                                "exec": {"command": ["docker", "info"]},
                                "periodSeconds": 2,
                                "failureThreshold": 120,
                            },
                            "securityContext": {"privileged": True},
                            "resources": {
                                "requests": {
                                    "cpu": "2",
                                    "memory": "2Gi",
                                    "ephemeral-storage": "10Gi",
                                },
                                "limits": {
                                    "cpu": "4",
                                    "memory": "4Gi",
                                    "ephemeral-storage": "40Gi",
                                },
                            },
                            "volumeMounts": shared_mounts,
                        },
                    ],
                    "containers": [
                        {
                            "name": "collector",
                            "image": CONTROLLER_IMAGE,
                            "command": ["/bin/bash", "/bootstrap/run.sh"],
                            "env": [
                                _env_literal("DOCKER_HOST", "unix:///var/run/docker.sock"),
                                _env_literal("DOCKER_TLS_CERTDIR", ""),
                                _env_literal("DOCKER_BIND_ROOT", "/docker-bind"),
                                _env_literal("COLLECTION_DATABASE", operation["database"]),
                                _env_literal("COLLECTION_PRIVATE_ROOT", operation["private_root"]),
                                _env_literal("COLLECTION_ROUTE", operation["route"]),
                                _env_literal("COLLECTION_WORKER_ID", operation["worker_id"]),
                                _env_literal("COLLECTION_LIMIT", str(operation["planned_cells"])),
                                _env_secret("FLEET_API_KEY", secrets["fleet"]),
                                _env_secret("ROLLOUT_DATABASE_URL", secrets["postgres"]),
                            ],
                            "resources": {
                                "requests": {
                                    "cpu": "4",
                                    "memory": "32Gi",
                                    "ephemeral-storage": "10Gi",
                                },
                                "limits": {
                                    "cpu": "8",
                                    "memory": "48Gi",
                                    "ephemeral-storage": "40Gi",
                                },
                            },
                            "volumeMounts": [
                                {"name": "bootstrap", "mountPath": "/bootstrap", "readOnly": True},
                                {
                                    "name": "docker-cli",
                                    "mountPath": "/docker-cli",
                                    "readOnly": True,
                                },
                                *[
                                    mount
                                    for mount in shared_mounts
                                    if mount["name"] != "docker-data"
                                ],
                            ],
                        }
                    ],
                    "volumes": [
                        {
                            "name": "bootstrap",
                            "configMap": {
                                "name": value["config_map_name"],
                                "defaultMode": 256,
                            },
                        },
                        {"name": "docker-cli", "emptyDir": {}},
                        {"name": "docker-data", "emptyDir": {"sizeLimit": "80Gi"}},
                        {"name": "docker-socket", "emptyDir": {}},
                        {"name": "docker-bind", "emptyDir": {"sizeLimit": "2Gi"}},
                        {"name": "workspace", "emptyDir": {"sizeLimit": "5Gi"}},
                        {
                            "name": "sfs",
                            "persistentVolumeClaim": {"claimName": SFS_PVC},
                        },
                    ],
                },
            },
        },
    }


def _containers(pod: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for field in ("initContainers", "containers"):
        rows = pod.get(field, [])
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise CollectionJobError("Job container inventory is invalid")
        result.extend(rows)
    return result


def _assert_zero_gpu(pod: dict[str, Any]) -> None:
    for container in _containers(pod):
        resources = _mapping(container.get("resources", {}), "container resources")
        for field in ("requests", "limits"):
            values = _mapping(resources.get(field, {}), "container resource quantities")
            if "nvidia.com/gpu" in values:
                raise CollectionJobError("collection Job must request and limit zero GPUs")


def build_package(packet_path: Path) -> Package:
    packet = load_packet(packet_path)
    value = packet.value
    config = _json(packet.files["eval_config"], "collection eval config")
    runtime = _collection_runtime(config)
    if set(packet.value["source"]["files"]) != set(_source_files(runtime)) | {"run.sh"}:
        raise CollectionJobError("source file set does not match the selected runtime")
    authorization_input = _json(packet.files["operation_authorization"], "operation authorization")
    admitted_packet = _json(packet.files["collection_packet"], "collection packet")
    plan = runtime.compile_eval(config, relative_to=packet.files["eval_config"].parent)
    authorization = runtime.validate_operation_authorization(authorization_input, plan)
    operation = _mapping(value.get("operation"), "operation identity")
    if set(operation) != {
        "campaign_id",
        "plan_sha256",
        "planned_cell_universe_sha256",
        "operation_authorization_sha256",
        "collection_packet_sha256",
        "planned_cells",
        "private_root",
        "operation_root",
        "database",
        "route",
        "worker_id",
    }:
        raise CollectionJobError("operation identity has unknown or missing fields")
    planned_cells = authorization["planned_cells"]
    worker_id = "base-v3" if runtime is collection_v3 else "base-v2"
    expected_operation = {
        "campaign_id": authorization["campaign_id"],
        "plan_sha256": authorization["plan_sha256"],
        "planned_cell_universe_sha256": authorization["planned_cell_universe_sha256"],
        "operation_authorization_sha256": authorization["sha256"],
        "collection_packet_sha256": admitted_packet.get("sha256"),
        "planned_cells": planned_cells,
        "private_root": PRIVATE_ROOT,
        "operation_root": f"{PRIVATE_ROOT}/{authorization['operation_root_name']}",
        "database": operation.get("database"),
        "route": "base",
        "worker_id": worker_id,
    }
    if (
        operation != expected_operation
        or DATABASE_NAME.fullmatch(str(operation["database"])) is None
    ):
        raise CollectionJobError("operation identity differs from the exact authorized wave")
    _validate_admitted_packet(admitted_packet, authorization, runtime)
    allowed_cells = {EXPECTED_CELLS} if runtime is collection_v2 else {CANARY_CELLS, EXPECTED_CELLS}
    if (
        plan.get("planned_cells") not in allowed_cells
        or plan.get("concurrency") != EXPECTED_CONCURRENCY
    ):
        raise CollectionJobError("collection Job cell count or concurrency differs")
    if value.get("images") != {
        "controller": CONTROLLER_IMAGE,
        "dind": DIND_IMAGE,
        "agent": {
            "image_id": AGENT_IMAGE_ID,
            "tar_sfs_path": HARNESS_TAR,
            "tar_sha256": HARNESS_TAR_SHA256,
            "build_receipt_sfs_path": HARNESS_RECEIPT,
            "build_receipt_sha256": HARNESS_RECEIPT_SHA256,
            "platform": "linux/amd64",
        },
        "proxy": PROXY_IMAGE,
    }:
        raise CollectionJobError("launch packet image identities differ")
    if value.get("storage") != {
        "pvc": SFS_PVC,
        "mount_path": "/mnt/sfs",
        "operation_root": operation["operation_root"],
    }:
        raise CollectionJobError("launch packet SFS identity differs")
    if value.get("secrets") != {
        "fleet": {"name": FLEET_SECRET, "key": FLEET_SECRET_KEY},
        "postgres": {"name": POSTGRES_SECRET, "key": POSTGRES_SECRET_KEY},
    }:
        raise CollectionJobError("launch packet secret references differ")
    if value.get("execution") != {
        "mode": "authorized_amd64_cpu_job_v1",
        "root_kind": "Job",
        "priority_class_name": PRIORITY_CLASS,
        "queue_name": QUEUE_NAME,
        "active_deadline_seconds": ACTIVE_DEADLINE_SECONDS,
        "backoff_limit": 0,
        "restart_policy": "Never",
        "requested_gpus": 0,
        "server_preview_count": 2,
        "create_request_count": 1,
        "automatic_create_retry": False,
        "exact_uid_cleanup_required": True,
        "failure_alerts": "off",
    }:
        raise CollectionJobError("launch packet execution safety differs")
    config_map = _build_config_map(packet)
    job = _build_job(packet)
    pod = job["spec"]["template"]["spec"]
    _assert_zero_gpu(pod)
    if (
        job["metadata"]["annotations"].get(FAILURE_ALERT_ANNOTATION) != "off"
        or job["spec"]["activeDeadlineSeconds"] != ACTIVE_DEADLINE_SECONDS
        or job["spec"]["backoffLimit"] != 0
        or pod.get("priorityClassName") != PRIORITY_CLASS
        or pod.get("restartPolicy") != "Never"
        or pod.get("nodeSelector", {}).get("kubernetes.io/arch") != "amd64"
    ):
        raise CollectionJobError("rendered collection Job lost its safety contract")
    return Package(packet=packet, config_map=config_map, job=job)


def _items(value: object, label: str) -> list[dict[str, Any]]:
    mapping = _mapping(value, label)
    items = mapping.get("items")
    if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
        raise CollectionJobError(f"{label} is not a Kubernetes list")
    return items


def _objects(value: object, label: str) -> list[dict[str, Any]]:
    mapping = _mapping(value, label)
    return _items(mapping, label) if mapping.get("kind") == "List" else [mapping]


def _metadata(value: dict[str, Any], label: str) -> dict[str, Any]:
    metadata = value.get("metadata")
    if not isinstance(metadata, dict):
        raise CollectionJobError(f"{label} metadata is invalid")
    return metadata


def _response_object(
    value: object, *, api_version: str, kind: str, namespace: str, name: str, label: str
) -> dict[str, Any]:
    matches = []
    for item in _objects(value, label):
        metadata = _metadata(item, label)
        if (
            item.get("apiVersion") == api_version
            and item.get("kind") == kind
            and metadata.get("namespace") == namespace
            and metadata.get("name") == name
        ):
            matches.append(item)
    if len(matches) != 1:
        raise CollectionJobError(f"{label} omitted the exact {kind}")
    return matches[0]


def _contains(actual: Any, expected: Any) -> bool:
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and _contains(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(_contains(left, right) for left, right in zip(actual, expected, strict=True))
        )
    return actual == expected


def validate_server_preview(value: dict[str, Any], package: Package) -> str:
    job = _response_object(
        value,
        api_version="batch/v1",
        kind="Job",
        namespace=NAMESPACE,
        name=package.packet.value["job_name"],
        label="server preview",
    )
    config_map = _response_object(
        value,
        api_version="v1",
        kind="ConfigMap",
        namespace=NAMESPACE,
        name=package.packet.value["config_map_name"],
        label="server preview",
    )
    if _metadata(job, "server Job").get("annotations", {}).get(FAILURE_ALERT_ANNOTATION) != "off":
        raise CollectionJobError("server-rendered root Job lacks failure-alerts off")
    try:
        stable_job = stable_job_preview(job)
        expected_job = stable_job_preview(package.job)
    except ValueError as error:
        raise CollectionJobError("server-rendered Job has no stable form") from error
    if not _contains(stable_job, expected_job):
        raise CollectionJobError("server-rendered Job differs from the sealed packet")
    stable_config = {
        "apiVersion": config_map.get("apiVersion"),
        "kind": config_map.get("kind"),
        "metadata": {
            "name": _metadata(config_map, "server ConfigMap").get("name"),
            "namespace": _metadata(config_map, "server ConfigMap").get("namespace"),
        },
        "immutable": config_map.get("immutable"),
        "data": config_map.get("data"),
    }
    expected_config = {
        "apiVersion": package.config_map["apiVersion"],
        "kind": package.config_map["kind"],
        "metadata": {
            "name": package.config_map["metadata"]["name"],
            "namespace": package.config_map["metadata"]["namespace"],
        },
        "immutable": True,
        "data": package.config_map["data"],
    }
    if stable_config != expected_config:
        raise CollectionJobError("server-rendered ConfigMap differs from sealed bytes")
    _assert_zero_gpu(job["spec"]["template"]["spec"])
    return _canonical_digest({"job": stable_job, "config_map": stable_config})


def exact_name_census(package: Package, cluster: Cluster) -> dict[str, int]:
    value = package.packet.value
    counts: dict[str, int] = {}
    for label, resource, name in (
        ("jobs", "jobs.batch", value["job_name"]),
        ("config_maps", "configmaps", value["config_map_name"]),
    ):
        rows = _items(
            cluster.list(
                resource,
                NAMESPACE,
                field_selector=f"metadata.name={name}",
            ),
            label,
        )
        exact = [row for row in rows if _metadata(row, label).get("name") == name]
        counts[label] = len(exact)
        if exact:
            raise CollectionJobError("exact collection Kubernetes identity already exists")
    pods = _items(
        cluster.list(
            "pods",
            NAMESPACE,
            label_selector=f"batch.kubernetes.io/job-name={value['job_name']}",
        ),
        "pods",
    )
    counts["pods"] = len(pods)
    if pods:
        raise CollectionJobError("exact collection Job-owned Pod already exists")
    return counts


def _created_name_observation(package: Package, cluster: Cluster) -> dict[str, bool]:
    result: dict[str, bool] = {}
    for label, resource, name in (
        ("job", "jobs.batch", package.packet.value["job_name"]),
        ("config_map", "configmaps", package.packet.value["config_map_name"]),
    ):
        try:
            rows = _items(
                cluster.list(resource, NAMESPACE, field_selector=f"metadata.name={name}"),
                label,
            )
            result[label] = any(_metadata(row, label).get("name") == name for row in rows)
        except Exception:
            result[label] = False
    return result


def launch_once(packet_path: Path, *, cluster: Cluster, journal: Path) -> dict[str, Any]:
    if journal.exists() or journal.is_symlink():
        raise CollectionJobError("create journal exists; reconcile and never retry")
    if not journal.parent.is_dir():
        raise CollectionJobError("create journal parent is unavailable")
    package = build_package(packet_path)
    first_census = exact_name_census(package, cluster)
    first_preview = validate_server_preview(
        cluster.server_dry_run(NAMESPACE, package.bundle), package
    )
    second_preview = validate_server_preview(
        cluster.server_dry_run(NAMESPACE, package.bundle), package
    )
    if first_preview != second_preview:
        raise CollectionJobError("server preview changed across identical requests")
    final_census = exact_name_census(package, cluster)
    intent = {
        "schema": CREATE_INTENT_SCHEMA,
        "state": "KUBERNETES_CREATE_INTENT_DO_NOT_RETRY",
        "packet_file_sha256": package.packet.packet_sha256,
        "packet_sha256": package.packet.value["sha256"],
        "operation_authorization_sha256": package.packet.value["operation"][
            "operation_authorization_sha256"
        ],
        "collection_packet_sha256": package.packet.value["operation"]["collection_packet_sha256"],
        "context": CONTEXT,
        "namespace": NAMESPACE,
        "job_name": package.packet.value["job_name"],
        "config_map_name": package.packet.value["config_map_name"],
        "server_preview_sha256": first_preview,
        "first_exact_name_census": first_census,
        "final_exact_name_census": final_census,
    }
    _write_jsonl_once(journal, intent)
    try:
        response = cluster.create_once(NAMESPACE, package.bundle)
        job = _response_object(
            response,
            api_version="batch/v1",
            kind="Job",
            namespace=NAMESPACE,
            name=package.packet.value["job_name"],
            label="create response",
        )
        config_map = _response_object(
            response,
            api_version="v1",
            kind="ConfigMap",
            namespace=NAMESPACE,
            name=package.packet.value["config_map_name"],
            label="create response",
        )
        job_metadata = _metadata(job, "created Job")
        config_metadata = _metadata(config_map, "created ConfigMap")
        job_uid = job_metadata.get("uid")
        config_map_uid = config_metadata.get("uid")
        if (
            job_metadata.get("annotations", {}).get(FAILURE_ALERT_ANNOTATION) != "off"
            or KUBERNETES_UID.fullmatch(str(job_uid)) is None
            or KUBERNETES_UID.fullmatch(str(config_map_uid)) is None
        ):
            raise CollectionJobError("create response lacks root alert annotation or exact UIDs")
    except Exception as error:
        _append_json(
            journal,
            {
                "state": "KUBERNETES_CREATE_RESPONSE_UNCERTAIN_DO_NOT_RETRY",
                "observed_exact_names": _created_name_observation(package, cluster),
            },
        )
        if isinstance(error, CollectionJobError):
            raise
        raise CollectionJobError("create response uncertain; reconcile and never retry") from None
    result = {
        "submitted": True,
        "gpus": 0,
        "job_name": package.packet.value["job_name"],
        "job_uid": job_uid,
        "config_map_name": package.packet.value["config_map_name"],
        "config_map_uid": config_map_uid,
        "operation_authorization_sha256": package.packet.value["operation"][
            "operation_authorization_sha256"
        ],
        "collection_packet_sha256": package.packet.value["operation"]["collection_packet_sha256"],
        "server_preview_sha256": first_preview,
    }
    _append_json(journal, {"state": "KUBERNETES_CREATE_RESPONSE", **result})
    return result


def _journal_binding(path: Path, package: Package) -> dict[str, Any]:
    try:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as error:
        raise CollectionJobError("create journal is unreadable") from error
    responses = [row for row in rows if row.get("state") == "KUBERNETES_CREATE_RESPONSE"]
    uncertain = any(
        row.get("state") == "KUBERNETES_CREATE_RESPONSE_UNCERTAIN_DO_NOT_RETRY" for row in rows
    )
    if uncertain or len(responses) != 1:
        raise CollectionJobError("create journal has no unambiguous exact-UID binding")
    response = responses[0]
    if (
        response.get("job_name") != package.packet.value["job_name"]
        or response.get("config_map_name") != package.packet.value["config_map_name"]
        or response.get("operation_authorization_sha256")
        != package.packet.value["operation"]["operation_authorization_sha256"]
        or KUBERNETES_UID.fullmatch(str(response.get("job_uid"))) is None
        or KUBERNETES_UID.fullmatch(str(response.get("config_map_uid"))) is None
    ):
        raise CollectionJobError("create journal binding differs from the packet")
    return response


def _owned_by(value: dict[str, Any], *, kind: str, name: str, uid: str) -> bool:
    owners = _metadata(value, value.get("kind", "resource")).get("ownerReferences")
    return isinstance(owners, list) and any(
        isinstance(owner, dict)
        and owner.get("kind") == kind
        and owner.get("name") == name
        and owner.get("uid") == uid
        and owner.get("controller") is True
        for owner in owners
    )


def _terminal_condition(job: dict[str, Any]) -> str:
    status = _mapping(job.get("status"), "created Job status")
    if status.get("active", 0) != 0:
        raise CollectionJobError("collection Job still has active Pods")
    conditions = status.get("conditions")
    if not isinstance(conditions, list):
        raise CollectionJobError("collection Job has no terminal condition")
    terminal = {
        row.get("type")
        for row in conditions
        if isinstance(row, dict)
        and row.get("status") == "True"
        and row.get("type") in {"Complete", "Failed"}
    }
    if len(terminal) != 1:
        raise CollectionJobError("collection Job is not unambiguously terminal")
    return terminal.pop()


def _bound_resource(value: dict[str, Any], *, label: str, name: str, uid: str) -> dict[str, Any]:
    metadata = _metadata(value, label)
    if metadata.get("name") != name or metadata.get("uid") != uid:
        raise CollectionJobError(f"{label} exact name or UID changed")
    return metadata


def _owned_inventory(
    cluster: Cluster, binding: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pods = _items(
        cluster.list(
            "pods",
            NAMESPACE,
            label_selector=f"batch.kubernetes.io/controller-uid={binding['job_uid']}",
        ),
        "owned Pods",
    )
    for pod in pods:
        if not _owned_by(pod, kind="Job", name=binding["job_name"], uid=binding["job_uid"]):
            raise CollectionJobError("terminal Pod owner binding differs")
    workloads = _items(
        cluster.list(
            "workloads.kueue.x-k8s.io",
            NAMESPACE,
            label_selector=f"kueue.x-k8s.io/job-uid={binding['job_uid']}",
        ),
        "owned Workloads",
    )
    for workload in workloads:
        if not _owned_by(workload, kind="Job", name=binding["job_name"], uid=binding["job_uid"]):
            raise CollectionJobError("terminal Workload owner binding differs")
    return pods, workloads


def observe_status(packet_path: Path, *, cluster: Cluster, journal: Path) -> dict[str, Any]:
    """Read only the exact bound Kubernetes identities; never read logs or data."""
    package = build_package(packet_path)
    binding = _journal_binding(journal, package)
    root = cluster.get_optional("jobs.batch", NAMESPACE, binding["job_name"])
    config_map = cluster.get_optional("configmaps", NAMESPACE, binding["config_map_name"])
    if root is not None:
        metadata = _bound_resource(
            root,
            label="created Job",
            name=binding["job_name"],
            uid=binding["job_uid"],
        )
        if metadata.get("annotations", {}).get(FAILURE_ALERT_ANNOTATION) != "off":
            raise CollectionJobError("created root Job lost failure-alerts off")
        _assert_zero_gpu(root["spec"]["template"]["spec"])
    if config_map is not None:
        _bound_resource(
            config_map,
            label="created ConfigMap",
            name=binding["config_map_name"],
            uid=binding["config_map_uid"],
        )
    pods, workloads = _owned_inventory(cluster, binding)
    phases: dict[str, int] = {}
    for pod in pods:
        phase = pod.get("status", {}).get("phase")
        name = phase if isinstance(phase, str) and phase else "Missing"
        phases[name] = phases.get(name, 0) + 1
    terminal = ""
    active = 0
    if root is not None:
        status = root.get("status", {})
        if not isinstance(status, dict) or type(status.get("active", 0)) is not int:
            raise CollectionJobError("created Job status is invalid")
        active = status.get("active", 0)
        conditions = status.get("conditions", [])
        if not isinstance(conditions, list):
            raise CollectionJobError("created Job conditions are invalid")
        observed = {
            row.get("type")
            for row in conditions
            if isinstance(row, dict)
            and row.get("status") == "True"
            and row.get("type") in {"Complete", "Failed"}
        }
        if len(observed) > 1:
            raise CollectionJobError("created Job has contradictory terminal conditions")
        terminal = next(iter(observed), "")
    active_phases = sum(phases.get(phase, 0) for phase in ("Pending", "Running", "Unknown"))
    if root is None and config_map is None and not pods and not workloads:
        state = "release_confirmed"
    elif terminal and active == 0 and active_phases == 0:
        state = "terminal_ready_for_cleanup"
    elif root is None:
        state = "cleanup_in_progress"
    else:
        state = "running_or_queued"
    return {
        "schema": STATUS_SCHEMA,
        "state": state,
        "operation_authorization_sha256": binding["operation_authorization_sha256"],
        "job": {
            "name": binding["job_name"],
            "uid": binding["job_uid"],
            "present": root is not None,
            "terminal_condition": terminal or None,
            "active": active,
        },
        "config_map": {
            "name": binding["config_map_name"],
            "uid": binding["config_map_uid"],
            "present": config_map is not None,
        },
        "owned_pod_phases": dict(sorted(phases.items())),
        "owned_workloads": len(workloads),
        "gpus": 0,
        "logs_traces_scores_or_credentials_read": False,
    }


def _load_cleanup_intent(path: Path, expected_body: dict[str, Any]) -> dict[str, Any] | None:
    if not path.exists() and not path.is_symlink():
        return None
    if path.is_symlink() or not path.is_file():
        raise CollectionJobError("cleanup intent path is not a regular file")
    value = _json(path, "cleanup intent")
    if value != {
        **expected_body,
        "sha256": _canonical_digest(expected_body),
    }:
        raise CollectionJobError("cleanup intent differs from the exact UID binding")
    return value


def _existing_cleanup_receipt(path: Path, binding: dict[str, Any]) -> dict[str, Any] | None:
    if not path.exists() and not path.is_symlink():
        return None
    if path.is_symlink() or not path.is_file():
        raise CollectionJobError("cleanup receipt path is not a regular file")
    value = _json(path, "cleanup receipt")
    unsigned = {key: item for key, item in value.items() if key != "sha256"}
    if (
        value.get("schema") != CLEANUP_RECEIPT_SCHEMA
        or value.get("sha256") != _canonical_digest(unsigned)
        or value.get("operation_authorization_sha256") != binding["operation_authorization_sha256"]
        or value.get("job") != {"name": binding["job_name"], "uid": binding["job_uid"]}
        or value.get("config_map")
        != {"name": binding["config_map_name"], "uid": binding["config_map_uid"]}
        or value.get("release_confirmed") is not True
    ):
        raise CollectionJobError("cleanup receipt is invalid")
    return value


def cleanup_once(
    packet_path: Path,
    *,
    cluster: Cluster,
    journal: Path,
    receipt_path: Path,
    timeout_seconds: float = 600,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    if timeout_seconds <= 0:
        raise CollectionJobError("cleanup timeout must be positive")
    package = build_package(packet_path)
    binding = _journal_binding(journal, package)
    intent_path = receipt_path.with_name(receipt_path.name + ".intent")
    completed = _existing_cleanup_receipt(receipt_path, binding)
    if completed is not None:
        return completed
    root = cluster.get_optional("jobs.batch", NAMESPACE, binding["job_name"])
    config_map = cluster.get_optional("configmaps", NAMESPACE, binding["config_map_name"])
    if root is not None:
        root_metadata = _bound_resource(
            root,
            label="created Job",
            name=binding["job_name"],
            uid=binding["job_uid"],
        )
        if root_metadata.get("annotations", {}).get(FAILURE_ALERT_ANNOTATION) != "off":
            raise CollectionJobError("terminal root Job lost failure-alerts off")
        condition = _terminal_condition(root)
        _assert_zero_gpu(root["spec"]["template"]["spec"])
    else:
        if not intent_path.exists() or intent_path.is_symlink():
            raise CollectionJobError(
                "exact terminal Job must be observed before the first cleanup intent"
            )
        raw_intent = _json(intent_path, "cleanup intent")
        condition = raw_intent.get("terminal_condition")
        if condition not in {"Complete", "Failed"}:
            raise CollectionJobError("cleanup intent has no exact terminal observation")
    if config_map is not None:
        _bound_resource(
            config_map,
            label="created ConfigMap",
            name=binding["config_map_name"],
            uid=binding["config_map_uid"],
        )
    pods, workloads = _owned_inventory(cluster, binding)
    for pod in pods:
        if pod.get("status", {}).get("phase") in {"Pending", "Running", "Unknown"}:
            raise CollectionJobError("terminal collection still has an active or unknown Pod")

    intent_body = {
        "schema": CLEANUP_INTENT_SCHEMA,
        "state": "EXACT_UID_FOREGROUND_DELETE_INTENT_RESUMABLE_V1",
        "packet_file_sha256": package.packet.packet_sha256,
        "operation_authorization_sha256": binding["operation_authorization_sha256"],
        "job": {"name": binding["job_name"], "uid": binding["job_uid"]},
        "config_map": {
            "name": binding["config_map_name"],
            "uid": binding["config_map_uid"],
        },
        "terminal_condition": condition,
    }
    intent = _load_cleanup_intent(intent_path, intent_body)
    if intent is None:
        intent = {**intent_body, "sha256": _canonical_digest(intent_body)}
        _write_json_once(intent_path, intent)
    if root is not None:
        try:
            cluster.delete_uid("jobs.batch", NAMESPACE, binding["job_name"], binding["job_uid"])
        except Exception:
            observed = cluster.get_optional("jobs.batch", NAMESPACE, binding["job_name"])
            if observed is not None:
                _bound_resource(
                    observed,
                    label="created Job",
                    name=binding["job_name"],
                    uid=binding["job_uid"],
                )
                raise CollectionJobError(
                    "exact-UID Job cleanup is incomplete; resume the sealed cleanup intent"
                ) from None
    deadline = monotonic() + timeout_seconds
    while True:
        remaining_root = cluster.get_optional("jobs.batch", NAMESPACE, binding["job_name"])
        if remaining_root is not None:
            _bound_resource(
                remaining_root,
                label="created Job",
                name=binding["job_name"],
                uid=binding["job_uid"],
            )
        remaining_pods, remaining_workloads = _owned_inventory(cluster, binding)
        if remaining_root is None and not remaining_pods and not remaining_workloads:
            break
        if monotonic() >= deadline:
            raise CollectionJobError(
                "exact-UID Job cleanup is incomplete; resume the sealed cleanup intent"
            )
        sleep(min(5.0, max(0.0, deadline - monotonic())))
    current_config = cluster.get_optional("configmaps", NAMESPACE, binding["config_map_name"])
    if current_config is not None:
        _bound_resource(
            current_config,
            label="created ConfigMap",
            name=binding["config_map_name"],
            uid=binding["config_map_uid"],
        )
        try:
            cluster.delete_uid(
                "configmaps",
                NAMESPACE,
                binding["config_map_name"],
                binding["config_map_uid"],
            )
        except Exception:
            observed = cluster.get_optional("configmaps", NAMESPACE, binding["config_map_name"])
            if observed is not None:
                _bound_resource(
                    observed,
                    label="created ConfigMap",
                    name=binding["config_map_name"],
                    uid=binding["config_map_uid"],
                )
                raise CollectionJobError(
                    "exact-UID ConfigMap cleanup is incomplete; resume the sealed cleanup intent"
                ) from None
    if cluster.get_optional("configmaps", NAMESPACE, binding["config_map_name"]) is not None:
        raise CollectionJobError(
            "exact-UID ConfigMap cleanup is incomplete; resume the sealed cleanup intent"
        )
    body = {
        "schema": CLEANUP_RECEIPT_SCHEMA,
        "operation_authorization_sha256": binding["operation_authorization_sha256"],
        "job": {"name": binding["job_name"], "uid": binding["job_uid"]},
        "config_map": {
            "name": binding["config_map_name"],
            "uid": binding["config_map_uid"],
        },
        "terminal_condition": condition,
        "owned_pods_observed": len(pods),
        "owned_workloads_observed": len(workloads),
        "release_confirmed": True,
        "gpus": 0,
        "logs_traces_scores_or_credentials_read": False,
        "cleanup_intent_sha256": intent["sha256"],
    }
    receipt = {**body, "sha256": _canonical_digest(body)}
    _write_json_once(receipt_path, receipt)
    return receipt


class KubectlCluster:
    """Explicit-context Kubernetes adapter with one create and UID deletes."""

    def __init__(self, context: str, *, binary: str = "kubectl") -> None:
        if context != CONTEXT or not binary or binary.startswith("-"):
            raise CollectionJobError("exact collection Kubernetes context is required")
        self.context = context
        self.binary = binary

    def _run(
        self,
        namespace: str,
        arguments: list[str],
        value: dict[str, Any] | None = None,
        *,
        allow_empty: bool = False,
    ) -> dict[str, Any] | None:
        command = [
            self.binary,
            "--context",
            self.context,
            "--request-timeout=60s",
            "--namespace",
            namespace,
            *arguments,
        ]
        try:
            result = subprocess.run(
                command,
                input=(
                    json.dumps(value, sort_keys=True, separators=(",", ":"))
                    if value is not None
                    else None
                ),
                text=True,
                capture_output=True,
                timeout=75,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            raise CollectionJobError("kubectl transport failed; state is unknown") from None
        if result.returncode:
            raise CollectionJobError("kubectl operation failed; private output suppressed")
        if allow_empty and not result.stdout.strip():
            return None
        try:
            decoder = json.JSONDecoder()
            rows: list[dict[str, Any]] = []
            offset = 0
            while offset < len(result.stdout):
                while offset < len(result.stdout) and result.stdout[offset].isspace():
                    offset += 1
                if offset == len(result.stdout):
                    break
                row, offset = decoder.raw_decode(result.stdout, offset)
                rows.append(_mapping(row, "kubectl response"))
        except json.JSONDecodeError as error:
            raise CollectionJobError("kubectl returned invalid JSON") from error
        if not rows:
            raise CollectionJobError("kubectl returned no JSON")
        return rows[0] if len(rows) == 1 else {"apiVersion": "v1", "kind": "List", "items": rows}

    def list(
        self,
        resource: str,
        namespace: str,
        *,
        field_selector: str | None = None,
        label_selector: str | None = None,
    ) -> dict[str, Any]:
        if resource not in {
            "jobs.batch",
            "configmaps",
            "pods",
            "workloads.kueue.x-k8s.io",
        } or (field_selector is None) == (label_selector is None):
            raise CollectionJobError("Kubernetes inventory request is not narrowly scoped")
        selector = (
            f"--field-selector={field_selector}"
            if field_selector is not None
            else f"--selector={label_selector}"
        )
        result = self._run(namespace, ["get", resource, selector, "--output=json"])
        assert result is not None
        return result

    def get(self, resource: str, namespace: str, name: str) -> dict[str, Any]:
        if resource not in {"jobs.batch", "configmaps"} or KUBERNETES_NAME.fullmatch(name) is None:
            raise CollectionJobError("Kubernetes get is not narrowly scoped")
        result = self._run(namespace, ["get", resource, name, "--output=json"])
        assert result is not None
        return result

    def get_optional(self, resource: str, namespace: str, name: str) -> dict[str, Any] | None:
        if resource not in {"jobs.batch", "configmaps"} or KUBERNETES_NAME.fullmatch(name) is None:
            raise CollectionJobError("Kubernetes get is not narrowly scoped")
        return self._run(
            namespace,
            ["get", resource, name, "--ignore-not-found", "--output=json"],
            allow_empty=True,
        )

    def server_dry_run(self, namespace: str, bundle: dict[str, Any]) -> dict[str, Any]:
        result = self._run(
            namespace,
            ["create", "--dry-run=server", "--filename=-", "--output=json"],
            bundle,
        )
        assert result is not None
        return result

    def create_once(self, namespace: str, bundle: dict[str, Any]) -> dict[str, Any]:
        result = self._run(namespace, ["create", "--filename=-", "--output=json"], bundle)
        assert result is not None
        return result

    def delete_uid(self, resource: str, namespace: str, name: str, uid: str) -> dict[str, Any]:
        prefixes = {
            "jobs.batch": "/apis/batch/v1",
            "configmaps": "/api/v1",
        }
        plurals = {"jobs.batch": "jobs", "configmaps": "configmaps"}
        if (
            resource not in prefixes
            or KUBERNETES_NAME.fullmatch(name) is None
            or KUBERNETES_UID.fullmatch(uid) is None
        ):
            raise CollectionJobError("Kubernetes delete is not exact-UID scoped")
        body = {
            "apiVersion": "v1",
            "kind": "DeleteOptions",
            "propagationPolicy": "Foreground",
            "preconditions": {"uid": uid},
        }
        path = (
            f"{prefixes[resource]}/namespaces/{quote(namespace, safe='')}/"
            f"{plurals[resource]}/{quote(name, safe='')}"
        )
        result = self._run(namespace, ["delete", "--raw", path, "-f", "-"], body)
        assert result is not None
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--repo-root", type=Path, required=True)
    prepare.add_argument("--config", type=Path, required=True)
    prepare.add_argument("--task-selection", type=Path, required=True)
    prepare.add_argument("--authorization", type=Path, required=True)
    prepare.add_argument("--collection-packet", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--expected-source-commit", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("packet", type=Path)
    launch = commands.add_parser("launch")
    launch.add_argument("packet", type=Path)
    launch.add_argument("--context", required=True)
    launch.add_argument("--journal", type=Path, required=True)
    status = commands.add_parser("status")
    status.add_argument("packet", type=Path)
    status.add_argument("--context", required=True)
    status.add_argument("--journal", type=Path, required=True)
    cleanup = commands.add_parser("cleanup")
    cleanup.add_argument("packet", type=Path)
    cleanup.add_argument("--context", required=True)
    cleanup.add_argument("--journal", type=Path, required=True)
    cleanup.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_packet(
            repo_root=args.repo_root,
            config_path=args.config,
            task_selection_path=args.task_selection,
            authorization_path=args.authorization,
            collection_packet_path=args.collection_packet,
            output=args.output,
            expected_source_commit=args.expected_source_commit,
        )
    elif args.command == "validate":
        package = build_package(args.packet)
        result = {
            "valid": True,
            "job_name": package.packet.value["job_name"],
            "planned_cells": package.packet.value["operation"]["planned_cells"],
            "gpus": 0,
            "external_mutations": 0,
        }
    elif args.command == "launch":
        result = launch_once(
            args.packet,
            cluster=KubectlCluster(args.context),
            journal=args.journal,
        )
    elif args.command == "status":
        result = observe_status(
            args.packet,
            cluster=KubectlCluster(args.context),
            journal=args.journal,
        )
    else:
        result = cleanup_once(
            args.packet,
            cluster=KubectlCluster(args.context),
            journal=args.journal,
            receipt_path=args.receipt,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
