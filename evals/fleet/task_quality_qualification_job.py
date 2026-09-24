"""Seal, preview, create, observe, and clean one QA33 CPU-only Job.

Preparation is offline.  Creation requires a separate exact-packet authorization,
two equal server previews, a second exact-name absence check, and a durable local
intent.  A missing or ambiguous create response is never retried.
"""

from __future__ import annotations

import argparse
import base64
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

from evals.fleet import task_quality_qualification as qualification
from evals.fleet import visible_action_collection_job as proven_rails
from evals.fleet.evaluate import stable_job_preview

PACKET_SCHEMA = "cyber_task_quality_cpu_job_packet_v1"
AUTHORIZATION_SCHEMA = "cyber_task_quality_cpu_job_launch_authorization_v1"
CREATE_INTENT_SCHEMA = "cyber_task_quality_cpu_job_create_intent_v1"
CLEANUP_INTENT_SCHEMA = "cyber_task_quality_cpu_job_cleanup_intent_v1"
CLEANUP_RECEIPT_SCHEMA = "cyber_task_quality_cpu_job_cleanup_receipt_v1"
PARTIAL_CREATE_CLEANUP_SCHEMA = "cyber_task_quality_cpu_job_partial_create_cleanup_v1"
CONTEXT = "nebius-mk8s-fleetai-training-dev-e04p03enwk5c0va9tb"
NAMESPACE = "fleet-train-jobs"
QUEUE_NAME = "training-lq"
PRIORITY_CLASS = "c1"
DEADLINE_SECONDS = 1800
TTL_SECONDS_AFTER_FINISHED = 3600
PRIVATE_ROOT = "/mnt/sfs/jobs"
SFS_PVC = "sfs-shared"
FLEET_SECRET = "fleet-api"
FLEET_SECRET_KEY = "FLEET_API_KEY"
FAILURE_ALERT_ANNOTATION = "fleet.ai/failure-alerts"
PACKET_ANNOTATION = "cyber-post-train.fleet.ai/task-quality-packet-sha256"
CONTROLLER_IMAGE = (
    "ghcr.io/astral-sh/uv:python3.12-bookworm@"
    "sha256:9aa60c50016c0485636ab9a830246a6ef3399aa4a8bab3d17ef4a2358fba2ca7"
)
SOURCE_FILES = {
    "evals-init.py": "evals/__init__.py",
    "fleet-init.py": "evals/fleet/__init__.py",
    "opencode_self_hosted.py": "evals/fleet/opencode_self_hosted.py",
    "task_quality_qualification.py": "evals/fleet/task_quality_qualification.py",
    "task_quality_qualification_job_entry.py": (
        "evals/fleet/task_quality_qualification_job_entry.py"
    ),
}
CONFIG_MAP_MAX_BYTES = 900_000
KUBE_NAME = re.compile(r"[a-z0-9](?:[-a-z0-9]{0,61}[a-z0-9])?")
KUBE_UID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
GIT_ID = re.compile(r"[0-9a-f]{40}")
SHA256 = re.compile(r"sha256:[0-9a-f]{64}")
CPU_RESOURCE_KEYS = {"cpu", "memory", "ephemeral-storage"}
CANONICAL_REMOTE_URLS = {"https://github.com/fleet-ai/cyber-post-train.git"}
QA33_TASK_KEY = "cysec1-2-concur-gen_atoms-1_blackbox-703e845ecc4c1e020bb99309__blackbox_ctf_v1"
QA33_TASK_VERSION_ID = "ae236a07-128d-4556-ba2b-25a248818497"
QA33_TASK_ID = "ad41ad1d-e092-4839-8217-f02f5e0c6ba0"
QA33_LINEAGE_BINDING_SHA256 = (
    "sha256:5c5fd694b7fbb57b155206dc7cff8349c5f2f7b6db2f64ebc5f3b6c41926c56e"
)
QA33_CANDIDATE_SHA256 = "sha256:349ef5126d7cbdd431518a19bf173a9deff40c79395319e9171b168177e1f875"
RECONCILE_STABILITY_SECONDS = 10.0
RECONCILE_POLL_SECONDS = 1.0
PARTIAL_CREATE_EXTENSION_SCHEMA = "cyber_task_quality_cpu_job_partial_create_extension_v1"
EXACT_PACKET_CLEANUP = "exact_packet"
MUTATED_CREATE_RESPONSE_CLEANUP = "mutated_create_response_identity"
ALLOWED_CONTAINER_KEYS = {
    "name",
    "image",
    "imagePullPolicy",
    "command",
    "env",
    "resources",
    "securityContext",
    "volumeMounts",
    "terminationMessagePath",
    "terminationMessagePolicy",
}
CANARY_SOURCE_INPUTS = {
    "qualification_controller": "evals/fleet/task_quality_qualification.py",
    "qualification_job_launcher": "evals/fleet/task_quality_qualification_job.py",
    "qualification_job_entry": "evals/fleet/task_quality_qualification_job_entry.py",
}

BOOTSTRAP = r"""set -euo pipefail
umask 077
: "${FLEET_API_KEY:?FLEET_API_KEY is required}"
: "${QUALIFICATION_PRIVATE_ROOT:?QUALIFICATION_PRIVATE_ROOT is required}"
root=/workspace/cyber-post-train
mkdir -p "$root/evals/fleet"
install -m 0644 /bootstrap/evals-init.py "$root/evals/__init__.py"
install -m 0644 /bootstrap/fleet-init.py "$root/evals/fleet/__init__.py"
install -m 0644 /bootstrap/opencode_self_hosted.py "$root/evals/fleet/opencode_self_hosted.py"
install -m 0644 /bootstrap/task_quality_qualification.py \
  "$root/evals/fleet/task_quality_qualification.py"
install -m 0644 /bootstrap/task_quality_qualification_job_entry.py \
  "$root/evals/fleet/task_quality_qualification_job_entry.py"
install -m 0600 /private/plan.json /workspace/plan.json
install -m 0600 /bootstrap/source-attestation.json /workspace/source-attestation.json
cd "$root"
exec uv run --no-project --with httpx==0.28.1 \
  python -m evals.fleet.task_quality_qualification_job_entry \
  --plan /workspace/plan.json \
  --source-attestation /workspace/source-attestation.json \
  --private-root "$QUALIFICATION_PRIVATE_ROOT"
"""


class QualificationJobError(RuntimeError):
    """A packet, preview, create, or cleanup invariant failed."""


class Cluster(Protocol):
    def list(
        self,
        resource: str,
        namespace: str,
        *,
        field_selector: str | None = None,
        label_selector: str | None = None,
    ) -> dict[str, Any]: ...

    def get_optional(self, resource: str, namespace: str, name: str) -> dict[str, Any] | None: ...

    def server_dry_run(self, namespace: str, bundle: dict[str, Any]) -> dict[str, Any]: ...

    def create_once(self, namespace: str, bundle: dict[str, Any]) -> dict[str, Any]: ...

    def delete_uid(
        self,
        resource: str,
        namespace: str,
        name: str,
        uid: str,
        resource_version: str,
        *,
        confirmed_live: bool,
    ) -> dict[str, Any]: ...


class KubectlCluster(proven_rails.KubectlCluster):
    """The proven exact-UID adapter, pinned to the development context."""

    def __init__(self, context: str = CONTEXT, *, binary: str = "kubectl") -> None:
        if context != CONTEXT or not binary or binary.startswith("-"):
            raise QualificationJobError("exact development Kubernetes context is required")
        self.context = context
        self.binary = binary

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
            "secrets",
            "pods",
            "workloads.kueue.x-k8s.io",
        } or (field_selector is None) == (label_selector is None):
            raise QualificationJobError("Kubernetes inventory request is not narrowly scoped")
        selector = (
            f"--field-selector={field_selector}"
            if field_selector is not None
            else f"--selector={label_selector}"
        )
        result = self._run(namespace, ["get", resource, selector, "--output=json"])
        assert result is not None
        return result

    def get_optional(self, resource: str, namespace: str, name: str) -> dict[str, Any] | None:
        if (
            resource
            not in {
                "jobs.batch",
                "configmaps",
                "secrets",
                "pods",
                "workloads.kueue.x-k8s.io",
            }
            or KUBE_NAME.fullmatch(name) is None
        ):
            raise QualificationJobError("Kubernetes get is not narrowly scoped")
        return self._run(
            namespace,
            ["get", resource, name, "--ignore-not-found", "--output=json"],
            allow_empty=True,
        )

    def delete_uid(
        self,
        resource: str,
        namespace: str,
        name: str,
        uid: str,
        resource_version: str,
        *,
        confirmed_live: bool,
    ) -> dict[str, Any]:
        if resource != "secrets":
            return super().delete_uid(
                resource,
                namespace,
                name,
                uid,
                resource_version,
                confirmed_live=confirmed_live,
            )
        if (
            KUBE_NAME.fullmatch(name) is None
            or KUBE_UID.fullmatch(uid) is None
            or not resource_version
            or confirmed_live is not True
        ):
            raise QualificationJobError("Secret delete identity is invalid")
        body = {
            "apiVersion": "v1",
            "kind": "DeleteOptions",
            "propagationPolicy": "Foreground",
            "preconditions": {"uid": uid, "resourceVersion": resource_version},
        }
        path = f"/api/v1/namespaces/{quote(namespace, safe='')}/secrets/{quote(name, safe='')}"
        result = self._run(namespace, ["delete", "--raw", path, "-f", "-"], body)
        assert result is not None
        return result


@dataclass(frozen=True)
class Packet:
    path: Path
    value: dict[str, Any]
    files: dict[str, Path]

    @property
    def file_sha256(self) -> str:
        return file_sha256(self.path)


@dataclass(frozen=True)
class Package:
    packet: Packet
    config_map: dict[str, Any]
    secret: dict[str, Any]
    job: dict[str, Any]

    @property
    def bundle(self) -> dict[str, Any]:
        return {
            "apiVersion": "v1",
            "kind": "List",
            "items": [self.config_map, self.secret, self.job],
        }


def canonical_digest(value: object) -> str:
    return (
        "sha256:"
        + hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
    )


def file_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return "sha256:" + value.hexdigest()


def read_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise QualificationJobError(f"{label} must be a regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QualificationJobError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise QualificationJobError(f"{label} must be an object")
    return value


def write_once(path: Path, data: bytes) -> None:
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


def write_json_once(path: Path, value: dict[str, Any]) -> None:
    write_once(path, json.dumps(value, sort_keys=True, indent=2).encode() + b"\n")


def write_jsonl_once(path: Path, value: dict[str, Any]) -> None:
    write_once(
        path,
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n",
    )


def append_journal(path: Path, value: dict[str, Any]) -> None:
    descriptor = os.open(path, os.O_RDWR | os.O_APPEND)
    data = os.read(descriptor, os.fstat(descriptor).st_size)
    if data and not data.endswith(b"\n"):
        os.ftruncate(descriptor, data.rfind(b"\n") + 1)
        os.fsync(descriptor)
    with os.fdopen(descriptor, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_journal(path: Path) -> list[dict[str, Any]]:
    try:
        data = path.read_bytes()
        data.decode("utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise QualificationJobError("create journal is unreadable") from error
    complete = data if data.endswith(b"\n") else data[: data.rfind(b"\n") + 1]
    try:
        rows = [json.loads(line) for line in complete.splitlines() if line.strip()]
    except json.JSONDecodeError as error:
        raise QualificationJobError("create journal has a corrupt committed row") from error
    if any(not isinstance(row, dict) for row in rows):
        raise QualificationJobError("create journal rows are invalid")
    return rows


def git(repo: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise QualificationJobError("git source check failed") from error
    if result.returncode:
        raise QualificationJobError("git source check failed")
    return result.stdout.strip()


def sealed(value: dict[str, Any]) -> dict[str, Any]:
    return {**value, "sha256": canonical_digest(value)}


def assert_sealed(value: dict[str, Any], schema: str, label: str) -> None:
    if value.get("schema") != schema or value.get("sha256") != canonical_digest(
        {key: item for key, item in value.items() if key != "sha256"}
    ):
        raise QualificationJobError(f"{label} seal is invalid")


def safe_file(root: Path, binding: object, label: str) -> Path:
    if not isinstance(binding, dict) or set(binding) != {"path", "file_sha256"}:
        raise QualificationJobError(f"{label} binding is invalid")
    relative = PurePosixPath(str(binding.get("path", "")))
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise QualificationJobError(f"{label} path is unsafe")
    path = (root / Path(relative)).resolve(strict=True)
    if root.resolve() not in path.parents or path.is_symlink() or not path.is_file():
        raise QualificationJobError(f"{label} is not a safe regular file")
    if binding.get("file_sha256") != file_sha256(path):
        raise QualificationJobError(f"{label} bytes changed")
    return path


def merge_witness(repo: Path, source_commit: str, *, refresh: bool) -> dict[str, Any]:
    remote = git(repo, "remote", "get-url", "origin")
    if remote not in CANONICAL_REMOTE_URLS:
        raise QualificationJobError("origin is not the canonical cyber-post-train remote")
    if refresh:
        git(repo, "fetch", "--quiet", "origin", "main")
    observed_main = git(repo, "rev-parse", "refs/remotes/origin/main")
    if GIT_ID.fullmatch(source_commit) is None or GIT_ID.fullmatch(observed_main) is None:
        raise QualificationJobError("source or canonical-main commit is invalid")
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "merge-base", "--is-ancestor", source_commit, observed_main],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise QualificationJobError("source ancestry check failed") from error
    if result.returncode != 0:
        raise QualificationJobError("source commit is not merged to canonical main")
    return {
        "canonical_remote": remote,
        "source_commit": source_commit,
        "observed_main_commit": observed_main,
        "is_ancestor": True,
    }


def _copy(source: Path, destination: Path) -> dict[str, str]:
    if source.is_symlink() or not source.is_file():
        raise QualificationJobError("packet source must be a regular file")
    write_once(destination, source.read_bytes())
    return {
        "path": destination.relative_to(destination.parents[1]).as_posix(),
        "file_sha256": file_sha256(destination),
    }


def _validate_exact_plan(
    plan: dict[str, Any],
    canary: dict[str, Any],
    *,
    source_files: dict[str, Path] | None = None,
) -> None:
    qualification._sealed(plan, qualification.PLAN_SCHEMA, "qualification plan")  # noqa: SLF001
    assert_sealed(
        canary,
        "fleet_blackbox_qa33_zero_model_canary_review_packet_v1",
        "canary review packet",
    )
    tasks = plan.get("tasks")
    candidate = canary.get("candidate")
    if (
        not isinstance(tasks, list)
        or len(tasks) != 1
        or not isinstance(tasks[0], dict)
        or not isinstance(candidate, dict)
        or candidate.get("task_key") != QA33_TASK_KEY
        or candidate.get("task_version_id") != QA33_TASK_VERSION_ID
        or candidate.get("task_id") != QA33_TASK_ID
        or candidate.get("lineage_binding_sha256") != QA33_LINEAGE_BINDING_SHA256
        or candidate.get("qa_status") != "clean"
        or canonical_digest(candidate) != QA33_CANDIDATE_SHA256
        or tasks[0].get("task_key") != candidate.get("task_key")
        or tasks[0].get("task_version_id") != candidate.get("task_version_id")
        or plan.get("selection", {}).get("exact_task_identity")
        != {
            "task_key": candidate.get("task_key"),
            "task_version_id": candidate.get("task_version_id"),
        }
        or plan.get("execution", {}).get("concurrency") != 1
        or plan.get("execution", {}).get("model_calls") != 0
        or plan.get("execution", {}).get("external_mutations_authorized") is not True
    ):
        raise QualificationJobError("plan is not the exact one-cell model-free canary")
    if source_files is not None:
        inputs = canary.get("inputs")
        if not isinstance(inputs, dict):
            raise QualificationJobError("canary source bindings are missing")
        for label, expected_path in CANARY_SOURCE_INPUTS.items():
            binding = inputs.get(label)
            source_path = source_files.get(label)
            if not isinstance(binding, dict) or binding != {
                "path": expected_path,
                "file_sha256": file_sha256(source_path) if source_path else None,
            }:
                raise QualificationJobError("canary source binding differs from reviewed bytes")


def prepare_packet(
    *,
    repo_root: Path,
    plan_path: Path,
    canary_path: Path,
    output: Path,
    expected_source_commit: str,
) -> dict[str, Any]:
    """Create one private packet; perform no Kubernetes or Fleet mutation."""
    repo_root = repo_root.resolve(strict=True)
    if git(repo_root, "status", "--porcelain"):
        raise QualificationJobError("source worktree must be completely clean")
    commit = git(repo_root, "rev-parse", "HEAD")
    tree = git(repo_root, "rev-parse", "HEAD^{tree}")
    if commit != expected_source_commit or GIT_ID.fullmatch(tree) is None:
        raise QualificationJobError("source is not the exact reviewed commit")
    witness = merge_witness(repo_root, commit, refresh=False)
    plan = read_json(plan_path, "qualification plan")
    canary = read_json(canary_path, "canary review packet")
    _validate_exact_plan(
        plan,
        canary,
        source_files={
            label: repo_root / relative for label, relative in CANARY_SOURCE_INPUTS.items()
        },
    )
    source = plan.get("source")
    if (
        not isinstance(source, dict)
        or source.get("git_commit") != commit
        or source.get("git_tree") != tree
        or source.get("merged_to_origin_main") is not True
        or source.get("controller_file_sha256")
        != file_sha256(repo_root / "evals/fleet/task_quality_qualification.py")
    ):
        raise QualificationJobError("plan source differs from the exact reviewed checkout")

    try:
        output.mkdir(mode=0o700)
    except FileExistsError as error:
        raise QualificationJobError("packet directory already exists") from error
    source_dir = output / "source"
    input_dir = output / "inputs"
    source_dir.mkdir(mode=0o700)
    input_dir.mkdir(mode=0o700)
    source_entries: dict[str, dict[str, str]] = {}
    attested_files: dict[str, dict[str, str]] = {}
    for key, relative in SOURCE_FILES.items():
        binding = _copy(repo_root / relative, source_dir / key)
        source_entries[key] = binding
        attested_files[relative] = {"file_sha256": binding["file_sha256"]}
    plan_binding = _copy(plan_path, input_dir / "plan.json")
    canary_binding = _copy(canary_path, input_dir / "canary.json")
    attestation = sealed(
        {
            "schema": qualification.PACKAGED_SOURCE_SCHEMA,
            "plan_sha256": plan["sha256"],
            "source": source,
            "image": CONTROLLER_IMAGE,
            "dependency_pins": {"httpx": "0.28.1"},
            "files": attested_files,
            "merge_witness": witness,
            "exact_task_identity": plan["selection"]["exact_task_identity"],
        }
    )
    attestation_path = input_dir / "source-attestation.json"
    write_json_once(attestation_path, attestation)
    attestation_binding = {
        "path": attestation_path.relative_to(output).as_posix(),
        "file_sha256": file_sha256(attestation_path),
    }
    suffix = plan["sha256"].removeprefix("sha256:")[:8]
    job_name = f"chris-cyber-qa33-zm-{suffix}"
    config_map_name = f"{job_name}-code"
    secret_name = f"{job_name}-private"
    operation_root = f"{PRIVATE_ROOT}/{job_name}"
    body = {
        "schema": PACKET_SCHEMA,
        "context": CONTEXT,
        "namespace": NAMESPACE,
        "job_name": job_name,
        "config_map_name": config_map_name,
        "secret_name": secret_name,
        "source": {
            "git_commit": commit,
            "git_tree": tree,
            "launcher_file_sha256": file_sha256(Path(__file__)),
            "merge_witness": witness,
            "files": source_entries,
        },
        "inputs": {
            "plan": plan_binding,
            "canary": canary_binding,
            "source_attestation": attestation_binding,
        },
        "operation": {
            "plan_sha256": plan["sha256"],
            "canary_packet_sha256": canary["sha256"],
            "private_root": operation_root,
        },
        "runtime": {
            "image": CONTROLLER_IMAGE,
            "fleet_secret": {"name": FLEET_SECRET, "key": FLEET_SECRET_KEY},
            "sfs_pvc": SFS_PVC,
            "bootstrap": BOOTSTRAP,
        },
        "execution": {
            "active_deadline_seconds": DEADLINE_SECONDS,
            "backoff_limit": 0,
            "restart_policy": "Never",
            "priority_class_name": PRIORITY_CLASS,
            "queue_name": QUEUE_NAME,
            "gpu_requests": 0,
            "gpu_limits": 0,
            "server_preview_count": 2,
            "create_attempts_maximum": 1,
            "automatic_create_retry": False,
            "launch_authorized": False,
            "root_annotation": {FAILURE_ALERT_ANNOTATION: "off"},
        },
    }
    packet = sealed(body)
    packet_path = output / "launch-packet.json"
    write_json_once(packet_path, packet)
    return {
        "packet": str(packet_path),
        "packet_sha256": packet["sha256"],
        "packet_file_sha256": file_sha256(packet_path),
        "plan_sha256": plan["sha256"],
        "external_mutations": 0,
        "launch_authorized": False,
    }


def load_packet(path: Path) -> Packet:
    value = read_json(path, "launch packet")
    assert_sealed(value, PACKET_SCHEMA, "launch packet")
    if value.get("context") != CONTEXT or value.get("namespace") != NAMESPACE:
        raise QualificationJobError("packet targets the wrong cluster")
    if any(
        KUBE_NAME.fullmatch(str(value.get(field))) is None
        for field in ("job_name", "config_map_name", "secret_name")
    ):
        raise QualificationJobError("packet Kubernetes name is invalid")
    source = value.get("source")
    if (
        not isinstance(source, dict)
        or set(source)
        != {"git_commit", "git_tree", "launcher_file_sha256", "merge_witness", "files"}
        or source.get("launcher_file_sha256") != file_sha256(Path(__file__))
    ):
        raise QualificationJobError("launcher bytes changed after packet preparation")
    root = path.resolve().parent
    files: dict[str, Path] = {}
    entries = source.get("files")
    if not isinstance(entries, dict) or set(entries) != set(SOURCE_FILES):
        raise QualificationJobError("packet source file set is incomplete")
    for key, binding in entries.items():
        files[key] = safe_file(root, binding, f"source {key}")
    inputs = value.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != {"plan", "canary", "source_attestation"}:
        raise QualificationJobError("packet input set is incomplete")
    for key, binding in inputs.items():
        files[key] = safe_file(root, binding, f"input {key}")
    plan = read_json(files["plan"], "qualification plan")
    canary = read_json(files["canary"], "canary review packet")
    _validate_exact_plan(
        plan,
        canary,
        source_files={
            "qualification_controller": files["task_quality_qualification.py"],
            "qualification_job_launcher": Path(__file__),
            "qualification_job_entry": files["task_quality_qualification_job_entry.py"],
        },
    )
    operation = value.get("operation")
    if operation != {
        "plan_sha256": plan["sha256"],
        "canary_packet_sha256": canary["sha256"],
        "private_root": f"{PRIVATE_ROOT}/{value['job_name']}",
    }:
        raise QualificationJobError("packet operation identity changed")
    attestation = read_json(files["source_attestation"], "source attestation")
    qualification._sealed(  # noqa: SLF001
        attestation, qualification.PACKAGED_SOURCE_SCHEMA, "source attestation"
    )
    if (
        attestation.get("plan_sha256") != plan["sha256"]
        or attestation.get("source") != plan.get("source")
        or attestation.get("image") != CONTROLLER_IMAGE
        or attestation.get("dependency_pins") != {"httpx": "0.28.1"}
        or attestation.get("exact_task_identity")
        != plan.get("selection", {}).get("exact_task_identity")
    ):
        raise QualificationJobError("source attestation differs from the packet")
    attested_files = attestation.get("files")
    expected_attested = {
        relative: {"file_sha256": file_sha256(files[key])} for key, relative in SOURCE_FILES.items()
    }
    if attested_files != expected_attested:
        raise QualificationJobError("source attestation file closure changed")
    if attestation.get("merge_witness") != source.get("merge_witness"):
        raise QualificationJobError("source attestation merge witness changed")
    if value.get("runtime") != {
        "image": CONTROLLER_IMAGE,
        "fleet_secret": {"name": FLEET_SECRET, "key": FLEET_SECRET_KEY},
        "sfs_pvc": SFS_PVC,
        "bootstrap": BOOTSTRAP,
    }:
        raise QualificationJobError("packet runtime differs from the reviewed launcher")
    if value.get("execution") != {
        "active_deadline_seconds": DEADLINE_SECONDS,
        "backoff_limit": 0,
        "restart_policy": "Never",
        "priority_class_name": PRIORITY_CLASS,
        "queue_name": QUEUE_NAME,
        "gpu_requests": 0,
        "gpu_limits": 0,
        "server_preview_count": 2,
        "create_attempts_maximum": 1,
        "automatic_create_retry": False,
        "launch_authorized": False,
        "root_annotation": {FAILURE_ALERT_ANNOTATION: "off"},
    }:
        raise QualificationJobError("packet execution differs from the reviewed launcher")
    return Packet(path=path.resolve(), value=value, files=files)


def _config_map(packet: Packet) -> dict[str, Any]:
    def exact_text(path: Path) -> str:
        data = path.read_bytes()
        try:
            value = data.decode("utf-8")
        except UnicodeDecodeError as error:
            raise QualificationJobError("ConfigMap source is not UTF-8") from error
        if value.encode("utf-8") != data:
            raise QualificationJobError("ConfigMap source text is not byte-stable")
        return value

    data = {
        **{key: exact_text(path) for key, path in packet.files.items() if key in SOURCE_FILES},
        "source-attestation.json": exact_text(packet.files["source_attestation"]),
    }
    value = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": packet.value["config_map_name"],
            "namespace": NAMESPACE,
            "annotations": {PACKET_ANNOTATION: packet.value["sha256"]},
        },
        "immutable": True,
        "data": data,
    }
    if (
        len(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())
        > CONFIG_MAP_MAX_BYTES
    ):
        raise QualificationJobError("source ConfigMap is too large")
    return value


def _secret(packet: Packet) -> dict[str, Any]:
    return {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {
            "name": packet.value["secret_name"],
            "namespace": NAMESPACE,
            "annotations": {PACKET_ANNOTATION: packet.value["sha256"]},
        },
        "immutable": True,
        "type": "Opaque",
        "data": {"plan.json": base64.b64encode(packet.files["plan"].read_bytes()).decode("ascii")},
    }


def _job(packet: Packet) -> dict[str, Any]:
    runtime = packet.value["runtime"]
    name = packet.value["job_name"]
    annotations = {
        FAILURE_ALERT_ANNOTATION: "off",
        PACKET_ANNOTATION: packet.value["sha256"],
    }
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": name,
            "namespace": NAMESPACE,
            "annotations": annotations,
            "labels": {"kueue.x-k8s.io/queue-name": QUEUE_NAME},
        },
        "spec": {
            "activeDeadlineSeconds": DEADLINE_SECONDS,
            "backoffLimit": 0,
            "completions": 1,
            "parallelism": 1,
            "ttlSecondsAfterFinished": TTL_SECONDS_AFTER_FINISHED,
            "template": {
                "metadata": {
                    "annotations": {PACKET_ANNOTATION: packet.value["sha256"]},
                    "labels": {"app.kubernetes.io/name": name},
                },
                "spec": {
                    "automountServiceAccountToken": False,
                    "priorityClassName": PRIORITY_CLASS,
                    "restartPolicy": "Never",
                    "terminationGracePeriodSeconds": 300,
                    "containers": [
                        {
                            "name": "qualification",
                            "image": runtime["image"],
                            "imagePullPolicy": "IfNotPresent",
                            "command": ["/bin/bash", "-ceu", runtime["bootstrap"]],
                            "env": [
                                {
                                    "name": "FLEET_API_KEY",
                                    "valueFrom": {
                                        "secretKeyRef": runtime["fleet_secret"],
                                    },
                                },
                                {
                                    "name": "QUALIFICATION_PRIVATE_ROOT",
                                    "value": packet.value["operation"]["private_root"],
                                },
                            ],
                            "resources": {
                                "requests": {"cpu": "1", "memory": "1Gi"},
                                "limits": {"cpu": "2", "memory": "2Gi"},
                            },
                            "securityContext": {
                                "allowPrivilegeEscalation": False,
                                "capabilities": {"drop": ["ALL"]},
                            },
                            "volumeMounts": [
                                {"name": "source", "mountPath": "/bootstrap", "readOnly": True},
                                {"name": "private", "mountPath": "/private", "readOnly": True},
                                {"name": "sfs", "mountPath": "/mnt/sfs"},
                            ],
                        }
                    ],
                    "volumes": [
                        {
                            "name": "source",
                            "configMap": {"name": packet.value["config_map_name"]},
                        },
                        {
                            "name": "private",
                            "secret": {
                                "secretName": packet.value["secret_name"],
                                "defaultMode": 256,
                            },
                        },
                        {"name": "sfs", "persistentVolumeClaim": {"claimName": runtime["sfs_pvc"]}},
                    ],
                },
            },
        },
    }


def _containers(pod: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for field in ("initContainers", "containers", "ephemeralContainers"):
        rows = pod.get(field, [])
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise QualificationJobError("container inventory is invalid")
        result.extend(rows)
    return result


def _assert_zero_accelerators(pod: dict[str, Any]) -> None:
    regular = pod.get("containers")
    if (
        not isinstance(regular, list)
        or len(regular) != 1
        or pod.get("initContainers")
        or pod.get("ephemeralContainers")
        or pod.get("automountServiceAccountToken") is not False
        or any(
            pod.get(field) is True
            for field in ("hostNetwork", "hostPID", "hostIPC", "shareProcessNamespace")
        )
    ):
        raise QualificationJobError("CPU-only Job Pod shape is unsafe")
    if pod.get("resourceClaims"):
        raise QualificationJobError("CPU-only Job cannot use resource claims")
    overhead = pod.get("overhead", {})
    if not isinstance(overhead, dict) or any(
        key not in CPU_RESOURCE_KEYS and not str(key).startswith("hugepages-") for key in overhead
    ):
        raise QualificationJobError("CPU-only Job has accelerator or extended-resource overhead")
    for container in _containers(pod):
        if set(container) - ALLOWED_CONTAINER_KEYS:
            raise QualificationJobError("CPU-only Job container has unreviewed fields")
        if (
            container.get("terminationMessagePath", "/dev/termination-log")
            != "/dev/termination-log"
        ):
            raise QualificationJobError("CPU-only Job termination path changed")
        if container.get("terminationMessagePolicy", "File") != "File":
            raise QualificationJobError("CPU-only Job termination policy changed")
        if container.get("securityContext") != {
            "allowPrivilegeEscalation": False,
            "capabilities": {"drop": ["ALL"]},
        }:
            raise QualificationJobError("CPU-only Job container security context is unsafe")
        resources = container.get("resources", {})
        if not isinstance(resources, dict) or set(resources) - {"requests", "limits", "claims"}:
            raise QualificationJobError("container resources are invalid")
        for field in ("requests", "limits"):
            quantities = resources.get(field, {})
            if not isinstance(quantities, dict) or any(
                key not in CPU_RESOURCE_KEYS and not str(key).startswith("hugepages-")
                for key in quantities
            ):
                raise QualificationJobError("CPU-only Job requests an accelerator")
        if resources.get("claims"):
            raise QualificationJobError("CPU-only container cannot use resource claims")


def _assert_exact_job(value: dict[str, Any], packet: Packet) -> None:
    metadata = value.get("metadata")
    spec = value.get("spec")
    if not isinstance(metadata, dict) or not isinstance(spec, dict):
        raise QualificationJobError("Job metadata or spec is invalid")
    pod = spec.get("template", {}).get("spec")
    if (
        value.get("apiVersion") != "batch/v1"
        or value.get("kind") != "Job"
        or metadata.get("name") != packet.value["job_name"]
        or metadata.get("namespace") != NAMESPACE
        or metadata.get("annotations", {}).get(FAILURE_ALERT_ANNOTATION) != "off"
        or metadata.get("annotations", {}).get(PACKET_ANNOTATION) != packet.value["sha256"]
        or metadata.get("labels", {}).get("kueue.x-k8s.io/queue-name") != QUEUE_NAME
        or type(spec.get("activeDeadlineSeconds")) is not int
        or not 1 <= spec["activeDeadlineSeconds"] <= DEADLINE_SECONDS
        or spec.get("backoffLimit") != 0
        or spec.get("completions") != 1
        or spec.get("parallelism") != 1
        or spec.get("ttlSecondsAfterFinished") != TTL_SECONDS_AFTER_FINISHED
        or not isinstance(pod, dict)
        or pod.get("priorityClassName") != PRIORITY_CLASS
        or pod.get("restartPolicy") != "Never"
    ):
        raise QualificationJobError("rendered root Job violates the sealed safety contract")
    _assert_zero_accelerators(pod)


def build_package(packet_path: Path) -> Package:
    packet = load_packet(packet_path)
    config_map = _config_map(packet)
    secret = _secret(packet)
    job = _job(packet)
    _assert_exact_job(job, packet)
    return Package(packet=packet, config_map=config_map, secret=secret, job=job)


def _objects(value: dict[str, Any]) -> list[dict[str, Any]]:
    rows = value.get("items") if value.get("kind") == "List" else [value]
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise QualificationJobError("Kubernetes response object list is invalid")
    return rows


def _response_object(
    value: dict[str, Any], *, api_version: str, kind: str, namespace: str, name: str
) -> dict[str, Any]:
    matches = [
        row
        for row in _objects(value)
        if row.get("apiVersion") == api_version
        and row.get("kind") == kind
        and row.get("metadata", {}).get("namespace") == namespace
        and row.get("metadata", {}).get("name") == name
    ]
    if len(matches) != 1:
        raise QualificationJobError("Kubernetes response lacks one exact object")
    return matches[0]


def _contains(actual: object, expected: object) -> bool:
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and _contains(actual[key], item) for key, item in expected.items()
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
    )
    config_map = _response_object(
        value,
        api_version="v1",
        kind="ConfigMap",
        namespace=NAMESPACE,
        name=package.packet.value["config_map_name"],
    )
    secret = _response_object(
        value,
        api_version="v1",
        kind="Secret",
        namespace=NAMESPACE,
        name=package.packet.value["secret_name"],
    )
    if len(_objects(value)) != 3:
        raise QualificationJobError("server preview contains unexpected objects")
    _assert_exact_job(job, package.packet)
    try:
        stable_job = stable_job_preview(job)
        expected_job = stable_job_preview(package.job)
    except ValueError as error:
        raise QualificationJobError("server preview Job has no stable form") from error
    if not _contains(stable_job, expected_job):
        raise QualificationJobError("server preview differs from the sealed Job bundle")
    stable_config = {
        "apiVersion": config_map.get("apiVersion"),
        "kind": config_map.get("kind"),
        "metadata": {
            "name": config_map.get("metadata", {}).get("name"),
            "namespace": config_map.get("metadata", {}).get("namespace"),
            "annotations": config_map.get("metadata", {}).get("annotations"),
        },
        "immutable": config_map.get("immutable"),
        "data": config_map.get("data"),
    }
    expected_config = {
        **package.config_map,
        "metadata": {
            "name": package.config_map["metadata"]["name"],
            "namespace": package.config_map["metadata"]["namespace"],
            "annotations": package.config_map["metadata"]["annotations"],
        },
    }
    if stable_config != expected_config:
        raise QualificationJobError("server preview changed source ConfigMap bytes")
    stable_secret = {
        "apiVersion": secret.get("apiVersion"),
        "kind": secret.get("kind"),
        "metadata": {
            "name": secret.get("metadata", {}).get("name"),
            "namespace": secret.get("metadata", {}).get("namespace"),
            "annotations": secret.get("metadata", {}).get("annotations"),
        },
        "immutable": secret.get("immutable"),
        "type": secret.get("type"),
        "data": secret.get("data"),
    }
    expected_secret = {
        **package.secret,
        "metadata": {
            "name": package.secret["metadata"]["name"],
            "namespace": package.secret["metadata"]["namespace"],
            "annotations": package.secret["metadata"]["annotations"],
        },
    }
    if stable_secret != expected_secret:
        raise QualificationJobError("server preview changed private plan Secret bytes")
    return canonical_digest(
        {"job": stable_job, "config_map": stable_config, "secret": stable_secret}
    )


def write_launch_authorization(
    packet_path: Path, output: Path, *, root_authorization_id: str
) -> dict[str, Any]:
    """Seal an explicit root authorization after it has actually been granted."""
    package = build_package(packet_path)
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{7,127}", root_authorization_id):
        raise QualificationJobError("root authorization id is invalid")
    authorization_path = output.resolve()
    create_journal_path = authorization_path.parent / "create.jsonl"
    value = sealed(
        {
            "schema": AUTHORIZATION_SCHEMA,
            "packet_sha256": package.packet.value["sha256"],
            "packet_file_sha256": package.packet.file_sha256,
            "context": CONTEXT,
            "namespace": NAMESPACE,
            "job_name": package.packet.value["job_name"],
            "config_map_name": package.packet.value["config_map_name"],
            "secret_name": package.packet.value["secret_name"],
            "root_authorization_id": root_authorization_id,
            "authorization_path": str(authorization_path),
            "create_journal_path": str(create_journal_path),
            "one_create_authorized": True,
        }
    )
    write_json_once(output, value)
    return {"authorization": str(output), "sha256": value["sha256"], "external_mutations": 0}


def load_authorization(path: Path, package: Package) -> dict[str, Any]:
    value = read_json(path, "launch authorization")
    assert_sealed(value, AUTHORIZATION_SCHEMA, "launch authorization")
    if set(value) != {
        "schema",
        "packet_sha256",
        "packet_file_sha256",
        "context",
        "namespace",
        "job_name",
        "config_map_name",
        "secret_name",
        "root_authorization_id",
        "authorization_path",
        "create_journal_path",
        "one_create_authorized",
        "sha256",
    }:
        raise QualificationJobError("launch authorization fields are invalid")
    expected = {
        "packet_sha256": package.packet.value["sha256"],
        "packet_file_sha256": package.packet.file_sha256,
        "context": CONTEXT,
        "namespace": NAMESPACE,
        "job_name": package.packet.value["job_name"],
        "config_map_name": package.packet.value["config_map_name"],
        "secret_name": package.packet.value["secret_name"],
        "authorization_path": str(path.resolve()),
        "create_journal_path": str(path.resolve().parent / "create.jsonl"),
        "one_create_authorized": True,
    }
    if any(value.get(key) != item for key, item in expected.items()) or not re.fullmatch(
        r"[a-z0-9][a-z0-9-]{7,127}", str(value.get("root_authorization_id", ""))
    ):
        raise QualificationJobError("launch authorization differs from the exact packet")
    return value


def _items(value: dict[str, Any], label: str) -> list[dict[str, Any]]:
    rows = value.get("items")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise QualificationJobError(f"{label} inventory is invalid")
    return rows


def exact_name_census(package: Package, cluster: Cluster) -> dict[str, int]:
    job_name = package.packet.value["job_name"]
    config_name = package.packet.value["config_map_name"]
    secret_name = package.packet.value["secret_name"]
    checks = {
        "jobs": cluster.list("jobs.batch", NAMESPACE, field_selector=f"metadata.name={job_name}"),
        "config_maps": cluster.list(
            "configmaps", NAMESPACE, field_selector=f"metadata.name={config_name}"
        ),
        "secrets": cluster.list(
            "secrets", NAMESPACE, field_selector=f"metadata.name={secret_name}"
        ),
        "pods": cluster.list(
            "pods", NAMESPACE, label_selector=f"batch.kubernetes.io/job-name={job_name}"
        ),
    }
    counts = {name: len(_items(value, name)) for name, value in checks.items()}
    if any(counts.values()):
        raise QualificationJobError(
            "exact Job, ConfigMap, Secret, or owned Pod identity already exists"
        )
    return counts


def _created_name_observation(package: Package, cluster: Cluster) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for label, resource, name in (
        ("job", "jobs.batch", package.packet.value["job_name"]),
        ("config_map", "configmaps", package.packet.value["config_map_name"]),
        ("secret", "secrets", package.packet.value["secret_name"]),
    ):
        try:
            value = cluster.get_optional(resource, NAMESPACE, name)
        except Exception:
            result[label] = {"observation": "unavailable"}
            continue
        metadata = value.get("metadata", {}) if isinstance(value, dict) else {}
        uid = metadata.get("uid") if isinstance(metadata, dict) else None
        resource_version = metadata.get("resourceVersion") if isinstance(metadata, dict) else None
        result[label] = {
            "present": value is not None,
            "uid": uid if KUBE_UID.fullmatch(str(uid)) else None,
            "resource_version": resource_version if isinstance(resource_version, str) else None,
        }
    return result


def _uncertain_observation(package: Package, cluster: Cluster) -> dict[str, Any]:
    objects = {
        "job": cluster.get_optional("jobs.batch", NAMESPACE, package.packet.value["job_name"]),
        "config_map": cluster.get_optional(
            "configmaps", NAMESPACE, package.packet.value["config_map_name"]
        ),
        "secret": cluster.get_optional("secrets", NAMESPACE, package.packet.value["secret_name"]),
    }
    children: list[dict[str, str]] = []
    job_name = package.packet.value["job_name"]
    for resource, selector in (("pods", f"batch.kubernetes.io/job-name={job_name}"),):
        for value in _items(
            cluster.list(resource, NAMESPACE, label_selector=selector),
            f"uncertain {resource}",
        ):
            metadata = _metadata(value, f"uncertain {resource}")
            name = metadata.get("name")
            uid = metadata.get("uid")
            if KUBE_NAME.fullmatch(str(name)) is None or KUBE_UID.fullmatch(str(uid)) is None:
                raise QualificationJobError("uncertain child identity is invalid")
            children.append({"resource": resource, "name": str(name), "uid": str(uid)})
    return {**objects, "children": sorted(children, key=lambda row: tuple(row.values()))}


def _observation_signature(value: dict[str, Any]) -> str:
    objects: dict[str, Any] = {}
    for label in ("job", "config_map", "secret"):
        item = value[label]
        if item is None:
            objects[label] = None
        else:
            metadata = _metadata(item, f"uncertain {label}")
            uid, _ = _identity(metadata, f"uncertain {label}")
            objects[label] = {"name": metadata.get("name"), "uid": uid}
    return canonical_digest({"objects": objects, "children": value["children"]})


def _stable_uncertain_observation(
    package: Package,
    cluster: Cluster,
    *,
    deadline: float,
    monotonic: Callable[[], float],
    sleep: Callable[[float], None],
) -> dict[str, Any]:
    previous: str | None = None
    stable_since: float | None = None
    while True:
        value = _uncertain_observation(package, cluster)
        signature = _observation_signature(value)
        now = monotonic()
        if signature != previous:
            previous = signature
            stable_since = now
        elif stable_since is not None and now - stable_since >= RECONCILE_STABILITY_SECONDS:
            return value
        if now >= deadline:
            raise QualificationJobError("uncertain create state did not become stably observable")
        sleep(min(RECONCILE_POLL_SECONDS, max(0.0, deadline - now)))


def _metadata(value: dict[str, Any], label: str) -> dict[str, Any]:
    metadata = value.get("metadata")
    if not isinstance(metadata, dict):
        raise QualificationJobError(f"{label} metadata is invalid")
    return metadata


def _identity(metadata: dict[str, Any], label: str) -> tuple[str, str]:
    uid = metadata.get("uid")
    resource_version = metadata.get("resourceVersion")
    if (
        KUBE_UID.fullmatch(str(uid)) is None
        or not isinstance(resource_version, str)
        or not resource_version
    ):
        raise QualificationJobError(f"{label} UID/resourceVersion is invalid")
    return str(uid), resource_version


def _create_response_bindings(package: Package, response: dict[str, Any]) -> list[dict[str, str]]:
    """Bind exact-name objects from the single create response without trusting their content."""
    if len(_objects(response)) != 3:
        raise QualificationJobError("create response contains unexpected objects")
    result: list[dict[str, str]] = []
    for label, resource, api_version, kind, name in (
        ("created Job", "jobs.batch", "batch/v1", "Job", package.packet.value["job_name"]),
        (
            "created ConfigMap",
            "configmaps",
            "v1",
            "ConfigMap",
            package.packet.value["config_map_name"],
        ),
        ("created Secret", "secrets", "v1", "Secret", package.packet.value["secret_name"]),
    ):
        value = _response_object(
            response,
            api_version=api_version,
            kind=kind,
            namespace=NAMESPACE,
            name=name,
        )
        uid, resource_version = _identity(_metadata(value, label), label)
        result.append(
            {
                "resource": resource,
                "name": name,
                "uid": uid,
                "resource_version": resource_version,
            }
        )
    return result


def _mutated_live_bindings(package: Package, response: dict[str, Any]) -> list[dict[str, str]]:
    """Bind a stable live mismatch only when every object retains the packet marker."""
    bindings = _create_response_bindings(package, response)
    for value in _objects(response):
        annotations = _metadata(value, "mutated live object").get("annotations")
        if (
            not isinstance(annotations, dict)
            or annotations.get(PACKET_ANNOTATION) != package.packet.value["sha256"]
        ):
            raise QualificationJobError("mismatched live object lacks the exact packet annotation")
    return bindings


def launch_once(
    packet_path: Path,
    *,
    authorization_path: Path,
    repo_root: Path,
    cluster: Cluster,
    journal: Path,
    cleanup_timeout_seconds: float = 600,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    if journal.exists() or journal.is_symlink() or not journal.parent.is_dir():
        raise QualificationJobError("create journal exists or its parent is unavailable")
    package = build_package(packet_path)
    authorization = load_authorization(authorization_path, package)
    if journal.resolve() != Path(authorization["create_journal_path"]):
        raise QualificationJobError("create journal is not the authorization-bound journal")
    source_commit = package.packet.value["source"]["git_commit"]
    live_witness = merge_witness(repo_root.resolve(strict=True), source_commit, refresh=True)
    first_census = exact_name_census(package, cluster)
    first_preview = validate_server_preview(
        cluster.server_dry_run(NAMESPACE, package.bundle), package
    )
    second_preview = validate_server_preview(
        cluster.server_dry_run(NAMESPACE, package.bundle), package
    )
    if first_preview != second_preview:
        raise QualificationJobError("server previews differ")
    final_census = exact_name_census(package, cluster)
    intent = sealed(
        {
            "schema": CREATE_INTENT_SCHEMA,
            "state": "KUBERNETES_CREATE_INTENT_DO_NOT_RETRY",
            "packet_sha256": package.packet.value["sha256"],
            "packet_file_sha256": package.packet.file_sha256,
            "authorization_sha256": authorization["sha256"],
            "plan_sha256": package.packet.value["operation"]["plan_sha256"],
            "context": CONTEXT,
            "namespace": NAMESPACE,
            "job_name": package.packet.value["job_name"],
            "config_map_name": package.packet.value["config_map_name"],
            "secret_name": package.packet.value["secret_name"],
            "server_preview_sha256": first_preview,
            "first_exact_name_census": first_census,
            "final_exact_name_census": final_census,
            "fresh_merge_witness": live_witness,
        }
    )
    write_jsonl_once(journal, intent)
    mutated_create_released = False
    try:
        response = cluster.create_once(NAMESPACE, package.bundle)
        response_bindings = _create_response_bindings(package, response)
        try:
            created_preview = validate_server_preview(response, package)
        except QualificationJobError:
            created_preview = None
        if created_preview != first_preview:
            cleanup_intent = sealed(
                {
                    "schema": PARTIAL_CREATE_CLEANUP_SCHEMA,
                    "state": "PARTIAL_CREATE_CLEANUP_INTENT",
                    "packet_sha256": package.packet.value["sha256"],
                    "objects": response_bindings,
                    "validation_mode": MUTATED_CREATE_RESPONSE_CLEANUP,
                    "never_retry_create": True,
                }
            )
            append_journal(journal, cleanup_intent)
            _finish_partial_create_cleanup(
                package,
                cluster=cluster,
                journal=journal,
                intent=cleanup_intent,
                timeout_seconds=cleanup_timeout_seconds,
                monotonic=monotonic,
                sleep=sleep,
            )
            mutated_create_released = True
            raise QualificationJobError(
                "created objects differed from the reviewed server preview and were released"
            )
        job = _response_object(
            response,
            api_version="batch/v1",
            kind="Job",
            namespace=NAMESPACE,
            name=package.packet.value["job_name"],
        )
        config_map = _response_object(
            response,
            api_version="v1",
            kind="ConfigMap",
            namespace=NAMESPACE,
            name=package.packet.value["config_map_name"],
        )
        secret = _response_object(
            response,
            api_version="v1",
            kind="Secret",
            namespace=NAMESPACE,
            name=package.packet.value["secret_name"],
        )
        _assert_exact_job(job, package.packet)
        job_uid, job_rv = _identity(_metadata(job, "created Job"), "created Job")
        config_uid, config_rv = _identity(
            _metadata(config_map, "created ConfigMap"), "created ConfigMap"
        )
        secret_uid, secret_rv = _identity(_metadata(secret, "created Secret"), "created Secret")
        result = {
            "state": "KUBERNETES_CREATE_RESPONSE",
            "submitted": True,
            "gpus": 0,
            "job_name": package.packet.value["job_name"],
            "job_uid": job_uid,
            "job_resource_version": job_rv,
            "config_map_name": package.packet.value["config_map_name"],
            "config_map_uid": config_uid,
            "config_map_resource_version": config_rv,
            "secret_name": package.packet.value["secret_name"],
            "secret_uid": secret_uid,
            "secret_resource_version": secret_rv,
            "packet_sha256": package.packet.value["sha256"],
            "plan_sha256": package.packet.value["operation"]["plan_sha256"],
            "server_preview_sha256": first_preview,
        }
    except Exception as error:
        if mutated_create_released:
            raise
        append_journal(
            journal,
            {
                "state": "KUBERNETES_CREATE_RESPONSE_UNCERTAIN_DO_NOT_RETRY",
                "observed_exact_names": _created_name_observation(package, cluster),
            },
        )
        if isinstance(error, QualificationJobError):
            raise
        raise QualificationJobError(
            "create response uncertain; reconcile and never retry"
        ) from None
    append_journal(journal, result)
    return result


def reconcile_uncertain(
    packet_path: Path,
    *,
    cluster: Cluster,
    journal: Path,
    confirm_partial_cleanup: bool = False,
    timeout_seconds: float = 600,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Bind or release a lost create response; never issue another create."""
    package = build_package(packet_path)
    rows = read_journal(journal)
    states = [row.get("state") for row in rows]
    create_intents = [
        row for row in rows if row.get("state") == "KUBERNETES_CREATE_INTENT_DO_NOT_RETRY"
    ]
    partial_intents = [row for row in rows if row.get("state") == "PARTIAL_CREATE_CLEANUP_INTENT"]
    if (
        len(create_intents) != 1
        or states.count("KUBERNETES_CREATE_RESPONSE_UNCERTAIN_DO_NOT_RETRY") not in {0, 1}
        or any(
            state
            in {
                "KUBERNETES_CREATE_RESPONSE",
                "KUBERNETES_CREATE_RECONCILED",
                "KUBERNETES_CREATE_RECONCILED_ABSENT",
                "KUBERNETES_CREATE_RECONCILED_PARTIAL_RELEASED",
            }
            for state in states
        )
        or len(partial_intents) > 1
    ):
        raise QualificationJobError("journal is not one unreconciled create response")
    create_intent = create_intents[0]
    assert_sealed(create_intent, CREATE_INTENT_SCHEMA, "create intent")
    if (
        create_intent.get("packet_sha256") != package.packet.value["sha256"]
        or create_intent.get("plan_sha256") != package.packet.value["operation"]["plan_sha256"]
        or create_intent.get("job_name") != package.packet.value["job_name"]
        or create_intent.get("config_map_name") != package.packet.value["config_map_name"]
        or create_intent.get("secret_name") != package.packet.value["secret_name"]
        or SHA256.fullmatch(str(create_intent.get("server_preview_sha256"))) is None
    ):
        raise QualificationJobError("create intent differs from the exact packet")
    if partial_intents:
        assert_sealed(
            partial_intents[0], PARTIAL_CREATE_CLEANUP_SCHEMA, "partial create cleanup intent"
        )
        return _finish_partial_create_cleanup(
            package,
            cluster=cluster,
            journal=journal,
            intent=partial_intents[0],
            timeout_seconds=timeout_seconds,
            monotonic=monotonic,
            sleep=sleep,
        )
    if timeout_seconds <= RECONCILE_STABILITY_SECONDS:
        raise QualificationJobError("reconcile timeout cannot prove stable create state")
    observation = _stable_uncertain_observation(
        package,
        cluster,
        deadline=monotonic() + timeout_seconds,
        monotonic=monotonic,
        sleep=sleep,
    )
    job_value = observation["job"]
    config_value = observation["config_map"]
    secret_value = observation["secret"]
    present = [job_value is not None, config_value is not None, secret_value is not None]
    if not any(present) and not observation["children"]:
        result = {
            "state": "KUBERNETES_CREATE_RECONCILED_ABSENT",
            "submitted": "unknown",
            "kubernetes_release_confirmed": False,
            "environment_cleanup_proven": False,
            "release_confirmed": False,
            "possible_external_environment_leak": True,
            "gpus": 0,
        }
        append_journal(journal, result)
        return result
    if observation["children"] and job_value is None:
        raise QualificationJobError("uncertain create has children without a bindable exact Job")
    if not all(present):
        if confirm_partial_cleanup is not True:
            raise QualificationJobError(
                "partial List create requires explicit exact-UID cleanup confirmation"
            )
        objects = _validate_reconciled_objects(
            package,
            job=job_value,
            config_map=config_value,
            secret=secret_value,
        )
        intent = sealed(
            {
                "schema": PARTIAL_CREATE_CLEANUP_SCHEMA,
                "state": "PARTIAL_CREATE_CLEANUP_INTENT",
                "packet_sha256": package.packet.value["sha256"],
                "objects": objects,
                "validation_mode": EXACT_PACKET_CLEANUP,
                "never_retry_create": True,
            }
        )
        append_journal(journal, intent)
        return _finish_partial_create_cleanup(
            package,
            cluster=cluster,
            journal=journal,
            intent=intent,
            timeout_seconds=timeout_seconds,
            monotonic=monotonic,
            sleep=sleep,
        )

    assert job_value is not None and config_value is not None and secret_value is not None
    live_bundle = {
        "apiVersion": "v1",
        "kind": "List",
        "items": [config_value, secret_value, job_value],
    }
    try:
        reconciled_preview = validate_server_preview(live_bundle, package)
    except QualificationJobError:
        reconciled_preview = None
    if reconciled_preview != create_intent["server_preview_sha256"]:
        if confirm_partial_cleanup is not True:
            raise QualificationJobError(
                "reconciled objects differ from the reviewed server preview; "
                "explicit exact-UID cleanup confirmation is required"
            )
        intent = sealed(
            {
                "schema": PARTIAL_CREATE_CLEANUP_SCHEMA,
                "state": "PARTIAL_CREATE_CLEANUP_INTENT",
                "packet_sha256": package.packet.value["sha256"],
                "objects": _mutated_live_bindings(package, live_bundle),
                "validation_mode": MUTATED_CREATE_RESPONSE_CLEANUP,
                "never_retry_create": True,
            }
        )
        append_journal(journal, intent)
        return _finish_partial_create_cleanup(
            package,
            cluster=cluster,
            journal=journal,
            intent=intent,
            timeout_seconds=timeout_seconds,
            monotonic=monotonic,
            sleep=sleep,
        )
    _validate_reconciled_objects(
        package, job=job_value, config_map=config_value, secret=secret_value
    )
    job_uid, job_rv = _identity(_metadata(job_value, "reconciled Job"), "reconciled Job")
    config_uid, config_rv = _identity(
        _metadata(config_value, "reconciled ConfigMap"), "reconciled ConfigMap"
    )
    secret_uid, secret_rv = _identity(
        _metadata(secret_value, "reconciled Secret"), "reconciled Secret"
    )
    result = {
        "state": "KUBERNETES_CREATE_RECONCILED",
        "submitted": True,
        "gpus": 0,
        "job_name": package.packet.value["job_name"],
        "job_uid": job_uid,
        "job_resource_version": job_rv,
        "config_map_name": package.packet.value["config_map_name"],
        "config_map_uid": config_uid,
        "config_map_resource_version": config_rv,
        "secret_name": package.packet.value["secret_name"],
        "secret_uid": secret_uid,
        "secret_resource_version": secret_rv,
        "packet_sha256": package.packet.value["sha256"],
        "plan_sha256": package.packet.value["operation"]["plan_sha256"],
    }
    append_journal(journal, result)
    return result


def _validate_reconciled_objects(
    package: Package,
    *,
    job: dict[str, Any] | None,
    config_map: dict[str, Any] | None,
    secret: dict[str, Any] | None,
) -> list[dict[str, str]]:
    """Validate and bind each object left by one uncertain List create."""
    result: list[dict[str, str]] = []
    for label, resource, expected, value in (
        ("Job", "jobs.batch", package.job, job),
        ("ConfigMap", "configmaps", package.config_map, config_map),
        ("Secret", "secrets", package.secret, secret),
    ):
        if value is None:
            continue
        if label == "Job":
            _assert_exact_job(value, package.packet)
            try:
                actual_job = stable_job_preview(value)
                expected_job = stable_job_preview(package.job)
            except ValueError as error:
                raise QualificationJobError("reconciled Job has no stable form") from error
            if not _contains(actual_job, expected_job):
                raise QualificationJobError("reconciled Job differs from the exact packet")
        elif (
            value.get("data") != expected["data"]
            or value.get("immutable") is not True
            or value.get("metadata", {}).get("annotations", {}).get(PACKET_ANNOTATION)
            != package.packet.value["sha256"]
            or (label == "Secret" and value.get("type") != expected["type"])
        ):
            raise QualificationJobError(f"reconciled {label} differs from the exact packet")
        metadata = _metadata(value, f"reconciled {label}")
        uid, resource_version = _identity(metadata, f"reconciled {label}")
        if (
            metadata.get("name") != expected["metadata"]["name"]
            or metadata.get("namespace") != NAMESPACE
        ):
            raise QualificationJobError(f"reconciled {label} identity differs from the packet")
        result.append(
            {
                "resource": resource,
                "name": expected["metadata"]["name"],
                "uid": uid,
                "resource_version": resource_version,
            }
        )
    return result


def _finish_partial_create_cleanup(
    package: Package,
    *,
    cluster: Cluster,
    journal: Path,
    intent: dict[str, Any],
    timeout_seconds: float,
    monotonic: Callable[[], float],
    sleep: Callable[[float], None],
) -> dict[str, Any]:
    if (
        timeout_seconds <= 0
        or set(intent)
        != {
            "schema",
            "state",
            "packet_sha256",
            "objects",
            "validation_mode",
            "never_retry_create",
            "sha256",
        }
        or intent.get("state") != "PARTIAL_CREATE_CLEANUP_INTENT"
        or intent.get("packet_sha256") != package.packet.value["sha256"]
        or intent.get("never_retry_create") is not True
    ):
        raise QualificationJobError("partial create cleanup intent is invalid")

    validation_mode = intent.get("validation_mode")
    if validation_mode not in {EXACT_PACKET_CLEANUP, MUTATED_CREATE_RESPONSE_CLEANUP}:
        raise QualificationJobError("partial create cleanup validation mode is invalid")

    expected_names = {
        "jobs.batch": package.packet.value["job_name"],
        "configmaps": package.packet.value["config_map_name"],
        "secrets": package.packet.value["secret_name"],
    }

    def validate_bindings(rows: object) -> list[dict[str, str]]:
        if (
            not isinstance(rows, list)
            or not rows
            or any(
                not isinstance(row, dict)
                or set(row) != {"resource", "name", "uid", "resource_version"}
                or row.get("resource") not in {"jobs.batch", "configmaps", "secrets"}
                or row.get("name") != expected_names.get(str(row.get("resource")))
                or KUBE_NAME.fullmatch(str(row.get("name"))) is None
                or KUBE_UID.fullmatch(str(row.get("uid"))) is None
                or not isinstance(row.get("resource_version"), str)
                or not row.get("resource_version")
                for row in rows
            )
        ):
            raise QualificationJobError("partial create cleanup object binding is invalid")
        return rows

    objects = validate_bindings(intent.get("objects"))
    expected_by_resource = {row["resource"]: row for row in objects}
    if len(expected_by_resource) != len(objects):
        raise QualificationJobError("partial create cleanup object binding is ambiguous")
    if validation_mode == MUTATED_CREATE_RESPONSE_CLEANUP and set(expected_by_resource) != set(
        expected_names
    ):
        raise QualificationJobError("mutated create cleanup must bind all three response objects")
    for extension in [
        row for row in read_journal(journal) if row.get("state") == "PARTIAL_CREATE_EXTENSION"
    ]:
        assert_sealed(extension, PARTIAL_CREATE_EXTENSION_SCHEMA, "partial create extension")
        if (
            extension.get("packet_sha256") != package.packet.value["sha256"]
            or extension.get("partial_intent_sha256") != intent["sha256"]
        ):
            raise QualificationJobError("partial create extension differs from the intent")
        for row in validate_bindings(extension.get("objects")):
            previous = expected_by_resource.get(row["resource"])
            if previous is not None and previous["uid"] != row["uid"]:
                raise QualificationJobError("partial create extension changes a bound UID")
            expected_by_resource[row["resource"]] = row

    deadline = monotonic() + timeout_seconds
    observed_children: list[dict[str, str]] = []
    while True:
        observation = _stable_uncertain_observation(
            package,
            cluster,
            deadline=deadline,
            monotonic=monotonic,
            sleep=sleep,
        )
        live = {
            "jobs.batch": observation["job"],
            "configmaps": observation["config_map"],
            "secrets": observation["secret"],
        }
        if validation_mode == EXACT_PACKET_CLEANUP:
            validated = _validate_reconciled_objects(
                package,
                job=live["jobs.batch"],
                config_map=live["configmaps"],
                secret=live["secrets"],
            )
        else:
            validated = []
            for resource, value in live.items():
                if value is None:
                    continue
                bound = expected_by_resource[resource]
                metadata = _bound_metadata(
                    value,
                    label=f"mutated create {resource}",
                    name=bound["name"],
                    uid=bound["uid"],
                )
                _, resource_version = _identity(metadata, f"mutated create {resource}")
                validated.append({**bound, "resource_version": resource_version})
        new_bindings: list[dict[str, str]] = []
        for row in validated:
            previous = expected_by_resource.get(row["resource"])
            if previous is not None and previous["uid"] != row["uid"]:
                raise QualificationJobError("partial create found a same-name replacement UID")
            if previous is None:
                new_bindings.append(row)
        if new_bindings:
            extension = sealed(
                {
                    "schema": PARTIAL_CREATE_EXTENSION_SCHEMA,
                    "state": "PARTIAL_CREATE_EXTENSION",
                    "packet_sha256": package.packet.value["sha256"],
                    "partial_intent_sha256": intent["sha256"],
                    "objects": new_bindings,
                }
            )
            append_journal(journal, extension)
            expected_by_resource.update({row["resource"]: row for row in new_bindings})

        job_binding = expected_by_resource.get("jobs.batch")
        if observation["children"] and job_binding is None:
            raise QualificationJobError("partial create has children without a bound Job")
        if job_binding is not None:
            binding = {"job_name": job_binding["name"], "job_uid": job_binding["uid"]}
            for child in _owned_inventory(cluster, binding):
                if child not in observed_children:
                    observed_children.append(child)
            current_job = live["jobs.batch"]
            if current_job is not None:
                metadata = _bound_metadata(
                    current_job,
                    label="partial Job",
                    name=job_binding["name"],
                    uid=job_binding["uid"],
                )
                _, resource_version = _identity(metadata, "partial Job")
                cluster.delete_uid(
                    "jobs.batch",
                    NAMESPACE,
                    job_binding["name"],
                    job_binding["uid"],
                    resource_version,
                    confirmed_live=True,
                )
            while True:
                root = cluster.get_optional("jobs.batch", NAMESPACE, job_binding["name"])
                if root is not None:
                    _bound_metadata(
                        root,
                        label="partial Job",
                        name=job_binding["name"],
                        uid=job_binding["uid"],
                    )
                if root is None and not _owned_inventory(cluster, binding):
                    break
                if monotonic() >= deadline:
                    raise QualificationJobError("partial create Job cleanup remains incomplete")
                sleep(min(RECONCILE_POLL_SECONDS, max(0.0, deadline - monotonic())))

        for resource in ("configmaps", "secrets"):
            bound = expected_by_resource.get(resource)
            value = live[resource]
            if bound is None or value is None:
                continue
            metadata = _bound_metadata(
                value, label="partial source object", name=bound["name"], uid=bound["uid"]
            )
            _, resource_version = _identity(metadata, "partial source object")
            cluster.delete_uid(
                resource,
                NAMESPACE,
                bound["name"],
                bound["uid"],
                resource_version,
                confirmed_live=True,
            )

        final = _stable_uncertain_observation(
            package,
            cluster,
            deadline=deadline,
            monotonic=monotonic,
            sleep=sleep,
        )
        if (
            not any(final[label] is not None for label in ("job", "config_map", "secret"))
            and not final["children"]
        ):
            break
    result = {
        "state": "KUBERNETES_CREATE_RECONCILED_PARTIAL_RELEASED",
        "submitted": "unknown",
        "kubernetes_release_confirmed": True,
        "environment_cleanup_proven": False,
        "release_confirmed": False,
        "possible_external_environment_leak": True,
        "packet_sha256": package.packet.value["sha256"],
        "released_objects": sorted(expected_by_resource.values(), key=lambda row: row["resource"]),
        "owned_children": sorted(observed_children, key=lambda row: (row["resource"], row["name"])),
        "gpus": 0,
    }
    append_journal(journal, result)
    return result


def _journal_binding(path: Path, package: Package) -> dict[str, Any]:
    rows = read_journal(path)
    responses = [
        row
        for row in rows
        if row.get("state") in {"KUBERNETES_CREATE_RESPONSE", "KUBERNETES_CREATE_RECONCILED"}
    ]
    if len(responses) != 1:
        raise QualificationJobError("create journal has no unambiguous exact-UID binding")
    value = responses[0]
    if (
        value.get("packet_sha256") != package.packet.value["sha256"]
        or value.get("plan_sha256") != package.packet.value["operation"]["plan_sha256"]
        or value.get("job_name") != package.packet.value["job_name"]
        or value.get("config_map_name") != package.packet.value["config_map_name"]
        or value.get("secret_name") != package.packet.value["secret_name"]
        or KUBE_UID.fullmatch(str(value.get("job_uid"))) is None
        or KUBE_UID.fullmatch(str(value.get("config_map_uid"))) is None
        or KUBE_UID.fullmatch(str(value.get("secret_uid"))) is None
        or not isinstance(value.get("job_resource_version"), str)
        or not isinstance(value.get("config_map_resource_version"), str)
        or not isinstance(value.get("secret_resource_version"), str)
    ):
        raise QualificationJobError("create journal binding differs from the packet")
    return value


def _bound_metadata(value: dict[str, Any], *, label: str, name: str, uid: str) -> dict[str, Any]:
    metadata = _metadata(value, label)
    if (
        metadata.get("name") != name
        or metadata.get("namespace") != NAMESPACE
        or metadata.get("uid") != uid
    ):
        raise QualificationJobError(f"{label} exact name, namespace, or UID changed")
    return metadata


def _owned_by(value: dict[str, Any], *, job_name: str, job_uid: str) -> bool:
    owners = _metadata(value, str(value.get("kind", "resource"))).get("ownerReferences")
    return isinstance(owners, list) and any(
        isinstance(owner, dict)
        and owner.get("kind") == "Job"
        and owner.get("name") == job_name
        and owner.get("uid") == job_uid
        and owner.get("controller") is True
        for owner in owners
    )


def _owned_inventory(cluster: Cluster, binding: dict[str, Any]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for resource, selector in (
        ("pods", f"batch.kubernetes.io/controller-uid={binding['job_uid']}"),
        ("workloads.kueue.x-k8s.io", f"kueue.x-k8s.io/job-uid={binding['job_uid']}"),
    ):
        rows = _items(
            cluster.list(resource, NAMESPACE, label_selector=selector), f"owned {resource}"
        )
        for row in rows:
            if not _owned_by(row, job_name=binding["job_name"], job_uid=binding["job_uid"]):
                raise QualificationJobError("owned child has a different root owner")
            metadata = _metadata(row, f"owned {resource}")
            name = metadata.get("name")
            uid = metadata.get("uid")
            if KUBE_NAME.fullmatch(str(name)) is None or KUBE_UID.fullmatch(str(uid)) is None:
                raise QualificationJobError("owned child identity is invalid")
            result.append({"resource": resource, "name": str(name), "uid": str(uid)})
    return sorted(result, key=lambda row: (row["resource"], row["name"]))


def _terminal(job: dict[str, Any]) -> str:
    status = job.get("status")
    if not isinstance(status, dict) or status.get("active", 0) != 0:
        raise QualificationJobError("Job is not terminal or still has an active Pod")
    conditions = status.get("conditions")
    if not isinstance(conditions, list):
        raise QualificationJobError("Job terminal conditions are missing")
    observed = {
        row.get("type")
        for row in conditions
        if isinstance(row, dict)
        and row.get("status") == "True"
        and row.get("type") in {"Complete", "Failed"}
    }
    if len(observed) != 1:
        raise QualificationJobError("Job is not unambiguously terminal")
    return observed.pop()


def status(packet_path: Path, *, cluster: Cluster, journal: Path) -> dict[str, Any]:
    package = build_package(packet_path)
    binding = _journal_binding(journal, package)
    job = cluster.get_optional("jobs.batch", NAMESPACE, binding["job_name"])
    config_map = cluster.get_optional("configmaps", NAMESPACE, binding["config_map_name"])
    secret = cluster.get_optional("secrets", NAMESPACE, binding["secret_name"])
    terminal: str | None = None
    if job is not None:
        _bound_metadata(job, label="Job", name=binding["job_name"], uid=binding["job_uid"])
        _assert_exact_job(job, package.packet)
        try:
            terminal = _terminal(job)
        except QualificationJobError:
            terminal = None
    if config_map is not None:
        _bound_metadata(
            config_map,
            label="ConfigMap",
            name=binding["config_map_name"],
            uid=binding["config_map_uid"],
        )
    if secret is not None:
        _bound_metadata(
            secret,
            label="Secret",
            name=binding["secret_name"],
            uid=binding["secret_uid"],
        )
    children = _owned_inventory(cluster, binding)
    return {
        "schema": "cyber_task_quality_cpu_job_status_v1",
        "job_present": job is not None,
        "config_map_present": config_map is not None,
        "secret_present": secret is not None,
        "terminal_condition": terminal,
        "owned_children": len(children),
        "gpus": 0,
        "logs_prompts_traces_scores_or_credentials_read": False,
    }


def _load_cleanup_intent(path: Path, package: Package, binding: dict[str, Any]) -> dict[str, Any]:
    value = read_json(path, "cleanup intent")
    assert_sealed(value, CLEANUP_INTENT_SCHEMA, "cleanup intent")
    if (
        set(value)
        != {
            "schema",
            "packet_sha256",
            "job",
            "config_map",
            "secret",
            "terminal_condition",
            "owned_children",
            "sha256",
        }
        or value.get("packet_sha256") != package.packet.value["sha256"]
        or value.get("job") != {"name": binding["job_name"], "uid": binding["job_uid"]}
        or value.get("config_map")
        != {"name": binding["config_map_name"], "uid": binding["config_map_uid"]}
        or value.get("secret") != {"name": binding["secret_name"], "uid": binding["secret_uid"]}
        or value.get("terminal_condition") not in {"Complete", "Failed"}
    ):
        raise QualificationJobError("cleanup intent differs from the exact UID binding")
    children = value.get("owned_children")
    if not isinstance(children, list) or any(
        not isinstance(row, dict)
        or set(row) != {"resource", "name", "uid"}
        or row.get("resource") not in {"pods", "workloads.kueue.x-k8s.io"}
        or KUBE_NAME.fullmatch(str(row.get("name"))) is None
        or KUBE_UID.fullmatch(str(row.get("uid"))) is None
        for row in children
    ):
        raise QualificationJobError("cleanup intent child binding is invalid")
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
    package = build_package(packet_path)
    binding = _journal_binding(journal, package)
    if receipt_path.exists() or receipt_path.is_symlink():
        receipt = read_json(receipt_path, "cleanup receipt")
        assert_sealed(receipt, CLEANUP_RECEIPT_SCHEMA, "cleanup receipt")
        if (
            receipt.get("packet_sha256") != package.packet.value["sha256"]
            or receipt.get("kubernetes_release_confirmed") is not True
        ):
            raise QualificationJobError("cleanup receipt differs from this packet")
        return receipt
    if timeout_seconds <= 0:
        raise QualificationJobError("cleanup timeout must be positive")
    intent_path = receipt_path.with_suffix(receipt_path.suffix + ".intent")
    if intent_path.exists() or intent_path.is_symlink():
        intent = _load_cleanup_intent(intent_path, package, binding)
        condition = intent["terminal_condition"]
        children = intent["owned_children"]
    else:
        job = cluster.get_optional("jobs.batch", NAMESPACE, binding["job_name"])
        if job is None:
            raise QualificationJobError("exact terminal Job must be observed before cleanup")
        _bound_metadata(job, label="Job", name=binding["job_name"], uid=binding["job_uid"])
        _assert_exact_job(job, package.packet)
        condition = _terminal(job)
        children = _owned_inventory(cluster, binding)
        for child in children:
            value = cluster.get_optional(child["resource"], NAMESPACE, child["name"])
            if value is None:
                continue
            _bound_metadata(value, label="owned child", name=child["name"], uid=child["uid"])
            if child["resource"] == "pods" and value.get("status", {}).get("phase") in {
                "Pending",
                "Running",
                "Unknown",
            }:
                raise QualificationJobError("terminal Job still has an active or unknown Pod")
        body = {
            "schema": CLEANUP_INTENT_SCHEMA,
            "packet_sha256": package.packet.value["sha256"],
            "job": {"name": binding["job_name"], "uid": binding["job_uid"]},
            "config_map": {
                "name": binding["config_map_name"],
                "uid": binding["config_map_uid"],
            },
            "secret": {"name": binding["secret_name"], "uid": binding["secret_uid"]},
            "terminal_condition": condition,
            "owned_children": children,
        }
        intent = sealed(body)
        write_json_once(intent_path, intent)
        intent = _load_cleanup_intent(intent_path, package, binding)
    job = cluster.get_optional("jobs.batch", NAMESPACE, binding["job_name"])
    if job is not None:
        job_metadata = _bound_metadata(
            job, label="Job", name=binding["job_name"], uid=binding["job_uid"]
        )
        _assert_exact_job(job, package.packet)
        if _terminal(job) != condition:
            raise QualificationJobError("terminal Job condition differs from cleanup intent")
        _, fresh_job_rv = _identity(job_metadata, "fresh Job")
        cluster.delete_uid(
            "jobs.batch",
            NAMESPACE,
            binding["job_name"],
            binding["job_uid"],
            fresh_job_rv,
            confirmed_live=True,
        )
    deadline = monotonic() + timeout_seconds
    while True:
        root = cluster.get_optional("jobs.batch", NAMESPACE, binding["job_name"])
        if root is not None:
            _bound_metadata(root, label="Job", name=binding["job_name"], uid=binding["job_uid"])
        live_children = _owned_inventory(cluster, binding)
        if root is None and not live_children:
            break
        if monotonic() >= deadline:
            raise QualificationJobError("foreground Job cleanup remains incomplete")
        sleep(min(5.0, max(0.0, deadline - monotonic())))
    config_map = cluster.get_optional("configmaps", NAMESPACE, binding["config_map_name"])
    if config_map is not None:
        config_metadata = _bound_metadata(
            config_map,
            label="ConfigMap",
            name=binding["config_map_name"],
            uid=binding["config_map_uid"],
        )
        _, fresh_config_rv = _identity(config_metadata, "fresh ConfigMap")
        cluster.delete_uid(
            "configmaps",
            NAMESPACE,
            binding["config_map_name"],
            binding["config_map_uid"],
            fresh_config_rv,
            confirmed_live=True,
        )
    secret = cluster.get_optional("secrets", NAMESPACE, binding["secret_name"])
    if secret is not None:
        secret_metadata = _bound_metadata(
            secret,
            label="Secret",
            name=binding["secret_name"],
            uid=binding["secret_uid"],
        )
        _, fresh_secret_rv = _identity(secret_metadata, "fresh Secret")
        cluster.delete_uid(
            "secrets",
            NAMESPACE,
            binding["secret_name"],
            binding["secret_uid"],
            fresh_secret_rv,
            confirmed_live=True,
        )
    if (
        cluster.get_optional("jobs.batch", NAMESPACE, binding["job_name"]) is not None
        or cluster.get_optional("configmaps", NAMESPACE, binding["config_map_name"]) is not None
        or cluster.get_optional("secrets", NAMESPACE, binding["secret_name"]) is not None
        or _owned_inventory(cluster, binding)
    ):
        raise QualificationJobError("final root or child absence is unproven")
    body = {
        "schema": CLEANUP_RECEIPT_SCHEMA,
        "packet_sha256": package.packet.value["sha256"],
        "plan_sha256": package.packet.value["operation"]["plan_sha256"],
        "job": {"name": binding["job_name"], "uid": binding["job_uid"]},
        "config_map": {"name": binding["config_map_name"], "uid": binding["config_map_uid"]},
        "secret": {"name": binding["secret_name"], "uid": binding["secret_uid"]},
        "terminal_condition": condition,
        "owned_children": children,
        "cleanup_intent_sha256": intent["sha256"],
        "kubernetes_release_confirmed": True,
        "environment_cleanup_proven": condition == "Complete",
        "release_confirmed": condition == "Complete",
        "possible_external_environment_leak": condition != "Complete",
        "gpus": 0,
        "logs_prompts_traces_scores_or_credentials_read": False,
    }
    receipt = sealed(body)
    write_json_once(receipt_path, receipt)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--repo-root", type=Path, required=True)
    prepare.add_argument("--plan", type=Path, required=True)
    prepare.add_argument("--canary", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--expected-source-commit", required=True)
    authorize = commands.add_parser("authorize")
    authorize.add_argument("--packet", type=Path, required=True)
    authorize.add_argument("--output", type=Path, required=True)
    authorize.add_argument("--root-authorization-id", required=True)
    launch = commands.add_parser("launch")
    launch.add_argument("--packet", type=Path, required=True)
    launch.add_argument("--authorization", type=Path, required=True)
    launch.add_argument("--repo-root", type=Path, required=True)
    launch.add_argument("--journal", type=Path, required=True)
    launch.add_argument("--confirm-live-create", action="store_true")
    read = commands.add_parser("status")
    read.add_argument("--packet", type=Path, required=True)
    read.add_argument("--journal", type=Path, required=True)
    reconcile = commands.add_parser("reconcile")
    reconcile.add_argument("--packet", type=Path, required=True)
    reconcile.add_argument("--journal", type=Path, required=True)
    reconcile.add_argument("--confirm-partial-delete", action="store_true")
    cleanup = commands.add_parser("cleanup")
    cleanup.add_argument("--packet", type=Path, required=True)
    cleanup.add_argument("--journal", type=Path, required=True)
    cleanup.add_argument("--receipt", type=Path, required=True)
    cleanup.add_argument("--confirm-live-delete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        if args.command == "prepare":
            result = prepare_packet(
                repo_root=args.repo_root,
                plan_path=args.plan,
                canary_path=args.canary,
                output=args.output,
                expected_source_commit=args.expected_source_commit,
            )
        elif args.command == "authorize":
            result = write_launch_authorization(
                args.packet,
                args.output,
                root_authorization_id=args.root_authorization_id,
            )
        elif args.command == "launch":
            if args.confirm_live_create is not True:
                raise QualificationJobError("live create requires explicit confirmation")
            result = launch_once(
                args.packet,
                authorization_path=args.authorization,
                repo_root=args.repo_root,
                cluster=KubectlCluster(),
                journal=args.journal,
            )
        elif args.command == "status":
            result = status(args.packet, cluster=KubectlCluster(), journal=args.journal)
        elif args.command == "reconcile":
            result = reconcile_uncertain(
                args.packet,
                cluster=KubectlCluster(),
                journal=args.journal,
                confirm_partial_cleanup=args.confirm_partial_delete,
            )
        else:
            if args.confirm_live_delete is not True:
                raise QualificationJobError("live cleanup requires explicit confirmation")
            result = cleanup_once(
                args.packet,
                cluster=KubectlCluster(),
                journal=args.journal,
                receipt_path=args.receipt,
            )
    except BaseException as error:  # noqa: BLE001 - suppress private provider output
        print(
            json.dumps(
                {
                    "schema": "cyber_task_quality_cpu_job_cli_failure_v1",
                    "status": "failed_no_automatic_retry",
                    "failure_code": type(error).__name__.lower(),
                    "private_content_included": False,
                },
                sort_keys=True,
            )
        )
        raise SystemExit(1) from None
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
