"""Guard one new CPU Fleet held-out evaluator from preview through collection.

This is intentionally not a general Kubernetes launcher.  A caller supplies a
sealed packet for one arm of a predeclared matched comparison.  The launcher
checks that its names, output, database and scientific identity are unused,
proves the *server-rendered root Job* opted out of failure alerts, records an
intent, and sends one create request for the immutable ConfigMap/Job pair.

The terminal collector is read-only.  It records only Kubernetes identities and
score-blind database counts; it never reads rollout logs, prompts, traces,
answers, flags, or scores.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Protocol
from urllib.parse import parse_qsl, quote, urlsplit, urlunsplit

import yaml

from evals.fleet.evaluate import stable_job_preview

PACKET_SCHEMA = "cyber_fleet_heldout_launch_packet_v1"
COMPARISON_PROTOCOL_SCHEMA = "cyber_fleet_matched_evaluation_protocol_v1"
TERMINAL_SCHEMA = "cyber_fleet_heldout_terminal_observation_v1"
NAMESPACE = "fleet-train-jobs"
FAILURE_ALERT_ANNOTATION = "fleet.ai/failure-alerts"
FAILURE_ALERT_OFF = "off"
CREATE_ONCE_ANNOTATION = "cyber-post-train.fleet.ai/create-once"
POSTGRES_CLIENT_LABEL = "cyber-post-train.fleet.ai/postgres-client"
POSTGRES_CLIENT_LABEL_VALUE = "true"
ROLLOUT_DATABASE_ENV = "ROLLOUT_DATABASE_URL"
ROLLOUT_DATABASE_SECRET = "chris-cyber-rollout-postgres-v1"
KUBERNETES_NAME = re.compile(r"[a-z0-9](?:[-a-z0-9]{0,61}[a-z0-9])?")
DATABASE_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,62}")
KUBERNETES_UID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
SHA256 = re.compile(r"sha256:[0-9a-f]{64}")
ENV_PREFIX = re.compile(r"(?:[A-Za-z_][A-Za-z0-9_]*)?")
SUPPORTED_SPLIT_SCHEMAS = {
    "cyber_representative_study_split_v2",
    "cyber_parameterized_task_family_split_v1",
}

SEALED_FILES = {
    "evaluation_config",
    "task_set",
    "split_manifest",
    "comparison_protocol",
    "config_map",
    "job",
    "checkpoint_provenance",
    "serving_route_proof",
}
ALL_FILES = {*SEALED_FILES, "evaluation_ledger"}
SEALED_EVALUATOR_MODULES = (
    "evaluate.py",
    "rollout_worker.py",
    "rollout_postgres.py",
    "rollout_ledger.py",
    "opencode_self_hosted.py",
    "fixed_proxy.py",
    "exact_pass4_crypto.py",
    "exact_pass4_universe.py",
    "rollout_campaign.py",
)
SEALED_RUNTIME_IDENTITY_FILES = SEALED_EVALUATOR_MODULES[:7]
IDENTITY_FIELDS = {
    "protocol_id",
    "comparison_arms",
    "arm_id",
    "evaluation_config_name",
    "evaluation_config_sha256",
    "task_selection_sha256",
    "split_manifest_file_sha256",
    "split_manifest_sha256",
    "comparison_protocol_file_sha256",
    "comparison_protocol_sha256",
    "checkpoint_provenance_sha256",
    "serving_route_proof_sha256",
    "model_revision",
    "harness",
    "harness_version",
    "context_management",
    "sampling_seed",
    "pass_k",
    "retry_limit",
    "output_root",
    "database",
}


class HeldoutLaunchError(ValueError):
    """A fail-closed launch or terminal-observation error."""


class Cluster(Protocol):
    """The only Kubernetes operations this narrow mechanism needs."""

    def list(
        self,
        resource: str,
        namespace: str,
        *,
        field_selector: str | None = None,
        label_selector: str | None = None,
    ) -> dict[str, Any]: ...

    def get(self, resource: str, namespace: str, name: str) -> dict[str, Any]: ...

    def server_dry_run(self, namespace: str, bundle: dict[str, Any]) -> dict[str, Any]: ...

    def create_once(self, namespace: str, bundle: dict[str, Any]) -> dict[str, Any]: ...


class Database(Protocol):
    """Score-blind database reads used for duplicate and terminal checks."""

    def exists(self, database: str) -> bool: ...

    def summary(self, database: str) -> dict[str, Any]: ...

    def cell_status(
        self, database: str, *, task_version_id: str, model_revision: str, attempt: int
    ) -> dict[str, Any]: ...


@dataclass(frozen=True)
class LaunchPacket:
    path: Path
    namespace: str
    job_name: str
    config_map_name: str
    output_root: str
    database: str
    files: dict[str, Path]
    file_sha256: dict[str, str]
    identity: dict[str, Any]
    identity_sha256: str


@dataclass(frozen=True)
class Package:
    packet: LaunchPacket
    evaluation_config: dict[str, Any]
    config_map: dict[str, Any]
    job: dict[str, Any]

    @property
    def bundle(self) -> dict[str, Any]:
        return {
            "apiVersion": "v1",
            "kind": "List",
            "items": [self.config_map, self.job],
        }


@dataclass(frozen=True)
class SealedEvaluation:
    """A plan and its rows compiled by the packet's exact evaluator bytes."""

    plan: dict[str, Any]
    rows: tuple[dict[str, Any], ...]


def _canonical_digest(value: Any) -> str:
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    except (TypeError, ValueError) as exc:
        raise HeldoutLaunchError("launch packet contains a non-canonical value") from exc
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise HeldoutLaunchError(f"{label} must be a mapping")
    return value


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HeldoutLaunchError(f"{label} is not readable JSON") from exc
    return _require_mapping(value, label)


def _load_manifest(path: Path, label: str) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise HeldoutLaunchError(f"{label} is not readable YAML") from exc
    return _require_mapping(value, label)


def _require_digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256.fullmatch(value) is None:
        raise HeldoutLaunchError(f"{label} must be a SHA-256 digest")
    return value


def _require_name(value: Any, label: str) -> str:
    if not isinstance(value, str) or KUBERNETES_NAME.fullmatch(value) is None:
        raise HeldoutLaunchError(f"{label} must be a Kubernetes name")
    return value


def _safe_packet_file(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise HeldoutLaunchError(f"{label} path is required")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise HeldoutLaunchError(f"{label} path must stay below the packet directory")
    path = root / Path(relative)
    if path.is_symlink() or not path.is_file():
        raise HeldoutLaunchError(f"{label} path is not a regular file")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise HeldoutLaunchError(f"{label} path cannot be resolved") from exc
    if root not in resolved.parents:
        raise HeldoutLaunchError(f"{label} path escapes the packet directory")
    return resolved


def _validate_output_root(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise HeldoutLaunchError("output_root is required")
    path = PurePosixPath(value)
    if (
        not path.is_absolute()
        or path.parts[:4] != ("/", "mnt", "sfs", "jobs")
        or len(path.parts) < 5
        or ".." in path.parts
        or path.as_posix() != value
    ):
        raise HeldoutLaunchError("output_root must be a canonical per-evaluation SFS path")
    return value


def _packet_file_entries(value: Any, root: Path) -> tuple[dict[str, Path], dict[str, str]]:
    entries = _require_mapping(value, "packet files")
    if set(entries) != ALL_FILES:
        raise HeldoutLaunchError("launch packet file set is incomplete or has unknown entries")
    paths: dict[str, Path] = {}
    digests: dict[str, str] = {}
    for name in sorted(ALL_FILES):
        entry = _require_mapping(entries[name], f"packet file {name}")
        expected = {"path"} | ({"sha256"} if name in SEALED_FILES else set())
        if set(entry) != expected:
            raise HeldoutLaunchError(f"packet file {name} fields are invalid")
        path = _safe_packet_file(root, entry["path"], name)
        paths[name] = path
        if name in SEALED_FILES:
            digest = _require_digest(entry["sha256"], f"packet file {name}")
            if _file_sha256(path) != digest:
                raise HeldoutLaunchError(f"packet file {name} bytes differ from its digest")
            digests[name] = digest
    return paths, digests


def _validate_identity(identity: Any, packet: LaunchPacket) -> None:
    value = _require_mapping(identity, "evaluation_identity")
    if not IDENTITY_FIELDS.issubset(value):
        raise HeldoutLaunchError("evaluation identity is incomplete")
    if not isinstance(value["protocol_id"], str) or not value["protocol_id"]:
        raise HeldoutLaunchError("evaluation identity protocol_id is invalid")
    arms = value["comparison_arms"]
    if (
        not isinstance(arms, list)
        or len(arms) < 2
        or any(not isinstance(arm, str) or not arm for arm in arms)
        or len(arms) != len(set(arms))
    ):
        raise HeldoutLaunchError("evaluation identity comparison_arms is invalid")
    if value["arm_id"] not in arms:
        raise HeldoutLaunchError("evaluation arm is absent from its matched protocol")
    for field in (
        "evaluation_config_name",
        "model_revision",
        "harness",
        "harness_version",
        "context_management",
    ):
        if not isinstance(value[field], str) or not value[field]:
            raise HeldoutLaunchError(f"evaluation identity {field} is invalid")
    for field in (
        "evaluation_config_sha256",
        "task_selection_sha256",
        "split_manifest_file_sha256",
        "split_manifest_sha256",
        "comparison_protocol_file_sha256",
        "comparison_protocol_sha256",
        "checkpoint_provenance_sha256",
        "serving_route_proof_sha256",
    ):
        _require_digest(value[field], f"evaluation identity {field}")
    if any(
        (
            value["evaluation_config_sha256"] != packet.file_sha256["evaluation_config"],
            value["task_selection_sha256"] != packet.file_sha256["task_set"],
            value["split_manifest_file_sha256"] != packet.file_sha256["split_manifest"],
            value["comparison_protocol_file_sha256"] != packet.file_sha256["comparison_protocol"],
            value["checkpoint_provenance_sha256"] != packet.file_sha256["checkpoint_provenance"],
            value["serving_route_proof_sha256"] != packet.file_sha256["serving_route_proof"],
            value["output_root"] != packet.output_root,
            value["database"] != packet.database,
        )
    ):
        raise HeldoutLaunchError("evaluation identity does not bind packet bytes and destinations")
    if type(value["sampling_seed"]) is not int or value["sampling_seed"] < 0:
        raise HeldoutLaunchError("evaluation identity sampling_seed is invalid")
    if type(value["pass_k"]) is not int or value["pass_k"] < 1:
        raise HeldoutLaunchError("evaluation identity pass_k is invalid")
    if type(value["retry_limit"]) is not int or value["retry_limit"] < 0:
        raise HeldoutLaunchError("evaluation identity retry_limit is invalid")


def load_packet(path: Path) -> LaunchPacket:
    """Load one sealed packet without touching Fleet, PostgreSQL, or SFS."""
    if path.is_symlink() or not path.is_file():
        raise HeldoutLaunchError("launch packet must be a regular file")
    packet_path = path.resolve()
    raw = _load_json(packet_path, "launch packet")
    expected = {
        "schema",
        "namespace",
        "job_name",
        "config_map_name",
        "output_root",
        "database",
        "files",
        "evaluation_identity",
        "evaluation_identity_sha256",
    }
    if set(raw) != expected or raw.get("schema") != PACKET_SCHEMA:
        raise HeldoutLaunchError("launch packet schema is unsupported")
    if raw["namespace"] != NAMESPACE:
        raise HeldoutLaunchError("held-out evaluation namespace is not authorized")
    job_name = _require_name(raw["job_name"], "job_name")
    config_map_name = _require_name(raw["config_map_name"], "config_map_name")
    database = raw["database"]
    if not isinstance(database, str) or DATABASE_NAME.fullmatch(database) is None:
        raise HeldoutLaunchError("database name is invalid")
    output_root = _validate_output_root(raw["output_root"])
    files, file_sha256 = _packet_file_entries(raw["files"], packet_path.parent)
    identity = _require_mapping(raw["evaluation_identity"], "evaluation_identity")
    identity_sha256 = _require_digest(
        raw["evaluation_identity_sha256"], "evaluation_identity_sha256"
    )
    if _canonical_digest(identity) != identity_sha256:
        raise HeldoutLaunchError("evaluation identity digest differs from packet")
    packet = LaunchPacket(
        path=packet_path,
        namespace=NAMESPACE,
        job_name=job_name,
        config_map_name=config_map_name,
        output_root=output_root,
        database=database,
        files=files,
        file_sha256=file_sha256,
        identity=identity,
        identity_sha256=identity_sha256,
    )
    _validate_identity(identity, packet)
    return packet


def _container_environment(container: dict[str, Any]) -> dict[str, str]:
    raw = container.get("env")
    if not isinstance(raw, list):
        raise HeldoutLaunchError("evaluator environment is missing")
    result: dict[str, str] = {}
    for entry in raw:
        if not isinstance(entry, dict) or set(entry) - {"name", "value", "valueFrom"}:
            raise HeldoutLaunchError("evaluator environment entry is invalid")
        name = entry.get("name")
        if not isinstance(name, str) or not name or name in result:
            raise HeldoutLaunchError("evaluator environment names are invalid")
        if "value" in entry:
            if not isinstance(entry["value"], str):
                raise HeldoutLaunchError("literal evaluator environment value is invalid")
            result[name] = entry["value"]
        else:
            result[name] = "<valueFrom>"
    return result


def _metadata(value: dict[str, Any], label: str) -> dict[str, Any]:
    metadata = value.get("metadata")
    if not isinstance(metadata, dict):
        raise HeldoutLaunchError(f"{label} metadata is invalid")
    return metadata


def require_postgres_client_label(job: dict[str, Any], *, label: str) -> bool:
    """Fail closed when a PostgreSQL-reading Job lacks its NetworkPolicy label.

    The live rollout PostgreSQL policy selects Pods, not root Jobs. Therefore
    the exact label belongs on ``spec.template.metadata.labels``. Jobs which do
    not receive the rollout database credential are intentionally unaffected.
    """
    if job.get("apiVersion") != "batch/v1" or job.get("kind") != "Job":
        raise HeldoutLaunchError(f"{label} is not a batch/v1 Job")
    spec = job.get("spec")
    template = spec.get("template") if isinstance(spec, dict) else None
    pod = template.get("spec") if isinstance(template, dict) else None
    if not isinstance(pod, dict):
        raise HeldoutLaunchError(f"{label} Pod template is invalid")
    reads_postgres = False
    for group in ("initContainers", "containers"):
        containers = pod.get(group, [])
        if not isinstance(containers, list) or any(
            not isinstance(container, dict) for container in containers
        ):
            raise HeldoutLaunchError(f"{label} {group} is invalid")
        for container in containers:
            environment = container.get("env", [])
            if not isinstance(environment, list) or any(
                not isinstance(entry, dict) for entry in environment
            ):
                raise HeldoutLaunchError(f"{label} container environment is invalid")
            for entry in environment:
                value_from = entry.get("valueFrom")
                secret_ref = (
                    value_from.get("secretKeyRef") if isinstance(value_from, dict) else None
                )
                if (
                    isinstance(value_from, dict)
                    and "secretKeyRef" in value_from
                    and (
                        not isinstance(secret_ref, dict)
                        or set(secret_ref) - {"name", "key", "optional"}
                        or not isinstance(secret_ref.get("name"), str)
                        or not secret_ref["name"]
                        or not isinstance(secret_ref.get("key"), str)
                        or not secret_ref["key"]
                        or ("optional" in secret_ref and type(secret_ref["optional"]) is not bool)
                    )
                ):
                    raise HeldoutLaunchError(f"{label} secretKeyRef is invalid")
                reads_postgres = reads_postgres or entry.get("name") == ROLLOUT_DATABASE_ENV
                reads_postgres = reads_postgres or (
                    isinstance(secret_ref, dict)
                    and (
                        secret_ref.get("name") == ROLLOUT_DATABASE_SECRET
                        or secret_ref.get("key") == ROLLOUT_DATABASE_ENV
                    )
                )
            environment_from = container.get("envFrom", [])
            if not isinstance(environment_from, list) or any(
                not isinstance(entry, dict) for entry in environment_from
            ):
                raise HeldoutLaunchError(f"{label} container envFrom is invalid")
            for entry in environment_from:
                prefix = entry.get("prefix", "")
                if (
                    set(entry) - {"prefix", "secretRef", "configMapRef"}
                    or (("secretRef" in entry) == ("configMapRef" in entry))
                    or not isinstance(prefix, str)
                    or ENV_PREFIX.fullmatch(prefix) is None
                ):
                    raise HeldoutLaunchError(f"{label} container envFrom entry is invalid")
                ref_name = "secretRef" if "secretRef" in entry else "configMapRef"
                reference = entry[ref_name]
                if (
                    not isinstance(reference, dict)
                    or set(reference) - {"name", "optional"}
                    or not isinstance(reference.get("name"), str)
                    or not reference["name"]
                    or ("optional" in reference and type(reference["optional"]) is not bool)
                ):
                    raise HeldoutLaunchError(f"{label} container envFrom entry is invalid")
                secret_ref = entry.get("secretRef")
                reads_postgres = reads_postgres or (
                    isinstance(secret_ref, dict)
                    and secret_ref.get("name") == ROLLOUT_DATABASE_SECRET
                )
    template_metadata = template.get("metadata")
    labels = template_metadata.get("labels") if isinstance(template_metadata, dict) else None
    label_present = isinstance(labels, dict) and POSTGRES_CLIENT_LABEL in labels
    label_exact = label_present and labels.get(POSTGRES_CLIENT_LABEL) == POSTGRES_CLIENT_LABEL_VALUE
    if reads_postgres and not label_exact:
        raise HeldoutLaunchError(
            f"{label} reads rollout PostgreSQL but its Pod template lacks the exact "
            f"{POSTGRES_CLIENT_LABEL}={POSTGRES_CLIENT_LABEL_VALUE} label"
        )
    if not reads_postgres and label_present:
        raise HeldoutLaunchError(
            f"{label} has a stale {POSTGRES_CLIENT_LABEL} label without a rollout "
            "PostgreSQL dependency"
        )
    return reads_postgres


def _namespace_name(value: dict[str, Any], label: str) -> tuple[str, str]:
    metadata = _metadata(value, label)
    name = metadata.get("name")
    namespace = metadata.get("namespace")
    if not isinstance(name, str) or not isinstance(namespace, str):
        raise HeldoutLaunchError(f"{label} name or namespace is invalid")
    return namespace, name


def _contains_config_map_volume(pod: dict[str, Any], name: str) -> bool:
    volumes = pod.get("volumes")
    if not isinstance(volumes, list):
        raise HeldoutLaunchError("Job volumes are invalid")
    for volume in volumes:
        if not isinstance(volume, dict):
            raise HeldoutLaunchError("Job volume is invalid")
        config_map = volume.get("configMap")
        if isinstance(config_map, dict) and config_map.get("name") == name:
            return True
    return False


def _assert_cpu_only(pod: dict[str, Any]) -> None:
    def is_gpu_resource(value: Any) -> bool:
        if not isinstance(value, str):
            raise HeldoutLaunchError("Job resource name is invalid")
        lowered = value.lower()
        return "gpu" in lowered or lowered.startswith("nvidia.com/")

    for group in ("containers", "initContainers", "ephemeralContainers"):
        containers = pod.get(group, [])
        if not isinstance(containers, list):
            raise HeldoutLaunchError(f"Job {group} is invalid")
        for container in containers:
            if not isinstance(container, dict):
                raise HeldoutLaunchError(f"Job {group} entry is invalid")
            resources = container.get("resources", {})
            if not isinstance(resources, dict):
                raise HeldoutLaunchError("Job resources are invalid")
            for limit in ("requests", "limits"):
                values = resources.get(limit, {})
                if not isinstance(values, dict):
                    raise HeldoutLaunchError("Job resource quantity is invalid")
                if any(is_gpu_resource(name) for name in values):
                    raise HeldoutLaunchError("held-out evaluator must not request GPUs")
    overhead = pod.get("overhead", {})
    if not isinstance(overhead, dict):
        raise HeldoutLaunchError("Job Pod overhead is invalid")
    if any(is_gpu_resource(name) for name in overhead):
        raise HeldoutLaunchError("held-out evaluator must not request GPUs")


def _split_index(value: dict[str, Any], expected_sha256: str) -> dict[str, tuple[str, str]]:
    """Validate only the stable, public split surface needed by an evaluator."""
    if (
        value.get("schema") not in SUPPORTED_SPLIT_SCHEMAS
        or value.get("sha256") != expected_sha256
        or _canonical_digest({key: item for key, item in value.items() if key != "sha256"})
        != expected_sha256
    ):
        raise HeldoutLaunchError("sealed representative split manifest is invalid")
    tasks = value.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise HeldoutLaunchError("sealed representative split has no tasks")
    index: dict[str, tuple[str, str]] = {}
    for row in tasks:
        if not isinstance(row, dict):
            raise HeldoutLaunchError("sealed representative split task is invalid")
        task_key = row.get("task_key")
        version = row.get("task_version_id")
        split = row.get("split")
        if (
            not isinstance(task_key, str)
            or not task_key
            or not isinstance(version, str)
            or not version
            or not isinstance(split, str)
            or not split
            or version in index
        ):
            raise HeldoutLaunchError("sealed representative split task is invalid")
        index[version] = (task_key, split)
    return index


def _selection_index(
    value: dict[str, Any], *, split_index: dict[str, tuple[str, str]], packet: LaunchPacket
) -> tuple[str, dict[str, str]]:
    """Require an exact, role-consistent subset of a sealed split manifest."""
    if value.get("schema") != "cyber_eval_task_selection_v2":
        raise HeldoutLaunchError("task selection schema is unsupported")
    binding = value.get("split_manifest")
    if not isinstance(binding, dict) or set(binding) - {"path", "file_sha256", "object_sha256"}:
        raise HeldoutLaunchError("task selection split binding is invalid")
    binding_path = binding.get("path")
    if (
        not isinstance(binding_path, str)
        or not binding_path
        or PurePosixPath(binding_path).is_absolute()
        or ".." in PurePosixPath(binding_path).parts
    ):
        raise HeldoutLaunchError("task selection split path is invalid")
    if (
        binding.get("file_sha256") != packet.file_sha256["split_manifest"]
        or binding.get("object_sha256") != packet.identity["split_manifest_sha256"]
    ):
        raise HeldoutLaunchError("task selection split binding differs from sealed packet")
    role = value.get("selection_role")
    tasks = value.get("tasks")
    if not isinstance(role, str) or not role or not isinstance(tasks, list) or not tasks:
        raise HeldoutLaunchError("task selection role or tasks are invalid")
    if role.casefold() in {"train", "training"}:
        raise HeldoutLaunchError("held-out evaluator selection cannot use a training role")
    if value.get("task_count") is not None and value["task_count"] != len(tasks):
        raise HeldoutLaunchError("task selection count differs from its tasks")
    selected: dict[str, str] = {}
    for row in tasks:
        if not isinstance(row, dict):
            raise HeldoutLaunchError("task selection task is invalid")
        task_key = row.get("task_key")
        version = row.get("task_version_id")
        if (
            not isinstance(task_key, str)
            or not task_key
            or not isinstance(version, str)
            or version in selected
            or split_index.get(version) != (task_key, role)
        ):
            raise HeldoutLaunchError("task selection is not the declared split subset")
        selected[version] = task_key
    return role, selected


def _validate_comparison_protocol(
    value: dict[str, Any], *, packet: LaunchPacket, config: dict[str, Any]
) -> None:
    """Bind this arm to one sealed, complete base-versus-checkpoint treatment."""
    expected = {
        "schema",
        "protocol_id",
        "comparison_arms",
        "model_revisions",
        "task_selection_sha256",
        "split_manifest_file_sha256",
        "split_manifest_sha256",
        "harness",
        "sampling",
        "images",
        "pass_k",
        "retry_limit",
        "sha256",
    }
    if set(value) != expected or value.get("schema") != COMPARISON_PROTOCOL_SCHEMA:
        raise HeldoutLaunchError("comparison protocol schema is unsupported")
    if any(
        not isinstance(value[field], dict) or not value[field]
        for field in ("harness", "sampling", "images")
    ):
        raise HeldoutLaunchError("comparison protocol treatment is invalid")
    images = value["images"]
    if set(images) != {"agent", "proxy"} or any(
        not isinstance(image, str)
        or re.fullmatch(r"(?:[^\s]+@)?sha256:[0-9a-f]{64}", image) is None
        for image in images.values()
    ):
        raise HeldoutLaunchError("comparison protocol evaluator images are invalid")
    protocol_sha256 = _require_digest(value.get("sha256"), "comparison protocol sha256")
    if (
        protocol_sha256 != packet.identity["comparison_protocol_sha256"]
        or _canonical_digest({key: item for key, item in value.items() if key != "sha256"})
        != protocol_sha256
    ):
        raise HeldoutLaunchError("comparison protocol digest differs from its sealed identity")
    identity = packet.identity
    if any(
        (
            value.get("protocol_id") != identity["protocol_id"],
            value.get("comparison_arms") != identity["comparison_arms"],
            value.get("task_selection_sha256") != identity["task_selection_sha256"],
            value.get("split_manifest_file_sha256") != identity["split_manifest_file_sha256"],
            value.get("split_manifest_sha256") != identity["split_manifest_sha256"],
            value.get("harness") != config.get("harness"),
            value.get("sampling") != config.get("sampling"),
            value.get("images") != config.get("images"),
            value.get("pass_k") != identity["pass_k"],
            value.get("retry_limit") != identity["retry_limit"],
        )
    ):
        raise HeldoutLaunchError("comparison protocol differs from this sealed evaluation arm")
    revisions = value.get("model_revisions")
    arms = identity["comparison_arms"]
    if (
        not isinstance(revisions, dict)
        or set(revisions) != set(arms)
        or any(not isinstance(revisions[arm], str) or not revisions[arm] for arm in arms)
        or revisions.get(identity["arm_id"]) != identity["model_revision"]
    ):
        raise HeldoutLaunchError("comparison protocol model revisions are invalid")


def build_package(packet_path: Path) -> Package:
    """Open and cross-bind all packet bytes before live duplicate checks."""
    packet = load_packet(packet_path)
    config = _load_json(packet.files["evaluation_config"], "evaluation config")
    task_selection = _load_json(packet.files["task_set"], "task selection")
    split_manifest = _load_json(packet.files["split_manifest"], "representative split manifest")
    comparison_protocol = _load_json(
        packet.files["comparison_protocol"], "matched comparison protocol"
    )
    split_index = _split_index(split_manifest, packet.identity["split_manifest_sha256"])
    _selection_role, selected_tasks = _selection_index(
        task_selection,
        split_index=split_index,
        packet=packet,
    )
    _validate_comparison_protocol(comparison_protocol, packet=packet, config=config)
    config_map = _load_manifest(packet.files["config_map"], "ConfigMap manifest")
    job = _load_manifest(packet.files["job"], "Job manifest")
    if config_map.get("apiVersion") != "v1" or config_map.get("kind") != "ConfigMap":
        raise HeldoutLaunchError("packet ConfigMap manifest is invalid")
    if job.get("apiVersion") != "batch/v1" or job.get("kind") != "Job":
        raise HeldoutLaunchError("packet Job manifest is invalid")
    require_postgres_client_label(job, label="sealed packet Job")
    if _namespace_name(config_map, "ConfigMap") != (packet.namespace, packet.config_map_name):
        raise HeldoutLaunchError("ConfigMap name differs from launch packet")
    if _namespace_name(job, "Job") != (packet.namespace, packet.job_name):
        raise HeldoutLaunchError("Job name differs from launch packet")
    if config_map.get("immutable") is not True:
        raise HeldoutLaunchError("held-out evaluator ConfigMap must be immutable")
    data = config_map.get("data")
    if not isinstance(data, dict):
        raise HeldoutLaunchError("ConfigMap data is invalid")
    spec = job.get("spec")
    if not isinstance(spec, dict) or spec.get("backoffLimit") != 0:
        raise HeldoutLaunchError("held-out evaluator Job must have backoffLimit zero")
    if spec.get("parallelism", 1) != 1 or spec.get("completions", 1) != 1:
        raise HeldoutLaunchError("held-out evaluator Job must have one completion")
    deadline = spec.get("activeDeadlineSeconds")
    if type(deadline) is not int or not 1 <= deadline <= 172800:
        raise HeldoutLaunchError("held-out evaluator Job needs a bounded active deadline")
    template = spec.get("template")
    if not isinstance(template, dict) or not isinstance(template.get("spec"), dict):
        raise HeldoutLaunchError("Job Pod template is invalid")
    pod = template["spec"]
    if pod.get("priorityClassName") != "c1" or pod.get("restartPolicy") != "Never":
        raise HeldoutLaunchError("held-out evaluator CPU policy is invalid")
    if "nodeName" in pod:
        raise HeldoutLaunchError("held-out evaluator must not pin one node")
    _assert_cpu_only(pod)
    containers = pod.get("containers")
    if not isinstance(containers, list):
        raise HeldoutLaunchError("evaluator container list is invalid")
    evaluator = [
        item for item in containers if isinstance(item, dict) and item.get("name") == "evaluator"
    ]
    if len(evaluator) != 1:
        raise HeldoutLaunchError("held-out evaluator requires one evaluator container")
    environment = _container_environment(evaluator[0])
    config_key = environment.get("EVAL_CONFIG_NAME")
    if not isinstance(config_key, str) or not config_key:
        raise HeldoutLaunchError("EVAL_CONFIG_NAME must be a literal ConfigMap key")
    if environment.get("EVAL_OUTPUT") != packet.output_root:
        raise HeldoutLaunchError("Job output root differs from launch packet")
    if environment.get("EVAL_DATABASE") != packet.database:
        raise HeldoutLaunchError("Job database differs from launch packet")
    selected = data.get(config_key)
    generic = data.get("config.json")
    if not isinstance(selected, str) or generic != selected:
        raise HeldoutLaunchError("selected and generic evaluator config bytes must match")
    try:
        embedded_config = json.loads(selected)
    except json.JSONDecodeError as exc:
        raise HeldoutLaunchError("embedded evaluator config is invalid JSON") from exc
    if embedded_config != config:
        raise HeldoutLaunchError("embedded evaluator config differs from sealed config bytes")
    if not _contains_config_map_volume(pod, packet.config_map_name):
        raise HeldoutLaunchError("Job does not mount its sealed ConfigMap")
    annotations = _metadata(job, "Job").get("annotations")
    if (
        not isinstance(annotations, dict)
        or annotations.get(FAILURE_ALERT_ANNOTATION) != FAILURE_ALERT_OFF
    ):
        raise HeldoutLaunchError("Job root failure-alert annotation is not off")
    if annotations.get(CREATE_ONCE_ANNOTATION) != "true":
        raise HeldoutLaunchError("Job create-once annotation is missing")
    identity = packet.identity
    if config.get("name") != identity["evaluation_config_name"]:
        raise HeldoutLaunchError("evaluation config name differs from its identity")
    if config.get("training_data_eligible") is not False:
        raise HeldoutLaunchError("held-out evaluator config must be training-data ineligible")
    task_set = config.get("task_set")
    if not isinstance(task_set, str) or not task_set:
        raise HeldoutLaunchError("evaluation config task set is invalid")
    configured_task_set = (packet.files["evaluation_config"].parent / task_set).resolve()
    if configured_task_set != packet.files["task_set"]:
        raise HeldoutLaunchError("evaluation config task set differs from sealed packet task set")
    models = config.get("models")
    routes = config.get("routes")
    harness = config.get("harness")
    if (
        not isinstance(models, dict)
        or len(models) != 1
        or not isinstance(routes, dict)
        or len(routes) != 1
    ):
        raise HeldoutLaunchError("narrow held-out launch requires one model and one route")
    model_alias, model = next(iter(models.items()))
    if not isinstance(model, dict) or model.get("revision") != identity["model_revision"]:
        raise HeldoutLaunchError("evaluation model revision differs from its identity")
    route = next(iter(routes.values()))
    if not isinstance(route, dict) or route.get("model") != model_alias:
        raise HeldoutLaunchError("evaluation route does not select its sole sealed model")
    task_versions = route.get("task_versions")
    if (
        not isinstance(task_versions, list)
        or any(not isinstance(version, str) or not version for version in task_versions)
        or len(task_versions) != len(set(task_versions))
        or set(task_versions) != set(selected_tasks)
    ):
        raise HeldoutLaunchError("evaluation route task versions differ from sealed task selection")
    if not isinstance(harness, dict) or any(
        (
            harness.get("harness") != identity["harness"],
            harness.get("harness_version") != identity["harness_version"],
            harness.get("context_management") != identity["context_management"],
        )
    ):
        raise HeldoutLaunchError("evaluation harness differs from its identity")
    sampling = config.get("sampling")
    if not isinstance(sampling, dict) or sampling.get("seed") != identity["sampling_seed"]:
        raise HeldoutLaunchError("evaluation sampling seed differs from its identity")
    if config.get("pass_k") != identity["pass_k"]:
        raise HeldoutLaunchError("evaluation pass_k differs from its identity")
    if config.get("max_reviewed_infrastructure_retries") != identity["retry_limit"]:
        raise HeldoutLaunchError("evaluation retry limit differs from its identity")
    return Package(packet=packet, evaluation_config=config, config_map=config_map, job=job)


def sealed_evaluation(package: Package) -> SealedEvaluation:
    """Compile plan and rows with the exact runtime sealed in a launch packet."""

    data = package.config_map.get("data")
    if (
        not isinstance(data, dict)
        or any(not isinstance(data.get(name), str) for name in SEALED_EVALUATOR_MODULES)
        or not isinstance(data.get("jobs.py"), str)
    ):
        raise HeldoutLaunchError("evaluation packet lacks its complete sealed evaluator runtime")
    scientific_config = dict(package.evaluation_config)
    scientific_config.pop("model_artifact_binding", None)
    with tempfile.TemporaryDirectory(prefix="fleet-sealed-evaluator-") as directory:
        root = Path(directory)
        package_root = root / "evals" / "fleet"
        jobs_root = root / "cyber_post_train"
        package_root.mkdir(parents=True, mode=0o700)
        jobs_root.mkdir(mode=0o700)
        (root / "evals" / "__init__.py").write_text("", encoding="utf-8")
        (package_root / "__init__.py").write_text("", encoding="utf-8")
        (jobs_root / "__init__.py").write_text("", encoding="utf-8")
        (jobs_root / "jobs.py").write_text(data["jobs.py"], encoding="utf-8")
        for name in SEALED_EVALUATOR_MODULES:
            (package_root / name).write_text(data[name], encoding="utf-8")
        task_set = data.get("task-set.json")
        task_set_name = scientific_config.get("task_set")
        if (
            not isinstance(task_set, str)
            or not isinstance(task_set_name, str)
            or not task_set_name
            or Path(task_set_name).name != task_set_name
        ):
            raise HeldoutLaunchError("evaluation packet lacks its sealed task set")
        (root / task_set_name).write_text(task_set, encoding="utf-8")
        (root / "config.json").write_text(
            json.dumps(scientific_config, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        command = (
            "import json; from pathlib import Path; from evals.fleet import evaluate; "
            "config=json.loads(Path('config.json').read_text(encoding='utf-8')); "
            "plan=evaluate.compile_eval(config, relative_to=Path('.')); "
            "print(json.dumps({'plan':plan,'rows':evaluate.plan_rows(plan)}, "
            "sort_keys=True, separators=(',', ':'), allow_nan=False))"
        )
        try:
            completed = subprocess.run(
                [sys.executable, "-c", command],
                cwd=root,
                env={
                    "PYTHONPATH": str(root),
                    "PYTHONNOUSERSITE": "1",
                    "PYTHONDONTWRITEBYTECODE": "1",
                },
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            result = json.loads(completed.stdout)
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
            raise HeldoutLaunchError("sealed evaluator could not compile its exact plan") from exc
    plan = result.get("plan") if isinstance(result, dict) else None
    rows = result.get("rows") if isinstance(result, dict) else None
    row_fields = {
        "experiment_id",
        "task_key",
        "task_version_id",
        "model_id",
        "model_revision",
        "serving_block",
        "endpoint_model_id",
        "harness_id",
        "attempt",
        "max_retries",
    }
    if (
        not isinstance(plan, dict)
        or plan.get("schema") != "cyber_fleet_eval_v1"
        or plan.get("sha256")
        != _canonical_digest({key: item for key, item in plan.items() if key != "sha256"})[7:]
        or not isinstance(rows, list)
        or not rows
        or any(not isinstance(row, dict) or set(row) != row_fields for row in rows)
        or {row["harness_id"] for row in rows} != {"protocol-" + plan["sha256"]}
    ):
        raise HeldoutLaunchError("sealed evaluator returned an invalid plan")
    return SealedEvaluation(plan=plan, rows=tuple(rows))


def sealed_evaluation_plan(package: Package) -> dict[str, Any]:
    """Return the packet-runtime plan for callers that do not need its rows."""

    return sealed_evaluation(package).plan


def _list_items(value: Any, label: str) -> list[dict[str, Any]]:
    mapping = _require_mapping(value, label)
    items = mapping.get("items")
    if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
        raise HeldoutLaunchError(f"{label} inventory is invalid")
    return items


def _object_name(item: dict[str, Any], label: str) -> str:
    metadata = item.get("metadata")
    name = metadata.get("name") if isinstance(metadata, dict) else None
    if not isinstance(name, str) or not name:
        raise HeldoutLaunchError(f"{label} inventory contains invalid metadata")
    return name


def _owner_matches(item: dict[str, Any], job_name: str) -> bool:
    metadata = item.get("metadata")
    if not isinstance(metadata, dict):
        raise HeldoutLaunchError("Kubernetes inventory contains invalid metadata")
    labels = metadata.get("labels", {})
    if not isinstance(labels, dict):
        raise HeldoutLaunchError("Kubernetes inventory labels are invalid")
    if labels.get("job-name") == job_name or labels.get("batch.kubernetes.io/job-name") == job_name:
        return True
    owners = metadata.get("ownerReferences", [])
    if not isinstance(owners, list):
        raise HeldoutLaunchError("Kubernetes inventory owner references are invalid")
    for owner in owners:
        if not isinstance(owner, dict):
            raise HeldoutLaunchError("Kubernetes inventory owner reference is invalid")
        if owner.get("kind") == "Job" and owner.get("name") == job_name:
            return True
    return False


def _workload_matches(item: dict[str, Any], job_name: str) -> bool:
    if _object_name(item, "Workload") == job_name or _owner_matches(item, job_name):
        return True
    spec = item.get("spec", {})
    pod_sets = spec.get("podSets", []) if isinstance(spec, dict) else None
    if not isinstance(pod_sets, list):
        raise HeldoutLaunchError("Kueue Workload podSets are invalid")
    for pod_set in pod_sets:
        template = pod_set.get("template", {}) if isinstance(pod_set, dict) else None
        metadata = template.get("metadata", {}) if isinstance(template, dict) else None
        labels = metadata.get("labels", {}) if isinstance(metadata, dict) else None
        if not isinstance(labels, dict):
            raise HeldoutLaunchError("Kueue Workload PodSet labels are invalid")
        if (
            labels.get("job-name") == job_name
            or labels.get("batch.kubernetes.io/job-name") == job_name
        ):
            return True
    return False


def _workload_binds_created_job(item: dict[str, Any], job_name: str, job_uid: str) -> bool:
    """Require both immutable Kueue UID label and exact Kubernetes Job ownership."""
    metadata = _metadata(item, "Workload")
    labels = metadata.get("labels")
    owners = metadata.get("ownerReferences")
    if not isinstance(labels, dict) or not isinstance(owners, list):
        raise HeldoutLaunchError("Workload ownership metadata is invalid")
    if labels.get("kueue.x-k8s.io/job-uid") != job_uid:
        return False
    for owner in owners:
        if not isinstance(owner, dict):
            raise HeldoutLaunchError("Workload owner reference is invalid")
        if (
            owner.get("apiVersion") == "batch/v1"
            and owner.get("kind") == "Job"
            and owner.get("name") == job_name
            and owner.get("uid") == job_uid
            and owner.get("controller") is True
        ):
            return True
    return False


def _scoped_items(
    cluster: Cluster,
    resource: str,
    namespace: str,
    *,
    label: str,
    field_selector: str | None = None,
    label_selector: str | None = None,
) -> list[dict[str, Any]]:
    """Read one bounded server-side Kubernetes selection, never a namespace census."""
    if (field_selector is None) == (label_selector is None):
        raise HeldoutLaunchError("Kubernetes list must have exactly one scoped selector")
    return _list_items(
        cluster.list(
            resource,
            namespace,
            field_selector=field_selector,
            label_selector=label_selector,
        ),
        label,
    )


def _merge_scoped_items(lists: list[list[dict[str, Any]]], *, label: str) -> list[dict[str, Any]]:
    """Union multiple exact label selections without silently accepting drift."""
    merged: dict[str, dict[str, Any]] = {}
    for items in lists:
        for item in items:
            name = _object_name(item, label)
            existing = merged.get(name)
            if existing is not None and existing != item:
                raise HeldoutLaunchError(
                    f"{label} changed between scoped Kubernetes reads; reconcile before launch"
                )
            merged[name] = item
    return [merged[name] for name in sorted(merged)]


def _owned_pods(cluster: Cluster, packet: LaunchPacket) -> list[dict[str, Any]]:
    """Read the two Kubernetes Job-label spellings used by supported clusters."""
    pods = _merge_scoped_items(
        [
            _scoped_items(
                cluster,
                "pods",
                packet.namespace,
                label="Pod",
                label_selector=f"job-name={packet.job_name}",
            ),
            _scoped_items(
                cluster,
                "pods",
                packet.namespace,
                label="Pod",
                label_selector=f"batch.kubernetes.io/job-name={packet.job_name}",
            ),
        ],
        label="Pod",
    )
    if any(not _owner_matches(item, packet.job_name) for item in pods):
        raise HeldoutLaunchError("scoped Pod read returned an object not owned by the Job")
    return pods


def _output_exists(path: str) -> bool:
    target = Path(path)
    root = Path("/mnt/sfs/jobs")
    try:
        root_stat = root.lstat()
    except OSError as exc:
        raise HeldoutLaunchError("shared SFS root cannot be inspected") from exc
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise HeldoutLaunchError("shared SFS root is not a real directory")
    try:
        target.lstat()
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise HeldoutLaunchError("evaluation output cannot be inspected") from exc
    return True


def _ledger_contains_identity(value: Any, identity_sha256: str) -> bool:
    if isinstance(value, dict):
        if value.get("evaluation_identity_sha256") == identity_sha256:
            return True
        identity = value.get("evaluation_identity")
        if isinstance(identity, dict) and _canonical_digest(identity) == identity_sha256:
            return True
        return any(_ledger_contains_identity(item, identity_sha256) for item in value.values())
    if isinstance(value, list):
        return any(_ledger_contains_identity(item, identity_sha256) for item in value)
    return False


def duplicate_census(
    package: Package,
    *,
    cluster: Cluster,
    database: Database,
    output_exists: Callable[[str], bool] = _output_exists,
) -> dict[str, int]:
    """Fail closed if any create-once or complete-identity duplicate exists."""
    packet = package.packet
    jobs = _scoped_items(
        cluster,
        "jobs.batch",
        packet.namespace,
        label="Job",
        field_selector=f"metadata.name={packet.job_name}",
    )
    if any(_object_name(item, "Job") == packet.job_name for item in jobs):
        raise HeldoutLaunchError(
            "exact held-out evaluator Job already exists; reconcile, never replay"
        )
    config_maps = _scoped_items(
        cluster,
        "configmaps",
        packet.namespace,
        label="ConfigMap",
        field_selector=f"metadata.name={packet.config_map_name}",
    )
    if any(_object_name(item, "ConfigMap") == packet.config_map_name for item in config_maps):
        raise HeldoutLaunchError(
            "exact held-out evaluator ConfigMap already exists; reconcile, never replay"
        )
    pods = _owned_pods(cluster, packet)
    if any(_owner_matches(item, packet.job_name) for item in pods):
        raise HeldoutLaunchError(
            "owned held-out evaluator Pod already exists; reconcile, never replay"
        )
    try:
        destination_exists = output_exists(packet.output_root)
    except HeldoutLaunchError:
        raise
    except Exception as exc:
        raise HeldoutLaunchError("evaluation output duplicate check failed") from exc
    if destination_exists:
        raise HeldoutLaunchError(
            "held-out evaluator output already exists; reconcile, never replay"
        )
    try:
        database_exists = database.exists(packet.database)
    except Exception as exc:
        raise HeldoutLaunchError("evaluation database duplicate check failed") from exc
    if database_exists:
        raise HeldoutLaunchError(
            "held-out evaluator database already exists; reconcile, never replay"
        )
    ledger = _load_json(packet.files["evaluation_ledger"], "evaluation ledger")
    if _ledger_contains_identity(ledger, packet.identity_sha256):
        raise HeldoutLaunchError("evaluation ledger already contains this complete identity")
    return {
        "jobs": len(jobs),
        "config_maps": len(config_maps),
        "pods": len(pods),
    }


def _objects_from_response(value: Any, label: str) -> list[dict[str, Any]]:
    mapping = _require_mapping(value, label)
    if mapping.get("kind") == "List":
        return _list_items(mapping, label)
    return [mapping]


def _response_object(
    response: Any,
    *,
    api_version: str,
    kind: str,
    namespace: str,
    name: str,
    label: str,
) -> dict[str, Any]:
    matches = [
        item
        for item in _objects_from_response(response, label)
        if item.get("apiVersion") == api_version
        and item.get("kind") == kind
        and _namespace_name(item, label) == (namespace, name)
    ]
    if len(matches) != 1:
        raise HeldoutLaunchError(f"{label} did not return the exact expected object")
    return matches[0]


def _contains(actual: Any, expected: Any) -> bool:
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            (key in actual and _contains(actual[key], value)) or (key not in actual and value == "")
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(
                _contains(actual_item, expected_item)
                for actual_item, expected_item in zip(actual, expected, strict=True)
            )
        )
    return actual == expected


def _validate_server_preview(response: dict[str, Any], package: Package) -> str:
    packet = package.packet
    job = _response_object(
        response,
        api_version="batch/v1",
        kind="Job",
        namespace=packet.namespace,
        name=packet.job_name,
        label="server dry-run",
    )
    config_map = _response_object(
        response,
        api_version="v1",
        kind="ConfigMap",
        namespace=packet.namespace,
        name=packet.config_map_name,
        label="server dry-run",
    )
    expected_reads_postgres = require_postgres_client_label(package.job, label="sealed packet Job")
    observed_reads_postgres = require_postgres_client_label(job, label="server-rendered Job")
    if observed_reads_postgres != expected_reads_postgres:
        raise HeldoutLaunchError(
            "server-rendered Job changed the rollout PostgreSQL dependency contract"
        )
    annotations = _metadata(job, "server-rendered Job").get("annotations")
    if (
        not isinstance(annotations, dict)
        or annotations.get(FAILURE_ALERT_ANNOTATION) != FAILURE_ALERT_OFF
    ):
        raise HeldoutLaunchError("server-rendered root Job is missing failure-alerts off")
    rendered_spec = job.get("spec")
    rendered_template = rendered_spec.get("template") if isinstance(rendered_spec, dict) else None
    rendered_pod = rendered_template.get("spec") if isinstance(rendered_template, dict) else None
    if not isinstance(rendered_pod, dict):
        raise HeldoutLaunchError("server-rendered Job Pod template is invalid")
    _assert_cpu_only(rendered_pod)
    try:
        stable_job = stable_job_preview(job)
        expected_job = stable_job_preview(package.job)
    except ValueError as exc:
        raise HeldoutLaunchError("server-rendered Job has no stable preview form") from exc
    if not _contains(stable_job, expected_job):
        raise HeldoutLaunchError("server-rendered Job differs from the sealed packet")
    stable_config_map = {
        "apiVersion": config_map.get("apiVersion"),
        "kind": config_map.get("kind"),
        "metadata": {
            "name": packet.config_map_name,
            "namespace": packet.namespace,
        },
        "immutable": config_map.get("immutable"),
        "data": config_map.get("data"),
    }
    expected_config_map = {
        "apiVersion": package.config_map.get("apiVersion"),
        "kind": package.config_map.get("kind"),
        "metadata": {
            "name": packet.config_map_name,
            "namespace": packet.namespace,
        },
        "immutable": package.config_map.get("immutable"),
        "data": package.config_map.get("data"),
    }
    if stable_config_map != expected_config_map:
        raise HeldoutLaunchError("server-rendered ConfigMap differs from sealed packet")
    return _canonical_digest({"job": stable_job, "config_map": stable_config_map})


def _write_new_record(path: Path, value: dict[str, Any]) -> None:
    if path.is_symlink() or path.exists():
        raise HeldoutLaunchError("create journal already exists; reconcile, never retry")
    if not path.parent.is_dir():
        raise HeldoutLaunchError("create journal parent does not exist")
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except OSError as exc:
        raise HeldoutLaunchError("could not create durable create intent") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    # A synced file is not enough on its own: persist the newly-created name
    # before the one mutation boundary becomes reachable.
    try:
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except OSError as exc:
        raise HeldoutLaunchError("could not durably record create intent") from exc


def _append_record(path: Path, value: dict[str, Any]) -> None:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_APPEND)
    except OSError:
        return
    with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _created_name_observation(cluster: Cluster, packet: LaunchPacket) -> dict[str, bool]:
    """Best-effort readback after uncertain create; never retries mutation."""
    observed: dict[str, bool] = {}
    for label, resource, name in (
        ("job", "jobs.batch", packet.job_name),
        ("config_map", "configmaps", packet.config_map_name),
    ):
        try:
            items = _scoped_items(
                cluster,
                resource,
                packet.namespace,
                label=resource,
                field_selector=f"metadata.name={name}",
            )
            observed[label] = any(_object_name(item, resource) == name for item in items)
        except Exception:
            observed[label] = False
    return observed


def launch_once(
    packet_path: Path,
    *,
    cluster: Cluster,
    database: Database,
    journal: Path,
    output_exists: Callable[[str], bool] = _output_exists,
) -> dict[str, Any]:
    """Preview and create exactly one immutable CPU evaluator bundle.

    This function has one mutation boundary: ``cluster.create_once``.  It is
    deliberately never retried, including when its response is uncertain.
    """
    if journal.exists() or journal.is_symlink():
        raise HeldoutLaunchError("create journal already exists; reconcile, never retry")
    package = build_package(packet_path)
    first_census = duplicate_census(
        package, cluster=cluster, database=database, output_exists=output_exists
    )
    first_preview = _validate_server_preview(
        cluster.server_dry_run(package.packet.namespace, package.bundle), package
    )
    second_preview = _validate_server_preview(
        cluster.server_dry_run(package.packet.namespace, package.bundle), package
    )
    if first_preview != second_preview:
        raise HeldoutLaunchError("server dry-run changed across identical previews")
    final_census = duplicate_census(
        package, cluster=cluster, database=database, output_exists=output_exists
    )
    intent = {
        "schema": "cyber_fleet_heldout_create_intent_v1",
        "state": "KUBECTL_CREATE_INTENT_DO_NOT_RETRY",
        "packet_sha256": _file_sha256(package.packet.path),
        "evaluation_identity_sha256": package.packet.identity_sha256,
        "comparison_protocol_sha256": package.packet.identity["comparison_protocol_sha256"],
        "protocol_id": package.packet.identity["protocol_id"],
        "arm_id": package.packet.identity["arm_id"],
        "namespace": package.packet.namespace,
        "job_name": package.packet.job_name,
        "config_map_name": package.packet.config_map_name,
        "output_root": package.packet.output_root,
        "database": package.packet.database,
        "server_preview_sha256": first_preview,
        "first_duplicate_census": first_census,
        "final_duplicate_census": final_census,
    }
    _write_new_record(journal, intent)
    try:
        created = cluster.create_once(package.packet.namespace, package.bundle)
        job = _response_object(
            created,
            api_version="batch/v1",
            kind="Job",
            namespace=package.packet.namespace,
            name=package.packet.job_name,
            label="create response",
        )
        config_map = _response_object(
            created,
            api_version="v1",
            kind="ConfigMap",
            namespace=package.packet.namespace,
            name=package.packet.config_map_name,
            label="create response",
        )
        expected_reads_postgres = require_postgres_client_label(
            package.job, label="sealed packet Job"
        )
        observed_reads_postgres = require_postgres_client_label(job, label="created Job")
        if observed_reads_postgres != expected_reads_postgres:
            raise HeldoutLaunchError(
                "created Job changed the rollout PostgreSQL dependency contract"
            )
        annotations = _metadata(job, "created Job").get("annotations")
        uid = _metadata(job, "created Job").get("uid")
        config_map_uid = _metadata(config_map, "created ConfigMap").get("uid")
        if (
            not isinstance(annotations, dict)
            or annotations.get(FAILURE_ALERT_ANNOTATION) != FAILURE_ALERT_OFF
            or not isinstance(uid, str)
            or KUBERNETES_UID.fullmatch(uid) is None
            or not isinstance(config_map_uid, str)
            or KUBERNETES_UID.fullmatch(config_map_uid) is None
        ):
            raise HeldoutLaunchError("create response lacks exact root alert annotation or UID")
        created_spec = job.get("spec")
        created_template = created_spec.get("template") if isinstance(created_spec, dict) else None
        created_pod = created_template.get("spec") if isinstance(created_template, dict) else None
        if not isinstance(created_pod, dict):
            raise HeldoutLaunchError("created Job Pod template is invalid")
        _assert_cpu_only(created_pod)
    except Exception as exc:
        observed = _created_name_observation(cluster, package.packet)
        _append_record(
            journal,
            {
                "state": "KUBECTL_CREATE_RESPONSE_UNCERTAIN_DO_NOT_RETRY",
                "observed_exact_names": observed,
            },
        )
        if isinstance(exc, HeldoutLaunchError):
            raise
        raise HeldoutLaunchError(
            "create response is uncertain; reconcile and never retry"
        ) from None
    result = {
        "submitted": True,
        "gpus": 0,
        "job_name": package.packet.job_name,
        "job_uid": uid,
        "config_map_name": package.packet.config_map_name,
        "config_map_uid": config_map_uid,
        "evaluation_identity_sha256": package.packet.identity_sha256,
        "comparison_protocol_sha256": package.packet.identity["comparison_protocol_sha256"],
    }
    _append_record(journal, {"state": "KUBECTL_CREATE_RESPONSE", **result})
    return result


def _terminal_condition(job: dict[str, Any]) -> str:
    status = job.get("status")
    conditions = status.get("conditions", []) if isinstance(status, dict) else None
    if not isinstance(conditions, list):
        raise HeldoutLaunchError("created Job has no readable conditions")
    terminal = {
        condition.get("type")
        for condition in conditions
        if isinstance(condition, dict)
        and condition.get("status") == "True"
        and condition.get("type") in {"Complete", "Failed"}
    }
    if len(terminal) != 1:
        raise HeldoutLaunchError(
            "held-out evaluator is not terminal or has contradictory conditions"
        )
    return terminal.pop()


def _terminal_count(job: dict[str, Any], field: str) -> int:
    status = job.get("status")
    if not isinstance(status, dict):
        raise HeldoutLaunchError("created Job status is invalid")
    value = status.get(field, 0)
    if type(value) is not int or value < 0:
        raise HeldoutLaunchError(f"created Job {field} count is invalid")
    return value


def _score_blind_summary(value: Any) -> dict[str, Any]:
    summary = _require_mapping(value, "score-blind database summary")
    expected = {
        "total",
        "local_results",
        "by_state",
        "by_serving_block",
        "stale_active",
        "plan_sha256",
    }
    if set(summary) != expected:
        raise HeldoutLaunchError("score-blind database summary shape is invalid")
    if any(
        type(summary[field]) is not int or summary[field] < 0
        for field in ("total", "local_results", "stale_active")
    ):
        raise HeldoutLaunchError("score-blind database summary count is invalid")
    by_state = summary["by_state"]
    if not isinstance(by_state, dict) or any(
        not isinstance(name, str) or type(count) is not int or count < 0
        for name, count in by_state.items()
    ):
        raise HeldoutLaunchError("score-blind database state counts are invalid")
    by_serving_block = summary["by_serving_block"]
    if not isinstance(by_serving_block, list):
        raise HeldoutLaunchError("score-blind database route counts are invalid")
    route_counts: list[dict[str, Any]] = []
    for row in by_serving_block:
        if (
            not isinstance(row, dict)
            or set(row) != {"serving_block", "state", "count"}
            or not isinstance(row["serving_block"], str)
            or not isinstance(row["state"], str)
            or type(row["count"]) is not int
            or row["count"] < 0
        ):
            raise HeldoutLaunchError("score-blind database route counts are invalid")
        route_counts.append(
            {
                "serving_block": row["serving_block"],
                "state": row["state"],
                "count": row["count"],
            }
        )
    plan = summary["plan_sha256"]
    if plan is not None and not isinstance(plan, str):
        raise HeldoutLaunchError("score-blind database plan identity is invalid")
    return {
        "total": summary["total"],
        "local_results": summary["local_results"],
        "by_state": dict(sorted(by_state.items())),
        "by_serving_block": sorted(
            route_counts, key=lambda row: (row["serving_block"], row["state"])
        ),
        "stale_active": summary["stale_active"],
        "plan_sha256": plan,
    }


def _absent_database_summary() -> dict[str, Any]:
    """Represent a twice-observed missing ledger without inventing rollout rows."""
    return _score_blind_summary(
        {
            "total": 0,
            "local_results": 0,
            "by_state": {
                state: 0
                for state in (
                    "accepted",
                    "claimed",
                    "grading",
                    "pending",
                    "retry_review",
                    "running",
                    "terminal",
                )
            },
            "by_serving_block": [],
            "stale_active": 0,
            "plan_sha256": None,
        }
    )


def _terminal_resource_rows(items: list[dict[str, Any]], *, kind: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in items:
        metadata = _metadata(item, kind)
        name = metadata.get("name")
        uid = metadata.get("uid")
        if (
            not isinstance(name, str)
            or not isinstance(uid, str)
            or KUBERNETES_UID.fullmatch(uid) is None
        ):
            raise HeldoutLaunchError(f"{kind} terminal identity is invalid")
        status = item.get("status", {})
        phase = status.get("phase") if isinstance(status, dict) else None
        rows.append({"name": name, "uid": uid, "phase": phase if isinstance(phase, str) else None})
    return sorted(rows, key=lambda row: (row["name"], row["uid"]))


def terminal_receipt_path(
    output_root: str,
    job_name: str,
    *,
    fallback_root: Path = Path("/mnt/sfs/jobs"),
) -> Path:
    """Choose a no-overwrite receipt path even when evaluation never made output."""
    if not isinstance(output_root, str) or not output_root:
        raise HeldoutLaunchError("terminal output root is invalid")
    if not isinstance(job_name, str) or KUBERNETES_NAME.fullmatch(job_name) is None:
        raise HeldoutLaunchError("terminal Job name is invalid")
    output = Path(output_root)
    if output.is_symlink():
        raise HeldoutLaunchError("terminal output root must not be a symlink")
    if output.is_dir():
        return output / "TERMINAL_OBSERVATION.json"
    if output.exists():
        raise HeldoutLaunchError("terminal output root exists but is not a directory")
    if fallback_root.is_symlink() or not fallback_root.is_dir():
        raise HeldoutLaunchError("terminal fallback root is not an exact directory")
    return fallback_root / f"{job_name}-TERMINAL_OBSERVATION.json"


def collect_terminal(
    packet_path: Path,
    *,
    cluster: Cluster,
    database: Database,
    receipt_path: Path,
    output_exists: Callable[[str], bool] = _output_exists,
) -> dict[str, Any]:
    """Record one score-blind terminal observation; never launch, retry, or score."""
    if receipt_path.exists() or receipt_path.is_symlink():
        raise HeldoutLaunchError("terminal receipt already exists; reconcile, never overwrite")
    if not receipt_path.parent.is_dir():
        raise HeldoutLaunchError("terminal receipt parent does not exist")
    package = build_package(packet_path)
    packet = package.packet
    job = cluster.get("jobs.batch", packet.namespace, packet.job_name)
    config_map = cluster.get("configmaps", packet.namespace, packet.config_map_name)
    if _namespace_name(job, "created Job") != (packet.namespace, packet.job_name):
        raise HeldoutLaunchError("terminal Job identity differs from packet")
    if _namespace_name(config_map, "created ConfigMap") != (
        packet.namespace,
        packet.config_map_name,
    ):
        raise HeldoutLaunchError("terminal ConfigMap identity differs from packet")
    job_annotations = _metadata(job, "created Job").get("annotations")
    job_uid = _metadata(job, "created Job").get("uid")
    config_map_uid = _metadata(config_map, "created ConfigMap").get("uid")
    if (
        not isinstance(job_annotations, dict)
        or job_annotations.get(FAILURE_ALERT_ANNOTATION) != FAILURE_ALERT_OFF
        or not isinstance(job_uid, str)
        or KUBERNETES_UID.fullmatch(job_uid) is None
        or not isinstance(config_map_uid, str)
        or KUBERNETES_UID.fullmatch(config_map_uid) is None
    ):
        raise HeldoutLaunchError("terminal resource identity or root alert annotation is invalid")
    terminal = _terminal_condition(job)
    workloads = _scoped_items(
        cluster,
        "workloads.kueue.x-k8s.io",
        packet.namespace,
        label="Workload",
        label_selector=f"kueue.x-k8s.io/job-uid={job_uid}",
    )
    if any(
        not _workload_matches(item, packet.job_name)
        or not _workload_binds_created_job(item, packet.job_name, job_uid)
        for item in workloads
    ):
        raise HeldoutLaunchError("scoped Workload read differs from the created Job")
    pods = _owned_pods(cluster, packet)
    try:
        database_exists = database.exists(packet.database)
        if database_exists:
            summary = _score_blind_summary(database.summary(packet.database))
        else:
            if database.exists(packet.database):
                raise HeldoutLaunchError("terminal database appeared between absence observations")
            summary = _absent_database_summary()
        destination_exists = output_exists(packet.output_root)
    except HeldoutLaunchError:
        raise
    except Exception as exc:
        raise HeldoutLaunchError("terminal reconciliation reads failed") from exc
    unresolved = sum(
        count
        for state, count in summary["by_state"].items()
        if state in {"pending", "claimed", "running", "grading", "retry_review"}
    )
    receipt = {
        "schema": TERMINAL_SCHEMA,
        "observed_at": datetime.now(UTC).isoformat(),
        "evaluation_identity_sha256": packet.identity_sha256,
        "comparison_protocol_sha256": packet.identity["comparison_protocol_sha256"],
        "protocol_id": packet.identity["protocol_id"],
        "arm_id": packet.identity["arm_id"],
        "job": {
            "name": packet.job_name,
            "uid": job_uid,
            "terminal_condition": terminal,
            "succeeded": _terminal_count(job, "succeeded"),
            "failed": _terminal_count(job, "failed"),
        },
        "config_map": {"name": packet.config_map_name, "uid": config_map_uid},
        "workloads": _terminal_resource_rows(workloads, kind="Workload"),
        "pods": _terminal_resource_rows(pods, kind="Pod"),
        "database": {"name": packet.database, "summary": summary},
        "output_root": {"path": packet.output_root, "exists": destination_exists},
        "decision": {
            "capability_result_status": "not_interpreted",
            "score_blind_reconciliation_required": unresolved > 0,
            "unresolved_cells": unresolved,
            "rollout_retry_performed": False,
            "score_read_or_generated": False,
        },
        "privacy": {
            "prompts_responses_flags_rewards_or_trace_content_included": False,
            "score_values_included": False,
            "credentials_included": False,
        },
    }
    receipt["sha256"] = _canonical_digest(receipt)
    _write_new_record(receipt_path, receipt)
    return receipt


class KubectlCluster:
    """No-shell production adapter; it exposes one create request only."""

    def __init__(self, context: str, *, binary: str = "kubectl") -> None:
        if (
            not isinstance(context, str)
            or not context
            or context.startswith("-")
            or re.fullmatch(r"[-A-Za-z0-9_.:@/]+", context) is None
        ):
            raise HeldoutLaunchError("an explicit Kubernetes context is required")
        if not isinstance(binary, str) or not binary or binary.startswith("-"):
            raise HeldoutLaunchError("kubectl binary is invalid")
        self.context = context
        self.binary = binary

    def _run(
        self, namespace: str, args: list[str], bundle: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        command = [
            self.binary,
            "--context",
            self.context,
            "--request-timeout=60s",
            "--namespace",
            namespace,
            *args,
        ]
        try:
            result = subprocess.run(
                command,
                input=(
                    json.dumps(bundle, sort_keys=True, separators=(",", ":")) if bundle else None
                ),
                text=True,
                capture_output=True,
                timeout=75,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            raise HeldoutLaunchError(
                "kubectl transport failed; do not infer cluster state"
            ) from None
        if result.returncode:
            raise HeldoutLaunchError("kubectl operation failed; private server output suppressed")
        try:
            decoder = json.JSONDecoder()
            values: list[dict[str, Any]] = []
            position = 0
            while position < len(result.stdout):
                while position < len(result.stdout) and result.stdout[position].isspace():
                    position += 1
                if position == len(result.stdout):
                    break
                value, position = decoder.raw_decode(result.stdout, position)
                values.append(_require_mapping(value, "kubectl response"))
        except json.JSONDecodeError as exc:
            raise HeldoutLaunchError("kubectl returned invalid JSON") from exc
        if not values:
            raise HeldoutLaunchError("kubectl returned invalid JSON")
        if len(values) == 1:
            return values[0]
        return {"apiVersion": "v1", "kind": "List", "items": values}

    def list(
        self,
        resource: str,
        namespace: str,
        *,
        field_selector: str | None = None,
        label_selector: str | None = None,
    ) -> dict[str, Any]:
        if resource not in {"jobs.batch", "configmaps", "workloads.kueue.x-k8s.io", "pods"}:
            raise HeldoutLaunchError("unsupported held-out duplicate-check resource")
        if (field_selector is None) == (label_selector is None):
            raise HeldoutLaunchError("Kubernetes list must have exactly one scoped selector")
        if field_selector is not None:
            match = re.fullmatch(
                r"metadata\.name=([a-z0-9](?:[-a-z0-9]{0,61}[a-z0-9])?)", field_selector
            )
            if match is None:
                raise HeldoutLaunchError("Kubernetes field selector is invalid")
            return self._run(
                namespace,
                ["get", resource, f"--field-selector={field_selector}", "--output=json"],
            )
        assert label_selector is not None
        match = re.fullmatch(
            r"(?:job-name|batch\.kubernetes\.io/job-name)=([a-z0-9](?:[-a-z0-9]{0,61}[a-z0-9])?)"
            r"|kueue\.x-k8s\.io/job-uid="
            r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})",
            label_selector,
        )
        if match is None:
            raise HeldoutLaunchError("Kubernetes label selector is invalid")
        return self._run(
            namespace,
            ["get", resource, f"--selector={label_selector}", "--output=json"],
        )

    def get(self, resource: str, namespace: str, name: str) -> dict[str, Any]:
        if resource not in {"jobs.batch", "configmaps"}:
            raise HeldoutLaunchError("unsupported held-out terminal resource")
        _require_name(name, "resource name")
        return self._run(namespace, ["get", resource, name, "--output=json"])

    def server_dry_run(self, namespace: str, bundle: dict[str, Any]) -> dict[str, Any]:
        return self._run(
            namespace,
            ["create", "--dry-run=server", "--filename=-", "--output=json"],
            bundle,
        )

    def create_once(self, namespace: str, bundle: dict[str, Any]) -> dict[str, Any]:
        return self._run(namespace, ["create", "--filename=-", "--output=json"], bundle)


class PostgresDatabase:
    """Read-only campaign database boundary. Credentials remain in one env var."""

    def __init__(self, dsn_env: str = "ROLLOUT_DATABASE_URL") -> None:
        if not isinstance(dsn_env, str) or not dsn_env:
            raise HeldoutLaunchError("database environment variable name is invalid")
        self.dsn_env = dsn_env

    def _dsn(self) -> str:
        value = os.environ.get(self.dsn_env)
        if not value:
            raise HeldoutLaunchError("database environment is missing")
        return value

    def _database_dsn(self, database: str) -> str:
        if not isinstance(database, str) or DATABASE_NAME.fullmatch(database) is None:
            raise HeldoutLaunchError("database name is invalid")
        original = urlsplit(self._dsn())
        if (
            original.scheme not in {"postgres", "postgresql"}
            or not original.netloc
            or original.fragment
        ):
            raise HeldoutLaunchError("database environment is not a supported PostgreSQL URI")
        query_keys = {
            key.casefold() for key, _ in parse_qsl(original.query, keep_blank_values=True)
        }
        if query_keys & {"database", "dbname"}:
            raise HeldoutLaunchError(
                "database environment must select its database only by URI path"
            )
        return urlunsplit(
            (
                original.scheme,
                original.netloc,
                "/" + quote(database, safe=""),
                original.query,
                "",
            )
        )

    def exists(self, database: str) -> bool:
        try:
            import psycopg

            for attempt in range(3):
                try:
                    connection = psycopg.connect(self._dsn(), connect_timeout=5)
                except psycopg.errors.ConnectionTimeout:
                    if attempt == 2:
                        raise
                    time.sleep(5)
                    continue
                with connection:
                    row = connection.execute(
                        "SELECT 1 FROM pg_database WHERE datname = %s", (database,)
                    ).fetchone()
                return row is not None
        except HeldoutLaunchError:
            raise
        except Exception:
            raise HeldoutLaunchError("database duplicate check failed") from None

    def summary(self, database: str) -> dict[str, Any]:
        try:
            from evals.fleet import rollout_postgres

            return rollout_postgres.summary(self._database_dsn(database))
        except HeldoutLaunchError:
            raise
        except Exception:
            raise HeldoutLaunchError("score-blind database summary failed") from None

    def cell_status(
        self, database: str, *, task_version_id: str, model_revision: str, attempt: int
    ) -> dict[str, Any]:
        try:
            from evals.fleet import rollout_postgres

            return rollout_postgres.cell_status(
                self._database_dsn(database),
                task_version_id=task_version_id,
                model_revision=model_revision,
                attempt=attempt,
            )
        except HeldoutLaunchError:
            raise
        except Exception:
            raise HeldoutLaunchError("score-blind database cell status failed") from None
