"""Caller-side preview and create-once rail for the bounded prod10 operator."""

from __future__ import annotations

import json
import os
import stat
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from training import dev_cleanup_observer as cleanup
from training import skyrl_prod10_operator as operator
from training import skyrl_prod9_direct as direct

from .direct_submit import Kubectl
from .jobs import JobsError, digest
from .skyrl_prod10_operator_job import (
    NAMESPACE,
    OperatorPackage,
    validate_operator_package,
    validate_server_response,
)

PREVIEW_SCHEMA = "cyber_skyrl_prod10_operator_preview_v1"
DUPLICATE_SCHEMA = "cyber_skyrl_prod10_operator_duplicate_absence_v2"
CREATED_SCHEMA = "cyber_skyrl_prod10_operator_created_v1"
RESULT_SCHEMA = "cyber_skyrl_prod10_operator_launch_result_v1"
_RESOURCES = (
    "configmaps",
    "jobs.batch",
    "pods",
    "workloads.kueue.x-k8s.io",
    "rayjobs.ray.io",
    "rayclusters.ray.io",
)


def _seal(value: dict[str, Any]) -> dict[str, Any]:
    body = {key: item for key, item in value.items() if key != "sha256"}
    return {**body, "sha256": "sha256:" + digest(body)}


def _validate_seal(value: object, schema: str) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema") != schema or value != _seal(value):
        raise JobsError("prod10 operator launcher evidence is invalid")
    return value


def _validate_config_map_response(
    actual: dict[str, Any], expected: dict[str, Any], *, require_uid: bool
) -> dict[str, Any]:
    if (
        not isinstance(actual, dict)
        or actual.get("apiVersion") != "v1"
        or actual.get("kind") != "ConfigMap"
        or actual.get("immutable") is not True
        or actual.get("binaryData") != expected.get("binaryData")
        or actual.get("data") not in (None, {})
    ):
        raise JobsError("prod10 operator ConfigMap response changed")
    metadata = actual.get("metadata")
    expected_metadata = expected.get("metadata")
    if (
        not isinstance(metadata, dict)
        or not isinstance(expected_metadata, dict)
        or metadata.get("name") != expected_metadata.get("name")
        or metadata.get("namespace") != NAMESPACE
        or metadata.get("annotations") != expected_metadata.get("annotations")
    ):
        raise JobsError("prod10 operator ConfigMap metadata changed")
    if require_uid:
        try:
            UUID(metadata.get("uid"))
        except (TypeError, ValueError) as exc:
            raise JobsError("prod10 operator ConfigMap response omitted its UID") from exc
    return {
        "name": metadata["name"],
        "uid": metadata.get("uid"),
        "server_sha256": "sha256:" + digest(actual),
    }


def server_previews(
    package: OperatorPackage,
    *,
    factory: type[Kubectl] = Kubectl,
) -> list[dict[str, Any]]:
    proof = validate_operator_package(package)
    results = []
    for context in (direct.DEV_CONTEXT, direct.PROD_CONTEXT):
        client = factory(context)
        source = client.dry_run(package.source_config_map)
        packet = client.dry_run(package.packet_config_map)
        job = client.dry_run(package.job)
        source_proof = _validate_config_map_response(
            source, package.source_config_map, require_uid=False
        )
        packet_proof = _validate_config_map_response(
            packet, package.packet_config_map, require_uid=False
        )
        job_proof = validate_server_response(job, package.job, require_uid=False)
        results.append(
            _seal(
                {
                    "schema": PREVIEW_SCHEMA,
                    "status": "passed",
                    "context": context,
                    "namespace": NAMESPACE,
                    "name": proof["name"],
                    "packet_sha256": proof["packet_sha256"],
                    "source_sha256": proof["source_sha256"],
                    "job_manifest_sha256": proof["job_manifest_sha256"],
                    "source_config_map_server_sha256": source_proof["server_sha256"],
                    "packet_config_map_server_sha256": packet_proof["server_sha256"],
                    "job_server_sha256": job_proof["server_sha256"],
                    "failure_alerts": job_proof["failure_alerts"],
                    "priority": job_proof["priority"],
                    "queue_priority": job_proof["queue_priority"],
                    "gpus": 0,
                    "submitted": False,
                    "checked_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                }
            )
        )
    return results


def _preview_set(package: OperatorPackage, values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    proof = validate_operator_package(package)
    contexts = []
    for item in values:
        value = _validate_seal(item, PREVIEW_SCHEMA)
        if (
            value.get("status") != "passed"
            or value.get("context") not in {direct.DEV_CONTEXT, direct.PROD_CONTEXT}
            or value.get("namespace") != NAMESPACE
            or value.get("name") != proof["name"]
            or value.get("packet_sha256") != proof["packet_sha256"]
            or value.get("source_sha256") != proof["source_sha256"]
            or value.get("job_manifest_sha256") != proof["job_manifest_sha256"]
            or value.get("failure_alerts") != "off"
            or value.get("priority") != "c1"
            or value.get("queue_priority") != "q1"
            or value.get("gpus") != 0
            or value.get("submitted") is not False
        ):
            raise JobsError("prod10 operator server preview changed")
        direct._fresh_at(value.get("checked_at"))
        contexts.append(value["context"])
    if sorted(contexts) != sorted({direct.DEV_CONTEXT, direct.PROD_CONTEXT}) or len(values) != 2:
        raise JobsError("prod10 operator server preview set is incomplete")
    return values


def duplicate_proof(
    package: OperatorPackage,
    *,
    factory: type[Kubectl] = Kubectl,
) -> dict[str, Any]:
    proof = validate_operator_package(package)
    names = {
        package.source_config_map["metadata"]["name"],
        package.packet_config_map["metadata"]["name"],
        package.job["metadata"]["name"],
    }
    checked = 0
    work = [
        (context, resource)
        for context in (direct.DEV_CONTEXT, direct.PROD_CONTEXT)
        for resource in _RESOURCES
    ]

    def read(context: str, resource: str) -> tuple[str, str, dict[str, Any]]:
        client = factory(context)
        if resource == "configmaps":
            items = [
                value
                for name in sorted(names)
                if name.endswith(("-source", "-packet"))
                for value in [client.get_operator_object("configmap", name)]
                if value is not None
            ]
            return context, resource, {"kind": "ConfigMapList", "items": items}
        return context, resource, client.list_operator_resources(resource)

    # Namespace inventories are deliberately exhaustive and can be tens of
    # megabytes in production. Serialize them so one large response cannot
    # starve another kubectl transport and turn absence into a flaky result.
    with ThreadPoolExecutor(max_workers=1) as pool:
        futures = {pool.submit(read, *item): item for item in work}
        for future in as_completed(futures):
            _context, _resource, inventory = future.result()
            items = inventory.get("items") if isinstance(inventory, dict) else None
            if not isinstance(items, list):
                raise JobsError("prod10 operator duplicate inventory is invalid")
            checked += 1
            for item in items:
                metadata = item.get("metadata") if isinstance(item, dict) else None
                if not isinstance(metadata, dict):
                    raise JobsError("prod10 operator duplicate inventory is invalid")
                labels = metadata.get("labels", {})
                annotations = metadata.get("annotations", {})
                if not isinstance(labels, dict) or not isinstance(annotations, dict):
                    raise JobsError("prod10 operator duplicate inventory is invalid")
                owners = metadata.get("ownerReferences", [])
                if not isinstance(owners, list):
                    raise JobsError("prod10 operator duplicate inventory is invalid")
                values = {
                    metadata.get("name"),
                    *labels.values(),
                    *annotations.values(),
                    *(
                        owner.get("name")
                        for owner in owners
                        if isinstance(owner, dict)
                    ),
                }
                if any(
                    value == name
                    or str(value).startswith(name + "-")
                    or str(value).startswith("job-" + name + "-")
                    for value in values
                    for name in names
                ):
                    raise JobsError("prod10 operator identity already exists")
    return _seal(
        {
            "schema": DUPLICATE_SCHEMA,
            "status": "identities_absent",
            "name": proof["name"],
            "phase": proof["phase"],
            "source_sha256": proof["source_sha256"],
            "names": sorted(names),
            "contexts": [direct.DEV_CONTEXT, direct.PROD_CONTEXT],
            "kubernetes_inventories_checked": checked,
            "checked_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    )


def _validate_duplicate(package: OperatorPackage, value: dict[str, Any]) -> dict[str, Any]:
    proof = validate_operator_package(package)
    checked = _validate_seal(value, DUPLICATE_SCHEMA)
    names = sorted(
        {
            package.source_config_map["metadata"]["name"],
            package.packet_config_map["metadata"]["name"],
            package.job["metadata"]["name"],
        }
    )
    if (
        checked.get("status") != "identities_absent"
        or checked.get("name") != proof["name"]
        or checked.get("phase") != proof["phase"]
        or checked.get("source_sha256") != proof["source_sha256"]
        or checked.get("names") != names
        or checked.get("contexts") != [direct.DEV_CONTEXT, direct.PROD_CONTEXT]
        or checked.get("kubernetes_inventories_checked") != len(_RESOURCES) * 2
    ):
        raise JobsError("prod10 operator duplicate proof changed")
    direct._fresh_at(checked.get("checked_at"))
    return checked


def _write_once(path: Path, value: dict[str, Any]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        stream.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _append(path: Path, value: dict[str, Any]) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _operation_directory(path: Path) -> Path:
    if not path.is_absolute() or path.is_symlink() or not path.is_dir() or path.resolve() != path:
        raise JobsError("prod10 operator local operation directory is invalid")
    identity = path.stat()
    if identity.st_uid != os.geteuid() or stat.S_IMODE(identity.st_mode) & 0o077:
        raise JobsError("prod10 operator local operation directory ownership/mode changed")
    return path


def create_once(
    package: OperatorPackage,
    *,
    previews: list[dict[str, Any]],
    duplicates: dict[str, Any],
    operation_directory: Path,
    factory: type[Kubectl] = Kubectl,
) -> dict[str, Any]:
    package_proof = validate_operator_package(package)
    preview_set = _preview_set(package, previews)
    duplicate = _validate_duplicate(package, duplicates)
    root = _operation_directory(operation_directory)
    journal = root / "OPERATOR_CREATE.jsonl"
    if journal.exists() or journal.is_symlink():
        raise JobsError("prod10 operator create intent exists; reconcile, never retry")
    observer = cleanup.Observer(
        context=direct.PROD_CONTEXT,
        namespace=NAMESPACE,
        kind="job",
        name=package_proof["name"],
        maximum_seconds=2500,
        expected_gpus=0,
        plan_sha256=package_proof["packet_sha256"],
        manifest_sha256=package_proof["job_manifest_sha256"],
        armed_path=root / "OPERATOR_OBSERVER_ARMED.json",
        result_path=root / "OPERATOR_OBSERVER_RESULT.json",
        profile="production-operator",
        poll_seconds=2,
    )
    state: dict[str, Any] = {}

    def observe() -> None:
        try:
            state["result"] = observer.run()
        except BaseException as exc:
            state["error"] = exc

    thread = threading.Thread(target=observe, name="prod10-operator-observer", daemon=True)
    thread.start()
    deadline = datetime.now(UTC).timestamp() + 30
    while not observer.armed_path.is_file():
        if not thread.is_alive() or datetime.now(UTC).timestamp() >= deadline:
            raise JobsError("prod10 operator observer did not arm")
        threading.Event().wait(0.25)
    armed = json.loads(observer.armed_path.read_bytes())
    _write_once(
        journal,
        {
            "state": "CREATE_INTENT_DO_NOT_RETRY",
            "package": package_proof,
            "preview_sha256": [value["sha256"] for value in preview_set],
            "duplicate_sha256": duplicate["sha256"],
            "observer_sha256": armed["sha256"],
        },
    )
    client = factory(direct.PROD_CONTEXT)
    source = client.create_once(package.source_config_map)
    source_proof = _validate_config_map_response(
        source, package.source_config_map, require_uid=True
    )
    _append(journal, {"state": "SOURCE_CONFIG_MAP_CREATED", **source_proof})
    packet = client.create_once(package.packet_config_map)
    packet_proof = _validate_config_map_response(
        packet, package.packet_config_map, require_uid=True
    )
    _append(journal, {"state": "PACKET_CONFIG_MAP_CREATED", **packet_proof})
    created = client.create_once(package.job)
    job_proof = validate_server_response(created, package.job, require_uid=True)
    _append(journal, {"state": "OPERATOR_JOB_CREATED", **job_proof})
    direct._publish_creator_binding(
        armed,
        kind="job",
        name=package_proof["name"],
        plan_sha256=package_proof["packet_sha256"],
        manifest_sha256=package_proof["job_manifest_sha256"],
        uid=job_proof["uid"],
    )
    thread.join(2760)
    if thread.is_alive():
        raise JobsError("prod10 operator observer exceeded its bound")
    error = state.get("error")
    if isinstance(error, BaseException):
        raise JobsError("prod10 operator release failed") from error
    observer_result = state.get("result")
    if not isinstance(observer_result, dict) or observer_result.get("status") != "released":
        raise JobsError("prod10 operator execution was not accepted and released")
    source_delete = client.delete_operator_object_uid_once(
        "configmap", source_proof["name"], source_proof["uid"]
    )
    packet_delete = client.delete_operator_object_uid_once(
        "configmap", packet_proof["name"], packet_proof["uid"]
    )
    result = _seal(
        {
            "schema": RESULT_SCHEMA,
            "status": "operator_succeeded_and_released",
            "package": package_proof,
            "created": {
                "source_config_map": source_proof,
                "packet_config_map": packet_proof,
                "job": job_proof,
            },
            "observer": observer_result,
            "config_map_release": {
                "source_status": source_delete.get("status"),
                "packet_status": packet_delete.get("status"),
            },
            "gpus": 0,
        }
    )
    _append(journal, {"state": "TERMINAL_RESULT", **result})
    return result
