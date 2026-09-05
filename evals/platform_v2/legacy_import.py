"""Plan and submit exact legacy Fleet cyber task imports to Platform v2.

Protocol v3 binds the Fleet task row and its expected current frozen version.
The legacy v2 protocol remains available for read-only current-pointer
diagnostics, but is never write-admissible.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import stat
from collections import Counter
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx

from evals.fleet.models import FLEET_TEAM_ID

REGISTRY = "https://registry-alpha.fleetai.me"
ORCHESTRATOR = "https://orchestrator.fleetai.com"
DISCOVERY_PATH = "/.well-known/fleet-registry"
IMPORT_PATH = "/api/v1/imports/legacy-tasks"
REPOSITORY_PATH = "/api/v1/repositories"
LEGACY_IMPORT_PROTOCOL = "fleet.legacy-task-import.v2"
EXPECTED_IMPORT_PROTOCOL = "fleet.legacy-task-import.v3"
SUPPORTED_IMPORT_PROTOCOLS = {LEGACY_IMPORT_PROTOCOL, EXPECTED_IMPORT_PROTOCOL}
IDENTITY_ROSTER_SCHEMA = "chris.cyber.v2.source-roster.v1"
PLAN_SCHEMA = "fleet_platform_v2_cyber_import_plan_v3"
DEFAULT_SELECTION = Path("configs/data/fleet-a62-task-split-v1.json")
DEFAULT_IDENTITY_ROSTER = Path("configs/data/fleet-a62-task-identity-roster-v1.json")
DEFAULT_NAMESPACE = "gentle-ember-ledger"
DEFAULT_REPOSITORY = "fleet-cyber-a62-frozen"
EXPECTED_SELECTION_DIGEST = (
    "sha256:fb09668f8de77e6baee37dc1e3162a7617ca9a0a450da9ca53422c17bfbc942a"
)
EXPECTED_IDENTITY_ROSTER_SHA256 = (
    "sha256:d58f146e1c5cdc2ba372f13b079a13eab997ba7addd47d943f2ab9eceeec6bb9"
)
TAG_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$")
UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")
IMPORT_STATES = {
    "requested",
    "extracting",
    "needs_input",
    "extracted",
    "importing",
    "finalizing",
    "published",
    "failed",
}
ALLOWED_RESOLUTION_AUTHORITIES = {
    "training_task_catalog",
    "fleet_job_roster+training_environment_catalog",
}
DEPLOYED_TOPOLOGY_OMITTED_ENV_KEYS = {"cysec1-2-current-fubspot-gen"}
ROW_RECEIPT_SCHEMA = "fleet_platform_v2_cyber_import_row_receipt_v3"


class PlatformImportError(RuntimeError):
    """A safe-to-display import planning or control-plane failure."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def sha256(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def digest_without(value: dict[str, Any], field: str) -> str:
    return sha256(canonical_json({key: item for key, item in value.items() if key != field}))


def load_registry_token(path: Path | None = None) -> str:
    explicit = os.environ.get("FLEET_REGISTRY_TOKEN")
    if explicit:
        return explicit
    source = path or Path.home() / ".config/fleet/credentials.json"
    try:
        document = json.loads(source.read_text())
        token = document["registries"]["registry-alpha.fleetai.me"]["token"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise PlatformImportError("Registry alpha login is unavailable") from exc
    if not isinstance(token, str) or not token:
        raise PlatformImportError("Registry alpha login is unavailable")
    return token


def load_json_object(path: Path, operation: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError):
        raise PlatformImportError(f"{operation} is unavailable or invalid") from None
    if not isinstance(value, dict):
        raise PlatformImportError(f"{operation} is not a JSON object")
    return value


def load_selection(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        document = json.loads(raw)
    except (OSError, json.JSONDecodeError):
        raise PlatformImportError("selection is unavailable or invalid") from None
    tasks = document.get("tasks")
    if (
        set(document) != {"schema", "source", "counts", "policy", "tasks", "manifest_digest"}
        or document.get("schema") != "fleet_rl_task_split_v1"
        or not isinstance(tasks, list)
    ):
        raise PlatformImportError("selection is not a Fleet frozen split manifest")
    task_fields = {
        "task_key",
        "task_version_id",
        "task_version",
        "environment_version_id",
        "env_key",
        "env_version",
        "data_key",
        "data_version",
        "split",
        "resolution_authority",
    }
    if (
        len(tasks) != 160
        or any(not isinstance(row, dict) or set(row) != task_fields for row in tasks)
        or len({row["task_version_id"] for row in tasks}) != 160
        or len({row["task_key"] for row in tasks}) != 160
        or any(not UUID_RE.fullmatch(str(row["task_version_id"])) for row in tasks)
        or any(not UUID_RE.fullmatch(str(row["environment_version_id"])) for row in tasks)
        or any(not TAG_RE.fullmatch(str(row["task_key"])) for row in tasks)
        or any(row["resolution_authority"] not in ALLOWED_RESOLUTION_AUTHORITIES for row in tasks)
    ):
        raise PlatformImportError("selection must contain 160 unique task-version UUIDs")
    if Counter(row["split"] for row in tasks) != {"train": 130, "dev": 10, "test": 20}:
        raise PlatformImportError("selection split counts drifted")
    expected = document.get("manifest_digest")
    computed = sha256(canonical_json({k: v for k, v in document.items() if k != "manifest_digest"}))
    if expected != EXPECTED_SELECTION_DIGEST or expected != computed:
        raise PlatformImportError("selection manifest digest does not verify")
    return document


def task_tag(row: dict[str, Any], split_index: int) -> str:
    value = f"{row['split']}-{split_index:03d}-{row['task_version_id'][:8]}"
    if not TAG_RE.fullmatch(value):
        raise PlatformImportError("generated Registry tag is invalid")
    return value


def load_identity_roster(path: Path, selection: dict[str, Any]) -> dict[str, Any]:
    """Load the separately reviewed task-key to stable eval-task-ID roster."""
    try:
        raw = path.read_bytes()
        roster = json.loads(raw)
    except (OSError, json.JSONDecodeError):
        raise PlatformImportError("identity roster is unavailable or invalid") from None
    if sha256(raw) != EXPECTED_IDENTITY_ROSTER_SHA256:
        raise PlatformImportError("identity roster does not match the reviewed file digest")
    _validate_identity_roster(roster, selection)
    return roster


def _validate_identity_roster(roster: Any, selection: dict[str, Any]) -> None:
    rows = roster.get("tasks") if isinstance(roster, dict) else None
    expected_keys = [row["task_key"] for row in selection["tasks"]]
    if (
        not isinstance(roster, dict)
        or set(roster) != {"schema", "source_job_id", "tasks"}
        or roster.get("schema") != IDENTITY_ROSTER_SCHEMA
        or roster.get("source_job_id") != selection["source"]["job_id"]
        or not isinstance(rows, list)
        or len(rows) != 160
        or any(
            not isinstance(row, dict)
            or set(row) != {"task_key", "eval_task_id", "current_task_version_id"}
            or not TAG_RE.fullmatch(str(row.get("task_key")))
            or not UUID_RE.fullmatch(str(row.get("eval_task_id")))
            or not UUID_RE.fullmatch(str(row.get("current_task_version_id")))
            for row in rows
        )
        or set(row["task_key"] for row in rows) != set(expected_keys)
        or len({row["task_key"] for row in rows}) != 160
        or len({row["eval_task_id"] for row in rows}) != 160
    ):
        raise PlatformImportError("identity roster does not match the frozen 160-task cohort")


def source_identity(row: dict[str, Any], eval_task_id: str) -> dict[str, str]:
    identity = {
        "task_key": row["task_key"],
        "expected_eval_task_id": eval_task_id,
        "expected_current_task_version_id": row["task_version_id"],
    }
    if not UUID_RE.fullmatch(eval_task_id):
        raise PlatformImportError("reviewed eval task ID is not a canonical UUID")
    return identity


def import_request(
    row: dict[str, Any],
    *,
    namespace: str,
    repository: str,
    split_index: int,
    protocol: str = EXPECTED_IMPORT_PROTOCOL,
    expected_eval_task_id: str | None = None,
) -> dict[str, Any]:
    if (namespace, repository) != (DEFAULT_NAMESPACE, DEFAULT_REPOSITORY):
        raise PlatformImportError("destination differs from the frozen import campaign")
    if protocol not in SUPPORTED_IMPORT_PROTOCOLS:
        raise PlatformImportError("unsupported legacy import protocol")
    request = {
        "idempotency_key": f"chris-cyber-a62-{row['task_version_id']}",
        "task_key": row["task_key"],
        "source_team_id": FLEET_TEAM_ID,
        "destination": {
            "namespace": namespace,
            "repository": repository,
            "tag": task_tag(row, split_index),
        },
        "modality_decision": {
            "task_key": row["task_key"],
            "modality": "tool_use",
            "reason": "Fleet cyber blackbox task with server-enforced agent tool capabilities.",
        },
    }
    if protocol == EXPECTED_IMPORT_PROTOCOL:
        if expected_eval_task_id is None:
            raise PlatformImportError("v3 import requires a reviewed eval task ID")
        request.update(source_identity(row, expected_eval_task_id))
    return request


def _json(response: httpx.Response, operation: str) -> dict[str, Any]:
    if response.status_code >= 400:
        if response.status_code == 403 and operation == "Legacy task import":
            raise PlatformImportError(
                "Legacy task import requires destination Repository write access"
            )
        raise PlatformImportError(f"{operation} failed with HTTP {response.status_code}")
    try:
        value = response.json()
    except ValueError:
        raise PlatformImportError(f"{operation} returned non-JSON data") from None
    if not isinstance(value, dict):
        raise PlatformImportError(f"{operation} returned an unexpected response")
    return value


def _request(
    client: httpx.Client,
    method: str,
    url: str,
    operation: str,
    *,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        response = client.request(method, url, json=payload)
    except httpx.HTTPError:
        raise PlatformImportError(f"{operation} failed before a valid response") from None
    return _json(response, operation)


def _current_identity(
    fleet_client: httpx.Client, source: dict[str, Any]
) -> tuple[str | None, str | None]:
    current = _request(
        fleet_client,
        "GET",
        f"{ORCHESTRATOR}/v1/tasks/{quote(source['task_key'], safe='')}",
        "v1 task identity read",
    )
    task_key_values = {
        value for value in (current.get("task_key"), current.get("key")) if value is not None
    }
    if task_key_values not in (set(), {source["task_key"]}):
        raise PlatformImportError("v1 task identity response named a different task")
    id_values = {
        value for value in (current.get("eval_task_id"), current.get("id")) if value is not None
    }
    if len(id_values) > 1:
        raise PlatformImportError("v1 task identity response named conflicting task IDs")
    eval_task_id = next(iter(id_values), None)
    version_id = current.get("eval_task_version_id")
    return (
        eval_task_id if isinstance(eval_task_id, str) else None,
        version_id if isinstance(version_id, str) else None,
    )


def build_plan(
    selection: dict[str, Any],
    *,
    identity_roster: dict[str, Any] | None,
    fleet_client: httpx.Client,
    registry_client: httpx.Client,
    namespace: str,
    repository: str,
    read_concurrency: int = 24,
) -> dict[str, Any]:
    protocol = _validate_discovery(registry_client)

    if not 1 <= read_concurrency <= 64:
        raise PlatformImportError("read_concurrency must be between 1 and 64")

    if protocol == EXPECTED_IMPORT_PROTOCOL and identity_roster is None:
        raise PlatformImportError("v3 planning requires the reviewed 160-row identity roster")
    if identity_roster is not None:
        _validate_identity_roster(identity_roster, selection)
    roster_by_key = {
        row["task_key"]: row for row in (identity_roster or {}).get("tasks", [])
    }

    def read_current(source: dict[str, Any]) -> tuple[str | None, str | None]:
        return _current_identity(fleet_client, source)

    with ThreadPoolExecutor(max_workers=read_concurrency) as executor:
        current_identities = list(executor.map(read_current, selection["tasks"]))

    split_seen: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    for source, (current_eval_task_id, current_version_id) in zip(
        selection["tasks"], current_identities, strict=True
    ):
        split_seen[source["split"]] += 1
        reviewed = roster_by_key.get(source["task_key"])
        expected_eval_task_id = reviewed.get("eval_task_id") if reviewed else None
        if protocol == EXPECTED_IMPORT_PROTOCOL and current_eval_task_id != expected_eval_task_id:
            disposition = "blocked_eval_task_identity_differs_from_reviewed"
        elif (
            protocol == EXPECTED_IMPORT_PROTOCOL
            and current_version_id != source["task_version_id"]
        ):
            disposition = "blocked_frozen_version_not_current"
        elif protocol == EXPECTED_IMPORT_PROTOCOL:
            disposition = "eligible_exact_current_frozen"
        elif source["env_key"] in DEPLOYED_TOPOLOGY_OMITTED_ENV_KEYS:
            disposition = "blocked_deployed_topology_omits_environment"
        elif current_version_id == source["task_version_id"]:
            disposition = "eligible_current_equals_frozen"
        else:
            disposition = "blocked_current_differs_from_frozen"
        request = import_request(
            source,
            namespace=namespace,
            repository=repository,
            split_index=split_seen[source["split"]],
            protocol=protocol,
            expected_eval_task_id=expected_eval_task_id,
        )
        identity_digest = (
            sha256(canonical_json(source_identity(source, expected_eval_task_id)))
            if protocol == EXPECTED_IMPORT_PROTOCOL
            else None
        )
        rows.append(
            {
                "source_index": len(rows) + 1,
                "split": source["split"],
                "task_key": source["task_key"],
                "frozen_task_version_id": source["task_version_id"],
                "expected_eval_task_id": expected_eval_task_id,
                "observed_eval_task_id": current_eval_task_id,
                "current_task_version_id": current_version_id,
                "destination_tag": request["destination"]["tag"],
                "idempotency_key": request["idempotency_key"],
                "disposition": disposition,
                "source_identity_sha256": identity_digest,
                "request_sha256": sha256(canonical_json(request)),
            }
        )
    counts = Counter(row["disposition"] for row in rows)
    plan = {
        "schema": PLAN_SCHEMA,
        "source_job_id": selection["source"]["job_id"],
        "source_manifest_digest": selection["manifest_digest"],
        "source_team_id": FLEET_TEAM_ID,
        "identity_roster_sha256": (
            EXPECTED_IDENTITY_ROSTER_SHA256
            if protocol == EXPECTED_IMPORT_PROTOCOL
            else None
        ),
        "destination": {"namespace": namespace, "repository": repository},
        "registry_protocol": protocol,
        "deployed_exact_current_source_pins": protocol == EXPECTED_IMPORT_PROTOCOL,
        "policy": {
            "one_import_per_frozen_task_version": True,
            "tool_use_modality_predeclared": True,
            "moved_current_versions_fail_closed": True,
            "partial_cohort_publication_forbidden": True,
            "task_content_retained": False,
            "exact_source_pin_fields": [
                "expected_eval_task_id",
                "expected_current_task_version_id",
            ],
        },
        "counts": {"total": len(rows), **dict(sorted(counts.items()))},
        "rows": rows,
    }
    plan["plan_sha256"] = sha256(canonical_json(plan))
    return plan


def _validate_discovery(registry_client: httpx.Client) -> str:
    discovery = _request(registry_client, "GET", REGISTRY + DISCOVERY_PATH, "Registry discovery")
    imports = discovery.get("legacy_task_imports") or {}
    if (
        imports.get("base_path") != IMPORT_PATH
        or imports.get("protocol") not in SUPPORTED_IMPORT_PROTOCOLS
    ):
        raise PlatformImportError(
            "Registry alpha legacy-import capability is unavailable or drifted"
        )
    return imports["protocol"]


def validate_plan(
    plan: dict[str, Any],
    selection: dict[str, Any],
    identity_roster: dict[str, Any] | None = None,
) -> None:
    expected_fields = {
        "schema",
        "source_job_id",
        "source_manifest_digest",
        "source_team_id",
        "identity_roster_sha256",
        "destination",
        "registry_protocol",
        "deployed_exact_current_source_pins",
        "policy",
        "counts",
        "rows",
        "plan_sha256",
    }
    rows = plan.get("rows")
    protocol = plan.get("registry_protocol")
    exact_protocol = protocol == EXPECTED_IMPORT_PROTOCOL
    if exact_protocol and identity_roster is None:
        raise PlatformImportError("v3 plan validation requires the reviewed identity roster")
    if identity_roster is not None:
        _validate_identity_roster(identity_roster, selection)
    roster_by_key = {
        row["task_key"]: row for row in (identity_roster or {}).get("tasks", [])
    }
    if (
        set(plan) != expected_fields
        or plan.get("schema") != PLAN_SCHEMA
        or plan.get("source_job_id") != selection["source"]["job_id"]
        or plan.get("source_manifest_digest") != selection["manifest_digest"]
        or plan.get("source_team_id") != FLEET_TEAM_ID
        or plan.get("identity_roster_sha256")
        != (EXPECTED_IDENTITY_ROSTER_SHA256 if exact_protocol else None)
        or protocol not in SUPPORTED_IMPORT_PROTOCOLS
        or plan.get("deployed_exact_current_source_pins") is not exact_protocol
        or plan.get("policy")
        != {
            "one_import_per_frozen_task_version": True,
            "tool_use_modality_predeclared": True,
            "moved_current_versions_fail_closed": True,
            "partial_cohort_publication_forbidden": True,
            "task_content_retained": False,
            "exact_source_pin_fields": [
                "expected_eval_task_id",
                "expected_current_task_version_id",
            ],
        }
        or not isinstance(rows, list)
        or len(rows) != len(selection["tasks"])
        or plan.get("plan_sha256") != digest_without(plan, "plan_sha256")
    ):
        raise PlatformImportError("import plan is not authoritative")
    destination = plan.get("destination")
    if not isinstance(destination, dict) or set(destination) != {"namespace", "repository"}:
        raise PlatformImportError("import plan destination drifted")
    split_seen: Counter[str] = Counter()
    dispositions: Counter[str] = Counter()
    for index, (row, source) in enumerate(zip(rows, selection["tasks"], strict=True), start=1):
        split_seen[source["split"]] += 1
        expected_eval_task_id = (
            roster_by_key.get(source["task_key"], {}).get("eval_task_id")
            if exact_protocol
            else None
        )
        request = import_request(
            source,
            namespace=destination["namespace"],
            repository=destination["repository"],
            split_index=split_seen[source["split"]],
            protocol=protocol,
            expected_eval_task_id=expected_eval_task_id,
        )
        if exact_protocol:
            if row.get("observed_eval_task_id") != expected_eval_task_id:
                disposition = "blocked_eval_task_identity_differs_from_reviewed"
            elif row.get("current_task_version_id") != source["task_version_id"]:
                disposition = "blocked_frozen_version_not_current"
            else:
                disposition = "eligible_exact_current_frozen"
            identity_digest = sha256(
                canonical_json(source_identity(source, expected_eval_task_id))
            )
        elif source["env_key"] in DEPLOYED_TOPOLOGY_OMITTED_ENV_KEYS:
            disposition = "blocked_deployed_topology_omits_environment"
            identity_digest = None
        elif row.get("current_task_version_id") == source["task_version_id"]:
            disposition = "eligible_current_equals_frozen"
            identity_digest = None
        else:
            disposition = "blocked_current_differs_from_frozen"
            identity_digest = None
        expected = {
            "source_index": index,
            "split": source["split"],
            "task_key": source["task_key"],
            "frozen_task_version_id": source["task_version_id"],
            "expected_eval_task_id": expected_eval_task_id,
            "observed_eval_task_id": row.get("observed_eval_task_id"),
            "current_task_version_id": row.get("current_task_version_id"),
            "destination_tag": request["destination"]["tag"],
            "idempotency_key": request["idempotency_key"],
            "disposition": disposition,
            "source_identity_sha256": identity_digest,
            "request_sha256": sha256(canonical_json(request)),
        }
        if row != expected:
            raise PlatformImportError("import plan row drifted from frozen selection")
        dispositions[disposition] += 1
    expected_counts = {"total": len(rows), **dict(sorted(dispositions.items()))}
    if plan.get("counts") != expected_counts:
        raise PlatformImportError("import plan counts drifted")


def require_submit_safe(plan: dict[str, Any]) -> None:
    """Reject writes until the deployed importer binds the exact frozen versions."""
    if (
        plan.get("registry_protocol") != EXPECTED_IMPORT_PROTOCOL
        or plan.get("deployed_exact_current_source_pins") is not True
    ):
        raise PlatformImportError(
            "import submission is held until the deployed importer binds exact task versions"
        )
    if plan.get("counts") != {
        "total": 160,
        "eligible_exact_current_frozen": 160,
    }:
        raise PlatformImportError("partial frozen-cohort publication is forbidden")


def ensure_repository(
    registry_client: httpx.Client, *, namespace: str, repository: str
) -> dict[str, Any]:
    path = f"{REGISTRY}/api/v1/namespaces/{namespace}/repositories/{repository}"
    try:
        response = registry_client.get(path)
    except httpx.HTTPError:
        raise PlatformImportError("Repository read failed before a valid response") from None
    if response.status_code == 200:
        value = _json(response, "Repository read")
        _validate_repository(value, namespace=namespace, repository=repository)
        return {"created": False, "repository": value}
    if response.status_code != 404:
        raise PlatformImportError(f"Repository read failed with HTTP {response.status_code}")
    payload = {
        "name": {"namespace": namespace, "name": repository},
        "kind": "taskset",
        "visibility": "private",
        "mutable_tag_patterns": ["latest"],
    }
    created = _request(
        registry_client,
        "POST",
        REGISTRY + REPOSITORY_PATH,
        "Repository creation",
        payload=payload,
    )
    _validate_repository(created, namespace=namespace, repository=repository)
    return {"created": True, "repository": created}


def _validate_repository(value: dict[str, Any], *, namespace: str, repository: str) -> None:
    if (
        value.get("name") != {"namespace": namespace, "name": repository}
        or value.get("kind") != "taskset"
        or value.get("visibility") != "private"
        or value.get("mutable_tag_patterns") != ["latest"]
    ):
        raise PlatformImportError("destination Repository configuration drifted")


def _row_receipt(
    *,
    plan: dict[str, Any],
    row: dict[str, Any],
    request: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    import_id = result.get("import_id")
    state = result.get("state")
    expected_eval_task_id = request.get("expected_eval_task_id")
    expected_version_id = request.get("expected_current_task_version_id")
    identity_digest = sha256(
        canonical_json(
            {
                "task_key": request["task_key"],
                "expected_eval_task_id": expected_eval_task_id,
                "expected_current_task_version_id": expected_version_id,
            }
        )
    )
    if (
        not isinstance(import_id, str)
        or not TAG_RE.fullmatch(import_id)
        or state not in IMPORT_STATES
        or result.get("task_key") != request["task_key"]
        or result.get("source_team_id") != FLEET_TEAM_ID
        or result.get("destination") != request["destination"]
        or result.get("modality_decision") != request["modality_decision"]
        or result.get("protocol_version") != EXPECTED_IMPORT_PROTOCOL
        or result.get("expected_eval_task_id") != expected_eval_task_id
        or result.get("expected_current_task_version_id") != expected_version_id
        or not isinstance(result.get("idempotent_replay", False), bool)
    ):
        raise PlatformImportError("Legacy task import returned mismatched receipt identity")
    receipt = {
        "schema": ROW_RECEIPT_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "request_sha256": row["request_sha256"],
        "task_key": row["task_key"],
        "task_version_id": row["frozen_task_version_id"],
        "protocol_version": EXPECTED_IMPORT_PROTOCOL,
        "expected_eval_task_id": expected_eval_task_id,
        "expected_current_task_version_id": expected_version_id,
        "source_identity_sha256": identity_digest,
        "destination": request["destination"],
        "import_id": import_id,
        "state": state,
        "idempotent_replay": result.get("idempotent_replay", False),
    }
    receipt["receipt_sha256"] = digest_without(receipt, "receipt_sha256")
    return receipt


def validate_row_receipt(receipt: dict[str, Any]) -> None:
    if (
        set(receipt)
        != {
            "schema",
            "plan_sha256",
            "request_sha256",
            "task_key",
            "task_version_id",
            "protocol_version",
            "expected_eval_task_id",
            "expected_current_task_version_id",
            "source_identity_sha256",
            "destination",
            "import_id",
            "state",
            "idempotent_replay",
            "receipt_sha256",
        }
        or receipt.get("schema") != ROW_RECEIPT_SCHEMA
        or not UUID_RE.fullmatch(str(receipt.get("task_version_id")))
        or not TAG_RE.fullmatch(str(receipt.get("task_key")))
        or not TAG_RE.fullmatch(str(receipt.get("import_id")))
        or receipt.get("protocol_version") != EXPECTED_IMPORT_PROTOCOL
        or not UUID_RE.fullmatch(str(receipt.get("expected_eval_task_id")))
        or receipt.get("expected_current_task_version_id") != receipt.get("task_version_id")
        or receipt.get("source_identity_sha256")
        != sha256(
            canonical_json(
                {
                    "task_key": receipt.get("task_key"),
                    "expected_eval_task_id": receipt.get("expected_eval_task_id"),
                    "expected_current_task_version_id": receipt.get(
                        "expected_current_task_version_id"
                    ),
                }
            )
        )
        or receipt.get("state") not in IMPORT_STATES
        or not isinstance(receipt.get("idempotent_replay"), bool)
        or not isinstance(receipt.get("destination"), dict)
        or set(receipt["destination"]) != {"namespace", "repository", "tag"}
        or any(
            not TAG_RE.fullmatch(str(receipt["destination"].get(key)))
            for key in ("namespace", "repository", "tag")
        )
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", str(receipt.get("plan_sha256")))
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", str(receipt.get("request_sha256")))
        or receipt.get("receipt_sha256") != digest_without(receipt, "receipt_sha256")
    ):
        raise PlatformImportError("import journal contains an invalid receipt")


@contextmanager
def receipt_journal(
    path: Path, *, create: bool = True
) -> Iterator[tuple[list[dict[str, Any]], Any]]:
    """Lock and append sanitized row receipts without replacing prior progress."""
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise PlatformImportError("import journal parent is not a safe operator directory")
    flags = os.O_RDWR | (os.O_CREAT if create else 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags, 0o600)
    except OSError:
        raise PlatformImportError("import journal cannot be opened safely") from None
    try:
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_mode & 0o077:
            raise PlatformImportError("import journal must be a private regular file")
        fcntl.flock(fd, fcntl.LOCK_EX)
        with os.fdopen(fd, "r+", encoding="utf-8") as stream:
            fd = -1
            records: list[dict[str, Any]] = []
            for line in stream:
                try:
                    receipt = json.loads(line)
                except json.JSONDecodeError:
                    raise PlatformImportError("import journal contains invalid JSON") from None
                if not isinstance(receipt, dict):
                    raise PlatformImportError("import journal contains an invalid receipt")
                validate_row_receipt(receipt)
                records.append(receipt)

            def append(receipt: dict[str, Any]) -> None:
                validate_row_receipt(receipt)
                stream.seek(0, os.SEEK_END)
                stream.write(canonical_json(receipt).decode() + "\n")
                stream.flush()
                os.fsync(stream.fileno())

            yield records, append
    finally:
        if fd >= 0:
            os.close(fd)


def submit_rows(
    plan: dict[str, Any],
    selection: dict[str, Any],
    *,
    identity_roster: dict[str, Any],
    registry_client: httpx.Client,
    limit: int | None,
    prior_receipts: list[dict[str, Any]] | None = None,
    receipt_sink: Any = None,
) -> dict[str, Any]:
    validate_plan(plan, selection, identity_roster)
    require_submit_safe(plan)
    source_by_version = {row["task_version_id"]: row for row in selection["tasks"]}
    eligible = [
        row
        for row in plan["rows"]
        if row["disposition"] == "eligible_exact_current_frozen"
    ]
    if limit is not None:
        eligible = eligible[:limit]
    prior: dict[str, dict[str, Any]] = {}
    planned_by_version = {row["frozen_task_version_id"]: row for row in plan["rows"]}
    for receipt in prior_receipts or []:
        validate_row_receipt(receipt)
        version = receipt["task_version_id"]
        row = planned_by_version.get(version)
        source = source_by_version.get(version)
        if row is None or source is None:
            raise PlatformImportError("import journal contains a task outside the frozen plan")
        request = import_request(
            source,
            namespace=plan["destination"]["namespace"],
            repository=plan["destination"]["repository"],
            split_index=int(row["destination_tag"].split("-")[1]),
            protocol=plan["registry_protocol"],
            expected_eval_task_id=row["expected_eval_task_id"],
        )
        if (
            version in prior
            or receipt["plan_sha256"] != plan["plan_sha256"]
            or receipt["request_sha256"] != row["request_sha256"]
            or receipt["task_key"] != row["task_key"]
            or receipt["destination"] != request["destination"]
            or receipt["expected_eval_task_id"] != request["expected_eval_task_id"]
            or receipt["expected_current_task_version_id"]
            != request["expected_current_task_version_id"]
            or receipt["source_identity_sha256"] != row["source_identity_sha256"]
        ):
            raise PlatformImportError("import journal contains conflicting task receipts")
        prior[version] = receipt
    receipts = []
    submitted_new = 0
    resumed = 0
    for row in eligible:
        source = source_by_version[row["frozen_task_version_id"]]
        request = import_request(
            source,
            namespace=plan["destination"]["namespace"],
            repository=plan["destination"]["repository"],
            split_index=int(row["destination_tag"].split("-")[1]),
            protocol=plan["registry_protocol"],
            expected_eval_task_id=row["expected_eval_task_id"],
        )
        if sha256(canonical_json(request)) != row["request_sha256"]:
            raise PlatformImportError("import request drifted from the frozen plan")
        existing = prior.get(row["frozen_task_version_id"])
        if existing is not None:
            if (
                existing["plan_sha256"] != plan["plan_sha256"]
                or existing["request_sha256"] != row["request_sha256"]
                or existing["task_key"] != row["task_key"]
                or existing["destination"] != request["destination"]
                or existing["expected_eval_task_id"] != request["expected_eval_task_id"]
                or existing["expected_current_task_version_id"]
                != request["expected_current_task_version_id"]
                or existing["source_identity_sha256"] != row["source_identity_sha256"]
            ):
                raise PlatformImportError("import journal receipt does not match the frozen plan")
            receipts.append(existing)
            resumed += 1
            continue
        result = _request(
            registry_client,
            "POST",
            REGISTRY + IMPORT_PATH,
            "Legacy task import",
            payload=request,
        )
        receipt = _row_receipt(plan=plan, row=row, request=request, result=result)
        if receipt_sink is not None:
            receipt_sink(receipt)
        receipts.append(receipt)
        submitted_new += 1
    submission = {
        "schema": "fleet_platform_v2_cyber_import_submission_v1",
        "plan_sha256": plan["plan_sha256"],
        "submitted": len(receipts),
        "submitted_new": submitted_new,
        "resumed_from_journal": resumed,
        "receipts": receipts,
    }
    submission["submission_sha256"] = digest_without(submission, "submission_sha256")
    return submission


def monitor_receipts(
    receipts: list[dict[str, Any]], *, registry_client: httpx.Client
) -> dict[str, Any]:
    rows = []
    for receipt in receipts:
        validate_row_receipt(receipt)
        value = _request(
            registry_client,
            "GET",
            f"{REGISTRY}{IMPORT_PATH}/{quote(receipt['import_id'], safe='')}",
            "Legacy task import status",
        )
        if (
            value.get("import_id") != receipt["import_id"]
            or value.get("task_key") != receipt["task_key"]
            or value.get("source_team_id") != FLEET_TEAM_ID
            or value.get("destination") != receipt["destination"]
            or value.get("protocol_version") != receipt["protocol_version"]
            or value.get("expected_eval_task_id") != receipt["expected_eval_task_id"]
            or value.get("expected_current_task_version_id")
            != receipt["expected_current_task_version_id"]
            or value.get("state") not in IMPORT_STATES
            or not isinstance(value.get("version"), int)
            or value.get("version", 0) < 1
        ):
            raise PlatformImportError("Legacy task import status identity drifted")
        error = value.get("error") or {}
        error_code = error.get("code") if isinstance(error, dict) else None
        if (
            not isinstance(error, dict)
            or set(error) - {"code"}
            or (error_code is not None and not TAG_RE.fullmatch(str(error_code)))
        ):
            raise PlatformImportError("Legacy task import status error shape drifted")
        rows.append(
            {
                "import_id": receipt["import_id"],
                "destination_tag": receipt["destination"]["tag"],
                "protocol_version": receipt["protocol_version"],
                "expected_eval_task_id": receipt["expected_eval_task_id"],
                "expected_current_task_version_id": receipt[
                    "expected_current_task_version_id"
                ],
                "source_identity_sha256": receipt["source_identity_sha256"],
                "state": value["state"],
                "version": value["version"],
                "error_code": error_code,
            }
        )
    return {
        "schema": "fleet_platform_v2_cyber_import_status_v3",
        "imports": len(rows),
        "states": dict(Counter(row["state"] for row in rows)),
        "rows": rows,
    }


def write_json_create_once(path: Path, value: dict[str, Any]) -> None:
    payload = canonical_json(value) + b"\n"
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise PlatformImportError("output parent is not a safe operator directory")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags, 0o600)
    except FileExistsError:
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise PlatformImportError("output already exists with different bytes") from None
        return
    except OSError:
        raise PlatformImportError("output cannot be created safely") from None
    try:
        with os.fdopen(fd, "wb") as stream:
            fd = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        parent_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    finally:
        if fd >= 0:
            os.close(fd)


def submission_summary(
    repository: dict[str, Any], submission: dict[str, Any], plan: dict[str, Any]
) -> dict[str, Any]:
    return {
        "dry_run": False,
        "repository_created": repository["created"],
        "submitted": submission["submitted"],
        "submitted_new": submission["submitted_new"],
        "resumed_from_journal": submission["resumed_from_journal"],
        "states": dict(Counter(row["state"] for row in submission["receipts"])),
        "plan_sha256": plan["plan_sha256"],
        "receipts": [
            {
                "import_id": row["import_id"],
                "destination_tag": row["destination"]["tag"],
                "protocol_version": row["protocol_version"],
                "source_identity_sha256": row["source_identity_sha256"],
                "state": row["state"],
                "idempotent_replay": row["idempotent_replay"],
            }
            for row in submission["receipts"]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--identity-roster", type=Path, default=DEFAULT_IDENTITY_ROSTER)
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--repository", default=DEFAULT_REPOSITORY)
    parser.add_argument("--plan-input", type=Path)
    parser.add_argument("--plan-output", type=Path)
    parser.add_argument("--journal", type=Path)
    parser.add_argument("--monitor", action="store_true")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")
    if args.monitor and args.submit:
        raise SystemExit("--monitor and --submit are mutually exclusive")
    if args.plan_input and args.plan_output:
        raise SystemExit("--plan-input and --plan-output are mutually exclusive")
    if args.submit and args.plan_input is None:
        raise SystemExit("--plan-input is required for a reviewed resumable submit")
    if (args.monitor or args.submit) and args.journal is None:
        raise SystemExit("--journal is required for submit and monitor")

    selection = load_selection(args.selection)
    registry_token = load_registry_token()
    if args.monitor:
        with (
            httpx.Client(
                headers={"Authorization": f"Bearer {registry_token}"}, timeout=120
            ) as registry,
            receipt_journal(args.journal, create=False) as (receipts, _append),
        ):
            status = monitor_receipts(receipts, registry_client=registry)
        print(json.dumps(status, indent=2))
        return
    fleet_key = os.environ.get("FLEET_API_KEY")
    if not fleet_key:
        raise SystemExit("FLEET_API_KEY is required for the source identity preflight")
    identity_roster = load_identity_roster(args.identity_roster, selection)
    with (
        httpx.Client(
            headers={"Authorization": f"Bearer {registry_token}"}, timeout=120
        ) as registry,
        httpx.Client(headers={"Authorization": f"Bearer {fleet_key}"}, timeout=120) as fleet,
    ):
        deployed_protocol = _validate_discovery(registry)
        if args.plan_input:
            plan = load_json_object(args.plan_input, "reviewed import plan")
        else:
            plan = build_plan(
                selection,
                identity_roster=identity_roster,
                fleet_client=fleet,
                registry_client=registry,
                namespace=args.namespace,
                repository=args.repository,
            )
        validate_plan(plan, selection, identity_roster)
        if plan["registry_protocol"] != deployed_protocol:
            raise PlatformImportError("reviewed plan protocol differs from deployed discovery")
        if plan["destination"] != {
            "namespace": args.namespace,
            "repository": args.repository,
        }:
            raise PlatformImportError("CLI destination differs from the reviewed plan")
        if args.plan_output:
            write_json_create_once(args.plan_output, plan)
        if not args.submit:
            print(
                json.dumps(
                    {"dry_run": True, **plan["counts"], "plan_sha256": plan["plan_sha256"]},
                    indent=2,
                )
            )
            return
        require_submit_safe(plan)
        repository = ensure_repository(
            registry, namespace=args.namespace, repository=args.repository
        )
        with receipt_journal(args.journal) as (prior, append):
            submission = submit_rows(
                plan,
                selection,
                identity_roster=identity_roster,
                registry_client=registry,
                limit=args.limit,
                prior_receipts=prior,
                receipt_sink=append,
            )
        print(
            json.dumps(
                submission_summary(repository, submission, plan),
                indent=2,
            )
        )
        if fleet is not None:
            fleet.close()


if __name__ == "__main__":
    try:
        main()
    except PlatformImportError as exc:
        raise SystemExit(f"error: {exc}") from None
