"""Qualify unproven Fleet cyber task versions without running a model.

The controller has two deliberately separate phases. ``prepare`` reads only
metadata, freezes an exact roster, and excludes any task sharing an atom family
with the protected dev or final split. ``run`` creates each exact environment once,
probes the two public task tools, records one genuine verifier outcome through
the task-version authority, ingests a zero-message evidence session, and
terminates the environment.  No prompt, tool output, flag, trace, or numeric
score is written by this module.

Every mutating request is preceded by an exclusive intent file and is attempted
once.  An ambiguous response quarantines that task version; rerunning an
existing operation root is forbidden.  The private catalog contains exact
task/runtime identities for the roster adapter.  The public aggregate receipt
contains counts and digests only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import httpx

from evals.fleet import opencode_self_hosted as self_hosted
from evals.fleet import rollout_worker

PLAN_SCHEMA = "cyber_task_quality_qualification_plan_v1"
RUN_INTENT_SCHEMA = "cyber_task_quality_qualification_run_intent_v1"
CELL_TERMINAL_SCHEMA = "cyber_task_quality_qualification_cell_terminal_v1"
PRIVATE_CATALOG_SCHEMA = "cyber_fleet_task_qualification_catalog_v1"
SUPPLY_CATALOG_SCHEMA = "cyber_fleet_blackbox_task_supply_catalog_v1"
ATTEMPTED_CATALOG_SCHEMA = "cyber_task_quality_qualification_attempted_catalog_v1"
AGGREGATE_SCHEMA = "cyber_task_quality_qualification_aggregate_receipt_v1"
CLEANUP_RESOLUTION_SCHEMA = "cyber_task_quality_cleanup_resolution_v1"
CLEANUP_AGGREGATE_SCHEMA = "cyber_task_quality_cleanup_aggregate_v1"
INVENTORY_SCHEMA = "fleet_current_production_blackbox_inventory_v1"
COVERAGE_SCHEMA = "fleet_current_blackbox_training_coverage_v1"
SPLIT_SCHEMA = "cyber_representative_study_split_v2"
HELDOUT_PROTOCOL_SCHEMA = "cyber_fleet_existing_checkpoint_holdout_protocol_v2"
EXPECTED_TEAM_ID = "a1025f0b-ad67-49fc-a023-51800ab43e84"
CREATE_CLAIM_ROUTE_TEMPLATE = "/v1/env/instances/create-requests/{request_id}"
SESSION_MODEL = "fleet/task-quality-runtime-probe-v1"
REQUIRED_TOOLS = ["bash", "submit_report"]
REQUIRED_CONTRACT = {
    "evidence_schema": "1.0.0",
    "submission_protocol": "2.0.0",
    "verifier_contract": "3.0.0",
}
SHA256 = re.compile(r"sha256:[0-9a-f]{64}")
UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


class QualificationError(RuntimeError):
    """A frozen input, live binding, or one-shot operation is invalid."""


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return "sha256:" + value.hexdigest()


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def source_provenance(*, require_merged: bool = True) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    if _git(root, "status", "--porcelain", "--untracked-files=no"):
        raise QualificationError("qualification source worktree must be clean")
    commit = _git(root, "rev-parse", "HEAD")
    origin_main = _git(root, "rev-parse", "refs/remotes/origin/main")
    ancestry = subprocess.run(
        ["git", "-C", str(root), "merge-base", "--is-ancestor", commit, origin_main],
        check=False,
        capture_output=True,
        text=True,
    )
    if ancestry.returncode not in {0, 1}:
        raise QualificationError("cannot verify qualification source ancestry")
    merged = ancestry.returncode == 0
    if require_merged and not merged:
        raise QualificationError("qualification source commit is not merged to origin/main")
    return {
        "git_commit": commit,
        "git_tree": _git(root, "rev-parse", "HEAD^{tree}"),
        "origin_main_commit": origin_main,
        "merged_to_origin_main": merged,
        "controller_path": "evals/fleet/task_quality_qualification.py",
        "controller_file_sha256": file_digest(Path(__file__).resolve()),
    }


def _source_matches_plan(source: object) -> None:
    if not isinstance(source, dict) or source.get("merged_to_origin_main") is not True:
        raise QualificationError("qualification plan is not authorized from merged source")
    observed = source_provenance(require_merged=True)
    exact_fields = {"git_commit", "git_tree", "controller_path", "controller_file_sha256"}
    if any(source.get(field) != observed.get(field) for field in exact_fields):
        raise QualificationError("qualification source differs from the frozen plan")


def sealed(value: dict[str, Any]) -> dict[str, Any]:
    body = dict(value)
    return {**body, "sha256": digest(body)}


def _write_once(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    payload = json.dumps(value, sort_keys=True, indent=2).encode() + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _read(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise QualificationError(f"{label} must be a regular file")
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise QualificationError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise QualificationError(f"{label} must be a JSON object")
    return value


def _sealed(value: dict[str, Any], schema: str, label: str) -> None:
    if value.get("schema") != schema or value.get("sha256") != digest(
        {key: item for key, item in value.items() if key != "sha256"}
    ):
        raise QualificationError(f"{label} is not an exact sealed {schema}")


def _uuid(value: object, label: str) -> str:
    if not isinstance(value, str) or UUID.fullmatch(value) is None:
        raise QualificationError(f"{label} must be a canonical UUID")
    try:
        if str(uuid.UUID(value)) != value:
            raise ValueError
    except ValueError as error:
        raise QualificationError(f"{label} must be a canonical UUID") from error
    return value


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise QualificationError(f"{label} SHA-256 is missing")
    normalized = value if value.startswith("sha256:") else f"sha256:{value}"
    if SHA256.fullmatch(normalized) is None:
        raise QualificationError(f"{label} SHA-256 is invalid")
    return normalized


def _client(api_key: str) -> httpx.Client:
    if not api_key:
        raise QualificationError("FLEET_API_KEY is required")
    return httpx.Client(
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        timeout=1800,
    )


def _account(client: httpx.Client) -> None:
    account = self_hosted._request(client, "GET", "/v1/account")  # noqa: SLF001
    if account.get("team_name") != "fleet" or account.get("team_id") != EXPECTED_TEAM_ID:
        raise QualificationError("FLEET_API_KEY is not scoped to the Fleet team")


def _assert_create_claim_routes_deployed(client: httpx.Client) -> dict[str, Any]:
    """Prove exact cleanup reconciliation exists before provisioning can mutate."""
    openapi = self_hosted._request(client, "GET", "/openapi.json")  # noqa: SLF001
    paths = openapi.get("paths") if isinstance(openapi, dict) else None
    route = paths.get(CREATE_CLAIM_ROUTE_TEMPLATE) if isinstance(paths, dict) else None
    required_methods = ("get", "delete")
    if not isinstance(route, dict) or any(
        not isinstance(route.get(method), dict) for method in required_methods
    ):
        raise QualificationError("durable create-request claim GET/DELETE routes are not deployed")
    return {
        "mode": "openapi",
        "route": CREATE_CLAIM_ROUTE_TEMPLATE,
        "methods": [method.upper() for method in required_methods],
    }


def _source_locator(value: object, label: str, *, bundle_kind: str) -> str:
    if not isinstance(value, str) or not value.startswith("cyber/") or ":" not in value:
        raise QualificationError(f"{label} is not an immutable Registry locator")
    locator, observed_bundle_kind = value.rsplit(":", 1)
    try:
        artifact_key, version_index = locator.rsplit("@", 1)
    except ValueError as error:
        raise QualificationError(f"{label} is not an immutable Registry locator") from error
    if (
        not artifact_key.startswith("cyber/")
        or not artifact_key.removeprefix("cyber/")
        or not version_index.isdigit()
        or observed_bundle_kind != bundle_kind
    ):
        raise QualificationError(f"{label} is not an immutable Registry locator")
    return locator


def safe_task_binding(
    task: dict[str, Any], selected: dict[str, Any], *, require_tool_declaration: bool = True
) -> dict[str, Any]:
    """Project an exact task GET onto content-free runtime and lineage metadata."""
    if task.get("key") != selected.get("task_key"):
        raise QualificationError("live task key differs from the frozen inventory")
    version_id = _uuid(task.get("eval_task_version_id"), "live task version")
    if version_id != _uuid(selected.get("task_version_id"), "inventory task version"):
        raise QualificationError("live task version differs from the frozen inventory")
    if selected.get("task_shape") != "blackbox":
        raise QualificationError("qualification supports only blackbox task versions")
    environment = {
        "id": task.get("environment_id"),
        "version": task.get("version"),
        "version_id": _uuid(task.get("environment_version_id"), "environment version"),
        "legacy_data_id": task.get("data_id"),
        "legacy_data_version": task.get("data_version"),
        "seed_config": task.get("seed_config"),
    }
    if not isinstance(environment["id"], str) or not environment["id"]:
        raise QualificationError("task environment ID is missing")
    if not isinstance(environment["version"], str) or not environment["version"]:
        raise QualificationError("task environment version is missing")
    seeds = environment["seed_config"]
    if not isinstance(seeds, dict) or not seeds:
        if not all(
            isinstance(environment[field], str) and environment[field]
            for field in ("legacy_data_id", "legacy_data_version")
        ):
            raise QualificationError("task has no exact starting-data binding")
        seeds = {
            environment["id"]: {
                "env_key": environment["id"],
                "data_key": environment["legacy_data_id"],
                "data_version": environment["legacy_data_version"],
            }
        }
        environment["seed_config"] = seeds
    for name, seed in seeds.items():
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(seed, dict)
            or set(seed) != {"env_key", "data_key", "data_version"}
            or any(not isinstance(seed[field], str) or not seed[field] for field in seed)
        ):
            raise QualificationError("task starting-data binding is invalid")

    metadata = task.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise QualificationError("task metadata is invalid")
    if metadata.get("projection_id") != "blackbox_ctf_v1":
        raise QualificationError("task is not the blackbox CTF projection")
    declared_tools = metadata.get("tools")
    # Older exact task versions predate the metadata mirror.  Missing metadata
    # is not positive evidence, so it is neither admitted nor rejected here;
    # the live MCP probe below is the qualification authority.  A conflicting
    # non-null declaration remains fatal.
    if declared_tools not in (None, REQUIRED_TOOLS):
        raise QualificationError("task has an unexpected tool declaration")
    if require_tool_declaration and declared_tools is None:
        tool_declaration_status = "absent_requires_live_probe"
    else:
        tool_declaration_status = "exact" if declared_tools == REQUIRED_TOOLS else "absent"
    if metadata.get("cyber_contract") != REQUIRED_CONTRACT:
        raise QualificationError("task cyber contract is not the exact supported contract")
    runtime_seed = (metadata.get("runtime_seed_manifest") or {}).get("content_sha256")
    if runtime_seed is None:
        runtime_seed_status = "absent_authoritative_task_version_and_seed_config_required"
    else:
        runtime_seed = _sha(runtime_seed, "runtime seed")
        runtime_seed_status = "exact"

    verifier = task.get("verifier") or {}
    if not isinstance(verifier, dict):
        raise QualificationError("task verifier is invalid")
    verifier_binding = {
        "id": _uuid(task.get("verifier_id"), "verifier"),
        "version_id": _uuid(verifier.get("verifier_version_id"), "verifier version"),
        "version": verifier.get("version"),
        "sha256": _sha(verifier.get("sha256"), "verifier"),
    }
    if type(verifier_binding["version"]) is not int or verifier_binding["version"] <= 0:
        raise QualificationError("verifier version number is invalid")

    subject = metadata.get("cyber_subject") or {}
    if not isinstance(subject, dict):
        raise QualificationError("task cyber subject is invalid")
    atom_sources = subject.get("atom_sources")
    if not isinstance(atom_sources, list) or not atom_sources:
        raise QualificationError("task has no reviewed atom-source lineage")
    atoms: list[dict[str, Any]] = []
    for source in atom_sources:
        if not isinstance(source, dict):
            raise QualificationError("task atom-source lineage is invalid")
        artifact_key = source.get("artifact_key")
        atom_id = source.get("atom_id")
        locator = _source_locator(source.get("locator"), "atom source", bundle_kind="atom_source")
        version_index = source.get("version_index")
        if (
            not isinstance(artifact_key, str)
            or not artifact_key.startswith("cyber/atoms/")
            or not isinstance(atom_id, str)
            or "/" not in atom_id
            or artifact_key != f"cyber/atoms/{atom_id}"
            or type(version_index) is not int
            or version_index < 0
            or locator != f"{artifact_key}@{version_index}"
        ):
            raise QualificationError("task atom-source binding is incomplete")
        atoms.append(
            {
                "artifact_key": artifact_key,
                "atom_id": atom_id,
                "locator": locator,
                "version_index": version_index,
            }
        )
    if len({row["artifact_key"] for row in atoms}) != len(atoms):
        raise QualificationError("task atom-source lineage contains duplicates")
    source_locator = _source_locator(
        subject.get("source_locator"),
        "task graph source",
        bundle_kind="task_graph_source",
    )
    task_graph_id = subject.get("task_graph_id")
    if not isinstance(task_graph_id, str) or not task_graph_id:
        raise QualificationError("task graph identity is missing")
    if source_locator.rsplit("@", 1)[0] != f"cyber/task-graphs/{task_graph_id}":
        raise QualificationError("task graph source is not bound to the task graph identity")
    applications = sorted({row["atom_id"].split("/", 1)[0] for row in atoms})
    family = atoms[0]["artifact_key"] if len(atoms) == 1 else source_locator.rsplit("@", 1)[0]
    difficulty = metadata.get("task_graph_band") or metadata.get("expected_difficulty")
    if not isinstance(difficulty, str) or not difficulty:
        raise QualificationError("task difficulty lineage is missing")
    lineage = {
        "application": "+".join(applications),
        "environment": environment["id"],
        "difficulty": difficulty,
        "vulnerability_family": sorted(f"atom:{row['artifact_key']}" for row in atoms),
        "task_family": family,
    }
    body = {
        "task_key": selected["task_key"],
        "task_version_id": version_id,
        "qa_status": selected["qa_status"],
        "task_id": _uuid(selected.get("task_id"), "inventory task"),
        "environment": environment,
        "runtime_seed_content_sha256": runtime_seed,
        "runtime_seed_status": runtime_seed_status,
        "starting_data_binding_sha256": digest(environment["seed_config"]),
        "verifier": verifier_binding,
        "lineage": lineage,
        "atom_artifact_keys": sorted(row["artifact_key"] for row in atoms),
        "task_graph": {"id": task_graph_id, "source_locator": source_locator},
        "task_content": {
            "prompt_sha256": self_hosted.sha256((task.get("prompt") or "").encode()),
            "env_variables_sha256": self_hosted.sha256(
                self_hosted.canonical_json(task.get("env_variables") or {})
            ),
            "output_json_schema_sha256": self_hosted.sha256(
                self_hosted.canonical_json(task.get("output_json_schema"))
            ),
        },
        "tool_declaration_status": tool_declaration_status,
    }
    return {**body, "binding_sha256": digest(body)}


def _fetch_binding(
    client: httpx.Client,
    selected: dict[str, Any],
    *,
    require_tool_declaration: bool = True,
) -> dict[str, Any]:
    task = self_hosted._request(  # noqa: SLF001
        client,
        "GET",
        f"/v1/tasks/{selected['task_key']}",
        params={"version_id": selected["task_version_id"]},
    )
    return safe_task_binding(task, selected, require_tool_declaration=require_tool_declaration)


def _input(path: Path, schema: str, label: str) -> dict[str, Any]:
    value = _read(path, label)
    if value.get("schema") != schema or value.get("sha256") != digest(
        {key: item for key, item in value.items() if key != "sha256"}
    ):
        raise QualificationError(f"{label} is not an exact sealed {schema}")
    return value


def _input_one_of(path: Path, schemas: set[str], label: str) -> dict[str, Any]:
    value = _read(path, label)
    schema = value.get("schema")
    if schema not in schemas or value.get("sha256") != digest(
        {key: item for key, item in value.items() if key != "sha256"}
    ):
        raise QualificationError(f"{label} is not an exact supported sealed input")
    return value


def _protected_rows(split: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    """Normalize the original split and the repaired clean-heldout protocol."""
    if split.get("schema") == SPLIT_SCHEMA:
        rows = split.get("tasks")
        if not isinstance(rows, list):
            raise QualificationError("protected split task roster is invalid")
        protected = [
            row
            for row in rows
            if isinstance(row, dict) and row.get("split") in {"dev", "final_test"}
        ]
        final_count = sum(row.get("split") == "final_test" for row in protected)
        if len(protected) != 25 or final_count != 8:
            raise QualificationError(
                "protected split must contain exactly 17 dev and eight final tasks"
            )
        return protected, final_count

    selection = split.get("selection")
    rows = selection.get("tasks") if isinstance(selection, dict) else None
    if not isinstance(rows, list) or not rows:
        raise QualificationError("repaired held-out protocol task roster is invalid")
    protected = []
    for row in rows:
        if not isinstance(row, dict) or row.get("source_role") not in {"dev", "final_test"}:
            raise QualificationError("repaired held-out protocol role is invalid")
        family = row.get("reviewed_task_family")
        if not isinstance(family, str) or not family.startswith("cyber/atoms/"):
            raise QualificationError("repaired held-out protocol family is invalid")
        protected.append(
            {
                "task_key": row.get("task_key"),
                "task_version_id": row.get("task_version_id"),
                "split": row["source_role"],
                "reviewed_task_family": family,
            }
        )
    if selection.get("exact_task_version_count") != len(protected):
        raise QualificationError("repaired held-out protocol count is invalid")
    if len({(row["task_key"], row["task_version_id"]) for row in protected}) != len(
        protected
    ):
        raise QualificationError("repaired held-out protocol contains duplicate task versions")
    return protected, sum(row["split"] == "final_test" for row in protected)


def build_plan(
    *,
    inventory: dict[str, Any],
    coverage: dict[str, Any],
    split: dict[str, Any],
    inventory_sha256: str,
    coverage_sha256: str,
    split_sha256: str,
    client: httpx.Client,
    wave_id: str,
    qa_statuses: set[str],
    limit: int,
    concurrency: int,
    source: dict[str, Any],
    excluded_catalog: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,62}", wave_id):
        raise QualificationError("wave ID is not DNS-safe")
    if not 1 <= limit <= 64 or not 1 <= concurrency <= limit:
        raise QualificationError("wave limit/concurrency must be within 1..64")
    if not qa_statuses or not qa_statuses <= {"clean", "agent_failure", "not_analyzed"}:
        raise QualificationError("wave QA statuses are unsupported")
    if "not_analyzed" in qa_statuses and (
        excluded_catalog is None or excluded_catalog.get("schema") != ATTEMPTED_CATALOG_SCHEMA
    ):
        raise QualificationError("not_analyzed waves require the cumulative attempted catalog")
    inventory_rows = inventory.get("tasks")
    if not isinstance(inventory_rows, list) or inventory.get("task_count") != len(inventory_rows):
        raise QualificationError("inventory task census is invalid")
    missing = coverage.get("no_exact_receipt_in_this_refresh_task_keys")
    if not isinstance(missing, list) or any(not isinstance(key, str) for key in missing):
        raise QualificationError("coverage missing-receipt roster is invalid")
    missing_keys = set(missing)
    row_by_identity = {
        (row.get("task_key"), row.get("task_version_id")): row
        for row in inventory_rows
        if isinstance(row, dict)
    }
    if len(row_by_identity) != len(inventory_rows):
        raise QualificationError("inventory contains malformed or duplicate task versions")
    protected_rows, final_count = _protected_rows(split)
    heldout_rows: list[dict[str, Any]] = []
    for row in protected_rows:
        key = (row.get("task_key"), row.get("task_version_id"))
        selected = row_by_identity.get(key)
        if selected is None:
            raise QualificationError("protected held-out task is absent from the current inventory")
        heldout_rows.append(selected)
    heldout_bindings = [
        _fetch_binding(client, row, require_tool_declaration=False) for row in heldout_rows
    ]
    reviewed_families = {
        (row["task_key"], row["task_version_id"]): row.get("reviewed_task_family")
        for row in protected_rows
        if row.get("reviewed_task_family") is not None
    }
    for binding in heldout_bindings:
        reviewed = reviewed_families.get((binding["task_key"], binding["task_version_id"]))
        if reviewed is not None:
            expected_key = reviewed.rsplit("@", 1)[0]
            if expected_key not in binding["atom_artifact_keys"]:
                raise QualificationError("protected held-out reviewed family differs live")
    heldout_atom_keys = {
        atom for binding in heldout_bindings for atom in binding["atom_artifact_keys"]
    }

    excluded_ids: set[tuple[str, str]] = set()
    excluded_families: set[str] = set()
    prior_attempted: list[dict[str, Any]] = []
    if excluded_catalog is not None:
        schema = excluded_catalog.get("schema")
        if schema not in {PRIVATE_CATALOG_SCHEMA, ATTEMPTED_CATALOG_SCHEMA}:
            raise QualificationError("excluded catalog schema is unsupported")
        _sealed(excluded_catalog, schema, "excluded qualification catalog")
        if excluded_catalog.get("catalog_snapshot_sha256") != inventory.get("sha256"):
            raise QualificationError("excluded catalog is bound to a different inventory snapshot")
        for row in excluded_catalog.get("task_versions") or []:
            if not isinstance(row, dict):
                raise QualificationError("excluded qualification catalog task is invalid")
            identity = (row.get("task_key"), row.get("task_version_id"))
            if identity not in row_by_identity or identity in excluded_ids:
                raise QualificationError("excluded catalog task is absent or duplicated")
            excluded_ids.add(identity)
            lineage = row.get("lineage") or {}
            family = lineage.get("task_family")
            if not isinstance(family, str) or not family:
                raise QualificationError("excluded catalog task family is invalid")
            excluded_families.add(family)
            prior_attempted.append(
                {
                    "task_key": row.get("task_key"),
                    "task_version_id": row.get("task_version_id"),
                    "lineage": lineage,
                }
            )

    candidates = sorted(
        (
            row
            for row in inventory_rows
            if isinstance(row, dict)
            and row.get("task_key") in missing_keys
            and row.get("qa_status") in qa_statuses
            and (row.get("task_key"), row.get("task_version_id")) not in excluded_ids
        ),
        key=lambda row: (str(row.get("task_key")), str(row.get("task_version_id"))),
    )
    selected_bindings: list[dict[str, Any]] = []
    seen_families = set(excluded_families)
    excluded_counts = {
        "protected_heldout_atom_overlap": 0,
        "duplicate_or_prior_task_family": 0,
        "invalid_live_binding": 0,
    }
    for row in candidates:
        try:
            binding = _fetch_binding(client, row)
        except Exception:  # noqa: BLE001 - aggregate-only planning classification
            excluded_counts["invalid_live_binding"] += 1
            continue
        if heldout_atom_keys.intersection(binding["atom_artifact_keys"]):
            excluded_counts["protected_heldout_atom_overlap"] += 1
            continue
        family = binding["lineage"]["task_family"]
        if family in seen_families:
            excluded_counts["duplicate_or_prior_task_family"] += 1
            continue
        selected_bindings.append(binding)
        seen_families.add(family)
        if len(selected_bindings) == limit:
            break
    if not selected_bindings:
        raise QualificationError("no candidate task versions passed the source-only plan gates")
    body = {
        "schema": PLAN_SCHEMA,
        "wave_id": wave_id,
        "source": source,
        "inputs": {
            "inventory_file_sha256": inventory_sha256,
            "inventory_sha256": inventory["sha256"],
            "coverage_file_sha256": coverage_sha256,
            "coverage_sha256": coverage["sha256"],
            "protected_split_file_sha256": split_sha256,
            "protected_split_sha256": split["sha256"],
        },
        "selection": {
            "qa_statuses": sorted(qa_statuses),
            "requested_limit": limit,
            "selected_task_versions": len(selected_bindings),
            "maximum_task_versions_per_family": 1,
            "excluded_counts": excluded_counts,
            "protected_heldout_task_versions": len(heldout_rows),
            "protected_final_task_versions": final_count,
            "protected_heldout_atom_key_count": len(heldout_atom_keys),
            "protected_heldout_atom_keys_sha256": digest(sorted(heldout_atom_keys)),
            "zero_protected_heldout_atom_intersection": True,
        },
        "execution": {
            "controller": "local_exactly_once_v1",
            "concurrency": concurrency,
            "model_calls": 0,
            "training_data_eligible": False,
            "mutating_request_attempts": 1,
            "automatic_retry": False,
            "ambiguous_mutation_replay": False,
            "external_mutations_authorized": source.get("merged_to_origin_main") is True,
            "tool_probe": {
                "bash": {"script": "printf task-quality-runtime-ok"},
                "submit_report": {
                    "verdict": "no_flag",
                    "explanation": (
                        "Automated task-runtime qualification probe; exploitability was not tested."
                    ),
                },
            },
        },
        "prior_attempted_task_versions": sorted(
            prior_attempted, key=lambda row: (row["task_key"], row["task_version_id"])
        ),
        "tasks": selected_bindings,
        "privacy": {
            "private_plan": True,
            "public_receipt_aggregate_only": True,
            "prompts_traces_answers_flags_credentials_persisted": False,
            "numeric_scores_persisted_in_operation_artifacts": False,
        },
    }
    return sealed(body)


def _config(binding: dict[str, Any], wave_id: str) -> dict[str, Any]:
    run_id = f"task-quality-{wave_id}-{binding['binding_sha256'].removeprefix('sha256:')[:12]}"
    return {
        "schema_version": "fleet-task-quality-qualification-cell-v1",
        "run_id": run_id,
        "task": {"key": binding["task_key"], "version_id": binding["task_version_id"]},
        "environment": {
            "id": binding["environment"]["id"],
            "version": binding["environment"]["version"],
            "version_id": binding["environment"]["version_id"],
        },
        "verifier": binding["verifier"],
        "authority": {
            **rollout_worker.AUTHORITY,
            "scoring_payload_mode": self_hosted.RUNTIME_EVIDENCE_ONLY_V3,
        },
        "model": {"session_model": SESSION_MODEL},
        "execution": {"training_data_eligible": False},
    }


def _mcp_result(value: dict[str, Any], request_id: int, label: str) -> None:
    if value.get("id") != request_id or not isinstance(value.get("result"), dict):
        raise QualificationError(f"{label} MCP response binding is invalid")
    result = value["result"]
    if result.get("isError") is True or result.get("is_error") is True:
        raise QualificationError(f"{label} MCP call reported an error")


def _probe_arguments() -> dict[str, dict[str, Any]]:
    return {
        "bash": {"script": "printf task-quality-runtime-ok"},
        "submit_report": {
            "verdict": "no_flag",
            "explanation": (
                "Automated task-runtime qualification probe; exploitability was not tested."
            ),
        },
    }


def _schema_admits(schema: object, value: object) -> bool:  # noqa: C901, PLR0911
    """Conservatively evaluate the JSON-Schema subset used by Fleet MCP tools."""
    if isinstance(schema, bool):
        return schema
    if not isinstance(schema, dict):
        return False
    annotations = {
        "$comment",
        "$id",
        "$schema",
        "default",
        "deprecated",
        "description",
        "examples",
        "readOnly",
        "title",
        "writeOnly",
    }
    supported = annotations | {
        "additionalProperties",
        "allOf",
        "anyOf",
        "const",
        "enum",
        "items",
        "maxItems",
        "maxLength",
        "maxProperties",
        "minItems",
        "minLength",
        "minProperties",
        "oneOf",
        "properties",
        "required",
        "type",
        "uniqueItems",
    }
    if set(schema) - supported:
        return False
    if "const" in schema and value != schema["const"]:
        return False
    if "enum" in schema and (not isinstance(schema["enum"], list) or value not in schema["enum"]):
        return False
    for keyword, predicate in (
        ("allOf", all),
        ("anyOf", any),
    ):
        branches = schema.get(keyword)
        if branches is not None:
            if not isinstance(branches, list) or not branches:
                return False
            admitted = [_schema_admits(branch, value) for branch in branches]
            if not predicate(admitted):
                return False
    branches = schema.get("oneOf")
    if branches is not None and (
        not isinstance(branches, list)
        or sum(_schema_admits(branch, value) for branch in branches) != 1
    ):
        return False
    declared_type = schema.get("type")
    allowed_types = [declared_type] if isinstance(declared_type, str) else declared_type
    if allowed_types is not None:
        if not isinstance(allowed_types, list) or not all(
            isinstance(item, str) for item in allowed_types
        ):
            return False
        observed_type = (
            "null"
            if value is None
            else "boolean"
            if isinstance(value, bool)
            else "object"
            if isinstance(value, dict)
            else "array"
            if isinstance(value, list)
            else "string"
            if isinstance(value, str)
            else "integer"
            if isinstance(value, int)
            else "number"
            if isinstance(value, float)
            else "unsupported"
        )
        if observed_type not in allowed_types and not (
            observed_type == "integer" and "number" in allowed_types
        ):
            return False
    if isinstance(value, dict):
        required = schema.get("required", [])
        properties = schema.get("properties", {})
        additional = schema.get("additionalProperties", True)
        if (
            not isinstance(required, list)
            or any(not isinstance(name, str) for name in required)
            or not isinstance(properties, dict)
            or not isinstance(additional, (bool, dict))
            or any(name not in value for name in required)
        ):
            return False
        for name, item in value.items():
            item_schema = properties.get(name, additional)
            if not _schema_admits(item_schema, item):
                return False
        for keyword, default, compare in (
            ("minProperties", 0, lambda size, bound: size >= bound),
            ("maxProperties", None, lambda size, bound: size <= bound),
        ):
            bound = schema.get(keyword, default)
            if bound is not None and (type(bound) is not int or not compare(len(value), bound)):
                return False
    if isinstance(value, list):
        items = schema.get("items", True)
        if not isinstance(items, (bool, dict)) or any(
            not _schema_admits(items, item) for item in value
        ):
            return False
        if schema.get("uniqueItems", False) is True and len(
            {canonical_bytes(item) for item in value}
        ) != len(value):
            return False
        for keyword, default, compare in (
            ("minItems", 0, lambda size, bound: size >= bound),
            ("maxItems", None, lambda size, bound: size <= bound),
        ):
            bound = schema.get(keyword, default)
            if bound is not None and (type(bound) is not int or not compare(len(value), bound)):
                return False
    if isinstance(value, str):
        for keyword, default, compare in (
            ("minLength", 0, lambda size, bound: size >= bound),
            ("maxLength", None, lambda size, bound: size <= bound),
        ):
            bound = schema.get(keyword, default)
            if bound is not None and (type(bound) is not int or not compare(len(value), bound)):
                return False
    return True


def probe_tools(root_url: str, runner_header: str, runner_token: str) -> dict[str, Any]:
    """Exercise bash and submit_report while retaining no tool response content."""
    headers = {
        runner_header: runner_token,
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
    }
    endpoint = root_url.rstrip("/") + "/mcp"
    tools: list[dict[str, Any]] = []
    names: list[str] = []
    with httpx.Client(timeout=120) as client:
        session_id = None
        try:
            init_response = client.post(
                endpoint,
                headers=headers,
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-03-26",
                        "capabilities": {},
                        "clientInfo": {"name": "task-quality-qualification", "version": "1"},
                    },
                },
            )
            init_response.raise_for_status()
            _mcp_result(  # noqa: SLF001
                self_hosted._mcp_json(init_response), 1, "initialize"
            )
            session_id = init_response.headers.get("mcp-session-id")
            if session_id:
                headers["Mcp-Session-Id"] = session_id
            ready = client.post(
                endpoint,
                headers=headers,
                json={"jsonrpc": "2.0", "method": "notifications/initialized"},
            )
            ready.raise_for_status()
            listed_response = client.post(
                endpoint,
                headers=headers,
                json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
            )
            listed_response.raise_for_status()
            listed = self_hosted._mcp_json(listed_response)  # noqa: SLF001
            _mcp_result(listed, 2, "tools/list")
            raw_tools = listed["result"].get("tools")
            if not isinstance(raw_tools, list) or any(
                not isinstance(row, dict) for row in raw_tools
            ):
                raise QualificationError("MCP tool catalog is invalid")
            tools = raw_tools
            by_name = {row.get("name"): row for row in tools if isinstance(row.get("name"), str)}
            names = sorted(by_name)
            if names != REQUIRED_TOOLS or len(by_name) != len(tools):
                raise QualificationError("MCP tool catalog differs from bash/submit_report")
            probes = _probe_arguments()
            if any(
                not _schema_admits(by_name[name].get("inputSchema"), probes[name])
                for name in REQUIRED_TOOLS
            ):
                raise QualificationError("MCP tool schema does not admit the exact negative probe")
            for request_id, name in enumerate(REQUIRED_TOOLS, start=3):
                response = client.post(
                    endpoint,
                    headers=headers,
                    json={
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "method": "tools/call",
                        "params": {"name": name, "arguments": probes[name]},
                    },
                )
                response.raise_for_status()
                _mcp_result(  # noqa: SLF001
                    self_hosted._mcp_json(response), request_id, name
                )
        finally:
            if session_id:
                closed = client.delete(endpoint, headers=headers)
                closed.raise_for_status()
    return {
        "tool_names": names,
        "tool_catalog_sha256": self_hosted.sha256(self_hosted.canonical_json(tools)),
        "bash_reachable": True,
        "submit_report_reachable": True,
        "tool_outputs_persisted": False,
    }


def _failure(error: BaseException, *, phase: str) -> dict[str, Any]:
    value: dict[str, Any] = {"phase": phase, "error_type": type(error).__name__}
    if isinstance(error, self_hosted.FleetRequestError):
        value.update(
            method=error.method,
            route=error.route,
            http_status=error.status_code,
            **error.diagnostic,
        )
    return value


def qualify_one(
    binding: dict[str, Any], *, wave_id: str, directory: Path, api_key: str
) -> dict[str, Any]:
    directory.mkdir(mode=0o700)
    config = _config(binding, wave_id)
    _write_once(
        directory / "CELL_INTENT.json",
        sealed(
            {
                "schema": "cyber_task_quality_qualification_cell_intent_v1",
                "wave_id": wave_id,
                "binding_sha256": binding["binding_sha256"],
                "run_id": config["run_id"],
                "mutating_request_attempts": 1,
                "automatic_retry": False,
            }
        ),
    )
    instance_id: str | None = None
    phase = "preflight"
    passed = False
    failure: dict[str, Any] | None = None
    ambiguous = False
    unbound_mutation = False
    cleanup_complete = False
    evidence: dict[str, Any] = {}
    client = _client(api_key)
    try:
        _account(client)
        observed = _fetch_binding(
            client,
            {
                "task_key": binding["task_key"],
                "task_version_id": binding["task_version_id"],
                "task_shape": "blackbox",
                "qa_status": binding["qa_status"],
                "task_id": binding["task_id"],
            },
        )
        if observed != binding:
            raise QualificationError("live task binding changed after qualification planning")
        authority_gate = self_hosted.assert_authoritative_routes_deployed(client, config)
        create_claim_gate = _assert_create_claim_routes_deployed(client)
        evidence["exact_task_binding"] = True
        evidence["authority_gate"] = authority_gate.get("mode")
        evidence["durable_create_claim_routes_deployed"] = (
            create_claim_gate.get("mode") == "openapi"
        )

        phase = "provisioning"
        request_id = self_hosted.provisioning_request_id(config)
        _write_once(
            directory / "PROVISION_INTENT.json",
            sealed(
                {
                    "schema": "cyber_task_quality_provision_intent_v1",
                    "run_id": config["run_id"],
                    "task_version_id": binding["task_version_id"],
                    "request_id": request_id,
                    "request_body_sha256": digest({}),
                }
            ),
        )
        unbound_mutation = True
        rollout_instance = self_hosted._request(  # noqa: SLF001
            client,
            "POST",
            self_hosted.authoritative_route(config, "provisioning"),
            headers={"X-Request-ID": request_id},
            json={},
        )
        validated_instance_id, evidence_run_id = self_hosted.validate_rollout_instance_response(
            config, rollout_instance
        )
        unbound_mutation = False
        instance = self_hosted._request(  # noqa: SLF001
            client, "GET", f"/v1/env/instances/{validated_instance_id}"
        )
        if (
            instance.get("instance_id") != validated_instance_id
            or instance.get("team_id") != EXPECTED_TEAM_ID
            or instance.get("env_key") != binding["environment"]["id"]
            or instance.get("version") != binding["environment"]["version"]
            or instance.get("status") != "running"
            or instance.get("terminated_at") is not None
            or not isinstance((instance.get("urls") or {}).get("root"), str)
        ):
            raise QualificationError("provisioned environment differs from the exact task binding")
        create_claim = _claim(client, request_id=request_id, config=config)
        if (
            create_claim.get("state") != "materialized"
            or create_claim.get("instance_id") != validated_instance_id
        ):
            raise QualificationError("provisioned instance differs from its exact create claim")
        # Inline cleanup is authorized only after both the response and the
        # live instance/readback claim prove this exact Fleet-owned request.
        instance_id = validated_instance_id
        _write_once(
            directory / "PROVISION_RECEIPT.json",
            sealed(
                {
                    "schema": "cyber_task_quality_provision_receipt_v1",
                    "request_id": request_id,
                    "instance_id": instance_id,
                    "evidence_run_id": evidence_run_id,
                    "task_version_id": binding["task_version_id"],
                }
            ),
        )
        evidence["environment_started"] = True

        phase = "tool_probe"
        token = self_hosted._request(client, "GET", "/v1/runner-auth/token")  # noqa: SLF001
        if not all(
            isinstance(token.get(field), str) and token[field] for field in ("header", "token")
        ):
            raise QualificationError("runner token response is invalid")
        tool_receipt = probe_tools(instance["urls"]["root"], token["header"], token["token"])
        _write_once(directory / "TOOL_PROBE_RECEIPT.json", sealed(tool_receipt))
        evidence["bash_reachable"] = True
        evidence["submit_report_reachable"] = True

        phase = "scoring"
        scoring_payload = self_hosted.build_scoring_payload(
            config, instance_id=instance_id, final_answer="", messages=[]
        )
        scoring_intent = self_hosted.build_scoring_intent(
            config,
            instance_id=instance_id,
            evidence_run_id=evidence_run_id,
            scoring_payload=scoring_payload,
        )
        _write_once(directory / "SCORING_INTENT.json", scoring_intent)
        unbound_mutation = True
        reward_response = self_hosted._request(  # noqa: SLF001
            client,
            "POST",
            self_hosted.authoritative_route(config, "scoring"),
            json=scoring_payload,
        )
        reward = self_hosted.sanitize_authoritative_reward_response(
            config,
            reward_response,
            instance_id=instance_id,
            evidence_run_id=evidence_run_id,
        )
        unbound_mutation = False
        score = reward["reward"]
        if score != 0.0:
            raise QualificationError("fixed no_flag negative control returned nonzero reward")
        verifier_execution_id = reward["verifier_execution_id"]
        _write_once(
            directory / "VERIFIER_RECEIPT.json",
            sealed(
                {
                    "schema": "cyber_task_quality_verifier_receipt_v1",
                    "task_version_id": binding["task_version_id"],
                    "instance_id": instance_id,
                    "evidence_run_id": evidence_run_id,
                    "verifier_execution_id": verifier_execution_id,
                    "direct_authority_attestation_sha256": digest(
                        reward["direct_authority_attestation"]
                    ),
                    "verifier_completed": True,
                    "finite_authoritative_outcome": True,
                    "numeric_score_in_controller_receipt": False,
                }
            ),
        )
        evidence["verifier_completed"] = True
        evidence["finite_authoritative_outcome"] = True

        phase = "session_ingest"
        _write_once(
            directory / "SESSION_INGEST_INTENT.json",
            sealed(
                {
                    "schema": "cyber_task_quality_session_ingest_intent_v1",
                    "task_version_id": binding["task_version_id"],
                    "instance_id": instance_id,
                    "evidence_run_id": evidence_run_id,
                    "verifier_execution_id": verifier_execution_id,
                    "message_count": 0,
                    "trace_persisted": False,
                }
            ),
        )
        unbound_mutation = True
        ingest = self_hosted.ingest_metadata_only_session(
            client,
            config=config,
            instance_id=instance_id,
            evidence_run_id=evidence_run_id,
            score=score,
            verifier_execution_id=verifier_execution_id,
        )
        unbound_mutation = False
        if ingest.get("created_new_session") is not True:
            raise QualificationError("metadata-only evidence session was not created exactly once")
        _write_once(
            directory / "SESSION_INGEST_RECEIPT.json",
            sealed(
                {
                    "schema": "cyber_task_quality_session_ingest_receipt_v1",
                    "session_id": ingest["session_id"],
                    "evidence_run_id": evidence_run_id,
                    "verifier_execution_id": verifier_execution_id,
                    "message_count": 0,
                    "trace_persisted": False,
                    "metadata_only": True,
                    "created_new_session": True,
                    "authoritative_session_outcome_persisted": True,
                    "numeric_score_in_controller_receipt": False,
                }
            ),
        )
        evidence["session_ingested"] = True
        passed = True
    except BaseException as error:  # noqa: BLE001
        failure = _failure(error, phase=phase)
        ambiguous = unbound_mutation
    finally:
        phase = "cleanup"
        if instance_id is not None:
            try:
                _write_once(
                    directory / "CLEANUP_INTENT.json",
                    sealed(
                        {
                            "schema": "cyber_task_quality_cleanup_intent_v1",
                            "instance_id": instance_id,
                            "delete_attempts": 1,
                        }
                    ),
                )
                deleted = self_hosted._request(  # noqa: SLF001
                    client, "DELETE", f"/v1/env/instances/{instance_id}"
                )
                if (
                    not isinstance(deleted.get("terminated_at"), str)
                    or not deleted["terminated_at"]
                ):
                    raise QualificationError("instance delete returned no termination evidence")
                cleanup_complete = True
                _write_once(
                    directory / "CLEANUP_RECEIPT.json",
                    sealed(
                        {
                            "schema": "cyber_task_quality_cleanup_receipt_v1",
                            "instance_id": instance_id,
                            "terminated": True,
                            "terminated_at": deleted["terminated_at"],
                        }
                    ),
                )
            except BaseException as cleanup_error:  # noqa: BLE001
                cleanup_complete = False
                if failure is None:
                    failure = _failure(cleanup_error, phase="cleanup")
                passed = False
        client.close()
    evidence_files = {
        path.name: file_digest(path)
        for path in sorted(directory.iterdir(), key=lambda item: item.name)
        if path.is_file() and path.name != "CELL_TERMINAL.json"
    }
    status = (
        "qualified"
        if passed and cleanup_complete
        else ("quarantined_ambiguous" if ambiguous else "infrastructure_invalid")
    )
    body = {
        "schema": CELL_TERMINAL_SCHEMA,
        "wave_id": wave_id,
        "binding_sha256": binding["binding_sha256"],
        "qualification_status": status,
        "checks": {
            "exact_task_binding": evidence.get("exact_task_binding", False),
            "durable_create_claim_routes_deployed": evidence.get(
                "durable_create_claim_routes_deployed", False
            ),
            "environment_started": evidence.get("environment_started", False),
            "bash_reachable": evidence.get("bash_reachable", False),
            "submit_report_reachable": evidence.get("submit_report_reachable", False),
            "verifier_completed": evidence.get("verifier_completed", False),
            "finite_authoritative_outcome": evidence.get("finite_authoritative_outcome", False),
            "metadata_only_session_ingested": evidence.get("session_ingested", False),
            "environment_cleanup_completed": cleanup_complete,
        },
        "failure": failure,
        "ambiguous_external_mutation": ambiguous,
        "automatic_retry_performed": False,
        "model_calls": 0,
        "training_data_eligible": False,
        "prompt_trace_answer_flag_or_credential_persisted": False,
        "numeric_score_persisted_in_controller_artifacts": False,
        "evidence_file_sha256": evidence_files,
    }
    receipt = sealed(body)
    _write_once(directory / "CELL_TERMINAL.json", receipt)
    return receipt


def execute_plan(plan: dict[str, Any], root: Path, *, api_key: str) -> dict[str, Any]:
    _sealed(plan, PLAN_SCHEMA, "qualification plan")
    _source_matches_plan(plan.get("source"))
    if plan.get("execution", {}).get("external_mutations_authorized") is not True:
        raise QualificationError("qualification plan does not authorize external mutations")
    intent = sealed(
        {
            "schema": RUN_INTENT_SCHEMA,
            "plan_sha256": plan["sha256"],
            "wave_id": plan["wave_id"],
            "task_versions": len(plan["tasks"]),
            "concurrency": plan["execution"]["concurrency"],
            "one_shot": True,
        }
    )
    _write_once(root / "RUN_INTENT.json", intent)
    cells_root = root / "cells"
    cells_root.mkdir(mode=0o700)
    with _client(api_key) as client:
        _account(client)
    terminals: list[tuple[dict[str, Any], dict[str, Any]]] = []
    with ThreadPoolExecutor(max_workers=plan["execution"]["concurrency"]) as pool:
        pending = {
            pool.submit(
                qualify_one,
                binding,
                wave_id=plan["wave_id"],
                directory=cells_root / f"cell-{index:03d}",
                api_key=api_key,
            ): binding
            for index, binding in enumerate(plan["tasks"])
        }
        for future in as_completed(pending):
            terminals.append((pending[future], future.result()))
    terminals.sort(key=lambda item: (item[0]["task_key"], item[0]["task_version_id"]))
    counts = {
        status: 0 for status in ("qualified", "infrastructure_invalid", "quarantined_ambiguous")
    }
    gate_counts = {
        name: 0
        for name in (
            "exact_task_binding",
            "durable_create_claim_routes_deployed",
            "environment_started",
            "bash_reachable",
            "submit_report_reachable",
            "verifier_completed",
            "finite_authoritative_outcome",
            "metadata_only_session_ingested",
            "environment_cleanup_completed",
        )
    }
    catalog_rows = []
    for binding, terminal in terminals:
        status = terminal["qualification_status"]
        counts[status] += 1
        for name, passed in terminal["checks"].items():
            gate_counts[name] += int(passed)
        if status != "qualified":
            continue
        environment = binding["environment"]
        legacy_data_id = environment.get("legacy_data_id")
        legacy_data_version = environment.get("legacy_data_version")
        if not isinstance(legacy_data_id, str) or not isinstance(legacy_data_version, str):
            first_seed = next(iter(environment["seed_config"].values()))
            legacy_data_id = first_seed["data_key"]
            legacy_data_version = first_seed["data_version"]
        catalog_rows.append(
            {
                "task_key": binding["task_key"],
                "task_version_id": binding["task_version_id"],
                "lineage": binding["lineage"],
                "runtime": {
                    "task_key": binding["task_key"],
                    "task_version_id": binding["task_version_id"],
                    "env_key": environment["id"],
                    "env_version": environment["version"],
                    "environment_version_id": environment["version_id"],
                    "data_key": legacy_data_id,
                    "data_version": legacy_data_version,
                },
                "qualification_receipt_sha256": terminal["sha256"],
            }
        )
    supply = sealed(
        {
            "schema": SUPPLY_CATALOG_SCHEMA,
            "catalog_snapshot_sha256": plan["inputs"]["inventory_sha256"],
            "task_versions": [
                {
                    "task_key": binding["task_key"],
                    "task_version_id": binding["task_version_id"],
                }
                for binding in plan["tasks"]
            ],
        }
    )
    _write_once(root / "SUPPLY_CATALOG.private.json", supply)
    attempted_rows = [
        *plan.get("prior_attempted_task_versions", []),
        *(
            {
                "task_key": binding["task_key"],
                "task_version_id": binding["task_version_id"],
                "lineage": binding["lineage"],
            }
            for binding in plan["tasks"]
        ),
    ]
    attempted = sealed(
        {
            "schema": ATTEMPTED_CATALOG_SCHEMA,
            "catalog_snapshot_sha256": plan["inputs"]["inventory_sha256"],
            "task_versions": sorted(
                attempted_rows, key=lambda row: (row["task_key"], row["task_version_id"])
            ),
        }
    )
    _write_once(root / "ATTEMPTED_CATALOG.private.json", attempted)
    catalog = sealed(
        {
            "schema": PRIVATE_CATALOG_SCHEMA,
            "supply_catalog_sha256": supply["sha256"],
            "catalog_snapshot_sha256": plan["inputs"]["inventory_sha256"],
            "task_validity_receipt_sha256": digest(
                sorted(row["qualification_receipt_sha256"] for row in catalog_rows)
            ),
            "task_versions": catalog_rows,
        }
    )
    _write_once(root / "QUALIFIED_CATALOG.private.json", catalog)
    aggregate = sealed(
        {
            "schema": AGGREGATE_SCHEMA,
            "plan_sha256": plan["sha256"],
            "wave_id": plan["wave_id"],
            "planned_task_versions": len(plan["tasks"]),
            "counts": counts,
            "gate_pass_counts": gate_counts,
            "qualified_catalog_sha256": catalog["sha256"],
            "supply_catalog_sha256": supply["sha256"],
            "attempted_catalog_sha256": attempted["sha256"],
            "protected_heldout_atom_keys_sha256": plan["selection"][
                "protected_heldout_atom_keys_sha256"
            ],
            "zero_protected_heldout_atom_intersection": True,
            "outcome_selected_promotion": False,
            "success_and_failure_outcomes_both_eligible": True,
            "model_calls": 0,
            "training_data_eligible": False,
            "public_evidence": {
                "aggregate_only": True,
                "task_session_instance_or_verifier_ids_included": False,
                "prompts_traces_answers_flags_credentials_included": False,
                "numeric_scores_included": False,
            },
        }
    )
    _write_once(root / "AGGREGATE_RECEIPT.json", aggregate)
    return aggregate


def _intent_once_or_exact(path: Path, body: dict[str, Any]) -> dict[str, Any]:
    expected = sealed(body)
    if path.is_file():
        observed = _read(path, path.name)
        if observed != expected:
            raise QualificationError(f"{path.name} differs from the exact cleanup target")
        return observed
    _write_once(path, expected)
    return expected


def _optional_instance(client: httpx.Client, instance_id: str) -> dict[str, Any] | None:
    response = client.get(f"{self_hosted.ORCHESTRATOR}/v1/env/instances/{instance_id}")
    if response.status_code == 404:
        return None
    if response.status_code >= 400:
        raise self_hosted.FleetRequestError(
            "GET",
            f"/v1/env/instances/{instance_id}",
            response.status_code,
            diagnostic=self_hosted.request_diagnostic(response),
        )
    value = response.json()
    if not isinstance(value, dict):
        raise QualificationError("instance readback is not an object")
    return value


def _claim(client: httpx.Client, *, request_id: str, config: dict[str, Any]) -> dict[str, Any]:
    value = self_hosted._request(  # noqa: SLF001
        client, "GET", f"/v1/env/instances/create-requests/{request_id}"
    )
    if (
        not isinstance(value, dict)
        or value.get("request_id") != request_id
        or value.get("run_id") != config["run_id"]
        or value.get("team_id") != EXPECTED_TEAM_ID
        or value.get("state") not in {"accepted", "materialized", "failed", "cancelled"}
    ):
        raise QualificationError("durable create-request claim binding is invalid")
    instance_id = value.get("instance_id")
    if instance_id is not None:
        self_hosted._instance_identifier(instance_id)  # noqa: SLF001
    if value["state"] == "materialized" and instance_id is None:
        raise QualificationError("materialized create-request claim has no instance")
    return value


def _cleanup_resolution(
    directory: Path,
    *,
    binding: dict[str, Any],
    request_id: str | None,
    instance_id: str | None,
    resolution: str,
) -> dict[str, Any]:
    receipt = sealed(
        {
            "schema": CLEANUP_RESOLUTION_SCHEMA,
            "binding_sha256": binding["binding_sha256"],
            "request_id": request_id,
            "instance_id": instance_id,
            "resolution": resolution,
            "instance_live_after": False,
            "resumable_exact_target_only": True,
        }
    )
    path = directory / "CLEANUP_RESOLUTION.json"
    if path.is_file():
        observed = _read(path, path.name)
        if observed != receipt:
            raise QualificationError("cleanup resolution differs from the exact target")
        return observed
    _write_once(path, receipt)
    return receipt


def cleanup_one(
    binding: dict[str, Any], *, directory: Path, wave_id: str, api_key: str
) -> dict[str, Any]:
    """Resolve one exact create claim/instance without replaying qualification."""
    resolution_path = directory / "CLEANUP_RESOLUTION.json"
    if resolution_path.is_file():
        resolution = _read(resolution_path, resolution_path.name)
        _sealed(resolution, CLEANUP_RESOLUTION_SCHEMA, "cleanup resolution")
        if resolution.get("binding_sha256") != binding["binding_sha256"]:
            raise QualificationError("cleanup resolution binding changed")
        return resolution

    provision_intent_path = directory / "PROVISION_INTENT.json"
    if not provision_intent_path.is_file():
        return _cleanup_resolution(
            directory,
            binding=binding,
            request_id=None,
            instance_id=None,
            resolution="no_provision_attempt",
        )
    provision_intent = _read(provision_intent_path, provision_intent_path.name)
    _sealed(provision_intent, "cyber_task_quality_provision_intent_v1", "provision intent")
    config = _config(binding, wave_id)
    request_id = self_hosted.provisioning_request_id(config)
    if (
        provision_intent.get("run_id") != config["run_id"]
        or provision_intent.get("task_version_id") != binding["task_version_id"]
        or provision_intent.get("request_id") != request_id
    ):
        raise QualificationError("provision intent differs from the cleanup target")

    client = _client(api_key)
    try:
        _account(client)
        provision_receipt_path = directory / "PROVISION_RECEIPT.json"
        instance_id: str | None = None
        if provision_receipt_path.is_file():
            provision_receipt = _read(provision_receipt_path, provision_receipt_path.name)
            _sealed(
                provision_receipt,
                "cyber_task_quality_provision_receipt_v1",
                "provision receipt",
            )
            if (
                provision_receipt.get("task_version_id") != binding["task_version_id"]
                or provision_receipt.get("request_id") != request_id
            ):
                raise QualificationError("provision receipt request/task binding changed")
            instance_id = self_hosted._instance_identifier(  # noqa: SLF001
                provision_receipt.get("instance_id")
            )
        else:
            claim = _claim(client, request_id=request_id, config=config)
            instance_id = claim.get("instance_id")
            if instance_id is None and claim["state"] == "accepted":
                _intent_once_or_exact(
                    directory / "CREATE_CLAIM_CANCELLATION_INTENT.json",
                    {
                        "schema": "cyber_task_quality_create_claim_cancellation_intent_v1",
                        "binding_sha256": binding["binding_sha256"],
                        "request_id": request_id,
                        "run_id": config["run_id"],
                    },
                )
                claim = self_hosted._request(  # noqa: SLF001
                    client,
                    "DELETE",
                    f"/v1/env/instances/create-requests/{request_id}",
                )
                if (
                    not isinstance(claim, dict)
                    or claim.get("request_id") != request_id
                    or claim.get("run_id") != config["run_id"]
                    or claim.get("team_id") != EXPECTED_TEAM_ID
                    or claim.get("state") not in {"materialized", "failed", "cancelled"}
                ):
                    raise QualificationError("create-request cancellation response is invalid")
                instance_id = claim.get("instance_id")
            if instance_id is None:
                if claim["state"] not in {"failed", "cancelled"}:
                    raise QualificationError("create-request claim remains unresolved")
                return _cleanup_resolution(
                    directory,
                    binding=binding,
                    request_id=request_id,
                    instance_id=None,
                    resolution=f"claim_{claim['state']}_without_instance",
                )
            instance_id = self_hosted._instance_identifier(instance_id)  # noqa: SLF001

        initial_cleanup = directory / "CLEANUP_RECEIPT.json"
        if initial_cleanup.is_file():
            receipt = _read(initial_cleanup, initial_cleanup.name)
            _sealed(receipt, "cyber_task_quality_cleanup_receipt_v1", "initial cleanup")
            if receipt.get("instance_id") != instance_id or receipt.get("terminated") is not True:
                raise QualificationError("initial cleanup receipt binding is invalid")
            return _cleanup_resolution(
                directory,
                binding=binding,
                request_id=request_id,
                instance_id=instance_id,
                resolution="initial_delete_terminated",
            )

        observed = _optional_instance(client, instance_id)
        if observed is None:
            return _cleanup_resolution(
                directory,
                binding=binding,
                request_id=request_id,
                instance_id=instance_id,
                resolution="instance_already_absent",
            )
        if (
            observed.get("instance_id") != instance_id
            or observed.get("team_id") != EXPECTED_TEAM_ID
            or observed.get("env_key") != binding["environment"]["id"]
            or observed.get("version") != binding["environment"]["version"]
        ):
            raise QualificationError("cleanup instance differs from the exact owned runtime")
        if observed.get("terminated_at") is not None:
            return _cleanup_resolution(
                directory,
                binding=binding,
                request_id=request_id,
                instance_id=instance_id,
                resolution="instance_already_terminated",
            )

        # Reuse this exact intent after a transport loss.  A later invocation
        # first re-reads the same instance and may safely reissue only this
        # exact-id delete; qualification/scoring/ingestion are never replayed.
        _intent_once_or_exact(
            directory / "CLEANUP_RECOVERY_INTENT.json",
            {
                "schema": "cyber_task_quality_cleanup_recovery_intent_v1",
                "binding_sha256": binding["binding_sha256"],
                "request_id": request_id,
                "instance_id": instance_id,
            },
        )
        deleted = self_hosted._request(  # noqa: SLF001
            client, "DELETE", f"/v1/env/instances/{instance_id}"
        )
        if not isinstance(deleted.get("terminated_at"), str) or not deleted["terminated_at"]:
            raise QualificationError("cleanup recovery returned no termination evidence")
        return _cleanup_resolution(
            directory,
            binding=binding,
            request_id=request_id,
            instance_id=instance_id,
            resolution="recovery_delete_terminated",
        )
    finally:
        client.close()


def cleanup_plan(plan: dict[str, Any], root: Path, *, api_key: str) -> dict[str, Any]:
    _sealed(plan, PLAN_SCHEMA, "qualification plan")
    _source_matches_plan(plan.get("source"))
    resolved = 0
    unresolved = 0
    receipt_digests: list[str] = []
    for index, binding in enumerate(plan["tasks"]):
        directory = root / "cells" / f"cell-{index:03d}"
        if not directory.is_dir():
            unresolved += 1
            continue
        try:
            receipt = cleanup_one(
                binding,
                directory=directory,
                wave_id=plan["wave_id"],
                api_key=api_key,
            )
        except BaseException:  # noqa: BLE001 - content-free resumable status
            unresolved += 1
            continue
        resolved += 1
        receipt_digests.append(receipt["sha256"])
    result = {
        "schema": CLEANUP_AGGREGATE_SCHEMA,
        "plan_sha256": plan["sha256"],
        "planned_task_versions": len(plan["tasks"]),
        "resolved_task_versions": resolved,
        "unresolved_task_versions": unresolved,
        "resolution_receipts_sha256": digest(sorted(receipt_digests)),
        "qualification_replayed": False,
        "exact_instance_or_create_claim_cleanup_only": True,
        "logs_prompts_traces_scores_or_credentials_read": False,
    }
    if unresolved == 0:
        receipt = sealed(result)
        path = root / "CLEANUP_AGGREGATE.json"
        if path.is_file():
            observed = _read(path, path.name)
            if observed != receipt:
                raise QualificationError("cleanup aggregate differs from existing receipt")
            return observed
        _write_once(path, receipt)
        return receipt
    return {**result, "complete": False}


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    inventory_path = Path(args.inventory)
    coverage_path = Path(args.coverage)
    split_path = Path(args.protected_split)
    inventory = _input(inventory_path, INVENTORY_SCHEMA, "current inventory")
    coverage = _input(coverage_path, COVERAGE_SCHEMA, "receipt coverage")
    split = _input_one_of(
        split_path,
        {SPLIT_SCHEMA, HELDOUT_PROTOCOL_SCHEMA},
        "protected family split",
    )
    excluded = (
        _read(Path(args.exclude_catalog), "excluded catalog") if args.exclude_catalog else None
    )
    with _client(os.environ.get("FLEET_API_KEY", "")) as client:
        _account(client)
        plan = build_plan(
            inventory=inventory,
            coverage=coverage,
            split=split,
            inventory_sha256=file_digest(inventory_path),
            coverage_sha256=file_digest(coverage_path),
            split_sha256=file_digest(split_path),
            client=client,
            wave_id=args.wave_id,
            qa_statuses=set(args.qa_status),
            limit=args.limit,
            concurrency=args.concurrency,
            source=source_provenance(require_merged=not args.allow_unmerged_preview),
            excluded_catalog=excluded,
        )
    root = Path(args.private_base) / (
        f"task-quality-{args.wave_id}-{plan['sha256'].removeprefix('sha256:')[:12]}"
    )
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    _write_once(root / "PLAN.json", plan)
    return {
        "schema": "cyber_task_quality_qualification_prepare_receipt_v1",
        "private_root": str(root),
        "plan_sha256": plan["sha256"],
        "selected_task_versions": len(plan["tasks"]),
        "zero_protected_heldout_atom_intersection": True,
        "external_mutations": 0,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.private_root)
    plan = _read(root / "PLAN.json", "qualification plan")
    return execute_plan(plan, root, api_key=os.environ.get("FLEET_API_KEY", ""))


def cleanup(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.private_root)
    plan = _read(root / "PLAN.json", "qualification plan")
    return cleanup_plan(plan, root, api_key=os.environ.get("FLEET_API_KEY", ""))


def status(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.private_root)
    plan = _read(root / "PLAN.json", "qualification plan")
    _sealed(plan, PLAN_SCHEMA, "qualification plan")
    terminals = list((root / "cells").glob("cell-*/CELL_TERMINAL.json"))
    aggregate_path = root / "AGGREGATE_RECEIPT.json"
    return {
        "schema": "cyber_task_quality_qualification_status_v1",
        "plan_sha256": plan["sha256"],
        "planned_task_versions": len(plan["tasks"]),
        "terminal_task_versions": len(terminals),
        "aggregate_receipt_present": aggregate_path.is_file(),
        "cleanup_aggregate_present": (root / "CLEANUP_AGGREGATE.json").is_file(),
        "logs_prompts_traces_scores_or_credentials_read": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--inventory", required=True)
    prepare_parser.add_argument("--coverage", required=True)
    prepare_parser.add_argument("--protected-split", required=True)
    prepare_parser.add_argument("--wave-id", required=True)
    prepare_parser.add_argument("--qa-status", action="append", required=True)
    prepare_parser.add_argument("--limit", type=int, required=True)
    prepare_parser.add_argument("--concurrency", type=int, required=True)
    prepare_parser.add_argument("--exclude-catalog")
    prepare_parser.add_argument("--private-base", required=True)
    prepare_parser.add_argument("--allow-unmerged-preview", action="store_true")
    run_parser = commands.add_parser("run")
    run_parser.add_argument("private_root")
    cleanup_parser = commands.add_parser("cleanup")
    cleanup_parser.add_argument("private_root")
    status_parser = commands.add_parser("status")
    status_parser.add_argument("private_root")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = {"prepare": prepare, "run": run, "cleanup": cleanup, "status": status}[
            args.command
        ](args)
    except BaseException as error:  # noqa: BLE001
        print(
            json.dumps(
                {
                    "schema": "cyber_task_quality_qualification_cli_failure_v1",
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
