"""UID-bound score-blind release observer for the peer-free GLM rank-30 Job."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import ssl
import stat
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evals.fleet import hosted_glm_exact_bulk_runtime_v1 as source_runtime
from evals.fleet import hosted_glm_rank30_peer_free_successor_v2 as successor

NAMESPACE = "fleet-train-jobs"
FLEET_TEAM_ID = "a1025f0b-ad67-49fc-a023-51800ab43e84"
ORCHESTRATOR = "https://orchestrator.fleetai.com"
SCHEMA = "fleet-hosted-glm-rank30-peer-free-observer-binding-v1"
PACKAGE_SCHEMA = "fleet-hosted-glm-rank30-peer-free-observer-package-v1"
OUTPUT_ROOT = Path("/mnt/sfs/jobs/chris-glm53-peer-free-release-observer-v1")
OUTPUT_PATH = OUTPUT_ROOT / "RELEASE.json"
OBSERVER_JOB_NAME = "chris-glm53-peer-free-release-observer-v1"
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_RECEIPT_BYTES = 2 * 1024 * 1024
SHA_RE = re.compile(r"sha256:[0-9a-f]{64}")
RANK30_PATH_MARKERS = ("r030", "rank30", "rank-30")
CONTROLLER_SOURCE_PATHS = {
    "Dockerfile.opencode": "evals/fleet/Dockerfile.opencode",
    "fixed_proxy.py": "evals/fleet/fixed_proxy.py",
    "self_hosted.py": "evals/fleet/self_hosted.py",
    "runner.py": "evals/fleet/opencode_train_sweep_runner.py",
    "endpoint_lease.py": "evals/fleet/endpoint_lease.py",
    "universe.py": "evals/fleet/exact_pass4_universe.py",
    "crypto.py": "evals/fleet/exact_pass4_crypto.py",
    "inventory.py": "evals/fleet/exact_pass4_task_inventory.py",
    "campaign.json": "evals/fleet/configs/q38-glm53-exact-easiest100-pass4-campaign-v1.json",
    "selection.json": "evals/fleet/configs/opencode-easiest-train100-selection-v2.json",
    "glm-template.json": "evals/fleet/configs/glm53-opencode-autocontinue-canary1-v1.json",
    "qwen-template.json": "evals/fleet/configs/qwen38-opencode-autocontinue-canary1-v1.json",
    "bulk-qwen-a.json": "evals/fleet/configs/exact-pass4-bulk-qwen-a-v3.json",
    "bulk-qwen-b.json": "evals/fleet/configs/exact-pass4-bulk-qwen-b-v3.json",
    "bulk-glm-a.json": "evals/fleet/configs/exact-pass4-bulk-glm-a-v3.json",
    "bulk-glm-b.json": "evals/fleet/configs/exact-pass4-bulk-glm-b-v3.json",
    "predecessor.py": "evals/fleet/exact_pass4_bulk_v3.py",
    "engine.py": "evals/fleet/hosted_glm_whole_task_engine_v1.py",
    "bulk.py": "evals/fleet/hosted_glm_rank30_peer_free_successor_v2.py",
    "bulk_runtime.py": "evals/fleet/hosted_glm_rank30_peer_free_runtime_v2.py",
    "whole_task_v1.py": "evals/fleet/hosted_glm_whole_task_successor_v1.py",
    "base_engine.py": "evals/fleet/exact_pass4_bulk_runtime_v3.py",
    "hosted_glm_exact_bulk_v1.py": "evals/fleet/hosted_glm_exact_bulk_v1.py",
    "hosted_glm_exact_bulk_runtime_v1.py": "evals/fleet/hosted_glm_exact_bulk_runtime_v1.py",
    "run.sh": "evals/fleet/scripts/run_hosted_glm_rank30_peer_free_v2.sh",
}
EXPECTED_CELL_IDS = [
    "sha256:5af2152499b8af043a847707e47cc3c04b81d24c47aa99c33c04a8688cbcafa6",
    "sha256:7e2fffc5a21d93d76e97ab387800a687669bf39e69763d8a5b874dfea4da1f41",
    "sha256:b77c81587baee31eabe1bf65903749e8a89cf7719c38e8f527fd5bc53e087121",
    "sha256:2b6f93f57e1862f339e55ff758ce6258a9665c66df80ad6d67fee21fa9fbe66f",
]
EXPECTED_TASK_KEYS = ["cysec1-2-fira-gen_blackbox-df372d5a31676a815c9f0751__blackbox_ctf_v1"]
EXPECTED_ALLOWED_PATHS = [
    "/mnt/sfs/jobs/chris-glm53-r030-preclaim-phase-observer-v2"
]
EXPECTED_TASK_VERSION_ID = "a0cacaaf-480b-4a4c-9ed7-6b6192bb6783"
EXPECTED_SESSION_MODEL = "fleet-cluster-opencode-1.18.27/glm-5.3-opencode11827-autocontinue-v1"
SESSION_IDENTITY_PATH = "/v1/sessions/identities"
SESSION_IDENTITY_KEYS = {
    "session_id",
    "eval_task_id",
    "eval_task_version_id",
    "task_key",
    "model_identity",
    "model_identity_status",
    "status",
}


class ObserverError(RuntimeError):
    """The score-blind observer failed closed."""


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def sha256(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def digest(value: dict[str, Any]) -> str:
    return sha256(canonical({key: item for key, item in value.items() if key != "receipt_sha256"}))


def controller_source_sha256(root: Path) -> str:
    data: dict[str, str] = {}
    for name, relative in CONTROLLER_SOURCE_PATHS.items():
        source = root / relative
        if source.is_symlink() or not source.is_file():
            raise ObserverError("controller_source_path_unsafe")
        data[name] = source.read_text()
    return sha256(canonical(data))


def strict_json(raw: bytes) -> dict[str, Any]:
    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in rows:
            if key in value:
                raise ObserverError("duplicate_json_key")
            value[key] = item
        return value

    try:
        value = json.loads(raw, object_pairs_hook=pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ObserverError("invalid_json") from exc
    if not isinstance(value, dict):
        raise ObserverError("json_root_invalid")
    return value


def load(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_JSON_BYTES:
        raise ObserverError("input_path_unsafe")
    return strict_json(path.read_bytes())


def load_projected(path: Path, package_root: Path) -> dict[str, Any]:
    try:
        root = package_root.resolve(strict=True)
        source = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ObserverError("projected_input_unsafe") from exc
    if not source.is_relative_to(root) or not source.is_file():
        raise ObserverError("projected_input_unsafe")
    return strict_json(source.read_bytes())


def write_once(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o400)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ObserverError("output_not_regular")
        raw = canonical(value) + b"\n"
        if os.write(fd, raw) != len(raw):
            raise ObserverError("output_short_write")
        os.fsync(fd)
    finally:
        os.close(fd)


def _uuid(value: Any, label: str) -> str:
    try:
        parsed = uuid.UUID(str(value))
    except (AttributeError, ValueError) as exc:
        raise ObserverError(f"{label}_invalid") from exc
    if parsed.int == 0:
        raise ObserverError(f"{label}_invalid")
    return str(parsed)


class _RejectRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *_args: Any, **_kwargs: Any) -> None:
        raise ObserverError("redirect_forbidden")


def _request_json(
    url: str, *, headers: dict[str, str], cafile: str | None = None
) -> dict[str, Any]:
    handlers: list[Any] = [_RejectRedirects()]
    if cafile is not None:
        handlers.append(
            urllib.request.HTTPSHandler(context=ssl.create_default_context(cafile=cafile))
        )
    request = urllib.request.Request(url, method="GET", headers=headers)
    try:
        with urllib.request.build_opener(*handlers).open(request, timeout=60) as response:
            raw = response.read(MAX_JSON_BYTES + 1)
    except (OSError, urllib.error.URLError) as exc:
        raise ObserverError("read_request_failed") from exc
    if len(raw) > MAX_JSON_BYTES:
        raise ObserverError("read_response_too_large")
    return strict_json(raw)


def _kubernetes(path: str) -> dict[str, Any]:
    host = os.environ.get("KUBERNETES_SERVICE_HOST")
    port = os.environ.get("KUBERNETES_SERVICE_PORT_HTTPS", "443")
    if not host or not path.startswith("/"):
        raise ObserverError("kubernetes_authority_invalid")
    token = Path("/var/run/secrets/kubernetes.io/serviceaccount/token").read_text().strip()
    return _request_json(
        f"https://{host}:{port}{path}",
        headers={"Authorization": "Bearer " + token, "Accept": "application/json"},
        cafile="/var/run/secrets/kubernetes.io/serviceaccount/ca.crt",
    )


def _kubernetes_items(path: str, label: str) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    token = ""
    gets = 0
    while True:
        query = {"limit": 500}
        if token:
            query["continue"] = token
        page = _kubernetes(path + "?" + urllib.parse.urlencode(query))
        gets += 1
        rows.extend(_list_items(page, label))
        metadata = page.get("metadata")
        if not isinstance(metadata, dict):
            raise ObserverError(f"{label}_inventory_metadata_invalid")
        next_token = metadata.get("continue", "")
        if not isinstance(next_token, str):
            raise ObserverError(f"{label}_inventory_continue_invalid")
        if not next_token:
            return rows, gets
        if next_token == token:
            raise ObserverError(f"{label}_inventory_pagination_stalled")
        token = next_token


def _list_items(value: dict[str, Any], label: str) -> list[dict[str, Any]]:
    items = value.get("items")
    if not isinstance(items, list) or not all(isinstance(row, dict) for row in items):
        raise ObserverError(f"{label}_inventory_invalid")
    return items


def _is_nonterminal_job(row: dict[str, Any]) -> bool:
    status = row.get("status") or {}
    conditions = status.get("conditions") or []
    terminal = any(
        item.get("type") in {"Complete", "Failed"} and item.get("status") == "True"
        for item in conditions
        if isinstance(item, dict)
    )
    return not terminal and (row.get("metadata") or {}).get("deletionTimestamp") is None


def _kubernetes_clear(
    binding: dict[str, Any],
    *,
    observer_job_uid: str | None = None,
    observer_pod_uid: str | None = None,
) -> dict[str, int]:
    jobs, job_gets = _kubernetes_items(
        f"/apis/batch/v1/namespaces/{NAMESPACE}/jobs", "job"
    )
    pods, pod_gets = _kubernetes_items(
        f"/api/v1/namespaces/{NAMESPACE}/pods", "pod"
    )
    configmaps, configmap_gets = _kubernetes_items(
        f"/api/v1/namespaces/{NAMESPACE}/configmaps", "configmap"
    )
    target_job = binding["fresh_job_name"]
    target_configmap = binding["fresh_configmap_name"]
    fresh_job = sum((row.get("metadata") or {}).get("name") == target_job for row in jobs)
    fresh_pod = sum(
        (row.get("metadata") or {}).get("labels", {}).get("job-name") == target_job
        for row in pods
    )
    fresh_configmap = sum(
        (row.get("metadata") or {}).get("name") == target_configmap for row in configmaps
    )
    active = 0
    for row in jobs:
        metadata = row.get("metadata") or {}
        labels = metadata.get("labels") or {}
        name = str(metadata.get("name", ""))
        experiment = str(labels.get("cyber-post-train.fleet.ai/experiment", ""))
        owner = labels.get("cyber-post-train.fleet.ai/owner")
        hosted_glm = "glm" in (name + experiment).lower() and "hosted" in (
            name + experiment
        ).lower()
        active += int(owner == "chris" and hosted_glm and _is_nonterminal_job(row))
    if (observer_job_uid is None) != (observer_pod_uid is None):
        raise ObserverError("observer_identity_incomplete")
    if observer_job_uid is not None and observer_pod_uid is not None:
        observer_jobs = [
            row
            for row in jobs
            if (row.get("metadata") or {}).get("name") == OBSERVER_JOB_NAME
        ]
        observer_pods = [
            row
            for row in pods
            if (row.get("metadata") or {}).get("labels", {}).get("job-name")
            == OBSERVER_JOB_NAME
        ]
        if len(observer_jobs) != 1 or len(observer_pods) != 1:
            raise ObserverError("observer_identity_not_unique")
        job_metadata = observer_jobs[0].get("metadata") or {}
        pod_metadata = observer_pods[0].get("metadata") or {}
        owners = pod_metadata.get("ownerReferences") or []
        expected_owner = {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "name": OBSERVER_JOB_NAME,
            "uid": observer_job_uid,
            "controller": True,
            "blockOwnerDeletion": True,
        }
        if any(
            (
                job_metadata.get("uid") != observer_job_uid,
                pod_metadata.get("uid") != observer_pod_uid,
                owners != [expected_owner],
                not _is_nonterminal_job(observer_jobs[0]),
            )
        ):
            raise ObserverError("observer_identity_mismatch")
    result = {
        "new_job_collisions": fresh_job,
        "new_pod_collisions": fresh_pod,
        "new_configmap_collisions": fresh_configmap,
        "active_hosted_controllers": active,
        "kubernetes_gets": job_gets + pod_gets + configmap_gets,
    }
    if any(result[key] for key in result if key != "kubernetes_gets"):
        raise ObserverError("kubernetes_or_controller_collision")
    return result


def _identity_values(value: Any, fields: set[str]) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in fields and isinstance(item, str):
                found.add(item)
            found.update(_identity_values(item, fields))
    elif isinstance(value, list):
        for item in value:
            found.update(_identity_values(item, fields))
    return found


def _safe_json_files(root: Path, *, accepted_only: bool) -> list[Path]:
    if root.is_symlink() or not root.is_dir():
        raise ObserverError("scan_root_unsafe")
    paths: list[Path] = []
    for directory, names, files in os.walk(root, followlinks=False):
        parent = Path(directory)
        if any((parent / name).is_symlink() for name in names):
            raise ObserverError("scan_directory_symlink_forbidden")
        for name in files:
            if accepted_only and name != "ACCEPTED.json" and not (
                parent.name == "accepted" and name.endswith(".json")
            ):
                continue
            if not accepted_only and not name.endswith(".json"):
                continue
            paths.append(parent / name)
    return paths


def _receipt_collisions(
    root: Path, cell_ids: set[str], *, accepted_only: bool
) -> tuple[int, int]:
    examined = collisions = 0
    for path in _safe_json_files(root, accepted_only=accepted_only):
        if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_RECEIPT_BYTES:
            raise ObserverError("scan_file_unsafe")
        value = strict_json(path.read_bytes())
        examined += 1
        collisions += int(bool(_identity_values(value, {"cell_id"}) & cell_ids))
    return examined, collisions


def _output_paths_clear(binding: dict[str, Any]) -> int:
    jobs_root = Path(binding["jobs_root"])
    if jobs_root.is_symlink() or not jobs_root.is_dir():
        raise ObserverError("jobs_root_unsafe")
    allowed = set(binding["allowed_rank30_paths"])
    collisions = 0
    for path in jobs_root.iterdir():
        if path.is_symlink():
            raise ObserverError("jobs_entry_symlink_forbidden")
        lowered = path.name.lower()
        if any(marker in lowered for marker in RANK30_PATH_MARKERS) and str(path) not in allowed:
            collisions += 1
    target = Path(binding["fresh_sfs_root"])
    if target.exists() or target.is_symlink():
        collisions += 1
    if collisions:
        raise ObserverError("output_root_collision")
    return 0


def _lease_clear(binding: dict[str, Any]) -> int:
    root = Path(binding["lease_root"]) / binding["endpoint_key"]
    if root.is_symlink() or not root.is_dir():
        raise ObserverError("endpoint_lease_root_unsafe")
    handles: list[Any] = []
    try:
        for slot in (1, 2):
            fd = os.open(root / f"slot-{slot}.lock", os.O_RDWR | getattr(os, "O_NOFOLLOW", 0))
            handle = os.fdopen(fd, "a+b")
            handles.append(handle)
            if not stat.S_ISREG(os.fstat(fd).st_mode) or os.fstat(fd).st_size != 0:
                raise ObserverError("endpoint_lease_inode_invalid")
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ObserverError("endpoint_lease_held") from exc
        return 2
    finally:
        for handle in reversed(handles):
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()


def _fleet_get(path: str, key: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    query = "" if not params else "?" + urllib.parse.urlencode(params)
    return _request_json(
        ORCHESTRATOR + path + query,
        headers={"Authorization": "Bearer " + key, "Accept": "application/json"},
    )


def _session_collisions(binding: dict[str, Any], key: str) -> tuple[int, int, int, str]:
    rows_examined = gets = collisions = 0
    snapshot: list[dict[str, Any]] = []
    for task_key in binding["task_keys"]:
        cursor: str | None = None
        snapshot_cursor: str | None = None
        while True:
            params: dict[str, Any] = {"task_key": task_key, "limit": 500}
            if cursor is not None:
                params["cursor"] = cursor
            page = _fleet_get(
                SESSION_IDENTITY_PATH,
                key,
                params,
            )
            gets += 1
            if set(page) != {"sessions", "limit", "has_more", "next_cursor", "snapshot"}:
                raise ObserverError("session_identity_page_shape_invalid")
            rows = page.get("sessions")
            if any(
                (
                    not isinstance(rows, list),
                    not all(isinstance(row, dict) for row in rows or []),
                    page.get("limit") != 500,
                    not isinstance(page.get("has_more"), bool),
                )
            ):
                raise ObserverError("session_inventory_invalid")
            observed_snapshot = page.get("snapshot")
            if snapshot_cursor is None:
                if rows and (not isinstance(observed_snapshot, str) or not observed_snapshot):
                    raise ObserverError("session_snapshot_cursor_invalid")
                snapshot_cursor = observed_snapshot
            elif observed_snapshot != snapshot_cursor:
                raise ObserverError("session_snapshot_cursor_changed")
            rows_examined += len(rows)
            for row in rows:
                session_id = row.get("session_id")
                if (
                    set(row) != SESSION_IDENTITY_KEYS
                    or not isinstance(session_id, str)
                    or not session_id
                    or len(session_id) > 128
                    or row.get("task_key") != task_key
                ):
                    raise ObserverError("session_identity_row_invalid")
                if row.get("eval_task_version_id") == EXPECTED_TASK_VERSION_ID:
                    model_status = row.get("model_identity_status")
                    model_identity = row.get("model_identity")
                    if model_status == "resolved" and model_identity == EXPECTED_SESSION_MODEL:
                        collisions += 1
                    elif model_status != "resolved" or not isinstance(model_identity, str):
                        raise ObserverError("target_session_model_identity_ambiguous")
                snapshot.append(row)
            has_more = page["has_more"]
            next_cursor = page.get("next_cursor")
            if not has_more:
                if next_cursor is not None:
                    raise ObserverError("session_terminal_cursor_invalid")
                break
            if not rows or not isinstance(next_cursor, str) or not next_cursor:
                raise ObserverError("session_pagination_stalled")
            if next_cursor == cursor:
                raise ObserverError("session_pagination_stalled")
            cursor = next_cursor
    snapshot.sort(key=lambda row: row["session_id"])
    if len({row["session_id"] for row in snapshot}) != len(snapshot):
        raise ObserverError("session_inventory_duplicate_id")
    return rows_examined, gets, collisions, sha256(canonical(snapshot))


def validate_binding(value: dict[str, Any]) -> None:
    required = {
        "schema_version", "release_schema", "cell_ids", "task_keys", "fresh_job_name",
        "fresh_configmap_name", "fresh_sfs_root", "claim_root", "jobs_root", "lease_root",
        "endpoint_key", "allowed_rank30_paths", "held_source_package_sha256",
        "ledger_authority", "live_ledger_validation", "diagnostic_v2",
        "superseded_identities", "binding_sha256",
    }
    cell_ids = value.get("cell_ids")
    task_keys = value.get("task_keys")
    if any(
        (
            set(value) != required,
            value.get("schema_version") != SCHEMA,
            value.get("release_schema") != successor.RELEASE_SCHEMA,
            not isinstance(cell_ids, list),
            cell_ids != EXPECTED_CELL_IDS,
            not isinstance(task_keys, list),
            task_keys != EXPECTED_TASK_KEYS,
            value.get("fresh_job_name") != successor.JOB_NAME,
            value.get("fresh_configmap_name") != successor.CONFIGMAP_NAME,
            value.get("fresh_sfs_root") != successor.SFS_ROOT,
            value.get("claim_root") != str(successor.prior.CLAIM_ROOT),
            value.get("jobs_root") != str(successor.prior.JOBS_ROOT),
            value.get("lease_root") != str(successor.prior.LEASE_ROOT),
            value.get("endpoint_key") != successor.prior.LEASE_ENDPOINT_KEY,
            value.get("allowed_rank30_paths") != EXPECTED_ALLOWED_PATHS,
            value.get("ledger_authority") != successor.LEDGER_AUTHORITY,
            value.get("live_ledger_validation") != successor.LIVE_LEDGER_VALIDATION,
            value.get("diagnostic_v2") != successor.DIAGNOSTIC_V2,
            value.get("superseded_identities") != successor.SUPERSEDED_IDENTITIES,
            SHA_RE.fullmatch(str(value.get("held_source_package_sha256"))) is None,
            value.get("binding_sha256")
            != sha256(
                canonical(
                    {key: item for key, item in value.items() if key != "binding_sha256"}
                )
            ),
        )
    ):
        raise ObserverError("observer_binding_invalid")


def validate_package(package: dict[str, Any], package_root: Path) -> None:
    files = package.get("files")
    if any(
        (
            set(package) != {"schema_version", "files", "file_count", "receipt_sha256"},
            package.get("schema_version") != PACKAGE_SCHEMA,
            not isinstance(files, dict),
            not files,
            package.get("file_count") != len(files or {}),
            package.get("receipt_sha256") != digest(package),
        )
    ):
        raise ObserverError("observer_package_invalid")
    root = package_root.resolve(strict=True)
    for relative, expected in files.items():
        target = (package_root / relative.replace("/", "__SLASH__")).resolve(strict=True)
        if any(
            (
                not target.is_relative_to(root),
                not target.is_file(),
                SHA_RE.fullmatch(str(expected)) is None,
                sha256(target.read_bytes()) != expected,
            )
        ):
            raise ObserverError("observer_package_file_invalid")


def _clear_state(
    binding: dict[str, Any],
    api_key: str,
    *,
    observer_job_uid: str | None = None,
    observer_pod_uid: str | None = None,
) -> dict[str, Any]:
    account = _fleet_get("/v1/account", api_key)
    if account.get("team_name") != "fleet" or account.get("team_id") != FLEET_TEAM_ID:
        raise ObserverError("fleet_team_invalid")
    kubernetes = _kubernetes_clear(
        binding,
        observer_job_uid=observer_job_uid,
        observer_pod_uid=observer_pod_uid,
    )
    cell_ids = set(binding["cell_ids"])
    claim_examined, claim_collisions = _receipt_collisions(
        Path(binding["claim_root"]), cell_ids, accepted_only=False
    )
    accepted_examined, accepted_collisions = _receipt_collisions(
        Path(binding["jobs_root"]), cell_ids, accepted_only=True
    )
    session_rows, session_gets, session_collisions, session_sha = _session_collisions(
        binding, api_key
    )
    output_collisions = _output_paths_clear(binding)
    slots = _lease_clear(binding)
    if claim_collisions or accepted_collisions or session_collisions or output_collisions:
        raise ObserverError("rank30_cell_collision")
    return {
        "claim_receipts_examined": claim_examined,
        "accepted_receipts_examined": accepted_examined,
        "session_rows_examined": session_rows,
        "session_inventory_scans": 1,
        "archived_sessions_included": True,
        "stable_session_snapshot": True,
        "session_identity_projection": SESSION_IDENTITY_PATH,
        "session_snapshot_sha256": session_sha,
        "fleet_gets": 1 + session_gets,
        **kubernetes,
        "all_generation_claim_collisions": 0,
        "authoritative_session_collisions": 0,
        "accepted_evidence_collisions": 0,
        "output_root_collisions": 0,
        "endpoint_lease_slots_available": slots,
        "both_endpoint_lease_slots_simultaneously_free": slots == 2,
    }


def collect(
    binding: dict[str, Any],
    root: Path,
    *,
    job_uid: str,
    pod_uid: str,
    api_key: str,
) -> dict[str, Any]:
    validate_binding(binding)
    plan = successor.build_runtime_plan(
        successor.CONTROLLER, successor.load(source_runtime.INVENTORY_PATH), root
    )
    source_sha = controller_source_sha256(root)
    if source_sha != binding["held_source_package_sha256"]:
        raise ObserverError("controller_source_package_drifted")
    bound_job_uid = _uuid(job_uid, "observer_job_uid")
    bound_pod_uid = _uuid(pod_uid, "observer_pod_uid")
    state = _clear_state(
        binding,
        api_key,
        observer_job_uid=bound_job_uid,
        observer_pod_uid=bound_pod_uid,
    )
    collision = {
        "checked_immediately_before_create": True,
        "observer_job_uid": bound_job_uid,
        "observer_pod_uid": bound_pod_uid,
        "observed_cells": 4,
        "all_generation_claim_collisions": state["all_generation_claim_collisions"],
        "authoritative_session_collisions": state["authoritative_session_collisions"],
        "accepted_evidence_collisions": state["accepted_evidence_collisions"],
        "output_root_collisions": state["output_root_collisions"],
        "new_job_collisions": state["new_job_collisions"],
        "new_pod_collisions": state["new_pod_collisions"],
        "new_configmap_collisions": state["new_configmap_collisions"],
        "active_hosted_controllers": state["active_hosted_controllers"],
        "endpoint_lease_slots_available": state["endpoint_lease_slots_available"],
        "both_endpoint_lease_slots_simultaneously_free": state[
            "both_endpoint_lease_slots_simultaneously_free"
        ],
        "session_inventory_scans": state["session_inventory_scans"],
        "archived_sessions_included": state["archived_sessions_included"],
        "stable_session_snapshot": state["stable_session_snapshot"],
        "session_identity_projection": state["session_identity_projection"],
        "session_snapshot_sha256": state["session_snapshot_sha256"],
        "api_mutations": 0,
    }
    body: dict[str, Any] = {
        "schema_version": successor.RELEASE_SCHEMA,
        "status": "CLEAR_PEER_FREE",
        "checked_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "launch_authorized": True,
        "scoring_authorized": True,
        "controller": successor.release_projection(plan),
        "source_package_sha256": source_sha,
        "ledger_authority": successor.LEDGER_AUTHORITY,
        "live_ledger_validation": successor.LIVE_LEDGER_VALIDATION,
        "diagnostic_v2": successor.DIAGNOSTIC_V2,
        "superseded_identities": successor.SUPERSEDED_IDENTITIES,
        "fresh_collision_reconciliation": collision,
        "privacy": {
            "scores_read": False,
            "prompts_traces_flags_read": False,
            "credentials_included": False,
        },
    }
    release = {
        **body,
        "receipt_sha256": successor.self_hosted.digest_without(body, "receipt_sha256"),
    }
    successor.validate_release(release, plan, source_sha)
    return release


def recheck(binding: dict[str, Any], *, api_key: str) -> dict[str, Any]:
    """Repeat every mutable collision/lease check immediately before submit."""
    validate_binding(binding)
    return _clear_state(binding, api_key)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args(argv)
    if args.output != OUTPUT_PATH:
        raise ObserverError("observer_output_identity_invalid")
    package_root = args.package.parent
    validate_package(load_projected(args.package, package_root), package_root)
    binding = load_projected(args.binding, package_root)
    key = os.environ.get("FLEET_API_KEY", "")
    if not key:
        raise ObserverError("fleet_api_key_absent")
    if args.output.exists() or args.output.is_symlink():
        raise ObserverError("observer_output_already_exists_do_not_retry")
    release = collect(
        binding,
        args.repo.resolve(strict=True),
        job_uid=os.environ.get("JOB_UID", ""),
        pod_uid=os.environ.get("POD_UID", ""),
        api_key=key,
    )
    write_once(args.output, release)
    print(
        json.dumps(
            {"status": release["status"], "receipt_sha256": release["receipt_sha256"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
