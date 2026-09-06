"""Score-blind, create-once release observer for hosted-Qwen rank 17.

The observer only reads the Fleet account/session inventory, Kubernetes object
metadata, canonical execution claims, sanitized acceptance receipts, output
root existence, and endpoint lease locks.  It never requests task bodies,
transcripts, verifier payloads, prompts, flags, or scores.
"""

from __future__ import annotations

import argparse
import contextlib
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

NAMESPACE = "fleet-train-jobs"
FLEET_TEAM_ID = "a1025f0b-ad67-49fc-a023-51800ab43e84"
ORCHESTRATOR = "https://orchestrator.fleetai.com"
SCHEMA = "fleet-qwen38-hosted-rank17-g22-release-observation-v1"
BINDING_SCHEMA = "fleet-qwen38-hosted-rank17-g22-release-binding-v1"
PACKAGE_SCHEMA = "fleet-qwen38-hosted-rank17-g22-release-package-v1"
PLAN_COMMIT = "e7f1772ee7727e348523735387f2a7861308553e"
EXPECTED_TALLY = {
    "accepted": 51,
    "active": 0,
    "blocked_nonrepeatable": 8,
    "unstarted": 341,
}
EXPECTED_PLAN_MODULE_SHA256 = (
    "sha256:406eec486d25cda5e201eb907e24ce4c4671f66a48502beea0503191d9e7571b"
)
EXPECTED_HELD_FILE_SHA256 = (
    "sha256:71a99f7a90a16c6e69eff13ad9d66d6287020e4776658a8250120c7c38a2364a"
)
EXPECTED_HELD_RECEIPT_SHA256 = (
    "sha256:05e054ad8dee94d64b3b7be8ee3b66c66909129e4d5d000e8fcd7bb280c01cbf"
)
EXPECTED_PLAN_SHA256 = (
    "sha256:626f9a03737b30996c7dedbda4f5b26a3db436b6f9a43aedd7f9f5d68c5e85b9"
)
EXPECTED_TASK_KEY = (
    "cysec1-2-fentry-gen_blackbox-ddc48a638b2d9f6001c2c935__blackbox_ctf_v1"
)
EXPECTED_TASK_VERSION_ID = "0b192133-c9b8-4211-b0a1-dbf98be46fe3"
EXPECTED_SESSION_MODEL = (
    "fleet-cluster-opencode-1.18.27/qwen3.8-27b-opencode11827-autocontinue-v1"
)
EXPECTED_FRESH_OBJECT = {
    "job_name": "chris-q38-hosted-r017-whole-task-g22-v1",
    "configmap_name": "chris-q38-hosted-r017-whole-task-g22-package-v1",
}
EXPECTED_SFS_ROOTS = [
    "/mnt/sfs/jobs/chris-q38-hosted-r017-whole-task-g22-v1",
    "/mnt/sfs/jobs/chris-q38-hosted-r017-whole-task-g22-v1-diagnostic",
]
EXPECTED_CELLS = [
    {
        "attempt": 1,
        "cell_id": "sha256:8ccf52cec63b5a8ab0f969148b1c1e1d211df5203fb27b23edb8c78d9426d225",
        "execution_id": "sha256:b519a1eb9b95ade2e40d397f6b491c674645449f865dc6f737b1d09421fe5b76",
        "run_id": "chris-q38-ac-g22v1-a-r017-a1-b519a1eb",
    },
    {
        "attempt": 2,
        "cell_id": "sha256:a5aae17551251dcd2c640f6df660f16f50772084a7f46229144cb68c3885fba6",
        "execution_id": "sha256:d99047e036e25d5e266144ed558f2a10707c9c0a8cb163708c5df936a6a4f5e2",
        "run_id": "chris-q38-ac-g22v1-a-r017-a2-d99047e0",
    },
    {
        "attempt": 3,
        "cell_id": "sha256:2095b8623eb8e218bd59f69f4d7a8390f1193226b9526c9accc74ddf895b8358",
        "execution_id": "sha256:f9bad78d60ff4457a089114bb38767899476b05261a87c8af064b7d8ac508e15",
        "run_id": "chris-q38-ac-g22v1-a-r017-a3-f9bad78d",
    },
    {
        "attempt": 4,
        "cell_id": "sha256:f3e9cda21b5ff340a64807df0b48387a57adbeb0ef6465e8258b1925bfaed280",
        "execution_id": "sha256:eb626440759bf817027566f63e06767746e3100a6c2504245774fbc56da02d9e",
        "run_id": "chris-q38-ac-g22v1-a-r017-a4-eb626440",
    },
]
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_RECEIPT_BYTES = 2 * 1024 * 1024
SHA_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
SAFE_FAILURE_CODES = frozenset(
    {
        "duplicate_json_key",
        "endpoint_lease_inode_drifted",
        "endpoint_lease_root_unsafe",
        "endpoint_lease_slot_held",
        "fleet_api_key_absent",
        "fleet_session_inventory_invalid",
        "fleet_session_identity_ambiguous",
        "fleet_session_pagination_stalled",
        "fleet_team_identity_invalid",
        "fresh_configmap_collision",
        "fresh_job_collision",
        "fresh_pod_collision",
        "input_path_unsafe",
        "invalid_json",
        "json_root_invalid",
        "kubernetes_kind_invalid",
        "kubernetes_pod_list_invalid",
        "kubernetes_service_unavailable",
        "observer_job_uid_invalid",
        "observer_pod_uid_invalid",
        "output_path_unsafe",
        "output_short_write",
        "observer_failed_safely",
        "planned_accepted_path_collision",
        "planned_claim_path_collision",
        "planned_statistical_cell_collision",
        "read_request_failed",
        "read_response_too_large",
        "redirect_forbidden",
        "release_binding_invalid",
        "release_observation_invalid",
        "release_package_file_drifted",
        "release_package_source_escape",
        "release_package_source_invalid",
        "scan_directory_symlink_forbidden",
        "scan_file_unsafe",
        "scan_receipt_invalid",
        "scan_root_unsafe",
        "sfs_output_root_collision",
    }
)


class GateError(RuntimeError):
    """The score-blind rank-17 release gate failed closed."""


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def sha256(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def digest(value: dict[str, Any]) -> str:
    return sha256(canonical({key: item for key, item in value.items() if key != "receipt_sha256"}))


def binding_digest(value: dict[str, Any]) -> str:
    return sha256(canonical({key: item for key, item in value.items() if key != "binding_sha256"}))


def seal(value: dict[str, Any]) -> dict[str, Any]:
    return {**value, "receipt_sha256": digest(value)}


def _strict_json(raw: bytes) -> dict[str, Any]:
    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in rows:
            if key in result:
                raise GateError("duplicate_json_key")
            result[key] = value
        return result

    try:
        value = json.loads(raw, object_pairs_hook=pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GateError("invalid_json") from exc
    if not isinstance(value, dict):
        raise GateError("json_root_invalid")
    return value


def load(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_JSON_BYTES:
        raise GateError("input_path_unsafe")
    return _strict_json(path.read_bytes())


def load_projected(path: Path, package_root: Path) -> dict[str, Any]:
    try:
        root = package_root.resolve(strict=True)
        source = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise GateError("input_path_unsafe") from exc
    if not source.is_relative_to(root):
        raise GateError("input_path_unsafe")
    return load(source)


def write_once(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o400)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise GateError("output_path_unsafe")
        raw = canonical(value) + b"\n"
        if os.write(fd, raw) != len(raw):
            raise GateError("output_short_write")
        os.fsync(fd)
    finally:
        os.close(fd)


def _uuid(value: Any, label: str) -> str:
    try:
        parsed = uuid.UUID(str(value))
    except (ValueError, AttributeError) as exc:
        raise GateError(f"{label}_invalid") from exc
    if parsed.int == 0:
        raise GateError(f"{label}_invalid")
    return str(parsed)


class _RejectRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        raise GateError("redirect_forbidden")


def _request_json(
    url: str, *, headers: dict[str, str], cafile: str | None = None
) -> dict[str, Any]:
    request = urllib.request.Request(url, method="GET", headers=headers)
    handlers: list[Any] = [_RejectRedirects()]
    if cafile is not None:
        handlers.append(
            urllib.request.HTTPSHandler(context=ssl.create_default_context(cafile=cafile))
        )
    try:
        with urllib.request.build_opener(*handlers).open(request, timeout=60) as response:
            raw = response.read(MAX_JSON_BYTES + 1)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise
        raise GateError("read_request_failed") from exc
    if len(raw) > MAX_JSON_BYTES:
        raise GateError("read_response_too_large")
    return _strict_json(raw)


def _kubernetes(kind: str, name: str) -> dict[str, Any] | list[dict[str, Any]] | None:
    host = os.environ.get("KUBERNETES_SERVICE_HOST")
    port = os.environ.get("KUBERNETES_SERVICE_PORT_HTTPS", "443")
    if not host:
        raise GateError("kubernetes_service_unavailable")
    quoted = urllib.parse.quote(name, safe="")
    if kind == "job":
        path = f"/apis/batch/v1/namespaces/{NAMESPACE}/jobs/{quoted}"
    elif kind == "configmap":
        path = f"/api/v1/namespaces/{NAMESPACE}/configmaps/{quoted}"
    elif kind == "pods":
        selector = urllib.parse.quote(f"job-name={name}", safe="")
        path = f"/api/v1/namespaces/{NAMESPACE}/pods?labelSelector={selector}"
    else:
        raise GateError("kubernetes_kind_invalid")
    token = Path("/var/run/secrets/kubernetes.io/serviceaccount/token").read_text().strip()
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    try:
        value = _request_json(
            f"https://{host}:{port}{path}",
            headers=headers,
            cafile="/var/run/secrets/kubernetes.io/serviceaccount/ca.crt",
        )
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return [] if kind == "pods" else None
        raise
    if kind == "pods":
        items = value.get("items")
        if not isinstance(items, list) or not all(isinstance(row, dict) for row in items):
            raise GateError("kubernetes_pod_list_invalid")
        return items
    return value


def _fresh_objects_absent(binding: dict[str, Any]) -> int:
    expected = binding["fresh_object"]
    if _kubernetes("job", expected["job_name"]) is not None:
        raise GateError("fresh_job_collision")
    if _kubernetes("pods", expected["job_name"]):
        raise GateError("fresh_pod_collision")
    if _kubernetes("configmap", expected["configmap_name"]) is not None:
        raise GateError("fresh_configmap_collision")
    return 3


def _identity_values(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"cell_id", "execution_id", "run_id"} and isinstance(item, str):
                found.add(item)
            found.update(_identity_values(item))
    elif isinstance(value, list):
        for item in value:
            found.update(_identity_values(item))
    return found


def _validate_scanned_receipt(value: dict[str, Any]) -> None:
    """Require self-authenticating sanitized metadata before using its identities."""
    schema = value.get("schema_version")
    receipt_sha256 = value.get("receipt_sha256")
    if (
        not isinstance(schema, str)
        or not schema
        or SHA_RE.fullmatch(str(receipt_sha256)) is None
        or receipt_sha256 != digest(value)
    ):
        raise GateError("scan_receipt_invalid")


def _planned_paths_absent(paths: list[str], *, failure_code: str) -> int:
    for raw in paths:
        path = Path(raw)
        if path.exists() or path.is_symlink():
            raise GateError(failure_code)
    return len(paths)


def _file_identity_collisions(
    root: Path, expected: set[str], *, accepted_only: bool
) -> tuple[int, int]:
    if root.is_symlink() or not root.is_dir():
        raise GateError("scan_root_unsafe")
    examined = 0
    collisions = 0
    paths: list[Path] = []
    for directory, names, files in os.walk(root, followlinks=False):
        parent = Path(directory)
        if any((parent / name).is_symlink() for name in names):
            raise GateError("scan_directory_symlink_forbidden")
        for name in files:
            if accepted_only and name != "ACCEPTED.json" and not (
                parent.name == "accepted" and name.endswith(".json")
            ):
                continue
            if not accepted_only and not name.endswith(".json"):
                continue
            paths.append(parent / name)
    for path in paths:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_RECEIPT_BYTES:
            raise GateError("scan_file_unsafe")
        examined += 1
        value = _strict_json(path.read_bytes())
        _validate_scanned_receipt(value)
        collisions += int(bool(_identity_values(value).intersection(expected)))
    return examined, collisions


def _sfs_roots_clear(binding: dict[str, Any]) -> None:
    for raw in binding["checked_sfs_roots"]:
        path = Path(raw)
        if path.exists() or path.is_symlink():
            raise GateError("sfs_output_root_collision")


def _lease_slots_clear(binding: dict[str, Any]) -> int:
    root = Path(binding["lease_root"]) / binding["endpoint_key"]
    if root.is_symlink() or not root.is_dir():
        raise GateError("endpoint_lease_root_unsafe")
    handles: list[Any] = []
    try:
        for slot in (1, 2):
            path = root / f"slot-{slot}.lock"
            fd = os.open(path, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0))
            handle = os.fdopen(fd, "a+b")
            handles.append(handle)
            if not stat.S_ISREG(os.fstat(fd).st_mode) or os.fstat(fd).st_size != 0:
                raise GateError("endpoint_lease_inode_drifted")
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise GateError("endpoint_lease_slot_held") from exc
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
        headers={"Authorization": f"Bearer {key}", "Accept": "application/json"},
    )


def _session_collisions(
    binding: dict[str, Any], key: str, expected: set[str]
) -> tuple[int, int, int]:
    rows_examined = 0
    gets = 0
    collisions = 0
    offset = 0
    while True:
        page = _fleet_get(
            "/v1/sessions",
            key,
            {"task_key": binding["task_key"], "limit": 500, "offset": offset},
        )
        gets += 1
        if "sessions" not in page or "has_more" not in page:
            raise GateError("fleet_session_inventory_invalid")
        rows = page["sessions"]
        has_more = page["has_more"]
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise GateError("fleet_session_inventory_invalid")
        if not isinstance(has_more, bool):
            raise GateError("fleet_session_inventory_invalid")
        rows_examined += len(rows)
        for row in rows:
            row_identities = _identity_values(row)
            collisions += int(bool(row_identities.intersection(expected)))
            model = row.get("model")
            if model is None:
                raise GateError("fleet_session_identity_ambiguous")
            if not isinstance(model, str):
                raise GateError("fleet_session_identity_ambiguous")
            if model != binding["session_model"]:
                continue
            metadata = row.get("metadata")
            projected_version = row.get("eval_task_version_id") or row.get("task_version_id")
            if not isinstance(metadata, dict) or projected_version != binding["task_version_id"]:
                raise GateError("fleet_session_identity_ambiguous")
            if not _identity_values(metadata):
                raise GateError("fleet_session_identity_ambiguous")
        if has_more is False:
            break
        if not rows:
            raise GateError("fleet_session_pagination_stalled")
        offset += len(rows)
    return rows_examined, gets, collisions


def validate_binding(binding: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "plan_commit",
        "plan_module_sha256",
        "held_file_sha256",
        "held_receipt_sha256",
        "plan_sha256",
        "authoritative_tally",
        "statistical_cell_count",
        "cells",
        "identity_values",
        "task_key",
        "task_version_id",
        "session_model",
        "fresh_object",
        "checked_sfs_roots",
        "claim_root",
        "planned_claim_paths",
        "jobs_root",
        "planned_accepted_paths",
        "lease_root",
        "endpoint_key",
        "binding_sha256",
    }
    cells = binding.get("cells") or []
    identities = binding.get("identity_values") or []
    if any(
        (
            set(binding) != required,
            binding.get("schema_version") != BINDING_SCHEMA,
            binding.get("plan_commit") != PLAN_COMMIT,
            binding.get("plan_module_sha256") != EXPECTED_PLAN_MODULE_SHA256,
            binding.get("held_file_sha256") != EXPECTED_HELD_FILE_SHA256,
            binding.get("held_receipt_sha256") != EXPECTED_HELD_RECEIPT_SHA256,
            binding.get("plan_sha256") != EXPECTED_PLAN_SHA256,
            binding.get("authoritative_tally") != EXPECTED_TALLY,
            binding.get("statistical_cell_count") != 4,
            not isinstance(cells, list),
            cells != EXPECTED_CELLS,
            not isinstance(identities, list),
            len(identities) != 12,
            len(identities) != len(set(identities)),
            identities != sorted(_identity_values(EXPECTED_CELLS)),
            binding.get("task_key") != EXPECTED_TASK_KEY,
            binding.get("task_version_id") != EXPECTED_TASK_VERSION_ID,
            binding.get("session_model") != EXPECTED_SESSION_MODEL,
            binding.get("fresh_object") != EXPECTED_FRESH_OBJECT,
            binding.get("checked_sfs_roots") != EXPECTED_SFS_ROOTS,
            binding.get("claim_root")
            != "/mnt/sfs/cell-execution-claims/opencode11827-autocontinue-v1",
            binding.get("planned_claim_paths")
            != [
                "/mnt/sfs/cell-execution-claims/opencode11827-autocontinue-v1/"
                + row["execution_id"].removeprefix("sha256:")
                + ".json"
                for row in EXPECTED_CELLS
            ],
            binding.get("jobs_root") != "/mnt/sfs/jobs",
            binding.get("planned_accepted_paths")
            != [f"/mnt/sfs/jobs/{row['run_id']}/ACCEPTED.json" for row in EXPECTED_CELLS],
            binding.get("lease_root")
            != "/mnt/sfs/endpoint-leases/opencode11827-autocontinue-primary-v1",
            binding.get("endpoint_key") != "qwen-hosted-autocontinue-v1",
            binding.get("binding_sha256") != binding_digest(binding),
        )
    ):
        raise GateError("release_binding_invalid")


def collect(binding: dict[str, Any], *, job_uid: str, pod_uid: str, api_key: str) -> dict[str, Any]:
    validate_binding(binding)
    account = _fleet_get("/v1/account", api_key)
    if account.get("team_name") != "fleet" or account.get("team_id") != FLEET_TEAM_ID:
        raise GateError("fleet_team_identity_invalid")
    expected = set(binding["identity_values"])
    kubernetes_gets = _fresh_objects_absent(binding)
    _sfs_roots_clear(binding)
    planned_claims_absent = _planned_paths_absent(
        binding["planned_claim_paths"], failure_code="planned_claim_path_collision"
    )
    planned_accepted_absent = _planned_paths_absent(
        binding["planned_accepted_paths"], failure_code="planned_accepted_path_collision"
    )
    claim_examined, claim_collisions = _file_identity_collisions(
        Path(binding["claim_root"]), expected, accepted_only=False
    )
    accepted_examined, accepted_collisions = _file_identity_collisions(
        Path(binding["jobs_root"]), expected, accepted_only=True
    )
    session_rows, session_gets, session_collisions = _session_collisions(
        binding, api_key, expected
    )
    slots = _lease_slots_clear(binding)
    if claim_collisions or accepted_collisions or session_collisions:
        raise GateError("planned_statistical_cell_collision")
    receipt = seal(
        {
            "schema_version": SCHEMA,
            "status": "CLEAR_PENDING_TERMINAL_JOB_VALIDATION",
            "observed_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "binding_sha256": binding["binding_sha256"],
            "plan_commit": PLAN_COMMIT,
            "plan_sha256": binding["plan_sha256"],
            "held_receipt_sha256": binding["held_receipt_sha256"],
            "authoritative_tally": EXPECTED_TALLY,
            "statistical_cell_count": 4,
            "all_planned_cells_observed_unstarted": True,
            "collisions": {
                "canonical_claims": 0,
                "accepted_receipts": 0,
                "authoritative_sessions": 0,
                "fresh_kubernetes_objects": 0,
                "sfs_output_roots": 0,
            },
            "observed_aggregates": {
                "canonical_claim_receipts_examined": claim_examined,
                "accepted_receipts_examined": accepted_examined,
                "planned_claim_paths_absent": planned_claims_absent,
                "planned_accepted_paths_absent": planned_accepted_absent,
                "authoritative_session_rows_examined": session_rows,
                "fresh_object_sets_absent": 1,
                "checked_sfs_roots_absent": 2,
                "endpoint_lease_slots_simultaneously_free": slots,
            },
            "request_counts": {
                "fleet_account_gets": 1,
                "fleet_session_inventory_gets": session_gets,
                "kubernetes_gets": kubernetes_gets,
                "transcript_prompt_task_verifier_or_score_gets": 0,
            },
            "runtime": {
                "namespace": NAMESPACE,
                "job_uid": _uuid(job_uid, "observer_job_uid"),
                "pod_uid": _uuid(pod_uid, "observer_pod_uid"),
            },
            "methods": ["GET"],
            "model_calls": 0,
            "task_calls": 0,
            "session_mutations": 0,
            "verifier_calls": 0,
            "scoring_calls": 0,
            "api_mutations": 0,
            "scores_included": False,
            "prompts_traces_flags_included": False,
            "credentials_included": False,
        }
    )
    validate_observation(receipt, binding=binding)
    return receipt


def validate_observation(receipt: Any, *, binding: dict[str, Any]) -> None:
    if not isinstance(receipt, dict):
        raise GateError("release_observation_invalid")
    aggregates = receipt.get("observed_aggregates") or {}
    requests = receipt.get("request_counts") or {}
    runtime = receipt.get("runtime") or {}
    required = {
        "schema_version",
        "status",
        "observed_at_utc",
        "binding_sha256",
        "plan_commit",
        "plan_sha256",
        "held_receipt_sha256",
        "authoritative_tally",
        "statistical_cell_count",
        "all_planned_cells_observed_unstarted",
        "collisions",
        "observed_aggregates",
        "request_counts",
        "runtime",
        "methods",
        "model_calls",
        "task_calls",
        "session_mutations",
        "verifier_calls",
        "scoring_calls",
        "api_mutations",
        "scores_included",
        "prompts_traces_flags_included",
        "credentials_included",
        "receipt_sha256",
    }
    if any(
        (
            set(receipt) != required,
            receipt.get("schema_version") != SCHEMA,
            receipt.get("status") != "CLEAR_PENDING_TERMINAL_JOB_VALIDATION",
            receipt.get("binding_sha256") != binding["binding_sha256"],
            receipt.get("plan_commit") != PLAN_COMMIT,
            receipt.get("plan_sha256") != binding["plan_sha256"],
            receipt.get("held_receipt_sha256") != binding["held_receipt_sha256"],
            receipt.get("authoritative_tally") != EXPECTED_TALLY,
            receipt.get("statistical_cell_count") != 4,
            receipt.get("all_planned_cells_observed_unstarted") is not True,
            receipt.get("collisions")
            != {
                "canonical_claims": 0,
                "accepted_receipts": 0,
                "authoritative_sessions": 0,
                "fresh_kubernetes_objects": 0,
                "sfs_output_roots": 0,
            },
            any(not isinstance(value, int) or value < 0 for value in aggregates.values()),
            set(aggregates)
            != {
                "canonical_claim_receipts_examined",
                "accepted_receipts_examined",
                "planned_claim_paths_absent",
                "planned_accepted_paths_absent",
                "authoritative_session_rows_examined",
                "fresh_object_sets_absent",
                "checked_sfs_roots_absent",
                "endpoint_lease_slots_simultaneously_free",
            },
            aggregates.get("fresh_object_sets_absent") != 1,
            aggregates.get("checked_sfs_roots_absent") != 2,
            aggregates.get("planned_claim_paths_absent") != 4,
            aggregates.get("planned_accepted_paths_absent") != 4,
            aggregates.get("endpoint_lease_slots_simultaneously_free") != 2,
            requests.get("fleet_account_gets") != 1,
            set(requests)
            != {
                "fleet_account_gets",
                "fleet_session_inventory_gets",
                "kubernetes_gets",
                "transcript_prompt_task_verifier_or_score_gets",
            },
            not isinstance(requests.get("fleet_session_inventory_gets"), int),
            requests.get("fleet_session_inventory_gets", 0) < 1,
            requests.get("kubernetes_gets") != 3,
            requests.get("transcript_prompt_task_verifier_or_score_gets") != 0,
            receipt.get("methods") != ["GET"],
            any(
                receipt.get(field) != 0
                for field in (
                    "model_calls",
                    "task_calls",
                    "session_mutations",
                    "verifier_calls",
                    "scoring_calls",
                    "api_mutations",
                )
            ),
            receipt.get("scores_included") is not False,
            receipt.get("prompts_traces_flags_included") is not False,
            receipt.get("credentials_included") is not False,
            receipt.get("receipt_sha256") != digest(receipt),
            UTC_RE.fullmatch(str(receipt.get("observed_at_utc"))) is None,
            set(runtime) != {"namespace", "job_uid", "pod_uid"},
            runtime.get("namespace") != NAMESPACE,
        )
    ):
        raise GateError("release_observation_invalid")
    _uuid(runtime.get("job_uid"), "observer_job_uid")
    _uuid(runtime.get("pod_uid"), "observer_pod_uid")


def validate_package_source(path: Path, package_root: Path) -> None:
    try:
        root = package_root.resolve(strict=True)
        source = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise GateError("release_package_source_invalid") from exc
    if not source.is_relative_to(root):
        raise GateError("release_package_source_escape")
    value = load(source)
    files = value.get("files") or {}
    expected_files = {
        "binding.json",
        "qwen_hosted_rank17_g22_release_observer_v1.py",
    }
    if any(
        (
            set(value) != {"schema_version", "files", "file_count", "receipt_sha256"},
            value.get("schema_version") != PACKAGE_SCHEMA,
            not isinstance(files, dict),
            set(files) != expected_files,
            value.get("file_count") != 2,
            value.get("receipt_sha256") != digest(value),
        )
    ):
        raise GateError("release_package_source_invalid")
    for name, expected in files.items():
        try:
            target = (package_root / name).resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise GateError("release_package_file_drifted") from exc
        if (
            not target.is_relative_to(root)
            or not target.is_file()
            or target.stat().st_size > MAX_JSON_BYTES
            or sha256(target.read_bytes()) != expected
        ):
            raise GateError("release_package_file_drifted")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--package-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    stage = "package-source"
    try:
        validate_package_source(args.package_source, args.package_source.parent)
        stage = "binding"
        binding = load_projected(args.binding, args.package_source.parent)
        key = os.environ.get("FLEET_API_KEY")
        if not key:
            raise GateError("fleet_api_key_absent")
        stage = "collect"
        receipt = collect(
            binding,
            job_uid=os.environ["JOB_UID"],
            pod_uid=os.environ["POD_UID"],
            api_key=key,
        )
        stage = "write"
        write_once(args.output, receipt)
    except Exception as exc:
        code = str(exc)
        failure = seal(
            {
                "schema_version": "fleet-qwen38-hosted-rank17-g22-release-failure-v1",
                "status": "FAILED",
                "last_stage": stage,
                "failure_code": code if code in SAFE_FAILURE_CODES else "redacted",
                "error_type": type(exc).__name__,
                "error_sha256": sha256(str(exc).encode()),
                "job_uid": os.environ.get("JOB_UID"),
                "pod_uid": os.environ.get("POD_UID"),
                "model_calls": 0,
                "task_calls": 0,
                "session_mutations": 0,
                "verifier_calls": 0,
                "scoring_calls": 0,
                "api_mutations": 0,
                "scores_included": False,
                "prompts_traces_flags_included": False,
                "credentials_included": False,
            }
        )
        with contextlib.suppress(Exception):
            write_once(args.output.with_name("FAILED.json"), failure)
        raise GateError("observer_failed_safely") from None
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
