"""Score-blind, create-once release observer for fresh hosted-Qwen cells.

The observer is deliberately read-only.  It checks the exact statistical-cell
lineage and both old and fresh execution/object identities, but emits only
counts and binding digests.  It never requests transcripts, verifier results,
scores, prompts or task bodies.
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
ACCOUNT_URL = "https://orchestrator.fleetai.com/v1/account"
ORCHESTRATOR = "https://orchestrator.fleetai.com"
SCHEMA = "fleet-qwen38-hosted-whole-task-release-gate-observation-v1"
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
        "planned_statistical_cell_collision",
        "predecessor_configmap_identity_drifted",
        "predecessor_job_identity_drifted",
        "predecessor_job_not_exclusively_failed",
        "predecessor_pod_count_drifted",
        "predecessor_pod_terminal_identity_drifted",
        "read_request_failed",
        "read_response_too_large",
        "redirect_forbidden",
        "release_gate_binding_invalid",
        "release_gate_observation_invalid",
        "release_gate_package_file_drifted",
        "release_gate_package_source_escape",
        "release_gate_package_source_invalid",
        "release_gate_runtime_namespace_invalid",
        "scan_directory_symlink_forbidden",
        "scan_file_unsafe",
        "scan_root_unsafe",
        "sfs_output_root_collision",
    }
)


class GateError(RuntimeError):
    """The score-blind release gate failed closed."""


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def sha256(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def digest(value: dict[str, Any]) -> str:
    return sha256(canonical({key: item for key, item in value.items() if key != "receipt_sha256"}))


def binding_digest(value: dict[str, Any]) -> str:
    return sha256(canonical({key: item for key, item in value.items() if key != "binding_sha256"}))


def _seal(value: dict[str, Any]) -> dict[str, Any]:
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
    """Load one ConfigMap-projected file only when it resolves inside its mount."""
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
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
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


def _object_checks(binding: dict[str, Any]) -> int:
    gets = 0
    for expected in binding["predecessor_objects"]:
        job = _kubernetes("job", expected["job_name"])
        pods = _kubernetes("pods", expected["job_name"])
        configmap = _kubernetes("configmap", expected["configmap_name"])
        gets += 3
        if (
            not isinstance(job, dict)
            or (job.get("metadata") or {}).get("uid") != expected["job_uid"]
        ):
            raise GateError("predecessor_job_identity_drifted")
        conditions = (job.get("status") or {}).get("conditions") or []
        failed = [
            row for row in conditions if row.get("type") == "Failed" and row.get("status") == "True"
        ]
        if len(failed) != 1 or failed[0].get("reason") != "BackoffLimitExceeded":
            raise GateError("predecessor_job_not_exclusively_failed")
        if not isinstance(pods, list) or len(pods) != 1:
            raise GateError("predecessor_pod_count_drifted")
        pod = pods[0]
        statuses = (pod.get("status") or {}).get("containerStatuses") or []
        if (
            (pod.get("metadata") or {}).get("uid") != expected["pod_uid"]
            or (pod.get("status") or {}).get("phase") != "Failed"
            or len(statuses) != 1
            or statuses[0].get("restartCount") != 0
        ):
            raise GateError("predecessor_pod_terminal_identity_drifted")
        if (
            not isinstance(configmap, dict)
            or (configmap.get("metadata") or {}).get("uid") != expected["configmap_uid"]
            or configmap.get("immutable") is not True
        ):
            raise GateError("predecessor_configmap_identity_drifted")
    for expected in binding["fresh_objects"]:
        if _kubernetes("job", expected["job_name"]) is not None:
            raise GateError("fresh_job_collision")
        if _kubernetes("pods", expected["job_name"]):
            raise GateError("fresh_pod_collision")
        if _kubernetes("configmap", expected["configmap_name"]) is not None:
            raise GateError("fresh_configmap_collision")
        gets += 3
    return gets


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
            if (
                accepted_only
                and name != "ACCEPTED.json"
                and not (parent.name == "accepted" and name.endswith(".json"))
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
            flags = os.O_RDWR | (getattr(os, "O_NOFOLLOW", 0))
            fd = os.open(path, flags)
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
    for task_key in binding["task_keys"]:
        offset = 0
        while True:
            page = _fleet_get(
                "/v1/sessions", key, {"task_key": task_key, "limit": 500, "offset": offset}
            )
            gets += 1
            rows = page.get("sessions") or []
            if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
                raise GateError("fleet_session_inventory_invalid")
            rows_examined += len(rows)
            collisions += sum(bool(_identity_values(row).intersection(expected)) for row in rows)
            if page.get("has_more") is False:
                break
            if not rows:
                raise GateError("fleet_session_pagination_stalled")
            offset += len(rows)
    return rows_examined, gets, collisions


def collect(binding: dict[str, Any], *, job_uid: str, pod_uid: str, api_key: str) -> dict[str, Any]:
    validate_binding(binding)
    account = _fleet_get("/v1/account", api_key)
    if account.get("team_name") != "fleet" or account.get("team_id") != FLEET_TEAM_ID:
        raise GateError("fleet_team_identity_invalid")
    expected = set(binding["identity_values"])
    kubernetes_gets = _object_checks(binding)
    _sfs_roots_clear(binding)
    claim_examined, claim_collisions = _file_identity_collisions(
        Path(binding["claim_root"]), expected, accepted_only=False
    )
    accepted_examined, accepted_collisions = _file_identity_collisions(
        Path(binding["jobs_root"]), expected, accepted_only=True
    )
    session_rows, session_gets, session_collisions = _session_collisions(binding, api_key, expected)
    slots = _lease_slots_clear(binding)
    if claim_collisions or accepted_collisions or session_collisions:
        raise GateError("planned_statistical_cell_collision")
    body = {
        "schema_version": SCHEMA,
        "status": "CLEAR_PENDING_TERMINAL_JOB_VALIDATION",
        "observed_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "binding_sha256": binding["binding_sha256"],
        "statistical_cell_count": 8,
        "controller_count": 2,
        "predecessor_object_set_sha256": binding["predecessor_object_set_sha256"],
        "fresh_object_set_sha256": binding["fresh_object_set_sha256"],
        "plan_set_sha256": binding["plan_set_sha256"],
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
            "authoritative_session_rows_examined": session_rows,
            "predecessor_jobs_exclusively_failed": 2,
            "predecessor_pods_terminal_restart_zero": 2,
            "predecessor_immutable_configmaps_bound": 2,
            "fresh_object_sets_absent": 2,
            "checked_sfs_roots_absent": len(binding["checked_sfs_roots"]),
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
    receipt = _seal(body)
    validate_observation(receipt, Path.cwd(), binding=binding)
    return receipt


def validate_binding(binding: dict[str, Any]) -> None:
    if any(
        (
            binding.get("schema_version")
            != "fleet-qwen38-hosted-whole-task-release-gate-binding-v1",
            binding.get("statistical_cell_count") != 8,
            len(binding.get("identity_values") or [])
            != len(set(binding.get("identity_values") or [])),
            len(binding.get("predecessor_objects") or []) != 2,
            len(binding.get("fresh_objects") or []) != 2,
            len(binding.get("task_keys") or []) != 2,
            len(binding.get("checked_sfs_roots") or []) != 6,
            binding.get("binding_sha256") != binding_digest(binding),
        )
    ):
        raise GateError("release_gate_binding_invalid")


def validate_observation(
    receipt: Any,
    root: Path,
    plans: dict[str, dict[str, Any]] | None = None,
    *,
    binding: dict[str, Any] | None = None,
) -> None:
    del root, plans
    if not isinstance(receipt, dict):
        raise GateError("release_gate_observation_invalid")
    aggregates = receipt.get("observed_aggregates") or {}
    requests = receipt.get("request_counts") or {}
    collisions = receipt.get("collisions") or {}
    if any(
        (
            receipt.get("schema_version") != SCHEMA,
            receipt.get("status") != "CLEAR_PENDING_TERMINAL_JOB_VALIDATION",
            SHA_RE.fullmatch(str(receipt.get("binding_sha256"))) is None,
            binding is not None and receipt.get("binding_sha256") != binding["binding_sha256"],
            receipt.get("statistical_cell_count") != 8,
            receipt.get("controller_count") != 2,
            any(
                SHA_RE.fullmatch(str(receipt.get(field))) is None
                for field in (
                    "predecessor_object_set_sha256",
                    "fresh_object_set_sha256",
                    "plan_set_sha256",
                )
            ),
            collisions
            != {
                "canonical_claims": 0,
                "accepted_receipts": 0,
                "authoritative_sessions": 0,
                "fresh_kubernetes_objects": 0,
                "sfs_output_roots": 0,
            },
            any(not isinstance(value, int) or value < 0 for value in aggregates.values()),
            aggregates.get("predecessor_jobs_exclusively_failed") != 2,
            aggregates.get("predecessor_pods_terminal_restart_zero") != 2,
            aggregates.get("predecessor_immutable_configmaps_bound") != 2,
            aggregates.get("fresh_object_sets_absent") != 2,
            aggregates.get("checked_sfs_roots_absent") != 6,
            aggregates.get("endpoint_lease_slots_simultaneously_free") != 2,
            requests.get("fleet_account_gets") != 1,
            not isinstance(requests.get("fleet_session_inventory_gets"), int),
            requests.get("fleet_session_inventory_gets", 0) < 2,
            requests.get("kubernetes_gets") != 12,
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
        )
    ):
        raise GateError("release_gate_observation_invalid")
    runtime = receipt.get("runtime") or {}
    if runtime.get("namespace") != NAMESPACE:
        raise GateError("release_gate_runtime_namespace_invalid")
    _uuid(runtime.get("job_uid"), "observer_job_uid")
    _uuid(runtime.get("pod_uid"), "observer_pod_uid")


def validate_package_source(path: Path, package_root: Path) -> None:
    try:
        root = package_root.resolve(strict=True)
        source = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise GateError("release_gate_package_source_invalid") from exc
    if not source.is_relative_to(root):
        raise GateError("release_gate_package_source_escape")
    value = load(source)
    files = value.get("files") or {}
    if any(
        (
            value.get("schema_version") != "fleet-qwen38-hosted-whole-task-release-gate-package-v1",
            not isinstance(files, dict),
            len(files) != 2,
            value.get("file_count") != 2,
            value.get("receipt_sha256") != digest(value),
        )
    ):
        raise GateError("release_gate_package_source_invalid")
    for name, expected in files.items():
        try:
            target = (package_root / name).resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise GateError("release_gate_package_file_drifted") from exc
        if (
            not target.is_relative_to(root)
            or not target.is_file()
            or target.stat().st_size > MAX_JSON_BYTES
            or sha256(target.read_bytes()) != expected
        ):
            raise GateError("release_gate_package_file_drifted")


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
        failure = _seal(
            {
                "schema_version": "fleet-qwen38-hosted-release-gate-failure-v1",
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
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
