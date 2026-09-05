"""Exact score-blind source/accept duplicate observer for final pass@4 v5."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import ssl
import stat
import subprocess
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

NAMESPACE = "fleet-train-jobs"
SECRET_NAME = "chris-cyber-opencode-evals-v2"
SECRET_UID = "e0febd8e-94a2-46b0-a0bf-dd6b3154187b"
FLEET_TEAM_ID = "a1025f0b-ad67-49fc-a023-51800ab43e84"
ACCOUNT_URL = "https://orchestrator.fleetai.com/v1/account"
SOURCE_SCHEMA = "fleet-exact-pass4-final-duplicate-source-v5"
RELEASE_SCHEMA = "fleet-exact-pass4-final-duplicate-observer-release-v5"
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_CLAIM_BYTES = 4 * 1024 * 1024
MAX_FRESH_SECONDS = 900
UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
SHA_RE = re.compile(r"^sha256:[0-9a-f]{64}$")

MODULE_PATH = "evals/fleet/exact_pass4_final_duplicate_observer_v5.py"
PACKAGE_PATH = "evals/fleet/exact_pass4_final_duplicate_package_v5.py"
RUN_PATH = "evals/fleet/scripts/run_exact_pass4_final_duplicate_v5.sh"
SUBMIT_PATH = "evals/fleet/scripts/submit_exact_pass4_final_duplicate_v5.sh"
RENDER_PATH = "evals/fleet/exact_pass4_final_duplicate_renderer_v5.py"
MANIFEST_PATH = "evals/fleet/cluster/exact-pass4-final-duplicate-held-v5.yaml"
HELD_PATH = (
    "docs/evidence/qwen38-study/"
    "2026-09-05-opencode-exact-pass4-final-duplicate-observer-held-v5.json"
)
DOC_PATH = "docs/EXACT_PASS4_FINAL_DUPLICATE_OBSERVER_V5.md"
RELEASE_PATH = (
    "docs/evidence/qwen38-study/"
    "2026-09-05-opencode-exact-pass4-final-duplicate-observer-release-v5.json"
)
PACKAGE_PATHS = (
    MODULE_PATH,
    PACKAGE_PATH,
    RENDER_PATH,
    RUN_PATH,
    SUBMIT_PATH,
    MANIFEST_PATH,
    HELD_PATH,
    DOC_PATH,
)


class DuplicateError(RuntimeError):
    """The final-v5 duplicate gate failed closed."""


def _bulk() -> Any:
    from evals.fleet import exact_pass4_final_bulk_v5

    return exact_pass4_final_bulk_v5


def _evidence() -> Any:
    """Return the reviewed score-blind helpers through the current v4 gate."""
    from evals.fleet import exact_pass4_prebulk_reconciliation_v4

    return exact_pass4_prebulk_reconciliation_v4.prior


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def sha256(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def digest(value: dict[str, Any], field: str = "receipt_sha256") -> str:
    return sha256(canonical({key: item for key, item in value.items() if key != field}))


def _seal(value: dict[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["receipt_sha256"] = digest(result)
    return result


def _uuid(value: Any, label: str) -> str:
    try:
        parsed = uuid.UUID(str(value))
    except (ValueError, AttributeError) as exc:
        raise DuplicateError(f"{label}_invalid") from exc
    if parsed.int == 0:
        raise DuplicateError(f"{label}_invalid")
    return str(parsed)


def _strict_json(raw: bytes) -> dict[str, Any]:
    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in rows:
            if key in result:
                raise DuplicateError("duplicate_json_key")
            result[key] = value
        return result

    try:
        value = json.loads(raw, object_pairs_hook=pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DuplicateError("invalid_json") from exc
    if not isinstance(value, dict):
        raise DuplicateError("json_root_invalid")
    return value


def load(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_JSON_BYTES:
        raise DuplicateError("receipt_path_unsafe")
    return _strict_json(path.read_bytes())


def write_once(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o400)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise DuplicateError("receipt_target_unsafe")
        raw = canonical(value) + b"\n"
        if os.write(fd, raw) != len(raw):
            raise DuplicateError("receipt_short_write")
        os.fsync(fd)
    finally:
        os.close(fd)


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def source_job(group: str) -> str:
    return f"chris-cyber-final-v5-dup-{group}-source"


def accept_job(group: str) -> str:
    return f"chris-cyber-final-v5-dup-{group}-accept"


def output_root(group: str) -> Path:
    return Path(f"/mnt/sfs/jobs/chris-cyber-final-v5-dup-{group}")


def package_configmap(group: str, component: str) -> str:
    if component not in {"core-a", "core-b", "core-c", "core-d", "runtime"}:
        raise DuplicateError("package_component_invalid")
    return f"chris-final-v5-dup-{group}-{component}"


def _groups() -> tuple[str, ...]:
    return tuple(_bulk().GROUPS)


def group_binding(root: Path, group: str) -> dict[str, Any]:
    bulk = _bulk()
    plans = bulk.validate_all(root)
    if group not in bulk.GROUPS:
        raise DuplicateError("group_invalid")
    selected = [plans[key] for key in bulk.GROUPS[group]]
    cells = [cell for plan in selected for cell in plan["attempts"]]
    identities = bulk._group_identity(group, root)  # noqa: SLF001 - final-v5 authority
    return {
        "group": group,
        "planned_execution_count": len(cells),
        "exact_identity_sha256": bulk.sha256(bulk.canonical(identities)),
        "checked_job_names": sorted(plan["job_name"] for plan in selected),
        "checked_configmap_names": sorted(plan["configmap_name"] for plan in selected),
        "checked_output_roots": sorted(plan["sfs_root"] for plan in selected),
        "cell_ids": sorted(cell["cell_id"] for cell in cells),
        "execution_ids": sorted(cell["execution_id"] for cell in cells),
        "run_ids": sorted(cell["run_id"] for cell in cells),
        "task_keys": sorted({cell["task_key"] for cell in cells}),
        "task_version_ids": sorted({cell["task_version_id"] for cell in cells}),
        "environment_version_ids": sorted({cell["environment_version_id"] for cell in cells}),
        "controller_plan_sha256": {plan["controller"]: plan["plan_sha256"] for plan in selected},
        "model_serving_bindings": sorted(
            (
                {
                    "model": plan["model"],
                    "serving_kind": plan["serving_kind"],
                    "serving_block": plan["serving_block"],
                }
                for plan in selected
            ),
            key=lambda row: (row["model"], row["serving_block"]),
        ),
        "execution_contract": {
            "harness": "opencode",
            "harness_version": "1.18.27",
            "context_window_size": 262144,
            "compaction_headroom_tokens": 20000,
            "required_task_tools": ["bash", "submit_report"],
            "same_task_max_inflight": 1,
            "attempts_per_task_sequential": True,
        },
    }


def validate_held(root: Path) -> dict[str, Any]:
    value = load(root / HELD_PATH)
    if (
        value.get("schema_version") != "fleet-exact-pass4-final-duplicate-observer-held-v5"
        or value.get("status") != "HELD"
        or value.get("groups") != list(_groups())
        or value.get("source_job_count") != len(_groups())
        or value.get("accept_job_count") != len(_groups())
        or value.get("create_once") is not True
        or value.get("methods") != ["GET"]
        or value.get("mutation_calls") != 0
        or value.get("credential_authority")
        != {
            "secret_name": SECRET_NAME,
            "secret_uid": SECRET_UID,
            "fleet_team_id": FLEET_TEAM_ID,
        }
        or value.get("resource_policy")
        != {
            "automatic_retry": False,
            "cpu_only": True,
            "preemption_policy": "Never",
            "priority_class": "fleet-serve-low",
        }
        or value.get("launch_authorized") is not False
        or value.get("objects_created") is not False
        or value.get("prompts_traces_flags_or_scores_included") is not False
        or value.get("receipt_sha256") != digest(value)
    ):
        raise DuplicateError("held_receipt_invalid")
    return value


class _RejectRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        raise DuplicateError("redirect_forbidden")


def _in_cluster_metadata(kind: str, name: str) -> dict[str, Any] | list[dict[str, Any]] | None:
    host = os.environ.get("KUBERNETES_SERVICE_HOST")
    port = os.environ.get("KUBERNETES_SERVICE_PORT_HTTPS", "443")
    if not host:
        raise DuplicateError("kubernetes_service_unavailable")
    quoted = urllib.parse.quote(name, safe="")
    if kind == "job":
        path = f"/apis/batch/v1/namespaces/{NAMESPACE}/jobs/{quoted}"
    elif kind == "configmap":
        path = f"/api/v1/namespaces/{NAMESPACE}/configmaps/{quoted}"
    elif kind == "secret":
        path = f"/api/v1/namespaces/{NAMESPACE}/secrets/{quoted}"
    elif kind == "pods":
        selector = urllib.parse.quote(f"job-name={name}", safe="")
        path = f"/api/v1/namespaces/{NAMESPACE}/pods?labelSelector={selector}"
    else:
        raise DuplicateError("metadata_query_kind_invalid")
    token = Path("/var/run/secrets/kubernetes.io/serviceaccount/token").read_text().strip()
    request = urllib.request.Request(
        f"https://{host}:{port}{path}",
        method="GET",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json;as=PartialObjectMetadata;g=meta.k8s.io;v=v1",
        },
    )
    context = ssl.create_default_context(
        cafile="/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"
    )
    opener = urllib.request.build_opener(
        _RejectRedirects(), urllib.request.HTTPSHandler(context=context)
    )
    try:
        with opener.open(request, timeout=30) as response:
            raw = response.read(MAX_JSON_BYTES + 1)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return [] if kind == "pods" else None
        raise DuplicateError("kubernetes_metadata_query_failed") from exc
    if len(raw) > MAX_JSON_BYTES:
        raise DuplicateError("kubernetes_metadata_response_too_large")
    value = _strict_json(raw)
    if kind == "pods":
        items = value.get("items")
        if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
            raise DuplicateError("kubernetes_metadata_list_invalid")
        return items
    return value


def _release_kubernetes(kind: str, name: str | None = None) -> dict[str, Any]:
    """Read exact terminal Job/Pod objects without transporting credentials.

    The release builder normally runs beside the create relay on the operator
    workstation, while source and accept collection run in-cluster.  Prefer the
    projected service-account reader when present and otherwise use the current
    read-only kubectl identity.  Only the two object kinds consumed by the
    UID-bound terminal validator are supported.
    """
    if os.environ.get("KUBERNETES_SERVICE_HOST"):
        return _evidence()._default_kubernetes(kind, name)  # noqa: SLF001
    if not name or kind not in {"job", "pods"}:
        raise DuplicateError("release_kubernetes_query_invalid")
    command = ["kubectl", "-n", NAMESPACE, "get"]
    if kind == "job":
        command.extend(("job", name, "-o", "json"))
    else:
        command.extend(("pods", "-l", f"job-name={name}", "-o", "json"))
    result = subprocess.run(command, capture_output=True, check=False)
    if result.returncode != 0 or len(result.stdout) > MAX_JSON_BYTES:
        raise DuplicateError("release_kubernetes_query_failed")
    return _strict_json(result.stdout)


def _fleet_account(api_key: str) -> dict[str, str]:
    request = urllib.request.Request(
        ACCOUNT_URL,
        method="GET",
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
    )
    with urllib.request.build_opener(_RejectRedirects()).open(request, timeout=30) as response:
        raw = response.read(1024 * 1024 + 1)
    if len(raw) > 1024 * 1024:
        raise DuplicateError("fleet_account_response_too_large")
    value = _strict_json(raw)
    return {"team_name": str(value.get("team_name")), "team_id": str(value.get("team_id"))}


def _identity_values(value: Any, keys: frozenset[str]) -> set[str]:
    observed: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in keys and isinstance(item, str):
                observed.add(item)
            observed.update(_identity_values(item, keys))
    elif isinstance(value, list):
        for item in value:
            observed.update(_identity_values(item, keys))
    return observed


def _reject_protected_keys(value: Any) -> None:
    forbidden = {
        "prompt",
        "prompts",
        "trace",
        "traces",
        "flag",
        "flags",
        "score",
        "scores",
        "solution",
        "solutions",
        "messages",
        "transcript",
        "transcripts",
    }
    if isinstance(value, dict):
        if forbidden.intersection(value):
            raise DuplicateError("protected_claim_key_forbidden")
        for item in value.values():
            _reject_protected_keys(item)
    elif isinstance(value, list):
        for item in value:
            _reject_protected_keys(item)


def _claim_files_clear(claim_root: Path, binding: dict[str, Any]) -> int:
    if claim_root.is_symlink():
        raise DuplicateError("claim_root_symlink_forbidden")
    if not claim_root.exists():
        return 0
    if not claim_root.is_dir():
        raise DuplicateError("claim_root_invalid")
    expected = {
        *binding["cell_ids"],
        *binding["execution_ids"],
        *binding["run_ids"],
        *(value.removeprefix("sha256:") for value in binding["execution_ids"]),
    }
    count = 0
    paths: list[Path] = []
    for directory, names, files in os.walk(claim_root, followlinks=False):
        parent = Path(directory)
        for name in names:
            if (parent / name).is_symlink():
                raise DuplicateError("claim_directory_symlink_forbidden")
        paths.extend(parent / name for name in files if name.endswith(".json"))
    for path in sorted(paths):
        if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_CLAIM_BYTES:
            raise DuplicateError("claim_file_unsafe")
        count += 1
        value = _strict_json(path.read_bytes())
        _reject_protected_keys(value)
        identities = _identity_values(value, frozenset({"cell_id", "execution_id", "run_id"}))
        if path.stem in expected or identities.intersection(expected):
            raise DuplicateError("global_generation_claim_collision")
    return count


def _metadata_absence(
    binding: dict[str, Any],
    metadata: Callable[[str, str], dict[str, Any] | list[dict[str, Any]] | None],
) -> None:
    for name in binding["checked_job_names"]:
        if metadata("job", name) is not None or metadata("pods", name):
            raise DuplicateError("kubernetes_job_or_pod_collision")
    for name in binding["checked_configmap_names"]:
        if metadata("configmap", name) is not None:
            raise DuplicateError("kubernetes_configmap_collision")


def _sfs_absence(binding: dict[str, Any]) -> None:
    for value in binding["checked_output_roots"]:
        path = Path(value)
        if path.exists() or path.is_symlink():
            raise DuplicateError("sfs_output_collision")


def _session_absence(
    binding: dict[str, Any],
    sessions: Callable[[str, str], list[dict[str, Any]]],
    api_key: str,
) -> int:
    expected = {*binding["cell_ids"], *binding["execution_ids"], *binding["run_ids"]}
    rows_examined = 0
    for task_key in binding["task_keys"]:
        rows = sessions(task_key, api_key)
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise DuplicateError("fleet_session_metadata_invalid")
        rows_examined += len(rows)
        for row in rows:
            if _identity_values(row, frozenset({"cell_id", "execution_id", "run_id"})).intersection(
                expected
            ):
                raise DuplicateError("fleet_session_collision")
    return rows_examined


def build_release(root: Path, package_commit: str) -> dict[str, Any]:
    if COMMIT_RE.fullmatch(package_commit) is None:
        raise DuplicateError("package_commit_invalid")
    from evals.fleet import exact_pass4_final_duplicate_package_v5 as package

    validate_held(root)
    built = package.build_package(root)
    for obj in built["manifest"]["objects"]:
        for entry in obj["entries"]:
            source = entry["source_path"]
            committed = subprocess.run(
                ["git", "-C", str(root), "show", f"{package_commit}:{source}"],
                check=True,
                capture_output=True,
            ).stdout
            if committed != (root / source).read_bytes():
                raise DuplicateError(f"package_commit_byte_drift:{source}")
    body = {
        "schema_version": RELEASE_SCHEMA,
        "status": "RELEASED",
        "append_only": True,
        "package_commit": package_commit,
        "package_aggregate_sha256": built["aggregate_sha256"],
        "groups": list(_groups()),
        "source_jobs": {group: source_job(group) for group in _groups()},
        "accept_jobs": {group: accept_job(group) for group in _groups()},
        "output_roots": {group: str(output_root(group)) for group in _groups()},
        "credential_authority": {
            "secret_name": SECRET_NAME,
            "secret_uid": SECRET_UID,
            "fleet_team_id": FLEET_TEAM_ID,
        },
        "resource_policy": {
            "cpu_only": True,
            "priority_class": "fleet-serve-low",
            "preemption_policy": "Never",
            "automatic_retry": False,
        },
        "create_once": True,
        "methods": ["GET"],
        "mutation_calls": 0,
        "launch_authorized": True,
        "prompts_traces_flags_or_scores_included": False,
        "credentials_included": False,
    }
    return _seal(body)


def validate_release(value: dict[str, Any], root: Path) -> None:
    package_commit = value.get("package_commit")
    if not isinstance(package_commit, str) or value != build_release(root, package_commit):
        raise DuplicateError("observer_release_invalid")


def collect(
    root: Path,
    group: str,
    *,
    job_uid: str,
    pod_uid: str,
    package_commit: str,
    sessions: Callable[[str, str], list[dict[str, Any]]] | None = None,
    metadata: Callable[
        [str, str], dict[str, Any] | list[dict[str, Any]] | None
    ] = _in_cluster_metadata,
    account: Callable[[str], dict[str, str]] = _fleet_account,
    api_key: str | None = None,
    observed_at: str | None = None,
    claim_root: Path | None = None,
    runtime_role: str = "source",
) -> dict[str, Any]:
    if COMMIT_RE.fullmatch(package_commit) is None:
        raise DuplicateError("package_commit_invalid")
    binding = group_binding(root, group)
    job_uid = _uuid(job_uid, "observer_job_uid")
    pod_uid = _uuid(pod_uid, "observer_pod_uid")
    key = api_key or os.environ.get("FLEET_API_KEY")
    if not key:
        raise DuplicateError("fleet_api_key_absent")
    secret = metadata("secret", SECRET_NAME)
    secret_metadata = secret.get("metadata") if isinstance(secret, dict) else None
    if (
        not isinstance(secret_metadata, dict)
        or secret_metadata.get("name") != SECRET_NAME
        or secret_metadata.get("namespace") != NAMESPACE
        or _uuid(secret_metadata.get("uid"), "secret_uid") != SECRET_UID
    ):
        raise DuplicateError("exact_secret_identity_invalid")
    fleet_account = account(key)
    if fleet_account != {"team_name": "fleet", "team_id": FLEET_TEAM_ID}:
        raise DuplicateError("fleet_team_identity_invalid")
    _metadata_absence(binding, metadata)
    _sfs_absence(binding)
    root_claims = claim_root or Path(_bulk().hosted.CLAIM_ROOT)
    claim_files = _claim_files_clear(root_claims, binding)
    session_reader = sessions or _evidence()._default_sessions  # noqa: SLF001
    session_rows = _session_absence(binding, session_reader, key)
    timestamp = observed_at or _now()
    if UTC_RE.fullmatch(timestamp) is None:
        raise DuplicateError("observation_time_invalid")
    if runtime_role not in {"source", "accept"}:
        raise DuplicateError("runtime_role_invalid")
    runtime_name = source_job(group) if runtime_role == "source" else accept_job(group)
    body = {
        "schema_version": SOURCE_SCHEMA,
        "status": "CLEAR_PENDING_SOURCE_JOB_ACCEPTANCE",
        "group": group,
        "observed_at_utc": timestamp,
        "observer_package_commit": package_commit,
        "planned_execution_count": binding["planned_execution_count"],
        "exact_identity_sha256": binding["exact_identity_sha256"],
        "checked_job_names": binding["checked_job_names"],
        "checked_configmap_names": binding["checked_configmap_names"],
        "checked_output_roots": binding["checked_output_roots"],
        "checked_cell_ids": binding["cell_ids"],
        "task_version_ids": binding["task_version_ids"],
        "environment_version_ids": binding["environment_version_ids"],
        "controller_plan_sha256": binding["controller_plan_sha256"],
        "model_serving_bindings": binding["model_serving_bindings"],
        "execution_contract": binding["execution_contract"],
        "collisions": {
            "fleet_api": 0,
            "kubernetes_job_pod_or_configmap": 0,
            "sfs_output": 0,
            "global_claim": 0,
            "accepted_active_or_model_started_cell": 0,
        },
        "global_generation_claim_files_examined": claim_files,
        "request_counts": {
            "fleet_account_get": 1,
            "fleet_task_session_gets": len(binding["task_keys"]),
            "fleet_session_rows_examined": session_rows,
            "kubernetes_metadata_gets": (
                1 + 2 * len(binding["checked_job_names"]) + len(binding["checked_configmap_names"])
            ),
            "transcript_or_score_gets": 0,
        },
        "credential_authority": {
            "secret_name": SECRET_NAME,
            "secret_uid": SECRET_UID,
            "fleet_team_id": FLEET_TEAM_ID,
        },
        "runtime": {
            "namespace": NAMESPACE,
            "kind": "Job",
            "role": runtime_role,
            "name": runtime_name,
            "job_uid": job_uid,
            "pod_uid": pod_uid,
        },
        "methods": ["GET"],
        "mutation_calls": 0,
        "prompts_traces_flags_or_scores_included": False,
        "credentials_included": False,
    }
    receipt = _seal(body)
    validate_source(receipt, group, root, package_commit, runtime_role=runtime_role)
    return receipt


def validate_source(
    receipt: dict[str, Any],
    group: str,
    root: Path,
    package_commit: str,
    *,
    runtime_role: str = "source",
) -> None:
    binding = group_binding(root, group)
    runtime = receipt.get("runtime") or {}
    requests = receipt.get("request_counts") or {}
    if (
        receipt.get("schema_version") != SOURCE_SCHEMA
        or receipt.get("status") != "CLEAR_PENDING_SOURCE_JOB_ACCEPTANCE"
        or receipt.get("group") != group
        or receipt.get("observer_package_commit") != package_commit
        or receipt.get("planned_execution_count") != binding["planned_execution_count"]
        or receipt.get("exact_identity_sha256") != binding["exact_identity_sha256"]
        or receipt.get("checked_job_names") != binding["checked_job_names"]
        or receipt.get("checked_configmap_names") != binding["checked_configmap_names"]
        or receipt.get("checked_output_roots") != binding["checked_output_roots"]
        or receipt.get("checked_cell_ids") != binding["cell_ids"]
        or receipt.get("task_version_ids") != binding["task_version_ids"]
        or receipt.get("environment_version_ids") != binding["environment_version_ids"]
        or receipt.get("controller_plan_sha256") != binding["controller_plan_sha256"]
        or receipt.get("model_serving_bindings") != binding["model_serving_bindings"]
        or receipt.get("execution_contract") != binding["execution_contract"]
        or receipt.get("collisions")
        != {
            "fleet_api": 0,
            "kubernetes_job_pod_or_configmap": 0,
            "sfs_output": 0,
            "global_claim": 0,
            "accepted_active_or_model_started_cell": 0,
        }
        or not isinstance(receipt.get("global_generation_claim_files_examined"), int)
        or receipt.get("global_generation_claim_files_examined", -1) < 0
        or requests.get("fleet_account_get") != 1
        or requests.get("fleet_task_session_gets") != len(binding["task_keys"])
        or not isinstance(requests.get("fleet_session_rows_examined"), int)
        or requests.get("fleet_session_rows_examined", -1) < 0
        or requests.get("kubernetes_metadata_gets")
        != 1 + 2 * len(binding["checked_job_names"]) + len(binding["checked_configmap_names"])
        or requests.get("transcript_or_score_gets") != 0
        or receipt.get("credential_authority")
        != {
            "secret_name": SECRET_NAME,
            "secret_uid": SECRET_UID,
            "fleet_team_id": FLEET_TEAM_ID,
        }
        or runtime.get("namespace") != NAMESPACE
        or runtime.get("kind") != "Job"
        or runtime.get("role") != runtime_role
        or runtime.get("name")
        != (source_job(group) if runtime_role == "source" else accept_job(group))
        or receipt.get("methods") != ["GET"]
        or receipt.get("mutation_calls") != 0
        or receipt.get("prompts_traces_flags_or_scores_included") is not False
        or receipt.get("credentials_included") is not False
        or receipt.get("receipt_sha256") != digest(receipt)
        or UTC_RE.fullmatch(str(receipt.get("observed_at_utc"))) is None
    ):
        raise DuplicateError("source_observation_invalid")
    _uuid(runtime.get("job_uid"), "source_job_uid")
    _uuid(runtime.get("pod_uid"), "source_pod_uid")


def accept(
    root: Path,
    group: str,
    observation: dict[str, Any],
    *,
    collector_job_uid: str,
    collector_pod_uid: str,
    package_commit: str,
    sessions: Callable[[str, str], list[dict[str, Any]]] | None = None,
    metadata: Callable[
        [str, str], dict[str, Any] | list[dict[str, Any]] | None
    ] = _in_cluster_metadata,
    kubernetes: Callable[[str, str | None], dict[str, Any]] | None = None,
    account: Callable[[str], dict[str, str]] = _fleet_account,
    api_key: str | None = None,
    observed_at: str | None = None,
    claim_root: Path | None = None,
) -> dict[str, Any]:
    validate_source(observation, group, root, package_commit)
    cluster = kubernetes or _evidence()._default_kubernetes  # noqa: SLF001
    source_runtime = observation["runtime"]
    source_state = _evidence()._succeeded_job(  # noqa: SLF001
        source_job(group), source_runtime["job_uid"], source_runtime["pod_uid"], cluster
    )
    fresh = collect(
        root,
        group,
        job_uid=collector_job_uid,
        pod_uid=collector_pod_uid,
        package_commit=package_commit,
        sessions=sessions,
        metadata=metadata,
        account=account,
        api_key=api_key,
        observed_at=observed_at,
        claim_root=claim_root,
        runtime_role="accept",
    )
    binding = group_binding(root, group)
    body = {
        "schema_version": _bulk().FRESH_SCHEMA,
        "status": "CLEAR",
        "group": group,
        "planned_execution_count": binding["planned_execution_count"],
        "exact_identity_sha256": binding["exact_identity_sha256"],
        "checked_job_names": binding["checked_job_names"],
        "checked_configmap_names": binding["checked_configmap_names"],
        "checked_output_roots": binding["checked_output_roots"],
        "checked_cell_ids": binding["cell_ids"],
        "task_version_ids": binding["task_version_ids"],
        "environment_version_ids": binding["environment_version_ids"],
        "controller_plan_sha256": binding["controller_plan_sha256"],
        "model_serving_bindings": binding["model_serving_bindings"],
        "execution_contract": binding["execution_contract"],
        "collisions": fresh["collisions"],
        "methods": ["GET"],
        "mutation_calls": 0,
        "checked_immediately_before_release": True,
        "observer_job_succeeded": True,
        "observer_pod_restarts": source_state["pod_restarts"],
        "observer_package_commit": package_commit,
        "observer_job_uid": source_runtime["job_uid"],
        "observer_pod_uid": source_runtime["pod_uid"],
        "acceptor_job_uid": _uuid(collector_job_uid, "acceptor_job_uid"),
        "acceptor_pod_uid": _uuid(collector_pod_uid, "acceptor_pod_uid"),
        "source_observation_receipt_sha256": observation["receipt_sha256"],
        "fresh_source_receipt_sha256": fresh["receipt_sha256"],
        "global_generation_claim_files_examined": fresh["global_generation_claim_files_examined"],
        "request_counts": fresh["request_counts"],
        "credential_authority": fresh["credential_authority"],
        "observed_at_utc": fresh["observed_at_utc"],
        "prompts_traces_flags_or_scores_included": False,
        "credentials_included": False,
    }
    receipt = _seal(body)
    _bulk().validate_fresh_duplicate(receipt, group, root, package_commit)
    validate_accepted_structure(receipt, group, root, package_commit)
    return receipt


def validate_accepted_structure(
    receipt: dict[str, Any], group: str, root: Path, package_commit: str
) -> None:
    _bulk().validate_fresh_duplicate(receipt, group, root, package_commit)
    binding = group_binding(root, group)
    requests = receipt.get("request_counts") or {}
    if (
        set(receipt)
        != {
            "schema_version",
            "status",
            "group",
            "planned_execution_count",
            "exact_identity_sha256",
            "checked_job_names",
            "checked_configmap_names",
            "checked_output_roots",
            "checked_cell_ids",
            "task_version_ids",
            "environment_version_ids",
            "controller_plan_sha256",
            "model_serving_bindings",
            "execution_contract",
            "collisions",
            "methods",
            "mutation_calls",
            "checked_immediately_before_release",
            "observer_job_succeeded",
            "observer_pod_restarts",
            "observer_package_commit",
            "observer_job_uid",
            "observer_pod_uid",
            "acceptor_job_uid",
            "acceptor_pod_uid",
            "source_observation_receipt_sha256",
            "fresh_source_receipt_sha256",
            "global_generation_claim_files_examined",
            "request_counts",
            "credential_authority",
            "observed_at_utc",
            "prompts_traces_flags_or_scores_included",
            "credentials_included",
            "receipt_sha256",
        }
        or receipt.get("checked_cell_ids") != binding["cell_ids"]
        or receipt.get("task_version_ids") != binding["task_version_ids"]
        or receipt.get("environment_version_ids") != binding["environment_version_ids"]
        or receipt.get("controller_plan_sha256") != binding["controller_plan_sha256"]
        or receipt.get("model_serving_bindings") != binding["model_serving_bindings"]
        or receipt.get("execution_contract") != binding["execution_contract"]
        or set(requests)
        != {
            "fleet_account_get",
            "fleet_task_session_gets",
            "fleet_session_rows_examined",
            "kubernetes_metadata_gets",
            "transcript_or_score_gets",
        }
        or requests.get("fleet_account_get") != 1
        or requests.get("fleet_task_session_gets") != len(binding["task_keys"])
        or not isinstance(requests.get("fleet_session_rows_examined"), int)
        or requests.get("fleet_session_rows_examined", -1) < 0
        or requests.get("kubernetes_metadata_gets")
        != 1 + 2 * len(binding["checked_job_names"]) + len(binding["checked_configmap_names"])
        or requests.get("transcript_or_score_gets") != 0
        or receipt.get("credential_authority")
        != {
            "secret_name": SECRET_NAME,
            "secret_uid": SECRET_UID,
            "fleet_team_id": FLEET_TEAM_ID,
        }
        or not isinstance(receipt.get("global_generation_claim_files_examined"), int)
        or receipt.get("global_generation_claim_files_examined", -1) < 0
        or SHA_RE.fullmatch(str(receipt.get("source_observation_receipt_sha256"))) is None
        or SHA_RE.fullmatch(str(receipt.get("fresh_source_receipt_sha256"))) is None
        or receipt.get("receipt_sha256") != digest(receipt)
    ):
        raise DuplicateError("accepted_duplicate_receipt_invalid")
    for field in ("observer_job_uid", "observer_pod_uid", "acceptor_job_uid", "acceptor_pod_uid"):
        _uuid(receipt.get(field), field)


def validate_accepted_for_release(
    receipt: dict[str, Any],
    group: str,
    root: Path,
    package_commit: str,
    *,
    kubernetes: Callable[[str, str | None], dict[str, Any]] | None = None,
) -> None:
    validate_accepted_structure(receipt, group, root, package_commit)
    cluster = kubernetes or _release_kubernetes
    _evidence()._succeeded_job(  # noqa: SLF001
        source_job(group), receipt["observer_job_uid"], receipt["observer_pod_uid"], cluster
    )
    _evidence()._succeeded_job(  # noqa: SLF001
        accept_job(group), receipt["acceptor_job_uid"], receipt["acceptor_pod_uid"], cluster
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("render-release", "validate-release", "source", "accept", "validate")
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--release", type=Path)
    parser.add_argument("--group")
    parser.add_argument("--package-commit")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args(argv)
    root = args.root.resolve(strict=True)
    if args.command == "render-release":
        if not args.package_commit:
            parser.error("render-release requires --package-commit")
        print(json.dumps(build_release(root, args.package_commit), sort_keys=True))
        return 0
    if args.release is None:
        parser.error(f"{args.command} requires --release")
    release = load(args.release)
    validate_release(release, root)
    if args.command == "validate-release":
        return 0
    if args.group not in _groups():
        parser.error(f"{args.command} requires a valid --group")
    group = str(args.group)
    target = args.output_root or output_root(group)
    if target != output_root(group):
        raise DuplicateError("output_root_invalid")
    package_commit = release["package_commit"]
    if args.command == "source":
        receipt = collect(
            root,
            group,
            job_uid=os.environ["JOB_UID"],
            pod_uid=os.environ["POD_UID"],
            package_commit=package_commit,
        )
        write_once(target / "OBSERVATION.json", receipt)
        return 0
    if args.command == "accept":
        observation = load(target / "OBSERVATION.json")
        receipt = accept(
            root,
            group,
            observation,
            collector_job_uid=os.environ["JOB_UID"],
            collector_pod_uid=os.environ["POD_UID"],
            package_commit=package_commit,
        )
        write_once(target / "ACCEPTED.json", receipt)
        return 0
    receipt = load(target / "ACCEPTED.json")
    validate_accepted_for_release(receipt, group, root, package_commit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
