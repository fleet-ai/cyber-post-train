"""Caller-side create-once guard for reviewed SkyRL production arms.

The guard performs only read-only checks until it writes the durable POST
intent.  It then issues exactly one Jobs API create request.  No retry path is
provided.  Credentials stay in the Jobs, Kubernetes, and W&B clients and are
never serialized into the receipt.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from cyber_post_train.jobs import JobsError, digest, safe_status, validate_request

from . import skyrl_production as production

ARMED_SCHEMA = "cyber_skyrl_release_observer_armed_v1"
KUBERNETES_RESOURCES = "rayjobs.ray.io,rayclusters.ray.io,jobs.batch,pods,workloads.kueue.x-k8s.io"
MAXIMUM_GUARD_SECONDS = 120
MAXIMUM_OBSERVER_ARM_AGE_SECONDS = 300


def _validate_seal(value: object, schema: str) -> dict:
    if not isinstance(value, dict):
        raise JobsError("release observer receipt must be an object")
    body = {key: item for key, item in value.items() if key != "sha256"}
    if value.get("schema") != schema or value.get("sha256") != "sha256:" + digest(body):
        raise JobsError("release observer receipt digest mismatch")
    return value


def _timestamp(value: object) -> float:
    if not isinstance(value, str):
        raise JobsError("release observer armed time is missing")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise JobsError("release observer armed time is invalid") from exc
    if parsed.tzinfo is None:
        raise JobsError("release observer armed time must include a timezone")
    return parsed.astimezone(UTC).timestamp()


def validate_armed_observer(
    plan: dict,
    request: dict,
    contract: dict,
    armed: object,
    *,
    now: float,
) -> dict:
    expected = production.release_observer_contract(plan, request)
    if contract != expected:
        raise JobsError("release observer contract differs from the prepared request")
    value = _validate_seal(armed, ARMED_SCHEMA)
    pid = value.get("observer_pid")
    armed_at = _timestamp(value.get("armed_at"))
    if (
        value.get("status") != "armed"
        or value.get("contract_sha256") != contract["sha256"]
        or value.get("plan_sha256") != contract["plan_sha256"]
        or value.get("request_sha256") != contract["request_sha256"]
        or value.get("kubernetes_context") != contract["kubernetes_context"]
        or value.get("namespace") != contract["namespace"]
        or value.get("expected_gpus") != 8
        or type(pid) is not int
        or pid < 1
        or not 0 <= now - armed_at <= MAXIMUM_OBSERVER_ARM_AGE_SECONDS
    ):
        raise JobsError("release observer is not freshly and exactly armed")
    try:
        os.kill(pid, 0)
    except OSError as exc:
        raise JobsError("release observer process is not alive") from exc
    return value


def _jobs_absent(client, request: dict) -> int:
    rows = client.all_runs()
    for row in rows:
        name = row.get("name", "")
        if (
            row.get("run_dir") == request["run_dir"]
            or name == request["name"]
            or name.startswith(request["name"] + "-")
        ):
            raise JobsError("Jobs history already owns the production name/output")
    return len(rows)


def _kubernetes_absent(
    binding: dict,
    request: dict,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]],
) -> int:
    result = runner(
        [
            "kubectl",
            "--context",
            binding["kubernetes_context"],
            "--namespace",
            binding["namespace"],
            "get",
            KUBERNETES_RESOURCES,
            "--output=json",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if result.returncode:
        raise JobsError("fresh Kubernetes duplicate check failed")
    try:
        payload = json.loads(result.stdout)
        items = payload["items"]
    except (KeyError, TypeError, ValueError) as exc:
        raise JobsError("fresh Kubernetes duplicate check returned invalid JSON") from exc
    if not isinstance(items, list):
        raise JobsError("fresh Kubernetes duplicate inventory is malformed")
    for item in items:
        metadata = item.get("metadata", {}) if isinstance(item, dict) else {}
        name = metadata.get("name", "")
        values = [
            *metadata.get("labels", {}).values(),
            *metadata.get("annotations", {}).values(),
        ]
        if (
            name == request["name"]
            or name.startswith(request["name"] + "-")
            or request["name"] in values
            or request["run_dir"] in values
        ):
            raise JobsError("Kubernetes already contains the production name/output")
    return len(items)


def _wandb_exists_default(entity: str, project: str, run_id: str) -> bool:
    try:
        import wandb
        from wandb.errors import CommError
    except ImportError as exc:
        raise JobsError("W&B client is required for the fresh run-ID check") from exc
    if not os.environ.get("WANDB_API_KEY"):
        raise JobsError("W&B read credential is required for the fresh run-ID check")
    try:
        run = wandb.Api(timeout=30).run(f"{entity}/{project}/{run_id}")
    except CommError as exc:
        response = getattr(exc, "response", None)
        if getattr(response, "status_code", None) == 404:
            return False
        raise JobsError("fresh W&B run-ID check failed") from None
    return run is not None


def _sealed_proof(value: dict) -> dict:
    result = dict(value)
    result["sha256"] = "sha256:" + digest(result)
    return result


def fresh_absence_checks(
    plan: dict,
    request: dict,
    client,
    contract: dict,
    armed: dict,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    wandb_exists: Callable[[str, str, str], bool] = _wandb_exists_default,
    now: Callable[[], float] = time.time,
) -> dict:
    """Check every create-once destination in one bounded caller transaction."""
    started = now()
    binding = production.validate_plan_binding(
        plan.get("qualification"), plan["data"], plan["arguments"], plan["model"]
    )
    validate_armed_observer(plan, request, contract, armed, now=started)
    staged = production.validate_staged_data(plan)
    jobs_rows = _jobs_absent(client, request)
    kubernetes_objects = _kubernetes_absent(binding, request, runner=runner)
    arguments = plan["arguments"]
    if wandb_exists(
        arguments["wandb_entity"], arguments["wandb_project"], arguments["wandb_run_id"]
    ):
        raise JobsError("W&B run ID already exists")
    finished = now()
    if finished < started or finished - started > MAXIMUM_GUARD_SECONDS:
        raise JobsError("fresh destination checks expired before submission")
    return _sealed_proof(
        {
            "schema": "cyber_skyrl_production_fresh_absence_v1",
            "status": "passed",
            "plan_sha256": "sha256:" + digest(plan),
            "request_sha256": "sha256:" + digest(request),
            "release_observer_armed_sha256": armed["sha256"],
            "jobs_history_rows_checked": jobs_rows,
            "kubernetes_objects_checked": kubernetes_objects,
            "sfs_validation_sha256": staged["sha256"],
            "sfs_output_absent": True,
            "wandb_run_id_absent": True,
            "observed_at_unix": finished,
            "elapsed_seconds": finished - started,
        }
    )


def _canary_sfs_absent(plan: dict) -> dict:
    """Reopen the exact prod4 manifest/payloads and prove its output is absent."""
    from .rl_reward_canary import validate_plan_binding

    validate_plan_binding(plan.get("qualification"), plan["data"], plan["arguments"])
    root = Path(plan["arguments"]["data_manifest"]).parent
    root_stat = root.lstat()
    if (
        root.is_symlink()
        or not root.is_dir()
        or (root_stat.st_uid, root_stat.st_gid) != (1000, 100)
        or stat.S_IMODE(root_stat.st_mode) != 0o700
    ):
        raise JobsError("prod4 SFS data root ownership or mode changed")
    expected_names = {
        "manifest.json",
        "split.json",
        "task-set.json",
        *(item["path"] for item in plan["data"]["files"].values()),
    }
    found = {path.name: path for path in root.iterdir()}
    if set(found) != expected_names:
        raise JobsError("prod4 SFS file set changed")
    try:
        manifest = json.loads((root / "manifest.json").read_bytes())
    except (OSError, ValueError) as exc:
        raise JobsError("prod4 SFS manifest is unavailable") from exc
    if manifest != plan["data"]:
        raise JobsError("prod4 SFS manifest differs from the prepared plan")
    documents = {
        "manifest.json": manifest,
        "split.json": json.loads((root / "split.json").read_bytes()),
        "task-set.json": json.loads((root / "task-set.json").read_bytes()),
    }
    if (
        documents["split.json"].get("sha256") != manifest["split_sha256"]
        or documents["task-set.json"].get("sha256") != manifest["selection_sha256"]
        or any(
            document.get("sha256")
            != "sha256:"
            + digest({key: value for key, value in document.items() if key != "sha256"})
            for document in documents.values()
        )
    ):
        raise JobsError("prod4 SFS manifest, split, or task-set seal changed")
    checked = []
    for name, path in sorted(found.items()):
        value = hashlib.sha256()
        try:
            before = path.lstat()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    value.update(block)
            after = path.lstat()
        except OSError as exc:
            raise JobsError("prod4 SFS input is unavailable") from exc
        expected_sha256 = next(
            (item["sha256"] for item in manifest["files"].values() if item["path"] == name),
            None,
        )
        if (
            path.is_symlink()
            or not path.is_file()
            or (before.st_uid, before.st_gid) != (1000, 100)
            or stat.S_IMODE(before.st_mode) != 0o600
            or (expected_sha256 is not None and "sha256:" + value.hexdigest() != expected_sha256)
            or any(
                getattr(before, field) != getattr(after, field)
                for field in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
            )
        ):
            raise JobsError("prod4 SFS input identity changed")
        checked.append({"path": name, "bytes": before.st_size, "sha256": value.hexdigest()})
    final_root_stat = root.lstat()
    if any(
        getattr(root_stat, field) != getattr(final_root_stat, field)
        for field in (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_uid",
            "st_gid",
            "st_mtime_ns",
            "st_ctime_ns",
        )
    ):
        raise JobsError("prod4 SFS data root changed during validation")
    output = Path(plan["output_root"])
    if output.exists() or output.is_symlink():
        raise JobsError("prod4 SFS output already exists")
    return {
        "manifest_sha256": manifest["sha256"],
        "files_checked": checked,
        "output_absent": True,
    }


def fresh_canary_absence_checks(
    plan: dict,
    request: dict,
    client,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    wandb_exists: Callable[[str, str, str], bool] = _wandb_exists_default,
    now: Callable[[], float] = time.time,
) -> dict:
    """Run the exact prod4 Jobs/Kubernetes/SFS/W&B checks before create."""
    started = now()
    sfs = _canary_sfs_absent(plan)
    jobs_rows = _jobs_absent(client, request)
    kubernetes_objects = _kubernetes_absent(
        {
            "kubernetes_context": production.PROD_CONTEXT,
            "namespace": production.NAMESPACE,
        },
        request,
        runner=runner,
    )
    arguments = plan["arguments"]
    if wandb_exists(
        arguments["wandb_entity"], arguments["wandb_project"], arguments["wandb_run_id"]
    ):
        raise JobsError("W&B run ID already exists")
    finished = now()
    if finished < started or finished - started > MAXIMUM_GUARD_SECONDS:
        raise JobsError("fresh destination checks expired before submission")
    return _sealed_proof(
        {
            "schema": "cyber_skyrl_reward_canary_fresh_absence_v1",
            "status": "passed",
            "plan_sha256": "sha256:" + digest(plan),
            "request_sha256": "sha256:" + digest(request),
            "jobs_history_rows_checked": jobs_rows,
            "kubernetes_objects_checked": kubernetes_objects,
            "sfs": sfs,
            "wandb_run_id_absent": True,
            "observed_at_unix": finished,
            "elapsed_seconds": finished - started,
        }
    )


def _journal_and_post(
    request: dict,
    client,
    journal: Path,
    preview_proof: dict,
    guard: dict,
) -> dict:
    journal.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(journal, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        stream.write(
            json.dumps(
                {
                    "state": "POST_INTENT_DO_NOT_RETRY",
                    "request_sha256": digest(request),
                    "fresh_absence": guard,
                    **preview_proof,
                },
                sort_keys=True,
            )
            + "\n"
        )
        stream.flush()
        os.fsync(stream.fileno())
    response = client.request("POST", "/v1/runs", json=request)
    if not re.fullmatch(re.escape(request["name"]) + r"-[a-f0-9]{8}", response.get("name", "")):
        raise JobsError("ambiguous submit response; reconcile journal, never repeat POST")
    if response.get("run_dir") not in (None, request["run_dir"]):
        raise JobsError("submitted output differs; reconcile resource ownership immediately")
    result = safe_status(response)
    with journal.open("a") as stream:
        stream.write(json.dumps({"state": "POST_RESPONSE", **result}, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return result


def submit_canary_once(
    plan: dict,
    request: dict,
    client,
    directory: Path,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    wandb_exists: Callable[[str, str, str], bool] = _wandb_exists_default,
    now: Callable[[], float] = time.time,
) -> dict:
    """Submit prod4 once only after its caller-specific fresh checks pass."""
    from .skyrl_training import validate_preview

    validate_request(request)
    journal = directory / "SUBMISSION.jsonl"
    if journal.exists() or journal.is_symlink():
        raise JobsError("submission journal already exists; reconcile, never repeat POST")
    preview_proof = validate_preview(plan, request, client.preview(request))
    guard = fresh_canary_absence_checks(
        plan,
        request,
        client,
        runner=runner,
        wandb_exists=wandb_exists,
        now=now,
    )
    return _journal_and_post(request, client, journal, preview_proof, guard)


def submit_once(
    plan: dict,
    request: dict,
    client,
    directory: Path,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    wandb_exists: Callable[[str, str, str], bool] = _wandb_exists_default,
    now: Callable[[], float] = time.time,
) -> dict:
    """Preview, recheck all destinations, journal, and issue one create POST."""
    from .skyrl_production_training import validate_preview

    validate_request(request)
    journal = directory / "SUBMISSION.jsonl"
    if journal.exists() or journal.is_symlink():
        raise JobsError("submission journal already exists; reconcile, never repeat POST")
    contract = json.loads((directory / "RELEASE_OBSERVER_CONTRACT.json").read_bytes())
    armed = json.loads((directory / "RELEASE_OBSERVER_ARMED.json").read_bytes())
    preview = client.preview(request)
    preview_proof = validate_preview(plan, request, preview)
    guard = fresh_absence_checks(
        plan,
        request,
        client,
        contract,
        armed,
        runner=runner,
        wandb_exists=wandb_exists,
        now=now,
    )
    return _journal_and_post(request, client, journal, preview_proof, guard)
