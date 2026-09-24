"""Create-once launcher and exact-UID cleanup coordinator for Miles96.

This module is caller-side only.  It performs a live Jobs API preview, arms a
separate cleanup process before the sole POST, rechecks every destination, and
writes a durable intent before creation.  It never retries an uncertain POST.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from cyber_post_train.gpu_capacity import ROLE_LABELS, live_capacity_census
from cyber_post_train.jobs import (
    Jobs,
    JobsError,
    digest,
    safe_status,
    validate_preview,
    validate_request,
)
from cyber_post_train.sfs_output import SFS_JOBS_ROOT, prove_output_absent
from training import dev_cleanup_observer as cleanup
from training import miles96_mechanics_canary as mechanics

KUBERNETES_RESOURCES = "rayjobs.ray.io,rayclusters.ray.io,jobs.batch,pods,workloads.kueue.x-k8s.io"
MAXIMUM_GUARD_SECONDS = 120
MAXIMUM_ARM_AGE_SECONDS = 300
CAPACITY_MAX_AGE_SECONDS = 120
SIGNAL_MODEL_STARTUP_GRACE_SECONDS = 1800
SIGNAL_SLOT_CLOSE_GRACE_SECONDS = 60
SIGNAL_RELEASE_GRACE_SECONDS = 300
POST_RECONCILIATION_SECONDS = 600
PROJECT_OWNER_PREFIXES = ("chris-q38-",)
PROJECT_MAX_NODES = 10
PROJECT_MAX_GPUS = 80
COORDINATOR_RESULT_SCHEMA = "cyber_miles96_cleanup_coordinator_v1"
CAPACITY_GATE_SCHEMA = "cyber_miles96_capacity_gate_v1"
_SFS_OBSERVER_ROLE = "sfs-output-check"
_SFS_OBSERVER_NAME = re.compile(
    r"(?P<job>[a-z0-9](?:[-a-z0-9]*[a-z0-9])?-sfs-a[0-9]{2})(?:-[a-z0-9]+)?"
)


def _validate_plan(plan: dict[str, Any]) -> dict[str, Any]:
    if plan.get("schema") == mechanics.SCHEMA:
        return mechanics.validate_plan(plan)
    from training import miles96_signal_qualification as signal

    return signal.validate_plan(plan)


def _identity_names(plan: dict[str, Any]) -> set[str]:
    identity = plan["identity"]
    return {identity[key] for key in ("name", "reload_name") if key in identity}


def _seal(value: dict[str, Any]) -> dict[str, Any]:
    return {**value, "sha256": "sha256:" + digest(value)}


def _write_json_once(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        stream.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _timestamp(value: object) -> float:
    if not isinstance(value, str):
        raise JobsError("cleanup observer armed time is absent")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise JobsError("cleanup observer armed time is invalid") from exc
    if parsed.tzinfo is None:
        raise JobsError("cleanup observer armed time lacks a timezone")
    return parsed.astimezone(UTC).timestamp()


def _paths(directory: Path) -> dict[str, Path]:
    return {
        "plan": directory / "PLAN.json",
        "request": directory / "REQUEST.json",
        "preview": directory / "SERVER_PREVIEW.json",
        "armed": directory / "OBSERVER_ARMED.json",
        "binding": directory / "EXACT_BINDING.json",
        "release": directory / "EXACT_RELEASE_CONTRACT.json",
        "observer": directory / "EXACT_OBSERVER_RESULT.json",
        "coordinator": directory / "COORDINATOR_RESULT.json",
        "journal": directory / "SUBMISSION.jsonl",
    }


def _maximum_seconds(plan: dict[str, Any], request: dict[str, Any]) -> int:
    """Return the active-allocation bound; queue time starts no clock.

    The mechanics canary has one episode and retains its reviewed two-hour
    bound.  Signal qualification has eight immutable slots but admits only two
    environments at once.  Its four serial waves therefore need their own
    deterministic envelope instead of inheriting the shorter mechanics bound.
    """
    from training import miles96_signal_qualification as signal

    if plan.get("schema") != signal.SCHEMA:
        return 7200
    episode = plan["episode"]
    waves = math.ceil(
        plan["qualification"]["samples"] / episode["max_concurrent_envs"]
    )
    per_wave = (
        episode["ready_timeout_s"]
        + episode["episode_timeout_s"]
        + episode["grade_timeout_s"]
        + 2 * episode["request_timeout_s"]
        + SIGNAL_SLOT_CLOSE_GRACE_SECONDS
    )
    maximum = (
        SIGNAL_MODEL_STARTUP_GRACE_SECONDS
        + waves * per_wave
        + SIGNAL_RELEASE_GRACE_SECONDS
    )
    if not 7200 < maximum <= 24 * 60 * 60:
        raise JobsError("signal qualification active-runtime envelope is invalid")
    return maximum


def _guard(
    plan: dict[str, Any], request: dict[str, Any], directory: Path
) -> cleanup.JobsApiPrefixGuard:
    paths = _paths(directory)
    preview = json.loads(paths["preview"].read_text())
    return cleanup.JobsApiPrefixGuard(
        context=plan["execution"]["kubernetes_context"],
        namespace=plan["execution"]["namespace"],
        run_name_prefix=request["name"],
        run_dir=request["run_dir"],
        image=request["image"],
        plan_sha256="sha256:" + mechanics.digest(plan),
        manifest_sha256=preview["manifest_sha256"],
        maximum_seconds=_maximum_seconds(plan, request),
        expected_gpus=request["workers"] * request["gpus_per_worker"],
        armed_path=paths["armed"],
        binding_path=paths["binding"],
    )


def _release_contract(binding: dict[str, Any]) -> dict[str, Any]:
    return _seal(
        {
            "schema": cleanup.JOBS_API_RELEASE_CONTRACT_SCHEMA,
            "status": "creator_authorized_exact_uid_release",
            "binding_sha256": binding["sha256"],
            "context": binding["context"],
            "namespace": binding["namespace"],
            "jobs_api_run_name": binding["jobs_api_run_name"],
            "jobs_api_run_id": binding["jobs_api_run_id"],
            "rayjob_name": binding["rayjob_name"],
            "rayjob_uid": binding["rayjob_uid"],
            "authorized_at": binding["bound_at"],
            "release_route": "raw_rayjob_uid_precondition_v1",
        }
    )


def _post_response(journal: Path, deadline: float) -> dict[str, Any] | None:
    while time.monotonic() < deadline:
        if journal.is_file():
            try:
                rows = [json.loads(line) for line in journal.read_text().splitlines() if line]
            except (OSError, ValueError):
                rows = []
            matches = [
                row
                for row in rows
                if row.get("state") in {"POST_RESPONSE", "POST_RESPONSE_RECONCILED"}
            ]
            if len(matches) > 1:
                raise cleanup.ObserverError("submission journal contains duplicate POST responses")
            if matches:
                return matches[0]
        time.sleep(0.25)
    return None


def _append_journal(path: Path, value: dict[str, Any]) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _reconcile_post_response(
    plan: dict[str, Any],
    request: dict[str, Any],
    journal: Path,
    *,
    deadline: float | None = None,
    poll_seconds: float = 5.0,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Resolve one uncertain POST from authenticated history, never by retrying it."""
    try:
        rows = [json.loads(line) for line in journal.read_text().splitlines() if line]
    except (OSError, ValueError) as exc:
        raise cleanup.ObserverError("submission intent is unavailable for reconciliation") from exc
    intents = [row for row in rows if row.get("state") == "POST_INTENT_DO_NOT_RETRY"]
    if len(intents) != 1 or intents[0].get("request_sha256") != digest(request):
        raise cleanup.ObserverError("submission intent differs during reconciliation")
    token = os.environ.get("FLEET_API_KEY")
    if not token:
        raise cleanup.ObserverError("Jobs API credential is unavailable for reconciliation")
    pattern = re.compile(re.escape(request["name"]) + r"-[a-f0-9]{8}")
    while True:
        try:
            with Jobs(token, base_url=plan["execution"]["jobs_api_base_url"]) as client:
                # Authenticated run reads do not expose the submitted title.
                # Bind the server-generated name to the exact create-once
                # output, immutable image, priority and no-requeue policy.
                matches = [
                    row
                    for row in client.all_runs()
                    if pattern.fullmatch(str(row.get("name", "")))
                    and row.get("run_dir") == request["run_dir"]
                    and row.get("image") == request["image"]
                    and row.get("priority_class") == request["priority_class"]
                    and row.get("requeueIfPreempted") is False
                ]
        except JobsError:
            matches = []
        if len(matches) > 1:
            raise cleanup.ObserverError("uncertain POST reconciled to multiple runs")
        if matches:
            break
        if deadline is None or monotonic() >= deadline:
            raise cleanup.ObserverError("uncertain POST did not reconcile to exactly one run")
        sleep(min(poll_seconds, max(0.0, deadline - monotonic())))
    observed = safe_status(matches[0])
    try:
        from uuid import UUID

        UUID(observed["job_id"])
    except (TypeError, ValueError) as exc:
        raise cleanup.ObserverError("reconciled Jobs API identity is invalid") from exc
    if observed.get("run_dir") != request["run_dir"]:
        raise cleanup.ObserverError("reconciled Jobs API output differs")
    value = {"state": "POST_RESPONSE_RECONCILED", **observed}
    _append_journal(journal, value)
    return value


def run_cleanup_coordinator(plan: dict[str, Any], request: dict[str, Any], directory: Path) -> None:
    """Arm before POST, then bind and watch only the exact creator-returned UID."""
    paths = _paths(directory)
    try:
        guard = _guard(plan, request, directory)
        guard.arm()
        response = _post_response(paths["journal"], time.monotonic() + 300)
        if response is None:
            response = _reconcile_post_response(
                plan,
                request,
                paths["journal"],
                deadline=time.monotonic() + POST_RECONCILIATION_SECONDS,
            )
        binding = guard.bind_exact(
            {
                "jobs_api_run_name": response.get("name"),
                "jobs_api_run_id": response.get("job_id"),
                "run_dir": response.get("run_dir"),
            }
        )
        _write_json_once(paths["release"], _release_contract(binding))
        result = cleanup.JobsApiExactUidObserver(
            binding_path=paths["binding"],
            result_path=paths["observer"],
            release_contract_path=paths["release"],
        ).run()
        final = _seal(
            {
                "schema": COORDINATOR_RESULT_SCHEMA,
                "status": "finished",
                "binding_sha256": binding["sha256"],
                "observer_result_sha256": result["sha256"],
                "release_confirmed": result["release_confirmed"],
            }
        )
    except Exception as exc:
        final = _seal(
            {
                "schema": COORDINATOR_RESULT_SCHEMA,
                "status": "release_uncertain",
                "reason": type(exc).__name__,
                "release_confirmed": False,
            }
        )
    _write_json_once(paths["coordinator"], final)


def validate_armed_observer(
    plan: dict[str, Any], request: dict[str, Any], directory: Path, *, now: float
) -> dict[str, Any]:
    paths = _paths(directory)
    try:
        value = json.loads(paths["armed"].read_text())
        preview = json.loads(paths["preview"].read_text())
    except (OSError, ValueError) as exc:
        raise JobsError("cleanup observer is not armed") from exc
    body = {key: item for key, item in value.items() if key != "sha256"}
    pid = value.get("observer_pid")
    if (
        value.get("schema") != cleanup.JOBS_API_PREFIX_GUARD_SCHEMA
        or value.get("status") != "armed_non_destructive_prefix_guard"
        or value.get("sha256") != "sha256:" + digest(body)
        or value.get("context") != plan["execution"]["kubernetes_context"]
        or value.get("namespace") != plan["execution"]["namespace"]
        or value.get("run_name_prefix") != request["name"]
        or value.get("generated_name_pattern")
        != "^" + re.escape(request["name"]) + r"-[a-f0-9]{8}$"
        or value.get("run_dir") != request["run_dir"]
        or value.get("image") != request["image"]
        or value.get("plan_sha256") != "sha256:" + mechanics.digest(plan)
        or value.get("manifest_sha256") != preview.get("manifest_sha256")
        or value.get("maximum_seconds") != _maximum_seconds(plan, request)
        or value.get("expected_gpus") != request["workers"] * request["gpus_per_worker"]
        or value.get("prefix_collision_count_before_post") != 0
        or type(pid) is not int
        or pid < 1
        or not 0 <= now - _timestamp(value.get("armed_at")) <= MAXIMUM_ARM_AGE_SECONDS
    ):
        raise JobsError("cleanup observer armed receipt differs from the request")
    try:
        os.kill(pid, 0)
    except OSError as exc:
        raise JobsError("cleanup observer process is not alive") from exc
    return value


def _jobs_absent(client: Any, request: dict[str, Any]) -> int:
    rows = client.all_runs()
    for row in rows:
        name = row.get("name", "")
        if (
            row.get("run_dir") == request["run_dir"]
            or name == request["name"]
            or name.startswith(request["name"] + "-")
            or row.get("title") == request["title"]
        ):
            raise JobsError("Jobs history already owns this name, title, or output")
    return len(rows)


def _kubernetes_absent(
    plan: dict[str, Any],
    request: dict[str, Any],
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]],
) -> int:
    result = runner(
        [
            "kubectl",
            "--context",
            plan["execution"]["kubernetes_context"],
            "--namespace",
            plan["execution"]["namespace"],
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
        items = json.loads(result.stdout)["items"]
    except (KeyError, TypeError, ValueError) as exc:
        raise JobsError("fresh Kubernetes duplicate inventory is invalid") from exc
    if not isinstance(items, list):
        raise JobsError("fresh Kubernetes duplicate inventory is invalid")
    for item in items:
        metadata = item.get("metadata", {}) if isinstance(item, dict) else {}
        name = metadata.get("name", "")
        values = list((metadata.get("labels") or {}).values()) + list(
            (metadata.get("annotations") or {}).values()
        )
        collision = (
            name == request["name"]
            or name.startswith(request["name"] + "-")
            or request["name"] in values
            or request["run_dir"] in values
        )
        if collision and _is_terminal_sfs_observer(item, plan, request):
            continue
        if collision:
            raise JobsError("Kubernetes already contains this name or output")
    return len(items)


def _is_terminal_sfs_observer(item: object, plan: dict[str, Any], request: dict[str, Any]) -> bool:
    """Recognize only the exact completed zero-GPU observer used by this launch.

    The SFS observer intentionally carries the target run name and output in its
    metadata, so a generic prefix scan would otherwise reject the evidence Job
    that produced the required fresh receipt.  This exemption is deliberately
    narrow: exact source bindings plus terminal success are required, and no
    Ray resource or non-observer Job can enter it.
    """
    if not isinstance(item, dict) or item.get("kind") not in {"Job", "Pod"}:
        return False
    metadata = item.get("metadata")
    if not isinstance(metadata, dict):
        return False
    name = metadata.get("name")
    match = _SFS_OBSERVER_NAME.fullmatch(name) if isinstance(name, str) else None
    if match is None or not match.group("job").startswith(request["name"] + "-sfs-a"):
        return False
    labels = metadata.get("labels")
    annotations = metadata.get("annotations")
    if not isinstance(labels, dict) or not isinstance(annotations, dict):
        return False
    if (
        labels.get("cyber-post-train.fleet.ai/role") != _SFS_OBSERVER_ROLE
        or labels.get("cyber-post-train.fleet.ai/owner") != "chris"
        or labels.get("kueue.x-k8s.io/queue-name") != "training-lq"
        or labels.get("kueue.x-k8s.io/priority-class") != "q1"
        or annotations.get("fleet.ai/failure-alerts") != "off"
        or annotations.get("cyber-post-train.fleet.ai/plan-sha256") != mechanics.digest(plan)
        or annotations.get("cyber-post-train.fleet.ai/request-sha256") != digest(request)
        or annotations.get("cyber-post-train.fleet.ai/training-output") != request["run_dir"]
        or not re.fullmatch(
            r"[0-9a-f]{64}",
            annotations.get("cyber-post-train.fleet.ai/driver-sha256", ""),
        )
    ):
        return False
    status = item.get("status")
    if not isinstance(status, dict):
        return False
    if item["kind"] == "Job":
        conditions = status.get("conditions")
        return (
            isinstance(conditions, list)
            and any(
                isinstance(value, dict)
                and value.get("type") == "Complete"
                and value.get("status") == "True"
                for value in conditions
            )
            and not any(
                isinstance(value, dict)
                and value.get("type") == "Failed"
                and value.get("status") == "True"
                for value in conditions
            )
        )
    statuses = status.get("containerStatuses")
    if status.get("phase") != "Succeeded" or not isinstance(statuses, list) or len(statuses) != 1:
        return False
    terminated = statuses[0].get("state", {}).get("terminated", {})
    return statuses[0].get("restartCount") == 0 and terminated.get("exitCode") == 0


def _prove_sfs_output_absent(
    plan: dict[str, Any],
    request: dict[str, Any],
    *,
    jobs_root: Path,
    receipt: dict[str, Any] | None,
    observed_at: float,
) -> dict[str, Any]:
    try:
        return prove_output_absent(
            plan,
            request,
            jobs_root=jobs_root,
            receipt=receipt,
            now=observed_at,
        )
    except ValueError as exc:
        raise JobsError(str(exc)) from None


def _wandb_exists_default(entity: str, project: str, run_id: str) -> bool:
    try:
        import wandb
        from wandb.errors import CommError
    except ImportError as exc:
        raise JobsError("W&B client is required for run-ID absence") from exc
    if not os.environ.get("WANDB_API_KEY"):
        raise JobsError("W&B credential is required for run-ID absence")
    try:
        return wandb.Api(timeout=30).run(f"{entity}/{project}/{run_id}") is not None
    except CommError as exc:
        if getattr(getattr(exc, "response", None), "status_code", None) == 404:
            return False
        raise JobsError("W&B run-ID absence check failed") from None


def fresh_absence_checks(
    plan: dict[str, Any],
    request: dict[str, Any],
    client: Any,
    directory: Path,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    jobs_root: Path = SFS_JOBS_ROOT,
    output_absence_receipt: dict[str, Any] | None = None,
    wandb_exists: Callable[[str, str, str], bool] = _wandb_exists_default,
    now: Callable[[], float] = time.time,
) -> dict[str, Any]:
    started = now()
    armed = validate_armed_observer(plan, request, directory, now=started)
    jobs_rows = _jobs_absent(client, request)
    kubernetes_objects = _kubernetes_absent(plan, request, runner=runner)
    sfs_proof = _prove_sfs_output_absent(
        plan,
        request,
        jobs_root=jobs_root,
        receipt=output_absence_receipt,
        observed_at=now(),
    )
    wandb_absent: bool | None = None
    if "wandb" in plan and request["name"] == plan["identity"]["name"]:
        value = plan["wandb"]
        wandb_absent = not wandb_exists(value["entity"], value["project"], value["run_id"])
        if not wandb_absent:
            raise JobsError("W&B run ID already exists")
    finished = now()
    if finished < started or finished - started > MAXIMUM_GUARD_SECONDS:
        raise JobsError("fresh destination checks expired before submission")
    return _seal(
        {
            "schema": "cyber_miles96_fresh_absence_v1",
            "status": "passed",
            "plan_sha256": "sha256:" + mechanics.digest(plan),
            "request_sha256": "sha256:" + digest(request),
            "observer_armed_sha256": armed["sha256"],
            "jobs_history_rows_checked": jobs_rows,
            "kubernetes_objects_checked": kubernetes_objects,
            "sfs_output_absent": True,
            "sfs_output_absence_receipt_sha256": sfs_proof["sha256"],
            "wandb_run_id_absent": wandb_absent,
            "observed_at_unix": finished,
            "elapsed_seconds": finished - started,
        }
    )


def final_prepost_checks(
    plan: dict[str, Any],
    request: dict[str, Any],
    client: Any,
    directory: Path,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    now: Callable[[], float] = time.time,
) -> dict[str, Any]:
    """Recheck observer liveness and duplicate destinations immediately pre-POST."""
    started = now()
    armed = validate_armed_observer(plan, request, directory, now=started)
    jobs_rows = _jobs_absent(client, request)
    kubernetes_objects = _kubernetes_absent(plan, request, runner=runner)
    finished = now()
    if finished < started or finished - started > MAXIMUM_GUARD_SECONDS:
        raise JobsError("final pre-POST checks expired")
    return _seal(
        {
            "schema": "cyber_miles96_final_prepost_gate_v1",
            "status": "passed",
            "plan_sha256": "sha256:" + mechanics.digest(plan),
            "request_sha256": "sha256:" + digest(request),
            "observer_armed_sha256": armed["sha256"],
            "jobs_history_rows_checked": jobs_rows,
            "kubernetes_objects_checked": kubernetes_objects,
            "observed_at_unix": finished,
            "elapsed_seconds": finished - started,
        }
    )


def _start_observer(directory: Path) -> subprocess.Popen[bytes]:
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "training.miles96_mechanics_launch",
            "--observe",
            "--plan",
            str(_paths(directory)["plan"]),
            "--request",
            str(_paths(directory)["request"]),
            "--directory",
            str(directory),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    deadline = time.monotonic() + 30
    armed = _paths(directory)["armed"]
    while not armed.is_file():
        if process.poll() is not None:
            raise JobsError("cleanup observer exited before arming")
        if time.monotonic() >= deadline:
            process.terminate()
            raise JobsError("cleanup observer did not arm in time")
        time.sleep(0.1)
    return process


def _expected_request(
    plan: dict[str, Any], request: dict[str, Any], receipt: dict[str, Any] | None
) -> dict[str, Any]:
    from training import miles96_signal_qualification as signal

    if plan.get("schema") == signal.SCHEMA and receipt is None:
        return signal.job_request(plan)
    if request.get("name") == plan["identity"]["name"] and receipt is None:
        return mechanics.job_request(plan)
    if request.get("name") == plan["identity"]["reload_name"] and receipt is not None:
        return mechanics.reload_request(plan, receipt)
    raise JobsError("request mode and receipt do not form a valid mechanics launch")


def _live_preview_proof(request: dict[str, Any], preview: dict[str, Any]) -> dict[str, Any]:
    """Validate the shared contract plus the canary's no-controller-retry rule."""
    proof = validate_preview(request, preview)
    try:
        rendered = yaml.safe_load(preview["manifest_yaml"])
        backoff_limit = rendered["spec"]["backoffLimit"]
    except (KeyError, TypeError, yaml.YAMLError) as exc:
        raise JobsError("live preview lacks an explicit controller retry limit") from exc
    if type(backoff_limit) is not int or backoff_limit != 0:
        raise JobsError("live preview must disable controller retries")
    return proof


def capacity_gate(
    plan: dict[str, Any],
    request: dict[str, Any],
    *,
    reader: Callable[..., dict[str, Any]] = live_capacity_census,
    now: Callable[[], float] = time.time,
) -> dict[str, Any]:
    """Read and seal one fresh cross-namespace census immediately before create."""
    planned_gpus = request.get("workers", 0) * request.get("gpus_per_worker", 0)
    if (
        request.get("name") not in _identity_names(plan)
        or not request["name"].startswith(PROJECT_OWNER_PREFIXES)
        or request.get("workers") != 1
        or planned_gpus not in {1, 8}
        or request.get("priority_class") != "c1"
        or request.get("failureAlerts") is not False
    ):
        raise JobsError("Miles96 GPU capacity binding changed")
    try:
        census = reader(
            plan["execution"]["kubernetes_context"],
            owner_prefixes=PROJECT_OWNER_PREFIXES,
            max_nodes=PROJECT_MAX_NODES,
            max_gpus=PROJECT_MAX_GPUS,
            planned_nodes=1,
            planned_gpus=planned_gpus,
        )
    except Exception as exc:
        raise JobsError("Miles96 cross-namespace GPU capacity census failed") from exc
    if not isinstance(census, dict):
        raise JobsError("Miles96 cross-namespace GPU capacity census is invalid")
    try:
        observed_at = _timestamp(census.get("observed_at"))
    except JobsError as exc:
        raise JobsError("Miles96 GPU capacity observation is invalid") from exc
    unsigned = {key: value for key, value in census.items() if key != "sha256"}
    current = census.get("current")
    projected = census.get("projected")
    scope = census.get("scope")
    role_counts = current.get("role_pod_counts") if isinstance(current, dict) else None
    age = now() - observed_at
    if (
        census.get("sha256") != digest(unsigned)
        or census.get("schema") != "cyber_project_gpu_capacity_census_v1"
        or scope
        != {
            "kubernetes_namespaces": "all",
            "owner_prefixes": list(PROJECT_OWNER_PREFIXES),
            "ownership_labels": ROLE_LABELS,
        }
        or census.get("limits") != {"nodes": PROJECT_MAX_NODES, "gpus": PROJECT_MAX_GPUS}
        or census.get("planned") != {"nodes": 1, "gpus": planned_gpus}
        or census.get("qualified") is not True
        or census.get("problems") != []
        or not isinstance(current, dict)
        or not isinstance(projected, dict)
        or not isinstance(role_counts, dict)
        or role_counts.get("unclassified") != 0
        or type(current.get("nodes")) is not int
        or type(current.get("gpus")) is not int
        or projected != {"nodes": current["nodes"] + 1, "gpus": current["gpus"] + planned_gpus}
        or projected["nodes"] > PROJECT_MAX_NODES
        or projected["gpus"] > PROJECT_MAX_GPUS
        or not 0 <= age <= CAPACITY_MAX_AGE_SECONDS
    ):
        raise JobsError("Miles96 GPU capacity proof is stale or incomplete")
    return _seal(
        {
            "schema": CAPACITY_GATE_SCHEMA,
            "status": "passed",
            "plan_sha256": "sha256:" + mechanics.digest(plan),
            "request_sha256": "sha256:" + digest(request),
            "planned": {"nodes": 1, "gpus": planned_gpus},
            "observed_at": census["observed_at"],
            "capacity_census": census,
        }
    )


def submit_once(
    plan: dict[str, Any],
    request: dict[str, Any],
    client: Any,
    directory: Path,
    *,
    receipt: dict[str, Any] | None = None,
    output_absence_receipt: dict[str, Any] | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    jobs_root: Path = SFS_JOBS_ROOT,
    wandb_exists: Callable[[str, str, str], bool] = _wandb_exists_default,
    capacity_reader: Callable[..., dict[str, Any]] = live_capacity_census,
    now: Callable[[], float] = time.time,
    start_observer: Callable[[Path], subprocess.Popen[bytes]] = _start_observer,
) -> dict[str, Any]:
    """Preview, arm, recheck, journal, and issue exactly one Jobs API POST."""
    plan = _validate_plan(plan)
    if request != _expected_request(plan, request, receipt):
        raise JobsError("request differs from the exact immutable mechanics plan")
    validate_request(request)
    paths = _paths(directory)
    if directory.exists() and any(directory.iterdir()):
        raise JobsError("launch evidence directory is not create-once empty")
    directory.mkdir(parents=True, exist_ok=True)
    preview_proof = _live_preview_proof(request, client.preview(request))
    preview_receipt = _seal(
        {
            "schema": "cyber_miles96_live_server_preview_v1",
            "plan_sha256": "sha256:" + mechanics.digest(plan),
            "request_sha256": "sha256:" + digest(request),
            "manifest_sha256": "sha256:" + preview_proof["manifest_sha256"],
            "root_failure_alerts": "off",
            "backoff_limit": 0,
            "shutdown_after_job_finishes": True,
            "nodes": preview_proof["nodes"],
            "gpus": preview_proof["gpus"],
        }
    )
    _write_json_once(paths["plan"], plan)
    _write_json_once(paths["request"], request)
    _write_json_once(paths["preview"], preview_receipt)
    observer = start_observer(directory)
    try:
        guard = fresh_absence_checks(
            plan,
            request,
            client,
            directory,
            runner=runner,
            jobs_root=jobs_root,
            output_absence_receipt=output_absence_receipt,
            wandb_exists=wandb_exists,
            now=now,
        )
    except Exception:
        observer.terminate()
        raise
    try:
        # Revalidate the exact source-bound SFS receipt immediately before the
        # durable create intent.  A runtime create-once guard remains required
        # because no read-side absence check can make the later POST atomic.
        final_sfs_proof = _prove_sfs_output_absent(
            plan,
            request,
            jobs_root=jobs_root,
            receipt=output_absence_receipt,
            observed_at=now(),
        )
        capacity = capacity_gate(plan, request, reader=capacity_reader, now=now)
    except Exception:
        observer.terminate()
        raise
    try:
        final_gate = final_prepost_checks(
            plan,
            request,
            client,
            directory,
            runner=runner,
            now=now,
        )
    except Exception:
        observer.terminate()
        raise
    descriptor = os.open(paths["journal"], os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        stream.write(
            json.dumps(
                {
                    "state": "POST_INTENT_DO_NOT_RETRY",
                    "request_sha256": digest(request),
                    "preview": preview_receipt,
                    "fresh_absence": guard,
                    "final_prepost_gate": final_gate,
                    "final_sfs_output_absence_receipt_sha256": final_sfs_proof["sha256"],
                    "capacity_gate": capacity,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )
        stream.flush()
        os.fsync(stream.fileno())
    try:
        # Keep this separate from the duplicate read so a coordinator that
        # exits during that final read cannot leave an unwatched allocation.
        validate_armed_observer(plan, request, directory, now=now())
    except Exception:
        observer.terminate()
        raise
    response = client.request("POST", "/v1/runs", json=request)
    # Persist the creator-returned identity before interpreting it.  If the
    # response is malformed, the detached coordinator still has the only
    # available reconciliation evidence and this caller still must not POST
    # again.
    observed = safe_status(response)
    if observed.get("run_dir") is None:
        observed["run_dir"] = request["run_dir"]
    _append_journal(paths["journal"], {"state": "POST_RESPONSE", **observed})
    if not re.fullmatch(re.escape(request["name"]) + r"-[a-f0-9]{8}", response.get("name", "")):
        raise JobsError("ambiguous submit response; reconcile journal, never repeat POST")
    if response.get("run_dir") not in (None, request["run_dir"]):
        raise JobsError("submitted output differs; reconcile ownership immediately")
    return observed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--observe", action="store_true")
    mode.add_argument("--submit", action="store_true")
    mode.add_argument("--sfs-output-job-create", action="store_true")
    mode.add_argument("--sfs-output-job-collect", action="store_true")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--request", required=True)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--receipt")
    parser.add_argument("--output-absence-receipt")
    parser.add_argument("--output")
    parser.add_argument("--attempt", type=int, default=1)
    args = parser.parse_args()
    plan = _validate_plan(json.loads(Path(args.plan).read_text()))
    request = json.loads(Path(args.request).read_text())
    directory = Path(args.directory)
    if args.observe:
        if args.receipt or args.output_absence_receipt or args.output:
            parser.error("--observe does not accept receipt or output arguments")
        run_cleanup_coordinator(plan, request, directory)
        return
    receipt = json.loads(Path(args.receipt).read_text()) if args.receipt else None
    if request != _expected_request(plan, request, receipt):
        parser.error("request differs from the exact immutable mechanics plan")
    if args.sfs_output_job_create or args.sfs_output_job_collect:
        if args.output_absence_receipt:
            parser.error("SFS output-check modes do not accept --output-absence-receipt")
        if not 1 <= args.attempt <= 99:
            parser.error("--attempt must be between 1 and 99")
        from cyber_post_train.direct_submit import (
            Kubectl,
            collect_sfs_output_check,
            create_sfs_output_check_once,
        )

        kubectl = Kubectl(plan["execution"]["kubernetes_context"])
        directory.mkdir(parents=True, exist_ok=True)
        if args.sfs_output_job_create:
            if args.output:
                parser.error("--sfs-output-job-create does not accept --output")
            result = create_sfs_output_check_once(
                plan=plan,
                request=request,
                attempt=args.attempt,
                kubectl=kubectl,
                journal=directory / f"SFS_OUTPUT_CHECK_A{args.attempt:02d}.jsonl",
            )
        else:
            if not args.output:
                parser.error("--sfs-output-job-collect requires --output")
            result = collect_sfs_output_check(
                plan=plan,
                request=request,
                attempt=args.attempt,
                kubectl=kubectl,
            )
            _write_json_once(Path(args.output), result)
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        return
    if args.output:
        parser.error("--submit does not accept --output")
    output_absence_receipt = (
        json.loads(Path(args.output_absence_receipt).read_text())
        if args.output_absence_receipt
        else None
    )
    token = os.environ.get("FLEET_API_KEY")
    if not token:
        parser.error("--submit requires FLEET_API_KEY in the process environment")
    with Jobs(token, base_url=plan["execution"]["jobs_api_base_url"]) as client:
        result = submit_once(
            plan,
            request,
            client,
            directory,
            receipt=receipt,
            output_absence_receipt=output_absence_receipt,
        )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
