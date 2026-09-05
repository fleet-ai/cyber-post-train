"""Execute one released exact-pass@4 bulk controller.

The runner is deliberately restart-safe.  A cell-wide execution claim is
created with O_EXCL before the first model call.  A restart skips every cell
that already has a claim and proceeds to the untouched tail.  Infrastructure
incompleteness quarantines only the claimed cell; it never silently retries it
and never truncates later statistical cells.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import subprocess
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from evals.fleet import endpoint_lease, self_hosted
from evals.fleet import exact_pass4_bulk_v3 as bulk

CLAIM_SCHEMA = "fleet-exact-pass4-bulk-cell-execution-claim-v3"
TERMINAL_SCHEMA = "fleet-exact-pass4-bulk-controller-terminal-v3"
QUARANTINE_SCHEMA = "fleet-exact-pass4-bulk-cell-quarantine-v3"
ABORT_SCHEMA = "fleet-exact-pass4-bulk-controller-abort-v3"
UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")
ISO_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z$")


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _seal(value: dict[str, Any]) -> dict[str, Any]:
    return {
        **value,
        "receipt_sha256": self_hosted.digest_without(value, "receipt_sha256"),
    }


def _write_once(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise RuntimeError("receipt target is not a regular file")
        with os.fdopen(fd, "wb") as handle:
            fd = -1
            handle.write(self_hosted.canonical_json(value) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if fd >= 0:
            os.close(fd)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def claim_cell(
    plan: dict[str, Any], item: dict[str, Any], *, claim_root: Path, job_uid: str, pod_uid: str
) -> dict[str, Any] | None:
    """Atomically claim a statistical execution, or return None if already claimed."""
    if UUID_RE.fullmatch(job_uid) is None or UUID_RE.fullmatch(pod_uid) is None:
        raise RuntimeError("bulk cell claim requires downward API UIDs")
    if claim_root.exists() and (claim_root.is_symlink() or not claim_root.is_dir()):
        raise RuntimeError("global claim root is unsafe")
    claim_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    name = item["execution_id"].removeprefix("sha256:") + ".json"
    receipt = _seal(
        {
            "schema_version": CLAIM_SCHEMA,
            "plan_sha256": plan["plan_sha256"],
            "controller": plan["controller"],
            "cell_id": item["cell_id"],
            "execution_id": item["execution_id"],
            "execution_generation": item["execution_generation"],
            "run_id": item["run_id"],
            "selection_rank": item["selection_rank"],
            "attempt": item["attempt"],
            "job_uid": job_uid,
            "pod_uid": pod_uid,
            "claimed_at_utc": _now(),
            "immutable": True,
            "automatic_retry": False,
            "model_call_started_when_claim_written": False,
            "scores_included": False,
            "prompts_or_traces_included": False,
        }
    )
    try:
        _write_once(claim_root / name, receipt)
    except FileExistsError:
        return None
    return receipt


def _validate_preserved_claim(
    plan: dict[str, Any], item: dict[str, Any], claim_root: Path
) -> dict[str, Any]:
    path = claim_root / (item["execution_id"].removeprefix("sha256:") + ".json")
    claim = bulk.load(path)
    if (
        claim.get("receipt_sha256") != self_hosted.digest_without(claim, "receipt_sha256")
        or claim.get("schema_version") != CLAIM_SCHEMA
        or claim.get("plan_sha256") != plan["plan_sha256"]
        or claim.get("controller") != plan["controller"]
        or claim.get("cell_id") != item["cell_id"]
        or claim.get("execution_id") != item["execution_id"]
        or claim.get("execution_generation") != item["execution_generation"]
        or claim.get("run_id") != item["run_id"]
        or claim.get("selection_rank") != item["selection_rank"]
        or claim.get("attempt") != item["attempt"]
        or UUID_RE.fullmatch(str(claim.get("job_uid"))) is None
        or UUID_RE.fullmatch(str(claim.get("pod_uid"))) is None
        or claim.get("immutable") is not True
        or claim.get("automatic_retry") is not False
        or claim.get("model_call_started_when_claim_written") is not False
        or ISO_UTC_RE.fullmatch(str(claim.get("claimed_at_utc"))) is None
        or claim.get("scores_included") is not False
        or claim.get("prompts_or_traces_included") is not False
    ):
        raise RuntimeError("existing global execution claim drifted")
    return claim


def _task_for_item(plan: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    matches = [row for row in plan["tasks"] if row["rank"] == item["selection_rank"]]
    if len(matches) != 1:
        raise RuntimeError("bulk runtime task binding is not unique")
    return matches[0]


def _attempt_config(
    plan: dict[str, Any], task: dict[str, Any], item: dict[str, Any]
) -> dict[str, Any]:
    config = {
        "schema_version": "fleet-hosted-opencode-task-boundary-attempt-v1",
        "run_id": item["run_id"],
        "campaign_id": plan["campaign_id"],
        "source_job_id": plan["source_job_id"],
        "task": task["task"],
        "environment": task["environment"],
        "verifier": task["verifier"],
        "authority": plan["authority"],
        "model": plan["model"],
        "harness": plan["harness"],
        "execution": {
            "pass_k": 1,
            "planned_full_pass_k": 4,
            "max_concurrent": 1,
            "network": item["network"],
            "cell_id": item["cell_id"],
            "execution_id": item["execution_id"],
            "training_data_eligible": False,
            "required_task_tools": ["bash", "submit_report"],
            "required_task_tool_catalog_sha256": plan["execution"][
                "required_task_tool_catalog_sha256"
            ],
        },
    }
    config["config_sha256"] = self_hosted.digest_without(config, "config_sha256")
    return config


@contextmanager
def _client(key: str) -> Iterator[httpx.Client]:
    with httpx.Client(
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        timeout=1800,
    ) as client:
        yield client


def _assert_run_absent(config: dict[str, Any], key: str) -> None:
    with _client(key) as client:
        rows = self_hosted._task_sessions(client, config["task"]["key"])
    expected_metadata = {
        "run_id": config["run_id"],
        **self_hosted.session_execution_metadata(config),
    }
    collisions = [
        row
        for row in rows
        if any(
            (row.get("metadata") or {}).get(field) == value
            for field, value in expected_metadata.items()
        )
    ]
    if collisions:
        raise RuntimeError("bulk planned run identity already exists")


def _fresh_route_check(plan: dict[str, Any], key: str) -> None:
    """Bind local context controls and reject any contradictory live route metadata."""
    model = plan.get("model") or {}
    harness = plan.get("harness") or {}
    treatment = plan.get("treatment") or {}
    served_id = model.get("served_id")
    settings = self_hosted.opencode_settings(plan)
    rendered_limit = (
        settings.get("provider", {})
        .get("fleet-cluster", {})
        .get("models", {})
        .get(served_id, {})
        .get("limit")
    )
    if (
        model.get("endpoint_origin") != "https://inference.flt.build"
        or harness.get("context_window_size") != 262144
        or harness.get("compaction_headroom_tokens") != 20000
        or treatment.get("context_window_size") != 262144
        or treatment.get("compaction_headroom_tokens") != 20000
        or rendered_limit
        != {"context": 262144, "output": 32768, "input": 262144 - 32768}
    ):
        raise RuntimeError("fresh hosted route context binding drifted")
    with _client(key) as client:
        account = self_hosted._request(client, "GET", "/v1/account")
        response = client.get(model["endpoint_origin"].rstrip("/") + "/v1/models")
        response.raise_for_status()
        roster = response.json()
    rows = roster.get("data") if isinstance(roster, dict) else None
    selected = (
        [row for row in rows if row.get("id") == served_id]
        if isinstance(rows, list) and all(isinstance(row, dict) for row in rows)
        else []
    )
    if (
        account.get("team_name") != "fleet"
        or account.get("team_id") != self_hosted.FLEET_TEAM_ID
        or len(selected) != 1
    ):
        raise RuntimeError("fresh hosted route check failed")
    context_values = [
        selected[0].get(field)
        for field in (
            "context_length",
            "context_window",
            "max_context_length",
            "max_model_len",
            "max_sequence_length",
        )
        if selected[0].get(field) is not None
    ]
    # The hosted roster currently omits context metadata. The immutable local
    # harness limit above remains authoritative; if the roster later exposes a
    # limit, every exposed value must agree rather than silently changing it.
    if context_values and any(
        type(value) is not int or value != 262144 for value in context_values
    ):
        raise RuntimeError("fresh hosted route context drifted")


def _runtime_release_gate_check(plan: dict[str, Any]) -> None:
    """Recheck immutable sanitized SFS gate identities inside every runtime Pod."""
    bindings = (
        ("BULK_INVENTORY_GATE_PATH", "BULK_INVENTORY_GATE_SHA256"),
        ("BULK_RECONCILIATION_GATE_PATH", "BULK_RECONCILIATION_GATE_SHA256"),
        ("BULK_QWEN_CANARY_GATE_PATH", "BULK_QWEN_CANARY_GATE_SHA256"),
        ("BULK_GLM_CANARY_GATE_PATH", "BULK_GLM_CANARY_GATE_SHA256"),
    )
    receipts: dict[str, dict[str, Any]] = {}
    for path_name, sha_name in bindings:
        path_value, expected_sha = os.environ.get(path_name), os.environ.get(sha_name)
        if not path_value or not expected_sha or bulk.SHA256_RE.fullmatch(expected_sha) is None:
            raise RuntimeError("bulk runtime release gate environment is incomplete")
        path = Path(path_value)
        if not path.is_absolute() or not path.is_relative_to(Path("/mnt/sfs/jobs")):
            raise RuntimeError("bulk runtime release gate path is unsafe")
        receipt = bulk.load(path)
        if receipt.get("receipt_sha256") != expected_sha or receipt.get(
            "receipt_sha256"
        ) != self_hosted.digest_without(receipt, "receipt_sha256"):
            raise RuntimeError("bulk runtime release gate receipt drifted")
        receipts[path_name] = receipt

    inventory = receipts["BULK_INVENTORY_GATE_PATH"]
    bulk.validate_inventory_gate(inventory, Path(plan["repo_root"]))
    reconciliation = receipts["BULK_RECONCILIATION_GATE_PATH"]
    all_plans = bulk.validate_all(Path(plan["repo_root"]))
    execution_ids = sorted(
        row["execution_id"] for built in all_plans.values() for row in built["attempts"]
    )
    expected_execution_sha = self_hosted.sha256(self_hosted.canonical_json(execution_ids))
    remaining_cells = sorted(
        (
            {
                "model": built["model"]["served_id"],
                "controller": controller,
                "selection_rank": row["selection_rank"],
                "attempt": row["attempt"],
                "task_version_id": row["task_version_id"],
                "cell_id": row["cell_id"],
                "execution_id": row["execution_id"],
                "run_id": row["run_id"],
            }
            for controller, built in all_plans.items()
            for row in built["attempts"]
        ),
        key=lambda row: (row["model"], row["selection_rank"], row["attempt"]),
    )
    remaining_cells_sha = self_hosted.sha256(self_hosted.canonical_json(remaining_cells))
    package_commit = os.environ.get("BULK_PACKAGE_COMMIT")
    qwen = receipts["BULK_QWEN_CANARY_GATE_PATH"]
    glm = receipts["BULK_GLM_CANARY_GATE_PATH"]
    refs = reconciliation.get("generation7_gate_receipts") or {}
    if (
        bulk.COMMIT_RE.fullmatch(str(package_commit)) is None
        or reconciliation.get("schema_version") != bulk.RECONCILIATION_GATE_SCHEMA
        or reconciliation.get("status") != "CLEAR"
        or reconciliation.get("observer_package_commit") != package_commit
        or reconciliation.get("planned_execution_count") != 798
        or reconciliation.get("planned_execution_ids_sha256") != expected_execution_sha
        or reconciliation.get("remaining_cells") != remaining_cells
        or reconciliation.get("remaining_cells_sha256") != remaining_cells_sha
        or reconciliation.get("checked_immediately_before_release") is not True
        or reconciliation.get("observer_job_succeeded") is not True
        or reconciliation.get("observer_pod_restarts") != 0
        or any(
            reconciliation.get(field) != 0
            for field in (
                "fleet_api_collisions",
                "kubernetes_job_or_pod_collisions",
                "sfs_output_collisions",
                "global_claim_collisions",
                "active_or_accepted_cell_collisions",
                "mutation_calls",
            )
        )
        or inventory.get("receipt_sha256")
        != (reconciliation.get("exact100_inventory") or {}).get("receipt_sha256")
        or (refs.get("qwen3.8-27b") or {}).get("receipt_sha256") != qwen.get("receipt_sha256")
        or (refs.get("glm-5.3") or {}).get("receipt_sha256") != glm.get("receipt_sha256")
        or qwen.get("schema_version") != bulk.CANARY_GATE_SCHEMA
        or qwen.get("status") != "ACCEPTED"
        or qwen.get("model") != "qwen3.8-27b"
        or glm.get("schema_version") != bulk.CANARY_GATE_SCHEMA
        or glm.get("status") != "ACCEPTED"
        or glm.get("model") != "glm-5.3"
    ):
        raise RuntimeError("bulk runtime release gate chain drifted")


def _classify_result(
    out_dir: Path,
    config: dict[str, Any],
    item: dict[str, Any],
    claim: dict[str, Any],
    key: str,
) -> dict[str, Any]:
    result = bulk.load(out_dir / "result.json")
    reward = bulk.load(out_dir / "reward-result.json")
    ingest = bulk.load(out_dir / "session-ingest.json")
    cleanup = bulk.load(out_dir / "cleanup.json")
    session_id = result.get("session_id")
    verifier_id = result.get("verifier_execution_id")
    try:
        uuid.UUID(str(session_id))
        uuid.UUID(str(verifier_id))
    except ValueError as exc:
        raise RuntimeError("bulk attempt lacks authoritative UUID evidence") from exc
    if not all(
        (
            result.get("run_id") == config["run_id"],
            result.get("task_key") == config["task"]["key"],
            result.get("task_version_id") == config["task"]["version_id"],
            result.get("agent_termination") == "completed",
            type(result.get("agent_exit_code")) is int,
            result.get("session_ingest_status") == "completed",
            ingest.get("status") == "completed",
            ingest.get("session_id") == session_id,
            reward.get("task_key") == config["task"]["key"],
            reward.get("task_version_id") == config["task"]["version_id"],
            reward.get("verifier_execution_id") == verifier_id,
            reward.get("instance_id") == result.get("instance_id"),
            isinstance(reward.get("reward"), (int, float)),
            not isinstance(reward.get("reward"), bool),
            cleanup
            == {"instance_created": True, "instance_closed": True, "containers_removed": True},
        )
    ):
        raise RuntimeError("bulk attempt is infrastructure-incomplete")
    authoritative: list[dict[str, Any]] = []
    for poll in range(12):
        with _client(key) as client:
            rows = self_hosted._task_sessions(client, config["task"]["key"])
        authoritative = [row for row in rows if row.get("session_id") == session_id]
        if authoritative:
            break
        if poll < 11:
            time.sleep(5)
    if not authoritative:
        raise RuntimeError("bulk attempt lacks authoritative scored session")
    if len(authoritative) != 1:
        raise RuntimeError("bulk authoritative scored session identity is duplicated")
    row = authoritative[0]
    verifier = row.get("verifier_execution") or {}
    projected_version = row.get("eval_task_version_id") or row.get("task_version_id")
    metadata = row.get("metadata") or {}
    projected_run = metadata.get("run_id")
    if (
        row.get("status") != "completed"
        or row.get("model") != self_hosted.persisted_session_model_identity(config)
        or verifier.get("id") != verifier_id
        or projected_version != config["task"]["version_id"]
        or projected_run != config["run_id"]
        or metadata.get("execution_id") != item["execution_id"]
        or metadata.get("cell_id") != item["cell_id"]
    ):
        raise RuntimeError("bulk authoritative scored session binding drifted")
    return _seal(
        {
            "schema_version": "fleet-exact-pass4-bulk-cell-accepted-v3",
            "accepted": True,
            "credited": True,
            "retry_allowed": False,
            "controller": plan_controller(config["campaign_id"]),
            "cell_id": item["cell_id"],
            "execution_id": item["execution_id"],
            "run_id": config["run_id"],
            "selection_rank": item["selection_rank"],
            "attempt": item["attempt"],
            "task_key": config["task"]["key"],
            "task_version_id": config["task"]["version_id"],
            "session_id": session_id,
            "verifier_execution_id": verifier_id,
            "agent_exit_code": result.get("agent_exit_code"),
            "agent_process_exit_success": result.get("agent_exit_code") == 0,
            "config_sha256": config["config_sha256"],
            "claim_sha256": claim["receipt_sha256"],
            "cleanup_completed": True,
            "session_ingest_completed": True,
            "scores_included": False,
            "prompts_or_traces_included": False,
        }
    )


def plan_controller(campaign_id: str) -> str:
    for controller, expected in bulk.CONTROLLERS.items():
        if campaign_id == expected["job_name"]:
            return controller
    raise RuntimeError("bulk runtime campaign identity drifted")


def _infrastructure_evidence(out_dir: Path, error: Exception) -> bool:
    try:
        ingest = out_dir / "session-ingest.json"
        if ingest.is_file():
            value = bulk.load(ingest)
            if value.get("status") == "failed":
                status = value.get("http_status")
                transient_http = (
                    value.get("error_code") == "fleet_http_error"
                    and type(status) is int
                    and (status == 429 or status >= 500)
                )
                transient_transport = value.get("error_type") in {
                    "ConnectError",
                    "ConnectTimeout",
                    "NetworkError",
                    "PoolTimeout",
                    "ReadError",
                    "ReadTimeout",
                    "RemoteProtocolError",
                    "TimeoutException",
                    "WriteError",
                    "WriteTimeout",
                }
                # Binding-drift/MissingSessionId/SessionIdChanged receipts are
                # protocol defects, not infrastructure attrition.  Fail the
                # controller so the remaining cells are not consumed under a
                # broken ingest contract.
                return transient_http or transient_transport
        result = out_dir / "result.json"
        if result.is_file() and bulk.load(result).get("session_ingest_status") == "failed":
            return False
        # The model runner can finish all local mutations successfully while the
        # final, read-only authoritative session lookup is delayed or returns a
        # transient transport error.  That is infrastructure-incomplete.  A
        # local binding mismatch is *not*: it is a systemic protocol defect and
        # must stop the controller instead of being mislabeled and repeated.
        if all(
            (out_dir / name).is_file()
            for name in (
                "result.json",
                "reward-result.json",
                "session-ingest.json",
                "cleanup.json",
            )
        ) and (
            str(error) == "bulk attempt lacks authoritative scored session"
            or (
                isinstance(error, self_hosted.FleetRequestError)
                and (error.status_code == 429 or error.status_code >= 500)
            )
            or isinstance(error, httpx.HTTPError)
        ):
            return True
        failure = out_dir / "failure.json"
        cleanup = out_dir / "cleanup.json"
        if failure.is_file() and cleanup.is_file():
            failure_row = bulk.load(failure)
            cleanup_row = bulk.load(cleanup)
            cleanup_safe = cleanup_row.get("containers_removed") is True and (
                cleanup_row.get("instance_created") is not True
                or cleanup_row.get("instance_closed") is True
            )
            # A generic RuntimeError can represent immutable binding, tool,
            # scoring-contract, or trace-format drift.  Never call those
            # infrastructure and burn the untouched tail.  Continue only for
            # explicitly transient transport/process failures with complete
            # cleanup, or the fixed-proxy readiness timeout before a model
            # request could be made.
            transient_fleet = isinstance(error, self_hosted.FleetRequestError) and (
                error.status_code == 429 or error.status_code >= 500
            )
            transient_process = isinstance(
                error,
                (httpx.HTTPError, subprocess.CalledProcessError, subprocess.TimeoutExpired),
            )
            proxy_readiness = isinstance(error, RuntimeError) and (
                str(error).startswith("fixed proxy ")
                and str(error).endswith(" did not become ready")
            )
            return (
                failure_row.get("error_type") == type(error).__name__
                and cleanup_safe
                and (transient_fleet or transient_process or proxy_readiness)
            )
        return False
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def _validate_existing_receipt(
    receipt: dict[str, Any], item: dict[str, Any], controller: str, kind: str
) -> None:
    if receipt.get("receipt_sha256") != self_hosted.digest_without(receipt, "receipt_sha256"):
        raise RuntimeError(f"existing bulk {kind} receipt digest drifted")
    common = {
        "controller": controller,
        "cell_id": item["cell_id"],
        "execution_id": item["execution_id"],
        "run_id": item["run_id"],
        "selection_rank": item["selection_rank"],
        "attempt": item["attempt"],
    }
    if any(receipt.get(field) != value for field, value in common.items()):
        raise RuntimeError(f"existing bulk {kind} receipt identity drifted")
    if kind == "accepted":
        if (
            receipt.get("schema_version") != "fleet-exact-pass4-bulk-cell-accepted-v3"
            or receipt.get("accepted") is not True
            or receipt.get("credited") is not True
            or receipt.get("retry_allowed") is not False
            or UUID_RE.fullmatch(str(receipt.get("session_id"))) is None
            or UUID_RE.fullmatch(str(receipt.get("verifier_execution_id"))) is None
            or type(receipt.get("agent_exit_code")) is not int
            or receipt.get("agent_process_exit_success")
            is not (receipt.get("agent_exit_code") == 0)
            or bulk.SHA256_RE.fullmatch(str(receipt.get("config_sha256"))) is None
            or bulk.SHA256_RE.fullmatch(str(receipt.get("claim_sha256"))) is None
            or receipt.get("cleanup_completed") is not True
            or receipt.get("session_ingest_completed") is not True
            or receipt.get("task_key") != item["task_key"]
            or receipt.get("task_version_id") != item["task_version_id"]
            or receipt.get("scores_included") is not False
            or receipt.get("prompts_or_traces_included") is not False
        ):
            raise RuntimeError("existing bulk accepted receipt is not authoritative")
    elif kind == "quarantine":
        if (
            receipt.get("schema_version") != QUARANTINE_SCHEMA
            or receipt.get("retry_allowed") is not False
            or receipt.get("classification") != "infrastructure_incomplete"
            or not isinstance(receipt.get("error_type"), str)
            or not receipt["error_type"]
            or receipt.get("only_exact_cell_quarantined") is not True
            or receipt.get("tail_continues") is not True
            or bulk.SHA256_RE.fullmatch(str(receipt.get("claim_sha256"))) is None
            or receipt.get("scores_included") is not False
            or receipt.get("prompts_or_traces_included") is not False
        ):
            raise RuntimeError("existing bulk quarantine receipt is invalid")
    elif kind == "preserved":
        if (
            receipt.get("schema_version") != "fleet-exact-pass4-bulk-existing-claim-preserved-v3"
            or receipt.get("retry_allowed") is not False
            or receipt.get("reason")
            not in {
                "planned_run_identity_already_exists",
                "global_execution_claim_already_exists",
            }
            or (
                receipt.get("reason") == "global_execution_claim_already_exists"
                and bulk.SHA256_RE.fullmatch(str(receipt.get("existing_claim_sha256"))) is None
            )
            or (
                receipt.get("reason") == "planned_run_identity_already_exists"
                and "existing_claim_sha256" in receipt
            )
        ):
            raise RuntimeError("existing bulk preserved receipt is invalid")
    else:  # pragma: no cover - internal caller fixes the vocabulary
        raise AssertionError("unsupported existing receipt kind")


def _prepare_output_root(
    out: Path, plan: dict[str, Any], claim_root: Path
) -> tuple[set[str], set[str], set[str], dict[str, Any] | None]:
    """Create a fresh root or validate an interrupted same-root continuation."""
    existing_terminal: dict[str, Any] | None = None
    if out.exists() or out.is_symlink():
        if out.is_symlink() or not out.is_dir():
            raise RuntimeError("bulk output root is unsafe")
        plan_path = out / "PLAN.json"
        if not plan_path.exists():
            if any(out.iterdir()):
                raise RuntimeError("bulk output root is initialized without a plan")
            _write_once(plan_path, plan)
        existing_plan = bulk.load(plan_path)
        if existing_plan != plan:
            raise RuntimeError("bulk resumed output plan drifted")
        abort_path = out / "ABORT.json"
        if abort_path.exists() or abort_path.is_symlink():
            abort = bulk.load(abort_path)
            item_by_execution = {item["execution_id"]: item for item in plan["attempts"]}
            item = item_by_execution.get(abort.get("execution_id"))
            if (
                abort.get("receipt_sha256") != self_hosted.digest_without(abort, "receipt_sha256")
                or abort.get("schema_version") != ABORT_SCHEMA
                or abort.get("plan_sha256") != plan["plan_sha256"]
                or abort.get("controller") != plan_controller(plan["campaign_id"])
                or item is None
                or abort.get("cell_id") != item["cell_id"]
                or abort.get("run_id") != item["run_id"]
                or abort.get("claim_sha256")
                != _validate_preserved_claim(plan, item, claim_root)["receipt_sha256"]
                or not isinstance(abort.get("error_type"), str)
                or not abort["error_type"]
                or abort.get("systemic_failure_requires_review") is not True
                or abort.get("automatic_tail_continuation") is not False
                or ISO_UTC_RE.fullmatch(str(abort.get("aborted_at_utc"))) is None
                or abort.get("scores_included") is not False
                or abort.get("prompts_or_traces_included") is not False
            ):
                raise RuntimeError("existing bulk controller abort receipt drifted")
            raise RuntimeError("bulk controller has a durable systemic abort receipt")
        terminal_path = out / "TERMINAL.json"
        if terminal_path.exists() or terminal_path.is_symlink():
            terminal = bulk.load(terminal_path)
            if (
                terminal.get("receipt_sha256")
                != self_hosted.digest_without(terminal, "receipt_sha256")
                or terminal.get("schema_version") != TERMINAL_SCHEMA
                or terminal.get("plan_sha256") != plan["plan_sha256"]
                or terminal.get("accounted_cells") != len(plan["attempts"])
                or terminal.get("planned_cells") != len(plan["attempts"])
            ):
                raise RuntimeError("existing bulk terminal receipt drifted")
            existing_terminal = terminal
    else:
        out.mkdir(mode=0o700, parents=True, exist_ok=False)
        _write_once(out / "PLAN.json", plan)

    for name in ("attempts", "accepted", "quarantine", "preserved-claims"):
        directory = out / name
        if not directory.exists():
            directory.mkdir(mode=0o700)
        if directory.is_symlink() or not directory.is_dir():
            raise RuntimeError(f"bulk resumed {name} directory is unsafe")

    item_by_run = {item["run_id"]: item for item in plan["attempts"]}
    states: dict[str, set[str]] = {
        "accepted": set(),
        "quarantine": set(),
        "preserved": set(),
    }
    for directory_name, kind in (
        ("accepted", "accepted"),
        ("quarantine", "quarantine"),
        ("preserved-claims", "preserved"),
    ):
        for path in (out / directory_name).iterdir():
            if path.is_symlink() or not path.is_file() or path.suffix != ".json":
                raise RuntimeError(f"bulk resumed {kind} directory has an unsafe entry")
            run_id = path.stem
            item = item_by_run.get(run_id)
            if item is None:
                raise RuntimeError(f"bulk resumed {kind} receipt is outside the plan")
            receipt = bulk.load(path)
            _validate_existing_receipt(receipt, item, plan_controller(plan["campaign_id"]), kind)
            if kind in {"accepted", "quarantine"} or receipt.get("reason") == (
                "global_execution_claim_already_exists"
            ):
                claim = _validate_preserved_claim(plan, item, claim_root)
                bound_claim = (
                    receipt.get("existing_claim_sha256")
                    if kind == "preserved"
                    else receipt.get("claim_sha256")
                )
                if bound_claim != claim["receipt_sha256"]:
                    raise RuntimeError(f"existing bulk {kind} claim binding drifted")
            states[kind].add(run_id)
    seen: set[str] = set()
    for kind, run_ids in states.items():
        overlap = seen & run_ids
        if overlap:
            raise RuntimeError(f"bulk resumed cell has conflicting {kind} state")
        seen |= run_ids
    if existing_terminal is not None and (
        existing_terminal.get("controller") != plan_controller(plan["campaign_id"])
        or existing_terminal.get("accepted_cells") != len(states["accepted"])
        or existing_terminal.get("quarantined_cells") != len(states["quarantine"])
        or existing_terminal.get("preserved_nonrepeatable_cells") != len(states["preserved"])
        or existing_terminal.get("accounted_cells")
        != sum(len(run_ids) for run_ids in states.values())
        or seen != set(item_by_run)
        or existing_terminal.get("endpoint_lease_released") is not True
        or existing_terminal.get("scientific_complete")
        is not (len(states["accepted"]) == len(plan["attempts"]))
        or existing_terminal.get("successor_reconciliation_required")
        is not bool(states["quarantine"] or states["preserved"])
        or existing_terminal.get("status") not in {"COMPLETE", "COMPLETE_WITH_QUARANTINE"}
        or existing_terminal.get("status")
        != (
            "COMPLETE_WITH_QUARANTINE"
            if states["quarantine"] or states["preserved"]
            else "COMPLETE"
        )
        or UUID_RE.fullmatch(str(existing_terminal.get("job_uid"))) is None
        or UUID_RE.fullmatch(str(existing_terminal.get("pod_uid"))) is None
        or ISO_UTC_RE.fullmatch(str(existing_terminal.get("terminal_at_utc"))) is None
        or existing_terminal.get("scores_included") is not False
        or existing_terminal.get("prompts_or_traces_included") is not False
    ):
        raise RuntimeError("existing bulk terminal state does not match durable receipts")
    return (
        states["accepted"],
        states["quarantine"],
        states["preserved"],
        existing_terminal,
    )


def run_controller(
    plan: dict[str, Any],
    *,
    out: Path,
    proxy: Path,
    claim_root: Path | None = None,
    model_runner: Callable[[dict[str, Any], Path, Path], dict[str, Any]] = self_hosted.run,
    classifier: Callable[
        [Path, dict[str, Any], dict[str, Any], dict[str, Any], str], dict[str, Any]
    ] = _classify_result,
    check_run_absent: Callable[[dict[str, Any], str], None] = _assert_run_absent,
    route_check: Callable[[dict[str, Any], str], None] = _fresh_route_check,
    runtime_gate_check: Callable[[dict[str, Any]], None] = _runtime_release_gate_check,
) -> dict[str, Any]:
    """Run untouched cells sequentially, preserving progress across infrastructure attrition."""
    controller = plan_controller(plan["campaign_id"])
    rebuilt = bulk.build_runtime_plan(
        controller, plan["inventory_receipt"], Path(plan["repo_root"])
    )
    if plan != rebuilt:
        raise ValueError("bulk executable runtime plan drifted")
    key = os.environ.get("FLEET_API_KEY")
    job_uid, pod_uid = os.environ.get("JOB_UID", ""), os.environ.get("POD_UID", "")
    if not key:
        raise RuntimeError("FLEET_API_KEY is required")
    runtime_gate_check(plan)
    root = claim_root or Path(plan["execution"]["claim_root"])
    accepted_runs, quarantined_runs, preserved_runs, existing_terminal = _prepare_output_root(
        out, plan, root
    )
    if existing_terminal is not None:
        return existing_terminal
    lease = plan["execution"]["endpoint_lease"]
    accepted = len(accepted_runs)
    quarantined = len(quarantined_runs)
    preserved = len(preserved_runs)
    with endpoint_lease.acquire_endpoint_lease(
        lease_root=Path(lease["lease_root"]),
        endpoint_key=lease["endpoint_key"],
        maximum_streams=lease["maximum_streams"],
    ):
        for item in plan["attempts"]:
            if item["run_id"] in accepted_runs | quarantined_runs | preserved_runs:
                continue
            task = _task_for_item(plan, item)
            config = _attempt_config(plan, task, item)
            try:
                check_run_absent(config, key)
            except RuntimeError as exc:
                if str(exc) != "bulk planned run identity already exists":
                    raise
                preserved += 1
                _write_once(
                    out / "preserved-claims" / f"{item['run_id']}.json",
                    _seal(
                        {
                            "schema_version": "fleet-exact-pass4-bulk-existing-claim-preserved-v3",
                            "controller": controller,
                            "cell_id": item["cell_id"],
                            "execution_id": item["execution_id"],
                            "run_id": item["run_id"],
                            "selection_rank": item["selection_rank"],
                            "attempt": item["attempt"],
                            "reason": "planned_run_identity_already_exists",
                            "retry_allowed": False,
                            "model_call_started_when_claim_written": None,
                        }
                    ),
                )
                continue
            # The public endpoint is shared mutable infrastructure.  Re-prove
            # the served model id and context immediately before each atomic
            # claim/model boundary, not merely once when a days-long Job starts.
            route_check(plan, key)
            claim = claim_cell(plan, item, claim_root=root, job_uid=job_uid, pod_uid=pod_uid)
            if claim is None:
                existing_claim = _validate_preserved_claim(plan, item, root)
                preserved += 1
                _write_once(
                    out / "preserved-claims" / f"{item['run_id']}.json",
                    _seal(
                        {
                            "schema_version": "fleet-exact-pass4-bulk-existing-claim-preserved-v3",
                            "controller": controller,
                            "cell_id": item["cell_id"],
                            "execution_id": item["execution_id"],
                            "run_id": item["run_id"],
                            "selection_rank": item["selection_rank"],
                            "attempt": item["attempt"],
                            "reason": "global_execution_claim_already_exists",
                            "retry_allowed": False,
                            "model_call_started_when_claim_written": existing_claim.get(
                                "model_call_started_when_claim_written"
                            ),
                            "existing_claim_sha256": existing_claim["receipt_sha256"],
                        }
                    ),
                )
                continue
            attempt_out = out / "attempts" / item["run_id"]
            try:
                model_runner(config, attempt_out, proxy)
                receipt = classifier(attempt_out, config, item, claim, key)
                _write_once(out / "accepted" / f"{item['run_id']}.json", receipt)
                accepted += 1
            except Exception as exc:
                if not _infrastructure_evidence(attempt_out, exc):
                    _write_once(
                        out / "ABORT.json",
                        _seal(
                            {
                                "schema_version": ABORT_SCHEMA,
                                "plan_sha256": plan["plan_sha256"],
                                "controller": controller,
                                "cell_id": item["cell_id"],
                                "execution_id": item["execution_id"],
                                "run_id": item["run_id"],
                                "claim_sha256": claim["receipt_sha256"],
                                "error_type": type(exc).__name__,
                                "systemic_failure_requires_review": True,
                                "automatic_tail_continuation": False,
                                "aborted_at_utc": _now(),
                                "scores_included": False,
                                "prompts_or_traces_included": False,
                            }
                        ),
                    )
                    raise
                quarantined += 1
                _write_once(
                    out / "quarantine" / f"{item['run_id']}.json",
                    _seal(
                        {
                            "schema_version": QUARANTINE_SCHEMA,
                            "controller": controller,
                            "cell_id": item["cell_id"],
                            "execution_id": item["execution_id"],
                            "run_id": item["run_id"],
                            "selection_rank": item["selection_rank"],
                            "attempt": item["attempt"],
                            "claim_sha256": claim["receipt_sha256"],
                            "classification": "infrastructure_incomplete",
                            "error_type": type(exc).__name__,
                            "retry_allowed": False,
                            "only_exact_cell_quarantined": True,
                            "tail_continues": True,
                            "scores_included": False,
                            "prompts_or_traces_included": False,
                        }
                    ),
                )
    terminal = _seal(
        {
            "schema_version": TERMINAL_SCHEMA,
            "status": "COMPLETE_WITH_QUARANTINE" if quarantined or preserved else "COMPLETE",
            "controller": controller,
            "plan_sha256": plan["plan_sha256"],
            "planned_cells": len(plan["attempts"]),
            "accepted_cells": accepted,
            "quarantined_cells": quarantined,
            "preserved_nonrepeatable_cells": preserved,
            "accounted_cells": accepted + quarantined + preserved,
            "scientific_complete": accepted == len(plan["attempts"]),
            "successor_reconciliation_required": bool(quarantined or preserved),
            "endpoint_lease_released": True,
            "terminal_at_utc": _now(),
            "job_uid": job_uid,
            "pod_uid": pod_uid,
            "scores_included": False,
            "prompts_or_traces_included": False,
        }
    )
    _write_once(out / "TERMINAL.json", terminal)
    return terminal


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", nargs="?")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--proxy", type=Path, required=True)
    args = parser.parse_args()
    run_controller(bulk.load(args.plan), out=args.out, proxy=args.proxy)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
