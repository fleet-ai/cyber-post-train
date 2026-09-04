"""Run one create-once OpenCode pass@4 plan with per-attempt acceptance receipts."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import queue
import threading
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

import httpx

from evals.fleet import self_hosted

PLAN_SCHEMA = "fleet-selfhosted-opencode-pass4-plan-v1"
PARALLEL_PLAN_SCHEMA = "fleet-selfhosted-opencode-pass4-parallel-plan-v1"
DRAIN_REQUEST_SCHEMA = "fleet-selfhosted-opencode-pass4-drain-request-v1"
DRAINED_SCHEMA = "fleet-selfhosted-opencode-pass4-drained-v1"
ALLOWED_DRAIN_REASONS = {
    "capacity_rebalance",
    "operator_maintenance",
    "protocol_change",
}
_CLAIM_GATE = threading.Lock()


class DrainRequested(RuntimeError):
    """A UID-bound operator request closed the attempt-claim gate."""

    def __init__(self, request: dict[str, Any]) -> None:
        super().__init__("campaign drain requested")
        self.request = request


def _digest_without(value: dict[str, Any], field: str) -> str:
    return self_hosted.digest_without(value, field)


def _load_drain_request(plan: dict[str, Any], root: Path) -> dict[str, Any] | None:
    path = root / "DRAIN-REQUEST.json"
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("campaign drain request is not a safe regular file")
    request = json.loads(path.read_text())
    if (
        request.get("schema_version") != DRAIN_REQUEST_SCHEMA
        or request.get("request_sha256") != _digest_without(request, "request_sha256")
        or request.get("plan_sha256") != plan.get("plan_sha256")
        or request.get("campaign_id") != plan.get("campaign_id")
        or request.get("reason") not in ALLOWED_DRAIN_REASONS
    ):
        raise RuntimeError("campaign drain request binding or digest drifted")
    job_uid = os.environ.get("JOB_UID")
    pod_uid = os.environ.get("POD_UID")
    if not job_uid or not pod_uid:
        raise RuntimeError("campaign drain request cannot be verified without Job/Pod UIDs")
    if request.get("target_job_uid") != job_uid or request.get("target_pod_uid") != pod_uid:
        raise RuntimeError("campaign drain request targets a different Job or Pod")
    return request


def _claim_or_raise_drained(
    plan: dict[str, Any], root: Path, claim_path: Path, claim: dict[str, Any]
) -> None:
    """Serialize a claim against an external drain writer on the shared filesystem."""
    gate_path = root / ".attempt-claim-gate.lock"
    with _CLAIM_GATE, gate_path.open("a+b") as gate:
        fcntl.flock(gate.fileno(), fcntl.LOCK_EX)
        request = _load_drain_request(plan, root)
        if request is not None:
            raise DrainRequested(request)
        self_hosted.write_json_once(claim_path, claim)


def _drain_completed_claims(plan: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    claims: dict[str, dict[str, Any]] = {}
    for path in sorted((root / "claims").glob("*.json")):
        claim = json.loads(path.read_text())
        run_id = claim.get("run_id")
        if (
            claim.get("claim_sha256") != _digest_without(claim, "claim_sha256")
            or claim.get("plan_sha256") != plan["plan_sha256"]
            or not isinstance(run_id, str)
            or run_id in claims
        ):
            raise RuntimeError("campaign drain claim evidence drifted")
        claims[run_id] = claim
    accepted = _accepted_attempts(root)
    if set(claims) != set(accepted):
        raise RuntimeError("campaign drain reached terminalization with unresolved claims")
    return [
        {
            "run_id": run_id,
            "claim_sha256": claims[run_id]["claim_sha256"],
            "acceptance_receipt_sha256": accepted[run_id]["receipt_sha256"],
        }
        for run_id in sorted(claims)
    ]


def _write_drained_unlocked(
    plan: dict[str, Any], root: Path, request: dict[str, Any]
) -> dict[str, Any]:
    completed_claims = _drain_completed_claims(plan, root)
    receipt = {
        "schema_version": DRAINED_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "campaign_id": plan["campaign_id"],
        "drain_request_sha256": request["request_sha256"],
        "job_uid": os.environ["JOB_UID"],
        "pod_uid": os.environ["POD_UID"],
        "accepted_new_sessions": len(_accepted_attempts(root)),
        "completed_claim_receipts": completed_claims,
        "completed_claim_receipts_sha256": self_hosted.sha256(
            self_hosted.canonical_json(completed_claims)
        ),
        "attempt_claim_gate_closed": True,
        "in_flight_attempts_completed_before_exit": True,
        "scores_included": False,
        "prompts_or_traces_included": False,
    }
    receipt["receipt_sha256"] = _digest_without(receipt, "receipt_sha256")
    self_hosted.write_json_once(root / "DRAINED.json", receipt)
    return receipt


def _write_drained(plan: dict[str, Any], root: Path, request: dict[str, Any]) -> dict[str, Any]:
    gate_path = root / ".attempt-claim-gate.lock"
    with _CLAIM_GATE, gate_path.open("a+b") as gate:
        fcntl.flock(gate.fileno(), fcntl.LOCK_EX)
        current = _load_drain_request(plan, root)
        if current is None or current.get("request_sha256") != request.get("request_sha256"):
            raise RuntimeError("campaign drain request changed before terminalization")
        if (root / "ACCEPTED.json").exists():
            raise RuntimeError("campaign was accepted before drain terminalization")
        return _write_drained_unlocked(plan, root, current)


def _write_accepted_or_drained(
    plan: dict[str, Any], root: Path, final: dict[str, Any]
) -> dict[str, Any]:
    gate_path = root / ".attempt-claim-gate.lock"
    with _CLAIM_GATE, gate_path.open("a+b") as gate:
        fcntl.flock(gate.fileno(), fcntl.LOCK_EX)
        request = _load_drain_request(plan, root)
        if request is not None:
            return _write_drained_unlocked(plan, root, request)
        self_hosted.write_json_once(root / "ACCEPTED.json", final)
        return final


def validate_plan(plan: dict[str, Any]) -> None:
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("unsupported full-plan schema")
    if plan.get("plan_sha256") != _digest_without(plan, "plan_sha256"):
        raise ValueError("full-plan digest mismatch")
    task_count = int(plan.get("task_count") or 0)
    if task_count not in {50, 100} or plan.get("pass_k") != 4:
        raise ValueError("full-plan task/pass@k shape drifted")
    if len(plan.get("tasks") or []) != task_count:
        raise ValueError("full-plan task count drifted")
    if len(plan.get("attempts") or []) != task_count * 4 - 1:
        raise ValueError("full-plan attempt count drifted")
    if (plan.get("execution") or {}).get("max_concurrent") != 1:
        raise ValueError("full-plan concurrency must remain one per model")
    if (plan.get("execution") or {}).get("required_task_tools") != [
        "bash",
        "submit_report",
    ]:
        raise ValueError("full-plan tool contract drifted")


def validate_parallel_plan(plan: dict[str, Any]) -> None:
    if plan.get("schema_version") != PARALLEL_PLAN_SCHEMA:
        raise ValueError("unsupported parallel full-plan schema")
    if plan.get("plan_sha256") != _digest_without(plan, "plan_sha256"):
        raise ValueError("parallel full-plan digest mismatch")
    task_count = int(plan.get("task_count") or 0)
    if task_count not in {50, 100} or plan.get("pass_k") != 4:
        raise ValueError("parallel full-plan task/pass@k shape drifted")
    tasks = plan.get("tasks") or []
    if len(tasks) != task_count or [row.get("rank") for row in tasks] != list(
        range(1, task_count + 1)
    ):
        raise ValueError("parallel full-plan task count drifted")
    attempts = plan.get("attempts") or []
    prior = plan.get("prior_accepted") or []
    if len(attempts) != int(plan.get("new_session_count") or -1):
        raise ValueError("parallel full-plan new-session count drifted")
    if int(plan.get("total_session_count") or 0) != task_count * 4:
        raise ValueError("parallel full-plan total-session count drifted")
    if 1 + len(prior) + len(attempts) != task_count * 4:
        raise ValueError("parallel full-plan pass@4 accounting drifted")
    run_ids = [row.get("run_id") for row in attempts]
    networks = [row.get("network") for row in attempts]
    if len(run_ids) != len(set(run_ids)) or len(networks) != len(set(networks)):
        raise ValueError("parallel full-plan attempt identities are not unique")
    credited = plan.get("credited_smoke") or {}
    cells = [
        (int(credited.get("rank") or 0), int(credited.get("attempt") or 0)),
        *[(int(row.get("rank") or 0), int(row.get("attempt") or 0)) for row in prior],
        *[(int(row.get("rank") or 0), int(row.get("attempt") or 0)) for row in attempts],
    ]
    expected_cells = {
        (rank, attempt) for rank in range(1, task_count + 1) for attempt in range(1, 5)
    }
    if len(cells) != len(set(cells)) or set(cells) != expected_cells:
        raise ValueError("parallel full-plan Cartesian pass@4 cells drifted")
    if [int(row.get("ordinal") or 0) for row in attempts] != list(range(1, len(attempts) + 1)):
        raise ValueError("parallel full-plan attempt ordinals drifted")
    prior_sessions = [row.get("session_id") for row in prior]
    prior_verifiers = [row.get("verifier_execution_id") for row in prior]
    if len(prior_sessions) != len(set(prior_sessions)) or len(prior_verifiers) != len(
        set(prior_verifiers)
    ):
        raise ValueError("parallel full-plan prior evidence identities are not unique")
    schedule = (plan.get("execution") or {}).get("concurrency_schedule") or {}
    if schedule != {
        "max_concurrent": 2,
        "infrastructure_valid_sessions_before_continuation": 4,
        "decision_inputs": [
            "accepted_outcome",
            "completed_session_ingest",
            "valid_verifier_execution_id",
            "complete_cleanup",
        ],
        "score_blind": True,
        "capacity_basis": "shared_endpoint_compute_hot_first_step_1_to_2",
    }:
        raise ValueError("parallel full-plan concurrency schedule drifted")
    if (plan.get("execution") or {}).get("required_task_tools") != [
        "bash",
        "submit_report",
    ]:
        raise ValueError("parallel full-plan tool contract drifted")


def _accepted_attempts(root: Path) -> dict[str, dict[str, Any]]:
    accepted: dict[str, dict[str, Any]] = {}
    attempts_root = root / "attempts"
    if not attempts_root.exists():
        return accepted
    for receipt_path in attempts_root.glob("*/ACCEPTED.json"):
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("receipt_sha256") != _digest_without(receipt, "receipt_sha256"):
            raise RuntimeError("attempt acceptance digest mismatch")
        run_id = receipt.get("run_id")
        if not isinstance(run_id, str) or run_id in accepted:
            raise RuntimeError("attempt acceptance identity is invalid or duplicated")
        accepted[run_id] = receipt
    return accepted


def _inventory_for_task(
    client: httpx.Client,
    task_key: str,
    persisted_model: str,
) -> list[dict[str, Any]]:
    return [
        row
        for row in self_hosted._task_sessions(client, task_key)
        if row.get("model") == persisted_model
    ]


def _attempt_config(
    plan: dict[str, Any], task_row: dict[str, Any], attempt: dict[str, Any]
) -> dict[str, Any]:
    config = {
        "schema_version": "fleet-selfhosted-opencode-pass4-attempt-v1",
        "run_id": attempt["run_id"],
        "campaign_id": plan["campaign_id"],
        "source_job_id": plan["source_job_id"],
        "task": task_row["task"],
        "environment": task_row["environment"],
        "verifier": task_row["verifier"],
        "authority": plan["authority"],
        "model": plan["model"],
        "harness": plan["harness"],
        "execution": {
            "pass_k": 1,
            "planned_full_pass_k": 4,
            "max_concurrent": 1,
            "network": attempt["network"],
            "training_data_eligible": plan["execution"]["training_data_eligible"],
            "required_task_tools": plan["execution"]["required_task_tools"],
            "required_task_tool_catalog_sha256": plan["execution"][
                "required_task_tool_catalog_sha256"
            ],
        },
    }
    config["config_sha256"] = _digest_without(config, "config_sha256")
    return config


def _accept_attempt(
    out_dir: Path,
    config: dict[str, Any],
    *,
    rank: int | None = None,
    attempt_number: int | None = None,
    claim_sha256: str | None = None,
) -> dict[str, Any]:
    result = json.loads((out_dir / "result.json").read_text())
    cleanup = json.loads((out_dir / "cleanup.json").read_text())
    ingest = json.loads((out_dir / "session-ingest.json").read_text())
    if any(
        (
            result.get("run_id") != config["run_id"],
            result.get("agent_exit_code") != 0,
            result.get("agent_termination") != "completed",
            result.get("session_ingest_status") != "completed",
            ingest.get("status") != "completed",
            cleanup
            != {
                "instance_created": True,
                "instance_closed": True,
                "containers_removed": True,
            },
        )
    ):
        raise RuntimeError("attempt is not an accepted valid model outcome")
    receipt = {
        "schema_version": "fleet-selfhosted-opencode-pass4-attempt-accepted-v1",
        "accepted": True,
        "run_id": config["run_id"],
        "config_sha256": config["config_sha256"],
        "session_id": result["session_id"],
        "verifier_execution_id": result["verifier_execution_id"],
        "cleanup_completed": True,
        "score_persisted_privately": True,
        "scores_included": False,
        "prompts_or_traces_included": False,
    }
    if rank is not None:
        receipt["rank"] = rank
        receipt["task_key"] = config["task"]["key"]
        receipt["task_version_id"] = config["task"]["version_id"]
    if attempt_number is not None:
        receipt["attempt"] = attempt_number
    if claim_sha256 is not None:
        receipt["claim_sha256"] = claim_sha256
    receipt["receipt_sha256"] = _digest_without(receipt, "receipt_sha256")
    self_hosted.write_json_once(out_dir / "ACCEPTED.json", receipt)
    return receipt


def _client(key: str) -> httpx.Client:
    return httpx.Client(
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        timeout=1800,
    )


def _parallel_allowed_sessions(
    plan: dict[str, Any], task_row: dict[str, Any], root: Path
) -> set[str]:
    rank = int(task_row["rank"])
    allowed = set()
    if int(plan["credited_smoke"]["rank"]) == rank:
        allowed.add(plan["credited_smoke"]["session_id"])
    allowed.update(row["session_id"] for row in plan["prior_accepted"] if int(row["rank"]) == rank)
    attempt_by_run = {row["run_id"]: row for row in plan["attempts"]}
    for receipt in _accepted_attempts(root).values():
        attempt = attempt_by_run.get(receipt["run_id"])
        if attempt is None:
            raise RuntimeError("accepted attempt is absent from the parallel plan")
        if int(attempt["rank"]) == rank:
            allowed.add(receipt["session_id"])
    return allowed


def _validate_prior_accepted(
    client: httpx.Client, plan: dict[str, Any], task_by_rank: dict[int, dict[str, Any]]
) -> None:
    persisted_model = self_hosted.persisted_session_model_identity(plan)
    by_rank: dict[int, list[dict[str, Any]]] = {}
    smoke = plan["credited_smoke"]
    by_rank.setdefault(int(smoke["rank"]), []).append(smoke)
    for row in plan["prior_accepted"]:
        by_rank.setdefault(int(row["rank"]), []).append(row)
    for rank, credited in by_rank.items():
        task_row = task_by_rank[rank]
        observed = {
            row.get("session_id"): row
            for row in _inventory_for_task(
                client,
                task_row["task"]["key"],
                persisted_model,
            )
        }
        for credit in credited:
            session = observed.get(credit["session_id"])
            verifier = (session or {}).get("verifier_execution") or {}
            if (
                session is None
                or session.get("status") != "completed"
                or verifier.get("id") != credit["verifier_execution_id"]
            ):
                raise RuntimeError("prior accepted session is not authoritative")


def _attempt_claim(
    plan: dict[str, Any], config: dict[str, Any], item: dict[str, Any]
) -> dict[str, Any]:
    claim = {
        "schema_version": "fleet-selfhosted-opencode-pass4-attempt-claim-v1",
        "plan_sha256": plan["plan_sha256"],
        "campaign_id": plan["campaign_id"],
        "run_id": item["run_id"],
        "network": item["network"],
        "rank": int(item["rank"]),
        "attempt": int(item["attempt"]),
        "task_key": config["task"]["key"],
        "task_version_id": config["task"]["version_id"],
        "config_sha256": config["config_sha256"],
        "model_revision": config["model"]["revision"],
        "session_model": config["model"]["session_model"],
        "harness_version": config["harness"]["version"],
        "harness_asset_sha256": config["harness"]["release_asset_sha256"],
        "verifier_id": config["verifier"]["id"],
        "verifier_version_id": config["verifier"]["version_id"],
    }
    claim["claim_sha256"] = _digest_without(claim, "claim_sha256")
    return claim


def _run_parallel_attempt(
    plan: dict[str, Any],
    task_row: dict[str, Any],
    item: dict[str, Any],
    root: Path,
    proxy_script: Path,
    key: str,
) -> dict[str, Any]:
    persisted_model = self_hosted.persisted_session_model_identity(plan)
    with _client(key) as client:
        observed = _inventory_for_task(
            client,
            task_row["task"]["key"],
            persisted_model,
        )
    allowed = _parallel_allowed_sessions(plan, task_row, root)
    unknown = {
        row.get("session_id")
        for row in observed
        if isinstance(row.get("session_id"), str) and row.get("session_id") not in allowed
    }
    if unknown:
        raise RuntimeError("unexpected equivalent-treatment session appeared; halting")
    config = _attempt_config(plan, task_row, item)
    out_dir = root / "attempts" / item["run_id"]
    if out_dir.exists():
        raise RuntimeError("parallel planned attempt output already exists; refusing duplicate")
    claim = _attempt_claim(plan, config, item)
    claims_dir = root / "claims"
    claims_dir.mkdir(exist_ok=True)
    _claim_or_raise_drained(
        plan,
        root,
        claims_dir / f"{item['run_id']}.json",
        claim,
    )
    self_hosted.run(config, out_dir, proxy_script)
    receipt = _accept_attempt(
        out_dir,
        config,
        rank=int(item["rank"]),
        attempt_number=int(item["attempt"]),
        claim_sha256=claim["claim_sha256"],
    )
    progress = {
        "schema_version": "fleet-selfhosted-opencode-pass4-parallel-progress-v1",
        "plan_sha256": plan["plan_sha256"],
        "run_id": item["run_id"],
        "rank": int(item["rank"]),
        "attempt": int(item["attempt"]),
        "accepted_new_sessions": len(_accepted_attempts(root)),
        "planned_new_sessions": plan["new_session_count"],
        "scores_included": False,
    }
    progress["receipt_sha256"] = _digest_without(progress, "receipt_sha256")
    progress_dir = root / "progress"
    progress_dir.mkdir(exist_ok=True)
    self_hosted.write_json_once(progress_dir / f"{int(item['ordinal']):04d}.json", progress)
    return receipt


def _run_attempt_wave(
    plan: dict[str, Any],
    task_by_rank: dict[int, dict[str, Any]],
    items: list[dict[str, Any]],
    root: Path,
    proxy_script: Path,
    key: str,
    max_workers: int,
) -> dict[str, Any] | None:
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        remaining = iter(items)
        active: set[Future[dict[str, Any]]] = set()

        def submit_next() -> bool:
            try:
                item = next(remaining)
            except StopIteration:
                return False
            active.add(
                pool.submit(
                    _run_parallel_attempt,
                    plan,
                    task_by_rank[int(item["rank"])],
                    item,
                    root,
                    proxy_script,
                    key,
                )
            )
            return True

        for _ in range(min(max_workers, len(items))):
            submit_next()
        failure: Exception | None = None
        drain_request: dict[str, Any] | None = None
        while active:
            done, active = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                try:
                    future.result()
                except DrainRequested as exc:
                    drain_request = exc.request
                except Exception as exc:
                    failure = exc
            if failure is None and drain_request is None:
                for _ in range(len(done)):
                    if not submit_next():
                        break
        if failure is not None:
            raise failure
        return drain_request


def _run_task_groups(
    plan: dict[str, Any],
    task_by_rank: dict[int, dict[str, Any]],
    groups: list[list[dict[str, Any]]],
    root: Path,
    proxy_script: Path,
    key: str,
    max_workers: int,
) -> dict[str, Any] | None:
    work: queue.Queue[list[dict[str, Any]]] = queue.Queue()
    for group in groups:
        work.put(group)
    stop = threading.Event()
    failures: list[Exception] = []
    drain_requests: list[dict[str, Any]] = []
    failure_lock = threading.Lock()

    def worker() -> None:
        while not stop.is_set():
            try:
                group = work.get_nowait()
            except queue.Empty:
                return
            try:
                for item in group:
                    if stop.is_set():
                        return
                    _run_parallel_attempt(
                        plan,
                        task_by_rank[int(item["rank"])],
                        item,
                        root,
                        proxy_script,
                        key,
                    )
            except DrainRequested as exc:
                with failure_lock:
                    drain_requests.append(exc.request)
                stop.set()
                return
            except Exception as exc:
                with failure_lock:
                    failures.append(exc)
                stop.set()
                return
            finally:
                work.task_done()

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(worker) for _ in range(max_workers)]
        for future in futures:
            future.result()
    if failures:
        raise failures[0]
    if drain_requests:
        first = drain_requests[0]
        if any(row.get("request_sha256") != first.get("request_sha256") for row in drain_requests):
            raise RuntimeError("conflicting campaign drain requests observed")
        return first
    return None


def run_parallel_plan(plan: dict[str, Any], root: Path, proxy_script: Path) -> dict[str, Any]:
    validate_parallel_plan(plan)
    root.mkdir(parents=True, exist_ok=False)
    root.chmod(0o700)
    (root / "attempts").mkdir(mode=0o700)
    (root / "claims").mkdir(mode=0o700)
    self_hosted.write_json_once(root / "PLAN.json", plan)
    key = os.environ.get("FLEET_API_KEY")
    if not key:
        raise RuntimeError("FLEET_API_KEY is required")
    task_by_rank = {int(row["rank"]): row for row in plan["tasks"]}
    with _client(key) as client:
        account = self_hosted._request(client, "GET", "/v1/account")
        if (
            account.get("team_name") != "fleet"
            or account.get("team_id") != self_hosted.FLEET_TEAM_ID
        ):
            raise RuntimeError("FLEET_API_KEY is not scoped to the Fleet team")
        _validate_prior_accepted(client, plan, task_by_rank)

    by_rank: dict[int, list[dict[str, Any]]] = {}
    for item in plan["attempts"]:
        by_rank.setdefault(int(item["rank"]), []).append(item)
    for group in by_rank.values():
        group.sort(key=lambda row: (int(row["attempt"]), int(row["ordinal"])))

    schedule = plan["execution"]["concurrency_schedule"]
    wave: list[dict[str, Any]] = []
    for rank in sorted(by_rank):
        if len(wave) == int(schedule["infrastructure_valid_sessions_before_continuation"]):
            break
        wave.append(by_rank[rank].pop(0))
    drain_request = _run_attempt_wave(
        plan,
        task_by_rank,
        wave,
        root,
        proxy_script,
        key,
        int(schedule["max_concurrent"]),
    )
    if drain_request is not None:
        return _write_drained(plan, root, drain_request)
    if len(_accepted_attempts(root)) != len(wave):
        raise RuntimeError("score-blind concurrency ramp gate did not reconcile")
    groups = [group for _, group in sorted(by_rank.items()) if group]
    drain_request = _run_task_groups(
        plan,
        task_by_rank,
        groups,
        root,
        proxy_script,
        key,
        int(schedule["max_concurrent"]),
    )
    if drain_request is not None:
        return _write_drained(plan, root, drain_request)
    accepted_new = len(_accepted_attempts(root))
    final = {
        "schema_version": "fleet-selfhosted-opencode-pass4-parallel-accepted-v1",
        "accepted": True,
        "plan_sha256": plan["plan_sha256"],
        "credited_smoke_sessions": 1,
        "prior_accepted_sessions": len(plan["prior_accepted"]),
        "accepted_new_sessions": accepted_new,
        "total_sessions": 1 + len(plan["prior_accepted"]) + accepted_new,
        "scores_included": False,
        "prompts_or_traces_included": False,
    }
    if accepted_new != plan["new_session_count"]:
        raise RuntimeError("parallel terminal accepted-session count drifted")
    if final["total_sessions"] != plan["total_session_count"]:
        raise RuntimeError("parallel terminal total-session count drifted")
    accepted_cells = {
        (int(row["rank"]), int(row["attempt"])) for row in _accepted_attempts(root).values()
    }
    prior_cells = {(int(row["rank"]), int(row["attempt"])) for row in plan["prior_accepted"]}
    credited_cell = {(int(plan["credited_smoke"]["rank"]), int(plan["credited_smoke"]["attempt"]))}
    if accepted_cells | prior_cells | credited_cell != {
        (rank, attempt) for rank in range(1, int(plan["task_count"]) + 1) for attempt in range(1, 5)
    }:
        raise RuntimeError("parallel terminal Cartesian pass@4 evidence drifted")
    final["receipt_sha256"] = _digest_without(final, "receipt_sha256")
    return _write_accepted_or_drained(plan, root, final)


def run_plan(plan: dict[str, Any], root: Path, proxy_script: Path) -> dict[str, Any]:
    validate_plan(plan)
    root.mkdir(parents=True, exist_ok=False)
    root.chmod(0o700)
    (root / "attempts").mkdir(mode=0o700)
    (root / "claims").mkdir(mode=0o700)
    self_hosted.write_json_once(root / "PLAN.json", plan)
    key = os.environ.get("FLEET_API_KEY")
    if not key:
        raise RuntimeError("FLEET_API_KEY is required")
    task_by_rank = {int(row["rank"]): row for row in plan["tasks"]}
    persisted_model = self_hosted.persisted_session_model_identity(plan)
    with httpx.Client(
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        timeout=1800,
    ) as client:
        account = self_hosted._request(client, "GET", "/v1/account")
        if (
            account.get("team_name") != "fleet"
            or account.get("team_id") != self_hosted.FLEET_TEAM_ID
        ):
            raise RuntimeError("FLEET_API_KEY is not scoped to the Fleet team")
        for item in plan["attempts"]:
            task_row = task_by_rank[int(item["rank"])]
            accepted = _accepted_attempts(root)
            allowed = set(task_row["baseline_session_ids"])
            allowed.update(
                receipt["session_id"]
                for receipt in accepted.values()
                if receipt["run_id"] != item["run_id"]
            )
            observed = _inventory_for_task(
                client,
                task_row["task"]["key"],
                persisted_model,
            )
            unknown = {
                row.get("session_id")
                for row in observed
                if isinstance(row.get("session_id"), str) and row.get("session_id") not in allowed
            }
            if unknown:
                raise RuntimeError("unexpected equivalent-treatment session appeared; halting")
            config = _attempt_config(plan, task_row, item)
            out_dir = root / "attempts" / item["run_id"]
            if out_dir.exists():
                raise RuntimeError("planned attempt output already exists; refusing duplicate")
            claim = _attempt_claim(plan, config, item)
            try:
                _claim_or_raise_drained(
                    plan,
                    root,
                    root / "claims" / f"{item['run_id']}.json",
                    claim,
                )
            except DrainRequested as exc:
                return _write_drained(plan, root, exc.request)
            self_hosted.run(config, out_dir, proxy_script)
            _accept_attempt(
                out_dir,
                config,
                rank=int(item["rank"]),
                attempt_number=int(item["attempt"]),
                claim_sha256=claim["claim_sha256"],
            )
            progress = {
                "schema_version": "fleet-selfhosted-opencode-pass4-progress-v1",
                "plan_sha256": plan["plan_sha256"],
                "accepted_new_sessions": len(_accepted_attempts(root)),
                "planned_new_sessions": plan["new_session_count"],
                "last_run_id": item["run_id"],
                "scores_included": False,
            }
            progress["receipt_sha256"] = _digest_without(progress, "receipt_sha256")
            progress_dir = root / "progress"
            progress_dir.mkdir(exist_ok=True)
            self_hosted.write_json_once(progress_dir / f"{item['ordinal']:04d}.json", progress)
    final = {
        "schema_version": "fleet-selfhosted-opencode-pass4-accepted-v1",
        "accepted": True,
        "plan_sha256": plan["plan_sha256"],
        "credited_smoke_sessions": 1,
        "accepted_new_sessions": len(_accepted_attempts(root)),
        "total_sessions": plan["total_session_count"],
        "scores_included": False,
        "prompts_or_traces_included": False,
    }
    if final["accepted_new_sessions"] != plan["new_session_count"]:
        raise RuntimeError("terminal accepted-session count drifted")
    final["receipt_sha256"] = _digest_without(final, "receipt_sha256")
    return _write_accepted_or_drained(plan, root, final)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--proxy-script", type=Path, required=True)
    parser.add_argument("--parallel", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if args.parallel:
        result = run_parallel_plan(plan, args.out_dir, args.proxy_script)
    else:
        result = run_plan(plan, args.out_dir, args.proxy_script)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
