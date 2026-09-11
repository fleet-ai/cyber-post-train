"""Run exact Fleet cyber rollout cells from the atomic campaign ledger.

The current distributed path uses PostgreSQL for cross-host coordination. Every
stochastic execution is fenced first by both an atomic ledger claim and the campaign's
global immutable execution-claim file. SQLite remains available only for historical
replay and local tests where one process owns the database.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import suppress
from pathlib import Path
from typing import Any

import httpx

from evals.fleet import exact_pass4_crypto as crypto
from evals.fleet import exact_pass4_universe as universe
from evals.fleet import opencode_self_hosted as self_hosted
from evals.fleet import rollout_campaign, rollout_ledger, rollout_postgres

AUTHORITY = {
    "multi_app_aggregation_mode": "fractional",
    "provisioning_route_template": (
        "/v1/rollout-rewards/{task_key}/versions/{task_version_id}/instances"
    ),
    "required_cyber_contract": {
        "evidence_schema": "1.0.0",
        "submission_protocol": "2.0.0",
        "verifier_contract": "3.0.0",
    },
    "scoring_mode": "partial",
    "scoring_route_template": "/v1/rollout-rewards/{task_key}/versions/{task_version_id}",
}
ORCHESTRATOR = "https://orchestrator.fleetai.com"
CLAIM_SCHEMA = "fleet-statistical-cell-execution-claim-v1"
ACCEPTED_SCHEMA = "fleet-rollout-ledger-cell-accepted-v1"
TERMINAL_SCHEMA = "fleet-rollout-ledger-controller-terminal-v1"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path.name} must contain a JSON object")
    return value


def _safe_write_once(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value = {**value, "receipt_sha256": crypto.digest_without(value, "receipt_sha256")}
    self_hosted.write_json_once(path, value)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _record_local_result(
    *,
    database: Path | str,
    ledger: Any = rollout_ledger,
    config: dict[str, Any],
    ledger_cell: dict[str, Any],
    result: dict[str, Any],
    out_dir: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Index private rollout evidence before the Fleet catalog acceptance gate."""

    artifact_directory = out_dir.resolve().relative_to(output_root.resolve()).as_posix()
    manifest = _load(out_dir / "trace-manifest.json")
    ingest = _load(out_dir / "session-ingest.json")
    trace_relative = Path(str(manifest.get("canonical_trace") or ""))
    if not trace_relative.parts or trace_relative.is_absolute() or ".." in trace_relative.parts:
        raise RuntimeError("canonical trace manifest path is unsafe")
    trace = (out_dir / trace_relative).resolve()
    trace.relative_to(out_dir.resolve())
    if not trace.is_file():
        raise RuntimeError("canonical trace file is missing")
    trace_sha256 = _file_sha256(trace)
    if trace_sha256 != manifest.get("canonical_trace_sha256"):
        raise RuntimeError("canonical trace digest differs from its manifest")

    files = {
        "result": out_dir / "result.json",
        "reward": out_dir / "reward-result.json",
        "session_ingest": out_dir / "session-ingest.json",
        "cleanup": out_dir / "cleanup.json",
    }
    if any(not path.is_file() for path in files.values()):
        raise RuntimeError("complete local rollout evidence is missing")
    execution = config["execution"]
    record = {
        "execution_id": execution["execution_id"],
        "execution_generation": execution["execution_generation"],
        "run_id": config["run_id"],
        "session_id": result.get("session_id"),
        "verifier_execution_id": result["verifier_execution_id"],
        "score": result["score"],
        "config_sha256": config["config_sha256"],
        "artifact_directory": artifact_directory,
        "trace_path": trace_relative.as_posix(),
        "trace_sha256": trace_sha256,
        "result_path": files["result"].name,
        "result_sha256": _file_sha256(files["result"]),
        "reward_path": files["reward"].name,
        "reward_sha256": _file_sha256(files["reward"]),
        "session_ingest_path": files["session_ingest"].name,
        "session_ingest_sha256": _file_sha256(files["session_ingest"]),
        "cleanup_path": files["cleanup"].name,
        "cleanup_sha256": _file_sha256(files["cleanup"]),
        "session_ingest_status": ingest["status"],
        "agent_exit_code": result["agent_exit_code"],
        "agent_termination": result["agent_termination"],
        "elapsed_seconds": result["elapsed_seconds"],
    }
    return ledger.record_local_result(
        database,
        cell_id=ledger_cell["cell_id"],
        worker_id=ledger_cell["worker_id"],
        claim_id=ledger_cell["claim_id"],
        record=record,
    )


def _selection_index(selection: dict[str, Any]) -> dict[str, dict[str, Any]]:
    tasks = selection.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise RuntimeError("frozen selection must contain tasks")
    result = {str(task["task_version_id"]): task for task in tasks}
    if len(result) != len(tasks):
        raise RuntimeError("frozen selection task versions are not unique")
    return result


def _universe_index(
    campaign: dict[str, Any], repo_root: Path
) -> dict[tuple[str, str, int], dict[str, Any]]:
    expanded = universe.build_universe(campaign, repo_root)
    return {
        (str(row["model"]), str(row["task_version_id"]), int(row["attempt"])): row
        for row in expanded["cells"]
    }


def _task_binding(
    client: httpx.Client,
    cell: dict[str, Any],
    selected: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    response = self_hosted._request(  # noqa: SLF001
        client,
        "GET",
        f"/v1/tasks/{cell['task_key']}",
        params={"version_id": cell["task_version_id"]},
    )
    return self_hosted.bind_task(
        response,
        {**selected, "task_key": cell["task_key"], "task_version_id": cell["task_version_id"]},
    )


def build_config(
    campaign: dict[str, Any],
    ledger_cell: dict[str, Any],
    scientific_cell: dict[str, Any],
    selected: dict[str, Any],
    client: httpx.Client,
) -> dict[str, Any]:
    task, environment, verifier = _task_binding(client, scientific_cell, selected)
    frozen = campaign.get("task_bindings", {}).get(scientific_cell["task_version_id"])
    if frozen is not None and frozen != [task, environment, verifier]:
        raise RuntimeError("task binding changed after evaluation preflight")
    execution = scientific_cell["initial_execution"]
    treatment = campaign["treatment"]
    model = campaign["models"][ledger_cell["model_id"]]
    route = campaign.get("routes", {}).get(ledger_cell.get("serving_block"), {})
    run_suffix = execution["execution_id"].removeprefix("sha256:")[:12]
    config = {
        "schema_version": "fleet-selfhosted-opencode-ledger-cell-v1",
        "run_id": f"{campaign.get('run_prefix', 'chris-cyber-cell')}-{run_suffix}-g1",
        "campaign_id": campaign["campaign_id"],
        "source_job_id": campaign["selection"]["source_job_id"],
        "task": task,
        "environment": environment,
        "verifier": verifier,
        "authority": AUTHORITY,
        "model": {
            **model,
            "endpoint_origin": route.get("endpoint_origin", "https://inference.flt.build"),
            "served_id": ledger_cell["endpoint_model_id"],
        },
        "harness": {
            "name": treatment["harness"],
            "version": treatment["harness_version"],
            "release_asset_sha256": treatment["release_asset_sha256"],
            "provider_adapter": treatment["provider_adapter"],
            "context_management": treatment["context_management"],
            "context_window_size": treatment["context_window_size"],
            "compaction_headroom_tokens": treatment["compaction_headroom_tokens"],
            "max_output_tokens": treatment["max_output_tokens"],
            "max_model_requests": treatment["max_model_requests"],
            "timeout_seconds": treatment["timeout_seconds"],
        },
        "execution": {
            **execution,
            "network": f"cyber-cell-{run_suffix}",
            "pass_k": 1,
            "planned_full_pass_k": campaign.get("pass_k", 4),
            "max_concurrent": 1,
            "training_data_eligible": campaign.get("training_data_eligible", True),
            "required_task_tools": treatment["tools"],
            "required_task_tool_catalog_sha256": treatment["tool_catalog_sha256"],
        },
    }
    if "sampling" in campaign:
        config["sampling"] = {
            **campaign["sampling"],
            "seed": (campaign["sampling"]["seed"] + int(scientific_cell["attempt"]) - 1) % 2**31,
        }
    config["config_sha256"] = self_hosted.digest_without(config, "config_sha256")
    return config


def _claim_receipt(config: dict[str, Any], ledger_cell: dict[str, Any]) -> dict[str, Any]:
    body = {
        "schema_version": CLAIM_SCHEMA,
        "campaign_id": config["campaign_id"],
        "cell_id": config["execution"]["cell_id"],
        "execution_id": config["execution"]["execution_id"],
        "execution_generation": config["execution"]["execution_generation"],
        "ledger_cell_id": ledger_cell["cell_id"],
        "run_id": config["run_id"],
        "serving_block": ledger_cell["serving_block"],
        "model_revision": ledger_cell["model_revision"],
        "task_version_id": ledger_cell["task_version_id"],
        "attempt": ledger_cell["attempt"],
        "harness_id": ledger_cell["harness_id"],
        "model_call_started_when_claim_written": False,
        "scores_included": False,
        "prompts_or_traces_included": False,
    }
    return {**body, "receipt_sha256": crypto.digest_without(body, "receipt_sha256")}


class _Heartbeat:
    def __init__(
        self,
        database: Path | str,
        cell: dict[str, Any],
        ledger: Any = rollout_ledger,
        interval: int = 60,
    ) -> None:
        self.database = database
        self.cell = cell
        self.ledger = ledger
        self.interval = interval
        self.lease_seconds = max(300, self.interval * 4)
        self.last_success = time.monotonic()
        self.failure: str | None = None
        self.closed = False
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        delay = self.interval
        while not self.stop.wait(delay):
            try:
                self._renew()
                delay = self.interval
            except rollout_ledger.LedgerError as exc:
                # Ownership/state failures are fences, not transient outages.
                self.failure = f"heartbeat_failed:{type(exc).__name__.lower()}"
                return
            except Exception:  # noqa: BLE001
                # Never print a driver exception: it may contain connection data.
                if self.failure is not None:
                    return
                delay = min(5, self.interval)

    def _renew(self) -> None:
        started = time.monotonic()
        # Leave a full minute for a bounded database operation. Do not revive a
        # stale lease after a long outage or process suspension.
        if started - self.last_success >= self.lease_seconds - 60:
            self.failure = "heartbeat_lease_window_exhausted"
            self.check()
        self.ledger.heartbeat(
            self.database,
            cell_id=self.cell["cell_id"],
            worker_id=self.cell["worker_id"],
            claim_id=self.cell["claim_id"],
            lease_seconds=self.lease_seconds,
        )
        self.last_success = started

    def check(self) -> None:
        if self.failure is not None:
            raise RuntimeError(self.failure) from None

    def close(self) -> None:
        if self.closed:
            self.check()
            return
        self.stop.set()
        self.thread.join(timeout=10)
        if self.thread.is_alive():
            self.failure = "heartbeat_shutdown_incomplete"
        self.check()
        # Refresh once after joining, before the short terminal transaction.
        # No background renewal may race an accepted terminal state.
        self._renew()
        self.closed = True

    def __enter__(self) -> _Heartbeat:
        self._renew()
        self.thread.start()
        return self

    def __exit__(self, exc_type: object, *_args: object) -> None:
        if exc_type is None:
            self.close()
        else:
            self.stop.set()
            self.thread.join(timeout=10)


def _accepted_receipt(
    client: httpx.Client,
    config: dict[str, Any],
    ledger_cell: dict[str, Any],
    result: dict[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    cleanup = _load(out_dir / "cleanup.json")
    ingest = _load(out_dir / "session-ingest.json")
    expected_cleanup = {
        "instance_created": True,
        "instance_closed": True,
        "containers_removed": True,
    }
    if any(
        (
            result.get("run_id") != config["run_id"],
            result.get("agent_exit_code") != 0,
            result.get("agent_termination") != "completed",
            result.get("session_ingest_status") != "completed",
            ingest.get("status") != "completed",
            ingest.get("session_id") != result.get("session_id"),
            cleanup != expected_cleanup,
        )
    ):
        raise RuntimeError("cell did not produce a complete valid rollout lifecycle")
    sessions = self_hosted._task_sessions(client, config["task"]["key"])  # noqa: SLF001
    matches = [
        row
        for row in sessions
        if row.get("session_id") == result.get("session_id")
        and row.get("model") == self_hosted.persisted_session_model_identity(config)
        and (row.get("verifier_execution") or {}).get("id") == result.get("verifier_execution_id")
        and row.get("status") == "completed"
    ]
    if len(matches) != 1:
        raise RuntimeError("cell lacks one exact authoritative completed session")
    body = {
        "schema_version": ACCEPTED_SCHEMA,
        "accepted": True,
        "campaign_id": config["campaign_id"],
        "cell_id": config["execution"]["cell_id"],
        "execution_id": config["execution"]["execution_id"],
        "ledger_cell_id": ledger_cell["cell_id"],
        "run_id": config["run_id"],
        "serving_block": ledger_cell["serving_block"],
        "session_id": result["session_id"],
        "verifier_execution_id": result["verifier_execution_id"],
        "config_sha256": config["config_sha256"],
        "session_ingest_completed": True,
        "cleanup_completed": True,
        "score_persisted_privately": True,
        "scores_included": False,
        "prompts_or_traces_included": False,
    }
    return {**body, "receipt_sha256": crypto.digest_without(body, "receipt_sha256")}


def deferred_task_versions() -> list[str]:
    """Read explicit scheduling deferrals; never change a deferred ledger row."""
    value = json.loads(os.environ.get("ROLLOUT_DEFERRED_TASK_VERSIONS", "[]"))
    if (
        not isinstance(value, list)
        or any(not isinstance(v, str) or str(uuid.UUID(v)) != v for v in value)
        or len(value) != len(set(value))
    ):
        raise RuntimeError("invalid deferred task version identities")
    return value


def run_one(
    *,
    database: Path | str,
    ledger: Any = rollout_ledger,
    campaign: dict[str, Any],
    selection: dict[str, Any],
    universe_index: dict[tuple[str, str, int], dict[str, Any]],
    serving_block: str,
    worker_id: str,
    output_root: Path,
    claim_root: Path,
    proxy_script: Path,
) -> dict[str, Any]:
    cell: dict[str, Any] | None = None
    out_dir: Path | None = None
    try:
        deferred = deferred_task_versions()
        if deferred and ledger is not rollout_postgres:
            raise RuntimeError("task scheduling deferral requires PostgreSQL")
        claim_options = {"excluded_task_versions": deferred} if ledger is rollout_postgres else {}
        cell = ledger.claim(
            database,
            worker_id=worker_id,
            serving_block=serving_block,
            lease_seconds=300,
            **claim_options,
        )
        if cell is None:
            return {"serving_block": serving_block, "claimed": False, "accepted": False}
        key = (cell["model_id"], cell["task_version_id"], int(cell["attempt"]))
        scientific = universe_index.get(key)
        selected = _selection_index(selection).get(cell["task_version_id"])
        if scientific is None or selected is None:
            raise RuntimeError("ledger cell is absent from the frozen universe")
        api_key = os.environ.get("FLEET_API_KEY")
        if not api_key:
            raise RuntimeError("FLEET_API_KEY is required")
        with (
            _Heartbeat(database, cell, ledger) as heartbeat,
            httpx.Client(
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                timeout=1800,
            ) as client,
        ):
            config = build_config(campaign, cell, scientific, selected, client)
            execution_name = config["execution"]["execution_id"].removeprefix("sha256:")
            claim = _claim_receipt(config, cell)
            self_hosted.write_json_once(claim_root / f"{execution_name}.json", claim)
            out_dir = output_root / "attempts" / execution_name
            heartbeat.check()
            result = self_hosted.run(config, out_dir, proxy_script)
            heartbeat.check()
            _record_local_result(
                database=database,
                ledger=ledger,
                config=config,
                ledger_cell=cell,
                result=result,
                out_dir=out_dir,
                output_root=output_root,
            )
            accepted = _accepted_receipt(client, config, cell, result, out_dir)
            heartbeat.check()
            _safe_write_once(out_dir / "ACCEPTED.json", accepted)
            ledger.start(
                database,
                cell_id=cell["cell_id"],
                worker_id=cell["worker_id"],
                claim_id=cell["claim_id"],
                session_id=result["session_id"],
                lease_seconds=300,
            )
            ledger.mark_grading(
                database,
                cell_id=cell["cell_id"],
                worker_id=cell["worker_id"],
                claim_id=cell["claim_id"],
            )
            heartbeat.close()
            ledger.accept(
                database,
                cell_id=cell["cell_id"],
                worker_id=cell["worker_id"],
                claim_id=cell["claim_id"],
                receipt_digest=accepted["receipt_sha256"],
            )
            return {
                "serving_block": serving_block,
                "claimed": True,
                "accepted": True,
                "ledger_cell_id": cell["cell_id"],
                "receipt_sha256": accepted["receipt_sha256"],
            }
    except BaseException as exc:  # noqa: BLE001
        if cell is None:
            return {
                "serving_block": serving_block,
                "claimed": False,
                "accepted": False,
                "controller_failure_code": type(exc).__name__.lower(),
            }
        stage = "post_claim"
        if out_dir and (out_dir / "scoring-intent.json").is_file():
            stage = "authoritative_scoring_started"
        elif out_dir and (out_dir / "trace-manifest.json").is_file():
            stage = "model_trace_created"
        code = f"{stage}.{type(exc).__name__.lower()}"[:128]
        with suppress(Exception):
            ledger.request_retry_review(
                database,
                cell_id=cell["cell_id"],
                worker_id=cell["worker_id"],
                claim_id=cell["claim_id"],
                failure_code=code,
            )
        return {
            "serving_block": serving_block,
            "claimed": True,
            "accepted": False,
            "ledger_cell_id": cell["cell_id"],
            "failure_code": code,
        }


def run_batch(
    *,
    repo_root: Path,
    database: Path | str,
    plan: Path,
    campaign_path: Path,
    selection_path: Path,
    snapshot_path: Path,
    output_root: Path,
    claim_root: Path,
    serving_blocks: list[str],
    cells_per_block: int,
    worker_id: str,
) -> dict[str, Any]:
    ledger = rollout_postgres if isinstance(database, str) else rollout_ledger
    campaign = _load(campaign_path)
    selection = _load(selection_path)
    rollout_campaign.build_plan_rows(campaign_path, selection_path, snapshot_path)
    if isinstance(database, str):
        rollout_postgres.verify_plan(database, plan)
    else:
        rollout_ledger.initialize(database, plan)
        rollout_campaign.import_snapshot(database, snapshot_path)
    expanded = _universe_index(campaign, repo_root)
    jobs = [(block, index) for block in serving_blocks for index in range(cells_per_block)]
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(serving_blocks)) as pool:
        active = {
            pool.submit(
                run_one,
                database=database,
                ledger=ledger,
                campaign=campaign,
                selection=selection,
                universe_index=expanded,
                serving_block=block,
                worker_id=f"{worker_id}-{block}-{index}",
                output_root=output_root,
                claim_root=claim_root,
                proxy_script=repo_root / "evals/fleet/fixed_proxy.py",
            ): (block, index)
            for block, index in jobs
        }
        for future in as_completed(active):
            results.append(future.result())
    status = ledger.summary(database)
    body = {
        "schema_version": TERMINAL_SCHEMA,
        "accepted": bool(results) and all(row.get("accepted") is True for row in results),
        "worker_id": worker_id,
        "requested_cells": len(jobs),
        "accepted_cells": sum(row.get("accepted") is True for row in results),
        "results": sorted(results, key=lambda row: row["serving_block"]),
        "ledger_status": status,
        "scores_included": False,
        "prompts_or_traces_included": False,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    terminal = output_root / f"TERMINAL-{worker_id}.json"
    _safe_write_once(terminal, body)
    return _load(terminal)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    database_group = parser.add_mutually_exclusive_group(required=True)
    database_group.add_argument("--database", type=Path)
    database_group.add_argument("--postgres-dsn-env")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--claim-root", type=Path, required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--cells-per-block", type=int, default=1)
    parser.add_argument(
        "--serving-block",
        action="append",
        choices=sorted({value[0] for value in rollout_campaign.BLOCKS.values()}),
    )
    args = parser.parse_args()
    if args.cells_per_block < 1:
        parser.error("--cells-per-block must be positive")
    blocks = args.serving_block or sorted({value[0] for value in rollout_campaign.BLOCKS.values()})
    try:
        database: Path | str
        if args.postgres_dsn_env:
            database = os.environ.get(args.postgres_dsn_env, "")
            if not database:
                raise RuntimeError(f"{args.postgres_dsn_env} is required")
        else:
            database = args.database
        result = run_batch(
            repo_root=args.repo_root.resolve(),
            database=database,
            plan=args.plan,
            campaign_path=args.campaign,
            selection_path=args.selection,
            snapshot_path=args.snapshot,
            output_root=args.output_root,
            claim_root=args.claim_root,
            serving_blocks=blocks,
            cells_per_block=args.cells_per_block,
            worker_id=args.worker_id,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return int(any("controller_failure_code" in row for row in result["results"]))
    except BaseException as exc:  # noqa: BLE001
        # A reviewed cell outcome is not a controller exception. Preserve the
        # sanitized failure and return nonzero; never hide a real runtime defect.
        body = {
            "schema_version": TERMINAL_SCHEMA,
            "accepted": False,
            "worker_id": args.worker_id,
            "controller_failure_code": type(exc).__name__.lower(),
            "scores_included": False,
            "prompts_or_traces_included": False,
        }
        terminal = args.output_root / f"TERMINAL-{args.worker_id}.json"
        with suppress(FileExistsError):
            _safe_write_once(terminal, body)
        print(json.dumps(body, indent=2, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
