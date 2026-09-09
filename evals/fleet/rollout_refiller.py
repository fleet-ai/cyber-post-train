"""Score-blind observation and refill planning for the four-route rollout pool.

The default command is deliberately read-only.  It reconciles live Kubernetes
controllers with active database claims and reports empty route slots without
creating, replacing, or deleting any object.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

NAMESPACE = "fleet-train-jobs"
JOB_PREFIX = "chris-cyber-rollout-ledger-"
REFILL_LABEL = "cyber-post-train.fleet.ai/refiller"
WORKER_PREFIX_ANNOTATION = "cyber-post-train.fleet.ai/worker-prefix"
ROUTES = (
    "glm-dedicated",
    "glm-shared",
    "qwen-dedicated",
    "qwen-shared",
)
ENDPOINTS = {
    "glm-dedicated": "chris-cyber-glm53-dedicated-v1",
    "glm-shared": "glm-5.3",
    "qwen-dedicated": "chris-cyber-qwen38-27b-dedicated-v1",
    "qwen-shared": "qwen3.8-27b",
}


class RefillerError(RuntimeError):
    """A safe, content-blind observation failure."""


@dataclass(frozen=True)
class RouteStatus:
    route: str
    target: int
    active_jobs: int
    active_claims: int
    occupancy: int
    deficit: int
    endpoint_ready: bool


def _route_from_job_name(name: str) -> str | None:
    if not name.startswith(JOB_PREFIX):
        return None
    suffix = name.removeprefix(JOB_PREFIX)
    for route in ROUTES:
        if suffix.startswith(f"{route}-"):
            return route
    return None


def _is_terminal(job: dict[str, Any]) -> bool:
    conditions = (job.get("status") or {}).get("conditions") or []
    return any(
        condition.get("status") == "True" and condition.get("type") in {"Complete", "Failed"}
        for condition in conditions
    )


def active_jobs_by_route(jobs: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts = dict.fromkeys(ROUTES, 0)
    for job in jobs:
        metadata = job.get("metadata") or {}
        route = _route_from_job_name(str(metadata.get("name") or ""))
        if route is None or metadata.get("deletionTimestamp") or _is_terminal(job):
            continue
        status = job.get("status") or {}
        labels = metadata.get("labels") or {}
        # An active Pod consumes a slot. A refiller-created Job also consumes a
        # slot while waiting for admission or initialization before its claim.
        if int(status.get("active") or 0) > 0 or labels.get(REFILL_LABEL):
            counts[route] += 1
    return counts


def active_worker_prefixes(jobs: Sequence[dict[str, Any]]) -> set[str]:
    prefixes: set[str] = set()
    for job in jobs:
        metadata = job.get("metadata") or {}
        if metadata.get("deletionTimestamp") or _is_terminal(job):
            continue
        if int((job.get("status") or {}).get("active") or 0) < 1:
            continue
        prefix = str((metadata.get("annotations") or {}).get(WORKER_PREFIX_ANNOTATION) or "")
        if prefix:
            prefixes.add(prefix)
    return prefixes


def endpoint_readiness(models: Sequence[dict[str, Any]]) -> dict[str, bool]:
    by_name = {(model.get("metadata") or {}).get("name"): model for model in models}
    result: dict[str, bool] = {}
    for route, name in ENDPOINTS.items():
        conditions = ((by_name.get(name) or {}).get("status") or {}).get("conditions") or []
        result[route] = any(
            condition.get("type") == "Ready" and condition.get("status") == "True"
            for condition in conditions
        )
    return result


def plan_routes(
    *,
    jobs: Sequence[dict[str, Any]],
    active_claims: dict[str, int],
    models: Sequence[dict[str, Any]],
    target_per_route: int,
) -> list[RouteStatus]:
    if target_per_route < 1:
        raise RefillerError("target_per_route must be positive")
    job_counts = active_jobs_by_route(jobs)
    ready = endpoint_readiness(models)
    statuses: list[RouteStatus] = []
    for route in ROUTES:
        claims = int(active_claims.get(route, 0))
        if claims < 0:
            raise RefillerError("active claim counts cannot be negative")
        # A controller usually appears in both sources. Taking the maximum
        # counts initialization-only Jobs and orphan claims without double-counting.
        occupancy = max(job_counts[route], claims)
        deficit = max(0, target_per_route - occupancy) if ready[route] else 0
        statuses.append(
            RouteStatus(
                route=route,
                target=target_per_route,
                active_jobs=job_counts[route],
                active_claims=claims,
                occupancy=occupancy,
                deficit=deficit,
                endpoint_ready=ready[route],
            )
        )
    return statuses


def _run_json(command: Sequence[str]) -> dict[str, Any]:
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RefillerError(f"read-only command failed: {command[0]} {command[1]}")
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RefillerError("read-only command returned invalid JSON") from exc
    if not isinstance(value, dict):
        raise RefillerError("read-only command did not return a JSON object")
    return value


def _sqlite_claims(namespace: str, pod: str) -> dict[str, Any]:
    code = """
import json, sqlite3
p = '/mnt/sfs/jobs/chris-cyber-q38-glm53-pass4-ledger-v1/ledger.sqlite3'
c = sqlite3.connect('file:' + p + '?mode=ro', uri=True, timeout=30)
r = {
    row[0]: row[1]
    for row in c.execute(
        "SELECT serving_block, COUNT(*) FROM rollout_cells "
        "WHERE state IN ('claimed','running','grading') GROUP BY serving_block"
    )
}
columns = ('serving_block', 'worker_id', 'claim_id', 'cell_id', 'stale')
o = [
    dict(zip(columns, row, strict=True))
    for row in c.execute(
        "SELECT serving_block, worker_id, claim_id, cell_id, "
        "julianday(lease_expires_at) < julianday('now') FROM rollout_cells "
        "WHERE state IN ('claimed','running','grading') "
        "ORDER BY serving_block, worker_id, cell_id"
    )
]
print(json.dumps({
    'active_claims': r,
    'stale_active': sum(bool(row['stale']) for row in o),
    'owners': o,
}, sort_keys=True))
"""
    value = _run_json(
        ["kubectl", "-n", namespace, "exec", pod, "-c", "evaluator", "--", "python", "-c", code]
    )
    return value


def _postgres_claims(namespace: str, pod: str) -> dict[str, Any]:
    sql = """
    SELECT json_build_object(
      'active_claims', COALESCE((
        SELECT json_object_agg(serving_block, count)
        FROM (
          SELECT serving_block, COUNT(*) AS count
          FROM rollout_cells
          WHERE state IN ('claimed','running','grading')
          GROUP BY serving_block
        ) counts
      ), '{}'::json),
      'stale_active', (
        SELECT COUNT(*) FROM rollout_cells
        WHERE state IN ('claimed','running','grading')
          AND lease_expires_at < CURRENT_TIMESTAMP
      ),
      'owners', COALESCE((
        SELECT json_agg(json_build_object(
          'serving_block', serving_block,
          'worker_id', worker_id,
          'claim_id', claim_id,
          'cell_id', cell_id,
          'stale', lease_expires_at < CURRENT_TIMESTAMP
        ) ORDER BY serving_block, worker_id, cell_id)
        FROM rollout_cells
        WHERE state IN ('claimed','running','grading')
      ), '[]'::json)
    )
    """
    return _run_json(
        [
            "kubectl",
            "-n",
            namespace,
            "exec",
            pod,
            "-c",
            "postgres",
            "--",
            "psql",
            "--username=rollout",
            "--dbname=rollout",
            "--tuples-only",
            "--no-align",
            "--quiet",
            "--command",
            sql,
        ]
    )


def _ledger_claims(namespace: str, pod: str, ledger_kind: str) -> dict[str, Any]:
    value = (
        _postgres_claims(namespace, pod)
        if ledger_kind == "postgres"
        else _sqlite_claims(namespace, pod)
    )
    claims = value.get("active_claims")
    if not isinstance(claims, dict):
        raise RefillerError("ledger observation omitted active_claims")
    owners = value.get("owners")
    if not isinstance(owners, list):
        raise RefillerError("ledger observation omitted active claim owners")
    return {
        "active_claims": {str(key): int(item) for key, item in claims.items()},
        "stale_active": int(value.get("stale_active", 0)),
        "owners": owners,
    }


def observe(
    *,
    namespace: str,
    ledger_pod: str,
    ledger_kind: str = "sqlite",
    target_per_route: int,
    runner: Callable[[Sequence[str]], dict[str, Any]] = _run_json,
) -> dict[str, Any]:
    jobs = runner(["kubectl", "-n", namespace, "get", "jobs", "-o", "json"])
    models = runner(["kubectl", "-n", "inference", "get", "inferencemodels", "-o", "json"])
    ledger = _ledger_claims(namespace, ledger_pod, ledger_kind)
    active_claims = ledger["active_claims"]
    stale_active = ledger["stale_active"]
    worker_prefixes = active_worker_prefixes(jobs.get("items") or [])
    orphan_claims = [
        owner
        for owner in ledger["owners"]
        if not any(
            str(owner.get("worker_id") or "").startswith(f"{prefix}-") for prefix in worker_prefixes
        )
    ]
    statuses = plan_routes(
        jobs=jobs.get("items") or [],
        active_claims=active_claims,
        models=models.get("items") or [],
        target_per_route=target_per_route,
    )
    blocked_reasons = []
    if stale_active:
        blocked_reasons.append("stale_active_claims")
    if orphan_claims:
        blocked_reasons.append("orphan_active_claims")
    body = {
        "schema_version": "fleet-rollout-refiller-observation-v1",
        "mode": "observe_only",
        "observed_at": datetime.now(UTC).isoformat(),
        "namespace": namespace,
        "target_total": target_per_route * len(ROUTES),
        "occupancy_total": sum(status.occupancy for status in statuses),
        "stale_active": stale_active,
        "orphan_active": len(orphan_claims),
        "orphan_claims": orphan_claims,
        "refill_blocked": bool(blocked_reasons),
        "blocked_reasons": blocked_reasons,
        "routes": [asdict(status) for status in statuses],
        "proposed_refills": [
            {"route": status.route, "count": status.deficit}
            for status in statuses
            if status.deficit and not blocked_reasons
        ],
        "would_create_objects": False,
        "scores_included": False,
        "prompts_or_traces_included": False,
    }
    return body


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", default=NAMESPACE)
    parser.add_argument("--ledger-pod", required=True)
    parser.add_argument("--ledger-kind", choices=("postgres", "sqlite"), default="postgres")
    parser.add_argument("--target-per-route", type=int, default=4)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result = observe(
        namespace=arguments.namespace,
        ledger_pod=arguments.ledger_pod,
        ledger_kind=arguments.ledger_kind,
        target_per_route=arguments.target_per_route,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
