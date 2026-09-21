"""Read-only, score-blind PostgreSQL diagnostics; never repair or initialize state.

Campaign slot decisions still require rollout_refiller's Kubernetes/ledger reconciliation.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence

from evals.fleet import rollout_postgres


def retry_review_summary(dsn: str) -> dict:
    """Group allowlisted lifecycle facts without changing sealed evaluator bytes."""
    with rollout_postgres._read_transaction(dsn) as connection:  # noqa: SLF001
        rows = connection.execute(
            """
            WITH latest_results AS (
                SELECT DISTINCT ON (cell_id)
                    cell_id, execution_id, session_id, agent_termination,
                    agent_exit_code, session_ingest_status
                FROM rollout_local_results
                ORDER BY cell_id, execution_generation DESC
            ), safe_rows AS (
                SELECT
                    COALESCE(c.failure_code, 'missing') AS failure_code,
                    CASE
                        WHEN c.result_class = 'infrastructure_invalid'
                            THEN c.result_class
                        WHEN c.result_class IS NULL THEN 'missing'
                        ELSE 'other'
                    END AS result_class,
                    CASE
                        WHEN latest.agent_termination IN (
                            'completed', 'execution_timeout', 'process_error',
                            'malformed_trace', 'harness_error', 'missing_terminal_step',
                            'incomplete_terminal_step', 'output_limit'
                        ) THEN latest.agent_termination
                        WHEN latest.agent_termination IS NULL THEN 'missing'
                        ELSE 'other'
                    END AS agent_termination,
                    latest.agent_exit_code,
                    CASE
                        WHEN latest.session_ingest_status IN ('completed', 'failed')
                            THEN latest.session_ingest_status
                        WHEN latest.session_ingest_status IS NULL THEN 'missing'
                        ELSE 'other'
                    END AS session_ingest_status,
                    latest.execution_id IS NOT NULL AS has_local_result,
                    latest.session_id IS NOT NULL AS has_session
                FROM rollout_cells AS c
                LEFT JOIN latest_results AS latest ON latest.cell_id = c.cell_id
                WHERE c.state = 'retry_review'
            )
            SELECT
                failure_code, result_class, agent_termination, agent_exit_code,
                session_ingest_status, has_local_result, has_session,
                COUNT(*) AS count
            FROM safe_rows
            GROUP BY
                failure_code, result_class, agent_termination,
                agent_exit_code, session_ingest_status,
                has_local_result, has_session
            ORDER BY
                failure_code, result_class, agent_termination,
                agent_exit_code NULLS FIRST, session_ingest_status,
                has_local_result, has_session
            """
        ).fetchall()
    groups = [dict(row) for row in rows]
    return {
        "retry_review": sum(row["count"] for row in groups),
        "with_local_result": sum(row["count"] for row in groups if row["has_local_result"]),
        "groups": groups,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsn-env", default="ROLLOUT_DATABASE_URL")
    parser.add_argument("view", choices=("summary", "active", "retry-review"))
    args = parser.parse_args(argv)
    dsn = os.environ.get(args.dsn_env)
    if not dsn:
        print(json.dumps({"ok": False, "error_code": "database_environment_missing"}))
        return 1
    try:
        operation = {
            "summary": rollout_postgres.summary,
            "active": rollout_postgres.active_claims,
            "retry-review": retry_review_summary,
        }[args.view]
        value = operation(dsn)
    except Exception:
        # Driver errors can contain connection strings. Never print or chain them.
        print(json.dumps({"ok": False, "error_code": "read_only_observation_failed"}))
        return 1
    print(json.dumps({"ok": True, "view": args.view, "observation": value}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
