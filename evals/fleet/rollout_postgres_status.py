"""Read-only, score-blind PostgreSQL diagnostics; never repair or initialize state.

Campaign slot decisions still require rollout_refiller's Kubernetes/ledger reconciliation.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence

from evals.fleet import rollout_postgres


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsn-env", default="ROLLOUT_DATABASE_URL")
    parser.add_argument("view", choices=("summary", "active"))
    args = parser.parse_args(argv)
    dsn = os.environ.get(args.dsn_env)
    if not dsn:
        print(json.dumps({"ok": False, "error_code": "database_environment_missing"}))
        return 1
    try:
        operation = (
            rollout_postgres.summary if args.view == "summary" else rollout_postgres.active_claims
        )
        value = operation(dsn)
    except Exception:
        # Driver errors can contain connection strings. Never print or chain them.
        print(json.dumps({"ok": False, "error_code": "read_only_observation_failed"}))
        return 1
    print(json.dumps({"ok": True, "view": args.view, "observation": value}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
