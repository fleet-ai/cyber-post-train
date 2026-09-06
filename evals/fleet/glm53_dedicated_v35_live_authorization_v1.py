"""Generation-exact zero-server authorization for the GLM v35 lifecycle."""

from __future__ import annotations

import argparse
import json
import math
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from evals.fleet import exact_pass4_crypto as crypto
from evals.fleet import glm53_dedicated_v33_live_authorization_v1 as engine

LIVE_SCHEMA = "fleet-glm53-dedicated-v35-live-create-observation-v1"
AUTHORIZATION_SCHEMA = "fleet-glm53-dedicated-v35-create-authorization-v1"
CONTROL_RESULT_PATH = (
    "/mnt/sfs/jobs/chris-cyber-evalserve-glm53-tp8-a-v35-create-control/CREATED.json"
)
SFS_OBSERVER_POD_NAME = engine.SFS_OBSERVER_POD_NAME
SFS_OBSERVER_POD_UID = engine.SFS_OBSERVER_POD_UID
SFS_OBSERVER_MOUNT_PATH = engine.SFS_OBSERVER_MOUNT_PATH
MAX_EXACT_GET_WORKERS = engine.MAX_EXACT_GET_WORKERS
LiveAuthorizationError = engine.LiveAuthorizationError
SystemBackend = engine.SystemBackend
_LOCK = threading.Lock()


@contextmanager
def bound_engine() -> Iterator[None]:
    if not _LOCK.acquire(blocking=False):
        raise LiveAuthorizationError("v35_live_authorization_engine_already_bound")
    values = {"LIVE_SCHEMA": LIVE_SCHEMA, "CONTROL_RESULT_PATH": CONTROL_RESULT_PATH}
    prior = {name: getattr(engine, name) for name in values}
    try:
        for name, value in values.items():
            setattr(engine, name, value)
        yield
    finally:
        for name, value in prior.items():
            setattr(engine, name, value)
        _LOCK.release()


def build_live_authorization(**kwargs: Any) -> dict[str, Any]:
    with bound_engine():
        value = engine.build_live_authorization(**kwargs)
    value["schema_version"] = AUTHORIZATION_SCHEMA
    value["receipt_sha256"] = crypto.digest_without(value, "receipt_sha256")
    return value


def bind_observation_completion(
    value: dict[str, Any], *, completed_at_epoch: float
) -> dict[str, Any]:
    """Bind freshness to completion of the full read-only observation.

    The inherited builder timestamps the beginning of its bounded scan.  That
    timestamp is useful for the embedded stale-run evidence, but it is not the
    right freshness boundary for the immediate create gate: a complete scan can
    itself take longer than the 60-second create-authorization lifetime.  The
    live observation timestamp therefore records when that scan completed,
    while the embedded stale reconciliation retains its own start timestamp.
    """

    live = value.get("live_observation")
    started_at = value.get("observed_at_epoch")
    if (
        not isinstance(live, dict)
        or not isinstance(started_at, (int, float))
        or isinstance(started_at, bool)
        or not math.isfinite(started_at)
        or not isinstance(completed_at_epoch, (int, float))
        or isinstance(completed_at_epoch, bool)
        or not math.isfinite(completed_at_epoch)
        or completed_at_epoch < started_at
    ):
        raise LiveAuthorizationError("v35_live_observation_completion_invalid")
    validate_live_observation(live, value)
    rebound = dict(value)
    rebound_live = dict(live)
    rebound_live["observed_at_epoch"] = completed_at_epoch
    rebound_live["receipt_sha256"] = crypto.digest_without(rebound_live, "receipt_sha256")
    rebound["observed_at_epoch"] = completed_at_epoch
    rebound["live_observation"] = rebound_live
    rebound["receipt_sha256"] = crypto.digest_without(rebound, "receipt_sha256")
    validate_live_observation(rebound_live, rebound)
    return rebound


def validate_live_observation(value: Mapping[str, Any], authorization: Mapping[str, Any]) -> None:
    with bound_engine():
        engine.validate_live_observation(value, authorization)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stale-reconciliation", type=Path, required=True)
    args = parser.parse_args(argv)
    from evals.fleet import glm53_dedicated_v32_stale_run_reconciliation_v1 as stale
    from evals.fleet import glm53_dedicated_v35_create_v1 as server

    try:
        reconciliation = json.loads(args.stale_reconciliation.read_text())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise LiveAuthorizationError("v35_stale_run_reconciliation_invalid") from exc
    if not isinstance(reconciliation, dict):
        raise LiveAuthorizationError("v35_stale_run_reconciliation_invalid")
    stale.validate_reconciliation(reconciliation, rows=reconciliation.get("project_rows"))
    value = build_live_authorization(
        backend=SystemBackend(),
        payload=server.payload(),
        title=server.TITLE,
        run_dir=server.RUN_DIR,
        control_result_path=server.RESULT_PATH,
        request_sha256=server.request_sha256(),
        priority_class=server.payload()["priority_class"],
        stale_run_reconciliation=reconciliation,
    )
    server.validate_authorization(value)
    engine.engine._write_once(args.output, value)  # noqa: SLF001
    print(
        json.dumps(
            {"status": value["status"], "receipt_sha256": value["receipt_sha256"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
