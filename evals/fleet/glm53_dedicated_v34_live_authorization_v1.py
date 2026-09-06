"""Generation-exact zero-server authorization for the GLM v34 lifecycle."""

from __future__ import annotations

import argparse
import json
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from evals.fleet import exact_pass4_crypto as crypto
from evals.fleet import glm53_dedicated_v33_live_authorization_v1 as engine

LIVE_SCHEMA = "fleet-glm53-dedicated-v34-live-create-observation-v1"
AUTHORIZATION_SCHEMA = "fleet-glm53-dedicated-v34-create-authorization-v1"
CONTROL_RESULT_PATH = (
    "/mnt/sfs/jobs/chris-cyber-evalserve-glm53-tp8-a-v34-create-control/CREATED.json"
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
        raise LiveAuthorizationError("v34_live_authorization_engine_already_bound")
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


def validate_live_observation(value: Mapping[str, Any], authorization: Mapping[str, Any]) -> None:
    with bound_engine():
        engine.validate_live_observation(value, authorization)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stale-reconciliation", type=Path, required=True)
    args = parser.parse_args(argv)
    from evals.fleet import glm53_dedicated_v32_stale_run_reconciliation_v1 as stale
    from evals.fleet import glm53_dedicated_v34_create_v1 as server

    try:
        reconciliation = json.loads(args.stale_reconciliation.read_text())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise LiveAuthorizationError("v34_stale_run_reconciliation_invalid") from exc
    if not isinstance(reconciliation, dict):
        raise LiveAuthorizationError("v34_stale_run_reconciliation_invalid")
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
