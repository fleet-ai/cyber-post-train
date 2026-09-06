"""Create, watchdog, and score-free generation-exact parity controller for GLM v35."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from evals.fleet import exact_pass4_crypto as crypto
from evals.fleet import glm53_dedicated_v32_stale_run_reconciliation_v1 as stale
from evals.fleet import glm53_dedicated_v33_controller_v1 as engine
from evals.fleet import glm53_dedicated_v35_create_v1 as server
from evals.fleet import glm53_dedicated_v35_incluster_parity_v1 as parity
from evals.fleet import glm53_dedicated_v35_live_authorization_v1 as live
from evals.fleet import glm53_dedicated_v35_watchdog_live_release_v1 as adapter

RESULT_SCHEMA = "fleet-glm53-dedicated-v35-controller-result-v1"
SEED_SCHEMA = "fleet-glm53-dedicated-v35-controller-seed-v1"
SEED_KEYS = {
    "schema_version",
    "status",
    "package_commit",
    "server_title",
    "server_run_dir",
    "fresh_in_controller_authorization_required",
    "fleet_task_instance_calls",
    "fleet_session_calls",
    "verifier_calls",
    "scoring_calls",
    "protected_content_included",
    "receipt_sha256",
}
READY_TIMEOUT_SECONDS = engine.READY_TIMEOUT_SECONDS
PARITY_TIMEOUT_SECONDS = engine.PARITY_TIMEOUT_SECONDS
POLL_SECONDS = engine.POLL_SECONDS
ControllerError = engine.ControllerError
_LOCK = threading.Lock()


@contextmanager
def bound_engine() -> Iterator[None]:
    if not _LOCK.acquire(blocking=False):
        raise ControllerError("v35_controller_engine_already_bound")
    values = {"server": server, "parity": parity, "adapter": adapter}
    prior = {name: getattr(engine, name) for name in values}
    try:
        for name, value in values.items():
            setattr(engine, name, value)
        yield
    finally:
        for name, value in prior.items():
            setattr(engine, name, value)
        _LOCK.release()


def build_seed(commit: str) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema_version": SEED_SCHEMA,
        "status": "HELD_FOR_FRESH_IN_CONTROLLER_AUTHORIZATION",
        "package_commit": commit,
        "server_title": server.TITLE,
        "server_run_dir": server.RUN_DIR,
        "fresh_in_controller_authorization_required": True,
        "fleet_task_instance_calls": 0,
        "fleet_session_calls": 0,
        "verifier_calls": 0,
        "scoring_calls": 0,
        "protected_content_included": False,
    }
    body["receipt_sha256"] = crypto.digest_without(body, "receipt_sha256")
    return body


def validate_seed(value: dict[str, Any], commit: str) -> None:
    if (
        set(value) != SEED_KEYS
        or value != build_seed(commit)
        or value.get("receipt_sha256") != crypto.digest_without(value, "receipt_sha256")
    ):
        raise ControllerError("v35_controller_seed_invalid")


def _load_seed(path: Path, commit: str) -> None:
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ControllerError("v35_controller_seed_invalid") from exc
    if not isinstance(value, dict):
        raise ControllerError("v35_controller_seed_invalid")
    validate_seed(value, commit)


def build_fresh_authorization(
    *,
    stale_backend_factory: Callable[[], Any] = stale.SystemBackend,
    live_backend_factory: Callable[[], Any] = live.SystemBackend,
    now: Callable[[], float] = time.time,
) -> dict[str, Any]:
    """Repeat every score-blind global gate after this Pod actually starts."""

    if not os.environ.get("FLEET_API_KEY"):
        raise ControllerError("v35_controller_fleet_credential_absent")
    observation_started_at = now()
    stale_backend = stale_backend_factory()
    rows, pages = stale_backend.list_runs()
    reconciliation = stale.build_reconciliation(
        backend=stale_backend,
        now=observation_started_at,
        rows_snapshot=(rows, pages),
    )
    stale.validate_reconciliation(
        reconciliation, rows=rows, now=observation_started_at
    )
    authorization = live.build_live_authorization(
        backend=live_backend_factory(),
        payload=server.payload(),
        title=server.TITLE,
        run_dir=server.RUN_DIR,
        control_result_path=server.RESULT_PATH,
        request_sha256=server.request_sha256(),
        priority_class=server.payload()["priority_class"],
        stale_run_reconciliation=reconciliation,
        now=observation_started_at,
    )
    authorization = live.bind_observation_completion(
        authorization, completed_at_epoch=now()
    )
    server.validate_authorization(authorization)
    return authorization


def run(
    root: Path,
    commit: str,
    seed_path: Path,
    *,
    stale_backend_factory: Callable[[], Any] = stale.SystemBackend,
    live_backend_factory: Callable[[], Any] = live.SystemBackend,
    now: Callable[[], float] = time.time,
) -> dict[str, Any]:
    _load_seed(seed_path, commit)
    authorization = build_fresh_authorization(
        stale_backend_factory=stale_backend_factory,
        live_backend_factory=live_backend_factory,
        now=now,
    )
    with tempfile.TemporaryDirectory(prefix="glm53-v35-auth-") as directory:
        authorization_path = Path(directory) / "CREATE-AUTHORIZED.json"
        descriptor = os.open(
            authorization_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(crypto.canonical_json(authorization) + b"\n")
        with bound_engine():
            receipt = engine.run(root, commit, authorization_path)
    if (
        receipt.get("schema_version") != engine.RESULT_SCHEMA
        or receipt.get("fleet_task_instance_calls") != 0
        or receipt.get("fleet_session_calls") != 0
        or receipt.get("verifier_calls") != 0
        or receipt.get("scoring_calls") != 0
        or receipt.get("qualification_launch_authorized") is not False
        or receipt.get("scored_launch_authorized") is not False
        or receipt.get("protected_content_included") is not False
        or receipt.get("receipt_sha256") != crypto.digest_without(receipt, "receipt_sha256")
    ):
        raise ControllerError("v35_controller_result_invalid")
    receipt["schema_version"] = RESULT_SCHEMA
    receipt["receipt_sha256"] = crypto.digest_without(receipt, "receipt_sha256")
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/workspace"))
    parser.add_argument("--package-commit", required=True)
    parser.add_argument("--authorization", dest="seed", type=Path, required=True)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            run(args.root, args.package_commit, args.seed),
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
