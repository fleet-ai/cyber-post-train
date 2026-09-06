"""Credentialed create/readiness/watchdog controller for the held GLM v28 server."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from evals.fleet import glm53_dedicated_v23_scorefree_package_v1 as source_engine
from evals.fleet import glm53_dedicated_v24_watchdog_live_release_v1 as live_engine
from evals.fleet import glm53_dedicated_v28_create_v1 as server
from evals.fleet import glm53_dedicated_v28_watchdog_live_release_v1 as adapter

READY_TIMEOUT_SECONDS = 600
POLL_SECONDS = 5


class ControllerError(RuntimeError):
    """The v28 credentialed lifecycle controller failed closed."""


@contextmanager
def materialized_source(root: Path) -> Iterator[None]:
    """Use digest-bound projected source instead of requiring a Git checkout."""

    original = source_engine._source  # noqa: SLF001

    def read(_root: Path, _commit: str, path: str) -> bytes:
        target = root / path
        if not target.is_file():
            raise ControllerError(f"v28_materialized_source_absent:{path}")
        return target.read_bytes()

    source_engine._source = read  # type: ignore[assignment]  # noqa: SLF001
    try:
        yield
    finally:
        source_engine._source = original  # type: ignore[assignment]  # noqa: SLF001


def _load_authorization(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ControllerError("v28_controller_authorization_invalid") from exc
    if not isinstance(value, dict):
        raise ControllerError("v28_controller_authorization_invalid")
    server.validate_authorization(value)
    return value


def _wait_ready() -> None:
    deadline = time.monotonic() + READY_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if Path(server.READY_PATH).is_file():
            return
        time.sleep(POLL_SECONDS)
    raise ControllerError("v28_application_ready_timeout")


def run(root: Path, commit: str, authorization_path: Path) -> dict[str, Any]:
    if not os.environ.get("FLEET_API_KEY"):
        raise ControllerError("v28_controller_fleet_credential_absent")
    authorization = _load_authorization(authorization_path)
    created = server.create_once(authorization, result_path=Path(server.RESULT_PATH))
    api_run_id = created["api_run_id"]
    try:
        _wait_ready()
        priorities = live_engine._kubectl_json(  # noqa: SLF001
            "priorityclasses.scheduling.k8s.io"
        ).get("items", [])
        with materialized_source(root):
            return adapter.launch(
                root,
                commit,
                api_run_id,
                priority_classes=priorities,
            )
    except Exception:
        live_engine._release_local(api_run_id)  # noqa: SLF001
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/workspace"))
    parser.add_argument("--package-commit", required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = run(args.root, args.package_commit, args.authorization)
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
