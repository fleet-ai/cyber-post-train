"""Create, watchdog, and score-free generation-exact parity controller for GLM v33."""

from __future__ import annotations

import argparse
import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from evals.fleet import exact_pass4_crypto as crypto
from evals.fleet import glm53_dedicated_v32_controller_v1 as engine
from evals.fleet import glm53_dedicated_v33_create_v1 as server
from evals.fleet import glm53_dedicated_v33_incluster_parity_v1 as parity
from evals.fleet import glm53_dedicated_v33_watchdog_live_release_v1 as adapter

RESULT_SCHEMA = "fleet-glm53-dedicated-v33-controller-result-v1"
READY_TIMEOUT_SECONDS = engine.READY_TIMEOUT_SECONDS
PARITY_TIMEOUT_SECONDS = engine.PARITY_TIMEOUT_SECONDS
POLL_SECONDS = engine.POLL_SECONDS
_LOCK = threading.Lock()


class ControllerError(RuntimeError):
    """The v33 score-free lifecycle controller failed closed."""


@contextmanager
def bound_engine() -> Iterator[None]:
    if not _LOCK.acquire(blocking=False):
        raise ControllerError("v33_controller_engine_already_bound")
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


def run(root: Path, commit: str, authorization_path: Path) -> dict[str, Any]:
    with bound_engine():
        receipt = engine.run(root, commit, authorization_path)
    if receipt.get("schema_version") != engine.RESULT_SCHEMA:
        raise ControllerError("v33_controller_result_invalid")
    receipt["schema_version"] = RESULT_SCHEMA
    receipt["receipt_sha256"] = crypto.digest_without(receipt, "receipt_sha256")
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/workspace"))
    parser.add_argument("--package-commit", required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            run(args.root, args.package_commit, args.authorization),
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
