"""Held create-once launcher for the reviewed GLM v34 controller package."""

from __future__ import annotations

import argparse
import json
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from evals.fleet import glm53_dedicated_v33_launch_v1 as engine
from evals.fleet import glm53_dedicated_v34_controller_package_v1 as package
from evals.fleet import glm53_dedicated_v34_create_v1 as server
from evals.fleet import glm53_dedicated_v34_live_authorization_v1 as live

FROZEN_PACKAGE_COMMIT = "3b373975dbbe2cf5aa5b5e20fcb474faeeaffdaa"
FROZEN_PACKAGE_SHA256 = "sha256:4b3d7573733258842004b73c47e572b6635f273f6ab61165b13d975ccc735755"
HELD_SCHEMA = "fleet-glm53-dedicated-v34-launch-held-v1"
RESULT_SCHEMA = "fleet-glm53-dedicated-v34-controller-submission-v1"
LaunchError = engine.LaunchError
KubernetesBackend = engine.KubernetesBackend
SystemKubernetesBackend = engine.SystemKubernetesBackend
_LOCK = threading.Lock()


@contextmanager
def bound_engine() -> Iterator[None]:
    if not _LOCK.acquire(blocking=False):
        raise LaunchError("v34_launcher_engine_already_bound")
    values = {
        "FROZEN_PACKAGE_COMMIT": FROZEN_PACKAGE_COMMIT,
        "FROZEN_PACKAGE_SHA256": FROZEN_PACKAGE_SHA256,
        "HELD_SCHEMA": HELD_SCHEMA,
        "RESULT_SCHEMA": RESULT_SCHEMA,
        "package": package,
        "server": server,
        "live": live,
    }
    prior = {name: getattr(engine, name) for name in values}
    try:
        for name, value in values.items():
            setattr(engine, name, value)
        yield
    finally:
        for name, value in prior.items():
            setattr(engine, name, value)
        _LOCK.release()


def _identities() -> tuple[tuple[str, str], ...]:
    return (
        ("configmap", package.CONFIGMAP_NAME),
        ("configmap", package.AUTHORIZATION_CONFIGMAP_NAME),
        ("job", package.JOB_NAME),
    )


def build_held(root: Path) -> dict[str, Any]:
    with bound_engine():
        value = engine.build_held(root)
    value.update(
        {
            "response_body_identity_required": False,
            "exhaustive_pre_post_jobs_api_reconciliation_required": True,
            "ambiguous_post_create_identity_release_required": True,
        }
    )
    from evals.fleet import exact_pass4_crypto as crypto

    value["receipt_sha256"] = crypto.digest_without(value, "receipt_sha256")
    return value


def execute(
    *,
    root: Path,
    output: Path,
    kubernetes: KubernetesBackend,
    stale_backend_factory: Callable[[], Any],
    live_backend_factory: Callable[[], Any],
    now: Callable[[], float],
) -> dict[str, Any]:
    with bound_engine():
        return engine.execute(
            root=root,
            output=output,
            kubernetes=kubernetes,
            stale_backend_factory=stale_backend_factory,
            live_backend_factory=live_backend_factory,
            now=now,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(build_held(args.root), sort_keys=True, separators=(",", ":")))
        return 0
    if args.output is None:
        raise LaunchError("v34_launch_output_required")
    result = execute(
        root=args.root,
        output=args.output,
        kubernetes=SystemKubernetesBackend(),
        stale_backend_factory=engine.stale.SystemBackend,
        live_backend_factory=live.SystemBackend,
        now=engine.time.time,
    )
    print(
        json.dumps(
            {"status": result["status"], "receipt_sha256": result["receipt_sha256"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
