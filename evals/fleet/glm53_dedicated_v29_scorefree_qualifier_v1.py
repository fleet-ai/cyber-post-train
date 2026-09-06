"""Generation-correct adapter for the reviewed GLM v23 score-free qualifier."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from evals.fleet import glm53_dedicated_v23_scorefree_qualifier_v1 as engine_module
from evals.fleet import glm53_dedicated_v29_create_v1 as server

SCHEMA = "fleet-glm53-dedicated-v29-scorefree-qualification-held-v1"
AUTH_SCHEMA = "fleet-glm53-dedicated-v29-scorefree-authorization-v1"
RAW_SCHEMA = "fleet-glm53-dedicated-v29-scorefree-raw-v1"
VERDICT_SCHEMA = "fleet-glm53-dedicated-v29-scorefree-qualified-v1"
JOB_NAME = "chris-glm53-dedicated-v29-scorefree-qualification-v1"
RESULT_ROOT = Path("/mnt/sfs/jobs") / JOB_NAME
SERVER_TITLE = server.TITLE
SERVER_RUN_DIR = server.RUN_DIR
LEASE_ROOT = Path("/mnt/sfs/endpoint-leases/opencode11827-dedicated-v29-v1")

MODEL_REVISION = engine_module.MODEL_REVISION
SERVED_ID = engine_module.SERVED_ID
CONTEXT_LENGTH = engine_module.CONTEXT_LENGTH
IDLE_RELEASE_SECONDS = engine_module.IDLE_RELEASE_SECONDS
CPU_PRIORITY_CLASS = engine_module.CPU_PRIORITY_CLASS
CPU_PRIORITY_VALUE = engine_module.CPU_PRIORITY_VALUE
CONCURRENCY = engine_module.CONCURRENCY
GPU_OBSERVER_SCHEMA = "fleet-glm53-dedicated-v29-scorefree-gpu-wave-v1"
GPU_OBSERVER_WAIT_SECONDS = engine_module.GPU_OBSERVER_WAIT_SECONDS
QualificationError = engine_module.QualificationError
engine = engine_module.engine

_LOCK = threading.Lock()


@contextmanager
def bound_engine() -> Iterator[None]:
    """Bind every generation-bearing v23 global for one exact v29 call."""

    if not _LOCK.acquire(blocking=False):
        raise QualificationError("v29_scorefree_engine_already_bound")
    values = {
        "SCHEMA": SCHEMA,
        "AUTH_SCHEMA": AUTH_SCHEMA,
        "RAW_SCHEMA": RAW_SCHEMA,
        "VERDICT_SCHEMA": VERDICT_SCHEMA,
        "JOB_NAME": JOB_NAME,
        "RESULT_ROOT": RESULT_ROOT,
        "SERVER_TITLE": SERVER_TITLE,
        "SERVER_RUN_DIR": SERVER_RUN_DIR,
        "LEASE_ROOT": LEASE_ROOT,
        "GPU_OBSERVER_SCHEMA": GPU_OBSERVER_SCHEMA,
    }
    prior = {name: getattr(engine_module, name) for name in values}
    try:
        for name, value in values.items():
            setattr(engine_module, name, value)
        yield
    finally:
        for name, value in prior.items():
            setattr(engine_module, name, value)
        _LOCK.release()


def build_held() -> dict[str, Any]:
    with bound_engine():
        return engine_module.build_held()


def load(path: Path) -> dict[str, Any]:
    return engine_module.load(path)


def _validate_binding(binding: dict[str, Any]) -> None:
    with bound_engine():
        engine_module._validate_binding(binding)  # noqa: SLF001


def validate_authorization(authorization: dict[str, Any]) -> None:
    with bound_engine():
        engine_module.validate_authorization(authorization)


def authorize(
    binding: dict[str, Any],
    parity: dict[str, Any],
    watchdog: dict[str, Any],
    live: dict[str, Any],
) -> dict[str, Any]:
    with bound_engine():
        return engine_module.authorize(binding, parity, watchdog, live)


def main() -> int:
    with bound_engine():
        return engine_module.main()


if __name__ == "__main__":
    raise SystemExit(main())
