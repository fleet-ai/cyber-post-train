"""Fresh v31 identity over the reviewed in-cluster DinD parity engine."""

from __future__ import annotations

import argparse
import json
import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from evals.fleet import exact_pass4_crypto as crypto
from evals.fleet import glm53_dedicated_v30_incluster_parity_v1 as engine

SCHEMA = "fleet-glm53-dedicated-v31-incluster-parity-authorization-v1"
RESULT_SCHEMA = "fleet-glm53-dedicated-v31-incluster-parity-result-v1"
PACKAGE_SCHEMA = "fleet-glm53-dedicated-v31-incluster-parity-package-v1"
JOB_NAME = "chris-glm53-dedicated-v31-incluster-actual-parity-v1"
CONFIGMAP_NAME = JOB_NAME + "-package"
AUTHORIZATION_CONFIGMAP_NAME = JOB_NAME + "-authorization"
RESULT_ROOT = Path("/mnt/sfs/jobs") / JOB_NAME
SERVER_TITLE = "chris-cyber-evalserve-glm53-tp8-a-v31"
SERVER_RUN_DIR = "/mnt/sfs/jobs/chris-cyber-evalserve-glm53-tp8-a-v31"
SERVED_ID = engine.SERVED_ID
MODEL_REVISION = engine.MODEL_REVISION
CONTEXT_LENGTH = engine.CONTEXT_LENGTH
NAMESPACE = engine.NAMESPACE
CPU_PRIORITY_CLASS = engine.CPU_PRIORITY_CLASS
PARITY_TEMP_ROOT = engine.PARITY_TEMP_ROOT
FILES = tuple(
    sorted(
        set(engine.FILES)
        | {"evals/fleet/glm53_dedicated_v31_incluster_parity_v1.py"}
    )
)
ParityPackageError = engine.ParityPackageError
_source = engine._source  # noqa: SLF001
_LOCK = threading.Lock()


@contextmanager
def bound_engine() -> Iterator[None]:
    if not _LOCK.acquire(blocking=False):
        raise ParityPackageError("v31_incluster_parity_engine_already_bound")
    values = {
        "SCHEMA": SCHEMA,
        "RESULT_SCHEMA": RESULT_SCHEMA,
        "PACKAGE_SCHEMA": PACKAGE_SCHEMA,
        "JOB_NAME": JOB_NAME,
        "CONFIGMAP_NAME": CONFIGMAP_NAME,
        "AUTHORIZATION_CONFIGMAP_NAME": AUTHORIZATION_CONFIGMAP_NAME,
        "RESULT_ROOT": RESULT_ROOT,
        "SERVER_TITLE": SERVER_TITLE,
        "SERVER_RUN_DIR": SERVER_RUN_DIR,
        "FILES": FILES,
        "_source": _source,
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


def validate_binding(binding: dict[str, Any]) -> None:
    with bound_engine():
        engine.validate_binding(binding)


def validate_authorization(value: dict[str, Any]) -> None:
    with bound_engine():
        engine.validate_authorization(value)


def execute(authorization: dict[str, Any], *, job_uid: str, pod_uid: str) -> dict[str, Any]:
    with bound_engine():
        return engine.execute(authorization, job_uid=job_uid, pod_uid=pod_uid)


def render(root: Path, commit: str, authorization: dict[str, Any]) -> dict[str, Any]:
    with bound_engine():
        return engine.render(root, commit, authorization)


def build_held() -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema_version": "fleet-glm53-dedicated-v31-incluster-parity-held-v1",
        "status": "PASSED_HELD_NO_LAUNCH",
        "server_title": SERVER_TITLE,
        "server_run_dir": SERVER_RUN_DIR,
        "job_name": JOB_NAME,
        "result_root": str(RESULT_ROOT),
        "nested_container_network": "host",
        "shared_temp_root": PARITY_TEMP_ROOT,
        "endpoint_origin_must_equal_internal_service_origin": True,
        "qualification_launch_authorized": False,
        "scored_launch_authorized": False,
        "api_mutation_calls": 0,
        "protected_content_included": False,
    }
    body["receipt_sha256"] = crypto.digest_without(body, "receipt_sha256")
    return body


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "plan"))
    parser.add_argument("--authorization", type=Path)
    args = parser.parse_args(argv)
    if args.command == "plan":
        print(json.dumps(build_held(), sort_keys=True, separators=(",", ":")))
        return 0
    if args.authorization is None:
        parser.error("run requires --authorization")
    authorization = json.loads(args.authorization.read_text())
    result = execute(
        authorization,
        job_uid=os.environ.get("JOB_UID", ""),
        pod_uid=os.environ.get("POD_UID", ""),
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
