"""Create, watchdog, and score-free in-cluster parity controller for GLM v30."""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from evals.fleet import exact_pass4_crypto as crypto
from evals.fleet import glm53_dedicated_v23_scorefree_package_v1 as source_engine
from evals.fleet import glm53_dedicated_v24_watchdog_live_release_v1 as live_engine
from evals.fleet import glm53_dedicated_v30_create_v1 as server
from evals.fleet import glm53_dedicated_v30_incluster_parity_v1 as parity
from evals.fleet import glm53_dedicated_v30_watchdog_live_release_v1 as adapter

READY_TIMEOUT_SECONDS = 600
PARITY_TIMEOUT_SECONDS = 600
POLL_SECONDS = 5


class ControllerError(RuntimeError):
    """The v30 score-free lifecycle controller failed closed."""


@contextmanager
def materialized_source(root: Path) -> Iterator[None]:
    """Use the immutable projected package instead of requiring a Git checkout."""

    original_watchdog = source_engine._source  # noqa: SLF001
    original_parity = parity._source  # noqa: SLF001

    def read(_root: Path, _commit: str, path: str) -> bytes:
        target = root / path
        if not target.is_file():
            raise ControllerError(f"v30_materialized_source_absent:{path}")
        return target.read_bytes()

    source_engine._source = read  # type: ignore[assignment]  # noqa: SLF001
    parity._source = read  # type: ignore[assignment]  # noqa: SLF001
    try:
        yield
    finally:
        source_engine._source = original_watchdog  # type: ignore[assignment]  # noqa: SLF001
        parity._source = original_parity  # type: ignore[assignment]  # noqa: SLF001


def _load_authorization(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ControllerError("v30_controller_authorization_invalid") from exc
    if not isinstance(value, dict):
        raise ControllerError("v30_controller_authorization_invalid")
    server.validate_authorization(value)
    return value


def _wait_ready() -> None:
    deadline = time.monotonic() + READY_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if Path(server.READY_PATH).is_file():
            return
        time.sleep(POLL_SECONDS)
    raise ControllerError("v30_application_ready_timeout")


def _uuid(value: Any) -> bool:
    try:
        return uuid.UUID(str(value)).int != 0
    except (ValueError, TypeError, AttributeError):
        return False


def build_parity_authorization(
    binding: dict[str, Any],
    live: dict[str, Any],
    watchdog_launch: dict[str, Any],
    *,
    observed_at_epoch: float,
) -> dict[str, Any]:
    runtime = watchdog_launch.get("runtime", {})
    if (
        not isinstance(runtime, dict)
        or not _uuid(runtime.get("watchdog_job_uid"))
        or not _uuid(runtime.get("watchdog_pod_uid"))
        or not isinstance(runtime.get("active_receipt_sha256"), str)
    ):
        raise ControllerError("v30_watchdog_active_binding_invalid")
    body: dict[str, Any] = {
        "schema_version": parity.SCHEMA,
        "status": "AUTHORIZED_SCORE_FREE_INCLUSTER_PARITY",
        "observed_at_epoch": observed_at_epoch,
        "server_binding": binding,
        "server_ready_receipt_sha256": live["application_ready_receipt_sha256"],
        "watchdog_active_receipt_sha256": runtime["active_receipt_sha256"],
        "job_identity_absent": True,
        "result_root_absent": True,
        "cpu_priority_class": parity.CPU_PRIORITY_CLASS,
        "cpu_preemption_policy": "Never",
        "qualification_launch_authorized": True,
        "scored_launch_authorized": False,
        "fleet_task_instance_calls": 0,
        "fleet_session_calls": 0,
        "verifier_calls": 0,
        "scoring_calls": 0,
        "protected_content_included": False,
    }
    body["receipt_sha256"] = crypto.digest_without(body, "receipt_sha256")
    parity.validate_authorization(body)
    return body


def _assert_parity_absent() -> None:
    for kind, name in (
        ("jobs", parity.JOB_NAME),
        ("configmaps", parity.CONFIGMAP_NAME),
        ("configmaps", parity.AUTHORIZATION_CONFIGMAP_NAME),
    ):
        if live_engine._kubectl_optional(kind, name) is not None:  # noqa: SLF001
            raise ControllerError(f"v30_parity_identity_collision:{kind}:{name}")
    if parity.RESULT_ROOT.exists() or parity.RESULT_ROOT.is_symlink():
        raise ControllerError("v30_parity_result_collision")


def _validate_parity_result(value: dict[str, Any], binding: dict[str, Any]) -> None:
    expected_keys = {
        "schema_version",
        "status",
        "server_binding_sha256",
        "authorization_receipt_sha256",
        "parity_receipt_sha256",
        "job_uid",
        "pod_uid",
        "nested_container_network",
        "local_proxy_bind_address",
        "endpoint_origin",
        "fleet_task_instance_calls",
        "fleet_session_calls",
        "verifier_calls",
        "scoring_calls",
        "scored_launch_authorized",
        "protected_content_included",
        "receipt_sha256",
    }
    if (
        set(value) != expected_keys
        or value.get("schema_version") != parity.RESULT_SCHEMA
        or value.get("status") != "PASSED_NON_SCORED_INCLUSTER_PARITY"
        or value.get("server_binding_sha256")
        != crypto.sha256(crypto.canonical_json(binding))
        or not _uuid(value.get("job_uid"))
        or not _uuid(value.get("pod_uid"))
        or value.get("nested_container_network") != "host"
        or value.get("local_proxy_bind_address") != "127.0.0.1"
        or value.get("endpoint_origin") != binding.get("service_origin")
        or any(
            value.get(key) != 0
            for key in (
                "fleet_task_instance_calls",
                "fleet_session_calls",
                "verifier_calls",
                "scoring_calls",
            )
        )
        or value.get("scored_launch_authorized") is not False
        or value.get("protected_content_included") is not False
        or value.get("receipt_sha256") != crypto.digest_without(value, "receipt_sha256")
    ):
        raise ControllerError("v30_parity_result_invalid")


def _wait_parity(binding: dict[str, Any]) -> dict[str, Any]:
    deadline = time.monotonic() + PARITY_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        job = live_engine._kubectl_optional("jobs", parity.JOB_NAME)  # noqa: SLF001
        if job is not None:
            status = job.get("status", {})
            if status.get("failed", 0):
                raise ControllerError("v30_parity_job_failed")
            if status.get("succeeded") == 1:
                try:
                    result = json.loads((parity.RESULT_ROOT / "RESULT.json").read_text())
                except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ControllerError("v30_parity_result_absent_or_invalid") from exc
                if not isinstance(result, dict):
                    raise ControllerError("v30_parity_result_absent_or_invalid")
                _validate_parity_result(result, binding)
                return result
        time.sleep(POLL_SECONDS)
    raise ControllerError("v30_parity_timeout")


def run(root: Path, commit: str, authorization_path: Path) -> dict[str, Any]:
    if not os.environ.get("FLEET_API_KEY"):
        raise ControllerError("v30_controller_fleet_credential_absent")
    authorization = _load_authorization(authorization_path)
    created = server.create_once(authorization, result_path=Path(server.RESULT_PATH))
    api_run_id = created["api_run_id"]
    parity_rendered: dict[str, Any] | None = None
    try:
        _wait_ready()
        priorities = live_engine._kubectl_json(  # noqa: SLF001
            "priorityclasses.scheduling.k8s.io"
        ).get("items", [])
        with materialized_source(root):
            watchdog = adapter.launch(
                root,
                commit,
                api_run_id,
                priority_classes=priorities,
            )
            binding, live = adapter.observe_live(api_run_id)
            _assert_parity_absent()
            parity_authorization = build_parity_authorization(
                binding,
                live,
                watchdog,
                observed_at_epoch=time.time(),
            )
            parity_rendered = parity.render(root, commit, parity_authorization)
            parity_objects = [
                {
                    "kind": item["kind"],
                    "name": item["metadata"]["name"],
                    "outcome": live_engine._create_or_verify_exact(item),  # noqa: SLF001
                }
                for item in parity_rendered["objects"]["items"]
            ]
        parity_result = _wait_parity(binding)
        receipt: dict[str, Any] = {
            "schema_version": "fleet-glm53-dedicated-v30-controller-result-v1",
            "status": "WATCHDOG_ACTIVE_INCLUSTER_PARITY_PASSED_NON_SCORED",
            "api_run_id": api_run_id,
            "server_binding_sha256": crypto.sha256(crypto.canonical_json(binding)),
            "watchdog_launch_receipt_sha256": watchdog["receipt_sha256"],
            "parity_objects": parity_objects,
            "parity_result_receipt_sha256": parity_result["receipt_sha256"],
            "qualification_launch_authorized": False,
            "scored_launch_authorized": False,
            "fleet_task_instance_calls": 0,
            "fleet_session_calls": 0,
            "verifier_calls": 0,
            "scoring_calls": 0,
            "protected_content_included": False,
        }
        receipt["receipt_sha256"] = crypto.digest_without(receipt, "receipt_sha256")
        return receipt
    except Exception:
        if parity_rendered is not None:
            live_engine._rollback_watcher_objects(parity_rendered)  # noqa: SLF001
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
