"""Held create-once launcher for the reviewed GLM v35 controller package."""

from __future__ import annotations

import argparse
import json
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from evals.fleet import exact_pass4_crypto as crypto
from evals.fleet import glm53_dedicated_v33_launch_v1 as engine
from evals.fleet import glm53_dedicated_v35_controller_package_v1 as package
from evals.fleet import glm53_dedicated_v35_controller_v1 as controller
from evals.fleet import glm53_dedicated_v35_create_v1 as server
from evals.fleet import glm53_dedicated_v35_live_authorization_v1 as live

FROZEN_PACKAGE_COMMIT = "dbb94e9f45827c745a2f281ea4a3c3265924f1d2"
FROZEN_PACKAGE_SHA256 = "sha256:eb9c158aafd1f6d2aaf125b93d1c9b37e9545f9af27529db1104f4d4e3f156a2"
HELD_SCHEMA = "fleet-glm53-dedicated-v35-launch-held-v1"
RESULT_SCHEMA = "fleet-glm53-dedicated-v35-controller-submission-v1"
LaunchError = engine.LaunchError
KubernetesBackend = engine.KubernetesBackend
SystemKubernetesBackend = engine.SystemKubernetesBackend
_LOCK = threading.Lock()


@contextmanager
def bound_engine() -> Iterator[None]:
    if not _LOCK.acquire(blocking=False):
        raise LaunchError("v35_launcher_engine_already_bound")
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
    source = package.build_source_configmap(root, FROZEN_PACKAGE_COMMIT)
    manifest = json.loads(source["data"]["package.json"])
    if manifest.get("package_sha256") != FROZEN_PACKAGE_SHA256:
        raise LaunchError("v35_frozen_package_digest_invalid")
    body: dict[str, Any] = {
        "schema_version": HELD_SCHEMA,
        "status": "PASSED_HELD_NO_LAUNCH",
        "package_commit": FROZEN_PACKAGE_COMMIT,
        "package_sha256": FROZEN_PACKAGE_SHA256,
        "controller_job_name": package.JOB_NAME,
        "source_configmap_name": package.CONFIGMAP_NAME,
        "seed_configmap_name": package.AUTHORIZATION_CONFIGMAP_NAME,
        "server_title": server.TITLE,
        "server_run_dir": server.RUN_DIR,
        "authorization_built_inside_controller": True,
        "launcher_authorization_reuse_forbidden": True,
        "fresh_stale_reconciliation_inside_controller": True,
        "fresh_global_kubernetes_sfs_capacity_checks_inside_controller": True,
        "immediate_create_preflight_after_authorization": True,
        "response_body_identity_required": False,
        "exhaustive_pre_post_jobs_api_reconciliation_required": True,
        "ambiguous_post_create_identity_release_required": True,
        "post_create_transient_error_limit": server.POST_CREATE_TRANSIENT_ERRORS,
        "post_create_cleanup_attempt_limit": server.POST_CREATE_CLEANUP_ATTEMPTS,
        "all_safe_candidates_release_attempted": True,
        "result_parent_and_write_failure_release_required": True,
        "bounded_exact_object_preflight_count": len(_identities()),
        "server_side_dry_run_required": True,
        "repeat_exact_object_preflight_before_create": True,
        "idle_release_seconds": 600,
        "api_mutation_calls": 0,
        "kubernetes_mutation_calls": 0,
        "fleet_task_instance_calls": 0,
        "fleet_session_calls": 0,
        "verifier_calls": 0,
        "scoring_calls": 0,
        "protected_content_included": False,
        "server_launch_authorized": False,
        "qualification_launch_authorized": False,
        "scored_launch_authorized": False,
    }
    body["receipt_sha256"] = crypto.digest_without(body, "receipt_sha256")
    return body


def execute(
    *,
    root: Path,
    output: Path,
    kubernetes: KubernetesBackend,
    now: Callable[[], float],
) -> dict[str, Any]:
    if output.exists():
        raise LaunchError("v35_launch_result_already_exists")
    with bound_engine():
        engine._require_absent(kubernetes)  # noqa: SLF001
    seed = controller.build_seed(FROZEN_PACKAGE_COMMIT)
    rendered = package.render(root, FROZEN_PACKAGE_COMMIT, seed)
    objects = rendered["objects"]
    source_manifest = json.loads(objects["items"][0]["data"]["package.json"])
    if source_manifest.get("package_sha256") != FROZEN_PACKAGE_SHA256:
        raise LaunchError("v35_frozen_package_digest_invalid")
    with bound_engine():
        engine._require_absent(kubernetes)  # noqa: SLF001
        kubernetes.dry_run(objects)
        engine._require_absent(kubernetes)  # noqa: SLF001
        kubernetes.create(objects)
        created = {name: kubernetes.get(kind, name) for kind, name in _identities()}
        if any(value is None for value in created.values()):
            raise LaunchError("v35_kubernetes_create_unknown_reconcile_do_not_retry")
        object_uids = {
            name: engine._uid(created[name], kind=kind, name=name)  # type: ignore[arg-type] # noqa: SLF001
            for kind, name in _identities()
        }
    body: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "status": "CONTROLLER_SUBMITTED_CREATE_ONCE",
        "observed_at_epoch": now(),
        "package_commit": FROZEN_PACKAGE_COMMIT,
        "package_sha256": FROZEN_PACKAGE_SHA256,
        "controller_seed_receipt_sha256": seed["receipt_sha256"],
        "controller_job_name": package.JOB_NAME,
        "object_uids": object_uids,
        "server_title": server.TITLE,
        "server_run_dir": server.RUN_DIR,
        "authorization_built_inside_controller": True,
        "jobs_api_mutation_calls": 0,
        "kubernetes_create_calls": 1,
        "fleet_task_instance_calls": 0,
        "fleet_session_calls": 0,
        "verifier_calls": 0,
        "scoring_calls": 0,
        "protected_content_included": False,
        "qualification_launch_authorized": False,
        "scored_launch_authorized": False,
    }
    body["receipt_sha256"] = crypto.digest_without(body, "receipt_sha256")
    engine._write_once(output, body)  # noqa: SLF001
    return body


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
        raise LaunchError("v35_launch_output_required")
    result = execute(
        root=args.root,
        output=args.output,
        kubernetes=SystemKubernetesBackend(),
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
