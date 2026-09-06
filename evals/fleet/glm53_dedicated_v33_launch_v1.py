"""Create-once launcher for the reviewed GLM v33 controller package.

The default command is held and performs no live reads or writes.  ``--execute``
loads the Fleet credential from the Kubernetes Secret directly into process
memory, builds a fresh stale-run reconciliation followed by a fresh v33 live
authorization, and creates the immutable controller objects exactly once.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import time
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Protocol

from evals.fleet import exact_pass4_crypto as crypto
from evals.fleet import glm53_dedicated_v32_stale_run_reconciliation_v1 as stale
from evals.fleet import glm53_dedicated_v33_controller_package_v1 as package
from evals.fleet import glm53_dedicated_v33_create_v1 as server
from evals.fleet import glm53_dedicated_v33_live_authorization_v1 as live

NAMESPACE = "fleet-train-jobs"
FLEET_SECRET_NAME = "chris-cyber-opencode-evals-v2"
FLEET_SECRET_KEY = "FLEET_API_KEY"
FROZEN_PACKAGE_COMMIT = "5087b2754be38b64aa403df1b6689218ebe018ae"
FROZEN_PACKAGE_SHA256 = (
    "sha256:9a1b3e8e61f711cb98ba567ffc223f17846efd5d8703ea97bebcaad3d028ed99"
)
HELD_SCHEMA = "fleet-glm53-dedicated-v33-launch-held-v1"
RESULT_SCHEMA = "fleet-glm53-dedicated-v33-controller-submission-v1"


class LaunchError(RuntimeError):
    """The v33 controller launch failed closed."""


class KubernetesBackend(Protocol):
    def load_fleet_token(self) -> str: ...

    def get(self, kind: str, name: str) -> dict[str, Any] | None: ...

    def dry_run(self, objects: Mapping[str, Any]) -> None: ...

    def create(self, objects: Mapping[str, Any]) -> None: ...


class SystemKubernetesBackend:
    """Bounded exact Kubernetes reads and one create operation."""

    @staticmethod
    def _run(args: list[str], *, stdin: bytes | None = None) -> subprocess.CompletedProcess[bytes]:
        try:
            return subprocess.run(
                ["kubectl", "-n", NAMESPACE, *args],
                input=stdin,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except OSError as exc:
            raise LaunchError("v33_kubectl_unavailable") from exc

    def load_fleet_token(self) -> str:
        completed = self._run(
            [
                "get",
                "secret",
                FLEET_SECRET_NAME,
                "-o",
                f"jsonpath={{.data.{FLEET_SECRET_KEY}}}",
            ]
        )
        if completed.returncode != 0 or not completed.stdout:
            raise LaunchError("v33_cluster_secret_absent")
        try:
            token = base64.b64decode(completed.stdout, validate=True).decode()
        except (UnicodeDecodeError, ValueError) as exc:
            raise LaunchError("v33_cluster_secret_invalid") from exc
        if not token or any(character.isspace() for character in token):
            raise LaunchError("v33_cluster_secret_invalid")
        return token

    def get(self, kind: str, name: str) -> dict[str, Any] | None:
        completed = self._run(
            ["get", kind, name, "--ignore-not-found", "-o", "json"]
        )
        if completed.returncode != 0:
            raise LaunchError("v33_kubernetes_exact_get_failed")
        if not completed.stdout.strip():
            return None
        try:
            value = json.loads(completed.stdout)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise LaunchError("v33_kubernetes_exact_get_invalid") from exc
        if not isinstance(value, dict):
            raise LaunchError("v33_kubernetes_exact_get_invalid")
        return value

    def dry_run(self, objects: Mapping[str, Any]) -> None:
        completed = self._run(
            ["create", "--dry-run=server", "-f", "-", "-o", "name"],
            stdin=crypto.canonical_json(dict(objects)),
        )
        if completed.returncode != 0:
            raise LaunchError("v33_kubernetes_server_dry_run_failed")

    def create(self, objects: Mapping[str, Any]) -> None:
        completed = self._run(
            ["create", "-f", "-", "-o", "name"],
            stdin=crypto.canonical_json(dict(objects)),
        )
        if completed.returncode != 0:
            raise LaunchError("v33_kubernetes_create_unknown_reconcile_do_not_retry")


@contextmanager
def _bound_token(token: str) -> Iterator[None]:
    prior = os.environ.get("FLEET_API_KEY")
    os.environ["FLEET_API_KEY"] = token
    try:
        yield
    finally:
        if prior is None:
            os.environ.pop("FLEET_API_KEY", None)
        else:
            os.environ["FLEET_API_KEY"] = prior


def _identities() -> tuple[tuple[str, str], ...]:
    return (
        ("configmap", package.CONFIGMAP_NAME),
        ("configmap", package.AUTHORIZATION_CONFIGMAP_NAME),
        ("job", package.JOB_NAME),
    )


def _require_absent(backend: KubernetesBackend) -> None:
    if any(backend.get(kind, name) is not None for kind, name in _identities()):
        raise LaunchError("v33_controller_identity_already_exists")


def _uid(value: Mapping[str, Any], *, kind: str, name: str) -> str:
    metadata = value.get("metadata")
    if not isinstance(metadata, dict):
        raise LaunchError("v33_created_object_identity_invalid")
    candidate = metadata.get("uid")
    try:
        valid = uuid.UUID(str(candidate)).int != 0
    except (TypeError, ValueError):
        valid = False
    if (
        value.get("kind", "").lower() != kind
        or metadata.get("name") != name
        or metadata.get("namespace") != NAMESPACE
        or not valid
    ):
        raise LaunchError("v33_created_object_identity_invalid")
    return str(candidate)


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(crypto.canonical_json(dict(value)) + b"\n")


def build_held(root: Path) -> dict[str, Any]:
    source = package.build_source_configmap(root, FROZEN_PACKAGE_COMMIT)
    manifest = json.loads(source["data"]["package.json"])
    if manifest.get("package_sha256") != FROZEN_PACKAGE_SHA256:
        raise LaunchError("v33_frozen_package_digest_invalid")
    body: dict[str, Any] = {
        "schema_version": HELD_SCHEMA,
        "status": "PASSED_HELD_NO_LAUNCH",
        "package_commit": FROZEN_PACKAGE_COMMIT,
        "package_sha256": FROZEN_PACKAGE_SHA256,
        "fleet_secret_name": FLEET_SECRET_NAME,
        "fleet_secret_key": FLEET_SECRET_KEY,
        "controller_job_name": package.JOB_NAME,
        "source_configmap_name": package.CONFIGMAP_NAME,
        "authorization_configmap_name": package.AUTHORIZATION_CONFIGMAP_NAME,
        "server_title": server.TITLE,
        "server_run_dir": server.RUN_DIR,
        "fresh_stale_reconciliation_required": True,
        "fresh_live_authorization_required": True,
        "bounded_exact_object_preflight_count": len(_identities()),
        "server_side_dry_run_required": True,
        "repeat_exact_object_preflight_before_create": True,
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
    stale_backend_factory: Callable[[], Any] = stale.SystemBackend,
    live_backend_factory: Callable[[], Any] = live.SystemBackend,
    now: Callable[[], float] = time.time,
) -> dict[str, Any]:
    if output.exists():
        raise LaunchError("v33_launch_result_already_exists")
    _require_absent(kubernetes)
    token = kubernetes.load_fleet_token()
    try:
        with _bound_token(token):
            stale_backend = stale_backend_factory()
            rows, pages = stale_backend.list_runs()
            reconciliation = stale.build_reconciliation(
                backend=stale_backend, rows_snapshot=(rows, pages)
            )
            stale.validate_reconciliation(reconciliation, rows=rows)
            authorization = live.build_live_authorization(
                backend=live_backend_factory(),
                payload=server.payload(),
                title=server.TITLE,
                run_dir=server.RUN_DIR,
                control_result_path=server.RESULT_PATH,
                request_sha256=server.request_sha256(),
                priority_class=server.payload()["priority_class"],
                stale_run_reconciliation=reconciliation,
            )
    finally:
        token = ""
    server.validate_authorization(authorization)
    rendered = package.render(root, FROZEN_PACKAGE_COMMIT, authorization)
    objects = rendered["objects"]
    source_manifest = json.loads(objects["items"][0]["data"]["package.json"])
    if source_manifest.get("package_sha256") != FROZEN_PACKAGE_SHA256:
        raise LaunchError("v33_frozen_package_digest_invalid")
    _require_absent(kubernetes)
    kubernetes.dry_run(objects)
    _require_absent(kubernetes)
    server.validate_authorization(authorization)
    kubernetes.create(objects)
    created = {
        name: kubernetes.get(kind, name) for kind, name in _identities()
    }
    if any(value is None for value in created.values()):
        raise LaunchError("v33_kubernetes_create_unknown_reconcile_do_not_retry")
    object_uids = {
        name: _uid(created[name], kind=kind, name=name)  # type: ignore[arg-type]
        for kind, name in _identities()
    }
    body: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "status": "CONTROLLER_SUBMITTED_CREATE_ONCE",
        "observed_at_epoch": now(),
        "package_commit": FROZEN_PACKAGE_COMMIT,
        "package_sha256": FROZEN_PACKAGE_SHA256,
        "stale_reconciliation_receipt_sha256": reconciliation["receipt_sha256"],
        "authorization_receipt_sha256": authorization["receipt_sha256"],
        "controller_job_name": package.JOB_NAME,
        "object_uids": object_uids,
        "server_title": server.TITLE,
        "server_run_dir": server.RUN_DIR,
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
    _write_once(output, body)
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
        raise LaunchError("v33_launch_output_required")
    result = execute(
        root=args.root,
        output=args.output,
        kubernetes=SystemKubernetesBackend(),
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
