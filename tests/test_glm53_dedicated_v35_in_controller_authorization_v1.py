from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

import pytest

from evals.fleet import exact_pass4_crypto as crypto
from evals.fleet import glm53_dedicated_v35_controller_package_v1 as package
from evals.fleet import glm53_dedicated_v35_controller_v1 as controller
from evals.fleet import glm53_dedicated_v35_create_v1 as server
from evals.fleet import glm53_dedicated_v35_launch_v1 as launch
from evals.fleet import glm53_dedicated_v35_live_authorization_v1 as live

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "1" * 40


def _engine_receipt() -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema_version": controller.engine.RESULT_SCHEMA,
        "fleet_task_instance_calls": 0,
        "fleet_session_calls": 0,
        "verifier_calls": 0,
        "scoring_calls": 0,
        "qualification_launch_authorized": False,
        "scored_launch_authorized": False,
        "protected_content_included": False,
    }
    body["receipt_sha256"] = crypto.digest_without(body, "receipt_sha256")
    return body


def test_controller_rebuilds_authorization_after_delayed_pod_start(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seed = controller.build_seed(COMMIT)
    seed_path = tmp_path / "authorization.json"
    seed_path.write_text(json.dumps(seed))
    observation_started_at = 1_000.0
    simulated_submission_at = observation_started_at - 61.0
    observation_completed_at = 1_061.0
    assert observation_completed_at - simulated_submission_at > server.AUTH_MAX_AGE_SECONDS
    stale_rows = [{"name": "terminal-history"}]
    reconciliation = {"receipt_sha256": "sha256:reconciliation"}
    fresh = {
        "observed_at_epoch": observation_started_at,
        "receipt_sha256": "sha256:fresh-authorization",
    }
    completed = {
        "observed_at_epoch": observation_completed_at,
        "receipt_sha256": "sha256:fresh-at-completion",
    }
    calls: list[str] = []

    class StaleBackend:
        def list_runs(self) -> tuple[list[dict[str, Any]], int]:
            calls.append("stale-list")
            return stale_rows, 1

    monkeypatch.setattr(
        controller.stale,
        "build_reconciliation",
        lambda **kwargs: (
            (
                calls.append("stale-build"),
                reconciliation,
            )[1]
            if kwargs["now"] == observation_started_at
            and kwargs["rows_snapshot"] == (stale_rows, 1)
            else pytest.fail("stale reconciliation did not use the in-Pod observation time")
        ),
    )
    monkeypatch.setattr(
        controller.stale,
        "validate_reconciliation",
        lambda value, *, rows, now: (
            calls.append("stale-validate")
            if value == reconciliation and rows == stale_rows and now == observation_started_at
            else pytest.fail("stale reconciliation validation drifted")
        ),
    )

    def build_live(**kwargs: Any) -> dict[str, Any]:
        calls.append("live-build")
        assert kwargs["now"] == observation_started_at
        assert kwargs["stale_run_reconciliation"] == reconciliation
        assert kwargs["title"] == server.TITLE
        assert kwargs["run_dir"] == server.RUN_DIR
        assert kwargs["control_result_path"] == server.RESULT_PATH
        return fresh

    monkeypatch.setattr(controller.live, "build_live_authorization", build_live)
    monkeypatch.setattr(
        controller.live,
        "bind_observation_completion",
        lambda value, *, completed_at_epoch: (
            calls.append("bind-observation-completion"),
            completed
            if value == fresh and completed_at_epoch == observation_completed_at
            else pytest.fail("authorization was not rebound at observation completion"),
        )[1],
    )
    monkeypatch.setattr(
        controller.server,
        "validate_authorization",
        lambda value: (
            calls.append("authorization-validate")
            if value == completed
            else pytest.fail("unexpected authorization")
        ),
    )

    def engine_run(root: Path, commit: str, authorization_path: Path) -> dict[str, Any]:
        calls.append("create-controller")
        assert root == tmp_path
        assert commit == COMMIT
        assert authorization_path.parent != seed_path.parent
        assert json.loads(authorization_path.read_text()) == completed
        return _engine_receipt()

    monkeypatch.setattr(controller.engine, "run", engine_run)
    monkeypatch.setenv("FLEET_API_KEY", "test-only")
    result = controller.run(
        tmp_path,
        COMMIT,
        seed_path,
        stale_backend_factory=StaleBackend,
        live_backend_factory=lambda: object(),
        now=iter((observation_started_at, observation_completed_at)).__next__,
    )
    assert result["schema_version"] == controller.RESULT_SCHEMA
    assert result["receipt_sha256"] == crypto.digest_without(result, "receipt_sha256")
    assert calls == [
        "stale-list",
        "stale-build",
        "stale-validate",
        "live-build",
        "bind-observation-completion",
        "authorization-validate",
        "create-controller",
    ]
    assert seed == controller.build_seed(COMMIT)
    assert "observed_at_epoch" not in seed


def test_invalid_or_rehashed_controller_seed_fails_before_live_reads(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seed = controller.build_seed(COMMIT)
    seed["server_title"] = "attacker"
    seed["receipt_sha256"] = crypto.digest_without(seed, "receipt_sha256")
    path = tmp_path / "authorization.json"
    path.write_text(json.dumps(seed))
    called = False

    def stale_factory() -> object:
        nonlocal called
        called = True
        return object()

    with pytest.raises(controller.ControllerError, match="seed_invalid"):
        controller.run(
            tmp_path,
            COMMIT,
            path,
            stale_backend_factory=stale_factory,
            live_backend_factory=lambda: object(),
        )
    assert called is False


def test_live_freshness_is_rebound_only_after_strict_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = {
        "observed_at_epoch": 100.0,
        "live_observation": {
            "observed_at_epoch": 100.0,
            "receipt_sha256": "sha256:old-live",
        },
        "receipt_sha256": "sha256:old-authorization",
    }
    validated: list[float] = []

    def validate(value: dict[str, Any], authorization: dict[str, Any]) -> None:
        assert value is authorization["live_observation"]
        validated.append(float(authorization["observed_at_epoch"]))

    monkeypatch.setattr(live, "validate_live_observation", validate)
    rebound = live.bind_observation_completion(original, completed_at_epoch=161.0)
    assert original["observed_at_epoch"] == 100.0
    assert rebound["observed_at_epoch"] == 161.0
    assert rebound["live_observation"]["observed_at_epoch"] == 161.0
    assert rebound["live_observation"]["receipt_sha256"] == crypto.digest_without(
        rebound["live_observation"], "receipt_sha256"
    )
    assert rebound["receipt_sha256"] == crypto.digest_without(rebound, "receipt_sha256")
    assert validated == [100.0, 161.0]
    for invalid in (99.0, float("nan"), float("inf")):
        with pytest.raises(live.LiveAuthorizationError, match="completion_invalid"):
            live.bind_observation_completion(original, completed_at_epoch=invalid)


def test_v35_identities_are_fresh_and_keep_v34_create_guarantees() -> None:
    from evals.fleet import glm53_dedicated_v35_incluster_parity_v1 as parity
    from evals.fleet import glm53_dedicated_v35_watchdog_package_v1 as watchdog

    assert server.API_NAME == "glm53-tp8-v35"
    assert server.TITLE.endswith("-v35")
    assert server.RUN_DIR.endswith("-v35")
    assert server.RESULT_PATH.endswith("v35-create-control/CREATED.json")
    assert package.JOB_NAME.endswith("v35-create-watchdog-parity-controller-v1")
    assert parity.JOB_NAME.endswith("v35-incluster-actual-parity-v1")
    assert watchdog.JOB_NAME.endswith("v35-request-watchdog-v1")
    assert server.POST_CREATE_ATTEMPTS == 60
    assert server.POST_CREATE_TRANSIENT_ERRORS == 3
    assert server.POST_CREATE_CLEANUP_ATTEMPTS == 3
    assert server.build_held()["response_body_identity_required"] is False
    assert server.build_held()["scored_launch_authorized"] is False


def test_v34_terminal_evidence_is_sanitized_and_digest_valid() -> None:
    path = (
        ROOT / "docs/evidence/glm53-study/"
        "2026-09-06-glm53-dedicated-v34-controller-authorization-stale-terminal-v1.json"
    )
    value = json.loads(path.read_text())
    assert value["receipt_sha256"] == crypto.digest_without(value, "receipt_sha256")
    assert value["status"] == "TERMINAL_INFRASTRUCTURE_FAILED_PRE_CREATE"
    assert value["failure_code"] == "v34_create_authorization_invalid"
    assert value["launch_receipt_sha256"] == (
        "sha256:91d05535d90fe36c68f516d9cb42501693a4273d10de30c412cf9888318aacf8"
    )
    assert value["jobs_api_create_calls"] == 0
    assert value["server_matching_gpu_pods"] == 0
    assert value["protected_content_included"] is False
    assert all(
        value[key] == 0
        for key in (
            "fleet_task_instance_calls",
            "fleet_session_calls",
            "verifier_calls",
            "scoring_calls",
        )
    )


class FakeKubernetes:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], dict[str, Any]] = {}
        self.dry_runs = 0
        self.creates = 0

    def load_fleet_token(self) -> str:
        pytest.fail("launcher must not build a reusable live authorization")

    def get(self, kind: str, name: str) -> dict[str, Any] | None:
        return self.objects.get((kind, name))

    def dry_run(self, _objects: dict[str, Any]) -> None:
        self.dry_runs += 1

    def create(self, _objects: dict[str, Any]) -> None:
        self.creates += 1
        for kind, name in launch._identities():  # noqa: SLF001
            self.objects[(kind, name)] = {
                "apiVersion": "v1" if kind == "configmap" else "batch/v1",
                "kind": "ConfigMap" if kind == "configmap" else "Job",
                "metadata": {
                    "name": name,
                    "namespace": launch.engine.NAMESPACE,
                    "uid": str(uuid.uuid4()),
                },
            }


def test_launcher_packages_seed_not_expiring_authorization(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source_manifest = {"package_sha256": launch.FROZEN_PACKAGE_SHA256}
    seed = controller.build_seed(launch.FROZEN_PACKAGE_COMMIT)
    rendered = {
        "objects": {
            "apiVersion": "v1",
            "kind": "List",
            "items": [
                {"data": {"package.json": json.dumps(source_manifest)}},
                {"data": {"authorization.json": json.dumps(seed)}},
                {},
            ],
        }
    }
    monkeypatch.setattr(package, "render", lambda *_args: rendered)
    kubernetes = FakeKubernetes()
    monkeypatch.setattr(launch.engine, "_write_once", lambda *_args: None)
    result = launch.execute(
        root=tmp_path,
        output=tmp_path / "SUBMITTED.json",
        kubernetes=kubernetes,
        now=lambda: 123.0,
    )
    assert result["authorization_built_inside_controller"] is True
    assert result["controller_seed_receipt_sha256"] == seed["receipt_sha256"]
    assert kubernetes.dry_runs == 1
    assert kubernetes.creates == 1
    assert "observed_at_epoch" not in seed


def test_materialized_package_imports_fresh_controller_and_builds_fresh_auth(
    tmp_path: Path,
) -> None:
    configmap = package.build_source_configmap(ROOT, launch.FROZEN_PACKAGE_COMMIT)
    manifest = json.loads(configmap["data"]["package.json"])
    assert manifest["package_sha256"] == launch.FROZEN_PACKAGE_SHA256
    projected = tmp_path / "work"
    for key, source in configmap["data"].items():
        if "__SLASH__" not in key:
            continue
        target = projected / key.replace("__SLASH__", "/")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from evals.fleet import glm53_dedicated_v35_controller_v1 as c; "
                "from evals.fleet import glm53_dedicated_v35_create_v1 as s; "
                "assert c.build_seed('1' * 40)['fresh_in_controller_authorization_required']; "
                "assert s.API_RUN_ID_RE.fullmatch('glm53-tp8-v35-e4888aa7')"
            ),
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(projected)},
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    job = package.build_job(launch.FROZEN_PACKAGE_COMMIT)
    command = job["spec"]["template"]["spec"]["containers"][0]["command"][-1]
    assert command.count("glm53_dedicated_v35_controller_v1") == 1
    assert "glm53_dedicated_v34_controller_v1" not in command
