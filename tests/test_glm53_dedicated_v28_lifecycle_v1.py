import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from evals.fleet import exact_pass4_crypto as crypto
from evals.fleet import glm53_dedicated_v24_server_v1 as v24
from evals.fleet import glm53_dedicated_v24_watchdog_live_release_v1 as live_engine
from evals.fleet import glm53_dedicated_v27_create_v1 as v27
from evals.fleet import glm53_dedicated_v28_controller_package_v1 as controller_package
from evals.fleet import glm53_dedicated_v28_controller_v1 as controller
from evals.fleet import glm53_dedicated_v28_create_v1 as server
from evals.fleet import glm53_dedicated_v28_watchdog_live_release_v1 as adapter
from evals.fleet import glm53_dedicated_v28_watchdog_package_v1 as watchdog_package

ROOT = Path(__file__).resolve().parents[1]
NOW = 2_000_000_000.0
COMMIT = "8c875af2126d4794843bee2fd06946dc25984415"
EVIDENCE_ROOT = ROOT / "docs/evidence/glm53-study"


def authorization() -> dict:
    value = {
        "schema_version": server.SCHEMA,
        "status": "PASSED_LIVE_CREATE_GATES",
        "observed_at_epoch": time.time(),
        "server_title": server.TITLE,
        "server_run_dir": server.RUN_DIR,
        "request_sha256": server.request_sha256(),
        "preview_http_status": 200,
        "jobs_api_title_matches": 0,
        "jobs_api_run_dir_matches": 0,
        "kubernetes_identity_or_remnant_matches": 0,
        "sfs_run_dir_absent": True,
        "control_result_absent": True,
        "active_dedicated_nodes": 0,
        "active_dedicated_gpus": 0,
        "planned_nodes_after_create": 1,
        "planned_gpus_after_create": 8,
        "priority_class": v24.PRIORITY_CLASS,
        "preemption_policy": v24.PREEMPTION_POLICY,
        "server_launch_authorized": True,
        "watchdog_handoff_required_immediately": True,
        "qualification_launch_authorized": False,
        "scored_launch_authorized": False,
        "api_mutation_calls": 0,
        "protected_content_included": False,
    }
    value["receipt_sha256"] = crypto.digest_without(value, "receipt_sha256")
    return value


def test_v28_server_identity_is_fresh_and_restores_v27_engine() -> None:
    before = (v27.TITLE, v27.RUN_DIR, v27.READY_SCHEMA)
    value = server.payload()
    server.validate_payload(value)
    assert value["title"] == server.TITLE
    assert value["run_dir"] == server.RUN_DIR
    assert server.READY_SCHEMA in server._observer_source()
    assert v27.TITLE not in crypto.canonical_json(value).decode()
    assert before == (v27.TITLE, v27.RUN_DIR, v27.READY_SCHEMA)


def test_v28_watchdog_and_adapter_are_generation_exact_and_held() -> None:
    held = adapter.build_held(COMMIT)
    assert held["runtime_auth_schema"] == watchdog_package.RUNTIME_AUTH_SCHEMA
    assert held["live_release_schema"] == watchdog_package.LIVE_RELEASE_SCHEMA
    assert held["controller_side_fleet_credential_required"] is True
    assert held["server_launch_authorized"] is False
    assert held["watchdog_launch_authorized"] is False
    assert held["qualification_launch_authorized"] is False
    assert held["scored_launch_authorized"] is False
    assert held["receipt_sha256"] == crypto.digest_without(held, "receipt_sha256")


def test_controller_job_keeps_credential_through_ready_and_handoff() -> None:
    job = controller_package.build_job(COMMIT)
    spec = job["spec"]["template"]["spec"]
    container = spec["containers"][0]
    command = container["command"][-1]
    assert spec["priorityClassName"] == "fleet-serve-low"
    assert spec["preemptionPolicy"] == "Never"
    assert spec["restartPolicy"] == "Never"
    assert job["spec"]["ttlSecondsAfterFinished"] == 604800
    assert spec["initContainers"][0]["image"] == controller_package.KUBECTL_IMAGE
    assert container["resources"]["requests"] == {"cpu": "100m", "memory": "256Mi"}
    assert any(row["name"] == "FLEET_API_KEY" for row in container["env"])
    assert "glm53_dedicated_v28_controller_v1" in command
    assert "uv sync --project /workspace --frozen" in command
    assert "export PYTHONPATH=/workspace" in command
    assert command.index("cd /workspace") < command.index(
        "python -m evals.fleet.glm53_dedicated_v28_controller_v1"
    )
    assert controller_package.JOB_NAME.endswith("controller-v2")
    assert spec["volumes"][2]["persistentVolumeClaim"]["claimName"] == "sfs-shared"
    held = controller_package.build_held()
    assert held["credentialed_release_required"] is True
    assert held["server_launch_authorized"] is False
    assert held["receipt_sha256"] == crypto.digest_without(held, "receipt_sha256")


def test_controller_package_is_exact_commit_closed_and_create_once(tmp_path: Path) -> None:
    rendered = controller_package.render(ROOT, COMMIT, authorization())
    source, auth, job = rendered["objects"]["items"]
    package = json.loads(source["data"]["package.json"])
    assert package["package_commit"] == COMMIT
    assert set(package["files"]) == set(controller_package.FILES)
    assert package["credentialed_release_required"] is True
    assert package["score_free"] is True
    assert len(json.dumps(source)) < 1_000_000
    assert job["metadata"]["name"].endswith("controller-v2")
    assert "export PYTHONPATH=/workspace" in job["spec"]["template"]["spec"][
        "containers"
    ][0]["command"][-1]
    for relative in controller_package.FILES:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source["data"][relative.replace("/", "__SLASH__")])
    result = subprocess.run(
        [sys.executable, "-c", "import evals.fleet.glm53_dedicated_v28_controller_v1"],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(tmp_path)},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert auth["immutable"] is True
    assert job["metadata"]["name"] == controller_package.JOB_NAME
    assert rendered["server_launch_authorized"] is True
    assert rendered["watchdog_handoff_required"] is True
    assert rendered["qualification_launch_authorized"] is False
    assert rendered["scored_launch_authorized"] is False


def test_controller_releases_with_its_credential_when_handoff_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    auth_path = tmp_path / "authorization.json"
    auth_path.write_text(json.dumps(authorization()))
    monkeypatch.setenv("FLEET_API_KEY", "test-only")
    monkeypatch.setattr(
        server,
        "create_once",
        lambda *_args, **_kwargs: {"api_run_id": "ft-run-deadbeef"},
    )
    monkeypatch.setattr(controller, "_wait_ready", lambda: None)
    monkeypatch.setattr(
        live_engine,
        "_kubectl_json",
        lambda *_args: {"items": []},
    )
    monkeypatch.setattr(
        adapter,
        "launch",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("handoff failed")),
    )
    releases: list[str] = []
    monkeypatch.setattr(
        live_engine,
        "_release_local",
        lambda api_run_id: releases.append(api_run_id),
    )
    with pytest.raises(RuntimeError, match="handoff failed"):
        controller.run(tmp_path, COMMIT, auth_path)
    assert releases == ["ft-run-deadbeef"]


def test_controller_refuses_to_create_without_fleet_credential(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("FLEET_API_KEY", raising=False)
    with pytest.raises(controller.ControllerError, match="credential_absent"):
        controller.run(tmp_path, COMMIT, tmp_path / "absent.json")


def test_controller_materialized_source_restores_reader(tmp_path: Path) -> None:
    target = tmp_path / "evals/fleet/exact_pass4_crypto.py"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"exact bytes")
    original = controller.source_engine._source
    with controller.materialized_source(tmp_path):
        observed = controller.source_engine._source(
            tmp_path, COMMIT, "evals/fleet/exact_pass4_crypto.py"
        )
        assert observed == b"exact bytes"
    assert controller.source_engine._source is original


def test_controller_source_closure_has_no_protected_or_scoring_artifacts() -> None:
    names = set(controller_package.FILES)
    assert "evals/fleet/glm53_dedicated_v28_controller_v1.py" in names
    assert "evals/fleet/glm53_dedicated_v28_watchdog_live_release_v1.py" in names
    assert not any(
        name.startswith(("prompts/", "traces/", "scoring/")) for name in names
    )
    assert os.path.isabs(server.RUN_DIR)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("2026-09-06-glm53-dedicated-v28-create-wrapper-held-v1.json", server.build_held()),
        (
            "2026-09-06-glm53-dedicated-v28-watchdog-adapter-held-v2.json",
            adapter.build_held(COMMIT),
        ),
        (
            "2026-09-06-glm53-dedicated-v28-controller-held-v2.json",
            controller_package.build_held(),
        ),
    ],
)
def test_tracked_held_receipts_are_exact(name: str, expected: dict[str, object]) -> None:
    observed = json.loads((EVIDENCE_ROOT / name).read_text())
    assert observed == expected
    assert observed["receipt_sha256"] == crypto.digest_without(
        observed, "receipt_sha256"
    )


def test_failed_controller_v1_is_a_zero_effect_tombstone() -> None:
    observed = json.loads(
        (
            EVIDENCE_ROOT
            / "2026-09-06-glm53-dedicated-v28-controller-bootstrap-v1-failure.json"
        ).read_text()
    )
    assert observed["status"] == "FAILED_PRECREATE_ZERO_EFFECT"
    assert observed["failed_controller_job"].endswith("controller-v1")
    assert observed["server_gpu_objects_present"] is False
    assert observed["gpu_allocation_count"] == 0
    assert observed["task_session_verifier_scoring_calls"] == 0
    assert observed["retry_same_identity"] is False
    assert observed["receipt_sha256"] == crypto.digest_without(
        observed, "receipt_sha256"
    )
