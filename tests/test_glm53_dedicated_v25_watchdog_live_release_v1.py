import datetime
import json
import subprocess
from pathlib import Path

import pytest

from evals.fleet import exact_pass4_crypto as crypto
from evals.fleet import glm53_dedicated_v23_request_counter_watchdog_v1 as runtime
from evals.fleet import glm53_dedicated_v24_server_v1 as v24_server
from evals.fleet import glm53_dedicated_v24_watchdog_live_release_v1 as engine
from evals.fleet import glm53_dedicated_v24_watchdog_package_v1 as v24_package
from evals.fleet import glm53_dedicated_v25_create_v1 as server
from evals.fleet import glm53_dedicated_v25_watchdog_live_release_v1 as adapter
from evals.fleet import glm53_dedicated_v25_watchdog_package_v1 as package

ROOT = Path(__file__).resolve().parents[1]
NOW = 2_000_000_000.0
COMMIT = subprocess.run(
    ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True, capture_output=True
).stdout.strip()


def binding() -> dict:
    service = "ft-run-deadbeef-abcde-head-svc"
    return {
        "server_title": server.TITLE,
        "server_run_dir": server.RUN_DIR,
        "api_run_id": "ft-run-deadbeef",
        "rayjob_uid": "11111111-1111-4111-8111-111111111111",
        "workload_uid": "22222222-2222-4222-8222-222222222222",
        "head_pod_uid": "33333333-3333-4333-8333-333333333333",
        "service_uid": "44444444-4444-4444-8444-444444444444",
        "service_origin": f"http://{service}.fleet-train-jobs.svc:8000",
        "served_id": server.SERVED_ID,
        "model_revision": server.MODEL_REVISION,
        "context_length": server.CONTEXT_LENGTH,
    }


def priorities() -> list[dict]:
    return [
        {
            "metadata": {"name": "fleet-infra-quiet"},
            "value": -1000,
            "preemptionPolicy": "Never",
        },
        {
            "metadata": {"name": "fleet-serve-low"},
            "value": 100,
            "preemptionPolicy": "Never",
        },
    ]


def ready() -> dict:
    epoch = NOW - 15
    value = {
        "schema_version": server.READY_SCHEMA,
        "status": "APPLICATION_HEALTH_HTTP_200",
        "server_title": server.TITLE,
        "server_run_dir": server.RUN_DIR,
        "served_id": server.SERVED_ID,
        "model_revision": server.MODEL_REVISION,
        "context_length": server.CONTEXT_LENGTH,
        "ready_at_epoch": epoch,
        "ready_at_utc": datetime.datetime.fromtimestamp(
            epoch, datetime.UTC
        ).isoformat().replace("+00:00", "Z"),
        "health_http_status": 200,
        "prompts_traces_flags_scores_or_model_outputs_included": False,
    }
    value["receipt_sha256"] = crypto.digest_without(value, "receipt_sha256")
    return value


def live() -> dict:
    server_binding = binding()
    value = {
        "schema_version": adapter.LIVE_STATE_SCHEMA,
        "status": "READY_POST_CREATE_UID_BOUND",
        "observed_at_epoch": NOW - 5,
        "ready_at_epoch": NOW - 15,
        "server_binding": server_binding,
        "request_sha256": server.request_sha256(),
        "api_get_http_status": 200,
        "api_run_id_match_count": 1,
        "api_run_state": "RUNNING",
        "api_title_match_count": 1,
        "api_run_dir_match_count": 1,
        "rayjob_running": True,
        "rayjob_name": server_binding["api_run_id"],
        "rayjob_uid_match_count": 1,
        "raycluster_name": "ft-run-deadbeef-abcde",
        "raycluster_uid": "66666666-6666-4666-8666-666666666666",
        "workload_admitted": True,
        "workload_name": "rayjob-workload-deadbeef",
        "workload_finished": False,
        "workload_preemption_events": 0,
        "workload_uid_match_count": 1,
        "head_pod_phase": "Running",
        "head_pod_name": "ft-run-deadbeef-abcde-head-xyz12",
        "head_pod_ready": True,
        "head_pod_restarts": 0,
        "head_pod_uid_match_count": 1,
        "service_present": True,
        "service_name": "ft-run-deadbeef-abcde-head-svc",
        "service_uid_match_count": 1,
        "sfs_pvc_name": "sfs-claim",
        "sfs_pvc_uid": "77777777-7777-4777-8777-777777777777",
        "head_pod_sfs_mount_path": "/mnt/sfs",
        "metrics_http_status": 200,
        "activity_metric_families": list(runtime.ACTIVITY_METRICS),
        "jobs_api_credential_secret_name": "ft-run-deadbeef-fleet-key",
        "jobs_api_credential_secret_uid": "55555555-5555-4555-8555-555555555555",
        "jobs_api_credential_owner_rayjob_uid": server_binding["rayjob_uid"],
        "jobs_api_credential_key": "FLEET_API_KEY",
        "jobs_api_credential_probe_http_status": 200,
        "watchdog_job_match_count": 0,
        "watchdog_configmap_match_count": 0,
        "server_run_dir_exists": True,
        "watchdog_result_root_absent": True,
        "application_ready_receipt_sha256": ready()["receipt_sha256"],
        "fleet_task_instance_calls": 0,
        "fleet_session_calls": 0,
        "verifier_calls": 0,
        "scoring_calls": 0,
        "protected_content_included": False,
    }
    value["receipt_sha256"] = crypto.digest_without(value, "receipt_sha256")
    return value


def test_v25_payload_and_ready_receipt_are_exact() -> None:
    server.validate_payload(server.payload())
    server.validate_binding(binding())
    assert server.READY_SCHEMA in server._observer_source()
    assert v24_server.READY_SCHEMA not in server._observer_source()


def test_held_receipt_is_digest_valid_and_non_authorizing() -> None:
    value = adapter.build_held(COMMIT)
    assert value["server_request_sha256"] == server.request_sha256()
    assert value["exact_v25_contract_bound"] is True
    assert value["idle_release_seconds"] == 600
    assert value["server_launch_authorized"] is False
    assert value["watchdog_launch_authorized"] is False
    assert value["qualification_launch_authorized"] is False
    assert value["scored_launch_authorized"] is False
    assert value["receipt_sha256"] == crypto.digest_without(value, "receipt_sha256")
    tracked = json.loads(
        (
            ROOT
            / "docs/evidence/glm53-study/"
            "2026-09-06-glm53-dedicated-v25-watchdog-adapter-held-v1.json"
        ).read_text()
    )
    assert tracked == adapter.build_held(tracked["package_commit"])


def test_bound_engine_is_exact_and_restored() -> None:
    prior = (
        engine.server,
        engine.package,
        engine.LIVE_STATE_SCHEMA,
        engine.RELEASE_SCHEMA,
        engine.LAUNCH_SCHEMA,
        engine.AUTHORIZATION_CONFIGMAP_NAME,
    )
    with adapter.bound_engine():
        assert engine.server is server
        assert engine.package is package
        assert engine.LIVE_STATE_SCHEMA == adapter.LIVE_STATE_SCHEMA
        assert engine.RELEASE_SCHEMA == adapter.RELEASE_SCHEMA
        assert engine.LAUNCH_SCHEMA == adapter.LAUNCH_SCHEMA
        assert engine.AUTHORIZATION_CONFIGMAP_NAME == adapter.AUTHORIZATION_CONFIGMAP_NAME
        with (
            pytest.raises(adapter.AdapterError, match="already_bound"),
            adapter.bound_engine(),
        ):
            pass
    assert prior == (
        engine.server,
        engine.package,
        engine.LIVE_STATE_SCHEMA,
        engine.RELEASE_SCHEMA,
        engine.LAUNCH_SCHEMA,
        engine.AUTHORIZATION_CONFIGMAP_NAME,
    )
    assert engine.server is v24_server
    assert engine.package is v24_package


def test_real_v25_render_builds_exact_watcher_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(engine.time, "time", lambda: NOW)
    with adapter.bound_engine():
        rendered = engine.render(
            ROOT,
            COMMIT,
            binding(),
            live(),
            priority_classes=priorities(),
        )
    configmap, authorization, job = rendered["objects"]["items"]
    assert configmap["metadata"]["name"] == package.JOB_NAME + "-package"
    assert authorization["metadata"]["name"] == adapter.AUTHORIZATION_CONFIGMAP_NAME
    assert job["metadata"]["name"] == package.JOB_NAME
    assert job["spec"]["template"]["spec"]["priorityClassName"] == "fleet-serve-low"
    release = json.loads(authorization["data"]["LIVE_RELEASE.json"])
    assert release["schema_version"] == adapter.RELEASE_SCHEMA
    assert release["server_binding"] == binding()
    assert release["request_sha256"] == server.request_sha256()
    assert release["watchdog_launch_authorized"] is True
    assert release["qualification_launch_authorized"] is False
    assert release["scored_launch_authorized"] is False


def test_executable_adapter_calls_engine_under_exact_binding_and_validates_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def fake_launch(*_args: object, **_kwargs: object) -> dict:
        observed["server"] = engine.server
        observed["package"] = engine.package
        observed["schema"] = engine.LAUNCH_SCHEMA
        return {
            "schema_version": adapter.LAUNCH_SCHEMA,
            "server_launch_authorized": False,
            "watchdog_launch_authorized": True,
            "qualification_launch_authorized": False,
            "scored_launch_authorized": False,
            "protected_content_included": False,
        }

    monkeypatch.setattr(engine, "launch", fake_launch)
    receipt = adapter.launch(
        ROOT, COMMIT, "ft-run-deadbeef", priority_classes=priorities()
    )
    assert observed == {
        "server": server,
        "package": package,
        "schema": adapter.LAUNCH_SCHEMA,
    }
    assert receipt["watchdog_launch_authorized"] is True
    assert engine.server is v24_server


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("schema_version", "v24"),
        ("watchdog_launch_authorized", False),
        ("qualification_launch_authorized", True),
        ("scored_launch_authorized", True),
        ("protected_content_included", True),
    ],
)
def test_adapter_rejects_mutated_launch_receipt(
    monkeypatch: pytest.MonkeyPatch, field: str, bad: object
) -> None:
    value = {
        "schema_version": adapter.LAUNCH_SCHEMA,
        "server_launch_authorized": False,
        "watchdog_launch_authorized": True,
        "qualification_launch_authorized": False,
        "scored_launch_authorized": False,
        "protected_content_included": False,
    }
    value[field] = bad
    monkeypatch.setattr(engine, "launch", lambda *_args, **_kwargs: value)
    with pytest.raises(adapter.AdapterError, match="receipt_invalid"):
        adapter.launch(ROOT, COMMIT, "ft-run-deadbeef", priority_classes=priorities())
    assert engine.server is v24_server
