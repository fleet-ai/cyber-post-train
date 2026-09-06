import datetime
import json
import subprocess
from pathlib import Path

import pytest

from evals.fleet import exact_pass4_crypto as crypto
from evals.fleet import glm53_dedicated_v23_request_counter_watchdog_v1 as runtime
from evals.fleet import glm53_dedicated_v24_server_v1 as v24
from evals.fleet import glm53_dedicated_v24_watchdog_live_release_v1 as engine
from evals.fleet import glm53_dedicated_v25_create_v1 as v25
from evals.fleet import glm53_dedicated_v26_create_v1 as server
from evals.fleet import glm53_dedicated_v26_watchdog_live_release_v1 as adapter
from evals.fleet import glm53_dedicated_v26_watchdog_package_v1 as package

ROOT = Path(__file__).resolve().parents[1]
NOW = 2_000_000_000.0
COMMIT = subprocess.run(
    ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True, capture_output=True
).stdout.strip()


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


def test_v26_create_identity_is_fresh_and_restores_engine() -> None:
    before = (v25.TITLE, v25.RUN_DIR, v25.READY_SCHEMA)
    value = server.payload()
    server.validate_payload(value)
    assert value["title"] == server.TITLE
    assert value["run_dir"] == server.RUN_DIR
    assert server.READY_SCHEMA in server._observer_source()
    encoded = crypto.canonical_json(value).decode()
    assert v25.TITLE not in encoded
    assert v25.RUN_DIR not in encoded
    assert v24.TITLE not in encoded
    assert before == (v25.TITLE, v25.RUN_DIR, v25.READY_SCHEMA)


def test_v26_watchdog_render_and_runtime_authorization_are_generation_exact(
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
    assert job["metadata"]["name"] == package.JOB_NAME
    assert job["spec"]["template"]["spec"]["priorityClassName"] == "fleet-serve-low"
    release = json.loads(authorization["data"]["LIVE_RELEASE.json"])
    assert release["schema_version"] == package.LIVE_RELEASE_SCHEMA
    command = job["spec"]["template"]["spec"]["containers"][0]["command"][-1]
    for expected in (
        package.RUNTIME_AUTH_SCHEMA,
        package.LIVE_RELEASE_SCHEMA,
        package.JOB_NAME,
        package.RESULT_ROOT,
    ):
        assert expected in command
    runtime.validate_launch_authorization(
        release,
        binding(),
        ready_at_epoch=release["ready_at_epoch"],
        package_commit=release["watchdog_package_commit"],
        package_sha256=release["watchdog_package_sha256"],
        expected_runtime_auth_schema=package.RUNTIME_AUTH_SCHEMA,
        expected_live_release_schema=package.LIVE_RELEASE_SCHEMA,
        expected_watchdog_job_name=package.JOB_NAME,
        expected_watchdog_result_root=package.RESULT_ROOT,
    )
    receipt = runtime.build_runtime_authorization_receipt(
        release,
        binding(),
        watcher_job_uid="88888888-8888-4888-8888-888888888888",
        watcher_pod_uid="99999999-9999-4999-8999-999999999999",
        package_commit=release["watchdog_package_commit"],
        package_sha256=release["watchdog_package_sha256"],
        expected_runtime_auth_schema=package.RUNTIME_AUTH_SCHEMA,
        expected_live_release_schema=package.LIVE_RELEASE_SCHEMA,
        expected_watchdog_job_name=package.JOB_NAME,
        expected_watchdog_result_root=package.RESULT_ROOT,
    )
    assert receipt["schema_version"] == package.RUNTIME_AUTH_SCHEMA
    assert receipt["expected_runtime_auth_schema"] == package.RUNTIME_AUTH_SCHEMA


def test_v26_held_receipts_are_digest_valid_and_non_authorizing() -> None:
    for value in (server.build_held(), adapter.build_held(COMMIT)):
        assert value["server_launch_authorized"] is False
        assert value["qualification_launch_authorized"] is False
        assert value["scored_launch_authorized"] is False
        assert value["receipt_sha256"] == crypto.digest_without(value, "receipt_sha256")


def test_v26_wrong_generation_contract_fails_closed(
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
    release = json.loads(rendered["objects"]["items"][1]["data"]["LIVE_RELEASE.json"])
    with pytest.raises(runtime.WatchdogError, match="release_authorization"):
        runtime.validate_launch_authorization(
            release,
            binding(),
            ready_at_epoch=release["ready_at_epoch"],
            package_commit=release["watchdog_package_commit"],
            package_sha256=release["watchdog_package_sha256"],
            expected_runtime_auth_schema=(
                "fleet-glm53-dedicated-v25-watchdog-runtime-auth-v1"
            ),
            expected_live_release_schema=package.LIVE_RELEASE_SCHEMA,
            expected_watchdog_job_name=package.JOB_NAME,
            expected_watchdog_result_root=package.RESULT_ROOT,
        )
