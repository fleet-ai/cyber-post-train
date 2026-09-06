import json
import subprocess
from pathlib import Path

from evals.fleet import exact_pass4_crypto as crypto
from evals.fleet import glm53_dedicated_v24_watchdog_live_release_v1 as live_engine
from evals.fleet import glm53_dedicated_v26_create_v1 as v26
from evals.fleet import glm53_dedicated_v27_create_v1 as server
from evals.fleet import glm53_dedicated_v27_watchdog_live_release_v1 as adapter
from evals.fleet import glm53_dedicated_v27_watchdog_package_v1 as package

ROOT = Path(__file__).resolve().parents[1]
COMMIT = subprocess.run(
    ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True, capture_output=True
).stdout.strip()


def binding() -> dict:
    return {
        "server_title": server.TITLE,
        "server_run_dir": server.RUN_DIR,
        "api_run_id": "ft-run-deadbeef",
        "rayjob_uid": "11111111-1111-4111-8111-111111111111",
        "workload_uid": "22222222-2222-4222-8222-222222222222",
        "head_pod_uid": "33333333-3333-4333-8333-333333333333",
        "service_uid": "44444444-4444-4444-8444-444444444444",
        "service_origin": (
            "http://ft-run-deadbeef-abcde-head-svc.fleet-train-jobs.svc:8000"
        ),
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


def test_v27_server_identity_is_fresh_and_nested_engine_restores() -> None:
    before = (v26.TITLE, v26.RUN_DIR, v26.READY_SCHEMA)
    value = server.payload()
    server.validate_payload(value)
    assert value["title"] == server.TITLE
    assert value["run_dir"] == server.RUN_DIR
    assert server.READY_SCHEMA in server._observer_source()
    encoded = crypto.canonical_json(value).decode()
    assert v26.TITLE not in encoded
    assert v26.RUN_DIR not in encoded
    assert before == (v26.TITLE, v26.RUN_DIR, v26.READY_SCHEMA)


def test_v27_watchdog_package_is_generation_exact_and_held() -> None:
    server.validate_binding(binding())
    rendered = package.render(
        ROOT,
        COMMIT,
        binding(),
        ready_at_epoch=2_000_000_000.0,
        priority_classes=priorities(),
    )
    configmap, job = rendered["objects"]["items"]
    assert configmap["metadata"]["name"] == package.JOB_NAME + "-package"
    assert job["metadata"]["name"] == package.JOB_NAME
    command = job["spec"]["template"]["spec"]["containers"][0]["command"][-1]
    for expected in (
        package.RUNTIME_AUTH_SCHEMA,
        package.LIVE_RELEASE_SCHEMA,
        package.JOB_NAME,
        package.RESULT_ROOT,
    ):
        assert expected in command
    assert rendered["watchdog_launch_authorized"] is False
    assert rendered["qualification_launch_authorized"] is False
    assert rendered["scored_launch_authorized"] is False


def test_v27_adapter_held_receipt_and_engine_restoration() -> None:
    before = (
        live_engine.server,
        live_engine.package,
        live_engine.LIVE_STATE_SCHEMA,
        live_engine.RELEASE_SCHEMA,
    )
    value = adapter.build_held(COMMIT)
    assert value["runtime_auth_schema"] == package.RUNTIME_AUTH_SCHEMA
    assert value["live_release_schema"] == package.LIVE_RELEASE_SCHEMA
    assert value["watchdog_launch_authorized"] is False
    assert value["qualification_launch_authorized"] is False
    assert value["scored_launch_authorized"] is False
    assert value["receipt_sha256"] == crypto.digest_without(value, "receipt_sha256")
    with adapter.bound_engine():
        assert live_engine.server is server
        assert live_engine.package is package
        assert live_engine.LIVE_STATE_SCHEMA == adapter.LIVE_STATE_SCHEMA
        assert live_engine.RELEASE_SCHEMA == adapter.RELEASE_SCHEMA
    assert before == (
        live_engine.server,
        live_engine.package,
        live_engine.LIVE_STATE_SCHEMA,
        live_engine.RELEASE_SCHEMA,
    )
    tracked_create = json.loads(
        (
            ROOT
            / "docs/evidence/glm53-study/"
            "2026-09-06-glm53-dedicated-v27-create-wrapper-held-v1.json"
        ).read_text()
    )
    assert tracked_create == server.build_held()
    tracked_watchdog = json.loads(
        (
            ROOT
            / "docs/evidence/glm53-study/"
            "2026-09-06-glm53-dedicated-v27-watchdog-adapter-held-v1.json"
        ).read_text()
    )
    assert tracked_watchdog == adapter.build_held(tracked_watchdog["package_commit"])


def test_v27_handoff_failure_is_terminal_zero_gpu_and_successor_held() -> None:
    value = json.loads(
        (
            ROOT
            / "docs/evidence/glm53-study/"
            "2026-09-06-glm53-dedicated-v27-watchdog-handoff-failure-v1.json"
        ).read_text()
    )
    assert value["status"] == "FAILED_CLOSED_RELEASED_DIAGNOSTIC_GAP"
    assert value["jobs_api_final_get_http_status"] == 404
    assert value["kubernetes_remnants_after_release"] == 0
    assert value["dedicated_gpus_after_release"] == 0
    assert value["watchdog_job_created"] is False
    assert value["qualification_job_created"] is False
    assert value["retry_same_server_identity"] is False
    assert value["successor_launch_authorized"] is False
    assert value["receipt_sha256"] == crypto.digest_without(value, "receipt_sha256")
