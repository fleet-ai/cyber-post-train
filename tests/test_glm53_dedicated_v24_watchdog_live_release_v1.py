import copy
import json
import subprocess
from pathlib import Path

import pytest

from evals.fleet import exact_pass4_crypto as crypto
from evals.fleet import glm53_dedicated_v23_request_counter_watchdog_v1 as runtime
from evals.fleet import glm53_dedicated_v24_server_v1 as server
from evals.fleet import glm53_dedicated_v24_watchdog_live_release_v1 as live_release
from evals.fleet import glm53_dedicated_v24_watchdog_package_v1 as package

ROOT = Path(__file__).resolve().parents[1]
NOW = 2_000_000_000.0


def _binding() -> dict:
    api_run_id = "ft-run-deadbeef"
    return {
        "server_title": server.TITLE,
        "server_run_dir": server.RUN_DIR,
        "api_run_id": api_run_id,
        "rayjob_uid": "11111111-1111-4111-8111-111111111111",
        "workload_uid": "22222222-2222-4222-8222-222222222222",
        "head_pod_uid": "33333333-3333-4333-8333-333333333333",
        "service_uid": "44444444-4444-4444-8444-444444444444",
        "service_origin": (
            f"http://{api_run_id}-head-svc.fleet-train-jobs.svc:8000"
        ),
        "served_id": server.SERVED_ID,
        "model_revision": server.MODEL_REVISION,
        "context_length": server.CONTEXT_LENGTH,
    }


def _priority_classes() -> list[dict]:
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


def _live(binding: dict | None = None) -> dict:
    binding = binding or _binding()
    value = {
        "schema_version": live_release.LIVE_STATE_SCHEMA,
        "status": "READY_POST_CREATE_UID_BOUND",
        "observed_at_epoch": NOW - 5,
        "ready_at_epoch": NOW - 15,
        "server_binding": binding,
        "request_sha256": crypto.sha256(crypto.canonical_json(server.payload())),
        "api_get_http_status": 200,
        "api_run_state": "RUNNING",
        "api_title_match_count": 1,
        "api_run_dir_match_count": 1,
        "rayjob_running": True,
        "rayjob_uid_match_count": 1,
        "workload_admitted": True,
        "workload_finished": False,
        "workload_preemption_events": 0,
        "workload_uid_match_count": 1,
        "head_pod_phase": "Running",
        "head_pod_ready": True,
        "head_pod_restarts": 0,
        "head_pod_uid_match_count": 1,
        "service_present": True,
        "service_uid_match_count": 1,
        "metrics_http_status": 200,
        "activity_metric_families": list(runtime.ACTIVITY_METRICS),
        "watchdog_job_match_count": 0,
        "watchdog_configmap_match_count": 0,
        "watchdog_result_root_absent": True,
        "fleet_task_instance_calls": 0,
        "fleet_session_calls": 0,
        "verifier_calls": 0,
        "scoring_calls": 0,
        "protected_content_included": False,
    }
    value["receipt_sha256"] = crypto.digest_without(value, "receipt_sha256")
    return value


def _rehash(value: dict) -> dict:
    value["receipt_sha256"] = crypto.digest_without(value, "receipt_sha256")
    return value


def test_precreate_package_remains_held_and_sentinel_cannot_release() -> None:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()
    sentinel = _binding()
    sentinel["api_run_id"] = "ft-run-v24exact"
    sentinel["service_origin"] = (
        "http://ft-run-v24exact-head-svc.fleet-train-jobs.svc:8000"
    )
    held = package.render(
        ROOT,
        commit,
        sentinel,
        ready_at_epoch=123.0,
        priority_classes=_priority_classes(),
    )
    assert held["watchdog_launch_authorized"] is False
    with pytest.raises(live_release.LiveReleaseError, match="live_state"):
        live_release.build_release(_live(sentinel), sentinel, now_epoch=NOW)


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("status", "HELD"),
        ("request_sha256", "sha256:" + "0" * 64),
        ("api_get_http_status", 503),
        ("api_run_state", "FAILED"),
        ("api_title_match_count", 2),
        ("api_run_dir_match_count", 0),
        ("rayjob_running", False),
        ("rayjob_uid_match_count", 0),
        ("workload_admitted", False),
        ("workload_finished", True),
        ("workload_preemption_events", 1),
        ("workload_uid_match_count", 2),
        ("head_pod_phase", "Pending"),
        ("head_pod_ready", False),
        ("head_pod_restarts", 1),
        ("head_pod_uid_match_count", 0),
        ("service_present", False),
        ("service_uid_match_count", 2),
        ("metrics_http_status", 500),
        ("activity_metric_families", ["sglang:num_requests_total"]),
        ("watchdog_job_match_count", 1),
        ("watchdog_configmap_match_count", 1),
        ("watchdog_result_root_absent", False),
        ("fleet_task_instance_calls", 1),
        ("fleet_session_calls", 1),
        ("verifier_calls", 1),
        ("scoring_calls", 1),
        ("protected_content_included", True),
    ],
)
def test_live_state_semantic_mutations_fail_closed(field: str, bad: object) -> None:
    value = _live()
    value[field] = bad
    _rehash(value)
    with pytest.raises(live_release.LiveReleaseError, match="live_state"):
        live_release.validate_live_state(value, _binding(), now_epoch=NOW)


def test_live_state_rejects_extra_missing_stale_future_and_old_ready() -> None:
    extra = _live()
    extra["unexpected"] = False
    _rehash(extra)
    missing = _live()
    del missing["service_present"]
    _rehash(missing)
    stale = _live()
    stale["observed_at_epoch"] = NOW - 61
    stale["ready_at_epoch"] = NOW - 62
    _rehash(stale)
    future = _live()
    future["observed_at_epoch"] = NOW + 1
    _rehash(future)
    old_ready = _live()
    old_ready["ready_at_epoch"] = NOW - 126
    _rehash(old_ready)
    for value in (extra, missing, stale, future, old_ready):
        with pytest.raises(live_release.LiveReleaseError, match="live_state"):
            live_release.validate_live_state(value, _binding(), now_epoch=NOW)


def test_live_binding_uid_and_service_origin_mutations_fail_closed() -> None:
    for field, bad in (
        ("rayjob_uid", "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"),
        ("workload_uid", "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"),
        ("head_pod_uid", "cccccccc-cccc-4ccc-8ccc-cccccccccccc"),
        ("service_uid", "dddddddd-dddd-4ddd-8ddd-dddddddddddd"),
        ("service_origin", "http://wrong.fleet-train-jobs.svc:8000"),
    ):
        binding = _binding()
        binding[field] = bad
        value = _live()
        with pytest.raises(live_release.LiveReleaseError):
            live_release.validate_live_state(value, binding, now_epoch=NOW)


def test_live_renderer_emits_authorization_and_exact_create_once_job() -> None:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()
    binding = _binding()
    live = _live(binding)
    rendered = live_release.render(
        ROOT,
        commit,
        binding,
        live,
        now_epoch=NOW,
        priority_classes=_priority_classes(),
    )
    configmap, authorization, job = rendered["objects"]["items"]
    release = rendered["live_release"]
    assert configmap["metadata"]["name"] == package.JOB_NAME + "-package"
    assert authorization["metadata"]["name"] == live_release.AUTHORIZATION_CONFIGMAP_NAME
    assert authorization["immutable"] is True
    assert json.loads(authorization["data"]["LIVE_RELEASE.json"]) == release
    assert job["metadata"]["name"] == package.JOB_NAME
    assert job["metadata"]["annotations"][
        "cyber-post-train.fleet.ai/create-once"
    ] == "true"
    assert job["metadata"]["annotations"][
        "cyber-post-train.fleet.ai/live-release-receipt-sha256"
    ] == release["receipt_sha256"]
    assert rendered["server_launch_authorized"] is False
    assert rendered["watchdog_launch_authorized"] is True
    assert rendered["qualification_launch_authorized"] is False
    assert rendered["scored_launch_authorized"] is False


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("status", "HELD"),
        ("server_binding_sha256", "sha256:" + "0" * 64),
        ("live_state_receipt_sha256", "sha256:" + "0" * 64),
        ("request_sha256", "sha256:" + "0" * 64),
        ("watchdog_job_name", "wrong"),
        ("watchdog_result_root", "/wrong"),
        ("watchdog_implementation_sha256", "sha256:" + "0" * 64),
        ("ready_at_epoch", NOW),
        ("idle_release_seconds", 601),
        ("create_once", False),
        ("server_launch_authorized", True),
        ("watchdog_launch_authorized", False),
        ("qualification_launch_authorized", True),
        ("scored_launch_authorized", True),
        ("protected_content_included", True),
    ],
)
def test_rehashed_live_release_mutations_fail_closed(field: str, bad: object) -> None:
    binding = _binding()
    live = _live(binding)
    release = live_release.build_release(live, binding, now_epoch=NOW)
    mutated = copy.deepcopy(release)
    mutated[field] = bad
    _rehash(mutated)
    with pytest.raises(live_release.LiveReleaseError, match="live_release"):
        live_release.validate_release(mutated, live, binding, now_epoch=NOW)


def test_tracked_live_release_receipt_is_self_digested_and_held() -> None:
    path = (
        ROOT
        / "docs/evidence/glm53-study/"
        "2026-09-06-glm53-dedicated-v24-watchdog-live-release-held-v1.json"
    )
    value = json.loads(path.read_text())
    assert value["status"] == "READY_HELD_FOR_EXACT_POST_CREATE_BINDING"
    assert value["watchdog_launch_authorized"] is False
    assert value["server_launch_authorized"] is False
    assert value["qualification_launch_authorized"] is False
    assert value["scored_launch_authorized"] is False
    assert value["receipt_sha256"] == crypto.digest_without(value, "receipt_sha256")
