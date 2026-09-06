import json
import subprocess
from pathlib import Path

import pytest

from evals.fleet import exact_pass4_crypto as crypto
from evals.fleet import glm53_dedicated_v23_request_counter_watchdog_v1 as watchdog_runtime
from evals.fleet import glm53_dedicated_v23_scorefree_package_v1 as package
from evals.fleet import glm53_dedicated_v23_scorefree_qualifier_v1 as qualifier
from evals.fleet import opencode_actual_harness_parity_v1 as actual_harness

ROOT = Path(__file__).resolve().parents[1]


def _binding() -> dict:
    return {
        "server_title": qualifier.SERVER_TITLE,
        "server_run_dir": qualifier.SERVER_RUN_DIR,
        "api_run_id": "ft-run-freshv23",
        "rayjob_uid": "11111111-1111-4111-8111-111111111111",
        "workload_uid": "22222222-2222-4222-8222-222222222222",
        "head_pod_uid": "33333333-3333-4333-8333-333333333333",
        "service_uid": "44444444-4444-4444-8444-444444444444",
        "service_origin": "http://fresh-v23-head-svc.fleet-train-jobs.svc.cluster.local:8000",
        "served_id": qualifier.SERVED_ID,
        "model_revision": qualifier.MODEL_REVISION,
        "context_length": qualifier.CONTEXT_LENGTH,
    }


def _evidence() -> tuple[dict, dict, dict]:
    binding = _binding()
    treatment = actual_harness.exact.EXPECTED_TREATMENT
    expected_model = actual_harness.exact.EXPECTED_MODELS["glm-5.3"]
    parity = {
        "schema_version": actual_harness.SCHEMA,
        "classification": "ACTUAL_HARNESS_PARITY",
        "status": "PASSED_NON_SCORED",
        "endpoint": {
            "kind": "dedicated_uid_bound_inference",
            "server_binding": qualifier.canonical_model_binding(binding),
        },
        "execution": {
            "final_marker_observed": True,
            "harness_exit_code": 0,
            "model_requests": 4,
            "scored_launch_authorized": False,
            "task_instance_session_verifier_scoring_calls": 0,
        },
        "harness": {
            "name": treatment["harness"],
            "version": treatment["harness_version"],
            "image": actual_harness.IMAGE,
            "image_id": actual_harness.IMAGE_ID,
            "release_asset_sha256": treatment["release_asset_sha256"],
            "provider_adapter": treatment["provider_adapter"],
            "context_management": treatment["context_management"],
            "context_window_size": treatment["context_window_size"],
            "compaction_headroom_tokens": treatment["compaction_headroom_tokens"],
            "max_output_tokens": treatment["max_output_tokens"],
            "max_model_requests": treatment["max_model_requests"],
            "timeout_seconds": treatment["timeout_seconds"],
        },
        "model": dict(expected_model),
        "tool_contract": {
            "names": treatment["tools"],
            "calls_observed_in_order": actual_harness.EXPECTED_CALL_ORDER,
            "mcp_catalog_sha256": treatment["tool_catalog_sha256"],
            "arguments_structurally_valid": True,
            "model_request_catalog_exact": True,
            "model_request_tool_names_exact": True,
            "model_request_tool_descriptions_exact": True,
            "model_request_tool_parameters_exact": True,
            "model_requests_with_tools": 3,
        },
        "privacy": {
            "benchmark_content_included": False,
            "credentials_included": False,
            "prompt_included": False,
            "responses_or_model_outputs_included": False,
            "stderr_or_stdout_included": False,
            "tool_arguments_included": False,
        },
    }
    parity["receipt_sha256"] = crypto.digest_without(parity, "receipt_sha256")
    watchdog = watchdog_runtime.build_active_receipt(
        binding,
        watcher_job_uid="55555555-5555-4555-8555-555555555555",
        watcher_pod_uid="66666666-6666-4666-8666-666666666666",
    )
    watchdog["initial_request_counter"] = 4
    watchdog["ready_at_epoch"] = 123.0
    watchdog["receipt_sha256"] = crypto.digest_without(watchdog, "receipt_sha256")
    live = {
        "schema_version": "fleet-glm53-dedicated-v23-scorefree-live-state-v1",
        "server_binding": binding,
        "rayjob_running": True,
        "workload_admitted": True,
        "workload_preemption_events": 0,
        "head_pod_ready": True,
        "head_pod_restarts": 0,
        "active_scored_controller_count": 0,
        "qualification_result_root_absent": True,
        "endpoint_lease_available": True,
        "watcher_job_uid": watchdog["watcher_job_uid"],
        "watcher_pod_uid": watchdog["watcher_pod_uid"],
        "watcher_job_active": 1,
        "watcher_pod_ready": True,
        "watcher_pod_restarts": 0,
        "seconds_since_last_model_request": 1,
    }
    live["receipt_sha256"] = crypto.digest_without(live, "receipt_sha256")
    return parity, watchdog, live


def test_held_contract_removes_false_acceptance_gate_and_forbids_scoring() -> None:
    value = qualifier.build_held()
    assert value["status"] == "READY_HELD"
    assert value["removed_false_prerequisite"]["rank51_attempt2_accepted_required"] is False
    assert value["score_free_boundary"]["concurrency_waves"] == [1, 2, 4]
    assert value["score_free_boundary"]["fleet_task_instance_calls"] == 0
    assert value["score_free_boundary"]["fleet_session_calls"] == 0
    assert value["score_free_boundary"]["verifier_calls"] == 0
    assert value["score_free_boundary"]["scoring_calls"] == 0
    assert value["required_live_evidence"]["idle_release_seconds"] == 600
    assert value["qualification_launch_authorized"] is False
    assert value["scored_launch_authorized"] is False
    assert value["receipt_sha256"] == crypto.digest_without(value, "receipt_sha256")


def test_tracked_held_receipt_is_self_digested_and_never_authorizes_launch() -> None:
    path = (
        ROOT / "docs/evidence/glm53-study/"
        "2026-09-06-glm53-dedicated-v23-scorefree-qualifier-held-v1.json"
    )
    value = json.loads(path.read_text())
    assert value["status"] == "READY_HELD_NO_LAUNCH"
    assert value["request_counter_watchdog"]["idle_release_seconds"] == 600
    assert value["server_launch_authorized"] is False
    assert value["qualification_launch_authorized"] is False
    assert value["scored_launch_authorized"] is False
    assert value["receipt_sha256"] == crypto.digest_without(value, "receipt_sha256")


def test_authorization_binds_uid_parity_watchdog_and_live_state() -> None:
    parity, watchdog, live = _evidence()
    authorization = qualifier.authorize(_binding(), parity, watchdog, live)
    assert authorization["qualification_launch_authorized"] is True
    assert authorization["scored_launch_authorized"] is False
    assert authorization["idle_release_seconds"] == 600
    assert authorization["server_binding"] == _binding()
    assert authorization["receipt_sha256"] == crypto.digest_without(authorization, "receipt_sha256")


def test_real_parity_validator_accepts_only_canonical_seven_key_binding() -> None:
    canonical = qualifier.canonical_model_binding(_binding())
    assert set(canonical) == {
        "api_run_id",
        "rayjob_uid",
        "head_pod_uid",
        "service_uid",
        "served_id",
        "model_revision",
        "context_length",
    }
    assert actual_harness.validate_server_binding(canonical, "glm-5.3") == canonical
    with pytest.raises(actual_harness.ActualHarnessParityError):
        actual_harness.validate_server_binding(_binding(), "glm-5.3")


@pytest.mark.parametrize(
    ("source", "field", "value", "message"),
    [
        ("watchdog", "health_or_process_liveness_refreshes", True, "watchdog"),
        ("watchdog", "idle_release_seconds", 601, "watchdog"),
        ("live", "active_scored_controller_count", 1, "boundary"),
        ("live", "seconds_since_last_model_request", 600, "boundary"),
        ("parity", "status", "FAILED", "parity"),
    ],
)
def test_authorization_fails_closed(source: str, field: str, value: object, message: str) -> None:
    parity, watchdog, live = _evidence()
    target = {"parity": parity, "watchdog": watchdog, "live": live}[source]
    target[field] = value
    if source != "live":
        target["receipt_sha256"] = crypto.digest_without(target, "receipt_sha256")
    with pytest.raises(qualifier.QualificationError, match=message):
        qualifier.authorize(_binding(), parity, watchdog, live)


@pytest.mark.parametrize(
    ("path", "field", "value"),
    [
        (("harness",), "image_id", "sha256:wrong"),
        (("tool_contract",), "model_request_tool_parameters_exact", False),
        (("model",), "revision", "wrong"),
        (("execution",), "task_instance_session_verifier_scoring_calls", 1),
    ],
)
def test_authorization_rejects_rehashed_but_noncanonical_actual_parity(
    path: tuple[str, ...], field: str, value: object
) -> None:
    parity, watchdog, live = _evidence()
    target = parity
    for part in path:
        target = target[part]
    target[field] = value
    parity["receipt_sha256"] = crypto.digest_without(parity, "receipt_sha256")
    with pytest.raises(qualifier.QualificationError, match="parity"):
        qualifier.authorize(_binding(), parity, watchdog, live)


def test_authorization_rejects_rehashed_live_or_wrong_watchdog_source() -> None:
    parity, watchdog, live = _evidence()
    live["active_scored_controller_count"] = 1
    live["receipt_sha256"] = crypto.digest_without(live, "receipt_sha256")
    with pytest.raises(qualifier.QualificationError, match="boundary"):
        qualifier.authorize(_binding(), parity, watchdog, live)

    parity, watchdog, live = _evidence()
    watchdog["implementation_sha256"] = "sha256:" + "0" * 64
    watchdog["receipt_sha256"] = crypto.digest_without(watchdog, "receipt_sha256")
    with pytest.raises(qualifier.QualificationError, match="watchdog"):
        qualifier.authorize(_binding(), parity, watchdog, live)


def test_request_counter_watchdog_refreshes_only_on_counter_growth() -> None:
    state = {"counter": 4, "last_model_request_at": 100.0}
    assert watchdog_runtime.advance(state, counter=4, now=200.0) == state
    assert watchdog_runtime.advance(state, counter=5, now=200.0) == {
        "counter": 5,
        "last_model_request_at": 200.0,
    }
    with pytest.raises(watchdog_runtime.WatchdogError, match="regressed"):
        watchdog_runtime.advance(state, counter=3, now=200.0)


def test_request_counter_parser_sums_realistic_labeled_glm_series_only() -> None:
    metrics = """
# HELP sglang:num_requests_total Number of requests
sglang:num_requests_total{model_name="glm-5.3",is_streaming="true",engine_type="tp"} 4.0
sglang:num_requests_total{engine_type="tp",model_name="glm-5.3",is_streaming="false"} 3
sglang:num_requests_total{model_name="another-model",is_streaming="true"} 900
"""
    assert watchdog_runtime.parse_counter(metrics) == 7


def test_request_counter_watchdog_releases_exact_run_after_600_idle_seconds() -> None:
    times = iter((599.0, 600.0))
    released: list[str] = []
    result = watchdog_runtime.watch(
        api_run_id="ft-run-freshv23",
        initial_counter=7,
        ready_at=0.0,
        read_counter=lambda: 7,
        release_via_jobs_api=released.append,
        clock=lambda: next(times),
        sleep=lambda _seconds: None,
    )
    assert result == "RELEASED_IDLE"
    assert released == ["ft-run-freshv23"]


def test_request_growth_prevents_false_idle_release() -> None:
    times = iter((599.0, 600.0, 1199.0, 1200.0))
    counters = iter((8, 8, 8, 8))
    released: list[str] = []
    result = watchdog_runtime.watch(
        api_run_id="ft-run-freshv23",
        initial_counter=7,
        ready_at=0.0,
        read_counter=lambda: next(counters),
        release_via_jobs_api=released.append,
        clock=lambda: next(times),
        sleep=lambda _seconds: None,
    )
    assert result == "RELEASED_IDLE"
    assert released == ["ft-run-freshv23"]


def test_gpu_observer_rejects_valid_uuid_for_the_wrong_server() -> None:
    observed = {
        "schema_version": qualifier.GPU_OBSERVER_SCHEMA,
        "status": "OBSERVED_SCORE_FREE_WAVE",
        "concurrency": 1,
        "server": qualifier.build_held()["server"],
        "devices_seen": 8,
        "samples_per_device": 2,
        "max_utilization_percent_by_device": [50] * 8,
        "identity": {
            "server_rayjob_uid": "99999999-9999-4999-8999-999999999999",
            "server_head_pod_uid": _binding()["head_pod_uid"],
            "qualifier_job_uid": "77777777-7777-4777-8777-777777777777",
            "qualifier_pod_uid": "88888888-8888-4888-8888-888888888888",
        },
        "server_identity_unchanged": True,
        "qualifier_identity_unchanged": True,
    }
    observed["receipt_sha256"] = crypto.digest_without(observed, "receipt_sha256")
    with pytest.raises(qualifier.QualificationError, match="gpu_wave"):
        qualifier.validate_gpu_wave(observed, 1, qualifier.build_held()["server"], _binding())


def test_watchdog_active_receipt_binds_exact_loaded_source_and_release_route() -> None:
    receipt = watchdog_runtime.build_active_receipt(
        _binding(),
        watcher_job_uid="55555555-5555-4555-8555-555555555555",
        watcher_pod_uid="66666666-6666-4666-8666-666666666666",
    )
    assert receipt["implementation_sha256"] == watchdog_runtime.source_sha256()
    assert receipt["release_route"] == "DELETE /v1/runs/{api_run_id}"
    assert receipt["receipt_sha256"] == crypto.digest_without(receipt, "receipt_sha256")


def test_watchdog_package_is_create_once_uid_bound_and_nonpreempting() -> None:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()
    rendered = package.render_watchdog(ROOT, commit, _binding(), ready_at_epoch=123.0)
    configmap, job = rendered["objects"]["items"]
    assert configmap["immutable"] is True
    assert job["metadata"]["annotations"]["cyber-post-train.fleet.ai/create-once"] == "true"
    pod = job["spec"]["template"]["spec"]
    assert pod["priorityClassName"] == "fleet-infra-quiet"
    assert pod["preemptionPolicy"] == "Never"
    container = pod["containers"][0]
    env = {row["name"]: row for row in container["env"]}
    assert env["JOB_UID"]["valueFrom"]["fieldRef"]["fieldPath"] == (
        "metadata.labels['batch.kubernetes.io/controller-uid']"
    )
    assert env["POD_UID"]["valueFrom"]["fieldRef"]["fieldPath"] == "metadata.uid"
    assert env["GH_TOKEN"]["valueFrom"]["secretKeyRef"] == {
        "name": "img-build-secrets",
        "key": "GH_TOKEN",
    }
    assert rendered["watchdog_launch_authorized"] is True
    assert rendered["qualification_launch_authorized"] is False
    assert rendered["scored_launch_authorized"] is False


def test_package_is_exact_create_once_nonpreempting_and_scorefree() -> None:
    parity, watchdog, live = _evidence()
    authorization = qualifier.authorize(_binding(), parity, watchdog, live)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()
    rendered = package.render(ROOT, commit, authorization)
    configmap, auth, job = rendered["objects"]["items"]
    assert configmap["immutable"] is auth["immutable"] is True
    assert job["metadata"]["name"] == qualifier.JOB_NAME
    assert job["metadata"]["annotations"]["cyber-post-train.fleet.ai/create-once"] == "true"
    assert job["metadata"]["annotations"]["cyber-post-train.fleet.ai/score-free"] == "true"
    assert job["spec"]["template"]["spec"]["priorityClassName"] == "fleet-infra-quiet"
    assert job["spec"]["template"]["spec"]["preemptionPolicy"] == "Never"
    assert rendered["server_launch_authorized"] is False
    assert rendered["qualification_launch_authorized"] is True
    assert rendered["scored_launch_authorized"] is False
    package_manifest = json.loads(configmap["data"]["package.json"])
    assert package_manifest["score_free"] is True
    assert package_manifest["scored_launch_authorized"] is False
    assert set(package_manifest["external_uid_bound_operator_files"]) == {
        "evals/fleet/glm53_dedicated_v23_scorefree_gpu_observer_v1.py",
        "evals/fleet/scripts/observe_glm53_dedicated_v23_scorefree_gpu_v1.sh",
    }
    assert "glm53_dedicated_v23_scorefree_qualifier_v1 run" in configmap["data"]["run.sh"]
    assert any(
        key.endswith("glm53_dedicated_v23_request_counter_watchdog_v1.py")
        for key in configmap["data"]
    )
    container = job["spec"]["template"]["spec"]["containers"][0]
    env = {row["name"]: row for row in container["env"]}
    assert env["DEDICATED_SERVICE_ORIGIN"]["value"] == _binding()["service_origin"]
    assert env["JOB_UID"]["valueFrom"]["fieldRef"]["fieldPath"] == (
        "metadata.labels['batch.kubernetes.io/controller-uid']"
    )
    assert env["POD_UID"]["valueFrom"]["fieldRef"]["fieldPath"] == "metadata.uid"


def test_execute_keeps_fleet_calls_zero_and_runs_frozen_waves(tmp_path, monkeypatch) -> None:
    parity, watchdog, live = _evidence()
    authorization = qualifier.authorize(_binding(), parity, watchdog, live)
    calls = []

    def wave(concurrency, **_kwargs):
        assert _kwargs["binding"] == qualifier.canonical_model_binding(_binding())
        calls.append(concurrency)
        return {
            "concurrency": concurrency,
            "streams_succeeded": concurrency,
            "errors": 0,
            "retries": 0,
            "timeouts": 0,
            "request_counter_delta": concurrency,
            "model_requests_observed": concurrency,
            "stream_latency_seconds": {"p95": 1.0},
            "throughput_streams_per_second": float(concurrency),
        }

    monkeypatch.setattr(qualifier.engine, "run_wave", wave)
    monkeypatch.setattr(qualifier.engine, "validate_runtime_ramp", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        qualifier.endpoint_lease,
        "acquire_endpoint_lease",
        lambda **_kwargs: __import__("contextlib").nullcontext(),
    )
    out = tmp_path / "RAW.json"

    def observed(_root, concurrency, _server):
        value = {
            "schema_version": qualifier.GPU_OBSERVER_SCHEMA,
            "status": "OBSERVED_SCORE_FREE_WAVE",
            "concurrency": concurrency,
            "server": qualifier.build_held()["server"],
            "devices_seen": 8,
            "samples_per_device": 2,
            "max_utilization_percent_by_device": [50] * 8,
            "identity": {
                "server_rayjob_uid": "11111111-1111-4111-8111-111111111111",
                "server_head_pod_uid": "33333333-3333-4333-8333-333333333333",
                "qualifier_job_uid": "77777777-7777-4777-8777-777777777777",
                "qualifier_pod_uid": "88888888-8888-4888-8888-888888888888",
            },
            "server_identity_unchanged": True,
            "qualifier_identity_unchanged": True,
        }
        value["receipt_sha256"] = crypto.digest_without(value, "receipt_sha256")
        return value

    result = qualifier.execute(
        authorization,
        out,
        origin="http://v23.invalid:8000",
        runner=lambda *_args, **_kwargs: {},
        counter=lambda _origin: 0,
        observer=observed,
        qualifier_identity={
            "job_uid": "77777777-7777-4777-8777-777777777777",
            "pod_uid": "88888888-8888-4888-8888-888888888888",
        },
    )
    assert calls == [1, 2, 4]
    assert result["fleet_task_instance_calls"] == 0
    assert result["fleet_session_calls"] == 0
    assert result["verifier_calls"] == 0
    assert result["scoring_calls"] == 0
    assert result["scored_launch_authorized"] is False
    assert result["receipt_sha256"] == crypto.digest_without(result, "receipt_sha256")
    qualifier.validate_raw(result)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("status", "FAILED"),
        ("authorization_receipt_sha256", "sha256:" + "0" * 64),
        ("fleet_session_calls", 1),
        ("scored_launch_authorized", True),
    ],
)
def test_raw_validation_fails_closed(field: str, value: object) -> None:
    parity, watchdog, live = _evidence()
    authorization = qualifier.authorize(_binding(), parity, watchdog, live)
    raw = {
        "schema_version": qualifier.RAW_SCHEMA,
        "status": "COMPLETED_SCORE_FREE_WAVES",
        "authorization_receipt_sha256": authorization["receipt_sha256"],
        "authorization": authorization,
        "server_binding": _binding(),
        "waves": [],
        "gpu_waves": [],
        "fleet_task_instance_calls": 0,
        "fleet_session_calls": 0,
        "verifier_calls": 0,
        "scoring_calls": 0,
        "scored_launch_authorized": False,
    }
    raw["receipt_sha256"] = crypto.digest_without(raw, "receipt_sha256")
    raw[field] = value
    if field != "authorization_receipt_sha256":
        raw["receipt_sha256"] = crypto.digest_without(raw, "receipt_sha256")
    with pytest.raises(qualifier.QualificationError, match="raw_authority"):
        qualifier.validate_raw(raw)


def test_raw_rejects_rehashed_authorization_with_nonzero_fleet_calls() -> None:
    parity, watchdog, live = _evidence()
    authorization = qualifier.authorize(_binding(), parity, watchdog, live)
    authorization["fleet_session_calls"] = 1
    authorization["receipt_sha256"] = crypto.digest_without(authorization, "receipt_sha256")
    raw = {
        "schema_version": qualifier.RAW_SCHEMA,
        "status": "COMPLETED_SCORE_FREE_WAVES",
        "authorization_receipt_sha256": authorization["receipt_sha256"],
        "authorization": authorization,
        "server_binding": _binding(),
        "waves": [],
        "gpu_waves": [],
        "fleet_task_instance_calls": 0,
        "fleet_session_calls": 0,
        "verifier_calls": 0,
        "scoring_calls": 0,
        "scored_launch_authorized": False,
    }
    raw["receipt_sha256"] = crypto.digest_without(raw, "receipt_sha256")
    with pytest.raises(qualifier.QualificationError, match="raw_authority"):
        qualifier.validate_raw(raw)
