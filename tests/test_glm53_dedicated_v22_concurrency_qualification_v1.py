import json
from contextlib import nullcontext
from pathlib import Path

import pytest

from evals.fleet import glm53_dedicated_v22_concurrency_authorization_v1 as authorization
from evals.fleet import glm53_dedicated_v22_concurrency_package_v1 as package
from evals.fleet import glm53_dedicated_v22_concurrency_qualification_v1 as held
from evals.fleet import self_hosted

ROOT = Path(__file__).resolve().parents[1]


def test_concurrency_qualification_is_score_free_and_fail_closed() -> None:
    value = held.render(ROOT)
    assert value["launch_authorized"] is False
    assert [phase["concurrency"] for phase in value["phases"]] == [1, 2, 4]
    assert value["treatment"]["tools"] == ["bash", "submit_report"]
    assert value["treatment"]["context_window_size"] == 262144
    assert value["score_free_boundary"]["fleet_task_instance_calls"] == 0
    assert value["score_free_boundary"]["fleet_session_calls"] == 0
    assert value["score_free_boundary"]["verifier_calls"] == 0
    assert value["score_free_boundary"]["scoring_calls"] == 0
    assert value["fail_closed_ramp"]["never_overlap_scored_controller"] is True
    assert value["fail_closed_ramp"]["gpu_observer_receipt_required_before_next_wave"] is True
    assert value["fail_closed_ramp"]["ramp_2_requires"]["all_eight_gpus_active_in_wave"] is True
    assert value["gpu_observer_execution"]["mode"] == "local_operator_uid_bound_kubectl"
    assert value["gpu_observer_execution"]["atomic_receipt_write_via_stdin"] is True
    assert str(held.LEASE_ROOT).endswith("/opencode11827-dedicated-v22-v1")
    assert value["launch_prerequisites"]["rank51_attempt2_terminal_accepted"] is True
    assert value["scored_concurrency_change_authorized"] is False


def _binding() -> dict:
    return json.loads((ROOT / held.BINDING).read_text())


def _harness_receipt(binding: dict) -> dict:
    return {
        "status": "PASSED_NON_SCORED",
        "receipt_sha256": "sha256:" + "1" * 64,
        "endpoint": {"server_binding": binding},
        "execution": {
            "model_requests": 2,
            "task_instance_session_verifier_scoring_calls": 0,
        },
    }


def test_wave_uses_distinct_streams_and_exact_counter_delta() -> None:
    binding = _binding()
    counts = iter([10, 14])

    def runner(*_args, **kwargs):
        assert kwargs["docker_add_host_gateway"] is True
        return _harness_receipt(binding)

    wave = held.run_wave(
        2,
        origin=held.ORIGIN,
        binding=binding,
        runner=runner,
        counter=lambda _origin: next(counts),
    )
    assert [row["stream_id"] for row in wave["streams"]] == ["synthetic-a", "synthetic-b"]
    assert wave["request_counter_delta"] == 4
    assert wave["errors"] == wave["retries"] == wave["timeouts"] == 0


def test_wave_fails_on_nonexclusive_request_counter() -> None:
    binding = _binding()
    counts = iter([10, 15])
    with pytest.raises(held.QualificationError, match="exclusive_request_counter_delta_mismatch"):
        held.run_wave(
            2,
            origin=held.ORIGIN,
            binding=binding,
            runner=lambda *_args, **_kwargs: _harness_receipt(binding),
            counter=lambda _origin: next(counts),
        )


def _wave(concurrency: int, p95: float, throughput: float) -> dict:
    return {
        "concurrency": concurrency,
        "streams_succeeded": concurrency,
        "errors": 0,
        "retries": 0,
        "timeouts": 0,
        "request_counter_delta": concurrency * 2,
        "model_requests_observed": concurrency * 2,
        "stream_latency_seconds": {"p95": p95},
        "throughput_streams_per_second": throughput,
    }


def test_evaluator_passes_c1_c2_c4_and_fails_closed() -> None:
    waves = [_wave(1, 10, 0.1), _wave(2, 15, 0.13), _wave(4, 19, 0.16)]
    server = held.render(ROOT)["server"]
    gpu_waves = []
    for value in (1, 2, 4):
        row = {
            "schema_version": held.GPU_OBSERVER_SCHEMA,
            "status": "OBSERVED_SCORE_FREE_WAVE",
            "concurrency": value,
            "server": server,
            "devices_seen": 8,
            "samples_per_device": 3,
            "max_utilization_percent_by_device": [50] * 8,
            "identity": {
                "server_rayjob_uid": "11111111-1111-4111-8111-111111111111",
                "server_head_pod_uid": "22222222-2222-4222-8222-222222222222",
                "qualifier_job_uid": "33333333-3333-4333-8333-333333333333",
                "qualifier_pod_uid": "44444444-4444-4444-8444-444444444444",
            },
            "server_identity_unchanged": True,
            "qualifier_identity_unchanged": True,
        }
        row["receipt_sha256"] = self_hosted.digest_without(row, "receipt_sha256")
        gpu_waves.append(row)
    gpu = {"server": server, "waves": gpu_waves}
    assert held.evaluate(waves, gpu)["status"] == "PASSED_SCORE_FREE"
    assert held.evaluate(waves, gpu)["qualified_concurrency_ceiling"] == 4
    waves[2]["throughput_streams_per_second"] = 0.14
    verdict = held.evaluate(waves, gpu)
    assert verdict["status"] == "FAILED"
    assert verdict["qualified_concurrency_ceiling"] == 0
    assert "c4_throughput_ratio" in verdict["failures"]
    gpu_waves[1]["max_utilization_percent_by_device"][0] = 0
    gpu_waves[1]["receipt_sha256"] = self_hosted.digest_without(gpu_waves[1], "receipt_sha256")
    assert "c2_gpu_or_identity" in held.evaluate(waves, gpu)["failures"]


@pytest.mark.parametrize(
    ("failure", "match"),
    (
        ("protocol", "c2_runtime_gate_failed"),
        ("counter", "c2_runtime_gate_failed"),
        ("latency", "c2_latency_gate_failed"),
    ),
)
def test_execute_stops_before_c4_when_c2_gate_fails(
    tmp_path: Path, monkeypatch, failure: str, match: str
) -> None:
    plan = held.render(ROOT)
    authorization_receipt = {
        "qualification_launch_authorized": True,
        "server": plan["server"],
        "no_active_scored_controller": True,
    }
    authorization_receipt["receipt_sha256"] = self_hosted.digest_without(
        authorization_receipt, "receipt_sha256"
    )
    authorization_path = tmp_path / "authorization.json"
    authorization_path.write_text(json.dumps(authorization_receipt))
    calls = []

    def fake_wave(concurrency, **_kwargs):
        calls.append(concurrency)
        wave = _wave(concurrency, 10 if concurrency == 1 or failure != "latency" else 25, 0.1)
        if concurrency == 2 and failure == "protocol":
            wave["streams_succeeded"] = 1
        if concurrency == 2 and failure == "counter":
            wave["request_counter_delta"] += 1
        return wave

    monkeypatch.setattr(held, "run_wave", fake_wave)
    monkeypatch.setattr(
        held.endpoint_lease, "acquire_endpoint_lease", lambda **_kwargs: nullcontext()
    )

    def observed(_root, concurrency, server):
        row = {
            "schema_version": held.GPU_OBSERVER_SCHEMA,
            "status": "OBSERVED_SCORE_FREE_WAVE",
            "concurrency": concurrency,
            "server": server,
            "devices_seen": 8,
            "samples_per_device": 2,
            "max_utilization_percent_by_device": [50] * 8,
            "identity": {
                "server_rayjob_uid": "11111111-1111-4111-8111-111111111111",
                "server_head_pod_uid": "22222222-2222-4222-8222-222222222222",
                "qualifier_job_uid": "33333333-3333-4333-8333-333333333333",
                "qualifier_pod_uid": "44444444-4444-4444-8444-444444444444",
            },
            "server_identity_unchanged": True,
            "qualifier_identity_unchanged": True,
        }
        row["receipt_sha256"] = self_hosted.digest_without(row, "receipt_sha256")
        return row

    with pytest.raises(held.QualificationError, match=match):
        held.execute(
            ROOT,
            authorization_path,
            tmp_path / "RAW.json",
            gpu_observer_root=tmp_path / "gpu",
            observer=observed,
        )
    assert calls == [1, 2]


def test_post_acceptance_authorization_is_score_free_only() -> None:
    accepted = {
        "status": "ACCEPTED_VALIDATED",
        "accepted": True,
        "credited": True,
        "cell_id": authorization.CELL_ID,
        "execution_id": authorization.EXECUTION_ID,
    }
    accepted["receipt_sha256"] = self_hosted.digest_without(accepted, "receipt_sha256")
    plan = held.render(ROOT)
    live = {
        "server": plan["server"],
        "binding": _binding(),
        "rayjob_running": True,
        "workload_admitted": True,
        "workload_preemption_events": 0,
        "head_pod_ready": True,
        "head_pod_restarts": 0,
        "active_scored_controller_count": 0,
        "qualification_result_root_absent": True,
        "endpoint_lease_available": True,
        "request_counter_watchdog_active": True,
    }
    value = authorization.build(ROOT, accepted, live)
    assert value["qualification_launch_authorized"] is True
    assert value["scored_successor_launch_authorized"] is False
    live["active_scored_controller_count"] = 1
    with pytest.raises(authorization.AuthorizationError, match="live_score_free_boundary_invalid"):
        authorization.build(ROOT, accepted, live)


def test_package_job_is_cpu_only_dind_and_create_once() -> None:
    auth = {
        "qualification_launch_authorized": True,
        "scored_successor_launch_authorized": False,
    }
    auth["receipt_sha256"] = self_hosted.digest_without(auth, "receipt_sha256")
    configmap = {
        "data": {
            "package.json": json.dumps(
                {"package_commit": "a" * 40, "package_sha256": "sha256:" + "2" * 64}
            )
        }
    }
    job = package.build_job(configmap, auth)
    pod = job["spec"]["template"]["spec"]
    assert job["metadata"]["annotations"]["cyber-post-train.fleet.ai/create-once"] == "true"
    assert job["metadata"]["annotations"]["cyber-post-train.fleet.ai/score-free"] == "true"
    assert pod["priorityClassName"] == "fleet-infra-quiet"
    assert pod["preemptionPolicy"] == "Never"
    assert all("nvidia.com/gpu" not in str(row.get("resources", {})) for row in pod["containers"])
    assert (
        next(row for row in pod["initContainers"] if row["name"] == "dind")["securityContext"][
            "privileged"
        ]
        is True
    )


def test_immutable_package_binds_external_uid_observer_sources(monkeypatch) -> None:
    monkeypatch.setattr(package, "_source", lambda _root, _commit, path: path.encode())
    configmap = package.build_configmap(ROOT, "a" * 40)
    value = json.loads(configmap["data"]["package.json"])
    assert set(value["external_uid_bound_operator_files"]) == set(package.OPERATOR_FILES)
    assert all(
        digest.startswith("sha256:") and len(digest) == 71
        for digest in value["external_uid_bound_operator_files"].values()
    )


def test_authenticated_jobs_api_preview_is_fail_closed_for_cpu_qualifier() -> None:
    path = (
        ROOT / "docs/evidence/glm53-study/"
        "2026-09-06-glm53-v22-concurrency-jobs-api-preview-held-v1.json"
    )
    value = json.loads(path.read_text())
    assert value["receipt_sha256"] == self_hosted.digest_without(value, "receipt_sha256")
    assert value["status"] == "HELD_API_CONTRACT_MISMATCH"
    assert value["authenticated_jobs_api_contract"]["preview_request_schema"] == "RLJobConfig"
    assert value["authenticated_jobs_api_contract"]["gpus_per_worker_minimum"] == 1
    assert value["package_contract"]["cpu_only"] is True
    assert value["preview_post_calls"] == 0
    assert value["api_mutation_calls"] == 0
    assert value["qualification_launch_authorized"] is False
