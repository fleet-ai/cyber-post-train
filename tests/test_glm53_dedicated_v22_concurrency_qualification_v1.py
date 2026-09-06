import json
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
    gpu = {
        "waves": [
            {
                "concurrency": value,
                "devices_seen": 8,
                "samples_per_device": 3,
                "server_identity_unchanged": True,
            }
            for value in (1, 2, 4)
        ]
    }
    assert held.evaluate(waves, gpu)["status"] == "PASSED_SCORE_FREE"
    waves[2]["throughput_streams_per_second"] = 0.14
    verdict = held.evaluate(waves, gpu)
    assert verdict["status"] == "FAILED"
    assert "c4_throughput_ratio" in verdict["failures"]


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
