from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from evals.fleet import exact_pass4_crypto as crypto
from evals.fleet import laptop_dedicated_qwen_rank3_v1 as lane

ROOT = Path(__file__).resolve().parents[1]


def _objects() -> tuple[dict, dict, dict]:
    rayjob = {
        "metadata": {"name": lane.RAYJOB_NAME, "uid": lane.RAYJOB_UID},
        "status": {
            "jobDeploymentStatus": "Running",
            "jobStatus": "RUNNING",
            "rayClusterStatus": {"state": "ready"},
        },
    }
    pod = {
        "metadata": {"name": lane.HEAD_POD_NAME, "uid": lane.HEAD_POD_UID},
        "status": {
            "phase": "Running",
            "conditions": [{"type": "Ready", "status": "True"}],
            "containerStatuses": [{"restartCount": 0}],
        },
    }
    service = {
        "metadata": {
            "name": lane.SERVICE_NAME,
            "namespace": lane.NAMESPACE,
            "uid": lane.SERVICE_UID,
        },
        "spec": {
            "ports": [
                {"name": "serve", "port": lane.SERVICE_PORT, "targetPort": lane.SERVICE_PORT}
            ]
        },
    }
    return rayjob, pod, service


def test_live_gate_binds_fresh_ready_restart_free_objects() -> None:
    assert lane.validate_live_objects(*_objects()) == {
        "rayjob_uid": lane.RAYJOB_UID,
        "head_pod_uid": lane.HEAD_POD_UID,
        "service_uid": lane.SERVICE_UID,
        "head_pod_restarts": 0,
        "ready": True,
    }


@pytest.mark.parametrize(
    ("object_index", "path", "value"),
    [
        (0, ("metadata", "uid"), "00000000-0000-0000-0000-000000000001"),
        (1, ("metadata", "uid"), "00000000-0000-0000-0000-000000000002"),
        (1, ("status", "phase"), "Failed"),
        (1, ("status", "containerStatuses"), [{"restartCount": 1}]),
        (2, ("metadata", "uid"), "00000000-0000-0000-0000-000000000003"),
        (2, ("spec", "ports"), [{"name": "serve", "port": 8001, "targetPort": 8001}]),
    ],
)
def test_live_gate_fails_closed_on_identity_or_health_drift(
    object_index: int, path: tuple[str, str], value: object
) -> None:
    objects = list(copy.deepcopy(_objects()))
    objects[object_index][path[0]][path[1]] = value
    with pytest.raises(lane.DedicatedLaptopError):
        lane.validate_live_objects(*objects)


def test_runner_reads_only_exact_three_objects() -> None:
    objects = _objects()
    expected = {
        ("rayjob.ray.io", lane.RAYJOB_NAME): objects[0],
        ("pod", lane.HEAD_POD_NAME): objects[1],
        ("service", lane.SERVICE_NAME): objects[2],
    }
    seen: list[list[str]] = []

    def runner(argv: list[str]) -> str:
        seen.append(argv)
        return json.dumps(expected[(argv[4], argv[5])])

    lane.validate_live_with_runner(runner)
    assert seen == [
        ["kubectl", "-n", lane.NAMESPACE, "get", kind, name, "-o", "json"]
        for kind, name in expected
    ]


def test_port_forward_is_uid_gateable_ipv4_local_and_secret_free() -> None:
    argv = lane.port_forward_argv()
    assert argv == [
        "kubectl",
        "-n",
        lane.NAMESPACE,
        "port-forward",
        "--address=127.0.0.1",
        f"service/{lane.SERVICE_NAME}",
        f"{lane.LOCAL_DEFAULT_PORT}:{lane.SERVICE_PORT}",
    ]
    assert "FLEET_API_KEY" not in " ".join(argv)


def test_rank3_plan_is_complete_sequential_held_and_dedicated_only() -> None:
    plan = lane.held_plan(ROOT)
    assert plan["selection_rank"] == 3
    assert [row["attempt"] for row in plan["cells"]] == [1, 2, 3, 4]
    assert plan["execution"] == {
        "scored_launch_authorized": False,
        "whole_task_partition": True,
        "attempt_order": [1, 2, 3, 4],
        "maximum_concurrent_attempts": 1,
        "automatic_retry": False,
        "required_attempt_order": [1, 2, 3, 4],
    }
    assert plan["route"]["shared_hosted_route_allowed"] is False
    assert plan["server"] == lane.server_binding()
    assert plan["plan_sha256"] == crypto.digest_without(plan, "plan_sha256")


def _stream(name: str, elapsed: float = 4.0) -> dict:
    value = {
        "schema_version": lane.STREAM_SCHEMA,
        "lane": name,
        "server_binding": lane.server_binding(),
        "non_scored": True,
        "generic_non_benchmark": True,
        "tools": ["bash", "submit_report"],
        "requests_started": 2,
        "requests_succeeded": 2,
        "protocol_valid_requests": 2,
        "elapsed_seconds": elapsed,
        "request_latency_seconds": {"min": 1.0, "max": 3.0},
        "response_content_persisted": False,
        "task_instance_session_verifier_scoring_mutations": 0,
        "passed": True,
    }
    value["receipt_sha256"] = crypto.digest_without(value, "receipt_sha256")
    return value


def test_local_stream_persists_only_aggregate_protocol_facts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ticks = iter([10.0, 14.0])
    monkeypatch.setattr(lane.time, "monotonic", lambda: next(ticks))
    receipt = lane.run_local_stream("http://127.0.0.1:18527", caller=lambda _o, _t: 2.0)
    assert receipt["requests_succeeded"] == 2
    assert receipt["tools"] == ["bash", "submit_report"]
    assert receipt["response_content_persisted"] is False
    assert receipt["task_instance_session_verifier_scoring_mutations"] == 0
    assert not ({"messages", "choices", "tool_calls", "arguments"} & set(receipt))


def test_two_stream_qualification_requires_one_cluster_and_one_local() -> None:
    receipt = lane.combine_concurrency2(
        _stream("cluster_internal", 3.0), _stream("local_port_forward", 4.0)
    )
    assert receipt["status"] == "PASSED_NON_SCORED"
    assert receipt["concurrency"] == 2
    assert receipt["streams_succeeded"] == 2
    assert receipt["requests_succeeded"] == 4
    assert receipt["streams_per_minute"] == 30.0
    assert receipt["scored_launch_authorized"] is False
    assert receipt["receipt_sha256"] == crypto.digest_without(receipt, "receipt_sha256")


@pytest.mark.parametrize(
    "mutator",
    [
        lambda row: row.__setitem__("lane", "local_port_forward"),
        lambda row: row.__setitem__("server_binding", {}),
        lambda row: row.__setitem__("requests_succeeded", 1),
        lambda row: row.__setitem__("task_instance_session_verifier_scoring_mutations", 1),
        lambda row: row.__setitem__("response_content_persisted", True),
    ],
)
def test_two_stream_qualification_fails_closed(mutator) -> None:
    cluster = _stream("cluster_internal")
    mutator(cluster)
    cluster["receipt_sha256"] = crypto.digest_without(cluster, "receipt_sha256")
    with pytest.raises(lane.DedicatedLaptopError):
        lane.combine_concurrency2(cluster, _stream("local_port_forward"))
