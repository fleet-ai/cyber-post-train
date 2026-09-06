from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from evals.fleet import qwen38_dp8_post_rank99_launch_v1 as launch
from evals.fleet import self_hosted

ROOT = Path(__file__).resolve().parents[1]


def _binding() -> dict[str, object]:
    return {
        "api_run_id": "ft-run-example",
        "rayjob_uid": "11111111-1111-4111-8111-111111111111",
        "head_pod_uid": "22222222-2222-4222-8222-222222222222",
        "service_uid": "33333333-3333-4333-8333-333333333333",
        "served_id": "qwen3.8-27b",
        "model_revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        "context_length": 262144,
    }


def _release() -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": launch.TRANSITION_SCHEMA,
        "status": "RELEASED_FOR_ONE_NON_SCORED_DP8_SERVER",
        "launch_authorized": True,
        "scoring_authorized": False,
        "title": "chris-cyber-evalserve-q38-dp8-b-v1",
        "run_dir": "/mnt/sfs/jobs/chris-cyber-evalserve-q38-dp8-b-v1",
        "serving_block": "dedicated-qwen-dp8-b-v1",
        "rank99_accepted_validated_count": 4,
        "rank99_accepted_validated_receipts": [f"sha256:{value:064x}" for value in range(1, 5)],
        "rank99_controller_terminal_success": True,
        "tp1_jobs_api_delete_http_status": 204,
        "tp1_jobs_api_get_http_status_after_delete": 404,
        "tp1_runtime_objects_absent": True,
        "tp1_gpu_released": True,
        "fresh_title_matches": 0,
        "fresh_run_dir_matches": 0,
        "fresh_sfs_run_dir_exists": False,
        "project_gpu_nodes_before_create": 1,
        "project_gpus_before_create": 8,
        "api_mutations": 0,
        "prompts_traces_flags_or_scores_included": False,
    }
    value["receipt_sha256"] = self_hosted.digest_without(value, "receipt_sha256")
    return value


def _live_gate() -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": launch.LIVE_SUBMIT_GATE_SCHEMA,
        "status": "CLEAR",
        "title": "chris-cyber-evalserve-q38-dp8-b-v1",
        "run_dir": "/mnt/sfs/jobs/chris-cyber-evalserve-q38-dp8-b-v1",
        "jobs_api_title_matches": 0,
        "jobs_api_run_dir_matches": 0,
        "kubernetes_identity_matches": 0,
        "sfs_run_dir_exists": False,
        "project_gpu_nodes_before_create": 1,
        "project_gpus_before_create": 8,
        "tp1_objects_absent": True,
        "tp1_gpu_released": True,
        "observed_after_transition_release": True,
        "freshness_seconds_at_submit": 5,
        "api_mutations": 0,
        "prompts_traces_flags_or_scores_included": False,
    }
    value["receipt_sha256"] = self_hosted.digest_without(value, "receipt_sha256")
    return value


def test_held_config_renders_exact_nonpreempting_full_node_payload() -> None:
    value = launch.spec(ROOT)
    payload = launch.jobs_payload(ROOT)
    assert value["launch_authorized"] is False
    assert value["scoring_authorized"] is False
    assert payload["title"] == "chris-cyber-evalserve-q38-dp8-b-v1"
    assert payload["run_dir"] == "/mnt/sfs/jobs/chris-cyber-evalserve-q38-dp8-b-v1"
    assert payload["workers"] == 1
    assert payload["gpus_per_worker"] == 8
    assert payload["priority_class"] == "fleet-infra-quiet"
    assert payload["privileged"] is False
    assert "--tp-size 1" in payload["command"]
    assert "--dp-size 8" in payload["command"]
    assert value["runtime"]["jobs_api_payload_sha256"] == self_hosted.sha256(
        self_hosted.canonical_json(payload)
    )


def test_transition_release_requires_complete_tp1_teardown_and_four_acceptances() -> None:
    launch.validate_transition_release(_release())
    for field, changed in (
        ("rank99_accepted_validated_count", 3),
        ("rank99_controller_terminal_success", False),
        ("tp1_jobs_api_get_http_status_after_delete", 200),
        ("tp1_runtime_objects_absent", False),
        ("tp1_gpu_released", False),
        ("fresh_title_matches", 1),
        ("project_gpu_nodes_before_create", 2),
        ("scoring_authorized", True),
    ):
        value = _release()
        value[field] = changed
        value["receipt_sha256"] = self_hosted.digest_without(value, "receipt_sha256")
        with pytest.raises(ValueError):
            launch.validate_transition_release(value)


def test_preview_validator_requires_exact_jobs_api_render() -> None:
    payload = launch.jobs_payload(ROOT)
    manifest = {
        "kind": "RayJob",
        "metadata": {"labels": {"kueue.x-k8s.io/queue-name": "training-lq"}},
        "spec": {
            "suspend": True,
            "entrypoint": payload["command"],
            "rayClusterSpec": {
                "headGroupSpec": {
                    "template": {
                        "metadata": {
                            "annotations": {
                                "kueue.x-k8s.io/podset-preferred-topology": (
                                    "topology.nebius.com/tier-1"
                                )
                            }
                        },
                        "spec": {
                            "imagePullSecrets": [{"name": "ghcr-pull"}],
                            "priorityClassName": "fleet-infra-quiet",
                            "containers": [
                                {
                                    "image": payload["image"],
                                    "env": [{"name": "RUN_DIR", "value": payload["run_dir"]}],
                                    "resources": {
                                        "requests": {"nvidia.com/gpu": 8},
                                        "limits": {"nvidia.com/gpu": 8},
                                    },
                                }
                            ],
                        },
                    }
                }
            },
        },
    }
    assert launch.preview_identity(yaml.safe_dump(manifest), ROOT)["gpus"] == 8
    manifest["spec"]["rayClusterSpec"]["headGroupSpec"]["template"]["spec"]["containers"][0][
        "resources"
    ]["requests"]["nvidia.com/gpu"] = 4
    with pytest.raises(RuntimeError):
        launch.preview_identity(yaml.safe_dump(manifest), ROOT)


def test_authenticated_preview_receipt_is_exact_and_non_mutating() -> None:
    value = json.loads((ROOT / launch.PREVIEW_RECEIPT_PATH).read_text())
    launch.validate_preview_receipt(value, ROOT)
    assert value["side_effects"] == {
        "api_mutations": 0,
        "launch_authorized": False,
        "scoring_authorized": False,
    }
    for path, replacement in (
        (("rendered", "gpus"), 4),
        (("rendered", "suspended"), False),
        (("side_effects", "api_mutations"), 1),
    ):
        changed = copy.deepcopy(value)
        cursor = changed
        for key in path[:-1]:
            cursor = cursor[key]
        cursor[path[-1]] = replacement
        changed["receipt_sha256"] = self_hosted.digest_without(changed, "receipt_sha256")
        with pytest.raises(ValueError):
            launch.validate_preview_receipt(changed, ROOT)


def test_create_once_submit_requires_release_and_immediate_live_gate(monkeypatch) -> None:
    payload = launch.jobs_payload(ROOT)
    manifest = {
        "kind": "RayJob",
        "metadata": {"labels": {"kueue.x-k8s.io/queue-name": "training-lq"}},
        "spec": {
            "suspend": True,
            "entrypoint": payload["command"],
            "rayClusterSpec": {
                "headGroupSpec": {
                    "template": {
                        "metadata": {
                            "annotations": {
                                "kueue.x-k8s.io/podset-preferred-topology": (
                                    "topology.nebius.com/tier-1"
                                )
                            }
                        },
                        "spec": {
                            "imagePullSecrets": [{"name": "ghcr-pull"}],
                            "priorityClassName": "fleet-infra-quiet",
                            "containers": [
                                {
                                    "image": payload["image"],
                                    "env": [{"name": "RUN_DIR", "value": payload["run_dir"]}],
                                    "resources": {
                                        "requests": {"nvidia.com/gpu": 8},
                                        "limits": {"nvidia.com/gpu": 8},
                                    },
                                }
                            ],
                        },
                    }
                }
            },
        },
    }

    class Response:
        def __init__(self, status_code, value):
            self.status_code = status_code
            self._value = value

        def raise_for_status(self):
            assert self.status_code < 400

        def json(self):
            return self._value

    class Client:
        def __init__(self):
            self.routes = []

        def post(self, route, json):
            assert json == payload
            self.routes.append(route)
            if route.endswith("preview"):
                return Response(200, {"manifest_yaml": yaml.safe_dump(manifest)})
            return Response(202, {"name": "ft-run-fresh"})

        def get(self, route, params=None):
            assert route == "/v1/runs"
            assert params is not None
            return Response(200, {"runs": [], "next_cursor": None})

    # Avoid depending on wall-clock formatting in this pure create-once unit test.
    class Clock:
        @classmethod
        def now(cls, _zone):
            class Value:
                def isoformat(self):
                    return "2026-09-06T05:00:00+00:00"

            return Value()

    monkeypatch.setattr(launch, "datetime", Clock)
    client = Client()
    receipt = launch.submit_create_once(
        client, _release(), _live_gate(), ROOT, source_commit="a" * 40
    )
    assert client.routes == ["/v1/runs/preview", "/v1/runs"]
    assert receipt["status"] == "SUBMITTED_NON_SCORED_SERVER"
    assert receipt["scored_calls"] == 0
    changed = _live_gate()
    changed["tp1_gpu_released"] = False
    changed["receipt_sha256"] = self_hosted.digest_without(changed, "receipt_sha256")
    with pytest.raises(ValueError):
        launch.submit_create_once(Client(), _release(), changed, ROOT, source_commit="a" * 40)


def test_server_binding_requires_exact_immutable_runtime_and_uids() -> None:
    submission = {
        "schema_version": launch.SUBMISSION_SCHEMA,
        "status": "SUBMITTED_NON_SCORED_SERVER",
        "api_run_id": "ft-run-fresh",
        "title": "chris-cyber-evalserve-q38-dp8-b-v1",
        "run_dir": "/mnt/sfs/jobs/chris-cyber-evalserve-q38-dp8-b-v1",
        "serving_block": "dedicated-qwen-dp8-b-v1",
        "route": "POST /v1/runs",
        "http_status": 202,
        "server_instances_created": 1,
        "scored_calls": 0,
        "prompts_traces_flags_or_scores_included": False,
    }
    submission["receipt_sha256"] = self_hosted.digest_without(submission, "receipt_sha256")
    value = {
        "schema_version": launch.SERVER_BINDING_SCHEMA,
        "status": "READY_NON_SCORED",
        "submission_receipt_sha256": submission["receipt_sha256"],
        "api_run_id": submission["api_run_id"],
        "title": "chris-cyber-evalserve-q38-dp8-b-v1",
        "run_dir": "/mnt/sfs/jobs/chris-cyber-evalserve-q38-dp8-b-v1",
        "serving_block": "dedicated-qwen-dp8-b-v1",
        "service_origin": "http://ft-run-fresh-example-head-svc:8000",
        "rayjob_uid": "11111111-1111-4111-8111-111111111111",
        "workload_uid": "22222222-2222-4222-8222-222222222222",
        "head_pod_uid": "33333333-3333-4333-8333-333333333333",
        "service_uid": "44444444-4444-4444-8444-444444444444",
        "image": launch.prior.IMAGE,
        "model_revision": launch.prior.MODEL_REVISION,
        "context_length": 262144,
        "tensor_parallel_size": 1,
        "data_parallel_size": 8,
        "head_pod_running_ready": True,
        "head_pod_restarts": 0,
        "kueue_preempted": False,
        "scoring_authorized": False,
        "prompts_traces_flags_or_scores_included": False,
    }
    value["receipt_sha256"] = self_hosted.digest_without(value, "receipt_sha256")
    assert launch.validate_server_binding_receipt(value, submission)["api_run_id"] == (
        "ft-run-fresh"
    )
    changed = copy.deepcopy(value)
    changed["data_parallel_size"] = 4
    changed["receipt_sha256"] = self_hosted.digest_without(changed, "receipt_sha256")
    with pytest.raises(ValueError):
        launch.validate_server_binding_receipt(changed, submission)


def _probe(status: str = "PASSED_NON_SCORED") -> dict[str, object]:
    value: dict[str, object] = {
        "status": status,
        "tool_contract": {
            "calls_observed_in_order": ["bash", "submit_report"],
            "arguments_structurally_valid": True,
        },
        "execution": {"task_instance_session_verifier_scoring_calls": 0},
        "privacy": {"responses_or_model_outputs_included": False},
    }
    value["receipt_sha256"] = self_hosted.digest_without(value, "receipt_sha256")
    return value


def test_actual_opencode_ladder_stops_before_next_level_on_failure() -> None:
    plan = launch.qualification_plan(_binding(), ROOT)
    calls = 0

    def probe():
        nonlocal calls
        calls += 1
        return _probe("FAILED" if calls == 3 else "PASSED_NON_SCORED")

    result = launch.run_qualification_ladder(plan, probe)
    assert [row["concurrency"] for row in result["levels"]] == [1, 2]
    assert result["highest_passing_concurrency"] == 1
    assert calls == 3


def test_scored_partition_includes_only_whole_unstarted_task_boundaries() -> None:
    plan = launch.qualification_plan(_binding(), ROOT)
    result = launch.run_qualification_ladder(plan, _probe)
    cells = [
        {
            "task_version_id": "task-a",
            "attempt": attempt,
            "cell_id": f"sha256:{attempt:064x}",
            "execution_id": f"sha256:{attempt + 10:064x}",
            "status": "unstarted",
            "accepted_receipt_sha256": None,
            "active_claim_sha256": None,
            "authoritative_session_matches": 0,
            "output_root_exists": False,
        }
        for attempt in range(1, 5)
    ]
    held = launch.held_scored_partition(result, cells)
    assert held["launch_authorized"] is False
    assert held["scoring_authorized"] is False
    assert held["qualified_concurrency_ceiling"] == 8
    assert [row["attempt"] for row in held["whole_task_partitions"][0]["cells"]] == [
        1,
        2,
        3,
        4,
    ]
    changed = copy.deepcopy(cells)
    changed[2]["status"] = "accepted"
    changed[2]["accepted_receipt_sha256"] = "sha256:" + "a" * 64
    with pytest.raises(ValueError):
        launch.held_scored_partition(result, changed)


def test_qualification_plan_is_exact_score_free_ladder() -> None:
    value = launch.qualification_plan(_binding(), ROOT)
    assert [row["concurrency"] for row in value["waves"]] == [1, 2, 4, 8]
    assert [row["request_count"] for row in value["waves"]] == [2, 4, 8, 16]
    assert all(row["tool_order"] == ["bash", "submit_report"] for row in value["waves"])
    assert set(value["side_effects"].values()) == {0}
    assert value["treatment"]["tool_catalog_sha256"] == (
        "sha256:85fad6bdc3a835bf52a11a99b3387740eb06eb3d1720ad9bb33f3feac215b44a"
    )


def _result(plan: dict[str, object], count: int = 4) -> dict[str, object]:
    rows = [
        {
            "concurrency": concurrency,
            "status": "PASSED",
            "request_count": concurrency * 2,
            "completed_count": concurrency * 2,
            "error_count": 0,
            "tool_order_exact": True,
            "tool_arguments_exact": True,
        }
        for concurrency in launch.LEVELS[:count]
    ]
    value: dict[str, object] = {
        "schema_version": launch.RESULT_SCHEMA,
        "plan_receipt_sha256": plan["receipt_sha256"],
        "levels": rows,
        "highest_passing_concurrency": rows[-1]["concurrency"],
        "scored_calls": 0,
        "prompts_traces_flags_or_scores_included": False,
    }
    value["receipt_sha256"] = self_hosted.digest_without(value, "receipt_sha256")
    return value


def test_qualification_result_fails_closed_on_partial_or_protocol_drift() -> None:
    plan = launch.qualification_plan(_binding(), ROOT)
    assert launch.validate_result(_result(plan), plan) == 8
    with pytest.raises(ValueError, match="stopped without"):
        launch.validate_result(_result(plan, 2), plan)
    changed = _result(plan)
    changed["levels"][2]["tool_arguments_exact"] = False
    changed["receipt_sha256"] = self_hosted.digest_without(changed, "receipt_sha256")
    with pytest.raises(ValueError):
        launch.validate_result(changed, plan)


def test_held_spec_rejects_authorization_or_runtime_drift() -> None:
    original = launch.spec(ROOT)
    for path, replacement in (
        (("launch_authorized",), True),
        (("scoring_authorized",), True),
        (("runtime", "data_parallel_size"), 4),
        (("qualification", "levels"), [8]),
    ):
        value = copy.deepcopy(original)
        cursor = value
        for key in path[:-1]:
            cursor = cursor[key]
        cursor[path[-1]] = replacement
        value["config_sha256"] = self_hosted.digest_without(value, "config_sha256")
        with pytest.raises(ValueError):
            launch.validate_spec(value, ROOT)
