from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from training.checkpoint_serving_route import (
    RouteError,
    _digest,
    _normalized_contract,
    _validate_plan,
    build_paused_spec,
    lifecycle,
    reconcile_create,
    wait_phase,
)

ROOT = Path(__file__).resolve().parents[1]
LR30_ROUTE = ROOT / "configs/evaluation/qwen38-lr30-step76-serving-route-plan-v1.json"
LR30_REGISTRATION = (
    ROOT / "docs/evidence/qwen38-lr30-step76-matched-serving-registration-20260921.json"
)
STEP600_ROUTE = ROOT / "configs/evaluation/qwen38-teacher3k32-step600-serving-route-plan-v1.json"


def _base() -> dict:
    return {
        "displayName": "Base",
        "desiredState": "serving",
        "capabilities": ["chat_completions", "reasoning", "streaming", "tool_calling"],
        "model": {
            "sourcePath": "/models/base/rev",
            "path": "/scratch/models/base/rev",
            "revision": "base-revision",
            "precision": "bf16",
            "tensorParallelSize": 1,
            "dataParallelSize": 8,
            "dataParallelAttention": False,
        },
        "runtime": {
            "engine": "sglang",
            "args": ["--model-path", "/scratch/models/base/rev", "--served-model-name", "base"],
        },
        "placement": {"nodeGroup": "gpu", "priorityClassName": "c0"},
        "resources": {
            "requests": {"cpu": "96", "memory": "1Ti", "nvidia.com/gpu": 8},
            "limits": {"cpu": "192", "memory": "2Ti", "nvidia.com/gpu": 8},
        },
        "routing": {"enabled": True},
        "scaling": {"minReplicas": 1, "replicas": 2},
    }


def test_build_paused_spec_changes_only_weights_identity_and_lifecycle() -> None:
    base = _base()
    candidate = build_paused_spec(
        base,
        model_id="chris-q38-test-v1",
        display_name="Candidate",
        source_path="/models/chris-q38-test-v1",
        revision="sha256:" + "a" * 64,
    )
    assert candidate["desiredState"] == "paused"
    assert candidate["placement"]["priorityClassName"] == "c1"
    assert candidate["model"]["tensorParallelSize"] == 1
    assert candidate["model"]["dataParallelSize"] == 8
    assert candidate["scaling"] == {"minReplicas": 0, "replicas": 1}
    assert _normalized_contract(candidate) == _normalized_contract(base)


def test_contract_detects_runtime_drift() -> None:
    base = _base()
    candidate = build_paused_spec(
        base,
        model_id="chris-q38-test-v1",
        display_name="Candidate",
        source_path="/models/chris-q38-test-v1",
        revision="sha256:" + "a" * 64,
    )
    drifted = copy.deepcopy(candidate)
    drifted["runtime"]["args"].append("--different-runtime")
    assert _normalized_contract(drifted) != _normalized_contract(base)


def test_lr30_step76_route_plan_is_offline_paused_exact_base_clone() -> None:
    plan = json.loads(LR30_ROUTE.read_text())
    _validate_plan(plan)
    registration = plan["registration"]
    spec = registration["spec"]

    assert plan["source_model_id"] == "chris-q38-available-a-lr30-step76-v1"
    assert plan["base_model_id"] == "qwen3.8-27b"
    assert plan["mutation_count"] == 0
    assert registration["id"] == "chris-q38-lr30-step76-web-v1"
    assert spec["desiredState"] == "paused"
    assert spec["scaling"] == {"minReplicas": 0, "replicas": 1}
    assert spec["placement"]["priorityClassName"] == "c1"
    assert spec["model"]["sourcePath"] == "/models/chris-q38-available-a-lr30-step76-v1"
    assert spec["model"]["revision"] == (
        "sha256:3bef11697759b11db150b705e1882fb2d31a3efbdadbebd948aa917dcacc03a8"
    )
    assert plan["normalized_contract_sha256"] == (
        "sha256:d82d78721f4ec8d4b0d6228242df4838b38086a48108e57fc6fe4e7be1fa3662"
    )


def test_teacher3k32_step600_route_plan_is_paused_and_exactly_bound() -> None:
    plan = json.loads(STEP600_ROUTE.read_text())
    _validate_plan(plan)
    registration = plan["registration"]
    spec = registration["spec"]

    assert plan["source_model_id"] == "chris-q38-t3k32-s600-v1"
    assert plan["source_resource_version"] == "32786336"
    assert plan["base_model_id"] == "qwen3.8-27b"
    assert plan["mutation_count"] == 0
    assert registration["id"] == "chris-q38-t3k32-s600-web-v1"
    assert spec["desiredState"] == "paused"
    assert spec["scaling"] == {"minReplicas": 0, "replicas": 1}
    assert spec["placement"]["priorityClassName"] == "c1"
    assert spec["model"]["sourcePath"] == "/models/chris-q38-t3k32-s600-v1"
    assert spec["model"]["revision"] == (
        "sha256:adf5d8c6609ea441744eab13ed0649c22eb4ad5f95baeebdcd60880c206702a6"
    )
    assert plan["normalized_contract_sha256"] == (
        "sha256:d82d78721f4ec8d4b0d6228242df4838b38086a48108e57fc6fe4e7be1fa3662"
    )


def test_lr30_step76_route_registration_is_paused_zero_gpu_and_bound_to_plan() -> None:
    plan = json.loads(LR30_ROUTE.read_text())
    evidence = json.loads(LR30_REGISTRATION.read_text())
    claimed = evidence.pop("sha256")
    committed_plan = evidence["registration"]["committed_plan"]

    assert claimed == _digest(evidence)
    assert committed_plan["plan_sha256"] == plan["plan_sha256"]
    assert committed_plan["registration_sha256"] == plan["registration_sha256"]
    assert evidence["registration"]["create_result"]["post_attempts"] == 1
    assert evidence["registration"]["create_result"]["second_post_performed"] is False
    assert evidence["target_readback"]["spec_exactly_matches_committed_plan"] is True
    assert evidence["target_readback"]["desired_state"] == "paused"
    assert evidence["target_readback"]["minimum_replicas"] == 0
    assert evidence["target_readback"]["active_pods"] == 0
    assert evidence["target_readback"]["ready_replicas"] == 0
    assert evidence["target_readback"]["pod_count"] == 0
    assert evidence["target_readback"]["gpus_allocated"] == 0
    assert evidence["scientific_boundary"]["live_route_resumed"] is False
    assert evidence["scientific_boundary"]["evaluation_launched"] is False
    assert evidence["next_gate"]["external_evaluation_launchable"] is False


def test_rejects_non_sha_revision() -> None:
    with pytest.raises(RouteError, match="invalid_revision"):
        build_paused_spec(
            _base(),
            model_id="chris-q38-test-v1",
            display_name="Candidate",
            source_path="/models/chris-q38-test-v1",
            revision="mutable",
        )


def test_reconcile_create_never_posts(tmp_path) -> None:
    spec = build_paused_spec(
        _base(),
        model_id="chris-q38-test-v1",
        display_name="Candidate",
        source_path="/models/chris-q38-test-v1",
        revision="sha256:" + "a" * 64,
    )
    registration = {"id": "chris-q38-test-v1", "spec": spec}
    plan = {
        "schema": "cyber_checkpoint_serving_route_plan_v1",
        "registration": registration,
        "registration_sha256": _digest(registration),
        "mutation_count": 0,
    }
    plan["plan_sha256"] = _digest(plan)
    intent = {
        "schema": "cyber_checkpoint_serving_route_intent_v1",
        "plan_sha256": plan["plan_sha256"],
        "registration_sha256": plan["registration_sha256"],
        "post_will_be_attempted": True,
    }
    intent["receipt_sha256"] = _digest(intent)
    plan_path = tmp_path / "plan.json"
    intent_path = tmp_path / "intent.json"
    result_path = tmp_path / "result.json"
    plan_path.write_text(json.dumps(plan))
    intent_path.write_text(json.dumps(intent))

    class FakeClient:
        methods: list[str] = []

        def request(self, method, url):
            self.methods.append(method)
            return 200, {
                "id": registration["id"],
                "spec": spec,
                "resource_version": "123",
                "status": {"phase": "paused"},
            }

    client = FakeClient()
    result = reconcile_create(client, plan_path, intent_path, result_path)
    assert client.methods == ["GET"]
    assert result["reconciled_after_create"] is True
    assert json.loads(result_path.read_text())["post_attempts"] == 1


def test_lifecycle_pause_uses_fresh_resource_version_once() -> None:
    class FakeClient:
        calls: list[tuple] = []

        def request(self, method, url, body=None, **kwargs):
            self.calls.append((method, url, body, kwargs))
            if method == "GET":
                return 200, {"resource_version": "123", "status": {"phase": "ready"}}
            assert method == "POST"
            return 202, {"resource_version": "124", "status": {"phase": "ready"}}

    client = FakeClient()
    result = lifecycle(client, "chris-q38-test-v1", "pause")

    assert client.calls == [
        (
            "GET",
            "https://inference.flt.build/fleet/v1/models/chris-q38-test-v1",
            None,
            {},
        ),
        (
            "POST",
            "https://inference.flt.build/fleet/v1/models/chris-q38-test-v1/pause",
            {},
            {"if_match": "123"},
        ),
    ]
    assert result == {
        "model_id": "chris-q38-test-v1",
        "action": "pause",
        "before_phase": "ready",
        "before_resource_version": "123",
        "accepted_phase": "ready",
        "accepted_resource_version": "124",
    }


def test_lifecycle_pause_rejects_non_ready_route_without_post() -> None:
    class FakeClient:
        methods: list[str] = []

        def request(self, method, url, body=None, **kwargs):
            self.methods.append(method)
            return 200, {"resource_version": "123", "status": {"phase": "paused"}}

    client = FakeClient()
    with pytest.raises(RouteError, match="route_not_ready"):
        lifecycle(client, "chris-q38-test-v1", "pause")
    assert client.methods == ["GET"]


def test_wait_phase_requires_exact_terminal_pause_state() -> None:
    class FakeClient:
        def request(self, method, url):
            assert method == "GET"
            return 200, {
                "resource_version": "125",
                "status": {"phase": "paused", "ready_replicas": 0, "active_pods": 0},
            }

    assert wait_phase(FakeClient(), "chris-q38-test-v1", "paused", 1) == {
        "model_id": "chris-q38-test-v1",
        "phase": "paused",
        "resource_version": "125",
        "ready_replicas": 0,
        "active_pods": 0,
    }
