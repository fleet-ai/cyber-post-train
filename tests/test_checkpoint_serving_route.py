from __future__ import annotations

import copy
import json
from datetime import timedelta
from pathlib import Path

import pytest

from cyber_post_train import gpu_capacity
from evals.external_ctf import execution_packet
from training import checkpoint_serving_route as route
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
EXTERNAL_CTF_CLONE_INTENT = (
    ROOT / "configs/evaluation/qwen38-external-ctf-base-c1-clone-intent-v1.json"
)
EXTERNAL_CTF_PROTOCOL = ROOT / "configs/evaluation/qwen38-external-ctf-paired-v1.json"


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


def _external_base_spec() -> dict:
    spec = copy.deepcopy(json.loads(STEP600_ROUTE.read_text())["registration"]["spec"])
    source_path = "/models/qwen3.8-27b/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
    scratch_path = "/scratch/models/qwen3.8-27b/base"
    spec["displayName"] = "Qwen3.8-27B base"
    spec["desiredState"] = "serving"
    spec["model"].update(
        {
            "sourcePath": source_path,
            "path": scratch_path,
            "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        }
    )
    arguments = spec["runtime"]["args"]
    arguments[arguments.index("--model-path") + 1] = scratch_path
    arguments[arguments.index("--served-model-name") + 1] = "qwen3.8-27b"
    spec["placement"]["priorityClassName"] = "c0"
    spec["scaling"] = {"minReplicas": 1, "replicas": 1}
    return spec


def _external_route_files(tmp_path: Path, monkeypatch):
    intent = route.load_external_ctf_clone_intent(EXTERNAL_CTF_CLONE_INTENT)
    base_spec = _external_base_spec()
    base = {
        "id": "qwen3.8-27b",
        "uid": "base-uid",
        "resource_version": "base-rv",
        "spec": base_spec,
        "status": {"phase": "ready", "ready_replicas": 1},
    }

    class PrepareClient:
        def request(self, method, url, _body=None, **kwargs):
            assert method == "GET"
            if url == route.ACCOUNT:
                return 200, {"team_id": route.TEAM_ID}
            if url.endswith("/" + route.EXTERNAL_CTF_BASE_CLONE_ID):
                assert kwargs.get("allow_404") is True
                return 404, None
            return 200, copy.deepcopy(base)

    plan = route.prepare(
        PrepareClient(),
        base_model_id="qwen3.8-27b",
        source_model_id="qwen3.8-27b",
        target_model_id=route.EXTERNAL_CTF_BASE_CLONE_ID,
        display_name=intent["target"]["display_name"],
        source_path=intent["source"]["source_path"],
        revision=intent["source"]["model_revision"],
        external_ctf_clone_intent=intent,
    )
    plan_path = tmp_path / "plan.json"
    plan_path.write_bytes(route._canonical(plan) + b"\n")  # noqa: SLF001
    result = {
        "schema": "cyber_checkpoint_serving_route_result_v1",
        "plan_sha256": plan["plan_sha256"],
        "registration_sha256": plan["registration_sha256"],
        "model_id": route.EXTERNAL_CTF_BASE_CLONE_ID,
        "resource_version": "target-rv",
        "phase": "paused",
        "post_attempts": 1,
    }
    result["receipt_sha256"] = route._digest(result)  # noqa: SLF001
    result_path = tmp_path / "result.json"
    result_path.write_bytes(route._canonical(result) + b"\n")  # noqa: SLF001
    target = {
        "id": route.EXTERNAL_CTF_BASE_CLONE_ID,
        "uid": "target-uid",
        "generation": 1,
        "resource_version": "target-rv",
        "spec": copy.deepcopy(plan["registration"]["spec"]),
        "status": {"phase": "paused", "ready_replicas": 0, "active_pods": 0},
    }
    candidate_spec = copy.deepcopy(plan["registration"]["spec"])
    candidate_path = intent["matched_candidate"]["source_path"]
    candidate_spec["displayName"] = "Step 1000"
    candidate_spec["desiredState"] = "serving"
    candidate_spec["model"].update(
        {
            "sourcePath": candidate_path,
            "path": "/scratch/models/chris-q38-t3k32-s1000-v1",
            "revision": intent["matched_candidate"]["model_revision"],
        }
    )
    arguments = candidate_spec["runtime"]["args"]
    arguments[arguments.index("--model-path") + 1] = candidate_spec["model"]["path"]
    arguments[arguments.index("--served-model-name") + 1] = route.EXTERNAL_CTF_CANDIDATE_ID
    candidate_spec["scaling"] = {"minReplicas": 1, "replicas": 1}
    candidate = {
        "id": route.EXTERNAL_CTF_CANDIDATE_ID,
        "uid": "candidate-uid",
        "generation": 4,
        "resource_version": "candidate-rv",
        "spec": candidate_spec,
        "status": {"phase": "ready", "ready_replicas": 1, "active_pods": 1},
    }
    capacity = gpu_capacity.build_capacity_census(
        {"items": []},
        {"items": []},
        {"items": []},
        {"items": []},
        owner_prefixes=("chris-q38-", "qwen3.8-27b"),
        max_nodes=route.EXTERNAL_CTF_CAPACITY_MAX_NODES,
        max_gpus=route.EXTERNAL_CTF_CAPACITY_MAX_GPUS,
        planned_nodes=1,
        planned_gpus=8,
    )

    class PreflightClient:
        def request(self, method, url, _body=None, **_kwargs):
            assert method == "GET"
            if url == route.ACCOUNT:
                return 200, {"team_id": route.TEAM_ID}
            if url.endswith("/" + route.EXTERNAL_CTF_BASE_CLONE_ID):
                return 200, copy.deepcopy(target)
            if url.endswith("/" + route.EXTERNAL_CTF_CANDIDATE_ID):
                return 200, copy.deepcopy(candidate)
            raise AssertionError(url)

    monkeypatch.setattr(
        route.gpu_capacity,
        "live_capacity_census",
        lambda *_args, **_kwargs: copy.deepcopy(capacity),
    )
    preflight_path = tmp_path / "resume-preflight.json"
    preflight = route.seal_external_ctf_resume_preflight(
        PreflightClient(),
        clone_intent_path=EXTERNAL_CTF_CLONE_INTENT,
        plan_path=plan_path,
        result_path=result_path,
        kubernetes_context="synthetic-context",
        output_path=preflight_path,
    )
    return {
        "intent": intent,
        "plan": plan,
        "plan_path": plan_path,
        "result": result,
        "result_path": result_path,
        "target": target,
        "candidate": candidate,
        "capacity": capacity,
        "preflight": preflight,
        "preflight_path": preflight_path,
    }


def test_external_ctf_clone_intent_is_exact_not_merely_self_digested(tmp_path: Path) -> None:
    value = json.loads(EXTERNAL_CTF_CLONE_INTENT.read_text())
    value["source"]["source_path"] = "/models/wrong-but-self-digested"
    unsigned = {key: item for key, item in value.items() if key != "intent_sha256"}
    value["intent_sha256"] = route._digest(unsigned)  # noqa: SLF001
    path = tmp_path / "tampered-intent.json"
    path.write_bytes(route._canonical(value) + b"\n")  # noqa: SLF001

    with pytest.raises(RouteError, match="external_ctf_clone_intent_digest_mismatch"):
        route.load_external_ctf_clone_intent(path)


def test_external_ctf_resume_preflight_binds_target_candidate_and_capacity(
    tmp_path: Path, monkeypatch
) -> None:
    fixture = _external_route_files(tmp_path, monkeypatch)

    loaded = route.load_external_ctf_resume_preflight(
        fixture["preflight_path"],
        clone_intent_path=EXTERNAL_CTF_CLONE_INTENT,
        plan_path=fixture["plan_path"],
        result_path=fixture["result_path"],
    )

    assert loaded == fixture["preflight"]
    assert loaded["target"]["id"] == route.EXTERNAL_CTF_BASE_CLONE_ID
    assert loaded["candidate"]["id"] == route.EXTERNAL_CTF_CANDIDATE_ID
    assert loaded["capacity"]["qualified"] is True


def test_external_ctf_resume_preflight_staleness_fails_closed(tmp_path: Path, monkeypatch) -> None:
    fixture = _external_route_files(tmp_path, monkeypatch)
    observed = route._timestamp(fixture["preflight"]["observed_at"])  # noqa: SLF001

    with pytest.raises(RouteError, match="external_ctf_resume_preflight_invalid_or_stale"):
        route.load_external_ctf_resume_preflight(
            fixture["preflight_path"],
            clone_intent_path=EXTERNAL_CTF_CLONE_INTENT,
            plan_path=fixture["plan_path"],
            result_path=fixture["result_path"],
            now=observed
            + timedelta(seconds=route.EXTERNAL_CTF_RESUME_PREFLIGHT_MAX_AGE_SECONDS + 1),
        )


def _lifecycle_arguments(fixture: dict) -> dict:
    return {
        "external_ctf_resume_preflight_path": fixture["preflight_path"],
        "external_ctf_clone_intent_path": EXTERNAL_CTF_CLONE_INTENT,
        "external_ctf_plan_path": fixture["plan_path"],
        "external_ctf_result_path": fixture["result_path"],
        "kubernetes_context": "synthetic-context",
    }


def _external_lifecycle_result(fixture: dict, output: Path) -> dict:
    class Client:
        def request(self, method, url, _body=None, **_kwargs):
            if method == "POST":
                return 202, {
                    "resource_version": "target-rv-resuming",
                    "status": {"phase": "resuming"},
                }
            if url == route.ACCOUNT:
                return 200, {"team_id": route.TEAM_ID}
            if url.endswith("/" + route.EXTERNAL_CTF_BASE_CLONE_ID):
                return 200, copy.deepcopy(fixture["target"])
            return 200, copy.deepcopy(fixture["candidate"])

    result = route.lifecycle(
        Client(),
        route.EXTERNAL_CTF_BASE_CLONE_ID,
        "resume",
        **_lifecycle_arguments(fixture),
    )
    output.write_bytes(route._canonical(result) + b"\n")  # noqa: SLF001
    return result


def test_external_ctf_resume_immediate_capacity_failure_makes_zero_posts(
    tmp_path: Path, monkeypatch
) -> None:
    fixture = _external_route_files(tmp_path, monkeypatch)
    rejected = copy.deepcopy(fixture["capacity"])
    rejected["qualified"] = False
    rejected["problems"] = ["synthetic_limit"]
    rejected["sha256"] = gpu_capacity._digest(  # noqa: SLF001
        {key: item for key, item in rejected.items() if key != "sha256"}
    )
    monkeypatch.setattr(
        route.gpu_capacity, "live_capacity_census", lambda *_args, **_kwargs: rejected
    )

    class Client:
        calls: list[tuple] = []

        def request(self, method, url, _body=None, **kwargs):
            self.calls.append((method, url, kwargs))
            assert url == route.ACCOUNT
            return 200, {"team_id": route.TEAM_ID}

    client = Client()
    with pytest.raises(RouteError, match="external_ctf_immediate_resume_capacity_not_qualified"):
        route.lifecycle(
            client,
            route.EXTERNAL_CTF_BASE_CLONE_ID,
            "resume",
            **_lifecycle_arguments(fixture),
        )
    assert all(call[0] != "POST" for call in client.calls)


@pytest.mark.parametrize("drift_arm", ["target", "candidate"])
def test_external_ctf_resume_route_drift_makes_zero_posts(
    tmp_path: Path, monkeypatch, drift_arm: str
) -> None:
    fixture = _external_route_files(tmp_path, monkeypatch)
    target = copy.deepcopy(fixture["target"])
    candidate = copy.deepcopy(fixture["candidate"])
    (target if drift_arm == "target" else candidate)["resource_version"] += "-drift"

    class Client:
        calls: list[tuple] = []

        def request(self, method, url, _body=None, **kwargs):
            self.calls.append((method, url, kwargs))
            assert method == "GET"
            if url == route.ACCOUNT:
                return 200, {"team_id": route.TEAM_ID}
            if url.endswith("/" + route.EXTERNAL_CTF_BASE_CLONE_ID):
                return 200, target
            return 200, candidate

    client = Client()
    with pytest.raises(
        RouteError, match=f"external_ctf_resume_{drift_arm}_drifted_after_preflight"
    ):
        route.lifecycle(
            client,
            route.EXTERNAL_CTF_BASE_CLONE_ID,
            "resume",
            **_lifecycle_arguments(fixture),
        )
    assert all(call[0] != "POST" for call in client.calls)


def test_external_ctf_resume_posts_once_with_if_match_and_durable_capacity(
    tmp_path: Path, monkeypatch
) -> None:
    fixture = _external_route_files(tmp_path, monkeypatch)

    class Client:
        calls: list[tuple] = []

        def request(self, method, url, _body=None, **kwargs):
            self.calls.append((method, url, kwargs))
            if method == "POST":
                return 202, {
                    "resource_version": "target-rv-resuming",
                    "status": {"phase": "resuming"},
                }
            if url == route.ACCOUNT:
                return 200, {"team_id": route.TEAM_ID}
            if url.endswith("/" + route.EXTERNAL_CTF_BASE_CLONE_ID):
                return 200, copy.deepcopy(fixture["target"])
            return 200, copy.deepcopy(fixture["candidate"])

    client = Client()
    result = route.lifecycle(
        client,
        route.EXTERNAL_CTF_BASE_CLONE_ID,
        "resume",
        **_lifecycle_arguments(fixture),
    )

    posts = [call for call in client.calls if call[0] == "POST"]
    assert len(posts) == 1
    assert posts[0][2]["if_match"] == fixture["target"]["resource_version"]
    assert result["immediate_capacity"] == fixture["capacity"]
    assert result["immediate_capacity_sha256"] == "sha256:" + fixture["capacity"]["sha256"]
    assert result["target_readback"]["uid"] == fixture["target"]["uid"]
    assert result["candidate_readback"]["uid"] == fixture["candidate"]["uid"]
    assert result["receipt_sha256"] == route._digest(  # noqa: SLF001
        {key: item for key, item in result.items() if key != "receipt_sha256"}
    )


def test_execution_packet_accepts_exact_base_clone_evidence(tmp_path: Path, monkeypatch) -> None:
    fixture = _external_route_files(tmp_path, monkeypatch)
    lifecycle_path = tmp_path / "lifecycle.json"
    lifecycle = _external_lifecycle_result(fixture, lifecycle_path)
    protocol = json.loads(EXTERNAL_CTF_PROTOCOL.read_text())

    evidence = execution_packet._clone_serving_evidence(  # noqa: SLF001
        protocol=protocol,
        clone_intent_path=EXTERNAL_CTF_CLONE_INTENT,
        clone_plan_path=fixture["plan_path"],
        clone_result_path=fixture["result_path"],
        clone_resume_preflight_path=fixture["preflight_path"],
        clone_lifecycle_result_path=lifecycle_path,
    )

    assert evidence["intent"]["intent_sha256"] == fixture["intent"]["intent_sha256"]
    assert evidence["plan"]["plan_sha256"] == fixture["plan"]["plan_sha256"]
    assert evidence["resume_result"]["receipt_sha256"] == lifecycle["receipt_sha256"]


def test_execution_packet_rejects_rehashed_unqualified_clone_capacity(
    tmp_path: Path, monkeypatch
) -> None:
    fixture = _external_route_files(tmp_path, monkeypatch)
    lifecycle_path = tmp_path / "lifecycle.json"
    lifecycle = _external_lifecycle_result(fixture, lifecycle_path)
    lifecycle["immediate_capacity"]["qualified"] = False
    lifecycle["immediate_capacity"]["sha256"] = gpu_capacity._digest(  # noqa: SLF001
        {key: item for key, item in lifecycle["immediate_capacity"].items() if key != "sha256"}
    )
    lifecycle["immediate_capacity_sha256"] = "sha256:" + lifecycle["immediate_capacity"]["sha256"]
    lifecycle["receipt_sha256"] = route._digest(  # noqa: SLF001
        {key: item for key, item in lifecycle.items() if key != "receipt_sha256"}
    )
    lifecycle_path.write_bytes(route._canonical(lifecycle) + b"\n")  # noqa: SLF001

    with pytest.raises(
        execution_packet.ExecutionPacketError,
        match="base_clone_serving_evidence_binding_invalid",
    ):
        execution_packet._clone_serving_evidence(  # noqa: SLF001
            protocol=json.loads(EXTERNAL_CTF_PROTOCOL.read_text()),
            clone_intent_path=EXTERNAL_CTF_CLONE_INTENT,
            clone_plan_path=fixture["plan_path"],
            clone_result_path=fixture["result_path"],
            clone_resume_preflight_path=fixture["preflight_path"],
            clone_lifecycle_result_path=lifecycle_path,
        )
