import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from cyber_post_train import gpu_capacity
from cyber_post_train.gpu_capacity import CapacityError, build_capacity_census

EVIDENCE = (
    Path(__file__).parents[1]
    / "docs/evidence/qwen38-project-gpu-capacity-reconciliation-20260921.json"
)


def pod(
    name: str,
    *,
    namespace: str,
    uid: str,
    node: str | None,
    labels: dict[str, str],
    gpus: int = 8,
    phase: str = "Running",
    init_containers: list[dict] | None = None,
) -> dict:
    return {
        "metadata": {
            "name": name,
            "namespace": namespace,
            "uid": uid,
            "resourceVersion": "10",
            "labels": labels,
        },
        "spec": {
            "nodeName": node,
            "initContainers": init_containers or [],
            "containers": [
                {
                    "resources": {
                        "requests": {"nvidia.com/gpu": str(gpus)},
                        "limits": {"nvidia.com/gpu": str(gpus)},
                    }
                }
            ],
        },
        "status": {"phase": phase, "containerStatuses": [{"restartCount": 0}]},
    }


def gpu_container(name: str, gpus: int, *, restartable: bool = False) -> dict:
    value = {
        "name": name,
        "resources": {
            "requests": {"nvidia.com/gpu": str(gpus)},
            "limits": {"nvidia.com/gpu": str(gpus)},
        },
    }
    if restartable:
        value["restartPolicy"] = "Always"
    return value


def model(
    name: str,
    *,
    active: int,
    phase: str = "ready",
    desired_state: str | None = None,
    generation: int = 1,
    observed_generation: int = 1,
    gpus: int = 8,
) -> dict:
    return {
        "metadata": {
            "name": name,
            "namespace": "inference",
            "uid": "model-" + name,
            "resourceVersion": "20",
            "generation": generation,
        },
        "spec": {
            "desiredState": desired_state or ("serving" if active else "paused"),
            "resources": {
                "requests": {"nvidia.com/gpu": gpus},
                "limits": {"nvidia.com/gpu": gpus},
            },
            "scaling": {"minReplicas": 0, "replicas": 1},
        },
        "status": {
            "phase": phase,
            "activePods": active,
            "readyReplicas": active,
            "observedGeneration": observed_generation,
        },
    }


def rayjob(
    name: str,
    *,
    run_name: str | None = None,
    uid: str = "rayjob-uid",
    gpus: int = 8,
) -> dict:
    return {
        "metadata": {
            "name": name,
            "namespace": "fleet-train-jobs",
            "uid": uid,
            "resourceVersion": "30",
            "labels": {"fleet.ai/run-name": run_name or name},
        },
        "spec": {
            "suspend": True,
            "rayClusterSpec": {
                "headGroupSpec": {
                    "template": {"spec": {"containers": [gpu_container("head", gpus)]}}
                },
                "workerGroupSpecs": [],
            },
        },
        "status": {},
    }


def workload(name: str, *, owner: str, owner_uid: str = "rayjob-uid", gpus: int = 8) -> dict:
    return {
        "metadata": {
            "name": name,
            "namespace": "fleet-train-jobs",
            "uid": "workload-uid",
            "resourceVersion": "40",
            "ownerReferences": [{"kind": "RayJob", "name": owner, "uid": owner_uid}],
        },
        "spec": {
            "podSets": [
                {
                    "count": 1,
                    "template": {"spec": {"containers": [gpu_container("head", gpus)]}},
                }
            ]
        },
        "status": {},
    }


def test_cross_namespace_census_counts_training_and_every_serving_route() -> None:
    pods = {
        "items": [
            pod(
                "chris-q38-train-head",
                namespace="fleet-train-jobs",
                uid="train",
                node="node-a",
                labels={"fleet.ai/run-name": "chris-q38-train"},
            ),
            pod(
                "inference-chris-q38-candidate-a",
                namespace="inference",
                uid="serve-a",
                node="node-b",
                labels={"inference.fleet.ai/model": "chris-q38-candidate-a"},
            ),
            # Regression for the omitted historical endpoint: it is not in the
            # training namespace or the current experiment list.
            pod(
                "inference-chris-q38-historical-step10",
                namespace="inference",
                uid="serve-old",
                node="node-c",
                labels={"inference.fleet.ai/model": "chris-q38-historical-step10"},
            ),
        ]
    }
    models = {
        "items": [
            model("chris-q38-candidate-a", active=1),
            model("chris-q38-historical-step10", active=1),
            model("chris-q38-paused", active=0, phase="paused"),
        ]
    }
    receipt = build_capacity_census(
        pods,
        models,
        observed_at="2026-09-21T00:00:00Z",
    )
    assert receipt["qualified"] is True
    assert receipt["current"]["nodes"] == 3
    assert receipt["current"]["gpus"] == 24
    assert receipt["current"]["role_pod_counts"] == {
        "training": 1,
        "serving": 2,
        "unclassified": 0,
    }
    assert [row["name"] for row in receipt["current"]["inference_models"]] == [
        "chris-q38-candidate-a",
        "chris-q38-historical-step10",
        "chris-q38-paused",
    ]
    assert len(receipt["sha256"]) == 64


def test_projected_create_fails_closed_above_eight_nodes() -> None:
    pods = {
        "items": [
            pod(
                f"chris-q38-train-{index}",
                namespace="fleet-train-jobs",
                uid=f"uid-{index}",
                node=f"node-{index}",
                labels={"fleet.ai/run-name": f"chris-q38-train-{index}"},
            )
            for index in range(8)
        ]
    }
    receipt = build_capacity_census(
        pods,
        {"items": []},
        planned_nodes=1,
        planned_gpus=8,
        observed_at="2026-09-21T00:00:00Z",
    )
    assert receipt["current"]["nodes"] == 8
    assert receipt["projected"] == {"nodes": 9, "gpus": 72}
    assert receipt["qualified"] is False
    assert "projected GPU nodes 9 exceed limit 8" in receipt["problems"]
    assert "projected GPUs 72 exceed limit 64" in receipt["problems"]


def test_pending_unscheduled_pod_occupies_capacity_before_a_new_create() -> None:
    pods = {
        "items": [
            *[
                pod(
                    f"chris-q38-running-{index}",
                    namespace="fleet-train-jobs",
                    uid=f"running-{index}",
                    node=f"node-{index}",
                    labels={"fleet.ai/run-name": f"chris-q38-running-{index}"},
                )
                for index in range(7)
            ],
            pod(
                "chris-q38-queued",
                namespace="fleet-train-jobs",
                uid="queued",
                node=None,
                labels={"fleet.ai/run-name": "chris-q38-queued"},
                phase="Pending",
            ),
        ]
    }
    receipt = build_capacity_census(
        pods,
        {"items": []},
        planned_nodes=1,
        planned_gpus=8,
        observed_at="2026-09-21T00:00:00Z",
    )
    assert receipt["current"]["nodes"] == 8
    assert receipt["current"]["gpus"] == 64
    assert receipt["projected"] == {"nodes": 9, "gpus": 72}
    assert receipt["qualified"] is False


def test_suspended_rayjob_workload_without_a_pod_occupies_capacity() -> None:
    name = "chris-q38-queued"
    receipt = build_capacity_census(
        {
            "items": [
                pod(
                    f"chris-q38-running-{index}",
                    namespace="fleet-train-jobs",
                    uid=f"running-{index}",
                    node=f"node-{index}",
                    labels={"fleet.ai/run-name": f"chris-q38-running-{index}"},
                )
                for index in range(7)
            ]
        },
        {"items": []},
        {"items": [rayjob(name)]},
        {"items": [workload("queued-workload", owner=name)]},
        planned_nodes=1,
        planned_gpus=8,
        observed_at="2026-09-21T00:00:00Z",
    )
    assert receipt["current"]["queued_claims"][0]["identity"] == name
    assert receipt["current"]["nodes"] == 8
    assert receipt["current"]["gpus"] == 64
    assert receipt["qualified"] is False


def test_api_suffixed_rayjobs_reconcile_to_six_live_run_name_pods() -> None:
    run_names = [f"chris-q38-t3k-{index}" for index in range(6)]
    controllers = [f"{run_name}-{index:08x}" for index, run_name in enumerate(run_names)]
    rayjob_uids = [f"rayjob-uid-{index}" for index in range(6)]
    receipt = build_capacity_census(
        {
            "items": [
                pod(
                    f"{run_name}-head",
                    namespace="fleet-train-jobs",
                    uid=f"pod-uid-{index}",
                    node=f"node-{index}",
                    labels={"fleet.ai/run-name": run_name},
                )
                for index, run_name in enumerate(run_names)
            ]
        },
        {"items": []},
        {
            "items": [
                rayjob(
                    controller,
                    run_name=run_name,
                    uid=rayjob_uid,
                )
                for controller, run_name, rayjob_uid in zip(
                    controllers, run_names, rayjob_uids, strict=True
                )
            ]
        },
        {
            "items": [
                workload(
                    f"workload-{index}",
                    owner=controller,
                    owner_uid=rayjob_uid,
                )
                for index, (controller, rayjob_uid) in enumerate(
                    zip(controllers, rayjob_uids, strict=True)
                )
            ]
        },
        observed_at="2026-09-22T08:23:52Z",
    )
    assert receipt["qualified"] is True
    assert receipt["problems"] == []
    assert receipt["current"]["nodes"] == 6
    assert receipt["current"]["gpus"] == 48
    assert receipt["current"]["queued_claims"] == []
    assert {row["uid"] for row in receipt["current"]["allocated_pods"]} == {
        f"pod-uid-{index}" for index in range(6)
    }


def test_uncorrelated_owned_workload_fails_closed() -> None:
    receipt = build_capacity_census(
        {"items": []},
        {"items": []},
        {"items": []},
        {"items": [workload("orphan", owner="chris-q38-missing")]},
        observed_at="2026-09-21T00:00:00Z",
    )
    assert receipt["qualified"] is False
    assert any("no exact RayJob owner" in problem for problem in receipt["problems"])


def test_regular_init_gpu_is_peak_not_added_to_app_gpu() -> None:
    receipt = build_capacity_census(
        {
            "items": [
                pod(
                    "chris-q38-train-head",
                    namespace="fleet-train-jobs",
                    uid="train",
                    node="node-a",
                    labels={"fleet.ai/run-name": "chris-q38-train"},
                    gpus=8,
                    init_containers=[
                        gpu_container("first-setup", 8),
                        gpu_container("second-setup", 8),
                    ],
                )
            ]
        },
        {"items": []},
        observed_at="2026-09-21T00:00:00Z",
    )
    assert receipt["qualified"] is True
    assert receipt["current"]["gpus"] == 8


def test_restartable_init_sidecar_is_added_to_app_and_later_init_peak() -> None:
    receipt = build_capacity_census(
        {
            "items": [
                pod(
                    "chris-q38-train-head",
                    namespace="fleet-train-jobs",
                    uid="train",
                    node="node-a",
                    labels={"fleet.ai/run-name": "chris-q38-train"},
                    gpus=7,
                    init_containers=[
                        gpu_container("gpu-sidecar", 1, restartable=True),
                        gpu_container("later-setup", 8),
                    ],
                )
            ]
        },
        {"items": []},
        max_gpus=9,
        observed_at="2026-09-21T00:00:00Z",
    )
    assert receipt["qualified"] is True
    assert receipt["current"]["gpus"] == 9


def test_unlabelled_owned_gpu_pod_is_counted_and_rejected() -> None:
    receipt = build_capacity_census(
        {
            "items": [
                pod(
                    "chris-q38-unknown",
                    namespace="other",
                    uid="unknown",
                    node="node-a",
                    labels={},
                )
            ]
        },
        {"items": []},
        observed_at="2026-09-21T00:00:00Z",
    )
    assert receipt["current"]["nodes"] == 1
    assert receipt["current"]["gpus"] == 8
    assert receipt["current"]["role_pod_counts"]["unclassified"] == 1
    assert receipt["qualified"] is False
    assert "no recognized training or serving ownership label" in receipt["problems"][0]


def test_inference_model_and_pod_count_must_reconcile() -> None:
    receipt = build_capacity_census(
        {"items": []},
        {"items": [model("chris-q38-serving", active=1)]},
        observed_at="2026-09-21T00:00:00Z",
    )
    assert receipt["qualified"] is False
    assert "activePods=1" in receipt["problems"][0]


@pytest.mark.parametrize("phase", ["ready", "queued", "resuming"])
def test_serving_or_starting_inference_model_without_a_pod_fails_closed(phase: str) -> None:
    receipt = build_capacity_census(
        {
            "items": [
                pod(
                    f"chris-q38-running-{index}",
                    namespace="fleet-train-jobs",
                    uid=f"running-{index}",
                    node=f"node-{index}",
                    labels={"fleet.ai/run-name": f"chris-q38-running-{index}"},
                )
                for index in range(7)
            ]
        },
        {
            "items": [
                model(
                    "chris-q38-resuming",
                    active=0,
                    phase=phase,
                    desired_state="serving",
                )
            ]
        },
        planned_nodes=1,
        planned_gpus=8,
        observed_at="2026-09-21T00:00:00Z",
    )
    assert receipt["current"]["gpus"] == 64
    assert receipt["current"]["nodes"] == 8
    assert receipt["projected"] == {"nodes": 9, "gpus": 72}
    assert receipt["current"]["serving_claims"] == [
        {
            "namespace": "inference",
            "name": "chris-q38-resuming",
            "uid": "model-chris-q38-resuming",
            "resource_version": "20",
            "identity": "chris-q38-resuming",
            "nodes": 1,
            "gpus": 8,
            "source": "inference_model",
        }
    ]
    assert receipt["qualified"] is False
    assert "projected GPUs 72 exceed limit 64" in receipt["problems"]


def test_stale_inference_model_generation_fails_closed_with_a_counted_pod() -> None:
    name = "chris-q38-serving"
    receipt = build_capacity_census(
        {
            "items": [
                pod(
                    "inference-" + name,
                    namespace="inference",
                    uid="serving",
                    node="node-a",
                    labels={"inference.fleet.ai/model": name},
                )
            ]
        },
        {
            "items": [
                model(
                    name,
                    active=1,
                    phase="ready",
                    generation=2,
                    observed_generation=1,
                )
            ]
        },
        observed_at="2026-09-21T00:00:00Z",
    )
    assert receipt["current"]["gpus"] == 8
    assert receipt["qualified"] is False
    assert any(
        "generation=2 but observedGeneration=1" in problem for problem in receipt["problems"]
    )


@pytest.mark.parametrize("phase", ["ready", "serving", "queued", "resuming"])
def test_stale_active_inference_route_without_pods_is_still_counted(phase: str) -> None:
    receipt = build_capacity_census(
        {"items": []},
        {
            "items": [
                model(
                    "chris-q38-stale-route",
                    active=0,
                    phase=phase,
                    desired_state="paused",
                    generation=2,
                    observed_generation=1,
                )
            ]
        },
        observed_at="2026-09-21T00:00:00Z",
    )
    assert receipt["current"]["nodes"] == 1
    assert receipt["current"]["gpus"] == 8
    assert receipt["current"]["serving_claims"][0]["name"] == "chris-q38-stale-route"
    assert receipt["qualified"] is False
    assert any(
        "generation=2 but observedGeneration=1" in problem for problem in receipt["problems"]
    )


def test_live_census_reads_pods_and_queued_claims_across_all_namespaces(monkeypatch) -> None:
    calls: list[list[str]] = []
    responses = iter(({"items": []}, {"items": []}, {"items": []}, {"items": []}))

    def run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, json.dumps(next(responses)), "")

    monkeypatch.setattr(gpu_capacity.subprocess, "run", run)
    receipt = gpu_capacity.live_capacity_census(
        "prod-context",
        observed_at="2026-09-21T00:00:00Z",
    )
    assert receipt["qualified"] is True
    assert len(calls) == 4
    assert calls[0][-4:] == ["get", "pods", "--all-namespaces", "-o", "json"][-4:]
    assert calls[1][-4:] == [
        "inferencemodels.inference.fleet.ai",
        "--all-namespaces",
        "-o",
        "json",
    ]
    assert "rayjobs.ray.io" in calls[2]
    assert "workloads.kueue.x-k8s.io" in calls[3]
    assert all("--context" in command and "prod-context" in command for command in calls)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"owner_prefixes": ()},
        {"max_nodes": 0},
        {"planned_nodes": -1},
        {"planned_nodes": 1, "planned_gpus": 0},
        {"planned_nodes": 2, "planned_gpus": 1},
    ],
)
def test_invalid_capacity_contract_is_rejected(kwargs: dict) -> None:
    with pytest.raises(CapacityError):
        build_capacity_census({"items": []}, {"items": []}, **kwargs)


def test_capacity_reconciliation_evidence_is_sanitized_and_self_digesting() -> None:
    value = json.loads(EVIDENCE.read_text())
    digest = value.pop("sha256")
    assert (
        digest
        == hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
    )
    assert value["private_content_persisted"] is False
    assert value["release"]["former_pod_uid_absent"] is True
    census = value["post_release_cross_namespace_census"]
    assert census["scope"]["kubernetes_namespaces"] == "all"
    assert census["current"]["role_pod_counts"]["unclassified"] == 0
    assert census["qualified"] is True
