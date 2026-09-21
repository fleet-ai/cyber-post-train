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


def model(name: str, *, active: int, phase: str = "ready") -> dict:
    return {
        "metadata": {
            "name": name,
            "namespace": "inference",
            "uid": "model-" + name,
            "resourceVersion": "20",
        },
        "spec": {"desiredState": "serving" if active else "paused"},
        "status": {
            "phase": phase,
            "activePods": active,
            "readyReplicas": active,
            "observedGeneration": 1,
        },
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


def test_live_census_reads_pods_and_inference_models_across_all_namespaces(monkeypatch) -> None:
    calls: list[list[str]] = []
    responses = iter(({"items": []}, {"items": []}))

    def run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, json.dumps(next(responses)), "")

    monkeypatch.setattr(gpu_capacity.subprocess, "run", run)
    receipt = gpu_capacity.live_capacity_census(
        "prod-context",
        observed_at="2026-09-21T00:00:00Z",
    )
    assert receipt["qualified"] is True
    assert len(calls) == 2
    assert calls[0][-4:] == ["get", "pods", "--all-namespaces", "-o", "json"][-4:]
    assert calls[1][-4:] == [
        "inferencemodels.inference.fleet.ai",
        "--all-namespaces",
        "-o",
        "json",
    ]
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
