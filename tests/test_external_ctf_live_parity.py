import copy
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from evals.external_ctf import live_parity

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/evaluation/qwen38-external-ctf-paired-v1.json"
PACKET_RECEIPT = "sha256:" + "8" * 64
CONTRACT = "sha256:" + "1" * 64
SPEC = "sha256:" + "2" * 64
POD_SPEC = "sha256:" + "3" * 64
IMAGE_ID = "docker-pullable://runtime@sha256:" + "4" * 64


def _protocol() -> dict:
    return json.loads(PROTOCOL.read_text())


def _containers() -> list[dict[str, str]]:
    return [
        {
            "name": "server",
            "image": "registry.invalid/runtime:fixed",
            "image_id": IMAGE_ID,
        }
    ]


def _catalog(protocol: dict, arm: str) -> dict:
    return {
        "id": protocol["arms"][arm]["served_model"],
        "model_revision": protocol["arms"][arm]["model_revision"],
        "status": "ready",
        "routed": True,
        "ready_replicas": 1,
        "engine": "sglang",
        "precision": "bf16",
        "tensor_parallel_size": 1,
        "data_parallel_size": 8,
        "data_parallel_attention": False,
        "capabilities": ["chat_completions", "reasoning", "tool_calling"],
    }


def _frozen(protocol: dict) -> dict:
    routes = {}
    for index, arm in enumerate(("base", "step_1000"), start=1):
        routes[arm] = {
            "served_model": protocol["arms"][arm]["served_model"],
            "inference_model": {
                "uid": f"inference-{index}",
                "generation": index,
                "resource_version": f"rv-{index}",
                "spec_sha256": SPEC,
                "priority_class": "c1",
            },
            "pods": [
                {
                    "uid": f"pod-{index}",
                    "resource_version": f"pod-rv-{index}",
                    "spec_sha256": POD_SPEC,
                    "scientific_spec_sha256": POD_SPEC,
                    "priority_class": "c1",
                    "containers": _containers(),
                }
            ],
        }
    return {
        "receipt_sha256": "sha256:" + "5" * 64,
        "routes": routes,
        "standard_live_parity": {
            "arms": {
                "base": {
                    "normalized_contract_sha256": CONTRACT,
                    "catalog": _catalog(protocol, "base"),
                },
                "candidate": {
                    "normalized_contract_sha256": CONTRACT,
                    "catalog": _catalog(protocol, "step_1000"),
                },
            }
        },
    }


def _start_receipt(protocol: dict, frozen: dict) -> dict:
    routes = {}
    for arm in ("base", "step_1000"):
        snapshot = frozen["routes"][arm]
        standard_name = "base" if arm == "base" else "candidate"
        catalog = frozen["standard_live_parity"]["arms"][standard_name]["catalog"]
        catalog_scientific = live_parity._catalog_scientific_projection(  # noqa: SLF001
            catalog, "synthetic_catalog"
        )
        routes[arm] = {
            "served_model": protocol["arms"][arm]["served_model"],
            "model_revision": protocol["arms"][arm]["model_revision"],
            "source_path": protocol["arms"][arm]["source_path"],
            "api_resource_version": "api-rv-1",
            "normalized_contract_sha256": CONTRACT,
            "catalog_sha256": live_parity._digest(  # noqa: SLF001
                live_parity._normalized_catalog(catalog, "synthetic_catalog")  # noqa: SLF001
            ),
            "catalog_scientific": catalog_scientific,
            "catalog_scientific_sha256": live_parity._digest(  # noqa: SLF001
                catalog_scientific
            ),
            "inference_model": {
                **snapshot["inference_model"],
                "resource_version": "fresh-inference-rv",
            },
            "pod": {
                **snapshot["pods"][0],
                "uid": "clean-replacement-pod-is-allowed",
                "resource_version": "fresh-pod-rv",
            },
        }
    value = {
        "schema": live_parity.START_PREFLIGHT_SCHEMA,
        "status": "passed",
        "observed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "protocol_sha256": protocol["protocol_sha256"],
        "execution_packet_receipt_sha256": PACKET_RECEIPT,
        "frozen_live_parity_receipt_sha256": frozen["receipt_sha256"],
        "cell": {
            "benchmark": "cvebench_zero_day",
            "task_index": 5,
            "task_id": protocol["benchmarks"]["cvebench_zero_day"]["task_ids"][5],
            "arm": "step_1000",
        },
        "fleet_account": {
            "team_id": "a1025f0b-ad67-49fc-a023-51800ab43e84",
            "team_name": "fleet",
        },
        "routes": routes,
        "benchmark_content_included": False,
        "response_content_recorded": False,
        "scores_observed": False,
        "external_mutations_performed": 0,
    }
    value["receipt_sha256"] = live_parity._digest(value)  # noqa: SLF001
    return value


def _validate(value: dict, protocol: dict, frozen: dict) -> dict:
    return live_parity.validate_start_preflight(
        value,
        protocol=protocol,
        execution_packet_receipt_sha256=PACKET_RECEIPT,
        benchmark="cvebench_zero_day",
        task_index=5,
        arm="step_1000",
        frozen_live_parity=frozen,
    )


def _resign(value: dict) -> None:
    value["receipt_sha256"] = live_parity._digest(  # noqa: SLF001
        {key: item for key, item in value.items() if key != "receipt_sha256"}
    )


def test_start_preflight_accepts_clean_pod_replacement_but_exact_route_contract() -> None:
    protocol = _protocol()
    frozen = _frozen(protocol)
    value = _start_receipt(protocol, frozen)

    assert _validate(value, protocol, frozen) == value


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (
            lambda value: value.update(execution_packet_receipt_sha256="sha256:" + "0" * 64),
            "start_route_preflight_invalid_or_stale",
        ),
        (lambda value: value["cell"].update(arm="base"), "start_route_preflight_invalid_or_stale"),
        (
            lambda value: value["routes"]["base"].update(source_path="wrong/source"),
            "start_route_preflight_route_binding_invalid",
        ),
        (
            lambda value: value["routes"]["base"]["inference_model"].update(uid="wrong"),
            "start_route_preflight_route_binding_invalid",
        ),
        (
            lambda value: value["routes"]["base"]["inference_model"].update(
                spec_sha256="sha256:" + "0" * 64
            ),
            "start_route_preflight_route_binding_invalid",
        ),
        (
            lambda value: value["routes"]["base"]["inference_model"].update(priority_class="c0"),
            "start_route_preflight_route_binding_invalid",
        ),
        (
            lambda value: value["routes"]["base"]["catalog_scientific"].update(engine="drifted"),
            "start_route_preflight_route_binding_invalid",
        ),
        (
            lambda value: value["routes"]["base"]["pod"].update(
                containers=[
                    {
                        "name": "server",
                        "image": "registry.invalid/runtime:fixed",
                        "image_id": "docker-pullable://runtime@sha256:" + "0" * 64,
                    }
                ]
            ),
            "start_route_preflight_route_binding_invalid",
        ),
    ],
)
def test_start_preflight_rejects_binding_drift(mutation, error: str) -> None:
    protocol = _protocol()
    frozen = _frozen(protocol)
    value = _start_receipt(protocol, frozen)
    mutation(value)
    _resign(value)

    with pytest.raises(live_parity.LiveParityError, match=error):
        _validate(value, protocol, frozen)


def test_start_preflight_rejects_stale_receipt() -> None:
    protocol = _protocol()
    frozen = _frozen(protocol)
    value = _start_receipt(protocol, frozen)
    value["observed_at"] = (
        (datetime.now(UTC) - timedelta(seconds=live_parity.START_PREFLIGHT_MAX_AGE_SECONDS + 1))
        .isoformat()
        .replace("+00:00", "Z")
    )
    _resign(value)

    with pytest.raises(
        live_parity.LiveParityError,
        match="start_route_preflight_invalid_or_stale",
    ):
        _validate(value, protocol, frozen)


def test_frozen_standard_catalog_rejects_cross_arm_scientific_drift() -> None:
    protocol = _protocol()
    arms = _frozen(protocol)["standard_live_parity"]["arms"]
    arms["candidate"]["catalog"]["engine"] = "drifted"

    with pytest.raises(
        live_parity.LiveParityError,
        match="standard_live_parity_catalog_drifted",
    ):
        live_parity._matched_catalog_projection(arms)  # noqa: SLF001


@pytest.mark.parametrize(
    ("priority", "restart_count"),
    [("c0", 0), ("c1", 1)],
)
def test_ready_pod_rejects_priority_or_restart_drift(priority: str, restart_count: int) -> None:
    pod = {
        "metadata": {"uid": "pod", "resourceVersion": "rv"},
        "spec": {"priorityClassName": priority},
        "status": {
            "phase": "Running",
            "conditions": [{"type": "Ready", "status": "True"}],
            "containerStatuses": [
                {
                    "name": "server",
                    "image": "registry.invalid/runtime:fixed",
                    "imageID": IMAGE_ID,
                    "ready": True,
                    "restartCount": restart_count,
                }
            ],
        },
    }

    with pytest.raises(live_parity.LiveParityError):
        live_parity._ready_pod(pod, "synthetic")  # noqa: SLF001


def test_route_snapshot_rejects_an_extra_active_pod(monkeypatch) -> None:
    standard = {
        "kubernetes": {
            "inference_model_uid": "inference-uid",
            "inference_model_resource_version": "inference-rv",
            "pod_uids": ["pod-1"],
        }
    }
    inference = {
        "metadata": {
            "uid": "inference-uid",
            "resourceVersion": "inference-rv",
            "generation": 1,
        },
        "spec": {"placement": {"priorityClassName": "c1"}},
        "status": {"phase": "ready", "readyReplicas": 1},
    }
    pod = {
        "metadata": {"uid": "pod-1", "resourceVersion": "pod-rv"},
        "spec": {"priorityClassName": "c1"},
        "status": {
            "phase": "Running",
            "conditions": [{"type": "Ready", "status": "True"}],
            "containerStatuses": [
                {
                    "name": "server",
                    "image": "registry.invalid/runtime:fixed",
                    "imageID": IMAGE_ID,
                    "ready": True,
                    "restartCount": 0,
                }
            ],
        },
    }

    def kubectl(_context, kind, *_args):
        if kind == "inferencemodel":
            return inference
        return {
            "items": [pod, {**copy.deepcopy(pod), "metadata": {**pod["metadata"], "uid": "pod-2"}}]
        }

    monkeypatch.setattr(live_parity.checkpoint_serving_parity, "_kubectl", kubectl)

    with pytest.raises(
        live_parity.LiveParityError,
        match="serving_pod_inventory_not_unique",
    ):
        live_parity._route_snapshot("context", "model", standard)  # noqa: SLF001
