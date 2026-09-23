"""Capture and validate prompt-free serving parity for external CTF evaluation."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import urllib.error
import urllib.request
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evals.webexploitbench.tensorlake import collection_provenance
from training import checkpoint_serving_parity, checkpoint_serving_route

from .protocol import FLEET_TEAM_ID, canonical, load_protocol

SCHEMA = "external_ctf_serving_live_parity_v1"
START_PREFLIGHT_SCHEMA = "external_ctf_scored_start_route_preflight_v1"
MAX_AGE_SECONDS = 1800
START_PREFLIGHT_MAX_AGE_SECONDS = 300
MAX_FUTURE_SKEW_SECONDS = 60
REQUIRED_PRIORITY_CLASS = "c1"
MODEL_LABEL = "inference.fleet.ai/model"
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
CATALOG_FIELDS = {
    "id",
    "model_revision",
    "status",
    "routed",
    "ready_replicas",
    "engine",
    "precision",
    "tensor_parallel_size",
    "data_parallel_size",
    "data_parallel_attention",
    "capabilities",
}


class LiveParityError(RuntimeError):
    """A live serving observation is stale, incomplete, or scientifically unmatched."""


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(canonical(value)).hexdigest()


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise LiveParityError(f"{label}_invalid")
    return dict(value)


def _catalog_scientific_projection(value: object, label: str) -> dict[str, Any]:
    catalog = _mapping(value, label)
    capabilities = catalog.get("capabilities")
    if (
        set(catalog) != CATALOG_FIELDS
        or catalog.get("status") != "ready"
        or catalog.get("routed") is not True
        or type(catalog.get("ready_replicas")) is not int
        or catalog["ready_replicas"] < 1
        or not isinstance(catalog.get("engine"), str)
        or not catalog["engine"]
        or not isinstance(catalog.get("precision"), str)
        or not catalog["precision"]
        or type(catalog.get("tensor_parallel_size")) is not int
        or catalog["tensor_parallel_size"] < 1
        or type(catalog.get("data_parallel_size")) is not int
        or catalog["data_parallel_size"] < 1
        or type(catalog.get("data_parallel_attention")) is not bool
        or not isinstance(capabilities, list)
        or not capabilities
        or any(not isinstance(item, str) or not item for item in capabilities)
        or len(capabilities) != len(set(capabilities))
    ):
        raise LiveParityError(label + "_scientific_fields_invalid")
    return {
        "status": catalog["status"],
        "routed": catalog["routed"],
        "ready_replicas": catalog["ready_replicas"],
        "engine": catalog["engine"],
        "precision": catalog["precision"],
        "tensor_parallel_size": catalog["tensor_parallel_size"],
        "data_parallel_size": catalog["data_parallel_size"],
        "data_parallel_attention": catalog["data_parallel_attention"],
        "capabilities": sorted(capabilities),
    }


def _normalized_catalog(value: object, label: str) -> dict[str, Any]:
    catalog = _mapping(value, label)
    if (
        not isinstance(catalog.get("id"), str)
        or not catalog["id"]
        or not isinstance(catalog.get("model_revision"), str)
        or not catalog["model_revision"]
    ):
        raise LiveParityError(label + "_identity_invalid")
    return {
        "id": catalog["id"],
        "model_revision": catalog["model_revision"],
        **_catalog_scientific_projection(catalog, label),
    }


def _matched_catalog_projection(arms: Mapping[str, Any]) -> dict[str, Any]:
    base = _mapping(arms.get("base"), "base_standard_arm")
    candidate = _mapping(arms.get("candidate"), "candidate_standard_arm")
    base_projection = _catalog_scientific_projection(base.get("catalog"), "base_standard_catalog")
    candidate_projection = _catalog_scientific_projection(
        candidate.get("catalog"), "candidate_standard_catalog"
    )
    if base_projection != candidate_projection:
        raise LiveParityError("standard_live_parity_catalog_drifted")
    return base_projection


def _metadata(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    metadata = _mapping(value.get("metadata"), label + "_metadata")
    if (
        not isinstance(metadata.get("uid"), str)
        or not metadata["uid"]
        or not isinstance(metadata.get("resourceVersion"), str)
        or not metadata["resourceVersion"]
    ):
        raise LiveParityError(label + "_metadata_invalid")
    return metadata


def _ready_pod(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    metadata = _metadata(value, label)
    spec = _mapping(value.get("spec"), label + "_spec")
    status = _mapping(value.get("status"), label + "_status")
    if spec.get("priorityClassName") != REQUIRED_PRIORITY_CLASS or status.get("phase") != "Running":
        raise LiveParityError(label + "_not_ready_or_priority_drifted")
    conditions = status.get("conditions")
    if not isinstance(conditions, list) or not any(
        isinstance(row, Mapping) and row.get("type") == "Ready" and row.get("status") == "True"
        for row in conditions
    ):
        raise LiveParityError(label + "_not_ready")
    statuses = status.get("containerStatuses")
    if not isinstance(statuses, list) or not statuses:
        raise LiveParityError(label + "_container_status_missing")
    containers = []
    for row in statuses:
        current = _mapping(row, label + "_container_status")
        if (
            not isinstance(current.get("name"), str)
            or not isinstance(current.get("image"), str)
            or not isinstance(current.get("imageID"), str)
            or "sha256:" not in current["imageID"]
            or current.get("ready") is not True
            or current.get("restartCount") != 0
        ):
            raise LiveParityError(label + "_container_not_clean")
        containers.append(
            {
                "name": current["name"],
                "image": current["image"],
                "image_id": current["imageID"],
            }
        )
    scientific_spec = copy.deepcopy(spec)
    scientific_spec.pop("nodeName", None)
    return {
        "uid": metadata["uid"],
        "resource_version": metadata["resourceVersion"],
        "spec_sha256": _digest(spec),
        "scientific_spec_sha256": _digest(scientific_spec),
        "priority_class": spec["priorityClassName"],
        "containers": sorted(containers, key=lambda row: row["name"]),
    }


def _route_snapshot(
    context: str,
    model: str,
    standard_arm: Mapping[str, Any],
) -> dict[str, Any]:
    inference = checkpoint_serving_parity._kubectl(  # noqa: SLF001
        context, "inferencemodel", model
    )
    metadata = _metadata(inference, "inference_model")
    spec = _mapping(inference.get("spec"), "inference_model_spec")
    placement = _mapping(spec.get("placement"), "inference_model_placement")
    status = _mapping(inference.get("status"), "inference_model_status")
    kubernetes = _mapping(standard_arm.get("kubernetes"), "standard_kubernetes")
    generation = metadata.get("generation")
    if (
        metadata["uid"] != kubernetes.get("inference_model_uid")
        or metadata["resourceVersion"] != kubernetes.get("inference_model_resource_version")
        or type(generation) is not int
        or generation < 1
        or placement.get("priorityClassName") != REQUIRED_PRIORITY_CLASS
        or status.get("phase") != "ready"
        or status.get("readyReplicas") != 1
    ):
        raise LiveParityError("inference_model_identity_or_priority_drifted")
    listing = checkpoint_serving_parity._kubectl(  # noqa: SLF001
        context, "pods", "-l", f"{MODEL_LABEL}={model}"
    )
    rows = listing.get("items") if isinstance(listing, Mapping) else None
    expected_uids = kubernetes.get("pod_uids")
    if not isinstance(rows, list) or not isinstance(expected_uids, list):
        raise LiveParityError("pod_inventory_invalid")
    active_rows = [
        row
        for row in rows
        if isinstance(row, Mapping)
        and isinstance(row.get("status"), Mapping)
        and row["status"].get("phase") not in {"Succeeded", "Failed"}
    ]
    if (
        len(active_rows) != 1
        or not isinstance(active_rows[0].get("metadata"), Mapping)
        or active_rows[0]["metadata"].get("uid") not in expected_uids
    ):
        raise LiveParityError("serving_pod_inventory_not_unique")
    pods = [_ready_pod(row, "serving_pod") for row in active_rows]
    if len(pods) != 1 or {pod["uid"] for pod in pods} != set(expected_uids):
        raise LiveParityError("serving_pod_identity_drifted")
    return {
        "served_model": model,
        "inference_model": {
            "uid": metadata["uid"],
            "generation": generation,
            "resource_version": metadata["resourceVersion"],
            "spec_sha256": _digest(spec),
            "priority_class": placement["priorityClassName"],
        },
        "pods": pods,
        "standard_arm_sha256": _digest(standard_arm),
    }


def _fleet_account(key: str) -> dict[str, str]:
    request = urllib.request.Request(
        "https://orchestrator.fleetai.com/v1/account",
        headers={"Authorization": "Bearer " + key, "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            value = json.load(response)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise LiveParityError("fleet_account_read_failed") from exc
    if (
        not isinstance(value, Mapping)
        or value.get("team_id") != FLEET_TEAM_ID
        or value.get("team_name") != "fleet"
    ):
        raise LiveParityError("fleet_account_identity_mismatch")
    return {"team_id": FLEET_TEAM_ID, "team_name": "fleet"}


def _refreshed_route(
    *,
    key: str,
    context: str,
    model: str,
    expected_revision: str,
    expected_source_path: str,
    frozen_route: Mapping[str, Any],
    frozen_standard_arm: Mapping[str, Any],
) -> dict[str, Any]:
    client = checkpoint_serving_route.Client(key)
    _status, api = client.request("GET", checkpoint_serving_route.API + "/" + model)
    api = _mapping(api, "route_api")
    status = _mapping(api.get("status"), "route_api_status")
    spec = _mapping(api.get("spec"), "route_api_spec")
    model_spec = _mapping(spec.get("model"), "route_api_model")
    revision = str(model_spec.get("revision"))
    resource_version = api.get("resource_version")
    if (
        status.get("phase") != "ready"
        or status.get("ready_replicas") != 1
        or revision != expected_revision
        or model_spec.get("sourcePath") != expected_source_path
        or not isinstance(resource_version, str)
        or not resource_version
    ):
        raise LiveParityError("route_api_identity_or_readiness_drifted")
    normalized_contract = checkpoint_serving_route._digest(  # noqa: SLF001
        checkpoint_serving_route._normalized_contract(spec)  # noqa: SLF001
    )
    if normalized_contract != frozen_standard_arm.get("normalized_contract_sha256"):
        raise LiveParityError("route_normalized_contract_drifted")
    catalog = checkpoint_serving_parity.catalog_projection(
        checkpoint_serving_parity._request(  # noqa: SLF001
            key, "/fleet/v1/model-catalog", model=model
        ),
        model,
        revision,
    )
    frozen_catalog = _mapping(frozen_standard_arm.get("catalog"), "frozen_standard_catalog")
    if _normalized_catalog(catalog, "refreshed_catalog") != _normalized_catalog(
        frozen_catalog, "frozen_standard_catalog"
    ):
        raise LiveParityError("route_catalog_drifted")
    catalog_scientific = _catalog_scientific_projection(catalog, "refreshed_catalog")

    inference = checkpoint_serving_parity._kubectl(  # noqa: SLF001
        context, "inferencemodel", model
    )
    metadata = _metadata(inference, "refreshed_inference_model")
    inference_spec = _mapping(inference.get("spec"), "refreshed_inference_model_spec")
    placement = _mapping(inference_spec.get("placement"), "refreshed_inference_placement")
    inference_status = _mapping(inference.get("status"), "refreshed_inference_model_status")
    frozen_inference = _mapping(frozen_route.get("inference_model"), "frozen_inference_model")
    if (
        metadata.get("uid") != frozen_inference.get("uid")
        or metadata.get("generation") != frozen_inference.get("generation")
        or _digest(inference_spec) != frozen_inference.get("spec_sha256")
        or placement.get("priorityClassName") != REQUIRED_PRIORITY_CLASS
        or inference_status.get("phase") != "ready"
        or inference_status.get("readyReplicas") != 1
    ):
        raise LiveParityError("refreshed_inference_model_drifted")
    listing = checkpoint_serving_parity._kubectl(  # noqa: SLF001
        context, "pods", "-l", f"{MODEL_LABEL}={model}"
    )
    rows = listing.get("items") if isinstance(listing, Mapping) else None
    active_rows = (
        []
        if not isinstance(rows, list)
        else [
            row
            for row in rows
            if isinstance(row, Mapping)
            and isinstance(row.get("status"), Mapping)
            and row["status"].get("phase") not in {"Succeeded", "Failed"}
        ]
    )
    if len(active_rows) != 1:
        raise LiveParityError("refreshed_serving_pod_inventory_not_unique")
    pod = _ready_pod(active_rows[0], "refreshed_serving_pod")
    frozen_pods = frozen_route.get("pods")
    if (
        not isinstance(frozen_pods, list)
        or len(frozen_pods) != 1
        or pod.get("scientific_spec_sha256") != frozen_pods[0].get("scientific_spec_sha256")
        or pod.get("priority_class") != frozen_pods[0].get("priority_class")
        or pod.get("containers") != frozen_pods[0].get("containers")
    ):
        raise LiveParityError("refreshed_serving_pod_drifted")
    return {
        "served_model": model,
        "model_revision": revision,
        "source_path": model_spec["sourcePath"],
        "api_resource_version": resource_version,
        "normalized_contract_sha256": normalized_contract,
        "catalog_sha256": _digest(_normalized_catalog(catalog, "refreshed_catalog")),
        "catalog_scientific": catalog_scientific,
        "catalog_scientific_sha256": _digest(catalog_scientific),
        "inference_model": {
            "uid": metadata["uid"],
            "generation": metadata["generation"],
            "resource_version": metadata["resourceVersion"],
            "spec_sha256": _digest(inference_spec),
            "priority_class": placement["priorityClassName"],
        },
        "pod": pod,
    }


def capture_start_preflight(
    *,
    protocol: dict[str, Any],
    execution_packet: Mapping[str, Any],
    benchmark: str,
    task_index: int,
    arm: str,
    key: str,
    context: str,
) -> dict[str, Any]:
    if (
        benchmark != protocol.get("execution_benchmark")
        or arm not in {"base", "step_1000"}
        or type(task_index) is not int
        or not 0 <= task_index < len(protocol["benchmarks"][benchmark]["task_ids"])
    ):
        raise LiveParityError("start_preflight_cell_invalid")
    packet_receipt = execution_packet.get("receipt_sha256")
    frozen = execution_packet.get("live_parity", {}).get("receipt")
    if not isinstance(packet_receipt, str) or _DIGEST.fullmatch(packet_receipt) is None:
        raise LiveParityError("start_preflight_packet_invalid")
    if not isinstance(frozen, dict):
        raise LiveParityError("start_preflight_frozen_parity_invalid")
    validate(frozen, protocol=protocol, now=_timestamp(frozen.get("observed_at")))
    routes = {
        name: _refreshed_route(
            key=key,
            context=context,
            model=protocol["arms"][name]["served_model"],
            expected_revision=protocol["arms"][name]["model_revision"],
            expected_source_path=protocol["arms"][name]["source_path"],
            frozen_route=frozen["routes"][name],
            frozen_standard_arm=frozen["standard_live_parity"]["arms"][
                "base" if name == "base" else "candidate"
            ],
        )
        for name in ("base", "step_1000")
    }
    if (
        routes["base"]["normalized_contract_sha256"]
        != routes["step_1000"]["normalized_contract_sha256"]
        or routes["base"]["catalog_scientific"] != routes["step_1000"]["catalog_scientific"]
        or routes["base"]["pod"]["containers"] != routes["step_1000"]["pod"]["containers"]
    ):
        raise LiveParityError("refreshed_routes_not_matched")
    value = {
        "schema": START_PREFLIGHT_SCHEMA,
        "status": "passed",
        "observed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "protocol_sha256": protocol["protocol_sha256"],
        "execution_packet_receipt_sha256": packet_receipt,
        "frozen_live_parity_receipt_sha256": frozen["receipt_sha256"],
        "cell": {
            "benchmark": benchmark,
            "task_index": task_index,
            "task_id": protocol["benchmarks"][benchmark]["task_ids"][task_index],
            "arm": arm,
        },
        "fleet_account": _fleet_account(key),
        "routes": routes,
        "benchmark_content_included": False,
        "response_content_recorded": False,
        "scores_observed": False,
        "external_mutations_performed": 0,
    }
    value["receipt_sha256"] = _digest(value)
    return value


def validate_start_preflight(
    value: dict[str, Any],
    *,
    protocol: dict[str, Any],
    execution_packet_receipt_sha256: str,
    benchmark: str,
    task_index: int,
    arm: str,
    frozen_live_parity: Mapping[str, Any],
    now: datetime | None = None,
) -> dict[str, Any]:
    expected_cell = {
        "benchmark": benchmark,
        "task_index": task_index,
        "task_id": protocol["benchmarks"][benchmark]["task_ids"][task_index],
        "arm": arm,
    }
    current = datetime.now(UTC) if now is None else now.astimezone(UTC)
    age = (current - _timestamp(value.get("observed_at"))).total_seconds()
    frozen_receipt = frozen_live_parity.get("receipt_sha256")
    routes = value.get("routes")
    if (
        value.get("schema") != START_PREFLIGHT_SCHEMA
        or value.get("status") != "passed"
        or value.get("protocol_sha256") != protocol["protocol_sha256"]
        or value.get("execution_packet_receipt_sha256") != execution_packet_receipt_sha256
        or value.get("frozen_live_parity_receipt_sha256") != frozen_receipt
        or value.get("cell") != expected_cell
        or value.get("fleet_account") != {"team_id": FLEET_TEAM_ID, "team_name": "fleet"}
        or not isinstance(routes, Mapping)
        or set(routes) != {"base", "step_1000"}
        or value.get("benchmark_content_included") is not False
        or value.get("response_content_recorded") is not False
        or value.get("scores_observed") is not False
        or value.get("external_mutations_performed") != 0
        or value.get("receipt_sha256")
        != _digest({key: item for key, item in value.items() if key != "receipt_sha256"})
        or age < -MAX_FUTURE_SKEW_SECONDS
        or age > START_PREFLIGHT_MAX_AGE_SECONDS
    ):
        raise LiveParityError("start_route_preflight_invalid_or_stale")
    frozen_routes = frozen_live_parity.get("routes")
    frozen_standard = frozen_live_parity.get("standard_live_parity", {}).get("arms")
    if not isinstance(frozen_routes, Mapping) or not isinstance(frozen_standard, Mapping):
        raise LiveParityError("start_route_preflight_frozen_parity_invalid")
    for name, standard_name in (("base", "base"), ("step_1000", "candidate")):
        route = _mapping(routes[name], name + "_start_route")
        inference = _mapping(route.get("inference_model"), name + "_start_inference")
        pod = _mapping(route.get("pod"), name + "_start_pod")
        frozen_route = _mapping(frozen_routes.get(name), name + "_frozen_route")
        frozen_inference = _mapping(frozen_route.get("inference_model"), name + "_frozen_inference")
        frozen_pods = frozen_route.get("pods")
        standard_arm = _mapping(frozen_standard.get(standard_name), name + "_standard_arm")
        frozen_catalog = _mapping(standard_arm.get("catalog"), name + "_frozen_standard_catalog")
        expected_catalog_scientific = _catalog_scientific_projection(
            frozen_catalog, name + "_frozen_standard_catalog"
        )
        if (
            route.get("served_model") != protocol["arms"][name]["served_model"]
            or route.get("model_revision") != protocol["arms"][name]["model_revision"]
            or route.get("source_path") != protocol["arms"][name]["source_path"]
            or route.get("normalized_contract_sha256")
            != standard_arm.get("normalized_contract_sha256")
            or not isinstance(route.get("api_resource_version"), str)
            or not route["api_resource_version"]
            or route.get("catalog_sha256")
            != _digest(_normalized_catalog(frozen_catalog, name + "_frozen_standard_catalog"))
            or route.get("catalog_scientific") != expected_catalog_scientific
            or route.get("catalog_scientific_sha256") != _digest(expected_catalog_scientific)
            or inference.get("uid") != frozen_inference.get("uid")
            or inference.get("generation") != frozen_inference.get("generation")
            or inference.get("spec_sha256") != frozen_inference.get("spec_sha256")
            or inference.get("priority_class") != REQUIRED_PRIORITY_CLASS
            or not isinstance(inference.get("resource_version"), str)
            or not inference["resource_version"]
            or not isinstance(frozen_pods, list)
            or len(frozen_pods) != 1
            or pod.get("scientific_spec_sha256") != frozen_pods[0].get("scientific_spec_sha256")
            or pod.get("priority_class") != REQUIRED_PRIORITY_CLASS
            or pod.get("containers") != frozen_pods[0].get("containers")
            or not isinstance(pod.get("resource_version"), str)
            or not pod["resource_version"]
        ):
            raise LiveParityError("start_route_preflight_route_binding_invalid")
    if (
        routes["base"]["normalized_contract_sha256"]
        != routes["step_1000"]["normalized_contract_sha256"]
        or routes["base"]["catalog_scientific"] != routes["step_1000"]["catalog_scientific"]
        or routes["base"]["pod"]["containers"] != routes["step_1000"]["pod"]["containers"]
    ):
        raise LiveParityError("start_route_preflight_routes_not_matched")
    return value


def load_start_preflight(
    path: Path,
    **bindings: Any,
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise LiveParityError("start_route_preflight_file_invalid")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LiveParityError("start_route_preflight_file_invalid") from exc
    if not isinstance(value, dict) or raw != canonical(value) + b"\n":
        raise LiveParityError("start_route_preflight_file_not_canonical")
    return validate_start_preflight(value, **bindings)


def _timestamp(value: object) -> datetime:
    if not isinstance(value, str):
        raise LiveParityError("live_parity_timestamp_invalid")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise LiveParityError("live_parity_timestamp_invalid") from exc
    if result.tzinfo is None or result.utcoffset() is None:
        raise LiveParityError("live_parity_timestamp_invalid")
    return result.astimezone(UTC)


def capture(
    *,
    protocol: dict[str, Any],
    key: str,
    context: str,
) -> dict[str, Any]:
    standard = checkpoint_serving_parity.capture(
        key=key,
        context=context,
        base_model=protocol["arms"]["base"]["served_model"],
        candidate_model=protocol["arms"]["step_1000"]["served_model"],
        expected_revision=protocol["arms"]["step_1000"]["model_revision"],
        expected_source_path=protocol["arms"]["step_1000"]["source_path"],
    )
    routes = {
        "base": _route_snapshot(
            context,
            protocol["arms"]["base"]["served_model"],
            standard["arms"]["base"],
        ),
        "step_1000": _route_snapshot(
            context,
            protocol["arms"]["step_1000"]["served_model"],
            standard["arms"]["candidate"],
        ),
    }
    _matched_catalog_projection(standard["arms"])
    if routes["base"]["pods"][0]["containers"] != routes["step_1000"]["pods"][0]["containers"]:
        raise LiveParityError("serving_container_image_ids_differ")
    value = {
        "schema": SCHEMA,
        "status": "passed",
        "protocol_sha256": protocol["protocol_sha256"],
        "observed_at": standard["observed_at"],
        "required_priority_class": REQUIRED_PRIORITY_CLASS,
        "standard_live_parity": standard,
        "fleet_account": _fleet_account(key),
        "routes": routes,
        "benchmark_content_included": False,
        "response_content_recorded": False,
        "scores_observed": False,
        "external_mutations_performed": 0,
    }
    value["receipt_sha256"] = _digest(value)
    return value


def validate(
    value: dict[str, Any],
    *,
    protocol: dict[str, Any],
    now: datetime | None = None,
) -> dict[str, Any]:
    expected_fields = {
        "schema",
        "status",
        "protocol_sha256",
        "observed_at",
        "required_priority_class",
        "standard_live_parity",
        "fleet_account",
        "routes",
        "benchmark_content_included",
        "response_content_recorded",
        "scores_observed",
        "external_mutations_performed",
        "receipt_sha256",
    }
    if (
        set(value) != expected_fields
        or value.get("schema") != SCHEMA
        or value.get("status") != "passed"
        or value.get("protocol_sha256") != protocol["protocol_sha256"]
        or value.get("required_priority_class") != REQUIRED_PRIORITY_CLASS
        or value.get("fleet_account") != {"team_id": FLEET_TEAM_ID, "team_name": "fleet"}
        or value.get("receipt_sha256")
        != _digest({key: item for key, item in value.items() if key != "receipt_sha256"})
        or value.get("benchmark_content_included") is not False
        or value.get("response_content_recorded") is not False
        or value.get("scores_observed") is not False
        or value.get("external_mutations_performed") != 0
    ):
        raise LiveParityError("live_parity_receipt_invalid")
    current = datetime.now(UTC) if now is None else now.astimezone(UTC)
    observed = _timestamp(value.get("observed_at"))
    age = (current - observed).total_seconds()
    if age > MAX_AGE_SECONDS or age < -MAX_FUTURE_SKEW_SECONDS:
        raise LiveParityError("live_parity_receipt_stale")
    standard = _mapping(value.get("standard_live_parity"), "standard_live_parity")
    if standard.get("receipt_sha256") != _digest(
        {key: item for key, item in standard.items() if key != "receipt_sha256"}
    ):
        raise LiveParityError("standard_live_parity_digest_invalid")
    try:
        base = collection_provenance._current_live_arm(  # noqa: SLF001
            standard.get("arms", {}).get("base"), "external CTF live base"
        )
        candidate = collection_provenance._current_live_arm(  # noqa: SLF001
            standard.get("arms", {}).get("candidate"), "external CTF live candidate"
        )
    except collection_provenance.ProvenanceError as exc:
        raise LiveParityError("standard_live_parity_invalid") from exc
    _matched_catalog_projection(standard["arms"])
    if (
        standard.get("schema") != "cyber_checkpoint_serving_live_parity_v2"
        or standard.get("status") != "passed"
        or standard.get("observed_at") != value["observed_at"]
        or base["served_model"] != protocol["arms"]["base"]["served_model"]
        or base["model_revision"] != protocol["arms"]["base"]["model_revision"]
        or base["source_path"] != protocol["arms"]["base"]["source_path"]
        or candidate["served_model"] != protocol["arms"]["step_1000"]["served_model"]
        or candidate["model_revision"] != protocol["arms"]["step_1000"]["model_revision"]
        or candidate["source_path"] != protocol["arms"]["step_1000"]["source_path"]
        or base["execution_contract_sha256"] != candidate["execution_contract_sha256"]
        or base["model_info"] != candidate["model_info"]
        or base["server_info"] != candidate["server_info"]
        or {key: base["probe"][key] for key in ("tool_name", "tool_argument_keys")}
        != {key: candidate["probe"][key] for key in ("tool_name", "tool_argument_keys")}
    ):
        raise LiveParityError("standard_live_parity_model_or_runtime_drifted")
    routes = _mapping(value.get("routes"), "routes")
    if set(routes) != {"base", "step_1000"}:
        raise LiveParityError("route_snapshots_invalid")
    for arm_name, standard_name, identity in (
        ("base", "base", base),
        ("step_1000", "candidate", candidate),
    ):
        route = _mapping(routes[arm_name], arm_name + "_route")
        inference = _mapping(route.get("inference_model"), arm_name + "_inference")
        pods = route.get("pods")
        if (
            route.get("served_model") != identity["served_model"]
            or route.get("standard_arm_sha256") != _digest(standard["arms"][standard_name])
            or inference.get("uid") != identity["inference_model_uid"]
            or inference.get("resource_version")
            != standard["arms"][standard_name]["kubernetes_resource_version"]
            or inference.get("priority_class") != REQUIRED_PRIORITY_CLASS
            or type(inference.get("generation")) is not int
            or inference["generation"] < 1
            or _DIGEST.fullmatch(str(inference.get("spec_sha256"))) is None
            or not isinstance(pods, list)
            or len(pods) != 1
            or pods[0].get("uid") != identity["pod_uids"][0]
            or pods[0].get("priority_class") != REQUIRED_PRIORITY_CLASS
            or _DIGEST.fullmatch(str(pods[0].get("spec_sha256"))) is None
            or _DIGEST.fullmatch(str(pods[0].get("scientific_spec_sha256"))) is None
            or not isinstance(pods[0].get("resource_version"), str)
            or not pods[0]["resource_version"]
        ):
            raise LiveParityError("route_snapshot_binding_invalid")
    if routes["base"]["pods"][0].get("containers") != routes["step_1000"]["pods"][0].get(
        "containers"
    ):
        raise LiveParityError("serving_container_image_ids_differ")
    return value


def load(path: Path, *, protocol: dict[str, Any], now: datetime | None = None) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise LiveParityError("live_parity_file_invalid")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LiveParityError("live_parity_file_invalid") from exc
    if not isinstance(value, dict) or raw != canonical(value) + b"\n":
        raise LiveParityError("live_parity_file_not_canonical")
    return validate(value, protocol=protocol, now=now)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--kubernetes-context", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = load_protocol(args.protocol)
    key = os.environ.pop("FLEET_API_KEY", "")
    if not key or key.strip() != key or any(character.isspace() for character in key):
        raise LiveParityError("fleet_api_key_missing_or_invalid")
    value = capture(
        protocol=protocol,
        key=key,
        context=args.kubernetes_context,
    )
    key = ""
    if args.output.exists():
        raise LiveParityError("live_parity_output_exists")
    args.output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    args.output.write_bytes(canonical(value) + b"\n")
    os.chmod(args.output, 0o600)
    print(json.dumps({"receipt_sha256": value["receipt_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
