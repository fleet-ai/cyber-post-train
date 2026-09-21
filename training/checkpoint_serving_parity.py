"""Capture sanitized live parity for a base and checkpoint serving route.

The command is read-only. It retains identities, execution-contract hashes,
Kubernetes UIDs and fixed-probe structure, never generated text or benchmark
content.
"""

from __future__ import annotations

import argparse
import copy
import http.client
import json
import math
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from training.checkpoint_serving_route import (
    API,
    Client,
    RouteError,
    _canonical,
    _digest,
    _mapping,
    _normalized_contract,
    _write_once,
)

ORIGIN = "https://inference.flt.build"
NAMESPACE = "inference"
MODEL_LABEL = "inference.fleet.ai/model"


def _need(condition: object, reason: str) -> None:
    if not condition:
        raise RouteError(reason)


def _request(
    key: str,
    path: str,
    *,
    model: str,
    payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _need(
        path in {"/fleet/v1/model-catalog", "/model_info", "/server_info", "/v1/chat/completions"},
        "unapproved_probe_path",
    )
    body = None if payload is None else _canonical(payload)
    headers = {
        "Authorization": "Bearer " + key,
        "Accept": "application/json",
        "X-Fleet-Model": model,
    }
    if body is not None:
        headers["Content-Type"] = "application/json"
    attempts = 3 if body is None else 1
    for attempt in range(1, attempts + 1):
        request = urllib.request.Request(
            ORIGIN + path,
            data=body,
            method="POST" if body is not None else "GET",
            headers=headers,
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                value = json.load(response)
            return _mapping(value, "invalid_probe_response")
        except urllib.error.HTTPError as exc:
            raise RouteError(f"probe_http_{exc.code}:{path}") from None
        except (
            urllib.error.URLError,
            TimeoutError,
            json.JSONDecodeError,
            http.client.IncompleteRead,
        ) as exc:
            if attempt == attempts:
                raise RouteError(f"probe_failed:{path}:{type(exc).__name__}") from None
            time.sleep(attempt)
    raise AssertionError("unreachable")


def _kubectl(context: str, *arguments: str) -> dict[str, Any]:
    command = [
        "kubectl",
        "--context",
        context,
        "-n",
        NAMESPACE,
        "get",
        *arguments,
        "-o",
        "json",
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, timeout=60)
        return _mapping(json.loads(completed.stdout), "invalid_kubectl_response")
    except (subprocess.SubprocessError, json.JSONDecodeError) as exc:
        raise RouteError("kubectl_read_failed") from exc


def _owner_uid(item: Mapping[str, Any], kind: str) -> str | None:
    owners = _mapping(item.get("metadata"), "invalid_metadata").get("ownerReferences") or []
    _need(isinstance(owners, list), "invalid_owner_references")
    matches = [
        owner.get("uid")
        for owner in owners
        if isinstance(owner, Mapping)
        and owner.get("kind") == kind
        and owner.get("controller") is True
    ]
    _need(len(matches) <= 1, "ambiguous_owner")
    return matches[0] if matches else None


def _metadata_uid(item: Mapping[str, Any], reason: str) -> str:
    uid = _mapping(item.get("metadata"), reason).get("uid")
    _need(isinstance(uid, str) and uid, reason)
    return uid


def _list(value: object, reason: str) -> list[dict[str, Any]]:
    listing = _mapping(value, reason)
    rows = listing.get("items")
    _need(isinstance(rows, list) and all(isinstance(row, Mapping) for row in rows), reason)
    return [copy.deepcopy(dict(row)) for row in rows]


def workload_projection(
    context: str, model: str, image_digest: str, *, expected_replicas: int | None
) -> dict[str, Any]:
    inference = _kubectl(context, "inferencemodel", model)
    inference_uid = _metadata_uid(inference, "invalid_inference_model_uid")
    inference_status = _mapping(inference.get("status"), "invalid_inference_status")
    ready_replicas = inference_status.get("readyReplicas")
    _need(
        inference_status.get("phase") == "ready"
        and type(ready_replicas) is int
        and ready_replicas >= 1,
        "inference_model_not_ready",
    )
    if expected_replicas is not None:
        _need(ready_replicas == expected_replicas, "unexpected_ready_replicas")
    deployments = _list(_kubectl(context, "deployments"), "invalid_deployments")
    owned_deployments = [
        row
        for row in deployments
        if _owner_uid(row, "InferenceModel") == inference_uid
        and _mapping(row.get("metadata"), "invalid_deployment_metadata")
        .get("labels", {})
        .get(MODEL_LABEL)
        == model
    ]
    _need(len(owned_deployments) == 1, "deployment_not_unique")
    deployment = owned_deployments[0]
    deployment_uid = _metadata_uid(deployment, "invalid_deployment_uid")
    deployment_status = _mapping(deployment.get("status"), "invalid_deployment_status")
    _need(
        deployment_status.get("replicas")
        == deployment_status.get("readyReplicas")
        == ready_replicas,
        "deployment_not_fully_ready",
    )
    replicasets = _list(_kubectl(context, "replicasets"), "invalid_replicasets")
    owned_sets = [row for row in replicasets if _owner_uid(row, "Deployment") == deployment_uid]
    set_uids = {_metadata_uid(row, "invalid_replicaset_uid") for row in owned_sets}
    pods = _list(_kubectl(context, "pods", "-l", f"{MODEL_LABEL}={model}"), "invalid_pods")
    owned_pods = [row for row in pods if _owner_uid(row, "ReplicaSet") in set_uids]
    _need(len(owned_pods) == ready_replicas, "pod_count_differs_from_ready_replicas")
    pod_uids: list[str] = []
    node_names: list[str] = []
    for pod in owned_pods:
        pod_status = _mapping(pod.get("status"), "invalid_pod_status")
        conditions = pod_status.get("conditions") or []
        ready = any(
            isinstance(row, Mapping) and row.get("type") == "Ready" and row.get("status") == "True"
            for row in conditions
        )
        statuses = pod_status.get("containerStatuses") or []
        _need(
            ready and isinstance(statuses, list) and len(statuses) == 1,
            "pod_not_ready",
        )
        container = _mapping(statuses[0], "invalid_container_status")
        observed_image = str(container.get("imageID", "")).rsplit("@", 1)[-1]
        _need(observed_image == image_digest, "observed_image_drift")
        _need(
            container.get("ready") is True and container.get("restartCount") == 0,
            "container_not_clean",
        )
        pod_uids.append(_metadata_uid(pod, "invalid_pod_uid"))
        node_names.append(str(_mapping(pod.get("spec"), "invalid_pod_spec").get("nodeName")))
    services = _list(
        _kubectl(context, "services", "-l", f"{MODEL_LABEL}={model}"), "invalid_services"
    )
    _need(len(services) == 1, "service_not_unique")
    service = services[0]
    service_uid = _metadata_uid(service, "invalid_service_uid")
    _need(_owner_uid(service, "InferenceModel") == inference_uid, "service_owner_drift")
    slices = _list(
        _kubectl(context, "endpointslices", "-l", f"{MODEL_LABEL}={model}"),
        "invalid_endpoint_slices",
    )
    owned_slices = [row for row in slices if _owner_uid(row, "Service") == service_uid]
    _need(owned_slices, "endpoint_slice_missing")
    endpoints = [endpoint for item in owned_slices for endpoint in (item.get("endpoints") or [])]
    _need(len(endpoints) == ready_replicas, "endpoint_count_differs_from_ready_replicas")
    for endpoint in endpoints:
        endpoint_conditions = _mapping(endpoint, "invalid_endpoint").get("conditions") or {}
        _need(endpoint_conditions.get("ready") is True, "endpoint_not_ready")
    return {
        "inference_model_uid": inference_uid,
        "inference_model_resource_version": _mapping(
            inference.get("metadata"), "invalid_inference_metadata"
        ).get("resourceVersion"),
        "deployment_uid": deployment_uid,
        "replicaset_uids": sorted(set_uids),
        "pod_uids": sorted(pod_uids),
        "service_uid": service_uid,
        "endpoint_slice_uids": sorted(
            _metadata_uid(row, "invalid_endpoint_slice_uid") for row in owned_slices
        ),
        "node_names": sorted(node_names),
        "ready_replicas": ready_replicas,
        "restart_counts": [0] * ready_replicas,
        "image_digest": image_digest,
        "ready_endpoint_count": ready_replicas,
        "normalized_inference_spec_sha256": _digest(_normalized_contract(inference.get("spec"))),
    }


def catalog_projection(value: object, model: str, revision: str) -> dict[str, Any]:
    rows = _mapping(value, "invalid_catalog").get("data")
    _need(isinstance(rows, list), "invalid_catalog_rows")
    matches = [row for row in rows if isinstance(row, Mapping) and row.get("id") == model]
    _need(len(matches) == 1, "catalog_route_not_unique")
    row = _mapping(matches[0], "invalid_catalog_row")
    _need(
        row.get("model_revision") == revision
        and row.get("status") == "ready"
        and row.get("routed") is True
        and row.get("ready_replicas", 0) >= 1,
        "catalog_route_not_ready",
    )
    return {
        field: copy.deepcopy(row.get(field))
        for field in (
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
        )
    }


def model_projection(value: object, path: str, model: str) -> dict[str, Any]:
    info = _mapping(value, "invalid_model_info")
    _need(
        info.get("model_path") == info.get("tokenizer_path") == path
        and info.get("served_model_name") == model,
        "model_info_identity_drift",
    )
    return {
        field: copy.deepcopy(info.get(field))
        for field in (
            "architectures",
            "model_type",
            "load_format",
            "reasoning_parser",
            "tool_call_parser",
            "weight_version",
        )
    }


def server_projection(value: object, path: str, model: str) -> dict[str, Any]:
    info = _mapping(value, "invalid_server_info")
    _need(
        info.get("model_path") == path and info.get("served_model_name") == model,
        "server_info_identity_drift",
    )
    fields = (
        "version",
        "context_length",
        "tp_size",
        "dp_size",
        "dtype",
        "quantization",
        "kv_cache_dtype",
        "attention_backend",
        "decode_attention_backend",
        "prefill_attention_backend",
        "chunked_prefill_size",
        "max_prefill_tokens",
        "reasoning_parser",
        "tool_call_parser",
        "speculative_algorithm",
        "speculative_num_steps",
        "speculative_eagle_topk",
        "speculative_num_draft_tokens",
        "load_format",
        "weight_version",
    )
    projection = {field: copy.deepcopy(info.get(field)) for field in fields}
    _need(
        projection["context_length"] == 262144
        and projection["tp_size"] == 1
        and projection["dp_size"] == 8
        and projection["quantization"] is None,
        "server_profile_unexpected",
    )
    return projection


def _tool_request(model: str) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": "Call identity exactly once with value parity."}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "identity",
                    "description": "Return the supplied value.",
                    "parameters": {
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                        "required": ["value"],
                        "additionalProperties": False,
                    },
                },
            }
        ],
        "tool_choice": {"type": "function", "function": {"name": "identity"}},
        "chat_template_kwargs": {"enable_thinking": False},
        "temperature": 0,
        "max_tokens": 128,
    }


def _logit_request(model: str) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with exactly: parity"}],
        "temperature": 0,
        "max_tokens": 8,
        "logprobs": True,
        "top_logprobs": 0,
    }


def probe_projection(
    tool: object, first_logits: object, second_logits: object, model: str
) -> dict[str, Any]:
    tool = _mapping(tool, "invalid_tool_response")
    _need(tool.get("model") == model, "tool_served_model_drift")
    choices = tool.get("choices")
    _need(isinstance(choices, list) and len(choices) == 1, "tool_choice_not_unique")
    message = _mapping(
        _mapping(choices[0], "invalid_tool_choice").get("message"), "invalid_tool_message"
    )
    calls = message.get("tool_calls")
    _need(isinstance(calls, list) and len(calls) == 1, "tool_call_not_unique")
    function = _mapping(
        _mapping(calls[0], "invalid_tool_call").get("function"), "invalid_tool_function"
    )
    try:
        arguments = json.loads(function.get("arguments"))
    except (TypeError, json.JSONDecodeError) as exc:
        raise RouteError("invalid_tool_arguments") from exc
    _need(
        function.get("name") == "identity" and arguments == {"value": "parity"},
        "tool_call_unexpected",
    )

    def logits(value: object) -> list[dict[str, Any]]:
        response = _mapping(value, "invalid_logit_response")
        _need(response.get("model") == model, "logit_served_model_drift")
        rows = response.get("choices")
        _need(isinstance(rows, list) and len(rows) == 1, "logit_choice_not_unique")
        logprobs = _mapping(
            _mapping(rows[0], "invalid_logit_choice").get("logprobs"), "missing_logprobs"
        )
        content = logprobs.get("content")
        _need(isinstance(content, list) and content, "empty_logprobs")
        projected = []
        for row in content:
            item = _mapping(row, "invalid_logprob_token")
            score = item.get("logprob")
            token_bytes = item.get("bytes")
            _need(
                isinstance(score, (int, float))
                and math.isfinite(score)
                and isinstance(item.get("token"), str)
                and isinstance(token_bytes, list)
                and all(type(byte) is int and 0 <= byte <= 255 for byte in token_bytes),
                "invalid_logprob_value",
            )
            projected.append(
                {"token": item["token"], "bytes": token_bytes, "logprob": float(score)}
            )
        return projected

    first = logits(first_logits)
    second = logits(second_logits)
    first_identity = [{"token": row["token"], "bytes": row["bytes"]} for row in first]
    second_identity = [{"token": row["token"], "bytes": row["bytes"]} for row in second]
    # The shared base route can load-balance identical greedy requests across
    # replicas. Floating-point log probabilities may then differ slightly even
    # when the generated token sequence is stable. Treat exact token identity
    # as the determinism gate and retain the finite score delta as evidence;
    # requiring bit-identical scores would make route qualification depend on
    # which healthy base replica happened to answer.
    _need(first_identity == second_identity, "fixed_logit_token_identity_not_deterministic")
    max_abs_logprob_delta = max(
        abs(left["logprob"] - right["logprob"]) for left, right in zip(first, second, strict=True)
    )
    return {
        "served_model": model,
        "tool_call_passed": True,
        "tool_name": "identity",
        "tool_argument_keys": ["value"],
        "tool_response_sha256": _digest(tool),
        "logits_finite": True,
        "logit_token_identity_deterministic": True,
        "logit_values_exactly_equal": first == second,
        "max_abs_logprob_delta": max_abs_logprob_delta,
        "logit_token_count": len(first),
        "logit_response_sha256": [_digest(first_logits), _digest(second_logits)],
        "logit_projection_sha256": _digest(first),
        "logit_token_identity_sha256": _digest(first_identity),
    }


def _capture_route(
    *,
    key: str,
    context: str,
    client: Client,
    label: str,
    model: str,
    expected_revision: str | None,
    expected_source_path: str | None,
    expected_replicas: int | None,
) -> dict[str, Any]:
    """Return one sanitized, content-free live route observation.

    ``capture`` uses this for the two arms of a matched comparison.  The
    standalone-base gate uses the exact same readbacks and fixed probes, but
    deliberately does not pretend that a base route is a checkpoint pair.
    """

    _need(
        (expected_revision is None) == (expected_source_path is None),
        "incomplete_expected_artifact_identity",
    )
    _, api = client.request("GET", API + "/" + model)
    api = _mapping(api, f"invalid_{label}_api")
    status = _mapping(api.get("status"), f"invalid_{label}_status")
    _need(
        status.get("phase") == "ready" and status.get("ready_replicas", 0) >= 1,
        f"{label}_not_ready",
    )
    spec = _mapping(api.get("spec"), f"invalid_{label}_spec")
    model_spec = _mapping(spec.get("model"), f"invalid_{label}_model")
    runtime = _mapping(spec.get("runtime"), f"invalid_{label}_runtime")
    revision = str(model_spec.get("revision"))
    if expected_revision is not None:
        _need(
            revision == expected_revision and model_spec.get("sourcePath") == expected_source_path,
            f"{label}_artifact_identity_drift",
        )
    image = _mapping(runtime.get("image"), f"invalid_{label}_image")
    image_digest = str(image.get("digest"))
    catalog = catalog_projection(
        _request(key, "/fleet/v1/model-catalog", model=model), model, revision
    )
    model_info_raw = _request(key, "/model_info", model=model)
    server_info_raw = _request(key, "/server_info", model=model)
    tool = _request(key, "/v1/chat/completions", model=model, payload=_tool_request(model))
    first = _request(key, "/v1/chat/completions", model=model, payload=_logit_request(model))
    second = _request(key, "/v1/chat/completions", model=model, payload=_logit_request(model))
    kubernetes = workload_projection(
        context,
        model,
        image_digest,
        expected_replicas=expected_replicas,
    )
    normalized_contract_sha256 = _digest(_normalized_contract(spec))
    _need(
        normalized_contract_sha256 == kubernetes["normalized_inference_spec_sha256"],
        f"{label}_api_kubernetes_spec_drift",
    )
    # Probe requests can update live routing status. Re-read the API after all
    # probes and bind the final resource-version observation while requiring
    # the immutable serving spec to remain unchanged. API and Kubernetes
    # resource versions are recorded separately because their status writes
    # need not be atomic.
    _, final_api = client.request("GET", API + "/" + model)
    final_api = _mapping(final_api, f"invalid_final_{label}_api")
    final_status = _mapping(final_api.get("status"), f"invalid_final_{label}_status")
    _need(
        final_status.get("phase") == "ready" and final_status.get("ready_replicas", 0) >= 1,
        f"final_{label}_not_ready",
    )
    final_spec = _mapping(final_api.get("spec"), f"invalid_final_{label}_spec")
    _need(
        _digest(_normalized_contract(final_spec)) == normalized_contract_sha256,
        f"{label}_spec_changed_during_capture",
    )
    return {
        "served_model": model,
        "resource_version": str(final_api.get("resource_version")),
        "kubernetes_resource_version": str(kubernetes["inference_model_resource_version"]),
        "model_revision": revision,
        "source_path": model_spec.get("sourcePath"),
        "serving_path": model_spec.get("path"),
        "normalized_contract_sha256": normalized_contract_sha256,
        "registration_spec_sha256": _digest(final_spec),
        "catalog": catalog,
        "model_info": model_projection(model_info_raw, str(model_spec.get("path")), model),
        "server_info": server_projection(server_info_raw, str(model_spec.get("path")), model),
        "kubernetes": kubernetes,
        "probes": probe_projection(tool, first, second, model),
    }


def capture(
    key: str,
    context: str,
    base_model: str,
    candidate_model: str,
    expected_revision: str,
    expected_source_path: str,
) -> dict[str, Any]:
    client = Client(key)
    arms = {
        "base": _capture_route(
            key=key,
            context=context,
            client=client,
            label="base",
            model=base_model,
            expected_revision=None,
            expected_source_path=None,
            expected_replicas=None,
        ),
        "candidate": _capture_route(
            key=key,
            context=context,
            client=client,
            label="candidate",
            model=candidate_model,
            expected_revision=expected_revision,
            expected_source_path=expected_source_path,
            expected_replicas=1,
        ),
    }
    base = arms["base"]
    candidate = arms["candidate"]
    _need(
        base["normalized_contract_sha256"] == candidate["normalized_contract_sha256"],
        "serving_execution_contract_drift",
    )
    _need(base["model_info"] == candidate["model_info"], "model_info_runtime_drift")
    _need(base["server_info"] == candidate["server_info"], "server_info_runtime_drift")
    logit_projection_differs = (
        base["probes"]["logit_projection_sha256"] != candidate["probes"]["logit_projection_sha256"]
    )
    receipt = {
        "schema": "cyber_checkpoint_serving_live_parity_v2",
        "status": "passed",
        "observed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "endpoint_origin": ORIGIN,
        "arms": arms,
        "held_constant": {
            "normalized_contract_sha256": base["normalized_contract_sha256"],
            "model_info_sha256": _digest(base["model_info"]),
            "server_info_sha256": _digest(base["server_info"]),
            "inference_precision": "bf16",
            "max_context_size": 262144,
            "weight_quantization": "none",
        },
        "benchmark_content_included": False,
        "response_content_recorded": False,
        "scores_observed": False,
        "task_content_included": False,
        "external_mutations_performed": 0,
        "resource_versions_are_non_atomic_live_observations": True,
        "fixed_probe_logit_projection_differs_between_weights": logit_projection_differs,
    }
    receipt["receipt_sha256"] = _digest(receipt)
    return receipt


def capture_base_route(
    key: str,
    context: str,
    base_model: str,
    expected_revision: str,
    expected_source_path: str,
) -> dict[str, Any]:
    """Capture the current, content-free identity of one exact base route.

    This is intentionally a route proof rather than a fake two-arm parity
    receipt.  A later paired checkpoint comparison still needs a fresh
    base-versus-candidate ``capture`` receipt.
    """

    route = _capture_route(
        key=key,
        context=context,
        client=Client(key),
        label="base",
        model=base_model,
        expected_revision=expected_revision,
        expected_source_path=expected_source_path,
        expected_replicas=None,
    )
    receipt = {
        "schema": "cyber_base_serving_live_route_proof_v1",
        "status": "passed",
        "observed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "endpoint_origin": ORIGIN,
        "route": route,
        "benchmark_content_included": False,
        "response_content_recorded": False,
        "scores_observed": False,
        "task_content_included": False,
        "external_mutations_performed": 0,
    }
    receipt["receipt_sha256"] = _digest(receipt)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model-id", default="qwen3.8-27b")
    parser.add_argument("--base-only", action="store_true")
    parser.add_argument("--candidate-model-id")
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-source-path", required=True)
    parser.add_argument("--kubernetes-context", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    try:
        if arguments.base_only:
            _need(arguments.candidate_model_id is None, "base_proof_rejects_candidate_model")
            value = capture_base_route(
                os.environ.get("FLEET_API_KEY", ""),
                arguments.kubernetes_context,
                arguments.base_model_id,
                arguments.expected_revision,
                arguments.expected_source_path,
            )
        else:
            _need(bool(arguments.candidate_model_id), "candidate_model_id_required")
            value = capture(
                os.environ.get("FLEET_API_KEY", ""),
                arguments.kubernetes_context,
                arguments.base_model_id,
                arguments.candidate_model_id,
                arguments.expected_revision,
                arguments.expected_source_path,
            )
        _write_once(arguments.output.resolve(), value)
        print(
            json.dumps(
                {
                    "base": arguments.base_model_id,
                    "candidate": arguments.candidate_model_id,
                    "receipt_sha256": value["receipt_sha256"],
                    "status": value["status"],
                },
                sort_keys=True,
            )
        )
        return 0
    except Exception as exc:
        print(
            json.dumps({"error_type": type(exc).__name__, "error": str(exc)}, sort_keys=True),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
