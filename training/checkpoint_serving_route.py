"""Create and control a base-matched checkpoint serving route.

The tool clones the current live base-model serving contract and changes only
the route identity, checkpoint paths, checkpoint revision, lifecycle state and
project priority.  Creation is paused and create-once.  Resume and pause use
the control plane's exact current resource version.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cyber_post_train import gpu_capacity

API = "https://inference.flt.build/fleet/v1/models"
ACCOUNT = "https://orchestrator.fleetai.com/v1/account"
TEAM_ID = "a1025f0b-ad67-49fc-a023-51800ab43e84"
_MODEL_ID = re.compile(r"[a-z0-9](?:[-a-z0-9]*[a-z0-9])?")
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}")
EXTERNAL_CTF_CLONE_INTENT_SCHEMA = "external_ctf_matched_base_clone_intent_v1"
EXTERNAL_CTF_CLONE_INTENT_SHA256 = (
    "sha256:74b6a6629e27562b55369a90960a9731c1af14d2ae057800542b902617d56c02"
)
EXTERNAL_CTF_RESUME_PREFLIGHT_SCHEMA = "external_ctf_base_clone_resume_preflight_v1"
EXTERNAL_CTF_BASE_CLONE_ID = "chris-q38-base-extctf-c1-v1"
EXTERNAL_CTF_CANDIDATE_ID = "chris-q38-t3k32-s1000-v1"
EXTERNAL_CTF_CAPACITY_MAX_NODES = 10
EXTERNAL_CTF_CAPACITY_MAX_GPUS = 80
EXTERNAL_CTF_RESUME_PREFLIGHT_MAX_AGE_SECONDS = 300


class RouteError(RuntimeError):
    """The serving route cannot be created or changed safely."""


def _need(condition: object, reason: str) -> None:
    if not condition:
        raise RouteError(reason)


def _mapping(value: object, reason: str) -> dict[str, Any]:
    _need(isinstance(value, Mapping), reason)
    return copy.deepcopy(dict(value))


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(_canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def _read(path: Path) -> dict[str, Any]:
    before = path.read_bytes()
    value = json.loads(before)
    _need(isinstance(value, Mapping) and path.read_bytes() == before, "plan_changed_while_reading")
    return dict(value)


def _file_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _read_canonical(path: Path, label: str) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RouteError(label + "_invalid") from exc
    _need(
        isinstance(value, Mapping) and raw == _canonical(value) + b"\n",
        label + "_not_canonical",
    )
    return dict(value)


def load_external_ctf_clone_intent(path: Path) -> dict[str, Any]:
    try:
        value = _read(path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RouteError("external_ctf_clone_intent_invalid") from exc
    claimed = value.get("intent_sha256")
    _need(
        claimed == EXTERNAL_CTF_CLONE_INTENT_SHA256
        and claimed
        == _digest({key: item for key, item in value.items() if key != "intent_sha256"}),
        "external_ctf_clone_intent_digest_mismatch",
    )
    source = _mapping(value.get("source"), "external_ctf_clone_source_invalid")
    target = _mapping(value.get("target"), "external_ctf_clone_target_invalid")
    candidate = _mapping(value.get("matched_candidate"), "external_ctf_clone_candidate_invalid")
    contract = _mapping(value.get("contract"), "external_ctf_clone_contract_invalid")
    prepare_value = _mapping(
        value.get("provider_free_prepare"), "external_ctf_clone_prepare_invalid"
    )
    quota = _mapping(value.get("quota"), "external_ctf_clone_quota_invalid")
    resume = _mapping(
        value.get("post_create_resume_preflight"),
        "external_ctf_clone_resume_preflight_invalid",
    )
    _need(
        value.get("schema") == EXTERNAL_CTF_CLONE_INTENT_SCHEMA
        and value.get("status") == "provider_free_no_mutation"
        and source.get("served_model") == "qwen3.8-27b"
        and source.get("model_revision") == "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
        and source.get("source_path")
        == "/models/qwen3.8-27b/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
        and target.get("served_model") == EXTERNAL_CTF_BASE_CLONE_ID
        and target.get("desired_state_at_create") == "paused"
        and target.get("minimum_replicas_at_create") == 0
        and target.get("replicas_when_resumed") == 1
        and target.get("priority_class") == "c1"
        and target.get("nodes_when_resumed") == 1
        and target.get("gpus_when_resumed") == 8
        and target.get("tensor_parallel_size") == 1
        and target.get("data_parallel_size") == 8
        and target.get("precision") == "bf16"
        and candidate.get("served_model") == EXTERNAL_CTF_CANDIDATE_ID
        and candidate.get("model_revision")
        == "sha256:023c5f8b0559ba050f0d672a6bc27aabecec7d5837595f8ea5bc914446d26db5"
        and candidate.get("source_path") == "/models/chris-q38-t3k32-s1000-v1"
        and candidate.get("priority_class") == "c1"
        and contract.get(
            "all_other_effective_serving_fields_must_match_after_reviewed_normalization"
        )
        is True
        and contract.get("all_other_source_spec_fields_must_be_byte_equal") is None
        and contract.get("existing_base_route_mutations") == 0
        and contract.get("existing_candidate_route_mutations") == 0
        and contract.get("target_name_must_be_absent_before_create") is True
        and contract.get("create_post_attempts") == 1
        and contract.get("ambiguous_create_requires_read_only_reconciliation") is True
        and prepare_value.get("external_mutations") == 0
        and prepare_value.get("required_readbacks")
        == [
            "fleet_team_identity",
            "exact_live_base_uid_generation_resource_version_and_spec",
            "target_route_absent",
        ]
        and quota
        == {
            "maximum_project_gpu_nodes": EXTERNAL_CTF_CAPACITY_MAX_NODES,
            "nodes_added_by_paused_create": 0,
            "nodes_added_by_single_resume": 1,
            "resume_requires_fresh_census_with_proposed_allocation_at_or_below_limit": True,
        }
        and resume.get("maximum_receipt_age_seconds")
        == EXTERNAL_CTF_RESUME_PREFLIGHT_MAX_AGE_SECONDS
        and resume.get("resume_requires_fresh_recheck_immediately_before_post") is True
        and value.get("kubernetes_job_created") is False
        and value.get("fleet_failure_alert_annotation_applicable") is False
        and value.get("external_mutations_performed") == 0,
        "external_ctf_clone_intent_binding_invalid",
    )
    return value


def _replace_arg(arguments: list[Any], flag: str, value: str) -> None:
    _need(arguments.count(flag) == 1, f"{flag}_not_unique")
    index = arguments.index(flag)
    _need(index + 1 < len(arguments) and isinstance(arguments[index + 1], str), "invalid_args")
    arguments[index + 1] = value


def _normalized_contract(spec_value: object) -> dict[str, Any]:
    spec = _mapping(spec_value, "invalid_spec")
    model = _mapping(spec.get("model"), "invalid_model")
    runtime = _mapping(spec.get("runtime"), "invalid_runtime")
    placement = _mapping(spec.get("placement"), "invalid_placement")
    resources = _mapping(spec.get("resources"), "invalid_resources")
    requests = _mapping(resources.get("requests"), "invalid_resource_requests")
    limits = _mapping(resources.get("limits"), "invalid_resource_limits")
    arguments = runtime.get("args")
    _need(isinstance(arguments, list), "invalid_runtime_args")
    _replace_arg(arguments, "--model-path", "$MODEL_PATH")
    _replace_arg(arguments, "--served-model-name", "$MODEL_ID")
    runtime["args"] = arguments
    for field, value in (
        ("path", "$MODEL_PATH"),
        ("sourcePath", "$SOURCE_PATH"),
        ("revision", "$REVISION"),
    ):
        _need(isinstance(model.get(field), str) and model[field], f"invalid_model_{field}")
        model[field] = value
    spec["model"] = model
    spec["runtime"] = runtime
    capabilities = spec.get("capabilities")
    _need(
        isinstance(capabilities, list) and all(isinstance(value, str) for value in capabilities),
        "invalid_capabilities",
    )
    spec["capabilities"] = sorted(capabilities)
    # Kubernetes defaults a missing request to the corresponding limit. The
    # inference API retains the explicit request, so normalize both views to
    # the same effective resource contract.
    requests.setdefault("nvidia.com/gpu", limits.get("nvidia.com/gpu"))
    resources["requests"] = requests
    resources["limits"] = limits
    spec["resources"] = resources
    spec["displayName"] = "$DISPLAY_NAME"
    spec["desiredState"] = "$DESIRED_STATE"
    placement["priorityClassName"] = "$PRIORITY"
    spec["placement"] = placement
    spec["scaling"] = {"lifecycle_managed": True}
    return spec


def build_paused_spec(
    base_spec_value: object,
    *,
    model_id: str,
    display_name: str,
    source_path: str,
    revision: str,
    allow_git_revision: bool = False,
) -> dict[str, Any]:
    _need(_MODEL_ID.fullmatch(model_id), "invalid_model_id")
    _need(source_path.startswith("/models/") and ".." not in source_path, "invalid_source_path")
    _need(
        _SHA256.fullmatch(revision) is not None
        or (allow_git_revision and re.fullmatch(r"[0-9a-f]{40}", revision) is not None),
        "invalid_revision",
    )
    base = _mapping(base_spec_value, "invalid_base_spec")
    candidate = copy.deepcopy(base)
    model = _mapping(candidate.get("model"), "invalid_base_model")
    runtime = _mapping(candidate.get("runtime"), "invalid_base_runtime")
    placement = _mapping(candidate.get("placement"), "invalid_base_placement")
    resources = _mapping(candidate.get("resources"), "invalid_base_resources")
    limits = _mapping(resources.get("limits"), "invalid_base_limits")
    requests = _mapping(resources.get("requests"), "invalid_base_requests")
    _need(limits.get("nvidia.com/gpu") == 8, "base_is_not_one_eight_gpu_replica")
    requests["nvidia.com/gpu"] = 8
    resources["requests"] = requests
    scratch_path = "/scratch/models/" + model_id
    model.update(
        {
            "sourcePath": source_path,
            "path": scratch_path,
            "revision": revision,
        }
    )
    arguments = runtime.get("args")
    _need(isinstance(arguments, list), "invalid_base_runtime_args")
    _replace_arg(arguments, "--model-path", scratch_path)
    _replace_arg(arguments, "--served-model-name", model_id)
    runtime["args"] = arguments
    placement["priorityClassName"] = "c1"
    candidate.update(
        {
            "displayName": display_name,
            "desiredState": "paused",
            "model": model,
            "runtime": runtime,
            "placement": placement,
            "resources": resources,
            "scaling": {"minReplicas": 0, "replicas": 1},
        }
    )
    _need(candidate.get("routing") == {"enabled": True}, "routing_not_enabled")
    _need(
        _normalized_contract(base) == _normalized_contract(candidate),
        "candidate_differs_from_base_beyond_weights_and_lifecycle",
    )
    return candidate


class Client:
    def __init__(self, key: str):
        _need(bool(key), "FLEET_API_KEY_required")
        self._key = key

    def request(
        self,
        method: str,
        url: str,
        body: Mapping[str, Any] | None = None,
        *,
        idempotency: str | None = None,
        if_match: str | None = None,
        allow_404: bool = False,
    ) -> tuple[int, dict[str, Any] | None]:
        _need(method in {"GET", "POST"}, "unsupported_method")
        _need(url in (ACCOUNT, API) or url.startswith(API + "/"), "unapproved_url")
        headers = {"Authorization": "Bearer " + self._key}
        if idempotency:
            headers["Idempotency-Key"] = idempotency
        if if_match:
            headers["If-Match"] = if_match
        data = None
        if body is not None:
            headers["Content-Type"] = "application/json"
            data = _canonical(body)
        request = urllib.request.Request(url, method=method, headers=headers, data=data)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                raw = response.read()
                value = json.loads(raw) if raw else {}
                _need(isinstance(value, Mapping), "invalid_response")
                return response.status, dict(value)
        except urllib.error.HTTPError as exc:
            if allow_404 and exc.code == 404:
                return 404, None
            raise RouteError(f"control_plane_http_{exc.code}") from None


def _team(client: Client) -> None:
    _, value = client.request("GET", ACCOUNT)
    value = _mapping(value, "invalid_account")
    team = (
        value.get("team_id")
        or _mapping(value.get("team", {}), "invalid_team").get("id")
        or value.get("id")
    )
    _need(team == TEAM_ID, "unexpected_fleet_team")


def prepare(
    client: Client,
    *,
    base_model_id: str,
    source_model_id: str,
    target_model_id: str,
    display_name: str,
    source_path: str,
    revision: str,
    external_ctf_clone_intent: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _team(client)
    _, base = client.request("GET", API + "/" + urllib.parse.quote(base_model_id, safe=""))
    _, source = client.request("GET", API + "/" + urllib.parse.quote(source_model_id, safe=""))
    code, target = client.request(
        "GET", API + "/" + urllib.parse.quote(target_model_id, safe=""), allow_404=True
    )
    _need(code == 404 and target is None, "target_route_already_exists")
    base = _mapping(base, "invalid_base")
    source = _mapping(source, "invalid_source")
    base_status = _mapping(base.get("status"), "invalid_base_status")
    source_status = _mapping(source.get("status"), "invalid_source_status")
    _need(
        base_status.get("phase") == "ready" and base_status.get("ready_replicas", 0) >= 1,
        "base_not_ready",
    )
    same_weight_base_clone = source_model_id == base_model_id
    if external_ctf_clone_intent is not None:
        intent_source = _mapping(
            external_ctf_clone_intent.get("source"), "external_ctf_clone_source_invalid"
        )
        intent_target = _mapping(
            external_ctf_clone_intent.get("target"), "external_ctf_clone_target_invalid"
        )
        intent_contract = _mapping(
            external_ctf_clone_intent.get("contract"),
            "external_ctf_clone_contract_invalid",
        )
        _need(
            same_weight_base_clone
            and base_model_id == intent_source.get("served_model")
            and source_model_id == intent_source.get("served_model")
            and target_model_id == intent_target.get("served_model")
            and display_name == intent_target.get("display_name")
            and source_path == intent_source.get("source_path")
            and revision == intent_source.get("model_revision"),
            "external_ctf_clone_prepare_arguments_drifted",
        )
    if same_weight_base_clone:
        _need(
            source.get("spec") == base.get("spec")
            and (source.get("uid") or source.get("metadata", {}).get("uid"))
            == (base.get("uid") or base.get("metadata", {}).get("uid"))
            and source_status.get("phase") == "ready"
            and source_status.get("ready_replicas", 0) >= 1,
            "base_clone_source_identity_mismatch",
        )
    else:
        _need(
            source_status.get("phase") == "paused" and source_status.get("active_pods") == 0,
            "source_not_paused_zero_gpu",
        )
    source_model = _mapping(
        _mapping(source.get("spec"), "invalid_source_spec").get("model"), "invalid_source_model"
    )
    _need(
        source_model.get("sourcePath") == source_path and source_model.get("revision") == revision,
        "source_artifact_identity_mismatch",
    )
    spec = build_paused_spec(
        base.get("spec"),
        model_id=target_model_id,
        display_name=display_name,
        source_path=source_path,
        revision=revision,
        allow_git_revision=same_weight_base_clone,
    )
    if external_ctf_clone_intent is not None:
        _need(
            _digest(_normalized_contract(spec))
            == intent_contract.get("normalized_serving_contract_sha256"),
            "external_ctf_clone_normalized_contract_drifted",
        )
    registration = {"id": target_model_id, "spec": spec}
    plan = {
        "schema": "cyber_checkpoint_serving_route_plan_v1",
        "base_model_id": base_model_id,
        "base_model_uid": base.get("uid") or base.get("metadata", {}).get("uid"),
        "base_resource_version": str(base.get("resource_version")),
        "source_model_id": source_model_id,
        "source_model_uid": source.get("uid") or source.get("metadata", {}).get("uid"),
        "source_resource_version": str(source.get("resource_version")),
        "source_role": (
            "same_weight_base_clone" if same_weight_base_clone else "checkpoint_artifact_route"
        ),
        "registration": registration,
        "registration_sha256": _digest(registration),
        "normalized_contract_sha256": _digest(_normalized_contract(spec)),
        "mutation_count": 0,
    }
    if external_ctf_clone_intent is not None:
        plan["external_ctf_clone_intent_sha256"] = external_ctf_clone_intent["intent_sha256"]
    plan["plan_sha256"] = _digest(plan)
    return plan


def _validate_plan(plan: Mapping[str, Any]) -> None:
    copy_value = dict(plan)
    claimed = copy_value.pop("plan_sha256", None)
    _need(claimed == _digest(copy_value), "plan_digest_mismatch")
    registration = _mapping(plan.get("registration"), "invalid_registration")
    _need(plan.get("registration_sha256") == _digest(registration), "registration_digest_mismatch")
    _need(plan.get("mutation_count") == 0, "plan_claims_mutation")


def _created_readback(
    client: Client,
    model_id: str,
    registration: Mapping[str, Any],
    *,
    timeout: int = 30,
) -> dict[str, Any]:
    """Wait through control-plane read-after-create propagation only."""

    deadline = time.monotonic() + timeout
    url = API + "/" + urllib.parse.quote(model_id, safe="")
    while True:
        code, response = client.request("GET", url, allow_404=True)
        if code == 200:
            response = _mapping(response, "invalid_creation_readback")
            _need(response.get("id") == model_id, "created_model_id_mismatch")
            _need(response.get("spec") == registration["spec"], "created_spec_mismatch")
            return response
        _need(time.monotonic() < deadline, "creation_readback_not_visible")
        time.sleep(1)


def execute(
    client: Client,
    plan_path: Path,
    intent_path: Path,
    result_path: Path,
    external_ctf_clone_intent_path: Path | None = None,
) -> dict[str, Any]:
    plan = _read(plan_path)
    _validate_plan(plan)
    clone_intent = (
        None
        if external_ctf_clone_intent_path is None
        else load_external_ctf_clone_intent(external_ctf_clone_intent_path)
    )
    _need(
        (plan.get("external_ctf_clone_intent_sha256") is None) == (clone_intent is None),
        "external_ctf_clone_intent_required_or_unexpected",
    )
    if clone_intent is not None:
        _need(
            plan.get("external_ctf_clone_intent_sha256") == clone_intent["intent_sha256"],
            "external_ctf_clone_intent_plan_mismatch",
        )
    registration = _mapping(plan.get("registration"), "invalid_registration")
    model_id = str(registration.get("id"))
    fresh = prepare(
        client,
        base_model_id=str(plan["base_model_id"]),
        source_model_id=str(plan["source_model_id"]),
        target_model_id=model_id,
        display_name=str(registration["spec"]["displayName"]),
        source_path=str(registration["spec"]["model"]["sourcePath"]),
        revision=str(registration["spec"]["model"]["revision"]),
        external_ctf_clone_intent=clone_intent,
    )
    _need(fresh["registration_sha256"] == plan["registration_sha256"], "live_preview_drift")
    intent = {
        "schema": "cyber_checkpoint_serving_route_intent_v1",
        "plan_sha256": plan["plan_sha256"],
        "registration_sha256": plan["registration_sha256"],
        "post_will_be_attempted": True,
    }
    intent["receipt_sha256"] = _digest(intent)
    _write_once(intent_path, intent)
    _, response = client.request(
        "POST",
        API,
        registration,
        idempotency="checkpoint-route-" + plan["registration_sha256"].removeprefix("sha256:"),
    )
    response = _mapping(response, "invalid_creation_response")
    _need(response.get("id") == model_id, "created_model_id_mismatch")
    if response.get("spec") != registration["spec"]:
        response = _created_readback(client, model_id, registration)
    _need(response.get("spec") == registration["spec"], "created_spec_mismatch")
    result = {
        "schema": "cyber_checkpoint_serving_route_result_v1",
        "plan_sha256": plan["plan_sha256"],
        "registration_sha256": plan["registration_sha256"],
        "model_id": model_id,
        "resource_version": str(response.get("resource_version")),
        "phase": _mapping(response.get("status"), "invalid_created_status").get("phase"),
        "post_attempts": 1,
    }
    _need(result["phase"] == "paused", "created_route_not_paused")
    result["receipt_sha256"] = _digest(result)
    _write_once(result_path, result)
    return result


def reconcile_create(
    client: Client,
    plan_path: Path,
    intent_path: Path,
    result_path: Path,
    external_ctf_clone_intent_path: Path | None = None,
) -> dict[str, Any]:
    """Reconcile an attempted create without issuing another POST."""

    plan = _read(plan_path)
    _validate_plan(plan)
    if plan.get("external_ctf_clone_intent_sha256") is not None:
        _need(
            external_ctf_clone_intent_path is not None
            and load_external_ctf_clone_intent(external_ctf_clone_intent_path)["intent_sha256"]
            == plan["external_ctf_clone_intent_sha256"],
            "external_ctf_clone_intent_plan_mismatch",
        )
    intent = _read(intent_path)
    _need(intent.get("plan_sha256") == plan["plan_sha256"], "intent_plan_mismatch")
    _need(
        intent.get("registration_sha256") == plan["registration_sha256"],
        "intent_registration_mismatch",
    )
    registration = _mapping(plan.get("registration"), "invalid_registration")
    model_id = str(registration.get("id"))
    _, response = client.request("GET", API + "/" + urllib.parse.quote(model_id, safe=""))
    response = _mapping(response, "invalid_creation_readback")
    _need(response.get("id") == model_id, "created_model_id_mismatch")
    _need(response.get("spec") == registration["spec"], "created_spec_mismatch")
    result = {
        "schema": "cyber_checkpoint_serving_route_result_v1",
        "plan_sha256": plan["plan_sha256"],
        "registration_sha256": plan["registration_sha256"],
        "model_id": model_id,
        "resource_version": str(response.get("resource_version")),
        "phase": _mapping(response.get("status"), "invalid_created_status").get("phase"),
        "post_attempts": 1,
        "reconciled_after_create": True,
    }
    _need(result["phase"] == "paused", "created_route_not_paused")
    result["receipt_sha256"] = _digest(result)
    _write_once(result_path, result)
    return result


def _route_readback(value: object, label: str) -> dict[str, Any]:
    route = _mapping(value, label + "_invalid")
    metadata = _mapping(route.get("metadata", {}), label + "_metadata_invalid")
    spec = _mapping(route.get("spec"), label + "_spec_invalid")
    status = _mapping(route.get("status"), label + "_status_invalid")
    uid = route.get("uid") or metadata.get("uid")
    generation = route.get("generation") or metadata.get("generation")
    resource_version = route.get("resource_version") or metadata.get("resourceVersion")
    _need(
        isinstance(uid, str)
        and uid
        and type(generation) is int
        and generation > 0
        and isinstance(resource_version, str)
        and resource_version,
        label + "_identity_invalid",
    )
    return {
        "id": route.get("id") or metadata.get("name"),
        "uid": uid,
        "generation": generation,
        "resource_version": resource_version,
        "spec_sha256": _digest(spec),
        "spec": spec,
        "phase": status.get("phase"),
        "ready_replicas": status.get("ready_replicas"),
        "active_pods": status.get("active_pods"),
    }


def _timestamp(value: object) -> datetime:
    _need(isinstance(value, str), "resume_preflight_timestamp_invalid")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RouteError("resume_preflight_timestamp_invalid") from exc
    _need(
        parsed.tzinfo is not None and parsed.utcoffset() is not None,
        "resume_preflight_timestamp_invalid",
    )
    return parsed.astimezone(UTC)


def _validate_external_ctf_clone_result(result: Mapping[str, Any], plan: Mapping[str, Any]) -> None:
    result_unsigned = {key: item for key, item in result.items() if key != "receipt_sha256"}
    _need(
        result.get("receipt_sha256") == _digest(result_unsigned)
        and result.get("schema") == "cyber_checkpoint_serving_route_result_v1"
        and result.get("plan_sha256") == plan.get("plan_sha256")
        and result.get("registration_sha256") == plan.get("registration_sha256")
        and result.get("model_id") == EXTERNAL_CTF_BASE_CLONE_ID
        and result.get("phase") == "paused"
        and result.get("post_attempts") == 1,
        "external_ctf_clone_result_invalid",
    )


def seal_external_ctf_resume_preflight(
    client: Client,
    *,
    clone_intent_path: Path,
    plan_path: Path,
    result_path: Path,
    kubernetes_context: str,
    output_path: Path,
) -> dict[str, Any]:
    clone_intent = load_external_ctf_clone_intent(clone_intent_path)
    plan = _read_canonical(plan_path, "external_ctf_clone_plan")
    _validate_plan(plan)
    _need(
        plan.get("external_ctf_clone_intent_sha256") == clone_intent["intent_sha256"],
        "external_ctf_clone_intent_plan_mismatch",
    )
    result = _read_canonical(result_path, "external_ctf_clone_result")
    _validate_external_ctf_clone_result(result, plan)
    _team(client)
    _, target_raw = client.request("GET", API + "/" + EXTERNAL_CTF_BASE_CLONE_ID)
    _, candidate_raw = client.request("GET", API + "/" + EXTERNAL_CTF_CANDIDATE_ID)
    target = _route_readback(target_raw, "external_ctf_clone_target")
    candidate = _route_readback(candidate_raw, "external_ctf_candidate")
    registration = _mapping(plan.get("registration"), "invalid_registration")
    candidate_spec = _mapping(candidate["spec"], "external_ctf_candidate_spec_invalid")
    candidate_model = _mapping(candidate_spec.get("model"), "external_ctf_candidate_model_invalid")
    candidate_placement = _mapping(
        candidate_spec.get("placement"), "external_ctf_candidate_placement_invalid"
    )
    _need(
        target["id"] == EXTERNAL_CTF_BASE_CLONE_ID
        and target["spec"] == registration["spec"]
        and target["phase"] == "paused"
        and target["active_pods"] == 0
        and candidate["id"] == EXTERNAL_CTF_CANDIDATE_ID
        and candidate["phase"] == "ready"
        and candidate["ready_replicas"] == 1
        and candidate_model.get("revision") == clone_intent["matched_candidate"]["model_revision"]
        and candidate_model.get("sourcePath") == clone_intent["matched_candidate"]["source_path"]
        and candidate_placement.get("priorityClassName") == "c1",
        "external_ctf_clone_or_candidate_readback_invalid",
    )
    _need(
        _digest(_normalized_contract(candidate_spec)) == plan["normalized_contract_sha256"],
        "external_ctf_candidate_normalized_contract_drifted",
    )
    capacity = gpu_capacity.live_capacity_census(
        kubernetes_context,
        owner_prefixes=("chris-q38-", "qwen3.8-27b"),
        max_nodes=EXTERNAL_CTF_CAPACITY_MAX_NODES,
        max_gpus=EXTERNAL_CTF_CAPACITY_MAX_GPUS,
        planned_nodes=1,
        planned_gpus=8,
    )
    _need(capacity.get("qualified") is True, "external_ctf_resume_capacity_not_qualified")
    value = {
        "schema": EXTERNAL_CTF_RESUME_PREFLIGHT_SCHEMA,
        "status": "passed_no_mutation",
        "observed_at": capacity["observed_at"],
        "fleet_team_id": TEAM_ID,
        "clone_intent_file_sha256": _file_digest(clone_intent_path),
        "clone_intent_sha256": clone_intent["intent_sha256"],
        "plan_file_sha256": _file_digest(plan_path),
        "plan_sha256": plan["plan_sha256"],
        "result_file_sha256": _file_digest(result_path),
        "result_receipt_sha256": result["receipt_sha256"],
        "target": {key: target[key] for key in target if key != "spec"},
        "candidate": {key: candidate[key] for key in candidate if key != "spec"},
        "capacity": capacity,
        "external_mutations_performed": 0,
    }
    value["receipt_sha256"] = _digest(value)
    _write_once(output_path, value)
    return value


def load_external_ctf_resume_preflight(
    path: Path,
    *,
    clone_intent_path: Path,
    plan_path: Path,
    result_path: Path,
    now: datetime | None = None,
) -> dict[str, Any]:
    value = _read_canonical(path, "external_ctf_resume_preflight")
    clone_intent = load_external_ctf_clone_intent(clone_intent_path)
    plan = _read_canonical(plan_path, "external_ctf_clone_plan")
    _validate_plan(plan)
    result = _read_canonical(result_path, "external_ctf_clone_result")
    _validate_external_ctf_clone_result(result, plan)
    current = datetime.now(UTC) if now is None else now.astimezone(UTC)
    age = (current - _timestamp(value.get("observed_at"))).total_seconds()
    capacity = value.get("capacity")
    target = value.get("target")
    candidate = value.get("candidate")
    registration = plan.get("registration")
    _need(
        value.get("receipt_sha256")
        == _digest({key: item for key, item in value.items() if key != "receipt_sha256"})
        and value.get("schema") == EXTERNAL_CTF_RESUME_PREFLIGHT_SCHEMA
        and value.get("status") == "passed_no_mutation"
        and value.get("fleet_team_id") == TEAM_ID
        and value.get("clone_intent_file_sha256") == _file_digest(clone_intent_path)
        and value.get("clone_intent_sha256") == clone_intent["intent_sha256"]
        and value.get("plan_file_sha256") == _file_digest(plan_path)
        and value.get("plan_sha256") == plan["plan_sha256"]
        and value.get("result_file_sha256") == _file_digest(result_path)
        and value.get("result_receipt_sha256") == result.get("receipt_sha256")
        and isinstance(registration, Mapping)
        and isinstance(target, Mapping)
        and target.get("id") == EXTERNAL_CTF_BASE_CLONE_ID
        and target.get("spec_sha256") == _digest(registration.get("spec"))
        and target.get("phase") == "paused"
        and target.get("active_pods") == 0
        and isinstance(candidate, Mapping)
        and candidate.get("id") == EXTERNAL_CTF_CANDIDATE_ID
        and candidate.get("phase") == "ready"
        and candidate.get("ready_replicas") == 1
        and isinstance(capacity, Mapping)
        and capacity.get("sha256")
        == gpu_capacity._digest(  # noqa: SLF001
            {key: item for key, item in capacity.items() if key != "sha256"}
        )
        and capacity.get("qualified") is True
        and capacity.get("limits")
        == {"nodes": EXTERNAL_CTF_CAPACITY_MAX_NODES, "gpus": EXTERNAL_CTF_CAPACITY_MAX_GPUS}
        and capacity.get("planned") == {"nodes": 1, "gpus": 8}
        and age >= -60
        and age <= EXTERNAL_CTF_RESUME_PREFLIGHT_MAX_AGE_SECONDS
        and value.get("external_mutations_performed") == 0,
        "external_ctf_resume_preflight_invalid_or_stale",
    )
    return value


def lifecycle(
    client: Client,
    model_id: str,
    action: str,
    *,
    external_ctf_resume_preflight_path: Path | None = None,
    external_ctf_clone_intent_path: Path | None = None,
    external_ctf_plan_path: Path | None = None,
    external_ctf_result_path: Path | None = None,
    kubernetes_context: str | None = None,
) -> dict[str, Any]:
    _need(action in {"resume", "pause"}, "invalid_lifecycle_action")
    resume_preflight = None
    immediate_capacity = None
    current_target = None
    current_candidate = None
    if model_id == EXTERNAL_CTF_BASE_CLONE_ID and action == "resume":
        _need(
            all(
                path is not None
                for path in (
                    external_ctf_resume_preflight_path,
                    external_ctf_clone_intent_path,
                    external_ctf_plan_path,
                    external_ctf_result_path,
                )
            )
            and isinstance(kubernetes_context, str)
            and bool(kubernetes_context),
            "external_ctf_resume_preflight_required",
        )
        resume_preflight = load_external_ctf_resume_preflight(
            external_ctf_resume_preflight_path,  # type: ignore[arg-type]
            clone_intent_path=external_ctf_clone_intent_path,  # type: ignore[arg-type]
            plan_path=external_ctf_plan_path,  # type: ignore[arg-type]
            result_path=external_ctf_result_path,  # type: ignore[arg-type]
        )
        _team(client)
        immediate_capacity = gpu_capacity.live_capacity_census(
            kubernetes_context,
            owner_prefixes=("chris-q38-", "qwen3.8-27b"),
            max_nodes=EXTERNAL_CTF_CAPACITY_MAX_NODES,
            max_gpus=EXTERNAL_CTF_CAPACITY_MAX_GPUS,
            planned_nodes=1,
            planned_gpus=8,
        )
        _need(
            immediate_capacity.get("qualified") is True,
            "external_ctf_immediate_resume_capacity_not_qualified",
        )
    url = API + "/" + urllib.parse.quote(model_id, safe="")
    _, before = client.request("GET", url)
    before = _mapping(before, "invalid_lifecycle_readback")
    status = _mapping(before.get("status"), "invalid_lifecycle_status")
    expected = "paused" if action == "resume" else "ready"
    _need(status.get("phase") == expected, f"route_not_{expected}")
    if resume_preflight is not None:
        current = _route_readback(before, "external_ctf_resume_target")
        current_target = {key: value for key, value in current.items() if key != "spec"}
        _need(
            all(
                current.get(field) == resume_preflight["target"].get(field)
                for field in ("id", "uid", "generation", "resource_version", "spec_sha256")
            ),
            "external_ctf_resume_target_drifted_after_preflight",
        )
        _, candidate_raw = client.request("GET", API + "/" + EXTERNAL_CTF_CANDIDATE_ID)
        candidate = _route_readback(candidate_raw, "external_ctf_resume_candidate")
        current_candidate = {key: value for key, value in candidate.items() if key != "spec"}
        _need(
            all(
                candidate.get(field) == resume_preflight["candidate"].get(field)
                for field in ("id", "uid", "generation", "resource_version", "spec_sha256")
            )
            and candidate.get("phase") == "ready"
            and candidate.get("ready_replicas") == 1,
            "external_ctf_resume_candidate_drifted_after_preflight",
        )
    resource_version = str(before.get("resource_version") or "")
    _need(bool(resource_version), "missing_resource_version")
    _, accepted = client.request(
        "POST",
        url + "/" + action,
        {},
        if_match=resource_version,
    )
    accepted = _mapping(accepted, "invalid_lifecycle_acceptance")
    result = {
        "model_id": model_id,
        "action": action,
        "before_phase": expected,
        "before_resource_version": resource_version,
        "accepted_phase": _mapping(accepted.get("status", {}), "invalid_accepted_status").get(
            "phase"
        ),
        "accepted_resource_version": str(accepted.get("resource_version") or ""),
    }
    if resume_preflight is not None and immediate_capacity is not None:
        result.update(
            {
                "schema": "external_ctf_base_clone_lifecycle_v1",
                "external_ctf_resume_preflight_receipt_sha256": resume_preflight["receipt_sha256"],
                "immediate_capacity_sha256": "sha256:" + immediate_capacity["sha256"],
                "immediate_capacity": immediate_capacity,
                "target_readback": current_target,
                "candidate_readback": current_candidate,
            }
        )
        result["receipt_sha256"] = _digest(result)
    return result


def wait_phase(client: Client, model_id: str, phase: str, timeout: int) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    url = API + "/" + urllib.parse.quote(model_id, safe="")
    while True:
        _, value = client.request("GET", url)
        value = _mapping(value, "invalid_wait_readback")
        status = _mapping(value.get("status"), "invalid_wait_status")
        current = status.get("phase")
        if current == phase:
            return {
                "model_id": model_id,
                "phase": current,
                "resource_version": str(value.get("resource_version")),
                "ready_replicas": status.get("ready_replicas"),
                "active_pods": status.get("active_pods"),
            }
        _need(current not in {"failed", "error", "retired"}, f"route_terminal_{current}")
        _need(time.monotonic() < deadline, "wait_timeout")
        time.sleep(10)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--base-model-id", default="qwen3.8-27b")
    prepare_parser.add_argument("--source-model-id", required=True)
    prepare_parser.add_argument("--target-model-id", required=True)
    prepare_parser.add_argument("--display-name", required=True)
    prepare_parser.add_argument("--source-path", required=True)
    prepare_parser.add_argument("--revision", required=True)
    prepare_parser.add_argument("--external-ctf-clone-intent", type=Path)
    prepare_parser.add_argument("--output", type=Path, required=True)
    execute_parser = commands.add_parser("execute")
    execute_parser.add_argument("--plan", type=Path, required=True)
    execute_parser.add_argument("--intent", type=Path, required=True)
    execute_parser.add_argument("--result", type=Path, required=True)
    execute_parser.add_argument("--external-ctf-clone-intent", type=Path)
    reconcile_parser = commands.add_parser("reconcile-create")
    reconcile_parser.add_argument("--plan", type=Path, required=True)
    reconcile_parser.add_argument("--intent", type=Path, required=True)
    reconcile_parser.add_argument("--result", type=Path, required=True)
    reconcile_parser.add_argument("--external-ctf-clone-intent", type=Path)
    resume_preflight_parser = commands.add_parser("external-ctf-resume-preflight")
    resume_preflight_parser.add_argument("--clone-intent", type=Path, required=True)
    resume_preflight_parser.add_argument("--plan", type=Path, required=True)
    resume_preflight_parser.add_argument("--result", type=Path, required=True)
    resume_preflight_parser.add_argument("--kubernetes-context", required=True)
    resume_preflight_parser.add_argument("--output", type=Path, required=True)
    lifecycle_parser = commands.add_parser("lifecycle")
    lifecycle_parser.add_argument("--model-id", required=True)
    lifecycle_parser.add_argument("--action", choices=("resume", "pause"), required=True)
    lifecycle_parser.add_argument("--external-ctf-resume-preflight", type=Path)
    lifecycle_parser.add_argument("--external-ctf-clone-intent", type=Path)
    lifecycle_parser.add_argument("--external-ctf-plan", type=Path)
    lifecycle_parser.add_argument("--external-ctf-create-result", type=Path)
    lifecycle_parser.add_argument("--kubernetes-context")
    lifecycle_parser.add_argument("--result", type=Path)
    wait_parser = commands.add_parser("wait")
    wait_parser.add_argument("--model-id", required=True)
    wait_parser.add_argument("--phase", choices=("ready", "paused"), required=True)
    wait_parser.add_argument("--timeout", type=int, default=1800)
    arguments = parser.parse_args(argv)
    try:
        client = Client(os.environ.get("FLEET_API_KEY", ""))
        if arguments.command == "prepare":
            clone_intent = (
                None
                if arguments.external_ctf_clone_intent is None
                else load_external_ctf_clone_intent(arguments.external_ctf_clone_intent.resolve())
            )
            value = prepare(
                client,
                base_model_id=arguments.base_model_id,
                source_model_id=arguments.source_model_id,
                target_model_id=arguments.target_model_id,
                display_name=arguments.display_name,
                source_path=arguments.source_path,
                revision=arguments.revision,
                external_ctf_clone_intent=clone_intent,
            )
            _write_once(arguments.output.resolve(), value)
        elif arguments.command == "execute":
            value = execute(
                client,
                arguments.plan.resolve(),
                arguments.intent.resolve(),
                arguments.result.resolve(),
                None
                if arguments.external_ctf_clone_intent is None
                else arguments.external_ctf_clone_intent.resolve(),
            )
        elif arguments.command == "reconcile-create":
            value = reconcile_create(
                client,
                arguments.plan.resolve(),
                arguments.intent.resolve(),
                arguments.result.resolve(),
                None
                if arguments.external_ctf_clone_intent is None
                else arguments.external_ctf_clone_intent.resolve(),
            )
        elif arguments.command == "external-ctf-resume-preflight":
            value = seal_external_ctf_resume_preflight(
                client,
                clone_intent_path=arguments.clone_intent.resolve(),
                plan_path=arguments.plan.resolve(),
                result_path=arguments.result.resolve(),
                kubernetes_context=arguments.kubernetes_context,
                output_path=arguments.output.resolve(),
            )
        elif arguments.command == "lifecycle":
            value = lifecycle(
                client,
                arguments.model_id,
                arguments.action,
                external_ctf_resume_preflight_path=(
                    None
                    if arguments.external_ctf_resume_preflight is None
                    else arguments.external_ctf_resume_preflight.resolve()
                ),
                external_ctf_clone_intent_path=(
                    None
                    if arguments.external_ctf_clone_intent is None
                    else arguments.external_ctf_clone_intent.resolve()
                ),
                external_ctf_plan_path=(
                    None
                    if arguments.external_ctf_plan is None
                    else arguments.external_ctf_plan.resolve()
                ),
                external_ctf_result_path=(
                    None
                    if arguments.external_ctf_create_result is None
                    else arguments.external_ctf_create_result.resolve()
                ),
                kubernetes_context=arguments.kubernetes_context,
            )
            if arguments.model_id == EXTERNAL_CTF_BASE_CLONE_ID and arguments.action == "resume":
                _need(arguments.result is not None, "external_ctf_lifecycle_result_required")
                _write_once(arguments.result.resolve(), value)
        else:
            value = wait_phase(client, arguments.model_id, arguments.phase, arguments.timeout)
        print(json.dumps(value, sort_keys=True))
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {"command": arguments.command, "error_type": type(exc).__name__, "error": str(exc)},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
