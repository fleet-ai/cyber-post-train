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
from pathlib import Path
from typing import Any

API = "https://inference.flt.build/fleet/v1/models"
ACCOUNT = "https://orchestrator.fleetai.com/v1/account"
TEAM_ID = "a1025f0b-ad67-49fc-a023-51800ab43e84"
_MODEL_ID = re.compile(r"[a-z0-9](?:[-a-z0-9]*[a-z0-9])?")
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}")


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
) -> dict[str, Any]:
    _need(_MODEL_ID.fullmatch(model_id), "invalid_model_id")
    _need(source_path.startswith("/models/") and ".." not in source_path, "invalid_source_path")
    _need(_SHA256.fullmatch(revision), "invalid_revision")
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
        "registration": registration,
        "registration_sha256": _digest(registration),
        "normalized_contract_sha256": _digest(_normalized_contract(spec)),
        "mutation_count": 0,
    }
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
    client: Client, plan_path: Path, intent_path: Path, result_path: Path
) -> dict[str, Any]:
    plan = _read(plan_path)
    _validate_plan(plan)
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
    client: Client, plan_path: Path, intent_path: Path, result_path: Path
) -> dict[str, Any]:
    """Reconcile an attempted create without issuing another POST."""

    plan = _read(plan_path)
    _validate_plan(plan)
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


def lifecycle(client: Client, model_id: str, action: str) -> dict[str, Any]:
    _need(action in {"resume", "pause"}, "invalid_lifecycle_action")
    url = API + "/" + urllib.parse.quote(model_id, safe="")
    _, before = client.request("GET", url)
    before = _mapping(before, "invalid_lifecycle_readback")
    status = _mapping(before.get("status"), "invalid_lifecycle_status")
    expected = "paused" if action == "resume" else "ready"
    _need(status.get("phase") == expected, f"route_not_{expected}")
    resource_version = str(before.get("resource_version") or "")
    _need(bool(resource_version), "missing_resource_version")
    _, accepted = client.request(
        "POST",
        url + "/" + action,
        {},
        if_match=resource_version,
    )
    accepted = _mapping(accepted, "invalid_lifecycle_acceptance")
    return {
        "model_id": model_id,
        "action": action,
        "before_phase": expected,
        "before_resource_version": resource_version,
        "accepted_phase": _mapping(accepted.get("status", {}), "invalid_accepted_status").get(
            "phase"
        ),
        "accepted_resource_version": str(accepted.get("resource_version") or ""),
    }


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
    prepare_parser.add_argument("--output", type=Path, required=True)
    execute_parser = commands.add_parser("execute")
    execute_parser.add_argument("--plan", type=Path, required=True)
    execute_parser.add_argument("--intent", type=Path, required=True)
    execute_parser.add_argument("--result", type=Path, required=True)
    reconcile_parser = commands.add_parser("reconcile-create")
    reconcile_parser.add_argument("--plan", type=Path, required=True)
    reconcile_parser.add_argument("--intent", type=Path, required=True)
    reconcile_parser.add_argument("--result", type=Path, required=True)
    lifecycle_parser = commands.add_parser("lifecycle")
    lifecycle_parser.add_argument("--model-id", required=True)
    lifecycle_parser.add_argument("--action", choices=("resume", "pause"), required=True)
    wait_parser = commands.add_parser("wait")
    wait_parser.add_argument("--model-id", required=True)
    wait_parser.add_argument("--phase", choices=("ready", "paused"), required=True)
    wait_parser.add_argument("--timeout", type=int, default=1800)
    arguments = parser.parse_args(argv)
    try:
        client = Client(os.environ.get("FLEET_API_KEY", ""))
        if arguments.command == "prepare":
            value = prepare(
                client,
                base_model_id=arguments.base_model_id,
                source_model_id=arguments.source_model_id,
                target_model_id=arguments.target_model_id,
                display_name=arguments.display_name,
                source_path=arguments.source_path,
                revision=arguments.revision,
            )
            _write_once(arguments.output.resolve(), value)
        elif arguments.command == "execute":
            value = execute(
                client,
                arguments.plan.resolve(),
                arguments.intent.resolve(),
                arguments.result.resolve(),
            )
        elif arguments.command == "reconcile-create":
            value = reconcile_create(
                client,
                arguments.plan.resolve(),
                arguments.intent.resolve(),
                arguments.result.resolve(),
            )
        elif arguments.command == "lifecycle":
            value = lifecycle(client, arguments.model_id, arguments.action)
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
