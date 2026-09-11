"""Validate nonlaunchable Qwen base-route component evidence.

V2 excludes Kubernetes ``resourceVersion`` from scientific identity because it
changes on status-only writes.  The current value is still required as a fresh
observation.  This module cannot issue a final certificate; simultaneous
base/post parity and a complete agent-image qualification remain separate gates.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

from evals.fleet import base_control_certification as v1
from training.io import digest_json, file_sha256

PLAN_SCHEMA = "cyber_fleet_base_control_certification_plan_v2"
ROUTE_SCHEMA = "cyber_fleet_base_route_component_evidence_v2"
PAYLOAD_SCHEMA = "cyber_fleet_base_payload_readback_v1"
PROXY_SCHEMA = "cyber_fleet_fixed_proxy_cpu_qualification_v1"
HARNESS_PARTIAL_SCHEMA = "cyber_fleet_harness_component_evidence_v2"
CERTIFICATE_SCHEMA = "cyber_fleet_base_control_certificate_v2"
PUBLICATION_SCHEMA = "cyber_opencode_agent_image_publication_plan_v1"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain one object")
    return value


def _seal(value: dict[str, Any], schema: str) -> str:
    unsigned = {key: item for key, item in value.items() if key != "sha256"}
    expected = digest_json(unsigned)
    if value.get("schema") != schema or value.get("sha256") != expected:
        raise ValueError(f"invalid {schema} seal")
    return expected


def _time(value: Any) -> None:
    if not isinstance(value, str):
        raise ValueError("expected a timezone-qualified timestamp")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("expected a timezone-qualified timestamp")


def _load_v1(plan: dict[str, Any], root: Path) -> dict[str, Any]:
    reference = plan.get("supersedes")
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256", "file_sha256"}:
        raise ValueError("v2 must bind one exact v1 plan")
    path = root / reference["path"]
    old = _read(path)
    if reference["file_sha256"] != file_sha256(path) or reference["sha256"] != old.get("sha256"):
        raise ValueError("v1 plan bytes or logical digest changed")
    v1.validate_plan(old, root=root)
    return old


def _stable_candidate(old: dict[str, Any]) -> dict[str, Any]:
    candidate = deepcopy(old["candidate_base_route"])
    candidate["object_identity"].pop("inference_model_resource_version")
    return candidate


def validate_plan(plan: dict[str, Any], *, root: Path) -> None:
    _seal(plan, PLAN_SCHEMA)
    if set(plan) != {
        "schema",
        "purpose",
        "split_variant",
        "supersedes",
        "identity_policy",
        "stable_candidate_base_route",
        "component_schemas",
        "final_gate",
        "launchable",
        "paid_or_scored_work_authorized",
        "sha256",
    }:
        raise ValueError("v2 certification plan contains unknown or missing fields")
    old = _load_v1(plan, root)
    policy = plan.get("identity_policy")
    if policy != {
        "stable_fields": [
            "inference_model_uid",
            "inference_model_generation",
            "deployment_uid",
            "deployment_generation",
            "pod_uid",
            "service_uid",
            "endpoint_slice_uid",
        ],
        "status_only_observation_fields": ["inference_model_resource_version"],
        "resource_version_is_scientific_identity": False,
        "fresh_resource_version_observation_required": True,
    }:
        raise ValueError("v2 status-versus-identity policy changed")
    if (
        plan.get("split_variant") != "a"
        or plan.get("stable_candidate_base_route") != _stable_candidate(old)
        or plan.get("component_schemas")
        != {
            "base_payload": PAYLOAD_SCHEMA,
            "base_route": ROUTE_SCHEMA,
            "harness_partial": HARNESS_PARTIAL_SCHEMA,
        }
        or plan.get("final_gate")
        != {
            "certificate_schema": CERTIFICATE_SCHEMA,
            "requires_complete_harness_receipt": v1.HARNESS_SCHEMA,
            "requires_simultaneous_live_pair_receipt": (
                "cyber_fleet_base_post_live_parity_receipt_v2"
            ),
            "base_and_post_runtime_must_match_except_weights": True,
            "certificate_does_not_authorize_scored_work": True,
        }
        or plan.get("launchable") is not False
        or plan.get("paid_or_scored_work_authorized") is not False
    ):
        raise ValueError("v2 plan opened or changed a scientific gate")


def validate_route_component(
    plan: dict[str, Any],
    payload: dict[str, Any],
    route: dict[str, Any],
    *,
    root: Path,
) -> None:
    validate_plan(plan, root=root)
    _seal(payload, PAYLOAD_SCHEMA)
    _seal(route, ROUTE_SCHEMA)
    if set(route) != {
        "schema",
        "status",
        "plan_sha256",
        "observed_at",
        "payload_receipt_sha256",
        "model_artifact",
        "registration",
        "readiness",
        "gateway_projections",
        "payload_execution",
        "scope",
        "launchable",
        "remaining_gates",
        "sha256",
    }:
        raise ValueError("route component contains unknown or missing fields")
    old = _load_v1(plan, root)
    if payload.get("plan_sha256") != old["sha256"]:
        raise ValueError("payload readback does not bind the superseded v1 plan")
    if route.get("status") != "passed_component_only" or route.get("plan_sha256") != plan["sha256"]:
        raise ValueError("route component does not bind this v2 plan")
    _time(route.get("observed_at"))
    if route.get("payload_receipt_sha256") != payload["sha256"]:
        raise ValueError("route component does not bind the payload readback")
    candidate = plan["stable_candidate_base_route"]
    registration = route.get("registration")
    if (
        not isinstance(registration, dict)
        or set(registration) != {"stable_route", "stable_route_sha256", "status_observation"}
        or registration.get("stable_route") != candidate
        or registration.get("stable_route_sha256") != digest_json(candidate)
    ):
        raise ValueError("current base registration differs from the stable candidate")
    status = registration.get("status_observation")
    if (
        not isinstance(status, dict)
        or not isinstance(status.get("inference_model_resource_version"), str)
        or not status["inference_model_resource_version"].isdigit()
        or status.get("inference_model_observed_generation")
        != candidate["object_identity"]["inference_model_generation"]
        or status.get("deployment_observed_generation")
        != candidate["object_identity"]["deployment_generation"]
        or status.get("inference_model_phase") != "ready"
        or status.get("inference_model_ready_replicas") != 1
        or status.get("deployment_ready_replicas") != 1
        or status.get("pod_phase") != "Running"
        or status.get("pod_ready") is not True
        or status.get("pod_restart_count") != 0
        or status.get("ready_endpoint_count") != 1
    ):
        raise ValueError("base route status observation is stale or incomplete")
    artifact = route.get("model_artifact")
    if artifact != {
        "repository": payload["model"]["repository"],
        "revision": payload["model"]["revision"],
        "base_snapshot_manifest_sha256": payload["payload"]["manifest_sha256"],
        "weights_manifest_sha256": payload["model"]["weights_manifest_sha256"],
        "tokenizer_manifest_sha256": payload["model"]["tokenizer_manifest_sha256"],
        "chat_template_sha256": payload["model"]["chat_template_sha256"],
        "payload_rehashed": True,
        "symlinks_absent": True,
    }:
        raise ValueError("route artifact differs from the measured payload")
    readiness = route.get("readiness")
    if readiness != {
        "catalog_status": "ready",
        "catalog_ready_replicas": 1,
        "catalog_routed": True,
        "catalog_revision": candidate["model"]["revision"],
        "models_listed_once": True,
        "model_info_http_status": 200,
        "model_info_repository_type": "qwen3_5",
        "model_info_architecture": "Qwen3_5ForConditionalGeneration",
        "server_info_http_status": 200,
        "server_info_runtime_matches": True,
    }:
        raise ValueError("GET-only base readiness is incomplete")
    root_path = f"/scratch/models/qwen3.8-27b/{candidate['model']['revision']}"
    if route.get("gateway_projections") != {
        "catalog": {
            "id": candidate["served_model_id"],
            "model_revision": candidate["model"]["revision"],
            "engine": candidate["runtime"]["engine"],
            "precision": candidate["runtime"]["precision"],
            "tensor_parallel_size": 8,
            "data_parallel_size": 1,
            "status": "ready",
            "ready_replicas": 1,
            "routed": True,
            "capabilities": ["chat_completions", "reasoning", "streaming", "tool_calling"],
        },
        "models": {"id": candidate["served_model_id"], "object": "model", "match_count": 1},
        "model_info": {
            "model_path": root_path,
            "model_type": "qwen3_5",
            "architectures": ["Qwen3_5ForConditionalGeneration"],
        },
        "server_info": {
            "model_path": root_path,
            "served_model_name": candidate["served_model_id"],
            "context_length": candidate["runtime"]["context_length"],
            "tp_size": candidate["runtime"]["tensor_parallel_size"],
            "dp_size": candidate["runtime"]["data_parallel_size"],
            "load_balance_method": "total_tokens",
            "quantization": None,
            "kv_cache_dtype": candidate["runtime"]["kv_cache_dtype"],
            "reasoning_parser": candidate["runtime"]["reasoning_parser"],
            "tool_call_parser": candidate["runtime"]["tool_call_parser"],
        },
    }:
        raise ValueError("gateway GET projections differ from the exact route")
    execution = route.get("payload_execution")
    execution_image = execution.get("image_id") if isinstance(execution, dict) else None
    if (
        not isinstance(execution, dict)
        or execution.get("server_dry_run_passed") is not True
        or execution.get("pod_uid") != payload["helper"]["pod_uid"]
        or execution.get("terminal_phase") != "Succeeded"
        or execution.get("exit_code") != 0
        or execution.get("restart_count") != 0
        or execution.get("gpu_requests") != 0
        or not isinstance(execution_image, str)
        or execution_image.split("@", 1)[-1]
        != candidate["runtime"]["serving_image_digest"]
        or execution.get("deleted") is not True
        or execution.get("absence_confirmed") is not True
    ):
        raise ValueError("payload helper execution or cleanup is incomplete")
    scope = route.get("scope")
    if scope != {
        "gateway_gets": 4,
        "cluster_reads": True,
        "prompt_requests": 0,
        "completion_requests": 0,
        "scoring_requests": 0,
        "task_or_grading_requests": 0,
        "evaluation_submissions": 0,
        "gpu_mutations": 0,
    }:
        raise ValueError("route component exceeded read-only certification scope")
    if route.get("launchable") is not False or route.get("remaining_gates") != [
        "complete_agent_image_qualification",
        "accepted_post_sft_checkpoint_export_staging_and_serving",
        "simultaneous_base_post_live_parity",
        "final_v2_certificate",
    ]:
        raise ValueError("component evidence was presented as a launch certificate")


def validate_harness_partial(
    plan: dict[str, Any], proxy: dict[str, Any], harness: dict[str, Any], *, root: Path
) -> None:
    validate_plan(plan, root=root)
    _seal(proxy, PROXY_SCHEMA)
    _seal(harness, HARNESS_PARTIAL_SCHEMA)
    old = _load_v1(plan, root)
    if proxy.get("plan_sha256") != old["sha256"] or not all(proxy.get("checks", {}).values()):
        raise ValueError("proxy qualification does not bind the exact v1 treatment")
    if set(harness) != {
        "schema",
        "status",
        "plan_sha256",
        "observed_at",
        "proxy_receipt_sha256",
        "fixed_proxy",
        "agent",
        "full_harness_receipt_present",
        "scope",
        "launchable",
        "sha256",
    }:
        raise ValueError("harness partial evidence contains unknown or missing fields")
    if (
        harness.get("status") != "partial_blocked"
        or harness.get("plan_sha256") != plan["sha256"]
        or harness.get("proxy_receipt_sha256") != proxy["sha256"]
        or harness.get("full_harness_receipt_present") is not False
        or harness.get("launchable") is not False
    ):
        raise ValueError("harness partial evidence is not safely blocked")
    _time(harness.get("observed_at"))
    fixed = harness.get("fixed_proxy")
    observed_image = fixed.get("observed_image_id") if isinstance(fixed, dict) else None
    if (
        not isinstance(fixed, dict)
        or fixed.get("requested_image") != proxy["requested_image"]
        or not isinstance(observed_image, str)
        or observed_image.split("@", 1)[-1]
        != proxy["requested_image"].split("@", 1)[-1]
        or fixed.get("pod_uid") != proxy["helper"]["pod_uid"]
        or fixed.get("proxy_source_sha256") != proxy["proxy_source_sha256"]
        or fixed.get("server_dry_run_passed") is not True
        or fixed.get("fresh_pull") is not True
        or fixed.get("terminal_phase") != "Succeeded"
        or fixed.get("exit_code") != 0
        or fixed.get("restart_count") != 0
        or fixed.get("gpu_requests") != 0
        or fixed.get("deleted") is not True
        or fixed.get("absence_confirmed") is not True
    ):
        raise ValueError("fixed-proxy pull, execution, or cleanup is incomplete")
    agent = harness.get("agent")
    if (
        not isinstance(agent, dict)
        or agent.get("status") != "blocked_no_pullable_immutable_reference"
        or agent.get("dev_pod_created") is not False
        or agent.get("registry_mutated") is not False
        or agent.get("image_build_job_created") is not False
    ):
        raise ValueError("agent-image blocker is missing or was bypassed")
    reference = agent.get("publication_plan")
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256", "file_sha256"}:
        raise ValueError("agent publication plan reference is incomplete")
    publication_path = root / reference["path"]
    publication = _read(publication_path)
    if (
        reference["file_sha256"] != file_sha256(publication_path)
        or reference["sha256"] != publication.get("sha256")
    ):
        raise ValueError("agent publication plan bytes or digest changed")
    validate_agent_publication_plan(publication, root=root)
    if harness.get("scope") != {
        "gpu_requests": 0,
        "model_requests": 0,
        "prompt_requests": 0,
        "completion_requests": 0,
        "scoring_requests": 0,
        "task_or_grading_requests": 0,
        "evaluation_submissions": 0,
    }:
        raise ValueError("harness component exceeded offline qualification scope")


def validate_agent_publication_plan(plan: dict[str, Any], *, root: Path) -> None:
    _seal(plan, PUBLICATION_SCHEMA)
    if set(plan) != {
        "schema",
        "status",
        "purpose",
        "source",
        "dev_builder",
        "historical_routes",
        "required_preconditions",
        "forbidden_shortcuts",
        "authorized_mutations",
        "launchable",
        "sha256",
    }:
        raise ValueError("agent publication plan contains unknown or missing fields")
    source = plan.get("source", {})
    dockerfile = root / source.get("dockerfile", "")
    local = source.get("local_image", {})
    builder = plan.get("dev_builder", {})
    if (
        plan.get("status") != "blocked_preparation_only"
        or source.get("dockerfile_sha256") != file_sha256(dockerfile)
        or source.get("opencode_version") != "1.18.27"
        or source.get("release_asset_sha256")
        != "sha256:4af5494f9433f59db8c1e344198f0ee72a50c06ec009fb4a8aeab4c2d4abd702"
        or local.get("pullable_registry_reference_proven") is not False
        or source.get("rebuild_byte_reproducible") is not False
        or builder.get("repository_exists_and_builder_iam_confirmed") is not False
        or builder.get("must_not_repurpose_trainer_repository") is not True
        or plan.get("authorized_mutations")
        != {
            "registry_publication": False,
            "image_build_job": False,
            "agent_qualification_pod": False,
            "evaluation_submission": False,
        }
        or plan.get("launchable") is not False
    ):
        raise ValueError("blocked agent publication plan was opened or changed")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    plan_command = sub.add_parser("validate-plan")
    plan_command.add_argument("plan", type=Path)
    route_command = sub.add_parser("validate-route-component")
    route_command.add_argument("plan", type=Path)
    route_command.add_argument("--payload", type=Path, required=True)
    route_command.add_argument("--route", type=Path, required=True)
    harness_command = sub.add_parser("validate-harness-partial")
    harness_command.add_argument("plan", type=Path)
    harness_command.add_argument("--proxy", type=Path, required=True)
    harness_command.add_argument("--harness", type=Path, required=True)
    publication_command = sub.add_parser("validate-agent-publication-plan")
    publication_command.add_argument("plan", type=Path)
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    plan = _read(args.plan)
    if args.command == "validate-agent-publication-plan":
        validate_agent_publication_plan(plan, root=root)
    elif args.command == "validate-plan":
        validate_plan(plan, root=root)
    elif args.command == "validate-route-component":
        validate_route_component(plan, _read(args.payload), _read(args.route), root=root)
    else:
        validate_harness_partial(plan, _read(args.proxy), _read(args.harness), root=root)
    print(json.dumps({"launchable": False, "plan_sha256": plan["sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
