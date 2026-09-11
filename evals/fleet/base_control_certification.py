"""Offline certification gate for the matched Qwen Fleet-dev base control.

The module validates sanitized, self-digesting receipts.  It never contacts a
model, opens an evaluation outcome, or submits work.  A certificate is emitted
only after current base bytes, the exact harness images and a simultaneous
base/post live-parity receipt all agree with the frozen split-A protocol.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from evals.fleet import dev_outcome_protocol as fleet
from evals.fleet import evaluate
from evals.fleet.opencode_self_hosted import write_json_once
from training.io import canonical_json, digest_json, file_sha256

PLAN_SCHEMA = "cyber_fleet_base_control_certification_plan_v1"
BASE_SCHEMA = "cyber_fleet_existing_base_route_receipt_v1"
HARNESS_SCHEMA = "cyber_fleet_harness_qualification_receipt_v1"
PAIR_SCHEMA = "cyber_fleet_base_post_live_parity_receipt_v1"
CERTIFICATE_SCHEMA = "cyber_fleet_base_control_certificate_v1"
SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
IMAGE = re.compile(r"^[^\s]+@sha256:[0-9a-f]{64}$")
POST_FIELDS = tuple(
    field for field in fleet.UNBOUND_CHECKPOINT_FIELDS if field != "live_parity_receipt_sha256"
)
RUNTIME_FIELDS = fleet.MATCHED_SERVING_FIELDS


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain one object")
    return value


def _check_seal(value: dict[str, Any], schema: str) -> str:
    unsigned = {key: item for key, item in value.items() if key != "sha256"}
    expected = digest_json(unsigned)
    if value.get("schema") != schema or value.get("sha256") != expected:
        raise ValueError(f"invalid {schema} seal")
    return expected


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be an exact SHA-256")
    return value


def _time(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a timezone-qualified timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{label} must be a timezone-qualified timestamp") from error
    if parsed.tzinfo is None:
        raise ValueError(f"{label} must be a timezone-qualified timestamp")
    return value


def _repo_reference(value: Any, root: Path, *, sealed: bool) -> dict[str, Any]:
    expected = {"path", "file_sha256"} | ({"sha256"} if sealed else set())
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("certification references require exact path and digests")
    path = Path(value["path"])
    if path.is_absolute():
        raise ValueError("certification reference must be repository-relative")
    resolved = (root / path).resolve()
    if root.resolve() not in resolved.parents:
        raise ValueError("certification reference escapes the repository")
    loaded = _read(resolved)
    if value["file_sha256"] != file_sha256(resolved):
        raise ValueError("certification reference file bytes changed")
    if sealed and value["sha256"] != loaded.get("sha256"):
        raise ValueError("certification reference logical digest changed")
    return loaded


def _candidate(audit: dict[str, Any], model_lock: dict[str, Any]) -> dict[str, Any]:
    shared = audit["routes"]["shared"]
    return {
        "endpoint_origin": "https://inference.flt.build",
        "served_model_id": shared["served_model_id"],
        "serving_block_kind": shared["serving_block_kind"],
        "model": {
            "repository": model_lock["repo"],
            "revision": model_lock["revision"],
        },
        "cluster": audit["cluster"],
        "object_identity": {
            "inference_model_uid": shared["inference_model"]["uid"],
            "inference_model_resource_version": shared["inference_model"]["resource_version"],
            "inference_model_generation": shared["inference_model"]["generation"],
            "deployment_uid": shared["deployment"]["uid"],
            "deployment_generation": shared["deployment"]["generation"],
            "pod_uid": shared["pod"]["uid"],
            "service_uid": shared["service"]["uid"],
            "endpoint_slice_uid": shared["service"]["endpoint_slice_uid"],
        },
        "runtime": {
            "engine": shared["runtime"]["engine"],
            "precision": shared["runtime"]["precision"],
            "quantization": shared["runtime"]["quantization"],
            "tensor_parallel_size": shared["runtime"]["runtime_tensor_parallel_size"],
            "data_parallel_size": shared["runtime"]["runtime_data_parallel_size"],
            "context_length": shared["runtime"]["context_length"],
            "kv_cache_dtype": shared["runtime"]["kv_cache_dtype"],
            "reasoning_parser": shared["runtime"]["reasoning_parser"],
            "tool_call_parser": shared["runtime"]["tool_call_parser"],
            "serving_image_digest": shared["runtime"]["serving_image_digest"],
            "normalized_server_arguments_sha256": shared["runtime"][
                "normalized_server_arguments_sha256"
            ],
        },
    }


def validate_plan(plan: dict[str, Any], *, root: Path) -> None:
    """Validate the immutable, deliberately nonlaunchable certification plan."""
    _check_seal(plan, PLAN_SCHEMA)
    if set(plan) != {
        "schema",
        "purpose",
        "split_variant",
        "references",
        "candidate_base_route",
        "harness_runtime",
        "workflow",
        "receipt_templates",
        "final_gate",
        "launchable",
        "paid_or_scored_work_authorized",
        "sha256",
    }:
        raise ValueError("certification plan has unknown or missing fields")
    if (
        plan.get("split_variant") != "a"
        or plan.get("launchable") is not False
        or plan.get("paid_or_scored_work_authorized") is not False
    ):
        raise ValueError("certification plan must remain a nonlaunchable split-A preparation")

    refs = plan.get("references")
    if not isinstance(refs, dict) or set(refs) != {
        "base_control",
        "parent_protocol",
        "task_set",
        "model_lock",
        "route_audit",
    }:
        raise ValueError("certification references are incomplete")
    base = _repo_reference(refs["base_control"], root, sealed=True)
    parent = _repo_reference(refs["parent_protocol"], root, sealed=True)
    tasks = _repo_reference(refs["task_set"], root, sealed=True)
    model_lock = _repo_reference(refs["model_lock"], root, sealed=False)
    audit = _repo_reference(refs["route_audit"], root, sealed=True)
    fleet.validate_base_control(base, parent, tasks)
    if (
        model_lock.get("repo") != fleet.MODEL_REPOSITORY
        or model_lock.get("revision") != fleet.MODEL_REVISION
    ):
        raise ValueError("certification model lock differs from the frozen base")
    _check_seal(audit, "cyber_fleet_dev_base_route_audit_v1")
    audit_scope = audit.get("scope", {})
    shared = audit.get("routes", {}).get("shared", {})
    if (
        audit_scope.get("read_only") is not True
        or any(
            audit_scope.get(key) != 0
            for key in (
                "prompt_or_completion_requests",
                "scoring_requests",
                "cluster_or_api_mutations",
                "evaluation_submissions",
            )
        )
        or shared.get("exact_frozen_revision") is not True
        or shared.get("assessment", {}).get("readiness_and_catalog_identity_passed") is not True
        or audit.get("decision", {}).get("base_control_launchable") is not False
    ):
        raise ValueError("route audit does not prove the expected safe blocked state")
    if plan.get("candidate_base_route") != _candidate(audit, model_lock):
        raise ValueError("candidate base route differs from the read-only audit")

    harness = plan.get("harness_runtime")
    expected_harness = {
        "harness_sha256": digest_json(parent["harness"]),
        "sampling_sha256": digest_json(parent["sampling"]),
        "pass_k": parent["pass_k"],
        "runtime_files": evaluate.runtime_identity(),
        "runtime_files_sha256": digest_json(evaluate.runtime_identity()),
    }
    if harness != expected_harness:
        raise ValueError("Fleet evaluator or v2 harness identity changed")

    workflow = plan.get("workflow")
    if not isinstance(workflow, dict) or [step.get("id") for step in workflow.get("order", [])] != [
        "base_route",
        "harness_images",
        "post_sft_route",
        "live_pair",
        "certificate",
    ]:
        raise ValueError("certification workflow order changed")
    availability = {step.get("id"): step.get("availability") for step in workflow["order"]}
    if availability != {
        "base_route": "available_now",
        "harness_images": "available_now",
        "post_sft_route": "wait_for_accepted_post_sft_export",
        "live_pair": "wait_for_both_routes_live",
        "certificate": "wait_for_all_receipts",
    }:
        raise ValueError("certification workflow availability changed")

    templates = plan.get("receipt_templates")
    if not isinstance(templates, dict) or set(templates) != {
        "base_route",
        "harness_images",
        "live_pair",
    }:
        raise ValueError("certification receipt templates are incomplete")
    expected_schemas = {
        "base_route": BASE_SCHEMA,
        "harness_images": HARNESS_SCHEMA,
        "live_pair": PAIR_SCHEMA,
    }
    for name, schema in expected_schemas.items():
        template = templates[name]
        if template.get("schema") != schema or template.get("status") != "template_not_evidence":
            raise ValueError("a receipt template was presented as evidence")
        if template.get("sha256") is not None:
            raise ValueError("receipt templates must not carry acceptance digests")
    base_template = templates["base_route"]
    if (
        set(base_template)
        != {
            "schema",
            "status",
            "plan_sha256",
            "observed_at",
            "model_artifact",
            "registration",
            "readiness",
            "sha256",
        }
        or any(value is not None for value in base_template["model_artifact"].values())
        or any(value is not None for value in base_template["registration"].values())
        or any(
            value is not None
            for key, value in base_template["readiness"].items()
            if key not in {"prompt_requests", "completion_requests", "scoring_requests"}
        )
        or any(
            base_template["readiness"][key] != 0
            for key in (
                "prompt_requests",
                "completion_requests",
                "scoring_requests",
            )
        )
    ):
        raise ValueError("base receipt template contains evidence or changed fields")
    harness_template = templates["harness_images"]
    if (
        set(harness_template)
        != {
            "schema",
            "status",
            "plan_sha256",
            "qualified_at",
            "platform",
            "agent_image_digest",
            "proxy_image_digest",
            "harness_sha256",
            "sampling_sha256",
            "runtime_files",
            "runtime_files_sha256",
            "checks",
            "sha256",
        }
        or any(value is not None for value in harness_template["checks"].values())
        or any(
            harness_template[key] is not None
            for key in set(harness_template) - {"schema", "status", "checks", "sha256"}
        )
    ):
        raise ValueError("harness receipt template contains evidence or changed fields")
    pair_template = templates["live_pair"]
    if set(pair_template) != {
        "schema",
        "status",
        "plan_sha256",
        "observed_at",
        "base_route_receipt_sha256",
        "harness_receipt_sha256",
        "post_sft",
        "base_runtime",
        "post_runtime",
        "checks",
        "observations",
        "sha256",
    } or any(
        pair_template[key] is not None
        for key in set(pair_template) - {"schema", "status", "sha256"}
    ):
        raise ValueError("live-pair receipt template contains evidence or changed fields")
    gate = plan.get("final_gate")
    if gate != {
        "required_receipts": [BASE_SCHEMA, HARNESS_SCHEMA, PAIR_SCHEMA],
        "output_schema": CERTIFICATE_SCHEMA,
        "only_scientific_difference": "weights_manifest_sha256",
        "base_control_template_remains_immutable": True,
        "certificate_does_not_authorize_scored_work": True,
    }:
        raise ValueError("final certification gate changed")


def _validate_base(plan: dict[str, Any], receipt: dict[str, Any], *, root: Path) -> None:
    _check_seal(receipt, BASE_SCHEMA)
    if set(receipt) != {
        "schema",
        "status",
        "plan_sha256",
        "observed_at",
        "model_artifact",
        "registration",
        "readiness",
        "sha256",
    }:
        raise ValueError("base route receipt contains unknown or missing fields")
    if receipt.get("status") != "passed" or receipt.get("plan_sha256") != plan["sha256"]:
        raise ValueError("base route receipt is not accepted for this plan")
    _time(receipt.get("observed_at"), "base route observed_at")
    artifact = receipt.get("model_artifact")
    lock = _read(root / plan["references"]["model_lock"]["path"])
    expected = {
        "repository": lock["repo"],
        "revision": lock["revision"],
        "weights_manifest_sha256": lock["weights"]["manifest_sha256"],
        "tokenizer_manifest_sha256": lock["tokenizer"]["manifest_sha256"],
        "chat_template_sha256": "sha256:"
        + next(
            row["sha256"]
            for row in lock["tokenizer"]["files"]
            if row["path"] == "chat_template.jinja"
        ),
    }
    if (
        not isinstance(artifact, dict)
        or set(artifact)
        != {
            *expected,
            "base_snapshot_manifest_sha256",
            "staging_receipt_sha256",
            "payload_rehashed",
            "symlinks_absent",
        }
        or any(artifact.get(k) != v for k, v in expected.items())
    ):
        raise ValueError("live base payload differs from the exact model lock")
    for key in ("base_snapshot_manifest_sha256", "staging_receipt_sha256"):
        _sha(artifact.get(key), f"base artifact {key}")
    if artifact.get("payload_rehashed") is not True or artifact.get("symlinks_absent") is not True:
        raise ValueError("base payload was not safely and completely rehashed")

    registration = receipt.get("registration")
    candidate = plan["candidate_base_route"]
    if not isinstance(registration, dict) or set(registration) != {
        "endpoint_origin",
        "served_model_id",
        "serving_block_kind",
        "object_identity",
        "runtime",
        "serving_route_profile_sha256",
        "serving_registration_receipt_sha256",
    }:
        raise ValueError("base registration projection is absent")
    for key in (
        "endpoint_origin",
        "served_model_id",
        "serving_block_kind",
        "object_identity",
        "runtime",
    ):
        if registration.get(key) != candidate[key]:
            raise ValueError(f"base registration {key} differs from the audited route")
    _sha(registration.get("serving_registration_receipt_sha256"), "base registration receipt")
    route = {
        key: registration[key]
        for key in (
            "endpoint_origin",
            "served_model_id",
            "serving_block_kind",
            "object_identity",
            "runtime",
        )
    }
    if registration.get("serving_route_profile_sha256") != digest_json(route):
        raise ValueError("base serving route profile digest differs")
    readiness = receipt.get("readiness")
    if readiness != {
        "catalog_ready": True,
        "routed": True,
        "ready_endpoint_count": 1,
        "model_info_http_status": 200,
        "server_info_http_status": 200,
        "prompt_requests": 0,
        "completion_requests": 0,
        "scoring_requests": 0,
    }:
        raise ValueError("base readiness is incomplete or contains model/scoring requests")


def _validate_harness(plan: dict[str, Any], receipt: dict[str, Any]) -> None:
    _check_seal(receipt, HARNESS_SCHEMA)
    if set(receipt) != {
        "schema",
        "status",
        "plan_sha256",
        "qualified_at",
        "platform",
        "agent_image_digest",
        "proxy_image_digest",
        "harness_sha256",
        "sampling_sha256",
        "runtime_files",
        "runtime_files_sha256",
        "checks",
        "sha256",
    }:
        raise ValueError("harness receipt contains unknown or missing fields")
    if receipt.get("status") != "passed" or receipt.get("plan_sha256") != plan["sha256"]:
        raise ValueError("harness receipt is not accepted for this plan")
    _time(receipt.get("qualified_at"), "harness qualified_at")
    for key in ("agent_image_digest", "proxy_image_digest"):
        if not isinstance(receipt.get(key), str) or IMAGE.fullmatch(receipt[key]) is None:
            raise ValueError(f"{key} must be an immutable image reference")
    expected = plan["harness_runtime"]
    for key in ("harness_sha256", "sampling_sha256", "runtime_files", "runtime_files_sha256"):
        if receipt.get(key) != expected[key]:
            raise ValueError(f"harness {key} differs from the frozen v2 treatment")
    if receipt.get("platform") != "linux/amd64" or receipt.get("checks") != {
        "fresh_digest_pull": True,
        "opencode_release_and_label": True,
        "private_home_startup": True,
        "ordered_bash_submit_report_tools": True,
        "native_compaction_trigger": True,
        "automatic_continuation_after_compaction": True,
        "fixed_sampling_seed_and_32768_output": True,
        "proxy_path_auth_size_and_request_limits": True,
        "zero_prompt_completion_and_scoring_requests": True,
    }:
        raise ValueError("harness images have not passed the exact offline v2 checks")


def _validate_post_fields(value: Any) -> None:
    if not isinstance(value, dict) or set(value) != set(POST_FIELDS):
        raise ValueError("post-SFT receipt binding is incomplete")
    for key, item in value.items():
        if key.endswith("sha256"):
            _sha(item, f"post-SFT {key}")
        elif not isinstance(item, str) or not item:
            raise ValueError(f"post-SFT {key} must be a nonempty identity")


def _validate_pair(
    plan: dict[str, Any],
    receipt: dict[str, Any],
    base: dict[str, Any],
    harness: dict[str, Any],
) -> None:
    _check_seal(receipt, PAIR_SCHEMA)
    if set(receipt) != {
        "schema",
        "status",
        "plan_sha256",
        "observed_at",
        "base_route_receipt_sha256",
        "harness_receipt_sha256",
        "post_sft",
        "base_runtime",
        "post_runtime",
        "checks",
        "observations",
        "sha256",
    }:
        raise ValueError("live-pair receipt contains unknown or missing fields")
    if (
        receipt.get("status") != "passed"
        or receipt.get("plan_sha256") != plan["sha256"]
        or receipt.get("base_route_receipt_sha256") != base["sha256"]
        or receipt.get("harness_receipt_sha256") != harness["sha256"]
    ):
        raise ValueError("live-pair receipt does not bind this exact preparation")
    _time(receipt.get("observed_at"), "live-pair observed_at")
    post = receipt.get("post_sft")
    _validate_post_fields(post)
    artifact = base["model_artifact"]
    if post["weights_manifest_sha256"] == artifact["weights_manifest_sha256"]:
        raise ValueError("post-SFT and base weights must differ")
    for key in ("tokenizer_manifest_sha256", "chat_template_sha256"):
        if post[key] != artifact[key]:
            raise ValueError(f"base/post {key} differs")
    for key in ("agent_image_digest", "proxy_image_digest"):
        if post[key] != harness[key]:
            raise ValueError(f"post-SFT {key} differs from the qualified harness")

    expected_runtime = {
        "endpoint_origin": plan["candidate_base_route"]["endpoint_origin"],
        "serving_block_kind": plan["candidate_base_route"]["serving_block_kind"],
        **plan["candidate_base_route"]["runtime"],
        "tokenizer_manifest_sha256": artifact["tokenizer_manifest_sha256"],
        "chat_template_sha256": artifact["chat_template_sha256"],
        "agent_image_digest": harness["agent_image_digest"],
        "proxy_image_digest": harness["proxy_image_digest"],
    }
    if set(expected_runtime) != set(RUNTIME_FIELDS):
        raise AssertionError("internal matched-runtime projection drifted")
    if (
        receipt.get("base_runtime") != expected_runtime
        or receipt.get("post_runtime") != expected_runtime
    ):
        raise ValueError("base/post serving or harness runtime differs")

    checks = receipt.get("checks")
    if checks != {
        "post_checkpoint_export_accepted": True,
        "post_staging_create_once": True,
        "base_and_post_registration_current": True,
        "tokenizer_encode_decode_match": True,
        "structured_tool_call_both": True,
        "fixed_logit_probe_finite_both": True,
        "fixed_logit_probe_deterministic_within_arm": True,
        "context_compaction_and_autocontinue_both": True,
        "same_runtime_except_weights": True,
    }:
        raise ValueError("live-pair scientific checks are incomplete")
    observations = receipt.get("observations")
    required_digests = {
        "normalized_tool_request_sha256",
        "normalized_logit_request_sha256",
        "tokenizer_probe_sha256",
        "base_response_manifest_sha256",
        "post_response_manifest_sha256",
    }
    if not isinstance(observations, dict) or set(observations) != required_digests | {
        "window_seconds",
        "non_scored_completion_requests",
        "task_instance_creates",
        "grading_requests",
    }:
        raise ValueError("live-pair observation projection is incomplete")
    for key in required_digests:
        _sha(observations[key], f"live-pair {key}")
    if (
        type(observations["window_seconds"]) is not int
        or not 0 < observations["window_seconds"] <= 900
        or type(observations["non_scored_completion_requests"]) is not int
        or observations["non_scored_completion_requests"] < 1
        or observations["task_instance_creates"] != 0
        or observations["grading_requests"] != 0
    ):
        raise ValueError("live parity was not fresh, non-scored and task-free")


def assemble_certificate(
    plan: dict[str, Any],
    base: dict[str, Any],
    harness: dict[str, Any],
    pair: dict[str, Any],
    *,
    root: Path,
) -> dict[str, Any]:
    """Return a launch-identity certificate; never authorize or launch scored work."""
    validate_plan(plan, root=root)
    _validate_base(plan, base, root=root)
    _validate_harness(plan, harness)
    _validate_pair(plan, pair, base, harness)
    artifact = base["model_artifact"]
    registration = base["registration"]
    pair_sha256 = pair["sha256"]
    base_fields = {
        "base_snapshot_manifest_sha256": artifact["base_snapshot_manifest_sha256"],
        "weights_manifest_sha256": artifact["weights_manifest_sha256"],
        "tokenizer_manifest_sha256": artifact["tokenizer_manifest_sha256"],
        "chat_template_sha256": artifact["chat_template_sha256"],
        "serving_image_digest": registration["runtime"]["serving_image_digest"],
        "normalized_server_arguments_sha256": registration["runtime"][
            "normalized_server_arguments_sha256"
        ],
        "staging_receipt_sha256": artifact["staging_receipt_sha256"],
        "serving_registration_receipt_sha256": registration["serving_registration_receipt_sha256"],
        "live_pair_parity_receipt_sha256": pair_sha256,
        "served_model_id": registration["served_model_id"],
        "serving_route_profile_sha256": registration["serving_route_profile_sha256"],
        "serving_block_kind": registration["serving_block_kind"],
        "agent_image_digest": harness["agent_image_digest"],
        "proxy_image_digest": harness["proxy_image_digest"],
    }
    post_fields = {**pair["post_sft"], "live_parity_receipt_sha256": pair_sha256}
    matched = {**pair["base_runtime"], "live_pair_parity_receipt_sha256": pair_sha256}
    certificate = {
        "schema": CERTIFICATE_SCHEMA,
        "plan_sha256": plan["sha256"],
        "parent_protocol_sha256": plan["references"]["parent_protocol"]["sha256"],
        "base_control_sha256": plan["references"]["base_control"]["sha256"],
        "input_receipts": {
            "base_route": base["sha256"],
            "harness_images": harness["sha256"],
            "live_pair": pair_sha256,
        },
        "base_serving_binding": {
            "state": "bound",
            "launchable": True,
            "fields": base_fields,
        },
        "post_sft_binding": post_fields,
        "matched_runtime": matched,
        "only_scientific_difference": "weights_manifest_sha256",
        "eligible_scope": "split-A matched child using this exact post-SFT pair",
        "paid_or_scored_work_authorized": False,
    }
    return {**certificate, "sha256": digest_json(certificate)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate-plan")
    validate.add_argument("plan", type=Path)
    certify = sub.add_parser("certify")
    certify.add_argument("plan", type=Path)
    certify.add_argument("--base-route", type=Path, required=True)
    certify.add_argument("--harness-images", type=Path, required=True)
    certify.add_argument("--live-pair", type=Path, required=True)
    certify.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    plan = _read(args.plan)
    if args.command == "validate-plan":
        validate_plan(plan, root=root)
        print(canonical_json({"launchable": False, "plan_sha256": plan["sha256"]}))
        return 0
    certificate = assemble_certificate(
        plan,
        _read(args.base_route),
        _read(args.harness_images),
        _read(args.live_pair),
        root=root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json_once(args.output, certificate)
    print(canonical_json({"launchable": True, "certificate_sha256": certificate["sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
