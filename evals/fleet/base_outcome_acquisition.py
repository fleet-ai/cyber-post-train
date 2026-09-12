"""Seal split-A base outcomes before a post-SFT checkpoint exists.

This module is deliberately offline.  It validates sanitized receipts and
builds create-once scientific-control artifacts; it has no HTTP, Kubernetes,
database, model, grader, or job-submission path.

The base acquisition is not itself a matched post-training comparison.  A
later certificate must bind the acquired result manifest to a fresh live
base/post parity receipt and an ordinary launchable Fleet-dev study child.
"""

from __future__ import annotations

import argparse
import json
import re
import stat
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from cyber_post_train.jobs import digest as raw_digest
from evals import study_sealing
from evals.fleet import base_control_certification as certification_v1
from evals.fleet import base_control_certification_v2 as certification_v2
from evals.fleet import dev_outcome_protocol as fleet
from evals.fleet import evaluate, opencode_image_context
from evals.fleet.opencode_self_hosted import write_json_once
from training.io import canonical_json, digest_json, file_sha256

PLAN_SCHEMA = "cyber_fleet_base_outcome_acquisition_plan_v1"
ACQUISITION_CERTIFICATE_SCHEMA = "cyber_fleet_base_outcome_acquisition_certificate_v1"
PREFLIGHT_SCHEMA = "cyber_fleet_base_outcome_acquisition_preflight_v1"
TRANSPORT_SCHEMA = "cyber_fleet_eval_v1_exact_binding_transport_v1"
CHILD_SCHEMA = "cyber_fleet_base_outcome_acquisition_child_v1"
PRIVATE_COMPLETION_SCHEMA = "cyber_fleet_base_outcome_private_completion_v1"
BASE_OUTCOME_SEAL_SCHEMA = "cyber_fleet_base_outcome_seal_v1"
PAIR_RECEIPT_SCHEMA = "cyber_fleet_base_post_live_parity_receipt_v3"
MATCHED_PAIR_CERTIFICATE_SCHEMA = "cyber_fleet_acquired_base_matched_pair_certificate_v1"
AGENT_IMAGE_QUALIFICATION_SCHEMA = "cyber_opencode_agent_image_qualification_receipt_v1"

SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
IMAGE = re.compile(r"^[^\s]+@sha256:[0-9a-f]{64}$")
NAME = re.compile(r"^[a-z0-9][a-z0-9-]{2,63}$")

ACQUISITION_BASE_FIELDS = tuple(
    field
    for field in fleet.UNBOUND_BASE_SERVING_FIELDS
    if field
    not in {
        "staging_receipt_sha256",
        "serving_registration_receipt_sha256",
        "live_pair_parity_receipt_sha256",
    }
)
PAIR_BASE_FIELDS = tuple(
    field
    for field in fleet.UNBOUND_BASE_SERVING_FIELDS
    if field != "live_pair_parity_receipt_sha256"
)
PAIR_POST_FIELDS = tuple(
    field for field in fleet.UNBOUND_CHECKPOINT_FIELDS if field != "live_parity_receipt_sha256"
)

RESULT_POLICY = {
    "raw_outcomes": "private_only",
    "wandb_export_allowed": False,
    "training_data_eligible": False,
    "valid_outcome_retry": "forbidden",
    "automatic_retry": False,
    "outcome_access": "only_after_the_predeclared_split_a_arm_set_is_frozen",
}


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain one object")
    return value


def _seal(value: dict[str, Any]) -> dict[str, Any]:
    return {**value, "sha256": digest_json(value)}


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


def _time(value: Any, label: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a timezone-qualified timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{label} must be a timezone-qualified timestamp") from error
    if parsed.tzinfo is None:
        raise ValueError(f"{label} must be a timezone-qualified timestamp")
    return parsed


def _repo_reference(value: Any, root: Path) -> tuple[dict[str, Any], Path]:
    if not isinstance(value, dict) or set(value) != {"path", "sha256", "file_sha256"}:
        raise ValueError("base-acquisition references require path and both digests")
    relative = Path(value["path"])
    if relative.is_absolute():
        raise ValueError("base-acquisition references must be repository-relative")
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise ValueError("base-acquisition reference escapes the repository")
    loaded = _read(path)
    if value["file_sha256"] != file_sha256(path) or value["sha256"] != loaded.get("sha256"):
        raise ValueError("base-acquisition reference identity changed")
    return loaded, path


def _reference(path: Path, root: Path) -> dict[str, str]:
    resolved = path.resolve()
    if root.resolve() not in resolved.parents:
        raise ValueError("base-acquisition reference escapes the repository")
    value = _read(resolved)
    _sha(value.get("sha256"), f"{resolved.name} logical digest")
    return {
        "path": resolved.relative_to(root.resolve()).as_posix(),
        "sha256": value["sha256"],
        "file_sha256": file_sha256(resolved),
    }


def _task_bindings(task_set: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in sorted(task_set["tasks"], key=lambda item: item["task_version_id"]):
        exact = row["exact_binding"]
        rows.append(
            {
                "task_key": row["task_key"],
                "task_version_id": row["task_version_id"],
                "environment_version_id": row["environment_version_id"],
                "exact_binding_sha256": row["exact_binding_sha256"],
                "prompt_sha256": exact["task"]["prompt_sha256"],
                "env_variables_sha256": exact["task"]["env_variables_sha256"],
                "output_json_schema_sha256": exact["task"]["output_json_schema_sha256"],
                "runtime_seed_content_sha256": exact["environment"]["runtime_seed_content_sha256"],
                "verifier_id": exact["verifier"]["id"],
                "verifier_version_id": exact["verifier"]["version_id"],
                "verifier_sha256": exact["verifier"]["sha256"],
            }
        )
    return rows


def _task_seed_design(bindings: list[dict[str, Any]], seeds: list[int]) -> list[dict[str, Any]]:
    return [
        {"task_version_id": row["task_version_id"], "attempt_seed": seed}
        for row in bindings
        for seed in seeds
    ]


def build_plan(
    *,
    parent_path: Path,
    task_set_path: Path,
    base_control_path: Path,
    certification_plan_path: Path,
    publication_plan_path: Path,
    fixed_proxy_path: Path,
    harness_component_path: Path,
    root: Path,
) -> dict[str, Any]:
    """Build the immutable metadata plan; never create an evaluation."""
    parent = _read(parent_path)
    task_set = _read(task_set_path)
    base_control = _read(base_control_path)
    certification_plan = _read(certification_plan_path)
    publication_plan = _read(publication_plan_path)
    fixed_proxy = _read(fixed_proxy_path)
    harness_component = _read(harness_component_path)
    fleet.validate_protocol(parent, task_set)
    fleet.validate_base_control(base_control, parent, task_set)
    certification_v2.validate_plan(certification_plan, root=root)
    opencode_image_context.validate_publication_plan(publication_plan, root=root)
    certification_v2.validate_harness_partial(
        certification_plan, fixed_proxy, harness_component, root=root
    )
    if parent.get("schema") != fleet.PROTOCOL_SCHEMA_V2 or parent.get("split_variant") != "a":
        raise ValueError("base acquisition is only defined for the frozen split-A v2 parent")
    stable = certification_plan["stable_candidate_base_route"]
    if stable.get("model") != {
        "repository": fleet.MODEL_REPOSITORY,
        "revision": fleet.MODEL_REVISION,
    }:
        raise ValueError("base route differs from the frozen Qwen3.8 base")
    bindings = _task_bindings(task_set)
    seeds = list(parent["sampling"]["attempt_seeds"])
    design = _task_seed_design(bindings, seeds)
    return _seal(
        {
            "schema": PLAN_SCHEMA,
            "purpose": (
                "Acquire one reusable, outcome-sealed split-A base vector before any "
                "post-SFT checkpoint exists; later reuse still requires fresh base/post parity"
            ),
            "split_variant": "a",
            "references": {
                "parent_protocol": _reference(parent_path, root),
                "task_set": _reference(task_set_path, root),
                "base_control": _reference(base_control_path, root),
                "base_certification": _reference(certification_plan_path, root),
                "agent_image_publication": _reference(publication_plan_path, root),
                "fixed_proxy_qualification": _reference(fixed_proxy_path, root),
                "harness_component": _reference(harness_component_path, root),
            },
            "frozen_design": {
                "model": {
                    "repository": fleet.MODEL_REPOSITORY,
                    "revision": fleet.MODEL_REVISION,
                    "tokenizer_revision": fleet.MODEL_REVISION,
                },
                "task_count": fleet.DEV_TASKS,
                "task_seed_pair_count": fleet.DEV_TASKS * fleet.PASS_K,
                "task_bindings": bindings,
                "task_binding_set_sha256": digest_json(bindings),
                "attempt_seeds": seeds,
                "task_seed_design_sha256": digest_json(design),
                "harness": parent["harness"],
                "harness_sha256": digest_json(parent["harness"]),
                "sampling": parent["sampling"],
                "sampling_sha256": digest_json(parent["sampling"]),
                "pass_k": fleet.PASS_K,
                "concurrency_per_worker": parent["concurrency_per_worker"],
                "transport_runtime_files": evaluate.runtime_identity(),
                "transport_runtime_files_sha256": digest_json(evaluate.runtime_identity()),
                "agent_image_publication": {
                    "plan_sha256": publication_plan["sha256"],
                    "repository_commit": publication_plan["source"]["repository_commit"],
                    "dockerfile_sha256": publication_plan["source"]["dockerfile_sha256"],
                    "opencode_version": publication_plan["source"]["opencode_version"],
                    "release_asset_sha256": publication_plan["source"]["release_asset_sha256"],
                    "opencode_binary_sha256": publication_plan["source"]["opencode_binary_sha256"],
                    "source_date_epoch": publication_plan["source"]["source_date_epoch"],
                    "context_archive_sha256": publication_plan["deterministic_context"][
                        "archive_sha256"
                    ],
                    "context_receipt_sha256": publication_plan["deterministic_context"][
                        "receipt_sha256"
                    ],
                    "registry_repository": (
                        f"{opencode_image_context.ECR_REGISTRY}/"
                        f"{publication_plan['dev_builder']['proposed_repository']}"
                    ),
                    "tag": publication_plan["exact_future_request"]["tag"],
                    "platform": publication_plan["exact_future_request"]["platform"],
                },
                "fixed_proxy_image_digest": harness_component["fixed_proxy"]["requested_image"],
                "fixed_proxy_receipt_sha256": fixed_proxy["sha256"],
            },
            "acquisition_gate": {
                "base_only": True,
                "post_sft_checkpoint_required": False,
                "live_base_post_parity_required": False,
                "requires_base_payload_and_route_component_v2": True,
                "requires_complete_immutable_harness_qualification": True,
                "requires_exact_agent_publication_and_clean_pull_qualification": True,
                "requires_fresh_exact_binding_preflight": True,
                "generic_transport_without_frozen_binding_extension_forbidden": True,
                "certificate_does_not_authorize_scored_work": True,
            },
            "completion_gate": {
                "all_80_predeclared_pairs_required": True,
                "valid_outcome_count": fleet.DEV_TASKS * fleet.PASS_K,
                "valid_outcomes_are_never_retried": True,
                "infrastructure_invalid_or_ambiguous_attempts_block_completion": True,
                "only_outcome_manifest_digest_may_leave_private_storage": True,
            },
            "reuse_gate": {
                "fresh_live_base_post_parity_required": True,
                "base_acquisition_runtime_must_be_reproved_exactly": True,
                "model_tokenizer_chat_template_harness_tools_and_runtime_must_match": True,
                "only_scientific_difference": "weights_manifest_sha256",
                "later_candidate_must_use_the_same_parent_and_base_control": True,
                "base_outcomes_remain_sealed_until_the_arm_set_is_frozen": True,
            },
            "result_policy": RESULT_POLICY,
            "launchable": False,
            "paid_or_scored_work_authorized": False,
        }
    )


def validate_plan(plan: dict[str, Any], *, root: Path) -> None:
    _check_seal(plan, PLAN_SCHEMA)
    if set(plan) != {
        "schema",
        "purpose",
        "split_variant",
        "references",
        "frozen_design",
        "acquisition_gate",
        "completion_gate",
        "reuse_gate",
        "result_policy",
        "launchable",
        "paid_or_scored_work_authorized",
        "sha256",
    }:
        raise ValueError("base-acquisition plan contains unknown or missing fields")
    refs = plan.get("references")
    if not isinstance(refs, dict) or set(refs) != {
        "parent_protocol",
        "task_set",
        "base_control",
        "base_certification",
        "agent_image_publication",
        "fixed_proxy_qualification",
        "harness_component",
    }:
        raise ValueError("base-acquisition references are incomplete")
    paths = {}
    for name, reference in refs.items():
        _, paths[name] = _repo_reference(reference, root)
    expected = build_plan(
        parent_path=paths["parent_protocol"],
        task_set_path=paths["task_set"],
        base_control_path=paths["base_control"],
        certification_plan_path=paths["base_certification"],
        publication_plan_path=paths["agent_image_publication"],
        fixed_proxy_path=paths["fixed_proxy_qualification"],
        harness_component_path=paths["harness_component"],
        root=root,
    )
    if plan != expected:
        raise ValueError("base-acquisition plan drifted from its frozen split-A inputs")


def _validate_exact_fields(value: Any, fields: tuple[str, ...], label: str) -> None:
    if not isinstance(value, dict) or set(value) != set(fields):
        raise ValueError(f"{label} binding fields are incomplete")
    for key, item in value.items():
        if key.endswith("sha256"):
            _sha(item, f"{label} {key}")
        elif key in {"tensor_parallel_size", "data_parallel_size", "context_length"}:
            if type(item) is not int or item < 1:
                raise ValueError(f"{label} {key} must be a positive integer")
        elif key in {"agent_image_digest", "proxy_image_digest"}:
            if not isinstance(item, str) or IMAGE.fullmatch(item) is None:
                raise ValueError(f"{label} {key} must be an immutable image reference")
        elif key == "serving_image_digest":
            _sha(item, f"{label} {key}")
        elif not isinstance(item, str) or not item:
            raise ValueError(f"{label} {key} must be a nonempty identity")


def _validate_runtime(value: Any) -> None:
    _validate_exact_fields(value, fleet.MATCHED_SERVING_FIELDS, "runtime")
    if value["endpoint_origin"] != "https://inference.flt.build":
        raise ValueError("base acquisition must use the frozen Fleet gateway")
    for key in ("tensor_parallel_size", "data_parallel_size", "context_length"):
        if type(value[key]) is not int or value[key] < 1:
            raise ValueError(f"runtime {key} must be a positive integer")


def _validate_agent_image_qualification(
    plan: dict[str, Any], receipt: dict[str, Any], *, root: Path
) -> None:
    """Require the exact publication and clean-pull proof behind the agent image."""
    _check_seal(receipt, AGENT_IMAGE_QUALIFICATION_SCHEMA)
    if set(receipt) != {
        "schema",
        "status",
        "publication_plan_sha256",
        "qualified_at",
        "agent_image_digest",
        "publication",
        "clean_pull",
        "checks",
        "scope",
        "paid_or_scored_work_authorized",
        "sha256",
    }:
        raise ValueError("agent-image qualification contains unknown or missing fields")
    frozen = plan["frozen_design"]["agent_image_publication"]
    image = receipt.get("agent_image_digest")
    if (
        receipt.get("status") != "passed"
        or receipt.get("publication_plan_sha256") != frozen["plan_sha256"]
        or not isinstance(image, str)
        or IMAGE.fullmatch(image) is None
        or not image.startswith(f"{frozen['registry_repository']}@sha256:")
        or receipt.get("paid_or_scored_work_authorized") is not False
    ):
        raise ValueError("agent image is not the exact qualified publication")
    qualified_at = _time(receipt.get("qualified_at"), "agent image qualified_at")
    publication_plan, _ = _repo_reference(plan["references"]["agent_image_publication"], root)
    if qualified_at < _time(publication_plan["observed_at"], "publication observed_at"):
        raise ValueError("agent-image qualification predates its publication plan")

    publication = receipt.get("publication")
    expected_publication = {
        "repository_commit": frozen["repository_commit"],
        "dockerfile_sha256": frozen["dockerfile_sha256"],
        "release_asset_sha256": frozen["release_asset_sha256"],
        "opencode_binary_sha256": frozen["opencode_binary_sha256"],
        "source_date_epoch": frozen["source_date_epoch"],
        "context_archive_sha256": frozen["context_archive_sha256"],
        "context_receipt_sha256": frozen["context_receipt_sha256"],
        "registry_repository": frozen["registry_repository"],
        "tag": frozen["tag"],
        "rewrite_timestamp": True,
        "terminal_status": "succeeded",
        "image_digest": image,
    }
    if publication != expected_publication:
        raise ValueError("agent image publication differs from the frozen build")

    clean_pull = receipt.get("clean_pull")
    if not isinstance(clean_pull, dict) or set(clean_pull) != {
        "cluster_context",
        "namespace",
        "pod",
        "pod_uid",
        "node",
        "server_dry_run_passed",
        "image_pull_policy",
        "observed_image_id",
        "platform",
        "terminal_phase",
        "exit_code",
        "restart_count",
        "gpu_requests",
        "deleted",
        "absence_confirmed",
    }:
        raise ValueError("agent-image clean-pull proof is incomplete")
    observed_image = clean_pull.get("observed_image_id")
    if (
        clean_pull.get("cluster_context") != publication_plan["dev_builder"]["cluster_context"]
        or any(
            not isinstance(clean_pull.get(key), str) or not clean_pull[key]
            for key in ("namespace", "pod", "pod_uid", "node")
        )
        or clean_pull.get("server_dry_run_passed") is not True
        or clean_pull.get("image_pull_policy") != "Always"
        or not isinstance(observed_image, str)
        or IMAGE.fullmatch(observed_image) is None
        or observed_image.rsplit("@", 1)[-1] != image.rsplit("@", 1)[-1]
        or clean_pull.get("platform") != frozen["platform"]
        or clean_pull.get("terminal_phase") != "Succeeded"
        or clean_pull.get("exit_code") != 0
        or clean_pull.get("restart_count") != 0
        or clean_pull.get("gpu_requests") != 0
        or clean_pull.get("deleted") is not True
        or clean_pull.get("absence_confirmed") is not True
    ):
        raise ValueError("agent image did not pass exact zero-GPU dev clean-pull qualification")
    if receipt.get("checks") != {
        "fresh_digest_pull": True,
        "exact_source_labels": True,
        "opencode_1_18_27": True,
        "private_home": True,
        "non_root_user": True,
        "working_directory": True,
        "ordered_bash_submit_report_tools": True,
        "native_compaction_autocontinue_v1": True,
        "native_compaction_autocontinue_v2": True,
    }:
        raise ValueError("agent image did not pass the complete OpenCode 1.18.27 checks")
    if receipt.get("scope") != {
        "gpu_requests": 0,
        "model_requests": 0,
        "prompt_requests": 0,
        "completion_requests": 0,
        "scoring_requests": 0,
        "task_or_grading_requests": 0,
        "evaluation_submissions": 0,
    }:
        raise ValueError("agent-image qualification exceeded offline scope")


def assemble_acquisition_certificate(
    plan: dict[str, Any],
    payload: dict[str, Any],
    route: dict[str, Any],
    harness: dict[str, Any],
    agent_image_qualification: dict[str, Any],
    *,
    root: Path,
) -> dict[str, Any]:
    """Bind base bytes/runtime and exact harness without requiring a post arm."""
    validate_plan(plan, root=root)
    v2_plan, _ = _repo_reference(plan["references"]["base_certification"], root)
    certification_v2.validate_route_component(v2_plan, payload, route, root=root)
    v1_plan, _ = _repo_reference(v2_plan["supersedes"], root)
    certification_v1._validate_harness(v1_plan, harness)  # noqa: SLF001
    _validate_agent_image_qualification(plan, agent_image_qualification, root=root)
    frozen = plan["frozen_design"]
    if (
        harness["harness_sha256"] != frozen["harness_sha256"]
        or harness["sampling_sha256"] != frozen["sampling_sha256"]
        or harness["runtime_files"] != frozen["transport_runtime_files"]
        or harness["runtime_files_sha256"] != frozen["transport_runtime_files_sha256"]
        or harness["agent_image_digest"] != agent_image_qualification["agent_image_digest"]
        or harness["proxy_image_digest"] != frozen["fixed_proxy_image_digest"]
    ):
        raise ValueError("qualified harness differs from the frozen acquisition treatment")
    artifact = route["model_artifact"]
    stable = route["registration"]["stable_route"]
    base = {
        "base_snapshot_manifest_sha256": artifact["base_snapshot_manifest_sha256"],
        "weights_manifest_sha256": artifact["weights_manifest_sha256"],
        "tokenizer_manifest_sha256": artifact["tokenizer_manifest_sha256"],
        "chat_template_sha256": artifact["chat_template_sha256"],
        "serving_image_digest": stable["runtime"]["serving_image_digest"],
        "normalized_server_arguments_sha256": stable["runtime"][
            "normalized_server_arguments_sha256"
        ],
        "served_model_id": stable["served_model_id"],
        "serving_route_profile_sha256": route["registration"]["stable_route_sha256"],
        "serving_block_kind": stable["serving_block_kind"],
        "agent_image_digest": harness["agent_image_digest"],
        "proxy_image_digest": harness["proxy_image_digest"],
    }
    runtime = {
        "endpoint_origin": stable["endpoint_origin"],
        "serving_block_kind": stable["serving_block_kind"],
        **stable["runtime"],
        "tokenizer_manifest_sha256": artifact["tokenizer_manifest_sha256"],
        "chat_template_sha256": artifact["chat_template_sha256"],
        "agent_image_digest": harness["agent_image_digest"],
        "proxy_image_digest": harness["proxy_image_digest"],
    }
    _validate_exact_fields(base, ACQUISITION_BASE_FIELDS, "base acquisition")
    _validate_runtime(runtime)
    certificate = _seal(
        {
            "schema": ACQUISITION_CERTIFICATE_SCHEMA,
            "plan_sha256": plan["sha256"],
            "parent_protocol_sha256": plan["references"]["parent_protocol"]["sha256"],
            "task_set_sha256": plan["references"]["task_set"]["sha256"],
            "base_control_sha256": plan["references"]["base_control"]["sha256"],
            "input_receipts": {
                "base_payload": payload["sha256"],
                "base_route_component": route["sha256"],
                "fixed_proxy_qualification": frozen["fixed_proxy_receipt_sha256"],
                "agent_image_qualification": agent_image_qualification["sha256"],
                "full_harness": harness["sha256"],
            },
            "task_binding_set_sha256": frozen["task_binding_set_sha256"],
            "task_seed_design_sha256": frozen["task_seed_design_sha256"],
            "task_seed_pair_count": frozen["task_seed_pair_count"],
            "base_binding": base,
            "runtime": runtime,
            "eligible_scope": "one reviewed split-A base-only acquisition child",
            "post_sft_or_live_pair_evidence_included": False,
            "paid_or_scored_work_authorized": False,
        }
    )
    validate_acquisition_certificate(plan, certificate, root=root)
    return certificate


def validate_acquisition_certificate(
    plan: dict[str, Any], certificate: dict[str, Any], *, root: Path
) -> None:
    validate_plan(plan, root=root)
    _check_seal(certificate, ACQUISITION_CERTIFICATE_SCHEMA)
    if set(certificate) != {
        "schema",
        "plan_sha256",
        "parent_protocol_sha256",
        "task_set_sha256",
        "base_control_sha256",
        "input_receipts",
        "task_binding_set_sha256",
        "task_seed_design_sha256",
        "task_seed_pair_count",
        "base_binding",
        "runtime",
        "eligible_scope",
        "post_sft_or_live_pair_evidence_included",
        "paid_or_scored_work_authorized",
        "sha256",
    }:
        raise ValueError("acquisition certificate contains unknown or missing fields")
    frozen = plan["frozen_design"]
    if (
        certificate["plan_sha256"] != plan["sha256"]
        or certificate["parent_protocol_sha256"] != plan["references"]["parent_protocol"]["sha256"]
        or certificate["task_set_sha256"] != plan["references"]["task_set"]["sha256"]
        or certificate["base_control_sha256"] != plan["references"]["base_control"]["sha256"]
        or certificate["task_binding_set_sha256"] != frozen["task_binding_set_sha256"]
        or certificate["task_seed_design_sha256"] != frozen["task_seed_design_sha256"]
        or certificate["task_seed_pair_count"] != fleet.DEV_TASKS * fleet.PASS_K
        or certificate["eligible_scope"] != "one reviewed split-A base-only acquisition child"
        or certificate["post_sft_or_live_pair_evidence_included"] is not False
        or certificate["paid_or_scored_work_authorized"] is not False
    ):
        raise ValueError("acquisition certificate changed scope or frozen design")
    receipts = certificate["input_receipts"]
    if not isinstance(receipts, dict) or set(receipts) != {
        "base_payload",
        "base_route_component",
        "fixed_proxy_qualification",
        "agent_image_qualification",
        "full_harness",
    }:
        raise ValueError("acquisition certificate receipt set is incomplete")
    for name, value in receipts.items():
        _sha(value, f"{name} receipt")
    _validate_exact_fields(certificate["base_binding"], ACQUISITION_BASE_FIELDS, "base")
    _validate_runtime(certificate["runtime"])
    v2_plan, _ = _repo_reference(plan["references"]["base_certification"], root)
    v1_plan, _ = _repo_reference(v2_plan["supersedes"], root)
    lock_path = root / v1_plan["references"]["model_lock"]["path"]
    lock = _read(lock_path)
    if file_sha256(lock_path) != v1_plan["references"]["model_lock"]["file_sha256"]:
        raise ValueError("base model lock bytes changed")
    chat_template = "sha256:" + next(
        row["sha256"] for row in lock["tokenizer"]["files"] if row["path"] == "chat_template.jinja"
    )
    base = certificate["base_binding"]
    stable = v2_plan["stable_candidate_base_route"]
    if (
        base["weights_manifest_sha256"] != lock["weights"]["manifest_sha256"]
        or base["tokenizer_manifest_sha256"] != lock["tokenizer"]["manifest_sha256"]
        or base["chat_template_sha256"] != chat_template
        or base["serving_image_digest"] != stable["runtime"]["serving_image_digest"]
        or base["normalized_server_arguments_sha256"]
        != stable["runtime"]["normalized_server_arguments_sha256"]
        or base["served_model_id"] != stable["served_model_id"]
        or base["serving_route_profile_sha256"] != digest_json(stable)
        or base["serving_block_kind"] != stable["serving_block_kind"]
        or base["proxy_image_digest"] != frozen["fixed_proxy_image_digest"]
        or not base["agent_image_digest"].startswith(
            f"{frozen['agent_image_publication']['registry_repository']}@sha256:"
        )
    ):
        raise ValueError("acquisition certificate base differs from the exact model or route")
    expected_runtime = {
        "endpoint_origin": stable["endpoint_origin"],
        "serving_block_kind": stable["serving_block_kind"],
        **stable["runtime"],
        "tokenizer_manifest_sha256": base["tokenizer_manifest_sha256"],
        "chat_template_sha256": base["chat_template_sha256"],
        "agent_image_digest": base["agent_image_digest"],
        "proxy_image_digest": base["proxy_image_digest"],
    }
    if certificate["runtime"] != expected_runtime:
        raise ValueError("acquisition certificate runtime differs from the exact base route")


def validate_preflight(
    plan: dict[str, Any],
    certificate: dict[str, Any],
    transport: dict[str, Any],
    preflight: dict[str, Any],
    *,
    root: Path,
    now: datetime | None = None,
) -> None:
    validate_acquisition_certificate(plan, certificate, root=root)
    validate_transport(plan, certificate, transport, root=root)
    _check_seal(preflight, PREFLIGHT_SCHEMA)
    if set(preflight) != {
        "schema",
        "status",
        "plan_sha256",
        "acquisition_certificate_sha256",
        "transport_plan_sha256",
        "observed_at",
        "expires_at",
        "task_bindings",
        "task_binding_set_sha256",
        "base_binding",
        "runtime",
        "harness",
        "scope",
        "sha256",
    }:
        raise ValueError("base-acquisition preflight contains unknown or missing fields")
    observed = _time(preflight.get("observed_at"), "preflight observed_at")
    expires = _time(preflight.get("expires_at"), "preflight expires_at")
    if expires <= observed or (expires - observed).total_seconds() > 900:
        raise ValueError("base-acquisition preflight exceeds its 15-minute freshness window")
    if now is not None and not observed <= now.astimezone(UTC) <= expires:
        raise ValueError("base-acquisition preflight is not currently fresh")
    frozen = plan["frozen_design"]
    if (
        preflight["status"] != "passed_read_only"
        or preflight["plan_sha256"] != plan["sha256"]
        or preflight["acquisition_certificate_sha256"] != certificate["sha256"]
        or preflight["transport_plan_sha256"] != transport["sha256"]
        or preflight["task_bindings"] != frozen["task_bindings"]
        or preflight["task_binding_set_sha256"] != frozen["task_binding_set_sha256"]
        or preflight["base_binding"] != certificate["base_binding"]
        or preflight["runtime"] != certificate["runtime"]
        or preflight["harness"]
        != {
            "harness_sha256": frozen["harness_sha256"],
            "sampling_sha256": frozen["sampling_sha256"],
            "agent_image_digest": certificate["base_binding"]["agent_image_digest"],
            "proxy_image_digest": certificate["base_binding"]["proxy_image_digest"],
            "opencode_version": "1.18.27",
            "ordered_tools": ["bash", "submit_report"],
        }
    ):
        raise ValueError("base-acquisition preflight differs from the frozen design")
    if preflight["scope"] != {
        "task_gets": fleet.DEV_TASKS,
        "route_gets_only": True,
        "prompt_requests": 0,
        "completion_requests": 0,
        "task_instance_creates": 0,
        "grading_requests": 0,
        "scoring_requests": 0,
        "scored_sessions": 0,
    }:
        raise ValueError("base-acquisition preflight crossed a scoring or mutation boundary")


def _expected_full_bindings(plan: dict[str, Any], root: Path) -> dict[str, list[dict[str, Any]]]:
    task_set, _ = _repo_reference(plan["references"]["task_set"], root)
    return {
        row["task_version_id"]: [
            row["exact_binding"]["task"],
            row["exact_binding"]["environment"],
            row["exact_binding"]["verifier"],
        ]
        for row in task_set["tasks"]
    }


def _validate_evaluator_plan(
    plan: dict[str, Any], certificate: dict[str, Any], evaluator: dict[str, Any], root: Path
) -> None:
    if evaluator.get("schema") != "cyber_fleet_eval_v1" or evaluator.get("sha256") != raw_digest(
        {key: value for key, value in evaluator.items() if key != "sha256"}
    ):
        raise ValueError("evaluator plan seal is invalid")
    if set(evaluator) != {
        "schema",
        "campaign_id",
        "run_prefix",
        "selection",
        "tasks",
        "models",
        "routes",
        "treatment",
        "images",
        "pass_k",
        "concurrency",
        "training_data_eligible",
        "automatic_retry",
        "runtime_files",
        "sampling",
        "interpretation",
        "sha256",
    }:
        raise ValueError("evaluator plan contains an unreviewed field")
    task_set, _ = _repo_reference(plan["references"]["task_set"], root)
    expected_tasks = [
        {
            key: row[key]
            for key in sorted(
                {
                    "task_key",
                    "task_version_id",
                    "env_key",
                    "env_version",
                    "environment_version_id",
                    "data_key",
                    "data_version",
                }
            )
        }
        for row in task_set["tasks"]
    ]
    expected_tasks.sort(key=lambda row: row["task_version_id"])
    if (
        evaluator["tasks"] != expected_tasks
        or evaluator["selection"] != {"source_job_id": task_set["source_job_id"]}
        or evaluator["treatment"] != plan["frozen_design"]["harness"]
        or evaluator["images"]
        != {
            "agent": certificate["base_binding"]["agent_image_digest"],
            "proxy": certificate["base_binding"]["proxy_image_digest"],
        }
        or evaluator["pass_k"] != fleet.PASS_K
        or evaluator["concurrency"] != plan["frozen_design"]["concurrency_per_worker"]
        or evaluator["training_data_eligible"] is not False
        or evaluator["automatic_retry"] is not False
        or evaluator["runtime_files"] != plan["frozen_design"]["transport_runtime_files"]
        or evaluator["sampling"]
        != {
            "temperature": plan["frozen_design"]["sampling"]["temperature"],
            "top_p": plan["frozen_design"]["sampling"]["top_p"],
            "seed": plan["frozen_design"]["sampling"]["base_seed"],
        }
    ):
        raise ValueError("evaluator plan differs from the frozen 80-cell design")
    models = evaluator["models"]
    routes = evaluator["routes"]
    if (
        not isinstance(models, dict)
        or len(models) != 1
        or not isinstance(routes, dict)
        or len(routes) != 1
    ):
        raise ValueError("base acquisition requires exactly one model and one serving route")
    model_alias, model = next(iter(models.items()))
    route = next(iter(routes.values()))
    runtime = certificate["runtime"]
    if model != {
        "repository": fleet.MODEL_REPOSITORY,
        "revision": fleet.MODEL_REVISION,
        "session_model": certificate["base_binding"]["served_model_id"],
    }:
        raise ValueError("evaluator model differs from the exact Qwen3.8 base")
    expected_versions = sorted(row["task_version_id"] for row in task_set["tasks"])
    server = route.get("server_info") if isinstance(route, dict) else None
    catalog = route.get("catalog") if isinstance(route, dict) else None
    model_path = f"/scratch/models/qwen3.8-27b/{fleet.MODEL_REVISION}"
    if (
        not isinstance(server, dict)
        or not isinstance(catalog, dict)
        or set(route)
        != {
            "model",
            "served_id",
            "task_versions",
            "catalog",
            "model_info",
            "server_info",
            "endpoint_origin",
        }
        or route.get("model") != model_alias
        or route.get("served_id") != certificate["base_binding"]["served_model_id"]
        or sorted(route.get("task_versions") or []) != expected_versions
        or route.get("endpoint_origin") != runtime["endpoint_origin"]
        or catalog
        != {
            "engine": runtime["engine"],
            "precision": runtime["precision"],
            "tensor_parallel_size": runtime["tensor_parallel_size"],
        }
        or route.get("model_info")
        != {
            "model_path": model_path,
            "model_type": "qwen3_5",
            "architectures": ["Qwen3_5ForConditionalGeneration"],
        }
        or server
        != {
            "model_path": model_path,
            "context_length": runtime["context_length"],
            "tp_size": runtime["tensor_parallel_size"],
            "dp_size": runtime["data_parallel_size"],
            "load_balance_method": "total_tokens",
            "quantization": None,
            "kv_cache_dtype": runtime["kv_cache_dtype"],
            "reasoning_parser": runtime["reasoning_parser"],
            "tool_call_parser": runtime["tool_call_parser"],
        }
    ):
        raise ValueError("evaluator serving route differs from the certified base runtime")


def _validate_evaluator_preflight(
    plan: dict[str, Any], evaluator: dict[str, Any], proof: dict[str, Any], root: Path
) -> None:
    if proof.get("schema") != "cyber_fleet_eval_preflight_v1" or proof.get("sha256") != raw_digest(
        {key: value for key, value in proof.items() if key != "sha256"}
    ):
        raise ValueError("evaluator preflight seal is invalid")
    if set(proof) != {
        "schema",
        "plan_sha256",
        "task_bindings",
        "routes",
        "images",
        "sha256",
    }:
        raise ValueError("evaluator preflight contains an unreviewed field")
    expected = _expected_full_bindings(plan, root)
    if (
        proof["plan_sha256"] != evaluator["sha256"]
        or proof["task_bindings"] != expected
        or proof["images"] != evaluator["images"]
        or set(proof["routes"]) != set(evaluator["routes"])
    ):
        raise ValueError("evaluator preflight dropped a frozen prompt/verifier/runtime-seed hash")
    for name, route in proof["routes"].items():
        expected_route = evaluator["routes"][name]
        model = evaluator["models"][expected_route["model"]]
        if route != {
            "served_id": expected_route["served_id"],
            "revision": model["revision"],
            "profile_sha256": raw_digest(expected_route),
            "ready": True,
        }:
            raise ValueError("evaluator preflight route identity differs")


def assemble_transport(
    plan: dict[str, Any],
    certificate: dict[str, Any],
    evaluator: dict[str, Any],
    evaluator_preflight: dict[str, Any],
    *,
    root: Path,
) -> dict[str, Any]:
    """Bind the legacy evaluator to the frozen hashes it otherwise omits."""
    validate_acquisition_certificate(plan, certificate, root=root)
    _validate_evaluator_plan(plan, certificate, evaluator, root)
    _validate_evaluator_preflight(plan, evaluator, evaluator_preflight, root)
    frozen = plan["frozen_design"]
    value = _seal(
        {
            "schema": TRANSPORT_SCHEMA,
            "status": "offline_exact_binding_extension_validated",
            "plan_sha256": plan["sha256"],
            "acquisition_certificate_sha256": certificate["sha256"],
            "evaluator_plan": evaluator,
            "evaluator_preflight": evaluator_preflight,
            "task_bindings": frozen["task_bindings"],
            "task_binding_set_sha256": frozen["task_binding_set_sha256"],
            "task_seed_design_sha256": frozen["task_seed_design_sha256"],
            "base_binding": certificate["base_binding"],
            "runtime": certificate["runtime"],
            "harness": frozen["harness"],
            "execution_contract": {
                "transport": "cyber_fleet_eval_v1",
                "binding_source": "sealed_transport_manifest_only",
                "rehash_before_each_session": True,
                "live_task_snapshot_fallback": False,
                "base_only": True,
                "candidate_sessions": 0,
                "attempt_seeds": frozen["attempt_seeds"],
                "session_count": fleet.DEV_TASKS * fleet.PASS_K,
            },
        }
    )
    validate_transport(plan, certificate, value, root=root)
    return value


def validate_transport(
    plan: dict[str, Any],
    certificate: dict[str, Any],
    transport: dict[str, Any],
    *,
    root: Path,
) -> None:
    """Reject the legacy live-snapshot evaluator that drops frozen task hashes."""
    validate_acquisition_certificate(plan, certificate, root=root)
    _check_seal(transport, TRANSPORT_SCHEMA)
    if set(transport) != {
        "schema",
        "status",
        "plan_sha256",
        "acquisition_certificate_sha256",
        "evaluator_plan",
        "evaluator_preflight",
        "task_bindings",
        "task_binding_set_sha256",
        "task_seed_design_sha256",
        "base_binding",
        "runtime",
        "harness",
        "execution_contract",
        "sha256",
    }:
        raise ValueError("exact-binding transport contains unknown or missing fields")
    frozen = plan["frozen_design"]
    _validate_evaluator_plan(plan, certificate, transport["evaluator_plan"], root)
    _validate_evaluator_preflight(
        plan, transport["evaluator_plan"], transport["evaluator_preflight"], root
    )
    if (
        transport["status"] != "offline_exact_binding_extension_validated"
        or transport["plan_sha256"] != plan["sha256"]
        or transport["acquisition_certificate_sha256"] != certificate["sha256"]
        or transport["task_bindings"] != frozen["task_bindings"]
        or transport["task_binding_set_sha256"] != frozen["task_binding_set_sha256"]
        or transport["task_seed_design_sha256"] != frozen["task_seed_design_sha256"]
        or transport["base_binding"] != certificate["base_binding"]
        or transport["runtime"] != certificate["runtime"]
        or transport["harness"] != frozen["harness"]
    ):
        raise ValueError("exact-binding transport differs from the frozen acquisition")
    if transport["execution_contract"] != {
        "transport": "cyber_fleet_eval_v1",
        "binding_source": "sealed_transport_manifest_only",
        "rehash_before_each_session": True,
        "live_task_snapshot_fallback": False,
        "base_only": True,
        "candidate_sessions": 0,
        "attempt_seeds": frozen["attempt_seeds"],
        "session_count": fleet.DEV_TASKS * fleet.PASS_K,
    }:
        raise ValueError("transport can drop or replace frozen task identities")


def assemble_preflight(
    plan: dict[str, Any],
    certificate: dict[str, Any],
    transport: dict[str, Any],
    *,
    observed_at: str,
    expires_at: str,
    root: Path,
) -> dict[str, Any]:
    """Project the exact evaluator proof into a short-lived, score-free gate."""
    validate_transport(plan, certificate, transport, root=root)
    frozen = plan["frozen_design"]
    value = _seal(
        {
            "schema": PREFLIGHT_SCHEMA,
            "status": "passed_read_only",
            "plan_sha256": plan["sha256"],
            "acquisition_certificate_sha256": certificate["sha256"],
            "transport_plan_sha256": transport["sha256"],
            "observed_at": observed_at,
            "expires_at": expires_at,
            "task_bindings": frozen["task_bindings"],
            "task_binding_set_sha256": frozen["task_binding_set_sha256"],
            "base_binding": certificate["base_binding"],
            "runtime": certificate["runtime"],
            "harness": {
                "harness_sha256": frozen["harness_sha256"],
                "sampling_sha256": frozen["sampling_sha256"],
                "agent_image_digest": certificate["base_binding"]["agent_image_digest"],
                "proxy_image_digest": certificate["base_binding"]["proxy_image_digest"],
                "opencode_version": "1.18.27",
                "ordered_tools": ["bash", "submit_report"],
            },
            "scope": {
                "task_gets": fleet.DEV_TASKS,
                "route_gets_only": True,
                "prompt_requests": 0,
                "completion_requests": 0,
                "task_instance_creates": 0,
                "grading_requests": 0,
                "scoring_requests": 0,
                "scored_sessions": 0,
            },
        }
    )
    validate_preflight(plan, certificate, transport, value, root=root)
    return value


def _private_directory(path: Path, root: Path) -> Path:
    if not path.is_absolute():
        raise ValueError("base outcome paths must be absolute and private")
    resolved = path.resolve()
    repository = root.resolve()
    private = (repository / "data/private").resolve()
    if (
        (resolved == repository or repository in resolved.parents)
        and resolved != private
        and private not in resolved.parents
    ):
        raise ValueError("repository-local base outcomes belong under data/private")
    info = resolved.stat()
    if not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode) & 0o077:
        raise ValueError("base outcome root must be an existing private directory")
    return resolved


def assemble_child(
    plan: dict[str, Any],
    certificate: dict[str, Any],
    transport: dict[str, Any],
    preflight: dict[str, Any],
    *,
    campaign_id: str,
    private_results_root: Path,
    claim_journal: Path,
    root: Path,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Create the base-only child after a fresh exact-binding preflight."""
    check_time = now or datetime.now(UTC)
    validate_preflight(plan, certificate, transport, preflight, root=root, now=check_time)
    if not NAME.fullmatch(campaign_id):
        raise ValueError("base-acquisition campaign id must be an owner-specific lowercase name")
    result_root = _private_directory(private_results_root, root)
    journal = claim_journal.resolve()
    if journal != result_root and result_root not in journal.parents:
        raise ValueError("claim journal must live inside the private result root")
    child = _seal(
        {
            "schema": CHILD_SCHEMA,
            "campaign_id": campaign_id,
            "plan_sha256": plan["sha256"],
            "acquisition_certificate_sha256": certificate["sha256"],
            "preflight_sha256": preflight["sha256"],
            "transport_plan_sha256": transport["sha256"],
            "parent_protocol_sha256": plan["references"]["parent_protocol"]["sha256"],
            "base_control_sha256": plan["references"]["base_control"]["sha256"],
            "task_set_sha256": plan["references"]["task_set"]["sha256"],
            "task_bindings": plan["frozen_design"]["task_bindings"],
            "task_binding_set_sha256": plan["frozen_design"]["task_binding_set_sha256"],
            "task_seed_design_sha256": plan["frozen_design"]["task_seed_design_sha256"],
            "task_seed_pair_count": fleet.DEV_TASKS * fleet.PASS_K,
            "bindings": {
                "base": certificate["base_binding"],
                "runtime_before_post_pairing": certificate["runtime"],
                "post_sft": None,
            },
            "execution_transport": {
                "schema": "cyber_fleet_eval_v1_exact_bindings_v1",
                "generic_transport_without_this_extension_forbidden": True,
                "preflight_and_each_execution_must_rehash_frozen_bindings": True,
                "base_only": True,
                "candidate_sessions": 0,
            },
            "execution": {
                "private_results_root": str(result_root),
                "claim_journal": str(journal),
            },
            "result_policy": RESULT_POLICY,
            "later_reuse_requires_fresh_live_pair_certificate": True,
            "launchable": True,
            "paid_or_scored_work_authorized_by_certificate": False,
        }
    )
    validate_child(plan, certificate, transport, preflight, child, root=root, now=check_time)
    return child


def validate_child(
    plan: dict[str, Any],
    certificate: dict[str, Any],
    transport: dict[str, Any],
    preflight: dict[str, Any],
    child: dict[str, Any],
    *,
    root: Path,
    now: datetime | None = None,
) -> None:
    _check_seal(child, CHILD_SCHEMA)
    if set(child) != {
        "schema",
        "campaign_id",
        "plan_sha256",
        "acquisition_certificate_sha256",
        "preflight_sha256",
        "transport_plan_sha256",
        "parent_protocol_sha256",
        "base_control_sha256",
        "task_set_sha256",
        "task_bindings",
        "task_binding_set_sha256",
        "task_seed_design_sha256",
        "task_seed_pair_count",
        "bindings",
        "execution_transport",
        "execution",
        "result_policy",
        "later_reuse_requires_fresh_live_pair_certificate",
        "launchable",
        "paid_or_scored_work_authorized_by_certificate",
        "sha256",
    }:
        raise ValueError("base-acquisition child contains unknown or missing fields")
    validate_preflight(plan, certificate, transport, preflight, root=root, now=now)
    if (
        child["campaign_id"] != transport["evaluator_plan"]["campaign_id"]
        or child["plan_sha256"] != plan["sha256"]
        or child["acquisition_certificate_sha256"] != certificate["sha256"]
        or child["preflight_sha256"] != preflight["sha256"]
        or child["transport_plan_sha256"] != transport["sha256"]
        or child["parent_protocol_sha256"] != plan["references"]["parent_protocol"]["sha256"]
        or child["base_control_sha256"] != plan["references"]["base_control"]["sha256"]
        or child["task_set_sha256"] != plan["references"]["task_set"]["sha256"]
        or child["task_bindings"] != plan["frozen_design"]["task_bindings"]
        or child["task_binding_set_sha256"] != plan["frozen_design"]["task_binding_set_sha256"]
        or child["task_seed_design_sha256"] != plan["frozen_design"]["task_seed_design_sha256"]
        or child["task_seed_pair_count"] != fleet.DEV_TASKS * fleet.PASS_K
        or child["bindings"]
        != {
            "base": certificate["base_binding"],
            "runtime_before_post_pairing": certificate["runtime"],
            "post_sft": None,
        }
        or child["execution_transport"]
        != {
            "schema": "cyber_fleet_eval_v1_exact_bindings_v1",
            "generic_transport_without_this_extension_forbidden": True,
            "preflight_and_each_execution_must_rehash_frozen_bindings": True,
            "base_only": True,
            "candidate_sessions": 0,
        }
        or child["result_policy"] != RESULT_POLICY
        or child["later_reuse_requires_fresh_live_pair_certificate"] is not True
        or child["launchable"] is not True
        or child["paid_or_scored_work_authorized_by_certificate"] is not False
    ):
        raise ValueError("base-acquisition child weakens or changes the frozen design")
    execution = child["execution"]
    if not isinstance(execution, dict) or set(execution) != {
        "private_results_root",
        "claim_journal",
    }:
        raise ValueError("base-acquisition execution identity is incomplete")
    root_path = _private_directory(Path(execution["private_results_root"]), root)
    journal = Path(execution["claim_journal"]).resolve()
    if journal != root_path and root_path not in journal.parents:
        raise ValueError("claim journal must live inside the private result root")


def seal_base_outcomes(
    plan: dict[str, Any],
    certificate: dict[str, Any],
    transport: dict[str, Any],
    preflight: dict[str, Any],
    child: dict[str, Any],
    completion: dict[str, Any],
    *,
    root: Path,
) -> dict[str, Any]:
    """Project only validity counts and digests from the private base result."""
    validate_child(plan, certificate, transport, preflight, child, root=root)
    _check_seal(completion, PRIVATE_COMPLETION_SCHEMA)
    if set(completion) != {
        "schema",
        "child_sha256",
        "terminal_classification",
        "terminal_evidence_sha256",
        "raw_result_manifest_sha256",
        "private_results_root",
        "task_seed_pair_count",
        "valid_outcome_count",
        "infrastructure_invalid_count",
        "interrupted_or_unknown_count",
        "resources_released",
        "outcomes_exposed",
        "wandb_exported",
        "sha256",
    }:
        raise ValueError("private base completion contains outcome-bearing or unknown fields")
    expected_root = child["execution"]["private_results_root"]
    if (
        completion["child_sha256"] != child["sha256"]
        or completion["terminal_classification"] != "valid_outcome_set"
        or completion["private_results_root"] != expected_root
        or completion["task_seed_pair_count"] != fleet.DEV_TASKS * fleet.PASS_K
        or completion["valid_outcome_count"] != fleet.DEV_TASKS * fleet.PASS_K
        or completion["infrastructure_invalid_count"] != 0
        or completion["interrupted_or_unknown_count"] != 0
        or completion["resources_released"] is not True
        or completion["outcomes_exposed"] is not False
        or completion["wandb_exported"] is not False
    ):
        raise ValueError("base completion is not the complete frozen 80-outcome design")
    _private_directory(Path(expected_root), root)
    for key in ("terminal_evidence_sha256", "raw_result_manifest_sha256"):
        _sha(completion[key], key)
    return _seal(
        {
            "schema": BASE_OUTCOME_SEAL_SCHEMA,
            "plan_sha256": plan["sha256"],
            "acquisition_certificate_sha256": certificate["sha256"],
            "child_sha256": child["sha256"],
            "parent_protocol_sha256": plan["references"]["parent_protocol"]["sha256"],
            "base_control_sha256": plan["references"]["base_control"]["sha256"],
            "task_binding_set_sha256": plan["frozen_design"]["task_binding_set_sha256"],
            "task_seed_design_sha256": plan["frozen_design"]["task_seed_design_sha256"],
            "task_seed_pair_count": fleet.DEV_TASKS * fleet.PASS_K,
            "all_frozen_pairs_valid": True,
            "terminal_evidence_sha256": completion["terminal_evidence_sha256"],
            "raw_result_manifest_sha256": completion["raw_result_manifest_sha256"],
            "outcomes_remaining_private": True,
            "wandb_exported": False,
        }
    )


def _validate_base_outcome_seal(
    plan: dict[str, Any], certificate: dict[str, Any], child: dict[str, Any], seal: dict[str, Any]
) -> None:
    _check_seal(seal, BASE_OUTCOME_SEAL_SCHEMA)
    if set(seal) != {
        "schema",
        "plan_sha256",
        "acquisition_certificate_sha256",
        "child_sha256",
        "parent_protocol_sha256",
        "base_control_sha256",
        "task_binding_set_sha256",
        "task_seed_design_sha256",
        "task_seed_pair_count",
        "all_frozen_pairs_valid",
        "terminal_evidence_sha256",
        "raw_result_manifest_sha256",
        "outcomes_remaining_private",
        "wandb_exported",
        "sha256",
    }:
        raise ValueError("base outcome seal contains unknown or missing fields")
    if (
        seal["plan_sha256"] != plan["sha256"]
        or seal["acquisition_certificate_sha256"] != certificate["sha256"]
        or seal["child_sha256"] != child["sha256"]
        or seal["parent_protocol_sha256"] != plan["references"]["parent_protocol"]["sha256"]
        or seal["base_control_sha256"] != plan["references"]["base_control"]["sha256"]
        or seal["task_binding_set_sha256"] != plan["frozen_design"]["task_binding_set_sha256"]
        or seal["task_seed_design_sha256"] != plan["frozen_design"]["task_seed_design_sha256"]
        or seal["task_seed_pair_count"] != fleet.DEV_TASKS * fleet.PASS_K
        or seal["all_frozen_pairs_valid"] is not True
        or seal["outcomes_remaining_private"] is not True
        or seal["wandb_exported"] is not False
    ):
        raise ValueError("base outcome seal does not bind the complete frozen acquisition")
    _sha(seal["terminal_evidence_sha256"], "terminal evidence")
    _sha(seal["raw_result_manifest_sha256"], "raw result manifest")


def assemble_matched_pair_certificate(
    plan: dict[str, Any],
    acquisition: dict[str, Any],
    transport: dict[str, Any],
    preflight: dict[str, Any],
    acquisition_child: dict[str, Any],
    base_outcomes: dict[str, Any],
    pair: dict[str, Any],
    candidate_child: dict[str, Any],
    *,
    root: Path,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Re-prove parity before reusing the early base vector for one candidate."""
    validate_child(plan, acquisition, transport, preflight, acquisition_child, root=root)
    _validate_base_outcome_seal(plan, acquisition, acquisition_child, base_outcomes)
    _check_seal(pair, PAIR_RECEIPT_SCHEMA)
    if set(pair) != {
        "schema",
        "status",
        "plan_sha256",
        "acquisition_certificate_sha256",
        "base_outcome_seal_sha256",
        "observed_at",
        "base_serving",
        "post_sft",
        "base_runtime",
        "post_runtime",
        "checks",
        "observations",
        "sha256",
    }:
        raise ValueError("matched-pair receipt contains unknown or missing fields")
    observed = _time(pair["observed_at"], "matched-pair observed_at")
    check_time = (now or datetime.now(UTC)).astimezone(UTC)
    if not observed <= check_time <= observed + timedelta(minutes=15):
        raise ValueError("matched-pair receipt is not fresh within 15 minutes")
    if (
        pair["status"] != "passed"
        or pair["plan_sha256"] != plan["sha256"]
        or pair["acquisition_certificate_sha256"] != acquisition["sha256"]
        or pair["base_outcome_seal_sha256"] != base_outcomes["sha256"]
    ):
        raise ValueError("matched-pair receipt does not bind the acquired base outcomes")
    _validate_exact_fields(pair["base_serving"], PAIR_BASE_FIELDS, "paired base")
    _validate_exact_fields(pair["post_sft"], PAIR_POST_FIELDS, "paired post-SFT")
    for field in ACQUISITION_BASE_FIELDS:
        if pair["base_serving"][field] != acquisition["base_binding"][field]:
            raise ValueError(f"paired base {field} differs from acquisition")
    if (
        pair["base_runtime"] != acquisition["runtime"]
        or pair["post_runtime"] != acquisition["runtime"]
    ):
        raise ValueError("fresh base/post runtime differs from base acquisition")
    if (
        pair["post_sft"]["weights_manifest_sha256"]
        == pair["base_serving"]["weights_manifest_sha256"]
    ):
        raise ValueError("post-SFT weights must differ from acquired base weights")
    for field in (
        "tokenizer_manifest_sha256",
        "chat_template_sha256",
        "agent_image_digest",
        "proxy_image_digest",
    ):
        if pair["post_sft"][field] != pair["base_serving"][field]:
            raise ValueError(f"fresh base/post {field} differs")
    if pair["checks"] != {
        "post_checkpoint_export_accepted": True,
        "post_staging_create_once": True,
        "base_and_post_registration_current": True,
        "base_acquisition_runtime_revalidated": True,
        "tokenizer_encode_decode_match": True,
        "structured_tool_call_both": True,
        "fixed_logit_probe_finite_both": True,
        "fixed_logit_probe_deterministic_within_arm": True,
        "context_compaction_and_autocontinue_both": True,
        "same_runtime_except_weights": True,
        "all_80_base_outcomes_still_sealed": True,
    }:
        raise ValueError("fresh live-pair checks are incomplete")
    observations = pair["observations"]
    digest_fields = {
        "normalized_tool_request_sha256",
        "normalized_logit_request_sha256",
        "tokenizer_probe_sha256",
        "base_response_manifest_sha256",
        "post_response_manifest_sha256",
    }
    if not isinstance(observations, dict) or set(observations) != digest_fields | {
        "window_seconds",
        "non_scored_completion_requests",
        "task_instance_creates",
        "grading_requests",
    }:
        raise ValueError("fresh live-pair observation projection is incomplete")
    for field in digest_fields:
        _sha(observations[field], field)
    if (
        type(observations["window_seconds"]) is not int
        or not 0 < observations["window_seconds"] <= 900
        or type(observations["non_scored_completion_requests"]) is not int
        or observations["non_scored_completion_requests"] < 1
        or observations["task_instance_creates"] != 0
        or observations["grading_requests"] != 0
    ):
        raise ValueError("fresh parity crossed a task or grading boundary")

    study_sealing.validate_child(candidate_child, root=root)
    if (
        candidate_child["surface"] != "fleet_dev"
        or candidate_child["parent_protocol"]["sha256"]
        != plan["references"]["parent_protocol"]["sha256"]
        or candidate_child["base_control"]["sha256"] != plan["references"]["base_control"]["sha256"]
    ):
        raise ValueError("candidate child uses a different split-A parent or base control")
    pair_sha256 = pair["sha256"]
    base_binding = {
        **pair["base_serving"],
        "live_pair_parity_receipt_sha256": pair_sha256,
    }
    post_binding = {**pair["post_sft"], "live_parity_receipt_sha256": pair_sha256}
    matched_runtime = {
        **pair["base_runtime"],
        "live_pair_parity_receipt_sha256": pair_sha256,
    }
    if candidate_child["bindings"] != {
        "base": base_binding,
        "post_sft": post_binding,
        "matched_runtime": matched_runtime,
    }:
        raise ValueError("candidate child bindings differ from fresh matched parity")
    certificate = _seal(
        {
            "schema": MATCHED_PAIR_CERTIFICATE_SCHEMA,
            "plan_sha256": plan["sha256"],
            "parent_protocol_sha256": plan["references"]["parent_protocol"]["sha256"],
            "base_control_sha256": plan["references"]["base_control"]["sha256"],
            "acquisition_certificate_sha256": acquisition["sha256"],
            "base_outcome_seal_sha256": base_outcomes["sha256"],
            "base_raw_result_manifest_sha256": base_outcomes["raw_result_manifest_sha256"],
            "live_pair_parity_receipt_sha256": pair_sha256,
            "candidate_child_sha256": candidate_child["sha256"],
            "task_seed_pair_count": fleet.DEV_TASKS * fleet.PASS_K,
            "base_serving_binding": base_binding,
            "post_sft_binding": post_binding,
            "matched_runtime": matched_runtime,
            "only_scientific_difference": "weights_manifest_sha256",
            "outcomes_remaining_private": True,
            "wandb_exported": False,
            "paid_or_scored_work_authorized": False,
        }
    )
    return certificate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-plan")
    validate.add_argument("plan", type=Path)
    certify = commands.add_parser("certify-acquisition")
    certify.add_argument("plan", type=Path)
    certify.add_argument("--payload", type=Path, required=True)
    certify.add_argument("--route", type=Path, required=True)
    certify.add_argument("--harness", type=Path, required=True)
    certify.add_argument("--agent-image-qualification", type=Path, required=True)
    certify.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    plan = _read(args.plan)
    if args.command == "validate-plan":
        validate_plan(plan, root=root)
        print(canonical_json({"launchable": False, "plan_sha256": plan["sha256"]}))
        return 0
    value = assemble_acquisition_certificate(
        plan,
        _read(args.payload),
        _read(args.route),
        _read(args.harness),
        _read(args.agent_image_qualification),
        root=root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json_once(args.output, value)
    print(
        canonical_json(
            {
                "certificate_sha256": value["sha256"],
                "paid_or_scored_work_authorized": False,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
