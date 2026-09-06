"""Held, non-mutating launch package for post-rank99 Qwen DP8 qualification."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import shlex
import subprocess
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from evals.fleet import glm53_dedicated_v7 as jobs_api
from evals.fleet import glm53_dedicated_v8_live as live_api
from evals.fleet import opencode_actual_harness_parity_v1 as parity
from evals.fleet import qwen38_dedicated_dp8_v1 as prior
from evals.fleet import qwen38_dedicated_dp8_v1_live as prior_live
from evals.fleet import qwen38_dp8_post_rank99_plan_v1 as held
from evals.fleet import self_hosted

SPEC_PATH = Path("evals/fleet/configs/qwen38-dedicated-dp8-post-rank99-v1-held.json")
PREVIEW_RECEIPT_PATH = Path(
    "docs/evidence/qwen38-study/2026-09-06-qwen38-dp8-post-rank99-authenticated-preview-v1.json"
)
SCHEMA = "fleet-qwen38-dedicated-dp8-post-rank99-launch-held-v1"
PREVIEW_RECEIPT_SCHEMA = "fleet-qwen38-dp8-post-rank99-authenticated-preview-v1"
TRANSITION_SCHEMA = "fleet-qwen38-rank99-to-dp8-transition-release-v1"
QUALIFICATION_SCHEMA = "fleet-qwen38-dp8-nonscored-qualification-plan-v1"
RESULT_SCHEMA = "fleet-qwen38-dp8-nonscored-qualification-result-v1"
LIVE_SUBMIT_GATE_SCHEMA = "fleet-qwen38-dp8-post-rank99-live-submit-gate-v1"
SUBMISSION_SCHEMA = "fleet-qwen38-dp8-post-rank99-submission-v1"
SERVER_BINDING_SCHEMA = "fleet-qwen38-dp8-post-rank99-server-binding-v1"
PARTITION_SCHEMA = "fleet-qwen38-dp8-post-rank99-scored-partition-held-v1"
SNAPSHOT_SCHEMA = "fleet-qwen38-dp8-rank-distribution-snapshot-v1"
DISTRIBUTION_SCHEMA = "fleet-qwen38-dp8-wave-distribution-v1"
LEVELS = (1, 2, 4, 8)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def jobs_payload(root: Path) -> dict[str, Any]:
    """Render exact Jobs API bytes without contacting or mutating the API."""
    lifecycle = (root / prior.LIFECYCLE_PATH).read_text()
    payload = {
        "image": prior.IMAGE,
        "command": (
            "bash -lc " + shlex.quote(lifecycle) + " -- " + shlex.join(prior.SERVER_ARGUMENTS)
        ),
        "workers": 1,
        "gpus_per_worker": 8,
        "env": {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "QWEN38_RUN_DIR": held.RUN_DIR,
        },
        "secrets": [],
        "resources": {
            "cpu_request": "32",
            "cpu_limit": "96",
            "memory_request": "256Gi",
            "memory_limit": "768Gi",
        },
        "priority_class": prior.PRIORITY_CLASS,
        "privileged": False,
        "run_dir": held.RUN_DIR,
        "title": held.TITLE,
    }
    if set(payload) != jobs_api.EXPECTED_API_FIELDS:
        raise AssertionError("post-rank99 DP8 Jobs API payload shape drifted")
    return payload


def validate_spec(value: dict[str, Any], root: Path) -> None:
    held_value = held.load_held(root)
    payload = jobs_payload(root)
    expected = {
        "schema_version": SCHEMA,
        "status": "HELD_NO_LAUNCH",
        "launch_authorized": False,
        "scoring_authorized": False,
        "held_plan_path": str(held.PLAN_PATH),
        "held_plan_receipt_sha256": held_value["receipt_sha256"],
        "identity": {
            "title": held.TITLE,
            "run_dir": held.RUN_DIR,
            "serving_block": held.SERVING_BLOCK,
            "create_once": True,
            "jobs_api_preview_route": "POST /v1/runs/preview",
            "jobs_api_submit_route": "POST /v1/runs",
        },
        "runtime": {
            "image": prior.IMAGE,
            "model_repository": "Qwen/Qwen3.8-27B",
            "model_revision": prior.MODEL_REVISION,
            "model_path": prior.MODEL_PATH,
            "served_id": "qwen3.8-27b",
            "context_length": 262144,
            "tensor_parallel_size": 1,
            "data_parallel_size": 8,
            "load_balance_method": "total_tokens",
            "reasoning_parser": "qwen3",
            "tool_call_parser": "qwen3_coder",
            "server_arguments_sha256": prior.SERVER_ARGUMENTS_SHA256,
            "jobs_api_payload_sha256": self_hosted.sha256(self_hosted.canonical_json(payload)),
        },
        "resources": {
            "workers": 1,
            "gpus_per_worker": 8,
            "max_project_gpu_nodes": 2,
            "max_project_gpus": 16,
            "priority_class": prior.PRIORITY_CLASS,
            "preemption_policy": "Never",
            "privileged": False,
        },
        "release_guard": {
            "required_schema": TRANSITION_SCHEMA,
            "all_four_rank99_acceptances_required": True,
            "rank99_controller_terminal_success_required": True,
            "tp1_jobs_api_delete_204_then_get_404_required": True,
            "tp1_runtime_objects_absent_required": True,
            "tp1_gpu_released_required": True,
            "fresh_duplicate_and_node_inventory_required": True,
            "release_must_authorize_one_server_and_zero_scoring": True,
        },
        "qualification": {
            "schema_version": QUALIFICATION_SCHEMA,
            "levels": list(LEVELS),
            "actual_opencode_version": "1.18.27",
            "context_management": ("opencode_1.18.27_native_compaction_autocontinue_v1"),
            "compaction_headroom_tokens": 20000,
            "max_output_tokens": 32768,
            "max_model_requests": 600,
            "tools": ["bash", "submit_report"],
            "tool_catalog_sha256": (
                "sha256:85fad6bdc3a835bf52a11a99b3387740eb06eb3d1720ad9bb33f3feac215b44a"
            ),
            "strict_tool_name_order_and_arguments": True,
            "stop_at_first_failed_level": True,
            "task_instance_session_verifier_scoring_calls": 0,
        },
        "privacy": {
            "credentials_included": False,
            "prompts_traces_flags_or_scores_included": False,
            "request_or_response_bodies_included": False,
        },
    }
    actual = {key: item for key, item in value.items() if key != "config_sha256"}
    if actual != expected:
        raise ValueError("post-rank99 DP8 held launch spec drifted")
    if value.get("config_sha256") != self_hosted.digest_without(value, "config_sha256"):
        raise ValueError("post-rank99 DP8 held launch spec digest drifted")


def spec(root: Path) -> dict[str, Any]:
    value = _load(root / SPEC_PATH)
    validate_spec(value, root)
    return value


def validate_preview_receipt(value: dict[str, Any], root: Path) -> None:
    """Validate the sanitized evidence from the authenticated preview-only API call."""
    expected = {
        "schema_version": PREVIEW_RECEIPT_SCHEMA,
        "status": "PASSED_NON_MUTATING",
        "observed_at": "2026-09-06T04:45:16Z",
        "held_config_path": str(SPEC_PATH),
        "held_config_sha256": spec(root)["config_sha256"],
        "request_sha256": self_hosted.sha256(self_hosted.canonical_json(jobs_payload(root))),
        "route": "POST /v1/runs/preview",
        "rendered": {
            "command_sha256": (
                "sha256:e0fda171b57c418bd499b9442b7285a6cdf629dd84523b28f75c7bf16ec2c8bd"
            ),
            "gpus": 8,
            "image": prior.IMAGE,
            "image_pull_secrets": ["ghcr-pull"],
            "preferred_topology": "topology.nebius.com/tier-1",
            "priority_class": prior.PRIORITY_CLASS,
            "privileged": False,
            "privileged_field_present": False,
            "queue_name": "training-lq",
            "run_dir": held.RUN_DIR,
            "suspended": True,
        },
        "side_effects": {
            "api_mutations": 0,
            "launch_authorized": False,
            "scoring_authorized": False,
        },
        "privacy": {
            "credentials_included": False,
            "response_body_included": False,
            "prompts_traces_flags_or_scores_included": False,
        },
    }
    actual = {key: item for key, item in value.items() if key != "receipt_sha256"}
    if actual != expected:
        raise ValueError("authenticated DP8 preview receipt drifted")
    if value.get("receipt_sha256") != self_hosted.digest_without(value, "receipt_sha256"):
        raise ValueError("authenticated DP8 preview receipt digest drifted")


def validate_transition_release(value: dict[str, Any]) -> None:
    """Validate a future append-only release; this function performs no reads or writes."""
    if value.get("receipt_sha256") != self_hosted.digest_without(value, "receipt_sha256"):
        raise ValueError("rank99 transition release digest drifted")
    if (
        value.get("schema_version") != TRANSITION_SCHEMA
        or value.get("status") != "RELEASED_FOR_ONE_NON_SCORED_DP8_SERVER"
        or value.get("launch_authorized") is not True
        or value.get("scoring_authorized") is not False
        or value.get("title") != held.TITLE
        or value.get("run_dir") != held.RUN_DIR
        or value.get("serving_block") != held.SERVING_BLOCK
        or value.get("rank99_accepted_validated_count") != 4
        or value.get("rank99_controller_terminal_success") is not True
        or value.get("tp1_jobs_api_delete_http_status") != 204
        or value.get("tp1_jobs_api_get_http_status_after_delete") != 404
        or value.get("tp1_runtime_objects_absent") is not True
        or value.get("tp1_gpu_released") is not True
        or value.get("fresh_title_matches") != 0
        or value.get("fresh_run_dir_matches") != 0
        or value.get("fresh_sfs_run_dir_exists") is not False
        or value.get("project_gpu_nodes_before_create", 99) > 1
        or value.get("project_gpus_before_create", 99) > 8
        or value.get("api_mutations") != 0
        or value.get("prompts_traces_flags_or_scores_included") is not False
    ):
        raise ValueError("rank99 transition release is not sufficient")
    receipts = value.get("rank99_accepted_validated_receipts")
    if (
        not isinstance(receipts, list)
        or len(receipts) != 4
        or len(set(receipts)) != 4
        or any(
            not isinstance(item, str) or not item.startswith("sha256:") or len(item) != 71
            for item in receipts
        )
    ):
        raise ValueError("rank99 acceptance chain is incomplete")


def validate_live_submit_gate(value: dict[str, Any]) -> None:
    """Require a fresh post-release duplicate/resource observation immediately before create."""
    if value.get("receipt_sha256") != self_hosted.digest_without(value, "receipt_sha256"):
        raise ValueError("DP8 live submit gate digest drifted")
    if (
        value.get("schema_version") != LIVE_SUBMIT_GATE_SCHEMA
        or value.get("status") != "CLEAR"
        or value.get("title") != held.TITLE
        or value.get("run_dir") != held.RUN_DIR
        or value.get("jobs_api_title_matches") != 0
        or value.get("jobs_api_run_dir_matches") != 0
        or value.get("kubernetes_identity_matches") != 0
        or value.get("sfs_run_dir_exists") is not False
        or value.get("project_gpu_nodes_before_create", 99) > 1
        or value.get("project_gpus_before_create", 99) > 8
        or value.get("tp1_objects_absent") is not True
        or value.get("tp1_gpu_released") is not True
        or value.get("observed_after_transition_release") is not True
        or not isinstance(value.get("freshness_seconds_at_submit"), int)
        or not 0 <= value["freshness_seconds_at_submit"] <= 30
        or value.get("api_mutations") != 0
        or value.get("prompts_traces_flags_or_scores_included") is not False
    ):
        raise ValueError("DP8 live submit gate is not clear")


def submit_create_once(
    client: Any,
    transition_release: dict[str, Any],
    live_gate: dict[str, Any],
    root: Path,
    *,
    source_commit: str,
) -> dict[str, Any]:
    """Create exactly one Jobs API run after immutable release and immediate live gates."""
    validate_transition_release(transition_release)
    validate_live_submit_gate(live_gate)
    if len(source_commit) != 40 or any(char not in "0123456789abcdef" for char in source_commit):
        raise ValueError("DP8 submit source commit is invalid")
    payload = jobs_payload(root)
    runs = live_api._runs(client)  # noqa: SLF001 - shared authenticated Jobs API reader
    if any(row.get("title") == held.TITLE or row.get("run_dir") == held.RUN_DIR for row in runs):
        raise RuntimeError("DP8 create-once identity appeared after the live gate")
    preview = client.post("/v1/runs/preview", json=payload)
    preview.raise_for_status()
    rendered = preview_identity(preview.json()["manifest_yaml"], root)
    response = client.post("/v1/runs", json=payload)
    response.raise_for_status()
    if response.status_code != 202:
        raise RuntimeError("DP8 create did not return 202")
    api_run_id = response.json().get("name")
    if not isinstance(api_run_id, str) or not api_run_id.startswith("ft-run-"):
        raise RuntimeError("DP8 create omitted API run identity")
    receipt = {
        "schema_version": SUBMISSION_SCHEMA,
        "status": "SUBMITTED_NON_SCORED_SERVER",
        "submitted_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "api_run_id": api_run_id,
        "title": held.TITLE,
        "run_dir": held.RUN_DIR,
        "serving_block": held.SERVING_BLOCK,
        "source_commit": source_commit,
        "request_sha256": self_hosted.sha256(self_hosted.canonical_json(payload)),
        "transition_release_sha256": transition_release["receipt_sha256"],
        "live_submit_gate_sha256": live_gate["receipt_sha256"],
        "rendered": rendered,
        "route": "POST /v1/runs",
        "http_status": 202,
        "server_instances_created": 1,
        "scored_calls": 0,
        "prompts_traces_flags_or_scores_included": False,
    }
    receipt["receipt_sha256"] = self_hosted.digest_without(receipt, "receipt_sha256")
    return receipt


def validate_server_binding_receipt(
    value: dict[str, Any], submission: Mapping[str, Any]
) -> dict[str, Any]:
    """Bind the one submitted server to exact Ray/Kubernetes UIDs before qualification."""
    validate_submission_receipt(submission)
    if value.get("receipt_sha256") != self_hosted.digest_without(value, "receipt_sha256"):
        raise ValueError("DP8 server binding digest drifted")
    for field in ("rayjob_uid", "workload_uid", "head_pod_uid", "service_uid"):
        try:
            parsed = uuid.UUID(str(value.get(field)))
        except ValueError as exc:
            raise ValueError("DP8 server binding UID is invalid") from exc
        if parsed.int == 0:
            raise ValueError("DP8 server binding UID is invalid")
    if (
        value.get("schema_version") != SERVER_BINDING_SCHEMA
        or value.get("status") != "READY_NON_SCORED"
        or value.get("submission_receipt_sha256") != submission.get("receipt_sha256")
        or value.get("api_run_id") != submission.get("api_run_id")
        or value.get("title") != held.TITLE
        or value.get("run_dir") != held.RUN_DIR
        or value.get("serving_block") != held.SERVING_BLOCK
        or not isinstance(value.get("service_origin"), str)
        or not value["service_origin"].startswith("http://")
        or not value["service_origin"].endswith(":8000")
        or not isinstance(value.get("head_pod_name"), str)
        or not value["head_pod_name"]
        or value.get("image") != prior.IMAGE
        or value.get("model_revision") != prior.MODEL_REVISION
        or value.get("context_length") != 262144
        or value.get("tensor_parallel_size") != 1
        or value.get("data_parallel_size") != 8
        or value.get("head_pod_running_ready") is not True
        or value.get("head_pod_restarts") != 0
        or value.get("kueue_preempted") is not False
        or value.get("scoring_authorized") is not False
        or value.get("prompts_traces_flags_or_scores_included") is not False
    ):
        raise ValueError("DP8 server binding is not qualification-ready")
    return parity.validate_server_binding(
        {
            "api_run_id": value["api_run_id"],
            "rayjob_uid": value["rayjob_uid"],
            "head_pod_uid": value["head_pod_uid"],
            "service_uid": value["service_uid"],
            "served_id": "qwen3.8-27b",
            "model_revision": value["model_revision"],
            "context_length": value["context_length"],
        },
        "qwen3.8-27b",
    )


def validate_submission_receipt(value: Mapping[str, Any]) -> None:
    if value.get("receipt_sha256") != self_hosted.digest_without(dict(value), "receipt_sha256"):
        raise ValueError("DP8 submission receipt digest drifted")
    if (
        value.get("schema_version") != SUBMISSION_SCHEMA
        or value.get("status") != "SUBMITTED_NON_SCORED_SERVER"
        or value.get("title") != held.TITLE
        or value.get("run_dir") != held.RUN_DIR
        or value.get("serving_block") != held.SERVING_BLOCK
        or value.get("route") != "POST /v1/runs"
        or value.get("http_status") != 202
        or value.get("server_instances_created") != 1
        or value.get("scored_calls") != 0
        or value.get("prompts_traces_flags_or_scores_included") is not False
    ):
        raise ValueError("DP8 submission receipt drifted")


def qualification_plan(
    server_binding: dict[str, Any], service_origin: str, root: Path
) -> dict[str, Any]:
    """Freeze a content-free c1->c2->c4->c8 qualification plan."""
    binding = parity.validate_server_binding(server_binding, "qwen3.8-27b")
    if not service_origin.startswith("http://") or not service_origin.endswith(":8000"):
        raise ValueError("DP8 qualification service origin is invalid")
    plan = {
        "schema_version": QUALIFICATION_SCHEMA,
        "status": "HELD_NON_SCORED",
        "launch_authorized": False,
        "scoring_authorized": False,
        "serving_block": held.SERVING_BLOCK,
        "server_binding": binding,
        "service_origin": service_origin,
        "treatment": spec(root)["qualification"],
        "waves": [
            {
                "concurrency": level,
                "parallel_streams": level,
                "requests_per_stream": 2,
                "request_count": level * 2,
                "tool_order": ["bash", "submit_report"],
                "strict_tool_arguments_required": True,
            }
            for level in LEVELS
        ],
        "acceptance": {
            "levels_must_run_in_order": True,
            "zero_http_model_or_tool_protocol_errors": True,
            "all_tool_names_order_and_arguments_exact": True,
            "stop_at_first_failed_level": True,
            "highest_passing_level_is_scored_concurrency_ceiling": True,
        },
        "side_effects": {
            "task_calls": 0,
            "instance_calls": 0,
            "session_calls": 0,
            "verifier_calls": 0,
            "scoring_calls": 0,
        },
        "privacy": {
            "credentials_included": False,
            "request_or_response_bodies_included": False,
            "prompts_traces_flags_or_scores_included": False,
        },
    }
    plan["receipt_sha256"] = self_hosted.digest_without(plan, "receipt_sha256")
    return plan


def validate_qualification_plan(value: Mapping[str, Any]) -> None:
    """Require the complete score-free ladder, exact treatment, and no extra fields."""
    if value.get("receipt_sha256") != self_hosted.digest_without(dict(value), "receipt_sha256"):
        raise ValueError("DP8 qualification plan digest drifted")
    if set(value) != {
        "schema_version",
        "status",
        "launch_authorized",
        "scoring_authorized",
        "serving_block",
        "server_binding",
        "service_origin",
        "treatment",
        "waves",
        "acceptance",
        "side_effects",
        "privacy",
        "receipt_sha256",
    }:
        raise ValueError("DP8 qualification plan contains an unreviewed field")
    expected_waves = [
        {
            "concurrency": level,
            "parallel_streams": level,
            "requests_per_stream": 2,
            "request_count": level * 2,
            "tool_order": ["bash", "submit_report"],
            "strict_tool_arguments_required": True,
        }
        for level in LEVELS
    ]
    if (
        value.get("schema_version") != QUALIFICATION_SCHEMA
        or value.get("status") != "HELD_NON_SCORED"
        or value.get("launch_authorized") is not False
        or value.get("scoring_authorized") is not False
        or value.get("serving_block") != held.SERVING_BLOCK
        or value.get("treatment") != spec(parity.REPO_ROOT)["qualification"]
        or value.get("waves") != expected_waves
        or value.get("acceptance")
        != {
            "levels_must_run_in_order": True,
            "zero_http_model_or_tool_protocol_errors": True,
            "all_tool_names_order_and_arguments_exact": True,
            "stop_at_first_failed_level": True,
            "highest_passing_level_is_scored_concurrency_ceiling": True,
        }
        or value.get("side_effects")
        != {
            "task_calls": 0,
            "instance_calls": 0,
            "session_calls": 0,
            "verifier_calls": 0,
            "scoring_calls": 0,
        }
        or value.get("privacy")
        != {
            "credentials_included": False,
            "request_or_response_bodies_included": False,
            "prompts_traces_flags_or_scores_included": False,
        }
    ):
        raise ValueError("DP8 qualification plan treatment drifted")
    parity.validate_server_binding(value.get("server_binding") or {}, "qwen3.8-27b")
    origin = value.get("service_origin")
    if (
        not isinstance(origin, str)
        or not origin.startswith("http://")
        or not origin.endswith(":8000")
    ):
        raise ValueError("DP8 qualification plan service origin drifted")


def validate_stream_receipt(value: Mapping[str, Any], plan: Mapping[str, Any]) -> None:
    """Validate one sanitized actual-OpenCode parity receipt against the exact server/treatment."""
    if value.get("receipt_sha256") != self_hosted.digest_without(dict(value), "receipt_sha256"):
        raise ValueError("DP8 stream receipt digest drifted")
    if set(value) != {
        "schema_version",
        "status",
        "classification",
        "model",
        "endpoint",
        "harness",
        "tool_contract",
        "execution",
        "privacy",
        "receipt_sha256",
    }:
        raise ValueError("DP8 stream receipt contains an unreviewed field")
    treatment = parity.treatment_config("qwen3.8-27b")
    endpoint = value.get("endpoint") or {}
    harness = value.get("harness") or {}
    tools = value.get("tool_contract") or {}
    execution = value.get("execution") or {}
    privacy = value.get("privacy") or {}
    expected_binding = plan.get("server_binding")
    expected_binding_sha = self_hosted.sha256(self_hosted.canonical_json(expected_binding))
    expected_mcp_sha = self_hosted.sha256(self_hosted.canonical_json(parity.mcp_tools()))
    expected_openai_sha = self_hosted.sha256(
        self_hosted.canonical_json(parity.expected_openai_tools())
    )
    harness_fields = treatment["harness"]
    expected_observed_image = {
        "image": parity.IMAGE,
        "image_id": parity.IMAGE_ID,
        "os": "linux",
        "architecture": "amd64",
        "user": "node",
        "working_dir": "/workspace",
    }
    observed_catalogs = tools.get("observed_model_request_catalog_sha256s")
    if (
        value.get("schema_version") != parity.SCHEMA
        or value.get("status") != "PASSED_NON_SCORED"
        or value.get("classification") != "ACTUAL_HARNESS_PARITY"
        or value.get("model") != treatment["model"]
        or set(endpoint) != {"origin", "kind", "server_binding", "server_binding_sha256"}
        or endpoint.get("origin") != plan.get("service_origin")
        or endpoint.get("kind") != "dedicated_uid_bound_inference"
        or endpoint.get("server_binding") != expected_binding
        or endpoint.get("server_binding_sha256") != expected_binding_sha
        or set(harness)
        != {*harness_fields, "image", "image_id", "observed_image", "settings_sha256"}
        or any(harness.get(key) != expected for key, expected in harness_fields.items())
        or harness.get("image") != parity.IMAGE
        or harness.get("image_id") != parity.IMAGE_ID
        or harness.get("observed_image") != expected_observed_image
        or not isinstance(harness.get("settings_sha256"), str)
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", harness["settings_sha256"])
        or set(tools)
        != {
            "names",
            "mcp_catalog_sha256",
            "production_catalog_provenance",
            "openai_catalog_sha256",
            "model_request_catalog_exact",
            "model_request_tool_names_exact",
            "model_request_tool_descriptions_exact",
            "model_request_tool_parameters_exact",
            "observed_model_request_catalog_sha256s",
            "model_requests_with_tools",
            "model_requests_without_tools",
            "calls_observed_in_order",
            "arguments_structurally_valid",
        }
        or tools.get("names") != ["bash", "submit_report"]
        or tools.get("mcp_catalog_sha256") != expected_mcp_sha
        or tools.get("production_catalog_provenance")
        != parity.production_tools.provenance(parity.REPO_ROOT)
        or tools.get("openai_catalog_sha256") != expected_openai_sha
        or tools.get("model_request_catalog_exact") is not True
        or tools.get("model_request_tool_names_exact") is not True
        or tools.get("model_request_tool_descriptions_exact") is not True
        or tools.get("model_request_tool_parameters_exact") is not True
        or observed_catalogs != [expected_openai_sha]
        or type(tools.get("model_requests_with_tools")) is not int
        or tools["model_requests_with_tools"] < 1
        or type(tools.get("model_requests_without_tools")) is not int
        or tools["model_requests_without_tools"] < 0
        or tools.get("calls_observed_in_order") != ["bash", "submit_report"]
        or tools.get("arguments_structurally_valid") is not True
        or set(execution)
        != {
            "harness_exit_code",
            "model_requests",
            "final_marker_observed",
            "task_instance_session_verifier_scoring_calls",
            "scored_launch_authorized",
        }
        or type(execution.get("model_requests")) is not int
        or not 1 <= execution["model_requests"] <= parity.MAX_MODEL_REQUESTS
        or tools.get("model_requests_with_tools") + tools.get("model_requests_without_tools")
        != execution.get("model_requests")
        or execution.get("harness_exit_code") != 0
        or execution.get("final_marker_observed") is not True
        or execution.get("task_instance_session_verifier_scoring_calls") != 0
        or execution.get("scored_launch_authorized") is not False
        or privacy
        != {
            "credentials_included": False,
            "prompt_included": False,
            "responses_or_model_outputs_included": False,
            "tool_arguments_included": False,
            "stderr_or_stdout_included": False,
            "benchmark_content_included": False,
        }
    ):
        raise ValueError("DP8 stream receipt treatment or server binding drifted")


def validate_distribution_receipt(
    value: Mapping[str, Any],
    plan: Mapping[str, Any],
    level: int,
    rows: list[Mapping[str, Any]],
) -> None:
    """Require UID-bound request and GPU activity across enough DP ranks for this wave."""
    if value.get("receipt_sha256") != self_hosted.digest_without(dict(value), "receipt_sha256"):
        raise ValueError("DP8 distribution receipt digest drifted")
    if set(value) != {
        "schema_version",
        "status",
        "plan_receipt_sha256",
        "server_binding",
        "service_origin",
        "concurrency",
        "stream_receipt_sha256s",
        "request_counters_before_by_rank",
        "request_counters_after_by_rank",
        "request_deltas_by_rank",
        "gpu_peak_utilization_percent_by_rank",
        "gpu_peak_memory_used_mib_by_rank",
        "gpu_device_count",
        "gpu_memory_loaded_count",
        "sampling_seconds",
        "observer_errors",
        "task_instance_session_verifier_scoring_calls",
        "prompts_traces_flags_or_scores_included",
        "receipt_sha256",
    }:
        raise ValueError("DP8 distribution receipt contains an unreviewed field")
    expected_streams = [row.get("receipt_sha256") for row in rows]
    before = value.get("request_counters_before_by_rank")
    after = value.get("request_counters_after_by_rank")
    request_deltas = value.get("request_deltas_by_rank")
    gpu_peaks = value.get("gpu_peak_utilization_percent_by_rank")
    gpu_memory = value.get("gpu_peak_memory_used_mib_by_rank")
    counters_valid = all(
        isinstance(row, list)
        and len(row) == 8
        and all(type(item) is int and item >= 0 for item in row)
        for row in (before, after, request_deltas)
    )
    gpu_valid = (
        isinstance(gpu_peaks, list)
        and len(gpu_peaks) == 8
        and all(type(item) is int and 0 <= item <= 100 for item in gpu_peaks)
        and isinstance(gpu_memory, list)
        and len(gpu_memory) == 8
        and all(type(item) is int and item > 0 for item in gpu_memory)
    )
    active_requests = (
        [index for index, count in enumerate(request_deltas) if type(count) is int and count > 0]
        if isinstance(request_deltas, list) and len(request_deltas) == 8
        else []
    )
    active_gpus = (
        [index for index, peak in enumerate(gpu_peaks) if type(peak) is int and peak > 0]
        if isinstance(gpu_peaks, list) and len(gpu_peaks) == 8
        else []
    )
    required = set(range(8)) if level == 8 else None
    if (
        value.get("schema_version") != DISTRIBUTION_SCHEMA
        or value.get("status") != "PASSED_NON_SCORED_DISTRIBUTION"
        or value.get("plan_receipt_sha256") != plan.get("receipt_sha256")
        or value.get("server_binding") != plan.get("server_binding")
        or value.get("service_origin") != plan.get("service_origin")
        or value.get("concurrency") != level
        or value.get("stream_receipt_sha256s") != expected_streams
        or not counters_valid
        or request_deltas != [after[index] - before[index] for index in range(8)]
        or not gpu_valid
        or value.get("gpu_device_count") != 8
        or value.get("gpu_memory_loaded_count") != 8
        or type(value.get("sampling_seconds")) is not int
        or value["sampling_seconds"] < 0
        or value.get("observer_errors") != []
        or len(active_requests) < level
        or len(active_gpus) < level
        or (required is not None and set(active_requests) != required)
        or (required is not None and set(active_gpus) != required)
        or value.get("task_instance_session_verifier_scoring_calls") != 0
        or value.get("prompts_traces_flags_or_scores_included") is not False
    ):
        raise ValueError("DP8 wave did not prove UID-bound request/GPU distribution")


_LABEL = re.compile(r'(?:^|,\s*)([A-Za-z_][A-Za-z0-9_]*)="([^"]*)"')


def _request_counters_by_rank(metrics: str) -> list[int]:
    """Extract one unambiguous eight-rank cumulative request counter family."""
    families: dict[str, dict[int, int]] = {}
    for raw in metrics.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "{" not in line or "}" not in line:
            continue
        head, rest = line.split("{", 1)
        labels_text, raw_value = rest.split("}", 1)
        if "requests_total" not in head:
            continue
        labels = dict(_LABEL.findall(labels_text))
        rank_text = labels.get("dp_rank") or labels.get("data_parallel_rank")
        if rank_text is None or not rank_text.isdigit():
            continue
        try:
            value = int(float(raw_value.strip().split()[0]))
        except (IndexError, ValueError):
            continue
        rank = int(rank_text)
        if 0 <= rank < 8 and value >= 0:
            families.setdefault(head, {})[rank] = value
    complete = [values for values in families.values() if set(values) == set(range(8))]
    if len(complete) != 1:
        raise ValueError("DP8 metrics did not expose one unambiguous eight-rank request family")
    return [complete[0][rank] for rank in range(8)]


def _gpu_sample(pod_name: str, pod_uid: str) -> tuple[list[int], list[int]]:
    pod = json.loads(
        subprocess.run(
            ["kubectl", "-n", live_api.NAMESPACE, "get", "pod", pod_name, "-o", "json"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    if pod.get("metadata", {}).get("uid") != pod_uid:
        raise RuntimeError("DP8 GPU observer Pod UID drifted")
    rows = subprocess.run(
        [
            "kubectl",
            "-n",
            live_api.NAMESPACE,
            "exec",
            pod_name,
            "--",
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    parsed: dict[int, tuple[int, int]] = {}
    for row in rows:
        fields = [field.strip() for field in row.split(",")]
        if len(fields) != 3 or not all(field.isdigit() for field in fields):
            raise RuntimeError("DP8 GPU observer returned invalid device data")
        parsed[int(fields[0])] = (int(fields[1]), int(fields[2]))
    if set(parsed) != set(range(8)):
        raise RuntimeError("DP8 GPU observer did not see exactly eight devices")
    return (
        [parsed[index][0] for index in range(8)],
        [parsed[index][1] for index in range(8)],
    )


def live_wave_observer(
    binding_receipt: Mapping[str, Any],
) -> Callable[
    [int, Callable[[], list[dict[str, Any]]], Mapping[str, Any]],
    tuple[list[dict[str, Any]], Mapping[str, Any]],
]:
    """Produce per-wave request-rank deltas and sampled GPU activity from exact server UIDs."""
    origin = str(binding_receipt["service_origin"])
    pod_name = str(binding_receipt["head_pod_name"])
    pod_uid = str(binding_receipt["head_pod_uid"])

    def observe(
        level: int,
        execute: Callable[[], list[dict[str, Any]]],
        plan: Mapping[str, Any],
    ) -> tuple[list[dict[str, Any]], Mapping[str, Any]]:
        before_response = httpx.get(origin + "/metrics", timeout=30)
        before_response.raise_for_status()
        before = _request_counters_by_rank(before_response.text)
        max_memory, max_utilization = _gpu_sample(pod_name, pod_uid)
        stop = threading.Event()
        errors: list[str] = []

        def sample() -> None:
            while not stop.is_set():
                try:
                    memory, utilization = _gpu_sample(pod_name, pod_uid)
                    for index in range(8):
                        max_memory[index] = max(max_memory[index], memory[index])
                        max_utilization[index] = max(max_utilization[index], utilization[index])
                except (
                    OSError,
                    RuntimeError,
                    subprocess.SubprocessError,
                    json.JSONDecodeError,
                ) as exc:
                    errors.append(type(exc).__name__)
                    return
                stop.wait(1)

        sampler = threading.Thread(target=sample, daemon=True)
        started = time.time()
        sampler.start()
        try:
            rows = execute()
        finally:
            stop.set()
            sampler.join(timeout=10)
        after_response = httpx.get(origin + "/metrics", timeout=30)
        after_response.raise_for_status()
        after = _request_counters_by_rank(after_response.text)
        deltas = [after[index] - before[index] for index in range(8)]
        if any(delta < 0 for delta in deltas):
            raise RuntimeError("DP8 request counters decreased during a qualification wave")
        receipt = {
            "schema_version": DISTRIBUTION_SCHEMA,
            "status": "PASSED_NON_SCORED_DISTRIBUTION" if not errors else "FAILED",
            "plan_receipt_sha256": plan["receipt_sha256"],
            "server_binding": plan["server_binding"],
            "service_origin": origin,
            "concurrency": level,
            "stream_receipt_sha256s": [row.get("receipt_sha256") for row in rows],
            "request_counters_before_by_rank": before,
            "request_counters_after_by_rank": after,
            "request_deltas_by_rank": deltas,
            "gpu_peak_utilization_percent_by_rank": max_utilization,
            "gpu_peak_memory_used_mib_by_rank": max_memory,
            "gpu_device_count": len(max_memory),
            "gpu_memory_loaded_count": sum(value > 0 for value in max_memory),
            "sampling_seconds": max(0, int(time.time() - started)),
            "observer_errors": errors,
            "task_instance_session_verifier_scoring_calls": 0,
            "prompts_traces_flags_or_scores_included": False,
        }
        receipt["receipt_sha256"] = self_hosted.digest_without(receipt, "receipt_sha256")
        return rows, receipt

    return observe


def run_qualification_ladder(
    plan: Mapping[str, Any],
    probe: Callable[[], Mapping[str, Any]],
    wave_observer: Callable[
        [int, Callable[[], list[dict[str, Any]]], Mapping[str, Any]],
        tuple[list[dict[str, Any]], Mapping[str, Any]],
    ],
) -> dict[str, Any]:
    """Run actual-OpenCode probes c1->c2->c4->c8, stopping before the next failed level."""
    validate_qualification_plan(plan)

    def safe_probe(_index: int) -> dict[str, Any]:
        try:
            return dict(probe())
        except Exception as exc:  # noqa: BLE001 - sanitized terminal classification
            value = {
                "status": "FAILED",
                "error_type": type(exc).__name__,
                "tool_contract": {},
                "execution": {"task_instance_session_verifier_scoring_calls": 0},
                "privacy": {"responses_or_model_outputs_included": False},
            }
            value["receipt_sha256"] = self_hosted.digest_without(value, "receipt_sha256")
            return value

    observations = []
    highest = 0
    for level in LEVELS:

        def execute_wave(wave_level: int = level) -> list[dict[str, Any]]:
            with concurrent.futures.ThreadPoolExecutor(max_workers=wave_level) as executor:
                return list(executor.map(safe_probe, range(wave_level)))

        rows, distribution = wave_observer(level, execute_wave, plan)
        stream_valid = []
        for row in rows:
            try:
                validate_stream_receipt(row, plan)
            except ValueError:
                stream_valid.append(False)
            else:
                stream_valid.append(True)
        try:
            validate_distribution_receipt(distribution, plan, level, rows)
        except ValueError:
            distribution_valid = False
        else:
            distribution_valid = True
        passed = len(rows) == level and all(stream_valid) and distribution_valid
        observations.append(
            {
                "concurrency": level,
                "status": "PASSED" if passed else "FAILED",
                "request_count": level * 2,
                "completed_count": level * 2 if passed else 0,
                "error_count": 0 if passed else 1,
                "tool_order_exact": passed,
                "tool_arguments_exact": passed,
                "stream_receipts": rows,
                "distribution_receipt": dict(distribution),
            }
        )
        if not passed:
            break
        highest = level
    result = {
        "schema_version": RESULT_SCHEMA,
        "plan_receipt_sha256": plan["receipt_sha256"],
        "qualification_plan": dict(plan),
        "levels": observations,
        "highest_passing_concurrency": highest,
        "scored_calls": 0,
        "prompts_traces_flags_or_scores_included": False,
    }
    result["receipt_sha256"] = self_hosted.digest_without(result, "receipt_sha256")
    validate_result(result, dict(plan))
    return result


def held_scored_partition(
    qualification_result: Mapping[str, Any], cells: list[Mapping[str, Any]]
) -> dict[str, Any]:
    """Freeze whole unstarted tasks; scoring remains closed until a separate canary release."""
    highest = validate_result(
        dict(qualification_result),
        {"receipt_sha256": qualification_result.get("plan_receipt_sha256")},
    )
    if highest < 1 or not cells:
        raise ValueError("DP8 qualification did not admit a scored canary")
    by_task: dict[str, list[Mapping[str, Any]]] = {}
    for row in cells:
        task = row.get("task_version_id")
        if not isinstance(task, str) or not task:
            raise ValueError("DP8 partition task identity is absent")
        by_task.setdefault(task, []).append(row)
    normalized = []
    for task, rows in sorted(by_task.items()):
        attempts = sorted(row.get("attempt") for row in rows)
        if attempts != [1, 2, 3, 4]:
            raise ValueError("DP8 partition must preserve complete pass@4 task boundaries")
        for row in rows:
            for field in ("cell_id", "execution_id"):
                identity = row.get(field)
                if (
                    not isinstance(identity, str)
                    or not identity.startswith("sha256:")
                    or len(identity) != 71
                ):
                    raise ValueError("DP8 partition cell identity is invalid")
            if (
                row.get("status") != "unstarted"
                or row.get("accepted_receipt_sha256") is not None
                or row.get("active_claim_sha256") is not None
                or row.get("authoritative_session_matches") != 0
                or row.get("output_root_exists") is not False
            ):
                raise ValueError("DP8 partition includes a non-unstarted cell")
        normalized.append(
            {
                "task_version_id": task,
                "cells": [
                    {
                        "attempt": row["attempt"],
                        "cell_id": row["cell_id"],
                        "execution_id": row["execution_id"],
                    }
                    for row in sorted(rows, key=lambda item: item["attempt"])
                ],
            }
        )
    all_cells = [row["cell_id"] for rows in by_task.values() for row in rows]
    all_executions = [row["execution_id"] for rows in by_task.values() for row in rows]
    if len(set(all_cells)) != len(all_cells) or len(set(all_executions)) != len(all_executions):
        raise ValueError("DP8 partition contains duplicate cell or execution identities")
    value = {
        "schema_version": PARTITION_SCHEMA,
        "status": "HELD_PENDING_ONE_ACCEPTED_CANARY",
        "launch_authorized": False,
        "scoring_authorized": False,
        "serving_block": held.SERVING_BLOCK,
        "qualification_result_sha256": qualification_result["receipt_sha256"],
        "qualified_concurrency_ceiling": highest,
        "canary": normalized[0]["cells"][0],
        "whole_task_partitions": normalized,
        "accepted_cells_excluded": True,
        "active_or_blocked_cells_excluded": True,
        "fresh_launch_time_collision_check_required": True,
        "one_formal_canary_acceptance_required_before_bulk": True,
        "max_bulk_streams_after_canary": highest,
        "prompts_traces_flags_or_scores_included": False,
    }
    value["receipt_sha256"] = self_hosted.digest_without(value, "receipt_sha256")
    return value


def validate_result(value: dict[str, Any], plan: dict[str, Any]) -> int:
    """Return the highest passing level, failing closed on partial or invalid evidence."""
    embedded_plan = value.get("qualification_plan")
    if not isinstance(embedded_plan, dict):
        raise ValueError("DP8 qualification result omitted its exact plan")
    validate_qualification_plan(embedded_plan)
    if set(plan) != {"receipt_sha256"} and plan != embedded_plan:
        raise ValueError("DP8 qualification embedded plan identity drifted")
    plan = embedded_plan
    if value.get("receipt_sha256") != self_hosted.digest_without(value, "receipt_sha256"):
        raise ValueError("DP8 qualification result digest drifted")
    if set(value) != {
        "schema_version",
        "plan_receipt_sha256",
        "qualification_plan",
        "levels",
        "highest_passing_concurrency",
        "scored_calls",
        "prompts_traces_flags_or_scores_included",
        "receipt_sha256",
    }:
        raise ValueError("DP8 qualification result contains an unreviewed field")
    if (
        value.get("schema_version") != RESULT_SCHEMA
        or value.get("plan_receipt_sha256") != plan.get("receipt_sha256")
        or value.get("scored_calls") != 0
        or value.get("prompts_traces_flags_or_scores_included") is not False
    ):
        raise ValueError("DP8 qualification result identity drifted")
    observations = value.get("levels")
    if not isinstance(observations, list) or not observations:
        raise ValueError("DP8 qualification result is empty")
    if any(not isinstance(row, dict) for row in observations):
        raise ValueError("DP8 qualification level is invalid")
    expected_prefix = list(LEVELS[: len(observations)])
    if [row.get("concurrency") for row in observations] != expected_prefix:
        raise ValueError("DP8 qualification ladder order drifted")
    highest = 0
    failed = False
    for row in observations:
        if set(row) != {
            "concurrency",
            "status",
            "request_count",
            "completed_count",
            "error_count",
            "tool_order_exact",
            "tool_arguments_exact",
            "stream_receipts",
            "distribution_receipt",
        }:
            raise ValueError("DP8 qualification level contains an unreviewed field")
        streams = row.get("stream_receipts")
        distribution = row.get("distribution_receipt")
        detailed_valid = isinstance(streams, list) and len(streams) == row.get("concurrency")
        if detailed_valid:
            try:
                for stream in streams:
                    validate_stream_receipt(stream, plan)
                validate_distribution_receipt(distribution or {}, plan, row["concurrency"], streams)
            except (TypeError, ValueError):
                detailed_valid = False
        passed = all(
            (
                row.get("status") == "PASSED",
                row.get("request_count") == row.get("concurrency") * 2,
                row.get("completed_count") == row.get("request_count"),
                row.get("error_count") == 0,
                row.get("tool_order_exact") is True,
                row.get("tool_arguments_exact") is True,
                detailed_valid,
            )
        )
        if failed or (row.get("status") == "PASSED" and not passed):
            raise ValueError("DP8 qualification result violates stop-on-failure")
        if passed:
            highest = row["concurrency"]
        else:
            failed = True
    if not failed and len(observations) != len(LEVELS):
        raise ValueError("DP8 qualification stopped without a failed level")
    if value.get("highest_passing_concurrency") != highest:
        raise ValueError("DP8 qualification ceiling drifted")
    return highest


def preview_identity(manifest_yaml: str, root: Path) -> dict[str, Any]:
    """Validate an authenticated preview response without submitting the run."""
    return prior_live._preview_identity(manifest_yaml, jobs_payload(root))  # noqa: SLF001


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("validate", "render-held", "validate-preview", "submit", "qualify", "partition"),
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--transition-release", type=Path)
    parser.add_argument("--live-submit-gate", type=Path)
    parser.add_argument("--submission", type=Path)
    parser.add_argument("--server-binding", type=Path)
    parser.add_argument("--qualification-result", type=Path)
    parser.add_argument("--cells", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path.cwd()
    value = spec(root)
    if args.command == "render-held":
        envelope = {
            "status": value["status"],
            "launch_authorized": False,
            "scoring_authorized": False,
            "config_sha256": value["config_sha256"],
            "jobs_api_payload": jobs_payload(root),
        }
        print(json.dumps(envelope, sort_keys=True))
    elif args.command == "validate-preview":
        if args.manifest is None:
            raise ValueError("--manifest is required")
        print(json.dumps(preview_identity(args.manifest.read_text(), root), sort_keys=True))
    elif args.command == "submit":
        if None in (args.transition_release, args.live_submit_gate, args.output):
            raise ValueError("submit requires transition release, live submit gate, and output")
        if args.output.exists() or args.output.is_symlink():
            raise FileExistsError(args.output)
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        if dirty:
            raise RuntimeError("DP8 submit requires a clean immutable worktree")
        source_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        release = _load(args.transition_release)
        gate = _load(args.live_submit_gate)
        with httpx.Client(
            base_url=live_api.BASE_URL,
            headers={
                "Authorization": f"Bearer {live_api._token()}",  # noqa: SLF001
                "Accept": "application/json",
            },
            timeout=60,
        ) as client:
            receipt = submit_create_once(client, release, gate, root, source_commit=source_commit)
        self_hosted.write_json_once(args.output, receipt)
        print(json.dumps({"status": receipt["status"], "api_run_id": receipt["api_run_id"]}))
    elif args.command == "qualify":
        if None in (args.submission, args.server_binding, args.output):
            raise ValueError("qualify requires submission, server binding, and output")
        if args.output.exists() or args.output.is_symlink():
            raise FileExistsError(args.output)
        submission = _load(args.submission)
        binding_receipt = _load(args.server_binding)
        binding = validate_server_binding_receipt(binding_receipt, submission)
        plan = qualification_plan(binding, binding_receipt["service_origin"], root)
        result = run_qualification_ladder(
            plan,
            lambda: parity.run(
                "qwen3.8-27b",
                "",
                upstream_origin=binding_receipt["service_origin"],
                server_binding=binding,
            ),
            live_wave_observer(binding_receipt),
        )
        self_hosted.write_json_once(args.output, result)
        print(json.dumps({"highest_passing_concurrency": result["highest_passing_concurrency"]}))
    elif args.command == "partition":
        if None in (args.qualification_result, args.cells, args.output):
            raise ValueError("partition requires qualification result, cells, and output")
        if args.output.exists() or args.output.is_symlink():
            raise FileExistsError(args.output)
        cells = json.loads(args.cells.read_text())
        if not isinstance(cells, list):
            raise ValueError("partition cells must be a list")
        receipt = held_scored_partition(_load(args.qualification_result), cells)
        self_hosted.write_json_once(args.output, receipt)
        print(json.dumps({"status": receipt["status"], "launch_authorized": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
