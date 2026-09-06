"""Held, non-mutating launch package for post-rank99 Qwen DP8 qualification."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import shlex
import subprocess
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


def qualification_plan(server_binding: dict[str, Any], root: Path) -> dict[str, Any]:
    """Freeze a content-free c1->c2->c4->c8 qualification plan."""
    binding = parity.validate_server_binding(server_binding, "qwen3.8-27b")
    plan = {
        "schema_version": QUALIFICATION_SCHEMA,
        "status": "HELD_NON_SCORED",
        "launch_authorized": False,
        "scoring_authorized": False,
        "serving_block": held.SERVING_BLOCK,
        "server_binding": binding,
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


def run_qualification_ladder(
    plan: Mapping[str, Any],
    probe: Callable[[], Mapping[str, Any]],
) -> dict[str, Any]:
    """Run actual-OpenCode probes c1->c2->c4->c8, stopping before the next failed level."""
    if (
        plan.get("schema_version") != QUALIFICATION_SCHEMA
        or plan.get("receipt_sha256") != self_hosted.digest_without(dict(plan), "receipt_sha256")
        or plan.get("status") != "HELD_NON_SCORED"
        or plan.get("launch_authorized") is not False
        or plan.get("scoring_authorized") is not False
    ):
        raise ValueError("DP8 qualification plan schema drifted")

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
        with concurrent.futures.ThreadPoolExecutor(max_workers=level) as executor:
            rows = list(executor.map(safe_probe, range(level)))
        passed = all(
            row.get("status") == "PASSED_NON_SCORED"
            and isinstance(row.get("receipt_sha256"), str)
            and row["receipt_sha256"].startswith("sha256:")
            and len(row["receipt_sha256"]) == 71
            and (row.get("tool_contract") or {}).get("calls_observed_in_order")
            == ["bash", "submit_report"]
            and (row.get("tool_contract") or {}).get("arguments_structurally_valid") is True
            and (row.get("execution") or {}).get("task_instance_session_verifier_scoring_calls")
            == 0
            and (row.get("privacy") or {}).get("responses_or_model_outputs_included") is False
            for row in rows
        )
        observations.append(
            {
                "concurrency": level,
                "status": "PASSED" if passed else "FAILED",
                "request_count": level * 2,
                "completed_count": level * 2 if passed else 0,
                "error_count": 0 if passed else 1,
                "tool_order_exact": passed,
                "tool_arguments_exact": passed,
                "stream_receipt_sha256s": [row.get("receipt_sha256") for row in rows],
            }
        )
        if not passed:
            break
        highest = level
    result = {
        "schema_version": RESULT_SCHEMA,
        "plan_receipt_sha256": plan["receipt_sha256"],
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
    if value.get("receipt_sha256") != self_hosted.digest_without(value, "receipt_sha256"):
        raise ValueError("DP8 qualification result digest drifted")
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
        passed = all(
            (
                row.get("status") == "PASSED",
                row.get("request_count") == row.get("concurrency") * 2,
                row.get("completed_count") == row.get("request_count"),
                row.get("error_count") == 0,
                row.get("tool_order_exact") is True,
                row.get("tool_arguments_exact") is True,
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
        plan = qualification_plan(binding, root)
        result = run_qualification_ladder(
            plan,
            lambda: parity.run(
                "qwen3.8-27b",
                "",
                upstream_origin=binding_receipt["service_origin"],
                server_binding=binding,
            ),
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
