"""Held, non-mutating launch package for post-rank99 Qwen DP8 qualification."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any

from evals.fleet import glm53_dedicated_v7 as jobs_api
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
    parser.add_argument("command", choices=("validate", "render-held", "validate-preview"))
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    value = spec(Path.cwd())
    if args.command == "render-held":
        envelope = {
            "status": value["status"],
            "launch_authorized": False,
            "scoring_authorized": False,
            "config_sha256": value["config_sha256"],
            "jobs_api_payload": jobs_payload(Path.cwd()),
        }
        print(json.dumps(envelope, sort_keys=True))
    elif args.command == "validate-preview":
        if args.manifest is None:
            raise ValueError("--manifest is required")
        print(json.dumps(preview_identity(args.manifest.read_text(), Path.cwd()), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
