"""Held score-free concurrency qualification for the UID-bound GLM v22 server."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evals.fleet import glm53_dedicated_v22_a2_package_v1 as server
from evals.fleet import glm53_dedicated_v22_a2_v1 as evidence
from evals.fleet import self_hosted

SCHEMA = "fleet-glm53-dedicated-v22-concurrency-qualification-held-v1"
JOB_NAME = "chris-glm53-dedicated-v22-concurrency-qualification-v1"
RESULT_ROOT = Path("/mnt/sfs/jobs") / JOB_NAME


def render(root: Path) -> dict[str, Any]:
    binding = evidence.load(root / server.BINDING)
    parity = evidence.load(root / server.PARITY)
    evidence._validate_evidence(parity, binding, server.ORIGIN)
    body: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": "READY_HELD",
        "launch_authorized": False,
        "job_name": JOB_NAME,
        "result_root": str(RESULT_ROOT),
        "server": {
            "api_run_id": binding["api_run_id"],
            "rayjob_uid": binding["rayjob_uid"],
            "head_pod_uid": binding["head_pod_uid"],
            "service_uid": binding["service_uid"],
            "origin": server.ORIGIN,
            "served_id": binding["served_id"],
            "model_revision": binding["model_revision"],
            "context_length": binding["context_length"],
            "parity_receipt_sha256": parity["receipt_sha256"],
        },
        "treatment": {
            "harness": parity["harness"]["name"],
            "harness_version": parity["harness"]["version"],
            "harness_image_id": parity["harness"]["image_id"],
            "provider_adapter": parity["harness"]["provider_adapter"],
            "context_management": parity["harness"]["context_management"],
            "compaction_headroom_tokens": parity["harness"]["compaction_headroom_tokens"],
            "context_window_size": parity["harness"]["context_window_size"],
            "max_output_tokens": parity["harness"]["max_output_tokens"],
            "tools": parity["tool_contract"]["names"],
            "tool_catalog_sha256": parity["tool_contract"]["mcp_catalog_sha256"],
            "automatic_retry": False,
        },
        "score_free_boundary": {
            "synthetic_distinct_workload_ids": [
                "synthetic-a",
                "synthetic-b",
                "synthetic-c",
                "synthetic-d",
            ],
            "fleet_task_instance_calls": 0,
            "fleet_session_calls": 0,
            "verifier_calls": 0,
            "scoring_calls": 0,
            "persist_response_content": False,
            "persist_prompts_traces_flags_scores": False,
        },
        "phases": [
            {"name": "baseline", "concurrency": 1, "streams": ["synthetic-a"]},
            {"name": "ramp-2", "concurrency": 2, "streams": ["synthetic-a", "synthetic-b"]},
            {
                "name": "ramp-4",
                "concurrency": 4,
                "streams": ["synthetic-a", "synthetic-b", "synthetic-c", "synthetic-d"],
            },
        ],
        "measurements": {
            "per_request_http_success": True,
            "per_request_latency_seconds": True,
            "aggregate_throughput_requests_per_second": True,
            "server_request_counter_delta": True,
            "gpu_utilization_percent_per_device": True,
            "gpu_memory_mib_per_device": True,
            "response_or_tool_argument_content": False,
        },
        "fail_closed_ramp": {
            "ramp_2_requires": {
                "baseline_all_requests_valid": True,
                "http_or_harness_failures": 0,
                "timeout_or_retry_policy_changes": 0,
                "p95_latency_ratio_to_baseline_lte": 2.0,
                "all_eight_gpus_visible": True,
                "server_identity_unchanged": True,
            },
            "ramp_4_requires": {
                "ramp_2_all_requests_valid": True,
                "http_or_harness_failures": 0,
                "timeout_or_retry_policy_changes": 0,
                "p95_latency_ratio_to_baseline_lte": 2.0,
                "throughput_ratio_to_baseline_gte": 1.5,
                "server_identity_unchanged": True,
            },
            "stop_on_first_failed_requirement": True,
            "never_overlap_scored_controller": True,
        },
        "launch_prerequisites": {
            "rank51_attempt2_terminal_accepted": True,
            "no_active_scored_controller": True,
            "fresh_server_uid_and_workload_history": True,
            "fresh_result_root_absence": True,
            "fresh_endpoint_lease_exclusive": True,
            "actual_model_request_counter_watchdog_active": True,
        },
        "scored_concurrency_change_authorized": False,
    }
    body["package_sha256"] = self_hosted.digest_without(body, "package_sha256")
    return body


def main() -> int:
    print(json.dumps(render(Path.cwd()), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
