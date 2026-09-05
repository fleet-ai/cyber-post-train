"""Render fail-closed GLM dedicated-v7 gate receipts from sanitized observations.

This module does not inspect model output.  Its inputs are score-blind observer
receipts produced beside the Jobs API/Kubernetes resources; every output is
revalidated by the dedicated-v7 authority before it can gate a controller.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evals.fleet import glm53_dedicated_v7 as dedicated
from evals.fleet import self_hosted

OBSERVATION_SCHEMA = "fleet-exact-pass4-final-dedicated-observation-v5"


def _validate_observation(value: dict[str, Any], kind: str, replica: str) -> None:
    expected = {
        "schema_version": OBSERVATION_SCHEMA,
        "kind": kind,
        "replica": replica,
        "methods": ["GET"],
        "mutation_calls": 0,
        "scores_included": False,
        "prompts_traces_flags_or_scores_included": False,
        "credentials_included": False,
    }
    if any(value.get(key) != item for key, item in expected.items()):
        raise ValueError("dedicated v5 sanitized observation contract drifted")
    if value.get("receipt_sha256") != self_hosted.digest_without(value, "receipt_sha256"):
        raise ValueError("dedicated v5 sanitized observation digest drifted")


def preview_receipt(observation: dict[str, Any], root: Path, replica: str) -> dict[str, Any]:
    _validate_observation(observation, "authenticated_preview", replica)
    value = dedicated.spec(root)
    payload = dedicated.server_payload(value, root, replica)
    rendered = observation.get("rendered") or {}
    body = {
        "schema_version": dedicated.PREVIEW_SCHEMA,
        "status": "PASSED",
        "replica": replica,
        "authenticated": True,
        "base_url": "https://api.ft.flt.build",
        "route": "/v1/runs/preview",
        "request_sha256": self_hosted.sha256(self_hosted.canonical_json(payload)),
        "requested_image": dedicated.IMAGE,
        "rendered_image": rendered.get("image"),
        "rendered_image_pull_secrets": rendered.get("image_pull_secrets"),
        "rendered_workers": rendered.get("workers"),
        "rendered_gpus_per_worker": rendered.get("gpus_per_worker"),
        "rendered_priority_class": rendered.get("priority_class"),
        "rendered_preemption_policy": rendered.get("preemption_policy"),
        "rendered_run_dir": rendered.get("run_dir"),
        "post_put_patch_delete": 0,
        "credentials_included": False,
    }
    receipt = {**body, "receipt_sha256": self_hosted.digest_without(body, "receipt_sha256")}
    dedicated.validate_authenticated_preview(receipt, value, root, replica)
    return receipt


def parity_receipt(observation: dict[str, Any], root: Path, replica: str) -> dict[str, Any]:
    _validate_observation(observation, "parity", replica)
    if observation.get("checks") != {
        "job_succeeded": True,
        "pod_restarts": 0,
        "health_http_200": True,
        "service_health_http_200": True,
        "parallel_requests_expected": 2,
        "parallel_requests_completed": 2,
        "parallel_requests_failed": 0,
        "maximum_observed_inflight": 2,
        "scored_requests": 0,
        "task_session_or_verifier_calls": 0,
        "content_blind": True,
    }:
        raise ValueError("dedicated v5 parity observation did not prove the exact checks")
    value = dedicated.spec(root)
    payload = dedicated.server_payload(value, root, replica)
    identity = dedicated._parity_runtime_identity(observation, value, replica)  # noqa: SLF001
    body = {
        "schema_version": dedicated.PARITY_SCHEMA,
        "status": "PASSED",
        "replica": replica,
        "request_sha256": self_hosted.sha256(self_hosted.canonical_json(payload)),
        "run_dir": value["replicas"][replica]["run_dir"],
        "image": dedicated.IMAGE,
        "resolved_image_id": dedicated.IMAGE,
        "model_revision": dedicated.MODEL_REVISION,
        "model_path": dedicated.MODEL_PATH,
        "served_id": "glm-5.3",
        "context_length": 262144,
        "priority_class": "fleet-infra-quiet",
        "preemption_policy": "Never",
        "gpus": 8,
        "pod_restarts": 0,
        "health_http_200": True,
        "service_port": 8000,
        "service_health_http_200": True,
        "server_arguments_sha256": dedicated.CANONICAL_SERVER_ARGUMENTS_SHA256,
        "rendered_command_sha256": self_hosted.sha256(payload["command"].encode()),
        "structured_tools": ["bash", "submit_report"],
        "tool_catalog_sha256": dedicated.TOOL_CATALOG_SHA,
        "content_blind_probe": True,
        "non_scored_model_requests": 2,
        "scored_sessions": 0,
        "concurrency_qualification": {
            "mode": "parallel_barrier",
            "requested_concurrency": 2,
            "maximum_observed_inflight": 2,
            "completed_requests": 2,
            "failed_requests": 0,
            "scored_requests": 0,
            "endpoint_traversal": identity["endpoint_traversal"],
        },
        "probe_resource_policy": {
            "cpu_only": True,
            "gpus": 0,
            "priority_class": "fleet-serve-low",
            "preemption_policy": "Never",
        },
        **identity,
        "prompts_traces_flags_or_scores_included": False,
    }
    receipt = {**body, "receipt_sha256": self_hosted.digest_without(body, "receipt_sha256")}
    dedicated.validate_parity_gate(receipt, value, root, replica)
    return receipt


def canary_receipt(
    observation: dict[str, Any], parity: dict[str, Any], root: Path, replica: str
) -> dict[str, Any]:
    _validate_observation(observation, "scored_canary", replica)
    if observation.get("checks") != {
        "source_binding_valid": True,
        "authoritative_session_matches": 1,
        "session_ingest_completed": True,
        "cleanup_completed": True,
        "claim_file_digest_valid": True,
        "claim_job_succeeded": True,
        "claim_pod_restarts": 0,
    }:
        raise ValueError("dedicated v5 canary observation did not prove acceptance")
    value = dedicated.spec(root)
    cell = dedicated.dedicated_canary_cell(value, root, replica)
    exit_code = observation.get("agent_exit_code")
    body = {
        "schema_version": dedicated.CANARY_SCHEMA,
        "accepted": True,
        "credited": True,
        "retry_allowed": False,
        "replica": replica,
        "serving_block": value["replicas"][replica]["serving_block"],
        **cell,
        "run_id": observation.get("run_id"),
        "session_id": observation.get("session_id"),
        "verifier_execution_id": observation.get("verifier_execution_id"),
        "claim_execution_id": cell["execution_id"],
        "claim_path": (
            f"{dedicated.bulk_v3.CLAIM_ROOT}/{cell['execution_id'].removeprefix('sha256:')}.json"
        ),
        "claim_job_kind": "Job",
        "claim_job_namespace": dedicated.NAMESPACE,
        "claim_job_name": observation.get("claim_job_name"),
        "claim_job_uid": observation.get("claim_job_uid"),
        "claim_pod_kind": "Pod",
        "claim_pod_name": observation.get("claim_pod_name"),
        "claim_pod_uid": observation.get("claim_pod_uid"),
        "claim_pod_owner_job_uid": observation.get("claim_job_uid"),
        "claim_receipt_sha256": observation.get("claim_receipt_sha256"),
        "source_binding_valid": True,
        "session_model": dedicated.exact.EXPECTED_MODELS["glm-5.3"]["session_model"],
        "model_revision": dedicated.MODEL_REVISION,
        "authoritative_metadata_run_id": observation.get("run_id"),
        "serving_parity_receipt_sha256": parity.get("receipt_sha256"),
        "serving_runtime": dedicated._parity_runtime_identity(  # noqa: SLF001
            parity, value, replica
        ),
        "authoritative_session_id": observation.get("session_id"),
        "authoritative_verifier_execution_id": observation.get("verifier_execution_id"),
        "authoritative_session_matches": 1,
        "agent_exit_code": exit_code,
        "agent_process_exit_success": exit_code == 0,
        "session_ingest_completed": True,
        "cleanup_completed": True,
        "cleanup_scope": "exact_session_and_task_environment",
        "scores_included": False,
        "prompts_traces_flags_or_scores_included": False,
    }
    receipt = {**body, "receipt_sha256": self_hosted.digest_without(body, "receipt_sha256")}
    dedicated.validate_canary_acceptance(receipt, parity, value, root, replica)
    return receipt


def runtime_receipt(
    parity: dict[str, Any], canary: dict[str, Any], root: Path, replica: str
) -> dict[str, Any]:
    value = dedicated.spec(root)
    body = {
        "schema_version": dedicated.RUNTIME_SCHEMA,
        "status": "PASSED",
        "replica": replica,
        "parity_receipt_sha256": parity["receipt_sha256"],
        "serving_runtime": dedicated._parity_runtime_identity(parity, value, replica),  # noqa: SLF001
        "scored_canary_accepted": True,
        "scored_canary_cell_id": canary["cell_id"],
        "scored_canary_execution_id": canary["execution_id"],
        "scored_canary_claim_path": canary["claim_path"],
        "scored_canary_claim_receipt_sha256": canary["claim_receipt_sha256"],
        "scored_canary_run_id": canary["run_id"],
        "scored_canary_session_id": canary["session_id"],
        "scored_canary_verifier_execution_id": canary["verifier_execution_id"],
        "scored_canary_authoritative_session_matches": 1,
        "scored_canary_session_ingest_completed": True,
        "scored_canary_cleanup_completed": True,
        "scored_canary_receipt_sha256": canary["receipt_sha256"],
        "dependent_bulk_release_authorized": True,
        "qualified_controller_concurrency": 2,
        "released_streams": [
            f"dedicated-{replica.lower()}-1",
            f"dedicated-{replica.lower()}-2",
        ],
        "maximum_streams_released": 2,
        "prompts_traces_flags_or_scores_included": False,
    }
    receipt = {**body, "receipt_sha256": self_hosted.digest_without(body, "receipt_sha256")}
    dedicated.validate_runtime_gate(receipt, parity, canary, value, root, replica)
    return receipt


def _write_once(path: Path, value: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise ValueError("dedicated evidence output already exists")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    path.chmod(0o400)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preview", "parity", "canary", "runtime"))
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--replica", choices=("A", "B"), required=True)
    parser.add_argument("--observation", type=Path)
    parser.add_argument("--parity", type=Path)
    parser.add_argument("--canary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.repo.resolve()

    def required(path: Path | None, label: str) -> dict[str, Any]:
        if path is None:
            parser.error(f"{args.command} requires {label}")
        return dedicated.load(path.resolve(strict=True))

    if args.command == "preview":
        value = preview_receipt(required(args.observation, "observation"), root, args.replica)
    elif args.command == "parity":
        value = parity_receipt(required(args.observation, "observation"), root, args.replica)
    elif args.command == "canary":
        value = canary_receipt(
            required(args.observation, "observation"),
            required(args.parity, "parity"),
            root,
            args.replica,
        )
    else:
        value = runtime_receipt(
            required(args.parity, "parity"),
            required(args.canary, "canary"),
            root,
            args.replica,
        )
    _write_once(args.output.resolve(), value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
