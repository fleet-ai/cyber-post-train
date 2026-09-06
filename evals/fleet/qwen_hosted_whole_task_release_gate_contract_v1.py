"""Small packaged validator for hosted-Qwen release-gate observations."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from typing import Any

SCHEMA = "fleet-qwen38-hosted-whole-task-release-gate-observation-v1"
SHA_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


class GateContractError(RuntimeError):
    """The sealed release-gate observation violates its public contract."""


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def sha256(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def digest(value: dict[str, Any]) -> str:
    return sha256(canonical({key: item for key, item in value.items() if key != "receipt_sha256"}))


def _uuid(value: Any, label: str) -> str:
    try:
        parsed = uuid.UUID(str(value))
    except (ValueError, AttributeError) as exc:
        raise GateContractError(f"{label}_invalid") from exc
    if parsed.int == 0:
        raise GateContractError(f"{label}_invalid")
    return str(parsed)


def validate_observation(receipt: Any, *, binding: dict[str, Any] | None = None) -> None:
    if not isinstance(receipt, dict):
        raise GateContractError("release_gate_observation_invalid")
    aggregates = receipt.get("observed_aggregates") or {}
    requests = receipt.get("request_counts") or {}
    collisions = receipt.get("collisions") or {}
    if any(
        (
            receipt.get("schema_version") != SCHEMA,
            receipt.get("status") != "CLEAR_PENDING_TERMINAL_JOB_VALIDATION",
            SHA_RE.fullmatch(str(receipt.get("binding_sha256"))) is None,
            binding is not None and receipt.get("binding_sha256") != binding["binding_sha256"],
            receipt.get("statistical_cell_count") != 8,
            receipt.get("controller_count") != 2,
            any(
                SHA_RE.fullmatch(str(receipt.get(field))) is None
                for field in (
                    "predecessor_object_set_sha256",
                    "fresh_object_set_sha256",
                    "plan_set_sha256",
                )
            ),
            collisions
            != {
                "canonical_claims": 0,
                "accepted_receipts": 0,
                "authoritative_sessions": 0,
                "fresh_kubernetes_objects": 0,
                "sfs_output_roots": 0,
            },
            any(not isinstance(value, int) or value < 0 for value in aggregates.values()),
            aggregates.get("predecessor_jobs_exclusively_failed") != 2,
            aggregates.get("predecessor_pods_terminal_restart_zero") != 2,
            aggregates.get("predecessor_immutable_configmaps_bound") != 2,
            aggregates.get("fresh_object_sets_absent") != 2,
            aggregates.get("checked_sfs_roots_absent") != 6,
            aggregates.get("endpoint_lease_slots_simultaneously_free") != 2,
            requests.get("fleet_account_gets") != 1,
            not isinstance(requests.get("fleet_session_inventory_gets"), int),
            requests.get("fleet_session_inventory_gets", 0) < 2,
            requests.get("kubernetes_gets") != 12,
            requests.get("transcript_prompt_task_verifier_or_score_gets") != 0,
            receipt.get("methods") != ["GET"],
            any(
                receipt.get(field) != 0
                for field in (
                    "model_calls",
                    "task_calls",
                    "session_mutations",
                    "verifier_calls",
                    "scoring_calls",
                    "api_mutations",
                )
            ),
            receipt.get("scores_included") is not False,
            receipt.get("prompts_traces_flags_included") is not False,
            receipt.get("credentials_included") is not False,
            receipt.get("receipt_sha256") != digest(receipt),
            UTC_RE.fullmatch(str(receipt.get("observed_at_utc"))) is None,
        )
    ):
        raise GateContractError("release_gate_observation_invalid")
    runtime = receipt.get("runtime") or {}
    if runtime.get("namespace") != "fleet-train-jobs":
        raise GateContractError("release_gate_runtime_namespace_invalid")
    _uuid(runtime.get("job_uid"), "observer_job_uid")
    _uuid(runtime.get("pod_uid"), "observer_pod_uid")
