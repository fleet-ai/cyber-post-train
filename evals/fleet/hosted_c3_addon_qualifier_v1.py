"""Non-scoring +1-stream qualifier beside two exact hosted Qwen controllers."""

from __future__ import annotations

import argparse
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evals.fleet import endpoint_lease
from evals.fleet import hosted_c4_addon_qualifier_v1 as c4
from evals.fleet import hosted_concurrency4_qualification_v1 as probe

SCHEMA = "fleet-hosted-qwen-c3-addon-qualification-v1"
MODEL = "qwen3.8-27b"
LEASE_ROOT = Path("/mnt/sfs/endpoint-leases/opencode11827-autocontinue-primary-v1")
LEASE_KEY = "qwen-hosted-autocontinue-v1"


class C3QualificationError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def run(
    out_dir: Path,
    *,
    reader: Any | None = None,
    caller: probe.CompletionCaller = c4.post_addon_completion,
    identity_checker: Any = c4.validate_qwen_identity,
    lease_root: Path = LEASE_ROOT,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
    api_key = os.environ.get("FLEET_API_KEY", "")
    if not api_key:
        raise C3QualificationError("fleet_api_key_absent")
    if os.environ.get("EXPECTED_SECRET_UID") != c4.EXPECTED_SECRET_UID:
        raise C3QualificationError("secret_uid_mismatch")
    job_uid = probe._safe_uuid(os.environ.get("JOB_UID", ""), "job_uid")  # noqa: SLF001
    pod_uid = probe._safe_uuid(os.environ.get("POD_UID", ""), "pod_uid")  # noqa: SLF001
    controllers = c4.validate_controllers(reader or c4.InClusterReader())
    context = identity_checker(api_key)

    with endpoint_lease.acquire_endpoint_lease(
        lease_root=lease_root,
        endpoint_key=LEASE_KEY,
        maximum_streams=3,
    ) as lease:
        if lease.slot != 3:
            raise C3QualificationError("existing_controller_slots_not_exactly_one_and_two")
        latencies = [caller(MODEL, tool, api_key) for tool in ("bash", "submit_report")]

    receipt: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": "PASSED_NON_SCORED",
        "classification": "operational_addon_capacity_gate_no_capability_claim",
        "observed_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "runtime": {"job_uid": job_uid, "pod_uid": pod_uid, "cpu_only": True},
        "exact_existing_controllers": controllers,
        "lease": {
            "root": str(lease_root),
            "endpoint_key": LEASE_KEY,
            "existing_slots_required": [1, 2],
            "addon_slot_acquired": 3,
            "maximum_total_owned_streams": 3,
            "released": True,
        },
        "model": {
            "served_id": MODEL,
            "repository": probe.EXPECTED_MODELS[MODEL]["repository"],
            "revision": probe.EXPECTED_MODELS[MODEL]["revision"],
            "endpoint_origin": probe.ORIGIN,
            "response_model_exact": True,
        },
        "context_contract": {
            "expected_context_length": probe.EXPECTED_CONTEXT_LENGTH,
            **context,
        },
        "tool_contract": {
            "names": ["bash", "submit_report"],
            "catalog_sha256": probe.sha256(probe.canonical_json(probe.TOOLS)),
            "forced_selection_order": ["bash", "submit_report"],
            "exact_name_and_argument_validation": True,
            "tool_execution_performed": False,
        },
        "stream": {
            "streams_started": 1,
            "streams_succeeded": 1,
            "requests_succeeded": 2,
            "request_latency_seconds": probe._stats(latencies),  # noqa: SLF001
        },
        "request_counts": {
            "chat_completions": 2,
            "task_instance": 0,
            "session": 0,
            "scoring": 0,
            "verifier": 0,
        },
        "scored_bulk_launch_authorized": False,
        "privacy": {
            "request_bodies_included": False,
            "response_bodies_included": False,
            "tool_arguments_included": False,
            "prompts_traces_flags_or_scores_included": False,
            "credentials_included": False,
        },
    }
    receipt["receipt_sha256"] = probe.digest_without(receipt, "receipt_sha256")
    probe.write_json_once(out_dir / "TERMINAL.json", receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", nargs="?")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.out_dir)
    except Exception as exc:
        code = getattr(exc, "code", "unexpected_failure")
        if args.out_dir.is_dir() and not (args.out_dir / "TERMINAL.json").exists():
            failure = {
                "schema_version": SCHEMA,
                "status": "FAILED_NON_SCORED",
                "failure_code": code,
                "observed_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "task_instance_session_scoring_or_verifier_calls": 0,
                "prompts_traces_flags_or_scores_included": False,
                "credentials_included": False,
            }
            failure["receipt_sha256"] = probe.digest_without(failure, "receipt_sha256")
            probe.write_json_once(args.out_dir / "TERMINAL.json", failure)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
