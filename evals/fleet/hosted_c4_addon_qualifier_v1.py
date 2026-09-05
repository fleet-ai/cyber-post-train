"""Non-scoring +2-stream qualifier beside two exact hosted Qwen controllers.

The qualifier is deliberately narrower than the historical c2/c4 benchmark:
it proves that two additional synthetic structured-tool streams can coexist
with the two exact generation-19 controllers.  It never creates a task,
instance, session, verifier record, or score.
"""

from __future__ import annotations

import argparse
import os
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evals.fleet import endpoint_lease
from evals.fleet import hosted_concurrency4_qualification_v1 as probe

SCHEMA = "fleet-hosted-qwen-c4-addon-qualification-v1"
NAMESPACE = "fleet-train-jobs"
MODEL = "qwen3.8-27b"
EXPECTED_SECRET_UID = "e0febd8e-94a2-46b0-a0bf-dd6b3154187b"
LEASE_ROOT = Path("/mnt/sfs/endpoint-leases/opencode11827-autocontinue-primary-v1")
LEASE_KEY = "qwen-hosted-autocontinue-v1"
EXPECTED_CONTROLLERS = {
    "a": {
        "job_name": "chris-q38-ac-exact100-g19-a192-v4",
        "job_uid": "515c370a-ecb4-4c41-a349-46a328c8fa68",
        "pod_uid": "4d86f0c5-7cef-4f8a-a798-ea5c725e4955",
    },
    "b": {
        "job_name": "chris-q38-ac-exact100-g19-b192-v4",
        "job_uid": "f23805b5-516b-4828-a3b5-82673a1b3e2f",
        "pod_uid": "d5d7b7cf-d639-4ad6-b012-3001c4935b2d",
    },
}


class AddonQualificationError(RuntimeError):
    """Stable response-content-free failure."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def validate_addon_completion(value: dict[str, Any], *, model: str, tool_name: str) -> None:
    """Accept provider terminal markers only after the exact forced call validates."""

    if value.get("model") != model:
        raise AddonQualificationError("completion_model_identity_mismatch")
    choices = value.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise AddonQualificationError("completion_choice_shape_mismatch")
    choice = choices[0]
    message = choice.get("message")
    calls = message.get("tool_calls") if isinstance(message, dict) else None
    if not isinstance(calls, list) or len(calls) != 1 or not isinstance(calls[0], dict):
        raise AddonQualificationError("completion_tool_call_count_mismatch")
    function = calls[0].get("function")
    if not isinstance(function, dict) or function.get("name") != tool_name:
        raise AddonQualificationError("completion_tool_name_mismatch")
    arguments = function.get("arguments")
    if not isinstance(arguments, str):
        raise AddonQualificationError("completion_tool_arguments_invalid")
    parsed = probe._load_json(  # noqa: SLF001
        arguments.encode(), code="completion_tool_arguments_invalid"
    )
    if tool_name == "bash":
        if set(parsed) != {"command"} or not isinstance(parsed["command"], str):
            raise AddonQualificationError("completion_bash_arguments_mismatch")
    elif (
        set(parsed) != {"flags", "explanation"}
        or not isinstance(parsed["flags"], list)
        or any(not isinstance(item, str) for item in parsed["flags"])
        or not isinstance(parsed["explanation"], str)
    ):
        raise AddonQualificationError("completion_submit_report_arguments_mismatch")
    if choice.get("finish_reason") not in {"tool_calls", "stop"}:
        raise AddonQualificationError("completion_finish_reason_mismatch")


def post_addon_completion(model: str, tool_name: str, api_key: str) -> float:
    body = probe.canonical_json(probe._request_payload(model, tool_name))  # noqa: SLF001
    request = urllib.request.Request(
        probe.COMPLETIONS_URL,
        method="POST",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=probe.REQUEST_TIMEOUT_SECONDS) as response:
            if response.status != 200:
                raise AddonQualificationError("completion_http_status")
            raw = response.read(4 * 1024 * 1024 + 1)
    except AddonQualificationError:
        raise
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as exc:
        raise AddonQualificationError("completion_request_failed") from exc
    elapsed = time.monotonic() - started
    if len(raw) > 4 * 1024 * 1024:
        raise AddonQualificationError("completion_response_too_large")
    validate_addon_completion(
        probe._load_json(raw, code="completion_invalid_json"),  # noqa: SLF001
        model=model,
        tool_name=tool_name,
    )
    if elapsed > probe.MAX_REQUEST_LATENCY_SECONDS:
        raise AddonQualificationError("completion_request_latency_exceeded")
    return elapsed


class InClusterReader:
    def __init__(self) -> None:
        host = os.environ.get("KUBERNETES_SERVICE_HOST")
        port = os.environ.get("KUBERNETES_SERVICE_PORT_HTTPS", "443")
        if not host:
            raise AddonQualificationError("in_cluster_service_unavailable")
        token = Path("/var/run/secrets/kubernetes.io/serviceaccount/token").read_text().strip()
        self.base = f"https://{host}:{port}"
        self.headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
        self.context = ssl.create_default_context(
            cafile="/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"
        )

    def get(self, path: str) -> dict[str, Any]:
        request = urllib.request.Request(self.base + path, headers=self.headers)
        try:
            with urllib.request.urlopen(request, timeout=30, context=self.context) as response:
                raw = response.read(4 * 1024 * 1024 + 1)
                status = response.status
        except urllib.error.HTTPError as exc:
            raw, status = exc.read(64 * 1024 + 1), exc.code
        if status != 200 or len(raw) > 4 * 1024 * 1024:
            raise AddonQualificationError("kubernetes_read_failed")
        value = probe._load_json(raw, code="kubernetes_response_invalid")  # noqa: SLF001
        return value


def validate_controllers(reader: Any) -> list[dict[str, str]]:
    """Require the two exact UID-bound controllers to be live and restart-free."""

    validated: list[dict[str, str]] = []
    for lane, expected in EXPECTED_CONTROLLERS.items():
        job = reader.get(
            f"/apis/batch/v1/namespaces/{NAMESPACE}/jobs/{expected['job_name']}"
        )
        selector = urllib.parse.quote(f"job-name={expected['job_name']}", safe="")
        pods = reader.get(f"/api/v1/namespaces/{NAMESPACE}/pods?labelSelector={selector}")
        items = pods.get("items")
        if (
            (job.get("metadata") or {}).get("uid") != expected["job_uid"]
            or (job.get("metadata") or {}).get("name") != expected["job_name"]
            or (job.get("status") or {}).get("active") != 1
            or (job.get("status") or {}).get("failed", 0) != 0
            or (job.get("status") or {}).get("succeeded", 0) != 0
            or not isinstance(items, list)
            or len(items) != 1
        ):
            raise AddonQualificationError("controller_job_identity_or_state_invalid")
        pod = items[0]
        metadata, status = pod.get("metadata") or {}, pod.get("status") or {}
        owners = metadata.get("ownerReferences") or []
        statuses = status.get("containerStatuses") or []
        if (
            metadata.get("uid") != expected["pod_uid"]
            or status.get("phase") != "Running"
            or not statuses
            or any(row.get("ready") is not True or row.get("restartCount") != 0 for row in statuses)
            or not any(
                owner.get("kind") == "Job" and owner.get("uid") == expected["job_uid"]
                for owner in owners
            )
        ):
            raise AddonQualificationError("controller_pod_identity_or_state_invalid")
        validated.append({"lane": lane, **expected})
    return validated


def validate_qwen_identity(api_key: str) -> dict[str, Any]:
    account = probe.get_json(probe.ACCOUNT_URL, api_key)
    if account.get("team_name") != "fleet" or account.get("team_id") != probe.FLEET_TEAM_ID:
        raise AddonQualificationError("fleet_team_identity_mismatch")
    roster = probe.get_json(probe.MODELS_URL, api_key)
    rows = roster.get("data")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise AddonQualificationError("model_roster_shape_mismatch")
    matches = [row for row in rows if row.get("id") == MODEL]
    if len(matches) != 1:
        raise AddonQualificationError("model_roster_identity_mismatch")
    values = sorted(set(probe._context_values(matches[0])))  # noqa: SLF001
    if values and values != [probe.EXPECTED_CONTEXT_LENGTH]:
        raise AddonQualificationError("model_roster_context_mismatch")
    return {
        "context_length_observable": bool(values),
        "observed_context_length": values[0] if values else None,
    }


def run(
    out_dir: Path,
    *,
    reader: Any | None = None,
    caller: probe.CompletionCaller = post_addon_completion,
    identity_checker: Any = validate_qwen_identity,
    lease_root: Path = LEASE_ROOT,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
    api_key = os.environ.get("FLEET_API_KEY", "")
    if not api_key:
        raise AddonQualificationError("fleet_api_key_absent")
    if os.environ.get("EXPECTED_SECRET_UID") != EXPECTED_SECRET_UID:
        raise AddonQualificationError("secret_uid_mismatch")
    job_uid = probe._safe_uuid(os.environ.get("JOB_UID", ""), "job_uid")  # noqa: SLF001
    pod_uid = probe._safe_uuid(os.environ.get("POD_UID", ""), "pod_uid")  # noqa: SLF001
    controllers = validate_controllers(reader or InClusterReader())
    context = identity_checker(api_key)

    with endpoint_lease.acquire_endpoint_leases(
        lease_root=lease_root,
        endpoint_key=LEASE_KEY,
        maximum_streams=4,
        count=2,
    ) as leases:
        if leases.slots != [3, 4]:
            raise AddonQualificationError("existing_controller_slots_not_exactly_one_and_two")
        wave = probe.run_wave(MODEL, 2, api_key, caller=caller)
        if (
            wave["streams_succeeded"] != 2
            or wave["requests_succeeded"] != 4
            or wave["protocol_valid_requests"] != 4
            or wave["errors"] != 0
        ):
            raise AddonQualificationError("addon_wave_failed")
        slots = leases.slots

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
            "addon_slots_acquired": slots,
            "maximum_total_owned_streams": 4,
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
            "forced_selection_order_per_stream": ["bash", "submit_report"],
            "tool_execution_performed": False,
        },
        "wave": wave,
        "request_counts": {
            "chat_completions": 4,
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
