"""Non-scored Fleet-team and hosted-model availability gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

FLEET_TEAM_ID = "a1025f0b-ad67-49fc-a023-51800ab43e84"
ACCOUNT_URL = "https://orchestrator.fleetai.com/v1/account"
MODELS_URL = "https://inference.flt.build/v1/models"
CAMPAIGN_SHA256 = "sha256:9df46502affc3ae892ead22fd13fa05e3e9f2950700f9ac67d76588bc29a6d96"
EXPECTED_CONTEXT_LENGTH = 262144
EXPECTED_MODELS = ("qwen3.8-27b", "glm-5.3")
CONTEXT_FIELDS = {
    "context_length",
    "context_window",
    "max_context_length",
    "max_model_len",
    "max_sequence_length",
}


class GateError(RuntimeError):
    """A sanitized qualification failure without an upstream response body."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def digest_without(value: dict[str, Any], field: str) -> str:
    body = {key: item for key, item in value.items() if key != field}
    return "sha256:" + hashlib.sha256(canonical_json(body)).hexdigest()


def write_json_once(path: Path, value: dict[str, Any]) -> None:
    payload = canonical_json(value) + b"\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            fd = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if fd >= 0:
            os.close(fd)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def fetch_json(
    name: str,
    url: str,
    api_key: str,
    *,
    opener: Callable[..., Any] = urlopen,
) -> dict[str, Any]:
    request = Request(
        url,
        method="GET",
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
    )
    try:
        with opener(request, timeout=30) as response:
            if response.status != 200:
                raise GateError(f"{name}_http_status")
            payload = response.read(4 * 1024 * 1024 + 1)
    except GateError:
        raise
    except Exception as exc:
        raise GateError(f"{name}_request_failed") from exc
    if len(payload) > 4 * 1024 * 1024:
        raise GateError(f"{name}_response_too_large")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GateError(f"{name}_invalid_json") from exc
    if not isinstance(value, dict):
        raise GateError(f"{name}_invalid_shape")
    return value


def _context_values(value: Any) -> list[int]:
    found: list[int] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key in CONTEXT_FIELDS and type(item) is int:
                found.append(item)
            else:
                found.extend(_context_values(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(_context_values(item))
    return found


def qualify(
    api_key: str,
    *,
    job_uid: str,
    pod_uid: str,
    secret_uid: str,
    opener: Callable[..., Any] = urlopen,
) -> dict[str, Any]:
    account = fetch_json("fleet_account", ACCOUNT_URL, api_key, opener=opener)
    if account.get("team_name") != "fleet" or account.get("team_id") != FLEET_TEAM_ID:
        raise GateError("fleet_team_identity_mismatch")

    roster = fetch_json("hosted_models", MODELS_URL, api_key, opener=opener)
    rows = roster.get("data")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise GateError("hosted_models_invalid_shape")
    observed: dict[str, dict[str, Any]] = {}
    for model in EXPECTED_MODELS:
        matches = [row for row in rows if row.get("id") == model]
        if len(matches) != 1:
            raise GateError(f"{model}_availability_mismatch")
        values = sorted(set(_context_values(matches[0])))
        if values and values != [EXPECTED_CONTEXT_LENGTH]:
            raise GateError(f"{model}_context_length_mismatch")
        observed[model] = {
            "available": True,
            "context_length_observable": bool(values),
            "context_length": values[0] if values else None,
        }

    receipt = {
        "schema_version": "fleet-opencode-autocontinue-hosted-health-v1",
        "status": "PASSED",
        "observed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "campaign_sha256": CAMPAIGN_SHA256,
        "fleet_account": {
            "team_name": "fleet",
            "team_id": FLEET_TEAM_ID,
            "authenticated_get_succeeded": True,
        },
        "hosted_inference": {
            "origin": "https://inference.flt.build",
            "authenticated_models_get_succeeded": True,
            "models": observed,
        },
        "runtime": {
            "job_uid": job_uid,
            "pod_uid": pod_uid,
            "secret_name": "chris-cyber-opencode-evals-v2",
            "secret_uid": secret_uid,
        },
        "request_counts": {
            "fleet_account_get": 1,
            "hosted_models_get": 1,
            "chat_completions": 0,
            "fleet_task_or_scoring": 0,
        },
        "scores_read_or_included": False,
        "prompts_traces_flags_or_tool_arguments_read_or_included": False,
        "credentials_included": False,
    }
    receipt["receipt_sha256"] = digest_without(receipt, "receipt_sha256")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    try:
        receipt = qualify(
            os.environ["FLEET_API_KEY"],
            job_uid=os.environ["JOB_UID"],
            pod_uid=os.environ["POD_UID"],
            secret_uid=os.environ["EXPECTED_SECRET_UID"],
        )
        write_json_once(args.out_dir / "TERMINAL.json", receipt)
        print("hosted endpoint health gate passed")
        return 0
    except Exception as exc:
        code = exc.code if isinstance(exc, GateError) else "unexpected_failure"
        failure = {
            "schema_version": "fleet-opencode-autocontinue-hosted-health-v1",
            "status": "FAILED",
            "observed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "campaign_sha256": CAMPAIGN_SHA256,
            "failure_code": code,
            "runtime": {
                "job_uid": os.environ.get("JOB_UID"),
                "pod_uid": os.environ.get("POD_UID"),
                "secret_name": "chris-cyber-opencode-evals-v2",
                "secret_uid": os.environ.get("EXPECTED_SECRET_UID"),
            },
            "request_policy": {
                "allowed_method": "GET",
                "allowed_urls": [ACCOUNT_URL, MODELS_URL],
                "chat_completions": 0,
                "fleet_task_or_scoring": 0,
            },
            "request_bodies_or_responses_included": False,
            "scores_read_or_included": False,
            "prompts_traces_flags_or_tool_arguments_read_or_included": False,
            "credentials_included": False,
        }
        failure["receipt_sha256"] = digest_without(failure, "receipt_sha256")
        write_json_once(args.out_dir / "TERMINAL.json", failure)
        print("hosted endpoint health gate failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
