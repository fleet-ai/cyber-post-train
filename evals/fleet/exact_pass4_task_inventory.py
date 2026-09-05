"""Hydrate the exact easiest-100 Fleet task inventory with GET requests only.

The corrected Qwen3.8/GLM5.3 campaign shares one immutable 100-task slate.
This observer validates that campaign and its complete 800-cell universe before
requesting every exact task version.  Task content is hashed in memory and is
never persisted; the terminal receipt contains only identities, digests, and
sanitized runtime bindings needed by later rollout plans.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener

from evals.fleet import exact_pass4_crypto as crypto
from evals.fleet import exact_pass4_universe as exact

FLEET_ORIGIN = "https://orchestrator.fleetai.com"
FLEET_TEAM_ID = "a1025f0b-ad67-49fc-a023-51800ab43e84"
CAMPAIGN_PATH = Path(
    "evals/fleet/configs/q38-glm53-exact-easiest100-pass4-campaign-v1.json"
)
EXPECTED_SCHEMA = "fleet-exact-easiest100-task-inventory-expected-v1"
RECEIPT_SCHEMA = "fleet-exact-easiest100-task-inventory-terminal-v1"
EXPECTED_TASK_FIELDS = {
    "selection_rank",
    "task_key",
    "task_version",
    "task_version_id",
    "split",
    "environment_version_id",
    "env_key",
    "env_version",
    "data_key",
    "data_version",
}
HEX_DIGEST_RE = re.compile(r"^(?:sha256:)?[0-9a-f]{64}$")


class GateError(RuntimeError):
    """A sanitized, stable failure code suitable for a terminal receipt."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class _RejectRedirects(HTTPRedirectHandler):
    """Reject redirects so bearer credentials cannot cross an origin boundary."""

    def redirect_request(
        self,
        req: Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        del req, fp, code, msg, headers, newurl
        raise GateError("fleet_get_redirect_rejected")


def canonical_json(value: Any) -> bytes:
    return crypto.canonical_json(value)


def sha256(value: bytes) -> str:
    return crypto.sha256(value)


def digest_without(value: dict[str, Any], field: str) -> str:
    return crypto.digest_without(value, field)


def _uuid(value: Any, code: str) -> str:
    if not isinstance(value, str):
        raise GateError(code)
    try:
        return str(uuid.UUID(value))
    except ValueError as exc:
        raise GateError(code) from exc


def _hex_digest(value: Any, code: str) -> str:
    if not isinstance(value, str) or HEX_DIGEST_RE.fullmatch(value) is None:
        raise GateError(code)
    return value


def write_once(path: Path, value: dict[str, Any]) -> None:
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


def prepare_expected(root: Path) -> dict[str, Any]:
    """Build a deterministic inventory expectation from the corrected universe."""
    campaign_path = root / CAMPAIGN_PATH
    try:
        campaign = exact.read_object(campaign_path)
        universe = exact.build_universe(campaign, root)
        selection = exact.validate_campaign(campaign, root)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise GateError("campaign_or_universe_invalid") from exc

    universe_tasks = {
        (cell["selection_rank"], cell["task_key"], cell["task_version_id"])
        for cell in universe["cells"]
    }
    rows = [
        {
            "selection_rank": row["rank"],
            "task_key": row["task_key"],
            "task_version": row["task_version"],
            "task_version_id": row["task_version_id"],
            "split": row["split"],
            "environment_version_id": row["environment_version_id"],
            "env_key": row["env_key"],
            "env_version": row["env_version"],
            "data_key": row["data_key"],
            "data_version": row["data_version"],
        }
        for row in selection
    ]
    selected_tasks = {
        (row["selection_rank"], row["task_key"], row["task_version_id"]) for row in rows
    }
    if universe_tasks != selected_tasks:
        raise GateError("universe_task_identity_mismatch")

    expected = {
        "schema_version": EXPECTED_SCHEMA,
        "campaign_path": str(CAMPAIGN_PATH),
        "campaign_id": campaign["campaign_id"],
        "campaign_sha256": sha256(canonical_json(campaign)),
        "universe_sha256": universe["universe_sha256"],
        "selection_sha256": universe["selection_sha256"],
        "models": campaign["models"],
        "task_count": 100,
        "cell_counts": universe["model_counts"],
        "total_cell_count": universe["cell_count"],
        "attempts": universe["attempts"],
        "tasks": rows,
        "request_policy": {
            "methods": ["GET"],
            "redirects_followed": 0,
            "account_get": 1,
            "exact_task_version_get": 100,
            "model_or_scoring_calls": 0,
            "session_calls": 0,
            "mutation_calls": 0,
        },
        "privacy": {
            "task_payloads_persisted": False,
            "task_content_included": False,
            "prompts_traces_flags_or_scores_included": False,
            "credentials_included": False,
        },
    }
    expected["expected_sha256"] = digest_without(expected, "expected_sha256")
    _validate_expected(expected)
    return expected


def _validate_expected(expected: dict[str, Any]) -> None:
    if expected.get("expected_sha256") != digest_without(expected, "expected_sha256"):
        raise GateError("expected_inventory_digest_mismatch")
    if expected.get("schema_version") != EXPECTED_SCHEMA:
        raise GateError("expected_schema_mismatch")
    if expected.get("campaign_id") != exact.EXPECTED_CAMPAIGN_ID:
        raise GateError("campaign_identity_mismatch")
    if expected.get("models") != exact.EXPECTED_MODELS:
        raise GateError("model_binding_mismatch")
    if (
        expected.get("selection_sha256") != exact.EXPECTED_SELECTION_SHA256
        or expected.get("task_count") != 100
        or expected.get("cell_counts") != {"qwen3.8-27b": 400, "glm-5.3": 400}
        or expected.get("total_cell_count") != 800
        or expected.get("attempts") != [1, 2, 3, 4]
    ):
        raise GateError("expected_denominator_mismatch")
    if expected.get("request_policy") != {
        "methods": ["GET"],
        "redirects_followed": 0,
        "account_get": 1,
        "exact_task_version_get": 100,
        "model_or_scoring_calls": 0,
        "session_calls": 0,
        "mutation_calls": 0,
    }:
        raise GateError("request_policy_mismatch")
    rows = expected.get("tasks")
    if not isinstance(rows, list) or len(rows) != 100:
        raise GateError("expected_task_count_mismatch")
    if [row.get("selection_rank") for row in rows if isinstance(row, dict)] != list(
        range(1, 101)
    ):
        raise GateError("expected_rank_coverage_mismatch")
    identities: set[tuple[str, str]] = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != EXPECTED_TASK_FIELDS:
            raise GateError("expected_task_shape_mismatch")
        if row.get("split") != "train":
            raise GateError("expected_split_mismatch")
        task_key = row.get("task_key")
        if not isinstance(task_key, str) or not task_key:
            raise GateError("expected_task_key_invalid")
        task_version_id = _uuid(row.get("task_version_id"), "expected_task_version_invalid")
        _uuid(row.get("environment_version_id"), "expected_environment_version_invalid")
        if any(
            not isinstance(row.get(field), str) or not row[field]
            for field in (
                "task_version",
                "env_key",
                "env_version",
                "data_key",
                "data_version",
            )
        ):
            raise GateError("expected_task_binding_incomplete")
        identity = (task_key, task_version_id)
        if identity in identities:
            raise GateError("expected_task_identity_duplicate")
        identities.add(identity)


def fetch_json(url: str, api_key: str) -> dict[str, Any]:
    """Perform one bounded GET; no generic method argument is accepted."""
    request = Request(
        url,
        method="GET",
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
    )
    try:
        with build_opener(_RejectRedirects()).open(request, timeout=30) as response:
            if response.status != 200:
                raise GateError("fleet_get_http_status")
            payload = response.read(8 * 1024 * 1024 + 1)
    except GateError:
        raise
    except Exception as exc:
        raise GateError("fleet_get_failed") from exc
    if len(payload) > 8 * 1024 * 1024:
        raise GateError("fleet_get_response_too_large")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GateError("fleet_get_invalid_json") from exc
    if not isinstance(value, dict):
        raise GateError("fleet_get_invalid_shape")
    return value


def _hydrate_task(task: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    response_version_id = task.get("version_id")
    if response_version_id not in (None, expected["task_version_id"]):
        raise GateError("live_task_version_mismatch")
    response_task_version = task.get("task_version")
    if response_task_version not in (None, expected["task_version"]):
        raise GateError("live_task_version_mismatch")
    actual_identity = {
        "task_key": task.get("key"),
        "env_key": task.get("environment_id"),
        "env_version": task.get("version"),
        "data_key": task.get("data_id"),
        "data_version": task.get("data_version"),
    }
    wanted_identity = {field: expected[field] for field in actual_identity}
    if actual_identity != wanted_identity:
        raise GateError("live_task_binding_mismatch")

    prompt = task.get("prompt")
    env_variables = task.get("env_variables")
    output_json_schema = task.get("output_json_schema")
    if not isinstance(prompt, str) or not prompt:
        raise GateError("live_prompt_invalid")
    if not isinstance(env_variables, dict):
        raise GateError("live_env_variables_invalid")
    # Report-only cyber task versions legitimately expose JSON null here. Bind
    # that exact value by digest, while rejecting every other non-object shape.
    if output_json_schema is not None and not isinstance(output_json_schema, dict):
        raise GateError("live_output_json_schema_invalid")

    metadata = task.get("metadata")
    if not isinstance(metadata, dict):
        raise GateError("live_task_metadata_missing")
    cyber_contract = metadata.get("cyber_contract")
    if cyber_contract != exact.EXPECTED_CYBER_CONTRACT:
        raise GateError("live_cyber_contract_mismatch")
    runtime_seed = metadata.get("runtime_seed_manifest")
    if not isinstance(runtime_seed, dict):
        raise GateError("runtime_seed_manifest_incomplete")
    runtime_seed_digest = _hex_digest(
        runtime_seed.get("content_sha256"), "runtime_seed_manifest_incomplete"
    )
    runtime_seed_files = runtime_seed.get("files")
    if not isinstance(runtime_seed_files, list) or not runtime_seed_files:
        raise GateError("runtime_seed_manifest_incomplete")
    for file_row in runtime_seed_files:
        if not isinstance(file_row, dict) or any(
            not isinstance(file_row.get(field), str) or not file_row[field]
            for field in ("target_path", "s3_key", "bucket")
        ):
            raise GateError("runtime_seed_manifest_incomplete")

    verifier = task.get("verifier")
    if not isinstance(verifier, dict):
        raise GateError("verifier_binding_incomplete")
    verifier_receipt = {
        "id": _uuid(task.get("verifier_id"), "verifier_binding_incomplete"),
        "version_id": _uuid(
            verifier.get("verifier_version_id"), "verifier_binding_incomplete"
        ),
        "version": verifier.get("version"),
        "sha256": _hex_digest(verifier.get("sha256"), "verifier_binding_incomplete"),
        "function_name": verifier.get("function_name") or "verify",
    }
    if (
        not isinstance(verifier_receipt["version"], int)
        or isinstance(verifier_receipt["version"], bool)
        or verifier_receipt["version"] < 1
        or not isinstance(verifier_receipt["function_name"], str)
        or not verifier_receipt["function_name"]
    ):
        raise GateError("verifier_binding_incomplete")

    return {
        "selection_rank": expected["selection_rank"],
        "task": {
            "key": expected["task_key"],
            "version": expected["task_version"],
            "version_id": expected["task_version_id"],
            "prompt_sha256": sha256(prompt.encode()),
            "env_variables_sha256": sha256(canonical_json(env_variables)),
            "output_json_schema_sha256": sha256(canonical_json(output_json_schema)),
            "cyber_contract": cyber_contract,
        },
        "environment": {
            "id": expected["env_key"],
            "version": expected["env_version"],
            "version_id": expected["environment_version_id"],
            "version_id_authority": "immutable_selection",
            "data_id": expected["data_key"],
            "data_version": expected["data_version"],
            "runtime_seed_content_sha256": runtime_seed_digest,
            "runtime_seed_file_count": len(runtime_seed_files),
        },
        "verifier": verifier_receipt,
        "exact_version_request": True,
        "response_version_id_observable": response_version_id is not None,
    }


def observe(
    expected: dict[str, Any],
    api_key: str,
    *,
    root: Path,
    job_uid: str,
    pod_uid: str,
    observed_at: str | None = None,
) -> dict[str, Any]:
    """Verify and hydrate all 100 tasks without making any write or score call."""
    _validate_expected(expected)
    if expected != prepare_expected(root):
        raise GateError("expected_source_mismatch")
    normalized_job_uid = _uuid(job_uid, "runtime_job_uid_invalid")
    normalized_pod_uid = _uuid(pod_uid, "runtime_pod_uid_invalid")
    account = fetch_json(f"{FLEET_ORIGIN}/v1/account", api_key)
    if account.get("team_name") != "fleet" or account.get("team_id") != FLEET_TEAM_ID:
        raise GateError("fleet_team_identity_mismatch")

    hydrated: list[dict[str, Any]] = []
    task_get_count = 0
    for row in expected["tasks"]:
        path = (
            f"/v1/tasks/{quote(row['task_key'], safe='')}?"
            f"{urlencode({'version_id': row['task_version_id']})}"
        )
        task = fetch_json(FLEET_ORIGIN + path, api_key)
        task_get_count += 1
        hydrated.append(_hydrate_task(task, row))
    if len(hydrated) != 100 or {row["selection_rank"] for row in hydrated} != set(
        range(1, 101)
    ):
        raise GateError("live_task_count_mismatch")

    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "PASSED",
        "observed_at": observed_at
        or datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "campaign_id": expected["campaign_id"],
        "campaign_sha256": expected["campaign_sha256"],
        "universe_sha256": expected["universe_sha256"],
        "selection_sha256": expected["selection_sha256"],
        "models": expected["models"],
        "expected_inventory_sha256": expected["expected_sha256"],
        "task_count": 100,
        "cell_counts": expected["cell_counts"],
        "total_cell_count": expected["total_cell_count"],
        "attempts": expected["attempts"],
        "tasks": hydrated,
        "task_bindings_sha256": sha256(canonical_json(hydrated)),
        "binding_mismatch_count": 0,
        "fleet_account": {"team_name": "fleet", "team_id": FLEET_TEAM_ID},
        "request_counts": {
            "account_get": 1,
            "exact_task_version_get": task_get_count,
            "redirects_followed": 0,
            "post_put_patch_delete": 0,
            "model_or_scoring_calls": 0,
            "session_calls": 0,
        },
        "runtime": {"job_uid": normalized_job_uid, "pod_uid": normalized_pod_uid},
        "privacy": expected["privacy"],
    }
    receipt["receipt_sha256"] = digest_without(receipt, "receipt_sha256")
    return receipt


def _failure_receipt(code: str, *, job_uid: str | None, pod_uid: str | None) -> dict[str, Any]:
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "FAILED",
        "observed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "failure_code": code,
        "runtime": {"job_uid": job_uid, "pod_uid": pod_uid},
        "request_policy": {
            "methods": ["GET"],
            "redirects_followed": 0,
            "model_or_scoring_calls": 0,
            "session_calls": 0,
            "mutation_calls": 0,
        },
        "privacy": {
            "task_payloads_persisted": False,
            "task_content_included": False,
            "prompts_traces_flags_or_scores_included": False,
            "credentials_included": False,
        },
    }
    receipt["receipt_sha256"] = digest_without(receipt, "receipt_sha256")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--prepare", type=Path)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    if args.prepare:
        write_once(args.prepare, prepare_expected(args.root))
        return 0
    if not args.out_dir:
        parser.error("observer requires --out-dir")
    args.out_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    try:
        expected = prepare_expected(args.root)
        receipt = observe(
            expected,
            os.environ["FLEET_API_KEY"],
            root=args.root,
            job_uid=os.environ["JOB_UID"],
            pod_uid=os.environ["POD_UID"],
        )
        write_once(args.out_dir / "TERMINAL.json", receipt)
        print("exact easiest-100 task inventory passed")
        return 0
    except Exception as exc:
        code = exc.code if isinstance(exc, GateError) else "unexpected_failure"
        write_once(
            args.out_dir / "TERMINAL.json",
            _failure_receipt(
                code,
                job_uid=os.environ.get("JOB_UID"),
                pod_uid=os.environ.get("POD_UID"),
            ),
        )
        print("exact easiest-100 task inventory failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
