"""Sequential exact-version Qwen Code evaluation on the sealed Fleet test split."""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import httpx

from evals.fleet import self_hosted

HOLDOUT_SCHEMA = "fleet-qwen-code-test20-plan-v1"
RECEIPT_SCHEMA = "fleet-qwen-code-test20-receipt-v1"
EXPECTED_SPLIT_SCHEMA = "fleet_rl_task_split_v1"
EXPECTED_TASK_COUNT = 20
BASE_EXECUTION_CONTROLS = {
    "pass_k": 1,
    "max_concurrent": 1,
    "training_data_eligible": False,
}
OPTIONAL_EXECUTION_CONTROLS = {
    "required_task_tools",
    "required_task_tool_catalog_sha256",
}


def _digest_without(value: dict[str, Any], field: str) -> str:
    unsigned = {key: item for key, item in value.items() if key != field}
    return self_hosted.sha256(self_hosted.canonical_json(unsigned))


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def validate_plan(plan: dict[str, Any], split: dict[str, Any]) -> list[dict[str, Any]]:
    if plan.get("schema_version") != HOLDOUT_SCHEMA:
        raise ValueError("unsupported holdout plan schema")
    if split.get("schema") != EXPECTED_SPLIT_SCHEMA:
        raise ValueError("unsupported task split schema")
    if split.get("manifest_digest") != _digest_without(split, "manifest_digest"):
        raise ValueError("task split manifest digest mismatch")
    if plan.get("split_manifest_digest") != split.get("manifest_digest"):
        raise ValueError("holdout plan does not pin this task split")
    if plan.get("task_count") != EXPECTED_TASK_COUNT:
        raise ValueError("holdout plan must declare exactly 20 tasks")
    if plan.get("source_job_id") != (split.get("source") or {}).get("job_id"):
        raise ValueError("holdout plan source job does not match split provenance")
    execution = plan.get("execution") or {}
    if any(execution.get(key) != value for key, value in BASE_EXECUTION_CONTROLS.items()):
        raise ValueError("holdout execution controls drifted")
    if set(execution) - (set(BASE_EXECUTION_CONTROLS) | OPTIONAL_EXECUTION_CONTROLS):
        raise ValueError("holdout execution controls contain unsupported fields")
    required_tools = execution.get("required_task_tools")
    required_tool_digest = execution.get("required_task_tool_catalog_sha256")
    if (required_tools is None) != (required_tool_digest is None):
        raise ValueError("holdout exact tool controls must be provided together")
    if required_tools is not None:
        if required_tools != ["bash", "submit_report"]:
            raise ValueError("holdout exact tool surface drifted")
        if not isinstance(required_tool_digest, str) or not required_tool_digest.startswith(
            "sha256:"
        ):
            raise ValueError("holdout exact tool catalog digest is invalid")
    resource_prefix = plan.get("resource_prefix", "q36qcode-test20-b1")
    if not isinstance(resource_prefix, str) or not re.fullmatch(
        r"[a-z0-9][a-z0-9-]{0,39}", resource_prefix
    ):
        raise ValueError("holdout resource prefix is invalid")

    rows = [row for row in split.get("tasks", []) if row.get("split") == "test"]
    if len(rows) != EXPECTED_TASK_COUNT:
        raise ValueError("task split does not contain exactly 20 test tasks")
    required = {
        "task_key",
        "task_version_id",
        "task_version",
        "environment_version_id",
        "env_key",
        "env_version",
        "data_key",
        "data_version",
    }
    for row in rows:
        if any(not row.get(field) for field in required):
            raise ValueError("test task binding is incomplete")
    if len({row["task_key"] for row in rows}) != EXPECTED_TASK_COUNT:
        raise ValueError("test task keys are not unique")
    if len({row["task_version_id"] for row in rows}) != EXPECTED_TASK_COUNT:
        raise ValueError("test task version ids are not unique")
    non_test = [row for row in split.get("tasks", []) if row.get("split") != "test"]
    if {row["task_key"] for row in rows} & {row["task_key"] for row in non_test}:
        raise ValueError("test task lineage overlaps another split")
    return sorted(rows, key=lambda row: row["task_key"])


def _gateway_request(
    client: httpx.Client, method: str, path: str, *, model: str | None = None, **kwargs: Any
) -> dict[str, Any]:
    headers = dict(kwargs.pop("headers", {}))
    if model:
        headers["X-Fleet-Model"] = model
    response = client.request(method, path, headers=headers, **kwargs)
    if response.status_code >= 400:
        raise RuntimeError(
            f"inference gateway {method} {path} failed with HTTP {response.status_code}"
        )
    value = response.json()
    if not isinstance(value, dict):
        raise RuntimeError("inference gateway returned a non-object response")
    return value


def live_model_identity(api_key: str, model: dict[str, Any]) -> dict[str, Any] | None:
    """Validate an optional exact self-hosted serving contract without exposing model output."""

    contract = model.get("live_identity")
    if contract is None:
        return None
    if not isinstance(contract, dict):
        raise ValueError("holdout live model identity contract is malformed")
    served_id = model["served_id"]
    origin = model["endpoint_origin"].rstrip("/")
    with httpx.Client(
        base_url=origin,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        timeout=180,
    ) as client:
        catalog = _gateway_request(client, "GET", "/fleet/v1/model-catalog")
        matches = [row for row in catalog.get("data", []) if row.get("id") == served_id]
        if len(matches) != 1:
            raise RuntimeError("holdout served model is missing or duplicated in the catalog")
        catalog_row = matches[0]
        expected_catalog = contract.get("catalog") or {}
        actual_catalog = {field: catalog_row.get(field) for field in expected_catalog}
        if actual_catalog != expected_catalog or int(catalog_row.get("ready_replicas") or 0) < 1:
            raise RuntimeError("holdout catalog model identity drifted")
        if "tool_calling" not in (catalog_row.get("capabilities") or []):
            raise RuntimeError("holdout served model no longer declares tool calling")

        model_info = _gateway_request(client, "GET", "/model_info", model=served_id)
        server_info = _gateway_request(client, "GET", "/server_info", model=served_id)
        expected_model_info = contract.get("model_info") or {}
        expected_server_info = contract.get("server_info") or {}
        actual_model_info = {field: model_info.get(field) for field in expected_model_info}
        actual_server_info = {field: server_info.get(field) for field in expected_server_info}
        if actual_model_info != expected_model_info:
            raise RuntimeError("holdout model_info identity drifted")
        if actual_server_info != expected_server_info:
            raise RuntimeError("holdout server_info identity drifted")

        probe = _gateway_request(
            client,
            "POST",
            "/v1/chat/completions",
            json={
                "model": served_id,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "For this protocol probe, call submit_report exactly once with "
                            "an empty flags list and explanation identity probe."
                        ),
                    }
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "description": "Run a shell command in an isolated challenge sandbox.",
                            "parameters": {
                                "type": "object",
                                "properties": {"command": {"type": "string"}},
                                "required": ["command"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "submit_report",
                            "description": "Submit recovered flags and an explanation.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "flags": {"type": "array", "items": {"type": "string"}},
                                    "explanation": {"type": "string"},
                                },
                                "required": ["flags", "explanation"],
                                "additionalProperties": False,
                            },
                        },
                    },
                ],
                "tool_choice": "auto",
                "temperature": 0,
                "max_tokens": 256,
            },
        )
    choices = probe.get("choices") or []
    calls = (
        (choices[0].get("message") or {}).get("tool_calls") or []
        if len(choices) == 1
        else []
    )
    call_names = [call.get("function", {}).get("name") for call in calls]
    if (
        len(choices) != 1
        or probe.get("model") != served_id
        or choices[0].get("finish_reason") != "tool_calls"
        or call_names != ["submit_report"]
    ):
        raise RuntimeError("holdout structured-tool identity probe failed")
    return {
        "catalog": actual_catalog,
        "model_info": actual_model_info,
        "server_info": actual_server_info,
        "structured_tool_probe": {
            "response_model": probe.get("model"),
            "finish_reason": choices[0].get("finish_reason"),
            "tool_names": call_names,
        },
    }


def _task_receipt(client: httpx.Client, row: dict[str, Any], index: int) -> dict[str, Any]:
    task = self_hosted._request(
        client,
        "GET",
        f"/v1/tasks/{row['task_key']}",
        params={"version_id": row["task_version_id"]},
    )
    actual_binding = {
        "task_key": task.get("key"),
        "env_key": task.get("environment_id"),
        "env_version": task.get("version"),
        "data_key": task.get("data_id"),
        "data_version": task.get("data_version"),
    }
    expected_binding = {field: row[field] for field in actual_binding}
    if actual_binding != expected_binding:
        raise RuntimeError(f"exact task binding drifted at test index {index}")
    verifier = task.get("verifier") or {}
    verifier_receipt = {
        "id": task.get("verifier_id"),
        "version_id": verifier.get("verifier_version_id"),
        "version": verifier.get("version"),
        "sha256": verifier.get("sha256"),
        "function_name": verifier.get("function_name") or "verify",
    }
    if any(value in (None, "") for value in verifier_receipt.values()):
        raise RuntimeError(f"verifier receipt is incomplete at test index {index}")
    runtime_seed = (task.get("metadata") or {}).get("runtime_seed_manifest") or {}
    return {
        "index": index,
        "task_key": row["task_key"],
        "task_version_id": row["task_version_id"],
        "task_version": row["task_version"],
        "environment_version_id": row["environment_version_id"],
        "env_key": row["env_key"],
        "env_version": row["env_version"],
        "data_key": row["data_key"],
        "data_version": row["data_version"],
        "prompt_sha256": self_hosted.sha256((task.get("prompt") or "").encode()),
        "env_variables_sha256": self_hosted.sha256(
            self_hosted.canonical_json(task.get("env_variables") or {})
        ),
        "output_json_schema_sha256": self_hosted.sha256(
            self_hosted.canonical_json(task.get("output_json_schema"))
        ),
        "runtime_seed_content_sha256": runtime_seed.get("content_sha256"),
        "runtime_seed_file_count": len(runtime_seed.get("files") or []),
        "verifier": verifier_receipt,
    }


def build_live_receipt(
    client: httpx.Client,
    plan: dict[str, Any],
    split: dict[str, Any],
    api_key: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = validate_plan(plan, split)
    account = self_hosted._request(client, "GET", "/v1/account")
    if account.get("team_name") != "fleet" or account.get("team_id") not in {
        None,
        self_hosted.FLEET_TEAM_ID,
    }:
        raise RuntimeError("FLEET_API_KEY is not scoped to the Fleet team")
    authority_gate = self_hosted.assert_authoritative_routes_deployed(client, plan)
    model_identity = live_model_identity(api_key, plan["model"]) if api_key else None
    if plan["model"].get("live_identity") is not None and model_identity is None:
        raise RuntimeError("holdout live model identity was not validated")
    tasks = [_task_receipt(client, row, index) for index, row in enumerate(rows, 1)]
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "campaign_id": plan["campaign_id"],
        "source_job_id": plan["source_job_id"],
        "split_manifest_digest": split["manifest_digest"],
        "task_count": len(tasks),
        "planned_sessions": len(tasks) * plan["execution"]["pass_k"],
        "model": copy.deepcopy(plan["model"]),
        "harness": copy.deepcopy(plan["harness"]),
        "authority": copy.deepcopy(plan["authority"]),
        "execution": copy.deepcopy(plan["execution"]),
        "tasks": tasks,
    }
    if model_identity is not None:
        receipt["live_model_identity"] = model_identity
    if "resource_prefix" in plan:
        receipt["resource_prefix"] = plan["resource_prefix"]
    if "launch_gate" in plan:
        receipt["launch_gate"] = copy.deepcopy(plan["launch_gate"])
    receipt["receipt_sha256"] = self_hosted.sha256(self_hosted.canonical_json(receipt))
    return receipt, authority_gate


def validate_frozen_receipt(receipt: dict[str, Any]) -> None:
    if receipt.get("schema_version") != RECEIPT_SCHEMA:
        raise ValueError("unsupported frozen receipt schema")
    if receipt.get("receipt_sha256") != _digest_without(receipt, "receipt_sha256"):
        raise ValueError("frozen receipt digest mismatch")
    if receipt.get("task_count") != EXPECTED_TASK_COUNT:
        raise ValueError("frozen receipt must contain exactly 20 tasks")
    if receipt.get("planned_sessions") != EXPECTED_TASK_COUNT:
        raise ValueError("frozen receipt must plan exactly 20 pass@1 sessions")
    tasks = receipt.get("tasks") or []
    if len(tasks) != EXPECTED_TASK_COUNT:
        raise ValueError("frozen receipt task rows are incomplete")
    if len({row["task_version_id"] for row in tasks}) != EXPECTED_TASK_COUNT:
        raise ValueError("frozen receipt task versions are not unique")


def task_config(plan: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    key_digest = self_hosted.sha256(row["task_key"].encode()).split(":", 1)[1][:8]
    resource_prefix = plan.get("resource_prefix", "q36qcode-test20-b1")
    return {
        "schema_version": "fleet-selfhosted-qwen-code-protocol-v1",
        "run_id": f"{plan['campaign_id']}-t{row['index']:02d}-{key_digest}",
        "source_job_id": plan["source_job_id"],
        "task": {
            "key": row["task_key"],
            "version_id": row["task_version_id"],
            "prompt_sha256": row["prompt_sha256"],
            "env_variables_sha256": row["env_variables_sha256"],
            "output_json_schema_sha256": row["output_json_schema_sha256"],
        },
        "environment": {
            "id": row["env_key"],
            "version": row["env_version"],
            "data_id": row["data_key"],
            "data_version": row["data_version"],
            "runtime_seed_content_sha256": row["runtime_seed_content_sha256"],
            "ttl_seconds": plan["environment_ttl_seconds"],
        },
        "verifier": copy.deepcopy(row["verifier"]),
        "authority": copy.deepcopy(plan["authority"]),
        "model": copy.deepcopy(plan["model"]),
        "harness": copy.deepcopy(plan["harness"]),
        "execution": {
            **copy.deepcopy(plan["execution"]),
            "network": f"{resource_prefix}-{row['index']:02d}-{key_digest}",
        },
    }


def run_campaign(
    plan: dict[str, Any],
    receipt: dict[str, Any],
    out_dir: Path,
    proxy_script: Path,
) -> dict[str, Any]:
    validate_frozen_receipt(receipt)
    out_dir.mkdir(parents=True, exist_ok=False)
    out_dir.chmod(0o700)
    outcomes = []
    for row in receipt["tasks"]:
        config = task_config(plan, row)
        task_out = out_dir / f"task-{row['index']:02d}-{config['run_id'].rsplit('-', 1)[-1]}"
        result = None
        try:
            result = self_hosted.run(config, task_out, proxy_script)
            cleanup = load_json(task_out / "cleanup.json")
            cleanup_verified = cleanup.get("containers_removed") is True and (
                cleanup.get("instance_created") is not True
                or cleanup.get("instance_closed") is True
            )
            if not cleanup_verified:
                raise RuntimeError("task cleanup receipt is incomplete")
            outcome = {
                "index": row["index"],
                "task_key": row["task_key"],
                "task_version_id": row["task_version_id"],
                "status": "model_outcome",
                **result,
            }
        except Exception as exc:  # noqa: BLE001
            outcome = {
                "index": row["index"],
                "task_key": row["task_key"],
                "task_version_id": row["task_version_id"],
            }
            if result is None:
                outcome.update(status="infrastructure_error", error_type=type(exc).__name__)
            else:
                outcome.update(
                    status="model_outcome",
                    **result,
                    cleanup_verified=False,
                    infrastructure_error="cleanup_incomplete",
                )
        outcomes.append(outcome)
        state = {
            "campaign_id": plan["campaign_id"],
            "planned_sessions": receipt["planned_sessions"],
            "completed_attempts": len(outcomes),
            "outcomes": outcomes,
        }
        (out_dir / "campaign-state.json").write_bytes(self_hosted.canonical_json(state) + b"\n")
        cleanup_path = task_out / "cleanup.json"
        if cleanup_path.exists():
            cleanup = load_json(cleanup_path)
            if (
                cleanup.get("instance_created") is True
                and cleanup.get("instance_closed") is not True
            ):
                break
    model_outcomes = [row for row in outcomes if row["status"] == "model_outcome"]
    infrastructure_errors = sum(
        row["status"] == "infrastructure_error" or "infrastructure_error" in row for row in outcomes
    )
    summary = {
        "campaign_id": plan["campaign_id"],
        "planned_sessions": receipt["planned_sessions"],
        "model_outcomes": len(model_outcomes),
        "infrastructure_errors": infrastructure_errors,
        "score_sum": sum(float(row["score"]) for row in model_outcomes),
        "outcomes": outcomes,
    }
    (out_dir / "summary.json").write_bytes(self_hosted.canonical_json(summary) + b"\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preflight", "run"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--receipt-out", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--proxy-script", type=Path)
    args = parser.parse_args()
    plan = load_json(args.config)
    split = load_json(args.split)
    api_key = os.environ.get("FLEET_API_KEY")
    if not api_key:
        raise RuntimeError("FLEET_API_KEY is required")
    with httpx.Client(
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        timeout=180,
    ) as client:
        live_receipt, authority_gate = build_live_receipt(client, plan, split, api_key)
    if args.receipt_out:
        args.receipt_out.write_bytes(self_hosted.canonical_json(live_receipt) + b"\n")
    if args.receipt:
        frozen = load_json(args.receipt)
        validate_frozen_receipt(frozen)
        if frozen != live_receipt:
            raise RuntimeError("live exact-task receipt does not match frozen receipt")
    if args.command == "preflight":
        print(
            json.dumps(
                {
                    "ok": True,
                    "campaign_id": plan["campaign_id"],
                    "planned_sessions": live_receipt["planned_sessions"],
                    "receipt_sha256": live_receipt["receipt_sha256"],
                    "authority_gate": authority_gate,
                    "model_revision": plan["model"]["revision"],
                    "harness_version": plan["harness"]["version"],
                    "max_concurrent": plan["execution"]["max_concurrent"],
                    "training_data_eligible": plan["execution"]["training_data_eligible"],
                },
                sort_keys=True,
            )
        )
        return 0
    if not args.receipt or not args.out_dir or not args.proxy_script:
        raise ValueError("run requires --receipt, --out-dir, and --proxy-script")
    summary = run_campaign(plan, live_receipt, args.out_dir, args.proxy_script)
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["infrastructure_errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
