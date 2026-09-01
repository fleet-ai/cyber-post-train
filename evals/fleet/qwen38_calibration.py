"""Pass@1 Qwen3.8-27B reward calibration on exact non-test Fleet tasks.

The first selected task is an in-protocol canary. The remaining nineteen tasks
start only after that attempt produces a positive authoritative reward and a
complete cleanup receipt. External benchmark and sealed Fleet test rows are
never read by this runner.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from uuid import UUID

import httpx

from evals.fleet import holdout, self_hosted

PLAN_SCHEMA = "fleet-qwen38-code-reward-calibration-plan-v1"
RECEIPT_SCHEMA = "fleet-qwen38-code-reward-calibration-receipt-v1"
EXPECTED_SPLIT_SCHEMA = "fleet_rl_task_split_v1"
EXPECTED_TASK_COUNT = 20
EXPECTED_PASS_K = 1
EXPECTED_RELEASE_GATE = {
    "task_index": 1,
    "criterion": "positive_authoritative_reward_with_valid_cleanup_on_non_test_fleet_tasks",
    "required_outcome_status": "model_outcome",
    "require_exact_canary_binding": True,
    "required_verifier_execution_id": "nonzero_uuid",
    "required_cleanup_verified": True,
    "required_binary_score": 1,
}
EXPECTED_PREDECESSOR_INCIDENT = {
    "campaign_id": "chris-cyber-q38-qcode-reward-cal-p1-v2",
    "classification": "terminal_infrastructure_interrupted",
    "scientific_disposition": "descriptive_only_not_training_gate_evidence",
    "incident_binding": {
        "path": (
            "docs/evidence/qwen38-study/"
            "2026-09-01-fleet-calibration-v2-infrastructure-incident.json"
        ),
        "binding_receipt_sha256": (
            "sha256:6f5b609d139c87a015263feac5f099df1a83e002d3146e038a98087ec8f9a1ad"
        ),
        "source_sanitized_receipt_raw_file_sha256": (
            "sha256:5ca91ad5db8d951d545cd78a4ea06982941fced369c54c48b6593c00f401235a"
        ),
        "source_sanitized_receipt_embedded_sha256": (
            "sha256:8f7a5c7842e499cce5618b563aae848cf5ddac26c7f9b359f941d2aff4b0c111"
        ),
        "source_sanitized_receipt_computed_canonical_sha256": (
            "sha256:507e8faecc6c686ca90ff6196e7b03ba95acb66543bcfcb547473c1225ae1e58"
        ),
        "source_sanitized_receipt_embedded_digest_valid": False,
    },
    "outcome_accounting": {
        "valid_scored_outcomes": 1,
        "valid_zero_outcomes": 1,
        "positive_reward_outcomes": 0,
        "unresolved_attempts_excluded_from_outcomes": 3,
        "never_launched_attempts": 16,
    },
    "positive_reward_gate_satisfied": False,
    "valid_scored_attempts_must_not_be_rerun": True,
    "unresolved_attempts_must_remain_excluded": True,
}
EXPECTED_TOOL_NAMES = ["bash", "submit_report"]
EXPECTED_TOOL_CATALOG_SHA256 = (
    "sha256:85fad6bdc3a835bf52a11a99b3387740eb06eb3d1720ad9bb33f3feac215b44a"
)
EXPECTED_FAMILY_COUNTS = {
    "current": 4,
    "fakelook": 4,
    "fentry": 3,
    "fira": 3,
    "fubspot": 3,
    "rops": 3,
}


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _digest_without(value: dict[str, Any], field: str) -> str:
    unsigned = {key: item for key, item in value.items() if key != field}
    return self_hosted.sha256(self_hosted.canonical_json(unsigned))


def validate_plan(plan: dict[str, Any], split: dict[str, Any]) -> list[dict[str, Any]]:
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("unsupported Qwen3.8 calibration plan schema")
    if split.get("schema") != EXPECTED_SPLIT_SCHEMA:
        raise ValueError("unsupported task split schema")
    if split.get("manifest_digest") != _digest_without(split, "manifest_digest"):
        raise ValueError("task split manifest digest mismatch")
    if plan.get("split_manifest_digest") != split.get("manifest_digest"):
        raise ValueError("calibration plan does not pin this task split")
    if plan.get("source_job_id") != (split.get("source") or {}).get("job_id"):
        raise ValueError("calibration source job does not match split provenance")
    if plan.get("study_role") != "reward_acquisition_model_calibration":
        raise ValueError("calibration study role drifted")
    if plan.get("release_gate") != EXPECTED_RELEASE_GATE:
        raise ValueError(
            "calibration release gate must explicitly require positive authoritative reward "
            "with valid cleanup"
        )
    if plan.get("supersedes_protocol") != EXPECTED_PREDECESSOR_INCIDENT:
        raise ValueError("calibration predecessor incident binding drifted")
    if plan.get("execution") != {
        "pass_k": EXPECTED_PASS_K,
        "max_concurrent": 3,
        "training_data_eligible": False,
        "required_task_tools": EXPECTED_TOOL_NAMES,
        "required_task_tool_catalog_sha256": EXPECTED_TOOL_CATALOG_SHA256,
    }:
        raise ValueError("calibration execution controls drifted")
    if (plan.get("authority") or {}).get("scoring_mode") != "binary":
        raise ValueError("calibration must use binary verifier rewards")
    if (plan.get("authority") or {}).get("multi_app_aggregation_mode") != "binary":
        raise ValueError("calibration must use binary multi-app aggregation")
    harness = plan.get("harness") or {}
    if harness.get("name") != "qwen_code" or harness.get("version") != "0.22.3":
        raise ValueError("calibration must use Qwen Code 0.22.3")
    if harness.get("max_model_requests") != 600:
        raise ValueError("calibration must preserve the 600-request horizon")
    if harness.get("context_window_size") != 262144:
        raise ValueError("calibration must preserve the 262144-token context window")

    allowed = {
        row["task_version_id"]: row
        for row in split.get("tasks", [])
        if row.get("split") in {"train", "dev"}
    }
    sealed_test = {
        row["task_version_id"] for row in split.get("tasks", []) if row.get("split") == "test"
    }
    selected = plan.get("tasks") or []
    if len(selected) != EXPECTED_TASK_COUNT:
        raise ValueError("calibration must select exactly 20 tasks")
    ids = [row.get("task_version_id") for row in selected]
    if len(set(ids)) != EXPECTED_TASK_COUNT:
        raise ValueError("calibration task versions are not unique")
    if set(ids) & sealed_test:
        raise ValueError("calibration selected a sealed test task")
    family_counts = {
        family: sum(row.get("family") == family for row in selected)
        for family in EXPECTED_FAMILY_COUNTS
    }
    if family_counts != EXPECTED_FAMILY_COUNTS:
        raise ValueError("calibration family balance drifted")

    rows: list[dict[str, Any]] = []
    for index, selected_row in enumerate(selected, 1):
        version_id = selected_row.get("task_version_id")
        row = allowed.get(version_id)
        if row is None:
            raise ValueError("calibration selected an unknown or disallowed task version")
        if row["task_key"] != selected_row.get("task_key"):
            raise ValueError("calibration task key/version binding drifted")
        rows.append(
            {
                **copy.deepcopy(row),
                "index": index,
                "family": selected_row["family"],
            }
        )
    return rows


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


def live_model_identity(api_key: str, model: dict[str, Any]) -> dict[str, Any]:
    origin = model["endpoint_origin"].rstrip("/")
    served_id = model["served_id"]
    revision = model["revision"]
    with httpx.Client(
        base_url=origin,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        timeout=180,
    ) as client:
        catalog = _gateway_request(client, "GET", "/fleet/v1/model-catalog")
        matches = [row for row in catalog.get("data", []) if row.get("id") == served_id]
        if len(matches) != 1:
            raise RuntimeError("Qwen3.8 served id is missing or duplicated in the model catalog")
        row = matches[0]
        expected_catalog = {
            "id": served_id,
            "model_revision": revision,
            "engine": "sglang",
            "precision": "bf16",
            "tensor_parallel_size": 1,
            "routed": True,
            "status": "ready",
        }
        actual_catalog = {field: row.get(field) for field in expected_catalog}
        if actual_catalog != expected_catalog or int(row.get("ready_replicas") or 0) < 1:
            raise RuntimeError("Qwen3.8 catalog identity or readiness drifted")
        if "tool_calling" not in (row.get("capabilities") or []):
            raise RuntimeError("Qwen3.8 catalog no longer declares tool calling")

        model_info = _gateway_request(client, "GET", "/model_info", model=served_id)
        server_info = _gateway_request(client, "GET", "/server_info", model=served_id)
        expected_path = f"/scratch/models/qwen3.8-27b/{revision}"
        model_fields = {
            "model_path": model_info.get("model_path"),
            "model_type": model_info.get("model_type"),
            "architectures": model_info.get("architectures"),
        }
        server_fields = {
            "model_path": server_info.get("model_path"),
            "served_model_name": server_info.get("served_model_name"),
            "context_length": server_info.get("context_length"),
            "tp_size": server_info.get("tp_size"),
            "quantization": server_info.get("quantization"),
            "kv_cache_dtype": server_info.get("kv_cache_dtype"),
            "reasoning_parser": server_info.get("reasoning_parser"),
            "tool_call_parser": server_info.get("tool_call_parser"),
        }
        if model_fields != {
            "model_path": expected_path,
            "model_type": "qwen3_5",
            "architectures": ["Qwen3_5ForConditionalGeneration"],
        }:
            raise RuntimeError("Qwen3.8 model_info identity drifted")
        if server_fields != {
            "model_path": expected_path,
            "served_model_name": served_id,
            "context_length": 262144,
            "tp_size": 1,
            "quantization": None,
            "kv_cache_dtype": "fp8_e4m3",
            "reasoning_parser": "qwen3",
            "tool_call_parser": "qwen3_coder",
        }:
            raise RuntimeError("Qwen3.8 server_info identity drifted")

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
    if len(choices) != 1:
        raise RuntimeError("Qwen3.8 structured-tool probe returned an unexpected choice count")
    choice = choices[0]
    calls = (choice.get("message") or {}).get("tool_calls") or []
    call_names = [call.get("function", {}).get("name") for call in calls]
    if choice.get("finish_reason") != "tool_calls" or call_names != ["submit_report"]:
        raise RuntimeError("Qwen3.8 structured-tool probe failed")
    return {
        "catalog": actual_catalog,
        "model_info": model_fields,
        "server_info": server_fields,
        "structured_tool_probe": {
            "response_model": probe.get("model"),
            "finish_reason": choice.get("finish_reason"),
            "tool_names": call_names,
        },
    }


def build_live_receipt(
    orchestrator: httpx.Client,
    plan: dict[str, Any],
    split: dict[str, Any],
    api_key: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = validate_plan(plan, split)
    account = self_hosted._request(orchestrator, "GET", "/v1/account")
    if account.get("team_name") != "fleet" or account.get("team_id") not in {
        None,
        self_hosted.FLEET_TEAM_ID,
    }:
        raise RuntimeError("FLEET_API_KEY is not scoped to the Fleet team")
    authority_gate = self_hosted.assert_authoritative_routes_deployed(orchestrator, plan)
    gateway_identity = live_model_identity(api_key, plan["model"])
    tasks = []
    for row in rows:
        receipt = holdout._task_receipt(orchestrator, row, row["index"])
        receipt["family"] = row["family"]
        receipt["split"] = row["split"]
        tasks.append(receipt)
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "campaign_id": plan["campaign_id"],
        "study_role": plan["study_role"],
        "retry_of": copy.deepcopy(plan.get("retry_of")),
        "source_job_id": plan["source_job_id"],
        "split_manifest_digest": split["manifest_digest"],
        "task_count": len(tasks),
        "planned_sessions": len(tasks),
        "model": copy.deepcopy(plan["model"]),
        "live_model_identity": gateway_identity,
        "harness": copy.deepcopy(plan["harness"]),
        "authority": copy.deepcopy(plan["authority"]),
        "execution": copy.deepcopy(plan["execution"]),
        "release_gate": copy.deepcopy(plan["release_gate"]),
        "supersedes_protocol": copy.deepcopy(plan["supersedes_protocol"]),
        "comparison": copy.deepcopy(plan["comparison"]),
        "known_non_parity": copy.deepcopy(plan["known_non_parity"]),
        "tasks": tasks,
    }
    receipt["receipt_sha256"] = self_hosted.sha256(self_hosted.canonical_json(receipt))
    return receipt, authority_gate


def validate_frozen_receipt(receipt: dict[str, Any]) -> None:
    if receipt.get("schema_version") != RECEIPT_SCHEMA:
        raise ValueError("unsupported Qwen3.8 calibration receipt schema")
    if receipt.get("receipt_sha256") != _digest_without(receipt, "receipt_sha256"):
        raise ValueError("frozen Qwen3.8 calibration receipt digest mismatch")
    if receipt.get("task_count") != EXPECTED_TASK_COUNT:
        raise ValueError("frozen receipt must contain exactly 20 tasks")
    if receipt.get("planned_sessions") != EXPECTED_TASK_COUNT:
        raise ValueError("frozen receipt must plan exactly 20 pass@1 sessions")
    if receipt.get("release_gate") != EXPECTED_RELEASE_GATE:
        raise ValueError("frozen receipt does not bind the positive-reward release gate")
    if receipt.get("supersedes_protocol") != EXPECTED_PREDECESSOR_INCIDENT:
        raise ValueError("frozen receipt does not bind the predecessor incident")
    tasks = receipt.get("tasks") or []
    if len(tasks) != EXPECTED_TASK_COUNT:
        raise ValueError("frozen receipt task rows are incomplete")
    if len({row["task_version_id"] for row in tasks}) != EXPECTED_TASK_COUNT:
        raise ValueError("frozen receipt task versions are not unique")
    if {row["split"] for row in tasks} - {"train", "dev"}:
        raise ValueError("frozen receipt contains a sealed test task")


def task_config(plan: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    key_digest = self_hosted.sha256(row["task_key"].encode()).split(":", 1)[1][:8]
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
            "network": f"q38cal-t{row['index']:02d}-{key_digest}",
        },
    }


def _one_task(
    plan: dict[str, Any], row: dict[str, Any], out_dir: Path, proxy_script: Path
) -> dict[str, Any]:
    config = task_config(plan, row)
    task_out = out_dir / f"task-{row['index']:02d}"
    try:
        result = self_hosted.run(config, task_out, proxy_script)
        cleanup = load_json(task_out / "cleanup.json")
        if cleanup.get("containers_removed") is not True or (
            cleanup.get("instance_created") is True and cleanup.get("instance_closed") is not True
        ):
            raise RuntimeError("task cleanup receipt is incomplete")
        return {
            "index": row["index"],
            "family": row["family"],
            "split": row["split"],
            "task_key": row["task_key"],
            "task_version_id": row["task_version_id"],
            "status": "model_outcome",
            "cleanup_verified": True,
            **result,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "index": row["index"],
            "family": row["family"],
            "split": row["split"],
            "task_key": row["task_key"],
            "task_version_id": row["task_version_id"],
            "status": "infrastructure_error",
            "error_type": type(exc).__name__,
        }


def _write_state(out_dir: Path, plan: dict[str, Any], outcomes: list[dict[str, Any]]) -> None:
    value = {
        "campaign_id": plan["campaign_id"],
        "planned_sessions": EXPECTED_TASK_COUNT,
        "completed_attempts": len(outcomes),
        "outcomes": sorted(outcomes, key=lambda row: row["index"]),
    }
    (out_dir / "campaign-state.json").write_bytes(self_hosted.canonical_json(value) + b"\n")


def can_release_remaining(plan: dict[str, Any], outcome: dict[str, Any]) -> bool:
    """Apply the preregistered canary gate without permissive fallbacks."""

    if plan.get("release_gate") != EXPECTED_RELEASE_GATE:
        raise ValueError("unsupported or missing calibration release gate")
    if outcome.get("status") != EXPECTED_RELEASE_GATE["required_outcome_status"]:
        return False
    if outcome.get("cleanup_verified") is not True:
        return False
    if outcome.get("index") != EXPECTED_RELEASE_GATE["task_index"]:
        return False
    selected = plan.get("tasks") or []
    if not selected or any(
        outcome.get(field) != selected[0].get(field) for field in ("task_key", "task_version_id")
    ):
        return False
    verifier_execution_id = outcome.get("verifier_execution_id")
    if not isinstance(verifier_execution_id, str):
        return False
    try:
        if UUID(verifier_execution_id).int == 0:
            return False
    except ValueError:
        return False
    try:
        score = float(outcome["score"])
    except (KeyError, TypeError, ValueError):
        return False
    return score == float(EXPECTED_RELEASE_GATE["required_binary_score"])


def run_campaign(
    plan: dict[str, Any], receipt: dict[str, Any], out_dir: Path, proxy_script: Path
) -> dict[str, Any]:
    validate_frozen_receipt(receipt)
    out_dir.mkdir(parents=True, exist_ok=False)
    out_dir.chmod(0o700)
    outcomes = [_one_task(plan, receipt["tasks"][0], out_dir, proxy_script)]
    _write_state(out_dir, plan, outcomes)
    canary_passed = can_release_remaining(plan, outcomes[0])

    if canary_passed:
        remaining = receipt["tasks"][1:]
        width = plan["execution"]["max_concurrent"]
        with ThreadPoolExecutor(max_workers=width) as pool:
            for start in range(0, len(remaining), width):
                wave = remaining[start : start + width]
                futures = {
                    pool.submit(_one_task, plan, row, out_dir, proxy_script): row for row in wave
                }
                wave_outcomes = [future.result() for future in as_completed(futures)]
                outcomes.extend(wave_outcomes)
                _write_state(out_dir, plan, outcomes)
                if any(row["status"] == "infrastructure_error" for row in wave_outcomes):
                    break

    model_outcomes = [row for row in outcomes if row["status"] == "model_outcome"]
    summary = {
        "campaign_id": plan["campaign_id"],
        "study_role": plan["study_role"],
        "planned_sessions": EXPECTED_TASK_COUNT,
        "canary_gate": {
            "task_index": 1,
            "passed": canary_passed,
            "criterion": plan["release_gate"]["criterion"],
            "required_binary_score": plan["release_gate"]["required_binary_score"],
            "verifier_execution_id": outcomes[0].get("verifier_execution_id"),
            "observed_status": outcomes[0].get("status"),
            "observed_score": outcomes[0].get("score"),
            "cleanup_verified": outcomes[0].get("cleanup_verified") is True,
        },
        "model_outcomes": len(model_outcomes),
        "infrastructure_errors": len(outcomes) - len(model_outcomes),
        "successes": sum(float(row["score"]) > 0 for row in model_outcomes),
        "outcomes": sorted(outcomes, key=lambda row: row["index"]),
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
    plan, split = load_json(args.config), load_json(args.split)
    api_key = os.environ.get("FLEET_API_KEY")
    if not api_key:
        raise RuntimeError("FLEET_API_KEY is required")
    with httpx.Client(
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        timeout=180,
    ) as orchestrator:
        live_receipt, authority_gate = build_live_receipt(orchestrator, plan, split, api_key)
    if args.receipt_out:
        args.receipt_out.parent.mkdir(parents=True, exist_ok=True)
        args.receipt_out.write_bytes(self_hosted.canonical_json(live_receipt) + b"\n")
    if args.receipt:
        frozen = load_json(args.receipt)
        validate_frozen_receipt(frozen)
        if frozen != live_receipt:
            raise RuntimeError("live calibration identities do not match the frozen receipt")
    if args.command == "preflight":
        print(
            json.dumps(
                {
                    "ok": True,
                    "campaign_id": plan["campaign_id"],
                    "planned_sessions": EXPECTED_TASK_COUNT,
                    "receipt_sha256": live_receipt["receipt_sha256"],
                    "authority_gate": authority_gate,
                    "model_revision": plan["model"]["revision"],
                    "sealed_test_tasks_selected": 0,
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
