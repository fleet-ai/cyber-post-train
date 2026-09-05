"""Immutable single-cell Qwen G18 hosted canary authority."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

from evals.fleet import exact_pass4_universe as exact
from evals.fleet import self_hosted

CONTROLLER = "qwen-canary"
JOB_NAME = "chris-q38-ac-r005-a1-g18-v1"
CONFIGMAP_NAME = JOB_NAME + "-run-v1"
PREFLIGHT_JOB_NAME = JOB_NAME + "-preflight-v2"
PREFLIGHT_CONFIGMAP_NAME = PREFLIGHT_JOB_NAME + "-run-v1"
PREFLIGHT_ROOT = f"/mnt/sfs/jobs/{PREFLIGHT_JOB_NAME}"
OUTPUT_ROOT = f"/mnt/sfs/jobs/{JOB_NAME}"
PLAN_PATH = "evals/fleet/configs/qwen-hosted-generation18-canary.json"
PARITY_PATH = "docs/evidence/qwen38-study/2026-09-05-qwen38-hosted-actual-opencode-parity-v1.json"
CLAIM_ROOT = "/mnt/sfs/cell-execution-claims/opencode11827-autocontinue-v1"
LEASE_ROOT = "/mnt/sfs/endpoint-leases/opencode11827-autocontinue-v1"
FLEET_API_KEY_SECRET = "chris-cyber-opencode-evals-v2"
SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
CONTROLLERS = {
    CONTROLLER: {
        "model": "qwen3.8-27b",
        "job_name": JOB_NAME,
        "configmap_name": CONFIGMAP_NAME,
        "serving_block": "qwen-hosted-autocontinue-v1",
        "cell_count": 1,
    }
}


def load(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"unsafe or absent immutable input: {path}")
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict) or raw != self_hosted.canonical_json(value) + b"\n":
        raise ValueError(f"non-canonical immutable input: {path}")
    return value


def validate_parity(value: dict[str, Any]) -> None:
    tool = value.get("tool_contract") or {}
    execution = value.get("execution") or {}
    privacy = value.get("privacy") or {}
    if any(
        (
            value.get("schema_version") != "fleet-opencode-actual-harness-hosted-parity-v1",
            value.get("status") != "PASSED_NON_SCORED",
            value.get("classification") != "ACTUAL_HARNESS_PARITY",
            value.get("endpoint")
            != {
                "kind": "shared_hosted_inference",
                "origin": "https://inference.flt.build",
                "server_binding": None,
            },
            value.get("model", {}).get("served_id") != "qwen3.8-27b",
            value.get("harness", {}).get("version") != "1.18.27",
            value.get("harness", {}).get("context_window_size") != 262144,
            value.get("harness", {}).get("compaction_headroom_tokens") != 20000,
            tool.get("calls_observed_in_order") != ["bash", "submit_report"],
            tool.get("mcp_catalog_sha256")
            != "sha256:85fad6bdc3a835bf52a11a99b3387740eb06eb3d1720ad9bb33f3feac215b44a",
            tool.get("model_request_catalog_exact") is not True,
            tool.get("arguments_structurally_valid") is not True,
            execution.get("harness_exit_code") != 0,
            execution.get("task_instance_session_verifier_scoring_calls") != 0,
            privacy.get("benchmark_content_included") is not False,
            privacy.get("responses_or_model_outputs_included") is not False,
            value.get("receipt_sha256") != self_hosted.digest_without(value, "receipt_sha256"),
        )
    ):
        raise ValueError("hosted actual-OpenCode parity receipt drifted")


def derive_plan(inventory: dict[str, Any], root: Path) -> dict[str, Any]:
    """Derive rank-5/attempt-1 G18 from the exact accepted inventory."""
    from evals.fleet import qwen_bulk_generation16 as g17

    g17.validate_inventory_gate(inventory, root)
    source = g17.build_runtime_plan("qwen-a", inventory, root)
    task = next(row for row in source["tasks"] if row["rank"] == 5)
    universe = exact.build_universe(
        exact.read_object(
            root / "evals/fleet/configs/q38-glm53-exact-easiest100-pass4-campaign-v1.json"
        ),
        root,
    )
    cell = next(
        row
        for row in universe["cells"]
        if row["model"] == "qwen3.8-27b" and row["selection_rank"] == 5 and row["attempt"] == 1
    )
    execution = exact.execution_for(cell["cell_id"], 18)
    run_id = f"chris-q38-ac-g18-r005-a1-{execution['execution_id'][7:15]}"
    attempt = {
        "ordinal": 1,
        "cell_id": cell["cell_id"],
        "selection_rank": 5,
        "attempt": 1,
        "task_version_id": cell["task_version_id"],
        "execution_generation": 18,
        "execution_id": execution["execution_id"],
        "task_key": task["task"]["key"],
        "environment_version_id": task["environment"]["version_id"],
        "run_id": run_id,
        "network": run_id.removeprefix("chris-")[:63],
    }
    body = {
        "schema_version": "fleet-qwen-generation18-hosted-canary-plan-v1",
        "controller": CONTROLLER,
        "campaign_id": JOB_NAME,
        "source_job_id": JOB_NAME,
        "sfs_root": OUTPUT_ROOT,
        "repo_root": ".",
        "inventory_receipt": {"receipt_sha256": inventory["receipt_sha256"]},
        "inventory_receipt_sha256": inventory["receipt_sha256"],
        "model": copy.deepcopy(source["model"]),
        "harness": copy.deepcopy(source["harness"]),
        "authority": copy.deepcopy(source["authority"]),
        "serving_block": "qwen-hosted-autocontinue-v1",
        "tasks": [copy.deepcopy(task)],
        "attempts": [attempt],
        "execution": {
            **copy.deepcopy(source["execution"]),
            "endpoint_lease": {
                "lease_root": LEASE_ROOT,
                "endpoint_key": "qwen-hosted-autocontinue-v1",
                "maximum_streams": 2,
            },
            "priority_class": "fleet-serve-low",
            "preemption_policy": "Never",
        },
        "treatment": copy.deepcopy(source["treatment"]),
        "release_required": True,
        "launch_authorized": True,
        "privacy": copy.deepcopy(source["privacy"]),
    }
    plan = {**body, "plan_sha256": self_hosted.digest_without(body, "plan_sha256")}
    validate_plan(plan)
    return plan


def validate_plan(plan: dict[str, Any]) -> None:
    attempts = plan.get("attempts") or []
    task = (plan.get("tasks") or [{}])[0]
    if any(
        (
            plan.get("schema_version") != "fleet-qwen-generation18-hosted-canary-plan-v1",
            plan.get("controller") != CONTROLLER,
            plan.get("campaign_id") != JOB_NAME,
            plan.get("source_job_id") != JOB_NAME,
            plan.get("sfs_root") != OUTPUT_ROOT,
            plan.get("model", {}).get("served_id") != "qwen3.8-27b",
            plan.get("harness", {}).get("version") != "1.18.27",
            plan.get("harness", {}).get("context_window_size") != 262144,
            plan.get("execution", {}).get("required_task_tools") != ["bash", "submit_report"],
            plan.get("execution", {}).get("required_task_tool_catalog_sha256")
            != "sha256:85fad6bdc3a835bf52a11a99b3387740eb06eb3d1720ad9bb33f3feac215b44a",
            plan.get("execution", {}).get("endpoint_lease", {}).get("maximum_streams") != 2,
            plan.get("execution", {}).get("priority_class") != "fleet-serve-low",
            plan.get("execution", {}).get("preemption_policy") != "Never",
            len(attempts) != 1,
            attempts[0].get("selection_rank") != 5,
            attempts[0].get("attempt") != 1,
            attempts[0].get("execution_generation") != 18,
            task.get("rank") != 5,
            task.get("task", {}).get("version_id") != attempts[0].get("task_version_id"),
            plan.get("plan_sha256") != self_hosted.digest_without(plan, "plan_sha256"),
        )
    ):
        raise ValueError("Generation-18 hosted canary plan drifted")


def validate_all(root: Path) -> dict[str, dict[str, Any]]:
    plan = load(root / PLAN_PATH)
    validate_plan(plan)
    return {CONTROLLER: plan}


def build_runtime_plan(
    controller: str, inventory_receipt: dict[str, Any], root: Path
) -> dict[str, Any]:
    if controller != CONTROLLER:
        raise ValueError("unknown Generation-18 controller")
    plan = validate_all(root)[CONTROLLER]
    if plan["inventory_receipt"] != inventory_receipt:
        raise ValueError("Generation-18 inventory projection drifted")
    return plan
