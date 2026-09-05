"""Held-only hosted-c4 repartition of the exact 798 pass@4 cells.

This module changes execution grouping only.  Every statistical cell,
execution id, run id, network, model, task, and treatment is inherited from
the accepted v3 plan.  A future release additionally requires a terminally
passing non-scoring c4 qualifier and a fresh duplicate reconciliation over
these exact successor identities.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import stat
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evals.fleet import exact_pass4_bulk_v3 as predecessor
from evals.fleet import hosted_concurrency4_qualification_release_v3 as qualifier_release
from evals.fleet import hosted_concurrency4_qualification_v1 as qualifier
from evals.fleet import self_hosted

SPEC_SCHEMA = "fleet-exact-pass4-hosted-c4-controller-spec-v4"
PLAN_SCHEMA = "fleet-exact-pass4-hosted-c4-controller-plan-v4"
HELD_SCHEMA = "fleet-exact-pass4-hosted-c4-held-v4"
RELEASE_SCHEMA = "fleet-exact-pass4-hosted-c4-release-v4"
BRIDGE_SCHEMA = "fleet-exact-pass4-hosted-c4-duplicate-reconciliation-v1"

BASE_COMMIT = "eb52d7a06e97620219ceabc394c2f5b0f096449d"
PREDECESSOR_COMMIT = "b2934446d93cf34facf5fd007216e6dedcc9c2b4"
G7_COMMIT = "3a66b8a3376be3e5954fceb09f0d226ae986d23f"
MODULE_PATH = "evals/fleet/exact_pass4_hosted_c4_bulk_v4.py"
RUNTIME_PATH = "evals/fleet/exact_pass4_hosted_c4_bulk_runtime_v4.py"
HELD_PATH = "docs/evidence/qwen38-study/2026-09-05-opencode-exact-pass4-hosted-c4-bulk-held-v4.json"
RELEASE_PATH = (
    "docs/evidence/qwen38-study/2026-09-05-opencode-exact-pass4-hosted-c4-bulk-release-v4.json"
)
QUALIFIER_TERMINAL = Path("/mnt/sfs/jobs/chris-cyber-hosted-c4-qualification-v1/TERMINAL.json")
QUALIFIER_MODEL_RECEIPTS = {
    model: QUALIFIER_TERMINAL.parent / f"{model}.json" for model in ("qwen3.8-27b", "glm-5.3")
}
PREBULK_TERMINAL = predecessor.RECONCILIATION_ROOT / "TERMINAL.json"
INVENTORY_TERMINAL = Path(predecessor.FIXED_GATE_PATHS["exact100_inventory"])
CLAIM_ROOT = predecessor.CLAIM_ROOT
LEASE_ROOT = predecessor.LEASE_ROOT
SHA_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
ISO_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z$")
SHA256_RE = SHA_RE
CANARY_GATE_SCHEMA = predecessor.CANARY_GATE_SCHEMA
RECONCILIATION_GATE_SCHEMA = predecessor.RECONCILIATION_GATE_SCHEMA
validate_inventory_gate = predecessor.validate_inventory_gate
PACKAGE_PATHS = (
    MODULE_PATH,
    RUNTIME_PATH,
    HELD_PATH,
    "docs/HOSTED_C4_BULK_V4.md",
    "tests/test_exact_pass4_hosted_c4_bulk_v4.py",
)


def _controllers() -> dict[str, dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    for model, short, canary_rank in (
        ("qwen3.8-27b", "q38", 4),
        ("glm-5.3", "glm53", 13),
    ):
        serving_block = predecessor.CONTROLLERS["qwen-a" if model == "qwen3.8-27b" else "glm-a"][
            "serving_block"
        ]
        for stream, (first, last) in enumerate(((1, 25), (26, 50), (51, 75), (76, 100)), 1):
            key = f"{short}-s{stream}"
            values[key] = {
                "model": model,
                "stream": stream,
                "task_ranks": list(range(first, last + 1)),
                "partial_attempts": {canary_rank: [2, 3, 4]}
                if first <= canary_rank <= last
                else {},
                "cell_count": 99 if first <= canary_rank <= last else 100,
                "job_name": f"chris-{short}-ac-exact100-hosted-c4-s{stream}-v4",
                "configmap_name": f"chris-{short}-ac-exact100-hosted-c4-s{stream}-run-v4",
                "serving_block": serving_block,
            }
    return values


CONTROLLERS = _controllers()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def sha256(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def digest(value: dict[str, Any], field: str) -> str:
    return sha256(canonical({key: item for key, item in value.items() if key != field}))


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def load(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"unsafe or absent JSON file: {path}")
    value = json.loads(path.read_bytes(), object_pairs_hook=_strict_object)
    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object")
    return value


def write_once(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError("release target is unsafe")
        with os.fdopen(fd, "wb") as handle:
            fd = -1
            handle.write(canonical(value) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if fd >= 0:
            os.close(fd)


def validate_package_commit(root: Path, package_commit: str) -> None:
    if COMMIT_RE.fullmatch(package_commit) is None:
        raise ValueError("hosted-c4 package commit is invalid")
    try:
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--", *PACKAGE_PATHS],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("hosted-c4 package Git authority unavailable") from exc
    if head != package_commit or status:
        raise ValueError("hosted-c4 package is not clean at its exact commit")
    for relative in PACKAGE_PATHS:
        committed = subprocess.run(
            ["git", "-C", str(root), "show", f"{package_commit}:{relative}"],
            check=True,
            capture_output=True,
        ).stdout
        if committed != (root / relative).read_bytes():
            raise ValueError(f"hosted-c4 package byte drifted: {relative}")


def _predecessor_plans(root: Path) -> dict[str, dict[str, Any]]:
    return predecessor.validate_all(root)


def _predecessor_rows(root: Path) -> dict[tuple[str, int, int], dict[str, Any]]:
    rows: dict[tuple[str, int, int], dict[str, Any]] = {}
    for source_controller, plan in _predecessor_plans(root).items():
        for row in plan["attempts"]:
            key = (plan["model"]["served_id"], row["selection_rank"], row["attempt"])
            if key in rows:
                raise ValueError("predecessor statistical cell overlap")
            rows[key] = {**copy.deepcopy(row), "source_controller": source_controller}
    if len(rows) != 798:
        raise ValueError("predecessor cell count drifted")
    return rows


def _selected_keys(controller: str) -> list[tuple[str, int, int]]:
    source = CONTROLLERS[controller]
    selected: list[tuple[str, int, int]] = []
    partial = source["partial_attempts"]
    for rank in source["task_ranks"]:
        for attempt in partial.get(rank, [1, 2, 3, 4]):
            selected.append((source["model"], rank, attempt))
    if len(selected) != source["cell_count"] or len(selected) != len(set(selected)):
        raise ValueError("hosted-c4 partition count drifted")
    return selected


def build_spec(controller: str, root: Path) -> dict[str, Any]:
    source = CONTROLLERS.get(controller)
    if source is None:
        raise ValueError("unsupported hosted-c4 controller")
    prior = _predecessor_rows(root)
    rows = [prior[key] for key in _selected_keys(controller)]
    identity = [
        {
            key: row[key]
            for key in (
                "cell_id",
                "execution_id",
                "run_id",
                "network",
                "selection_rank",
                "attempt",
                "task_version_id",
            )
        }
        for row in rows
    ]
    body = {
        "schema_version": SPEC_SCHEMA,
        "controller": controller,
        "model": source["model"],
        "job_name": source["job_name"],
        "configmap_name": source["configmap_name"],
        "serving_block": source["serving_block"],
        "partition": {
            "stream": source["stream"],
            "task_ranks": source["task_ranks"],
            "partial_attempts": {
                str(rank): attempts for rank, attempts in source["partial_attempts"].items()
            },
            "whole_task_serving_blocks": True,
            "cell_count": source["cell_count"],
        },
        "preserved_cell_identities_sha256": sha256(canonical(identity)),
        "execution": {
            "workers": 1,
            "same_task_max_inflight": 1,
            "attempts_per_task_sequential": True,
            "global_execution_claim_before_model_call": True,
            "claim_root": CLAIM_ROOT,
            "accepted_active_claimed_or_model_started_cells_are_nonrepeatable": True,
            "automatic_retry": False,
            "restart_resumes_only_unclaimed_cells": True,
            "endpoint_lease": {
                "lease_root": LEASE_ROOT,
                "endpoint_key": source["serving_block"],
                "maximum_streams": 4,
            },
            "priority_class": "fleet-serve-low",
            "preemption_policy": "Never",
            "cpu_only": True,
        },
        "treatment": predecessor.exact.EXPECTED_TREATMENT,
        "serving_treatment": {
            "mode": "hosted_only",
            "dedicated_and_hosted_not_pooled": True,
            "dedicated_glm_requires_append_only_whole_task_repartition": True,
        },
        "launch_authorized": False,
    }
    return {**body, "spec_sha256": digest(body, "spec_sha256")}


def build_plan(controller: str, root: Path) -> dict[str, Any]:
    spec = build_spec(controller, root)
    source = CONTROLLERS[controller]
    prior = _predecessor_rows(root)
    rows = [prior[key] for key in _selected_keys(controller)]
    template_controller = "qwen-a" if source["model"] == "qwen3.8-27b" else "glm-a"
    prior_plan = _predecessor_plans(root)[template_controller]
    execution = copy.deepcopy(prior_plan["execution"])
    execution.update(spec["execution"])
    attempts = []
    for ordinal, row in enumerate(rows, 1):
        attempts.append({**row, "ordinal": ordinal})
    body = {
        "schema_version": PLAN_SCHEMA,
        "campaign_id": predecessor.exact.EXPECTED_CAMPAIGN_ID,
        "controller": controller,
        "job_name": source["job_name"],
        "configmap_name": source["configmap_name"],
        "sfs_root": f"/mnt/sfs/jobs/{source['job_name']}",
        "model": copy.deepcopy(prior_plan["model"]),
        "serving_block": source["serving_block"],
        "task_count": len(source["task_ranks"]),
        "new_session_count": len(attempts),
        "attempts": attempts,
        "execution": execution,
        "treatment": copy.deepcopy(prior_plan["treatment"]),
        "serving_treatment": spec["serving_treatment"],
        "release_required": True,
        "launch_authorized": False,
        "privacy": copy.deepcopy(prior_plan["privacy"]),
    }
    return {**body, "plan_sha256": digest(body, "plan_sha256")}


def validate_all(root: Path) -> dict[str, dict[str, Any]]:
    plans = {controller: build_plan(controller, root) for controller in CONTROLLERS}
    prior = _predecessor_rows(root)
    current: dict[tuple[str, int, int], dict[str, Any]] = {}
    task_owners: dict[tuple[str, int], str] = {}
    for controller, plan in plans.items():
        for row in plan["attempts"]:
            key = (plan["model"]["served_id"], row["selection_rank"], row["attempt"])
            task_key = key[:2]
            if key in current or (task_key in task_owners and task_owners[task_key] != controller):
                raise ValueError("hosted-c4 task or cell overlap")
            task_owners[task_key] = controller
            current[key] = row
    if set(current) != set(prior):
        raise ValueError("hosted-c4 does not cover exact predecessor cells")
    preserved = ("cell_id", "execution_id", "run_id", "network", "task_version_id")
    if any(any(current[key][field] != prior[key][field] for field in preserved) for key in prior):
        raise ValueError("hosted-c4 changed a statistical or run identity")
    for model in ("qwen3.8-27b", "glm-5.3"):
        model_plans = [plan for plan in plans.values() if plan["model"]["served_id"] == model]
        if len(model_plans) != 4 or sum(plan["new_session_count"] for plan in model_plans) != 399:
            raise ValueError("hosted-c4 model partition count drifted")
    return plans


def build_runtime_plan(
    controller: str, inventory_receipt: dict[str, Any], root: Path
) -> dict[str, Any]:
    """Hydrate the unchanged v3 execution engine with one c4 partition."""
    validate_inventory_gate(inventory_receipt, root)
    compact = validate_all(root)[controller]
    template_path = predecessor.RUNTIME_TEMPLATE_PATHS[compact["model"]["served_id"]]
    template = load(root / template_path)
    tasks_by_rank = {row["selection_rank"]: row for row in inventory_receipt["tasks"]}
    tasks = []
    for rank in sorted({row["selection_rank"] for row in compact["attempts"]}):
        source = tasks_by_rank[rank]
        environment = copy.deepcopy(source["environment"])
        environment.pop("version_id_authority", None)
        environment.pop("runtime_seed_file_count", None)
        environment["ttl_seconds"] = 32400
        task = copy.deepcopy(source["task"])
        task.pop("version", None)
        tasks.append(
            {
                "rank": rank,
                "source_rank": rank,
                "environment": environment,
                "task": task,
                "verifier": copy.deepcopy(source["verifier"]),
            }
        )
    attempts = [
        {**copy.deepcopy(row), "rank": row["selection_rank"], "source_rank": row["selection_rank"]}
        for row in compact["attempts"]
    ]
    execution = copy.deepcopy(compact["execution"])
    execution.update(
        {
            "required_task_tools": ["bash", "submit_report"],
            "required_task_tool_catalog_sha256": template["execution"][
                "required_task_tool_catalog_sha256"
            ],
            "training_data_eligible": False,
        }
    )
    runtime_harness = copy.deepcopy(template["harness"])
    runtime_harness["compaction_headroom_tokens"] = compact["treatment"][
        "compaction_headroom_tokens"
    ]
    body = {
        "schema_version": "fleet-exact-pass4-hosted-c4-executable-plan-v4",
        "controller": controller,
        "campaign_id": compact["job_name"],
        "source_job_id": compact["job_name"],
        "sfs_root": compact["sfs_root"],
        "repo_root": ".",
        "inventory_receipt": inventory_receipt,
        "inventory_receipt_sha256": inventory_receipt["receipt_sha256"],
        "model": copy.deepcopy(template["model"]),
        "harness": runtime_harness,
        "authority": copy.deepcopy(template["authority"]),
        "serving_block": compact["serving_block"],
        "serving_treatment": compact["serving_treatment"],
        "tasks": tasks,
        "attempts": attempts,
        "execution": execution,
        "treatment": compact["treatment"],
        "release_required": True,
        "launch_authorized": True,
        "privacy": compact["privacy"],
    }
    plan = {**body, "plan_sha256": digest(body, "plan_sha256")}
    settings = self_hosted.opencode_settings(plan)
    canonical_settings = self_hosted.canonical_json(settings)
    if (
        plan["harness"]["settings_canonical_sha256"] != self_hosted.sha256(canonical_settings)
        or plan["harness"]["settings_file_sha256"] != self_hosted.sha256(canonical_settings + b"\n")
        or settings.get("compaction") != {"auto": True, "reserved": 20000}
        or "plugin" in settings
        or plan["execution"]["required_task_tools"] != ["bash", "submit_report"]
    ):
        raise ValueError("hosted-c4 executable treatment drifted")
    return plan


def _identity_rows(root: Path) -> list[dict[str, Any]]:
    plans = validate_all(root)
    return sorted(
        (
            {
                "model": plan["model"]["served_id"],
                "selection_rank": row["selection_rank"],
                "attempt": row["attempt"],
                "task_version_id": row["task_version_id"],
                "cell_id": row["cell_id"],
                "execution_id": row["execution_id"],
                "run_id": row["run_id"],
                "network": row["network"],
            }
            for plan in plans.values()
            for row in plan["attempts"]
        ),
        key=lambda row: (row["model"], row["selection_rank"], row["attempt"]),
    )


def expected_held(root: Path) -> dict[str, Any]:
    rows = _identity_rows(root)
    receipt: dict[str, Any] = {
        "schema_version": HELD_SCHEMA,
        "append_only": True,
        "status": "HELD",
        "launch_authorized": False,
        "objects_created": False,
        "base_commit": BASE_COMMIT,
        "predecessor_bulk_commit": PREDECESSOR_COMMIT,
        "generation7_release_commit": G7_COMMIT,
        "planned_sessions": {"qwen3.8-27b": 399, "glm-5.3": 399, "total": 798},
        "controller_count": {"qwen3.8-27b": 4, "glm-5.3": 4, "total": 8},
        "exact_identity_sha256": sha256(canonical(rows)),
        "release_conditions": {
            "generation7_canaries_accepted": True,
            "prebulk_v3_terminal_clear": True,
            "qualifier_v3_job_exclusively_complete": True,
            "qualifier_v3_terminal_status_passed": True,
            "both_qualifier_model_receipts_passed": True,
            "fresh_v4_duplicate_reconciliation_clear": True,
            "all_v4_jobs_configmaps_pods_output_roots_absent": True,
            "all_798_cells_absent_from_accepted_active_claimed_and_model_started_state": True,
        },
        "execution_contract": {
            "four_independent_one_stream_controllers_per_model": True,
            "endpoint_lease_maximum_streams_per_model": 4,
            "whole_task_serving_blocks": True,
            "cpu_only": True,
            "priority_class": "fleet-serve-low",
            "preemption_policy": "Never",
            "global_create_once_claim_before_model_call": True,
            "restart_skips_claimed_cells": True,
        },
        "integration": {
            "hosted_treatment_only": True,
            "dedicated_glm_treatment_included": False,
            "dedicated_glm_must_use_mutually_exclusive_append_only_successor": True,
            "dedicated_glm_transfer_granularity": "complete_task_rank",
            "statistical_cell_identities_must_remain_byte_identical": True,
        },
        "privacy": {
            "credentials_included": False,
            "prompts_traces_flags_or_scores_included": False,
        },
    }
    receipt["receipt_sha256"] = digest(receipt, "receipt_sha256")
    return receipt


def validate_held(receipt: dict[str, Any], root: Path) -> None:
    if receipt != expected_held(root):
        raise ValueError("hosted-c4 held receipt drifted")


def validate_bridge(receipt: dict[str, Any], root: Path) -> None:
    rows = _identity_rows(root)
    plans = validate_all(root)
    if (
        receipt.get("schema_version") != BRIDGE_SCHEMA
        or receipt.get("status") != "CLEAR"
        or receipt.get("planned_execution_count") != 798
        or receipt.get("exact_identity_sha256") != sha256(canonical(rows))
        or receipt.get("checked_job_names") != sorted(plan["job_name"] for plan in plans.values())
        or receipt.get("checked_output_roots")
        != sorted(plan["sfs_root"] for plan in plans.values())
        or receipt.get("checked_configmap_names")
        != sorted(plan["configmap_name"] for plan in plans.values())
        or receipt.get("collisions")
        != {
            "fleet_api": 0,
            "kubernetes_job_pod_or_configmap": 0,
            "sfs_output": 0,
            "global_claim": 0,
            "accepted_active_or_model_started_cell": 0,
        }
        or receipt.get("methods") != ["GET"]
        or receipt.get("mutation_calls") != 0
        or receipt.get("checked_immediately_before_release") is not True
        or receipt.get("observer_job_succeeded") is not True
        or receipt.get("observer_pod_restarts") != 0
        or COMMIT_RE.fullmatch(str(receipt.get("observer_package_commit"))) is None
        or UTC_RE.fullmatch(str(receipt.get("observed_at_utc"))) is None
        or receipt.get("receipt_sha256") != digest(receipt, "receipt_sha256")
        or receipt.get("prompts_traces_flags_or_scores_included") is not False
    ):
        raise ValueError("hosted-c4 duplicate reconciliation is not clear")
    for field in ("observer_job_uid", "observer_pod_uid"):
        uuid.UUID(str(receipt.get(field)))


def validate_qualifier_pass(
    release: dict[str, Any],
    terminal: dict[str, Any],
    model_receipts: dict[str, dict[str, Any]],
    job: dict[str, Any],
    pods: dict[str, Any],
    root: Path,
) -> dict[str, Any]:
    qualifier_release.validate_release(release, root)
    metadata, status = job.get("metadata") or {}, job.get("status") or {}
    conditions = status.get("conditions") or []
    items = pods.get("items")
    if (
        metadata.get("name") != qualifier_release.package.JOB_NAME
        or not any(
            row.get("type") == "Complete" and row.get("status") == "True" for row in conditions
        )
        or any(row.get("type") == "Failed" and row.get("status") == "True" for row in conditions)
        or status.get("active", 0) not in (0, None)
        or status.get("succeeded") != 1
        or status.get("failed", 0) not in (0, None)
        or not isinstance(items, list)
        or len(items) != 1
    ):
        raise ValueError("hosted-c4 qualifier Job is not exclusively complete")
    pod = items[0]
    job_uid = str(uuid.UUID(metadata["uid"]))
    pod_uid = str(uuid.UUID(pod["metadata"]["uid"]))
    owners = pod["metadata"].get("ownerReferences") or []
    statuses = (pod.get("status") or {}).get("containerStatuses") or []
    if (
        owners
        != [
            {
                "apiVersion": "batch/v1",
                "kind": "Job",
                "name": qualifier_release.package.JOB_NAME,
                "uid": job_uid,
                "controller": True,
            }
        ]
        or (pod.get("status") or {}).get("phase") != "Succeeded"
        or len(statuses) != 1
        or statuses[0].get("name") != "qualifier"
        or statuses[0].get("restartCount") != 0
        or ((statuses[0].get("state") or {}).get("terminated") or {}).get("exitCode") != 0
    ):
        raise ValueError("hosted-c4 qualifier Pod did not cleanly succeed")
    expected_models = ("qwen3.8-27b", "glm-5.3")
    if set(model_receipts) != set(expected_models):
        raise ValueError("hosted-c4 qualifier model receipt set drifted")
    model_bindings = []
    for model in expected_models:
        value = model_receipts[model]
        waves = value.get("waves")
        model_identity = value.get("model") or {}
        context = value.get("context_contract") or {}
        tools = value.get("tool_contract") or {}
        expected_model = qualifier.EXPECTED_MODELS[model]
        if (
            value.get("schema_version") != qualifier.SCHEMA
            or value.get("status") != "PASSED"
            or model_identity
            != {
                "served_id": model,
                "repository": expected_model["repository"],
                "revision": expected_model["revision"],
                "endpoint_origin": qualifier.ORIGIN,
                "response_model_exact": True,
            }
            or context
            not in (
                {
                    "expected_context_length": 262144,
                    "context_length_observable": False,
                    "observed_context_length": None,
                    "unobservable_roster_context_requires_existing_exact_harness_gate": True,
                },
                {
                    "expected_context_length": 262144,
                    "context_length_observable": True,
                    "observed_context_length": 262144,
                    "unobservable_roster_context_requires_existing_exact_harness_gate": True,
                },
            )
            or tools
            != {
                "names": ["bash", "submit_report"],
                "openai_tool_schema_sha256": qualifier.sha256(
                    qualifier.canonical_json(qualifier.TOOLS)
                ),
                "forced_selection_order_per_stream": ["bash", "submit_report"],
                "argument_shapes_valid": True,
                "tool_execution_performed": False,
            }
            or not isinstance(waves, list)
            or value.get("decision") != qualifier.evaluate_waves(waves)
            or value["decision"].get("accepted") is not True
            or value.get("lease") != {"namespace": qualifier.LEASE_NAMESPACE, "exclusive": True}
            or value.get("request_counts")
            != {
                "chat_completions": 12,
                "task_instance": 0,
                "session": 0,
                "scoring": 0,
                "verifier": 0,
            }
            or value.get("privacy")
            != {
                "request_bodies_included": False,
                "response_bodies_included": False,
                "tool_arguments_included": False,
                "prompts_traces_flags_or_scores_included": False,
                "credentials_included": False,
            }
            or value.get("receipt_sha256") != digest(value, "receipt_sha256")
        ):
            raise ValueError("hosted-c4 qualifier model did not pass")
        model_bindings.append({"model": model, "receipt_sha256": value["receipt_sha256"]})
    runtime = terminal.get("runtime") or {}
    if (
        terminal.get("schema_version") != qualifier.SCHEMA
        or terminal.get("status") != "PASSED"
        or terminal.get("classification") != "operational_gate_no_capability_claim"
        or ISO_UTC_RE.fullmatch(str(terminal.get("observed_at_utc"))) is None
        or runtime.get("job_uid") != job_uid
        or runtime.get("pod_uid") != pod_uid
        or runtime.get("package_commit") != release["implementation"]["commit"]
        or runtime.get("package_sha256") != release["probe_package"]["sha256"]
        or runtime.get("priority_class") != "fleet-serve-low"
        or runtime.get("preemption_policy") != "Never"
        or runtime.get("cpu_only") is not True
        or terminal.get("models")
        != [
            {"served_id": row["model"], "status": "PASSED", "receipt_sha256": row["receipt_sha256"]}
            for row in model_bindings
        ]
        or terminal.get("request_counts")
        != {
            "fleet_account_get": 1,
            "hosted_models_get": 1,
            "chat_completions": 24,
            "task_instance": 0,
            "session": 0,
            "scoring": 0,
            "verifier": 0,
        }
        or terminal.get("endpoint_lease")
        != {
            "namespace": qualifier.LEASE_NAMESPACE,
            "separate_from_scored_endpoint_leases": True,
            "released": True,
        }
        or terminal.get("scored_bulk_launch_authorized") is not False
        or terminal.get("privacy")
        != {
            "request_bodies_included": False,
            "response_bodies_included": False,
            "tool_arguments_included": False,
            "prompts_traces_flags_or_scores_included": False,
            "credentials_included": False,
        }
        or terminal.get("receipt_sha256") != digest(terminal, "receipt_sha256")
    ):
        raise ValueError("hosted-c4 qualifier terminal is not a PASS")
    return {
        "release_receipt_sha256": release["receipt_sha256"],
        "terminal_receipt_sha256": terminal["receipt_sha256"],
        "job_uid": job_uid,
        "pod_uid": pod_uid,
        "models": model_bindings,
    }


def build_release(
    root: Path,
    implementation_commit: str,
    released_at_utc: str,
    *,
    qualifier_release_receipt: dict[str, Any],
    qualifier_terminal: dict[str, Any],
    qualifier_models: dict[str, dict[str, Any]],
    qualifier_job: dict[str, Any],
    qualifier_pods: dict[str, Any],
    bridge: dict[str, Any],
    prebulk_terminal: dict[str, Any],
    inventory_terminal: dict[str, Any],
    prebulk_evidence_root: Path | None = None,
) -> dict[str, Any]:
    validate_package_commit(root, implementation_commit)
    plans = validate_all(root)
    validate_bridge(bridge, root)
    if bridge["observer_package_commit"] != implementation_commit:
        raise ValueError("hosted-c4 bridge is not from the released package")
    if UTC_RE.fullmatch(released_at_utc) is None:
        raise ValueError("hosted-c4 release timestamp is invalid")
    observed = datetime.strptime(bridge["observed_at_utc"], "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=UTC
    )
    released = datetime.strptime(released_at_utc, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    if released < observed or (released - observed).total_seconds() > 900:
        raise ValueError("hosted-c4 bridge is stale at release")
    if (released - datetime.now(UTC)).total_seconds() > 60:
        raise ValueError("hosted-c4 release timestamp is in the future")
    predecessor.validate_inventory_gate(inventory_terminal, root)
    # The prebulk gate validates its complete G7, source/accept, inventory, and
    # duplicate chain from the provided sanitized evidence mirror.
    predecessor.validate_reconciliation_gate(
        prebulk_terminal, root, evidence_root=prebulk_evidence_root
    )
    qualifier_binding = validate_qualifier_pass(
        qualifier_release_receipt,
        qualifier_terminal,
        qualifier_models,
        qualifier_job,
        qualifier_pods,
        root,
    )
    receipt: dict[str, Any] = {
        "schema_version": RELEASE_SCHEMA,
        "append_only": True,
        "status": "RELEASED",
        "launch_authorized": True,
        "hosted_only": True,
        "dedicated_serving_authorized": False,
        "implementation_commit": implementation_commit,
        "base_commit": BASE_COMMIT,
        "predecessor_bulk_commit": PREDECESSOR_COMMIT,
        "generation7_release_commit": G7_COMMIT,
        "released_at_utc": released_at_utc,
        "qualifier": qualifier_binding,
        "prebulk_terminal_receipt_sha256": prebulk_terminal["receipt_sha256"],
        "inventory_terminal_receipt_sha256": inventory_terminal["receipt_sha256"],
        "fresh_duplicate_reconciliation_receipt_sha256": bridge["receipt_sha256"],
        "exact_identity_sha256": sha256(canonical(_identity_rows(root))),
        "controllers": [
            {
                "controller": key,
                "job_name": plan["job_name"],
                "configmap_name": plan["configmap_name"],
                "sfs_root": plan["sfs_root"],
                "model": plan["model"]["served_id"],
                "session_count": plan["new_session_count"],
                "plan_sha256": plan["plan_sha256"],
            }
            for key, plan in plans.items()
        ],
        "execution": {
            "endpoint_lease_maximum_streams_per_model": 4,
            "cpu_only": True,
            "priority_class": "fleet-serve-low",
            "preemption_policy": "Never",
            "whole_task_serving_blocks": True,
            "create_once": True,
            "restart_safe_claims": True,
        },
        "integration": {
            "dedicated_glm_pooled": False,
            "dedicated_glm_requires_mutually_exclusive_append_only_successor": True,
        },
        "privacy": {
            "credentials_included": False,
            "prompts_traces_flags_or_scores_included": False,
        },
    }
    receipt["receipt_sha256"] = digest(receipt, "receipt_sha256")
    return receipt


def validate_runtime_release(
    release: dict[str, Any],
    plan: dict[str, Any],
    root: Path,
    *,
    package_commit: str,
) -> None:
    """Bind one runtime plan to the exact append-only v4 release.

    The full release builder performs the live G7, prebulk, qualifier, and
    duplicate checks. Runtime Pods consume only its immutable digest-bound
    result; they must not reinterpret the predecessor v3 reconciliation using
    the new v4 controller names.
    """
    plans = validate_all(root)
    expected_controllers = [
        {
            "controller": key,
            "job_name": built["job_name"],
            "configmap_name": built["configmap_name"],
            "sfs_root": built["sfs_root"],
            "model": built["model"]["served_id"],
            "session_count": built["new_session_count"],
            "plan_sha256": built["plan_sha256"],
        }
        for key, built in plans.items()
    ]
    qualifier_binding = release.get("qualifier") or {}
    qualifier_models = qualifier_binding.get("models")
    if (
        release.get("schema_version") != RELEASE_SCHEMA
        or release.get("append_only") is not True
        or release.get("status") != "RELEASED"
        or release.get("launch_authorized") is not True
        or release.get("hosted_only") is not True
        or release.get("dedicated_serving_authorized") is not False
        or release.get("implementation_commit") != package_commit
        or release.get("base_commit") != BASE_COMMIT
        or release.get("predecessor_bulk_commit") != PREDECESSOR_COMMIT
        or release.get("generation7_release_commit") != G7_COMMIT
        or UTC_RE.fullmatch(str(release.get("released_at_utc"))) is None
        or release.get("exact_identity_sha256") != sha256(canonical(_identity_rows(root)))
        or release.get("controllers") != expected_controllers
        or release.get("execution")
        != {
            "endpoint_lease_maximum_streams_per_model": 4,
            "cpu_only": True,
            "priority_class": "fleet-serve-low",
            "preemption_policy": "Never",
            "whole_task_serving_blocks": True,
            "create_once": True,
            "restart_safe_claims": True,
        }
        or release.get("integration")
        != {
            "dedicated_glm_pooled": False,
            "dedicated_glm_requires_mutually_exclusive_append_only_successor": True,
        }
        or release.get("privacy")
        != {
            "credentials_included": False,
            "prompts_traces_flags_or_scores_included": False,
        }
        or SHA_RE.fullmatch(str(release.get("prebulk_terminal_receipt_sha256"))) is None
        or SHA_RE.fullmatch(str(release.get("inventory_terminal_receipt_sha256"))) is None
        or SHA_RE.fullmatch(str(release.get("fresh_duplicate_reconciliation_receipt_sha256")))
        is None
        or SHA_RE.fullmatch(str(qualifier_binding.get("release_receipt_sha256"))) is None
        or SHA_RE.fullmatch(str(qualifier_binding.get("terminal_receipt_sha256"))) is None
        or not isinstance(qualifier_models, list)
        or len(qualifier_models) != 2
        or qualifier_models
        != [
            {
                "model": model,
                "receipt_sha256": qualifier_models[index].get("receipt_sha256"),
            }
            for index, model in enumerate(("qwen3.8-27b", "glm-5.3"))
        ]
        or any(SHA_RE.fullmatch(str(row.get("receipt_sha256"))) is None for row in qualifier_models)
        or release.get("receipt_sha256") != digest(release, "receipt_sha256")
    ):
        raise ValueError("hosted-c4 runtime release drifted")
    for field in ("job_uid", "pod_uid"):
        uuid.UUID(str(qualifier_binding.get(field)))
    if plan != build_runtime_plan(plan["controller"], plan["inventory_receipt"], root):
        raise ValueError("hosted-c4 runtime plan drifted from release")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preview", "validate-held"))
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--held", type=Path)
    args = parser.parse_args()
    if args.command == "validate-held":
        if args.held is None:
            parser.error("validate-held requires --held")
        validate_held(load(args.held), args.repo)
        return 0
    plans = validate_all(args.repo)
    print(
        json.dumps(
            {
                "ok": True,
                "status": "HELD",
                "launch_authorized": False,
                "objects_created": False,
                "controllers": len(plans),
                "planned_sessions": sum(plan["new_session_count"] for plan in plans.values()),
                "maximum_streams_per_model": 4,
                "whole_task_serving_blocks": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
