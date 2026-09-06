"""Held, score-blind plan for the next untouched hosted-Qwen whole task.

This module deliberately cannot launch.  It freezes rank 17 and all four of
its pass@4 cells behind a fresh runtime inventory, lease, session, claim, and
output-absence gate.  A separate reviewed release must authorize execution.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from evals.fleet import exact_pass4_universe as exact
from evals.fleet import self_hosted

SCHEMA = "fleet-qwen38-hosted-atomic-whole-task-plan-v5"
HELD_SCHEMA = "fleet-qwen38-hosted-rank17-g22-held-v1"
SOURCE_PATH = Path("evals/fleet/configs/qwen-hosted-generation19-qwen-a-v4.json")
SOURCE_PLAN_SHA256 = "sha256:5fcc7e90a4ffbb093c09ca3cbb9cf907ab755ad873e68853629b283dad57542e"
RANK = 17
TASK_VERSION_ID = "0b192133-c9b8-4211-b0a1-dbf98be46fe3"
EXECUTION_GENERATION = 22
CONTROLLER = "qwen-a"
JOB_NAME = "chris-q38-hosted-r017-whole-task-g22-v1"
CONFIGMAP_NAME = "chris-q38-hosted-r017-whole-task-g22-package-v1"
SFS_ROOT = f"/mnt/sfs/jobs/{JOB_NAME}"
DIAGNOSTIC_ROOT = f"{SFS_ROOT}-diagnostic"
HELD_PATH = Path(
    "docs/evidence/qwen38-study/2026-09-06-qwen38-hosted-rank17-g22-held-v1.json"
)

EXPECTED_MODEL = {
    "endpoint_origin": "https://inference.flt.build",
    "repository": "Qwen/Qwen3.8-27B",
    "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    "served_id": "qwen3.8-27b",
    "session_model": (
        "fleet-cluster-opencode-1.18.27/"
        "qwen3.8-27b-opencode11827-autocontinue-v1"
    ),
}
EXPECTED_HARNESS = {
    "name": "opencode",
    "version": "1.18.27",
    "release_asset_sha256": (
        "sha256:4af5494f9433f59db8c1e344198f0ee72a50c06ec009fb4a8aeab4c2d4abd702"
    ),
    "provider_adapter": "@ai-sdk/openai-compatible",
    "context_management": "opencode_1.18.27_native_compaction_autocontinue_v1",
    "context_window_size": 262144,
    "compaction_headroom_tokens": 20000,
    "max_output_tokens": 32768,
    "max_model_requests": 600,
    "timeout_seconds": 28800,
    "settings_canonical_sha256": (
        "sha256:0458cabed67de9e2001661e61707fb16d91669774add3adbf4c97dd559fb5531"
    ),
    "settings_file_sha256": (
        "sha256:1328f6eb97861443b712625d9038a924de9c220c146667565dd4c5bb73a6d8f8"
    ),
}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _seal(body: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(body)
    value["receipt_sha256"] = self_hosted.digest_without(value, "receipt_sha256")
    return value


def _fresh_attempt(row: Mapping[str, Any]) -> dict[str, Any]:
    item = copy.deepcopy(dict(row))
    execution = exact.execution_for(str(item["cell_id"]), EXECUTION_GENERATION)
    run_id = (
        f"chris-q38-ac-g22v1-a-r{RANK:03d}-a{item['attempt']}-"
        f"{execution['execution_id'][7:15]}"
    )
    item.update(
        execution_generation=EXECUTION_GENERATION,
        execution_id=execution["execution_id"],
        run_id=run_id,
        network=run_id.removeprefix("chris-")[:63],
    )
    return item


def build_plan(root: Path) -> dict[str, Any]:
    source = _load(root / SOURCE_PATH)
    if source.get("plan_sha256") != SOURCE_PLAN_SHA256:
        raise ValueError("rank17 source plan digest drifted")
    tasks = [copy.deepcopy(row) for row in source.get("tasks", []) if row.get("rank") == RANK]
    prior_attempts = [
        copy.deepcopy(row)
        for row in source.get("attempts", [])
        if row.get("selection_rank") == RANK
    ]
    body = copy.deepcopy(source)
    body.pop("plan_sha256")
    body.update(
        schema_version=SCHEMA,
        controller=CONTROLLER,
        campaign_id=JOB_NAME,
        source_job_id=JOB_NAME,
        sfs_root=SFS_ROOT,
        tasks=tasks,
        attempts=[_fresh_attempt(row) for row in prior_attempts],
        predecessor_plan_sha256=SOURCE_PLAN_SHA256,
        launch_authorized=False,
        release_required=True,
        atomic_whole_task_reservation={
            "required": True,
            "claim_count": 4,
            "all_claims_validated_before_model_call": True,
            "attempt_order": [1, 2, 3, 4],
            "same_task_max_inflight": 1,
            "restart_recovers_partial_publication": True,
            "model_boundary_is_durable_and_nonrepeatable": True,
        },
        execution_rollforward={
            "statistical_cell_identity_unchanged": True,
            "predecessor_execution_generation": 19,
            "execution_generation": EXECUTION_GENERATION,
            "predecessor_execution_retry_forbidden": True,
            "fresh_execution_and_output_identities_required": True,
        },
    )
    plan = {**body, "plan_sha256": self_hosted.digest_without(body, "plan_sha256")}
    validate_plan(plan, source)
    return plan


def validate_plan(plan: Mapping[str, Any], source: Mapping[str, Any]) -> None:
    source_rows = [row for row in source.get("attempts", []) if row.get("selection_rank") == RANK]
    rows = plan.get("attempts")
    tasks = plan.get("tasks")
    if any(
        (
            plan.get("receipt_sha256") is not None,
            plan.get("schema_version") != SCHEMA,
            plan.get("controller") != CONTROLLER,
            plan.get("campaign_id") != JOB_NAME,
            plan.get("source_job_id") != JOB_NAME,
            plan.get("sfs_root") != SFS_ROOT,
            plan.get("predecessor_plan_sha256") != SOURCE_PLAN_SHA256,
            plan.get("launch_authorized") is not False,
            plan.get("release_required") is not True,
            plan.get("model") != EXPECTED_MODEL,
            plan.get("harness") != EXPECTED_HARNESS,
            plan.get("treatment", {}).get("tools") != ["bash", "submit_report"],
            plan.get("treatment", {}).get("context_management")
            != "opencode_1.18.27_native_compaction_autocontinue_v1",
            plan.get("execution", {}).get("global_execution_claim_before_model_call") is not True,
            plan.get("execution", {}).get("automatic_retry") is not False,
            plan.get("execution", {}).get("same_task_max_inflight") != 1,
            plan.get("execution", {}).get("endpoint_lease", {}).get("maximum_streams") != 2,
            not isinstance(tasks, list),
            len(tasks or []) != 1,
            (tasks or [{}])[0].get("rank") != RANK,
            not isinstance(rows, list),
            len(rows or []) != 4,
            [row.get("attempt") for row in rows or []] != [1, 2, 3, 4],
            any(row.get("selection_rank") != RANK for row in rows or []),
            any(row.get("task_version_id") != TASK_VERSION_ID for row in rows or []),
            plan.get("plan_sha256") != self_hosted.digest_without(dict(plan), "plan_sha256"),
        )
    ):
        raise ValueError("held rank17 hosted plan drifted")
    for row, old in zip(rows or [], source_rows, strict=True):
        stable = {
            key: value
            for key, value in row.items()
            if key not in {"execution_generation", "execution_id", "run_id", "network"}
        }
        old_stable = {
            key: value
            for key, value in old.items()
            if key not in {"execution_generation", "execution_id", "run_id", "network"}
        }
        expected = exact.execution_for(str(row["cell_id"]), EXECUTION_GENERATION)
        if any(
            (
                stable != old_stable,
                row.get("execution_generation") != EXECUTION_GENERATION,
                row.get("execution_id") != expected["execution_id"],
                row.get("execution_id") == old.get("execution_id"),
                row.get("run_id") == old.get("run_id"),
            )
        ):
            raise ValueError("held rank17 execution identity drifted")


def held_receipt(root: Path) -> dict[str, Any]:
    plan = build_plan(root)
    return _seal(
        {
            "schema_version": HELD_SCHEMA,
            "status": "HELD_PENDING_FRESH_SCORE_BLIND_RELEASE",
            "launch_authorized": False,
            "scoring_authorized": False,
            "controller": CONTROLLER,
            "selection_rank": RANK,
            "task_version_id": TASK_VERSION_ID,
            "job_name": JOB_NAME,
            "configmap_name": CONFIGMAP_NAME,
            "sfs_root": SFS_ROOT,
            "diagnostic_root": DIAGNOSTIC_ROOT,
            "plan_sha256": plan["plan_sha256"],
            "cells": [
                {
                    "attempt": row["attempt"],
                    "cell_id": row["cell_id"],
                    "execution_id": row["execution_id"],
                    "run_id": row["run_id"],
                }
                for row in plan["attempts"]
            ],
            "required_fresh_release": {
                "authoritative_ledger_counts": {
                    "accepted": 51,
                    "active": 0,
                    "blocked_nonrepeatable": 8,
                    "unstarted": 341,
                },
                "all_four_cells_exactly_unstarted": True,
                "canonical_claim_collisions": 0,
                "accepted_receipt_collisions": 0,
                "authoritative_session_collisions": 0,
                "fresh_job_configmap_pod_collisions": 0,
                "fresh_sfs_output_collisions": 0,
                "endpoint_lease_slot_required": True,
                "uid_bound_observer_succeeded": True,
                "observer_restarts": 0,
                "independent_review_required": True,
            },
            "scientific_binding": {
                "model": EXPECTED_MODEL,
                "harness": EXPECTED_HARNESS,
                "tools": ["bash", "submit_report"],
                "tool_catalog_sha256": plan["treatment"]["tool_catalog_sha256"],
                "atomic_whole_task": True,
                "pass_k": 4,
            },
            "side_effects": {
                "claims_created": 0,
                "model_calls": 0,
                "task_calls": 0,
                "session_calls": 0,
                "verifier_calls": 0,
                "scoring_calls": 0,
                "kubernetes_mutations": 0,
            },
            "privacy": {
                "scores_read": False,
                "prompts_traces_flags_read": False,
                "credentials_included": False,
            },
        }
    )


def validate_held(value: Mapping[str, Any], root: Path) -> None:
    if dict(value) != held_receipt(root):
        raise ValueError("held rank17 receipt drifted")
