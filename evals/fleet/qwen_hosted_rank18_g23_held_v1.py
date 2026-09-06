"""Held whole-task plan for the next conservative hosted-Qwen task."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from evals.fleet import exact_pass4_universe as exact
from evals.fleet import qwen_hosted_rank17_g22_held_v1 as prior
from evals.fleet import self_hosted

SCHEMA = "fleet-qwen38-hosted-atomic-whole-task-plan-v5"
HELD_SCHEMA = "fleet-qwen38-hosted-rank18-g23-held-v1"
SOURCE_PATH = Path("evals/fleet/configs/qwen-hosted-generation19-qwen-b-v4.json")
SOURCE_PLAN_SHA256 = "sha256:c4dc5c95bb5ccc9e5d4e0ae14745d5533bb4c1d6eb64fa50d0cfe171007f93c5"
RANK = 18
TASK_VERSION_ID = "3aafc5a8-2144-4bc3-919f-fcbbebd9f0d7"
EXECUTION_GENERATION = 23
CONTROLLER = "qwen-b"
JOB_NAME = "chris-q38-hosted-r018-whole-task-g23-v1"
CONFIGMAP_NAME = "chris-q38-hosted-r018-whole-task-g23-package-v1"
SFS_ROOT = f"/mnt/sfs/jobs/{JOB_NAME}"
DIAGNOSTIC_ROOT = f"{SFS_ROOT}-diagnostic"
HELD_PATH = Path(
    "docs/evidence/qwen38-study/2026-09-06-qwen38-hosted-rank18-g23-held-v1.json"
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("rank18 source plan invalid")
    return value


def _fresh_attempt(row: Mapping[str, Any]) -> dict[str, Any]:
    item = copy.deepcopy(dict(row))
    execution = exact.execution_for(str(item["cell_id"]), EXECUTION_GENERATION)
    run_id = (
        f"chris-q38-ac-g23v1-b-r{RANK:03d}-a{item['attempt']}-"
        f"{execution['execution_id'][7:15]}"
    )
    item.update(
        execution_generation=EXECUTION_GENERATION,
        execution_id=execution["execution_id"],
        run_id=run_id,
        network=run_id.removeprefix("chris-")[:63],
    )
    return item


def _validate_source(source: Mapping[str, Any]) -> None:
    value = dict(source)
    if (
        value.get("plan_sha256") != SOURCE_PLAN_SHA256
        or self_hosted.digest_without(value, "plan_sha256") != SOURCE_PLAN_SHA256
    ):
        raise ValueError("rank18 source plan digest drifted")


def _expected_plan(source: Mapping[str, Any]) -> dict[str, Any]:
    _validate_source(source)
    tasks = [copy.deepcopy(row) for row in source.get("tasks", []) if row.get("rank") == RANK]
    attempts = [
        _fresh_attempt(row)
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
        attempts=attempts,
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
    return {**body, "plan_sha256": self_hosted.digest_without(body, "plan_sha256")}


def build_plan(root: Path) -> dict[str, Any]:
    source = _load(root / SOURCE_PATH)
    plan = _expected_plan(source)
    validate_plan(plan, source)
    return plan


def validate_plan(plan: Mapping[str, Any], source: Mapping[str, Any]) -> None:
    if dict(plan) != _expected_plan(source):
        raise ValueError("held rank18 hosted plan drifted")


def held_receipt(root: Path) -> dict[str, Any]:
    plan = build_plan(root)
    value = {
        "schema_version": HELD_SCHEMA,
        "status": "HELD_PENDING_CONSERVATIVE_SCORE_BLIND_RELEASE",
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
        "rank17_ledger_treatment": {
            "authoritative_tally_unchanged": True,
            "no_reclassification_by_rank18_plan": True,
            "rank17_release_held_on_metadata_contract": True,
        },
        "required_fresh_release": {
            "authoritative_ledger_counts": prior.held_receipt(root)[
                "required_fresh_release"
            ]["authoritative_ledger_counts"],
            "all_four_rank18_cells_exactly_unstarted": True,
            "exact_model_session_summary_rows": 0,
            "missing_or_malformed_session_models": 0,
            "canonical_claim_collisions": 0,
            "accepted_receipt_collisions": 0,
            "fresh_job_configmap_pod_collisions": 0,
            "fresh_sfs_output_collisions": 0,
            "endpoint_lease_slot_required": True,
            "streaming_allowlist_observer_succeeded": True,
            "observer_restarts": 0,
            "independent_review_required": True,
        },
        "scientific_binding": {
            "model": prior.EXPECTED_MODEL,
            "harness": prior.EXPECTED_HARNESS,
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
            "session_inventory_parser": "bounded_streaming_allowlist_v1",
            "verifier_execution_values_materialized": False,
            "scores_read": False,
            "prompts_traces_flags_read": False,
            "credentials_included": False,
        },
    }
    value["receipt_sha256"] = self_hosted.digest_without(value, "receipt_sha256")
    return value


def validate_held(value: Mapping[str, Any], root: Path) -> None:
    if dict(value) != held_receipt(root):
        raise ValueError("held rank18 receipt drifted")
