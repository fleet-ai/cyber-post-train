"""HELD authority for a hosted/dedicated exact-pass@4 bulk successor.

This module is intentionally side-effect free. It freezes the statistical-cell
partition and the gates a later release must satisfy; it cannot submit work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from evals.fleet import exact_pass4_bulk_v3 as v3
from evals.fleet import exact_pass4_universe as exact
from evals.fleet import glm53_dedicated_v7 as dedicated
from evals.fleet import self_hosted

SPEC_PATH = "evals/fleet/configs/exact-pass4-dedicated-bulk-v4-held.json"
MODULE_PATH = "evals/fleet/exact_pass4_dedicated_bulk_v4.py"
SUBMIT_PATH = "evals/fleet/scripts/submit_exact_pass4_dedicated_bulk_v4.sh"
HELD_PATH = "docs/evidence/qwen38-study/2026-09-05-opencode-exact-pass4-dedicated-bulk-held-v4.json"
SPEC_SCHEMA = "fleet-exact-pass4-dedicated-bulk-v4-held-spec-v1"
PLAN_SCHEMA = "fleet-exact-pass4-dedicated-bulk-v4-plan-v1"
CANARY_PLAN_SCHEMA = "fleet-exact-pass4-dedicated-pre-canary-v4-plan-v1"
HELD_SCHEMA = "fleet-exact-pass4-dedicated-bulk-v4-held-receipt-v1"
PREBULK_SCHEMA = "fleet-exact-pass4-dedicated-prebulk-v4"
PREBULK_ROOT = "/mnt/sfs/jobs/chris-cyber-exact100-prebulk-reconcile-v4"
SOURCE_COMMIT = "b2934446d93cf34facf5fd007216e6dedcc9c2b4"
G7_COMMIT = "e549ed9588159af9aaf6186e5b458a9eb070c114"


def load(path: Path) -> dict[str, Any]:
    return v3.load(path)


def file_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _ranks(row: dict[str, Any]) -> list[int]:
    if "explicit_ranks" in row:
        result = row["explicit_ranks"]
    else:
        bounds = row.get("ranks")
        step = row.get("rank_step")
        if (
            not isinstance(bounds, list)
            or len(bounds) != 2
            or not all(type(item) is int for item in bounds)
            or type(step) is not int
            or step < 1
        ):
            raise ValueError("bulk v4 rank selection is invalid")
        result = list(range(bounds[0], bounds[1] + 1, step))
    if (
        not isinstance(result, list)
        or not result
        or any(type(rank) is not int or not 1 <= rank <= 100 for rank in result)
        or len(result) != len(set(result))
    ):
        raise ValueError("bulk v4 rank selection is invalid")
    return result


def _cells(root: Path) -> dict[tuple[str, int, int], dict[str, Any]]:
    campaign = exact.read_object(root / v3.CAMPAIGN_PATH)
    universe = exact.build_universe(campaign, root)
    return {(row["model"], row["selection_rank"], row["attempt"]): row for row in universe["cells"]}


def validate_spec(value: dict[str, Any], root: Path) -> None:
    if (
        value.get("schema_version") != SPEC_SCHEMA
        or value.get("status") != "HELD"
        or value.get("launch_authorized") is not False
        or value.get("source_bulk_commit") != SOURCE_COMMIT
        or value.get("generation7_package_commit") != G7_COMMIT
        or value.get("prebulk_root") != PREBULK_ROOT
        or value.get("claim_root") != v3.CLAIM_ROOT
    ):
        raise ValueError("bulk v4 held identity drifted")
    if value.get("execution") != {
        "global_create_once_claim_before_model_call": True,
        "restart_safe_receipt_reconciliation": True,
        "automatic_retry": False,
        "same_task_max_inflight": 1,
        "attempts_per_task_sequential": True,
        "source_binding_required": True,
        "authoritative_session_match_count": 1,
        "verifier_execution_id_required": True,
        "session_ingest_completed_required": True,
        "cleanup_completed_required": True,
        "dedicated_server_uid_binding_required": True,
        "dedicated_controller_heartbeat_required": True,
        "post_ready_no_heartbeat_release_seconds": 600,
        "dedicated_controller_concurrency_after_runtime_gate": 2,
        "controller_cpu_only": True,
        "controller_priority_class": "fleet-serve-low",
        "controller_preemption_policy": "Never",
    }:
        raise ValueError("bulk v4 execution contract drifted")
    expected_treatment = {
        "hosted_and_dedicated_blocks_stratified": True,
        "pooling_across_serving_blocks_allowed": False,
        "task_crosses_serving_block": False,
        "harness": "opencode",
        "harness_version": "1.18.27",
        "tools": ["bash", "submit_report"],
        "context_window_size": 262144,
        "compaction_headroom_tokens": 20000,
        "max_model_requests": 600,
        "timeout_seconds": 28800,
    }
    if value.get("treatment") != expected_treatment:
        raise ValueError("bulk v4 treatment drifted")
    if value.get("privacy") != {
        "credentials_included": False,
        "prompts_traces_flags_or_scores_included": False,
    }:
        raise ValueError("bulk v4 privacy contract drifted")
    if value.get("release_order") != [
        "qwen_and_glm_generation7_acceptance",
        "fresh_v4_prebulk_reconciliation",
        "hosted_controllers",
        "dedicated_a_jobs_api_preview_and_duplicate_gate",
        "dedicated_a_non_scored_parity",
        "dedicated_a_rank51_attempt1_parity_only_scored_canary",
        "dedicated_a_post_canary_runtime_acceptance",
        "dedicated_a_bulk_streams",
        "dedicated_b_jobs_api_preview_and_duplicate_gate",
        "dedicated_b_non_scored_parity",
        "dedicated_b_rank76_attempt1_parity_only_scored_canary",
        "dedicated_b_post_canary_runtime_acceptance",
        "dedicated_b_bulk_streams",
    ]:
        raise ValueError("bulk v4 ramp sequence drifted")
    plans(value, root)


def _controller_cells(
    row: dict[str, Any], universe: dict[tuple[str, int, int], dict[str, Any]]
) -> list[dict[str, Any]]:
    partial = {int(rank): attempts for rank, attempts in row["partial_attempts"].items()}
    selected_ranks = sorted(set(_ranks(row)) | set(partial))
    canary = {"A": (51, 1), "B": (76, 1)}.get(row.get("replica"))
    selected = [
        universe[(row["model"], rank, attempt)]
        for rank in selected_ranks
        for attempt in partial.get(rank, [1, 2, 3, 4])
        if canary != (rank, attempt)
    ]
    if len(selected) != row["cell_count"]:
        raise ValueError(f"bulk v4 {row['name']} cell count drifted")
    return selected


def plans(value: dict[str, Any], root: Path) -> dict[str, dict[str, Any]]:
    rows = value.get("controllers")
    if not isinstance(rows, list) or len(rows) != 8:
        raise ValueError("bulk v4 controller set drifted")
    universe = _cells(root)
    dedicated_value = dedicated.spec(root)
    result: dict[str, dict[str, Any]] = {}
    seen_cells: set[str] = set()
    glm_rank_owner: dict[int, str] = {}
    for row in rows:
        name = row.get("name")
        if not isinstance(name, str) or name in result:
            raise ValueError("bulk v4 controller identity drifted")
        cells = _controller_cells(row, universe)
        for cell in cells:
            if cell["cell_id"] in seen_cells:
                raise ValueError("bulk v4 cell overlap")
            seen_cells.add(cell["cell_id"])
            if row["model"] == "glm-5.3":
                rank = cell["selection_rank"]
                prior = glm_rank_owner.setdefault(rank, row["serving_block"])
                if prior != row["serving_block"]:
                    raise ValueError("bulk v4 GLM task crosses serving treatment")
        dedicated_kind = row["serving_kind"] == "dedicated"
        if dedicated_kind:
            if row.get("replica") not in {"A", "B"} or row.get("stream") not in {1, 2}:
                raise ValueError("bulk v4 dedicated identity drifted")
        elif row["serving_kind"] != "hosted" or "replica" in row or "stream" in row:
            raise ValueError("bulk v4 hosted identity drifted")
        compact = [
            {
                "cell_id": cell["cell_id"],
                "selection_rank": cell["selection_rank"],
                "attempt": cell["attempt"],
                "task_key": cell["task_key"],
                "task_version_id": cell["task_version_id"],
                "execution_id": cell["initial_execution"]["execution_id"],
                "execution_generation": 1,
            }
            for cell in cells
        ]
        body = {
            "schema_version": PLAN_SCHEMA,
            "controller": name,
            "job_name": f"chris-cyber-exact100-{name}-v4",
            "sfs_root": f"/mnt/sfs/jobs/chris-cyber-exact100-{name}-v4",
            "stage": "dependent_bulk" if dedicated_kind else "hosted_bulk",
            "model": row["model"],
            "serving_kind": row["serving_kind"],
            "serving_block": row["serving_block"],
            "replica": row.get("replica"),
            "stream": row.get("stream"),
            "cells": compact,
            "cell_count": len(compact),
            "global_claim_root": v3.CLAIM_ROOT,
            "prebulk_gate": f"{PREBULK_ROOT}/TERMINAL.json",
            "generation7_gate": (
                f"{PREBULK_ROOT}/"
                f"{'qwen3.8-27b' if row['model'] == 'qwen3.8-27b' else 'glm-5.3'}"
                "-generation7-gate.json"
            ),
            "dedicated_runtime_gate": (
                f"{value['prebulk_root']}/glm-dedicated-"
                f"{row.get('replica', '').lower()}-runtime.json"
                if dedicated_kind
                else None
            ),
            "runtime_gate_required_before_any_scored_bulk_cell": dedicated_kind,
            "dedicated_server": (
                {
                    "title": dedicated_value["replicas"][row["replica"]]["title"],
                    "run_dir": dedicated_value["replicas"][row["replica"]]["run_dir"],
                    "request_sha256": self_hosted.sha256(
                        self_hosted.canonical_json(
                            dedicated.server_payload(dedicated_value, root, row["replica"])
                        )
                    ),
                    "jobs_api_observation_receipt_sha256": dedicated_value["jobs_api"][
                        "observation_receipt_sha256"
                    ],
                    "priority_class": "fleet-infra-quiet",
                    "preemption_policy": "Never",
                }
                if dedicated_kind
                else None
            ),
            "heartbeat_script": dedicated.HEARTBEAT_PATH if dedicated_kind else None,
            "heartbeat_relative_path": (
                f"lifecycle/traffic-stream-{row['stream']}" if dedicated_kind else None
            ),
            "qualified_controller_concurrency": 2 if dedicated_kind else None,
            "concurrency_qualification_receipt": (
                f"{value['prebulk_root']}/glm-dedicated-"
                f"{row.get('replica', '').lower()}-parity.json"
                if dedicated_kind
                else None
            ),
            "automatic_retry": False,
            "controller_resource_policy": {
                "cpu_only": True,
                "gpus": 0,
                "priority_class": "fleet-serve-low",
                "preemption_policy": "Never",
            },
            "launch_authorized": False,
        }
        result[name] = {
            **body,
            "plan_sha256": self_hosted.sha256(self_hosted.canonical_json(body)),
        }
    expected = set(universe) - {
        (model, canary["selection_rank"], canary["attempt"])
        for model, canary in v3.CANARIES.items()
    } - {("glm-5.3", 51, 1), ("glm-5.3", 76, 1)}
    selected = {
        (plan["model"], cell["selection_rank"], cell["attempt"])
        for plan in result.values()
        for cell in plan["cells"]
    }
    if selected != expected or len(seen_cells) != 796:
        raise ValueError("bulk v4 dependent plans do not cover exact post-canary universe")
    if set(glm_rank_owner) != set(range(1, 101)):
        raise ValueError("bulk v4 GLM rank ownership drifted")
    return result


def scored_canary_plans(value: dict[str, Any], root: Path) -> dict[str, dict[str, Any]]:
    """Materialize the two one-cell scored canaries gated only by live parity."""
    dependent = plans(value, root)
    universe = _cells(root)
    result: dict[str, dict[str, Any]] = {}
    for replica, rank in (("A", 51), ("B", 76)):
        lower = replica.lower()
        cell = universe[("glm-5.3", rank, 1)]
        compact = {
            "cell_id": cell["cell_id"],
            "selection_rank": rank,
            "attempt": 1,
            "task_key": cell["task_key"],
            "task_version_id": cell["task_version_id"],
            "execution_id": cell["initial_execution"]["execution_id"],
            "execution_generation": 1,
        }
        body = {
            "schema_version": CANARY_PLAN_SCHEMA,
            "controller": f"glm-dedicated-{lower}-canary",
            "job_name": f"chris-cyber-exact100-glm-dedicated-{lower}-canary-v4",
            "sfs_root": f"/mnt/sfs/jobs/chris-cyber-exact100-glm-dedicated-{lower}-canary-v4",
            "stage": "parity_only_scored_canary",
            "model": "glm-5.3",
            "serving_kind": "dedicated",
            "serving_block": f"glm-dedicated-{lower}-v7",
            "replica": replica,
            "cells": [compact],
            "cell_count": 1,
            "global_claim_root": v3.CLAIM_ROOT,
            "prebulk_gate": f"{PREBULK_ROOT}/TERMINAL.json",
            "generation7_gate": f"{PREBULK_ROOT}/glm-5.3-generation7-gate.json",
            "parity_gate": f"{PREBULK_ROOT}/glm-dedicated-{lower}-parity.json",
            "runtime_gate": None,
            "runtime_gate_required_before_canary": False,
            "post_canary_runtime_gate": f"{PREBULK_ROOT}/glm-dedicated-{lower}-runtime.json",
            "dependent_controller_plan_sha256": {
                name: plan["plan_sha256"]
                for name, plan in dependent.items()
                if name.startswith(f"glm-dedicated-{lower}-")
            },
            "heartbeat_script": dedicated.HEARTBEAT_PATH,
            "heartbeat_relative_path": "lifecycle/traffic-stream-1",
            "controller_concurrency_limit": 1,
            "automatic_retry": False,
            "controller_resource_policy": {
                "cpu_only": True,
                "gpus": 0,
                "priority_class": "fleet-serve-low",
                "preemption_policy": "Never",
            },
            "launch_authorized": False,
        }
        result[replica] = {
            **body,
            "plan_sha256": self_hosted.sha256(self_hosted.canonical_json(body)),
        }
    selected = {
        (plan["model"], cell["selection_rank"], cell["attempt"])
        for plan in plans(value, root).values()
        for cell in plan["cells"]
    }
    selected.update(
        (plan["model"], cell["selection_rank"], cell["attempt"])
        for plan in result.values()
        for cell in plan["cells"]
    )
    expected = set(_cells(root)) - {
        (model, canary["selection_rank"], canary["attempt"])
        for model, canary in v3.CANARIES.items()
    }
    if selected != expected or len(selected) != 798:
        raise ValueError("bulk v4 canary plus dependent plans do not cover 798 cells")
    return result


def prebulk_requirements(value: dict[str, Any], root: Path) -> dict[str, Any]:
    built = plans(value, root)
    canaries = scored_canary_plans(value, root)
    return {
        "schema_version": PREBULK_SCHEMA,
        "status": "REQUIRED_BEFORE_RELEASE",
        "root": PREBULK_ROOT,
        "controller_plan_sha256": {name: plan["plan_sha256"] for name, plan in built.items()},
        "scored_canary_plan_sha256": {
            replica: plan["plan_sha256"] for replica, plan in canaries.items()
        },
        "required_checks": [
            "exact_inventory_digest_and_100_rank_order",
            "qwen_generation7_rank4_attempt1_accepted_once",
            "glm_generation7_rank13_attempt1_accepted_once",
            "all_798_planned_cells_have_zero_accepted_active_claimed_or_model_started_collisions",
            "all_hosted_v3_has_no_release_or_active_controller",
            "global_claim_root_matches_generation7",
            "model_task_harness_tool_and_context_bindings_match",
            "dedicated_tasks_are_whole_task_stratified_blocks",
            "no_prompts_traces_flags_scores_or_credentials_persisted",
        ],
        "jobs_api_server_receipts_required": {
            "A": [
                "authenticated_preview",
                "duplicate_gate",
                "uid_parity",
                "parity_only_scored_canary",
                "post_canary_runtime_gate",
            ],
            "B": [
                "authenticated_preview",
                "duplicate_gate",
                "uid_parity",
                "parity_only_scored_canary",
                "post_canary_runtime_gate",
            ],
        },
        "launch_authorized": False,
    }


def held_receipt(root: Path) -> dict[str, Any]:
    value = load(root / SPEC_PATH)
    validate_spec(value, root)
    built = plans(value, root)
    canaries = scored_canary_plans(value, root)
    receipt = {
        "schema_version": HELD_SCHEMA,
        "status": "HELD",
        "launch_authorized": False,
        "spec_path": SPEC_PATH,
        "spec_file_sha256": file_sha256(root / SPEC_PATH),
        "module_path": MODULE_PATH,
        "module_file_sha256": file_sha256(root / MODULE_PATH),
        "submit_path": SUBMIT_PATH,
        "submit_file_sha256": file_sha256(root / SUBMIT_PATH),
        "source_bulk_commit": SOURCE_COMMIT,
        "generation7_package_commit": G7_COMMIT,
        "dedicated_v7_spec_sha256": file_sha256(root / dedicated.SPEC_PATH),
        "generation7_acceptance_gates": {
            "qwen3.8-27b": f"{PREBULK_ROOT}/qwen3.8-27b-generation7-gate.json",
            "glm-5.3": f"{PREBULK_ROOT}/glm-5.3-generation7-gate.json",
        },
        "jobs_api_observation_receipt_sha256": dedicated.spec(root)["jobs_api"][
            "observation_receipt_sha256"
        ],
        "controller_counts": {name: plan["cell_count"] for name, plan in built.items()},
        "controller_plan_sha256": {name: plan["plan_sha256"] for name, plan in built.items()},
        "scored_canary_counts": {replica: plan["cell_count"] for replica, plan in canaries.items()},
        "scored_canary_plan_sha256": {
            replica: plan["plan_sha256"] for replica, plan in canaries.items()
        },
        "qwen_hosted_cells": 399,
        "glm_hosted_cells": 199,
        "glm_dedicated_a_bulk_cells": 99,
        "glm_dedicated_b_bulk_cells": 99,
        "glm_dedicated_scored_canary_cells": 2,
        "total_cells": 798,
        "prebulk": prebulk_requirements(value, root),
        "dedicated_ramp": {
            "A_canary": {"selection_rank": 51, "attempt": 1},
            "B_canary": {"selection_rank": 76, "attempt": 1},
            "A_before_B": True,
            "maximum_nodes": 2,
            "maximum_gpus": 16,
            "no_heartbeat_release_seconds": 600,
        },
        "objects_created": False,
        "privacy": value["privacy"],
    }
    receipt["receipt_sha256"] = self_hosted.digest_without(receipt, "receipt_sha256")
    return receipt


def validate_held(receipt: dict[str, Any], root: Path) -> None:
    if receipt != held_receipt(root):
        raise ValueError("bulk v4 held receipt drifted")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preview", "render-held", "validate"))
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.repo.resolve(strict=True)
    if args.command == "validate":
        validate_held(load(root / HELD_PATH), root)
        output = {"status": "HELD", "valid": True, "objects_created": False}
    elif args.command == "render-held":
        output = held_receipt(root)
    else:
        value = load(root / SPEC_PATH)
        validate_spec(value, root)
        output = {
            "status": "HELD",
            "launch_authorized": False,
            "controller_counts": {
                name: plan["cell_count"] for name, plan in plans(value, root).items()
            },
            "scored_canary_counts": {
                replica: plan["cell_count"]
                for replica, plan in scored_canary_plans(value, root).items()
            },
            "total_cells": 798,
            "objects_created": False,
        }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
