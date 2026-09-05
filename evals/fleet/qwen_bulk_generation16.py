"""Exact two-controller Qwen G17 successor after the accepted G15 canary.

This module is deliberately Qwen-only.  Rank 2 is reserved in its entirety for
the dedicated-serving block, and the accepted rank-4/attempt-1 G15 cell is
excluded.  The two hosted controllers therefore own exactly 395 cells.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from evals.fleet import exact_pass4_bulk_v3 as base
from evals.fleet import exact_pass4_universe as exact
from evals.fleet import self_hosted

SPEC_SCHEMA = "fleet-qwen-generation17-bulk-controller-spec-v1"
PLAN_SCHEMA = "fleet-qwen-generation17-bulk-controller-plan-v1"
MODULE_PATH = "evals/fleet/qwen_bulk_generation16.py"
RUNTIME_PATH = "evals/fleet/qwen_bulk_generation16_runtime.py"
RUN_PATH = "evals/fleet/scripts/run_qwen_bulk_generation16.sh"
MANIFEST_PATH = "evals/fleet/cluster/qwen-bulk-generation17.yaml"
G15_GATE_PATH = (
    "docs/evidence/qwen38-study/2026-09-05-qwen38-generation15-accepted-gate-v1.json"
)
EXECUTION_GENERATION = 17
FLEET_API_KEY_SECRET = "chris-cyber-opencode-evals-v2"
CLAIM_ROOT = base.CLAIM_ROOT
LEASE_ROOT = base.LEASE_ROOT
SHA256_RE = base.SHA256_RE
COMMIT_RE = base.COMMIT_RE
UUID_RE = base.UUID_RE
load = base.load
validate_inventory_gate = base.validate_inventory_gate

CONTROLLERS = {
    "qwen-a": {
        "model": "qwen3.8-27b",
        "job_name": "chris-q38-ac-exact100-g17-a199-v1",
        "configmap_name": "chris-q38-ac-exact100-g17-a199-run-v1",
        "serving_block": "qwen-hosted-autocontinue-v1",
        "full_ranks": list(range(1, 98, 2)),
        "partial_attempts": {4: [2, 3, 4]},
        "cell_count": 199,
    },
    "qwen-b": {
        "model": "qwen3.8-27b",
        "job_name": "chris-q38-ac-exact100-g17-b196-v1",
        "configmap_name": "chris-q38-ac-exact100-g17-b196-run-v1",
        "serving_block": "qwen-hosted-autocontinue-v1",
        "full_ranks": [*range(6, 99, 2), 99, 100],
        "partial_attempts": {},
        "cell_count": 196,
    },
}
SPEC_PATHS = {
    key: f"evals/fleet/configs/qwen-bulk-generation17-{key}.json" for key in CONTROLLERS
}


def _selected_keys(controller: str) -> list[tuple[str, int, int]]:
    row = CONTROLLERS[controller]
    selected: list[tuple[str, int, int]] = []
    partial = {int(rank): attempts for rank, attempts in row["partial_attempts"].items()}
    for rank in sorted(set(row["full_ranks"]) | set(partial)):
        for attempt in partial.get(rank, [1, 2, 3, 4]):
            selected.append((row["model"], rank, attempt))
    if len(selected) != row["cell_count"] or len(selected) != len(set(selected)):
        raise ValueError("Generation-17 controller partition drifted")
    return selected


def _compact_cells(controller: str, root: Path) -> list[dict[str, Any]]:
    cells = base._cells_by_key(root)  # noqa: SLF001 - immutable universe projection
    compact: list[dict[str, Any]] = []
    for key in _selected_keys(controller):
        row = cells[key]
        execution = exact.execution_for(row["cell_id"], EXECUTION_GENERATION)
        compact.append(
            {
                "cell_id": row["cell_id"],
                "selection_rank": row["selection_rank"],
                "attempt": row["attempt"],
                "task_version_id": row["task_version_id"],
                "execution_generation": EXECUTION_GENERATION,
                "execution_id": execution["execution_id"],
            }
        )
    return compact


def build_spec(controller: str, root: Path) -> dict[str, Any]:
    row = CONTROLLERS[controller]
    compact = _compact_cells(controller, root)
    body = {
        "schema_version": SPEC_SCHEMA,
        "controller": controller,
        "model": row["model"],
        "job_name": row["job_name"],
        "configmap_name": row["configmap_name"],
        "serving_block": row["serving_block"],
        "partition": {
            "full_ranks": row["full_ranks"],
            "partial_attempts": {str(k): v for k, v in row["partial_attempts"].items()},
            "cell_count": row["cell_count"],
            "rank2_reserved_for_dedicated": True,
            "g15_accepted_rank4_attempt1_excluded": True,
        },
        "cells_sha256": self_hosted.sha256(self_hosted.canonical_json(compact)),
        "execution": {
            "workers": 1,
            "same_task_max_inflight": 1,
            "attempts_per_task_sequential": True,
            "global_execution_claim_before_model_call": True,
            "claim_root": CLAIM_ROOT,
            "accepted_active_or_claimed_cells_are_nonrepeatable": True,
            "automatic_retry": False,
            "future_nonzero_exit_policy": base.NONZERO_EXIT_POLICY,
            "infrastructure_failure_policy": "quarantine_exact_cell_and_continue_tail",
            "tail_survives_single_cell_failure": True,
            "endpoint_lease": {
                "lease_root": LEASE_ROOT,
                "endpoint_key": row["serving_block"],
                "maximum_streams": 2,
            },
            "priority_class": "fleet-serve-low",
            "preemption_policy": "Never",
        },
        "treatment": exact.EXPECTED_TREATMENT,
        "launch_authorized": True,
    }
    return {**body, "spec_sha256": self_hosted.digest_without(body, "spec_sha256")}


def build_plan(spec: dict[str, Any], root: Path) -> dict[str, Any]:
    controller = spec["controller"]
    if spec != build_spec(controller, root):
        raise ValueError("Generation-17 source spec drifted")
    selected = exact.validate_selection(exact.read_object(root / base.CAMPAIGN_PATH), root)
    tasks = {row["rank"]: row for row in selected}
    attempts: list[dict[str, Any]] = []
    for ordinal, cell in enumerate(_compact_cells(controller, root), 1):
        task = tasks[cell["selection_rank"]]
        run_id = (
            f"chris-q38-ac-g17-{controller[-1]}-r{cell['selection_rank']:03d}-"
            f"a{cell['attempt']}-{cell['execution_id'][7:15]}"
        )
        attempts.append(
            {
                "ordinal": ordinal,
                **cell,
                "task_key": task["task_key"],
                "environment_version_id": task["environment_version_id"],
                "run_id": run_id,
                "network": run_id.removeprefix("chris-")[:63],
            }
        )
    body = {
        "schema_version": PLAN_SCHEMA,
        "campaign_id": spec["job_name"],
        "controller": controller,
        "job_name": spec["job_name"],
        "configmap_name": spec["configmap_name"],
        "sfs_root": f"/mnt/sfs/jobs/{spec['job_name']}",
        "model": exact.EXPECTED_MODELS["qwen3.8-27b"],
        "serving_block": spec["serving_block"],
        "task_count": len({row["selection_rank"] for row in attempts}),
        "new_session_count": len(attempts),
        "attempts": attempts,
        "execution": spec["execution"],
        "treatment": spec["treatment"],
        "release_required": True,
        "launch_authorized": True,
        "privacy": {
            "credentials_included": False,
            "prompts_included": False,
            "scores_included": False,
            "transcripts_included": False,
        },
    }
    return {**body, "plan_sha256": self_hosted.digest_without(body, "plan_sha256")}


def validate_all(root: Path) -> dict[str, dict[str, Any]]:
    plans: dict[str, dict[str, Any]] = {}
    identities: set[tuple[int, int]] = set()
    for controller, path in SPEC_PATHS.items():
        spec = load(root / path)
        if spec != build_spec(controller, root):
            raise ValueError("Generation-17 immutable spec drifted")
        plan = build_plan(spec, root)
        plans[controller] = plan
        for row in plan["attempts"]:
            identity = (row["selection_rank"], row["attempt"])
            if identity in identities:
                raise ValueError("Generation-17 partitions overlap")
            identities.add(identity)
    expected = {
        (rank, attempt)
        for rank in range(1, 101)
        if rank != 2
        for attempt in range(1, 5)
    } - {(4, 1)}
    if identities != expected or {k: v["new_session_count"] for k, v in plans.items()} != {
        "qwen-a": 199,
        "qwen-b": 196,
    }:
        raise ValueError("Generation-17 coverage drifted")
    return plans


def build_runtime_plan(
    controller: str, inventory_receipt: dict[str, Any], root: Path
) -> dict[str, Any]:
    validate_inventory_gate(inventory_receipt, root)
    compact = validate_all(root)[controller]
    template = load(root / base.RUNTIME_TEMPLATE_PATHS["qwen3.8-27b"])
    tasks_by_rank = {row["selection_rank"]: row for row in inventory_receipt["tasks"]}
    tasks: list[dict[str, Any]] = []
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
    execution = copy.deepcopy(compact["execution"])
    execution.update(
        required_task_tools=["bash", "submit_report"],
        required_task_tool_catalog_sha256=template["execution"][
            "required_task_tool_catalog_sha256"
        ],
        training_data_eligible=False,
    )
    harness = copy.deepcopy(template["harness"])
    harness["compaction_headroom_tokens"] = compact["treatment"]["compaction_headroom_tokens"]
    body = {
        "schema_version": "fleet-qwen-generation17-bulk-executable-plan-v1",
        "controller": controller,
        "campaign_id": compact["job_name"],
        "source_job_id": compact["job_name"],
        "sfs_root": compact["sfs_root"],
        "repo_root": ".",
        "inventory_receipt": inventory_receipt,
        "inventory_receipt_sha256": inventory_receipt["receipt_sha256"],
        "model": copy.deepcopy(template["model"]),
        "harness": harness,
        "authority": copy.deepcopy(template["authority"]),
        "serving_block": compact["serving_block"],
        "tasks": tasks,
        "attempts": copy.deepcopy(compact["attempts"]),
        "execution": execution,
        "treatment": compact["treatment"],
        "release_required": True,
        "launch_authorized": True,
        "privacy": compact["privacy"],
    }
    plan = {**body, "plan_sha256": self_hosted.digest_without(body, "plan_sha256")}
    settings = self_hosted.opencode_settings(plan)
    canonical = self_hosted.canonical_json(settings)
    if (
        plan["harness"]["settings_canonical_sha256"] != self_hosted.sha256(canonical)
        or plan["harness"]["settings_file_sha256"] != self_hosted.sha256(canonical + b"\n")
        or settings.get("compaction") != {"auto": True, "reserved": 20000}
        or "plugin" in settings
    ):
        raise ValueError("Generation-17 harness treatment drifted")
    return plan
