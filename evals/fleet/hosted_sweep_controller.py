"""Create and run duplicate-safe hosted OpenCode task-boundary shards."""

from __future__ import annotations

import argparse
import copy
import json
import os
import threading
import uuid
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

import httpx

from evals.fleet import opencode_train_sweep_runner as legacy
from evals.fleet import self_hosted

PLAN_SCHEMA = "fleet-hosted-opencode-task-boundary-shard-v1"
SOURCE_SCHEMA = "opencode-hosted-successor-source-v1"
REMAINDER_SOURCE_SCHEMA = "hosted-opencode-remainder-source-v1"
CAMPAIGNS = {
    "qwen38": "chris-cyber-q38-opencode11827-hosted-complete49-p4-v5",
    "glm53": "chris-cyber-glm53-opencode11827-hosted-complete99-p4-v5",
    "glm53_clean": "chris-cyber-glm53-opencode11827-hosted-complete98-p4-v6",
    "glm53_hosted_odd": "chris-cyber-glm53-opencode11827-hosted-odd49-p4-v7",
}
SOURCE_MODEL_KEYS = {
    "qwen38": "qwen38",
    "glm53": "glm53",
    "glm53_clean": "glm53",
    "glm53_hosted_odd": "glm53",
}
EXPECTED_SOURCE_TASK_COUNTS = {
    "qwen38": 50,
    "glm53": 100,
    "glm53_clean": 100,
    "glm53_hosted_odd": 100,
}
EXPECTED_INCLUDED_TASK_COUNTS = {
    "qwen38": 49,
    "glm53": 99,
    "glm53_clean": 98,
    "glm53_hosted_odd": 49,
}
ADDITIONAL_DEFERRED_SOURCE_RANKS = {
    "qwen38": set(),
    "glm53": set(),
    "glm53_clean": {1},
    "glm53_hosted_odd": {1},
}
RESERVED_SOURCE_RANKS = {
    "qwen38": set(),
    "glm53": set(),
    "glm53_clean": set(),
    "glm53_hosted_odd": set(range(4, 101, 2)),
}
REMAINDER_CAMPAIGNS = {
    "qwen38_remainder": "chris-cyber-q38-opencode11827-hosted-complete48-p4-v6",
    "glm53_remainder": "chris-cyber-glm53-opencode11827-hosted-odd47-p4-v8",
    "qwen38_remainder2": "chris-cyber-q38-opencode11827-hosted-complete47-p4-v7",
    "glm53_remainder2": "chris-cyber-glm53-opencode11827-hosted-odd46-p4-v9",
}
EXPECTED_INCLUDED_TASK_COUNTS.update(
    {
        "qwen38_remainder": 48,
        "glm53_remainder": 47,
        "qwen38_remainder2": 47,
        "glm53_remainder2": 46,
    }
)
SCHEDULE = [
    {
        "accepted_outcomes_at_least": 0,
        "workers": 1,
        "headroom_gate": "fixed_at_launch_no_automatic_widening",
    },
]


def _emit(event: str, **fields: Any) -> None:
    """Emit only score- and content-blind controller progress."""
    print(json.dumps({"event": event, **fields}, sort_keys=True), flush=True)


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def digest_without(value: dict[str, Any], field: str) -> str:
    return self_hosted.digest_without(value, field)


def validate_source(value: dict[str, Any]) -> None:
    if value.get("schema_version") != SOURCE_SCHEMA:
        raise ValueError("unsupported hosted successor source schema")
    if value.get("receipt_sha256") != digest_without(value, "receipt_sha256"):
        raise ValueError("hosted successor source digest mismatch")
    if value.get("scores_read") is not False or value.get("prompts_or_traces_read") is not False:
        raise ValueError("hosted successor source crossed the sealed-content boundary")


def validate_remainder_source(value: dict[str, Any]) -> None:
    if value.get("schema_version") != REMAINDER_SOURCE_SCHEMA:
        raise ValueError("unsupported hosted remainder source schema")
    if value.get("receipt_sha256") != digest_without(value, "receipt_sha256"):
        raise ValueError("hosted remainder source digest mismatch")
    if value.get("scores_read") is not False or value.get("prompts_or_traces_read") is not False:
        raise ValueError("hosted remainder source crossed the sealed-content boundary")


def build_remainder_plan(
    predecessor: dict[str, Any], source: dict[str, Any], model_key: str
) -> dict[str, Any]:
    validate_plan(predecessor)
    validate_remainder_source(source)
    if model_key not in REMAINDER_CAMPAIGNS:
        raise ValueError("unsupported hosted remainder shard")
    source_key = model_key.split("_", 1)[0]
    evidence = source["runs"][source_key]
    if (
        evidence.get("predecessor_plan_sha256") != predecessor["plan_sha256"]
        or evidence.get("active_attempts") != 0
        or not evidence.get("job_uid")
        or not evidence.get("pod_uid")
    ):
        raise ValueError("hosted remainder predecessor is not terminal-bound")
    excluded_source_ranks = {int(rank) for rank in evidence["excluded_source_ranks"]}
    expected_excluded_by_model = {
        "qwen38_remainder": {2},
        "glm53_remainder": {3, 5},
        "qwen38_remainder2": {3},
        "glm53_remainder2": {7},
    }
    expected_excluded = expected_excluded_by_model[model_key]
    if excluded_source_ranks != expected_excluded:
        raise ValueError("hosted remainder excluded-task boundary drifted")
    for outcome in evidence["outcomes"]:
        if int(outcome["source_rank"]) not in excluded_source_ranks:
            raise ValueError("hosted remainder outcome escaped excluded task")
        if outcome.get("retry_allowed") is not False:
            raise ValueError("hosted remainder outcome became retryable")

    tasks = []
    for source_task in predecessor["tasks"]:
        source_rank = int(source_task["source_rank"])
        if source_rank in excluded_source_ranks:
            continue
        task = copy.deepcopy(source_task)
        task["rank"] = len(tasks) + 1
        tasks.append(task)
    if len(tasks) != EXPECTED_INCLUDED_TASK_COUNTS[model_key]:
        raise ValueError("hosted remainder included-task count drifted")

    campaign = REMAINDER_CAMPAIGNS[model_key]
    attempts = []
    for task in tasks:
        rank = int(task["rank"])
        source_rank = int(task["source_rank"])
        key_digest = self_hosted.sha256(task["task"]["key"].encode()).split(":", 1)[1][:8]
        for attempt_number in range(1, 5):
            attempts.append(
                {
                    "ordinal": len(attempts) + 1,
                    "rank": rank,
                    "source_rank": source_rank,
                    "attempt": attempt_number,
                    "run_id": f"{campaign}-sr{source_rank:03d}-a{attempt_number}-{key_digest}",
                    "network": f"{model_key}-sr{source_rank:03d}-a{attempt_number}-{key_digest}",
                }
            )
    predecessor_tasks = {int(row["source_rank"]): row for row in predecessor["tasks"]}
    excluded_tasks = [
        {
            "source_rank": source_rank,
            "task_key": predecessor_tasks[source_rank]["task"]["key"],
            "task_version_id": predecessor_tasks[source_rank]["task"]["version_id"],
            "reason": "predecessor_task_partially_touched_and_fenced",
            "retry_allowed": False,
            "credited": False,
        }
        for source_rank in sorted(excluded_source_ranks)
    ]
    plan = {
        "schema_version": PLAN_SCHEMA,
        "shard_key": model_key,
        "campaign_id": campaign,
        "source_job_id": predecessor["source_job_id"],
        "source": {
            "predecessor_plan_sha256": predecessor["plan_sha256"],
            "source_state_receipt_sha256": source["receipt_sha256"],
            "source_job_name": evidence["job_name"],
            "source_job_uid": evidence["job_uid"],
            "source_pod_uid": evidence["pod_uid"],
        },
        "treatment_block": predecessor["treatment_block"],
        "model": predecessor["model"],
        "harness": predecessor["harness"],
        "authority": predecessor["authority"],
        "task_count": len(tasks),
        "pass_k": 4,
        "total_session_count": len(tasks) * 4,
        "credited_sessions": [],
        "new_session_count": len(attempts),
        "upstream_excluded_tasks": predecessor["excluded_tasks"],
        "excluded_tasks": excluded_tasks,
        "execution": {
            **predecessor["execution"],
            "inventory_policy": (
                "plan_identity_plus_authoritative_receipt_v1"
                if model_key.startswith("qwen38_")
                else "conservative_no_same_model_session_for_task_key_v1"
            ),
        },
        "tasks": tasks,
        "attempts": attempts,
        "privacy": predecessor["privacy"],
    }
    if predecessor.get("reserved_tasks"):
        plan["reserved_tasks"] = predecessor["reserved_tasks"]
    plan["plan_sha256"] = digest_without(plan, "plan_sha256")
    validate_plan(plan)
    return plan


def _validate_source_plan(plan: dict[str, Any]) -> None:
    if plan.get("schema_version") == legacy.PLAN_SCHEMA:
        legacy.validate_plan(plan)
    elif plan.get("schema_version") == legacy.PARALLEL_PLAN_SCHEMA:
        legacy.validate_parallel_plan(plan)
    else:
        raise ValueError("unsupported hosted predecessor plan")


def build_plan(
    source_plan: dict[str, Any], source: dict[str, Any], model_key: str
) -> dict[str, Any]:
    validate_source(source)
    _validate_source_plan(source_plan)
    if model_key not in CAMPAIGNS:
        raise ValueError("unsupported hosted model shard")
    evidence = source["runs"][SOURCE_MODEL_KEYS[model_key]]
    if (
        evidence.get("source_plan_sha256") != source_plan["plan_sha256"]
        or evidence.get("source_terminal") is not True
        or evidence.get("source_active_attempts") != 0
        or len(evidence.get("fenced_noncreditable") or []) != 1
    ):
        raise ValueError("hosted predecessor is not exactly fenced")
    if int(source_plan["task_count"]) != EXPECTED_SOURCE_TASK_COUNTS[model_key]:
        raise ValueError("hosted predecessor task count drifted")
    evidence_excluded = {int(rank) for rank in evidence["excluded_source_ranks"]}
    blocked = evidence["fenced_noncreditable"][0]
    if evidence_excluded != {int(blocked["source_rank"])}:
        raise ValueError("fenced cell must exclude its complete source task")
    deferred = ADDITIONAL_DEFERRED_SOURCE_RANKS[model_key]
    if deferred & evidence_excluded:
        raise ValueError("deferred source tasks overlap fenced source tasks")
    reserved = RESERVED_SOURCE_RANKS[model_key]
    if reserved & (evidence_excluded | deferred):
        raise ValueError("reserved source tasks overlap fenced source tasks")
    excluded = evidence_excluded | deferred | reserved
    if (
        blocked.get("agent_exit_code") != 1
        or blocked.get("session_ingest_completed") is not True
        or blocked.get("cleanup_completed") is not True
        or blocked.get("retry_allowed") is not False
        or blocked.get("credited") is not False
    ):
        raise ValueError("fenced noncreditable outcome policy drifted")

    source_tasks = {int(row["rank"]): row for row in source_plan["tasks"]}
    source_attempts = {row["run_id"]: row for row in source_plan["attempts"]}
    accepted = evidence.get("accepted") or []
    for row in accepted:
        attempt = source_attempts.get(row["run_id"])
        if attempt is None or (int(attempt["rank"]), int(attempt["attempt"])) != (
            int(row["source_rank"]),
            int(row["attempt"]),
        ):
            raise ValueError("accepted predecessor outcome is not plan-bound")

    credit_rows = [evidence["credited_smoke"], *accepted]
    credits_by_source_rank: dict[int, list[dict[str, Any]]] = {}
    for row in credit_rows:
        credits_by_source_rank.setdefault(int(row["source_rank"]), []).append(row)

    tasks: list[dict[str, Any]] = []
    credits: list[dict[str, Any]] = []
    source_rank_to_rank: dict[int, int] = {}
    for source_rank in sorted(source_tasks):
        if source_rank in excluded:
            continue
        rank = len(tasks) + 1
        source_rank_to_rank[source_rank] = rank
        task = copy.deepcopy(source_tasks[source_rank])
        task["rank"] = rank
        task["source_rank"] = source_rank
        task.pop("baseline_session_ids", None)
        tasks.append(task)
        for row in credits_by_source_rank.get(source_rank, []):
            credits.append(
                {
                    "rank": rank,
                    "source_rank": source_rank,
                    "attempt": int(row["attempt"]),
                    "session_id": row["session_id"],
                    "verifier_execution_id": row["verifier_execution_id"],
                    "source_run_id": row.get("run_id", "credited-smoke"),
                }
            )
    if len(tasks) != EXPECTED_INCLUDED_TASK_COUNTS[model_key]:
        raise ValueError("hosted included task count drifted")

    credited_cells = {(int(row["rank"]), int(row["attempt"])) for row in credits}
    attempts: list[dict[str, Any]] = []
    campaign = CAMPAIGNS[model_key]
    for task in tasks:
        rank = int(task["rank"])
        source_rank = int(task["source_rank"])
        key_digest = self_hosted.sha256(task["task"]["key"].encode()).split(":", 1)[1][:8]
        for attempt_number in range(1, 5):
            if (rank, attempt_number) in credited_cells:
                continue
            attempts.append(
                {
                    "ordinal": len(attempts) + 1,
                    "rank": rank,
                    "source_rank": source_rank,
                    "attempt": attempt_number,
                    "run_id": (
                        f"{campaign}-sr{source_rank:03d}-a{attempt_number}-{key_digest}"
                    ),
                    "network": (
                        f"{model_key}-hosted-"
                        f"{'v7' if model_key == 'glm53_hosted_odd' else 'v5'}-"
                        f"sr{source_rank:03d}-a{attempt_number}-{key_digest}"
                    ),
                }
            )

    blocked_task = source_tasks[int(blocked["source_rank"])]
    plan = {
        "schema_version": PLAN_SCHEMA,
        "campaign_id": campaign,
        "source_job_id": source_plan["source_job_id"],
        "source": {
            "source_plan_sha256": source_plan["plan_sha256"],
            "source_state_receipt_sha256": source["receipt_sha256"],
            "source_job_name": evidence["source_job_name"],
            "source_job_uid": evidence["source_job_uid"],
            "source_pod_uid": evidence["source_pod_uid"],
        },
        "treatment_block": {
            "kind": "hosted_inference_endpoint_v1",
            "endpoint_origin": source_plan["model"]["endpoint_origin"],
            "served_id": source_plan["model"]["served_id"],
            "model_revision": source_plan["model"]["revision"],
            "session_model": source_plan["model"]["session_model"],
            "harness": source_plan["harness"],
            "required_task_tools": source_plan["execution"]["required_task_tools"],
            "required_task_tool_catalog_sha256": source_plan["execution"][
                "required_task_tool_catalog_sha256"
            ],
        },
        "model": source_plan["model"],
        "harness": source_plan["harness"],
        "authority": source_plan["authority"],
        "task_count": len(tasks),
        "pass_k": 4,
        "total_session_count": len(tasks) * 4,
        "credited_sessions": credits,
        "new_session_count": len(attempts),
        "excluded_tasks": [
            {
                "source_rank": int(blocked["source_rank"]),
                "task_key": blocked_task["task"]["key"],
                "task_version_id": blocked_task["task"]["version_id"],
                "reason": "fenced_verifier_backed_ingested_cleaned_opencode_exit_1",
                "run_id": blocked["run_id"],
                "session_id": blocked["session_id"],
                "verifier_execution_id": blocked["verifier_execution_id"],
                "retry_allowed": False,
                "credited": False,
            }
        ],
        "execution": {
            "task_partition": "complete_task_boundary",
            "same_task_max_inflight": 1,
            "retry_policy": "never_repeat_any_verifier_backed_outcome",
            "future_nonzero_exit_policy": "fence_task_and_continue_other_tasks",
            "training_data_eligible": True,
            "required_task_tools": source_plan["execution"]["required_task_tools"],
            "required_task_tool_catalog_sha256": source_plan["execution"][
                "required_task_tool_catalog_sha256"
            ],
            "score_blind_concurrency_schedule": SCHEDULE,
        },
        "tasks": tasks,
        "attempts": attempts,
        "privacy": {
            "scores_included": False,
            "prompts_included": False,
            "transcripts_included": False,
            "credentials_included": False,
        },
    }
    if model_key in {"glm53_clean", "glm53_hosted_odd"}:
        plan["shard_key"] = model_key
        plan["execution"]["inventory_policy"] = (
            "conservative_no_same_model_session_for_task_key_v1"
        )
    for source_rank in sorted(deferred):
        task = source_tasks[source_rank]
        plan["excluded_tasks"].append(
            {
                "source_rank": source_rank,
                "task_key": task["task"]["key"],
                "task_version_id": task["task"]["version_id"],
                "reason": "retained_history_requires_authoritative_metadata_projection",
                "retry_allowed": False,
                "credited": False,
            }
        )
    if reserved:
        plan["reserved_tasks"] = [
            {
                "source_rank": source_rank,
                "task_key": source_tasks[source_rank]["task"]["key"],
                "task_version_id": source_tasks[source_rank]["task"]["version_id"],
                "reserved_treatment_block": "dedicated_inference_endpoint_v1",
            }
            for source_rank in sorted(reserved)
        ]
    plan["plan_sha256"] = digest_without(plan, "plan_sha256")
    validate_plan(plan)
    return plan


def validate_plan(plan: dict[str, Any]) -> None:
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("unsupported hosted shard plan schema")
    if plan.get("plan_sha256") != digest_without(plan, "plan_sha256"):
        raise ValueError("hosted shard plan digest mismatch")
    campaign_to_shard = {
        value: key for key, value in {**CAMPAIGNS, **REMAINDER_CAMPAIGNS}.items()
    }
    shard_key = plan.get("shard_key") or campaign_to_shard.get(plan.get("campaign_id"))
    task_count = int(plan.get("task_count") or 0)
    if (
        shard_key not in EXPECTED_INCLUDED_TASK_COUNTS
        or task_count != EXPECTED_INCLUDED_TASK_COUNTS[shard_key]
        or plan.get("pass_k") != 4
    ):
        raise ValueError("hosted shard task/pass shape drifted")
    tasks = plan.get("tasks") or []
    if [row.get("rank") for row in tasks] != list(range(1, task_count + 1)):
        raise ValueError("hosted shard task ranks drifted")
    credits = plan.get("credited_sessions") or []
    attempts = plan.get("attempts") or []
    cells = [
        *[(int(row["rank"]), int(row["attempt"])) for row in credits],
        *[(int(row["rank"]), int(row["attempt"])) for row in attempts],
    ]
    expected = {(rank, attempt) for rank in range(1, task_count + 1) for attempt in range(1, 5)}
    if len(cells) != len(set(cells)) or set(cells) != expected:
        raise ValueError("hosted shard Cartesian cells drifted")
    if len(attempts) != int(plan["new_session_count"]):
        raise ValueError("hosted shard new-session count drifted")
    if int(plan["total_session_count"]) != task_count * 4:
        raise ValueError("hosted shard total-session count drifted")
    if [row.get("ordinal") for row in attempts] != list(range(1, len(attempts) + 1)):
        raise ValueError("hosted shard attempt ordinals drifted")
    if len({row["run_id"] for row in attempts}) != len(attempts):
        raise ValueError("hosted shard run identities drifted")
    execution = plan.get("execution") or {}
    expected_inventory_policy = (
        "conservative_no_same_model_session_for_task_key_v1"
        if shard_key
        in {"glm53_clean", "glm53_hosted_odd", "glm53_remainder", "glm53_remainder2"}
        else (
            "plan_identity_plus_authoritative_receipt_v1"
            if shard_key in {"qwen38_remainder", "qwen38_remainder2"}
            else None
        )
    )
    if (
        execution.get("task_partition") != "complete_task_boundary"
        or execution.get("same_task_max_inflight") != 1
        or execution.get("score_blind_concurrency_schedule") != SCHEDULE
        or execution.get("required_task_tools") != ["bash", "submit_report"]
        or execution.get("inventory_policy") != expected_inventory_policy
    ):
        raise ValueError("hosted shard execution policy drifted")
    excluded = plan.get("excluded_tasks") or []
    expected_excluded = (
        2
        if shard_key in {"glm53_clean", "glm53_hosted_odd", "glm53_remainder"}
        else 1
    )
    if len(excluded) != expected_excluded or any(
        row.get("retry_allowed") is not False for row in excluded
    ):
        raise ValueError("hosted shard fenced-task policy drifted")
    included_keys = {row["task"]["key"] for row in tasks}
    if any(row["task_key"] in included_keys for row in excluded):
        raise ValueError("fenced task leaked into hosted shard")
    reserved = plan.get("reserved_tasks") or []
    expected_reserved = (
        49
        if shard_key in {"glm53_hosted_odd", "glm53_remainder", "glm53_remainder2"}
        else 0
    )
    if (
        len(reserved) != expected_reserved
        or any(row["task_key"] in included_keys for row in reserved)
        or any(
            row.get("reserved_treatment_block") != "dedicated_inference_endpoint_v1"
            for row in reserved
        )
    ):
        raise ValueError("hosted shard reserved-task policy drifted")


def _client(key: str) -> httpx.Client:
    return httpx.Client(
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        timeout=1800,
    )


def _exact_treatment_sessions(
    client: httpx.Client, plan: dict[str, Any], task_key: str
) -> list[dict[str, Any]]:
    model = self_hosted.persisted_session_model_identity(plan)
    harness = f"opencode-{plan['harness']['version']}"
    tool_digest = plan["execution"]["required_task_tool_catalog_sha256"]
    exact = []
    for row in self_hosted._task_sessions(client, task_key):
        metadata = row.get("metadata") or {}
        if (
            row.get("model") == model
            and metadata.get("self_hosted_harness") == harness
            and metadata.get("tool_catalog_sha256") == tool_digest
        ):
            exact.append(row)
    return exact


def _credits_for_rank(plan: dict[str, Any], rank: int) -> list[dict[str, Any]]:
    return [row for row in plan["credited_sessions"] if int(row["rank"]) == rank]


def _accepted(plan: dict[str, Any], root: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    paths = list((root / "attempts").glob("*/ACCEPTED.json"))
    if paths:
        persisted_plan = load_object(root / "PLAN.json")
        if persisted_plan.get("plan_sha256") != plan["plan_sha256"]:
            raise RuntimeError("hosted accepted root is not plan-bound")
    plan_attempts = {row["run_id"]: row for row in plan["attempts"]}
    for path in paths:
        row = load_object(path)
        if row.get("receipt_sha256") != digest_without(row, "receipt_sha256"):
            raise RuntimeError("hosted accepted receipt digest mismatch")
        claim = load_object(root / "claims" / f"{row['run_id']}.json")
        planned = plan_attempts.get(row["run_id"])
        if (
            planned is None
            or claim.get("claim_sha256") != digest_without(claim, "claim_sha256")
            or claim.get("plan_sha256") != plan["plan_sha256"]
            or claim.get("run_id") != row["run_id"]
            or claim.get("claim_sha256") != row.get("claim_sha256")
            or claim.get("config_sha256") != row.get("config_sha256")
            or int(claim.get("rank") or 0) != int(planned["rank"])
            or int(claim.get("attempt") or 0) != int(planned["attempt"])
        ):
            raise RuntimeError("hosted accepted receipt is not claim-bound")
        if row["run_id"] in rows:
            raise RuntimeError("hosted accepted run identity duplicated")
        rows[row["run_id"]] = row
    return rows


def _allowed_sessions(plan: dict[str, Any], root: Path, rank: int) -> set[str]:
    allowed = {row["session_id"] for row in _credits_for_rank(plan, rank)}
    attempt_rank = {row["run_id"]: int(row["rank"]) for row in plan["attempts"]}
    for run_id, row in _accepted(plan, root).items():
        if attempt_rank.get(run_id) == rank:
            allowed.add(row["session_id"])
    return allowed


def _validate_inventory_for_task(
    plan: dict[str, Any], root: Path, task: dict[str, Any], key: str
) -> int:
    with _client(key) as client:
        all_rows = self_hosted._task_sessions(client, task["task"]["key"])
    rank = int(task["rank"])
    planned_run_ids = {
        row["run_id"] for row in plan["attempts"] if int(row["rank"]) == rank
    }
    accepted_by_run = {
        run_id: row
        for run_id, row in _accepted(plan, root).items()
        if run_id in planned_run_ids
    }
    for row in all_rows:
        run_id = (row.get("metadata") or {}).get("run_id")
        accepted = accepted_by_run.get(run_id)
        if run_id in planned_run_ids and (
            accepted is None or row.get("session_id") != accepted["session_id"]
        ):
            raise RuntimeError("hosted current-plan API run identity already exists")
    policy = plan["execution"].get(
        "inventory_policy", "exact_hosted_treatment_metadata_v1"
    )
    allowed = _allowed_sessions(plan, root, rank)
    if policy in {
        "exact_hosted_treatment_metadata_v1",
        "plan_identity_plus_authoritative_receipt_v1",
    }:
        model = self_hosted.persisted_session_model_identity(plan)
        harness = f"opencode-{plan['harness']['version']}"
        tool_digest = plan["execution"]["required_task_tool_catalog_sha256"]
        rows = [
            row
            for row in all_rows
            if row.get("model") == model
            and (row.get("metadata") or {}).get("self_hosted_harness") == harness
            and (row.get("metadata") or {}).get("tool_catalog_sha256") == tool_digest
        ]
        if policy == "plan_identity_plus_authoritative_receipt_v1":
            exact_ids = {row.get("session_id") for row in rows}
            rows.extend(
                row
                for row in all_rows
                if row.get("session_id") in allowed
                and row.get("session_id") not in exact_ids
            )
    elif policy == "conservative_no_same_model_session_for_task_key_v1":
        model = self_hosted.persisted_session_model_identity(plan)
        rows = [row for row in all_rows if row.get("model") == model]
    else:
        raise RuntimeError("hosted inventory policy drifted")
    observed = {row.get("session_id") for row in rows if isinstance(row.get("session_id"), str)}
    if observed != allowed:
        raise RuntimeError("hosted exact-treatment session inventory drifted")
    by_id = {row.get("session_id"): row for row in rows}
    credits = [*_credits_for_rank(plan, rank)]
    local_accepted = _accepted(plan, root)
    receipts = [*credits]
    attempt_rank = {row["run_id"]: int(row["rank"]) for row in plan["attempts"]}
    receipts.extend(
        row
        for run_id, row in local_accepted.items()
        if attempt_rank.get(run_id) == rank
    )
    for receipt in receipts:
        row = by_id.get(receipt["session_id"])
        verifier = (row or {}).get("verifier_execution") or {}
        local_receipt = receipt.get("run_id") in local_accepted
        if (
            row is None
            or row.get("status") != "completed"
            or verifier.get("id") != receipt["verifier_execution_id"]
            or (
                not local_receipt
                and (
                    row.get("eval_task_version_id") or row.get("task_version_id")
                )
                != task["task"]["version_id"]
            )
        ):
            raise RuntimeError("hosted credited session is not authoritative")
    return len(rows)


def _attempt_config(
    plan: dict[str, Any], task: dict[str, Any], item: dict[str, Any]
) -> dict[str, Any]:
    config = {
        "schema_version": "fleet-hosted-opencode-task-boundary-attempt-v1",
        "run_id": item["run_id"],
        "campaign_id": plan["campaign_id"],
        "source_job_id": plan["source_job_id"],
        "task": task["task"],
        "environment": task["environment"],
        "verifier": task["verifier"],
        "authority": plan["authority"],
        "model": plan["model"],
        "harness": plan["harness"],
        "execution": {
            "pass_k": 1,
            "planned_full_pass_k": 4,
            "max_concurrent": 1,
            "network": item["network"],
            "training_data_eligible": plan["execution"]["training_data_eligible"],
            "required_task_tools": plan["execution"]["required_task_tools"],
            "required_task_tool_catalog_sha256": plan["execution"][
                "required_task_tool_catalog_sha256"
            ],
        },
    }
    config["config_sha256"] = digest_without(config, "config_sha256")
    return config


def _classify_result(
    out_dir: Path, config: dict[str, Any], item: dict[str, Any], claim_sha256: str
) -> dict[str, Any]:
    result = load_object(out_dir / "result.json")
    ingest = load_object(out_dir / "session-ingest.json")
    cleanup = load_object(out_dir / "cleanup.json")
    verifier_id = result.get("verifier_execution_id")
    try:
        uuid.UUID(str(verifier_id))
        uuid.UUID(str(result.get("session_id")))
    except ValueError as exc:
        raise RuntimeError("hosted attempt lacks authoritative UUID evidence") from exc
    operationally_terminal = all(
        (
            result.get("run_id") == config["run_id"],
            result.get("agent_termination") == "completed",
            result.get("session_ingest_status") == "completed",
            ingest.get("status") == "completed",
            cleanup == {
                "instance_created": True,
                "instance_closed": True,
                "containers_removed": True,
            },
        )
    )
    if not operationally_terminal:
        raise RuntimeError("hosted attempt is infrastructure-incomplete")
    accepted = result.get("agent_exit_code") == 0
    row = {
        "schema_version": (
            "fleet-hosted-opencode-attempt-accepted-v1"
            if accepted
            else "fleet-hosted-opencode-attempt-noncreditable-v1"
        ),
        "accepted": accepted,
        "credited": accepted,
        "retry_allowed": False,
        "run_id": config["run_id"],
        "rank": int(item["rank"]),
        "source_rank": int(item["source_rank"]),
        "attempt": int(item["attempt"]),
        "task_key": config["task"]["key"],
        "task_version_id": config["task"]["version_id"],
        "session_id": result["session_id"],
        "verifier_execution_id": verifier_id,
        "agent_exit_code": result.get("agent_exit_code"),
        "config_sha256": config["config_sha256"],
        "claim_sha256": claim_sha256,
        "cleanup_completed": True,
        "session_ingest_completed": True,
        "scores_included": False,
        "prompts_or_traces_included": False,
    }
    row["receipt_sha256"] = digest_without(row, "receipt_sha256")
    name = "ACCEPTED.json" if accepted else "NONCREDITABLE.json"
    self_hosted.write_json_once(out_dir / name, row)
    return row


def _run_task(
    plan: dict[str, Any],
    task: dict[str, Any],
    items: list[dict[str, Any]],
    root: Path,
    proxy: Path,
    key: str,
) -> dict[str, Any]:
    rank = int(task["rank"])
    task_claim = {
        "schema_version": "fleet-hosted-opencode-task-claim-v1",
        "plan_sha256": plan["plan_sha256"],
        "rank": rank,
        "source_rank": int(task["source_rank"]),
        "task_key": task["task"]["key"],
        "task_version_id": task["task"]["version_id"],
        "run_ids": [item["run_id"] for item in items],
    }
    task_claim["claim_sha256"] = digest_without(task_claim, "claim_sha256")
    self_hosted.write_json_once(root / "task-claims" / f"rank-{rank:03d}.json", task_claim)
    _emit(
        "task_claimed",
        plan_sha256=plan["plan_sha256"],
        rank=rank,
        source_rank=int(task["source_rank"]),
        attempts=[int(item["attempt"]) for item in items],
        claim_sha256=task_claim["claim_sha256"],
    )
    accepted_count = 0
    for item in items:
        _validate_inventory_for_task(plan, root, task, key)
        config = _attempt_config(plan, task, item)
        claim = {
            "schema_version": "fleet-hosted-opencode-attempt-claim-v1",
            "plan_sha256": plan["plan_sha256"],
            "task_claim_sha256": task_claim["claim_sha256"],
            "run_id": item["run_id"],
            "rank": rank,
            "source_rank": int(item["source_rank"]),
            "attempt": int(item["attempt"]),
            "network": item["network"],
            "config_sha256": config["config_sha256"],
        }
        claim["claim_sha256"] = digest_without(claim, "claim_sha256")
        self_hosted.write_json_once(root / "claims" / f"{item['run_id']}.json", claim)
        out_dir = root / "attempts" / item["run_id"]
        self_hosted.run(config, out_dir, proxy)
        outcome = _classify_result(out_dir, config, item, claim["claim_sha256"])
        if not outcome["accepted"]:
            task_receipt = {
                "schema_version": "fleet-hosted-opencode-task-fenced-v1",
                "plan_sha256": plan["plan_sha256"],
                "rank": rank,
                "source_rank": int(task["source_rank"]),
                "accepted_new_outcomes": accepted_count,
                "noncreditable_receipt_sha256": outcome["receipt_sha256"],
                "remaining_cells_not_started": [
                    next_item["attempt"]
                    for next_item in items
                    if int(next_item["attempt"]) > int(item["attempt"])
                ],
                "retry_allowed": False,
                "scores_included": False,
            }
            task_receipt["receipt_sha256"] = digest_without(task_receipt, "receipt_sha256")
            self_hosted.write_json_once(
                root / "task-results" / f"rank-{rank:03d}.json", task_receipt
            )
            _emit(
                "task_fenced_noncreditable",
                rank=rank,
                source_rank=int(task["source_rank"]),
                attempt=int(item["attempt"]),
                receipt_sha256=outcome["receipt_sha256"],
            )
            return {"complete": False, "accepted": accepted_count}
        accepted_count += 1
        _emit(
            "attempt_accepted",
            rank=rank,
            source_rank=int(item["source_rank"]),
            attempt=int(item["attempt"]),
            receipt_sha256=outcome["receipt_sha256"],
        )
    task_receipt = {
        "schema_version": "fleet-hosted-opencode-task-pass4-accepted-v1",
        "plan_sha256": plan["plan_sha256"],
        "rank": rank,
        "source_rank": int(task["source_rank"]),
        "credited_outcomes": len(_credits_for_rank(plan, rank)),
        "accepted_new_outcomes": accepted_count,
        "total_outcomes": len(_credits_for_rank(plan, rank)) + accepted_count,
        "scores_included": False,
    }
    if task_receipt["total_outcomes"] != 4:
        raise RuntimeError("hosted task did not reach exact pass@4")
    task_receipt["receipt_sha256"] = digest_without(task_receipt, "receipt_sha256")
    self_hosted.write_json_once(root / "task-results" / f"rank-{rank:03d}.json", task_receipt)
    _emit(
        "task_pass4_complete",
        rank=rank,
        source_rank=int(task["source_rank"]),
        receipt_sha256=task_receipt["receipt_sha256"],
    )
    return {"complete": True, "accepted": accepted_count}


def _worker_cap(accepted: int) -> int:
    cap = 1
    for stage in SCHEDULE:
        if accepted >= int(stage["accepted_outcomes_at_least"]):
            cap = int(stage["workers"])
    return cap


def _validate_plan_identity_absence(plan: dict[str, Any], root: Path) -> int:
    """Prove no current-plan run identity or claim exists anywhere on shared storage."""
    if root.exists():
        raise RuntimeError("hosted shard output root already exists")
    parent = root.parent
    if parent.is_symlink() or not parent.is_dir():
        raise RuntimeError("hosted shared job root is not an authoritative directory")
    planned = {row["run_id"] for row in plan["attempts"]}
    roots_reconciled = 0
    for job_root in parent.iterdir():
        if job_root.is_symlink():
            raise RuntimeError("hosted shared job root contains a symlink")
        if not job_root.is_dir():
            continue
        roots_reconciled += 1
        for run_id in planned:
            if (job_root / "claims" / f"{run_id}.json").exists() or (
                job_root / "attempts" / run_id
            ).exists():
                raise RuntimeError("hosted current-plan run identity already exists")
        for claim_path in (job_root / "task-claims").glob("*.json"):
            claim = load_object(claim_path)
            run_ids = claim.get("run_ids") or []
            if not isinstance(run_ids, list) or any(run_id in planned for run_id in run_ids):
                raise RuntimeError("hosted current-plan task claim already exists")
    return roots_reconciled


def preflight_plan(plan: dict[str, Any], root: Path, key: str) -> dict[str, Any]:
    validate_plan(plan)
    roots_reconciled = _validate_plan_identity_absence(plan, root)
    with _client(key) as client:
        account = self_hosted._request(client, "GET", "/v1/account")
    if (
        account.get("team_name") != "fleet"
        or account.get("team_id") != self_hosted.FLEET_TEAM_ID
    ):
        raise RuntimeError("FLEET_API_KEY is not scoped to the Fleet team")
    # A temporary empty root lets the same exact-inventory function prove that
    # no plan-owned new receipt is being silently credited before launch.
    scratch = Path("/tmp/hosted-preflight-empty")
    if scratch.exists():
        raise RuntimeError("hosted preflight scratch unexpectedly exists")
    (scratch / "attempts").mkdir(parents=True)
    inventory_counts: dict[int, int] = {}
    lock = threading.Lock()

    def inspect(task: dict[str, Any]) -> None:
        count = _validate_inventory_for_task(plan, scratch, task, key)
        with lock:
            inventory_counts[int(task["rank"])] = count

    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(inspect, plan["tasks"]))
    finally:
        (scratch / "attempts").rmdir()
        scratch.rmdir()
    receipt = {
        "schema_version": "fleet-hosted-opencode-preflight-v1",
        "plan_sha256": plan["plan_sha256"],
        "fleet_team_id": self_hosted.FLEET_TEAM_ID,
        "tasks_reconciled": len(inventory_counts),
        "exact_treatment_sessions_reconciled": sum(inventory_counts.values()),
        "active_source_attempts": 0,
        "output_root_absent": True,
        "sfs_job_roots_reconciled": roots_reconciled,
        "current_plan_run_and_claim_identities_absent": True,
        "public_list_treatment_metadata_authoritative": False,
        "empty_exact_treatment_set_used_as_sole_proof": False,
        "scores_read": False,
        "prompts_or_traces_read": False,
    }
    receipt["receipt_sha256"] = digest_without(receipt, "receipt_sha256")
    return receipt


def run_plan(plan: dict[str, Any], root: Path, proxy: Path) -> dict[str, Any]:
    validate_plan(plan)
    key = os.environ.get("FLEET_API_KEY")
    if not key:
        raise RuntimeError("FLEET_API_KEY is required")
    roots_reconciled = _validate_plan_identity_absence(plan, root)
    root.mkdir(parents=True, exist_ok=False)
    root.chmod(0o700)
    for name in ("attempts", "claims", "task-claims", "task-results", "ramps"):
        (root / name).mkdir(mode=0o700)
    self_hosted.write_json_once(root / "PLAN.json", plan)
    with _client(key) as client:
        account = self_hosted._request(client, "GET", "/v1/account")
    if account.get("team_name") != "fleet" or account.get("team_id") != self_hosted.FLEET_TEAM_ID:
        raise RuntimeError("FLEET_API_KEY is not scoped to the Fleet team")

    inventory_counts: dict[int, int] = {}
    lock = threading.Lock()
    def inspect(task: dict[str, Any]) -> None:
        count = _validate_inventory_for_task(plan, root, task, key)
        with lock:
            inventory_counts[int(task["rank"])] = count
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(inspect, plan["tasks"]))
    preflight = {
        "schema_version": "fleet-hosted-opencode-preflight-v1",
        "plan_sha256": plan["plan_sha256"],
        "fleet_team_id": self_hosted.FLEET_TEAM_ID,
        "tasks_reconciled": len(inventory_counts),
        "exact_treatment_sessions_reconciled": sum(inventory_counts.values()),
        "active_source_attempts": 0,
        "sfs_job_roots_reconciled": roots_reconciled,
        "current_plan_run_and_claim_identities_absent": True,
        "public_list_treatment_metadata_authoritative": False,
        "empty_exact_treatment_set_used_as_sole_proof": False,
        "scores_read": False,
        "prompts_or_traces_read": False,
    }
    preflight["receipt_sha256"] = digest_without(preflight, "receipt_sha256")
    self_hosted.write_json_once(root / "PREFLIGHT.json", preflight)

    tasks = {int(row["rank"]): row for row in plan["tasks"]}
    groups: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for rank in sorted(tasks):
        items = sorted(
            (row for row in plan["attempts"] if int(row["rank"]) == rank),
            key=lambda row: int(row["attempt"]),
        )
        groups.append((tasks[rank], items))

    pending = iter(groups)
    active: dict[Future[dict[str, Any]], int] = {}
    accepted_count = 0
    complete_tasks = 0
    fenced_tasks = 0
    last_cap = 1
    infrastructure_failure: Exception | None = None
    with ThreadPoolExecutor(max_workers=max(stage["workers"] for stage in SCHEDULE)) as pool:
        def submit_until_cap() -> None:
            nonlocal last_cap
            cap = _worker_cap(accepted_count)
            if cap != last_cap:
                ramp = {
                    "schema_version": "fleet-hosted-opencode-score-blind-ramp-v1",
                    "plan_sha256": plan["plan_sha256"],
                    "from_workers": last_cap,
                    "to_workers": cap,
                    "accepted_operational_outcomes": accepted_count,
                    "scores_read": False,
                    "decision_inputs": [
                        "accepted",
                        "session_ingest_completed",
                        "verifier_uuid",
                        "cleanup_completed",
                    ],
                }
                ramp["receipt_sha256"] = digest_without(ramp, "receipt_sha256")
                self_hosted.write_json_once(root / "ramps" / f"workers-{cap}.json", ramp)
                last_cap = cap
            while len(active) < cap and infrastructure_failure is None:
                try:
                    task, items = next(pending)
                except StopIteration:
                    return
                future = pool.submit(_run_task, plan, task, items, root, proxy, key)
                active[future] = int(task["rank"])

        submit_until_cap()
        while active:
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                active.pop(future)
                try:
                    result = future.result()
                    accepted_count += int(result["accepted"])
                    if result["complete"]:
                        complete_tasks += 1
                    else:
                        fenced_tasks += 1
                except Exception as exc:  # noqa: BLE001
                    infrastructure_failure = exc
            submit_until_cap()
    if infrastructure_failure is not None:
        raise infrastructure_failure

    terminal = {
        "schema_version": "fleet-hosted-opencode-collection-terminal-v1",
        "plan_sha256": plan["plan_sha256"],
        "planned_tasks": int(plan["task_count"]),
        "pass4_complete_tasks": complete_tasks,
        "fenced_noncreditable_tasks": fenced_tasks,
        "accepted_new_outcomes": accepted_count,
        "credited_prior_outcomes": len(plan["credited_sessions"]),
        "scores_included": False,
        "prompts_or_traces_included": False,
    }
    if complete_tasks + fenced_tasks != int(plan["task_count"]):
        raise RuntimeError("hosted task terminal accounting drifted")
    terminal["receipt_sha256"] = digest_without(terminal, "receipt_sha256")
    self_hosted.write_json_once(root / "TERMINAL.json", terminal)
    return terminal


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--source-plan", type=Path, required=True)
    build.add_argument("--source-state", type=Path, required=True)
    build.add_argument("--model", choices=sorted(CAMPAIGNS), required=True)
    build.add_argument("--output", type=Path, required=True)
    remainder = sub.add_parser("build-remainder")
    remainder.add_argument("--predecessor-plan", type=Path, required=True)
    remainder.add_argument("--source-state", type=Path, required=True)
    remainder.add_argument("--model", choices=sorted(REMAINDER_CAMPAIGNS), required=True)
    remainder.add_argument("--output", type=Path, required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--plan", type=Path, required=True)
    preflight = sub.add_parser("preflight")
    preflight.add_argument("--plan", type=Path, required=True)
    preflight.add_argument("--out-dir", type=Path, required=True)
    run = sub.add_parser("run")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--out-dir", type=Path, required=True)
    run.add_argument("--proxy-script", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        value = build_plan(
            load_object(args.source_plan), load_object(args.source_state), args.model
        )
        self_hosted.write_json_once(args.output, value)
        return 0
    if args.command == "validate":
        validate_plan(load_object(args.plan))
        return 0
    if args.command == "build-remainder":
        value = build_remainder_plan(
            load_object(args.predecessor_plan),
            load_object(args.source_state),
            args.model,
        )
        self_hosted.write_json_once(args.output, value)
        return 0
    if args.command == "preflight":
        key = os.environ.get("FLEET_API_KEY")
        if not key:
            raise RuntimeError("FLEET_API_KEY is required")
        result = preflight_plan(load_object(args.plan), args.out_dir, key)
        print(json.dumps(result, sort_keys=True))
        return 0
    result = run_plan(load_object(args.plan), args.out_dir, args.proxy_script)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
