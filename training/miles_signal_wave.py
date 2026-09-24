"""Validate the immutable four-lane Miles reward-signal authority packet."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "configs/qualification/qwen38-miles-signal-wave-v1.json"
SHA = re.compile(r"sha256:[a-f0-9]{64}")


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _file(authority: dict[str, str]) -> dict[str, Any]:
    path = ROOT / authority["path"]
    if "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest() != authority["file_sha256"]:
        raise ValueError(f"authority file changed: {authority['path']}")
    return json.loads(path.read_text())


def _normalized_sha(value: str) -> str:
    return value if value.startswith("sha256:") else "sha256:" + value


def task_binding(plan: dict[str, Any], candidate: dict[str, Any]) -> dict[str, str]:
    """Return the exact phase-2-compatible authority tuple for one lane."""
    return {
        "task_key": candidate["task"]["key"],
        "task_version_id": candidate["task"]["version_id"],
        "verifier_version_id": candidate["verifier"]["version_id"],
        "task_set_sha256": plan["authorities"]["task_set"]["self_sha256"],
        "tool_catalog_sha256": plan["authorities"]["tool_catalog"]["self_sha256"],
    }


def live_binding(plan: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Canonical richer binding observed through the live exact-version GET."""
    return {
        "task_set_sha256": plan["authorities"]["task_set"]["self_sha256"],
        "production_split_sha256": plan["authorities"]["production_split"]["self_sha256"],
        "tool_catalog_sha256": plan["authorities"]["tool_catalog"]["self_sha256"],
        "split": candidate["split"],
        "lineage": {"task_family": candidate["lineage"]["task_family"]},
        "task": {
            k: v for k, v in candidate["task"].items() if k not in {"id", "live_response_sha256"}
        },
        "environment": candidate["environment"],
        "verifier": candidate["verifier"],
        "cyber_contract_sha256": candidate["cyber_contract_sha256"],
        "current_binding_sha256": candidate["legacy_current_binding_sha256"],
    }


def validate(plan: dict[str, Any]) -> dict[str, Any]:
    value = json.loads(json.dumps(plan))
    body = {k: v for k, v in value.items() if k != "sha256"}
    if value.get("schema") != "cyber_qwen38_miles_signal_wave_v1" or value.get(
        "sha256"
    ) != "sha256:" + digest(body):
        raise ValueError("signal-wave seal is invalid")
    if value.get("launchable") is not False or value.get("state") != "parent_review_required":
        raise ValueError("signal wave must remain unlaunchable pending review")

    authorities = value["authorities"]
    task_set = _file(authorities["task_set"])
    split = _file(authorities["production_split"])
    eligible = _file(authorities["eligible_inventory"])
    ledger = _file(authorities["selection_prior"])
    catalog = _file(authorities["tool_catalog"])
    for name, document in (
        ("task_set", task_set),
        ("production_split", split),
        ("eligible_inventory", eligible),
    ):
        if _normalized_sha(document.get("sha256", "")) != authorities[name]["self_sha256"]:
            raise ValueError(f"{name} self digest changed")
    if authorities["selection_prior"]["receipt_sha256"] != ledger.get("receipt_sha256"):
        raise ValueError("selection-prior receipt changed")
    if [tool.get("name") for tool in catalog] != authorities["tool_catalog"]["ordered_tools"]:
        raise ValueError("tool catalog order changed")
    if (
        "sha256:"
        + hashlib.sha256(
            json.dumps(catalog, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        != authorities["tool_catalog"]["self_sha256"]
    ):
        raise ValueError("tool catalog digest changed")

    task_rows = {(row["task_key"], row["task_version_id"]): row for row in task_set["tasks"]}
    split_rows = {(row["task_key"], row["task_version_id"]): row for row in split["tasks"]}
    eligible_rows = {
        (row["task_key"], row["task_version_id"]): row for row in eligible["task_versions"]
    }
    candidates = value["candidates"]
    names = {row["identity"]["name"] for row in candidates}
    families = {row["lineage"]["task_family"] for row in candidates}
    if len(candidates) != len(names) or len(candidates) != len(families) or len(candidates) != 4:
        raise ValueError("signal wave needs four unique names and task families")
    if value["phase1"] != {
        "samples_per_candidate": 8,
        "max_concurrent_episodes": 2,
        "optimizer_steps": 0,
        "checkpoint": False,
        "max_turns": 32,
        "max_tokens_per_turn": 8192,
        "episode_timeout_s": 2400,
        "tool_timeout_s": 330,
        "tool_output_max_chars": 50000,
        "failure_handling": {
            "outer_retry_or_replacement": False,
            "maintained_fti_native_request_behavior_unchanged": True,
            "every_slot_predeclared_once": True,
            "every_slot_terminally_accounted": True,
            "infra_invalid_cells_excluded": True,
            "accepted_or_errored_cell_never_replayed": True,
        },
        "acceptance": {
            "minimum_completed_gradeable_episodes": 2,
            "minimum_distinct_finite_rewards": 2,
            "every_reward_requires_exact_verifier_execution_id": True,
            "all_planned_slots_terminally_accounted": True,
            "all_instances_released": True,
        },
        "successor": value["phase1"]["successor"],
    }:
        raise ValueError("phase-1 zero-update contract drift")
    motivation_evidence = value["historical_low_concurrency_motivation"]
    if (
        motivation_evidence.get("receipt_sha256")
        != "sha256:ee591a4f3b002486f0fe16ec06b668b9ca36d0698c5d87f02df69c659094e3e9"
        or motivation_evidence.get("scored_error_terminal_receipt_sha256")
        != "sha256:b4d670cd7e014fad7e21b748f6ab01201f99d374197d908c7927625130cece95"
        or motivation_evidence.get("scored_error_predicate_sha256")
        != "sha256:0e342297a6f4d7e1029ce338e998ccee67463f0a4eb9f8fda461d0f6e354ee9d"
    ):
        raise ValueError("429 classifier or scored-error evidence changed")
    cluster = value["cluster"]
    if not (
        cluster["priority_class"] == "c1"
        and cluster["queue_priority"] == "q1"
        and cluster["nodes_per_candidate"] == 1
        and cluster["gpus_per_node"] == 8
        and cluster["parallel_candidates"] == 4
        and cluster["failureAlerts"] is False
        and cluster["root_annotations"] == {"fleet.ai/failure-alerts": "off"}
        and cluster["backoff_limit"] == 0
    ):
        raise ValueError("cluster safety contract drift")

    for rank, row in enumerate(candidates, 1):
        if row["rank"] != rank or row["split"] != "train":
            raise ValueError("candidate ordering or split changed")
        key = (row["task"]["key"], row["task"]["version_id"])
        selected, assigned, inventory = task_rows[key], split_rows[key], eligible_rows[key]
        if assigned["split"] != "train" or selected["lineage"] != row["lineage"]:
            raise ValueError("candidate is outside the exact production train split")
        if any(
            selected[field] != row["environment"][target]
            for field, target in (
                ("env_key", "id"),
                ("env_version", "version"),
                ("environment_version_id", "version_id"),
                ("data_key", "data_id"),
                ("data_version", "data_version"),
            )
        ):
            raise ValueError("task-set runtime binding changed")
        if (
            inventory["environment"] != row["environment"]
            or inventory["verifier"] != row["verifier"]
        ):
            raise ValueError("eligible inventory binding changed")
        if inventory["current_binding_sha256"] != row["legacy_current_binding_sha256"]:
            raise ValueError("legacy inventory binding digest changed")
        authority = task_binding(value, row)
        if row["authority_receipt_sha256"] != "sha256:" + digest(authority):
            raise ValueError("mechanics authority receipt is invalid")
        if row["live_binding_receipt_sha256"] != "sha256:" + digest(live_binding(value, row)):
            raise ValueError("rich live-binding receipt is invalid")
        prior = row["selection_prior"]
        if not (
            prior["completed_accepted_episodes"] == 4
            and prior["passes"] in {2, 3}
            and prior["mixed_binary_outcomes"] is True
        ):
            raise ValueError("selection prior is not mixed")
        for field in (
            "split_group_sha256",
            "safe_task_binding_sha256",
            "authority_receipt_sha256",
            "live_binding_receipt_sha256",
            "cyber_contract_sha256",
        ):
            if not SHA.fullmatch(row[field]):
                raise ValueError(f"invalid {field}")
        if row["identity"]["run_dir"] != "/mnt/sfs/jobs/" + row["identity"]["name"]:
            raise ValueError("run output is not create-once identity bound")
    return value


def load() -> dict[str, Any]:
    return validate(json.loads(PLAN_PATH.read_text()))
