"""Fresh-current release authority for the held Qwen DP6-f server."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from evals.fleet import qwen38_dp6_f_live_v1 as base
from evals.fleet import qwen38_dp6_f_scorefree_v1 as held
from evals.fleet import self_hosted

SERVER_RELEASE_SCHEMA = "fleet-qwen38-dp6-f-server-release-v2"
CURRENT_INVENTORY_SCHEMA = "fleet-qwen38-dp6-f-current-live-inventory-v2"
LIVE_GATE_SCHEMA = base.LIVE_GATE_SCHEMA
SUBMISSION_SCHEMA = base.SUBMISSION_SCHEMA
MAX_INVENTORY_AGE_SECONDS = 300

shared = base.shared
_observer_pod = base._observer_pod
submit_create_once = base.submit_create_once
validate_submission = base.validate_submission


def _digest(value: Mapping[str, Any]) -> str:
    return self_hosted.digest_without(dict(value), "receipt_sha256")


def current_inventory(client: httpx.Client, source_commit: str) -> dict[str, Any]:
    active = base._active_project_runs(client)
    shape = base._kubernetes_gate(active)
    sfs = base._sfs_absence(base._observer_pod())
    value = {
        "schema_version": CURRENT_INVENTORY_SCHEMA,
        "status": "CLEAR_CURRENT_ZERO_SERVING",
        "observed_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "source_commit": source_commit,
        "active_project_serving_runs": 0,
        "project_resource_shape": shape,
        "target_identity_matches": {"jobs_api": 0, "kubernetes": 0, "sfs": 0},
        "sfs_observation": sfs,
        "api_mutations": 0,
        "scored_calls": 0,
        "prompts_traces_flags_or_scores_included": False,
    }
    value["receipt_sha256"] = _digest(value)
    return value


def validate_current_inventory(value: Mapping[str, Any], source_commit: str) -> None:
    observed = datetime.fromisoformat(str(value.get("observed_at_utc", "")).replace("Z", "+00:00"))
    age = (datetime.now(UTC) - observed).total_seconds()
    if value.get("receipt_sha256") != _digest(value) or (
        value.get("schema_version") != CURRENT_INVENTORY_SCHEMA
        or value.get("status") != "CLEAR_CURRENT_ZERO_SERVING"
        or value.get("source_commit") != source_commit
        or value.get("active_project_serving_runs") != 0
        or value.get("target_identity_matches") != {"jobs_api": 0, "kubernetes": 0, "sfs": 0}
        or value.get("sfs_observation", {}).get("run_dir_exists") is not False
        or not base._shape_safe(value.get("project_resource_shape"))
        or value.get("api_mutations") != 0
        or value.get("scored_calls") != 0
        or value.get("prompts_traces_flags_or_scores_included") is not False
        or not -30 <= age <= MAX_INVENTORY_AGE_SECONDS
    ):
        raise ValueError("DP6-f current live inventory is not executable")


def validate_server_release(
    value: Mapping[str, Any], root: Path, source_commit: str
) -> None:
    config, plan, preview, historical_inventory, held_release = held.load_all(root)
    current = value.get("current_inventory")
    if not isinstance(current, dict):
        raise ValueError("DP6-f release omitted current live inventory")
    validate_current_inventory(current, source_commit)
    expected = {
        "schema_version": SERVER_RELEASE_SCHEMA,
        "status": "RELEASED_FOR_ONE_SCORE_FREE_DP6_F_SERVER",
        "launch_authorized": True,
        "scoring_authorized": False,
        "source_commit": source_commit,
        "title": held.TITLE,
        "run_dir": held.RUN_DIR,
        "serving_block": held.SERVING_BLOCK,
        "config_sha256": config["config_sha256"],
        "plan_receipt_sha256": plan["receipt_sha256"],
        "preview_receipt_sha256": preview["receipt_sha256"],
        "historical_blocked_inventory_receipt_sha256": historical_inventory["receipt_sha256"],
        "held_release_receipt_sha256": held_release["receipt_sha256"],
        "current_inventory_receipt_sha256": current["receipt_sha256"],
        "current_inventory": current,
        "release_first_reconciled": True,
        "server_create_limit": 1,
        "qualifier_create_limit": 0,
        "statistical_cells_selected": 0,
        "scored_calls": 0,
        "prompts_traces_flags_or_scores_included": False,
    }
    if value.get("receipt_sha256") != _digest(value) or (
        {key: item for key, item in value.items() if key != "receipt_sha256"} != expected
    ):
        raise ValueError("DP6-f current server release is not executable")


def live_gate(
    client: httpx.Client,
    release: Mapping[str, Any],
    root: Path,
    source_commit: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    validate_server_release(release, root, source_commit)
    payload = held.jobs_payload(root)
    active = base._active_project_runs(client)
    shape = base._kubernetes_gate(active)
    preview = client.post("/v1/runs/preview", json=payload)
    preview.raise_for_status()
    rendered = held.preview_identity(preview.json()["manifest_yaml"], root)
    sfs = base._sfs_absence(base._observer_pod())
    value = {
        "schema_version": LIVE_GATE_SCHEMA,
        "status": "PASSED_IMMEDIATELY_BEFORE_CREATE",
        "observed_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "source_commit": source_commit,
        "server_release_receipt_sha256": release["receipt_sha256"],
        "config_sha256": held.config(root)["config_sha256"],
        "request_sha256": held.config(root)["request_sha256"],
        "active_project_serving_runs": 0,
        "project_resource_shape": shape,
        "target_identity_matches": {"jobs_api": 0, "kubernetes": 0, "sfs": 0},
        "sfs_observation": sfs,
        "rendered": rendered,
        "api_mutations": 0,
        "scored_calls": 0,
        "prompts_traces_flags_or_scores_included": False,
    }
    value["receipt_sha256"] = _digest(value)
    return payload, value
