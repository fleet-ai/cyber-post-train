"""Build the fail-closed review transition for the four Miles signal lanes."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

from cyber_post_train.jobs import digest
from evals.fleet import opencode_self_hosted as fleet

from . import miles_signal_wave
from .fleet import FleetClient

ROOT = Path(__file__).resolve().parents[1]
SHA256 = re.compile(r"sha256:[0-9a-f]{64}")
COMMIT = re.compile(r"[0-9a-f]{40}")
LIVE_SCHEMA = "cyber_qwen38_miles_signal_live_task_receipt_v1"
CANDIDATE_SCHEMA = "cyber_qwen38_miles_signal_transition_candidate_v1"
REVIEW_SCHEMA = "cyber_qwen38_miles_signal_parent_review_v1"
APPROVED_SCHEMA = "cyber_qwen38_miles_signal_reviewed_transition_v1"
FRESH_TASK_SECONDS = 900
FRESH_PREVIEW_SECONDS = 300

GATES = (
    "frozen_adapter_and_exact_image_preflight",
    "frozen_operator_commit",
    "authority_seal_and_fresh_task_gets",
    "four_immutable_plan_request_digests",
    "live_server_previews",
    "duplicate_and_create_once_absence_checks",
    "observer_armed_and_uid_release_safety",
    "private_material_cleanup",
    "parent_review",
)


def _now(value: dt.datetime | None = None) -> dt.datetime:
    value = value or dt.datetime.now(dt.UTC)
    if value.tzinfo is None:
        raise ValueError("time must be timezone-aware")
    return value.astimezone(dt.UTC)


def _time(value: str) -> dt.datetime:
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return parsed.astimezone(dt.UTC)


def _stamp(value: dt.datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _sealed(body: dict[str, Any]) -> dict[str, Any]:
    return {**body, "sha256": "sha256:" + digest(body)}


def _verify_seal(value: dict[str, Any], schema: str) -> None:
    if value.get("schema") != schema or not SHA256.fullmatch(str(value.get("sha256", ""))):
        raise ValueError(f"invalid {schema} identity")
    if value["sha256"] != "sha256:" + digest({k: v for k, v in value.items() if k != "sha256"}):
        raise ValueError(f"invalid {schema} seal")


def _fresh(observed_at: str, now: dt.datetime, seconds: int) -> bool:
    try:
        age = (now - _time(observed_at)).total_seconds()
    except (TypeError, ValueError):
        return False
    return 0 <= age <= seconds


def _expected_live_row(wave: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    row = {
        "name": candidate["identity"]["name"],
        "task": {
            "key": candidate["task"]["key"],
            "version_id": candidate["task"]["version_id"],
            "prompt_sha256": candidate["task"]["prompt_sha256"],
            "env_variables_sha256": candidate["task"]["env_variables_sha256"],
            "output_json_schema_sha256": candidate["task"]["output_json_schema_sha256"],
            "cyber_contract_sha256": candidate["cyber_contract_sha256"],
            "lifecycle_status": "production",
        },
        "environment": candidate["environment"],
        "verifier": candidate["verifier"],
        "split": "train",
        "authority_receipt_sha256": candidate["authority_receipt_sha256"],
        "live_binding_receipt_sha256": candidate["live_binding_receipt_sha256"],
    }
    return {**row, "binding_sha256": "sha256:" + digest(row)}


def collect_live_task_receipt(
    client: FleetClient, *, observed_at: dt.datetime | None = None
) -> dict[str, Any]:
    """GET exact task versions and retain identities/hashes, never private bodies."""
    wave = miles_signal_wave.load()
    account = client._get("/v1/account")
    if account.get("team_name") != "fleet" or account.get("team_id") != fleet.FLEET_TEAM_ID:
        raise ValueError("live task receipt requires the Fleet team")
    rows = []
    for candidate in wave["candidates"]:
        response = client._get(
            f"/v1/tasks/{candidate['task']['key']}",
            {"version_id": candidate["task"]["version_id"]},
        )
        selected = {
            "task_key": candidate["task"]["key"],
            "task_version_id": candidate["task"]["version_id"],
            "env_key": candidate["environment"]["id"],
            "env_version": candidate["environment"]["version"],
            "environment_version_id": candidate["environment"]["version_id"],
            "data_key": candidate["environment"]["data_id"],
            "data_version": candidate["environment"]["data_version"],
        }
        task, environment, verifier = fleet.bind_task(response, selected)
        observed = {
            "name": candidate["identity"]["name"],
            "task": {
                "key": task["key"],
                "version_id": task["version_id"],
                "prompt_sha256": task["prompt_sha256"],
                "env_variables_sha256": task["env_variables_sha256"],
                "output_json_schema_sha256": task["output_json_schema_sha256"],
                "cyber_contract_sha256": "sha256:"
                + digest((response.get("metadata") or {}).get("cyber_contract")),
                "lifecycle_status": response.get("task_lifecycle_status"),
            },
            "environment": {k: environment[k] for k in candidate["environment"]},
            "verifier": {k: verifier[k] for k in candidate["verifier"]},
            "split": candidate["split"],
            "authority_receipt_sha256": candidate["authority_receipt_sha256"],
            "live_binding_receipt_sha256": candidate["live_binding_receipt_sha256"],
        }
        observed["binding_sha256"] = "sha256:" + digest(observed)
        expected = _expected_live_row(wave, candidate)
        if observed != expected:
            raise ValueError(f"live task binding drifted: {candidate['identity']['name']}")
        rows.append(observed)
    body = {
        "schema": LIVE_SCHEMA,
        "observed_at": _stamp(_now(observed_at)),
        "authority_config_sha256": wave["sha256"],
        "authority_source_commit": wave["source_commit"],
        "account": {"team_id": fleet.FLEET_TEAM_ID, "team_name": "fleet"},
        "task_set_sha256": wave["authorities"]["task_set"]["self_sha256"],
        "production_split_sha256": wave["authorities"]["production_split"]["self_sha256"],
        "tool_catalog_sha256": wave["authorities"]["tool_catalog"]["self_sha256"],
        "candidates": rows,
        "privacy": {
            "prompt_text_included": False,
            "verifier_code_included": False,
            "env_values_included": False,
            "credentials_included": False,
        },
    }
    return _sealed(body)


def validate_live_task_receipt(
    receipt: dict[str, Any], *, now: dt.datetime | None = None, require_fresh: bool = False
) -> dict[str, Any]:
    wave = miles_signal_wave.load()
    _verify_seal(receipt, LIVE_SCHEMA)
    expected = {
        "authority_config_sha256": wave["sha256"],
        "authority_source_commit": wave["source_commit"],
        "account": {"team_id": fleet.FLEET_TEAM_ID, "team_name": "fleet"},
        "task_set_sha256": wave["authorities"]["task_set"]["self_sha256"],
        "production_split_sha256": wave["authorities"]["production_split"]["self_sha256"],
        "tool_catalog_sha256": wave["authorities"]["tool_catalog"]["self_sha256"],
        "candidates": [_expected_live_row(wave, row) for row in wave["candidates"]],
        "privacy": {
            "prompt_text_included": False,
            "verifier_code_included": False,
            "env_values_included": False,
            "credentials_included": False,
        },
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError(f"live task receipt drifted: {key}")
    _time(receipt["observed_at"])
    if require_fresh and not _fresh(receipt["observed_at"], _now(now), FRESH_TASK_SECONDS):
        raise ValueError("live task receipt is stale")
    return receipt


def _sha(value: Any) -> bool:
    return isinstance(value, str) and SHA256.fullmatch(value) is not None


def _commit(value: Any) -> bool:
    return isinstance(value, str) and COMMIT.fullmatch(value) is not None


def _candidate_gate_status(
    wave: dict[str, Any], receipt: dict[str, Any], evidence: dict[str, Any], now: dt.datetime
) -> dict[str, bool]:
    adapter = evidence.get("adapter") or {}
    operator = evidence.get("operator") or {}
    lanes = evidence.get("lanes") or []
    cleanup = evidence.get("cleanup") or {}
    names = [row["identity"]["name"] for row in wave["candidates"]]
    lane_names = [row.get("name") for row in lanes if isinstance(row, dict)]
    exact_lanes = len(lanes) == 4 and lane_names == names
    adapter_ok = (
        _commit(adapter.get("commit"))
        and adapter.get("clean") is True
        and _sha(adapter.get("source_closure_sha256"))
        and isinstance(adapter.get("runtime_image"), str)
        and "@sha256:" in adapter["runtime_image"]
        and _sha(adapter.get("tests_receipt_sha256"))
        and _sha(adapter.get("exact_image_preflight_sha256"))
        and adapter.get("sample_indexes") == list(range(8))
        and adapter.get("max_concurrent_episodes") == 2
        and adapter.get("optimizer_steps") == 0
        and adapter.get("checkpoint") is False
        and adapter.get("live_session_open_tool_schema_gate") is True
        and adapter.get("tool_catalog_sha256") == wave["authorities"]["tool_catalog"]["self_sha256"]
        and adapter.get("current_binding_source") == "live_binding_receipt_sha256"
        and adapter.get("no_outer_retry_or_replacement") is True
        and adapter.get("all_slots_terminally_accounted") is True
        and adapter.get("unique_verifier_execution_ids") is True
        and adapter.get("exact_cleanup_required") is True
    )
    operator_ok = (
        _commit(operator.get("commit"))
        and operator.get("clean") is True
        and _sha(operator.get("tests_receipt_sha256"))
    )
    plans_ok = (
        exact_lanes
        and len({row.get("plan_sha256") for row in lanes}) == 4
        and len({row.get("request_sha256") for row in lanes}) == 4
        and all(
            _sha(row.get("plan_sha256"))
            and _sha(row.get("request_sha256"))
            and row.get("optimizer_steps") == 0
            and row.get("checkpoint") is False
            and row.get("authority_config_sha256") == wave["sha256"]
            and row.get("live_binding_receipt_sha256")
            == wave["candidates"][index]["live_binding_receipt_sha256"]
            for index, row in enumerate(lanes)
        )
    )
    previews_ok = (
        exact_lanes
        and len({(row.get("preview") or {}).get("receipt_sha256") for row in lanes}) == 4
        and all(
            (preview := row.get("preview") or {}).get("request_sha256") == row.get("request_sha256")
            and _sha(preview.get("receipt_sha256"))
            and _fresh(preview.get("observed_at", ""), now, FRESH_PREVIEW_SECONDS)
            and preview.get("root_annotations") == {"fleet.ai/failure-alerts": "off"}
            and preview.get("priority_class") == "c1"
            and preview.get("queue_priority") == "q1"
            and preview.get("backoff_limit") == 0
            and preview.get("nodes") == 1
            and preview.get("gpus_per_node") == 8
            and preview.get("requeue_if_preempted") is False
            and preview.get("optimizer_steps") == 0
            and preview.get("checkpoint") is False
            for row in lanes
        )
    )
    absence_ok = (
        exact_lanes
        and len({(row.get("absence") or {}).get("receipt_sha256") for row in lanes}) == 4
        and all(
            _sha((absence := row.get("absence") or {}).get("receipt_sha256"))
            and _fresh(absence.get("observed_at", ""), now, FRESH_PREVIEW_SECONDS)
            and absence.get("jobs_api_duplicates") == 0
            and absence.get("kubernetes_duplicates") == 0
            and absence.get("sfs_output_absent") is True
            for row in lanes
        )
    )
    observer_ok = (
        exact_lanes
        and len({(row.get("observer") or {}).get("receipt_sha256") for row in lanes}) == 4
        and all(
            _sha((observer := row.get("observer") or {}).get("receipt_sha256"))
            and observer.get("armed") is True
            and observer.get("healthy") is True
            and observer.get("uid_bound_release") is True
            and observer.get("delete_drain_supervised") is True
            and isinstance(observer.get("active_deadline_s"), int)
            and observer["active_deadline_s"] >= 18_900
            for row in lanes
        )
    )
    cleanup_ok = (
        _sha(cleanup.get("receipt_sha256"))
        and cleanup.get("prompt_removed_after_terminal") is True
        and cleanup.get("private_episode_material_restricted") is True
        and cleanup.get("output_create_once") is True
    )
    return {
        GATES[0]: bool(adapter_ok),
        GATES[1]: bool(operator_ok),
        GATES[2]: bool(_fresh(receipt["observed_at"], now, FRESH_TASK_SECONDS)),
        GATES[3]: bool(plans_ok),
        GATES[4]: bool(previews_ok),
        GATES[5]: bool(absence_ok),
        GATES[6]: bool(observer_ok),
        GATES[7]: bool(cleanup_ok),
        GATES[8]: False,
    }


def build_review_candidate(
    task_receipt: dict[str, Any],
    evidence: dict[str, Any] | None = None,
    *,
    observed_at: dt.datetime | None = None,
) -> dict[str, Any]:
    """Build a parent-review candidate; this function can never make it launchable."""
    wave = miles_signal_wave.load()
    now = _now(observed_at)
    validate_live_task_receipt(task_receipt)
    evidence = deepcopy(evidence or {})
    statuses = _candidate_gate_status(wave, task_receipt, evidence, now)
    body = {
        "schema": CANDIDATE_SCHEMA,
        "state": "parent_review_required",
        "launchable": False,
        "observed_at": _stamp(now),
        "authority": {
            "config_path": str(miles_signal_wave.PLAN_PATH.relative_to(ROOT)),
            "config_sha256": wave["sha256"],
            "source_commit": wave["source_commit"],
            "task_get_receipt_sha256": task_receipt["sha256"],
            "task_get_observed_at": task_receipt["observed_at"],
        },
        "future_bindings": evidence,
        "gates": [{"id": gate, "required": True, "passed": statuses[gate]} for gate in GATES],
        "transition_rule": (
            "Only approve_review_candidate may change launchable after all first eight "
            "gates pass and a separately sealed parent review binds this exact candidate."
        ),
    }
    return _sealed(body)


def validate_review_candidate(
    candidate: dict[str, Any], task_receipt: dict[str, Any] | None = None
) -> dict[str, Any]:
    _verify_seal(candidate, CANDIDATE_SCHEMA)
    if (
        candidate.get("state") != "parent_review_required"
        or candidate.get("launchable") is not False
    ):
        raise ValueError("review candidate must remain launchable false")
    gates = candidate.get("gates")
    if not isinstance(gates, list) or [row.get("id") for row in gates] != list(GATES):
        raise ValueError("review candidate does not enumerate the nine gates")
    if any(row.get("required") is not True for row in gates):
        raise ValueError("every transition gate is mandatory")
    if gates[-1].get("passed") is not False:
        raise ValueError("parent-review gate cannot pass in a review candidate")
    if task_receipt is not None:
        wave = miles_signal_wave.load()
        validate_live_task_receipt(task_receipt)
        if candidate["authority"].get("task_get_receipt_sha256") != task_receipt["sha256"]:
            raise ValueError("review candidate binds a different task receipt")
        statuses = _candidate_gate_status(
            wave,
            task_receipt,
            candidate.get("future_bindings") or {},
            _time(candidate["observed_at"]),
        )
        expected = [{"id": gate, "required": True, "passed": statuses[gate]} for gate in GATES]
        if gates != expected:
            raise ValueError("review candidate gate status is not reproducible")
    return candidate


def approve_review_candidate(
    candidate: dict[str, Any],
    parent_review: dict[str, Any],
    task_receipt: dict[str, Any],
    *,
    reviewed_at: dt.datetime | None = None,
) -> dict[str, Any]:
    """Create a launchable successor only from an exact separate parent review."""
    now = _now(reviewed_at)
    validate_review_candidate(candidate, task_receipt)
    validate_live_task_receipt(task_receipt, now=now, require_fresh=True)
    _verify_seal(parent_review, REVIEW_SCHEMA)
    if (
        parent_review.get("candidate_sha256") != candidate["sha256"]
        or parent_review.get("approved") is not True
        or parent_review.get("reviewer") != "root"
    ):
        raise ValueError("parent review does not approve this exact candidate")
    current = _candidate_gate_status(
        miles_signal_wave.load(), task_receipt, candidate["future_bindings"], now
    )
    if not all(current[gate] is True for gate in GATES[:-1]):
        raise ValueError("cannot approve an incomplete transition candidate")
    body = {
        **{k: deepcopy(v) for k, v in candidate.items() if k != "sha256"},
        "schema": APPROVED_SCHEMA,
        "state": "reviewed_launchable",
        "launchable": True,
        "predecessor_candidate_sha256": candidate["sha256"],
        "parent_review_receipt_sha256": parent_review["sha256"],
    }
    body["gates"][-1]["passed"] = True
    return _sealed(body)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect-live", action="store_true")
    args = parser.parse_args(argv)
    if not args.collect_live:
        parser.error("use --collect-live")
    import os

    value = collect_live_task_receipt(FleetClient(os.environ.get("FLEET_API_KEY", "")))
    print(json.dumps(value, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
