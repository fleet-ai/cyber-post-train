"""Build the fail-closed review transition for the four Miles signal lanes."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
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

ADAPTER_FREEZE_SCHEMA = "cyber_qwen38_miles96_adapter_freeze_v1"
OPERATOR_FREEZE_SCHEMA = "cyber_qwen38_miles96_operator_freeze_v1"
POST_BUNDLE_SCHEMA = "cyber_qwen38_miles_signal_post_receipt_bundle_v1"
TEST_RECEIPT_SCHEMA = "cyber_qwen38_miles96_focused_tests_v1"
IMAGE_PREFLIGHT_SCHEMA = "cyber_qwen38_miles96_exact_image_preflight_v1"
OPERATOR_SOURCE_PATHS = (
    "cyber_post_train/gpu_capacity.py",
    "cyber_post_train/jobs.py",
    "cyber_post_train/sfs_output.py",
    "evals/fleet/opencode_self_hosted.py",
    "training/dev_cleanup_observer.py",
    "training/fleet.py",
    "training/miles96_mechanics_launch.py",
    "training/miles_signal_transition.py",
    "training/miles_signal_wave.py",
)

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


def load_receipt(path: Path, expected_file_sha256: str) -> dict[str, Any]:
    """Load an exact JSON file and retain both its byte and body identities."""
    raw = path.read_bytes()
    observed = "sha256:" + hashlib.sha256(raw).hexdigest()
    if observed != expected_file_sha256:
        raise ValueError("review receipt file digest changed")
    body = json.loads(raw)
    return {
        "file_sha256": observed,
        "body_sha256": "sha256:" + digest(body),
        "body": body,
    }


def _verify_loaded_receipt(value: dict[str, Any], expected_file_sha256: str) -> dict[str, Any]:
    if set(value) != {"file_sha256", "body_sha256", "body"}:
        raise ValueError("review receipt wrapper fields changed")
    if value["file_sha256"] != expected_file_sha256:
        raise ValueError("review receipt file identity changed")
    body = value["body"]
    if not isinstance(body, dict) or value["body_sha256"] != "sha256:" + digest(body):
        raise ValueError("review receipt body identity changed")
    return body


def _operator_source_manifest() -> dict[str, str]:
    return {
        path: "sha256:" + hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        for path in OPERATOR_SOURCE_PATHS
    }


def _lane_objects() -> list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]:
    """Rebuild every lane from the sealed task authority and current adapter bytes."""
    from training import miles96_signal_qualification as signal

    wave = miles_signal_wave.load()
    result = []
    for row in wave["candidates"]:
        task = {
            **miles_signal_wave.task_binding(wave, row),
            "authority_receipt_sha256": row["authority_receipt_sha256"],
        }
        plan = signal.build_plan(
            name=row["identity"]["name"],
            model_root=signal.HF_MODEL_ROOT,
            model_binding_sha256=signal.HF_MODEL_BINDING_SHA256,
            task_binding=task,
            authority_config_sha256=wave["sha256"],
            current_binding_sha256=row["live_binding_receipt_sha256"],
            production_split_sha256=wave["authorities"]["production_split"]["self_sha256"],
        )
        request = signal.job_request(plan)
        result.append((row, plan, request))
    return result


def _lane_receipts() -> list[dict[str, Any]]:
    from training import miles96_signal_qualification as signal

    return [
        {
            "name": row["identity"]["name"],
            "plan_sha256": "sha256:" + digest(plan),
            "request_sha256": "sha256:" + digest(request),
            "runtime_bundle_sha256": request["env"]["CYBER_RUNTIME_BUNDLE_SHA256"],
            "request_binding_sha256": request["env"]["CYBER_REQUEST_BINDING_SHA256"],
            "live_binding_receipt_sha256": row["live_binding_receipt_sha256"],
            "optimizer_steps": plan["qualification"]["optimizer_steps"],
            "checkpoint": plan["acceptance"].get("checkpoint", False),
            "sample_indexes": [
                slot["sample_index"] for slot in plan["qualification"]["sample_slots"]
            ],
            "max_concurrent_episodes": plan["episode"]["max_concurrent_envs"],
            "runtime_source_manifest_sha256": "sha256:" + digest(signal.runtime_source_manifest()),
        }
        for row, plan, request in _lane_objects()
    ]


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
        if response.get("environment_version_id") != candidate["environment"]["version_id"]:
            raise ValueError(
                f"live environment-version binding drifted: {candidate['identity']['name']}"
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


def _validate_test_receipt(
    wrapper: dict[str, Any],
    *,
    expected_file_sha256: str,
    expected_commit: str,
    expected_subject: str,
) -> dict[str, Any]:
    body = _verify_loaded_receipt(wrapper, expected_file_sha256)
    _verify_seal(body, TEST_RECEIPT_SCHEMA)
    if (
        body.get("subject") != expected_subject
        or body.get("commit") != expected_commit
        or body.get("status") != "passed"
        or not isinstance(body.get("commands"), list)
        or not body["commands"]
        or not all(isinstance(command, str) and command for command in body["commands"])
        or type(body.get("passed")) is not int
        or body["passed"] < 1
        or body.get("failed") != 0
        or body.get("ruff_check") is not True
        or body.get("ruff_format_check") is not True
        or body.get("git_diff_check") is not True
    ):
        raise ValueError("focused test receipt is incomplete")
    _time(body["observed_at"])
    return body


def _validate_image_preflight(
    wrapper: dict[str, Any],
    *,
    expected_file_sha256: str,
    expected_runtime_image: str,
    expected_source_closure_sha256: str,
) -> dict[str, Any]:
    body = _verify_loaded_receipt(wrapper, expected_file_sha256)
    _verify_seal(body, IMAGE_PREFLIGHT_SCHEMA)
    image_digest = "sha256:" + expected_runtime_image.rsplit("@sha256:", 1)[1]
    checks = body.get("checks") or {}
    if (
        body.get("status") != "passed"
        or body.get("runtime_image") != expected_runtime_image
        or body.get("image_digest") != image_digest
        or body.get("source_closure_sha256") != expected_source_closure_sha256
        or checks
        != {
            "image_digest_exact": True,
            "pinned_fti_imports": True,
            "pinned_miles_sources": True,
            "zero_update_entrypoint": True,
        }
    ):
        raise ValueError("exact-image preflight receipt is incomplete")
    _time(body["observed_at"])
    return body


def _validate_adapter_freeze(
    wrapper: dict[str, Any], *, expected_file_sha256: str, expected_commit: str
) -> dict[str, Any]:
    from training import miles96_mechanics_canary as mechanics
    from training import miles96_signal_qualification as signal

    body = _verify_loaded_receipt(wrapper, expected_file_sha256)
    excluded = {
        "optimizer_steps",
        "checkpoint",
        "sample_indexes",
        "max_concurrent_episodes",
        "runtime_source_manifest_sha256",
    }
    expected_lanes = [
        {key: value for key, value in lane.items() if key not in excluded}
        for lane in _lane_receipts()
    ]
    tests = body.get("tests") or {}
    if (
        set(body)
        != {
            "schema",
            "branch",
            "commit",
            "clean",
            "authority_sha256",
            "runtime_image",
            "source_file_count",
            "source_closure_sha256",
            "tests_receipt_sha256",
            "exact_image_preflight_receipt_sha256",
            "tests",
            "lanes",
        }
        or body.get("schema") != ADAPTER_FREEZE_SCHEMA
        or body.get("commit") != expected_commit
        or body.get("clean") is not True
        or body.get("authority_sha256") != miles_signal_wave.load()["sha256"]
        or body.get("runtime_image") != mechanics.IMAGE
        or body.get("source_file_count") != len(signal.runtime_source_manifest())
        or body.get("source_closure_sha256") != "sha256:" + digest(signal.runtime_source_manifest())
        or not _sha(body.get("tests_receipt_sha256"))
        or not _sha(body.get("exact_image_preflight_receipt_sha256"))
        or type(tests.get("passed")) is not int
        or tests["passed"] < 1
        or tests.get("failed") != 0
        or tests.get("ruff_check") is not True
        or tests.get("ruff_format_check") is not True
        or tests.get("git_diff_check") is not True
        or body.get("lanes") != expected_lanes
    ):
        raise ValueError("adapter freeze receipt differs from rebuilt lanes")
    return body


def _validate_operator_freeze(
    wrapper: dict[str, Any], *, expected_file_sha256: str, expected_commit: str
) -> dict[str, Any]:
    body = _verify_loaded_receipt(wrapper, expected_file_sha256)
    manifest = _operator_source_manifest()
    tests = body.get("tests") or {}
    if (
        set(body)
        != {
            "schema",
            "commit",
            "clean",
            "source_manifest",
            "source_closure_sha256",
            "tests_receipt_sha256",
            "tests",
        }
        or body.get("schema") != OPERATOR_FREEZE_SCHEMA
        or body.get("commit") != expected_commit
        or body.get("clean") is not True
        or body.get("source_manifest") != manifest
        or body.get("source_closure_sha256") != "sha256:" + digest(manifest)
        or not _sha(body.get("tests_receipt_sha256"))
        or type(tests.get("passed")) is not int
        or tests["passed"] < 1
        or tests.get("failed") != 0
    ):
        raise ValueError("operator freeze receipt differs from current launch sources")
    return body


def _candidate_gate_status(
    wave: dict[str, Any], receipt: dict[str, Any], evidence: dict[str, Any], now: dt.datetime
) -> dict[str, bool]:
    pins = evidence.get("pins") or {}
    adapter_ok = operator_ok = plans_ok = False
    try:
        adapter = _validate_adapter_freeze(
            evidence["adapter_freeze"],
            expected_file_sha256=pins["adapter_freeze_file_sha256"],
            expected_commit=pins["adapter_commit"],
        )
        adapter_tests = _validate_test_receipt(
            evidence["adapter_test_receipt"],
            expected_file_sha256=pins["adapter_test_receipt_file_sha256"],
            expected_commit=pins["adapter_commit"],
            expected_subject="adapter",
        )
        image = _validate_image_preflight(
            evidence["exact_image_preflight_receipt"],
            expected_file_sha256=pins["exact_image_preflight_receipt_file_sha256"],
            expected_runtime_image=adapter["runtime_image"],
            expected_source_closure_sha256=adapter["source_closure_sha256"],
        )
        if (
            adapter["tests_receipt_sha256"] != adapter_tests["sha256"]
            or adapter["exact_image_preflight_receipt_sha256"] != image["sha256"]
        ):
            raise ValueError("adapter freeze references different supporting receipts")
        adapter_ok = True
        operator = _validate_operator_freeze(
            evidence["operator_freeze"],
            expected_file_sha256=pins["operator_freeze_file_sha256"],
            expected_commit=pins["operator_commit"],
        )
        operator_tests = _validate_test_receipt(
            evidence["operator_test_receipt"],
            expected_file_sha256=pins["operator_test_receipt_file_sha256"],
            expected_commit=pins["operator_commit"],
            expected_subject="operator",
        )
        if operator["tests_receipt_sha256"] != operator_tests["sha256"]:
            raise ValueError("operator freeze references a different test receipt")
        operator_ok = True
        rebuilt = _lane_receipts()
        plans_ok = (
            evidence.get("lanes") == rebuilt
            and len(rebuilt) == len(wave["candidates"]) == 4
            and len({row["plan_sha256"] for row in rebuilt}) == 4
            and len({row["request_sha256"] for row in rebuilt}) == 4
        )
    except (KeyError, TypeError, ValueError):
        pass
    return {
        GATES[0]: bool(adapter_ok),
        GATES[1]: bool(operator_ok),
        GATES[2]: bool(_fresh(receipt["observed_at"], now, FRESH_TASK_SECONDS)),
        GATES[3]: bool(plans_ok),
        GATES[4]: False,
        GATES[5]: False,
        GATES[6]: False,
        # The exact adapter source closure contains the create-once private
        # evidence directory, terminal prompt removal, and zero-checkpoint
        # assertions. Runtime receipts re-prove those claims after execution.
        GATES[7]: bool(adapter_ok),
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
    if evidence:
        evidence["lanes"] = _lane_receipts()
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


def validate_parent_review(
    candidate: dict[str, Any],
    parent_review: dict[str, Any],
    *,
    expected_parent_review_sha256: str,
) -> dict[str, Any]:
    """Require an independently supplied digest for the root-issued review."""
    validate_review_candidate(candidate)
    _verify_seal(parent_review, REVIEW_SCHEMA)
    if (
        parent_review["sha256"] != expected_parent_review_sha256
        or parent_review.get("candidate_sha256") != candidate["sha256"]
        or parent_review.get("approved") is not True
        or parent_review.get("reviewer") != "root"
    ):
        raise ValueError("independently pinned parent review does not approve this candidate")
    return parent_review


def _verify_receipt(value: dict[str, Any], schema: str, *, prefixed: bool = True) -> None:
    if not isinstance(value, dict) or value.get("schema") != schema:
        raise ValueError(f"invalid {schema} body")
    body = {key: item for key, item in value.items() if key != "sha256"}
    expected = ("sha256:" if prefixed else "") + digest(body)
    if value.get("sha256") != expected:
        raise ValueError(f"invalid {schema} seal")


def validate_post_receipt_bundle(
    bundle: dict[str, Any],
    plan: dict[str, Any],
    request: dict[str, Any],
    *,
    now: dt.datetime | None = None,
) -> dict[str, Any]:
    """Validate the exact sanitized receipts used by one imminent POST."""
    from cyber_post_train.sfs_output import validate_output_absence_receipt
    from training import dev_cleanup_observer as cleanup
    from training import miles96_mechanics_launch as launch
    from training import miles96_signal_qualification as signal

    now = _now(now)
    plan = signal.validate_plan(plan)
    if request != signal.job_request(plan):
        raise ValueError("POST receipt bundle request differs from rebuilt request")
    if set(bundle) != {
        "schema",
        "name",
        "plan_sha256",
        "request_sha256",
        "server_preview",
        "observer_armed",
        "fresh_absence",
        "final_prepost_gate",
        "sfs_output_absence",
        "capacity_gate",
        "sha256",
    }:
        raise ValueError("POST receipt bundle fields changed")
    _verify_seal(bundle, POST_BUNDLE_SCHEMA)
    plan_sha = "sha256:" + digest(plan)
    request_sha = "sha256:" + digest(request)
    if (
        bundle["name"] != request["name"]
        or bundle["plan_sha256"] != plan_sha
        or bundle["request_sha256"] != request_sha
    ):
        raise ValueError("POST receipt bundle identity changed")

    preview = bundle["server_preview"]
    _verify_receipt(preview, "cyber_miles96_live_server_preview_v1")
    if (
        preview.get("plan_sha256") != plan_sha
        or preview.get("request_sha256") != request_sha
        or preview.get("root_failure_alerts") != "off"
        or preview.get("backoff_limit") != 0
        or preview.get("shutdown_after_job_finishes") is not True
        or preview.get("nodes") != 1
        or preview.get("gpus") != 8
        or preview.get("preview_count") != 2
        or preview.get("priority_class") != "c1"
        or preview.get("queue_priority") != "q1"
        or preview.get("requeue_if_preempted") is not False
        or type(preview.get("observed_at_unix")) not in {int, float}
        or not 0 <= now.timestamp() - float(preview["observed_at_unix"]) <= FRESH_PREVIEW_SECONDS
    ):
        raise ValueError("server preview receipt is stale or unsafe")

    observer = bundle["observer_armed"]
    _verify_receipt(observer, cleanup.JOBS_API_PREFIX_GUARD_SCHEMA)
    if (
        observer.get("status") != "armed_non_destructive_prefix_guard"
        or observer.get("run_name_prefix") != request["name"]
        or observer.get("run_dir") != request["run_dir"]
        or observer.get("image") != request["image"]
        or observer.get("plan_sha256") != plan_sha
        or observer.get("manifest_sha256") != preview.get("manifest_sha256")
        or observer.get("maximum_seconds") != launch._maximum_seconds(plan, request)
        or observer.get("expected_gpus") != 8
        or observer.get("prefix_collision_count_before_post") != 0
        or type(observer.get("observer_pid")) is not int
        or not _fresh(observer.get("armed_at", ""), now, FRESH_PREVIEW_SECONDS)
    ):
        raise ValueError("observer receipt is stale or differs from the lane")
    try:
        os.kill(observer["observer_pid"], 0)
    except OSError as exc:
        raise ValueError("observer process is not alive") from exc

    absence = bundle["fresh_absence"]
    _verify_receipt(absence, "cyber_miles96_fresh_absence_v1")
    final = bundle["final_prepost_gate"]
    _verify_receipt(final, "cyber_miles96_final_prepost_gate_v1")
    now_epoch = now.timestamp()
    for value in (absence, final):
        if (
            value.get("status") != "passed"
            or value.get("plan_sha256") != plan_sha
            or value.get("request_sha256") != request_sha
            or value.get("observer_armed_sha256") != observer["sha256"]
            or type(value.get("observed_at_unix")) not in {int, float}
            or not 0 <= now_epoch - float(value["observed_at_unix"]) <= FRESH_PREVIEW_SECONDS
        ):
            raise ValueError("absence or final pre-POST receipt is stale or mismatched")
    if absence.get("sfs_output_absent") is not True:
        raise ValueError("fresh absence receipt does not prove SFS absence")

    sfs = bundle["sfs_output_absence"]
    validate_output_absence_receipt(sfs, plan, request, now=now_epoch)
    if absence.get("sfs_output_absence_receipt_sha256") != sfs["sha256"]:
        raise ValueError("fresh absence receipt names a different SFS proof")

    capacity = bundle["capacity_gate"]
    _verify_receipt(capacity, launch.CAPACITY_GATE_SCHEMA)
    census = capacity.get("capacity_census") or {}
    if (
        capacity.get("status") != "passed"
        or capacity.get("plan_sha256") != plan_sha
        or capacity.get("request_sha256") != request_sha
        or capacity.get("planned") != {"nodes": 1, "gpus": 8}
        or not 0
        <= now.timestamp() - launch._timestamp(capacity.get("observed_at"))
        <= launch.CAPACITY_MAX_AGE_SECONDS
        or census.get("qualified") is not True
        or census.get("limits") != {"nodes": 10, "gpus": 80}
        or (census.get("projected") or {}).get("nodes", 11) > 10
        or (census.get("projected") or {}).get("gpus", 81) > 80
    ):
        raise ValueError("capacity receipt does not admit this one-node lane")
    return bundle


def approve_review_candidate(
    candidate: dict[str, Any],
    parent_review: dict[str, Any],
    task_receipt: dict[str, Any],
    post_receipt_bundle: dict[str, Any],
    plan: dict[str, Any],
    request: dict[str, Any],
    *,
    expected_parent_review_sha256: str,
    reviewed_at: dt.datetime | None = None,
) -> dict[str, Any]:
    """Create one lane successor only from exact static review and POST-time receipts."""
    now = _now(reviewed_at)
    validate_review_candidate(candidate, task_receipt)
    validate_live_task_receipt(task_receipt, now=now, require_fresh=True)
    validate_parent_review(
        candidate,
        parent_review,
        expected_parent_review_sha256=expected_parent_review_sha256,
    )
    current = _candidate_gate_status(
        miles_signal_wave.load(), task_receipt, candidate["future_bindings"], now
    )
    if not all(current[gate] is True for gate in (*GATES[:4], GATES[7])):
        raise ValueError("cannot approve a statically incomplete transition candidate")
    validate_post_receipt_bundle(post_receipt_bundle, plan, request, now=now)
    lane = next(
        (row for row in candidate["future_bindings"]["lanes"] if row["name"] == request["name"]),
        None,
    )
    if (
        lane is None
        or lane["plan_sha256"] != "sha256:" + digest(plan)
        or lane["request_sha256"] != "sha256:" + digest(request)
    ):
        raise ValueError("POST-time receipts belong to a different reviewed lane")
    body = {
        "schema": APPROVED_SCHEMA,
        "state": "reviewed_launchable",
        "launchable": True,
        "lane": request["name"],
        "plan_sha256": lane["plan_sha256"],
        "request_sha256": lane["request_sha256"],
        "predecessor_candidate_sha256": candidate["sha256"],
        "parent_review_receipt_sha256": parent_review["sha256"],
        "task_get_receipt_sha256": task_receipt["sha256"],
        "post_receipt_bundle_sha256": post_receipt_bundle["sha256"],
        "approved_at": _stamp(now),
        "gates": [{"id": gate, "required": True, "passed": True} for gate in GATES],
    }
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
