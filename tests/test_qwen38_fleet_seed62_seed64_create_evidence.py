from __future__ import annotations

import copy
import hashlib
import json
import uuid
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/qwen38-study"
CREATE = EVIDENCE / "2026-09-23-q38-dev17-seed62-seed64-launcher-create-receipt.json"
HANDOFF = EVIDENCE / "2026-09-23-q38-dev17-seed62-seed64-postcreate-handoff.json"
REPOSITORY_COMMIT = "955416264369013d5584edb8c4c473b5a5be306f"

EXPECTED = {
    CREATE: (
        "e7073e4a8dfef52dc9d4df9365282f8b6da36e40e1c340fb136af338a2c20b2a",
        "sha256:bb848c1b0afc0a259438e623280626adfe6d0b18f95a7f5d3803df52777f82d6",
        "cyber_fleet_seed62_seed63_seed64_launcher_create_receipt_v1",
    ),
    HANDOFF: (
        "6f3c88ab31963062024c5099a22c67f6813371426509bb522b4f73f3970f64c3",
        "sha256:9b766e91256aaa5905ba4ae123ce680b6ac88c5327be5b5c4f7cd3924d0daa91",
        "cyber_fleet_seed62_seed63_seed64_postcreate_handoff_v1",
    ),
}

FORBIDDEN_PRIVATE_KEYS = {
    "answer",
    "api_key",
    "cell_id",
    "credentials",
    "flag",
    "prompt",
    "prompt_text",
    "response",
    "response_content",
    "reward",
    "score",
    "scores",
    "session",
    "session_id",
    "task_id",
    "task_key",
    "task_version_id",
    "trace",
    "trace_content",
    "verifier_execution_id",
}
ALLOWED_INFRASTRUCTURE_KEYS = {
    "admitted",
    "arm_id",
    "config_maps_created",
    "count",
    "create_calls_performed",
    "create_intent_file_sha256",
    "create_intent_sha256",
    "create_once",
    "create_receipt_file_sha256",
    "create_receipt_sha256",
    "create_response_file_sha256",
    "create_response_resource_version",
    "current_resource_version",
    "evaluator_config_map_name",
    "evaluator_config_map_resource_version",
    "evaluator_config_map_uid",
    "evaluator_config_maps_created_by_launchers",
    "evaluator_job_name",
    "evaluator_job_resource_version",
    "evaluator_job_uid",
    "evaluator_jobs_created_by_launchers",
    "external_mutations",
    "failure_alerts_off",
    "gpu_requests",
    "infrastructure_state_only",
    "inner_evaluators",
    "inner_evaluators_created_count",
    "job_active",
    "job_failed",
    "job_succeeded",
    "jobs_created",
    "kind",
    "launcher_config_maps_created",
    "launcher_jobs_created",
    "name",
    "objects",
    "observed_at",
    "other",
    "outer_launcher_objects",
    "outer_launchers_terminal_success_count",
    "pod_name",
    "pod_phase",
    "pod_restarts",
    "pod_uid",
    "postgres_client_label",
    "priority_class",
    "privacy",
    "prompts_responses_flags_rewards_or_trace_content_read",
    "repeat_create_calls_performed",
    "repository_commit",
    "resource_version_note",
    "root_job_policy",
    "schema",
    "scores_read",
    "seed65_created",
    "sha256",
    "successor_seed",
    "uid",
    "workload_name",
    "workload_uid",
}
INNER_REQUIRED_UID_FIELDS = (
    "evaluator_config_map_uid",
    "evaluator_job_uid",
    "workload_uid",
)
INNER_UID_FIELDS = (*INNER_REQUIRED_UID_FIELDS, "pod_uid")


def _canonical(value: dict[str, Any]) -> str:
    body = {key: item for key, item in value.items() if key != "sha256"}
    return (
        "sha256:"
        + hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
    )


def _read(path: Path) -> dict[str, Any]:
    assert path.is_file() and not path.is_symlink()
    value = json.loads(path.read_text())
    assert isinstance(value, dict)
    return value


def _assert_sanitized(value: Any) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            assert key in ALLOWED_INFRASTRUCTURE_KEYS
            assert key.lower() not in FORBIDDEN_PRIVATE_KEYS
            _assert_sanitized(nested)
    elif isinstance(value, list):
        for nested in value:
            _assert_sanitized(nested)


def _object_index(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    result = {(row["kind"], row["name"]): row for row in rows}
    assert len(result) == len(rows)
    return result


def _assert_inner_evaluator_bindings(rows: list[dict[str, Any]]) -> None:
    assert len(rows) == 6
    assert {(row["successor_seed"], row["arm_id"]) for row in rows} == {
        (seed, arm) for seed in (62, 63, 64) for arm in ("base", "candidate")
    }
    all_uids: list[str] = []
    for row in rows:
        route = "base" if row["arm_id"] == "base" else "t3k32s1000"
        job_name = f"chris-q38-dev17-s{row['successor_seed']}-{route}-repl-p1-v2"
        assert row["evaluator_job_name"] == job_name
        assert row["evaluator_config_map_name"] == job_name.replace("-p1-v2", "-code-v2")
        assert row["workload_name"].startswith(f"job-{job_name}-")
        for field in INNER_REQUIRED_UID_FIELDS:
            uuid.UUID(row[field])
            all_uids.append(row[field])
        if row["admitted"]:
            assert row["pod_name"].startswith(f"{job_name}-")
            uuid.UUID(row["pod_uid"])
            all_uids.append(row["pod_uid"])
        else:
            assert row["pod_name"] is row["pod_uid"] is None
    for field in INNER_REQUIRED_UID_FIELDS:
        assert len({row[field] for row in rows}) == 6
    pod_uids = [row["pod_uid"] for row in rows if row["pod_uid"] is not None]
    assert len(pod_uids) == sum(row["admitted"] for row in rows)
    assert len(pod_uids) == len(set(pod_uids))
    assert len(all_uids) == len(set(all_uids))


def _require_same_created_uid(created: dict[str, Any], current: dict[str, Any]) -> dict[str, str]:
    """Bind an immutable UID while retaining, not equating, two resource versions."""
    assert (created["kind"], created["name"]) == (current["kind"], current["name"])
    assert created["uid"] == current["uid"], "created resource UID drift"
    uuid.UUID(created["uid"])
    create_version = created["create_response_resource_version"]
    current_version = current["current_resource_version"]
    assert isinstance(create_version, str) and create_version.isdecimal()
    assert isinstance(current_version, str) and current_version.isdecimal()
    return {
        "uid": created["uid"],
        "create_response_resource_version": create_version,
        "current_resource_version": current_version,
    }


def test_exact_create_and_postcreate_evidence_bytes_are_preserved_and_sanitized() -> None:
    for path, (file_sha256, self_sha256, schema) in EXPECTED.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == file_sha256
        value = _read(path)
        assert value["schema"] == schema
        assert value["sha256"] == self_sha256 == _canonical(value)
        assert value["repository_commit"] == REPOSITORY_COMMIT
        _assert_sanitized(value)


@pytest.mark.parametrize("key", sorted(FORBIDDEN_PRIVATE_KEYS))
def test_private_key_variants_fail_closed(key: str) -> None:
    with pytest.raises(AssertionError):
        _assert_sanitized({key: "private"})


@pytest.mark.parametrize(
    "key",
    (
        "auth_credentials",
        "cell",
        "cell_ids",
        "credential",
        "model_response",
        "prompt_content",
        "score_values",
        "session_ids",
        "task",
        "task_keys",
        "trace_data",
    ),
)
def test_unknown_private_key_aliases_fail_closed(key: str) -> None:
    with pytest.raises(AssertionError):
        _assert_sanitized({key: "private"})


def test_single_create_is_uid_bound_while_controller_resource_versions_may_advance() -> None:
    create = _read(CREATE)
    handoff = _read(HANDOFF)
    assert create["create_calls_performed"] == handoff["create_calls_performed"] == 1
    assert create["repeat_create_calls_performed"] == handoff["repeat_create_calls_performed"] == 0
    assert create["seed65_created"] is handoff["seed65_created"] is False
    assert handoff["create_receipt_file_sha256"] == "sha256:" + EXPECTED[CREATE][0]
    assert handoff["create_receipt_sha256"] == EXPECTED[CREATE][1]
    assert handoff["create_response_file_sha256"] == create["create_response_file_sha256"]

    created = _object_index(create["objects"])
    observed = _object_index(handoff["outer_launcher_objects"])
    assert set(created) == set(observed)
    bindings = {key: _require_same_created_uid(created[key], observed[key]) for key in created}
    assert len({row["uid"] for row in bindings.values()}) == 12
    assert sum(kind == "Job" for kind, _ in bindings) == 6
    assert sum(kind == "ConfigMap" for kind, _ in bindings) == 6

    changed_versions = {
        key
        for key, row in bindings.items()
        if row["current_resource_version"] != row["create_response_resource_version"]
    }
    assert changed_versions == {key for key in bindings if key[0] == "Job"}
    assert create["resource_version_note"] == (
        "Job resourceVersions advanced after create through normal admission-controller updates; "
        "immutable UIDs match the single create response"
    )


def test_uid_drift_fails_closed_even_when_resource_version_change_is_valid() -> None:
    uid = "11111111-2222-4333-8444-555555555555"
    created = {
        "kind": "Job",
        "name": "example",
        "uid": uid,
        "create_response_resource_version": "100",
    }
    advanced = {
        "kind": "Job",
        "name": "example",
        "uid": uid,
        "current_resource_version": "101",
    }
    assert _require_same_created_uid(created, advanced) == {
        "uid": uid,
        "create_response_resource_version": "100",
        "current_resource_version": "101",
    }
    drifted = copy.deepcopy(advanced)
    drifted["uid"] = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
    with pytest.raises(AssertionError, match="UID drift"):
        _require_same_created_uid(created, drifted)


def test_postcreate_handoff_is_infrastructure_only_and_complete() -> None:
    create = _read(CREATE)
    handoff = _read(HANDOFF)
    assert create["root_job_policy"] == {
        "count": 6,
        "create_once": True,
        "failure_alerts_off": True,
        "gpu_requests": 0,
        "postgres_client_label": True,
        "priority_class": "c1",
    }
    assert create["external_mutations"] == {
        "config_maps_created": 6,
        "jobs_created": 6,
        "other": 0,
    }
    assert handoff["external_mutations"] == {
        "evaluator_config_maps_created_by_launchers": 6,
        "evaluator_jobs_created_by_launchers": 6,
        "launcher_config_maps_created": 6,
        "launcher_jobs_created": 6,
        "other": 0,
    }
    assert handoff["outer_launchers_terminal_success_count"] == 6
    assert handoff["inner_evaluators_created_count"] == 6
    _assert_inner_evaluator_bindings(handoff["inner_evaluators"])
    assert handoff["privacy"] == {
        "infrastructure_state_only": True,
        "prompts_responses_flags_rewards_or_trace_content_read": False,
        "scores_read": False,
    }


@pytest.mark.parametrize("field", INNER_UID_FIELDS)
def test_inner_evaluator_uid_bindings_fail_closed(field: str) -> None:
    rows = copy.deepcopy(_read(HANDOFF)["inner_evaluators"])
    rows[0][field] = "not-a-kubernetes-uid"
    with pytest.raises(ValueError):
        _assert_inner_evaluator_bindings(rows)

    rows = copy.deepcopy(_read(HANDOFF)["inner_evaluators"])
    duplicate_index = 4 if field == "pod_uid" else 1
    rows[duplicate_index][field] = rows[0][field]
    with pytest.raises(AssertionError):
        _assert_inner_evaluator_bindings(rows)


def test_inner_evaluator_uid_bindings_reject_cross_kind_duplicates() -> None:
    rows = copy.deepcopy(_read(HANDOFF)["inner_evaluators"])
    rows[1]["evaluator_job_uid"] = rows[0]["evaluator_config_map_uid"]
    with pytest.raises(AssertionError):
        _assert_inner_evaluator_bindings(rows)
