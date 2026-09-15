from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

import evals.platform_v2.legacy_import as legacy_import
from evals.platform_v2.legacy_import import (
    EXPECTED_IMPORT_PROTOCOL,
    FLEET_TEAM_ID,
    LEGACY_IMPORT_PROTOCOL,
    PlatformImportError,
    build_plan,
    digest_without,
    ensure_repository,
    import_request,
    load_identity_roster,
    load_selection,
    monitor_receipts,
    receipt_journal,
    require_submit_safe,
    sha256,
    source_identity,
    submission_summary,
    submit_rows,
    validate_plan,
    write_json_create_once,
)

SELECTION = Path("configs/data/fleet-a62-task-split-v1.json")


def _roster(selection: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "chris.cyber.v2.source-roster.v1",
        "source_job_id": selection["source"]["job_id"],
        "tasks": [
            {
                "task_key": row["task_key"],
                "eval_task_id": f"00000000-0000-4000-8000-{index:012d}",
                "current_task_version_id": row["task_version_id"],
            }
            for index, row in enumerate(selection["tasks"], start=1)
        ],
    }


def _clients(
    selection: dict[str, Any],
    roster: dict[str, Any],
    *,
    protocol: str = EXPECTED_IMPORT_PROTOCOL,
    moved: int = 0,
    mismatched_eval_id: bool = False,
    forbidden: bool = False,
    response_pin_drift: bool = False,
) -> tuple[httpx.Client, httpx.Client, list[dict[str, Any]], list[str]]:
    frozen = {row["task_key"]: row["task_version_id"] for row in selection["tasks"]}
    eval_ids = {row["task_key"]: row["eval_task_id"] for row in roster["tasks"]}
    movable_keys = [
        row["task_key"]
        for row in selection["tasks"]
        if row["env_key"] != "cysec1-2-current-fubspot-gen"
    ]
    moved_keys = set(movable_keys[:moved])
    posts: list[dict[str, Any]] = []
    fleet_reads: list[str] = []

    def fleet_handler(request: httpx.Request) -> httpx.Response:
        task_key = request.url.path.rsplit("/", 1)[-1]
        fleet_reads.append(task_key)
        version = frozen[task_key]
        if task_key in moved_keys:
            version = "ffffffff-ffff-4fff-8fff-ffffffffffff"
        eval_task_id = eval_ids[task_key]
        if mismatched_eval_id and task_key == selection["tasks"][0]["task_key"]:
            eval_task_id = "eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee"
        return httpx.Response(
            200,
            json={"id": eval_task_id, "key": task_key, "eval_task_version_id": version},
        )

    def registry_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/.well-known/fleet-registry":
            return httpx.Response(
                200,
                json={
                    "legacy_task_imports": {
                        "base_path": "/api/v1/imports/legacy-tasks",
                        "protocol": protocol,
                    }
                },
            )
        payload = json.loads(request.content)
        posts.append(payload)
        if forbidden:
            return httpx.Response(403, json={"private": "must-not-surface"})
        expected_eval_task_id = payload["expected_eval_task_id"]
        if response_pin_drift:
            expected_eval_task_id = "dddddddd-dddd-4ddd-8ddd-dddddddddddd"
        return httpx.Response(
            202,
            json={
                "import_id": f"imp_{len(posts):03d}",
                "state": "requested",
                "task_key": payload["task_key"],
                "source_team_id": FLEET_TEAM_ID,
                "protocol_version": EXPECTED_IMPORT_PROTOCOL,
                "expected_eval_task_id": expected_eval_task_id,
                "expected_current_task_version_id": payload[
                    "expected_current_task_version_id"
                ],
                "destination": payload["destination"],
                "modality_decision": payload["modality_decision"],
            },
        )

    return (
        httpx.Client(transport=httpx.MockTransport(fleet_handler)),
        httpx.Client(transport=httpx.MockTransport(registry_handler)),
        posts,
        fleet_reads,
    )


def _plan(
    selection: dict[str, Any],
    roster: dict[str, Any],
    *,
    protocol: str = EXPECTED_IMPORT_PROTOCOL,
    moved: int = 0,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    fleet, registry, posts, reads = _clients(
        selection, roster, protocol=protocol, moved=moved
    )
    with fleet, registry:
        plan = build_plan(
            selection,
            identity_roster=roster if protocol == EXPECTED_IMPORT_PROTOCOL else None,
            fleet_client=fleet,
            registry_client=registry,
            namespace="gentle-ember-ledger",
            repository="fleet-cyber-a62-frozen",
        )
    return plan, posts, reads


def _receipt_for_plan(plan: dict[str, Any], *, state: str = "requested") -> dict[str, Any]:
    row = next(
        row for row in plan["rows"] if row["disposition"] == "eligible_exact_current_frozen"
    )
    receipt = {
        "schema": "fleet_platform_v2_cyber_import_row_receipt_v3",
        "plan_sha256": plan["plan_sha256"],
        "request_sha256": row["request_sha256"],
        "task_key": row["task_key"],
        "task_version_id": row["frozen_task_version_id"],
        "protocol_version": EXPECTED_IMPORT_PROTOCOL,
        "expected_eval_task_id": row["expected_eval_task_id"],
        "expected_current_task_version_id": row["frozen_task_version_id"],
        "source_identity_sha256": row["source_identity_sha256"],
        "destination": {**plan["destination"], "tag": row["destination_tag"]},
        "import_id": "lti_resume_example",
        "state": state,
        "idempotent_replay": False,
    }
    receipt["receipt_sha256"] = digest_without(receipt, "receipt_sha256")
    return receipt


def test_frozen_selection_and_reviewed_identity_roster_verify(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection = load_selection(SELECTION)
    assert len(selection["tasks"]) == 160
    reviewed = load_identity_roster(
        Path("configs/data/fleet-a62-task-identity-roster-v1.json"), selection
    )
    assert len(reviewed["tasks"]) == 160
    assert len({row["eval_task_id"] for row in reviewed["tasks"]}) == 160
    frozen_by_key = {row["task_key"]: row["task_version_id"] for row in selection["tasks"]}
    assert sum(
        row["current_task_version_id"] == frozen_by_key[row["task_key"]]
        for row in reviewed["tasks"]
    ) == 146
    assert sum(
        row["current_task_version_id"] != frozen_by_key[row["task_key"]]
        for row in reviewed["tasks"]
    ) == 14
    roster = _roster(selection)
    path = tmp_path / "roster.json"
    raw = json.dumps(roster, separators=(",", ":")).encode()
    path.write_bytes(raw)
    monkeypatch.setattr(legacy_import, "EXPECTED_IDENTITY_ROSTER_SHA256", sha256(raw))
    assert load_identity_roster(path, selection) == roster

    invalid = copy.deepcopy(roster)
    invalid["tasks"][0]["eval_task_id"] = invalid["tasks"][1]["eval_task_id"]
    changed = tmp_path / "changed.json"
    changed_raw = json.dumps(invalid, separators=(",", ":")).encode()
    changed.write_bytes(changed_raw)
    monkeypatch.setattr(legacy_import, "EXPECTED_IDENTITY_ROSTER_SHA256", sha256(changed_raw))
    with pytest.raises(PlatformImportError, match="frozen 160-task cohort"):
        load_identity_roster(changed, selection)


def test_v3_request_matches_platform_905_contract() -> None:
    selection = load_selection(SELECTION)
    roster = _roster(selection)
    row = selection["tasks"][0]
    eval_task_id = roster["tasks"][0]["eval_task_id"]
    request = import_request(
        row,
        namespace="gentle-ember-ledger",
        repository="fleet-cyber-a62-frozen",
        split_index=1,
        expected_eval_task_id=eval_task_id,
    )
    assert request["expected_eval_task_id"] == eval_task_id
    assert request["expected_current_task_version_id"] == row["task_version_id"]
    assert "source_selection" not in request
    assert source_identity(row, eval_task_id) == {
        "task_key": row["task_key"],
        "expected_eval_task_id": eval_task_id,
        "expected_current_task_version_id": row["task_version_id"],
    }
    legacy = import_request(
        row,
        namespace="gentle-ember-ledger",
        repository="fleet-cyber-a62-frozen",
        split_index=1,
        protocol=LEGACY_IMPORT_PROTOCOL,
    )
    assert "expected_eval_task_id" not in legacy
    assert "expected_current_task_version_id" not in legacy


def test_v2_stays_diagnostics_only_and_held() -> None:
    selection = load_selection(SELECTION)
    roster = _roster(selection)
    plan, posts, reads = _plan(selection, roster, protocol=LEGACY_IMPORT_PROTOCOL, moved=1)
    assert len(reads) == 160
    assert posts == []
    assert plan["counts"] == {
        "total": 160,
        "blocked_current_differs_from_frozen": 1,
        "blocked_deployed_topology_omits_environment": 2,
        "eligible_current_equals_frozen": 157,
    }
    with pytest.raises(PlatformImportError, match="binds exact task versions"):
        require_submit_safe(plan)


def test_v3_models_905_current_pin_limit_and_holds_partial_cohort() -> None:
    selection = load_selection(SELECTION)
    roster = _roster(selection)
    plan, posts, reads = _plan(selection, roster, moved=14)
    assert len(reads) == 160
    assert posts == []
    assert plan["counts"] == {
        "total": 160,
        "blocked_frozen_version_not_current": 14,
        "eligible_exact_current_frozen": 146,
    }
    assert all(row["source_identity_sha256"] for row in plan["rows"])
    with pytest.raises(PlatformImportError, match="partial frozen-cohort"):
        require_submit_safe(plan)


def test_v3_fails_closed_on_reviewed_eval_task_identity_drift() -> None:
    selection = load_selection(SELECTION)
    roster = _roster(selection)
    fleet, registry, _posts, _reads = _clients(
        selection, roster, mismatched_eval_id=True
    )
    with fleet, registry:
        plan = build_plan(
            selection,
            identity_roster=roster,
            fleet_client=fleet,
            registry_client=registry,
            namespace="gentle-ember-ledger",
            repository="fleet-cyber-a62-frozen",
        )
    assert plan["counts"] == {
        "total": 160,
        "blocked_eval_task_identity_differs_from_reviewed": 1,
        "eligible_exact_current_frozen": 159,
    }
    with pytest.raises(PlatformImportError, match="partial frozen-cohort"):
        require_submit_safe(plan)


def test_v3_full_plan_can_submit_one_exact_create_once_canary() -> None:
    selection = load_selection(SELECTION)
    roster = _roster(selection)
    fleet, registry, posts, _reads = _clients(selection, roster)
    with fleet, registry:
        plan = build_plan(
            selection,
            identity_roster=roster,
            fleet_client=fleet,
            registry_client=registry,
            namespace="gentle-ember-ledger",
            repository="fleet-cyber-a62-frozen",
        )
        assert plan["counts"] == {"total": 160, "eligible_exact_current_frozen": 160}
        require_submit_safe(plan)
        submission = submit_rows(
            plan,
            selection,
            identity_roster=roster,
            registry_client=registry,
            limit=1,
        )
    assert len(posts) == 1
    assert posts[0]["expected_eval_task_id"] == roster["tasks"][0]["eval_task_id"]
    assert posts[0]["expected_current_task_version_id"] == selection["tasks"][0][
        "task_version_id"
    ]
    assert submission["receipts"][0]["request_sha256"] == plan["rows"][0][
        "request_sha256"
    ]


def test_create_receipt_rejects_protocol_source_pin_drift() -> None:
    selection = load_selection(SELECTION)
    roster = _roster(selection)
    fleet, registry, _posts, _reads = _clients(
        selection, roster, response_pin_drift=True
    )
    with fleet, registry:
        plan = build_plan(
            selection,
            identity_roster=roster,
            fleet_client=fleet,
            registry_client=registry,
            namespace="gentle-ember-ledger",
            repository="fleet-cyber-a62-frozen",
        )
        with pytest.raises(PlatformImportError, match="mismatched receipt identity"):
            submit_rows(
                plan,
                selection,
                identity_roster=roster,
                registry_client=registry,
                limit=1,
            )


def test_resealed_plan_cannot_change_reviewed_eval_task_id() -> None:
    selection = load_selection(SELECTION)
    roster = _roster(selection)
    plan, _posts, _reads = _plan(selection, roster)
    tampered = copy.deepcopy(plan)
    tampered["rows"][0]["expected_eval_task_id"] = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
    tampered["rows"][0]["source_identity_sha256"] = sha256(
        legacy_import.canonical_json(
            source_identity(selection["tasks"][0], tampered["rows"][0]["expected_eval_task_id"])
        )
    )
    tampered["plan_sha256"] = digest_without(tampered, "plan_sha256")
    with pytest.raises(PlatformImportError, match="plan row drifted"):
        validate_plan(tampered, selection, roster)


def test_journal_resumes_sanitized_receipts(tmp_path: Path) -> None:
    selection = load_selection(SELECTION)
    roster = _roster(selection)
    plan, _posts, _reads = _plan(selection, roster)
    receipt = _receipt_for_plan(plan, state="extracting")
    journal = tmp_path / "receipts.jsonl"
    with receipt_journal(journal) as (_prior, append):
        append(receipt)
    with receipt_journal(journal) as (prior, _append):
        assert prior == [receipt]
    assert len(journal.read_text().splitlines()) == 1


def test_status_monitor_binds_protocol_and_source_pins_without_private_data() -> None:
    selection = load_selection(SELECTION)
    roster = _roster(selection)
    plan, _posts, _reads = _plan(selection, roster)
    receipt = _receipt_for_plan(plan)

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "import_id": receipt["import_id"],
                "state": "extracting",
                "version": 2,
                "task_key": receipt["task_key"],
                "source_team_id": FLEET_TEAM_ID,
                "protocol_version": receipt["protocol_version"],
                "expected_eval_task_id": receipt["expected_eval_task_id"],
                "expected_current_task_version_id": receipt[
                    "expected_current_task_version_id"
                ],
                "destination": receipt["destination"],
                "private_task_content": "must-not-be-copied",
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        status = monitor_receipts([receipt], registry_client=client)
    assert status["rows"] == [
        {
            "import_id": receipt["import_id"],
            "destination_tag": receipt["destination"]["tag"],
            "protocol_version": EXPECTED_IMPORT_PROTOCOL,
            "expected_eval_task_id": receipt["expected_eval_task_id"],
            "expected_current_task_version_id": receipt["expected_current_task_version_id"],
            "source_identity_sha256": receipt["source_identity_sha256"],
            "state": "extracting",
            "version": 2,
            "error_code": None,
        }
    ]
    assert "private_task_content" not in json.dumps(status)
    summary = submission_summary(
        {"created": True},
        {
            "submitted": 1,
            "submitted_new": 0,
            "resumed_from_journal": 1,
            "receipts": [receipt],
        },
        plan,
    )
    assert receipt["task_key"] not in json.dumps(summary)


def test_status_monitor_rejects_source_pin_drift() -> None:
    selection = load_selection(SELECTION)
    roster = _roster(selection)
    plan, _posts, _reads = _plan(selection, roster)
    receipt = _receipt_for_plan(plan)

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "import_id": receipt["import_id"],
                "state": "published",
                "version": 2,
                "task_key": receipt["task_key"],
                "source_team_id": FLEET_TEAM_ID,
                "protocol_version": receipt["protocol_version"],
                "expected_eval_task_id": "dddddddd-dddd-4ddd-8ddd-dddddddddddd",
                "expected_current_task_version_id": receipt[
                    "expected_current_task_version_id"
                ],
                "destination": receipt["destination"],
            },
        )

    with (
        httpx.Client(transport=httpx.MockTransport(handler)) as client,
        pytest.raises(PlatformImportError, match="identity drifted"),
    ):
        monitor_receipts([receipt], registry_client=client)


def test_forbidden_errors_and_journal_failures_are_sanitized(tmp_path: Path) -> None:
    selection = load_selection(SELECTION)
    roster = _roster(selection)
    fleet, registry, _posts, _reads = _clients(selection, roster, forbidden=True)
    with fleet, registry:
        plan = build_plan(
            selection,
            identity_roster=roster,
            fleet_client=fleet,
            registry_client=registry,
            namespace="gentle-ember-ledger",
            repository="fleet-cyber-a62-frozen",
        )
        with pytest.raises(PlatformImportError, match="destination write access") as error:
            submit_rows(
                plan,
                selection,
                identity_roster=roster,
                registry_client=registry,
                limit=1,
            )
    assert "must-not-surface" not in str(error.value)

    corrupt = tmp_path / "corrupt.jsonl"
    corrupt.write_text("{truncated")
    corrupt.chmod(0o600)
    with pytest.raises(PlatformImportError, match="invalid JSON"), receipt_journal(corrupt):
        pass
    public = tmp_path / "public.jsonl"
    public.write_text("")
    public.chmod(0o644)
    with pytest.raises(PlatformImportError, match="private regular file"), receipt_journal(public):
        pass


def test_repository_and_operator_outputs_fail_closed(tmp_path: Path) -> None:
    good = {
        "name": {"namespace": "gentle-ember-ledger", "name": "fleet-cyber-a62-frozen"},
        "kind": "taskset",
        "visibility": "private",
        "mutable_tag_patterns": ["latest"],
    }
    with httpx.Client(
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, json=good))
    ) as client:
        assert not ensure_repository(
            client,
            namespace="gentle-ember-ledger",
            repository="fleet-cyber-a62-frozen",
        )["created"]
    wrong = {**good, "visibility": "public"}
    with (
        httpx.Client(
            transport=httpx.MockTransport(lambda _request: httpx.Response(200, json=wrong))
        ) as client,
        pytest.raises(PlatformImportError, match="configuration drifted"),
    ):
        ensure_repository(
            client,
            namespace="gentle-ember-ledger",
            repository="fleet-cyber-a62-frozen",
        )

    output = tmp_path / "plan.json"
    write_json_create_once(output, {"safe": True})
    write_json_create_once(output, {"safe": True})
    with pytest.raises(PlatformImportError, match="different bytes"):
        write_json_create_once(output, {"safe": False})
