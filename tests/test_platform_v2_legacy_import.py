from __future__ import annotations

import copy
import json
from pathlib import Path

import httpx
import pytest

from evals.platform_v2.legacy_import import (
    EXPECTED_IMPORT_PROTOCOL,
    FLEET_TEAM_ID,
    PlatformImportError,
    build_plan,
    digest_without,
    ensure_repository,
    import_request,
    load_selection,
    monitor_receipts,
    receipt_journal,
    submission_summary,
    submit_rows,
    validate_plan,
    write_json_create_once,
)

SELECTION = Path("configs/data/fleet-a62-task-split-v1.json")


def test_frozen_selection_verifies() -> None:
    selection = load_selection(SELECTION)
    assert len(selection["tasks"]) == 160
    assert {row["resolution_authority"] for row in selection["tasks"]} == {
        "training_task_catalog",
        "fleet_job_roster+training_environment_catalog",
    }


def test_request_is_exact_and_stable() -> None:
    row = load_selection(SELECTION)["tasks"][0]
    request = import_request(
        row, namespace="gentle-ember-ledger", repository="fleet-cyber-a62-frozen", split_index=1
    )
    assert request["destination"]["tag"].endswith(row["task_version_id"][:8])
    assert request["modality_decision"]["modality"] == "tool_use"
    assert "task_version_id" not in request


def test_plan_blocks_mutable_version_drift_and_omitted_topology_before_submit() -> None:
    selection = load_selection(SELECTION)
    expected = {row["task_key"]: row["task_version_id"] for row in selection["tasks"]}

    def fleet_handler(request: httpx.Request) -> httpx.Response:
        task_key = request.url.path.rsplit("/", 1)[-1]
        version = expected[task_key]
        if task_key == selection["tasks"][1]["task_key"]:
            version = "11111111-1111-4111-8111-111111111111"
        return httpx.Response(200, json={"eval_task_version_id": version})

    def registry_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/.well-known/fleet-registry":
            return httpx.Response(
                200,
                json={
                    "legacy_task_imports": {
                        "base_path": "/api/v1/imports/legacy-tasks",
                        "protocol": EXPECTED_IMPORT_PROTOCOL,
                    }
                },
            )
        raise AssertionError("a held plan must not submit imports")

    with (
        httpx.Client(transport=httpx.MockTransport(fleet_handler)) as fleet,
        httpx.Client(transport=httpx.MockTransport(registry_handler)) as registry,
    ):
        plan = build_plan(
            selection,
            fleet_client=fleet,
            registry_client=registry,
            namespace="gentle-ember-ledger",
            repository="fleet-cyber-a62-frozen",
        )
        with pytest.raises(PlatformImportError, match="binds exact task versions"):
            submit_rows(
                plan,
                selection,
                fleet_client=fleet,
                registry_client=registry,
                limit=None,
            )

    assert plan["counts"] == {
        "total": 160,
        "blocked_current_differs_from_frozen": 1,
        "blocked_deployed_topology_omits_environment": 2,
        "eligible_current_equals_frozen": 157,
    }
    omitted = [
        row
        for row in plan["rows"]
        if row["disposition"] == "blocked_deployed_topology_omits_environment"
    ]
    assert len(omitted) == 2


def _clients(
    selection: dict, *, move_after_plan: bool = False, forbidden: bool = False
) -> tuple[httpx.Client, httpx.Client, list[dict[str, object]]]:
    expected = {row["task_key"]: row["task_version_id"] for row in selection["tasks"]}
    reads = 0
    posts: list[dict[str, object]] = []

    def fleet_handler(request: httpx.Request) -> httpx.Response:
        nonlocal reads
        reads += 1
        task_key = request.url.path.rsplit("/", 1)[-1]
        version = expected[task_key]
        if move_after_plan and reads > len(selection["tasks"]):
            version = "11111111-1111-4111-8111-111111111111"
        return httpx.Response(200, json={"task_key": task_key, "eval_task_version_id": version})

    def registry_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/.well-known/fleet-registry":
            return httpx.Response(
                200,
                json={
                    "legacy_task_imports": {
                        "base_path": "/api/v1/imports/legacy-tasks",
                        "protocol": EXPECTED_IMPORT_PROTOCOL,
                    }
                },
            )
        payload = json.loads(request.content)
        posts.append(payload)
        if forbidden:
            return httpx.Response(403, json={"secret": "must-not-surface"})
        return httpx.Response(
            202,
            json={
                "import_id": f"imp_{len(posts):03d}",
                "state": "requested",
                "task_key": payload["task_key"],
                "source_team_id": FLEET_TEAM_ID,
                "destination": payload["destination"],
                "modality_decision": payload["modality_decision"],
            },
        )

    return (
        httpx.Client(transport=httpx.MockTransport(fleet_handler)),
        httpx.Client(transport=httpx.MockTransport(registry_handler)),
        posts,
    )


def _receipt_for_plan(plan: dict, *, state: str = "requested") -> dict:
    row = next(
        row for row in plan["rows"] if row["disposition"] == "eligible_current_equals_frozen"
    )
    receipt = {
        "schema": "fleet_platform_v2_cyber_import_row_receipt_v1",
        "plan_sha256": plan["plan_sha256"],
        "request_sha256": row["request_sha256"],
        "task_key": row["task_key"],
        "task_version_id": row["frozen_task_version_id"],
        "destination": {**plan["destination"], "tag": row["destination_tag"]},
        "import_id": "lti_resume_example",
        "state": state,
        "idempotent_replay": False,
    }
    receipt["receipt_sha256"] = digest_without(receipt, "receipt_sha256")
    return receipt


def test_submit_is_held_before_version_reads_or_posts() -> None:
    selection = load_selection(SELECTION)
    fleet, registry, posts = _clients(selection, move_after_plan=True)
    with fleet, registry:
        plan = build_plan(
            selection,
            fleet_client=fleet,
            registry_client=registry,
            namespace="gentle-ember-ledger",
            repository="fleet-cyber-a62-frozen",
        )
        with pytest.raises(PlatformImportError, match="binds exact task versions"):
            submit_rows(
                plan,
                selection,
                fleet_client=fleet,
                registry_client=registry,
                limit=1,
            )
    assert posts == []


def test_resealed_plan_cannot_promote_blocked_or_change_destination() -> None:
    selection = load_selection(SELECTION)
    fleet, registry, _posts = _clients(selection)
    with fleet, registry:
        plan = build_plan(
            selection,
            fleet_client=fleet,
            registry_client=registry,
            namespace="gentle-ember-ledger",
            repository="fleet-cyber-a62-frozen",
        )
    tampered = copy.deepcopy(plan)
    blocked = next(
        row
        for row in tampered["rows"]
        if row["disposition"] == "blocked_deployed_topology_omits_environment"
    )
    blocked["disposition"] = "eligible_current_equals_frozen"
    tampered["plan_sha256"] = digest_without(tampered, "plan_sha256")
    with pytest.raises(PlatformImportError, match="plan row drifted"):
        validate_plan(tampered, selection)

    tampered = copy.deepcopy(plan)
    tampered["destination"]["repository"] = "other"
    tampered["plan_sha256"] = digest_without(tampered, "plan_sha256")
    with pytest.raises(PlatformImportError, match="frozen import campaign"):
        validate_plan(tampered, selection)


def test_journal_resumes_sanitized_receipts_without_reposting(tmp_path: Path) -> None:
    selection = load_selection(SELECTION)
    fleet, registry, _posts = _clients(selection)
    journal = tmp_path / "receipts.jsonl"
    with fleet, registry:
        plan = build_plan(
            selection,
            fleet_client=fleet,
            registry_client=registry,
            namespace="gentle-ember-ledger",
            repository="fleet-cyber-a62-frozen",
        )
        receipt = _receipt_for_plan(plan, state="extracting")
        with receipt_journal(journal) as (_prior, append):
            append(receipt)
        with receipt_journal(journal) as (prior, append):
            assert prior == [receipt]
            assert append is not None
    assert len(journal.read_text().splitlines()) == 1

    missing = tmp_path / "missing.jsonl"
    with (
        pytest.raises(PlatformImportError, match="cannot be opened safely"),
        receipt_journal(missing, create=False),
    ):
        pass


def test_import_forbidden_is_sanitized_and_journal_stays_empty(tmp_path: Path) -> None:
    selection = load_selection(SELECTION)
    fleet, registry, _posts = _clients(selection)
    journal = tmp_path / "receipts.jsonl"
    with fleet, registry:
        plan = build_plan(
            selection,
            fleet_client=fleet,
            registry_client=registry,
            namespace="gentle-ember-ledger",
            repository="fleet-cyber-a62-frozen",
        )
        receipt = _receipt_for_plan(plan)
        with receipt_journal(journal) as (_prior, append):
            append(receipt)

    def forbidden_status(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"secret": "must-not-surface"})

    with (
        httpx.Client(transport=httpx.MockTransport(forbidden_status)) as status_client,
        pytest.raises(PlatformImportError, match="HTTP 403") as error,
    ):
        monitor_receipts([receipt], registry_client=status_client)
    assert "must-not-surface" not in str(error.value)
    assert len(journal.read_text().splitlines()) == 1


def test_status_monitor_emits_only_sanitized_ids_tags_and_state(tmp_path: Path) -> None:
    selection = load_selection(SELECTION)
    fleet, registry, _posts = _clients(selection)
    journal = tmp_path / "receipts.jsonl"
    with fleet, registry:
        plan = build_plan(
            selection,
            fleet_client=fleet,
            registry_client=registry,
            namespace="gentle-ember-ledger",
            repository="fleet-cyber-a62-frozen",
        )
        receipt = _receipt_for_plan(plan)
        with receipt_journal(journal) as (_prior, append):
            append(receipt)

    def status_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "import_id": receipt["import_id"],
                "state": "extracting",
                "version": 2,
                "task_key": receipt["task_key"],
                "source_team_id": FLEET_TEAM_ID,
                "destination": receipt["destination"],
                "private_task_content": "must-not-be-copied",
            },
        )

    with httpx.Client(transport=httpx.MockTransport(status_handler)) as status_client:
        status = monitor_receipts([receipt], registry_client=status_client)
    assert status["rows"] == [
        {
            "import_id": receipt["import_id"],
            "destination_tag": receipt["destination"]["tag"],
            "state": "extracting",
            "version": 2,
            "error_code": None,
        }
    ]
    assert "private_task_content" not in json.dumps(status)
    submission = {
        "submitted": 1,
        "submitted_new": 0,
        "resumed_from_journal": 1,
        "receipts": [receipt],
    }
    summary = submission_summary({"created": True}, submission, plan)
    assert summary["receipts"] == [
        {
            "import_id": receipt["import_id"],
            "destination_tag": receipt["destination"]["tag"],
            "state": "requested",
            "idempotent_replay": False,
        }
    ]
    assert receipt["task_key"] not in json.dumps(summary)


def test_status_monitor_fails_closed_on_relocated_or_malformed_import() -> None:
    receipt = {
        "schema": "fleet_platform_v2_cyber_import_row_receipt_v1",
        "plan_sha256": "sha256:" + "1" * 64,
        "request_sha256": "sha256:" + "2" * 64,
        "task_key": "task-one",
        "task_version_id": "11111111-1111-4111-8111-111111111111",
        "destination": {
            "namespace": "gentle-ember-ledger",
            "repository": "fleet-cyber-a62-frozen",
            "tag": "train-001-11111111",
        },
        "import_id": "imp_one",
        "state": "requested",
        "idempotent_replay": False,
    }
    receipt["receipt_sha256"] = digest_without(receipt, "receipt_sha256")

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "import_id": "imp_other",
                "state": "published",
                "version": 2,
                "task_key": "task-one",
                "source_team_id": FLEET_TEAM_ID,
                "destination": receipt["destination"],
                "private": "must-not-surface",
            },
        )

    with (
        httpx.Client(transport=httpx.MockTransport(handler)) as client,
        pytest.raises(PlatformImportError, match="identity drifted") as error,
    ):
        monitor_receipts([receipt], registry_client=client)
    assert "must-not-surface" not in str(error.value)


def test_corrupt_or_public_journal_fails_closed(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.jsonl"
    corrupt.write_text("{truncated")
    corrupt.chmod(0o600)
    with (
        pytest.raises(PlatformImportError, match="invalid JSON"),
        receipt_journal(corrupt),
    ):
        pass

    public = tmp_path / "public.jsonl"
    public.write_text("")
    public.chmod(0o644)
    with (
        pytest.raises(PlatformImportError, match="private regular file"),
        receipt_journal(public),
    ):
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
        assert (
            ensure_repository(
                client,
                namespace="gentle-ember-ledger",
                repository="fleet-cyber-a62-frozen",
            )["created"]
            is False
        )
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
