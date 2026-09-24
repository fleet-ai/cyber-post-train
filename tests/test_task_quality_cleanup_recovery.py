from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from evals.fleet import opencode_self_hosted as self_hosted
from evals.fleet import task_quality_cleanup_recovery as recovery
from evals.fleet import task_quality_qualification as qualification

TASK_VERSION = "11111111-1111-4111-8111-111111111111"
ENV_VERSION = "22222222-2222-4222-8222-222222222222"


def _binding() -> dict:
    body = {
        "task_key": "task-a",
        "task_version_id": TASK_VERSION,
        "qa_status": "clean",
        "environment": {
            "id": "cysec1-2-fixture",
            "version": "v0.0.1",
            "version_id": ENV_VERSION,
        },
        "verifier": {"verifier_version_id": "verifier-a"},
    }
    return {**body, "binding_sha256": qualification.digest(body)}


def _source() -> dict:
    return {
        "git_commit": "a" * 40,
        "git_tree": "b" * 40,
        "origin_main_commit": "a" * 40,
        "merged_to_origin_main": True,
        "controller_path": "evals/fleet/task_quality_qualification.py",
        "controller_file_sha256": "sha256:" + "c" * 64,
    }


def _operation_root(tmp_path: Path) -> tuple[Path, dict, dict]:
    root = tmp_path / "operation"
    root.mkdir(mode=0o700)
    binding = _binding()
    plan = qualification.sealed(
        {
            "schema": qualification.PLAN_SCHEMA,
            "wave_id": "wave-a",
            "source": _source(),
            "inputs": {"inventory_sha256": "sha256:" + "1" * 64},
            "selection": {},
            "execution": {"concurrency": 1, "external_mutations_authorized": True},
            "tasks": [binding],
        }
    )
    qualification._write_once(root / "PLAN.json", plan)  # noqa: SLF001
    qualification._write_once(  # noqa: SLF001
        root / "RUN_INTENT.json",
        qualification.sealed(
            {
                "schema": qualification.RUN_INTENT_SCHEMA,
                "plan_sha256": plan["sha256"],
                "wave_id": plan["wave_id"],
                "task_versions": 1,
                "concurrency": 1,
                "one_shot": True,
            }
        ),
    )
    qualification._write_once(  # noqa: SLF001
        root / "AGGREGATE_RECEIPT.json",
        qualification.sealed(
            {
                "schema": qualification.AGGREGATE_SCHEMA,
                "plan_sha256": plan["sha256"],
                "planned_task_versions": 1,
                "model_calls": 0,
            }
        ),
    )
    directory = root / "cells/cell-000"
    config = qualification._config(binding, plan["wave_id"])  # noqa: SLF001
    request_id = self_hosted.provisioning_request_id(config)
    qualification._write_once(  # noqa: SLF001
        directory / "CELL_INTENT.json",
        qualification.sealed(
            {
                "schema": "cyber_task_quality_qualification_cell_intent_v1",
                "binding_sha256": binding["binding_sha256"],
                "wave_id": plan["wave_id"],
                "run_id": config["run_id"],
            }
        ),
    )
    qualification._write_once(  # noqa: SLF001
        directory / "PROVISION_INTENT.json",
        qualification.sealed(
            {
                "schema": "cyber_task_quality_provision_intent_v1",
                "run_id": config["run_id"],
                "task_version_id": binding["task_version_id"],
                "request_id": request_id,
            }
        ),
    )
    qualification._write_once(  # noqa: SLF001
        directory / "CELL_TERMINAL.json",
        qualification.sealed(
            {
                "schema": qualification.CELL_TERMINAL_SCHEMA,
                "binding_sha256": binding["binding_sha256"],
                "wave_id": plan["wave_id"],
                "qualification_status": "infrastructure_invalid",
            }
        ),
    )
    return root, plan, binding


def _install_source_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    current = {
        "git_commit": "d" * 40,
        "git_tree": "e" * 40,
        "origin_main_commit": "d" * 40,
        "canonical_remote": recovery.CANONICAL_REMOTE,
        "module_path": "evals/fleet/task_quality_cleanup_recovery.py",
        "module_file_sha256": "sha256:" + "f" * 64,
    }
    monkeypatch.setattr(recovery, "recovery_source_provenance", lambda: current)
    monkeypatch.setattr(
        recovery,
        "verify_historical_source",
        lambda source, *, current_commit: {
            key: source[key]
            for key in ("git_commit", "git_tree", "controller_path", "controller_file_sha256")
        },
    )


def _install_client(
    monkeypatch: pytest.MonkeyPatch,
    *,
    claim_status: int = 404,
    list_responses: list[list[dict]] | None = None,
) -> list[str]:
    methods: list[str] = []
    responses = iter(list_responses or [[], []])

    def handler(request: httpx.Request) -> httpx.Response:
        methods.append(request.method)
        if request.url.path == "/v1/account":
            return httpx.Response(
                200,
                json={"team_name": "fleet", "team_id": qualification.EXPECTED_TEAM_ID},
            )
        if request.url.path == "/openapi.json":
            return httpx.Response(
                200,
                json={
                    "paths": {
                        qualification.CREATE_CLAIM_ROUTE_TEMPLATE: {
                            "get": {},
                            "delete": {},
                        }
                    }
                },
            )
        if request.url.path.startswith("/v1/env/instances/create-requests/"):
            return httpx.Response(claim_status, json={"state": "materialized"})
        if request.url.path == "/v1/env/instances":
            return httpx.Response(200, json=next(responses))
        raise AssertionError(request.url)

    monkeypatch.setattr(
        qualification,
        "_client",
        lambda _api_key: httpx.Client(
            transport=httpx.MockTransport(handler),
            headers={"Authorization": "Bearer fixture"},
        ),
    )
    return methods


def test_recovery_seals_two_empty_reads_and_never_mutates(tmp_path, monkeypatch):
    root, plan, binding = _operation_root(tmp_path)
    _install_source_stubs(monkeypatch)
    methods = _install_client(monkeypatch, list_responses=[[], [], [], []])

    result = recovery.recover(root, api_key="fixture", settle_seconds=0)

    assert result["resolution_receipts"] == 1
    assert result["all_claims_absent"] is True
    assert result["all_exact_run_instance_lists_empty"] is True
    assert result["external_mutations"] == 0
    assert set(methods) == {"GET"}
    directory = root / "cells/cell-000"
    evidence = json.loads((directory / recovery.RECOVERY_EVIDENCE_FILE).read_text())
    assert evidence["plan_sha256"] == plan["sha256"]
    assert evidence["binding_sha256"] == binding["binding_sha256"]
    assert len(evidence["observations"]) == 2
    for observation in evidence["observations"]:
        assert observation["claim_http_status"] == 404
        assert observation["exact_run_instance_count"] == 0
        observed_at = datetime.fromisoformat(observation["observed_at_utc"])
        assert observed_at.tzinfo is not None
        assert observed_at.utcoffset() == UTC.utcoffset(observed_at)
    resolution = json.loads((directory / "CLEANUP_RESOLUTION.json").read_text())
    assert resolution["resolution"] == recovery.RECOVERY_RESOLUTION
    assert resolution["instance_id"] is None
    assert resolution["independent_absence_evidence_sha256"] == evidence["sha256"]
    monkeypatch.setattr(qualification, "_source_matches_plan", lambda _source, **_kwargs: None)
    closed = qualification.cleanup_plan(plan, root, api_key="fixture")
    assert closed["resolved_task_versions"] == 1
    assert closed["unresolved_task_versions"] == 0
    assert recovery.recover(root, api_key="fixture", settle_seconds=0) == result


def test_recovery_rejects_existing_claim_without_writing_resolution(tmp_path, monkeypatch):
    root, _, _ = _operation_root(tmp_path)
    _install_source_stubs(monkeypatch)
    methods = _install_client(monkeypatch, claim_status=200)

    with pytest.raises(recovery.RecoveryError, match="claim is not absent"):
        recovery.recover(root, api_key="fixture", settle_seconds=0)

    assert not (root / "cells/cell-000/CLEANUP_RESOLUTION.json").exists()
    assert "DELETE" not in methods


def test_recovery_rejects_nonempty_or_racing_exact_run_list(tmp_path, monkeypatch):
    root, _, _ = _operation_root(tmp_path)
    _install_source_stubs(monkeypatch)
    methods = _install_client(monkeypatch, list_responses=[[], [{"instance_id": "peer"}]])

    with pytest.raises(recovery.RecoveryError, match="listing is not empty"):
        recovery.recover(root, api_key="fixture", settle_seconds=0)

    assert not (root / "cells/cell-000/CLEANUP_RESOLUTION.json").exists()
    assert "DELETE" not in methods


def test_recovery_rejects_run_intent_mismatch_before_network(tmp_path, monkeypatch):
    root, _, _ = _operation_root(tmp_path)
    intent_path = root / "RUN_INTENT.json"
    intent = json.loads(intent_path.read_text())
    intent["task_versions"] = 2
    intent["sha256"] = qualification.digest(
        {key: value for key, value in intent.items() if key != "sha256"}
    )
    intent_path.write_text(json.dumps(intent))
    _install_source_stubs(monkeypatch)
    methods = _install_client(monkeypatch)

    with pytest.raises(recovery.RecoveryError, match="run intent differs"):
        recovery.recover(root, api_key="fixture", settle_seconds=0)

    assert methods == []


@pytest.mark.parametrize("artifact", ["CELL_INTENT.json", "CELL_TERMINAL.json"])
def test_recovery_rejects_cross_wave_cell_evidence_before_network(tmp_path, monkeypatch, artifact):
    root, _, _ = _operation_root(tmp_path)
    path = root / "cells/cell-000" / artifact
    value = json.loads(path.read_text())
    value["wave_id"] = "other-wave"
    value["sha256"] = qualification.digest(
        {key: item for key, item in value.items() if key != "sha256"}
    )
    path.write_text(json.dumps(value))
    _install_source_stubs(monkeypatch)
    methods = _install_client(monkeypatch)

    with pytest.raises(recovery.RecoveryError, match="frozen binding"):
        recovery.recover(root, api_key="fixture", settle_seconds=0)

    assert methods == []


def test_recovery_source_requires_exact_freshly_fetched_main(monkeypatch):
    monkeypatch.setattr(recovery, "_fresh_canonical_main", lambda _root: "b" * 40)

    def fake_git(_root, *args):
        if args == ("status", "--porcelain", "--untracked-files=no"):
            return ""
        if args == ("rev-parse", "HEAD"):
            return "a" * 40
        raise AssertionError(args)

    monkeypatch.setattr(recovery, "_git", fake_git)
    with pytest.raises(recovery.RecoveryError, match="freshly fetched origin/main"):
        recovery.recovery_source_provenance()


def test_canonical_main_observation_fetches_exact_remote_branch(monkeypatch, tmp_path):
    calls = []

    def fake_git(_root, *args):
        if args == ("remote", "get-url", "origin"):
            return recovery.CANONICAL_REMOTE
        if args == ("rev-parse", "FETCH_HEAD"):
            return "a" * 40
        raise AssertionError(args)

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        return object()

    monkeypatch.setattr(recovery, "_git", fake_git)
    monkeypatch.setattr(recovery.subprocess, "run", fake_run)

    assert recovery._fresh_canonical_main(tmp_path) == "a" * 40
    assert calls[0][0] == [
        "git",
        "-C",
        str(tmp_path),
        "fetch",
        "--quiet",
        "--no-tags",
        "origin",
        "main",
    ]
    assert calls[0][1]["check"] is True
