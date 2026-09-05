from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse
from urllib.request import Request

import pytest

from evals.fleet import exact_pass4_task_inventory as inventory

ROOT = Path(__file__).resolve().parents[1]
LEGACY_GAP = {
    1: "90338ce0-194b-4cea-bef4-d64e5d426319",
    2: "09a3fea6-f691-4841-9218-d04459041a1f",
    3: "33d37078-0669-478e-af39-43cd245f0da8",
    5: "037704a7-fc03-4cf6-a8da-3f3e3933717e",
}
JOB_UID = "20000000-0000-4000-8000-000000000001"
POD_UID = "30000000-0000-4000-8000-000000000001"


def _expected() -> dict:
    return inventory.prepare_expected(ROOT)


def _live_task(row: dict, *, sentinel: str = "private-task-content") -> dict:
    rank = row["selection_rank"]
    return {
        "key": row["task_key"],
        "version_id": row["task_version_id"],
        "task_version": row["task_version"],
        "environment_id": row["env_key"],
        "version": row["env_version"],
        "data_id": row["data_key"],
        "data_version": row["data_version"],
        "prompt": f"{sentinel}-prompt-{rank}",
        "env_variables": {"secret": f"{sentinel}-env-{rank}"},
        "output_json_schema": {"private": f"{sentinel}-schema-{rank}"},
        "metadata": {
            "cyber_contract": {
                "evidence_schema": "1.0.0",
                "submission_protocol": "2.0.0",
                "verifier_contract": "3.0.0",
            },
            "runtime_seed_manifest": {
                "content_sha256": f"{rank:064x}",
                "files": [
                    {
                        "target_path": f"/task/{sentinel}-{rank}",
                        "s3_key": f"private/{sentinel}-{rank}",
                        "bucket": "private-bucket",
                    }
                ],
            },
        },
        "verifier_id": f"00000000-0000-4000-8000-{rank:012d}",
        "verifier": {
            "verifier_version_id": f"10000000-0000-4000-8000-{rank:012d}",
            "version": 1,
            "sha256": f"{rank + 100:064x}",
            "function_name": "verify",
            "code": f"{sentinel}-verifier-{rank}",
        },
    }


def _install_fake_gets(monkeypatch: pytest.MonkeyPatch, expected: dict) -> list[str]:
    rows = {row["task_version_id"]: row for row in expected["tasks"]}
    calls: list[str] = []

    def fake_fetch(url: str, api_key: str) -> dict:
        assert api_key == "test-key"
        calls.append(url)
        parsed = urlparse(url)
        if parsed.path == "/v1/account":
            return {"team_name": "fleet", "team_id": inventory.FLEET_TEAM_ID}
        assert parsed.path.startswith("/v1/tasks/")
        version_ids = parse_qs(parsed.query).get("version_id")
        assert version_ids and len(version_ids) == 1
        return _live_task(rows[version_ids[0]])

    monkeypatch.setattr(inventory, "fetch_json", fake_fetch)
    return calls


def test_expected_inventory_is_exact_shared_100_and_includes_legacy_gap() -> None:
    expected = _expected()
    assert expected["task_count"] == 100
    assert expected["total_cell_count"] == 800
    assert expected["cell_counts"] == {"qwen3.8-27b": 400, "glm-5.3": 400}
    assert [row["selection_rank"] for row in expected["tasks"]] == list(range(1, 101))
    by_rank = {row["selection_rank"]: row["task_version_id"] for row in expected["tasks"]}
    assert {rank: by_rank[rank] for rank in LEGACY_GAP} == LEGACY_GAP
    assert expected["expected_sha256"] == inventory.digest_without(
        expected, "expected_sha256"
    )


def test_observe_uses_only_exact_gets_and_emits_sanitized_self_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = _expected()
    calls = _install_fake_gets(monkeypatch, expected)
    receipt = inventory.observe(
        expected,
        "test-key",
        root=ROOT,
        job_uid=JOB_UID,
        pod_uid=POD_UID,
        observed_at="2026-09-05T00:00:00Z",
    )
    assert len(calls) == 101
    assert calls[0].endswith("/v1/account")
    assert all("version_id=" in url for url in calls[1:])
    assert receipt["status"] == "PASSED"
    assert len(receipt["tasks"]) == 100
    assert receipt["request_counts"] == {
        "account_get": 1,
        "exact_task_version_get": 100,
        "redirects_followed": 0,
        "post_put_patch_delete": 0,
        "model_or_scoring_calls": 0,
        "session_calls": 0,
    }
    assert receipt["receipt_sha256"] == inventory.digest_without(receipt, "receipt_sha256")
    serialized = json.dumps(receipt)
    assert "private-task-content" not in serialized
    assert "private-bucket" not in serialized
    assert '"code"' not in serialized


def test_team_mismatch_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        inventory,
        "fetch_json",
        lambda _url, _key: {"team_name": "other", "team_id": "other"},
    )
    with pytest.raises(inventory.GateError, match="fleet_team_identity_mismatch"):
        inventory.observe(
            _expected(), "test-key", root=ROOT, job_uid=JOB_UID, pod_uid=POD_UID
        )


def test_redirects_are_rejected_before_authorization_can_be_forwarded() -> None:
    request = Request(
        "https://orchestrator.fleetai.com/v1/account",
        headers={"Authorization": "Bearer private"},
    )
    with pytest.raises(inventory.GateError, match="fleet_get_redirect_rejected"):
        inventory._RejectRedirects().redirect_request(
            request,
            None,
            302,
            "Found",
            {},
            "https://attacker.invalid/collect",
        )


def test_model_binding_mismatch_fails_before_network(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = _expected()
    expected["models"]["qwen3.8-27b"]["revision"] = "main"
    expected["expected_sha256"] = inventory.digest_without(expected, "expected_sha256")
    monkeypatch.setattr(
        inventory,
        "fetch_json",
        lambda *_args, **_kwargs: pytest.fail("network must not be called"),
    )
    with pytest.raises(inventory.GateError, match="model_binding_mismatch"):
        inventory.observe(
            expected, "test-key", root=ROOT, job_uid=JOB_UID, pod_uid=POD_UID
        )


@pytest.mark.parametrize("field", ["version_id", "task_version"])
def test_task_version_mismatch_fails_closed(
    monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    expected = _expected()
    first = expected["tasks"][0]
    calls = 0

    def fake_fetch(url: str, _key: str) -> dict:
        nonlocal calls
        calls += 1
        if url.endswith("/v1/account"):
            return {"team_name": "fleet", "team_id": inventory.FLEET_TEAM_ID}
        task = _live_task(first)
        task[field] = "wrong"
        return task

    monkeypatch.setattr(inventory, "fetch_json", fake_fetch)
    with pytest.raises(inventory.GateError, match="live_task_version_mismatch"):
        inventory.observe(
            expected, "test-key", root=ROOT, job_uid=JOB_UID, pod_uid=POD_UID
        )
    assert calls == 2


def test_task_binding_mismatch_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = _expected()
    first = expected["tasks"][0]

    def fake_fetch(url: str, _key: str) -> dict:
        if url.endswith("/v1/account"):
            return {"team_name": "fleet", "team_id": inventory.FLEET_TEAM_ID}
        task = _live_task(first)
        task["data_version"] = "mutable-latest"
        return task

    monkeypatch.setattr(inventory, "fetch_json", fake_fetch)
    with pytest.raises(inventory.GateError, match="live_task_binding_mismatch"):
        inventory.observe(
            expected, "test-key", root=ROOT, job_uid=JOB_UID, pod_uid=POD_UID
        )


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        (
            lambda task: task["metadata"].update({"runtime_seed_manifest": {"files": []}}),
            "runtime_seed_manifest_incomplete",
        ),
        (lambda task: task.update({"verifier": {}}), "verifier_binding_incomplete"),
    ],
)
def test_incomplete_runtime_bindings_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    mutation: Callable[[dict[str, Any]], None],
    code: str,
) -> None:
    expected = _expected()
    task = _live_task(expected["tasks"][0])
    mutation(task)
    with pytest.raises(inventory.GateError, match=code):
        inventory._hydrate_task(task, expected["tasks"][0])


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("prompt", "", "live_prompt_invalid"),
        ("prompt", None, "live_prompt_invalid"),
        ("env_variables", [], "live_env_variables_invalid"),
        ("output_json_schema", None, "live_output_json_schema_invalid"),
        ("output_json_schema", [], "live_output_json_schema_invalid"),
    ],
)
def test_private_task_content_requires_exact_safe_shapes(
    field: str, value: object, code: str
) -> None:
    expected = _expected()["tasks"][0]
    task = _live_task(expected)
    task[field] = value
    with pytest.raises(inventory.GateError, match=code):
        inventory._hydrate_task(task, expected)


@pytest.mark.parametrize("bad_file", [None, {}, {"target_path": "/task/a"}])
def test_every_runtime_seed_file_requires_complete_storage_binding(
    bad_file: object,
) -> None:
    expected = _expected()["tasks"][0]
    task = _live_task(expected)
    task["metadata"]["runtime_seed_manifest"]["files"] = [bad_file]
    with pytest.raises(inventory.GateError, match="runtime_seed_manifest_incomplete"):
        inventory._hydrate_task(task, expected)


@pytest.mark.parametrize(
    ("job_uid", "pod_uid", "code"),
    [
        ("not-a-uuid", POD_UID, "runtime_job_uid_invalid"),
        (JOB_UID, "not-a-uuid", "runtime_pod_uid_invalid"),
    ],
)
def test_runtime_uids_must_be_valid_before_network(
    monkeypatch: pytest.MonkeyPatch, job_uid: str, pod_uid: str, code: str
) -> None:
    monkeypatch.setattr(
        inventory,
        "fetch_json",
        lambda *_args, **_kwargs: pytest.fail("network must not be called"),
    )
    with pytest.raises(inventory.GateError, match=code):
        inventory.observe(
            _expected(),
            "test-key",
            root=ROOT,
            job_uid=job_uid,
            pod_uid=pod_uid,
        )


def test_source_contains_no_write_or_session_routes() -> None:
    source = Path(inventory.__file__).read_text()
    assert 'method="GET"' in source
    assert 'method="POST"' not in source
    assert 'method="PUT"' not in source
    assert 'method="PATCH"' not in source
    assert 'method="DELETE"' not in source
    assert "/v1/sessions" not in source
    assert "/v1/jobs" not in source
    assert "/v1/rollout-rewards" not in source


def test_expected_digest_tamper_fails_closed() -> None:
    expected = copy.deepcopy(_expected())
    expected["tasks"].pop()
    with pytest.raises(inventory.GateError, match="expected_inventory_digest_mismatch"):
        inventory._validate_expected(expected)
