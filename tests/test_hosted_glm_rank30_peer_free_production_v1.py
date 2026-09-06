from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from evals.fleet import hosted_glm_rank30_peer_free_package_v2 as controller_package
from evals.fleet import hosted_glm_rank30_peer_free_release_bootstrap_v1 as bootstrap
from evals.fleet import hosted_glm_rank30_peer_free_release_observer_v1 as observer
from evals.fleet import hosted_glm_rank30_peer_free_release_package_v1 as release_package
from evals.fleet import hosted_glm_rank30_peer_free_submit_v1 as submit
from evals.fleet import hosted_glm_rank30_peer_free_successor_v2 as successor

ROOT = Path(__file__).parents[1]
JOB_UID = "11111111-1111-4111-8111-111111111111"
POD_UID = "22222222-2222-4222-8222-222222222222"


def _runtime_plan() -> dict:
    plan = copy.deepcopy(successor.validate_all(ROOT)[successor.CONTROLLER])
    plan["launch_authorized"] = True
    plan.pop("plan_sha256")
    plan["plan_sha256"] = successor.self_hosted.digest_without(plan, "plan_sha256")
    return plan


def _clear_state() -> dict:
    return {
        "claim_receipts_examined": 1,
        "accepted_authority_projections_examined": 2,
        "accepted_authority_snapshot_sha256": "sha256:" + "4" * 64,
        "session_rows_examined": 1,
        "session_inventory_scans": 1,
        "archived_sessions_included": True,
        "stable_session_snapshot": True,
        "session_identity_projection": "/v1/sessions/identities",
        "session_snapshot_sha256": "sha256:" + "3" * 64,
        "fleet_gets": 2,
        "new_job_collisions": 0,
        "new_pod_collisions": 0,
        "new_configmap_collisions": 0,
        "active_hosted_controllers": 0,
        "kubernetes_gets": 3,
        "all_generation_claim_collisions": 0,
        "authoritative_session_collisions": 0,
        "accepted_evidence_collisions": 0,
        "output_root_collisions": 0,
        "endpoint_lease_slots_available": 2,
        "both_endpoint_lease_slots_simultaneously_free": True,
    }


def _release(monkeypatch: pytest.MonkeyPatch) -> dict:
    plan = _runtime_plan()
    monkeypatch.setattr(successor, "load", lambda _path: {})
    monkeypatch.setattr(successor, "build_runtime_plan", lambda *_args: plan)
    monkeypatch.setattr(observer, "_clear_state", lambda *_args, **_kwargs: _clear_state())
    binding = release_package.build_binding(ROOT)
    return observer.collect(
        binding,
        ROOT,
        job_uid=JOB_UID,
        pod_uid=POD_UID,
        api_key="not-persisted",
    )


def test_strict_json_rejects_duplicate_keys() -> None:
    with pytest.raises(observer.ObserverError, match="duplicate_json_key"):
        observer.strict_json(b'{"status":"CLEAR","status":"MALICIOUS"}')


def test_binding_rejects_rehashed_mutation() -> None:
    binding = release_package.build_binding(ROOT)
    binding["fresh_job_name"] = "attacker-job"
    binding["binding_sha256"] = observer.sha256(
        observer.canonical(
            {key: value for key, value in binding.items() if key != "binding_sha256"}
        )
    )
    with pytest.raises(observer.ObserverError, match="observer_binding_invalid"):
        observer.validate_binding(binding)


def test_session_collision_finds_exact_version_and_model_without_caller_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = release_package.build_binding(ROOT)
    requests: list[tuple[str, dict]] = []

    def identity_page(path: str, _key: str, params: dict) -> dict:
        requests.append((path, params))
        return {
            "sessions": [
                {
                    "session_id": "archived-session",
                    "eval_task_id": "task-id",
                    "eval_task_version_id": observer.EXPECTED_TASK_VERSION_ID,
                    "task_key": observer.EXPECTED_TASK_KEYS[0],
                    "model_id": observer.EXPECTED_SESSION_MODEL_ID,
                    "model_identity": observer.EXPECTED_SESSION_MODEL,
                    "model_identity_status": "resolved",
                    "status": "completed",
                }
            ],
            "limit": 500,
            "has_more": False,
            "next_cursor": None,
            "snapshot": "immutable-snapshot",
        }

    monkeypatch.setattr(
        observer,
        "_fleet_get",
        identity_page,
    )
    rows, gets, collisions, snapshot_sha = observer._session_collisions(binding, "not-persisted")
    assert (rows, gets, collisions) == (1, 1, 1)
    assert observer.SHA_RE.fullmatch(snapshot_sha)
    assert requests == [
        (
            "/v1/sessions/identities",
            {"task_key": observer.EXPECTED_TASK_KEYS[0], "limit": 500},
        )
    ]


def test_exact_version_ambiguous_model_identity_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = release_package.build_binding(ROOT)
    monkeypatch.setattr(
        observer,
        "_fleet_get",
        lambda *_args, **_kwargs: {
            "sessions": [
                {
                    "session_id": "ambiguous-session",
                    "eval_task_id": "task-id",
                    "eval_task_version_id": observer.EXPECTED_TASK_VERSION_ID,
                    "task_key": observer.EXPECTED_TASK_KEYS[0],
                    "model_id": None,
                    "model_identity": None,
                    "model_identity_status": "ambiguous",
                    "status": "completed",
                }
            ],
            "limit": 500,
            "has_more": False,
            "next_cursor": None,
            "snapshot": "immutable-snapshot",
        },
    )
    with pytest.raises(observer.ObserverError, match="model_identity_ambiguous"):
        observer._session_collisions(binding, "not-persisted")


def test_session_inventory_requires_stable_keyset_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = release_package.build_binding(ROOT)
    calls: list[dict] = []

    def changing(_path: str, _key: str, params: dict) -> dict:
        calls.append(params)
        page = len(calls)
        return {
            "sessions": [
                {
                    "session_id": f"session-{page}",
                    "eval_task_id": "task-id",
                    "eval_task_version_id": "33333333-3333-4333-8333-333333333333",
                    "task_key": observer.EXPECTED_TASK_KEYS[0],
                    "model_id": None,
                    "model_identity": None,
                    "model_identity_status": "ambiguous",
                    "status": "completed",
                }
            ],
            "limit": 500,
            "has_more": page == 1,
            "next_cursor": "cursor-2" if page == 1 else None,
            "snapshot": "snapshot-1" if page == 1 else "snapshot-drift",
        }

    monkeypatch.setattr(observer, "_fleet_get", changing)
    with pytest.raises(observer.ObserverError, match="snapshot_cursor_changed"):
        observer._session_collisions(binding, "not-persisted")
    assert len(calls) == 2
    assert "cursor" not in calls[0]
    assert calls[1]["cursor"] == "cursor-2"


@pytest.mark.parametrize("task_version", [None, "", "ambiguous-version"])
def test_same_task_unknown_version_fails_closed(
    monkeypatch: pytest.MonkeyPatch, task_version: object
) -> None:
    binding = release_package.build_binding(ROOT)
    monkeypatch.setattr(
        observer,
        "_fleet_get",
        lambda *_args, **_kwargs: {
            "sessions": [
                {
                    "session_id": "unknown-version-session",
                    "eval_task_id": "task-id",
                    "eval_task_version_id": task_version,
                    "task_key": observer.EXPECTED_TASK_KEYS[0],
                    "model_id": observer.EXPECTED_SESSION_MODEL_ID,
                    "model_identity": observer.EXPECTED_SESSION_MODEL,
                    "model_identity_status": "resolved",
                    "status": "completed",
                }
            ],
            "limit": 500,
            "has_more": False,
            "next_cursor": None,
            "snapshot": "immutable-snapshot",
        },
    )
    with pytest.raises(observer.ObserverError, match="task_version_ambiguous"):
        observer._session_collisions(binding, "not-persisted")


def test_session_route_exact_schema_requires_model_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = release_package.build_binding(ROOT)
    row = {
        "session_id": "missing-model-id",
        "eval_task_id": "task-id",
        "eval_task_version_id": observer.EXPECTED_TASK_VERSION_ID,
        "task_key": observer.EXPECTED_TASK_KEYS[0],
        "model_identity": observer.EXPECTED_SESSION_MODEL,
        "model_identity_status": "resolved",
        "status": "completed",
    }
    monkeypatch.setattr(
        observer,
        "_fleet_get",
        lambda *_args, **_kwargs: {
            "sessions": [row],
            "limit": 500,
            "has_more": False,
            "next_cursor": None,
            "snapshot": "immutable-snapshot",
        },
    )
    with pytest.raises(observer.ObserverError, match="session_identity_row_invalid"):
        observer._session_collisions(binding, "not-persisted")


def test_session_route_rejects_rehashed_model_id_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = release_package.build_binding(ROOT)
    monkeypatch.setattr(
        observer,
        "_fleet_get",
        lambda *_args, **_kwargs: {
            "sessions": [
                {
                    "session_id": "model-id-drift",
                    "eval_task_id": "task-id",
                    "eval_task_version_id": observer.EXPECTED_TASK_VERSION_ID,
                    "task_key": observer.EXPECTED_TASK_KEYS[0],
                    "model_id": "different-model",
                    "model_identity": observer.EXPECTED_SESSION_MODEL,
                    "model_identity_status": "resolved",
                    "status": "completed",
                }
            ],
            "limit": 500,
            "has_more": False,
            "next_cursor": None,
            "snapshot": "immutable-snapshot",
        },
    )
    with pytest.raises(observer.ObserverError, match="model_identity_row_invalid"):
        observer._session_collisions(binding, "not-persisted")


def test_acceptance_check_reads_only_exact_sanitized_projections(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    for authority in (successor.LEDGER_AUTHORITY, successor.LIVE_LEDGER_VALIDATION):
        source = ROOT / authority["path"]
        target = tmp_path / authority["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    protected = tmp_path / "jobs" / "arbitrary" / "accepted" / "result.json"
    protected.parent.mkdir(parents=True)
    protected.write_text('{"score":"must-not-be-read"}')
    monkeypatch.setattr(
        observer.os,
        "walk",
        lambda *_args, **_kwargs: pytest.fail("acceptance check must not recursively scan"),
    )
    examined, collisions, projection_sha = observer._accepted_authority_clear(tmp_path)
    assert (examined, collisions) == (2, 0)
    assert observer.SHA_RE.fullmatch(projection_sha)

    validation_path = tmp_path / successor.LIVE_LEDGER_VALIDATION["path"]
    malicious = json.loads(validation_path.read_text())
    malicious["score"] = "forbidden"
    malicious["receipt_sha256"] = observer.digest(malicious)
    validation_path.write_text(json.dumps(malicious))
    with pytest.raises(observer.ObserverError, match="digest_invalid"):
        observer._accepted_authority_clear(tmp_path)


def test_observer_emits_exact_valid_release(monkeypatch: pytest.MonkeyPatch) -> None:
    release = _release(monkeypatch)
    successor.validate_release(
        release, _runtime_plan(), controller_package.source_package_sha256(ROOT)
    )
    assert release["status"] == "CLEAR_PEER_FREE"
    assert release["fresh_collision_reconciliation"]["api_mutations"] == 0
    assert release["privacy"] == {
        "scores_read": False,
        "prompts_traces_flags_read": False,
        "credentials_included": False,
    }


def test_stale_and_extra_field_release_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    release = _release(monkeypatch)
    stale = copy.deepcopy(release)
    stale["checked_at_utc"] = (
        (datetime.now(UTC) - timedelta(seconds=601)).isoformat().replace("+00:00", "Z")
    )
    stale["receipt_sha256"] = successor.self_hosted.digest_without(stale, "receipt_sha256")
    with pytest.raises(RuntimeError, match="stale"):
        successor.validate_release(
            stale, _runtime_plan(), controller_package.source_package_sha256(ROOT)
        )
    malicious = copy.deepcopy(release)
    malicious["prompt"] = "must never survive exact-key validation"
    malicious["receipt_sha256"] = successor.self_hosted.digest_without(malicious, "receipt_sha256")
    with pytest.raises(RuntimeError, match="release drifted"):
        successor.validate_release(
            malicious, _runtime_plan(), controller_package.source_package_sha256(ROOT)
        )


def test_controller_source_authority_and_bootstrap_are_complete(tmp_path: Path) -> None:
    assert controller_package.PATHS == observer.CONTROLLER_SOURCE_PATHS
    assert observer.controller_source_sha256(ROOT) == controller_package.source_package_sha256(ROOT)
    rendered = release_package.render(ROOT)
    data = rendered["items"][0]["data"]
    projected = tmp_path / "bootstrap"
    repo = tmp_path / "repo"
    projected.mkdir()
    for name, content in data.items():
        (projected / name).write_text(content)
    for relative in data["package.json"] and release_package.FILES:
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(data[relative.replace("/", "__SLASH__")])
    bootstrap.validate(projected / "package.json", projected, repo)
    environment = {**os.environ, "PYTHONPATH": str(repo)}
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from evals.fleet import hosted_glm_rank30_peer_free_release_observer_v1",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_controller_materialized_import_includes_exact_missing_modules(tmp_path: Path) -> None:
    rendered = controller_package.render(ROOT)
    data = rendered["objects"]["items"][0]["data"]
    repo = tmp_path / "controller"
    fleet = repo / "evals" / "fleet"
    config = fleet / "configs"
    config.mkdir(parents=True)
    (repo / "evals" / "__init__.py").write_text("")
    (fleet / "__init__.py").write_text("")
    code_bindings = {
        "self_hosted.py": "self_hosted.py",
        "runner.py": "opencode_train_sweep_runner.py",
        "endpoint_lease.py": "endpoint_lease.py",
        "predecessor.py": "exact_pass4_bulk_v3.py",
        "base_engine.py": "exact_pass4_bulk_runtime_v3.py",
        "engine.py": "hosted_glm_whole_task_engine_v1.py",
        "universe.py": "exact_pass4_universe.py",
        "crypto.py": "exact_pass4_crypto.py",
        "inventory.py": "exact_pass4_task_inventory.py",
        "whole_task_v1.py": "hosted_glm_whole_task_successor_v1.py",
        "bulk.py": "hosted_glm_rank30_peer_free_successor_v2.py",
        "bulk_runtime.py": "hosted_glm_rank30_peer_free_runtime_v2.py",
        "hosted_glm_exact_bulk_v1.py": "hosted_glm_exact_bulk_v1.py",
        "hosted_glm_exact_bulk_runtime_v1.py": "hosted_glm_exact_bulk_runtime_v1.py",
        "fixed_proxy.py": "fixed_proxy.py",
    }
    for source, target in code_bindings.items():
        (fleet / target).write_text(data[source])
    config_bindings = {
        "campaign.json": "q38-glm53-exact-easiest100-pass4-campaign-v1.json",
        "selection.json": "opencode-easiest-train100-selection-v2.json",
        "glm-template.json": "glm53-opencode-autocontinue-canary1-v1.json",
        "qwen-template.json": "qwen38-opencode-autocontinue-canary1-v1.json",
        "bulk-qwen-a.json": "exact-pass4-bulk-qwen-a-v3.json",
        "bulk-qwen-b.json": "exact-pass4-bulk-qwen-b-v3.json",
        "bulk-glm-a.json": "exact-pass4-bulk-glm-a-v3.json",
        "bulk-glm-b.json": "exact-pass4-bulk-glm-b-v3.json",
    }
    for source, target in config_bindings.items():
        (config / target).write_text(data[source])
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from evals.fleet import hosted_glm_rank30_peer_free_runtime_v2",
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(repo)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_submit_render_rechecks_and_uses_exact_release(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    release = _release(monkeypatch)
    release_path = tmp_path / "RELEASE.json"
    release_path.write_text(json.dumps(release))
    plan = _runtime_plan()
    monkeypatch.setattr(
        submit, "_runtime_plan", lambda _root: (plan, release["source_package_sha256"])
    )
    monkeypatch.setattr(observer, "recheck", lambda *_args, **_kwargs: _clear_state())
    rendered = submit.render(ROOT, release_path, api_key="not-persisted")
    assert rendered["release_receipt_sha256"] == release["receipt_sha256"]
    assert [item["kind"] for item in rendered["items"]] == ["ConfigMap", "Job"]
    assert rendered["items"][0]["immutable"] is True


def test_submit_collision_and_repeat_fail_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    release = _release(monkeypatch)
    release_path = tmp_path / "RELEASE.json"
    release_path.write_text(json.dumps(release))
    plan = _runtime_plan()
    monkeypatch.setattr(
        submit, "_runtime_plan", lambda _root: (plan, release["source_package_sha256"])
    )
    collision = _clear_state()
    collision["authoritative_session_collisions"] = 1
    monkeypatch.setattr(observer, "recheck", lambda *_args, **_kwargs: collision)
    with pytest.raises(submit.SubmitError, match="reconciliation_failed"):
        submit.render(ROOT, release_path, api_key="not-persisted")

    receipt_path = tmp_path / "SUBMITTED.json"
    monkeypatch.setattr(submit, "SUBMIT_RECEIPT", receipt_path)
    monkeypatch.setattr(observer, "recheck", lambda *_args, **_kwargs: _clear_state())
    manifest = submit.render(ROOT, release_path, api_key="not-persisted")
    calls = []

    def runner(payload: str) -> subprocess.CompletedProcess[str]:
        calls.append(payload)
        return subprocess.CompletedProcess([], 0, "created", "")

    receipt = submit.submit_once(manifest, receipt_path, runner=runner)
    assert receipt["create_invocations"] == 1
    assert len(calls) == 1
    with pytest.raises(submit.SubmitError, match="do_not_repeat"):
        submit.submit_once(manifest, receipt_path, runner=runner)
    assert len(calls) == 1


def test_submit_rejects_stale_or_rehashed_extra_preview(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    release = _release(monkeypatch)
    release_path = tmp_path / "RELEASE.json"
    release_path.write_text(json.dumps(release))
    plan = _runtime_plan()
    monkeypatch.setattr(
        submit, "_runtime_plan", lambda _root: (plan, release["source_package_sha256"])
    )
    monkeypatch.setattr(observer, "recheck", lambda *_args, **_kwargs: _clear_state())
    stale = submit.render(ROOT, release_path, api_key="not-persisted")
    stale["previewed_at_utc"] = (
        (datetime.now(UTC) - timedelta(seconds=61)).isoformat().replace("+00:00", "Z")
    )
    stale["preview_sha256"] = submit.preview_digest(stale)
    with pytest.raises(submit.SubmitError, match="preview_stale"):
        submit.validate_preview(stale)
    malicious = copy.deepcopy(stale)
    malicious["prompt"] = "forbidden"
    malicious["previewed_at_utc"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    malicious["preview_sha256"] = submit.preview_digest(malicious)
    with pytest.raises(submit.SubmitError, match="preview_invalid"):
        submit.validate_preview(malicious)


def test_submit_release_loader_rejects_duplicate_key(tmp_path: Path) -> None:
    release = tmp_path / "RELEASE.json"
    release.write_bytes(b'{"status":"CLEAR_PEER_FREE","status":"MALICIOUS"}')
    with pytest.raises(observer.ObserverError, match="duplicate_json_key"):
        submit.validate_release(release, ROOT)


def test_observer_package_is_score_free_and_held() -> None:
    rendered = release_package.render(ROOT)
    job = rendered["items"][1]
    assert rendered["launch_authorized"] is False
    assert rendered["scoring_authorized"] is False
    assert job["metadata"]["annotations"]["cyber-post-train.fleet.ai/score-free"] == "true"
    command = job["spec"]["template"]["spec"]["containers"][0]["command"][2]
    assert "release_bootstrap_v1" in command
    assert "--projected-root /bootstrap" in command
    assert "POST" not in Path(observer.__file__).read_text()


def test_production_held_receipt_rebuilds() -> None:
    path = (
        ROOT / "docs/evidence/glm53-study/"
        "2026-09-06-glm53-hosted-rank30-peer-free-production-held-v1.json"
    )
    value = json.loads(path.read_text())
    rendered = release_package.render(ROOT)
    configmap = rendered["items"][0]
    objects = {"apiVersion": "v1", "kind": "List", "items": rendered["items"]}
    assert value["receipt_sha256"] == observer.digest(value)
    assert (
        value["release_observer"]["binding_sha256"]
        == json.loads(configmap["data"]["binding.json"])["binding_sha256"]
    )
    assert value["release_observer"]["configmap_sha256"] == successor.self_hosted.sha256(
        successor.self_hosted.canonical_json(configmap)
    )
    assert value["release_observer"]["rendered_list_sha256"] == successor.self_hosted.sha256(
        successor.self_hosted.canonical_json(objects)
    )
