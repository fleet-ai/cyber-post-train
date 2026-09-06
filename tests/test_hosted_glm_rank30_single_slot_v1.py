import copy
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from evals.fleet import hosted_glm_rank30_single_slot_package_v1 as package
from evals.fleet import hosted_glm_rank30_single_slot_release_package_v1 as release_package
from evals.fleet import hosted_glm_rank30_single_slot_release_v1 as release
from evals.fleet import hosted_glm_rank30_single_slot_v1 as successor
from evals.fleet import hosted_glm_whole_task_engine_v1 as engine
from evals.fleet import hosted_glm_whole_task_successor_v1 as whole
from evals.fleet import self_hosted

ROOT = Path(__file__).resolve().parents[1]
JOB_UID = "11111111-1111-4111-8111-111111111111"
POD_UID = "22222222-2222-4222-8222-222222222222"
HELD_RECEIPT = (
    ROOT
    / "docs/evidence/glm53-study/"
    "2026-09-06-glm53-hosted-rank30-single-slot-held-v1.json"
)


def _binding(slot: int = 2) -> dict:
    return {
        "slot": slot,
        "path": str(
            successor.LEASE_ROOT
            / successor.LEASE_ENDPOINT_KEY
            / f"slot-{slot}.lock"
        ),
        "device": 42,
        "inode": 101,
        "size": 0,
    }


def _receipt(plan: dict, source_sha: str) -> dict:
    body = {
        "schema_version": successor.RELEASE_SCHEMA,
        "status": "CLEAR",
        "checked_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "launch_authorized": True,
        "scoring_authorized": True,
        "controller_cap": 1,
        "controller": successor.release_projection(plan),
        "source_package_sha256": source_sha,
        "ledger_authority": whole.LEDGER_AUTHORITY,
        "selection_authority": whole.SELECTION_AUTHORITY,
        "live_rank29_peer": {
            "job_name": successor.PEER_JOB,
            "job_uid": successor.PEER_JOB_UID,
            "job_active": 1,
            "pod_name": successor.PEER_POD,
            "pod_uid": successor.PEER_POD_UID,
            "pod_phase": "Running",
            "pod_ready": True,
            "pod_restarts": 0,
            "endpoint_slot_held": True,
            "run_id": successor.PEER_RUN_ID,
            "stream_path": str(successor.PEER_STREAM),
            "stream_bytes": 1_000_000,
            "stream_mtime_epoch": int(datetime.now(UTC).timestamp()),
        },
        "endpoint_lease_observer": {
            "lease_root": str(successor.LEASE_ROOT),
            "endpoint_key": successor.LEASE_ENDPOINT_KEY,
            "maximum_streams": 2,
            "held_slots": 1,
            "held_slot_numbers": [1],
            "available_slots": 1,
            "available_slot_bindings": [_binding()],
            "probe_released": True,
        },
        "fresh_collision_reconciliation": {
            "checked_immediately_before_create": True,
            "observer_job_uid": JOB_UID,
            "observer_pod_uid": POD_UID,
            "observed_cells": 4,
            "claim_files_examined": 93,
            "accepted_files_examined": 57,
            "session_rows_examined": 47,
            "canonical_claim_collisions": 0,
            "authoritative_session_collisions": 0,
            "accepted_evidence_collisions": 0,
            "output_root_collisions": 0,
            "new_job_collisions": 0,
            "new_configmap_collisions": 0,
            "api_mutations": 0,
        },
        "privacy": {
            "scores_read": False,
            "prompts_traces_flags_read": False,
            "credentials_included": False,
        },
    }
    return {**body, "receipt_sha256": self_hosted.digest_without(body, "receipt_sha256")}


def test_rank30_only_preserves_rank31_and_rank29() -> None:
    plans = successor.validate_all(ROOT)
    assert set(plans) == {successor.CONTROLLER}
    plan = plans[successor.CONTROLLER]
    assert plan["partition"]["whole_task_rank"] == 30
    assert [row["attempt"] for row in plan["attempts"]] == [1, 2, 3, 4]
    assert all(row["selection_rank"] == 30 for row in plan["attempts"])
    assert "glm-hosted-r31-whole-task" not in successor.CONTROLLERS
    engine.base.validate_bulk_adapter(successor)


def test_held_scored_and_observer_packages_are_create_once_and_closed() -> None:
    scored = package.render(ROOT)
    observer = release_package.render(ROOT)
    assert scored["launch_authorized"] is False
    assert scored["scoring_authorized"] is False
    assert observer["launch_authorized"] is False
    assert observer["scoring_authorized"] is False
    assert observer["model_calls_authorized"] is False
    assert len(scored["objects"]["items"]) == 2
    assert scored["objects"]["items"][0]["immutable"] is True
    assert "rank31" not in json.dumps(scored["objects"])
    assert "docker build" not in observer["objects"]["items"][0]["data"]["run.sh"]
    for rendered in (scored, observer):
        job = rendered["objects"]["items"][1]
        assert job["metadata"]["annotations"][
            "cyber-post-train.fleet.ai/create-once"
        ] == "true"


def test_held_receipt_binds_exact_packages_and_stays_closed() -> None:
    receipt = json.loads(HELD_RECEIPT.read_text())
    assert receipt["receipt_sha256"] == self_hosted.digest_without(
        receipt, "receipt_sha256"
    )
    assert receipt["source_package_sha256"] == package.source_package_sha256(ROOT)
    assert receipt["held_scored_package_sha256"] == package.render(ROOT)[
        "package_sha256"
    ]
    assert receipt["held_release_observer_package_sha256"] == release_package.render(
        ROOT
    )["package_sha256"]
    assert receipt["controller"] == successor.release_projection(
        successor.validate_all(ROOT)[successor.CONTROLLER]
    )
    assert receipt["launch_authorized"] is False
    assert receipt["scoring_authorized"] is False
    assert receipt["rank31_disposition"] == {
        "cells_claimed": 0,
        "held": True,
        "selected": False,
    }


def test_release_binds_live_peer_single_slot_and_all_collisions() -> None:
    plan = successor.validate_all(ROOT)[successor.CONTROLLER]
    source_sha = package.source_package_sha256(ROOT)
    valid = _receipt(plan, source_sha)
    successor.validate_release(valid, plan, source_sha)
    mutations = [
        ("live_rank29_peer", "job_active", 0),
        ("endpoint_lease_observer", "held_slots", 0),
        ("endpoint_lease_observer", "held_slot_numbers", [2]),
        ("endpoint_lease_observer", "available_slots", 2),
        ("fresh_collision_reconciliation", "canonical_claim_collisions", 1),
        ("fresh_collision_reconciliation", "authoritative_session_collisions", 1),
        ("fresh_collision_reconciliation", "accepted_evidence_collisions", 1),
        ("fresh_collision_reconciliation", "output_root_collisions", 1),
        ("fresh_collision_reconciliation", "new_job_collisions", 1),
        ("fresh_collision_reconciliation", "new_configmap_collisions", 1),
    ]
    for section, field, value in mutations:
        changed = copy.deepcopy(valid)
        changed[section][field] = value
        changed["receipt_sha256"] = self_hosted.digest_without(
            changed, "receipt_sha256"
        )
        with pytest.raises(RuntimeError, match="release drifted"):
            successor.validate_release(changed, plan, source_sha)

    changed = copy.deepcopy(valid)
    changed["endpoint_lease_observer"]["available_slot_bindings"][0]["slot"] = 1
    changed["endpoint_lease_observer"]["available_slot_bindings"][0]["path"] = str(
        successor.LEASE_ROOT / successor.LEASE_ENDPOINT_KEY / "slot-1.lock"
    )
    changed["receipt_sha256"] = self_hosted.digest_without(
        changed, "receipt_sha256"
    )
    with pytest.raises(RuntimeError, match="release drifted"):
        successor.validate_release(changed, plan, source_sha)

    stale = copy.deepcopy(valid)
    stale["live_rank29_peer"]["stream_mtime_epoch"] = 1
    stale["receipt_sha256"] = self_hosted.digest_without(stale, "receipt_sha256")
    with pytest.raises(RuntimeError, match="peer stream is stale"):
        successor.validate_release(stale, plan, source_sha)


def test_observer_build_is_sanitized_and_validated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = successor.validate_all(ROOT)[successor.CONTROLLER]
    source_sha = package.source_package_sha256(ROOT)
    inventory = tmp_path / "inventory.json"
    inventory.write_text("{}")
    monkeypatch.setenv("FLEET_API_KEY", "inert")
    monkeypatch.setenv("JOB_UID", JOB_UID)
    monkeypatch.setenv("POD_UID", POD_UID)
    monkeypatch.setenv("GLM_HOSTED_R30_SOURCE_SHA256", source_sha)
    monkeypatch.setattr(release.source_runtime, "INVENTORY_PATH", inventory)
    monkeypatch.setattr(successor, "build_runtime_plan", lambda *_args: plan)
    monkeypatch.setattr(
        release,
        "_validate_live_peer",
        lambda: _receipt(plan, source_sha)["live_rank29_peer"],
    )
    monkeypatch.setattr(
        release,
        "_probe_slots",
        lambda: _receipt(plan, source_sha)["endpoint_lease_observer"],
    )
    monkeypatch.setattr(
        release,
        "_collisions",
        lambda *_args: _receipt(plan, source_sha)["fresh_collision_reconciliation"],
    )
    receipt = release.build(ROOT)
    successor.validate_release(receipt, plan, source_sha)
    assert receipt["privacy"] == {
        "scores_read": False,
        "prompts_traces_flags_read": False,
        "credentials_included": False,
    }
    assert "prompt" not in json.dumps(receipt).lower().replace(
        "prompts_traces_flags_read", ""
    )
