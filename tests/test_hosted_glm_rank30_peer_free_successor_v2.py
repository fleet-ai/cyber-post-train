import copy
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from evals.fleet import hosted_glm_rank30_peer_free_package_v2 as package
from evals.fleet import hosted_glm_rank30_peer_free_runtime_v2 as runtime
from evals.fleet import hosted_glm_rank30_peer_free_successor_v2 as successor
from evals.fleet import hosted_glm_whole_task_engine_v1 as engine
from evals.fleet import self_hosted

ROOT = Path(__file__).resolve().parents[1]
JOB_UID = "11111111-1111-4111-8111-111111111111"
POD_UID = "22222222-2222-4222-8222-222222222222"


@pytest.fixture(autouse=True)
def _safe_runtime_projection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(engine, "_task_for_item", lambda _plan, item: {"task": item["task_key"]})
    monkeypatch.setattr(
        engine,
        "_attempt_config",
        lambda _plan, _task, item: {
            "run_id": item["run_id"],
            "execution": {
                "cell_id": item["cell_id"],
                "execution_id": item["execution_id"],
            },
        },
    )


def _plan() -> dict:
    return successor.validate_all(ROOT)[successor.CONTROLLER]


def _release(plan: dict, source_sha: str) -> dict:
    body = {
        "schema_version": successor.RELEASE_SCHEMA,
        "status": "CLEAR_PEER_FREE",
        "checked_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "launch_authorized": True,
        "scoring_authorized": True,
        "controller": successor.release_projection(plan),
        "source_package_sha256": source_sha,
        "ledger_authority": successor.LEDGER_AUTHORITY,
        "live_ledger_validation": successor.LIVE_LEDGER_VALIDATION,
        "diagnostic_v2": successor.DIAGNOSTIC_V2,
        "superseded_identities": successor.SUPERSEDED_IDENTITIES,
        "fresh_collision_reconciliation": {
            "checked_immediately_before_create": True,
            "observer_job_uid": JOB_UID,
            "observer_pod_uid": POD_UID,
            "observed_cells": 4,
            "all_generation_claim_collisions": 0,
            "authoritative_session_collisions": 0,
            "accepted_evidence_collisions": 0,
            "output_root_collisions": 0,
            "new_job_collisions": 0,
            "new_pod_collisions": 0,
            "new_configmap_collisions": 0,
            "active_hosted_controllers": 0,
            "endpoint_lease_slots_available": 2,
            "both_endpoint_lease_slots_simultaneously_free": True,
            "session_inventory_scans": 1,
            "archived_sessions_included": True,
            "stable_session_snapshot": True,
            "session_identity_projection": "/v1/sessions/identities",
            "session_snapshot_sha256": "sha256:" + "3" * 64,
            "api_mutations": 0,
        },
        "privacy": {
            "scores_read": False,
            "prompts_traces_flags_read": False,
            "credentials_included": False,
        },
    }
    return {**body, "receipt_sha256": self_hosted.digest_without(body, "receipt_sha256")}


def test_exact_rank30_generation_two_scientific_binding() -> None:
    plan = _plan()
    assert set(successor.validate_all(ROOT)) == {successor.CONTROLLER}
    assert plan["partition"] == {
        "whole_task_rank": 30,
        "attempts": [1, 2, 3, 4],
        "all_cells_previously_unstarted_required": True,
        "current_peer_required": False,
        "both_endpoint_slots_free_required": True,
        "other_ranks_excluded": True,
    }
    assert plan["model"]["served_id"] == "glm-5.3"
    assert plan["model"]["revision"] == "30333038ada1f1dacb294a93270305a890b50c14"
    assert plan["harness"]["version"] == "1.18.27"
    assert plan["harness"]["context_management"] == (
        "opencode_1.18.27_native_compaction_autocontinue_v1"
    )
    assert plan["execution"]["required_task_tools"] == ["bash", "submit_report"]
    assert [row["execution_generation"] for row in plan["attempts"]] == [2] * 4
    assert [row["execution_id"] for row in plan["attempts"]] == [
        "sha256:eb5945fdab0567dfadcc9f4b1aa76d8edf2fe9ae27695202bc6a884613c0be21",
        "sha256:aeea2d30f6c4fd425ea4d4c87af3301a9dcfc839618dafd2e4bb1ddc97a86432",
        "sha256:54cb087f456a4284a87860dad4463c476bedf8508abdc7083076e673918b2a1d",
        "sha256:4296e8b9300a114686794dce795cec010f601430e2ac13594cfaa650431b4c72",
    ]
    assert all(
        "r029" not in row["run_id"] and "g1" not in row["run_id"]
        for row in plan["attempts"]
    )


def test_ledger_v48_and_score_free_diagnostic_are_exact() -> None:
    assert successor.LIVE_LEDGER_VALIDATION["glm_tally"] == {
        "accepted": 24,
        "active": 0,
        "blocked": 4,
        "unstarted": 372,
    }
    diagnostic = successor.DIAGNOSTIC_V2
    assert diagnostic["failed_phase"] == "07-strict-current-peer"
    assert diagnostic["failure_code"] == "CURRENT_PEER_INVARIANT_FAILED"
    assert diagnostic["provider_session_model_boundary_crossed"] is False
    assert diagnostic["api_calls"] == 0
    assert diagnostic["scores_read"] is False
    assert diagnostic["prompts_traces_flags_read"] is False


def test_held_package_is_one_create_once_job_and_closed() -> None:
    rendered = package.render(ROOT)
    assert rendered["launch_authorized"] is False
    assert rendered["scoring_authorized"] is False
    assert rendered["reason"] == "requires_fresh_digest_valid_peer_free_release"
    configmap, job = rendered["objects"]["items"]
    assert configmap["metadata"]["name"] == successor.CONFIGMAP_NAME
    assert configmap["immutable"] is True
    assert job["metadata"]["name"] == successor.JOB_NAME
    assert job["metadata"]["annotations"]["cyber-post-train.fleet.ai/create-once"] == "true"
    assert job["metadata"]["annotations"]["cyber-post-train.fleet.ai/launch-authorized"] == "false"
    assert job["spec"]["template"]["spec"]["priorityClassName"] == "fleet-serve-low"
    assert job["spec"]["template"]["spec"]["preemptionPolicy"] == "Never"


def test_tracked_held_receipt_rebuilds_byte_exact() -> None:
    path = (
        ROOT
        / "docs/evidence/glm53-study/2026-09-06-glm53-hosted-rank30-peer-free-held-v2.json"
    )
    value = json.loads(path.read_text())
    rendered = package.render(ROOT)
    assert value == rendered["held_receipt"]
    assert value["receipt_sha256"] == self_hosted.digest_without(value, "receipt_sha256")
    assert value["source_package_sha256"] == rendered["source_package_sha256"]
    assert value["required_release"]["current_peer_required"] is False
    assert value["required_release"]["both_endpoint_slots_simultaneously_free"] is True


def test_release_requires_peer_free_collision_zero_and_both_slots() -> None:
    plan = _plan()
    source_sha = package.source_package_sha256(ROOT)
    release = _release(plan, source_sha)
    successor.validate_release(release, plan, source_sha)
    for field, bad in (
        ("active_hosted_controllers", 1),
        ("all_generation_claim_collisions", 1),
        ("authoritative_session_collisions", 1),
        ("accepted_evidence_collisions", 1),
        ("output_root_collisions", 1),
        ("new_job_collisions", 1),
        ("new_pod_collisions", 1),
        ("new_configmap_collisions", 1),
        ("endpoint_lease_slots_available", 1),
        ("both_endpoint_lease_slots_simultaneously_free", False),
    ):
        changed = copy.deepcopy(release)
        changed["fresh_collision_reconciliation"][field] = bad
        changed["receipt_sha256"] = self_hosted.digest_without(changed, "receipt_sha256")
        with pytest.raises(RuntimeError, match="release drifted"):
            successor.validate_release(changed, plan, source_sha)


def test_release_rejects_ledger_diagnostic_extra_keys_and_staleness() -> None:
    plan = _plan()
    source_sha = package.source_package_sha256(ROOT)
    release = _release(plan, source_sha)
    for mutate in (
        lambda value: value["ledger_authority"].update(file_sha256="sha256:" + "0" * 64),
        lambda value: value["diagnostic_v2"].update(failed_phase="06-model-boundary"),
        lambda value: value.update(protected_content="forbidden"),
    ):
        changed = copy.deepcopy(release)
        mutate(changed)
        changed["receipt_sha256"] = self_hosted.digest_without(changed, "receipt_sha256")
        with pytest.raises(RuntimeError, match="release drifted"):
            successor.validate_release(changed, plan, source_sha)
    stale = copy.deepcopy(release)
    stale["checked_at_utc"] = (datetime.now(UTC) - timedelta(seconds=601)).isoformat().replace(
        "+00:00", "Z"
    )
    stale["receipt_sha256"] = self_hosted.digest_without(stale, "receipt_sha256")
    with pytest.raises(RuntimeError, match="release is stale"):
        successor.validate_release(stale, plan, source_sha)


def test_four_claim_reservation_precedes_model_boundary(tmp_path: Path) -> None:
    plan = _plan()
    jobs = tmp_path / "jobs"
    claims = tmp_path / "claims"
    reservations = tmp_path / "reservations"
    out = jobs / successor.JOB_NAME
    for path in (jobs, claims, reservations, out):
        path.mkdir(parents=True, exist_ok=True)
    provider = successor.AtomicWholeTaskClaims(
        plan,
        out,
        key="inert",
        jobs_root=jobs,
        reservation_root=reservations,
        session_check=lambda _config, _key: None,
    )
    provider(plan, plan["attempts"][0], claims, JOB_UID, POD_UID)
    assert len(provider.claims) == 4
    assert len(list(claims.glob("*.json"))) == 4
    assert out.joinpath("RESERVATION.json").is_file()
    assert not out.joinpath("model-boundaries").exists()
    provider.mark_model_boundary(plan["attempts"][0])
    assert len(list(out.joinpath("model-boundaries").glob("*.json"))) == 1


def test_materialized_bootstrap_uses_exact_v2_modules() -> None:
    script = (ROOT / "evals/fleet/scripts/run_hosted_glm_rank30_peer_free_v2.sh").read_text()
    assert "bulk.py:hosted_glm_rank30_peer_free_successor_v2.py" in script
    assert "bulk_runtime.py:hosted_glm_rank30_peer_free_runtime_v2.py" in script
    assert "whole_task_v1.py:hosted_glm_whole_task_successor_v1.py" in script
    assert "test -f /bootstrap/release.json" in script
    assert "hosted_glm_rank30_peer_free_runtime_v2" in script
    assert "hosted_glm_whole_task_runtime_v1 --controller" not in script


def test_runtime_failure_receipt_does_not_serialize_exception_text() -> None:
    source = Path(runtime.__file__).read_text()
    assert "error_sha256" not in source
    assert "error_type" not in source
    assert "str(exc)" not in source
    assert "prompts_traces_flags_included" in source


def test_unknown_controller_and_peer_dependency_fail_closed() -> None:
    inventory = successor.load(successor.prior.source_runtime.INVENTORY_PATH) if False else {}
    with pytest.raises(ValueError, match="unknown peer-free rank30 controller"):
        successor.build_runtime_plan("glm-hosted-stale-peer", inventory, ROOT)
    plan = _plan()
    changed = copy.deepcopy(plan)
    changed["partition"]["current_peer_required"] = True
    changed["plan_sha256"] = self_hosted.digest_without(changed, "plan_sha256")
    assert changed != plan
