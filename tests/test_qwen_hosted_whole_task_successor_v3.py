from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from evals.fleet import exact_pass4_bulk_runtime_v3 as engine
from evals.fleet import qwen_hosted_whole_task_release_gate_package_v5 as gate_package
from evals.fleet import qwen_hosted_whole_task_successor_v2 as predecessor
from evals.fleet import qwen_hosted_whole_task_successor_v3 as successor
from evals.fleet import qwen_hosted_whole_task_successor_v3_package as package
from evals.fleet import self_hosted

ROOT = Path(__file__).parents[1]


def test_generation21_preserves_exact_eight_cell_treatment_with_fresh_identities() -> None:
    old = predecessor.build_plans(ROOT)
    new = successor.build_plans(ROOT)
    identities = {
        (row["selection_rank"], row["attempt"])
        for plan in new.values()
        for row in plan["attempts"]
    }
    assert identities == {
        (rank, attempt) for rank in (15, 16) for attempt in range(1, 5)
    }
    for controller, plan in new.items():
        prior = old[controller]
        for field in ("model", "harness", "authority", "tasks", "execution", "treatment"):
            assert plan[field] == prior[field]
        assert plan["model"]["revision"] == "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
        assert plan["harness"]["version"] == "1.18.27"
        assert plan["treatment"]["tools"] == ["bash", "submit_report"]
        assert plan["treatment"]["context_window_size"] == 262144
        assert plan["treatment"]["max_model_requests"] == 600
        assert plan["treatment"]["max_output_tokens"] == 32768
        assert plan["campaign_id"] != prior["campaign_id"]
        assert plan["sfs_root"] != prior["sfs_root"]
        for current, stale in zip(plan["attempts"], prior["attempts"], strict=True):
            assert current["cell_id"] == stale["cell_id"]
            assert current["task_version_id"] == stale["task_version_id"]
            assert current["execution_generation"] == 21
            assert current["execution_id"] != stale["execution_id"]
            assert current["run_id"] != stale["run_id"]


def test_bulk_adapter_contract_is_complete_and_each_missing_member_fails() -> None:
    engine.validate_bulk_adapter(successor)
    assert tuple(engine.BULK_ADAPTER_MEMBERS) == (
        "CANARY_GATE_SCHEMA",
        "COMMIT_RE",
        "CONTROLLERS",
        "RECONCILIATION_GATE_SCHEMA",
        "SHA256_RE",
        "build_runtime_plan",
        "load",
        "validate_all",
        "validate_inventory_gate",
    )
    for missing in engine.BULK_ADAPTER_MEMBERS:
        incomplete = type("IncompleteAdapter", (), {})()
        for name in engine.BULK_ADAPTER_MEMBERS:
            if name != missing:
                setattr(incomplete, name, getattr(successor, name))
        with pytest.raises(TypeError, match=missing):
            engine.validate_bulk_adapter(incomplete)


def test_held_render_and_canary_are_create_once_and_not_launch_authorized() -> None:
    plans = successor.build_plans(ROOT)
    sources, _ = package.package_sources(ROOT, plans)
    held = successor.load(ROOT / successor.HELD_PATH)
    successor.validate_held(held, plans, sources)
    rendered = package.render(ROOT)
    assert len(rendered["items"]) == 4
    jobs = [item for item in rendered["items"] if item["kind"] == "Job"]
    assert all(
        job["metadata"]["annotations"]["cyber-post-train.fleet.ai/launch-authorized"]
        == "false"
        for job in jobs
    )
    assert all(job["metadata"]["name"].endswith("g21-v1") for job in jobs)
    assert all("g20-v2" not in job["metadata"]["name"] for job in jobs)

    canary = package.render_runtime_gate_canary(ROOT)
    _, canary_job = canary["items"]
    annotations = canary_job["metadata"]["annotations"]
    assert canary_job["metadata"]["name"] == successor.RUNTIME_GATE_V3_CANARY_JOB
    assert annotations["cyber-post-train.fleet.ai/create-once"] == "true"
    assert annotations["cyber-post-train.fleet.ai/score-free"] == "true"
    assert annotations["cyber-post-train.fleet.ai/launch-authorized"] == "false"
    assert all(
        row["name"] != "FLEET_API_KEY"
        for row in canary_job["spec"]["template"]["spec"]["containers"][0]["env"]
    )


def test_v3_runtime_canary_receipt_binds_complete_adapter_and_zero_effects() -> None:
    plans = successor.build_plans(ROOT)
    sources, _ = package.package_sources(ROOT, plans)
    held = successor.load(ROOT / successor.HELD_PATH)
    receipt = successor.runtime_gate_v3_receipt(
        plans["qwen-a"],
        held,
        sources["qwen-a"],
        job_uid="11111111-1111-4111-8111-111111111111",
        pod_uid="22222222-2222-4222-8222-222222222222",
    )
    successor.validate_runtime_gate_v3_receipt(
        receipt,
        plans,
        sources,
        held_receipt_sha256=held["receipt_sha256"],
    )
    assert receipt["bulk_adapter_validated"] is True
    assert receipt["bulk_adapter_members"] == list(engine.BULK_ADAPTER_MEMBERS)
    for field in (
        "output_roots_created",
        "endpoint_leases_acquired",
        "canonical_claims_created",
        "model_calls",
        "task_calls",
        "session_calls",
        "verifier_calls",
        "scoring_calls",
        "api_mutations",
    ):
        assert receipt[field] == 0


def test_fresh_observer_binding_covers_failed_g20_and_new_g21_only() -> None:
    binding = gate_package.build_binding(ROOT)
    assert binding["binding_sha256"] == successor.RELEASE_GATE_BINDING["binding_sha256"]
    assert len(binding["identity_values"]) == 40
    assert len(binding["checked_sfs_roots"]) == 6
    assert {row["job_uid"] for row in binding["predecessor_objects"]} == {
        "c5d02921-3271-4186-89b8-65d227a44f2c",
        "ea058426-820e-4b0c-af9e-499f0350dbd7",
    }
    manifest = gate_package.render(ROOT)
    _, job = manifest["items"]
    assert job["metadata"]["name"] == "chris-q38-hosted-r15-r16-release-gate-v5"
    assert job["metadata"]["annotations"]["cyber-post-train.fleet.ai/score-free"] == "true"
    assert job["metadata"]["annotations"]["cyber-post-train.fleet.ai/launch-authorized"] == "false"
    assert job["spec"]["template"]["spec"]["preemptionPolicy"] == "Never"


def test_sanitized_incident_is_self_digest_valid_and_zero_effect() -> None:
    path = ROOT / successor.INCIDENT_EVIDENCE["path"]
    value = json.loads(path.read_text())
    assert value["receipt_sha256"] == successor.INCIDENT_EVIDENCE["receipt_sha256"]
    assert value["receipt_sha256"] == self_hosted.digest_without(value, "receipt_sha256")
    assert "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest() == successor.INCIDENT_EVIDENCE[
        "file_sha256"
    ]
    assert len(value["failed_objects"]) == 2
    assert all(row["production_sfs_root_absent"] is True for row in value["failed_objects"])
    assert value["failure"]["reservation_committed"] is False
    assert value["statistical_effects"] == {
        "accepted_receipts_created": 0,
        "model_calls": 0,
        "reservations_committed": 0,
        "scored_sessions_created": 0,
        "statistical_cells_consumed": 0,
    }
    assert value["privacy"] == {
        "credentials_included": False,
        "prompts_traces_flags_included": False,
        "scores_included": False,
        "source_logs_read": False,
    }


def test_release_stays_fail_closed_without_fresh_observer_and_v3_canary() -> None:
    plans = successor.build_plans(ROOT)
    sources, _ = package.package_sources(ROOT, plans)
    held = successor.load(ROOT / successor.HELD_PATH)
    candidate = copy.deepcopy(held)
    candidate.update(
        schema_version=successor.RELEASE_SCHEMA,
        status="CLEAR",
        launch_authorized=True,
        scoring_authorized=True,
    )
    candidate["receipt_sha256"] = self_hosted.digest_without(candidate, "receipt_sha256")
    with pytest.raises(RuntimeError, match="observation"):
        successor.validate_release(candidate, plans, sources)
