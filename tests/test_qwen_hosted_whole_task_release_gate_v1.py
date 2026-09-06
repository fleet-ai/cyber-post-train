from __future__ import annotations

import copy
import fcntl
import json
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from evals.fleet import qwen_hosted_whole_task_release_gate_package_v1 as gate_package
from evals.fleet import qwen_hosted_whole_task_release_gate_v1 as gate
from evals.fleet import qwen_hosted_whole_task_successor_v1 as prior
from evals.fleet import qwen_hosted_whole_task_successor_v2 as successor
from evals.fleet import qwen_hosted_whole_task_successor_v2_package as package
from evals.fleet import self_hosted

ROOT = Path(__file__).parents[1]


def test_generation20_changes_only_execution_and_object_identity() -> None:
    old = prior.build_plans(ROOT)
    new = successor.build_plans(ROOT)
    for controller in new:
        assert new[controller]["model"] == old[controller]["model"]
        assert new[controller]["harness"] == old[controller]["harness"]
        assert new[controller]["treatment"] == old[controller]["treatment"]
        assert new[controller]["tasks"] == old[controller]["tasks"]
        assert new[controller]["execution"] == old[controller]["execution"]
        assert new[controller]["campaign_id"] != old[controller]["campaign_id"]
        for fresh, stale in zip(
            new[controller]["attempts"], old[controller]["attempts"], strict=True
        ):
            assert fresh["cell_id"] == stale["cell_id"]
            assert fresh["selection_rank"] == stale["selection_rank"]
            assert fresh["attempt"] == stale["attempt"]
            assert fresh["task_version_id"] == stale["task_version_id"]
            assert fresh["execution_generation"] == 20
            assert fresh["execution_id"] != stale["execution_id"]
            assert fresh["run_id"] != stale["run_id"]


def test_canary_v5_terminal_evidence_is_zero_effect_and_digest_valid() -> None:
    value = successor.load(ROOT / successor.CANARY_PASS["path"])
    assert value["receipt_sha256"] == successor.CANARY_PASS["receipt_sha256"]
    assert value["receipt_sha256"] == self_hosted.digest_without(value, "receipt_sha256")
    assert value["status"] == "PASS"
    assert value["job"]["uid"] == "b17cd3d6-0b72-485d-a90c-d8cbb1c0de63"
    assert value["pod"]["uid"] == "09e66c45-3742-4289-9141-9b1339859c65"
    assert value["pod"]["restart_count"] == 0
    for field in (
        "api_mutations",
        "canonical_claims_created",
        "endpoint_leases_acquired",
        "model_calls",
        "output_roots_created",
        "scoring_calls",
        "session_calls",
        "task_calls",
        "verifier_calls",
    ):
        assert value[field] == 0


def test_held_scored_successors_cannot_launch_before_observer() -> None:
    plans = successor.build_plans(ROOT)
    sources, _ = package.package_sources(ROOT, plans)
    held = successor.load(ROOT / successor.HELD_PATH)
    successor.validate_held(held, plans, sources)
    rendered = package.render(ROOT)
    jobs = [item for item in rendered["items"] if item["kind"] == "Job"]
    assert held["launch_authorized"] is False
    assert held["scoring_authorized"] is False
    assert len(jobs) == 2
    assert all(
        job["metadata"]["annotations"]["cyber-post-train.fleet.ai/launch-authorized"] == "false"
        for job in jobs
    )
    assert all(
        "apt-get" not in job["spec"]["template"]["spec"]["containers"][0]["args"][0] for job in jobs
    )


def test_release_gate_binding_covers_old_and_new_cell_lineage() -> None:
    binding = gate_package.build_binding(ROOT)
    gate.validate_binding(binding)
    assert binding["statistical_cell_count"] == 8
    assert len(binding["predecessor_objects"]) == 2
    assert len(binding["fresh_objects"]) == 2
    assert len(binding["checked_sfs_roots"]) == 6
    assert all(
        plan["sfs_root"] + "-diagnostic" in binding["checked_sfs_roots"]
        for plan in successor.build_plans(ROOT).values()
    )
    assert len(binding["identity_values"]) == 40
    assert binding["binding_sha256"] == gate.binding_digest(binding)
    for field, expected in successor.RELEASE_GATE_BINDING.items():
        assert binding[field] == expected


def test_real_filesystem_collision_and_lease_checks(tmp_path: Path) -> None:
    identities = {"sha256:" + "a" * 64}
    claims = tmp_path / "claims"
    jobs = tmp_path / "jobs"
    leases = tmp_path / "leases" / "endpoint"
    claims.mkdir()
    jobs.mkdir()
    leases.mkdir(parents=True)
    (claims / "unrelated.json").write_text(json.dumps({"cell_id": "sha256:" + "b" * 64}))
    (jobs / "ACCEPTED.json").write_text(json.dumps({"cell_id": "sha256:" + "c" * 64}))
    for slot in (1, 2):
        (leases / f"slot-{slot}.lock").touch()
    assert gate._file_identity_collisions(claims, identities, accepted_only=False) == (1, 0)  # noqa: SLF001
    assert gate._file_identity_collisions(jobs, identities, accepted_only=True) == (1, 0)  # noqa: SLF001
    assert (
        gate._lease_slots_clear(  # noqa: SLF001
            {"lease_root": str(tmp_path / "leases"), "endpoint_key": "endpoint"}
        )
        == 2
    )
    with (leases / "slot-1.lock").open("r+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(gate.GateError, match="endpoint_lease_slot_held"):
            gate._lease_slots_clear(  # noqa: SLF001
                {"lease_root": str(tmp_path / "leases"), "endpoint_key": "endpoint"}
            )


def test_observer_manifest_is_create_once_score_free_and_read_only() -> None:
    rendered = gate_package.render(ROOT)
    configmap, job = rendered["items"]
    assert configmap["immutable"] is True
    assert job["metadata"]["name"] == gate_package.JOB_NAME
    assert job["metadata"]["annotations"]["cyber-post-train.fleet.ai/create-once"] == "true"
    assert job["metadata"]["annotations"]["cyber-post-train.fleet.ai/launch-authorized"] == "false"
    assert job["metadata"]["annotations"]["cyber-post-train.fleet.ai/score-free"] == "true"
    assert job["spec"]["backoffLimit"] == 0
    assert job["spec"]["template"]["spec"]["preemptionPolicy"] == "Never"
    raw = json.dumps(rendered)
    assert re.search(r"sk_[A-Za-z0-9]{12,}", raw) is None
    assert "prompt" not in configmap["data"]["binding.json"]
    assert "score" not in configmap["data"]["binding.json"]


def test_release_stays_closed_without_real_observer_receipt() -> None:
    plans = successor.build_plans(ROOT)
    sources, _ = package.package_sources(ROOT, plans)
    held = successor.load(ROOT / successor.HELD_PATH)
    fake = copy.deepcopy(held)
    fake.update(
        schema_version=successor.RELEASE_SCHEMA,
        status="CLEAR",
        launch_authorized=True,
        scoring_authorized=True,
    )
    with pytest.raises(RuntimeError, match="release-gate observation"):
        successor.validate_release(fake, plans, sources)


def _observation(binding: dict[str, object], observed_at: datetime) -> dict[str, object]:
    body = {
        "schema_version": gate.SCHEMA,
        "status": "CLEAR_PENDING_TERMINAL_JOB_VALIDATION",
        "observed_at_utc": observed_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "binding_sha256": binding["binding_sha256"],
        "statistical_cell_count": 8,
        "controller_count": 2,
        "predecessor_object_set_sha256": binding["predecessor_object_set_sha256"],
        "fresh_object_set_sha256": binding["fresh_object_set_sha256"],
        "plan_set_sha256": binding["plan_set_sha256"],
        "collisions": {
            "canonical_claims": 0,
            "accepted_receipts": 0,
            "authoritative_sessions": 0,
            "fresh_kubernetes_objects": 0,
            "sfs_output_roots": 0,
        },
        "observed_aggregates": {
            "canonical_claim_receipts_examined": 1,
            "accepted_receipts_examined": 1,
            "authoritative_session_rows_examined": 1,
            "predecessor_jobs_exclusively_failed": 2,
            "predecessor_pods_terminal_restart_zero": 2,
            "predecessor_immutable_configmaps_bound": 2,
            "fresh_object_sets_absent": 2,
            "checked_sfs_roots_absent": 6,
            "endpoint_lease_slots_simultaneously_free": 2,
        },
        "request_counts": {
            "fleet_account_gets": 1,
            "fleet_session_inventory_gets": 2,
            "kubernetes_gets": 12,
            "transcript_prompt_task_verifier_or_score_gets": 0,
        },
        "runtime": {
            "namespace": gate.NAMESPACE,
            "job_uid": "11111111-1111-4111-8111-111111111111",
            "pod_uid": "22222222-2222-4222-8222-222222222222",
        },
        "methods": ["GET"],
        "model_calls": 0,
        "task_calls": 0,
        "session_mutations": 0,
        "verifier_calls": 0,
        "scoring_calls": 0,
        "api_mutations": 0,
        "scores_included": False,
        "prompts_traces_flags_included": False,
        "credentials_included": False,
    }
    return gate._seal(body)  # noqa: SLF001


def test_full_observation_contract_rejects_rehashed_wrong_binding_and_stale_receipt() -> None:
    binding = gate_package.build_binding(ROOT)
    now = datetime.now(UTC).replace(microsecond=0)
    valid = _observation(binding, now)
    gate.validate_observation(valid, ROOT, binding=binding)
    wrong = copy.deepcopy(valid)
    wrong["binding_sha256"] = "sha256:" + "0" * 64
    wrong["receipt_sha256"] = gate.digest(wrong)
    with pytest.raises(RuntimeError, match="release-gate observation drifted"):
        successor.validate_release(
            {"release_gate_observation": wrong},
            successor.build_plans(ROOT),
            {},
            now=now,
        )
    with pytest.raises(RuntimeError, match="observation is stale"):
        successor.validate_release(
            {"release_gate_observation": _observation(binding, now - timedelta(hours=2))},
            successor.build_plans(ROOT),
            {},
            now=now,
        )


def test_v2_canary_uses_exact_production_package_bytes_and_runtime_entry() -> None:
    production = package.render(ROOT)
    canary = package.render_runtime_gate_canary(ROOT)
    assert canary["items"][0]["data"] == production["items"][0]["data"]
    job = canary["items"][1]
    assert job["metadata"]["name"] == package.RUNTIME_GATE_CANARY_JOB
    assert job["metadata"]["annotations"]["cyber-post-train.fleet.ai/score-free"] == "true"
    raw = json.dumps(job)
    assert "run_qwen_hosted_whole_task_successor_v2.sh" in raw
    assert "FLEET_API_KEY" not in raw


def test_last_moment_claim_scan_rejects_alternate_generation_same_cell(tmp_path: Path) -> None:
    plan = successor.build_plans(ROOT)["qwen-a"]
    jobs = tmp_path / "jobs"
    claims = tmp_path / "claims"
    reservations = tmp_path / "reservations"
    jobs.mkdir()
    claims.mkdir()
    collision = {
        "cell_id": plan["attempts"][0]["cell_id"],
        "execution_id": "sha256:" + "9" * 64,
    }
    (claims / "alternate-generation.json").write_text(json.dumps(collision))
    provider = successor.AtomicWholeTaskClaims(
        plan,
        tmp_path / "out",
        key="unused",
        jobs_root=jobs,
        reservation_root=reservations,
        session_check=lambda _config, _key: None,
    )
    with pytest.raises(RuntimeError, match="statistical cell claim collision"):
        provider._fresh_checks(claims)  # noqa: SLF001
