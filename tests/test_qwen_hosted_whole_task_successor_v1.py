from __future__ import annotations

import copy
import fcntl
import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from evals.fleet import exact_pass4_bulk_runtime_v3 as engine
from evals.fleet import qwen_hosted_generation19_v4 as source
from evals.fleet import qwen_hosted_prebootstrap_v4 as prebootstrap
from evals.fleet import qwen_hosted_private_stage_v5 as private_stage
from evals.fleet import qwen_hosted_whole_task_successor_v1 as successor
from evals.fleet import qwen_hosted_whole_task_successor_v1_package as package
from evals.fleet import qwen_hosted_whole_task_successor_v1_runtime as runtime
from evals.fleet import self_hosted

ROOT = Path(__file__).parents[1]


def _sources(plans: dict[str, dict]) -> dict[str, dict]:
    sources, _ = package.package_sources(ROOT, plans)
    return sources


def _lease_observer() -> dict:
    return successor._seal(  # noqa: SLF001
        {
            "schema_version": successor.LEASE_OBSERVER_SCHEMA,
            "status": "BOTH_SLOTS_FREE",
            "observer_pod_uid": "33333333-3333-4333-8333-333333333333",
            "lease_root_sha256": self_hosted.sha256(str(successor.ENDPOINT_LEASE_ROOT).encode()),
            "endpoint_key_sha256": self_hosted.sha256(successor.ENDPOINT_KEY.encode()),
            "slots": [
                {
                    "slot": 1,
                    "device": 1,
                    "inode": 101,
                    "size": 0,
                    "path_sha256": self_hosted.sha256(
                        str(
                            successor.ENDPOINT_LEASE_ROOT / successor.ENDPOINT_KEY / "slot-1.lock"
                        ).encode()
                    ),
                    "file_sha256": self_hosted.sha256(b""),
                },
                {
                    "slot": 2,
                    "device": 1,
                    "inode": 102,
                    "size": 0,
                    "path_sha256": self_hosted.sha256(
                        str(
                            successor.ENDPOINT_LEASE_ROOT / successor.ENDPOINT_KEY / "slot-2.lock"
                        ).encode()
                    ),
                    "file_sha256": self_hosted.sha256(b""),
                },
            ],
            "simultaneous_nonblocking_exclusive_acquisition": True,
            "all_locks_released": True,
            "files_created": 0,
            "files_deleted": 0,
            "api_mutations": 0,
            "scores_included": False,
            "prompts_or_traces_included": False,
        }
    )


def _runtime_gate_canary(plans: dict[str, dict], sources: dict[str, dict]) -> dict:
    return successor._seal(  # noqa: SLF001
        {
            "schema_version": successor.RUNTIME_GATE_CANARY_SCHEMA,
            "status": "PASS",
            "controller": "qwen-a",
            "plan_sha256": plans["qwen-a"]["plan_sha256"],
            "job_uid": "44444444-4444-4444-8444-444444444444",
            "pod_uid": "55555555-5555-4555-8555-555555555555",
            "authority_schema_version": successor.HELD_SCHEMA,
            "authority_receipt_sha256": "sha256:" + "a" * 64,
            "package_source_receipt_sha256": sources["qwen-a"]["receipt_sha256"],
            "bootstrap_stage_reached": "06-runtime-exec",
            "output_roots_created": 0,
            "endpoint_leases_acquired": 0,
            "canonical_claims_created": 0,
            "model_calls": 0,
            "task_calls": 0,
            "session_calls": 0,
            "verifier_calls": 0,
            "scoring_calls": 0,
            "api_mutations": 0,
            "scores_included": False,
            "prompts_or_traces_included": False,
            "credentials_included": False,
        }
    )


def _release(plans: dict[str, dict], sources: dict[str, dict] | None = None) -> dict:
    sources = sources or _sources(plans)
    body = {
        "schema_version": successor.RELEASE_SCHEMA,
        "status": "CLEAR",
        "checked_at_utc": "2026-09-06T08:40:00Z",
        "launch_authorized": True,
        "scoring_authorized": True,
        "controller_cap": 2,
        "endpoint_maximum_streams": 2,
        "controllers": successor.release_projection(plans, sources),
        "ledger_snapshot_path": successor.LEDGER_PATH,
        "ledger_snapshot_receipt_sha256": successor.LEDGER_SELF_SHA256,
        "ledger_snapshot_file_sha256": successor.LEDGER_FILE_SHA256,
        "predecessor_tombstones": successor.PREDECESSOR_TOMBSTONES,
        "endpoint_lease_observer": _lease_observer(),
        "runtime_gate_canary": _runtime_gate_canary(plans, sources),
        "predecessor_disposition": {
            "retry_forbidden_selection_ranks": [13, 14],
            "prior_job_uids": [
                "515c370a-ecb4-4c41-a349-46a328c8fa68",
                "f23805b5-516b-4828-a3b5-82673a1b3e2f",
            ],
            "prior_pod_uids": [
                "4d86f0c5-7cef-4f8a-a798-ea5c725e4955",
                "d5d7b7cf-d639-4ad6-b012-3001c4935b2d",
            ],
            "prior_jobs_terminal": True,
            "prior_pods_absent": True,
            "relevant_active_hosted_q_jobs": 0,
            "relevant_active_hosted_q_pods": 0,
            "new_jobs_absent": True,
            "new_configmaps_absent": True,
            "kubernetes_api_mutations": 0,
        },
        "fresh_collision_reconciliation": {
            "checked_immediately_before_release": True,
            "checked_from_uid_bound_sfs_pod": True,
            "observer_pod_uid": "33333333-3333-4333-8333-333333333333",
            "observed_cells": 8,
            "canonical_claim_collisions": 0,
            "authoritative_session_collisions": 0,
            "accepted_evidence_collisions": 0,
            "output_root_collisions": 0,
            "api_mutations": 0,
        },
        "privacy": {
            "scores_read": False,
            "prompts_traces_flags_read": False,
            "credentials_included": False,
        },
    }
    return successor._seal(body)  # noqa: SLF001


def test_plans_select_only_two_untouched_complete_tasks() -> None:
    plans = successor.build_plans(ROOT)
    predecessors = source.validate_all(ROOT)
    assert [(key, len(plan["attempts"])) for key, plan in plans.items()] == [
        ("qwen-a", 4),
        ("qwen-b", 4),
    ]
    for controller, plan in plans.items():
        rank = successor.CONTROLLERS[controller]["rank"]
        expected = [
            row for row in predecessors[controller]["attempts"] if row["selection_rank"] == rank
        ]
        assert plan["attempts"] == expected
        assert plan["model"] == predecessors[controller]["model"]
        assert plan["harness"] == predecessors[controller]["harness"]
        assert plan["treatment"] == predecessors[controller]["treatment"]
        assert plan["authority"] == predecessors[controller]["authority"]
        assert plan["execution"] == predecessors[controller]["execution"]
        assert plan["execution"]["endpoint_lease"]["maximum_streams"] == 2
    assert {row["selection_rank"] for plan in plans.values() for row in plan["attempts"]} == {
        15,
        16,
    }


def test_held_evidence_is_digest_valid_and_never_launches() -> None:
    plans = successor.build_plans(ROOT)
    sources = _sources(plans)
    held = successor.load(ROOT / successor.HELD_PATH)
    successor.validate_held(held, plans, sources)
    assert held["receipt_sha256"] == self_hosted.digest_without(held, "receipt_sha256")
    assert held["launch_authorized"] is False
    assert held["scoring_authorized"] is False
    assert held["retry_forbidden_selection_ranks"] == [13, 14]


def test_preclaim_failure_receipt_is_score_blind_and_retry_closed() -> None:
    receipt = successor.load(ROOT / successor.PRECLAIM_FAILURE["path"])
    assert receipt["receipt_sha256"] == successor.PRECLAIM_FAILURE["receipt_sha256"]
    assert receipt["receipt_sha256"] == self_hosted.digest_without(receipt, "receipt_sha256")
    assert receipt["status"] == "INFRASTRUCTURE_INVALID_PRECLAIM"
    assert receipt["observed_failure_stage"] == {
        "classification": "PRE_RUNTIME_UNCLASSIFIED",
        "runtime_failed_receipts_present": 0,
        "bootstrap_stage_receipts_present": 0,
        "possible_boundaries": [
            "package-source-validation",
            "source-install",
            "docker-readiness",
            "image-build",
            "harness-version-validation",
            "pre-try-runtime-construction",
        ],
    }
    assert receipt["latent_runtime_gate_defect"]["classification"] == (
        "PROVEN_BY_STATIC_CODE_AND_REGRESSION"
    )
    assert receipt["latent_runtime_gate_defect"]["established_as_observed_exit"] is False
    assert receipt["score_blind_reconciliation"] == {
        "observer_pod_uid": "35a00aeb-fb66-4bc5-a7d2-1056337ade69",
        "canonical_claims_present": 0,
        "output_roots_present": 0,
        "accepted_evidence_matches": 0,
        "authoritative_session_matches": 0,
        "verifier_execution_matches": 0,
        "diagnostic_failure_receipts_present": 0,
        "model_calls": 0,
        "task_calls": 0,
        "session_calls": 0,
        "verifier_calls": 0,
        "scoring_calls": 0,
        "api_mutations": 0,
    }
    assert receipt["disposition"]["failed_job_identities_retry_authorized"] is False
    assert receipt["disposition"]["launch_authorized"] is False


def test_canary_v4_failure_is_stage06_projected_symlink_and_zero_effect() -> None:
    receipt = successor.load(ROOT / successor.CANARY_FAILURE["path"])
    assert receipt["receipt_sha256"] == successor.CANARY_FAILURE["receipt_sha256"]
    assert receipt["receipt_sha256"] == self_hosted.digest_without(receipt, "receipt_sha256")
    assert receipt["status"] == "FAILED_AT_BOOTSTRAP_STAGE_SIX"
    assert receipt["sanitized_failure_receipt"]["last_completed_stage"] == (
        "06-runtime-exec"
    )
    assert receipt["sanitized_failure_receipt"]["last_completed_phase"] == (
        "build-plans-done"
    )
    assert receipt["deterministic_static_diagnosis"]["cause_classification"] == (
        "PROJECTED_CONFIGMAP_SYMLINK_REJECTED_BY_STRICT_RUNTIME_LOADER"
    )
    assert receipt["deterministic_static_diagnosis"]["logs_used"] is False
    assert receipt["deterministic_correction"] == {
        "projected_files_are_transport_only": True,
        "private_destination_mode": "0700",
        "private_file_mode": "0600",
        "copy_is_create_once": True,
        "source_and_copy_bytes_match": True,
        "source_and_copy_self_digests_validated": True,
        "runtime_accepts_only_private_regular_files": True,
        "strict_runtime_loader_weakened": False,
    }
    assert set(receipt["score_blind_effects"].values()) == {0}
    assert receipt["disposition"]["canary_v4_identity_retry_authorized"] is False
    assert receipt["disposition"]["scored_successor_authorized"] is False


def test_score_free_runtime_gate_canary_stops_before_any_scored_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plans = successor.build_plans(ROOT)
    plan = plans["qwen-a"]
    package_source = _sources(plans)["qwen-a"]
    held = successor.load(ROOT / successor.HELD_PATH)
    monkeypatch.setattr(
        runtime, "_load_bound_inputs", lambda _plan, **_kwargs: (plans, package_source)
    )
    monkeypatch.setattr(
        runtime,
        "runtime_gate_check",
        lambda actual_plans, actual_source, *, canary=False, phase_callback=None: (
            held
            if actual_plans == plans and actual_source == package_source and canary
            else pytest.fail("runtime gate canary did not use the exact held gate")
        ),
    )
    monkeypatch.setattr(
        engine,
        "run_controller",
        lambda *_args, **_kwargs: pytest.fail("canary crossed into the scored engine"),
    )
    monkeypatch.setenv("JOB_UID", "11111111-1111-4111-8111-111111111111")
    monkeypatch.setenv("POD_UID", "22222222-2222-4222-8222-222222222222")
    receipt_path = tmp_path / "diagnostic" / "RUNTIME-GATE-CANARY.json"
    receipt = runtime.run_gate_canary(plan, receipt_path=receipt_path)
    successor.validate_runtime_gate_canary(receipt)
    assert successor.load(receipt_path) == receipt
    assert receipt["bootstrap_stage_reached"] == "06-runtime-exec"
    assert receipt["output_roots_created"] == 0
    assert receipt["endpoint_leases_acquired"] == 0
    assert receipt["canonical_claims_created"] == 0


def test_runtime_gate_canary_rejects_any_nonexact_held_authority_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    plans = successor.build_plans(ROOT)
    package_source = _sources(plans)["qwen-a"]
    held = successor.load(ROOT / successor.HELD_PATH)
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    held_path = private / "release.json"
    held_path.write_text(json.dumps(held))
    held_path.chmod(0o600)
    monkeypatch.setenv("QWEN_HOSTED_WHOLE_TASK_RELEASE_PATH", str(held_path.resolve()))
    monkeypatch.setenv("QWEN_HOSTED_WHOLE_TASK_RELEASE_SHA256", "sha256:" + "b" * 64)
    with pytest.raises(RuntimeError, match="canary held digest drifted"):
        runtime.runtime_gate_check(plans, package_source, canary=True)


def test_runtime_gate_failure_receipt_contains_only_sanitized_error_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("JOB_UID", "11111111-1111-4111-8111-111111111111")
    monkeypatch.setenv("POD_UID", "22222222-2222-4222-8222-222222222222")
    monkeypatch.setenv("QWEN_HOSTED_WHOLE_TASK_RELEASE_SHA256", "sha256:" + "a" * 64)
    monkeypatch.setenv("QWEN_HOSTED_WHOLE_TASK_PACKAGE_SOURCE_SHA256", "sha256:" + "b" * 64)
    path = tmp_path / "RUNTIME-GATE-CANARY-FAILED.json"
    receipt = runtime.write_gate_canary_failure(
        path,
        RuntimeError("protected detail"),
        last_completed_phase="package-source-digest-done",
    )
    assert successor.load(path) == receipt
    assert receipt["last_completed_stage"] == "06-runtime-exec"
    assert receipt["last_completed_phase"] == "package-source-digest-done"
    assert receipt["error_type"] == "RuntimeError"
    assert receipt["error_sha256"] == self_hosted.sha256(b"protected detail")
    assert "protected detail" not in path.read_text()


def test_scored_release_is_unconditionally_closed_for_consumed_identities() -> None:
    plans = successor.build_plans(ROOT)
    sources = _sources(plans)
    release = _release(plans, sources)
    with pytest.raises(RuntimeError, match="closed for consumed object and execution identities"):
        successor.validate_release(release, plans, sources)
    with pytest.raises(RuntimeError, match="closed for consumed object and execution identities"):
        successor.validate_release(successor.load(ROOT / successor.HELD_PATH), plans, sources)


def _lease_root(tmp_path: Path) -> tuple[Path, list[Path]]:
    root = tmp_path / "leases"
    endpoint = root / successor.ENDPOINT_KEY
    endpoint.mkdir(parents=True)
    paths = [endpoint / "slot-1.lock", endpoint / "slot-2.lock"]
    for path in paths:
        path.touch(mode=0o600)
    return root, paths


def test_persistent_lease_inodes_are_simultaneously_probed_and_never_deleted(
    tmp_path: Path,
) -> None:
    root, paths = _lease_root(tmp_path)
    before = [(path.stat().st_dev, path.stat().st_ino) for path in paths]
    receipt = successor.observe_endpoint_lease_slots(
        "33333333-3333-4333-8333-333333333333", lease_root=root
    )
    assert receipt["status"] == "BOTH_SLOTS_FREE"
    assert receipt["all_locks_released"] is True
    assert receipt["files_created"] == receipt["files_deleted"] == 0
    assert [(path.stat().st_dev, path.stat().st_ino) for path in paths] == before
    handles = [path.open("a+b") for path in paths]
    try:
        for handle in handles:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        for handle in reversed(handles):
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()


def test_held_second_slot_fails_and_releases_partial_first_slot(tmp_path: Path) -> None:
    root, paths = _lease_root(tmp_path)
    held = paths[1].open("a+b")
    fcntl.flock(held.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with pytest.raises(RuntimeError, match="lease slot is held"):
            successor.observe_endpoint_lease_slots(
                "33333333-3333-4333-8333-333333333333", lease_root=root
            )
        first = paths[0].open("a+b")
        try:
            fcntl.flock(first.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            fcntl.flock(first.fileno(), fcntl.LOCK_UN)
            first.close()
    finally:
        fcntl.flock(held.fileno(), fcntl.LOCK_UN)
        held.close()
    assert all(path.exists() for path in paths)


def test_lease_observer_fails_closed_on_path_and_inode_drift(tmp_path: Path) -> None:
    root, paths = _lease_root(tmp_path)
    paths[1].unlink()
    with pytest.raises(RuntimeError, match="inode drifted"):
        successor.observe_endpoint_lease_slots(
            "33333333-3333-4333-8333-333333333333", lease_root=root
        )
    paths[1].mkdir()
    with pytest.raises(RuntimeError, match="inode drifted"):
        successor.observe_endpoint_lease_slots(
            "33333333-3333-4333-8333-333333333333", lease_root=root
        )
    bad = _lease_observer()
    bad["slots"][1]["inode"] = bad["slots"][0]["inode"]
    bad["receipt_sha256"] = self_hosted.digest_without(bad, "receipt_sha256")
    with pytest.raises(RuntimeError, match="lease observer drifted"):
        successor.validate_endpoint_lease_observer(bad)
    bad = _lease_observer()
    bad["slots"][1]["path_sha256"] = bad["slots"][0]["path_sha256"]
    bad["receipt_sha256"] = self_hosted.digest_without(bad, "receipt_sha256")
    with pytest.raises(RuntimeError, match="lease observer drifted"):
        successor.validate_endpoint_lease_observer(bad)


def test_release_requires_zero_relevant_active_hosted_objects() -> None:
    plans = successor.build_plans(ROOT)
    sources = _sources(plans)
    release = _release(plans, sources)
    for field in ("relevant_active_hosted_q_jobs", "relevant_active_hosted_q_pods"):
        bad = copy.deepcopy(release)
        bad["predecessor_disposition"][field] = 1
        bad["receipt_sha256"] = self_hosted.digest_without(bad, "receipt_sha256")
        with pytest.raises(
            RuntimeError, match="closed for consumed object and execution identities"
        ):
            successor.validate_release(bad, plans, sources)


def test_atomic_provider_publishes_and_validates_all_four_before_return(
    tmp_path: Path,
) -> None:
    plan = successor.build_plans(ROOT)["qwen-a"]
    out = tmp_path / "out"
    out.mkdir()
    claim_root = tmp_path / "claims"
    claim_root.mkdir()
    (tmp_path / "jobs").mkdir()
    checked: list[str] = []
    provider = successor.AtomicWholeTaskClaims(
        plan,
        out,
        key="test-only",
        jobs_root=tmp_path / "jobs",
        reservation_root=tmp_path / "reservations",
        session_check=lambda config, _key: checked.append(config["run_id"]),
    )
    first = provider(
        plan,
        plan["attempts"][0],
        claim_root,
        "11111111-1111-4111-8111-111111111111",
        "22222222-2222-4222-8222-222222222222",
    )
    assert first == provider.claims[plan["attempts"][0]["run_id"]]
    assert checked == [row["run_id"] for row in plan["attempts"]]
    assert len(list(claim_root.glob("*.json"))) == 4
    reservation = successor.load(out / "RESERVATION.json")
    assert reservation["all_claims_before_model_call"] is True
    assert reservation["all_claims_byte_validated"] is True
    assert reservation["claim_sha256s"] == [
        provider.claims[row["run_id"]]["receipt_sha256"] for row in plan["attempts"]
    ]
    assert reservation["receipt_sha256"] == self_hosted.digest_without(
        reservation, "receipt_sha256"
    )


def test_partial_claim_publication_rolls_back_without_touching_existing_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = successor.build_plans(ROOT)["qwen-b"]
    out = tmp_path / "out"
    out.mkdir()
    claim_root = tmp_path / "claims"
    claim_root.mkdir()
    unrelated = claim_root / "unrelated.json"
    unrelated.write_text("{}\n")
    (tmp_path / "jobs").mkdir()
    real = engine.claim_cell
    calls = 0

    def collide_on_second(*args: object, **kwargs: object) -> dict | None:
        nonlocal calls
        calls += 1
        if calls == 2:
            return None
        return real(*args, **kwargs)

    monkeypatch.setattr(engine, "claim_cell", collide_on_second)
    provider = successor.AtomicWholeTaskClaims(
        plan,
        out,
        key="test-only",
        jobs_root=tmp_path / "jobs",
        reservation_root=tmp_path / "reservations",
        session_check=lambda *_args: None,
    )
    with pytest.raises(RuntimeError, match="transaction collided"):
        provider(
            plan,
            plan["attempts"][0],
            claim_root,
            "11111111-1111-4111-8111-111111111111",
            "22222222-2222-4222-8222-222222222222",
        )
    assert list(claim_root.glob("*.json")) == [unrelated]
    assert not (out / "RESERVATION.json").exists()


@pytest.mark.parametrize(
    "stage",
    [
        "after_preparing",
        "after_claim_1",
        "after_claim_2",
        "after_claim_3",
        "after_claim_4",
        "after_validation_1",
        "after_validation_2",
        "after_validation_3",
        "after_validation_4",
        "before_reservation_write",
        "after_reservation_write",
    ],
)
@pytest.mark.parametrize("error_type", [RuntimeError, SystemExit])
def test_every_transaction_boundary_is_rollback_or_restart_recoverable(
    tmp_path: Path, stage: str, error_type: type[BaseException]
) -> None:
    plan = successor.build_plans(ROOT)["qwen-a"]
    out = tmp_path / "out"
    out.mkdir()
    claim_root = tmp_path / "claims"
    claim_root.mkdir()
    jobs_root = tmp_path / "jobs"
    jobs_root.mkdir()
    reservations = tmp_path / "reservations"

    def fail_at(observed: str) -> None:
        if observed == stage:
            raise error_type(stage)

    interrupted = successor.AtomicWholeTaskClaims(
        plan,
        out,
        key="test-only",
        jobs_root=jobs_root,
        reservation_root=reservations,
        session_check=lambda *_args: None,
        fault_hook=fail_at,
    )
    with pytest.raises(error_type, match=stage):
        interrupted(
            plan,
            plan["attempts"][0],
            claim_root,
            "11111111-1111-4111-8111-111111111111",
            "22222222-2222-4222-8222-222222222222",
        )
    committed = stage == "after_reservation_write"
    assert len(list(claim_root.glob("*.json"))) == (4 if committed else 0)
    assert (out / "RESERVATION.json").exists() is committed
    assert list((reservations / "preparing" / interrupted.task_version).glob("*.json"))

    recovered = successor.AtomicWholeTaskClaims(
        plan,
        out,
        key="test-only",
        jobs_root=jobs_root,
        reservation_root=reservations,
        session_check=lambda *_args: None,
    )
    recovered(
        plan,
        plan["attempts"][0],
        claim_root,
        "11111111-1111-4111-8111-111111111111",
        "22222222-2222-4222-8222-222222222222",
    )
    assert len(list(claim_root.glob("*.json"))) == 4
    assert (out / "RESERVATION.json").is_file()


def test_torn_reservation_write_rolls_back_claims_and_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = successor.build_plans(ROOT)["qwen-b"]
    out = tmp_path / "out"
    out.mkdir()
    claim_root = tmp_path / "claims"
    claim_root.mkdir()
    jobs_root = tmp_path / "jobs"
    jobs_root.mkdir()
    real = engine._write_once  # noqa: SLF001

    def torn(path: Path, value: dict) -> None:
        if path.name == "RESERVATION.json":
            path.write_bytes(b'{"torn":')
            raise OSError("injected torn reservation")
        real(path, value)

    monkeypatch.setattr(engine, "_write_once", torn)
    provider = successor.AtomicWholeTaskClaims(
        plan,
        out,
        key="test-only",
        jobs_root=jobs_root,
        reservation_root=tmp_path / "reservations",
        session_check=lambda *_args: None,
    )
    with pytest.raises(OSError, match="torn reservation"):
        provider(
            plan,
            plan["attempts"][0],
            claim_root,
            "11111111-1111-4111-8111-111111111111",
            "22222222-2222-4222-8222-222222222222",
        )
    assert not list(claim_root.glob("*.json"))
    assert not (out / "RESERVATION.json").exists()


def test_restart_recovers_crash_like_preparing_and_partial_claim_state(tmp_path: Path) -> None:
    plan = successor.build_plans(ROOT)["qwen-b"]
    out = tmp_path / "out"
    out.mkdir()
    claim_root = tmp_path / "claims"
    claim_root.mkdir()
    jobs_root = tmp_path / "jobs"
    jobs_root.mkdir()
    reservations = tmp_path / "reservations"
    crashed = successor.AtomicWholeTaskClaims(
        plan,
        out,
        key="test-only",
        jobs_root=jobs_root,
        reservation_root=reservations,
        session_check=lambda *_args: None,
    )
    job_uid = "11111111-1111-4111-8111-111111111111"
    old_pod_uid = "22222222-2222-4222-8222-222222222222"
    crashed._ensure_preparing(job_uid, old_pod_uid)  # noqa: SLF001
    for item in plan["attempts"][-2:]:
        assert engine.claim_cell(
            plan,
            item,
            claim_root=claim_root,
            job_uid=job_uid,
            pod_uid=old_pod_uid,
        )
    assert len(list(claim_root.glob("*.json"))) == 2

    restarted = successor.AtomicWholeTaskClaims(
        plan,
        out,
        key="test-only",
        jobs_root=jobs_root,
        reservation_root=reservations,
        session_check=lambda *_args: None,
    )
    restarted(
        plan,
        plan["attempts"][0],
        claim_root,
        job_uid,
        "33333333-3333-4333-8333-333333333333",
    )
    assert len(list(claim_root.glob("*.json"))) == 4
    assert {successor.load(path)["pod_uid"] for path in claim_root.glob("*.json")} == {
        "33333333-3333-4333-8333-333333333333"
    }
    assert (out / "RESERVATION.json").is_file()


def test_missing_canonical_mount_fails_before_claim(tmp_path: Path) -> None:
    plan = successor.build_plans(ROOT)["qwen-a"]
    out = tmp_path / "out"
    out.mkdir()
    provider = successor.AtomicWholeTaskClaims(
        plan,
        out,
        key="test-only",
        jobs_root=tmp_path / "missing-jobs",
        reservation_root=tmp_path / "reservations",
        session_check=lambda *_args: None,
    )
    with pytest.raises(RuntimeError, match="canonical jobs root is unavailable"):
        provider(
            plan,
            plan["attempts"][0],
            tmp_path / "missing-claims",
            "11111111-1111-4111-8111-111111111111",
            "22222222-2222-4222-8222-222222222222",
        )
    assert not (tmp_path / "missing-claims").exists()


def test_engine_reaches_model_boundary_only_after_four_claims(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = copy.deepcopy(successor.build_plans(ROOT)["qwen-a"])
    plan["execution"]["endpoint_lease"]["lease_root"] = str(tmp_path / "leases")
    plan["plan_sha256"] = self_hosted.digest_without(plan, "plan_sha256")
    monkeypatch.setattr(successor, "build_runtime_plan", lambda *_args: plan)
    out = tmp_path / "out"
    claim_root = tmp_path / "claims"
    claim_root.mkdir()
    (tmp_path / "jobs").mkdir()
    provider = successor.AtomicWholeTaskClaims(
        plan,
        out,
        key="test-only",
        jobs_root=tmp_path / "jobs",
        reservation_root=tmp_path / "reservations",
        session_check=lambda *_args: None,
    )

    def stop_at_model(*_args: object) -> dict:
        assert len(list(claim_root.glob("*.json"))) == 4
        assert (out / "RESERVATION.json").is_file()
        assert len(list((out / "model-boundaries").glob("*.json"))) == 1
        raise SystemExit("model boundary reached")

    def observe(stage: str, item: dict | None) -> None:
        if stage == "07-claim-written":
            assert item is not None
            provider.mark_model_boundary(item)

    prior = engine.bulk
    engine.bulk = successor
    monkeypatch.setenv("FLEET_API_KEY", "test-only")
    monkeypatch.setenv("JOB_UID", "11111111-1111-4111-8111-111111111111")
    monkeypatch.setenv("POD_UID", "22222222-2222-4222-8222-222222222222")
    try:
        with pytest.raises(SystemExit, match="model boundary"):
            engine.run_controller(
                plan,
                out=out,
                proxy=tmp_path / "proxy.py",
                claim_root=claim_root,
                model_runner=stop_at_model,
                classifier=lambda *_args: {},
                check_run_absent=lambda *_args: None,
                route_check=lambda *_args: None,
                runtime_gate_check=lambda *_args: None,
                claim_provider=provider,
                stage_observer=observe,
            )
    finally:
        engine.bulk = prior


def test_production_runtime_gate_revalidates_exact_release_signature(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plans = successor.build_plans(ROOT)
    plan = plans["qwen-a"]
    sources = _sources(plans)
    package_source = sources["qwen-a"]
    release = _release(plans, sources)
    calls: list[str] = []
    validator_calls: list[tuple[dict, dict, dict]] = []

    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    package_source_path = private / "package-source.json"
    package_source_path.write_text(json.dumps(package_source))
    package_source_path.chmod(0o600)
    release_path = private / "release.json"
    release_path.write_text(json.dumps(release))
    release_path.chmod(0o600)

    def run_controller(actual_plan: dict, *, runtime_gate_check: object, **_kwargs: object) -> dict:
        assert actual_plan == plan
        runtime_gate_check(actual_plan)  # type: ignore[operator]
        calls.append("production-runtime-gate")
        return {"status": "gate-passed"}

    monkeypatch.setattr(successor, "build_plans", lambda _root: plans)
    monkeypatch.setattr(
        successor,
        "validate_release",
        lambda actual_release, actual_plans, actual_sources: validator_calls.append(
            (actual_release, actual_plans, actual_sources)
        ),
    )
    monkeypatch.setattr(engine, "run_controller", run_controller)
    monkeypatch.setenv("FLEET_API_KEY", "test-only")
    monkeypatch.setenv("QWEN_HOSTED_WHOLE_TASK_RELEASE_PATH", str(release_path.resolve()))
    monkeypatch.setenv("QWEN_HOSTED_WHOLE_TASK_RELEASE_SHA256", release["receipt_sha256"])
    monkeypatch.setenv(
        "QWEN_HOSTED_WHOLE_TASK_PACKAGE_SOURCE_SHA256",
        package_source["receipt_sha256"],
    )
    result = runtime.run(
        plan,
        out=tmp_path / "out",
        proxy=tmp_path / "proxy.py",
        diagnostic_root=tmp_path / "diagnostic",
        package_source_path=package_source_path.resolve(),
    )
    assert result == {"status": "gate-passed"}
    assert calls == ["production-runtime-gate"]
    assert len(validator_calls) == 2
    assert all(call[0] == release and call[1] == plans for call in validator_calls)
    assert all(set(call[2]) == set(plans) for call in validator_calls)


def test_restart_never_repeats_a_durable_model_boundary(tmp_path: Path) -> None:
    plan = successor.build_plans(ROOT)["qwen-a"]
    out = tmp_path / "out"
    out.mkdir()
    claim_root = tmp_path / "claims"
    claim_root.mkdir()
    jobs_root = tmp_path / "jobs"
    jobs_root.mkdir()
    reservations = tmp_path / "reservations"
    first = successor.AtomicWholeTaskClaims(
        plan,
        out,
        key="test-only",
        jobs_root=jobs_root,
        reservation_root=reservations,
        session_check=lambda *_args: None,
    )
    item = plan["attempts"][0]
    first(
        plan,
        item,
        claim_root,
        "11111111-1111-4111-8111-111111111111",
        "22222222-2222-4222-8222-222222222222",
    )
    first.mark_model_boundary(item)

    restarted = successor.AtomicWholeTaskClaims(
        plan,
        out,
        key="test-only",
        jobs_root=jobs_root,
        reservation_root=reservations,
        session_check=lambda *_args: None,
    )
    restarted(
        plan,
        item,
        claim_root,
        "11111111-1111-4111-8111-111111111111",
        "33333333-3333-4333-8333-333333333333",
    )
    with pytest.raises(RuntimeError, match="already crossed; retry prohibited"):
        restarted.mark_model_boundary(item)


def test_reservation_drift_after_model_boundary_never_rolls_back(tmp_path: Path) -> None:
    plan = successor.build_plans(ROOT)["qwen-a"]
    out = tmp_path / "out"
    out.mkdir()
    claim_root = tmp_path / "claims"
    claim_root.mkdir()
    jobs_root = tmp_path / "jobs"
    jobs_root.mkdir()
    reservations = tmp_path / "reservations"
    provider = successor.AtomicWholeTaskClaims(
        plan,
        out,
        key="test-only",
        jobs_root=jobs_root,
        reservation_root=reservations,
        session_check=lambda *_args: None,
    )
    item = plan["attempts"][0]
    provider(
        plan,
        item,
        claim_root,
        "11111111-1111-4111-8111-111111111111",
        "22222222-2222-4222-8222-222222222222",
    )
    provider.mark_model_boundary(item)
    (out / "RESERVATION.json").write_text("{}\n")

    restarted = successor.AtomicWholeTaskClaims(
        plan,
        out,
        key="test-only",
        jobs_root=jobs_root,
        reservation_root=reservations,
        session_check=lambda *_args: None,
    )
    with pytest.raises(RuntimeError, match="drifted after model boundary"):
        restarted(
            plan,
            item,
            claim_root,
            "11111111-1111-4111-8111-111111111111",
            "33333333-3333-4333-8333-333333333333",
        )
    assert len(list(claim_root.glob("*.json"))) == 4
    assert (out / "RESERVATION.json").read_text() == "{}\n"


def test_held_package_is_closed_create_once_and_cap_two(tmp_path: Path) -> None:
    rendered = package.render(ROOT)
    assert [item["kind"] for item in rendered["items"]] == [
        "ConfigMap",
        "Job",
        "ConfigMap",
        "Job",
    ]
    for cm, job in zip(rendered["items"][::2], rendered["items"][1::2], strict=True):
        assert cm["immutable"] is True
        assert job["metadata"]["annotations"] == {
            "cyber-post-train.fleet.ai/create-once": "true",
            "cyber-post-train.fleet.ai/launch-authorized": "false",
        }
        assert job["spec"]["backoffLimit"] == 0
        assert job["spec"]["template"]["spec"]["restartPolicy"] == "Never"
        assert job["spec"]["activeDeadlineSeconds"] >= 4 * 28_800
        plan = json.loads(cm["data"]["plan.json"])
        source_receipt = json.loads(cm["data"]["package-source.json"])
        source_data = {
            name: value
            for name, value in cm["data"].items()
            if name not in {"release.json", "package-source.json"}
        }
        assert source_receipt == successor.package_source_receipt(
            plan["controller"], plan, source_data
        )
        source_env = next(
            row["value"]
            for row in job["spec"]["template"]["spec"]["containers"][0]["env"]
            if row["name"] == "QWEN_HOSTED_WHOLE_TASK_PACKAGE_SOURCE_SHA256"
        )
        assert source_env == source_receipt["receipt_sha256"]
        assert len(plan["attempts"]) == 4
        assert plan["execution"]["endpoint_lease"]["maximum_streams"] == 2
        assert [row["attempt"] for row in plan["attempts"]] == [1, 2, 3, 4]
        assert {row["selection_rank"] for row in plan["attempts"]}.isdisjoint({13, 14})
        bootstrap = cm["data"]["run_qwen_hosted_whole_task_successor_v1.sh"]
        assert "bootstrap_stage 00-started" in bootstrap
        assert "bootstrap_stage 06-runtime-exec" in bootstrap
        assert "fleet-qwen38-hosted-whole-task-bootstrap-stage-v1" in bootstrap
        assert "os.O_WRONLY | os.O_CREAT | os.O_EXCL" in bootstrap
        assert "QWEN_HOSTED_WHOLE_TASK_RUNTIME_GATE_CANARY" in bootstrap
        assert "RUNTIME-GATE-CANARY-FAILED.json" in bootstrap
        assert '"error_type": "ProcessExitNonzero"' in bootstrap

        module_root = tmp_path / cm["metadata"]["name"] / "evals" / "fleet"
        config_root = module_root / "configs"
        config_root.mkdir(parents=True)
        (module_root.parent / "__init__.py").touch()
        (module_root / "__init__.py").touch()
        for name, value in cm["data"].items():
            if name.endswith(".py"):
                (module_root / name).write_text(value)
        (config_root / "qwen-hosted-generation19-qwen-a-v4.json").write_text(
            cm["data"]["source-plan-v4-a.json"]
        )
        (config_root / "qwen-hosted-generation19-qwen-b-v4.json").write_text(
            cm["data"]["source-plan-v4-b.json"]
        )
        subprocess.run(
            [
                sys.executable,
                "-c",
                textwrap.dedent(
                    """
                    from pathlib import Path
                    from evals.fleet import qwen_hosted_whole_task_successor_v1 as successor
                    from evals.fleet import qwen_hosted_whole_task_successor_v1_runtime
                    plans = successor.build_plans(Path.cwd())
                    assert {len(plan['attempts']) for plan in plans.values()} == {4}
                    """
                ),
            ],
            cwd=module_root.parents[1],
            check=True,
            capture_output=True,
            text=True,
        )


def test_score_free_canary_and_held_scored_job_share_exact_bootstrap_bytes() -> None:
    held_cm, held_job = package.render(ROOT)["items"][:2]
    canary_cm, canary_job = package.render_runtime_gate_canary(ROOT)["items"]
    assert held_cm["data"] == canary_cm["data"]
    held_pod = held_job["spec"]["template"]["spec"]
    canary_pod = canary_job["spec"]["template"]["spec"]
    held_env = {row["name"]: row for row in held_pod["containers"][0]["env"]}
    canary_env = {row["name"]: row for row in canary_pod["containers"][0]["env"]}
    held_diagnostic = held_env["QWEN_HOSTED_WHOLE_TASK_DIAGNOSTIC_ROOT"]["value"]
    held_private = str(Path(held_env["QWEN_HOSTED_WHOLE_TASK_RELEASE_PATH"]["value"]).parent)

    def normalize(value: object, diagnostic: str, private: str) -> object:
        return json.loads(
            json.dumps(value)
            .replace(diagnostic, "__DIAGNOSTIC_ROOT__")
            .replace(private, "__PRIVATE_ROOT__")
        )

    assert normalize(
        held_pod["initContainers"], held_diagnostic, held_private
    ) == normalize(
        canary_pod["initContainers"],
        package.RUNTIME_GATE_CANARY_DIAGNOSTIC_ROOT,
        package.RUNTIME_GATE_CANARY_PRIVATE_ROOT,
    )
    assert normalize(
        held_pod["containers"][0]["args"], held_diagnostic, held_private
    ) == normalize(
        canary_pod["containers"][0]["args"],
        package.RUNTIME_GATE_CANARY_DIAGNOSTIC_ROOT,
        package.RUNTIME_GATE_CANARY_PRIVATE_ROOT,
    )
    assert "apt-get" not in json.dumps(held_pod)
    assert "apt-get" not in json.dumps(canary_pod)
    assert "FLEET_API_KEY" in held_env
    assert "FLEET_API_KEY" not in canary_env
    assert "QWEN_HOSTED_WHOLE_TASK_RUNTIME_GATE_CANARY" not in held_env
    assert canary_env["QWEN_HOSTED_WHOLE_TASK_RUNTIME_GATE_CANARY"]["value"] == "true"
    assert held_job["metadata"]["annotations"]["cyber-post-train.fleet.ai/launch-authorized"] == (
        "false"
    )
    assert canary_env["QWEN_HOSTED_WHOLE_TASK_OUTPUT_ROOT"]["value"] == "/dev/null"


def test_runtime_gate_canary_package_is_new_score_free_create_once_identity() -> None:
    rendered = package.render_runtime_gate_canary(ROOT)
    assert [item["kind"] for item in rendered["items"]] == ["ConfigMap", "Job"]
    cm, job = rendered["items"]
    assert cm["metadata"]["name"] == package.RUNTIME_GATE_CANARY_CONFIGMAP
    assert cm["immutable"] is True
    assert job["metadata"]["name"] == package.RUNTIME_GATE_CANARY_JOB
    assert job["metadata"]["annotations"] == {
        "cyber-post-train.fleet.ai/create-once": "true",
        "cyber-post-train.fleet.ai/diagnostic-only": "true",
        "cyber-post-train.fleet.ai/launch-authorized": "false",
    }
    assert job["spec"]["backoffLimit"] == 0
    container = job["spec"]["template"]["spec"]["containers"][0]
    env = {row["name"]: row for row in container["env"]}
    assert "FLEET_API_KEY" not in env
    assert env["QWEN_HOSTED_WHOLE_TASK_RUNTIME_GATE_CANARY"]["value"] == "true"
    assert env["QWEN_HOSTED_WHOLE_TASK_OUTPUT_ROOT"]["value"] == "/dev/null"
    assert env["QWEN_HOSTED_WHOLE_TASK_DIAGNOSTIC_ROOT"]["value"] == (
        package.RUNTIME_GATE_CANARY_DIAGNOSTIC_ROOT
    )
    held = json.loads(cm["data"]["release.json"])
    assert held["launch_authorized"] is False
    assert env["QWEN_HOSTED_WHOLE_TASK_RELEASE_SHA256"]["value"] == held["receipt_sha256"]
    assert env["QWEN_HOSTED_WHOLE_TASK_RELEASE_PATH"]["value"] == (
        package.RUNTIME_GATE_CANARY_PRIVATE_ROOT + "/release.json"
    )
    assert env["QWEN_HOSTED_WHOLE_TASK_PACKAGE_SOURCE_PATH"]["value"] == (
        package.RUNTIME_GATE_CANARY_PRIVATE_ROOT + "/package-source.json"
    )
    assert "/bootstrap/release.json" not in json.dumps(container)
    assert package.RUNTIME_GATE_CANARY_JOB.endswith("-v5")
    assert not cm["metadata"]["name"].startswith(
        "chris-q38-hosted-whole-task-runtime-gate-canary-v4"
    )
    assert not job["metadata"]["name"].startswith(
        "chris-q38-hosted-whole-task-runtime-gate-canary-v4"
    )


def test_runtime_gate_canary_records_prebootstrap_and_uses_pinned_cli_without_apt() -> None:
    cm, job = package.render_runtime_gate_canary(ROOT)["items"]
    pod = job["spec"]["template"]["spec"]
    assert [item["name"] for item in pod["initContainers"]] == [
        "prebootstrap-evidence",
        "docker-cli",
        "dind",
    ]
    pre, cli, _dind = pod["initContainers"]
    assert pre["command"] == ["python", "/bootstrap/qwen_hosted_prebootstrap_v4.py"]
    assert pre["args"][2:4] == ["00-prebootstrap-entry", "--root"]
    assert {row["name"] for row in pre["env"]} == {
        "JOB_UID",
        "POD_UID",
        "QWEN_HOSTED_WHOLE_TASK_DIAGNOSTIC_ROOT",
        "QWEN_HOSTED_WHOLE_TASK_RELEASE_SHA256",
        "QWEN_HOSTED_WHOLE_TASK_PACKAGE_SOURCE_SHA256",
    }
    assert cli["image"] == package.base.DIND_IMAGE
    assert package.DOCKER_CLI_SHA256 in cli["args"][0]
    assert package.DOCKER_BUILDX_SHA256 in cli["args"][0]
    assert str(package.DOCKER_CLI_BYTES) in cli["args"][0]
    docker_volume = next(row for row in pod["volumes"] if row["name"] == "docker-cli")
    assert docker_volume == {"name": "docker-cli", "emptyDir": {"sizeLimit": "256Mi"}}
    evaluator = pod["containers"][0]
    command = evaluator["args"][0]
    assert command.index("01-network-package-install-bypassed") < command.index(
        "02-pinned-docker-cli-ready"
    )
    assert command.index("03-private-input-stage-started") < command.index(
        "qwen_hosted_private_stage_v5.py stage-all"
    )
    assert command.index("qwen_hosted_private_stage_v5.py stage-all") < command.index(
        "04-private-input-stage-done"
    )
    assert "apt-get" not in json.dumps(pod)
    assert "qwen_hosted_prebootstrap_v4.py" in cm["data"]
    assert "qwen_hosted_private_stage_v5.py" in cm["data"]
    package_source = json.loads(cm["data"]["package-source.json"])
    assert package_source["files"]["qwen_hosted_prebootstrap_v4.py"] == self_hosted.sha256(
        cm["data"]["qwen_hosted_prebootstrap_v4.py"].encode()
    )
    for required in (
        "qwen_hosted_prebootstrap_v4.py",
        "qwen_hosted_private_stage_v5.py",
        "run_qwen_hosted_whole_task_successor_v1.sh",
    ):
        assert f"/bootstrap/{required}" in json.dumps(pod)
        assert required in cm["data"]
    bootstrap_references = set(
        re.findall(r"/bootstrap/([A-Za-z0-9_.-]+)", json.dumps(pod))
    )
    assert bootstrap_references <= set(cm["data"])


def test_materialized_prebootstrap_package_writes_self_digesting_score_free_receipts(
    tmp_path: Path,
) -> None:
    cm, _job = package.render_runtime_gate_canary(ROOT)["items"]
    module_path = tmp_path / "qwen_hosted_prebootstrap_v4.py"
    module_path.write_text(cm["data"]["qwen_hosted_prebootstrap_v4.py"])
    evidence_root = tmp_path / "evidence"
    env = os.environ.copy()
    env.update(
        {
            "JOB_UID": "11111111-1111-4111-8111-111111111111",
            "POD_UID": "22222222-2222-4222-8222-222222222222",
            "QWEN_HOSTED_WHOLE_TASK_RELEASE_SHA256": "sha256:" + "a" * 64,
            "QWEN_HOSTED_WHOLE_TASK_PACKAGE_SOURCE_SHA256": "sha256:" + "b" * 64,
        }
    )
    for phase in prebootstrap.PHASES:
        subprocess.run(
            [
                sys.executable,
                str(module_path),
                "record",
                "--phase",
                phase,
                "--root",
                str(evidence_root.resolve()),
            ],
            check=True,
            env=env,
            capture_output=True,
            text=True,
        )
        receipt = json.loads(
            (evidence_root / "prebootstrap-phases" / f"{phase}.json").read_text()
        )
        assert receipt["receipt_sha256"] == self_hosted.digest_without(
            receipt, "receipt_sha256"
        )
        assert receipt["network_package_install"] == "BYPASSED"
        assert receipt["docker_cli_source"] == "PINNED_IMAGE_SHARED_EMPTYDIR"
        assert receipt["credentials_included"] is False
        assert {
            receipt[field]
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
            )
        } == {0}


def test_private_stage_materializes_projected_inputs_create_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cm, job = package.render_runtime_gate_canary(ROOT)["items"]
    bootstrap = tmp_path / "bootstrap"
    projected = bootstrap / "..data"
    projected.mkdir(parents=True)
    for name in private_stage.SOURCE_NAMES:
        (projected / name).write_text(cm["data"][name])
        (bootstrap / name).symlink_to(Path("..data") / name)
    package_source = json.loads(cm["data"]["package-source.json"])
    release = json.loads(cm["data"]["release.json"])
    monkeypatch.setenv(
        "QWEN_HOSTED_WHOLE_TASK_PACKAGE_SOURCE_SHA256",
        package_source["receipt_sha256"],
    )
    monkeypatch.setenv("QWEN_HOSTED_WHOLE_TASK_RELEASE_SHA256", release["receipt_sha256"])
    monkeypatch.setenv("JOB_UID", "11111111-1111-4111-8111-111111111111")
    monkeypatch.setenv("POD_UID", "22222222-2222-4222-8222-222222222222")
    destination = tmp_path / "private"
    receipt = private_stage.stage_all(bootstrap.resolve(), destination.resolve())
    assert receipt["receipt_sha256"] == self_hosted.digest_without(
        receipt, "receipt_sha256"
    )
    assert {row["name"] for row in receipt["files"]} == set(private_stage.SOURCE_NAMES)
    assert {row["source_was_projected_symlink"] for row in receipt["files"]} == {True}
    assert {row["destination_is_private_regular_file"] for row in receipt["files"]} == {
        True
    }
    assert destination.stat().st_mode & 0o777 == 0o700
    for name in private_stage.SOURCE_NAMES:
        staged = destination / name
        assert staged.read_bytes() == (projected / name).read_bytes()
        assert staged.stat().st_mode & 0o777 == 0o600
        assert not staged.is_symlink()
    with pytest.raises(FileExistsError):
        private_stage.stage_all(bootstrap.resolve(), destination.resolve())
    assert "FLEET_API_KEY" not in {
        row["name"] for row in job["spec"]["template"]["spec"]["containers"][0]["env"]
    }


def test_private_runtime_loader_rejects_projected_symlink_and_nonprivate_mode(
    tmp_path: Path,
) -> None:
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    regular = private / "release.json"
    regular.write_text('{"status":"test"}\n')
    regular.chmod(0o600)
    assert runtime._load_private_regular(regular.resolve()) == {"status": "test"}  # noqa: SLF001
    regular.chmod(0o644)
    with pytest.raises(RuntimeError, match="mode drifted"):
        runtime._load_private_regular(regular.resolve())  # noqa: SLF001
    regular.chmod(0o600)
    projected = private / "projected.json"
    projected.symlink_to(regular.name)
    with pytest.raises(RuntimeError, match="path drifted"):
        runtime._load_private_regular(projected.absolute())  # noqa: SLF001


def test_runtime_gate_canary_records_every_safe_gate_phase(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plans = successor.build_plans(ROOT)
    sources = _sources(plans)
    package_source_path = tmp_path / "package-source.json"
    package_source_path.write_text(json.dumps(sources["qwen-a"], sort_keys=True) + "\n")
    package_source_path.chmod(0o600)
    held_path = ROOT / successor.HELD_PATH
    held = successor.load(held_path)
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    private_held_path = private / "release.json"
    private_held_path.write_text(json.dumps(held))
    private_held_path.chmod(0o600)
    private_package_source_path = private / "package-source.json"
    private_package_source_path.write_bytes(package_source_path.read_bytes())
    private_package_source_path.chmod(0o600)
    monkeypatch.setenv(
        "QWEN_HOSTED_WHOLE_TASK_PACKAGE_SOURCE_SHA256",
        sources["qwen-a"]["receipt_sha256"],
    )
    monkeypatch.setenv(
        "QWEN_HOSTED_WHOLE_TASK_RELEASE_PATH", str(private_held_path.resolve())
    )
    monkeypatch.setenv("QWEN_HOSTED_WHOLE_TASK_RELEASE_SHA256", held["receipt_sha256"])
    phases: list[str] = []
    loaded_plans, package_source = runtime._load_bound_inputs(  # noqa: SLF001
        plans["qwen-a"],
        package_source_path=private_package_source_path.resolve(),
        phase_callback=phases.append,
    )
    assert (
        runtime.runtime_gate_check(
            loaded_plans,
            package_source,
            canary=True,
            phase_callback=phases.append,
        )
        == held
    )
    assert phases == [
        "build-plans-started",
        "build-plans-done",
        "package-source-load-done",
        "package-source-digest-done",
        "plan-match-done",
        "held-load-started",
        "held-load-done",
        "held-digest-done",
        "package-source-validate-done",
        "held-validate-done",
    ]


def test_runtime_gate_canary_failure_preserves_last_completed_phase(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = successor.build_plans(ROOT)["qwen-a"]
    phases: list[str] = []
    monkeypatch.setattr(
        successor,
        "build_plans",
        lambda _root: (_ for _ in ()).throw(ValueError("protected phase detail")),
    )
    with pytest.raises(ValueError, match="protected phase detail"):
        runtime.run_gate_canary(
            plan,
            receipt_path=tmp_path / "unused.json",
            package_source_path=(tmp_path / "unused-package.json").resolve(),
            phase_callback=phases.append,
        )
    assert phases == ["build-plans-started"]


def test_unmocked_materialized_canary_package_runs_exact_runtime_gate(tmp_path: Path) -> None:
    cm, _job = package.render_runtime_gate_canary(ROOT)["items"]
    data = cm["data"]
    root = tmp_path / "workspace" / "cyber-post-train"
    module_root = root / "evals" / "fleet"
    config_root = module_root / "configs"
    bootstrap = tmp_path / "bootstrap"
    config_root.mkdir(parents=True)
    bootstrap.mkdir()
    projected = bootstrap / "..data"
    projected.mkdir()
    (module_root.parent / "__init__.py").touch()
    (module_root / "__init__.py").touch()
    for name, value in data.items():
        if name.endswith(".py"):
            (module_root / name).write_text(value)
    (config_root / "qwen-hosted-generation19-qwen-a-v4.json").write_text(
        data["source-plan-v4-a.json"]
    )
    (config_root / "qwen-hosted-generation19-qwen-b-v4.json").write_text(
        data["source-plan-v4-b.json"]
    )
    plan_path = config_root / "runtime-plan.json"
    plan_path.write_text(data["plan.json"])
    projected_package_source = projected / "package-source.json"
    projected_package_source.write_text(data["package-source.json"])
    projected_authority = projected / "release.json"
    projected_authority.write_text(data["release.json"])
    (bootstrap / "package-source.json").symlink_to(Path("..data/package-source.json"))
    (bootstrap / "release.json").symlink_to(Path("..data/release.json"))
    private = tmp_path / "private"
    diagnostic = tmp_path / "diagnostic"
    receipt_path = diagnostic / "RUNTIME-GATE-CANARY.json"
    forbidden_output = tmp_path / "forbidden-output"
    env = os.environ.copy()
    env.pop("FLEET_API_KEY", None)
    env.update(
        {
            "QWEN_HOSTED_WHOLE_TASK_PACKAGE_SOURCE_SHA256": json.loads(data["package-source.json"])[
                "receipt_sha256"
            ],
            "QWEN_HOSTED_WHOLE_TASK_RELEASE_SHA256": json.loads(data["release.json"])[
                "receipt_sha256"
            ],
            "JOB_UID": "11111111-1111-4111-8111-111111111111",
            "POD_UID": "22222222-2222-4222-8222-222222222222",
        }
    )
    subprocess.run(
        [
            sys.executable,
            str(module_root / "qwen_hosted_private_stage_v5.py"),
            "stage-all",
            "--bootstrap-root",
            str(bootstrap.resolve()),
            "--destination-root",
            str(private.resolve()),
        ],
        check=True,
        env=env,
        capture_output=True,
        text=True,
    )
    package_source_path = private / "package-source.json"
    authority_path = private / "release.json"
    assert package_source_path.is_file() and not package_source_path.is_symlink()
    assert authority_path.is_file() and not authority_path.is_symlink()
    assert package_source_path.stat().st_mode & 0o777 == 0o600
    assert authority_path.stat().st_mode & 0o777 == 0o600
    env["QWEN_HOSTED_WHOLE_TASK_RELEASE_PATH"] = str(authority_path)
    subprocess.run(
        [
            "uv",
            "run",
            "--no-project",
            "--with",
            "httpx==0.28.1",
            "python",
            "-m",
            "evals.fleet.qwen_hosted_whole_task_successor_v1_runtime",
            "--plan",
            str(plan_path),
            "--out",
            str(forbidden_output),
            "--diagnostic-root",
            str(diagnostic),
            "--proxy",
            str(module_root / "fixed_proxy.py"),
            "--runtime-gate-canary-receipt",
            str(receipt_path),
            "--package-source",
            str(package_source_path),
        ],
        cwd=root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    receipt = successor.load(receipt_path)
    successor.validate_runtime_gate_canary(receipt)
    assert receipt["status"] == "PASS"
    assert not forbidden_output.exists()
    assert not receipt_path.with_name("RUNTIME-GATE-CANARY-FAILED.json").exists()


def test_released_render_is_closed_for_consumed_identities(tmp_path: Path) -> None:
    plans = successor.build_plans(ROOT)
    release = _release(plans)
    release_path = tmp_path / "release.json"
    release_path.write_text(json.dumps(release))
    with pytest.raises(RuntimeError, match="closed for consumed object and execution identities"):
        package.render(ROOT, release_path=release_path)
