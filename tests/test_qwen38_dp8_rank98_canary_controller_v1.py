from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.fleet import qwen38_dp8_rank98_canary_controller_v1 as controller
from evals.fleet import qwen38_dp8_rank98_canary_package_v1 as held_package
from evals.fleet import qwen38_dp8_rank98_canary_partition_v1 as partition
from evals.fleet import self_hosted
from tests.test_qwen38_dp8_rank98_canary_package_v1 import _root_with_parity

JOB_UID = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
POD_UID = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"


def _root_release(binding: dict, qualification: dict, held: dict, package_sha: str) -> dict:
    value = {
        "schema_version": controller.ROOT_RELEASE_SCHEMA,
        "status": "RELEASED_ONE_A1_CANARY",
        "root_reviewed": True,
        "root_review_authority": "root",
        "root_review_receipt_sha256": "sha256:" + "9" * 64,
        "server_binding_receipt_sha256": binding["receipt_sha256"],
        "qualification_result_sha256": qualification["receipt_sha256"],
        "held_partition_receipt_sha256": held["receipt_sha256"],
        "held_package_sha256": package_sha,
        "fresh_live_scan_receipt_sha256": held["fresh_live_scan_receipt_sha256"],
        "controller_job_name": held["held_controller_identity"]["job_name"],
        "controller_configmap_name": held["held_controller_identity"]["configmap_name"],
        "reservation_run_id": "chris-cyber-q38-dp8-c-r098-reservation-v1",
        "active_attempt": 1,
        "held_attempts": [2, 3, 4],
        "all_four_claims_absent": True,
        "all_four_outputs_absent": True,
        "exact_session_and_verifier_matches": 0,
        "accepted_receipt_matches": 0,
        "launch_authorized": True,
        "scoring_authorized": True,
        "controller_create_permitted": True,
        "prompts_traces_flags_scores_or_model_outputs_included": False,
    }
    value["receipt_sha256"] = self_hosted.digest_without(value, "receipt_sha256")
    return value


def test_root_release_is_exact_and_builds_only_one_active_plan(tmp_path: Path, monkeypatch) -> None:
    root, binding, qualification, held_release = _root_with_parity(tmp_path, monkeypatch)
    held_bundle = held_package.render_bundle(root, binding, qualification, held_release)
    held = held_release["held_partition"]
    release = _root_release(binding, qualification, held, held_bundle["package_sha256"])
    controller.validate_root_release(
        release, binding, qualification, held, held_bundle["package_sha256"]
    )
    plans = controller.build_plans(root, binding, release)
    assert [plan["launch_authorized"] for plan in plans] == [True, False, False, False]
    assert [plan["reservation_state"] for plan in plans] == [
        "active",
        "reservation_only",
        "reservation_only",
        "reservation_only",
    ]
    assert [plan["item"]["cell_id"] for plan in plans] == partition.CELL_IDS


def test_root_release_rejects_stale_or_missing_independent_review(
    tmp_path: Path, monkeypatch
) -> None:
    root, binding, qualification, held_release = _root_with_parity(tmp_path, monkeypatch)
    held_bundle = held_package.render_bundle(root, binding, qualification, held_release)
    held = held_release["held_partition"]
    release = _root_release(binding, qualification, held, held_bundle["package_sha256"])
    release["root_reviewed"] = False
    release["receipt_sha256"] = self_hosted.digest_without(release, "receipt_sha256")
    with pytest.raises(controller.ControllerError, match="root release"):
        controller.validate_root_release(
            release, binding, qualification, held, held_bundle["package_sha256"]
        )


def test_four_claim_reservation_rolls_back_partial_failure_with_same_uid_proof(
    tmp_path: Path, monkeypatch
) -> None:
    root, binding, qualification, held_release = _root_with_parity(tmp_path, monkeypatch)
    held_bundle = held_package.render_bundle(root, binding, qualification, held_release)
    release = _root_release(
        binding,
        qualification,
        held_release["held_partition"],
        held_bundle["package_sha256"],
    )
    plans = controller.build_plans(root, binding, release)
    claim_root = tmp_path / "claims"
    rollback_root = tmp_path / "rollbacks"
    claim_root.mkdir()
    original = self_hosted.write_json_once
    claim_calls = 0

    def fail_second_claim(path: Path, value: dict) -> None:
        nonlocal claim_calls
        if path.parent == claim_root:
            claim_calls += 1
            if claim_calls == 2:
                raise OSError("injected pre-model reservation failure")
        original(path, value)

    monkeypatch.setattr(self_hosted, "write_json_once", fail_second_claim)
    with pytest.raises(OSError, match="injected"):
        controller.reserve_all(
            plans,
            release,
            JOB_UID,
            POD_UID,
            claim_root=claim_root,
            rollback_root=rollback_root,
        )
    assert list(claim_root.iterdir()) == []
    proofs = list(rollback_root.iterdir())
    assert len(proofs) == 1
    proof = json.loads(proofs[0].read_text())
    assert proof["model_requests"] == 0
    assert proof["job_uid"] == JOB_UID
    assert proof["pod_uid"] == POD_UID
    assert proof["receipt_sha256"] == self_hosted.digest_without(
        proof, "receipt_sha256"
    )


def test_successful_reservation_creates_exact_g19_claims(tmp_path: Path, monkeypatch) -> None:
    root, binding, qualification, held_release = _root_with_parity(tmp_path, monkeypatch)
    held_bundle = held_package.render_bundle(root, binding, qualification, held_release)
    release = _root_release(
        binding,
        qualification,
        held_release["held_partition"],
        held_bundle["package_sha256"],
    )
    plans = controller.build_plans(root, binding, release)
    claim_root = tmp_path / "claims"
    claim_root.mkdir()
    claims = controller.reserve_all(
        plans, release, JOB_UID, POD_UID, claim_root=claim_root
    )
    assert sorted(claims) == [1, 2, 3, 4]
    assert [claims[index]["reservation_state"] for index in range(1, 5)] == [
        "active",
        "reservation_only",
        "reservation_only",
        "reservation_only",
    ]
    assert {path.name for path in claim_root.iterdir()} == {
        execution.removeprefix("sha256:") + ".json"
        for execution in partition.EXECUTION_IDS
    }


def test_post_reservation_session_gate_ignores_unrelated_but_rejects_exact(
    tmp_path: Path, monkeypatch
) -> None:
    root, binding, qualification, held_release = _root_with_parity(tmp_path, monkeypatch)
    held_bundle = held_package.render_bundle(root, binding, qualification, held_release)
    release = _root_release(
        binding,
        qualification,
        held_release["held_partition"],
        held_bundle["package_sha256"],
    )
    plans = controller.build_plans(root, binding, release)

    class Client:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    monkeypatch.setattr(controller.httpx, "Client", Client)
    monkeypatch.setattr(
        self_hosted,
        "_request",
        lambda *_args, **_kwargs: {
            "team_name": "fleet",
            "team_id": self_hosted.FLEET_TEAM_ID,
        },
    )
    unrelated = [{"metadata": {"run_id": f"historical-{index}"}} for index in range(26)]
    monkeypatch.setattr(self_hosted, "_task_sessions", lambda *_args: unrelated)
    controller._assert_authoritative_sessions_clear(plans, "key")

    collision = [
        {"metadata": {"execution_id": plans[0]["item"]["execution_id"]}}
    ]
    monkeypatch.setattr(self_hosted, "_task_sessions", lambda *_args: collision)
    with pytest.raises(controller.ControllerError, match="session collision"):
        controller._assert_authoritative_sessions_clear(plans, "key")


def test_callback_order_rechecks_sessions_after_claims_before_model() -> None:
    source = Path(controller.__file__).read_text()
    prepare = source.index("def prepare_before_model_call")
    reserve = source.index("claims = reserve_all", prepare)
    second_session_gate = source.index("_assert_authoritative_sessions_clear", reserve)
    publish = source.index('reservation_root = Path("/mnt/sfs/jobs")', second_session_gate)
    run = source.index("def run_a1", publish)
    model = source.index("self_hosted.run", run)
    assert reserve < second_session_gate < publish < model


def test_a1_failure_persists_only_sanitized_boundary_and_never_runs_held_attempts(
    tmp_path: Path, monkeypatch
) -> None:
    binding = {
        "service_origin": "http://example:8000",
        "service_uid": "1",
        "rayjob_uid": "2",
        "workload_uid": "3",
        "traffic_path": str(tmp_path / "traffic"),
        "serving_block": "dedicated-qwen-dp8-c-v1",
        "receipt_sha256": "sha256:" + "1" * 64,
    }
    output = tmp_path / "a1"
    plan = {
        "output_root": str(output),
        "config": {},
        "item": {"attempt": 1},
        "plan_sha256": "sha256:" + "2" * 64,
    }
    claims = {1: {"receipt_sha256": "sha256:" + "3" * 64}}
    reservation = tmp_path / "reservation"
    reservation.mkdir()
    release = {"receipt_sha256": "sha256:" + "4" * 64}
    monkeypatch.setenv("FLEET_API_KEY", "test")
    monkeypatch.setenv("JOB_UID", JOB_UID)
    monkeypatch.setenv("POD_UID", POD_UID)
    monkeypatch.setattr(
        controller,
        "prepare_before_model_call",
        lambda *_args: ([plan], claims, reservation),
    )
    monkeypatch.setattr(controller.legacy, "_traffic_loop", lambda stop: None)

    def fail(*_args, **_kwargs):
        raise self_hosted.FleetRequestError("POST", "/v1/task-versions/x/score", 503)

    monkeypatch.setattr(self_hosted, "run", fail)
    with pytest.raises(self_hosted.FleetRequestError):
        controller.run_a1(tmp_path, tmp_path / "proxy.py", binding, release)
    abort = json.loads((reservation / "CANARY-ABORT.json").read_text())
    assert abort["error_type"] == "FleetRequestError"
    assert abort["method"] == "POST"
    assert abort["route"] == "/v1/task-versions/x/score"
    assert abort["http_status"] == 503
    assert abort["a1_retry_authorized"] is False
    assert abort["attempts_2_to_4_transfer_authorized"] is False
    assert "response" not in abort


def test_formal_acceptance_barrier_rejects_incomplete_result() -> None:
    plan = {
        "item": {
            "attempt": 1,
            "cell_id": partition.CELL_IDS[0],
            "execution_id": partition.EXECUTION_IDS[0],
            "run_id": partition.RUN_IDS[0],
        }
    }
    incomplete = {
        "accepted": True,
        "credited": True,
        "retry_allowed": False,
        **plan["item"],
        "session_ingest_completed": False,
        "cleanup_completed": True,
        "session_id": "session",
        "verifier_execution_id": "verifier",
    }
    with pytest.raises(controller.ControllerError, match="formal authoritative"):
        controller._assert_formal_acceptance(incomplete, plan)
