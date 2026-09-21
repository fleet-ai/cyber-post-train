from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "configs/evaluation/qwen38-checkpoint-serving-readiness-v1.json"
LEDGER = ROOT / "configs/evaluation/qwen38-checkpoint-eval-ledger-v1.json"
RECONCILIATION = ROOT / "docs/evidence/qwen38-checkpoint-eval-ledger-reconciliation-20260921.json"

ACCEPTED_IDS = {
    "q38-fresh75-step230",
    "q38-teacher-dense-v5-step186",
    "q38-self-sft-step44",
    "q38-available-a-lr30-step76",
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _digest(value: dict) -> str:
    unsigned = dict(value)
    unsigned.pop("sha256")
    encoded = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _file_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _by_id(rows: list[dict]) -> dict[str, dict]:
    result = {row["artifact_id"]: row for row in rows}
    assert len(result) == len(rows), "duplicate artifact_id"
    return result


def _all_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        keys = set(value)
        for child in value.values():
            keys.update(_all_keys(child))
        return keys
    if isinstance(value, list):
        keys: set[str] = set()
        for child in value:
            keys.update(_all_keys(child))
        return keys
    return set()


def test_ledger_matrix_and_reconciliation_are_self_digesting() -> None:
    ledger = _load(LEDGER)
    matrix = _load(MATRIX)
    reconciliation = _load(RECONCILIATION)

    assert ledger["sha256"] == _digest(ledger)
    assert matrix["sha256"] == _digest(matrix)
    assert reconciliation["sha256"] == _digest(reconciliation)

    for document in (ledger, matrix):
        binding = document["current_reconciliation"]
        assert binding["path"] == str(RECONCILIATION.relative_to(ROOT))
        assert binding["receipt_sha256"] == reconciliation["sha256"]


def test_exactly_four_checkpoints_remain_accepted_everywhere() -> None:
    ledger = _load(LEDGER)
    matrix = _load(MATRIX)
    reconciliation = _load(RECONCILIATION)

    ledger_rows = _by_id(ledger["accepted_checkpoints"])
    matrix_rows = _by_id(matrix["artifacts"])
    receipt_rows = _by_id(reconciliation["accepted_checkpoints"])
    assert set(ledger_rows) == ACCEPTED_IDS
    assert set(matrix_rows) == ACCEPTED_IDS
    assert set(receipt_rows) == ACCEPTED_IDS
    assert reconciliation["accepted_checkpoint_count"] == 4
    assert set(reconciliation["accepted_checkpoint_ids"]) == ACCEPTED_IDS
    assert reconciliation["acceptance_semantics"] == {
        "checkpoint_acceptance_scope": (
            "Exact source checkpoint, BF16 export, and zero-update reload qualification only."
        ),
        "checkpoint_acceptance_is_matched_evaluation_acceptance": False,
        "checkpoint_acceptance_authorizes_evaluation_launch": False,
        "matched_evaluation_requires_separate_live_path_binding_parity_and_duplicate_gates": True,
        "accepted_matched_web_results": 0,
        "accepted_matched_fleet_results": 0,
    }

    for artifact_id in ACCEPTED_IDS:
        assert (
            ledger_rows[artifact_id]["checkpoint_path"]
            == receipt_rows[artifact_id]["checkpoint_path"]
        )
        assert ledger_rows[artifact_id]["export_path"] == receipt_rows[artifact_id]["export_path"]
        assert (
            matrix_rows[artifact_id]["checkpoint"]["path"]
            == receipt_rows[artifact_id]["checkpoint_path"]
        )
        assert (
            matrix_rows[artifact_id]["export"]["path"] == receipt_rows[artifact_id]["export_path"]
        )


def test_accepted_checkpoint_receipts_and_current_routes_are_exact() -> None:
    matrix = _by_id(_load(MATRIX)["artifacts"])
    reconciliation = _load(RECONCILIATION)
    accepted = _by_id(reconciliation["accepted_checkpoints"])

    assert accepted["q38-fresh75-step230"]["export_payload_manifest_sha256"] == (
        "sha256:36eec01f1dd3f0d47f6b099e79970d9533cf59c37d8e15478907413dd162e029"
    )
    assert accepted["q38-teacher-dense-v5-step186"]["export_receipt_sha256"] == (
        "sha256:e6d59f5a58ce54b913d775cd71b81c64bd54637b53795df82baec0f71efefaf9"
    )
    assert accepted["q38-self-sft-step44"]["checkpoint_manifest_file_sha256"] == (
        "sha256:0094cfeb62f120b55d9e2f7fdfd4217c4c0aab66f369e3c52c89fc5e7ccfff9b"
    )

    lr30 = matrix["q38-available-a-lr30-step76"]
    assert lr30["export_reload"]["status"] == "accepted_one_gpu_finite_forward"
    assert lr30["export_reload"]["rayjob_uid"] == ("e5da1477-9678-4f59-960c-97067ea7e9b9")
    assert lr30["export_reload"]["optimizer_steps_executed"] == 0
    assert lr30["stage"]["path"] == "/models/chris-q38-available-a-lr30-step76-v1"
    assert lr30["stage"]["receipt_sha256"] == (
        "sha256:eed9a359eec69e1ba9a4a51169323fab1ff46ad0b138a7bb261d98707e1aee49"
    )
    assert lr30["registration"]["model_id"] == "chris-q38-lr30-step76-web-v1"
    assert lr30["registration"]["active_pods"] == 0

    lifecycle = _by_id(reconciliation["serving_lifecycle"])
    assert set(lifecycle) == ACCEPTED_IDS
    assert all(row["observed_state"] == "paused_zero_active_pods" for row in lifecycle.values())
    assert all(row["live_target"]["status"] == "paused_zero_active_pods" for row in matrix.values())


def test_fresh75_frozen_config_path_mismatch_fails_closed() -> None:
    reconciliation = _load(RECONCILIATION)
    binding = reconciliation["fresh75_path_binding"]
    matrix = _by_id(_load(MATRIX)["artifacts"])["q38-fresh75-step230"]

    assert binding["accepted_export_path"] == (
        "/mnt/sfs/jobs/chris-q38-f75-max-full-v4/hf-export-step230-v1"
    )
    assert binding["frozen_config_repository"] == (
        "/mnt/sfs/jobs/chris-q38-fresh75-e1-v2/hf-export-v1"
    )
    mismatches = {row["path"]: row for row in binding["known_mismatched_frozen_configs"]}
    assert set(mismatches) == {
        "configs/evaluation/qwen38-fresh75-fleet-dev17-opencode-seed43-pass1-v1.json",
        "configs/evaluation/qwen38-fresh75-fleet-dev17-opencode-pass1-v2.json",
        "configs/evaluation/qwen38-fresh75-fleet-dev17-matched-pass1-v2.json",
    }
    assert binding["known_mismatch_count"] == 3
    assert {row["seed"] for row in mismatches.values()} == {42, 43}
    assert all(
        row["repository"] == "/mnt/sfs/jobs/chris-q38-fresh75-e1-v2/hf-export-v1"
        for row in mismatches.values()
    )
    for path, row in mismatches.items():
        assert _file_digest(ROOT / path) == row["file_sha256"]
    assert binding["config_path_binding_matches_accepted_export"] is False
    assert binding["frozen_config_repository_presence_status"] == "not_reopened_in_this_audit"
    assert binding["launchable_from_frozen_config"] is False
    assert matrix["path_binding"]["exact_path_match"] is False
    assert matrix["path_binding"]["launchable"] is False


def test_web_v23_and_v4_are_terminal_infrastructure_evidence_only() -> None:
    web = _load(RECONCILIATION)["webexploitbench"]
    v23 = web["v23_terminal"]
    assert v23["planned_collections"] == 60
    assert v23["accepted_collections"] == 0
    assert v23["failure_preserved_and_released"] == 60
    assert v23["collection_export_failures"] == 45
    assert v23["collection_run_failures"] == 15
    assert v23["active_supervisors"] == 0
    assert v23["scoring_campaign_created"] is False
    assert v23["capability_result"] is False

    v4 = web["v4_latest_qualification"]
    assert v4["transaction"] == "wbe-prompt-parity-p338-v4"
    assert v4["sandbox_id"] == "bvwusw7l9b6s6wrm4kq7n"
    assert v4["terminal_status"] == "failed_infrastructure_only"
    assert v4["reason"] == "prompt_runtime_dependency_preflight_failed"
    assert v4["sanitized_packet_path"] == "/private/tmp/cpt-wbe-prompt-qualification-pr338-v4"
    assert v4["failure_capture_path"] == "state/FAILURE_CAPTURED.json"
    assert v4["release_path"] == "state/RELEASED.json"
    assert v4["release_reconciliation_path"] == "release.reconciliation.json"
    assert (
        sum(
            v4[field]
            for field in ("student_calls", "judge_calls", "scoring_calls", "benchmark_run_calls")
        )
        == 0
    )
    assert v4["provider_resource_status"] == "terminated_owned_inventory_empty"
    assert v4["replacement_snapshot_id"] is None
    assert v4["matched_collection_accepted"] is False
    assert web["accepted_matched_capability_results"] == 0
    assert web["campaign_launchable"] is False


def test_fleet_seed43_counts_are_isolated_and_lr30_v2_changed_nothing() -> None:
    fleet = _load(RECONCILIATION)["fleet_dev17_seed43"]
    arms = _by_id(fleet["arms"])
    assert fleet["accepted_matched_capability_results"] == 0
    assert fleet["final_test_status"] == "sealed_not_accessed"
    assert "Never splice seed42 and seed43" in fleet["seed_isolation_rule"]
    assert {row["seed"] for row in arms.values()} == {43}

    assert arms["qwen38-27b-base-matched-control"] == {
        "artifact_id": "qwen38-27b-base-matched-control",
        "seed": 43,
        "job_uid": "6dc5320f-02f5-470d-ae83-a5dec0fe2a62",
        "status": "terminal_infrastructure_incomplete",
        "accepted": 7,
        "retry_review": 10,
        "complete_capability_result": False,
    }
    assert arms["q38-fresh75-step230"]["accepted"] == 14
    assert arms["q38-fresh75-step230"]["retry_review"] == 3
    assert arms["q38-teacher-dense-v5-step186"]["accepted"] == 12
    assert arms["q38-teacher-dense-v5-step186"]["retry_review"] == 5
    assert arms["q38-self-sft-step44"]["last_safe_observation"] == {
        "accepted": 8,
        "claimed": 5,
        "retry_review": 4,
    }

    lr30 = arms["q38-available-a-lr30-step76"]
    assert (lr30["accepted"], lr30["retry_review"], lr30["claimed"]) == (6, 10, 1)
    v2 = lr30["reconciliation_v2"]
    assert v2["job_uid"] == "9e3d9d6f-8475-46dc-bc8a-f084c7081979"
    assert v2["pod_uid"] == "55709be8-ddcb-44cd-9ccd-aa65a40bc6d7"
    assert v2["workload_uid"] == "8febf409-e805-412a-934b-3fa037546095"
    assert v2["status"] == "failed_before_database_mutation"
    assert v2["reason"] == "postgresql_ledger_plan_digest_differs"
    assert v2["database_changed"] is False
    assert v2["successful_reconciliation_receipts"] == 0
    assert v2["model_generation_calls"] == 0
    assert v2["scoring_calls"] == 0
    assert v2["gpu_allocated"] is False
    assert v2["active_compute_held"] is False

    other_seeds = fleet["other_seed_descriptive_only"]
    assert {row["seed"] for row in other_seeds} == {42}
    assert all(
        row["complete_capability_result_for_seed43_comparison"] is False for row in other_seeds
    )


def test_fresh75_v2_eight_checkpoint_cohort_stays_out_of_eval_backlog() -> None:
    backlog = _load(RECONCILIATION)["qualification_backlog"]
    cohort = backlog["fresh75_v2_cohort"]
    assert backlog["fresh75_v2_cohort_status"] == {
        "training_terminal_candidate_checkpoints": 8,
        "exact_full_checkpoint_receipt_digests_available": 7,
        "unresolved_exact_full_checkpoint_receipt_digests": 1,
        "accepted_exports": 0,
        "matched_eval_backlog_eligible": 0,
    }
    assert len(cohort) == 8
    assert len({row["artifact_id"] for row in cohort}) == 8
    assert all(row["export_path"] is None for row in cohort)
    assert all(row["matched_eval_backlog_eligible"] is False for row in cohort)

    exact = [row for row in cohort if row["checkpoint_receipt_sha256"] is not None]
    unresolved = [row for row in cohort if row["checkpoint_receipt_sha256"] is None]
    assert len(exact) == 7
    assert all(row["checkpoint_receipt_file_sha256"].startswith("sha256:") for row in exact)
    assert all(row["checkpoint_receipt_sha256"].startswith("sha256:") for row in exact)
    assert unresolved == [
        {
            "artifact_id": "q38-fresh75-v2-b8-lr1e5-e4-step460",
            "arm_id": "b8-lr1e5-e4",
            "rayjob_uid": "56ca3cb5-452b-47f3-9c0b-a31403c04c59",
            "checkpoint_path": "/mnt/sfs/jobs/chris-q38-f75-e4-v2/checkpoints/global_step_460",
            "checkpoint_receipt_path": "/mnt/sfs/jobs/chris-q38-f75-e4-v2/checkpoint_receipts/step-000460.json",  # noqa: E501
            "checkpoint_receipt_file_sha256": None,
            "checkpoint_receipt_sha256": None,
            "receipt_digest_status": (
                "unresolved_exact_full_hashes_unavailable_do_not_expand_prefixes"
            ),
            "training_terminal_status": "succeeded_uid_bound_control_plane_observation",
            "export_path": None,
            "matched_eval_backlog_eligible": False,
        }
    ]

    replicate = next(row for row in cohort if row["arm_id"] == "b8-lr1e5-e2")
    assert replicate["relationship_to_accepted_fresh75"] == (
        "same_declared_treatment_same_seed_replicate_not_independent_arm"
    )


def test_backlog_gate_cannot_be_satisfied_by_training_completion_alone() -> None:
    gate = _load(RECONCILIATION)["qualification_backlog"]["minimum_entry_gate"]
    assert gate["training_completion_alone_is_sufficient"] is False
    assert gate["checkpoint_full_payload_manifest_and_independent_rehash"] is True
    assert gate["cpu_only_bf16_zero_update_export"] == {
        "trained_tensors": 1184,
        "restored_base_tensors": 15,
        "tensor_count": 1199,
        "tensor_bytes": 55562855904,
    }
    assert gate["source_size_mtime_stable_and_byte_equal_reopen"] is True
    assert gate["cpu_integrity_and_metadata_reload"] is True
    assert gate["one_gpu_finite_two_token_zero_update_reload"] is True
    assert gate["uid_bound_terminal_and_release_receipt"] is True
    assert gate["serving_stage_and_route_are_later_launchability_gates"] is True


def test_lora_step60_and_broad_sft_terminal_rows_are_not_accepted() -> None:
    backlog = _load(RECONCILIATION)["qualification_backlog"]
    lora = backlog["lora_step60"]
    assert lora["checkpoint_path"] == (
        "/mnt/sfs/jobs/chris-q38-lora-r1-s60-v3/checkpoints/global_step_60"
    )
    assert lora["checkpoint_manifest_file_sha256"] == (
        "sha256:cd53865f869eeb1975aa0e099aef143a736167c5c7b095c283f0da292ffecd78"
    )
    assert lora["sealed_and_independently_rehashed"] is True
    assert lora["gpu_reload_verified"] is False
    assert lora["export_path"] is None
    assert lora["matched_eval_backlog_eligible"] is False
    assert lora["artifact_id"] not in ACCEPTED_IDS

    broad = _by_id(backlog["broad_sft_terminal"])
    assert set(broad) == {
        "q38-full-weight-broad-b8-lr3e-6",
        "q38-full-weight-broad-b16-lr3e-6",
        "q38-full-weight-broad-b8-lr1e-6",
        "q38-full-weight-broad-64k-b8-lr3e-6",
    }
    assert {row["last_optimizer_step"] for row in broad.values()} == {275, 137, 271, 37}
    assert all(row["checkpoint_present"] is False for row in broad.values())
    assert all(row["matched_eval_backlog_eligible"] is False for row in broad.values())
    assert all("running" not in row["resource_status"] for row in broad.values())
    assert broad["q38-full-weight-broad-64k-b8-lr3e-6"]["terminal_marker_flushed"] is False


def test_evidence_source_file_digests_are_full_and_bound_when_present() -> None:
    reconciliation = _load(RECONCILIATION)
    for source in reconciliation["evidence_sources"]:
        assert source["file_sha256"].startswith("sha256:")
        assert len(source["file_sha256"]) == 71
        source_path = ROOT / source["path"]
        if source_path.exists():
            assert _file_digest(source_path) == source["file_sha256"]
        else:
            assert len(source["git_commit"]) == 40
            int(source["git_commit"], 16)

    v2 = next(
        source
        for source in reconciliation["evidence_sources"]
        if source["path"].endswith(
            "qwen38-lr30-step76-stored-session-reconciliation-v2-"
            "database-selection-failure-20260921.json"
        )
    )
    assert v2["required_in_current_tree"] is True
    assert "git_commit" not in v2
    assert (ROOT / v2["path"]).is_file()


def test_status_documents_are_score_blind_and_contain_no_private_payloads() -> None:
    matrix = _load(MATRIX)
    reconciliation = _load(RECONCILIATION)
    assert matrix["scientific_boundary"]["capability_claimed"] is False
    assert matrix["scientific_boundary"]["evaluation_launched_by_this_audit"] is False
    assert matrix["privacy"]["credentials_included"] is False
    assert reconciliation["privacy"] == {
        "scores_included": False,
        "private_cell_task_session_or_trace_identifiers_included": False,
        "prompts_responses_flags_rewards_answers_or_benchmark_content_included": False,
        "credentials_or_database_authorization_material_included": False,
        "private_logs_included": False,
        "raw_log_sha256_metadata_only": True,
    }
    assert _all_keys(reconciliation).isdisjoint(
        {"score", "session_id", "task_key", "task_version_id", "prompt", "trace", "flag", "answer"}
    )
    serialized = (MATRIX.read_text() + RECONCILIATION.read_text()).lower()
    assert "bearer " not in serialized
    assert "private_key" not in serialized


def test_old_candidate_snapshots_are_explicitly_superseded() -> None:
    ledger = _load(LEDGER)
    matrix = _load(MATRIX)
    assert "in_progress_candidates" not in ledger
    assert "in_progress_candidates" not in matrix
    assert "superseded_candidate_observations" in ledger
    assert "superseded_candidate_observations" in matrix
