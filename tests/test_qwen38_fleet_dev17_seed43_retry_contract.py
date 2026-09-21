"""Guards for the score-blind seed-43 reviewed recovery contract."""

import hashlib
import json
from pathlib import Path

from cyber_post_train.jobs import digest
from evals.fleet import evaluate, retry_review_policy

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/evaluation/qwen38-fleet-dev17-seed43-reviewed-recovery-v1.json"
MASTER = ROOT / "configs/evaluation/qwen38-fleet-dev17-seed43-matched-protocol-v1.json"
PREVIEW = ROOT / "docs/evidence/qwen38-fleet-dev17-seed43-recovery-policy-preview-20260921.json"


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def test_contract_is_self_digesting_review_only_and_protocol_bound():
    contract, master = read(CONTRACT), read(MASTER)
    assert contract["sha256"] == digest(
        {key: value for key, value in contract.items() if key != "sha256"}
    )
    assert contract["status"] == "review_only_do_not_apply"
    assert contract["protocol"] == {
        "path": str(MASTER.relative_to(ROOT)),
        "file_sha256": sha256(MASTER),
        "sha256": master["sha256"],
        "sampling_seed": 43,
        "accepted_cells_required_per_arm": 17,
        "final_eight_task_set_access": "sealed",
    }
    assert set(contract["arms"]) == set(master["arms"])


def test_every_arm_keeps_the_frozen_identity_and_reserves_unique_recovery_names():
    contract, master = read(CONTRACT), read(MASTER)
    reserved_jobs, reserved_maps, reserved_outputs = set(), set(), set()
    primary_jobs, primary_databases, primary_outputs = set(), set(), set()
    for name, arm in contract["arms"].items():
        frozen = master["arms"][name]
        assert arm["config_sha256"] == frozen["config_sha256"]
        assert arm["primary_job"] == frozen["primary_job"]
        assert arm["database"] == frozen["primary_database"]
        assert arm["served_id"] == frozen["served_id"]
        primary_jobs.add(frozen["primary_job"])
        primary_databases.add(frozen["primary_database"])
        primary_outputs.add(frozen["primary_output_root"])
        for kind in ("scoring", "rollout"):
            reserved_jobs.add(arm[f"{kind}_job"])
            reserved_maps.add(arm[f"{kind}_config_map"])
            reserved_outputs.add(arm[f"{kind}_output_root"])
    assert len(reserved_jobs) == len(reserved_maps) == len(reserved_outputs) == 8
    assert not reserved_jobs & primary_jobs
    assert not reserved_outputs & primary_outputs
    assert len(primary_jobs) == len(primary_databases) == len(primary_outputs) == 4


def test_contract_is_arm_blind_and_never_conditions_on_score_or_capability():
    contract = read(CONTRACT)
    assert contract["arm_blindness"] == {
        "same_rules_for_every_arm": True,
        "score_values_are_not_inputs": True,
        "capability_outcomes_are_not_inputs": True,
        "accepted_cells_are_final_and_never_replayed": True,
        "cross_arm_splicing_forbidden": True,
        "classification_must_finish_before_any_apply": True,
    }
    serialized = json.dumps(contract).lower()
    assert '"score":' not in serialized
    assert '"reward":' not in serialized
    assert contract["classification"]["no_outcome_transport_failure"][
        "allowed_failure_codes"
    ] == sorted(retry_review_policy.ROLLOUT_RETRY_FAILURE_CODES)
    assert retry_review_policy.MAX_ROLLOUT_RETRIES == 1
    assert retry_review_policy.MAX_SCORING_RECOVERIES == 1


def test_stored_session_and_rollout_recovery_are_independent_and_bounded():
    classification = read(CONTRACT)["classification"]
    assert classification["existing_exact_scored_session"] == {
        "action": "accept_existing_scored_session",
        "model_generation_allowed": False,
        "requirements": [
            "one exact authoritative session",
            "completed session status",
            "exact model task-version and verifier identities",
            "finite authoritative score presence without reading or recording its value",
            "complete immutable attempt evidence",
        ],
    }
    assert classification["existing_unscored_session"]["action"] == ("scoring_only_recovery")
    assert classification["existing_unscored_session"]["model_generation_allowed"] is False
    assert classification["existing_unscored_session"]["maximum_attempts"] == 1
    assert classification["no_outcome_transport_failure"]["action"] == "rollout_retry"
    assert classification["no_outcome_transport_failure"]["model_generation_allowed"] is True
    assert classification["no_outcome_transport_failure"]["maximum_attempts"] == 1
    assert classification["output_limit_or_process_error"] == {
        "model_generation_allowed": False,
        "rule": (
            "Use an exact existing scored session or scoring-only recovery; "
            "never repeat generation."
        ),
    }


def test_apply_gate_is_terminal_create_once_alert_silent_c1_cpu_only():
    execution = read(CONTRACT)["execution"]
    assert execution == {
        "wait_for_all_four_primary_jobs_terminal": True,
        "apply_requires_a_private_cell_manifest_self_digest": True,
        "apply_requires_current_readback_of_every_selected_cell": True,
        "scoring_only_actions_before_any_rollout_retry": True,
        "root_failure_alert_annotation": {"fleet.ai/failure-alerts": "off"},
        "priority_class": "c1",
        "gpu_request": 0,
        "cpu_request": "4",
        "memory_request": "32Gi",
        "server_dry_run_required": True,
        "job_config_map_pod_workload_output_duplicate_absence_required": True,
        "uncertain_create_must_not_be_replayed": True,
    }


def test_review_policy_is_not_part_of_the_frozen_evaluator_runtime():
    assert "retry_review_policy.py" not in evaluate.RUNTIME_FILES


def test_preview_is_self_digesting_sanitized_and_created_nothing():
    preview = read(PREVIEW)
    assert preview["sha256"] == digest(
        {key: value for key, value in preview.items() if key != "sha256"}
    )
    assert preview["privacy"] == {
        "score_values_read_or_recorded": False,
        "capability_outcomes_read_or_recorded": False,
        "cell_session_task_or_trace_identifiers_recorded": False,
        "prompt_response_flag_reward_or_trace_content_recorded": False,
        "final_eight_task_set_accessed": False,
    }
    assert preview["launch_gate"]["open"] is False
    assert preview["interpretation"] == {
        "jobs_created": 0,
        "config_maps_created": 0,
        "sessions_created": 0,
        "cells_replayed_or_mutated": 0,
        "capability_claim_made": False,
        "eligibility_census_is_final": False,
    }
    duplicate = preview["current_duplicate_preview"]
    assert all(
        duplicate[key] == 0
        for key in (
            "reserved_jobs_found",
            "reserved_config_maps_found",
            "reserved_pods_found",
            "reserved_workloads_found",
            "reserved_output_roots_found",
        )
    )
