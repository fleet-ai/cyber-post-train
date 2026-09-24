from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN_COMMIT = "0ae3e0923c1e74e65bea3a91f102c1ac7a12917d"
PLAN_PATH = ROOT / "configs" / "qualification" / "qwen38-corrected-teacher-sft-minimum-plan-v1.json"
EVAL_PATH = (
    ROOT / "configs" / "evaluation" / "qwen38-corrected-teacher3k-heldout20-pass4-successor-v1.json"
)
DOC_PATH = ROOT / "docs" / "QWEN38_CORRECTED_TEACHER_SFT_MINIMUM_PLAN_2026-09-24.md"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _logical_digest(value: dict) -> str:
    body = {key: item for key, item in value.items() if key != "sha256"}
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _file_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _all_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {key for item in value.values() for key in _all_keys(item)}
    if isinstance(value, list):
        return {key for item in value for key in _all_keys(item)}
    return set()


def test_minimum_plan_is_self_digesting_blocked_and_cross_binds_evaluation() -> None:
    plan = _load(PLAN_PATH)
    evaluation = _load(EVAL_PATH)
    documentation = DOC_PATH.read_text(encoding="utf-8")

    assert plan["sha256"] == _logical_digest(plan)
    assert evaluation["sha256"] == _logical_digest(evaluation)
    assert plan["sha256"].removeprefix("sha256:") in documentation
    assert evaluation["sha256"].removeprefix("sha256:") in documentation
    assert "TO_BE_RECOMPUTED" not in documentation
    assert plan["status"] == "proposal_blocked_no_launch"
    assert plan["launch_authorized"] is False
    assert evaluation["launch_authorized"] is False
    assert plan["prepared_from_main"] == MAIN_COMMIT
    assert evaluation["prepared_from_main"] == MAIN_COMMIT
    assert plan["evidence"]["heldout_successor"] == {
        "path": "configs/evaluation/qwen38-corrected-teacher3k-heldout20-pass4-successor-v1.json",
        "file_sha256": _file_digest(EVAL_PATH),
        "sha256": evaluation["sha256"],
    }

    assert set(plan["phases"]["corpus_canary"]["pr584_blockers"][0]) == {
        "id",
        "required_evidence",
    }
    assert {row["id"] for row in plan["phases"]["corpus_canary"]["pr584_blockers"]} == {
        "exact-family-role-roster",
        "successful-report-call-binding",
        "exact-opencode-model-tool-contract",
        "private-source-readback",
    }
    admission = plan["phases"]["corpus_canary"]["additional_admission_requirement"]
    required_unknowns = {
        "unique_supervised_target_tokens",
        "source_count",
        "task_family_count",
        "per_task_token_concentration",
        "per_session_token_concentration",
        "per_teacher_token_concentration",
        "per_source_token_cap",
        "per_family_token_cap",
        "cap_policy_sha256",
        "cap_implementation_commit",
        "cap_implementation_file_sha256",
        "cap_regression_test_file_sha256",
        "prepacking_cap_enforcement_receipt_sha256",
        "admission_receipt_sha256",
    }
    assert all(admission[field] is None for field in required_unknowns)
    assert admission["status"] == "blocked_missing_cap_policy_implementation_and_receipts"
    assert "before packing" in admission["reason"]

    boundary = plan["scientific_boundary"]
    assert boundary["historical_step1000_role"]["label"] == "historical flawed-treatment comparator"
    assert "only that combined treatment" in boundary["causality_boundary"]
    retained_limits = boundary["retained_mismatches_and_limits"]
    assert any("training-seed variance" in limit for limit in retained_limits)
    assert any("262K" in limit for limit in retained_limits)
    followup = plan["fixed_vs_changed"]["smallest_followup_after_confirmed_step1000_regression"]
    assert "matched supervised-target-token exposure" in followup
    assert "no launch authority" in followup

    formulas = plan["phases"]["optimizer_canary"]["formulas"]
    assert formulas == {
        "full_epoch_steps": "S=ceil(accepted_corpus_rows/8)",
        "warmup_steps": "W=ceil(0.05*S)",
        "canary_pause_step": "C=max(W+1,ceil(0.10*S)); require C<S",
        "checkpoint_interval": "C",
    }
    optimizer = plan["phases"]["optimizer_canary"]["proposed_optimizer"]
    assert optimizer["peak_learning_rate"] == 1e-6
    assert optimizer["scheduler"] == "constant_with_warmup"
    assert optimizer["weight_decay"] == 0.01
    assert optimizer["max_grad_norm"] == 1.0
    assert plan["fixed_inputs"]["training_shape"]["epochs"] == 1
    assert plan["phases"]["meaningful_run"]["scientific_changes_from_canary"] == []

    assert not any(plan["operation"].values())
    assert plan["safety"]["every_future_root_job_or_rayjob_annotation"] == {
        "fleet.ai/failure-alerts": "off"
    }


def test_heldout_successor_fails_closed_on_new_corpus_and_outcome_validity() -> None:
    evaluation = _load(EVAL_PATH)

    assert evaluation["predecessor_protocol"]["sha256"] == (
        "sha256:2ac58c3ded7ba5c9bacd27dc3a379c726912a3f0cafc3ee542672de730e1fe0e"
    )
    assert evaluation["selection"]["corrected_corpus_manifest_sha256"] is None
    assert evaluation["selection"]["lineage_reconciliation_sha256"] is None
    assert evaluation["required_runtime_integration"]["accepted_integration_commit"] is None
    assert evaluation["selection"]["task_family_count"] == 20
    assert evaluation["selection"]["source_role_counts"] == {"dev": 13, "final_test": 7}
    assert evaluation["selection"]["pass_k"] == 4
    assert evaluation["selection"]["planned_valid_rollouts_per_arm"] == 80
    assert evaluation["treatment"]["attempt_seeds"] == [
        975414717,
        3965467,
        807861398,
        1399333715,
    ]
    assert evaluation["outcome_policy"]["valid_model_outcomes_are_final"] is True
    assert evaluation["outcome_policy"]["output_limit_and_process_error_are_not_capability_zeros"]
    assert evaluation["outcome_policy"]["zero_imputation_for_missing_or_invalid_cells"] is False
    assert evaluation["treatment"]["job_contract"]["root_metadata_annotations"] == {
        "fleet.ai/failure-alerts": "off"
    }
    analysis = evaluation["reporting"]["analysis_contract"]
    assert analysis["primary_stratum"] == "dev13"
    assert analysis["primary_test"] == {
        "method": "one-sided exact McNemar conditional binomial test",
        "null": "Pr(corrected win | discordant pair) <= 0.5",
        "alternative": "Pr(corrected win | discordant pair) > 0.5",
        "p_value": "Pr[Binomial(D,0.5)>=b] evaluated exactly; when D=0 set p=1",
        "alpha": 0.05,
        "mid_p": False,
    }
    assert analysis["uncertainty"]["method"] == "two-sided 95% Clopper-Pearson exact interval"
    assert analysis["development_decision"]["minimum_absolute_delta"] == {
        "numerator": 3,
        "denominator": 13,
    }
    assert analysis["final_test_role"]["unseal_condition"].startswith("Unseal once only")
    assert analysis["final_test_role"]["directional_consistency_floor"] == {
        "numerator": 2,
        "denominator": 7,
    }
    assert evaluation["reporting"]["union20"] == "descriptive_only"
    assert evaluation["containment"] == {
        "external_evaluations_in_scope": False,
        "external_benchmark_content_present": False,
        "private_task_content_present": False,
        "training_data_eligible": False,
    }
    assert not any(evaluation["operation"].values())
    assert "task_key" not in _all_keys(evaluation)
    assert "task_version_id" not in _all_keys(evaluation)
