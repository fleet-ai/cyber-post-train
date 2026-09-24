import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/evaluation/qwen38-teacher3k96-step350-heldout20-pass4-successor-v1.json"
EVIDENCE = ROOT / "docs/evidence/qwen38-teacher3k96-step350-heldout20-preparation-20260924.json"


def _digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _read(path: Path) -> dict:
    value = json.loads(path.read_text())
    assert value["sha256"] == _digest({key: item for key, item in value.items() if key != "sha256"})
    return value


def test_step350_packet_is_exact_and_fail_closed() -> None:
    value = _read(CONFIG)
    checkpoint = value["candidate_checkpoint"]
    assert value["status"] == "prepared_inert_fail_closed"
    assert value["launchable"] is False
    assert value["repository"]["prepared_against_main_commit"] == (
        "9859a27d09b4fd881dfe04bbf9a1dcf7413083b8"
    )
    assert value["scientific_boundary"]["external_mutations"] == 0
    assert checkpoint["artifact_id"] == "q38-teacher3k-96k-step350"
    assert checkpoint["optimizer_step"] == 350
    assert checkpoint["receipt_file_sha256"] == (
        "sha256:71dd83bdcb3449d3ccdf095530f0b12a9011f6cd415067afad570ecce2502a50"
    )
    assert checkpoint["receipt_sha256"] == (
        "sha256:12c92a50587ab9828f17a8515df1b7978f6be0def34a4eac4886bbd3ebf20e3e"
    )
    assert checkpoint["immutable_manifest_present"] is False
    assert value["training_identity"]["plan_sha256"] == (
        "sha256:6741fb125ce20856df85fa065ccfc8723bb99f327a809aa10725a57fc0558792"
    )
    assert value["baseline_arm"]["revision"] == "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
    assert value["promotion"]["proposed_model_id"] == "chris-q38-t3k96-s350-v1"
    assert value["promotion"]["status"] == "non_executable_future_plan"
    assert value["promotion"]["exact_workload_identity_plan"] is None
    assert (
        value["promotion"]["exact_workload_identity_amendment_required_before_any_create"] is True
    )
    assert value["serving"]["fresh_model_revision"] is None
    assert value["serving"]["fresh_live_parity_receipt_sha256"] is None


def test_step350_packet_preserves_matched_eval_contract() -> None:
    value = _read(CONFIG)
    evaluation = value["evaluation"]
    harness = evaluation["harness"]
    assert evaluation["task_count"] == 20
    assert evaluation["strata"] == {"dev": 13, "final_test": 7}
    assert evaluation["pass_k"] == 4
    assert evaluation["attempt_seeds"] == [975414717, 3965467, 807861398, 1399333715]
    assert evaluation["sampling"] == {"temperature": 0.6, "top_p": 0.95}
    assert harness["version"] == "1.18.27"
    assert harness["context_window_size"] == 262144
    assert harness["context_management"] == "opencode_1.18.27_native_compaction_autocontinue_v2"
    assert harness["tools_in_order"] == ["bash", "submit_report"]
    assert harness["tool_catalog_sha256"] == (
        "sha256:85fad6bdc3a835bf52a11a99b3387740eb06eb3d1720ad9bb33f3feac215b44a"
    )
    assert (
        value["baseline_arm"]["tokenizer_sha256"] == value["training_identity"]["tokenizer_sha256"]
    )
    assert (
        value["baseline_arm"]["chat_template_sha256"]
        == value["training_identity"]["chat_template_sha256"]
    )


def test_step350_jobs_are_c1_alert_off_and_zero_gpu_for_eval() -> None:
    value = _read(CONFIG)
    assert value["training_identity"]["requested_priority_class"] == "c1"
    assert value["training_identity"]["observed_rayjob_priority_class"] == "q1"
    assert value["training_identity"]["observed_effective_workload_priority"] == 10000
    for contract in (value["promotion"]["job_contract"], value["evaluation"]["job_contract"]):
        assert contract["priority_class"] == "c1"
        assert contract["root_metadata_annotations"] == {"fleet.ai/failure-alerts": "off"}
    assert value["evaluation"]["job_contract"]["gpus_per_job"] == 0
    assert value["evaluation"]["job_contract"]["backoff_limit"] == 0
    assert value["evaluation"]["job_contract"]["source_job_count"] == 8


def test_step350_packet_requires_predecessor_acceptance_and_fresh_promotion() -> None:
    value = _read(CONFIG)
    predecessor = value["predecessor_study"]
    assert predecessor["status_required"] == "accepted_terminal_before_successor_launch"
    assert predecessor["source_protocol"]["sha256"] == (
        "sha256:2ac58c3ded7ba5c9bacd27dc3a379c726912a3f0cafc3ee542672de730e1fe0e"
    )
    assert predecessor["baseline_reuse_gate"]["expected_valid_base_cells"] == 80
    assert predecessor["baseline_reuse_gate"]["silent_zero_imputation_forbidden"] is True
    assert predecessor["baseline_reuse_gate"]["accepted_terminal_receipt_set_path"] is None
    assert predecessor["baseline_reuse_gate"]["accepted_terminal_receipt_set_sha256"] is None
    assert (
        predecessor["baseline_reuse_gate"][
            "successor_amendment_importing_exact_receipt_set_required"
        ]
        is True
    )
    assert value["duplicate_and_absence_gates"][
        "promotion_destinations_must_be_absent_before_create"
    ] == [
        "/mnt/sfs/jobs/chris-q38-t3k96-b8-v3/checkpoint-seals-v1/step-350.json",
        "/mnt/sfs/jobs/chris-q38-t3k96-b8-v3/hf-export-step350-v1",
        "/mnt/sfs/jobs/chris-q38-t3k96-b8-v3/hf-export-step350-v1-cpu-check.json",
        "/mnt/sfs/jobs/chris-q38-study-corpora-v1/launch-controls/chris-q38-t3k96-s350-gpu-check-v1",
        "/models/chris-q38-t3k96-s350-v1",
    ]
    assert value["promotion"]["required_gates_in_order"] == [
        "seal_checkpoint_350_with_a_create_once_byte_manifest",
        "create_once_bf16_export_bound_to_the_checkpoint_seal",
        "validate_exact_tensor_layout_dtype_and_sidecar_hashes_on_cpu",
        "perform_a_zero_optimizer_update_gpu_reload_with_finite_output_and_unchanged_source",
        "create_once_immutable_stage_and_acceptance_receipt",
        "register_the_exact_model_paused_at_zero_replicas",
        "perform_fresh_readback_and_live_parity_before_evaluation",
    ]


def test_step350_candidate_execution_identities_are_exact_and_unique() -> None:
    value = _read(CONFIG)
    plan = value["evaluation"]["candidate_execution_identity_plan"]
    assert plan["sha256"] == _digest({key: item for key, item in plan.items() if key != "sha256"})
    assert value["duplicate_and_absence_gates"]["candidate_identity_plan_sha256"] == plan["sha256"]
    rows = plan["rows"]
    assert len(rows) == 8
    assert sum(row["task_count"] for row in rows) == 80
    for field in (
        "job_name",
        "config_map_name",
        "output_root",
        "database",
        "evaluation_config_name",
        "run_id_template",
    ):
        assert len({row[field] for row in rows}) == 8
    assert {row["attempt_seed"] for row in rows} == {
        975414717,
        3965467,
        807861398,
        1399333715,
    }
    assert all(row["partial_root"] is None for row in rows)
    assert "source_protocol.selection.tasks" in plan["derivation"]["task_ordinal"]


def test_step350_evidence_binds_config_and_records_no_mutation() -> None:
    value = _read(CONFIG)
    evidence = _read(EVIDENCE)
    assert evidence["configuration_sha256"] == value["sha256"]
    assert evidence["configuration_file_sha256"] == (
        "sha256:" + hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    )
    assert evidence["effects"] == {
        "cluster_mutations": 0,
        "serving_mutations": 0,
        "eval_launches": 0,
        "model_calls": 0,
        "scoring_calls": 0,
    }
    assert evidence["checks"]["checkpoint350_promotion_destinations_absent"] is True
    assert evidence["checks"]["checkpoint350_serving_route_absent"] is True
    observed = evidence["live_read_only_observation"]
    assert observed["rayjob"]["name"] == value["training_identity"]["api_name"]
    assert observed["rayjob"]["uid"] == value["training_identity"]["rayjob_uid"]
    assert (
        observed["rayjob"]["priority_class"]
        == (value["training_identity"]["observed_rayjob_priority_class"])
    )
    assert observed["workload"]["uid"] == value["training_identity"]["workload_uid"]
    assert (
        observed["workload"]["effective_priority"]
        == (value["training_identity"]["observed_effective_workload_priority"])
    )
    assert (
        observed["checkpoint_350"]["receipt_file_sha256"]
        == (value["candidate_checkpoint"]["receipt_file_sha256"])
    )
    assert (
        observed["checkpoint_350"]["receipt_sha256"]
        == (value["candidate_checkpoint"]["receipt_sha256"])
    )
    assert (
        observed["current_96k_serving_identity"]["model_id"]
        == (value["serving"]["reference_route"]["model_id"])
    )
    assert (
        observed["current_96k_serving_identity"]["revision"]
        == (value["serving"]["reference_route"]["revision"])
    )
