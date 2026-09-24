import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/evaluation/qwen38-teacher3k96-step400-heldout20-pass4-successor-v1.json"
EVIDENCE = ROOT / "docs/evidence/qwen38-teacher3k96-step400-heldout20-preparation-20260924.json"
LINEAGE = ROOT / "configs/data/qwen38-teacher3k-training-lineage-map-20260924-v1.json"
PROTOCOL = ROOT / "configs/evaluation/qwen38-teacher3k-fleet-heldout20-pass4-protocol-v2.json"


def _digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _read(path: Path) -> dict:
    value = json.loads(path.read_text())
    assert value["sha256"] == _digest({key: item for key, item in value.items() if key != "sha256"})
    return value


def test_step400_successor_is_inert_and_contains_no_step350_identity() -> None:
    value = _read(CONFIG)
    serialized = json.dumps(value, sort_keys=True)
    assert value["status"] == "prepared_inert_fail_closed"
    assert value["launchable"] is False
    assert value["repository"] == {
        "prepared_against_main_commit": "0ae3e0923c1e74e65bea3a91f102c1ac7a12917d",
        "derived_from_pr587_commit": "fe26e6483b00a5fc4ac026133009df95801ab81a",
        "lineage_gate_bound_against_main_commit": "ea5f67dd6a955218c4d4ee2946e365189cfd3a8d",
    }
    for stale in ("step350", "step-350", "step_350", "s350", "000350"):
        assert stale not in serialized


def test_step400_checkpoint_and_promotion_facts_remain_unobserved() -> None:
    value = _read(CONFIG)
    checkpoint = value["candidate_checkpoint"]
    assert checkpoint["artifact_id"] == "q38-teacher3k-96k-step400"
    assert checkpoint["optimizer_step"] == 400
    assert checkpoint["path"].endswith("/checkpoints/global_step_400")
    assert checkpoint["receipt_path"].endswith("/checkpoint_receipts/step-000400.json")
    for key in (
        "receipt_file_sha256",
        "receipt_sha256",
        "saved_at_utc",
        "supervised_tokens_seen",
        "observed_file_count",
        "observed_total_bytes",
        "immutable_manifest_present",
    ):
        assert checkpoint[key] is None
    assert value["promotion"]["status"] == "non_executable_future_plan"
    assert value["promotion"]["exact_workload_identity_plan"] is None
    assert value["serving"]["fresh_model_revision"] is None
    assert value["serving"]["fresh_route_uid"] is None
    assert value["serving"]["fresh_registration_sha256"] is None
    assert value["serving"]["fresh_live_parity_receipt_sha256"] is None


def test_step400_lineage_gate_binds_current_main_proof() -> None:
    value = _read(CONFIG)
    lineage = _read(LINEAGE)
    protocol = _read(PROTOCOL)
    gate = value["lineage_gate"]
    assert gate["status"] == "passed"
    assert gate["selection_change_authorized"] is False
    assert gate["training_lineage_map"]["file_sha256"] == (
        "sha256:" + hashlib.sha256(LINEAGE.read_bytes()).hexdigest()
    )
    assert gate["training_lineage_map"]["sha256"] == lineage["sha256"]
    assert gate["training_lineage_map"]["training_task_keys"] == 496
    assert gate["training_lineage_map"]["training_task_versions"] == 1176
    assert gate["training_lineage_map"]["unique_reviewed_atom_lineages"] == 389
    assert gate["training_lineage_map"]["unproven_training_versions"] == 0
    assert gate["heldout_protocol"]["file_sha256"] == (
        "sha256:" + hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    )
    assert gate["heldout_protocol"]["sha256"] == protocol["sha256"]
    assert gate["heldout_protocol"]["lineage_clean_task_families"] == 20
    assert gate["heldout_protocol"]["excluded_exposed_task_families"] == 5
    assert gate["heldout_protocol"]["unproven_task_families"] == 0
    training_atoms = {
        atom for row in lineage["training_task_keys"] for atom in row["atom_lineages"]
    }
    selected_atoms = {
        row["reviewed_task_family"] + ":atom_source" for row in protocol["selection"]["tasks"]
    }
    assert len(training_atoms) == 389
    assert not training_atoms & selected_atoms
    assert gate["proof"] == {
        "exact_task_key_overlap": 0,
        "exact_task_version_overlap": 0,
        "shared_reviewed_atom_lineages": 0,
        "all_selected_task_families_have_reviewed_lineage": True,
    }


def test_step400_packet_generation_is_blocked_until_checkpoint_is_complete_and_sealed() -> None:
    value = _read(CONFIG)
    gate = value["packet_generation_gate"]
    checkpoint = value["candidate_checkpoint"]
    assert gate["status"] == "blocked_checkpoint400_not_complete_or_sealed"
    assert gate["promotion_packet_generation_authorized"] is False
    assert gate["evaluation_packet_generation_authorized"] is False
    assert gate["first_openable_gate"] == "checkpoint400_complete_and_immutable_seal_accepted"
    assert checkpoint["receipt_file_sha256"] is None
    assert checkpoint["receipt_sha256"] is None
    assert checkpoint["immutable_manifest_present"] is None
    required = set(gate["checkpoint400_required_evidence"])
    assert {
        "checkpoint_receipt_file_sha256",
        "checkpoint_receipt_sha256",
        "receipt_optimizer_step_equals_400",
        "receipt_checkpoint_path_matches_frozen_path",
        "receipt_source_plan_sha256_matches_training_plan",
        "seal_manifest_file_sha256",
        "seal_manifest_sha256",
        "seal_optimizer_step_equals_400",
        "seal_checkpoint_path_matches_frozen_path",
        "seal_source_plan_sha256_matches_training_plan",
        "seal_source_unchanged_before_and_after",
    } <= required


def test_step400_packet_reuses_current_main_promotion_and_eval_rails() -> None:
    value = _read(CONFIG)
    assert value["existing_rails"] == {
        "checkpoint_receipt_and_seal": "training/checkpoints.py:receipt,seal,verify",
        "bf16_export": "training/export.py:export",
        "cpu_layout_and_zero_update_reload": "training/export_check.py:check",
        "immutable_stage": "training/model_stage.py:read_plan,validate_plan,execute_stage",
        "paused_serving_registration": (
            "training/checkpoint_serving_route.py:prepare,execute,lifecycle"
        ),
        "live_serving_parity": "training/checkpoint_serving_parity.py:capture",
        "fleet_heldout_compile_and_run": (
            "evals/fleet/evaluate.py:compile_eval,prepare,preflight,run"
        ),
        "fleet_heldout_create_once_launch": (
            "evals/fleet/heldout_launch.py:build_package,sealed_evaluation,"
            "duplicate_census,launch_once,collect_terminal"
        ),
    }
    for binding in value["existing_rails"].values():
        path, functions = binding.split(":", 1)
        source = (ROOT / path).read_text()
        assert (ROOT / path).is_file()
        assert all(f"def {function}(" in source for function in functions.split(","))


def test_step400_preserves_exact_heldout20_contract() -> None:
    value = _read(CONFIG)
    predecessor = value["predecessor_study"]
    evaluation = value["evaluation"]
    harness = evaluation["harness"]
    assert predecessor["source_protocol"]["sha256"] == (
        "sha256:2ac58c3ded7ba5c9bacd27dc3a379c726912a3f0cafc3ee542672de730e1fe0e"
    )
    assert predecessor["fresh_study_plan"]["sha256"] == (
        "sha256:edb1bbfcc48b3c97c5ad01602dd668cae3aa7603125f3a25acf5b2b55840c39f"
    )
    assert evaluation["task_count"] == 20
    assert evaluation["strata"] == {"dev": 13, "final_test": 7}
    assert evaluation["pass_k"] == 4
    assert evaluation["candidate_rollouts"] == 80
    assert evaluation["baseline_rollouts_reused"] == 80
    assert evaluation["new_total_rollouts"] == 80
    assert evaluation["attempt_seeds"] == [975414717, 3965467, 807861398, 1399333715]
    assert evaluation["sampling"] == {"temperature": 0.6, "top_p": 0.95}
    assert harness["version"] == "1.18.27"
    assert harness["context_management"] == "opencode_1.18.27_native_compaction_autocontinue_v2"
    assert harness["context_window_size"] == 262144
    assert harness["compaction_headroom_tokens"] == 20000
    assert harness["max_output_tokens"] == 32768
    assert harness["max_model_requests"] == 600
    assert harness["timeout_seconds"] == 28800
    assert harness["tools_in_order"] == ["bash", "submit_report"]


def test_step400_preserves_runtime_contract_and_create_only_destinations() -> None:
    value = _read(CONFIG)
    candidate = value["serving"]["candidate_route"]
    assert candidate == {
        "model_id": "chris-q38-t3k96-s400-v1",
        "source_path": "/models/chris-q38-t3k96-s400-v1",
        "precision": "bf16",
        "quantization": None,
        "tensor_parallel_size": 1,
        "data_parallel_size": 8,
        "context_length": 262144,
        "kv_cache_dtype": "fp8_e4m3",
        "runtime_image": (
            "lmsysorg/sglang@sha256:"
            "d6e7288627be8b02be88e4bba38e73f6d50e2826869f753c13a4c4385ab3eda9"
        ),
        "reasoning_parser": "qwen3",
        "tool_call_parser": "qwen3_coder",
        "priority_class": "c1",
        "initial_state": "paused_zero_replicas",
        "allowed_differences_from_reference": [
            "display_name",
            "model_id",
            "model_path",
            "model_revision",
            "source_path",
            "served_model_name",
            "idempotency_key",
            "request_fingerprint",
        ],
    }
    destinations = value["duplicate_and_absence_gates"][
        "promotion_destinations_must_be_absent_before_create"
    ]
    assert len(destinations) == 5
    assert all("400" in destination for destination in destinations)
    assert value["promotion"]["job_contract"] == {
        "priority_class": "c1",
        "root_metadata_annotations": {"fleet.ai/failure-alerts": "off"},
        "preview_required_before_create": True,
        "patch_after_create_forbidden": True,
        "create_once_destinations_required": True,
    }


def test_step400_execution_identities_are_exact_unique_and_unbound() -> None:
    value = _read(CONFIG)
    plan = value["evaluation"]["candidate_execution_identity_plan"]
    assert plan["sha256"] == _digest({key: item for key, item in plan.items() if key != "sha256"})
    assert value["duplicate_and_absence_gates"]["candidate_identity_plan_sha256"] == plan["sha256"]
    assert plan["campaign_id"] == "q38-heldout20-t3k96-s400-p4-v1"
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
        assert all("400" in row[field] for row in rows)
    assert {row["attempt_seed"] for row in rows} == {
        975414717,
        3965467,
        807861398,
        1399333715,
    }
    assert all(row["partial_root"] is None for row in rows)
    assert plan["runtime_fields_required_null_before_create"] == [
        "job_uid",
        "config_map_uid",
        "pod_uid",
        "workload_uid",
        "session_id",
        "experiment_identity_sha256",
    ]


def test_step400_predecessor_receipts_remain_fail_closed() -> None:
    value = _read(CONFIG)
    gate = value["predecessor_study"]["baseline_reuse_gate"]
    assert gate["expected_valid_base_cells"] == 80
    assert gate["normal_completion_required_for_every_reused_cell"] is True
    assert gate["accepted_terminal_receipt_set_path"] is None
    assert gate["accepted_terminal_receipt_set_sha256"] is None
    assert gate["successor_amendment_importing_exact_receipt_set_required"] is True
    assert gate["silent_zero_imputation_forbidden"] is True


def test_step400_evidence_binds_only_frozen_facts() -> None:
    value = _read(CONFIG)
    evidence = _read(EVIDENCE)
    assert evidence["configuration_sha256"] == value["sha256"]
    assert evidence["configuration_file_sha256"] == (
        "sha256:" + hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    )
    assert evidence["status"] == "prepared_inert_fail_closed"
    assert evidence["observed_at_utc"] is None
    assert evidence["checks"] == {
        "launchable": False,
        "lineage_gate_bound": True,
        "checkpoint400_complete": False,
        "checkpoint_receipt_bound": False,
        "checkpoint_seal_bound": False,
        "packet_generation_authorized": False,
        "promotion_chain_bound": False,
        "predecessor_baseline_receipts_bound": False,
        "fresh_prelaunch_receipts_bound": False,
    }
    for section in (
        "future_checkpoint_observation",
        "future_promotion_observation",
        "future_predecessor_observation",
        "future_launch_observation",
    ):
        for key, item in evidence[section].items():
            if key not in {"optimizer_step", "path", "receipt_path"}:
                assert item is None, (section, key)
    assert evidence["effects"] == {
        "cluster_mutations": 0,
        "serving_mutations": 0,
        "eval_launches": 0,
        "model_calls": 0,
        "scoring_calls": 0,
    }
