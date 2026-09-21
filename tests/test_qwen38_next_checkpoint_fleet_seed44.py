"""Review-only Base/Fresh75/LoRA step-60 Fleet dev17 seed-44 backlog."""

import hashlib
import json
from pathlib import Path

from cyber_post_train.jobs import digest
from evals.fleet import evaluate

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "configs/evaluation"
MASTER = EVAL / "qwen38-next-checkpoints-fleet-dev17-seed44-matched-backlog-v1.json"
BASE = EVAL / "qwen38-base-fleet-dev17-opencode-seed44-pass1-v1.json"
FRESH = EVAL / "qwen38-fresh75-fleet-dev17-opencode-seed44-pass1-v1.json"
FRESH_ARTIFACT = EVAL / "qwen38-fresh75-fleet-dev17-opencode-seed44-pass1-v1.artifact.json"
FRESH_PROVENANCE = EVAL / "qwen38-fresh75-fleet-dev17-opencode-seed43-pass1-v3.json"
FRESH_PROVENANCE_ARTIFACT = (
    EVAL / "qwen38-fresh75-fleet-dev17-opencode-seed43-pass1-v3.artifact.json"
)
LORA = EVAL / "qwen38-lora-step60-fleet-dev17-seed44-preparation-v1.json"
TASK_SET = EVAL / "qwen38-fresh75-fleet-dev17-task-set-v1.json"
SPLIT = ROOT / "configs/data/fleet-blackbox-current-study-split-20260914-v2.json"
SEED43 = EVAL / "qwen38-fleet-dev17-seed43-matched-protocol-v1.json"


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def file_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def assert_self_digest(value: dict) -> None:
    assert value["sha256"] == digest({key: item for key, item in value.items() if key != "sha256"})


def compile_config(path: Path) -> dict:
    value = read(path)
    value.pop("model_artifact_binding", None)
    return evaluate.compile_eval(value, relative_to=path.parent)


def test_preparation_receipts_are_self_digesting_and_create_nothing() -> None:
    master, lora, artifact = read(MASTER), read(LORA), read(FRESH_ARTIFACT)
    for value in (master, lora, artifact):
        assert_self_digest(value)
    assert master["status"] == "prepared_not_launchable"
    assert master["launchable"] is False
    assert lora["status"] == "blocked_not_launchable"
    assert lora["launchable"] is False
    assert artifact["authorization"] == "provenance_only_not_launch_authorization"
    assert master["operation"] == {
        "jobs_submitted": 0,
        "config_maps_created": 0,
        "databases_created": 0,
        "outputs_created": 0,
        "routes_resumed_or_created": 0,
        "evaluation_sessions_created": 0,
    }


def test_seed44_is_a_new_matched_experiment_not_a_seed43_retry() -> None:
    master, seed43 = read(MASTER), read(SEED43)
    assert master["sampling"] == {"temperature": 0.6, "top_p": 0.95, "seed": 44}
    assert seed43["sampling"] == {"temperature": 0.6, "top_p": 0.95, "seed": 43}
    assert master["predecessor_protocol"] == {
        "path": "configs/evaluation/qwen38-fleet-dev17-seed43-matched-protocol-v1.json",
        "file_sha256": file_sha256(SEED43),
        "sha256": "sha256:" + seed43["sha256"],
        "sampling_seed": 43,
        "frozen_and_not_a_retry_parent": True,
    }
    independence = master["independence"]
    assert independence["seed43_cells_replayed"] is False
    assert independence["seed43_results_spliced"] is False
    assert independence["final_eight_task_set_access"] == "sealed"
    census = master["score_blind_live_reconciliation"]["fresh75_seed43_primary"]
    assert census["total"] == census["accepted"] + census["retry_review"] == 17
    assert census["claimed"] == census["pending"] == census["stale_active"] == 0


def test_base_and_fresh_compile_to_the_same_frozen_treatment() -> None:
    master = read(MASTER)
    task_set, split = read(TASK_SET), read(SPLIT)
    plans = {"base": compile_config(BASE), "fresh75": compile_config(FRESH)}
    held = None
    for name, path in (("base", BASE), ("fresh75", FRESH)):
        plan = plans[name]
        arm = master["arms"][name]
        assert arm["config_file_sha256"] == file_sha256(path)
        assert arm["evaluation_plan_sha256"] == "sha256:" + plan["sha256"]
        assert plan["sampling"] == master["sampling"]
        assert plan["images"] == master["images"]
        assert plan["treatment"] == master["treatment"]
        assert plan["pass_k"] == master["pass_k"] == 1
        assert plan["automatic_retry"] is False
        assert plan["max_reviewed_infrastructure_retries"] == 1
        assert plan["training_data_eligible"] is False
        assert len(evaluate.plan_rows(plan)) == 17
        constant = {
            "tasks": plan["tasks"],
            "treatment": plan["treatment"],
            "images": plan["images"],
            "sampling": plan["sampling"],
            "pass_k": plan["pass_k"],
            "retry": plan["max_reviewed_infrastructure_retries"],
        }
        held = constant if held is None else held
        assert constant == held
    dev_ids = {row["task_version_id"] for row in split["evaluation"]["dev"]["tasks"]}
    final_ids = {row["task_version_id"] for row in split["evaluation"]["final_test"]["tasks"]}
    task_ids = {row["task_version_id"] for row in task_set["tasks"]}
    assert task_ids == dev_ids
    assert not task_ids & final_ids
    assert master["selection"]["split_file_sha256"] == file_sha256(SPLIT)
    assert master["selection"]["task_set_file_sha256"] == file_sha256(TASK_SET)


def test_base_and_fresh_declare_the_same_serving_contract() -> None:
    plans = {"base": compile_config(BASE), "fresh75": compile_config(FRESH)}
    routes = {name: next(iter(plan["routes"].values())) for name, plan in plans.items()}
    base, fresh = routes["base"], routes["fresh75"]
    assert (
        base["catalog"]
        == fresh["catalog"]
        == {
            "engine": "sglang",
            "precision": "bf16",
            "tensor_parallel_size": 1,
        }
    )
    assert base["endpoint_origin"] == fresh["endpoint_origin"]
    assert {key: value for key, value in base["model_info"].items() if key != "model_path"} == {
        key: value for key, value in fresh["model_info"].items() if key != "model_path"
    }
    assert {key: value for key, value in base["server_info"].items() if key != "model_path"} == {
        key: value for key, value in fresh["server_info"].items() if key != "model_path"
    }
    for name, plan in plans.items():
        route = routes[name]
        model = next(iter(plan["models"].values()))
        assert model["session_model"] == f"qwen/{route['served_id']}"


def test_fresh_seed44_reuses_only_the_merged_accepted_artifact_chain() -> None:
    master = read(MASTER)
    fresh = read(FRESH)
    packet = read(FRESH_ARTIFACT)
    provenance = read(FRESH_PROVENANCE)
    provenance_packet = read(FRESH_PROVENANCE_ARTIFACT)
    assert master["prepared_from_main"] == "c56fab19877a610911d80d2406948098c4ed74c5"
    assert master["arms"]["fresh75"]["merged_provenance_config_file_sha256"] == file_sha256(
        FRESH_PROVENANCE
    )
    assert fresh["models"] == provenance["models"]
    assert packet["models"] == provenance_packet["models"]
    assert packet["campaign_name"] == fresh["name"]
    binding = fresh["model_artifact_binding"]
    assert binding["packet_path"] == str(FRESH_ARTIFACT.relative_to(ROOT))
    assert binding["packet_file_sha256"] == file_sha256(FRESH_ARTIFACT)
    assert binding["packet_sha256"] == "sha256:" + packet["sha256"]
    route = master["score_blind_live_reconciliation"]["fresh75_route"]
    assert route["phase"] == "paused"
    assert route["active_pods"] == route["ready_pods"] == 0
    assert route["fresh_resume_and_live_parity_required"] is True


def test_lora_stays_fail_closed_until_exact_export_reload_and_route_exist() -> None:
    master, lora = read(MASTER), read(LORA)
    arm = master["arms"]["lora60"]
    assert arm["preparation_file_sha256"] == file_sha256(LORA)
    assert arm["preparation_sha256"] == "sha256:" + lora["sha256"]
    assert arm["model_revision"] is None
    assert arm["session_model"] is None
    assert arm["served_id"] is None
    assert arm["evaluation_config"] is None
    assert arm["evaluation_plan_sha256"] is None
    predecessor = lora["qualification_predecessor"]
    assert predecessor["accepted_export_receipt"] is None
    assert predecessor["accepted_gpu_reload_receipt"] is None
    assert predecessor["accepted_payload_revision"] is None
    assert predecessor["accepted_serving_route"] is None
    assert predecessor["state"] == "qualification_not_accepted"
    assert lora["source_checkpoint"]["gpu_reload_verified"] is False
    assert lora["resource_identities"] == {
        "evaluation_config": None,
        "job": None,
        "config_map": None,
        "database": None,
        "output_root": None,
        "workload": None,
        "pod": None,
        "reason": (
            "Resource identities are minted only after the accepted export, route and "
            "fresh live-parity receipts exist on current main."
        ),
    }


def test_future_jobs_fail_closed_on_alert_capacity_and_duplicate_gates() -> None:
    master, lora = read(MASTER), read(LORA)
    contract = master["future_resource_contract"]
    assert contract["resource_identities_minted"] is False
    assert all(contract[key] is None for key in ("job", "config_map", "database", "output_root"))
    assert contract["root_job_api_version"] == "batch/v1"
    assert contract["root_job_kind"] == "Job"
    assert contract["root_failure_alert_annotation"] == "off"
    assert contract["create_once_annotation"] == "true"
    assert contract["priority_class"] == "c1"
    assert contract["backoff_limit"] == contract["evaluator_gpu_request"] == 0
    assert contract["server_preview_required"] is True
    assert contract["job_configmap_workload_pod_output_database_absence_required"] is True
    assert contract["uncertain_create_must_not_be_repeated"] is True
    assert lora["eventual_root_job_contract"]["fleet.ai/failure-alerts"] == "off"
    assert lora["eventual_root_job_contract"]["priority_class"] == "c1"
    assert lora["eventual_root_job_contract"]["evaluator_gpu_request"] == 0


def test_protocol_forbids_teacher_token_ce_and_requires_complete_matched_arms() -> None:
    master, lora = read(MASTER), read(LORA)
    assert master["teacher_token_heldout_cross_entropy"] is False
    assert lora["evaluation_contract"]["teacher_token_heldout_cross_entropy"] is False
    assert master["terminal_policy"] == {
        "accepted_cells_required_per_arm": 17,
        "comparison_requires_all_three_arms_complete": True,
        "pending_claimed_retry_review_or_terminal_invalid_means_incomplete": True,
        "incomplete_cells_are_not_capability_results": True,
        "cross_arm_splicing_forbidden": True,
        "selective_replay_forbidden": True,
    }
    assert master["retry_policy"]["decision_rule_is_arm_blind"] is True
    assert master["retry_policy"]["cross_experiment_recovery_forbidden"] is True
