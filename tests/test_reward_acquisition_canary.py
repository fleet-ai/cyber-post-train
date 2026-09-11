from __future__ import annotations

import hashlib
import json
from pathlib import Path

from training.io import digest_json
from training.rl_preview import rl_paid_launch_blockers

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "configs/runs/qwen36-27b-native-rl-reward-acquisition-canary.pre-submit.json"
PREVIEW_PATH = ROOT / "configs/runs/qwen36-27b-native-rl-reward-acquisition-canary.preview.json"


def _plan() -> dict:
    return json.loads(PLAN_PATH.read_text())


def test_reward_acquisition_canary_is_exactly_shaped_and_explicitly_blocked() -> None:
    plan = _plan()
    request = plan["run_request_template"]
    source_path = ROOT / plan["source_evidence"]["two_task_gate"]
    source = json.loads(source_path.read_text())

    assert plan["status"] == "blocked_pre_submission"
    assert plan["paid_submission_authorized"] is False
    assert [row["source_task_version_id"] for row in plan["task_successors"]] == [
        "54425601-6fd2-43d8-8cb9-e565b767676a",
        "f31ebe83-0ff1-4660-bcba-59ffa4b82d5a",
    ]
    assert plan["source_evidence"]["two_task_gate_file_sha256"] == (
        f"sha256:{hashlib.sha256(source_path.read_bytes()).hexdigest()}"
    )
    source_by_id = {row["task_version_id"]: row for row in source["tasks"]}
    for successor in plan["task_successors"]:
        frozen = source_by_id[successor["source_task_version_id"]]
        assert successor["task_key"] == frozen["task_key"]
        assert successor["source_task_version"] == frozen["task_version"]
        for field in (
            "environment_version_id",
            "env_key",
            "env_version",
            "data_key",
            "data_version",
        ):
            assert successor[field] == frozen[field]
    expected_successors = {
        "54425601-6fd2-43d8-8cb9-e565b767676a": (
            "c99340e2-3801-5c3c-a50c-3b96cee4572f",
            "9",
        ),
        "f31ebe83-0ff1-4660-bcba-59ffa4b82d5a": (
            "ab5f2956-9fbb-54fb-afd8-67fe11402fa4",
            "13",
        ),
    }
    assert all(row["status"] == "created_and_identity_verified" for row in plan["task_successors"])
    for row in plan["task_successors"]:
        assert (
            row["metadata_only_successor_task_version_id"],
            row["metadata_only_successor_task_version"],
        ) == expected_successors[row["source_task_version_id"]]

    task_versions = request["tasks"]["task_versions"]
    assert [row["task_version_id"] for row in task_versions] == [
        "c99340e2-3801-5c3c-a50c-3b96cee4572f",
        "ab5f2956-9fbb-54fb-afd8-67fe11402fa4",
    ]
    for successor, bound in zip(plan["task_successors"], task_versions, strict=True):
        assert bound["task_key"] == successor["task_key"]
        assert bound["task_version_id"] == successor["metadata_only_successor_task_version_id"]
        assert bound["task_version"] == successor["metadata_only_successor_task_version"]
        for field in (
            "environment_version_id",
            "env_key",
            "env_version",
            "data_key",
            "data_version",
        ):
            assert bound[field] == successor[field]
        assert bound["env_variables"] == {}
        assert bound["env_variable_deletions"] == []

    operation = plan["successor_operation"]
    assert operation["schema"] == "fleet_exact_tool_successors_v1"
    assert operation["idempotency_key"] == "qwen36-harness-parity-bash-submit-report-v1"
    assert operation["production_revision"] == "bb9cc39d9386dc4376cb501c72908444ed0de033"
    assert operation["created_count"] == 2
    assert operation["reused_count"] == 0
    assert operation["ordered_metadata_tools"] == ["bash", "submit_report"]
    assert operation["identity_receipt_assertions"] == {
        "all_non_tool_fields_equal": True,
        "only_semantic_diff_path": "/metadata/tools",
        "source_parent_rows_unchanged": True,
        "source_current_version_pointers_unchanged": True,
        "successors_promoted_to_current": False,
        "all_or_nothing_transaction": True,
    }
    assert request["trainer"]["trainer_version_id"] is None

    assert request["num_workers"] == 1
    assert request["gpus_per_worker"] == 8
    assert request["model"] == {
        "staged_model": "qwen3.6-27b",
        "precision": "bf16",
        "engine_tensor_parallel_size": 4,
        "max_context_length": 65_536,
        "max_prompt_length": 16_384,
        "max_generate_length": 49_152,
        "sequence_parallel_size": 1,
    }
    assert request["grpo"] == {
        "group_size": 4,
        "train_batch_size": 2,
        "policy_mini_batch_size": 2,
        "kl_coefficient": 0.001,
        "kl_estimator": "k3",
        "policy_loss_type": "regular",
        "checkpoint_interval": 1,
        "max_checkpoints_to_keep": 1,
        "learning_rate": 0.000001,
        "max_steps": 1,
        "temperature": 1.0,
    }
    assert "trainer.max_training_steps=1" in request["trainer"]["args"]
    assert "generator.sampling_params.top_p=1.0" in request["trainer"]["args"]
    assert request["rollout"] == {
        "harness": "native",
        "mode": "tool-use",
        "max_turns": 600,
        "max_tokens_per_turn": 2048,
        "tool_result_max_chars": 16000,
        "required_task_tools": ["bash", "submit_report"],
        "partial_verifier_scoring": False,
        "pass_conversation_to_verifier": False,
        "multi_app_aggregation_mode": "binary",
    }
    assert request["eval"]["task_versions"] == []
    assert request["eval"]["before_train"] is False
    assert plan["derived_shape"] == {
        "accelerator": "B300",
        "workers": 1,
        "gpus_per_worker": 8,
        "total_gpus": 8,
        "engine_tensor_parallel_size": 4,
        "num_inference_engines": 2,
        "planned_rollouts": 8,
        "true_optimizer_steps": 1,
        "checkpoints_retained": 1,
        "evaluation_episodes": 0,
        "context_compaction": "disabled_single_flat_trajectory",
    }


def test_checked_in_canary_cannot_pass_the_current_paid_launch_gate() -> None:
    request = _plan()["run_request_template"]
    preview = {
        "manifest_yaml": (
            "spec:\n"
            "  entrypoint: python -m rl_rollout.entrypoint "
            "trainer.epochs=1 trainer.max_training_steps=1\n"
        )
    }

    blockers = rl_paid_launch_blockers(request, preview)

    assert blockers == [
        "Train API preview lacks authoritative task_tool_allowlist_evidence; exact "
        "task-version metadata.tools must be populated and version-bound before paid RL"
    ]


def test_exact_successors_can_pass_the_local_tool_evidence_gate() -> None:
    request = _plan()["run_request_template"]
    bindings = [
        {"task_version_id": row["task_version_id"], "tools": ["bash", "submit_report"]}
        for row in request["tasks"]["task_versions"]
    ]
    preview = {
        "manifest_yaml": (
            "spec:\n"
            "  entrypoint: python -m rl_rollout.entrypoint "
            "trainer.epochs=1 trainer.max_training_steps=1\n"
        ),
        "task_tool_allowlist_evidence": {
            "schema": "fleet_rl_task_tool_allowlists_v1",
            "source": "authoritative_task_version_metadata",
            "source_field": "metadata.tools",
            "bindings": bindings,
            "bindings_sha256": digest_json(bindings),
        },
    }

    assert rl_paid_launch_blockers(request, preview) == []


def test_authoritative_preview_receipt_binds_the_exact_successors_without_submission() -> None:
    plan = _plan()
    request = plan["run_request_template"]
    receipt = json.loads(PREVIEW_PATH.read_text())

    assert plan["source_evidence"]["authoritative_preview_receipt"] == str(
        PREVIEW_PATH.relative_to(ROOT)
    )
    assert receipt["submitted"] is False
    assert receipt["http_status"] == 200
    assert receipt["errors"] == []
    assert receipt["source_config_canonical_sha256"] == digest_json(request)
    assert plan["source_evidence"][
        "authoritative_preview_source_config_canonical_sha256"
    ] == digest_json(request)

    evidence = receipt["task_tool_allowlist_evidence"]
    assert evidence["schema"] == "fleet_rl_task_tool_allowlists_v1"
    assert evidence["source"] == "authoritative_task_version_metadata"
    assert evidence["source_field"] == "metadata.tools"
    assert evidence["bindings"] == [
        {"task_version_id": row["task_version_id"], "tools": ["bash", "submit_report"]}
        for row in request["tasks"]["task_versions"]
    ]
    assert evidence["bindings_sha256"] == digest_json(evidence["bindings"])

    assignments = receipt["manifest"]["entrypoint_exact_assignments"]
    assert assignments["trainer.max_training_steps"] == "1"
    assert assignments["trainer.max_prompt_length"] == "16384"
    assert assignments["generator.sampling_params.max_generate_length"] == "49152"
    assert receipt["manifest"]["trainer_version_id"] is None
    assert receipt["prompt_schema_token_preflight"]["status"] == "blocked"
    assert receipt["scientific_interpretation"] == {
        "authoritative_tool_gate": "passed",
        "exact_one_step_render_gate": "passed",
        "trainer_catalog_readiness": "not_proven",
        "prompt_schema_token_fit": "not_proven",
        "paid_launch_ready": False,
        "paid_job_created": False,
    }


def test_canary_names_every_nonadministrative_launch_blocker() -> None:
    plan = _plan()
    blocker_ids = [row["id"] for row in plan["launch_blockers"]]
    assert blocker_ids == [
        "trainer_catalog_readiness",
        "prompt_schema_token_preflight",
        "per_episode_verifier_identity",
    ]
    assert "approval" not in json.dumps(plan["launch_blockers"]).lower()


def test_feasibility_and_multi_call_semantics_are_measured_not_launch_blockers() -> None:
    plan = _plan()
    outcome = plan["canary_outcome_contract"]["b300_65k_feasibility"]
    diagnostic = plan["diagnostic_protocol_differences"]

    assert "infrastructure/configuration failure" in outcome["failure_classification"]
    assert "Do not claim an optimizer result" in outcome["prohibited_claim_on_failure"]
    assert [row["id"] for row in diagnostic] == ["multiple_tool_calls_per_turn"]
    assert diagnostic[0]["status"] == "accepted_for_native_only_reward_acquisition_canary"
    assert "Agent Runtime harness parity" in diagnostic[0]["later_parity_gate"]
