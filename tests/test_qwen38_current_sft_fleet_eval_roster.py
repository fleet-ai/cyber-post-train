"""No-launch roster for current Qwen3.8 SFT Fleet dev17 candidates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from cyber_post_train.jobs import digest

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "configs/evaluation"
ROSTER = EVAL / "qwen38-current-sft-fleet-dev17-eval-roster-20260922-v1.json"
PROTOCOL = EVAL / "qwen38-fleet-dev17-seed43-matched-protocol-v1.json"
TASK_SET = EVAL / "qwen38-fresh75-fleet-dev17-task-set-v1.json"
BINDINGS = EVAL / "qwen38-fleet-dev17-exact-binding-roster-20260922-v1.json"
SPLIT = ROOT / "configs/data/fleet-blackbox-current-study-split-20260914-v2.json"
BASE = EVAL / "qwen38-base-fleet-dev17-opencode-seed43-pass1-v1.json"


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def file_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def test_roster_is_self_bound_and_authorizes_no_mutation() -> None:
    roster = read(ROSTER)
    assert roster["sha256"] == digest(
        {key: value for key, value in roster.items() if key != "sha256"}
    )
    assert roster["status"] == "prepared_no_launch"
    assert roster["launchable"] is False
    assert all(value == 0 for value in roster["operation"].values())
    assert roster["scope"]["capability_scores_or_private_traces_accessed"] is False
    assert roster["scope"]["final_test_split_access"] == "sealed"


def test_roster_freezes_exact_seed43_opencode_protocol() -> None:
    roster = read(ROSTER)
    frozen = roster["frozen_comparison"]
    protocol, task_set, bindings = read(PROTOCOL), read(TASK_SET), read(BINDINGS)
    assert frozen["protocol_file_sha256"] == file_sha256(PROTOCOL)
    assert frozen["protocol_sha256"] == "sha256:" + protocol["sha256"]
    assert frozen["task_set_file_sha256"] == file_sha256(TASK_SET)
    assert frozen["selection_sha256"] == task_set["selection_sha256"]
    assert frozen["binding_roster_file_sha256"] == file_sha256(BINDINGS)
    assert frozen["bindings_sha256"] == bindings["bindings_sha256"]
    assert frozen["split_file_sha256"] == file_sha256(SPLIT)
    assert frozen["task_count"] == len(task_set["tasks"]) == len(bindings["bindings"]) == 17
    assert all(task["task_version_id"] for task in task_set["tasks"])
    assert frozen["mutable_task_keys_allowed"] is False
    assert frozen["exact_task_version_ids_required"] is True
    assert frozen["harness"] == "opencode"
    assert frozen["harness_version"] == "1.18.27"
    assert frozen["context_management"] == ("opencode_1.18.27_native_compaction_autocontinue_v2")
    assert frozen["context_window_size"] == 262144
    assert frozen["sampling"] == {"temperature": 0.6, "top_p": 0.95, "seed": 43}
    assert frozen["pass_k"] == 1
    assert frozen["training_data_eligible"] is False


def test_accepted_base_is_complete_but_conditionally_reusable() -> None:
    base = read(ROSTER)["accepted_base_control"]
    assert base["config_file_sha256"] == file_sha256(BASE)
    assert base["planned_cells"] == base["accepted_cells"] == base["valid_cells"] == 17
    assert base["automatic_checker_results"] == 17
    assert base["technical_invalid_cells"] == 0
    assert base["result_authority"]["outcome_not_copied_into_roster"] is True
    assert "fresh live parity" in base["reuse_rule"]
    assert "Never selectively replay" in base["reuse_rule"]


def test_each_arm_selects_only_an_exact_accepted_native_receipt() -> None:
    roster = read(ROSTER)
    candidates = {row["arm_id"]: row for row in roster["candidates"]}
    assert list(candidates) == [
        "q38-t3k32-lead",
        "q38-t3k32-lr1",
        "q38-sft32-dense-m1",
        "q38-d32-b16-lr5e6",
        "q38-t3k64-b8",
        "q38-t3k96-b8",
    ]
    expected = {
        "q38-t3k32-lead": (
            1000,
            "sha256:bae4b7215a6b9ebd38461555a2de02dfe822013222280ee0ca6498ad27e892b7",
            31035607,
        ),
        "q38-t3k32-lr1": (
            700,
            "sha256:fca45725b3ccede801c6fef5f143c51c508a0872413cbd0bbdf734d970fdbe98",
            21596883,
        ),
        "q38-sft32-dense-m1": (
            600,
            "sha256:3eebf6a6bb06cf3cb6e990fb492d071d68c40a6c87825d9beb141f4a32f33288",
            18752000,
        ),
        "q38-d32-b16-lr5e6": (
            300,
            "sha256:cf48c5d3542581b84c9ee888b6e77921f2fd469d6dabd4d4f38622a64c1e2e5c",
            18752000,
        ),
        "q38-t3k64-b8": (
            285,
            "sha256:fb0068f4d3f60ded3d3a725273343672822eeaeea718f4802edbc33a67e432d0",
            15380675,
        ),
        "q38-t3k96-b8": (
            100,
            "sha256:e1dfd6718926f6d19ec5710ed7ba53e753d81526f95dbd8313cc8a72d564e06d",
            6734083,
        ),
    }
    for arm_id, (step, receipt, tokens) in expected.items():
        row, checkpoint = candidates[arm_id], candidates[arm_id]["selected_checkpoint"]
        assert checkpoint["optimizer_step"] == row["latest_accepted_checkpoint_step"] == step
        assert checkpoint["checkpoint_receipt_sha256"] == receipt
        assert checkpoint["supervised_tokens"] == tokens
        assert checkpoint["world_size"] == 8
        assert checkpoint["file_count"] == 33
        assert row["latest_observed_training_step"] >= step
        assert row["newer_metric_is_not_checkpoint_evidence"] is (
            row["latest_observed_training_step"] > step
        )


def test_lead_retargets_to_step1000_and_keeps_step900_only_as_fallback() -> None:
    candidates = {row["arm_id"]: row for row in read(ROSTER)["candidates"]}
    lead = candidates["q38-t3k32-lead"]
    assert lead["selected_checkpoint"]["optimizer_step"] == 1000
    assert lead["promotion_state"] == "native_checkpoint_only"
    assert lead["serving_model_id"] is None
    assert lead["route_state"] == "absent"
    assert lead["live_parity_state"] == "not_run"
    assert lead["eval_state"] == "blocked_before_promotion"
    assert lead["fallback_checkpoint"] == {
        "artifact_id": "q38-teacher3k-32k-step900",
        "optimizer_step": 900,
        "checkpoint_receipt_sha256": (
            "sha256:f7eb50db657486f901db6b03e1a17ebf0dcd0d0124c8c7491cb5a405110e54e8"
        ),
        "promotion_evidence": "docs/evidence/qwen38-teacher3k32-step900-preservation-20260922.json",
        "serving_model_id": "chris-q38-t3k32-s900-v1",
        "route_state": "paused_routing_disabled_zero_pods",
        "selected_for_next_eval": False,
        "reason": "superseded_by_newer_accepted_step1000_before_live_parity",
    }
    assert lead["retarget_before_live_parity"] == {
        "current_selection_step": 1000,
        "next_accepted_receipt_wins": True,
        "promotion_required": True,
        "resource_identities_minted": False,
    }


def test_other_native_only_rows_do_not_skip_promotion() -> None:
    candidates = {row["arm_id"]: row for row in read(ROSTER)["candidates"]}
    for arm_id in (
        "q38-t3k32-lr1",
        "q38-sft32-dense-m1",
        "q38-d32-b16-lr5e6",
        "q38-t3k64-b8",
    ):
        row = candidates[arm_id]
        assert row["promotion_state"] == "native_checkpoint_only"
        assert row["serving_model_id"] is None
        assert row["route_state"] == "absent"
        assert row["eval_state"] == "blocked_before_promotion"
    ninety_six = candidates["q38-t3k96-b8"]
    assert ninety_six["promotion_state"] == "accepted_through_cpu_validation_gpu_reload_absent"
    assert ninety_six["serving_model_id"] is None
    assert ninety_six["route_state"] == "absent"
    assert ninety_six["eval_state"] == "blocked_before_gpu_reload"


def test_64k_retargets_to_step285_and_keeps_step270_only_as_fallback() -> None:
    row = {row["arm_id"]: row for row in read(ROSTER)["candidates"]}["q38-t3k64-b8"]
    assert row["selected_checkpoint"]["optimizer_step"] == 285
    assert row["fallback_checkpoint"] == {
        "artifact_id": "q38-t3k64-b8-step270",
        "optimizer_step": 270,
        "checkpoint_receipt_sha256": (
            "sha256:31a4db6702a6442f92051bb27885e1f45d1bfe865ceefdd00a0a7a34be655d80"
        ),
        "selected_for_next_eval": False,
        "reason": "superseded_by_newer_accepted_step285_before_promotion",
    }


def test_atomic_latest_rule_applies_to_every_running_arm() -> None:
    roster = read(ROSTER)
    assert roster["selection_policy"]["unit"] == "training_arm"
    assert roster["selection_policy"]["candidate_key"] == [
        "source_rayjob_uid",
        "checkpoint_receipt_sha256",
    ]
    latest = roster["selection_policy"]["latest_rule"]
    assert "greatest optimizer step" in latest
    assert "independently validated immutable native checkpoint receipt" in latest
    assert "never supersedes an accepted receipt" in latest
    assert len(roster["scope"]["included_training_arms"]) == len(roster["candidates"]) == 6
    assert set(roster["scope"]["included_training_arms"]) == {
        row["source_run"]["api_name"] for row in roster["candidates"]
    }


def test_capacity_and_launch_gates_stay_closed() -> None:
    roster = read(ROSTER)
    capacity = roster["capacity_snapshot"]
    assert capacity["current_logical_nodes"] == 8
    assert capacity["current_logical_gpus"] == 64
    assert capacity["projected_if_resumed_nodes"] == 9
    assert capacity["projected_if_resumed_gpus"] == 72
    assert capacity["within_eight_node_cap"] is False
    gates = "\n".join(roster["remaining_gates"])
    for required in (
        "all-namespace capacity census",
        "live parity receipt",
        "task_version_id roster",
        "duplicate identities are absent",
        "Two byte-equivalent Kubernetes server dry-runs",
        "fleet.ai/failure-alerts=off",
    ):
        assert required in gates
