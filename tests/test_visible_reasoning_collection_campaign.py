"""Source-only tests for the first large Qwen visible-reasoning campaign."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from training import collection_campaign
from training import visible_reasoning_collection_campaign as campaign

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "configs/collection/qwen38-self-visible-reasoning-train50-target20m-v1.source.json"
OUTPUT = ROOT / "configs/collection/qwen38-self-visible-reasoning-train50-target20m-v1"
INVENTORY_SHA = "sha256:d84b822611129059a4d60165a2ea7f2e0d44f084f2d74c45f904b445a83e0fbf"
SPLIT_SHA = "sha256:8859da6ac391add1b405ee5c453924657dfd9a14b2826dfa2a681456556af4df"
ANCHOR_SHA = "sha256:48350b8fc23143abe297db3b2364590c72d564553ebb4e9a349fa06ec246a26b"
LOCK_SHA = "sha256:62ff0f7c1ade3c6ed6e0b5be025efac10f496617d99f369149a8aa1c8253339e"
RUNTIME_SHA = "sha256:849542785db0cb71fdc9ecd29e1008c8af26e0bf25ad0dd74e22b6d6fb3cb694"
SELECTION_SHA = "sha256:93e264d7cc5a7c90101f4c510a37a2d02fd4c87e94845a4012db7fa0d486d923"


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _reseal(value: dict) -> dict:
    value["sha256"] = collection_campaign.canonical_digest(
        {name: item for name, item in value.items() if name != "sha256"}
    )
    return value


@pytest.fixture(scope="module")
def rendered() -> dict[str, dict]:
    return campaign.render(_load(SPEC), root=ROOT)


def test_committed_large_campaign_is_reproducible_and_source_only(rendered) -> None:
    campaign.check(OUTPUT, rendered)
    receipt = rendered["review-receipt.json"]
    assert receipt["status"] == "source_only_review_required"
    assert receipt["train_task_versions"] == 50
    assert receipt["heldout_task_versions"] == 25
    assert receipt["planned_cells"] == 20_000
    assert receipt["minimum_unique_supervised_tokens"] == 20_000_000
    assert receipt["private_or_hidden_reasoning_allowed"] is False
    assert receipt["external_submission"] is False
    assert receipt["review_completed"] is False


def test_campaign_uses_all_train_families_and_zero_heldout_cells(rendered) -> None:
    plan = rendered["campaign-plan.json"]
    boundary = plan["task_boundary"]
    assert boundary == {
        "inventory_sha256": INVENTORY_SHA,
        "family_split_sha256": SPLIT_SHA,
        "root_role_anchor_id": "fleet-blackbox-current-study-20260914-v2",
        "family_role_anchor_sha256": ANCHOR_SHA,
        "protected_family_lock_sha256": LOCK_SHA,
        "runtime_bindings_sha256": RUNTIME_SHA,
        "task_selection_sha256": SELECTION_SHA,
        "train_task_versions": 50,
        "heldout_task_versions": 25,
        "heldout_collection_cells": 0,
    }
    assert plan["collection"]["minimum_successful_families"] == 20
    assert plan["collection"]["maximum_family_target_token_fraction"] == 0.25


def test_wave_plan_is_balanced_create_once_and_target_driven(rendered) -> None:
    plan = rendered["campaign-plan.json"]
    wave_plan = rendered["wave-plan.json"]
    waves = wave_plan["waves"]
    assert len(waves) == 40
    assert sum(wave["planned_cells"] for wave in waves) == 20_000
    assert all(wave["task_versions"] == 50 for wave in waves)
    assert all(wave["planned_cells"] == 500 for wave in waves)
    assert [(wave["attempt_first"], wave["attempt_last"]) for wave in waves] == [
        (1 + index * 10, 10 + index * 10) for index in range(40)
    ]
    assert [(wave["seed_first"], wave["seed_last"]) for wave in waves] == [
        (43 + index * 10, 52 + index * 10) for index in range(40)
    ]
    assert wave_plan["scheduling"] == {
        "maximum_rollouts_per_day": 500,
        "one_full_wave_per_day": True,
        "finish_active_wave_before_target_stop": True,
        "start_next_wave_only_if_aggregate_target_not_reached": True,
    }
    identity = plan["identity"]
    assert identity["planned_unique_cell_ids"] == 20_000
    assert identity["ambiguous_cell_replay_allowed"] is False
    assert identity["cell_identity_universe_sha256"].startswith("sha256:")


def test_campaign_only_accepts_explicit_student_visible_qwen_reasoning(rendered) -> None:
    plan = rendered["campaign-plan.json"]
    source = plan["source_treatment"]
    assert source["kind"] == "qwen_self"
    assert source["thinking"] == {
        "enable_thinking": True,
        "preserve_thinking": True,
        "reasoning_visibility": "student_visible",
    }
    assert source["opencode"]["context_window_tokens"] == 262_144
    assert source["opencode"]["context_headroom_tokens"] == 20_000
    assert source["opencode"]["tools"] == ["bash", "submit_report"]
    rejected = set(plan["admission"]["reject"])
    assert {
        "private_or_unknown_reasoning",
        "teacher_or_provider_private_reasoning",
        "inferred_or_reconstructed_reasoning",
        "unknown_serialization",
    } <= rejected
    assert plan["matched_ablation"] == {
        "derive_action_only_from_same_selected_turns": True,
        "visible_action_spans_identical": True,
        "reasoning_spans_masked_only_in_action_arm": True,
        "mix_corpora": False,
    }


def test_compaction_requires_exact_visible_summary_lineage(rendered) -> None:
    compaction = rendered["campaign-plan.json"]["compaction"]
    assert compaction == {
        "online": "opencode_1.18.27_native_compaction_autocontinue_v2",
        "accepted_offline_kind": "student_generated_exact_continuation_v1",
        "actual_post_summary_prompt_required": True,
        "compaction_summary_loss_mask": 0,
        "opaque_or_unreconstructable_continuation": "reject_target_and_later_continuation",
    }


def test_20m_budget_has_an_explicit_aggregate_planning_basis(rendered) -> None:
    basis = rendered["campaign-plan.json"]["planning_basis"]
    assert basis["historical_verified_success_sessions"] == 2_886
    assert basis["historical_unique_supervised_tokens"] == 57_384_881
    assert basis["reference_successes_needed_for_20m"] == 1_006
    assert basis["minimum_reference_success_rate_over_planned_cells"] == 0.0503
    assert basis["not_a_success_or_token_yield_guarantee"] is True


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda value: value["source"]["thinking"].update({"preserve_thinking": False}),
            "explicitly generated and preserved",
        ),
        (
            lambda value: value["admission"]["reject"].remove(
                "teacher_or_provider_private_reasoning"
            ),
            "admission policy drift",
        ),
        (
            lambda value: value["selection"].update({"heldout_collection_cells": 1}),
            "family selection drift",
        ),
        (
            lambda value: value["execution"].update({"ambiguous_cell_replay_allowed": True}),
            "execution boundary drift",
        ),
    ],
)
def test_treatment_safety_and_leakage_tampering_fail_closed(mutate, match: str) -> None:
    spec = copy.deepcopy(_load(SPEC))
    mutate(spec)
    _reseal(spec)
    with pytest.raises(ValueError, match=match):
        campaign.render(spec, root=ROOT)


def test_independent_review_cannot_be_self_asserted() -> None:
    spec = copy.deepcopy(_load(SPEC))
    spec["review"]["review_completed"] = True
    _reseal(spec)
    with pytest.raises(ValueError, match="review gate drift"):
        campaign.render(spec, root=ROOT)
