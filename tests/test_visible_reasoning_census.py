"""Aggregate-only authorization tests for the distinct written-reasoning arm."""

from __future__ import annotations

import copy

import pytest

from training import visible_reasoning_census as census
from training.io import digest_json


def _sha(character: str) -> str:
    return "sha256:" + character * 64


def _seal(value: dict) -> dict:
    value.pop("sha256", None)
    value["sha256"] = digest_json(value)
    return value


def _manifest() -> dict:
    arm_manifests = {
        "reasoning_plus_action": {
            "path": "reasoning-plus-action.manifest.json",
            "file_sha256": _sha("9"),
            "logical_sha256": _sha("a"),
        },
        "matched_action_only": {
            "path": "matched-action-only.manifest.json",
            "file_sha256": _sha("0"),
            "logical_sha256": _sha("b"),
        },
    }
    paired = _sha("c")
    source_target = _sha("d")
    return _seal(
        {
            "schema": "cyber_qwen_opencode_visible_reasoning_sft_corpus_v1",
            "source_profile_sha256": _sha("a"),
            "source_authorization_sha256": _sha("b"),
            "collection_packet_sha256": _sha("c"),
            "selection_sha256": _sha("d"),
            "success_evidence_sha256": _sha("e"),
            "source_census_sha256": _sha("f"),
            "catalog_inventory_sha256": _sha("1"),
            "family_split_sha256": _sha("2"),
            "root_role_anchor_id": "fleet-blackbox-current-study-20260914-v2",
            "family_role_anchor_sha256": _sha("3"),
            "protected_family_lock_sha256": _sha("4"),
            "runtime_bindings_sha256": _sha("5"),
            "files": {
                "reasoning_plus_action": {
                    "path": "train-reasoning-plus-action.parquet",
                    "sha256": _sha("6"),
                    "rows": 20,
                },
                "matched_action_only": {
                    "path": "train-matched-action-only.parquet",
                    "sha256": _sha("7"),
                    "rows": 20,
                },
            },
            "arm_manifests": arm_manifests,
            "counts": {
                "source_records": 20,
                "windows": 20,
                "student_visible_reasoning_target_tokens": 8_000_000,
                "visible_action_target_tokens": 20_000_000,
                "supervised_tokens": 28_000_000,
                "matched_action_only_supervised_tokens": 20_000_000,
            },
            "campaign_identity": {
                "campaign_plan_sha256": _sha("1"),
                "wave_plan_sha256": _sha("2"),
                "cell_identity_universe_sha256": _sha("3"),
                "operation_authorization_sha256": _sha("4"),
                "task_selection_sha256": _sha("5"),
            },
            "matched_ablation": {
                "paired_window_identity_sha256": paired,
                "source_target_identity_sha256": source_target,
                "same_selected_windows": True,
                "same_input_ids": True,
                "only_reasoning_loss_mask_differs": True,
                "cross_arm_sha256": digest_json(
                    {
                        "reasoning_plus_action_manifest_sha256": arm_manifests[
                            "reasoning_plus_action"
                        ]["logical_sha256"],
                        "matched_action_only_manifest_sha256": arm_manifests["matched_action_only"][
                            "logical_sha256"
                        ],
                        "paired_window_identity_sha256": paired,
                        "source_target_identity_sha256": source_target,
                    }
                ),
            },
            "validation_mode": "pending_reasoning_selection",
            "coverage_sha256": _sha("7"),
            "limitations": ["synthetic"],
            "builder_sha256": {"fleet_visible_reasoning_corpus.py": _sha("8")},
        }
    )


def _source_census(manifest: dict) -> dict:
    return _seal(
        {
            "schema": census.SOURCE_CENSUS_SCHEMA,
            "source_profile_sha256": manifest["source_profile_sha256"],
            "source_authorization_sha256": manifest["source_authorization_sha256"],
            "collection_packet_sha256": manifest["collection_packet_sha256"],
            "private_selection_sha256": manifest["selection_sha256"],
            "success_evidence_sha256": manifest["success_evidence_sha256"],
            "sessions": {"candidates": 22, "verified_successes": 21, "selected": 20},
            "visibility": {"student_visible": 20, "private_or_unknown": 1, "absent": 1},
            "target_tokens": {
                "student_visible_reasoning": 8_000_000,
                "visible_action": 20_000_000,
            },
            "compaction": {"none": 21, "exact_student_generated": 1, "opaque_rejected": 0},
        }
    )


def _coverage(manifest: dict, source: dict) -> dict:
    return _seal(
        {
            "schema": "cyber_qwen_opencode_visible_reasoning_coverage_v1",
            "selection_sha256": manifest["selection_sha256"],
            "success_evidence_sha256": manifest["success_evidence_sha256"],
            "source_census_sha256": source["sha256"],
            "collection_packet_sha256": manifest["collection_packet_sha256"],
            "source_profile_sha256": manifest["source_profile_sha256"],
            "source_authorization_sha256": manifest["source_authorization_sha256"],
            "selected_source_records": 20,
            "visible_reasoning_windows": 20,
            "student_visible_reasoning_target_tokens": 8_000_000,
            "visible_action_target_tokens": 20_000_000,
            "unique_supervised_tokens": 28_000_000,
            "arm_unique_supervised_tokens": {
                "reasoning_plus_action": 28_000_000,
                "matched_action_only": 20_000_000,
            },
            "minimum_unique_supervised_tokens": 20_000_000,
            "arm_token_goal_reached": {
                "reasoning_plus_action": True,
                "matched_action_only": True,
            },
            "token_goal_reached": True,
            "minimum_successful_families": 20,
            "successful_family_goal_reached": True,
            "target_goal_reached": True,
            "family_token_concentration": {
                "families_with_targets": 20,
                "largest_family_target_token_fraction": 0.05,
                "maximum_allowed_fraction": 0.25,
                "within_limit": True,
            },
            "compaction": {
                "uncompacted_windows": 20,
                "exact_student_generated_continuation_windows": 0,
                "opaque_compaction_windows": 0,
            },
            "heldout_families_materialized": 0,
            "raw_text_written": False,
        }
    )


def test_separate_visible_reasoning_selection_binds_all_aggregates() -> None:
    manifest = _manifest()
    source = _source_census(manifest)
    manifest["source_census_sha256"] = source["sha256"]
    _seal(manifest)
    coverage = _coverage(manifest, source)
    manifest["coverage_sha256"] = coverage["sha256"]
    _seal(manifest)
    coverage = _coverage(manifest, source)

    selection = census.select(source, manifest, coverage)

    assert selection["objective"] == census.OBJECTIVE
    assert selection["selected"]["supervised_tokens"] == 28_000_000
    assert census.validate_selection(
        selection, census=source, corpus_manifest=manifest, coverage=coverage
    )


@pytest.mark.parametrize(
    "mutate,match",
    [
        (
            lambda source, manifest, coverage: source["visibility"].update(
                {"private_or_unknown": 2}
            ),
            "visibility totals",
        ),
        (
            lambda source, manifest, coverage: source["sessions"].update({"verified_successes": 5}),
            "session totals",
        ),
        (
            lambda source, manifest, coverage: source["target_tokens"].update(
                {"student_visible_reasoning": 21}
            ),
            "does not bind the exact manifest",
        ),
        (
            lambda source, manifest, coverage: coverage.update(
                {"source_authorization_sha256": _sha("9")}
            ),
            "does not bind the exact manifest",
        ),
        (
            lambda source, manifest, coverage: manifest.update({"validation_mode": "sft_ready"}),
            "bypassed final selection",
        ),
        (
            lambda source, manifest, coverage: coverage.update(
                {"heldout_families_materialized": 1}
            ),
            "private family boundary",
        ),
        (
            lambda source, manifest, coverage: manifest.update(
                {"root_role_anchor_id": "untrusted-root"}
            ),
            "trusted family split",
        ),
        (
            lambda source, manifest, coverage: coverage["family_token_concentration"].update(
                {"within_limit": False}
            ),
            "family concentration limit",
        ),
        (
            lambda source, manifest, coverage: coverage["compaction"].update(
                {"opaque_compaction_windows": 1, "uncompacted_windows": 2}
            ),
            "opaque or unbound compaction",
        ),
        (
            lambda source, manifest, coverage: coverage.update({"target_goal_reached": False}),
            "family gate",
        ),
        (
            lambda source, manifest, coverage: coverage.update(
                {
                    "minimum_successful_families": 19,
                    "successful_family_goal_reached": True,
                }
            ),
            "weakens the successful-family gate",
        ),
        (
            lambda source, manifest, coverage: manifest["matched_ablation"].update(
                {"cross_arm_sha256": _sha("9")}
            ),
            "matched visible-reasoning arm binding",
        ),
    ],
)
def test_visible_reasoning_selection_fails_closed_on_aggregate_drift(mutate, match: str) -> None:
    manifest = _manifest()
    source = _source_census(manifest)
    manifest["source_census_sha256"] = source["sha256"]
    _seal(manifest)
    coverage = _coverage(manifest, source)
    manifest["coverage_sha256"] = coverage["sha256"]
    _seal(manifest)
    coverage = _coverage(manifest, source)
    source, manifest, coverage = (
        copy.deepcopy(source),
        copy.deepcopy(manifest),
        copy.deepcopy(coverage),
    )
    mutate(source, manifest, coverage)
    _seal(source)
    _seal(manifest)
    _seal(coverage)
    with pytest.raises(ValueError, match=match):
        census.select(source, manifest, coverage)


def test_combined_20m_cannot_hide_an_underfilled_action_only_arm() -> None:
    manifest = _manifest()
    manifest["counts"].update(
        {
            "visible_action_target_tokens": 12_000_000,
            "supervised_tokens": 20_000_000,
            "matched_action_only_supervised_tokens": 12_000_000,
        }
    )
    source = _source_census(manifest)
    source["target_tokens"]["visible_action"] = 12_000_000
    _seal(source)
    manifest["source_census_sha256"] = source["sha256"]
    _seal(manifest)
    coverage = _coverage(manifest, source)
    coverage.update(
        {
            "visible_action_target_tokens": 12_000_000,
            "unique_supervised_tokens": 20_000_000,
            "arm_unique_supervised_tokens": {
                "reasoning_plus_action": 20_000_000,
                "matched_action_only": 12_000_000,
            },
            # This is the old, unsafe aggregate-only conclusion.
            "arm_token_goal_reached": {
                "reasoning_plus_action": True,
                "matched_action_only": True,
            },
            "token_goal_reached": True,
            "target_goal_reached": True,
        }
    )
    _seal(coverage)
    manifest["coverage_sha256"] = coverage["sha256"]
    _seal(manifest)
    coverage = copy.deepcopy(coverage)
    coverage.update(
        {
            "selection_sha256": manifest["selection_sha256"],
            "source_census_sha256": source["sha256"],
        }
    )
    _seal(coverage)

    with pytest.raises(ValueError, match="per-arm token gates"):
        census.select(source, manifest, coverage)
