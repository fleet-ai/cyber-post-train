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
    return _seal(
        {
            "schema": "cyber_qwen_opencode_visible_reasoning_sft_corpus_v1",
            "source_profile_sha256": _sha("a"),
            "collection_packet_sha256": _sha("b"),
            "selection_sha256": _sha("c"),
            "success_evidence_sha256": _sha("d"),
            "source_census_sha256": _sha("e"),
            "catalog_inventory_sha256": _sha("f"),
            "family_split_sha256": _sha("1"),
            "root_role_anchor_id": "fleet-blackbox-current-study-20260914-v2",
            "family_role_anchor_sha256": _sha("2"),
            "protected_family_lock_sha256": _sha("3"),
            "runtime_bindings_sha256": _sha("4"),
            "files": {"train": {"path": "train.parquet", "sha256": _sha("5"), "rows": 3}},
            "counts": {
                "source_records": 2,
                "windows": 3,
                "student_visible_reasoning_target_tokens": 20,
                "visible_action_target_tokens": 30,
                "supervised_tokens": 50,
            },
            "validation_mode": "pending_reasoning_selection",
            "coverage_sha256": _sha("6"),
            "limitations": ["synthetic"],
            "builder_sha256": {"fleet_visible_reasoning_corpus.py": _sha("7")},
        }
    )


def _source_census(manifest: dict) -> dict:
    return _seal(
        {
            "schema": census.SOURCE_CENSUS_SCHEMA,
            "source_profile_sha256": manifest["source_profile_sha256"],
            "collection_packet_sha256": manifest["collection_packet_sha256"],
            "private_selection_sha256": manifest["selection_sha256"],
            "success_evidence_sha256": manifest["success_evidence_sha256"],
            "sessions": {"candidates": 4, "verified_successes": 3, "selected": 2},
            "visibility": {"student_visible": 2, "private_or_unknown": 1, "absent": 1},
            "target_tokens": {"student_visible_reasoning": 20, "visible_action": 30},
            "compaction": {"none": 3, "exact_student_generated": 1, "opaque_rejected": 0},
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
            "selected_source_records": 2,
            "visible_reasoning_windows": 3,
            "student_visible_reasoning_target_tokens": 20,
            "visible_action_target_tokens": 30,
            "unique_supervised_tokens": 50,
            "minimum_unique_supervised_tokens": 20,
            "target_goal_reached": True,
            "family_token_concentration": {"within_limit": True},
            "compaction": {},
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
    assert selection["selected"]["supervised_tokens"] == 50
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
            lambda source, manifest, coverage: source["target_tokens"].update(
                {"student_visible_reasoning": 21}
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
