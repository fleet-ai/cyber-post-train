from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from cyber_post_train import interim_eval_import as importer
from cyber_post_train import public_eval_import as final_importer


def _signed(value: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(value)
    result["receipt_sha256"] = importer._sha256(importer._canonical(result))  # noqa: SLF001
    return result


def _aggregate(*, zero_k: bool = False) -> dict[str, Any]:
    qualified = {(0, 0)} if zero_k else ({(task_index, 0) for task_index in range(15)} | {(0, 1)})
    qualified_rows = [
        {"public_task_index": task_index, "attempt_index": attempt_index}
        for task_index, attempt_index in sorted(qualified)
    ]
    missing_pairs = []
    missing_cells = []
    for task_index in range(15):
        for attempt_index in range(8):
            if (task_index, attempt_index) in qualified:
                continue
            arm_coverage = {
                arm: {
                    "terminal_status": "missing",
                    "primary_score_status": "not_assessable_without_terminal",
                }
                for arm in importer.ARMS
            }
            missing_pairs.append(
                {
                    "public_task_index": task_index,
                    "attempt_index": attempt_index,
                    "missing_arms": list(importer.ARMS),
                    "arm_coverage": arm_coverage,
                }
            )
            for arm in importer.ARMS:
                missing_cells.append(
                    {
                        "public_task_index": task_index,
                        "attempt_index": attempt_index,
                        "arm": arm,
                        **arm_coverage[arm],
                    }
                )
    by_task = {
        task_index: sorted(attempt for task, attempt in qualified if task == task_index)
        for task_index in range(15)
    }
    uniform_k = min(len(attempts) for attempts in by_task.values())
    denominators = [8] * 5 + [7] * 10
    rows = []
    for task_index, denominator in enumerate(denominators):
        attempts = by_task[task_index]
        if not attempts:
            rows.append(
                {
                    "public_task_index": task_index,
                    "planned_pair_denominator": 8,
                    "qualified_pair_denominator": 0,
                    "matched_attempt_indices": [],
                    "metric_denominator": None,
                    "base": None,
                    "candidate": None,
                    "uniform_headline": None,
                }
            )
            continue
        rows.append(
            {
                "public_task_index": task_index,
                "planned_pair_denominator": 8,
                "qualified_pair_denominator": len(attempts),
                "matched_attempt_indices": attempts,
                "metric_denominator": denominator,
                "base": {
                    "valid_matched_attempts": len(attempts),
                    "observed_union_numerator": 4 if len(attempts) == 2 else 1,
                },
                "candidate": {
                    "valid_matched_attempts": len(attempts),
                    "observed_union_numerator": 5 if len(attempts) == 2 else 2,
                },
                "uniform_headline": (
                    {
                        "base": {"metric_numerator": 1},
                        "candidate": {"metric_numerator": 2},
                    }
                    if uniform_k
                    else None
                ),
            }
        )
    headline = None
    if uniform_k:
        headline = {
            "metric": "matched_complete_case_union_at_k",
            "interpretation": "descriptive_not_pass_at_8",
            "selection_method": importer.HEADLINE_METHOD,
            "uniform_k": 1,
            "metric_denominator": 110,
            "base": {"metric_numerator": 15},
            "candidate": {"metric_numerator": 30},
        }
    return _signed(
        {
            "schema_version": importer.INPUT_SCHEMA,
            "status": "interim",
            "benchmark": importer.BENCHMARK,
            "comparison_definition_sha256": importer.COMPARISON_DEFINITION_SHA256,
            "anonymization_method": "private_random_permutation_v1",
            "anonymization_receipt_sha256": "sha256:" + "a" * 64,
            "headline": headline,
            "coverage": {
                "denominators": {
                    "official_metric_denominator": 110,
                    "planned_attempts_per_task": 8,
                    "planned_task_count": 15,
                    "planned_pair_denominator": 120,
                    "planned_cell_denominator": 240,
                    "qualified_pair_denominator": len(qualified),
                    "qualified_cell_denominator": 2 * len(qualified),
                },
                "counts": {
                    "accepted_terminal_cells": 2 * len(qualified),
                    "completed_primary_score_cells": 2 * len(qualified),
                    "missing_cells": len(missing_cells),
                    "missing_pairs": len(missing_pairs),
                },
                "qualified_pair_roster": qualified_rows,
                "missing_cell_roster": missing_cells,
                "missing_pair_roster": missing_pairs,
                "headline_selection": {
                    "method": importer.HEADLINE_METHOD,
                    "uniform_k": uniform_k,
                    "tasks": [
                        {
                            "public_task_index": task_index,
                            "attempt_indices": attempts[:uniform_k],
                        }
                        for task_index, attempts in by_task.items()
                    ],
                },
            },
            "task_rows": rows,
            "evidence": {
                "coverage_file_sha256": "sha256:" + "b" * 64,
                "coverage_receipt_sha256": "sha256:" + "c" * 64,
                "accepted_cell_index_sha256": "sha256:" + "d" * 64,
                "terminal_schemas": copy.deepcopy(importer.TERMINAL_SCHEMAS),
            },
        }
    )


def _write(path: Path, value: dict[str, Any]) -> tuple[Path, str]:
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return path, importer._sha256(path.read_bytes())  # noqa: SLF001


def _resign(value: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(value)
    value.pop("receipt_sha256", None)
    return _signed(value)


def test_builds_public_uniform_k_descriptive_result_without_private_material(
    tmp_path: Path,
) -> None:
    source, digest = _write(tmp_path / "interim.json", _aggregate())

    value = importer.build_public_interim_results(source, file_sha256=digest)

    assert value["schema_version"] == importer.OUTPUT_SCHEMA
    assert value["status"] == "interim"
    study = value["studies"][0]
    assert study["uniform_k"] == 1
    assert study["summary"] == {
        "metric_population": "all_tasks_score_blind_uniform_matched_complete_cases",
        "interpretation": "descriptive_not_pass_at_8",
        "uniform_k": 1,
        "metric_denominator": 110,
        "base": {"metric_percent": pytest.approx(15 * 100 / 110)},
        "candidate": {"metric_percent": pytest.approx(30 * 100 / 110)},
        "delta_percentage_points": pytest.approx(15 * 100 / 110),
        "confidence_interval": None,
    }
    assert study["coverage"]["denominators"]["qualified_pair_denominator"] == 16
    assert len(study["coverage"]["missing_pair_roster"]) == 104
    assert study["task_rows"][0]["qualified_pair_denominator"] == 2
    serialized = json.dumps(value).lower()
    assert "metric_numerator" not in serialized
    assert "public_to_private_task_index" not in serialized
    for forbidden in (
        "private_task_index",
        "task_key",
        "target",
        "objective",
        "prompt",
        "trace",
        "answer",
        "flag",
        "/private/",
    ):
        assert forbidden not in serialized


def test_zero_uniform_k_keeps_headline_null(tmp_path: Path) -> None:
    source, digest = _write(tmp_path / "interim.json", _aggregate(zero_k=True))

    study = importer.build_public_interim_results(source, file_sha256=digest)["studies"][0]

    assert study["uniform_k"] == 0
    assert study["summary"] is None
    assert study["task_rows"][0]["qualified_pair_denominator"] == 1
    assert study["task_rows"][1]["base"] is None


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda value: value.update(status="final"), "not a supported interim"),
        (
            lambda value: value["coverage"]["headline_selection"]["tasks"][0].update(
                attempt_indices=[1]
            ),
            "headline task attempts differ",
        ),
        (
            lambda value: value["headline"]["base"].update(metric_numerator=14),
            "headline numerator differs",
        ),
        (
            lambda value: value["task_rows"][0].update(private_task_index=0),
            "fields are incomplete or unknown",
        ),
        (
            lambda value: value["coverage"]["missing_cell_roster"][0].update(
                primary_score_status="complete"
            ),
            "missing cell status is invalid",
        ),
    ],
)
def test_rejects_forged_private_or_inconsistent_input(tmp_path: Path, mutation, match: str) -> None:
    value = _aggregate()
    mutation(value)
    source, digest = _write(tmp_path / "bad.json", _resign(value))

    with pytest.raises(importer.InterimEvalImportError, match=match):
        importer.build_public_interim_results(source, file_sha256=digest)


def test_final_importer_remains_strictly_final(tmp_path: Path) -> None:
    value = _aggregate()

    with pytest.raises(final_importer.PublicEvalImportError, match="fields are incomplete"):
        final_importer._validate(value, "sha256:" + "0" * 64)  # noqa: SLF001


def test_writer_is_atomic_and_idempotent(tmp_path: Path) -> None:
    value = {"schema_version": importer.OUTPUT_SCHEMA, "status": "interim", "studies": []}
    output = tmp_path / "interim-results.json"

    assert importer.write_public_interim_results(value, output) is True
    first_stat = output.stat()
    assert importer.write_public_interim_results(value, output) is False
    assert output.stat().st_mtime_ns == first_stat.st_mtime_ns


def test_rejects_changed_file_and_symlinked_input(tmp_path: Path) -> None:
    source, digest = _write(tmp_path / "interim.json", _aggregate())
    alias = tmp_path / "alias.json"
    alias.symlink_to(source)

    with pytest.raises(importer.InterimEvalImportError, match="reviewed digest"):
        importer.build_public_interim_results(source, file_sha256="sha256:" + "0" * 64)
    with pytest.raises(importer.InterimEvalImportError, match="exact regular file"):
        importer.build_public_interim_results(alias, file_sha256=digest)
