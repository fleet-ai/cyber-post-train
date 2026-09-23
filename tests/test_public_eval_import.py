from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from cyber_post_train import public_eval_import as importer


def _signed(value: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(value)
    result["receipt_sha256"] = importer._sha256(importer._canonical(result))  # noqa: SLF001
    return result


def _denominators(benchmark: str) -> list[int]:
    return [8] * 5 + [7] * 10 if benchmark == "webexploitbench_level0" else [1] * 17


def _aggregate(benchmark: str, *, invalid_task: int | None = None) -> dict[str, Any]:
    spec = importer.BENCHMARKS[benchmark]
    rows = []
    for task_index, denominator in enumerate(_denominators(benchmark)):
        incomplete = task_index == invalid_task
        if benchmark == "webexploitbench_level0":
            base_numerator = task_index % 3
            candidate_numerator = min(denominator, base_numerator + task_index % 2)
        else:
            base_numerator = int(task_index % 4 == 0)
            candidate_numerator = int(task_index % 2 == 0)
        rows.append(
            {
                "metric_denominator": denominator,
                "base": {
                    "valid_attempts": 7 if incomplete else 8,
                    "infrastructure_invalid_attempts": 1 if incomplete else 0,
                    "metric_numerator": None if incomplete else base_numerator,
                },
                "candidate": {
                    "valid_attempts": 8,
                    "infrastructure_invalid_attempts": 0,
                    "metric_numerator": candidate_numerator,
                },
            }
        )
    return _signed(
        {
            "schema_version": importer.INPUT_SCHEMA,
            "status": "final",
            "benchmark": benchmark,
            "comparison_definition_sha256": spec["comparison_definition_sha256"],
            "anonymization_method": "private_random_permutation_v1",
            "anonymization_receipt_sha256": "sha256:" + "a" * 64,
            "task_rows": rows,
            "evidence": {
                "terminal_index_sha256": "sha256:" + "2" * 64,
                "scored_outcome_index_sha256": "sha256:" + "3" * 64,
                "terminal_schemas": copy.deepcopy(spec["terminal_schemas"]),
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


def _build(web: tuple[Path, str], fleet: tuple[Path, str]) -> dict[str, Any]:
    return importer.build_public_results(
        web[0], fleet[0], web_file_sha256=web[1], fleet_file_sha256=fleet[1]
    )


def test_builds_recomputed_public_studies_without_private_material(tmp_path: Path) -> None:
    web = _write(tmp_path / "web.json", _aggregate("webexploitbench_level0", invalid_task=4))
    fleet = _write(tmp_path / "fleet.json", _aggregate("fleet_development_dev17"))

    value = _build(web, fleet)

    assert [study["benchmark"] for study in value["studies"]] == [
        "webexploitbench_level0",
        "fleet_development_dev17",
    ]
    web_study = value["studies"][0]
    assert web_study["summary"]["paired_valid_tasks"] == 14
    assert web_study["summary"]["paired_technical_failure_tasks"] == 1
    assert web_study["task_rows"][4]["base"] == {
        "valid_attempts": 7,
        "technical_failures": 1,
        "metric_percent": None,
    }
    assert "metric_numerator" not in json.dumps(value)
    serialized = json.dumps(value).lower()
    for forbidden in ("task_key", "task_version_id", "session_id", "prompt", "trace", "flag"):
        assert forbidden not in serialized


def test_recomputes_headline_delta_and_bootstrap_from_task_counts(tmp_path: Path) -> None:
    web = _write(tmp_path / "web.json", _aggregate("webexploitbench_level0"))
    fleet = _write(tmp_path / "fleet.json", _aggregate("fleet_development_dev17"))

    first = _build(web, fleet)["studies"][0]["summary"]
    changed = _aggregate("webexploitbench_level0")
    changed["task_rows"][1]["candidate"]["metric_numerator"] = 0
    changed = _write(tmp_path / "web-changed.json", _resign(changed))
    second = _build(changed, fleet)["studies"][0]["summary"]

    assert second["candidate"]["metric_percent"] < first["candidate"]["metric_percent"]
    assert second["delta_percentage_points"] < first["delta_percentage_points"]
    assert second["confidence_interval"] != first["confidence_interval"]


def test_writer_is_atomic_and_idempotent(tmp_path: Path) -> None:
    value = {"schema_version": importer.OUTPUT_SCHEMA, "studies": []}
    output = tmp_path / "results.json"

    assert importer.write_public_results(value, output) is True
    first_stat = output.stat()
    assert importer.write_public_results(value, output) is False
    assert output.stat().st_mtime_ns == first_stat.st_mtime_ns


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda value: value.update(status="incomplete"), "not a final"),
        (
            lambda value: value.update(comparison_definition_sha256="sha256:" + "9" * 64),
            "comparison definition",
        ),
        (
            lambda value: value["task_rows"][0]["base"].update(metric_numerator=99),
            "numerator is invalid",
        ),
        (
            lambda value: value["task_rows"][0]["base"].update(
                valid_attempts=7, infrastructure_invalid_attempts=0
            ),
            "all eight attempts",
        ),
        (
            lambda value: value["task_rows"][0].update(metric_denominator=9),
            "frozen benchmark",
        ),
        (
            lambda value: value.update(anonymization_method="original_task_order"),
            "anonymous task ordering",
        ),
        (
            lambda value: value["evidence"]["terminal_schemas"].update(arm_terminal="wrong"),
            "terminal schemas",
        ),
        (lambda value: value.update(prompt="private"), "fields are incomplete"),
    ],
)
def test_rejects_incomplete_forged_or_private_input(tmp_path: Path, mutation, match: str) -> None:
    value = _aggregate("webexploitbench_level0")
    mutation(value)
    web = _write(tmp_path / "bad.json", _resign(value))
    fleet = _write(tmp_path / "fleet.json", _aggregate("fleet_development_dev17"))

    with pytest.raises(importer.PublicEvalImportError, match=match):
        _build(web, fleet)


def test_rejects_file_that_differs_from_reviewed_digest(tmp_path: Path) -> None:
    web = _write(tmp_path / "web.json", _aggregate("webexploitbench_level0"))
    fleet = _write(tmp_path / "fleet.json", _aggregate("fleet_development_dev17"))

    with pytest.raises(importer.PublicEvalImportError, match="reviewed digest"):
        importer.build_public_results(
            web[0],
            fleet[0],
            web_file_sha256="sha256:" + "0" * 64,
            fleet_file_sha256=fleet[1],
        )


def test_rejects_changed_payload_with_stale_self_digest(tmp_path: Path) -> None:
    value = _aggregate("webexploitbench_level0")
    value["comparison_definition_sha256"] = "sha256:" + "9" * 64
    web = _write(tmp_path / "web.json", value)
    fleet = _write(tmp_path / "fleet.json", _aggregate("fleet_development_dev17"))

    with pytest.raises(importer.PublicEvalImportError, match="self-digest"):
        _build(web, fleet)


def test_infrastructure_invalid_attempt_is_disclosed_not_scored_as_zero(tmp_path: Path) -> None:
    source = _aggregate("webexploitbench_level0", invalid_task=3)
    assert source["task_rows"][3]["base"]["metric_numerator"] is None
    web = _write(tmp_path / "web.json", source)
    fleet = _write(tmp_path / "fleet.json", _aggregate("fleet_development_dev17"))

    public = _build(web, fleet)

    assert public["studies"][0]["summary"]["base"]["technical_failures"] == 1
    assert public["studies"][0]["summary"]["paired_technical_failure_tasks"] == 1


def test_rejects_symlinked_input(tmp_path: Path) -> None:
    target = _write(tmp_path / "target.json", _aggregate("webexploitbench_level0"))
    alias = tmp_path / "alias.json"
    alias.symlink_to(target[0])
    fleet = _write(tmp_path / "fleet.json", _aggregate("fleet_development_dev17"))

    with pytest.raises(importer.PublicEvalImportError, match="exact regular file"):
        importer.build_public_results(
            alias,
            fleet[0],
            web_file_sha256=target[1],
            fleet_file_sha256=fleet[1],
        )
