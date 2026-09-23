"""Validate one sanitized interim WebExploitBench comparison for the public site.

This importer is deliberately separate from ``public_eval_import``.  It accepts
only the interim schema, recomputes a descriptive uniform-k comparison, emits no
confidence interval, and never labels the result pass@8.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any

INPUT_SCHEMA = "cyber_sanitized_matched_pass8_interim_v1"
OUTPUT_SCHEMA = "cyber_public_eval_interim_results_v1"
BENCHMARK = "webexploitbench_level0"
COMPARISON_DEFINITION_SHA256 = (
    "sha256:ce3e0706e9c8f73bb552d83bd0841602dbb68c0f0d2909b1588432dff86fe95f"
)
HEADLINE_METHOD = "score_blind_lowest_matched_attempts_uniform_k_v1"
TASK_COUNT = 15
ATTEMPT_COUNT = 8
PAIR_COUNT = TASK_COUNT * ATTEMPT_COUNT
CELL_COUNT = PAIR_COUNT * 2
METRIC_DENOMINATOR = 110
ARMS = ("base", "candidate")
TERMINAL_SCHEMAS = {
    "experiment": "webexploitbench_collection_replica_set_v2",
    "arm_terminal": "webexploitbench_collection_replica_arm_terminal_v1",
    "score_acceptance": "webexploitbench_deferred_score_replica_cell_acceptance_v1",
}

_INPUT_FIELDS = {
    "schema_version",
    "status",
    "benchmark",
    "comparison_definition_sha256",
    "anonymization_method",
    "anonymization_receipt_sha256",
    "headline",
    "coverage",
    "task_rows",
    "evidence",
    "receipt_sha256",
}
_DENOMINATOR_FIELDS = {
    "official_metric_denominator",
    "planned_attempts_per_task",
    "planned_task_count",
    "planned_pair_denominator",
    "planned_cell_denominator",
    "qualified_pair_denominator",
    "qualified_cell_denominator",
}
_COUNT_FIELDS = {
    "accepted_terminal_cells",
    "completed_primary_score_cells",
    "missing_cells",
    "missing_pairs",
}
_COVERAGE_FIELDS = {
    "denominators",
    "counts",
    "qualified_pair_roster",
    "missing_cell_roster",
    "missing_pair_roster",
    "headline_selection",
}
_STATUS_FIELDS = {"terminal_status", "primary_score_status"}
_TASK_FIELDS = {
    "public_task_index",
    "planned_pair_denominator",
    "qualified_pair_denominator",
    "matched_attempt_indices",
    "metric_denominator",
    "base",
    "candidate",
    "uniform_headline",
}
_OBSERVED_ARM_FIELDS = {"valid_matched_attempts", "observed_union_numerator"}
_UNIFORM_ARM_FIELDS = {"metric_numerator"}
_HEADLINE_FIELDS = {
    "metric",
    "interpretation",
    "selection_method",
    "uniform_k",
    "metric_denominator",
    "base",
    "candidate",
}
_EVIDENCE_FIELDS = {
    "coverage_file_sha256",
    "coverage_receipt_sha256",
    "accepted_cell_index_sha256",
    "terminal_schemas",
}


class InterimEvalImportError(ValueError):
    """The proposed interim summary is inconsistent, unsafe, or not supported."""


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha256(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 71
        and value.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in value[7:])
    )


def _exact_fields(value: object, fields: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise InterimEvalImportError(f"{label} fields are incomplete or unknown")
    return value


def _integer(value: object, *, minimum: int, maximum: int, label: str) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise InterimEvalImportError(f"{label} is invalid")
    return value


def _load(path: Path, expected_sha256: str) -> tuple[dict[str, Any], str]:
    if path.is_symlink() or not path.is_file():
        raise InterimEvalImportError("interim input must be an exact regular file")
    raw = path.read_bytes()
    file_sha256 = _sha256(raw)
    if not _is_sha256(expected_sha256) or file_sha256 != expected_sha256:
        raise InterimEvalImportError("interim file digest differs from the reviewed digest")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InterimEvalImportError("interim input is not valid JSON") from exc
    if not isinstance(value, dict):
        raise InterimEvalImportError("interim input must be a JSON object")
    return value, file_sha256


def _coordinate(row: Mapping[str, Any], label: str) -> tuple[int, int]:
    return (
        _integer(row.get("public_task_index"), minimum=0, maximum=14, label=f"{label} task"),
        _integer(row.get("attempt_index"), minimum=0, maximum=7, label=f"{label} attempt"),
    )


def _status(raw: object, label: str) -> dict[str, str]:
    value = _exact_fields(raw, _STATUS_FIELDS, label)
    terminal = value["terminal_status"]
    score = value["primary_score_status"]
    if terminal == "missing":
        valid = score == "not_assessable_without_terminal"
    else:
        valid = terminal == "accepted" and score in {"missing", "complete"}
    if not valid:
        raise InterimEvalImportError(f"{label} status is invalid")
    return {"terminal_status": terminal, "primary_score_status": score}


def _validate_coverage(raw: object) -> tuple[dict[str, Any], dict[int, list[int]], int]:
    coverage = _exact_fields(raw, _COVERAGE_FIELDS, "coverage")
    denominators = _exact_fields(coverage["denominators"], _DENOMINATOR_FIELDS, "denominators")
    fixed = {
        "official_metric_denominator": METRIC_DENOMINATOR,
        "planned_attempts_per_task": ATTEMPT_COUNT,
        "planned_task_count": TASK_COUNT,
        "planned_pair_denominator": PAIR_COUNT,
        "planned_cell_denominator": CELL_COUNT,
    }
    if any(denominators[field] != value for field, value in fixed.items()):
        raise InterimEvalImportError("fixed interim denominators differ")
    qualified_count = _integer(
        denominators["qualified_pair_denominator"],
        minimum=0,
        maximum=PAIR_COUNT,
        label="qualified pair denominator",
    )
    if denominators["qualified_cell_denominator"] != 2 * qualified_count:
        raise InterimEvalImportError("qualified cell denominator differs")

    qualified_raw = coverage["qualified_pair_roster"]
    if not isinstance(qualified_raw, list) or len(qualified_raw) != qualified_count:
        raise InterimEvalImportError("qualified pair roster differs")
    qualified: set[tuple[int, int]] = set()
    for raw_row in qualified_raw:
        row = _exact_fields(raw_row, {"attempt_index", "public_task_index"}, "qualified pair")
        coordinate = _coordinate(row, "qualified pair")
        if coordinate in qualified:
            raise InterimEvalImportError("qualified pair roster has duplicates")
        qualified.add(coordinate)

    missing_pair_raw = coverage["missing_pair_roster"]
    if not isinstance(missing_pair_raw, list):
        raise InterimEvalImportError("missing pair roster is invalid")
    missing_pairs: set[tuple[int, int]] = set()
    derived_missing_cells: set[tuple[int, int, str]] = set()
    accepted_terminals = 2 * len(qualified)
    completed_scores = 2 * len(qualified)
    for raw_row in missing_pair_raw:
        row = _exact_fields(
            raw_row,
            {"attempt_index", "public_task_index", "missing_arms", "arm_coverage"},
            "missing pair",
        )
        coordinate = _coordinate(row, "missing pair")
        if coordinate in qualified or coordinate in missing_pairs:
            raise InterimEvalImportError("pair coverage overlaps or duplicates")
        missing_pairs.add(coordinate)
        arm_coverage = _exact_fields(row["arm_coverage"], set(ARMS), "pair arm coverage")
        statuses = {arm: _status(arm_coverage[arm], f"missing pair {arm}") for arm in ARMS}
        missing_arms = [arm for arm in ARMS if statuses[arm]["primary_score_status"] != "complete"]
        if row["missing_arms"] != missing_arms or not missing_arms:
            raise InterimEvalImportError("missing pair arms differ")
        for arm in ARMS:
            status = statuses[arm]
            if status["terminal_status"] == "accepted":
                accepted_terminals += 1
            if status["primary_score_status"] == "complete":
                completed_scores += 1
            else:
                derived_missing_cells.add((*coordinate, arm))
    if len(qualified | missing_pairs) != PAIR_COUNT:
        raise InterimEvalImportError("pair coverage is incomplete")

    missing_cell_raw = coverage["missing_cell_roster"]
    if not isinstance(missing_cell_raw, list):
        raise InterimEvalImportError("missing cell roster is invalid")
    missing_cells: set[tuple[int, int, str]] = set()
    for raw_row in missing_cell_raw:
        row = _exact_fields(
            raw_row,
            {
                "attempt_index",
                "public_task_index",
                "arm",
                "terminal_status",
                "primary_score_status",
            },
            "missing cell",
        )
        task_index, attempt_index = _coordinate(row, "missing cell")
        arm = row["arm"]
        if arm not in ARMS:
            raise InterimEvalImportError("missing cell arm is invalid")
        _status(
            {
                "terminal_status": row["terminal_status"],
                "primary_score_status": row["primary_score_status"],
            },
            "missing cell",
        )
        coordinate = (task_index, attempt_index, arm)
        if coordinate in missing_cells:
            raise InterimEvalImportError("missing cell roster has duplicates")
        missing_cells.add(coordinate)
    if missing_cells != derived_missing_cells:
        raise InterimEvalImportError("missing cell roster differs from pair coverage")

    counts = _exact_fields(coverage["counts"], _COUNT_FIELDS, "coverage counts")
    expected_counts = {
        "accepted_terminal_cells": accepted_terminals,
        "completed_primary_score_cells": completed_scores,
        "missing_cells": len(missing_cells),
        "missing_pairs": len(missing_pairs),
    }
    if counts != expected_counts or completed_scores + len(missing_cells) != CELL_COUNT:
        raise InterimEvalImportError("coverage counts differ from rosters")

    by_task = {
        task_index: sorted(attempt_index for task, attempt_index in qualified if task == task_index)
        for task_index in range(TASK_COUNT)
    }
    uniform_k = min(len(attempts) for attempts in by_task.values())
    selection = _exact_fields(
        coverage["headline_selection"], {"method", "uniform_k", "tasks"}, "headline selection"
    )
    if selection["method"] != HEADLINE_METHOD or selection["uniform_k"] != uniform_k:
        raise InterimEvalImportError("uniform headline selection differs")
    selection_rows = selection["tasks"]
    if not isinstance(selection_rows, list) or len(selection_rows) != TASK_COUNT:
        raise InterimEvalImportError("headline task selection is incomplete")
    seen_tasks: set[int] = set()
    for raw_row in selection_rows:
        row = _exact_fields(raw_row, {"public_task_index", "attempt_indices"}, "headline task")
        task_index = _integer(
            row["public_task_index"], minimum=0, maximum=14, label="headline public task"
        )
        attempts = row["attempt_indices"]
        if task_index in seen_tasks or attempts != by_task[task_index][:uniform_k]:
            raise InterimEvalImportError("headline task attempts differ")
        seen_tasks.add(task_index)
    return dict(coverage), by_task, uniform_k


def _percent(numerator: int, denominator: int) -> float:
    return numerator * 100.0 / denominator


def _validate(value: dict[str, Any], file_sha256: str) -> dict[str, Any]:
    aggregate = _exact_fields(value, _INPUT_FIELDS, "aggregate")
    if aggregate["schema_version"] != INPUT_SCHEMA or aggregate["status"] != "interim":
        raise InterimEvalImportError("aggregate is not a supported interim summary")
    claimed = aggregate["receipt_sha256"]
    unsigned = dict(aggregate)
    unsigned.pop("receipt_sha256")
    if not _is_sha256(claimed) or claimed != _sha256(_canonical(unsigned)):
        raise InterimEvalImportError("aggregate self-digest is invalid")
    if (
        aggregate["benchmark"] != BENCHMARK
        or aggregate["comparison_definition_sha256"] != COMPARISON_DEFINITION_SHA256
    ):
        raise InterimEvalImportError("interim benchmark definition differs")
    if aggregate["anonymization_method"] != "private_random_permutation_v1" or not _is_sha256(
        aggregate["anonymization_receipt_sha256"]
    ):
        raise InterimEvalImportError("anonymous task ordering contract is invalid")

    coverage, qualified_by_task, uniform_k = _validate_coverage(aggregate["coverage"])
    rows = aggregate["task_rows"]
    if not isinstance(rows, list) or len(rows) != TASK_COUNT:
        raise InterimEvalImportError("interim task rows are incomplete")
    public_rows: list[dict[str, Any]] = []
    headline_totals = {arm: 0 for arm in ARMS}
    denominator_total = 0
    for expected_index, raw_row in enumerate(rows):
        row = _exact_fields(raw_row, _TASK_FIELDS, "task row")
        if row["public_task_index"] != expected_index:
            raise InterimEvalImportError("public task order differs")
        attempts = qualified_by_task[expected_index]
        if (
            row["planned_pair_denominator"] != ATTEMPT_COUNT
            or row["qualified_pair_denominator"] != len(attempts)
            or row["matched_attempt_indices"] != attempts
        ):
            raise InterimEvalImportError("task pair coverage differs")
        denominator = row["metric_denominator"]
        public_arms: dict[str, Any] = {}
        if not attempts:
            if (
                denominator is not None
                or row["uniform_headline"] is not None
                or any(row[arm] is not None for arm in ARMS)
            ):
                raise InterimEvalImportError("task without matched pairs contains outcomes")
            public_arms = {arm: None for arm in ARMS}
        else:
            denominator = _integer(
                denominator, minimum=1, maximum=METRIC_DENOMINATOR, label="task denominator"
            )
            denominator_total += denominator
            for arm in ARMS:
                arm_row = _exact_fields(row[arm], _OBSERVED_ARM_FIELDS, f"task {arm}")
                if arm_row["valid_matched_attempts"] != len(attempts):
                    raise InterimEvalImportError("task matched attempts differ")
                numerator = _integer(
                    arm_row["observed_union_numerator"],
                    minimum=0,
                    maximum=denominator,
                    label="task observed numerator",
                )
                public_arms[arm] = {
                    "valid_matched_attempts": len(attempts),
                    "observed_union_percent": _percent(numerator, denominator),
                }
        uniform_public = None
        if uniform_k:
            uniform = _exact_fields(row["uniform_headline"], set(ARMS), "uniform task headline")
            uniform_public = {}
            for arm in ARMS:
                arm_row = _exact_fields(uniform[arm], _UNIFORM_ARM_FIELDS, f"uniform task {arm}")
                numerator = _integer(
                    arm_row["metric_numerator"],
                    minimum=0,
                    maximum=denominator,
                    label="uniform task numerator",
                )
                headline_totals[arm] += numerator
                uniform_public[arm] = {"metric_percent": _percent(numerator, denominator)}
        elif row["uniform_headline"] is not None:
            raise InterimEvalImportError("zero-k task contains a uniform headline")
        public_rows.append(
            {
                "public_task_index": expected_index,
                "planned_pair_denominator": ATTEMPT_COUNT,
                "qualified_pair_denominator": len(attempts),
                "matched_attempt_indices": attempts,
                "base": public_arms["base"],
                "candidate": public_arms["candidate"],
                "uniform_headline": uniform_public,
            }
        )

    headline = aggregate["headline"]
    summary = None
    if uniform_k == 0:
        if headline is not None:
            raise InterimEvalImportError("zero-k aggregate must not contain a headline")
    else:
        if denominator_total != METRIC_DENOMINATOR:
            raise InterimEvalImportError("uniform headline denominator differs")
        source = _exact_fields(headline, _HEADLINE_FIELDS, "headline")
        if (
            source["metric"] != "matched_complete_case_union_at_k"
            or source["interpretation"] != "descriptive_not_pass_at_8"
            or source["selection_method"] != HEADLINE_METHOD
            or source["uniform_k"] != uniform_k
            or source["metric_denominator"] != METRIC_DENOMINATOR
        ):
            raise InterimEvalImportError("headline definition differs")
        for arm in ARMS:
            arm_row = _exact_fields(source[arm], _UNIFORM_ARM_FIELDS, f"headline {arm}")
            if arm_row["metric_numerator"] != headline_totals[arm]:
                raise InterimEvalImportError("headline numerator differs from task rows")
        base_percent = _percent(headline_totals["base"], METRIC_DENOMINATOR)
        candidate_percent = _percent(headline_totals["candidate"], METRIC_DENOMINATOR)
        summary = {
            "metric_population": "all_tasks_score_blind_uniform_matched_complete_cases",
            "interpretation": "descriptive_not_pass_at_8",
            "uniform_k": uniform_k,
            "metric_denominator": METRIC_DENOMINATOR,
            "base": {"metric_percent": base_percent},
            "candidate": {"metric_percent": candidate_percent},
            "delta_percentage_points": candidate_percent - base_percent,
            "confidence_interval": None,
        }

    evidence = _exact_fields(aggregate["evidence"], _EVIDENCE_FIELDS, "evidence")
    if (
        any(not _is_sha256(evidence[field]) for field in _EVIDENCE_FIELDS - {"terminal_schemas"})
        or evidence["terminal_schemas"] != TERMINAL_SCHEMAS
    ):
        raise InterimEvalImportError("interim evidence contract differs")
    return {
        "benchmark": BENCHMARK,
        "title": "Public web-security benchmark",
        "metric": "Known weaknesses found across the same completed attempts for both models",
        "task_count": TASK_COUNT,
        "uniform_k": uniform_k,
        "task_rows": public_rows,
        "coverage": coverage,
        "summary": summary,
        "evidence": {
            "source_file_sha256": file_sha256,
            "source_receipt_sha256": claimed,
            "anonymization_receipt_sha256": aggregate["anonymization_receipt_sha256"],
            "coverage_receipt_sha256": evidence["coverage_receipt_sha256"],
            "accepted_cell_index_sha256": evidence["accepted_cell_index_sha256"],
        },
    }


def build_public_interim_results(path: Path, *, file_sha256: str) -> dict[str, Any]:
    """Validate one reviewed interim aggregate and return public-only site data."""

    value, actual_sha256 = _load(path, file_sha256)
    return {
        "schema_version": OUTPUT_SCHEMA,
        "status": "interim",
        "studies": [_validate(value, actual_sha256)],
    }


def write_public_interim_results(value: Mapping[str, Any], output: Path) -> bool:
    """Atomically write public interim data; return False for an identical rerun."""

    if output.is_symlink() or output.parent.is_symlink() or not output.parent.is_dir():
        raise InterimEvalImportError("public interim output path is unsafe")
    raw = json.dumps(value, indent=2, sort_keys=True).encode() + b"\n"
    if output.exists() and output.read_bytes() == raw:
        return False
    descriptor, temporary = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    try:
        os.fchmod(descriptor, 0o644)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    except BaseException:
        with suppress(OSError):
            os.close(descriptor)
        Path(temporary).unlink(missing_ok=True)
        raise
    return True


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--web-interim", type=Path, required=True)
    parser.add_argument("--web-interim-file-sha256", required=True)
    parser.add_argument("--output", type=Path, default=Path("site/evaluation-interim-results.json"))
    args = parser.parse_args(argv)
    try:
        value = build_public_interim_results(
            args.web_interim, file_sha256=args.web_interim_file_sha256
        )
        changed = write_public_interim_results(value, args.output)
    except (OSError, InterimEvalImportError) as exc:
        parser.error(str(exc))
    print(json.dumps({"changed": changed, "study_count": len(value["studies"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
