"""Import final sanitized matched-evaluation summaries into the public site.

The importer accepts aggregate-only, self-digesting JSON. It never reads task
prompts, traces, per-attempt scores, or benchmark artifacts. All public metrics
and confidence intervals are recomputed from sanitized task-level counts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any

INPUT_SCHEMA = "cyber_sanitized_matched_pass8_aggregate_v1"
OUTPUT_SCHEMA = "cyber_public_eval_results_v1"
PASS_K = 8
ARMS = ("base", "candidate")
CI_METHOD = "paired_task_sha256_bootstrap_percentile_v1"
CI_UNIT = "percentage_points"
CI_REPLICATES = 10_000
# SHA-256 of the public seed label ``cyber-public-report-qwen38-step1000-v1``.
CI_SEED_SHA256 = "sha256:276f713e7022bd5e6daf20a47031df0add5381ef8f994d8511fb75a438b48285"
CI_QUANTILE_RULE = "linear_interpolation_n_minus_one"

BENCHMARKS = {
    "webexploitbench_level0": {
        "task_count": 15,
        "metric_denominator": 110,
        "title": "Public web-security benchmark",
        "metric": "Known weaknesses found at least once across eight attempts",
        "comparison_definition_sha256": (
            "sha256:ce3e0706e9c8f73bb552d83bd0841602dbb68c0f0d2909b1588432dff86fe95f"
        ),
        "terminal_schemas": {
            "experiment": "webexploitbench_collection_replica_set_v1",
            "arm_terminal": "webexploitbench_collection_replica_arm_terminal_v1",
            "score_acceptance": "webexploitbench_deferred_score_acceptance_v1",
        },
    },
    "fleet_development_dev17": {
        "task_count": 17,
        "metric_denominator": 17,
        "title": "Fleet development tasks",
        "metric": "Tasks solved at least once across eight attempts",
        "comparison_definition_sha256": (
            "sha256:bff3b01e6dcc4b189c9acb6140fbf868fda74e9bef4288c39bab0490cfc49fd2"
        ),
        "terminal_schemas": {
            "launch_terminal": "cyber_fleet_heldout_terminal_observation_v1",
            "accepted_cell": "fleet-rollout-ledger-cell-accepted-v1",
            "controller_terminal": "fleet-rollout-ledger-controller-terminal-v1",
        },
    },
}

_INPUT_FIELDS = {
    "schema_version",
    "status",
    "benchmark",
    "comparison_definition_sha256",
    "anonymization_method",
    "anonymization_receipt_sha256",
    "task_rows",
    "evidence",
    "receipt_sha256",
}
_TASK_FIELDS = {
    "metric_denominator",
    "base",
    "candidate",
}
_ARM_FIELDS = {
    "valid_attempts",
    "infrastructure_invalid_attempts",
    "metric_numerator",
}
_EVIDENCE_FIELDS = {
    "terminal_index_sha256",
    "scored_outcome_index_sha256",
    "terminal_schemas",
}


class PublicEvalImportError(ValueError):
    """A proposed public summary is incomplete, inconsistent, or unsafe."""


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
        raise PublicEvalImportError(f"{label} fields are incomplete or unknown")
    return value


def _load(path: Path, expected_sha256: str) -> tuple[dict[str, Any], str]:
    if path.is_symlink() or not path.is_file():
        raise PublicEvalImportError("aggregate input must be an exact regular file")
    raw = path.read_bytes()
    file_sha256 = _sha256(raw)
    if not _is_sha256(expected_sha256) or file_sha256 != expected_sha256:
        raise PublicEvalImportError("aggregate file digest differs from the reviewed digest")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PublicEvalImportError("aggregate input is not valid JSON") from exc
    if not isinstance(value, dict):
        raise PublicEvalImportError("aggregate input must be a JSON object")
    return value, file_sha256


def _metric_percent(numerator: int, denominator: int) -> float:
    return numerator * 100.0 / denominator


def _validate_arm(raw: object, *, denominator: int, label: str) -> dict[str, Any]:
    arm = _exact_fields(raw, _ARM_FIELDS, label)
    valid = arm["valid_attempts"]
    invalid = arm["infrastructure_invalid_attempts"]
    if type(valid) is not int or type(invalid) is not int or valid < 0 or invalid < 0:
        raise PublicEvalImportError(f"{label} attempt counts are invalid")
    if valid + invalid != PASS_K:
        raise PublicEvalImportError(f"{label} does not account for all eight attempts")
    numerator = arm["metric_numerator"]
    if valid != PASS_K:
        if numerator is not None:
            raise PublicEvalImportError(f"{label} incomplete result must not be scored")
        return {"valid_attempts": valid, "technical_failures": invalid, "metric_percent": None}
    if type(numerator) is not int or not 0 <= numerator <= denominator:
        raise PublicEvalImportError(f"{label} metric numerator is invalid")
    return {
        "valid_attempts": valid,
        "technical_failures": invalid,
        "metric_numerator": numerator,
        "metric_percent": _metric_percent(numerator, denominator),
    }


def _bootstrap_delta(rows: list[dict[str, Any]], selected: list[int]) -> float:
    denominator = sum(rows[index]["metric_denominator"] for index in selected)
    base = sum(rows[index]["base"]["metric_numerator"] for index in selected)
    candidate = sum(rows[index]["candidate"]["metric_numerator"] for index in selected)
    return _metric_percent(candidate, denominator) - _metric_percent(base, denominator)


def _quantile(values: list[float], fraction: float) -> float:
    position = (len(values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def _bootstrap_ci(rows: list[dict[str, Any]]) -> tuple[float, float]:
    """Return the exact deterministic task-bootstrap interval used by the report."""

    if not rows:
        raise PublicEvalImportError("at least one complete task pair is required")
    deltas = []
    for replicate in range(CI_REPLICATES):
        selected = []
        for position in range(len(rows)):
            material = f"{CI_SEED_SHA256}:{replicate}:{position}".encode()
            selected.append(int.from_bytes(hashlib.sha256(material).digest(), "big") % len(rows))
        deltas.append(_bootstrap_delta(rows, selected))
    deltas.sort()
    return _quantile(deltas, 0.025), _quantile(deltas, 0.975)


def _validate(value: dict[str, Any], file_sha256: str) -> dict[str, Any]:
    aggregate = _exact_fields(value, _INPUT_FIELDS, "aggregate")
    if aggregate["schema_version"] != INPUT_SCHEMA or aggregate["status"] != "final":
        raise PublicEvalImportError("aggregate is not a final supported summary")
    claimed = aggregate["receipt_sha256"]
    unsigned = dict(aggregate)
    unsigned.pop("receipt_sha256")
    if not _is_sha256(claimed) or claimed != _sha256(_canonical(unsigned)):
        raise PublicEvalImportError("aggregate self-digest is invalid")

    benchmark = aggregate["benchmark"]
    if not isinstance(benchmark, str):
        raise PublicEvalImportError("aggregate benchmark is invalid")
    spec = BENCHMARKS.get(benchmark)
    if spec is None:
        raise PublicEvalImportError("aggregate benchmark is unsupported")
    task_count = spec["task_count"]
    if aggregate["comparison_definition_sha256"] != spec["comparison_definition_sha256"]:
        raise PublicEvalImportError("comparison definition differs")
    if aggregate["anonymization_method"] != "private_random_permutation_v1" or not _is_sha256(
        aggregate["anonymization_receipt_sha256"]
    ):
        raise PublicEvalImportError("anonymous task ordering contract is invalid")

    rows = aggregate["task_rows"]
    if not isinstance(rows, list) or len(rows) != task_count:
        raise PublicEvalImportError("aggregate task rows are incomplete")
    public_rows: list[dict[str, Any]] = []
    calculation_rows: list[dict[str, Any]] = []
    totals = {arm: {"valid": 0, "invalid": 0} for arm in ARMS}
    denominator_total = 0
    for expected_index, raw_row in enumerate(rows):
        row = _exact_fields(raw_row, _TASK_FIELDS, "task row")
        denominator = row["metric_denominator"]
        if type(denominator) is not int or denominator < 1:
            raise PublicEvalImportError("task metric denominator is invalid")
        denominator_total += denominator
        arms = {
            arm: _validate_arm(
                row[arm], denominator=denominator, label=f"task {expected_index} {arm}"
            )
            for arm in ARMS
        }
        for arm in ARMS:
            totals[arm]["valid"] += arms[arm]["valid_attempts"]
            totals[arm]["invalid"] += arms[arm]["technical_failures"]
        complete_pair = all(arms[arm]["valid_attempts"] == PASS_K for arm in ARMS)
        if complete_pair:
            calculation_rows.append({"metric_denominator": denominator, **arms})
        public_rows.append(
            {
                "public_task_index": expected_index,
                "base": {
                    key: value for key, value in arms["base"].items() if key != "metric_numerator"
                },
                "candidate": {
                    key: value
                    for key, value in arms["candidate"].items()
                    if key != "metric_numerator"
                },
                "delta_percentage_points": (
                    arms["candidate"]["metric_percent"] - arms["base"]["metric_percent"]
                    if complete_pair
                    else None
                ),
            }
        )
    if denominator_total != spec["metric_denominator"]:
        raise PublicEvalImportError("task metric denominators differ from the frozen benchmark")

    paired_valid = len(calculation_rows)
    if paired_valid < 1:
        raise PublicEvalImportError("at least one complete task pair is required")
    paired_denominator = sum(row["metric_denominator"] for row in calculation_rows)
    public_summary: dict[str, Any] = {}
    for arm in ARMS:
        numerator = sum(row[arm]["metric_numerator"] for row in calculation_rows)
        public_summary[arm] = {
            "valid_attempts": totals[arm]["valid"],
            "technical_failures": totals[arm]["invalid"],
            "metric_percent": _metric_percent(numerator, paired_denominator),
        }
    delta = public_summary["candidate"]["metric_percent"] - public_summary["base"]["metric_percent"]
    lower, upper = _bootstrap_ci(calculation_rows)
    expected_ci = {
        "confidence_level": 0.95,
        "lower": lower,
        "upper": upper,
        "method": CI_METHOD,
        "unit": CI_UNIT,
        "resampling_unit": "task",
        "replicates": CI_REPLICATES,
        "seed_sha256": CI_SEED_SHA256,
        "quantile_rule": CI_QUANTILE_RULE,
    }

    evidence = _exact_fields(aggregate["evidence"], _EVIDENCE_FIELDS, "evidence")
    if any(not _is_sha256(evidence[field]) for field in _EVIDENCE_FIELDS - {"terminal_schemas"}):
        raise PublicEvalImportError("evidence digest is invalid")
    if evidence["terminal_schemas"] != spec["terminal_schemas"]:
        raise PublicEvalImportError("terminal schemas differ from the exact evaluator contracts")

    return {
        "benchmark": benchmark,
        "title": spec["title"],
        "metric": spec["metric"],
        "task_count": task_count,
        "pass_k": PASS_K,
        "task_rows": public_rows,
        "summary": {
            **public_summary,
            "metric_population": "paired_tasks_with_complete_eight_attempts",
            "paired_valid_tasks": paired_valid,
            "paired_technical_failure_tasks": task_count - paired_valid,
            "delta_percentage_points": delta,
            "confidence_interval": expected_ci,
        },
        "evidence": {
            "source_file_sha256": file_sha256,
            "source_receipt_sha256": claimed,
            "anonymization_receipt_sha256": aggregate["anonymization_receipt_sha256"],
            **{field: evidence[field] for field in _EVIDENCE_FIELDS - {"terminal_schemas"}},
        },
    }


def build_public_results(
    web_path: Path,
    fleet_path: Path,
    *,
    web_file_sha256: str,
    fleet_file_sha256: str,
) -> dict[str, Any]:
    """Validate both final studies and return deterministic public-only data."""

    studies = []
    for expected, path, digest in (
        ("webexploitbench_level0", web_path, web_file_sha256),
        ("fleet_development_dev17", fleet_path, fleet_file_sha256),
    ):
        value, file_sha256 = _load(path, digest)
        study = _validate(value, file_sha256)
        if study["benchmark"] != expected:
            raise PublicEvalImportError("aggregate paths are assigned to the wrong benchmark")
        studies.append(study)
    return {"schema_version": OUTPUT_SCHEMA, "studies": studies}


def write_public_results(value: Mapping[str, Any], output: Path) -> bool:
    """Atomically write deterministic site data; return False for an identical rerun."""

    if output.is_symlink() or output.parent.is_symlink() or not output.parent.is_dir():
        raise PublicEvalImportError("public output path is unsafe")
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
    parser.add_argument("--web", type=Path, required=True)
    parser.add_argument("--web-file-sha256", required=True)
    parser.add_argument("--fleet", type=Path, required=True)
    parser.add_argument("--fleet-file-sha256", required=True)
    parser.add_argument("--output", type=Path, default=Path("site/evaluation-results.json"))
    args = parser.parse_args(argv)
    try:
        value = build_public_results(
            args.web,
            args.fleet,
            web_file_sha256=args.web_file_sha256,
            fleet_file_sha256=args.fleet_file_sha256,
        )
        changed = write_public_results(value, args.output)
    except (OSError, PublicEvalImportError) as exc:
        parser.error(str(exc))
    print(json.dumps({"changed": changed, "study_count": len(value["studies"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
