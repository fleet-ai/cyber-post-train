"""Estimate the v2 Fleet development outcome signal from matched binary outcomes.

The input and output contain development evaluation results and belong in private
storage.  This module has no network, launch, W&B, or benchmark-integration path.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import tempfile
from pathlib import Path
from typing import Any

from evals.fleet import dev_outcome_protocol as protocol_module
from training.io import canonical_json, digest_json

OUTCOME_SCHEMA = "cyber_fleet_dev_matched_binary_outcomes_v1"
ESTIMATE_SCHEMA = "cyber_fleet_dev_paired_estimate_v1"
SELECTION_SCHEMA = "cyber_fleet_dev_primary_selection_v1"
PRIMARY_METRIC = "fleet_dev_paired_mean_success_delta_4fixed"
NAME = re.compile(r"^[a-z0-9][a-z0-9._-]{1,127}$")
SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
ROOT = Path(__file__).resolve().parents[2]


def _sealed(value: dict[str, Any]) -> dict[str, Any]:
    return {**value, "sha256": digest_json(value)}


def _check_seal(value: dict[str, Any], schema: str) -> None:
    unsigned = {key: item for key, item in value.items() if key != "sha256"}
    if value.get("schema") != schema or value.get("sha256") != digest_json(unsigned):
        raise ValueError(f"invalid {schema} seal")


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be an exact SHA-256")
    return value


def _arm(value: Any, label: str) -> dict[str, str]:
    fields = {"arm_id", "evaluation_child_sha256", "raw_result_manifest_sha256"}
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{label} arm identity differs")
    arm_id = value["arm_id"]
    if not isinstance(arm_id, str) or NAME.fullmatch(arm_id) is None:
        raise ValueError(f"{label} arm id is invalid")
    _sha(value["evaluation_child_sha256"], f"{label} evaluation child")
    _sha(value["raw_result_manifest_sha256"], f"{label} result manifest")
    return value


def _result(value: Any, label: str) -> bool:
    if not isinstance(value, dict) or set(value) != {"valid_outcome", "full_task_success"}:
        raise ValueError(f"{label} outcome differs")
    if type(value["valid_outcome"]) is not bool:
        raise ValueError(f"{label} validity must be boolean")
    if value["valid_outcome"] is not True:
        if value["full_task_success"] is not None:
            raise ValueError(f"{label} invalid outcome cannot carry a capability value")
        raise ValueError("paired primary comparison is incomplete")
    if type(value["full_task_success"]) is not bool:
        raise ValueError(f"{label} full-task success must be boolean")
    return value["full_task_success"]


def _quantile(sorted_values: list[float], quantile: float) -> float:
    rank = (len(sorted_values) - 1) * quantile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return sorted_values[lower]
    weight = rank - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def task_clustered_bootstrap(
    per_task_deltas: list[float], *, resamples: int, seed: int, interval: float
) -> list[float]:
    """Return a paired percentile interval while retaining all attempts per task."""
    if not per_task_deltas:
        raise ValueError("task-clustered bootstrap needs at least one task")
    if type(resamples) is not int or resamples < 1:
        raise ValueError("bootstrap resamples must be positive")
    if type(seed) is not int:
        raise ValueError("bootstrap seed must be an integer")
    if not isinstance(interval, (int, float)) or not 0.0 < interval < 1.0:
        raise ValueError("bootstrap interval must be between zero and one")
    generator = random.Random(seed)
    task_count = len(per_task_deltas)
    draws = sorted(
        sum(per_task_deltas[generator.randrange(task_count)] for _ in range(task_count))
        / task_count
        for _ in range(resamples)
    )
    tail = (1.0 - float(interval)) / 2.0
    return [_quantile(draws, tail), _quantile(draws, 1.0 - tail)]


def estimate(
    protocol: dict[str, Any], task_set: dict[str, Any], outcomes: dict[str, Any]
) -> dict[str, Any]:
    """Build one sealed private base-versus-candidate estimate."""
    protocol_module.validate_protocol(protocol, task_set)
    if protocol.get("schema") != protocol_module.PROTOCOL_SCHEMA_V2:
        raise ValueError("the corrected estimator requires a v2 Fleet development protocol")
    _check_seal(outcomes, OUTCOME_SCHEMA)
    if set(outcomes) != {
        "schema",
        "parent_protocol_sha256",
        "task_set_sha256",
        "arms",
        "rows",
        "sha256",
    }:
        raise ValueError("matched outcome document has unknown or missing fields")
    if outcomes["parent_protocol_sha256"] != protocol["sha256"]:
        raise ValueError("matched outcomes reference a different protocol")
    if outcomes["task_set_sha256"] != task_set["sha256"]:
        raise ValueError("matched outcomes reference a different task set")
    arms = outcomes["arms"]
    if not isinstance(arms, dict) or set(arms) != {"base", "candidate"}:
        raise ValueError("matched outcomes require base and candidate arms")
    base_arm = _arm(arms["base"], "base")
    candidate_arm = _arm(arms["candidate"], "candidate")
    if base_arm["arm_id"] == candidate_arm["arm_id"]:
        raise ValueError("base and candidate arm ids must differ")

    task_ids = sorted(row["task_version_id"] for row in task_set["tasks"])
    seeds = list(protocol["sampling"]["attempt_seeds"])
    expected_pairs = {(task_id, seed) for task_id in task_ids for seed in seeds}
    rows = outcomes["rows"]
    if not isinstance(rows, list):
        raise ValueError("matched outcome rows must be a list")
    observed: dict[tuple[str, int], tuple[bool, bool]] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "task_version_id",
            "attempt_seed",
            "base",
            "candidate",
        }:
            raise ValueError("matched outcome row has unknown or missing fields")
        pair = (row["task_version_id"], row["attempt_seed"])
        if pair not in expected_pairs:
            raise ValueError("matched outcome row is outside the frozen task-seed design")
        if pair in observed:
            raise ValueError("matched outcome document repeats a task-seed pair")
        observed[pair] = (_result(row["base"], "base"), _result(row["candidate"], "candidate"))
    if set(observed) != expected_pairs:
        raise ValueError("paired primary comparison requires every fixed task-seed pair")

    base_vector = []
    candidate_vector = []
    base_task_means = []
    candidate_task_means = []
    per_task_deltas = []
    base_pass_at_4 = []
    candidate_pass_at_4 = []
    for task_id in task_ids:
        base_values = [observed[(task_id, seed)][0] for seed in seeds]
        candidate_values = [observed[(task_id, seed)][1] for seed in seeds]
        base_mean = sum(base_values) / len(seeds)
        candidate_mean = sum(candidate_values) / len(seeds)
        base_task_means.append(base_mean)
        candidate_task_means.append(candidate_mean)
        per_task_deltas.append(candidate_mean - base_mean)
        base_pass_at_4.append(float(any(base_values)))
        candidate_pass_at_4.append(float(any(candidate_values)))
        for seed, base_success, candidate_success in zip(
            seeds, base_values, candidate_values, strict=True
        ):
            base_vector.append(
                {"task_version_id": task_id, "attempt_seed": seed, "success": base_success}
            )
            candidate_vector.append(
                {"task_version_id": task_id, "attempt_seed": seed, "success": candidate_success}
            )

    uncertainty = protocol["metrics"]["uncertainty"]
    paired_interval = task_clustered_bootstrap(
        per_task_deltas,
        resamples=uncertainty["resamples"],
        seed=uncertainty["seed"],
        interval=uncertainty["interval"],
    )
    task_count = len(task_ids)
    return _sealed(
        {
            "schema": ESTIMATE_SCHEMA,
            "parent_protocol_sha256": protocol["sha256"],
            "task_set_sha256": task_set["sha256"],
            "arms": {"base": dict(base_arm), "candidate": dict(candidate_arm)},
            "pairing": {
                "task_count": task_count,
                "attempt_seeds": seeds,
                "matched_pair_count": len(observed),
                "base_pair_outcomes_sha256": digest_json(base_vector),
                "candidate_pair_outcomes_sha256": digest_json(candidate_vector),
            },
            "primary": {
                "name": PRIMARY_METRIC,
                "base_mean_success": sum(base_task_means) / task_count,
                "candidate_mean_success": sum(candidate_task_means) / task_count,
                "candidate_minus_base_paired_delta": sum(per_task_deltas) / task_count,
            },
            "secondary_only": {
                "name": "fleet_dev_pass_at_4",
                "base": sum(base_pass_at_4) / task_count,
                "candidate": sum(candidate_pass_at_4) / task_count,
                "candidate_minus_base": (sum(candidate_pass_at_4) - sum(base_pass_at_4))
                / task_count,
                "hyperparameter_or_checkpoint_selection_eligible": False,
            },
            "uncertainty": {
                "method": uncertainty["method"],
                "cluster_unit": uncertainty["cluster_unit"],
                "within_cluster": uncertainty["within_cluster"],
                "resamples": uncertainty["resamples"],
                "seed": uncertainty["seed"],
                "interval": uncertainty["interval"],
                "candidate_minus_base_percentile_interval": paired_interval,
            },
            "release_policy": {
                "private_only": True,
                "wandb_export_allowed": False,
                "webexploitbench_input_or_tiebreaker": False,
            },
        }
    )


def _validate_estimate(value: dict[str, Any]) -> None:
    _check_seal(value, ESTIMATE_SCHEMA)
    if set(value) != {
        "schema",
        "parent_protocol_sha256",
        "task_set_sha256",
        "arms",
        "pairing",
        "primary",
        "secondary_only",
        "uncertainty",
        "release_policy",
        "sha256",
    }:
        raise ValueError("estimate has unknown or missing fields")
    _sha(value["parent_protocol_sha256"], "estimate parent protocol")
    _sha(value["task_set_sha256"], "estimate task set")
    arms = value.get("arms")
    if not isinstance(arms, dict) or set(arms) != {"base", "candidate"}:
        raise ValueError("estimate arms differ")
    base_arm = _arm(arms["base"], "base")
    candidate_arm = _arm(arms["candidate"], "candidate")
    if base_arm["arm_id"] == candidate_arm["arm_id"]:
        raise ValueError("estimate base and candidate ids must differ")
    pairing = value.get("pairing")
    if not isinstance(pairing, dict) or set(pairing) != {
        "task_count",
        "attempt_seeds",
        "matched_pair_count",
        "base_pair_outcomes_sha256",
        "candidate_pair_outcomes_sha256",
    }:
        raise ValueError("estimate pairing contract differs")
    if (
        pairing["task_count"] != protocol_module.DEV_TASKS
        or pairing["attempt_seeds"] != protocol_module.SAMPLING["attempt_seeds"]
        or pairing["matched_pair_count"] != protocol_module.DEV_TASKS * protocol_module.PASS_K
    ):
        raise ValueError("estimate does not contain the fixed matched design")
    _sha(pairing["base_pair_outcomes_sha256"], "base pair outcomes")
    _sha(pairing["candidate_pair_outcomes_sha256"], "candidate pair outcomes")
    primary = value.get("primary")
    if not isinstance(primary, dict) or set(primary) != {
        "name",
        "base_mean_success",
        "candidate_mean_success",
        "candidate_minus_base_paired_delta",
    }:
        raise ValueError("estimate primary fields differ")
    if primary["name"] != PRIMARY_METRIC:
        raise ValueError("estimate does not carry the corrected primary metric")
    base, candidate, delta = (
        primary["base_mean_success"],
        primary["candidate_mean_success"],
        primary["candidate_minus_base_paired_delta"],
    )
    if any(
        isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item)
        for item in (base, candidate, delta)
    ):
        raise ValueError("estimate primary values must be finite numbers")
    if not 0.0 <= base <= 1.0 or not 0.0 <= candidate <= 1.0 or not -1.0 <= delta <= 1.0:
        raise ValueError("estimate primary values are outside binary-outcome bounds")
    if not math.isclose(candidate - base, delta, abs_tol=1e-12):
        raise ValueError("estimate paired delta differs from its arm means")
    secondary = value.get("secondary_only")
    if not isinstance(secondary, dict) or set(secondary) != {
        "name",
        "base",
        "candidate",
        "candidate_minus_base",
        "hyperparameter_or_checkpoint_selection_eligible",
    }:
        raise ValueError("estimate secondary fields differ")
    if (
        secondary["name"] != "fleet_dev_pass_at_4"
        or secondary["hyperparameter_or_checkpoint_selection_eligible"] is not False
    ):
        raise ValueError("pass@4 cannot select an arm")
    secondary_values = (
        secondary["base"],
        secondary["candidate"],
        secondary["candidate_minus_base"],
    )
    if (
        any(
            isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item)
            for item in secondary_values
        )
        or not (
            0.0 <= secondary["base"] <= 1.0
            and 0.0 <= secondary["candidate"] <= 1.0
            and -1.0 <= secondary["candidate_minus_base"] <= 1.0
        )
        or not math.isclose(
            secondary["candidate"] - secondary["base"],
            secondary["candidate_minus_base"],
            abs_tol=1e-12,
        )
    ):
        raise ValueError("pass@4 summary is inconsistent")
    uncertainty = value.get("uncertainty")
    if not isinstance(uncertainty, dict) or set(uncertainty) != {
        "method",
        "cluster_unit",
        "within_cluster",
        "resamples",
        "seed",
        "interval",
        "candidate_minus_base_percentile_interval",
    }:
        raise ValueError("estimate uncertainty fields differ")
    if (
        uncertainty["method"] != "paired task-clustered percentile bootstrap"
        or uncertainty["cluster_unit"] != "exact task_version_id"
        or uncertainty["within_cluster"] != "retain all four matched base/candidate seed pairs"
        or uncertainty["resamples"] != 10000
        or uncertainty["seed"] != 20260911
        or uncertainty["interval"] != 0.95
    ):
        raise ValueError("estimate does not use the frozen task-clustered bootstrap")
    bounds = uncertainty["candidate_minus_base_percentile_interval"]
    if (
        not isinstance(bounds, list)
        or len(bounds) != 2
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(item)
            or not -1.0 <= item <= 1.0
            for item in bounds
        )
        or bounds[0] > bounds[1]
    ):
        raise ValueError("estimate bootstrap interval is invalid")
    release = value.get("release_policy", {})
    if release != {
        "private_only": True,
        "wandb_export_allowed": False,
        "webexploitbench_input_or_tiebreaker": False,
    }:
        raise ValueError("estimate release policy drifted")


def select_primary_arm(estimates: list[dict[str, Any]]) -> dict[str, Any]:
    """Select only on the paired four-attempt Fleet development point estimate."""
    if not estimates:
        raise ValueError("selection requires at least one estimate")
    for value in estimates:
        _validate_estimate(value)
    first = estimates[0]
    common = {
        "parent_protocol_sha256": first["parent_protocol_sha256"],
        "task_set_sha256": first["task_set_sha256"],
        "base_arm": first["arms"]["base"],
        "base_pair_outcomes_sha256": first["pairing"]["base_pair_outcomes_sha256"],
        "attempt_seeds": first["pairing"]["attempt_seeds"],
    }
    arm_ids = []
    for value in estimates:
        observed = {
            "parent_protocol_sha256": value["parent_protocol_sha256"],
            "task_set_sha256": value["task_set_sha256"],
            "base_arm": value["arms"]["base"],
            "base_pair_outcomes_sha256": value["pairing"]["base_pair_outcomes_sha256"],
            "attempt_seeds": value["pairing"]["attempt_seeds"],
        }
        if observed != common:
            raise ValueError("candidate estimates do not share one exact matched base design")
        arm_ids.append(value["arms"]["candidate"]["arm_id"])
    if len(set(arm_ids)) != len(arm_ids):
        raise ValueError("candidate arm ids must be unique")
    best = max(value["primary"]["candidate_minus_base_paired_delta"] for value in estimates)
    winners = sorted(
        value["arms"]["candidate"]["arm_id"]
        for value in estimates
        if value["primary"]["candidate_minus_base_paired_delta"] == best
    )
    return _sealed(
        {
            "schema": SELECTION_SCHEMA,
            **common,
            "primary_metric": PRIMARY_METRIC,
            "candidate_estimate_sha256": sorted(value["sha256"] for value in estimates),
            "winner_arm_ids": winners,
            "status": (
                "selected" if len(winners) == 1 else "tie_requires_fresh_predeclared_confirmation"
            ),
            "pass_at_4_used": False,
            "webexploitbench_used": False,
        }
    )


def _private_path(path: Path) -> Path:
    resolved = path.resolve()
    private = (ROOT / "data/private").resolve()
    if (resolved == ROOT or ROOT in resolved.parents) and not (
        resolved == private or private in resolved.parents
    ):
        raise ValueError("development outcomes and estimates belong under data/private")
    return resolved


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain one object")
    return value


def _write_private_once(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    estimate_parser = commands.add_parser("estimate")
    estimate_parser.add_argument("--protocol", type=Path, required=True)
    estimate_parser.add_argument("--task-set", type=Path, required=True)
    estimate_parser.add_argument("--outcomes", type=Path, required=True)
    estimate_parser.add_argument("--output", type=Path, required=True)
    select_parser = commands.add_parser("select")
    select_parser.add_argument("--estimate", type=Path, action="append", required=True)
    select_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = _private_path(args.output)
    if output.exists():
        raise FileExistsError("refusing to replace a private estimate or decision")
    if args.command == "estimate":
        outcomes_path = _private_path(args.outcomes)
        result = estimate(_read(args.protocol), _read(args.task_set), _read(outcomes_path))
    else:
        result = select_primary_arm([_read(_private_path(path)) for path in args.estimate])
    _write_private_once(output, result)
    print(canonical_json({"output_sha256": result["sha256"], "written": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
