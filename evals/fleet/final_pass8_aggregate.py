"""Private final gate for the matched Fleet dev17 pass@8 comparison.

The gate opens scores only after all sixteen pass@1 replica ledgers are held in
repeatable-read snapshots and every score-blind readiness check has passed.
Its public artifact contains only anonymized task-level counts and evidence
digests; task, cell, session, prompt, trace, and verifier identities stay in
mode-0600 private receipts.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import tempfile
import uuid
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import parse_qsl, quote, urlsplit, urlunsplit

PLAN_SCHEMA = "cyber_fleet_matched_pass8_protocol_v2_final_plan_v1"
PUBLIC_SCHEMA = "cyber_sanitized_matched_pass8_aggregate_v1"
BENCHMARK = "fleet_development_dev17"
PREDECESSOR_COMPARISON_DEFINITION_SHA256 = (
    "sha256:1154b450624a8b9a567464916da95874c44933175c408b86a80ff5bca0eada70"
)
MIGRATION_SCHEMA = "cyber_qwen38_fleet_protocol_v2_replica_migration_receipt_v1"
COMPARISON_DEFINITION_SCHEMA = "cyber_qwen38_fleet_dev17_pass8_comparison_definition_v2"
TERMINAL_SCHEMA = "cyber_fleet_heldout_terminal_observation_v1"
ACCEPTED_SCHEMA = "fleet-rollout-ledger-cell-accepted-v1"
CONTROLLER_TERMINAL_SCHEMA = "fleet-rollout-ledger-controller-terminal-v1"
SOURCE_SEEDS = tuple(range(46, 54))
FROZEN_INVALID_SEEDS = (47, 51, 52, 53)
ARMS = ("base", "candidate")
TASK_COUNT = 17
PASS_K = 8
EXACT_PLAN_FIELD_DIGESTS = {
    "tasks": "sha256:3398757c75f22de021ca325262914d7af52b8e8d32617cb0ba5c25ec1ee463f4",
    "harness": "sha256:cc289a5221a6bfdaf57bf24652c3a8e67d1469d21149e6af889628207a5172bb",
    "images": "sha256:4d9238bf16805846d9ef355364546c762903e29d85080be7b49ba90f2802257f",
    "sampling": "sha256:67a5e03019869fbcc5c68c694e9b0daf72ab20a79c9e8122577889913b42f3c5",
    "arms": "sha256:ba923970850f4eb59456f6788d43b04c712a3787f1a46224affbed8bde4a2c30",
}
SHA256 = re.compile(r"(?:sha256:)?[0-9a-f]{64}")
UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")

CELL_FIELDS = (
    "cell_id",
    "experiment_id",
    "task_key",
    "task_version_id",
    "model_id",
    "model_revision",
    "serving_block",
    "endpoint_model_id",
    "harness_id",
    "attempt",
    "state",
    "session_id",
    "lease_expires_at",
    "retry_count",
    "max_retries",
    "result_class",
    "receipt_digest",
    "failure_code",
    "reconciliation_digest",
)
LOCAL_RECORD_FIELDS = (
    "execution_id",
    "cell_id",
    "execution_generation",
    "run_id",
    "session_id",
    "verifier_execution_id",
    "score",
    "config_sha256",
    "artifact_directory",
    "trace_path",
    "trace_sha256",
    "result_path",
    "result_sha256",
    "reward_path",
    "reward_sha256",
    "session_ingest_path",
    "session_ingest_sha256",
    "cleanup_path",
    "cleanup_sha256",
    "session_ingest_status",
    "agent_exit_code",
    "agent_termination",
    "elapsed_seconds",
)
PLAN_STORED_FIELDS = (
    "cell_id",
    "experiment_id",
    "task_key",
    "task_version_id",
    "model_id",
    "model_revision",
    "serving_block",
    "endpoint_model_id",
    "harness_id",
    "attempt",
    "max_retries",
)
RUNTIME_FILES = {
    "evaluate.py",
    "rollout_worker.py",
    "rollout_postgres.py",
    "rollout_ledger.py",
    "opencode_self_hosted.py",
    "fixed_proxy.py",
    "exact_pass4_crypto.py",
}
ALLOWED_EXCLUSION_REASONS = {
    "terminal_replica_incomplete",
    "replica_not_started",
    "mixed_infrastructure_invalid_replica",
}


class FinalAggregateError(ValueError):
    """The final comparison is incomplete, mismatched, or unsafe to open."""


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, ValueError) as exc:
        raise FinalAggregateError("value cannot be canonically encoded") from exc


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _plain_digest(value: object) -> str:
    return _digest(value).removeprefix("sha256:")


def _cell_id(row: Mapping[str, Any]) -> str:
    identity = "\0".join(
        str(row[field]) for field in ("experiment_id", "task_version_id", "model_id", "attempt")
    )
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"fleet-cyber-rollout-cell-v1:{identity}"))


def _normalized_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or SHA256.fullmatch(value) is None:
        raise FinalAggregateError(f"{label} is not a SHA-256 digest")
    return value.removeprefix("sha256:")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FinalAggregateError(f"{label} is not an exact regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FinalAggregateError(f"{label} is not readable JSON") from exc
    if not isinstance(value, dict):
        raise FinalAggregateError(f"{label} must be a JSON object")
    return value


def _require_self_digest(value: dict[str, Any], field: str, label: str) -> str:
    claimed = value.get(field)
    if not isinstance(claimed, str) or SHA256.fullmatch(claimed) is None:
        raise FinalAggregateError(f"{label} self digest is missing")
    unsigned = {key: item for key, item in value.items() if key != field}
    expected = _digest(unsigned)
    if claimed.removeprefix("sha256:") != expected.removeprefix("sha256:"):
        raise FinalAggregateError(f"{label} self digest differs")
    return expected


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    raw = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _task_inputs(task_set_path: Path, roster_path: Path) -> tuple[list[dict[str, str]], dict]:
    task_set = _read_json(task_set_path, "dev17 task selection")
    roster = _read_json(roster_path, "dev17 exact binding roster")
    _require_self_digest(roster, "sha256", "dev17 exact binding roster")
    if any(
        (
            task_set.get("schema") != "cyber_eval_task_selection_v2",
            task_set.get("selection_role") != "dev",
            task_set.get("selection_sha256")
            != "sha256:38ea6686afa068c19e69fea2493e01027fdf99f7e01b20376e107b1a0cfa0b68",
            roster.get("binding_count") != TASK_COUNT,
            roster.get("bindings_sha256")
            != "sha256:32230df8759bb74eb306d16cd9e32686618b529aa7c0b0ac1fa1d751ecd38728",
        )
    ):
        raise FinalAggregateError("immutable dev17 selection identity differs")
    tasks = task_set.get("tasks")
    bindings = roster.get("bindings")
    if not isinstance(tasks, list) or not isinstance(bindings, list) or len(tasks) != TASK_COUNT:
        raise FinalAggregateError("immutable dev17 selection is incomplete")
    selected: list[dict[str, str]] = []
    for task, binding in zip(tasks, bindings, strict=True):
        if not isinstance(task, dict) or not isinstance(binding, dict):
            raise FinalAggregateError("immutable dev17 row is invalid")
        version = task.get("task_version_id")
        key = task.get("task_key")
        bound = binding.get("task", {})
        if (
            not isinstance(version, str)
            or UUID.fullmatch(version) is None
            or not isinstance(key, str)
            or not key
            or not isinstance(bound, dict)
            or bound.get("version_id") != version
            or bound.get("key") != key
        ):
            raise FinalAggregateError("dev17 task selection and binding roster differ")
        selected.append(
            {
                field: str(task[field])
                for field in (
                    "task_key",
                    "task_version_id",
                    "env_key",
                    "env_version",
                    "environment_version_id",
                    "data_key",
                    "data_version",
                )
            }
        )
    if len({row["task_version_id"] for row in selected}) != TASK_COUNT:
        raise FinalAggregateError("dev17 task versions are not unique")
    return selected, roster


def _exact_digest_fields(value: Mapping[str, Any], fields: Sequence[str], label: str) -> None:
    for field in fields:
        _normalized_digest(value.get(field), f"{label} {field}")


def _deterministic_mapping(originals: Sequence[int]) -> list[dict[str, int]]:
    if (
        not originals
        or list(originals) != sorted(originals)
        or tuple(originals) != FROZEN_INVALID_SEEDS
        or len(set(originals)) != len(originals)
        or any(type(seed) is not int or seed not in SOURCE_SEEDS for seed in originals)
    ):
        raise FinalAggregateError("protocol-v2 excluded original seeds differ from the freeze")
    return [
        {"invalid_original_seed": original, "replacement_seed": 54 + index}
        for index, original in enumerate(originals)
    ]


def _comparison_contract(
    comparison: dict[str, Any],
) -> tuple[list[dict[str, int]], list[int], dict[int, dict[str, Any]]]:
    """Validate the stable scientific definition embedded by protocol v2."""

    _require_self_digest(comparison, "sha256", "protocol-v2 comparison definition")
    excluded = comparison.get("excluded_original_seeds")
    if not isinstance(excluded, list):
        raise FinalAggregateError("protocol-v2 excluded original seeds are missing")
    mapping = _deterministic_mapping(excluded)
    included = sorted(
        [
            *(seed for seed in SOURCE_SEEDS if seed not in set(excluded)),
            *(row["replacement_seed"] for row in mapping),
        ]
    )
    if any(
        (
            comparison.get("schema") != COMPARISON_DEFINITION_SCHEMA,
            comparison.get("protocol_study_id") != "q38-dev17-base-step1000-p8-v2",
            comparison.get("predecessor_comparison_definition_sha256")
            != PREDECESSOR_COMPARISON_DEFINITION_SHA256,
            comparison.get("aggregation") != "eight_predeclared_pass1_replicas_per_task_and_arm",
            comparison.get("original_seeds") != list(SOURCE_SEEDS),
            comparison.get("replacement_mapping") != mapping,
            comparison.get("included_seeds") != included,
            len(included) != PASS_K,
            len(set(included)) != PASS_K,
            comparison.get("task_count") != TASK_COUNT,
            comparison.get("sessions_per_arm") != TASK_COUNT * PASS_K,
            comparison.get("total_sessions") != TASK_COUNT * PASS_K * len(ARMS),
            comparison.get("comparison_arms") != list(ARMS),
            comparison.get("pass_k_per_replica") != 1,
            comparison.get("retry_limit") != 0,
            comparison.get("training_data_eligible") is not False,
            comparison.get("whole_replica_pairs_only") is not True,
            comparison.get("cell_level_replacement_forbidden") is not True,
            comparison.get("task_selection_sha256")
            != "sha256:38ea6686afa068c19e69fea2493e01027fdf99f7e01b20376e107b1a0cfa0b68",
            comparison.get("split_manifest_sha256")
            != "sha256:05b3a8dc90ca93adc9671942d75ecb54ff8d0951b0dd48a087e29ec1e641840c",
            comparison.get("binding_roster_sha256")
            != "sha256:39ae49c2db322725800b15347d5da5d171e7e03b21dec5b570683414d70e83c5",
        )
    ):
        raise FinalAggregateError("protocol-v2 comparison identity differs")
    expected_models = {
        "base": {
            "model_id": "qwen3.8-27b",
            "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        },
        "candidate": {
            "model_id": "chris-q38-t3k32-s1000-v1",
            "revision": ("sha256:023c5f8b0559ba050f0d672a6bc27aabecec7d5837595f8ea5bc914446d26db5"),
        },
    }
    if (
        comparison.get("models") != expected_models
        or _digest(comparison.get("harness")) != EXACT_PLAN_FIELD_DIGESTS["harness"]
        or _digest(comparison.get("images")) != EXACT_PLAN_FIELD_DIGESTS["images"]
        or comparison.get("sampling_without_seed") != {"temperature": 0.6, "top_p": 0.95}
    ):
        raise FinalAggregateError("protocol-v2 model or treatment identity differs")
    protocols = comparison.get("replica_protocols")
    if (
        not isinstance(protocols, list)
        or len(protocols) != PASS_K
        or [row.get("seed") for row in protocols if isinstance(row, dict)] != included
    ):
        raise FinalAggregateError("protocol-v2 included replica protocol roster is incomplete")
    protocol_by_seed: dict[int, dict[str, Any]] = {}
    replacement_seeds = {row["replacement_seed"] for row in mapping}
    for row in protocols:
        if not isinstance(row, dict) or set(row) != {
            "seed",
            "origin",
            "protocol_id",
            "comparison_protocol_sha256",
        }:
            raise FinalAggregateError("protocol-v2 replica protocol row is malformed")
        seed = row.get("seed")
        expected_origin = (
            "whole_pair_replacement" if seed in replacement_seeds else "retained_original"
        )
        if (
            type(seed) is not int
            or seed not in included
            or seed in protocol_by_seed
            or row.get("origin") != expected_origin
            or not isinstance(row.get("protocol_id"), str)
            or not row["protocol_id"]
        ):
            raise FinalAggregateError("protocol-v2 replica protocol identity differs")
        _normalized_digest(row.get("comparison_protocol_sha256"), "replica protocol")
        protocol_by_seed[seed] = row
    if set(protocol_by_seed) != set(included):
        raise FinalAggregateError("protocol-v2 included replica protocol seeds differ")
    return mapping, included, protocol_by_seed


def _migration_input(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt = _read_json(path, "protocol-v2 migration receipt")
    _require_self_digest(receipt, "sha256", "protocol-v2 migration receipt")
    comparison = receipt.get("comparison_definition")
    if not isinstance(comparison, dict):
        raise FinalAggregateError("protocol-v2 comparison definition is missing")
    mapping, included, protocol_by_seed = _comparison_contract(comparison)
    comparison_path = path.parent / "COMPARISON_DEFINITION.json"
    comparison_file = _read_json(comparison_path, "protocol-v2 comparison definition file")
    if comparison_file != comparison or receipt.get("comparison_definition_file_sha256") != (
        _file_digest(comparison_path)
    ):
        raise FinalAggregateError("protocol-v2 comparison definition file differs")
    excluded = comparison["excluded_original_seeds"]
    source = receipt.get("source")
    if not isinstance(source, dict):
        raise FinalAggregateError("protocol-v2 source study evidence is missing")
    _exact_digest_fields(
        source,
        ("preparation_receipt_sha256", "preparation_receipt_file_sha256"),
        "protocol-v2 source study",
    )
    _normalized_digest(receipt.get("migration_intent_sha256"), "migration intent")
    if any(
        (
            receipt.get("schema") != MIGRATION_SCHEMA,
            source.get("protocol_study_id") != "q38-dev17-seeds46to53-base-step1000-p8-v1",
            source.get("seeds") != list(SOURCE_SEEDS),
            source.get("arm_count") != len(SOURCE_SEEDS) * len(ARMS),
            receipt.get("mapping_rule")
            != (
                "invalid original seeds sorted ascending map to the smallest unused seeds "
                "greater than 53, ascending"
            ),
            receipt.get("excluded_original_seeds") != excluded,
            receipt.get("included_seeds") != included,
            receipt.get("external_mutations") != 0,
            receipt.get("launch_performed") is not False,
        )
    ):
        raise FinalAggregateError("protocol-v2 migration or comparison identity differs")
    replacement_seeds = {row["replacement_seed"] for row in mapping}
    migrations = receipt.get("migrations")
    if not isinstance(migrations, list) or len(migrations) != len(mapping):
        raise FinalAggregateError("protocol-v2 excluded source evidence is incomplete")
    for expected, row in zip(mapping, migrations, strict=True):
        if (
            not isinstance(row, dict)
            or row.get("invalid_original_seed") != expected["invalid_original_seed"]
            or row.get("replacement_seed") != expected["replacement_seed"]
            or row.get("reason_class") not in ALLOWED_EXCLUSION_REASONS
            or row.get("whole_pair_excluded") is not True
            or not isinstance(row.get("evidence_receipt_sha256s"), list)
            or not row["evidence_receipt_sha256s"]
            or set(row.get("excluded_source_arms", {})) != set(ARMS)
        ):
            raise FinalAggregateError("protocol-v2 excluded source evidence differs")
        for value in row["evidence_receipt_sha256s"]:
            _normalized_digest(value, "excluded source evidence")
        source_arms = row["excluded_source_arms"]
        for arm in ARMS:
            evidence = source_arms[arm]
            if not isinstance(evidence, dict):
                raise FinalAggregateError("protocol-v2 excluded source arm evidence is invalid")
            _exact_digest_fields(
                evidence,
                (
                    "packet_file_sha256",
                    "evaluation_identity_sha256",
                    "evaluation_config_sha256",
                    "comparison_protocol_file_sha256",
                    "comparison_protocol_sha256",
                ),
                "excluded source arm",
            )
    replacement_protocols = receipt.get("replacement_protocols")
    replacement_arms = receipt.get("replacement_arms")
    if (
        not isinstance(replacement_protocols, list)
        or len(replacement_protocols) != len(mapping)
        or not isinstance(replacement_arms, list)
        or len(replacement_arms) != len(mapping) * len(ARMS)
    ):
        raise FinalAggregateError("protocol-v2 replacement evidence is incomplete")
    replacement_protocol_by_seed: dict[int, dict[str, Any]] = {}
    for expected, row in zip(mapping, replacement_protocols, strict=True):
        seed = expected["replacement_seed"]
        if (
            not isinstance(row, dict)
            or row.get("invalid_original_seed") != expected["invalid_original_seed"]
            or row.get("replacement_seed") != seed
            or row.get("protocol_id") != protocol_by_seed[seed]["protocol_id"]
            or row.get("sha256") != protocol_by_seed[seed]["comparison_protocol_sha256"]
        ):
            raise FinalAggregateError("protocol-v2 replacement protocol differs")
        _exact_digest_fields(row, ("sha256", "file_sha256"), "replacement protocol")
        replacement_protocol_by_seed[seed] = row
    arm_by_identity: dict[tuple[int, str], dict[str, Any]] = {}
    original_by_replacement = {
        row["replacement_seed"]: row["invalid_original_seed"] for row in mapping
    }
    for row in replacement_arms:
        if not isinstance(row, dict):
            raise FinalAggregateError("protocol-v2 replacement arm evidence is invalid")
        seed = row.get("replacement_seed")
        arm = row.get("arm_id")
        identity = (seed, arm)
        if (
            type(seed) is not int
            or seed not in replacement_seeds
            or arm not in ARMS
            or identity in arm_by_identity
            or row.get("invalid_original_seed") != original_by_replacement[seed]
            or row.get("comparison_protocol_sha256")
            != protocol_by_seed[seed]["comparison_protocol_sha256"]
        ):
            raise FinalAggregateError("protocol-v2 replacement arm identity differs")
        _exact_digest_fields(
            row,
            (
                "packet_file_sha256",
                "evaluation_identity_sha256",
                "serving_proof_file_sha256",
                "comparison_protocol_file_sha256",
                "comparison_protocol_sha256",
            ),
            "replacement arm",
        )
        arm_by_identity[identity] = row
    if set(arm_by_identity) != {(seed, arm) for seed in replacement_seeds for arm in ARMS}:
        raise FinalAggregateError("protocol-v2 replacement arm roster differs")
    scientific = receipt.get("scientific_identity")
    capacity = receipt.get("capacity")
    privacy = receipt.get("privacy")
    selection = receipt.get("selection")
    retirement = receipt.get("retirement_evidence")
    if isinstance(retirement, dict):
        _exact_digest_fields(
            retirement,
            ("sha256", "file_sha256"),
            "protocol-v2 retirement evidence",
        )
    _exact_digest_fields(
        receipt,
        ("live_parity_file_sha256", "live_parity_receipt_sha256"),
        "protocol-v2 migration",
    )
    planned = len(mapping) * len(ARMS) * TASK_COUNT
    if any(
        (
            scientific
            != {
                "task_count_per_arm": TASK_COUNT,
                "comparison_arms": list(ARMS),
                "pass_k": 1,
                "retry_limit": 0,
                "same_task_versions_models_harness_budgets_and_sampling_recipe": True,
                "whole_replica_pairs_only": True,
                "cell_level_replacement_forbidden": True,
                "seed_reuse_forbidden": True,
                "later_invalid_seed_requires_versioned_successor_before_score_unseal": True,
            },
            not isinstance(capacity, dict),
            isinstance(capacity, dict) and capacity.get("new_replacement_rollouts") != planned,
            isinstance(capacity, dict)
            and capacity.get("final_comparison_rollouts") != TASK_COUNT * PASS_K * len(ARMS),
            isinstance(capacity, dict)
            and capacity.get("original_base_rollouts") != TASK_COUNT * PASS_K,
            isinstance(capacity, dict)
            and capacity.get("original_candidate_started_seeds") != [46, 47, 48, 49, 50, 51],
            isinstance(capacity, dict)
            and capacity.get("original_candidate_started_seed_rollouts") != TASK_COUNT * 6,
            isinstance(capacity, dict)
            and capacity.get("original_candidate_retired_before_start_seeds") != [52, 53],
            isinstance(capacity, dict)
            and capacity.get("original_candidate_retired_before_start_rollouts") != 0,
            isinstance(capacity, dict)
            and capacity.get("scoring_or_metadata_cpu_model_rollouts") != 0,
            isinstance(capacity, dict)
            and capacity.get("cumulative_model_rollouts_consumed_or_planned_today")
            != TASK_COUNT * PASS_K + TASK_COUNT * 6 + planned,
            isinstance(capacity, dict) and capacity.get("daily_rollout_cap") != 500,
            isinstance(capacity, dict) and capacity.get("within_daily_cap") is not True,
            not isinstance(retirement, dict),
            isinstance(retirement, dict) and retirement.get("targets") != 2,
            isinstance(retirement, dict) and retirement.get("model_rollouts") != 0,
            isinstance(retirement, dict)
            and retirement.get("outputs_or_databases_deleted") is not False,
            privacy
            != {
                "score_values_read": False,
                "prompts_responses_flags_rewards_or_trace_content_read": False,
                "infrastructure_reason_classes_only": True,
            },
            not isinstance(selection, dict),
            isinstance(selection, dict)
            and selection.get("selection_sha256")
            != "sha256:38ea6686afa068c19e69fea2493e01027fdf99f7e01b20376e107b1a0cfa0b68",
            isinstance(selection, dict)
            and selection.get("split_sha256")
            != "sha256:05b3a8dc90ca93adc9671942d75ecb54ff8d0951b0dd48a087e29ec1e641840c",
            isinstance(selection, dict)
            and selection.get("binding_roster_sha256")
            != "sha256:39ae49c2db322725800b15347d5da5d171e7e03b21dec5b570683414d70e83c5",
        )
    ):
        raise FinalAggregateError("protocol-v2 migration safety contract differs")
    return receipt, comparison


def _replica_descriptor(
    *,
    seed: int,
    arm: str,
    origin: str,
    protocol: Mapping[str, Any],
    replacement_evidence: Mapping[tuple[int, str], Mapping[str, Any]],
    original_by_replacement: Mapping[int, int],
) -> dict[str, Any]:
    if origin == "retained_original":
        suffix = "base-p1-v1" if arm == "base" else "t3k32s1000-p1-v2"
        experiment = f"q38-dev17-s{seed}-{suffix}"
        job_name = f"chris-{experiment}"
        config_map_name = (
            f"chris-q38-dev17-s{seed}-base-code-v1"
            if arm == "base"
            else f"chris-q38-dev17-s{seed}-t3k32s1000-code-v2"
        )
        output_root = f"/mnt/sfs/jobs/chris-q38-fleet-dev17-s{seed}-{suffix}"
        database = experiment.replace("-", "_")
        expected_evaluation_identity = None
        replaces_seed = None
    else:
        model = "base" if arm == "base" else "t3k32s1000"
        experiment = f"q38-dev17-s{seed}-{model}-replacement-p1-v2"
        short = f"{model}-repl-p1-v2"
        job_name = f"chris-q38-dev17-s{seed}-{short}"
        config_map_name = f"chris-q38-dev17-s{seed}-{model}-repl-code-v2"
        output_root = f"/mnt/sfs/jobs/chris-q38-fleet-dev17-s{seed}-{short}"
        database = f"q38_dev17_s{seed}_{model}_repl_p1_v2"
        expected_evaluation_identity = replacement_evidence[(seed, arm)][
            "evaluation_identity_sha256"
        ]
        replaces_seed = original_by_replacement[seed]
    return {
        "seed": seed,
        "arm": arm,
        "origin": origin,
        "replaces_seed": replaces_seed,
        "experiment_id": experiment,
        "database": database,
        "job_name": job_name,
        "config_map_name": config_map_name,
        "output_root": output_root,
        "terminal_receipt_path": f"{output_root}/TERMINAL_OBSERVATION.json",
        "protocol_id": protocol["protocol_id"],
        "comparison_protocol_sha256": protocol["comparison_protocol_sha256"],
        "evaluation_identity_sha256": expected_evaluation_identity,
    }


def build_current_study_plan(
    *,
    task_set_path: Path,
    roster_path: Path,
    base_config_path: Path,
    migration_receipt_path: Path,
) -> dict[str, Any]:
    """Build the exact score-blind gate plan for the protocol-v2 pass@8 study."""

    tasks, roster = _task_inputs(task_set_path, roster_path)
    migration, comparison = _migration_input(migration_receipt_path)
    base = _read_json(base_config_path, "base evaluation config")
    harness = base.get("harness")
    images = base.get("images")
    routes = base.get("routes")
    if (
        not isinstance(harness, dict)
        or harness.get("harness") != "opencode"
        or harness.get("harness_version") != "1.18.27"
        or harness.get("context_management") != "opencode_1.18.27_native_compaction_autocontinue_v2"
        or not isinstance(images, dict)
        or not isinstance(routes, dict)
        or set(routes) != {"base"}
    ):
        raise FinalAggregateError("base evaluation treatment identity differs")
    base_route = routes["base"]
    if base_route.get("task_versions") != [row["task_version_id"] for row in tasks]:
        raise FinalAggregateError("base route task order differs from dev17")
    candidate_route = json.loads(json.dumps(base_route))
    candidate_route.update(model="teacher3k32-step1000", served_id="chris-q38-t3k32-s1000-v1")
    candidate_route["model_info"]["model_path"] = "/scratch/models/chris-q38-t3k32-s1000-v1"
    candidate_route["server_info"]["model_path"] = "/scratch/models/chris-q38-t3k32-s1000-v1"
    arms = {
        "base": {
            "model": base["models"]["qwen3.8-27b-base"],
            "model_id": "qwen3.8-27b-base",
            "serving_block": "base",
            "route": base_route,
        },
        "candidate": {
            "model": {
                "repository": "/mnt/sfs/jobs/chris-q38-t3k32-b8-v3/hf-export-step1000-v1",
                "revision": (
                    "sha256:023c5f8b0559ba050f0d672a6bc27aabecec7d5837595f8ea5bc914446d26db5"
                ),
                "session_model": "qwen/chris-q38-t3k32-s1000-v1",
            },
            "model_id": "teacher3k32-step1000",
            "serving_block": "teacher3k32",
            "route": candidate_route,
        },
    }
    replacement_evidence = {
        (row["replacement_seed"], row["arm_id"]): row for row in migration["replacement_arms"]
    }
    original_by_replacement = {
        row["replacement_seed"]: row["invalid_original_seed"]
        for row in comparison["replacement_mapping"]
    }
    protocols = {row["seed"]: row for row in comparison["replica_protocols"]}
    replicas = []
    for seed in comparison["included_seeds"]:
        for arm in ARMS:
            replicas.append(
                _replica_descriptor(
                    seed=seed,
                    arm=arm,
                    origin=protocols[seed]["origin"],
                    protocol=protocols[seed],
                    replacement_evidence=replacement_evidence,
                    original_by_replacement=original_by_replacement,
                )
            )
    plan = {
        "schema": PLAN_SCHEMA,
        "study_id": comparison["protocol_study_id"],
        "benchmark": BENCHMARK,
        "comparison_definition": comparison,
        "comparison_definition_sha256": comparison["sha256"],
        "comparison_definition_file_sha256": migration["comparison_definition_file_sha256"],
        "migration_receipt_sha256": migration["sha256"],
        "migration_receipt_file_sha256": _file_digest(migration_receipt_path),
        "source_seeds": list(SOURCE_SEEDS),
        "excluded_original_seeds": comparison["excluded_original_seeds"],
        "replacement_mapping": comparison["replacement_mapping"],
        "included_seeds": comparison["included_seeds"],
        "excluded_source_evidence": migration["migrations"],
        "task_set_file_sha256": _file_digest(task_set_path),
        "task_selection_sha256": (
            "sha256:38ea6686afa068c19e69fea2493e01027fdf99f7e01b20376e107b1a0cfa0b68"
        ),
        "binding_roster_file_sha256": _file_digest(roster_path),
        "binding_roster_sha256": roster["sha256"],
        "bindings_sha256": roster["bindings_sha256"],
        "tasks": tasks,
        "seeds": comparison["included_seeds"],
        "pass_k": PASS_K,
        "replica_pass_k": 1,
        "retry_limit": 0,
        "harness": harness,
        "images": images,
        "sampling": {"temperature": 0.6, "top_p": 0.95},
        "arms": arms,
        "replicas": replicas,
        "public_output_schema": PUBLIC_SCHEMA,
        "private_output_root": "/mnt/sfs/jobs/chris-q38-dev17-pass8-final-v2",
    }
    return {**plan, "sha256": _digest(plan)}


def validate_plan(plan: dict[str, Any]) -> None:
    if plan.get("schema") != PLAN_SCHEMA:
        raise FinalAggregateError("final aggregate plan schema differs")
    _require_self_digest(plan, "sha256", "final aggregate plan")
    comparison = plan.get("comparison_definition")
    if not isinstance(comparison, dict):
        raise FinalAggregateError("final aggregate comparison definition is missing")
    mapping, included, protocol_by_seed = _comparison_contract(comparison)
    if any(
        (
            plan.get("study_id") != "q38-dev17-base-step1000-p8-v2",
            plan.get("benchmark") != BENCHMARK,
            plan.get("comparison_definition_sha256") != comparison["sha256"],
            plan.get("source_seeds") != list(SOURCE_SEEDS),
            plan.get("included_seeds") != included,
            plan.get("seeds") != included,
            plan.get("excluded_original_seeds") != comparison["excluded_original_seeds"],
            plan.get("replacement_mapping") != mapping,
            plan.get("pass_k") != PASS_K,
            plan.get("replica_pass_k") != 1,
            plan.get("retry_limit") != 0,
            plan.get("public_output_schema") != PUBLIC_SCHEMA,
            plan.get("task_set_file_sha256")
            != "sha256:79c834e739246da29aca9513965ecfc7032f8df7744eb2245ce0303aba1b97c5",
            plan.get("task_selection_sha256")
            != "sha256:38ea6686afa068c19e69fea2493e01027fdf99f7e01b20376e107b1a0cfa0b68",
            plan.get("binding_roster_file_sha256")
            != "sha256:9d6fc1e4780e49f5054ac44dfabe13310a9b387a8361e4c01f09a9870c76aed3",
            plan.get("binding_roster_sha256")
            != "sha256:39ae49c2db322725800b15347d5da5d171e7e03b21dec5b570683414d70e83c5",
            plan.get("bindings_sha256")
            != "sha256:32230df8759bb74eb306d16cd9e32686618b529aa7c0b0ac1fa1d751ecd38728",
        )
    ):
        raise FinalAggregateError("final aggregate study identity differs")
    _exact_digest_fields(
        plan,
        (
            "comparison_definition_file_sha256",
            "migration_receipt_sha256",
            "migration_receipt_file_sha256",
        ),
        "final aggregate plan",
    )
    if any(
        _digest(plan.get(field)) != digest for field, digest in EXACT_PLAN_FIELD_DIGESTS.items()
    ):
        raise FinalAggregateError(
            "final aggregate immutable model, task, or protocol binding differs"
        )
    tasks = plan.get("tasks")
    replicas = plan.get("replicas")
    arms = plan.get("arms")
    if (
        not isinstance(tasks, list)
        or len(tasks) != TASK_COUNT
        or len({row.get("task_version_id") for row in tasks if isinstance(row, dict)}) != TASK_COUNT
        or not isinstance(replicas, list)
        or len(replicas) != PASS_K * len(ARMS)
        or not isinstance(arms, dict)
        or set(arms) != set(ARMS)
    ):
        raise FinalAggregateError("final aggregate roster is incomplete")
    identities = {(row.get("seed"), row.get("arm")) for row in replicas if isinstance(row, dict)}
    if identities != {(seed, arm) for seed in included for arm in ARMS}:
        raise FinalAggregateError("final aggregate does not contain eight included pairs once")
    original_by_replacement = {
        row["replacement_seed"]: row["invalid_original_seed"] for row in mapping
    }
    replacement_evidence: dict[tuple[int, str], dict[str, str]] = {}
    for replica in replicas:
        if not isinstance(replica, dict):
            raise FinalAggregateError("final aggregate replica identity differs")
        seed = replica["seed"]
        arm = replica["arm"]
        replacement = seed in original_by_replacement
        if replacement:
            _normalized_digest(
                replica.get("evaluation_identity_sha256"),
                "replacement evaluation identity",
            )
            replacement_evidence[(seed, arm)] = {
                "evaluation_identity_sha256": replica["evaluation_identity_sha256"]
            }
        if (
            replica.get("origin") != protocol_by_seed[seed]["origin"]
            or replica.get("protocol_id") != protocol_by_seed[seed]["protocol_id"]
            or replica.get("comparison_protocol_sha256")
            != protocol_by_seed[seed]["comparison_protocol_sha256"]
            or (not replacement and replica.get("evaluation_identity_sha256") is not None)
        ):
            raise FinalAggregateError("final aggregate replica protocol identity differs")
    for replica in replicas:
        seed = replica["seed"]
        arm = replica["arm"]
        expected = _replica_descriptor(
            seed=seed,
            arm=arm,
            origin=protocol_by_seed[seed]["origin"],
            protocol=protocol_by_seed[seed],
            replacement_evidence=replacement_evidence,
            original_by_replacement=original_by_replacement,
        )
        output_root = replica.get("output_root")
        if not isinstance(output_root, str) or not output_root.startswith("/"):
            raise FinalAggregateError("final aggregate replica output root is invalid")
        expected["output_root"] = output_root
        expected["terminal_receipt_path"] = f"{output_root}/TERMINAL_OBSERVATION.json"
        if replica != expected:
            raise FinalAggregateError("final aggregate replica identity differs")
    excluded_source_evidence = plan.get("excluded_source_evidence")
    if not isinstance(excluded_source_evidence, list) or len(excluded_source_evidence) != len(
        mapping
    ):
        raise FinalAggregateError("final aggregate excluded source evidence is incomplete")
    for expected, evidence in zip(mapping, excluded_source_evidence, strict=True):
        if (
            not isinstance(evidence, dict)
            or evidence.get("invalid_original_seed") != expected["invalid_original_seed"]
            or evidence.get("replacement_seed") != expected["replacement_seed"]
            or evidence.get("whole_pair_excluded") is not True
            or evidence.get("reason_class") not in ALLOWED_EXCLUSION_REASONS
        ):
            raise FinalAggregateError("final aggregate excluded source evidence differs")
    if plan.get("harness", {}).get("harness_version") != "1.18.27":
        raise FinalAggregateError("final aggregate harness identity differs")


@dataclass(frozen=True)
class GateSnapshot:
    plan_sha256: str
    cells: list[dict[str, Any]]
    local_metadata: list[dict[str, Any]]
    events: list[dict[str, Any]]
    reconciliations: list[dict[str, Any]]


class Snapshot(Protocol):
    def gate(self) -> GateSnapshot: ...

    def scored_results(self) -> list[dict[str, Any]]: ...


def _evaluation_plan(replica: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(str(replica["output_root"])) / "EVAL.json"
    value = _read_json(path, f"seed {replica['seed']} {replica['arm']} EVAL")
    claimed = value.get("sha256")
    if not isinstance(claimed, str) or not re.fullmatch(r"[0-9a-f]{64}", claimed):
        raise FinalAggregateError("evaluation plan digest is invalid")
    if _plain_digest({key: item for key, item in value.items() if key != "sha256"}) != claimed:
        raise FinalAggregateError("evaluation plan self digest differs")
    runtime = value.get("runtime_files")
    if (
        not isinstance(runtime, dict)
        or set(runtime) != RUNTIME_FILES
        or any(
            not isinstance(item, str) or re.fullmatch(r"[0-9a-f]{64}", item) is None
            for item in runtime.values()
        )
    ):
        raise FinalAggregateError("evaluation runtime identity is incomplete")
    return value


def _terminal_receipt(replica: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    path = Path(str(replica["terminal_receipt_path"]))
    receipt = _read_json(path, f"seed {replica['seed']} {replica['arm']} terminal receipt")
    self_digest = _require_self_digest(receipt, "sha256", "held-out terminal receipt")
    job = receipt.get("job", {})
    terminal_condition = job.get("terminal_condition")
    terminal_counts_match = (
        terminal_condition == "Complete" and job.get("succeeded") == 1 and job.get("failed") == 0
    ) or (terminal_condition == "Failed" and job.get("succeeded") == 0 and job.get("failed") == 1)
    if any(
        (
            receipt.get("schema") != TERMINAL_SCHEMA,
            receipt.get("protocol_id") != replica["protocol_id"],
            receipt.get("comparison_protocol_sha256") != replica["comparison_protocol_sha256"],
            replica.get("evaluation_identity_sha256") is not None
            and receipt.get("evaluation_identity_sha256") != replica["evaluation_identity_sha256"],
            receipt.get("arm_id") != replica["arm"],
            job.get("name") != replica["job_name"],
            not isinstance(job.get("uid"), str),
            isinstance(job.get("uid"), str) and UUID.fullmatch(job["uid"]) is None,
            receipt.get("config_map", {}).get("name") != replica["config_map_name"],
            not isinstance(receipt.get("config_map", {}).get("uid"), str),
            isinstance(receipt.get("config_map", {}).get("uid"), str)
            and UUID.fullmatch(receipt["config_map"]["uid"]) is None,
            not terminal_counts_match,
            receipt.get("database", {}).get("name") != replica["database"],
            receipt.get("output_root", {}).get("path") != replica["output_root"],
            receipt.get("decision", {}).get("score_read_or_generated") is not False,
            receipt.get("privacy", {}).get(
                "prompts_responses_flags_rewards_or_trace_content_included"
            )
            is not False,
            receipt.get("privacy", {}).get("score_values_included") is not False,
            receipt.get("privacy", {}).get("credentials_included") is not False,
        )
    ):
        raise FinalAggregateError("held-out terminal receipt identity differs")
    return receipt, self_digest


def _expected_task_map(plan: Mapping[str, Any]) -> dict[str, str]:
    return {row["task_version_id"]: row["task_key"] for row in plan["tasks"]}


def _event_detail(row: Mapping[str, Any]) -> dict[str, Any]:
    value = row.get("detail_json")
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise FinalAggregateError("ledger accepted event detail is invalid") from exc
    if not isinstance(value, dict):
        raise FinalAggregateError("ledger accepted event detail is invalid")
    return value


def _validate_acceptance_evidence(
    cell: Mapping[str, Any], events: Sequence[Mapping[str, Any]], reconciliations: Sequence[dict]
) -> None:
    cell_events = [row for row in events if row.get("cell_id") == cell["cell_id"]]
    if len(cell_events) != 1 or cell_events[0].get("to_state") != "accepted":
        raise FinalAggregateError("accepted cell lacks one exact acceptance event")
    detail = _event_detail(cell_events[0])
    receipt = _normalized_digest(cell.get("receipt_digest"), "accepted receipt")
    event_receipt = detail.get("receipt_digest", detail.get("cell_receipt_sha256"))
    if _normalized_digest(event_receipt, "accepted event receipt") != receipt:
        raise FinalAggregateError("accepted event receipt differs from ledger cell")
    reconciliation = cell.get("reconciliation_digest")
    if reconciliation is None:
        if cell_events[0].get("event") != "accepted":
            raise FinalAggregateError("unreconciled acceptance event kind differs")
        return
    reconciliation = _normalized_digest(reconciliation, "reconciliation digest")
    event_intent = detail.get("reviewed_intent_sha256", detail.get("reconciliation_digest"))
    if _normalized_digest(event_intent, "accepted reconciliation event") != reconciliation:
        raise FinalAggregateError("accepted reconciliation event differs")
    matches = []
    for value in reconciliations:
        if not isinstance(value, dict):
            continue
        reviewed = value.get("reviewed_intent_sha256")
        if (
            isinstance(reviewed, str)
            and SHA256.fullmatch(reviewed)
            and reviewed.removeprefix("sha256:") == reconciliation
        ):
            matches.append(value)
    if len(matches) != 1:
        raise FinalAggregateError("accepted reconciliation receipt is missing or ambiguous")
    evidence = matches[0]
    _require_self_digest(evidence, "receipt_sha256", "accepted reconciliation receipt")
    if any(
        (
            evidence.get("schema_version")
            not in {
                "fleet-stored-session-reconciliation-v1",
                "fleet-stored-session-reconciliation-v2",
            },
            evidence.get("accepted_existing_completed_session_count", 0) < 1,
            evidence.get("model_generation_performed") is not False,
            evidence.get("scoring_call_performed") is not False,
            evidence.get("score_values_included") is not False,
        )
    ):
        raise FinalAggregateError("accepted reconciliation receipt is invalid")


def _validate_eval_identity(
    plan: Mapping[str, Any], replica: Mapping[str, Any], evaluation: Mapping[str, Any]
) -> None:
    arm = plan["arms"][replica["arm"]]
    expected_tasks = sorted(plan["tasks"], key=lambda row: row["task_version_id"])
    if any(
        (
            evaluation.get("schema") != "cyber_fleet_eval_v1",
            evaluation.get("campaign_id") != replica["experiment_id"],
            evaluation.get("tasks") != expected_tasks,
            evaluation.get("models") != {arm["model_id"]: arm["model"]},
            evaluation.get("routes") != {arm["serving_block"]: arm["route"]},
            evaluation.get("treatment") != plan["harness"],
            evaluation.get("images") != plan["images"],
            evaluation.get("sampling") != {**plan["sampling"], "seed": replica["seed"]},
            evaluation.get("pass_k") != 1,
            evaluation.get("automatic_retry") is not False,
            evaluation.get("max_reviewed_infrastructure_retries") != 0,
            evaluation.get("training_data_eligible") is not False,
        )
    ):
        raise FinalAggregateError("evaluation plan model, task, harness, or protocol differs")


def validate_gate(
    plan: Mapping[str, Any],
    replica: Mapping[str, Any],
    snapshot: GateSnapshot,
    evaluation: Mapping[str, Any],
    terminal: Mapping[str, Any],
) -> None:
    """Validate one replica without reading its score column."""

    _validate_eval_identity(plan, replica, evaluation)
    arm = plan["arms"][replica["arm"]]
    tasks = _expected_task_map(plan)
    expected_harness = "protocol-" + str(evaluation["sha256"])
    if snapshot.plan_sha256 != terminal.get("database", {}).get("summary", {}).get("plan_sha256"):
        raise FinalAggregateError("terminal and live database plan identities differ")
    cells = snapshot.cells
    if len(cells) != TASK_COUNT or {row.get("task_version_id") for row in cells} != set(tasks):
        raise FinalAggregateError("replica ledger does not contain the exact dev17 roster")
    for cell in cells:
        version = cell["task_version_id"]
        if any(
            (
                cell.get("experiment_id") != replica["experiment_id"],
                cell.get("task_key") != tasks[version],
                cell.get("model_id") != arm["model_id"],
                cell.get("model_revision") != arm["model"]["revision"],
                cell.get("serving_block") != arm["serving_block"],
                cell.get("endpoint_model_id") != arm["route"]["served_id"],
                cell.get("harness_id") != expected_harness,
                cell.get("attempt") != 1,
                cell.get("state") != "accepted",
                cell.get("result_class") != "valid",
                cell.get("lease_expires_at") is not None,
                cell.get("retry_count") != 0,
                cell.get("max_retries") != 0,
                cell.get("failure_code") is not None,
                not isinstance(cell.get("session_id"), str),
            )
        ):
            raise FinalAggregateError("replica contains a non-authoritative or mismatched cell")
        _normalized_digest(cell.get("receipt_digest"), "accepted receipt")
        if cell.get("cell_id") != _cell_id(cell):
            raise FinalAggregateError("replica cell identity differs from its immutable plan row")
        _validate_acceptance_evidence(cell, snapshot.events, snapshot.reconciliations)
    stored_plan = [
        {field: row.get(field) for field in PLAN_STORED_FIELDS}
        for row in sorted(
            cells,
            key=lambda row: (
                row["experiment_id"],
                row["task_version_id"],
                row["model_id"],
                row["attempt"],
            ),
        )
    ]
    if _plain_digest(stored_plan) != snapshot.plan_sha256:
        raise FinalAggregateError("replica immutable ledger plan digest differs")
    metadata = snapshot.local_metadata
    counts = Counter(row.get("cell_id") for row in metadata)
    if (
        len(metadata) != TASK_COUNT
        or set(counts) != {row["cell_id"] for row in cells}
        or any(count != 1 for count in counts.values())
    ):
        raise FinalAggregateError("accepted cells do not each have one private local result")
    cells_by_id = {row["cell_id"]: row for row in cells}
    for row in metadata:
        if (
            row.get("session_ingest_status") != "completed"
            or row.get("session_id") != cells_by_id[row["cell_id"]]["session_id"]
        ):
            raise FinalAggregateError("accepted local result lacks completed session ingestion")


def _validate_scored_results(
    snapshot: GateSnapshot, scored: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    if len(scored) != TASK_COUNT:
        raise FinalAggregateError("score opening did not return exactly seventeen rows")
    cells = {row["cell_id"]: row for row in snapshot.cells}
    if {row.get("cell_id") for row in scored} != set(cells):
        raise FinalAggregateError("opened scores differ from the accepted cell roster")
    metadata_fields = tuple(field for field in LOCAL_RECORD_FIELDS if field != "score") + (
        "record_sha256",
    )
    expected_metadata = sorted(
        ({field: row.get(field) for field in metadata_fields} for row in snapshot.local_metadata),
        key=lambda row: row["cell_id"],
    )
    opened_metadata = sorted(
        ({field: row.get(field) for field in metadata_fields} for row in scored),
        key=lambda row: row["cell_id"],
    )
    if opened_metadata != expected_metadata:
        raise FinalAggregateError(
            "opened private result metadata differs from the score-blind gate"
        )
    output = []
    for row in scored:
        normalized = {field: row.get(field) for field in LOCAL_RECORD_FIELDS}
        score = normalized["score"]
        elapsed = normalized["elapsed_seconds"]
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
            or not 0.0 <= float(score) <= 1.0
            or isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not math.isfinite(float(elapsed))
            or float(elapsed) < 0
        ):
            raise FinalAggregateError("opened private result has an invalid numeric value")
        for field in (
            "config_sha256",
            "trace_sha256",
            "result_sha256",
            "reward_sha256",
            "session_ingest_sha256",
            "cleanup_sha256",
        ):
            normalized[field] = _normalized_digest(normalized[field], field)
        if _plain_digest(normalized) != _normalized_digest(row.get("record_sha256"), "record"):
            raise FinalAggregateError("opened private result record digest differs")
        if normalized["session_id"] != cells[normalized["cell_id"]]["session_id"]:
            raise FinalAggregateError("opened private result session differs")
        output.append({**normalized, "record_sha256": row["record_sha256"]})
    return output


def finalize(
    plan: dict[str, Any],
    snapshots: Mapping[tuple[int, str], Snapshot],
    *,
    output_root: Path,
) -> dict[str, Any]:
    """Gate all replicas, then and only then open scores and publish receipts."""

    validate_plan(plan)
    if output_root.as_posix() != plan["private_output_root"]:
        raise FinalAggregateError("final output root differs from the sealed plan")
    if output_root.exists() or output_root.is_symlink():
        raise FinalAggregateError("final aggregate output already exists")
    if not output_root.parent.is_dir() or output_root.parent.is_symlink():
        raise FinalAggregateError("final aggregate parent is not an exact directory")
    replicas = {(row["seed"], row["arm"]): row for row in plan["replicas"]}
    if set(snapshots) != set(replicas):
        raise FinalAggregateError("database snapshots differ from the sealed replica roster")

    gated: dict[tuple[int, str], tuple[GateSnapshot, dict, dict, str]] = {}
    common_runtime: dict[str, str] | None = None
    protocol_by_seed: dict[int, dict[str, str]] = defaultdict(dict)
    for identity in sorted(replicas):
        replica = replicas[identity]
        evaluation = _evaluation_plan(replica)
        runtime = evaluation["runtime_files"]
        if common_runtime is None:
            common_runtime = runtime
        elif runtime != common_runtime:
            raise FinalAggregateError("base and candidate evaluator runtime files differ")
        terminal, terminal_sha = _terminal_receipt(replica)
        protocol_by_seed[replica["seed"]][replica["arm"]] = terminal.get(
            "comparison_protocol_sha256"
        )
        snapshot = snapshots[identity].gate()
        validate_gate(plan, replica, snapshot, evaluation, terminal)
        gated[identity] = (snapshot, evaluation, terminal, terminal_sha)
    if any(
        set(values) != set(ARMS) or values["base"] != values["candidate"]
        for values in protocol_by_seed.values()
    ):
        raise FinalAggregateError("matched base and candidate protocol identity differs")

    # This is the sole score-opening boundary. No scored_results() call occurs
    # until all sixteen terminal, identity, state, and evidence gates above pass.
    outcomes: list[dict[str, Any]] = []
    for identity in sorted(replicas):
        replica = replicas[identity]
        snapshot, evaluation, _terminal, _terminal_sha = gated[identity]
        for row in _validate_scored_results(snapshot, snapshots[identity].scored_results()):
            cell = next(item for item in snapshot.cells if item["cell_id"] == row["cell_id"])
            outcomes.append(
                {
                    "seed": replica["seed"],
                    "arm": replica["arm"],
                    "task_key": cell["task_key"],
                    "task_version_id": cell["task_version_id"],
                    "score": float(row["score"]),
                    "record_sha256": "sha256:" + _normalized_digest(row["record_sha256"], "record"),
                    "accepted_receipt_sha256": "sha256:"
                    + _normalized_digest(cell["receipt_digest"], "accepted receipt"),
                    "evaluation_plan_sha256": "sha256:" + evaluation["sha256"],
                }
            )
    coverage = Counter((row["task_version_id"], row["arm"], row["seed"]) for row in outcomes)
    if len(outcomes) != TASK_COUNT * len(ARMS) * PASS_K or any(
        coverage[(task["task_version_id"], arm, seed)] != 1
        for task in plan["tasks"]
        for arm in ARMS
        for seed in plan["included_seeds"]
    ):
        raise FinalAggregateError("opened outcomes do not cover each task, arm, and seed once")

    permutation = list(range(TASK_COUNT))
    secrets.SystemRandom().shuffle(permutation)
    anonymization_body = {
        "schema": "cyber_private_task_anonymization_v1",
        "method": "private_random_permutation_v1",
        "rows": [
            {
                "public_task_index": public_index,
                "task_key": plan["tasks"][source_index]["task_key"],
                "task_version_id": plan["tasks"][source_index]["task_version_id"],
            }
            for public_index, source_index in enumerate(permutation)
        ],
    }
    anonymization = {
        **anonymization_body,
        "receipt_sha256": _digest(anonymization_body),
    }
    terminal_body = {
        "schema": "cyber_fleet_matched_pass8_terminal_index_v2",
        "study_plan_sha256": plan["sha256"],
        "protocol_v2_migration": {
            "migration_receipt_sha256": plan["migration_receipt_sha256"],
            "migration_receipt_file_sha256": plan["migration_receipt_file_sha256"],
            "comparison_definition_sha256": plan["comparison_definition_sha256"],
            "comparison_definition_file_sha256": plan["comparison_definition_file_sha256"],
            "source_seeds": plan["source_seeds"],
            "excluded_original_seeds": plan["excluded_original_seeds"],
            "replacement_mapping": plan["replacement_mapping"],
            "included_seeds": plan["included_seeds"],
            "excluded_source_evidence": plan["excluded_source_evidence"],
        },
        "rows": [
            {
                "seed": replica["seed"],
                "arm": replica["arm"],
                "origin": replica["origin"],
                "replaces_seed": replica["replaces_seed"],
                "evaluation_identity_sha256": terminal["evaluation_identity_sha256"],
                "comparison_protocol_sha256": terminal["comparison_protocol_sha256"],
                "job_uid": terminal["job"]["uid"],
                "terminal_receipt_file_sha256": _file_digest(
                    Path(replica["terminal_receipt_path"])
                ),
                "terminal_receipt_sha256": terminal_sha,
                "database_plan_sha256": snapshot.plan_sha256,
            }
            for identity, replica in sorted(replicas.items())
            for snapshot, _evaluation, terminal, terminal_sha in [gated[identity]]
        ],
    }
    terminal_index = {**terminal_body, "receipt_sha256": _digest(terminal_body)}
    outcome_body = {
        "schema": "cyber_fleet_matched_pass8_protocol_v2_private_scored_outcome_index_v1",
        "study_plan_sha256": plan["sha256"],
        "outcome_count": len(outcomes),
        "rows": sorted(
            outcomes,
            key=lambda row: (row["task_version_id"], row["arm"], row["seed"]),
        ),
    }
    outcome_index = {**outcome_body, "receipt_sha256": _digest(outcome_body)}
    outcomes_by_task = defaultdict(list)
    for row in outcomes:
        outcomes_by_task[row["task_version_id"]].append(row)
    public_rows = []
    for source_index in permutation:
        version = plan["tasks"][source_index]["task_version_id"]
        task_outcomes = outcomes_by_task[version]
        row = {"metric_denominator": 1}
        for arm in ARMS:
            scores = [item["score"] for item in task_outcomes if item["arm"] == arm]
            if len(scores) != PASS_K:
                raise FinalAggregateError("anonymized task row is missing an arm attempt")
            row[arm] = {
                "valid_attempts": PASS_K,
                "infrastructure_invalid_attempts": 0,
                "metric_numerator": int(any(score == 1.0 for score in scores)),
            }
        public_rows.append(row)
    public_body = {
        "schema_version": PUBLIC_SCHEMA,
        "status": "final",
        "benchmark": BENCHMARK,
        "comparison_definition_sha256": plan["comparison_definition_sha256"],
        "anonymization_method": "private_random_permutation_v1",
        "anonymization_receipt_sha256": anonymization["receipt_sha256"],
        "task_rows": public_rows,
        "evidence": {
            "terminal_index_sha256": terminal_index["receipt_sha256"],
            "scored_outcome_index_sha256": outcome_index["receipt_sha256"],
            "terminal_schemas": {
                "launch_terminal": TERMINAL_SCHEMA,
                "accepted_cell": ACCEPTED_SCHEMA,
                "controller_terminal": CONTROLLER_TERMINAL_SCHEMA,
            },
        },
    }
    public = {**public_body, "receipt_sha256": _digest(public_body)}
    final_body = {
        "schema": "cyber_fleet_matched_pass8_protocol_v2_final_receipt_v1",
        "study_plan_sha256": plan["sha256"],
        "status": "final",
        "task_count": TASK_COUNT,
        "valid_outcomes_per_task_arm": PASS_K,
        "private_terminal_index_sha256": terminal_index["receipt_sha256"],
        "private_scored_outcome_index_sha256": outcome_index["receipt_sha256"],
        "private_anonymization_receipt_sha256": anonymization["receipt_sha256"],
        "sanitized_aggregate_receipt_sha256": public["receipt_sha256"],
        "comparison_definition_sha256": plan["comparison_definition_sha256"],
        "migration_receipt_sha256": plan["migration_receipt_sha256"],
        "included_seeds": plan["included_seeds"],
        "excluded_original_seeds": plan["excluded_original_seeds"],
        "public_contains_private_identifiers": False,
        "prompts_responses_flags_or_traces_read": False,
    }
    final = {**final_body, "receipt_sha256": _digest(final_body)}

    temporary = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent))
    os.chmod(temporary, 0o700)
    try:
        for name, value in (
            ("PRIVATE_TERMINAL_INDEX.json", terminal_index),
            ("PRIVATE_SCORED_OUTCOME_INDEX.json", outcome_index),
            ("PRIVATE_ANONYMIZATION.json", anonymization),
            ("SANITIZED_AGGREGATE.json", public),
            ("FINAL.json", final),
        ):
            _write_exclusive(temporary / name, value)
        os.rename(temporary, output_root)
    except BaseException:
        # The final path is never exposed partially. A failed temporary tree is
        # retained for private diagnosis and cannot be mistaken for acceptance.
        raise
    return final


class _PostgresSnapshot:
    def __init__(self, connection: Any) -> None:
        self.connection = connection
        self.score_opened = False

    def gate(self) -> GateSnapshot:
        plan = self.connection.execute(
            "SELECT value FROM ledger_metadata WHERE key = 'plan_sha256'"
        ).fetchall()
        if len(plan) != 1:
            raise FinalAggregateError("replica database plan identity is missing")
        cells = self.connection.execute(
            f"SELECT {', '.join(CELL_FIELDS)} FROM rollout_cells ORDER BY cell_id"
        ).fetchall()
        metadata_fields = tuple(field for field in LOCAL_RECORD_FIELDS if field != "score") + (
            "record_sha256",
        )
        metadata = self.connection.execute(
            f"SELECT {', '.join(metadata_fields)} FROM rollout_local_results ORDER BY cell_id"
        ).fetchall()
        events = self.connection.execute(
            """
            SELECT cell_id, event, to_state, detail_json
            FROM rollout_events WHERE to_state = 'accepted' ORDER BY sequence
            """
        ).fetchall()
        reconciliation_rows = self.connection.execute(
            "SELECT receipt_json FROM ledger_reconciliations ORDER BY receipt_sha256"
        ).fetchall()
        reconciliations = []
        for row in reconciliation_rows:
            value = row["receipt_json"]
            if isinstance(value, str):
                value = json.loads(value)
            if not isinstance(value, dict):
                raise FinalAggregateError("database reconciliation receipt is invalid")
            reconciliations.append(value)
        return GateSnapshot(
            plan_sha256=plan[0]["value"],
            cells=[dict(row) for row in cells],
            local_metadata=[dict(row) for row in metadata],
            events=[dict(row) for row in events],
            reconciliations=reconciliations,
        )

    def scored_results(self) -> list[dict[str, Any]]:
        self.score_opened = True
        rows = self.connection.execute(
            f"SELECT {', '.join(LOCAL_RECORD_FIELDS)}, record_sha256 "
            "FROM rollout_local_results ORDER BY cell_id"
        ).fetchall()
        return [dict(row) for row in rows]


def _database_dsn(root_dsn: str, database: str) -> str:
    parsed = urlsplit(root_dsn)
    query = {key.casefold() for key, _value in parse_qsl(parsed.query, keep_blank_values=True)}
    if (
        parsed.scheme not in {"postgres", "postgresql"}
        or not parsed.netloc
        or parsed.fragment
        or query & {"database", "dbname"}
        or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,62}", database)
    ):
        raise FinalAggregateError("PostgreSQL root connection or database name is invalid")
    return urlunsplit(
        (parsed.scheme, parsed.netloc, "/" + quote(database, safe=""), parsed.query, "")
    )


@contextmanager
def open_postgres_snapshots(
    plan: Mapping[str, Any], root_dsn: str
) -> Iterator[dict[tuple[int, str], Snapshot]]:
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError as exc:
        raise FinalAggregateError("psycopg is required for the private final gate") from exc
    with ExitStack() as stack:
        snapshots: dict[tuple[int, str], Snapshot] = {}
        for replica in plan["replicas"]:
            connection = stack.enter_context(
                psycopg.connect(
                    _database_dsn(root_dsn, replica["database"]),
                    autocommit=False,
                    row_factory=dict_row,
                    connect_timeout=30,
                    application_name="fleet-dev17-pass8-final-v2",
                )
            )
            connection.execute("BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            snapshots[(replica["seed"], replica["arm"])] = _PostgresSnapshot(connection)
        yield snapshots


def run_from_environment(plan_path: Path) -> dict[str, Any]:
    plan = _read_json(plan_path, "bundled final aggregate plan")
    expected = os.environ.get("FINAL_STUDY_PLAN_FILE_SHA256")
    if expected != _file_digest(plan_path):
        raise FinalAggregateError("bundled final aggregate plan bytes differ")
    validate_plan(plan)
    dsn = os.environ.get("ROLLOUT_DATABASE_URL")
    if not dsn:
        raise FinalAggregateError("PostgreSQL root connection is missing")
    output = Path(os.environ.get("FINAL_OUTPUT_ROOT", ""))
    if output.as_posix() != plan["private_output_root"]:
        raise FinalAggregateError("final output environment differs from the sealed plan")
    with open_postgres_snapshots(plan, dsn) as snapshots:
        return finalize(plan, snapshots, output_root=output)
