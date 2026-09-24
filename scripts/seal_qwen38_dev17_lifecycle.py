#!/usr/bin/env python3
"""Seal score-blind dev17 lifecycle evidence and a fresh heldout20 seed plan.

The database query is deliberately allowlisted: it never selects the score
column or trace contents.  Private task/session identities stay in the private
output; the checked-in receipt contains only counts and cryptographic digests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

import psycopg
from psycopg import sql
from psycopg.rows import dict_row

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_SCHEMA = "cyber_qwen38_dev17_private_lifecycle_map_v1"
SOURCE_SCHEMA = "cyber_qwen38_dev17_private_database_sources_v1"
SEED_PLAN_SCHEMA = "cyber_qwen38_teacher3k_heldout20_pass4_seed_plan_v1"
RECEIPT_SCHEMA = "cyber_qwen38_dev17_lifecycle_and_heldout20_seed_receipt_v1"
CORRECTED_PROTOCOL_SHA256 = (
    "sha256:2ac58c3ded7ba5c9bacd27dc3a379c726912a3f0cafc3ee542672de730e1fe0e"
)
CORRECTED_PROTOCOL_SOURCE_COMMIT = "1117c8c19b7eaeb15911e2f2ab270e46892510d0"
CORRECTED_PROTOCOL_FILE_SHA256 = (
    "sha256:a2a0389494fa1bca26911081c1d5f55d235cc52c74c15dcd0d98355b7a4dd106"
)
SUPERSESSION_PATH = (
    ROOT / "docs/evidence/qwen38-dev17-pass4-outcome-validity-supersession-20260924.json"
)
SEED_DERIVATION_MATERIAL = "qwen38-heldout20-fresh-base-vs-step1000-pass4-20260924-v1"
SEED_DERIVATION_SHA256 = "sha256:3a23a5bc003c821ab026fc95d3682352a79db8dc05852a74948a3d2bddd657c7"
INITIAL_SEEDS = (975414717, 3965467, 807861398, 1399333715)
ARMS = ("base", "step1000")

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
    "started_at",
    "completed_at",
    "retry_count",
    "max_retries",
    "result_class",
    "receipt_digest",
    "failure_code",
    "reconciliation_digest",
)
RESULT_FIELDS = (
    "execution_id",
    "cell_id",
    "execution_generation",
    "run_id",
    "session_id",
    "verifier_execution_id",
    "config_sha256",
    "artifact_directory",
    "trace_path",
    "trace_sha256",
    "session_ingest_status",
    "agent_exit_code",
    "agent_termination",
    "elapsed_seconds",
    "recorded_at",
)


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=_json_default
    ).encode()


def _json_default(value: Any) -> str:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    raise TypeError(f"unsupported value for canonical JSON: {type(value).__name__}")


def digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical(value)).hexdigest()


def file_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def logical_digest(value: dict[str, Any]) -> str:
    return digest({key: item for key, item in value.items() if key != "sha256"})


def repo_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return path.name


def read_verified(path: Path, schema: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != schema:
        raise ValueError(f"{path} has the wrong schema")
    if value.get("sha256") != logical_digest(value):
        raise ValueError(f"{path} has an invalid self digest")
    return value


def write_sealed(path: Path, value: dict[str, Any], *, private: bool = False) -> None:
    output = {**value, "sha256": logical_digest(value)}
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "x" if private else "w"
    with path.open(mode, encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                output,
                indent=2,
                sort_keys=True,
                allow_nan=False,
                default=_json_default,
            )
            + "\n"
        )
    if private:
        path.chmod(0o600)


def classify_lifecycle(exit_code: object, termination: object) -> str:
    if type(exit_code) is int and exit_code == 0 and termination == "completed":
        return "normal_completed"
    if type(exit_code) is int and exit_code == 0 and termination == "output_limit":
        return "held_output_limit"
    if termination == "process_error":
        return "infrastructure_invalid_process_error"
    return "infrastructure_invalid_other"


def _database_url(template: str, database: str) -> str:
    parsed = urlparse(template)
    if parsed.scheme not in {"postgres", "postgresql"} or not parsed.netloc:
        raise ValueError("database URL must be PostgreSQL")
    return urlunparse(
        (parsed.scheme, parsed.netloc, f"/{database}", parsed.params, parsed.query, parsed.fragment)
    )


def _select(cursor: psycopg.Cursor[Any], table: str, fields: tuple[str, ...]) -> list[dict]:
    query = sql.SQL("SELECT {} FROM {} ORDER BY cell_id").format(
        sql.SQL(", ").join(map(sql.Identifier, fields)), sql.Identifier(table)
    )
    cursor.execute(query)
    return [dict(row) for row in cursor.fetchall()]


def read_database(template: str, database: str) -> dict[str, Any]:
    """Read only explicitly allowlisted lifecycle evidence from one database."""

    with psycopg.connect(
        _database_url(template, database),
        row_factory=dict_row,
    ) as connection:
        connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
        with connection.cursor() as cursor:
            cursor.execute("SELECT key, value FROM ledger_metadata ORDER BY key")
            metadata = {row["key"]: row["value"] for row in cursor.fetchall()}
            cells = _select(cursor, "rollout_cells", CELL_FIELDS)
            results = _select(cursor, "rollout_local_results", RESULT_FIELDS)
    return {"ledger_metadata": metadata, "cells": cells, "results": results}


def read_database_catalog(template: str) -> list[str]:
    with psycopg.connect(template, row_factory=dict_row) as connection:
        connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
        with connection.cursor() as cursor:
            cursor.execute("SELECT datname FROM pg_database WHERE datallowconn ORDER BY datname")
            return [row["datname"] for row in cursor.fetchall()]


def seal_source(source: dict[str, Any], snapshot: dict[str, Any]) -> dict[str, Any]:
    cells, results = snapshot["cells"], snapshot["results"]
    if len(cells) != 17 or len(results) != 17:
        raise ValueError("every retained source database must contain exactly 17 cells")
    result_by_cell = {row["cell_id"]: row for row in results}
    if len(result_by_cell) != len(results) or set(result_by_cell) != {
        row["cell_id"] for row in cells
    }:
        raise ValueError("local results do not form a one-to-one join with cells")

    attempts = []
    for cell in cells:
        local = result_by_cell[cell["cell_id"]]
        if local["session_id"] != cell["session_id"]:
            raise ValueError("cell and local result session identities differ")
        outcome = classify_lifecycle(local["agent_exit_code"], local["agent_termination"])
        row = {
            "arm": source["arm"],
            "seed": source["seed"],
            "cell": cell,
            "local_result": local,
            "outcome_class": outcome,
        }
        row["database_row_sha256"] = digest(row)
        attempts.append(row)

    trace_manifest = [
        {
            "cell_id": row["cell"]["cell_id"],
            "trace_path": row["local_result"]["trace_path"],
            "trace_sha256": row["local_result"]["trace_sha256"],
        }
        for row in attempts
    ]
    evidence = {
        "source_label": source["label"],
        "arm": source["arm"],
        "seed": source["seed"],
        "database": source["database"],
        "job": source["job"],
        "output_root": source["output_root"],
        "ledger_metadata": snapshot["ledger_metadata"],
        "attempts": attempts,
        "recorded_trace_manifest_sha256": digest(trace_manifest),
        "recorded_trace_digests_rehashed_from_raw_bytes": False,
    }
    evidence["database_evidence_snapshot_sha256"] = digest(evidence)
    return evidence


def build_private_map(
    sources: dict[str, Any],
    snapshots: dict[str, dict[str, Any]],
    database_catalog: list[str],
    input_bindings: dict[str, Any],
    old_task_versions: set[str],
    observed_at: str,
) -> dict[str, Any]:
    raw_rows = sources.get("sources")
    if not isinstance(raw_rows, list) or len(raw_rows) != 8:
        raise ValueError("source manifest must name eight databases")
    if any(not isinstance(row, dict) for row in raw_rows):
        raise ValueError("source manifest rows must be objects")
    rows = sorted(raw_rows, key=lambda row: (row["seed"], row["arm"]))
    expected = {(arm, seed) for arm in ARMS for seed in (48, 54, 61, 62)}
    identities = {(row.get("arm"), row.get("seed")) for row in rows if isinstance(row, dict)}
    if identities != expected or len(identities) != len(rows):
        raise ValueError("source manifest does not exactly cover the retained dev17 roster")
    source_fields = {"arm", "database", "job", "label", "output_root", "seed"}
    job_fields = {"kind", "name", "namespace", "terminal_state", "uid"}
    if any(
        set(row) != source_fields
        or set(row["job"]) != job_fields
        or row["job"]["kind"] != "Job"
        or row["job"]["namespace"] != "fleet-train-jobs"
        or row["job"]["terminal_state"] not in {"Failed", "Complete"}
        for row in rows
    ):
        raise ValueError("source manifest contains an unsupported source binding")
    for field in ("database", "label", "output_root"):
        values = [row[field] for row in rows]
        if len(values) != len(set(values)):
            raise ValueError(f"source manifest repeats {field}")
    if len({row["job"]["uid"] for row in rows}) != 8:
        raise ValueError("source manifest repeats a Job UID")
    if set(snapshots) != {row["label"] for row in rows}:
        raise ValueError("database snapshots do not exactly match the source manifest")

    sealed_sources = [seal_source(row, snapshots[row["label"]]) for row in rows]
    if any(
        {attempt["cell"]["task_version_id"] for attempt in source["attempts"]} != old_task_versions
        for source in sealed_sources
    ):
        raise ValueError("a database task roster differs from the exact old dev17 roster")
    attempts = [attempt for source in sealed_sources for attempt in source["attempts"]]
    identities = [(row["cell"]["task_version_id"], row["arm"], row["seed"]) for row in attempts]
    if len(attempts) != 136 or len(set(identities)) != 136:
        raise ValueError("private map is not an exact 136-cell task×arm×seed roster")
    census = Counter((row["arm"], row["outcome_class"]) for row in attempts)
    catalog_seeds = sorted(
        {
            int(match.group(1))
            for name in database_catalog
            if (match := re.search(r"(?:^|_)s(?:eed)?(\d+)(?:_|$)", name))
            and ("dev17" in name or "q38" in name)
        }
    )
    if set(INITIAL_SEEDS) & set(catalog_seeds):
        raise ValueError("fresh-study seed collides with an existing Qwen database")
    return {
        "schema": PRIVATE_SCHEMA,
        "observed_at": observed_at,
        "scope": {"task_count": 17, "attempts": 136, "arms": list(ARMS)},
        "inputs": input_bindings,
        "sources": sealed_sources,
        "database_catalog": {
            "names": database_catalog,
            "catalog_sha256": digest(database_catalog),
            "qwen_or_dev17_parsed_seeds": catalog_seeds,
        },
        "lifecycle_census": {
            arm: {
                key: census[(arm, key)]
                for key in (
                    "normal_completed",
                    "held_output_limit",
                    "infrastructure_invalid_process_error",
                    "infrastructure_invalid_other",
                )
            }
            for arm in ARMS
        },
        "privacy": {
            "score_column_selected": False,
            "prompt_response_flag_or_trace_content_read": False,
            "private_identifiers_included": True,
            "must_not_commit": True,
        },
    }


def validate_private_map(value: dict[str, Any]) -> None:
    if value.get("schema") != PRIVATE_SCHEMA or value.get("sha256") != logical_digest(value):
        raise ValueError("private lifecycle map self binding differs")
    sources = value.get("sources")
    if not isinstance(sources, list) or len(sources) != 8:
        raise ValueError("private lifecycle map source roster differs")
    attempts: list[dict[str, Any]] = []
    for source in sources:
        expected_source = digest(
            {
                key: item
                for key, item in source.items()
                if key != "database_evidence_snapshot_sha256"
            }
        )
        if source.get("database_evidence_snapshot_sha256") != expected_source:
            raise ValueError("private database evidence snapshot binding differs")
        for attempt in source.get("attempts", []):
            expected_row = digest(
                {key: item for key, item in attempt.items() if key != "database_row_sha256"}
            )
            if attempt.get("database_row_sha256") != expected_row:
                raise ValueError("private database row binding differs")
            attempts.append(attempt)
    if len(attempts) != 136:
        raise ValueError("private lifecycle map does not contain 136 attempts")
    identities = {(row["cell"]["task_version_id"], row["arm"], row["seed"]) for row in attempts}
    if len(identities) != 136:
        raise ValueError("private lifecycle map contains duplicate task×arm×seed cells")


def build_seed_plan(protocol: dict[str, Any]) -> dict[str, Any]:
    if protocol.get("sha256") != CORRECTED_PROTOCOL_SHA256:
        raise ValueError("corrected protocol logical digest differs")
    tasks = protocol.get("selection", {}).get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 20:
        raise ValueError("corrected protocol must contain exactly 20 tasks")
    versions = sorted(row["task_version_id"] for row in tasks)
    if len(set(versions)) != 20:
        raise ValueError("corrected protocol task versions are not unique")
    seed_bytes = hashlib.sha256(SEED_DERIVATION_MATERIAL.encode()).digest()
    derived = tuple(
        (int.from_bytes(seed_bytes[index : index + 4], "big") & 0x7FFFFFFF) + 1
        for index in range(0, 16, 4)
    )
    raw_material_digest = "sha256:" + hashlib.sha256(SEED_DERIVATION_MATERIAL.encode()).hexdigest()
    if raw_material_digest != SEED_DERIVATION_SHA256 or derived != INITIAL_SEEDS:
        raise AssertionError("frozen fresh-study seed derivation differs")
    paired_cells = [
        {"task_version_id": task, "arm": arm, "seed": seed}
        for task in versions
        for arm in ARMS
        for seed in INITIAL_SEEDS
    ]
    if len(paired_cells) != 160 or len({tuple(row.values()) for row in paired_cells}) != 160:
        raise AssertionError("initial task×arm×seed roster contains duplicates")
    return {
        "schema": SEED_PLAN_SCHEMA,
        "status": "sealed_not_launched",
        "corrected_protocol": {
            "path": "configs/evaluation/qwen38-teacher3k-fleet-heldout20-pass4-protocol-v2.json",
            "source_commit": CORRECTED_PROTOCOL_SOURCE_COMMIT,
            "file_sha256": CORRECTED_PROTOCOL_FILE_SHA256,
            "sha256": protocol["sha256"],
        },
        "comparison": {
            "arms": list(ARMS),
            "task_count": 20,
            "pass_k": 4,
            "initial_seeds": list(INITIAL_SEEDS),
            "initial_attempts_per_arm": 80,
            "initial_total_attempts": 160,
            "task_arm_seed_roster_sha256": digest(paired_cells),
            "execution_bindings_complete": False,
            "launchable": False,
        },
        "seed_freshness": {
            "derivation_material": SEED_DERIVATION_MATERIAL,
            "derivation_material_sha256": SEED_DERIVATION_SHA256,
            "derivation_rule": (
                "Split the SHA-256 digest into the first four big-endian 32-bit chunks, "
                "mask each to positive 31-bit, then add one."
            ),
            "old_dev17_attempts_reused": 0,
            "reason": (
                "This is a wholly fresh matched study over the corrected 20-task estimand; "
                "historical dev17 results are diagnostic only."
            ),
        },
        "paired_replacement_policy": {
            "rule": (
                "If either arm is not exit-code zero with termination completed, preserve both "
                "originals and hold the paired cell out. Freeze a separate versioned symmetric "
                "successor map before any replacement execution."
            ),
            "score_blind": True,
            "silent_zero_imputation_forbidden": True,
            "replacement_map_in_this_plan": False,
        },
        "effects": {"eval_launches": 0, "model_calls": 0, "cluster_mutations": 0},
    }


def build_public_receipt(
    private_map: dict[str, Any],
    private_path: Path,
    seed_plan: dict[str, Any],
    seed_path: Path,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    validate_private_map(private_map)
    census = private_map["lifecycle_census"]
    sources = [
        {
            "arm": row["arm"],
            "seed": row["seed"],
            "database_evidence_snapshot_sha256": row["database_evidence_snapshot_sha256"],
            "recorded_trace_manifest_sha256": row["recorded_trace_manifest_sha256"],
        }
        for row in private_map["sources"]
    ]
    old_versions = {
        attempt["cell"]["task_version_id"]
        for source in private_map["sources"]
        for attempt in source["attempts"]
    }
    corrected_versions = {row["task_version_id"] for row in protocol["selection"]["tasks"]}
    if len(old_versions) != 17 or len(corrected_versions) != 20:
        raise ValueError("old or corrected task roster is incomplete")
    if (len(old_versions & corrected_versions), len(corrected_versions - old_versions)) != (13, 7):
        raise ValueError("old-to-corrected task roster relation differs")
    observed_seeds = private_map["database_catalog"]["qwen_or_dev17_parsed_seeds"]
    if set(INITIAL_SEEDS) & set(observed_seeds):
        raise ValueError("fresh-study seed exists in the observed database catalog")
    supersession = read_verified(
        SUPERSESSION_PATH, "cyber_qwen38_dev17_pass4_outcome_validity_supersession_v1"
    )
    return {
        "schema": RECEIPT_SCHEMA,
        "status": "sealed_not_launched",
        "private_lifecycle_map": {
            "file_sha256": file_digest(private_path),
            "sha256": private_map["sha256"],
            "records": 136,
            "source_evidence": sources,
            "raw_trace_bytes_independently_rehashed": False,
            "trace_digest_provenance": (
                "The receipt binds the per-result trace SHA-256 values recorded by the rollout "
                "worker. Raw SFS trace bytes were not mounted or opened in this audit."
            ),
            "database_catalog_sha256": private_map["database_catalog"]["catalog_sha256"],
            "database_catalog_count": len(private_map["database_catalog"]["names"]),
            "observed_qwen_or_dev17_seeds": private_map["database_catalog"][
                "qwen_or_dev17_parsed_seeds"
            ],
        },
        "score_blind_lifecycle_census": census,
        "extends_sanitized_supersession": {
            "path": repo_path(SUPERSESSION_PATH),
            "file_sha256": file_digest(SUPERSESSION_PATH),
            "sha256": supersession["sha256"],
            "old_study_remains_incomplete": True,
        },
        "roster_relation": {
            "old_dev17_tasks": len(old_versions),
            "corrected_heldout_tasks": len(corrected_versions),
            "exact_task_version_overlap": len(old_versions & corrected_versions),
            "corrected_only": len(corrected_versions - old_versions),
            "old_only": len(old_versions - corrected_versions),
            "old_attempts_reused": 0,
        },
        "corrected_heldout20_seed_plan": {
            "path": repo_path(seed_path),
            "file_sha256": file_digest(seed_path),
            "sha256": seed_plan["sha256"],
            "initial_seeds": list(INITIAL_SEEDS),
            "old_attempts_reused": 0,
        },
        "scientific_boundary": {
            "existing_dev17_pass_at_4_complete": False,
            "corrected_heldout20_pass_at_4_complete": False,
            "scores_opened": False,
            "future_denominator": (
                "Exactly four normally completed paired attempts per corrected task; held or "
                "infrastructure-invalid attempts are replaced symmetrically, never scored zero."
            ),
        },
        "privacy": {
            "scores_included": False,
            "private_task_session_database_or_path_identifiers_included": False,
            "prompts_responses_flags_answers_or_trace_content_included": False,
            "credentials_included": False,
        },
        "effects": {"eval_launches": 0, "model_calls": 0, "cluster_mutations": 0},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--corrected-protocol", type=Path, required=True)
    parser.add_argument("--old-roster", type=Path, required=True)
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--seed-plan-output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    parser.add_argument("--observed-at", required=True)
    parser.add_argument("--database-url-env", default="ROLLOUT_DATABASE_URL")
    args = parser.parse_args()

    sources = read_verified(args.sources, SOURCE_SCHEMA)
    protocol = read_verified(
        args.corrected_protocol, "cyber_fleet_existing_checkpoint_holdout_protocol_v2"
    )
    if file_digest(args.corrected_protocol) != CORRECTED_PROTOCOL_FILE_SHA256:
        raise ValueError("corrected protocol file digest differs")
    old_roster = read_verified(args.old_roster, "qwen38_fleet_dev17_exact_binding_roster_v1")
    template = os.environ.get(args.database_url_env)
    if not template:
        raise ValueError(f"{args.database_url_env} is not set")
    snapshots = {
        row["label"]: read_database(template, row["database"]) for row in sources["sources"]
    }
    database_catalog = read_database_catalog(template)
    input_bindings = {
        "old_dev17_roster": {
            "path": str(args.old_roster.resolve()),
            "file_sha256": file_digest(args.old_roster),
            "sha256": old_roster["sha256"],
        },
        "corrected_heldout20_protocol": {
            "source_commit": CORRECTED_PROTOCOL_SOURCE_COMMIT,
            "path": str(args.corrected_protocol.resolve()),
            "file_sha256": file_digest(args.corrected_protocol),
            "sha256": protocol["sha256"],
        },
        "query_contract": {
            "cell_fields": list(CELL_FIELDS),
            "local_result_fields": list(RESULT_FIELDS),
            "database_isolation": "repeatable_read_read_only",
            "select_star_used": False,
            "score_bearing_fields_selected": False,
        },
    }
    private_map = build_private_map(
        sources,
        snapshots,
        database_catalog,
        input_bindings,
        {row["task"]["version_id"] for row in old_roster["bindings"]},
        args.observed_at,
    )
    write_sealed(args.private_output, private_map, private=True)
    private_map = read_verified(args.private_output, PRIVATE_SCHEMA)
    if stat.S_IMODE(args.private_output.stat().st_mode) != 0o600:
        raise ValueError("private lifecycle map must be mode 0600")
    validate_private_map(private_map)

    seed_plan = build_seed_plan(protocol)
    write_sealed(args.seed_plan_output, seed_plan)
    seed_plan = read_verified(args.seed_plan_output, SEED_PLAN_SCHEMA)
    receipt = build_public_receipt(
        private_map, args.private_output, seed_plan, args.seed_plan_output, protocol
    )
    write_sealed(args.receipt_output, receipt)


if __name__ == "__main__":
    main()
