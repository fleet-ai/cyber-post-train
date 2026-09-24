from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path

import pytest

from scripts import seal_qwen38_dev17_lifecycle as seal

ROOT = Path(__file__).parents[1]
TASKS = {f"task-version-{index:02d}" for index in range(17)}
COUNTS = {
    ("base", 48): (14, 3, 0),
    ("step1000", 48): (6, 11, 0),
    ("base", 54): (16, 1, 0),
    ("step1000", 54): (4, 12, 1),
    ("base", 61): (11, 6, 0),
    ("step1000", 61): (4, 13, 0),
    ("base", 62): (11, 5, 1),
    ("step1000", 62): (4, 13, 0),
}


def _source(arm: str, seed: int) -> dict:
    label = f"seed{seed}-{arm}"
    return {
        "arm": arm,
        "database": f"private_{label}",
        "job": {
            "kind": "Job",
            "name": f"job-{label}",
            "namespace": "fleet-train-jobs",
            "terminal_state": "Failed",
            "uid": f"00000000-0000-4000-8000-{seed:04d}{0 if arm == 'base' else 1:08d}",
        },
        "label": label,
        "output_root": f"/mnt/sfs/jobs/{label}",
        "seed": seed,
    }


def _snapshot(arm: str, seed: int) -> dict:
    complete, output_limit, process_error = COUNTS[(arm, seed)]
    outcomes = [*("completed" for _ in range(complete))]
    outcomes += [*("output_limit" for _ in range(output_limit))]
    outcomes += [*("process_error" for _ in range(process_error))]
    cells, results = [], []
    for index, (task, termination) in enumerate(zip(sorted(TASKS), outcomes, strict=True)):
        cell = f"cell-{seed}-{arm}-{index:02d}"
        session = f"session-{seed}-{arm}-{index:02d}"
        cells.append(
            {
                "cell_id": cell,
                "experiment_id": "experiment",
                "task_key": f"task-key-{index:02d}",
                "task_version_id": task,
                "model_id": arm,
                "model_revision": "revision",
                "serving_block": "serving",
                "endpoint_model_id": "endpoint",
                "harness_id": "opencode",
                "attempt": 1,
                "state": "accepted",
                "session_id": session,
                "started_at": "2026-09-23T00:00:00+00:00",
                "completed_at": "2026-09-23T00:01:00+00:00",
                "retry_count": 0,
                "max_retries": 0,
                "result_class": "valid",
                "receipt_digest": "sha256:" + "a" * 64,
                "failure_code": None,
                "reconciliation_digest": None,
            }
        )
        results.append(
            {
                "execution_id": f"execution-{cell}",
                "cell_id": cell,
                "execution_generation": 1,
                "run_id": f"run-{cell}",
                "session_id": session,
                "verifier_execution_id": f"verifier-{cell}",
                "config_sha256": "sha256:" + "b" * 64,
                "artifact_directory": f"/private/{cell}",
                "trace_path": f"/private/{cell}/trace.json",
                "trace_sha256": "sha256:" + f"{index + seed:064x}",
                "session_ingest_status": "complete",
                "agent_exit_code": 1 if termination == "process_error" else 0,
                "agent_termination": termination,
                "elapsed_seconds": 60.0,
                "recorded_at": "2026-09-23T00:01:00+00:00",
            }
        )
    return {
        "ledger_metadata": {"plan_sha256": "sha256:" + "c" * 64, "schema_version": "1"},
        "cells": cells,
        "results": results,
    }


def _private_map() -> dict:
    sources = [_source(arm, seed) for seed in (48, 54, 61, 62) for arm in seal.ARMS]
    snapshots = {row["label"]: _snapshot(row["arm"], row["seed"]) for row in sources}
    return seal.build_private_map(
        {"sources": sources},
        snapshots,
        ["q38_dev17_s68_inert", "q38_dev17_s62_source"],
        {"query_contract": "test"},
        TASKS,
        "2026-09-24T00:00:00Z",
    )


def _protocol() -> dict:
    tasks = [
        {"task_key": f"corrected-key-{index:02d}", "task_version_id": f"corrected-{index:02d}"}
        for index in range(20)
    ]
    body = {
        "schema": "cyber_fleet_existing_checkpoint_holdout_protocol_v2",
        "selection": {"tasks": tasks},
    }
    return {**body, "sha256": seal.CORRECTED_PROTOCOL_SHA256}


def test_query_allowlist_cannot_read_score_bearing_fields() -> None:
    forbidden = {
        "score",
        "record_sha256",
        "result_path",
        "result_sha256",
        "reward_path",
        "reward_sha256",
        "session_ingest_path",
        "session_ingest_sha256",
        "cleanup_path",
        "cleanup_sha256",
    }
    assert forbidden.isdisjoint(seal.RESULT_FIELDS)
    assert "*" not in {*seal.CELL_FIELDS, *seal.RESULT_FIELDS}


def test_strict_lifecycle_classification() -> None:
    assert seal.classify_lifecycle(0, "completed") == "normal_completed"
    assert seal.classify_lifecycle(0, "output_limit") == "held_output_limit"
    assert seal.classify_lifecycle(1, "process_error") == ("infrastructure_invalid_process_error")
    assert seal.classify_lifecycle(False, "completed") == "infrastructure_invalid_other"
    assert seal.classify_lifecycle(0, "unknown") == "infrastructure_invalid_other"


def test_private_map_is_exact_score_blind_and_deterministic() -> None:
    value = _private_map()
    assert value["scope"] == {"task_count": 17, "attempts": 136, "arms": list(seal.ARMS)}
    assert value["lifecycle_census"] == {
        "base": {
            "normal_completed": 52,
            "held_output_limit": 15,
            "infrastructure_invalid_process_error": 1,
            "infrastructure_invalid_other": 0,
        },
        "step1000": {
            "normal_completed": 18,
            "held_output_limit": 49,
            "infrastructure_invalid_process_error": 1,
            "infrastructure_invalid_other": 0,
        },
    }
    assert all(
        source["recorded_trace_digests_rehashed_from_raw_bytes"] is False
        for source in value["sources"]
    )
    reversed_sources = list(reversed([_source(a, s) for s in (48, 54, 61, 62) for a in seal.ARMS]))
    snapshots = {row["label"]: _snapshot(row["arm"], row["seed"]) for row in reversed_sources}
    reordered = seal.build_private_map(
        {"sources": reversed_sources},
        snapshots,
        ["q38_dev17_s68_inert", "q38_dev17_s62_source"],
        {"query_contract": "test"},
        TASKS,
        "2026-09-24T00:00:00Z",
    )
    assert seal.digest(value) == seal.digest(reordered)


def test_private_map_rejects_missing_duplicate_and_seed_collision() -> None:
    sources = [_source(arm, seed) for seed in (48, 54, 61, 62) for arm in seal.ARMS]
    snapshots = {row["label"]: _snapshot(row["arm"], row["seed"]) for row in sources}
    broken = copy.deepcopy(snapshots)
    broken[sources[0]["label"]]["results"].pop()
    with pytest.raises(ValueError, match="17 cells"):
        seal.build_private_map(
            {"sources": sources},
            broken,
            [],
            {},
            TASKS,
            "2026-09-24T00:00:00Z",
        )
    with pytest.raises(ValueError, match="collides"):
        seal.build_private_map(
            {"sources": sources},
            snapshots,
            [f"q38_dev17_s{seal.INITIAL_SEEDS[0]}_collision"],
            {},
            TASKS,
            "2026-09-24T00:00:00Z",
        )


def test_fresh_study_seed_plan_is_whole_study_and_high_entropy() -> None:
    plan = seal.build_seed_plan(_protocol())
    assert plan["comparison"]["initial_seeds"] == list(seal.INITIAL_SEEDS)
    assert plan["comparison"]["initial_total_attempts"] == 160
    assert len(set(seal.INITIAL_SEEDS)) == 4
    assert all(0 < seed < 2**31 for seed in seal.INITIAL_SEEDS)
    assert plan["seed_freshness"]["old_dev17_attempts_reused"] == 0
    assert plan["paired_replacement_policy"]["replacement_map_in_this_plan"] is False
    material_sha = "sha256:" + hashlib.sha256(seal.SEED_DERIVATION_MATERIAL.encode()).hexdigest()
    assert material_sha == seal.SEED_DERIVATION_SHA256


def test_public_receipt_has_only_sanitized_aggregate_evidence(tmp_path) -> None:
    private = _private_map()
    private_path = tmp_path / "private.json"
    seal.write_sealed(private_path, private, private=True)
    private = seal.read_verified(private_path, seal.PRIVATE_SCHEMA)
    seal.validate_private_map(private)
    drifted = copy.deepcopy(private)
    drifted["sources"][0]["attempts"][0]["local_result"]["trace_sha256"] = "sha256:" + "f" * 64
    drifted["sha256"] = seal.logical_digest(drifted)
    with pytest.raises(ValueError, match="binding differs"):
        seal.validate_private_map(drifted)
    seed = seal.build_seed_plan(_protocol())
    seed_path = tmp_path / "seed.json"
    seal.write_sealed(seed_path, seed)
    seed = seal.read_verified(seed_path, seal.SEED_PLAN_SCHEMA)
    protocol = _protocol()
    # Make 13 task versions overlap without exposing any of them in the receipt.
    for index, task in enumerate(protocol["selection"]["tasks"][:13]):
        task["task_version_id"] = f"task-version-{index:02d}"
    receipt = seal.build_public_receipt(private, private_path, seed, seed_path, protocol)
    text = json.dumps(receipt, sort_keys=True)
    assert receipt["roster_relation"] == {
        "old_dev17_tasks": 17,
        "corrected_heldout_tasks": 20,
        "exact_task_version_overlap": 13,
        "corrected_only": 7,
        "old_only": 4,
        "old_attempts_reused": 0,
    }
    assert re.search(r"00000000-0000-4000-8000-", text) is None
    for forbidden in (
        "/mnt/",
        "private_seed",
        "task-version-",
        "session-seed",
        "trace.json",
    ):
        assert forbidden not in text


def test_checked_in_fresh_study_plan_and_receipt_are_bound_and_sanitized() -> None:
    plan_path = ROOT / "configs/evaluation/qwen38-teacher3k-heldout20-pass4-fresh-study-v1.json"
    receipt_path = ROOT / "docs/evidence/qwen38-dev17-lifecycle-heldout20-fresh-study-20260924.json"
    plan = seal.read_verified(plan_path, seal.SEED_PLAN_SCHEMA)
    receipt = seal.read_verified(receipt_path, seal.RECEIPT_SCHEMA)
    assert plan["comparison"]["initial_seeds"] == list(seal.INITIAL_SEEDS)
    assert plan["comparison"]["initial_total_attempts"] == 160
    assert plan["seed_freshness"]["old_dev17_attempts_reused"] == 0
    assert receipt["corrected_heldout20_seed_plan"] == {
        "path": str(plan_path.relative_to(ROOT)),
        "file_sha256": seal.file_digest(plan_path),
        "sha256": plan["sha256"],
        "initial_seeds": list(seal.INITIAL_SEEDS),
        "old_attempts_reused": 0,
    }
    assert receipt["roster_relation"] == {
        "old_dev17_tasks": 17,
        "corrected_heldout_tasks": 20,
        "exact_task_version_overlap": 13,
        "corrected_only": 7,
        "old_only": 4,
        "old_attempts_reused": 0,
    }
    text = json.dumps(receipt, sort_keys=True)
    assert re.search(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", text) is None
    for forbidden in (
        "/mnt/",
        "q38_dev17_",
        "chris-q38-dev17",
        "task_version_id",
        "session_id",
        "trace_path",
        "FLAG{",
    ):
        assert forbidden not in text
