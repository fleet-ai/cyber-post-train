#!/usr/bin/env python3
"""Prepare the score-blind seed-62/63/64 wave and defer seed 65.

The accepted seed-61 packet and comparison stay immutable.  This provider-free
command binds those exact bytes, excludes the irrecoverable seed-55, seed-57,
seed-58, and seed-59 pairs, prepares three fresh symmetric pairs, and freezes
the seed-65 protocol without preparing its packets.  It never previews or
creates a cluster object and never reads scores or rollout content.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from evals.fleet import heldout_launch
from scripts import prepare_qwen38_fleet_protocol_v2_replacements as protocol_v2
from scripts import prepare_qwen38_fleet_seed46_pass8_packets as source

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "cyber_qwen38_fleet_protocol_v2_seed62_seed63_seed64_successor_receipt_v2"
COMPARISON_SCHEMA = "cyber_qwen38_fleet_dev17_pass8_comparison_definition_v5"
PREDECESSOR_RECEIPT_SHA256 = (
    "sha256:f67f16d4de3bd3542e20500bb6946acd6180332160d93fd75a88ebb38422b8da"
)
PREDECESSOR_RECEIPT_FILE_SHA256 = (
    "sha256:955d6cd575cb27d9dfbbbf1ca02a749120bc583061555e0abdd458620c307c87"
)
PREDECESSOR_COMPARISON_SHA256 = (
    "sha256:fe5ea9cb30979e1449356f63750958b57d27bee1fbdfde6b78e08618e1a5b68e"
)
PREDECESSOR_COMPARISON_FILE_SHA256 = (
    "sha256:ce9c0d11a30ca3589e538cef481580f5d4d4dccb170e86a4593492fb68a1780d"
)
PREDECESSOR_PACKET_SHA256 = {
    "base": "sha256:fe004007ff5d5bbe174dc665b52631d4caf424546b2856df333226be03bc60be",
    "candidate": "sha256:8857a112b53f438603ac1db23738154f0e8d68a2ad21e71caed2945b9d9a1189",
}
PREDECESSOR_SEED = 61
PREDECESSOR_PROJECTED_ROLLOUTS = 384
SUPERSEDED_TO_SUCCESSOR = {55: 62, 57: 63, 58: 65, 59: 64}
SUPERSEDED_TO_ORIGINAL = {55: 47, 57: 50, 58: 51, 59: 52}
RENDERED_SUCCESSOR_SEEDS = (62, 63, 64)
DEFERRED_SUCCESSOR_SEEDS = (65,)
FINAL_SUCCESSOR_SEEDS = (*RENDERED_SUCCESSOR_SEEDS, *DEFERRED_SUCCESSOR_SEEDS)
INCLUDED_SEEDS = (48, 54, 60, 61, 62, 63, 64, 65)
EXCLUDED_PROTOCOL_V2_SEEDS = (55, 56, 57, 58, 59)
NEW_ROLLOUTS = protocol_v2.TASKS_PER_ARM * len(protocol_v2.ARMS) * len(RENDERED_SUCCESSOR_SEEDS)
DEFERRED_SEED65_ROLLOUTS = protocol_v2.TASKS_PER_ARM * len(protocol_v2.ARMS)
PROJECTED_IMMEDIATE_ROLLOUTS = 485
PROJECTED_WITH_DEFERRED_SEED65 = PROJECTED_IMMEDIATE_ROLLOUTS + DEFERRED_SEED65_ROLLOUTS

EVIDENCE_ROOT = ROOT / "docs/evidence/qwen38-study"
EVIDENCE_FILES = {
    "seed55_invalidation": (
        "2026-09-23-q38-dev17-seed55-pair-invalidation.json",
        "sha256:6ac8ce9b0c9db5337d392e5705ae4f4fbb1ebdedd38b1f6d47a9f28a19bb3f45",
        "sha256:321e750c7a66be292bd724dc28dfcbd8b79f02344da6ce3ddcf3a716392e7080",
    ),
    "seed55_terminal": (
        "2026-09-23-q38-dev17-seed55-base-terminal-observation.json",
        "sha256:6d5525309f0d275f5940257fd4ec9a011134a5e9392607760187c9ca1af517d1",
        "sha256:ea7a7258890c1fd583dcf714c239ee84ed97aaec02808d36a72dea30b6a611fb",
    ),
    "seed55_pre": (
        "2026-09-23-q38-dev17-seed55-candidate-retirement-pre.json",
        "sha256:93e68b175f1f526709c10dd5b9582b7d84c9984d455c34ea2b6f1a6c9102d11d",
        "sha256:02d5c71f722d2aecfbe3719b7f57016369ea5d09aa9eae8801ba074d6894e3e7",
    ),
    "seed55_post": (
        "2026-09-23-q38-dev17-seed55-candidate-retirement-receipt.json",
        "sha256:3ad589068ace08483d93cd36eb87de446fc02d227f207f0b3e4a5aa1a441bb81",
        "sha256:6e6807cfddfc94a903320221221a10d79f44ed3f27b8d48ba753b2d3f119b4f9",
    ),
    "seed57_invalidation": (
        "2026-09-23-q38-dev17-seed57-pair-invalidation.json",
        "sha256:c26c4a86d580faa69f469f76144a75285bf6ff559b616f9735c0b5818906fb3c",
        "sha256:8334a43d901d1b8a8f8d7d52f0038adc54e98b03a8963e4d45f5a7915bdf88cd",
    ),
    "seed57_pre": (
        "2026-09-23-q38-dev17-seed57-pair-retirement-pre.json",
        "sha256:5a74a84acf7acf3f98831cdc5116941e373dbc85dacec306661bb8b9c41f764c",
        "sha256:aff75bc2005d49e77712eb9a586fe66052fde49692efa6bd1ba448767afb8cfa",
    ),
    "seed57_post": (
        "2026-09-23-q38-dev17-seed57-pair-retirement-receipt.json",
        "sha256:336ae0293eec925bf7705d03c155a914f0e14a82c060a33e6f2f304ef32ec5b2",
        "sha256:70d268a69069ee21d4a72fd08f61f9bfaded4a79dbbfbaf596c6fd955fcc0944",
    ),
    "seed58_classification": (
        "2026-09-23-q38-dev17-seed58-pair-classification.json",
        "sha256:523bc8c521652c43e5c566c4827502faa0bdd1039fc5afda1fbe694cbad99213",
        "sha256:8656ec064ddcdcb54e78c8f51a8c3b487719ec334c367b136c3f3fa87f373c42",
    ),
    "seed59_snapshot": (
        "2026-09-23-q38-dev17-seed59-pair-provisional-noartifact.json",
        "sha256:5d1a9c70699e375e9bddf82d4e3746be98a6c6b2a25c23199089d47bd0d4b630",
        "sha256:196771c9acd4cd684f46d337c62f34ee987451df52bd9986d3bb6549d3e58386",
    ),
    "seed59_pre": (
        "2026-09-23-q38-dev17-seed59-pair-retirement-pre.json",
        "sha256:a2b341b186a23701c240d16a57cc54c670fad94b5bdea3a31eeb3c15de174666",
        "sha256:aa58aa7d4934dfa9813340f8234b1a49d6d742679e83cc4684e75b83ec4803e2",
    ),
    "seed59_post_v2": (
        "2026-09-23-q38-dev17-seed59-pair-retirement-receipt-v2.json",
        "sha256:b1bca260da6d9b5b869b2a576fff33d539e14fdd183b8f8c721e45bda75379f6",
        "sha256:b45f9e2b5e88489d9b43e75c5abd1b2325d75f9fa32839bef31da113545290d6",
    ),
}
SEED59_INVALID_V1 = (
    "2026-09-23-q38-dev17-seed59-pair-retirement-invalid-v1.json",
    "sha256:35235bd522ba7b188eaee9a3f0de3c82e7a3327270d4e8ef2ce6c591de5ea11e",
    "sha256:5c4d2812706ffd69b1ed96a9c9bfff1f9b0c336e2c50e680ba82e6fd536dd02a",
    "sha256:7b740197ad415576804ecdbc74b118f9974a6de33ac557962488ef3c367f44f1",
)
EVIDENCE_DIGEST_KEYS_BY_SEED = {
    55: ("seed55_invalidation", "seed55_terminal", "seed55_pre", "seed55_post"),
    57: ("seed57_invalidation", "seed57_pre", "seed57_post"),
    58: ("seed58_classification",),
    59: ("seed59_snapshot", "seed59_pre", "seed59_post_v2"),
}
BUDGET_FILE_SHA256 = "sha256:520ccffb5011e12e899386bc480c5e26a30ea5d21b4accdaa32e7ce3d17faa73"
BUDGET_SHA256 = "sha256:b32a0429d5e46cb9b2c079793b354e7e47ec7b08b845d27dab2e0954fe45732b"
ABSENCE_FILE_SHA256 = "sha256:916e6e540fb6d2c0f776029bc7e4c20e85bd67326d31b005c5698f57ce1b30f1"
ABSENCE_SHA256 = "sha256:968befbce2bb4bbc9a1c7008c66685ff9939adac7292531f9278c52591efe2ce"
ABSENCE_LEDGER_COMMIT = "0bc7236565e8719e232bc5085b08eeace42fe64d"
ABSENCE_LEDGER_FILE_SHA256 = (
    "sha256:312602dd20397ba66028073e3b45cb695db6c3931df0be6d954553a39b3b304a"
)


def _full_comparison_packet_state() -> dict[str, Any]:
    return {
        "state": "seed65_deferred_current_utc_cap",
        "full_roster_launch_ready": False,
        "final_aggregation_allowed": False,
        "score_unseal_allowed": False,
        "rendered_successor_seeds": list(RENDERED_SUCCESSOR_SEEDS),
        "deferred_successor_seeds": list(DEFERRED_SUCCESSOR_SEEDS),
        "rendered_packet_count": len(RENDERED_SUCCESSOR_SEEDS) * len(protocol_v2.ARMS),
        "missing_packet_cells": [{"seed": 65, "arm_id": arm_id} for arm_id in protocol_v2.ARMS],
        "requires_fresh_budget_refresh": True,
        "requires_fresh_destination_absence": True,
        "requires_fresh_live_parity": True,
        "requires_versioned_successor_receipt": True,
    }


def _binding(path: Path, value: dict[str, Any]) -> dict[str, str]:
    return {
        "path": str(path.relative_to(ROOT)) if ROOT in path.resolve().parents else str(path),
        "file_sha256": protocol_v2._file_sha(path),  # noqa: SLF001
        "sha256": value["sha256"],
    }


def _evidence() -> dict[str, Any]:
    values: dict[str, dict[str, Any]] = {}
    bindings: dict[str, dict[str, str]] = {}
    for label, (name, file_sha256, self_sha256) in EVIDENCE_FILES.items():
        path = EVIDENCE_ROOT / name
        value = protocol_v2._verified(path, label)  # noqa: SLF001
        if protocol_v2._file_sha(path) != file_sha256 or value.get("sha256") != self_sha256:  # noqa: SLF001
            raise ValueError(f"{label} differs from the reviewed evidence")
        values[label] = value
        bindings[label] = _binding(path, value)
    invalid_name, invalid_file_sha256, invalid_embedded_sha256, invalid_canonical_sha256 = (
        SEED59_INVALID_V1
    )
    invalid_path = EVIDENCE_ROOT / invalid_name
    invalid = protocol_v2._read(invalid_path, "seed59 invalid v1 receipt")  # noqa: SLF001
    actual_invalid_canonical = protocol_v2._canonical(  # noqa: SLF001
        {key: item for key, item in invalid.items() if key != "sha256"}
    )
    if (
        protocol_v2._file_sha(invalid_path) != invalid_file_sha256  # noqa: SLF001
        or invalid.get("sha256") != invalid_embedded_sha256
        or actual_invalid_canonical != invalid_canonical_sha256
        or invalid_embedded_sha256 == invalid_canonical_sha256
    ):
        raise ValueError("seed59 invalid v1 receipt does not match the reviewed bad bytes")
    bindings["seed59_invalid_post_v1"] = {
        "path": (
            str(invalid_path.relative_to(ROOT))
            if ROOT in invalid_path.resolve().parents
            else str(invalid_path)
        ),
        "file_sha256": invalid_file_sha256,
        "embedded_self_sha256": invalid_embedded_sha256,
        "canonical_self_sha256": invalid_canonical_sha256,
        "authoritative": False,
    }
    privacy = {
        "cell_session_task_ids_included": False,
        "credentials_included": False,
        "prompts_responses_flags_rewards_or_trace_content_read_or_included": False,
        "score_blind": True,
        "scores_read_or_included": False,
    }
    seed55 = values["seed55_invalidation"]
    seed57 = values["seed57_invalidation"]
    seed58 = values["seed58_classification"]
    seed59 = values["seed59_snapshot"]
    seed59_post = values["seed59_post_v2"]
    if (
        seed55.get("seed") != 55
        or seed57.get("seed") != 57
        or seed55.get("privacy") != privacy
        or seed57.get("privacy") != privacy
        or seed55.get("classification")
        != "whole_pair_infrastructure_invalid_requires_fresh_paired_successor"
        or seed57.get("classification")
        != "whole_pair_infrastructure_invalid_requires_fresh_paired_successor"
        or seed55.get("decision", {}).get("cell_level_splicing_allowed") is not False
        or seed57.get("decision", {}).get("partial_splicing_allowed") is not False
        or values["seed55_terminal"].get("database", {}).get("summary", {}).get("by_state")
        != {
            "accepted": 12,
            "claimed": 0,
            "grading": 0,
            "pending": 0,
            "retry_review": 5,
            "running": 0,
            "terminal": 0,
        }
        or values["seed55_post"].get("post_release", {}).get("exact_job_count") != 0
        or values["seed57_post"].get("post_release", {}).get("exact_job_count") != 0
        or values["seed57_post"].get("post_release", {}).get("kueue_quota_claims_released")
        is not True
        or values["seed57_post"].get("post_release", {}).get("ledger_claims_mutated") is not False
        or values["seed55_post"].get("scope", {}).get("whole_pair_excluded") is not True
        or values["seed57_post"].get("scope", {}).get("whole_pair_excluded") is not True
        or seed58.get("schema") != "q38_dev17_s58_score_blind_pair_classification_v1"
        or seed58.get("protocol_id") != "q38-dev17-s58-base-t3k32s1000-replacement-p1-v2"
        or seed58.get("comparison_protocol_sha256")
        != "sha256:fd645d784d659b0931881827cfd9ef0141c90213058e234eaa06225766e7521a"
        or seed58.get("frozen_score_blind_counts", {}).get("base")
        != {
            "total": 17,
            "accepted": 17,
            "pending": 0,
            "retry_review": 0,
            "active_states": 0,
            "unexpired_leases": 0,
        }
        or seed58.get("candidate_unresolved_classification", {}).get("pending")
        != {
            "count": 1,
            "retry_count": 0,
            "max_retries": 0,
            "worker_present": False,
            "claim_present": False,
            "lease_present": False,
            "session_present": False,
            "local_result_or_artifact_present": False,
            "events": ["initialized"],
            "recoverable_without_new_rollout": False,
        }
        or seed58.get("release_census")
        != {
            "source_job_active_count": 0,
            "source_active_pod_count": 0,
            "source_unfinished_workload_count": 0,
            "active_session_claim_state_count": 0,
            "unexpired_lease_count": 0,
            "resources_held": False,
        }
        or seed58.get("decision", {}).get("classification") != "infrastructure_invalid_whole_pair"
        or seed58.get("decision", {}).get("roster_eligible") is not False
        or seed58.get("decision", {}).get("reroll_performed") is not False
        or seed58.get("decision", {}).get("rescore_performed") is not False
        or seed58.get("decision", {}).get("reconciliation_performed") is not False
        or seed58.get("decision", {}).get("ledger_mutated") is not False
        or seed58.get("privacy")
        != {
            "scores_read_or_included": False,
            "prompts_responses_flags_rewards_or_trace_content_included": False,
            "private_cell_session_task_or_claim_ids_included": False,
            "credentials_included": False,
        }
        or seed59.get("provisional_classification", {}).get("unsupported_no_artifact_cell_present")
        is not True
        or seed59.get("provisional_classification", {}).get("unsupported_signature")
        != "post_claim.connecterror"
        or seed59.get("provisional_classification", {}).get("counterpart_sessions_started") != 0
        or values["seed59_pre"].get("reason", {}).get("class")
        != "exact_valid8_whole_pair_exclusion"
        or seed59_post.get("schema") != "q38_dev17_protocol_v2_exact_pair_retirement_receipt_v2"
        or seed59_post.get("supersedes_invalid_receipt")
        != {
            "path": (
                "/private/tmp/q38-s59-provisional-noartifact.20260923T121234Z/"
                "RETIREMENT_RECEIPT_INVALID_V1.json"
            ),
            "file_sha256": invalid_file_sha256,
            "embedded_self_sha256": invalid_embedded_sha256,
            "canonical_self_sha256": invalid_canonical_sha256,
            "reason": (
                "embedded self digest included a trailing newline and failed canonical verification"
            ),
        }
        or seed59_post.get("scope", {}).get("whole_pair_excluded") is not True
        or seed59_post.get("post_release", {}).get("exact_job_count") != 0
        or seed59_post.get("post_release", {}).get("exact_owner_workload_count") != 0
        or seed59_post.get("post_release", {}).get("exact_owner_pod_count") != 0
        or seed59_post.get("post_release", {}).get("kueue_quota_claims_released") is not True
        or seed59_post.get("post_release", {}).get("ledger_claims_mutated") is not False
        or seed59_post.get("post_release", {}).get("live_session_ids_attached_to_claims") != 0
        or seed59_post.get("post_release", {}).get("residual_lease_bound_claims_without_session")
        != 4
    ):
        raise ValueError("seed-55/57/58/59 whole-pair exclusion evidence differs")
    return bindings


def _budget(path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    value = protocol_v2._verified(path, "seed-62/63/64 budget refresh with seed-65 hold")  # noqa: SLF001
    if (
        protocol_v2._file_sha(path) != BUDGET_FILE_SHA256  # noqa: SLF001
        or value.get("sha256") != BUDGET_SHA256
        or value.get("schema") != "cyber_qwen38_fleet_global_daily_rollout_budget_refresh_v2"
        or value.get("window_utc")
        != {
            "start_inclusive": "2026-09-23T00:00:00Z",
            "end_exclusive": "2026-09-24T00:00:00Z",
        }
        or value.get("budget")
        != {
            "actual_started_rollouts": 199,
            "daily_cap": protocol_v2.DAILY_ROLLOUT_CAP,
            "deferred_seed65_pair_additional_cost": DEFERRED_SEED65_ROLLOUTS,
            "immediate_seed62_seed63_seed64_reserved_not_started": 286,
            "projected_after_immediate_seed62_seed63_seed64": PROJECTED_IMMEDIATE_ROLLOUTS,
            "projected_with_deferred_seed65_pair": PROJECTED_WITH_DEFERRED_SEED65,
            "remaining_after_immediate_seed62_seed63_seed64": 15,
            "seed65_over_cap_if_launched_same_day": 19,
            "within_cap_for_immediate_seed62_seed63_seed64": True,
            "within_cap_with_deferred_seed65_pair": False,
        }
        or value.get("reservation")
        != {
            "arms_per_seed": 2,
            "deferred_successor_seed": 65,
            "immediate_successor_seeds": list(RENDERED_SUCCESSOR_SEEDS),
            "prior_reviewed_projected_rollouts": 486,
            "released_seed58_unstarted_reservation": 1,
            "rollouts_per_seed_pair": protocol_v2.TASKS_PER_ARM * len(protocol_v2.ARMS),
            "seed58_classification_file_sha256": EVIDENCE_FILES["seed58_classification"][1],
            "seed58_classification_self_sha256": EVIDENCE_FILES["seed58_classification"][2],
            "tasks_per_arm": protocol_v2.TASKS_PER_ARM,
            "zero_credit_for_started_excluded_seed58_sessions": True,
        }
        or value.get("census", {}).get("started_rollouts") != 199
        or value.get("census", {}).get("outside_dev17_started_rollouts") != 0
        or value.get("census", {}).get("unreadable_databases") != 0
        or value.get("proof", {}).get("provider_requests") != 0
        or value.get("proof", {}).get("port_forward_closed_after_query") is not True
        or value.get("privacy", {}).get("score_blind") is not True
        or value.get("external_mutations") != 0
    ):
        raise ValueError("seed-62/63/64 budget refresh differs")
    return value, _binding(path, value)


def _destinations(seed: int) -> dict[str, dict[str, str]]:
    return {
        "base": {
            "job": f"chris-q38-dev17-s{seed}-base-repl-p1-v2",
            "config_map": f"chris-q38-dev17-s{seed}-base-repl-code-v2",
            "output": f"/mnt/sfs/jobs/chris-q38-fleet-dev17-s{seed}-base-repl-p1-v2",
            "database": f"q38_dev17_s{seed}_base_repl_p1_v2",
            "identity": f"q38-s{seed}-base-repl-p1-v2",
        },
        "candidate": {
            "job": f"chris-q38-dev17-s{seed}-t3k32s1000-repl-p1-v2",
            "config_map": f"chris-q38-dev17-s{seed}-t3k32s1000-repl-code-v2",
            "output": f"/mnt/sfs/jobs/chris-q38-fleet-dev17-s{seed}-t3k32s1000-repl-p1-v2",
            "database": f"q38_dev17_s{seed}_t3k32s1000_repl_p1_v2",
            "identity": f"q38-s{seed}-t3k32s1000-repl-p1-v2",
        },
    }


def _absence(path: Path) -> dict[str, str]:
    value = protocol_v2._verified(path, "seed-62/63/64 destination absence")  # noqa: SLF001
    expected = [
        _destinations(seed)[arm] for seed in RENDERED_SUCCESSOR_SEEDS for arm in protocol_v2.ARMS
    ]
    checks = value.get("checks", {})
    if (
        protocol_v2._file_sha(path) != ABSENCE_FILE_SHA256  # noqa: SLF001
        or value.get("sha256") != ABSENCE_SHA256
        or value.get("schema") != "cyber_qwen38_fleet_successor_absence_receipt_v1"
        or checks.get("jobs_absent") != sorted(row["job"] for row in expected)
        or checks.get("config_maps_absent") != sorted(row["config_map"] for row in expected)
        or checks.get("databases_absent") != sorted(row["database"] for row in expected)
        or checks.get("output_roots_absent") != sorted(row["output"] for row in expected)
        or checks.get("ledger_identities_absent") != sorted(row["identity"] for row in expected)
        or value.get("proof", {}).get("ledger_commit") != ABSENCE_LEDGER_COMMIT
        or value.get("proof", {}).get("ledger_file_sha256") != ABSENCE_LEDGER_FILE_SHA256
        or value.get("proof", {}).get("output_probe", {}).get("pod_uid")
        != "b6605292-f0a8-47bb-8a1b-113cd2518c43"
        or value.get("proof", {}).get("output_probe", {}).get("source_job_uid")
        != "5232b375-09d8-43ff-ac35-ab259e582d0b"
        or value.get("proof", {}).get("provider_requests") != 0
        or value.get("privacy", {}).get("score_blind") is not True
        or value.get("external_mutations") != 0
    ):
        raise ValueError("seed-62/63/64 destination absence receipt differs")
    return _binding(path, value)


def _preserve_input(
    source_path: Path, destination: Path, binding: dict[str, str], relative_path: str
) -> dict[str, str]:
    destination.write_bytes(source_path.read_bytes())
    destination.chmod(0o600)
    if destination.is_symlink() or protocol_v2._file_sha(destination) != binding["file_sha256"]:  # noqa: SLF001
        raise ValueError("preserved successor evidence bytes differ")
    return {**binding, "path": relative_path}


def _frozen_config_contract(value: dict[str, Any]) -> dict[str, Any]:
    contract = copy.deepcopy(value)
    contract.pop("name")
    contract.pop("model_artifact_binding", None)
    contract["sampling"].pop("seed")
    return contract


def _predecessor(
    packet_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    receipt_path = packet_root / "SUCCESSOR_RECEIPT.json"
    definition_path = packet_root / "COMPARISON_DEFINITION.json"
    receipt = protocol_v2._verified(receipt_path, "seed-61 successor receipt")  # noqa: SLF001
    definition = protocol_v2._verified(definition_path, "seed-61 comparison")  # noqa: SLF001
    if (
        receipt.get("sha256") != PREDECESSOR_RECEIPT_SHA256
        or protocol_v2._file_sha(receipt_path) != PREDECESSOR_RECEIPT_FILE_SHA256  # noqa: SLF001
        or definition.get("sha256") != PREDECESSOR_COMPARISON_SHA256
        or protocol_v2._file_sha(definition_path)  # noqa: SLF001
        != PREDECESSOR_COMPARISON_FILE_SHA256
        or receipt.get("comparison_definition") != definition
        or receipt.get("comparison_definition_file_sha256") != PREDECESSOR_COMPARISON_FILE_SHA256
        or receipt.get("included_seeds") != [48, 54, 55, 57, 58, 59, 60, 61]
        or definition.get("excluded_protocol_v2_seeds") != [56]
        or definition.get("included_seeds") != receipt.get("included_seeds")
        or receipt.get("capacity", {}).get("projected_after_seed61_reservation")
        != PREDECESSOR_PROJECTED_ROLLOUTS
        or receipt.get("privacy")
        != {
            "score_values_read": False,
            "prompts_responses_flags_rewards_or_trace_content_read": False,
            "infrastructure_reason_classes_only": True,
        }
        or receipt.get("external_mutations") != 0
        or receipt.get("server_preview_performed") is not False
        or receipt.get("launch_performed") is not False
    ):
        raise ValueError("seed-61 predecessor differs from the accepted bytes")
    rows = {
        row.get("arm_id"): row
        for row in receipt.get("replacement_arms", [])
        if isinstance(row, dict)
    }
    if set(rows) != set(protocol_v2.ARMS):
        raise ValueError("seed-61 predecessor lacks its complete pair")
    contracts: dict[str, dict[str, Any]] = {}
    for arm in protocol_v2.ARMS:
        packet_path = packet_root / "seed61" / arm / "LAUNCH_PACKET.json"
        binding = protocol_v2._evaluation_binding(packet_path)  # noqa: SLF001
        package = heldout_launch.build_package(packet_path)
        row = rows[arm]
        if (
            binding["packet_file_sha256"] != PREDECESSOR_PACKET_SHA256[arm]
            or binding["packet_file_sha256"] != row.get("packet_file_sha256")
            or binding["evaluation_identity_sha256"] != row.get("evaluation_identity_sha256")
            or binding["evaluation_plan_sha256"] != row.get("evaluation_plan_sha256")
            or binding["runtime_files_sha256"] != row.get("runtime_files_sha256")
            or package.packet.identity.get("sampling_seed") != PREDECESSOR_SEED
        ):
            raise ValueError("seed-61 packet bytes differ from the predecessor receipt")
        contracts[arm] = _frozen_config_contract(package.evaluation_config)
    return receipt, definition, contracts


def _validate_successor_package(packet_path: Path) -> heldout_launch.Package:
    package = heldout_launch.build_package(packet_path)
    pod = package.job.get("spec", {}).get("template", {}).get("spec")
    if not isinstance(pod, dict) or pod.get("automountServiceAccountToken") is not False:
        raise ValueError("new successor Job must disable service-account token automount")
    if (
        heldout_launch.require_postgres_client_label(package.job, label="new successor Job")
        is not True
    ):
        raise ValueError("new successor Job must carry the exact PostgreSQL client label")
    return package


def _require_weight_probe_difference(parity: dict[str, Any]) -> None:
    if parity.get("fixed_probe_logit_projection_differs_between_weights") is not True:
        raise ValueError("live parity did not prove distinct base and candidate weights")


def _validate_packet_partition(
    replacement_arms: list[dict[str, Any]],
    protocols: dict[int, dict[str, Any]],
    packet_state: dict[str, Any],
) -> None:
    expected_cells = {
        (seed, arm_id) for seed in RENDERED_SUCCESSOR_SEEDS for arm_id in protocol_v2.ARMS
    }
    actual_cells = {(row.get("successor_seed"), row.get("arm_id")) for row in replacement_arms}
    if (
        actual_cells != expected_cells
        or len(replacement_arms) != len(expected_cells)
        or set(protocols) != set(FINAL_SUCCESSOR_SEEDS)
        or packet_state != _full_comparison_packet_state()
        or set(RENDERED_SUCCESSOR_SEEDS) & set(DEFERRED_SUCCESSOR_SEEDS)
        or set(RENDERED_SUCCESSOR_SEEDS) | set(DEFERRED_SUCCESSOR_SEEDS)
        != set(FINAL_SUCCESSOR_SEEDS)
    ):
        raise ValueError("rendered and deferred successor packet sets differ")


def _harden_and_reseal(packet_path: Path) -> heldout_launch.Package:
    packet = heldout_launch.load_packet(packet_path)
    raw = json.loads(packet_path.read_text(encoding="utf-8"))
    job_path = packet.files["job"]
    job = yaml.safe_load(job_path.read_text(encoding="utf-8"))
    pod = job.get("spec", {}).get("template", {}).get("spec") if isinstance(job, dict) else None
    if not isinstance(pod, dict):
        raise ValueError("new successor Job Pod spec is malformed")
    existing = pod.get("automountServiceAccountToken")
    if existing is not None and existing is not False:
        raise ValueError("new successor Job has a conflicting service-account token policy")
    pod["automountServiceAccountToken"] = False
    job_path.write_text(yaml.safe_dump(job, sort_keys=False), encoding="utf-8")
    raw["files"]["job"]["sha256"] = protocol_v2._file_sha(job_path)  # noqa: SLF001
    source._write_json(packet_path, raw)  # noqa: SLF001
    return _validate_successor_package(packet_path)


def _definition(
    predecessor: dict[str, Any], protocols: dict[int, dict[str, Any]]
) -> dict[str, Any]:
    value = copy.deepcopy(predecessor)
    value.pop("sha256")
    prior_parent = value["predecessor_comparison_definition_sha256"]
    prior_retirement = value.pop("superseded_replacement")
    value["schema"] = COMPARISON_SCHEMA
    value["protocol_v2_migration_comparison_definition_sha256"] = prior_parent
    value["predecessor_comparison_definition_sha256"] = PREDECESSOR_COMPARISON_SHA256
    value["predecessor_successor_receipt_sha256"] = PREDECESSOR_RECEIPT_SHA256
    value["excluded_protocol_v2_seeds"] = list(EXCLUDED_PROTOCOL_V2_SEEDS)
    value["included_seeds"] = list(INCLUDED_SEEDS)
    replacements = SUPERSEDED_TO_SUCCESSOR
    value["replacement_mapping"] = [
        {
            **row,
            "replacement_seed": replacements.get(row["replacement_seed"], row["replacement_seed"]),
        }
        for row in predecessor["replacement_mapping"]
    ]
    value["superseded_replacements"] = [
        prior_retirement,
        *[
            {
                "invalid_original_seed": SUPERSEDED_TO_ORIGINAL[old],
                "superseded_replacement_seed": old,
                "successor_seed": new,
                "whole_pair_excluded": True,
                "evidence_receipt_sha256s": [
                    EVIDENCE_FILES[key][2] for key in EVIDENCE_DIGEST_KEYS_BY_SEED[old]
                ],
            }
            for old, new in SUPERSEDED_TO_SUCCESSOR.items()
        ],
    ]
    value["replica_protocols"] = [
        row
        for row in predecessor["replica_protocols"]
        if row["seed"] not in SUPERSEDED_TO_SUCCESSOR
    ] + [
        {
            "seed": seed,
            "origin": "whole_pair_successor",
            "protocol_id": protocols[seed]["protocol_id"],
            "comparison_protocol_sha256": protocols[seed]["sha256"],
        }
        for seed in FINAL_SUCCESSOR_SEEDS
    ]
    value["replica_protocols"].sort(key=lambda row: row["seed"])
    expected_mapping = [
        (46, 54),
        (47, 62),
        (49, 61),
        (50, 63),
        (51, 65),
        (52, 64),
        (53, 60),
    ]
    value["full_comparison_packet_state"] = _full_comparison_packet_state()
    if (
        [
            (row["invalid_original_seed"], row["replacement_seed"])
            for row in value["replacement_mapping"]
        ]
        != expected_mapping
        or [row["seed"] for row in value["replica_protocols"]] != list(INCLUDED_SEEDS)
        or value["sessions_per_arm"] != protocol_v2.TASKS_PER_ARM * len(INCLUDED_SEEDS)
        or value["total_sessions"]
        != protocol_v2.TASKS_PER_ARM * len(INCLUDED_SEEDS) * len(protocol_v2.ARMS)
    ):
        raise ValueError("combined successor does not preserve exact pass@8")
    value["sha256"] = protocol_v2._canonical(value)  # noqa: SLF001
    return value


def prepare(
    *,
    predecessor_packets: Path,
    live_parity: Path,
    budget_refresh: Path,
    absence_receipt: Path,
    output: Path,
    now: datetime | None = None,
) -> dict[str, Any]:
    if output.exists() or output.is_symlink():
        raise FileExistsError("seed-62/63/64 successor output already exists")
    if not output.parent.is_dir():
        raise ValueError("seed-62/63/64 successor output parent does not exist")
    predecessor_receipt, predecessor_definition, predecessor_contracts = _predecessor(
        predecessor_packets
    )
    evidence = _evidence()
    budget, budget_input_binding = _budget(budget_refresh)
    absence_input_binding = _absence(absence_receipt)
    base_template, task_set, split, _corpus, roster = source._inputs()  # noqa: SLF001
    parity_base, parity_candidate = protocol_v2._configs(  # noqa: SLF001
        base_template, RENDERED_SUCCESSOR_SEEDS[0]
    )
    parity_arms = {
        "base": {
            "model_revision": source.BASE_REVISION,
            "served_id": "qwen3.8-27b",
            "expected_source_path": f"/models/qwen3.8-27b/{source.BASE_REVISION}",
        },
        "candidate": {
            "model_revision": source.CANDIDATE_REVISION,
            "served_id": source.CANDIDATE_ID,
            "expected_source_path": f"/models/{source.CANDIDATE_ID}",
        },
    }
    parity = source.shared._live_parity(  # noqa: SLF001
        live_parity,
        readiness={"arms": {"base": parity_arms["base"], "fresh75": parity_arms["candidate"]}},
        configs={"base": parity_base, "fresh75": parity_candidate},
        now=now or datetime.now(UTC),
    )
    _require_weight_probe_difference(parity)
    temporary = Path(tempfile.mkdtemp(prefix=".fleet-seed626364-successor-", dir=output.parent))
    try:
        evidence_dir = temporary / "evidence"
        evidence_dir.mkdir(mode=0o700)
        budget_binding = _preserve_input(
            budget_refresh,
            evidence_dir / "BUDGET_REFRESH.json",
            budget_input_binding,
            "evidence/BUDGET_REFRESH.json",
        )
        absence_binding = _preserve_input(
            absence_receipt,
            evidence_dir / "ABSENCE_RECEIPT.json",
            absence_input_binding,
            "evidence/ABSENCE_RECEIPT.json",
        )
        replacement_arms: list[dict[str, Any]] = []
        protocols: dict[int, dict[str, Any]] = {}
        treatment_sha256 = {
            arm: protocol_v2._canonical(contract)  # noqa: SLF001
            for arm, contract in predecessor_contracts.items()
        }
        deferred_seed = DEFERRED_SUCCESSOR_SEEDS[0]
        deferred_seed_dir = temporary / f"seed{deferred_seed}"
        deferred_seed_dir.mkdir(mode=0o700)
        deferred_base, _deferred_candidate = protocol_v2._configs(  # noqa: SLF001
            base_template, deferred_seed
        )
        deferred_protocol = protocol_v2._protocol(deferred_base, deferred_seed)  # noqa: SLF001
        protocols[deferred_seed] = deferred_protocol
        source._write_json(  # noqa: SLF001
            deferred_seed_dir
            / (
                f"qwen38-fleet-dev17-seed{deferred_seed}-base-step1000-replacement-protocol-v2.json"
            ),
            deferred_protocol,
        )
        for seed in RENDERED_SUCCESSOR_SEEDS:
            seed_dir = temporary / f"seed{seed}"
            seed_dir.mkdir(mode=0o700)
            base, candidate = protocol_v2._configs(base_template, seed)  # noqa: SLF001
            protocol = protocol_v2._protocol(base, seed)  # noqa: SLF001
            protocols[seed] = protocol
            base_path = seed_dir / f"qwen38-base-fleet-dev17-opencode-seed{seed}-pass1-v2.json"
            candidate_path = seed_dir / (
                f"qwen38-teacher3k32-step1000-fleet-dev17-opencode-seed{seed}-pass1-v2.json"
            )
            protocol_path = seed_dir / (
                f"qwen38-fleet-dev17-seed{seed}-base-step1000-replacement-protocol-v2.json"
            )
            base_provenance = seed_dir / f"qwen38-base-seed{seed}-replacement-provenance-v2.json"
            candidate_provenance = seed_dir / (
                f"qwen38-step1000-seed{seed}-replacement-provenance-v2.json"
            )
            source._write_json(base_path, base)  # noqa: SLF001
            source._write_json(candidate_path, candidate)  # noqa: SLF001
            source._write_json(protocol_path, protocol)  # noqa: SLF001
            source._write_json(  # noqa: SLF001
                base_provenance,
                source._base_provenance(base_path, roster, base["name"]),  # noqa: SLF001
            )
            source._write_json(  # noqa: SLF001
                candidate_provenance,
                source._candidate_packet(candidate["name"]),  # noqa: SLF001
            )
            destinations = _destinations(seed)
            arms = {
                arm: {
                    **parity_arms[arm],
                    "job_name": destination["job"],
                    "config_map_name": destination["config_map"],
                    "output_root": destination["output"],
                    "database": destination["database"],
                }
                for arm, destination in destinations.items()
            }
            for arm_id, config, config_path, checkpoint in (
                ("base", base, base_path, base_provenance),
                ("candidate", candidate, candidate_path, candidate_provenance),
            ):
                row = source.shared._prepare_arm(  # noqa: SLF001
                    seed_dir / arm_id,
                    arm_id=arm_id,
                    arm=arms[arm_id],
                    config_path=config_path,
                    config=config,
                    task_set_path=source.TASK_SET,
                    split_path=source.SPLIT,
                    protocol_path=protocol_path,
                    protocol=protocol,
                    checkpoint_path=checkpoint,
                    proof_path=live_parity,
                    ledger_path=source.LEDGER,
                    source_files=source.V3_SOURCE_FILES if arm_id == "candidate" else None,
                )
                packet_path = seed_dir / arm_id / "LAUNCH_PACKET.json"
                package = _harden_and_reseal(packet_path)
                row["packet_path"] = f"seed{seed}/{arm_id}/LAUNCH_PACKET.json"
                row["packet_file_sha256"] = protocol_v2._file_sha(packet_path)  # noqa: SLF001
                route = next(iter(package.evaluation_config["routes"].values()))
                if (
                    _frozen_config_contract(package.evaluation_config)
                    != predecessor_contracts[arm_id]
                    or package.packet.identity["sampling_seed"] != seed
                    or package.packet.identity["pass_k"] != 1
                    or package.packet.identity["retry_limit"] != 0
                    or package.evaluation_config["max_reviewed_infrastructure_retries"] != 0
                    or len(route["task_versions"]) != protocol_v2.TASKS_PER_ARM
                    or package.job["metadata"]["annotations"].get(
                        heldout_launch.FAILURE_ALERT_ANNOTATION
                    )
                    != heldout_launch.FAILURE_ALERT_OFF
                    or package.job["spec"]["template"]["spec"].get("priorityClassName") != "c1"
                    or "nvidia.com/gpu" in json.dumps(package.job)
                ):
                    raise ValueError(
                        "successor arm changed the frozen treatment or resource policy"
                    )
                old_seed = next(old for old, new in SUPERSEDED_TO_SUCCESSOR.items() if new == seed)
                replacement_arms.append(
                    {
                        **row,
                        **protocol_v2._evaluation_binding(packet_path),  # noqa: SLF001
                        "invalid_original_seed": SUPERSEDED_TO_ORIGINAL[old_seed],
                        "superseded_replacement_seed": old_seed,
                        "successor_seed": seed,
                        "job_name": package.packet.job_name,
                        "config_map_name": package.packet.config_map_name,
                        "output_root": package.packet.output_root,
                        "database": package.packet.database,
                        "ledger_identity": package.packet.identity["evaluation_config_name"],
                        "comparison_protocol_file_sha256": package.packet.identity[
                            "comparison_protocol_file_sha256"
                        ],
                        "comparison_protocol_sha256": package.packet.identity[
                            "comparison_protocol_sha256"
                        ],
                    }
                )
        unique_fields = (
            "packet_path",
            "job_name",
            "config_map_name",
            "output_root",
            "database",
            "ledger_identity",
        )
        if any(
            len({row[field] for row in replacement_arms}) != len(replacement_arms)
            for field in unique_fields
        ):
            raise ValueError("successor packet and destination identities must be unique")
        packet_state = _full_comparison_packet_state()
        _validate_packet_partition(replacement_arms, protocols, packet_state)
        if any(
            (temporary / f"seed{deferred_seed}" / arm_id / "LAUNCH_PACKET.json").exists()
            for arm_id in protocol_v2.ARMS
        ):
            raise ValueError("deferred seed-65 must not have a launch packet")
        definition = _definition(predecessor_definition, protocols)
        source._write_json(temporary / "COMPARISON_DEFINITION.json", definition)  # noqa: SLF001
        receipt = {
            "schema": SCHEMA,
            "predecessor": {
                "successor_receipt_sha256": predecessor_receipt["sha256"],
                "successor_receipt_file_sha256": PREDECESSOR_RECEIPT_FILE_SHA256,
                "comparison_definition_sha256": predecessor_definition["sha256"],
                "comparison_definition_file_sha256": PREDECESSOR_COMPARISON_FILE_SHA256,
                "included_seeds": predecessor_receipt["included_seeds"],
                "packet_file_sha256": PREDECESSOR_PACKET_SHA256,
            },
            "whole_pair_exclusion_evidence": evidence,
            "lineage": [
                {
                    "invalid_original_seed": SUPERSEDED_TO_ORIGINAL[old],
                    "superseded_replacement_seed": old,
                    "successor_seed": new,
                    "whole_pair_excluded": True,
                    "cell_level_replacement_forbidden": True,
                }
                for old, new in SUPERSEDED_TO_SUCCESSOR.items()
            ],
            "comparison_definition": definition,
            "comparison_definition_file_sha256": protocol_v2._file_sha(  # noqa: SLF001
                temporary / "COMPARISON_DEFINITION.json"
            ),
            "included_seeds": list(INCLUDED_SEEDS),
            "excluded_protocol_v2_seeds": list(EXCLUDED_PROTOCOL_V2_SEEDS),
            "replacement_protocols": [
                {
                    "seed": seed,
                    "protocol_id": protocols[seed]["protocol_id"],
                    "sha256": protocols[seed]["sha256"],
                    "file_sha256": protocol_v2._file_sha(  # noqa: SLF001
                        temporary
                        / f"seed{seed}"
                        / (
                            f"qwen38-fleet-dev17-seed{seed}-base-step1000-"
                            "replacement-protocol-v2.json"
                        )
                    ),
                }
                for seed in FINAL_SUCCESSOR_SEEDS
            ],
            "replacement_arms": replacement_arms,
            "full_comparison_packet_state": packet_state,
            "capacity": {
                **budget["budget"],
                "budget_refresh": budget_binding,
                "window_utc": budget["window_utc"],
                "immediate_successor_rollouts": NEW_ROLLOUTS,
                "seed58_accounting": {
                    "released_unstarted_reservations": 1,
                    "credit_claimed_for_started_excluded_sessions": 0,
                },
                "deferred_seed65": {
                    "rollouts": DEFERRED_SEED65_ROLLOUTS,
                    "reserved_in_this_window": False,
                    "included_in_projected_after_reservation": False,
                    "projected_if_reserved_in_same_window": PROJECTED_WITH_DEFERRED_SEED65,
                    "within_current_utc_cap": False,
                    "over_cap_by": PROJECTED_WITH_DEFERRED_SEED65 - protocol_v2.DAILY_ROLLOUT_CAP,
                },
            },
            "destination_absence": absence_binding,
            "destination_absence_scope": {
                "successor_seeds": list(RENDERED_SUCCESSOR_SEEDS),
                "deferred_seed65_checked": False,
            },
            "scientific_identity": {
                "task_count_per_arm": protocol_v2.TASKS_PER_ARM,
                "comparison_arms": list(protocol_v2.ARMS),
                "pass_k": 1,
                "retry_limit": 0,
                "same_task_versions_models_harness_budgets_and_sampling_recipe": True,
                "whole_replica_pairs_only": True,
                "score_blind_exclusion": True,
                "predecessor_treatment_sha256": treatment_sha256,
            },
            "job_security": {
                "root_failure_alert_annotation": {
                    heldout_launch.FAILURE_ALERT_ANNOTATION: heldout_launch.FAILURE_ALERT_OFF
                },
                "pod_postgres_client_label": {
                    heldout_launch.POSTGRES_CLIENT_LABEL: heldout_launch.POSTGRES_CLIENT_LABEL_VALUE
                },
                "automountServiceAccountToken": False,
            },
            "selection": {
                "selection_sha256": task_set["selection_sha256"],
                "split_sha256": split["sha256"],
                "binding_roster_sha256": roster["sha256"],
            },
            "live_parity_file_sha256": protocol_v2._file_sha(live_parity),  # noqa: SLF001
            "live_parity_receipt_sha256": parity["receipt_sha256"],
            "privacy": {
                "score_values_read": False,
                "prompts_responses_flags_rewards_or_trace_content_read": False,
                "infrastructure_reason_classes_only": True,
            },
            "external_mutations": 0,
            "server_preview_performed": False,
            "launch_performed": False,
        }
        if (
            receipt["capacity"]["within_cap_for_immediate_seed62_seed63_seed64"] is not True
            or receipt["capacity"]["within_cap_with_deferred_seed65_pair"] is not False
            or receipt["capacity"]["projected_after_immediate_seed62_seed63_seed64"]
            != PROJECTED_IMMEDIATE_ROLLOUTS
            or receipt["capacity"]["projected_with_deferred_seed65_pair"]
            != PROJECTED_WITH_DEFERRED_SEED65
            or receipt["full_comparison_packet_state"]
            != receipt["comparison_definition"]["full_comparison_packet_state"]
        ):
            raise ValueError("immediate successor budget or deferred seed-65 hold differs")
        receipt["sha256"] = protocol_v2._canonical(receipt)  # noqa: SLF001
        source._write_json(temporary / "SUCCESSOR_RECEIPT.json", receipt)  # noqa: SLF001
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predecessor-packets", type=Path, required=True)
    parser.add_argument("--live-parity", type=Path, required=True)
    parser.add_argument("--budget-refresh", type=Path, required=True)
    parser.add_argument("--absence-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                predecessor_packets=args.predecessor_packets,
                live_parity=args.live_parity,
                budget_refresh=args.budget_refresh,
                absence_receipt=args.absence_receipt,
                output=args.output,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
