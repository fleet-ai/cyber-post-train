"""Freeze the no-launch 50-component Qwen self-SFT pass@8 successor plan."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from training import task_family_split

SCHEMA = "cyber_qwen38_self_sft_full50_pass8_successor_plan_v1"
OUTPUT = "configs/collection/qwen38-base-train50-pass8-successor-v1.plan.json"
SOURCE_COMMIT = "ea5f67dd6a955218c4d4ee2946e365189cfd3a8d"
TARGET_UTC_DAY = "2026-09-25"
SEEDS = tuple(range(51, 59))
DAILY_CAP = 500

CATALOG = "configs/data/qwen38-self-sft-blackbox-task-catalog-20260924-v1.json"
SPLIT = "configs/data/fleet-blackbox-lineage-safe-split-20260924-v2.json"
SELECTION = "configs/collection/qwen38-base-current75-actions-pass4-v2/task-selection.json"
METADATA = "configs/collection/qwen38-base-current75-actions-pass4-v2/metadata-inventory.json"
EVAL_CONFIG = "configs/collection/qwen38-base-current75-actions-pass4-v2/eval-config.json"
COLLECTION_PACKET = (
    "configs/collection/qwen38-base-current75-actions-pass4-v2/collection-packet.json"
)
LIVE_CENSUS = "docs/evidence/qwen38-study/2026-09-24-self-sft-live-operation-census-v1.json"

# Exact bytes already reviewed on SOURCE_COMMIT or, for LIVE_CENSUS, in the
# independently reviewed historical evidence packet.  The successor plan never
# resolves a mutable alias or an unbound task version.
SOURCES = {
    CATALOG: (
        "sha256:dcdea7fbfaabc71d38483e14e65ca5a59eac66d19878dfca0664f828f9f06de2",
        "sha256:3e92c3a0789355d31cd1eeb5f0003bcc0a00c7aec249c64efebad15cd25db20c",
    ),
    SPLIT: (
        "sha256:76e0169c189dd5a0ec2c9858f7172a9d958357cef88c86a8c596b8c8da5663a1",
        "sha256:24b7e98a98b97aca8cc2c63eccbeb28a2963a7cd459078b2ddcb3f386fce1323",
    ),
    SELECTION: (
        "sha256:93e264d7cc5a7c90101f4c510a37a2d02fd4c87e94845a4012db7fa0d486d923",
        "sha256:e7b849eead6418cd1156a52027f234801bbf3bbe196023c56061ae68e3f0af19",
    ),
    METADATA: (
        "sha256:d84b822611129059a4d60165a2ea7f2e0d44f084f2d74c45f904b445a83e0fbf",
        "sha256:32432fe02c1f2253f77a4190c8730bc9348e9053aa2ff39420cb40d6f525c767",
    ),
    EVAL_CONFIG: (
        "sha256:bb9e2c5bd6a7c9924f16cb86474012e9dd19b24ec2c571be7082c7eef7addf9c",
        "sha256:d82657e5c63454171021ceb82dce6b072f0426fa9d22861beaa9c1ad2ee332fc",
    ),
    COLLECTION_PACKET: (
        "sha256:d164781e9d704611c35637a23a61976bab0372906cfd65ee2ccd014a9d1caeba",
        "sha256:ae9798d9aa3ea12137f81a6b6590a45ee6d6eab386813765d0aeae048616ea4e",
    ),
    LIVE_CENSUS: (
        "sha256:2430ce493357bd7305bdb17090e584e0c170e73f789e6dc3659441682d85cf79",
        "sha256:b15975607a839996bfe892f9b122985c7ddbf4fbb27793919521de58f7fdb39c",
    ),
}


def _bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_bytes(value)).hexdigest()


def _file_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _load_sources(root: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for relative, (logical, file_digest) in SOURCES.items():
        path = root / relative
        value = json.loads(path.read_text())
        unsigned = {key: item for key, item in value.items() if key != "sha256"}
        actual_logical = task_family_split.canonical_digest(unsigned)
        if "sha256" not in value:
            actual_logical = _digest(value)
        if actual_logical != logical or _file_digest(path) != file_digest:
            raise ValueError(f"{relative} drifted from its exact reviewed bytes")
        result[relative] = value
    return result


def _identity(row: dict[str, Any]) -> tuple[str, str]:
    key, version = row.get("task_key"), row.get("task_version_id")
    if not isinstance(key, str) or not key or not isinstance(version, str) or not version:
        raise ValueError("exact task key and version are required")
    return key, version


def _pilot(
    train: list[dict[str, Any]], metadata: dict[tuple[str, str], dict[str, Any]]
) -> list[dict[str, Any]]:
    """Pick five deterministic, metadata-diverse train components."""
    remaining = list(train)
    chosen: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    while len(chosen) < 5:
        ranked: list[tuple[int, str, str, dict[str, Any], set[tuple[str, str]]]] = []
        for row in remaining:
            lineage = metadata[_identity(row)]["lineage"]
            facets = {
                ("application", lineage["application"]),
                ("environment", lineage["environment"]),
                ("difficulty", lineage["difficulty"]),
                *(("vulnerability_family", item) for item in lineage["vulnerability_family"]),
            }
            ranked.append(
                (len(facets - seen), row["task_key"], row["task_version_id"], row, facets)
            )
        _, _, _, selected, facets = min(ranked, key=lambda item: (-item[0], item[1], item[2]))
        chosen.append(selected)
        seen |= facets
        remaining.remove(selected)
    return chosen


def build(root: Path) -> dict[str, Any]:
    values = _load_sources(root)
    catalog = values[CATALOG]
    split = values[SPLIT]
    selection = values[SELECTION]
    metadata_value = values[METADATA]
    config = values[EVAL_CONFIG]
    packet = values[COLLECTION_PACKET]
    live_census = values[LIVE_CENSUS]
    waves = {row["name"]: row for row in live_census["waves"]}
    if (
        set(waves) != {"wave_a", "wave_b"}
        or live_census["planned_cells"] != 400
        or live_census["launch_authorization"] is not False
        or live_census["global_exact_source_trajectory_exclusion_proven"] is not False
        or waves["wave_a"]["operation_state"] != "consumed_non_relaunchable"
        or waves["wave_a"]["postgresql"]["reusable_verified_success_trajectories"] != 0
        or waves["wave_b"]["operation_state"] != "unused_at_observation_not_launch_authorized"
    ):
        raise ValueError("historical operation evidence does not preserve the retired state")
    retired_evidence = {
        "schema": live_census["schema"],
        "logical_sha256": live_census["sha256"],
        "file_sha256": SOURCES[LIVE_CENSUS][1],
        "wave_a_state": waves["wave_a"]["operation_state"],
        "wave_a_reusable_verified_success_trajectories": waves["wave_a"]["postgresql"][
            "reusable_verified_success_trajectories"
        ],
        "wave_b_state": waves["wave_b"]["operation_state"],
    }

    train_catalog = {
        _identity(row): row
        for row in catalog["task_versions"]
        if row["classification"] == "immediately_train_eligible"
    }
    heldout_components = {
        row["transitive_family"]["component_id"]
        for row in catalog["task_versions"]
        if row["classification"] == "heldout_family_overlap"
    }
    train_split = [row for row in split["tasks"] if row["split"] == "train"]
    selected = {_identity(row): row for row in selection["tasks"]}
    metadata = {_identity(row): row for row in metadata_value["task_versions"]}
    if (
        len(train_catalog) != 50
        or len(train_split) != 50
        or len(selected) != 50
        or set(train_catalog) != {_identity(row) for row in train_split}
        or set(train_catalog) != set(selected)
        or not set(train_catalog) <= set(metadata)
    ):
        raise ValueError("exact 50-component train authorities do not agree")
    component_rows = []
    for row in sorted(train_split, key=_identity):
        identity = _identity(row)
        catalog_row = train_catalog[identity]
        component = row["shared_atom_component_id"]
        if (
            component != catalog_row["transitive_family"]["component_id"]
            or component in heldout_components
        ):
            raise ValueError("train component overlaps or contradicts protected heldout")
        component_rows.append(
            {
                "task_key": identity[0],
                "task_version_id": identity[1],
                "shared_atom_component_id": component,
            }
        )
    if len({row["shared_atom_component_id"] for row in component_rows}) != 50:
        raise ValueError("the plan requires exactly one task version per train component")

    harness = config["harness"]
    model = config["models"]["source"]
    route = config["routes"]["base"]
    source_identity = {
        "model": model,
        "route": {
            key: route[key]
            for key in ("endpoint_origin", "served_id", "catalog", "model_info", "server_info")
        },
        "images": config["images"],
        "harness": {
            key: harness[key]
            for key in (
                "harness",
                "harness_version",
                "provider_adapter",
                "release_asset_sha256",
                "context_window_size",
                "context_management",
                "compaction_headroom_tokens",
                "max_model_requests",
                "max_output_tokens",
                "timeout_seconds",
                "thinking_mode",
                "tool_catalog_sha256",
                "tools",
            )
        },
        "template_sha256": packet["source"]["template_sha256"],
        "sampling": {"temperature": 0.6, "top_p": 0.95, "seeds": list(SEEDS)},
        "reasoning_policy": packet["admission_policy"]["reasoning_policy"],
        "offline_compaction_policy": packet["admission_policy"]["offline_compaction_policy"],
    }
    source_identity_sha256 = _digest(source_identity)
    cells = [{**row, "seed": seed} for row in component_rows for seed in SEEDS]
    universe = _digest({"source_identity_sha256": source_identity_sha256, "cells": cells})
    component_roster_sha256 = _digest(component_rows)
    pilot = _pilot(component_rows, metadata)
    pilot_cells = [{**row, "seed": seed} for row in pilot for seed in SEEDS]
    pilot_digest = _digest({"source_identity_sha256": source_identity_sha256, "cells": pilot_cells})
    pilot_identities = {_identity(row) for row in pilot}
    expansion_cells = [cell for cell in cells if _identity(cell) not in pilot_identities]
    expansion_digest = _digest(
        {"source_identity_sha256": source_identity_sha256, "cells": expansion_cells}
    )
    operation_name = f"q38-base-train50-pass8-s2-{universe[7:15]}"
    reservation_id = f"self-sft-train50-pass8-{universe[7:19]}"

    reservation_groups = [
        {
            "name": "operational_canary",
            "components": 5,
            "count": 40,
            "cell_universe_sha256": pilot_digest,
        },
        {
            "name": "expansion",
            "components": 45,
            "count": 360,
            "cell_universe_sha256": expansion_digest,
        },
    ]
    body = {
        "schema": SCHEMA,
        "mode": "prepared_no_launch_no_reservation",
        "launch_authorized": False,
        "source_authority_git_commit": SOURCE_COMMIT,
        "sources": {
            path: {"logical_sha256": digests[0], "file_sha256": digests[1]}
            for path, digests in SOURCES.items()
        },
        "source_identity": source_identity,
        "source_identity_sha256": source_identity_sha256,
        "train_roster": {
            "exact_task_versions": 50,
            "transitive_components": 50,
            "one_exact_version_per_component": True,
            "component_task_version_roster": component_rows,
            "component_task_version_roster_sha256": component_roster_sha256,
            "selection_sha256": selection["sha256"],
            "split_sha256": split["sha256"],
            "catalog_sha256": catalog["sha256"],
            "protected_heldout_components": len(heldout_components),
            "heldout_overlap": 0,
        },
        "cell_universe": {
            "planned_cells": 400,
            "seeds": list(SEEDS),
            "sha256": universe,
            "fresh_relative_to_retired_seed_range": [43, 50],
        },
        "stages": {
            "qa_one_cell": {
                "required": True,
                "accepted_cleanup_receipt_sha256": None,
                "status": "missing",
            },
            "operational_canary": {
                "exact_components": 5,
                "planned_cells": 40,
                "task_versions": pilot,
                "cell_universe_sha256": pilot_digest,
                "is_prefix_of_full_collection": True,
                "replay_during_expansion_allowed": False,
                "acceptance_requires": [
                    "all 40 cells have terminal exact identities",
                    "zero infrastructure-invalid or ambiguous cells",
                    "all task environments, sessions, and cluster objects are cleaned",
                    "at least one verifier-confirmed successful trajectory passes corpus admission",
                ],
                "accepted_cleanup_receipt_sha256": None,
                "status": "missing",
            },
            "expansion": {
                "exact_components": 45,
                "planned_cells": 360,
                "cell_universe_sha256": expansion_digest,
                "opens_only_after_both_prior_stages_are_accepted": True,
                "status": "blocked",
            },
        },
        "daily_reservation": {
            "schema": "cyber_fleet_daily_self_sft_collection_reservation_v1",
            "target_utc_day": TARGET_UTC_DAY,
            "expires_after_target_day": True,
            "canonical_cap": DAILY_CAP,
            "count": 400,
            "reservation_id": reservation_id,
            "cell_universe_sha256": universe,
            "component_task_version_roster_sha256": component_roster_sha256,
            "source_identity_sha256": source_identity_sha256,
            "groups": reservation_groups,
            "packet_set_sha256": None,
            "reservation_must_bind_final_plan_sha256": True,
            "shared_lock": f"fleet-daily-rollouts-v1/{TARGET_UTC_DAY}/.lock",
            "requirements": [
                "count legacy, strict-wave, QA, and self-SFT reservation schemas together",
                "fail closed on unknown schemas or a missing exact daily baseline",
                "under one exclusive lock prove committed plus 400 is at most 500",
                "count the separate QA1 reservation too if it occurs on the target UTC day",
                "publish one create-once no-replace row and fsync its parent directory",
                "recover an exact interrupted write without replacing or decrementing it",
            ],
            "reservation_receipt_sha256": None,
            "status": "missing_not_reserved",
        },
        "operation": {
            "operation_root_name": operation_name,
            "dedicated_ledger_id": _digest(
                {"kind": "self_sft_train50_pass8_v1", "cell_universe_sha256": universe}
            ),
            "new_output_database_and_ledger_required": True,
            "retired_operation_evidence": retired_evidence,
            "retired_cell_universe_sha256": (
                "sha256:39cbe899153782e5cdb4ec9fbe5a5b77b1e9ae4f0ed0a128ce7fc6c8101e224a"
            ),
            "retired_names_or_authorizations_may_be_reused": False,
        },
        "fresh_precreate_gates": {
            "exact_task_runtime_freshness": False,
            "global_exact_cell_and_source_trajectory_duplicate_exclusion": False,
            "no_active_queued_intended_or_accepted_overlap": False,
            "new_job_configmap_database_ledger_and_output_absence": False,
            "two_identical_server_previews": False,
            "exact_daily_reservation": False,
            "uid_bound_cleanup_armed": False,
        },
        "implementation_gates": {
            "pass8_400_cell_successor_runtime_reviewed": False,
            "pilot40_packet_set_sealed": False,
            "expansion360_packet_set_sealed": False,
            "dedicated_self_sft_reservation_writer_reviewed": False,
        },
        "cluster_root_contract": {
            "kind": "Job",
            "priority_class_name": "c1",
            "required_top_level_annotation": {"fleet.ai/failure-alerts": "off"},
            "worker_gpu_requests_and_limits": 0,
            "backoff_limit": 0,
            "restart_policy": "Never",
            "create_attempts_maximum": 1,
            "ambiguous_create_replay_allowed": False,
            "exact_uid_bound_terminal_reconciliation_and_cleanup_required": True,
        },
        "privacy": {
            "prompts_or_trajectories_persisted": False,
            "numeric_scores_persisted": False,
            "credentials_persisted": False,
        },
        "external_mutations": 0,
    }
    body["sha256"] = _digest(body)
    return body


def check(root: Path) -> None:
    expected = build(root)
    actual = json.loads((root / OUTPUT).read_text())
    if actual != expected:
        raise ValueError(f"{OUTPUT} does not match exact source authorities")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        check(args.root)
    else:
        print(json.dumps(build(args.root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
