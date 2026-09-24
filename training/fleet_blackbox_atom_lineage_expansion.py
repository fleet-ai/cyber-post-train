"""Reproduce the Sep24 shared-atom lineage census and no-extension split."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from training import shared_atom_lineage, task_family_split

PROVEN = "configs/data/fleet-blackbox-receipt-proven-20260924-v1.json"
CANDIDATES = "configs/data/fleet-blackbox-qa-candidates-20260924-v1.json"
LIVE_LINEAGE = "configs/data/fleet-blackbox-qa33-live-lineage-20260924-v1.json"
PARENT_SPLIT = "configs/data/fleet-blackbox-lineage-safe-split-20260924-v1.json"
PARENT_DECISION = (
    "docs/evidence/qwen38-study/2026-09-24-fleet-blackbox-lineage-safe-split-design-v1.json"
)
TEACHER3K = "configs/data/qwen38-teacher3k-training-lineage-map-20260924-v1.json"

LINEAGE_CENSUS = "configs/data/fleet-blackbox-shared-atom-lineage-census-20260924-v1.json"
SPLIT = "configs/data/fleet-blackbox-lineage-safe-split-20260924-v2.json"
DECISION = "docs/evidence/qwen38-study/2026-09-24-fleet-blackbox-qa33-lineage-qualification-v1.json"

SOURCES = {
    PROVEN: (
        "sha256:23cfc25af183ff474e914443682e4707162110ddc2aa16103a13d984fe05b83c",
        "sha256:6c211259a29376de195411e5cc182bb6f26c5b8027ab86fec855b44f064b0fd0",
    ),
    CANDIDATES: (
        "sha256:4f61fd78b4d92c933030f026e0137abe199e0157fd72b2f959df0354df489be8",
        "sha256:b3d551863655bc8ad92c76bb1b2b0f87c8bcec812b5c42987fc9c0c28c1a5b2a",
    ),
    LIVE_LINEAGE: (
        "sha256:7622ce3870f74a3a9896575d1e42a122826743b6cf15307f1d411694135bea6b",
        "sha256:8f5a0e8f08a8b193ef785bd18741f46f9355fa24b49694b49e9f7ca07c58373a",
    ),
    PARENT_SPLIT: (
        "sha256:b613abaae0e5b52ef279a46a4ab0145dba4517442d1e4198fd52af7de0161d10",
        "sha256:d40c0fc2fd7db91103978777c364354e68d5e6fae57311a44e583902a09c9580",
    ),
    PARENT_DECISION: (
        "sha256:5338adac4e0abed6e5416f05b538349feabf82e1717f4f3253be687750376ee1",
        "sha256:5507acff41b5684f503e0208aa37efd4741d9fce5b38299ae3d4987fafe67465",
    ),
    TEACHER3K: (
        "sha256:6e067606fc04162c8765b288982651773aa28c6f72390fe83428a20b9b39e75f",
        "sha256:49be490fed7726e49b92197853f3b2a5504f08dee1617ac6e2cdbfa69d02e5ea",
    ),
}


def _file_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _load_sources(root: Path) -> dict[str, dict[str, Any]]:
    values = {}
    for relative, (logical, file_digest) in SOURCES.items():
        value = json.loads((root / relative).read_text())
        unsigned = {key: item for key, item in value.items() if key != "sha256"}
        if (
            value.get("sha256") != logical
            or task_family_split.canonical_digest(unsigned) != logical
        ):
            raise ValueError(f"{relative} logical digest drift")
        if _file_digest(root / relative) != file_digest:
            raise ValueError(f"{relative} file digest drift")
        values[relative] = value
    return values


def _proven_atom_key(row: dict[str, Any]) -> str:
    family = row.get("lineage", {}).get("task_family")
    if not isinstance(family, str):
        raise ValueError("receipt-proven row omits reviewed task family")
    return shared_atom_lineage.atom_artifact_key(f"{family}:atom_source")


def _teacher3k_exposure(value: dict[str, Any]) -> tuple[set[tuple[str, str]], set[str]]:
    exact = set()
    atoms = set()
    for task in value["training_task_keys"]:
        atoms.update(shared_atom_lineage.atom_artifact_key(item) for item in task["atom_lineages"])
        exact.update((task["task_key"], row["task_version_id"]) for row in task["versions"])
    return exact, atoms


def build(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    values = _load_sources(root)
    proven = values[PROVEN]["task_versions"]
    candidates = values[CANDIDATES]["tasks"]
    live_rows = values[LIVE_LINEAGE]["task_versions"]
    parent_tasks = values[PARENT_SPLIT]["tasks"]
    role_by_identity = {
        (row["task_key"], row["task_version_id"]): row["split"] for row in parent_tasks
    }
    candidate_identities = {(row["task_key"], row["task_version_id"]) for row in candidates}
    live_identities = {(row["task_key"], row["task_version_id"]) for row in live_rows}
    if candidate_identities != live_identities or len(candidate_identities) != 33:
        raise ValueError("live candidate lineage does not exactly cover the frozen roster")
    proven_identities = {(row["task_key"], row["task_version_id"]) for row in proven}
    if proven_identities != set(role_by_identity) or proven_identities & candidate_identities:
        raise ValueError("receipt-proven, split, and candidate identities do not partition cleanly")

    source_rows = [
        {
            "task_key": row["task_key"],
            "task_version_id": row["task_version_id"],
            "atom_artifact_keys": [_proven_atom_key(row)],
        }
        for row in proven
    ] + [
        {
            "task_key": row["task_key"],
            "task_version_id": row["task_version_id"],
            "atom_artifact_keys": row["atom_artifact_keys"],
        }
        for row in live_rows
    ]
    component_graph = shared_atom_lineage.build_components(source_rows)
    inherited_roles = shared_atom_lineage.component_roles(component_graph, role_by_identity)
    component_by_identity = {
        (row["task_key"], row["task_version_id"]): row["component_id"]
        for row in component_graph["task_versions"]
    }
    teacher_exact, teacher_atoms = _teacher3k_exposure(values[TEACHER3K])
    live_by_identity = {(row["task_key"], row["task_version_id"]): row for row in live_rows}

    components = []
    candidate_component_ids = set()
    candidate_overlap_with_proven = 0
    for component in component_graph["components"]:
        members = []
        sources = Counter()
        for identity in component["task_versions"]:
            key = (identity["task_key"], identity["task_version_id"])
            source = "receipt_proven" if key in proven_identities else "qa_candidate_unqualified"
            sources[source] += 1
            members.append({**identity, "source": source})
        candidate_count = sources["qa_candidate_unqualified"]
        proven_count = sources["receipt_proven"]
        if candidate_count:
            candidate_component_ids.add(component["component_id"])
            candidate_overlap_with_proven += int(proven_count > 0)
        components.append(
            {
                **component,
                "task_versions": members,
                "source_counts": dict(sorted(sources.items())),
                "inherited_role": inherited_roles.get(component["component_id"]),
            }
        )

    candidate_rows = []
    for identity in sorted(candidate_identities):
        live = live_by_identity[identity]
        atom_overlap = sorted(set(live["atom_artifact_keys"]) & teacher_atoms)
        candidate_rows.append(
            {
                "task_key": identity[0],
                "task_version_id": identity[1],
                "qa_status": live["qa_status"],
                "component_id": component_by_identity[identity],
                "atom_artifact_keys": live["atom_artifact_keys"],
                "teacher3k_exact_version_exposed": identity in teacher_exact,
                "teacher3k_shared_atom_exposed": bool(atom_overlap),
                "teacher3k_shared_atom_keys": atom_overlap,
                "runtime_receipt_qualified": False,
                "admission": "excluded_missing_complete_runtime_receipts",
            }
        )

    component_counts = Counter(len(row["task_versions"]) for row in components)
    census_body = {
        "schema": "fleet_blackbox_shared_atom_lineage_census_v1",
        "qualification_authority_git_commit": values[PARENT_DECISION]["qualification_authority"][
            "git_commit"
        ],
        "sources": {
            path: {"logical_sha256": digest[0], "file_sha256": digest[1]}
            for path, digest in SOURCES.items()
        },
        "grouping": {
            "unit": "transitive closure of shared immutable atom artifact keys",
            "successor_atom_versions_share_artifact_key": True,
            "same_task_key_versions_stay_together": True,
            "composites_union_all_constituent_atoms": True,
        },
        "counts": {
            "exact_task_versions": len(source_rows),
            "receipt_proven_task_versions": len(proven_identities),
            "qa_candidate_task_versions": len(candidate_identities),
            "atom_artifact_keys": len(
                {atom for row in source_rows for atom in row["atom_artifact_keys"]}
            ),
            "components": len(components),
            "candidate_components": len(candidate_component_ids),
            "components_by_task_version_count": {
                str(size): count for size, count in sorted(component_counts.items())
            },
            "candidate_components_overlapping_receipt_proven": candidate_overlap_with_proven,
        },
        "components": sorted(components, key=lambda row: row["component_id"]),
        "candidate_task_versions": candidate_rows,
        "safety": {"launch_authorized": False, "external_mutations": 0},
    }
    census_body["sha256"] = task_family_split.canonical_digest(census_body)

    split_tasks = [
        {
            **row,
            "shared_atom_component_id": component_by_identity[
                (row["task_key"], row["task_version_id"])
            ],
        }
        for row in parent_tasks
    ]
    roles = sorted({row["split"] for row in split_tasks})
    split_body = {
        "schema": "cyber_transitive_atom_lineage_split_v1",
        "status": "no_extension_zero_new_receipt_qualified_candidates",
        "parent": {
            "path": PARENT_SPLIT,
            "sha256": values[PARENT_SPLIT]["sha256"],
        },
        "lineage_census": {"path": LINEAGE_CENSUS, "sha256": census_body["sha256"]},
        "split_unit": "transitive shared-atom component",
        "tasks": sorted(split_tasks, key=lambda row: (row["task_key"], row["task_version_id"])),
        "counts": {
            role: {
                "components": len(
                    {row["shared_atom_component_id"] for row in split_tasks if row["split"] == role}
                ),
                "task_versions": sum(row["split"] == role for row in split_tasks),
            }
            for role in roles
        },
        "candidate_decision": {
            "exact_lineage_bound": len(candidate_rows),
            "new_runtime_receipt_qualified": 0,
            "excluded_missing_complete_runtime_receipts": len(candidate_rows),
            "shared_atom_components": len(candidate_component_ids),
        },
        "leakage_checks": {
            "admitted_component_role_conflicts": 0,
            "candidate_components_overlapping_admitted_components": candidate_overlap_with_proven,
            "all_parent_roles_preserved": True,
            "candidate_task_versions_admitted": 0,
        },
        "safety": {"launch_authorized": False, "external_mutations": 0},
    }
    split_body["sha256"] = task_family_split.canonical_digest(split_body)

    decision_body = {
        "schema": "fleet_blackbox_qa33_lineage_qualification_decision_v1",
        "status": "lineage_bound_no_new_runtime_qualification",
        "lineage_census": {"path": LINEAGE_CENSUS, "sha256": census_body["sha256"]},
        "split": {"path": SPLIT, "sha256": split_body["sha256"]},
        "counts": {
            "candidate_task_versions": len(candidate_rows),
            "exact_live_lineage_bound": len(candidate_rows),
            "runtime_receipt_qualified": 0,
            "teacher3k_exact_version_exposed": sum(
                row["teacher3k_exact_version_exposed"] for row in candidate_rows
            ),
            "teacher3k_shared_atom_exposed": sum(
                row["teacher3k_shared_atom_exposed"] for row in candidate_rows
            ),
            "teacher3k_shared_atom_unexposed": sum(
                not row["teacher3k_shared_atom_exposed"] for row in candidate_rows
            ),
        },
        "qualification_boundary": {
            "checked_in_qa_status_is_not_a_runtime_receipt": True,
            "read_only_task_metadata_proves_lineage_only": True,
            "paid_or_mutating_qualification_performed": False,
            "next_gate": (
                "A later authorized zero-model qualification must prove environment startup, "
                "tool reachability, verifier execution, a finite outcome, and cleanup."
            ),
        },
        "safety": {
            "launch_authorized": False,
            "external_mutations": 0,
            "private_content_persisted": False,
        },
    }
    decision_body["sha256"] = task_family_split.canonical_digest(decision_body)
    return census_body, split_body, decision_body


def _encoded(value: dict[str, Any]) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def write(root: Path) -> None:
    for relative, value in zip((LINEAGE_CENSUS, SPLIT, DECISION), build(root), strict=True):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("x") as stream:
                stream.write(_encoded(value))
        except FileExistsError:
            if path.read_text() != _encoded(value):
                raise ValueError(f"refusing to replace immutable artifact {path}") from None


def check(root: Path) -> None:
    for relative, value in zip((LINEAGE_CENSUS, SPLIT, DECISION), build(root), strict=True):
        if (root / relative).read_text() != _encoded(value):
            raise ValueError(f"{relative} reproduction drift")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if args.write:
        write(args.root)
    else:
        check(args.root)


if __name__ == "__main__":
    main()
