#!/usr/bin/env python3
"""Build the teacher3k lineage map and its corrected Fleet holdout protocol.

The inputs may include private session metadata. Outputs deliberately contain
only task identities, aggregate counts, and reviewed source lineage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).parents[1]
SUBJECT_RE = re.compile(r"(?:^|_)blackbox-([0-9a-f]{24})(?=__|_)")
LEGACY_ALIASES = {
    "wiz-blackbox-client-settings-proto-pollution": (
        "cyber/atoms/fira/client-settings-proto-pollution@1:atom_source"
    ),
    "wiz-blackbox-remote-link-preview-credentialed-ssrf": (
        "cyber/atoms/fira/remote-link-preview-credentialed-ssrf@3:atom_source"
    ),
    "wiz-blackbox-user-search-real-ldap-filter-injection": (
        "cyber/atoms/fira/user-search-ldap-filter-injection@0:atom_source"
    ),
}


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical(value)).hexdigest()


def file_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def logical_digest(value: dict[str, Any]) -> str:
    return digest({key: item for key, item in value.items() if key != "sha256"})


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def repo_path(path: Path) -> str:
    return str(path.relative_to(ROOT))


def write_json(path: Path, value: dict[str, Any]) -> None:
    value["sha256"] = logical_digest(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def legacy_alias(task_key: str) -> str | None:
    for needle, locator in LEGACY_ALIASES.items():
        if needle in task_key:
            return locator
    return None


def variant_kind(task_key: str) -> str:
    if "hint" in task_key:
        return "hinted_alias"
    if task_key.startswith("cysec1-2-cmp-"):
        return "composite_alias"
    if task_key.startswith("cysec1-2-wiz-blackbox_"):
        return "legacy_wiz_alias"
    if "__deepmind_level_" in task_key:
        return "difficulty_alias"
    return "canonical_or_named_task"


def wilson(n: int, p: float = 0.5) -> dict[str, float]:
    z = 1.959963984540054
    denominator = 1 + z * z / n
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    zero_upper = (z * z / n + z * math.sqrt(z * z / (n * n))) / (2 * denominator)
    return {
        "maximum_half_width_at_approximately_50_percent": round(half, 6),
        "zero_success_upper_95_percent": round(zero_upper, 6),
    }


def build(args: argparse.Namespace) -> None:
    split_path = ROOT / args.split
    inventory_path = ROOT / args.inventory
    manifests = [ROOT / item for item in args.manifest]
    source_path = Path(args.source_selection)
    safe_path = Path(args.safe_lineage)
    map_path = ROOT / args.map_output
    protocol_path = ROOT / args.protocol_output

    split, inventory = load(split_path), load(inventory_path)
    manifest_values = [load(path) for path in manifests]
    source_rows, safe = load_jsonl(source_path), load(safe_path)
    if safe.get("sha256") != logical_digest(safe):
        raise ValueError("safe lineage cache self-digest mismatch")

    source_pairs = {(row["task_key"], row["task_version_id"]) for row in source_rows}
    safe_rows = safe["results"]
    safe_pairs = {(row["task_key"], row["task_version_id"]) for row in safe_rows}
    if source_pairs != safe_pairs or any(row.get("error") for row in safe_rows):
        raise ValueError("safe lineage cache does not exactly cover source versions")

    source_by_pair: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    metadata_by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        source_by_pair[(row["task_key"], row["task_version_id"])].append(row)
    for row in safe_rows:
        metadata_by_key[row["task_key"]].append(row)

    version_rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    unresolved: list[tuple[str, str]] = []
    for task_key in sorted(metadata_by_key):
        metadata = metadata_by_key[task_key]
        sibling_atoms = sorted(
            {
                atom["locator"]
                for row in metadata
                for atom in row.get("atom_sources", [])
                if atom.get("locator")
            }
        )
        versions = []
        for row in sorted(metadata, key=lambda item: item["task_version_id"]):
            pair = (task_key, row["task_version_id"])
            atoms = sorted(
                atom["locator"]
                for atom in row.get("atom_sources", [])
                if atom.get("locator")
            )
            resolution = "fleet_api_exact_atom_metadata"
            if not atoms and len(sibling_atoms) == 1:
                atoms = sibling_atoms
                resolution = "exact_task_key_sibling_version"
            if not atoms and (alias := legacy_alias(task_key)):
                atoms = [alias]
                resolution = "reviewed_legacy_wiz_alias"
            if not atoms:
                unresolved.append(pair)
                resolution = "unproven"
            sessions = source_by_pair[pair]
            version = {
                "task_version_id": row["task_version_id"],
                "source_sessions": len(sessions),
                "supervised_tokens": sum(int(item["supervised_tokens"]) for item in sessions),
                "atom_lineages": atoms,
                "lineage_resolution": resolution,
                "task_graph_source_locators": sorted(
                    {row["task_graph_source_locator"]} - {None}
                ),
            }
            versions.append(version)
            version_rows.append({"task_key": task_key, **version})

        subject = SUBJECT_RE.search(task_key)
        key_rows.append(
            {
                "task_key": task_key,
                "variant_kind": variant_kind(task_key),
                "subject_slug": subject.group(1) if subject else None,
                "source_sessions": sum(item["source_sessions"] for item in versions),
                "source_versions": len(versions),
                "supervised_tokens": sum(item["supervised_tokens"] for item in versions),
                "atom_lineages": sorted(
                    {atom for item in versions for atom in item["atom_lineages"]}
                ),
                "lineage_status": (
                    "reviewed" if all(item["lineage_resolution"] != "unproven" for item in versions)
                    else "unproven"
                ),
                "versions": versions,
            }
        )
    if unresolved:
        raise ValueError(f"unproven source lineage remains: {len(unresolved)} versions")

    manifest_task_counts = {
        int(value["catalog_provenance"]["selected_task_keys"]) for value in manifest_values
    }
    if manifest_task_counts != {len(key_rows)}:
        raise ValueError("manifest selected-task count does not match lineage map")
    selection_digests = {
        value["catalog_provenance"]["source_selection_file_sha256"]
        for value in manifest_values
    }
    if len(selection_digests) != 1:
        raise ValueError("32K/64K/96K manifests do not bind one source selection")
    if {
        value["catalog_provenance"]["source_selection_file_sha256"]
        for value in manifest_values
    } != {file_digest(source_path)}:
        raise ValueError("source selection file does not match corpus manifests")
    if any(
        value[section]["held_out_task_families_excluded_across_all_versions"] != 25
        for value in manifest_values
        for section in ("catalog_provenance", "rechunk_provenance")
    ):
        raise ValueError("unexpected historical manifest exclusion claim")

    lineage_map = {
        "schema": "cyber_qwen38_teacher3k_training_lineage_map_v1",
        "purpose": (
            "Exact 496-key and 1,176-version lineage map for the source shared by the "
            "teacher3k 32K, 64K, and 96K corpora. No prompts, answers, traces, flags, "
            "session identifiers, or capability results are retained."
        ),
        "source": {
            "selection_file_sha256": file_digest(source_path),
            "safe_fleet_metadata_file_sha256": file_digest(safe_path),
            "safe_fleet_metadata_logical_sha256": safe["sha256"],
            "corpus_manifests": [
                {
                    "path": repo_path(path),
                    "file_sha256": file_digest(path),
                    "logical_sha256": value["sha256"],
                    "max_length": value["max_length"],
                }
                for path, value in zip(manifests, manifest_values, strict=True)
            ],
        },
        "method": {
            "primary": (
                "Read-only Fleet GET /v1/tasks/{task_key}?version_id={task_version_id}; "
                "retain only task identity and metadata.cyber_subject lineage."
            ),
            "exact_api_metadata_versions": sum(bool(row.get("atom_sources")) for row in safe_rows),
            "exact_key_sibling_versions": sum(
                item["lineage_resolution"] == "exact_task_key_sibling_version"
                for item in version_rows
            ),
            "reviewed_legacy_alias_versions": sum(
                item["lineage_resolution"] == "reviewed_legacy_wiz_alias"
                for item in version_rows
            ),
            "unproven_versions": 0,
            "fallback_task_key_group_rows": sum(
                row["group_id"] == f"task-key:{row['task_key']}" for row in source_rows
            ),
            "fallback_task_key_groups": len({row["group_id"] for row in source_rows}),
            "legacy_aliases": LEGACY_ALIASES,
        },
        "corrected_historical_claim": {
            "claim": "held_out_task_families_excluded_across_all_versions = 25",
            "locations": [
                f"{repo_path(path)}:catalog_provenance"
                for path in manifests
            ]
            + [f"{repo_path(path)}:rechunk_provenance" for path in manifests],
            "status": "false_for_the_bound_source_selection",
            "replacement": "20 lineage-clean, 5 exposed through 6 alias keys",
            "treatment": (
                "Keep immutable historical manifests and checkpoint identity unchanged; "
                "do not use this manifest field as leakage evidence."
            ),
        },
        "counts": {
            "training_task_keys": len(key_rows),
            "training_task_versions": len(version_rows),
            "source_sessions": len(source_rows),
            "reviewed_training_task_keys": sum(
                row["lineage_status"] == "reviewed" for row in key_rows
            ),
            "unproven_training_task_keys": 0,
        },
        "training_task_keys": key_rows,
    }
    write_json(map_path, lineage_map)

    inventory_by_pair = {
        (row["task_key"], row["task_version_id"]): row for row in inventory["task_versions"]
    }
    all_lineage_hits: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in version_rows:
        for atom in row["atom_lineages"]:
            all_lineage_hits[atom].append(
                {
                    "task_key": row["task_key"],
                    "task_version_id": row["task_version_id"],
                    "variant_kind": variant_kind(row["task_key"]),
                    "lineage_resolution": row["lineage_resolution"],
                    "source_sessions": row["source_sessions"],
                    "supervised_tokens": row["supervised_tokens"],
                }
            )

    heldout = []
    for selected in split["tasks"]:
        if selected["split"] == "train":
            continue
        fact = inventory_by_pair[(selected["task_key"], selected["task_version_id"])]
        locator = fact["lineage"]["task_family"] + ":atom_source"
        hits = sorted(
            all_lineage_hits.get(locator, []),
            key=lambda item: (item["task_key"], item["task_version_id"]),
        )
        heldout.append(
            {
                "source_role": selected["split"],
                "application": fact["lineage"]["application"],
                "group_id": selected["group_id"],
                "task_key": selected["task_key"],
                "task_version_id": selected["task_version_id"],
                "reviewed_task_family": fact["lineage"]["task_family"],
                "training_lineage_status": "exposed" if hits else "clean",
                "training_aliases": hits,
            }
        )
    clean = [row for row in heldout if row["training_lineage_status"] == "clean"]
    exposed = [row for row in heldout if row["training_lineage_status"] == "exposed"]
    unproven_heldout = [row for row in heldout if row["training_lineage_status"] == "unproven"]
    if (len(clean), len(exposed), len(unproven_heldout)) != (20, 5, 0):
        raise ValueError("unexpected heldout classification")
    if sum(len(row["training_aliases"]) for row in exposed) != 6:
        raise ValueError("unexpected exposed alias count")
    training_keys = {row["task_key"] for row in key_rows}
    training_versions = {row["task_version_id"] for row in version_rows}
    exact_key_overlap = len(training_keys & {row["task_key"] for row in heldout})
    exact_version_overlap = len(
        training_versions & {row["task_version_id"] for row in heldout}
    )
    if exact_key_overlap or exact_version_overlap:
        raise ValueError("unexpected exact heldout identity overlap")

    clean_tasks = [
        {key: row[key] for key in (
            "source_role", "application", "group_id", "task_key", "task_version_id",
            "reviewed_task_family",
        )}
        for row in clean
    ]
    protocol = {
        "schema": "cyber_fleet_existing_checkpoint_holdout_protocol_v2",
        "status": "corrected_proposal_not_launched",
        "supersedes": {
            "path": "configs/evaluation/qwen38-teacher3k-fleet-heldout25-proposal-v1.json",
            "sha256": "sha256:4f9b8357679d39aed8383429491dcaece10fa13aad4a690d6a177ebbe01c2750",
            "reason": (
                "Rejected before activation: exact task-key comparison missed six alias keys "
                "covering five reviewed heldout atom lineages."
            ),
        },
        "corrects_historical_manifest_claim": {
            "claim": "held_out_task_families_excluded_across_all_versions = 25",
            "affected_manifest_sha256": [value["sha256"] for value in manifest_values],
            "status": "false_for_the_bound_source_selection",
            "replacement": "20 lineage-clean, 5 exposed through 6 alias keys",
            "treatment": (
                "The manifests remain immutable checkpoint provenance. Their exclusion field "
                "must not be used as leakage evidence."
            ),
        },
        "source": {
            "study_split_path": repo_path(split_path),
            "study_split_file_sha256": file_digest(split_path),
            "study_split_sha256": split["sha256"],
            "quality_inventory_path": repo_path(inventory_path),
            "quality_inventory_file_sha256": file_digest(inventory_path),
            "quality_inventory_sha256": inventory["sha256"],
            "training_lineage_map_path": repo_path(map_path),
            "training_lineage_map_file_sha256": file_digest(map_path),
            "training_lineage_map_sha256": load(map_path)["sha256"],
        },
        "classification": {
            "original_heldout_tasks": len(heldout),
            "clean": len(clean),
            "exposed": len(exposed),
            "unproven": len(unproven_heldout),
            "exposed_training_aliases": sum(len(row["training_aliases"]) for row in exposed),
            "exact_task_key_overlap": exact_key_overlap,
            "exact_task_version_overlap": exact_version_overlap,
            "tasks": heldout,
        },
        "selection": {
            "rule": (
                "Include only reviewed dev/final task versions whose exact atom locator is "
                "absent from every resolved source version in the shared teacher3k corpus."
            ),
            "task_family_count": len(clean_tasks),
            "exact_task_version_count": len(clean_tasks),
            "source_role_counts": {
                "dev": sum(row["source_role"] == "dev" for row in clean_tasks),
                "final_test": sum(row["source_role"] == "final_test" for row in clean_tasks),
            },
            "pass_k": 4,
            "planned_rollouts_per_arm": len(clean_tasks) * 4,
            "tasks": clean_tasks,
        },
        "compatibility": {
            "valid_checkpoint_corpora": [value["sha256"] for value in manifest_values],
            "valid_context_lengths": [value["max_length"] for value in manifest_values],
            "scope": (
                "Any checkpoint trained only from one of the bound teacher3k manifests. "
                "Bind the exact checkpoint, serving route, model revision, harness, and budget "
                "in a separate immutable execution plan before launch."
            ),
        },
        "reporting": {
            "primary_unit": "reviewed task family",
            "per_task_metric": "pass@4: one or more verified successes in four attempts",
            "required_strata": ["dev13", "final7"],
            "union20": "descriptive only; do not hide the dev/final strata",
            "matched_comparison": (
                "Use the same 20 exact task versions, four attempts, harness, prompt contract, "
                "tool and token budgets, sampling recipe, infrastructure, and scoring path per arm."
            ),
            "interpretation_boundary": (
                "Training-lineage heldout for the teacher3k corpus, not globally untouched: these "
                "tasks have prior execution certification and some may have other historical use."
            ),
        },
        "uncertainty": {
            "method": "95% Wilson interval across task-family pass@4 outcomes",
            "union20": wilson(20),
            "dev13": wilson(13),
            "final7": wilson(7),
        },
        "effects": {
            "api_mutations": 0,
            "cluster_mutations": 0,
            "eval_launches": 0,
            "model_calls": 0,
            "scoring_calls": 0,
        },
    }
    write_json(protocol_path, protocol)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-selection", required=True)
    parser.add_argument("--safe-lineage", required=True)
    parser.add_argument(
        "--split", default="configs/data/fleet-blackbox-current-study-split-20260914-v2.json"
    )
    parser.add_argument(
        "--inventory", default="configs/data/fleet-blackbox-current-high-quality-20260914-v1.json"
    )
    parser.add_argument(
        "--manifest",
        action="append",
        default=[],
        help="Pass each corpus manifest once (32K, 64K, 96K).",
    )
    parser.add_argument(
        "--map-output",
        default="configs/data/qwen38-teacher3k-training-lineage-map-20260924-v1.json",
    )
    parser.add_argument(
        "--protocol-output",
        default="configs/evaluation/qwen38-teacher3k-fleet-heldout20-pass4-protocol-v2.json",
    )
    args = parser.parse_args()
    if not args.manifest:
        args.manifest = [
            "configs/data/qwen38-teacher3k-32k-v1.manifest.json",
            "configs/data/qwen38-teacher3k-64k-v1.manifest.json",
            "configs/data/qwen38-teacher3k-96k-v1.manifest.json",
        ]
    build(args)


if __name__ == "__main__":
    main()
