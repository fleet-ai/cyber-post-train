"""Reproduce the Sep24 lineage-safe Fleet blackbox split design.

The output is metadata only.  It never reads task prompts, trajectories, scores,
or credentials, and it cannot launch an evaluation or training workload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from training import task_family_split

SCHEMA = "fleet_blackbox_lineage_safe_split_design_v1"
QUALIFICATION_COMMIT = "915b385196842674114a4e9891d4e2cd02613d57"
SEED = "fleet-blackbox-lineage-safe-20260924-v1"
RATIOS = {"train": 50 / 75, "dev": 17 / 75, "final_test": 8 / 75}

INVENTORY = "configs/data/fleet-blackbox-current-production-20260924-v1.json"
PROVEN = "configs/data/fleet-blackbox-receipt-proven-20260924-v1.json"
CANDIDATES = "configs/data/fleet-blackbox-qa-candidates-20260924-v1.json"
COVERAGE = "configs/data/fleet-blackbox-training-coverage-20260924-v1.json"
LEGACY_INVENTORY = "configs/data/fleet-blackbox-current-high-quality-20260914-v1.json"
LEGACY_SPLIT = "configs/data/fleet-blackbox-current-study-split-20260914-v2.json"
TEACHER3K_PROTOCOL = "configs/evaluation/qwen38-teacher3k-fleet-heldout20-pass4-protocol-v2.json"
EVIDENCE_INDEX = "configs/discovery/qwen38-lora-evidence-index-v1.json"
SEP7_TEMPLATE = "configs/runs/qwen38-27b-sft-full.template.json"
FRESH75_MANIFEST = "configs/data/qwen38-fresh75-teacher-sft-train-v1.manifest.json"
FRESH75_FINAL_LOCK = "configs/data/qwen38-fresh75-teacher-sft-final-lock-free-v2.manifest.json"

ROLE_ANCHOR = "configs/data/fleet-blackbox-lineage-role-anchor-20260924-v1.json"
SPLIT = "configs/data/fleet-blackbox-lineage-safe-split-20260924-v1.json"
DESIGN = "docs/evidence/qwen38-study/2026-09-24-fleet-blackbox-lineage-safe-split-design-v1.json"

SOURCE_DIGESTS = {
    INVENTORY: (
        "sha256:3de1271d11769989a4d80b0e329a53744642402a2754544eec1851db7322efe7",
        "sha256:71564b9abfa4d3bd3c47a31e7cc5a4a7acf6406120772478f5e85bc83bde4d30",
    ),
    PROVEN: (
        "sha256:23cfc25af183ff474e914443682e4707162110ddc2aa16103a13d984fe05b83c",
        "sha256:6c211259a29376de195411e5cc182bb6f26c5b8027ab86fec855b44f064b0fd0",
    ),
    CANDIDATES: (
        "sha256:4f61fd78b4d92c933030f026e0137abe199e0157fd72b2f959df0354df489be8",
        "sha256:b3d551863655bc8ad92c76bb1b2b0f87c8bcec812b5c42987fc9c0c28c1a5b2a",
    ),
    COVERAGE: (
        "sha256:acf38af53ef50d35347b7d7618a7fa0b24058a1948a584e3251fb7b4fd5f3139",
        "sha256:800cfcc808e754e060fc467a65675ba74d6f2cd47e953cb6e52e22e18ba5c832",
    ),
    LEGACY_INVENTORY: (
        "sha256:23cfc25af183ff474e914443682e4707162110ddc2aa16103a13d984fe05b83c",
        "sha256:6c211259a29376de195411e5cc182bb6f26c5b8027ab86fec855b44f064b0fd0",
    ),
    LEGACY_SPLIT: (
        "sha256:05b3a8dc90ca93adc9671942d75ecb54ff8d0951b0dd48a087e29ec1e641840c",
        "sha256:28a3dcaf31f14d724def9023d9435681772d5b8b3a8647cacc7f72ea8fc8adcb",
    ),
    TEACHER3K_PROTOCOL: (
        "sha256:2ac58c3ded7ba5c9bacd27dc3a379c726912a3f0cafc3ee542672de730e1fe0e",
        "sha256:a2a0389494fa1bca26911081c1d5f55d235cc52c74c15dcd0d98355b7a4dd106",
    ),
}

FILE_ONLY_DIGESTS = {
    EVIDENCE_INDEX: "sha256:adb6b89328a3d8fb5249fea01f36c67da94ad0a4a2e295d08e5477ae2b9866c9",
    SEP7_TEMPLATE: "sha256:395c483e53750971e3daad7bdd834ff1dc4b570b9dceffd16a2fafbd75fde506",
    FRESH75_MANIFEST: "sha256:f71b0b7549a73a2ca3e33d1dc4cb1d8477d271fe3538dbde14afcb03f812cad1",
    FRESH75_FINAL_LOCK: ("sha256:9e144c94b6ad85dc715e100ac5ae6689d6d385972fdfa378d39d8da1b0ccbed5"),
}

UNRESOLVED_EXACT_LINEAGE = (
    "q38-sep7-original-teacher",
    "q38-dense-teacher-v4-v5",
    "q38-self-sft-step44",
    "q38-available-a-lr-screen",
    "q38-retired-dataminer-v2-lora-grpo",
)

ATTEMPT_BOUNDARIES = {
    "q38-sep7-original-teacher": "exact_trained_families_not_retained",
    "q38-cleanup-sft-recovery-lineage": "non_capability_fixture_out_of_scope",
    "q38-dense-teacher-v4-v5": "exact_trained_families_not_retained",
    "q38-self-sft-step44": "exact_trained_families_not_retained",
    "q38-available-a-lr-screen": "exact_trained_families_not_retained",
    "q38-fresh75-v2-v4": "frozen_train_roles_manifest_bound",
    "q38-teacher3k-262k-lineage": "teacher3k_protocol_clean20_exposed5",
    "q38-teacher3k-two-node-v1-v2": "teacher3k_protocol_clean20_exposed5",
    "q38-retired-dataminer-v2-lora-grpo": "external_exact_lineage_not_retained",
    "q38-cleanup-miles-v1-v7": "no_accepted_project_checkpoint_boundary",
    "q38-cleanup-skyrl-v3-and-engine-diagnostics": "no_accepted_project_checkpoint_boundary",
    "q38-miles-dev-and-prod-reward-canaries": "no_accepted_project_checkpoint_boundary",
    "q38-fti-v1-a1-a3": "no_accepted_project_checkpoint_boundary",
    "maintained-fti-miles-full-weight-reference": "external_reference_not_project_checkpoint",
}


def _file_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _load(root: Path, relative: str) -> dict[str, Any]:
    value = json.loads((root / relative).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{relative} must contain a JSON object")
    return value


def _require_sources(root: Path) -> dict[str, dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    for relative, (logical, file_digest) in SOURCE_DIGESTS.items():
        value = _load(root, relative)
        if (
            value.get("sha256") != logical
            or task_family_split.canonical_digest(
                {key: item for key, item in value.items() if key != "sha256"}
            )
            != logical
        ):
            raise ValueError(f"{relative} logical digest drift")
        if _file_digest(root / relative) != file_digest:
            raise ValueError(f"{relative} file digest drift")
        values[relative] = value
    for relative, digest in FILE_ONLY_DIGESTS.items():
        if _file_digest(root / relative) != digest:
            raise ValueError(f"{relative} file digest drift")
        values[relative] = _load(root, relative)
    return values


def _known_lineage(
    values: dict[str, dict[str, Any]], heldout_keys: set[str], heldout_family_count: int
) -> dict[str, Any]:
    protocol = values[TEACHER3K_PROTOCOL]
    evidence_index = values[EVIDENCE_INDEX]
    known_ids = {row["id"] for row in evidence_index["sft_attempts"]}
    if not set(UNRESOLVED_EXACT_LINEAGE) <= known_ids:
        raise ValueError("known training lineage inventory drift")
    all_attempt_ids = known_ids | {row["id"] for row in evidence_index["rl_attempts"]}
    if set(ATTEMPT_BOUNDARIES) != all_attempt_ids:
        raise ValueError("known training lineage classification is incomplete")

    sep7_keys = set(values[SEP7_TEMPLATE]["data"]["task_keys"])
    fresh75 = values[FRESH75_MANIFEST]
    fresh75_final = values[FRESH75_FINAL_LOCK]
    if fresh75.get("split_sha256") != values[LEGACY_SPLIT]["sha256"]:
        raise ValueError("fresh75 corpus split binding drift")
    if (
        fresh75_final.get("catalog_provenance", {}).get("study_split_sha256")
        != values[LEGACY_SPLIT]["sha256"]
    ):
        raise ValueError("fresh75 final-lock split binding drift")

    return {
        "inventory": {
            "source": EVIDENCE_INDEX,
            "file_sha256": FILE_ONLY_DIGESTS[EVIDENCE_INDEX],
            "sft_attempt_ids": sorted(known_ids),
            "rl_attempt_ids": sorted(row["id"] for row in evidence_index["rl_attempts"]),
            "attempt_boundaries": [
                {"attempt_id": attempt_id, "lineage_boundary": ATTEMPT_BOUNDARIES[attempt_id]}
                for attempt_id in sorted(ATTEMPT_BOUNDARIES)
            ],
        },
        "checkpoint_specific_results": {
            "teacher3k": {
                "protocol_path": TEACHER3K_PROTOCOL,
                "protocol_sha256": protocol["sha256"],
                "clean_heldout_families": protocol["classification"]["clean"],
                "exposed_heldout_families": protocol["classification"]["exposed"],
                "exposed_training_aliases": protocol["classification"]["exposed_training_aliases"],
                "status": "20_checkpoint_specific_lineage_clean",
            },
            "fresh75": {
                "corpus_sha256": fresh75["sha256"],
                "final_lock_corpus_sha256": fresh75_final["sha256"],
                "frozen_role_heldout_families": heldout_family_count,
                "status": "manifest_bound_to_frozen_train_roles",
            },
            "sep7_original_teacher": {
                "candidate_key_overlap_with_current_heldout": len(sep7_keys & heldout_keys),
                "actual_trained_family_identities": "not_retained",
                "status": "not_certified_for_current_heldout",
            },
        },
        "global_historical_boundary": {
            "certified_globally_untouched_families": 0,
            "status": "not_certifiable_from_current_public_evidence",
            "unresolved_exact_lineage_attempts": list(UNRESOLVED_EXACT_LINEAGE),
            "reason": (
                "At least one accepted historical corpus lacks a complete exact task-family "
                "identity list, so absence across every historical training lineage cannot be "
                "proven."
            ),
        },
    }


def build(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    values = _require_sources(root)
    legacy_inventory = values[LEGACY_INVENTORY]
    proven = values[PROVEN]
    if legacy_inventory != proven:
        raise ValueError("Sep24 receipt-proven inventory is not the frozen 75-task inventory")

    legacy_split = values[LEGACY_SPLIT]
    anchor = task_family_split.freeze_study_v2_role_anchor(
        legacy_split,
        legacy_inventory_path=root / LEGACY_INVENTORY,
        legacy_inventory_display_path=Path(LEGACY_INVENTORY),
    )
    split = task_family_split.build_anchored(
        proven["task_versions"],
        inventory_sha256=proven["sha256"],
        role_anchor=anchor,
        seed=SEED,
        ratios=RATIOS,
        max_group_task_version_fraction=0.2,
    )
    task_family_split.validate(split, proven["task_versions"], role_anchor=anchor)

    heldout_rows = [row for row in split["tasks"] if row["split"] != "train"]
    heldout_keys = {row["task_key"] for row in heldout_rows}
    heldout_families = {row["group_id"] for row in heldout_rows}
    coverage = values[COVERAGE]["counts"]
    qa_status_counts = Counter(row["qa_status"] for row in values[INVENTORY]["tasks"])
    qa_count = coverage["qa_cleared_pending_receipts"]
    target_total = 300
    optimistic_total = coverage["exact_receipt_proven"] + qa_count
    optimistic_heldout = round(optimistic_total / 3)
    design = {
        "schema": SCHEMA,
        "status": "no_launch_insufficient_qualified_population_for_heldout_100",
        "qualification_authority": {
            "git_commit": QUALIFICATION_COMMIT,
            "sources": {
                path: {"logical_sha256": digests[0], "file_sha256": digests[1]}
                for path, digests in SOURCE_DIGESTS.items()
                if path in {INVENTORY, PROVEN, CANDIDATES, COVERAGE}
            },
        },
        "current_prospective_split": {
            "role_anchor_path": ROLE_ANCHOR,
            "role_anchor_sha256": anchor["sha256"],
            "split_path": SPLIT,
            "split_sha256": split["sha256"],
            "counts": split["counts"],
            "heldout_families": len(heldout_families),
            "newly_qualified_families_since_anchor": split["anchor_audit"]["new_group_count"],
            "use": (
                "Prospective only: future training may use train; dev/final_test remain excluded. "
                "Existing checkpoints require a separate lineage audit."
            ),
        },
        "population_decision": {
            "current_inventory_task_versions": coverage["current_production_blackbox"],
            "exact_receipt_proven_task_versions": coverage["exact_receipt_proven"],
            "known_broken_task_versions": qa_status_counts["broken_task"],
            "qa_cleared_but_unproven_task_versions": qa_count,
            "not_analyzed_without_exact_receipt": coverage["qa_not_analyzed_without_exact_receipt"],
            "heldout_100_supported_now": False,
            "reason": (
                "Only 75 exact task versions are receipt-proven and all 75 already have immutable "
                "roles. QA status without runtime, verifier, outcome, cleanup, and lineage "
                "receipts does not admit a task."
            ),
        },
        "expansion_gate": {
            "target": {"train": 200, "dev": 68, "final_test": 32, "heldout": 100},
            "minimum_total_distinct_qualified_families": target_total,
            "minimum_additional_distinct_qualified_families": target_total
            - coverage["exact_receipt_proven"],
            "qa_candidate_upper_bound_if_all_distinct_and_qualified": {
                "total_families": optimistic_total,
                "heldout_families": optimistic_heldout,
                "shortfall_to_100": 100 - optimistic_heldout,
            },
            "required_before_extension": [
                "exact good-environment, verifier, finite-outcome, and cleanup receipts",
                "exact (task_key, task_version_id) binding",
                "reviewed application and task-family binding for every exact version",
                (
                    "transitive grouping by shared reviewed atom identity, including composites "
                    "and successor atom versions"
                ),
                "family-level exclusion against every training corpus that will be evaluated",
                "anchored allocation that never moves an existing family role",
            ],
        },
        "balance": {
            "grouping_unit": split["split_unit"],
            "lineage_safety_scope": {
                "current_population": (
                    "75 single-atom families grouped by reviewed application and task_family"
                ),
                "shared_atom_transitive_closure_implemented": False,
                "extension_status": (
                    "blocked until composite and successor aliases are grouped by shared "
                    "reviewed atom identity"
                ),
            },
            "balanced_dimensions": split["policy"]["balanced_dimensions"],
            "current_representation": split["representation"],
            "role_drift": split["leakage_checks"]["immutable_inherited_role_drift"],
        },
        "known_training_lineage": _known_lineage(values, heldout_keys, len(heldout_families)),
        "safety": {
            "launch_authorized": False,
            "external_mutations": 0,
            "private_content_persisted": False,
            "task_prompts_or_trajectories_read": False,
        },
    }
    design["sha256"] = task_family_split.canonical_digest(design)
    return anchor, split, design


def _encoded(value: dict[str, Any]) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def _write_once_or_equal(path: Path, value: dict[str, Any]) -> None:
    encoded = _encoded(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x") as stream:
            stream.write(encoded)
    except FileExistsError:
        if path.read_text() != encoded:
            raise ValueError(f"refusing to replace immutable artifact {path}") from None


def check(root: Path) -> None:
    expected = build(root)
    for relative, value in zip((ROLE_ANCHOR, SPLIT, DESIGN), expected, strict=True):
        if (root / relative).read_text() != _encoded(value):
            raise ValueError(f"{relative} reproduction drift")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if not args.write:
        check(args.root)
        return
    for relative, value in zip((ROLE_ANCHOR, SPLIT, DESIGN), build(args.root), strict=True):
        _write_once_or_equal(args.root / relative, value)


if __name__ == "__main__":
    main()
