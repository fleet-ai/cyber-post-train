from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).parents[1]
EVIDENCE = ROOT / "docs/evidence/qwen38-teacher3k-step1000-corpus-audit-20260924.json"
MANIFEST = ROOT / "configs/data/qwen38-teacher3k-32k-v1.manifest.json"
MATERIALIZATION = ROOT / "docs/evidence/qwen38-teacher3k-32k-materialization-receipt-20260920.json"
TARGET_AUDIT = ROOT / "docs/evidence/qwen38-teacher3k-32k-target-audit-20260920.json"
LINEAGE = ROOT / "configs/data/qwen38-teacher3k-training-lineage-map-20260924-v1.json"
HELDOUT = ROOT / "configs/evaluation/qwen38-teacher3k-fleet-heldout20-pass4-protocol-v2.json"
RECONCILIATION = ROOT / "docs/QWEN38_TEACHER3K_FLEET_HELDOUT_LINEAGE_RECONCILIATION_2026-09-24.md"


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _digest(value: dict) -> str:
    unsigned = {key: item for key, item in value.items() if key != "sha256"}
    payload = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _pct(value: int | Fraction, total: int | Fraction) -> float:
    return round(float(value * 100 / total), 6)


def _effective_units(values: list[int]) -> float:
    total = sum(values)
    hhi = sum((value / total) ** 2 for value in values)
    return round(1 / hhi, 3)


def _fraction(value: dict) -> Fraction:
    return Fraction(value["numerator"], value["denominator"])


def _assert_ranked_concentration(
    section: dict, values: list[int], *, top_ten_percent: bool = True
) -> None:
    ordered = sorted(values, reverse=True)
    total = sum(ordered)
    assert section["unit_count"] == len(ordered)
    assert section["effective_units_by_token_hhi"] == _effective_units(ordered)
    for count, label in ((1, "top1"), (5, "top5"), (10, "top10")):
        tokens = sum(ordered[:count])
        assert section[f"{label}_tokens"] == tokens
        assert section[f"{label}_token_pct"] == _pct(tokens, total)
    if top_ten_percent:
        count = math.ceil(len(ordered) * 0.1)
        tokens = sum(ordered[:count])
        assert section["top10pct_units_tokens"] == tokens
        assert section["top10pct_units_token_pct"] == _pct(tokens, total)


def test_receipt_self_digest_and_repository_artifact_bindings() -> None:
    evidence = _load(EVIDENCE)
    manifest = _load(MANIFEST)
    materialization = _load(MATERIALIZATION)
    target = _load(TARGET_AUDIT)
    lineage = _load(LINEAGE)
    heldout = _load(HELDOUT)
    bindings = evidence["artifact_bindings"]

    assert evidence["sha256"] == _digest(evidence)

    derived = bindings["derived_32k_manifest"]
    assert derived["path"] == str(MANIFEST.relative_to(ROOT))
    assert derived["file_sha256"] == _file_sha256(MANIFEST)
    assert derived["logical_sha256"] == manifest["sha256"]
    assert derived["train_parquet_sha256"] == manifest["files"]["train"]["sha256"]
    assert (
        bindings["private_source_selection"]["file_sha256"]
        == (manifest["catalog_provenance"]["source_selection_file_sha256"])
    )
    source = bindings["source_manifest"]
    assert source["file_sha256"] == manifest["rechunk_provenance"]["source_manifest_file_sha256"]
    assert source["logical_sha256"] == manifest["rechunk_provenance"]["source_manifest_sha256"]
    assert source["train_parquet_sha256"] == manifest["rechunk_provenance"]["source_train_sha256"]

    receipt = bindings["materialization_receipt"]
    assert receipt["path"] == str(MATERIALIZATION.relative_to(ROOT))
    assert receipt["file_sha256"] == _file_sha256(MATERIALIZATION)
    assert receipt["logical_sha256"] == materialization["sha256"]
    assert materialization["manifest_sha256"] == manifest["sha256"]
    assert materialization["train_parquet_sha256"] == manifest["files"]["train"]["sha256"]

    target_binding = bindings["target_length_audit"]
    assert target_binding["path"] == str(TARGET_AUDIT.relative_to(ROOT))
    assert target_binding["file_sha256"] == _file_sha256(TARGET_AUDIT)
    assert target["source_manifest_sha256"] == source["logical_sha256"]
    assert target["source_train_parquet_sha256"] == source["train_parquet_sha256"]

    lineage_binding = bindings["training_lineage_map"]
    assert lineage_binding["path"] == str(LINEAGE.relative_to(ROOT))
    assert lineage_binding["file_sha256"] == _file_sha256(LINEAGE)
    assert lineage_binding["logical_sha256"] == lineage["sha256"]
    safe = bindings["safe_lineage_metadata"]
    assert safe["file_sha256"] == lineage["source"]["safe_fleet_metadata_file_sha256"]
    assert safe["logical_sha256"] == lineage["source"]["safe_fleet_metadata_logical_sha256"]

    heldout_binding = bindings["corrected_heldout_protocol"]
    assert heldout_binding["path"] == str(HELDOUT.relative_to(ROOT))
    assert heldout_binding["file_sha256"] == _file_sha256(HELDOUT)
    assert heldout_binding["logical_sha256"] == heldout["sha256"]
    assert heldout["source"]["training_lineage_map_file_sha256"] == _file_sha256(LINEAGE)
    assert heldout["source"]["training_lineage_map_sha256"] == lineage["sha256"]

    reconciliation = bindings["lineage_reconciliation_report"]
    assert reconciliation["path"] == str(RECONCILIATION.relative_to(ROOT))
    assert reconciliation["file_sha256"] == _file_sha256(RECONCILIATION)


def test_corpus_target_salvage_duplicate_and_holdout_arithmetic() -> None:
    evidence = _load(EVIDENCE)
    manifest = _load(MANIFEST)
    materialization = _load(MATERIALIZATION)
    target = _load(TARGET_AUDIT)
    corpus = evidence["corpus"]
    composition = evidence["target_composition"]
    provenance = evidence["success_provenance"]
    duplicate = evidence["duplicate_window_audit"]
    holdout = evidence["lineage_and_holdout"]["heldout25_reconciliation"]

    train = manifest["files"]["train"]
    assert (
        corpus["source_sessions"] == train["source_sessions"] == materialization["source_sessions"]
    )
    assert corpus["task_versions"] == materialization["task_versions"]
    assert corpus["derived_rows"] == train["rows"] == materialization["rows"]
    assert corpus["assistant_targets"] == train["assistant_responses"]
    assert (
        corpus["supervised_tokens"]
        == train["supervised_tokens"]
        == materialization["supervised_tokens"]
    )
    assert corpus["source_windows"] == target["source_rows"]
    assert corpus["maximum_single_target_tokens"] == target["maximum_target_tokens"]
    assert corpus["targets_over_16384_tokens"] == target["targets_over_16384"] == 0
    exposure = corpus["step1000_exposure"]
    assert exposure["planned_epoch_token_pct"] == _pct(
        exposure["supervised_tokens"], corpus["supervised_tokens"]
    )

    counts = composition["response_counts"]
    assert sum(counts.values()) == corpus["assistant_targets"]
    for key, value in counts.items():
        assert composition["response_pct"][key] == _pct(value, corpus["assistant_targets"])
    assert composition["final_submit_token_pct"] == _pct(
        composition["final_submit_tokens"], corpus["supervised_tokens"]
    )
    assert (
        composition["final_submit_tokens"] + composition["remaining_non_submit_or_other_tokens"]
        == corpus["supervised_tokens"]
    )
    assert (
        composition["sessions_with_retained_submit"]
        + composition["sessions_without_retained_submit"]
        == corpus["source_sessions"]
    )

    salvage = provenance["lifecycle_prefix_salvage"]
    assert sum(salvage["reason_counts"].values()) == salvage["sessions"]
    assert (
        salvage["sessions_with_retained_submit"] + salvage["sessions_without_retained_submit"]
        == salvage["sessions"]
    )
    assert salvage["session_pct"] == _pct(salvage["sessions"], corpus["source_sessions"])
    assert salvage["retained_token_pct"] == _pct(
        salvage["retained_supervised_tokens"], corpus["supervised_tokens"]
    )

    assert (
        duplicate["single_window_source_sessions"] + duplicate["multi_window_source_sessions"]
        == corpus["source_sessions"]
    )
    assert (
        corpus["source_windows"]
        == corpus["source_sessions"] + duplicate["source_windows_beyond_first_per_session"]
    )
    assert all(value == 0 for value in duplicate["selected_metadata_duplicates"].values())
    assert (
        duplicate["exact_source_window_payload_duplicates_removed"]
        == manifest["whole_source_exclusions"]["exact_window_payload_duplicate"]
    )

    assert holdout["original_families"] == (
        holdout["lineage_exposed_families"] + holdout["lineage_clean_families"]
    )
    assert holdout["lineage_clean_families"] == (
        holdout["lineage_clean_dev_families"] + holdout["lineage_clean_final_test_families"]
    )
    assert holdout["lineage_exposed_families"] == (
        holdout["exposed_dev_families"] + holdout["exposed_final_test_families"]
    )


def test_teacher_and_private_session_concentration_is_closed_and_sanitized() -> None:
    evidence = _load(EVIDENCE)
    corpus = evidence["corpus"]
    teachers = evidence["concentration"]["teacher_models"]
    entries = teachers["entries"]

    assert len(entries) == teachers["unit_count"]
    assert sum(row["sessions"] for row in entries) == corpus["source_sessions"]
    assert sum(row["source_windows"] for row in entries) == corpus["source_windows"]
    assert sum(row["supervised_tokens"] for row in entries) == corpus["supervised_tokens"]
    tokens = sorted((row["supervised_tokens"] for row in entries), reverse=True)
    assert teachers["top1_token_pct"] == _pct(sum(tokens[:1]), sum(tokens))
    assert teachers["top5_token_pct"] == _pct(sum(tokens[:5]), sum(tokens))
    assert teachers["effective_units_by_token_hhi"] == _effective_units(tokens)
    for row in entries:
        assert row["token_pct"] == _pct(row["supervised_tokens"], corpus["supervised_tokens"])

    sessions = evidence["concentration"]["source_sessions"]
    assert sessions["unit_count"] == corpus["source_sessions"]
    assert sessions["top1_tokens"] == sessions["supervised_token_distribution"]["maximum"]
    assert sessions["top1_token_pct"] == _pct(sessions["top1_tokens"], corpus["supervised_tokens"])
    assert sessions["top1_tokens"] <= sessions["top5_tokens"] <= sessions["top10_tokens"]
    assert sessions["top10_tokens"] <= sessions["top10pct_units_tokens"]
    distribution = sessions["supervised_token_distribution"]
    assert list(distribution.values()) == sorted(distribution.values())

    text = EVIDENCE.read_text()
    assert not re.search(
        r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
        text,
    )
    assert "FLAG{" not in text


def test_task_version_lineage_and_application_concentration_recompute() -> None:
    evidence = _load(EVIDENCE)
    lineage = _load(LINEAGE)
    heldout = _load(HELDOUT)
    rows = lineage["training_task_keys"]
    concentration = evidence["concentration"]
    lineage_receipt = evidence["lineage_and_holdout"]

    assert len(rows) == evidence["corpus"]["task_keys"]
    _assert_ranked_concentration(
        concentration["exact_task_keys"], [row["supervised_tokens"] for row in rows]
    )
    versions = [version for row in rows for version in row["versions"]]
    _assert_ranked_concentration(
        concentration["exact_task_versions"],
        [version["supervised_tokens"] for version in versions],
    )
    assert len(versions) == evidence["corpus"]["task_versions"]
    assert Counter(version["lineage_resolution"] for version in versions) == {
        key: value for key, value in lineage_receipt["version_resolution"].items() if value
    }

    atom_to_tasks: dict[str, set[int]] = defaultdict(set)
    for index, row in enumerate(rows):
        for atom in row["atom_lineages"]:
            atom_to_tasks[atom].add(index)
    assert (
        len(atom_to_tasks)
        == concentration["reviewed_atom_lineage_components"]["unique_atom_lineages"]
    )
    assert (
        sum(len(row["atom_lineages"]) for row in rows)
        == concentration["reviewed_atom_lineage_components"]["task_atom_memberships"]
    )
    assert (
        sum(len(row["atom_lineages"]) == 1 for row in rows)
        == lineage_receipt["task_keys_with_one_atom_lineage"]
    )
    assert (
        sum(len(row["atom_lineages"]) > 1 for row in rows)
        == lineage_receipt["task_keys_with_multiple_atom_lineages"]
    )
    assert (
        max(len(row["atom_lineages"]) for row in rows)
        == lineage_receipt["maximum_atom_lineages_per_task_key"]
    )
    assert (
        sum(len(task_indexes) > 1 for task_indexes in atom_to_tasks.values())
        == lineage_receipt["atom_lineages_used_by_multiple_task_keys"]
    )

    parent: dict[tuple[str, object], tuple[str, object]] = {}

    def find(item: tuple[str, object]) -> tuple[str, object]:
        parent.setdefault(item, item)
        if parent[item] != item:
            parent[item] = find(parent[item])
        return parent[item]

    def union(left: tuple[str, object], right: tuple[str, object]) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for index, row in enumerate(rows):
        task = ("task", index)
        find(task)
        for atom in row["atom_lineages"]:
            union(task, ("atom", atom))
    components: dict[tuple[str, object], list[dict]] = defaultdict(list)
    for index, row in enumerate(rows):
        components[find(("task", index))].append(row)
    component_tokens = sorted(
        (sum(row["supervised_tokens"] for row in members) for members in components.values()),
        reverse=True,
    )
    component_receipt = concentration["reviewed_atom_lineage_components"]
    assert len(components) == component_receipt["transitive_component_count"]
    assert (
        _effective_units(component_tokens) == component_receipt["effective_components_by_token_hhi"]
    )
    for count, label in ((1, "top1"), (5, "top5"), (10, "top10")):
        tokens = sum(component_tokens[:count])
        assert component_receipt[f"{label}_tokens"] == tokens
        assert component_receipt[f"{label}_token_pct"] == _pct(tokens, sum(component_tokens))
    assert (
        sum(len(members) > 1 for members in components.values())
        == component_receipt["components_with_multiple_task_keys"]
    )
    assert (
        max(len(members) for members in components.values())
        == component_receipt["maximum_task_keys_in_component"]
    )
    assert (
        max(
            len({atom for row in members for atom in row["atom_lineages"]})
            for members in components.values()
        )
        == component_receipt["maximum_atom_lineages_in_component"]
    )

    missing = lineage_receipt["missing_subject_slug"]
    missing_rows = [row for row in rows if row["subject_slug"] is None]
    assert missing["task_keys"] == len(missing_rows)
    assert missing["source_sessions"] == sum(row["source_sessions"] for row in missing_rows)
    assert missing["supervised_tokens"] == sum(row["supervised_tokens"] for row in missing_rows)
    assert missing["supervised_token_pct"] == _pct(
        missing["supervised_tokens"], evidence["corpus"]["supervised_tokens"]
    )

    fractional: dict[str, dict[str, Fraction]] = {
        key: defaultdict(Fraction) for key in ("tasks", "sessions", "tokens")
    }
    multi_application = 0
    for row in rows:
        applications = {atom.split("/")[2] for atom in row["atom_lineages"]}
        multi_application += len(applications) > 1
        for application in applications:
            fractional["tasks"][application] += Fraction(1, len(applications))
            fractional["sessions"][application] += Fraction(
                row["source_sessions"], len(applications)
            )
            fractional["tokens"][application] += Fraction(
                row["supervised_tokens"], len(applications)
            )
    applications = concentration["applications"]
    assert multi_application == applications["task_keys_spanning_multiple_applications"]
    assert sum(fractional["tasks"].values()) == evidence["corpus"]["task_keys"]
    assert sum(fractional["sessions"].values()) == evidence["corpus"]["source_sessions"]
    assert sum(fractional["tokens"].values()) == evidence["corpus"]["supervised_tokens"]
    for entry in applications["entries"]:
        application = entry["application"]
        assert _fraction(entry["fractional_task_keys"]) == fractional["tasks"][application]
        assert _fraction(entry["fractional_source_sessions"]) == fractional["sessions"][application]
        assert _fraction(entry["fractional_supervised_tokens"]) == fractional["tokens"][application]
        assert entry["token_pct"] == _pct(
            fractional["tokens"][application], evidence["corpus"]["supervised_tokens"]
        )

    classification = heldout["classification"]
    reconciled = lineage_receipt["heldout25_reconciliation"]
    assert classification["original_heldout_tasks"] == reconciled["original_families"]
    assert classification["clean"] == reconciled["lineage_clean_families"]
    assert classification["exposed"] == reconciled["lineage_exposed_families"]
    assert classification["exposed_training_aliases"] == reconciled["exposing_training_aliases"]
    assert classification["exact_task_key_overlap"] == reconciled["exact_task_key_overlap"]
    assert classification["exact_task_version_overlap"] == reconciled["exact_task_version_overlap"]
    assert heldout["selection"]["source_role_counts"] == {
        "dev": reconciled["lineage_clean_dev_families"],
        "final_test": reconciled["lineage_clean_final_test_families"],
    }


def test_defects_hypotheses_privacy_and_no_launch_boundary_are_explicit() -> None:
    evidence = _load(EVIDENCE)
    confirmed = {row["id"] for row in evidence["confirmed_defects"]}
    unproven = {row["id"] for row in evidence["unproven_or_unavailable"]}

    assert confirmed == {
        "anchorless_raw_token_rechunking",
        "training_evaluation_tool_contract_mismatch",
        "salvaged_prefix_inherited_whole_session_success",
        "task_key_only_holdout_exclusion",
        "unbalanced_supervised_token_weighting",
        "historical_manifest_holdout_claim_false",
    }
    assert unproven == {
        "shared_harness_drift",
        "step1000_prefix_concentration_matches_full_corpus",
        "zero_duplicate_derived_rows",
        "aggressive_optimizer_recipe_as_causal_amplifier",
        "global_holdout_pristineness",
        "visible_reasoning_vs_tool_token_split",
    }
    assert not confirmed & unproven

    anchor = next(
        row
        for row in evidence["confirmed_defects"]
        if row["id"] == "anchorless_raw_token_rechunking"
    )["evidence"]
    assert anchor["row_pct"] == _pct(
        anchor["rows_with_first_target_at_clip_offset"], anchor["derived_rows"]
    )
    assert anchor["step1000_clipped_row_pct"] == _pct(
        anchor["step1000_clipped_rows"], anchor["step1000_rows"]
    )

    privacy = evidence["privacy"]
    assert set(privacy.values()) == {False}
    scope = evidence["scope"]
    assert scope["audit_mode"] == "local_read_only_aggregate_metadata"
    assert all(value is False for key, value in scope.items() if key.endswith("_read"))
    assert set(evidence["effects"].values()) == {0}

    replacement = evidence["replacement_state"]
    assert replacement["status"] == "blocked_pending_exact_source_evidence"
    assert replacement["launch_authorized"] is False
    assert replacement["private_data_materialized"] is False
    assert replacement["gpu_jobs_submitted"] == 0
    assert replacement["blocking_evidence_count"] == len(replacement["remaining_required_evidence"])
    source_ref = evidence["artifact_bindings"]["replacement_corpus_plan"]["source_ref"]
    assert re.fullmatch(r"codex/[a-z0-9-]+@[0-9a-f]{40}", source_ref)
