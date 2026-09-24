from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parents[1]
EVIDENCE = ROOT / "docs/evidence/qwen38-fleet-final8-exposure-audit-20260923.json"
SPLIT = ROOT / "configs/data/fleet-blackbox-current-study-split-20260914-v2.json"
INVENTORY = ROOT / "configs/data/fleet-blackbox-current-high-quality-20260914-v1.json"
SELECTION = ROOT / "evals/fleet/configs/opencode-easiest-train100-selection-v2.json"
CAMPAIGN = ROOT / "evals/fleet/configs/q38-glm53-exact-easiest100-pass4-campaign-v1.json"
LEDGER = ROOT / "docs/evidence/qwen38-study/2026-09-06-exact-pass4-ledger-v50.json"
PRESPLIT = ROOT / "configs/data/qwen-blackbox-study-presplit-v1.json"
CHECKPOINT_LEDGER = ROOT / "configs/evaluation/qwen38-checkpoint-eval-ledger-v2.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _digest(value: dict) -> str:
    unsigned = {key: item for key, item in value.items() if key != "sha256"}
    payload = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _id_digest(values: set[str]) -> str:
    payload = ("\n".join(sorted(values)) + "\n").encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def test_audit_is_self_digesting_and_binds_every_cited_repository_file() -> None:
    evidence, split, inventory = _load(EVIDENCE), _load(SPLIT), _load(INVENTORY)
    selection, campaign = _load(SELECTION), _load(CAMPAIGN)
    historical = evidence["historical_execution_evidence"]["historical_model_comparison"]

    assert evidence["sha256"] == _digest(evidence)
    assert evidence["split"]["path"] == str(SPLIT.relative_to(ROOT))
    assert evidence["split"]["file_sha256"] == _file_sha256(SPLIT)
    assert evidence["split"]["logical_sha256"] == split["sha256"]
    assert evidence["split"]["inventory_file_sha256"] == _file_sha256(INVENTORY)
    assert evidence["split"]["inventory_path"] == str(INVENTORY.relative_to(ROOT))
    assert evidence["split"]["inventory_logical_sha256"] == inventory["sha256"]
    assert evidence["split"]["dev_selection_sha256"] == split["evaluation"]["dev"]["sha256"]
    assert evidence["split"]["final_test_selection_sha256"] == (
        split["evaluation"]["final_test"]["sha256"]
    )
    assert historical["selection_file_sha256"] == _file_sha256(SELECTION)
    assert historical["selection_path"] == str(SELECTION.relative_to(ROOT))
    assert historical["selection_sha256"] == selection["selection_sha256"]
    assert historical["campaign_file_sha256"] == _file_sha256(CAMPAIGN)
    assert historical["campaign_path"] == str(CAMPAIGN.relative_to(ROOT))
    assert historical["campaign_id"] == campaign["campaign_id"]
    assert historical["ledger_receipt_file_sha256"] == _file_sha256(LEDGER)
    assert historical["ledger_receipt_path"] == str(LEDGER.relative_to(ROOT))
    assert evidence["scoped_presplit_claim"]["file_sha256"] == _file_sha256(PRESPLIT)
    assert evidence["scoped_presplit_claim"]["path"] == str(PRESPLIT.relative_to(ROOT))
    assert evidence["post_split_lock_audit"]["checkpoint_ledger_file_sha256"] == (
        _file_sha256(CHECKPOINT_LEDGER)
    )
    assert evidence["post_split_lock_audit"]["checkpoint_ledger_path"] == str(
        CHECKPOINT_LEDGER.relative_to(ROOT)
    )


def test_final8_roster_is_exact_unique_certified_and_disjoint() -> None:
    evidence, split, inventory = _load(EVIDENCE), _load(SPLIT), _load(INVENTORY)
    final_rows = evidence["final_test_task_versions"]
    inventory_by_version = {row["task_version_id"]: row for row in inventory["task_versions"]}
    expected = []
    for row in split["evaluation"]["final_test"]["tasks"]:
        source = inventory_by_version[row["task_version_id"]]
        assert source["provenance"]["evidence_class"] == (
            "prior_exact_execution_receipt_current_not_known_broken"
        )
        assert source["provenance"]["certification"]["status"] == "accepted"
        expected.append(
            {
                "task_key": row["task_key"],
                "task_version_id": row["task_version_id"],
                "group_id": row["group_id"],
                "prior_exact_execution_certification_sha256": source["provenance"]
                ["certification"]["receipt_sha256"],
            }
        )

    assert final_rows == expected
    assert len(final_rows) == 8
    assert evidence["historical_execution_evidence"]["all_eight_task_certification"][
        "accepted_prior_exact_execution_certifications"
    ] == len(final_rows)
    for key in ("task_key", "task_version_id", "group_id"):
        assert len({row[key] for row in final_rows}) == 8
    other = [row for row in split["tasks"] if row["split"] != "final_test"]
    assert not {row["task_version_id"] for row in final_rows} & {
        row["task_version_id"] for row in other
    }
    assert not {row["group_id"] for row in final_rows} & {row["group_id"] for row in other}
    assert evidence["split"]["exact_identity_overlap"] == 0
    assert evidence["split"]["reviewed_family_overlap"] == 0


def test_historical_model_execution_counts_are_derived_score_independently() -> None:
    evidence, ledger = _load(EVIDENCE), _load(LEDGER)
    final_ids = {row["task_version_id"] for row in evidence["final_test_task_versions"]}
    cells = [cell for cell in ledger["cells"] if cell["task_version_id"] in final_ids]
    historical = evidence["historical_execution_evidence"]["historical_model_comparison"]

    assert ledger["receipt_sha256"] == historical["ledger_receipt_sha256"]
    assert len(cells) == historical["planned_cells"] == 64
    assert len({cell["task_version_id"] for cell in cells}) == (
        historical["planned_exact_task_versions"]
    ) == 8
    assert Counter(cell["model"] for cell in cells) == {
        "qwen3.8-27b": historical["qwen_cells"],
        "glm-5.3": historical["glm_cells"],
    }
    assert Counter(cell["state"] for cell in cells) == historical["state_counts"]
    accepted_ids = {cell["task_version_id"] for cell in cells if cell["state"] == "accepted"}
    assert len(accepted_ids) == historical["exact_task_versions_with_accepted_cells"] == 2
    assert _id_digest(accepted_ids) == historical["accepted_version_ids_sha256"]
    only_unstarted = {
        task_version_id
        for task_version_id in final_ids
        if {cell["state"] for cell in cells if cell["task_version_id"] == task_version_id}
        == {"unstarted"}
    }
    assert len(only_unstarted) == historical["exact_task_versions_with_only_unstarted_cells"]


def test_scoped_claim_and_post_split_selection_rule_are_not_overstated() -> None:
    evidence, presplit, checkpoint = _load(EVIDENCE), _load(PRESPLIT), _load(CHECKPOINT_LEDGER)
    claim = evidence["scoped_presplit_claim"]
    for key in ("frozen_at", "status", "scope", "historical_global_exposure_claimed"):
        assert claim[key] == presplit[key]
    assert claim["receipt_sha256"] == presplit["sha256"]

    lock = evidence["post_split_lock_audit"]
    binding = checkpoint["comparison_contract"]["fleet_holdout_binding"]
    assert lock["checkpoint_ledger_observed_at"] == checkpoint["observed_at"]
    assert lock["checkpoint_ledger_rule"] == binding["selection_rule"]
    assert binding["final_test_selection_sha256"] == evidence["split"][
        "final_test_selection_sha256"
    ]
    search = lock["repository_search"]
    matches = []
    for root_name in search["roots"]:
        for path in (ROOT / root_name).rglob("*"):
            if not path.is_file() or str(path.relative_to(ROOT)) == search["excluded_path"]:
                continue
            try:
                contains_needle = search["needle"] in path.read_text()
            except UnicodeDecodeError:
                continue
            if contains_needle:
                matches.append(str(path.relative_to(ROOT)))
    assert sorted(matches) == search["matches"]
    assert lock["current_final_selection_terminal_result_found_in_repository"] is False
    assert lock["current_final_results_used_for_checkpoint_selection_found_in_repository"] is False


def test_public_receipt_has_a_closed_schema_and_explicit_integrity_disclosure() -> None:
    evidence = _load(EVIDENCE)
    assert set(evidence) == {
        "schema",
        "observed_at",
        "repository_commit",
        "scope",
        "split",
        "final_test_task_versions",
        "historical_execution_evidence",
        "post_split_lock_audit",
        "scoped_presplit_claim",
        "integrity_disclosure",
        "effects",
        "recommended_prospective_confirmation",
        "disposition",
        "conclusion",
        "sha256",
    }
    assert all(
        set(row)
        == {
            "task_key",
            "task_version_id",
            "group_id",
            "prior_exact_execution_certification_sha256",
        }
        for row in evidence["final_test_task_versions"]
    )
    assert set(evidence["split"]) == {
        "path",
        "file_sha256",
        "logical_sha256",
        "dev_selection_sha256",
        "final_test_selection_sha256",
        "counts",
        "exact_identity_overlap",
        "reviewed_family_overlap",
        "generalization_scope",
        "inventory_path",
        "inventory_file_sha256",
        "inventory_logical_sha256",
        "introduced_by_commit",
        "introduced_at",
    }
    assert set(evidence["split"]["counts"]) == {"train", "dev", "final_test"}
    historical = evidence["historical_execution_evidence"]
    assert set(historical) == {"all_eight_task_certification", "historical_model_comparison"}
    assert set(historical["all_eight_task_certification"]) == {
        "accepted_prior_exact_execution_certifications",
        "evidence_source",
        "boundary",
    }
    assert set(historical["historical_model_comparison"]) == {
        "campaign_id",
        "source_api_job_id",
        "selection_path",
        "selection_file_sha256",
        "selection_sha256",
        "campaign_path",
        "campaign_file_sha256",
        "ledger_receipt_path",
        "ledger_receipt_file_sha256",
        "ledger_receipt_sha256",
        "files_first_committed_by",
        "files_first_committed_at",
        "planned_exact_task_versions",
        "planned_cells",
        "qwen_cells",
        "glm_cells",
        "state_counts",
        "exact_task_versions_with_accepted_cells",
        "accepted_version_ids_sha256",
        "accepted_version_ids_digest_rule",
        "exact_task_versions_with_only_unstarted_cells",
        "boundary",
    }
    assert set(historical["historical_model_comparison"]["state_counts"]) == {
        "accepted",
        "blocked_nonrepeatable",
        "unstarted",
    }
    assert set(evidence["post_split_lock_audit"]) == {
        "checkpoint_ledger_path",
        "checkpoint_ledger_file_sha256",
        "checkpoint_ledger_observed_at",
        "checkpoint_ledger_rule",
        "repository_search",
        "current_final_selection_terminal_result_found_in_repository",
        "current_final_results_used_for_checkpoint_selection_found_in_repository",
        "search_boundary",
    }
    assert set(evidence["post_split_lock_audit"]["repository_search"]) == {
        "needle",
        "roots",
        "excluded_path",
        "matches",
    }
    assert set(evidence["scoped_presplit_claim"]) == {
        "path",
        "file_sha256",
        "receipt_sha256",
        "frozen_at",
        "status",
        "scope",
        "historical_global_exposure_claimed",
        "interpretation",
    }
    recommendation = evidence["recommended_prospective_confirmation"]
    assert set(recommendation) == {
        "preferred_action",
        "fallback_use_of_this_set",
        "matched_design",
        "interpretation",
    }
    assert set(recommendation["matched_design"]) == {
        "arms",
        "tasks",
        "attempts_per_task_per_arm",
        "planned_outcomes",
        "requirements",
    }
    disclosure = evidence["integrity_disclosure"]
    false_fields = {
        "current_final8_task_level_scores_or_rewards_read",
        "current_final8_task_level_scores_or_rewards_recorded",
        "prompts_read",
        "traces_read",
        "outputs_read",
        "flags_or_answers_read",
        "session_content_read",
        "session_ids_recorded",
        "capability_values_used_in_finding",
        "strict_no_score_display_attestation",
    }
    assert all(disclosure[key] is False for key in false_fields)
    assert disclosure["conclusions_are_score_independent"] is True
    assert set(disclosure) == false_fields | {"conclusions_are_score_independent", "reason"}
    serialized = json.dumps(evidence)
    assert '"session_id"' not in serialized
    assert '"score"' not in serialized
    assert '"reward"' not in serialized
    forbidden_keys = {
        "session_id",
        "session_ids",
        "score",
        "scores",
        "score_value",
        "score_values",
        "reward",
        "rewards",
        "reward_value",
        "reward_values",
        "prompt",
        "prompts",
        "trace",
        "traces",
        "trace_excerpt",
        "output",
        "outputs",
        "flag",
        "flags",
        "answer",
        "answers",
        "session_content",
    }

    def nested_keys(value: object) -> set[str]:
        if isinstance(value, dict):
            return set(value) | set().union(*(nested_keys(item) for item in value.values()))
        if isinstance(value, list):
            return set().union(*(nested_keys(item) for item in value))
        return set()

    assert not nested_keys(evidence) & forbidden_keys
    assert evidence["effects"] == {
        "api_mutations": 0,
        "cluster_mutations": 0,
        "model_calls": 0,
        "scoring_calls": 0,
        "sessions_created": 0,
        "results_unsealed": 0,
    }


def test_recommended_confirmation_arithmetic_and_disposition() -> None:
    evidence = _load(EVIDENCE)
    protocol = evidence["recommended_prospective_confirmation"]["matched_design"]
    assert protocol["planned_outcomes"] == (
        protocol["arms"] * protocol["tasks"] * protocol["attempts_per_task_per_arm"]
    )
    assert protocol["arms"] == 6
    assert protocol["tasks"] == 8
    assert protocol["attempts_per_task_per_arm"] == 4
    assert protocol["planned_outcomes"] == 192
    assert evidence["disposition"] == "historically_exposed_locked_confirmation_set"
    assert "not pristine or globally untouched" in evidence["conclusion"]
