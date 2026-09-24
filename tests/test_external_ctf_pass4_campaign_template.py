from __future__ import annotations

import hashlib
import json
from pathlib import Path

from evals.external_ctf import opencode_scored
from evals.external_ctf.protocol import DEFAULT_PROTOCOL, canonical, digest, load_protocol

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "configs/evaluation/qwen38-external-ctf-pass4-six-arm-v1.template.json"
MATRIX = ROOT / "configs/evaluation/qwen38-top5-multibench-pass4-matrix-20260923-v1.json"


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _load() -> dict:
    return json.loads(TEMPLATE.read_text())


def test_template_is_held_and_binds_the_current_external_protocol() -> None:
    value = _load()
    protocol = load_protocol()
    binding = value["external_protocol"]
    controller = value["controller_contract"]

    assert value["status"].startswith("held_")
    assert value["launch_authorized"] is False
    assert binding["path"] == str(DEFAULT_PROTOCOL.relative_to(ROOT))
    assert binding["file_sha256"] == _sha256(DEFAULT_PROTOCOL)
    assert binding["protocol_sha256"] == protocol["protocol_sha256"]
    assert value["checkpoint_matrix"]["selection_must_predate_external_results"] is True
    assert value["checkpoint_matrix"] == {
        "schema": "cyber_qwen38_top5_multibench_pass4_matrix_v1",
        "path": str(MATRIX.relative_to(ROOT)),
        "file_sha256": _sha256(MATRIX),
        "receipt_sha256": "sha256:" + json.loads(MATRIX.read_bytes())["sha256"],
        "required_selected_checkpoint_count": 5,
        "total_arm_count_with_baseline": 6,
        "selection_must_predate_external_results": True,
        "external_results_may_not_change_selection": True,
    }
    assert value["data_policy"] == "evaluation_only_never_training_tuning_or_checkpoint_selection"
    assert controller["source_sha256"] == _sha256(ROOT / controller["source_path"])
    assert controller["driver_adapter_sha256"] == _sha256(ROOT / controller["driver_adapter_path"])
    assert controller["binding_schema"] == "external_ctf_campaign_bindings_v1"


def test_six_arm_pass4_universe_preserves_official_unavailable_rows() -> None:
    value = _load()
    protocol = load_protocol()
    slots = value["model_slots"]
    matrix = json.loads(MATRIX.read_bytes())
    benchmarks = {row["id"]: row for row in value["benchmarks"]}

    assert len(slots) == 6
    assert [row["role"] for row in slots].count("baseline") == 1
    assert [row["role"] for row in slots].count("selected_checkpoint") == 5
    baseline = next(row for row in slots if row["role"] == "baseline")
    assert baseline["external_protocol_arm"] == "base"
    assert baseline["external_protocol_arm_sha256"] == digest(protocol["arms"]["base"])
    assert [row["matrix_arm_id"] for row in slots] == [row["arm_id"] for row in matrix["arms"]]
    assert [row["matrix_artifact_id"] for row in slots] == [
        row["artifact_id"] for row in matrix["arms"]
    ]
    assert [row["matrix_arm_sha256"] for row in slots] == [digest(row) for row in matrix["arms"]]
    assert value["pass_k"] == 4
    assert value["attempts"] == [1, 2, 3, 4]

    candidate_tasks = 0
    unavailable_tasks = 0
    for benchmark_id, row in benchmarks.items():
        source = protocol["benchmarks"][benchmark_id]
        unavailable = set(source["source_unavailable_task_ids"]) | set(
            source["execution_unavailable_task_ids"]
        )
        assert row["official_task_count"] == source["task_count"]
        assert row["maximum_candidate_task_count"] == source["task_count"] - len(unavailable)
        assert (
            row["maximum_candidate_task_indices"] == source["runtime_qualification"]["task_indices"]
        )
        assert row["maximum_candidate_task_ids"] == [
            task_id for task_id in source["task_ids"] if task_id not in unavailable
        ]
        assert set(row["declared_unavailable_task_ids"]) == unavailable
        assert row["task_ids_sha256"] == source["task_ids_sha256"]
        assert (
            row["qualification_contract_sha256"]
            == source["runtime_qualification"]["contract_sha256"]
        )
        assert row["accepted_qualification_summary_sha256"] is None
        assert row["maximum_candidate_cell_count"] == (
            row["maximum_candidate_task_count"] * len(slots) * value["pass_k"]
        )
        candidate_tasks += row["maximum_candidate_task_count"]
        unavailable_tasks += len(unavailable)

    universe = value["universe"]
    assert universe == {
        "official_task_count": 65,
        "maximum_candidate_task_count": candidate_tasks,
        "declared_unavailable_task_count": unavailable_tasks,
        "arm_count": 6,
        "attempt_count": 4,
        "official_cell_count": 65 * 6 * 4,
        "declared_unavailable_cell_count": unavailable_tasks * 6 * 4,
        "maximum_candidate_cell_count": candidate_tasks * 6 * 4,
        "resolved_launchable_task_count": None,
        "resolved_cell_count": None,
        "target_admission": "accepted_and_released_exact_runtime_qualification_rows_only",
        "no_substitute_tasks": True,
    }
    assert universe["maximum_candidate_cell_count"] == 1464
    assert [row["maximum_candidate_cell_count"] for row in value["benchmarks"]] == [
        960,
        384,
        120,
    ]


def test_template_cannot_silently_change_science_or_evade_deduplication() -> None:
    value = _load()
    requirements = value["model_binding_requirements"]
    cells = value["stable_statistical_cell"]
    dedup = value["deduplication_preflight"]
    gates = value["launch_gates"]

    assert {
        "weights_sha256",
        "matrix_arm_sha256",
        "checkpoint_or_base_manifest_sha256",
        "export_or_base_clone_receipt_sha256",
        "staging_receipt_sha256",
        "serving_registration_receipt_sha256",
        "tokenizer_sha256",
        "chat_template_sha256",
        "served_model",
        "serving_runtime_sha256",
        "live_parity_receipt_sha256",
        "context_length",
        "inference_precision",
        "quantization",
    } <= set(requirements["required_per_arm"])
    assert requirements["new_serving_create_permitted"] is False
    assert (
        value["sampling"]["seed_policy"]
        == "official_protocol_seed_is_null_attempt_identity_is_ordinal"
    )
    assert value["sampling"]["seed_base"] is None
    assert value["retry_policy"]["maximum_replacements_per_cell"] == 0
    assert value["retry_policy"]["pre_model_infrastructure_replacement_permitted"] is False

    identity = set(cells["identity_fields"])
    excluded = set(cells["excluded_from_identity"])
    assert {
        "model.weights_sha256",
        "target.official_task_index",
        "target.version_sha256",
        "attempt",
        "sampling",
    } <= identity
    assert "benchmark.task_set_sha256" not in identity
    assert "evaluation_protocol_sha256" not in identity
    assert cells["campaign_science_bindings_not_cell_identity_fields"] == [
        "benchmark.task_set_sha256",
        "evaluation_protocol_sha256",
    ]
    assert {
        "campaign_id",
        "rollout_driver.source_sha256",
        "score_driver.source_sha256",
        "execution_generation",
    } <= excluded
    assert identity.isdisjoint(excluded)
    assert cells["code_changes_do_not_create_a_duplicate_escape"] is True

    assert dedup["canonical_registry_binding_required"] is True
    assert dedup["canonical_registry_path"] is None
    assert dedup["canonical_registry_receipt_sha256"] is None
    assert dedup["atomic_create_once_canonical_cell_claim_required_before_any_provider_post"]
    assert dedup["canonical_claim_must_use_o_excl_or_equivalent_atomic_semantics"]
    assert dedup["fresh_census_does_not_replace_the_atomic_claim"]
    assert dedup["same_cell_in_a_different_campaign_or_registry_is_still_a_duplicate"] is True
    assert dedup["provider_ambiguous_create_uses_reconciliation_never_a_second_post"] is True
    assert set(dedup["required_sources"]) == {
        "canonical_campaign_registry_claims",
        "external_execution_packet_claims_and_terminals",
        "shared_capacity_reservations_and_releases",
        "provider_inventory_for_exact_owned_names",
        "local_output_roots_and_intents",
    }
    assert gates["all_gates_must_be_true_before_any_create"] is True
    assert all(
        value is False
        for key, value in gates.items()
        if key != "all_gates_must_be_true_before_any_create"
    )


def test_results_are_native_sealed_and_benchmark_separated() -> None:
    value = _load()
    execution = value["execution_policy"]
    reporting = value["reporting"]

    assert [row["scoring_authority"] for row in value["benchmarks"]] == [
        "official_native_deterministic_exploit_grader",
        "project_defined_pinned_manifest_exact_hidden_flag",
        "official_native_exact_flag",
    ]
    assert execution["frozen_deterministic_grading_only"] is True
    assert execution["gpt_scoring_permitted"] is False
    assert execution["score_blind_rollout_execution"] is True
    assert execution["maximum_parallel_external_cells_after_canaries"] == 4
    assert execution["initial_canaries"] == [
        {
            "benchmark": "cvebench_zero_day",
            "task_index": 5,
            "task_id": "CVE-2024-2624",
        },
        {
            "benchmark": "nyu_ctf_web_test",
            "task_index": 6,
            "task_id": "2020f-web-picgram",
        },
        {
            "benchmark": "cybench_web",
            "task_index": 2,
            "task_id": "benchmark/project-sekai-ctf/sekaictf-2023/web/frog-waf",
        },
    ]
    assert reporting["separate_benchmark_aggregates"] is True
    assert reporting["cross_benchmark_primary_score"] is False
    assert reporting["one_shared_baseline_not_repeated_per_pair"] is True
    assert (
        reporting["pass_at_4_validity"]
        == "four_independent_frozen_deterministic_pass1_accepted_outcomes_per_arm_task"
    )
    assert reporting["infrastructure_invalid_attempt_makes_arm_task_pass_at_4_incomplete_not_zero"]
    assert reporting["prompts_traces_flags_answers_private_scores_and_secrets_included"] is False
    assert reporting["results_remain_sealed_until_a_complete_predeclared_matched_aggregate"] is True


def test_scored_adapter_is_exact_and_truthfully_held_for_remote_qualification() -> None:
    value = _load()["scored_adapter"]

    assert value["status"] == "code_complete_remote_credential_isolation_qualification_pending"
    assert value["shared_source_sha256"] == opencode_scored.source_sha256()
    assert value["source_files"] == {
        path: _sha256(ROOT / path)
        for path in (
            "evals/external_ctf/opencode_scored.py",
            "evals/external_ctf/external_proxy.py",
            "evals/fleet/fixed_proxy.py",
        )
    }
    assert value["proxy_image"] == opencode_scored.PROXY_IMAGE
    assert value["native_grading_only"] is True
    assert value["gpt_scoring_permitted"] is False
    assert value["credentials_live_only_in_fixed_proxy"] is True
    assert value["benchmark_containers_receive_provider_credentials"] is False
    assert value["credential_isolation_receipt_sha256"] is None


def test_template_self_digest_is_canonical() -> None:
    value = _load()
    expected = (
        "sha256:"
        + hashlib.sha256(
            canonical({key: item for key, item in value.items() if key != "template_sha256"})
        ).hexdigest()
    )
    assert value["template_sha256"] == expected
