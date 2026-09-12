"""The teacher-SFT successor has one producer per cell and serial safety gates."""

from __future__ import annotations

import json
import math
from pathlib import Path

from training.io import digest_json, file_sha256

ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT / "configs/studies/qwen-blackbox-teacher-staged-search-v3.json"
V4 = ROOT / "configs/studies/qwen-blackbox-teacher-staged-search-v4.json"
V5 = ROOT / "configs/studies/qwen-blackbox-teacher-staged-search-v5.json"
CADENCE_RELOAD_EVIDENCE = (
    ROOT / "docs/evidence/qwen38-study/2026-09-12-checkpoint-cadence-reload-dev-v1.json"
)


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sealed(path: Path) -> dict:
    value = read(path)
    assert value["sha256"] == digest_json(
        {key: item for key, item in value.items() if key != "sha256"}
    )
    return value


def test_v5_is_inert_and_binds_the_exact_v4_successor() -> None:
    v4, v5 = sealed(V4), sealed(V5)
    assert v5["study_id"] == "qwen38-blackbox-teacher-staged-search-v5"
    assert v5["status"] == "frozen_nonlaunchable_successor"
    assert v5["supersedes"]["file_sha256"] == file_sha256(V4)
    assert v5["supersedes"]["embedded_sha256"] == v4["sha256"]
    assert v5["execution"] == {
        "kind": "sealed_metadata_only_not_a_job_request",
        "new_jobs_authorized": False,
        "cluster_or_api_mutations_performed": False,
        "all_cells_currently_launchable": False,
        "next_safe_transition": (
            "bind the missing source cadence terminal artifacts, close the combined "
            "cadence gate and accept the split-A base control, then materialize and "
            "preview only the canonical 1e-5 first production cell"
        ),
    }


def test_v5_binds_only_the_independently_accepted_cadence_reload_scope() -> None:
    value = sealed(V5)
    evidence = sealed(CADENCE_RELOAD_EVIDENCE)
    gate = value["opening_gates"]["cadence_training_and_periodic_reload"]

    assert gate["state"] == "partial_reload_accepted_source_terminal_artifacts_unbound"
    assert gate["accepted_reload_evidence"] == {
        "path": str(CADENCE_RELOAD_EVIDENCE.relative_to(ROOT)),
        "file_sha256": file_sha256(CADENCE_RELOAD_EVIDENCE),
        "embedded_sha256": evidence["sha256"],
        "proof_self_digest": evidence["independent_verifier"]["proof_self_digest"],
    }
    assert gate["combined_gate_closed"] is False
    assert gate["accepted_receipt_sha256"] is None
    assert (
        gate["unresolved_source_terminal_evidence"]
        == evidence["source_cadence_run"]["unresolved_terminal_evidence"]
    )
    assert (
        "opening_gates.cadence_training_and_periodic_reload.accepted_receipt_sha256"
        in value["first_production_cell"]["unresolved_dependencies"]
    )


def test_cadence_reload_evidence_is_exact_terminal_and_zero_update() -> None:
    evidence = sealed(CADENCE_RELOAD_EVIDENCE)
    source = evidence["source_cadence_run"]
    reload = evidence["run"]

    assert source["config"]["file_sha256"] == file_sha256(ROOT / source["config"]["path"])
    assert source["api_run_id"] == "35155f25-aa37-4c75-9f8a-18af24ccaa79"
    assert source["rayjob_uid"] == "ae0811e2-ae26-4f3f-b27b-43b1b2a055ba"
    assert source["workload_uid"] == "e797dc9e-ae21-4620-bc16-a49b55c50266"
    assert source["terminal_state"] == {
        "jobs_api": "SUCCEEDED",
        "rayjob": "SUCCEEDED",
        "workload": "Succeeded",
        "released_gpus": 8,
        "raycluster_and_pods_absent": True,
    }
    assert source["admission"] == {
        "requested_priority": "c1",
        "effective_queue_priority": "q1",
        "effective_priority_value": 10_000,
        "requeue_if_preempted": False,
    }
    assert len(source["unresolved_terminal_evidence"]) == 4

    assert reload["config"]["file_sha256"] == file_sha256(ROOT / reload["config"]["path"])
    assert reload["api_run_id"] == "db29f2b4-e03f-4e7d-99dd-c16f7c811931"
    assert reload["rayjob_uid"] == "275e2b3a-7340-4a2e-8837-25f2f1fd9db7"
    assert reload["workload_uid"] == "c7c63eac-4a94-4805-899a-6fef86740554"
    assert reload["terminal_state"] == {
        "jobs_api": "SUCCEEDED",
        "rayjob": "SUCCEEDED",
        "workload": "Succeeded",
        "released_gpus": 8,
    }

    checkpoint = evidence["source_checkpoint"]
    assert checkpoint == {
        "manifest_path": (
            "/mnt/sfs/jobs/chris-q38-study-corpora-v1/ta8-cad20-dev5-reload-v1/"
            "checkpoint-step-20.json"
        ),
        "manifest_file_sha256": (
            "sha256:0c0f5f4fcb24b6eee79706174705f3bd9d568120644601ca8eb46945e2d3929b"
        ),
        "manifest_receipt_sha256": (
            "sha256:0132d4866f0e9180ce8ec97354501ddf5c06a60365eeb774764b294d4b90c9d6"
        ),
        "optimizer_step": 20,
        "world_size": 8,
        "files": 33,
        "bytes": 324_627_486_731,
    }
    terminal = evidence["terminal_evidence"]
    assert terminal["optimizer_step"] == 20
    assert terminal["optimizer_updates_executed"] == 0
    assert terminal["rank_count"] == 8
    assert terminal["sampler_state_restored"] is True
    assert terminal["scheduler_state_restored"] is True
    assert terminal["training_artifacts_written"] == 0
    assert terminal["validation_scope"] == "checkpoint_and_sampler_reload_only_no_ce"
    assert terminal["recovered"]["file_sha256"] == (
        "sha256:87c0cd7ac84e3ef602eb39762e1aec09975ca32faba875658267e60934673199"
    )
    assert terminal["reload_validated"]["file_sha256"] == (
        "sha256:9653eff813c2aa79831f61bc88666233d2a596284ce4d16ad5813ac3efd67bf7"
    )
    assert terminal["wandb_sync"]["file_sha256"] == (
        "sha256:28896894a1fca05bba566f430626cd6e9b90a459b67ecd9e7451b3c267d9d730"
    )

    verifier = evidence["independent_verifier"]
    assert verifier["pod_uid"] == "0a127b39-4e67-48b2-9c28-a8fa1bf6e624"
    assert (verifier["gpus"], verifier["container_exit_code"], verifier["container_restarts"]) == (
        0,
        0,
        0,
    )
    assert verifier["phase"] == "Succeeded"
    assert verifier["deleted_after_validation"] is True
    assert verifier["proof_self_digest"] == (
        "sha256:6cf2c2431db0b2cd7f87368b775d6891242805cf85a1a1b2eb9e342a59560ad4"
    )
    assert evidence["accepted_scope"]["combined_cadence_training_and_reload_gate_closed"] is False


def test_representative_splits_are_exact_lineage_safe_and_share_final_test() -> None:
    value = sealed(V5)
    splits = value["representative_split_bindings"]
    loaded = {}
    for name in ("a", "b"):
        binding = splits[name]
        payload = sealed(ROOT / binding["path"])
        loaded[name] = payload
        assert binding["file_sha256"] == file_sha256(ROOT / binding["path"])
        assert binding["embedded_sha256"] == payload["sha256"]
        assert binding["groups"] == {"train": 59, "dev": 20, "final_test": 10}
        by_split = {
            split: {row["group_id"] for row in payload["tasks"] if row["split"] == split}
            for split in ("train", "dev", "final_test")
        }
        assert not (by_split["train"] & by_split["dev"])
        assert not (by_split["train"] & by_split["final_test"])
        assert not (by_split["dev"] & by_split["final_test"])

    final_binding = splits["shared_final_test"]
    final = sealed(ROOT / final_binding["path"])
    assert final_binding["file_sha256"] == file_sha256(ROOT / final_binding["path"])
    assert final_binding["embedded_sha256"] == final["sha256"]
    expected = set(final["group_ids"])
    assert expected == {
        row["group_id"]
        for payload in loaded.values()
        for row in payload["tasks"]
        if row["split"] == "final_test"
    }
    assert (
        len(
            {row["group_id"] for row in loaded["a"]["tasks"] if row["split"] == "dev"}
            & {row["group_id"] for row in loaded["b"]["tasks"] if row["split"] == "dev"}
        )
        == 5
    )


def test_first_production_cell_counts_as_the_existing_lr_bracket_member_once() -> None:
    v3, v4, v5 = sealed(V3), sealed(V4), sealed(V5)
    first = v5["first_production_cell"]
    inherited = next(stage for stage in v3["stages"] if stage["id"] == "split-a-broad-lr")
    inherited_lr10 = next(arm for arm in inherited["arms"] if arm["lr"] == 1e-5)
    v4_first = v4["production_gate"]["first_arm"]

    assert first["fulfills_inherited_cell"] == (f"{inherited['id']}.{inherited_lr10['id']}")
    assert first["recipe"]["lr"] == v4_first["recipe"]["lr"] == inherited_lr10["lr"]
    assert first["data"]["manifest_sha256"] == v4_first["data"]["manifest_sha256"]
    assert first["counts_toward_lr_bracket"] is True
    assert first["must_not_be_repeated_in_remaining_lr_wave"] is True

    remaining = v5["remaining_lr_production_wave"]["cells"]
    assert {cell["lr"] for cell in remaining} == {1e-6, 3e-5, 1e-4}
    assert first["recipe"]["lr"] not in {cell["lr"] for cell in remaining}
    assert len(v5["lr_outcome_barrier"]["exact_cells"]) == 4
    assert len(set(v5["lr_outcome_barrier"]["exact_cells"])) == 4
    assert v5["lr_outcome_barrier"]["duplicate_count_per_cell"] == 1


def test_each_changed_lr_has_a_matching_dev_gate_before_production() -> None:
    value = sealed(V5)
    qualification = value["remaining_lr_qualification"]
    numeric = read(ROOT / qualification["historical_numeric_evidence"]["path"])
    assert qualification["historical_numeric_evidence"]["file_sha256"] == file_sha256(
        ROOT / qualification["historical_numeric_evidence"]["path"]
    )
    assert numeric["shared_bindings"]["nodes_per_arm"] == 1
    assert numeric["shared_bindings"]["gpus_per_arm"] == 4
    assert numeric["limits"][1].startswith("The runtime used the historical constant")

    dev_cells = [cell for wave in qualification["dev_waves"] for cell in wave["cells"]]
    assert {cell["lr"] for cell in dev_cells} == {1e-6, 3e-5, 1e-4}
    assert max(wave["max_nodes"] for wave in qualification["dev_waves"]) == 2
    assert qualification["accepted_receipts"] == {
        "lr-1e-6": None,
        "lr-3e-5": None,
        "lr-1e-4": None,
    }
    for cell in value["remaining_lr_production_wave"]["cells"]:
        receipt_key = f"lr-{cell['lr']:.0e}".replace("e-0", "e-")
        assert cell["dev_receipt_binding"].endswith(receipt_key)


def test_step_schedule_checkpoint_and_wave_arithmetic_is_explicit() -> None:
    value = sealed(V5)
    first = value["first_production_cell"]
    recipe = first["recipe"]
    assert recipe["max_steps"] == math.ceil(first["data"]["rows"] / 8) == 76
    assert recipe["warmup_steps"] == math.ceil(recipe["max_steps"] * 0.05) == 4
    assert recipe["checkpoint_interval"] == 20

    remaining = value["remaining_lr_production_wave"]
    assert remaining["wave_nodes"] == len(remaining["cells"]) == 3
    assert remaining["common"]["max_steps"] == 76
    assert remaining["common"]["warmup_steps"] == 4
    assert remaining["common"]["checkpoint_interval"] == 20

    later = value["later_stage_order"]
    assert [stage["order"] for stage in later] == [1, 2, 3, 4]
    assert max(stage["wave_nodes"] for stage in later) == 3
    refinement = later[2]
    stress = later[3]
    assert {cell["id"] for cell in refinement["cells"]} == {
        "batch-16-epoch-1",
        "batch-32-epoch-1",
        "batch-8-epoch-2",
    }
    assert [cell["id"] for cell in stress["cells"]] == ["batch-8-epoch-4"]
    expected = {
        "batch-16-epoch-1": (38, 2, 19),
        "batch-32-epoch-1": (19, 1, 9),
        "batch-8-epoch-2": (152, 8, 20),
        "batch-8-epoch-4": (304, 16, 20),
    }
    for cell in refinement["cells"] + stress["cells"]:
        assert (
            cell["max_steps"],
            cell["warmup_steps"],
            cell["max_checkpoint_interval"],
        ) == expected[cell["id"]]
        assert cell["max_checkpoint_interval"] <= max(1, cell["max_steps"] // 2)
    assert "batch-8-epoch-2 accepted checkpoint" in stress["requires"][0]


def test_every_accepted_cell_gets_matched_eval_and_scalar_only_tracking() -> None:
    value = sealed(V5)
    lifecycle = value["per_accepted_training_cell_lifecycle"]
    assert any("80 split-matched Fleet-dev" in item for item in lifecycle)
    assert any("Tensorlake WebExploitBench" in item for item in lifecycle)
    policy = value["telemetry_and_selection"]
    assert policy["training_loss"] == "diagnostic_only"
    assert policy["teacher_reference_cross_entropy"] == "not_computed_or_used"
    assert "only HPO" in policy["fleet_dev"]
    assert "remain sealed" in policy["webexploitbench"]
    for forbidden in ("prompts", "traces", "flags", "answers", "token IDs", "scores"):
        assert forbidden in policy["wandb"]
    capacity = value["capacity_policy"]
    assert capacity["max_active_experiment_nodes"] == 8
    assert capacity["excluded_existing_inference_endpoint_nodes"] == 4
    assert capacity["largest_training_wave_nodes"] == 3
