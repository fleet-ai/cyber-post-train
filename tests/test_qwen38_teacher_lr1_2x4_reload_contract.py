"""LR1 2x4 sealing/reload stays inert and does not waive the 1x8 study gate."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from training.io import digest_json, file_sha256

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "configs/qualification/qwen38-teacher-lr1-cosine-layout-dev-v2.json"
FALLBACK = ROOT / "configs/qualification/qwen38-teacher-lr1-cosine-2x4-dev-fallback-v1.json"
TEMPLATE = ROOT / "configs/qualification/qwen38-teacher-lr1-2x4-reload-dev-v1.template.json"
V6 = ROOT / "configs/studies/qwen-blackbox-teacher-staged-search-v6.json"
V7 = ROOT / "configs/studies/qwen-blackbox-teacher-staged-search-v7.json"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sealed(path: Path) -> dict:
    value = read(path)
    assert value["sha256"] == digest_json(
        {key: item for key, item in value.items() if key != "sha256"}
    )
    return value


def nested(value: dict, path: str):
    current = value
    for part in path.split("."):
        current = current[part]
    return current


def null_paths(value, prefix: str = "") -> set[str]:
    paths = set()
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else key
            paths.update(null_paths(item, child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            paths.update(null_paths(item, f"{prefix}[{index}]"))
    elif value is None:
        paths.add(prefix)
    return paths


def test_reload_template_is_self_digesting_inert_and_enumerates_every_null() -> None:
    value = sealed(TEMPLATE)
    assert value["status"] == "blocked_waiting_step6_checkpoint_seal"
    assert value["launchable"] is False
    assert value["cluster_or_api_mutations_performed"] is False
    assert value["reload"]["jobs_api_post_calls_performed"] == 0
    assert null_paths(value) == set(value["unresolved_bindings"])
    assert len(value["unresolved_bindings"]) == len(set(value["unresolved_bindings"]))
    for path in value["unresolved_bindings"]:
        assert nested(value, path) is None

    reload = value["reload"]
    for key in (
        "config_path",
        "new_run_name",
        "new_output_root",
        "new_wandb_run_id",
        "new_wandb_name",
        "prepared_dir",
    ):
        assert reload[key] is None
        assert key in reload["identity_policy"]["must_be_new_and_mutually_disjoint"]
    assert reload["identity_policy"]["materialize_only_after_seal_is_independently_verified"]


def test_reload_template_binds_exact_source_and_cpu_step6_seal() -> None:
    value, source = sealed(TEMPLATE), read(FALLBACK)
    terminal_evidence = sealed(
        ROOT / "docs/evidence/qwen38-study/2026-09-12-lr1-2x4-dev-terminal-v1.json"
    )
    binding, terminal, seal = (
        value["source"],
        value["source"]["expected_terminal"],
        value["checkpoint_seal"],
    )
    assert binding["config_path"] == str(FALLBACK.relative_to(ROOT))
    assert binding["config_file_sha256"] == file_sha256(FALLBACK)
    assert binding["config_name"] == source["name"]
    assert binding["output_root"] == source["output_root"]
    assert binding["prepared_dir"] == (
        "/mnt/sfs/jobs/chris-q38-study-corpora-v1/lr-dev-v2-qualified-inputs-v1/lr1-2x4-fallback-v1"
    )
    assert binding["plan_sha256"] == (
        "3b554d48bb61eeac041415c951f1bc327f70288aa8e2590246bd5eecb2f04e58"
    )
    assert binding["request_sha256"] == (
        "698dd2804479ad58c19430688b061cd571a64e3324ee2f3eef2cf1c195f79ebd"
    )
    assert binding["api_run_id"] == terminal_evidence["run"]["api_run_id"]
    assert terminal["evidence_path"] == (
        "docs/evidence/qwen38-study/2026-09-12-lr1-2x4-dev-terminal-v1.json"
    )
    assert terminal["evidence_file_sha256"] == file_sha256(ROOT / terminal["evidence_path"])
    assert terminal["evidence_receipt_sha256"] == terminal_evidence["sha256"]
    assert (
        terminal["jobs_api_state"],
        terminal["rayjob_state"],
        terminal["workload_state"],
    ) == ("SUCCEEDED", "SUCCEEDED", "Succeeded")
    assert terminal["optimizer_step"] == terminal["optimizer_steps_executed"] == 6
    assert (
        terminal["file_sha256"] == terminal_evidence["receipts"]["training_paused"]["file_sha256"]
    )
    assert terminal["embedded_receipt_digest_valid"] is True
    assert terminal["wandb_tracking_status"] == "synced"
    assert terminal["wandb_remote_state"] == "finished"
    assert terminal["wandb_local_remote_exact_parity"] is True
    assert "wandb_sync_file_sha256" not in terminal
    assert "wandb_sync_receipt_sha256" not in terminal
    assert terminal["ready_observation_restarts"] == {"head": 0, "worker": 0}
    assert terminal["terminal_final_restart_counts_known"] is False
    assert terminal["zero_restarts_for_full_lifecycle_claimed"] is False
    assert terminal["pod_provenance_exception_preserved"] is True
    assert terminal["resource_release_verified"] is True
    assert source["pause_after_step"] == source["recipe"]["checkpoint_interval"] == 6
    assert seal["optimizer_step"] == 6
    assert seal["world_size"] == source["recipe"]["nodes"] * source["recipe"]["gpus_per_node"]
    assert seal["world_size"] == 8
    assert seal["gpus"] == 0
    assert seal["operation"] == "cpu_only_create_once"
    assert seal["source_checkpoint_path"] == (
        "/mnt/sfs/jobs/chris-q38-ta8-lr1-2x4-dev-v1/checkpoints/global_step_6"
    )
    assert seal["source_checkpoint_receipt_path"] == (
        "/mnt/sfs/jobs/chris-q38-ta8-lr1-2x4-dev-v1/checkpoint_receipts/step-000006.json"
    )
    assert (
        seal["source_checkpoint_receipt_file_sha256"]
        == terminal_evidence["receipts"]["checkpoint_step_6"]["file_sha256"]
    )
    assert seal["source_checkpoint_embedded_receipt_digest_valid"] is True
    assert seal["manifest_path"] == (
        "/mnt/sfs/jobs/chris-q38-study-corpora-v1/lr1-2x4-dev-v1-reload-v1/checkpoint-step-6.json"
    )
    assert seal["command"] == [
        "uv",
        "run",
        "--locked",
        "cyber-post-train",
        "checkpoint-seal",
        binding["prepared_dir"],
        "6",
        "--output",
        seal["manifest_path"],
    ]
    assert seal["pinned_image_digest"] == binding["requested_image_digest"]


def test_future_reload_transform_preserves_science_and_exact_two_by_four_layout() -> None:
    value, source = sealed(TEMPLATE), read(FALLBACK)
    reload = value["reload"]
    candidate = copy.deepcopy(source)
    candidate["name"] = "future-unique-reload"
    candidate["output_root"] = "/mnt/sfs/jobs/future-unique-reload"
    candidate["wandb"]["run_id"] = "future-unique-reload"
    candidate["wandb"]["name"] = "future-unique-reload"
    del candidate["pause_after_step"]
    candidate["recovery"] = {
        "manifest": value["checkpoint_seal"]["manifest_path"],
        "sha256": "a" * 64,
        "mode": "validate",
    }

    restored = copy.deepcopy(candidate)
    restored["name"] = source["name"]
    restored["output_root"] = source["output_root"]
    restored["wandb"]["run_id"] = source["wandb"]["run_id"]
    restored["wandb"]["name"] = source["wandb"]["name"]
    restored["pause_after_step"] = source["pause_after_step"]
    del restored["recovery"]
    assert restored == source

    assert reload["source_config_transform"]["preserve_every_other_field"] is True
    assert reload["source_config_transform"]["remove"] == ["pause_after_step"]
    assert reload["source_config_transform"]["add"]["recovery"]["mode"] == "validate"
    assert reload["optimizer_steps_authorized"] == 0
    assert (reload["expected_workers"], reload["expected_gpus_per_worker"]) == (2, 4)
    assert reload["expected_world_size"] == 8
    assert value["topology_scope"]["source_layout"] == value["topology_scope"]["reload_layout"]
    assert value["topology_scope"]["production_layout_equivalence_claimed"] is False
    assert value["topology_scope"]["other_lr_cells_covered"] is False
    assert reload["priority_class"] == "c1"
    assert (reload["expected_queue_priority"], reload["expected_priority_value"]) == (
        "q1",
        10_000,
    )
    assert reload["requeueIfPreempted"] is False


def test_v7_supersedes_v6_exactly_but_is_metadata_only_and_unresolved() -> None:
    old, value = sealed(V6), sealed(V7)
    assert value["supersedes"]["path"] == str(V6.relative_to(ROOT))
    assert value["supersedes"]["file_sha256"] == file_sha256(V6)
    assert value["supersedes"]["embedded_sha256"] == old["sha256"]
    assert value["launchable"] is False
    assert value["execution"] == {
        "kind": "metadata_only_not_a_job_request",
        "cluster_or_api_mutations_performed": False,
        "reload_config_materialized": False,
        "reload_submitted": False,
        "production_submission_authorized": False,
    }
    assert null_paths(value) == set(value["unresolved_bindings"])
    assert len(value["unresolved_bindings"]) == len(set(value["unresolved_bindings"]))
    assert value["acceptance"]["state"] == "blocked"
    assert value["acceptance"]["lr1_operational_dev_receipt_sha256"] is None


def test_v7_reconciles_literal_one_by_eight_gate_without_claiming_equivalence() -> None:
    old, value = sealed(V6), sealed(V7)
    gate = value["literal_gate_reconciliation"]
    assert gate["v6_required_text"] in old["acceptance"]["required"]
    assert gate["satisfied_by_two_by_four_successor"] is False
    assert gate["topology_equivalence_claimed"] is False
    assert "only the existing LR1 development cell" in gate["replacement_rule"]
    assert "Never cite" in gate["replacement_rule"]
    assert value["scope"] == {
        "exact_predecessor_pointer": "dev_waves[0].cells[0]",
        "canonical_cell_id": "teacher.available-a.lr-1e-6.batch-8.epoch-1.seed-42",
        "cluster": "dev",
        "operational_cell_only": True,
        "other_lr_cells_changed": False,
        "production_changed": False,
        "new_training_producer_authorized": False,
        "new_reload_producer_authorized": False,
    }
    predecessor, successor = value["predecessor_one_by_eight"], value["successor_two_by_four"]
    assert predecessor["topology"] == {"nodes": 1, "gpus_per_node": 8, "world_size": 8}
    assert successor["topology"] == {"nodes": 2, "gpus_per_node": 4, "world_size": 8}
    assert predecessor["execution_reconciliation"]["optimizer_steps"] == 0
    assert predecessor["execution_reconciliation"]["gpus_allocated"] == 0
    assert predecessor["execution_reconciliation"]["must_never_be_replayed"] is True
    assert successor["terminal_proof_scope"] == {
        "jobs_api": "SUCCEEDED",
        "rayjob": "SUCCEEDED",
        "workload": "Succeeded",
        "released_gpus": 8,
        "ready_observation_restarts": {"head": 0, "worker": 0},
        "failure_or_rejection_markers_present": False,
        "terminal_final_restart_counts_known": False,
        "zero_restarts_for_full_lifecycle_claimed": False,
        "pod_provenance_exception_preserved": True,
    }
    assert value["production"] == {
        "state": "blocked",
        "v6_literal_one_by_eight_gate_waived": False,
        "two_by_four_accepted_as_production_layout": False,
        "authorization_added": False,
    }


def test_v7_binds_fallback_transition_terminal_and_inert_reload_template() -> None:
    value, transition, terminal, preparation, template = (
        sealed(V7),
        sealed(ROOT / "docs/evidence/qwen38-study/2026-09-12-lr1-topology-transition-dev-v1.json"),
        sealed(ROOT / "docs/evidence/qwen38-study/2026-09-12-lr1-2x4-dev-terminal-v1.json"),
        sealed(
            ROOT / "docs/evidence/qwen38-study/2026-09-12-lr1-2x4-dev-fallback-preparation-v1.json"
        ),
        sealed(TEMPLATE),
    )
    for name, artifact in (
        ("fallback_preparation", preparation),
        ("topology_transition", transition),
        ("reload_template", template),
    ):
        binding = value["provenance"][name]
        path = ROOT / binding["path"]
        assert binding["file_sha256"] == file_sha256(path)
        assert binding["embedded_sha256"] == artifact["sha256"]

    predecessor, successor = value["predecessor_one_by_eight"], value["successor_two_by_four"]
    assert predecessor["rayjob_uid"] == transition["predecessor"]["rayjob_uid"]
    assert predecessor["workload_uid"] == transition["predecessor"]["workload_uid"]
    assert successor["rayjob_uid"] == transition["successor"]["rayjob_uid"]
    assert successor["workload_uid"] == transition["successor"]["workload_uid"]
    assert successor["prepared_dir"] == transition["successor"]["prepared_artifacts"]["directory"]
    assert successor["plan_sha256"] == transition["successor"]["prepared_artifacts"]["plan_sha256"]
    assert (
        successor["request_sha256"]
        == transition["successor"]["prepared_artifacts"]["request_sha256"]
    )
    assert successor["terminal_evidence_path"] == (
        "docs/evidence/qwen38-study/2026-09-12-lr1-2x4-dev-terminal-v1.json"
    )
    assert successor["terminal_evidence_file_sha256"] == file_sha256(
        ROOT / successor["terminal_evidence_path"]
    )
    assert successor["terminal_evidence_receipt_sha256"] == terminal["sha256"]
    assert (
        successor["terminal_proof_scope"]["released_gpus"]
        == terminal["terminal_controller_reconciliation"]["released_gpus"]
    )
    assert successor["terminal_proof_scope"]["ready_observation_restarts"] == {
        "head": terminal["pod_provenance"]["ready_observation"]["head_pod_restarts"],
        "worker": terminal["pod_provenance"]["ready_observation"]["worker_pod_restarts"],
    }
    assert successor["terminal_proof_scope"]["terminal_final_restart_counts_known"] is False
    assert terminal["pod_provenance"]["zero_restarts_for_full_lifecycle_claimed"] is False
    assert template["source"]["config_path"] == successor["config_path"]
    assert template["source"]["rayjob_uid"] == successor["rayjob_uid"]
    assert template["source"]["workload_uid"] == successor["workload_uid"]
    assert template["topology_scope"]["source_layout"] == successor["topology"]


def test_v7_changes_only_lr1_operational_layout_and_preserves_science() -> None:
    old, value, source, fallback = sealed(V6), sealed(V7), read(SOURCE), read(FALLBACK)
    old_lr1 = old["dev_waves"][0]["cells"][0]
    assert old_lr1["config_path"] == value["predecessor_one_by_eight"]["config_path"]
    assert old_lr1["config_file_sha256"] == value["predecessor_one_by_eight"]["config_file_sha256"]

    restored = copy.deepcopy(fallback)
    restored["name"] = source["name"]
    restored["output_root"] = source["output_root"]
    restored["recipe"]["nodes"] = source["recipe"]["nodes"]
    restored["recipe"]["gpus_per_node"] = source["recipe"]["gpus_per_node"]
    restored["wandb"]["run_id"] = source["wandb"]["run_id"]
    restored["wandb"]["name"] = source["wandb"]["name"]
    restored["wandb"]["tags"] = source["wandb"]["tags"]
    assert restored == source
    assert source["recipe"]["nodes"] * source["recipe"]["gpus_per_node"] == 8
    assert fallback["recipe"]["nodes"] * fallback["recipe"]["gpus_per_node"] == 8
    assert value["scientific_preservation"]["world_size_unchanged"] is True
    assert value["scientific_preservation"]["optimizer_horizon_steps"] == 76
    assert value["scientific_preservation"]["pause_after_optimizer_step"] == 6
    assert value["scientific_preservation"]["checkpoint_step"] == 6
    assert value["scientific_preservation"]["learning_rate"] == 1e-6

    inherited = {cell["id"]: cell for cell in value["unchanged_cells"]}
    old_cells = {
        "lr30": old["dev_waves"][0]["cells"][1],
        "lr100": old["dev_waves"][1]["cells"][0],
    }
    for label, old_cell in old_cells.items():
        binding = inherited[label]
        config = read(ROOT / binding["config_path"])
        assert binding["config_path"] == old_cell["config_path"]
        assert binding["config_file_sha256"] == old_cell["config_file_sha256"]
        assert binding["config_file_sha256"] == file_sha256(ROOT / binding["config_path"])
        assert (config["recipe"]["nodes"], config["recipe"]["gpus_per_node"]) == (1, 8)
        assert binding["topology"] == {"nodes": 1, "gpus_per_node": 8, "world_size": 8}
