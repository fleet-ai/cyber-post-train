"""Offline contract for the create-once LR1 fragmented-layout fallback."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

from cyber_post_train.jobs import digest
from training.io import digest_json, file_sha256
from training.sft import compile_sft, job_request
from training.sft_runtime import optimizer_schedule

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "configs/qualification/qwen38-teacher-lr1-cosine-layout-dev-v2.json"
FALLBACK = (
    ROOT / "configs/qualification/qwen38-teacher-lr1-cosine-2x4-dev-fallback-v1.json"
)
CADENCE_CONFIG = (
    ROOT / "configs/qualification/qwen38-teacher-checkpoint-cadence-dev-v1.json"
)
CADENCE_EVIDENCE = (
    ROOT
    / "docs/evidence/qwen38-study/2026-09-12-checkpoint-cadence-reload-dev-v1.json"
)
PREPARATION_EVIDENCE = (
    ROOT
    / "docs/evidence/qwen38-study/2026-09-12-lr1-2x4-dev-fallback-preparation-v1.json"
)


def read(path: Path) -> dict:
    return json.loads(path.read_bytes())


def test_fallback_changes_only_identity_topology_and_layout_tag() -> None:
    source, fallback = read(SOURCE), read(FALLBACK)
    restored = copy.deepcopy(fallback)
    restored["name"] = source["name"]
    restored["output_root"] = source["output_root"]
    restored["recipe"]["nodes"] = source["recipe"]["nodes"]
    restored["recipe"]["gpus_per_node"] = source["recipe"]["gpus_per_node"]
    restored["wandb"]["run_id"] = source["wandb"]["run_id"]
    restored["wandb"]["name"] = source["wandb"]["name"]
    restored["wandb"]["tags"] = source["wandb"]["tags"]
    assert restored == source


def test_fallback_has_unique_create_once_identities_and_same_world_size() -> None:
    source, fallback = read(SOURCE), read(FALLBACK)
    assert (source["recipe"]["nodes"], source["recipe"]["gpus_per_node"]) == (1, 8)
    assert (fallback["recipe"]["nodes"], fallback["recipe"]["gpus_per_node"]) == (2, 4)
    assert source["recipe"]["nodes"] * source["recipe"]["gpus_per_node"] == 8
    assert fallback["recipe"]["nodes"] * fallback["recipe"]["gpus_per_node"] == 8

    name = fallback["name"]
    assert re.fullmatch(r"[a-z0-9](?:[-a-z0-9]{0,29}[a-z0-9])?", name)
    assert name == fallback["wandb"]["run_id"] == fallback["wandb"]["name"]
    assert name not in {source["name"], source["wandb"]["run_id"], source["wandb"]["name"]}
    assert fallback["output_root"] != source["output_root"]
    assert not Path(fallback["output_root"]).is_relative_to(source["output_root"])
    assert not Path(source["output_root"]).is_relative_to(fallback["output_root"])
    assert fallback["wandb"]["group"] == source["wandb"]["group"]
    assert "fragmented-dev-layout" in fallback["wandb"]["tags"]
    assert "production-layout" not in fallback["wandb"]["tags"]


def test_fallback_compiles_offline_to_exact_two_by_four_request(tmp_path: Path) -> None:
    config = read(FALLBACK)
    manifest = {
        "tokenizer": {
            "repo": "Qwen/Qwen3.8-27B",
            "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        },
        "split_sha256": "sha256:" + "a" * 64,
        "validation_mode": "task_outcomes_only",
        "fleet_dev_protocol_sha256": config["fleet_dev_protocol_sha256"],
        "files": {
            "train": {
                "path": "train.parquet",
                "rows": 602,
                "sha256": "b" * 64,
                "task_keys": ["synthetic-offline-task"],
            }
        },
    }
    manifest["sha256"] = "sha256:" + digest(manifest)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    config["data"]["manifest"] = str(manifest_path)
    config["data"]["manifest_sha256"] = manifest["sha256"]

    plan = compile_sft(config, relative_to=FALLBACK.parent)
    request = job_request(plan)
    assert plan["recipe"]["max_steps"] == 76
    assert plan["pause_after_step"] == plan["recipe"]["checkpoint_interval"] == 6
    assert plan["recipe"]["lr"] == 1e-6
    assert optimizer_schedule(plan["recipe"]) == {
        "scheduler": "cosine",
        "warmup_ratio": 0.05,
        "num_warmup_steps": 4,
    }
    assert (request["workers"], request["gpus_per_worker"]) == (2, 4)
    assert request["workers"] * request["gpus_per_worker"] == 8
    assert request["priority_class"] == "c1"
    assert request["requeueIfPreempted"] is False
    assert request["secrets"] == ["wandb-api"]
    assert request["name"] == request["title"] == config["name"]
    assert request["run_dir"] == config["output_root"]
    assert request["env"]["WANDB_RUN_ID"] == config["wandb"]["run_id"]


def test_preparation_evidence_is_self_digesting_and_binds_source_and_cadence() -> None:
    proof = read(PREPARATION_EVIDENCE)
    assert proof["sha256"] == digest_json(
        {key: value for key, value in proof.items() if key != "sha256"}
    )
    assert proof["canonical_arm"]["config_file_sha256"] == file_sha256(SOURCE)
    assert proof["fallback"]["config_file_sha256"] == file_sha256(FALLBACK)

    cadence = read(CADENCE_EVIDENCE)
    binding = proof["accepted_2x4_cadence_evidence"]
    assert binding["file_sha256"] == file_sha256(CADENCE_EVIDENCE)
    assert binding["embedded_sha256"] == cadence["sha256"]
    assert binding["source_config_file_sha256"] == file_sha256(CADENCE_CONFIG)
    assert binding["combined_gate_closed_for_bounded_lr_dev_qualification"] is True
    assert cadence["accepted_scope"]["combined_gate_closed_for_bounded_lr_dev_qualification"]
    assert binding["limitations_preserved"] == {
        "operational_only": cadence["accepted_scope"]["operational_only"],
        "full_terminal_provenance_complete": cadence["accepted_scope"][
            "full_terminal_provenance_complete"
        ],
        "source_zero_restarts_verified": cadence["accepted_scope"][
            "source_zero_restarts_verified"
        ],
        "production_training_authorized": cadence["accepted_scope"][
            "production_training_authorized"
        ],
    }
    cadence_config = read(CADENCE_CONFIG)
    assert (cadence_config["recipe"]["nodes"], cadence_config["recipe"]["gpus_per_node"]) == (
        2,
        4,
    )
    assert cadence_config["recipe"]["lr"] != read(FALLBACK)["recipe"]["lr"]


def test_preparation_contract_is_cpu_only_and_does_not_authorize_a_second_producer() -> None:
    proof = read(PREPARATION_EVIDENCE)
    preparation, gate, safety = (
        proof["cpu_preparation"],
        proof["submission_gate"],
        proof["safety"],
    )
    assert preparation["state"] == "not_run_by_this_change"
    assert preparation["network_required"] is preparation["gpu_required"] is False
    assert preparation["prepare_command"][4] == "train"
    assert preparation["prepare_command"][5] == proof["fallback"]["config_path"]
    assert preparation["prepare_command"][-1] == proof["fallback"]["prepared_directory"]
    assert preparation["preflight_command"][-1] == proof["fallback"]["prepared_directory"]
    assert not {"submit", "preview"}.intersection(preparation["prepare_command"])
    assert not {"submit", "preview"}.intersection(preparation["preflight_command"])
    assert gate["authorized_by_this_evidence"] is gate["production_authorized"] is False
    assert "Never submit" in gate["mutual_exclusion"]
    assert "retire" in gate["retire_if"].lower()
    assert safety == {
        "cpu_preparation_performed": False,
        "gpu_training_job_posts": 0,
        "jobs_api_calls": 0,
        "kubernetes_calls": 0,
        "cluster_mutations": 0,
        "canonical_arm_files_modified": False,
        "submission_journals_read_or_modified": False,
        "authentication_material_read_or_modified": False,
        "private_training_records_read": False,
    }
