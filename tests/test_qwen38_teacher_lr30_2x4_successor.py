"""LR30 retires the unadmitted 1x8 attempt before a fresh 2x4 gate."""

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
RETIRED = ROOT / "configs/qualification/qwen38-teacher-lr30-cosine-layout-dev-v2.json"
SUCCESSOR = ROOT / "configs/qualification/qwen38-teacher-lr30-cosine-2x4-dev-v3.json"
EVIDENCE = ROOT / (
    "docs/evidence/qwen38-study/2026-09-12-lr30-dev-v2-queue-contract-retirement-v1.json"
)
V8 = ROOT / "configs/studies/qwen-blackbox-teacher-staged-search-v8.json"
V9 = ROOT / "configs/studies/qwen-blackbox-teacher-staged-search-v9.json"


def read(path: Path) -> dict:
    return json.loads(path.read_bytes())


def sealed(path: Path) -> dict:
    value = read(path)
    assert value["sha256"] == digest_json(
        {key: item for key, item in value.items() if key != "sha256"}
    )
    return value


def synthetic_manifest(config: dict) -> dict:
    value = {
        "tokenizer": {
            "repo": "Qwen/Qwen3.8-27B",
            "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        },
        "split_sha256": "sha256:" + "a" * 64,
        "validation_mode": config["validation_mode"],
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
    value["sha256"] = "sha256:" + digest(value)
    return value


def test_successor_changes_only_create_once_identity_layout_and_layout_tag() -> None:
    retired, successor = read(RETIRED), read(SUCCESSOR)
    restored = copy.deepcopy(successor)
    restored["name"] = retired["name"]
    restored["output_root"] = retired["output_root"]
    restored["recipe"]["nodes"] = retired["recipe"]["nodes"]
    restored["recipe"]["gpus_per_node"] = retired["recipe"]["gpus_per_node"]
    restored["wandb"]["run_id"] = retired["wandb"]["run_id"]
    restored["wandb"]["name"] = retired["wandb"]["name"]
    restored["wandb"]["tags"] = retired["wandb"]["tags"]
    assert restored == retired

    assert (retired["recipe"]["nodes"], retired["recipe"]["gpus_per_node"]) == (1, 8)
    assert (successor["recipe"]["nodes"], successor["recipe"]["gpus_per_node"]) == (
        2,
        4,
    )
    assert "fragmented-dev-layout" in successor["wandb"]["tags"]
    assert "production-layout" not in successor["wandb"]["tags"]
    assert re.fullmatch(r"[a-z0-9](?:[-a-z0-9]{0,29}[a-z0-9])?", successor["name"])
    assert successor["name"] == successor["wandb"]["run_id"] == successor["wandb"]["name"]
    assert successor["name"] != retired["name"]
    assert successor["output_root"] != retired["output_root"]


def test_successor_compiles_to_exact_lr30_two_by_four_six_step_gate(
    tmp_path: Path,
) -> None:
    config = read(SUCCESSOR)
    manifest = synthetic_manifest(config)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    config["data"]["manifest"] = str(manifest_path)
    config["data"]["manifest_sha256"] = manifest["sha256"]

    plan = compile_sft(config, relative_to=SUCCESSOR.parent)
    request = job_request(plan)
    assert plan["recipe"]["max_steps"] == 76
    assert plan["recipe"]["batch_size"] == 8
    assert plan["recipe"]["microbatch_per_gpu"] == 1
    assert plan["recipe"]["lr"] == 3e-5
    assert plan["recipe"]["seed"] == 42
    assert plan["pause_after_step"] == plan["recipe"]["checkpoint_interval"] == 6
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


def test_retirement_evidence_is_self_digesting_and_terminally_reconciled() -> None:
    value = sealed(EVIDENCE)
    retired, successor = value["retired_attempt"], value["fresh_successor"]
    assert retired["config"]["file_sha256"] == file_sha256(RETIRED)
    assert successor["config"]["file_sha256"] == file_sha256(SUCCESSOR)
    assert retired["classification"] == "dev_queue_contract_mismatch_no_runtime_started"
    assert retired["queue_condition"] == {
        "type": "QuotaReserved",
        "status": "False",
        "reason": "Pending",
        "message": (
            'couldn\'t assign flavors to pod set head: Flavor "cpu-head" '
            "does not support TopologyAwareScheduling"
        ),
        "last_transition_time": "2026-09-12T16:30:00Z",
    }
    assert set(retired["execution"].values()) == {0}
    reconciliation = value["terminal_reconciliation"]
    assert reconciliation["jobs_api_delete_http_status"] == 204
    assert reconciliation["jobs_api_exact_get_http_status"] == 404
    assert all(
        reconciliation[key]
        for key in ("rayjob_absent", "workload_absent", "raycluster_absent", "pods_absent")
    )
    assert reconciliation["gpus_held"] == 0
    assert reconciliation["submission_journal_must_never_be_replayed"] is True
    assert value["immutability"] == {
        "retired_config_modified": False,
        "retired_output_reused": False,
        "retired_wandb_identity_reused": False,
        "retired_submission_journal_replayed": False,
        "canonical_scientific_cell_preserved": True,
    }
    assert not any(value["authoring_boundaries"].values())


def test_v9_is_a_sealed_delta_successor_bound_to_retirement_and_v3() -> None:
    old, value, incident = sealed(V8), sealed(V9), sealed(EVIDENCE)
    supersedes = value["supersedes"]
    assert supersedes["path"] == str(V8.relative_to(ROOT))
    assert supersedes["file_sha256"] == file_sha256(V8)
    assert supersedes["embedded_sha256"] == old["sha256"]

    retired = value["retired_lr30_v2"]
    assert retired["config"]["file_sha256"] == file_sha256(RETIRED)
    assert retired["evidence"] == {
        "path": str(EVIDENCE.relative_to(ROOT)),
        "file_sha256": file_sha256(EVIDENCE),
        "embedded_sha256": incident["sha256"],
    }
    assert retired["api_name"] == incident["retired_attempt"]["api_name"]
    assert retired["rayjob_uid"] == incident["retired_attempt"]["rayjob_uid"]
    assert retired["workload_uid"] == incident["retired_attempt"]["workload_uid"]
    assert retired["active"] is retired["accepted"] is False
    assert retired["optimizer_steps"] == retired["gpus_allocated"] == 0
    assert retired["terminally_reconciled_absent"] is True

    successor = value["successor_lr30_two_by_four"]
    config = read(SUCCESSOR)
    assert successor["config"]["file_sha256"] == file_sha256(SUCCESSOR)
    assert successor["identity"] == {
        "name": config["name"],
        "output_root": config["output_root"],
        "wandb_run_id": config["wandb"]["run_id"],
        "wandb_name": config["wandb"]["name"],
    }
    assert successor["topology"] == {"nodes": 2, "gpus_per_node": 4, "world_size": 8}
    assert successor["fresh_gates_verified"] is False
    assert successor["jobs_api_post_authorized_by_this_document"] is False
    assert (
        successor["operational_precedent"]["lr30_runtime_or_scientific_acceptance_transferred"]
        is False
    )


def test_v9_replaces_only_lr30_wave_and_preserves_all_closed_boundaries() -> None:
    old, value = sealed(V8), sealed(V9)
    assert value["inherited_state"] == [
        {"json_pointer": pointer, "preserve_source_value_verbatim": True}
        for pointer in (
            "/accepted_lr1",
            "/fresh_gate_contract",
            "/claim_boundaries",
            "/production",
        )
    ]
    assert value["claim_boundaries"] == old["claim_boundaries"]
    assert value["production"] == old["production"]
    old_lr30, new_lr30 = old["remaining_dev_waves"][0], value["remaining_dev_waves"][0]
    assert old_lr30["cell"]["canonical_cell_id"] == new_lr30["cell"]["canonical_cell_id"]
    assert new_lr30["cell"]["config_path"] == str(SUCCESSOR.relative_to(ROOT))
    assert new_lr30["cell"]["config_file_sha256"] == file_sha256(SUCCESSOR)
    assert new_lr30["cell"]["topology"] == {"nodes": 2, "gpus_per_node": 4, "world_size": 8}
    assert new_lr30["cell"]["fresh_gates_verified"] is False
    assert new_lr30["cell"]["jobs_api_post_authorized_by_this_document"] is False
    assert value["remaining_dev_waves"][1] == old["remaining_dev_waves"][1]
    assert value["invariants"]["retired_lr30_v2_must_never_be_replayed"] is True
    assert value["invariants"]["c1_q1_effective_priority"] == 10_000
    assert value["invariants"]["automatic_requeue"] is False
    assert value["execution"] == {
        "kind": "metadata_only_not_a_job_request",
        "cluster_or_api_mutations_performed": False,
        "new_training_producer_created": False,
        "production_submission_authorized": False,
    }
    gates = " ".join(value["required_fresh_lr30_gates"])
    for phrase in (
        "pinned-image CPU preflight",
        "authenticated dev Jobs API preview",
        "W&B run ID",
        "durable journals",
        "four-GPU workers",
        "one durable pre-POST intent",
        "runtime image IDs",
    ):
        assert phrase in gates
