"""The cadence reload stays inert until the terminal source and seal exist."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

from training.io import digest_json, file_sha256

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = (
    ROOT / "configs/qualification/qwen38-teacher-checkpoint-cadence-reload-dev-v1.template.json"
)


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def nested(value: dict, path: str):
    current = value
    for part in path.split("."):
        current = current[part]
    return current


def load_template() -> dict:
    value = read(TEMPLATE)
    assert value["sha256"] == digest_json(
        {key: item for key, item in value.items() if key != "sha256"}
    )
    return value


def test_cadence_reload_template_is_digest_bound_and_fail_closed() -> None:
    value = load_template()
    assert value["launchable"] is False
    assert value["cluster_or_api_mutations_performed"] is False
    assert value["status"] == "blocked_waiting_source_terminal_and_step20_seal"
    unresolved = value["unresolved_bindings"]
    assert len(unresolved) == len(set(unresolved)) == 15
    assert all(nested(value, path) in (None, False) for path in unresolved)


def test_cadence_reload_binds_exact_source_and_periodic_checkpoint() -> None:
    value = load_template()
    source_path = ROOT / value["source"]["config_path"]
    source = read(source_path)
    assert file_sha256(source_path) == value["source"]["config_file_sha256"]
    assert source["name"] == "chris-q38-ta8-cad20-dev5"
    assert source["output_root"] == value["source"]["output_root"]
    assert (
        source["pause_after_step"] == value["source"]["expected_terminal"]["optimizer_step"] == 21
    )
    assert (
        source["recipe"]["checkpoint_interval"] == value["checkpoint_seal"]["optimizer_step"] == 20
    )
    assert value["checkpoint_seal"]["source_checkpoint_path"].endswith(
        "/checkpoints/global_step_20"
    )
    assert value["checkpoint_seal"]["source_checkpoint_receipt_path"].endswith(
        "/checkpoint_receipts/step-000020.json"
    )


def test_materialized_reload_would_change_only_identity_pause_and_recovery() -> None:
    value = load_template()
    source = read(ROOT / value["source"]["config_path"])
    reload = copy.deepcopy(source)
    reload["name"] = value["reload"]["new_run_name"]
    reload["output_root"] = value["reload"]["new_output_root"]
    reload["wandb"]["run_id"] = value["reload"]["new_wandb_run_id"]
    reload["wandb"]["name"] = value["reload"]["new_wandb_run_id"]
    del reload["pause_after_step"]
    reload["recovery"] = {
        "manifest": value["checkpoint_seal"]["manifest_path"],
        "sha256": "a" * 64,
        "mode": "validate",
    }

    restored = copy.deepcopy(reload)
    restored["name"] = source["name"]
    restored["output_root"] = source["output_root"]
    restored["wandb"]["run_id"] = source["wandb"]["run_id"]
    restored["wandb"]["name"] = source["wandb"]["name"]
    restored["pause_after_step"] = source["pause_after_step"]
    del restored["recovery"]
    assert restored == source

    assert (reload["recipe"]["nodes"], reload["recipe"]["gpus_per_node"]) == (2, 4)
    assert reload["recipe"]["nodes"] * reload["recipe"]["gpus_per_node"] == 8
    assert reload["recipe"]["scheduler"] == "cosine"
    assert reload["recipe"]["warmup_ratio"] == 0.05
    assert reload["recipe"]["checkpoint_interval"] == 20
    assert reload["recovery"]["mode"] == "validate"
    assert "pause_after_step" not in reload
    assert "requeueIfPreempted" not in reload
    assert "queue_priority_class" not in reload["cluster"]
    assert reload["cluster"]["priority"] == "c1"


def test_reload_identities_are_new_create_once_candidates_and_preview_is_exact() -> None:
    value = load_template()
    source, reload = value["source"], value["reload"]
    name = reload["new_run_name"]
    assert re.fullmatch(r"[a-z0-9](?:[-a-z0-9]{0,29}[a-z0-9])?", name)
    assert name == reload["new_wandb_run_id"]
    assert name != source["config_name"]
    assert reload["new_output_root"] != source["output_root"]
    assert not Path(reload["new_output_root"]).is_relative_to(Path(source["output_root"]))
    assert not Path(source["output_root"]).is_relative_to(Path(reload["new_output_root"]))
    assert reload["cluster"] == "dev"
    assert reload["optimizer_steps_authorized"] == 0
    assert (reload["expected_workers"], reload["expected_gpus_per_worker"]) == (2, 4)
    assert reload["expected_world_size"] == 8
    assert (reload["priority_class"], reload["expected_queue_priority"]) == ("c1", "q1")
    assert reload["expected_priority_value"] == 10_000
    assert reload["requeueIfPreempted"] is False


def test_seal_procedure_is_cpu_only_create_once_and_uses_original_prepared_plan() -> None:
    value = load_template()
    seal = value["checkpoint_seal"]
    assert seal["operation"] == "cpu_only_create_once"
    assert seal["world_size"] == 8
    assert seal["command"] == [
        "uv",
        "run",
        "cyber-post-train",
        "checkpoint-seal",
        value["source"]["prepared_dir"],
        "20",
        "--output",
        seal["manifest_path"],
    ]
    assert "optimizer_steps_executed_equals_zero" in value["acceptance"]
    assert "source_checkpoint_inventory_and_hashes_remain_unchanged" in value["acceptance"]
