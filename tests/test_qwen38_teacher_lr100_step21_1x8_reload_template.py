"""The LR100 step-21 reload stays inert until pair verification is accepted."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from training.io import digest_json, file_sha256

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "configs/qualification/qwen38-teacher-lr100-cosine-layout-dev-v2.json"
TEMPLATE = (
    ROOT
    / "configs/qualification/qwen38-teacher-lr100-cosine-1x8-step21-reload-dev-v1.template.json"
)
VERIFIER = (
    ROOT / "configs/qualification/qwen38-teacher-lr100-step20-step21-seal-verify-dev-v2.pod.yaml"
)
HANDOFF = ROOT / "docs/evidence/qwen38-study/2026-09-14-lr100-training-seal-handoff-v1.json"


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


def test_template_is_self_digesting_inert_and_enumerates_every_null() -> None:
    value = sealed(TEMPLATE)
    assert value["status"] == "blocked_waiting_independent_checkpoint_pair_verification"
    assert value["launchable"] is False
    assert value["cluster_or_api_mutations_performed"] is False
    assert null_paths(value) == set(value["unresolved_bindings"])
    assert len(value["unresolved_bindings"]) == len(set(value["unresolved_bindings"]))
    for path in value["unresolved_bindings"]:
        assert nested(value, path) is None
    assert value["execution"] == {
        "kind": "offline_preparation_only_not_a_job_request",
        "kubernetes_calls_performed": 0,
        "jobs_api_preview_calls_performed": 0,
        "jobs_api_submit_calls_performed": 0,
        "wandb_runs_created": 0,
    }


def test_template_binds_immutable_handoff_and_verifier_artifact() -> None:
    value = sealed(TEMPLATE)
    handoff = sealed(HANDOFF)
    source = read(SOURCE)
    binding = value["source"]
    assert binding["config_file_sha256"] == file_sha256(SOURCE)
    assert binding["config_name"] == source["name"]
    assert binding["output_root"] == source["output_root"]
    assert binding["terminal_handoff"]["file_sha256"] == file_sha256(HANDOFF)
    assert binding["terminal_handoff"]["embedded_sha256"] == handoff["sha256"]
    assert binding["terminal_handoff"]["optimizer_step"] == 21
    assert binding["terminal_handoff"]["resource_release_verified"] is True
    pair = value["checkpoint_pair_verification"]
    assert pair["verifier_artifact_file_sha256"] == file_sha256(VERIFIER)
    assert pair["active_deadline_seconds"] == 3600
    assert pair["gpus"] == 0
    assert pair["read_only_sfs"] is True
    for step in (20, 21):
        item = pair[f"step_{step}"]
        assert item["files"] == 33
        assert item["bytes"] == 324_627_486_731
        assert item["manifest_file_sha256"] is None


def test_reload_transform_preserves_science_and_exact_one_by_eight_layout() -> None:
    value = sealed(TEMPLATE)
    source = read(SOURCE)
    reload = value["reload"]
    candidate = copy.deepcopy(source)
    replacement = reload["source_config_transform"]["replace"]
    candidate["name"] = replacement["name"]
    candidate["output_root"] = replacement["output_root"]
    candidate["wandb"]["run_id"] = replacement["wandb.run_id"]
    candidate["wandb"]["name"] = replacement["wandb.name"]
    del candidate["pause_after_step"]
    candidate["recovery"] = {
        **reload["source_config_transform"]["add"]["recovery"],
        "sha256": "a" * 64,
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
    assert reload["optimizer_steps_authorized"] == 0
    assert (reload["expected_workers"], reload["expected_gpus_per_worker"]) == (1, 8)
    assert reload["expected_world_size"] == 8
    assert source["recipe"]["nodes"] == 1
    assert source["recipe"]["gpus_per_node"] == 8
    assert reload["priority_class"] == "c1"
    assert (reload["expected_queue_priority"], reload["expected_priority_value"]) == (
        "q1",
        10_000,
    )
    assert reload["requeueIfPreempted"] is False


def test_step21_reload_and_every_later_gate_remain_closed() -> None:
    value = sealed(TEMPLATE)
    pair = value["checkpoint_pair_verification"]
    reload = value["reload"]
    scope = value["topology_scope"]
    assert pair["both_source_checkpoints_fully_rehashed"] is None
    assert pair["required_terminal_evidence_file_sha256"] is None
    assert reload["materialized_config_file_sha256"] is None
    assert reload["source_config_transform"]["add"]["recovery"]["sha256"] is None
    assert reload["plan_sha256"] is reload["request_sha256"] is None
    assert reload["pinned_image_preflight_sha256"] is reload["dev_preview_sha256"] is None
    assert reload["jobs_api_post_calls_performed"] == 0
    assert (
        scope["source_layout"]
        == scope["reload_layout"]
        == {
            "nodes": 1,
            "gpus_per_node": 8,
            "world_size": 8,
        }
    )
    assert scope["production_authorization_added"] is False
    assert scope["step20_reload_covered"] is False
