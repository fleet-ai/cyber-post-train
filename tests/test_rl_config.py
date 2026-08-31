from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from training.io import digest_json
from training.rl_config import (
    TaskCatalogError,
    build_full_rl_config,
    build_treatment_set,
    freeze_task_split,
    verify_task_split,
    verify_treatment_set,
)


def trajectory(task_id: str, task_key: str, split: str) -> dict:
    return {
        "lineage": {
            "eval_task_version_id": task_id,
            "task_key": task_key,
            "task_version": "v0.0.3",
        },
        "environment": {
            "env_key": "cysec1-2-current-gen",
            "version": "v0.0.3",
            "data_key": "commercial",
            "data_version": "v0.0.9",
        },
        "split": split,
    }


def option(task_id: str, task_key: str) -> dict:
    return {
        "key": task_key,
        "task_version_id": task_id,
        "task_version": "8",
        "environment_version_id": "d70c4fe9-70c5-4020-91b1-a23d886a1e22",
        "env_key": "cysec1-2-current-gen",
        "env_version": "v0.0.3",
        "data_key": "commercial",
        "data_version": "v0.0.9",
        "image_url": "private-image",
        "env_variables": {"PLANTED_FLAG": "must-not-survive"},
    }


def snapshot() -> dict:
    rows = [
        trajectory("00000000-0000-4000-8000-000000000001", "train-task", "train"),
        trajectory("00000000-0000-4000-8000-000000000002", "dev-task", "dev"),
        trajectory("00000000-0000-4000-8000-000000000003", "test-task", "test"),
    ]
    options = {row["lineage"]["task_key"]: option(
        row["lineage"]["eval_task_version_id"], row["lineage"]["task_key"]
    ) for row in rows}
    return freeze_task_split(
        rows,
        lambda key: [options[key]],
        source_job_id="job-id",
        dataset_manifest_digest="sha256:dataset",
    )


def template() -> dict:
    return {
        "fleet_env": None,
        "tasks": {"selection_mode": "tasks", "task_keys": [], "task_versions": []},
        "eval": {"task_keys": [], "task_versions": []},
    }


def test_snapshot_drops_catalog_payload_and_has_verifiable_digest() -> None:
    value = snapshot()
    verify_task_split(value)
    serialized = json.dumps(value)
    assert "PLANTED_FLAG" not in serialized
    assert "must-not-survive" not in serialized
    assert "private-image" not in serialized
    assert value["counts"] == {"tasks": 3, "splits": {"dev": 1, "test": 1, "train": 1}}


def test_builder_uses_train_and_dev_only_with_empty_environment_variables() -> None:
    config = build_full_rl_config(template(), snapshot())
    assert [task["task_key"] for task in config["tasks"]["task_versions"]] == ["train-task"]
    assert [task["task_key"] for task in config["eval"]["task_versions"]] == ["dev-task"]
    assert all(task["env_variables"] == {} for task in config["tasks"]["task_versions"])
    assert "test-task" not in json.dumps(config)


def test_builder_rejects_modified_snapshot() -> None:
    value = snapshot()
    value["tasks"][0]["task_key"] = "tampered"
    with pytest.raises(TaskCatalogError, match="digest mismatch"):
        build_full_rl_config(template(), value)


def test_explicit_treatment_set_excludes_only_named_train_task() -> None:
    rows = [
        trajectory("00000000-0000-4000-8000-000000000001", "train-a", "train"),
        trajectory("00000000-0000-4000-8000-000000000004", "train-b", "train"),
        trajectory("00000000-0000-4000-8000-000000000002", "dev-task", "dev"),
    ]
    options = {
        row["lineage"]["task_key"]: option(
            row["lineage"]["eval_task_version_id"], row["lineage"]["task_key"]
        )
        for row in rows
    }
    value = freeze_task_split(
        rows,
        lambda key: [options[key]],
        source_job_id="job-id",
        dataset_manifest_digest="sha256:dataset",
    )
    train = next(task for task in value["tasks"] if task["task_key"] == "train-a")
    treatment = build_treatment_set(
        value,
        [
            {
                "task_version_id": train["task_version_id"],
                "reason": "training_api_preview_422_no_longer_runnable",
                "evidence": "configs/runs/preview.json",
            }
        ],
    )
    assert verify_treatment_set(treatment, value) == {train["task_version_id"]}
    config = build_full_rl_config(template(), value, treatment)
    assert [item["task_key"] for item in config["tasks"]["task_versions"]] == ["train-b"]
    assert len(config["eval"]["task_versions"]) == 1
    assert treatment["counts"] == {
        "intent_to_treat_train_tasks": 2,
        "as_treated_train_tasks": 1,
        "excluded_train_tasks": 1,
    }


def test_treatment_set_rejects_holdout_exclusion_and_tampering() -> None:
    value = snapshot()
    dev = next(task for task in value["tasks"] if task["split"] == "dev")
    with pytest.raises(TaskCatalogError, match="only train"):
        build_treatment_set(
            value,
            [{"task_version_id": dev["task_version_id"], "reason": "bad", "evidence": "x"}],
        )
    train = next(task for task in value["tasks"] if task["split"] == "train")
    treatment = build_treatment_set(
        value,
        [{"task_version_id": train["task_version_id"], "reason": "blocked", "evidence": "x"}],
    )
    treatment["counts"]["as_treated_train_tasks"] = 1
    with pytest.raises(TaskCatalogError, match="digest mismatch"):
        verify_treatment_set(treatment, value)


def test_snapshot_rejects_catalog_contradiction() -> None:
    row = trajectory("00000000-0000-4000-8000-000000000001", "task", "train")
    bad = option(row["lineage"]["eval_task_version_id"], "task")
    bad["data_version"] = "different"
    with pytest.raises(TaskCatalogError, match="contradicts"):
        freeze_task_split(
            [row],
            lambda _key: [bad],
            source_job_id="job-id",
            dataset_manifest_digest="sha256:dataset",
        )


def test_archived_task_falls_back_to_roster_and_exact_environment_catalog() -> None:
    row = trajectory("00000000-0000-4000-8000-000000000001", "archived", "train")
    value = freeze_task_split(
        [row],
        lambda _key: [],
        source_job_id="job-id",
        dataset_manifest_digest="sha256:dataset",
        fetch_environment_versions=lambda _key: [
            {
                "id": "00000000-0000-4000-8000-000000000010",
                "env_key": "cysec1-2-current-gen",
                "version": "v0.0.3",
                "data_key": "commercial",
                "data_version": "v0.0.9",
                "env_variables": {"SECRET": "must-not-survive"},
            }
        ],
    )
    task = value["tasks"][0]
    assert task["task_version_id"] == row["lineage"]["eval_task_version_id"]
    assert task["environment_version_id"] == "00000000-0000-4000-8000-000000000010"
    assert task["resolution_authority"] == "fleet_job_roster+training_environment_catalog"
    assert "must-not-survive" not in json.dumps(value)


def test_snapshot_digest_excludes_only_digest_field() -> None:
    value = snapshot()
    unsigned = {key: item for key, item in value.items() if key != "manifest_digest"}
    assert value["manifest_digest"] == digest_json(unsigned)


def test_committed_full_config_is_exactly_reproducible() -> None:
    root = Path(__file__).resolve().parents[1]
    split = json.loads((root / "configs/data/fleet-a62-task-split-v1.json").read_text())
    template_value = json.loads(
        (root / "configs/runs/qwen36-27b-rl-base-full.template.json").read_text()
    )
    committed = json.loads((root / "configs/runs/qwen36-27b-rl-base-full.json").read_text())
    assert split["manifest_digest"] == (
        "sha256:fb09668f8de77e6baee37dc1e3162a7617ca9a0a450da9ca53422c17bfbc942a"
    )
    assert split["counts"] == {
        "tasks": 160,
        "splits": {"dev": 10, "test": 20, "train": 130},
    }
    assert build_full_rl_config(template_value, split) == committed
    assert len(committed["tasks"]["task_versions"]) == 130
    assert len(committed["eval"]["task_versions"]) == 10
    serialized = json.dumps(committed).lower()
    assert "flag{" not in serialized
    assert "planted_flag" not in serialized
    assert "sk_pw" not in serialized


def test_qwen_rl_configs_disable_microbatch_padding_for_vision_inputs() -> None:
    """Qwen vision inputs require SkyRL's unpacked reference-forward path."""
    root = Path(__file__).resolve().parents[1]
    expected = "trainer.remove_microbatch_padding=false"
    for relative in (
        "configs/runs/qwen36-27b-rl-base-full.template.json",
        "configs/runs/qwen36-27b-rl-base-full.json",
        "configs/runs/qwen36-27b-rl-base-full-runnable.json",
        "configs/runs/qwen36-27b-rl-base-smoke.json",
    ):
        config = json.loads((root / relative).read_text())
        assert expected in config["trainer"]["args"], relative


def test_runnable_qwen_rl_configs_pin_the_expected_trainer() -> None:
    root = Path(__file__).resolve().parents[1]
    expected_by_config = {
        "configs/runs/qwen36-27b-rl-base-full-runnable.json": (
            "4e800585-12d2-57aa-9a5d-9fb2927302eb"
        ),
        "configs/runs/qwen36-27b-rl-base-smoke.json": (
            "4b4dc57c-c7dc-5562-bbdd-9e1d6764ede0"
        ),
    }
    for relative, expected in expected_by_config.items():
        config = json.loads((root / relative).read_text())
        assert config["trainer"]["trainer_version_id"] == expected, relative
        assert (
            "trainer.policy.model_config_kwargs.fleet_force_qwen35_torch_gdn=true"
            in config["trainer"]["args"]
        ), relative
        assert (
            "trainer.ref.model_config_kwargs.fleet_force_qwen35_torch_gdn=true"
            in config["trainer"]["args"]
        ), relative

    full = json.loads(
        (
            root / "configs/runs/qwen36-27b-rl-base-full-runnable.json"
        ).read_text()
    )
    assert len(full["tasks"]["task_versions"]) == 129
    assert all(
        row["task_key"].endswith("__blackbox_ctf_v1")
        for row in full["tasks"]["task_versions"]
    )


def test_successor_template_adds_true_step_cap_without_rewriting_historical_request() -> None:
    root = Path(__file__).resolve().parents[1]
    split = json.loads((root / "configs/data/fleet-a62-task-split-v1.json").read_text())
    treatment = json.loads((root / "configs/data/fleet-a62-rl-treatment-v1.json").read_text())
    historical = json.loads(
        (root / "configs/runs/qwen36-27b-rl-base-full-runnable.json").read_text()
    )
    successor = json.loads(
        (root / "configs/runs/qwen36-27b-rl-successor.template.json").read_text()
    )
    assert historical["grpo"]["max_steps"] == successor["grpo"]["max_steps"] == 130
    assert "trainer.max_training_steps=130" not in historical["trainer"]["args"]
    assert successor["trainer"]["args"].count("trainer.max_training_steps=130") == 1
    assert successor["trainer"]["trainer_version_id"] == (
        "4e800585-12d2-57aa-9a5d-9fb2927302eb"
    )
    built = build_full_rl_config(successor, split, treatment)
    assert len(built["tasks"]["task_versions"]) == 129
    assert len(built["eval"]["task_versions"]) == 10
    assert built["trainer"]["args"].count("trainer.max_training_steps=130") == 1


def test_authoritative_native_rl_gate_matches_frozen_two_task_receipt() -> None:
    root = Path(__file__).resolve().parents[1]
    config = json.loads(
        (root / "configs/runs/qwen36-27b-native-rl-authoritative-smoke.json").read_text()
    )
    receipt = json.loads(
        (root / "configs/data/qwen36-27b-native-rl-two-task-gate-v1.json").read_text()
    )
    split = json.loads((root / "configs/data/fleet-a62-task-split-v1.json").read_text())
    original_path = root / "configs/data/qwen36-27b-native-rl-smoke-selection-v1.json"

    assert receipt["split_manifest_digest"] == split["manifest_digest"]
    assert receipt["original_selection_file_sha256"] == (
        f"sha256:{hashlib.sha256(original_path.read_bytes()).hexdigest()}"
    )
    split_by_version = {task["task_version_id"]: task for task in split["tasks"]}
    identity_fields = (
        "task_key",
        "task_version_id",
        "task_version",
        "environment_version_id",
        "env_key",
        "env_version",
        "data_key",
        "data_version",
    )
    config_tasks = config["tasks"]["task_versions"]
    assert len(config_tasks) == len(receipt["tasks"]) == 2
    for configured, recorded in zip(config_tasks, receipt["tasks"], strict=True):
        assert configured["env_variables"] == {}
        assert configured["env_variable_deletions"] == []
        frozen = split_by_version[configured["task_version_id"]]
        assert frozen["split"] == "train"
        for field in identity_fields:
            assert configured[field] == recorded[field] == frozen[field]

    assert config["trainer"]["trainer_version_id"] == receipt["trainer"]["trainer_version_id"]
    assert config["grpo"]["group_size"] == 4
    assert config["grpo"]["train_batch_size"] == 2
    assert config["grpo"]["policy_mini_batch_size"] == 2
    assert config["grpo"]["max_steps"] == 1
    assert receipt["selection_policy"]["planned_rollouts"] == (
        config["grpo"]["group_size"] * config["grpo"]["train_batch_size"]
    )
    assert config["gpus_per_worker"] == 8
    assert config["rollout"] == {
        "harness": "native",
        "mode": "tool-use",
        "partial_verifier_scoring": False,
        "pass_conversation_to_verifier": False,
        "multi_app_aggregation_mode": "binary",
    }
    assert config["eval"]["task_keys"] == []
    assert config["eval"]["task_versions"] == []
    assert "trainer.remove_microbatch_padding=false" in config["trainer"]["args"]
    assert (
        "trainer.policy.model_config_kwargs.fleet_force_qwen35_torch_gdn=true"
        in config["trainer"]["args"]
    )
    assert (
        "trainer.ref.model_config_kwargs.fleet_force_qwen35_torch_gdn=true"
        in config["trainer"]["args"]
    )
    serialized = json.dumps({"config": config, "receipt": receipt}).lower()
    assert "flag{" not in serialized
    assert "sk_pw" not in serialized


def test_qwenxml_corrected_gate_changes_only_trainer_identity() -> None:
    root = Path(__file__).resolve().parents[1]
    original = json.loads(
        (root / "configs/runs/qwen36-27b-native-rl-authoritative-smoke.json").read_text()
    )
    corrected = json.loads(
        (
            root
            / "configs/runs/qwen36-27b-native-rl-authoritative-smoke-q36xml.json"
        ).read_text()
    )

    assert corrected["trainer"]["trainer_version_id"] == (
        "83f8d256-aea8-52d7-8711-080c56dbbc3a"
    )
    assert corrected["title"].endswith("q36xml-1a74092a")
    original.pop("title")
    corrected.pop("title")
    original["trainer"].pop("trainer_version_id")
    corrected["trainer"].pop("trainer_version_id")
    assert corrected == original


def test_qwen_sft_configs_pin_and_record_the_torch_gdn_fallback() -> None:
    root = Path(__file__).resolve().parents[1]
    expected = "4b4dc57c-c7dc-5562-bbdd-9e1d6764ede0"
    for relative in (
        "configs/runs/qwen36-27b-sft-smoke.json",
        "configs/runs/qwen36-27b-sft-full.json",
    ):
        config = json.loads((root / relative).read_text())
        assert config["trainer"]["trainer_version_id"] == expected, relative
        assert config["trainer"]["args"] == [
            "model_config_kwargs.fleet_force_qwen35_torch_gdn=true"
        ], relative
        assert "env" not in config["trainer"], relative
