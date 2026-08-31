import copy
import json
from pathlib import Path

import pytest

from training.io import digest_json, file_sha256
from training.post_sft import (
    build_fleet_test_holdout_receipt,
    build_post_sft_comparison_receipt,
    build_zero_step_hf_export_request,
    checkpoint_uuid,
    derive_post_sft_registration,
    derive_webexploit_config,
    freeze_final_promoted_checkpoint,
    validate_selection_receipt,
)

ROOT = Path(__file__).resolve().parents[1]
RUN = "ft-run-574bd7b3"
CONFIG_SHA = "sha256:" + "1" * 64
IMAGE = "registry.example/trainer@sha256:" + "2" * 64


def _selection_inputs(status: str = "succeeded"):
    run = {
        "schema": "fleet_training_run_observation_v1",
        "run_name": RUN,
        "status": status,
        "rayjob_uid": "uid-1",
        "run_config_sha256": CONFIG_SHA,
        "trainer_image": IMAGE,
        "entrypoint_sha256": "sha256:" + "3" * 64,
        "latest_checkpoint_step": 318,
    }
    checkpoints = [
        {
            "run_name": RUN,
            "step": 310,
            "complete": True,
            "is_promoted": False,
            "s3_available": True,
            "s3_uri": "s3://bucket/run/310",
            "archive_manifest_sha256": "sha256:" + "4" * 64,
        },
        {
            "run_name": RUN,
            "step": 318,
            "complete": True,
            "is_promoted": True,
            "s3_available": True,
            "s3_uri": "s3://bucket/run/318",
            "archive_manifest_sha256": "sha256:" + "5" * 64,
        },
    ]
    return run, checkpoints


def _selection():
    run, checkpoints = _selection_inputs()
    return freeze_final_promoted_checkpoint(
        run,
        checkpoints,
        expected_run_name=RUN,
        expected_run_config_sha256=CONFIG_SHA,
        expected_rayjob_uid="uid-1",
        expected_trainer_image=IMAGE,
    )


def _export(selection):
    value = {
        "schema": "cyber_sft_hf_export_v1",
        "source_checkpoint": {
            "uuid": selection["checkpoint"]["uuid"],
            "archive_manifest_sha256": selection["checkpoint"]["archive_manifest_sha256"],
        },
        "output": {
            "format": "safetensors",
            "dtype": "bfloat16",
            "source_path": "/models/cyber-sft/ft-run-574bd7b3/step-318",
            "weights_manifest_sha256": "sha256:" + "6" * 64,
            "tokenizer_manifest_sha256": "sha256:" + "7" * 64,
            "chat_template_sha256": "sha256:" + "8" * 64,
        },
        "conversion": {
            "image": IMAGE,
            "optimizer_steps": 0,
            "export_request_receipt_sha256": "sha256:" + "9" * 64,
            "command_sha256": "sha256:" + "a" * 64,
        },
        "verification": {
            "all_shards_present": True,
            "safetensors_load_passed": True,
            "parameter_count_matches": True,
        },
    }
    value["export_receipt_sha256"] = digest_json(value)
    return value


def test_checkpoint_selection_is_final_promoted_only():
    selection = _selection()
    assert selection["checkpoint"]["step"] == 318
    assert selection["checkpoint"]["uuid"] == checkpoint_uuid(RUN, 318)
    assert selection["checkpoint"]["fleet_model"] == f"fleet/{RUN}-step-318"
    assert "intermediate_dev_loss_ranking" in selection["selection_excludes"]


def test_checkpoint_selection_fails_before_success_or_with_two_promoted():
    run, checkpoints = _selection_inputs(status="running")
    with pytest.raises(ValueError, match="must be succeeded"):
        freeze_final_promoted_checkpoint(
            run,
            checkpoints,
            expected_run_name=RUN,
            expected_run_config_sha256=CONFIG_SHA,
            expected_rayjob_uid="uid-1",
            expected_trainer_image=IMAGE,
        )
    run["status"] = "succeeded"
    checkpoints[0]["is_promoted"] = True
    with pytest.raises(ValueError, match="exactly one"):
        freeze_final_promoted_checkpoint(
            run,
            checkpoints,
            expected_run_name=RUN,
            expected_run_config_sha256=CONFIG_SHA,
            expected_rayjob_uid="uid-1",
            expected_trainer_image=IMAGE,
        )


def test_selection_receipt_rejects_tampering():
    selection = _selection()
    assert validate_selection_receipt(selection) == selection["selection_receipt_sha256"]
    selection["checkpoint"]["step"] = 317
    with pytest.raises(ValueError, match="UUID does not match"):
        validate_selection_receipt(selection)


def test_zero_step_export_request_preserves_recipe_and_cannot_mutate_source():
    sft = json.loads((ROOT / "configs/runs/qwen36-27b-sft-full.json").read_text())
    selection = _selection()
    receipt = build_zero_step_hf_export_request(
        sft,
        selection,
        expected_trainer_version_id=str(sft["trainer"]["trainer_version_id"]),
        expected_trainer_image=IMAGE,
    )
    request = receipt["request"]
    assert receipt["submit"] is False
    assert request["sft"]["max_steps"] == 318
    assert request["sft"]["num_epochs"] is None
    assert request["data"] == sft["data"]
    assert request["model"] == sft["model"]
    assert f"resume_from=/mnt/sfs/checkpoints/{RUN}/global_step_318" in request["trainer"][
        "args"
    ]
    assert "hf_save_interval=319" in request["trainer"]["args"]
    assert request["title"].startswith("Chris cyber zero-step HF export")
    assert "name" not in request
    assert receipt["expected_output"]["path"].endswith("/global_step_318/policy")


def test_post_sft_registration_preserves_runtime_and_precision():
    base = json.loads(
        (ROOT / "evals/webexploitbench/serving/qwen36-27b-6a9e13bd-registration.json")
        .read_text()
    )
    selection = _selection()
    export = _export(selection)
    receipt = derive_post_sft_registration(
        base,
        selection,
        export,
        expected_tokenizer_manifest_sha256="sha256:" + "7" * 64,
        expected_chat_template_sha256="sha256:" + "8" * 64,
    )
    candidate = receipt["registration"]
    assert candidate["id"] == f"{RUN}-step-318"
    assert candidate["spec"]["model"]["precision"] == "bf16"
    assert candidate["spec"]["runtime"]["image"] == base["spec"]["runtime"]["image"]
    assert candidate["spec"]["runtime"]["engine"] == "sglang"

    old_args = base["spec"]["runtime"]["args"]
    new_args = candidate["spec"]["runtime"]["args"]
    allowed = {
        base["id"]: candidate["id"],
        base["spec"]["model"]["path"]: candidate["spec"]["model"]["path"],
    }
    normalized = [
        next((old for old, new in allowed.items() if arg == new), arg) for arg in new_args
    ]
    assert normalized == old_args


def test_webexploit_config_changes_only_model_and_run_id():
    base = json.loads(
        (ROOT / "evals/webexploitbench/configs/qwen36-27b-6a9e13bd-level0-full.json")
        .read_text()
    )
    candidate = derive_webexploit_config(
        base,
        served_model_id=f"{RUN}-step-318",
        run_id="webexploit-q36-sft-574bd7b3-l0-p1-v1",
    )
    changed = {key for key in base if base[key] != candidate[key]}
    assert changed == {"model", "run_id"}
    assert candidate["agent"] == "claude_code"


def test_fleet_holdout_is_exact_and_has_no_training_leakage():
    split = json.loads((ROOT / "configs/data/fleet-a62-task-split-v1.json").read_text())
    sft = json.loads((ROOT / "configs/runs/qwen36-27b-sft-full.json").read_text())
    receipt = build_fleet_test_holdout_receipt(split, sft)
    assert receipt["task_count"] == 20
    assert len({row["task_version_id"] for row in receipt["tasks"]}) == 20
    assert receipt["training_overlap"] == 0
    assert receipt["development_overlap"] == 0
    assert receipt["evaluation_policy"]["resolve_current_task_version"] is False

    leaked = copy.deepcopy(sft)
    leaked["data"]["task_keys"].append(receipt["tasks"][0]["task_key"])
    with pytest.raises(ValueError, match="overlap"):
        build_fleet_test_holdout_receipt(split, leaked)


def test_comparison_receipt_binds_all_four_handoffs():
    selection = _selection()
    export = _export(selection)
    base_registration = json.loads(
        (ROOT / "evals/webexploitbench/serving/qwen36-27b-6a9e13bd-registration.json")
        .read_text()
    )
    serving = derive_post_sft_registration(
        base_registration,
        selection,
        export,
        expected_tokenizer_manifest_sha256="sha256:" + "7" * 64,
        expected_chat_template_sha256="sha256:" + "8" * 64,
    )
    split = json.loads((ROOT / "configs/data/fleet-a62-task-split-v1.json").read_text())
    sft = json.loads((ROOT / "configs/runs/qwen36-27b-sft-full.json").read_text())
    holdout = build_fleet_test_holdout_receipt(split, sft)
    receipt = build_post_sft_comparison_receipt(
        selection=selection,
        export=export,
        serving=serving,
        base_webexploit_config_sha256="sha256:" + "b" * 64,
        post_webexploit_config_sha256="sha256:" + "c" * 64,
        fleet_holdout=holdout,
    )
    assert receipt["allowed_model_difference"] == "checkpoint_weights_only"
    assert receipt["fleet"]["exact_task_versions"] == 20


def test_checked_in_plan_binds_current_files():
    plan = json.loads(
        (ROOT / "configs/evaluation/qwen36-27b-ft-run-574bd7b3-post-sft.json").read_text()
    )
    for section, binding in (
        ("serving", plan["serving"]["base_registration"]),
        ("webexploitbench", plan["webexploitbench"]["base_config"]),
        ("split", plan["fleet"]["split_manifest"]),
        ("sft", plan["fleet"]["sft_config"]),
    ):
        assert file_sha256(ROOT / binding["path"]) == binding["sha256"], section
