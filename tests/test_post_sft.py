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
    freeze_final_promoted_sfs_checkpoint,
    validate_selection_receipt,
)

ROOT = Path(__file__).resolve().parents[1]
RUN = "ft-run-574bd7b3"
CONFIG_SHA = "sha256:" + "1" * 64
IMAGE = "registry.example/trainer@sha256:" + "2" * 64
ENTRYPOINT_SHA = "sha256:" + "3" * 64


def _export_binding():
    output_root = f"/mnt/sfs/exports/cyber-sft/{RUN}/step-318-v1"
    return {
        "strategy": "resume_final_checkpoint_with_zero_optimizer_steps_v1",
        "source_run_name": RUN,
        "source_global_step": 318,
        "source_checkpoint_path": f"/mnt/sfs/checkpoints/{RUN}/global_step_318",
        "output_root": output_root,
        "expected_output_path": f"{output_root}/global_step_318/policy",
        "destination_preflight": {
            "observed_at": "2026-08-31T16:05:07Z",
            "state": "absent",
            "matching_rayjobs": [{"name": "ft-run-export", "uid": "export-uid"}],
        },
        "export_run": {
            "name": "ft-run-export",
            "run_id": "export-run-id",
            "rayjob_uid": "export-uid",
            "trainer_version_id": "trainer-version",
            "trainer_image": IMAGE,
            "resume_from": f"/mnt/sfs/checkpoints/{RUN}/global_step_318",
            "num_steps": 318,
            "hf_save_interval": 319,
            "optimizer_steps_expected": 0,
        },
    }


def _selection_inputs(status: str = "succeeded"):
    run = {
        "schema": "fleet_training_run_observation_v1",
        "run_name": RUN,
        "status": status,
        "rayjob_uid": "uid-1",
        "run_config_sha256": CONFIG_SHA,
        "trainer_image": IMAGE,
        "entrypoint_sha256": ENTRYPOINT_SHA,
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
            "sfs_path": f"/mnt/sfs/checkpoints/{RUN}/global_step_310",
            "sfs_available": True,
            "archive_manifest_sha256": "sha256:" + "4" * 64,
        },
        {
            "run_name": RUN,
            "step": 318,
            "complete": True,
            "is_promoted": True,
            "s3_available": True,
            "s3_uri": "s3://bucket/run/318",
            "sfs_path": f"/mnt/sfs/checkpoints/{RUN}/global_step_318",
            "sfs_available": True,
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
        expected_entrypoint_sha256=ENTRYPOINT_SHA,
    )


def _export(selection):
    value = {
        "schema": "cyber_sft_hf_export_v1",
        "source_checkpoint": {
            "uuid": selection["checkpoint"]["uuid"],
            "source_manifest_sha256": selection["checkpoint"].get(
                "source_manifest_sha256", selection["checkpoint"].get("archive_manifest_sha256")
            ),
        },
        "output": {
            "format": "safetensors",
            "dtype": "bfloat16",
            "source_path": "/models/cyber-sft/ft-run-574bd7b3/step-318",
            "weights_manifest_sha256": "sha256:" + "6" * 64,
            "files_manifest_sha256": "sha256:" + "f" * 64,
            "tokenizer_manifest_sha256": "sha256:" + "7" * 64,
            "chat_template_sha256": "sha256:" + "8" * 64,
            "config_sha256": "sha256:" + "0" * 64,
        },
        "conversion": {
            "image": IMAGE,
            "optimizer_steps": 0,
            "run": {
                "name": "ft-run-export",
                "run_id": "export-run-id",
                "rayjob_uid": "export-uid",
                "trainer_version_id": "trainer-version",
            },
            "resume_from": f"/mnt/sfs/checkpoints/{RUN}/global_step_318",
            "num_steps": 318,
            "hf_save_interval": 319,
            "export_request_receipt_sha256": "sha256:" + "9" * 64,
            "command_sha256": "sha256:" + "a" * 64,
            "output_path": _export_binding()["expected_output_path"],
            "destination_preflight": _export_binding()["destination_preflight"],
        },
        "staging": {
            "source_path": _export_binding()["expected_output_path"],
            "destination_path": "/models/cyber-sft/ft-run-574bd7b3/step-318",
            "image": "registry.example/stager@sha256:" + "b" * 64,
            "command_sha256": "sha256:" + "c" * 64,
            "source_manifest_sha256": "sha256:" + "6" * 64,
            "destination_manifest_sha256": "sha256:" + "6" * 64,
            "byte_identical": True,
            "acceptance_manifest_sha256": "sha256:" + "d" * 64,
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
            expected_entrypoint_sha256=ENTRYPOINT_SHA,
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
            expected_entrypoint_sha256=ENTRYPOINT_SHA,
        )


def test_selection_receipt_rejects_tampering():
    selection = _selection()
    assert validate_selection_receipt(selection) == selection["selection_receipt_sha256"]
    selection["checkpoint"]["step"] = 317
    with pytest.raises(ValueError, match="UUID does not match"):
        validate_selection_receipt(selection)


def test_sfs_checkpoint_selection_requires_two_leg_stability_and_full_hash():
    run, _ = _selection_inputs()
    pipeline = {
        "argocd_application": "checkpoint-pipeline",
        "application_uid": "app-uid",
        "sync_revision": "a" * 40,
        "helm_values_sha256": "sha256:" + "b" * 64,
        "apply": False,
        "archive_enabled": False,
        "observed_at": "2026-08-31T16:16:51Z",
    }
    observation = {
        "schema": "fleet_sft_sfs_checkpoint_observation_v1",
        "run_name": RUN,
        "step": 318,
        "sfs_path": f"/mnt/sfs/checkpoints/{RUN}/global_step_318",
        "api_checkpoint_rows": 0,
        "checkpoint_pipeline": pipeline,
        "markers": {
            "promoted": True,
            "milestone": True,
            "expected_shards": 1,
            "complete_shards": 1,
            "latest_step": 318,
        },
        "structural_manifest_before_sha256": "sha256:" + "c" * 64,
        "structural_manifest_after_sha256": "sha256:" + "c" * 64,
        "full_file_manifest_sha256": "sha256:" + "d" * 64,
        "full_manifest_file_count": 37,
        "full_manifest_total_bytes": 302_000_000_000,
    }
    receipt = freeze_final_promoted_sfs_checkpoint(
        run,
        observation,
        expected_run_name=RUN,
        expected_run_config_sha256=CONFIG_SHA,
        expected_rayjob_uid="uid-1",
        expected_trainer_image=IMAGE,
        expected_entrypoint_sha256=ENTRYPOINT_SHA,
        expected_pipeline=pipeline,
        expected_structural_manifest_before_sha256="sha256:" + "c" * 64,
    )
    assert receipt["schema"] == "cyber_sft_checkpoint_selection_v2"
    assert receipt["checkpoint"]["source_manifest_kind"] == "sfs_sha256_all_files_v1"
    assert validate_selection_receipt(receipt) == receipt["selection_receipt_sha256"]

    changed = copy.deepcopy(observation)
    changed["structural_manifest_after_sha256"] = "sha256:" + "e" * 64
    with pytest.raises(ValueError, match="structure changed"):
        freeze_final_promoted_sfs_checkpoint(
            run,
            changed,
            expected_run_name=RUN,
            expected_run_config_sha256=CONFIG_SHA,
            expected_rayjob_uid="uid-1",
            expected_trainer_image=IMAGE,
            expected_entrypoint_sha256=ENTRYPOINT_SHA,
            expected_pipeline=pipeline,
            expected_structural_manifest_before_sha256="sha256:" + "c" * 64,
        )

    unindexed_without_explanation = copy.deepcopy(observation)
    unindexed_without_explanation["checkpoint_pipeline"]["apply"] = True
    expected = {**pipeline, "apply": True}
    with pytest.raises(ValueError, match="empty API index is not explained"):
        freeze_final_promoted_sfs_checkpoint(
            run,
            unindexed_without_explanation,
            expected_run_name=RUN,
            expected_run_config_sha256=CONFIG_SHA,
            expected_rayjob_uid="uid-1",
            expected_trainer_image=IMAGE,
            expected_entrypoint_sha256=ENTRYPOINT_SHA,
            expected_pipeline=expected,
            expected_structural_manifest_before_sha256="sha256:" + "c" * 64,
        )


def test_checkpoint_selection_rejects_unplanned_entrypoint():
    run, checkpoints = _selection_inputs()
    with pytest.raises(ValueError, match="entrypoint digest"):
        freeze_final_promoted_checkpoint(
            run,
            checkpoints,
            expected_run_name=RUN,
            expected_run_config_sha256=CONFIG_SHA,
            expected_rayjob_uid="uid-1",
            expected_trainer_image=IMAGE,
            expected_entrypoint_sha256="sha256:" + "f" * 64,
        )


def test_zero_step_export_request_preserves_recipe_and_cannot_mutate_source():
    sft = json.loads((ROOT / "configs/runs/qwen36-27b-sft-full.json").read_text())
    selection = _selection()
    receipt = build_zero_step_hf_export_request(
        sft,
        selection,
        expected_trainer_version_id=str(sft["trainer"]["trainer_version_id"]),
        expected_trainer_image=IMAGE,
        expected_export_binding={
            **_export_binding(),
            "export_run": {
                **_export_binding()["export_run"],
                "trainer_version_id": str(sft["trainer"]["trainer_version_id"]),
            },
        },
    )
    request = receipt["request"]
    assert receipt["submit"] is False
    assert request["sft"]["max_steps"] == 318
    assert request["sft"]["num_epochs"] is None
    assert request["data"] == sft["data"]
    assert request["model"] == sft["model"]
    assert f"resume_from=/mnt/sfs/checkpoints/{RUN}/global_step_318" in request["trainer"]["args"]
    assert "hf_save_interval=319" in request["trainer"]["args"]
    assert request["title"].startswith("Chris cyber zero-step HF export")
    assert "name" not in request
    assert receipt["expected_output"]["path"].endswith("/global_step_318/policy")


def test_post_sft_registration_preserves_runtime_and_precision():
    base = json.loads(
        (ROOT / "evals/webexploitbench/serving/qwen36-27b-6a9e13bd-registration.json").read_text()
    )
    selection = _selection()
    export = _export(selection)
    receipt = derive_post_sft_registration(
        base,
        selection,
        export,
        expected_tokenizer_manifest_sha256="sha256:" + "7" * 64,
        expected_chat_template_sha256="sha256:" + "8" * 64,
        expected_config_sha256="sha256:" + "0" * 64,
        expected_export_binding=_export_binding(),
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


def test_post_sft_registration_rejects_unproven_inference_staging():
    base = json.loads(
        (ROOT / "evals/webexploitbench/serving/qwen36-27b-6a9e13bd-registration.json").read_text()
    )
    selection = _selection()
    export = _export(selection)
    export["staging"]["destination_manifest_sha256"] = "sha256:" + "e" * 64
    export["export_receipt_sha256"] = digest_json(
        {key: value for key, value in export.items() if key != "export_receipt_sha256"}
    )
    with pytest.raises(ValueError, match="staging destination manifest"):
        derive_post_sft_registration(
            base,
            selection,
            export,
            expected_tokenizer_manifest_sha256="sha256:" + "7" * 64,
            expected_chat_template_sha256="sha256:" + "8" * 64,
            expected_config_sha256="sha256:" + "0" * 64,
            expected_export_binding=_export_binding(),
        )


def test_post_sft_registration_requires_exact_composed_runtime_sidecars():
    base = json.loads(
        (ROOT / "evals/webexploitbench/serving/qwen36-27b-6a9e13bd-registration.json").read_text()
    )
    selection = _selection()
    export = _export(selection)
    export["output"]["sidecar_sha256"] = {"config.json": "sha256:" + "1" * 64}
    export["export_receipt_sha256"] = digest_json(
        {key: value for key, value in export.items() if key != "export_receipt_sha256"}
    )
    with pytest.raises(ValueError, match="runtime sidecars"):
        derive_post_sft_registration(
            base,
            selection,
            export,
            expected_tokenizer_manifest_sha256="sha256:" + "7" * 64,
            expected_chat_template_sha256="sha256:" + "8" * 64,
            expected_config_sha256="sha256:" + "0" * 64,
            expected_export_binding=_export_binding(),
            expected_runtime_sidecar_sha256={"config.json": "sha256:" + "2" * 64},
        )


def test_export_binding_rejects_path_mismatch_and_destination_collision():
    sft = json.loads((ROOT / "configs/runs/qwen36-27b-sft-full.json").read_text())
    selection = _selection()
    binding = _export_binding()
    binding["expected_output_path"] = "/mnt/sfs/exports/wrong/global_step_318/policy"
    with pytest.raises(ValueError, match="output path is inconsistent"):
        build_zero_step_hf_export_request(
            sft,
            selection,
            expected_trainer_version_id=str(sft["trainer"]["trainer_version_id"]),
            expected_trainer_image=IMAGE,
            expected_export_binding={
                **binding,
                "export_run": {
                    **binding["export_run"],
                    "trainer_version_id": str(sft["trainer"]["trainer_version_id"]),
                },
            },
        )

    binding = _export_binding()
    binding["destination_preflight"]["matching_rayjobs"].append(
        {"name": "ft-run-collision", "uid": "collision-uid"}
    )
    with pytest.raises(ValueError, match="uniquely assigned"):
        build_zero_step_hf_export_request(
            sft,
            selection,
            expected_trainer_version_id=str(sft["trainer"]["trainer_version_id"]),
            expected_trainer_image=IMAGE,
            expected_export_binding={
                **binding,
                "export_run": {
                    **binding["export_run"],
                    "trainer_version_id": str(sft["trainer"]["trainer_version_id"]),
                },
            },
        )


def test_export_receipt_rejects_wrong_run_identity_and_missing_output_hash():
    base = json.loads(
        (ROOT / "evals/webexploitbench/serving/qwen36-27b-6a9e13bd-registration.json").read_text()
    )
    selection = _selection()
    export = _export(selection)
    export["conversion"]["run"]["rayjob_uid"] = "wrong-uid"
    export["export_receipt_sha256"] = digest_json(
        {key: value for key, value in export.items() if key != "export_receipt_sha256"}
    )
    with pytest.raises(ValueError, match="rayjob_uid differs"):
        derive_post_sft_registration(
            base,
            selection,
            export,
            expected_tokenizer_manifest_sha256="sha256:" + "7" * 64,
            expected_chat_template_sha256="sha256:" + "8" * 64,
            expected_config_sha256="sha256:" + "0" * 64,
            expected_export_binding=_export_binding(),
        )

    export = _export(selection)
    del export["output"]["files_manifest_sha256"]
    export["export_receipt_sha256"] = digest_json(
        {key: value for key, value in export.items() if key != "export_receipt_sha256"}
    )
    with pytest.raises(ValueError, match="files_manifest_sha256"):
        derive_post_sft_registration(
            base,
            selection,
            export,
            expected_tokenizer_manifest_sha256="sha256:" + "7" * 64,
            expected_chat_template_sha256="sha256:" + "8" * 64,
            expected_config_sha256="sha256:" + "0" * 64,
            expected_export_binding=_export_binding(),
        )


def test_webexploit_config_changes_only_model_and_run_id():
    base = json.loads(
        (ROOT / "evals/webexploitbench/configs/qwen36-27b-6a9e13bd-level0-full.json").read_text()
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
        (ROOT / "evals/webexploitbench/serving/qwen36-27b-6a9e13bd-registration.json").read_text()
    )
    serving = derive_post_sft_registration(
        base_registration,
        selection,
        export,
        expected_tokenizer_manifest_sha256="sha256:" + "7" * 64,
        expected_chat_template_sha256="sha256:" + "8" * 64,
        expected_config_sha256="sha256:" + "0" * 64,
        expected_export_binding=_export_binding(),
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
