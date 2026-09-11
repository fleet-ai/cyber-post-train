import copy
import json
from pathlib import Path

import pytest

from evals.webexploitbench import post_sft_evidence as evidence
from evals.webexploitbench.config import ExperimentConfig
from evals.webexploitbench.paired import (
    assert_cage_run_root_available,
    claim_cage_launch,
    derive_post_sft_qwen_pair,
    validate_paired_identity_for_launch,
    validate_post_sft_launch_evidence,
)
from evals.webexploitbench.post_sft_evidence import (
    ARTIFACT_COMMAND_SHA256,
    ARTIFACT_CONFIG_MAP,
    ARTIFACT_CONFIG_MAP_FILES,
    ARTIFACT_IMAGE,
    ARTIFACT_JOB,
    ARTIFACT_NAMESPACE,
    BASE_MODEL_ID,
    BASE_MODEL_REPOSITORY,
    BASE_MODEL_REVISION,
    BASE_MODEL_ROOT,
    BASE_NON_SERVING_SIDECARS,
    REGISTRATION_IMAGE,
    _artifact_projection,
    _artifacts_from_receipts,
    _atomic_write_json_new,
    _read_tokenizer_probe,
    _tool_request,
    assemble_live_parity,
    assemble_registration_completion,
    inspect_base_artifact,
)
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
    normalize_zero_step_stored_config,
    validate_selection_receipt,
)
from training.post_sft_export_observation import _validated_stored_config
from training.register_post_sft import validate_registration_receipt

ROOT = Path(__file__).resolve().parents[1]
RUN = "ft-run-574bd7b3"
CONFIG_SHA = "sha256:" + "1" * 64
IMAGE = "registry.example/trainer@sha256:" + "2" * 64
ENTRYPOINT_SHA = "sha256:" + "3" * 64


def test_base_artifact_inspection_uses_only_validated_inference_surface(tmp_path, monkeypatch):
    plan = json.loads(
        (ROOT / "configs/evaluation/qwen36-27b-ft-run-574bd7b3-post-sft.json").read_text()
    )
    model = plan["base_model"]
    root = tmp_path / "model"
    root.mkdir()
    lock = {
        "schema": "cyber_post_train_checkpoint_lock_v1",
        "repo": BASE_MODEL_REPOSITORY,
        "revision": BASE_MODEL_REVISION,
        "weights_manifest_sha256": model["weights_manifest_sha256"],
        "verified_shards": 15,
    }
    (root / ".cyber-post-train-lock.json").write_text(json.dumps(lock))
    control_names = set(BASE_NON_SERVING_SIDECARS)
    runtime_names = set(model["runtime_sidecar_sha256"])
    required_names = {"model.safetensors.index.json", "model-00001-of-00001.safetensors"}
    for name in sorted(control_names | runtime_names | required_names):
        path = root / name
        if not path.exists():
            path.write_text(name)
    cache = root / ".cache/huggingface/trees"
    cache.mkdir(parents=True)
    (cache / "must-not-be-read.json").write_text("control metadata")
    surface_manifest = {
        "files": [{"path": name} for name in sorted(required_names | runtime_names)],
        "excluded_non_artifact_files": [{"path": name} for name in sorted(control_names)],
    }

    def validate_surface(observed_root, _surface):
        assert observed_root == root.resolve()
        return surface_manifest

    def inspect_view(view, **_kwargs):
        observed = {path.name for path in view.iterdir()}
        assert observed == required_names | runtime_names | control_names
        assert ".cache" not in observed
        assert all(path.is_symlink() for path in view.iterdir())
        sidecars = {
            **model["runtime_sidecar_sha256"],
            **{name: "sha256:" + "a" * 64 for name in control_names},
        }
        return {
            "schema": "cyber_sft_hf_output_inspection_v1",
            "root": str(view),
            "format": "safetensors",
            "dtype": "bf16",
            "all_shards_present": True,
            "safetensors_load_passed": True,
            "parameter_count_matches": True,
            "weights_manifest_sha256": model["weights_manifest_sha256"],
            "tokenizer_manifest_sha256": model["tokenizer_manifest_sha256"],
            "chat_template_sha256": model["chat_template_sha256"],
            "config_sha256": model["config_sha256"],
            "sidecar_sha256": sidecars,
        }

    monkeypatch.setattr(evidence, "BASE_MODEL_ROOT", str(root.resolve()))
    monkeypatch.setattr(
        evidence.post_sft_cast, "_base_inference_artifact_manifest", validate_surface
    )
    monkeypatch.setattr(evidence, "inspect_hf_export", inspect_view)
    monkeypatch.setattr(evidence, "_artifact_execution_provenance", lambda _plan: {})

    def read_base_fixture(path):
        if str(path).endswith(".cyber-post-train-lock.json"):
            return lock
        return {
            "event": "stage_complete",
            "model": BASE_MODEL_REPOSITORY,
            "revision": BASE_MODEL_REVISION,
            "resolved_path": str(root.resolve()),
            "file_count": sum(
                1
                for path in root.iterdir()
                if path.is_file() and path.name != ".fleet-acceptance.json"
            ),
            "total_bytes": sum(
                path.stat().st_size
                for path in root.iterdir()
                if path.is_file() and path.name != ".fleet-acceptance.json"
            ),
            "staging_partitions": 1,
            "weights_manifest_sha256": model["weights_manifest_sha256"],
        }

    monkeypatch.setattr(evidence, "_read", read_base_fixture)

    receipt = inspect_base_artifact(root, plan)

    assert receipt["inspection"]["root"] == str(root.resolve())
    assert receipt["inference_artifact_manifest"] == surface_manifest


def _export_binding():
    output_root = f"/mnt/sfs/exports/cyber-sft/{RUN}/step-318-v1"
    return {
        "strategy": "resume_final_checkpoint_with_zero_optimizer_steps_v1",
        "source_run_name": RUN,
        "source_global_step": 318,
        "source_checkpoint_path": f"/mnt/sfs/checkpoints/{RUN}/global_step_318",
        "output_root": output_root,
        "expected_output_path": f"{output_root}/global_step_318/policy",
        "bf16_cast_destination": (
            f"/mnt/sfs/exports/cyber-sft/{RUN}/step-318-bf16-v1/global_step_318/policy"
        ),
        "inference_staging_destination": "/models/cyber-sft/ft-run-574bd7b3/step-318",
        "destination_preflight": {
            "observed_at": "2026-08-31T16:05:07Z",
            "state": "absent",
            "matching_rayjobs": [{"name": "ft-run-export", "uid": "export-uid"}],
        },
        "export_run": {
            "name": "ft-run-export",
            "run_id": "export-run-id",
            "rayjob_uid": "export-uid",
            "title": "Chris cyber test zero-step export",
            "trainer_version_id": "trainer-version",
            "trainer_image": IMAGE,
            "resume_from": f"/mnt/sfs/checkpoints/{RUN}/global_step_318",
            "num_steps": 318,
            "hf_save_interval": 319,
            "optimizer_steps_expected": 0,
            "server_normalization": {
                "schema": "fleet_training_stored_config_normalization_v1",
                "submitted_by_email": "scientist@example.com",
                "node_pool": "fleetai-training-ng-gpu",
                "data_defaults": {
                    "env_keys": None,
                    "models": None,
                    "session_ids": None,
                    "since": None,
                    "team_ids": ["a1025f0b-ad67-49fc-a023-51800ab43e84"],
                    "until": None,
                },
                "trainer_defaults": {"command": None, "env": {}},
            },
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
            "tokenizer_manifest_sha256": (
                "sha256:27f02770d03b60343350ad948bf1065673968c60274f1e34f4afaed16940a1ea"
            ),
            "chat_template_sha256": (
                "sha256:e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259"
            ),
            "config_sha256": (
                "sha256:69db4eb7196bc8190813231b3018ca05d8c2e3abc7b1af19d55c157af44a9d9c"
            ),
            "sidecar_sha256": json.loads(
                (ROOT / "configs/evaluation/qwen36-27b-ft-run-574bd7b3-post-sft.json").read_text()
            )["base_model"]["runtime_sidecar_sha256"],
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
            "observed_raw_dtype": "F32",
            "raw_weights_manifest_sha256": "sha256:" + "7" * 64,
            "raw_full_manifest_sha256": "sha256:" + "8" * 64,
        },
        "precision_correction": {
            "schema": "cyber_sft_fp32_to_bf16_precision_correction_v1",
            "source_path": _export_binding()["expected_output_path"],
            "destination_path": _export_binding()["bf16_cast_destination"],
            "source_dtype": "F32",
            "destination_dtype": "BF16",
            "policy": "deterministic_sorted_tensor_fp32_to_bf16_v1",
            "source_weights_manifest_sha256": "sha256:" + "7" * 64,
            "destination_weights_manifest_sha256": "sha256:" + "6" * 64,
            "cast_rows_sha256": "sha256:" + "8" * 64,
            "source_layout_sha256": "sha256:" + "9" * 64,
            "cast_receipt_sha256": "sha256:" + "a" * 64,
            "cast_full_manifest_sha256": "sha256:" + "b" * 64,
            "exact_cast_bits_verified": True,
        },
        "staging": {
            "source_path": _export_binding()["bf16_cast_destination"],
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
    assert request["title"] == "Chris cyber test zero-step export"
    assert "name" not in request
    assert receipt["expected_output"]["path"].endswith("/global_step_318/policy")


def test_live_export_job_server_normalization_fixture_matches_frozen_plan():
    fixture = json.loads(
        (ROOT / "tests/fixtures/ft-run-29f2bedf-config-normalization.json").read_text()
    )
    plan = json.loads(
        (ROOT / "configs/evaluation/qwen36-27b-ft-run-574bd7b3-post-sft.json").read_text()
    )
    export_run = plan["export"]["export_run"]
    for field in ("name", "run_id", "rayjob_uid", "title"):
        fixture_field = "run_name" if field == "name" else field
        assert export_run[field] == fixture[fixture_field]
    assert export_run["server_normalization"] == fixture["server_normalization"]

    request = json.loads((ROOT / "configs/runs/qwen36-27b-sft-full.json").read_text())
    request["title"] = export_run["title"]
    request["sft"]["num_epochs"] = None
    request["sft"]["max_steps"] = 318
    request["eval"] = {"task_keys": [], "interval": 1, "before_train": False}
    request["trainer"]["args"].extend(
        [
            "resume_from=/mnt/sfs/checkpoints/ft-run-574bd7b3/global_step_318",
            "hf_save_interval=319",
            "export_path=/mnt/sfs/exports/cyber-sft/ft-run-574bd7b3/step-318-v1",
        ]
    )
    stored = normalize_zero_step_stored_config(request, export_run)
    assert digest_json(stored) == fixture["fleet_run_config_sha256"]
    assert (
        _validated_stored_config(
            request,
            export_run,
            stored,
            "prompt-free live FLEET_RUN_CONFIG fixture",
        )
        == stored
    )
    assert stored["title"] == fixture["title"]
    assert stored["node_pool"] == "fleetai-training-ng-gpu"
    normalized_data = fixture["server_normalization"]["data_defaults"]
    assert {key: stored["data"][key] for key in normalized_data} == normalized_data
    assert {key: stored["trainer"][key] for key in ("command", "env")} == {
        "command": None,
        "env": {},
    }


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
        expected_tokenizer_manifest_sha256=export["output"]["tokenizer_manifest_sha256"],
        expected_chat_template_sha256=export["output"]["chat_template_sha256"],
        expected_config_sha256=export["output"]["config_sha256"],
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
            expected_tokenizer_manifest_sha256=export["output"]["tokenizer_manifest_sha256"],
            expected_chat_template_sha256=export["output"]["chat_template_sha256"],
            expected_config_sha256=export["output"]["config_sha256"],
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
            expected_tokenizer_manifest_sha256=export["output"]["tokenizer_manifest_sha256"],
            expected_chat_template_sha256=export["output"]["chat_template_sha256"],
            expected_config_sha256=export["output"]["config_sha256"],
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
            expected_tokenizer_manifest_sha256=export["output"]["tokenizer_manifest_sha256"],
            expected_chat_template_sha256=export["output"]["chat_template_sha256"],
            expected_config_sha256=export["output"]["config_sha256"],
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
            expected_tokenizer_manifest_sha256=export["output"]["tokenizer_manifest_sha256"],
            expected_chat_template_sha256=export["output"]["chat_template_sha256"],
            expected_config_sha256=export["output"]["config_sha256"],
            expected_export_binding=_export_binding(),
        )


def test_webexploit_config_changes_only_model_and_run_id():
    base = json.loads(
        (
            ROOT / "evals/webexploitbench/configs/qwen36-27b-6a9e13bd-level0-qwen-code-full.json"
        ).read_text()
    )
    candidate = derive_webexploit_config(
        base,
        served_model_id=f"{RUN}-step-318",
        run_id="webexploit-q36-sft-574bd7b3-qc0223-l0-p1-v1",
    )
    changed = {key for key in base if base[key] != candidate[key]}
    assert changed == {"model", "run_id"}
    assert candidate["agent"] == "qwen_code"
    assert candidate["agent_version"] == "0.22.3"


def _post_serving_receipt():
    selection = _selection()
    export = _export(selection)
    base_registration = json.loads(
        (ROOT / "evals/webexploitbench/serving/qwen36-27b-6a9e13bd-registration.json").read_text()
    )
    serving = derive_post_sft_registration(
        base_registration,
        selection,
        export,
        expected_tokenizer_manifest_sha256=export["output"]["tokenizer_manifest_sha256"],
        expected_chat_template_sha256=export["output"]["chat_template_sha256"],
        expected_config_sha256=export["output"]["config_sha256"],
        expected_runtime_sidecar_sha256=export["output"]["sidecar_sha256"],
        expected_export_binding=_export_binding(),
    )
    return base_registration, serving


def test_post_serving_receipt_is_accepted_by_registration_boundary():
    _, serving = _post_serving_receipt()
    registration = validate_registration_receipt(serving)
    assert registration["id"] == "ft-run-574bd7b3-step-318"
    assert registration["spec"]["model"]["sourcePath"] == (
        "/models/cyber-sft/ft-run-574bd7b3/step-318"
    )


def _paired_inputs(tmp_path):
    base_config_path = (
        ROOT / "evals/webexploitbench/configs/qwen36-27b-6a9e13bd-level0-qwen-code-full.json"
    )
    base = json.loads(base_config_path.read_text())
    base_registration, serving = _post_serving_receipt()
    post = derive_webexploit_config(
        base,
        served_model_id=serving["registration"]["id"],
        run_id="webexploit-q36-sft-574bd7b3-qc0223-l0-p1-v1",
    )
    post_path = tmp_path / "post.json"
    post_path.write_text(json.dumps(post, indent=2) + "\n")
    return {
        "baseline_terminal_path": ROOT
        / "evals/webexploitbench/manifests/qwen36-27b-qwen-code-l0-pass1-terminal.json",
        "baseline_protocol_path": ROOT
        / "evals/webexploitbench/manifests/qwen36-27b-qwen-code-protocol-v3.json",
        "baseline_config_path": base_config_path,
        "post_config_path": post_path,
        "harness_lock_path": ROOT / "evals/webexploitbench/harnesses/qwen-code-0.22.3.lock.json",
        "base_registration": base_registration,
        "post_serving_receipt": serving,
    }


def test_webexploit_pair_binds_exact_qwen_code_baseline_and_runtime(tmp_path):
    inputs = _paired_inputs(tmp_path)
    protocol, receipt = derive_post_sft_qwen_pair(**inputs)
    assert protocol.agent == "qwen_code"
    assert protocol.agent_version == "0.22.3"
    assert receipt["controlled_identity"]["benchmark"]["target_count"] == 15
    assert receipt["controlled_identity"]["benchmark"]["vulnerability_count"] == 110
    assert receipt["controlled_identity"]["execution"]["budgets"] == {
        "pass_k": 1,
        "max_concurrent": 1,
        "max_context_size": 262144,
        "judge_max_context_size": 65536,
        "request_timeout_seconds": 3600,
        "trial_timeout_seconds": 7200,
        "max_model_requests_per_target": 150,
    }
    assert (
        validate_paired_identity_for_launch(receipt, protocol, inputs["post_config_path"])
        == receipt["paired_identity_receipt_sha256"]
    )

    tampered = copy.deepcopy(receipt)
    tampered["controlled_identity"]["harness"]["version"] = "0.22.4"
    with pytest.raises(ValueError, match="receipt digest"):
        validate_paired_identity_for_launch(tampered, protocol, inputs["post_config_path"])


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        ("harness", "version", "0.22.4", "harness version"),
        ("harness", "image", "cage/qwen-code@sha256:" + "f" * 64, "harness image"),
        ("harness", "cage_commit", "f" * 40, "CAGE commit"),
        ("benchmark", "prompt_revision", "sha256:" + "f" * 64, "prompt revision"),
        ("benchmark", "max_model_requests_per_target", 151, "execution budgets"),
    ],
)
def test_webexploit_pair_rejects_terminal_protocol_drift(tmp_path, section, field, value, message):
    inputs = _paired_inputs(tmp_path)
    terminal = json.loads(Path(inputs["baseline_terminal_path"]).read_text())
    terminal[section][field] = value
    terminal_path = tmp_path / "terminal.json"
    terminal_path.write_text(json.dumps(terminal, indent=2) + "\n")
    inputs["baseline_terminal_path"] = terminal_path
    with pytest.raises(ValueError, match=message):
        derive_post_sft_qwen_pair(**inputs)


def test_webexploit_pair_rejects_target_set_drift(tmp_path):
    inputs = _paired_inputs(tmp_path)
    terminal = json.loads(Path(inputs["baseline_terminal_path"]).read_text())
    terminal["targets"][0]["id"] = terminal["targets"][1]["id"]
    terminal_path = tmp_path / "terminal.json"
    terminal_path.write_text(json.dumps(terminal, indent=2) + "\n")
    inputs["baseline_terminal_path"] = terminal_path
    with pytest.raises(ValueError, match="target set"):
        derive_post_sft_qwen_pair(**inputs)


def test_webexploit_pair_rejects_serving_runtime_drift(tmp_path):
    inputs = _paired_inputs(tmp_path)
    serving = copy.deepcopy(inputs["post_serving_receipt"])
    serving["registration"]["spec"]["runtime"]["args"][-1] = "--unpaired-runtime-flag"
    serving["serving_receipt_sha256"] = digest_json(
        {key: value for key, value in serving.items() if key != "serving_receipt_sha256"}
    )
    inputs["post_serving_receipt"] = serving
    with pytest.raises(ValueError, match="runtime differs"):
        derive_post_sft_qwen_pair(**inputs)


def test_webexploit_pair_rejects_claude_code_or_extra_config_change(tmp_path):
    inputs = _paired_inputs(tmp_path)
    post = json.loads(Path(inputs["post_config_path"]).read_text())
    post["agent"] = "claude_code"
    post["agent_version"] = None
    Path(inputs["post_config_path"]).write_text(json.dumps(post, indent=2) + "\n")
    with pytest.raises(ValueError, match="differ only by model and run_id"):
        derive_post_sft_qwen_pair(**inputs)


def _post_sft_launch_evidence(paired_identity, post_registration):
    base_registration = json.loads(
        (ROOT / "evals/webexploitbench/serving/qwen36-27b-6a9e13bd-registration.json").read_text()
    )
    registration_job_spec = {"template": {"spec": {"restartPolicy": "Never"}}}
    registration_job_status = {
        "succeeded": 1,
        "conditions": [{"type": "Complete", "status": "True"}],
    }
    registration_projection = json.loads(
        (ROOT / "configs/evaluation/qwen36-27b-ft-run-574bd7b3-post-sft.json").read_text()
    )["evidence_execution"]["registration"]["pod_spec"]
    registration = {
        "schema": "cyber_post_sft_registration_completion_v1",
        "job": {
            "namespace": "fleet-train-jobs",
            "name": "chris-cyber-qwen36-sft-register-574bd7b3-v3",
            "uid": "registration-job-uid",
            "complete": True,
            "succeeded": 1,
            "failed": 0,
            "spec": registration_job_spec,
            "status": registration_job_status,
            "spec_sha256": digest_json(registration_job_spec),
            "status_sha256": digest_json(registration_job_status),
            "reviewed_job_projection": copy.deepcopy(registration_projection),
            "reviewed_job_projection_sha256": digest_json(registration_projection),
            "pod": {
                "name": "registration-pod",
                "uid": "registration-pod-uid",
                "image": REGISTRATION_IMAGE,
                "image_id": REGISTRATION_IMAGE,
                "reviewed_spec_projection": copy.deepcopy(registration_projection),
                "reviewed_spec_projection_sha256": digest_json(registration_projection),
            },
            "config_map": {
                "name": "chris-cyber-qwen36-sft-register-574bd7b3-v3",
                "uid": "registration-config-uid",
                "resource_version": "1",
                "immutable": True,
                "data_sha256": "sha256:" + "2" * 64,
                "mounted_file_sha256": {"training/register_post_sft.py": "sha256:" + "3" * 64},
                "serving_receipt_sha256": paired_identity["post_sft"]["serving_contract_sha256"],
            },
        },
        "api_result": {
            "id": paired_identity["post_sft"]["model"],
            "phase": "pending",
            "model_revision": paired_identity["post_sft"]["model_revision"],
            "registration": copy.deepcopy(post_registration),
            "registration_sha256": digest_json(post_registration),
            "weights_manifest_sha256": "sha256:" + "6" * 64,
        },
        "source_receipts": {
            "export_receipt_sha256": _export(_selection())["export_receipt_sha256"]
        },
    }
    registration["registration_completion_sha256"] = digest_json(registration)
    runtime_image = (
        "lmsysorg/sglang@sha256:febfb971c7352570fc445c466ebd6ffc9d896024958e544a60f2137fd85856b1"
    )
    server_info = {
        "tokenizer_mode": "auto",
        "tokenizer_backend": "huggingface",
        "context_length": 262144,
        "chat_template": "qwen-frozen-template",
        "hf_chat_template_name": "default",
        "completion_template": None,
        "reasoning_parser": "qwen3",
        "tool_call_parser": "qwen3_coder",
        "sampling_defaults": {"temperature": 0.6},
        "dtype": "auto",
        "quantization": None,
        "kv_cache_dtype": "fp8_e4m3",
        "tp_size": 1,
        "version": "0.0.0.dev0+qwen38.27b.g561c8f3",
    }

    def arm(section, serving_registration):
        identity = paired_identity[section]
        model_spec = serving_registration["spec"]["model"]
        runtime = serving_registration["spec"]["runtime"]
        cr_status = {
            "phase": "ready",
            "readyReplicas": 1,
            "observedGeneration": 1,
            "conditions": [{"type": "Ready", "status": "True"}],
        }
        return {
            "inference_model": {
                "id": identity["model"],
                "uid": section + "-uid",
                "generation": 1,
                "observed_generation": 1,
                "registration": copy.deepcopy(serving_registration),
                "registration_sha256": digest_json(serving_registration),
                "canonical_spec": copy.deepcopy(serving_registration["spec"]),
                "canonical_status": cr_status,
                "spec_sha256": digest_json(serving_registration["spec"]),
                "status_sha256": digest_json(cr_status),
                "model": {
                    "source_path": model_spec["sourcePath"],
                    "serving_path": model_spec["path"],
                    "revision": model_spec["revision"],
                    "precision": model_spec["precision"],
                    "tensor_parallel_size": model_spec["tensorParallelSize"],
                },
                "runtime": {
                    "engine": runtime["engine"],
                    "image": runtime_image,
                    "command_sha256": digest_json(runtime["command"]),
                    "normalized_args_sha256": paired_identity["controlled_identity"][
                        "serving_runtime"
                    ]["normalized_registration_sha256"],
                    "env_sha256": digest_json(runtime["env"]),
                },
                "placement_sha256": digest_json(serving_registration["spec"]["placement"]),
                "scaling_sha256": digest_json(serving_registration["spec"]["scaling"]),
                "status": "ready",
                "ready_replicas": 1,
            },
            "model_info": {
                "model_path": model_spec["path"],
                "tokenizer_path": model_spec["path"],
                "model_type": "qwen3_5",
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "weight_version": "default",
            },
            "server_info": {
                **copy.deepcopy(server_info),
                "model_path": model_spec["path"],
                "tokenizer_path": model_spec["path"],
                "served_model_name": identity["model"],
                "revision": None,
                "weight_version": "default",
            },
        }

    export_receipt = _export(_selection())
    base_artifacts = copy.deepcopy(_base_artifact_receipt()["artifact_identity"])
    post_artifacts = copy.deepcopy(base_artifacts)
    post_artifacts["weights_manifest_sha256"] = "sha256:" + "6" * 64
    acceptance_sha256 = export_receipt["staging"]["acceptance_manifest_sha256"]
    post_artifacts["excluded_non_serving_files"] = {
        ".fleet-acceptance.json": {
            "sha256": acceptance_sha256,
            "reviewed_reason": (
                "post-SFT atomic-staging acceptance receipt; never loaded by SGLang"
            ),
        }
    }
    post_artifacts["full_non_weight_manifest_sha256"] = digest_json(
        [
            {"path": name, "sha256": digest}
            for name, digest in sorted(post_artifacts["serving_non_weight_file_sha256"].items())
        ]
        + [{"path": ".fleet-acceptance.json", "sha256": acceptance_sha256}]
    )
    live = {
        "schema": "webexploitbench_post_sft_live_parity_v1",
        "paired_identity_receipt_sha256": paired_identity["paired_identity_receipt_sha256"],
        "registration_completion_sha256": registration["registration_completion_sha256"],
        "arms": {
            "base": arm("baseline", base_registration),
            "post_sft": arm("post_sft", post_registration),
        },
        "artifact_identity": {
            "base": base_artifacts,
            "post_sft": post_artifacts,
            "only_difference": "checkpoint_weights",
        },
        "artifact_evidence": {
            "base_artifact_receipt_sha256": _base_artifact_receipt()[
                "base_artifact_receipt_sha256"
            ],
            "base_execution": _base_artifact_receipt()["execution"],
            "export_receipt_sha256": export_receipt["export_receipt_sha256"],
        },
        "probes": {
            "tokenizer": {
                "receipt_sha256": (
                    "sha256:3f2f72d77fda3d9e7a68cfa2ee16df031675e8f380266aa1a1eb98b00833b49e"
                ),
                "base_passed": True,
                "post_sft_passed": True,
                "exact_encode_decode_match": True,
            },
            "tool_call": {
                "request_sha256": "sha256:" + "b" * 64,
                "base_revision": paired_identity["baseline"]["model_revision"],
                "post_sft_revision": paired_identity["post_sft"]["model_revision"],
                "base_passed": True,
                "post_sft_passed": True,
                "tool_name": "identity",
                "argument_value": "parity",
            },
            "fixed_prompt_logits": {
                "request_sha256": "sha256:" + "c" * 64,
                "token_ids_sha256": "sha256:" + "d" * 64,
                "base_revision": paired_identity["baseline"]["model_revision"],
                "post_sft_revision": paired_identity["post_sft"]["model_revision"],
                "base_passed": True,
                "post_sft_passed": True,
                "finite": True,
                "deterministic": True,
            },
        },
    }
    live["live_parity_sha256"] = digest_json(live)
    return registration, live


def test_post_sft_launch_requires_registration_and_live_weights_only_parity(tmp_path):
    inputs = _paired_inputs(tmp_path)
    _, paired_identity = derive_post_sft_qwen_pair(**inputs)
    registration, live = _post_sft_launch_evidence(
        paired_identity, inputs["post_serving_receipt"]["registration"]
    )
    registration_sha256, live_sha256 = validate_post_sft_launch_evidence(
        paired_identity, registration, live
    )
    assert registration_sha256 == registration["registration_completion_sha256"]
    assert live_sha256 == live["live_parity_sha256"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda registration, live: registration["job"].update(complete=False), "complete"),
        (
            lambda registration, live: registration["job"]["spec"].update(activeDeadlineSeconds=1),
            "exact spec digest differs",
        ),
        (
            lambda registration, live: live["arms"]["base"]["inference_model"][
                "canonical_status"
            ].update(readyReplicas=2),
            "exact status digest differs",
        ),
        (
            lambda registration, live: live["arms"]["post_sft"]["server_info"].update(
                tool_call_parser="drifted"
            ),
            "server_info differs",
        ),
        (
            lambda registration, live: live["artifact_identity"]["post_sft"].update(
                non_weight_manifest_sha256="sha256:" + "e" * 64
            ),
            "non-weight artifact",
        ),
        (
            lambda registration, live: live["probes"]["fixed_prompt_logits"].update(finite=False),
            "not finite and deterministic",
        ),
    ],
)
def test_post_sft_launch_evidence_fails_closed(tmp_path, mutation, message):
    inputs = _paired_inputs(tmp_path)
    _, paired_identity = derive_post_sft_qwen_pair(**inputs)
    registration, live = _post_sft_launch_evidence(
        paired_identity, inputs["post_serving_receipt"]["registration"]
    )
    mutation(registration, live)
    registration["registration_completion_sha256"] = digest_json(
        {
            key: value
            for key, value in registration.items()
            if key != "registration_completion_sha256"
        }
    )
    live["registration_completion_sha256"] = registration["registration_completion_sha256"]
    live["live_parity_sha256"] = digest_json(
        {key: value for key, value in live.items() if key != "live_parity_sha256"}
    )
    with pytest.raises(ValueError, match=message):
        validate_post_sft_launch_evidence(paired_identity, registration, live)


def test_post_sft_launch_refuses_preexisting_cage_run_root(tmp_path):
    inputs = _paired_inputs(tmp_path)
    config = ExperimentConfig.load(inputs["post_config_path"])
    run_root = assert_cage_run_root_available(tmp_path, config)
    run_root.mkdir(parents=True)
    with pytest.raises(ValueError, match="CAGE run root already exists"):
        assert_cage_run_root_available(tmp_path, config)


def test_post_sft_launch_claim_is_atomic_and_persistent(tmp_path):
    inputs = _paired_inputs(tmp_path)
    config = ExperimentConfig.load(inputs["post_config_path"])
    run_root = assert_cage_run_root_available(tmp_path, config)
    runtime = {
        "interpreter": "/cage/.venv/bin/python",
        "console_script": "/cage/.venv/bin/cage",
        "console_interpreter": "/cage/.venv/bin/python",
        "prefix": "/cage/.venv",
        "cage_module": "/cage/cage/__init__.py",
        "registry_module": "/cage/cage/benchmarks/registry.py",
        "repository_root": "/cage",
        "project_file": "/cage/examples/agent_pentest_bench/default_web_exploit.yml",
        "benchmark_root": "/cage/examples/agent_pentest_bench/datasets/web_exploit_bench",
        "runs_root": str(run_root.parent.parent),
        "claim_root": str(run_root),
    }
    claim = claim_cage_launch(
        tmp_path,
        config,
        protocol_sha256="sha256:" + "1" * 64,
        paired_identity_sha256="sha256:" + "2" * 64,
        registration_completion_sha256="sha256:" + "3" * 64,
        live_parity_sha256="sha256:" + "4" * 64,
        cage_runtime_identity=runtime,
    )
    assert claim.is_file()
    claim_receipt = json.loads(claim.read_text())
    assert claim_receipt["schema"] == "webexploitbench_paid_launch_claim_v2"
    assert claim_receipt["cage_runtime_identity"] == runtime
    with pytest.raises(ValueError, match="launch claim already exists"):
        claim_cage_launch(
            tmp_path,
            config,
            protocol_sha256="sha256:" + "1" * 64,
            paired_identity_sha256="sha256:" + "2" * 64,
            registration_completion_sha256="sha256:" + "3" * 64,
            live_parity_sha256="sha256:" + "4" * 64,
            cage_runtime_identity=runtime,
        )

    wrong_runtime = {**runtime, "claim_root": str(run_root.parent / "wrong")}
    claim.unlink()
    with pytest.raises(ValueError, match="does not bind the paid claim root"):
        claim_cage_launch(
            tmp_path,
            config,
            protocol_sha256="sha256:" + "1" * 64,
            paired_identity_sha256="sha256:" + "2" * 64,
            registration_completion_sha256="sha256:" + "3" * 64,
            live_parity_sha256="sha256:" + "4" * 64,
            cage_runtime_identity=wrong_runtime,
        )

    wrong_runtime = {**runtime, "runs_root": str(run_root.parent / "wrong")}
    with pytest.raises(ValueError, match="does not bind the paid runs root"):
        claim_cage_launch(
            tmp_path,
            config,
            protocol_sha256="sha256:" + "1" * 64,
            paired_identity_sha256="sha256:" + "2" * 64,
            registration_completion_sha256="sha256:" + "3" * 64,
            live_parity_sha256="sha256:" + "4" * 64,
            cage_runtime_identity=wrong_runtime,
        )


def _registration_job_observation(serving):
    job_uid = "registration-job-uid"
    name = "chris-cyber-qwen36-sft-register-574bd7b3-v3"
    contract = json.loads(
        (ROOT / "configs/evaluation/qwen36-27b-ft-run-574bd7b3-post-sft.json").read_text()
    )["evidence_execution"]["registration"]
    pod_spec = copy.deepcopy(contract["pod_spec"])
    container = pod_spec.pop("container")
    pod_spec["containers"] = [container]
    return {
        "job": {
            "metadata": {
                "namespace": "fleet-train-jobs",
                "name": name,
                "uid": job_uid,
                "labels": copy.deepcopy(contract["job_labels"]),
            },
            "spec": {
                **contract["job"],
                "suspend": False,
                "template": {
                    "metadata": {"labels": copy.deepcopy(contract["pod_labels"])},
                    "spec": copy.deepcopy(pod_spec),
                },
            },
            "status": {
                "conditions": [{"type": "Complete", "status": "True"}],
                "succeeded": 1,
                "completionTime": "2026-08-31T23:59:59Z",
            },
        },
        "pod": {
            "metadata": {
                "name": "registration-pod",
                "uid": "registration-pod-uid",
                "labels": copy.deepcopy(contract["pod_labels"]),
                "ownerReferences": [
                    {"kind": "Job", "name": name, "uid": job_uid, "controller": True}
                ],
            },
            "spec": copy.deepcopy(pod_spec),
            "status": {"containerStatuses": [{"name": "register", "imageID": REGISTRATION_IMAGE}]},
        },
        "config_map": {
            "metadata": {
                "namespace": "fleet-train-jobs",
                "name": name,
                "uid": "registration-config-uid",
                "resourceVersion": "1",
            },
            "immutable": True,
            "data": {
                "training__init__.py": (
                    ROOT / "evals/post_sft/runtime/training__init__.py"
                ).read_text(),
                "training_io.py": (ROOT / "training/io.py").read_text(),
                "training_register_post_sft.py": (
                    ROOT / "training/register_post_sft.py"
                ).read_text(),
                "serving-registration-receipt.json": json.dumps(serving),
            },
        },
    }


def _assembled_registration(paired_identity, serving):
    export = _export(_selection())
    return assemble_registration_completion(
        paired_identity,
        _registration_job_observation(serving),
        {
            "id": paired_identity["post_sft"]["model"],
            "object": "inference.model",
            "phase": "pending",
            "registration": copy.deepcopy(serving["registration"]),
        },
        serving,
        export,
    )


def test_registration_accepts_only_the_exact_server_owned_region_label(tmp_path):
    inputs = _paired_inputs(tmp_path)
    _, paired_identity = derive_post_sft_qwen_pair(**inputs)
    serving = inputs["post_serving_receipt"]
    api_result = {
        "id": paired_identity["post_sft"]["model"],
        "object": "inference.model",
        "phase": "pending",
        "registration": copy.deepcopy(serving["registration"]),
    }
    observation = _registration_job_observation(serving)
    observation["pod"]["metadata"]["labels"]["topology.kubernetes.io/region"] = "eu-west2"
    observation["pod"]["spec"]["imagePullSecrets"] = [{"name": "ecr-pull"}]
    observation["pod"]["spec"]["tolerations"].extend(
        [
            {
                "effect": "NoExecute",
                "key": "node.kubernetes.io/not-ready",
                "operator": "Exists",
                "tolerationSeconds": 300,
            },
            {
                "effect": "NoExecute",
                "key": "node.kubernetes.io/unreachable",
                "operator": "Exists",
                "tolerationSeconds": 300,
            },
        ]
    )
    assemble_registration_completion(
        paired_identity,
        observation,
        api_result,
        serving,
        _export(_selection()),
    )

    observation["pod"]["metadata"]["labels"]["topology.kubernetes.io/region"] = "unexpected-region"
    with pytest.raises(ValueError, match="server-owned label .* differs"):
        assemble_registration_completion(
            paired_identity,
            observation,
            api_result,
            serving,
            _export(_selection()),
        )

    observation["pod"]["metadata"]["labels"]["topology.kubernetes.io/region"] = "eu-west2"
    observation["pod"]["spec"]["imagePullSecrets"] = [{"name": "unreviewed"}]
    with pytest.raises(ValueError, match="execution spec differs"):
        assemble_registration_completion(
            paired_identity,
            observation,
            api_result,
            serving,
            _export(_selection()),
        )


def _base_artifact_receipt():
    plan_path = ROOT / "configs/evaluation/qwen36-27b-ft-run-574bd7b3-post-sft.json"
    plan = json.loads(plan_path.read_text())
    model = plan["base_model"]
    contract = plan["evidence_execution"]["base_artifact_inspector"]
    inspection = {
        "schema": "cyber_sft_hf_output_inspection_v1",
        "root": BASE_MODEL_ROOT,
        "format": "safetensors",
        "dtype": "bf16",
        "all_shards_present": True,
        "safetensors_load_passed": True,
        "parameter_count_matches": True,
        "weights_manifest_sha256": model["weights_manifest_sha256"],
        "tokenizer_manifest_sha256": model["tokenizer_manifest_sha256"],
        "chat_template_sha256": model["chat_template_sha256"],
        "config_sha256": model["config_sha256"],
        "sidecar_sha256": model["runtime_sidecar_sha256"],
    }
    inspection["sidecar_sha256"] = {
        **inspection["sidecar_sha256"],
        ".cyber-post-train-lock.json": "sha256:" + "a" * 64,
        ".fleet-acceptance.json": "sha256:" + "f" * 64,
        ".gitattributes": "sha256:" + "c" * 64,
        "LICENSE": "sha256:" + "d" * 64,
        "README.md": "sha256:" + "e" * 64,
    }
    artifacts = _artifact_projection(
        inspection,
        "base fixture",
        allowed_non_serving_sidecars=BASE_NON_SERVING_SIDECARS,
    )
    receipt = {
        "schema": "webexploitbench_base_artifact_inspection_v1",
        "model": {
            "id": BASE_MODEL_ID,
            "repository": BASE_MODEL_REPOSITORY,
            "revision": BASE_MODEL_REVISION,
            "source_path": BASE_MODEL_ROOT,
        },
        "checkpoint_lock_sha256": "sha256:" + "9" * 64,
        "inspection": inspection,
        "artifact_identity": artifacts,
        "execution": {
            "job": {
                "namespace": ARTIFACT_NAMESPACE,
                "name": ARTIFACT_JOB,
                "uid": "artifact-job-uid",
                "spec": {
                    "template": {
                        "spec": {
                            **{
                                key: copy.deepcopy(value)
                                for key, value in contract["pod_spec"].items()
                                if key != "container"
                            },
                            "containers": [copy.deepcopy(contract["pod_spec"]["container"])],
                        }
                    }
                },
            },
            "pod": {
                "name": "artifact-pod",
                "uid": "artifact-pod-uid",
                "reviewed_spec_projection": copy.deepcopy(contract["pod_spec"]),
                "reviewed_spec_projection_sha256": digest_json(contract["pod_spec"]),
            },
            "image": ARTIFACT_IMAGE,
            "image_id": ARTIFACT_IMAGE,
            "command_sha256": ARTIFACT_COMMAND_SHA256,
            "config_map": {
                "name": ARTIFACT_CONFIG_MAP,
                "uid": "artifact-config-uid",
                "resource_version": "1",
                "immutable": True,
                "data_sha256": "sha256:" + "7" * 64,
                "mounted_file_sha256": {
                    **copy.deepcopy(contract["config_map_code_sha256"]),
                    "post-sft-plan.json": file_sha256(plan_path),
                },
            },
        },
    }
    job_spec = receipt["execution"]["job"]["spec"]
    job_projection = copy.deepcopy(contract["pod_spec"])
    receipt["execution"]["job"].update(
        {
            "spec_sha256": digest_json(job_spec),
            "reviewed_spec_projection": job_projection,
            "reviewed_spec_projection_sha256": digest_json(job_projection),
        }
    )
    receipt["base_artifact_receipt_sha256"] = digest_json(receipt)
    return receipt


def _live_observations(paired_identity, expected_live, post_registration):
    def response(model, token="parity"):
        return {
            "model": model,
            "choices": [
                {
                    "logprobs": {
                        "content": [
                            {
                                "token": token,
                                "bytes": list(token.encode()),
                                "logprob": -0.125,
                            }
                        ]
                    }
                }
            ],
        }

    arms = {}
    base_registration = json.loads(
        (ROOT / "evals/webexploitbench/serving/qwen36-27b-6a9e13bd-registration.json").read_text()
    )
    for label, identity_key in (("base", "baseline"), ("post_sft", "post_sft")):
        model = paired_identity[identity_key]["model"]
        expected = expected_live["arms"][label]
        serving_registration = base_registration if label == "base" else post_registration
        live_spec = copy.deepcopy(serving_registration["spec"])
        live_spec["capabilities"] = sorted(live_spec["capabilities"])
        token = "base" if label == "base" else "post"
        arms[label] = {
            "inference_model": {
                "metadata": {
                    "name": model,
                    "uid": label + "-uid",
                    "generation": 1,
                },
                "spec": live_spec,
                "status": {
                    "phase": "ready",
                    "readyReplicas": 1,
                    "observedGeneration": 1,
                    "conditions": [{"type": "Ready", "status": "True"}],
                },
            },
            "model_info": copy.deepcopy(expected["model_info"]),
            "server_info": copy.deepcopy(expected["server_info"]),
            "tool_response": {
                "model": model,
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "identity",
                                        "arguments": json.dumps({"value": "parity"}),
                                    }
                                }
                            ]
                        }
                    }
                ],
            },
            "logit_responses": [response(model, token), response(model, token)],
        }
    return {"arms": arms}


def test_production_evidence_builders_bind_job_export_and_live_routes(tmp_path):
    inputs = _paired_inputs(tmp_path)
    _, paired_identity = derive_post_sft_qwen_pair(**inputs)
    registration = _assembled_registration(paired_identity, inputs["post_serving_receipt"])
    _, expected_live = _post_sft_launch_evidence(
        paired_identity, inputs["post_serving_receipt"]["registration"]
    )
    live = assemble_live_parity(
        paired_identity,
        registration,
        _live_observations(
            paired_identity, expected_live, inputs["post_serving_receipt"]["registration"]
        ),
        inputs["base_registration"],
        inputs["post_serving_receipt"],
        _base_artifact_receipt(),
        _export(_selection()),
        expected_live["probes"]["tokenizer"],
    )
    assert registration["api_result"]["weights_manifest_sha256"] == "sha256:" + "6" * 64
    assert (
        registration["source_receipts"]["export_receipt_sha256"]
        == _export(_selection())["export_receipt_sha256"]
    )
    assert live["artifact_identity"]["only_difference"] == "checkpoint_weights"
    assert "parity" not in json.dumps(live["probes"]["fixed_prompt_logits"])
    validate_post_sft_launch_evidence(paired_identity, registration, live)


def test_base_artifact_receipt_is_digest_bound_and_names_runtime_provenance():
    receipt = _base_artifact_receipt()
    assert receipt["artifact_identity"]["weights_manifest_sha256"].startswith("sha256:")
    assert receipt["execution"]["job"]["name"] == ARTIFACT_JOB
    assert receipt["base_artifact_receipt_sha256"] == digest_json(
        {key: value for key, value in receipt.items() if key != "base_artifact_receipt_sha256"}
    )


def test_base_artifact_config_map_files_match_the_frozen_execution_contract():
    plan = json.loads(
        (ROOT / "configs/evaluation/qwen36-27b-ft-run-574bd7b3-post-sft.json").read_text()
    )
    contract = plan["evidence_execution"]["base_artifact_inspector"]
    assert set(ARTIFACT_CONFIG_MAP_FILES.values()) == set(contract["config_map_code_sha256"])
    assert set(ARTIFACT_CONFIG_MAP_FILES) | {"post-sft-plan.json"} == {
        item["key"]
        for volume in contract["pod_spec"]["volumes"]
        if volume["name"] == "bundle"
        for item in volume["configMap"]["items"]
    }


def test_web_evidence_outputs_are_no_clobber_including_dangling_symlinks(tmp_path):
    output = tmp_path / "receipt.json"
    _atomic_write_json_new(output, {"ready": True})
    with pytest.raises(ValueError, match="pre-existing output"):
        _atomic_write_json_new(output, {"ready": False})
    dangling = tmp_path / "dangling.json"
    dangling.symlink_to(tmp_path / "missing-target")
    with pytest.raises(ValueError, match="pre-existing output"):
        _atomic_write_json_new(dangling, {"ready": True})


def test_tool_probe_disables_thinking_and_remains_bounded():
    request = _tool_request("model")
    assert request["chat_template_kwargs"] == {"enable_thinking": False}
    assert request["temperature"] == 0
    assert request["max_tokens"] == 128


def test_live_builder_reads_only_the_exact_frozen_tokenizer_receipt(tmp_path):
    source = ROOT / "docs/evidence/post_sft/2026-08-31-tokenizer-equivalence.json"
    projection = _read_tokenizer_probe(str(source))
    assert projection == {
        "receipt_sha256": (
            "sha256:3f2f72d77fda3d9e7a68cfa2ee16df031675e8f380266aa1a1eb98b00833b49e"
        ),
        "base_passed": True,
        "post_sft_passed": True,
        "exact_encode_decode_match": True,
    }
    changed = tmp_path / "tokenizer.json"
    value = json.loads(source.read_text())
    value["corpus_parity"]["all_decode_parity"] = False
    changed.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="differs from the frozen receipt"):
        _read_tokenizer_probe(str(changed))


def test_registration_builder_rejects_incomplete_or_wrong_export(tmp_path):
    inputs = _paired_inputs(tmp_path)
    _, paired_identity = derive_post_sft_qwen_pair(**inputs)
    job = _registration_job_observation(inputs["post_serving_receipt"])
    job["job"]["status"]["conditions"] = [{"type": "Failed", "status": "True"}]
    with pytest.raises(ValueError, match="not exclusively Complete"):
        assemble_registration_completion(
            paired_identity,
            job,
            {
                "id": paired_identity["post_sft"]["model"],
                "phase": "pending",
                "registration": inputs["post_serving_receipt"]["registration"],
            },
            inputs["post_serving_receipt"],
            _export(_selection()),
        )

    wrong_export = _export(_selection())
    wrong_export["output"]["weights_manifest_sha256"] = "sha256:" + "e" * 64
    wrong_export["export_receipt_sha256"] = digest_json(
        {key: value for key, value in wrong_export.items() if key != "export_receipt_sha256"}
    )
    with pytest.raises(ValueError, match="does not bind the supplied export"):
        assemble_registration_completion(
            paired_identity,
            _registration_job_observation(inputs["post_serving_receipt"]),
            {
                "id": paired_identity["post_sft"]["model"],
                "phase": "pending",
                "registration": inputs["post_serving_receipt"]["registration"],
            },
            inputs["post_serving_receipt"],
            wrong_export,
        )


def test_registration_builder_rejects_unreviewed_workload_and_mutable_configmap(tmp_path):
    inputs = _paired_inputs(tmp_path)
    _, paired_identity = derive_post_sft_qwen_pair(**inputs)
    serving = inputs["post_serving_receipt"]
    api_result = {
        "id": paired_identity["post_sft"]["model"],
        "object": "inference.model",
        "phase": "pending",
        "registration": copy.deepcopy(serving["registration"]),
    }
    observation = _registration_job_observation(serving)
    extra = {"name": "unreviewed", "image": "alpine:latest"}
    observation["job"]["spec"]["template"]["spec"]["containers"].append(extra)
    observation["pod"]["spec"]["containers"].append(extra)
    with pytest.raises(ValueError, match="containers names differ"):
        assemble_registration_completion(
            paired_identity,
            observation,
            api_result,
            serving,
            _export(_selection()),
        )

    observation = _registration_job_observation(serving)
    observation["job"]["spec"]["template"]["spec"]["imagePullSecrets"] = [{"name": "unreviewed"}]
    observation["pod"]["spec"]["imagePullSecrets"] = [{"name": "unreviewed"}]
    with pytest.raises(ValueError, match="execution spec differs"):
        assemble_registration_completion(
            paired_identity,
            observation,
            api_result,
            serving,
            _export(_selection()),
        )

    observation = _registration_job_observation(serving)
    observation["config_map"]["immutable"] = False
    with pytest.raises(ValueError, match="ConfigMap is not immutable"):
        assemble_registration_completion(
            paired_identity,
            observation,
            api_result,
            serving,
            _export(_selection()),
        )


def test_artifact_evidence_rejects_unexpected_sidecars_and_self_asserted_execution():
    base = _base_artifact_receipt()
    base["inspection"]["sidecar_sha256"]["runtime_override.json"] = "sha256:" + "1" * 64
    with pytest.raises(ValueError, match="unexpected=.*runtime_override.json"):
        _artifact_projection(
            base["inspection"],
            "adversarial base",
            allowed_non_serving_sidecars=BASE_NON_SERVING_SIDECARS,
        )

    base = _base_artifact_receipt()
    base["execution"]["config_map"]["mounted_file_sha256"] = {"untrusted.py": "sha256:" + "0" * 64}
    base["base_artifact_receipt_sha256"] = digest_json(
        {key: value for key, value in base.items() if key != "base_artifact_receipt_sha256"}
    )
    with pytest.raises(ValueError, match="mounted bytes differ from the frozen plan"):
        _artifacts_from_receipts(base, _export(_selection()))


def test_live_builder_rejects_changed_cr_spec_and_nondeterminism(tmp_path):
    inputs = _paired_inputs(tmp_path)
    _, paired_identity = derive_post_sft_qwen_pair(**inputs)
    registration = _assembled_registration(paired_identity, inputs["post_serving_receipt"])
    _, expected_live = _post_sft_launch_evidence(
        paired_identity, inputs["post_serving_receipt"]["registration"]
    )
    observations = _live_observations(
        paired_identity, expected_live, inputs["post_serving_receipt"]["registration"]
    )
    observations["arms"]["base"]["inference_model"]["spec"]["runtime"]["args"].append("--drifted")
    with pytest.raises(ValueError, match="spec differs"):
        assemble_live_parity(
            paired_identity,
            registration,
            observations,
            inputs["base_registration"],
            inputs["post_serving_receipt"],
            _base_artifact_receipt(),
            _export(_selection()),
            expected_live["probes"]["tokenizer"],
        )

    observations = _live_observations(
        paired_identity, expected_live, inputs["post_serving_receipt"]["registration"]
    )
    observations["arms"]["post_sft"]["logit_responses"][1]["choices"][0]["logprobs"]["content"][0][
        "logprob"
    ] = -0.5
    with pytest.raises(ValueError, match="not deterministic"):
        assemble_live_parity(
            paired_identity,
            registration,
            observations,
            inputs["base_registration"],
            inputs["post_serving_receipt"],
            _base_artifact_receipt(),
            _export(_selection()),
            expected_live["probes"]["tokenizer"],
        )


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


def test_comparison_receipt_binds_all_four_handoffs(tmp_path):
    selection = _selection()
    export = _export(selection)
    base_registration = json.loads(
        (ROOT / "evals/webexploitbench/serving/qwen36-27b-6a9e13bd-registration.json").read_text()
    )
    serving = derive_post_sft_registration(
        base_registration,
        selection,
        export,
        expected_tokenizer_manifest_sha256=export["output"]["tokenizer_manifest_sha256"],
        expected_chat_template_sha256=export["output"]["chat_template_sha256"],
        expected_config_sha256=export["output"]["config_sha256"],
        expected_export_binding=_export_binding(),
    )
    split = json.loads((ROOT / "configs/data/fleet-a62-task-split-v1.json").read_text())
    sft = json.loads((ROOT / "configs/runs/qwen36-27b-sft-full.json").read_text())
    holdout = build_fleet_test_holdout_receipt(split, sft)
    _, paired_identity = derive_post_sft_qwen_pair(**_paired_inputs(tmp_path))
    receipt = build_post_sft_comparison_receipt(
        selection=selection,
        export=export,
        serving=serving,
        base_webexploit_config_sha256="sha256:" + "b" * 64,
        post_webexploit_config_sha256="sha256:" + "c" * 64,
        webexploit_paired_identity=paired_identity,
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
        ("web terminal", plan["webexploitbench"]["baseline_terminal_receipt"]),
        ("web protocol", plan["webexploitbench"]["baseline_protocol"]),
        ("web harness", plan["webexploitbench"]["harness_lock"]),
        ("tokenizer equivalence", plan["base_model"]["tokenizer_equivalence_evidence"]),
        ("split", plan["fleet"]["split_manifest"]),
        ("sft", plan["fleet"]["sft_config"]),
    ):
        assert file_sha256(ROOT / binding["path"]) == binding["sha256"], section
    assert plan["webexploitbench"]["primary_harness"] == "qwen_code"
    assert plan["webexploitbench"]["primary_harness_version"] == "0.22.3"
    assert plan["webexploitbench"]["paired_identity_required"] is True
