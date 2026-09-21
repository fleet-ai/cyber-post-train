"""Fresh75 evaluation provenance uses public identities only, never task payloads."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from cyber_post_train.jobs import digest
from evals.fleet import cluster_entry, evaluate, model_artifact
from training.io import file_sha256

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs/evaluation"
SUCCESSOR = CONFIGS / "qwen38-fresh75-fleet-dev17-opencode-seed43-pass1-v2.json"
ARTIFACT_PACKET = CONFIGS / "qwen38-fresh75-fleet-dev17-opencode-seed43-pass1-v2.artifact.json"
REPAIR = ROOT / "docs/evidence/qwen38-fresh75-step230-eval-path-repair-20260921.json"
ACCEPTANCE = ROOT / "docs/evidence/qwen38-fresh75-step230-reload-accepted-20260915.json"
FALSE_PATH = "/mnt/sfs/jobs/chris-q38-fresh75-e1-v2/hf-export-v1"
ACCEPTED_PATH = "/mnt/sfs/jobs/chris-q38-f75-max-full-v4/hf-export-step230-v1"
REVISION = "sha256:36eec01f1dd3f0d47f6b099e79970d9533cf59c37d8e15478907413dd162e029"

HISTORICAL = (
    CONFIGS / "qwen38-fresh75-fleet-dev17-opencode-seed43-pass1-v1.json",
    CONFIGS / "qwen38-fresh75-fleet-dev17-opencode-pass1-v2.json",
    CONFIGS / "qwen38-fresh75-fleet-dev17-matched-pass1-v2.json",
)


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def test_repair_receipt_preserves_old_bytes_and_binds_the_accepted_chain() -> None:
    repair = read(REPAIR)
    assert repair["sha256"] == digest(
        {key: value for key, value in repair.items() if key != "sha256"}
    )
    assert repair["status"] == "review_only_do_not_launch"
    assert repair["launchable"] is False
    accepted = repair["accepted_artifact"]
    assert accepted == {
        "checkpoint_path": "/mnt/sfs/jobs/chris-q38-f75-max-full-v4/checkpoints/global_step_230",
        "checkpoint_manifest_path": (
            "/mnt/sfs/jobs/chris-q38-f75-max-full-v4/checkpoint-seals-step230-v1/step-230.json"
        ),
        "checkpoint_manifest_file_sha256": (
            "sha256:bf92e29d9f19dd589201214f25fe8d0e3b0f65de14ec08fc573217f50580a486"
        ),
        "checkpoint_manifest_receipt_sha256": (
            "sha256:9542502631b47cd730ccecb5938650103ed999bf4a4a6cf7d16d9be09c317cab"
        ),
        "export_path": ACCEPTED_PATH,
        "export_receipt_file_sha256": (
            "sha256:7b4ed0c07cc0f9ed6a5cbdac7d3ad0b3bc08ba63447fc2d56e47885c3daa7c4f"
        ),
        "export_receipt_sha256": (
            "sha256:1bd81c78575c32fa52dc4ab9c7002d95db5028047fa52cc36297aa1954c3d2d3"
        ),
        "payload_model_revision": REVISION,
        "trained_tensors": 1184,
        "restored_mtp_tensors": 15,
        "total_tensors": 1199,
        "tensor_bytes": 55_562_855_904,
        "gpu_reload_receipt_path": (
            "/mnt/sfs/jobs/chris-q38-f75-step230-export-gpu-check-v1/GPU_CHECK.json"
        ),
        "gpu_reload_receipt_file_sha256": (
            "sha256:30cb5f9687d1cb08f8e78b9f425391807f0f16c0cefe712d97d9b155cf73b791"
        ),
        "gpu_reload_receipt_sha256": (
            "sha256:43e40b5397f3db4fb6b603cb9774f9aa862d570c3191d8f71e7c14b599be6f01"
        ),
        "gpu_reload_rayjob_uid": "64bc14b2-c418-4706-acad-7fc1535dfb9d",
        "gpu_reload_pod_uid": "0b0c47e8-39a4-4ec5-a2ff-f336ae44402a",
        "gpu_reload_was_finite_two_token_zero_optimizer_step": True,
        "gpu_allocation_released": True,
    }
    rows = {ROOT / row["path"]: row for row in repair["historical_non_launchable_configs"]}
    assert set(rows) == set(HISTORICAL)
    for path in HISTORICAL:
        row = rows[path]
        assert row["file_sha256"] == file_sha256(path)
        assert read(path)["models"]["fresh75-step230"]["repository"] == FALSE_PATH
        assert row["config_path_binding_matches_accepted_export"] is False
        assert row["sfs_path_presence"] == "not_reopened_in_this_change"
        assert row["launchable"] is False


def test_successor_is_new_non_authorized_identity_with_exact_accepted_artifact() -> None:
    config, packet, repair, acceptance = (
        read(SUCCESSOR),
        read(ARTIFACT_PACKET),
        read(REPAIR),
        read(ACCEPTANCE),
    )
    plan = evaluate.compile_eval(config, relative_to=SUCCESSOR.parent)
    model = plan["models"]["fresh75-step230"]
    binding = packet["models"]["fresh75-step230"]
    assert config["name"] == "q38-dev17-s43-fresh75-p1-v2"
    assert packet["campaign_name"] == config["name"]
    assert model["repository"] == repair["accepted_artifact"]["export_path"] == ACCEPTED_PATH
    assert model["revision"] == repair["accepted_artifact"]["payload_model_revision"] == REVISION
    assert model_artifact.validate_packet(plan["models"], packet) == packet
    assert model_artifact.validate_binding(model, binding) == binding
    assert binding["payload"] == {
        "manifest_sha256": REVISION,
        "file_count": 29,
        "trained_tensors": 1184,
        "restored_mtp_tensors": sorted(model_artifact.EXACT_MTP_TENSORS),
        "tensor_count": 1199,
        "tensor_bytes": 55_562_855_904,
    }
    qualification = binding["qualification"]
    assert qualification["acceptance_evidence_file_sha256"] == file_sha256(ACCEPTANCE)
    assert qualification["acceptance_evidence_sha256"] == acceptance["sha256"]
    assert repair["successor"]["file_sha256"] == file_sha256(SUCCESSOR)
    assert repair["successor"]["artifact_packet"] == {
        "path": str(ARTIFACT_PACKET.relative_to(ROOT)),
        "file_sha256": file_sha256(ARTIFACT_PACKET),
        "sha256": packet["sha256"],
        "authorization": "provenance_only_not_launch_authorization",
    }
    assert repair["successor"]["launchable"] is False
    assert repair["operation"] == {
        "jobs_submitted": 0,
        "resources_created": 0,
        "external_evaluations_launched": 0,
        "cluster_mutations_performed": 0,
    }


def test_successor_changes_only_create_once_identity_and_false_repository_path() -> None:
    historical = read(HISTORICAL[0])
    successor = read(SUCCESSOR)
    historical["name"] = successor["name"]
    historical["models"]["fresh75-step230"]["repository"] = ACCEPTED_PATH
    assert successor == historical


@pytest.mark.parametrize("path", HISTORICAL)
def test_all_false_path_historical_configs_are_rejected_from_live_execution(path: Path) -> None:
    config = read(path)
    assert config["models"]["fresh75-step230"]["repository"] == FALSE_PATH
    with pytest.raises(ValueError, match="requires an exact accepted artifact binding packet"):
        model_artifact.validate_live_models(config["models"], None)


def _write_receipt(path: Path, value: dict) -> tuple[str, str]:
    unsigned = copy.deepcopy(value)
    receipt = hashlib.sha256(
        json.dumps(unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    value = {**unsigned, "receipt_sha256": receipt}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return file_sha256(path), "sha256:" + receipt


def _packet_with_digest(value: dict) -> dict:
    packet = copy.deepcopy(value)
    packet["sha256"] = hashlib.sha256(
        json.dumps(packet, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return packet


def synthetic_local_model(tmp_path: Path, monkeypatch) -> tuple[dict, dict]:
    local_root = str(tmp_path / "jobs") + "/"
    monkeypatch.setattr(model_artifact, "LOCAL_ROOT", local_root)
    repository = tmp_path / "jobs/run/export"
    checkpoint_path = tmp_path / "jobs/run/checkpoints/global_step_1"
    checkpoint_file, checkpoint_receipt = _write_receipt(
        tmp_path / "jobs/run/seals/step-1.json",
        {
            "schema": "cyber_skyrl_checkpoint_manifest_v1",
            "checkpoint_path": str(checkpoint_path),
            "optimizer_step": 1,
            "world_size": 8,
            "total_bytes": 100,
            "gpu_reload_verified": False,
        },
    )
    files = {"model.safetensors": {"bytes": 10, "sha256": "0" * 64}}
    revision = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )
    export_file, export_receipt = _write_receipt(
        repository / "EXPORT.json",
        {
            "schema": "cyber_native_checkpoint_hf_export_v1",
            "source_checkpoint_receipt_sha256": checkpoint_receipt.removeprefix("sha256:"),
            "source_manifest_file_sha256": checkpoint_file.removeprefix("sha256:"),
            "source_plan_sha256": "3" * 64,
            "model_repo": "Qwen/synthetic",
            "model_revision": "4" * 40,
            "output_root": str(repository),
            "optimizer_step": 1,
            "optimizer_steps_executed": 0,
            "gpu_reload_verified": False,
            "trained_tensors": 1,
            "restored_base_tensors": sorted(model_artifact.EXACT_MTP_TENSORS),
            "tensor_values": 16,
            "tensor_bytes": 32,
            "dtype": "BF16",
            "files": files,
            "source_inventory_sizes_mtimes_unchanged": True,
            "all_output_tensors_reopened_equal": True,
        },
    )
    gpu_file, gpu_receipt = _write_receipt(
        tmp_path / "jobs/run/reload/GPU_CHECK.json",
        {
            "schema": "cyber_hf_export_check_v1",
            "status": "passed",
            "export_sha256": export_file.removeprefix("sha256:"),
            "export_receipt_sha256": export_receipt.removeprefix("sha256:"),
            "optimizer_steps_executed": 0,
            "gpus": 1,
            "gpu_reload_verified": True,
            "serving_qualified": False,
            "synthetic_only": True,
            "source_unchanged": True,
            "attention_implementation": "eager",
            "finite_logits": True,
            "generated_tokens": 2,
        },
    )
    model = {
        "repository": str(repository),
        "revision": revision,
        "session_model": "qwen/synthetic",
    }
    binding = {
        "schema": model_artifact.SCHEMA,
        "checkpoint_manifest": {
            "path": str(tmp_path / "jobs/run/seals/step-1.json"),
            "file_sha256": checkpoint_file,
            "receipt_sha256": checkpoint_receipt,
            "required_fields": {
                "schema": "cyber_skyrl_checkpoint_manifest_v1",
                "checkpoint_path": str(checkpoint_path),
                "optimizer_step": 1,
                "world_size": 8,
                "total_bytes": 100,
                "gpu_reload_verified": False,
            },
        },
        "export_receipt": {
            "path": str(repository / "EXPORT.json"),
            "file_sha256": export_file,
            "receipt_sha256": export_receipt,
            "required_fields": {
                "schema": "cyber_native_checkpoint_hf_export_v1",
                "source_checkpoint_receipt_sha256": checkpoint_receipt.removeprefix("sha256:"),
                "source_manifest_file_sha256": checkpoint_file.removeprefix("sha256:"),
                "source_plan_sha256": "3" * 64,
                "model_repo": "Qwen/synthetic",
                "model_revision": "4" * 40,
                "output_root": str(repository),
                "optimizer_step": 1,
                "optimizer_steps_executed": 0,
                "gpu_reload_verified": False,
                "trained_tensors": 1,
                "tensor_values": 16,
                "tensor_bytes": 32,
                "dtype": "BF16",
                "source_inventory_sizes_mtimes_unchanged": True,
                "all_output_tensors_reopened_equal": True,
            },
        },
        "gpu_reload_receipt": {
            "path": str(tmp_path / "jobs/run/reload/GPU_CHECK.json"),
            "file_sha256": gpu_file,
            "receipt_sha256": gpu_receipt,
            "required_fields": {
                "schema": "cyber_hf_export_check_v1",
                "status": "passed",
                "export_sha256": export_file.removeprefix("sha256:"),
                "export_receipt_sha256": export_receipt.removeprefix("sha256:"),
                "optimizer_steps_executed": 0,
                "gpus": 1,
                "gpu_reload_verified": True,
                "serving_qualified": False,
                "synthetic_only": True,
                "source_unchanged": True,
                "attention_implementation": "eager",
                "finite_logits": True,
                "generated_tokens": 2,
            },
        },
        "payload": {
            "manifest_sha256": revision,
            "file_count": 1,
            "trained_tensors": 1,
            "restored_mtp_tensors": sorted(model_artifact.EXACT_MTP_TENSORS),
            "tensor_count": 16,
            "tensor_bytes": 32,
        },
        "qualification": {
            "acceptance_evidence_path": "docs/evidence/synthetic.json",
            "acceptance_evidence_file_sha256": "sha256:" + "1" * 64,
            "acceptance_evidence_sha256": "sha256:" + "2" * 64,
            "gpu_reload_rayjob_uid": "11111111-1111-4111-8111-111111111111",
            "gpu_reload_pod_uid": "22222222-2222-4222-8222-222222222222",
            "gpu_allocation_released": True,
        },
    }
    packet = _packet_with_digest(
        {
            "schema": model_artifact.PACKET_SCHEMA,
            "campaign_name": "synthetic",
            "models": {"student": binding},
            "authorization": "provenance_only_not_launch_authorization",
        }
    )
    return model, packet


def test_live_guard_reopens_exact_checkpoint_export_and_reload_chain(tmp_path, monkeypatch) -> None:
    model, packet = synthetic_local_model(tmp_path, monkeypatch)
    proof = model_artifact.validate_live_models({"student": model}, packet)["student"]
    assert proof["repository"] == model["repository"]
    assert proof["revision"] == model["revision"]
    assert proof["gpu_allocation_released"] is True


@pytest.mark.parametrize("fault", ["swapped_hash", "missing_hash", "wrong_repository"])
def test_live_guard_rejects_swapped_missing_or_wrong_receipt_binding(
    tmp_path, monkeypatch, fault
) -> None:
    model, packet = synthetic_local_model(tmp_path, monkeypatch)
    binding = packet["models"]["student"]
    if fault == "swapped_hash":
        binding["export_receipt"]["file_sha256"] = binding["gpu_reload_receipt"]["file_sha256"]
    elif fault == "missing_hash":
        binding["checkpoint_manifest"].pop("receipt_sha256")
    else:
        model["repository"] += "-different"
    packet = _packet_with_digest({key: value for key, value in packet.items() if key != "sha256"})
    with pytest.raises((ValueError, FileNotFoundError)):
        model_artifact.validate_live_models({"student": model}, packet)


def test_cluster_entry_rejects_historical_config_before_image_staging(
    tmp_path, monkeypatch
) -> None:
    staged = []
    monkeypatch.setattr(cluster_entry, "stage_images", lambda **kwargs: staged.append(kwargs))
    args = SimpleNamespace(
        config=str(HISTORICAL[0]),
        output=str(tmp_path / "output"),
        database="unused",
        harness_tar="unused",
        harness_receipt="unused",
        harness_receipt_sha256="sha256:" + "0" * 64,
    )
    with pytest.raises(ValueError, match="requires an exact accepted artifact binding packet"):
        cluster_entry.execute(args)
    assert staged == []
    assert not (tmp_path / "output").exists()
