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
SUCCESSOR = CONFIGS / "qwen38-fresh75-fleet-dev17-opencode-seed43-pass1-v3.json"
ARTIFACT_PACKET = CONFIGS / "qwen38-fresh75-fleet-dev17-opencode-seed43-pass1-v3.artifact.json"
RETIRED_SUCCESSOR = CONFIGS / "qwen38-fresh75-fleet-dev17-opencode-seed43-pass1-v2.json"
RETIRED_PACKET = CONFIGS / "qwen38-fresh75-fleet-dev17-opencode-seed43-pass1-v2.artifact.json"
REPAIR = ROOT / "docs/evidence/qwen38-fresh75-step230-eval-path-repair-20260921.json"
ACCEPTANCE = ROOT / "docs/evidence/qwen38-fresh75-step230-reload-accepted-20260915.json"
PROTOCOL = CONFIGS / "qwen38-fleet-dev17-seed43-matched-protocol-v1.json"
RECOVERY = ROOT / "docs/evidence/qwen38-fleet-dev17-seed43-recovery-policy-preview-20260921.json"
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
    scientific_config = copy.deepcopy(config)
    scientific_config.pop("model_artifact_binding")
    plan = evaluate.compile_eval(scientific_config, relative_to=SUCCESSOR.parent)
    model = plan["models"]["fresh75-step230"]
    binding = packet["models"]["fresh75-step230"]
    assert config["name"] == "q38-dev17-s43-fresh75-p1-v3"
    assert packet["campaign_name"] == config["name"]
    assert model["repository"] == repair["accepted_artifact"]["export_path"] == ACCEPTED_PATH
    assert model["revision"] == repair["accepted_artifact"]["payload_model_revision"] == REVISION
    assert model_artifact.validate_packet(plan["models"], packet) == packet
    assert model_artifact.validate_binding(model, binding) == binding
    model_artifact._validate_acceptance_evidence(  # noqa: SLF001
        acceptance,
        binding,
        binding["checkpoint_manifest"]["required_fields"],
        binding["export_receipt"]["required_fields"],
        binding["gpu_reload_receipt"]["required_fields"],
    )
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
    assert config["model_artifact_binding"] == {
        "validator_file_sha256": file_sha256(Path(model_artifact.__file__)),
        "packet_path": str(ARTIFACT_PACKET.relative_to(ROOT)),
        "packet_file_sha256": file_sha256(ARTIFACT_PACKET),
        "packet_sha256": "sha256:" + packet["sha256"],
        "acceptance_evidence_path": str(ACCEPTANCE.relative_to(ROOT)),
        "acceptance_evidence_file_sha256": file_sha256(ACCEPTANCE),
        "acceptance_evidence_sha256": acceptance["sha256"],
    }
    assert repair["successor"]["file_sha256"] == file_sha256(SUCCESSOR)
    assert repair["successor"]["artifact_packet"] == {
        "path": str(ARTIFACT_PACKET.relative_to(ROOT)),
        "file_sha256": file_sha256(ARTIFACT_PACKET),
        "sha256": packet["sha256"],
        "authorization": "provenance_only_not_launch_authorization",
    }
    assert repair["successor"]["launchable"] is False
    assert repair["runtime_repair"]["validator_file_sha256"] == file_sha256(
        Path(model_artifact.__file__)
    )
    assert repair["operation"] == {
        "jobs_submitted": 0,
        "resources_created": 0,
        "external_evaluations_launched": 0,
        "cluster_mutations_performed": 0,
    }
    artifact_plan = cluster_entry.model_artifact_plan(
        config,
        {
            "fresh75-step230": {
                "repository": ACCEPTED_PATH,
                "revision": REVISION,
                "gpu_reload_rayjob_uid": repair["accepted_artifact"]["gpu_reload_rayjob_uid"],
                "gpu_reload_pod_uid": repair["accepted_artifact"]["gpu_reload_pod_uid"],
                "gpu_allocation_released": True,
            }
        },
    )
    assert artifact_plan["schema"] == "fleet_eval_model_artifact_plan_v1"
    assert artifact_plan["binding"] == config["model_artifact_binding"]
    assert artifact_plan["sha256"] == digest(
        {key: value for key, value in artifact_plan.items() if key != "sha256"}
    )


def test_successor_changes_only_create_once_identity_and_false_repository_path() -> None:
    historical = read(HISTORICAL[0])
    successor = read(SUCCESSOR)
    successor.pop("model_artifact_binding")
    historical["name"] = successor["name"]
    historical["models"]["fresh75-step230"]["repository"] = ACCEPTED_PATH
    assert successor == historical


def test_reserved_v2_successor_is_retired_after_primary_claims() -> None:
    protocol, recovery, retired, packet = (
        read(PROTOCOL),
        read(RECOVERY),
        read(RETIRED_SUCCESSOR),
        read(RETIRED_PACKET),
    )
    arm = protocol["arms"]["fresh75"]
    census = recovery["score_blind_partial_census"]["fresh75"]
    assert retired["name"] == packet["campaign_name"] == "q38-dev17-s43-fresh75-p1-v2"
    assert arm["zero_claim_setup_successor_job"] == "chris-q38-dev17-s43-fresh75-p1-v2"
    assert arm["zero_claim_setup_successor_database"] == "q38_dev17_s43_fresh75_p1_v2"
    assert arm["zero_claim_setup_successor_output_root"] == (
        "/mnt/sfs/jobs/chris-q38-fleet-dev17-s43-fresh75-p1-v2"
    )
    assert protocol["retry_policy"]["setup_successor_forbidden_after_any_claim"] is True
    assert census["job_uid"] == "d8fd2e3f-db6a-4ec5-89f3-fec695668486"
    assert census["counts"]["accepted"] == 14
    with pytest.raises(ValueError, match="retired evaluation campaign"):
        model_artifact.validate_packet(retired["models"], packet)
    repair = read(REPAIR)["retired_non_launchable_successor"]
    assert repair["config_file_sha256"] == file_sha256(RETIRED_SUCCESSOR)
    assert repair["artifact_packet_file_sha256"] == file_sha256(RETIRED_PACKET)
    assert repair["primary_job_readback"]["job_uid"] == census["job_uid"]
    assert repair["claim_evidence"]["accepted_cells"] == census["counts"]["accepted"]
    assert repair["launchable"] is False


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


def _write_evidence(path: Path, value: dict) -> tuple[str, str]:
    evidence = {**copy.deepcopy(value), "sha256": "sha256:" + digest(value)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(evidence, sort_keys=True) + "\n")
    return file_sha256(path), evidence["sha256"]


def _write_packet(path: Path, value: dict) -> tuple[dict, str]:
    packet = _packet_with_digest({key: item for key, item in value.items() if key != "sha256"})
    path.write_text(json.dumps(packet, sort_keys=True) + "\n")
    return packet, file_sha256(path)


def synthetic_local_model(tmp_path: Path, monkeypatch) -> tuple[dict, dict, dict, Path, Path]:
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
    rayjob_uid = "11111111-1111-4111-8111-111111111111"
    pod_uid = "22222222-2222-4222-8222-222222222222"
    acceptance_path = tmp_path / "acceptance.json"
    acceptance_file, acceptance_self = _write_evidence(
        acceptance_path,
        {
            "schema": "cyber_qwen38_fresh75_step230_reload_acceptance_v1",
            "training": {
                "status": "SUCCEEDED",
                "checkpoint_path": str(checkpoint_path),
                "optimizer_steps_executed": 1,
                "checkpoint_manifest_file_sha256": checkpoint_file,
                "checkpoint_receipt_sha256": checkpoint_receipt,
            },
            "export": {
                "receipt_path": str(repository / "EXPORT.json"),
                "receipt_file_sha256": export_file,
                "receipt_sha256": export_receipt,
                "source_plan_sha256": "sha256:" + "3" * 64,
                "payload_manifest_sha256": revision,
                "dtype": "BF16",
                "tensor_count": 16,
                "tensor_bytes": 32,
                "payload_files": 1,
                "source_unchanged": True,
                "all_output_tensors_reopened_equal": True,
                "optimizer_steps_executed": 0,
            },
            "cpu_check": {
                "status": "passed",
                "gpus": 0,
                "gpu_reload_verified": False,
                "source_unchanged": True,
                "optimizer_steps_executed": 0,
            },
            "gpu_check": {
                "jobs_api_status": "SUCCEEDED",
                "jobs_api_output": str(tmp_path / "jobs/run/reload"),
                "rayjob_uid": rayjob_uid,
                "pod_uid": pod_uid,
                "gpu_check_file_sha256": gpu_file,
                "gpu_check_receipt_sha256": gpu_receipt,
                "status": "passed",
                "gpus": 1,
                "gpu_reload_verified": True,
                "finite_logits": True,
                "generated_tokens": 2,
                "source_unchanged": True,
                "serving_qualified": False,
                "optimizer_steps_executed": 0,
            },
            "release": {
                "raycluster_absent": True,
                "pod_absent": True,
                "workload_finished": True,
                "gpu_allocation_released": True,
            },
        },
    )
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
            "acceptance_evidence_file_sha256": acceptance_file,
            "acceptance_evidence_sha256": acceptance_self,
            "gpu_reload_rayjob_uid": rayjob_uid,
            "gpu_reload_pod_uid": pod_uid,
            "gpu_allocation_released": True,
        },
    }
    packet_path = tmp_path / "packet.json"
    packet, packet_file = _write_packet(
        packet_path,
        {
            "schema": model_artifact.PACKET_SCHEMA,
            "campaign_name": "synthetic",
            "models": {"student": binding},
            "authorization": "provenance_only_not_launch_authorization",
        },
    )
    config_binding = {
        "validator_file_sha256": file_sha256(Path(model_artifact.__file__)),
        "packet_path": "configs/evaluation/synthetic.json",
        "packet_file_sha256": packet_file,
        "packet_sha256": "sha256:" + packet["sha256"],
        "acceptance_evidence_path": "docs/evidence/synthetic.json",
        "acceptance_evidence_file_sha256": acceptance_file,
        "acceptance_evidence_sha256": acceptance_self,
    }
    return model, packet, config_binding, packet_path, acceptance_path


def test_live_guard_reopens_exact_checkpoint_export_and_reload_chain(tmp_path, monkeypatch) -> None:
    model, packet, config_binding, packet_path, acceptance_path = synthetic_local_model(
        tmp_path, monkeypatch
    )
    proof = model_artifact.validate_live_models(
        {"student": model},
        packet,
        config_binding=config_binding,
        packet_path=packet_path,
        acceptance_path=acceptance_path,
    )["student"]
    assert proof["repository"] == model["repository"]
    assert proof["revision"] == model["revision"]
    assert proof["gpu_allocation_released"] is True
    assert proof["gpu_reload_pod_uid"] == "22222222-2222-4222-8222-222222222222"
    assert proof["acceptance_evidence_sha256"] == config_binding["acceptance_evidence_sha256"]


@pytest.mark.parametrize("fault", ["swapped_hash", "missing_hash", "wrong_repository"])
def test_live_guard_rejects_swapped_missing_or_wrong_receipt_binding(
    tmp_path, monkeypatch, fault
) -> None:
    model, packet, config_binding, packet_path, acceptance_path = synthetic_local_model(
        tmp_path, monkeypatch
    )
    binding = packet["models"]["student"]
    if fault == "swapped_hash":
        binding["export_receipt"]["file_sha256"] = binding["gpu_reload_receipt"]["file_sha256"]
    elif fault == "missing_hash":
        binding["checkpoint_manifest"].pop("receipt_sha256")
    else:
        model["repository"] += "-different"
    packet, config_binding["packet_file_sha256"] = _write_packet(packet_path, packet)
    config_binding["packet_sha256"] = "sha256:" + packet["sha256"]
    with pytest.raises((ValueError, FileNotFoundError)):
        model_artifact.validate_live_models(
            {"student": model},
            packet,
            config_binding=config_binding,
            packet_path=packet_path,
            acceptance_path=acceptance_path,
        )


def test_live_guard_rejects_uid_mismatch_even_when_packet_anchor_is_updated(
    tmp_path, monkeypatch
) -> None:
    model, packet, config_binding, packet_path, acceptance_path = synthetic_local_model(
        tmp_path, monkeypatch
    )
    packet["models"]["student"]["qualification"]["gpu_reload_pod_uid"] = (
        "33333333-3333-4333-8333-333333333333"
    )
    packet, config_binding["packet_file_sha256"] = _write_packet(packet_path, packet)
    config_binding["packet_sha256"] = "sha256:" + packet["sha256"]
    with pytest.raises(ValueError, match="finite zero-update GPU reload"):
        model_artifact.validate_live_models(
            {"student": model},
            packet,
            config_binding=config_binding,
            packet_path=packet_path,
            acceptance_path=acceptance_path,
        )


def test_cluster_entry_rejects_same_campaign_packet_swap_before_image_staging(
    tmp_path, monkeypatch
) -> None:
    packet = read(ARTIFACT_PACKET)
    packet["models"]["fresh75-step230"]["qualification"]["gpu_reload_pod_uid"] = (
        "33333333-3333-4333-8333-333333333333"
    )
    swapped = _packet_with_digest({key: value for key, value in packet.items() if key != "sha256"})
    swapped_path = tmp_path / "model-artifact.json"
    swapped_path.write_text(json.dumps(swapped, sort_keys=True) + "\n")
    staged = []
    monkeypatch.setattr(cluster_entry, "stage_images", lambda **kwargs: staged.append(kwargs))
    args = SimpleNamespace(
        config=str(SUCCESSOR),
        output=str(tmp_path / "output"),
        database="unused",
        harness_tar="unused",
        harness_receipt="unused",
        harness_receipt_sha256="sha256:" + "0" * 64,
        model_artifact_binding=str(swapped_path),
        model_artifact_acceptance=str(ACCEPTANCE),
    )
    with pytest.raises(ValueError, match="config-bound packet"):
        cluster_entry.execute(args)
    assert staged == []
    assert not (tmp_path / "output").exists()


def test_cluster_entry_rejects_validator_drift_before_image_staging(tmp_path, monkeypatch) -> None:
    config = read(SUCCESSOR)
    config["model_artifact_binding"]["validator_file_sha256"] = "sha256:" + "0" * 64
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config, sort_keys=True) + "\n")
    staged = []
    monkeypatch.setattr(cluster_entry, "stage_images", lambda **kwargs: staged.append(kwargs))
    args = SimpleNamespace(
        config=str(config_path),
        output=str(tmp_path / "output"),
        database="unused",
        harness_tar="unused",
        harness_receipt="unused",
        harness_receipt_sha256="sha256:" + "0" * 64,
        model_artifact_binding=str(ARTIFACT_PACKET),
        model_artifact_acceptance=str(ACCEPTANCE),
    )
    with pytest.raises(ValueError, match="config-bound code"):
        cluster_entry.execute(args)
    assert staged == []
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize(
    "script",
    [
        ROOT / "evals/fleet/scripts/run_qwen38_dev17_matched_v1.sh",
        ROOT / "evals/fleet/scripts/run_qwen38_dev17_single_arm_v1.sh",
    ],
)
def test_cluster_wrappers_require_packet_and_acceptance_evidence_together(script: Path) -> None:
    text = script.read_text()
    assert "/bootstrap/model-artifact.json || -e /bootstrap/model-artifact-acceptance.json" in text
    assert "-f /bootstrap/model-artifact.json" in text
    assert "-f /bootstrap/model-artifact-acceptance.json" in text
    assert "--model-artifact-binding /bootstrap/model-artifact.json" in text
    assert "--model-artifact-acceptance /bootstrap/model-artifact-acceptance.json" in text


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
