"""Synthetic, value-free checks for the create-once Miles HF handoff."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from cyber_post_train.jobs import digest
from training import miles
from training import miles_hf_export as export


def _sealed(value: dict) -> dict:
    return {**value, "sha256": digest(value)}


def _write_json(path: Path, value: dict) -> str:
    path.write_text(json.dumps(value, sort_keys=True))
    return export._hash(path)


def _hf_tree(root: Path, tensors: dict[str, torch.Tensor]) -> list[dict]:
    root.mkdir()
    save_file(tensors, root / "model.safetensors")
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "total_size": sum(
                        tensor.numel() * tensor.element_size() for tensor in tensors.values()
                    )
                },
                "weight_map": {name: "model.safetensors" for name in tensors},
            }
        )
    )
    (root / "config.json").write_text("{}")
    (root / "tokenizer.json").write_text("{}")
    return [{"path": path.name, "sha256": export._hash(path)} for path in sorted(root.iterdir())]


@pytest.fixture
def case(tmp_path: Path, monkeypatch) -> dict:
    base = tmp_path / "base"
    language = torch.arange(8, dtype=torch.float32).reshape(2, 4).bfloat16()
    visual = torch.arange(6, dtype=torch.float32).reshape(2, 3).bfloat16()
    mtp = torch.arange(4, dtype=torch.float32).reshape(2, 2).bfloat16()
    files = _hf_tree(
        base,
        {
            "model.language_model.weight": language,
            "model.visual.weight": visual,
            "mtp.weight": mtp,
        },
    )
    raw = tmp_path / "raw"
    _hf_tree(raw, {"model.language_model.weight": language + 1})

    checkpoint_root = tmp_path / "source" / "checkpoints"
    generation = checkpoint_root / "iter_0000000"
    generation.mkdir(parents=True)
    (checkpoint_root / "latest_checkpointed_iteration.txt").write_text("0\n")
    (generation / ".metadata").write_bytes(b"metadata")
    (generation / "common.pt").write_bytes(b"trusted common metadata")
    for rank in range(8):
        (generation / f"__{rank}_0.distcp").write_bytes(f"rank-{rank}".encode())
    inventory = []
    for path in sorted(checkpoint_root.rglob("*")):
        if path.is_file():
            inventory.append(
                {
                    "path": str(path.relative_to(checkpoint_root)),
                    "size": path.stat().st_size,
                    "sha256": export._hash(path),
                }
            )
    model = {
        "repo": export.MODEL_REPO,
        "revision": export.MODEL_REVISION,
        "root": str(base),
        "files": files,
        "weight_manifest_sha256": "sha256:" + "a" * 64,
    }
    checkpoint = _sealed(
        {
            "schema": export.CHECKPOINT_SCHEMA,
            "image": miles.IMAGE,
            "root": str(checkpoint_root),
            "rollout_index": 0,
            "next_rollout_id": 1,
            "world_size": 8,
            "topology": {"nodes": 1, "gpus_per_node": 8},
            "model": model,
            "source": {
                "run_name": "synthetic",
                "output_root": str(checkpoint_root.parent.parent),
                "plan_sha256": "b" * 64,
                "completion_sha256": "c" * 64,
                "arguments": {"nodes": 1, "gpus_per_node": 8},
                "execution": {"image": miles.IMAGE},
                "native_driver_sha256": "d" * 64,
            },
            "files": inventory,
            "source_optimizer_update_claimed": False,
            "gpu_reload_verified": False,
            "optimizer_update_during_reload": False,
        }
    )
    checkpoint_path = tmp_path / "MILES_TRAINING_CHECKPOINT.json"
    checkpoint_file_sha256 = _write_json(checkpoint_path, checkpoint)
    terminal = _sealed(
        {
            "schema": export.TERMINAL_SCHEMA,
            "status": "accepted",
            "source_plan_sha256": "sha256:" + checkpoint["source"]["plan_sha256"],
            "checkpoint_manifest": {
                "path": str(checkpoint_path),
                "file_sha256": checkpoint_file_sha256,
                "receipt_sha256": checkpoint["sha256"],
            },
            "reward_values_included": False,
            "task_content_included": False,
            "production_promotion_requires_reload_acceptance": True,
        }
    )
    terminal_path = tmp_path / "MILES_TERMINAL_ACCEPTED.json"
    terminal_file_sha256 = _write_json(terminal_path, terminal)
    native_reload = _sealed(
        {
            "schema": export.NATIVE_RELOAD_SCHEMA,
            "status": "accepted",
            "reload_plan": {"source_manifest": checkpoint},
            "source_manifest_sha256": checkpoint["sha256"],
        }
    )
    native_reload_path = tmp_path / "RELOAD_ACCEPTED.json"
    native_reload_file_sha256 = _write_json(native_reload_path, native_reload)
    monkeypatch.setattr(
        "training.miles_reload_acceptance.validate_accepted",
        lambda value, check_files: {
            "source_manifest_sha256": value["source_manifest_sha256"],
            "source_terminal_acceptance_sha256": terminal["sha256"],
        },
    )

    def invoke(_source: Path, destination: Path, metadata: Path, _log: Path) -> None:
        shutil.copytree(raw, destination)
        for name in ("config.json", "tokenizer.json"):
            shutil.copyfile(metadata / name, destination / name)
        _log.write_text("synthetic private converter log")

    monkeypatch.setattr(export, "_invoke_converter", invoke)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    return {
        "base": base,
        "raw": raw,
        "checkpoint_root": checkpoint_root,
        "checkpoint": checkpoint,
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": checkpoint_file_sha256,
        "terminal_path": terminal_path,
        "terminal_sha256": terminal_file_sha256,
        "native_reload_path": native_reload_path,
        "native_reload_sha256": native_reload_file_sha256,
        "output": tmp_path / "hf",
    }


def _run(case: dict) -> dict:
    return export.export(
        checkpoint_path=case["checkpoint_path"],
        checkpoint_sha256=case["checkpoint_sha256"],
        terminal_path=case["terminal_path"],
        terminal_sha256=case["terminal_sha256"],
        native_reload_path=case["native_reload_path"],
        native_reload_sha256=case["native_reload_sha256"],
        output=case["output"],
    )


def test_export_is_exhaustive_bf16_create_once_and_source_bound(case: dict) -> None:
    before = {
        str(path): export._hash(path)
        for path in case["checkpoint_root"].rglob("*")
        if path.is_file()
    }
    result = _run(case)

    assert result["schema"] == export.EXPORT_SCHEMA
    assert result["source"]["checkpoint"]["receipt_sha256"] == case["checkpoint"]["sha256"]
    assert result["miles"]["source_commit"] == export.MILES_SOURCE_COMMIT
    assert result["model"]["native_parallelism"] == {"tensor": 4, "context": 2, "world_size": 8}
    assert result["dtype"] == "BF16" and result["optimizer_updates_executed"] == 0
    assert result["restored_base_tensor_count"] == 2
    assert {row["source"] for row in result["tensor_inventory"]} == {
        "trained",
        "frozen_base_auxiliary",
    }
    assert before == {
        str(path): export._hash(path)
        for path in case["checkpoint_root"].rglob("*")
        if path.is_file()
    }
    inspected, tensors = export.inspect_export(
        case["output"] / "EXPORT.json", export._hash(case["output"] / "EXPORT.json")
    )
    assert inspected == result and len(tensors) == 3
    assert not case["output"].with_name(case["output"].name + ".partial").exists()
    with pytest.raises(FileExistsError):
        _run(case)


def test_export_refuses_a_preexisting_attempt_directory(case: dict) -> None:
    case["output"].with_name(case["output"].name + ".partial").mkdir()
    with pytest.raises(FileExistsError, match="attempt"):
        _run(case)


def test_converter_invocation_is_offline_and_has_no_force_flag(tmp_path: Path, monkeypatch) -> None:
    observed = {}

    def run(argv, **kwargs):
        observed.update(argv=argv, **kwargs)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(export, "_miles_root", lambda: tmp_path)
    monkeypatch.setattr(export.subprocess, "run", run)
    source = tmp_path / "checkpoint"
    metadata = tmp_path / "metadata"
    destination = tmp_path / "output"
    log = tmp_path / "private.log"
    export._invoke_converter(source, destination, metadata, log)

    assert "-f" not in observed["argv"] and "--force" not in observed["argv"]
    assert observed["argv"][2:] == [
        "--input-dir",
        str(source),
        "--output-dir",
        str(destination),
        "--origin-hf-dir",
        str(metadata),
        "--chunk-size",
        str(export.MAX_SHARD_BYTES),
        "--vocab-size",
        str(export.VOCAB_SIZE),
    ]
    assert observed["env"]["HF_HUB_OFFLINE"] == "1"
    assert observed["env"]["TRANSFORMERS_OFFLINE"] == "1"
    assert observed["check"] is False


@pytest.mark.parametrize("defect", ["fp32", "missing_language", "extra", "acceptance"])
def test_export_rejects_drift_outside_frozen_auxiliary(case: dict, defect: str) -> None:
    raw = case["raw"]
    if defect == "fp32":
        tensors = {"model.language_model.weight": torch.ones((2, 4), dtype=torch.float32)}
    elif defect == "missing_language":
        tensors = {"different.weight": torch.ones((2, 4), dtype=torch.bfloat16)}
    elif defect == "extra":
        tensors = {
            "model.language_model.weight": torch.ones((2, 4), dtype=torch.bfloat16),
            "unexpected.weight": torch.ones((1,), dtype=torch.bfloat16),
        }
    else:
        terminal = json.loads(case["terminal_path"].read_text())
        terminal["checkpoint_manifest"]["receipt_sha256"] = "0" * 64
        terminal.pop("sha256")
        terminal = _sealed(terminal)
        case["terminal_sha256"] = _write_json(case["terminal_path"], terminal)
        with pytest.raises(ValueError, match="does not bind"):
            _run(case)
        return
    shutil.rmtree(raw)
    _hf_tree(raw, tensors)
    with pytest.raises(ValueError, match="differs outside"):
        _run(case)


def test_export_detects_source_mutation_during_conversion(case: dict, monkeypatch) -> None:
    original = export._invoke_converter

    def mutate(source: Path, destination: Path, metadata: Path, log: Path) -> None:
        original(source, destination, metadata, log)
        (case["checkpoint_root"] / "iter_0000000/__0_0.distcp").write_bytes(b"changed")

    monkeypatch.setattr(export, "_invoke_converter", mutate)
    with pytest.raises(ValueError, match="checkpoint"):
        _run(case)


def test_gpu_reload_is_separate_and_zero_work(case: dict, monkeypatch) -> None:
    result = _run(case)
    tensor_rows = {row["name"]: row for row in result["tensor_inventory"]}

    class Tensor:
        dtype = torch.bfloat16
        device = SimpleNamespace(type="cuda")

        def __init__(self, shape):
            self.shape = torch.Size(shape)

    state = {
        name: Tensor(row["shape"])
        for name, row in tensor_rows.items()
        if not name.startswith("mtp.")
    }
    model = SimpleNamespace(state_dict=lambda: state)
    config = SimpleNamespace(auto_map=None, get_text_config=lambda: SimpleNamespace(auto_map=None))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "set_device", lambda _device: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _device: 123)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _device: "synthetic")
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr("transformers.AutoConfig.from_pretrained", lambda *args, **kwargs: config)
    monkeypatch.setattr(
        "transformers.AutoModelForImageTextToText.from_pretrained",
        lambda *args, **kwargs: (model, {"unexpected_keys": ["mtp.weight"]}),
    )
    evidence_root = case["output"].parent / "reload-evidence"
    evidence_root.mkdir()
    reload_output = evidence_root / "HF_RELOAD_VALIDATED.json"
    reloaded = export.gpu_reload(
        case["output"] / "EXPORT.json",
        export._hash(case["output"] / "EXPORT.json"),
        reload_output,
        run_name="hf-reload",
    )

    assert reloaded["schema"] == export.RELOAD_SCHEMA
    assert reloaded["all_runtime_weights_loaded"] is True
    assert reloaded["external_gpu_release_verified"] is False
    assert {
        reloaded[key]
        for key in (
            "optimizer_updates",
            "rollouts",
            "verifier_calls",
            "forwards",
            "backwards",
            "checkpoint_writes",
            "wandb_events",
        )
    } == {0}

    result_file_sha256 = export._hash(reload_output)
    run_id = "11111111-1111-4111-8111-111111111111"
    rayjob_uid = "22222222-2222-4222-8222-222222222222"
    workload_uid = "33333333-3333-4333-8333-333333333333"
    raycluster_uid = "44444444-4444-4444-8444-444444444444"
    pod_uid = "55555555-5555-4555-8555-555555555555"
    namespace_uid = "66666666-6666-4666-8666-666666666666"
    job_name = f"hf-reload-{run_id[:8]}"
    image_digest = miles.IMAGE.rsplit("@sha256:", 1)[-1]
    controller = _sealed(
        {
            "schema": export.RELOAD_CONTROLLER_SCHEMA,
            "status": "succeeded",
            "cluster": "dev",
            "api_base_url": export.API_URLS["dev"],
            "kube_context": "synthetic-dev",
            "namespace": export.RELOAD_NAMESPACE,
            "namespace_uid": namespace_uid,
            "run_name": "hf-reload",
            "reload_result_path": str(reload_output),
            "reload_result_file_sha256": result_file_sha256,
            "reload_result_sha256": reloaded["sha256"],
            "api_run_id": run_id,
            "api_run_name": job_name,
            "rayjob_name": job_name,
            "rayjob_uid": rayjob_uid,
            "workload_name": "hf-reload-workload",
            "workload_uid": workload_uid,
            "workload_owner_rayjob_uid": rayjob_uid,
            "raycluster_name": "hf-reload-cluster",
            "raycluster_uid": raycluster_uid,
            "raycluster_owner_rayjob_uid": rayjob_uid,
            "pods": [
                {
                    "name": "hf-reload-worker",
                    "uid": pod_uid,
                    "owner_raycluster_uid": raycluster_uid,
                    "phase": "Succeeded",
                    "exit_code": 0,
                    "termination_reason": "Completed",
                    "terminated_at": reloaded["completed_at"] + 1,
                    "runtime_image_id": "containerd://registry/image@sha256:" + image_digest,
                    "container_restarts": 0,
                    "gpus": 1,
                }
            ],
            "api_status": "SUCCEEDED",
            "controller_status": "SUCCEEDED",
            "effective_priority": 10000,
            "automatic_requeue": False,
            "workers": 1,
            "gpus_per_worker": 1,
            "total_gpus": 1,
            "observed_at": reloaded["completed_at"] + 2,
        }
    )
    controller_path = evidence_root / "HF_RELOAD_CONTROLLER_TERMINAL.json"
    controller_file_sha256 = _write_json(controller_path, controller)
    release = _sealed(
        {
            "schema": export.RELOAD_RELEASE_SCHEMA,
            "status": "released",
            "cluster": "dev",
            "api_base_url": export.API_URLS["dev"],
            "kube_context": controller["kube_context"],
            "namespace": export.RELOAD_NAMESPACE,
            "namespace_uid": namespace_uid,
            "run_name": "hf-reload",
            "reload_result_path": str(reload_output),
            "reload_result_file_sha256": result_file_sha256,
            "reload_result_sha256": reloaded["sha256"],
            "controller_terminal_path": str(controller_path),
            "controller_terminal_file_sha256": controller_file_sha256,
            "controller_terminal_sha256": controller["sha256"],
            "api_run_id": run_id,
            "api_run_name": job_name,
            "rayjob_name": job_name,
            "rayjob_uid": rayjob_uid,
            "workload_name": controller["workload_name"],
            "workload_uid": workload_uid,
            "raycluster_name": controller["raycluster_name"],
            "raycluster_uid": raycluster_uid,
            "pod_uids": [pod_uid],
            "api_status": "SUCCEEDED",
            "controller_status": "SUCCEEDED",
            "rayjob_present": False,
            "workload_present": False,
            "quota_reservation_present": False,
            "raycluster_present": False,
            "gpu_pods_present": False,
            "active_gpu_pod_uids": [],
            "active_gpus": 0,
            "observed_at": reloaded["completed_at"] + 3,
        }
    )
    release_path = evidence_root / "HF_RELOAD_RELEASE.json"
    _write_json(release_path, release)
    accepted_path = evidence_root / "HF_RELOAD_ACCEPTED.json"
    accepted = export.accept_gpu_reload(
        result_path=reload_output,
        controller_path=controller_path,
        release_path=release_path,
        output=accepted_path,
    )
    assert accepted["schema"] == export.RELOAD_ACCEPTED_SCHEMA
    assert accepted["external_gpu_release_verified"] is True
    assert accepted["serving_qualified"] is False
    assert export.validate_reload_accepted(accepted)["export_receipt_sha256"] == result["sha256"]

    release_path.unlink()
    release.pop("sha256")
    release["active_gpus"] = 1
    _write_json(release_path, _sealed(release))
    accepted_path.unlink()
    with pytest.raises(ValueError, match="external release"):
        export.accept_gpu_reload(
            result_path=reload_output,
            controller_path=controller_path,
            release_path=release_path,
            output=accepted_path,
        )
