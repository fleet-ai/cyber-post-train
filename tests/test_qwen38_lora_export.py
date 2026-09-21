from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from cyber_post_train.jobs import digest, validate_request
from training import qwen38_lora_export as export
from training import qwen38_lora_export_control as control


def checkpoint_identity() -> dict:
    return {
        "receipt_sha256": "1" * 64,
        "source_plan_sha256": "2" * 64,
        "checkpoint_path": "/mnt/sfs/jobs/source/checkpoints/global_step_1",
        "optimizer_step": 1,
        "model_repository": "Qwen/Qwen3.8-27B",
        "base_model_revision": "3" * 40,
        "checkpoint_inventory_sha256": "4" * 64,
        "target_census_sha256": "5" * 64,
        "adapter_parameter_inventory_sha256": "6" * 64,
        "base_model_inventory_sha256": "7" * 64,
    }


def checkpoint_receipt() -> dict:
    return {
        "source_plan": {"output_root": "/mnt/sfs/jobs/source"},
        "checkpoint_path": "/mnt/sfs/jobs/source/checkpoints/global_step_1",
    }


def continuation_identity() -> dict:
    return {
        "receipt_sha256": "1" * 64,
        "source_plan_sha256": "2" * 64,
        "checkpoint_path": "/mnt/sfs/jobs/source/checkpoints/global_step_60",
        "optimizer_step": 60,
        "model_repository": "Qwen/Qwen3.8-27B",
        "base_model_revision": "3" * 40,
        "checkpoint_inventory_sha256": "4" * 64,
        "base_model_inventory_sha256": "7" * 64,
    }


def continuation_manifest() -> dict:
    return {
        "source_plan": {
            "schema": "cyber_sft_runtime_dense_v1",
            "run_name": "source",
            "output_root": "/mnt/sfs/jobs/source",
            "recipe": {"lr": 1e-5, "max_steps": 100},
            "datasets": {"train": {"sha256": "8" * 64}},
            "recovery": {"mode": "resume", "checkpoint": {"optimizer_step": 42}},
            "recovery_runtime_sha256": "9" * 64,
            "pause_after_step": 60,
            "wandb": {
                "entity": "thefleet",
                "project": "cyber-post-train",
                "group": "qwen38-lora",
                "run_id": "source",
                "name": "source",
                "tags": ["qwen38", "lora"],
            },
        },
        "checkpoint_path": "/mnt/sfs/jobs/source/checkpoints/global_step_60",
    }


def write_checkpoint(path: Path) -> str:
    path.write_text(json.dumps(checkpoint_receipt(), sort_keys=True) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_seal_plan_consumes_only_an_accepted_checkpoint_receipt(
    tmp_path: Path, monkeypatch
) -> None:
    receipt = tmp_path / export.CHECKPOINT_FILENAME
    file_sha256 = write_checkpoint(receipt)
    monkeypatch.setattr(export, "validate_checkpoint_receipt", lambda value: checkpoint_identity())

    plan = export.seal_plan(
        receipt,
        checkpoint_file_sha256=file_sha256,
        run_name="chris-q38-lora-export-v1",
        run_dir="/mnt/sfs/jobs/chris-q38-lora-export-v1",
    )

    assert plan["checkpoint_receipt"] == {
        "path": str(receipt),
        "file_sha256": file_sha256,
        "receipt_sha256": "1" * 64,
    }
    assert plan["checkpoint_identity"]["checkpoint_path"].endswith("global_step_1")
    assert plan["optimizer_steps_executed"] == 0
    assert plan["external_evaluation"] is False
    assert plan["create_once"] is True
    assert plan["priority_class"] == "c1"
    assert plan["output_root"] == plan["run_dir"] + "/merged-hf"


def test_seal_continuation_plan_consumes_only_an_exact_native_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    manifest = tmp_path / "step-000060-megatron-v1.json"
    manifest.write_text(json.dumps(continuation_manifest(), sort_keys=True) + "\n")
    file_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
    monkeypatch.setattr(
        export,
        "validate_continuation_source_manifest",
        lambda value: continuation_identity(),
    )
    monkeypatch.setattr("training.recovery.validate", lambda value, check_files: None)

    plan = export.seal_continuation_plan(
        manifest,
        checkpoint_file_sha256=file_sha256,
        checkpoint_runtime_path=(
            "/mnt/sfs/jobs/source/checkpoint_manifests/step-000060-megatron-v1.json"
        ),
        run_name="chris-q38-lora-s60-export-v1",
        run_dir="/mnt/sfs/jobs/chris-q38-lora-s60-export-v1",
    )

    assert plan["schema"] == export.CONTINUATION_PLAN_SCHEMA
    assert plan["checkpoint_manifest"]["file_sha256"] == file_sha256
    assert plan["checkpoint_identity"]["optimizer_step"] == 60
    assert plan["optimizer_steps_executed"] == 0
    assert plan["create_once"] is True
    request = export.job_request(plan)
    assert request["failureAlerts"] is False
    assert request["priority_class"] == "c1"


def test_committed_step60_plan_binds_current_producer_and_exact_source() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "configs/qualification/qwen38-lora-step60-zero-update-export-v1.json"
    )
    plan = json.loads(path.read_text())

    assert export.validate_plan(plan) == plan
    assert digest(plan) == "31c9548c7119e01e3941049bcdaa79f9ee7a787a0191ce02c53644959d2d4547"
    assert plan["checkpoint_manifest"] == {
        "path": (
            "/mnt/sfs/jobs/chris-q38-lora-r1-s60-v3/"
            "checkpoint_manifests/step-000060-megatron-v1.json"
        ),
        "file_sha256": "cd53865f869eeb1975aa0e099aef143a736167c5c7b095c283f0da292ffecd78",
        "receipt_sha256": "6bae9ef75e0a60f598eb32b63d31a64c0f42716611a43daf19a177d3c0ffebd2",
    }
    request = export.job_request(plan)
    assert request["failureAlerts"] is False
    assert request["priority_class"] == "c1"
    assert request["gpus_per_worker"] == 8


def test_control_seals_continuation_without_treating_it_as_step_one(
    tmp_path: Path, monkeypatch
) -> None:
    manifest = tmp_path / "step-000060-megatron-v1.json"
    manifest.write_text(json.dumps(continuation_manifest(), sort_keys=True) + "\n")
    file_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
    monkeypatch.setattr(
        export,
        "validate_continuation_source_manifest",
        lambda value: continuation_identity(),
    )
    monkeypatch.setattr("training.recovery.validate", lambda value, check_files: None)
    monkeypatch.setattr(control, "_runtime_source_inventory", lambda: {})
    args = SimpleNamespace(
        checkpoint_receipt=None,
        checkpoint_manifest=str(manifest),
        checkpoint_file_sha256=file_sha256,
        checkpoint_runtime_path=(
            "/mnt/sfs/jobs/source/checkpoint_manifests/step-000060-megatron-v1.json"
        ),
        run_name="chris-q38-lora-s60-export-v1",
        run_dir="/mnt/sfs/jobs/chris-q38-lora-s60-export-v1",
        control_dir=str(tmp_path / "control"),
    )

    result = control.seal(args)
    plan = json.loads((tmp_path / "control" / "plan.json").read_text())

    assert result["checkpoint_reference_kind"] == "native_continuation_manifest"
    assert result["checkpoint_receipt_sha256"] == "1" * 64
    assert plan["schema"] == export.CONTINUATION_PLAN_SCHEMA
    assert plan["checkpoint_manifest"]["path"] == args.checkpoint_runtime_path
    assert plan["optimizer_steps_executed"] == 0


def test_continuation_runtime_derivation_preserves_every_scientific_field(
    tmp_path: Path, monkeypatch
) -> None:
    manifest_path = tmp_path / "step-000060-megatron-v1.json"
    manifest = continuation_manifest()
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
    file_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    monkeypatch.setattr(
        export,
        "validate_continuation_source_manifest",
        lambda value: continuation_identity(),
    )
    monkeypatch.setattr("training.recovery.validate", lambda value, check_files: None)
    plan = export.seal_continuation_plan(
        manifest_path,
        checkpoint_file_sha256=file_sha256,
        checkpoint_runtime_path=(
            "/mnt/sfs/jobs/source/checkpoint_manifests/step-000060-megatron-v1.json"
        ),
        run_name="chris-q38-lora-s60-export-v1",
        run_dir="/mnt/sfs/jobs/chris-q38-lora-s60-export-v1",
    )
    observed = {}
    monkeypatch.setattr(
        "training.recovery.validate",
        lambda value, check_files: observed.update(
            {"plan": copy.deepcopy(value), "check_files": check_files}
        ),
    )

    runtime = export._continuation_runtime_plan(manifest, plan, check_files=True)

    assert observed == {"plan": runtime, "check_files": True}
    assert runtime["recipe"] == manifest["source_plan"]["recipe"]
    assert runtime["datasets"] == manifest["source_plan"]["datasets"]
    assert runtime["recovery"]["mode"] == "validate"
    assert runtime["recovery"]["checkpoint"] == manifest
    assert "pause_after_step" not in runtime

    broken = copy.deepcopy(runtime)
    broken["recipe"]["lr"] = 5e-5
    with pytest.raises(ValueError, match="scientific plan"):
        export._validate_continuation_runtime_derivation(
            manifest["source_plan"], broken, manifest, plan
        )


def test_continuation_reload_reconciles_exact_tp8_checkpoint_tensors(
    tmp_path: Path, monkeypatch
) -> None:
    torch = pytest.importorskip("torch")
    name = "model.layers.0.linear_qkv.adapter.linear_in.weight"
    tensor = torch.ones((2, 2), dtype=torch.bfloat16)
    saved_sha = export._tensor_sha256(tensor)

    def row(rank: int, *, after: bool) -> dict:
        return {
            "target_census": {"selector": "all-linear"},
            "adapters": {
                name: {
                    "logical_target": "linear_qkv",
                    "dtype": "BF16",
                    "global_shape": [2, 16],
                    "local_shape": [2, 2],
                    "sharding": {
                        "tensor_parallel": True,
                        "partition_dim": 1,
                        "partition_stride": 1,
                    },
                    "sha256": saved_sha if after else f"{rank + 1:064x}",
                }
            },
            "trainable": {
                "parameter_count": 1,
                "elements": 4,
                "manifest_sha256": f"{100 + rank:064x}",
            },
            "frozen_base": {
                "parameter_count": 10,
                "elements": 20,
                "bytes": 40,
                "manifest_sha256": f"{200 + rank:064x}",
            },
            "successful_optimizer_updates": 0,
            "last_gradient_norm": None,
        }

    before = {rank: row(rank, after=False) for rank in range(8)}
    after = {rank: row(rank, after=True) for rank in range(8)}
    monkeypatch.setattr(
        "training.sft_runtime._qwen38_rank_snapshots",
        lambda snapshots, stage: snapshots,
    )
    monkeypatch.setattr(
        torch,
        "load",
        lambda path, map_location, weights_only: {"model_state_dict": {name: tensor}},
    )
    ranks = [
        {
            "rank": rank,
            "optimizer_step": 60,
            "optimizer_states": 2,
            "scheduler_restored": True,
            "backend": "megatron",
        }
        for rank in range(8)
    ]
    evidence = export.verify_continuation_reloaded_snapshots(
        {
            "checkpoint_path": str(tmp_path / "checkpoint"),
            "optimizer_step": 60,
        },
        before,
        after,
        ranks,
        [1e-5] * 8,
    )

    assert evidence["optimizer_steps_executed"] == 0
    assert evidence["strict_census"] is True
    assert evidence["optimizer_reload"]["ranks"] == ranks
    assert evidence["adapter_parameters"][name]["rank_shards"][0]["checkpoint_sha256"] == saved_sha

    after[0]["frozen_base"]["manifest_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="frozen base"):
        export.verify_continuation_reloaded_snapshots(
            {"checkpoint_path": str(tmp_path / "checkpoint"), "optimizer_step": 60},
            before,
            after,
            ranks,
            [1e-5] * 8,
        )

    with pytest.raises(ValueError, match="optimizer/scheduler reload"):
        export.verify_continuation_reloaded_snapshots(
            {"checkpoint_path": str(tmp_path / "checkpoint"), "optimizer_step": 60},
            before,
            {rank: row(rank, after=True) for rank in range(8)},
            ranks,
            [1e-5] * 7 + [2e-5],
        )


def test_seal_plan_rejects_noncanonical_input_or_overlapping_output(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(export, "validate_checkpoint_receipt", lambda value: checkpoint_identity())
    wrong = tmp_path / "checkpoint.json"
    wrong_sha = write_checkpoint(wrong)
    with pytest.raises(ValueError, match="exact QWEN38_LORA_CHECKPOINT"):
        export.seal_plan(
            wrong,
            checkpoint_file_sha256=wrong_sha,
            run_name="chris-q38-lora-export-v1",
            run_dir="/mnt/sfs/jobs/chris-q38-lora-export-v1",
        )


def test_seal_plan_can_bind_separate_exact_sfs_runtime_receipt(tmp_path: Path, monkeypatch) -> None:
    receipt = tmp_path / export.CHECKPOINT_FILENAME
    file_sha256 = write_checkpoint(receipt)
    monkeypatch.setattr(export, "validate_checkpoint_receipt", lambda value: checkpoint_identity())

    plan = export.seal_plan(
        receipt,
        checkpoint_file_sha256=file_sha256,
        checkpoint_runtime_path=(
            "/mnt/sfs/jobs/chris-q38-lora-prod-can-v1/QWEN38_LORA_CHECKPOINT.json"
        ),
        run_name="chris-q38-lora-prod-exp-v1",
        run_dir="/mnt/sfs/jobs/chris-q38-lora-prod-exp-v1",
    )

    assert plan["checkpoint_receipt"]["path"] == (
        "/mnt/sfs/jobs/chris-q38-lora-prod-can-v1/QWEN38_LORA_CHECKPOINT.json"
    )

    with pytest.raises(ValueError, match="direct SFS job receipt"):
        export.seal_plan(
            receipt,
            checkpoint_file_sha256=file_sha256,
            checkpoint_runtime_path="/tmp/QWEN38_LORA_CHECKPOINT.json",
            run_name="chris-q38-lora-prod-exp-v2",
            run_dir="/mnt/sfs/jobs/chris-q38-lora-prod-exp-v2",
        )

    exact = tmp_path / export.CHECKPOINT_FILENAME
    exact_sha = write_checkpoint(exact)
    with pytest.raises(ValueError, match="must not overlap"):
        export.seal_plan(
            exact,
            checkpoint_file_sha256=exact_sha,
            run_name="source",
            run_dir="/mnt/sfs/jobs/source",
        )


def test_job_request_is_exact_image_create_once_c1_and_score_free(
    tmp_path: Path, monkeypatch
) -> None:
    receipt = tmp_path / export.CHECKPOINT_FILENAME
    file_sha256 = write_checkpoint(receipt)
    monkeypatch.setattr(export, "validate_checkpoint_receipt", lambda value: checkpoint_identity())
    plan = export.seal_plan(
        receipt,
        checkpoint_file_sha256=file_sha256,
        run_name="chris-q38-lora-export-v1",
        run_dir="/mnt/sfs/jobs/chris-q38-lora-export-v1",
    )

    request = export.job_request(plan)
    assert request["failureAlerts"] is False
    validate_request(request)
    assert request["image"] == export.QWEN38_MEGATRON_IMAGE
    assert request["workers"] == 1 and request["gpus_per_worker"] == 8
    assert request["priority_class"] == "c1"
    assert request["requeueIfPreempted"] is False
    assert request["secrets"] == []
    assert request["image_pull_secrets"] == ["ghcr-pull"]
    assert request["run_dir"] == plan["run_dir"]
    assert request["env"]["HF_HUB_OFFLINE"] == "1"
    assert request["env"]["WANDB_MODE"] == "disabled"
    assert "FLEET" not in json.dumps(request)
    assert any(name.startswith("CYBER_RUNTIME_BUNDLE") for name in request["env"])
    assert request == export.job_request(plan)


def synthetic_checkpoint_and_snapshots() -> tuple[dict, list[dict]]:
    adapters = {
        "layer.linear_qkv.adapter.linear_in.weight": {
            "rank_shards": [
                {"tp_rank": rank, "checkpoint_sha256": f"{100 + rank:064x}"} for rank in range(8)
            ]
        },
        "layer.linear_fc1.adapter.linear_out.weight": {
            "rank_shards": [
                {"tp_rank": rank, "checkpoint_sha256": f"{200 + rank:064x}"} for rank in range(8)
            ]
        },
    }
    trainable = [{"tp_rank": rank, "after_sha256": f"{300 + rank:064x}"} for rank in range(8)]
    frozen = [{"tp_rank": rank, "after_sha256": f"{400 + rank:064x}"} for rank in range(8)]
    receipt = {
        "adapter_parameters": adapters,
        "trainable_parameter_census": {"rank_manifests": trainable},
        "frozen_base": {"rank_manifests": frozen},
    }
    snapshots = []
    for rank in range(8):
        snapshots.append(
            {
                "rank": {"tp_rank": rank},
                "adapters": {
                    name: {"sha256": row["rank_shards"][rank]["checkpoint_sha256"]}
                    for name, row in adapters.items()
                },
                "trainable": {"manifest_sha256": trainable[rank]["after_sha256"]},
                "frozen_base": {"manifest_sha256": frozen[rank]["after_sha256"]},
                "successful_optimizer_updates": 0,
                "last_gradient_norm": None,
            }
        )
    return receipt, snapshots


def test_reload_reconciles_every_adapter_value_on_every_tp8_rank() -> None:
    receipt, snapshots = synthetic_checkpoint_and_snapshots()
    assert export.verify_reloaded_snapshots(receipt, snapshots) == {
        "tp_ranks": list(range(8)),
        "optimizer_steps_executed": 0,
        "strict_census": True,
    }


@pytest.mark.parametrize("fault", ["missing_rank", "adapter", "base", "optimizer"])
def test_reload_fails_closed_on_non_strict_native_load_gap(fault: str) -> None:
    receipt, snapshots = synthetic_checkpoint_and_snapshots()
    if fault == "missing_rank":
        snapshots.pop()
    elif fault == "adapter":
        first = next(iter(snapshots[3]["adapters"]))
        snapshots[3]["adapters"][first]["sha256"] = "f" * 64
    elif fault == "base":
        snapshots[4]["frozen_base"]["manifest_sha256"] = "f" * 64
    else:
        snapshots[5]["successful_optimizer_updates"] = 1
    with pytest.raises(ValueError):
        export.verify_reloaded_snapshots(receipt, snapshots)


def test_plan_and_runtime_expose_no_evaluation_or_optimizer_path() -> None:
    source = Path(export.__file__).read_text()
    assert 'external_evaluation": False' in source
    assert ".optim_step(" not in source
    assert ".train()" not in source
    assert "trainer.dispatch.load_checkpoint" in source
    assert source.count("trainer.dispatch.save_hf_model") == 2
    assert "AutoModelForCausalLM.from_pretrained" in source
    assert "AutoTokenizer.from_pretrained" in source
    assert "ray.remote(num_cpus=4, num_gpus=1)" not in source
    assert "_start_reload_model_subprocess" in source


def test_full_reload_uses_an_isolated_visible_gpu_without_a_ray_lease(
    tmp_path: Path, monkeypatch
) -> None:
    captured = {}

    class Process:
        def wait(self, timeout):
            captured["timeout"] = timeout
            return 0

        def poll(self):
            return 0

    def popen(args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return Process()

    monkeypatch.setattr(export.subprocess, "Popen", popen)
    process, result_path = export._start_reload_model_subprocess(
        tmp_path / "merged-hf",
        tmp_path,
    )
    assert captured["args"][:2] == [export.sys.executable, "-c"]
    assert captured["kwargs"]["env"]["CUDA_VISIBLE_DEVICES"] == "0"
    assert str(Path(export.__file__).resolve().parents[1]) in captured["kwargs"]["env"][
        "PYTHONPATH"
    ].split(export.os.pathsep)
    assert (tmp_path / "private_logs" / "merged-model-reload.log").is_file()

    result_path.write_text(
        json.dumps(
            {
                "model_class": "QwenModel",
                "tokenizer_class": "QwenTokenizer",
                "logit_values": 42,
                "finite_logits": True,
            }
        )
    )
    assert export._finish_reload_model_subprocess(process, result_path) == {
        "model_class": "QwenModel",
        "tokenizer_class": "QwenTokenizer",
        "logit_values": 42,
        "finite_logits": True,
    }
    assert captured["timeout"] == 900
    assert not result_path.exists()


def test_plan_validation_rejects_any_mutated_scientific_boundary(
    tmp_path: Path, monkeypatch
) -> None:
    receipt = tmp_path / export.CHECKPOINT_FILENAME
    file_sha256 = write_checkpoint(receipt)
    monkeypatch.setattr(export, "validate_checkpoint_receipt", lambda value: checkpoint_identity())
    plan = export.seal_plan(
        receipt,
        checkpoint_file_sha256=file_sha256,
        run_name="chris-q38-lora-export-v1",
        run_dir="/mnt/sfs/jobs/chris-q38-lora-export-v1",
    )
    for field, value in (
        ("image", "registry/other@sha256:" + "a" * 64),
        ("priority_class", "c2"),
        ("optimizer_steps_executed", 1),
        ("external_evaluation", True),
        ("create_once", False),
    ):
        broken = copy.deepcopy(plan)
        broken[field] = value
        with pytest.raises(ValueError, match="reviewed exact-image lane"):
            export.validate_plan(broken)


def test_hf_inspection_reopens_every_indexed_bf16_tensor(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    save_file = pytest.importorskip("safetensors.torch").save_file
    shard = "model-00001-of-00001.safetensors"
    save_file(
        {
            "model.embed_tokens.weight": torch.arange(12, dtype=torch.bfloat16).reshape(3, 4),
            "model.norm.weight": torch.ones(4, dtype=torch.bfloat16),
        },
        tmp_path / shard,
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": (12 + 4) * 2},
                "weight_map": {
                    "model.embed_tokens.weight": shard,
                    "model.norm.weight": shard,
                },
            }
        )
    )
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        (tmp_path / name).write_text("{}\n")
    (tmp_path / "chat_template.jinja").write_text("{{ messages }}\n")

    first = export.inspect_hf_export(tmp_path)
    second = export.inspect_hf_export(tmp_path)
    assert first == second
    assert first["tensor_count"] == 2
    assert first["tensor_values"] == 16
    assert first["tensor_bytes"] == 32
    assert set(first["tensors"]) == {
        "model.embed_tokens.weight",
        "model.norm.weight",
    }
    accepted_surface = {path.name for path in tmp_path.iterdir() if path.is_file()}
    (tmp_path / ".cache").mkdir()
    (tmp_path / ".cache" / "unrelated-metadata").write_text("ignored\n")
    assert export.inspect_hf_export(tmp_path, allowed_files=accepted_surface) == first
    with pytest.raises(ValueError, match="flat"):
        export.inspect_hf_export(tmp_path)


def test_completion_restores_only_frozen_visual_and_mtp_base_layout(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    save_file = pytest.importorskip("safetensors.torch").save_file
    base = tmp_path / "base"
    partial = tmp_path / "partial"
    completed = tmp_path / "completed"
    repeated = tmp_path / "repeated"
    base.mkdir()
    partial.mkdir()
    shard = "model-00001-of-00001.safetensors"
    base_tensors = {
        "model.language.weight": torch.zeros((2, 2), dtype=torch.bfloat16),
        "model.visual.proj.weight": torch.ones((2, 2), dtype=torch.bfloat16),
        "mtp.norm.weight": torch.full((2,), 2, dtype=torch.bfloat16),
    }
    save_file(base_tensors, base / shard)
    save_file(
        {"model.language.weight": torch.full((2, 2), 3, dtype=torch.bfloat16)},
        partial / shard,
    )

    def write_surface(root: Path, names: list[str]) -> None:
        (root / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": 20}, "weight_map": {n: shard for n in names}})
        )
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
            (root / name).write_text("{}\n")
        (root / "chat_template.jinja").write_text("{{ messages }}\n")

    write_surface(base, list(base_tensors))
    write_surface(partial, ["model.language.weight"])
    base_files = {path.name for path in base.iterdir() if path.is_file()}
    result = export.complete_native_export_from_base(
        partial,
        base,
        completed,
        base_files=base_files,
    )
    repeated_result = export.complete_native_export_from_base(
        partial,
        base,
        repeated,
        base_files=base_files,
    )
    assert (
        result
        == repeated_result
        == {
            "base_tensor_count": 3,
            "native_tensor_count": 1,
            "restored_frozen_tensor_count": 2,
            "restored_prefixes": ["model", "mtp"],
        }
    )
    base_inspection = export.inspect_hf_export(base)
    completed_inspection = export.inspect_hf_export(completed)
    assert completed_inspection == export.inspect_hf_export(repeated)
    assert completed_inspection["layout_sha256"] == base_inspection["layout_sha256"]
    assert (
        completed_inspection["tensors"]["model.language.weight"]["sha256"]
        != base_inspection["tensors"]["model.language.weight"]["sha256"]
    )
    for name in ("model.visual.proj.weight", "mtp.norm.weight"):
        assert (
            completed_inspection["tensors"][name]["sha256"]
            == base_inspection["tensors"][name]["sha256"]
        )


def test_completion_rejects_an_omitted_language_tensor(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    save_file = pytest.importorskip("safetensors.torch").save_file
    base = tmp_path / "base"
    partial = tmp_path / "partial"
    base.mkdir()
    partial.mkdir()
    shard = "model-00001-of-00001.safetensors"
    save_file(
        {
            "model.language.weight": torch.zeros((2, 2), dtype=torch.bfloat16),
            "model.language.bias": torch.zeros(2, dtype=torch.bfloat16),
        },
        base / shard,
    )
    save_file({"model.language.weight": torch.ones((2, 2), dtype=torch.bfloat16)}, partial / shard)
    for root, names in (
        (base, ["model.language.weight", "model.language.bias"]),
        (partial, ["model.language.weight"]),
    ):
        (root / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {name: shard for name in names}})
        )
    with pytest.raises(ValueError, match="non-frozen language tensor"):
        export.complete_native_export_from_base(
            partial,
            base,
            tmp_path / "completed",
            base_files={path.name for path in base.iterdir()},
        )
