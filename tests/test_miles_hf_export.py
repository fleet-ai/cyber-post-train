"""Synthetic, value-free checks for the create-once Miles HF handoff."""

from __future__ import annotations

import importlib.machinery
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from cyber_post_train.jobs import digest
from training import miles, miles_promotion
from training import miles_hf_export as export
from training import miles_hf_export_job as export_job


def _sealed(value: dict) -> dict:
    return {**value, "sha256": digest(value)}


def _prediction_probe(prediction: str = "a" * 64) -> dict:
    return {
        "schema": export.PREDICTION_SCHEMA,
        "probe_id": export.PREDICTION_PROBE_ID,
        "input_ids_sha256": "sha256:" + digest(list(export.PREDICTION_INPUT_IDS)),
        "sequence_length": len(export.PREDICTION_INPUT_IDS),
        "top_k": export.PREDICTION_TOP_K,
        "selection_margin_threshold": export.PREDICTION_MARGIN,
        "selection_margin_satisfied": True,
        "prediction_sha256": "sha256:" + prediction,
        "logits_included": False,
        "task_content_included": False,
        "benchmark_content_included": False,
    }


def _active_binding(source_plan_sha256: str) -> dict:
    value = {
        "schema": miles_promotion.ACTIVE_CANARY_SCHEMA,
        "status": "accepted_dev_canary",
        "reward_terminal_receipt_sha256": "sha256:" + "e" * 64,
        "source_run_name": "synthetic-canary",
        "source_commit": "f" * 40,
        "source_plan_sha256": "sha256:" + source_plan_sha256,
        "source_request_sha256": "sha256:" + "1" * 64,
        "runtime_bundle_sha256": "sha256:" + "2" * 64,
        "api_base_url": miles_promotion.DEV_API_BASE_URL,
        "api_run_id": "11111111-1111-4111-8111-111111111111",
        "api_run_name": "synthetic-canary-11111111",
        "rayjob_uid": "22222222-2222-4222-8222-222222222222",
        "workload_uid": "33333333-3333-4333-8333-333333333333",
    }
    return _sealed(value)


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
    active_canary_binding = _active_binding(checkpoint["source"]["plan_sha256"])
    active_canary_binding_path = tmp_path / "ACTIVE_CANARY_BINDING.json"
    active_canary_binding_file_sha256 = _write_json(
        active_canary_binding_path, active_canary_binding
    )
    native_reload = _sealed(
        {
            "schema": export.NATIVE_RELOAD_SCHEMA,
            "status": "accepted",
            "reload_plan": {"source_manifest": checkpoint},
            "source_manifest_sha256": checkpoint["sha256"],
            "prediction_probe": _prediction_probe(),
        }
    )
    native_reload_path = tmp_path / "RELOAD_ACCEPTED.json"
    native_reload_file_sha256 = _write_json(native_reload_path, native_reload)
    monkeypatch.setattr(
        "training.miles_reload_acceptance.validate_accepted",
        lambda value, check_files: {
            "source_manifest_sha256": value["source_manifest_sha256"],
            "source_terminal_acceptance_sha256": terminal["sha256"],
            "prediction_probe": value["prediction_probe"],
        },
    )
    monkeypatch.setattr(
        "training.miles_acceptance.validate_terminal", lambda _value, *, check_files: {}
    )
    monkeypatch.setattr(
        "training.miles_promotion._exact_active_canary", lambda _terminal, _binding: None
    )

    def invoke(_source: Path, destination: Path, metadata: Path, _log: Path) -> None:
        shutil.copytree(raw, destination)
        for name in ("config.json", "tokenizer.json"):
            shutil.copyfile(metadata / name, destination / name)
        _log.write_text("synthetic private converter log")

    monkeypatch.setattr(export, "_invoke_converter", invoke)
    monkeypatch.setattr(
        export,
        "_source_value_inventory",
        lambda _generation: [
            {
                "name": row["name"],
                "shape": row["shape"],
                "dtype": row["dtype"],
                "value_sha256": row["value_sha256"],
            }
            for row in export.tensor_inventory(raw)
        ],
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    return {
        "base": base,
        "raw": raw,
        "checkpoint_root": checkpoint_root,
        "checkpoint": checkpoint,
        "active_canary_binding_path": active_canary_binding_path,
        "active_canary_binding_sha256": active_canary_binding_file_sha256,
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
        active_canary_binding_path=case["active_canary_binding_path"],
        active_canary_binding_sha256=case["active_canary_binding_sha256"],
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
    assert result["all_tensor_values_finite"] is True
    assert result["source_equivalence"]["all_trained_values_match_source"] is True
    assert result["resource_guard"]["converter_passes"] == 1
    assert result["source_equivalence"]["converter_input"].endswith("iter_0000000")
    assert result["source_equivalence"]["common_pt_sha256"] == export._hash(
        case["checkpoint_root"] / "iter_0000000/common.pt"
    )
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


@pytest.mark.parametrize("defect", ["extra_field", "sidecar_claim", "source_digest"])
def test_export_inspector_rejects_self_digested_receipt_false_accepts(
    case: dict, defect: str
) -> None:
    _run(case)
    path = case["output"] / "EXPORT.json"
    receipt = json.loads(path.read_text())
    receipt.pop("sha256")
    if defect == "extra_field":
        receipt["unreviewed_claim"] = True
    elif defect == "sidecar_claim":
        receipt["sidecars"].pop("tokenizer.json")
    else:
        receipt["source_equivalence"]["source_value_inventory_sha256"] = "0" * 64
    path.write_text(json.dumps(_sealed(receipt), sort_keys=True))

    with pytest.raises(ValueError, match="receipt|inventory"):
        export.inspect_export(path, export._hash(path))


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
    assert observed["env"]["WANDB_MODE"] == "disabled"
    assert observed["env"]["PYTHONPATH"] == str(tmp_path)
    assert observed["env"]["PYTHONNOUSERSITE"] == "1"
    assert "FLEET_API_KEY" not in observed["env"]
    assert observed["check"] is False


def test_converter_import_closure_is_exact_and_rejects_a_new_eager_module(
    tmp_path: Path, monkeypatch
) -> None:
    assert len(export.MILES_CONVERTER_SOURCES) == 47
    assert {
        "miles/backends/megatron_utils/__init__.py",
        "miles/backends/megatron_utils/sglang.py",
        "miles/backends/megatron_utils/update_weight/common.py",
        "miles/utils/fp8_kernel.py",
        "miles/utils/mxfp8.py",
        "miles/utils/nvfp4.py",
        "miles_plugins/megatron_bridge/__init__.py",
        "miles_plugins/models/qwen3_vl.py",
    }.issubset(export.MILES_CONVERTER_SOURCES)
    root = tmp_path / "miles-root"
    origin = root / "miles/__init__.py"
    origin.parent.mkdir(parents=True)
    origin.write_text("")
    eager = root / "miles/backends/megatron_utils/megatron_to_hf"
    for name in export.MILES_CONVERTER_SOURCES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    monkeypatch.setattr(
        export.importlib.util,
        "find_spec",
        lambda _name: importlib.machinery.ModuleSpec("miles", loader=None, origin=str(origin)),
    )
    monkeypatch.setattr(
        export,
        "_hash",
        lambda path: export.MILES_CONVERTER_SOURCES[str(path.relative_to(root))],
    )
    assert export._miles_root() == root

    (eager / "unreviewed.py").write_text("")
    with pytest.raises(ValueError, match="import closure"):
        export._miles_root()


def test_export_rejects_terminal_outside_exact_active_canary(case: dict, monkeypatch) -> None:
    def reject(_terminal: dict, _binding: dict) -> None:
        raise ValueError("not exact active canary")

    monkeypatch.setattr("training.miles_promotion._exact_active_canary", reject)
    with pytest.raises(ValueError, match="exact active canary"):
        _run(case)


def test_export_fails_closed_when_active_canary_digest_is_null(case: dict) -> None:
    case["active_canary_binding_sha256"] = None
    with pytest.raises(ValueError, match="digest is absent"):
        _run(case)


def test_observer_backed_native_reload_schema_and_prepared_runtime_reopen(case: dict) -> None:
    native = json.loads(case["native_reload_path"].read_text())
    native.pop("sha256")
    native["schema"] = export.OBSERVER_NATIVE_RELOAD_SCHEMA
    native["source_terminal_acceptance_sha256"] = json.loads(case["terminal_path"].read_text())[
        "sha256"
    ]
    native = _sealed(native)
    case["native_reload_path"].write_text(json.dumps(native))
    case["native_reload_sha256"] = export._hash(case["native_reload_path"])

    prepared = export.bind_source(
        active_canary_binding_path=case["active_canary_binding_path"],
        active_canary_binding_sha256=case["active_canary_binding_sha256"],
        checkpoint_path=case["checkpoint_path"],
        checkpoint_sha256=case["checkpoint_sha256"],
        terminal_path=case["terminal_path"],
        terminal_sha256=case["terminal_sha256"],
        native_reload_path=case["native_reload_path"],
        native_reload_sha256=case["native_reload_sha256"],
        validate_historical=False,
    )
    prepared.pop("checkpoint_manifest")
    result = export.export(
        active_canary_binding_path=case["active_canary_binding_path"],
        active_canary_binding_sha256=case["active_canary_binding_sha256"],
        checkpoint_path=case["checkpoint_path"],
        checkpoint_sha256=case["checkpoint_sha256"],
        terminal_path=case["terminal_path"],
        terminal_sha256=case["terminal_sha256"],
        native_reload_path=case["native_reload_path"],
        native_reload_sha256=case["native_reload_sha256"],
        output=case["output"],
        prepared_source=prepared,
    )
    assert result["source"]["native_reload_acceptance"]["receipt_sha256"] == native["sha256"]


def test_tensor_inventory_rejects_nonfinite_values_and_false_index_total(tmp_path: Path) -> None:
    root = tmp_path / "nonfinite"
    _hf_tree(root, {"weight": torch.tensor([float("nan")], dtype=torch.bfloat16)})
    with pytest.raises(ValueError, match="non-finite"):
        export.tensor_inventory(root)

    shutil.rmtree(root)
    _hf_tree(root, {"weight": torch.ones((2,), dtype=torch.bfloat16)})
    index = json.loads((root / "model.safetensors.index.json").read_text())
    index["metadata"]["total_size"] = 0
    (root / "model.safetensors.index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match="total size"):
        export.tensor_inventory(root)


def test_export_runs_one_create_once_converter_pass(case: dict, monkeypatch) -> None:
    original = export._invoke_converter
    calls = 0

    def counted(source: Path, destination: Path, metadata: Path, log: Path) -> None:
        nonlocal calls
        original(source, destination, metadata, log)
        calls += 1

    monkeypatch.setattr(export, "_invoke_converter", counted)
    result = _run(case)

    assert calls == 1
    assert result["source_equivalence"]["all_trained_values_match_source"] is True


def test_post_job_inspection_never_materializes_dcp_values(
    case: dict, monkeypatch
) -> None:
    """Controller/export/reload acceptance must stay within operator RAM."""

    result = _run(case)
    artifact_path = case["output"] / "EXPORT.json"
    artifact_file_sha256 = export._hash(artifact_path)
    plan = {
        "stage": "export",
        "artifact_path": str(case["output"]),
        "source": result["source"],
    }
    prepared_export = {
        "path": str(artifact_path),
        "file_sha256": artifact_file_sha256,
        "receipt_sha256": result["sha256"],
        "tensor_inventory_sha256": result["tensor_inventory_sha256"],
        "active_canary_binding": result["source"]["active_canary_binding"],
    }

    def forbidden(_generation: Path) -> list[dict]:
        raise AssertionError("post-Job validation materialized DCP values")

    monkeypatch.setattr(export, "_source_value_inventory", forbidden)
    monkeypatch.setattr(export_job, "validate_plan", lambda *_args, **_kwargs: None)
    reopened, _ = export_job._inspect_plan_bound_export(
        plan, artifact_path, artifact_file_sha256
    )
    assert reopened == result
    changed_plan = {
        **plan,
        "source": {**plan["source"], "source_plan_sha256": "f" * 64},
    }
    with pytest.raises(ValueError, match="plan-bound source evidence"):
        export_job._inspect_plan_bound_export(
            changed_plan, artifact_path, artifact_file_sha256
        )

    monkeypatch.setattr(export, "_validate_reload_result", lambda _value: 1.0)
    monkeypatch.setattr(export, "_validate_controller", lambda *_args: 2.0)
    monkeypatch.setattr(export, "_validate_release", lambda *_args: None)
    reload_result = {
        "export_path": str(artifact_path),
        "export_file_sha256": artifact_file_sha256,
        "export_receipt_sha256": result["sha256"],
        "export_tensor_inventory_sha256": result["tensor_inventory_sha256"],
        "native_prediction_probe": result["source"]["prediction_probe"],
        "hf_prediction_probe": result["source"]["prediction_probe"],
    }
    validated = export._validate_reload_evidence(
        result=reload_result,
        result_path=case["output"].parent / "reload" / "HF_RELOAD_VALIDATED.json",
        result_file_sha256="b" * 64,
        controller={},
        controller_path=case["output"].parent / "reload" / "HF_RELOAD_CONTROLLER_TERMINAL.json",
        controller_file_sha256="c" * 64,
        release={},
        plan={"source": {"export": prepared_export}},
        submission={},
    )
    assert validated == result


def test_export_rejects_values_that_do_not_match_independent_dcp_reload(
    case: dict, monkeypatch
) -> None:
    monkeypatch.setattr(
        export,
        "_source_value_inventory",
        lambda _generation: [
            {
                "name": "model.language_model.weight",
                "shape": [2, 4],
                "dtype": "BF16",
                "value_sha256": "0" * 64,
            }
        ],
    )
    with pytest.raises(ValueError, match="independent DCP reload"):
        _run(case)


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


def test_prediction_receipt_is_digest_only_and_requires_a_robust_boundary() -> None:
    logits = torch.arange(32, dtype=torch.float32)
    receipt = export._prediction_receipt(logits)
    assert receipt["prediction_sha256"] == "sha256:" + digest(list(range(31, 15, -1)))
    assert set(receipt) == export._PREDICTION_FIELDS
    assert receipt["logits_included"] is False
    assert not any(isinstance(value, list) for value in receipt.values())

    logits[15:17] = 16
    with pytest.raises(ValueError, match="not numerically robust"):
        export._prediction_receipt(logits)


def test_gpu_reload_is_separate_and_matches_native_prediction(case: dict, monkeypatch) -> None:
    result = _run(case)
    tensor_rows = {row["name"]: row for row in result["tensor_inventory"]}
    value_observer = export._runtime_value_observation

    class Tensor:
        dtype = torch.bfloat16
        device = SimpleNamespace(type="cuda")

        def __init__(self, shape, value_sha256):
            self.shape = torch.Size(shape)
            self.value_sha256 = value_sha256

    state = {
        name: Tensor(row["shape"], row["value_sha256"])
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
    monkeypatch.setattr(
        export,
        "_runtime_value_observation",
        lambda tensor: (True, tensor.value_sha256),
    )
    monkeypatch.setattr(export, "_hf_prediction_probe", lambda _model: _prediction_probe())
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
    assert reloaded["all_runtime_values_finite"] is True
    assert reloaded["all_runtime_values_match_export"] is True
    assert reloaded["semantic_prediction_match"] is True
    assert reloaded["hf_prediction_probe"] == reloaded["native_prediction_probe"]
    assert reloaded["external_gpu_release_verified"] is False
    assert reloaded["forwards"] == 1
    assert all(
        reloaded[key] == 0
        for key in (
            "optimizer_updates",
            "rollouts",
            "verifier_calls",
            "backwards",
            "checkpoint_writes",
            "wandb_events",
        )
    )

    changed = _prediction_probe("b" * 64)
    monkeypatch.setattr(export, "_hf_prediction_probe", lambda _model: changed)
    with pytest.raises(ValueError, match="differs from the native"):
        export.gpu_reload(
            case["output"] / "EXPORT.json",
            export._hash(case["output"] / "EXPORT.json"),
            evidence_root / "MISMATCH.json",
            run_name="hf-reload-mismatch",
        )

    assert value_observer(torch.tensor([1, 2], dtype=torch.bfloat16))[0]
    assert not value_observer(torch.tensor([float("inf")], dtype=torch.bfloat16))[0]


def test_reload_acceptance_requires_bound_plan_and_submission(case: dict) -> None:
    _run(case)
    evidence_root = case["output"].parent / "reload-evidence"
    evidence_root.mkdir()
    result_path = evidence_root / "HF_RELOAD_VALIDATED.json"
    result_path.write_text("{}")
    for name in ("HF_RELOAD_CONTROLLER_TERMINAL.json", "HF_RELOAD_RELEASE.json"):
        (evidence_root / name).write_text("{}")

    with pytest.raises(TypeError):
        export.accept_gpu_reload(
            result_path=result_path,
            controller_path=evidence_root / "HF_RELOAD_CONTROLLER_TERMINAL.json",
            release_path=evidence_root / "HF_RELOAD_RELEASE.json",
            output=evidence_root / "HF_RELOAD_ACCEPTED.json",
        )
