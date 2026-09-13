"""Fail-closed SFT → Miles initial-policy binding; no model bytes or GPU work."""

import json
from pathlib import Path

import pytest

from cyber_post_train.jobs import digest
from training import export_check, miles, miles_conversion, miles_training
from training.io import digest_json
from training.models import bound_model
from training.post_sft_artifacts import QWEN36_EXACT_MTP_OMISSION_KEYS
from training.sft_runtime import write_receipt

ROOT = Path(__file__).resolve().parents[1]


def _base(root: str) -> dict:
    return bound_model(
        json.loads((ROOT / "configs/models/qwen38-27b-1d4bf0f2.lock.json").read_text()),
        json.loads((ROOT / "configs/models/qwen38-27b-1d4bf0f2.weights.json").read_text()),
        root,
    )


@pytest.fixture
def accepted(tmp_path, monkeypatch):
    model_root = "/mnt/sfs/jobs/synthetic-sft/hf-export"
    base = _base(model_root)
    sidecars = {
        row["path"]: {"bytes": 1, "sha256": row["sha256"]}
        for row in base["files"]
        if not row["path"].endswith(".safetensors")
        and row["path"] != "model.safetensors.index.json"
    }
    export = {
        "schema": "cyber_native_checkpoint_hf_export_v1",
        "model_repo": base["repo"],
        "model_revision": base["revision"],
        "output_root": model_root,
        "optimizer_step": 44,
        "optimizer_steps_executed": 0,
        "gpu_reload_verified": False,
        "dtype": "BF16",
        "all_output_tensors_reopened_equal": True,
        "source_inventory_sizes_mtimes_unchanged": True,
        "source_checkpoint_receipt_sha256": "a" * 64,
        "source_manifest_file_sha256": "e" * 64,
        "source_plan_sha256": "b" * 64,
        "code_sha256": {"training/export.py": "f" * 64},
        "trained_tensors": 1,
        "restored_base_tensors": sorted(QWEN36_EXACT_MTP_OMISSION_KEYS),
        "sidecars": {name: item["sha256"] for name, item in sidecars.items()},
        "files": {
            "model-00001-of-00001.safetensors": {"bytes": 2, "sha256": "c" * 64},
            "model.safetensors.index.json": {"bytes": 1, "sha256": "d" * 64},
            **sidecars,
        },
    }
    export_path = tmp_path / "EXPORT.json"
    write_receipt(export_path, export)
    exported = json.loads(export_path.read_text())
    export_file_sha256 = miles_conversion._hash(export_path)
    gpu = {
        "schema": "cyber_hf_export_check_v1",
        "status": "passed",
        "gpus": 1,
        "gpu_reload_verified": True,
        "source_unchanged": True,
        "finite_logits": True,
        "optimizer_steps_executed": 0,
        "synthetic_only": True,
        "serving_qualified": False,
        "attention_implementation": "eager",
        "generated_tokens": 2,
        "patched_linear_layers": 1,
        "checker_sha256": "9" * 64,
        "loader_contract": {
            "model_class": export_check.MODEL_CLASS,
            "parameter_values": 1,
        },
        "export_sha256": export_file_sha256,
        "export_receipt_sha256": exported["receipt_sha256"],
    }
    gpu_path = tmp_path / "GPU_CHECK.json"
    write_receipt(gpu_path, gpu)
    declared_export = model_root + "/EXPORT.json"
    declared_gpu = "/mnt/sfs/jobs/synthetic-sft/GPU_CHECK.json"
    config = {
        "lock": str(ROOT / "configs/models/qwen38-27b-1d4bf0f2.lock.json"),
        "weights": str(ROOT / "configs/models/qwen38-27b-1d4bf0f2.weights.json"),
        "root": model_root,
        "export": {
            "path": declared_export,
            "snapshot": str(export_path),
            "sha256": export_file_sha256,
        },
        "gpu_check": {
            "path": declared_gpu,
            "snapshot": str(gpu_path),
            "sha256": miles_conversion._hash(gpu_path),
        },
        "sft_source": {
            "plan_sha256": "b" * 64,
            "checkpoint_receipt_sha256": "a" * 64,
            "checkpoint_manifest_sha256": "e" * 64,
            "export_code_sha256": digest_json(exported["code_sha256"]),
            "checker_sha256": "9" * 64,
        },
    }
    local = {declared_export: export_path, declared_gpu: gpu_path}

    def strict(export_sfs, export_sha, gpu_sfs, gpu_sha, checker_sha):
        assert (str(export_sfs), str(gpu_sfs)) == (declared_export, declared_gpu)
        observed_export = json.loads(export_path.read_text())
        observed_gpu = json.loads(gpu_path.read_text())
        if (
            miles_conversion._hash(export_path) != export_sha
            or miles_conversion._hash(gpu_path) != gpu_sha
            or observed_gpu.get("gpu_reload_verified") is not True
            or observed_gpu.get("checker_sha256")
            != checker_sha.removeprefix("sha256:")
        ):
            raise ValueError("strict synthetic acceptance failed")
        return observed_export, {}, observed_gpu

    def reopen(reference, label):
        path = local[reference["path"]]
        receipt = json.loads(path.read_text())
        if (
            miles_conversion._hash(path) != reference["file_sha256"]
            or receipt["receipt_sha256"] != reference["receipt_sha256"]
            or receipt["receipt_sha256"]
            != digest({k: v for k, v in receipt.items() if k != "receipt_sha256"})
        ):
            raise ValueError(f"{label} file digest mismatch")
        return receipt, reference

    monkeypatch.setattr(miles_conversion, "_reopen_reference", reopen)
    monkeypatch.setattr(export_check, "inspect_accepted_export", strict)
    return config, base, export_path, gpu_path


def test_conversion_binds_accepted_sft_export_without_changing_base_path(accepted, tmp_path):
    config, base, _, _ = accepted
    frozen = miles_conversion.bind_model_source(
        {key: config[key] for key in ("lock", "weights", "root")}, relative_to=tmp_path
    )
    assert frozen == base and "initial_policy" not in frozen

    plan = miles_conversion.compile_conversion(
        {
            "name": "synthetic-sft-conversion",
            "output_root": "/mnt/sfs/jobs/synthetic-sft-conversion",
            "model": config,
        },
        relative_to=tmp_path,
    )
    assert plan["optimizer_steps"] == 0
    assert plan["model"]["initial_policy"]["kind"] == "sft_hf_export"
    assert plan["model"]["initial_policy"]["sft_optimizer_step"] == 44
    assert plan["model"]["initial_policy"]["source_plan_sha256"] == "sha256:" + "b" * 64
    assert plan["model"]["initial_policy"]["source_checkpoint_receipt_sha256"] == (
        "sha256:" + "a" * 64
    )
    assert plan["model"]["initial_policy"]["source_checkpoint_manifest_sha256"] == (
        "sha256:" + "e" * 64
    )
    assert plan["model"]["initial_policy"]["checker_sha256"] == "sha256:" + "9" * 64
    assert plan["model"]["weight_manifest_sha256"] != base["weight_manifest_sha256"]
    assert "snapshot" not in json.dumps(plan)
    miles_conversion._reopen_initial_policy(plan["model"])
    miles_conversion._reopen_initial_policy(plan["model"], strict=True)


def _stage(config, export_path, gpu_path, tmp_path):
    export = json.loads(export_path.read_text())
    gpu = json.loads(gpu_path.read_text())
    runtime_root = "/mnt/sfs/jobs/synthetic-sft-stage/payload"
    stage_path = tmp_path / "STAGE.json"
    source_export = {
        "path": config["export"]["path"],
        "file_sha256": miles_conversion._hash(export_path),
        "receipt_sha256": export["receipt_sha256"],
    }
    source_gpu = {
        "path": config["gpu_check"]["path"],
        "file_sha256": miles_conversion._hash(gpu_path),
        "receipt_sha256": gpu["receipt_sha256"],
    }
    value = {
        "schema": miles_conversion.SFT_STAGE_SCHEMA,
        "status": "passed",
        "source_root": config["root"],
        "runtime_root": runtime_root,
        "source_export": source_export,
        "source_gpu_check": source_gpu,
        "files": miles_conversion._export_files(export),
        "export_copy": {
            "path": "EXPORT.json",
            "size": export_path.stat().st_size,
            "sha256": miles_conversion._hash(export_path),
        },
        "source_stable": True,
        "copied_equal": True,
        "zero_gpus": 0,
        "runtime_identity": {
            "uid": 1000,
            "gid": 2000,
            "supplemental_groups": [100, 2000],
        },
        "permissions": {"directory_mode": "0750", "file_mode": "0640", "gid": 2000},
    }
    write_receipt(stage_path, value)
    config["runtime_stage"] = {
        "root": runtime_root,
        "receipt": {
            "path": "/mnt/sfs/jobs/synthetic-sft-stage/STAGE.json",
            "snapshot": str(stage_path),
            "sha256": miles_conversion._hash(stage_path),
        },
    }
    return stage_path, value


def test_conversion_separates_accepted_evidence_from_group_readable_runtime_copy(
    accepted, tmp_path
):
    config, _, export_path, gpu_path = accepted
    stage_path, _ = _stage(config, export_path, gpu_path, tmp_path)
    plan = miles_conversion.compile_conversion(
        {
            "name": "synthetic-staged-sft-conversion",
            "output_root": "/mnt/sfs/jobs/synthetic-staged-sft-conversion",
            "model": config,
        },
        relative_to=tmp_path,
    )
    assert plan["model"]["root"] == config["runtime_stage"]["root"]
    assert plan["model"]["initial_policy"]["accepted_root"] == config["root"]
    assert plan["model"]["initial_policy"]["runtime_stage"] == {
        "path": config["runtime_stage"]["receipt"]["path"],
        "file_sha256": miles_conversion._hash(stage_path),
        "receipt_sha256": json.loads(stage_path.read_text())["receipt_sha256"],
    }
    assert plan["model"]["initial_policy"]["export"]["path"] == config["export"]["path"]


@pytest.mark.parametrize(
    "defect",
    (
        "source",
        "runtime",
        "files",
        "export_copy",
        "identity",
        "permissions",
        "gpus",
        "inside",
        "base_only",
    ),
)
def test_runtime_stage_rejects_unbound_or_unsafe_copy(accepted, tmp_path, defect):
    config, _, export_path, gpu_path = accepted
    stage_path, value = _stage(config, export_path, gpu_path, tmp_path)
    if defect == "base_only":
        base = {key: config[key] for key in ("lock", "weights", "root")}
        base["runtime_stage"] = config["runtime_stage"]
        with pytest.raises(ValueError, match="accepted SFT"):
            miles_conversion.bind_model_source(base, relative_to=tmp_path)
        return
    value.pop("receipt_sha256", None)
    if defect == "source":
        value["source_root"] += "-other"
    elif defect == "runtime":
        value["runtime_root"] += "-other"
    elif defect == "files":
        value["files"][0]["sha256"] = "0" * 64
    elif defect == "export_copy":
        value["export_copy"]["sha256"] = "0" * 64
    elif defect == "identity":
        value["runtime_identity"]["uid"] = 0
    elif defect == "permissions":
        value["permissions"]["file_mode"] = "0600"
    elif defect == "gpus":
        value["zero_gpus"] = True
    else:
        config["runtime_stage"]["receipt"]["path"] = (
            config["runtime_stage"]["root"] + "/STAGE.json"
        )
    stage_path.unlink()
    write_receipt(stage_path, value)
    config["runtime_stage"]["receipt"]["sha256"] = miles_conversion._hash(stage_path)
    with pytest.raises(ValueError):
        miles_conversion.bind_model_source(config, relative_to=tmp_path)


@pytest.mark.parametrize(
    "defect",
    (
        "missing_check",
        "wrong_revision",
        "changed_sidecar",
        "zero_step",
        "no_gpu",
        "wrong_source",
        "wrong_checkpoint_manifest",
        "wrong_exporter",
        "wrong_checker",
    ),
)
def test_incomplete_or_mismatched_sft_handoff_is_rejected(accepted, tmp_path, defect):
    config, _, export_path, gpu_path = accepted
    if defect == "missing_check":
        config.pop("gpu_check")
    elif defect == "wrong_source":
        config["sft_source"]["plan_sha256"] = "0" * 64
    elif defect == "wrong_checkpoint_manifest":
        config["sft_source"]["checkpoint_manifest_sha256"] = "0" * 64
    elif defect == "wrong_exporter":
        config["sft_source"]["export_code_sha256"] = "0" * 64
    elif defect == "wrong_checker":
        config["sft_source"]["checker_sha256"] = "0" * 64
    else:
        path = gpu_path if defect == "no_gpu" else export_path
        value = json.loads(path.read_text())
        value.pop("receipt_sha256")
        if defect == "wrong_revision":
            value["model_revision"] = "0" * 40
        elif defect == "changed_sidecar":
            value["files"]["tokenizer.json"]["sha256"] = "0" * 64
        elif defect == "zero_step":
            value["optimizer_step"] = 0
        else:
            value["gpu_reload_verified"] = False
        path.unlink()
        write_receipt(path, value)
        config["gpu_check" if path == gpu_path else "export"]["sha256"] = (
            miles_conversion._hash(path)
        )
        if path == export_path:
            gpu = json.loads(gpu_path.read_text())
            gpu.pop("receipt_sha256")
            gpu["export_sha256"] = miles_conversion._hash(export_path)
            gpu["export_receipt_sha256"] = json.loads(export_path.read_text())["receipt_sha256"]
            gpu_path.unlink()
            write_receipt(gpu_path, gpu)
            config["gpu_check"]["sha256"] = miles_conversion._hash(gpu_path)
    with pytest.raises(ValueError):
        miles_conversion.bind_model_source(config, relative_to=tmp_path)


def test_sft_acceptance_receipts_must_be_cluster_visible(tmp_path):
    path = tmp_path / "EXPORT.json"
    write_receipt(path, {"schema": "synthetic"})
    with pytest.raises(ValueError, match="cluster-visible"):
        miles_conversion._signed_reference(
            {
                "path": str(path),
                "snapshot": str(path),
                "sha256": miles_conversion._hash(path),
            },
            tmp_path,
            "SFT export",
        )


def test_runtime_reopens_exact_sft_acceptance_receipts(accepted, tmp_path):
    config, _, _, gpu_path = accepted
    model = miles_conversion.bind_model_source(config, relative_to=tmp_path)
    miles_conversion._reopen_initial_policy(model)
    gpu_path.write_bytes(gpu_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="file digest"):
        miles_conversion._reopen_initial_policy(model)


def test_miles_rl_consumes_only_the_matching_zero_step_sft_conversion(accepted, tmp_path):
    config, _, _, _ = accepted
    model = miles_conversion.bind_model_source(config, relative_to=tmp_path)
    data = {
        "schema": "cyber_miles_data_v1",
        "name": "synthetic-sft-rl",
        "tokenizer": {key: model[key] for key in ("repo", "revision")},
        "template_sha256": "sha256:" + miles.TEMPLATE_SHA256,
        "limits": {
            "context_tokens": 98304,
            "response_tokens": 81920,
            "max_tokens_per_turn": 4096,
        },
        "files": {
            split: {"path": split + ".jsonl", "rows": 1, "sha256": "sha256:" + "1" * 64}
            for split in ("train", "dev")
        },
    }
    data["sha256"] = "sha256:" + digest(data)
    data_path = tmp_path / "data.json"
    data_path.write_text(json.dumps(data))
    checkpoint = {
        "schema": "cyber_miles_checkpoint_v1",
        "model": model,
        "image": miles.IMAGE,
        "root": "/mnt/sfs/jobs/synthetic-sft-conversion/torch-dist",
        "optimizer_steps": 0,
        "files": [],
        "gpu_reload_verified": False,
    }
    checkpoint["sha256"] = digest(checkpoint)
    checkpoint_path = tmp_path / "checkpoint.json"
    checkpoint_path.write_text(json.dumps(checkpoint))
    run = {
        "backend": "miles",
        "name": data["name"],
        "output_root": "/mnt/sfs/jobs/synthetic-sft-rl",
        "model": config,
        "data": {"manifest": str(data_path), "root": "/mnt/sfs/jobs/synthetic-sft-data"},
        "checkpoint": {
            "manifest": str(checkpoint_path),
            "sha256": miles_conversion._hash(checkpoint_path),
        },
        "recipe": {"steps": 1, "groups": 1, "samples_per_prompt": 8},
        "wandb": {"entity": "synthetic", "project": "synthetic", "run_id": data["name"]},
        "cluster": {"target": "dev", "priority": "c1"},
    }
    plan = miles_training.compile_rl(run, relative_to=tmp_path)
    assert plan["model"] == checkpoint["model"]
    assert plan["arguments"]["model_root"] == config["root"]
    assert plan["execution"]["cluster_target"] == "dev"
    assert plan["arguments"]["wandb_run_id"] == data["name"]
    assert "webexploitbench" not in json.dumps(plan).lower()

    run["cluster"]["target"] = "prod"
    with pytest.raises(ValueError, match="exact reviewed run"):
        miles_training.compile_rl(run, relative_to=tmp_path)
    run["cluster"]["target"] = "dev"

    checkpoint["model"] = _base(config["root"])
    checkpoint["sha256"] = digest({k: v for k, v in checkpoint.items() if k != "sha256"})
    checkpoint_path.write_text(json.dumps(checkpoint))
    run["checkpoint"]["sha256"] = miles_conversion._hash(checkpoint_path)
    with pytest.raises(ValueError, match="initial policy"):
        miles_training.compile_rl(run, relative_to=tmp_path)
