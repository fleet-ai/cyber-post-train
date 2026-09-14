"""Exact HF → native Miles checkpoint preparation, never an optimizer step.

The pinned native converter owns model construction and distributed saving.
We own input identity, one bounded child process group, and create-once evidence.
No downloads, model edits, Ray lifecycle commands or process-name kills.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import signal
import stat
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path, PurePosixPath

from cyber_post_train.jobs import bundled_request, digest, quantity

from .miles import IMAGE

SCHEMA = "cyber_miles_conversion_v1"
CONVERTER_SHA256 = "0c2541d30073777a30344273a3773844a70ca1961287520c0496a1cec18d43f6"
DEADLINE_SECONDS = 1800
SFT_SOURCE_SCHEMA = "cyber_miles_sft_initial_policy_v1"
SFT_STAGE_SCHEMA = "cyber_sft_export_stage_v1"


def _sha256(value: object, label: str) -> str:
    normalized = str(value).removeprefix("sha256:")
    if re.fullmatch(r"[a-f0-9]{64}", normalized) is None:
        raise ValueError(f"{label} is not a SHA-256 digest")
    return normalized


def _json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _hash(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError("missing or indirect input file")
    before = path.stat()
    with path.open("rb") as stream:
        result = hashlib.file_digest(stream, "sha256").hexdigest()
    after = path.stat()
    if any(
        getattr(before, k) != getattr(after, k)
        for k in (
            "st_dev",
            "st_ino",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
    ):
        raise ValueError("file changed during hashing")
    return result


def _write(path: Path, value: dict) -> dict:
    value = {**value, "sha256": digest(value)}
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    return value


def _runtime_path(value: object, label: str) -> Path:
    if not isinstance(value, str):
        raise ValueError(f"{label} path is invalid")
    declared = PurePosixPath(value)
    if (
        declared.parts[:3] != ("/", "mnt", "sfs")
        or len(declared.parts) < 6
        or ".." in declared.parts
        or str(declared) != value
    ):
        raise ValueError(f"{label} must be a cluster-visible immutable SFS file")
    return Path(str(declared))


def _signed_reference(value: dict, relative_to: Path, label: str) -> tuple[dict, dict]:
    """Bind a local receipt snapshot to its exact cluster-visible runtime path."""
    if not isinstance(value, dict) or set(value) != {"path", "snapshot", "sha256"}:
        raise ValueError(f"{label} needs an SFS path, local snapshot and file digest")
    runtime_path = _runtime_path(value["path"], label)
    snapshot = Path(value["snapshot"])
    if not snapshot.is_absolute():
        snapshot = relative_to / snapshot
    file_sha256 = _hash(snapshot)
    if file_sha256 != _sha256(value["sha256"], f"{label} file digest"):
        raise ValueError(f"{label} file digest mismatch")
    receipt = json.loads(snapshot.read_text())
    if not isinstance(receipt, dict):
        raise ValueError(f"{label} must be a JSON object")
    receipt_sha256 = _sha256(receipt.get("receipt_sha256"), f"{label} receipt digest")
    if receipt_sha256 != digest({k: v for k, v in receipt.items() if k != "receipt_sha256"}):
        raise ValueError(f"{label} self-digest mismatch")
    return receipt, {
        "path": str(runtime_path),
        "file_sha256": file_sha256,
        "receipt_sha256": receipt_sha256,
    }


def _reopen_reference(reference: dict, label: str) -> tuple[dict, dict]:
    if set(reference) != {"path", "file_sha256", "receipt_sha256"}:
        raise ValueError(f"{label} reference fields changed")
    path = _runtime_path(reference["path"], label)
    if _hash(path) != _sha256(reference["file_sha256"], f"{label} file digest"):
        raise ValueError(f"{label} file digest mismatch")
    receipt = json.loads(path.read_text())
    if not isinstance(receipt, dict):
        raise ValueError(f"{label} must be a JSON object")
    self_sha256 = _sha256(receipt.get("receipt_sha256"), f"{label} receipt digest")
    if self_sha256 != _sha256(
        reference["receipt_sha256"], f"{label} receipt digest"
    ) or self_sha256 != digest({k: v for k, v in receipt.items() if k != "receipt_sha256"}):
        raise ValueError(f"{label} self-digest mismatch")
    return receipt, {**reference, "file_sha256": _sha256(reference["file_sha256"], label)}


def _export_files(export: dict) -> list[dict]:
    files = export.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("SFT HF export file inventory is absent")
    normalized = []
    for name, item in sorted(files.items()):
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or not isinstance(item, dict)
            or set(item) != {"bytes", "sha256"}
            or type(item["bytes"]) is not int
            or item["bytes"] <= 0
        ):
            raise ValueError("SFT HF export file inventory is invalid")
        normalized.append(
            {
                "path": name,
                "size": item["bytes"],
                "sha256": _sha256(item["sha256"], "SFT HF export payload digest"),
            }
        )
    return normalized


def _validate_runtime_stage(
    value: dict,
    *,
    accepted_root: str,
    runtime_root: str,
    files: list[dict],
    export_reference: dict,
    gpu_check_reference: dict,
    check_payload: bool,
) -> None:
    """Validate a byte-identical, group-readable copy without trusting it as new evidence."""
    expected_keys = {
        "schema",
        "status",
        "source_root",
        "runtime_root",
        "source_export",
        "source_gpu_check",
        "files",
        "export_copy",
        "source_stable",
        "copied_equal",
        "zero_gpus",
        "runtime_identity",
        "permissions",
        "receipt_sha256",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ValueError("SFT runtime-stage receipt fields changed")
    if (
        value.get("schema") != SFT_STAGE_SCHEMA
        or value.get("status") != "passed"
        or value.get("source_root") != accepted_root
        or value.get("runtime_root") != runtime_root
        or value.get("source_export") != export_reference
        or value.get("source_gpu_check") != gpu_check_reference
        or value.get("files") != files
        or value.get("source_stable") is not True
        or value.get("copied_equal") is not True
        or type(value.get("zero_gpus")) is not int
        or value.get("zero_gpus") != 0
        or value.get("runtime_identity")
        != {"uid": 1000, "gid": 2000, "supplemental_groups": [100, 2000]}
        or value.get("permissions") != {"directory_mode": "0750", "file_mode": "0640", "gid": 2000}
    ):
        raise ValueError("SFT runtime-stage receipt does not bind the accepted export")
    source = Path(accepted_root)
    runtime = Path(runtime_root)
    if source == runtime or source in runtime.parents or runtime in source.parents:
        raise ValueError("accepted and runtime SFT roots must not overlap")
    copied_export = value.get("export_copy")
    if (
        not isinstance(copied_export, dict)
        or set(copied_export) != {"path", "size", "sha256"}
        or copied_export.get("path") != "EXPORT.json"
        or type(copied_export.get("size")) is not int
        or copied_export["size"] <= 0
        or _sha256(copied_export.get("sha256"), "staged EXPORT digest")
        != _sha256(export_reference.get("file_sha256"), "accepted EXPORT digest")
    ):
        raise ValueError("SFT runtime-stage EXPORT copy is invalid")
    if not check_payload:
        return
    if runtime.is_symlink() or not runtime.is_dir():
        raise ValueError("SFT runtime-stage payload root is absent or indirect")
    root_stat = runtime.stat()
    if stat.S_IMODE(root_stat.st_mode) != 0o750 or root_stat.st_gid != 2000:
        raise ValueError("SFT runtime-stage directory permissions changed")
    for item in [*files, copied_export]:
        path = runtime / item["path"]
        observed = path.stat()
        if (
            path.is_symlink()
            or not path.is_file()
            or observed.st_size != item["size"]
            or stat.S_IMODE(observed.st_mode) != 0o640
            or observed.st_gid != 2000
            or _hash(path) != _sha256(item["sha256"], "staged SFT payload digest")
        ):
            raise ValueError("SFT runtime-stage payload differs from its accepted inventory")
    copied = json.loads((runtime / "EXPORT.json").read_text())
    if (
        not isinstance(copied, dict)
        or _sha256(copied.get("receipt_sha256"), "copied EXPORT receipt digest")
        != _sha256(export_reference.get("receipt_sha256"), "accepted EXPORT receipt digest")
        or copied.get("output_root") != accepted_root
        or _export_files(copied) != files
    ):
        raise ValueError("staged EXPORT receipt differs from accepted evidence")


def _accepted_sft_model(
    base: dict,
    accepted_root: str,
    runtime_root: str,
    export: dict,
    export_reference: dict,
    gpu_check_reference: dict,
    expected_source: dict,
    runtime_stage_reference: dict | None,
) -> dict:
    """Bind an already exported and independently reloaded SFT initial policy."""
    from .post_sft_artifacts import QWEN36_EXACT_MTP_OMISSION_KEYS

    if (
        export.get("schema") != "cyber_native_checkpoint_hf_export_v1"
        or export.get("model_repo") != base["repo"]
        or export.get("model_revision") != base["revision"]
        or export.get("output_root") != accepted_root
        or export.get("dtype") != "BF16"
        or export.get("optimizer_steps_executed") != 0
        or type(export.get("optimizer_step")) is not int
        or export["optimizer_step"] < 1
        or export.get("gpu_reload_verified") is not False
        or export.get("all_output_tensors_reopened_equal") is not True
        or export.get("source_inventory_sizes_mtimes_unchanged") is not True
        or set(export.get("restored_base_tensors", [])) != set(QWEN36_EXACT_MTP_OMISSION_KEYS)
        or type(export.get("trained_tensors")) is not int
        or export["trained_tensors"] < 1
    ):
        raise ValueError("SFT HF export is not a complete zero-update handoff")
    normalized = _export_files(export)
    names = {item["path"] for item in normalized}
    weight_files = [item for item in normalized if item["path"].endswith(".safetensors")]
    if not weight_files or "model.safetensors.index.json" not in names:
        raise ValueError("SFT HF export lacks indexed safetensors")
    expected_sidecars = {
        item["path"]: _sha256(item["sha256"], "base sidecar digest")
        for item in base["files"]
        if not item["path"].endswith(".safetensors")
        and item["path"] != "model.safetensors.index.json"
    }
    observed_sidecars = {
        item["path"]: item["sha256"]
        for item in normalized
        if not item["path"].endswith(".safetensors")
        and item["path"] != "model.safetensors.index.json"
    }
    if observed_sidecars != expected_sidecars:
        raise ValueError("SFT HF export tokenizer/runtime sidecars differ from the model lock")
    reported_sidecars = export.get("sidecars")
    if (
        not isinstance(reported_sidecars, dict)
        or {
            name: _sha256(value, "SFT HF export sidecar digest")
            for name, value in reported_sidecars.items()
        }
        != expected_sidecars
    ):
        raise ValueError("SFT HF export sidecar receipt differs from the model lock")
    export_receipt_sha256 = _sha256(
        export_reference.get("receipt_sha256"), "SFT HF export receipt digest"
    )
    if export_receipt_sha256 != _sha256(
        export.get("receipt_sha256"), "SFT HF export receipt digest"
    ):
        raise ValueError("SFT HF export reference changed")
    expected = {
        "source_checkpoint_receipt_sha256": _sha256(
            expected_source.get("checkpoint_receipt_sha256"), "expected SFT checkpoint digest"
        ),
        "source_manifest_file_sha256": _sha256(
            expected_source.get("checkpoint_manifest_sha256"),
            "expected SFT checkpoint manifest digest",
        ),
        "source_plan_sha256": _sha256(
            expected_source.get("plan_sha256"), "expected SFT plan digest"
        ),
    }
    if any(
        _sha256(export.get(key), f"SFT HF export {key}") != value for key, value in expected.items()
    ):
        raise ValueError("SFT HF export differs from the selected source checkpoint or plan")
    weight_manifest_sha256 = _json_sha256(weight_files)
    if _json_sha256(export.get("code_sha256")) != expected_source["export_code_sha256"]:
        raise ValueError("SFT HF export producer identity differs from the selected source")
    if weight_manifest_sha256 == base["weight_manifest_sha256"]:
        raise ValueError("SFT initial policy must not claim the frozen base weight inventory")
    return {
        "repo": base["repo"],
        "revision": base["revision"],
        "root": runtime_root,
        "files": normalized,
        "weight_manifest_sha256": weight_manifest_sha256,
        "initial_policy": {
            "schema": SFT_SOURCE_SCHEMA,
            "kind": "sft_hf_export",
            "base_weight_manifest_sha256": base["weight_manifest_sha256"],
            "sft_optimizer_step": export["optimizer_step"],
            "source_checkpoint_receipt_sha256": "sha256:"
            + expected["source_checkpoint_receipt_sha256"],
            "source_checkpoint_manifest_sha256": "sha256:"
            + expected["source_manifest_file_sha256"],
            "source_plan_sha256": "sha256:" + expected["source_plan_sha256"],
            "export_code_sha256": expected_source["export_code_sha256"],
            "checker_sha256": "sha256:" + expected_source["checker_sha256"],
            "accepted_root": accepted_root,
            "export": export_reference,
            "gpu_check": gpu_check_reference,
            **(
                {"runtime_stage": runtime_stage_reference}
                if runtime_stage_reference is not None
                else {}
            ),
        },
    }


def bind_model_source(config: dict, *, relative_to: Path) -> dict:
    """Bind either the exact frozen base or an accepted SFT HF export."""
    from .models import bound_model
    from .sft import _known, _sfs_root, read_mapping

    _known(
        config,
        {
            "lock",
            "weights",
            "root",
            "export",
            "gpu_check",
            "sft_source",
            "runtime_stage",
        },
        "model",
    )
    accepted_root = _sfs_root(config["root"], "model root")
    base = bound_model(
        read_mapping(relative_to / config["lock"]),
        read_mapping(relative_to / config["weights"]),
        accepted_root,
    )
    references = (config.get("export"), config.get("gpu_check"), config.get("sft_source"))
    if references == (None, None, None):
        if config.get("runtime_stage") is not None:
            raise ValueError("a runtime stage is valid only for an accepted SFT initial policy")
        return base
    if any(value is None for value in references):
        raise ValueError("SFT initial policy requires export, GPU check, and source identity")
    expected_source = config["sft_source"]
    if not isinstance(expected_source, dict) or set(expected_source) != {
        "plan_sha256",
        "checkpoint_receipt_sha256",
        "checkpoint_manifest_sha256",
        "export_code_sha256",
        "checker_sha256",
    }:
        raise ValueError("SFT source identity needs exact plan, checkpoint and producer digests")
    expected_source = {
        **expected_source,
        "export_code_sha256": "sha256:"
        + _sha256(expected_source["export_code_sha256"], "SFT exporter code digest"),
        "checker_sha256": _sha256(expected_source["checker_sha256"], "SFT checker digest"),
    }
    export, export_reference = _signed_reference(config["export"], relative_to, "SFT export")
    gpu_check, gpu_check_reference = _signed_reference(
        config["gpu_check"], relative_to, "SFT GPU check"
    )
    export_path = Path(export_reference["path"])
    if export_path != Path(accepted_root) / "EXPORT.json":
        raise ValueError("SFT export receipt must be inside the selected model root")
    from .export_check import validate_accepted_export_receipts

    validate_accepted_export_receipts(
        export,
        gpu_check,
        export_reference["file_sha256"],
        expected_source["checker_sha256"],
    )
    accepted = _accepted_sft_model(
        base,
        accepted_root,
        accepted_root,
        export,
        export_reference,
        gpu_check_reference,
        expected_source,
        None,
    )
    stage = config.get("runtime_stage")
    if stage is None:
        return accepted
    if not isinstance(stage, dict) or set(stage) != {"root", "receipt"}:
        raise ValueError("SFT runtime stage needs an exact root and signed receipt")
    runtime_root = _sfs_root(stage["root"], "SFT runtime-stage root")
    stage_receipt, stage_reference = _signed_reference(
        stage["receipt"], relative_to, "SFT runtime stage"
    )
    receipt_path = Path(stage_reference["path"])
    runtime_path = Path(runtime_root)
    if receipt_path == runtime_path or runtime_path in receipt_path.parents:
        raise ValueError("SFT runtime-stage receipt must be outside its payload root")
    _validate_runtime_stage(
        stage_receipt,
        accepted_root=accepted_root,
        runtime_root=runtime_root,
        files=accepted["files"],
        export_reference=export_reference,
        gpu_check_reference=gpu_check_reference,
        check_payload=False,
    )
    accepted["root"] = runtime_root
    accepted["initial_policy"]["runtime_stage"] = stage_reference
    return accepted


def _reopen_initial_policy(model: dict, *, strict: bool = False) -> None:
    source = model.get("initial_policy")
    if source is None:
        return
    if (
        not isinstance(source, dict)
        or source.get("schema") != SFT_SOURCE_SCHEMA
        or source.get("kind") != "sft_hf_export"
        or source.get("base_weight_manifest_sha256") == model.get("weight_manifest_sha256")
        or type(source.get("sft_optimizer_step")) is not int
        or source["sft_optimizer_step"] < 1
    ):
        raise ValueError("SFT initial-policy binding is invalid")
    stage_reference = source.get("runtime_stage")
    if stage_reference is not None:
        accepted_root = source.get("accepted_root")
        if not isinstance(accepted_root, str):
            raise ValueError("SFT accepted evidence root is absent")
        stage, reopened = _reopen_reference(stage_reference, "SFT runtime stage")
        if reopened != stage_reference:
            raise ValueError("SFT runtime-stage reference changed")
        _validate_runtime_stage(
            stage,
            accepted_root=accepted_root,
            runtime_root=model["root"],
            files=model["files"],
            export_reference=source["export"],
            gpu_check_reference=source["gpu_check"],
            check_payload=strict,
        )
        return
    export, export_reference = _reopen_reference(source["export"], "SFT export")
    gpu, gpu_reference = _reopen_reference(source["gpu_check"], "SFT GPU check")
    if strict:
        from .export_check import inspect_accepted_export

        strict_export, _, strict_gpu = inspect_accepted_export(
            Path(export_reference["path"]),
            export_reference["file_sha256"],
            Path(gpu_reference["path"]),
            gpu_reference["file_sha256"],
            source["checker_sha256"],
        )
        if strict_export != export or strict_gpu != gpu:
            raise ValueError("SFT acceptance evidence changed during validation")
    if (
        export_reference != source["export"]
        or gpu_reference != source["gpu_check"]
        or export.get("output_root") != source.get("accepted_root", model["root"])
        or export.get("optimizer_step") != source["sft_optimizer_step"]
        or _export_files(export) != model["files"]
        or _sha256(export.get("source_checkpoint_receipt_sha256"), "SFT checkpoint digest")
        != _sha256(source.get("source_checkpoint_receipt_sha256"), "SFT checkpoint digest")
        or _sha256(export.get("source_manifest_file_sha256"), "SFT checkpoint manifest digest")
        != _sha256(
            source.get("source_checkpoint_manifest_sha256"),
            "SFT checkpoint manifest digest",
        )
        or _sha256(export.get("source_plan_sha256"), "SFT plan digest")
        != _sha256(source.get("source_plan_sha256"), "SFT plan digest")
        or _json_sha256(export.get("code_sha256")) != source.get("export_code_sha256")
        or _sha256(gpu.get("checker_sha256"), "SFT checker digest")
        != _sha256(source.get("checker_sha256"), "SFT checker digest")
        or _sha256(gpu.get("export_sha256"), "GPU check export digest")
        != export_reference["file_sha256"]
        or _sha256(gpu.get("export_receipt_sha256"), "GPU check export receipt digest")
        != export_reference["receipt_sha256"]
    ):
        raise ValueError("SFT initial-policy evidence changed")


def compile_conversion(config: dict, *, relative_to: Path) -> dict:
    from .sft import RESOURCES, _known, _sfs_root

    _known(config, {"name", "output_root", "model", "cluster"}, "Miles conversion")
    model = config["model"]
    cluster = config.get("cluster", {})
    _known(cluster, {"priority", "resources"}, "cluster")
    bound = bind_model_source(model, relative_to=relative_to)
    output = _sfs_root(config["output_root"], "output root")
    if bound["repo"] != "Qwen/Qwen3.8-27B":
        raise ValueError("only the exact native Qwen3.8 text conversion is supported")
    input_roots = {Path(bound["root"])}
    if "initial_policy" in bound:
        input_roots.add(Path(bound["initial_policy"]["accepted_root"]))
    if any(
        Path(output) == root or Path(output) in root.parents or root in Path(output).parents
        for root in input_roots
    ):
        raise ValueError("model and output directories must not overlap")
    plan = {
        "schema": SCHEMA,
        "run_name": config["name"],
        "output_root": output,
        "model": bound,
        "runtime_sha256": _hash(Path(__file__)),
        "native_converter_sha256": CONVERTER_SHA256,
        "optimizer_steps": 0,
        "deadline_seconds": DEADLINE_SECONDS,
        "execution": {
            "image": IMAGE,
            "priority": cluster.get("priority", "c1"),
            "resources": {**RESOURCES, **cluster.get("resources", {})},
        },
    }
    job_request(plan)
    return plan


def job_request(plan: dict) -> dict:
    if (
        plan["schema"] != SCHEMA
        or plan["runtime_sha256"] != _hash(Path(__file__))
        or plan["native_converter_sha256"] != CONVERTER_SHA256
        or plan["execution"]["image"] != IMAGE
        or plan["optimizer_steps"] != 0
        or plan["deadline_seconds"] != DEADLINE_SECONDS
    ):
        raise ValueError("conversion runtime/plan drift")
    resources = plan["execution"]["resources"]
    if quantity(resources["cpu_request"]) < 64 or quantity(resources["memory_request"]) < quantity(
        "512Gi"
    ):
        raise ValueError("conversion requires its reviewed 64 CPU / 512Gi loading envelope")
    root = Path(__file__).resolve().parents[1]
    files = {
        name: (root / name).read_text()
        for name in (
            "training/miles_conversion.py",
            "training/miles.py",
            "training/export_check.py",
            "training/checkpoints.py",
            "training/post_sft_artifacts.py",
            "training/sft_runtime.py",
            "training/io.py",
            "cyber_post_train/jobs.py",
        )
    }
    files.update({"training/__init__.py": "", "cyber_post_train/__init__.py": ""})
    files["plan.json"] = json.dumps(plan, sort_keys=True, separators=(",", ":"))
    return bundled_request(
        {
            "name": plan["run_name"],
            "title": plan["run_name"] + " zero-step native conversion",
            "run_dir": plan["output_root"],
            "image": IMAGE,
            "workers": 1,
            "gpus_per_worker": 8,
            "resources": plan["execution"]["resources"],
            "priority_class": plan["execution"]["priority"],
            "requeueIfPreempted": False,
            "secrets": [],
            "env": {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "CUDA_DEVICE_MAX_CONNECTIONS": "1",
                "WANDB_MODE": "disabled",
                "PYTHONUNBUFFERED": "1",
            },
        },
        files,
        "training.miles_conversion",
        ["--plan", "plan.json", "--sha256", digest(plan)],
    )


def check_inputs(plan: dict) -> None:
    root = Path(plan["model"]["root"])
    if "initial_policy" in plan["model"]:
        _reopen_initial_policy(plan["model"], strict=True)
    else:
        for item in plan["model"]["files"]:
            if _hash(root / item["path"]) != item["sha256"].removeprefix("sha256:"):
                raise ValueError("staged model differs from exact inventory")
    config = json.loads((root / "config.json").read_text())
    if config.get("auto_map") or config.get("text_config", {}).get("auto_map"):
        raise ValueError("native conversion must not resolve remote model code")


def native_arguments(plan: dict) -> list[str]:
    from fti.trainers.miles.run_fleet import _RECIPES
    from miles.utils.external_utils.command_utils import repo_base_dir
    from miles.utils.external_utils.model_args_utils import load_model_args

    profile = _RECIPES["qwen3.8-27b"]
    source = Path(repo_base_dir) / "tools/convert_hf_to_torch_dist.py"
    if (
        _hash(source) != CONVERTER_SHA256
        or profile.backend != "megatron"
        or profile.vision
        or profile.megatron_model_type != "qwen3.8-27B"
    ):
        raise ValueError("native conversion source/model profile changed")
    # Native auto-PP splits eight ranks rather than building eight whole models.
    # The source has no optimizer and calls save_checkpoint with optimizer=None.
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc-per-node=8",
        str(source),
        *shlex.split(load_model_args(profile.megatron_model_type)),
        "--hf-checkpoint",
        plan["model"]["root"],
        "--save",
        plan["output_root"] + "/torch-dist",
    ]


def preflight(plan: dict) -> dict:
    import torch
    from transformers import AutoConfig

    if torch.cuda.is_available():
        raise ValueError("conversion preflight must not allocate GPUs")
    request = job_request(plan)
    if Path(plan["output_root"]).exists():
        raise FileExistsError("conversion output already exists")
    check_inputs(plan)
    config = AutoConfig.from_pretrained(
        plan["model"]["root"],
        trust_remote_code=False,
        local_files_only=True,
    )
    text = config.get_text_config()
    if (text.num_hidden_layers, text.hidden_size, text.vocab_size) != (64, 5120, 248320):
        raise ValueError("staged architecture differs from the Qwen conversion profile")
    argv = native_arguments(plan)
    return {
        "schema": "cyber_miles_conversion_cpu_preflight_v1",
        "status": "passed",
        "gpus": 0,
        "plan_sha256": digest(plan),
        "request_sha256": digest(request),
        "native_argv_sha256": digest(argv),
        "model_files": len(plan["model"]["files"]),
        "gpu_conversion_verified": False,
    }


def _child(argv: list[str], log: Path, timeout: float) -> None:
    if timeout <= 0:
        raise TimeoutError("conversion startup allowance exhausted")
    env = {**os.environ, "PYTHONPATH": str(Path(argv[6]).parent.parent) + ":/root/Megatron-LM"}
    env.pop("CONVERT_KEEP_PP1", None)
    with log.open("x") as stream:
        os.chmod(log, 0o600)
        process = subprocess.Popen(
            argv,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            if process.wait(timeout=timeout) != 0:
                raise RuntimeError("native conversion failed; private output preserved")
        finally:
            # Reap only our own torchrun group, including on interruption. Never
            # address processes by name or touch the Jobs API's Ray processes.
            if process.poll() is None:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)
                with suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=30)
            # A dead torchrun parent is not proof every child exited. Reap the
            # remaining owned group even if its parent has already returned.
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=30)


def inventory(root: Path) -> list[dict]:
    if root.is_symlink():
        raise ValueError("indirect checkpoint root")
    if (root / "latest_checkpointed_iteration.txt").read_text().strip() != "release":
        raise ValueError("native release checkpoint tracker absent")
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("indirect checkpoint payload")
        if path.is_file():
            size = path.stat().st_size
            if size == 0:
                raise ValueError("empty checkpoint payload")
            files.append({"path": str(path.relative_to(root)), "size": size})
    names = {r["path"] for r in files}
    if "release/.metadata" not in names or not any(n.endswith(".distcp") for n in names):
        raise ValueError("incomplete native distributed checkpoint")
    return files


def run(plan: dict) -> dict:
    import torch

    if os.environ.get("RUN_DIR") != plan["output_root"]:
        raise ValueError("Jobs API output binding drift")
    if torch.cuda.device_count() != 8:
        raise ValueError("native conversion must own exactly eight visible GPUs")
    started = time.monotonic()
    root = Path(plan["output_root"])
    _write(root / "STARTED.json", {"plan_sha256": digest(plan), "started_at": time.time()})
    destination = root / "torch-dist"

    def interrupted(signum, frame):
        raise TimeoutError("conversion interrupted or exceeded its fixed deadline")

    previous = {s: signal.signal(s, interrupted) for s in (signal.SIGALRM, signal.SIGTERM)}
    signal.alarm(DEADLINE_SECONDS)
    try:
        destination.mkdir(mode=0o700, exist_ok=False)
        check_inputs(plan)
        argv = native_arguments(plan)
        _child(
            argv, root / "private-conversion.log", DEADLINE_SECONDS - (time.monotonic() - started)
        )
        files = inventory(destination)
        return _write(
            root / "CONVERSION_COMPLETE.json",
            {
                "schema": SCHEMA,
                "status": "native_conversion_complete",
                "plan_sha256": digest(plan),
                "native_argv_sha256": digest(argv),
                "optimizer_steps": 0,
                "files": files,
                "completed_at": time.time(),
                "checkpoint_sha256_verified": False,
                "gpu_reload_verified": False,
            },
        )
    except BaseException as exc:
        _write(
            root / "FAILED.json",
            {
                "status": "failed",
                "error_class": type(exc).__name__,
                "plan_sha256": digest(plan),
            },
        )
        raise RuntimeError("conversion failed; preserve output, no automatic retry") from None
    finally:
        signal.alarm(0)
        for number, handler in previous.items():
            signal.signal(number, handler)


def seal(plan: dict, output: Path) -> dict:
    """Hash a terminal owned conversion on CPU after GPU release; not reload proof."""
    import torch

    if torch.cuda.is_available():
        raise ValueError("checkpoint sealing must not allocate GPUs")
    root = Path(plan["output_root"])
    receipt = json.loads((root / "CONVERSION_COMPLETE.json").read_text())
    if (
        (root / "FAILED.json").exists()
        or receipt["sha256"] != digest({k: v for k, v in receipt.items() if k != "sha256"})
        or receipt["plan_sha256"] != digest(plan)
        or receipt["optimizer_steps"] != 0
        or receipt["status"] != "native_conversion_complete"
    ):
        raise ValueError("unaccepted conversion completion")
    check_inputs(plan)
    checkpoint = root / "torch-dist"
    if output == checkpoint or checkpoint in output.parents:
        raise ValueError("seal must not alter the native checkpoint directory")
    files = inventory(checkpoint)
    if files != receipt["files"]:
        raise ValueError("checkpoint inventory changed after completion")
    hashed = [{**f, "sha256": _hash(checkpoint / f["path"])} for f in files]
    if inventory(checkpoint) != files:
        raise ValueError("checkpoint changed while sealing")
    return _write(
        output,
        {
            "schema": "cyber_miles_checkpoint_v1",
            "model": plan["model"],
            "image": IMAGE,
            "root": str(checkpoint),
            "plan_sha256": digest(plan),
            "conversion_receipt_sha256": receipt["sha256"],
            "files": hashed,
            "optimizer_steps": 0,
            "gpu_reload_verified": False,
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    ray = None
    task = None
    try:
        plan = json.loads(args.plan.read_text())
        if digest(plan) != args.sha256:
            raise ValueError("plan digest mismatch")
        import ray

        # HTTPMode starts a zero-GPU driver. Claim all eight GPUs for this one
        # native torchrun worker inside the existing allocation; do not start
        # another Ray cluster or override CUDA visibility by hand.
        ray.init(
            address="auto",
            log_to_driver=False,
            runtime_env={
                "env_vars": {
                    "PYTHONPATH": str(Path(__file__).resolve().parents[1])
                    + ":"
                    + os.environ.get("PYTHONPATH", ""),
                }
            },
        )
        task = ray.remote(num_cpus=1, num_gpus=8)(run).remote(plan)
        result = ray.get(task, timeout=DEADLINE_SECONDS + 120)
        print(json.dumps({k: result[k] for k in ("status", "optimizer_steps", "sha256")}))
    except BaseException as exc:
        print(json.dumps({"status": "failed", "error_class": type(exc).__name__}))
        raise SystemExit(1) from None
    finally:
        if ray is not None and ray.is_initialized():
            if task is not None:
                ray.cancel(task, force=True)  # no-op when done; only this task
            ray.shutdown()  # disconnect this driver, not a node-wide Ray stop


if __name__ == "__main__":
    main()
