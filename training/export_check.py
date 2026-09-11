"""Check a sealed Qwen HF export on CPU, then optionally smoke-test one GPU.

This is not checkpoint/optimizer recovery, a serving-engine qualification, or
an evaluation. Only a fixed synthetic text is used; no output tokens are logged.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from .checkpoints import receipt
from .post_sft_artifacts import QWEN36_EXACT_MTP_OMISSION_KEYS, _safetensor_layout
from .sft_runtime import SOURCE_SHA256, _checked_file, _unsigned_digest, digest, write_receipt

PATCH = "skyrl/backends/skyrl_train/patches/fleet_qwen_torch_gdn.py"
MODEL_CLASS = "Qwen3_5ForConditionalGeneration"
SYNTHETIC_TEXT = "Return the number 2."


def inspect_export(path: Path, sha256: str) -> tuple[dict, dict]:
    _checked_file(path, sha256)
    proof = receipt(path)
    root = path.parent
    if (
        path.name != "EXPORT.json"
        or root.is_symlink()
        or proof.get("schema") != "cyber_native_checkpoint_hf_export_v1"
        or proof.get("model_repo") != "Qwen/Qwen3.8-27B"
        or proof.get("output_root") != str(root)
        or proof.get("dtype") != "BF16"
        or proof.get("optimizer_steps_executed") != 0
        or proof.get("all_output_tensors_reopened_equal") is not True
        or proof.get("source_inventory_sizes_mtimes_unchanged") is not True
        or set(proof.get("restored_base_tensors", [])) != set(QWEN36_EXACT_MTP_OMISSION_KEYS)
    ):
        raise ValueError("unsupported or incomplete export receipt")
    if {p.name for p in root.iterdir()} != set(proof["files"]) | {path.name}:
        raise ValueError("export inventory differs from receipt")
    for name, spec in proof["files"].items():
        # Equality to immediate directory-entry names above excludes traversal.
        item = root / name
        _checked_file(item, spec["sha256"])
        if item.stat().st_size != spec["bytes"]:
            raise ValueError("export payload size mismatch")
    layout, _ = _safetensor_layout(root)
    if len(layout) != proof["trained_tensors"] + len(proof["restored_base_tensors"]) or any(
        spec["dtype"] != "BF16" for spec in layout.values()
    ):
        raise ValueError("export tensor count/dtype mismatch")
    return proof, layout


def model_contract(model, layout: dict, restored: list[str], *, gpu: bool) -> dict:
    """Artifact-only MTP tensors are not missing runtime model parameters."""
    import torch

    shapes = {name: list(t.shape) for name, t in model.state_dict().items()}
    if (
        type(model).__name__ != MODEL_CLASS
        or set(layout) - set(shapes) != set(restored)
        or set(shapes) - set(layout)
        or any(shape != layout[name]["shape"] for name, shape in shapes.items())
    ):
        raise ValueError("loaded model class/key/shape contract mismatch")
    parameters = list(model.parameters())
    if any(
        p.device != torch.device("cuda:0" if gpu else "meta") or p.dtype != torch.bfloat16
        for p in parameters
    ):
        raise ValueError("loaded parameter device/dtype mismatch")
    return {
        "model_class": type(model).__name__,
        "state_tensors": len(shapes),
        "named_parameters": len(parameters),
        "parameter_values": sum(p.numel() for p in parameters),
        "state_shapes_sha256": _unsigned_digest(shapes),
    }


def enable_native_patch(model) -> int:
    spec = importlib.util.find_spec(PATCH.removesuffix(".py").replace("/", "."))
    if spec is None or spec.origin is None:
        raise ValueError("qualified Qwen Torch GDN patch is absent")
    _checked_file(Path(spec.origin), SOURCE_SHA256[PATCH])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.enable_qwen35_torch_gdn(model)


def synthetic_forward(model, tokenizer, *, device: str) -> dict:
    import torch

    tokens = tokenizer(SYNTHETIC_TEXT, return_tensors="pt").to(device)
    model.eval().requires_grad_(False)
    with torch.inference_mode():
        output = model(**tokens, use_cache=False)
        finite = bool(torch.isfinite(output.logits).all().item())
        del output
        generated = model.generate(
            **tokens,
            do_sample=False,
            min_new_tokens=2,
            max_new_tokens=2,
            pad_token_id=tokenizer.eos_token_id,
        )
    if not finite or generated.shape != (1, tokens["input_ids"].shape[1] + 2):
        raise ValueError("synthetic forward/generation did not satisfy the smoke test")
    return {
        "finite_logits": True,
        "generated_tokens": 2,
        "input_tokens": tokens["input_ids"].shape[1],
    }


def check(path: Path, sha256: str, output: Path, *, gpu: bool = False) -> dict:
    import torch
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer

    if output.exists() or output.is_symlink():
        raise FileExistsError("check receipt already exists")
    if output.resolve().is_relative_to(path.parent.resolve()):
        raise ValueError("check receipt must be outside the immutable export")
    if torch.cuda.is_available() != gpu or (gpu and torch.cuda.device_count() != 1):
        raise ValueError("CPU preflight needs zero GPUs; GPU check needs exactly one")
    before, layout = inspect_export(path, sha256)
    root = path.parent
    kwargs = {"local_files_only": True, "trust_remote_code": False}
    config = AutoConfig.from_pretrained(root, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(root, **kwargs)
    config.dtype = torch.bfloat16
    config._attn_implementation = "eager"
    extra = {}
    if gpu:
        torch.cuda.set_device(0)
        model, info = AutoModelForImageTextToText.from_pretrained(
            root,
            dtype=torch.bfloat16,
            device_map={"": "cuda:0"},
            attn_implementation="eager",
            output_loading_info=True,
            **kwargs,
        )
        # HF versions may intentionally omit MTP from their diagnostics. The
        # complete state/layout comparison below, not warning text, is authority.
        if any(info.get(key) for key in ("missing_keys", "mismatched_keys", "error_msgs")) or (
            set(info.get("unexpected_keys", [])) - set(before["restored_base_tensors"])
        ):
            raise ValueError("HF load reported missing/mismatched or unexpected tensors")
        extra["patched_linear_layers"] = enable_native_patch(model)
    else:
        with torch.device("meta"):
            model = AutoModelForImageTextToText.from_config(config, trust_remote_code=False)
    contract = model_contract(model, layout, before["restored_base_tensors"], gpu=gpu)
    if gpu:
        extra.update(synthetic_forward(model, tokenizer, device="cuda:0"))
        torch.cuda.synchronize()
        extra["gpu_name"] = torch.cuda.get_device_name(0)
        extra["peak_memory_bytes"] = torch.cuda.max_memory_allocated(0)
        # Rehash every file after loading/forward, not just sizes/mtimes.
        inspect_export(path, sha256)
    result = {
        "schema": "cyber_hf_export_check_v1",
        "status": "passed",
        "export_sha256": sha256.removeprefix("sha256:"),
        "export_receipt_sha256": before["receipt_sha256"],
        "checker_sha256": digest(Path(__file__)),
        "optimizer_steps_executed": 0,
        "gpus": int(gpu),
        "gpu_reload_verified": gpu,
        "serving_qualified": False,
        "synthetic_only": True,
        "source_unchanged": True,
        "attention_implementation": "eager",
        "loader_contract": contract,
        **extra,
    }
    write_receipt(output, result)
    return receipt(output)
