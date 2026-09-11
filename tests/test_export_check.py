import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import AutoModelForImageTextToText, PreTrainedTokenizerFast, Qwen3_5Config

from training import export_check as c
from training.sft_runtime import digest, write_receipt


def tiny_model():
    config = Qwen3_5Config(
        text_config={
            "vocab_size": 128,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "linear_num_value_heads": 4,
            "linear_num_key_heads": 2,
            "linear_key_head_dim": 8,
            "linear_value_head_dim": 8,
            "layer_types": ["linear_attention", "full_attention"],
        },
        vision_config={
            "depth": 1,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 4,
            "out_hidden_size": 32,
            "num_position_embeddings": 16,
            "patch_size": 2,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
        },
        image_token_id=125,
        video_token_id=126,
        vision_start_token_id=123,
        vision_end_token_id=124,
    )
    config._attn_implementation = "eager"
    return AutoModelForImageTextToText.from_config(config).bfloat16()


@pytest.fixture
def fixture(tmp_path):
    model = tiny_model()
    root = tmp_path / "export"
    root.mkdir()
    model.config.save_pretrained(root)
    native = Tokenizer(
        models.WordLevel(
            {"<unk>": 0, "</s>": 1, "Return": 2, "the": 3, "number": 4, "2": 5, ".": 6},
            unk_token="<unk>",
        )
    )
    native.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=native, unk_token="<unk>", eos_token="</s>"
    )
    tokenizer.save_pretrained(root)
    weights = {name: value.clone() for name, value in model.state_dict().items()}
    weights.update(
        {k: torch.ones(1, dtype=torch.bfloat16) for k in c.QWEN36_EXACT_MTP_OMISSION_KEYS}
    )
    save_file(weights, root / "model.safetensors", metadata={"format": "pt"})
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: "model.safetensors" for k in weights}})
    )
    proof = {
        "schema": "cyber_native_checkpoint_hf_export_v1",
        "model_repo": "Qwen/Qwen3.8-27B",
        "output_root": str(root),
        "dtype": "BF16",
        "optimizer_steps_executed": 0,
        "all_output_tensors_reopened_equal": True,
        "source_inventory_sizes_mtimes_unchanged": True,
        "restored_base_tensors": list(c.QWEN36_EXACT_MTP_OMISSION_KEYS),
        "trained_tensors": len(model.state_dict()),
        "files": {p.name: {"bytes": p.stat().st_size, "sha256": digest(p)} for p in root.iterdir()},
    }
    path = root / "EXPORT.json"
    write_receipt(path, proof)
    return path, model, tokenizer, tmp_path / "checked.json"


def test_real_cpu_meta_contract_and_synthetic_forward(fixture):
    path, model, tokenizer, output = fixture
    result = c.check(path, digest(path), output)
    assert result["status"] == "passed" and not result["gpu_reload_verified"]
    assert result["loader_contract"]["parameter_values"] == sum(
        p.numel() for p in model.parameters()
    )
    assert result["optimizer_steps_executed"] == 0 and result["serving_qualified"] is False
    assert result["loader_contract"]["state_tensors"] == len(model.state_dict())
    assert c.synthetic_forward(model, tokenizer, device="cpu")["generated_tokens"] == 2
    # Exercise the installed HF loader's diagnostics on a genuine small model.
    loaded, info = AutoModelForImageTextToText.from_pretrained(
        path.parent,
        dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=False,
        attn_implementation="eager",
        output_loading_info=True,
    )
    assert not info.get("missing_keys") and not info.get("mismatched_keys")
    assert not set(info["unexpected_keys"]) - set(c.QWEN36_EXACT_MTP_OMISSION_KEYS)
    assert c.synthetic_forward(loaded, tokenizer, device="cpu")["finite_logits"]


@pytest.mark.parametrize(
    "defect", ["receipt", "inventory", "hash", "size", "dtype", "count", "path"]
)
def test_bad_export_rejected_before_loading(fixture, defect):
    path, _, _, _ = fixture
    proof = json.loads(path.read_text())
    proof.pop("receipt_sha256")
    if defect == "receipt":
        proof["model_repo"] = "different-model"
    elif defect == "inventory":
        (path.parent / "extra").touch()
    elif defect == "hash":
        proof["files"]["config.json"]["sha256"] = "0" * 64
    elif defect == "size":
        proof["files"]["config.json"]["bytes"] += 1
    elif defect == "dtype":
        proof["dtype"] = "FP32"
    elif defect == "count":
        proof["trained_tensors"] += 1
    else:
        proof["files"]["../outside"] = proof["files"].pop("config.json")
    path.unlink()
    write_receipt(path, proof)
    with pytest.raises(ValueError):
        c.inspect_export(path, digest(path))


@pytest.mark.parametrize(
    "defect", ["existing", "inside", "unexpected_gpu", "missing_gpu", "two_gpu"]
)
def test_output_and_device_boundary(fixture, monkeypatch, defect):
    path, _, _, output = fixture
    gpu = defect in {"missing_gpu", "two_gpu"}
    if defect == "existing":
        output.touch()
    if defect == "inside":
        output = path.parent / "bad.json"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: defect in {"unexpected_gpu", "two_gpu"})
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    with pytest.raises((ValueError, FileExistsError)):
        c.check(path, digest(path), output, gpu=gpu)


def test_model_layout_and_device_failures(fixture):
    path, model, _, _ = fixture
    proof, layout = c.inspect_export(path, digest(path))
    with pytest.raises(ValueError, match="device/dtype"):
        c.model_contract(model, layout, proof["restored_base_tensors"], gpu=False)
    model.to("meta")
    layout[next(iter(model.state_dict()))]["shape"] = [999]
    with pytest.raises(ValueError, match="class/key/shape"):
        c.model_contract(model, layout, proof["restored_base_tensors"], gpu=False)


@pytest.mark.parametrize("defect", ["nonfinite", "short_generation"])
def test_invalid_synthetic_result_fails(fixture, monkeypatch, defect):
    _, model, tokenizer, _ = fixture
    if defect == "nonfinite":
        monkeypatch.setattr(
            model, "forward", lambda **k: SimpleNamespace(logits=torch.tensor([float("nan")]))
        )
    monkeypatch.setattr(
        model,
        "generate",
        lambda **k: torch.zeros(
            (1, k["input_ids"].shape[1] + (1 if defect == "short_generation" else 2)),
            dtype=torch.long,
        ),
    )
    with pytest.raises(ValueError, match="smoke test"):
        c.synthetic_forward(model, tokenizer, device="cpu")


def test_gpu_orchestration_uses_actual_parameters_not_device_map(fixture, monkeypatch):
    path, model, _, output = fixture
    called = []
    model.hf_device_map = {"": 0}  # HF integer ordinal is legitimate, not a failure.

    def load(*args, **kwargs):
        called.append(kwargs)
        return model, {"unexpected_keys": list(c.QWEN36_EXACT_MTP_OMISSION_KEYS)}

    monkeypatch.setattr(AutoModelForImageTextToText, "from_pretrained", load)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "set_device", lambda *_: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *_: "synthetic GPU")
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *_: 1)
    monkeypatch.setattr(c, "model_contract", lambda *a, **k: {"unit_test_only": True})
    monkeypatch.setattr(c, "enable_native_patch", lambda *_: 1)
    monkeypatch.setattr(
        c, "synthetic_forward", lambda *a, **k: {"finite_logits": True, "generated_tokens": 2}
    )
    result = c.check(path, digest(path), output, gpu=True)
    assert result["gpu_reload_verified"] and called[0]["device_map"] == {"": "cuda:0"}
    output.unlink()
    monkeypatch.setattr(
        AutoModelForImageTextToText,
        "from_pretrained",
        lambda *a, **k: (model, {"missing_keys": ["weight"]}),
    )
    with pytest.raises(ValueError, match="HF load"):
        c.check(path, digest(path), output, gpu=True)


def test_native_patch_identity_checked(tmp_path, monkeypatch):
    monkeypatch.setattr(c.importlib.util, "find_spec", lambda *_: None)
    with pytest.raises(ValueError, match="absent"):
        c.enable_native_patch(None)
    patch = tmp_path / "patch.py"
    patch.write_text("def enable_qwen35_torch_gdn(model): return 3\n")
    spec = c.importlib.util.spec_from_file_location("fixture_patch", patch)
    monkeypatch.setattr(c.importlib.util, "find_spec", lambda *_: spec)
    with pytest.raises(ValueError, match="digest"):
        c.enable_native_patch(None)
    monkeypatch.setitem(c.SOURCE_SHA256, c.PATCH, digest(patch))
    assert c.enable_native_patch(None) == 3
