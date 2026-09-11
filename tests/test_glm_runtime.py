"""Actual installed HF/PEFT paths on tiny synthetic FP8 checkpoints, CPU only."""

import dataclasses
import importlib.util
import json
import random
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

if importlib.util.find_spec("torch") is None or importlib.util.find_spec("peft") is None:
    raise unittest.SkipTest("synthetic GLM test requires the pinned Torch/PEFT runtime")
import torch
from safetensors.torch import load_file, save_file
from transformers import GlmMoeDsaConfig, GlmMoeDsaForCausalLM

from training.glm_runtime import (
    BaseIdentity,
    adapter_parameters,
    bf16_fp8_conversion,
    load_checkpoint,
    load_lora_model,
    make_optimizer,
    save_checkpoint,
    sha_file,
    use_worker,
    worker_class,
)


def tiny_config_kwargs():
    return {
        "vocab_size": 128,
        "hidden_size": 64,
        "intermediate_size": 128,
        "moe_intermediate_size": 32,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "n_shared_experts": 1,
        "n_routed_experts": 4,
        "num_experts_per_tok": 2,
        "n_group": 1,
        "topk_group": 1,
        "q_lora_rank": 32,
        "kv_lora_rank": 16,
        "qk_rope_head_dim": 8,
        "qk_nope_head_dim": 8,
        "v_head_dim": 16,
        "index_head_dim": 16,
        "index_n_heads": 2,
        "index_topk": 4,
        "mlp_layer_types": ["dense", "dense", "dense", "sparse"],
        "indexer_types": ["full", "full", "shared", "shared"],
        "max_position_embeddings": 64,
        "use_cache": False,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0},
        "attention_dropout": 0.0,
        "tie_word_embeddings": False,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
    }


def synthetic_base(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(1729)
    root = tmp_path / "synthetic-fp8"
    config = GlmMoeDsaConfig(**tiny_config_kwargs())
    config._attn_implementation = "eager"
    model = GlmMoeDsaForCausalLM(config).cpu()
    model.save_pretrained(root, safe_serialization=True)
    path = root / "model.safetensors"
    weights = load_file(str(path))
    # The exact staged GLM artifact stores separate 2-D expert projections,
    # whereas HF's in-memory model fuses them into 3-D expert parameters.
    for name, tensor in list(weights.items()):
        if ".experts." not in name or tensor.ndim != 3:
            continue
        prefix, projection = name.split(".experts.")
        assert projection in {"gate_up_proj", "down_proj"}, projection
        del weights[name]
        for expert, matrix in enumerate(tensor):
            if projection == "gate_up_proj":
                gate, up = matrix.chunk(2, dim=0)
                weights[f"{prefix}.experts.{expert}.gate_proj.weight"] = gate.contiguous()
                weights[f"{prefix}.experts.{expert}.up_proj.weight"] = up.contiguous()
            else:
                weights[f"{prefix}.experts.{expert}.down_proj.weight"] = matrix.contiguous()
    for name, tensor in list(weights.items()):
        if (
            tensor.ndim == 2
            and name.endswith(".weight")
            and "embed_tokens" not in name
            and "lm_head" not in name
            and "indexer.weights_proj" not in name
        ):
            weights[name] = tensor.to(torch.float8_e4m3fn)
            weights[name.removesuffix(".weight") + ".weight_scale_inv"] = torch.ones(
                (1, 1), dtype=torch.float32
            )
    save_file(weights, str(path))
    cfg = json.loads((root / "config.json").read_text())
    cfg["quantization_config"] = {
        "quant_method": "fp8",
        "fmt": "e4m3",
        "activation_scheme": "dynamic",
        "weight_block_size": [128, 128],
    }
    (root / "config.json").write_text(json.dumps(cfg))
    identity = BaseIdentity(
        "zai-org/GLM-5.3", "3" * 40, sha_file(root / "config.json"), sha_file(path), "a" * 64
    )
    return root, identity


def loaded(base, meta=False):
    return load_lora_model(*base, meta_init=meta, rank=4, alpha=8, attention="eager")


def step(model, optimizer, scheduler, ids):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    labels = ids.clone()
    labels[:, :4] = -100
    loss = model(input_ids=ids, labels=labels, use_cache=False).loss
    assert torch.isfinite(loss)
    loss.backward()
    gradients = [p.grad for p in adapter_parameters(model).values() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert any((g != 0).any() for g in gradients)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)


def check_fp8_cpu_dequantization_bf16_and_strict_router_meta_parity(base):
    real = loaded(base)
    meta = loaded(base, meta=True)
    real_layout = {n: (tuple(p.shape), p.dtype) for n, p in real.named_parameters()}
    meta_layout = {n: (tuple(p.shape), p.dtype) for n, p in meta.named_parameters()}
    assert real_layout == meta_layout
    assert all(p.is_meta for p in meta.parameters())
    assert all(p.device.type == "cpu" for p in real.parameters())
    buffers = dict(real.named_buffers())
    empty_buffers = dict(meta.named_buffers())
    assert set(buffers) == set(empty_buffers)
    assert all(v.dtype == empty_buffers[n].dtype for n, v in buffers.items())
    strict = [b for n, b in buffers.items() if n.endswith("e_score_correction_bias")]
    assert strict and all(b.dtype == torch.float32 for b in strict)
    frozen = [p for p in real.parameters() if not p.requires_grad]
    assert frozen and all(p.dtype == torch.bfloat16 for p in frozen)
    assert any(".experts.gate_up_proj" in n and p.ndim == 3 for n, p in real.named_parameters())
    assert not any(type(m).__name__.startswith("FP8") for m in real.modules())
    stored = load_file(str(base[0] / "model.safetensors"))
    for name, parameter in real.named_parameters():
        if not name.endswith(".experts.gate_up_proj"):
            continue
        prefix = name.removeprefix("base_model.model.").removesuffix(".gate_up_proj")
        expected = torch.stack(
            [
                torch.cat(
                    [
                        stored[f"{prefix}.{i}.gate_proj.weight"].to(torch.bfloat16),
                        stored[f"{prefix}.{i}.up_proj.weight"].to(torch.bfloat16),
                    ],
                    dim=0,
                )
                for i in range(4)
            ]
        )
        assert torch.equal(parameter, expected)


def check_native_fused_base_save_reload_has_no_adapters_or_requantization(base, tmp_path):
    from transformers import AutoModelForCausalLM

    source_sha = sha_file(base[0] / "model.safetensors")
    # Remove the untouched adapters, without merging or taking an optimizer step.
    model = loaded(base).base_model.unload()
    expected = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    assert not any("lora_" in name for name in expected)
    destination = tmp_path / "native-bf16"
    model.save_pretrained(destination, save_original_format=False, max_shard_size="64KB")
    assert (destination / "model.safetensors.index.json").is_file()
    assert not json.loads((destination / "config.json").read_text()).get("quantization_config")
    reread, info = AutoModelForCausalLM.from_pretrained(
        destination,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        device_map={"": "cpu"},
        attn_implementation="eager",
        output_loading_info=True,
    )
    assert not any(info.values())
    actual = reread.state_dict()
    assert actual.keys() == expected.keys()
    assert all(
        t.dtype == actual[n].dtype and torch.equal(t, actual[n]) for n, t in expected.items()
    )
    assert sha_file(base[0] / "model.safetensors") == source_sha


def check_adapter_optimizer_checkpoint_exact_next_update_without_base_save(base, tmp_path):
    random.seed(42)
    torch.manual_seed(42)
    model = loaded(base)
    optimizer = make_optimizer(model, lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda t: 1.0 / (t + 1))
    frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    step(model, optimizer, scheduler, torch.arange(3, 19).reshape(1, 16))
    folder = tmp_path / "checkpoint"
    receipt = save_checkpoint(
        folder,
        model,
        optimizer,
        scheduler,
        identity=base[1],
        plan_sha256="b" * 64,
        optimizer_step=1,
        next_batch=1,
        epoch=0,
        at_optimizer_boundary=True,
    )
    expected_random = random.random()
    next_ids = torch.randint(3, 100, (1, 16))
    step(model, optimizer, scheduler, next_ids)
    expected = {n: p.detach().clone() for n, p in adapter_parameters(model).items()}
    restored = loaded(base)
    resumed_optimizer = make_optimizer(restored, lr=1e-3)
    resumed_scheduler = torch.optim.lr_scheduler.LambdaLR(
        resumed_optimizer, lambda t: 1.0 / (t + 1)
    )
    reread = load_checkpoint(
        folder,
        restored,
        resumed_optimizer,
        resumed_scheduler,
        identity=base[1],
        plan_sha256="b" * 64,
    )
    assert reread == receipt and reread["optimizer_step"] == 1
    assert random.random() == expected_random
    assert torch.equal(torch.randint(3, 100, (1, 16)), next_ids)
    step(restored, resumed_optimizer, resumed_scheduler, next_ids)
    assert all(
        torch.equal(expected[n], p.detach()) for n, p in adapter_parameters(restored).items()
    )
    assert all(
        torch.equal(frozen[n], p.detach()) for n, p in restored.named_parameters() if n in frozen
    )
    saved = load_file(str(folder / "adapter_tensors.safetensors"))
    assert set(saved) == set(expected) and all(".lora_" in n for n in saved)
    assert sum(t.numel() for t in saved.values()) == 8320
    assert receipt["frozen_base_saved"] is False
    before = sha_file(folder / "COMPLETE.json")
    with unittest.TestCase().assertRaisesRegex(ValueError, "FileExistsError"):
        save_checkpoint(
            folder,
            restored,
            resumed_optimizer,
            resumed_scheduler,
            identity=base[1],
            plan_sha256="b" * 64,
            optimizer_step=2,
            next_batch=2,
            epoch=0,
            at_optimizer_boundary=True,
        )
    assert sha_file(folder / "COMPLETE.json") == before
    wrong_base = dataclasses.replace(base[1], weights_manifest_sha256="c" * 64)
    with unittest.TestCase().assertRaises(ValueError):
        load_checkpoint(
            folder,
            restored,
            resumed_optimizer,
            resumed_scheduler,
            identity=wrong_base,
            plan_sha256="b" * 64,
        )
    with unittest.TestCase().assertRaises(ValueError):
        load_checkpoint(
            folder,
            restored,
            resumed_optimizer,
            resumed_scheduler,
            identity=base[1],
            plan_sha256="c" * 64,
        )
    with (folder / "training_state.pt").open("ab") as stream:
        stream.write(b"tampered")
    with unittest.TestCase().assertRaises(ValueError):
        load_checkpoint(
            folder,
            restored,
            resumed_optimizer,
            resumed_scheduler,
            identity=base[1],
            plan_sha256="b" * 64,
        )


def check_invalid_base_and_optimizer_rejected(base, tmp_path):
    with unittest.TestCase().assertRaises(ValueError):
        BaseIdentity("zai-org/GLM-5.3-Flash", "3" * 40, "a" * 64, "b" * 64, "c" * 64)
    with unittest.TestCase().assertRaises(ValueError):
        load_lora_model(
            base[0], dataclasses.replace(base[1], config_sha256="f" * 64), meta_init=False
        )
    model = loaded(base)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
    with unittest.TestCase().assertRaisesRegex(ValueError, "optimizer"):
        save_checkpoint(
            tmp_path / "bad",
            model,
            optimizer,
            scheduler,
            identity=base[1],
            plan_sha256="b" * 64,
            optimizer_step=1,
            next_batch=1,
            epoch=0,
            at_optimizer_boundary=True,
        )
    assert not (tmp_path / "bad").exists()


@unittest.skipUnless(importlib.util.find_spec("skyrl") is not None, "requires pinned native image")
class TestGlm53LoraCompatibility(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="glm53-compat-unit-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.base = synthetic_base(self.root)

    def test_fp8_dequantization_and_meta_parity(self):
        check_fp8_cpu_dequantization_bf16_and_strict_router_meta_parity(self.base)

    def test_native_fused_base_save_reload_has_no_adapters_or_requantization(self):
        check_native_fused_base_save_reload_has_no_adapters_or_requantization(self.base, self.root)

    def test_checkpoint_exact_next_update(self):
        check_adapter_optimizer_checkpoint_exact_next_update_without_base_save(self.base, self.root)

    def test_fail_closed(self):
        check_invalid_base_and_optimizer_rejected(self.base, self.root)

    def test_per_block_arithmetic_and_hook_restoration(self):
        from transformers.integrations.finegrained_fp8 import Fp8Dequantize

        original = Fp8Dequantize._dequantize_one
        matrix = torch.linspace(-4, 4, 256 * 256).reshape(256, 256).to(torch.float8_e4m3fn)
        scales = torch.tensor([[0.75, 1.25], [2.0, 0.5]], dtype=torch.float32)
        expected = (
            (matrix.float().reshape(2, 128, 2, 128) * scales[:, None, :, None])
            .reshape(256, 256)
            .to(torch.bfloat16)
        )
        with bf16_fp8_conversion():
            converted = Fp8Dequantize(None)._dequantize_one(matrix, scales)
            assert converted.dtype == torch.bfloat16 and torch.equal(converted, expected)
        assert Fp8Dequantize._dequantize_one is original
        with self.assertRaisesRegex(ValueError, "synthetic rejection"), bf16_fp8_conversion():
            raise ValueError("synthetic rejection")
        assert Fp8Dequantize._dequantize_one is original

    def test_cpu_threads_are_scoped_and_arithmetic_is_identical(self):
        from transformers.integrations.finegrained_fp8 import Fp8Dequantize

        original = Fp8Dequantize._dequantize_one
        initial_threads = torch.get_num_threads()
        weight = torch.linspace(-4, 4, 576 * 256).reshape(576, 256).to(torch.float8_e4m3fn)
        scales = torch.linspace(0.5, 1.5, 10).reshape(5, 2)
        with bf16_fp8_conversion(cpu_threads=1):
            expected = Fp8Dequantize(None)._dequantize_one(weight, scales)
        for threads in (None, 1, 4, 16):
            with self.subTest(threads=threads):
                with bf16_fp8_conversion(cpu_threads=threads):
                    assert torch.get_num_threads() == (threads or initial_threads)
                    assert torch.equal(
                        Fp8Dequantize(None)._dequantize_one(weight, scales), expected
                    )
                assert torch.get_num_threads() == initial_threads
                with (
                    self.assertRaisesRegex(RuntimeError, "load failed"),
                    bf16_fp8_conversion(cpu_threads=threads),
                ):
                    raise RuntimeError("load failed")
                assert torch.get_num_threads() == initial_threads
                assert Fp8Dequantize._dequantize_one is original
        for invalid in (True, False, 0, -1, 17, 4.0, "4"):
            with (
                self.subTest(invalid=invalid),
                self.assertRaisesRegex(ValueError, "threads"),
                bf16_fp8_conversion(cpu_threads=invalid),
            ):
                self.fail("invalid thread setting entered the loader")
            assert torch.get_num_threads() == initial_threads
            assert Fp8Dequantize._dequantize_one is original

    def test_real_loader_uses_sixteen_threads_then_restores(self):
        from transformers import AutoModelForCausalLM

        initial_threads = torch.get_num_threads()
        native = AutoModelForCausalLM.from_pretrained
        observed = []

        def load(*args, **kwargs):
            observed.append(torch.get_num_threads())
            return native(*args, **kwargs)

        with patch.object(AutoModelForCausalLM, "from_pretrained", side_effect=load):
            loaded(self.base)
            loaded(self.base, meta=True)
        assert observed == [16]  # Meta-only ranks do not enter the CPU payload loader.
        assert torch.get_num_threads() == initial_threads
        with (
            patch.object(AutoModelForCausalLM, "from_pretrained", side_effect=RuntimeError),
            self.assertRaises(RuntimeError),
        ):
            loaded(self.base)
        assert torch.get_num_threads() == initial_threads

    def test_invalid_loader_inputs_fail_before_reading_weights(self):
        from transformers import AutoConfig, AutoModelForCausalLM

        from training import glm_runtime

        for kwargs in ({"rank": 0}, {"rank": True}, {"alpha": -1}, {"attention": "unknown"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                load_lora_model(*self.base, meta_init=False, **kwargs)
        for change in (
            {"model_type": "qwen3_5"},
            {"quantization_config": {"quant_method": "int8"}},
            {"quantization_config": {"quant_method": "fp8", "weight_block_size": [64, 128]}},
        ):
            cfg = AutoConfig.from_pretrained(self.base[0], local_files_only=True)
            for key, value in change.items():
                setattr(cfg, key, value)
            with (
                self.subTest(change=change),
                patch.object(AutoConfig, "from_pretrained", return_value=cfg),
                patch.object(AutoModelForCausalLM, "from_pretrained") as payload,
                self.assertRaises(ValueError),
            ):
                loaded(self.base)
            payload.assert_not_called()
        with (
            patch.object(glm_runtime, "FP8_INTEGRATION_SHA256", "0" * 64),
            self.assertRaisesRegex(ValueError, "unqualified"),
            bf16_fp8_conversion(),
        ):
            self.fail("a different native integration entered the loader")
        from transformers.integrations.finegrained_fp8 import Fp8Dequantize

        with self.assertRaisesRegex(ValueError, "E4M3"), bf16_fp8_conversion():
            Fp8Dequantize(None)._dequantize_one(
                torch.ones((2, 2), dtype=torch.int8), torch.ones((1, 1))
            )
        for lr in (0, -1, float("nan"), float("inf"), 0.011):
            with self.subTest(lr=lr), self.assertRaisesRegex(ValueError, "learning rate"):
                make_optimizer(None, lr=lr)
        for sha in (None, "short", "z" * 64):
            with self.subTest(sha=sha), self.assertRaisesRegex(ValueError, "SHA-256"):
                BaseIdentity("zai-org/GLM-5.3", "3" * 40, sha, "b" * 64, "c" * 64)

    def test_checkpoint_decode_rejects_structural_and_numeric_defects_before_restore(self):
        import hashlib

        from training.glm_runtime import canonical

        model = loaded(self.base)
        optimizer = make_optimizer(model, lr=1e-3)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
        step(model, optimizer, scheduler, torch.arange(3, 19).reshape(1, 16))
        source = self.root / "original"
        save_checkpoint(
            source,
            model,
            optimizer,
            scheduler,
            identity=self.base[1],
            plan_sha256="b" * 64,
            optimizer_step=1,
            next_batch=1,
            epoch=0,
            at_optimizer_boundary=True,
        )
        unchanged = {n: p.detach().clone() for n, p in adapter_parameters(model).items()}
        for case in (
            "receipt_digest",
            "extra_file",
            "payload_set",
            "tensor_names",
            "tensor_shape",
            "nonfinite",
            "state_keys",
            "rng_count",
            "cuda_rng_count",
        ):
            path = self.root / case
            shutil.copytree(source, path)
            receipt = json.loads((path / "COMPLETE.json").read_text())
            if case in {"tensor_names", "tensor_shape", "nonfinite"}:
                payload = path / "adapter_tensors.safetensors"
                tensors = load_file(str(payload))
                key = next(iter(tensors))
                if case == "tensor_names":
                    del tensors[key]
                elif case == "tensor_shape":
                    tensors[key] = tensors[key].flatten()
                else:
                    tensors[key].fill_(float("nan"))
                save_file(tensors, str(payload))
            elif case in {"state_keys", "rng_count", "cuda_rng_count"}:
                payload = path / "training_state.pt"
                state = torch.load(payload, map_location="cpu", weights_only=True)
                if case == "state_keys":
                    state["extra"] = 1
                elif case == "rng_count":
                    state["rng_by_rank"] = []
                else:
                    state["rng_by_rank"][0]["cuda"] = [torch.zeros(1, dtype=torch.uint8)]
                torch.save(state, payload)
            elif case == "extra_file":
                (path / "unexpected").touch()
            elif case == "payload_set":
                receipt["payloads"].pop("training_state.pt")
            for name in receipt["payloads"]:
                payload = path / name
                receipt["payloads"][name] = {
                    "bytes": payload.stat().st_size,
                    "sha256": sha_file(payload),
                }
            receipt.pop("receipt_sha256")
            receipt["receipt_sha256"] = (
                "0" * 64
                if case == "receipt_digest"
                else hashlib.sha256(canonical(receipt)).hexdigest()
            )
            (path / "COMPLETE.json").write_bytes(canonical(receipt))
            with self.subTest(case=case), self.assertRaises(ValueError):
                load_checkpoint(
                    path, model, optimizer, scheduler, identity=self.base[1], plan_sha256="b" * 64
                )
            assert all(torch.equal(unchanged[n], p) for n, p in adapter_parameters(model).items())

    def test_partial_fp8_blocks_follow_declared_block_size(self):
        from transformers.integrations.finegrained_fp8 import Fp8Dequantize

        # 258 is divisible by its three scale rows; deriving an 86-row block
        # would silently use the wrong scale even though native code accepts it.
        for rows, cols in ((576, 128), (258, 256), (129, 255), (64, 64)):
            with self.subTest(shape=(rows, cols)):
                weight = (
                    torch.linspace(-4, 4, rows * cols).reshape(rows, cols).to(torch.float8_e4m3fn)
                )
                sr, sc = (rows + 127) // 128, (cols + 127) // 128
                scale = torch.linspace(0.5, 1.5, sr * sc).reshape(sr, sc)
                expected = torch.empty((rows, cols), dtype=torch.bfloat16)
                for i in range(sr):
                    for j in range(sc):
                        section = (slice(i * 128, (i + 1) * 128), slice(j * 128, (j + 1) * 128))
                        expected[section] = (weight[section].float() * scale[i, j]).to(
                            torch.bfloat16
                        )
                with bf16_fp8_conversion():
                    actual = Fp8Dequantize(None)._dequantize_one(weight, scale)
                assert torch.equal(actual, expected)
        with self.assertRaisesRegex(ValueError, "scale grid"), bf16_fp8_conversion():
            Fp8Dequantize(None)._dequantize_one(
                torch.ones((256, 256), dtype=torch.bfloat16), torch.ones((4, 4))
            )

    def plan(self):
        from test_sft_runtime import plan

        from training import glm_runtime

        value = plan(self.root)
        identity = self.base[1]
        value["model"] = {
            "repo": identity.repository,
            "revision": identity.revision,
            "root": str(self.base[0]),
            "files": [
                {"path": "config.json", "sha256": identity.config_sha256},
                {"path": "tokenizer_config.json", "sha256": "a" * 64},
            ],
            "weight_manifest_sha256": "sha256:" + identity.weights_manifest_sha256,
            "tokenizer_manifest_sha256": "sha256:" + identity.tokenizer_manifest_sha256,
        }
        value["lora"] = {"rank": 4, "alpha": 8}
        value["glm_runtime_sha256"] = sha_file(Path(glm_runtime.__file__))
        return value

    def test_real_native_configuration_and_actor_selection(self):
        from skyrl.backends.skyrl_train.workers.fsdp import fsdp_worker

        from training.sft_runtime import (
            build_runtime_configs,
            validate_plan,
            validate_runtime_sources,
        )

        value = self.plan()
        validate_plan(value, check_files=False)
        validate_runtime_sources()
        cfg, native = build_runtime_configs(value)
        assert cfg.model.lora.rank == native.trainer.policy.model.lora.rank == 4
        assert not cfg.remove_microbatch_padding and not cfg.use_sequence_packing
        assert not native.trainer.flash_attn
        assert not native.trainer.policy.fsdp_config.cpu_offload
        assert not native.trainer.policy.optimizer_config.offload_after_step
        assert not native.trainer.placement.colocate_policy_ref
        assert not native.trainer.policy.inference_only_init
        original = fsdp_worker.PolicyWorker
        with self.assertRaisesRegex(ValueError, "synthetic rejection"), use_worker(value):
            assert fsdp_worker.PolicyWorker is not original
            raise ValueError("synthetic rejection")
        assert fsdp_worker.PolicyWorker is original

    def test_native_worker_glue_keeps_optimizer_and_adapter_checkpoint(self):
        from skyrl.backends.skyrl_train.workers.fsdp import fsdp_worker

        value = self.plan()
        cls = worker_class(value)
        worker = cls.__new__(cls)
        worker.cfg = SimpleNamespace(policy=SimpleNamespace(inference_only_init=False))
        original = fsdp_worker.HFModelWrapper

        def initialize(w, model_path, num_training_steps=None):
            # Execute the actual wrapper's nn.Module path. Only distributed
            # placement is omitted here; this is not a GPU/FSDP qualification.
            w.model = fsdp_worker.HFModelWrapper(model_path, meta_init=False)
            w.optimizer = torch.optim.AdamW(w.model.model.parameters(), lr=1e-3)
            w.scheduler = torch.optim.lr_scheduler.LambdaLR(w.optimizer, lambda _: 1)

        with patch.object(fsdp_worker.FSDPPolicyWorkerBase, "init_model", initialize):
            worker.init_model(value["model"]["root"], num_training_steps=6)
        assert fsdp_worker.HFModelWrapper is original
        assert set(map(id, worker.optimizer.param_groups[0]["params"])) == {
            id(p) for p in adapter_parameters(worker.model.model).values()
        }

        def update(optimizer, model, scheduler, name):
            assert optimizer is worker.optimizer and name == "actor"
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            return norm

        worker.strategy = SimpleNamespace(optimizer_step=update)
        inputs = torch.tensor([[1, 2, 3, 4, 5]])
        loss = worker.model.model(input_ids=inputs, labels=inputs).loss
        loss.backward()
        assert worker.optim_step() > 0
        directory = self.root / "global_step_1"
        directory.mkdir()
        saved = worker.save_checkpoint(str(directory / "policy"))
        assert saved["optimizer_step"] == saved["next_batch"] == 1
        assert saved["frozen_base_saved"] is False
        restored = worker.load_checkpoint(str(directory / "policy"))
        assert restored["receipt_sha256"] == saved["receipt_sha256"]
        assert worker._completed_optimizer_steps == 1
        with self.assertRaises(ValueError):
            worker.load_checkpoint(str(directory / "policy"), load_optimizer_states=False)
        with self.assertRaises(ValueError):
            worker.save_checkpoint(str(self.root / "global_step_2/policy"))
        with self.assertRaises(ValueError):
            worker.save_hf_model("unused")
        with (
            patch.object(fsdp_worker.FSDPPolicyWorkerBase, "init_model", side_effect=RuntimeError),
            self.assertRaises(RuntimeError),
        ):
            worker.init_model(value["model"]["root"])
        assert fsdp_worker.HFModelWrapper is original


if __name__ == "__main__":
    unittest.main()
