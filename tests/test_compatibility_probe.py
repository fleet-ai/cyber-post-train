import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import training.compatibility_probe as probe
from training.compatibility_probe import (
    CHAT_TEMPLATE_SHA256,
    MODEL_REVISION,
    VERIFIED_SHARDS,
    WEIGHTS_MANIFEST_SHA256,
    validate_checkpoint_lock,
    validate_megatron_bridge_config,
)


def checkpoint_dir(tmp_path: Path) -> Path:
    template = tmp_path / "chat_template.jinja"
    template.write_bytes(b"probe")
    digest = "sha256:" + hashlib.sha256(b"probe").hexdigest()
    assert digest != CHAT_TEMPLATE_SHA256
    lock = {
        "schema": "cyber_post_train_checkpoint_lock_v1",
        "repo": "zai-org/GLM-5.2",
        "revision": MODEL_REVISION,
        "weights_manifest_sha256": WEIGHTS_MANIFEST_SHA256,
        "verified_shards": VERIFIED_SHARDS,
        "verified_bytes": 1,
    }
    (tmp_path / ".cyber-post-train-lock.json").write_text(json.dumps(lock))
    return tmp_path


def test_lock_validation_fails_on_template_bytes(tmp_path: Path):
    root = checkpoint_dir(tmp_path)
    with pytest.raises(ValueError, match="chat_template"):
        validate_checkpoint_lock(root)


def test_lock_validation_fails_on_revision_before_reading_template(tmp_path: Path):
    root = checkpoint_dir(tmp_path)
    lock_path = root / ".cyber-post-train-lock.json"
    lock = json.loads(lock_path.read_text())
    lock["revision"] = "moving-main"
    lock_path.write_text(json.dumps(lock))
    with pytest.raises(ValueError, match="checkpoint lock mismatch"):
        validate_checkpoint_lock(root)


def test_lock_validation_accepts_exact_identity(tmp_path: Path, monkeypatch):
    root = checkpoint_dir(tmp_path)
    monkeypatch.setattr(
        probe,
        "CHAT_TEMPLATE_SHA256",
        "sha256:" + hashlib.sha256(b"probe").hexdigest(),
    )
    assert validate_checkpoint_lock(root)["verified_bytes"] == 1


def test_megatron_bridge_config_preserves_moe_topology(tmp_path: Path, monkeypatch):
    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            assert args == (str(tmp_path),)
            assert kwargs["local_files_only"] is True
            assert kwargs["qk_rope_head_dim"] == 64
            return SimpleNamespace(model_type="glm_moe_dsa", qk_rope_head_dim=64)

    class FakeBridge:
        def to_megatron_provider(self, *, load_weights):
            assert load_weights is False
            return SimpleNamespace(
                num_layers=78,
                hidden_size=6144,
                num_attention_heads=64,
                num_moe_experts=256,
                moe_ffn_hidden_size=2048,
            )

    class FakeAutoBridge:
        @staticmethod
        def from_hf_config(config):
            assert config.model_type == "glm_moe_dsa"
            return FakeBridge()

    modules = {
        "transformers": SimpleNamespace(AutoConfig=FakeAutoConfig),
        "megatron.bridge": SimpleNamespace(AutoBridge=FakeAutoBridge),
    }
    monkeypatch.setattr(probe.importlib, "import_module", modules.__getitem__)
    result = validate_megatron_bridge_config(tmp_path)
    assert result["hf_model_type"] == "glm_moe_dsa"
    assert result["provider_topology"]["num_moe_experts"] == 256
