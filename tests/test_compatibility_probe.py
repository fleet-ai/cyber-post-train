import hashlib
import json
from pathlib import Path

import pytest

import training.compatibility_probe as probe
from training.compatibility_probe import (
    CHAT_TEMPLATE_SHA256,
    MODEL_REVISION,
    VERIFIED_SHARDS,
    WEIGHTS_MANIFEST_SHA256,
    validate_checkpoint_lock,
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
