import json
from pathlib import Path

import pytest

from training.io import digest_json
from training.post_sft_artifacts import (
    full_file_manifest,
    inspect_hf_export,
    structural_manifest,
    structural_tsv_sha256,
)


def test_manifests_are_sorted_relative_and_content_sensitive(tmp_path: Path):
    (tmp_path / "nested").mkdir()
    (tmp_path / "z").write_bytes(b"z")
    (tmp_path / "nested" / "a").write_bytes(b"alpha")
    structural = structural_manifest(tmp_path)
    full = full_file_manifest(tmp_path)
    assert [row["path"] for row in structural["files"]] == ["nested/a", "z"]
    assert [row["path"] for row in full["files"]] == ["nested/a", "z"]
    assert full["total_bytes"] == 6
    assert structural_tsv_sha256(tmp_path).startswith("sha256:")
    before = full["manifest_sha256"]
    (tmp_path / "z").write_bytes(b"changed")
    assert full_file_manifest(tmp_path)["manifest_sha256"] != before


def test_hf_inspection_binds_shards_tokenizer_config_and_parameter_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    class Slice:
        def get_shape(self):
            return [2, 3]

        def get_dtype(self):
            return "BF16"

    class SafeFile:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def keys(self):
            return ["weight"]

        def __iter__(self):
            return iter(self.keys())

        def get_slice(self, _key):
            return Slice()

    import safetensors

    monkeypatch.setattr(safetensors, "safe_open", lambda *_a, **_k: SafeFile())
    (tmp_path / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": "model-00001-of-00001.safetensors"}})
    )
    rows = []
    for name in (
        "chat_template.jinja",
        "merges.txt",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    ):
        (tmp_path / name).write_text(name)
        from training.post_sft_artifacts import sha256_file

        rows.append({"path": name, "sha256": sha256_file(tmp_path / name).removeprefix("sha256:")})
    (tmp_path / "config.json").write_text("config")
    result = inspect_hf_export(
        tmp_path,
        expected_tokenizer_manifest_sha256=digest_json(sorted(rows, key=lambda row: row["path"])),
        expected_chat_template_sha256=next(
            "sha256:" + row["sha256"] for row in rows if row["path"] == "chat_template.jinja"
        ),
        expected_config_sha256=(
            "sha256:b79606fb3afea5bd1609ed40b622142f1c98125abcfe89a76a661b0e8e343910"
        ),
        expected_parameter_count=6,
    )
    assert result["shard_count"] == 1
    assert result["parameter_count"] == 6

    with pytest.raises(ValueError, match="parameter count"):
        inspect_hf_export(
            tmp_path,
            expected_tokenizer_manifest_sha256=result["tokenizer_manifest_sha256"],
            expected_chat_template_sha256=result["chat_template_sha256"],
            expected_config_sha256=result["config_sha256"],
            expected_parameter_count=7,
        )
