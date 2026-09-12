import json
import os
from pathlib import Path

import pytest

from training.corpus_acceptance import accept
from training.io import digest_json, file_sha256
from training.study_data import seal


def _write_private(path: Path, payload: bytes) -> None:
    path.write_bytes(payload)
    path.chmod(0o600)


def _fixture(tmp_path: Path) -> tuple[Path, dict]:
    root = tmp_path / "corpus"
    root.mkdir(mode=0o700, parents=True)
    train = root / "train.parquet"
    # Deliberately not valid Parquet: acceptance hashes opaque bytes and never
    # decodes token arrays. Semantic construction belongs to the sealed builder.
    _write_private(train, b"opaque-private-training-bytes")
    study = seal({"schema": "test_study_split"})
    split = seal({"schema": "test_training_split"})
    target_policy = "sha256:" + "4" * 64
    selection = seal(
        {
            "schema": "cyber_sft_source_selection_v1",
            "status": "ready",
            "blockers": [],
            "study_split_sha256": study["sha256"],
            "training_split_sha256": split["sha256"],
            "target_policy_sha256": target_policy,
            "selected_episode_ids": ["episode-1", "episode-2"],
            "totals": {"episodes": 2, "supervised_tokens": 11},
        }
    )
    builder = {"dense.py": "sha256:" + "5" * 64}
    manifest = seal(
        {
            "schema": "cyber_dense_sft_corpus_v1",
            "files": {
                "train": {
                    "path": "train.parquet",
                    "format": "pretokenized_assistant_segments_v1",
                    "sha256": file_sha256(train),
                    "source_sessions": 2,
                    "rows": 3,
                    "supervised_tokens": 11,
                }
            },
            "dev_windows": 0,
            "validation_mode": "task_outcomes_only",
            "max_length": 16,
            "context_tokens": 4,
            "tokenizer": {
                "repo": "example/model",
                "revision": "a" * 40,
                "chat_template_sha256": "1" * 64,
                "backend_sha256": "2" * 64,
            },
            "builder_sha256": builder,
            "source_selection": {
                "source_selection_sha256": selection["sha256"],
                "study_split_sha256": study["sha256"],
                "target_policy_sha256": target_policy,
                "selected_episode_count": 2,
            },
        }
    )
    for name, value in (
        ("manifest.json", manifest),
        ("source-selection.json", selection),
        ("split.json", split),
        ("study-split.json", study),
    ):
        _write_private(root / name, (os.linesep.join([json.dumps(value), ""])).encode())
    expected = {
        "manifest_sha256": manifest["sha256"],
        "train_sha256": file_sha256(train),
        "source_selection_sha256": selection["sha256"],
        "study_split_sha256": study["sha256"],
        "training_split_sha256": split["sha256"],
        "target_policy_sha256": target_policy,
        "episodes": 2,
        "segments": 3,
        "supervised_tokens": 11,
        "model_repo": "example/model",
        "model_revision": "a" * 40,
        "chat_template_sha256": "sha256:" + "1" * 64,
        "tokenizer_backend_sha256": "sha256:" + "2" * 64,
        "max_length": 16,
        "context_tokens": 4,
        "builder_components_sha256": digest_json(builder),
        "frozen_materialization_runtime_sha256": "sha256:" + "3" * 64,
    }
    return root, expected


def test_accepts_exact_private_train_only_corpus_without_decoding_payload(tmp_path):
    root, expected = _fixture(tmp_path)
    receipt = accept(root, expected)
    assert receipt["status"] == "accepted"
    assert receipt["read_policy"] == "opaque_sha256_only_no_parquet_or_token_decoding"
    assert receipt["train_sha256"] == expected["train_sha256"]
    assert receipt["counts"] == {"episodes": 2, "segments": 3, "supervised_tokens": 11}


def test_rejects_payload_tampering(tmp_path):
    root, expected = _fixture(tmp_path)
    _write_private(root / "train.parquet", b"different-bytes")
    with pytest.raises(ValueError, match="payload identity"):
        accept(root, expected)


def test_rejects_symlink_or_nonprivate_member(tmp_path):
    root, expected = _fixture(tmp_path)
    (root / "train.parquet").unlink()
    (root / "train.parquet").symlink_to(root / "manifest.json")
    with pytest.raises(ValueError, match="regular non-symlink"):
        accept(root, expected)

    root, expected = _fixture(tmp_path / "second")
    (root / "train.parquet").chmod(0o644)
    with pytest.raises(ValueError, match="mode 0600"):
        accept(root, expected)


def test_rejects_path_escape_and_extra_files(tmp_path):
    root, expected = _fixture(tmp_path)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["train"]["path"] = "../train.parquet"
    manifest = seal({key: row for key, row in manifest.items() if key != "sha256"})
    _write_private(manifest_path, (json.dumps(manifest) + "\n").encode())
    expected["manifest_sha256"] = manifest["sha256"]
    with pytest.raises(ValueError, match="frozen interface"):
        accept(root, expected)

    root, expected = _fixture(tmp_path / "second")
    _write_private(root / "extra", b"unexpected")
    with pytest.raises(ValueError, match="inventory"):
        accept(root, expected)
