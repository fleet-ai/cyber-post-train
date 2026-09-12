"""Metadata-only protocol succession; no corpus or held-out content is opened."""

import copy
import json
import stat

import pytest

from cyber_post_train.jobs import digest
from training.corpus_protocol import publish, successor


def seal(value):
    value = {k: v for k, v in value.items() if k != "sha256"}
    return {**value, "sha256": "sha256:" + digest(value)}


@pytest.fixture
def metadata():
    manifest = seal(
        {
            "schema": "cyber_dense_sft_corpus_v1",
            "validation_mode": "task_outcomes_only",
            "fleet_dev_protocol_sha256": "sha256:" + "a" * 64,
            "files": {
                "train": {
                    "path": "deliberately-absent.parquet",
                    "rows": 2,
                    "supervised_tokens": 7,
                    "sha256": "sha256:" + "b" * 64,
                }
            },
            "source_sha256": "sha256:" + "c" * 64,
            "split_sha256": "sha256:" + "d" * 64,
            "tokenizer": {"repo": "synthetic", "revision": "immutable"},
            "source_selection": {
                "source_selection_sha256": "sha256:" + "e" * 64,
                "target_policy_sha256": "sha256:" + "f" * 64,
            },
            "builder_sha256": {"builder.py": "sha256:" + "1" * 64},
        }
    )
    protocol = seal({"schema": "cyber_fleet_dev_outcome_protocol_v2", "purpose": "synthetic"})
    return manifest, protocol


def test_successor_changes_only_protocol_and_seal_without_mutating_inputs(metadata):
    manifest, protocol = metadata
    original = copy.deepcopy(manifest)
    result = successor(
        manifest, protocol, source_sha256=manifest["sha256"], protocol_sha256=protocol["sha256"]
    )
    assert manifest == original
    assert {k for k in manifest if manifest[k] != result[k]} == {
        "fleet_dev_protocol_sha256",
        "sha256",
    }
    assert result == seal(result)
    assert result["fleet_dev_protocol_sha256"] == protocol["sha256"]


@pytest.mark.parametrize("defect", ["source_digest", "protocol_digest", "ce", "dev", "provenance"])
def test_invalid_or_unreviewed_source_fails(metadata, defect):
    manifest, protocol = metadata
    source_sha, protocol_sha = manifest["sha256"], protocol["sha256"]
    if defect == "source_digest":
        source_sha = "sha256:" + "0" * 64
    elif defect == "protocol_digest":
        protocol_sha = "sha256:" + "0" * 64
    else:
        if defect == "ce":
            manifest["validation_mode"] = "teacher_cross_entropy"
        elif defect == "dev":
            manifest["files"]["dev"] = dict(manifest["files"]["train"])
        else:
            manifest.pop("builder_sha256")
        manifest = seal(manifest)
        source_sha = manifest["sha256"]
    with pytest.raises(ValueError):
        successor(manifest, protocol, source_sha256=source_sha, protocol_sha256=protocol_sha)


def test_already_bound_manifest_is_not_republished(metadata):
    manifest, protocol = metadata
    manifest["fleet_dev_protocol_sha256"] = protocol["sha256"]
    manifest = seal(manifest)
    with pytest.raises(ValueError, match="already bound"):
        successor(
            manifest, protocol, source_sha256=manifest["sha256"], protocol_sha256=protocol["sha256"]
        )


def test_publication_is_private_create_once_and_metadata_only(tmp_path, metadata):
    manifest, protocol = metadata
    source, protocol_path = tmp_path / "source.json", tmp_path / "protocol.json"
    source.write_text(json.dumps(manifest))
    protocol_path.write_text(json.dumps(protocol))
    output = tmp_path / "new"
    kwargs = {"source_sha256": manifest["sha256"], "protocol_sha256": protocol["sha256"]}
    proof = publish(source, protocol_path, output, **kwargs)
    assert proof == seal(proof)
    assert proof["parquet_or_source_records_read"] is False
    assert proof["remote_operations_performed"] is False
    assert proof["launch_authorized"] is False
    assert proof["preserved"]["builder_sha256"] == manifest["builder_sha256"]
    assert stat.S_IMODE(output.stat().st_mode) == 0o700
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    assert set(before) == {"manifest.json", "SUCCESSOR.json"}
    assert all(stat.S_IMODE(p.stat().st_mode) == 0o600 for p in output.iterdir())
    with pytest.raises(FileExistsError):
        publish(source, protocol_path, output, **kwargs)
    assert before == {p.name: p.read_bytes() for p in output.iterdir()}
    link = tmp_path / "linked-source.json"
    link.symlink_to(source)
    with pytest.raises(ValueError, match="symlinks"):
        publish(link, protocol_path, tmp_path / "other", **kwargs)
