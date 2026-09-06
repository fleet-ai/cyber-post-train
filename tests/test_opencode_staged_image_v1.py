from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from evals.fleet import opencode_staged_image_v1 as staged
from evals.fleet import self_hosted


def _write_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    archive = tmp_path / "image.tar.gz"
    archive.write_bytes(b"exact-image")
    monkeypatch.setattr(staged, "ARCHIVE_PATH", archive)
    monkeypatch.setattr(staged, "ARCHIVE_BYTES", archive.stat().st_size)
    monkeypatch.setattr(
        staged,
        "ARCHIVE_SHA256",
        "sha256:" + hashlib.sha256(archive.read_bytes()).hexdigest(),
    )
    receipt = tmp_path / "STAGED.json"
    monkeypatch.setattr(staged, "RECEIPT_PATH", receipt)
    value = {
        "schema_version": staged.SCHEMA,
        "status": "STAGED_EXACT",
        "archive_path": str(archive),
        "archive_sha256": staged.ARCHIVE_SHA256,
        "source_oci_index_digest": staged.OCI_INDEX_DIGEST,
        "runtime_image_id": staged.RUNTIME_IMAGE_ID,
        "platform": {"architecture": "amd64", "os": "linux"},
        "runtime": {
            "opencode_version": "1.18.27",
            "user": "node",
            "working_dir": "/workspace",
        },
        "credentials_included": False,
        "prompts_traces_flags_or_scores_included": False,
    }
    value["receipt_sha256"] = self_hosted.digest_without(value, "receipt_sha256")
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    receipt.write_text(payload)
    monkeypatch.setattr(staged, "RECEIPT_SHA256", value["receipt_sha256"])
    monkeypatch.setattr(
        staged,
        "RECEIPT_FILE_SHA256",
        "sha256:" + hashlib.sha256(payload.encode()).hexdigest(),
    )
    return receipt, archive


def test_validate_binds_receipt_and_archive_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt, archive = _write_fixture(tmp_path, monkeypatch)
    value = staged.validate(receipt, archive)
    assert value["runtime_image_id"] == staged.RUNTIME_IMAGE_ID
    archive.write_bytes(b"drift")
    with pytest.raises(ValueError, match="artifact drifted"):
        staged.validate(receipt, archive)


def test_validate_rejects_symlinked_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt, archive = _write_fixture(tmp_path, monkeypatch)
    link = tmp_path / "linked.json"
    link.symlink_to(receipt)
    with pytest.raises(ValueError, match="regular files"):
        staged.validate(link, archive)
