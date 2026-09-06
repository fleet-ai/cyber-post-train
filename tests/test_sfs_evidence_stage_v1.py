import json
from pathlib import Path

import pytest

from evals.fleet import self_hosted
from evals.fleet import sfs_evidence_stage_v1 as stage


def _receipt(**body):
    value = dict(body)
    value["receipt_sha256"] = self_hosted.digest_without(value, "receipt_sha256")
    return value


def _write(path: Path, value: dict) -> str:
    data = self_hosted.canonical_json(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return self_hosted.sha256(data)


def test_stage_validates_transitive_sfs_authority_and_publishes_regular_files(tmp_path):
    projected, sfs, out = tmp_path / "projected", tmp_path / "sfs", tmp_path / "out"
    predecessor = _receipt(status="ACCEPTED")
    release = _receipt(predecessor_acceptance_receipt_sha256=predecessor["receipt_sha256"])
    _write(projected / "parity.json", _receipt(status="PASSED_NON_SCORED"))
    _write(projected / "binding.json", _receipt(status="BOUND"))
    release_digest = _write(sfs / "release.json", release)
    _write(sfs / "accepted.json", predecessor)
    staged = stage.stage(projected_root=projected, sfs_release=sfs/"release.json", predecessor=sfs/"accepted.json", output_root=out, release_file_sha256=release_digest)
    assert staged["release_receipt_sha256"] == release["receipt_sha256"]
    assert all((out/name).is_file() and not (out/name).is_symlink() for name in ("parity.json","binding.json","release.json"))


def test_failed_source_cannot_leave_empty_target(tmp_path):
    source = tmp_path / "source.json"
    source.touch()
    target = tmp_path / "output" / "evidence.json"
    with pytest.raises(ValueError, match="empty"):
        value = stage.load_receipt(source)
        stage.write_regular_once(target, value)
    assert not target.exists()


def test_kubernetes_style_projected_symlink_is_copied_to_regular_file(tmp_path):
    projected_root = tmp_path / "projected"
    real = projected_root / "..data" / "projected.json"
    _write(real, _receipt(status="OK"))
    projected = projected_root / "projected.json"
    projected.symlink_to(Path("..data") / "projected.json")
    assert stage.load_receipt(projected, projected_root=projected_root)["status"] == "OK"


def test_projected_symlink_escape_is_rejected(tmp_path):
    outside = tmp_path / "outside.json"
    _write(outside, _receipt(status="OK"))
    projected_root = tmp_path / "projected"
    projected_root.mkdir()
    projected = projected_root / "projected.json"
    projected.symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        stage.load_receipt(projected, projected_root=projected_root)
