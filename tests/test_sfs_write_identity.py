import hashlib
import json
import os
from pathlib import Path

import pytest

from cyber_post_train.sfs_write_identity import (
    validate_owned_output_binding,
    verify_owned_output_runtime,
)


def test_binding_requires_one_hidden_direct_child_of_owned_job_tree() -> None:
    owned = "/mnt/sfs/jobs/chris-q38-study-corpora-v1/launch-controls"
    output = owned + "/.preflight-control-step60-a21cbe7c"
    validate_owned_output_binding(owned, output)
    for bad in (
        "/mnt/sfs/jobs/.preflight-control-step60-a21cbe7c",
        owned + "/nested/.preflight-control-step60-a21cbe7c",
        owned + "/prepared-step60",
        owned + "/../launch-controls/.preflight-control-step60-a21cbe7c",
    ):
        with pytest.raises(ValueError):
            validate_owned_output_binding(owned, bad)


def test_runtime_rechecks_non_root_owner_mode_and_absence(tmp_path: Path, monkeypatch) -> None:
    # Keep the real lexical contract while redirecting only the lstat/exists calls
    # to this synthetic directory through a small Path proxy.
    owned_path = "/mnt/sfs/jobs/chris-q38-study-corpora-v1/launch-controls"
    output_path = owned_path + "/.preflight-control-step60-a21cbe7c"
    root = tmp_path / "owned"
    root.mkdir(mode=0o700)
    monkeypatch.setattr(os, "geteuid", lambda: root.stat().st_uid)
    monkeypatch.setattr(os, "getegid", lambda: root.stat().st_gid)

    # The runtime helper deliberately accepts Path-like objects.  Override only
    # their string form so lexical validation sees the production binding.
    class Proxy:
        def __init__(self, path: Path, rendered: str):
            self.path, self.rendered = path, rendered

        def __str__(self):
            return self.rendered

        def lstat(self):
            return self.path.lstat()

        def is_symlink(self):
            return self.path.is_symlink()

        def exists(self):
            return self.path.exists()

    receipt = verify_owned_output_runtime(
        Proxy(root, owned_path),
        Proxy(tmp_path / "absent", output_path),
        writable_mount=Proxy(root, "/controls"),
        uid=root.stat().st_uid,
        gid=root.stat().st_gid,
    )
    assert receipt["owned_root"] == owned_path
    assert receipt["output_root"] == output_path
    assert receipt["writable_mount"] == "/controls"


def test_runtime_rejects_wrong_identity_or_existing_output(tmp_path: Path, monkeypatch) -> None:
    owned_path = "/mnt/sfs/jobs/chris-q38-study-corpora-v1/launch-controls"
    output_path = owned_path + "/.preflight-control-step60-a21cbe7c"
    root = tmp_path / "owned"
    root.mkdir(mode=0o700)
    output = tmp_path / "output"

    class Proxy:
        def __init__(self, path: Path, rendered: str):
            self.path, self.rendered = path, rendered

        def __str__(self):
            return self.rendered

        def lstat(self):
            return self.path.lstat()

        def is_symlink(self):
            return self.path.is_symlink()

        def exists(self):
            return self.path.exists()

    owned = Proxy(root, owned_path)
    target = Proxy(output, output_path)
    monkeypatch.setattr(os, "geteuid", lambda: root.stat().st_uid + 1)
    monkeypatch.setattr(os, "getegid", lambda: root.stat().st_gid)
    with pytest.raises(ValueError, match="pinned trainer identity"):
        verify_owned_output_runtime(owned, target, uid=root.stat().st_uid, gid=root.stat().st_gid)
    monkeypatch.setattr(os, "geteuid", lambda: root.stat().st_uid)
    other = tmp_path / "other"
    other.mkdir(mode=0o700)
    with pytest.raises(ValueError, match="exact reviewed control parent"):
        verify_owned_output_runtime(
            owned,
            target,
            writable_mount=Proxy(other, "/controls"),
            uid=root.stat().st_uid,
            gid=root.stat().st_gid,
        )
    output.mkdir()
    with pytest.raises(FileExistsError):
        verify_owned_output_runtime(owned, target, uid=root.stat().st_uid, gid=root.stat().st_gid)


def test_p4_evidence_receipt_is_self_consistent() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "docs/evidence/qwen38-lora-step60-cpu-preflight-p4-sfs-write-rejection-20260921.json"
    )
    value = json.loads(path.read_text())
    expected = value.pop("receipt_sha256")
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(encoded).hexdigest() == expected
    assert value["pod"]["gpus"] == 0
    assert value["failure_boundary"]["sfs_write_operations_succeeded"] == 0
    assert value["repair_binding"]["path_isolation"]["full_sfs_mount_read_only"] is True
