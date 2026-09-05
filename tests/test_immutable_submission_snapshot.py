from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from evals.fleet import immutable_submission_snapshot as snapshot


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def _repo(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init")
    _git(root, "config", "user.email", "test@example.invalid")
    _git(root, "config", "user.name", "Test")
    (root / "source.txt").write_text("sealed\n")
    _git(root, "add", "source.txt")
    _git(root, "commit", "-m", "sealed")
    return root, _git(root, "rev-parse", "HEAD")


def test_snapshot_rejects_dirty_repository(tmp_path: Path) -> None:
    root, commit = _repo(tmp_path)
    (root / "source.txt").write_text("dirty\n")
    with pytest.raises(RuntimeError, match="dirty"):
        snapshot.assert_stable(root, commit)


def test_snapshot_rejects_head_change(tmp_path: Path) -> None:
    root, commit = _repo(tmp_path)
    (root / "second.txt").write_text("new\n")
    _git(root, "add", "second.txt")
    _git(root, "commit", "-m", "second")
    with pytest.raises(RuntimeError, match="HEAD changed"):
        snapshot.assert_stable(root, commit)


def test_snapshot_rejects_materialized_byte_drift(tmp_path: Path) -> None:
    root, commit = _repo(tmp_path)
    destination = tmp_path / "snapshot"
    snapshot.materialize(root, commit, destination)
    snapshot.verify_paths(root, commit, destination, {"source.txt"})
    (destination / "source.txt").write_text("drift\n")
    with pytest.raises(RuntimeError, match="differs"):
        snapshot.verify_paths(root, commit, destination, {"source.txt"})
