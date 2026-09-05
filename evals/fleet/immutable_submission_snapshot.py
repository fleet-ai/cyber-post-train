"""Fail-closed Git snapshot helpers for create-once submission bundles."""

from __future__ import annotations

import argparse
import io
import re
import subprocess
import tarfile
from pathlib import Path

COMMIT_RE = re.compile(r"[0-9a-f]{40}")


def _git(root: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, stdout=subprocess.PIPE
    ).stdout


def assert_stable(root: Path, expected_head: str) -> None:
    if not COMMIT_RE.fullmatch(expected_head):
        raise ValueError("expected HEAD must be a full Git commit")
    observed = _git(root, "rev-parse", "HEAD").decode().strip()
    if observed != expected_head:
        raise RuntimeError("repository HEAD changed during submission")
    if _git(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("repository is dirty during submission")


def materialize(root: Path, commit: str, destination: Path) -> None:
    if not COMMIT_RE.fullmatch(commit):
        raise ValueError("snapshot commit must be a full Git commit")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("snapshot destination already exists")
    archive = _git(root, "archive", "--format=tar", commit)
    destination.mkdir(mode=0o700)
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as bundle:
        for member in bundle.getmembers():
            target = Path(member.name)
            if target.is_absolute() or ".." in target.parts or member.issym() or member.islnk():
                raise RuntimeError("unsafe path in Git snapshot archive")
        bundle.extractall(destination, filter="data")


def verify_paths(root: Path, commit: str, snapshot: Path, paths: set[str]) -> None:
    if not paths:
        raise ValueError("snapshot verification requires source paths")
    for path in sorted(paths):
        if Path(path).is_absolute() or ".." in Path(path).parts:
            raise ValueError("snapshot source path is unsafe")
        expected = _git(root, "show", f"{commit}:{path}")
        candidate = snapshot / path
        if candidate.is_symlink() or not candidate.is_file():
            raise RuntimeError(f"snapshot source is missing or unsafe: {path}")
        if candidate.read_bytes() != expected:
            raise RuntimeError(f"snapshot source differs from package commit: {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    stable = subparsers.add_parser("assert-stable")
    stable.add_argument("--repo", type=Path, required=True)
    stable.add_argument("--expected-head", required=True)
    snapshot = subparsers.add_parser("materialize")
    snapshot.add_argument("--repo", type=Path, required=True)
    snapshot.add_argument("--commit", required=True)
    snapshot.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "assert-stable":
        assert_stable(args.repo, args.expected_head)
    else:
        materialize(args.repo, args.commit, args.destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
