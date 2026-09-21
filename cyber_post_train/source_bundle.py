"""Small exact-byte boundary for self-contained source bundles."""

from __future__ import annotations

import hashlib
import re
import tarfile
from pathlib import Path

_COMMIT = re.compile(r"[0-9a-f]{40}")


def canonical_source_commit_bytes(commit: str) -> bytes:
    """Return the one accepted on-disk encoding of a Git commit."""
    if not isinstance(commit, str) or _COMMIT.fullmatch(commit) is None:
        raise ValueError("source commit must be one lowercase 40-hex Git commit")
    return (commit + "\n").encode()


def verify_source_commit_file(path: Path, expected_commit: str) -> dict:
    """Require one canonical newline-terminated commit file."""
    expected = canonical_source_commit_bytes(expected_commit)
    actual = path.read_bytes()
    if actual != expected:
        raise ValueError("source commit file is not the exact canonical text")
    return {
        "source_commit": expected_commit,
        "source_commit_file_sha256": hashlib.sha256(actual).hexdigest(),
    }


def verify_source_archive_commit(path: Path, expected_commit: str) -> dict:
    """Verify the exact SOURCE_COMMIT bytes inside a gzip/tar source bundle."""
    expected = canonical_source_commit_bytes(expected_commit)
    with tarfile.open(path, "r:gz") as archive:
        matches = [member for member in archive.getmembers() if member.name == "SOURCE_COMMIT"]
        if len(matches) != 1 or not matches[0].isfile():
            raise ValueError("source archive must contain one regular SOURCE_COMMIT file")
        stream = archive.extractfile(matches[0])
        if stream is None:
            raise ValueError("source archive SOURCE_COMMIT is unreadable")
        actual = stream.read()
    if actual != expected:
        raise ValueError("source archive commit is not the exact canonical text")
    return {
        "source_commit": expected_commit,
        "source_commit_file_sha256": hashlib.sha256(actual).hexdigest(),
        "source_archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
