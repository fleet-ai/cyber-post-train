import io
import tarfile
from pathlib import Path

import pytest

from cyber_post_train.source_bundle import (
    canonical_source_commit_bytes,
    verify_source_archive_commit,
    verify_source_commit_file,
)

COMMIT = "a108edff2062558359cc72ebdfaf5d40cadeb333"


def archive(path: Path, value: bytes, *, duplicate: bool = False) -> None:
    with tarfile.open(path, "w:gz") as output:
        for _ in range(2 if duplicate else 1):
            member = tarfile.TarInfo("SOURCE_COMMIT")
            member.size = len(value)
            output.addfile(member, io.BytesIO(value))


def test_source_commit_file_and_archive_accept_exact_canonical_bytes(tmp_path):
    value = canonical_source_commit_bytes(COMMIT)
    assert value == (COMMIT + "\n").encode()
    source_file = tmp_path / "SOURCE_COMMIT"
    source_file.write_bytes(value)
    source_archive = tmp_path / "source.tgz"
    archive(source_archive, value)

    assert verify_source_commit_file(source_file, COMMIT)["source_commit"] == COMMIT
    receipt = verify_source_archive_commit(source_archive, COMMIT)
    assert receipt["source_commit"] == COMMIT
    assert len(receipt["source_archive_sha256"]) == 64


@pytest.mark.parametrize(
    "value",
    [COMMIT.encode(), (COMMIT + "\n\n").encode(), ("0" * 40 + "\n").encode()],
)
def test_source_bundle_rejects_noncanonical_or_wrong_commit_bytes(tmp_path, value):
    source_file = tmp_path / "SOURCE_COMMIT"
    source_file.write_bytes(value)
    source_archive = tmp_path / "source.tgz"
    archive(source_archive, value)

    with pytest.raises(ValueError, match="exact canonical text"):
        verify_source_commit_file(source_file, COMMIT)
    with pytest.raises(ValueError, match="exact canonical text"):
        verify_source_archive_commit(source_archive, COMMIT)


def test_source_bundle_rejects_duplicate_commit_entry(tmp_path):
    source_archive = tmp_path / "source.tgz"
    archive(source_archive, canonical_source_commit_bytes(COMMIT), duplicate=True)
    with pytest.raises(ValueError, match="one regular SOURCE_COMMIT"):
        verify_source_archive_commit(source_archive, COMMIT)
