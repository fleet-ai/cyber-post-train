import hashlib
import tarfile
from pathlib import Path

import pytest

from evals.fleet import opencode_image_context as context


def _root(tmp_path: Path) -> Path:
    dockerfile = tmp_path / context.DOCKERFILE
    dockerfile.parent.mkdir(parents=True)
    dockerfile.write_text("FROM scratch\nCOPY opencode-linux-x64.tar.gz /asset\n")
    return tmp_path


def test_context_archive_is_deterministic_and_minimal(tmp_path: Path) -> None:
    root = _root(tmp_path)
    asset = tmp_path / "release.tar.gz"
    asset.write_bytes(b"exact synthetic release")
    expected = hashlib.sha256(asset.read_bytes()).hexdigest()
    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"
    one = context.prepare_context(
        root=root, asset=asset, output=first, expected_asset_sha256=expected
    )
    two = context.prepare_context(
        root=root, asset=asset, output=second, expected_asset_sha256=expected
    )
    assert first.read_bytes() == second.read_bytes()
    assert one == two
    with tarfile.open(first, "r:gz") as archive:
        assert archive.getnames() == ["Dockerfile", context.ASSET_NAME]
        for member in archive.getmembers():
            assert (member.uid, member.gid, member.mtime, member.mode) == (0, 0, 0, 0o644)


def test_context_rejects_tamper_and_replace(tmp_path: Path) -> None:
    root = _root(tmp_path)
    asset = tmp_path / "release.tar.gz"
    asset.write_bytes(b"wrong")
    with pytest.raises(context.ContextError, match="digest mismatch"):
        context.prepare_context(root=root, asset=asset, output=tmp_path / "out.tar.gz")
    output = tmp_path / "existing.tar.gz"
    output.write_bytes(b"preserve")
    expected = hashlib.sha256(asset.read_bytes()).hexdigest()
    with pytest.raises(context.ContextError, match="already exists"):
        context.prepare_context(
            root=root, asset=asset, output=output, expected_asset_sha256=expected
        )
    assert output.read_bytes() == b"preserve"


def test_build_request_is_exact_and_create_once() -> None:
    request = context.build_request(
        context_id="a" * 32,
        context_sha256="b" * 64,
        source_commit="c" * 40,
    )
    assert request["ecr_repository"] == "fleet/cyber-post-train-opencode-runtime"
    assert request["tag"] == "opencode11827-" + "b" * 16
    assert request["platform"] == "linux/amd64"
    assert request["no_cache"] is True
    assert request["pull"] is False
    context.validate_request(request, context_sha256="b" * 64, source_commit="c" * 40)
    request["ecr_repository"] = "fleet/miles-trainer"
    with pytest.raises(context.ContextError, match="differs"):
        context.validate_request(request, context_sha256="b" * 64, source_commit="c" * 40)


def test_real_dockerfile_has_no_build_network_or_mutable_packages() -> None:
    dockerfile = (Path(__file__).parents[1] / context.DOCKERFILE).read_text()
    assert f"FROM {context.BASE_IMAGE}" in dockerfile
    assert "COPY opencode-linux-x64.tar.gz" in dockerfile
    assert "apt-get" not in dockerfile
    assert "curl " not in dockerfile
    assert "wget " not in dockerfile
    assert dockerfile.count("RUN --network=none") == 2
    assert "cyber.opencode.release-sha256" in dockerfile
