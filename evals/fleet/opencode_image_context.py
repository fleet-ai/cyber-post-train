"""Prepare the exact, network-free OpenCode 1.18.27 dev build context."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import tarfile
import tempfile
from pathlib import Path
from typing import Any

from training.io import digest_json, file_sha256

BASE_IMAGE = (
    "docker.io/library/node@sha256:4d676821dff059fd00d277ee4261ef34ea712317fed0737c03941481b5760c96"
)
OPENCODE_VERSION = "1.18.27"
OPENCODE_SHA256 = "4af5494f9433f59db8c1e344198f0ee72a50c06ec009fb4a8aeab4c2d4abd702"
ECR_REGISTRY = "661864827319.dkr.ecr.us-east-1.amazonaws.com"
ECR_REPOSITORY = "fleet/cyber-post-train-opencode-runtime"
PLATFORM = "linux/amd64"
DOCKERFILE = Path("evals/fleet/images/opencode11827/Dockerfile")
ASSET_NAME = "opencode-linux-x64.tar.gz"
CONTEXT_SCHEMA = "cyber_opencode_agent_build_context_v1"


class ContextError(ValueError):
    """The build context or immutable request is not exact."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _tar_member(name: str, data: bytes) -> tuple[tarfile.TarInfo, bytes]:
    member = tarfile.TarInfo(name)
    member.size = len(data)
    member.mode = 0o644
    member.mtime = 0
    member.uid = member.gid = 0
    member.uname = member.gname = ""
    return member, data


def prepare_context(
    *,
    root: Path,
    asset: Path,
    output: Path,
    expected_asset_sha256: str = OPENCODE_SHA256,
) -> dict[str, Any]:
    """Create a deterministic gzip tar without replacing an existing output."""
    dockerfile = (root / DOCKERFILE).resolve()
    dockerfile_bytes = dockerfile.read_bytes()
    asset_bytes = asset.read_bytes()
    if _sha256(asset_bytes) != expected_asset_sha256:
        raise ContextError("OpenCode release asset digest mismatch")
    if output.exists():
        raise ContextError("build context output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output.parent, prefix=".opencode-context-", delete=False
        ) as raw:
            temporary = Path(raw.name)
            with (
                gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed,
                tarfile.open(fileobj=compressed, mode="w", format=tarfile.USTAR_FORMAT) as archive,
            ):
                for name, data in (
                    ("Dockerfile", dockerfile_bytes),
                    (ASSET_NAME, asset_bytes),
                ):
                    member, payload = _tar_member(name, data)
                    archive.addfile(member, io.BytesIO(payload))
        os.link(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    receipt: dict[str, Any] = {
        "schema": CONTEXT_SCHEMA,
        "dockerfile": str(DOCKERFILE),
        "dockerfile_sha256": f"sha256:{_sha256(dockerfile_bytes)}",
        "base_image": BASE_IMAGE,
        "opencode_version": OPENCODE_VERSION,
        "release_asset_sha256": f"sha256:{expected_asset_sha256}",
        "archive_sha256": file_sha256(output),
        "archive_size_bytes": output.stat().st_size,
        "archive_entries": ["Dockerfile", ASSET_NAME],
        "dockerfile_run_network_mode": "none",
    }
    receipt["sha256"] = digest_json(receipt)
    return receipt


def build_request(*, context_id: str, context_sha256: str, source_commit: str) -> dict[str, Any]:
    if len(context_id) != 32 or any(char not in "0123456789abcdef" for char in context_id):
        raise ContextError("context_id must be 32 lowercase hexadecimal characters")
    if len(context_sha256) != 64 or any(char not in "0123456789abcdef" for char in context_sha256):
        raise ContextError("context_sha256 must be 64 lowercase hexadecimal characters")
    if len(source_commit) != 40 or any(char not in "0123456789abcdef" for char in source_commit):
        raise ContextError("source_commit must be one full lowercase Git SHA")
    return {
        "context_id": context_id,
        "dockerfile": "Dockerfile",
        "ecr_repository": ECR_REPOSITORY,
        "tag": f"opencode11827-{context_sha256[:16]}",
        "build_args": {
            "OPENCODE_VERSION": OPENCODE_VERSION,
            "OPENCODE_SHA256": OPENCODE_SHA256,
            "SOURCE_COMMIT": source_commit,
            "BUILD_CONTEXT_SHA256": context_sha256,
        },
        "platform": PLATFORM,
        "no_cache": True,
        "pull": False,
    }


def validate_request(request: dict[str, Any], *, context_sha256: str, source_commit: str) -> None:
    if request != build_request(
        context_id=str(request.get("context_id", "")),
        context_sha256=context_sha256,
        source_commit=source_commit,
    ):
        raise ContextError("dev image-build request differs from the exact publication request")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--asset", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    request = sub.add_parser("request")
    request.add_argument("--context-id", required=True)
    request.add_argument("--context-sha256", required=True)
    request.add_argument("--source-commit", required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        value = prepare_context(root=Path.cwd(), asset=args.asset, output=args.output)
    else:
        value = build_request(
            context_id=args.context_id,
            context_sha256=args.context_sha256,
            source_commit=args.source_commit,
        )
    print(json.dumps(value, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
