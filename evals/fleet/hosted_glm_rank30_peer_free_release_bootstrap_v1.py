"""Verify and enter the immutable peer-free rank-30 observer package."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

SCHEMA = "fleet-hosted-glm-rank30-peer-free-observer-package-v1"
MAX_BYTES = 16 * 1024 * 1024


class BootstrapError(RuntimeError):
    """The projected package did not match its immutable manifest."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _sha(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _strict(raw: bytes) -> dict[str, Any]:
    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in rows:
            if key in result:
                raise BootstrapError("duplicate_json_key")
            result[key] = value
        return result

    try:
        value = json.loads(raw, object_pairs_hook=pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BootstrapError("invalid_json") from exc
    if not isinstance(value, dict):
        raise BootstrapError("json_root_invalid")
    return value


def validate(package_path: Path, projected_root: Path, repo: Path) -> None:
    root = projected_root.resolve(strict=True)
    package_source = package_path.resolve(strict=True)
    if not package_source.is_relative_to(root) or package_source.stat().st_size > MAX_BYTES:
        raise BootstrapError("package_path_invalid")
    package = _strict(package_source.read_bytes())
    files = package.get("files")
    receipt = package.get("receipt_sha256")
    unsigned = {key: value for key, value in package.items() if key != "receipt_sha256"}
    if any(
        (
            set(package) != {"schema_version", "files", "file_count", "receipt_sha256"},
            package.get("schema_version") != SCHEMA,
            not isinstance(files, dict),
            not files,
            package.get("file_count") != len(files or {}),
            receipt != _sha(_canonical(unsigned)),
        )
    ):
        raise BootstrapError("package_manifest_invalid")
    repo_root = repo.resolve(strict=True)
    for relative, expected in files.items():
        if (
            not isinstance(relative, str)
            or relative.startswith("/")
            or ".." in Path(relative).parts
        ):
            raise BootstrapError("package_member_name_invalid")
        projected = (projected_root / relative.replace("/", "__SLASH__")).resolve(strict=True)
        materialized = (repo / relative).resolve(strict=True)
        if any(
            (
                not projected.is_relative_to(root),
                not materialized.is_relative_to(repo_root),
                not projected.is_file(),
                not materialized.is_file(),
                projected.stat().st_size > MAX_BYTES,
                materialized.stat().st_size > MAX_BYTES,
                _sha(projected.read_bytes()) != expected,
                _sha(materialized.read_bytes()) != expected,
            )
        ):
            raise BootstrapError("package_member_invalid")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--projected-root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("observer_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    validate(args.package, args.projected_root, args.repo)
    if not args.observer_args or args.observer_args[0] != "--":
        raise BootstrapError("observer_args_invalid")
    os.execv(
        sys.executable,
        [
            sys.executable,
            "-m",
            "evals.fleet.hosted_glm_rank30_peer_free_release_observer_v1",
            *args.observer_args[1:],
        ],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
