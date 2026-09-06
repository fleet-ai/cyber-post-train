"""Materialize the rank-30 observer from a Kubernetes ConfigMap projection."""

from __future__ import annotations

import json
import os
from contextlib import suppress
from pathlib import Path

SCHEMA = "fleet-hosted-glm-rank30-preclaim-source-manifest-v1"


def _projected_regular_file(path: Path, root: Path) -> Path:
    try:
        resolved_root = root.resolve(strict=True)
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise RuntimeError("rank-30 projected source is unsafe") from error
    if not resolved.is_file() or not resolved.is_relative_to(resolved_root):
        raise RuntimeError("rank-30 projected source is unsafe")
    return resolved


def _safe_relative(value: object) -> Path:
    if not isinstance(value, str):
        raise RuntimeError("rank-30 diagnostic source path is invalid")
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise RuntimeError("rank-30 diagnostic source path is invalid")
    return path


def _write_once(root: Path, relative: Path, payload: bytes) -> None:
    relative = _safe_relative(str(relative))
    try:
        root.mkdir(parents=True, exist_ok=True)
        root_descriptor = os.open(
            root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        )
    except OSError as error:
        raise RuntimeError("rank-30 diagnostic destination is unsafe") from error
    parent_descriptor = root_descriptor
    try:
        for part in relative.parts[:-1]:
            with suppress(FileExistsError):
                os.mkdir(part, 0o700, dir_fd=parent_descriptor)
            next_descriptor = os.open(
                part,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=parent_descriptor,
            )
            if parent_descriptor != root_descriptor:
                os.close(parent_descriptor)
            parent_descriptor = next_descriptor
        descriptor = os.open(
            relative.name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=parent_descriptor,
        )
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
    except OSError as error:
        raise RuntimeError("rank-30 diagnostic destination is unsafe") from error
    finally:
        if parent_descriptor != root_descriptor:
            os.close(parent_descriptor)
        os.close(root_descriptor)


def materialize(bootstrap: Path, root: Path) -> None:
    manifest_source = _projected_regular_file(
        bootstrap / "frozen-source-manifest.json", bootstrap
    )
    manifest = json.loads(manifest_source.read_text())
    files = manifest.get("files") if isinstance(manifest, dict) else None
    if (
        set(manifest) != {"schema_version", "source_package_sha256", "files"}
        or manifest.get("schema_version") != SCHEMA
        or not isinstance(files, dict)
        or not files
    ):
        raise RuntimeError("rank-30 diagnostic source manifest is invalid")
    for key, binding in files.items():
        if (
            not isinstance(key, str)
            or not isinstance(binding, dict)
            or set(binding) != {"relative_path", "sha256"}
        ):
            raise RuntimeError("rank-30 diagnostic source binding is invalid")
        source = _projected_regular_file(bootstrap / ("source__" + key), bootstrap)
        _write_once(root, _safe_relative(binding["relative_path"]), source.read_bytes())
    for initializer in (Path("evals/__init__.py"), Path("evals/fleet/__init__.py")):
        _write_once(root, initializer, b"")
    _write_once(
        root,
        Path("evals/fleet/hosted_glm_rank30_preclaim_phase_observer_v1.py"),
        _projected_regular_file(bootstrap / "diagnostic-v1.py", bootstrap).read_bytes(),
    )
    _write_once(
        root,
        Path("evals/fleet/hosted_glm_rank30_preclaim_phase_observer_v2.py"),
        _projected_regular_file(bootstrap / "diagnostic-v2.py", bootstrap).read_bytes(),
    )


def main() -> int:
    materialize(
        Path(os.environ.get("BOOTSTRAP_ROOT", "/bootstrap")),
        Path(os.environ.get("REPO_ROOT", "/workspace/cyber-post-train")),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
