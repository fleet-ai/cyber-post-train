"""Materialize the exact frozen rank-30 package from a projected ConfigMap."""

from __future__ import annotations

import json
import os
from pathlib import Path

SCHEMA = "fleet-hosted-glm-rank30-preclaim-source-manifest-v1"


def _safe_relative(value: object) -> Path:
    if not isinstance(value, str):
        raise RuntimeError("rank-30 diagnostic source path is invalid")
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise RuntimeError("rank-30 diagnostic source path is invalid")
    return path


def materialize(bootstrap: Path, root: Path) -> None:
    manifest_path = bootstrap / "frozen-source-manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise RuntimeError("rank-30 diagnostic source manifest is unsafe")
    manifest = json.loads(manifest_path.read_text())
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
        source = bootstrap / ("source__" + key)
        relative = _safe_relative(binding["relative_path"])
        target = root / relative
        if source.is_symlink() or not source.is_file() or target.exists():
            raise RuntimeError("rank-30 diagnostic source materialization is unsafe")
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(source.read_bytes())
    for package in (root / "evals", root / "evals/fleet"):
        package.mkdir(parents=True, exist_ok=True)
        initializer = package / "__init__.py"
        if not initializer.exists():
            initializer.write_text("")
    diagnostic = root / "evals/fleet/hosted_glm_rank30_preclaim_phase_observer_v1.py"
    descriptor = os.open(diagnostic, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write((bootstrap / "diagnostic.py").read_bytes())


def main() -> int:
    materialize(
        Path(os.environ.get("BOOTSTRAP_ROOT", "/bootstrap")),
        Path(os.environ.get("REPO_ROOT", "/workspace/cyber-post-train")),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
