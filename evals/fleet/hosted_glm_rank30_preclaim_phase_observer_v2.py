"""ConfigMap-projection-safe successor to the rank-30 phase observer."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from evals.fleet import hosted_glm_rank30_preclaim_phase_observer_v1 as prior
from evals.fleet import self_hosted

SCHEMA = "fleet-hosted-glm-rank30-preclaim-phase-observer-v2"
JOB_NAME = "chris-glm53-r030-preclaim-phase-observer-v2"
CONFIGMAP_NAME = JOB_NAME + "-run"
OUTPUT_PATH = Path("/mnt/sfs/jobs") / JOB_NAME / "DIAGNOSTIC.json"
MANIFEST_PATH = Path("/bootstrap/frozen-source-manifest.json")


def _projected_regular_file(path: Path, root: Path) -> Path:
    try:
        resolved_root = root.resolve(strict=True)
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise RuntimeError("rank-30 projected source is unsafe") from error
    if not resolved.is_file() or not resolved.is_relative_to(resolved_root):
        raise RuntimeError("rank-30 projected source is unsafe")
    return resolved


def _validate_sources(root: Path, state: dict[str, Any]) -> None:
    bootstrap_root = MANIFEST_PATH.parent
    manifest_source = _projected_regular_file(MANIFEST_PATH, bootstrap_root)
    raw = manifest_source.read_bytes()
    manifest = json.loads(raw)
    files = manifest.get("files") if isinstance(manifest, dict) else None
    if (
        set(manifest) != {"schema_version", "source_package_sha256", "files"}
        or manifest.get("schema_version")
        != "fleet-hosted-glm-rank30-preclaim-source-manifest-v1"
        or manifest.get("source_package_sha256") != prior.SOURCE_PACKAGE_SHA256
        or not isinstance(files, dict)
        or not files
    ):
        raise RuntimeError("rank-30 frozen source manifest drifted")
    source_data: dict[str, str] = {}
    for key, binding in files.items():
        if not isinstance(binding, dict) or set(binding) != {"relative_path", "sha256"}:
            raise RuntimeError("rank-30 frozen source binding drifted")
        projected = _projected_regular_file(
            bootstrap_root / ("source__" + key), bootstrap_root
        )
        installed = root / str(binding["relative_path"])
        if installed.is_symlink() or not installed.is_file():
            raise RuntimeError("rank-30 frozen source file is unsafe")
        payload = projected.read_bytes()
        if prior._file_sha256(payload) != binding["sha256"] or installed.read_bytes() != payload:  # noqa: SLF001
            raise RuntimeError("rank-30 frozen source digest drifted")
        source_data[key] = payload.decode()
    if self_hosted.sha256(self_hosted.canonical_json(source_data)) != prior.SOURCE_PACKAGE_SHA256:
        raise RuntimeError("rank-30 frozen package digest drifted")
    state["source_manifest_sha256"] = prior._file_sha256(raw)  # noqa: SLF001


_CANONICAL_PHASES = ((prior.PHASES[0][0], _validate_sources), *prior.PHASES[1:])
PHASES = _CANONICAL_PHASES


def _validate_phase_bindings() -> None:
    if PHASES is not _CANONICAL_PHASES or len(PHASES) != len(_CANONICAL_PHASES):
        raise RuntimeError("rank-30 v2 observer phase binding drifted")
    for actual, expected in zip(PHASES, _CANONICAL_PHASES, strict=True):
        if actual[0] != expected[0] or actual[1] is not expected[1]:
            raise RuntimeError("rank-30 v2 observer phase binding drifted")


def _activate() -> None:
    _validate_phase_bindings()
    prior.SCHEMA = SCHEMA
    prior.JOB_NAME = JOB_NAME
    prior.CONFIGMAP_NAME = CONFIGMAP_NAME
    prior.OUTPUT_PATH = OUTPUT_PATH
    prior.MANIFEST_PATH = MANIFEST_PATH
    prior.PHASES = PHASES
    prior._CANONICAL_PHASES = _CANONICAL_PHASES  # noqa: SLF001


def run(root: Path, *, output_path: Path = OUTPUT_PATH) -> int:
    _activate()
    return prior.run(root, output_path=output_path)


def main() -> int:
    return run(Path(os.environ.get("REPO_ROOT", "/workspace/cyber-post-train")))


if __name__ == "__main__":
    raise SystemExit(main())
