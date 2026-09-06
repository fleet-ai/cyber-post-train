"""Render rank 30 with the current rank-29 attempt-4 release authority."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from evals.fleet import hosted_glm_rank30_single_slot_package_v1 as prior
from evals.fleet import hosted_glm_rank30_single_slot_v3 as successor

PATHS = {
    **prior.PATHS,
    "single_slot_v1.py": "evals/fleet/hosted_glm_rank30_single_slot_v1.py",
    "bulk.py": "evals/fleet/hosted_glm_rank30_single_slot_v3.py",
    "bulk_runtime.py": "evals/fleet/hosted_glm_rank30_single_slot_runtime_v3.py",
    "run.sh": "evals/fleet/scripts/run_hosted_glm_rank30_single_slot_v3.sh",
}


def _data(root: Path) -> dict[str, str]:
    result = {}
    for name, relative in PATHS.items():
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"unsafe rank-30 v3 package source: {relative}")
        result[name] = path.read_text()
    return result


def source_package_sha256(root: Path) -> str:
    return prior.self_hosted.sha256(prior.self_hosted.canonical_json(_data(root)))


def render(root: Path, *, release_receipt: Path | None = None) -> dict[str, Any]:
    original_paths = prior.PATHS
    original_successor = prior.successor
    try:
        prior.PATHS = PATHS
        prior.successor = successor
        return prior.render(root, release_receipt=release_receipt)
    finally:
        prior.successor = original_successor
        prior.PATHS = original_paths
