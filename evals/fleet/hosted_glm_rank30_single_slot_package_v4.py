"""Render the v6-gated rank-30 scored controller package."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from evals.fleet import hosted_glm_rank30_single_slot_package_v3 as prior
from evals.fleet import hosted_glm_rank30_single_slot_v5 as successor

PATHS = {
    **prior.PATHS,
    "bulk_source.py": "evals/fleet/hosted_glm_exact_bulk_v1.py",
    "source_runtime.py": "evals/fleet/hosted_glm_exact_bulk_runtime_v1.py",
    "whole.py": "evals/fleet/hosted_glm_whole_task_successor_v1.py",
    "single_slot_v3.py": "evals/fleet/hosted_glm_rank30_single_slot_v3.py",
    "single_slot_v4.py": "evals/fleet/hosted_glm_rank30_single_slot_v4.py",
    "bulk.py": "evals/fleet/hosted_glm_rank30_single_slot_v5.py",
    "bulk_runtime.py": "evals/fleet/hosted_glm_rank30_single_slot_runtime_v4.py",
    "run.sh": "evals/fleet/scripts/run_hosted_glm_rank30_single_slot_v4.sh",
}


def _data(root: Path) -> dict[str, str]:
    result = {}
    for name, relative in PATHS.items():
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"unsafe rank-30 v4 package source: {relative}")
        result[name] = path.read_text()
    return result


def source_package_sha256(root: Path) -> str:
    return prior.prior.self_hosted.sha256(
        prior.prior.self_hosted.canonical_json(_data(root))
    )


def render(root: Path, *, release_receipt: Path | None = None) -> dict[str, Any]:
    original_paths = prior.prior.PATHS
    original_successor = prior.prior.successor
    try:
        prior.prior.PATHS = PATHS
        prior.prior.successor = successor
        return prior.prior.render(root, release_receipt=release_receipt)
    finally:
        prior.prior.successor = original_successor
        prior.prior.PATHS = original_paths
