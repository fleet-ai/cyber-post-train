"""Append-only checkpoint adapter for exact Fast3 training plans.

The frozen SkyRL post-training implementation owns checkpoint and rollout
verification. Its only Fast3-incompatible boundary is the closed historical
plan-schema validator. These wrappers temporarily substitute the exact Fast3
validator while preserving the original plan in every manifest.
"""

from __future__ import annotations

from pathlib import Path
from types import FunctionType

from . import skyrl_fast3_training as fast3
from . import skyrl_posttrain as historical

MANIFEST_SCHEMA = historical.MANIFEST_SCHEMA
EXPORT_SCHEMA = historical.EXPORT_SCHEMA


def _with_fast3_validator(function):
    """Copy one frozen entrypoint with only its plan validator replaced."""
    namespace = {**function.__globals__, "_validate_plan": fast3._validated}
    adapted = FunctionType(
        function.__code__,
        namespace,
        name=function.__name__,
        argdefs=function.__defaults__,
        closure=function.__closure__,
    )
    adapted.__kwdefaults__ = function.__kwdefaults__
    return adapted


_seal_checkpoint = _with_fast3_validator(historical.seal_checkpoint)
_verify_manifest = _with_fast3_validator(historical.verify_manifest)


def seal_checkpoint(plan: dict, output: Path, *, progress=None) -> dict:
    return _seal_checkpoint(plan, output, progress=progress)


def verify_manifest(manifest: dict, *, check_files: bool = True) -> None:
    _verify_manifest(manifest, check_files=check_files)


def export_checkpoint(
    manifest_path: Path, expected_sha256: str, output: Path, *, progress=None
) -> dict:
    """Export only after the manifest's exact Fast3 plan validates."""
    from .export import _export_verified

    return _export_verified(
        manifest_path,
        expected_sha256,
        output,
        verify_manifest=verify_manifest,
        receipt_schema=EXPORT_SCHEMA,
        code_files=("skyrl_posttrain.py", "skyrl_fast3_posttrain.py"),
        progress=progress,
    )
