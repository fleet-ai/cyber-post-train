"""Fail-closed authorization gate for scored Kubernetes Job manifests."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

LAUNCH_AUTHORIZED = "cyber-post-train.fleet.ai/launch-authorized"
PREVIEW_ONLY = "cyber-post-train.fleet.ai/preview-only"


def validate_scored_manifest(
    path: Path,
    expected_job_names: tuple[str, ...],
    *,
    launch_authorized: bool = True,
) -> None:
    try:
        documents = list(yaml.safe_load_all(path.read_text(encoding="utf-8")))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError("scored manifest could not be loaded") from exc

    if not documents or not expected_job_names or len(set(expected_job_names)) != len(
        expected_job_names
    ):
        raise ValueError("scored manifest authorization inputs are invalid")

    observed_names: list[str] = []
    for document in documents:
        if not isinstance(document, dict):
            raise ValueError("scored manifest contains a non-object document")
        metadata: Any = document.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError("scored manifest Job metadata is invalid")
        annotations: Any = metadata.get("annotations")
        name = metadata.get("name")
        if (
            document.get("apiVersion") != "batch/v1"
            or document.get("kind") != "Job"
            or not isinstance(name, str)
            or not isinstance(annotations, dict)
            or annotations.get(LAUNCH_AUTHORIZED)
            != ("true" if launch_authorized else "false")
            or annotations.get(PREVIEW_ONLY)
            != ("false" if launch_authorized else "true")
        ):
            raise ValueError("scored manifest contains an unauthorized Job")
        observed_names.append(name)

    if len(set(observed_names)) != len(observed_names) or set(observed_names) != set(
        expected_job_names
    ):
        raise ValueError("scored manifest Job identity does not match the launch plan")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-job", action="append", required=True)
    parser.add_argument("--held", action="store_true")
    args = parser.parse_args()
    validate_scored_manifest(
        args.manifest,
        tuple(args.expected_job),
        launch_authorized=not args.held,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
