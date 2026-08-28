"""Dry-run-by-default Nebius submitter with live Kueue safety checks."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

from .cluster_policy import (
    EXPECTED_CLUSTER_QUEUE,
    EXPECTED_CONTEXT,
    EXPECTED_NAMESPACE,
    EXPECTED_QUEUE,
    OWNER_LABEL,
    validate_scheduler_state,
    validate_training_manifest,
)


def _run_json(argv: list[str]) -> dict[str, Any]:
    result = subprocess.run(argv, check=True, capture_output=True, text=True)
    value = json.loads(result.stdout)
    if not isinstance(value, dict):
        raise ValueError(f"{' '.join(argv)} returned non-object JSON")
    return value


def submit(path: Path, *, execute: bool = False) -> dict[str, Any]:
    manifest = yaml.safe_load(path.read_text())
    if not isinstance(manifest, dict):
        raise ValueError("workload manifest must contain one YAML object")
    validate_training_manifest(manifest)
    context = subprocess.run(
        ["kubectl", "config", "current-context"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if context != EXPECTED_CONTEXT:
        raise ValueError(f"refusing unexpected Kubernetes context {context!r}")
    cluster_queue = _run_json(
        ["kubectl", "get", "clusterqueue", EXPECTED_CLUSTER_QUEUE, "-o", "json"]
    )
    local_queue = _run_json(
        [
            "kubectl",
            "-n",
            EXPECTED_NAMESPACE,
            "get",
            "localqueue",
            EXPECTED_QUEUE,
            "-o",
            "json",
        ]
    )
    validate_scheduler_state(cluster_queue, local_queue)
    kind = str(manifest["kind"]).lower()
    name = str(manifest["metadata"]["name"])
    existing = subprocess.run(
        [
            "kubectl",
            "-n",
            EXPECTED_NAMESPACE,
            "get",
            kind,
            name,
            "--ignore-not-found",
            "-o",
            "json",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if existing:
        live = json.loads(existing)
        live_labels = live.get("metadata", {}).get("labels", {})
        if live_labels.get(OWNER_LABEL) != "chris":
            raise ValueError("refusing to modify an existing workload not owned by chris")
    subprocess.run(
        ["kubectl", "apply", "--dry-run=server", "-f", str(path), "-o", "name"],
        check=True,
        capture_output=True,
        text=True,
    )
    metadata = manifest["metadata"]
    receipt = {
        "dry_run": not execute,
        "context": context,
        "namespace": EXPECTED_NAMESPACE,
        "queue": EXPECTED_QUEUE,
        "name": name,
        "owner": metadata["labels"][OWNER_LABEL],
    }
    if execute:
        subprocess.run(["kubectl", "apply", "-f", str(path)], check=True)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--execute", action="store_true", help="submit after all checks")
    args = parser.parse_args(argv)
    print(json.dumps(submit(args.manifest, execute=args.execute), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
