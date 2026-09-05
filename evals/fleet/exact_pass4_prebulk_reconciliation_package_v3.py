"""Immutable split package for the held exact-pass@4 pre-bulk observer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from evals.fleet import autocontinue_generation7_authority_package_v1 as generation7

SCHEMA = "fleet-exact-pass4-prebulk-reconciliation-split-package-v3"
CORE_A_NAME = "chris-exact100-prebulk-runtime-core-a-v3"
CORE_B_NAME = "chris-exact100-prebulk-runtime-core-b-v3"
OBSERVER_NAME = "chris-exact100-prebulk-runtime-observer-v3"
OBJECT_LIMIT = 1_048_576
SAFETY_LIMIT = 900_000
MANIFEST_PATH = "evals/fleet/cluster/exact-pass4-prebulk-reconciliation-held-v3.yaml"
RUN_PATH = "evals/fleet/scripts/run_exact_pass4_prebulk_reconciliation_v3.sh"
SUBMIT_PATH = "evals/fleet/scripts/submit_exact_pass4_prebulk_reconciliation_v3.sh"
HELD_PATH = (
    "docs/evidence/qwen38-study/2026-09-05-opencode-exact-pass4-prebulk-reconciliation-held-v3.json"
)
MODULE_PATH = "evals/fleet/exact_pass4_prebulk_reconciliation_v3.py"
PACKAGE_PATH = "evals/fleet/exact_pass4_prebulk_reconciliation_package_v3.py"
CORE_A_PATHS = generation7.CORE_A_PATHS
CORE_B_PATHS = generation7.CORE_B_PATHS
OBSERVER_PATHS = (
    MODULE_PATH,
    PACKAGE_PATH,
    RUN_PATH,
    SUBMIT_PATH,
    MANIFEST_PATH,
    HELD_PATH,
    "evals/fleet/exact_pass4_bulk_v3.py",
    "evals/fleet/exact_pass4_task_inventory.py",
    "evals/fleet/exact_pass4_task_inventory_package.py",
    "evals/fleet/configs/exact-pass4-bulk-qwen-a-v3.json",
    "evals/fleet/configs/exact-pass4-bulk-qwen-b-v3.json",
    "evals/fleet/configs/exact-pass4-bulk-glm-a-v3.json",
    "evals/fleet/configs/exact-pass4-bulk-glm-b-v3.json",
)


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def sha256(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def data_key(path: str) -> str:
    if path == RUN_PATH:
        return "run.sh"
    if path == PACKAGE_PATH:
        return "package.py"
    return "f-" + hashlib.sha256(path.encode()).hexdigest()[:24]


def _payload(root: Path, paths: tuple[str, ...]) -> tuple[dict[str, str], dict[str, Any]]:
    data: dict[str, str] = {}
    entries: list[dict[str, Any]] = []
    for source_path in paths:
        raw = (root / source_path).read_bytes()
        key = data_key(source_path)
        if key in data:
            raise ValueError("prebulk package data key collision")
        data[key] = raw.decode()
        entries.append(
            {
                "data_key": key,
                "source_path": source_path,
                "bytes": len(raw),
                "sha256": sha256(raw),
            }
        )
    return data, {"entries": entries}


def _configmap(name: str, data: dict[str, str]) -> dict[str, Any]:
    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": name,
            "namespace": "fleet-train-jobs",
            "annotations": {
                "cyber-post-train.fleet.ai/preview-only": "true",
                "cyber-post-train.fleet.ai/launch-authorized": "false",
            },
        },
        "immutable": True,
        "data": data,
    }


def build_package(root: Path) -> dict[str, Any]:
    # Keep the package itself fail-closed: a changed or missing held authority
    # may not be packaged even when the caller bypasses the submit wrapper.
    from evals.fleet import exact_pass4_prebulk_reconciliation_v3 as reconcile

    reconcile.validate_held(root)
    groups = (
        (CORE_A_NAME, CORE_A_PATHS),
        (CORE_B_NAME, CORE_B_PATHS),
        (OBSERVER_NAME, OBSERVER_PATHS),
    )
    configmaps: dict[str, dict[str, Any]] = {}
    objects: list[dict[str, Any]] = []
    for name, paths in groups:
        data, body = _payload(root, paths)
        unsigned = {"name": name, **body}
        obj = {**unsigned, "payload_sha256": sha256(canonical(unsigned))}
        configmaps[name] = _configmap(name, data)
        objects.append(obj)
    manifest = {
        "schema_version": SCHEMA,
        "objects": objects,
        "aggregate_sha256": sha256(canonical(objects)),
        "source_job": "chris-cyber-exact100-prebulk-reconcile-source-v3",
        "accept_job": "chris-cyber-exact100-prebulk-reconcile-accept-v3",
        "release_included": False,
        "launch_authorized": False,
    }
    configmaps[OBSERVER_NAME]["data"]["package-manifest.json"] = canonical(manifest).decode() + "\n"
    sizes = {name: len(canonical(value)) for name, value in configmaps.items()}
    if any(size >= OBJECT_LIMIT for size in sizes.values()):
        raise ValueError("prebulk package exceeds Kubernetes object limit")
    if any(size >= SAFETY_LIMIT for size in sizes.values()):
        raise ValueError("prebulk package exceeds ConfigMap safety budget")
    return {
        "schema_version": SCHEMA,
        "configmaps": configmaps,
        "manifest": manifest,
        "object_json_bytes": sizes,
        "aggregate_sha256": manifest["aggregate_sha256"],
        "release_included": False,
        "launch_authorized": False,
    }


def materialize(projected: Path, destination: Path, expected: str) -> dict[str, Any]:
    manifest = json.loads((projected / "package-manifest.json").read_text())
    if (
        manifest.get("schema_version") != SCHEMA
        or manifest.get("aggregate_sha256") != expected
        or manifest.get("aggregate_sha256") != sha256(canonical(manifest.get("objects")))
        or manifest.get("release_included") is not False
        or manifest.get("launch_authorized") is not False
    ):
        raise ValueError("prebulk mounted package manifest drifted")
    destination.mkdir(mode=0o700, parents=True, exist_ok=False)
    for path in (destination / "evals/__init__.py", destination / "evals/fleet/__init__.py"):
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        path.write_bytes(b"")
    for obj in manifest["objects"]:
        body = {key: value for key, value in obj.items() if key != "payload_sha256"}
        if obj.get("payload_sha256") != sha256(canonical(body)):
            raise ValueError("prebulk mounted object manifest drifted")
        for entry in obj["entries"]:
            raw = (projected / entry["data_key"]).read_bytes()
            if len(raw) != entry["bytes"] or sha256(raw) != entry["sha256"]:
                raise ValueError("prebulk mounted package payload drifted")
            target = destination / entry["source_path"]
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            target.write_bytes(raw)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preview", "materialize"))
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--projected", type=Path)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--expected")
    args = parser.parse_args()
    if args.command == "preview":
        built = build_package(args.repo)
        print(
            json.dumps(
                {
                    "status": "HELD",
                    "aggregate_sha256": built["aggregate_sha256"],
                    "object_json_bytes": built["object_json_bytes"],
                    "launch_authorized": False,
                    "objects_created": False,
                },
                sort_keys=True,
            )
        )
        return 0
    if args.projected is None or args.destination is None or args.expected is None:
        parser.error("materialize requires --projected, --destination, and --expected")
    materialize(args.projected, args.destination, args.expected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
