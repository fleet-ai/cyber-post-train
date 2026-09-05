"""Build the immutable, held-only package for the exact pass@4 bulk plan."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from evals.fleet import exact_pass4_bulk_v1 as bulk

SCHEMA = "fleet-exact-pass4-bulk-split-package-v1"
CORE_NAME = "chris-exact-pass4-bulk-runtime-core-v1"
KUBERNETES_OBJECT_LIMIT = 1_048_576
PACKAGE_OBJECT_LIMIT = 900_000
CORE_PATHS = (
    bulk.MODULE_PATH,
    bulk.PACKAGE_PATH,
    bulk.RENDER_PATH,
    bulk.SUBMIT_PATH,
    bulk.SNAPSHOT_PATH,
    "evals/fleet/exact_pass4_universe.py",
    "evals/fleet/exact_pass4_crypto.py",
    "evals/fleet/exact_pass4_task_inventory.py",
    "evals/fleet/exact_pass4_task_inventory_package.py",
    "evals/fleet/endpoint_lease.py",
    "evals/fleet/self_hosted.py",
    "evals/fleet/fixed_proxy.py",
    "evals/fleet/Dockerfile.opencode",
    bulk.CAMPAIGN_PATH,
    bulk.SELECTION_PATH,
    bulk.HELD_PATH,
    bulk.MANIFEST_PATH,
    *bulk.SPEC_PATHS.values(),
    *bulk.CANARY_SPEC_PATHS.values(),
    *bulk.RUNTIME_TEMPLATE_PATHS.values(),
)
CONTROLLER_NAMES = {key: bulk.CONTROLLERS[key]["configmap_name"] for key in bulk.CONTROLLERS}
CONTROLLER_PATHS = {
    key: (bulk.RUNTIME_PATH, bulk.RUN_PATH) for key in bulk.CONTROLLERS
}


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def data_key(path: str) -> str:
    return "f-" + hashlib.sha256(path.encode()).hexdigest()[:24]


def _payload(root: Path, paths: tuple[str, ...]) -> tuple[dict[str, str], list[dict[str, Any]]]:
    data: dict[str, str] = {}
    entries: list[dict[str, Any]] = []
    for source_path in paths:
        raw = (root / source_path).read_bytes()
        key = data_key(source_path)
        if key in data:
            raise ValueError("bulk package data-key collision")
        data[key] = raw.decode("utf-8")
        entries.append(
            {
                "data_key": key,
                "source_path": source_path,
                "bytes": len(raw),
                "sha256": sha256(raw),
            }
        )
    return data, entries


def _object(name: str, entries: list[dict[str, Any]]) -> dict[str, Any]:
    body = {"name": name, "entries": entries}
    return {**body, "payload_sha256": sha256(canonical(body))}


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


def _size(value: dict[str, Any]) -> int:
    return len(canonical(value))


def build_package(root: Path) -> dict[str, Any]:
    plans = bulk.validate_all(root)
    bulk.validate_held(bulk.load(root / bulk.HELD_PATH), root)
    payloads: dict[str, dict[str, str]] = {}
    objects: dict[str, dict[str, Any]] = {}
    payloads[CORE_NAME], entries = _payload(root, CORE_PATHS)
    objects[CORE_NAME] = _object(CORE_NAME, entries)
    for controller, name in CONTROLLER_NAMES.items():
        payloads[name], entries = _payload(root, CONTROLLER_PATHS[controller])
        objects[name] = _object(name, entries)

    controller_manifests: dict[str, dict[str, Any]] = {}
    for controller, name in CONTROLLER_NAMES.items():
        selected = [objects[CORE_NAME], objects[name]]
        manifest = {
            "schema_version": SCHEMA,
            "controller": controller,
            "spec_sha256": bulk.load(root / bulk.SPEC_PATHS[controller])["spec_sha256"],
            "plan_sha256": plans[controller]["plan_sha256"],
            "objects": selected,
            "aggregate_sha256": sha256(canonical(selected)),
            "release_included": False,
            "launch_authorized": False,
        }
        payloads[name]["package-manifest.json"] = canonical(manifest).decode() + "\n"
        payloads[name]["package_aggregate_sha256"] = manifest["aggregate_sha256"]
        controller_manifests[controller] = manifest

    configmaps = {name: _configmap(name, data) for name, data in payloads.items()}
    sizes = {name: _size(value) for name, value in configmaps.items()}
    if any(size >= KUBERNETES_OBJECT_LIMIT for size in sizes.values()):
        raise ValueError("bulk package exceeds Kubernetes object limit")
    if any(size >= PACKAGE_OBJECT_LIMIT for size in sizes.values()):
        raise ValueError("bulk package exceeds safety budget")
    all_objects = [objects[name] for name in sorted(objects)]
    return {
        "schema_version": SCHEMA,
        "configmaps": configmaps,
        "object_manifests": objects,
        "controller_manifests": controller_manifests,
        "object_json_bytes": sizes,
        "aggregate_sha256": sha256(canonical(all_objects)),
        "release_included": False,
        "launch_authorized": False,
    }


def verify_mounted(manifest_path: Path, bootstrap: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema_version") != SCHEMA
        or manifest.get("release_included") is not False
        or manifest.get("launch_authorized") is not False
        or manifest.get("aggregate_sha256") != sha256(canonical(manifest.get("objects")))
    ):
        raise ValueError("mounted bulk package manifest drifted")
    for obj in manifest["objects"]:
        body = {key: value for key, value in obj.items() if key != "payload_sha256"}
        if obj.get("payload_sha256") != sha256(canonical(body)):
            raise ValueError("mounted bulk package object manifest drifted")
        for entry in obj["entries"]:
            raw = (bootstrap / entry["data_key"]).read_bytes()
            if len(raw) != entry["bytes"] or sha256(raw) != entry["sha256"]:
                raise ValueError("mounted bulk package payload drifted")
    return manifest


def materialize_mounted(manifest_path: Path, bootstrap: Path, destination: Path) -> None:
    manifest = verify_mounted(manifest_path, bootstrap)
    for obj in manifest["objects"]:
        for entry in obj["entries"]:
            target = destination / entry["source_path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((bootstrap / entry["data_key"]).read_bytes())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preview", "verify-mounted", "materialize-mounted"))
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--bootstrap", type=Path)
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    if args.command == "preview":
        package = build_package(args.repo.resolve())
        print(
            json.dumps(
                {
                    "ok": True,
                    "status": "HELD",
                    "aggregate_sha256": package["aggregate_sha256"],
                    "object_json_bytes": package["object_json_bytes"],
                    "release_included": False,
                    "launch_authorized": False,
                    "objects_created": False,
                },
                sort_keys=True,
            )
        )
        return 0
    if args.manifest is None or args.bootstrap is None:
        parser.error(f"{args.command} requires --manifest and --bootstrap")
    if args.command == "verify-mounted":
        verify_mounted(args.manifest, args.bootstrap)
        return 0
    if args.destination is None:
        parser.error("materialize-mounted requires --destination")
    materialize_mounted(args.manifest, args.bootstrap, args.destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
