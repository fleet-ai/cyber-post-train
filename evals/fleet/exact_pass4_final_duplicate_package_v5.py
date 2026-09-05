"""Immutable split package for the final-v5 duplicate source/accept Jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA = "fleet-exact-pass4-final-duplicate-package-v5"
CORE_NAMES = tuple(f"chris-final-v5-dup-core-{letter}" for letter in "abcd")
OBSERVER_NAME = "chris-final-v5-dup-runtime"
PACKAGE_OBJECT_LIMIT = 900_000
PACKAGE_PATH = "evals/fleet/exact_pass4_final_duplicate_package_v5.py"
RUN_PATH = "evals/fleet/scripts/run_exact_pass4_final_duplicate_v5.sh"
OBSERVER_PATHS = (
    "evals/fleet/exact_pass4_final_duplicate_observer_v5.py",
    PACKAGE_PATH,
    "evals/fleet/exact_pass4_final_duplicate_renderer_v5.py",
    RUN_PATH,
    "evals/fleet/scripts/submit_exact_pass4_final_duplicate_v5.sh",
    "evals/fleet/cluster/exact-pass4-final-duplicate-held-v5.yaml",
    "docs/evidence/qwen38-study/"
    "2026-09-05-opencode-exact-pass4-final-duplicate-observer-held-v5.json",
    "docs/EXACT_PASS4_FINAL_DUPLICATE_OBSERVER_V5.md",
)


def _final_package() -> Any:
    from evals.fleet import exact_pass4_final_bulk_package_v5

    return exact_pass4_final_bulk_package_v5


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def sha256(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def data_key(path: str) -> str:
    if path == PACKAGE_PATH:
        return "package.py"
    if path == RUN_PATH:
        return "run.sh"
    return "f-" + hashlib.sha256(path.encode()).hexdigest()[:24]


def _payload(root: Path, paths: tuple[str, ...]) -> tuple[dict[str, str], list[dict[str, Any]]]:
    data: dict[str, str] = {}
    entries: list[dict[str, Any]] = []
    for source_path in paths:
        raw = (root / source_path).read_bytes()
        key = data_key(source_path)
        if key in data:
            raise ValueError("final duplicate package data-key collision")
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


def build_package(root: Path) -> dict[str, Any]:
    from evals.fleet import exact_pass4_final_duplicate_observer_v5 as observer

    final = _final_package()
    bulk = observer._bulk()  # noqa: SLF001 - exact final-v5 authority
    bulk.validate_held(bulk.load(root / bulk.HELD_PATH), root)
    observer.validate_held(root)
    groups = dict(zip(CORE_NAMES, final.CORE_PATH_GROUPS.values(), strict=True))
    groups[OBSERVER_NAME] = OBSERVER_PATHS
    payloads: dict[str, dict[str, str]] = {}
    objects: list[dict[str, Any]] = []
    for name, paths in groups.items():
        payload, entries = _payload(root, paths)
        payloads[name] = payload
        objects.append(_object(name, entries))
    manifest = {
        "schema_version": SCHEMA,
        "objects": objects,
        "aggregate_sha256": sha256(canonical(objects)),
        "authorized_groups": list(bulk.GROUPS),
        "source_jobs": {group: observer.source_job(group) for group in bulk.GROUPS},
        "accept_jobs": {group: observer.accept_job(group) for group in bulk.GROUPS},
        "release_included": False,
        "launch_authorized": False,
    }
    payloads[OBSERVER_NAME]["package-manifest.json"] = canonical(manifest).decode() + "\n"
    configmaps = {
        name: _configmap(name, data)
        for name, data in payloads.items()
    }
    sizes = {name: len(canonical(value)) for name, value in configmaps.items()}
    if any(size >= PACKAGE_OBJECT_LIMIT for size in sizes.values()):
        raise ValueError("final duplicate package exceeds Kubernetes object limit")
    if any(size >= 900_000 for size in sizes.values()):
        raise ValueError("final duplicate package exceeds ConfigMap safety budget")
    return {
        "schema_version": SCHEMA,
        "configmaps": configmaps,
        "manifest": manifest,
        "object_json_bytes": sizes,
        "aggregate_sha256": manifest["aggregate_sha256"],
        "release_included": False,
        "launch_authorized": False,
    }


def verify_mounted(
    manifest_path: Path, bootstrap: Path, destination: Path, expected: str
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema_version") != SCHEMA
        or manifest.get("aggregate_sha256") != expected
        or manifest.get("aggregate_sha256") != sha256(canonical(manifest.get("objects")))
        or manifest.get("release_included") is not False
        or manifest.get("launch_authorized") is not False
    ):
        raise ValueError("final duplicate mounted package manifest drifted")
    destination.mkdir(mode=0o700, parents=True, exist_ok=False)
    for path in (destination / "evals/__init__.py", destination / "evals/fleet/__init__.py"):
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        path.write_bytes(b"")
    for obj in manifest["objects"]:
        body = {key: value for key, value in obj.items() if key != "payload_sha256"}
        if obj.get("payload_sha256") != sha256(canonical(body)):
            raise ValueError("final duplicate mounted package object drifted")
        for entry in obj["entries"]:
            raw = (bootstrap / entry["data_key"]).read_bytes()
            if len(raw) != entry["bytes"] or sha256(raw) != entry["sha256"]:
                raise ValueError("final duplicate mounted package payload drifted")
            target = destination / entry["source_path"]
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            target.write_bytes(raw)
    return {"manifest": manifest, "root": destination}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preview", "verify-mounted"))
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--bootstrap", type=Path)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--expected")
    args = parser.parse_args(argv)
    if args.command == "preview":
        built = build_package(args.repo.resolve(strict=True))
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
    if (
        args.manifest is None
        or args.bootstrap is None
        or args.destination is None
        or args.expected is None
    ):
        parser.error(
            "verify-mounted requires --manifest, --bootstrap, --destination, and --expected"
        )
    value = verify_mounted(args.manifest, args.bootstrap, args.destination, args.expected)
    print(json.dumps({"status": "VALID", "root": str(value["root"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
