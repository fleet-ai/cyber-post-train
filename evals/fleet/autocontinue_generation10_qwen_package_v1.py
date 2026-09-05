"""Split immutable package for the held Qwen Generation-10 canary."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from evals.fleet import autocontinue_generation3_executable_package as base
from evals.fleet import autocontinue_generation8_package_v1 as generation8_package
from evals.fleet import autocontinue_generation10_qwen_v1 as generation10

SCHEMA = "fleet-opencode-autocontinue-generation10-qwen-split-package-v1"
CORE_A_NAME = "chris-q38-ac-g10-runtime-core-a-v1"
CORE_B_NAME = "chris-q38-ac-g10-runtime-core-b-v1"
MODEL_NAME = generation10.CONFIGMAP_NAME
CORE_A_PATHS = generation8_package.CORE_A_PATHS
CORE_B_PATHS = generation8_package.CORE_B_PATHS
QWEN_PATHS = (
    *generation8_package.G8_PATHS,
    generation10.MODULE_PATH,
    generation10.PACKAGE_PATH,
    generation10.RUN_PATH,
    generation10.SUBMIT_PATH,
    generation10.MANIFEST_PATH,
    generation10.HELD_PATH,
    generation10.TOMBSTONE_PATH,
    generation10.DIAGNOSIS_PATH,
)


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def sha256(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def data_key(path: str) -> str:
    if path == generation10.RUN_PATH:
        return "run-g10-qwen-v1.sh"
    return "f-" + hashlib.sha256(path.encode()).hexdigest()[:24]


def _payload(root: Path, paths: tuple[str, ...]) -> tuple[dict[str, str], list[dict[str, Any]]]:
    data: dict[str, str] = {}
    entries = []
    for source_path in paths:
        path = root / source_path
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Generation-10 package source is unsafe: {source_path}")
        raw = path.read_bytes()
        key = data_key(source_path)
        if key in data:
            raise ValueError("Generation-10 package data key collision")
        data[key] = raw.decode()
        entries.append(
            {"data_key": key, "source_path": source_path, "bytes": len(raw), "sha256": sha256(raw)}
        )
    return data, entries


def build_package(root: Path) -> dict[str, Any]:
    spec, plan, held = generation10.static(root)
    payloads: dict[str, dict[str, str]] = {}
    objects: dict[str, dict[str, Any]] = {}
    for name, paths in (
        (CORE_A_NAME, CORE_A_PATHS),
        (CORE_B_NAME, CORE_B_PATHS),
        (MODEL_NAME, QWEN_PATHS),
    ):
        payloads[name], entries = _payload(root, paths)
        objects[name] = base.object_manifest(name, entries)
    selected = [objects[CORE_A_NAME], objects[CORE_B_NAME], objects[MODEL_NAME]]
    manifest = {
        "schema_version": SCHEMA,
        "model": generation10.MODEL,
        "generation10_spec_sha256": spec["generation10_spec_sha256"],
        "rendered_plan_sha256": plan["plan_sha256"],
        "held_receipt_sha256": held["receipt_sha256"],
        "objects": selected,
        "aggregate_sha256": sha256(canonical(selected)),
        "release_included": False,
        "launch_authorized": False,
    }
    payloads[MODEL_NAME]["package-manifest.json"] = canonical(manifest).decode() + "\n"
    payloads[MODEL_NAME]["package_aggregate_sha256"] = manifest["aggregate_sha256"]
    configmaps = {name: base.configmap(name, data) for name, data in payloads.items()}
    sizes = {name: base.configmap_size(value) for name, value in configmaps.items()}
    if any(size >= base.PACKAGE_OBJECT_LIMIT for size in sizes.values()):
        raise ValueError("Generation-10 package exceeds ConfigMap safety budget")
    return {
        "schema_version": SCHEMA,
        "configmaps": configmaps,
        "object_manifests": objects,
        "model_manifest": manifest,
        "object_json_bytes": sizes,
        "aggregate_sha256": sha256(canonical([objects[name] for name in sorted(objects)])),
        "release_included": False,
        "launch_authorized": False,
    }


def released_configmaps(
    root: Path,
    release: dict[str, Any],
    release_raw: bytes,
    duplicate: dict[str, Any],
    duplicate_raw: bytes,
    package_commit: str,
    launch_route: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    built = build_package(root)
    generation10.validate_release(
        release,
        root,
        built["model_manifest"],
        package_commit,
        duplicate,
        launch_route,
    )
    if release_raw != canonical(release) + b"\n" or duplicate_raw != canonical(duplicate) + b"\n":
        raise ValueError("Generation-10 external receipt bytes drifted")
    configmaps = json.loads(json.dumps(built["configmaps"]))
    data = configmaps[MODEL_NAME]["data"]
    data.update(
        {
            "release.json": release_raw.decode(),
            "release_file_sha256": sha256(release_raw),
            "duplicate-preflight.json": duplicate_raw.decode(),
            "duplicate_preflight_file_sha256": sha256(duplicate_raw),
            "package_commit": package_commit,
            "launch-route.json": canonical(launch_route).decode() + "\n",
        }
    )
    if base.configmap_size(configmaps[MODEL_NAME]) >= base.PACKAGE_OBJECT_LIMIT:
        raise ValueError("released Generation-10 ConfigMap exceeds safety budget")
    return configmaps


def verify_mounted(manifest_path: Path, bootstrap: Path) -> dict[str, Any]:
    value = json.loads(manifest_path.read_text())
    if (
        value.get("schema_version") != SCHEMA
        or value.get("release_included") is not False
        or value.get("launch_authorized") is not False
        or value.get("aggregate_sha256") != sha256(canonical(value.get("objects")))
    ):
        raise ValueError("Generation-10 mounted package manifest drifted")
    for obj in value["objects"]:
        unsigned = {key: item for key, item in obj.items() if key != "payload_sha256"}
        if obj.get("payload_sha256") != sha256(canonical(unsigned)):
            raise ValueError("Generation-10 mounted object manifest drifted")
        for entry in obj["entries"]:
            raw = (bootstrap / entry["data_key"]).read_bytes()
            if len(raw) != entry["bytes"] or sha256(raw) != entry["sha256"]:
                raise ValueError("Generation-10 mounted payload drifted")
    return value


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preview", "render-package-manifest", "verify-mounted"))
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--bootstrap", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command in {"preview", "render-package-manifest"}:
        value = build_package(args.repo.resolve())
        if args.command == "render-package-manifest":
            if not args.output:
                parser.error("render-package-manifest requires an unused output")
            from evals.fleet import self_hosted

            self_hosted.write_json_once(args.output, value["model_manifest"])
            return 0
        print(
            json.dumps(
                {
                    "status": "HELD",
                    "launch_authorized": False,
                    "objects_created": False,
                    "aggregate_sha256": value["aggregate_sha256"],
                    "object_json_bytes": value["object_json_bytes"],
                },
                sort_keys=True,
            )
        )
        return 0
    if not args.manifest or not args.bootstrap:
        parser.error("verify-mounted requires manifest and bootstrap")
    verify_mounted(args.manifest, args.bootstrap)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
