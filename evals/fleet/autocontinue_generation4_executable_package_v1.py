"""Split immutable held package for generation-4 canaries."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from evals.fleet import autocontinue_generation3_authority_package_v1 as prior
from evals.fleet import autocontinue_generation3_executable_package as base
from evals.fleet import autocontinue_generation3_executable_package_v2 as semantic
from evals.fleet import autocontinue_generation4_canary as generation4

SCHEMA = "fleet-opencode-autocontinue-generation4-split-package-v1"
CORE_A_NAME = "chris-ac-g4-runtime-core-a-v1"
CORE_B_NAME = "chris-ac-g4-runtime-core-b-v1"
MODEL_NAMES = {
    model: generation4.EXPECTED[model]["configmap_name"] for model in generation4.EXPECTED
}
KUBERNETES_OBJECT_LIMIT = base.KUBERNETES_OBJECT_LIMIT
PACKAGE_OBJECT_LIMIT = base.PACKAGE_OBJECT_LIMIT
CORE_A_PATHS = tuple(path for path in prior.CORE_A_PATHS if path != prior.authority.RUN_PATH) + (
    generation4.RUN_PATH,
)
CORE_B_PATHS = prior.CORE_B_PATHS + (
    generation4.MODULE_PATH,
    generation4.PACKAGE_PATH,
    generation4.INCIDENT_PATH,
    generation4.TOMBSTONE_PATH,
    generation4.HELD_PATH,
    generation4.MANIFEST_PATH,
    generation4.SUBMIT_PATH,
    semantic.HELD_PATH,
    semantic.MANIFEST_PATH,
    semantic.RUN_PATH,
    semantic.SUBMIT_PATH,
)
MODEL_PATHS = {
    model: (generation4.G4_SPEC_PATHS[model],) for model in generation4.EXPECTED
}


def canonical(value: Any) -> bytes:
    return base.canonical(value)


def sha256(data: bytes) -> str:
    return base.sha256(data)


def data_key(path: str) -> str:
    if path == generation4.RUN_PATH:
        return "run-g4-v1.sh"
    return "f-" + hashlib.sha256(path.encode()).hexdigest()[:24]


def _payload(
    root: Path, paths: tuple[str, ...]
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    data: dict[str, str] = {}
    entries: list[dict[str, Any]] = []
    for source_path in paths:
        raw = (root / source_path).read_bytes()
        key = data_key(source_path)
        if key in data:
            raise ValueError(f"duplicate generation-4 package key: {key}")
        data[key] = raw.decode()
        entries.append(
            {
                "data_key": key,
                "source_path": source_path,
                "bytes": len(raw),
                "sha256": sha256(raw),
            }
        )
    return data, entries


def build_package(root: Path) -> dict[str, Any]:
    generation4.validate_incident(generation4.load(root / generation4.INCIDENT_PATH), root)
    generation4.validate_tombstones(
        generation4.load(root / generation4.TOMBSTONE_PATH), root
    )
    held = generation4.load(root / generation4.HELD_PATH)
    generation4.validate_held(held, root)
    specs = [
        generation4.load(root / generation4.G4_SPEC_PATHS[model])
        for model in generation4.EXPECTED
    ]
    plans = [generation4.validate_spec(spec, root) for spec in specs]
    payloads: dict[str, dict[str, str]] = {}
    objects: dict[str, dict[str, Any]] = {}
    for name, paths in ((CORE_A_NAME, CORE_A_PATHS), (CORE_B_NAME, CORE_B_PATHS)):
        payloads[name], entries = _payload(root, paths)
        objects[name] = base.object_manifest(name, entries)
    for model, name in MODEL_NAMES.items():
        payloads[name], entries = _payload(root, MODEL_PATHS[model])
        objects[name] = base.object_manifest(name, entries)
    model_manifests: dict[str, dict[str, Any]] = {}
    for model, model_name in MODEL_NAMES.items():
        selected = [objects[CORE_A_NAME], objects[CORE_B_NAME], objects[model_name]]
        spec = next(spec for spec in specs if spec["model"] == model)
        plan = next(
            plan
            for current, plan in zip(specs, plans, strict=True)
            if current["model"] == model
        )
        manifest = {
            "schema_version": SCHEMA,
            "model": model,
            "generation4_spec_sha256": spec["generation4_spec_sha256"],
            "rendered_plan_sha256": plan["plan_sha256"],
            "incident_receipt_sha256": generation4.INCIDENT_SHA,
            "tombstone_receipt_sha256": generation4.TOMBSTONE_SHA,
            "generation4_held_receipt_sha256": held["receipt_sha256"],
            "objects": selected,
            "aggregate_sha256": sha256(canonical(selected)),
            "launch_authorized": False,
        }
        payloads[model_name]["package-manifest.json"] = canonical(manifest).decode() + "\n"
        payloads[model_name]["package_aggregate_sha256"] = manifest["aggregate_sha256"]
        model_manifests[model] = manifest
    configmaps = {name: base.configmap(name, data) for name, data in payloads.items()}
    sizes = {name: base.configmap_size(value) for name, value in configmaps.items()}
    if any(size >= KUBERNETES_OBJECT_LIMIT for size in sizes.values()):
        raise ValueError("generation-4 package exceeds Kubernetes object limit")
    if any(size >= PACKAGE_OBJECT_LIMIT for size in sizes.values()):
        raise ValueError("generation-4 package exceeds ConfigMap safety budget")
    all_objects = [objects[name] for name in sorted(objects)]
    return {
        "schema_version": SCHEMA,
        "configmaps": configmaps,
        "object_manifests": objects,
        "model_manifests": model_manifests,
        "object_json_bytes": sizes,
        "aggregate_sha256": sha256(canonical(all_objects)),
        "launch_authorized": False,
    }


def verify_mounted(manifest_path: Path, bootstrap: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema_version") != SCHEMA
        or manifest.get("launch_authorized") is not False
        or manifest.get("aggregate_sha256") != sha256(canonical(manifest.get("objects")))
    ):
        raise ValueError("mounted generation-4 package manifest drifted")
    for obj in manifest["objects"]:
        unsigned = {key: value for key, value in obj.items() if key != "payload_sha256"}
        if obj.get("payload_sha256") != sha256(canonical(unsigned)):
            raise ValueError("mounted generation-4 object manifest drifted")
        for entry in obj["entries"]:
            raw = (bootstrap / entry["data_key"]).read_bytes()
            if len(raw) != entry["bytes"] or sha256(raw) != entry["sha256"]:
                raise ValueError("mounted generation-4 payload drifted")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preview", "verify-mounted"))
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--bootstrap", type=Path)
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
                    "execution_generation": 4,
                    "launch_authorized": False,
                    "objects_created": False,
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
