"""Split immutable held package for the optimized Generation-8 fallback."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from evals.fleet import autocontinue_generation3_executable_package as base
from evals.fleet import autocontinue_generation7_authority_package_v1 as prior
from evals.fleet import autocontinue_generation8_optimized_v1 as generation8

SCHEMA = "fleet-opencode-autocontinue-generation8-split-package-v1"
CORE_A_NAME = "chris-ac-g8-runtime-core-a-v1"
CORE_B_NAME = "chris-ac-g8-runtime-core-b-v1"
MODEL_NAMES = {
    model: f"chris-{row['short']}-ac-r{row['rank']:03d}-a1-g8-run-v1"
    for model, row in generation8.MODELS.items()
}
CORE_A_PATHS = prior.CORE_A_PATHS
CORE_B_PATHS = prior.CORE_B_PATHS
G8_PATHS = (
    *prior.G7_PATHS,
    generation8.MODULE_PATH,
    generation8.PACKAGE_PATH,
    *(row["spec"] for row in generation8.MODELS.values()),
    *(row["plan"] for row in generation8.MODELS.values()),
    generation8.HELD_PATH,
    generation8.MANIFEST_PATH,
    generation8.RUN_PATH,
    generation8.SUBMIT_PATH,
    generation8.DOC_PATH,
)


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def sha256(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def data_key(path: str) -> str:
    if path == generation8.RUN_PATH:
        return "run-g8-v1.sh"
    return "f-" + hashlib.sha256(path.encode()).hexdigest()[:24]


def _payload(root: Path, paths: tuple[str, ...]) -> tuple[dict[str, str], list[dict[str, Any]]]:
    data: dict[str, str] = {}
    entries = []
    for source_path in paths:
        path = root / source_path
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Generation-8 package source is unsafe: {source_path}")
        raw = path.read_bytes()
        key = data_key(source_path)
        if key in data:
            raise ValueError("Generation-8 package data key collision")
        data[key] = raw.decode()
        entries.append(
            {"data_key": key, "source_path": source_path, "bytes": len(raw), "sha256": sha256(raw)}
        )
    return data, entries


def build_package(root: Path) -> dict[str, Any]:
    payloads: dict[str, dict[str, str]] = {}
    objects: dict[str, dict[str, Any]] = {}
    for name, paths in ((CORE_A_NAME, CORE_A_PATHS), (CORE_B_NAME, CORE_B_PATHS)):
        payloads[name], entries = _payload(root, paths)
        objects[name] = base.object_manifest(name, entries)
    for _model, name in MODEL_NAMES.items():
        payloads[name], entries = _payload(root, G8_PATHS)
        objects[name] = base.object_manifest(name, entries)
    manifests = {}
    for model, name in MODEL_NAMES.items():
        spec = json.loads((root / generation8.MODELS[model]["spec"]).read_text())
        plan = json.loads((root / generation8.MODELS[model]["plan"]).read_text())
        selected = [objects[CORE_A_NAME], objects[CORE_B_NAME], objects[name]]
        manifest = {
            "schema_version": SCHEMA,
            "model": model,
            "generation8_spec_sha256": spec["generation8_spec_sha256"],
            "rendered_plan_sha256": plan["plan_sha256"],
            "held_receipt_sha256": json.loads((root / generation8.HELD_PATH).read_text())[
                "receipt_sha256"
            ],
            "objects": selected,
            "aggregate_sha256": sha256(canonical(selected)),
            "release_included": False,
            "launch_authorized": False,
        }
        payloads[name]["package-manifest.json"] = canonical(manifest).decode() + "\n"
        payloads[name]["package_aggregate_sha256"] = manifest["aggregate_sha256"]
        manifests[model] = manifest
    configmaps = {name: base.configmap(name, data) for name, data in payloads.items()}
    sizes = {name: base.configmap_size(value) for name, value in configmaps.items()}
    if any(size >= base.PACKAGE_OBJECT_LIMIT for size in sizes.values()):
        raise ValueError("Generation-8 package exceeds ConfigMap safety budget")
    all_objects = [objects[name] for name in sorted(objects)]
    return {
        "schema_version": SCHEMA,
        "configmaps": configmaps,
        "object_manifests": objects,
        "model_manifests": manifests,
        "object_json_bytes": sizes,
        "aggregate_sha256": sha256(canonical(all_objects)),
        "release_included": False,
        "launch_authorized": False,
    }


def released_configmaps(
    root: Path,
    releases: dict[str, tuple[dict[str, Any], bytes]],
    tombstone: dict[str, Any],
    package_commit: str,
    launch_route: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Add external releases without changing the held package authority."""
    from evals.fleet import immutable_submission_snapshot as snapshot

    snapshot.assert_stable(root, package_commit)
    built = build_package(root)
    configmaps = json.loads(json.dumps(built["configmaps"]))
    ledger, held, static = generation8._all_static(root)  # noqa: SLF001
    generation8.validate_tombstone(tombstone, static)
    if set(releases) != set(MODEL_NAMES):
        raise ValueError("both Generation-8 releases are required")
    for model, (release, raw) in releases.items():
        spec, plan = static[model]
        manifest = built["model_manifests"][model]
        expected = generation8._build_release_from_validated(  # noqa: SLF001
            model, static, held, tombstone, manifest, package_commit
        )
        generation8.validate_release(release, expected)
        if raw != canonical(release) + b"\n":
            raise ValueError("Generation-8 raw release bytes drifted")
        name = MODEL_NAMES[model]
        configmaps[name]["data"].update(
            {
                "release.json": raw.decode(),
                "release_file_sha256": sha256(raw),
                "g7-preclaim-stop.json": canonical(tombstone).decode() + "\n",
                "package_commit": package_commit,
                "launch-route.json": canonical(launch_route).decode() + "\n",
            }
        )
        if base.configmap_size(configmaps[name]) >= base.PACKAGE_OBJECT_LIMIT:
            raise ValueError("released Generation-8 ConfigMap exceeds safety budget")
    if any(count != 1 for count in ledger.read_counts.values()):
        raise RuntimeError("Generation-8 release validation was not exactly once")
    return configmaps


def verify_mounted(manifest_path: Path, bootstrap: Path) -> dict[str, Any]:
    value = json.loads(manifest_path.read_text())
    if (
        value.get("schema_version") != SCHEMA
        or value.get("release_included") is not False
        or value.get("launch_authorized") is not False
        or value.get("aggregate_sha256") != sha256(canonical(value.get("objects")))
    ):
        raise ValueError("Generation-8 mounted package manifest drifted")
    for obj in value["objects"]:
        unsigned = {key: item for key, item in obj.items() if key != "payload_sha256"}
        if obj.get("payload_sha256") != sha256(canonical(unsigned)):
            raise ValueError("Generation-8 mounted object manifest drifted")
        for entry in obj["entries"]:
            raw = (bootstrap / entry["data_key"]).read_bytes()
            if len(raw) != entry["bytes"] or sha256(raw) != entry["sha256"]:
                raise ValueError("Generation-8 mounted payload drifted")
    return value


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preview", "verify-mounted"))
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--bootstrap", type=Path)
    args = parser.parse_args()
    if args.command == "preview":
        value = build_package(args.repo.resolve())
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
