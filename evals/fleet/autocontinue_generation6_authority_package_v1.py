"""Split immutable ConfigMap package for authority-bound generation-6 canaries."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from evals.fleet import autocontinue_generation3_executable_package as base
from evals.fleet import autocontinue_generation5_authority_package_v1 as prior
from evals.fleet import autocontinue_generation6_authority_v1 as authority
from evals.fleet import autocontinue_generation6_canary as generation6

SCHEMA = "fleet-opencode-autocontinue-generation6-authority-split-package-v1"
CORE_A_NAME = "chris-ac-g6-runtime-core-a-v1"
CORE_B_NAME = "chris-ac-g6-runtime-core-b-v1"
MODEL_NAMES = {
    model: generation6.EXPECTED[model]["configmap_name"] for model in generation6.EXPECTED
}
KUBERNETES_OBJECT_LIMIT = base.KUBERNETES_OBJECT_LIMIT
PACKAGE_OBJECT_LIMIT = base.PACKAGE_OBJECT_LIMIT
CORE_A_PATHS = prior.CORE_A_PATHS
CORE_B_PATHS = prior.CORE_B_PATHS
G6_PATHS = (
    generation6.MODULE_PATH,
    generation6.FAILURE_PATH,
    *generation6.G6_SPEC_PATHS.values(),
    authority.MODULE_PATH,
    authority.PACKAGE_MODULE_PATH,
    authority.AUTH_PATH,
    authority.MANIFEST_PATH,
    authority.RUN_PATH,
    authority.SUBMIT_PATH,
)
MODEL_PATHS = {model: G6_PATHS for model in MODEL_NAMES}


def canonical(value: Any) -> bytes:
    return base.canonical(value)


def sha256(data: bytes) -> str:
    return base.sha256(data)


def file_sha256(path: Path) -> str:
    return base.file_sha256(path)


def data_key(path: str) -> str:
    if path == authority.RUN_PATH:
        return "run-g6-v1.sh"
    return "f-" + hashlib.sha256(path.encode()).hexdigest()[:24]


def _payload(root: Path, paths: tuple[str, ...]) -> tuple[dict[str, str], list[dict[str, Any]]]:
    data: dict[str, str] = {}
    entries: list[dict[str, Any]] = []
    for source_path in paths:
        raw = (root / source_path).read_bytes()
        key = data_key(source_path)
        if key in data:
            raise ValueError(f"duplicate generation-6 package key: {key}")
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


def build_package(root: Path) -> dict[str, Any]:
    authority_receipt = authority.load(root / authority.AUTH_PATH)
    plans = authority.validate_root_authorization(authority_receipt, root)
    selected_specs = authority.specs(root)
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
        spec = next(row for row in selected_specs if row["model"] == model)
        plan = next(
            plan for row, plan in zip(selected_specs, plans, strict=True)
            if row["model"] == model
        )
        manifest = {
            "schema_version": SCHEMA,
            "model": model,
            "generation6_spec_sha256": spec["generation6_spec_sha256"],
            "rendered_plan_sha256": plan["plan_sha256"],
            "root_authorization_receipt_sha256": authority_receipt["receipt_sha256"],
            "generation5_failure_receipt_sha256": authority.load(
                root / generation6.FAILURE_PATH
            )["receipt_sha256"],
            "objects": selected,
            "aggregate_sha256": sha256(canonical(selected)),
            "release_included": False,
            "launch_authorized": False,
        }
        payloads[model_name]["package-manifest.json"] = canonical(manifest).decode() + "\n"
        payloads[model_name]["package_aggregate_sha256"] = manifest["aggregate_sha256"]
        model_manifests[model] = manifest
    configmaps = {name: base.configmap(name, data) for name, data in payloads.items()}
    sizes = {name: base.configmap_size(value) for name, value in configmaps.items()}
    if any(size >= KUBERNETES_OBJECT_LIMIT for size in sizes.values()):
        raise ValueError("generation-6 package exceeds Kubernetes object limit")
    if any(size >= PACKAGE_OBJECT_LIMIT for size in sizes.values()):
        raise ValueError("generation-6 package exceeds ConfigMap safety budget")
    all_objects = [objects[name] for name in sorted(objects)]
    return {
        "schema_version": SCHEMA,
        "configmaps": configmaps,
        "object_manifests": objects,
        "model_manifests": model_manifests,
        "object_json_bytes": sizes,
        "aggregate_sha256": sha256(canonical(all_objects)),
        "release_included": False,
        "launch_authorized": False,
    }


def released_configmaps(
    root: Path,
    releases: dict[str, tuple[dict[str, Any], str]],
    package_commit: str,
    launch_route: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    built = build_package(root)
    configmaps = json.loads(json.dumps(built["configmaps"]))
    auth = authority.load(root / authority.AUTH_PATH)
    if set(releases) != set(MODEL_NAMES):
        raise ValueError("both generation-6 model releases are required")
    for model, (release, file_digest) in releases.items():
        spec = authority.load(root / generation6.G6_SPEC_PATHS[model])
        authority.validate_release(release, spec, auth, root, package_commit)
        raw = (root / authority.RELEASE_PATHS[model]).read_bytes()
        if sha256(raw) != file_digest or json.loads(raw) != release:
            raise ValueError("generation-6 raw release bytes do not match binding")
        name = MODEL_NAMES[model]
        configmaps[name]["data"].update(
            {
                "release.json": raw.decode("utf-8"),
                "release_file_sha256": file_digest,
                "package_commit": package_commit,
                "launch-route.json": canonical(launch_route).decode() + "\n",
            }
        )
        if base.configmap_size(configmaps[name]) >= PACKAGE_OBJECT_LIMIT:
            raise ValueError("released generation-6 ConfigMap exceeds safety budget")
    return configmaps


def verify_mounted(manifest_path: Path, bootstrap: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema_version") != SCHEMA
        or manifest.get("launch_authorized") is not False
        or manifest.get("release_included") is not False
        or manifest.get("aggregate_sha256") != sha256(canonical(manifest.get("objects")))
    ):
        raise ValueError("mounted generation-6 package manifest drifted")
    for obj in manifest["objects"]:
        unsigned = {key: value for key, value in obj.items() if key != "payload_sha256"}
        if obj.get("payload_sha256") != sha256(canonical(unsigned)):
            raise ValueError("mounted generation-6 object manifest drifted")
        for entry in obj["entries"]:
            raw = (bootstrap / entry["data_key"]).read_bytes()
            if len(raw) != entry["bytes"] or sha256(raw) != entry["sha256"]:
                raise ValueError("mounted generation-6 payload drifted")
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
        print(json.dumps({
            "ok": True,
            "status": "HELD",
            "aggregate_sha256": package["aggregate_sha256"],
            "object_json_bytes": package["object_json_bytes"],
            "launch_authorized": False,
            "objects_created": False,
        }, sort_keys=True))
        return 0
    if not args.manifest or not args.bootstrap:
        parser.error("verify-mounted requires manifest and bootstrap")
    verify_mounted(args.manifest, args.bootstrap)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
