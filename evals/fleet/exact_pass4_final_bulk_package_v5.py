"""Build the immutable held package for phased final-v5 controllers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evals.fleet import exact_pass4_bulk_package_v3 as prior
from evals.fleet import exact_pass4_final_bulk_v5 as bulk

SCHEMA = "fleet-exact-pass4-final-split-package-v5"
CORE_NAMES = tuple(f"chris-exact-pass4-final-runtime-core-{letter}-v5" for letter in "abcd")
CORE_A_PATHS = prior.CORE_A_PATHS
CORE_B_PATHS = prior.CORE_B_PATHS
CORE_C_PATHS = prior.CORE_C_PATHS
CORE_D_PATHS = tuple(
    dict.fromkeys(
        (
            bulk.MODULE_PATH,
            bulk.PACKAGE_PATH,
            bulk.RENDER_PATH,
            bulk.DEDICATED_EVIDENCE_PATH,
            bulk.HELD_PATH,
            bulk.DOC_PATH,
            bulk.MANIFEST_PATH,
            "evals/fleet/exact_pass4_hosted_c4_bulk_v4.py",
            "evals/fleet/exact_pass4_dedicated_bulk_v4.py",
            "evals/fleet/exact_pass4_prebulk_reconciliation_v4.py",
            "evals/fleet/glm53_dedicated_v7.py",
            "evals/fleet/glm53_dedicated_v6.py",
            "evals/fleet/scripts/glm53_dedicated_v7_controller_heartbeat.sh",
            "evals/fleet/configs/exact-pass4-dedicated-bulk-v4-held.json",
            "evals/fleet/configs/glm53-dedicated-serving-v7-held.json",
            "docs/evidence/qwen38-study/2026-09-05-glm53-dedicated-serving-v7-jobs-api-observation.json",
        )
    )
)
CORE_PATH_GROUPS = dict(
    zip(CORE_NAMES, (CORE_A_PATHS, CORE_B_PATHS, CORE_C_PATHS, CORE_D_PATHS), strict=True)
)
CONTROLLER_NAMES = {key: row["configmap_name"] for key, row in bulk.CONTROLLERS.items()}
CONTROLLER_PATHS = {key: (bulk.RUNTIME_PATH, bulk.RUN_PATH) for key in bulk.CONTROLLERS}


def canonical(value: Any) -> bytes:
    return prior.canonical(value)


def sha256(value: bytes) -> str:
    return prior.sha256(value)


def data_key(path: str) -> str:
    return prior.data_key(path)


def build_package(root: Path) -> dict[str, Any]:
    plans = bulk.validate_all(root)
    bulk.validate_held(bulk.load(root / bulk.HELD_PATH), root)
    payloads: dict[str, dict[str, str]] = {}
    objects: dict[str, dict[str, Any]] = {}
    for name, paths in CORE_PATH_GROUPS.items():
        payloads[name], entries = prior._payload(root, paths)  # noqa: SLF001
        objects[name] = prior._object(name, entries)  # noqa: SLF001
    for controller, name in CONTROLLER_NAMES.items():
        payloads[name], entries = prior._payload(root, CONTROLLER_PATHS[controller])  # noqa: SLF001
        objects[name] = prior._object(name, entries)  # noqa: SLF001
    manifests = {}
    for controller, name in CONTROLLER_NAMES.items():
        selected = [*(objects[core] for core in CORE_NAMES), objects[name]]
        manifest = {
            "schema_version": SCHEMA,
            "controller": controller,
            "plan_sha256": plans[controller]["plan_sha256"],
            "objects": selected,
            "aggregate_sha256": sha256(canonical(selected)),
            "release_included": False,
            "launch_authorized": False,
        }
        payloads[name]["package-manifest.json"] = canonical(manifest).decode() + "\n"
        payloads[name]["package_aggregate_sha256"] = manifest["aggregate_sha256"]
        manifests[controller] = manifest
    configmaps = {
        name: prior._configmap(name, data)  # noqa: SLF001
        for name, data in payloads.items()
    }
    sizes = {name: len(canonical(value)) for name, value in configmaps.items()}
    if any(size >= prior.PACKAGE_OBJECT_LIMIT for size in sizes.values()):
        raise ValueError("final v5 package exceeds safety budget")
    all_objects = [objects[name] for name in sorted(objects)]
    return {
        "schema_version": SCHEMA,
        "configmaps": configmaps,
        "object_manifests": objects,
        "controller_manifests": manifests,
        "object_json_bytes": sizes,
        "aggregate_sha256": sha256(canonical(all_objects)),
        "release_included": False,
        "launch_authorized": False,
    }


def verify_mounted(manifest_path: Path, bootstrap: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != SCHEMA or manifest.get("aggregate_sha256") != sha256(
        canonical(manifest.get("objects"))
    ):
        raise ValueError("final v5 mounted manifest drifted")
    for obj in manifest["objects"]:
        body = {key: value for key, value in obj.items() if key != "payload_sha256"}
        if obj.get("payload_sha256") != sha256(canonical(body)):
            raise ValueError("final v5 mounted object drifted")
        for entry in obj["entries"]:
            raw = (bootstrap / entry["data_key"]).read_bytes()
            if len(raw) != entry["bytes"] or sha256(raw) != entry["sha256"]:
                raise ValueError("final v5 mounted payload drifted")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("preview", nargs="?")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()
    package = build_package(args.repo.resolve())
    print(
        json.dumps(
            {
                "status": "HELD",
                "launch_authorized": False,
                "objects_created": False,
                "aggregate_sha256": package["aggregate_sha256"],
                "object_json_bytes": package["object_json_bytes"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
