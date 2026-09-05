"""Build the split immutable package for the two Qwen G16 controllers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from evals.fleet import exact_pass4_bulk_package_v3 as prior
from evals.fleet import qwen_bulk_generation16 as bulk

SCHEMA = "fleet-qwen-generation16-bulk-split-package-v1"
CORE_A_NAME = "chris-q38-g16-runtime-core-a-v1"
CORE_B_NAME = "chris-q38-g16-runtime-core-b-v1"
CORE_C_NAME = "chris-q38-g16-runtime-core-c-v1"
CORE_NAMES = (CORE_A_NAME, CORE_B_NAME, CORE_C_NAME)
CORE_A_PATHS = prior.CORE_A_PATHS
CORE_B_PATHS = prior.CORE_B_PATHS
CORE_C_PATHS = tuple(
    dict.fromkeys(
        (
            *prior.CORE_C_PATHS,
            "evals/fleet/exact_pass4_bulk_runtime_v3.py",
            bulk.MODULE_PATH,
            "evals/fleet/qwen_bulk_generation16_preflight.py",
            bulk.G15_GATE_PATH,
            *bulk.SPEC_PATHS.values(),
        )
    )
)
CORE_PATH_GROUPS = {
    CORE_A_NAME: CORE_A_PATHS,
    CORE_B_NAME: CORE_B_PATHS,
    CORE_C_NAME: CORE_C_PATHS,
}
CONTROLLER_NAMES = {key: row["configmap_name"] for key, row in bulk.CONTROLLERS.items()}
CONTROLLER_PATHS = {key: (bulk.RUNTIME_PATH, bulk.RUN_PATH) for key in bulk.CONTROLLERS}
data_key = prior.data_key
canonical = prior.canonical
sha256 = prior.sha256


def build_package(root: Path) -> dict[str, Any]:
    plans = bulk.validate_all(root)
    payloads: dict[str, dict[str, str]] = {}
    objects: dict[str, dict[str, Any]] = {}
    for name, paths in CORE_PATH_GROUPS.items():
        payloads[name], entries = prior._payload(root, paths)  # noqa: SLF001
        objects[name] = prior._object(name, entries)  # noqa: SLF001
    for controller, name in CONTROLLER_NAMES.items():
        payloads[name], entries = prior._payload(  # noqa: SLF001
            root, CONTROLLER_PATHS[controller]
        )
        objects[name] = prior._object(name, entries)  # noqa: SLF001
    controller_manifests: dict[str, dict[str, Any]] = {}
    for controller, name in CONTROLLER_NAMES.items():
        selected = [*(objects[core] for core in CORE_NAMES), objects[name]]
        manifest = {
            "schema_version": SCHEMA,
            "controller": controller,
            "spec_sha256": bulk.load(root / bulk.SPEC_PATHS[controller])["spec_sha256"],
            "plan_sha256": plans[controller]["plan_sha256"],
            "objects": selected,
            "aggregate_sha256": sha256(canonical(selected)),
            "release_included": True,
            "launch_authorized": True,
        }
        payloads[name]["package-manifest.json"] = canonical(manifest).decode() + "\n"
        controller_manifests[controller] = manifest
    configmaps = {name: prior._configmap(name, data) for name, data in payloads.items()}  # noqa: SLF001
    sizes = {name: prior._size(value) for name, value in configmaps.items()}  # noqa: SLF001
    if any(size >= prior.PACKAGE_OBJECT_LIMIT for size in sizes.values()):
        raise ValueError("Generation-16 package exceeds ConfigMap safety budget")
    all_objects = [objects[name] for name in sorted(objects)]
    return {
        "schema_version": SCHEMA,
        "configmaps": configmaps,
        "controller_manifests": controller_manifests,
        "object_json_bytes": sizes,
        "aggregate_sha256": sha256(canonical(all_objects)),
        "launch_authorized": True,
    }
