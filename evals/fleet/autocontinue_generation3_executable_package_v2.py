"""Semantic, split-ConfigMap package gate for held generation-3 canaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evals.fleet import autocontinue_generation3_canary as generation3
from evals.fleet import autocontinue_generation3_executable_package as prior

SCHEMA = "fleet-opencode-autocontinue-generation3-split-package-v2"
HELD_SCHEMA = "fleet-opencode-autocontinue-generation3-split-package-held-v2"
KUBERNETES_OBJECT_LIMIT = prior.KUBERNETES_OBJECT_LIMIT
PACKAGE_OBJECT_LIMIT = prior.PACKAGE_OBJECT_LIMIT

MODULE_PATH = "evals/fleet/autocontinue_generation3_executable_package_v2.py"
MANIFEST_PATH = (
    "evals/fleet/cluster/opencode-autocontinue-generation3-executable-held-v3.yaml"
)
RUN_PATH = "evals/fleet/scripts/run_opencode_autocontinue_generation3_executable_v3.sh"
SUBMIT_PATH = (
    "evals/fleet/scripts/submit_opencode_autocontinue_generation3_executable_v3.sh"
)
HELD_PATH = (
    "docs/evidence/qwen38-study/"
    "2026-09-04-opencode-autocontinue-generation3-executable-package-held-v3.json"
)

CORE_A_NAME = "chris-ac-g3-runtime-core-a-v3"
CORE_B_NAME = "chris-ac-g3-runtime-core-b-v3"
MODEL_NAMES = dict(prior.MODEL_NAMES)
CORE_A_PATHS = tuple(path for path in prior.CORE_A_PATHS if path != prior.RUN_PATH) + (
    RUN_PATH,
)
CORE_B_PATHS = prior.CORE_B_PATHS + (MODULE_PATH,)
MODEL_PATHS = dict(prior.MODEL_PATHS)


def canonical(value: Any) -> bytes:
    return prior.canonical(value)


def sha256(data: bytes) -> str:
    return prior.sha256(data)


def file_sha256(path: Path) -> str:
    return prior.file_sha256(path)


def validate_generation3_chain(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    incident = generation3.load(root / generation3.INCIDENT_PATH)
    tombstones = generation3.load(root / generation3.TOMBSTONE_PATH)
    generation3.validate_incident(incident, root)
    generation3.validate_tombstones(tombstones, root)
    specs = [
        generation3.load(root / generation3.G3_SPEC_PATHS[model])
        for model in ("qwen3.8-27b", "glm-5.3")
    ]
    plans = [generation3.validate_spec(spec, root) for spec in specs]
    generation3.validate_held(generation3.load(root / generation3.HELD_PATH), root)
    return specs, plans


def build_package(root: Path) -> dict[str, Any]:
    specs, plans = validate_generation3_chain(root)
    payloads: dict[str, dict[str, str]] = {}
    objects: dict[str, dict[str, Any]] = {}
    for name, paths in ((CORE_A_NAME, CORE_A_PATHS), (CORE_B_NAME, CORE_B_PATHS)):
        payloads[name], entries = prior._payload(root, paths)
        objects[name] = prior.object_manifest(name, entries)
    for model, name in MODEL_NAMES.items():
        payloads[name], entries = prior._payload(root, MODEL_PATHS[model])
        objects[name] = prior.object_manifest(name, entries)

    model_manifests: dict[str, dict[str, Any]] = {}
    for model, model_name in MODEL_NAMES.items():
        selected = [objects[CORE_A_NAME], objects[CORE_B_NAME], objects[model_name]]
        manifest = {
            "schema_version": SCHEMA,
            "model": model,
            "generation3_spec_sha256": next(
                spec["generation3_spec_sha256"] for spec in specs if spec["model"] == model
            ),
            "rendered_plan_sha256": next(
                plan["plan_sha256"]
                for spec, plan in zip(specs, plans, strict=True)
                if spec["model"] == model
            ),
            "incident_receipt_sha256": generation3.INCIDENT_SHA,
            "tombstone_receipt_sha256": generation3.TOMBSTONE_SHA,
            "generation3_held_receipt_sha256": generation3.load(
                root / generation3.HELD_PATH
            )["receipt_sha256"],
            "objects": selected,
            "aggregate_sha256": sha256(canonical(selected)),
            "launch_authorized": False,
        }
        payloads[model_name]["package-manifest.json"] = (
            canonical(manifest).decode() + "\n"
        )
        payloads[model_name]["package_aggregate_sha256"] = manifest[
            "aggregate_sha256"
        ]
        model_manifests[model] = manifest

    configmaps = {
        name: prior.configmap(name, data) for name, data in payloads.items()
    }
    sizes = {name: prior.configmap_size(value) for name, value in configmaps.items()}
    if any(size >= KUBERNETES_OBJECT_LIMIT for size in sizes.values()):
        raise ValueError("semantic split package exceeds the Kubernetes object limit")
    if any(size >= PACKAGE_OBJECT_LIMIT for size in sizes.values()):
        raise ValueError("semantic split package exceeds its ConfigMap safety budget")
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


def validate_held(receipt: dict[str, Any], root: Path) -> dict[str, Any]:
    package = build_package(root)
    generation3_held = generation3.load(root / generation3.HELD_PATH)
    expected = {
        "schema_version": HELD_SCHEMA,
        "append_only": True,
        "status": "HELD",
        "execution_generation": 3,
        "supersedes_split_package": {
            "path": prior.HELD_PATH,
            "receipt_sha256": json.loads((root / prior.HELD_PATH).read_text())[
                "receipt_sha256"
            ],
        },
        "semantic_evidence": {
            "incident_path": generation3.INCIDENT_PATH,
            "incident_receipt_sha256": generation3.INCIDENT_SHA,
            "tombstone_path": generation3.TOMBSTONE_PATH,
            "tombstone_receipt_sha256": generation3.TOMBSTONE_SHA,
            "generation3_held_path": generation3.HELD_PATH,
            "generation3_held_receipt_sha256": generation3_held["receipt_sha256"],
            "both_specs_and_rendered_plans_validated": True,
        },
        "package_layout": "split_immutable_configmaps_projected_volume_v3",
        "objects": sorted(package["configmaps"]),
        "object_json_bytes": package["object_json_bytes"],
        "kubernetes_object_limit_bytes": KUBERNETES_OBJECT_LIMIT,
        "package_object_safety_limit_bytes": PACKAGE_OBJECT_LIMIT,
        "aggregate_sha256": package["aggregate_sha256"],
        "model_aggregate_sha256s": {
            model: value["aggregate_sha256"]
            for model, value in package["model_manifests"].items()
        },
        "jobs": [
            generation3.EXPECTED[model]["job_name"]
            for model in ("qwen3.8-27b", "glm-5.3")
        ],
        "priority": {"class": "fleet-serve-low", "preemption_policy": "Never"},
        "implementation": {
            "module_path": MODULE_PATH,
            "module_sha256": file_sha256(root / MODULE_PATH),
            "manifest_path": MANIFEST_PATH,
            "manifest_sha256": file_sha256(root / MANIFEST_PATH),
            "run_path": RUN_PATH,
            "run_sha256": file_sha256(root / RUN_PATH),
            "submit_path": SUBMIT_PATH,
            "submit_sha256": file_sha256(root / SUBMIT_PATH),
        },
        "launch_authorized": False,
        "objects_created": False,
        "remaining_gates": [
            "fresh_route_and_duplicate_inventory",
            "append_only_model_specific_generation3_scoring_releases",
            "explicit_root_launch_authorization",
        ],
        "privacy": {
            "prompts_traces_flags_or_scores_included": False,
            "credentials_included": False,
        },
    }
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if unsigned != expected or receipt.get("receipt_sha256") != sha256(canonical(unsigned)):
        raise ValueError("generation-3 semantic split package held receipt drifted")
    return package


def verify_mounted(manifest_path: Path, bootstrap: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema_version") != SCHEMA
        or manifest.get("launch_authorized") is not False
        or manifest.get("incident_receipt_sha256") != generation3.INCIDENT_SHA
        or manifest.get("tombstone_receipt_sha256") != generation3.TOMBSTONE_SHA
        or manifest.get("aggregate_sha256")
        != sha256(canonical(manifest.get("objects")))
    ):
        raise ValueError("mounted semantic package manifest drifted")
    for obj in manifest["objects"]:
        if obj.get("payload_sha256") != sha256(
            canonical({key: value for key, value in obj.items() if key != "payload_sha256"})
        ):
            raise ValueError("mounted semantic object manifest drifted")
        for entry in obj["entries"]:
            raw = (bootstrap / entry["data_key"]).read_bytes()
            if len(raw) != entry["bytes"] or sha256(raw) != entry["sha256"]:
                raise ValueError("mounted semantic package payload drifted")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preview", "validate-held", "verify-mounted"))
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--held", type=Path)
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
                    "semantic_evidence_validated": True,
                    "launch_authorized": False,
                    "objects_created": False,
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "validate-held":
        if not args.held:
            parser.error("validate-held requires --held")
        validate_held(json.loads(args.held.read_text()), args.repo.resolve())
        return 0
    if not args.manifest or not args.bootstrap:
        parser.error("verify-mounted requires --manifest and --bootstrap")
    verify_mounted(args.manifest, args.bootstrap)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
