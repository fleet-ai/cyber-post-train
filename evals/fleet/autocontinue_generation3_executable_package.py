"""Deterministic split-ConfigMap package for the held generation-3 canaries.

This module packages the already-reviewed generation-3 treatment without
authorizing a launch.  It deliberately keeps package publication separate from
the later, model-specific scoring releases.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from evals.fleet import autocontinue_generation3_canary as generation3

SCHEMA = "fleet-opencode-autocontinue-generation3-split-package-v1"
OBJECT_SCHEMA = "fleet-immutable-configmap-payload-v1"
HELD_SCHEMA = "fleet-opencode-autocontinue-generation3-split-package-held-v1"
KUBERNETES_OBJECT_LIMIT = 1_048_576
PACKAGE_OBJECT_LIMIT = 950 * 1024

MODULE_PATH = "evals/fleet/autocontinue_generation3_executable_package.py"
MANIFEST_PATH = "evals/fleet/cluster/opencode-autocontinue-generation3-executable-held-v2.yaml"
RUN_PATH = "evals/fleet/scripts/run_opencode_autocontinue_generation3_executable_v2.sh"
SUBMIT_PATH = "evals/fleet/scripts/submit_opencode_autocontinue_generation3_executable_v2.sh"
HELD_PATH = (
    "docs/evidence/qwen38-study/"
    "2026-09-04-opencode-autocontinue-generation3-executable-package-held-v2.json"
)

CORE_A_NAME = "chris-ac-g3-runtime-core-a-v2"
CORE_B_NAME = "chris-ac-g3-runtime-core-b-v2"
MODEL_NAMES = {
    "qwen3.8-27b": generation3.EXPECTED["qwen3.8-27b"]["configmap_name"],
    "glm-5.3": generation3.EXPECTED["glm-5.3"]["configmap_name"],
}

CORE_A_PATHS = (
    RUN_PATH,
    "evals/fleet/hosted_sweep_controller.py",
    "evals/fleet/self_hosted.py",
    "evals/fleet/opencode_train_sweep_runner.py",
    "evals/fleet/endpoint_lease.py",
    "evals/fleet/fixed_proxy.py",
    "evals/fleet/Dockerfile.opencode",
)

CORE_B_PATHS = (
    "evals/fleet/autocontinue_canary_controller.py",
    "evals/fleet/autocontinue_canary_hosted_release.py",
    "evals/fleet/autocontinue_canary_hosted_runtime.py",
    "evals/fleet/autocontinue_hosted_health.py",
    "evals/fleet/exact_pass4_universe.py",
    "evals/fleet/autocontinue_generation2_canary.py",
    "evals/fleet/autocontinue_generation2_canary_v2.py",
    "evals/fleet/autocontinue_generation2_canary_v3.py",
    "evals/fleet/autocontinue_generation2_runtime_v3.py",
    "evals/fleet/autocontinue_generation2_authority_v4.py",
    "evals/fleet/autocontinue_generation2_authority_v5.py",
    "evals/fleet/autocontinue_generation3_canary.py",
    MODULE_PATH,
    "evals/fleet/configs/q38-glm53-exact-easiest100-pass4-campaign-v1.json",
    "evals/fleet/configs/opencode-easiest-train100-selection-v2.json",
    "evals/fleet/configs/qwen38-opencode-autocontinue-canary1-v2.json",
    "evals/fleet/configs/glm53-opencode-autocontinue-canary1-v2.json",
    "evals/fleet/configs/qwen38-opencode-autocontinue-canary-generation2-v1.json",
    "evals/fleet/configs/glm53-opencode-autocontinue-canary-generation2-v1.json",
    "evals/fleet/configs/qwen38-opencode-autocontinue-canary-generation2-v2.json",
    "evals/fleet/configs/glm53-opencode-autocontinue-canary-generation2-v2.json",
    "evals/fleet/cluster/opencode-autocontinue-generation2-canaries-v1.yaml",
    "evals/fleet/cluster/opencode-autocontinue-generation2-canaries-v2.yaml",
    "evals/fleet/cluster/opencode-autocontinue-generation2-canaries-v3.yaml",
    "evals/fleet/cluster/opencode-autocontinue-generation2-canaries-v4.yaml",
    "evals/fleet/cluster/opencode-autocontinue-generation2-canaries-v5.yaml",
    "evals/fleet/scripts/run_opencode_autocontinue_generation2_canary.sh",
    "evals/fleet/scripts/run_opencode_autocontinue_generation2_canary_v2.sh",
    "evals/fleet/scripts/run_opencode_autocontinue_generation2_canary_v3.sh",
    "evals/fleet/scripts/run_opencode_autocontinue_generation2_canary_v4.sh",
    "evals/fleet/scripts/run_opencode_autocontinue_generation2_canary_v5.sh",
    "evals/fleet/scripts/submit_opencode_autocontinue_generation2_canaries_v1.sh",
    "evals/fleet/scripts/submit_opencode_autocontinue_generation2_canaries_v2.sh",
    "evals/fleet/scripts/submit_opencode_autocontinue_generation2_canaries_v3.sh",
    "evals/fleet/scripts/submit_opencode_autocontinue_generation2_canaries_v4.sh",
    "evals/fleet/scripts/submit_opencode_autocontinue_generation2_canaries_v5.sh",
    generation3.INCIDENT_PATH,
    generation3.TOMBSTONE_PATH,
    generation3.HELD_PATH,
    "docs/evidence/qwen38-study/2026-09-04-opencode-autocontinue-canary-v2-pre-model-failure.json",
    "docs/evidence/qwen38-study/2026-09-04-opencode-autocontinue-canary-generation1-tombstones-v1.json",
    "docs/evidence/qwen38-study/2026-09-04-opencode-autocontinue-generation2-canaries-held-v1.json",
    "docs/evidence/qwen38-study/2026-09-04-opencode-autocontinue-generation2-canaries-held-v2.json",
    "docs/evidence/qwen38-study/2026-09-04-opencode-autocontinue-generation2-canaries-held-v3.json",
    "docs/evidence/qwen38-study/2026-09-04-opencode-autocontinue-generation2-canaries-executable-held-v1.json",
    "docs/evidence/qwen38-study/2026-09-04-opencode-autocontinue-generation2-root-authorization-v1.json",
    "docs/evidence/qwen38-study/2026-09-04-opencode-autocontinue-generation2-authority-bound-held-v1.json",
    "docs/evidence/qwen38-study/2026-09-04-opencode-autocontinue-generation2-authority-bound-held-v2.json",
    "docs/evidence/qwen38-study/2026-09-04-qwen38-autocontinue-generation2-scoring-release-v5.json",
    "docs/evidence/qwen38-study/2026-09-04-glm53-autocontinue-generation2-scoring-release-v5.json",
)

MODEL_PATHS = {model: (generation3.G3_SPEC_PATHS[model],) for model in MODEL_NAMES}


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def data_key(path: str) -> str:
    if path == RUN_PATH:
        return "run-g3-v2.sh"
    return "f-" + hashlib.sha256(path.encode()).hexdigest()[:24]


def _payload(root: Path, paths: tuple[str, ...]) -> tuple[dict[str, str], list[dict[str, Any]]]:
    data: dict[str, str] = {}
    entries: list[dict[str, Any]] = []
    for source_path in paths:
        raw = (root / source_path).read_bytes()
        key = data_key(source_path)
        if key in data:
            raise ValueError(f"duplicate package key: {key}")
        text = raw.decode("utf-8")
        data[key] = text
        entries.append(
            {
                "data_key": key,
                "source_path": source_path,
                "bytes": len(raw),
                "sha256": sha256(raw),
            }
        )
    return data, entries


def object_manifest(name: str, entries: list[dict[str, Any]]) -> dict[str, Any]:
    value = {
        "schema_version": OBJECT_SCHEMA,
        "name": name,
        "immutable": True,
        "entries": entries,
    }
    value["payload_sha256"] = sha256(canonical(value))
    return value


def configmap(name: str, data: dict[str, str]) -> dict[str, Any]:
    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": name, "namespace": "fleet-train-jobs"},
        "immutable": True,
        "data": data,
    }


def configmap_size(value: dict[str, Any]) -> int:
    return len(canonical(value))


def build_package(root: Path) -> dict[str, Any]:
    payloads: dict[str, dict[str, str]] = {}
    objects: dict[str, dict[str, Any]] = {}
    for name, paths in ((CORE_A_NAME, CORE_A_PATHS), (CORE_B_NAME, CORE_B_PATHS)):
        payloads[name], entries = _payload(root, paths)
        objects[name] = object_manifest(name, entries)
    for model, name in MODEL_NAMES.items():
        payloads[name], entries = _payload(root, MODEL_PATHS[model])
        objects[name] = object_manifest(name, entries)

    model_manifests: dict[str, dict[str, Any]] = {}
    for model, model_name in MODEL_NAMES.items():
        selected = [objects[CORE_A_NAME], objects[CORE_B_NAME], objects[model_name]]
        aggregate = sha256(canonical(selected))
        manifest = {
            "schema_version": SCHEMA,
            "model": model,
            "objects": selected,
            "aggregate_sha256": aggregate,
            "launch_authorized": False,
        }
        payloads[model_name]["package-manifest.json"] = canonical(manifest).decode() + "\n"
        payloads[model_name]["package_aggregate_sha256"] = aggregate
        model_manifests[model] = manifest

    configmaps = {name: configmap(name, data) for name, data in payloads.items()}
    sizes = {name: configmap_size(value) for name, value in configmaps.items()}
    if any(size >= KUBERNETES_OBJECT_LIMIT for size in sizes.values()):
        raise ValueError("split package exceeds the Kubernetes ConfigMap object limit")
    if any(size >= PACKAGE_OBJECT_LIMIT for size in sizes.values()):
        raise ValueError("split package exceeds its ConfigMap safety budget")
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


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes())


def validate_held(receipt: dict[str, Any], root: Path) -> dict[str, Any]:
    package = build_package(root)
    expected = {
        "schema_version": HELD_SCHEMA,
        "append_only": True,
        "status": "HELD",
        "execution_generation": 3,
        "package_layout": "split_immutable_configmaps_projected_volume_v2",
        "objects": sorted(package["configmaps"]),
        "object_json_bytes": package["object_json_bytes"],
        "kubernetes_object_limit_bytes": KUBERNETES_OBJECT_LIMIT,
        "package_object_safety_limit_bytes": PACKAGE_OBJECT_LIMIT,
        "aggregate_sha256": package["aggregate_sha256"],
        "model_aggregate_sha256s": {
            model: value["aggregate_sha256"] for model, value in package["model_manifests"].items()
        },
        "jobs": [generation3.EXPECTED[model]["job_name"] for model in ("qwen3.8-27b", "glm-5.3")],
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
        raise ValueError("generation-3 split package held receipt drifted")
    return package


def verify_mounted(manifest_path: Path, bootstrap: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema_version") != SCHEMA
        or manifest.get("launch_authorized") is not False
        or manifest.get("aggregate_sha256") != sha256(canonical(manifest.get("objects")))
    ):
        raise ValueError("mounted package manifest drifted")
    for obj in manifest["objects"]:
        if obj.get("payload_sha256") != sha256(
            canonical({k: v for k, v in obj.items() if k != "payload_sha256"})
        ):
            raise ValueError("mounted object manifest drifted")
        for entry in obj["entries"]:
            raw = (bootstrap / entry["data_key"]).read_bytes()
            if len(raw) != entry["bytes"] or sha256(raw) != entry["sha256"]:
                raise ValueError("mounted package payload drifted")
    return manifest


def install_mounted(manifest: dict[str, Any], bootstrap: Path, destination: Path) -> None:
    for obj in manifest["objects"]:
        for entry in obj["entries"]:
            target = destination / entry["source_path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((bootstrap / entry["data_key"]).read_bytes())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("preview", "validate-held", "verify-mounted", "install-mounted"),
    )
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--bootstrap", type=Path)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--held", type=Path)
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
        package = validate_held(json.loads(args.held.read_text()), args.repo.resolve())
        print(json.dumps({"ok": True, "aggregate_sha256": package["aggregate_sha256"]}))
        return 0
    if not args.manifest or not args.bootstrap:
        parser.error("mounted commands require --manifest and --bootstrap")
    manifest = verify_mounted(args.manifest, args.bootstrap)
    if args.command == "install-mounted":
        if not args.destination:
            parser.error("install-mounted requires --destination")
        install_mounted(manifest, args.bootstrap, args.destination)
    print(json.dumps({"ok": True, "aggregate_sha256": manifest["aggregate_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
