"""Render exactly two create-once Qwen G16 bulk controller Jobs."""

from __future__ import annotations

import copy
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import yaml

from evals.fleet import exact_pass4_bulk_release_renderer_v3 as prior
from evals.fleet import qwen_bulk_generation16 as bulk
from evals.fleet import qwen_bulk_generation16_package as package
from evals.fleet.qwen_bulk_generation16_runtime import validate_g15_gate

NAMESPACE_RE = re.compile(r"[a-z0-9](?:[-a-z0-9]*[a-z0-9])?")


def _job_runtime_dependencies(job: dict[str, Any]) -> tuple[set[str], set[str]]:
    pod = job["spec"]["template"]["spec"]
    secrets = {
        ref["name"]
        for container in [*pod.get("initContainers", []), *pod.get("containers", [])]
        for env in container.get("env", [])
        if (ref := env.get("valueFrom", {}).get("secretKeyRef")) is not None
    }
    secrets.update(
        ref["name"]
        for container in [*pod.get("initContainers", []), *pod.get("containers", [])]
        for env_from in container.get("envFrom", [])
        if (ref := env_from.get("secretRef")) is not None
    )
    configmaps = {
        ref["name"]
        for container in [*pod.get("initContainers", []), *pod.get("containers", [])]
        for env_from in container.get("envFrom", [])
        if (ref := env_from.get("configMapRef")) is not None
    }
    secrets.update(row["name"] for row in pod.get("imagePullSecrets", []))
    for volume in pod.get("volumes", []):
        if ref := volume.get("configMap"):
            configmaps.add(ref["name"])
        if ref := volume.get("secret"):
            secrets.add(ref["secretName"])
        for source in volume.get("projected", {}).get("sources", []):
            if ref := source.get("configMap"):
                configmaps.add(ref["name"])
            if ref := source.get("secret"):
                secrets.add(ref["name"])
    return secrets, configmaps


def validate_runtime_dependencies(
    rendered: dict[str, Any],
    *,
    available_secrets: set[str],
    available_configmaps: set[str],
) -> None:
    declared_configmaps = {
        item["metadata"]["name"]
        for item in rendered.get("items", [])
        if item.get("kind") == "ConfigMap"
    }
    required_secrets: set[str] = set()
    required_configmaps: set[str] = set()
    for item in rendered.get("items", []):
        if item.get("kind") == "Job":
            secrets, configmaps = _job_runtime_dependencies(item)
            required_secrets.update(secrets)
            required_configmaps.update(configmaps)
    missing_secrets = required_secrets - available_secrets
    missing_configmaps = required_configmaps - available_configmaps - declared_configmaps
    if missing_secrets or missing_configmaps:
        raise ValueError(
            "Generation-16 runtime dependencies are absent: "
            f"secrets={sorted(missing_secrets)!r}, "
            f"configmaps={sorted(missing_configmaps)!r}"
        )


def _kubectl_names(kind: str, namespace: str) -> set[str]:
    if kind not in {"secrets", "configmaps"} or NAMESPACE_RE.fullmatch(namespace) is None:
        raise ValueError("Generation-16 dependency inventory request is invalid")
    result = subprocess.run(
        ["kubectl", "-n", namespace, "get", kind, "-o", "name"],
        check=True,
        capture_output=True,
        text=True,
    )
    prefix = {"secrets": "secret/", "configmaps": "configmap/"}[kind]
    names: set[str] = set()
    for line in result.stdout.splitlines():
        if line and not line.startswith(prefix):
            raise ValueError("Generation-16 dependency inventory output drifted")
        if line:
            names.add(line.removeprefix(prefix))
    return names


def validate_live_runtime_dependencies(
    rendered: dict[str, Any], namespace: str = "fleet-train-jobs"
) -> None:
    validate_runtime_dependencies(
        rendered,
        available_secrets=_kubectl_names("secrets", namespace),
        available_configmaps=_kubectl_names("configmaps", namespace),
    )


def render(root: Path, inventory: dict[str, Any], package_commit: str) -> dict[str, Any]:
    bulk.validate_inventory_gate(inventory, root)
    if bulk.COMMIT_RE.fullmatch(package_commit) is None:
        raise ValueError("Generation-16 package commit must be immutable")
    gate = bulk.load(root / bulk.G15_GATE_PATH)
    validate_g15_gate(gate)
    built = package.build_package(root)
    plans = {
        controller: bulk.build_runtime_plan(controller, inventory, root)
        for controller in bulk.CONTROLLERS
    }
    items: list[dict[str, Any]] = []
    for name in package.CORE_NAMES:
        cm = copy.deepcopy(built["configmaps"][name])
        cm["metadata"]["annotations"] = {
            "cyber-post-train.fleet.ai/preview-only": "false",
            "cyber-post-train.fleet.ai/launch-authorized": "true",
            "cyber-post-train.fleet.ai/package-commit": package_commit,
        }
        items.append(cm)
    for controller, plan in plans.items():
        name = bulk.CONTROLLERS[controller]["configmap_name"]
        cm = copy.deepcopy(built["configmaps"][name])
        cm["metadata"]["annotations"] = {
            "cyber-post-train.fleet.ai/preview-only": "false",
            "cyber-post-train.fleet.ai/launch-authorized": "true",
            "cyber-post-train.fleet.ai/package-commit": package_commit,
            "cyber-post-train.fleet.ai/g15-gate-sha256": gate["receipt_sha256"],
        }
        cm["data"]["runtime-plan.json"] = json.dumps(
            plan, sort_keys=True, separators=(",", ":")
        ) + "\n"
        items.append(cm)
    prior.bulk = bulk
    prior.package = package
    for controller, plan in plans.items():
        manifest = {
            "release_receipt_sha256": gate["receipt_sha256"],
            "package_aggregate_sha256": built["controller_manifests"][controller][
                "aggregate_sha256"
            ],
            "package_commit": package_commit,
            "runtime_gates": {"QWEN_G15_ACCEPTED_GATE_SHA256": gate["receipt_sha256"]},
        }
        job = prior._job(controller, manifest, plan)  # noqa: SLF001
        for container in job["spec"]["template"]["spec"]["containers"]:
            for env in container.get("env", []):
                if env.get("name") == "FLEET_API_KEY":
                    env["valueFrom"]["secretKeyRef"]["name"] = bulk.FLEET_API_KEY_SECRET
        job["metadata"]["labels"]["cyber-post-train.fleet.ai/experiment"] = (
            "qwen-exact100-g16"
        )
        job["spec"]["backoffLimit"] = 0
        pod_labels = job["spec"]["template"]["metadata"]["labels"]
        pod_labels["cyber-post-train.fleet.ai/experiment"] = "qwen-exact100-g16"
        items.append(job)
    return {"apiVersion": "v1", "kind": "List", "items": items}


def dump(root: Path, inventory: Path, package_commit: str, output: Path) -> None:
    output.write_text(
        yaml.safe_dump(render(root, bulk.load(inventory), package_commit), sort_keys=False)
    )
