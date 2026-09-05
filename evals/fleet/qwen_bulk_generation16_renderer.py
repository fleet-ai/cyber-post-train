"""Render exactly two create-once Qwen G16 bulk controller Jobs."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml

from evals.fleet import exact_pass4_bulk_release_renderer_v3 as prior
from evals.fleet import qwen_bulk_generation16 as bulk
from evals.fleet import qwen_bulk_generation16_package as package
from evals.fleet.qwen_bulk_generation16_runtime import validate_g15_gate


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
