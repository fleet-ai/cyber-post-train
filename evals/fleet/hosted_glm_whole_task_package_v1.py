"""Render held or freshly released atomic hosted GLM whole-task controllers."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml

from evals.fleet import hosted_glm_exact_bulk_package_v1 as base
from evals.fleet import hosted_glm_exact_bulk_runtime_v1 as source_runtime
from evals.fleet import hosted_glm_whole_task_successor_v1 as successor
from evals.fleet import self_hosted

PATHS = {
    **{
        name: relative
        for name, relative in base.PATHS.items()
        if name not in {"bulk.py", "bulk_runtime.py", "run.sh"}
    },
    "bulk.py": "evals/fleet/hosted_glm_whole_task_successor_v1.py",
    "bulk_runtime.py": "evals/fleet/hosted_glm_whole_task_runtime_v1.py",
    "engine.py": "evals/fleet/hosted_glm_whole_task_engine_v1.py",
    "base_engine.py": "evals/fleet/exact_pass4_bulk_runtime_v3.py",
    "run.sh": "evals/fleet/scripts/run_hosted_glm_whole_task_v1.sh",
}


def _data(root: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    for name, relative in PATHS.items():
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"unsafe hosted GLM whole-task package source: {relative}")
        data[name] = path.read_text()
    return data


def source_package_sha256(root: Path) -> str:
    return self_hosted.sha256(self_hosted.canonical_json(_data(root)))


def _job(
    template: dict[str, Any], controller: str, configmap: str, source_sha: str, *, authorized: bool
) -> dict[str, Any]:
    authority = successor.CONTROLLERS[controller]
    job = copy.deepcopy(template)
    name = authority["job_name"]
    job["metadata"]["name"] = name
    job["metadata"]["labels"]["cyber-post-train.fleet.ai/experiment"] = name
    job["metadata"]["annotations"][
        "cyber-post-train.fleet.ai/launch-authorized"
    ] = str(authorized).lower()
    job["spec"]["activeDeadlineSeconds"] = 129_600
    pod = job["spec"]["template"]
    pod["metadata"]["labels"]["cyber-post-train.fleet.ai/experiment"] = name
    pod["spec"]["volumes"][0]["configMap"]["name"] = configmap
    container = pod["spec"]["containers"][0]
    container["env"] = [
        row
        for row in container["env"]
        if row["name"] not in {"CONTROLLER", "JOB_NAME", "SECRET_UID"}
    ]
    container["env"].extend(
        [
            {"name": "CONTROLLER", "value": controller},
            {"name": "JOB_NAME", "value": name},
            {"name": "GLM_HOSTED_WHOLE_TASK_SOURCE_SHA256", "value": source_sha},
        ]
    )
    return job


def render(root: Path, *, release_receipt: Path | None = None) -> dict[str, Any]:
    static_plans = successor.validate_all(root)
    data = _data(root)
    source_sha = self_hosted.sha256(self_hosted.canonical_json(data))
    authorized = release_receipt is not None
    release: dict[str, Any] | None = None
    if authorized:
        release = successor.load(release_receipt)  # type: ignore[arg-type]
        inventory = successor.load(source_runtime.INVENTORY_PATH)
        runtime_plans = {
            name: successor.build_runtime_plan(name, inventory, root)
            for name in successor.CONTROLLERS
        }
        successor.validate_release(release, runtime_plans, source_sha)
        data["release.json"] = release_receipt.read_text()  # type: ignore[union-attr]
    template = yaml.safe_load(
        (root / "evals/fleet/cluster/hosted-glm-exact-r001-a1-canary-v1.yaml").read_text()
    )
    items: list[dict[str, Any]] = []
    for controller, authority in successor.CONTROLLERS.items():
        configmap = {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": authority["configmap_name"],
                "namespace": "fleet-train-jobs",
            },
            "immutable": True,
            "data": copy.deepcopy(data),
        }
        items.extend(
            [
                configmap,
                _job(
                    template,
                    controller,
                    authority["configmap_name"],
                    source_sha,
                    authorized=authorized,
                ),
            ]
        )
    objects = {"apiVersion": "v1", "kind": "List", "items": items}
    oversized = any(
        len(json.dumps(item).encode()) >= 900_000
        for item in items
        if item["kind"] == "ConfigMap"
    )
    if oversized:
        raise ValueError("hosted GLM whole-task ConfigMap exceeds safety budget")
    return {
        "objects": objects,
        "package_sha256": self_hosted.sha256(self_hosted.canonical_json(objects)),
        "source_package_sha256": source_sha,
        "controller_plan_sha256s": {
            name: plan["plan_sha256"] for name, plan in static_plans.items()
        },
        "launch_authorized": authorized,
        "scoring_authorized": authorized,
        "reason": None if authorized else "requires_fresh_digest_valid_release",
    }
