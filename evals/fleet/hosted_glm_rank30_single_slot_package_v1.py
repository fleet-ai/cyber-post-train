"""Render the rank-30-only atomic hosted GLM controller."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml

from evals.fleet import hosted_glm_exact_bulk_runtime_v1 as source_runtime
from evals.fleet import hosted_glm_rank30_single_slot_v1 as successor
from evals.fleet import hosted_glm_whole_task_package_v1 as whole_package
from evals.fleet import self_hosted

PATHS = {
    **whole_package.PATHS,
    "bulk.py": "evals/fleet/hosted_glm_rank30_single_slot_v1.py",
    "bulk_runtime.py": "evals/fleet/hosted_glm_rank30_single_slot_runtime_v1.py",
    "run.sh": "evals/fleet/scripts/run_hosted_glm_rank30_single_slot_v1.sh",
}


def _data(root: Path) -> dict[str, str]:
    result = {}
    for name, relative in PATHS.items():
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"unsafe rank-30 package source: {relative}")
        result[name] = path.read_text()
    return result


def source_package_sha256(root: Path) -> str:
    return self_hosted.sha256(self_hosted.canonical_json(_data(root)))


def render(root: Path, *, release_receipt: Path | None = None) -> dict[str, Any]:
    data = _data(root)
    source_sha = self_hosted.sha256(self_hosted.canonical_json(data))
    authorized = release_receipt is not None
    plan = successor.validate_all(root)[successor.CONTROLLER]
    if release_receipt is not None:
        release = successor.load(release_receipt)
        inventory = successor.load(source_runtime.INVENTORY_PATH)
        runtime_plan = successor.build_runtime_plan(
            successor.CONTROLLER, inventory, root
        )
        successor.validate_release(release, runtime_plan, source_sha)
        data["release.json"] = release_receipt.read_text()
    authority = successor.CONTROLLERS[successor.CONTROLLER]
    template = yaml.safe_load(
        (root / "evals/fleet/cluster/hosted-glm-exact-r001-a1-canary-v1.yaml").read_text()
    )
    configmap = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": authority["configmap_name"],
            "namespace": "fleet-train-jobs",
        },
        "immutable": True,
        "data": data,
    }
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
    pod["spec"]["volumes"][0]["configMap"]["name"] = authority["configmap_name"]
    container = pod["spec"]["containers"][0]
    container["env"] = [
        row
        for row in container["env"]
        if row["name"] not in {"CONTROLLER", "JOB_NAME", "SECRET_UID"}
    ]
    container["env"].extend(
        [
            {"name": "JOB_NAME", "value": name},
            {"name": "GLM_HOSTED_R30_SOURCE_SHA256", "value": source_sha},
        ]
    )
    objects = {"apiVersion": "v1", "kind": "List", "items": [configmap, job]}
    if len(json.dumps(configmap).encode()) >= 900_000:
        raise ValueError("rank-30 ConfigMap exceeds safety budget")
    return {
        "objects": objects,
        "package_sha256": self_hosted.sha256(self_hosted.canonical_json(objects)),
        "source_package_sha256": source_sha,
        "plan_sha256": plan["plan_sha256"],
        "launch_authorized": authorized,
        "scoring_authorized": authorized,
    }
