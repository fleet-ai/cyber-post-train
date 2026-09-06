"""Render held or freshly released peer-free rank-30 hosted GLM objects."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml

from evals.fleet import hosted_glm_exact_bulk_package_v1 as base
from evals.fleet import hosted_glm_exact_bulk_runtime_v1 as source_runtime
from evals.fleet import hosted_glm_rank30_peer_free_successor_v2 as successor
from evals.fleet import self_hosted

PATHS = {
    **{
        name: relative
        for name, relative in base.PATHS.items()
        if name not in {"bulk.py", "bulk_runtime.py", "run.sh"}
    },
    "bulk.py": "evals/fleet/hosted_glm_rank30_peer_free_successor_v2.py",
    "bulk_runtime.py": "evals/fleet/hosted_glm_rank30_peer_free_runtime_v2.py",
    "whole_task_v1.py": "evals/fleet/hosted_glm_whole_task_successor_v1.py",
    "engine.py": "evals/fleet/hosted_glm_whole_task_engine_v1.py",
    "base_engine.py": "evals/fleet/exact_pass4_bulk_runtime_v3.py",
    "run.sh": "evals/fleet/scripts/run_hosted_glm_rank30_peer_free_v2.sh",
}


def _data(root: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    for name, relative in PATHS.items():
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"unsafe peer-free rank30 package source: {relative}")
        data[name] = path.read_text()
    return data


def source_package_sha256(root: Path) -> str:
    return self_hosted.sha256(self_hosted.canonical_json(_data(root)))


def _job(
    template: dict[str, Any], configmap: str, source_sha: str, *, authorized: bool
) -> dict[str, Any]:
    job = copy.deepcopy(template)
    job["metadata"]["name"] = successor.JOB_NAME
    job["metadata"]["labels"]["cyber-post-train.fleet.ai/experiment"] = successor.JOB_NAME
    job["metadata"]["annotations"][
        "cyber-post-train.fleet.ai/launch-authorized"
    ] = str(authorized).lower()
    job["spec"]["activeDeadlineSeconds"] = 129_600
    pod = job["spec"]["template"]
    pod["metadata"]["labels"]["cyber-post-train.fleet.ai/experiment"] = successor.JOB_NAME
    pod["spec"]["volumes"][0]["configMap"]["name"] = configmap
    container = pod["spec"]["containers"][0]
    container["env"] = [
        row
        for row in container["env"]
        if row["name"] not in {"CONTROLLER", "JOB_NAME", "SECRET_UID"}
    ]
    container["env"].extend(
        [
            {"name": "CONTROLLER", "value": successor.CONTROLLER},
            {"name": "JOB_NAME", "value": successor.JOB_NAME},
            {"name": "GLM_HOSTED_R30_SOURCE_SHA256", "value": source_sha},
        ]
    )
    return job


def render(root: Path, *, release_receipt: Path | None = None) -> dict[str, Any]:
    plan = successor.validate_all(root)[successor.CONTROLLER]
    data = _data(root)
    source_sha = self_hosted.sha256(self_hosted.canonical_json(data))
    held = successor.expected_held(plan, source_sha)
    authorized = release_receipt is not None
    bound = held
    if authorized:
        inventory = successor.load(source_runtime.INVENTORY_PATH)
        runtime_plan = successor.build_runtime_plan(successor.CONTROLLER, inventory, root)
        bound = successor.load(release_receipt)  # type: ignore[arg-type]
        successor.validate_release(bound, runtime_plan, source_sha)
    data["release.json"] = json.dumps(bound, sort_keys=True, separators=(",", ":")) + "\n"
    template = yaml.safe_load(
        (root / "evals/fleet/cluster/hosted-glm-exact-r001-a1-canary-v1.yaml").read_text()
    )
    configmap = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": successor.CONFIGMAP_NAME, "namespace": "fleet-train-jobs"},
        "immutable": True,
        "data": data,
    }
    job = _job(template, successor.CONFIGMAP_NAME, source_sha, authorized=authorized)
    if len(json.dumps(configmap).encode()) >= 900_000:
        raise ValueError("peer-free rank30 ConfigMap exceeds safety budget")
    objects = {"apiVersion": "v1", "kind": "List", "items": [configmap, job]}
    return {
        "objects": objects,
        "held_receipt": held,
        "package_sha256": self_hosted.sha256(self_hosted.canonical_json(objects)),
        "source_package_sha256": source_sha,
        "controller_plan_sha256": plan["plan_sha256"],
        "launch_authorized": authorized,
        "scoring_authorized": authorized,
        "reason": None if authorized else "requires_fresh_digest_valid_peer_free_release",
    }
