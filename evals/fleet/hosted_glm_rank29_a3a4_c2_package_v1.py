"""Render held release and gated scored packages for rank-29 attempts 3 and 4."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from evals.fleet import hosted_glm_exact_bulk_package_v1 as base
from evals.fleet import hosted_glm_exact_bulk_release_package_v1 as release_base
from evals.fleet import hosted_glm_rank29_a3a4_c2_release_v1 as release
from evals.fleet import hosted_glm_rank29_a3a4_c2_runtime_v1 as runtime
from evals.fleet import hosted_glm_rank29_a3a4_c2_successor_v1 as successor
from evals.fleet import self_hosted


def _data(root: Path, *, observer: bool, receipt: Path | None = None) -> dict[str, str]:
    files = {
        "successor.py": "evals/fleet/hosted_glm_rank29_a3a4_c2_successor_v1.py",
        "successor_runtime.py": "evals/fleet/hosted_glm_rank29_a3a4_c2_runtime_v1.py",
        "run.sh": (
            "evals/fleet/scripts/run_hosted_glm_rank29_a3a4_c2_release_v1.sh"
            if observer
            else "evals/fleet/scripts/run_hosted_glm_rank29_a3a4_c2_successor_v1.sh"
        ),
    }
    if observer:
        files.update(
            {
                "original_release.py": "evals/fleet/hosted_glm_exact_bulk_release_v1.py",
                "release.py": "evals/fleet/hosted_glm_rank29_a3a4_c2_release_v1.py",
            }
        )
    data = {name: (root / relative).read_text() for name, relative in files.items()}
    if receipt is not None:
        data["release.json"] = receipt.read_text()
    return data


def _identity(job: dict[str, Any], name: str, configmap: str) -> None:
    job["metadata"]["name"] = name
    job["metadata"]["labels"]["cyber-post-train.fleet.ai/experiment"] = name
    pod = job["spec"]["template"]
    pod["metadata"]["labels"]["cyber-post-train.fleet.ai/experiment"] = name
    pod["spec"]["priorityClassName"] = "fleet-serve-low"
    pod["spec"]["preemptionPolicy"] = "Never"
    pod["spec"]["volumes"][0]["configMap"]["name"] = configmap


def render_release(root: Path, *, authorized: bool = False) -> dict[str, Any]:
    value = copy.deepcopy(release_base.render(root))
    configmap, job = value["objects"]["items"]
    configmap["metadata"]["name"] = release.CONFIGMAP_NAME
    configmap["data"].update(_data(root, observer=True))
    _identity(job, release.JOB_NAME, release.CONFIGMAP_NAME)
    job["metadata"]["annotations"][
        "cyber-post-train.fleet.ai/launch-authorized"
    ] = str(authorized).lower()
    objects = {"apiVersion": "v1", "kind": "List", "items": [configmap, job]}
    return {
        "objects": objects,
        "package_sha256": self_hosted.sha256(self_hosted.canonical_json(objects)),
        "launch_authorized": authorized,
        "scored_launch_authorized": False,
    }


def render_scored(root: Path, receipt: Path) -> dict[str, Any]:
    plan = successor.validate_all(root)[successor.CONTROLLER]
    value = successor.load(receipt)
    if any(
        (
            value.get("schema_version") != runtime.RELEASE_SCHEMA,
            value.get("status") != "CLEAR",
            value.get("successor_job") != successor.JOB_NAME,
            value.get("successor_configmap") != successor.CONFIGMAP_NAME,
            value.get("plan_sha256")
            != successor.build_runtime_plan(
                successor.CONTROLLER,
                successor.load(base.runtime.INVENTORY_PATH),
                root,
            )["plan_sha256"],
            value.get("planned_cells") != 2,
            value.get("blocked_a2_claim_sha256") != successor.BLOCKED_A2_CLAIM_SHA,
            value.get("receipt_sha256")
            != self_hosted.digest_without(value, "receipt_sha256"),
        )
    ):
        raise RuntimeError("rank-29 successor scored release drifted")
    configmap, job = copy.deepcopy(base.render(root)["objects"]["items"][:2])
    configmap["metadata"]["name"] = successor.CONFIGMAP_NAME
    configmap["data"].update(_data(root, observer=False, receipt=receipt))
    _identity(job, successor.JOB_NAME, successor.CONFIGMAP_NAME)
    job["metadata"]["annotations"]["cyber-post-train.fleet.ai/launch-authorized"] = "true"
    job["spec"]["activeDeadlineSeconds"] = runtime.ACTIVE_DEADLINE_SECONDS
    container = job["spec"]["template"]["spec"]["containers"][0]
    container["env"] = [
        row
        for row in container["env"]
        if row["name"] not in {"JOB_NAME", "SECRET_UID", "CONTROLLER"}
    ]
    container["env"].append({"name": "JOB_NAME", "value": successor.JOB_NAME})
    objects = {"apiVersion": "v1", "kind": "List", "items": [configmap, job]}
    return {
        "objects": objects,
        "package_sha256": self_hosted.sha256(self_hosted.canonical_json(objects)),
        "launch_authorized": True,
        "held_plan_sha256": plan["plan_sha256"],
        "active_deadline_seconds": runtime.ACTIVE_DEADLINE_SECONDS,
        "preclaim_guard_seconds": runtime.CLAIM_GUARD_SECONDS,
    }
