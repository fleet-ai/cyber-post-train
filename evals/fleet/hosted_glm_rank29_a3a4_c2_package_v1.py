"""Render held release and gated scored packages for rank-29 attempts 3 and 4."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from evals.fleet import hosted_glm_exact_bulk_package_v1 as base
from evals.fleet import hosted_glm_exact_bulk_release_package_v1 as release_base
from evals.fleet import hosted_glm_rank29_a3a4_c2_runtime_v1 as runtime
from evals.fleet import hosted_glm_rank29_a3a4_c2_successor_v1 as successor
from evals.fleet import self_hosted

RELEASE_JOB_NAME = "chris-glm53-exact100-hosted-r029-a3a4-release-v1"
RELEASE_CONFIGMAP_NAME = RELEASE_JOB_NAME + "-run"
RELEASE_BINDING_SENTINEL = "fleet-rank29-release-receipt-bound-by-self-digest-v1\n"


def _data(
    root: Path, *, observer: bool, receipt_text: str | None = None
) -> dict[str, str]:
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
    if receipt_text is not None:
        data["release.json"] = receipt_text
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
    configmap["metadata"]["name"] = RELEASE_CONFIGMAP_NAME
    configmap["data"].update(_data(root, observer=True))
    _identity(job, RELEASE_JOB_NAME, RELEASE_CONFIGMAP_NAME)
    bindings = scored_bindings(root)
    env = job["spec"]["template"]["spec"]["containers"][0]["env"]
    env.extend({"name": key.upper(), "value": value} for key, value in bindings.items())
    job["metadata"]["annotations"][
        "cyber-post-train.fleet.ai/launch-authorized"
    ] = str(authorized).lower()
    objects = {"apiVersion": "v1", "kind": "List", "items": [configmap, job]}
    return {
        "objects": objects,
        "package_sha256": self_hosted.sha256(self_hosted.canonical_json(objects)),
        "launch_authorized": authorized,
        "scored_launch_authorized": False,
        **bindings,
    }


def _scored_objects(root: Path, release_text: str) -> dict[str, Any]:
    configmap, job = copy.deepcopy(base.render(root)["objects"]["items"][:2])
    configmap["metadata"]["name"] = successor.CONFIGMAP_NAME
    configmap["data"].update(_data(root, observer=False, receipt_text=release_text))
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
    return {"apiVersion": "v1", "kind": "List", "items": [configmap, job]}


def scored_bindings(root: Path) -> dict[str, str]:
    objects = _scored_objects(root, RELEASE_BINDING_SENTINEL)
    source_data = dict(objects["items"][0]["data"])
    source_data.pop("release.json")
    return {
        "scored_source_sha256": self_hosted.sha256(self_hosted.canonical_json(source_data)),
        "scored_package_template_sha256": self_hosted.sha256(
            self_hosted.canonical_json(objects)
        ),
    }


def render_scored(root: Path, receipt: Path) -> dict[str, Any]:
    plan = successor.validate_all(root)[successor.CONTROLLER]
    value = successor.load(receipt)
    runtime_plan = successor.build_runtime_plan(
        successor.CONTROLLER,
        successor.load(base.runtime.INVENTORY_PATH),
        root,
    )
    runtime.validate_release_receipt(runtime_plan, value)
    if any(value.get(key) != expected for key, expected in scored_bindings(root).items()):
        raise RuntimeError("rank-29 successor scored release drifted")
    objects = _scored_objects(root, receipt.read_text())
    return {
        "objects": objects,
        "package_sha256": self_hosted.sha256(self_hosted.canonical_json(objects)),
        "launch_authorized": True,
        "held_plan_sha256": plan["plan_sha256"],
        "active_deadline_seconds": runtime.ACTIVE_DEADLINE_SECONDS,
        "preclaim_guard_seconds": runtime.CLAIM_GUARD_SECONDS,
        **scored_bindings(root),
    }
