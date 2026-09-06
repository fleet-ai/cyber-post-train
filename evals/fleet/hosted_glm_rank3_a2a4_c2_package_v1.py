"""Render release-observer and scored-controller packages for rank-3 attempts 2-4."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from evals.fleet import hosted_glm_exact_bulk_package_v1 as base
from evals.fleet import hosted_glm_exact_bulk_release_package_v1 as release_base
from evals.fleet import hosted_glm_rank3_a2a4_c2_release_v1 as release
from evals.fleet import hosted_glm_rank3_a2a4_c2_runtime_v1 as runtime
from evals.fleet import hosted_glm_rank3_a2a4_c2_successor_v1 as successor
from evals.fleet import self_hosted

VALIDATION = "docs/evidence/glm53-study/2026-09-06-glm53-hosted-rank3-a1-v2-accepted-validated.json"


def _data(root: Path, *, observer: bool, receipt: Path | None = None) -> dict[str, str]:
    files = {
        "successor.py": "evals/fleet/hosted_glm_rank3_a2a4_c2_successor_v1.py",
        "successor_runtime.py": "evals/fleet/hosted_glm_rank3_a2a4_c2_runtime_v1.py",
        "run.sh": (
            "evals/fleet/scripts/run_hosted_glm_rank3_a2a4_c2_release_v1.sh"
            if observer
            else "evals/fleet/scripts/run_hosted_glm_rank3_a2a4_c2_successor_v1.sh"
        ),
    }
    if observer:
        files.update(
            {
                "original_release.py": "evals/fleet/hosted_glm_exact_bulk_release_v1.py",
                "release.py": "evals/fleet/hosted_glm_rank3_a2a4_c2_release_v1.py",
                "canary-validation.json": VALIDATION,
            }
        )
    data = {name: (root / rel).read_text() for name, rel in files.items()}
    if receipt is not None:
        data["release.json"] = receipt.read_text()
    return data


def render_release(root: Path) -> dict[str, Any]:
    package = copy.deepcopy(release_base.render(root))
    configmap, job = package["objects"]["items"]
    configmap["metadata"]["name"] = release.CONFIGMAP_NAME
    configmap["data"].update(_data(root, observer=True))
    job["metadata"]["name"] = release.JOB_NAME
    job["metadata"]["labels"]["cyber-post-train.fleet.ai/experiment"] = release.JOB_NAME
    job["metadata"]["annotations"]["cyber-post-train.fleet.ai/launch-authorized"] = "true"
    job["spec"]["template"]["metadata"]["labels"]["cyber-post-train.fleet.ai/experiment"] = release.JOB_NAME
    job["spec"]["template"]["spec"]["priorityClassName"] = "fleet-infra-quiet"
    job["spec"]["template"]["spec"]["volumes"][0]["configMap"]["name"] = release.CONFIGMAP_NAME
    objects = {"apiVersion": "v1", "kind": "List", "items": [configmap, job]}
    return {"objects": objects, "package_sha256": self_hosted.sha256(self_hosted.canonical_json(objects)), "launch_authorized": True}


def render_scored(root: Path, receipt: Path) -> dict[str, Any]:
    plan = successor.validate_all(root)[successor.CONTROLLER]
    value = successor.load(receipt)
    if any(
        (
            value.get("schema_version") != runtime.RELEASE_SCHEMA,
            value.get("status") != "CLEAR",
            value.get("successor_job") != successor.JOB_NAME,
            value.get("successor_configmap") != successor.CONFIGMAP_NAME,
            not isinstance(value.get("plan_sha256"), str),
            not value.get("plan_sha256", "").startswith("sha256:"),
            value.get("receipt_sha256") != self_hosted.digest_without(value, "receipt_sha256"),
        )
    ):
        raise RuntimeError("rank-3 successor scored release drifted")
    configmap, job = copy.deepcopy(base.render(root)["objects"]["items"][:2])
    configmap["metadata"]["name"] = successor.CONFIGMAP_NAME
    configmap["data"].update(_data(root, observer=False, receipt=receipt))
    job["metadata"]["name"] = successor.JOB_NAME
    job["metadata"]["labels"]["cyber-post-train.fleet.ai/experiment"] = successor.JOB_NAME
    job["metadata"]["annotations"]["cyber-post-train.fleet.ai/launch-authorized"] = "true"
    pod = job["spec"]["template"]
    pod["metadata"]["labels"]["cyber-post-train.fleet.ai/experiment"] = successor.JOB_NAME
    pod["spec"]["priorityClassName"] = "fleet-infra-quiet"
    pod["spec"]["volumes"][0]["configMap"]["name"] = successor.CONFIGMAP_NAME
    pod["spec"]["containers"][0]["env"] = [row for row in pod["spec"]["containers"][0]["env"] if row["name"] not in {"JOB_NAME", "SECRET_UID", "CONTROLLER"}]
    objects = {"apiVersion": "v1", "kind": "List", "items": [configmap, job]}
    return {"objects": objects, "package_sha256": self_hosted.sha256(self_hosted.canonical_json(objects)), "launch_authorized": True, "held_plan_sha256": plan["plan_sha256"]}
