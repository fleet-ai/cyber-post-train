"""Render fresh release/controller packages after the v1 adapter failure."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from evals.fleet import hosted_glm_rank3_a2a4_c2_package_v1 as prior
from evals.fleet import hosted_glm_rank3_a2a4_c2_release_v2 as release
from evals.fleet import hosted_glm_rank3_a2a4_c2_runtime_v2 as runtime
from evals.fleet import hosted_glm_rank3_a2a4_c2_successor_v2 as successor
from evals.fleet import self_hosted


def _replace(
    data: dict[str, str],
    root: Path,
    *,
    observer: bool,
    receipt: Path | None = None,
) -> None:
    paths = {
        "prior_successor.py": "evals/fleet/hosted_glm_rank3_a2a4_c2_successor_v1.py",
        "successor.py": "evals/fleet/hosted_glm_rank3_a2a4_c2_successor_v2.py",
        "successor_runtime.py": "evals/fleet/hosted_glm_rank3_a2a4_c2_runtime_v2.py",
        "run.sh": (
            "evals/fleet/scripts/run_hosted_glm_rank3_a2a4_c2_release_v2.sh"
            if observer else "evals/fleet/scripts/run_hosted_glm_rank3_a2a4_c2_successor_v2.sh"
        ),
    }
    if observer:
        paths.update(
            {
                "prior_release.py": "evals/fleet/hosted_glm_rank3_a2a4_c2_release_v1.py",
                "release.py": "evals/fleet/hosted_glm_rank3_a2a4_c2_release_v2.py",
            }
        )
    data.update({name: (root / rel).read_text() for name, rel in paths.items()})
    if receipt is not None:
        data["release.json"] = receipt.read_text()


def render_release(root: Path) -> dict[str, Any]:
    value = copy.deepcopy(prior.render_release(root))
    configmap, job = value["objects"]["items"]
    configmap["metadata"]["name"] = release.CONFIGMAP_NAME
    _replace(configmap["data"], root, observer=True)
    job["metadata"]["name"] = release.JOB_NAME
    job["metadata"]["labels"]["cyber-post-train.fleet.ai/experiment"] = release.JOB_NAME
    job["spec"]["template"]["metadata"]["labels"][
        "cyber-post-train.fleet.ai/experiment"
    ] = release.JOB_NAME
    job["spec"]["template"]["spec"]["volumes"][0]["configMap"]["name"] = release.CONFIGMAP_NAME
    objects = {"apiVersion": "v1", "kind": "List", "items": [configmap, job]}
    return {
        "objects": objects,
        "package_sha256": self_hosted.sha256(self_hosted.canonical_json(objects)),
        "launch_authorized": True,
    }


def render_scored(root: Path, receipt: Path) -> dict[str, Any]:
    value = successor.load(receipt)
    if any(
        (
            value.get("schema_version") != runtime.RELEASE_SCHEMA,
            value.get("status") != "CLEAR",
            value.get("successor_job") != successor.JOB_NAME,
            value.get("successor_configmap") != successor.CONFIGMAP_NAME,
            value.get("failed_v1") != runtime.FAILED_V1,
            value.get("receipt_sha256") != self_hosted.digest_without(value, "receipt_sha256"),
        )
    ):
        raise RuntimeError("rank-3 v2 successor scored release drifted")
    base = copy.deepcopy(prior.base.render(root))
    configmap, job = base["objects"]["items"][:2]
    configmap["metadata"]["name"] = successor.CONFIGMAP_NAME
    _replace(configmap["data"], root, observer=False, receipt=receipt)
    job["metadata"]["name"] = successor.JOB_NAME
    job["metadata"]["labels"]["cyber-post-train.fleet.ai/experiment"] = successor.JOB_NAME
    job["metadata"]["annotations"]["cyber-post-train.fleet.ai/launch-authorized"] = "true"
    pod = job["spec"]["template"]
    pod["metadata"]["labels"]["cyber-post-train.fleet.ai/experiment"] = successor.JOB_NAME
    pod["spec"]["priorityClassName"] = "fleet-infra-quiet"
    pod["spec"]["volumes"][0]["configMap"]["name"] = successor.CONFIGMAP_NAME
    pod["spec"]["containers"][0]["env"] = [
        row
        for row in pod["spec"]["containers"][0]["env"]
        if row["name"] not in {"JOB_NAME", "SECRET_UID", "CONTROLLER"}
    ]
    objects = {"apiVersion": "v1", "kind": "List", "items": [configmap, job]}
    return {
        "objects": objects,
        "package_sha256": self_hosted.sha256(self_hosted.canonical_json(objects)),
        "launch_authorized": True,
    }
