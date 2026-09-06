"""Render the config-complete rank-2 hosted GLM v3 release observer."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from evals.fleet import hosted_glm_s1_r2_c2_release_package_v2 as prior
from evals.fleet import hosted_glm_s1_r2_c2_release_v3 as release
from evals.fleet import self_hosted


def render(root: Path) -> dict[str, Any]:
    rendered = prior.render(root)
    configmap, job = copy.deepcopy(rendered["objects"]["items"])
    configmap["metadata"]["name"] = release.CONFIGMAP_NAME
    configmap["data"]["successor_release_v2.py"] = (
        root / "evals/fleet/hosted_glm_s1_r2_c2_release_v2.py"
    ).read_text()
    configmap["data"]["successor_release.py"] = (
        root / "evals/fleet/hosted_glm_s1_r2_c2_release_v3.py"
    ).read_text()
    configmap["data"]["run.sh"] = (
        root / "evals/fleet/scripts/run_hosted_glm_s1_r2_c2_release_v3.sh"
    ).read_text()
    job["metadata"]["name"] = release.JOB_NAME
    job["metadata"]["labels"]["cyber-post-train.fleet.ai/experiment"] = release.JOB_NAME
    pod = job["spec"]["template"]
    pod["metadata"]["labels"]["cyber-post-train.fleet.ai/experiment"] = release.JOB_NAME
    pod["spec"]["volumes"][0]["configMap"]["name"] = release.CONFIGMAP_NAME
    objects = {"apiVersion": "v1", "kind": "List", "items": [configmap, job]}
    return {
        "objects": objects,
        "package_sha256": self_hosted.sha256(self_hosted.canonical_json(objects)),
        "launch_authorized": True,
    }
