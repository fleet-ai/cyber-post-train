"""Render the engine-complete rank-2 v5 release observer."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from evals.fleet import hosted_glm_s1_r2_c2_package_v3 as closure
from evals.fleet import hosted_glm_s1_r2_c2_release_package_v2 as base
from evals.fleet import hosted_glm_s1_r2_c2_release_v5 as release
from evals.fleet import self_hosted


def render(root: Path) -> dict[str, Any]:
    configmap, job = copy.deepcopy(base.render(root)["objects"]["items"])
    configmap["metadata"]["name"] = release.CONFIGMAP_NAME
    additions = {
        "prior_successor_v2.py": "evals/fleet/hosted_glm_s1_r2_c2_successor_v2.py",
        "prior_successor_v3.py": "evals/fleet/hosted_glm_s1_r2_c2_successor_v3.py",
        "release_v2.py": "evals/fleet/hosted_glm_s1_r2_c2_release_v2.py",
        "successor.py": "evals/fleet/hosted_glm_s1_r2_c2_successor_v4.py",
        "successor_release.py": "evals/fleet/hosted_glm_s1_r2_c2_release_v5.py",
        "run.sh": "evals/fleet/scripts/run_hosted_glm_s1_r2_c2_release_v5.sh",
    }
    for name, path in additions.items():
        configmap["data"][name] = (root / path).read_text()
    closure.validate_install_closure(configmap["data"])
    job["metadata"]["name"] = release.JOB_NAME
    job["metadata"]["labels"]["cyber-post-train.fleet.ai/experiment"] = release.JOB_NAME
    pod = job["spec"]["template"]
    pod["metadata"]["labels"]["cyber-post-train.fleet.ai/experiment"] = release.JOB_NAME
    pod["spec"]["volumes"][0]["configMap"]["name"] = release.CONFIGMAP_NAME
    objects = {"apiVersion": "v1", "kind": "List", "items": [configmap, job]}
    return {"objects": objects, "package_sha256": self_hosted.sha256(self_hosted.canonical_json(objects)), "launch_authorized": True}
