"""Render the held/released complete rank-2 hosted GLM successor."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from evals.fleet import hosted_glm_exact_bulk_package_v1 as base
from evals.fleet import hosted_glm_s1_r2_c2_runtime_v1 as runtime
from evals.fleet import hosted_glm_s1_r2_c2_successor_v1 as successor
from evals.fleet import self_hosted


def render(root: Path, *, release_path: Path | None = None) -> dict[str, Any]:
    successor.validate_all(root)
    authorized = release_path is not None
    if authorized:
        runtime.validate_release_static(successor.load(release_path))
    configmap, job = copy.deepcopy(base.render(root)["objects"]["items"][:2])
    configmap["metadata"]["name"] = successor.CONFIGMAP_NAME
    additions = {
        "successor.py": "evals/fleet/hosted_glm_s1_r2_c2_successor_v1.py",
        "successor_release.py": "evals/fleet/hosted_glm_s1_r2_c2_release_v1.py",
        "successor_runtime.py": "evals/fleet/hosted_glm_s1_r2_c2_runtime_v1.py",
        "original_release.py": "evals/fleet/hosted_glm_exact_bulk_release_v1.py",
        "run.sh": "evals/fleet/scripts/run_hosted_glm_s1_r2_c2_successor_v1.sh",
    }
    for name, path in additions.items():
        configmap["data"][name] = (root / path).read_text()
    job["metadata"]["name"] = successor.JOB_NAME
    job["metadata"]["labels"]["cyber-post-train.fleet.ai/experiment"] = successor.JOB_NAME
    job["metadata"]["annotations"]["cyber-post-train.fleet.ai/launch-authorized"] = str(authorized).lower()
    pod = job["spec"]["template"]
    pod["metadata"]["labels"]["cyber-post-train.fleet.ai/experiment"] = successor.JOB_NAME
    pod["spec"]["volumes"][0]["configMap"]["name"] = successor.CONFIGMAP_NAME
    objects = {"apiVersion": "v1", "kind": "List", "items": [configmap, job]}
    return {
        "objects": objects,
        "package_sha256": self_hosted.sha256(self_hosted.canonical_json(objects)),
        "launch_authorized": authorized,
        "reason": None if authorized else "requires_fresh_rank2_release",
    }
