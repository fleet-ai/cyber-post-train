"""Render the import-complete fresh rank-3 tail release observer."""

import copy
from pathlib import Path
from typing import Any

from evals.fleet import hosted_glm_rank3_a2a4_c2_package_v2 as prior
from evals.fleet import hosted_glm_rank3_a2a4_c2_release_v3 as release
from evals.fleet import self_hosted


def render(root: Path) -> dict[str, Any]:
    value = copy.deepcopy(prior.render_release(root))
    configmap, job = value["objects"]["items"]
    configmap["metadata"]["name"] = release.CONFIGMAP_NAME
    configmap["data"].update(
        {
            "prior_runtime.py": (root / "evals/fleet/hosted_glm_rank3_a2a4_c2_runtime_v1.py").read_text(),
            "release.py": (root / "evals/fleet/hosted_glm_rank3_a2a4_c2_release_v3.py").read_text(),
            "run.sh": (root / "evals/fleet/scripts/run_hosted_glm_rank3_a2a4_c2_release_v3.sh").read_text(),
        }
    )
    job["metadata"]["name"] = release.JOB_NAME
    job["metadata"]["labels"]["cyber-post-train.fleet.ai/experiment"] = release.JOB_NAME
    job["spec"]["template"]["metadata"]["labels"]["cyber-post-train.fleet.ai/experiment"] = release.JOB_NAME
    job["spec"]["template"]["spec"]["volumes"][0]["configMap"]["name"] = release.CONFIGMAP_NAME
    objects = {"apiVersion": "v1", "kind": "List", "items": [configmap, job]}
    return {"objects": objects, "package_sha256": self_hosted.sha256(self_hosted.canonical_json(objects)), "launch_authorized": True}
