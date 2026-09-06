"""Run the exact v3 controller package to the preclaim boundary on CPU."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from evals.fleet import hosted_glm_s1_r2_c2_package_v3 as controller
from evals.fleet import self_hosted

JOB_NAME = "chris-glm53-exact100-hosted-s1-r002-c2-bootstrap-v1"
CONFIGMAP_NAME = JOB_NAME + "-run"
SFS_ROOT = "/mnt/sfs/jobs/" + JOB_NAME


def render(root: Path, *, release_path: Path) -> dict[str, Any]:
    configmap, job = copy.deepcopy(controller.render(root, release_path=release_path)["objects"]["items"])
    configmap["metadata"]["name"] = CONFIGMAP_NAME
    job["metadata"]["name"] = JOB_NAME
    job["metadata"]["labels"]["cyber-post-train.fleet.ai/experiment"] = JOB_NAME
    job["metadata"]["annotations"]["cyber-post-train.fleet.ai/launch-authorized"] = "false"
    pod = job["spec"]["template"]
    pod["metadata"]["labels"]["cyber-post-train.fleet.ai/experiment"] = JOB_NAME
    pod["spec"]["volumes"][0]["configMap"]["name"] = CONFIGMAP_NAME
    pod["spec"]["containers"][0]["env"].extend([
        {"name": "HOSTED_BOOTSTRAP_ONLY", "value": "1"},
        {"name": "HOSTED_BOOTSTRAP_RECEIPT", "value": SFS_ROOT + "/BOOTSTRAP.json"},
    ])
    objects = {"apiVersion": "v1", "kind": "List", "items": [configmap, job]}
    return {
        "objects": objects,
        "package_sha256": self_hosted.sha256(self_hosted.canonical_json(objects)),
        "launch_authorized": False,
        "claims_authorized": False,
    }
