"""Render the fresh held/released rank-2 hosted GLM successor."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from evals.fleet import hosted_glm_exact_bulk_package_v1 as base
from evals.fleet import hosted_glm_s1_r2_c2_runtime_v2 as runtime
from evals.fleet import hosted_glm_s1_r2_c2_successor_v2 as successor
from evals.fleet import self_hosted


def render(root: Path, *, release_path: Path | None = None) -> dict[str, Any]:
    successor.validate_all(root)
    authorized = release_path is not None
    configmap, job = copy.deepcopy(base.render(root)["objects"]["items"][:2])
    configmap["metadata"]["name"] = successor.CONFIGMAP_NAME
    additions = {
        "successor.py": "evals/fleet/hosted_glm_s1_r2_c2_successor_v2.py",
        "successor_release.py": "evals/fleet/hosted_glm_s1_r2_c2_release_v2.py",
        "successor_runtime.py": "evals/fleet/hosted_glm_s1_r2_c2_runtime_v2.py",
        "prior_successor.py": "evals/fleet/hosted_glm_s1_r2_c2_successor_v1.py",
        "prior_release.py": "evals/fleet/hosted_glm_s1_r2_c2_release_v1.py",
        "run.sh": "evals/fleet/scripts/run_hosted_glm_s1_r2_c2_successor_v2.sh",
    }
    for name, path in additions.items():
        configmap["data"][name] = (root / path).read_text()
    if authorized:
        receipt = successor.load(release_path)
        # Full live-plan validation runs again inside the Pod; package creation
        # admits only a self-digesting release tied to this fresh identity.
        if any(
            (
                receipt.get("schema_version") != "fleet-hosted-glm-s1-r2-c2-successor-release-v2",
                receipt.get("status") != "CLEAR",
                receipt.get("successor_job") != successor.JOB_NAME,
                receipt.get("successor_configmap") != successor.CONFIGMAP_NAME,
                receipt.get("receipt_sha256") != self_hosted.digest_without(receipt, "receipt_sha256"),
            )
        ):
            raise RuntimeError("rank-2 hosted v2 package release drifted")
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
        "reason": None if authorized else "requires_fresh_rank2_v2_release",
    }
