"""Render the held v6-gated rank-30 release observer."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from evals.fleet import hosted_glm_rank30_single_slot_package_v4 as scored
from evals.fleet import hosted_glm_rank30_single_slot_release_package_v3 as prior
from evals.fleet import hosted_glm_rank30_single_slot_release_v4 as release
from evals.fleet import hosted_glm_rank30_single_slot_v5 as successor
from evals.fleet import self_hosted


def render(root: Path, *, authorized: bool = False) -> dict[str, Any]:
    value = copy.deepcopy(prior.render(root, authorized=False))
    configmap, job = value["objects"]["items"]
    configmap["metadata"]["name"] = release.CONFIGMAP_NAME
    configmap["data"].update(
        {
            "single_slot_v3.py": (
                root / "evals/fleet/hosted_glm_rank30_single_slot_v3.py"
            ).read_text(),
            "single_slot_v4.py": (
                root / "evals/fleet/hosted_glm_rank30_single_slot_v4.py"
            ).read_text(),
            "successor.py": (
                root / "evals/fleet/hosted_glm_rank30_single_slot_v5.py"
            ).read_text(),
            "release_v3.py": configmap["data"]["release.py"],
            "release.py": (
                root / "evals/fleet/hosted_glm_rank30_single_slot_release_v4.py"
            ).read_text(),
            "run.sh": (
                root
                / "evals/fleet/scripts/run_hosted_glm_rank30_single_slot_release_v4.sh"
            ).read_text(),
        }
    )
    prior.prior.prior.base.source._identity(  # noqa: SLF001
        job, release.JOB_NAME, release.CONFIGMAP_NAME
    )
    job["metadata"]["annotations"][
        "cyber-post-train.fleet.ai/launch-authorized"
    ] = str(authorized).lower()
    source_sha = scored.source_package_sha256(root)
    env = job["spec"]["template"]["spec"]["containers"][0]["env"]
    env[:] = [row for row in env if row["name"] != "GLM_HOSTED_R30_SOURCE_SHA256"]
    env.append({"name": "GLM_HOSTED_R30_SOURCE_SHA256", "value": source_sha})
    objects = {"apiVersion": "v1", "kind": "List", "items": [configmap, job]}
    return {
        "objects": objects,
        "package_sha256": self_hosted.sha256(self_hosted.canonical_json(objects)),
        "source_package_sha256": source_sha,
        "launch_authorized": authorized,
        "scoring_authorized": False,
        "model_calls_authorized": False,
        "v6_diagnostic_receipt_sha256": successor.DIAGNOSTIC_SELF_SHA256,
    }
