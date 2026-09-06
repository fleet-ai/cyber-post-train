"""Render the held fresh rank-30 current-peer phase diagnostic."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from evals.fleet import hosted_glm_rank30_single_slot_release_diagnostic_v4 as diagnostic
from evals.fleet import hosted_glm_rank30_single_slot_release_package_v3 as prior
from evals.fleet import self_hosted


def render(root: Path, *, authorized: bool = False) -> dict[str, Any]:
    value = copy.deepcopy(prior.render(root, authorized=False))
    configmap, job = value["objects"]["items"]
    configmap["metadata"]["name"] = diagnostic.CONFIGMAP_NAME
    configmap["data"].update(
        {
            "diagnostic.py": (
                root
                / "evals/fleet/hosted_glm_rank30_single_slot_release_diagnostic_v4.py"
            ).read_text(),
            "run.sh": (
                root
                / "evals/fleet/scripts/run_hosted_glm_rank30_single_slot_release_diagnostic_v4.sh"
            ).read_text(),
        }
    )
    prior.prior.prior.base.source._identity(  # noqa: SLF001
        job, diagnostic.JOB_NAME, diagnostic.CONFIGMAP_NAME
    )
    job["metadata"]["annotations"][
        "cyber-post-train.fleet.ai/launch-authorized"
    ] = str(authorized).lower()
    container = job["spec"]["template"]["spec"]["containers"][0]
    container["env"] = [
        row for row in container["env"]
        if row["name"] not in {"FLEET_API_KEY", "GLM_HOSTED_R30_SOURCE_SHA256"}
    ]
    objects = {"apiVersion": "v1", "kind": "List", "items": [configmap, job]}
    return {
        "objects": objects,
        "package_sha256": self_hosted.sha256(self_hosted.canonical_json(objects)),
        "launch_authorized": authorized,
        "scored_launch_authorized": False,
        "model_calls_authorized": False,
        "task_session_verifier_calls_authorized": False,
        "supersedes_failed_observer_job_uid": diagnostic.FAILED_RELEASE_JOB_UID,
    }
