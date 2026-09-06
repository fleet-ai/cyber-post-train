"""Render the held one-shot rank-29 release phase diagnostic."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from evals.fleet import hosted_glm_rank29_a3a4_c2_package_v1 as prior
from evals.fleet import hosted_glm_rank29_a3a4_c2_release_diagnostic_v1 as diagnostic
from evals.fleet import self_hosted


def render(root: Path, *, authorized: bool = False) -> dict[str, Any]:
    value = copy.deepcopy(prior.render_release(root, authorized=False))
    configmap, job = value["objects"]["items"]
    configmap["metadata"]["name"] = diagnostic.CONFIGMAP_NAME
    configmap["data"].update(
        {
            "diagnostic.py": (
                root
                / "evals/fleet/hosted_glm_rank29_a3a4_c2_release_diagnostic_v1.py"
            ).read_text(),
            "source_package.py": (
                root / "evals/fleet/hosted_glm_rank29_a3a4_c2_package_v1.py"
            ).read_text(),
            "run.sh": (
                root
                / "evals/fleet/scripts/run_hosted_glm_rank29_release_diagnostic_v1.sh"
            ).read_text(),
        }
    )
    prior._identity(job, diagnostic.JOB_NAME, diagnostic.CONFIGMAP_NAME)  # noqa: SLF001
    job["metadata"]["annotations"][
        "cyber-post-train.fleet.ai/launch-authorized"
    ] = str(authorized).lower()
    container = job["spec"]["template"]["spec"]["containers"][0]
    container["env"] = [
        row
        for row in container["env"]
        if row["name"]
        not in {"FLEET_API_KEY", "SCORED_SOURCE_SHA256", "SCORED_PACKAGE_TEMPLATE_SHA256"}
    ]
    objects = value["objects"]
    return {
        "objects": objects,
        "package_sha256": self_hosted.sha256(self_hosted.canonical_json(objects)),
        "source_commit": diagnostic.SOURCE_COMMIT,
        "launch_authorized": authorized,
        "scored_launch_authorized": False,
        "model_calls_authorized": False,
        "task_session_verifier_calls_authorized": False,
    }
