"""Render the fresh Path-normalized rank-29 release observer."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from evals.fleet import hosted_glm_rank29_a3a4_c2_package_v1 as source
from evals.fleet import hosted_glm_rank29_a3a4_c2_release_v2 as release
from evals.fleet import self_hosted


def render(root: Path, *, authorized: bool = False) -> dict[str, Any]:
    value = copy.deepcopy(source.render_release(root, authorized=False))
    configmap, job = value["objects"]["items"]
    configmap["metadata"]["name"] = release.CONFIGMAP_NAME
    configmap["data"].update(
        {
            "release.py": (
                root / "evals/fleet/hosted_glm_rank29_a3a4_c2_release_v2.py"
            ).read_text(),
            "run.sh": (
                root
                / "evals/fleet/scripts/run_hosted_glm_rank29_a3a4_c2_release_v2.sh"
            ).read_text(),
        }
    )
    source._identity(job, release.JOB_NAME, release.CONFIGMAP_NAME)  # noqa: SLF001
    job["metadata"]["annotations"][
        "cyber-post-train.fleet.ai/launch-authorized"
    ] = str(authorized).lower()
    objects = {"apiVersion": "v1", "kind": "List", "items": [configmap, job]}
    return {
        "objects": objects,
        "package_sha256": self_hosted.sha256(self_hosted.canonical_json(objects)),
        "launch_authorized": authorized,
        "scored_launch_authorized": False,
        "model_calls_authorized": False,
        "task_session_verifier_calls_authorized": False,
        **source.scored_bindings(root),
    }
