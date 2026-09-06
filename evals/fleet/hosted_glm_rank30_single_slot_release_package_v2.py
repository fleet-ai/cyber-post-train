"""Render the fresh, dependency-closed rank-30 release observer."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from evals.fleet import hosted_glm_rank30_single_slot_release_package_v1 as prior
from evals.fleet import hosted_glm_rank30_single_slot_release_v2 as release
from evals.fleet import self_hosted


def render(root: Path, *, authorized: bool = False) -> dict[str, Any]:
    value = copy.deepcopy(prior.render(root, authorized=False))
    configmap, job = value["objects"]["items"]
    configmap["metadata"]["name"] = release.CONFIGMAP_NAME
    additions = {
        "rank29_successor.py": (
            "evals/fleet/hosted_glm_rank29_a3a4_c2_successor_v1.py"
        ),
        "rank29_runtime.py": "evals/fleet/hosted_glm_rank29_a3a4_c2_runtime_v1.py",
        "release_v2.py": "evals/fleet/hosted_glm_rank30_single_slot_release_v2.py",
        "run.sh": (
            "evals/fleet/scripts/run_hosted_glm_rank30_single_slot_release_v2.sh"
        ),
    }
    configmap["data"].update(
        {name: (root / relative).read_text() for name, relative in additions.items()}
    )
    prior.base.source._identity(  # noqa: SLF001
        job, release.JOB_NAME, release.CONFIGMAP_NAME
    )
    job["metadata"]["annotations"][
        "cyber-post-train.fleet.ai/launch-authorized"
    ] = str(authorized).lower()
    objects = {"apiVersion": "v1", "kind": "List", "items": [configmap, job]}
    return {
        "objects": objects,
        "package_sha256": self_hosted.sha256(self_hosted.canonical_json(objects)),
        "source_package_sha256": value["source_package_sha256"],
        "launch_authorized": authorized,
        "scoring_authorized": False,
        "model_calls_authorized": False,
        "supersedes_failed_observer_job_uid": (
            "2b538d17-d5a8-4841-b152-47cfd737e0d9"
        ),
    }
