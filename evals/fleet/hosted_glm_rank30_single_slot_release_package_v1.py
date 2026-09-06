"""Render the create-once score-blind rank-30 single-slot release observer."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from evals.fleet import hosted_glm_rank29_a3a4_c2_release_package_v2 as base
from evals.fleet import hosted_glm_rank30_single_slot_package_v1 as scored
from evals.fleet import hosted_glm_rank30_single_slot_release_v1 as release
from evals.fleet import self_hosted


def render(root: Path, *, authorized: bool = False) -> dict[str, Any]:
    value = copy.deepcopy(base.render(root, authorized=False))
    configmap, job = value["objects"]["items"]
    configmap["metadata"]["name"] = release.CONFIGMAP_NAME
    additions = {
        "base_engine.py": "evals/fleet/exact_pass4_bulk_runtime_v3.py",
        "source.py": "evals/fleet/hosted_glm_exact_bulk_v1.py",
        "source_runtime.py": "evals/fleet/hosted_glm_exact_bulk_runtime_v1.py",
        "whole.py": "evals/fleet/hosted_glm_whole_task_successor_v1.py",
        "engine.py": "evals/fleet/hosted_glm_whole_task_engine_v1.py",
        "successor.py": "evals/fleet/hosted_glm_rank30_single_slot_v1.py",
        "kube.py": "evals/fleet/hosted_glm_rank29_a3a4_c2_release_v2.py",
        "release.py": "evals/fleet/hosted_glm_rank30_single_slot_release_v1.py",
        "run.sh": "evals/fleet/scripts/run_hosted_glm_rank30_single_slot_release_v1.sh",
    }
    configmap["data"].update(
        {name: (root / relative).read_text() for name, relative in additions.items()}
    )
    base.source._identity(job, release.JOB_NAME, release.CONFIGMAP_NAME)  # noqa: SLF001
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
    }
