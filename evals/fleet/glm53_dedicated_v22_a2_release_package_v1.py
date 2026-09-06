"""Render GLM v22 attempt2 duplicate release observer."""

from __future__ import annotations

import copy
from pathlib import Path

from evals.fleet import configmap_python_closure_v1 as closure
from evals.fleet import glm53_dedicated_v21_a2_release_package_v1 as old
from evals.fleet import glm53_dedicated_v22_a2_package_v1 as controller
from evals.fleet import glm53_dedicated_v22_a2_release_v1 as release
from evals.fleet import self_hosted


def render(root: Path):
    package = controller.render(root)
    prior = old.render(root)
    data = copy.deepcopy(prior["objects"]["items"][0]["data"])
    data["v21_canary.py"] = (root / "evals/fleet/glm53_dedicated_v21_a2_v1.py").read_text()
    data["dedicated_canary.py"] = (root / "evals/fleet/glm53_dedicated_v22_a2_v1.py").read_text()
    data["dedicated_runtime.py"] = (
        root / "evals/fleet/glm53_dedicated_v22_a2_runtime_v1.py"
    ).read_text()
    data["release.py"] = (root / "evals/fleet/glm53_dedicated_v22_a2_release_v1.py").read_text()
    data["run.sh"] = (
        data["run.sh"]
        .replace(
            "dedicated_canary.py:glm53_dedicated_v21_a2_v1.py",
            "v21_canary.py:glm53_dedicated_v21_a2_v1.py "
            "dedicated_canary.py:glm53_dedicated_v22_a2_v1.py",
        )
        .replace(
            "dedicated_runtime.py:glm53_dedicated_v20_a2_runtime_v1.py",
            "dedicated_runtime.py:glm53_dedicated_v22_a2_runtime_v1.py",
        )
        .replace(
            "release.py:glm53_dedicated_v21_a2_release_v1.py",
            "release.py:glm53_dedicated_v22_a2_release_v1.py",
        )
        .replace("glm53_dedicated_v21_a2_release_v1", "glm53_dedicated_v22_a2_release_v1")
    )
    closure.validate(data, dynamic_import_allowlist=controller.old.old.DORMANT_BUILD_ONLY_IMPORTS)
    data["controller-package.json"] = self_hosted.canonical_json(
        {key: value for key, value in package.items() if key != "objects"}
    ).decode()
    data["parity.json"] = (root / controller.PARITY).read_text()
    data["binding.json"] = (root / controller.BINDING).read_text()
    configmap = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": release.CONFIGMAP_NAME, "namespace": "fleet-train-jobs"},
        "immutable": True,
        "data": data,
    }
    job = copy.deepcopy(prior["objects"]["items"][1])
    job["metadata"]["name"] = release.JOB_NAME
    job["metadata"]["labels"]["cyber-post-train.fleet.ai/experiment"] = release.JOB_NAME
    pod = job["spec"]["template"]
    pod["metadata"]["labels"]["cyber-post-train.fleet.ai/experiment"] = release.JOB_NAME
    pod["spec"]["volumes"][0]["configMap"]["name"] = release.CONFIGMAP_NAME
    for env in pod["spec"]["containers"][0]["env"]:
        if env["name"] == "DEDICATED_SERVICE_ORIGIN":
            env.pop("valueFrom", None)
            env["value"] = controller.ORIGIN
    objects = {"apiVersion": "v1", "kind": "List", "items": [configmap, job]}
    return {
        "objects": objects,
        "package_sha256": self_hosted.sha256(self_hosted.canonical_json(objects)),
        "launch_authorized": True,
    }
