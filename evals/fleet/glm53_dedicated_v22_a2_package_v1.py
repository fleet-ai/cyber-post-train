"""Render held GLM v22 rank51 attempt2 package."""

from __future__ import annotations

import copy
from pathlib import Path

from evals.fleet import configmap_python_closure_v1 as closure
from evals.fleet import glm53_dedicated_v21_a2_package_v1 as old
from evals.fleet import glm53_dedicated_v22_a2_v1 as canary
from evals.fleet import self_hosted

PARITY = Path(
    "docs/evidence/glm53-study/2026-09-06-glm53-dedicated-v22-actual-opencode-parity.json"
)
BINDING = Path("docs/evidence/glm53-study/2026-09-06-glm53-dedicated-v22-server-binding.json")
RELEASE = Path("docs/evidence/glm53-study/2026-09-06-glm53-v22-a2-release-v2.json")
ORIGIN = "http://ft-run-d2dff491-gfw5m-head-svc.fleet-train-jobs.svc.cluster.local:8000"
SOURCE_CONFIGMAP = canary.JOB_NAME + "-source"
EVIDENCE_CONFIGMAP = canary.JOB_NAME + "-evidence"


def render(root: Path):
    prior = old.render(root)
    data = copy.deepcopy(prior["objects"]["items"][0]["data"])
    data["v21_canary.py"] = (root / "evals/fleet/glm53_dedicated_v21_a2_v1.py").read_text()
    data["dedicated_canary.py"] = (root / "evals/fleet/glm53_dedicated_v22_a2_v1.py").read_text()
    data["dedicated_runtime.py"] = (
        root / "evals/fleet/glm53_dedicated_v22_a2_runtime_v1.py"
    ).read_text()
    data["run.sh"] = (
        data["run.sh"]
        .replace(
            "dedicated_canary.py:glm53_dedicated_v21_a2_v1.py",
            "v21_canary.py:glm53_dedicated_v21_a2_v1.py "
            "dedicated_canary.py:glm53_dedicated_v22_a2_v1.py",
        )
        .replace(
            "dedicated_runtime.py:glm53_dedicated_v21_a2_runtime_v1.py",
            "dedicated_runtime.py:glm53_dedicated_v22_a2_runtime_v1.py",
        )
        .replace("glm53_dedicated_v21_a2_runtime_v1", "glm53_dedicated_v22_a2_runtime_v1")
    )
    closure.validate(data, dynamic_import_allowlist=old.old.DORMANT_BUILD_ONLY_IMPORTS)
    held = canary.build_plan(
        root, service_origin=ORIGIN, parity_path=root / PARITY, binding_path=root / BINDING
    )
    source = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": SOURCE_CONFIGMAP, "namespace": "fleet-train-jobs"},
        "immutable": True,
        "data": data,
    }
    evidence = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": EVIDENCE_CONFIGMAP, "namespace": "fleet-train-jobs"},
        "immutable": True,
        "data": {
            "parity.json": (root / PARITY).read_text(),
            "binding.json": (root / BINDING).read_text(),
            "release.json": (root / RELEASE).read_text(),
        },
    }
    job = copy.deepcopy(prior["objects"]["items"][1])
    job["metadata"]["name"] = canary.JOB_NAME
    job["metadata"]["labels"]["cyber-post-train.fleet.ai/experiment"] = canary.JOB_NAME
    pod = job["spec"]["template"]
    pod["metadata"]["labels"]["cyber-post-train.fleet.ai/experiment"] = canary.JOB_NAME
    pod["spec"]["volumes"][0]["configMap"]["name"] = SOURCE_CONFIGMAP
    next(v for v in pod["spec"]["volumes"] if v["name"] == "evidence")["configMap"]["name"] = (
        EVIDENCE_CONFIGMAP
    )
    for env in pod["spec"]["containers"][0]["env"]:
        if env["name"] == "JOB_NAME":
            env["value"] = canary.JOB_NAME
        if env["name"] == "DEDICATED_SERVICE_ORIGIN":
            env.pop("valueFrom", None)
            env["value"] = ORIGIN
    objects = {"apiVersion": "v1", "kind": "List", "items": [source, evidence, job]}
    body = {
        "schema_version": "fleet-glm53-dedicated-v22-a2-package-v1",
        "status": "READY_HELD",
        "launch_authorized": False,
        "job_name": canary.JOB_NAME,
        "held_plan_sha256": held["plan_sha256"],
        "reserved_cell_ids": held["whole_task_reservation"]["cell_ids"],
        "reserved_execution_ids": held["whole_task_reservation"]["execution_ids"],
        "objects_sha256": self_hosted.sha256(self_hosted.canonical_json(objects)),
    }
    body["package_sha256"] = self_hosted.digest_without(body, "package_sha256")
    return {**body, "objects": objects}
