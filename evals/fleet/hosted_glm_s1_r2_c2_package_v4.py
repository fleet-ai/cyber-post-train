"""Render the engine-complete rank-2 v4 controller package."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from evals.fleet import hosted_glm_exact_bulk_package_v1 as base
from evals.fleet import hosted_glm_s1_r2_c2_package_v3 as closure
from evals.fleet import hosted_glm_s1_r2_c2_release_v5 as release
from evals.fleet import hosted_glm_s1_r2_c2_successor_v4 as successor
from evals.fleet import self_hosted


def render(root: Path, *, release_path: Path | None = None, bootstrap_path: Path | None = None, bootstrap_mode: bool = False) -> dict[str, Any]:
    plan = successor.validate_all(root)[successor.CONTROLLER]
    if not bootstrap_mode and (release_path is None) != (bootstrap_path is None):
        raise ValueError("scored authorization requires release and engine bootstrap")
    authorized = release_path is not None and bootstrap_path is not None
    configmap, job = copy.deepcopy(base.render(root)["objects"]["items"][:2])
    configmap["metadata"]["name"] = successor.CONFIGMAP_NAME
    additions = {
        "original_release.py": "evals/fleet/hosted_glm_exact_bulk_release_v1.py",
        "prior_successor_v1.py": "evals/fleet/hosted_glm_s1_r2_c2_successor_v1.py",
        "prior_successor_v2.py": "evals/fleet/hosted_glm_s1_r2_c2_successor_v2.py",
        "prior_successor_v3.py": "evals/fleet/hosted_glm_s1_r2_c2_successor_v3.py",
        "prior_release.py": "evals/fleet/hosted_glm_s1_r2_c2_release_v1.py",
        "release_v2.py": "evals/fleet/hosted_glm_s1_r2_c2_release_v2.py",
        "successor.py": "evals/fleet/hosted_glm_s1_r2_c2_successor_v4.py",
        "successor_release.py": "evals/fleet/hosted_glm_s1_r2_c2_release_v5.py",
        "successor_runtime.py": "evals/fleet/hosted_glm_s1_r2_c2_runtime_v4.py",
        "run.sh": "evals/fleet/scripts/run_hosted_glm_s1_r2_c2_successor_v4.sh",
    }
    for name, path in additions.items():
        configmap["data"][name] = (root / path).read_text()
    closure.validate_install_closure(configmap["data"])
    if release_path is not None:
        receipt = successor.load(release_path)
        if any((receipt.get("schema_version") != release.SCHEMA, receipt.get("status") != "CLEAR", receipt.get("successor_job") != successor.JOB_NAME, receipt.get("plan_sha256") != plan["plan_sha256"], receipt.get("receipt_sha256") != self_hosted.digest_without(receipt, "receipt_sha256"))):
            raise RuntimeError("rank-2 hosted v4 package release drifted")
        if authorized:
            bootstrap = successor.load(bootstrap_path)
            required = {"schema_version": "fleet-hosted-glm-rank2-controller-bootstrap-v2", "status": "PASSED_ENGINE_PRECLAIM", "controller_job": successor.JOB_NAME, "runtime_plan_sha256": receipt["plan_sha256"], "release_receipt_sha256": receipt["receipt_sha256"], "bulk_adapter_members_valid": True, "engine_runtime_rebuild_equal": True, "claims": 0, "model_requests": 0, "task_instance_session_verifier_scoring_calls": 0}
            if any(bootstrap.get(key) != value for key, value in required.items()) or bootstrap.get("receipt_sha256") != self_hosted.digest_without(bootstrap, "receipt_sha256"):
                raise RuntimeError("rank-2 hosted v4 engine bootstrap drifted")
    job["metadata"]["name"] = successor.JOB_NAME
    job["metadata"]["labels"]["cyber-post-train.fleet.ai/experiment"] = successor.JOB_NAME
    job["metadata"]["annotations"]["cyber-post-train.fleet.ai/launch-authorized"] = str(authorized).lower()
    pod = job["spec"]["template"]
    pod["metadata"]["labels"]["cyber-post-train.fleet.ai/experiment"] = successor.JOB_NAME
    pod["spec"]["volumes"][0]["configMap"]["name"] = successor.CONFIGMAP_NAME
    objects = {"apiVersion": "v1", "kind": "List", "items": [configmap, job]}
    return {"objects": objects, "package_sha256": self_hosted.sha256(self_hosted.canonical_json(objects)), "launch_authorized": authorized}
