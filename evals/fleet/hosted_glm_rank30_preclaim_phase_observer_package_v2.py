"""Render the corrected create-once rank-30 preclaim phase observer."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from evals.fleet import hosted_glm_rank30_preclaim_phase_observer_package_v1 as prior
from evals.fleet import hosted_glm_rank30_preclaim_phase_observer_v2 as observer
from evals.fleet import self_hosted

RUN_SCRIPT = "evals/fleet/scripts/run_hosted_glm_rank30_preclaim_phase_observer_v2.sh"
HELD_SCHEMA = "fleet-hosted-glm-rank30-preclaim-phase-observer-held-v2"


def render(root: Path, *, authorized: bool = False) -> dict[str, Any]:
    value = copy.deepcopy(prior.render(root, authorized=False))
    configmap, job = value["objects"]["items"]
    data = configmap["data"]
    data["diagnostic-v1.py"] = data.pop("diagnostic.py")
    data["diagnostic-v2.py"] = (
        root / "evals/fleet/hosted_glm_rank30_preclaim_phase_observer_v2.py"
    ).read_text()
    data["bootstrap.py"] = (
        root / "evals/fleet/hosted_glm_rank30_preclaim_phase_bootstrap_v2.py"
    ).read_text()
    data["run.sh"] = (root / RUN_SCRIPT).read_text()
    configmap["metadata"]["name"] = observer.CONFIGMAP_NAME
    configmap["metadata"]["labels"][
        "cyber-post-train.fleet.ai/experiment"
    ] = observer.JOB_NAME
    job["metadata"]["name"] = observer.JOB_NAME
    job["metadata"]["labels"][
        "cyber-post-train.fleet.ai/experiment"
    ] = observer.JOB_NAME
    job["metadata"]["annotations"][
        "cyber-post-train.fleet.ai/launch-authorized"
    ] = str(authorized).lower()
    pod = job["spec"]["template"]
    pod["metadata"]["labels"][
        "cyber-post-train.fleet.ai/experiment"
    ] = observer.JOB_NAME
    pod["spec"]["volumes"][0]["configMap"]["name"] = observer.CONFIGMAP_NAME
    objects = {"apiVersion": "v1", "kind": "List", "items": [configmap, job]}
    if len(json.dumps(configmap).encode()) >= 900_000:
        raise RuntimeError("rank-30 observer ConfigMap exceeds safety budget")
    return {
        "objects": objects,
        "package_sha256": self_hosted.sha256(self_hosted.canonical_json(objects)),
        "source_package_sha256": value["source_package_sha256"],
        "source_manifest_sha256": value["source_manifest_sha256"],
        "launch_authorized": authorized,
        "scored_successor_launch_authorized": False,
        "model_calls_authorized": False,
        "fleet_task_session_verifier_scoring_calls_authorized": False,
        "supersedes_failed_job_uid": "98ec91a3-168a-4b87-bb46-e42e16825a25",
        "supersedes_failed_pod_uid": "d96061b5-0ee5-4df9-b578-f3c35db575b8",
    }


def build_held(root: Path, package_commit: str) -> dict[str, Any]:
    rendered = render(root, authorized=False)
    body: dict[str, Any] = {
        "schema_version": HELD_SCHEMA,
        "status": "PASSED_HELD_NO_LAUNCH",
        "package_commit": package_commit,
        "package_sha256": rendered["package_sha256"],
        "source_package_sha256": rendered["source_package_sha256"],
        "source_manifest_sha256": rendered["source_manifest_sha256"],
        "observer_job_name": observer.JOB_NAME,
        "observer_configmap_name": observer.CONFIGMAP_NAME,
        "output_path": str(observer.OUTPUT_PATH),
        "supersedes_failed_job_uid": rendered["supersedes_failed_job_uid"],
        "supersedes_failed_pod_uid": rendered["supersedes_failed_pod_uid"],
        "phase_order": list(prior.diagnostic.PHASE_ORDER),
        "stops_before_provider_session_model": True,
        "create_once": True,
        "model_calls": 0,
        "fleet_task_instance_calls": 0,
        "fleet_session_calls": 0,
        "verifier_calls": 0,
        "scoring_calls": 0,
        "api_mutation_calls": 0,
        "scores_read": False,
        "prompts_traces_flags_read": False,
        "protected_content_included": False,
        "observer_launch_authorized": False,
        "scored_successor_launch_authorized": False,
    }
    body["receipt_sha256"] = self_hosted.digest_without(body, "receipt_sha256")
    return body
