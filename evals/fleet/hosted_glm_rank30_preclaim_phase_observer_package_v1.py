"""Render the held score-free rank-30 preclaim phase observer."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from evals.fleet import hosted_glm_rank30_preclaim_phase_bootstrap_v1 as bootstrap
from evals.fleet import hosted_glm_rank30_preclaim_phase_observer_v1 as diagnostic
from evals.fleet import hosted_glm_rank30_single_slot_package_v4 as scored
from evals.fleet import hosted_glm_rank30_single_slot_release_diagnostic_package_v6 as template
from evals.fleet import self_hosted

RUN_SCRIPT = "evals/fleet/scripts/run_hosted_glm_rank30_preclaim_phase_observer_v1.sh"
HELD_SCHEMA = "fleet-hosted-glm-rank30-preclaim-phase-observer-held-v1"


def _source_manifest(root: Path) -> tuple[dict[str, str], dict[str, Any]]:
    source = scored._data(root)  # noqa: SLF001
    source_digest = self_hosted.sha256(self_hosted.canonical_json(source))
    if source_digest != diagnostic.SOURCE_PACKAGE_SHA256:
        raise RuntimeError("rank-30 observer frozen source package drifted")
    files: dict[str, Any] = {}
    for key, relative in scored.PATHS.items():
        files[key] = {
            "relative_path": relative,
            "sha256": self_hosted.sha256(source[key].encode()),
        }
    manifest = {
        "schema_version": bootstrap.SCHEMA,
        "source_package_sha256": source_digest,
        "files": files,
    }
    return source, manifest


def render(root: Path, *, authorized: bool = False) -> dict[str, Any]:
    source, manifest = _source_manifest(root)
    value = copy.deepcopy(template.render(root, authorized=False))
    _old_configmap, job = value["objects"]["items"]
    data = {"source__" + key: text for key, text in source.items()}
    data.update(
        {
            "bootstrap.py": (
                root
                / "evals/fleet/hosted_glm_rank30_preclaim_phase_bootstrap_v1.py"
            ).read_text(),
            "diagnostic.py": (
                root / "evals/fleet/hosted_glm_rank30_preclaim_phase_observer_v1.py"
            ).read_text(),
            "frozen-source-manifest.json": json.dumps(
                manifest, sort_keys=True, separators=(",", ":")
            ),
            "run.sh": (root / RUN_SCRIPT).read_text(),
        }
    )
    configmap = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": diagnostic.CONFIGMAP_NAME,
            "namespace": "fleet-train-jobs",
            "labels": {
                "cyber-post-train.fleet.ai/owner": "chris",
                "cyber-post-train.fleet.ai/experiment": diagnostic.JOB_NAME,
            },
        },
        "immutable": True,
        "data": data,
    }
    job["metadata"]["name"] = diagnostic.JOB_NAME
    job["metadata"]["labels"][
        "cyber-post-train.fleet.ai/experiment"
    ] = diagnostic.JOB_NAME
    job["metadata"]["annotations"][
        "cyber-post-train.fleet.ai/create-once"
    ] = "true"
    job["metadata"]["annotations"][
        "cyber-post-train.fleet.ai/launch-authorized"
    ] = str(authorized).lower()
    pod = job["spec"]["template"]
    pod["metadata"]["labels"][
        "cyber-post-train.fleet.ai/experiment"
    ] = diagnostic.JOB_NAME
    pod["spec"]["volumes"][0]["configMap"]["name"] = diagnostic.CONFIGMAP_NAME
    job["spec"]["activeDeadlineSeconds"] = 600
    job["spec"]["backoffLimit"] = 0
    container = pod["spec"]["containers"][0]
    container["env"] = [
        row for row in container["env"] if row["name"] in {"JOB_UID", "POD_UID"}
    ]
    container["command"] = ["/bin/bash", "/bootstrap/run.sh"]
    objects = {"apiVersion": "v1", "kind": "List", "items": [configmap, job]}
    if len(json.dumps(configmap).encode()) >= 900_000:
        raise RuntimeError("rank-30 observer ConfigMap exceeds safety budget")
    return {
        "objects": objects,
        "package_sha256": self_hosted.sha256(self_hosted.canonical_json(objects)),
        "source_package_sha256": diagnostic.SOURCE_PACKAGE_SHA256,
        "source_manifest_sha256": self_hosted.sha256(
            self_hosted.canonical_json(manifest)
        ),
        "launch_authorized": authorized,
        "scored_successor_launch_authorized": False,
        "model_calls_authorized": False,
        "fleet_task_session_verifier_scoring_calls_authorized": False,
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
        "observer_job_name": diagnostic.JOB_NAME,
        "observer_configmap_name": diagnostic.CONFIGMAP_NAME,
        "output_path": str(diagnostic.OUTPUT_PATH),
        "failed_controller_job_uid": diagnostic.FAILED_JOB_UID,
        "failed_controller_pod_uid": diagnostic.FAILED_POD_UID,
        "release_file_sha256": diagnostic.RELEASE_FILE_SHA256,
        "release_receipt_sha256": diagnostic.RELEASE_RECEIPT_SHA256,
        "runtime_plan_sha256": diagnostic.RUNTIME_PLAN_SHA256,
        "phase_order": [name for name, _function in diagnostic.PHASES],
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
