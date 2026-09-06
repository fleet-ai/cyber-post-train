"""Render the held create-once task-scoped Qwen identity selector package."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evals.fleet import qwen_hosted_generation18_package as job_base
from evals.fleet import qwen_hosted_identity_selector_v1 as v1
from evals.fleet import qwen_hosted_identity_selector_v2 as selector
from evals.fleet import qwen_hosted_rank17_g22_release_observer_v1 as base
from evals.fleet import score_blind_session_inventory_v1 as stream

JOB_NAME = "chris-q38-hosted-identity-selector-v2"
CONFIGMAP_NAME = JOB_NAME + "-package"
OUTPUT_ROOT = selector.OUTPUT_ROOT
HELD_PATH = Path(
    "docs/evidence/qwen38-study/2026-09-06-qwen38-hosted-identity-selector-held-v2.json"
)
SECRET_NAME = "chris-cyber-opencode-evals-v2"


def build_binding(root: Path) -> dict[str, Any]:
    selection_path = root / v1.SELECTION_PATH
    if base.sha256(selection_path.read_bytes()) != v1.SELECTION_FILE_SHA256:
        raise ValueError("selector selection file drifted")
    selection = json.loads(selection_path.read_text())
    if (
        not isinstance(selection, dict)
        or selection.get("selection_sha256") != v1.SELECTION_SHA256
        or selection.get("selected_count") != 100
        or not isinstance(selection.get("tasks"), list)
    ):
        raise ValueError("selector selection authority drifted")
    roster = [
        {"rank": row.get("rank"), "task_key": row.get("task_key")}
        for row in selection["tasks"]
    ]
    binding = {
        "schema_version": selector.BINDING_SCHEMA,
        "selection_path": v1.SELECTION_PATH,
        "selection_file_sha256": v1.SELECTION_FILE_SHA256,
        "selection_sha256": v1.SELECTION_SHA256,
        "roster_sha256": v1.ROSTER_SHA256,
        "task_count": 100,
        "task_roster": roster,
        "session_model": v1.EXPECTED_MODEL,
        "authoritative_tally": v1.EXPECTED_TALLY,
        "output_root": OUTPUT_ROOT,
    }
    binding["binding_sha256"] = base.binding_digest(binding)
    selector.validate_binding(binding)
    return binding


def render(root: Path) -> dict[str, Any]:
    files = {
        "binding.json": json.dumps(build_binding(root), sort_keys=True, separators=(",", ":"))
        + "\n",
        "qwen_hosted_identity_selector_v2.py": Path(selector.__file__).read_text(),
        "qwen_hosted_identity_selector_v1.py": Path(v1.__file__).read_text(),
        "qwen_hosted_rank17_g22_release_observer_v1.py": Path(base.__file__).read_text(),
        "score_blind_session_inventory_v1.py": Path(stream.__file__).read_text(),
    }
    package = base.seal(
        {
            "schema_version": selector.PACKAGE_SCHEMA,
            "files": {name: base.sha256(value.encode()) for name, value in files.items()},
            "file_count": len(files),
        }
    )
    files["package-source.json"] = json.dumps(
        package, sort_keys=True, separators=(",", ":")
    ) + "\n"
    configmap = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": CONFIGMAP_NAME, "namespace": base.NAMESPACE},
        "immutable": True,
        "data": files,
    }
    job = job_base._base_job(JOB_NAME, CONFIGMAP_NAME, scored=False)  # noqa: SLF001
    job["metadata"]["annotations"].update(
        {
            "cyber-post-train.fleet.ai/create-once": "true",
            "cyber-post-train.fleet.ai/launch-authorized": "false",
            "cyber-post-train.fleet.ai/diagnostic-only": "true",
            "cyber-post-train.fleet.ai/score-free": "true",
        }
    )
    experiment = "q38-hosted-identity-selector-v2"
    job["metadata"]["labels"]["cyber-post-train.fleet.ai/experiment"] = experiment
    job["spec"]["template"]["metadata"]["labels"][
        "cyber-post-train.fleet.ai/experiment"
    ] = experiment
    container = job["spec"]["template"]["spec"]["containers"][0]
    container["args"] = [
        "exec python /bootstrap/qwen_hosted_identity_selector_v2.py "
        "--binding /bootstrap/binding.json --package-source /bootstrap/package-source.json "
        f"--output {OUTPUT_ROOT}/OBSERVATION.json"
    ]
    container["env"] = [row for row in container["env"] if row.get("name") != "FLEET_API_KEY"]
    container["env"].append(
        {
            "name": "FLEET_API_KEY",
            "valueFrom": {"secretKeyRef": {"name": SECRET_NAME, "key": "FLEET_API_KEY"}},
        }
    )
    container["resources"] = {
        "requests": {"cpu": "100m", "memory": "256Mi", "ephemeral-storage": "256Mi"},
        "limits": {"cpu": "1", "memory": "1Gi", "ephemeral-storage": "1Gi"},
    }
    return {"apiVersion": "v1", "kind": "List", "items": [configmap, job]}


def held(root: Path) -> dict[str, Any]:
    rendered = render(root)
    return base.seal(
        {
            "schema_version": "fleet-qwen38-hosted-identity-selector-held-v2",
            "status": "HELD_PENDING_INDEPENDENT_REVIEW",
            "launch_authorized": False,
            "scoring_authorized": False,
            "job_name": JOB_NAME,
            "configmap_name": CONFIGMAP_NAME,
            "output_root": OUTPUT_ROOT,
            "selection_file_sha256": v1.SELECTION_FILE_SHA256,
            "selection_sha256": v1.SELECTION_SHA256,
            "roster_sha256": v1.ROSTER_SHA256,
            "binding_sha256": build_binding(root)["binding_sha256"],
            "authoritative_tally": v1.EXPECTED_TALLY,
            "manifest_sha256": base.sha256(base.canonical(rendered)),
            "create_once": True,
            "score_free": True,
            "task_scoped_session_inventory": True,
            "emits_only_aggregate_counts_and_earliest_clear_rank": True,
            "protected_values_materialized": False,
            "side_effects": {
                "model_calls": 0,
                "task_calls": 0,
                "session_mutations": 0,
                "verifier_calls": 0,
                "scoring_calls": 0,
                "api_mutations": 0,
                "kubernetes_objects_created": 0,
            },
        }
    )
