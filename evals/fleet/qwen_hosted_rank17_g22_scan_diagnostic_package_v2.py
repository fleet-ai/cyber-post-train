"""Render the held payload-blind rank-17 scan diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from evals.fleet import qwen_hosted_generation18_package as base
from evals.fleet import qwen_hosted_rank17_g22_scan_diagnostic_v2 as diagnostic
from evals.fleet import self_hosted

JOB_NAME = "chris-q38-hosted-r017-release-scan-diag-g22-v2"
CONFIGMAP_NAME = JOB_NAME + "-package"
OUTPUT_ROOT = f"/mnt/sfs/jobs/{JOB_NAME}"
PATH_HASH_SALT = "sha256:126e7fe4c894f44a7f08fab949288354060057a725ae47607fc26f055fb71476"
HELD_PATH = Path(
    "docs/evidence/qwen38-study/2026-09-06-qwen38-hosted-rank17-g22-scan-diagnostic-held-v2.json"
)


def render(root: Path) -> dict[str, Any]:
    source_name = "qwen_hosted_rank17_g22_scan_diagnostic_v2.py"
    data = {source_name: (root / "evals/fleet" / source_name).read_text()}
    package_body = {
        "schema_version": diagnostic.PACKAGE_SCHEMA,
        "files": {source_name: diagnostic.sha256(data[source_name].encode())},
        "file_count": 1,
    }
    data["package-source.json"] = (
        json.dumps(diagnostic.seal(package_body), sort_keys=True, separators=(",", ":")) + "\n"
    )
    configmap = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": CONFIGMAP_NAME, "namespace": "fleet-train-jobs"},
        "immutable": True,
        "data": data,
    }
    job = base._base_job(JOB_NAME, CONFIGMAP_NAME, scored=False)  # noqa: SLF001
    job["metadata"]["annotations"].update(
        {
            "cyber-post-train.fleet.ai/create-once": "true",
            "cyber-post-train.fleet.ai/launch-authorized": "false",
            "cyber-post-train.fleet.ai/diagnostic-only": "true",
            "cyber-post-train.fleet.ai/score-free": "true",
            "cyber-post-train.fleet.ai/incident-job-uid": diagnostic.INCIDENT_JOB_UID,
        }
    )
    job["spec"]["activeDeadlineSeconds"] = 900
    container = job["spec"]["template"]["spec"]["containers"][0]
    container["args"] = [
        f"exec python /bootstrap/{source_name} --output {OUTPUT_ROOT}/DIAGNOSTIC.json "
        f"--path-hash-salt {PATH_HASH_SALT} --package-source /bootstrap/package-source.json"
    ]
    container["env"] = []
    container["resources"] = {
        "requests": {"cpu": "100m", "memory": "256Mi", "ephemeral-storage": "256Mi"},
        "limits": {"cpu": "1", "memory": "1Gi", "ephemeral-storage": "1Gi"},
    }
    return {"apiVersion": "v1", "kind": "List", "items": [configmap, job]}


def held(root: Path) -> dict[str, Any]:
    rendered = render(root)
    configmap, job = rendered["items"]
    return diagnostic.seal(
        {
            "schema_version": "fleet-qwen38-hosted-rank17-g22-scan-diagnostic-held-v2",
            "status": "HELD_PENDING_INDEPENDENT_REVIEW",
            "launch_authorized": False,
            "scoring_authorized": False,
            "incident": {
                "job_uid": diagnostic.INCIDENT_JOB_UID,
                "pod_uid": diagnostic.INCIDENT_POD_UID,
                "failure_code": "scan_receipt_invalid",
            },
            "diagnostic": {
                "job_name": JOB_NAME,
                "configmap_name": CONFIGMAP_NAME,
                "output_root": OUTPUT_ROOT,
                "manifest_sha256": self_hosted.sha256(
                    json.dumps(rendered, sort_keys=True, separators=(",", ":")).encode()
                ),
                "configmap_immutable": configmap["immutable"],
                "create_once": True,
                "score_free": True,
                "preemption_policy": job["spec"]["template"]["spec"]["preemptionPolicy"],
                "priority_class": job["spec"]["template"]["spec"]["priorityClassName"],
            },
            "privacy": {
                "payload_values_decoded": False,
                "paths_emitted": False,
                "scores_read": False,
                "prompts_traces_flags_read": False,
                "credentials_included": False,
            },
            "side_effects": {
                "claims_created": 0,
                "model_calls": 0,
                "task_calls": 0,
                "session_calls": 0,
                "verifier_calls": 0,
                "scoring_calls": 0,
                "api_mutations": 0,
                "scored_jobs_created": 0,
            },
        }
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    root = args.root.resolve(strict=True)
    rendered = yaml.safe_dump(render(root), sort_keys=False)
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.write_text(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
