"""Render the held, score-blind hosted-Qwen rank-17 release observer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from evals.fleet import qwen_hosted_generation18_package as base
from evals.fleet import qwen_hosted_rank17_g22_held_v1 as held
from evals.fleet import qwen_hosted_rank17_g22_release_observer_v1 as observer
from evals.fleet import self_hosted

JOB_NAME = "chris-q38-hosted-r017-release-gate-g22-v1"
CONFIGMAP_NAME = JOB_NAME + "-package"
OUTPUT_ROOT = f"/mnt/sfs/jobs/{JOB_NAME}"
SECRET_NAME = "chris-cyber-opencode-evals-v2"
SECRET_UID = "e0febd8e-94a2-46b0-a0bf-dd6b3154187b"
PLAN_MODULE_SHA256 = (
    "sha256:406eec486d25cda5e201eb907e24ce4c4671f66a48502beea0503191d9e7571b"
)
HELD_FILE_SHA256 = (
    "sha256:71a99f7a90a16c6e69eff13ad9d66d6287020e4776658a8250120c7c38a2364a"
)
HELD_RELEASE_SCHEMA = "fleet-qwen38-hosted-rank17-g22-observer-held-v1"
HELD_RELEASE_PATH = Path(
    "docs/evidence/qwen38-study/2026-09-06-qwen38-hosted-rank17-g22-observer-held-v1.json"
)


def build_binding(root: Path) -> dict[str, Any]:
    plan_module_path = root / "evals/fleet/qwen_hosted_rank17_g22_held_v1.py"
    held_path = root / held.HELD_PATH
    if observer.sha256(plan_module_path.read_bytes()) != PLAN_MODULE_SHA256:
        raise ValueError("rank17 held-plan module bytes drifted")
    if observer.sha256(held_path.read_bytes()) != HELD_FILE_SHA256:
        raise ValueError("rank17 held receipt file bytes drifted")
    plan = held.build_plan(root)
    held_receipt = json.loads(held_path.read_text())
    held.validate_held(held_receipt, root)
    tally = held_receipt["required_fresh_release"]["authoritative_ledger_counts"]
    if tally != observer.EXPECTED_TALLY:
        raise ValueError("rank17 authoritative tally drifted")
    cells = [
        {
            "attempt": row["attempt"],
            "cell_id": row["cell_id"],
            "execution_id": row["execution_id"],
            "run_id": row["run_id"],
        }
        for row in plan["attempts"]
    ]
    task_keys = {row["task_key"] for row in plan["attempts"]}
    if len(task_keys) != 1:
        raise ValueError("rank17 task key set drifted")
    body = {
        "schema_version": observer.BINDING_SCHEMA,
        "plan_commit": observer.PLAN_COMMIT,
        "plan_module_sha256": PLAN_MODULE_SHA256,
        "held_file_sha256": HELD_FILE_SHA256,
        "held_receipt_sha256": held_receipt["receipt_sha256"],
        "plan_sha256": plan["plan_sha256"],
        "authoritative_tally": tally,
        "statistical_cell_count": 4,
        "cells": cells,
        "identity_values": sorted(
            value
            for row in cells
            for value in (row["cell_id"], row["execution_id"], row["run_id"])
        ),
        "task_key": task_keys.pop(),
        "task_version_id": plan["tasks"][0]["task"]["version_id"],
        "session_model": plan["model"]["session_model"],
        "fresh_object": {
            "job_name": held.JOB_NAME,
            "configmap_name": held.CONFIGMAP_NAME,
        },
        "checked_sfs_roots": sorted({held.SFS_ROOT, held.DIAGNOSTIC_ROOT}),
        "claim_root": plan["execution"]["claim_root"],
        "planned_claim_paths": [
            f"{plan['execution']['claim_root']}/{row['execution_id'].removeprefix('sha256:')}.json"
            for row in cells
        ],
        "jobs_root": "/mnt/sfs/jobs",
        "planned_accepted_paths": [
            f"/mnt/sfs/jobs/{row['run_id']}/ACCEPTED.json" for row in cells
        ],
        "lease_root": plan["execution"]["endpoint_lease"]["lease_root"],
        "endpoint_key": plan["execution"]["endpoint_lease"]["endpoint_key"],
    }
    binding = {**body, "binding_sha256": observer.binding_digest(body)}
    observer.validate_binding(binding)
    return binding


def render(root: Path) -> dict[str, Any]:
    binding = build_binding(root)
    source_name = "qwen_hosted_rank17_g22_release_observer_v1.py"
    data = {
        source_name: (root / f"evals/fleet/{source_name}").read_text(),
        "binding.json": json.dumps(binding, sort_keys=True, separators=(",", ":")) + "\n",
    }
    package_body = {
        "schema_version": observer.PACKAGE_SCHEMA,
        "files": {
            name: self_hosted.sha256(value.encode()) for name, value in sorted(data.items())
        },
        "file_count": len(data),
    }
    data["package-source.json"] = (
        json.dumps(observer.seal(package_body), sort_keys=True, separators=(",", ":")) + "\n"
    )
    configmap = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": CONFIGMAP_NAME, "namespace": observer.NAMESPACE},
        "immutable": True,
        "data": data,
    }
    job = base._base_job(JOB_NAME, CONFIGMAP_NAME, scored=False)  # noqa: SLF001
    experiment = "q38-hosted-r017-release-gate-g22-v1"
    job["metadata"]["labels"]["cyber-post-train.fleet.ai/experiment"] = experiment
    job["metadata"]["annotations"].update(
        {
            "cyber-post-train.fleet.ai/create-once": "true",
            "cyber-post-train.fleet.ai/launch-authorized": "false",
            "cyber-post-train.fleet.ai/diagnostic-only": "true",
            "cyber-post-train.fleet.ai/score-free": "true",
            "cyber-post-train.fleet.ai/plan-commit": observer.PLAN_COMMIT,
        }
    )
    job["spec"]["activeDeadlineSeconds"] = 900
    job["spec"]["template"]["metadata"]["labels"][
        "cyber-post-train.fleet.ai/experiment"
    ] = experiment
    container = job["spec"]["template"]["spec"]["containers"][0]
    container["args"] = [
        f"exec python /bootstrap/{source_name} "
        "--binding /bootstrap/binding.json "
        "--package-source /bootstrap/package-source.json "
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
    if len(json.dumps(configmap).encode()) >= 900_000:
        raise ValueError("rank17 release observer ConfigMap exceeds safety budget")
    return {"apiVersion": "v1", "kind": "List", "items": [configmap, job]}


def held_release(root: Path) -> dict[str, Any]:
    binding = build_binding(root)
    rendered = render(root)
    configmap, job = rendered["items"]
    return observer.seal(
        {
            "schema_version": HELD_RELEASE_SCHEMA,
            "status": "HELD_PENDING_SCORE_BLIND_OBSERVER",
            "launch_authorized": False,
            "scoring_authorized": False,
            "plan_commit": observer.PLAN_COMMIT,
            "plan_sha256": binding["plan_sha256"],
            "held_receipt_sha256": binding["held_receipt_sha256"],
            "binding_sha256": binding["binding_sha256"],
            "authoritative_tally": observer.EXPECTED_TALLY,
            "all_four_cells_required_unstarted": True,
            "observer": {
                "job_name": JOB_NAME,
                "configmap_name": CONFIGMAP_NAME,
                "output_root": OUTPUT_ROOT,
                "create_once": True,
                "score_free": True,
                "configmap_immutable": configmap["immutable"],
                "manifest_sha256": observer.sha256(observer.canonical(rendered)),
                "preemption_policy": job["spec"]["template"]["spec"]["preemptionPolicy"],
                "priority_class": job["spec"]["template"]["spec"]["priorityClassName"],
                "secret_name": SECRET_NAME,
                "secret_uid": SECRET_UID,
            },
            "required_clear_observation": {
                "canonical_claim_collisions": 0,
                "accepted_receipt_collisions": 0,
                "authoritative_session_collisions": 0,
                "fresh_job_configmap_pod_collisions": 0,
                "fresh_sfs_output_collisions": 0,
                "endpoint_lease_slots_simultaneously_free": 2,
                "observer_job_succeeded": True,
                "observer_restarts": 0,
                "independent_terminal_review_required": True,
            },
            "side_effects": {
                "claims_created": 0,
                "model_calls": 0,
                "task_calls": 0,
                "session_mutations": 0,
                "verifier_calls": 0,
                "scoring_calls": 0,
                "api_mutations": 0,
                "scored_kubernetes_jobs_created": 0,
            },
            "privacy": {
                "scores_read": False,
                "prompts_traces_flags_read": False,
                "credentials_included": False,
            },
        }
    )


def validate_held_release(value: dict[str, Any], root: Path) -> None:
    if value != held_release(root):
        raise ValueError("rank17 held observer release drifted")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    rendered = yaml.safe_dump(render(args.root.resolve(strict=True)), sort_keys=False)
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.write_text(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
