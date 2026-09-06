"""Render the held conservative hosted-Qwen rank-18 release observer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from evals.fleet import qwen_hosted_generation18_package as job_base
from evals.fleet import qwen_hosted_rank17_g22_release_observer_v1 as observer_base
from evals.fleet import qwen_hosted_rank18_g23_held_v1 as held
from evals.fleet import qwen_hosted_rank18_g23_release_observer_v1 as observer
from evals.fleet import self_hosted

JOB_NAME = "chris-q38-hosted-r018-release-gate-g23-v1"
CONFIGMAP_NAME = JOB_NAME + "-package"
OUTPUT_ROOT = f"/mnt/sfs/jobs/{JOB_NAME}"
SECRET_NAME = "chris-cyber-opencode-evals-v2"
HELD_RELEASE_PATH = Path(
    "docs/evidence/qwen38-study/2026-09-06-qwen38-hosted-rank18-g23-release-held-v1.json"
)
ENGINE_PATH = Path("evals/fleet/exact_pass4_bulk_runtime_v3.py")
ENGINE_SHA256 = "sha256:b80b39412906d3cb5163e55a882d0f5ae936fd03417fae61bc0a497db1f18daa"


def _prove_engine_paths(root: Path) -> None:
    source = (root / ENGINE_PATH).read_text()
    if self_hosted.sha256(source.encode()) != ENGINE_SHA256 or any(
        marker not in source
        for marker in (
            'return execution_id.removeprefix("sha256:") + ".json"',
            'out / "accepted" / f"{item[\'run_id\']}.json"',
            'attempt_out = out / "attempts" / item["run_id"]',
        )
    ):
        raise ValueError("rank18 engine path contract drifted")


def build_binding(root: Path) -> dict[str, Any]:
    _prove_engine_paths(root)
    plan_module = root / "evals/fleet/qwen_hosted_rank18_g23_held_v1.py"
    held_path = root / held.HELD_PATH
    if observer_base.sha256(plan_module.read_bytes()) != observer.EXPECTED_PLAN_MODULE_SHA256:
        raise ValueError("rank18 held-plan module bytes drifted")
    if observer_base.sha256(held_path.read_bytes()) != observer.EXPECTED_HELD_FILE_SHA256:
        raise ValueError("rank18 held receipt file bytes drifted")
    plan = held.build_plan(root)
    held_receipt = json.loads(held_path.read_text())
    held.validate_held(held_receipt, root)
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
        raise ValueError("rank18 task key set drifted")
    body = {
        "schema_version": observer.BINDING_SCHEMA,
        "plan_commit": observer.PLAN_COMMIT,
        "plan_module_sha256": observer.EXPECTED_PLAN_MODULE_SHA256,
        "held_file_sha256": observer.EXPECTED_HELD_FILE_SHA256,
        "held_receipt_sha256": held_receipt["receipt_sha256"],
        "plan_sha256": plan["plan_sha256"],
        "authoritative_tally": held_receipt["required_fresh_release"][
            "authoritative_ledger_counts"
        ],
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
    binding = {**body, "binding_sha256": observer_base.binding_digest(body)}
    observer.validate_binding(binding)
    return binding


def render(root: Path) -> dict[str, Any]:
    binding = build_binding(root)
    names = (
        "qwen_hosted_rank17_g22_release_observer_v1.py",
        "qwen_hosted_rank18_g23_release_observer_v1.py",
        "score_blind_session_inventory_v1.py",
    )
    data = {name: (root / "evals/fleet" / name).read_text() for name in names}
    data["binding.json"] = json.dumps(binding, sort_keys=True, separators=(",", ":")) + "\n"
    package_body = {
        "schema_version": observer.PACKAGE_SCHEMA,
        "files": {
            name: observer_base.sha256(value.encode()) for name, value in sorted(data.items())
        },
        "file_count": len(data),
    }
    data["package-source.json"] = (
        json.dumps(observer_base.seal(package_body), sort_keys=True, separators=(",", ":"))
        + "\n"
    )
    configmap = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": CONFIGMAP_NAME, "namespace": observer_base.NAMESPACE},
        "immutable": True,
        "data": data,
    }
    job = job_base._base_job(JOB_NAME, CONFIGMAP_NAME, scored=False)  # noqa: SLF001
    experiment = "q38-hosted-r018-release-gate-g23-v1"
    job["metadata"]["labels"]["cyber-post-train.fleet.ai/experiment"] = experiment
    job["metadata"]["annotations"].update(
        {
            "cyber-post-train.fleet.ai/create-once": "true",
            "cyber-post-train.fleet.ai/launch-authorized": "false",
            "cyber-post-train.fleet.ai/diagnostic-only": "true",
            "cyber-post-train.fleet.ai/score-free": "true",
            "cyber-post-train.fleet.ai/plan-commit": observer.PLAN_COMMIT,
            "cyber-post-train.fleet.ai/rank17-release-held": "true",
        }
    )
    job["spec"]["activeDeadlineSeconds"] = 900
    job["spec"]["template"]["metadata"]["labels"][
        "cyber-post-train.fleet.ai/experiment"
    ] = experiment
    container = job["spec"]["template"]["spec"]["containers"][0]
    container["args"] = [
        "exec python /bootstrap/qwen_hosted_rank18_g23_release_observer_v1.py "
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


def held_release(root: Path) -> dict[str, Any]:
    binding = build_binding(root)
    rendered = render(root)
    configmap, job = rendered["items"]
    return observer_base.seal(
        {
            "schema_version": "fleet-qwen38-hosted-rank18-g23-release-held-v1",
            "status": "HELD_PENDING_INDEPENDENT_REVIEW",
            "launch_authorized": False,
            "scoring_authorized": False,
            "plan_commit": observer.PLAN_COMMIT,
            "plan_sha256": binding["plan_sha256"],
            "held_receipt_sha256": binding["held_receipt_sha256"],
            "binding_sha256": binding["binding_sha256"],
            "authoritative_tally": observer.EXPECTED_TALLY,
            "rank17_authority_unchanged": True,
            "observer": {
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
            "required_clear_observation": {
                "all_four_rank18_cells_unstarted": True,
                "any_exact_model_session_row_collides": True,
                "missing_or_malformed_model_fails_closed": True,
                "finite_claim_and_accept_paths_absent": 12,
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
                "session_inventory_parser": "bounded_streaming_allowlist_v1",
                "verifier_execution_values_materialized": False,
                "scores_read": False,
                "prompts_traces_flags_read": False,
                "credentials_included": False,
            },
        }
    )


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
