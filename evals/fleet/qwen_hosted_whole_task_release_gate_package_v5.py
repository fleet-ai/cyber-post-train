"""Render the fresh, score-free hosted-Qwen release-gate observer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from evals.fleet import qwen_hosted_generation18_package as base
from evals.fleet import qwen_hosted_whole_task_release_gate_v1 as gate
from evals.fleet import qwen_hosted_whole_task_successor_v2 as prior
from evals.fleet import qwen_hosted_whole_task_successor_v3 as successor
from evals.fleet import self_hosted

JOB_NAME = "chris-q38-hosted-r15-r16-release-gate-v5"
CONFIGMAP_NAME = JOB_NAME + "-package"
OUTPUT_ROOT = f"/mnt/sfs/jobs/{JOB_NAME}"
SECRET_NAME = "chris-cyber-opencode-evals-v2"
SECRET_UID = "e0febd8e-94a2-46b0-a0bf-dd6b3154187b"
PREDECESSOR_OBJECTS = [
    {
        "controller": "qwen-a",
        "job_name": "chris-q38-hosted-r015-whole-task-g20-v2",
        "job_uid": "c5d02921-3271-4186-89b8-65d227a44f2c",
        "pod_uid": "d3e39095-adb0-40d9-91b2-95f9ae27693b",
        "configmap_name": "chris-q38-hosted-r015-whole-task-g20-package-v2",
        "configmap_uid": "a93c42df-01a9-4fbc-a75c-122bc889fd6c",
    },
    {
        "controller": "qwen-b",
        "job_name": "chris-q38-hosted-r016-whole-task-g20-v2",
        "job_uid": "ea058426-820e-4b0c-af9e-499f0350dbd7",
        "pod_uid": "ff722cc1-e36d-4758-b7b6-f8edc8b8f4c6",
        "configmap_name": "chris-q38-hosted-r016-whole-task-g20-package-v2",
        "configmap_uid": "ad31b0da-7ac5-481f-a73b-1bd4b4a063d7",
    },
]


def build_binding(root: Path) -> dict[str, Any]:
    plans = successor.build_plans(root)
    old = prior.build_plans(root)
    new_attempts = [row for plan in plans.values() for row in plan["attempts"]]
    old_attempts = [row for plan in old.values() for row in plan["attempts"]]
    identities = sorted(
        {
            value
            for row in (*new_attempts, *old_attempts)
            for value in (row["cell_id"], row["execution_id"], row["run_id"])
        }
    )
    fresh_objects = [
        {
            "controller": controller,
            "job_name": authority["job_name"],
            "configmap_name": authority["configmap_name"],
        }
        for controller, authority in sorted(successor.CONTROLLERS.items())
    ]
    body = {
        "schema_version": "fleet-qwen38-hosted-whole-task-release-gate-binding-v1",
        "statistical_cell_count": 8,
        "identity_values": identities,
        "task_keys": sorted({row["task_key"] for row in new_attempts}),
        "predecessor_objects": PREDECESSOR_OBJECTS,
        "fresh_objects": fresh_objects,
        "checked_sfs_roots": sorted(
            {plan["sfs_root"] for plan in plans.values()}
            | {plan["sfs_root"] for plan in old.values()}
            | {plan["sfs_root"] + "-diagnostic" for plan in plans.values()}
        ),
        "claim_root": str(successor.CLAIM_ROOT),
        "jobs_root": str(successor.JOBS_ROOT),
        "lease_root": str(successor.ENDPOINT_LEASE_ROOT),
        "endpoint_key": successor.ENDPOINT_KEY,
        "predecessor_object_set_sha256": gate.sha256(gate.canonical(PREDECESSOR_OBJECTS)),
        "fresh_object_set_sha256": gate.sha256(gate.canonical(fresh_objects)),
        "plan_set_sha256": gate.sha256(
            gate.canonical(
                {
                    controller: {
                        "plan_sha256": plan["plan_sha256"],
                        "statistical_cells_sha256": gate.sha256(
                            gate.canonical(
                                [
                                    {
                                        "cell_id": row["cell_id"],
                                        "selection_rank": row["selection_rank"],
                                        "attempt": row["attempt"],
                                    }
                                    for row in plan["attempts"]
                                ]
                            )
                        ),
                    }
                    for controller, plan in sorted(plans.items())
                }
            )
        ),
    }
    binding = {**body, "binding_sha256": gate.binding_digest(body)}
    gate.validate_binding(binding)
    return binding


def render(root: Path) -> dict[str, Any]:
    binding = build_binding(root)
    data = {
        "qwen_hosted_whole_task_release_gate_v1.py": (
            root / "evals/fleet/qwen_hosted_whole_task_release_gate_v1.py"
        ).read_text(),
        "binding.json": json.dumps(binding, sort_keys=True, separators=(",", ":")) + "\n",
    }
    package_body = {
        "schema_version": "fleet-qwen38-hosted-whole-task-release-gate-package-v5",
        "files": {name: self_hosted.sha256(value.encode()) for name, value in sorted(data.items())},
        "file_count": len(data),
    }
    data["package-source.json"] = (
        json.dumps(gate._seal(package_body), sort_keys=True, separators=(",", ":")) + "\n"
    )
    configmap = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": CONFIGMAP_NAME, "namespace": gate.NAMESPACE},
        "immutable": True,
        "data": data,
    }
    job = base._base_job(JOB_NAME, CONFIGMAP_NAME, scored=False)  # noqa: SLF001
    job["metadata"]["labels"]["cyber-post-train.fleet.ai/experiment"] = (
        "q38-hosted-whole-task-release-gate-v5"
    )
    job["metadata"]["annotations"].update(
        {
            "cyber-post-train.fleet.ai/create-once": "true",
            "cyber-post-train.fleet.ai/launch-authorized": "false",
            "cyber-post-train.fleet.ai/diagnostic-only": "true",
            "cyber-post-train.fleet.ai/score-free": "true",
        }
    )
    job["spec"]["activeDeadlineSeconds"] = 900
    job["spec"]["template"]["metadata"]["labels"]["cyber-post-train.fleet.ai/experiment"] = (
        "q38-hosted-whole-task-release-gate-v5"
    )
    container = job["spec"]["template"]["spec"]["containers"][0]
    container["args"] = [
        "exec python /bootstrap/qwen_hosted_whole_task_release_gate_v1.py "
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
        raise ValueError("release-gate observer ConfigMap exceeds safety budget")
    return {"apiVersion": "v1", "kind": "List", "items": [configmap, job]}


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
