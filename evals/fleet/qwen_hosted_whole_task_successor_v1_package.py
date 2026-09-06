"""Render held create-once rank-15/rank-16 hosted successor Jobs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evals.fleet import qwen_hosted_generation18_package as base
from evals.fleet import qwen_hosted_generation19_v4_package as predecessor_package
from evals.fleet import qwen_hosted_whole_task_successor_v1 as successor

COMMON = (
    *predecessor_package.COMMON,
    "evals/fleet/qwen_hosted_generation19_v4_runtime.py",
    "evals/fleet/qwen_hosted_whole_task_successor_v1.py",
    "evals/fleet/qwen_hosted_whole_task_successor_v1_runtime.py",
)


def _read(root: Path, paths: tuple[str, ...]) -> dict[str, str]:
    return {Path(path).name: (root / path).read_text() for path in dict.fromkeys(paths)}


def _source_data(root: Path, plan: dict[str, Any]) -> dict[str, str]:
    paths = (
        *COMMON,
        "evals/fleet/opencode_train_sweep_runner.py",
        "evals/fleet/fixed_proxy.py",
        "evals/fleet/Dockerfile.opencode",
        "evals/fleet/scripts/run_qwen_hosted_whole_task_successor_v1.sh",
    )
    data = _read(root, paths)
    data["source-plan-v4-a.json"] = (root / successor.source.PLAN_PATHS["qwen-a"]).read_text()
    data["source-plan-v4-b.json"] = (root / successor.source.PLAN_PATHS["qwen-b"]).read_text()
    data["plan.json"] = json.dumps(plan, sort_keys=True, separators=(",", ":")) + "\n"
    return data


def package_sources(
    root: Path, plans: dict[str, dict[str, Any]]
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, str]]]:
    data = {controller: _source_data(root, plan) for controller, plan in plans.items()}
    receipts = {
        controller: successor.package_source_receipt(controller, plans[controller], values)
        for controller, values in data.items()
    }
    return receipts, data


def render(root: Path, *, release_path: Path | None = None) -> dict[str, Any]:
    plans = successor.build_plans(root)
    sources, source_data = package_sources(root, plans)
    held = successor.load(root / successor.HELD_PATH)
    successor.validate_held(held, plans, sources)
    release: dict[str, Any] | None = None
    if release_path is not None:
        release = successor.load(release_path)
        successor.validate_release(release, plans, sources)
    items: list[dict[str, Any]] = []
    for controller, plan in plans.items():
        authority = successor.CONTROLLERS[controller]
        data = source_data[controller]
        bound = release or held
        data["release.json"] = json.dumps(bound, sort_keys=True, separators=(",", ":")) + "\n"
        data["package-source.json"] = (
            json.dumps(sources[controller], sort_keys=True, separators=(",", ":")) + "\n"
        )
        cm = {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": authority["configmap_name"],
                "namespace": base.NAMESPACE,
            },
            "immutable": True,
            "data": data,
        }
        job = base._base_job(authority["job_name"], authority["configmap_name"], scored=True)  # noqa: SLF001
        launched = release is not None
        job["metadata"]["annotations"]["cyber-post-train.fleet.ai/launch-authorized"] = str(
            launched
        ).lower()
        job["metadata"]["labels"]["cyber-post-train.fleet.ai/experiment"] = (
            "q38-hosted-atomic-whole-task-successor"
        )
        job["spec"]["activeDeadlineSeconds"] = 120000
        job["spec"]["template"]["metadata"]["labels"]["cyber-post-train.fleet.ai/experiment"] = (
            "q38-hosted-atomic-whole-task-successor"
        )
        container = job["spec"]["template"]["spec"]["containers"][0]
        container["args"] = [
            "apt-get update; apt-get install --yes --no-install-recommends "
            "docker.io=20.10.24+dfsg1-1+deb12u1+b6; "
            "exec /bin/bash /bootstrap/run_qwen_hosted_whole_task_successor_v1.sh"
        ]
        container["env"] = [
            row
            for row in container["env"]
            if row["name"] not in {"G18_PREFLIGHT_PATH", "G18_PREFLIGHT_SHA256"}
        ]
        container["env"].extend(
            [
                {"name": "QWEN_HOSTED_WHOLE_TASK_CONTROLLER", "value": controller},
                {"name": "QWEN_HOSTED_WHOLE_TASK_OUTPUT_ROOT", "value": plan["sfs_root"]},
                {
                    "name": "QWEN_HOSTED_WHOLE_TASK_DIAGNOSTIC_ROOT",
                    "value": plan["sfs_root"] + "-diagnostic",
                },
                {
                    "name": "QWEN_HOSTED_WHOLE_TASK_RELEASE_PATH",
                    "value": "/bootstrap/release.json",
                },
                {
                    "name": "QWEN_HOSTED_WHOLE_TASK_RELEASE_SHA256",
                    "value": bound["receipt_sha256"],
                },
                {
                    "name": "QWEN_HOSTED_WHOLE_TASK_PACKAGE_SOURCE_SHA256",
                    "value": sources[controller]["receipt_sha256"],
                },
            ]
        )
        if len(json.dumps(cm).encode()) >= 900_000:
            raise ValueError("hosted whole-task ConfigMap exceeds safety budget")
        items.extend([cm, job])
    return {"apiVersion": "v1", "kind": "List", "items": items}
