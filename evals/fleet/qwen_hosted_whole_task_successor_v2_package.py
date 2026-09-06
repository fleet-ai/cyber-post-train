"""Render held fresh-generation rank-15/rank-16 hosted-Qwen Jobs."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from evals.fleet import qwen_hosted_generation18_package as base
from evals.fleet import qwen_hosted_whole_task_successor_v1_package as prior_package
from evals.fleet import qwen_hosted_whole_task_successor_v2 as successor

COMMON = (
    *(
        path
        for path in prior_package.COMMON
        if path
        not in {
            "evals/fleet/qwen_hosted_generation19_v2_runtime.py",
            "evals/fleet/qwen_hosted_generation19_v4_runtime.py",
            "evals/fleet/qwen_hosted_whole_task_successor_v1_runtime.py",
        }
    ),
    "evals/fleet/qwen_hosted_whole_task_successor_v2.py",
    "evals/fleet/qwen_hosted_whole_task_release_gate_contract_v1.py",
    "evals/fleet/qwen_hosted_whole_task_successor_v2_runtime.py",
)
RUNTIME_GATE_CANARY_JOB = "chris-q38-hosted-whole-task-runtime-gate-canary-v6"
RUNTIME_GATE_CANARY_CONFIGMAP = RUNTIME_GATE_CANARY_JOB + "-package"
RUNTIME_GATE_CANARY_DIAGNOSTIC_ROOT = f"/mnt/sfs/jobs/{RUNTIME_GATE_CANARY_JOB}-diagnostic"
RUNTIME_GATE_CANARY_PRIVATE_ROOT = "/workspace/q38-hosted-runtime-gate-v6-private"


def _configure_v2_runtime_bootstrap(
    job: dict[str, Any], *, diagnostic_root: str, private_root: str
) -> None:
    prior_package._configure_runtime_bootstrap(  # noqa: SLF001
        job, diagnostic_root=diagnostic_root, private_root=private_root
    )
    container = job["spec"]["template"]["spec"]["containers"][0]
    container["args"] = [
        container["args"][0].replace(
            "run_qwen_hosted_whole_task_successor_v1.sh",
            "run_qwen_hosted_whole_task_successor_v2.sh",
        )
    ]


def _replace_operational_string(value: Any, replacements: dict[str, str]) -> Any:
    if isinstance(value, str):
        for old, new in replacements.items():
            value = value.replace(old, new)
        return value
    if isinstance(value, list):
        return [_replace_operational_string(item, replacements) for item in value]
    if isinstance(value, dict):
        return {
            key: _replace_operational_string(item, replacements)
            for key, item in value.items()
        }
    return value


def _source_data(root: Path, plan: dict[str, Any]) -> dict[str, str]:
    paths = (
        *COMMON,
        "evals/fleet/opencode_train_sweep_runner.py",
        "evals/fleet/fixed_proxy.py",
        "evals/fleet/Dockerfile.opencode",
        "evals/fleet/scripts/run_qwen_hosted_whole_task_successor_v2.sh",
    )
    data = prior_package._read(root, paths)  # noqa: SLF001
    data["source-plan-v4-a.json"] = (root / successor.prior.source.PLAN_PATHS["qwen-a"]).read_text()
    data["source-plan-v4-b.json"] = (root / successor.prior.source.PLAN_PATHS["qwen-b"]).read_text()
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
        bound = release or held
        data = source_data[controller]
        data["release.json"] = json.dumps(bound, sort_keys=True, separators=(",", ":")) + "\n"
        data["package-source.json"] = (
            json.dumps(sources[controller], sort_keys=True, separators=(",", ":")) + "\n"
        )
        configmap = {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {"name": authority["configmap_name"], "namespace": base.NAMESPACE},
            "immutable": True,
            "data": data,
        }
        job = base._base_job(authority["job_name"], authority["configmap_name"], scored=True)  # noqa: SLF001
        launched = release is not None
        job["metadata"]["annotations"]["cyber-post-train.fleet.ai/launch-authorized"] = str(
            launched
        ).lower()
        job["metadata"]["labels"]["cyber-post-train.fleet.ai/experiment"] = (
            "q38-hosted-atomic-whole-task-successor-v2"
        )
        job["spec"]["activeDeadlineSeconds"] = 120000
        job["spec"]["template"]["metadata"]["labels"]["cyber-post-train.fleet.ai/experiment"] = (
            "q38-hosted-atomic-whole-task-successor-v2"
        )
        container = job["spec"]["template"]["spec"]["containers"][0]
        container["env"] = [
            row
            for row in container["env"]
            if row["name"] not in {"G18_PREFLIGHT_PATH", "G18_PREFLIGHT_SHA256"}
        ]
        private_root = f"/workspace/{authority['job_name']}-private"
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
                    "value": private_root + "/release.json",
                },
                {
                    "name": "QWEN_HOSTED_WHOLE_TASK_PACKAGE_SOURCE_PATH",
                    "value": private_root + "/package-source.json",
                },
                {"name": "QWEN_HOSTED_WHOLE_TASK_RELEASE_SHA256", "value": bound["receipt_sha256"]},
                {
                    "name": "QWEN_HOSTED_WHOLE_TASK_PACKAGE_SOURCE_SHA256",
                    "value": sources[controller]["receipt_sha256"],
                },
            ]
        )
        _configure_v2_runtime_bootstrap(
            job,
            diagnostic_root=plan["sfs_root"] + "-diagnostic",
            private_root=private_root,
        )
        if len(json.dumps(configmap).encode()) >= 900_000:
            raise ValueError("fresh hosted whole-task ConfigMap exceeds safety budget")
        items.extend([configmap, job])
    return {"apiVersion": "v1", "kind": "List", "items": items}


def render_runtime_gate_canary(root: Path) -> dict[str, Any]:
    """Render the exact v2 package/runtime and stop before every scored boundary."""
    rendered = render(root)
    source_cm, source_job = rendered["items"][:2]
    cm = copy.deepcopy(source_cm)
    cm["metadata"]["name"] = RUNTIME_GATE_CANARY_CONFIGMAP
    job = copy.deepcopy(source_job)
    job["metadata"]["name"] = RUNTIME_GATE_CANARY_JOB
    job["metadata"]["annotations"].update(
        {
            "cyber-post-train.fleet.ai/create-once": "true",
            "cyber-post-train.fleet.ai/launch-authorized": "false",
            "cyber-post-train.fleet.ai/diagnostic-only": "true",
            "cyber-post-train.fleet.ai/score-free": "true",
        }
    )
    job["metadata"]["labels"]["cyber-post-train.fleet.ai/experiment"] = (
        "q38-hosted-runtime-gate-canary-v2"
    )
    source_plan = successor.build_plans(root)["qwen-a"]
    production_private_root = (
        f"/workspace/{successor.CONTROLLERS['qwen-a']['job_name']}-private"
    )
    job = _replace_operational_string(
        job,
        {
            source_plan["sfs_root"] + "-diagnostic": RUNTIME_GATE_CANARY_DIAGNOSTIC_ROOT,
            production_private_root: RUNTIME_GATE_CANARY_PRIVATE_ROOT,
            successor.CONTROLLERS["qwen-a"]["configmap_name"]: RUNTIME_GATE_CANARY_CONFIGMAP,
        },
    )
    pod = job["spec"]["template"]
    pod["metadata"]["labels"]["cyber-post-train.fleet.ai/experiment"] = (
        "q38-hosted-runtime-gate-canary-v2"
    )
    pod["spec"]["containers"][0]["env"] = [
        row
        for row in pod["spec"]["containers"][0]["env"]
        if row["name"] != "FLEET_API_KEY"
    ]
    container = pod["spec"]["containers"][0]
    replacements = {
        "QWEN_HOSTED_WHOLE_TASK_OUTPUT_ROOT": "/dev/null",
        "QWEN_HOSTED_WHOLE_TASK_DIAGNOSTIC_ROOT": RUNTIME_GATE_CANARY_DIAGNOSTIC_ROOT,
        "QWEN_HOSTED_WHOLE_TASK_RELEASE_PATH": RUNTIME_GATE_CANARY_PRIVATE_ROOT + "/release.json",
        "QWEN_HOSTED_WHOLE_TASK_PACKAGE_SOURCE_PATH": (
            RUNTIME_GATE_CANARY_PRIVATE_ROOT + "/package-source.json"
        ),
    }
    for row in container["env"]:
        if row["name"] in replacements:
            row["value"] = replacements[row["name"]]
    container["env"].append(
        {"name": "QWEN_HOSTED_WHOLE_TASK_RUNTIME_GATE_CANARY", "value": "true"}
    )
    job["spec"]["activeDeadlineSeconds"] = 1800
    return {"apiVersion": "v1", "kind": "List", "items": [cm, job]}
