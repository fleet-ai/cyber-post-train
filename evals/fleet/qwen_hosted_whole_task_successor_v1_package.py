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
    "evals/fleet/qwen_hosted_prebootstrap_v4.py",
    "evals/fleet/qwen_hosted_private_stage_v5.py",
)
RUNTIME_GATE_CANARY_JOB = "chris-q38-hosted-whole-task-runtime-gate-canary-v5"
RUNTIME_GATE_CANARY_CONFIGMAP = RUNTIME_GATE_CANARY_JOB + "-package"
RUNTIME_GATE_CANARY_DIAGNOSTIC_ROOT = (
    "/mnt/sfs/jobs/chris-q38-hosted-whole-task-runtime-gate-canary-v5-diagnostic"
)
RUNTIME_GATE_CANARY_PRIVATE_ROOT = "/workspace/q38-hosted-runtime-gate-v5-private"
DOCKER_CLI_SHA256 = "242c7a8de606afba2acada7c7af00d77f92c3601678b2f3a60911b49a892c722"
DOCKER_BUILDX_SHA256 = "8c38f60308a895fa570f1410e453c5de11aafd65a99fa99965d96d24b6225a78"
DOCKER_CLI_BYTES = 105_594_160


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


def _configure_runtime_bootstrap(
    job: dict[str, Any], *, diagnostic_root: str, private_root: str
) -> None:
    """Install the one pinned, evidenced bootstrap shared by canary and scored Jobs."""
    pod = job["spec"]["template"]["spec"]
    container = pod["containers"][0]
    container["env"].extend(
        [
            {
                "name": "PATH",
                "value": (
                    "/docker-cli/bin:/usr/local/sbin:/usr/local/bin:"
                    "/usr/sbin:/usr/bin:/sbin:/bin"
                ),
            },
            {"name": "DOCKER_CONFIG", "value": "/workspace/docker-config"},
        ]
    )
    prebootstrap_env = [
        row
        for row in container["env"]
        if row["name"]
        in {
            "JOB_UID",
            "POD_UID",
            "QWEN_HOSTED_WHOLE_TASK_DIAGNOSTIC_ROOT",
            "QWEN_HOSTED_WHOLE_TASK_RELEASE_SHA256",
            "QWEN_HOSTED_WHOLE_TASK_PACKAGE_SOURCE_SHA256",
        }
    ]
    dind = pod["initContainers"][0]
    pod["initContainers"] = [
        {
            "name": "prebootstrap-evidence",
            "image": base.UV_IMAGE,
            "command": ["python", "/bootstrap/qwen_hosted_prebootstrap_v4.py"],
            "args": [
                "record",
                "--phase",
                "00-prebootstrap-entry",
                "--root",
                diagnostic_root,
            ],
            "env": prebootstrap_env,
            "resources": {
                "requests": {"cpu": "10m", "memory": "32Mi", "ephemeral-storage": "32Mi"},
                "limits": {"cpu": "100m", "memory": "128Mi", "ephemeral-storage": "128Mi"},
            },
            "volumeMounts": [
                {"name": "bootstrap", "mountPath": "/bootstrap", "readOnly": True},
                {"name": "sfs", "mountPath": "/mnt/sfs"},
            ],
        },
        {
            "name": "docker-cli",
            "image": base.DIND_IMAGE,
            "command": ["/bin/sh", "-ec"],
            "args": [
                "install -D -m 0755 /usr/local/bin/docker /cli/bin/docker; "
                "install -D -m 0755 /usr/local/libexec/docker/cli-plugins/docker-buildx "
                "/cli/plugins/docker-buildx; "
                f"test \"$(sha256sum /cli/bin/docker | awk '{{print $1}}')\" = "
                f"{DOCKER_CLI_SHA256}; "
                f"test \"$(sha256sum /cli/plugins/docker-buildx | awk '{{print $1}}')\" = "
                f"{DOCKER_BUILDX_SHA256}; "
                "test \"$(( $(wc -c < /cli/bin/docker) + "
                "$(wc -c < /cli/plugins/docker-buildx) ))\" = "
                f"{DOCKER_CLI_BYTES}"
            ],
            "resources": {
                "requests": {"cpu": "10m", "memory": "32Mi", "ephemeral-storage": "32Mi"},
                "limits": {"cpu": "100m", "memory": "128Mi", "ephemeral-storage": "128Mi"},
            },
            "volumeMounts": [{"name": "docker-cli", "mountPath": "/cli"}],
        },
        dind,
    ]
    pod["volumes"].append({"name": "docker-cli", "emptyDir": {"sizeLimit": "256Mi"}})
    container["volumeMounts"].append(
        {"name": "docker-cli", "mountPath": "/docker-cli", "readOnly": True}
    )
    container["args"] = [
        "python /bootstrap/qwen_hosted_prebootstrap_v4.py record "
        "--phase 01-network-package-install-bypassed "
        f"--root {diagnostic_root}; "
        "test \"$(command -v docker)\" = /docker-cli/bin/docker; "
        f"test \"$(sha256sum /docker-cli/bin/docker | awk '{{print $1}}')\" = "
        f"{DOCKER_CLI_SHA256}; "
        f"test \"$(sha256sum /docker-cli/plugins/docker-buildx | awk '{{print $1}}')\" = "
        f"{DOCKER_BUILDX_SHA256}; "
        "mkdir -p /workspace/docker-config/cli-plugins; "
        "install -m 0755 /docker-cli/plugins/docker-buildx "
        "/workspace/docker-config/cli-plugins/docker-buildx; "
        "python /bootstrap/qwen_hosted_prebootstrap_v4.py record "
        "--phase 02-pinned-docker-cli-ready "
        f"--root {diagnostic_root}; "
        "python /bootstrap/qwen_hosted_prebootstrap_v4.py record "
        "--phase 03-private-input-stage-started "
        f"--root {diagnostic_root}; "
        "python /bootstrap/qwen_hosted_private_stage_v5.py stage-all "
        "--bootstrap-root /bootstrap "
        f"--destination-root {private_root}; "
        "python /bootstrap/qwen_hosted_prebootstrap_v4.py record "
        "--phase 04-private-input-stage-done "
        f"--root {diagnostic_root}; "
        "exec /bin/bash /bootstrap/run_qwen_hosted_whole_task_successor_v1.sh"
    ]


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
                    "value": f"/workspace/{authority['job_name']}-private/release.json",
                },
                {
                    "name": "QWEN_HOSTED_WHOLE_TASK_PACKAGE_SOURCE_PATH",
                    "value": f"/workspace/{authority['job_name']}-private/package-source.json",
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
        private_root = f"/workspace/{authority['job_name']}-private"
        _configure_runtime_bootstrap(
            job,
            diagnostic_root=plan["sfs_root"] + "-diagnostic",
            private_root=private_root,
        )
        if len(json.dumps(cm).encode()) >= 900_000:
            raise ValueError("hosted whole-task ConfigMap exceeds safety budget")
        items.extend([cm, job])
    return {"apiVersion": "v1", "kind": "List", "items": items}


def render_runtime_gate_canary(root: Path) -> dict[str, Any]:
    """Render one score-free create-once Job that stops after the runtime gate."""
    plans = successor.build_plans(root)
    sources, source_data = package_sources(root, plans)
    held = successor.load(root / successor.HELD_PATH)
    successor.validate_held(held, plans, sources)
    controller = "qwen-a"
    data = source_data[controller]
    data["release.json"] = json.dumps(held, sort_keys=True, separators=(",", ":")) + "\n"
    data["package-source.json"] = (
        json.dumps(sources[controller], sort_keys=True, separators=(",", ":")) + "\n"
    )
    cm = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": RUNTIME_GATE_CANARY_CONFIGMAP, "namespace": base.NAMESPACE},
        "immutable": True,
        "data": data,
    }
    job = base._base_job(  # noqa: SLF001
        RUNTIME_GATE_CANARY_JOB, RUNTIME_GATE_CANARY_CONFIGMAP, scored=True
    )
    job["metadata"]["annotations"]["cyber-post-train.fleet.ai/launch-authorized"] = "false"
    job["metadata"]["annotations"]["cyber-post-train.fleet.ai/diagnostic-only"] = "true"
    job["metadata"]["labels"]["cyber-post-train.fleet.ai/experiment"] = (
        "q38-hosted-runtime-gate-canary"
    )
    job["spec"]["template"]["metadata"]["labels"]["cyber-post-train.fleet.ai/experiment"] = (
        "q38-hosted-runtime-gate-canary"
    )
    job["spec"]["activeDeadlineSeconds"] = 1800
    container = job["spec"]["template"]["spec"]["containers"][0]
    container["env"] = [
        row
        for row in container["env"]
        if row["name"] not in {"FLEET_API_KEY", "G18_PREFLIGHT_PATH", "G18_PREFLIGHT_SHA256"}
    ]
    container["env"].extend(
        [
            {"name": "QWEN_HOSTED_WHOLE_TASK_CONTROLLER", "value": controller},
            {"name": "QWEN_HOSTED_WHOLE_TASK_OUTPUT_ROOT", "value": "/dev/null"},
            {
                "name": "QWEN_HOSTED_WHOLE_TASK_DIAGNOSTIC_ROOT",
                "value": RUNTIME_GATE_CANARY_DIAGNOSTIC_ROOT,
            },
            {
                "name": "QWEN_HOSTED_WHOLE_TASK_RELEASE_PATH",
                "value": RUNTIME_GATE_CANARY_PRIVATE_ROOT + "/release.json",
            },
            {
                "name": "QWEN_HOSTED_WHOLE_TASK_PACKAGE_SOURCE_PATH",
                "value": RUNTIME_GATE_CANARY_PRIVATE_ROOT + "/package-source.json",
            },
            {
                "name": "QWEN_HOSTED_WHOLE_TASK_RELEASE_SHA256",
                "value": held["receipt_sha256"],
            },
            {
                "name": "QWEN_HOSTED_WHOLE_TASK_PACKAGE_SOURCE_SHA256",
                "value": sources[controller]["receipt_sha256"],
            },
            {"name": "QWEN_HOSTED_WHOLE_TASK_RUNTIME_GATE_CANARY", "value": "true"},
        ]
    )
    _configure_runtime_bootstrap(
        job,
        diagnostic_root=RUNTIME_GATE_CANARY_DIAGNOSTIC_ROOT,
        private_root=RUNTIME_GATE_CANARY_PRIVATE_ROOT,
    )
    if len(json.dumps(cm).encode()) >= 900_000:
        raise ValueError("hosted whole-task runtime-gate canary ConfigMap exceeds safety budget")
    return {"apiVersion": "v1", "kind": "List", "items": [cm, job]}
