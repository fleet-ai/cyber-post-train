"""Render two instrumented, create-once Qwen G19 v2 successor controllers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evals.fleet import qwen_hosted_generation18_package as base
from evals.fleet import qwen_hosted_generation19_package as prior_package
from evals.fleet import qwen_hosted_generation19_v2 as g19

PREFLIGHT_ROOT = "/mnt/sfs/jobs/chris-q38-ac-exact100-g19-preflight-v4"
PREFLIGHT_SHA256 = "sha256:5e2ea6d4d1417f0db242984f85b3466736dcb29b13272b5187fa340ae90c1ac7"
COMMON = (
    *prior_package.COMMON,
    "evals/fleet/qwen_hosted_generation19_v2.py",
    "evals/fleet/qwen_hosted_generation19_v2_runtime.py",
    "evals/fleet/exact_pass4_bulk_runtime_v3.py",
)


def render(root: Path) -> dict[str, Any]:
    plans = g19.validate_all(root)
    items: list[dict[str, Any]] = []
    for controller, plan in plans.items():
        authority = g19.CONTROLLERS[controller]
        paths = (
            *COMMON,
            "evals/fleet/opencode_train_sweep_runner.py",
            "evals/fleet/fixed_proxy.py",
            "evals/fleet/Dockerfile.opencode",
            "evals/fleet/scripts/run_qwen_hosted_generation19_v2.sh",
        )
        data = prior_package._read(root, tuple(dict.fromkeys(paths)))  # noqa: SLF001
        data["plan-v2-a.json"] = (root / g19.PLAN_PATHS["qwen-a"]).read_text()
        data["plan-v2-b.json"] = (root / g19.PLAN_PATHS["qwen-b"]).read_text()
        cm = {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {"name": authority["configmap_name"], "namespace": base.NAMESPACE},
            "data": data,
        }
        job = base._base_job(authority["job_name"], authority["configmap_name"], scored=True)  # noqa: SLF001
        job["metadata"]["annotations"]["cyber-post-train.fleet.ai/launch-authorized"] = "true"
        job["metadata"]["labels"]["cyber-post-train.fleet.ai/experiment"] = (
            "q38-g19-hosted-runtime-diagnostic"
        )
        job["spec"]["template"]["metadata"]["labels"][
            "cyber-post-train.fleet.ai/experiment"
        ] = "q38-g19-hosted-runtime-diagnostic"
        container = job["spec"]["template"]["spec"]["containers"][0]
        container["args"] = [
            "apt-get update; apt-get install --yes --no-install-recommends "
            "docker.io=20.10.24+dfsg1-1+deb12u1+b6; "
            "exec /bin/bash /bootstrap/run_qwen_hosted_generation19_v2.sh"
        ]
        container["env"] = [
            row
            for row in container["env"]
            if row["name"] not in {"G18_PREFLIGHT_PATH", "G18_PREFLIGHT_SHA256"}
        ]
        container["env"].extend(
            [
                {"name": "G19_OUTPUT_ROOT", "value": plan["sfs_root"]},
                {"name": "G19_CONTROLLER", "value": controller},
                {
                    "name": "G19_DIAGNOSTIC_ROOT",
                    "value": plan["sfs_root"] + "-diagnostic",
                },
                {"name": "G19_PREFLIGHT_PATH", "value": PREFLIGHT_ROOT + "/CLEAR.json"},
                {"name": "G19_PREFLIGHT_SHA256", "value": PREFLIGHT_SHA256},
            ]
        )
        if len(json.dumps(cm).encode()) >= 900_000:
            raise ValueError("Generation-19 v2 ConfigMap exceeds safety budget")
        items.extend([cm, job])
    return {"apiVersion": "v1", "kind": "List", "items": items}
