"""Render G19 read-only preflight and two held hosted bulk controller Jobs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evals.fleet import qwen_hosted_generation18_package as base
from evals.fleet import qwen_hosted_generation19_bulk as g19

PREFLIGHT_JOB = "chris-q38-ac-exact100-g19-preflight-v4"
PREFLIGHT_CM = PREFLIGHT_JOB + "-run-v1"
PREFLIGHT_ROOT = f"/mnt/sfs/jobs/{PREFLIGHT_JOB}"
COMMON = (
    "evals/fleet/self_hosted.py",
    "evals/fleet/exact_pass4_bulk_v3.py",
    "evals/fleet/exact_pass4_bulk_runtime_v3.py",
    "evals/fleet/exact_pass4_universe.py",
    "evals/fleet/exact_pass4_crypto.py",
    "evals/fleet/endpoint_lease.py",
    "evals/fleet/qwen_bulk_generation16.py",
    "evals/fleet/qwen_hosted_generation18.py",
    "evals/fleet/qwen_hosted_generation19_bulk.py",
)


def _read(root: Path, paths: tuple[str, ...]) -> dict[str, str]:
    return {Path(path).name: (root / path).read_text() for path in paths}


def render(root: Path, preflight_sha256: str | None = None) -> dict[str, Any]:
    plans = g19.validate_all(root)
    pre_data = _read(
        root,
        (
            *COMMON,
            "evals/fleet/qwen_bulk_generation16_preflight.py",
            "evals/fleet/qwen_bulk_generation16_runtime.py",
            "evals/fleet/qwen_hosted_generation19_preflight.py",
            "evals/fleet/scripts/run_qwen_hosted_generation19_preflight.sh",
        ),
    )
    pre_data["plan-a.json"] = (root / g19.PLAN_PATHS["qwen-a"]).read_text()
    pre_data["plan-b.json"] = (root / g19.PLAN_PATHS["qwen-b"]).read_text()
    pre_cm = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": PREFLIGHT_CM, "namespace": base.NAMESPACE},
        "data": pre_data,
    }
    pre_job = base._base_job(PREFLIGHT_JOB, PREFLIGHT_CM, scored=False)  # noqa: SLF001
    pre_job["spec"]["template"]["spec"]["containers"][0]["args"] = [
        "exec /bin/bash /bootstrap/run_qwen_hosted_generation19_preflight.sh"
    ]
    items: list[dict[str, Any]] = [pre_cm, pre_job]
    for controller, plan in plans.items():
        authority = g19.CONTROLLERS[controller]
        data = _read(
            root,
            (
                *COMMON,
                "evals/fleet/opencode_train_sweep_runner.py",
                "evals/fleet/qwen_hosted_generation19_runtime.py",
                "evals/fleet/fixed_proxy.py",
                "evals/fleet/Dockerfile.opencode",
                "evals/fleet/scripts/run_qwen_hosted_generation19.sh",
            ),
        )
        data["plan-a.json"] = (root / g19.PLAN_PATHS["qwen-a"]).read_text()
        data["plan-b.json"] = (root / g19.PLAN_PATHS["qwen-b"]).read_text()
        data["runtime-plan.json"] = json.dumps(plan, sort_keys=True, separators=(",", ":")) + "\n"
        cm = {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {"name": authority["configmap_name"], "namespace": base.NAMESPACE},
            "data": data,
        }
        job = base._base_job(authority["job_name"], authority["configmap_name"], scored=True)  # noqa: SLF001
        job["metadata"]["annotations"]["cyber-post-train.fleet.ai/launch-authorized"] = (
            "true" if preflight_sha256 is not None else "false"
        )
        container = job["spec"]["template"]["spec"]["containers"][0]
        container["args"] = [
            "apt-get update; apt-get install --yes --no-install-recommends "
            "docker.io=20.10.24+dfsg1-1+deb12u1+b6; "
            "exec /bin/bash /bootstrap/run_qwen_hosted_generation19.sh"
        ]
        container["env"] = [
            row
            for row in container["env"]
            if row["name"] not in {"G18_PREFLIGHT_PATH", "G18_PREFLIGHT_SHA256"}
        ]
        container["env"].extend(
            [
                {"name": "G19_OUTPUT_ROOT", "value": plan["sfs_root"]},
                {"name": "G19_PREFLIGHT_PATH", "value": PREFLIGHT_ROOT + "/CLEAR.json"},
                {"name": "G19_PREFLIGHT_SHA256", "value": preflight_sha256 or "__HELD__"},
            ]
        )
        for obj in (cm,):
            if len(json.dumps(obj).encode()) >= 900_000:
                raise ValueError("Generation-19 ConfigMap exceeds safety budget")
        items.extend([cm, job])
    return {"apiVersion": "v1", "kind": "List", "items": items}
