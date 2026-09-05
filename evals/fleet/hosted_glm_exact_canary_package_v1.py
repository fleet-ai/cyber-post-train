"""Render the immutable ConfigMap + Job for the hosted GLM exact canary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from evals.fleet import hosted_glm_exact_canary_v1 as canary
from evals.fleet import self_hosted

PATHS = {
    "Dockerfile.opencode": "evals/fleet/Dockerfile.opencode",
    "fixed_proxy.py": "evals/fleet/fixed_proxy.py",
    "self_hosted.py": "evals/fleet/self_hosted.py",
    "runner.py": "evals/fleet/opencode_train_sweep_runner.py",
    "endpoint_lease.py": "evals/fleet/endpoint_lease.py",
    "bulk.py": "evals/fleet/exact_pass4_bulk_v3.py",
    "bulk_runtime.py": "evals/fleet/exact_pass4_bulk_runtime_v3.py",
    "universe.py": "evals/fleet/exact_pass4_universe.py",
    "crypto.py": "evals/fleet/exact_pass4_crypto.py",
    "inventory.py": "evals/fleet/exact_pass4_task_inventory.py",
    "canary.py": "evals/fleet/hosted_glm_exact_canary_v1.py",
    "run.sh": "evals/fleet/scripts/run_hosted_glm_exact_canary_v1.sh",
    "campaign.json": "evals/fleet/configs/q38-glm53-exact-easiest100-pass4-campaign-v1.json",
    "selection.json": "evals/fleet/configs/opencode-easiest-train100-selection-v2.json",
    "glm-template.json": "evals/fleet/configs/glm53-opencode-autocontinue-canary1-v1.json",
    "qwen-template.json": "evals/fleet/configs/qwen38-opencode-autocontinue-canary1-v1.json",
    "bulk-qwen-a.json": "evals/fleet/configs/exact-pass4-bulk-qwen-a-v3.json",
    "bulk-qwen-b.json": "evals/fleet/configs/exact-pass4-bulk-qwen-b-v3.json",
    "bulk-glm-a.json": "evals/fleet/configs/exact-pass4-bulk-glm-a-v3.json",
    "bulk-glm-b.json": "evals/fleet/configs/exact-pass4-bulk-glm-b-v3.json",
    "parity.json": str(canary.PARITY_PATH),
}
INSTALL_PATHS = {
    "self_hosted.py": "evals/fleet/self_hosted.py",
    "runner.py": "evals/fleet/opencode_train_sweep_runner.py",
    "endpoint_lease.py": "evals/fleet/endpoint_lease.py",
    "bulk.py": "evals/fleet/exact_pass4_bulk_v3.py",
    "bulk_runtime.py": "evals/fleet/exact_pass4_bulk_runtime_v3.py",
    "universe.py": "evals/fleet/exact_pass4_universe.py",
    "crypto.py": "evals/fleet/exact_pass4_crypto.py",
    "inventory.py": "evals/fleet/exact_pass4_task_inventory.py",
    "canary.py": "evals/fleet/hosted_glm_exact_canary_v1.py",
    "fixed_proxy.py": "evals/fleet/fixed_proxy.py",
    "campaign.json": "evals/fleet/configs/q38-glm53-exact-easiest100-pass4-campaign-v1.json",
    "selection.json": "evals/fleet/configs/opencode-easiest-train100-selection-v2.json",
    "glm-template.json": "evals/fleet/configs/glm53-opencode-autocontinue-canary1-v1.json",
    "qwen-template.json": "evals/fleet/configs/qwen38-opencode-autocontinue-canary1-v1.json",
    "bulk-qwen-a.json": "evals/fleet/configs/exact-pass4-bulk-qwen-a-v3.json",
    "bulk-qwen-b.json": "evals/fleet/configs/exact-pass4-bulk-qwen-b-v3.json",
    "bulk-glm-a.json": "evals/fleet/configs/exact-pass4-bulk-glm-a-v3.json",
    "bulk-glm-b.json": "evals/fleet/configs/exact-pass4-bulk-glm-b-v3.json",
    "parity.json": str(canary.PARITY_PATH),
}


def render(root: Path) -> dict[str, Any]:
    canary._validate_parity(root)  # noqa: SLF001 - package freezes this exact gate
    canary.bulk.validate_all(root)
    data: dict[str, str] = {}
    for name, relative in PATHS.items():
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"unsafe package source: {relative}")
        data[name] = path.read_text()
    run_script = data["run.sh"]
    for name, target in INSTALL_PATHS.items():
        if f"/bootstrap/{name}" not in run_script or target not in run_script:
            raise ValueError(f"packaged runtime install closure omits {name}")
    configmap = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": canary.CONFIGMAP_NAME, "namespace": "fleet-train-jobs"},
        "immutable": True,
        "data": data,
    }
    job = yaml.safe_load(
        (root / "evals/fleet/cluster/hosted-glm-exact-r001-a1-canary-v1.yaml").read_text()
    )
    value = {"apiVersion": "v1", "kind": "List", "items": [configmap, job]}
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    configmap_bytes = len(json.dumps(configmap).encode())
    if configmap_bytes >= 900_000:
        raise ValueError("hosted GLM canary ConfigMap exceeds the safety budget")
    return {
        "objects": value,
        "package_sha256": self_hosted.sha256(encoded),
        "configmap_json_bytes": configmap_bytes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preview", "render"))
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    built = render(args.repo.resolve())
    if args.command == "preview":
        print(
            json.dumps(
                {
                    "status": "READY",
                    "launch_authorized": True,
                    "objects_created": False,
                    "job_name": canary.JOB_NAME,
                    "selection_rank": 1,
                    "attempt": 1,
                    "whole_task_reservation": [1, 2, 3, 4],
                    "package_sha256": built["package_sha256"],
                    "configmap_json_bytes": built["configmap_json_bytes"],
                },
                sort_keys=True,
            )
        )
        return 0
    if args.output is None or args.output.exists() or args.output.is_symlink():
        parser.error("render requires an unused --output")
    args.output.write_text(yaml.safe_dump(built["objects"], sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
