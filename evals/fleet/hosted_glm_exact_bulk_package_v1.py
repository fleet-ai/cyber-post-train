"""Render the held immutable four-stream hosted GLM bulk package."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import yaml

from evals.fleet import hosted_glm_exact_bulk_v1 as bulk
from evals.fleet import hosted_glm_exact_canary_package_v1 as canary_package
from evals.fleet import self_hosted

CONFIGMAP_NAME = "chris-glm53-exact100-hosted-bulk-v1-run"
PATHS = {
    **{
        name: relative
        for name, relative in canary_package.PATHS.items()
        if name not in {"canary.py", "run.sh", "parity.json"}
    },
    "predecessor.py": "evals/fleet/exact_pass4_bulk_v3.py",
    "engine.py": "evals/fleet/exact_pass4_bulk_runtime_v3.py",
    "bulk.py": "evals/fleet/hosted_glm_exact_bulk_v1.py",
    "bulk_runtime.py": "evals/fleet/hosted_glm_exact_bulk_runtime_v1.py",
    "run.sh": "evals/fleet/scripts/run_hosted_glm_exact_bulk_v1.sh",
}


def _job(template: dict[str, Any], controller: str) -> dict[str, Any]:
    job = copy.deepcopy(template)
    source = bulk.CONTROLLERS[controller]
    name = source["job_name"]
    job["metadata"]["name"] = name
    job["metadata"]["labels"]["cyber-post-train.fleet.ai/experiment"] = name
    job["metadata"]["annotations"]["cyber-post-train.fleet.ai/launch-authorized"] = "false"
    pod = job["spec"]["template"]
    pod["metadata"]["labels"]["cyber-post-train.fleet.ai/experiment"] = name
    env = pod["spec"]["containers"][0]["env"]
    env[:] = [row for row in env if row["name"] not in {"JOB_NAME", "SECRET_UID"}]
    env.append({"name": "CONTROLLER", "value": controller})
    pod["spec"]["volumes"][0]["configMap"]["name"] = CONFIGMAP_NAME
    return job


def render(root: Path) -> dict[str, Any]:
    bulk.validate_all(root)
    data = {}
    for name, relative in PATHS.items():
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"unsafe package source: {relative}")
        data[name] = path.read_text()
    configmap = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": CONFIGMAP_NAME, "namespace": "fleet-train-jobs"},
        "immutable": True,
        "data": data,
    }
    template = yaml.safe_load(
        (root / "evals/fleet/cluster/hosted-glm-exact-r001-a1-canary-v1.yaml").read_text()
    )
    jobs = [_job(template, controller) for controller in bulk.CONTROLLERS]
    objects = {"apiVersion": "v1", "kind": "List", "items": [configmap, *jobs]}
    encoded = self_hosted.canonical_json(objects)
    if len(json.dumps(configmap).encode()) >= 900_000:
        raise ValueError("hosted GLM bulk ConfigMap exceeds safety budget")
    return {
        "objects": objects,
        "package_sha256": self_hosted.sha256(encoded),
        "launch_authorized": False,
        "reason": "requires_validated_canary_and_fresh_release_receipt",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preview", "render"))
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    package = render(args.repo.resolve())
    if args.command == "preview":
        print(
            json.dumps(
                {key: value for key, value in package.items() if key != "objects"},
                sort_keys=True,
            )
        )
        return 0
    if args.output is None or args.output.exists() or args.output.is_symlink():
        parser.error("render requires an unused --output")
    args.output.write_text(yaml.safe_dump(package["objects"], sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
