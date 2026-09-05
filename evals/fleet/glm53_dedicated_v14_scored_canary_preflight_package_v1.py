"""Render the create-once score-blind dedicated GLM canary preflight."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import yaml

from evals.fleet import glm53_dedicated_v14_scored_canary_package_v1 as controller
from evals.fleet import glm53_dedicated_v14_scored_canary_preflight_v1 as preflight
from evals.fleet import hosted_glm_exact_bulk_release_package_v1 as base
from evals.fleet import self_hosted

EXTRA_PATHS = {
    "hosted_bulk.py": "evals/fleet/hosted_glm_exact_bulk_v1.py",
    "hosted_bulk_runtime.py": "evals/fleet/hosted_glm_exact_bulk_runtime_v1.py",
    "hosted_release.py": "evals/fleet/hosted_glm_exact_bulk_release_v1.py",
    "dedicated_canary.py": "evals/fleet/glm53_dedicated_v14_scored_canary_v1.py",
    "preflight.py": "evals/fleet/glm53_dedicated_v14_scored_canary_preflight_v1.py",
    "run.sh": "evals/fleet/scripts/run_glm53_dedicated_v14_scored_canary_preflight_v1.sh",
}


def render(root: Path) -> dict[str, Any]:
    package = controller.render(root)
    preview = {key: value for key, value in package.items() if key != "objects"}
    data: dict[str, str] = {}
    for name, relative in base.PATHS.items():
        if name in {"release.py", "run.sh"}:
            continue
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"unsafe preflight package source: {relative}")
        data[name] = path.read_text()
    for name, relative in EXTRA_PATHS.items():
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"unsafe preflight package source: {relative}")
        data[name] = path.read_text()
    data["controller-package.json"] = self_hosted.canonical_json(preview).decode()
    run_script = data["run.sh"]
    for name in (
        "self_hosted.py", "runner.py", "endpoint_lease.py", "predecessor.py",
        "engine.py", "universe.py", "crypto.py", "inventory.py",
        "hosted_bulk.py", "hosted_bulk_runtime.py", "hosted_release.py",
        "dedicated_canary.py", "preflight.py",
        "campaign.json", "selection.json", "glm-template.json", "qwen-template.json",
        "bulk-qwen-a.json", "bulk-qwen-b.json", "bulk-glm-a.json", "bulk-glm-b.json",
    ):
        if f"{name}:" not in run_script and f"/bootstrap/{name}" not in run_script:
            raise ValueError(f"preflight runtime install closure omits {name}")
    configmap = {
        "apiVersion": "v1", "kind": "ConfigMap",
        "metadata": {"name": preflight.CONFIGMAP_NAME, "namespace": "fleet-train-jobs"},
        "immutable": True, "data": data,
    }
    job = copy.deepcopy(base.render(root)["objects"]["items"][1])
    job["metadata"]["name"] = preflight.JOB_NAME
    job["metadata"]["labels"]["cyber-post-train.fleet.ai/experiment"] = preflight.JOB_NAME
    job["metadata"]["annotations"]["cyber-post-train.fleet.ai/launch-authorized"] = "true"
    pod = job["spec"]["template"]
    pod["metadata"]["labels"]["cyber-post-train.fleet.ai/experiment"] = preflight.JOB_NAME
    pod["spec"]["priorityClassName"] = "fleet-infra-quiet"
    pod["spec"]["preemptionPolicy"] = "Never"
    pod["spec"]["volumes"][0]["configMap"]["name"] = preflight.CONFIGMAP_NAME
    objects = {"apiVersion": "v1", "kind": "List", "items": [configmap, job]}
    if len(json.dumps(configmap).encode()) >= 900_000:
        raise ValueError("dedicated preflight ConfigMap exceeds safety budget")
    return {
        "objects": objects,
        "package_sha256": self_hosted.sha256(self_hosted.canonical_json(objects)),
        "launch_authorized": True,
        "controller_package_sha256": preview["package_sha256"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preview", "render"))
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()
    value = render(args.repo.resolve())
    if args.command == "preview":
        print(json.dumps({k: v for k, v in value.items() if k != "objects"}, sort_keys=True))
    else:
        print(yaml.safe_dump(value["objects"], sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
