"""Render the dynamic v15 rank-51 canary duplicate-release observer."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import yaml

from evals.fleet import glm53_dedicated_v14_scored_canary_package_v1 as controller
from evals.fleet import glm53_dedicated_v14_scored_canary_preflight_package_v1 as base
from evals.fleet import glm53_dedicated_v15_canary_release_v1 as release
from evals.fleet import self_hosted


def render(root: Path, parity: Path, binding: Path, service_origin: str) -> dict[str, Any]:
    package = controller.render(root)
    preview = {key: value for key, value in package.items() if key != "objects"}
    if package["package_sha256"] != "sha256:e26bb6dc20e87b56fb9ed8d746e631d8f8c969bfe2717c7ab4d100975afdda3c":
        raise ValueError("dedicated v15 pre-admitted controller package drifted")
    data = copy.deepcopy(base.render(root)["objects"]["items"][0]["data"])
    data["release.py"] = (root / "evals/fleet/glm53_dedicated_v15_canary_release_v1.py").read_text()
    data["run.sh"] = (
        root / "evals/fleet/scripts/run_glm53_dedicated_v15_canary_release_v1.sh"
    ).read_text()
    data["controller-package.json"] = self_hosted.canonical_json(preview).decode()
    data["parity.json"] = parity.read_text()
    data["binding.json"] = binding.read_text()
    configmap = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": release.CONFIGMAP_NAME, "namespace": "fleet-train-jobs"},
        "immutable": True,
        "data": data,
    }
    job = copy.deepcopy(base.render(root)["objects"]["items"][1])
    job["metadata"]["name"] = release.JOB_NAME
    job["metadata"]["labels"]["cyber-post-train.fleet.ai/experiment"] = release.JOB_NAME
    job["metadata"]["annotations"]["cyber-post-train.fleet.ai/launch-authorized"] = "true"
    pod = job["spec"]["template"]
    pod["metadata"]["labels"]["cyber-post-train.fleet.ai/experiment"] = release.JOB_NAME
    pod["spec"]["volumes"][0]["configMap"]["name"] = release.CONFIGMAP_NAME
    pod["spec"]["containers"][0]["env"].append(
        {"name": "DEDICATED_SERVICE_ORIGIN", "value": service_origin}
    )
    objects = {"apiVersion": "v1", "kind": "List", "items": [configmap, job]}
    if len(json.dumps(configmap).encode()) >= 900_000:
        raise ValueError("dedicated v15 release ConfigMap exceeds safety budget")
    return {
        "objects": objects,
        "package_sha256": self_hosted.sha256(self_hosted.canonical_json(objects)),
        "controller_package_sha256": package["package_sha256"],
        "launch_authorized": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preview", "render"))
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--parity", type=Path, required=True)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--service-origin", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    value = render(args.repo.resolve(), args.parity, args.binding, args.service_origin)
    if args.command == "preview":
        print(json.dumps({key: item for key, item in value.items() if key != "objects"}))
        return 0
    if args.output is None or args.output.exists() or args.output.is_symlink():
        parser.error("render requires unused --output")
    args.output.write_text(yaml.safe_dump(value["objects"], sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
