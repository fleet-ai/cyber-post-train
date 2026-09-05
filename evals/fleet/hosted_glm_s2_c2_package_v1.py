"""Render a held immutable package for the one-cell GLM concurrency-two canary."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import yaml

from evals.fleet import hosted_glm_exact_bulk_package_v1 as base
from evals.fleet import hosted_glm_s2_c2_canary_v1 as c2
from evals.fleet import hosted_glm_s2_c2_runtime_v1 as runtime
from evals.fleet import self_hosted


def render(
    root: Path,
    *,
    release_receipt: Path | None = None,
    inventory_receipt: Path | None = None,
) -> dict[str, Any]:
    c2.validate_all(root)
    authorized = release_receipt is not None or inventory_receipt is not None
    if authorized:
        if release_receipt is None or inventory_receipt is None:
            raise ValueError("GLM c2 authorization requires release and inventory receipts")
        inventory = c2.load(inventory_receipt)
        plan = c2.build_runtime_plan(c2.CONTROLLER, inventory, root)
        runtime.validate_release_receipt(plan, c2.load(release_receipt))
    rendered = base.render(root)
    configmap, template = copy.deepcopy(rendered["objects"]["items"][:2])
    configmap["metadata"]["name"] = c2.CONFIGMAP_NAME
    configmap["data"]["c2.py"] = (root / "evals/fleet/hosted_glm_s2_c2_canary_v1.py").read_text()
    configmap["data"]["c2_release.py"] = (
        root / "evals/fleet/hosted_glm_s2_c2_release_v1.py"
    ).read_text()
    configmap["data"]["original_release.py"] = (
        root / "evals/fleet/hosted_glm_exact_bulk_release_v1.py"
    ).read_text()
    configmap["data"]["c2_runtime.py"] = (
        root / "evals/fleet/hosted_glm_s2_c2_runtime_v1.py"
    ).read_text()
    configmap["data"]["run.sh"] = (
        root / "evals/fleet/scripts/run_hosted_glm_s2_c2_canary_v1.sh"
    ).read_text()
    job = template
    job["metadata"]["name"] = c2.JOB_NAME
    job["metadata"]["labels"]["cyber-post-train.fleet.ai/experiment"] = c2.JOB_NAME
    job["metadata"]["annotations"]["cyber-post-train.fleet.ai/launch-authorized"] = str(
        authorized
    ).lower()
    pod = job["spec"]["template"]
    pod["metadata"]["labels"]["cyber-post-train.fleet.ai/experiment"] = c2.JOB_NAME
    pod["spec"]["volumes"][0]["configMap"]["name"] = c2.CONFIGMAP_NAME
    objects = {"apiVersion": "v1", "kind": "List", "items": [configmap, job]}
    return {
        "objects": objects,
        "package_sha256": self_hosted.sha256(self_hosted.canonical_json(objects)),
        "launch_authorized": authorized,
        "reason": None if authorized else "requires_s1_acceptance_and_fresh_c2_release_receipt",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preview", "render"))
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--release-receipt", type=Path)
    parser.add_argument("--inventory-receipt", type=Path)
    args = parser.parse_args()
    value = render(
        args.repo.resolve(),
        release_receipt=args.release_receipt,
        inventory_receipt=args.inventory_receipt,
    )
    if args.command == "preview":
        print(json.dumps({key: item for key, item in value.items() if key != "objects"}))
    else:
        print(yaml.safe_dump(value["objects"], sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
