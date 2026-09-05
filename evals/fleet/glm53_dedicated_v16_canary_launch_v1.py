"""Render the one-cell GLM v16 canary after fresh parity and duplicate release."""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path
from typing import Any

import yaml

from evals.fleet import glm53_dedicated_v14_scored_canary_package_v1 as package
from evals.fleet import glm53_dedicated_v14_scored_canary_runtime_v1 as runtime
from evals.fleet import glm53_dedicated_v14_scored_canary_v1 as canary
from evals.fleet import glm53_dedicated_v15_canary_release_package_v1 as release_package
from evals.fleet import self_hosted


def render(
    root: Path,
    *,
    parity_path: Path,
    binding_path: Path,
    release_path: Path,
    service_origin: str,
) -> dict[str, Any]:
    parity = canary.load(parity_path)
    binding = canary.load(binding_path)
    release = canary.load(release_path)
    plan = canary.build_plan(
        root,
        service_origin=service_origin,
        parity_path=parity_path,
        binding_path=binding_path,
    )
    item = plan["attempts"][0]
    if (
        release.get("schema_version") != runtime.RELEASE_SCHEMA
        or release.get("status") != "CLEAR"
        or not isinstance(release.get("plan_sha256"), str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", release["plan_sha256"]) is None
        or release.get("selection_rank") != 51
        or release.get("attempt") != 1
        or release.get("cell_id") != item["cell_id"]
        or release.get("execution_id") != item["execution_id"]
        or release.get("all_four_rank_cells_unstarted") is not True
        or release.get("fleet_session_collisions") != 0
        or release.get("global_claim_collisions") != 0
        or release.get("sfs_output_collisions") != 0
        or release.get("kubernetes_object_collisions") != 0
        or release.get("checked_immediately_before_create") is not True
        or release.get("receipt_sha256")
        != self_hosted.digest_without(release, "receipt_sha256")
    ):
        raise ValueError("dedicated v16 canary release drifted")
    built = package.render(root)
    if built["package_sha256"] != release_package.CONTROLLER_PACKAGE_SHA256:
        raise ValueError("dedicated v16 controller package drifted")
    source, job = copy.deepcopy(built["objects"]["items"])
    job["metadata"]["annotations"]["cyber-post-train.fleet.ai/launch-authorized"] = "true"
    evidence = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": package.EVIDENCE_CONFIGMAP, "namespace": "fleet-train-jobs"},
        "immutable": True,
        "data": {
            "parity.json": parity_path.read_text(),
            "binding.json": binding_path.read_text(),
            "release.json": release_path.read_text(),
            "service_origin": service_origin,
        },
    }
    objects = {"apiVersion": "v1", "kind": "List", "items": [source, evidence, job]}
    body = {
        "schema_version": "fleet-glm53-dedicated-v16-scored-canary-launch-package-v1",
        "status": "READY",
        "launch_authorized": True,
        "controller_package_sha256": built["package_sha256"],
        "controller_objects_sha256": built["objects_sha256"],
        "runtime_plan_sha256": release["plan_sha256"],
        "parity_receipt_sha256": parity["receipt_sha256"],
        "server_binding_sha256": self_hosted.sha256(self_hosted.canonical_json(binding)),
        "release_receipt_sha256": release["receipt_sha256"],
        "cell_id": item["cell_id"],
        "execution_id": item["execution_id"],
        "objects": objects,
    }
    body["package_sha256"] = self_hosted.digest_without(body, "package_sha256")
    return body


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preview", "render"))
    parser.add_argument("--parity", type=Path, required=True)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--service-origin", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    value = render(
        Path.cwd(),
        parity_path=args.parity,
        binding_path=args.binding,
        release_path=args.release,
        service_origin=args.service_origin,
    )
    if args.command == "preview":
        print(json.dumps({key: item for key, item in value.items() if key != "objects"}, sort_keys=True))
        return 0
    if args.output is None or args.output.exists() or args.output.is_symlink():
        parser.error("render requires an unused --output")
    args.output.write_text(yaml.safe_dump(value["objects"], sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
