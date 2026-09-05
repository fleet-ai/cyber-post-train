"""Render the held create-once score-blind GLM concurrency-two release observer."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import yaml

from evals.fleet import hosted_glm_exact_bulk_release_package_v1 as base
from evals.fleet import hosted_glm_s2_c2_release_v1 as release
from evals.fleet import self_hosted


def render(root: Path, *, s1_accepted: Path | None = None) -> dict[str, Any]:
    authorized = s1_accepted is not None
    if authorized:
        receipt = release.bulk.load(s1_accepted)  # type: ignore[arg-type]
        # Cell identities are already immutable in the held plan; authorization
        # must not depend on a desktop mount of the cluster-only inventory.
        plan = release.bulk.validate_all(root)["glm-hosted-s1"]
        first = min(plan["attempts"], key=lambda row: row["ordinal"])
        if any(
            (
                receipt.get("accepted") is not True,
                receipt.get("credited") is not True,
                receipt.get("cell_id") != first["cell_id"],
                receipt.get("execution_id") != first["execution_id"],
                receipt.get("receipt_sha256")
                != self_hosted.digest_without(receipt, "receipt_sha256"),
            )
        ):
            raise ValueError("GLM c2 release package requires exact s1 first acceptance")
    rendered = base.render(root)
    configmap, job = copy.deepcopy(rendered["objects"]["items"])
    configmap["metadata"]["name"] = release.CONFIGMAP_NAME
    configmap["data"]["bulk_runtime.py"] = (
        root / "evals/fleet/hosted_glm_exact_bulk_runtime_v1.py"
    ).read_text()
    configmap["data"]["original_release.py"] = (
        root / "evals/fleet/hosted_glm_exact_bulk_release_v1.py"
    ).read_text()
    configmap["data"]["c2.py"] = (root / "evals/fleet/hosted_glm_s2_c2_canary_v1.py").read_text()
    configmap["data"]["c2_release.py"] = (
        root / "evals/fleet/hosted_glm_s2_c2_release_v1.py"
    ).read_text()
    configmap["data"]["run.sh"] = (
        root / "evals/fleet/scripts/run_hosted_glm_s2_c2_release_v1.sh"
    ).read_text()
    job["metadata"]["name"] = release.JOB_NAME
    job["metadata"]["labels"]["cyber-post-train.fleet.ai/experiment"] = release.JOB_NAME
    job["metadata"]["annotations"]["cyber-post-train.fleet.ai/launch-authorized"] = str(
        authorized
    ).lower()
    job["spec"]["template"]["metadata"]["labels"][
        "cyber-post-train.fleet.ai/experiment"
    ] = release.JOB_NAME
    job["spec"]["template"]["spec"]["volumes"][0]["configMap"][
        "name"
    ] = release.CONFIGMAP_NAME
    objects = {"apiVersion": "v1", "kind": "List", "items": [configmap, job]}
    return {
        "objects": objects,
        "package_sha256": self_hosted.sha256(self_hosted.canonical_json(objects)),
        "launch_authorized": authorized,
        "reason": None if authorized else "requires_digest_valid_s1_first_acceptance",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preview", "render"))
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--s1-accepted", type=Path)
    args = parser.parse_args()
    value = render(args.repo.resolve(), s1_accepted=args.s1_accepted)
    if args.command == "preview":
        print(json.dumps({key: item for key, item in value.items() if key != "objects"}))
    else:
        print(yaml.safe_dump(value["objects"], sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
