"""Render the held immutable four-stream hosted GLM bulk package."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import yaml

from evals.fleet import hosted_glm_exact_bulk_runtime_v1 as runtime
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
INSTALL_PATHS = {
    "self_hosted.py": "evals/fleet/self_hosted.py",
    "runner.py": "evals/fleet/opencode_train_sweep_runner.py",
    "endpoint_lease.py": "evals/fleet/endpoint_lease.py",
    "predecessor.py": "evals/fleet/exact_pass4_bulk_v3.py",
    "engine.py": "evals/fleet/exact_pass4_bulk_runtime_v3.py",
    "universe.py": "evals/fleet/exact_pass4_universe.py",
    "crypto.py": "evals/fleet/exact_pass4_crypto.py",
    "inventory.py": "evals/fleet/exact_pass4_task_inventory.py",
    "bulk.py": "evals/fleet/hosted_glm_exact_bulk_v1.py",
    "bulk_runtime.py": "evals/fleet/hosted_glm_exact_bulk_runtime_v1.py",
    "fixed_proxy.py": "evals/fleet/fixed_proxy.py",
    "campaign.json": "evals/fleet/configs/q38-glm53-exact-easiest100-pass4-campaign-v1.json",
    "selection.json": "evals/fleet/configs/opencode-easiest-train100-selection-v2.json",
    "glm-template.json": "evals/fleet/configs/glm53-opencode-autocontinue-canary1-v1.json",
    "qwen-template.json": "evals/fleet/configs/qwen38-opencode-autocontinue-canary1-v1.json",
    "bulk-qwen-a.json": "evals/fleet/configs/exact-pass4-bulk-qwen-a-v3.json",
    "bulk-qwen-b.json": "evals/fleet/configs/exact-pass4-bulk-qwen-b-v3.json",
    "bulk-glm-a.json": "evals/fleet/configs/exact-pass4-bulk-glm-a-v3.json",
    "bulk-glm-b.json": "evals/fleet/configs/exact-pass4-bulk-glm-b-v3.json",
}


def _job(template: dict[str, Any], controller: str, *, authorized: bool) -> dict[str, Any]:
    job = copy.deepcopy(template)
    source = bulk.CONTROLLERS[controller]
    name = source["job_name"]
    job["metadata"]["name"] = name
    job["metadata"]["labels"]["cyber-post-train.fleet.ai/experiment"] = name
    job["metadata"]["annotations"]["cyber-post-train.fleet.ai/launch-authorized"] = str(
        authorized
    ).lower()
    pod = job["spec"]["template"]
    pod["metadata"]["labels"]["cyber-post-train.fleet.ai/experiment"] = name
    env = pod["spec"]["containers"][0]["env"]
    env[:] = [row for row in env if row["name"] not in {"JOB_NAME", "SECRET_UID"}]
    env.append({"name": "CONTROLLER", "value": controller})
    pod["spec"]["volumes"][0]["configMap"]["name"] = CONFIGMAP_NAME
    return job


def render(
    root: Path,
    *,
    canary_accepted: Path | None = None,
    canary_terminal: Path | None = None,
    release_receipt: Path | None = None,
) -> dict[str, Any]:
    plans = bulk.validate_all(root)
    evidence = (canary_accepted, canary_terminal, release_receipt)
    authorized = any(path is not None for path in evidence)
    if authorized:
        if any(path is None for path in evidence):
            raise ValueError("bulk authorization requires canary and release receipts")
        accepted = bulk.load(canary_accepted)  # type: ignore[arg-type]
        terminal = bulk.load(canary_terminal)  # type: ignore[arg-type]
        release = bulk.load(release_receipt)  # type: ignore[arg-type]
        runtime.validate_canary_receipts(accepted, terminal)
        first = next(iter(plans.values()))
        runtime.validate_release_receipt({**first, "repo_root": str(root)}, release, terminal)
    data = {}
    for name, relative in PATHS.items():
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"unsafe package source: {relative}")
        data[name] = path.read_text()
    run_script = data["run.sh"]
    for name in INSTALL_PATHS:
        if f"{name}:" not in run_script and f"/bootstrap/{name}" not in run_script:
            raise ValueError(f"packaged runtime install closure omits {name}")
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
    jobs = [_job(template, controller, authorized=authorized) for controller in bulk.CONTROLLERS]
    objects = {"apiVersion": "v1", "kind": "List", "items": [configmap, *jobs]}
    encoded = self_hosted.canonical_json(objects)
    if len(json.dumps(configmap).encode()) >= 900_000:
        raise ValueError("hosted GLM bulk ConfigMap exceeds safety budget")
    return {
        "objects": objects,
        "package_sha256": self_hosted.sha256(encoded),
        "launch_authorized": authorized,
        "reason": None if authorized else "requires_validated_canary_and_fresh_release_receipt",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preview", "render"))
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    parser.add_argument("--canary-accepted", type=Path)
    parser.add_argument("--canary-terminal", type=Path)
    parser.add_argument("--release-receipt", type=Path)
    args = parser.parse_args()
    package = render(
        args.repo.resolve(),
        canary_accepted=args.canary_accepted,
        canary_terminal=args.canary_terminal,
        release_receipt=args.release_receipt,
    )
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
