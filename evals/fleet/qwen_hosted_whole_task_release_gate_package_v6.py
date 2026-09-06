"""Fresh observer identity with the generic release-gate package-v1 contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from evals.fleet import qwen_hosted_whole_task_release_gate_package_v5 as prior
from evals.fleet import qwen_hosted_whole_task_release_gate_v1 as gate

JOB_NAME = "chris-q38-hosted-r15-r16-release-gate-v6"
CONFIGMAP_NAME = JOB_NAME + "-package"
OUTPUT_ROOT = f"/mnt/sfs/jobs/{JOB_NAME}"


def _replace_identity(value: Any) -> Any:
    if isinstance(value, str):
        return value.replace(prior.CONFIGMAP_NAME, CONFIGMAP_NAME).replace(
            prior.JOB_NAME, JOB_NAME
        )
    if isinstance(value, list):
        return [_replace_identity(item) for item in value]
    if isinstance(value, dict):
        return {key: _replace_identity(item) for key, item in value.items()}
    return value


def render(root: Path) -> dict[str, Any]:
    rendered = _replace_identity(prior.render(root))
    configmap, job = rendered["items"]
    package_source = json.loads(configmap["data"]["package-source.json"])
    package_source.pop("receipt_sha256", None)
    package_source["schema_version"] = (
        "fleet-qwen38-hosted-whole-task-release-gate-package-v1"
    )
    configmap["data"]["package-source.json"] = (
        json.dumps(gate._seal(package_source), sort_keys=True, separators=(",", ":"))
        + "\n"
    )
    experiment = "q38-hosted-whole-task-release-gate-v6"
    job["metadata"]["labels"]["cyber-post-train.fleet.ai/experiment"] = experiment
    job["spec"]["template"]["metadata"]["labels"][
        "cyber-post-train.fleet.ai/experiment"
    ] = experiment
    return rendered


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    payload = yaml.safe_dump(render(args.root.resolve(strict=True)), sort_keys=False)
    if args.output is None:
        print(payload, end="")
    else:
        args.output.write_text(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
